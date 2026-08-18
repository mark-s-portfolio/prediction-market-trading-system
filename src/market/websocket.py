"""
Resilient Polymarket market-data WebSocket client for the public portfolio edition.

Responsibilities:
- maintain one canonical market-channel subscription
- receive and normalize book, price-change and top-of-book events
- publish market data into OrderBookStore
- track initial full-book coverage
- detect prolonged socket silence
- send the application-level heartbeat
- deduplicate immediately repeated raw frames
- reconnect with bounded backoff
- drain socket-owned child tasks before the next connection generation

The client deliberately does not contain trading admission, position sizing,
asset-specific strategy rules, execution policy, or historical setup knowledge.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
import hashlib
import inspect
import json
import math
import time
from typing import Awaitable, Callable, Dict, Iterable, Optional, Sequence, Tuple

import websockets
from websockets.exceptions import ConnectionClosed

from src.market.orderbook import OrderBookStore
from src.market.types import MarketDefinition, OrderBookSnapshot
from src.runtime.config import AppSettings, settings
from src.runtime.logging import runtime_print


class MarketDataEventType(str, Enum):
    BOOK = "BOOK"
    BOOK_DELTA = "BOOK_DELTA"
    TOP_OF_BOOK = "TOP_OF_BOOK"
    TICK_SIZE = "TICK_SIZE"
    CONNECTION = "CONNECTION"


class ConnectionState(str, Enum):
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    DISCONNECTED = "DISCONNECTED"
    STOPPED = "STOPPED"


@dataclass(frozen=True, slots=True)
class MarketDataEvent:
    """Public event emitted after market-data state has been normalized."""

    event_type: MarketDataEventType
    token_id: str = ""
    snapshot: Optional[OrderBookSnapshot] = None
    tick_size: Optional[float] = None
    connection_state: Optional[ConnectionState] = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class WebSocketPolicy:
    """Transport/observability settings only."""

    ping_interval_seconds: float = 10.0
    protocol_ping_interval_seconds: float = 20.0
    protocol_ping_timeout_seconds: float = 10.0
    close_timeout_seconds: float = 5.0
    reconnect_backoff_seconds: float = 3.0

    initial_coverage_check_seconds: float = 8.0
    silence_check_seconds: float = 5.0
    silence_warn_seconds: float = 60.0
    silence_reconnect_seconds: float = 90.0

    max_queue: int = 4096
    duplicate_frame_window_seconds: float = 0.030
    duplicate_frame_cache_max: int = 512

    cooperative_yield_every_frames: int = 16
    cooperative_yield_max_work_seconds: float = 0.004

    # Handler delivery is isolated from the socket receiver. When the bounded
    # queue is saturated, the oldest notification is coalesced away; the
    # OrderBookStore already contains the newer normalized state.
    event_queue_max: int = 1024
    event_shutdown_drain_seconds: float = 1.0


@dataclass(frozen=True, slots=True)
class WebSocketStats:
    connection_generation: int
    frames_received: int
    events_received: int
    duplicate_frames_dropped: int
    parse_errors: int
    reconnects: int
    handler_events_dropped: int
    initial_books_seen: int
    subscribed_tokens: int
    last_message_age_seconds: float

    @property
    def initial_coverage_ratio(self) -> float:
        if self.subscribed_tokens <= 0:
            return 1.0
        return self.initial_books_seen / self.subscribed_tokens


EventHandler = Callable[[MarketDataEvent], Optional[Awaitable[None]]]


def _as_event_list(payload: object) -> list[dict]:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _finite_price(value: object) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result) or not 0.0 < result < 1.0:
        return None

    return result


class MarketWebSocketClient:
    """Long-running WebSocket market-data service."""

    def __init__(
        self,
        orderbooks: OrderBookStore,
        *,
        app_settings: AppSettings = settings,
        policy: WebSocketPolicy = WebSocketPolicy(),
        event_handler: Optional[EventHandler] = None,
    ) -> None:
        self.orderbooks = orderbooks
        self.settings = app_settings
        self.policy = policy
        self.event_handler = event_handler

        self._markets: Tuple[MarketDefinition, ...] = ()
        self._tokens: Tuple[str, ...] = ()
        self._known_tokens: set[str] = set()

        self._stop_event = asyncio.Event()
        self._current_socket = None

        self._connection_generation = 0
        self._frames_received = 0
        self._events_received = 0
        self._duplicate_frames_dropped = 0
        self._parse_errors = 0
        self._reconnects = 0
        self._handler_events_dropped = 0

        self._event_queue: asyncio.Queue[MarketDataEvent] = asyncio.Queue(
            maxsize=max(1, int(self.policy.event_queue_max))
        )
        self._event_worker_task: Optional[asyncio.Task] = None

        self._initial_book_seen: set[str] = set()
        self._last_message_time = 0.0

        self._raw_recent: Dict[bytes, float] = {}

    @property
    def subscribed_tokens(self) -> Tuple[str, ...]:
        return self._tokens

    @property
    def markets(self) -> Tuple[MarketDefinition, ...]:
        return self._markets

    def configure_markets(
        self,
        markets: Sequence[MarketDefinition],
    ) -> Tuple[str, ...]:
        """Install one immutable token set for the next socket generation."""

        unique_markets: list[MarketDefinition] = []
        seen_slugs: set[str] = set()
        tokens: list[str] = []
        seen_tokens: set[str] = set()

        for market in markets:
            if market.slug not in seen_slugs:
                seen_slugs.add(market.slug)
                unique_markets.append(market)

            for token_id in (market.yes_token, market.no_token):
                token_id = str(token_id or "").strip()
                if token_id and token_id not in seen_tokens:
                    seen_tokens.add(token_id)
                    tokens.append(token_id)

        self._markets = tuple(unique_markets)
        self._tokens = tuple(tokens)
        self._known_tokens = set(tokens)
        return self._tokens

    async def stop(self) -> None:
        self._stop_event.set()

        socket = self._current_socket
        if socket is not None:
            try:
                await socket.close(code=1000, reason="client shutdown")
            except Exception:
                pass

    def stats(self) -> WebSocketStats:
        now = time.time()
        age = (
            max(0.0, now - self._last_message_time)
            if self._last_message_time > 0.0
            else math.inf
        )

        return WebSocketStats(
            connection_generation=self._connection_generation,
            frames_received=self._frames_received,
            events_received=self._events_received,
            duplicate_frames_dropped=self._duplicate_frames_dropped,
            parse_errors=self._parse_errors,
            reconnects=self._reconnects,
            handler_events_dropped=self._handler_events_dropped,
            initial_books_seen=len(self._initial_book_seen),
            subscribed_tokens=len(self._tokens),
            last_message_age_seconds=age,
        )

    def _ensure_event_worker(self) -> None:
        if self.event_handler is None:
            return

        task = self._event_worker_task
        if task is not None and not task.done():
            return

        self._event_worker_task = asyncio.create_task(
            self._event_worker(),
            name="market-ws-event-handler",
        )

    async def _event_worker(self) -> None:
        while True:
            event = await self._event_queue.get()
            try:
                handler = self.event_handler
                if handler is None:
                    continue

                try:
                    result = handler(event)
                    if inspect.isawaitable(result):
                        await result
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    runtime_print(
                        f"[websocket] event handler error: "
                        f"{type(exc).__name__}: {exc}"
                    )
            finally:
                self._event_queue.task_done()

    async def _shutdown_event_worker(self) -> None:
        task = self._event_worker_task
        if task is None:
            return

        drain = max(
            0.0,
            float(self.policy.event_shutdown_drain_seconds),
        )
        if drain > 0.0 and not task.done():
            try:
                await asyncio.wait_for(
                    self._event_queue.join(),
                    timeout=drain,
                )
            except asyncio.TimeoutError:
                pass

        if not task.done():
            task.cancel()

        await asyncio.gather(task, return_exceptions=True)
        self._event_worker_task = None

        # Balance unfinished-task accounting for notifications intentionally
        # discarded during shutdown.
        while True:
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self._event_queue.task_done()

    async def _emit(self, event: MarketDataEvent) -> None:
        if self.event_handler is None:
            return

        self._ensure_event_worker()

        try:
            self._event_queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass

        # The normalized market state is already committed to OrderBookStore.
        # Coalesce the oldest notification rather than blocking socket receive
        # on an arbitrarily slow async consumer.
        try:
            self._event_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        else:
            self._event_queue.task_done()
            self._handler_events_dropped += 1

        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            # A concurrent producer won the slot; preserve receive-loop liveness.
            self._handler_events_dropped += 1

    async def _emit_connection(
        self,
        state: ConnectionState,
        detail: str = "",
    ) -> None:
        await self._emit(
            MarketDataEvent(
                event_type=MarketDataEventType.CONNECTION,
                connection_state=state,
                detail=detail,
            )
        )

    def _subscription_payload(self) -> dict:
        return {
            "assets_ids": list(self._tokens),
            "type": "market",
            "custom_feature_enabled": True,
        }

    async def run(
        self,
        markets: Optional[Sequence[MarketDefinition]] = None,
    ) -> None:
        """Run until `stop()` is called.

        Reconnects keep the same configured market set. Market rotation belongs to
        the higher-level engine, which may stop this client and start a new
        generation with a newly discovered market set.
        """

        if markets is not None:
            self.configure_markets(markets)

        if not self._tokens:
            raise ValueError("at least one market token is required")

        self._stop_event.clear()
        reconnect_delay = max(0.0, float(self.policy.reconnect_backoff_seconds))

        while not self._stop_event.is_set():
            await self._emit_connection(
                ConnectionState.CONNECTING,
                f"tokens={len(self._tokens)}",
            )

            try:
                await self._run_connection()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stop_event.is_set():
                    break

                runtime_print(
                    f"[websocket] connection generation failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                await self._emit_connection(
                    ConnectionState.DISCONNECTED,
                    f"{type(exc).__name__}: {exc}",
                )

            if self._stop_event.is_set():
                break

            self._reconnects += 1
            if reconnect_delay > 0.0:
                await asyncio.sleep(reconnect_delay)

        await self._emit_connection(ConnectionState.STOPPED, "client stopped")
        await self._shutdown_event_worker()

    async def _run_connection(self) -> None:
        self._connection_generation += 1
        generation = self._connection_generation
        self._initial_book_seen = set()
        self._raw_recent.clear()

        child_tasks: set[asyncio.Task] = set()

        runtime_print(
            f"[websocket] connecting generation={generation} "
            f"tokens={len(self._tokens)}"
        )

        connect_kwargs = {
            "ping_interval": float(self.policy.protocol_ping_interval_seconds),
            "ping_timeout": float(self.policy.protocol_ping_timeout_seconds),
            "close_timeout": float(self.policy.close_timeout_seconds),
            "max_queue": max(1, int(self.policy.max_queue)),
        }

        try:
            async with websockets.connect(
                self.settings.network.market_ws_url,
                **connect_kwargs,
            ) as websocket:
                self._current_socket = websocket
                self._last_message_time = time.time()

                await websocket.send(json.dumps(self._subscription_payload()))

                runtime_print(
                    f"[websocket] subscribed generation={generation} "
                    f"assets={len(self._tokens)}"
                )
                await self._emit_connection(
                    ConnectionState.CONNECTED,
                    f"generation={generation}",
                )

                child_tasks.add(
                    asyncio.create_task(
                        self._heartbeat_loop(websocket, generation),
                        name=f"market-ws-heartbeat-{generation}",
                    )
                )
                child_tasks.add(
                    asyncio.create_task(
                        self._silence_watchdog(websocket, generation),
                        name=f"market-ws-silence-{generation}",
                    )
                )
                child_tasks.add(
                    asyncio.create_task(
                        self._coverage_watchdog(generation),
                        name=f"market-ws-coverage-{generation}",
                    )
                )

                await self._receive_loop(websocket, generation)

        except ConnectionClosed as exc:
            if not self._stop_event.is_set():
                runtime_print(
                    f"[websocket] disconnected generation={generation} "
                    f"code={getattr(exc, 'code', None)} "
                    f"reason={getattr(exc, 'reason', '') or '-'}"
                )
                await self._emit_connection(
                    ConnectionState.DISCONNECTED,
                    f"code={getattr(exc, 'code', None)} "
                    f"reason={getattr(exc, 'reason', '') or '-'}",
                )
        finally:
            self._current_socket = None
            await self._cancel_and_drain(child_tasks)

    async def _cancel_and_drain(
        self,
        tasks: Iterable[asyncio.Task],
    ) -> None:
        """Cancel socket-owned children and wait until cancellation is delivered."""

        owned = [task for task in tasks if task is not None]

        for task in owned:
            if not task.done():
                task.cancel()

        if owned:
            await asyncio.gather(*owned, return_exceptions=True)

    async def _heartbeat_loop(self, websocket, generation: int) -> None:
        interval = max(1.0, float(self.policy.ping_interval_seconds))

        while not self._stop_event.is_set():
            await asyncio.sleep(interval)

            if generation != self._connection_generation:
                return

            try:
                await websocket.send("PING")
            except Exception:
                return

    async def _silence_watchdog(self, websocket, generation: int) -> None:
        check_interval = max(1.0, float(self.policy.silence_check_seconds))
        warn_after = max(check_interval, float(self.policy.silence_warn_seconds))
        reconnect_after = max(
            warn_after,
            float(self.policy.silence_reconnect_seconds),
        )
        warned = False

        while not self._stop_event.is_set():
            await asyncio.sleep(check_interval)

            if generation != self._connection_generation:
                return

            silent_for = time.time() - self._last_message_time

            if silent_for >= warn_after and not warned:
                warned = True
                runtime_print(
                    f"[websocket] market-data silence: {silent_for:.0f}s"
                )
                await self._emit_connection(
                    ConnectionState.DEGRADED,
                    f"market-data silence {silent_for:.0f}s",
                )

            if silent_for >= reconnect_after:
                runtime_print(
                    f"[websocket] forcing reconnect after "
                    f"{silent_for:.0f}s of silence"
                )
                try:
                    await websocket.close(
                        code=1012,
                        reason="market data silence",
                    )
                except Exception:
                    pass
                return

    async def _coverage_watchdog(self, generation: int) -> None:
        delay = max(
            1.0,
            float(self.policy.initial_coverage_check_seconds),
        )
        await asyncio.sleep(delay)

        if (
            self._stop_event.is_set()
            or generation != self._connection_generation
        ):
            return

        missing = [
            token_id
            for token_id in self._tokens
            if token_id not in self._initial_book_seen
        ]

        if not missing:
            runtime_print(
                f"[websocket] initial full-book coverage "
                f"{len(self._initial_book_seen)}/{len(self._tokens)}"
            )
            return

        runtime_print(
            f"[websocket] partial initial full-book coverage "
            f"{len(self._initial_book_seen)}/{len(self._tokens)} "
            f"missing={len(missing)}"
        )
        await self._emit_connection(
            ConnectionState.DEGRADED,
            f"initial-book coverage missing={len(missing)}",
        )

    def _is_duplicate_raw_frame(self, raw: object) -> bool:
        try:
            raw_bytes = (
                raw.encode("utf-8", errors="surrogatepass")
                if isinstance(raw, str)
                else bytes(raw)
            )
        except Exception:
            return False

        digest = hashlib.blake2b(raw_bytes, digest_size=16).digest()
        now = time.monotonic()
        previous = float(self._raw_recent.get(digest, 0.0) or 0.0)
        window = max(
            0.0,
            float(self.policy.duplicate_frame_window_seconds),
        )

        if previous > 0.0 and now - previous <= window:
            self._duplicate_frames_dropped += 1
            return True

        self._raw_recent[digest] = now

        cap = max(32, int(self.policy.duplicate_frame_cache_max))
        if len(self._raw_recent) > cap:
            cutoff = now - max(0.25, window * 4.0)
            self._raw_recent = {
                key: timestamp
                for key, timestamp in self._raw_recent.items()
                if timestamp >= cutoff
            }

            if len(self._raw_recent) > cap:
                newest = sorted(
                    self._raw_recent.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:cap]
                self._raw_recent = dict(newest)

        return False

    async def _receive_loop(self, websocket, generation: int) -> None:
        frames_since_yield = 0
        last_yield_perf = time.perf_counter()

        while not self._stop_event.is_set():
            raw = await websocket.recv()
            self._last_message_time = time.time()

            if self._is_duplicate_raw_frame(raw):
                continue

            self._frames_received += 1

            try:
                payload = json.loads(raw)
            except Exception:
                self._parse_errors += 1
                continue

            events = _as_event_list(payload)
            self._events_received += len(events)

            frames_since_yield += 1
            now_perf = time.perf_counter()

            if (
                frames_since_yield
                >= max(1, int(self.policy.cooperative_yield_every_frames))
                or now_perf - last_yield_perf
                >= max(
                    0.0,
                    float(self.policy.cooperative_yield_max_work_seconds),
                )
            ):
                await asyncio.sleep(0)
                frames_since_yield = 0
                last_yield_perf = time.perf_counter()

            for event_index, event in enumerate(events):
                if event_index and event_index % 48 == 0:
                    await asyncio.sleep(0)

                await self._process_event(event)

            if generation != self._connection_generation:
                return

    async def _process_event(self, event: dict) -> None:
        event_type = str(event.get("event_type", "") or "").strip().lower()
        asset_id = str(event.get("asset_id", "") or "").strip()

        if event_type == "tick_size_change" and asset_id:
            await self._process_tick_size(event, asset_id)
            return

        # Polymarket may package top-price updates inside price_changes, and the
        # top-level asset_id isn't always the token updated by each child row.
        price_changes = event.get("price_changes")
        if (
            isinstance(price_changes, list)
            and price_changes
            and not event.get("bids")
            and not event.get("asks")
        ):
            for index, change in enumerate(price_changes):
                if index and index % 48 == 0:
                    await asyncio.sleep(0)
                if isinstance(change, dict):
                    await self._process_price_change(
                        change,
                        fallback_asset_id=asset_id,
                    )
            return

        if not asset_id or asset_id not in self._known_tokens:
            return

        if event_type == "best_bid_ask":
            await self._process_best_bid_ask(event, asset_id)
            return

        bids = event.get("bids", [])
        asks = event.get("asks", [])

        if bids or asks:
            await self._process_book_event(
                event_type=event_type,
                asset_id=asset_id,
                bids=bids,
                asks=asks,
            )

    async def _process_tick_size(
        self,
        event: dict,
        asset_id: str,
    ) -> None:
        if asset_id not in self._known_tokens:
            return

        raw_tick = event.get("new_tick_size", event.get("tick_size"))
        try:
            tick_size = float(raw_tick)
        except (TypeError, ValueError):
            return

        if not math.isfinite(tick_size) or tick_size <= 0.0:
            return

        await self._emit(
            MarketDataEvent(
                event_type=MarketDataEventType.TICK_SIZE,
                token_id=asset_id,
                tick_size=tick_size,
            )
        )

    async def _process_price_change(
        self,
        change: dict,
        *,
        fallback_asset_id: str,
    ) -> None:
        asset_id = str(
            change.get("asset_id", fallback_asset_id) or ""
        ).strip()

        if not asset_id or asset_id not in self._known_tokens:
            return

        best_bid = _finite_price(change.get("best_bid"))
        best_ask = _finite_price(change.get("best_ask"))

        if (
            best_bid is None
            or best_ask is None
            or best_bid >= best_ask
        ):
            return

        snapshot = self.orderbooks.publish_ws_top(
            token_id=asset_id,
            best_bid=best_bid,
            best_ask=best_ask,
            timestamp=time.time(),
        )

        if snapshot is not None:
            await self._emit(
                MarketDataEvent(
                    event_type=MarketDataEventType.TOP_OF_BOOK,
                    token_id=asset_id,
                    snapshot=snapshot,
                    detail="price_change",
                )
            )

    async def _process_best_bid_ask(
        self,
        event: dict,
        asset_id: str,
    ) -> None:
        best_bid = _finite_price(event.get("best_bid"))
        best_ask = _finite_price(event.get("best_ask"))

        if (
            best_bid is None
            or best_ask is None
            or best_bid >= best_ask
        ):
            return

        snapshot = self.orderbooks.publish_ws_top(
            token_id=asset_id,
            best_bid=best_bid,
            best_ask=best_ask,
            timestamp=time.time(),
        )

        if snapshot is not None:
            await self._emit(
                MarketDataEvent(
                    event_type=MarketDataEventType.TOP_OF_BOOK,
                    token_id=asset_id,
                    snapshot=snapshot,
                    detail="best_bid_ask",
                )
            )

    async def _process_book_event(
        self,
        *,
        event_type: str,
        asset_id: str,
        bids: object,
        asks: object,
    ) -> None:
        raw_bids = bids if isinstance(bids, (list, tuple)) else ()
        raw_asks = asks if isinstance(asks, (list, tuple)) else ()

        has_bids = bool(raw_bids)
        has_asks = bool(raw_asks)

        if not has_bids and not has_asks:
            return

        timestamp = time.time()

        if has_bids and has_asks:
            snapshot = self.orderbooks.publish_ws_book(
                token_id=asset_id,
                bids=raw_bids,
                asks=raw_asks,
                timestamp=timestamp,
                source="WS_FULL",
            )
            public_type = MarketDataEventType.BOOK
        else:
            snapshot = self.orderbooks.publish_ws_delta(
                token_id=asset_id,
                bids=raw_bids,
                asks=raw_asks,
                timestamp=timestamp,
            )
            public_type = MarketDataEventType.BOOK_DELTA

        if snapshot is None:
            return

        # Initial coverage requires an actual full, non-crossed WS book. A partial
        # delta or synthetic top update cannot satisfy this observation contract.
        if (
            event_type == "book"
            and snapshot.depth_proven
            and snapshot.best_bid is not None
            and snapshot.best_ask is not None
            and snapshot.best_bid.price < snapshot.best_ask.price
        ):
            self._initial_book_seen.add(asset_id)

        await self._emit(
            MarketDataEvent(
                event_type=public_type,
                token_id=asset_id,
                snapshot=snapshot,
                detail=event_type or "book_update",
            )
        )
