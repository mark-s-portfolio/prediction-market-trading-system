"""
Order-book state and execution-math utilities for the public portfolio edition.

The module separates authoritative WebSocket market-data state from bounded
on-demand REST recovery snapshots.  A REST rescue never silently becomes
WebSocket depth evidence.

No entry thresholds, asset-specific setup logic, position sizing, or historical
trading policy live in this module.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
import math
import time
from typing import Deque, Dict, Iterable, Optional, Sequence, Tuple

import aiohttp

from src.market.types import BookLevel, BookSource, OrderBookSnapshot
from src.runtime.config import AppSettings, settings
from src.runtime.logging import runtime_print


@dataclass(frozen=True, slots=True)
class FillEstimate:
    """Depth-aware estimate for executing a requested quantity."""

    requested_size: float
    filled_size: float
    remaining_size: float
    weighted_average_price: float
    limit_price: float

    @property
    def fully_executable(self) -> bool:
        return self.requested_size > 0.0 and self.remaining_size <= 1e-9


@dataclass(frozen=True, slots=True)
class TopOfBookStats:
    """Small, policy-neutral summary of recent top-of-book movement."""

    count: int
    first_bid: Optional[float]
    last_bid: Optional[float]
    first_ask: Optional[float]
    last_ask: Optional[float]
    bid_range: float
    ask_range: float
    age_seconds: float


@dataclass(frozen=True, slots=True)
class RestBookObservation:
    """REST recovery result plus exact-response side provenance."""

    snapshot: OrderBookSnapshot
    response_had_bids: bool
    response_had_asks: bool
    reason: str = ""


def estimate_fill(
    levels: Sequence[BookLevel] | Iterable[BookLevel],
    size: float,
) -> FillEstimate:
    """Estimate average price, residual quantity and required limit price.

    For a BUY, `limit_price` is the worst ask level that must be crossed to
    execute the returned filled quantity.  For a SELL, callers can pass bids in
    best-to-worst order; the same field then represents the worst accepted bid.
    """

    requested = max(0.0, float(size))
    if requested <= 0.0:
        return FillEstimate(
            requested_size=requested,
            filled_size=0.0,
            remaining_size=requested,
            weighted_average_price=0.0,
            limit_price=0.0,
        )

    remaining = requested
    total_notional = 0.0
    filled = 0.0
    last_executable_price = 0.0

    for level in levels or ():
        try:
            price = float(level.price)
            available = float(level.size)
        except Exception:
            continue

        if not (
            math.isfinite(price)
            and math.isfinite(available)
            and price > 0.0
            and available > 0.0
        ):
            continue

        take = min(remaining, available)
        if take <= 0.0:
            continue

        total_notional += take * price
        filled += take
        remaining -= take
        last_executable_price = price

        if remaining <= 1e-12:
            remaining = 0.0
            break

    weighted = total_notional / filled if filled > 0.0 else 0.0

    return FillEstimate(
        requested_size=requested,
        filled_size=filled,
        remaining_size=max(0.0, remaining),
        weighted_average_price=weighted,
        limit_price=last_executable_price,
    )


def weighted_fill_price(
    levels: Sequence[BookLevel] | Iterable[BookLevel],
    size: float,
) -> Tuple[float, float]:
    """Compatibility helper returning (weighted_average_price, remaining_size)."""

    result = estimate_fill(levels, size)
    return result.weighted_average_price, result.remaining_size


def weighted_fill_price_with_limit(
    levels: Sequence[BookLevel] | Iterable[BookLevel],
    size: float,
) -> Tuple[float, float, float]:
    """Return (weighted_average_price, remaining_size, executable_limit_price)."""

    result = estimate_fill(levels, size)
    return (
        result.weighted_average_price,
        result.remaining_size,
        result.limit_price,
    )


class OrderBookStore:
    """Source-aware in-memory order-book service.

    Design rules:
    1. WebSocket state is authoritative for live scanner depth.
    2. A one-sided WS delta may carry forward the opposite *WS* side.
    3. A top-only WS update proves top prices, not full ladder depth.
    4. On-demand REST recovery is stored separately from WS state.
    5. REST requests are bounded and singleflight per token.
    """

    def __init__(
        self,
        app_settings: AppSettings = settings,
        *,
        history_size: int = 90,
        rest_max_concurrent: int = 2,
    ) -> None:
        self.settings = app_settings

        self._ws_books: Dict[str, OrderBookSnapshot] = {}
        self._rest_books: Dict[str, RestBookObservation] = {}

        self._last_ws_activity: Dict[str, float] = {}
        self._last_rest_activity: Dict[str, float] = {}

        self._history: Dict[
            str,
            Deque[tuple[float, Optional[float], Optional[float]]],
        ] = defaultdict(lambda: deque(maxlen=max(2, int(history_size))))

        self._rest_tasks: Dict[str, asyncio.Task[Optional[RestBookObservation]]] = {}
        self._rest_gate = asyncio.Semaphore(max(1, int(rest_max_concurrent)))
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "OrderBookStore":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def start(self) -> None:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=max(
                    1.0,
                    float(self.settings.network.connect_timeout_seconds)
                    + float(self.settings.network.read_timeout_seconds),
                ),
                connect=max(
                    0.1,
                    float(self.settings.network.connect_timeout_seconds),
                ),
                sock_read=max(
                    0.1,
                    float(self.settings.network.read_timeout_seconds),
                ),
            )
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        tasks = tuple(self._rest_tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._rest_tasks.clear()

        if self._session is not None and not self._session.closed:
            await self._session.close()

        self._session = None

    @staticmethod
    def _normalize_levels(
        levels: Sequence[object] | Iterable[object],
        *,
        bids: bool,
    ) -> Tuple[BookLevel, ...]:
        parsed = []

        for raw in levels or ():
            try:
                parsed.append(BookLevel.from_raw(raw))
            except (TypeError, ValueError):
                continue

        parsed.sort(key=lambda level: level.price, reverse=bids)
        return tuple(parsed)

    @staticmethod
    def _top_prices(
        snapshot: OrderBookSnapshot,
    ) -> tuple[Optional[float], Optional[float]]:
        best_bid = snapshot.best_bid.price if snapshot.best_bid else None
        best_ask = snapshot.best_ask.price if snapshot.best_ask else None
        return best_bid, best_ask

    def _record_top(self, snapshot: OrderBookSnapshot) -> None:
        bid, ask = self._top_prices(snapshot)
        if bid is None and ask is None:
            return

        history = self._history[snapshot.token_id]
        if history:
            last_ts, last_bid, last_ask = history[-1]
            same_bid = (
                (last_bid is None and bid is None)
                or (
                    last_bid is not None
                    and bid is not None
                    and abs(last_bid - bid) < 5e-4
                )
            )
            same_ask = (
                (last_ask is None and ask is None)
                or (
                    last_ask is not None
                    and ask is not None
                    and abs(last_ask - ask) < 5e-4
                )
            )
            if same_bid and same_ask and snapshot.timestamp - last_ts < 0.025:
                return

        history.append((snapshot.timestamp, bid, ask))

    def publish_ws_book(
        self,
        *,
        token_id: str,
        bids: Sequence[object] | Iterable[object],
        asks: Sequence[object] | Iterable[object],
        timestamp: Optional[float] = None,
        source: str = "WS_FULL",
    ) -> Optional[OrderBookSnapshot]:
        """Publish a full WebSocket depth snapshot."""

        token_id = str(token_id or "").strip()
        if not token_id:
            return None

        parsed_bids = self._normalize_levels(bids, bids=True)
        parsed_asks = self._normalize_levels(asks, bids=False)

        if not parsed_bids and not parsed_asks:
            return None

        ts = float(timestamp or time.time())

        snapshot = OrderBookSnapshot(
            token_id=token_id,
            bids=parsed_bids,
            asks=parsed_asks,
            timestamp=ts,
            source=BookSource.WEBSOCKET,
            depth_proven=True,
            synthetic_depth=False,
        )

        self._ws_books[token_id] = snapshot
        self._last_ws_activity[token_id] = ts
        self._record_top(snapshot)
        return snapshot

    def publish_ws_delta(
        self,
        *,
        token_id: str,
        bids: Sequence[object] | Iterable[object] = (),
        asks: Sequence[object] | Iterable[object] = (),
        timestamp: Optional[float] = None,
    ) -> Optional[OrderBookSnapshot]:
        """Apply a one- or two-sided WS delta without importing REST state.

        If one side is omitted, only the prior WebSocket side may be carried
        forward.  This preserves source affinity across partial market-data events.
        """

        token_id = str(token_id or "").strip()
        if not token_id:
            return None

        incoming_bids = self._normalize_levels(bids, bids=True)
        incoming_asks = self._normalize_levels(asks, bids=False)

        if not incoming_bids and not incoming_asks:
            return None

        previous = self._ws_books.get(token_id)

        merged_bids = (
            incoming_bids
            if incoming_bids
            else (previous.bids if previous is not None else ())
        )
        merged_asks = (
            incoming_asks
            if incoming_asks
            else (previous.asks if previous is not None else ())
        )

        if not merged_bids and not merged_asks:
            return None

        ts = float(timestamp or time.time())

        # A partial WS delta retains depth provenance only when the missing side
        # came from a previous genuine WS depth snapshot.
        missing_side = not incoming_bids or not incoming_asks
        prior_depth_proven = bool(previous and previous.depth_proven)
        depth_proven = bool(
            (incoming_bids and incoming_asks)
            or (missing_side and prior_depth_proven)
        )

        snapshot = OrderBookSnapshot(
            token_id=token_id,
            bids=tuple(merged_bids),
            asks=tuple(merged_asks),
            timestamp=ts,
            source=BookSource.WEBSOCKET,
            depth_proven=depth_proven,
            synthetic_depth=not depth_proven,
        )

        self._ws_books[token_id] = snapshot
        self._last_ws_activity[token_id] = ts
        self._record_top(snapshot)
        return snapshot

    def publish_ws_top(
        self,
        *,
        token_id: str,
        best_bid: Optional[float],
        best_ask: Optional[float],
        timestamp: Optional[float] = None,
    ) -> Optional[OrderBookSnapshot]:
        """Publish top prices without carrying stale ladder depth forward.

        A top-of-book event proves only the reported best prices. It does not
        prove that any previously observed deeper levels still exist, nor does it
        refresh their sizes. The stored snapshot is therefore deliberately
        synthetic, one level per reported side, and never depth-proven.
        """

        token_id = str(token_id or "").strip()
        if not token_id:
            return None

        bids: list[BookLevel] = []
        asks: list[BookLevel] = []

        if best_bid is not None:
            bid_px = float(best_bid)
            if not math.isfinite(bid_px) or bid_px <= 0.0:
                return None
            bids.append(BookLevel(bid_px, 1.0))

        if best_ask is not None:
            ask_px = float(best_ask)
            if not math.isfinite(ask_px) or ask_px <= 0.0:
                return None
            asks.append(BookLevel(ask_px, 1.0))

        if not bids and not asks:
            return None

        ts = float(timestamp or time.time())

        snapshot = OrderBookSnapshot(
            token_id=token_id,
            bids=tuple(bids),
            asks=tuple(asks),
            timestamp=ts,
            source=BookSource.WEBSOCKET,
            depth_proven=False,
            synthetic_depth=True,
        )

        self._ws_books[token_id] = snapshot
        self._last_ws_activity[token_id] = ts
        self._record_top(snapshot)
        return snapshot

    def ws_book(self, token_id: str) -> Optional[OrderBookSnapshot]:
        return self._ws_books.get(str(token_id or ""))

    def rest_book(self, token_id: str) -> Optional[RestBookObservation]:
        return self._rest_books.get(str(token_id or ""))

    def latest_ws_activity(self, token_id: str) -> Optional[float]:
        return self._last_ws_activity.get(str(token_id or ""))

    def latest_rest_activity(self, token_id: str) -> Optional[float]:
        return self._last_rest_activity.get(str(token_id or ""))

    def ws_age_seconds(
        self,
        token_id: str,
        *,
        now: Optional[float] = None,
    ) -> float:
        snapshot = self.ws_book(token_id)
        if snapshot is None:
            return math.inf
        return snapshot.age_seconds(float(now or time.time()))

    def top_stats(
        self,
        token_id: str,
        *,
        lookback_seconds: float = 2.0,
        now: Optional[float] = None,
    ) -> TopOfBookStats:
        """Return recent movement metrics without making a trading decision."""

        current_time = float(now or time.time())
        history = list(self._history.get(str(token_id or ""), ()))

        if not history:
            return TopOfBookStats(
                count=0,
                first_bid=None,
                last_bid=None,
                first_ask=None,
                last_ask=None,
                bid_range=0.0,
                ask_range=0.0,
                age_seconds=math.inf,
            )

        recent = [
            row
            for row in history
            if current_time - row[0] <= max(0.0, float(lookback_seconds))
        ]
        if not recent:
            recent = [history[-1]]

        bids = [float(row[1]) for row in recent if row[1] is not None]
        asks = [float(row[2]) for row in recent if row[2] is not None]

        return TopOfBookStats(
            count=len(recent),
            first_bid=bids[0] if bids else None,
            last_bid=bids[-1] if bids else None,
            first_ask=asks[0] if asks else None,
            last_ask=asks[-1] if asks else None,
            bid_range=(max(bids) - min(bids)) if bids else 0.0,
            ask_range=(max(asks) - min(asks)) if asks else 0.0,
            age_seconds=max(0.0, current_time - recent[-1][0]),
        )

    async def refresh_rest_once(
        self,
        token_id: str,
        *,
        reason: str = "",
        timeout_seconds: Optional[float] = None,
    ) -> Optional[RestBookObservation]:
        """Singleflight on-demand CLOB REST refresh.

        The result is written only to the REST recovery cache. It never mutates
        the authoritative WebSocket book or its depth provenance.
        """

        token_id = str(token_id or "").strip()
        if not token_id:
            return None

        existing = self._rest_tasks.get(token_id)
        if existing is not None and not existing.done():
            try:
                return await asyncio.shield(existing)
            except Exception:
                return None

        async def _fetch() -> Optional[RestBookObservation]:
            if self._session is None or self._session.closed:
                await self.start()

            assert self._session is not None

            try:
                async with self._rest_gate:
                    request_timeout = aiohttp.ClientTimeout(
                        total=max(
                            0.1,
                            float(
                                timeout_seconds
                                if timeout_seconds is not None
                                else self.settings.network.read_timeout_seconds
                            ),
                        )
                    )

                    async with self._session.get(
                        f"{self.settings.network.clob_api_url.rstrip('/')}/book",
                        params={"token_id": token_id},
                        timeout=request_timeout,
                    ) as response:
                        response.raise_for_status()
                        payload = await response.json()

                raw_bids = payload.get("bids", []) if isinstance(payload, dict) else []
                raw_asks = payload.get("asks", []) if isinstance(payload, dict) else []

                bids = self._normalize_levels(raw_bids, bids=True)
                asks = self._normalize_levels(raw_asks, bids=False)

                if not bids and not asks:
                    return None

                previous = self._rest_books.get(token_id)

                merged_bids = (
                    bids
                    if bids
                    else (
                        previous.snapshot.bids
                        if previous is not None
                        else ()
                    )
                )
                merged_asks = (
                    asks
                    if asks
                    else (
                        previous.snapshot.asks
                        if previous is not None
                        else ()
                    )
                )

                timestamp = time.time()

                snapshot = OrderBookSnapshot(
                    token_id=token_id,
                    bids=tuple(merged_bids),
                    asks=tuple(merged_asks),
                    timestamp=timestamp,
                    source=BookSource.REST_ON_DEMAND,
                    depth_proven=False,
                    synthetic_depth=False,
                )

                observation = RestBookObservation(
                    snapshot=snapshot,
                    response_had_bids=bool(bids),
                    response_had_asks=bool(asks),
                    reason=str(reason or ""),
                )

                self._rest_books[token_id] = observation
                self._last_rest_activity[token_id] = timestamp
                return observation

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                runtime_print(
                    f"[orderbook] REST refresh failed for {token_id[:8]}: "
                    f"{type(exc).__name__}: {exc}"
                )
                return None

        task = asyncio.create_task(
            _fetch(),
            name=f"rest-book-{token_id[:8]}",
        )
        self._rest_tasks[token_id] = task

        def _release(done_task: asyncio.Task[Optional[RestBookObservation]]) -> None:
            if self._rest_tasks.get(token_id) is done_task:
                self._rest_tasks.pop(token_id, None)

            if done_task.cancelled():
                return

            try:
                done_task.exception()
            except Exception:
                pass

        task.add_done_callback(_release)

        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # The underlying task retains strong ownership until its done callback.
            raise
        except Exception:
            return None

    async def get_book(
        self,
        token_id: str,
        *,
        max_ws_age_seconds: Optional[float] = None,
        allow_rest_fallback: bool = False,
        rest_reason: str = "",
    ) -> Optional[OrderBookSnapshot]:
        """WS-first book accessor with optional separate REST recovery.

        Freshness is caller-supplied because acceptable age is a consumer concern,
        not a hidden policy of the market-data layer.
        """

        token_id = str(token_id or "").strip()
        if not token_id:
            return None

        ws = self.ws_book(token_id)
        if ws is not None:
            if max_ws_age_seconds is None:
                return ws
            if ws.age_seconds(time.time()) <= max(0.0, float(max_ws_age_seconds)):
                return ws

        if not allow_rest_fallback:
            return None

        rest = await self.refresh_rest_once(
            token_id,
            reason=rest_reason,
        )
        return rest.snapshot if rest is not None else None

    def prune(self, active_tokens: Iterable[str]) -> None:
        """Bound token-keyed state across market rotations."""

        keep = {str(token) for token in active_tokens if str(token)}

        for table in (
            self._ws_books,
            self._rest_books,
            self._last_ws_activity,
            self._last_rest_activity,
            self._history,
        ):
            for token_id in list(table.keys()):
                if token_id not in keep:
                    table.pop(token_id, None)

        # Do not cancel in-flight REST work here.  It remains strongly owned until
        # completion; its done callback removes the task entry safely.
