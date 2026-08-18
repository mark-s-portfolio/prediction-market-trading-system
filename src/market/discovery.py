"""
Resilient market discovery for the public portfolio edition.

This module preserves the infrastructure ideas from the private system:
- exact sprint-slug discovery
- bounded metadata caching
- retry/backoff around Gamma
- short circuit-breaking after 403/429 responses
- exact-slug CLOB fallback
- same-window discovery reuse
- explicit venue-confirmed resolution lookup

It intentionally contains no entry/admission rules, market-quality thresholds,
position sizing, historical setup families, or asset-specific trading edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import time
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import requests

from src.market.types import MarketDefinition, MarketResolution
from src.runtime.config import AppSettings, settings
from src.runtime.logging import runtime_print


DEFAULT_MARKET_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "bitcoin": ("bitcoin", "btc"),
    "ethereum": ("ethereum", "eth"),
    "solana": ("solana", "sol"),
    "xrp": ("xrp",),
    "doge": ("doge", "dogecoin"),
}


@dataclass(frozen=True, slots=True)
class DiscoveryPolicy:
    """Infrastructure-only discovery behavior.

    None of these values decide whether a discovered market should be traded.
    """

    slug_cache_ttl_seconds: float = 20 * 60
    slug_cache_max_entries: int = 64
    same_window_reuse_seconds: float = 45.0
    recent_discovery_fallback_seconds: float = 5 * 60
    http_circuit_breaker_seconds: float = 20.0
    clob_fallback_pages: int = 4
    discovery_attempts: int = 3
    retry_sleep_seconds: float = 1.0


@dataclass(frozen=True, slots=True)
class _ResolvedMarket:
    market: MarketDefinition
    cached: bool = False
    source: str = "GAMMA"


def _coerce_optional_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def _decode_sequence(value: object) -> list:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except Exception:
            return []
        return list(decoded) if isinstance(decoded, (list, tuple)) else []

    if isinstance(value, (list, tuple)):
        return list(value)

    return []


def _parse_end_time(slug: str, end_date: object, interval_minutes: int) -> datetime:
    """Resolve market end time from venue metadata, then exact sprint identity.

    The fallback is deliberately strict: it derives an end only from a slug whose
    final component is a Unix epoch. It never guesses from a title or current time.
    """

    if isinstance(end_date, datetime):
        dt = end_date
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

    if end_date not in (None, ""):
        try:
            text = str(end_date).strip()
            if text:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
                return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        except Exception:
            pass

    parts = str(slug or "").rsplit("-", 1)
    if len(parts) != 2 or not parts[1].isdigit() or len(parts[1]) != 10:
        raise ValueError(f"market end time unavailable for slug: {slug!r}")

    start_ts = int(parts[1])
    end_ts = start_ts + int(interval_minutes) * 60
    return datetime.fromtimestamp(end_ts, tz=timezone.utc)


def current_sprint_slug(market_alias: str, now_utc: datetime, interval_minutes: int) -> str:
    """Build the exact current sprint slug used by Polymarket crypto Up/Down markets."""

    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)

    window_seconds = int(interval_minutes) * 60
    if window_seconds <= 0:
        raise ValueError("interval_minutes must be positive")

    # Sprint slugs are epoch-window based. Epoch timestamps are timezone invariant;
    # explicit UTC use makes that fact clear and avoids manual DST calculations.
    timestamp = int(now_utc.timestamp())
    start = (timestamp // window_seconds) * window_seconds
    return f"{market_alias}-updown-{interval_minutes}m-{start}"


class MarketDiscovery:
    """Discover active binary markets with bounded resilient fallbacks."""

    def __init__(
        self,
        app_settings: AppSettings = settings,
        policy: DiscoveryPolicy = DiscoveryPolicy(),
        aliases: Mapping[str, Tuple[str, ...]] = DEFAULT_MARKET_ALIASES,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.settings = app_settings
        self.policy = policy
        self.aliases = dict(aliases)
        self.session = session or requests.Session()

        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Origin": "https://polymarket.com",
                "Referer": "https://polymarket.com/",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            }
        )

        self._slug_cache: Dict[str, tuple[float, MarketDefinition]] = {}
        self._last_discovered: Tuple[MarketDefinition, ...] = ()
        self._last_discovered_at: float = 0.0
        self._http_blocked_until: float = 0.0
        self._http_blocked_code: Optional[int] = None

    @property
    def gamma_markets_url(self) -> str:
        return f"{self.settings.network.gamma_api_url.rstrip('/')}/markets"

    @property
    def clob_markets_url(self) -> str:
        return f"{self.settings.network.clob_api_url.rstrip('/')}/markets"

    def _cache_market(self, market: MarketDefinition) -> None:
        now = time.time()
        ttl = max(1.0, float(self.policy.slug_cache_ttl_seconds))

        for slug, (created_at, _) in list(self._slug_cache.items()):
            if now - created_at > ttl:
                self._slug_cache.pop(slug, None)

        self._slug_cache[market.slug] = (now, market)

        cap = max(16, int(self.policy.slug_cache_max_entries))
        if len(self._slug_cache) > cap:
            oldest = sorted(self._slug_cache.items(), key=lambda item: item[1][0])
            excess = len(self._slug_cache) - cap
            for slug, _ in oldest[:excess]:
                self._slug_cache.pop(slug, None)

    def _cached_market(self, slug: str) -> Optional[MarketDefinition]:
        item = self._slug_cache.get(str(slug))
        if item is None:
            return None

        created_at, market = item
        if time.time() - created_at > float(self.policy.slug_cache_ttl_seconds):
            self._slug_cache.pop(str(slug), None)
            return None

        return market

    def _parse_gamma_market(
        self,
        *,
        slug: str,
        payload: object,
        interval_minutes: int,
    ) -> Optional[MarketDefinition]:
        if not payload:
            return None

        market = payload[0] if isinstance(payload, list) else payload
        if not isinstance(market, dict):
            return None

        tokens = _decode_sequence(market.get("clobTokenIds", []))
        outcomes = _decode_sequence(market.get("outcomes", ["Yes", "No"]))

        if len(tokens) < 2:
            return None

        yes_index = next(
            (i for i, outcome in enumerate(outcomes)
             if str(outcome).strip().lower() in {"yes", "up"}),
            0,
        )
        no_index = next(
            (i for i, outcome in enumerate(outcomes)
             if str(outcome).strip().lower() in {"no", "down"}),
            1 if yes_index == 0 else 0,
        )

        if yes_index >= len(tokens) or no_index >= len(tokens):
            return None

        yes_token = str(tokens[yes_index]).strip()
        no_token = str(tokens[no_index]).strip()
        condition_id = str(
            market.get("conditionId") or market.get("condition_id") or ""
        ).strip()

        if not yes_token or not no_token or yes_token == no_token or not condition_id:
            return None

        try:
            end_time = _parse_end_time(
                slug,
                market.get("endDate", market.get("end_date")),
                interval_minutes,
            )
        except ValueError:
            return None

        tick_raw = (
            market.get("orderPriceMinTickSize")
            or market.get("minimum_tick_size")
            or market.get("tickSize")
        )
        try:
            tick_size = float(tick_raw) if tick_raw not in (None, "") else None
        except (TypeError, ValueError):
            tick_size = None

        return MarketDefinition(
            slug=slug,
            question=str(market.get("question") or slug),
            yes_token=yes_token,
            no_token=no_token,
            condition_id=condition_id,
            interval_minutes=int(interval_minutes),
            end_time=end_time,
            tick_size=tick_size,
            neg_risk=_coerce_optional_bool(
                market.get("negRisk")
                if "negRisk" in market
                else market.get("neg_risk")
            ),
        )

    def _parse_clob_market(
        self,
        *,
        slug: str,
        payload: object,
        interval_minutes: int,
    ) -> Optional[MarketDefinition]:
        """Parse only an exact slug match from a CLOB market response."""

        if not isinstance(payload, dict):
            return None

        possible_slugs = {
            str(value)
            for value in (
                payload.get("market_slug"),
                payload.get("slug"),
                payload.get("marketSlug"),
                payload.get("event_slug"),
                payload.get("eventSlug"),
                payload.get("condition_slug"),
            )
            if value
        }
        if slug not in possible_slugs:
            return None

        tokens = payload.get("tokens") or payload.get("clobTokenIds") or []
        outcomes = _decode_sequence(payload.get("outcomes") or [])

        if isinstance(tokens, str):
            tokens = _decode_sequence(tokens)

        parsed: list[tuple[str, str]] = []

        if isinstance(tokens, list) and tokens and isinstance(tokens[0], dict):
            for token in tokens:
                token_id = token.get("token_id") or token.get("tokenId") or token.get("id")
                outcome = token.get("outcome") or token.get("name") or ""
                if token_id:
                    parsed.append(
                        (str(outcome).strip().lower(), str(token_id).strip())
                    )
        elif isinstance(tokens, list):
            for index, token_id in enumerate(tokens):
                fallback_outcome = "Yes" if index == 0 else "No"
                outcome = outcomes[index] if index < len(outcomes) else fallback_outcome
                parsed.append(
                    (str(outcome).strip().lower(), str(token_id).strip())
                )

        if len(parsed) < 2:
            return None

        yes_token = next(
            (token_id for outcome, token_id in parsed if outcome in {"yes", "up"}),
            parsed[0][1],
        )
        no_token = next(
            (token_id for outcome, token_id in parsed if outcome in {"no", "down"}),
            parsed[1][1],
        )

        condition_id = str(
            payload.get("condition_id") or payload.get("conditionId") or ""
        ).strip()

        if not yes_token or not no_token or yes_token == no_token or not condition_id:
            return None

        try:
            end_time = _parse_end_time(
                slug,
                payload.get("end_date_iso")
                or payload.get("endDate")
                or payload.get("end_date"),
                interval_minutes,
            )
        except ValueError:
            return None

        tick_raw = (
            payload.get("minimum_tick_size")
            or payload.get("orderPriceMinTickSize")
            or payload.get("tick_size")
        )
        try:
            tick_size = float(tick_raw) if tick_raw not in (None, "") else None
        except (TypeError, ValueError):
            tick_size = None

        return MarketDefinition(
            slug=slug,
            question=str(payload.get("question") or payload.get("title") or slug),
            yes_token=yes_token,
            no_token=no_token,
            condition_id=condition_id,
            interval_minutes=int(interval_minutes),
            end_time=end_time,
            tick_size=tick_size,
            neg_risk=_coerce_optional_bool(
                payload.get("neg_risk")
                if "neg_risk" in payload
                else payload.get("negRisk")
            ),
        )

    def _resolve_from_clob(
        self,
        *,
        slug: str,
        interval_minutes: int,
    ) -> Optional[MarketDefinition]:
        """Best-effort exact-slug fallback when Gamma is unavailable."""

        cursor = ""
        timeout = (
            float(self.settings.network.connect_timeout_seconds),
            float(self.settings.network.read_timeout_seconds),
        )

        for page in range(max(1, int(self.policy.clob_fallback_pages))):
            try:
                params = {"next_cursor": cursor} if cursor else {}
                response = self.session.get(
                    self.clob_markets_url,
                    params=params,
                    timeout=timeout,
                )
                response.raise_for_status()

                payload = response.json()
                rows = payload.get("data", payload) if isinstance(payload, dict) else payload
                if not isinstance(rows, list):
                    return None

                for row in rows:
                    market = self._parse_clob_market(
                        slug=slug,
                        payload=row,
                        interval_minutes=interval_minutes,
                    )
                    if market is not None:
                        self._cache_market(market)
                        runtime_print(
                            f"[discovery] exact CLOB fallback resolved {slug} "
                            f"(page {page + 1})"
                        )
                        return market

                if not isinstance(payload, dict):
                    break

                cursor = str(
                    payload.get("next_cursor") or payload.get("nextCursor") or ""
                )
                if not cursor or cursor.upper() == "LTE=":
                    break

            except Exception as exc:
                runtime_print(
                    f"[discovery] CLOB fallback failed for {slug}: "
                    f"{type(exc).__name__}: {exc}"
                )
                return None

        return None

    def resolve_slug(
        self,
        slug: str,
        interval_minutes: int,
    ) -> Optional[_ResolvedMarket]:
        """Resolve one exact sprint slug to canonical market metadata."""

        slug = str(slug or "").strip()
        if not slug:
            return None

        cached = self._cached_market(slug)
        now = time.time()

        if self._http_blocked_until > now:
            if cached is not None:
                return _ResolvedMarket(cached, cached=True, source="CACHE")

            fallback = self._resolve_from_clob(
                slug=slug,
                interval_minutes=interval_minutes,
            )
            return (
                _ResolvedMarket(fallback, source="CLOB")
                if fallback is not None
                else None
            )

        attempts = max(1, int(self.settings.network.retry_attempts))
        last_error: Optional[BaseException] = None

        for attempt in range(1, attempts + 1):
            try:
                response = self.session.get(
                    self.gamma_markets_url,
                    params={"slug": slug},
                    timeout=(
                        float(self.settings.network.connect_timeout_seconds),
                        float(self.settings.network.read_timeout_seconds),
                    ),
                )

                if response.status_code in {403, 429}:
                    self._http_blocked_code = int(response.status_code)
                    self._http_blocked_until = (
                        time.time()
                        + float(self.policy.http_circuit_breaker_seconds)
                    )

                    fallback = self._resolve_from_clob(
                        slug=slug,
                        interval_minutes=interval_minutes,
                    )
                    if fallback is not None:
                        return _ResolvedMarket(fallback, source="CLOB")
                    break

                response.raise_for_status()

                market = self._parse_gamma_market(
                    slug=slug,
                    payload=response.json(),
                    interval_minutes=interval_minutes,
                )
                if market is not None:
                    self._cache_market(market)
                    return _ResolvedMarket(market, source="GAMMA")

                # An empty exact-slug response is a real miss. A recent cached copy
                # remains preferable to inventing/fuzzy-matching a different market.
                return (
                    _ResolvedMarket(cached, cached=True, source="CACHE")
                    if cached is not None
                    else None
                )

            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(
                        max(
                            0.0,
                            float(self.settings.network.retry_backoff_seconds)
                            * attempt,
                        )
                    )
                    continue
            except Exception as exc:
                last_error = exc
                break

        if cached is not None:
            runtime_print(
                f"[discovery] using cached metadata for {slug} after "
                f"{type(last_error).__name__ if last_error else 'HTTP block'}"
            )
            return _ResolvedMarket(cached, cached=True, source="CACHE")

        if self._http_blocked_code in {403, 429} and self._http_blocked_until > time.time():
            runtime_print(
                f"[discovery] Gamma HTTP {self._http_blocked_code}; "
                f"no exact fallback for {slug}"
            )
        elif last_error is not None:
            runtime_print(
                f"[discovery] failed to resolve {slug}: "
                f"{type(last_error).__name__}: {last_error}"
            )

        return None

    def _try_asset_interval(
        self,
        *,
        asset: str,
        interval_minutes: int,
        now_utc: datetime,
    ) -> Optional[_ResolvedMarket]:
        aliases = self.aliases.get(asset, (asset,))

        for alias in aliases:
            slug = current_sprint_slug(alias, now_utc, interval_minutes)
            resolved = self.resolve_slug(slug, interval_minutes)
            if resolved is not None:
                return resolved

        return None

    def expected_current_slugs(
        self,
        now_utc: Optional[datetime] = None,
    ) -> set[str]:
        now_utc = now_utc or datetime.now(timezone.utc)
        slugs: set[str] = set()

        for asset in self.settings.market_data.assets:
            aliases = self.aliases.get(asset, (asset,))
            for interval in self.settings.market_data.intervals_minutes:
                for alias in aliases:
                    slugs.add(current_sprint_slug(alias, now_utc, interval))

        return slugs

    def _reuse_recent_same_window(
        self,
        now_utc: datetime,
    ) -> Optional[Tuple[MarketDefinition, ...]]:
        if not self._last_discovered:
            return None

        age = time.time() - self._last_discovered_at
        if not 0.0 <= age <= float(self.policy.same_window_reuse_seconds):
            return None

        expected_slugs = self.expected_current_slugs(now_utc)
        current = tuple(
            market
            for market in self._last_discovered
            if market.slug in expected_slugs
        )

        expected_market_count = (
            len(self.settings.market_data.assets)
            * len(self.settings.market_data.intervals_minutes)
        )

        if len(current) >= expected_market_count:
            runtime_print(
                f"[discovery] reusing {len(current)} same-window markets "
                f"(age {age:.1f}s)"
            )
            return current

        return None

    def _merge_recent_fallback(
        self,
        markets: Sequence[MarketDefinition],
        now_utc: datetime,
    ) -> Tuple[MarketDefinition, ...]:
        if not self._last_discovered:
            return tuple(markets)

        age = time.time() - self._last_discovered_at
        if age > float(self.policy.recent_discovery_fallback_seconds):
            return tuple(markets)

        expected_slugs = self.expected_current_slugs(now_utc)
        by_slug = {market.slug: market for market in markets}

        for prior in self._last_discovered:
            if prior.slug in expected_slugs:
                by_slug.setdefault(prior.slug, prior)

        return tuple(by_slug.values())

    def discover_active_markets(
        self,
        now_utc: Optional[datetime] = None,
    ) -> Tuple[MarketDefinition, ...]:
        """Discover the configured current sprint markets.

        Discovery determines market identity and metadata only. Whether any returned
        market is suitable for execution belongs to the strategy/risk layers.
        """

        now_utc = now_utc or datetime.now(timezone.utc)
        reused = self._reuse_recent_same_window(now_utc)
        if reused is not None:
            return reused

        expected_market_count = (
            len(self.settings.market_data.assets)
            * len(self.settings.market_data.intervals_minutes)
        )

        markets: Tuple[MarketDefinition, ...] = ()

        for attempt in range(max(1, int(self.policy.discovery_attempts))):
            discovered: Dict[str, MarketDefinition] = {}

            for asset in self.settings.market_data.assets:
                for interval in self.settings.market_data.intervals_minutes:
                    resolved = self._try_asset_interval(
                        asset=asset,
                        interval_minutes=int(interval),
                        now_utc=now_utc,
                    )
                    if resolved is None:
                        continue

                    market = resolved.market
                    discovered.setdefault(market.slug, market)

                    cache_note = " cache" if resolved.cached else ""
                    runtime_print(
                        f"[discovery] found{cache_note}: {market.question}"
                    )

            markets = self._merge_recent_fallback(
                tuple(discovered.values()),
                now_utc,
            )

            if len(markets) >= expected_market_count:
                break

            if attempt < int(self.policy.discovery_attempts) - 1:
                sleep_seconds = max(0.0, float(self.policy.retry_sleep_seconds))
                runtime_print(
                    f"[discovery] found {len(markets)}/{expected_market_count}; "
                    f"retrying in {sleep_seconds:.1f}s"
                )
                if sleep_seconds:
                    time.sleep(sleep_seconds)

        if markets:
            self._last_discovered = tuple(markets)
            self._last_discovered_at = time.time()

        runtime_print(f"[discovery] active markets: {len(markets)}")
        return tuple(markets)

    def resolve_market_resolution(
        self,
        market: MarketDefinition,
    ) -> MarketResolution:
        """Perform a fresh venue lookup and confirm only an explicit binary terminal state.

        A winner is never inferred from the live order book.
        """

        try:
            response = self.session.get(
                self.gamma_markets_url,
                params={"slug": market.slug},
                timeout=(
                    float(self.settings.network.connect_timeout_seconds),
                    float(self.settings.network.read_timeout_seconds),
                ),
            )
            response.raise_for_status()

            payload = response.json()
            row = payload[0] if isinstance(payload, list) and payload else payload
            if not isinstance(row, dict):
                return MarketResolution(market=market, confirmed=False)

            outcomes = _decode_sequence(row.get("outcomes", ["Yes", "No"]))
            prices = _decode_sequence(
                row.get("outcomePrices", row.get("outcome_prices", []))
            )

            if len(outcomes) < 2 or len(prices) < 2:
                return MarketResolution(market=market, confirmed=False)

            yes_index = next(
                (i for i, outcome in enumerate(outcomes)
                 if str(outcome).strip().lower() in {"yes", "up"}),
                0,
            )
            no_index = next(
                (i for i, outcome in enumerate(outcomes)
                 if str(outcome).strip().lower() in {"no", "down"}),
                1 if yes_index == 0 else 0,
            )

            yes_price = float(prices[yes_index])
            no_price = float(prices[no_index])

            if not (
                math.isfinite(yes_price)
                and math.isfinite(no_price)
                and 0.0 <= yes_price <= 1.0
                and 0.0 <= no_price <= 1.0
            ):
                return MarketResolution(market=market, confirmed=False)

            closed = bool(_coerce_optional_bool(row.get("closed")))
            resolved = bool(
                _coerce_optional_bool(
                    row.get("resolved")
                    if "resolved" in row
                    else row.get("isResolved", row.get("is_resolved"))
                )
            )

            binary_terminal = (
                (yes_price >= 0.999 and no_price <= 0.001)
                or (no_price >= 0.999 and yes_price <= 0.001)
            )

            return MarketResolution(
                market=market,
                confirmed=bool((closed or resolved) and binary_terminal),
                yes_price=yes_price,
                no_price=no_price,
            )

        except Exception:
            return MarketResolution(market=market, confirmed=False)
