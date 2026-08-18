"""
Public portfolio domain types for prediction-market market data.

These models deliberately contain no trading thresholds, admission policy,
position sizing rules, or asset-specific strategy knowledge. They define the
stable data contracts shared by discovery, market-data, execution, and risk
layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Iterable, Optional, Sequence, Tuple


class OutcomeSide(str, Enum):
    """Binary prediction-market outcome side."""

    YES = "YES"
    NO = "NO"

    @classmethod
    def from_value(cls, value: str) -> "OutcomeSide":
        normalized = str(value or "").strip().upper()
        if normalized in {"YES", "UP"}:
            return cls.YES
        if normalized in {"NO", "DOWN"}:
            return cls.NO
        raise ValueError(f"unsupported outcome side: {value!r}")


class BookSource(str, Enum):
    """Origin of an order-book observation."""

    WEBSOCKET = "WEBSOCKET"
    REST_BOOTSTRAP = "REST_BOOTSTRAP"
    REST_ON_DEMAND = "REST_ON_DEMAND"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_value(cls, value: str) -> "BookSource":
        normalized = str(value or "").strip().upper()
        if normalized.startswith("WS"):
            return cls.WEBSOCKET
        if normalized.startswith("REST_BOOTSTRAP"):
            return cls.REST_BOOTSTRAP
        if normalized.startswith("REST_ON_DEMAND"):
            return cls.REST_ON_DEMAND
        return cls.UNKNOWN


@dataclass(frozen=True, slots=True, order=True)
class BookLevel:
    """One price/size level in a CLOB order book."""

    price: float
    size: float

    def __post_init__(self) -> None:
        price = float(self.price)
        size = float(self.size)

        if not math.isfinite(price) or not 0.0 < price <= 1.0:
            raise ValueError(f"invalid prediction-market price: {price!r}")
        if not math.isfinite(size) or size <= 0.0:
            raise ValueError(f"invalid order-book size: {size!r}")

        object.__setattr__(self, "price", price)
        object.__setattr__(self, "size", size)

    @classmethod
    def from_raw(cls, value: object) -> "BookLevel":
        """Normalize tuple/list or mapping-shaped API levels."""

        if isinstance(value, dict):
            return cls(float(value.get("price", 0.0)), float(value.get("size", 0.0)))

        if isinstance(value, (tuple, list)) and len(value) >= 2:
            return cls(float(value[0]), float(value[1]))

        raise TypeError(f"unsupported book level: {value!r}")


def _normalize_levels(
    levels: Iterable[object],
    *,
    descending: bool,
) -> Tuple[BookLevel, ...]:
    parsed = []
    for raw in levels or ():
        try:
            parsed.append(BookLevel.from_raw(raw))
        except (TypeError, ValueError):
            continue

    parsed.sort(key=lambda level: level.price, reverse=descending)
    return tuple(parsed)


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    """Timestamped, source-aware order-book snapshot for one outcome token.

    `depth_proven` differentiates a real depth ladder from a top-of-book-only
    observation whose sizes may not be suitable for liquidity calculations.
    """

    token_id: str
    bids: Tuple[BookLevel, ...] = field(default_factory=tuple)
    asks: Tuple[BookLevel, ...] = field(default_factory=tuple)
    timestamp: float = 0.0
    source: BookSource = BookSource.UNKNOWN
    depth_proven: bool = False
    synthetic_depth: bool = False

    def __post_init__(self) -> None:
        token_id = str(self.token_id or "").strip()
        if not token_id:
            raise ValueError("token_id is required")

        timestamp = float(self.timestamp)
        if not math.isfinite(timestamp) or timestamp <= 0.0:
            raise ValueError("timestamp must be a positive Unix timestamp")

        object.__setattr__(self, "token_id", token_id)
        object.__setattr__(self, "timestamp", timestamp)

    @classmethod
    def from_raw(
        cls,
        *,
        token_id: str,
        bids: Sequence[object],
        asks: Sequence[object],
        timestamp: float,
        source: str | BookSource,
        depth_proven: bool = False,
        synthetic_depth: bool = False,
    ) -> "OrderBookSnapshot":
        parsed_source = (
            source if isinstance(source, BookSource) else BookSource.from_value(source)
        )
        return cls(
            token_id=str(token_id),
            bids=_normalize_levels(bids, descending=True),
            asks=_normalize_levels(asks, descending=False),
            timestamp=float(timestamp),
            source=parsed_source,
            depth_proven=bool(depth_proven),
            synthetic_depth=bool(synthetic_depth),
        )

    @property
    def best_bid(self) -> Optional[BookLevel]:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[BookLevel]:
        return self.asks[0] if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return max(0.0, self.best_ask.price - self.best_bid.price)

    @property
    def is_two_sided(self) -> bool:
        return self.best_bid is not None and self.best_ask is not None

    def age_seconds(self, now: float) -> float:
        return max(0.0, float(now) - self.timestamp)


@dataclass(frozen=True, slots=True)
class MarketDefinition:
    """Canonical identity and venue metadata for a binary market."""
