"""
Typed execution contracts for the public portfolio edition.

The private production system contains venue- and strategy-specific lifecycle
details. This module exposes only the reusable execution concepts needed by the
portfolio architecture:

- immutable order intent
- exact lifecycle/attempt identity
- explicit submission ambiguity
- normalized venue order state
- partial/full fill accounting
- cancellation outcomes
- reconciliation results

A transport timeout is never silently treated as a failed order. Ambiguous
submission/cancellation states retain lifecycle ownership until reconciliation
produces terminal evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Mapping, Optional, Tuple

from src.market.types import OutcomeSide


class OrderSide(str, Enum):
    """Venue order direction."""

    BUY = "BUY"
    SELL = "SELL"

    @classmethod
    def from_value(cls, value: str) -> "OrderSide":
        normalized = str(value or "").strip().upper()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"unsupported order side: {value!r}") from exc


class TimeInForce(str, Enum):
    """Public subset of order persistence semantics."""

    GTC = "GTC"


class OrderIntentRole(str, Enum):
    """Execution purpose without embedding proprietary admission policy."""

    OPENING = "OPENING"
    COMPLETION = "COMPLETION"
    RISK_REDUCTION = "RISK_REDUCTION"
    REBALANCE = "REBALANCE"
    UNKNOWN = "UNKNOWN"


class OrderLifecycleState(str, Enum):
    """Normalized local lifecycle state.

    `SUBMISSION_UNKNOWN` and `CANCEL_UNKNOWN` are intentionally first-class.
    They mean the venue outcome is ambiguous and reconciliation still owns the
    lifecycle; they are not aliases for FAILED/CANCELLED.
    """

    CREATED = "CREATED"
    SIGNING = "SIGNING"
    SUBMITTING = "SUBMITTING"

    WORKING = "WORKING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"

    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCEL_UNKNOWN = "CANCEL_UNKNOWN"
    CANCELLED = "CANCELLED"

    SUBMISSION_UNKNOWN = "SUBMISSION_UNKNOWN"
    RECONCILING = "RECONCILING"

    REJECTED = "REJECTED"
    FAILED = "FAILED"
    CLOSED = "CLOSED"

    @property
    def is_terminal(self) -> bool:
        return self in {
            self.FILLED,
            self.CANCELLED,
            self.REJECTED,
            self.FAILED,
            self.CLOSED,
        }

    @property
    def is_ambiguous(self) -> bool:
        return self in {
            self.SUBMISSION_UNKNOWN,
            self.CANCEL_UNKNOWN,
            self.RECONCILING,
        }

    @property
    def may_own_live_order(self) -> bool:
        return self in {
            self.SUBMITTING,
            self.WORKING,
            self.PARTIALLY_FILLED,
            self.CANCEL_REQUESTED,
            self.CANCEL_UNKNOWN,
            self.SUBMISSION_UNKNOWN,
            self.RECONCILING,
        }


class SubmissionOutcome(str, Enum):
    """Result classification at the raw order-submission boundary."""

    CONFIRMED_WORKING = "CONFIRMED_WORKING"
    CONFIRMED_FILLED = "CONFIRMED_FILLED"
    REJECTED = "REJECTED"
    FAILED_BEFORE_SUBMIT = "FAILED_BEFORE_SUBMIT"
    UNKNOWN = "UNKNOWN"


class CancellationOutcome(str, Enum):
    """Result classification for a cancellation request."""

    CONFIRMED_CANCELLED = "CONFIRMED_CANCELLED"
    ALREADY_TERMINAL = "ALREADY_TERMINAL"
    NOT_FOUND = "NOT_FOUND"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


class ReconciliationOutcome(str, Enum):
    """What an exact-order reconciliation pass established."""

    WORKING = "WORKING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED_ZERO_FILL = "CANCELLED_ZERO_FILL"
    REJECTED_ZERO_FILL = "REJECTED_ZERO_FILL"
    STILL_AMBIGUOUS = "STILL_AMBIGUOUS"


@dataclass(frozen=True, slots=True)
class LifecycleIdentity:
    """Stable local identity spanning retries, watchers and reconciliation.

    `attempt_id` identifies one logical submit attempt.
    `generation` changes when an order lifecycle is superseded or replaced.
    A stale status result from an older generation must not become authoritative
    for a newer generation.
    """

    lifecycle_id: str
    attempt_id: str
    generation: int = 0

    def __post_init__(self) -> None:
        lifecycle_id = str(self.lifecycle_id or "").strip()
        attempt_id = str(self.attempt_id or "").strip()
        generation = int(self.generation)

        if not lifecycle_id:
            raise ValueError("lifecycle_id is required")
        if not attempt_id:
            raise ValueError("attempt_id is required")
        if generation < 0:
            raise ValueError("generation must be non-negative")

        object.__setattr__(self, "lifecycle_id", lifecycle_id)
        object.__setattr__(self, "attempt_id", attempt_id)
        object.__setattr__(self, "generation", generation)

    def next_generation(self) -> "LifecycleIdentity":
        return LifecycleIdentity(
            lifecycle_id=self.lifecycle_id,
            attempt_id=self.attempt_id,
            generation=self.generation + 1,
        )


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """Immutable instruction passed from policy to execution.

    The execution layer receives a fully formed intent. It does not decide
    whether the market setup is desirable.
    """

    token_id: str
    market_id: str
    outcome_side: OutcomeSide
    order_side: OrderSide
    price: float
    size: float
    role: OrderIntentRole
    lifecycle: LifecycleIdentity
    time_in_force: TimeInForce = TimeInForce.GTC
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        token_id = str(self.token_id or "").strip()
        market_id = str(self.market_id or "").strip()
        price = float(self.price)
        size = float(self.size)
        created_at = float(self.created_at)

        if not token_id:
            raise ValueError("token_id is required")
        if not market_id:
            raise ValueError("market_id is required")
        if not math.isfinite(price) or not 0.0 < price <= 1.0:
            raise ValueError(f"invalid order price: {price!r}")
        if not math.isfinite(size) or size <= 0.0:
            raise ValueError(f"invalid order size: {size!r}")
        if not math.isfinite(created_at) or created_at <= 0.0:
            raise ValueError("created_at must be a positive Unix timestamp")

        object.__setattr__(self, "token_id", token_id)
        object.__setattr__(self, "market_id", market_id)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "created_at", created_at)


@dataclass(frozen=True, slots=True)
class VenueOrderSnapshot:
    """Normalized exact-order state returned by the venue/status layer."""

    order_id: str
    token_id: str
    order_side: OrderSide
    state: OrderLifecycleState
    requested_size: float
    matched_size: float
    limit_price: Optional[float] = None
    average_fill_price: Optional[float] = None
    observed_at: float = field(default_factory=time.time)
    raw_status: str = ""

    def __post_init__(self) -> None:
        order_id = str(self.order_id or "").strip()
        token_id = str(self.token_id or "").strip()
        requested = float(self.requested_size)
        matched = float(self.matched_size)
        observed = float(self.observed_at)

        if not order_id:
            raise ValueError("order_id is required")
        if not token_id:
            raise ValueError("token_id is required")
        if not math.isfinite(requested) or requested <= 0.0:
            raise ValueError("requested_size must be positive")
        if not math.isfinite(matched) or matched < 0.0:
            raise ValueError("matched_size must be non-negative")
        if not math.isfinite(observed) or observed <= 0.0:
            raise ValueError("observed_at must be a positive Unix timestamp")

        limit_price = self.limit_price
        if limit_price is not None:
            limit_price = float(limit_price)
            if not math.isfinite(limit_price) or not 0.0 < limit_price <= 1.0:
                raise ValueError("limit_price must be in (0, 1]")

        avg = self.average_fill_price
        if avg is not None:
            avg = float(avg)
            if not math.isfinite(avg) or not 0.0 < avg <= 1.0:
                raise ValueError("average_fill_price must be in (0, 1]")

        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "token_id", token_id)
        object.__setattr__(self, "requested_size", requested)
        object.__setattr__(self, "matched_size", matched)
        object.__setattr__(self, "limit_price", limit_price)
        object.__setattr__(self, "average_fill_price", avg)
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "raw_status", str(self.raw_status or ""))

    @property
    def remaining_size(self) -> float:
        return max(0.0, self.requested_size - self.matched_size)

    @property
    def has_fill(self) -> bool:
        return self.matched_size > 0.0

    @property
    def fully_filled(self) -> bool:
        return self.matched_size >= self.requested_size - 1e-9


@dataclass(frozen=True, slots=True)
class Fill:
    """One normalized fill contribution belonging to an exact order."""

    fill_id: str
    order_id: str
    token_id: str
    order_side: OrderSide
    size: float
    price: float
    timestamp: float

    def __post_init__(self) -> None:
        fill_id = str(self.fill_id or "").strip()
        order_id = str(self.order_id or "").strip()
        token_id = str(self.token_id or "").strip()
        size = float(self.size)
        price = float(self.price)
        timestamp = float(self.timestamp)

        if not fill_id:
            raise ValueError("fill_id is required")
        if not order_id:
            raise ValueError("order_id is required")
        if not token_id:
            raise ValueError("token_id is required")
        if not math.isfinite(size) or size <= 0.0:
            raise ValueError("fill size must be positive")
        if not math.isfinite(price) or not 0.0 < price <= 1.0:
            raise ValueError("fill price must be in (0, 1]")
        if not math.isfinite(timestamp) or timestamp <= 0.0:
            raise ValueError("fill timestamp must be positive")

        object.__setattr__(self, "fill_id", fill_id)
        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "token_id", token_id)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "timestamp", timestamp)


@dataclass(frozen=True, slots=True)
class FillSummary:
    """Aggregated fill accounting for one exact order."""

    order_id: str
    requested_size: float
    filled_size: float
    average_price: Optional[float]
    fills: Tuple[Fill, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        order_id = str(self.order_id or "").strip()
        requested = float(self.requested_size)
        filled = float(self.filled_size)

        if not order_id:
            raise ValueError("order_id is required")
        if not math.isfinite(requested) or requested <= 0.0:
            raise ValueError("requested_size must be positive")
        if not math.isfinite(filled) or filled < 0.0:
            raise ValueError("filled_size must be non-negative")

        average_price = self.average_price
        if average_price is not None:
            average_price = float(average_price)
            if (
                not math.isfinite(average_price)
                or not 0.0 < average_price <= 1.0
            ):
                raise ValueError("average_price must be in (0, 1]")

        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "requested_size", requested)
        object.__setattr__(self, "filled_size", filled)
        object.__setattr__(self, "average_price", average_price)
        object.__setattr__(self, "fills", tuple(self.fills))

    @property
    def remaining_size(self) -> float:
        return max(0.0, self.requested_size - self.filled_size)

    @property
    def has_fill(self) -> bool:
        return self.filled_size > 0.0

    @property
    def fully_filled(self) -> bool:
        return self.filled_size >= self.requested_size - 1e-9


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    """Typed result of one submission boundary crossing."""

    outcome: SubmissionOutcome
    intent: OrderIntent
    order_id: Optional[str] = None
    venue_snapshot: Optional[VenueOrderSnapshot] = None
    post_call_entered: bool = False
    reason: str = ""
    observed_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        order_id = str(self.order_id or "").strip() or None
        observed = float(self.observed_at)

        if not math.isfinite(observed) or observed <= 0.0:
            raise ValueError("observed_at must be a positive Unix timestamp")

        if self.outcome is SubmissionOutcome.CONFIRMED_WORKING and not order_id:
            raise ValueError("confirmed working submission requires order_id")

        if self.outcome is SubmissionOutcome.CONFIRMED_FILLED and not order_id:
            raise ValueError("confirmed filled submission requires order_id")

        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "reason", str(self.reason or ""))
        object.__setattr__(self, "observed_at", observed)

    @property
    def confirmed(self) -> bool:
        return self.outcome in {
            SubmissionOutcome.CONFIRMED_WORKING,
            SubmissionOutcome.CONFIRMED_FILLED,
        }

    @property
    def ambiguous(self) -> bool:
        return self.outcome is SubmissionOutcome.UNKNOWN

    @property
    def requires_reconciliation(self) -> bool:
        # Reconciliation is owned only by the explicit UNKNOWN outcome. A
        # FAILED_BEFORE_SUBMIT result is, by contract, a confirmed local no-post
        # result and must not be reinterpreted from a contradictory flag.
        return self.ambiguous


@dataclass(frozen=True, slots=True)
class CancellationResult:
    """Typed cancellation outcome retaining ambiguity explicitly."""

    outcome: CancellationOutcome
    order_id: str
    lifecycle: LifecycleIdentity
    venue_snapshot: Optional[VenueOrderSnapshot] = None
    reason: str = ""
    observed_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        order_id = str(self.order_id or "").strip()
        observed = float(self.observed_at)

        if not order_id:
            raise ValueError("order_id is required")
        if not math.isfinite(observed) or observed <= 0.0:
            raise ValueError("observed_at must be positive")

        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "reason", str(self.reason or ""))
        object.__setattr__(self, "observed_at", observed)

    @property
    def confirmed_terminal(self) -> bool:
        return self.outcome in {
            CancellationOutcome.CONFIRMED_CANCELLED,
            CancellationOutcome.ALREADY_TERMINAL,
        }

    @property
    def ambiguous(self) -> bool:
        return self.outcome is CancellationOutcome.UNKNOWN


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Evidence produced by an exact-order reconciliation pass."""

    outcome: ReconciliationOutcome
    lifecycle: LifecycleIdentity
    order_id: Optional[str]
    snapshot: Optional[VenueOrderSnapshot] = None
    fill_summary: Optional[FillSummary] = None
    reason: str = ""
    evidence: Mapping[str, object] = field(default_factory=dict)
    observed_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        order_id = str(self.order_id or "").strip() or None
        observed = float(self.observed_at)

        if not math.isfinite(observed) or observed <= 0.0:
            raise ValueError("observed_at must be positive")

        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "reason", str(self.reason or ""))
        object.__setattr__(self, "evidence", dict(self.evidence))
        object.__setattr__(self, "observed_at", observed)

    @property
    def terminal(self) -> bool:
        return self.outcome in {
            ReconciliationOutcome.FILLED,
            ReconciliationOutcome.CANCELLED_ZERO_FILL,
            ReconciliationOutcome.REJECTED_ZERO_FILL,
        }

    @property
    def ambiguous(self) -> bool:
        return self.outcome is ReconciliationOutcome.STILL_AMBIGUOUS


@dataclass(frozen=True, slots=True)
class WorkingOrder:
    """Local exact-owner record used by lifecycle and reconciliation services."""

    intent: OrderIntent
    state: OrderLifecycleState
    order_id: Optional[str] = None
    filled_size: float = 0.0
    average_fill_price: Optional[float] = None
    last_observed_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        order_id = str(self.order_id or "").strip() or None
        filled = float(self.filled_size)
        observed = float(self.last_observed_at)

        if not math.isfinite(filled) or filled < 0.0:
            raise ValueError("filled_size must be non-negative")
        if not math.isfinite(observed) or observed <= 0.0:
            raise ValueError("last_observed_at must be positive")

        average = self.average_fill_price
        if average is not None:
            average = float(average)
            if not math.isfinite(average) or not 0.0 < average <= 1.0:
                raise ValueError("average_fill_price must be in (0, 1]")

        if (
            self.state in {
                OrderLifecycleState.WORKING,
                OrderLifecycleState.PARTIALLY_FILLED,
                OrderLifecycleState.FILLED,
                OrderLifecycleState.CANCEL_REQUESTED,
                OrderLifecycleState.CANCEL_UNKNOWN,
            }
            and order_id is None
        ):
            raise ValueError(f"{self.state.value} requires an exact order_id")

        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "filled_size", filled)
        object.__setattr__(self, "average_fill_price", average)
        object.__setattr__(self, "last_observed_at", observed)

    @property
    def remaining_size(self) -> float:
        return max(0.0, self.intent.size - self.filled_size)

    @property
    def owns_lifecycle(self) -> bool:
        """Whether this row must block duplicate/superseding execution."""
        return self.state.may_own_live_order or self.filled_size > 0.0
