"""
Confirmed inventory state for the public portfolio edition.

This module owns economic position truth after execution evidence has established
inventory.  It is intentionally separate from order lifecycle state:

    order terminal != inventory terminal

A cancelled, rejected, filled, or otherwise terminal order may still leave
positive token inventory.  A position becomes economically flat only when
confirmed inventory has actually been reduced to zero or settled.

Public invariants:
- requested order size is never treated as inventory evidence
- confirmed BUY execution increases token inventory
- confirmed SELL execution reduces only confirmed inventory
- cumulative execution updates are idempotent per exact order ID
- cumulative notional deltas, not latest_average * quantity_delta, drive accounting
- stale regressing cumulative snapshots cannot rewrite newer accounting
- price/cost-basis uncertainty remains explicit
- wallet-recovered quantity with unknown entry price is preserved as inventory
  without inventing average cost or P&L
- exact order/lifecycle affinity prevents stale execution evidence from being
  applied to a newer owner
- order-terminal observations never mutate confirmed position quantity
- binary YES/NO inventory can be summarized without embedding trading policy

The module contains no admission thresholds, sizing policy, stop logic, target
logic, or asset-specific strategy knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
import math
import threading
import time
from typing import Dict, Iterable, Mapping, Optional, Tuple

from src.execution.types import LifecycleIdentity, OrderSide
from src.market.types import OutcomeSide


class PositionStatus(str, Enum):
    """Economic inventory state, independent from venue order status."""

    OPEN = "OPEN"
    PARTIALLY_PRICED = "PARTIALLY_PRICED"
    SETTLEMENT_PENDING = "SETTLEMENT_PENDING"
    CLOSED = "CLOSED"


class CostBasisState(str, Enum):
    """Completeness of entry-cost evidence for currently owned inventory."""

    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


class ExecutionPriceQuality(IntEnum):
    """Strength of cumulative execution price evidence.

    A stronger equal-quantity correction may replace a weaker one.  A weaker
    observation may never overwrite stronger already-booked price evidence.
    """

    UNKNOWN = 0
    CONSERVATIVE_BOUND = 1
    VENUE_AVERAGE = 2
    WEIGHTED_FILLS = 3
    EXACT_TRADES = 4


@dataclass(frozen=True, slots=True)
class OrderAffinity:
    """Immutable mapping from an exact venue order to one local lifecycle/token."""

    order_id: str
    lifecycle: LifecycleIdentity
    token_id: str
    market_id: str
    outcome_side: OutcomeSide
    order_side: OrderSide

    def __post_init__(self) -> None:
        order_id = str(self.order_id or "").strip()
        token_id = str(self.token_id or "").strip()
        market_id = str(self.market_id or "").strip()

        if not order_id:
            raise ValueError("order_id is required")
        if not token_id:
            raise ValueError("token_id is required")
        if not market_id:
            raise ValueError("market_id is required")

        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "token_id", token_id)
        object.__setattr__(self, "market_id", market_id)


@dataclass(frozen=True, slots=True)
class CumulativeExecution:
    """Latest cumulative execution evidence for one exact order."""

    affinity: OrderAffinity
    cumulative_size: float
    cumulative_average_price: Optional[float]
    cumulative_notional: Optional[float]
    price_quality: ExecutionPriceQuality
    observed_at: float
    price_source: str = ""

    def __post_init__(self) -> None:
        size = float(self.cumulative_size)
        observed = float(self.observed_at)

        if not math.isfinite(size) or size < 0.0:
            raise ValueError("cumulative_size must be non-negative")
        if not math.isfinite(observed) or observed <= 0.0:
            raise ValueError("observed_at must be positive")

        price = self.cumulative_average_price
        notional = self.cumulative_notional

        if price is not None:
            price = float(price)
            if not math.isfinite(price) or not 0.0 < price <= 1.0:
                raise ValueError("cumulative_average_price must be in (0, 1]")

        if notional is not None:
            notional = float(notional)
            if not math.isfinite(notional) or notional < 0.0:
                raise ValueError("cumulative_notional must be non-negative")

        if price is not None and notional is None:
            notional = size * price

        object.__setattr__(self, "cumulative_size", size)
        object.__setattr__(self, "cumulative_average_price", price)
        object.__setattr__(self, "cumulative_notional", notional)
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "price_source", str(self.price_source or ""))


@dataclass(frozen=True, slots=True)
class PositionAnomaly:
    code: str
    token_id: str
    message: str
    observed_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", str(self.code or "UNKNOWN"))
        object.__setattr__(self, "token_id", str(self.token_id or ""))
        object.__setattr__(self, "message", str(self.message or ""))


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """Immutable economic state for one outcome token."""

    token_id: str
    market_id: str
    outcome_side: OutcomeSide

    quantity: float
    priced_quantity: float
    cost_basis: float

    realized_proceeds: float
    known_realized_pnl: float
    realized_pnl_complete: bool

    mark_price: Optional[float]
    status: PositionStatus

    opened_at: float
    updated_at: float
    closed_at: Optional[float] = None

    settlement_pending: bool = False
    settlement_price: Optional[float] = None

    order_ids: Tuple[str, ...] = field(default_factory=tuple)
    lifecycle_ids: Tuple[str, ...] = field(default_factory=tuple)
    terminal_order_ids: Tuple[str, ...] = field(default_factory=tuple)
    anomalies: Tuple[PositionAnomaly, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        token_id = str(self.token_id or "").strip()
        market_id = str(self.market_id or "").strip()

        quantity = max(0.0, float(self.quantity))
        priced_quantity = max(0.0, float(self.priced_quantity))
        cost_basis = max(0.0, float(self.cost_basis))
        proceeds = max(0.0, float(self.realized_proceeds))
        known_pnl = float(self.known_realized_pnl)
        opened = float(self.opened_at)
        updated = float(self.updated_at)

        if not token_id:
            raise ValueError("token_id is required")
        if not market_id:
            raise ValueError("market_id is required")
        if priced_quantity > quantity + 1e-9:
            raise ValueError("priced_quantity cannot exceed quantity")
        if not all(
            math.isfinite(value)
            for value in (
                quantity,
                priced_quantity,
                cost_basis,
                proceeds,
                known_pnl,
                opened,
                updated,
            )
        ):
            raise ValueError("position contains non-finite numeric state")
        if opened <= 0.0 or updated <= 0.0:
            raise ValueError("timestamps must be positive")

        mark = self.mark_price
        if mark is not None:
            mark = float(mark)
            if not math.isfinite(mark) or not 0.0 <= mark <= 1.0:
                raise ValueError("mark_price must be in [0, 1]")

        settlement = self.settlement_price
        if settlement is not None:
            settlement = float(settlement)
            if not math.isfinite(settlement) or not 0.0 <= settlement <= 1.0:
                raise ValueError("settlement_price must be in [0, 1]")

        object.__setattr__(self, "token_id", token_id)
        object.__setattr__(self, "market_id", market_id)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "priced_quantity", priced_quantity)
        object.__setattr__(self, "cost_basis", cost_basis)
        object.__setattr__(self, "realized_proceeds", proceeds)
        object.__setattr__(self, "known_realized_pnl", known_pnl)
        object.__setattr__(self, "mark_price", mark)
        object.__setattr__(self, "opened_at", opened)
        object.__setattr__(self, "updated_at", updated)
        object.__setattr__(self, "settlement_price", settlement)
        object.__setattr__(self, "order_ids", tuple(self.order_ids))
        object.__setattr__(self, "lifecycle_ids", tuple(self.lifecycle_ids))
        object.__setattr__(
            self,
            "terminal_order_ids",
            tuple(self.terminal_order_ids),
        )
        object.__setattr__(self, "anomalies", tuple(self.anomalies))

    @property
    def unpriced_quantity(self) -> float:
        return max(0.0, self.quantity - self.priced_quantity)

    @property
    def cost_basis_state(self) -> CostBasisState:
        if self.quantity <= 1e-9:
            return CostBasisState.COMPLETE
        if self.priced_quantity <= 1e-9:
            return CostBasisState.UNKNOWN
        if self.priced_quantity >= self.quantity - 1e-9:
            return CostBasisState.COMPLETE
        return CostBasisState.PARTIAL

    @property
    def average_cost(self) -> Optional[float]:
        if self.cost_basis_state is not CostBasisState.COMPLETE:
            return None
        if self.quantity <= 1e-9:
            return None
        return self.cost_basis / self.quantity

    @property
    def known_average_cost(self) -> Optional[float]:
        if self.priced_quantity <= 1e-9:
            return None
        return self.cost_basis / self.priced_quantity

    @property
    def realized_pnl(self) -> Optional[float]:
        return (
            self.known_realized_pnl
            if self.realized_pnl_complete
            else None
        )

    @property
    def unrealized_pnl(self) -> Optional[float]:
        if (
            self.quantity <= 1e-9
            or self.mark_price is None
            or self.cost_basis_state is not CostBasisState.COMPLETE
        ):
            return None
        return self.quantity * self.mark_price - self.cost_basis

    @property
    def total_pnl(self) -> Optional[float]:
        unrealized = self.unrealized_pnl
        realized = self.realized_pnl

        if realized is None:
            return None
        if self.quantity <= 1e-9:
            return realized
        if unrealized is None:
            return None
        return realized + unrealized

    @property
    def economically_flat(self) -> bool:
        return self.quantity <= 1e-9


@dataclass(frozen=True, slots=True)
class BinaryInventoryView:
    """Policy-neutral YES/NO inventory summary for one binary market."""

    market_id: str
    yes_quantity: float
    no_quantity: float
    paired_quantity: float
    yes_residual: float
    no_residual: float

    yes_average_cost: Optional[float]
    no_average_cost: Optional[float]

    paired_cost_basis: Optional[float]
    paired_settlement_value: float
    paired_value_at_resolution: Optional[float]

    @property
    def fully_priced_pair(self) -> bool:
        return self.paired_cost_basis is not None


@dataclass(slots=True)
class _PositionRecord:
    token_id: str
    market_id: str
    outcome_side: OutcomeSide

    quantity: float = 0.0
    priced_quantity: float = 0.0
    cost_basis: float = 0.0

    realized_proceeds: float = 0.0
    known_realized_pnl: float = 0.0
    realized_pnl_complete: bool = True

    mark_price: Optional[float] = None

    opened_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    closed_at: Optional[float] = None

    settlement_pending: bool = False
    settlement_price: Optional[float] = None

    order_ids: set[str] = field(default_factory=set)
    lifecycle_ids: set[str] = field(default_factory=set)
    terminal_order_ids: set[str] = field(default_factory=set)
    anomalies: list[PositionAnomaly] = field(default_factory=list)

    def snapshot(self) -> PositionSnapshot:
        if self.quantity <= 1e-9:
            status = PositionStatus.CLOSED
        elif self.settlement_pending:
            status = PositionStatus.SETTLEMENT_PENDING
        elif self.priced_quantity >= self.quantity - 1e-9:
            status = PositionStatus.OPEN
        else:
            status = PositionStatus.PARTIALLY_PRICED

        return PositionSnapshot(
            token_id=self.token_id,
            market_id=self.market_id,
            outcome_side=self.outcome_side,
            quantity=self.quantity,
            priced_quantity=self.priced_quantity,
            cost_basis=self.cost_basis,
            realized_proceeds=self.realized_proceeds,
            known_realized_pnl=self.known_realized_pnl,
            realized_pnl_complete=self.realized_pnl_complete,
            mark_price=self.mark_price,
            status=status,
            opened_at=self.opened_at,
            updated_at=self.updated_at,
            closed_at=self.closed_at,
            settlement_pending=self.settlement_pending,
            settlement_price=self.settlement_price,
            order_ids=tuple(sorted(self.order_ids)),
            lifecycle_ids=tuple(sorted(self.lifecycle_ids)),
            terminal_order_ids=tuple(sorted(self.terminal_order_ids)),
            anomalies=tuple(self.anomalies),
        )


class PositionConflict(RuntimeError):
    """Execution evidence conflicts with already-owned position identity."""


class InsufficientConfirmedInventory(PositionConflict):
    """A SELL attempted to consume more inventory than is confirmed locally."""


class PositionBook:
    """Thread-safe inventory and average-cost accounting service."""

    def __init__(self, *, quantity_epsilon: float = 1e-9) -> None:
        self.quantity_epsilon = max(0.0, float(quantity_epsilon))

        self._gate = threading.RLock()
        self._positions: Dict[str, _PositionRecord] = {}

        self._affinity_by_order_id: Dict[str, OrderAffinity] = {}
        self._execution_by_order_id: Dict[str, CumulativeExecution] = {}
        self._settlement_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Identity / read models
    # ------------------------------------------------------------------

    def _ensure_position(
        self,
        *,
        token_id: str,
        market_id: str,
        outcome_side: OutcomeSide,
        observed_at: float,
    ) -> _PositionRecord:
        token_id = str(token_id or "").strip()
        market_id = str(market_id or "").strip()

        if not token_id:
            raise ValueError("token_id is required")
        if not market_id:
            raise ValueError("market_id is required")

        record = self._positions.get(token_id)

        if record is None:
            record = _PositionRecord(
                token_id=token_id,
                market_id=market_id,
                outcome_side=outcome_side,
                opened_at=observed_at,
                updated_at=observed_at,
            )
            self._positions[token_id] = record
            return record

        if record.market_id != market_id:
            raise PositionConflict(
                f"token {token_id} already belongs to market "
                f"{record.market_id}, not {market_id}"
            )

        if record.outcome_side is not outcome_side:
            raise PositionConflict(
                f"token {token_id} outcome affinity mismatch"
            )

        return record

    def register_order_affinity(
        self,
        affinity: OrderAffinity,
    ) -> OrderAffinity:
        with self._gate:
            existing = self._affinity_by_order_id.get(affinity.order_id)

            if existing is not None and existing != affinity:
                raise PositionConflict(
                    f"order {affinity.order_id} affinity changed"
                )

            self._affinity_by_order_id[affinity.order_id] = affinity
            return affinity

    def snapshot(self, token_id: str) -> Optional[PositionSnapshot]:
        with self._gate:
            record = self._positions.get(str(token_id or ""))
            return record.snapshot() if record is not None else None

    def snapshots(self) -> Tuple[PositionSnapshot, ...]:
        with self._gate:
            return tuple(
                record.snapshot()
                for record in self._positions.values()
            )

    def open_snapshots(self) -> Tuple[PositionSnapshot, ...]:
        return tuple(
            snapshot
            for snapshot in self.snapshots()
            if not snapshot.economically_flat
        )

    def execution(
        self,
        order_id: str,
    ) -> Optional[CumulativeExecution]:
        with self._gate:
            return self._execution_by_order_id.get(
                str(order_id or "")
            )

    # ------------------------------------------------------------------
    # Anomaly helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _append_anomaly(
        record: _PositionRecord,
        *,
        code: str,
        message: str,
        observed_at: float,
    ) -> None:
        anomaly = PositionAnomaly(
            code=code,
            token_id=record.token_id,
            message=message,
            observed_at=observed_at,
        )

        if any(
            existing.code == anomaly.code
            and existing.message == anomaly.message
            for existing in record.anomalies
        ):
            return

        record.anomalies.append(anomaly)

    # ------------------------------------------------------------------
    # Exact cumulative execution accounting
    # ------------------------------------------------------------------

    def apply_cumulative_execution(
        self,
        *,
        affinity: OrderAffinity,
        cumulative_size: float,
        cumulative_average_price: Optional[float],
        price_quality: ExecutionPriceQuality = ExecutionPriceQuality.VENUE_AVERAGE,
        observed_at: Optional[float] = None,
        price_source: str = "",
    ) -> PositionSnapshot:
        """Apply cumulative exact-order execution evidence idempotently.

        The cumulative order notional is compared with the previously booked
        cumulative notional.  This correctly handles both additional fills and
        same-quantity average-price corrections.
        """

        now = float(observed_at or time.time())
        size = max(0.0, float(cumulative_size))

        if not math.isfinite(size):
            raise ValueError("cumulative_size must be finite")
        if not math.isfinite(now) or now <= 0.0:
            raise ValueError("observed_at must be positive")

        price = cumulative_average_price
        if price is not None:
            price = float(price)
            if not math.isfinite(price) or not 0.0 < price <= 1.0:
                raise ValueError(
                    "cumulative_average_price must be in (0, 1]"
                )

        self.register_order_affinity(affinity)

        with self._gate:
            record = self._ensure_position(
                token_id=affinity.token_id,
                market_id=affinity.market_id,
                outcome_side=affinity.outcome_side,
                observed_at=now,
            )

            previous = self._execution_by_order_id.get(
                affinity.order_id
            )

            if previous is not None:
                if previous.affinity != affinity:
                    raise PositionConflict(
                        f"order {affinity.order_id} execution affinity changed"
                    )

                if size + self.quantity_epsilon < previous.cumulative_size:
                    self._append_anomaly(
                        record,
                        code="STALE_CUMULATIVE_QUANTITY_REGRESSION",
                        message=(
                            f"order {affinity.order_id} quantity regressed "
                            f"{previous.cumulative_size:.8f}->{size:.8f}; "
                            "newer accounting preserved"
                        ),
                        observed_at=now,
                    )
                    return record.snapshot()

                # Same quantity with weaker price evidence cannot overwrite a
                # stronger already-booked realized average.
                if (
                    abs(size - previous.cumulative_size)
                    <= self.quantity_epsilon
                    and int(price_quality) < int(previous.price_quality)
                ):
                    self._append_anomaly(
                        record,
                        code="LOWER_QUALITY_PRICE_CORRECTION_IGNORED",
                        message=(
                            f"order {affinity.order_id} weaker "
                            f"price evidence ignored"
                        ),
                        observed_at=now,
                    )
                    return record.snapshot()

            previous_size = (
                previous.cumulative_size
                if previous is not None
                else 0.0
            )
            previous_notional = (
                previous.cumulative_notional
                if previous is not None
                else 0.0
            )

            # Missing price evidence can still establish BUY inventory quantity.
            new_notional = (
                size * price if price is not None else None
            )

            if (
                previous is not None
                and new_notional is None
                and previous.cumulative_notional is not None
                and abs(size - previous_size)
                <= self.quantity_epsilon
            ):
                # Do not downgrade already-priced equal-quantity execution.
                return record.snapshot()

            delta_size = max(0.0, size - previous_size)

            previous_known_notional = (
                float(previous_notional)
                if previous_notional is not None
                else None
            )
            delta_notional = (
                new_notional - previous_known_notional
                if new_notional is not None
                and previous_known_notional is not None
                else (
                    new_notional
                    if new_notional is not None
                    and previous is None
                    else None
                )
            )

            if affinity.order_side is OrderSide.BUY:
                self._apply_buy_cumulative_delta(
                    record,
                    delta_size=delta_size,
                    delta_notional=delta_notional,
                    new_size=size,
                    new_notional=new_notional,
                    previous=previous,
                    observed_at=now,
                )
            else:
                self._apply_sell_cumulative_delta(
                    record,
                    delta_size=delta_size,
                    delta_notional=delta_notional,
                    new_size=size,
                    new_notional=new_notional,
                    previous=previous,
                    observed_at=now,
                    order_id=affinity.order_id,
                )

            record.order_ids.add(affinity.order_id)
            record.lifecycle_ids.add(
                affinity.lifecycle.lifecycle_id
            )
            record.updated_at = max(record.updated_at, now)

            if record.quantity > self.quantity_epsilon:
                record.closed_at = None

            execution = CumulativeExecution(
                affinity=affinity,
                cumulative_size=size,
                cumulative_average_price=price,
                cumulative_notional=new_notional,
                price_quality=price_quality,
                observed_at=now,
                price_source=price_source,
            )
            self._execution_by_order_id[affinity.order_id] = execution

            return record.snapshot()

    def _apply_buy_cumulative_delta(
        self,
        record: _PositionRecord,
        *,
        delta_size: float,
        delta_notional: Optional[float],
        new_size: float,
        new_notional: Optional[float],
        previous: Optional[CumulativeExecution],
        observed_at: float,
    ) -> None:
        if delta_size > self.quantity_epsilon:
            record.quantity += delta_size

            if delta_notional is not None:
                if delta_notional < -1e-9:
                    self._append_anomaly(
                        record,
                        code="NEGATIVE_INCREMENTAL_BUY_NOTIONAL",
                        message=(
                            "cumulative average correction implies negative "
                            "incremental BUY notional"
                        ),
                        observed_at=observed_at,
                    )
                else:
                    record.priced_quantity += delta_size
                    record.cost_basis += max(0.0, delta_notional)

        # Equal-quantity stronger price correction adjusts known cost basis by the
        # cumulative notional correction rather than inventing another fill.
        if (
            previous is not None
            and abs(new_size - previous.cumulative_size)
            <= self.quantity_epsilon
            and new_notional is not None
            and previous.cumulative_notional is not None
        ):
            correction = new_notional - previous.cumulative_notional

            if (
                previous.cumulative_size
                <= record.priced_quantity + self.quantity_epsilon
            ):
                record.cost_basis = max(
                    0.0,
                    record.cost_basis + correction,
                )

    def _apply_sell_cumulative_delta(
        self,
        record: _PositionRecord,
        *,
        delta_size: float,
        delta_notional: Optional[float],
        new_size: float,
        new_notional: Optional[float],
        previous: Optional[CumulativeExecution],
        observed_at: float,
        order_id: str,
    ) -> None:
        if delta_size > record.quantity + self.quantity_epsilon:
            self._append_anomaly(
                record,
                code="SELL_EXCEEDS_CONFIRMED_INVENTORY",
                message=(
                    f"order {order_id} sell delta {delta_size:.8f} "
                    f"exceeds confirmed inventory {record.quantity:.8f}"
                ),
                observed_at=observed_at,
            )
            raise InsufficientConfirmedInventory(
                f"sell {delta_size} exceeds confirmed "
                f"inventory {record.quantity}"
            )

        prior_basis_complete = (
            record.quantity <= self.quantity_epsilon
            or record.priced_quantity
            >= record.quantity - self.quantity_epsilon
        )

        if delta_size > self.quantity_epsilon:
            if delta_notional is None:
                # Quantity can be proven while proceeds price is still unknown.
                record.realized_pnl_complete = False
            else:
                record.realized_proceeds += delta_notional

            if prior_basis_complete and record.quantity > self.quantity_epsilon:
                average_cost = (
                    record.cost_basis / record.quantity
                    if record.quantity > self.quantity_epsilon
                    else 0.0
                )
                removed_cost = average_cost * delta_size

                record.cost_basis = max(
                    0.0,
                    record.cost_basis - removed_cost,
                )
                record.priced_quantity = max(
                    0.0,
                    record.priced_quantity - delta_size,
                )

                if delta_notional is not None:
                    record.known_realized_pnl += (
                        delta_notional - removed_cost
                    )
                else:
                    record.realized_pnl_complete = False
            else:
                # We cannot know which part of mixed priced/unpriced inventory was
                # sold without inventing a lot-allocation rule. Preserve quantity
                # truth, invalidate remaining exact cost-basis precision, and mark
                # realized P&L incomplete.
                record.cost_basis = 0.0
                record.priced_quantity = 0.0
                record.realized_pnl_complete = False

                self._append_anomaly(
                    record,
                    code="COST_BASIS_UNRESOLVED_AFTER_UNPRICED_SELL",
                    message=(
                        "SELL reduced inventory whose cost basis was incomplete; "
                        "exact realized P&L intentionally left unknown"
                    ),
                    observed_at=observed_at,
                )

            record.quantity = max(0.0, record.quantity - delta_size)

            if record.quantity <= self.quantity_epsilon:
                record.quantity = 0.0
                record.priced_quantity = 0.0
                record.cost_basis = 0.0
                record.closed_at = observed_at
                record.settlement_pending = False

        # Same-quantity stronger cumulative-average correction changes proceeds
        # without changing sold quantity or cost basis.
        if (
            previous is not None
            and abs(new_size - previous.cumulative_size)
            <= self.quantity_epsilon
            and new_notional is not None
            and previous.cumulative_notional is not None
        ):
            correction = new_notional - previous.cumulative_notional
            record.realized_proceeds += correction

            if record.realized_pnl_complete:
                record.known_realized_pnl += correction

    # ------------------------------------------------------------------
    # Inventory recovery without price evidence
    # ------------------------------------------------------------------

    def observe_inventory_floor(
        self,
        *,
        token_id: str,
        market_id: str,
        outcome_side: OutcomeSide,
        confirmed_quantity: float,
        lifecycle: Optional[LifecycleIdentity] = None,
        source_order_id: Optional[str] = None,
        observed_at: Optional[float] = None,
        reason: str = "inventory evidence",
    ) -> PositionSnapshot:
        """Raise inventory to a newly confirmed absolute floor without inventing cost.

        This is useful for wallet/reconciliation recovery where quantity is proven
        but an exact entry average is not yet available.
        """

        now = float(observed_at or time.time())
        quantity = max(0.0, float(confirmed_quantity))

        if not math.isfinite(quantity):
            raise ValueError("confirmed_quantity must be finite")

        with self._gate:
            record = self._ensure_position(
                token_id=token_id,
                market_id=market_id,
                outcome_side=outcome_side,
                observed_at=now,
            )

            if quantity > record.quantity + self.quantity_epsilon:
                delta = quantity - record.quantity
                record.quantity = quantity

                self._append_anomaly(
                    record,
                    code="UNPRICED_CONFIRMED_INVENTORY",
                    message=(
                        f"{delta:.8f} additional confirmed inventory has no "
                        f"exact cost basis yet ({reason})"
                    ),
                    observed_at=now,
                )

            if lifecycle is not None:
                record.lifecycle_ids.add(lifecycle.lifecycle_id)

            if source_order_id:
                record.order_ids.add(str(source_order_id))

            record.updated_at = max(record.updated_at, now)
            if record.quantity > self.quantity_epsilon:
                record.closed_at = None

            return record.snapshot()

    # ------------------------------------------------------------------
    # Order terminal metadata
    # ------------------------------------------------------------------

    def note_order_terminal(
        self,
        *,
        order_id: str,
        lifecycle: LifecycleIdentity,
        observed_at: Optional[float] = None,
    ) -> Optional[PositionSnapshot]:
        """Record terminal execution metadata without changing inventory."""

        order_id = str(order_id or "").strip()
        if not order_id:
            raise ValueError("order_id is required")

        now = float(observed_at or time.time())

        with self._gate:
            affinity = self._affinity_by_order_id.get(order_id)
            if affinity is None:
                return None

            if affinity.lifecycle != lifecycle:
                raise PositionConflict(
                    f"terminal observation lifecycle mismatch for {order_id}"
                )

            record = self._positions.get(affinity.token_id)
            if record is None:
                return None

            record.terminal_order_ids.add(order_id)
            record.updated_at = max(record.updated_at, now)

            # Deliberately no quantity/cost/P&L mutation here.
            return record.snapshot()

    # ------------------------------------------------------------------
    # Mark-to-market / settlement
    # ------------------------------------------------------------------

    def mark(
        self,
        token_id: str,
        price: Optional[float],
        *,
        observed_at: Optional[float] = None,
    ) -> Optional[PositionSnapshot]:
        now = float(observed_at or time.time())

        if price is not None:
            price = float(price)
            if not math.isfinite(price) or not 0.0 <= price <= 1.0:
                raise ValueError("mark price must be in [0, 1]")

        with self._gate:
            record = self._positions.get(str(token_id or ""))
            if record is None:
                return None

            record.mark_price = price
            record.updated_at = max(record.updated_at, now)
            return record.snapshot()

    def mark_settlement_pending(
        self,
        token_id: str,
        *,
        observed_at: Optional[float] = None,
    ) -> Optional[PositionSnapshot]:
        """Move positive inventory to settlement-pending without fabricating P&L."""

        now = float(observed_at or time.time())

        with self._gate:
            record = self._positions.get(str(token_id or ""))
            if record is None:
                return None

            if record.quantity > self.quantity_epsilon:
                record.settlement_pending = True

            record.updated_at = max(record.updated_at, now)
            return record.snapshot()

    def settle(
        self,
        *,
        token_id: str,
        settlement_id: str,
        payout_price: float,
        quantity: Optional[float] = None,
        observed_at: Optional[float] = None,
    ) -> PositionSnapshot:
        """Apply venue-confirmed settlement/redeem proceeds exactly once."""

        token_id = str(token_id or "").strip()
        settlement_id = str(settlement_id or "").strip()
        payout = float(payout_price)
        now = float(observed_at or time.time())

        if not token_id:
            raise ValueError("token_id is required")
        if not settlement_id:
            raise ValueError("settlement_id is required")
        if not math.isfinite(payout) or not 0.0 <= payout <= 1.0:
            raise ValueError("payout_price must be in [0, 1]")

        with self._gate:
            record = self._positions.get(token_id)
            if record is None:
                raise KeyError(f"unknown position: {token_id}")

            if settlement_id in self._settlement_ids:
                return record.snapshot()

            settle_quantity = (
                record.quantity
                if quantity is None
                else max(0.0, float(quantity))
            )

            if settle_quantity > record.quantity + self.quantity_epsilon:
                raise InsufficientConfirmedInventory(
                    "settlement quantity exceeds confirmed inventory"
                )

            proceeds = settle_quantity * payout
            basis_complete = (
                record.priced_quantity
                >= record.quantity - self.quantity_epsilon
            )

            record.realized_proceeds += proceeds

            if basis_complete and record.quantity > self.quantity_epsilon:
                average_cost = record.cost_basis / record.quantity
                removed_cost = average_cost * settle_quantity

                record.cost_basis = max(
                    0.0,
                    record.cost_basis - removed_cost,
                )
                record.priced_quantity = max(
                    0.0,
                    record.priced_quantity - settle_quantity,
                )
                record.known_realized_pnl += proceeds - removed_cost
            else:
                record.cost_basis = 0.0
                record.priced_quantity = 0.0
                record.realized_pnl_complete = False

                self._append_anomaly(
                    record,
                    code="SETTLEMENT_WITH_INCOMPLETE_COST_BASIS",
                    message=(
                        "settlement proceeds are known but exact realized "
                        "P&L remains unknown because cost basis was incomplete"
                    ),
                    observed_at=now,
                )

            record.quantity = max(
                0.0,
                record.quantity - settle_quantity,
            )
            record.settlement_price = payout
            record.updated_at = max(record.updated_at, now)
            self._settlement_ids.add(settlement_id)

            if record.quantity <= self.quantity_epsilon:
                record.quantity = 0.0
                record.priced_quantity = 0.0
                record.cost_basis = 0.0
                record.closed_at = now
                record.settlement_pending = False
            else:
                record.settlement_pending = True

            return record.snapshot()

    # ------------------------------------------------------------------
    # Binary market read model
    # ------------------------------------------------------------------

    def binary_inventory(
        self,
        *,
        market_id: str,
        yes_token: str,
        no_token: str,
    ) -> BinaryInventoryView:
        """Summarize paired and residual binary inventory without policy decisions."""

        yes = self.snapshot(yes_token)
        no = self.snapshot(no_token)

        yes_quantity = (
            yes.quantity
            if yes is not None and yes.market_id == market_id
            else 0.0
        )
        no_quantity = (
            no.quantity
            if no is not None and no.market_id == market_id
            else 0.0
        )

        paired = min(yes_quantity, no_quantity)
        yes_residual = max(0.0, yes_quantity - paired)
        no_residual = max(0.0, no_quantity - paired)

        yes_avg = (
            yes.average_cost if yes is not None else None
        )
        no_avg = (
            no.average_cost if no is not None else None
        )

        paired_cost = None
        paired_value = None

        if paired > self.quantity_epsilon and yes_avg is not None and no_avg is not None:
            paired_cost = paired * (yes_avg + no_avg)
            # One YES + one NO token in the same resolved binary market has a
            # combined terminal payout of one unit.
            paired_value = paired - paired_cost

        return BinaryInventoryView(
            market_id=str(market_id or ""),
            yes_quantity=yes_quantity,
            no_quantity=no_quantity,
            paired_quantity=paired,
            yes_residual=yes_residual,
            no_residual=no_residual,
            yes_average_cost=yes_avg,
            no_average_cost=no_avg,
            paired_cost_basis=paired_cost,
            paired_settlement_value=paired,
            paired_value_at_resolution=paired_value,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def prune_closed(
        self,
        *,
        older_than_seconds: float = 900.0,
        now: Optional[float] = None,
    ) -> int:
        """Prune only economically flat positions.

        Terminal order metadata never makes a positive-inventory row eligible.
        """

        current_time = float(now or time.time())
        minimum_age = max(0.0, float(older_than_seconds))
        removed = 0

        with self._gate:
            for token_id, record in list(self._positions.items()):
                if record.quantity > self.quantity_epsilon:
                    continue
                if record.closed_at is None:
                    continue
                if current_time - record.closed_at < minimum_age:
                    continue

                order_ids = set(record.order_ids)
                self._positions.pop(token_id, None)

                for order_id in order_ids:
                    self._execution_by_order_id.pop(order_id, None)
                    self._affinity_by_order_id.pop(order_id, None)

                removed += 1

        return removed
