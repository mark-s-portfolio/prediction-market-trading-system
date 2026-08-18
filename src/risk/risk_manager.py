"""
Generic portfolio risk management for the public portfolio edition.

This module consumes confirmed inventory and exact execution ownership. It does
not score trading candidates and it does not contain strategy-specific thresholds.

Risk truth is intentionally split into two independent domains:

1. Economic inventory
   Confirmed positions from PositionBook remain exposure until actually sold or
   settled. A terminal order does not remove inventory.

2. Execution ownership
   A live/unknown/cancel-uncertain lifecycle may still create or modify exposure
   even before inventory is visible. It therefore consumes execution-risk capacity.

Public responsibilities:
- aggregate gross/marked/unknown-cost exposure
- summarize binary-market paired and residual inventory
- count exact unresolved execution owners
- track account equity and drawdown
- enforce configurable portfolio capacity
- prevent duplicate new exposure on a token with an existing execution owner
- switch new exposure into reduce-only mode after configurable risk breaches
- preserve risk-reducing and settlement actions during reduce-only operation
- expose structured, auditable risk checks and snapshots

No production bankroll, stop-loss, completion, slot, asset-family, admission,
quality, or historical trading thresholds are embedded here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
import math
import threading
import time
from typing import Dict, Iterable, Optional, Protocol, Sequence, Tuple

from src.execution.order_lifecycle import LifecycleSnapshot
from src.execution.types import OrderLifecycleState
from src.market.types import OutcomeSide
from src.risk.position_state import CostBasisState, PositionBook, PositionSnapshot


class RiskAction(str, Enum):
    """High-level economic intent presented to the risk layer."""

    NEW_EXPOSURE = "NEW_EXPOSURE"
    INCREASE_EXPOSURE = "INCREASE_EXPOSURE"
    REDUCE_EXPOSURE = "REDUCE_EXPOSURE"
    CLOSE_EXPOSURE = "CLOSE_EXPOSURE"
    SETTLE = "SETTLE"
    RECONCILE = "RECONCILE"

    @property
    def creates_exposure(self) -> bool:
        return self in {
            self.NEW_EXPOSURE,
            self.INCREASE_EXPOSURE,
        }

    @property
    def risk_reducing(self) -> bool:
        return self in {
            self.REDUCE_EXPOSURE,
            self.CLOSE_EXPOSURE,
            self.SETTLE,
            self.RECONCILE,
        }


class RiskMode(str, Enum):
    """Portfolio operating mode derived from current risk state."""

    NORMAL = "NORMAL"
    REDUCE_ONLY = "REDUCE_ONLY"
    HALTED = "HALTED"


class RiskCheckSeverity(IntEnum):
    INFO = 0
    WARNING = 1
    HARD = 2


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Configurable generic portfolio limits.

    All numeric limits default to ``None`` so this public module does not reveal
    production-tuned values. Consumers explicitly supply the limits appropriate
    to their own deployment.
    """

    max_gross_cost_exposure: Optional[float] = None
    max_marked_exposure: Optional[float] = None

    max_open_tokens: Optional[int] = None
    max_open_markets: Optional[int] = None

    max_unresolved_executions: Optional[int] = None
    max_unresolved_per_token: Optional[int] = None

    max_market_residual_quantity: Optional[float] = None
    max_total_residual_quantity: Optional[float] = None

    max_absolute_drawdown: Optional[float] = None
    max_drawdown_fraction: Optional[float] = None

    block_new_exposure_on_unknown_cost_basis: bool = True
    block_new_exposure_when_equity_unknown: bool = False

    quantity_epsilon: float = 1e-9

    def __post_init__(self) -> None:
        non_negative_float_fields = (
            "max_gross_cost_exposure",
            "max_marked_exposure",
            "max_market_residual_quantity",
            "max_total_residual_quantity",
            "max_absolute_drawdown",
            "max_drawdown_fraction",
        )

        for name in non_negative_float_fields:
            value = getattr(self, name)
            if value is None:
                continue

            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

            object.__setattr__(self, name, numeric)

        for name in (
            "max_open_tokens",
            "max_open_markets",
            "max_unresolved_executions",
            "max_unresolved_per_token",
        ):
            value = getattr(self, name)
            if value is None:
                continue

            integer = int(value)
            if integer < 0:
                raise ValueError(f"{name} must be non-negative")

            object.__setattr__(self, name, integer)

        fraction = self.max_drawdown_fraction
        if fraction is not None and fraction > 1.0:
            raise ValueError("max_drawdown_fraction must be in [0, 1]")

        epsilon = float(self.quantity_epsilon)
        if not math.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError("quantity_epsilon must be finite and non-negative")

        object.__setattr__(self, "quantity_epsilon", epsilon)


@dataclass(frozen=True, slots=True)
class EquityObservation:
    """One externally supplied total-account-equity observation."""

    equity: float
    observed_at: float = field(default_factory=time.time)
    source: str = ""

    def __post_init__(self) -> None:
        equity = float(self.equity)
        observed = float(self.observed_at)

        if not math.isfinite(equity) or equity < 0.0:
            raise ValueError("equity must be finite and non-negative")
        if not math.isfinite(observed) or observed <= 0.0:
            raise ValueError("observed_at must be positive")

        object.__setattr__(self, "equity", equity)
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "source", str(self.source or ""))


@dataclass(frozen=True, slots=True)
class DrawdownSnapshot:
    current_equity: Optional[float]
    peak_equity: Optional[float]
    absolute_drawdown: Optional[float]
    drawdown_fraction: Optional[float]
    observed_at: Optional[float]
    source: str = ""

    @property
    def known(self) -> bool:
        return self.current_equity is not None and self.peak_equity is not None


@dataclass(frozen=True, slots=True)
class MarketExposure:
    """Policy-neutral binary-market inventory summary."""

    market_id: str
    yes_quantity: float
    no_quantity: float
    paired_quantity: float
    residual_quantity: float
    residual_side: Optional[OutcomeSide]

    gross_cost_basis: float
    unknown_cost_quantity: float

    @property
    def balanced(self) -> bool:
        return self.residual_quantity <= 1e-9


@dataclass(frozen=True, slots=True)
class PortfolioRiskSnapshot:
    """Aggregated risk truth from position + lifecycle + equity state."""

    observed_at: float
    mode: RiskMode

    open_tokens: int
    open_markets: int

    gross_quantity: float
    gross_cost_exposure: float
    marked_exposure: Optional[float]
    marked_exposure_complete: bool

    unknown_cost_quantity: float
    unknown_cost_positions: int

    total_paired_quantity: float
    total_residual_quantity: float
    largest_market_residual_quantity: float
    market_exposures: Tuple[MarketExposure, ...]

    unresolved_executions: int
    unresolved_tokens: Tuple[str, ...]
    unresolved_states: Tuple[Tuple[str, str], ...]

    drawdown: DrawdownSnapshot

    manual_halt: bool
    manual_halt_reason: str = ""


@dataclass(frozen=True, slots=True)
class ProposedExposure:
    """Incremental exposure presented for a risk-capacity decision.

    This object contains execution economics only. It intentionally carries no
    candidate quality, model score, setup family, or strategy provenance.
    """

    action: RiskAction
    token_id: str
    market_id: str
    outcome_side: Optional[OutcomeSide] = None

    quantity: float = 0.0
    estimated_unit_cost: Optional[float] = None

    def __post_init__(self) -> None:
        token_id = str(self.token_id or "").strip()
        market_id = str(self.market_id or "").strip()
        quantity = max(0.0, float(self.quantity))

        if not token_id:
            raise ValueError("token_id is required")
        if not market_id:
            raise ValueError("market_id is required")
        if not math.isfinite(quantity):
            raise ValueError("quantity must be finite")

        unit_cost = self.estimated_unit_cost
        if unit_cost is not None:
            unit_cost = float(unit_cost)
            if not math.isfinite(unit_cost) or not 0.0 <= unit_cost <= 1.0:
                raise ValueError("estimated_unit_cost must be in [0, 1]")

        object.__setattr__(self, "token_id", token_id)
        object.__setattr__(self, "market_id", market_id)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "estimated_unit_cost", unit_cost)

    @property
    def estimated_incremental_cost(self) -> Optional[float]:
        if self.estimated_unit_cost is None:
            return None
        return self.quantity * self.estimated_unit_cost


@dataclass(frozen=True, slots=True)
class RiskCheck:
    code: str
    passed: bool
    severity: RiskCheckSeverity
    reason: str

    observed: Optional[float] = None
    limit: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", str(self.code or "UNKNOWN"))
        object.__setattr__(self, "reason", str(self.reason or ""))


@dataclass(frozen=True, slots=True)
class RiskDecision:
    action: RiskAction
    allowed: bool
    mode: RiskMode
    checks: Tuple[RiskCheck, ...]
    snapshot: PortfolioRiskSnapshot

    @property
    def hard_failures(self) -> Tuple[RiskCheck, ...]:
        return tuple(
            check
            for check in self.checks
            if not check.passed
            and check.severity is RiskCheckSeverity.HARD
        )

    @property
    def warnings(self) -> Tuple[RiskCheck, ...]:
        return tuple(
            check
            for check in self.checks
            if not check.passed
            and check.severity is RiskCheckSeverity.WARNING
        )


class ExecutionOwnershipReader(Protocol):
    """Minimal lifecycle interface consumed by the risk layer."""

    def owned_snapshots(self) -> Tuple[LifecycleSnapshot, ...]:
        ...

    def has_owner_for_token(self, token_id: str) -> bool:
        ...


class RiskManager:
    """Portfolio risk aggregator and configurable capacity gate."""

    def __init__(
        self,
        *,
        positions: PositionBook,
        execution_ownership: ExecutionOwnershipReader,
        limits: RiskLimits = RiskLimits(),
    ) -> None:
        self.positions = positions
        self.execution_ownership = execution_ownership
        self.limits = limits

        self._gate = threading.RLock()

        self._latest_equity: Optional[EquityObservation] = None
        self._peak_equity: Optional[float] = None

        self._manual_halt = False
        self._manual_halt_reason = ""

    # ------------------------------------------------------------------
    # Equity / drawdown ownership
    # ------------------------------------------------------------------

    def observe_equity(
        self,
        equity: float,
        *,
        observed_at: Optional[float] = None,
        source: str = "",
    ) -> DrawdownSnapshot:
        observation = EquityObservation(
            equity=equity,
            observed_at=float(observed_at or time.time()),
            source=source,
        )

        with self._gate:
            latest = self._latest_equity

            # Ignore a stale equity observation rather than moving drawdown state
            # backwards in time.
            if (
                latest is not None
                and observation.observed_at + 1e-9 < latest.observed_at
            ):
                return self._drawdown_locked()

            self._latest_equity = observation

            if self._peak_equity is None:
                self._peak_equity = observation.equity
            else:
                self._peak_equity = max(
                    self._peak_equity,
                    observation.equity,
                )

            return self._drawdown_locked()

    def reset_equity_peak(
        self,
        *,
        new_peak: Optional[float] = None,
    ) -> DrawdownSnapshot:
        """Explicitly reset the drawdown reference.

        This is an operator action, never an automatic consequence of a loss.
        """

        with self._gate:
            if new_peak is None:
                if self._latest_equity is None:
                    self._peak_equity = None
                else:
                    self._peak_equity = self._latest_equity.equity
            else:
                value = float(new_peak)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        "new_peak must be finite and non-negative"
                    )
                self._peak_equity = value

            return self._drawdown_locked()

    def _drawdown_locked(self) -> DrawdownSnapshot:
        latest = self._latest_equity
        peak = self._peak_equity

        if latest is None or peak is None:
            return DrawdownSnapshot(
                current_equity=None,
                peak_equity=peak,
                absolute_drawdown=None,
                drawdown_fraction=None,
                observed_at=(
                    latest.observed_at if latest is not None else None
                ),
                source=(latest.source if latest is not None else ""),
            )

        drawdown = max(0.0, peak - latest.equity)
        fraction = (
            drawdown / peak
            if peak > 0.0
            else (0.0 if drawdown <= 0.0 else None)
        )

        return DrawdownSnapshot(
            current_equity=latest.equity,
            peak_equity=peak,
            absolute_drawdown=drawdown,
            drawdown_fraction=fraction,
            observed_at=latest.observed_at,
            source=latest.source,
        )

    def drawdown(self) -> DrawdownSnapshot:
        with self._gate:
            return self._drawdown_locked()

    # ------------------------------------------------------------------
    # Manual halt
    # ------------------------------------------------------------------

    def set_manual_halt(self, reason: str) -> None:
        with self._gate:
            self._manual_halt = True
            self._manual_halt_reason = str(
                reason or "manual risk halt"
            )

    def clear_manual_halt(self) -> None:
        with self._gate:
            self._manual_halt = False
            self._manual_halt_reason = ""

    # ------------------------------------------------------------------
    # Portfolio aggregation
    # ------------------------------------------------------------------

    def _market_exposures(
        self,
        positions: Sequence[PositionSnapshot],
    ) -> Tuple[MarketExposure, ...]:
        by_market: Dict[str, Dict[OutcomeSide, PositionSnapshot]] = {}

        for position in positions:
            if position.economically_flat:
                continue

            row = by_market.setdefault(position.market_id, {})
            existing = row.get(position.outcome_side)

            if existing is None:
                row[position.outcome_side] = position
                continue

            # A market can theoretically expose multiple token rows only if the
            # upstream identity layer is malformed. Keep aggregation conservative
            # by synthesizing quantity/cost totals below rather than discarding one.
            # PositionBook normally provides one token per market/outcome.
            if existing.token_id != position.token_id:
                # Retain the larger row as the read-model representative; gross
                # portfolio totals are computed independently from all positions.
                if position.quantity > existing.quantity:
                    row[position.outcome_side] = position

        output: list[MarketExposure] = []

        # Aggregate from the full position sequence to avoid depending on the
        # representative selection above.
        for market_id in sorted(
            {
                position.market_id
                for position in positions
                if not position.economically_flat
            }
        ):
            market_positions = [
                position
                for position in positions
                if position.market_id == market_id
                and not position.economically_flat
            ]

            yes_qty = sum(
                position.quantity
                for position in market_positions
                if position.outcome_side is OutcomeSide.YES
            )
            no_qty = sum(
                position.quantity
                for position in market_positions
                if position.outcome_side is OutcomeSide.NO
            )

            paired = min(yes_qty, no_qty)
            difference = yes_qty - no_qty
            residual = abs(difference)

            residual_side = None
            if residual > self.limits.quantity_epsilon:
                residual_side = (
                    OutcomeSide.YES
                    if difference > 0.0
                    else OutcomeSide.NO
                )

            gross_cost_basis = sum(
                position.cost_basis
                for position in market_positions
            )
            unknown_cost_quantity = sum(
                position.unpriced_quantity
                for position in market_positions
            )

            output.append(
                MarketExposure(
                    market_id=market_id,
                    yes_quantity=yes_qty,
                    no_quantity=no_qty,
                    paired_quantity=paired,
                    residual_quantity=residual,
                    residual_side=residual_side,
                    gross_cost_basis=gross_cost_basis,
                    unknown_cost_quantity=unknown_cost_quantity,
                )
            )

        return tuple(output)

    def _risk_mode(
        self,
        *,
        drawdown: DrawdownSnapshot,
        unknown_cost_positions: int,
        unresolved_executions: int,
        open_tokens: int,
        open_markets: int,
        gross_cost_exposure: float,
        marked_exposure: Optional[float],
        total_residual: float,
        largest_residual: float,
    ) -> RiskMode:
        with self._gate:
            if self._manual_halt:
                return RiskMode.HALTED

        # Portfolio-limit breaches are reduce-only conditions. They block adding
        # exposure but preserve close/reconcile/settlement reachability.
        if (
            self.limits.block_new_exposure_on_unknown_cost_basis
            and unknown_cost_positions > 0
        ):
            return RiskMode.REDUCE_ONLY

        if (
            self.limits.block_new_exposure_when_equity_unknown
            and not drawdown.known
        ):
            return RiskMode.REDUCE_ONLY

        checks = (
            (
                self.limits.max_gross_cost_exposure,
                gross_cost_exposure,
            ),
            (
                self.limits.max_marked_exposure,
                marked_exposure,
            ),
            (
                (
                    float(self.limits.max_open_tokens)
                    if self.limits.max_open_tokens is not None
                    else None
                ),
                float(open_tokens),
            ),
            (
                (
                    float(self.limits.max_open_markets)
                    if self.limits.max_open_markets is not None
                    else None
                ),
                float(open_markets),
            ),
            (
                (
                    float(self.limits.max_unresolved_executions)
                    if self.limits.max_unresolved_executions is not None
                    else None
                ),
                float(unresolved_executions),
            ),
            (
                self.limits.max_total_residual_quantity,
                total_residual,
            ),
            (
                self.limits.max_market_residual_quantity,
                largest_residual,
            ),
        )

        for limit, observed in checks:
            if limit is None or observed is None:
                continue
            if observed > limit + self.limits.quantity_epsilon:
                return RiskMode.REDUCE_ONLY

        if (
            self.limits.max_absolute_drawdown is not None
            and drawdown.absolute_drawdown is not None
            and drawdown.absolute_drawdown
            >= self.limits.max_absolute_drawdown
            - self.limits.quantity_epsilon
        ):
            return RiskMode.REDUCE_ONLY

        if (
            self.limits.max_drawdown_fraction is not None
            and drawdown.drawdown_fraction is not None
            and drawdown.drawdown_fraction
            >= self.limits.max_drawdown_fraction
            - self.limits.quantity_epsilon
        ):
            return RiskMode.REDUCE_ONLY

        return RiskMode.NORMAL

    def snapshot(self) -> PortfolioRiskSnapshot:
        now = time.time()
        positions = self.positions.open_snapshots()
        market_exposures = self._market_exposures(positions)

        open_tokens = len(positions)
        open_markets = len(
            {
                position.market_id
                for position in positions
            }
        )

        gross_quantity = sum(
            position.quantity
            for position in positions
        )
        gross_cost_exposure = sum(
            position.cost_basis
            for position in positions
        )

        marked_known = [
            position.quantity * position.mark_price
            for position in positions
            if position.mark_price is not None
        ]
        marked_complete = all(
            position.mark_price is not None
            for position in positions
        )
        marked_exposure = (
            sum(marked_known)
            if marked_complete
            else None
        )

        unknown_cost_positions = sum(
            1
            for position in positions
            if position.cost_basis_state
            is not CostBasisState.COMPLETE
        )
        unknown_cost_quantity = sum(
            position.unpriced_quantity
            for position in positions
        )

        total_paired = sum(
            market.paired_quantity
            for market in market_exposures
        )
        total_residual = sum(
            market.residual_quantity
            for market in market_exposures
        )
        largest_residual = max(
            (
                market.residual_quantity
                for market in market_exposures
            ),
            default=0.0,
        )

        owners = self.execution_ownership.owned_snapshots()

        unresolved_tokens = tuple(
            sorted(
                {
                    owner.working_order.intent.token_id
                    for owner in owners
                }
            )
        )
        unresolved_states = tuple(
            sorted(
                (
                    owner.working_order.intent.token_id,
                    owner.state.value,
                )
                for owner in owners
            )
        )

        drawdown = self.drawdown()

        mode = self._risk_mode(
            drawdown=drawdown,
            unknown_cost_positions=unknown_cost_positions,
            unresolved_executions=len(owners),
            open_tokens=open_tokens,
            open_markets=open_markets,
            gross_cost_exposure=gross_cost_exposure,
            marked_exposure=marked_exposure,
            total_residual=total_residual,
            largest_residual=largest_residual,
        )

        with self._gate:
            manual_halt = self._manual_halt
            manual_reason = self._manual_halt_reason

        return PortfolioRiskSnapshot(
            observed_at=now,
            mode=mode,
            open_tokens=open_tokens,
            open_markets=open_markets,
            gross_quantity=gross_quantity,
            gross_cost_exposure=gross_cost_exposure,
            marked_exposure=marked_exposure,
            marked_exposure_complete=marked_complete,
            unknown_cost_quantity=unknown_cost_quantity,
            unknown_cost_positions=unknown_cost_positions,
            total_paired_quantity=total_paired,
            total_residual_quantity=total_residual,
            largest_market_residual_quantity=largest_residual,
            market_exposures=market_exposures,
            unresolved_executions=len(owners),
            unresolved_tokens=unresolved_tokens,
            unresolved_states=unresolved_states,
            drawdown=drawdown,
            manual_halt=manual_halt,
            manual_halt_reason=manual_reason,
        )

    # ------------------------------------------------------------------
    # Capacity assessment
    # ------------------------------------------------------------------

    @staticmethod
    def _check(
        code: str,
        *,
        passed: bool,
        severity: RiskCheckSeverity,
        reason: str,
        observed: Optional[float] = None,
        limit: Optional[float] = None,
    ) -> RiskCheck:
        return RiskCheck(
            code=code,
            passed=bool(passed),
            severity=severity,
            reason=reason,
            observed=observed,
            limit=limit,
        )

    def assess(
        self,
        proposal: ProposedExposure,
    ) -> RiskDecision:
        """Evaluate generic portfolio capacity for one economic action."""

        snapshot = self.snapshot()
        checks: list[RiskCheck] = []

        # Risk reduction and reconciliation must remain reachable in reduce-only
        # mode. A manual HALT is intentionally stronger and stops normal actions;
        # explicit reconciliation/settlement remain allowed because they reduce
        # uncertainty/exposure rather than create it.
        if snapshot.manual_halt:
            allowed_during_halt = proposal.action in {
                RiskAction.RECONCILE,
                RiskAction.SETTLE,
            }
            checks.append(
                self._check(
                    "MANUAL_HALT",
                    passed=allowed_during_halt,
                    severity=RiskCheckSeverity.HARD,
                    reason=(
                        "manual halt active; only reconciliation/settlement "
                        "remains reachable"
                    ),
                )
            )

        if proposal.action.risk_reducing:
            # Reduction paths do not need to satisfy new-exposure capacity. They
            # still receive current risk telemetry in the returned snapshot.
            allowed = not any(
                not check.passed
                and check.severity is RiskCheckSeverity.HARD
                for check in checks
            )

            return RiskDecision(
                action=proposal.action,
                allowed=allowed,
                mode=snapshot.mode,
                checks=tuple(checks),
                snapshot=snapshot,
            )

        # From this point onward the action creates/increases economic exposure.
        checks.append(
            self._check(
                "RISK_MODE",
                passed=snapshot.mode is RiskMode.NORMAL,
                severity=RiskCheckSeverity.HARD,
                reason=(
                    "new exposure requires NORMAL risk mode"
                    if snapshot.mode is not RiskMode.NORMAL
                    else "risk mode normal"
                ),
            )
        )

        same_token_owners = [
            owner
            for owner in self.execution_ownership.owned_snapshots()
            if owner.working_order.intent.token_id == proposal.token_id
        ]

        checks.append(
            self._check(
                "TOKEN_EXECUTION_OWNERSHIP",
                passed=not same_token_owners,
                severity=RiskCheckSeverity.HARD,
                reason=(
                    "token has no unresolved/live execution owner"
                    if not same_token_owners
                    else (
                        "token already has unresolved/live execution ownership; "
                        "duplicate new exposure denied"
                    )
                ),
                observed=float(len(same_token_owners)),
                limit=0.0,
            )
        )

        if self.limits.max_unresolved_per_token is not None:
            checks.append(
                self._check(
                    "UNRESOLVED_PER_TOKEN",
                    passed=(
                        len(same_token_owners)
                        < self.limits.max_unresolved_per_token
                    ),
                    severity=RiskCheckSeverity.HARD,
                    reason="per-token unresolved execution capacity",
                    observed=float(len(same_token_owners)),
                    limit=float(self.limits.max_unresolved_per_token),
                )
            )

        if self.limits.max_unresolved_executions is not None:
            checks.append(
                self._check(
                    "GLOBAL_UNRESOLVED_EXECUTIONS",
                    passed=(
                        snapshot.unresolved_executions
                        < self.limits.max_unresolved_executions
                    ),
                    severity=RiskCheckSeverity.HARD,
                    reason="global unresolved execution capacity",
                    observed=float(snapshot.unresolved_executions),
                    limit=float(
                        self.limits.max_unresolved_executions
                    ),
                )
            )

        if (
            self.limits.block_new_exposure_on_unknown_cost_basis
        ):
            checks.append(
                self._check(
                    "UNKNOWN_COST_BASIS",
                    passed=snapshot.unknown_cost_positions == 0,
                    severity=RiskCheckSeverity.HARD,
                    reason=(
                        "all confirmed inventory has complete cost basis"
                        if snapshot.unknown_cost_positions == 0
                        else (
                            "confirmed inventory has incomplete cost basis; "
                            "new exposure remains reduce-only"
                        )
                    ),
                    observed=float(snapshot.unknown_cost_positions),
                    limit=0.0,
                )
            )

        if self.limits.block_new_exposure_when_equity_unknown:
            checks.append(
                self._check(
                    "EQUITY_KNOWN",
                    passed=snapshot.drawdown.known,
                    severity=RiskCheckSeverity.HARD,
                    reason=(
                        "equity/drawdown observation available"
                        if snapshot.drawdown.known
                        else "equity unknown; new exposure blocked"
                    ),
                )
            )

        incremental_cost = proposal.estimated_incremental_cost

        if self.limits.max_gross_cost_exposure is not None:
            projected = (
                snapshot.gross_cost_exposure + incremental_cost
                if incremental_cost is not None
                else None
            )

            checks.append(
                self._check(
                    "GROSS_COST_EXPOSURE",
                    passed=(
                        projected is not None
                        and projected
                        <= self.limits.max_gross_cost_exposure
                        + self.limits.quantity_epsilon
                    ),
                    severity=RiskCheckSeverity.HARD,
                    reason=(
                        "projected gross cost exposure within limit"
                        if projected is not None
                        else (
                            "incremental cost unknown; cannot prove "
                            "gross-cost capacity"
                        )
                    ),
                    observed=projected,
                    limit=self.limits.max_gross_cost_exposure,
                )
            )

        if self.limits.max_marked_exposure is not None:
            projected_marked = (
                snapshot.marked_exposure + incremental_cost
                if snapshot.marked_exposure is not None
                and incremental_cost is not None
                else None
            )

            checks.append(
                self._check(
                    "MARKED_EXPOSURE",
                    passed=(
                        projected_marked is not None
                        and projected_marked
                        <= self.limits.max_marked_exposure
                        + self.limits.quantity_epsilon
                    ),
                    severity=RiskCheckSeverity.HARD,
                    reason=(
                        "projected marked exposure within limit"
                        if projected_marked is not None
                        else (
                            "marked or incremental exposure unknown; "
                            "capacity not proven"
                        )
                    ),
                    observed=projected_marked,
                    limit=self.limits.max_marked_exposure,
                )
            )

        current_token = self.positions.snapshot(proposal.token_id)
        token_was_open = bool(
            current_token is not None
            and not current_token.economically_flat
        )

        if self.limits.max_open_tokens is not None:
            projected_tokens = (
                snapshot.open_tokens
                if token_was_open
                else snapshot.open_tokens + 1
            )

            checks.append(
                self._check(
                    "OPEN_TOKEN_CAPACITY",
                    passed=(
                        projected_tokens
                        <= self.limits.max_open_tokens
                    ),
                    severity=RiskCheckSeverity.HARD,
                    reason="projected open-token capacity",
                    observed=float(projected_tokens),
                    limit=float(self.limits.max_open_tokens),
                )
            )

        market_was_open = any(
            exposure.market_id == proposal.market_id
            for exposure in snapshot.market_exposures
        )

        if self.limits.max_open_markets is not None:
            projected_markets = (
                snapshot.open_markets
                if market_was_open
                else snapshot.open_markets + 1
            )

            checks.append(
                self._check(
                    "OPEN_MARKET_CAPACITY",
                    passed=(
                        projected_markets
                        <= self.limits.max_open_markets
                    ),
                    severity=RiskCheckSeverity.HARD,
                    reason="projected open-market capacity",
                    observed=float(projected_markets),
                    limit=float(self.limits.max_open_markets),
                )
            )

        if (
            proposal.outcome_side is not None
            and proposal.quantity > 0.0
            and (
                self.limits.max_market_residual_quantity is not None
                or self.limits.max_total_residual_quantity is not None
            )
        ):
            existing_market = next(
                (
                    exposure
                    for exposure in snapshot.market_exposures
                    if exposure.market_id == proposal.market_id
                ),
                None,
            )

            yes_quantity = (
                existing_market.yes_quantity
                if existing_market is not None
                else 0.0
            )
            no_quantity = (
                existing_market.no_quantity
                if existing_market is not None
                else 0.0
            )
            old_residual = abs(yes_quantity - no_quantity)

            if proposal.outcome_side is OutcomeSide.YES:
                yes_quantity += proposal.quantity
            else:
                no_quantity += proposal.quantity

            projected_market_residual = abs(
                yes_quantity - no_quantity
            )
            projected_total_residual = (
                snapshot.total_residual_quantity
                - old_residual
                + projected_market_residual
            )

            if self.limits.max_market_residual_quantity is not None:
                checks.append(
                    self._check(
                        "MARKET_RESIDUAL_CAPACITY",
                        passed=(
                            projected_market_residual
                            <= self.limits.max_market_residual_quantity
                            + self.limits.quantity_epsilon
                        ),
                        severity=RiskCheckSeverity.HARD,
                        reason="projected market residual inventory",
                        observed=projected_market_residual,
                        limit=self.limits.max_market_residual_quantity,
                    )
                )

            if self.limits.max_total_residual_quantity is not None:
                checks.append(
                    self._check(
                        "TOTAL_RESIDUAL_CAPACITY",
                        passed=(
                            projected_total_residual
                            <= self.limits.max_total_residual_quantity
                            + self.limits.quantity_epsilon
                        ),
                        severity=RiskCheckSeverity.HARD,
                        reason="projected total residual inventory",
                        observed=projected_total_residual,
                        limit=self.limits.max_total_residual_quantity,
                    )
                )

        if self.limits.max_absolute_drawdown is not None:
            dd = snapshot.drawdown.absolute_drawdown
            checks.append(
                self._check(
                    "ABSOLUTE_DRAWDOWN",
                    passed=(
                        dd is not None
                        and dd
                        < self.limits.max_absolute_drawdown
                        - self.limits.quantity_epsilon
                    ),
                    severity=RiskCheckSeverity.HARD,
                    reason=(
                        "absolute drawdown below reduce-only boundary"
                        if dd is not None
                        else "drawdown unknown"
                    ),
                    observed=dd,
                    limit=self.limits.max_absolute_drawdown,
                )
            )

        if self.limits.max_drawdown_fraction is not None:
            fraction = snapshot.drawdown.drawdown_fraction
            checks.append(
                self._check(
                    "DRAWDOWN_FRACTION",
                    passed=(
                        fraction is not None
                        and fraction
                        < self.limits.max_drawdown_fraction
                        - self.limits.quantity_epsilon
                    ),
                    severity=RiskCheckSeverity.HARD,
                    reason=(
                        "drawdown fraction below reduce-only boundary"
                        if fraction is not None
                        else "drawdown fraction unknown"
                    ),
                    observed=fraction,
                    limit=self.limits.max_drawdown_fraction,
                )
            )

        allowed = not any(
            not check.passed
            and check.severity is RiskCheckSeverity.HARD
            for check in checks
        )

        return RiskDecision(
            action=proposal.action,
            allowed=allowed,
            mode=snapshot.mode,
            checks=tuple(checks),
            snapshot=snapshot,
        )
