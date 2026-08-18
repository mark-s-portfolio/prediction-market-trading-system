"""
Exact-order reconciliation for the public portfolio edition.

Reconciliation sits between transport, fill accounting, and lifecycle ownership.
It resolves ambiguous execution state from independent venue evidence without
turning missing data into a zero-fill conclusion.

Core invariants:
- UNKNOWN status is not ZERO
- a successful local cancel call is not, by itself, zero-fill proof
- exact positive fill evidence dominates older/provisional zero evidence
- terminal-zero requires exact terminal order evidence plus a fresh same-token
  inventory proof
- wallet quantity is attributable to an order only with a valid pre-order baseline
- when no baseline exists, a fresh absolute zero can prove no current token
  inventory after exact terminal/zero-matched evidence; a non-zero balance remains
  ambiguous
- the opposite token is never queried to prove one order's zero fill
- repeated probes are singleflight per lifecycle
- unresolved no-OID submissions remain fail-closed unless an optional exact-order
  locator can recover their venue order identity

The module contains no strategy thresholds, asset-specific setup knowledge, or
admission policy.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import inspect
import math
import time
from typing import Awaitable, Callable, Dict, Mapping, Optional, Protocol, Union

from src.execution.clob_transport import ClobTransport
from src.execution.fill_accounting import (
    FillAccounting,
    FillEvidenceSource,
    OrderFillRecord,
    WalletBaseline,
    WalletDeltaObservation,
)
from src.execution.order_lifecycle import (
    LifecycleSnapshot,
    OrderLifecycleService,
    normalize_venue_order_snapshot,
)
from src.execution.types import (
    FillSummary,
    LifecycleIdentity,
    OrderLifecycleState,
    ReconciliationOutcome,
    ReconciliationResult,
    VenueOrderSnapshot,
)
from src.runtime.logging import runtime_print


MaybeAwaitable = Union[object, Awaitable[object]]


class WalletBalanceReader(Protocol):
    """Return the current conditional-token balance for one exact token."""

    def __call__(self, token_id: str) -> Union[float, Awaitable[float]]:
        ...


class ExactTradeReader(Protocol):
    """Return venue trade/fill payload scoped to one exact order ID."""

    def __call__(self, order_id: str) -> MaybeAwaitable:
        ...


class ExactOrderLocator(Protocol):
    """Optionally recover an exact venue order for an ambiguous no-OID submit.

    Implementations should match immutable submission identity (for example a
    caller-generated client reference plus token/side/price/size/time envelope).
    Returning a fuzzy or merely same-token order would violate lifecycle affinity.
    """

    def __call__(self, snapshot: LifecycleSnapshot) -> MaybeAwaitable:
        ...


class ExactTerminalProofReader(Protocol):
    """Return exact local/durable terminal proof for one known order ID.

    This hook exists for systems where a cancellation service has already obtained
    and persisted venue terminal evidence. A bare "cancel request returned 200" is
    not sufficient proof.
    """

    def __call__(self, order_id: str) -> Union[bool, Awaitable[bool]]:
        ...


@dataclass(frozen=True, slots=True)
class ReconciliationPolicy:
    """Operational reconciliation behavior only."""

    attempts: int = 3
    retry_delay_seconds: float = 0.50

    quantity_epsilon: float = 1e-9
    absolute_zero_epsilon: float = 1e-6

    allow_baselineless_absolute_zero: bool = True
    query_trades_after_positive_status: bool = True
    query_trades_on_terminal_zero_candidate: bool = True

    apply_result_to_lifecycle: bool = True


@dataclass(frozen=True, slots=True)
class ReconciliationEvidence:
    """Auditable evidence collected by one reconciliation pass."""

    lifecycle: LifecycleIdentity
    order_id: Optional[str]
    status_observed: bool
    raw_status: str
    status_state: Optional[OrderLifecycleState]
    status_matched_size: float

    trade_observed: bool
    accounting_filled_size: float
    accounting_average_price: Optional[float]

    wallet_observed: bool
    wallet_baseline_valid: bool
    wallet_balance: Optional[float]
    wallet_delta: Optional[float]

    terminal_proven: bool
    baselineless_absolute_zero: bool

    attempts_used: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_mapping(self) -> dict[str, object]:
        return {
            "lifecycle_id": self.lifecycle.lifecycle_id,
            "attempt_id": self.lifecycle.attempt_id,
            "generation": self.lifecycle.generation,
            "order_id": self.order_id,
            "status_observed": self.status_observed,
            "raw_status": self.raw_status,
            "status_state": (
                self.status_state.value if self.status_state is not None else None
            ),
            "status_matched_size": self.status_matched_size,
            "trade_observed": self.trade_observed,
            "accounting_filled_size": self.accounting_filled_size,
            "accounting_average_price": self.accounting_average_price,
            "wallet_observed": self.wallet_observed,
            "wallet_baseline_valid": self.wallet_baseline_valid,
            "wallet_balance": self.wallet_balance,
            "wallet_delta": self.wallet_delta,
            "terminal_proven": self.terminal_proven,
            "baselineless_absolute_zero": self.baselineless_absolute_zero,
            "attempts_used": self.attempts_used,
            "notes": self.notes,
        }


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _mapping_has_status_evidence(payload: object) -> bool:
    if not isinstance(payload, Mapping) or not payload:
        return False

    return any(
        key in payload
        for key in (
            "status",
            "state",
            "size_matched",
            "sizeMatched",
            "matched_size",
            "matchedSize",
            "filled_size",
            "filledSize",
            "filled",
        )
    )


def _terminal_cancel_state(state: Optional[OrderLifecycleState]) -> bool:
    return state is OrderLifecycleState.CANCELLED


def _terminal_reject_state(state: Optional[OrderLifecycleState]) -> bool:
    return state in {
        OrderLifecycleState.REJECTED,
        OrderLifecycleState.FAILED,
    }


class ReconciliationService:
    """Singleflight reconciliation coordinator for exact execution lifecycles."""

    def __init__(
        self,
        *,
        transport: ClobTransport,
        lifecycle: OrderLifecycleService,
        accounting: FillAccounting,
        wallet_balance_reader: Optional[WalletBalanceReader] = None,
        trade_reader: Optional[ExactTradeReader] = None,
        order_locator: Optional[ExactOrderLocator] = None,
        terminal_proof_reader: Optional[ExactTerminalProofReader] = None,
        policy: ReconciliationPolicy = ReconciliationPolicy(),
    ) -> None:
        self.transport = transport
        self.lifecycle = lifecycle
        self.accounting = accounting

        self.wallet_balance_reader = wallet_balance_reader
        self.trade_reader = trade_reader
        self.order_locator = order_locator
        self.terminal_proof_reader = terminal_proof_reader
        self.policy = policy

        self._tasks: Dict[str, asyncio.Task[ReconciliationResult]] = {}
        self._task_gate = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public singleflight API
    # ------------------------------------------------------------------

    async def reconcile(
        self,
        lifecycle_identity: LifecycleIdentity,
        *,
        wallet_baseline: Optional[WalletBaseline] = None,
        reason: str = "",
    ) -> ReconciliationResult:
        """Reconcile one lifecycle, sharing concurrent callers for the same owner."""

        key = (
            f"{lifecycle_identity.lifecycle_id}:"
            f"{lifecycle_identity.attempt_id}:"
            f"{lifecycle_identity.generation}"
        )

        async with self._task_gate:
            existing = self._tasks.get(key)
            if existing is not None and not existing.done():
                task = existing
            else:
                task = asyncio.create_task(
                    self._reconcile_owned(
                        lifecycle_identity,
                        wallet_baseline=wallet_baseline,
                        reason=reason,
                    ),
                    name=f"reconcile-{lifecycle_identity.lifecycle_id[:16]}",
                )
                self._tasks[key] = task

                def _release(done_task: asyncio.Task[ReconciliationResult]) -> None:
                    current = self._tasks.get(key)
                    if current is done_task:
                        self._tasks.pop(key, None)

                    if done_task.cancelled():
                        return

                    try:
                        done_task.exception()
                    except Exception:
                        pass

                task.add_done_callback(_release)

        return await asyncio.shield(task)

    def active_reconciliations(self) -> int:
        return sum(1 for task in self._tasks.values() if not task.done())

    # ------------------------------------------------------------------
    # Evidence acquisition
    # ------------------------------------------------------------------

    async def _read_status(self, order_id: str) -> object:
        return await asyncio.to_thread(self.transport.get_order, order_id)

    async def _read_trades(self, order_id: str) -> object:
        reader = self.trade_reader
        if reader is None:
            return None
        return await _maybe_await(reader(order_id))

    async def _read_wallet(self, token_id: str) -> Optional[float]:
        reader = self.wallet_balance_reader
        if reader is None:
            return None

        value = await _maybe_await(reader(token_id))
        result = float(value)

        if not math.isfinite(result) or result < 0.0:
            raise ValueError("wallet balance reader returned invalid quantity")

        return result

    async def _read_terminal_proof(self, order_id: str) -> bool:
        reader = self.terminal_proof_reader
        if reader is None:
            return False

        return bool(await _maybe_await(reader(order_id)))

    async def _locate_exact_order(
        self,
        snapshot: LifecycleSnapshot,
    ) -> Optional[object]:
        locator = self.order_locator
        if locator is None:
            return None

        located = await _maybe_await(locator(snapshot))
        if located is None:
            return None

        if isinstance(located, str):
            return {"orderID": located}

        if isinstance(located, Mapping):
            return located

        raise TypeError(
            "ExactOrderLocator must return None, an order ID string, or mapping"
        )

    # ------------------------------------------------------------------
    # Accounting helpers
    # ------------------------------------------------------------------

    def _ensure_accounting(
        self,
        snapshot: LifecycleSnapshot,
        order_id: str,
    ) -> OrderFillRecord:
        existing = self.accounting.get(order_id)
        if existing is not None:
            return existing

        intent = snapshot.working_order.intent

        return self.accounting.register_order(
            order_id=order_id,
            token_id=intent.token_id,
            lifecycle=intent.lifecycle,
            order_side=intent.order_side,
            requested_size=intent.size,
            submitted_limit_price=intent.price,
            created_at=intent.created_at,
        )

    def _accounting_summary(
        self,
        order_id: Optional[str],
    ) -> Optional[FillSummary]:
        if not order_id:
            return None

        return self.accounting.fill_summary(order_id)

    async def _apply_trade_evidence(
        self,
        *,
        order_id: str,
    ) -> tuple[Optional[OrderFillRecord], bool, Optional[str]]:
        if self.trade_reader is None:
            return self.accounting.get(order_id), False, None

        try:
            payload = await self._read_trades(order_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return (
                self.accounting.get(order_id),
                False,
                f"trade_read_error:{type(exc).__name__}",
            )

        if payload is None:
            return self.accounting.get(order_id), False, "trade_payload_missing"

        try:
            record = self.accounting.observe_trade_payload(
                order_id=order_id,
                payload=payload,
            )
            return record, True, None
        except Exception as exc:
            return (
                self.accounting.get(order_id),
                False,
                f"trade_accounting_error:{type(exc).__name__}",
            )

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    def _result(
        self,
        *,
        outcome: ReconciliationOutcome,
        lifecycle_identity: LifecycleIdentity,
        order_id: Optional[str],
        snapshot: Optional[VenueOrderSnapshot],
        evidence: ReconciliationEvidence,
        reason: str,
    ) -> ReconciliationResult:
        result = ReconciliationResult(
            outcome=outcome,
            lifecycle=lifecycle_identity,
            order_id=order_id,
            snapshot=snapshot,
            fill_summary=self._accounting_summary(order_id),
            reason=reason,
            evidence=evidence.as_mapping(),
        )

        if self.policy.apply_result_to_lifecycle:
            try:
                self.lifecycle.apply_reconciliation(result)
            except Exception as exc:
                # Evidence remains valid even when a newer generation raced this
                # result.  Never mutate/relabel it; surface the ownership conflict.
                runtime_print(
                    f"[reconcile] lifecycle apply rejected "
                    f"{lifecycle_identity.lifecycle_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

        return result

    # ------------------------------------------------------------------
    # Core state machine
    # ------------------------------------------------------------------

    async def _reconcile_owned(
        self,
        lifecycle_identity: LifecycleIdentity,
        *,
        wallet_baseline: Optional[WalletBaseline],
        reason: str,
    ) -> ReconciliationResult:
        initial = self.lifecycle.get(lifecycle_identity)
        if initial is None:
            raise KeyError(
                f"unknown lifecycle: {lifecycle_identity.lifecycle_id}"
            )

        # Preserve the exact generation throughout this reconciliation task.
        if initial.lifecycle != lifecycle_identity:
            raise RuntimeError("lifecycle generation changed before reconciliation")

        self.lifecycle.mark_reconciling(
            lifecycle_identity,
            reason=reason or "reconciliation started",
        )

        intent = initial.working_order.intent
        order_id = initial.order_id

        notes: list[str] = []
        status_payload: object = None
        status_snapshot: Optional[VenueOrderSnapshot] = None
        status_observed = False
        terminal_proven = False

        trade_observed = False
        wallet_observed = False
        wallet_balance: Optional[float] = None
        wallet_delta_observation: Optional[WalletDeltaObservation] = None
        baselineless_absolute_zero = False

        attempts_used = 0

        # Ambiguous raw POSTs can lack an OID.  Exact identity recovery is optional
        # and intentionally delegated to a venue-specific locator.
        if not order_id:
            try:
                located = await self._locate_exact_order(initial)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                located = None
                notes.append(f"order_locator_error:{type(exc).__name__}")

            if isinstance(located, Mapping):
                located_id = str(
                    located.get("orderID")
                    or located.get("order_id")
                    or located.get("id")
                    or located.get("orderId")
                    or ""
                ).strip()

                if located_id:
                    order_id = located_id
                    status_payload = located
                    status_observed = _mapping_has_status_evidence(located)
                    notes.append("exact_order_recovered_by_locator")
                else:
                    notes.append("locator_returned_no_exact_order_id")

        # No exact OID means we cannot query exact status/trades.  A baseline-backed
        # positive wallet delta is still important exposure evidence, but without a
        # venue order identity we do not invent a synthetic order or terminal state.
        if not order_id:
            if self.wallet_balance_reader is not None:
                try:
                    wallet_balance = await self._read_wallet(intent.token_id)
                    wallet_observed = wallet_balance is not None

                    if wallet_observed:
                        from src.execution.fill_accounting import wallet_delta_from_baseline

                        wallet_delta_observation = wallet_delta_from_baseline(
                            token_id=intent.token_id,
                            current_balance=float(wallet_balance),
                            requested_size=intent.size,
                            baseline=wallet_baseline,
                            epsilon=self.policy.quantity_epsilon,
                        )

                        if wallet_delta_observation.proves_positive_inventory:
                            notes.append(
                                "positive_wallet_delta_without_exact_order_id"
                            )
                        elif wallet_delta_observation.baseline_valid:
                            notes.append(
                                "zero_wallet_delta_does_not_prove_no_venue_order"
                            )
                        elif float(wallet_balance) > self.policy.absolute_zero_epsilon:
                            notes.append(
                                "baselineless_nonzero_wallet_without_exact_order_id"
                            )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    notes.append(f"wallet_read_error:{type(exc).__name__}")

            evidence = ReconciliationEvidence(
                lifecycle=lifecycle_identity,
                order_id=None,
                status_observed=False,
                raw_status="",
                status_state=None,
                status_matched_size=0.0,
                trade_observed=False,
                accounting_filled_size=0.0,
                accounting_average_price=None,
                wallet_observed=wallet_observed,
                wallet_baseline_valid=bool(
                    wallet_delta_observation
                    and wallet_delta_observation.baseline_valid
                ),
                wallet_balance=wallet_balance,
                wallet_delta=(
                    wallet_delta_observation.delta
                    if wallet_delta_observation is not None
                    else None
                ),
                terminal_proven=False,
                baselineless_absolute_zero=False,
                attempts_used=0,
                notes=tuple(notes),
            )

            return self._result(
                outcome=ReconciliationOutcome.STILL_AMBIGUOUS,
                lifecycle_identity=lifecycle_identity,
                order_id=None,
                snapshot=None,
                evidence=evidence,
                reason=(
                    "no exact venue order identity; preserve ambiguous ownership"
                ),
            )

        # From this point onward all evidence is exact-order affine.
        self._ensure_accounting(initial, order_id)

        attempts = max(1, int(self.policy.attempts))
        retry_delay = max(0.0, float(self.policy.retry_delay_seconds))

        for attempt in range(1, attempts + 1):
            attempts_used = attempt

            # If the locator already supplied a status-bearing exact-order mapping,
            # consume it before issuing another network read.
            payload = status_payload if attempt == 1 and status_payload is not None else None

            if payload is None:
                try:
                    payload = await self._read_status(order_id)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    notes.append(f"status_read_error:{type(exc).__name__}")
                    payload = None

            observed = _mapping_has_status_evidence(payload)

            if observed:
                status_observed = True
                status_payload = payload

                status_snapshot = normalize_venue_order_snapshot(
                    payload,
                    intent=intent,
                    order_id=order_id,
                )

                try:
                    self.accounting.observe_order_payload(
                        order_id=order_id,
                        payload=payload,
                        source=FillEvidenceSource.ORDER_STATUS,
                        confirm_quantity=True,
                    )
                except Exception as exc:
                    notes.append(
                        f"status_accounting_error:{type(exc).__name__}"
                    )

                break

            # UNKNOWN is not ZERO. Retry within the bounded owner task.
            notes.append("status_unknown")
            if attempt < attempts and retry_delay > 0.0:
                await asyncio.sleep(retry_delay)

        accounting_record = self.accounting.get(order_id)
        accounting_fill = (
            accounting_record.confirmed_filled_size
            if accounting_record is not None
            else 0.0
        )

        # Positive exact status evidence gets a trade-endpoint refinement before
        # classification.  Trades can be fresher than cumulative status and can
        # also provide a true weighted average.
        status_matched = (
            status_snapshot.matched_size
            if status_snapshot is not None
            else 0.0
        )

        positive_status = max(status_matched, accounting_fill) > self.policy.quantity_epsilon

        if positive_status and self.policy.query_trades_after_positive_status:
            record, trade_observed, trade_note = await self._apply_trade_evidence(
                order_id=order_id,
            )
            if trade_note:
                notes.append(trade_note)
            if record is not None:
                accounting_record = record
                accounting_fill = record.confirmed_filled_size

        # Positive evidence always wins over provisional/no-fill interpretations.
        positive_fill = max(status_matched, accounting_fill)

        if positive_fill > self.policy.quantity_epsilon:
            requested = intent.size

            outcome = (
                ReconciliationOutcome.FILLED
                if positive_fill >= requested - self.policy.quantity_epsilon
                else ReconciliationOutcome.PARTIALLY_FILLED
            )

            evidence = ReconciliationEvidence(
                lifecycle=lifecycle_identity,
                order_id=order_id,
                status_observed=status_observed,
                raw_status=(
                    status_snapshot.raw_status
                    if status_snapshot is not None
                    else ""
                ),
                status_state=(
                    status_snapshot.state
                    if status_snapshot is not None
                    else None
                ),
                status_matched_size=status_matched,
                trade_observed=trade_observed,
                accounting_filled_size=accounting_fill,
                accounting_average_price=(
                    accounting_record.realized_average_price
                    if accounting_record is not None
                    else None
                ),
                wallet_observed=False,
                wallet_baseline_valid=False,
                wallet_balance=None,
                wallet_delta=None,
                terminal_proven=bool(
                    status_snapshot
                    and status_snapshot.state.is_terminal
                ),
                baselineless_absolute_zero=False,
                attempts_used=attempts_used,
                notes=tuple(notes),
            )

            return self._result(
                outcome=outcome,
                lifecycle_identity=lifecycle_identity,
                order_id=order_id,
                snapshot=status_snapshot,
                evidence=evidence,
                reason="exact positive fill evidence",
            )

        # An unreadable/empty exact status response is not zero-fill proof.
        if not status_observed or status_snapshot is None:
            evidence = ReconciliationEvidence(
                lifecycle=lifecycle_identity,
                order_id=order_id,
                status_observed=False,
                raw_status="",
                status_state=None,
                status_matched_size=0.0,
                trade_observed=trade_observed,
                accounting_filled_size=accounting_fill,
                accounting_average_price=(
                    accounting_record.realized_average_price
                    if accounting_record is not None
                    else None
                ),
                wallet_observed=False,
                wallet_baseline_valid=False,
                wallet_balance=None,
                wallet_delta=None,
                terminal_proven=False,
                baselineless_absolute_zero=False,
                attempts_used=attempts_used,
                notes=tuple(notes),
            )

            return self._result(
                outcome=ReconciliationOutcome.STILL_AMBIGUOUS,
                lifecycle_identity=lifecycle_identity,
                order_id=order_id,
                snapshot=None,
                evidence=evidence,
                reason="exact order status unknown; preserve owner",
            )

        # A live/working order with zero current fill remains an exact working owner.
        if not status_snapshot.state.is_terminal:
            evidence = ReconciliationEvidence(
                lifecycle=lifecycle_identity,
                order_id=order_id,
                status_observed=True,
                raw_status=status_snapshot.raw_status,
                status_state=status_snapshot.state,
                status_matched_size=status_snapshot.matched_size,
                trade_observed=trade_observed,
                accounting_filled_size=accounting_fill,
                accounting_average_price=(
                    accounting_record.realized_average_price
                    if accounting_record is not None
                    else None
                ),
                wallet_observed=False,
                wallet_baseline_valid=False,
                wallet_balance=None,
                wallet_delta=None,
                terminal_proven=False,
                baselineless_absolute_zero=False,
                attempts_used=attempts_used,
                notes=tuple(notes),
            )

            return self._result(
                outcome=ReconciliationOutcome.WORKING,
                lifecycle_identity=lifecycle_identity,
                order_id=order_id,
                snapshot=status_snapshot,
                evidence=evidence,
                reason="exact order remains working",
            )

        # Terminal status with zero matched can still race a late/hidden fill.
        # Before terminal-zero classification, optionally query exact trades.
        if self.policy.query_trades_on_terminal_zero_candidate:
            record, trade_observed_now, trade_note = await self._apply_trade_evidence(
                order_id=order_id,
            )
            trade_observed = bool(trade_observed or trade_observed_now)
            if trade_note:
                notes.append(trade_note)

            if record is not None:
                accounting_record = record
                accounting_fill = record.confirmed_filled_size

            if accounting_fill > self.policy.quantity_epsilon:
                outcome = (
                    ReconciliationOutcome.FILLED
                    if accounting_fill
                    >= intent.size - self.policy.quantity_epsilon
                    else ReconciliationOutcome.PARTIALLY_FILLED
                )

                evidence = ReconciliationEvidence(
                    lifecycle=lifecycle_identity,
                    order_id=order_id,
                    status_observed=True,
                    raw_status=status_snapshot.raw_status,
                    status_state=status_snapshot.state,
                    status_matched_size=status_snapshot.matched_size,
                    trade_observed=trade_observed,
                    accounting_filled_size=accounting_fill,
                    accounting_average_price=(
                        accounting_record.realized_average_price
                        if accounting_record is not None
                        else None
                    ),
                    wallet_observed=False,
                    wallet_baseline_valid=False,
                    wallet_balance=None,
                    wallet_delta=None,
                    terminal_proven=True,
                    baselineless_absolute_zero=False,
                    attempts_used=attempts_used,
                    notes=tuple(notes + ["positive_trade_after_terminal_status"]),
                )

                return self._result(
                    outcome=outcome,
                    lifecycle_identity=lifecycle_identity,
                    order_id=order_id,
                    snapshot=status_snapshot,
                    evidence=evidence,
                    reason="exact trade evidence overrides terminal-zero candidate",
                )

        # Exact persisted terminal proof can supplement a status label, but never
        # substitutes for same-token inventory proof.
        terminal_proven = bool(
            _terminal_cancel_state(status_snapshot.state)
            or _terminal_reject_state(status_snapshot.state)
        )

        if not terminal_proven:
            try:
                terminal_proven = await self._read_terminal_proof(order_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                notes.append(f"terminal_proof_error:{type(exc).__name__}")

        if not terminal_proven:
            evidence = ReconciliationEvidence(
                lifecycle=lifecycle_identity,
                order_id=order_id,
                status_observed=True,
                raw_status=status_snapshot.raw_status,
                status_state=status_snapshot.state,
                status_matched_size=status_snapshot.matched_size,
                trade_observed=trade_observed,
                accounting_filled_size=accounting_fill,
                accounting_average_price=(
                    accounting_record.realized_average_price
                    if accounting_record is not None
                    else None
                ),
                wallet_observed=False,
                wallet_baseline_valid=False,
                wallet_balance=None,
                wallet_delta=None,
                terminal_proven=False,
                baselineless_absolute_zero=False,
                attempts_used=attempts_used,
                notes=tuple(notes),
            )

            return self._result(
                outcome=ReconciliationOutcome.STILL_AMBIGUOUS,
                lifecycle_identity=lifecycle_identity,
                order_id=order_id,
                snapshot=status_snapshot,
                evidence=evidence,
                reason="terminal order state not proven",
            )

        # Terminal + zero matched still requires a fresh same-token inventory proof.
        if self.wallet_balance_reader is None:
            notes.append("wallet_reader_unavailable")

            evidence = ReconciliationEvidence(
                lifecycle=lifecycle_identity,
                order_id=order_id,
                status_observed=True,
                raw_status=status_snapshot.raw_status,
                status_state=status_snapshot.state,
                status_matched_size=status_snapshot.matched_size,
                trade_observed=trade_observed,
                accounting_filled_size=accounting_fill,
                accounting_average_price=(
                    accounting_record.realized_average_price
                    if accounting_record is not None
                    else None
                ),
                wallet_observed=False,
                wallet_baseline_valid=False,
                wallet_balance=None,
                wallet_delta=None,
                terminal_proven=True,
                baselineless_absolute_zero=False,
                attempts_used=attempts_used,
                notes=tuple(notes),
            )

            return self._result(
                outcome=ReconciliationOutcome.STILL_AMBIGUOUS,
                lifecycle_identity=lifecycle_identity,
                order_id=order_id,
                snapshot=status_snapshot,
                evidence=evidence,
                reason="terminal-zero candidate lacks fresh wallet proof",
            )

        try:
            wallet_balance = await self._read_wallet(intent.token_id)
            wallet_observed = wallet_balance is not None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            notes.append(f"wallet_read_error:{type(exc).__name__}")
            wallet_balance = None
            wallet_observed = False

        if not wallet_observed or wallet_balance is None:
            evidence = ReconciliationEvidence(
                lifecycle=lifecycle_identity,
                order_id=order_id,
                status_observed=True,
                raw_status=status_snapshot.raw_status,
                status_state=status_snapshot.state,
                status_matched_size=status_snapshot.matched_size,
                trade_observed=trade_observed,
                accounting_filled_size=accounting_fill,
                accounting_average_price=(
                    accounting_record.realized_average_price
                    if accounting_record is not None
                    else None
                ),
                wallet_observed=False,
                wallet_baseline_valid=False,
                wallet_balance=None,
                wallet_delta=None,
                terminal_proven=True,
                baselineless_absolute_zero=False,
                attempts_used=attempts_used,
                notes=tuple(notes),
            )

            return self._result(
                outcome=ReconciliationOutcome.STILL_AMBIGUOUS,
                lifecycle_identity=lifecycle_identity,
                order_id=order_id,
                snapshot=status_snapshot,
                evidence=evidence,
                reason="fresh wallet proof unavailable",
            )

        # With an authoritative baseline, account the exact delta and preserve
        # overfills rather than clamping them.
        if (
            wallet_baseline is not None
            and wallet_baseline.valid
            and wallet_baseline.token_id == intent.token_id
        ):
            try:
                accounting_record, wallet_delta_observation = (
                    self.accounting.observe_wallet_balance(
                        order_id=order_id,
                        current_balance=wallet_balance,
                        baseline=wallet_baseline,
                    )
                )
                accounting_fill = accounting_record.confirmed_filled_size
            except Exception as exc:
                notes.append(
                    f"wallet_accounting_error:{type(exc).__name__}"
                )
                wallet_delta_observation = None

            if (
                wallet_delta_observation is not None
                and wallet_delta_observation.delta
                > self.policy.quantity_epsilon
            ):
                outcome = (
                    ReconciliationOutcome.FILLED
                    if accounting_fill
                    >= intent.size - self.policy.quantity_epsilon
                    else ReconciliationOutcome.PARTIALLY_FILLED
                )

                evidence = ReconciliationEvidence(
                    lifecycle=lifecycle_identity,
                    order_id=order_id,
                    status_observed=True,
                    raw_status=status_snapshot.raw_status,
                    status_state=status_snapshot.state,
                    status_matched_size=status_snapshot.matched_size,
                    trade_observed=trade_observed,
                    accounting_filled_size=accounting_fill,
                    accounting_average_price=(
                        accounting_record.realized_average_price
                        if accounting_record is not None
                        else None
                    ),
                    wallet_observed=True,
                    wallet_baseline_valid=True,
                    wallet_balance=wallet_balance,
                    wallet_delta=wallet_delta_observation.delta,
                    terminal_proven=True,
                    baselineless_absolute_zero=False,
                    attempts_used=attempts_used,
                    notes=tuple(notes + ["positive_post_terminal_wallet_delta"]),
                )

                return self._result(
                    outcome=outcome,
                    lifecycle_identity=lifecycle_identity,
                    order_id=order_id,
                    snapshot=status_snapshot,
                    evidence=evidence,
                    reason="fresh baseline-backed wallet delta proves late fill",
                )

            # Valid baseline + fresh zero delta completes terminal-zero proof.
            zero_proven = bool(
                wallet_delta_observation is not None
                and wallet_delta_observation.baseline_valid
                and wallet_delta_observation.delta
                <= self.policy.quantity_epsilon
            )

            if not zero_proven:
                notes.append("baseline_wallet_zero_not_proven")

        else:
            # No baseline: only a fresh absolute zero may prove that this token
            # currently has no inventory. A non-zero balance cannot be attributed
            # either to this order or to pre-existing inventory, so remain ambiguous.
            baselineless_absolute_zero = bool(
                self.policy.allow_baselineless_absolute_zero
                and wallet_balance
                <= max(
                    0.0,
                    float(self.policy.absolute_zero_epsilon),
                )
            )
            zero_proven = baselineless_absolute_zero

            if not zero_proven:
                notes.append(
                    "baselineless_nonzero_wallet_is_ambiguous"
                )

        if not zero_proven:
            evidence = ReconciliationEvidence(
                lifecycle=lifecycle_identity,
                order_id=order_id,
                status_observed=True,
                raw_status=status_snapshot.raw_status,
                status_state=status_snapshot.state,
                status_matched_size=status_snapshot.matched_size,
                trade_observed=trade_observed,
                accounting_filled_size=accounting_fill,
                accounting_average_price=(
                    accounting_record.realized_average_price
                    if accounting_record is not None
                    else None
                ),
                wallet_observed=True,
                wallet_baseline_valid=bool(
                    wallet_delta_observation
                    and wallet_delta_observation.baseline_valid
                ),
                wallet_balance=wallet_balance,
                wallet_delta=(
                    wallet_delta_observation.delta
                    if wallet_delta_observation is not None
                    else None
                ),
                terminal_proven=True,
                baselineless_absolute_zero=baselineless_absolute_zero,
                attempts_used=attempts_used,
                notes=tuple(notes),
            )

            return self._result(
                outcome=ReconciliationOutcome.STILL_AMBIGUOUS,
                lifecycle_identity=lifecycle_identity,
                order_id=order_id,
                snapshot=status_snapshot,
                evidence=evidence,
                reason="terminal order but inventory attribution remains ambiguous",
            )

        terminal_zero_outcome = (
            ReconciliationOutcome.CANCELLED_ZERO_FILL
            if _terminal_cancel_state(status_snapshot.state)
            else ReconciliationOutcome.REJECTED_ZERO_FILL
        )

        # Resolve the accounting row only at the same exact terminal-zero commit.
        try:
            self.accounting.resolve(
                order_id,
                reason=terminal_zero_outcome.value,
            )
        except Exception as exc:
            notes.append(f"accounting_resolve_error:{type(exc).__name__}")

        accounting_record = self.accounting.get(order_id)

        evidence = ReconciliationEvidence(
            lifecycle=lifecycle_identity,
            order_id=order_id,
            status_observed=True,
            raw_status=status_snapshot.raw_status,
            status_state=status_snapshot.state,
            status_matched_size=status_snapshot.matched_size,
            trade_observed=trade_observed,
            accounting_filled_size=(
                accounting_record.confirmed_filled_size
                if accounting_record is not None
                else 0.0
            ),
            accounting_average_price=(
                accounting_record.realized_average_price
                if accounting_record is not None
                else None
            ),
            wallet_observed=True,
            wallet_baseline_valid=bool(
                wallet_delta_observation
                and wallet_delta_observation.baseline_valid
            ),
            wallet_balance=wallet_balance,
            wallet_delta=(
                wallet_delta_observation.delta
                if wallet_delta_observation is not None
                else None
            ),
            terminal_proven=True,
            baselineless_absolute_zero=baselineless_absolute_zero,
            attempts_used=attempts_used,
            notes=tuple(notes),
        )

        return self._result(
            outcome=terminal_zero_outcome,
            lifecycle_identity=lifecycle_identity,
            order_id=order_id,
            snapshot=status_snapshot,
            evidence=evidence,
            reason="exact terminal + zero matched + fresh same-token zero proof",
        )
