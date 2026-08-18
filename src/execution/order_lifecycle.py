"""
Order lifecycle state machine for the public portfolio edition.

This module owns local execution state, not trading strategy.  It translates
transport outcomes into explicit lifecycle ownership and applies exact-order
reconciliation evidence without guessing from elapsed time.

Core invariants:
- a raw-POST transport ambiguity remains owned until reconciliation resolves it
- a success-like submission without an exact order ID is still ambiguous
- cancel intent invalidates the prior working-state assumption immediately
- cancel uncertainty is not equivalent to cancellation
- terminal order status is not automatically equivalent to zero fill
- partial fills remain economic ownership even after the order itself is terminal
- stale lifecycle generations cannot overwrite a newer generation
- timeouts and watcher expiry never release unresolved ownership by themselves

Heavy reconciliation (wallet deltas, trade endpoints, venue-specific fill
evidence) belongs to execution/reconciliation.py and is consumed here through
typed ReconciliationResult objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
import threading
import time
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

from src.execution.clob_transport import (
    ClobTransport,
    ClobTransportError,
    PreSubmitContext,
    PreSubmitRejected,
    RawPostEnterObserver,
    TransportStage,
)
from src.execution.types import (
    CancellationOutcome,
    CancellationResult,
    FillSummary,
    LifecycleIdentity,
    OrderIntent,
    OrderLifecycleState,
    ReconciliationOutcome,
    ReconciliationResult,
    SubmissionOutcome,
    SubmissionResult,
    VenueOrderSnapshot,
    WorkingOrder,
)
from src.runtime.logging import runtime_print


class LifecycleEventType(str, Enum):
    CREATED = "CREATED"
    SUBMIT_STARTED = "SUBMIT_STARTED"
    RAW_POST_ENTERED = "RAW_POST_ENTERED"
    SUBMIT_CONFIRMED = "SUBMIT_CONFIRMED"
    SUBMIT_UNKNOWN = "SUBMIT_UNKNOWN"
    SUBMIT_FAILED = "SUBMIT_FAILED"
    STATUS_OBSERVED = "STATUS_OBSERVED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCEL_CONFIRMED = "CANCEL_CONFIRMED"
    CANCEL_UNKNOWN = "CANCEL_UNKNOWN"
    RECONCILIATION_APPLIED = "RECONCILIATION_APPLIED"
    TERMINAL = "TERMINAL"


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    lifecycle: LifecycleIdentity
    event_type: LifecycleEventType
    state: OrderLifecycleState
    order_id: Optional[str] = None
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        observed = float(self.timestamp)
        if not math.isfinite(observed) or observed <= 0.0:
            raise ValueError("timestamp must be positive")

        object.__setattr__(
            self,
            "order_id",
            str(self.order_id or "").strip() or None,
        )
        object.__setattr__(self, "reason", str(self.reason or ""))
        object.__setattr__(self, "timestamp", observed)


LifecycleListener = Callable[[LifecycleEvent], None]


@dataclass(frozen=True, slots=True)
class LifecycleSnapshot:
    """Read-only public view of one locally owned order lifecycle."""

    working_order: WorkingOrder
    raw_post_entered: bool
    created_at: float
    updated_at: float
    last_reason: str = ""
    last_submission: Optional[SubmissionResult] = None
    last_cancellation: Optional[CancellationResult] = None
    last_reconciliation: Optional[ReconciliationResult] = None

    @property
    def lifecycle(self) -> LifecycleIdentity:
        return self.working_order.intent.lifecycle

    @property
    def state(self) -> OrderLifecycleState:
        return self.working_order.state

    @property
    def order_id(self) -> Optional[str]:
        return self.working_order.order_id

    @property
    def owns_execution(self) -> bool:
        return self.working_order.owns_lifecycle


@dataclass(slots=True)
class _LifecycleEntry:
    working_order: WorkingOrder
    raw_post_entered: bool
    created_at: float
    updated_at: float
    last_reason: str = ""
    last_submission: Optional[SubmissionResult] = None
    last_cancellation: Optional[CancellationResult] = None
    last_reconciliation: Optional[ReconciliationResult] = None

    def snapshot(self) -> LifecycleSnapshot:
        return LifecycleSnapshot(
            working_order=self.working_order,
            raw_post_entered=bool(self.raw_post_entered),
            created_at=float(self.created_at),
            updated_at=float(self.updated_at),
            last_reason=str(self.last_reason or ""),
            last_submission=self.last_submission,
            last_cancellation=self.last_cancellation,
            last_reconciliation=self.last_reconciliation,
        )


class LifecycleConflict(RuntimeError):
    """Raised when a transition would violate exact lifecycle ownership."""


class StaleLifecycleGeneration(LifecycleConflict):
    """Raised when old evidence attempts to overwrite a newer generation."""


def _first_value(
    payload: Mapping[str, object],
    names: Iterable[str],
) -> object:
    for name in names:
        if name in payload and payload.get(name) not in (None, ""):
            return payload.get(name)
    return None


def _float_or_none(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _extract_order_id(payload: object) -> Optional[str]:
    if not isinstance(payload, Mapping):
        return None

    value = _first_value(
        payload,
        ("orderID", "order_id", "id", "orderId"),
    )
    return str(value or "").strip() or None


def _raw_status(payload: Mapping[str, object]) -> str:
    value = _first_value(
        payload,
        ("status", "state", "order_status", "orderStatus"),
    )
    return str(value or "").strip().upper()


def _requested_size(
    payload: Mapping[str, object],
    *,
    fallback: float,
) -> float:
    value = _first_value(
        payload,
        (
            "original_size",
            "originalSize",
            "size",
            "amount",
            "order_size",
            "orderSize",
        ),
    )
    parsed = _float_or_none(value)
    if parsed is None or parsed <= 0.0:
        return float(fallback)
    return parsed


def _matched_size(payload: Mapping[str, object]) -> float:
    value = _first_value(
        payload,
        (
            "size_matched",
            "sizeMatched",
            "matched_size",
            "matchedSize",
            "filled_size",
            "filledSize",
            "filled",
        ),
    )
    parsed = _float_or_none(value)
    return max(0.0, float(parsed or 0.0))


def _average_fill_price(
    payload: Mapping[str, object],
) -> Optional[float]:
    value = _first_value(
        payload,
        (
            "average_price",
            "averagePrice",
            "avg_price",
            "avgPrice",
            "price_avg",
        ),
    )
    parsed = _float_or_none(value)
    if parsed is None or not 0.0 < parsed <= 1.0:
        return None
    return parsed


def _limit_price(
    payload: Mapping[str, object],
    *,
    fallback: Optional[float],
) -> Optional[float]:
    value = _first_value(
        payload,
        ("price", "limit_price", "limitPrice"),
    )
    parsed = _float_or_none(value)

    if parsed is None:
        parsed = fallback

    if parsed is None or not 0.0 < float(parsed) <= 1.0:
        return None

    return float(parsed)


def _normalize_status_state(
    raw_status: str,
    *,
    matched_size: float,
    requested_size: float,
) -> OrderLifecycleState:
    status = str(raw_status or "").strip().upper()

    if requested_size > 0.0 and matched_size >= requested_size - 1e-9:
        return OrderLifecycleState.FILLED

    if status in {
        "MATCHED",
        "FILLED",
        "FULLY_FILLED",
        "EXECUTED",
        "COMPLETE",
        "COMPLETED",
    }:
        return OrderLifecycleState.FILLED

    if status in {
        "CANCELLED",
        "CANCELED",
    }:
        return OrderLifecycleState.CANCELLED

    if status in {
        "REJECTED",
        "INVALID",
    }:
        return OrderLifecycleState.REJECTED

    if status in {
        "FAILED",
        "ERROR",
    }:
        return OrderLifecycleState.FAILED

    if matched_size > 0.0:
        return OrderLifecycleState.PARTIALLY_FILLED

    if status in {
        "LIVE",
        "OPEN",
        "ACTIVE",
        "PENDING",
        "PLACED",
        "UNMATCHED",
        "WORKING",
    }:
        return OrderLifecycleState.WORKING

    # An exact OID returned by a successful status endpoint with an unfamiliar
    # non-terminal label remains a working/owned order, not terminal-zero proof.
    return OrderLifecycleState.WORKING


def normalize_venue_order_snapshot(
    payload: object,
    *,
    intent: OrderIntent,
    order_id: Optional[str] = None,
    observed_at: Optional[float] = None,
) -> Optional[VenueOrderSnapshot]:
    """Normalize a venue status/post response into the public execution contract."""

    if not isinstance(payload, Mapping):
        return None

    exact_order_id = (
        str(order_id or "").strip()
        or _extract_order_id(payload)
        or ""
    )
    if not exact_order_id:
        return None

    requested = _requested_size(payload, fallback=intent.size)
    matched = _matched_size(payload)
    raw_status = _raw_status(payload)

    state = _normalize_status_state(
        raw_status,
        matched_size=matched,
        requested_size=requested,
    )

    return VenueOrderSnapshot(
        order_id=exact_order_id,
        token_id=intent.token_id,
        order_side=intent.order_side,
        state=state,
        requested_size=requested,
        matched_size=matched,
        limit_price=_limit_price(
            payload,
            fallback=intent.price,
        ),
        average_fill_price=_average_fill_price(payload),
        observed_at=float(observed_at or time.time()),
        raw_status=raw_status,
    )


class OrderLifecycleService:
    """Thread-safe exact-owner lifecycle registry and transition service."""

    def __init__(
        self,
        transport: ClobTransport,
        *,
        listener: Optional[LifecycleListener] = None,
    ) -> None:
        self.transport = transport
        self.listener = listener

        self._gate = threading.RLock()
        self._entries: Dict[str, _LifecycleEntry] = {}
        self._lifecycle_by_order_id: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Registry / observability
    # ------------------------------------------------------------------

    @staticmethod
    def _key(lifecycle: LifecycleIdentity | str) -> str:
        if isinstance(lifecycle, LifecycleIdentity):
            return lifecycle.lifecycle_id
        return str(lifecycle or "").strip()

    def _emit(
        self,
        entry: _LifecycleEntry,
        event_type: LifecycleEventType,
        *,
        reason: str = "",
    ) -> None:
        listener = self.listener
        if listener is None:
            return

        event = LifecycleEvent(
            lifecycle=entry.working_order.intent.lifecycle,
            event_type=event_type,
            state=entry.working_order.state,
            order_id=entry.working_order.order_id,
            reason=reason,
        )

        try:
            listener(event)
        except Exception:
            # Observability must not alter lifecycle semantics.
            pass

    def create(self, intent: OrderIntent) -> LifecycleSnapshot:
        """Register one exact logical execution attempt before signing/submission."""

        key = intent.lifecycle.lifecycle_id
        now = time.time()

        with self._gate:
            existing = self._entries.get(key)

            if existing is not None:
                current_intent = existing.working_order.intent
                current = current_intent.lifecycle
                requested = intent.lifecycle

                if current.attempt_id != requested.attempt_id:
                    raise LifecycleConflict(
                        f"attempt mismatch for lifecycle {key}: "
                        f"{current.attempt_id} != {requested.attempt_id}"
                    )

                if current == requested:
                    if current_intent != intent:
                        raise LifecycleConflict(
                            f"lifecycle {key} identity is already bound to "
                            "a different order intent"
                        )
                    return existing.snapshot()

                if current.generation > requested.generation:
                    raise StaleLifecycleGeneration(
                        f"lifecycle {key} already at generation "
                        f"{current.generation}"
                    )

                if current.generation == requested.generation:
                    raise LifecycleConflict(
                        f"lifecycle {key} generation {current.generation} "
                        "is already bound"
                    )

                if existing.working_order.owns_lifecycle:
                    raise LifecycleConflict(
                        f"cannot replace owned lifecycle {key} "
                        f"state={existing.working_order.state.value}"
                    )

                if existing.raw_post_entered:
                    recon = existing.last_reconciliation
                    if (
                        recon is None
                        or recon.outcome
                        not in {
                            ReconciliationOutcome.CANCELLED_ZERO_FILL,
                            ReconciliationOutcome.REJECTED_ZERO_FILL,
                        }
                    ):
                        raise LifecycleConflict(
                            f"cannot replace raw-post lifecycle {key} "
                            "without terminal-zero reconciliation proof"
                        )

                old_order_id = existing.working_order.order_id
                if old_order_id:
                    self._lifecycle_by_order_id.pop(old_order_id, None)

            working = WorkingOrder(
                intent=intent,
                state=OrderLifecycleState.CREATED,
                order_id=None,
                filled_size=0.0,
                average_fill_price=None,
                last_observed_at=now,
            )
            entry = _LifecycleEntry(
                working_order=working,
                raw_post_entered=False,
                created_at=now,
                updated_at=now,
                last_reason="created",
            )
            self._entries[key] = entry
            self._emit(entry, LifecycleEventType.CREATED, reason="created")
            return entry.snapshot()

    def get(
        self,
        lifecycle: LifecycleIdentity | str,
    ) -> Optional[LifecycleSnapshot]:
        key = self._key(lifecycle)
        with self._gate:
            entry = self._entries.get(key)
            return entry.snapshot() if entry is not None else None

    def get_by_order_id(
        self,
        order_id: str,
    ) -> Optional[LifecycleSnapshot]:
        order_id = str(order_id or "").strip()
        if not order_id:
            return None

        with self._gate:
            key = self._lifecycle_by_order_id.get(order_id)
            entry = self._entries.get(key) if key else None
            return entry.snapshot() if entry is not None else None

    def snapshots(self) -> Tuple[LifecycleSnapshot, ...]:
        with self._gate:
            return tuple(entry.snapshot() for entry in self._entries.values())

    def owned_snapshots(self) -> Tuple[LifecycleSnapshot, ...]:
        return tuple(
            snapshot
            for snapshot in self.snapshots()
            if snapshot.owns_execution
        )

    def has_owner_for_token(self, token_id: str) -> bool:
        token_id = str(token_id or "")
        if not token_id:
            return False

        return any(
            snapshot.working_order.intent.token_id == token_id
            and snapshot.owns_execution
            for snapshot in self.snapshots()
        )

    def _require_entry(
        self,
        lifecycle: LifecycleIdentity | str,
    ) -> _LifecycleEntry:
        key = self._key(lifecycle)
        if not key:
            raise KeyError("lifecycle_id is required")

        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(f"unknown lifecycle: {key}")

        return entry

    def _assert_generation(
        self,
        entry: _LifecycleEntry,
        lifecycle: LifecycleIdentity,
    ) -> None:
        current = entry.working_order.intent.lifecycle

        if current.lifecycle_id != lifecycle.lifecycle_id:
            raise LifecycleConflict("lifecycle identity mismatch")

        if current.attempt_id != lifecycle.attempt_id:
            raise LifecycleConflict(
                f"attempt mismatch for lifecycle {current.lifecycle_id}"
            )

        if current.generation != lifecycle.generation:
            raise StaleLifecycleGeneration(
                f"lifecycle {current.lifecycle_id} generation "
                f"{lifecycle.generation} is stale; current={current.generation}"
            )

    def _set_working(
        self,
        entry: _LifecycleEntry,
        *,
        state: OrderLifecycleState,
        order_id: Optional[str] = None,
        filled_size: Optional[float] = None,
        average_fill_price: Optional[float] = None,
        reason: str = "",
    ) -> None:
        current = entry.working_order
        next_order_id = (
            str(order_id or "").strip()
            or current.order_id
        )

        next_filled = (
            current.filled_size
            if filled_size is None
            else max(current.filled_size, float(filled_size))
        )

        next_average = (
            average_fill_price
            if average_fill_price is not None
            else current.average_fill_price
        )

        updated = WorkingOrder(
            intent=current.intent,
            state=state,
            order_id=next_order_id,
            filled_size=next_filled,
            average_fill_price=next_average,
            last_observed_at=time.time(),
        )

        if (
            current.order_id
            and next_order_id
            and current.order_id != next_order_id
        ):
            raise LifecycleConflict(
                f"lifecycle {current.intent.lifecycle.lifecycle_id} "
                f"cannot change exact order ID "
                f"{current.order_id} -> {next_order_id}"
            )

        entry.working_order = updated
        entry.updated_at = time.time()
        entry.last_reason = str(reason or "")

        if next_order_id:
            owner = self._lifecycle_by_order_id.get(next_order_id)
            key = current.intent.lifecycle.lifecycle_id

            if owner and owner != key:
                raise LifecycleConflict(
                    f"order {next_order_id} already owned by lifecycle {owner}"
                )

            self._lifecycle_by_order_id[next_order_id] = key

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(
        self,
        intent: OrderIntent,
        signed_order: object,
        *,
        pre_submit_context: Optional[PreSubmitContext] = None,
        raw_post_enter_observer: Optional[RawPostEnterObserver] = None,
        post_kwargs: Optional[Mapping[str, object]] = None,
    ) -> SubmissionResult:
        """Post one already-created/signed order and commit exact ownership.

        The transport's final pre-submit validator runs after transport/SDK locks
        are acquired.  This service records the irreversible raw-post handoff via
        the transport observer before the venue call begins.
        """

        self.create(intent)
        context = pre_submit_context or PreSubmitContext(
            token_id=intent.token_id,
            market_id=intent.market_id,
            lifecycle_id=intent.lifecycle.lifecycle_id,
            attempt_id=intent.lifecycle.attempt_id,
        )

        with self._gate:
            entry = self._require_entry(intent.lifecycle)
            self._assert_generation(entry, intent.lifecycle)

            if (
                entry.last_submission is not None
                or entry.raw_post_entered
                or entry.working_order.state
                not in {
                    OrderLifecycleState.CREATED,
                    OrderLifecycleState.SIGNING,
                }
            ):
                raise LifecycleConflict(
                    "lifecycle generation already received a submission handoff "
                    f"(state={entry.working_order.state.value})"
                )

            self._set_working(
                entry,
                state=OrderLifecycleState.SUBMITTING,
                reason="submit started",
            )
            self._emit(
                entry,
                LifecycleEventType.SUBMIT_STARTED,
                reason="submit started",
            )

        def _raw_enter(ctx: PreSubmitContext) -> None:
            with self._gate:
                current = self._require_entry(intent.lifecycle)
                self._assert_generation(current, intent.lifecycle)
                current.raw_post_entered = True
                current.updated_at = time.time()
                current.last_reason = "raw post entered"
                self._emit(
                    current,
                    LifecycleEventType.RAW_POST_ENTERED,
                    reason="raw post entered",
                )

            if raw_post_enter_observer is not None:
                raw_post_enter_observer(ctx)

        kwargs = dict(post_kwargs or {})

        try:
            response = self.transport.post_order(
                signed_order,
                pre_submit_context=context,
                raw_post_enter_observer=_raw_enter,
                **kwargs,
            )

        except PreSubmitRejected as exc:
            result = SubmissionResult(
                outcome=SubmissionOutcome.FAILED_BEFORE_SUBMIT,
                intent=intent,
                order_id=None,
                venue_snapshot=None,
                post_call_entered=False,
                reason=f"pre-submit rejected: {exc.result.reason}",
            )

            with self._gate:
                entry = self._require_entry(intent.lifecycle)
                self._assert_generation(entry, intent.lifecycle)
                self._set_working(
                    entry,
                    state=OrderLifecycleState.REJECTED,
                    reason=result.reason,
                )
                entry.last_submission = result
                self._emit(
                    entry,
                    LifecycleEventType.SUBMIT_FAILED,
                    reason=result.reason,
                )

            return result

        except ClobTransportError as exc:
            ambiguous = bool(
                exc.stage is TransportStage.RAW_POST_ENTERED
                or (
                    self.get(intent.lifecycle) is not None
                    and bool(self.get(intent.lifecycle).raw_post_entered)
                )
            )

            if ambiguous:
                result = SubmissionResult(
                    outcome=SubmissionOutcome.UNKNOWN,
                    intent=intent,
                    order_id=None,
                    venue_snapshot=None,
                    post_call_entered=True,
                    reason=str(exc),
                )

                with self._gate:
                    entry = self._require_entry(intent.lifecycle)
                    self._assert_generation(entry, intent.lifecycle)
                    self._set_working(
                        entry,
                        state=OrderLifecycleState.SUBMISSION_UNKNOWN,
                        reason=result.reason,
                    )
                    entry.last_submission = result
                    self._emit(
                        entry,
                        LifecycleEventType.SUBMIT_UNKNOWN,
                        reason=result.reason,
                    )

                return result

            result = SubmissionResult(
                outcome=SubmissionOutcome.FAILED_BEFORE_SUBMIT,
                intent=intent,
                order_id=None,
                venue_snapshot=None,
                post_call_entered=False,
                reason=str(exc),
            )

            with self._gate:
                entry = self._require_entry(intent.lifecycle)
                self._assert_generation(entry, intent.lifecycle)
                self._set_working(
                    entry,
                    state=OrderLifecycleState.FAILED,
                    reason=result.reason,
                )
                entry.last_submission = result
                self._emit(
                    entry,
                    LifecycleEventType.SUBMIT_FAILED,
                    reason=result.reason,
                )

            return result

        except Exception as exc:
            # Unknown exceptions are classified from the irreversible local handoff
            # rather than guessed from their message text.
            with self._gate:
                entry = self._require_entry(intent.lifecycle)
                self._assert_generation(entry, intent.lifecycle)
                raw_entered = bool(entry.raw_post_entered)

            outcome = (
                SubmissionOutcome.UNKNOWN
                if raw_entered
                else SubmissionOutcome.FAILED_BEFORE_SUBMIT
            )
            state = (
                OrderLifecycleState.SUBMISSION_UNKNOWN
                if raw_entered
                else OrderLifecycleState.FAILED
            )

            result = SubmissionResult(
                outcome=outcome,
                intent=intent,
                order_id=None,
                venue_snapshot=None,
                post_call_entered=raw_entered,
                reason=f"{type(exc).__name__}: {exc}",
            )

            with self._gate:
                entry = self._require_entry(intent.lifecycle)
                self._set_working(
                    entry,
                    state=state,
                    reason=result.reason,
                )
                entry.last_submission = result
                self._emit(
                    entry,
                    (
                        LifecycleEventType.SUBMIT_UNKNOWN
                        if raw_entered
                        else LifecycleEventType.SUBMIT_FAILED
                    ),
                    reason=result.reason,
                )

            return result

        order_id = _extract_order_id(response)

        # A success-like response without an exact OID cannot prove a live order.
        # The venue write has definitely been entered, so preserve unknown ownership.
        if not order_id:
            result = SubmissionResult(
                outcome=SubmissionOutcome.UNKNOWN,
                intent=intent,
                order_id=None,
                venue_snapshot=None,
                post_call_entered=True,
                reason="submission response did not contain an exact order ID",
            )

            with self._gate:
                entry = self._require_entry(intent.lifecycle)
                self._assert_generation(entry, intent.lifecycle)
                self._set_working(
                    entry,
                    state=OrderLifecycleState.SUBMISSION_UNKNOWN,
                    reason=result.reason,
                )
                entry.last_submission = result
                self._emit(
                    entry,
                    LifecycleEventType.SUBMIT_UNKNOWN,
                    reason=result.reason,
                )

            return result

        venue_snapshot = normalize_venue_order_snapshot(
            response,
            intent=intent,
            order_id=order_id,
        )

        state = (
            venue_snapshot.state
            if venue_snapshot is not None
            else OrderLifecycleState.WORKING
        )
        filled_size = (
            venue_snapshot.matched_size
            if venue_snapshot is not None
            else 0.0
        )
        average = (
            venue_snapshot.average_fill_price
            if venue_snapshot is not None
            else None
        )

        if state is OrderLifecycleState.FILLED:
            outcome = SubmissionOutcome.CONFIRMED_FILLED
        else:
            # An exact returned OID is the local working-order owner even when the
            # venue response uses an unfamiliar/non-terminal status label.
            outcome = SubmissionOutcome.CONFIRMED_WORKING
            if state in {
                OrderLifecycleState.REJECTED,
                OrderLifecycleState.FAILED,
                OrderLifecycleState.CANCELLED,
            }:
                # A terminal status returned atomically with the post is still an
                # exact result; preserve that state and classify the submit as
                # rejected rather than inventing a working order.
                outcome = SubmissionOutcome.REJECTED

        result = SubmissionResult(
            outcome=outcome,
            intent=intent,
            order_id=order_id,
            venue_snapshot=venue_snapshot,
            post_call_entered=True,
            reason="exact order ID returned by venue",
        )

        with self._gate:
            entry = self._require_entry(intent.lifecycle)
            self._assert_generation(entry, intent.lifecycle)
            self._set_working(
                entry,
                state=state,
                order_id=order_id,
                filled_size=filled_size,
                average_fill_price=average,
                reason=result.reason,
            )
            entry.last_submission = result
            self._emit(
                entry,
                (
                    LifecycleEventType.TERMINAL
                    if state.is_terminal
                    else LifecycleEventType.SUBMIT_CONFIRMED
                ),
                reason=result.reason,
            )

        return result

    # ------------------------------------------------------------------
    # Exact status observations
    # ------------------------------------------------------------------

    def observe_status(
        self,
        lifecycle: LifecycleIdentity,
        payload: object,
    ) -> Optional[VenueOrderSnapshot]:
        """Apply one exact-OID status result if its lifecycle generation is current."""

        with self._gate:
            entry = self._require_entry(lifecycle)
            self._assert_generation(entry, lifecycle)

            order_id = entry.working_order.order_id
            if not order_id:
                return None

            intent = entry.working_order.intent

        payload_order_id = _extract_order_id(payload)
        if payload_order_id and payload_order_id != order_id:
            raise LifecycleConflict(
                "status result order ID does not match lifecycle owner"
            )

        snapshot = normalize_venue_order_snapshot(
            payload,
            intent=intent,
            order_id=order_id,
        )
        if snapshot is None:
            return None

        with self._gate:
            entry = self._require_entry(lifecycle)
            self._assert_generation(entry, lifecycle)
            current = entry.working_order

            if current.order_id != snapshot.order_id:
                raise LifecycleConflict(
                    "status result order ID does not match lifecycle owner"
                )

            observed_filled = float(snapshot.matched_size)
            current_filled = float(current.filled_size)
            stale_quantity = observed_filled + 1e-9 < current_filled
            next_filled = max(current_filled, observed_filled)

            next_state = snapshot.state
            if stale_quantity:
                next_state = current.state
            elif current.state is OrderLifecycleState.FILLED:
                next_state = OrderLifecycleState.FILLED
            elif current.state.is_terminal and not snapshot.state.is_terminal:
                next_state = current.state
            elif (
                current.state is OrderLifecycleState.PARTIALLY_FILLED
                and snapshot.state is OrderLifecycleState.WORKING
                and next_filled > 0.0
            ):
                next_state = OrderLifecycleState.PARTIALLY_FILLED

            next_average = current.average_fill_price
            if (
                not stale_quantity
                and snapshot.average_fill_price is not None
            ):
                next_average = snapshot.average_fill_price

            self._set_working(
                entry,
                state=next_state,
                order_id=snapshot.order_id,
                filled_size=next_filled,
                average_fill_price=next_average,
                reason=f"status:{snapshot.raw_status or snapshot.state.value}",
            )
            self._emit(
                entry,
                (
                    LifecycleEventType.TERMINAL
                    if next_state.is_terminal
                    else LifecycleEventType.STATUS_OBSERVED
                ),
                reason=entry.last_reason,
            )

        return snapshot

    def refresh_status(
        self,
        lifecycle: LifecycleIdentity,
    ) -> Optional[VenueOrderSnapshot]:
        """Fetch and apply the current exact-order status through ClobTransport."""

        with self._gate:
            entry = self._require_entry(lifecycle)
            self._assert_generation(entry, lifecycle)
            order_id = entry.working_order.order_id

        if not order_id:
            return None

        payload = self.transport.get_order(order_id)
        if not payload:
            return None

        return self.observe_status(lifecycle, payload)

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel(
        self,
        lifecycle: LifecycleIdentity,
    ) -> CancellationResult:
        """Request cancellation without converting ambiguity into terminal state."""

        with self._gate:
            entry = self._require_entry(lifecycle)
            self._assert_generation(entry, lifecycle)

            order_id = entry.working_order.order_id
            state = entry.working_order.state

            if state.is_terminal:
                result = CancellationResult(
                    outcome=CancellationOutcome.ALREADY_TERMINAL,
                    order_id=order_id or "terminal-without-order-id",
                    lifecycle=lifecycle,
                    reason=f"already terminal: {state.value}",
                )
                entry.last_cancellation = result
                return result

            if not order_id:
                # No exact OID means a raw-post ambiguity cannot safely be cancelled
                # by pretending the order does not exist.
                raise LifecycleConflict(
                    f"lifecycle {lifecycle.lifecycle_id} has no exact order ID; "
                    "submission reconciliation must locate/resolve it first"
                )

            self._set_working(
                entry,
                state=OrderLifecycleState.CANCEL_REQUESTED,
                reason="cancel requested",
            )
            self._emit(
                entry,
                LifecycleEventType.CANCEL_REQUESTED,
                reason="cancel requested",
            )

        try:
            response = self.transport.cancel_order(order_id)

        except ClobTransportError as exc:
            result = CancellationResult(
                outcome=CancellationOutcome.UNKNOWN,
                order_id=order_id,
                lifecycle=lifecycle,
                reason=str(exc),
            )

            with self._gate:
                entry = self._require_entry(lifecycle)
                self._assert_generation(entry, lifecycle)
                self._set_working(
                    entry,
                    state=OrderLifecycleState.CANCEL_UNKNOWN,
                    reason=result.reason,
                )
                entry.last_cancellation = result
                self._emit(
                    entry,
                    LifecycleEventType.CANCEL_UNKNOWN,
                    reason=result.reason,
                )

            return result

        except Exception as exc:
            result = CancellationResult(
                outcome=CancellationOutcome.UNKNOWN,
                order_id=order_id,
                lifecycle=lifecycle,
                reason=f"{type(exc).__name__}: {exc}",
            )

            with self._gate:
                entry = self._require_entry(lifecycle)
                self._set_working(
                    entry,
                    state=OrderLifecycleState.CANCEL_UNKNOWN,
                    reason=result.reason,
                )
                entry.last_cancellation = result
                self._emit(
                    entry,
                    LifecycleEventType.CANCEL_UNKNOWN,
                    reason=result.reason,
                )

            return result

        # A transport-level success is not itself proof that cancellation won.
        # Classify only explicit terminal status or an explicit cancel ack.
        status_snapshot = None
        explicit_cancel_ack = False

        if isinstance(response, Mapping):
            payload_order_id = _extract_order_id(response)
            if payload_order_id and payload_order_id != order_id:
                raise LifecycleConflict(
                    "cancel response order ID does not match lifecycle owner"
                )

            status_snapshot = normalize_venue_order_snapshot(
                response,
                intent=self.get(lifecycle).working_order.intent,
                order_id=order_id,
            )

            cancelled_values = response.get(
                "canceled",
                response.get("cancelled"),
            )
            if isinstance(cancelled_values, (list, tuple, set)):
                explicit_cancel_ack = order_id in {
                    str(value) for value in cancelled_values
                }
            elif cancelled_values is True:
                explicit_cancel_ack = True

        elif response is True:
            explicit_cancel_ack = True

        terminal_state = (
            status_snapshot.state
            if status_snapshot is not None
            else None
        )

        confirmed_cancelled = bool(
            explicit_cancel_ack
            or terminal_state is OrderLifecycleState.CANCELLED
        )
        already_terminal = bool(
            terminal_state in {
                OrderLifecycleState.FILLED,
                OrderLifecycleState.REJECTED,
                OrderLifecycleState.FAILED,
                OrderLifecycleState.CLOSED,
            }
        )

        if confirmed_cancelled:
            outcome = CancellationOutcome.CONFIRMED_CANCELLED
            next_state = OrderLifecycleState.CANCELLED
            reason = "explicit cancellation acknowledgement"
            event_type = LifecycleEventType.CANCEL_CONFIRMED
        elif already_terminal:
            outcome = CancellationOutcome.ALREADY_TERMINAL
            next_state = terminal_state
            reason = f"venue reported terminal state {terminal_state.value}"
            event_type = LifecycleEventType.TERMINAL
        else:
            outcome = CancellationOutcome.UNKNOWN
            next_state = OrderLifecycleState.CANCEL_UNKNOWN
            reason = "cancel request returned without terminal cancellation proof"
            event_type = LifecycleEventType.CANCEL_UNKNOWN

        result = CancellationResult(
            outcome=outcome,
            order_id=order_id,
            lifecycle=lifecycle,
            venue_snapshot=status_snapshot,
            reason=reason,
        )

        with self._gate:
            entry = self._require_entry(lifecycle)
            self._assert_generation(entry, lifecycle)

            filled = (
                max(
                    entry.working_order.filled_size,
                    status_snapshot.matched_size,
                )
                if status_snapshot is not None
                else entry.working_order.filled_size
            )
            average = entry.working_order.average_fill_price
            if (
                status_snapshot is not None
                and status_snapshot.matched_size
                + 1e-9
                >= entry.working_order.filled_size
                and status_snapshot.average_fill_price is not None
            ):
                average = status_snapshot.average_fill_price

            self._set_working(
                entry,
                state=next_state,
                order_id=order_id,
                filled_size=filled,
                average_fill_price=average,
                reason=result.reason,
            )
            entry.last_cancellation = result
            self._emit(
                entry,
                event_type,
                reason=result.reason,
            )

        return result

    # ------------------------------------------------------------------
    # Reconciliation handoff
    # ------------------------------------------------------------------

    def apply_reconciliation(
        self,
        result: ReconciliationResult,
    ) -> LifecycleSnapshot:
        """Consume authoritative reconciliation evidence from another service.

        This is the only generic path that can turn unresolved no-OID/unknown
        ownership into confirmed terminal-zero state.
        """

        lifecycle = result.lifecycle

        with self._gate:
            entry = self._require_entry(lifecycle)
            self._assert_generation(entry, lifecycle)

            current = entry.working_order

            if (
                current.order_id
                and result.order_id
                and current.order_id != result.order_id
            ):
                raise LifecycleConflict(
                    f"reconciliation order ID {result.order_id} "
                    f"does not match owner {current.order_id}"
                )

            order_id = result.order_id or current.order_id
            snapshot = result.snapshot
            fill_summary = result.fill_summary

            filled_size = current.filled_size
            average = current.average_fill_price

            if snapshot is not None:
                if order_id and snapshot.order_id != order_id:
                    raise LifecycleConflict(
                        "reconciliation snapshot order ID mismatch"
                    )
                order_id = order_id or snapshot.order_id
                filled_size = max(filled_size, snapshot.matched_size)
                if snapshot.average_fill_price is not None:
                    average = snapshot.average_fill_price

            if fill_summary is not None:
                if order_id and fill_summary.order_id != order_id:
                    raise LifecycleConflict(
                        "fill summary order ID mismatch"
                    )
                order_id = order_id or fill_summary.order_id
                filled_size = max(
                    filled_size,
                    fill_summary.filled_size,
                )
                if fill_summary.average_price is not None:
                    average = fill_summary.average_price

            if result.outcome is ReconciliationOutcome.WORKING:
                if not order_id:
                    raise LifecycleConflict(
                        "WORKING reconciliation requires an exact order ID"
                    )
                next_state = (
                    OrderLifecycleState.PARTIALLY_FILLED
                    if filled_size > 0.0
                    else OrderLifecycleState.WORKING
                )

            elif result.outcome is ReconciliationOutcome.PARTIALLY_FILLED:
                if not order_id:
                    raise LifecycleConflict(
                        "PARTIALLY_FILLED reconciliation requires order ID"
                    )
                if filled_size <= 0.0:
                    raise LifecycleConflict(
                        "PARTIALLY_FILLED reconciliation requires fill quantity"
                    )
                if (
                    snapshot is not None
                    and snapshot.state
                    in {
                        OrderLifecycleState.CANCELLED,
                        OrderLifecycleState.REJECTED,
                        OrderLifecycleState.FAILED,
                    }
                ):
                    next_state = snapshot.state
                elif current.state in {
                    OrderLifecycleState.CANCELLED,
                    OrderLifecycleState.REJECTED,
                    OrderLifecycleState.FAILED,
                }:
                    next_state = current.state
                else:
                    next_state = OrderLifecycleState.PARTIALLY_FILLED

            elif result.outcome is ReconciliationOutcome.FILLED:
                if not order_id:
                    raise LifecycleConflict(
                        "FILLED reconciliation requires order ID"
                    )
                if filled_size <= 0.0:
                    raise LifecycleConflict(
                        "FILLED reconciliation requires fill quantity"
                    )
                next_state = OrderLifecycleState.FILLED

            elif result.outcome in {
                ReconciliationOutcome.CANCELLED_ZERO_FILL,
                ReconciliationOutcome.REJECTED_ZERO_FILL,
            }:
                # Explicit terminal-zero is stronger than raw labels.  Never infer it
                # merely because a watcher elapsed or a local cancel call returned.
                if filled_size > 1e-9:
                    raise LifecycleConflict(
                        "terminal-zero reconciliation conflicts with known fill"
                    )
                next_state = (
                    OrderLifecycleState.CANCELLED
                    if result.outcome
                    is ReconciliationOutcome.CANCELLED_ZERO_FILL
                    else OrderLifecycleState.REJECTED
                )

            elif result.outcome is ReconciliationOutcome.STILL_AMBIGUOUS:
                if current.state in {
                    OrderLifecycleState.CANCEL_REQUESTED,
                    OrderLifecycleState.CANCEL_UNKNOWN,
                }:
                    next_state = OrderLifecycleState.CANCEL_UNKNOWN
                else:
                    next_state = OrderLifecycleState.RECONCILING

            else:  # pragma: no cover - defensive future enum extension
                raise ValueError(
                    f"unsupported reconciliation outcome: {result.outcome}"
                )

            self._set_working(
                entry,
                state=next_state,
                order_id=order_id,
                filled_size=filled_size,
                average_fill_price=average,
                reason=result.reason or result.outcome.value,
            )
            entry.last_reconciliation = result

            self._emit(
                entry,
                LifecycleEventType.RECONCILIATION_APPLIED,
                reason=entry.last_reason,
            )

            if next_state.is_terminal:
                self._emit(
                    entry,
                    LifecycleEventType.TERMINAL,
                    reason=entry.last_reason,
                )

            return entry.snapshot()

    def mark_reconciling(
        self,
        lifecycle: LifecycleIdentity,
        *,
        reason: str = "reconciliation started",
    ) -> LifecycleSnapshot:
        """Move unresolved ownership into RECONCILING without releasing it."""

        with self._gate:
            entry = self._require_entry(lifecycle)
            self._assert_generation(entry, lifecycle)

            if entry.working_order.state.is_terminal:
                return entry.snapshot()

            self._set_working(
                entry,
                state=OrderLifecycleState.RECONCILING,
                reason=reason,
            )
            entry.updated_at = time.time()
            return entry.snapshot()

    # ------------------------------------------------------------------
    # Generation handoff / cleanup
    # ------------------------------------------------------------------

    def supersede(
        self,
        current: LifecycleIdentity,
        replacement_intent: OrderIntent,
    ) -> LifecycleSnapshot:
        """Install a newer generation only after the current owner is truly clear."""

        if (
            current.lifecycle_id
            != replacement_intent.lifecycle.lifecycle_id
        ):
            raise LifecycleConflict(
                "replacement must keep the same lifecycle_id"
            )

        if (
            current.attempt_id
            != replacement_intent.lifecycle.attempt_id
        ):
            raise LifecycleConflict(
                "replacement must keep the same attempt_id"
            )

        if (
            replacement_intent.lifecycle.generation
            <= current.generation
        ):
            raise StaleLifecycleGeneration(
                "replacement generation must increase"
            )

        with self._gate:
            entry = self._require_entry(current)
            self._assert_generation(entry, current)

            if entry.working_order.owns_lifecycle:
                raise LifecycleConflict(
                    f"cannot supersede owned state "
                    f"{entry.working_order.state.value}"
                )

            if entry.raw_post_entered:
                recon = entry.last_reconciliation
                if (
                    recon is None
                    or recon.outcome
                    not in {
                        ReconciliationOutcome.CANCELLED_ZERO_FILL,
                        ReconciliationOutcome.REJECTED_ZERO_FILL,
                    }
                ):
                    raise LifecycleConflict(
                        "cannot supersede raw-post lifecycle without "
                        "terminal-zero reconciliation proof"
                    )

            old_order_id = entry.working_order.order_id
            if old_order_id:
                self._lifecycle_by_order_id.pop(old_order_id, None)

            now = time.time()
            entry.working_order = WorkingOrder(
                intent=replacement_intent,
                state=OrderLifecycleState.CREATED,
                order_id=None,
                filled_size=0.0,
                average_fill_price=None,
                last_observed_at=now,
            )
            entry.raw_post_entered = False
            entry.created_at = now
            entry.updated_at = now
            entry.last_reason = "superseded by newer generation"
            entry.last_submission = None
            entry.last_cancellation = None
            entry.last_reconciliation = None

            self._emit(
                entry,
                LifecycleEventType.CREATED,
                reason=entry.last_reason,
            )
            return entry.snapshot()

    def release_terminal_zero(
        self,
        lifecycle: LifecycleIdentity,
    ) -> bool:
        """Remove only a terminal lifecycle with no known economic fill.

        This is deliberately narrow.  FILLED and partially-filled/cancelled rows
        remain available to the later position/fill-accounting layers.
        """

        with self._gate:
            entry = self._require_entry(lifecycle)
            self._assert_generation(entry, lifecycle)

            working = entry.working_order

            if not working.state.is_terminal:
                return False

            if working.filled_size > 1e-9:
                return False

            # A plain local cancel is not enough.  For CANCELLED/REJECTED rows that
            # ever crossed raw POST, require explicit terminal-zero reconciliation.
            if entry.raw_post_entered:
                recon = entry.last_reconciliation
                if (
                    recon is None
                    or recon.outcome
                    not in {
                        ReconciliationOutcome.CANCELLED_ZERO_FILL,
                        ReconciliationOutcome.REJECTED_ZERO_FILL,
                    }
                ):
                    return False

            order_id = working.order_id
            if order_id:
                self._lifecycle_by_order_id.pop(order_id, None)

            self._entries.pop(lifecycle.lifecycle_id, None)
            return True

    def prune_released(
        self,
        *,
        keep_last_seconds: float = 300.0,
        now: Optional[float] = None,
    ) -> int:
        """Prune old non-owning terminal rows without touching unresolved owners."""

        current_time = float(now or time.time())
        minimum_age = max(0.0, float(keep_last_seconds))
        removed = 0

        with self._gate:
            for key, entry in list(self._entries.items()):
                working = entry.working_order

                if working.owns_lifecycle:
                    continue

                if not working.state.is_terminal:
                    continue

                if current_time - entry.updated_at < minimum_age:
                    continue

                # Preserve raw-post terminal rows until explicit zero reconciliation
                # has proven that no inventory belongs to them.
                if entry.raw_post_entered:
                    recon = entry.last_reconciliation
                    if (
                        recon is None
                        or recon.outcome
                        not in {
                            ReconciliationOutcome.CANCELLED_ZERO_FILL,
                            ReconciliationOutcome.REJECTED_ZERO_FILL,
                        }
                    ):
                        continue

                if working.order_id:
                    self._lifecycle_by_order_id.pop(
                        working.order_id,
                        None,
                    )

                self._entries.pop(key, None)
                removed += 1

        return removed
