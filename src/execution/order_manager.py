"""
High-level execution façade for the public portfolio edition.

The private production OrderManager grew into a large mixed-responsibility
component containing strategy, risk, venue transport, fill accounting, wallet
reconciliation, cooldowns, and position state.  The portfolio edition keeps the
useful execution orchestration but makes the ownership boundaries explicit.

This manager coordinates already-separated services:

    OrderIntent
        -> optional wallet baseline capture
        -> local order construction/signing
        -> OrderLifecycleService submission
        -> FillAccounting registration
        -> status/cancel/reconciliation orchestration

It deliberately does NOT decide:
- whether a market should be traded
- whether an entry is high quality
- position size
- completion/hedge strategy
- asset-specific policy
- historical GOOD/BAD setup families

State ownership remains in the dedicated lifecycle, accounting, transport, and
reconciliation services rather than being duplicated in this façade.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import inspect
import math
import threading
import time
from typing import Awaitable, Callable, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple, Union

from src.execution.clob_transport import (
    ClobTransport,
    PreSubmitContext,
    PrewarmResult,
    RawPostEnterObserver,
)
from src.execution.fill_accounting import (
    FillAccounting,
    FillEvidenceSource,
    FillObservation,
    OrderFillRecord,
    PriceEvidence,
    PriceEvidenceKind,
    WalletBaseline,
)
from src.execution.order_lifecycle import (
    LifecycleSnapshot,
    OrderLifecycleService,
)
from src.execution.reconciliation import ReconciliationService
from src.execution.types import (
    CancellationResult,
    LifecycleIdentity,
    OrderIntent,
    ReconciliationResult,
    SubmissionOutcome,
    SubmissionResult,
    VenueOrderSnapshot,
)
from src.market.types import MarketDefinition
from src.runtime.logging import runtime_print


MaybeAwaitable = Union[object, Awaitable[object]]


class OrderArgumentFactory(Protocol):
    """Build venue SDK order arguments from an immutable OrderIntent."""

    def __call__(self, intent: OrderIntent) -> object:
        ...


class WalletBalanceReader(Protocol):
    """Return the current conditional-token balance for one token."""

    def __call__(self, token_id: str) -> Union[float, Awaitable[float]]:
        ...


@dataclass(frozen=True, slots=True)
class ExecutionManagerPolicy:
    """Operational façade behavior only.

    `capture_wallet_baseline_by_default` is false because baseline acquisition can
    require an extra venue/wallet read.  Consumers opt in when they need wallet
    attribution for later ambiguous/cancel reconciliation.
    """

    capture_wallet_baseline_by_default: bool = False
    reconcile_ambiguous_submission: bool = True
    reconcile_after_cancel: bool = True
    prewarm_market_metadata: bool = True

    resolved_accounting_keep_seconds: float = 15 * 60
    terminal_lifecycle_keep_seconds: float = 5 * 60


@dataclass(frozen=True, slots=True)
class ManagedSubmission:
    """Result returned by the high-level submission façade."""

    submission: SubmissionResult
    lifecycle: Optional[LifecycleSnapshot]
    fill_record: Optional[OrderFillRecord]
    wallet_baseline: Optional[WalletBaseline]
    reconciliation: Optional[ReconciliationResult] = None

    @property
    def order_id(self) -> Optional[str]:
        return self.submission.order_id

    @property
    def confirmed(self) -> bool:
        return self.submission.confirmed

    @property
    def ambiguous(self) -> bool:
        return self.submission.ambiguous


@dataclass(frozen=True, slots=True)
class ManagedCancellation:
    cancellation: CancellationResult
    lifecycle: Optional[LifecycleSnapshot]
    reconciliation: Optional[ReconciliationResult]


@dataclass(frozen=True, slots=True)
class ExecutionManagerStats:
    active_lifecycles: int
    owned_lifecycles: int
    accounting_records: int
    owned_background_tasks: int
    active_reconciliations: int
    wallet_baselines: int


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


class ExecutionManager:
    """Thin asynchronous façade over the execution subsystem."""

    def __init__(
        self,
        *,
        transport: ClobTransport,
        lifecycle: OrderLifecycleService,
        accounting: FillAccounting,
        reconciliation: ReconciliationService,
        order_argument_factory: Optional[OrderArgumentFactory] = None,
        wallet_balance_reader: Optional[WalletBalanceReader] = None,
        policy: ExecutionManagerPolicy = ExecutionManagerPolicy(),
    ) -> None:
        self.transport = transport
        self.lifecycle = lifecycle
        self.accounting = accounting
        self.reconciliation = reconciliation

        self.order_argument_factory = order_argument_factory
        self.wallet_balance_reader = wallet_balance_reader
        self.policy = policy

        self._baseline_gate = threading.RLock()
        self._wallet_baselines: Dict[str, WalletBaseline] = {}

        # Strong ownership is intentional: asyncio's loop does not make a Task a
        # durable application lifecycle owner merely because create_task() returned.
        self._owned_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Lifecycle/task ownership
    # ------------------------------------------------------------------

    @staticmethod
    def _lifecycle_key(identity: LifecycleIdentity) -> str:
        return (
            f"{identity.lifecycle_id}:"
            f"{identity.attempt_id}:"
            f"{identity.generation}"
        )

    def launch_owned_task(
        self,
        awaitable: Awaitable,
        *,
        label: str = "execution-background",
    ) -> asyncio.Task:
        """Create and strongly own background execution work until completion."""

        task = asyncio.create_task(awaitable, name=str(label or "execution-background"))
        self._owned_tasks.add(task)

        def _done(done_task: asyncio.Task) -> None:
            self._owned_tasks.discard(done_task)

            if done_task.cancelled():
                return

            try:
                exc = done_task.exception()
            except asyncio.CancelledError:
                return
            except Exception as callback_exc:
                runtime_print(
                    f"[execution] task callback error {label}: "
                    f"{type(callback_exc).__name__}: {callback_exc}"
                )
                return

            if exc is not None:
                runtime_print(
                    f"[execution] background task failed {label}: "
                    f"{type(exc).__name__}: {exc}"
                )

        task.add_done_callback(_done)
        return task

    async def close(self) -> None:
        """Cancel and drain manager-owned background work."""

        tasks = tuple(self._owned_tasks)

        for task in tasks:
            if not task.done():
                task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._owned_tasks.clear()

    # ------------------------------------------------------------------
    # Market metadata / signing hot path
    # ------------------------------------------------------------------

    async def prepare_markets(
        self,
        markets: Sequence[MarketDefinition],
        *,
        prewarm: Optional[bool] = None,
    ) -> PrewarmResult:
        """Register discovery metadata and optionally prewarm local signing state.

        Market identity metadata is operational venue data, not trading policy.
        """

        tokens: list[str] = []

        for market in markets:
            for token_id in (market.yes_token, market.no_token):
                self.transport.register_market_metadata(
                    token_id,
                    tick_size=market.tick_size,
                    neg_risk=market.neg_risk,
                    condition_id=market.condition_id,
                )
                tokens.append(token_id)

        unique_tokens = tuple(dict.fromkeys(tokens))
        do_prewarm = (
            self.policy.prewarm_market_metadata
            if prewarm is None
            else bool(prewarm)
        )

        if not do_prewarm:
            return PrewarmResult(
                tokens=len(unique_tokens),
                ready=0,
                resolved=0,
                failed=0,
                ready_tokens=(),
                failed_tokens=(),
                version=None,
            )

        return await asyncio.to_thread(
            self.transport.prewarm_create_hotpath,
            unique_tokens,
        )

    # ------------------------------------------------------------------
    # Wallet baseline ownership
    # ------------------------------------------------------------------

    async def capture_wallet_baseline(
        self,
        intent: OrderIntent,
    ) -> Optional[WalletBaseline]:
        """Capture an explicit pre-order token balance.

        Failure produces no baseline.  It never produces a synthetic zero baseline.
        """

        reader = self.wallet_balance_reader
        if reader is None:
            return None

        try:
            value = await _maybe_await(reader(intent.token_id))
            balance = float(value)

            if not math.isfinite(balance) or balance < 0.0:
                raise ValueError("wallet reader returned invalid balance")

            baseline = WalletBaseline(
                token_id=intent.token_id,
                balance=balance,
                observed_at=time.time(),
                valid=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            runtime_print(
                f"[execution] wallet baseline unavailable "
                f"{intent.token_id[:8]}: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

        with self._baseline_gate:
            self._wallet_baselines[
                self._lifecycle_key(intent.lifecycle)
            ] = baseline

        return baseline

    def wallet_baseline(
        self,
        lifecycle_identity: LifecycleIdentity,
    ) -> Optional[WalletBaseline]:
        with self._baseline_gate:
            return self._wallet_baselines.get(
                self._lifecycle_key(lifecycle_identity)
            )

    def _drop_wallet_baseline(
        self,
        lifecycle_identity: LifecycleIdentity,
    ) -> None:
        with self._baseline_gate:
            self._wallet_baselines.pop(
                self._lifecycle_key(lifecycle_identity),
                None,
            )

    # ------------------------------------------------------------------
    # Order construction
    # ------------------------------------------------------------------

    def _build_order_arguments(
        self,
        intent: OrderIntent,
        provided: Optional[object],
    ) -> object:
        if provided is not None:
            return provided

        factory = self.order_argument_factory
        if factory is None:
            raise RuntimeError(
                "no order arguments provided and no OrderArgumentFactory configured"
            )

        return factory(intent)

    async def create_signed_order(
        self,
        intent: OrderIntent,
        *,
        order_args: Optional[object] = None,
        create_options: Optional[object] = None,
    ) -> object:
        """Construct/sign one order without posting it."""

        args = self._build_order_arguments(intent, order_args)

        return await asyncio.to_thread(
            self.transport.create_order,
            args,
            create_options,
        )

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def _register_submission_accounting(
        self,
        submission: SubmissionResult,
    ) -> Optional[OrderFillRecord]:
        order_id = submission.order_id
        if not order_id:
            return None

        intent = submission.intent

        record = self.accounting.register_order(
            order_id=order_id,
            token_id=intent.token_id,
            lifecycle=intent.lifecycle,
            order_side=intent.order_side,
            requested_size=intent.size,
            submitted_limit_price=intent.price,
            created_at=intent.created_at,
        )

        snapshot = submission.venue_snapshot
        if snapshot is None:
            return record

        price_evidence = None
        if snapshot.average_fill_price is not None:
            price_evidence = PriceEvidence(
                price=snapshot.average_fill_price,
                kind=PriceEvidenceKind.EXPLICIT_AVERAGE,
                source="submission response average",
                represented_size=snapshot.matched_size,
            )

        observation = FillObservation(
            order_id=order_id,
            token_id=intent.token_id,
            order_side=intent.order_side,
            source=FillEvidenceSource.ORDER_STATUS,
            observed_size=snapshot.matched_size,
            cumulative=True,
            observed_at=snapshot.observed_at,
            price_evidence=price_evidence,
            evidence_id=(
                f"submission:{order_id}:"
                f"{snapshot.raw_status}:"
                f"{snapshot.matched_size:.12f}"
            ),
        )

        return self.accounting.apply_observation(
            observation,
            confirm_quantity=True,
        )

    async def submit(
        self,
        intent: OrderIntent,
        *,
        order_args: Optional[object] = None,
        create_options: Optional[object] = None,
        pre_submit_context: Optional[PreSubmitContext] = None,
        raw_post_enter_observer: Optional[RawPostEnterObserver] = None,
        post_kwargs: Optional[Mapping[str, object]] = None,
        capture_wallet_baseline: Optional[bool] = None,
        reconcile_ambiguity: Optional[bool] = None,
    ) -> ManagedSubmission:
        """Create/sign and submit one immutable execution intent."""

        capture_baseline = (
            self.policy.capture_wallet_baseline_by_default
            if capture_wallet_baseline is None
            else bool(capture_wallet_baseline)
        )

        baseline = (
            await self.capture_wallet_baseline(intent)
            if capture_baseline
            else self.wallet_baseline(intent.lifecycle)
        )

        # Order construction happens before lifecycle registration.  If local
        # signing/metadata resolution fails here, no venue POST has begun and no
        # unresolved execution owner needs to be created.
        try:
            signed_order = await self.create_signed_order(
                intent,
                order_args=order_args,
                create_options=create_options,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result = SubmissionResult(
                outcome=SubmissionOutcome.FAILED_BEFORE_SUBMIT,
                intent=intent,
                order_id=None,
                venue_snapshot=None,
                post_call_entered=False,
                reason=f"order creation failed: {type(exc).__name__}: {exc}",
            )
            return ManagedSubmission(
                submission=result,
                lifecycle=None,
                fill_record=None,
                wallet_baseline=baseline,
                reconciliation=None,
            )

        # Register exact local ownership only after a signable order exists and
        # immediately before the submission state machine takes over.
        self.lifecycle.create(intent)

        result = await asyncio.to_thread(
            self.lifecycle.submit,
            intent,
            signed_order,
            pre_submit_context=pre_submit_context,
            raw_post_enter_observer=raw_post_enter_observer,
            post_kwargs=post_kwargs,
        )

        fill_record = None

        if result.order_id:
            try:
                fill_record = self._register_submission_accounting(result)
            except Exception as exc:
                runtime_print(
                    f"[execution] initial accounting error "
                    f"{result.order_id[:10]}: "
                    f"{type(exc).__name__}: {exc}"
                )

        lifecycle_snapshot = self.lifecycle.get(intent.lifecycle)

        should_reconcile = (
            self.policy.reconcile_ambiguous_submission
            if reconcile_ambiguity is None
            else bool(reconcile_ambiguity)
        )

        reconciliation_result = None

        if result.ambiguous and should_reconcile:
            reconciliation_result = await self.reconcile(
                intent.lifecycle,
                reason="ambiguous submission",
            )
            lifecycle_snapshot = self.lifecycle.get(intent.lifecycle)
            if reconciliation_result.order_id:
                fill_record = self.accounting.get(
                    reconciliation_result.order_id
                )

        return ManagedSubmission(
            submission=result,
            lifecycle=lifecycle_snapshot,
            fill_record=fill_record,
            wallet_baseline=baseline,
            reconciliation=reconciliation_result,
        )

    # ------------------------------------------------------------------
    # Status / accounting refresh
    # ------------------------------------------------------------------

    async def refresh_status(
        self,
        lifecycle_identity: LifecycleIdentity,
    ) -> Optional[LifecycleSnapshot]:
        """Fetch one exact status payload and apply it to lifecycle + accounting."""

        current = self.lifecycle.get(lifecycle_identity)
        if current is None:
            return None

        order_id = current.order_id
        if not order_id:
            return current

        try:
            payload = await asyncio.to_thread(
                self.transport.get_order,
                order_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            runtime_print(
                f"[execution] status refresh failed {order_id[:10]}: "
                f"{type(exc).__name__}: {exc}"
            )
            return self.lifecycle.get(lifecycle_identity)

        if not isinstance(payload, Mapping) or not payload:
            # Missing/unknown status is intentionally not rewritten as zero/closed.
            return self.lifecycle.get(lifecycle_identity)

        try:
            self.lifecycle.observe_status(
                lifecycle_identity,
                payload,
            )
        except Exception as exc:
            runtime_print(
                f"[execution] lifecycle status apply failed "
                f"{order_id[:10]}: {type(exc).__name__}: {exc}"
            )

        try:
            if self.accounting.get(order_id) is None:
                intent = current.working_order.intent
                self.accounting.register_order(
                    order_id=order_id,
                    token_id=intent.token_id,
                    lifecycle=intent.lifecycle,
                    order_side=intent.order_side,
                    requested_size=intent.size,
                    submitted_limit_price=intent.price,
                    created_at=intent.created_at,
                )

            self.accounting.observe_order_payload(
                order_id=order_id,
                payload=payload,
                source=FillEvidenceSource.ORDER_STATUS,
                confirm_quantity=True,
            )
        except Exception as exc:
            runtime_print(
                f"[execution] status accounting failed "
                f"{order_id[:10]}: {type(exc).__name__}: {exc}"
            )

        return self.lifecycle.get(lifecycle_identity)

    # ------------------------------------------------------------------
    # Reconciliation / cancellation
    # ------------------------------------------------------------------

    async def reconcile(
        self,
        lifecycle_identity: LifecycleIdentity,
        *,
        reason: str = "",
    ) -> ReconciliationResult:
        """Run exact lifecycle reconciliation using its captured wallet baseline."""

        return await self.reconciliation.reconcile(
            lifecycle_identity,
            wallet_baseline=self.wallet_baseline(lifecycle_identity),
            reason=reason,
        )

    def schedule_reconciliation(
        self,
        lifecycle_identity: LifecycleIdentity,
        *,
        reason: str = "",
    ) -> asyncio.Task:
        """Launch strongly-owned reconciliation without blocking the caller."""

        return self.launch_owned_task(
            self.reconcile(
                lifecycle_identity,
                reason=reason,
            ),
            label=f"reconcile-{lifecycle_identity.lifecycle_id[:16]}",
        )

    async def cancel(
        self,
        lifecycle_identity: LifecycleIdentity,
        *,
        reconcile_after: Optional[bool] = None,
    ) -> ManagedCancellation:
        """Cancel an exact order and optionally reconcile late/partial fill state."""

        result = await asyncio.to_thread(
            self.lifecycle.cancel,
            lifecycle_identity,
        )

        should_reconcile = (
            self.policy.reconcile_after_cancel
            if reconcile_after is None
            else bool(reconcile_after)
        )

        reconciliation_result = None

        if should_reconcile and result.order_id:
            reconciliation_result = await self.reconcile(
                lifecycle_identity,
                reason="post-cancel reconciliation",
            )

        return ManagedCancellation(
            cancellation=result,
            lifecycle=self.lifecycle.get(lifecycle_identity),
            reconciliation=reconciliation_result,
        )

    # ------------------------------------------------------------------
    # Read models
    # ------------------------------------------------------------------

    def snapshot(
        self,
        lifecycle_identity: LifecycleIdentity,
    ) -> Optional[LifecycleSnapshot]:
        return self.lifecycle.get(lifecycle_identity)

    def snapshot_by_order_id(
        self,
        order_id: str,
    ) -> Optional[LifecycleSnapshot]:
        return self.lifecycle.get_by_order_id(order_id)

    def fill_record(
        self,
        order_id: str,
    ) -> Optional[OrderFillRecord]:
        return self.accounting.get(order_id)

    def owns_token(self, token_id: str) -> bool:
        """Whether unresolved/live execution currently owns this token."""
        return self.lifecycle.has_owner_for_token(token_id)

    def stats(self) -> ExecutionManagerStats:
        snapshots = self.lifecycle.snapshots()

        with self._baseline_gate:
            baseline_count = len(self._wallet_baselines)

        return ExecutionManagerStats(
            active_lifecycles=len(snapshots),
            owned_lifecycles=sum(
                1 for snapshot in snapshots if snapshot.owns_execution
            ),
            accounting_records=len(self.accounting.records()),
            owned_background_tasks=sum(
                1 for task in self._owned_tasks if not task.done()
            ),
            active_reconciliations=self.reconciliation.active_reconciliations(),
            wallet_baselines=baseline_count,
        )

    # ------------------------------------------------------------------
    # Bounded cleanup
    # ------------------------------------------------------------------

    def prune(
        self,
        *,
        active_tokens: Iterable[str] = (),
        now: Optional[float] = None,
    ) -> dict[str, int]:
        """Prune only safely rebuildable/resolved state.

        Unresolved lifecycle/accounting ownership is retained regardless of the
        current scanner token set.
        """

        current_time = float(now or time.time())
        active_token_set = {
            str(token)
            for token in active_tokens
            if str(token)
        }

        owned = self.lifecycle.owned_snapshots()

        # Ownership beats scanner rotation.  Never prune token metadata for an
        # unresolved/live execution just because market discovery rotated away.
        keep_tokens = set(active_token_set)
        active_order_ids: set[str] = set()
        live_lifecycle_keys: set[str] = set()

        for snapshot in owned:
            intent = snapshot.working_order.intent
            keep_tokens.add(intent.token_id)
            live_lifecycle_keys.add(
                self._lifecycle_key(intent.lifecycle)
            )

            if snapshot.order_id:
                active_order_ids.add(snapshot.order_id)

        transport_removed = self.transport.prune_token_state(keep_tokens)

        accounting_removed = self.accounting.prune(
            resolved_older_than_seconds=(
                self.policy.resolved_accounting_keep_seconds
            ),
            active_order_ids=active_order_ids,
            now=current_time,
        )

        lifecycle_removed = self.lifecycle.prune_released(
            keep_last_seconds=self.policy.terminal_lifecycle_keep_seconds,
            now=current_time,
        )

        with self._baseline_gate:
            for key in list(self._wallet_baselines):
                if key in live_lifecycle_keys:
                    continue

                # Keep baselines while the corresponding lifecycle still exists.
                if any(
                    self._lifecycle_key(snapshot.lifecycle) == key
                    for snapshot in self.lifecycle.snapshots()
                ):
                    continue

                self._wallet_baselines.pop(key, None)

        return {
            "transport": int(transport_removed),
            "accounting": int(accounting_removed),
            "lifecycle": int(lifecycle_removed),
        }
