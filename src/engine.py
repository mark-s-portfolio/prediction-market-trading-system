"""
Event-driven orchestration engine for the public portfolio edition.

The engine is intentionally thin.  It does not own market-data truth, model
truth, risk truth, admission truth, or execution truth.  Those remain in their
dedicated modules.

Pipeline:

    MarketDataEvent
        -> coalesced market work queue
        -> CandidateProducer
        -> optional CandidateEnricher(s)
        -> CandidateQualityMeasurer
        -> CandidateExecutionAdapter.risk_proposal()
        -> RiskManager.assess()
        -> AdmissionPolicy.evaluate()
        -> immutable AdmissionPermit
        -> final synchronous current-state validator
        -> ExecutionManager.submit()

Important architecture boundaries:
- websocket callbacks enqueue work and return quickly
- candidate generation/model enrichment run outside the portfolio commit lock
- only the final current-state risk/admission/permit/submit handoff is serialized
- raw-POST validation is networkless and re-consumes current in-memory facts
- a newer market-data generation can invalidate an older candidate before POST
- execution lifecycle/accounting/reconciliation remain owned by execution modules
- position/risk state remain owned by risk modules
- admission permits remain owned by AdmissionPermitLedger
- the engine never recreates private mutable pair-state or a second order ledger

No production setup families, thresholds, bankroll parameters, proprietary model
authority, or asset-specific trading rules are embedded here.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import inspect
import math
import threading
import time
from typing import (
    Awaitable,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from src.execution.clob_transport import (
    PreSubmitContext,
    PreSubmitValidator,
    ValidationResult,
)
from src.execution.order_manager import (
    ExecutionManager,
    ManagedSubmission,
)
from src.execution.types import (
    LifecycleIdentity,
    OrderIntent,
    OrderIntentRole,
    SubmissionOutcome,
)
from src.market.orderbook import OrderBookStore
from src.market.types import MarketBooks, MarketDefinition
from src.market.websocket import (
    MarketDataEvent,
    MarketDataEventType,
)
from src.risk.risk_manager import (
    ProposedExposure,
    RiskAction,
    RiskDecision,
    RiskManager,
)
from src.strategy.admission import (
    AdmissionContext,
    AdmissionDecision,
    AdmissionVerdict,
)
from src.strategy.candidate import (
    CandidatePurpose,
    CandidateRegistry,
    StrategyCandidate,
)
from src.strategy.public_policy import (
    AdmissionTransportBridge,
    PermitPreSubmitValidator,
    PreparedSubmissionAuthority,
    PublicPolicyBundle,
)
from src.strategy.quality import (
    CandidateQualityMeasurer,
    CandidateQualitySnapshot,
)
from src.runtime.logging import runtime_print


MaybeAwaitable = Union[object, Awaitable[object]]


class EngineMode(str, Enum):
    """Whether ALLOW decisions may reach execution."""

    OBSERVE_ONLY = "OBSERVE_ONLY"
    EXECUTE = "EXECUTE"


class PipelineOutcome(str, Enum):
    """High-level result of one candidate pipeline pass."""

    DENIED = "DENIED"
    DEFERRED = "DEFERRED"
    OBSERVED_ALLOW = "OBSERVED_ALLOW"
    SUBMITTED = "SUBMITTED"
    NOT_SUBMITTED = "NOT_SUBMITTED"
    STALE = "STALE"
    INVALID = "INVALID"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class CandidateGenerationContext:
    """Immutable context supplied to candidate producers/enrichers."""

    market_data_generation: int
    observed_at: float

    def __post_init__(self) -> None:
        generation = int(self.market_data_generation)
        observed = float(self.observed_at)

        if generation < 0:
            raise ValueError(
                "market_data_generation must be non-negative"
            )
        if not math.isfinite(observed) or observed <= 0.0:
            raise ValueError("observed_at must be positive")

        object.__setattr__(
            self,
            "market_data_generation",
            generation,
        )
        object.__setattr__(self, "observed_at", observed)


class CandidateProducer(Protocol):
    """Generate zero or more immutable candidates from one current market view."""

    def generate(
        self,
        books: MarketBooks,
        context: CandidateGenerationContext,
    ) -> Union[
        Iterable[StrategyCandidate],
        Awaitable[Iterable[StrategyCandidate]],
    ]:
        ...


class CandidateEnricher(Protocol):
    """Optionally attach model/diagnostic evidence to one candidate."""

    def enrich(
        self,
        candidate: StrategyCandidate,
        books: MarketBooks,
        context: CandidateGenerationContext,
    ) -> Union[
        StrategyCandidate,
        Awaitable[StrategyCandidate],
    ]:
        ...


class CandidateExecutionAdapter(Protocol):
    """Bind a strategy candidate to generic risk and execution contracts.

    This adapter is explicit because economic purpose cannot safely be guessed by
    the engine.  For example, a completion order may reduce residual risk while
    still increasing one token's absolute quantity.
    """

    def risk_proposal(
        self,
        candidate: StrategyCandidate,
    ) -> ProposedExposure:
        ...

    def lifecycle_identity(
        self,
        candidate: StrategyCandidate,
    ) -> LifecycleIdentity:
        ...

    def order_intent(
        self,
        candidate: StrategyCandidate,
        lifecycle: LifecycleIdentity,
    ) -> OrderIntent:
        ...


@dataclass(frozen=True, slots=True)
class DirectExecutionAdapter:
    """Simple one-candidate/one-order adapter for portfolio demonstrations.

    Risk action mapping is explicit.  By default only OPENING is mapped; other
    purposes must be configured by the caller instead of being guessed.
    """

    risk_action_by_purpose: Mapping[CandidatePurpose, RiskAction] = field(
        default_factory=lambda: {
            CandidatePurpose.OPENING: RiskAction.NEW_EXPOSURE,
        }
    )

    def risk_proposal(
        self,
        candidate: StrategyCandidate,
    ) -> ProposedExposure:
        action = self.risk_action_by_purpose.get(
            candidate.quote.purpose
        )

        if action is None:
            raise ValueError(
                "candidate purpose has no explicit risk-action mapping: "
                f"{candidate.quote.purpose.value}"
            )

        return ProposedExposure(
            action=action,
            token_id=candidate.token_id,
            market_id=candidate.market_id,
            outcome_side=candidate.subject.outcome_side,
            quantity=candidate.quote.quantity,
            estimated_unit_cost=(
                candidate.quote.limit_price
                if action.creates_exposure
                else None
            ),
        )

    def lifecycle_identity(
        self,
        candidate: StrategyCandidate,
    ) -> LifecycleIdentity:
        return LifecycleIdentity(
            lifecycle_id=f"life-{candidate.candidate_id}",
            attempt_id=candidate.attempt_id,
            generation=0,
        )

    def order_intent(
        self,
        candidate: StrategyCandidate,
        lifecycle: LifecycleIdentity,
    ) -> OrderIntent:
        try:
            role = OrderIntentRole(
                candidate.quote.purpose.value
            )
        except ValueError:
            role = OrderIntentRole.UNKNOWN

        return OrderIntent(
            token_id=candidate.token_id,
            market_id=candidate.market_id,
            outcome_side=candidate.subject.outcome_side,
            order_side=candidate.quote.order_side,
            price=candidate.quote.limit_price,
            size=candidate.quote.quantity,
            role=role,
            lifecycle=lifecycle,
            time_in_force=candidate.quote.time_in_force,
            created_at=candidate.created_at,
        )


@dataclass(frozen=True, slots=True)
class EnginePolicy:
    """Operational orchestration behavior only."""

    mode: EngineMode = EngineMode.OBSERVE_ONLY

    market_queue_max_entries: int = 256
    market_worker_count: int = 2

    expected_model_keys: Tuple[str, ...] = field(
        default_factory=tuple
    )
    expected_feature_names: Tuple[str, ...] = field(
        default_factory=tuple
    )

    require_current_snapshot_match_before_execution: bool = True
    require_expiring_permit_for_execution: bool = True

    capture_wallet_baseline: Optional[bool] = None
    reconcile_ambiguity: Optional[bool] = None

    def __post_init__(self) -> None:
        queue_max = int(self.market_queue_max_entries)
        workers = int(self.market_worker_count)

        if queue_max <= 0:
            raise ValueError(
                "market_queue_max_entries must be positive"
            )
        if workers <= 0:
            raise ValueError(
                "market_worker_count must be positive"
            )

        model_keys = tuple(
            sorted(
                {
                    str(key or "").strip()
                    for key in self.expected_model_keys
                    if str(key or "").strip()
                }
            )
        )
        feature_names = tuple(
            sorted(
                {
                    str(name or "").strip()
                    for name in self.expected_feature_names
                    if str(name or "").strip()
                }
            )
        )

        object.__setattr__(
            self,
            "market_queue_max_entries",
            queue_max,
        )
        object.__setattr__(
            self,
            "market_worker_count",
            workers,
        )
        object.__setattr__(
            self,
            "expected_model_keys",
            model_keys,
        )
        object.__setattr__(
            self,
            "expected_feature_names",
            feature_names,
        )


@dataclass(frozen=True, slots=True)
class CandidatePipelineResult:
    candidate: StrategyCandidate
    outcome: PipelineOutcome

    quality: Optional[CandidateQualitySnapshot]
    risk: Optional[RiskDecision]
    admission: Optional[AdmissionDecision]
    execution: Optional[ManagedSubmission]

    observed_at: float
    reason: str = ""

    @property
    def submitted(self) -> bool:
        return self.execution is not None


@dataclass(frozen=True, slots=True)
class EngineStats:
    configured_markets: int
    queued_markets: int
    active_market_workers: int

    market_events: int
    queue_coalesced_events: int
    queue_overflow_events: int

    market_evaluations: int
    candidates_generated: int
    candidates_denied: int
    candidates_deferred: int
    candidates_stale: int
    candidates_submitted: int
    candidates_not_submitted: int
    pipeline_errors: int


PipelineListener = Callable[
    [CandidatePipelineResult],
    Union[None, Awaitable[None]],
]


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


class EnginePreSubmitValidator:
    """Final networkless current-state seal at the raw POST boundary.

    It first validates the immutable permit envelope, then re-consumes current
    in-memory market/risk/admission evidence.  It does not perform network I/O and
    does not consume the permit; the raw-post-enter observer remains the one-shot
    permit consumer.
    """

    def __init__(
        self,
        *,
        base_validator: PermitPreSubmitValidator,
        candidate_lookup: Callable[
            [str],
            Optional[StrategyCandidate],
        ],
        current_books_lookup: Callable[
            [str],
            tuple[Optional[MarketBooks], Optional[int]],
        ],
        quality_measurer: CandidateQualityMeasurer,
        risk_manager: RiskManager,
        execution_adapter: CandidateExecutionAdapter,
        policy_bundle: PublicPolicyBundle,
        engine_policy: EnginePolicy,
    ) -> None:
        self.base_validator = base_validator
        self.candidate_lookup = candidate_lookup
        self.current_books_lookup = current_books_lookup
        self.quality_measurer = quality_measurer
        self.risk_manager = risk_manager
        self.execution_adapter = execution_adapter
        self.policy_bundle = policy_bundle
        self.engine_policy = engine_policy

    @staticmethod
    def _deny(
        reason: str,
        *,
        evidence: Optional[Mapping[str, object]] = None,
    ) -> ValidationResult:
        return ValidationResult(
            allowed=False,
            reason=reason,
            evidence=dict(evidence or {}),
        )

    def validate(
        self,
        context: PreSubmitContext,
    ) -> ValidationResult:
        base = self.base_validator.validate(context)
        if not base.allowed:
            return base

        candidate_id = str(
            context.metadata.get("candidate_id") or ""
        )
        if not candidate_id:
            return self._deny(
                "candidate identity missing at final consumer"
            )

        candidate = self.candidate_lookup(candidate_id)
        if candidate is None:
            return self._deny(
                "candidate unavailable at final consumer",
                evidence={"candidate_id": candidate_id},
            )

        books, generation = self.current_books_lookup(
            candidate.market_id
        )
        if books is None or generation is None:
            return self._deny(
                "current in-memory market evidence unavailable",
                evidence={
                    "candidate_id": candidate_id,
                    "market_id": candidate.market_id,
                },
            )

        now = time.time()

        try:
            quality = self.quality_measurer.measure(
                candidate,
                books,
                now=now,
                market_data_generation=generation,
                expected_model_keys=(
                    self.engine_policy.expected_model_keys
                ),
                expected_feature_names=(
                    self.engine_policy.expected_feature_names
                ),
            )
        except Exception as exc:
            return self._deny(
                "final quality measurement failed",
                evidence={
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                },
            )

        if (
            self.engine_policy
            .require_current_snapshot_match_before_execution
            and not quality.books.candidate_snapshot_match
        ):
            return self._deny(
                "candidate origin snapshot is no longer current",
                evidence={
                    "candidate_snapshot_id": (
                        candidate.market_snapshot.snapshot_id
                    ),
                    "candidate_generation": (
                        candidate.market_snapshot
                        .market_data_generation
                    ),
                    "current_generation": generation,
                    "current_yes_timestamp": (
                        books.yes.timestamp
                    ),
                    "current_no_timestamp": (
                        books.no.timestamp
                    ),
                },
            )

        try:
            current_lifecycle = (
                self.execution_adapter.lifecycle_identity(
                    candidate
                )
            )

            if (
                current_lifecycle.lifecycle_id
                != context.lifecycle_id
                or current_lifecycle.attempt_id
                != context.attempt_id
            ):
                return self._deny(
                    "final execution lifecycle binding mismatch",
                    evidence={
                        "context_lifecycle_id": context.lifecycle_id,
                        "expected_lifecycle_id": (
                            current_lifecycle.lifecycle_id
                        ),
                        "context_attempt_id": context.attempt_id,
                        "expected_attempt_id": (
                            current_lifecycle.attempt_id
                        ),
                    },
                )

            proposal = self.execution_adapter.risk_proposal(
                candidate
            )
            risk = self.risk_manager.assess(
                proposal,
                exclude_lifecycle=current_lifecycle,
            )

            decision = self.policy_bundle.policy.evaluate(
                AdmissionContext(
                    candidate=candidate,
                    quality=quality,
                    risk_proposal=proposal,
                    risk_decision=risk,
                    evaluated_at=now,
                )
            )
        except Exception as exc:
            return self._deny(
                "final risk/admission evaluation failed",
                evidence={
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                },
            )

        if decision.verdict is not AdmissionVerdict.ALLOW:
            return self._deny(
                "current evidence no longer admits candidate",
                evidence={
                    "verdict": decision.verdict.value,
                    "decision_fingerprint": (
                        decision.decision_fingerprint
                    ),
                    "blocking_checks": tuple(
                        check.rule_id
                        for check in decision.blocking_checks
                    ),
                    "deferring_checks": tuple(
                        check.rule_id
                        for check in decision.deferring_checks
                    ),
                },
            )

        return ValidationResult(
            allowed=True,
            reason=(
                "permit and current normalized authority both valid"
            ),
            evidence={
                "candidate_id": candidate.candidate_id,
                "current_generation": generation,
                "current_decision_fingerprint": (
                    decision.decision_fingerprint
                ),
                "permit_decision_fingerprint": (
                    context.metadata.get(
                        "admission_decision_fingerprint"
                    )
                ),
            },
        )


class TradingEngine:
    """Thin event-driven coordinator for the public architecture."""

    def __init__(
        self,
        *,
        orderbooks: OrderBookStore,
        candidate_producer: CandidateProducer,
        execution_adapter: CandidateExecutionAdapter,
        quality_measurer: CandidateQualityMeasurer,
        risk_manager: RiskManager,
        policy_bundle: PublicPolicyBundle,
        execution_manager: ExecutionManager,
        candidate_registry: Optional[CandidateRegistry] = None,
        enrichers: Sequence[CandidateEnricher] = (),
        policy: EnginePolicy = EnginePolicy(),
        result_listener: Optional[PipelineListener] = None,
    ) -> None:
        self.orderbooks = orderbooks
        self.candidate_producer = candidate_producer
        self.execution_adapter = execution_adapter
        self.quality_measurer = quality_measurer
        self.risk_manager = risk_manager
        self.policy_bundle = policy_bundle
        self.execution_manager = execution_manager
        self.candidate_registry = (
            candidate_registry or CandidateRegistry()
        )
        self.enrichers = tuple(enrichers)
        self.policy = policy
        self.result_listener = result_listener

        self._markets_by_id: Dict[
            str,
            MarketDefinition,
        ] = {}
        self._market_by_token: Dict[
            str,
            MarketDefinition,
        ] = {}

        self._current_market_gate = threading.RLock()
        self._current_books: Dict[str, MarketBooks] = {}
        self._market_generation: Dict[str, int] = {}

        self._candidate_gate = threading.RLock()

        self._queue: asyncio.Queue[str] = asyncio.Queue(
            maxsize=self.policy.market_queue_max_entries
        )
        self._queued_markets: set[str] = set()

        self._worker_tasks: set[asyncio.Task] = set()
        self._running = False

        # Tiny final portfolio commit lock: never held through websocket receive,
        # candidate production, model enrichment, or market discovery.
        self._portfolio_commit_lock = asyncio.Lock()

        self._stats_gate = threading.RLock()
        self._market_events = 0
        self._queue_coalesced_events = 0
        self._queue_overflow_events = 0
        self._market_evaluations = 0
        self._candidates_generated = 0
        self._candidates_denied = 0
        self._candidates_deferred = 0
        self._candidates_stale = 0
        self._candidates_submitted = 0
        self._candidates_not_submitted = 0
        self._pipeline_errors = 0

        self._final_validator = EnginePreSubmitValidator(
            base_validator=policy_bundle.pre_submit_validator,
            candidate_lookup=self._candidate_lookup,
            current_books_lookup=self._current_books_lookup,
            quality_measurer=quality_measurer,
            risk_manager=risk_manager,
            execution_adapter=execution_adapter,
            policy_bundle=policy_bundle,
            engine_policy=policy,
        )

        self._install_final_validator()

    # ------------------------------------------------------------------
    # Startup contract
    # ------------------------------------------------------------------

    def _install_final_validator(self) -> None:
        transport = self.execution_manager.transport
        current = transport.pre_submit_validator

        if current is self._final_validator:
            return

        if current is not self.policy_bundle.pre_submit_validator:
            raise RuntimeError(
                "execution transport has a different pre-submit validator; "
                "refusing to overwrite an unknown safety consumer"
            )

        transport.pre_submit_validator = self._final_validator

    async def configure_markets(
        self,
        markets: Sequence[MarketDefinition],
        *,
        prepare_execution: bool = True,
    ) -> None:
        normalized = tuple(markets)

        by_id: Dict[str, MarketDefinition] = {}
        by_token: Dict[str, MarketDefinition] = {}

        for market in normalized:
            key = market.pair_key

            if key in by_id and by_id[key] != market:
                raise ValueError(
                    f"duplicate market identity: {key}"
                )

            for token_id in (
                market.yes_token,
                market.no_token,
            ):
                existing = by_token.get(token_id)
                if existing is not None and existing != market:
                    raise ValueError(
                        "token belongs to multiple configured markets"
                    )
                by_token[token_id] = market

            by_id[key] = market

        self._markets_by_id = by_id
        self._market_by_token = by_token

        with self._current_market_gate:
            for key in list(self._current_books):
                if key not in by_id:
                    self._current_books.pop(key, None)
                    self._market_generation.pop(key, None)

        if prepare_execution and normalized:
            await self.execution_manager.prepare_markets(
                normalized
            )

    # ------------------------------------------------------------------
    # Fast market-event ingress
    # ------------------------------------------------------------------

    async def handle_market_event(
        self,
        event: MarketDataEvent,
    ) -> None:
        """Fast websocket callback: update current cache, enqueue, return."""

        if event.event_type not in {
            MarketDataEventType.BOOK,
            MarketDataEventType.BOOK_DELTA,
            MarketDataEventType.TOP_OF_BOOK,
        }:
            return

        token_id = str(event.token_id or "")
        if not token_id:
            return

        market = self._market_by_token.get(token_id)
        if market is None:
            return

        with self._stats_gate:
            self._market_events += 1

        self._refresh_current_cache_sync(market)

        key = market.pair_key

        if key in self._queued_markets:
            with self._stats_gate:
                self._queue_coalesced_events += 1
            return

        try:
            self._queue.put_nowait(key)
        except asyncio.QueueFull:
            with self._stats_gate:
                self._queue_overflow_events += 1
            return

        self._queued_markets.add(key)

    def _refresh_current_cache_sync(
        self,
        market: MarketDefinition,
    ) -> Optional[MarketBooks]:
        yes = self.orderbooks.ws_book(market.yes_token)
        no = self.orderbooks.ws_book(market.no_token)

        if yes is None or no is None:
            return None

        books = MarketBooks(
            market=market,
            yes=yes,
            no=no,
        )

        with self._current_market_gate:
            key = market.pair_key
            previous = self._current_books.get(key)

            if (
                previous is None
                or previous.yes.timestamp != books.yes.timestamp
                or previous.no.timestamp != books.no.timestamp
            ):
                self._market_generation[key] = (
                    self._market_generation.get(key, 0) + 1
                )

            self._current_books[key] = books

        return books

    def _current_books_lookup(
        self,
        market_id: str,
    ) -> tuple[Optional[MarketBooks], Optional[int]]:
        with self._current_market_gate:
            books = self._current_books.get(str(market_id or ""))
            generation = self._market_generation.get(
                str(market_id or "")
            )
            return books, generation

    def _candidate_lookup(
        self,
        candidate_id: str,
    ) -> Optional[StrategyCandidate]:
        with self._candidate_gate:
            return self.candidate_registry.get(candidate_id)

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return

        self._running = True

        for index in range(self.policy.market_worker_count):
            task = asyncio.create_task(
                self._market_worker(index),
                name=f"market-pipeline-{index}",
            )
            self._worker_tasks.add(task)

            def _done(
                done_task: asyncio.Task,
                *,
                worker_index: int = index,
            ) -> None:
                self._worker_tasks.discard(done_task)

                if done_task.cancelled():
                    return

                try:
                    exc = done_task.exception()
                except Exception:
                    return

                if exc is not None:
                    runtime_print(
                        f"[engine] market worker {worker_index} failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

            task.add_done_callback(_done)

    async def close(self) -> None:
        self._running = False

        tasks = tuple(self._worker_tasks)
        for task in tasks:
            if not task.done():
                task.cancel()

        if tasks:
            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

        self._worker_tasks.clear()

    async def _market_worker(
        self,
        worker_index: int,
    ) -> None:
        while self._running:
            key = await self._queue.get()

            try:
                self._queued_markets.discard(key)

                await self.process_market_once(key)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                with self._stats_gate:
                    self._pipeline_errors += 1

                runtime_print(
                    f"[engine] market pipeline error {key}: "
                    f"{type(exc).__name__}: {exc}"
                )
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------
    # Market -> candidate generation
    # ------------------------------------------------------------------

    async def process_market_once(
        self,
        market_id: str,
    ) -> Tuple[CandidatePipelineResult, ...]:
        market_id = str(market_id or "").strip()
        market = self._markets_by_id.get(market_id)

        if market is None:
            raise KeyError(
                f"market not configured: {market_id}"
            )

        books = self._refresh_current_cache_sync(market)
        if books is None:
            return ()

        _, generation = self._current_books_lookup(market_id)
        if generation is None:
            return ()

        context = CandidateGenerationContext(
            market_data_generation=generation,
            observed_at=time.time(),
        )

        with self._stats_gate:
            self._market_evaluations += 1

        generated = await _maybe_await(
            self.candidate_producer.generate(
                books,
                context,
            )
        )

        candidates = tuple(generated or ())

        with self._stats_gate:
            self._candidates_generated += len(candidates)

        results: list[CandidatePipelineResult] = []

        for candidate in candidates:
            try:
                result = await self._process_generated_candidate(
                    candidate,
                    books=books,
                    generation_context=context,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                with self._stats_gate:
                    self._pipeline_errors += 1

                result = CandidatePipelineResult(
                    candidate=candidate,
                    outcome=PipelineOutcome.ERROR,
                    quality=None,
                    risk=None,
                    admission=None,
                    execution=None,
                    observed_at=time.time(),
                    reason=(
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

            results.append(result)
            await self._publish_result(result)

        return tuple(results)

    def _validate_candidate_origin(
        self,
        candidate: StrategyCandidate,
        *,
        books: MarketBooks,
        generation_context: CandidateGenerationContext,
    ) -> Optional[str]:
        if candidate.market_id != books.market.pair_key:
            return "candidate market differs from producer market"

        if candidate.token_id not in {
            books.market.yes_token,
            books.market.no_token,
        }:
            return "candidate token not in producer market"

        snapshot = candidate.market_snapshot

        if snapshot.market_id != candidate.market_id:
            return "candidate snapshot market mismatch"

        if (
            abs(
                snapshot.yes_book_timestamp
                - books.yes.timestamp
            )
            > 1e-9
            or abs(
                snapshot.no_book_timestamp
                - books.no.timestamp
            )
            > 1e-9
        ):
            return (
                "candidate origin timestamps do not match "
                "producer MarketBooks"
            )

        if (
            snapshot.market_data_generation
            != generation_context.market_data_generation
        ):
            return (
                "candidate origin generation does not match "
                "producer generation"
            )

        return None

    async def _process_generated_candidate(
        self,
        candidate: StrategyCandidate,
        *,
        books: MarketBooks,
        generation_context: CandidateGenerationContext,
    ) -> CandidatePipelineResult:
        invalid_reason = self._validate_candidate_origin(
            candidate,
            books=books,
            generation_context=generation_context,
        )

        if invalid_reason:
            return CandidatePipelineResult(
                candidate=candidate,
                outcome=PipelineOutcome.INVALID,
                quality=None,
                risk=None,
                admission=None,
                execution=None,
                observed_at=time.time(),
                reason=invalid_reason,
            )

        enriched = candidate

        for enricher in self.enrichers:
            enriched = await _maybe_await(
                enricher.enrich(
                    enriched,
                    books,
                    generation_context,
                )
            )

            if not isinstance(enriched, StrategyCandidate):
                raise TypeError(
                    "CandidateEnricher must return StrategyCandidate"
                )

            if enriched.candidate_id != candidate.candidate_id:
                raise ValueError(
                    "enricher cannot replace candidate identity"
                )
            if enriched.fingerprint != candidate.fingerprint:
                raise ValueError(
                    "enricher cannot mutate immutable candidate subject"
                )

        with self._candidate_gate:
            self.candidate_registry.put(enriched)

        # Outside the portfolio lock: descriptive pre-measurement for diagnostics.
        preliminary_quality = self.quality_measurer.measure(
            enriched,
            books,
            now=time.time(),
            market_data_generation=(
                generation_context.market_data_generation
            ),
            expected_model_keys=self.policy.expected_model_keys,
            expected_feature_names=(
                self.policy.expected_feature_names
            ),
        )

        return await self._final_commit(
            enriched,
            preliminary_quality=preliminary_quality,
        )

    # ------------------------------------------------------------------
    # Tiny final portfolio commit
    # ------------------------------------------------------------------

    async def _final_commit(
        self,
        candidate: StrategyCandidate,
        *,
        preliminary_quality: CandidateQualitySnapshot,
    ) -> CandidatePipelineResult:
        async with self._portfolio_commit_lock:
            current_books, generation = self._current_books_lookup(
                candidate.market_id
            )

            if current_books is None or generation is None:
                with self._stats_gate:
                    self._candidates_stale += 1

                return CandidatePipelineResult(
                    candidate=candidate,
                    outcome=PipelineOutcome.STALE,
                    quality=preliminary_quality,
                    risk=None,
                    admission=None,
                    execution=None,
                    observed_at=time.time(),
                    reason="current market evidence unavailable",
                )

            quality = self.quality_measurer.measure(
                candidate,
                current_books,
                now=time.time(),
                market_data_generation=generation,
                expected_model_keys=(
                    self.policy.expected_model_keys
                ),
                expected_feature_names=(
                    self.policy.expected_feature_names
                ),
            )

            if (
                self.policy
                .require_current_snapshot_match_before_execution
                and not quality.books.candidate_snapshot_match
            ):
                with self._stats_gate:
                    self._candidates_stale += 1

                return CandidatePipelineResult(
                    candidate=candidate,
                    outcome=PipelineOutcome.STALE,
                    quality=quality,
                    risk=None,
                    admission=None,
                    execution=None,
                    observed_at=time.time(),
                    reason=(
                        "new market-data generation arrived "
                        "before final admission"
                    ),
                )

            proposal = self.execution_adapter.risk_proposal(
                candidate
            )
            risk = self.risk_manager.assess(proposal)

            admission = self.policy_bundle.policy.evaluate(
                AdmissionContext(
                    candidate=candidate,
                    quality=quality,
                    risk_proposal=proposal,
                    risk_decision=risk,
                    evaluated_at=time.time(),
                )
            )

            if admission.verdict is AdmissionVerdict.DENY:
                with self._stats_gate:
                    self._candidates_denied += 1

                return CandidatePipelineResult(
                    candidate=candidate,
                    outcome=PipelineOutcome.DENIED,
                    quality=quality,
                    risk=risk,
                    admission=admission,
                    execution=None,
                    observed_at=time.time(),
                    reason="admission denied",
                )

            if admission.verdict is AdmissionVerdict.DEFER:
                with self._stats_gate:
                    self._candidates_deferred += 1

                return CandidatePipelineResult(
                    candidate=candidate,
                    outcome=PipelineOutcome.DEFERRED,
                    quality=quality,
                    risk=risk,
                    admission=admission,
                    execution=None,
                    observed_at=time.time(),
                    reason="admission deferred",
                )

            if self.policy.mode is EngineMode.OBSERVE_ONLY:
                return CandidatePipelineResult(
                    candidate=candidate,
                    outcome=PipelineOutcome.OBSERVED_ALLOW,
                    quality=quality,
                    risk=risk,
                    admission=admission,
                    execution=None,
                    observed_at=time.time(),
                    reason="observe-only engine mode",
                )

            permit = admission.permit
            if permit is None:
                raise RuntimeError(
                    "ALLOW decision missing admission permit"
                )

            if (
                self.policy.require_expiring_permit_for_execution
                and permit.expires_at is None
            ):
                return CandidatePipelineResult(
                    candidate=candidate,
                    outcome=PipelineOutcome.INVALID,
                    quality=quality,
                    risk=risk,
                    admission=admission,
                    execution=None,
                    observed_at=time.time(),
                    reason=(
                        "execution mode requires an expiring "
                        "admission permit"
                    ),
                )

            lifecycle = self.execution_adapter.lifecycle_identity(
                candidate
            )
            intent = self.execution_adapter.order_intent(
                candidate,
                lifecycle,
            )
            self._assert_execution_binding(
                candidate,
                lifecycle=lifecycle,
                intent=intent,
            )

            authority = self.policy_bundle.transport_bridge.prepare(
                decision=admission,
                candidate=candidate,
                lifecycle_id=lifecycle.lifecycle_id,
            )

            try:
                execution = await self.execution_manager.submit(
                    intent,
                    pre_submit_context=(
                        authority.pre_submit_context
                    ),
                    raw_post_enter_observer=(
                        authority.raw_post_enter_observer
                    ),
                    capture_wallet_baseline=(
                        self.policy.capture_wallet_baseline
                    ),
                    reconcile_ambiguity=(
                        self.policy.reconcile_ambiguity
                    ),
                )
            except asyncio.CancelledError:
                self._retire_unconsumed_permit(
                    authority,
                    reason="pipeline-cancelled-before-raw-post",
                )
                raise
            except Exception:
                self._retire_unconsumed_permit(
                    authority,
                    reason="pipeline-exception-before-raw-post",
                )
                raise

            self._retire_unconsumed_permit(
                authority,
                reason="confirmed-local-submit-closed",
            )

            raw_post_entered = bool(
                execution.submission.post_call_entered
            )

            with self._stats_gate:
                if raw_post_entered:
                    self._candidates_submitted += 1
                else:
                    self._candidates_not_submitted += 1

            return CandidatePipelineResult(
                candidate=candidate,
                outcome=(
                    PipelineOutcome.SUBMITTED
                    if raw_post_entered
                    else PipelineOutcome.NOT_SUBMITTED
                ),
                quality=quality,
                risk=risk,
                admission=admission,
                execution=execution,
                observed_at=time.time(),
                reason=execution.submission.outcome.value,
            )

    @staticmethod
    def _assert_execution_binding(
        candidate: StrategyCandidate,
        *,
        lifecycle: LifecycleIdentity,
        intent: OrderIntent,
    ) -> None:
        mismatches = []

        if lifecycle.attempt_id != candidate.attempt_id:
            mismatches.append("lifecycle.attempt_id")
        if intent.lifecycle != lifecycle:
            mismatches.append("intent.lifecycle")
        if intent.token_id != candidate.token_id:
            mismatches.append("intent.token_id")
        if intent.market_id != candidate.market_id:
            mismatches.append("intent.market_id")
        if (
            intent.outcome_side
            is not candidate.subject.outcome_side
        ):
            mismatches.append("intent.outcome_side")
        if intent.order_side is not candidate.quote.order_side:
            mismatches.append("intent.order_side")
        if (
            abs(intent.price - candidate.quote.limit_price)
            > 1e-12
        ):
            mismatches.append("intent.price")
        if (
            abs(intent.size - candidate.quote.quantity)
            > 1e-12
        ):
            mismatches.append("intent.size")
        if (
            intent.time_in_force
            is not candidate.quote.time_in_force
        ):
            mismatches.append("intent.time_in_force")

        if mismatches:
            raise ValueError(
                "execution adapter changed admitted candidate economics: "
                + ", ".join(mismatches)
            )

    def _retire_unconsumed_permit(
        self,
        authority: PreparedSubmissionAuthority,
        *,
        reason: str,
    ) -> None:
        ledger = self.policy_bundle.permit_ledger
        permit_id = authority.permit.permit_id

        if ledger.get(permit_id) is None:
            return

        try:
            ledger.consume(
                permit_id,
                candidate=authority.candidate,
                consumer=reason,
                now=time.time(),
            )
        except Exception as exc:
            runtime_print(
                f"[engine] failed to retire unconsumed permit "
                f"{permit_id}: {type(exc).__name__}: {exc}"
            )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    async def _publish_result(
        self,
        result: CandidatePipelineResult,
    ) -> None:
        listener = self.result_listener
        if listener is None:
            return

        try:
            await _maybe_await(listener(result))
        except Exception as exc:
            runtime_print(
                f"[engine] result listener failed: "
                f"{type(exc).__name__}: {exc}"
            )

    def stats(self) -> EngineStats:
        with self._stats_gate:
            return EngineStats(
                configured_markets=len(self._markets_by_id),
                queued_markets=self._queue.qsize(),
                active_market_workers=sum(
                    1
                    for task in self._worker_tasks
                    if not task.done()
                ),
                market_events=self._market_events,
                queue_coalesced_events=(
                    self._queue_coalesced_events
                ),
                queue_overflow_events=(
                    self._queue_overflow_events
                ),
                market_evaluations=self._market_evaluations,
                candidates_generated=self._candidates_generated,
                candidates_denied=self._candidates_denied,
                candidates_deferred=self._candidates_deferred,
                candidates_stale=self._candidates_stale,
                candidates_submitted=(
                    self._candidates_submitted
                ),
                candidates_not_submitted=(
                    self._candidates_not_submitted
                ),
                pipeline_errors=self._pipeline_errors,
            )

    def current_books(
        self,
        market_id: str,
    ) -> Optional[MarketBooks]:
        books, _ = self._current_books_lookup(market_id)
        return books

    def market_generation(
        self,
        market_id: str,
    ) -> Optional[int]:
        _, generation = self._current_books_lookup(market_id)
        return generation
