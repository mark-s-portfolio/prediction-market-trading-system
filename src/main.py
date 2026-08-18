"""
Application wiring for the sanitized public portfolio edition.

This entrypoint intentionally runs the repository in OBSERVE_ONLY mode.

The project contains reusable execution/lifecycle/reconciliation components, but
this public `main.py` does not turn environment credentials into a live trading
process.  Live order placement requires an explicit application-specific SDK and
inventory-integration adapter outside this sanitized portfolio snapshot.

Runtime topology:

    AppSettings
        -> MarketDiscovery
        -> OrderBookStore
        -> MarketWebSocketClient
        -> TradingEngine
            -> CandidateProducer
            -> CandidateQualityMeasurer
            -> RiskManager
            -> AdmissionPolicy
            -> observe-only result

The execution subsystem is still constructed so the dependency graph remains
visible and testable, but its raw client is a fail-closed disabled adapter and the
engine never enters EXECUTE mode from this entrypoint.

Operational behavior:
- synchronous market discovery runs off the asyncio event loop
- active market-set changes rotate the WebSocket generation
- the engine owns its bounded/coalesced market work queue
- SIGINT/SIGTERM trigger cooperative graceful shutdown
- logging is non-blocking
- no credentials or strategy parameters are stored in source

No production strategy thresholds, setup families, bankroll parameters,
asset-specific admission rules, or proprietary causal authority are included.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import signal
import time
from typing import (
    Awaitable,
    Callable,
    Iterable,
    Optional,
    Sequence,
    Tuple,
)

from src.engine import (
    CandidateEnricher,
    CandidatePipelineResult,
    CandidateProducer,
    DirectExecutionAdapter,
    EngineMode,
    EnginePolicy,
    PipelineListener,
    TradingEngine,
)
from src.execution.clob_transport import ClobTransport
from src.execution.fill_accounting import FillAccounting
from src.execution.order_lifecycle import OrderLifecycleService
from src.execution.order_manager import (
    ExecutionManager,
    ExecutionManagerPolicy,
)
from src.execution.reconciliation import (
    ReconciliationPolicy,
    ReconciliationService,
)
from src.market.discovery import MarketDiscovery
from src.market.orderbook import OrderBookStore
from src.market.types import MarketDefinition
from src.market.websocket import MarketWebSocketClient
from src.risk.position_state import PositionBook
from src.risk.risk_manager import RiskLimits, RiskManager
from src.runtime.config import AppSettings, settings
from src.runtime.logging import (
    install_disconnect_resilience,
    runtime_logger,
    runtime_print,
)
from src.strategy.candidate import CandidateRegistry, StrategyCandidate
from src.strategy.public_policy import (
    PublicPolicyBundle,
    PublicPolicyConfig,
    build_public_policy_bundle,
)
from src.strategy.quality import CandidateQualityMeasurer


class NoOpCandidateProducer:
    """Safe default producer for the portfolio entrypoint.

    The repository demonstrates the orchestration architecture without publishing
    the private strategy that generates production candidates.
    """

    def generate(
        self,
        books,
        context,
    ) -> Tuple[StrategyCandidate, ...]:
        return ()


class DisabledExecutionClient:
    """Fail-closed raw-client placeholder for observe-only application wiring."""

    _MESSAGE = (
        "live execution is disabled in the sanitized portfolio entrypoint"
    )

    def _disabled(self, *args, **kwargs):
        raise RuntimeError(self._MESSAGE)

    create_order = _disabled
    post_order = _disabled
    get_order = _disabled
    get_trades = _disabled
    cancel_order = _disabled
    get_balance_allowance = _disabled
    get_tick_size = _disabled
    get_neg_risk = _disabled
    get_version = _disabled
    get_fee_rate_bps = _disabled


def _disabled_order_argument_factory(intent):
    raise RuntimeError(
        "order construction is disabled in the sanitized portfolio entrypoint"
    )


@dataclass(frozen=True, slots=True)
class ApplicationServices:
    """Constructed application dependency graph."""

    settings: AppSettings

    discovery: MarketDiscovery
    orderbooks: OrderBookStore

    policy_bundle: PublicPolicyBundle

    transport: ClobTransport
    lifecycle: OrderLifecycleService
    accounting: FillAccounting
    reconciliation: ReconciliationService
    execution: ExecutionManager

    positions: PositionBook
    risk: RiskManager

    quality: CandidateQualityMeasurer
    candidates: CandidateRegistry

    engine: TradingEngine


def _default_result_listener(
    app_settings: AppSettings,
) -> PipelineListener:
    async def _listener(
        result: CandidatePipelineResult,
    ) -> None:
        if not app_settings.runtime.detailed_logs:
            return

        runtime_print(
            "[pipeline]",
            result.outcome.value,
            f"candidate={result.candidate.candidate_id}",
            f"market={result.candidate.market_id}",
            f"token={result.candidate.token_id[:12]}",
            f"reason={result.reason}",
        )

    return _listener


def build_application(
    *,
    app_settings: AppSettings = settings,
    candidate_producer: Optional[CandidateProducer] = None,
    enrichers: Sequence[CandidateEnricher] = (),
    risk_limits: Optional[RiskLimits] = None,
    public_policy_config: Optional[PublicPolicyConfig] = None,
    result_listener: Optional[PipelineListener] = None,
) -> ApplicationServices:
    """Build the sanitized observe-only application graph.

    A custom public/demo CandidateProducer may be injected by tests or examples,
    but this entrypoint deliberately refuses live execution.
    """

    if app_settings.runtime.live_execution_requested:
        raise RuntimeError(
            "The sanitized portfolio main.py is observe-only. "
            "Live execution requires an explicit external application adapter "
            "for the current venue SDK and inventory synchronization."
        )

    producer = candidate_producer or NoOpCandidateProducer()
    limits = risk_limits or RiskLimits()
    policy_config = public_policy_config or PublicPolicyConfig()

    discovery = MarketDiscovery(app_settings=app_settings)
    orderbooks = OrderBookStore(app_settings=app_settings)

    policy_bundle = build_public_policy_bundle(
        policy_config
    )

    transport = ClobTransport(
        DisabledExecutionClient(),
        pre_submit_validator=(
            policy_bundle.pre_submit_validator
        ),
    )

    lifecycle = OrderLifecycleService(transport)
    accounting = FillAccounting()

    reconciliation = ReconciliationService(
        transport=transport,
        lifecycle=lifecycle,
        accounting=accounting,
        policy=ReconciliationPolicy(),
    )

    execution = ExecutionManager(
        transport=transport,
        lifecycle=lifecycle,
        accounting=accounting,
        reconciliation=reconciliation,
        order_argument_factory=(
            _disabled_order_argument_factory
        ),
        policy=ExecutionManagerPolicy(
            capture_wallet_baseline_by_default=False,
            reconcile_ambiguous_submission=True,
            reconcile_after_cancel=True,
            prewarm_market_metadata=False,
        ),
    )

    positions = PositionBook()
    risk = RiskManager(
        positions=positions,
        execution_ownership=lifecycle,
        limits=limits,
    )

    quality = CandidateQualityMeasurer()
    candidates = CandidateRegistry()

    engine = TradingEngine(
        orderbooks=orderbooks,
        candidate_producer=producer,
        execution_adapter=DirectExecutionAdapter(),
        quality_measurer=quality,
        risk_manager=risk,
        policy_bundle=policy_bundle,
        execution_manager=execution,
        candidate_registry=candidates,
        enrichers=tuple(enrichers),
        policy=EnginePolicy(
            mode=EngineMode.OBSERVE_ONLY,
            expected_model_keys=(
                policy_config.required_model_keys
            ),
            expected_feature_names=(
                policy_config.required_feature_names
            ),
            require_current_snapshot_match_before_execution=True,
            require_expiring_permit_for_execution=True,
            capture_wallet_baseline=False,
            reconcile_ambiguity=False,
        ),
        result_listener=(
            result_listener
            or _default_result_listener(app_settings)
        ),
    )

    return ApplicationServices(
        settings=app_settings,
        discovery=discovery,
        orderbooks=orderbooks,
        policy_bundle=policy_bundle,
        transport=transport,
        lifecycle=lifecycle,
        accounting=accounting,
        reconciliation=reconciliation,
        execution=execution,
        positions=positions,
        risk=risk,
        quality=quality,
        candidates=candidates,
        engine=engine,
    )


WebSocketFactory = Callable[
    [
        OrderBookStore,
        AppSettings,
        Callable,
    ],
    MarketWebSocketClient,
]


def _default_websocket_factory(
    orderbooks: OrderBookStore,
    app_settings: AppSettings,
    event_handler: Callable,
) -> MarketWebSocketClient:
    return MarketWebSocketClient(
        orderbooks,
        app_settings=app_settings,
        event_handler=event_handler,
    )


def _market_set_identity(
    markets: Sequence[MarketDefinition],
) -> Tuple[tuple[str, str, str, str], ...]:
    """Stable runtime identity for deciding whether socket rotation is needed."""

    return tuple(
        sorted(
            (
                market.pair_key,
                market.slug,
                market.yes_token,
                market.no_token,
            )
            for market in markets
        )
    )


class ApplicationRuntime:
    """Own discovery rotation, WebSocket generation, and graceful shutdown."""

    def __init__(
        self,
        services: ApplicationServices,
        *,
        websocket_factory: WebSocketFactory = _default_websocket_factory,
        discovery_interval_seconds: Optional[float] = None,
    ) -> None:
        self.services = services
        self.websocket_factory = websocket_factory

        configured_interval = (
            services.settings.market_data.discovery_cache_seconds
            if discovery_interval_seconds is None
            else discovery_interval_seconds
        )
        interval = float(configured_interval)

        if interval <= 0.0:
            raise ValueError(
                "discovery_interval_seconds must be positive"
            )

        # Operational floor only: avoid a tight error loop if discovery fails.
        self.discovery_interval_seconds = max(1.0, interval)

        self._stop_event = asyncio.Event()

        self._active_markets: Tuple[
            MarketDefinition,
            ...,
        ] = ()
        self._active_market_identity: Tuple[
            tuple[str, str, str, str],
            ...,
        ] = ()

        self._websocket: Optional[
            MarketWebSocketClient
        ] = None
        self._websocket_task: Optional[
            asyncio.Task
        ] = None

        self._started = False

    @property
    def active_markets(
        self,
    ) -> Tuple[MarketDefinition, ...]:
        return self._active_markets

    def request_stop(self) -> None:
        self._stop_event.set()

    async def _discover_markets(
        self,
    ) -> Tuple[MarketDefinition, ...]:
        return await asyncio.to_thread(
            self.services.discovery.discover_active_markets
        )

    async def _stop_websocket_generation(
        self,
    ) -> None:
        websocket = self._websocket
        task = self._websocket_task

        self._websocket = None
        self._websocket_task = None

        if websocket is not None:
            try:
                await websocket.stop()
            except Exception as exc:
                runtime_print(
                    "[main] websocket stop error:",
                    f"{type(exc).__name__}: {exc}",
                )

        if task is not None and not task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                task.cancel()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

        if task is not None:
            await asyncio.gather(
                task,
                return_exceptions=True,
            )

    def _websocket_running(self) -> bool:
        task = self._websocket_task
        return bool(
            task is not None
            and not task.done()
        )

    async def _start_websocket_generation(
        self,
        markets: Sequence[MarketDefinition],
    ) -> None:
        normalized = tuple(markets)
        if not normalized:
            return

        websocket = self.websocket_factory(
            self.services.orderbooks,
            self.services.settings,
            self.services.engine.handle_market_event,
        )
        websocket.configure_markets(normalized)

        task = asyncio.create_task(
            websocket.run(),
            name="market-websocket",
        )

        self._websocket = websocket
        self._websocket_task = task

        runtime_print(
            "[main] websocket generation started",
            f"markets={len(normalized)}",
            f"tokens={len(websocket.subscribed_tokens)}",
        )

    async def _replace_market_generation(
        self,
        markets: Sequence[MarketDefinition],
    ) -> None:
        normalized = tuple(markets)
        identity = _market_set_identity(normalized)

        await self._stop_websocket_generation()

        # Execution prewarm remains disabled in the portfolio entrypoint.
        await self.services.engine.configure_markets(
            normalized,
            prepare_execution=False,
        )

        self._active_markets = normalized
        self._active_market_identity = identity

        if normalized:
            await self._start_websocket_generation(
                normalized
            )

        runtime_print(
            "[main] active market generation",
            f"markets={len(normalized)}",
        )

    async def _refresh_market_generation(
        self,
    ) -> None:
        try:
            markets = await self._discover_markets()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            runtime_print(
                "[main] market discovery error:",
                f"{type(exc).__name__}: {exc}",
            )
            return

        identity = _market_set_identity(markets)

        changed = identity != self._active_market_identity
        socket_missing = bool(
            markets and not self._websocket_running()
        )

        if changed or socket_missing:
            await self._replace_market_generation(
                markets
            )

    async def start(self) -> None:
        if self._started:
            return

        self._started = True

        try:
            await self.services.orderbooks.start()
            await self.services.engine.start()
            await self._refresh_market_generation()
        except BaseException:
            await self.close()
            raise

    async def run(self) -> None:
        await self.start()

        runtime_print(
            "[main] portfolio runtime started",
            "mode=OBSERVE_ONLY",
        )

        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=(
                            self.discovery_interval_seconds
                        ),
                    )
                except asyncio.TimeoutError:
                    pass

                if self._stop_event.is_set():
                    break

                await self._refresh_market_generation()
        finally:
            await self.close()

    async def close(self) -> None:
        if not self._started:
            return

        self._started = False
        self._stop_event.set()

        await self._stop_websocket_generation()

        try:
            await self.services.engine.close()
        finally:
            try:
                await self.services.execution.close()
            finally:
                await self.services.orderbooks.close()

        session = getattr(
            self.services.discovery,
            "session",
            None,
        )
        if session is not None:
            try:
                await asyncio.to_thread(session.close)
            except Exception:
                pass

        runtime_print(
            "[main] portfolio runtime stopped"
        )


def _install_process_signal_handlers(
    runtime: ApplicationRuntime,
) -> None:
    """Install cooperative process shutdown where the platform supports it."""

    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        runtime_print(
            "[main] shutdown requested"
        )
        runtime.request_stop()

    for sig in (
        signal.SIGINT,
        signal.SIGTERM,
    ):
        try:
            loop.add_signal_handler(
                sig,
                _request_stop,
            )
        except (
            NotImplementedError,
            RuntimeError,
            ValueError,
        ):
            # Platforms without asyncio signal support still receive
            # KeyboardInterrupt through asyncio.run().
            pass


async def async_main(
    *,
    app_settings: AppSettings = settings,
    candidate_producer: Optional[
        CandidateProducer
    ] = None,
    enrichers: Sequence[CandidateEnricher] = (),
    risk_limits: Optional[RiskLimits] = None,
    public_policy_config: Optional[
        PublicPolicyConfig
    ] = None,
    result_listener: Optional[
        PipelineListener
    ] = None,
) -> None:
    services = build_application(
        app_settings=app_settings,
        candidate_producer=candidate_producer,
        enrichers=enrichers,
        risk_limits=risk_limits,
        public_policy_config=(
            public_policy_config
        ),
        result_listener=result_listener,
    )

    runtime = ApplicationRuntime(services)
    _install_process_signal_handlers(runtime)

    await runtime.run()


def main() -> None:
    """CLI entrypoint for the sanitized observe-only portfolio application."""

    install_disconnect_resilience()
    runtime_logger.start()

    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        runtime_print(
            "[main] interrupted"
        )
    finally:
        runtime_logger.stop()


if __name__ == "__main__":
    main()
