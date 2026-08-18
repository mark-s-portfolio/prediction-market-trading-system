"""Regression coverage for generic infrastructure hardening."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.execution.clob_transport import (
    ClobTransport,
    PreSubmitContext,
    PreSubmitRejected,
    ValidationResult,
)
from src.execution.fill_accounting import extract_incremental_fills
from src.execution.types import (
    LifecycleIdentity,
    OrderIntent,
    OrderIntentRole,
    OrderSide,
    SubmissionOutcome,
    SubmissionResult,
)
from src.market.orderbook import OrderBookStore
from src.market.types import OutcomeSide
from src.market.websocket import (
    MarketDataEvent,
    MarketDataEventType,
    MarketWebSocketClient,
    WebSocketPolicy,
)
from src.risk.position_state import (
    ExecutionPriceQuality,
    OrderAffinity,
    PositionBook,
)


def make_intent() -> OrderIntent:
    return OrderIntent(
        token_id="token-yes",
        market_id="market-1",
        outcome_side=OutcomeSide.YES,
        order_side=OrderSide.BUY,
        price=0.45,
        size=5.0,
        role=OrderIntentRole.OPENING,
        lifecycle=LifecycleIdentity("life-1", "attempt-1", 0),
    )


def test_top_only_update_drops_stale_ladder_depth() -> None:
    store = OrderBookStore()
    store.publish_ws_book(
        token_id="token",
        bids=((0.50, 10.0), (0.49, 20.0)),
        asks=((0.51, 10.0), (0.52, 20.0)),
        timestamp=100.0,
    )

    top = store.publish_ws_top(
        token_id="token",
        best_bid=0.48,
        best_ask=0.53,
        timestamp=101.0,
    )

    assert top is not None
    assert top.best_bid.price == pytest.approx(0.48)
    assert top.best_ask.price == pytest.approx(0.53)
    assert len(top.bids) == 1
    assert len(top.asks) == 1
    assert not top.depth_proven
    assert top.synthetic_depth


def test_native_fill_id_deduplicates_across_payload_aliases() -> None:
    payload = {
        "fills": [{"id": "exec-1", "size": "1", "price": "0.44"}],
        "trades": [{"id": "exec-1", "size": "1", "price": "0.44"}],
    }

    fills = extract_incremental_fills(
        payload,
        order_id="oid-1",
        token_id="token-yes",
        side=OrderSide.BUY,
        observed_at=100.0,
    )

    assert len(fills) == 1
    assert fills[0].size == pytest.approx(1.0)


def test_late_exact_buy_price_hydrates_confirmed_unpriced_inventory() -> None:
    book = PositionBook()
    affinity = OrderAffinity(
        order_id="oid-1",
        lifecycle=LifecycleIdentity("life-1", "attempt-1", 0),
        token_id="token-yes",
        market_id="market-1",
        outcome_side=OutcomeSide.YES,
        order_side=OrderSide.BUY,
    )

    first = book.apply_cumulative_execution(
        affinity=affinity,
        cumulative_size=5.0,
        cumulative_average_price=None,
        price_quality=ExecutionPriceQuality.UNKNOWN,
        observed_at=100.0,
    )
    assert first.quantity == pytest.approx(5.0)
    assert first.priced_quantity == pytest.approx(0.0)

    hydrated = book.apply_cumulative_execution(
        affinity=affinity,
        cumulative_size=5.0,
        cumulative_average_price=0.40,
        price_quality=ExecutionPriceQuality.VENUE_AVERAGE,
        observed_at=101.0,
    )

    assert hydrated.quantity == pytest.approx(5.0)
    assert hydrated.priced_quantity == pytest.approx(5.0)
    assert hydrated.cost_basis == pytest.approx(2.0)
    assert hydrated.average_cost == pytest.approx(0.40)


def test_failed_before_submit_never_requests_reconciliation() -> None:
    result = SubmissionResult(
        outcome=SubmissionOutcome.FAILED_BEFORE_SUBMIT,
        intent=make_intent(),
        post_call_entered=True,  # contradictory legacy telemetry must not win
    )
    assert not result.requires_reconciliation


class _DenyValidator:
    def validate(self, context: PreSubmitContext) -> ValidationResult:
        return ValidationResult(False, reason="deny before network")


class _RawClient:
    def __init__(self) -> None:
        self.posts = 0

    def post_order(self, signed_order):
        self.posts += 1
        return {"orderID": "oid-1"}


def test_pre_submit_rejection_does_not_advance_network_timestamp() -> None:
    raw = _RawClient()
    transport = ClobTransport(
        raw,
        pre_submit_validator=_DenyValidator(),
    )
    before = transport._last_network_call_ts

    with pytest.raises(PreSubmitRejected):
        transport.post_order(
            {"signed": True},
            pre_submit_context=PreSubmitContext(
                token_id="token-yes",
                market_id="market-1",
                lifecycle_id="life-1",
                attempt_id="attempt-1",
            ),
        )

    assert raw.posts == 0
    assert transport._last_network_call_ts == before


def test_slow_async_handler_does_not_block_socket_event_ingress() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def handler(event: MarketDataEvent) -> None:
            started.set()
            await release.wait()

        client = MarketWebSocketClient(
            OrderBookStore(),
            policy=WebSocketPolicy(
                event_queue_max=4,
                event_shutdown_drain_seconds=0.01,
            ),
            event_handler=handler,
        )

        started_at = time.perf_counter()
        await client._emit(
            MarketDataEvent(
                event_type=MarketDataEventType.CONNECTION,
                detail="test",
            )
        )
        elapsed = time.perf_counter() - started_at

        assert elapsed < 0.05
        await asyncio.wait_for(started.wait(), timeout=0.2)

        release.set()
        await client._shutdown_event_worker()

    asyncio.run(scenario())
