"""
Regression tests for src.execution.reconciliation.

The suite exercises the real ReconciliationService, OrderLifecycleService, and
FillAccounting implementations. Only venue/network evidence readers are replaced
with small boundary doubles.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from src.execution.fill_accounting import (
    FillAccounting,
    WalletBaseline,
)
from src.execution.order_lifecycle import OrderLifecycleService
from src.execution.reconciliation import (
    ReconciliationPolicy,
    ReconciliationService,
)
from src.execution.types import (
    LifecycleIdentity,
    OrderIntent,
    OrderIntentRole,
    OrderLifecycleState,
    OrderSide,
    ReconciliationOutcome,
    SubmissionOutcome,
)
from src.market.types import OutcomeSide


class FakeTransport:
    """Minimal venue boundary used by lifecycle + reconciliation."""

    def __init__(
        self,
        *,
        post_response=None,
        post_error: Exception | None = None,
        enter_raw_post: bool = True,
        status_responses=None,
        status_delay_seconds: float = 0.0,
    ) -> None:
        self.post_response = (
            {"orderID": "oid-1", "status": "LIVE", "size_matched": "0"}
            if post_response is None
            else post_response
        )
        self.post_error = post_error
        self.enter_raw_post = enter_raw_post

        self.status_responses = list(
            status_responses
            if status_responses is not None
            else (
                {"orderID": "oid-1", "status": "LIVE", "size_matched": "0"},
            )
        )
        self.status_delay_seconds = float(status_delay_seconds)

        self.post_calls = 0
        self.status_calls = 0
        self.status_order_ids: list[str] = []
        self._status_gate = threading.Lock()

    def post_order(
        self,
        signed_order,
        *,
        pre_submit_context=None,
        raw_post_enter_observer=None,
        **kwargs,
    ):
        self.post_calls += 1

        if self.enter_raw_post and raw_post_enter_observer is not None:
            raw_post_enter_observer(pre_submit_context)

        if self.post_error is not None:
            raise self.post_error

        return self.post_response

    def get_order(self, order_id: str):
        with self._status_gate:
            self.status_calls += 1
            self.status_order_ids.append(order_id)
            index = min(
                self.status_calls - 1,
                len(self.status_responses) - 1,
            )
            payload = self.status_responses[index]

        if self.status_delay_seconds > 0.0:
            time.sleep(self.status_delay_seconds)

        if isinstance(payload, Exception):
            raise payload

        return payload


class SequenceReader:
    """Sync reader returning one configured value per call."""

    def __init__(self, values) -> None:
        self.values = list(values)
        self.calls = 0
        self.args = []

    def __call__(self, value):
        self.calls += 1
        self.args.append(value)
        index = min(self.calls - 1, len(self.values) - 1)
        result = self.values[index]

        if isinstance(result, Exception):
            raise result

        return result


def run(coro):
    return asyncio.run(coro)


def make_intent(
    *,
    lifecycle_id: str = "life-1",
    attempt_id: str = "attempt-1",
    generation: int = 0,
    token_id: str = "token-yes",
    size: float = 5.0,
) -> OrderIntent:
    return OrderIntent(
        token_id=token_id,
        market_id="market-1",
        outcome_side=OutcomeSide.YES,
        order_side=OrderSide.BUY,
        price=0.45,
        size=size,
        role=OrderIntentRole.OPENING,
        lifecycle=LifecycleIdentity(
            lifecycle_id,
            attempt_id,
            generation,
        ),
    )


def build_known_oid(
    *,
    status_responses,
    post_response=None,
    trade_reader=None,
    wallet_reader=None,
    terminal_proof_reader=None,
    policy=None,
    size: float = 5.0,
):
    transport = FakeTransport(
        post_response=(
            post_response
            or {
                "orderID": "oid-1",
                "status": "LIVE",
                "size_matched": "0",
            }
        ),
        status_responses=status_responses,
    )
    lifecycle = OrderLifecycleService(transport)
    accounting = FillAccounting()
    intent = make_intent(size=size)

    submission = lifecycle.submit(
        intent,
        {"signed": True},
    )
    assert submission.order_id == "oid-1"

    service = ReconciliationService(
        transport=transport,
        lifecycle=lifecycle,
        accounting=accounting,
        trade_reader=trade_reader,
        wallet_balance_reader=wallet_reader,
        terminal_proof_reader=terminal_proof_reader,
        policy=(
            policy
            or ReconciliationPolicy(
                attempts=1,
                retry_delay_seconds=0.0,
            )
        ),
    )

    return transport, lifecycle, accounting, service, intent


def build_no_oid(
    *,
    wallet_reader=None,
    order_locator=None,
    status_responses=None,
    policy=None,
):
    transport = FakeTransport(
        post_error=TimeoutError("raw POST response lost"),
        enter_raw_post=True,
        status_responses=(
            status_responses
            if status_responses is not None
            else (
                {"orderID": "oid-recovered", "status": "LIVE", "size_matched": "0"},
            )
        ),
    )
    lifecycle = OrderLifecycleService(transport)
    accounting = FillAccounting()
    intent = make_intent(
        lifecycle_id="life-no-oid",
        attempt_id="attempt-no-oid",
    )

    submission = lifecycle.submit(
        intent,
        {"signed": True},
    )
    assert submission.outcome is SubmissionOutcome.UNKNOWN
    assert submission.post_call_entered
    assert submission.order_id is None

    service = ReconciliationService(
        transport=transport,
        lifecycle=lifecycle,
        accounting=accounting,
        wallet_balance_reader=wallet_reader,
        order_locator=order_locator,
        policy=(
            policy
            or ReconciliationPolicy(
                attempts=1,
                retry_delay_seconds=0.0,
            )
        ),
    )

    return transport, lifecycle, accounting, service, intent


def test_exact_working_zero_remains_working_owner() -> None:
    transport, lifecycle, accounting, service, intent = build_known_oid(
        status_responses=(
            {"orderID": "oid-1", "status": "LIVE", "size_matched": "0"},
        ),
    )

    result = run(service.reconcile(intent.lifecycle))

    assert result.outcome is ReconciliationOutcome.WORKING
    assert result.order_id == "oid-1"
    assert result.fill_summary.filled_size == pytest.approx(0.0)

    current = lifecycle.get(intent.lifecycle)
    assert current is not None
    assert current.state is OrderLifecycleState.WORKING
    assert current.owns_execution

    assert transport.status_calls == 1
    assert accounting.get("oid-1") is not None


def test_unknown_status_is_not_zero_even_when_wallet_is_zero() -> None:
    wallet = SequenceReader((0.0,))
    transport, lifecycle, accounting, service, intent = build_known_oid(
        status_responses=({},),
        wallet_reader=wallet,
        policy=ReconciliationPolicy(
            attempts=2,
            retry_delay_seconds=0.0,
        ),
    )

    result = run(service.reconcile(intent.lifecycle))

    assert result.outcome is ReconciliationOutcome.STILL_AMBIGUOUS
    assert result.evidence["status_observed"] is False
    assert result.evidence["terminal_proven"] is False

    # Wallet is not consulted to turn unknown exact-order status into zero.
    assert wallet.calls == 0
    assert transport.status_calls == 2

    current = lifecycle.get(intent.lifecycle)
    assert current is not None
    assert current.state is OrderLifecycleState.RECONCILING
    assert current.owns_execution


def test_foreign_status_oid_is_rejected_before_accounting() -> None:
    transport, lifecycle, accounting, service, intent = build_known_oid(
        status_responses=(
            {
                "orderID": "oid-other",
                "status": "FILLED",
                "size_matched": "5",
                "average_price": "0.40",
            },
        ),
    )

    result = run(service.reconcile(intent.lifecycle))

    assert result.outcome is ReconciliationOutcome.STILL_AMBIGUOUS
    assert "status_order_id_mismatch" in result.evidence["notes"]
    assert result.evidence["accounting_filled_size"] == pytest.approx(0.0)

    record = accounting.get("oid-1")
    assert record is not None
    assert record.confirmed_filled_size == pytest.approx(0.0)

    current = lifecycle.get(intent.lifecycle)
    assert current is not None
    assert current.order_id == "oid-1"
    assert current.state is OrderLifecycleState.RECONCILING


def test_foreign_status_oid_can_retry_to_correct_exact_oid() -> None:
    transport, lifecycle, accounting, service, intent = build_known_oid(
        status_responses=(
            {
                "orderID": "oid-other",
                "status": "FILLED",
                "size_matched": "5",
            },
            {
                "orderID": "oid-1",
                "status": "LIVE",
                "size_matched": "0",
            },
        ),
        policy=ReconciliationPolicy(
            attempts=2,
            retry_delay_seconds=0.0,
        ),
    )

    result = run(service.reconcile(intent.lifecycle))

    assert result.outcome is ReconciliationOutcome.WORKING
    assert "status_order_id_mismatch" in result.evidence["notes"]
    assert transport.status_calls == 2


def test_no_oid_positive_wallet_delta_remains_ambiguous_without_synthetic_oid() -> None:
    wallet = SequenceReader((3.0,))
    transport, lifecycle, accounting, service, intent = build_no_oid(
        wallet_reader=wallet,
    )
    baseline = WalletBaseline(
        token_id=intent.token_id,
        balance=0.0,
        observed_at=time.time() - 1.0,
        valid=True,
    )

    result = run(
        service.reconcile(
            intent.lifecycle,
            wallet_baseline=baseline,
        )
    )

    assert result.outcome is ReconciliationOutcome.STILL_AMBIGUOUS
    assert result.order_id is None
    assert result.fill_summary is None
    assert result.evidence["wallet_baseline_valid"] is True
    assert result.evidence["wallet_delta"] == pytest.approx(3.0)
    assert (
        "positive_wallet_delta_without_exact_order_id"
        in result.evidence["notes"]
    )

    assert accounting.records() == ()
    current = lifecycle.get(intent.lifecycle)
    assert current is not None
    assert current.order_id is None
    assert current.owns_execution


def test_no_oid_zero_wallet_delta_does_not_prove_no_venue_order() -> None:
    wallet = SequenceReader((2.0,))
    transport, lifecycle, accounting, service, intent = build_no_oid(
        wallet_reader=wallet,
    )
    baseline = WalletBaseline(
        token_id=intent.token_id,
        balance=2.0,
        observed_at=time.time() - 1.0,
        valid=True,
    )

    result = run(
        service.reconcile(
            intent.lifecycle,
            wallet_baseline=baseline,
        )
    )

    assert result.outcome is ReconciliationOutcome.STILL_AMBIGUOUS
    assert result.order_id is None
    assert result.evidence["wallet_delta"] == pytest.approx(0.0)
    assert (
        "zero_wallet_delta_does_not_prove_no_venue_order"
        in result.evidence["notes"]
    )


def test_id_only_locator_queries_status_in_same_attempt() -> None:
    locator = SequenceReader(("oid-recovered",))

    transport, lifecycle, accounting, service, intent = build_no_oid(
        order_locator=locator,
        status_responses=(
            {
                "orderID": "oid-recovered",
                "status": "LIVE",
                "size_matched": "0",
            },
        ),
        policy=ReconciliationPolicy(
            attempts=1,
            retry_delay_seconds=0.0,
        ),
    )

    result = run(service.reconcile(intent.lifecycle))

    assert result.outcome is ReconciliationOutcome.WORKING
    assert result.order_id == "oid-recovered"
    assert transport.status_calls == 1
    assert transport.status_order_ids == ["oid-recovered"]

    current = lifecycle.get(intent.lifecycle)
    assert current is not None
    assert current.order_id == "oid-recovered"
    assert current.state is OrderLifecycleState.WORKING


def test_status_bearing_locator_avoids_redundant_status_read() -> None:
    locator = SequenceReader(
        (
            {
                "orderID": "oid-recovered",
                "status": "LIVE",
                "size_matched": "0",
            },
        )
    )

    transport, lifecycle, accounting, service, intent = build_no_oid(
        order_locator=locator,
    )

    result = run(service.reconcile(intent.lifecycle))

    assert result.outcome is ReconciliationOutcome.WORKING
    assert result.order_id == "oid-recovered"
    assert transport.status_calls == 0


def test_cancelled_zero_without_wallet_proof_stays_ambiguous() -> None:
    transport, lifecycle, accounting, service, intent = build_known_oid(
        status_responses=(
            {
                "orderID": "oid-1",
                "status": "CANCELLED",
                "size_matched": "0",
            },
        ),
    )

    result = run(service.reconcile(intent.lifecycle))

    assert result.outcome is ReconciliationOutcome.STILL_AMBIGUOUS
    assert result.evidence["terminal_proven"] is True
    assert result.evidence["wallet_observed"] is False

    current = lifecycle.get(intent.lifecycle)
    assert current is not None
    # Ambiguity preserves reconciliation ownership rather than terminal-zero.
    assert current.state is OrderLifecycleState.RECONCILING
    assert current.owns_execution
    assert not accounting.get("oid-1").resolved


def test_cancelled_zero_plus_fresh_baselineless_absolute_zero_resolves() -> None:
    wallet = SequenceReader((0.0,))
    transport, lifecycle, accounting, service, intent = build_known_oid(
        status_responses=(
            {
                "orderID": "oid-1",
                "status": "CANCELLED",
                "size_matched": "0",
            },
        ),
        wallet_reader=wallet,
    )

    result = run(service.reconcile(intent.lifecycle))

    assert result.outcome is ReconciliationOutcome.CANCELLED_ZERO_FILL
    assert result.evidence["baselineless_absolute_zero"] is True
    assert result.evidence["wallet_balance"] == pytest.approx(0.0)

    record = accounting.get("oid-1")
    assert record is not None
    assert record.resolved
    assert record.confirmed_filled_size == pytest.approx(0.0)

    current = lifecycle.get(intent.lifecycle)
    assert current is not None
    assert current.state is OrderLifecycleState.CANCELLED
    assert lifecycle.release_terminal_zero(intent.lifecycle)


def test_cancelled_zero_plus_baselineless_nonzero_wallet_stays_ambiguous() -> None:
    wallet = SequenceReader((2.0,))
    transport, lifecycle, accounting, service, intent = build_known_oid(
        status_responses=(
            {
                "orderID": "oid-1",
                "status": "CANCELLED",
                "size_matched": "0",
            },
        ),
        wallet_reader=wallet,
    )

    result = run(service.reconcile(intent.lifecycle))

    assert result.outcome is ReconciliationOutcome.STILL_AMBIGUOUS
    assert result.evidence["wallet_balance"] == pytest.approx(2.0)
    assert result.evidence["wallet_baseline_valid"] is False
    assert (
        "baselineless_nonzero_wallet_is_ambiguous"
        in result.evidence["notes"]
    )
    assert not accounting.get("oid-1").resolved


def test_valid_baseline_zero_delta_completes_terminal_zero_proof() -> None:
    wallet = SequenceReader((4.0,))
    transport, lifecycle, accounting, service, intent = build_known_oid(
        status_responses=(
            {
                "orderID": "oid-1",
                "status": "CANCELLED",
                "size_matched": "0",
            },
        ),
        wallet_reader=wallet,
    )
    baseline = WalletBaseline(
        token_id=intent.token_id,
        balance=4.0,
        observed_at=time.time() - 1.0,
        valid=True,
    )

    result = run(
        service.reconcile(
            intent.lifecycle,
            wallet_baseline=baseline,
        )
    )

    assert result.outcome is ReconciliationOutcome.CANCELLED_ZERO_FILL
    assert result.evidence["wallet_baseline_valid"] is True
    assert result.evidence["wallet_delta"] == pytest.approx(0.0)


def test_valid_baseline_positive_delta_overrides_cancelled_zero_candidate() -> None:
    wallet = SequenceReader((2.0,))
    transport, lifecycle, accounting, service, intent = build_known_oid(
        status_responses=(
            {
                "orderID": "oid-1",
                "status": "CANCELLED",
                "size_matched": "0",
            },
        ),
        wallet_reader=wallet,
    )
    baseline = WalletBaseline(
        token_id=intent.token_id,
        balance=0.0,
        observed_at=time.time() - 1.0,
        valid=True,
    )

    result = run(
        service.reconcile(
            intent.lifecycle,
            wallet_baseline=baseline,
        )
    )

    assert result.outcome is ReconciliationOutcome.PARTIALLY_FILLED
    assert result.fill_summary.filled_size == pytest.approx(2.0)
    assert result.evidence["wallet_delta"] == pytest.approx(2.0)

    current = lifecycle.get(intent.lifecycle)
    assert current is not None
    # Cancel-raced partial fill preserves terminal order state.
    assert current.state is OrderLifecycleState.CANCELLED
    assert current.working_order.filled_size == pytest.approx(2.0)
    assert current.owns_execution
    assert not lifecycle.release_terminal_zero(intent.lifecycle)


def test_exact_trade_after_cancelled_zero_overrides_zero_before_wallet() -> None:
    trade = SequenceReader(
        (
            {
                "fills": [
                    {
                        "id": "fill-1",
                        "size": "2",
                        "price": "0.44",
                    }
                ]
            },
        )
    )
    wallet = SequenceReader((0.0,))

    transport, lifecycle, accounting, service, intent = build_known_oid(
        status_responses=(
            {
                "orderID": "oid-1",
                "status": "CANCELLED",
                "size_matched": "0",
            },
        ),
        trade_reader=trade,
        wallet_reader=wallet,
    )

    result = run(service.reconcile(intent.lifecycle))

    assert result.outcome is ReconciliationOutcome.PARTIALLY_FILLED
    assert result.fill_summary.filled_size == pytest.approx(2.0)
    assert result.fill_summary.average_price == pytest.approx(0.44)
    assert result.evidence["trade_observed"] is True
    assert wallet.calls == 0

    current = lifecycle.get(intent.lifecycle)
    assert current is not None
    assert current.state is OrderLifecycleState.CANCELLED
    assert current.working_order.filled_size == pytest.approx(2.0)


def test_trade_endpoint_can_upgrade_positive_status_to_full_fill() -> None:
    trade = SequenceReader(
        (
            {
                "fills": [
                    {"id": "fill-a", "size": "2", "price": "0.40"},
                    {"id": "fill-b", "size": "3", "price": "0.50"},
                ]
            },
        )
    )

    transport, lifecycle, accounting, service, intent = build_known_oid(
        status_responses=(
            {
                "orderID": "oid-1",
                "status": "LIVE",
                "size_matched": "2",
                "average_price": "0.40",
            },
        ),
        trade_reader=trade,
        size=5.0,
    )

    result = run(service.reconcile(intent.lifecycle))

    assert result.outcome is ReconciliationOutcome.FILLED
    assert result.fill_summary.filled_size == pytest.approx(5.0)
    assert result.fill_summary.average_price == pytest.approx(0.46)
    assert trade.calls == 1

    current = lifecycle.get(intent.lifecycle)
    assert current is not None
    assert current.state is OrderLifecycleState.FILLED
    assert current.working_order.filled_size == pytest.approx(5.0)


def test_existing_positive_accounting_evidence_beats_later_zero_status() -> None:
    transport, lifecycle, accounting, service, intent = build_known_oid(
        status_responses=(
            {
                "orderID": "oid-1",
                "status": "CANCELLED",
                "size_matched": "0",
            },
        ),
        policy=ReconciliationPolicy(
            attempts=1,
            retry_delay_seconds=0.0,
            query_trades_after_positive_status=False,
            query_trades_on_terminal_zero_candidate=False,
        ),
    )

    accounting.register_order(
        order_id="oid-1",
        token_id=intent.token_id,
        lifecycle=intent.lifecycle,
        order_side=intent.order_side,
        requested_size=intent.size,
        submitted_limit_price=intent.price,
        created_at=intent.created_at,
    )
    accounting.observe_trade_payload(
        order_id="oid-1",
        payload={
            "fills": [
                {"id": "historic-fill", "size": "1", "price": "0.43"},
            ]
        },
    )

    result = run(service.reconcile(intent.lifecycle))

    assert result.outcome is ReconciliationOutcome.PARTIALLY_FILLED
    assert result.fill_summary.filled_size == pytest.approx(1.0)

    current = lifecycle.get(intent.lifecycle)
    assert current is not None
    # Reconciliation sees the terminal status snapshot and preserves it for a
    # partial fill even though the positive evidence came from accounting.
    assert current.state is OrderLifecycleState.CANCELLED
    assert current.working_order.filled_size == pytest.approx(1.0)


def test_rejected_terminal_zero_uses_rejected_zero_outcome() -> None:
    wallet = SequenceReader((0.0,))
    transport, lifecycle, accounting, service, intent = build_known_oid(
        status_responses=(
            {
                "orderID": "oid-1",
                "status": "REJECTED",
                "size_matched": "0",
            },
        ),
        wallet_reader=wallet,
    )

    result = run(service.reconcile(intent.lifecycle))

    assert result.outcome is ReconciliationOutcome.REJECTED_ZERO_FILL

    current = lifecycle.get(intent.lifecycle)
    assert current is not None
    assert current.state is OrderLifecycleState.REJECTED
    assert lifecycle.release_terminal_zero(intent.lifecycle)


def test_filled_label_with_zero_matched_is_not_relabelled_rejected_zero() -> None:
    wallet = SequenceReader((0.0,))
    terminal_proof = SequenceReader((True,))

    transport, lifecycle, accounting, service, intent = build_known_oid(
        status_responses=(
            {
                "orderID": "oid-1",
                "status": "FILLED",
                "size_matched": "0",
            },
        ),
        wallet_reader=wallet,
        terminal_proof_reader=terminal_proof,
    )

    result = run(service.reconcile(intent.lifecycle))

    assert result.outcome is ReconciliationOutcome.STILL_AMBIGUOUS
    assert result.evidence["status_state"] == OrderLifecycleState.FILLED.value
    assert result.evidence["terminal_proven"] is False
    assert (
        "durable_terminal_proof_without_zero_classification"
        in result.evidence["notes"]
    )
    # Since the zero class is contradictory, wallet zero is not used to invent
    # CANCELLED/REJECTED semantics.
    assert wallet.calls == 0


def test_singleflight_shares_one_reconciliation_task_per_exact_lifecycle() -> None:
    transport = FakeTransport(
        status_responses=(
            {"orderID": "oid-1", "status": "LIVE", "size_matched": "0"},
        ),
        status_delay_seconds=0.05,
    )
    lifecycle = OrderLifecycleService(transport)
    accounting = FillAccounting()
    intent = make_intent()

    submission = lifecycle.submit(intent, {"signed": True})
    assert submission.order_id == "oid-1"

    service = ReconciliationService(
        transport=transport,
        lifecycle=lifecycle,
        accounting=accounting,
        policy=ReconciliationPolicy(
            attempts=1,
            retry_delay_seconds=0.0,
        ),
    )

    async def scenario():
        first, second = await asyncio.gather(
            service.reconcile(intent.lifecycle, reason="caller-a"),
            service.reconcile(intent.lifecycle, reason="caller-b"),
        )
        return first, second

    first, second = run(scenario())

    assert first.outcome is ReconciliationOutcome.WORKING
    assert second.outcome is ReconciliationOutcome.WORKING
    assert first is second
    assert transport.status_calls == 1
    assert service.active_reconciliations() == 0


def test_baseline_wallet_overfill_is_preserved_not_clamped() -> None:
    wallet = SequenceReader((7.0,))
    transport, lifecycle, accounting, service, intent = build_known_oid(
        status_responses=(
            {
                "orderID": "oid-1",
                "status": "CANCELLED",
                "size_matched": "0",
            },
        ),
        wallet_reader=wallet,
        size=5.0,
    )
    baseline = WalletBaseline(
        token_id=intent.token_id,
        balance=0.0,
        observed_at=time.time() - 1.0,
        valid=True,
    )

    result = run(
        service.reconcile(
            intent.lifecycle,
            wallet_baseline=baseline,
        )
    )

    assert result.outcome is ReconciliationOutcome.FILLED
    assert result.fill_summary.filled_size == pytest.approx(7.0)

    record = accounting.get("oid-1")
    assert record is not None
    assert record.confirmed_filled_size == pytest.approx(7.0)
    assert record.overfilled
    assert record.overfilled_size == pytest.approx(2.0)
    assert any(
        anomaly.code == "BASELINE_WALLET_OVERFILL"
        for anomaly in record.anomalies
    )


def test_smaller_trade_payload_cannot_regress_stronger_status_evidence() -> None:
    trade = SequenceReader(
        (
            {
                "fills": [
                    {"id": "small-trade", "size": "1", "price": "0.41"},
                ]
            },
        )
    )

    transport, lifecycle, accounting, service, intent = build_known_oid(
        status_responses=(
            {
                "orderID": "oid-1",
                "status": "LIVE",
                "size_matched": "3",
                "average_price": "0.44",
            },
        ),
        trade_reader=trade,
        size=5.0,
    )

    result = run(service.reconcile(intent.lifecycle))

    assert result.outcome is ReconciliationOutcome.PARTIALLY_FILLED
    assert result.fill_summary.filled_size == pytest.approx(3.0)
    assert result.fill_summary.average_price == pytest.approx(0.44)

    record = accounting.get("oid-1")
    assert record is not None
    assert record.confirmed_filled_size == pytest.approx(3.0)
    assert record.realized_average_price == pytest.approx(0.44)


def test_baselineless_absolute_zero_can_be_disabled_by_policy() -> None:
    wallet = SequenceReader((0.0,))
    transport, lifecycle, accounting, service, intent = build_known_oid(
        status_responses=(
            {
                "orderID": "oid-1",
                "status": "CANCELLED",
                "size_matched": "0",
            },
        ),
        wallet_reader=wallet,
        policy=ReconciliationPolicy(
            attempts=1,
            retry_delay_seconds=0.0,
            allow_baselineless_absolute_zero=False,
        ),
    )

    result = run(service.reconcile(intent.lifecycle))

    assert result.outcome is ReconciliationOutcome.STILL_AMBIGUOUS
    assert result.evidence["wallet_balance"] == pytest.approx(0.0)
    assert result.evidence["baselineless_absolute_zero"] is False
    assert not accounting.get("oid-1").resolved
