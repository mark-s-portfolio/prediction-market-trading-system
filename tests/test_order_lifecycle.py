"""
Regression tests for src.execution.order_lifecycle.

These tests focus on ownership/provenance invariants rather than venue-specific
strategy behavior.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.execution.order_lifecycle import (
    LifecycleConflict,
    OrderLifecycleService,
    StaleLifecycleGeneration,
)
from src.execution.types import (
    CancellationOutcome,
    FillSummary,
    LifecycleIdentity,
    OrderIntent,
    OrderIntentRole,
    OrderLifecycleState,
    OrderSide,
    ReconciliationOutcome,
    ReconciliationResult,
    SubmissionOutcome,
    VenueOrderSnapshot,
)
from src.market.types import OutcomeSide


class FakeTransport:
    """Minimal transport double preserving the raw-enter ordering contract."""

    def __init__(
        self,
        *,
        post_response=None,
        post_error: Exception | None = None,
        enter_raw_post: bool = True,
        cancel_response=True,
        cancel_error: Exception | None = None,
        status_response=None,
    ) -> None:
        self.post_response = (
            {"orderID": "oid-1", "status": "LIVE", "size_matched": "0"}
            if post_response is None
            else post_response
        )
        self.post_error = post_error
        self.enter_raw_post = enter_raw_post

        self.cancel_response = cancel_response
        self.cancel_error = cancel_error

        self.status_response = (
            {"orderID": "oid-1", "status": "LIVE", "size_matched": "0"}
            if status_response is None
            else status_response
        )

        self.post_calls = 0
        self.cancel_calls = 0
        self.status_calls = 0

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

    def cancel_order(self, order_id):
        self.cancel_calls += 1

        if self.cancel_error is not None:
            raise self.cancel_error

        return self.cancel_response

    def get_order(self, order_id):
        self.status_calls += 1
        return self.status_response


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


def submit_working(
    service: OrderLifecycleService,
    intent: OrderIntent,
    *,
    order_id: str = "oid-1",
) -> None:
    service.transport.post_response = {
        "orderID": order_id,
        "status": "LIVE",
        "size_matched": "0",
    }

    result = service.submit(intent, {"signed": True})

    assert result.outcome is SubmissionOutcome.CONFIRMED_WORKING
    assert result.order_id == order_id


def test_confirmed_submit_binds_exact_oid_and_owner() -> None:
    transport = FakeTransport()
    service = OrderLifecycleService(transport)
    intent = make_intent()

    result = service.submit(intent, {"signed": True})
    snapshot = service.get(intent.lifecycle)

    assert result.outcome is SubmissionOutcome.CONFIRMED_WORKING
    assert result.post_call_entered
    assert result.order_id == "oid-1"

    assert snapshot is not None
    assert snapshot.raw_post_entered
    assert snapshot.order_id == "oid-1"
    assert snapshot.state is OrderLifecycleState.WORKING
    assert snapshot.owns_execution

    by_oid = service.get_by_order_id("oid-1")
    assert by_oid is not None
    assert by_oid.lifecycle == intent.lifecycle


def test_raw_post_exception_preserves_unknown_ownership() -> None:
    transport = FakeTransport(
        post_error=TimeoutError("connection dropped"),
        enter_raw_post=True,
    )
    service = OrderLifecycleService(transport)
    intent = make_intent()

    result = service.submit(intent, {"signed": True})
    snapshot = service.get(intent.lifecycle)

    assert result.outcome is SubmissionOutcome.UNKNOWN
    assert result.post_call_entered
    assert snapshot is not None
    assert snapshot.raw_post_entered
    assert snapshot.state is OrderLifecycleState.SUBMISSION_UNKNOWN
    assert snapshot.order_id is None
    assert snapshot.owns_execution


def test_success_like_response_without_oid_stays_ambiguous() -> None:
    transport = FakeTransport(
        post_response={"status": "LIVE"},
    )
    service = OrderLifecycleService(transport)
    intent = make_intent()

    result = service.submit(intent, {"signed": True})
    snapshot = service.get(intent.lifecycle)

    assert result.outcome is SubmissionOutcome.UNKNOWN
    assert result.order_id is None
    assert snapshot is not None
    assert snapshot.state is OrderLifecycleState.SUBMISSION_UNKNOWN
    assert snapshot.owns_execution


def test_failure_before_raw_post_is_not_ambiguous_and_can_release() -> None:
    transport = FakeTransport(
        post_error=RuntimeError("local transport failure"),
        enter_raw_post=False,
    )
    service = OrderLifecycleService(transport)
    intent = make_intent()

    result = service.submit(intent, {"signed": True})

    assert result.outcome is SubmissionOutcome.FAILED_BEFORE_SUBMIT
    assert not result.post_call_entered

    snapshot = service.get(intent.lifecycle)
    assert snapshot is not None
    assert snapshot.state is OrderLifecycleState.FAILED
    assert not snapshot.raw_post_entered
    assert not snapshot.owns_execution

    assert service.release_terminal_zero(intent.lifecycle)
    assert service.get(intent.lifecycle) is None


def test_exact_status_payload_cannot_switch_order_id() -> None:
    transport = FakeTransport()
    service = OrderLifecycleService(transport)
    intent = make_intent()
    submit_working(service, intent, order_id="oid-owned")

    with pytest.raises(LifecycleConflict):
        service.observe_status(
            intent.lifecycle,
            {
                "orderID": "oid-other",
                "status": "LIVE",
                "size_matched": "0",
            },
        )

    snapshot = service.get(intent.lifecycle)
    assert snapshot is not None
    assert snapshot.order_id == "oid-owned"
    assert snapshot.state is OrderLifecycleState.WORKING


def test_partial_fill_cannot_regress_to_working_on_stale_status() -> None:
    transport = FakeTransport()
    service = OrderLifecycleService(transport)
    intent = make_intent()
    submit_working(service, intent)

    service.observe_status(
        intent.lifecycle,
        {
            "orderID": "oid-1",
            "status": "LIVE",
            "size_matched": "2",
            "average_price": "0.44",
        },
    )

    first = service.get(intent.lifecycle)
    assert first is not None
    assert first.state is OrderLifecycleState.PARTIALLY_FILLED
    assert first.working_order.filled_size == pytest.approx(2.0)

    # Later-arriving stale payload reports zero matched again.
    service.observe_status(
        intent.lifecycle,
        {
            "orderID": "oid-1",
            "status": "LIVE",
            "size_matched": "0",
        },
    )

    current = service.get(intent.lifecycle)
    assert current is not None
    assert current.state is OrderLifecycleState.PARTIALLY_FILLED
    assert current.working_order.filled_size == pytest.approx(2.0)


def test_filled_state_cannot_reopen_on_late_working_status() -> None:
    transport = FakeTransport()
    service = OrderLifecycleService(transport)
    intent = make_intent(size=5.0)
    submit_working(service, intent)

    service.observe_status(
        intent.lifecycle,
        {
            "orderID": "oid-1",
            "status": "FILLED",
            "size_matched": "5",
            "average_price": "0.45",
        },
    )

    assert service.get(intent.lifecycle).state is OrderLifecycleState.FILLED

    service.observe_status(
        intent.lifecycle,
        {
            "orderID": "oid-1",
            "status": "LIVE",
            "size_matched": "0",
        },
    )

    current = service.get(intent.lifecycle)
    assert current is not None
    assert current.state is OrderLifecycleState.FILLED
    assert current.working_order.filled_size == pytest.approx(5.0)


def test_cancel_live_status_response_is_not_fake_confirmed_cancel() -> None:
    transport = FakeTransport(
        cancel_response={
            "orderID": "oid-1",
            "status": "LIVE",
            "size_matched": "0",
        }
    )
    service = OrderLifecycleService(transport)
    intent = make_intent()
    submit_working(service, intent)

    result = service.cancel(intent.lifecycle)
    snapshot = service.get(intent.lifecycle)

    assert result.outcome is CancellationOutcome.UNKNOWN
    assert snapshot is not None
    assert snapshot.state is OrderLifecycleState.CANCEL_UNKNOWN
    assert snapshot.owns_execution


def test_explicit_cancel_ack_is_terminal_but_not_zero_fill_release_proof() -> None:
    transport = FakeTransport(
        cancel_response={"canceled": ["oid-1"]},
    )
    service = OrderLifecycleService(transport)
    intent = make_intent()
    submit_working(service, intent)

    result = service.cancel(intent.lifecycle)
    snapshot = service.get(intent.lifecycle)

    assert result.outcome is CancellationOutcome.CONFIRMED_CANCELLED
    assert snapshot is not None
    assert snapshot.state is OrderLifecycleState.CANCELLED
    assert snapshot.raw_post_entered

    # Raw POST happened. Cancel acknowledgement alone is not zero-fill proof.
    assert not service.release_terminal_zero(intent.lifecycle)


def test_cancel_exception_preserves_cancel_unknown_owner() -> None:
    transport = FakeTransport(
        cancel_error=TimeoutError("cancel response lost"),
    )
    service = OrderLifecycleService(transport)
    intent = make_intent()
    submit_working(service, intent)

    result = service.cancel(intent.lifecycle)
    snapshot = service.get(intent.lifecycle)

    assert result.outcome is CancellationOutcome.UNKNOWN
    assert snapshot is not None
    assert snapshot.state is OrderLifecycleState.CANCEL_UNKNOWN
    assert snapshot.owns_execution


def test_unreleased_raw_post_terminal_zero_cannot_be_superseded() -> None:
    transport = FakeTransport(
        cancel_response={"canceled": ["oid-1"]},
    )
    service = OrderLifecycleService(transport)
    intent = make_intent()
    submit_working(service, intent)
    service.cancel(intent.lifecycle)

    replacement = replace(
        intent,
        lifecycle=intent.lifecycle.next_generation(),
    )

    with pytest.raises(LifecycleConflict):
        service.supersede(
            intent.lifecycle,
            replacement,
        )

    with pytest.raises(LifecycleConflict):
        service.create(replacement)


def test_terminal_zero_reconciliation_allows_release() -> None:
    transport = FakeTransport(
        cancel_response={"canceled": ["oid-1"]},
    )
    service = OrderLifecycleService(transport)
    intent = make_intent()
    submit_working(service, intent)
    service.cancel(intent.lifecycle)

    result = ReconciliationResult(
        outcome=ReconciliationOutcome.CANCELLED_ZERO_FILL,
        lifecycle=intent.lifecycle,
        order_id="oid-1",
        reason="exact terminal zero proof",
    )
    snapshot = service.apply_reconciliation(result)

    assert snapshot.state is OrderLifecycleState.CANCELLED
    assert snapshot.working_order.filled_size == pytest.approx(0.0)

    assert service.release_terminal_zero(intent.lifecycle)
    assert service.get(intent.lifecycle) is None
    assert service.get_by_order_id("oid-1") is None


def test_cancel_raced_partial_fill_preserves_terminal_order_state() -> None:
    transport = FakeTransport()
    service = OrderLifecycleService(transport)
    intent = make_intent(size=5.0)
    submit_working(service, intent)

    terminal_snapshot = VenueOrderSnapshot(
        order_id="oid-1",
        token_id=intent.token_id,
        order_side=intent.order_side,
        state=OrderLifecycleState.CANCELLED,
        requested_size=5.0,
        matched_size=2.0,
        limit_price=0.45,
        average_fill_price=0.44,
        raw_status="CANCELLED",
    )
    fill_summary = FillSummary(
        order_id="oid-1",
        requested_size=5.0,
        filled_size=2.0,
        average_price=0.44,
    )

    reconciled = service.apply_reconciliation(
        ReconciliationResult(
            outcome=ReconciliationOutcome.PARTIALLY_FILLED,
            lifecycle=intent.lifecycle,
            order_id="oid-1",
            snapshot=terminal_snapshot,
            fill_summary=fill_summary,
            reason="cancel-raced partial fill",
        )
    )

    assert reconciled.state is OrderLifecycleState.CANCELLED
    assert reconciled.working_order.filled_size == pytest.approx(2.0)
    # Terminal order, but economic fill keeps lifecycle ownership visible.
    assert reconciled.owns_execution
    assert not service.release_terminal_zero(intent.lifecycle)


def test_stale_generation_cannot_overwrite_newer_generation() -> None:
    transport = FakeTransport()
    service = OrderLifecycleService(transport)

    old = make_intent(
        lifecycle_id="life-generation",
        attempt_id="attempt-generation",
        generation=0,
    )
    service.create(old)

    replacement = replace(
        old,
        lifecycle=old.lifecycle.next_generation(),
    )
    service.supersede(old.lifecycle, replacement)

    with pytest.raises(StaleLifecycleGeneration):
        service.observe_status(
            old.lifecycle,
            {
                "orderID": "stale-oid",
                "status": "LIVE",
                "size_matched": "0",
            },
        )

    current = service.get(replacement.lifecycle)
    assert current is not None
    assert current.lifecycle == replacement.lifecycle
    assert current.state is OrderLifecycleState.CREATED


def test_attempt_identity_cannot_change_inside_same_lifecycle_id() -> None:
    transport = FakeTransport()
    service = OrderLifecycleService(transport)

    first = make_intent(
        lifecycle_id="life-stable",
        attempt_id="attempt-a",
        generation=0,
    )
    service.create(first)

    different_attempt = replace(
        first,
        lifecycle=LifecycleIdentity(
            lifecycle_id="life-stable",
            attempt_id="attempt-b",
            generation=1,
        ),
    )

    with pytest.raises(LifecycleConflict):
        service.create(different_attempt)


def test_same_lifecycle_identity_cannot_change_order_intent() -> None:
    transport = FakeTransport()
    service = OrderLifecycleService(transport)

    first = make_intent()
    service.create(first)

    changed_price = replace(
        first,
        price=0.46,
    )

    with pytest.raises(LifecycleConflict):
        service.create(changed_price)


def test_submit_is_one_shot_per_lifecycle_generation() -> None:
    transport = FakeTransport(
        post_error=RuntimeError("confirmed local no-post"),
        enter_raw_post=False,
    )
    service = OrderLifecycleService(transport)
    intent = make_intent()

    first = service.submit(intent, {"signed": True})
    assert first.outcome is SubmissionOutcome.FAILED_BEFORE_SUBMIT
    assert transport.post_calls == 1

    # The same generation cannot silently become a second submit attempt.
    with pytest.raises(LifecycleConflict):
        service.submit(intent, {"signed": True})

    assert transport.post_calls == 1


def test_safe_generation_replacement_removes_old_oid_index() -> None:
    transport = FakeTransport(
        cancel_response={"canceled": ["oid-1"]},
    )
    service = OrderLifecycleService(transport)
    intent = make_intent()
    submit_working(service, intent)
    service.cancel(intent.lifecycle)

    service.apply_reconciliation(
        ReconciliationResult(
            outcome=ReconciliationOutcome.CANCELLED_ZERO_FILL,
            lifecycle=intent.lifecycle,
            order_id="oid-1",
            reason="exact terminal zero proof",
        )
    )

    replacement = replace(
        intent,
        lifecycle=intent.lifecycle.next_generation(),
    )
    created = service.create(replacement)

    assert created.lifecycle == replacement.lifecycle
    assert created.state is OrderLifecycleState.CREATED
    assert created.order_id is None
    assert service.get_by_order_id("oid-1") is None


def test_stale_lower_quantity_status_cannot_rewrite_average_fill_price() -> None:
    transport = FakeTransport()
    service = OrderLifecycleService(transport)
    intent = make_intent()
    submit_working(service, intent)

    service.observe_status(
        intent.lifecycle,
        {
            "orderID": "oid-1",
            "status": "LIVE",
            "size_matched": "2",
            "average_price": "0.44",
        },
    )

    service.observe_status(
        intent.lifecycle,
        {
            "orderID": "oid-1",
            "status": "LIVE",
            "size_matched": "1",
            "average_price": "0.41",
        },
    )

    current = service.get(intent.lifecycle)
    assert current is not None
    assert current.state is OrderLifecycleState.PARTIALLY_FILLED
    assert current.working_order.filled_size == pytest.approx(2.0)
    assert current.working_order.average_fill_price == pytest.approx(0.44)
