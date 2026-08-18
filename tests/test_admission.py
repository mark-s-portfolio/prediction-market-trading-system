"""
Regression tests for candidate admission and one-shot execution authority.

The suite exercises the real candidate, quality, risk, admission, public-policy,
permit-ledger, and CLOB pre-submit boundary implementations. Only the raw venue
client is replaced with a small boundary double.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone, timedelta
import time

import pytest

from src.execution.clob_transport import (
    ClobTransport,
    PreSubmitRejected,
)
from src.execution.types import OrderSide
from src.market.types import (
    BookLevel,
    BookSource,
    MarketBooks,
    MarketDefinition,
    OrderBookSnapshot,
    OutcomeSide,
)
from src.risk.position_state import PositionBook
from src.risk.risk_manager import (
    ProposedExposure,
    RiskAction,
    RiskLimits,
    RiskManager,
)
from src.strategy.admission import (
    AdmissionContext,
    AdmissionPermitLedger,
    AdmissionPolicy,
    AdmissionPolicyConfig,
    AdmissionVerdict,
    ComparisonOperator,
    FailureDisposition,
    MeasurementRule,
    RequiredModelEvidenceRule,
)
from src.strategy.candidate import (
    CandidateFactory,
    CandidatePurpose,
    CandidateSource,
    CandidateSubject,
    MarketSnapshotRef,
    ModelObservation,
    ObservationKind,
    QuoteIntent,
    QuoteStyle,
)
from src.strategy.public_policy import (
    PermitPreSubmitValidator,
    PublicPolicyConfig,
    build_public_admission_policy,
    build_public_policy_bundle,
)
from src.strategy.quality import CandidateQualityMeasurer


class EmptyOwnership:
    def owned_snapshots(self):
        return ()

    def has_owner_for_token(self, token_id):
        return False


class RawClient:
    def __init__(self):
        self.posts = 0

    def post_order(self, signed_order):
        self.posts += 1
        return {
            "orderID": f"oid-{self.posts}",
            "status": "LIVE",
            "size_matched": "0",
        }


def make_market() -> MarketDefinition:
    return MarketDefinition(
        slug="market-1",
        question="Will the example event resolve YES?",
        yes_token="yes-token",
        no_token="no-token",
        condition_id="condition-1",
        interval_minutes=15,
        end_time=datetime.now(timezone.utc) + timedelta(minutes=15),
        tick_size=0.01,
        neg_risk=False,
    )


def make_books(
    market: MarketDefinition,
    *,
    now: float,
    generation_shift: float = 0.0,
) -> MarketBooks:
    yes = OrderBookSnapshot(
        token_id=market.yes_token,
        bids=(BookLevel(0.49, 10.0),),
        asks=(BookLevel(0.50, 12.0),),
        timestamp=now - 0.05 + generation_shift,
        source=BookSource.WEBSOCKET,
        depth_proven=True,
    )
    no = OrderBookSnapshot(
        token_id=market.no_token,
        bids=(BookLevel(0.49, 11.0),),
        asks=(BookLevel(0.50, 13.0),),
        timestamp=now - 0.04 + generation_shift,
        source=BookSource.WEBSOCKET,
        depth_proven=True,
    )
    return MarketBooks(
        market=market,
        yes=yes,
        no=no,
    )


def make_candidate(
    market: MarketDefinition,
    books: MarketBooks,
    *,
    now: float,
    generation: int = 5,
    valid_until: float | None = None,
    observations=(),
):
    subject = CandidateSubject.from_market(
        market,
        OutcomeSide.YES,
    )
    snapshot = MarketSnapshotRef.from_books(
        books,
        captured_at=now - 0.02,
        market_data_generation=generation,
    )

    return CandidateFactory(
        namespace="admission-test"
    ).create(
        subject=subject,
        quote=QuoteIntent(
            order_side=OrderSide.BUY,
            purpose=CandidatePurpose.OPENING,
            style=QuoteStyle.PASSIVE_LIMIT,
            limit_price=0.49,
            quantity=2.0,
            paired_token_id=market.no_token,
        ),
        market_snapshot=snapshot,
        attempt_id="attempt-1",
        producer="test-producer",
        source=CandidateSource.MARKET_DATA,
        created_at=now - 0.01,
        valid_until=(
            now + 30.0
            if valid_until is None
            else valid_until
        ),
        model_observations=tuple(observations),
    )


def make_stack(
    *,
    with_model: bool = False,
    public_config: PublicPolicyConfig | None = None,
):
    now = time.time()
    market = make_market()
    books = make_books(market, now=now)

    observations = ()
    expected_model_keys = ()

    if with_model:
        observations = (
            ModelObservation(
                model_name="demo_model",
                metric="probability",
                kind=ObservationKind.PROBABILITY,
                value=0.60,
                generation=2,
                observed_at=now - 0.10,
            ),
        )
        expected_model_keys = (
            "demo_model:probability",
        )

    candidate = make_candidate(
        market,
        books,
        now=now,
        observations=observations,
    )

    quality = CandidateQualityMeasurer().measure(
        candidate,
        books,
        now=now,
        market_data_generation=5,
        expected_model_keys=expected_model_keys,
    )

    risk_manager = RiskManager(
        positions=PositionBook(),
        execution_ownership=EmptyOwnership(),
        limits=RiskLimits(),
    )

    proposal = ProposedExposure(
        action=RiskAction.NEW_EXPOSURE,
        token_id=candidate.token_id,
        market_id=candidate.market_id,
        outcome_side=candidate.subject.outcome_side,
        quantity=candidate.quote.quantity,
        estimated_unit_cost=candidate.quote.limit_price,
    )
    risk = risk_manager.assess(proposal)

    config = (
        public_config
        or PublicPolicyConfig(
            name="admission-test-policy",
            generation=3,
            permit_ttl_seconds=10.0,
        )
    )
    policy = build_public_admission_policy(config)

    context = AdmissionContext(
        candidate=candidate,
        quality=quality,
        risk_proposal=proposal,
        risk_decision=risk,
        evaluated_at=now,
    )

    return {
        "now": now,
        "market": market,
        "books": books,
        "candidate": candidate,
        "quality": quality,
        "risk_manager": risk_manager,
        "proposal": proposal,
        "risk": risk,
        "config": config,
        "policy": policy,
        "context": context,
    }


def test_exact_bound_context_allows_and_issues_permit() -> None:
    stack = make_stack()

    decision = stack["policy"].evaluate(
        stack["context"]
    )

    assert decision.verdict is AdmissionVerdict.ALLOW
    assert decision.permit is not None
    assert decision.permit.matches_candidate(
        stack["candidate"]
    )
    assert (
        decision.candidate_fingerprint
        == stack["candidate"].fingerprint
    )
    assert (
        decision.policy_fingerprint
        == stack["policy"].policy_fingerprint
    )


def test_quality_candidate_fingerprint_mismatch_denies() -> None:
    stack = make_stack()

    bad_quality = replace(
        stack["quality"],
        candidate_fingerprint="different-fingerprint",
    )
    decision = stack["policy"].evaluate(
        replace(
            stack["context"],
            quality=bad_quality,
        )
    )

    assert decision.verdict is AdmissionVerdict.DENY
    assert decision.permit is None
    assert any(
        check.rule_id
        == "binding.candidate_quality_identity"
        and check.blocks
        for check in decision.checks
    )


def test_origin_snapshot_provenance_mismatch_denies() -> None:
    stack = make_stack()

    bad_provenance = replace(
        stack["quality"].provenance,
        candidate_snapshot_id="snapshot-other",
    )
    bad_quality = replace(
        stack["quality"],
        provenance=bad_provenance,
    )

    decision = stack["policy"].evaluate(
        replace(
            stack["context"],
            quality=bad_quality,
        )
    )

    assert decision.verdict is AdmissionVerdict.DENY
    assert decision.permit is None
    assert any(
        check.rule_id == "binding.origin_snapshot"
        and check.blocks
        for check in decision.checks
    )


def test_candidate_risk_proposal_understated_cost_denies() -> None:
    stack = make_stack()

    understated = replace(
        stack["proposal"],
        estimated_unit_cost=0.01,
    )
    understated_risk = stack["risk_manager"].assess(
        understated
    )

    decision = stack["policy"].evaluate(
        replace(
            stack["context"],
            risk_proposal=understated,
            risk_decision=understated_risk,
        )
    )

    assert decision.verdict is AdmissionVerdict.DENY
    assert decision.permit is None
    assert any(
        check.rule_id
        == "binding.risk_proposal_subject"
        and check.blocks
        for check in decision.checks
    )


def test_risk_approval_from_different_proposal_cannot_be_reused() -> None:
    stack = make_stack()

    other_proposal = replace(
        stack["proposal"],
        estimated_unit_cost=0.48,
    )
    other_risk = stack["risk_manager"].assess(
        other_proposal
    )
    assert other_risk.allowed
    assert other_risk.action is stack["proposal"].action

    decision = stack["policy"].evaluate(
        replace(
            stack["context"],
            risk_decision=other_risk,
        )
    )

    assert decision.verdict is AdmissionVerdict.DENY
    assert decision.permit is None
    assert any(
        check.rule_id
        == "binding.risk_decision_proposal"
        and check.blocks
        for check in decision.checks
    )


def test_risk_decision_carries_exact_proposal() -> None:
    stack = make_stack()

    assert stack["risk"].proposal == stack["proposal"]
    assert (
        stack["risk"].action
        is stack["risk"].proposal.action
    )


def test_risk_denial_is_consumed_not_recomputed_away() -> None:
    stack = make_stack()
    stack["risk_manager"].set_manual_halt(
        "test halt"
    )
    denied_risk = stack["risk_manager"].assess(
        stack["proposal"]
    )
    assert not denied_risk.allowed

    decision = stack["policy"].evaluate(
        replace(
            stack["context"],
            risk_decision=denied_risk,
        )
    )

    assert decision.verdict is AdmissionVerdict.DENY
    assert decision.permit is None
    assert any(
        check.rule_id == "risk.approval"
        and check.blocks
        for check in decision.checks
    )


def test_expired_candidate_denies_without_permit() -> None:
    now = time.time()
    market = make_market()
    books = make_books(market, now=now)

    candidate = make_candidate(
        market,
        books,
        now=now - 2.0,
        valid_until=now - 1.0,
    )
    quality = CandidateQualityMeasurer().measure(
        candidate,
        books,
        now=now,
        market_data_generation=5,
    )

    risk_manager = RiskManager(
        positions=PositionBook(),
        execution_ownership=EmptyOwnership(),
        limits=RiskLimits(),
    )
    proposal = ProposedExposure(
        action=RiskAction.NEW_EXPOSURE,
        token_id=candidate.token_id,
        market_id=candidate.market_id,
        outcome_side=candidate.subject.outcome_side,
        quantity=candidate.quote.quantity,
        estimated_unit_cost=candidate.quote.limit_price,
    )
    risk = risk_manager.assess(proposal)

    policy = build_public_admission_policy(
        PublicPolicyConfig(
            permit_ttl_seconds=10.0,
        )
    )
    decision = policy.evaluate(
        AdmissionContext(
            candidate=candidate,
            quality=quality,
            risk_proposal=proposal,
            risk_decision=risk,
            evaluated_at=now,
        )
    )

    assert decision.verdict is AdmissionVerdict.DENY
    assert decision.permit is None
    assert any(
        check.rule_id == "candidate.explicit_validity"
        and check.blocks
        for check in decision.checks
    )


def test_missing_measurement_can_explicitly_defer() -> None:
    stack = make_stack()

    policy = AdmissionPolicy(
        rules=(
            MeasurementRule(
                rule_id="missing.measurement",
                measurement_name="does.not.exist",
                operator=ComparisonOperator.PRESENT,
                missing_disposition=FailureDisposition.DEFER,
            ),
        )
    )

    decision = policy.evaluate(
        stack["context"]
    )

    assert decision.verdict is AdmissionVerdict.DEFER
    assert decision.permit is None
    assert decision.deferring_checks


def test_missing_measurement_can_fail_closed_deny() -> None:
    stack = make_stack()

    policy = AdmissionPolicy(
        rules=(
            MeasurementRule(
                rule_id="missing.measurement",
                measurement_name="does.not.exist",
                operator=ComparisonOperator.PRESENT,
                missing_disposition=FailureDisposition.DENY,
            ),
        )
    )

    decision = policy.evaluate(
        stack["context"]
    )

    assert decision.verdict is AdmissionVerdict.DENY
    assert decision.permit is None
    assert decision.blocking_checks


def test_required_model_evidence_missing_denies() -> None:
    stack = make_stack()

    policy = AdmissionPolicy(
        rules=(
            RequiredModelEvidenceRule(
                rule_id="model.required",
                required_keys=(
                    "demo_model:probability",
                ),
            ),
        )
    )

    decision = policy.evaluate(
        stack["context"]
    )

    assert decision.verdict is AdmissionVerdict.DENY
    assert decision.permit is None


def test_required_model_evidence_present_allows() -> None:
    stack = make_stack(with_model=True)

    policy = AdmissionPolicy(
        rules=(
            RequiredModelEvidenceRule(
                rule_id="model.required",
                required_keys=(
                    "demo_model:probability",
                ),
            ),
        ),
        config=AdmissionPolicyConfig(
            name="model-policy",
            generation=1,
            permit_ttl_seconds=10.0,
        ),
    )

    decision = policy.evaluate(
        stack["context"]
    )

    assert decision.verdict is AdmissionVerdict.ALLOW
    assert decision.permit is not None


def test_rule_exception_fails_closed() -> None:
    stack = make_stack()

    class ExplodingRule:
        rule_id = "explode"

        def policy_payload(self):
            return {
                "type": "exploding-test-rule",
                "rule_id": self.rule_id,
            }

        def evaluate(self, context):
            raise RuntimeError("rule failure")

    policy = AdmissionPolicy(
        rules=(ExplodingRule(),)
    )

    decision = policy.evaluate(
        stack["context"]
    )

    assert decision.verdict is AdmissionVerdict.DENY
    assert decision.permit is None
    assert any(
        check.rule_id == "explode"
        and check.blocks
        and "rule evaluation error" in check.reason
        for check in decision.checks
    )


def test_policy_generation_changes_policy_and_permit_identity() -> None:
    stack = make_stack()

    first_policy = build_public_admission_policy(
        PublicPolicyConfig(
            name="same-policy",
            generation=1,
            permit_ttl_seconds=10.0,
        )
    )
    second_policy = build_public_admission_policy(
        PublicPolicyConfig(
            name="same-policy",
            generation=2,
            permit_ttl_seconds=10.0,
        )
    )

    first = first_policy.evaluate(
        stack["context"]
    )
    second = second_policy.evaluate(
        stack["context"]
    )

    assert first.allowed and second.allowed
    assert (
        first.policy_fingerprint
        != second.policy_fingerprint
    )
    assert (
        first.permit.policy_generation == 1
    )
    assert (
        second.permit.policy_generation == 2
    )
    assert (
        first.permit.permit_id
        != second.permit.permit_id
    )


def test_candidate_validity_is_upper_bound_on_permit_expiry() -> None:
    stack = make_stack()

    decision = stack["policy"].evaluate(
        stack["context"]
    )
    permit = decision.permit

    assert permit is not None
    assert permit.expires_at is not None
    assert (
        permit.expires_at
        <= stack["candidate"].valid_until
    )


def test_consumed_permit_cannot_be_registered_again() -> None:
    stack = make_stack()
    decision = stack["policy"].evaluate(
        stack["context"]
    )
    ledger = AdmissionPermitLedger()

    permit = ledger.register(decision)
    ledger.consume(
        permit.permit_id,
        candidate=stack["candidate"],
        consumer="test-consumer",
    )

    with pytest.raises(
        ValueError,
        match="consumed admission permit",
    ):
        ledger.register(decision)


def test_expired_permit_cannot_be_registered() -> None:
    stack = make_stack()
    decision = stack["policy"].evaluate(
        stack["context"]
    )
    assert decision.permit is not None

    old = time.time() - 20.0
    expired_permit = replace(
        decision.permit,
        issued_at=old,
        expires_at=old + 1.0,
    )
    expired_decision = replace(
        decision,
        permit=expired_permit,
    )

    ledger = AdmissionPermitLedger()

    with pytest.raises(
        ValueError,
        match="expired admission permit",
    ):
        ledger.register(expired_decision)


def test_mismatched_candidate_cannot_consume_active_permit() -> None:
    stack = make_stack()
    decision = stack["policy"].evaluate(
        stack["context"]
    )
    ledger = AdmissionPermitLedger()
    permit = ledger.register(decision)

    changed_quote = replace(
        stack["candidate"].quote,
        limit_price=0.48,
    )
    changed_candidate = replace(
        stack["candidate"],
        quote=changed_quote,
    )

    assert not permit.matches_candidate(
        changed_candidate
    )

    with pytest.raises(
        ValueError,
        match="does not match candidate",
    ):
        ledger.consume(
            permit.permit_id,
            candidate=changed_candidate,
            consumer="wrong-candidate",
        )

    assert ledger.get(permit.permit_id) == permit


def test_pre_submit_validator_rejects_missing_permit_identity() -> None:
    bundle = build_public_policy_bundle(
        PublicPolicyConfig(
            permit_ttl_seconds=10.0,
        )
    )

    from src.execution.clob_transport import PreSubmitContext

    result = bundle.pre_submit_validator.validate(
        PreSubmitContext(
            token_id="yes-token",
            market_id="market-1",
            lifecycle_id="life-1",
            attempt_id="attempt-1",
            metadata={},
        )
    )

    assert not result.allowed
    assert "missing admission permit" in result.reason


def test_pre_submit_validator_rejects_tampered_context() -> None:
    stack = make_stack()
    bundle = build_public_policy_bundle(
        stack["config"]
    )
    decision = bundle.policy.evaluate(
        stack["context"]
    )

    authority = bundle.transport_bridge.prepare(
        decision=decision,
        candidate=stack["candidate"],
        lifecycle_id="life-1",
    )

    tampered = replace(
        authority.pre_submit_context,
        metadata={
            **authority.pre_submit_context.metadata,
            "candidate_fingerprint": "tampered",
        },
    )

    result = bundle.pre_submit_validator.validate(
        tampered
    )

    assert not result.allowed
    assert "binding mismatch" in result.reason


def test_validator_rejects_noncurrent_policy_generation() -> None:
    stack = make_stack()
    bundle = build_public_policy_bundle(
        stack["config"]
    )
    decision = bundle.policy.evaluate(
        stack["context"]
    )
    authority = bundle.transport_bridge.prepare(
        decision=decision,
        candidate=stack["candidate"],
        lifecycle_id="life-1",
    )

    future_validator = PermitPreSubmitValidator(
        ledger=bundle.permit_ledger,
        expected_policy_name=bundle.policy.config.name,
        expected_policy_generation=(
            bundle.policy.config.generation + 1
        ),
        expected_policy_fingerprint=None,
    )

    result = future_validator.validate(
        authority.pre_submit_context
    )

    assert not result.allowed
    assert "generation is not current" in result.reason


def test_same_active_permit_cannot_prepare_two_lifecycles() -> None:
    stack = make_stack()
    bundle = build_public_policy_bundle(
        stack["config"]
    )
    decision = bundle.policy.evaluate(
        stack["context"]
    )

    bundle.transport_bridge.prepare(
        decision=decision,
        candidate=stack["candidate"],
        lifecycle_id="life-a",
    )

    with pytest.raises(
        ValueError,
        match="already prepared for lifecycle",
    ):
        bundle.transport_bridge.prepare(
            decision=decision,
            candidate=stack["candidate"],
            lifecycle_id="life-b",
        )


def test_raw_observer_context_mismatch_does_not_consume_permit() -> None:
    stack = make_stack()
    bundle = build_public_policy_bundle(
        stack["config"]
    )
    decision = bundle.policy.evaluate(
        stack["context"]
    )
    authority = bundle.transport_bridge.prepare(
        decision=decision,
        candidate=stack["candidate"],
        lifecycle_id="life-1",
    )

    wrong_context = replace(
        authority.pre_submit_context,
        lifecycle_id="life-other",
    )

    with pytest.raises(
        RuntimeError,
        match="different pre-submit context",
    ):
        authority.raw_post_enter_observer(
            wrong_context
        )

    assert (
        bundle.permit_ledger.get(
            authority.permit.permit_id
        )
        == authority.permit
    )
    assert (
        bundle.permit_ledger.consumed(
            authority.permit.permit_id
        )
        is None
    )


def test_exact_raw_boundary_consumes_once_and_posts_once() -> None:
    stack = make_stack()
    bundle = build_public_policy_bundle(
        stack["config"]
    )
    decision = bundle.policy.evaluate(
        stack["context"]
    )
    authority = bundle.transport_bridge.prepare(
        decision=decision,
        candidate=stack["candidate"],
        lifecycle_id="life-1",
    )

    raw = RawClient()
    transport = ClobTransport(
        raw,
        pre_submit_validator=(
            bundle.pre_submit_validator
        ),
    )

    response = transport.post_order(
        {"signed": True},
        **authority.submission_kwargs(),
    )

    assert response["orderID"] == "oid-1"
    assert raw.posts == 1

    consumption = bundle.permit_ledger.consumed(
        authority.permit.permit_id
    )
    assert consumption is not None
    assert consumption.consumer == "raw-post-enter"

    with pytest.raises(PreSubmitRejected):
        transport.post_order(
            {"signed": True},
            **authority.submission_kwargs(),
        )

    assert raw.posts == 1


def test_consumed_permit_cannot_be_prepared_again() -> None:
    stack = make_stack()
    bundle = build_public_policy_bundle(
        stack["config"]
    )
    decision = bundle.policy.evaluate(
        stack["context"]
    )
    authority = bundle.transport_bridge.prepare(
        decision=decision,
        candidate=stack["candidate"],
        lifecycle_id="life-1",
    )

    authority.raw_post_enter_observer(
        authority.pre_submit_context
    )

    with pytest.raises(
        ValueError,
        match="already consumed",
    ):
        bundle.transport_bridge.prepare(
            decision=decision,
            candidate=stack["candidate"],
            lifecycle_id="life-1",
        )


def test_deny_and_defer_decisions_never_have_permits() -> None:
    stack = make_stack()

    deny_policy = AdmissionPolicy(
        rules=(
            MeasurementRule(
                rule_id="forced-deny",
                measurement_name="books.both_two_sided",
                operator=ComparisonOperator.FALSE,
            ),
        )
    )
    defer_policy = AdmissionPolicy(
        rules=(
            MeasurementRule(
                rule_id="forced-defer",
                measurement_name="missing",
                operator=ComparisonOperator.PRESENT,
                missing_disposition=FailureDisposition.DEFER,
            ),
        )
    )

    denied = deny_policy.evaluate(
        stack["context"]
    )
    deferred = defer_policy.evaluate(
        stack["context"]
    )

    assert denied.verdict is AdmissionVerdict.DENY
    assert denied.permit is None
    assert deferred.verdict is AdmissionVerdict.DEFER
    assert deferred.permit is None
