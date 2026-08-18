"""
Public demonstration policy and exact pre-submit permit bridge.

This module wires the generic strategy contracts into a usable portfolio-safe
policy without embedding private production parameters.

It has two responsibilities:

1. Build a small demonstration AdmissionPolicy from structural market-data
   requirements only. Numeric trading thresholds remain deployment-owned.

2. Carry one immutable AdmissionPermit to the transport's final synchronous
   PRE_NETWORK boundary and consume it exactly once immediately before raw POST.

The transport integration intentionally uses two phases:

    PreSubmitValidator.validate()
        read-only permit verification
        no permit consumption
        no network I/O

    RawPostEnterObserver()
        exact context re-check
        one-shot permit consumption
        returns only after ownership is durably recorded

The raw SDK POST begins only after both phases succeed.

This ordering preserves an important distinction:
- validator failure -> confirmed local no-post
- permit consumption followed by POST exception -> the submission authority was
  already consumed and the execution lifecycle must treat the write as potentially
  ambiguous rather than retrying the same permit

No asset-specific setup knowledge, historical trade families, production
thresholds, bankroll values, or proprietary admission graph are included.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Callable, Mapping, Optional, Tuple

from src.execution.clob_transport import (
    PreSubmitContext,
    RawPostEnterObserver,
    ValidationResult,
)
from src.strategy.admission import (
    AdmissionDecision,
    AdmissionPermit,
    AdmissionPermitLedger,
    AdmissionPolicy,
    AdmissionPolicyConfig,
    AdmissionVerdict,
    ComparisonOperator,
    FailureDisposition,
    MeasurementRule,
    RequiredFeatureRule,
    RequiredModelEvidenceRule,
)
from src.strategy.candidate import StrategyCandidate


PERMIT_ID_KEY = "admission_permit_id"
CANDIDATE_ID_KEY = "candidate_id"
CANDIDATE_FINGERPRINT_KEY = "candidate_fingerprint"
SNAPSHOT_ID_KEY = "candidate_snapshot_id"
CANDIDATE_MARKET_GENERATION_KEY = "candidate_market_data_generation"
MEASURED_MARKET_GENERATION_KEY = "measured_market_data_generation"
QUOTE_FINGERPRINT_KEY = "quote_fingerprint"
POLICY_NAME_KEY = "admission_policy_name"
POLICY_GENERATION_KEY = "admission_policy_generation"
POLICY_FINGERPRINT_KEY = "admission_policy_fingerprint"
DECISION_FINGERPRINT_KEY = "admission_decision_fingerprint"


@dataclass(frozen=True, slots=True)
class PublicPolicyConfig:
    """Neutral demonstration policy configuration.

    Defaults contain no numeric market-quality or trading thresholds.  The policy
    demonstrates structural evidence handling only.
    """

    name: str = "public-demo-policy"
    generation: int = 0
    permit_ttl_seconds: Optional[float] = None

    require_two_sided_books: bool = True
    reject_crossed_books: bool = True
    require_proven_depth: bool = False

    required_model_keys: Tuple[str, ...] = field(default_factory=tuple)
    required_feature_names: Tuple[str, ...] = field(default_factory=tuple)

    missing_evidence_disposition: FailureDisposition = FailureDisposition.DENY

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        generation = int(self.generation)

        if not name:
            raise ValueError("policy name is required")
        if generation < 0:
            raise ValueError("generation must be non-negative")

        ttl = self.permit_ttl_seconds
        if ttl is not None:
            ttl = float(ttl)
            if not math.isfinite(ttl) or ttl <= 0.0:
                raise ValueError("permit_ttl_seconds must be positive")

        model_keys = tuple(
            sorted(
                {
                    str(key or "").strip()
                    for key in self.required_model_keys
                    if str(key or "").strip()
                }
            )
        )
        feature_names = tuple(
            sorted(
                {
                    str(name or "").strip()
                    for name in self.required_feature_names
                    if str(name or "").strip()
                }
            )
        )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "permit_ttl_seconds", ttl)
        object.__setattr__(self, "required_model_keys", model_keys)
        object.__setattr__(
            self,
            "required_feature_names",
            feature_names,
        )


def build_public_admission_policy(
    config: PublicPolicyConfig = PublicPolicyConfig(),
) -> AdmissionPolicy:
    """Build the portfolio demonstration policy.

    The default policy requires a normal risk approval through admission.py plus
    structurally usable two-sided/non-crossed books. Proven full depth and model
    evidence are opt-in so no hidden numeric liquidity/model policy is implied.
    """

    rules = []

    if config.require_two_sided_books:
        rules.append(
            MeasurementRule(
                rule_id="public.books.two_sided",
                measurement_name="books.both_two_sided",
                operator=ComparisonOperator.TRUE,
                failure_disposition=FailureDisposition.DENY,
                missing_disposition=config.missing_evidence_disposition,
            )
        )

    if config.reject_crossed_books:
        rules.extend(
            (
                MeasurementRule(
                    rule_id="public.books.yes_not_crossed",
                    measurement_name="books.yes.crossed",
                    operator=ComparisonOperator.FALSE,
                    failure_disposition=FailureDisposition.DENY,
                    missing_disposition=config.missing_evidence_disposition,
                ),
                MeasurementRule(
                    rule_id="public.books.no_not_crossed",
                    measurement_name="books.no.crossed",
                    operator=ComparisonOperator.FALSE,
                    failure_disposition=FailureDisposition.DENY,
                    missing_disposition=config.missing_evidence_disposition,
                ),
            )
        )

    if config.require_proven_depth:
        rules.append(
            MeasurementRule(
                rule_id="public.books.proven_depth",
                measurement_name="books.both_depth_proven",
                operator=ComparisonOperator.TRUE,
                failure_disposition=FailureDisposition.DENY,
                missing_disposition=config.missing_evidence_disposition,
            )
        )

    if config.required_model_keys:
        rules.append(
            RequiredModelEvidenceRule(
                rule_id="public.models.required",
                required_keys=config.required_model_keys,
                missing_disposition=config.missing_evidence_disposition,
            )
        )

    if config.required_feature_names:
        rules.append(
            RequiredFeatureRule(
                rule_id="public.features.required",
                required_names=config.required_feature_names,
                missing_disposition=config.missing_evidence_disposition,
            )
        )

    return AdmissionPolicy(
        rules=tuple(rules),
        config=AdmissionPolicyConfig(
            name=config.name,
            generation=config.generation,
            permit_ttl_seconds=config.permit_ttl_seconds,
            require_candidate_not_expired=True,
            require_risk_approval=True,
        ),
    )


def _context_metadata(
    *,
    candidate: StrategyCandidate,
    permit: AdmissionPermit,
) -> dict[str, object]:
    return {
        PERMIT_ID_KEY: permit.permit_id,
        CANDIDATE_ID_KEY: candidate.candidate_id,
        CANDIDATE_FINGERPRINT_KEY: candidate.fingerprint,
        SNAPSHOT_ID_KEY: candidate.market_snapshot.snapshot_id,
        CANDIDATE_MARKET_GENERATION_KEY: (
            candidate.market_snapshot.market_data_generation
        ),
        MEASURED_MARKET_GENERATION_KEY: (
            permit.measured_market_data_generation
        ),
        QUOTE_FINGERPRINT_KEY: permit.quote_fingerprint,
        POLICY_NAME_KEY: permit.policy_name,
        POLICY_GENERATION_KEY: permit.policy_generation,
        POLICY_FINGERPRINT_KEY: permit.policy_fingerprint,
        DECISION_FINGERPRINT_KEY: permit.decision_fingerprint,
    }


def build_pre_submit_context(
    *,
    candidate: StrategyCandidate,
    permit: AdmissionPermit,
    lifecycle_id: str,
) -> PreSubmitContext:
    """Build the exact transport context for one admitted candidate."""

    lifecycle_id = str(lifecycle_id or "").strip()
    if not lifecycle_id:
        raise ValueError("lifecycle_id is required")

    if not permit.matches_candidate(candidate):
        raise ValueError("permit does not match candidate")

    return PreSubmitContext(
        token_id=candidate.token_id,
        market_id=candidate.market_id,
        lifecycle_id=lifecycle_id,
        attempt_id=candidate.attempt_id,
        metadata=_context_metadata(
            candidate=candidate,
            permit=permit,
        ),
    )


class PermitPreSubmitValidator:
    """Synchronous, networkless exact permit validator.

    Validation is read-only.  The permit remains active until the paired
    RawPostEnterObserver consumes it at the final local ownership handoff.
    """

    def __init__(
        self,
        *,
        ledger: AdmissionPermitLedger,
        expected_policy_name: Optional[str] = None,
        expected_policy_generation: Optional[int] = None,
        expected_policy_fingerprint: Optional[str] = None,
    ) -> None:
        self.ledger = ledger
        self.expected_policy_name = (
            str(expected_policy_name or "").strip() or None
        )
        self.expected_policy_generation = (
            int(expected_policy_generation)
            if expected_policy_generation is not None
            else None
        )
        self.expected_policy_fingerprint = (
            str(expected_policy_fingerprint or "").strip() or None
        )

        if (
            self.expected_policy_generation is not None
            and self.expected_policy_generation < 0
        ):
            raise ValueError(
                "expected_policy_generation must be non-negative"
            )

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

    @staticmethod
    def _required_metadata(
        context: PreSubmitContext,
        key: str,
    ) -> Optional[object]:
        value = context.metadata.get(key)
        if value in (None, ""):
            return None
        return value

    def validate(
        self,
        context: PreSubmitContext,
    ) -> ValidationResult:
        permit_id_value = self._required_metadata(
            context,
            PERMIT_ID_KEY,
        )

        if permit_id_value is None:
            return self._deny(
                "missing admission permit identity",
                evidence={"missing": PERMIT_ID_KEY},
            )

        permit_id = str(permit_id_value)
        permit = self.ledger.get(permit_id)

        if permit is None:
            consumed = self.ledger.consumed(permit_id)
            return self._deny(
                (
                    "admission permit already consumed"
                    if consumed is not None
                    else "admission permit not active"
                ),
                evidence={
                    "permit_id": permit_id,
                    "consumed_by": (
                        consumed.consumer
                        if consumed is not None
                        else None
                    ),
                },
            )

        if permit.expired():
            return self._deny(
                "admission permit expired",
                evidence={
                    "permit_id": permit.permit_id,
                    "expires_at": permit.expires_at,
                },
            )

        expected = {
            "token_id": permit.token_id,
            "market_id": permit.market_id,
            "attempt_id": permit.attempt_id,
            CANDIDATE_ID_KEY: permit.candidate_id,
            CANDIDATE_FINGERPRINT_KEY: (
                permit.candidate_fingerprint
            ),
            SNAPSHOT_ID_KEY: permit.snapshot_id,
            CANDIDATE_MARKET_GENERATION_KEY: (
                permit.candidate_market_data_generation
            ),
            MEASURED_MARKET_GENERATION_KEY: (
                permit.measured_market_data_generation
            ),
            QUOTE_FINGERPRINT_KEY: permit.quote_fingerprint,
            POLICY_NAME_KEY: permit.policy_name,
            POLICY_GENERATION_KEY: permit.policy_generation,
            POLICY_FINGERPRINT_KEY: permit.policy_fingerprint,
            DECISION_FINGERPRINT_KEY: (
                permit.decision_fingerprint
            ),
        }

        observed = {
            "token_id": context.token_id,
            "market_id": context.market_id,
            "attempt_id": context.attempt_id,
            CANDIDATE_ID_KEY: context.metadata.get(
                CANDIDATE_ID_KEY
            ),
            CANDIDATE_FINGERPRINT_KEY: context.metadata.get(
                CANDIDATE_FINGERPRINT_KEY
            ),
            SNAPSHOT_ID_KEY: context.metadata.get(
                SNAPSHOT_ID_KEY
            ),
            CANDIDATE_MARKET_GENERATION_KEY: context.metadata.get(
                CANDIDATE_MARKET_GENERATION_KEY
            ),
            MEASURED_MARKET_GENERATION_KEY: context.metadata.get(
                MEASURED_MARKET_GENERATION_KEY
            ),
            QUOTE_FINGERPRINT_KEY: context.metadata.get(
                QUOTE_FINGERPRINT_KEY
            ),
            POLICY_NAME_KEY: context.metadata.get(
                POLICY_NAME_KEY
            ),
            POLICY_GENERATION_KEY: context.metadata.get(
                POLICY_GENERATION_KEY
            ),
            POLICY_FINGERPRINT_KEY: context.metadata.get(
                POLICY_FINGERPRINT_KEY
            ),
            DECISION_FINGERPRINT_KEY: context.metadata.get(
                DECISION_FINGERPRINT_KEY
            ),
        }

        mismatches = {
            key: {
                "expected": expected[key],
                "observed": observed[key],
            }
            for key in expected
            if observed.get(key) != expected.get(key)
        }

        if mismatches:
            return self._deny(
                "pre-submit permit/context binding mismatch",
                evidence={
                    "permit_id": permit.permit_id,
                    "mismatches": mismatches,
                },
            )

        if (
            self.expected_policy_name is not None
            and permit.policy_name != self.expected_policy_name
        ):
            return self._deny(
                "permit policy name is not current",
                evidence={
                    "permit_policy": permit.policy_name,
                    "expected_policy": self.expected_policy_name,
                },
            )

        if (
            self.expected_policy_generation is not None
            and permit.policy_generation
            != self.expected_policy_generation
        ):
            return self._deny(
                "permit policy generation is not current",
                evidence={
                    "permit_generation": permit.policy_generation,
                    "expected_generation": (
                        self.expected_policy_generation
                    ),
                },
            )

        if (
            self.expected_policy_fingerprint is not None
            and permit.policy_fingerprint
            != self.expected_policy_fingerprint
        ):
            return self._deny(
                "permit policy fingerprint is not current",
                evidence={
                    "permit_fingerprint": (
                        permit.policy_fingerprint
                    ),
                    "expected_fingerprint": (
                        self.expected_policy_fingerprint
                    ),
                },
            )

        return ValidationResult(
            allowed=True,
            reason="exact admission permit active and bound",
            evidence={
                "permit_id": permit.permit_id,
                "candidate_id": permit.candidate_id,
                "policy_generation": permit.policy_generation,
                "decision_fingerprint": (
                    permit.decision_fingerprint
                ),
            },
        )


@dataclass(frozen=True, slots=True)
class PreparedSubmissionAuthority:
    """Exact admission authority pair passed to ExecutionManager.submit()."""

    candidate: StrategyCandidate
    permit: AdmissionPermit
    pre_submit_context: PreSubmitContext
    raw_post_enter_observer: RawPostEnterObserver

    def submission_kwargs(self) -> dict[str, object]:
        return {
            "pre_submit_context": self.pre_submit_context,
            "raw_post_enter_observer": self.raw_post_enter_observer,
        }


class AdmissionTransportBridge:
    """Producer->consumer bridge for one-shot admission permits."""

    def __init__(
        self,
        *,
        ledger: AdmissionPermitLedger,
        consumer_name: str = "raw-post-enter",
    ) -> None:
        consumer_name = str(consumer_name or "").strip()
        if not consumer_name:
            raise ValueError("consumer_name is required")

        self.ledger = ledger
        self.consumer_name = consumer_name

        # Diagnostic only: the ledger is still the one-shot authority owner.
        self._gate = threading.RLock()
        self._prepared_lifecycle_by_permit: dict[str, str] = {}
        self._last_consumed_permit_id: Optional[str] = None
        self._last_consumed_at: Optional[float] = None

    def prepare(
        self,
        *,
        decision: AdmissionDecision,
        candidate: StrategyCandidate,
        lifecycle_id: str,
    ) -> PreparedSubmissionAuthority:
        """Register ALLOW authority and bind it to one transport context."""

        if decision.verdict is not AdmissionVerdict.ALLOW:
            raise ValueError("only ALLOW decisions may prepare submission")

        permit = decision.permit
        if permit is None:
            raise ValueError("ALLOW decision has no admission permit")

        if not permit.matches_candidate(candidate):
            raise ValueError(
                "admission decision permit does not match candidate"
            )

        if permit.expired():
            raise ValueError("admission permit expired before preparation")

        if self.ledger.consumed(permit.permit_id) is not None:
            raise ValueError("admission permit was already consumed")

        lifecycle_id = str(lifecycle_id or "").strip()
        if not lifecycle_id:
            raise ValueError("lifecycle_id is required")

        with self._gate:
            prepared_for = self._prepared_lifecycle_by_permit.get(
                permit.permit_id
            )
            if prepared_for is not None:
                raise ValueError(
                    "admission permit already prepared for lifecycle "
                    f"{prepared_for}"
                )

            permit = self.ledger.register(decision)
            self._prepared_lifecycle_by_permit[
                permit.permit_id
            ] = lifecycle_id

        context = build_pre_submit_context(
            candidate=candidate,
            permit=permit,
            lifecycle_id=lifecycle_id,
        )

        expected_context = context

        def raw_post_enter_observer(
            observed_context: PreSubmitContext,
        ) -> None:
            # This callback is deliberately synchronous and does no network I/O.
            # It runs after validation and immediately before the raw SDK POST.
            if observed_context != expected_context:
                raise RuntimeError(
                    "raw-post observer received a different pre-submit context"
                )

            consumption = self.ledger.consume(
                permit.permit_id,
                candidate=candidate,
                consumer=self.consumer_name,
                now=time.time(),
            )

            with self._gate:
                self._prepared_lifecycle_by_permit.pop(
                    permit.permit_id,
                    None,
                )
                self._last_consumed_permit_id = (
                    consumption.permit.permit_id
                )
                self._last_consumed_at = consumption.consumed_at

        return PreparedSubmissionAuthority(
            candidate=candidate,
            permit=permit,
            pre_submit_context=context,
            raw_post_enter_observer=raw_post_enter_observer,
        )

    def last_consumption(self) -> tuple[Optional[str], Optional[float]]:
        with self._gate:
            return (
                self._last_consumed_permit_id,
                self._last_consumed_at,
            )


@dataclass(frozen=True, slots=True)
class PublicPolicyBundle:
    """Convenience bundle for application wiring."""

    policy: AdmissionPolicy
    permit_ledger: AdmissionPermitLedger
    pre_submit_validator: PermitPreSubmitValidator
    transport_bridge: AdmissionTransportBridge


def build_public_policy_bundle(
    config: PublicPolicyConfig = PublicPolicyConfig(),
    *,
    permit_ledger_max_entries: int = 2048,
) -> PublicPolicyBundle:
    """Build one coherent public policy/permit/transport authority stack."""

    policy = build_public_admission_policy(config)
    ledger = AdmissionPermitLedger(
        max_entries=int(permit_ledger_max_entries)
    )

    validator = PermitPreSubmitValidator(
        ledger=ledger,
        expected_policy_name=policy.config.name,
        expected_policy_generation=policy.config.generation,
        expected_policy_fingerprint=policy.policy_fingerprint,
    )

    bridge = AdmissionTransportBridge(
        ledger=ledger,
        consumer_name="raw-post-enter",
    )

    return PublicPolicyBundle(
        policy=policy,
        permit_ledger=ledger,
        pre_submit_validator=validator,
        transport_bridge=bridge,
    )
