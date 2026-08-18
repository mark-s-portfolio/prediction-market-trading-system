"""
Generic candidate admission framework for the public portfolio edition.

This is the first strategy layer that may produce an ALLOW/DENY/DEFER verdict.
It consumes already-normalized candidate, measurement, and risk contracts rather
than re-measuring market state or recomputing portfolio risk itself.

The separation is deliberate:

    candidate.py
        immutable proposal identity

    quality.py
        measurement-only evidence

    risk_manager.py
        portfolio/execution capacity

    admission.py
        explicit policy evaluation + immutable one-shot permit

Core invariants:
- admission evaluates one exact candidate fingerprint
- quality evidence must be bound to the same candidate and origin snapshot
- risk capacity must be bound to the same economic proposal
- missing evidence is explicit and fail-closed according to the rule contract
- ALLOW creates one immutable AdmissionPermit
- a permit cannot be silently rebuilt from later mutable state
- permit identity includes policy generation and evidence provenance
- permit consumption is one-shot
- no permit exists for DENY or DEFER
- policy rules are configuration/contracts, not hidden candidate producers
- this module does not submit orders or mutate execution lifecycle state

No production strategy thresholds, asset-specific setup families, historical
GOOD/BAD labels, bankroll values, or proprietary causal-authority graph are
embedded here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import threading
import time
from typing import Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

from src.risk.risk_manager import ProposedExposure, RiskDecision
from src.strategy.candidate import StrategyCandidate
from src.strategy.quality import (
    CandidateQualitySnapshot,
    MeasurementAvailability,
    QualityMeasurement,
)


class AdmissionVerdict(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    DEFER = "DEFER"


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    MISSING = "MISSING"


class FailureDisposition(str, Enum):
    """What a failed/missing rule means to final admission."""

    DENY = "DENY"
    DEFER = "DEFER"


class ComparisonOperator(str, Enum):
    """Generic comparison semantics for normalized measurements."""

    PRESENT = "PRESENT"
    TRUE = "TRUE"
    FALSE = "FALSE"
    EQUAL = "EQUAL"
    NOT_EQUAL = "NOT_EQUAL"
    LESS_THAN = "LESS_THAN"
    LESS_OR_EQUAL = "LESS_OR_EQUAL"
    GREATER_THAN = "GREATER_THAN"
    GREATER_OR_EQUAL = "GREATER_OR_EQUAL"
    BETWEEN_INCLUSIVE = "BETWEEN_INCLUSIVE"
    IN_SET = "IN_SET"


@dataclass(frozen=True, slots=True)
class AdmissionCheck:
    """Auditable result from one structural or policy requirement."""

    rule_id: str
    status: CheckStatus
    failure_disposition: FailureDisposition
    reason: str

    observed: object = None
    expected: object = None
    source: str = ""

    def __post_init__(self) -> None:
        rule_id = str(self.rule_id or "").strip()
        if not rule_id:
            raise ValueError("rule_id is required")

        object.__setattr__(self, "rule_id", rule_id)
        object.__setattr__(self, "reason", str(self.reason or ""))
        object.__setattr__(self, "source", str(self.source or ""))

    @property
    def passed(self) -> bool:
        return self.status is CheckStatus.PASS

    @property
    def blocks(self) -> bool:
        return (
            not self.passed
            and self.failure_disposition is FailureDisposition.DENY
        )

    @property
    def defers(self) -> bool:
        return (
            not self.passed
            and self.failure_disposition is FailureDisposition.DEFER
        )


@dataclass(frozen=True, slots=True)
class AdmissionContext:
    """Exact normalized evidence bundle consumed by admission."""

    candidate: StrategyCandidate
    quality: CandidateQualitySnapshot
    risk_proposal: ProposedExposure
    risk_decision: RiskDecision
    evaluated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        evaluated = float(self.evaluated_at)
        if not math.isfinite(evaluated) or evaluated <= 0.0:
            raise ValueError("evaluated_at must be positive")
        object.__setattr__(self, "evaluated_at", evaluated)


class AdmissionRule(Protocol):
    """Extension point for public/custom admission requirements."""

    @property
    def rule_id(self) -> str:
        ...

    def evaluate(self, context: AdmissionContext) -> AdmissionCheck:
        ...

    def policy_payload(self) -> Mapping[str, object]:
        ...


@dataclass(frozen=True, slots=True)
class MeasurementRule:
    """Generic policy constraint over one quality measurement.

    Numeric/string/bool values are compared only after the measurement layer has
    explicitly reported them as OBSERVED. Missing/unavailable evidence follows
    `missing_disposition`; an observed comparison failure follows
    `failure_disposition`.
    """

    rule_id: str
    measurement_name: str
    operator: ComparisonOperator

    expected: object = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    allowed_values: Tuple[object, ...] = field(default_factory=tuple)

    failure_disposition: FailureDisposition = FailureDisposition.DENY
    missing_disposition: FailureDisposition = FailureDisposition.DENY

    def __post_init__(self) -> None:
        rule_id = str(self.rule_id or "").strip()
        measurement_name = str(self.measurement_name or "").strip()

        if not rule_id:
            raise ValueError("rule_id is required")
        if not measurement_name:
            raise ValueError("measurement_name is required")

        minimum = self.minimum
        maximum = self.maximum

        if minimum is not None:
            minimum = float(minimum)
            if not math.isfinite(minimum):
                raise ValueError("minimum must be finite")

        if maximum is not None:
            maximum = float(maximum)
            if not math.isfinite(maximum):
                raise ValueError("maximum must be finite")

        if (
            minimum is not None
            and maximum is not None
            and minimum > maximum
        ):
            raise ValueError("minimum cannot exceed maximum")

        if (
            self.operator is ComparisonOperator.BETWEEN_INCLUSIVE
            and (minimum is None or maximum is None)
        ):
            raise ValueError(
                "BETWEEN_INCLUSIVE requires minimum and maximum"
            )

        if (
            self.operator is ComparisonOperator.IN_SET
            and not self.allowed_values
        ):
            raise ValueError("IN_SET requires allowed_values")

        object.__setattr__(self, "rule_id", rule_id)
        object.__setattr__(
            self,
            "measurement_name",
            measurement_name,
        )
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(
            self,
            "allowed_values",
            tuple(self.allowed_values),
        )

    def policy_payload(self) -> Mapping[str, object]:
        return {
            "type": "measurement",
            "rule_id": self.rule_id,
            "measurement_name": self.measurement_name,
            "operator": self.operator.value,
            "expected": self.expected,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "allowed_values": self.allowed_values,
            "failure_disposition": self.failure_disposition.value,
            "missing_disposition": self.missing_disposition.value,
        }

    @staticmethod
    def _numeric(value: object) -> Optional[float]:
        if isinstance(value, bool):
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    def _compare(self, value: object) -> tuple[bool, str]:
        operator = self.operator

        if operator is ComparisonOperator.PRESENT:
            return True, "measurement present"

        if operator is ComparisonOperator.TRUE:
            return bool(value is True), "expected boolean true"

        if operator is ComparisonOperator.FALSE:
            return bool(value is False), "expected boolean false"

        if operator is ComparisonOperator.EQUAL:
            return value == self.expected, "value equality"

        if operator is ComparisonOperator.NOT_EQUAL:
            return value != self.expected, "value inequality"

        if operator is ComparisonOperator.IN_SET:
            return value in self.allowed_values, "allowed value membership"

        numeric = self._numeric(value)
        if numeric is None:
            return False, "measurement is not numeric"

        if operator is ComparisonOperator.LESS_THAN:
            bound = self._numeric(self.expected)
            return (
                bound is not None and numeric < bound,
                "numeric less-than constraint",
            )

        if operator is ComparisonOperator.LESS_OR_EQUAL:
            bound = self._numeric(self.expected)
            return (
                bound is not None and numeric <= bound,
                "numeric upper-bound constraint",
            )

        if operator is ComparisonOperator.GREATER_THAN:
            bound = self._numeric(self.expected)
            return (
                bound is not None and numeric > bound,
                "numeric greater-than constraint",
            )

        if operator is ComparisonOperator.GREATER_OR_EQUAL:
            bound = self._numeric(self.expected)
            return (
                bound is not None and numeric >= bound,
                "numeric lower-bound constraint",
            )

        if operator is ComparisonOperator.BETWEEN_INCLUSIVE:
            assert self.minimum is not None
            assert self.maximum is not None
            return (
                self.minimum <= numeric <= self.maximum,
                "numeric inclusive range constraint",
            )

        raise ValueError(f"unsupported comparison operator: {operator}")

    def evaluate(self, context: AdmissionContext) -> AdmissionCheck:
        measurement = context.quality.measurement(
            self.measurement_name
        )

        if (
            measurement is None
            or measurement.status
            is not MeasurementAvailability.OBSERVED
        ):
            reason = (
                measurement.reason
                if measurement is not None and measurement.reason
                else "required measurement unavailable"
            )

            return AdmissionCheck(
                rule_id=self.rule_id,
                status=CheckStatus.MISSING,
                failure_disposition=self.missing_disposition,
                reason=reason,
                observed=None,
                expected=self.policy_payload(),
                source=(
                    measurement.source
                    if measurement is not None
                    else "quality"
                ),
            )

        passed, comparison_reason = self._compare(
            measurement.value
        )

        return AdmissionCheck(
            rule_id=self.rule_id,
            status=(
                CheckStatus.PASS if passed else CheckStatus.FAIL
            ),
            failure_disposition=self.failure_disposition,
            reason=comparison_reason,
            observed=measurement.value,
            expected=self.policy_payload(),
            source=measurement.source,
        )


@dataclass(frozen=True, slots=True)
class RequiredModelEvidenceRule:
    """Require named model observations to be present in quality coverage."""

    rule_id: str
    required_keys: Tuple[str, ...]
    missing_disposition: FailureDisposition = FailureDisposition.DENY

    def __post_init__(self) -> None:
        rule_id = str(self.rule_id or "").strip()
        required = tuple(
            sorted(
                {
                    str(key or "").strip()
                    for key in self.required_keys
                    if str(key or "").strip()
                }
            )
        )

        if not rule_id:
            raise ValueError("rule_id is required")
        if not required:
            raise ValueError("required_keys cannot be empty")

        object.__setattr__(self, "rule_id", rule_id)
        object.__setattr__(self, "required_keys", required)

    def policy_payload(self) -> Mapping[str, object]:
        return {
            "type": "required_model_evidence",
            "rule_id": self.rule_id,
            "required_keys": self.required_keys,
            "missing_disposition": self.missing_disposition.value,
        }

    def evaluate(self, context: AdmissionContext) -> AdmissionCheck:
        present = set(context.quality.model_evidence.present_keys)
        missing = tuple(
            key for key in self.required_keys if key not in present
        )

        return AdmissionCheck(
            rule_id=self.rule_id,
            status=(
                CheckStatus.PASS
                if not missing
                else CheckStatus.MISSING
            ),
            failure_disposition=self.missing_disposition,
            reason=(
                "all required model observations present"
                if not missing
                else f"missing model observations: {missing}"
            ),
            observed=tuple(sorted(present)),
            expected=self.required_keys,
            source="quality.model_evidence",
        )


@dataclass(frozen=True, slots=True)
class RequiredFeatureRule:
    """Require named generic candidate features to be present."""

    rule_id: str
    required_names: Tuple[str, ...]
    missing_disposition: FailureDisposition = FailureDisposition.DENY

    def __post_init__(self) -> None:
        rule_id = str(self.rule_id or "").strip()
        required = tuple(
            sorted(
                {
                    str(name or "").strip()
                    for name in self.required_names
                    if str(name or "").strip()
                }
            )
        )

        if not rule_id:
            raise ValueError("rule_id is required")
        if not required:
            raise ValueError("required_names cannot be empty")

        object.__setattr__(self, "rule_id", rule_id)
        object.__setattr__(self, "required_names", required)

    def policy_payload(self) -> Mapping[str, object]:
        return {
            "type": "required_feature",
            "rule_id": self.rule_id,
            "required_names": self.required_names,
            "missing_disposition": self.missing_disposition.value,
        }

    def evaluate(self, context: AdmissionContext) -> AdmissionCheck:
        present = set(context.quality.feature_evidence.present_names)
        missing = tuple(
            name for name in self.required_names if name not in present
        )

        return AdmissionCheck(
            rule_id=self.rule_id,
            status=(
                CheckStatus.PASS
                if not missing
                else CheckStatus.MISSING
            ),
            failure_disposition=self.missing_disposition,
            reason=(
                "all required features present"
                if not missing
                else f"missing candidate features: {missing}"
            ),
            observed=tuple(sorted(present)),
            expected=self.required_names,
            source="quality.feature_evidence",
        )


def _json_safe(value: object) -> object:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(
                value.items(),
                key=lambda row: str(row[0]),
            )
        }

    if isinstance(value, (tuple, list, set)):
        items = [_json_safe(item) for item in value]
        if isinstance(value, set):
            items = sorted(items, key=repr)
        return items

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value cannot be fingerprinted")
        return {"float_hex": value.hex()}

    return str(value)


def _stable_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _digest(payload: Mapping[str, object], *, size: int = 16) -> str:
    return hashlib.blake2b(
        _stable_json(payload).encode("utf-8"),
        digest_size=size,
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class AdmissionPermit:
    """Immutable, one-candidate admission authority.

    A permit records the exact evidence envelope that was allowed.  Later
    pre-submit validation can reject a stale/mismatched candidate without
    reconstructing admission from mutable state.
    """

    permit_id: str

    candidate_id: str
    candidate_fingerprint: str
    attempt_id: str

    market_id: str
    token_id: str

    snapshot_id: str
    candidate_market_data_generation: int
    measured_market_data_generation: Optional[int]

    quote_fingerprint: str

    policy_name: str
    policy_generation: int
    policy_fingerprint: str

    issued_at: float
    expires_at: Optional[float]

    quality_measured_at: float
    risk_observed_at: float

    decision_fingerprint: str

    def __post_init__(self) -> None:
        string_fields = (
            "permit_id",
            "candidate_id",
            "candidate_fingerprint",
            "attempt_id",
            "market_id",
            "token_id",
            "snapshot_id",
            "quote_fingerprint",
            "policy_name",
            "policy_fingerprint",
            "decision_fingerprint",
        )

        for name in string_fields:
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, value)

        generation = int(self.policy_generation)
        candidate_generation = int(
            self.candidate_market_data_generation
        )

        if generation < 0 or candidate_generation < 0:
            raise ValueError("generations must be non-negative")

        measured_generation = self.measured_market_data_generation
        if measured_generation is not None:
            measured_generation = int(measured_generation)
            if measured_generation < 0:
                raise ValueError(
                    "measured_market_data_generation must be non-negative"
                )

        issued = float(self.issued_at)
        quality_ts = float(self.quality_measured_at)
        risk_ts = float(self.risk_observed_at)

        if not all(
            math.isfinite(value) and value > 0.0
            for value in (issued, quality_ts, risk_ts)
        ):
            raise ValueError("permit timestamps must be positive")

        expires = self.expires_at
        if expires is not None:
            expires = float(expires)
            if not math.isfinite(expires) or expires <= issued:
                raise ValueError("expires_at must be after issued_at")

        object.__setattr__(self, "policy_generation", generation)
        object.__setattr__(
            self,
            "candidate_market_data_generation",
            candidate_generation,
        )
        object.__setattr__(
            self,
            "measured_market_data_generation",
            measured_generation,
        )
        object.__setattr__(self, "issued_at", issued)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(
            self,
            "quality_measured_at",
            quality_ts,
        )
        object.__setattr__(self, "risk_observed_at", risk_ts)

    def expired(self, *, now: Optional[float] = None) -> bool:
        if self.expires_at is None:
            return False
        current = float(now or time.time())
        return current > self.expires_at + 1e-9

    def matches_candidate(
        self,
        candidate: StrategyCandidate,
    ) -> bool:
        return bool(
            self.candidate_id == candidate.candidate_id
            and self.candidate_fingerprint == candidate.fingerprint
            and self.attempt_id == candidate.attempt_id
            and self.market_id == candidate.market_id
            and self.token_id == candidate.token_id
            and self.snapshot_id
            == candidate.market_snapshot.snapshot_id
            and self.candidate_market_data_generation
            == candidate.market_snapshot.market_data_generation
            and self.quote_fingerprint
            == AdmissionPolicy.quote_fingerprint(candidate)
        )


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    verdict: AdmissionVerdict

    candidate_id: str
    candidate_fingerprint: str

    policy_name: str
    policy_generation: int
    policy_fingerprint: str

    evaluated_at: float
    checks: Tuple[AdmissionCheck, ...]

    decision_fingerprint: str
    permit: Optional[AdmissionPermit] = None

    @property
    def allowed(self) -> bool:
        return self.verdict is AdmissionVerdict.ALLOW

    @property
    def denied(self) -> bool:
        return self.verdict is AdmissionVerdict.DENY

    @property
    def deferred(self) -> bool:
        return self.verdict is AdmissionVerdict.DEFER

    @property
    def blocking_checks(self) -> Tuple[AdmissionCheck, ...]:
        return tuple(check for check in self.checks if check.blocks)

    @property
    def deferring_checks(self) -> Tuple[AdmissionCheck, ...]:
        return tuple(check for check in self.checks if check.defers)


@dataclass(frozen=True, slots=True)
class AdmissionPolicyConfig:
    """Generic policy engine behavior; contains no strategy thresholds."""

    name: str = "public-admission-policy"
    generation: int = 0

    permit_ttl_seconds: Optional[float] = None

    require_candidate_not_expired: bool = True
    require_risk_approval: bool = True

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
                raise ValueError(
                    "permit_ttl_seconds must be positive"
                )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "permit_ttl_seconds", ttl)


class AdmissionPolicy:
    """Evaluate structural bindings, risk capacity, and configured rules."""

    def __init__(
        self,
        *,
        rules: Sequence[AdmissionRule] = (),
        config: AdmissionPolicyConfig = AdmissionPolicyConfig(),
    ) -> None:
        self.rules = tuple(rules)
        self.config = config

        rule_ids: set[str] = set()
        for rule in self.rules:
            rule_id = str(rule.rule_id or "").strip()
            if not rule_id:
                raise ValueError("all admission rules need rule_id")
            if rule_id in rule_ids:
                raise ValueError(
                    f"duplicate admission rule_id: {rule_id}"
                )
            rule_ids.add(rule_id)

        self._policy_fingerprint = _digest(
            self.policy_payload(),
            size=16,
        )

    @property
    def policy_fingerprint(self) -> str:
        return self._policy_fingerprint

    def policy_payload(self) -> Mapping[str, object]:
        return {
            "name": self.config.name,
            "generation": self.config.generation,
            "permit_ttl_seconds": (
                self.config.permit_ttl_seconds
            ),
            "require_candidate_not_expired": (
                self.config.require_candidate_not_expired
            ),
            "require_risk_approval": (
                self.config.require_risk_approval
            ),
            "rules": tuple(
                rule.policy_payload() for rule in self.rules
            ),
        }

    @staticmethod
    def quote_fingerprint(
        candidate: StrategyCandidate,
    ) -> str:
        quote = candidate.quote
        return _digest(
            {
                "candidate_id": candidate.candidate_id,
                "attempt_id": candidate.attempt_id,
                "market_id": candidate.market_id,
                "token_id": candidate.token_id,
                "order_side": quote.order_side.value,
                "purpose": quote.purpose.value,
                "style": quote.style.value,
                "limit_price": quote.limit_price,
                "quantity": quote.quantity,
                "time_in_force": quote.time_in_force.value,
                "paired_token_id": quote.paired_token_id,
                "client_reference": quote.client_reference,
            },
            size=16,
        )

    # ------------------------------------------------------------------
    # Canonical structural checks
    # ------------------------------------------------------------------

    def _binding_checks(
        self,
        context: AdmissionContext,
    ) -> list[AdmissionCheck]:
        candidate = context.candidate
        quality = context.quality
        proposal = context.risk_proposal
        risk = context.risk_decision

        checks: list[AdmissionCheck] = []

        identity_ok = bool(
            quality.candidate_id == candidate.candidate_id
            and quality.candidate_fingerprint
            == candidate.fingerprint
        )
        checks.append(
            AdmissionCheck(
                rule_id="binding.candidate_quality_identity",
                status=(
                    CheckStatus.PASS
                    if identity_ok
                    else CheckStatus.FAIL
                ),
                failure_disposition=FailureDisposition.DENY,
                reason=(
                    "quality evidence bound to candidate"
                    if identity_ok
                    else "quality candidate identity mismatch"
                ),
                observed=(
                    quality.candidate_id,
                    quality.candidate_fingerprint,
                ),
                expected=(
                    candidate.candidate_id,
                    candidate.fingerprint,
                ),
                source="quality",
            )
        )

        snapshot_ok = bool(
            quality.provenance.candidate_snapshot_id
            == candidate.market_snapshot.snapshot_id
            and quality.provenance.candidate_market_data_generation
            == candidate.market_snapshot.market_data_generation
        )
        checks.append(
            AdmissionCheck(
                rule_id="binding.origin_snapshot",
                status=(
                    CheckStatus.PASS
                    if snapshot_ok
                    else CheckStatus.FAIL
                ),
                failure_disposition=FailureDisposition.DENY,
                reason=(
                    "quality provenance preserves candidate origin snapshot"
                    if snapshot_ok
                    else "candidate origin snapshot mismatch"
                ),
                observed=(
                    quality.provenance.candidate_snapshot_id,
                    quality.provenance.candidate_market_data_generation,
                ),
                expected=(
                    candidate.market_snapshot.snapshot_id,
                    candidate.market_snapshot.market_data_generation,
                ),
                source="quality.provenance",
            )
        )

        market_ok = bool(
            quality.provenance.measured_market_id
            == candidate.market_id
        )
        checks.append(
            AdmissionCheck(
                rule_id="binding.measured_market",
                status=(
                    CheckStatus.PASS
                    if market_ok
                    else CheckStatus.FAIL
                ),
                failure_disposition=FailureDisposition.DENY,
                reason=(
                    "measured market matches candidate"
                    if market_ok
                    else "measured market mismatch"
                ),
                observed=quality.provenance.measured_market_id,
                expected=candidate.market_id,
                source="quality.provenance",
            )
        )

        subject_ok = bool(
            proposal.token_id == candidate.token_id
            and proposal.market_id == candidate.market_id
            and (
                proposal.outcome_side is None
                or proposal.outcome_side
                is candidate.subject.outcome_side
            )
            and abs(
                proposal.quantity - candidate.quote.quantity
            )
            <= 1e-9
        )
        checks.append(
            AdmissionCheck(
                rule_id="binding.risk_proposal_subject",
                status=(
                    CheckStatus.PASS
                    if subject_ok
                    else CheckStatus.FAIL
                ),
                failure_disposition=FailureDisposition.DENY,
                reason=(
                    "risk proposal bound to candidate economics"
                    if subject_ok
                    else "risk proposal/candidate subject mismatch"
                ),
                observed={
                    "token_id": proposal.token_id,
                    "market_id": proposal.market_id,
                    "outcome_side": (
                        proposal.outcome_side.value
                        if proposal.outcome_side is not None
                        else None
                    ),
                    "quantity": proposal.quantity,
                },
                expected={
                    "token_id": candidate.token_id,
                    "market_id": candidate.market_id,
                    "outcome_side": candidate.subject.outcome_side.value,
                    "quantity": candidate.quote.quantity,
                },
                source="risk.proposal",
            )
        )

        risk_action_ok = risk.action is proposal.action
        checks.append(
            AdmissionCheck(
                rule_id="binding.risk_decision_action",
                status=(
                    CheckStatus.PASS
                    if risk_action_ok
                    else CheckStatus.FAIL
                ),
                failure_disposition=FailureDisposition.DENY,
                reason=(
                    "risk decision action matches proposal"
                    if risk_action_ok
                    else "risk decision action mismatch"
                ),
                observed=risk.action.value,
                expected=proposal.action.value,
                source="risk.decision",
            )
        )

        if self.config.require_candidate_not_expired:
            not_expired = not candidate.expired(
                now=context.evaluated_at
            )
            checks.append(
                AdmissionCheck(
                    rule_id="candidate.explicit_validity",
                    status=(
                        CheckStatus.PASS
                        if not_expired
                        else CheckStatus.FAIL
                    ),
                    failure_disposition=FailureDisposition.DENY,
                    reason=(
                        "candidate explicit validity window active"
                        if not_expired
                        else "candidate explicit validity window expired"
                    ),
                    observed=context.evaluated_at,
                    expected=candidate.valid_until,
                    source="candidate",
                )
            )

        if self.config.require_risk_approval:
            checks.append(
                AdmissionCheck(
                    rule_id="risk.approval",
                    status=(
                        CheckStatus.PASS
                        if risk.allowed
                        else CheckStatus.FAIL
                    ),
                    failure_disposition=FailureDisposition.DENY,
                    reason=(
                        "risk capacity approved"
                        if risk.allowed
                        else "risk capacity denied"
                    ),
                    observed={
                        "allowed": risk.allowed,
                        "mode": risk.mode.value,
                        "hard_failures": tuple(
                            check.code
                            for check in risk.hard_failures
                        ),
                    },
                    expected=True,
                    source="risk.decision",
                )
            )

        return checks

    # ------------------------------------------------------------------
    # Final decision / immutable permit
    # ------------------------------------------------------------------

    def evaluate(
        self,
        context: AdmissionContext,
    ) -> AdmissionDecision:
        checks = self._binding_checks(context)

        for rule in self.rules:
            try:
                check = rule.evaluate(context)
            except Exception as exc:
                check = AdmissionCheck(
                    rule_id=str(rule.rule_id),
                    status=CheckStatus.FAIL,
                    failure_disposition=FailureDisposition.DENY,
                    reason=(
                        f"rule evaluation error: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    source="admission.rule",
                )
            checks.append(check)

        has_deny = any(check.blocks for check in checks)
        has_defer = any(check.defers for check in checks)

        if has_deny:
            verdict = AdmissionVerdict.DENY
        elif has_defer:
            verdict = AdmissionVerdict.DEFER
        else:
            verdict = AdmissionVerdict.ALLOW

        decision_payload = {
            "candidate_id": context.candidate.candidate_id,
            "candidate_fingerprint": (
                context.candidate.fingerprint
            ),
            "policy_name": self.config.name,
            "policy_generation": self.config.generation,
            "policy_fingerprint": self.policy_fingerprint,
            "evaluated_at": context.evaluated_at,
            "verdict": verdict.value,
            "quality_provenance": {
                "candidate_snapshot_id": (
                    context.quality.provenance.candidate_snapshot_id
                ),
                "candidate_market_data_generation": (
                    context.quality.provenance
                    .candidate_market_data_generation
                ),
                "measured_market_data_generation": (
                    context.quality.provenance
                    .measured_market_data_generation
                ),
                "measured_at": (
                    context.quality.provenance.measured_at
                ),
            },
            "risk": {
                "action": context.risk_decision.action.value,
                "mode": context.risk_decision.mode.value,
                "allowed": context.risk_decision.allowed,
                "observed_at": (
                    context.risk_decision.snapshot.observed_at
                ),
            },
            "checks": tuple(
                {
                    "rule_id": check.rule_id,
                    "status": check.status.value,
                    "failure_disposition": (
                        check.failure_disposition.value
                    ),
                    "reason": check.reason,
                    "observed": check.observed,
                    "expected": check.expected,
                    "source": check.source,
                }
                for check in checks
            ),
        }

        decision_fingerprint = _digest(
            decision_payload,
            size=16,
        )

        permit = None

        if verdict is AdmissionVerdict.ALLOW:
            issued_at = context.evaluated_at
            expires_at = None

            ttl = self.config.permit_ttl_seconds
            if ttl is not None:
                expires_at = issued_at + ttl

            # A candidate's own explicit validity window is an upper bound on
            # any admission permit created from that candidate.
            if context.candidate.valid_until is not None:
                expires_at = (
                    context.candidate.valid_until
                    if expires_at is None
                    else min(
                        expires_at,
                        context.candidate.valid_until,
                    )
                )

            permit_payload = {
                "candidate_id": context.candidate.candidate_id,
                "candidate_fingerprint": (
                    context.candidate.fingerprint
                ),
                "attempt_id": context.candidate.attempt_id,
                "market_id": context.candidate.market_id,
                "token_id": context.candidate.token_id,
                "snapshot_id": (
                    context.candidate.market_snapshot.snapshot_id
                ),
                "candidate_market_data_generation": (
                    context.candidate.market_snapshot
                    .market_data_generation
                ),
                "measured_market_data_generation": (
                    context.quality.provenance
                    .measured_market_data_generation
                ),
                "quote_fingerprint": self.quote_fingerprint(
                    context.candidate
                ),
                "policy_name": self.config.name,
                "policy_generation": self.config.generation,
                "policy_fingerprint": self.policy_fingerprint,
                "issued_at": issued_at,
                "expires_at": expires_at,
                "quality_measured_at": (
                    context.quality.provenance.measured_at
                ),
                "risk_observed_at": (
                    context.risk_decision.snapshot.observed_at
                ),
                "decision_fingerprint": decision_fingerprint,
            }

            permit_id = f"permit-{_digest(permit_payload, size=12)}"

            permit = AdmissionPermit(
                permit_id=permit_id,
                **permit_payload,
            )

        return AdmissionDecision(
            verdict=verdict,
            candidate_id=context.candidate.candidate_id,
            candidate_fingerprint=context.candidate.fingerprint,
            policy_name=self.config.name,
            policy_generation=self.config.generation,
            policy_fingerprint=self.policy_fingerprint,
            evaluated_at=context.evaluated_at,
            checks=tuple(checks),
            decision_fingerprint=decision_fingerprint,
            permit=permit,
        )


@dataclass(frozen=True, slots=True)
class PermitConsumption:
    """Immutable record of a one-shot permit handoff."""

    permit: AdmissionPermit
    consumer: str
    consumed_at: float

    def __post_init__(self) -> None:
        consumer = str(self.consumer or "").strip()
        consumed_at = float(self.consumed_at)

        if not consumer:
            raise ValueError("consumer is required")
        if not math.isfinite(consumed_at) or consumed_at <= 0.0:
            raise ValueError("consumed_at must be positive")

        object.__setattr__(self, "consumer", consumer)
        object.__setattr__(self, "consumed_at", consumed_at)


class AdmissionPermitLedger:
    """Thread-safe one-shot permit registry.

    This ledger is intentionally small. It is a producer->consumer continuity
    mechanism, not a second source of candidate or execution truth.
    """

    def __init__(self, *, max_entries: int = 2048) -> None:
        max_entries = int(max_entries)
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")

        self.max_entries = max_entries
        self._gate = threading.RLock()

        self._permits: Dict[str, AdmissionPermit] = {}
        self._permit_by_candidate_id: Dict[str, str] = {}
        self._consumed: Dict[str, PermitConsumption] = {}

    def register(
        self,
        decision: AdmissionDecision,
    ) -> AdmissionPermit:
        permit = decision.permit

        if not decision.allowed or permit is None:
            raise ValueError(
                "only ALLOW decisions with a permit may be registered"
            )

        with self._gate:
            existing = self._permits.get(permit.permit_id)

            if existing is not None and existing != permit:
                raise ValueError(
                    "permit_id cannot be reused for another envelope"
                )

            prior_id = self._permit_by_candidate_id.get(
                permit.candidate_id
            )
            if prior_id and prior_id != permit.permit_id:
                prior = self._permits.get(prior_id)

                if prior is not None and not prior.expired():
                    raise ValueError(
                        "candidate already has a different active permit"
                    )

            self._permits[permit.permit_id] = permit
            self._permit_by_candidate_id[
                permit.candidate_id
            ] = permit.permit_id

            self._prune_bound_locked()
            return permit

    def get(
        self,
        permit_id: str,
    ) -> Optional[AdmissionPermit]:
        with self._gate:
            return self._permits.get(str(permit_id or ""))

    def for_candidate(
        self,
        candidate_id: str,
    ) -> Optional[AdmissionPermit]:
        candidate_id = str(candidate_id or "")

        with self._gate:
            permit_id = self._permit_by_candidate_id.get(
                candidate_id
            )
            return (
                self._permits.get(permit_id)
                if permit_id is not None
                else None
            )

    def consumed(
        self,
        permit_id: str,
    ) -> Optional[PermitConsumption]:
        with self._gate:
            return self._consumed.get(str(permit_id or ""))

    def consume(
        self,
        permit_id: str,
        *,
        candidate: StrategyCandidate,
        consumer: str,
        now: Optional[float] = None,
    ) -> PermitConsumption:
        permit_id = str(permit_id or "").strip()
        consumer = str(consumer or "").strip()
        consumed_at = float(now or time.time())

        if not permit_id:
            raise ValueError("permit_id is required")
        if not consumer:
            raise ValueError("consumer is required")

        with self._gate:
            prior = self._consumed.get(permit_id)
            if prior is not None:
                raise ValueError(
                    f"permit already consumed by {prior.consumer}"
                )

            permit = self._permits.get(permit_id)
            if permit is None:
                raise KeyError(f"unknown permit: {permit_id}")

            if permit.expired(now=consumed_at):
                raise ValueError("admission permit expired")

            if not permit.matches_candidate(candidate):
                raise ValueError(
                    "admission permit does not match candidate envelope"
                )

            consumption = PermitConsumption(
                permit=permit,
                consumer=consumer,
                consumed_at=consumed_at,
            )

            self._consumed[permit_id] = consumption
            self._permits.pop(permit_id, None)

            if (
                self._permit_by_candidate_id.get(
                    permit.candidate_id
                )
                == permit_id
            ):
                self._permit_by_candidate_id.pop(
                    permit.candidate_id,
                    None,
                )

            return consumption

    def prune(
        self,
        *,
        now: Optional[float] = None,
        consumed_keep_seconds: float = 300.0,
    ) -> dict[str, int]:
        current = float(now or time.time())
        keep = max(0.0, float(consumed_keep_seconds))

        expired_removed = 0
        consumed_removed = 0

        with self._gate:
            for permit_id, permit in list(self._permits.items()):
                if not permit.expired(now=current):
                    continue

                self._permits.pop(permit_id, None)

                if (
                    self._permit_by_candidate_id.get(
                        permit.candidate_id
                    )
                    == permit_id
                ):
                    self._permit_by_candidate_id.pop(
                        permit.candidate_id,
                        None,
                    )

                expired_removed += 1

            for permit_id, consumption in list(
                self._consumed.items()
            ):
                if current - consumption.consumed_at < keep:
                    continue

                self._consumed.pop(permit_id, None)
                consumed_removed += 1

        return {
            "expired": expired_removed,
            "consumed": consumed_removed,
        }

    def _prune_bound_locked(self) -> None:
        overflow = len(self._permits) - self.max_entries
        if overflow <= 0:
            return

        oldest = sorted(
            self._permits.values(),
            key=lambda permit: permit.issued_at,
        )[:overflow]

        for permit in oldest:
            self._permits.pop(permit.permit_id, None)

            if (
                self._permit_by_candidate_id.get(
                    permit.candidate_id
                )
                == permit.permit_id
            ):
                self._permit_by_candidate_id.pop(
                    permit.candidate_id,
                    None,
                )
