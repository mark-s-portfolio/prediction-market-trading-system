"""
Typed strategy-candidate contracts for the public portfolio edition.

This module describes *what a candidate is*.  It does not decide whether a
candidate is good, admissible, safe, or executable.

The production system historically carried candidate state through mutable
dictionaries and large signal-context payloads.  The public portfolio edition
uses immutable, explicitly scoped contracts instead:

    CandidateSubject
        stable market/token/side identity

    CandidateIdentity
        one generated candidate/attempt identity

    QuoteIntent
        proposed execution shape, not an authorization

    MarketSnapshotRef
        immutable reference to the market-data observation used to create it

    ModelObservation
        diagnostic/model outputs with generation and provenance

    CandidateProvenance
        producer and pipeline lineage

    StrategyCandidate
        immutable envelope joining the above

Core invariants:
- candidate identity is independent from mutable strategy state
- a quote intent is a proposal, never an execution permit
- model observations are evidence/diagnostics, never allow/block authority here
- the snapshot that produced a candidate is explicit and immutable
- provenance is append-only and does not silently rewrite candidate identity
- fingerprints contain no historical GOOD/BAD family or production threshold data
- candidate freshness is represented as time metadata, not a hidden policy rule
- execution lifecycle and risk state remain outside this module

No production admission thresholds, setup families, ranking weights, bankroll
parameters, or asset-specific rules are embedded here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
import time
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from src.execution.types import OrderSide, TimeInForce
from src.market.types import BookSource, MarketBooks, MarketDefinition, OutcomeSide


class CandidatePurpose(str, Enum):
    """Economic purpose of a candidate before admission."""

    OPENING = "OPENING"
    COMPLETION = "COMPLETION"
    RISK_REDUCTION = "RISK_REDUCTION"
    REBALANCE = "REBALANCE"
    UNKNOWN = "UNKNOWN"


class QuoteStyle(str, Enum):
    """Generic order-shape preference without strategy-specific thresholds."""

    PASSIVE_LIMIT = "PASSIVE_LIMIT"
    MARKETABLE_LIMIT = "MARKETABLE_LIMIT"
    LIMIT = "LIMIT"
    UNKNOWN = "UNKNOWN"


class CandidateSource(str, Enum):
    """High-level producer category for portfolio observability."""

    MARKET_DATA = "MARKET_DATA"
    MODEL = "MODEL"
    RECOVERY = "RECOVERY"
    SCHEDULER = "SCHEDULER"
    MANUAL = "MANUAL"
    UNKNOWN = "UNKNOWN"


class ObservationKind(str, Enum):
    """Semantic category for a model/diagnostic observation."""

    PROBABILITY = "PROBABILITY"
    SCORE = "SCORE"
    EXPECTED_VALUE = "EXPECTED_VALUE"
    STATE = "STATE"
    BOOLEAN = "BOOLEAN"
    COUNT = "COUNT"
    DURATION = "DURATION"
    DISTANCE = "DISTANCE"
    OTHER = "OTHER"


@dataclass(frozen=True, slots=True)
class CandidateSubject:
    """Immutable economic subject addressed by a candidate."""

    market_id: str
    token_id: str
    opposite_token_id: str
    outcome_side: OutcomeSide
    interval_minutes: int

    def __post_init__(self) -> None:
        market_id = str(self.market_id or "").strip()
        token_id = str(self.token_id or "").strip()
        opposite = str(self.opposite_token_id or "").strip()
        interval = int(self.interval_minutes)

        if not market_id:
            raise ValueError("market_id is required")
        if not token_id:
            raise ValueError("token_id is required")
        if not opposite:
            raise ValueError("opposite_token_id is required")
        if token_id == opposite:
            raise ValueError("token and opposite token must be different")
        if interval <= 0:
            raise ValueError("interval_minutes must be positive")

        object.__setattr__(self, "market_id", market_id)
        object.__setattr__(self, "token_id", token_id)
        object.__setattr__(self, "opposite_token_id", opposite)
        object.__setattr__(self, "interval_minutes", interval)

    @classmethod
    def from_market(
        cls,
        market: MarketDefinition,
        side: OutcomeSide,
    ) -> "CandidateSubject":
        opposite = (
            market.no_token
            if side is OutcomeSide.YES
            else market.yes_token
        )

        return cls(
            market_id=market.pair_key,
            token_id=market.token_for(side),
            opposite_token_id=opposite,
            outcome_side=side,
            interval_minutes=market.interval_minutes,
        )

    @property
    def subject_key(self) -> str:
        return (
            f"{self.market_id}:"
            f"{self.outcome_side.value}:"
            f"{self.token_id}"
        )


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    """Identity for one immutable candidate generation."""

    candidate_id: str
    attempt_id: str
    sequence: int
    created_at: float

    def __post_init__(self) -> None:
        candidate_id = str(self.candidate_id or "").strip()
        attempt_id = str(self.attempt_id or "").strip()
        sequence = int(self.sequence)
        created = float(self.created_at)

        if not candidate_id:
            raise ValueError("candidate_id is required")
        if not attempt_id:
            raise ValueError("attempt_id is required")
        if sequence < 0:
            raise ValueError("sequence must be non-negative")
        if not math.isfinite(created) or created <= 0.0:
            raise ValueError("created_at must be positive")

        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "attempt_id", attempt_id)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "created_at", created)


@dataclass(frozen=True, slots=True)
class QuoteIntent:
    """Proposed order economics prior to admission.

    The presence of this object does *not* authorize execution.
    """

    order_side: OrderSide
    purpose: CandidatePurpose
    style: QuoteStyle

    limit_price: float
    quantity: float
    time_in_force: TimeInForce = TimeInForce.GTC

    paired_token_id: Optional[str] = None
    client_reference: str = ""

    def __post_init__(self) -> None:
        price = float(self.limit_price)
        quantity = float(self.quantity)
        paired = str(self.paired_token_id or "").strip() or None

        if not math.isfinite(price) or not 0.0 < price <= 1.0:
            raise ValueError("limit_price must be in (0, 1]")
        if not math.isfinite(quantity) or quantity <= 0.0:
            raise ValueError("quantity must be positive")

        object.__setattr__(self, "limit_price", price)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "paired_token_id", paired)
        object.__setattr__(
            self,
            "client_reference",
            str(self.client_reference or ""),
        )

    @property
    def notional(self) -> float:
        return self.limit_price * self.quantity


@dataclass(frozen=True, slots=True)
class MarketSnapshotRef:
    """Immutable reference to the market-data observation behind a candidate.

    This is intentionally a compact reference rather than a deep copy of the full
    order book.  The market-data service remains the owner of the normalized book.
    """

    snapshot_id: str
    market_id: str

    yes_book_timestamp: float
    no_book_timestamp: float
    captured_at: float

    yes_source: BookSource
    no_source: BookSource

    yes_depth_proven: bool
    no_depth_proven: bool

    market_data_generation: int = 0

    def __post_init__(self) -> None:
        snapshot_id = str(self.snapshot_id or "").strip()
        market_id = str(self.market_id or "").strip()

        yes_ts = float(self.yes_book_timestamp)
        no_ts = float(self.no_book_timestamp)
        captured = float(self.captured_at)
        generation = int(self.market_data_generation)

        if not snapshot_id:
            raise ValueError("snapshot_id is required")
        if not market_id:
            raise ValueError("market_id is required")
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (yes_ts, no_ts, captured)
        ):
            raise ValueError("snapshot timestamps must be positive")
        if generation < 0:
            raise ValueError("market_data_generation must be non-negative")

        object.__setattr__(self, "snapshot_id", snapshot_id)
        object.__setattr__(self, "market_id", market_id)
        object.__setattr__(self, "yes_book_timestamp", yes_ts)
        object.__setattr__(self, "no_book_timestamp", no_ts)
        object.__setattr__(self, "captured_at", captured)
        object.__setattr__(self, "market_data_generation", generation)

    @property
    def oldest_book_timestamp(self) -> float:
        return min(self.yes_book_timestamp, self.no_book_timestamp)

    @property
    def newest_book_timestamp(self) -> float:
        return max(self.yes_book_timestamp, self.no_book_timestamp)

    @property
    def book_skew_seconds(self) -> float:
        return abs(self.yes_book_timestamp - self.no_book_timestamp)

    @property
    def both_depth_proven(self) -> bool:
        return self.yes_depth_proven and self.no_depth_proven

    def age_seconds(self, *, now: Optional[float] = None) -> float:
        current = float(now or time.time())
        return max(0.0, current - self.oldest_book_timestamp)

    @classmethod
    def from_books(
        cls,
        books: MarketBooks,
        *,
        snapshot_id: Optional[str] = None,
        captured_at: Optional[float] = None,
        market_data_generation: int = 0,
    ) -> "MarketSnapshotRef":
        captured = float(captured_at or time.time())

        if snapshot_id is None:
            payload = (
                books.market.pair_key,
                books.yes.token_id,
                round(books.yes.timestamp, 9),
                books.no.token_id,
                round(books.no.timestamp, 9),
                int(market_data_generation),
            )
            digest = hashlib.blake2b(
                repr(payload).encode("utf-8"),
                digest_size=12,
            ).hexdigest()
            snapshot_id = f"snapshot-{digest}"

        return cls(
            snapshot_id=snapshot_id,
            market_id=books.market.pair_key,
            yes_book_timestamp=books.yes.timestamp,
            no_book_timestamp=books.no.timestamp,
            captured_at=captured,
            yes_source=books.yes.source,
            no_source=books.no.source,
            yes_depth_proven=books.yes.depth_proven,
            no_depth_proven=books.no.depth_proven,
            market_data_generation=int(market_data_generation),
        )


@dataclass(frozen=True, slots=True)
class ModelObservation:
    """One typed model/diagnostic fact attached to a candidate.

    `value` is intentionally generic.  Model semantics belong to the named model,
    and policy semantics belong to later strategy modules.
    """

    model_name: str
    metric: str
    kind: ObservationKind
    value: object

    generation: int
    observed_at: float

    sample_size: Optional[int] = None
    source: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        model_name = str(self.model_name or "").strip()
        metric = str(self.metric or "").strip()
        generation = int(self.generation)
        observed = float(self.observed_at)

        if not model_name:
            raise ValueError("model_name is required")
        if not metric:
            raise ValueError("metric is required")
        if generation < 0:
            raise ValueError("generation must be non-negative")
        if not math.isfinite(observed) or observed <= 0.0:
            raise ValueError("observed_at must be positive")

        sample_size = self.sample_size
        if sample_size is not None:
            sample_size = int(sample_size)
            if sample_size < 0:
                raise ValueError("sample_size must be non-negative")

        # Numeric observations should not silently carry NaN/inf.
        if isinstance(self.value, (int, float)) and not isinstance(
            self.value,
            bool,
        ):
            numeric = float(self.value)
            if not math.isfinite(numeric):
                raise ValueError("numeric observation value must be finite")

        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "sample_size", sample_size)
        object.__setattr__(self, "source", str(self.source or ""))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def key(self) -> str:
        return f"{self.model_name}:{self.metric}"


@dataclass(frozen=True, slots=True)
class CandidateProvenance:
    """One append-only producer/pipeline lineage record."""

    producer: str
    source: CandidateSource
    stage: str
    observed_at: float

    event_id: str = ""
    parent_candidate_id: str = ""
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        producer = str(self.producer or "").strip()
        stage = str(self.stage or "").strip()
        observed = float(self.observed_at)

        if not producer:
            raise ValueError("producer is required")
        if not stage:
            raise ValueError("stage is required")
        if not math.isfinite(observed) or observed <= 0.0:
            raise ValueError("observed_at must be positive")

        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "event_id", str(self.event_id or ""))
        object.__setattr__(
            self,
            "parent_candidate_id",
            str(self.parent_candidate_id or ""),
        )
        object.__setattr__(self, "details", dict(self.details))


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

    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _stable_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


@dataclass(frozen=True, slots=True)
class StrategyCandidate:
    """Immutable candidate envelope passed between strategy pipeline stages."""

    identity: CandidateIdentity
    subject: CandidateSubject
    quote: QuoteIntent
    market_snapshot: MarketSnapshotRef

    model_observations: Tuple[ModelObservation, ...] = field(
        default_factory=tuple
    )
    provenance: Tuple[CandidateProvenance, ...] = field(
        default_factory=tuple
    )
    features: Mapping[str, object] = field(default_factory=dict)

    valid_until: Optional[float] = None

    def __post_init__(self) -> None:
        if self.market_snapshot.market_id != self.subject.market_id:
            raise ValueError(
                "market snapshot does not match candidate subject"
            )

        if (
            self.quote.paired_token_id is not None
            and self.quote.paired_token_id
            != self.subject.opposite_token_id
        ):
            raise ValueError(
                "quote paired_token_id does not match candidate subject"
            )

        valid_until = self.valid_until
        if valid_until is not None:
            valid_until = float(valid_until)
            if (
                not math.isfinite(valid_until)
                or valid_until <= self.identity.created_at
            ):
                raise ValueError(
                    "valid_until must be after candidate creation"
                )

        # Observation keys are unique inside one immutable envelope.  A later
        # generation should replace the observation explicitly rather than create
        # two contradictory rows for the same model metric.
        seen: set[str] = set()
        for observation in self.model_observations:
            if observation.key in seen:
                raise ValueError(
                    f"duplicate model observation key: {observation.key}"
                )
            seen.add(observation.key)

        object.__setattr__(
            self,
            "model_observations",
            tuple(self.model_observations),
        )
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "features", dict(self.features))
        object.__setattr__(self, "valid_until", valid_until)

    @property
    def candidate_id(self) -> str:
        return self.identity.candidate_id

    @property
    def attempt_id(self) -> str:
        return self.identity.attempt_id

    @property
    def token_id(self) -> str:
        return self.subject.token_id

    @property
    def market_id(self) -> str:
        return self.subject.market_id

    @property
    def created_at(self) -> float:
        return self.identity.created_at

    def age_seconds(self, *, now: Optional[float] = None) -> float:
        current = float(now or time.time())
        return max(0.0, current - self.identity.created_at)

    def expired(self, *, now: Optional[float] = None) -> bool:
        if self.valid_until is None:
            return False
        current = float(now or time.time())
        return current > self.valid_until + 1e-9

    def observation(
        self,
        model_name: str,
        metric: str,
    ) -> Optional[ModelObservation]:
        key = f"{str(model_name or '').strip()}:{str(metric or '').strip()}"

        return next(
            (
                observation
                for observation in self.model_observations
                if observation.key == key
            ),
            None,
        )

    def with_model_observation(
        self,
        observation: ModelObservation,
        *,
        provenance: Optional[CandidateProvenance] = None,
    ) -> "StrategyCandidate":
        """Return a new immutable envelope with one model metric replaced.

        Candidate identity/fingerprint does not change merely because diagnostics
        were refreshed.
        """

        observations = {
            existing.key: existing
            for existing in self.model_observations
        }

        prior = observations.get(observation.key)

        if (
            prior is not None
            and observation.generation < prior.generation
        ):
            raise ValueError(
                f"stale model generation for {observation.key}: "
                f"{observation.generation} < {prior.generation}"
            )

        observations[observation.key] = observation

        provenance_rows = self.provenance
        if provenance is not None:
            provenance_rows = provenance_rows + (provenance,)

        return replace(
            self,
            model_observations=tuple(
                observations[key]
                for key in sorted(observations)
            ),
            provenance=provenance_rows,
        )

    def with_feature(
        self,
        name: str,
        value: object,
        *,
        provenance: Optional[CandidateProvenance] = None,
    ) -> "StrategyCandidate":
        """Return a new candidate view with one non-authoritative feature."""

        name = str(name or "").strip()
        if not name:
            raise ValueError("feature name is required")

        features = dict(self.features)
        features[name] = value

        provenance_rows = self.provenance
        if provenance is not None:
            provenance_rows = provenance_rows + (provenance,)

        return replace(
            self,
            features=features,
            provenance=provenance_rows,
        )

    def append_provenance(
        self,
        row: CandidateProvenance,
    ) -> "StrategyCandidate":
        return replace(
            self,
            provenance=self.provenance + (row,),
        )

    def immutable_subject_payload(self) -> dict[str, object]:
        """Canonical payload used for stable candidate subject fingerprinting."""

        return {
            "candidate_id": self.identity.candidate_id,
            "attempt_id": self.identity.attempt_id,
            "sequence": self.identity.sequence,
            "subject": {
                "market_id": self.subject.market_id,
                "token_id": self.subject.token_id,
                "opposite_token_id": self.subject.opposite_token_id,
                "outcome_side": self.subject.outcome_side.value,
                "interval_minutes": self.subject.interval_minutes,
            },
            "quote": {
                "order_side": self.quote.order_side.value,
                "purpose": self.quote.purpose.value,
                "style": self.quote.style.value,
                "limit_price": round(self.quote.limit_price, 12),
                "quantity": round(self.quote.quantity, 12),
                "time_in_force": self.quote.time_in_force.value,
                "paired_token_id": self.quote.paired_token_id,
                "client_reference": self.quote.client_reference,
            },
            "snapshot": {
                "snapshot_id": self.market_snapshot.snapshot_id,
                "market_id": self.market_snapshot.market_id,
                "market_data_generation": (
                    self.market_snapshot.market_data_generation
                ),
            },
        }

    @property
    def fingerprint(self) -> str:
        """Stable fingerprint of the immutable candidate subject.

        Model observations, diagnostic features and later provenance are excluded
        so they cannot retroactively change which candidate was originally created.
        """

        digest = hashlib.blake2b(
            _stable_json(
                self.immutable_subject_payload()
            ).encode("utf-8"),
            digest_size=16,
        ).hexdigest()

        return f"candidate-{digest}"

    def diagnostic_payload(self) -> dict[str, object]:
        """Serializable public observability view.

        This remains descriptive.  It contains no admission verdict.
        """

        return {
            "candidate_id": self.identity.candidate_id,
            "attempt_id": self.identity.attempt_id,
            "fingerprint": self.fingerprint,
            "subject": {
                "market_id": self.subject.market_id,
                "token_id": self.subject.token_id,
                "opposite_token_id": self.subject.opposite_token_id,
                "outcome_side": self.subject.outcome_side.value,
                "interval_minutes": self.subject.interval_minutes,
            },
            "quote": {
                "order_side": self.quote.order_side.value,
                "purpose": self.quote.purpose.value,
                "style": self.quote.style.value,
                "limit_price": self.quote.limit_price,
                "quantity": self.quote.quantity,
                "notional": self.quote.notional,
                "time_in_force": self.quote.time_in_force.value,
            },
            "market_snapshot": {
                "snapshot_id": self.market_snapshot.snapshot_id,
                "yes_book_timestamp": (
                    self.market_snapshot.yes_book_timestamp
                ),
                "no_book_timestamp": (
                    self.market_snapshot.no_book_timestamp
                ),
                "yes_source": self.market_snapshot.yes_source.value,
                "no_source": self.market_snapshot.no_source.value,
                "both_depth_proven": (
                    self.market_snapshot.both_depth_proven
                ),
                "market_data_generation": (
                    self.market_snapshot.market_data_generation
                ),
            },
            "model_observations": [
                {
                    "model_name": observation.model_name,
                    "metric": observation.metric,
                    "kind": observation.kind.value,
                    "value": _json_safe(observation.value),
                    "generation": observation.generation,
                    "sample_size": observation.sample_size,
                    "source": observation.source,
                }
                for observation in self.model_observations
            ],
            "features": _json_safe(self.features),
            "provenance": [
                {
                    "producer": row.producer,
                    "source": row.source.value,
                    "stage": row.stage,
                    "event_id": row.event_id,
                    "parent_candidate_id": row.parent_candidate_id,
                    "observed_at": row.observed_at,
                }
                for row in self.provenance
            ],
            "created_at": self.identity.created_at,
            "valid_until": self.valid_until,
        }


class CandidateFactory:
    """Small identity factory for immutable candidate creation.

    The factory only allocates IDs and joins contracts; it performs no market
    scoring, ranking, or admission.
    """

    def __init__(self, *, namespace: str = "candidate") -> None:
        namespace = str(namespace or "").strip()
        if not namespace:
            raise ValueError("namespace is required")

        self.namespace = namespace
        self._sequence = 0

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def create(
        self,
        *,
        subject: CandidateSubject,
        quote: QuoteIntent,
        market_snapshot: MarketSnapshotRef,
        attempt_id: str,
        producer: str,
        source: CandidateSource = CandidateSource.MARKET_DATA,
        event_id: str = "",
        created_at: Optional[float] = None,
        valid_until: Optional[float] = None,
        model_observations: Iterable[ModelObservation] = (),
        features: Optional[Mapping[str, object]] = None,
    ) -> StrategyCandidate:
        created = float(created_at or time.time())
        sequence = self._next_sequence()

        identity_seed = {
            "namespace": self.namespace,
            "attempt_id": str(attempt_id or ""),
            "sequence": sequence,
            "created_at": round(created, 9),
            "market_id": subject.market_id,
            "token_id": subject.token_id,
            "snapshot_id": market_snapshot.snapshot_id,
        }

        digest = hashlib.blake2b(
            _stable_json(identity_seed).encode("utf-8"),
            digest_size=12,
        ).hexdigest()

        identity = CandidateIdentity(
            candidate_id=f"{self.namespace}-{digest}",
            attempt_id=str(attempt_id or ""),
            sequence=sequence,
            created_at=created,
        )

        provenance = CandidateProvenance(
            producer=producer,
            source=source,
            stage="GENERATED",
            observed_at=created,
            event_id=event_id,
        )

        return StrategyCandidate(
            identity=identity,
            subject=subject,
            quote=quote,
            market_snapshot=market_snapshot,
            model_observations=tuple(model_observations),
            provenance=(provenance,),
            features=dict(features or {}),
            valid_until=valid_until,
        )


class CandidateRegistry:
    """Bounded immutable candidate registry for scheduler/diagnostic use.

    This is not a ranking engine.  It keeps identity/provenance reachable across
    short asynchronous handoffs without introducing a second strategy truth.
    """

    def __init__(self, *, max_entries: int = 1024) -> None:
        max_entries = int(max_entries)
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")

        self.max_entries = max_entries
        self._by_id: Dict[str, StrategyCandidate] = {}
        self._latest_by_subject: Dict[str, str] = {}

    def put(self, candidate: StrategyCandidate) -> StrategyCandidate:
        self._by_id[candidate.candidate_id] = candidate
        self._latest_by_subject[
            candidate.subject.subject_key
        ] = candidate.candidate_id

        self._prune_bound()
        return candidate

    def get(self, candidate_id: str) -> Optional[StrategyCandidate]:
        return self._by_id.get(str(candidate_id or ""))

    def latest_for_subject(
        self,
        subject: CandidateSubject,
    ) -> Optional[StrategyCandidate]:
        candidate_id = self._latest_by_subject.get(subject.subject_key)
        return self._by_id.get(candidate_id) if candidate_id else None

    def replace_view(
        self,
        candidate: StrategyCandidate,
    ) -> StrategyCandidate:
        """Replace diagnostics/provenance for the same immutable candidate ID."""

        existing = self._by_id.get(candidate.candidate_id)

        if existing is not None and existing.fingerprint != candidate.fingerprint:
            raise ValueError(
                "candidate_id cannot be reused for a different immutable subject"
            )

        return self.put(candidate)

    def values(self) -> Tuple[StrategyCandidate, ...]:
        return tuple(
            sorted(
                self._by_id.values(),
                key=lambda candidate: (
                    candidate.identity.created_at,
                    candidate.identity.sequence,
                ),
            )
        )

    def prune_expired(
        self,
        *,
        now: Optional[float] = None,
    ) -> int:
        current = float(now or time.time())
        removed = 0

        for candidate_id, candidate in list(self._by_id.items()):
            if not candidate.expired(now=current):
                continue

            self._remove(candidate_id)
            removed += 1

        return removed

    def _remove(self, candidate_id: str) -> None:
        candidate = self._by_id.pop(candidate_id, None)
        if candidate is None:
            return

        subject_key = candidate.subject.subject_key

        if self._latest_by_subject.get(subject_key) == candidate_id:
            replacement = max(
                (
                    row
                    for row in self._by_id.values()
                    if row.subject.subject_key == subject_key
                ),
                key=lambda row: (
                    row.identity.created_at,
                    row.identity.sequence,
                ),
                default=None,
            )

            if replacement is None:
                self._latest_by_subject.pop(subject_key, None)
            else:
                self._latest_by_subject[
                    subject_key
                ] = replacement.candidate_id

    def _prune_bound(self) -> None:
        overflow = len(self._by_id) - self.max_entries
        if overflow <= 0:
            return

        oldest = sorted(
            self._by_id.values(),
            key=lambda candidate: (
                candidate.identity.created_at,
                candidate.identity.sequence,
            ),
        )[:overflow]

        for candidate in oldest:
            self._remove(candidate.candidate_id)
