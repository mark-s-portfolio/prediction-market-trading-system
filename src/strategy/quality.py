"""
Measurement-only candidate quality facts for the public portfolio edition.

This module measures the state attached to a StrategyCandidate.  It deliberately
does not rank candidates, compute a composite quality score, or issue an
admission verdict.

The separation is intentional:

    candidate.py
        immutable subject / proposal / provenance

    quality.py
        observable market-data and model-evidence facts

    admission.py
        later policy consumer of those facts

Public responsibilities:
- measure candidate and snapshot age
- measure YES/NO book skew
- expose top-of-book prices, spreads, sizes, and level counts
- distinguish proven depth from synthetic/top-only depth
- expose known aggregate depth only when depth is actually proven
- detect crossed or one-sided books as descriptive facts
- compare the current market-data view with the candidate's source snapshot
- measure quote position relative to the current book
- summarize model-observation coverage, generations, and age
- summarize named feature coverage
- preserve measurement provenance and missing-data reasons

Core invariants:
- missing evidence remains missing
- top-only/synthetic depth is not promoted to proven depth
- measurement age is a fact, not a hidden freshness threshold
- a newer current book does not rewrite the candidate's original snapshot identity
- model output remains descriptive evidence
- no aggregate quality score is produced
- no admission or execution authority is produced here

No production strategy thresholds, setup-family knowledge, asset-specific rules,
historical trade labels, ranking weights, or bankroll parameters are embedded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
import time
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from src.execution.types import OrderSide
from src.market.types import MarketBooks, OrderBookSnapshot
from src.strategy.candidate import ModelObservation, StrategyCandidate


class MeasurementKind(str, Enum):
    """Semantic type of one measured fact."""

    BOOLEAN = "BOOLEAN"
    COUNT = "COUNT"
    DURATION = "DURATION"
    PRICE = "PRICE"
    QUANTITY = "QUANTITY"
    RATIO = "RATIO"
    IDENTIFIER = "IDENTIFIER"
    OTHER = "OTHER"


class MeasurementAvailability(str, Enum):
    """Whether one fact was observable from the supplied evidence."""

    OBSERVED = "OBSERVED"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True)
class QualityMeasurement:
    """One typed descriptive fact.

    `status` describes evidence availability only.  It is not a pass/fail result.
    """

    name: str
    kind: MeasurementKind
    status: MeasurementAvailability
    value: object = None

    observed_at: float = field(default_factory=time.time)
    source: str = ""
    reason: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        observed = float(self.observed_at)

        if not name:
            raise ValueError("measurement name is required")
        if not math.isfinite(observed) or observed <= 0.0:
            raise ValueError("observed_at must be positive")

        if (
            self.status is MeasurementAvailability.OBSERVED
            and self.value is None
        ):
            raise ValueError(
                "OBSERVED measurements must carry a value"
            )

        if isinstance(self.value, (int, float)) and not isinstance(
            self.value,
            bool,
        ):
            numeric = float(self.value)
            if not math.isfinite(numeric):
                raise ValueError(
                    "numeric measurement values must be finite"
                )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "source", str(self.source or ""))
        object.__setattr__(self, "reason", str(self.reason or ""))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class BookSideFacts:
    """Descriptive facts for one outcome token's current book."""

    token_id: str

    timestamp: float
    age_seconds: float

    source: str
    two_sided: bool
    crossed: bool

    depth_proven: bool
    synthetic_depth: bool

    bid_levels: int
    ask_levels: int

    best_bid_price: Optional[float]
    best_bid_size: Optional[float]
    best_ask_price: Optional[float]
    best_ask_size: Optional[float]

    spread: Optional[float]
    spread_ticks: Optional[float]

    known_bid_depth: Optional[float]
    known_ask_depth: Optional[float]

    def __post_init__(self) -> None:
        token_id = str(self.token_id or "").strip()
        timestamp = float(self.timestamp)
        age = max(0.0, float(self.age_seconds))

        if not token_id:
            raise ValueError("token_id is required")
        if not math.isfinite(timestamp) or timestamp <= 0.0:
            raise ValueError("timestamp must be positive")
        if not math.isfinite(age):
            raise ValueError("age_seconds must be finite")

        for name in (
            "best_bid_price",
            "best_ask_price",
            "spread",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, numeric)

        for name in (
            "best_bid_size",
            "best_ask_size",
            "known_bid_depth",
            "known_ask_depth",
            "spread_ticks",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative"
                )
            object.__setattr__(self, name, numeric)

        object.__setattr__(self, "token_id", token_id)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "age_seconds", age)
        object.__setattr__(self, "source", str(self.source or ""))
        object.__setattr__(self, "bid_levels", max(0, int(self.bid_levels)))
        object.__setattr__(self, "ask_levels", max(0, int(self.ask_levels)))


@dataclass(frozen=True, slots=True)
class PairBookFacts:
    """Descriptive YES/NO book facts for one binary market."""

    market_id: str
    measured_at: float

    yes: BookSideFacts
    no: BookSideFacts

    book_skew_seconds: float
    oldest_book_age_seconds: float
    newest_book_age_seconds: float

    both_two_sided: bool
    both_depth_proven: bool

    combined_best_bid: Optional[float]
    combined_best_ask: Optional[float]

    candidate_snapshot_match: bool
    candidate_snapshot_advanced: bool
    market_data_generation: Optional[int]

    def __post_init__(self) -> None:
        market_id = str(self.market_id or "").strip()
        measured = float(self.measured_at)
        skew = max(0.0, float(self.book_skew_seconds))
        oldest_age = max(
            0.0,
            float(self.oldest_book_age_seconds),
        )
        newest_age = max(
            0.0,
            float(self.newest_book_age_seconds),
        )

        if not market_id:
            raise ValueError("market_id is required")
        if not math.isfinite(measured) or measured <= 0.0:
            raise ValueError("measured_at must be positive")
        if not all(
            math.isfinite(value)
            for value in (skew, oldest_age, newest_age)
        ):
            raise ValueError("book timing facts must be finite")

        generation = self.market_data_generation
        if generation is not None:
            generation = int(generation)
            if generation < 0:
                raise ValueError(
                    "market_data_generation must be non-negative"
                )

        object.__setattr__(self, "market_id", market_id)
        object.__setattr__(self, "measured_at", measured)
        object.__setattr__(self, "book_skew_seconds", skew)
        object.__setattr__(
            self,
            "oldest_book_age_seconds",
            oldest_age,
        )
        object.__setattr__(
            self,
            "newest_book_age_seconds",
            newest_age,
        )
        object.__setattr__(
            self,
            "market_data_generation",
            generation,
        )


@dataclass(frozen=True, slots=True)
class QuoteBookFacts:
    """Candidate quote position relative to the measured current book."""

    token_id: str
    order_side: OrderSide
    quote_price: float

    best_same_side_price: Optional[float]
    best_opposite_side_price: Optional[float]

    distance_to_same_side: Optional[float]
    distance_to_opposite_side: Optional[float]

    crosses_current_book: Optional[bool]

    def __post_init__(self) -> None:
        token_id = str(self.token_id or "").strip()
        quote_price = float(self.quote_price)

        if not token_id:
            raise ValueError("token_id is required")
        if not math.isfinite(quote_price) or not 0.0 < quote_price <= 1.0:
            raise ValueError("quote_price must be in (0, 1]")

        for name in (
            "best_same_side_price",
            "best_opposite_side_price",
            "distance_to_same_side",
            "distance_to_opposite_side",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, numeric)

        object.__setattr__(self, "token_id", token_id)
        object.__setattr__(self, "quote_price", quote_price)


@dataclass(frozen=True, slots=True)
class ModelEvidenceCoverage:
    """Descriptive model-evidence completeness for one candidate."""

    expected_keys: Tuple[str, ...]
    present_keys: Tuple[str, ...]
    missing_keys: Tuple[str, ...]

    observation_count: int

    oldest_observation_age_seconds: Optional[float]
    newest_observation_age_seconds: Optional[float]

    minimum_generation: Optional[int]
    maximum_generation: Optional[int]

    @property
    def expected_count(self) -> int:
        return len(self.expected_keys)

    @property
    def present_expected_count(self) -> int:
        expected = set(self.expected_keys)
        return sum(1 for key in self.present_keys if key in expected)

    @property
    def coverage_fraction(self) -> Optional[float]:
        if not self.expected_keys:
            return None
        return self.present_expected_count / len(self.expected_keys)


@dataclass(frozen=True, slots=True)
class FeatureCoverage:
    """Descriptive named-feature completeness for one candidate."""

    expected_names: Tuple[str, ...]
    present_names: Tuple[str, ...]
    missing_names: Tuple[str, ...]

    @property
    def coverage_fraction(self) -> Optional[float]:
        if not self.expected_names:
            return None
        expected = set(self.expected_names)
        present = sum(
            1 for name in self.present_names if name in expected
        )
        return present / len(self.expected_names)


@dataclass(frozen=True, slots=True)
class CandidateTimingFacts:
    """Candidate and explicit validity-window timing facts."""

    created_at: float
    measured_at: float
    candidate_age_seconds: float

    valid_until: Optional[float]
    validity_remaining_seconds: Optional[float]
    explicit_validity_expired: bool


@dataclass(frozen=True, slots=True)
class QualityProvenance:
    """Identity of the evidence used for one measurement snapshot."""

    candidate_id: str
    candidate_fingerprint: str
    candidate_snapshot_id: str
    candidate_market_data_generation: int

    measured_market_id: str
    measured_yes_timestamp: float
    measured_no_timestamp: float
    measured_market_data_generation: Optional[int]

    measured_at: float


@dataclass(frozen=True, slots=True)
class CandidateQualitySnapshot:
    """Immutable measurement-only view of candidate evidence."""

    candidate_id: str
    candidate_fingerprint: str

    timing: CandidateTimingFacts
    books: PairBookFacts
    quote: QuoteBookFacts
    model_evidence: ModelEvidenceCoverage
    feature_evidence: FeatureCoverage
    provenance: QualityProvenance

    measurements: Tuple[QualityMeasurement, ...]

    def measurement(
        self,
        name: str,
    ) -> Optional[QualityMeasurement]:
        name = str(name or "").strip()

        return next(
            (
                row
                for row in self.measurements
                if row.name == name
            ),
            None,
        )

    def diagnostic_payload(self) -> dict[str, object]:
        """Serializable descriptive view with no policy verdict."""

        return {
            "candidate_id": self.candidate_id,
            "candidate_fingerprint": self.candidate_fingerprint,
            "timing": {
                "candidate_age_seconds": (
                    self.timing.candidate_age_seconds
                ),
                "valid_until": self.timing.valid_until,
                "validity_remaining_seconds": (
                    self.timing.validity_remaining_seconds
                ),
                "explicit_validity_expired": (
                    self.timing.explicit_validity_expired
                ),
            },
            "books": {
                "book_skew_seconds": self.books.book_skew_seconds,
                "oldest_book_age_seconds": (
                    self.books.oldest_book_age_seconds
                ),
                "newest_book_age_seconds": (
                    self.books.newest_book_age_seconds
                ),
                "both_two_sided": self.books.both_two_sided,
                "both_depth_proven": (
                    self.books.both_depth_proven
                ),
                "combined_best_bid": (
                    self.books.combined_best_bid
                ),
                "combined_best_ask": (
                    self.books.combined_best_ask
                ),
                "candidate_snapshot_match": (
                    self.books.candidate_snapshot_match
                ),
                "candidate_snapshot_advanced": (
                    self.books.candidate_snapshot_advanced
                ),
                "yes": {
                    "age_seconds": self.books.yes.age_seconds,
                    "source": self.books.yes.source,
                    "two_sided": self.books.yes.two_sided,
                    "crossed": self.books.yes.crossed,
                    "depth_proven": (
                        self.books.yes.depth_proven
                    ),
                    "synthetic_depth": (
                        self.books.yes.synthetic_depth
                    ),
                    "best_bid_price": (
                        self.books.yes.best_bid_price
                    ),
                    "best_ask_price": (
                        self.books.yes.best_ask_price
                    ),
                    "spread": self.books.yes.spread,
                    "known_bid_depth": (
                        self.books.yes.known_bid_depth
                    ),
                    "known_ask_depth": (
                        self.books.yes.known_ask_depth
                    ),
                },
                "no": {
                    "age_seconds": self.books.no.age_seconds,
                    "source": self.books.no.source,
                    "two_sided": self.books.no.two_sided,
                    "crossed": self.books.no.crossed,
                    "depth_proven": (
                        self.books.no.depth_proven
                    ),
                    "synthetic_depth": (
                        self.books.no.synthetic_depth
                    ),
                    "best_bid_price": (
                        self.books.no.best_bid_price
                    ),
                    "best_ask_price": (
                        self.books.no.best_ask_price
                    ),
                    "spread": self.books.no.spread,
                    "known_bid_depth": (
                        self.books.no.known_bid_depth
                    ),
                    "known_ask_depth": (
                        self.books.no.known_ask_depth
                    ),
                },
            },
            "quote": {
                "token_id": self.quote.token_id,
                "order_side": self.quote.order_side.value,
                "quote_price": self.quote.quote_price,
                "best_same_side_price": (
                    self.quote.best_same_side_price
                ),
                "best_opposite_side_price": (
                    self.quote.best_opposite_side_price
                ),
                "distance_to_same_side": (
                    self.quote.distance_to_same_side
                ),
                "distance_to_opposite_side": (
                    self.quote.distance_to_opposite_side
                ),
                "crosses_current_book": (
                    self.quote.crosses_current_book
                ),
            },
            "model_evidence": {
                "expected_keys": self.model_evidence.expected_keys,
                "present_keys": self.model_evidence.present_keys,
                "missing_keys": self.model_evidence.missing_keys,
                "observation_count": (
                    self.model_evidence.observation_count
                ),
                "oldest_observation_age_seconds": (
                    self.model_evidence.oldest_observation_age_seconds
                ),
                "newest_observation_age_seconds": (
                    self.model_evidence.newest_observation_age_seconds
                ),
                "minimum_generation": (
                    self.model_evidence.minimum_generation
                ),
                "maximum_generation": (
                    self.model_evidence.maximum_generation
                ),
                "coverage_fraction": (
                    self.model_evidence.coverage_fraction
                ),
            },
            "feature_evidence": {
                "expected_names": (
                    self.feature_evidence.expected_names
                ),
                "present_names": (
                    self.feature_evidence.present_names
                ),
                "missing_names": (
                    self.feature_evidence.missing_names
                ),
                "coverage_fraction": (
                    self.feature_evidence.coverage_fraction
                ),
            },
            "provenance": {
                "candidate_snapshot_id": (
                    self.provenance.candidate_snapshot_id
                ),
                "candidate_market_data_generation": (
                    self.provenance.candidate_market_data_generation
                ),
                "measured_market_data_generation": (
                    self.provenance.measured_market_data_generation
                ),
                "measured_yes_timestamp": (
                    self.provenance.measured_yes_timestamp
                ),
                "measured_no_timestamp": (
                    self.provenance.measured_no_timestamp
                ),
                "measured_at": self.provenance.measured_at,
            },
        }


def _known_depth(book: OrderBookSnapshot) -> tuple[Optional[float], Optional[float]]:
    """Return aggregate depth only when the ladder is genuinely proven."""

    if not book.depth_proven or book.synthetic_depth:
        return None, None

    bid_depth = sum(level.size for level in book.bids)
    ask_depth = sum(level.size for level in book.asks)

    return bid_depth, ask_depth


def _book_crossed(book: OrderBookSnapshot) -> bool:
    if book.best_bid is None or book.best_ask is None:
        return False

    return book.best_bid.price >= book.best_ask.price - 1e-12


def _book_facts(
    book: OrderBookSnapshot,
    *,
    now: float,
    tick_size: Optional[float],
) -> BookSideFacts:
    best_bid = book.best_bid
    best_ask = book.best_ask

    raw_spread = None
    spread_ticks = None

    if best_bid is not None and best_ask is not None:
        raw_spread = best_ask.price - best_bid.price

        if tick_size is not None and tick_size > 0.0:
            spread_ticks = abs(raw_spread) / tick_size

    bid_depth, ask_depth = _known_depth(book)

    return BookSideFacts(
        token_id=book.token_id,
        timestamp=book.timestamp,
        age_seconds=book.age_seconds(now),
        source=book.source.value,
        two_sided=book.is_two_sided,
        crossed=_book_crossed(book),
        depth_proven=book.depth_proven,
        synthetic_depth=book.synthetic_depth,
        bid_levels=len(book.bids),
        ask_levels=len(book.asks),
        best_bid_price=(
            best_bid.price if best_bid is not None else None
        ),
        best_bid_size=(
            best_bid.size if best_bid is not None else None
        ),
        best_ask_price=(
            best_ask.price if best_ask is not None else None
        ),
        best_ask_size=(
            best_ask.size if best_ask is not None else None
        ),
        spread=raw_spread,
        spread_ticks=spread_ticks,
        known_bid_depth=bid_depth,
        known_ask_depth=ask_depth,
    )


def _model_coverage(
    candidate: StrategyCandidate,
    *,
    expected_keys: Iterable[str],
    now: float,
) -> ModelEvidenceCoverage:
    expected = tuple(
        sorted(
            {
                str(key or "").strip()
                for key in expected_keys
                if str(key or "").strip()
            }
        )
    )

    observations = tuple(candidate.model_observations)
    present = tuple(
        sorted(observation.key for observation in observations)
    )
    present_set = set(present)
    missing = tuple(
        key for key in expected if key not in present_set
    )

    ages = tuple(
        max(0.0, now - observation.observed_at)
        for observation in observations
    )
    generations = tuple(
        observation.generation
        for observation in observations
    )

    return ModelEvidenceCoverage(
        expected_keys=expected,
        present_keys=present,
        missing_keys=missing,
        observation_count=len(observations),
        oldest_observation_age_seconds=(
            max(ages) if ages else None
        ),
        newest_observation_age_seconds=(
            min(ages) if ages else None
        ),
        minimum_generation=(
            min(generations) if generations else None
        ),
        maximum_generation=(
            max(generations) if generations else None
        ),
    )


def _feature_coverage(
    candidate: StrategyCandidate,
    *,
    expected_names: Iterable[str],
) -> FeatureCoverage:
    expected = tuple(
        sorted(
            {
                str(name or "").strip()
                for name in expected_names
                if str(name or "").strip()
            }
        )
    )
    present = tuple(sorted(str(name) for name in candidate.features))
    present_set = set(present)

    return FeatureCoverage(
        expected_names=expected,
        present_names=present,
        missing_names=tuple(
            name for name in expected if name not in present_set
        ),
    )


def _timing_facts(
    candidate: StrategyCandidate,
    *,
    now: float,
) -> CandidateTimingFacts:
    valid_until = candidate.valid_until

    remaining = None
    expired = False

    if valid_until is not None:
        remaining = valid_until - now
        expired = remaining < -1e-9

    return CandidateTimingFacts(
        created_at=candidate.created_at,
        measured_at=now,
        candidate_age_seconds=max(
            0.0,
            now - candidate.created_at,
        ),
        valid_until=valid_until,
        validity_remaining_seconds=remaining,
        explicit_validity_expired=expired,
    )


def _quote_facts(
    candidate: StrategyCandidate,
    *,
    books: MarketBooks,
) -> QuoteBookFacts:
    token_id = candidate.subject.token_id

    if token_id == books.market.yes_token:
        book = books.yes
    elif token_id == books.market.no_token:
        book = books.no
    else:
        raise ValueError(
            "candidate token does not belong to measured market"
        )

    quote_price = candidate.quote.limit_price

    if candidate.quote.order_side is OrderSide.BUY:
        same = book.best_bid
        opposite = book.best_ask

        same_price = same.price if same is not None else None
        opposite_price = (
            opposite.price if opposite is not None else None
        )

        distance_same = (
            quote_price - same_price
            if same_price is not None
            else None
        )
        distance_opposite = (
            opposite_price - quote_price
            if opposite_price is not None
            else None
        )
        crosses = (
            quote_price >= opposite_price - 1e-12
            if opposite_price is not None
            else None
        )
    else:
        same = book.best_ask
        opposite = book.best_bid

        same_price = same.price if same is not None else None
        opposite_price = (
            opposite.price if opposite is not None else None
        )

        distance_same = (
            same_price - quote_price
            if same_price is not None
            else None
        )
        distance_opposite = (
            quote_price - opposite_price
            if opposite_price is not None
            else None
        )
        crosses = (
            quote_price <= opposite_price + 1e-12
            if opposite_price is not None
            else None
        )

    return QuoteBookFacts(
        token_id=token_id,
        order_side=candidate.quote.order_side,
        quote_price=quote_price,
        best_same_side_price=same_price,
        best_opposite_side_price=opposite_price,
        distance_to_same_side=distance_same,
        distance_to_opposite_side=distance_opposite,
        crosses_current_book=crosses,
    )


class CandidateQualityMeasurer:
    """Create measurement-only evidence snapshots for StrategyCandidate objects."""

    def measure(
        self,
        candidate: StrategyCandidate,
        books: MarketBooks,
        *,
        now: Optional[float] = None,
        market_data_generation: Optional[int] = None,
        expected_model_keys: Iterable[str] = (),
        expected_feature_names: Iterable[str] = (),
    ) -> CandidateQualitySnapshot:
        measured_at = float(now or time.time())

        if not math.isfinite(measured_at) or measured_at <= 0.0:
            raise ValueError("now must be a positive Unix timestamp")

        if books.market.pair_key != candidate.subject.market_id:
            raise ValueError(
                "measured market does not match candidate subject"
            )

        if candidate.subject.token_id not in {
            books.market.yes_token,
            books.market.no_token,
        }:
            raise ValueError(
                "candidate token does not belong to measured market"
            )

        if candidate.subject.opposite_token_id not in {
            books.market.yes_token,
            books.market.no_token,
        }:
            raise ValueError(
                "candidate opposite token does not belong to measured market"
            )

        generation = market_data_generation
        if generation is not None:
            generation = int(generation)
            if generation < 0:
                raise ValueError(
                    "market_data_generation must be non-negative"
                )

        tick_size = books.market.tick_size

        yes = _book_facts(
            books.yes,
            now=measured_at,
            tick_size=tick_size,
        )
        no = _book_facts(
            books.no,
            now=measured_at,
            tick_size=tick_size,
        )

        snapshot_tolerance = 1e-9
        snapshot_match = bool(
            abs(
                books.yes.timestamp
                - candidate.market_snapshot.yes_book_timestamp
            )
            <= snapshot_tolerance
            and abs(
                books.no.timestamp
                - candidate.market_snapshot.no_book_timestamp
            )
            <= snapshot_tolerance
            and (
                generation is None
                or generation
                == candidate.market_snapshot.market_data_generation
            )
        )

        snapshot_advanced = bool(
            books.yes.timestamp
            > candidate.market_snapshot.yes_book_timestamp
            + snapshot_tolerance
            or books.no.timestamp
            > candidate.market_snapshot.no_book_timestamp
            + snapshot_tolerance
            or (
                generation is not None
                and generation
                > candidate.market_snapshot.market_data_generation
            )
        )

        combined_bid = None
        if yes.best_bid_price is not None and no.best_bid_price is not None:
            combined_bid = yes.best_bid_price + no.best_bid_price

        combined_ask = None
        if yes.best_ask_price is not None and no.best_ask_price is not None:
            combined_ask = yes.best_ask_price + no.best_ask_price

        pair = PairBookFacts(
            market_id=books.market.pair_key,
            measured_at=measured_at,
            yes=yes,
            no=no,
            book_skew_seconds=abs(
                books.yes.timestamp - books.no.timestamp
            ),
            oldest_book_age_seconds=max(
                yes.age_seconds,
                no.age_seconds,
            ),
            newest_book_age_seconds=min(
                yes.age_seconds,
                no.age_seconds,
            ),
            both_two_sided=books.both_two_sided,
            both_depth_proven=bool(
                yes.depth_proven
                and no.depth_proven
                and not yes.synthetic_depth
                and not no.synthetic_depth
            ),
            combined_best_bid=combined_bid,
            combined_best_ask=combined_ask,
            candidate_snapshot_match=snapshot_match,
            candidate_snapshot_advanced=snapshot_advanced,
            market_data_generation=generation,
        )

        quote = _quote_facts(
            candidate,
            books=books,
        )
        model_evidence = _model_coverage(
            candidate,
            expected_keys=expected_model_keys,
            now=measured_at,
        )
        feature_evidence = _feature_coverage(
            candidate,
            expected_names=expected_feature_names,
        )
        timing = _timing_facts(
            candidate,
            now=measured_at,
        )

        provenance = QualityProvenance(
            candidate_id=candidate.candidate_id,
            candidate_fingerprint=candidate.fingerprint,
            candidate_snapshot_id=(
                candidate.market_snapshot.snapshot_id
            ),
            candidate_market_data_generation=(
                candidate.market_snapshot.market_data_generation
            ),
            measured_market_id=books.market.pair_key,
            measured_yes_timestamp=books.yes.timestamp,
            measured_no_timestamp=books.no.timestamp,
            measured_market_data_generation=generation,
            measured_at=measured_at,
        )

        measurements = self._flatten_measurements(
            candidate=candidate,
            timing=timing,
            pair=pair,
            quote=quote,
            model_evidence=model_evidence,
            feature_evidence=feature_evidence,
            measured_at=measured_at,
        )

        return CandidateQualitySnapshot(
            candidate_id=candidate.candidate_id,
            candidate_fingerprint=candidate.fingerprint,
            timing=timing,
            books=pair,
            quote=quote,
            model_evidence=model_evidence,
            feature_evidence=feature_evidence,
            provenance=provenance,
            measurements=measurements,
        )

    @staticmethod
    def _observed(
        name: str,
        kind: MeasurementKind,
        value: object,
        *,
        observed_at: float,
        source: str,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> QualityMeasurement:
        return QualityMeasurement(
            name=name,
            kind=kind,
            status=MeasurementAvailability.OBSERVED,
            value=value,
            observed_at=observed_at,
            source=source,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _unavailable(
        name: str,
        kind: MeasurementKind,
        *,
        observed_at: float,
        source: str,
        reason: str,
    ) -> QualityMeasurement:
        return QualityMeasurement(
            name=name,
            kind=kind,
            status=MeasurementAvailability.UNAVAILABLE,
            value=None,
            observed_at=observed_at,
            source=source,
            reason=reason,
        )

    def _optional_measurement(
        self,
        *,
        name: str,
        kind: MeasurementKind,
        value: object,
        observed_at: float,
        source: str,
        unavailable_reason: str,
    ) -> QualityMeasurement:
        if value is None:
            return self._unavailable(
                name,
                kind,
                observed_at=observed_at,
                source=source,
                reason=unavailable_reason,
            )

        return self._observed(
            name,
            kind,
            value,
            observed_at=observed_at,
            source=source,
        )

    def _flatten_measurements(
        self,
        *,
        candidate: StrategyCandidate,
        timing: CandidateTimingFacts,
        pair: PairBookFacts,
        quote: QuoteBookFacts,
        model_evidence: ModelEvidenceCoverage,
        feature_evidence: FeatureCoverage,
        measured_at: float,
    ) -> Tuple[QualityMeasurement, ...]:
        rows: list[QualityMeasurement] = []

        rows.extend(
            (
                self._observed(
                    "candidate.age_seconds",
                    MeasurementKind.DURATION,
                    timing.candidate_age_seconds,
                    observed_at=measured_at,
                    source="candidate",
                ),
                self._observed(
                    "candidate.explicit_validity_expired",
                    MeasurementKind.BOOLEAN,
                    timing.explicit_validity_expired,
                    observed_at=measured_at,
                    source="candidate",
                ),
                self._observed(
                    "books.skew_seconds",
                    MeasurementKind.DURATION,
                    pair.book_skew_seconds,
                    observed_at=measured_at,
                    source="market_data",
                ),
                self._observed(
                    "books.oldest_age_seconds",
                    MeasurementKind.DURATION,
                    pair.oldest_book_age_seconds,
                    observed_at=measured_at,
                    source="market_data",
                ),
                self._observed(
                    "books.both_two_sided",
                    MeasurementKind.BOOLEAN,
                    pair.both_two_sided,
                    observed_at=measured_at,
                    source="market_data",
                ),
                self._observed(
                    "books.both_depth_proven",
                    MeasurementKind.BOOLEAN,
                    pair.both_depth_proven,
                    observed_at=measured_at,
                    source="market_data",
                ),
                self._observed(
                    "books.candidate_snapshot_match",
                    MeasurementKind.BOOLEAN,
                    pair.candidate_snapshot_match,
                    observed_at=measured_at,
                    source="market_data",
                ),
                self._observed(
                    "books.candidate_snapshot_advanced",
                    MeasurementKind.BOOLEAN,
                    pair.candidate_snapshot_advanced,
                    observed_at=measured_at,
                    source="market_data",
                ),
                self._observed(
                    "books.yes.crossed",
                    MeasurementKind.BOOLEAN,
                    pair.yes.crossed,
                    observed_at=measured_at,
                    source="market_data",
                ),
                self._observed(
                    "books.no.crossed",
                    MeasurementKind.BOOLEAN,
                    pair.no.crossed,
                    observed_at=measured_at,
                    source="market_data",
                ),
                self._observed(
                    "models.observation_count",
                    MeasurementKind.COUNT,
                    model_evidence.observation_count,
                    observed_at=measured_at,
                    source="candidate_models",
                ),
                self._observed(
                    "models.missing_expected_count",
                    MeasurementKind.COUNT,
                    len(model_evidence.missing_keys),
                    observed_at=measured_at,
                    source="candidate_models",
                ),
                self._observed(
                    "features.missing_expected_count",
                    MeasurementKind.COUNT,
                    len(feature_evidence.missing_names),
                    observed_at=measured_at,
                    source="candidate_features",
                ),
            )
        )

        for prefix, side in (
            ("books.yes", pair.yes),
            ("books.no", pair.no),
        ):
            rows.extend(
                (
                    self._optional_measurement(
                        name=f"{prefix}.spread",
                        kind=MeasurementKind.PRICE,
                        value=side.spread,
                        observed_at=measured_at,
                        source="market_data",
                        unavailable_reason="book is not two-sided",
                    ),
                    self._optional_measurement(
                        name=f"{prefix}.best_bid_size",
                        kind=MeasurementKind.QUANTITY,
                        value=side.best_bid_size,
                        observed_at=measured_at,
                        source="market_data",
                        unavailable_reason="best bid unavailable",
                    ),
                    self._optional_measurement(
                        name=f"{prefix}.best_ask_size",
                        kind=MeasurementKind.QUANTITY,
                        value=side.best_ask_size,
                        observed_at=measured_at,
                        source="market_data",
                        unavailable_reason="best ask unavailable",
                    ),
                    self._optional_measurement(
                        name=f"{prefix}.known_bid_depth",
                        kind=MeasurementKind.QUANTITY,
                        value=side.known_bid_depth,
                        observed_at=measured_at,
                        source="market_data",
                        unavailable_reason=(
                            "full bid depth is not proven"
                        ),
                    ),
                    self._optional_measurement(
                        name=f"{prefix}.known_ask_depth",
                        kind=MeasurementKind.QUANTITY,
                        value=side.known_ask_depth,
                        observed_at=measured_at,
                        source="market_data",
                        unavailable_reason=(
                            "full ask depth is not proven"
                        ),
                    ),
                )
            )

        rows.extend(
            (
                self._optional_measurement(
                    name="books.combined_best_bid",
                    kind=MeasurementKind.PRICE,
                    value=pair.combined_best_bid,
                    observed_at=measured_at,
                    source="market_data",
                    unavailable_reason=(
                        "one or both best bids are unavailable"
                    ),
                ),
                self._optional_measurement(
                    name="books.combined_best_ask",
                    kind=MeasurementKind.PRICE,
                    value=pair.combined_best_ask,
                    observed_at=measured_at,
                    source="market_data",
                    unavailable_reason=(
                        "one or both best asks are unavailable"
                    ),
                ),
                self._optional_measurement(
                    name="quote.distance_to_same_side",
                    kind=MeasurementKind.PRICE,
                    value=quote.distance_to_same_side,
                    observed_at=measured_at,
                    source="candidate_vs_market",
                    unavailable_reason=(
                        "same-side top-of-book unavailable"
                    ),
                ),
                self._optional_measurement(
                    name="quote.distance_to_opposite_side",
                    kind=MeasurementKind.PRICE,
                    value=quote.distance_to_opposite_side,
                    observed_at=measured_at,
                    source="candidate_vs_market",
                    unavailable_reason=(
                        "opposite-side top-of-book unavailable"
                    ),
                ),
                self._optional_measurement(
                    name="quote.crosses_current_book",
                    kind=MeasurementKind.BOOLEAN,
                    value=quote.crosses_current_book,
                    observed_at=measured_at,
                    source="candidate_vs_market",
                    unavailable_reason=(
                        "opposite-side top-of-book unavailable"
                    ),
                ),
                self._optional_measurement(
                    name="models.oldest_age_seconds",
                    kind=MeasurementKind.DURATION,
                    value=(
                        model_evidence.oldest_observation_age_seconds
                    ),
                    observed_at=measured_at,
                    source="candidate_models",
                    unavailable_reason="no model observations",
                ),
                self._optional_measurement(
                    name="models.newest_age_seconds",
                    kind=MeasurementKind.DURATION,
                    value=(
                        model_evidence.newest_observation_age_seconds
                    ),
                    observed_at=measured_at,
                    source="candidate_models",
                    unavailable_reason="no model observations",
                ),
                self._optional_measurement(
                    name="models.coverage_fraction",
                    kind=MeasurementKind.RATIO,
                    value=model_evidence.coverage_fraction,
                    observed_at=measured_at,
                    source="candidate_models",
                    unavailable_reason=(
                        "no expected model keys supplied"
                    ),
                ),
                self._optional_measurement(
                    name="features.coverage_fraction",
                    kind=MeasurementKind.RATIO,
                    value=feature_evidence.coverage_fraction,
                    observed_at=measured_at,
                    source="candidate_features",
                    unavailable_reason=(
                        "no expected feature names supplied"
                    ),
                ),
            )
        )

        return tuple(rows)
