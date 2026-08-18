"""
Exact-order fill accounting for the public portfolio edition.

This module is deliberately separate from strategy and lifecycle policy.  It
turns venue/order/trade/wallet observations into auditable fill evidence.

Public invariants:
- requested quantity and actual filled quantity are different facts
- cumulative matched quantity is never silently rewritten to requested quantity
- duplicate fill rows are deduplicated by stable evidence identity
- top-level order `price` is treated as a limit/bound, not a realized average
- realized average price comes from explicit average fields or weighted fill rows
- a wallet balance proves an order fill only when a valid pre-order baseline exists
- baseline-backed wallet overfill is preserved and flagged, never hidden by clamp
- late fills remain attributable to their exact order/lifecycle
- incomplete price evidence is labelled incomplete instead of inventing improvement

The later reconciliation service decides which evidence source is authoritative
for a particular unresolved lifecycle.  This module records and summarizes facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
import threading
import time
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from src.execution.types import Fill, FillSummary, LifecycleIdentity, OrderSide


class FillEvidenceSource(str, Enum):
    """Origin of a fill observation."""

    ORDER_STATUS = "ORDER_STATUS"
    EMBEDDED_FILL = "EMBEDDED_FILL"
    TRADE_ENDPOINT = "TRADE_ENDPOINT"
    WALLET_DELTA = "WALLET_DELTA"
    MANUAL = "MANUAL"


class PriceEvidenceKind(str, Enum):
    """How a realized price was established."""

    EXPLICIT_AVERAGE = "EXPLICIT_AVERAGE"
    WEIGHTED_ROWS = "WEIGHTED_ROWS"
    ORDER_LIMIT_BOUND = "ORDER_LIMIT_BOUND"
    MISSING = "MISSING"


@dataclass(frozen=True, slots=True)
class WalletBaseline:
    """Pre-order token balance required for wallet-delta attribution."""

    token_id: str
    balance: float
    observed_at: float
    valid: bool = True

    def __post_init__(self) -> None:
        token_id = str(self.token_id or "").strip()
        balance = float(self.balance)
        observed = float(self.observed_at)

        if not token_id:
            raise ValueError("token_id is required")
        if not math.isfinite(balance) or balance < 0.0:
            raise ValueError("baseline balance must be non-negative")
        if not math.isfinite(observed) or observed <= 0.0:
            raise ValueError("observed_at must be positive")

        object.__setattr__(self, "token_id", token_id)
        object.__setattr__(self, "balance", balance)
        object.__setattr__(self, "observed_at", observed)


@dataclass(frozen=True, slots=True)
class WalletDeltaObservation:
    """Wallet evidence attributable to one token after a valid baseline."""

    token_id: str
    baseline_valid: bool
    baseline_balance: Optional[float]
    current_balance: float
    delta: float
    requested_size: float
    overfill: bool
    observed_at: float

    @property
    def proves_positive_inventory(self) -> bool:
        return self.baseline_valid and self.delta > 0.0

    @property
    def proves_zero_delta(self) -> bool:
        return self.baseline_valid and self.delta <= 0.0


@dataclass(frozen=True, slots=True)
class PriceEvidence:
    """Realized-price evidence plus represented fill quantity."""

    price: Optional[float]
    kind: PriceEvidenceKind
    source: str
    represented_size: float = 0.0

    def __post_init__(self) -> None:
        represented = max(0.0, float(self.represented_size))
        price = self.price

        if price is not None:
            price = float(price)
            if not math.isfinite(price) or not 0.0 < price <= 1.0:
                raise ValueError("price must be in (0, 1]")

        object.__setattr__(self, "price", price)
        object.__setattr__(self, "source", str(self.source or ""))
        object.__setattr__(self, "represented_size", represented)


@dataclass(frozen=True, slots=True)
class FillObservation:
    """One cumulative or incremental fill fact before ledger application."""

    order_id: str
    token_id: str
    order_side: OrderSide
    source: FillEvidenceSource
    observed_size: float
    cumulative: bool
    observed_at: float
    price_evidence: Optional[PriceEvidence] = None
    evidence_id: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        order_id = str(self.order_id or "").strip()
        token_id = str(self.token_id or "").strip()
        size = float(self.observed_size)
        observed = float(self.observed_at)

        if not order_id:
            raise ValueError("order_id is required")
        if not token_id:
            raise ValueError("token_id is required")
        if not math.isfinite(size) or size < 0.0:
            raise ValueError("observed_size must be non-negative")
        if not math.isfinite(observed) or observed <= 0.0:
            raise ValueError("observed_at must be positive")

        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "token_id", token_id)
        object.__setattr__(self, "observed_size", size)
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "evidence_id", str(self.evidence_id or ""))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class FillAnomaly:
    """Accounting anomaly retained for observability instead of being hidden."""

    code: str
    order_id: str
    message: str
    observed_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", str(self.code or "UNKNOWN"))
        object.__setattr__(self, "order_id", str(self.order_id or ""))
        object.__setattr__(self, "message", str(self.message or ""))


@dataclass(frozen=True, slots=True)
class OrderFillRecord:
    """Immutable snapshot of one exact order's fill ledger."""

    order_id: str
    token_id: str
    lifecycle: LifecycleIdentity
    order_side: OrderSide
    requested_size: float
    submitted_limit_price: float
    created_at: float

    observed_filled_size: float = 0.0
    confirmed_filled_size: float = 0.0
    realized_average_price: Optional[float] = None
    realized_price_kind: PriceEvidenceKind = PriceEvidenceKind.MISSING
    realized_price_source: str = ""
    price_covered_size: float = 0.0

    fills: Tuple[Fill, ...] = field(default_factory=tuple)
    observation_ids: Tuple[str, ...] = field(default_factory=tuple)
    sources: Tuple[FillEvidenceSource, ...] = field(default_factory=tuple)
    anomalies: Tuple[FillAnomaly, ...] = field(default_factory=tuple)

    last_observed_at: float = 0.0
    resolved: bool = False
    resolved_reason: str = ""

    def __post_init__(self) -> None:
        order_id = str(self.order_id or "").strip()
        token_id = str(self.token_id or "").strip()
        requested = float(self.requested_size)
        limit_price = float(self.submitted_limit_price)
        created = float(self.created_at)
        observed = max(0.0, float(self.observed_filled_size))
        confirmed = max(0.0, float(self.confirmed_filled_size))
        covered = max(0.0, float(self.price_covered_size))

        if not order_id:
            raise ValueError("order_id is required")
        if not token_id:
            raise ValueError("token_id is required")
        if not math.isfinite(requested) or requested <= 0.0:
            raise ValueError("requested_size must be positive")
        if not math.isfinite(limit_price) or not 0.0 < limit_price <= 1.0:
            raise ValueError("submitted_limit_price must be in (0, 1]")
        if not math.isfinite(created) or created <= 0.0:
            raise ValueError("created_at must be positive")

        average = self.realized_average_price
        if average is not None:
            average = float(average)
            if not math.isfinite(average) or not 0.0 < average <= 1.0:
                raise ValueError("realized_average_price must be in (0, 1]")

        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "token_id", token_id)
        object.__setattr__(self, "requested_size", requested)
        object.__setattr__(self, "submitted_limit_price", limit_price)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "observed_filled_size", observed)
        object.__setattr__(self, "confirmed_filled_size", confirmed)
        object.__setattr__(self, "realized_average_price", average)
        object.__setattr__(self, "price_covered_size", covered)
        object.__setattr__(self, "fills", tuple(self.fills))
        object.__setattr__(self, "observation_ids", tuple(self.observation_ids))
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "anomalies", tuple(self.anomalies))
        object.__setattr__(self, "resolved_reason", str(self.resolved_reason or ""))

    @property
    def remaining_requested_size(self) -> float:
        """Unfilled part of the original request, never negative."""
        return max(0.0, self.requested_size - self.confirmed_filled_size)

    @property
    def overfilled_size(self) -> float:
        return max(0.0, self.confirmed_filled_size - self.requested_size)

    @property
    def overfilled(self) -> bool:
        return self.overfilled_size > 1e-9

    @property
    def has_fill(self) -> bool:
        return self.confirmed_filled_size > 0.0

    @property
    def request_fully_filled(self) -> bool:
        return self.confirmed_filled_size >= self.requested_size - 1e-9

    @property
    def price_coverage_complete(self) -> bool:
        if self.confirmed_filled_size <= 0.0:
            return False
        return self.price_covered_size >= self.confirmed_filled_size - 1e-9

    def to_fill_summary(self) -> FillSummary:
        return FillSummary(
            order_id=self.order_id,
            requested_size=self.requested_size,
            filled_size=self.confirmed_filled_size,
            average_price=self.realized_average_price,
            fills=self.fills,
        )


def _float_or_none(value: object) -> Optional[float]:
    if value in (None, ""):
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    # Some venue/UI payloads expose cents rather than decimal dollars.
    if result > 1.5 and result <= 100.0:
        result /= 100.0

    return result


def _positive_amount(value: object) -> Optional[float]:
    """Parse a positive quantity without applying price/cents normalization."""

    if value in (None, ""):
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result) or result <= 0.0:
        return None

    return result


def _amount_units(value: object) -> Optional[float]:
    """Normalize obvious integer micro-unit amount fields conservatively."""

    if value is None:
        return None

    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result) or result <= 0.0:
        return None

    if result > 10_000:
        result /= 1_000_000.0

    return result if result > 0.0 else None


MATCHED_SIZE_KEYS = (
    "size_matched",
    "sizeMatched",
    "matched_size",
    "matchedSize",
    "filled",
    "filled_size",
    "filledSize",
)

FILL_LIST_KEYS = (
    "fills",
    "trades",
    "transactions",
    "matches",
)

FILL_SIZE_KEYS = (
    "size",
    "matched_size",
    "size_matched",
    "amount",
    "shares",
    "filled",
    "filled_size",
)

EXPLICIT_AVERAGE_KEYS = (
    "avgPrice",
    "avg_price",
    "averagePrice",
    "average_price",
    "avgFillPrice",
    "avg_fill_price",
    "averageFillPrice",
    "average_fill_price",
    "filledAvgPrice",
    "filled_avg_price",
    "fillAvgPrice",
    "fill_avg_price",
    "matchedAvgPrice",
    "matched_avg_price",
    "matchAvgPrice",
    "match_avg_price",
    "executionPrice",
    "execution_price",
    "executedPrice",
    "executed_price",
    "matchedPrice",
    "matched_price",
    "fillPrice",
    "fill_price",
)

ROW_PRICE_KEYS = (
    "avgPrice",
    "avg_price",
    "averagePrice",
    "average_price",
    "fillPrice",
    "fill_price",
    "matchedPrice",
    "matched_price",
    "executionPrice",
    "execution_price",
    "executedPrice",
    "executed_price",
    "price",
)

LIMIT_PRICE_KEYS = (
    "price",
    "limitPrice",
    "limit_price",
    "orderPrice",
    "order_price",
)


def extract_cumulative_matched_size(payload: object) -> float:
    """Return venue-reported cumulative matched quantity without clamping."""

    if not isinstance(payload, Mapping):
        return 0.0

    for key in MATCHED_SIZE_KEYS:
        value = _positive_amount(payload.get(key))
        if value is not None:
            return value

    total = 0.0

    for list_key in FILL_LIST_KEYS:
        rows = payload.get(list_key)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue

        list_total = 0.0
        found = False

        for row in rows:
            if not isinstance(row, Mapping):
                continue

            for size_key in FILL_SIZE_KEYS:
                amount = _amount_units(row.get(size_key))
                if amount is not None:
                    list_total += amount
                    found = True
                    break

        if found:
            total = max(total, list_total)

    return max(0.0, total)


def extract_order_limit_price(payload: object) -> Optional[float]:
    """Return an order limit/bound without mislabelling it as realized average."""

    if not isinstance(payload, Mapping):
        return None

    for key in LIMIT_PRICE_KEYS:
        price = _float_or_none(payload.get(key))
        if price is not None and 0.0 < price <= 1.0:
            return price

    return None


def _price_from_amounts(
    payload: Mapping[str, object],
    *,
    side: OrderSide,
) -> Optional[float]:
    """Best-effort average from making/taking amount pairs.

    Different SDK/feed shapes can represent base/share and quote/collateral
    amounts with different field names.  Only compute a price when both positive
    amounts are available and the ratio is a valid prediction-market price.
    """

    making = None
    taking = None

    for key in (
        "making_amount",
        "makingAmount",
        "maker_amount",
        "makerAmount",
    ):
        making = _amount_units(payload.get(key))
        if making is not None:
            break

    for key in (
        "taking_amount",
        "takingAmount",
        "taker_amount",
        "takerAmount",
    ):
        taking = _amount_units(payload.get(key))
        if taking is not None:
            break

    if making is None or taking is None:
        return None

    candidates = []

    if side is OrderSide.BUY:
        candidates.extend(
            (
                making / taking if taking else 0.0,
                taking / making if making else 0.0,
            )
        )
    else:
        candidates.extend(
            (
                taking / making if making else 0.0,
                making / taking if taking else 0.0,
            )
        )

    valid = [
        value
        for value in candidates
        if math.isfinite(value) and 0.0 < value <= 1.0
    ]

    return valid[0] if len(valid) == 1 else None


def extract_realized_price_evidence(
    payload: object,
    *,
    side: OrderSide,
) -> PriceEvidence:
    """Extract only realized-price evidence.

    A lone top-level `price` is intentionally excluded from true-average
    detection because CLOB order payloads commonly use it for the submitted
    limit.  Per-fill rows may use `price` because those rows themselves represent
    executions.
    """

    if not isinstance(payload, Mapping):
        return PriceEvidence(
            price=None,
            kind=PriceEvidenceKind.MISSING,
            source="missing payload",
        )

    for key in EXPLICIT_AVERAGE_KEYS:
        price = _float_or_none(payload.get(key))
        if price is not None and 0.0 < price <= 1.0:
            represented = extract_cumulative_matched_size(payload)
            return PriceEvidence(
                price=price,
                kind=PriceEvidenceKind.EXPLICIT_AVERAGE,
                source=key,
                represented_size=represented,
            )

    amount_price = _price_from_amounts(payload, side=side)
    if amount_price is not None:
        represented = extract_cumulative_matched_size(payload)
        return PriceEvidence(
            price=amount_price,
            kind=PriceEvidenceKind.EXPLICIT_AVERAGE,
            source="making/taking amounts",
            represented_size=represented,
        )

    for list_key in FILL_LIST_KEYS:
        rows = payload.get(list_key)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue

        numerator = 0.0
        denominator = 0.0

        for row in rows:
            if not isinstance(row, Mapping):
                continue

            price = None
            size = None

            for price_key in ROW_PRICE_KEYS:
                candidate = _float_or_none(row.get(price_key))
                if candidate is not None and 0.0 < candidate <= 1.0:
                    price = candidate
                    break

            if price is None:
                price = _price_from_amounts(row, side=side)

            for size_key in FILL_SIZE_KEYS:
                candidate_size = _amount_units(row.get(size_key))
                if candidate_size is not None:
                    size = candidate_size
                    break

            if price is not None and size is not None and size > 0.0:
                numerator += price * size
                denominator += size

        if denominator > 0.0:
            return PriceEvidence(
                price=numerator / denominator,
                kind=PriceEvidenceKind.WEIGHTED_ROWS,
                source=f"{list_key}.weighted",
                represented_size=denominator,
            )

    return PriceEvidence(
        price=None,
        kind=PriceEvidenceKind.MISSING,
        source="missing realized average",
    )


def limit_price_bound_evidence(
    payload: object,
    *,
    fallback_limit: Optional[float] = None,
) -> PriceEvidence:
    """Return a conservative order-local price bound, not a realized average."""

    limit_price = extract_order_limit_price(payload)
    if limit_price is None:
        limit_price = fallback_limit

    if limit_price is None:
        return PriceEvidence(
            price=None,
            kind=PriceEvidenceKind.MISSING,
            source="missing order-local limit",
        )

    limit_price = float(limit_price)

    if not math.isfinite(limit_price) or not 0.0 < limit_price <= 1.0:
        return PriceEvidence(
            price=None,
            kind=PriceEvidenceKind.MISSING,
            source="invalid order-local limit",
        )

    return PriceEvidence(
        price=limit_price,
        kind=PriceEvidenceKind.ORDER_LIMIT_BOUND,
        source="order-local limit",
        represented_size=0.0,
    )


def wallet_delta_from_baseline(
    *,
    token_id: str,
    current_balance: float,
    requested_size: float,
    baseline: Optional[WalletBaseline],
    observed_at: Optional[float] = None,
    epsilon: float = 1e-9,
) -> WalletDeltaObservation:
    """Compute token inventory delta only from an explicit valid baseline.

    Without a baseline, total wallet quantity may include older positions and is
    therefore not attributed to the current order.
    """

    token_id = str(token_id or "").strip()
    if not token_id:
        raise ValueError("token_id is required")

    current = float(current_balance)
    requested = float(requested_size)
    now = float(observed_at or time.time())

    if not math.isfinite(current) or current < 0.0:
        raise ValueError("current_balance must be non-negative")
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("requested_size must be positive")

    baseline_valid = bool(
        baseline is not None
        and baseline.valid
        and baseline.token_id == token_id
    )

    if not baseline_valid:
        return WalletDeltaObservation(
            token_id=token_id,
            baseline_valid=False,
            baseline_balance=None,
            current_balance=current,
            delta=0.0,
            requested_size=requested,
            overfill=False,
            observed_at=now,
        )

    assert baseline is not None
    delta = max(0.0, current - baseline.balance)
    if delta <= max(0.0, float(epsilon)):
        delta = 0.0

    # Preserve real observed inventory above the requested quantity.  This can
    # reveal duplicate, late or otherwise untracked fills.
    return WalletDeltaObservation(
        token_id=token_id,
        baseline_valid=True,
        baseline_balance=baseline.balance,
        current_balance=current,
        delta=delta,
        requested_size=requested,
        overfill=delta > requested + max(0.0, float(epsilon)),
        observed_at=now,
    )


def _stable_json(value: Mapping[str, object]) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        return repr(sorted(value.items(), key=lambda item: str(item[0])))


def _row_evidence_id(
    *,
    order_id: str,
    row: Mapping[str, object],
    list_key: str,
    index: int,
) -> str:
    """Build a stable identity for a venue fill row.

    Prefer venue-native IDs/transaction hashes.  The content fingerprint fallback
    is deterministic across repeated endpoint reads.
    """

    for key in (
        "id",
        "fill_id",
        "fillId",
        "trade_id",
        "tradeId",
        "transaction_hash",
        "transactionHash",
        "tx_hash",
        "txHash",
        "match_id",
        "matchId",
    ):
        value = str(row.get(key) or "").strip()
        if value:
            return f"{list_key}:{value}"

    fingerprint_payload = {
        "order_id": order_id,
        "list_key": list_key,
        "price": next(
            (
                row.get(key)
                for key in ROW_PRICE_KEYS
                if row.get(key) not in (None, "")
            ),
            None,
        ),
        "size": next(
            (
                row.get(key)
                for key in FILL_SIZE_KEYS
                if row.get(key) not in (None, "")
            ),
            None,
        ),
        "timestamp": next(
            (
                row.get(key)
                for key in (
                    "timestamp",
                    "time",
                    "created_at",
                    "createdAt",
                    "match_time",
                    "matchTime",
                )
                if row.get(key) not in (None, "")
            ),
            None,
        ),
        "index_hint": index,
    }

    digest = hashlib.blake2b(
        _stable_json(fingerprint_payload).encode("utf-8"),
        digest_size=16,
    ).hexdigest()

    return f"{list_key}:fingerprint:{digest}"


def _row_timestamp(
    row: Mapping[str, object],
    *,
    fallback: float,
) -> float:
    for key in (
        "timestamp",
        "time",
        "created_at",
        "createdAt",
        "match_time",
        "matchTime",
    ):
        value = row.get(key)
        if value in (None, ""):
            continue

        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue

        # Millisecond Unix timestamp.
        if parsed > 10_000_000_000:
            parsed /= 1000.0

        if math.isfinite(parsed) and parsed > 0.0:
            return parsed

    return fallback


def extract_incremental_fills(
    payload: object,
    *,
    order_id: str,
    token_id: str,
    side: OrderSide,
    observed_at: Optional[float] = None,
) -> Tuple[Fill, ...]:
    """Extract per-fill rows when the venue response exposes them."""

    if not isinstance(payload, Mapping):
        return ()

    now = float(observed_at or time.time())
    fills: list[Fill] = []

    for list_key in FILL_LIST_KEYS:
        rows = payload.get(list_key)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue

        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue

            price = None
            size = None

            for price_key in ROW_PRICE_KEYS:
                candidate = _float_or_none(row.get(price_key))
                if candidate is not None and 0.0 < candidate <= 1.0:
                    price = candidate
                    break

            if price is None:
                price = _price_from_amounts(row, side=side)

            for size_key in FILL_SIZE_KEYS:
                candidate_size = _amount_units(row.get(size_key))
                if candidate_size is not None:
                    size = candidate_size
                    break

            if price is None or size is None or size <= 0.0:
                continue

            fills.append(
                Fill(
                    fill_id=_row_evidence_id(
                        order_id=order_id,
                        row=row,
                        list_key=list_key,
                        index=index,
                    ),
                    order_id=order_id,
                    token_id=token_id,
                    order_side=side,
                    size=size,
                    price=price,
                    timestamp=_row_timestamp(row, fallback=now),
                )
            )

    # The same execution can occasionally be nested under multiple aliases in one
    # payload.  Deduplicate before returning.
    unique: Dict[str, Fill] = {}
    for fill in fills:
        unique.setdefault(fill.fill_id, fill)

    return tuple(unique.values())


class FillAccounting:
    """Thread-safe exact-order fill ledger."""

    def __init__(self) -> None:
        self._gate = threading.RLock()
        self._records: Dict[str, OrderFillRecord] = {}

    def register_order(
        self,
        *,
        order_id: str,
        token_id: str,
        lifecycle: LifecycleIdentity,
        order_side: OrderSide,
        requested_size: float,
        submitted_limit_price: float,
        created_at: Optional[float] = None,
    ) -> OrderFillRecord:
        order_id = str(order_id or "").strip()
        token_id = str(token_id or "").strip()
        requested = float(requested_size)
        limit_price = float(submitted_limit_price)
        created = float(created_at or time.time())

        record = OrderFillRecord(
            order_id=order_id,
            token_id=token_id,
            lifecycle=lifecycle,
            order_side=order_side,
            requested_size=requested,
            submitted_limit_price=limit_price,
            created_at=created,
            last_observed_at=created,
        )

        with self._gate:
            existing = self._records.get(order_id)

            if existing is not None:
                if existing.lifecycle != lifecycle:
                    raise ValueError(
                        f"order {order_id} already belongs to another lifecycle"
                    )
                if existing.token_id != token_id:
                    raise ValueError(
                        f"order {order_id} token affinity mismatch"
                    )
                return existing

            self._records[order_id] = record
            return record

    def get(self, order_id: str) -> Optional[OrderFillRecord]:
        with self._gate:
            return self._records.get(str(order_id or ""))

    def records(self) -> Tuple[OrderFillRecord, ...]:
        with self._gate:
            return tuple(self._records.values())

    @staticmethod
    def _append_anomaly(
        record: OrderFillRecord,
        *,
        code: str,
        message: str,
        observed_at: Optional[float] = None,
    ) -> OrderFillRecord:
        anomaly = FillAnomaly(
            code=code,
            order_id=record.order_id,
            message=message,
            observed_at=float(observed_at or time.time()),
        )

        # Avoid endlessly duplicating the same anomaly on repeated cumulative polls.
        if any(
            existing.code == anomaly.code
            and existing.message == anomaly.message
            for existing in record.anomalies
        ):
            return record

        return replace(
            record,
            anomalies=record.anomalies + (anomaly,),
        )

    @staticmethod
    def _merge_sources(
        current: Tuple[FillEvidenceSource, ...],
        source: FillEvidenceSource,
    ) -> Tuple[FillEvidenceSource, ...]:
        if source in current:
            return current
        return current + (source,)

    @staticmethod
    def _weighted_average(fills: Iterable[Fill]) -> tuple[Optional[float], float]:
        numerator = 0.0
        denominator = 0.0

        for fill in fills:
            if fill.size <= 0.0:
                continue
            numerator += fill.price * fill.size
            denominator += fill.size

        if denominator <= 0.0:
            return None, 0.0

        return numerator / denominator, denominator

    @staticmethod
    def _choose_price_evidence(
        *,
        record: OrderFillRecord,
        new_evidence: Optional[PriceEvidence],
        deduped_fills: Tuple[Fill, ...],
    ) -> tuple[Optional[float], PriceEvidenceKind, str, float]:
        """Choose strongest realized-price evidence without inventing coverage."""

        fill_average, fill_covered = FillAccounting._weighted_average(
            deduped_fills
        )

        # Per-fill evidence is naturally auditable and exact for its represented
        # quantity.  Prefer it when it covers at least as much quantity as the
        # current realized-price record.
        if (
            fill_average is not None
            and fill_covered >= record.price_covered_size - 1e-9
        ):
            return (
                fill_average,
                PriceEvidenceKind.WEIGHTED_ROWS,
                "deduplicated fills weighted",
                fill_covered,
            )

        if (
            new_evidence is not None
            and new_evidence.price is not None
            and new_evidence.kind
            in {
                PriceEvidenceKind.EXPLICIT_AVERAGE,
                PriceEvidenceKind.WEIGHTED_ROWS,
            }
        ):
            if (
                new_evidence.represented_size
                >= record.price_covered_size - 1e-9
            ):
                return (
                    new_evidence.price,
                    new_evidence.kind,
                    new_evidence.source,
                    new_evidence.represented_size,
                )

        return (
            record.realized_average_price,
            record.realized_price_kind,
            record.realized_price_source,
            record.price_covered_size,
        )

    def apply_observation(
        self,
        observation: FillObservation,
        *,
        fills: Iterable[Fill] = (),
        confirm_quantity: bool = True,
    ) -> OrderFillRecord:
        """Apply one cumulative/incremental evidence observation.

        Cumulative status/wallet quantities advance by max(old, observed).
        Incremental observations advance by addition, but should normally carry a
        stable evidence_id so repeat delivery remains idempotent.
        """

        with self._gate:
            record = self._records.get(observation.order_id)
            if record is None:
                raise KeyError(
                    f"order not registered: {observation.order_id}"
                )

            if record.token_id != observation.token_id:
                raise ValueError("fill observation token affinity mismatch")

            seen_ids = set(record.observation_ids)
            observation_id = str(observation.evidence_id or "")

            if observation_id and observation_id in seen_ids:
                return record

            deduped_by_id: Dict[str, Fill] = {
                fill.fill_id: fill
                for fill in record.fills
            }

            for fill in fills:
                if fill.order_id != record.order_id:
                    raise ValueError("fill order affinity mismatch")
                if fill.token_id != record.token_id:
                    raise ValueError("fill token affinity mismatch")
                if fill.order_side is not record.order_side:
                    raise ValueError("fill side affinity mismatch")
                deduped_by_id.setdefault(fill.fill_id, fill)

            deduped_fills = tuple(
                sorted(
                    deduped_by_id.values(),
                    key=lambda fill: (fill.timestamp, fill.fill_id),
                )
            )

            fill_sum = sum(fill.size for fill in deduped_fills)

            if observation.cumulative:
                observed_size = max(
                    record.observed_filled_size,
                    observation.observed_size,
                    fill_sum,
                )
            else:
                observed_size = max(
                    record.observed_filled_size,
                    record.observed_filled_size
                    + observation.observed_size,
                    fill_sum,
                )

            confirmed_size = record.confirmed_filled_size
            if confirm_quantity:
                confirmed_size = max(
                    confirmed_size,
                    observation.observed_size
                    if observation.cumulative
                    else (
                        confirmed_size + observation.observed_size
                    ),
                    fill_sum,
                )

            (
                average,
                average_kind,
                average_source,
                covered_size,
            ) = self._choose_price_evidence(
                record=record,
                new_evidence=observation.price_evidence,
                deduped_fills=deduped_fills,
            )

            observation_ids = record.observation_ids
            if observation_id:
                observation_ids = observation_ids + (observation_id,)

            updated = replace(
                record,
                observed_filled_size=observed_size,
                confirmed_filled_size=confirmed_size,
                realized_average_price=average,
                realized_price_kind=average_kind,
                realized_price_source=average_source,
                price_covered_size=covered_size,
                fills=deduped_fills,
                observation_ids=observation_ids,
                sources=self._merge_sources(
                    record.sources,
                    observation.source,
                ),
                last_observed_at=max(
                    record.last_observed_at,
                    observation.observed_at,
                ),
            )

            if updated.confirmed_filled_size > updated.requested_size + 1e-9:
                updated = self._append_anomaly(
                    updated,
                    code="ACTUAL_FILL_EXCEEDS_REQUEST",
                    message=(
                        f"confirmed fill {updated.confirmed_filled_size:.8f} "
                        f"exceeds requested {updated.requested_size:.8f}"
                    ),
                    observed_at=observation.observed_at,
                )

            if (
                updated.realized_average_price is not None
                and updated.price_covered_size + 1e-9
                < updated.confirmed_filled_size
            ):
                updated = self._append_anomaly(
                    updated,
                    code="PARTIAL_PRICE_COVERAGE",
                    message=(
                        f"realized price covers {updated.price_covered_size:.8f} "
                        f"of {updated.confirmed_filled_size:.8f} filled"
                    ),
                    observed_at=observation.observed_at,
                )

            self._records[record.order_id] = updated
            return updated

    def observe_order_payload(
        self,
        *,
        order_id: str,
        payload: object,
        observed_at: Optional[float] = None,
        source: FillEvidenceSource = FillEvidenceSource.ORDER_STATUS,
        confirm_quantity: bool = True,
    ) -> OrderFillRecord:
        """Apply an order/status response using cumulative matched semantics."""

        with self._gate:
            record = self._records.get(str(order_id or ""))
            if record is None:
                raise KeyError(f"order not registered: {order_id}")

        now = float(observed_at or time.time())
        matched = extract_cumulative_matched_size(payload)
        price_evidence = extract_realized_price_evidence(
            payload,
            side=record.order_side,
        )
        fills = extract_incremental_fills(
            payload,
            order_id=record.order_id,
            token_id=record.token_id,
            side=record.order_side,
            observed_at=now,
        )

        payload_fingerprint = hashlib.blake2b(
            repr(
                (
                    record.order_id,
                    source.value,
                    round(matched, 10),
                    tuple(fill.fill_id for fill in fills),
                    price_evidence.kind.value,
                    price_evidence.source,
                    price_evidence.price,
                    price_evidence.represented_size,
                )
            ).encode("utf-8"),
            digest_size=16,
        ).hexdigest()

        observation = FillObservation(
            order_id=record.order_id,
            token_id=record.token_id,
            order_side=record.order_side,
            source=source,
            observed_size=matched,
            cumulative=True,
            observed_at=now,
            price_evidence=price_evidence,
            evidence_id=f"{source.value}:{payload_fingerprint}",
        )

        return self.apply_observation(
            observation,
            fills=fills,
            confirm_quantity=confirm_quantity,
        )

    def observe_trade_payload(
        self,
        *,
        order_id: str,
        payload: object,
        observed_at: Optional[float] = None,
    ) -> OrderFillRecord:
        """Apply an exact-order trade endpoint response.

        Trade rows are deduplicated, so repeated endpoint polling is idempotent.
        """

        return self.observe_order_payload(
            order_id=order_id,
            payload=payload,
            observed_at=observed_at,
            source=FillEvidenceSource.TRADE_ENDPOINT,
            confirm_quantity=True,
        )

    def observe_wallet_balance(
        self,
        *,
        order_id: str,
        current_balance: float,
        baseline: Optional[WalletBaseline],
        observed_at: Optional[float] = None,
    ) -> tuple[OrderFillRecord, WalletDeltaObservation]:
        """Apply baseline-backed wallet inventory evidence.

        If no valid baseline exists, the observation is retained only as a
        non-authoritative zero-attribution fact; total wallet balance is not
        claimed by the current order.
        """

        with self._gate:
            record = self._records.get(str(order_id or ""))
            if record is None:
                raise KeyError(f"order not registered: {order_id}")

        delta = wallet_delta_from_baseline(
            token_id=record.token_id,
            current_balance=current_balance,
            requested_size=record.requested_size,
            baseline=baseline,
            observed_at=observed_at,
        )

        if not delta.baseline_valid:
            with self._gate:
                current = self._records[record.order_id]
                current = self._append_anomaly(
                    current,
                    code="WALLET_BASELINE_MISSING",
                    message=(
                        "wallet balance observed without a valid pre-order "
                        "baseline; no fill quantity attributed"
                    ),
                    observed_at=delta.observed_at,
                )
                self._records[record.order_id] = current
                return current, delta

        fingerprint = hashlib.blake2b(
            repr(
                (
                    record.order_id,
                    baseline.observed_at if baseline else None,
                    baseline.balance if baseline else None,
                    delta.current_balance,
                    delta.delta,
                )
            ).encode("utf-8"),
            digest_size=16,
        ).hexdigest()

        observation = FillObservation(
            order_id=record.order_id,
            token_id=record.token_id,
            order_side=record.order_side,
            source=FillEvidenceSource.WALLET_DELTA,
            observed_size=delta.delta,
            cumulative=True,
            observed_at=delta.observed_at,
            price_evidence=None,
            evidence_id=f"wallet:{fingerprint}",
            metadata={
                "baseline_balance": delta.baseline_balance,
                "current_balance": delta.current_balance,
            },
        )

        updated = self.apply_observation(
            observation,
            fills=(),
            confirm_quantity=True,
        )

        if delta.overfill:
            with self._gate:
                updated = self._append_anomaly(
                    updated,
                    code="BASELINE_WALLET_OVERFILL",
                    message=(
                        f"baseline-backed wallet delta {delta.delta:.8f} "
                        f"exceeds requested {record.requested_size:.8f}; "
                        "inventory preserved for reconciliation"
                    ),
                    observed_at=delta.observed_at,
                )
                self._records[record.order_id] = updated

        return updated, delta

    def resolve(
        self,
        order_id: str,
        *,
        reason: str,
    ) -> OrderFillRecord:
        """Mark the accounting row resolved without changing any fill facts."""

        with self._gate:
            record = self._records.get(str(order_id or ""))
            if record is None:
                raise KeyError(f"order not registered: {order_id}")

            updated = replace(
                record,
                resolved=True,
                resolved_reason=str(reason or "resolved"),
                last_observed_at=max(record.last_observed_at, time.time()),
            )
            self._records[record.order_id] = updated
            return updated

    def fill_summary(self, order_id: str) -> Optional[FillSummary]:
        record = self.get(order_id)
        return record.to_fill_summary() if record is not None else None

    def conservative_accounting_price(
        self,
        order_id: str,
    ) -> PriceEvidence:
        """Return realized average when known, otherwise the order-local limit bound.

        This method makes the distinction explicit: fallback limit is suitable as
        a conservative accounting bound but is not relabelled as a true average.
        """

        record = self.get(order_id)
        if record is None:
            raise KeyError(f"order not registered: {order_id}")

        if record.realized_average_price is not None:
            return PriceEvidence(
                price=record.realized_average_price,
                kind=record.realized_price_kind,
                source=record.realized_price_source,
                represented_size=record.price_covered_size,
            )

        return PriceEvidence(
            price=record.submitted_limit_price,
            kind=PriceEvidenceKind.ORDER_LIMIT_BOUND,
            source="registered submitted limit",
            represented_size=0.0,
        )

    def prune(
        self,
        *,
        resolved_older_than_seconds: float = 900.0,
        active_order_ids: Iterable[str] = (),
        now: Optional[float] = None,
    ) -> int:
        """Bound old resolved rows while preserving active/unresolved accounting."""

        current_time = float(now or time.time())
        minimum_age = max(0.0, float(resolved_older_than_seconds))
        keep = {
            str(order_id)
            for order_id in active_order_ids
            if str(order_id)
        }

        removed = 0

        with self._gate:
            for order_id, record in list(self._records.items()):
                if order_id in keep:
                    continue
                if not record.resolved:
                    continue
                if current_time - record.last_observed_at < minimum_age:
                    continue

                self._records.pop(order_id, None)
                removed += 1

        return removed
