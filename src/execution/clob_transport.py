"""
Thread-safe CLOB transport adapter for the public portfolio edition.

The production system uses a synchronous venue SDK from multiple asynchronous
execution/lifecycle services.  This adapter isolates the concurrency and
transport concerns that are safe to demonstrate publicly:

- one serialized raw SDK instance
- write-priority transport coordination
- bounded lifecycle-read fairness
- transient retry/backoff for idempotent reads/cancels
- no automatic retry of ambiguous order POSTs
- generation-aware exact-order status caching
- short trade-query coalescing
- explicit market-metadata prewarming for local signing
- an exact, synchronous pre-submit validation boundary

The pre-submit hook is intentionally generic.  It can consume a caller-produced
execution permit, but this module contains no strategy/admission logic and no
asset-specific trading rules.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
import math
import random
import threading
import time
from typing import Callable, Dict, Iterable, Mapping, Optional, Protocol

from src.runtime.logging import runtime_print

try:
    from py_clob_client.clob_types import PartialCreateOrderOptions
except Exception:  # pragma: no cover - optional dependency during static review
    PartialCreateOrderOptions = None


class TransportStage(str, Enum):
    """How far a failed call progressed toward the venue."""

    PRE_NETWORK = "PRE_NETWORK"
    NETWORK_CALL = "NETWORK_CALL"
    RAW_POST_ENTERED = "RAW_POST_ENTERED"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result returned by a synchronous last-moment submission validator."""

    allowed: bool
    reason: str = ""
    evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", str(self.reason or ""))
        object.__setattr__(self, "evidence", dict(self.evidence))


@dataclass(frozen=True, slots=True)
class PreSubmitContext:
    """Caller-owned context presented immediately before the raw POST.

    The transport treats the fields as opaque execution metadata.  It neither
    creates nor interprets trading policy.
    """

    token_id: str = ""
    market_id: str = ""
    lifecycle_id: str = ""
    attempt_id: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_id", str(self.token_id or ""))
        object.__setattr__(self, "market_id", str(self.market_id or ""))
        object.__setattr__(self, "lifecycle_id", str(self.lifecycle_id or ""))
        object.__setattr__(self, "attempt_id", str(self.attempt_id or ""))
        object.__setattr__(self, "metadata", dict(self.metadata))


class PreSubmitValidator(Protocol):
    """Synchronous, networkless final validation interface."""

    def validate(self, context: PreSubmitContext) -> ValidationResult:
        ...


class AllowAllPreSubmitValidator:
    """Default validator used when a higher layer has no final permit to check."""

    def validate(self, context: PreSubmitContext) -> ValidationResult:
        return ValidationResult(allowed=True, reason="no validator policy configured")


RawPostEnterObserver = Callable[[PreSubmitContext], None]
TransportErrorListener = Callable[["TransportErrorEvent"], None]


@dataclass(frozen=True, slots=True)
class TransportErrorEvent:
    method: str
    stage: TransportStage
    transient: bool
    error_type: str
    message: str
    timestamp: float = field(default_factory=time.time)


class ClobTransportError(RuntimeError):
    """Transport exception carrying write provenance.

    `RAW_POST_ENTERED` is intentionally stronger than a plain timeout message:
    the venue may have accepted the order, so the lifecycle must reconcile rather
    than assuming that no order exists.
    """

    def __init__(
        self,
        *,
        method: str,
        stage: TransportStage,
        cause: BaseException,
        transient: bool = False,
    ) -> None:
        self.method = str(method or "")
        self.stage = stage
        self.cause = cause
        self.transient = bool(transient)

        super().__init__(
            f"{self.method} failed at {self.stage.value}: "
            f"{type(cause).__name__}: {cause}"
        )

    @property
    def venue_write_may_have_happened(self) -> bool:
        return (
            self.method == "post_order"
            and self.stage is TransportStage.RAW_POST_ENTERED
        )


class PreSubmitRejected(ClobTransportError):
    """Confirmed local no-post outcome produced by the final validator."""

    def __init__(
        self,
        *,
        context: PreSubmitContext,
        result: ValidationResult,
    ) -> None:
        self.context = context
        self.result = result
        cause = RuntimeError(result.reason or "pre-submit validation rejected")
        super().__init__(
            method="post_order",
            stage=TransportStage.PRE_NETWORK,
            cause=cause,
            transient=False,
        )


@dataclass(frozen=True, slots=True)
class TransportPolicy:
    """Operational transport settings for the portfolio edition.

    These are public demonstration defaults, not production strategy parameters.
    """

    serialize_sdk_calls: bool = True

    write_min_intercall_gap_seconds: float = 0.005
    read_min_intercall_gap_seconds: float = 0.020

    transport_error_cooldown_seconds: float = 0.25
    transport_final_error_cooldown_seconds: float = 0.50
    write_max_cooldown_wait_seconds: float = 0.05

    lifecycle_read_max_write_starve_seconds: float = 0.20

    get_order_attempts: int = 2
    get_trades_attempts: int = 1
    cancel_order_attempts: int = 2
    balance_attempts: int = 1

    transient_retry_sleep_seconds: float = 0.05
    transient_retry_jitter_seconds: float = 0.025

    get_order_cache_seconds: float = 0.15
    get_trades_cache_seconds: float = 0.10

    get_order_cache_max_entries: int = 256
    get_trades_cache_max_entries: int = 64
    order_generation_max_entries: int = 4096


@dataclass(frozen=True, slots=True)
class MarketOrderMetadata:
    """Venue metadata required to create/sign an order locally."""

    token_id: str
    tick_size: Optional[float] = None
    neg_risk: Optional[bool] = None
    condition_id: str = ""

    def __post_init__(self) -> None:
        token_id = str(self.token_id or "").strip()
        if not token_id:
            raise ValueError("token_id is required")

        tick_size = self.tick_size
        if tick_size is not None:
            tick_size = float(tick_size)
            if not math.isfinite(tick_size) or tick_size <= 0.0:
                raise ValueError("tick_size must be positive")

        object.__setattr__(self, "token_id", token_id)
        object.__setattr__(self, "tick_size", tick_size)
        object.__setattr__(self, "condition_id", str(self.condition_id or ""))


@dataclass(frozen=True, slots=True)
class PrewarmResult:
    tokens: int
    ready: int
    resolved: int
    failed: int
    ready_tokens: tuple[str, ...]
    failed_tokens: tuple[str, ...]
    version: Optional[int] = None


class _RefCountedLock:
    """Lock that can be pruned only when nobody holds or waits for it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._meta = threading.Lock()
        self._users = 0

    def __enter__(self) -> "_RefCountedLock":
        with self._meta:
            self._users += 1
        try:
            self._lock.acquire()
            return self
        except BaseException:
            with self._meta:
                self._users = max(0, self._users - 1)
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._lock.release()
        finally:
            with self._meta:
                self._users = max(0, self._users - 1)

    @property
    def in_use(self) -> bool:
        with self._meta:
            return self._users > 0


def _coerce_optional_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


class ClobTransport:
    """Thread-safe coordinator around one synchronous CLOB SDK instance."""

    _GENERIC_NETWORK_METHODS = {
        "get_orders",
        "get_open_orders",
        "get_order_book",
        "get_order_books",
        "get_midpoint",
        "get_price",
        "get_spread",
        "get_last_trade_price",
        "get_server_time",
        "get_ok",
        "cancel_all",
        "cancel_orders",
        "delete_api_key",
        "get_api_keys",
        "get_notifications",
        "get_tick_size",
        "get_neg_risk",
        "get_clob_market_info",
        "get_version",
        "get_fee_rate_bps",
    }

    _LIFECYCLE_READ_METHODS = {
        "get_order",
        "get_balance_allowance",
        "get_orders",
        "get_open_orders",
    }

    def __init__(
        self,
        raw_client,
        *,
        policy: TransportPolicy = TransportPolicy(),
        pre_submit_validator: Optional[PreSubmitValidator] = None,
        error_listener: Optional[TransportErrorListener] = None,
    ) -> None:
        self._raw = raw_client
        self.policy = policy
        self.pre_submit_validator = (
            pre_submit_validator or AllowAllPreSubmitValidator()
        )
        self.error_listener = error_listener

        self._state_gate = threading.RLock()
        self._cache_gate = threading.RLock()
        self._sdk_instance_gate = threading.RLock()

        self._transport_condition = threading.Condition(threading.Lock())
        self._transport_active = False
        self._transport_waiting_writes = 0
        self._transport_last_lifecycle_read_escape = 0.0

        self._last_network_call_ts = 0.0
        self._transport_backoff_until = 0.0

        self._get_order_cache: Dict[str, tuple[float, int, object]] = {}
        self._get_trades_cache: Dict[str, tuple[float, object]] = {}
        self._get_order_key_locks: Dict[str, _RefCountedLock] = {}
        self._get_trades_key_locks: Dict[str, _RefCountedLock] = {}

        self._order_generation_by_oid: Dict[str, int] = {}

        self._metadata_by_token: Dict[str, MarketOrderMetadata] = {}
        self._create_metadata_key_locks: Dict[str, _RefCountedLock] = {}
        self._create_version_lock = threading.Lock()
        self._create_hotpath_required_tokens: set[str] = set()
        self._create_hotpath_ready_tokens: set[str] = set()
        self._create_hotpath_version: Optional[int] = None

        self._request_stats: Dict[str, float] = {}
        self._request_stats_gate = threading.Lock()

    def __getattr__(self, name: str):
        attr = getattr(self._raw, name)
        if callable(attr) and name in self._GENERIC_NETWORK_METHODS:
            def _coordinated(*args, **kwargs):
                return self._call_with_retry(name, attr, *args, **kwargs)
            return _coordinated
        return attr

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @staticmethod
    def _clone(value):
        try:
            return copy.deepcopy(value)
        except Exception:
            return value

    def _stat_inc(self, key: str, amount: float = 1.0) -> None:
        try:
            with self._request_stats_gate:
                self._request_stats[key] = (
                    float(self._request_stats.get(key, 0.0) or 0.0)
                    + float(amount)
                )
        except Exception:
            pass

    def request_stats(self, *, reset: bool = False) -> dict[str, float]:
        with self._request_stats_gate:
            result = dict(self._request_stats)
            if reset:
                self._request_stats.clear()
            return result

    def _notify_error(
        self,
        *,
        method: str,
        stage: TransportStage,
        exc: BaseException,
        transient: bool,
    ) -> None:
        listener = self.error_listener
        if listener is None:
            return

        try:
            listener(
                TransportErrorEvent(
                    method=str(method),
                    stage=stage,
                    transient=bool(transient),
                    error_type=type(exc).__name__,
                    message=str(exc)[:500],
                )
            )
        except Exception:
            # Error telemetry must never mutate transport semantics.
            pass

    # ------------------------------------------------------------------
    # Error classification / retry
    # ------------------------------------------------------------------

    @staticmethod
    def _root_exception(exc: BaseException) -> BaseException:
        if isinstance(exc, ClobTransportError):
            return exc.cause
        return exc

    @classmethod
    def _is_transient_error(cls, exc: BaseException) -> bool:
        root = cls._root_exception(exc)
        text = str(root).lower()

        status_code = getattr(root, "status_code", None)
        if status_code is None:
            response = getattr(root, "response", None)
            status_code = getattr(response, "status_code", None)

        if status_code in {408, 425, 429, 500, 502, 503, 504}:
            return True

        markers = (
            "server disconnected",
            "connection reset",
            "connection aborted",
            "remote disconnected",
            "read timed out",
            "timeout",
            "temporarily unavailable",
            "too many requests",
            "rate limit",
            "rate_limit",
            "cloudflare",
            "status_code=429",
            "status code 429",
            "http 429",
            "request exception",
            "status_code=none",
            "eof occurred in violation of protocol",
        )
        if any(marker in text for marker in markers):
            return True

        return "ssl" in text and "eof" in text

    @staticmethod
    def _is_write_method(method: str) -> bool:
        method = str(method or "").lower()
        return method.startswith(("post", "cancel", "delete", "update"))

    def _attempts_for(self, method: str) -> int:
        method = str(method or "")

        # Never automatically retry an ambiguous order POST.  If the socket failed
        # after raw POST entry, reconciliation must determine whether an order exists.
        if method == "post_order":
            return 1
        if method == "get_order":
            return max(1, int(self.policy.get_order_attempts))
        if method == "get_trades":
            return max(1, int(self.policy.get_trades_attempts))
        if method.startswith("cancel"):
            return max(1, int(self.policy.cancel_order_attempts))
        if method == "get_balance_allowance":
            return max(1, int(self.policy.balance_attempts))
        return 1

    def _arm_transport_backoff(self, *, final: bool = False) -> None:
        cooldown = max(
            0.0,
            float(self.policy.transport_error_cooldown_seconds),
        )
        if final:
            cooldown = max(
                cooldown,
                float(self.policy.transport_final_error_cooldown_seconds),
            )

        with self._state_gate:
            self._transport_backoff_until = max(
                float(self._transport_backoff_until or 0.0),
                time.monotonic() + cooldown,
            )

    # ------------------------------------------------------------------
    # Write-priority transport serialization
    # ------------------------------------------------------------------

    def _acquire_transport_slot(self, method: str) -> bool:
        is_write = self._is_write_method(method)
        lifecycle_read = str(method) in self._LIFECYCLE_READ_METHODS
        max_starve = max(
            0.05,
            float(self.policy.lifecycle_read_max_write_starve_seconds),
        )

        queued_at = time.monotonic()
        escaped_write_queue = False

        with self._transport_condition:
            if is_write:
                self._transport_waiting_writes += 1

            try:
                while True:
                    now = time.monotonic()
                    waited = max(0.0, now - queued_at)

                    escape_ready = bool(
                        lifecycle_read
                        and self._transport_waiting_writes > 0
                        and waited >= max_starve
                        and (
                            now - self._transport_last_lifecycle_read_escape
                            >= max_starve
                        )
                    )

                    write_priority_blocks = bool(
                        not is_write
                        and self._transport_waiting_writes > 0
                        and not escape_ready
                    )

                    if not self._transport_active and not write_priority_blocks:
                        escaped_write_queue = escape_ready
                        if escape_ready:
                            self._transport_last_lifecycle_read_escape = now
                        break

                    self._transport_condition.wait(
                        timeout=0.05 if lifecycle_read else 0.25
                    )

                self._transport_active = True
            finally:
                if is_write:
                    self._transport_waiting_writes = max(
                        0,
                        self._transport_waiting_writes - 1,
                    )

        queue_wait = max(0.0, time.monotonic() - queued_at)
        if queue_wait > 0.001:
            self._stat_inc(f"queue_wait_{method}_seconds", queue_wait)

        if escaped_write_queue:
            self._stat_inc("lifecycle_read_write_starvation_escape")

        return is_write

    def _release_transport_slot(self) -> None:
        with self._transport_condition:
            self._transport_active = False
            self._transport_condition.notify_all()

    # ------------------------------------------------------------------
    # Exact raw POST boundary
    # ------------------------------------------------------------------

    def _validate_before_raw_post(
        self,
        *,
        context: PreSubmitContext,
        observer: Optional[RawPostEnterObserver],
    ) -> None:
        try:
            result = self.pre_submit_validator.validate(context)
        except PreSubmitRejected:
            raise
        except Exception as exc:
            # Validator failure occurs before the raw SDK call and is therefore a
            # confirmed local no-post outcome.
            raise ClobTransportError(
                method="post_order",
                stage=TransportStage.PRE_NETWORK,
                cause=exc,
                transient=False,
            ) from exc

        if not isinstance(result, ValidationResult):
            raise ClobTransportError(
                method="post_order",
                stage=TransportStage.PRE_NETWORK,
                cause=TypeError(
                    "PreSubmitValidator.validate() must return ValidationResult"
                ),
                transient=False,
            )

        if not result.allowed:
            raise PreSubmitRejected(context=context, result=result)

        if observer is not None:
            try:
                observer(context)
            except Exception as exc:
                # The observer is the final local ownership handoff.  If it cannot
                # record that handoff, raw POST must not begin.
                raise ClobTransportError(
                    method="post_order",
                    stage=TransportStage.PRE_NETWORK,
                    cause=exc,
                    transient=False,
                ) from exc

    def _network_call_once(
        self,
        method: str,
        fn,
        *args,
        pre_submit_context: Optional[PreSubmitContext] = None,
        raw_post_enter_observer: Optional[RawPostEnterObserver] = None,
        **kwargs,
    ):
        if not self.policy.serialize_sdk_calls:
            if method == "post_order":
                self._validate_before_raw_post(
                    context=pre_submit_context or PreSubmitContext(),
                    observer=raw_post_enter_observer,
                )
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    raise ClobTransportError(
                        method=method,
                        stage=TransportStage.RAW_POST_ENTERED,
                        cause=exc,
                        transient=self._is_transient_error(exc),
                    ) from exc

            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                raise ClobTransportError(
                    method=method,
                    stage=TransportStage.NETWORK_CALL,
                    cause=exc,
                    transient=self._is_transient_error(exc),
                ) from exc

        is_write = self._acquire_transport_slot(method)

        try:
            with self._state_gate:
                cooldown_left = (
                    float(self._transport_backoff_until or 0.0)
                    - time.monotonic()
                )
                last_call = float(self._last_network_call_ts or 0.0)

            if cooldown_left > 0.0:
                if is_write:
                    cooldown_left = min(
                        cooldown_left,
                        max(
                            0.0,
                            float(self.policy.write_max_cooldown_wait_seconds),
                        ),
                    )
                self._stat_inc("transport_cooldown_seconds", cooldown_left)
                time.sleep(cooldown_left)

            min_gap = (
                float(self.policy.write_min_intercall_gap_seconds)
                if is_write
                else float(self.policy.read_min_intercall_gap_seconds)
            )
            wait_for_gap = min_gap - (time.monotonic() - last_call)
            if wait_for_gap > 0.0:
                time.sleep(wait_for_gap)

            try:
                with self._sdk_instance_gate:
                    # This is the exact last networkless observation point.
                    # No await, HTTP request or strategy computation is performed
                    # between validation/ownership handoff and raw post_order().
                    if method == "post_order":
                        self._validate_before_raw_post(
                            context=pre_submit_context or PreSubmitContext(),
                            observer=raw_post_enter_observer,
                        )

                    try:
                        return fn(*args, **kwargs)
                    except Exception as exc:
                        stage = (
                            TransportStage.RAW_POST_ENTERED
                            if method == "post_order"
                            else TransportStage.NETWORK_CALL
                        )
                        raise ClobTransportError(
                            method=method,
                            stage=stage,
                            cause=exc,
                            transient=self._is_transient_error(exc),
                        ) from exc
            finally:
                with self._state_gate:
                    self._last_network_call_ts = time.monotonic()
        finally:
            self._release_transport_slot()

    def _call_with_retry(self, method: str, fn, *args, **kwargs):
        attempts = self._attempts_for(method)
        last_exc: Optional[BaseException] = None

        for attempt in range(attempts):
            try:
                self._stat_inc(f"network_{method}")
                return self._network_call_once(method, fn, *args, **kwargs)

            except PreSubmitRejected:
                self._stat_inc("pre_submit_rejected")
                raise

            except ClobTransportError as exc:
                last_exc = exc
                transient = bool(exc.transient)
                self._stat_inc(f"error_{method}")

                # A raw order POST may have reached the venue.  Retrying would risk
                # a duplicate order, so reconciliation owns the next step.
                if (
                    method == "post_order"
                    and exc.stage is TransportStage.RAW_POST_ENTERED
                ):
                    self._arm_transport_backoff(final=True)
                    self._notify_error(
                        method=method,
                        stage=exc.stage,
                        exc=exc.cause,
                        transient=transient,
                    )
                    raise

                if transient:
                    self._arm_transport_backoff(
                        final=(attempt >= attempts - 1)
                    )

                if not transient or attempt >= attempts - 1:
                    self._notify_error(
                        method=method,
                        stage=exc.stage,
                        exc=exc.cause,
                        transient=transient,
                    )
                    raise

                self._stat_inc(f"retry_{method}")

                sleep_seconds = (
                    max(0.0, float(self.policy.transient_retry_sleep_seconds))
                    * (attempt + 1)
                    + random.uniform(
                        0.0,
                        max(
                            0.0,
                            float(self.policy.transient_retry_jitter_seconds),
                        ),
                    )
                )
                runtime_print(
                    f"[clob] retry {method} after transient error "
                    f"({attempt + 2}/{attempts}) in {sleep_seconds:.2f}s"
                )
                time.sleep(sleep_seconds)

        if isinstance(last_exc, BaseException):
            raise last_exc
        raise RuntimeError(f"{method} failed without an exception")

    # ------------------------------------------------------------------
    # Order-generation / cache ownership
    # ------------------------------------------------------------------

    def order_generation(self, order_id: str) -> int:
        order_id = str(order_id or "")
        if not order_id:
            return 0

        with self._cache_gate:
            return int(self._order_generation_by_oid.get(order_id, 0) or 0)

    def _bump_order_generation(self, order_id: str) -> int:
        order_id = str(order_id or "")
        if not order_id:
            return 0

        with self._cache_gate:
            generation = (
                int(self._order_generation_by_oid.get(order_id, 0) or 0)
                + 1
            )
            self._order_generation_by_oid[order_id] = generation
            self._get_order_cache.pop(order_id, None)

            cap = max(128, int(self.policy.order_generation_max_entries))
            if len(self._order_generation_by_oid) > cap:
                remove_count = max(1, cap // 4)
                for old_oid in list(self._order_generation_by_oid)[:remove_count]:
                    if old_oid != order_id:
                        self._order_generation_by_oid.pop(old_oid, None)

            return generation

    @staticmethod
    def _extract_order_id(args, kwargs) -> str:
        if args:
            return str(args[0] or "")
        return str(
            kwargs.get("order_id", kwargs.get("orderID", "")) or ""
        )

    def _key_lock(
        self,
        table: Dict[str, _RefCountedLock],
        key: str,
    ) -> _RefCountedLock:
        with self._cache_gate:
            lock = table.get(key)
            if lock is None:
                lock = _RefCountedLock()
                table[key] = lock
            return lock

    def _prune_unused_key_locks(
        self,
        table: Dict[str, _RefCountedLock],
        *,
        keep_keys: set[str],
        target_size: int,
    ) -> None:
        if len(table) <= target_size:
            return

        for key, lock in list(table.items()):
            if len(table) <= target_size:
                break
            if key in keep_keys or lock.in_use:
                continue
            table.pop(key, None)

    # ------------------------------------------------------------------
    # Public coordinated SDK methods
    # ------------------------------------------------------------------

    def post_order(
        self,
        *args,
        pre_submit_context: Optional[PreSubmitContext] = None,
        raw_post_enter_observer: Optional[RawPostEnterObserver] = None,
        **kwargs,
    ):
        """Post one signed order through the exact validation boundary."""

        try:
            result = self._call_with_retry(
                "post_order",
                self._raw.post_order,
                *args,
                pre_submit_context=pre_submit_context,
                raw_post_enter_observer=raw_post_enter_observer,
                **kwargs,
            )
        except Exception:
            # An ambiguous post may have created an order whose ID is not known
            # locally.  Cached status from older orders must not be reused as proof.
            with self._cache_gate:
                self._get_order_cache.clear()
            raise

        order_id = ""
        if isinstance(result, dict):
            order_id = str(
                result.get("orderID")
                or result.get("order_id")
                or result.get("id")
                or ""
            )

        if order_id:
            self._bump_order_generation(order_id)
        else:
            with self._cache_gate:
                self._get_order_cache.clear()

        return result

    def get_order(self, *args, **kwargs):
        """Generation-aware exact-order status lookup with short coalescing."""

        order_id = self._extract_order_id(args, kwargs)
        key = order_id or repr((args, sorted(kwargs.items())))
        ttl = max(0.0, float(self.policy.get_order_cache_seconds))

        with self._key_lock(self._get_order_key_locks, key):
            # A cancel can race the read.  Retry once in the new generation; if a
            # second transition races too, return unknown rather than stale state.
            for _ in range(2):
                generation_before = (
                    self.order_generation(order_id) if order_id else 0
                )

                if order_id and ttl > 0.0:
                    with self._cache_gate:
                        record = self._get_order_cache.get(order_id)

                    if (
                        record
                        and int(record[1]) == generation_before
                        and time.monotonic() - float(record[0]) <= ttl
                    ):
                        self._stat_inc("cache_hit_get_order")
                        return self._clone(record[2])

                result = self._call_with_retry(
                    "get_order",
                    self._raw.get_order,
                    *args,
                    **kwargs,
                )

                if not order_id:
                    return self._clone(result)

                generation_after = self.order_generation(order_id)
                if generation_after != generation_before:
                    self._stat_inc("discard_stale_get_order_generation")
                    continue

                with self._cache_gate:
                    current = int(
                        self._order_generation_by_oid.get(order_id, 0) or 0
                    )
                    if current != generation_before:
                        self._stat_inc(
                            "discard_stale_get_order_generation"
                        )
                        continue

                    self._get_order_cache[order_id] = (
                        time.monotonic(),
                        generation_before,
                        self._clone(result),
                    )

                    cap = max(
                        16,
                        int(self.policy.get_order_cache_max_entries),
                    )
                    if len(self._get_order_cache) > cap:
                        remove_count = max(1, cap // 4)
                        oldest = sorted(
                            self._get_order_cache.items(),
                            key=lambda item: item[1][0],
                        )[:remove_count]

                        for old_key, _record in oldest:
                            self._get_order_cache.pop(old_key, None)

                    self._prune_unused_key_locks(
                        self._get_order_key_locks,
                        keep_keys=set(self._get_order_cache),
                        target_size=max(32, cap * 2),
                    )

                return self._clone(result)

            return {}

    def get_trades(self, *args, **kwargs):
        """Short singleflight/coalescing window for duplicate trade reads."""

        fn = getattr(self._raw, "get_trades")
        key = repr((args, sorted(kwargs.items())))
        ttl = max(0.0, float(self.policy.get_trades_cache_seconds))

        with self._key_lock(self._get_trades_key_locks, key):
            if ttl > 0.0:
                with self._cache_gate:
                    record = self._get_trades_cache.get(key)

                if record and time.monotonic() - float(record[0]) <= ttl:
                    self._stat_inc("cache_hit_get_trades")
                    return self._clone(record[1])

            result = self._call_with_retry(
                "get_trades",
                fn,
                *args,
                **kwargs,
            )

            with self._cache_gate:
                self._get_trades_cache[key] = (
                    time.monotonic(),
                    self._clone(result),
                )

                cap = max(
                    8,
                    int(self.policy.get_trades_cache_max_entries),
                )
                if len(self._get_trades_cache) > cap:
                    remove_count = max(1, cap // 4)
                    oldest = sorted(
                        self._get_trades_cache.items(),
                        key=lambda item: item[1][0],
                    )[:remove_count]

                    for old_key, _record in oldest:
                        self._get_trades_cache.pop(old_key, None)

                self._prune_unused_key_locks(
                    self._get_trades_key_locks,
                    keep_keys=set(self._get_trades_cache),
                    target_size=max(16, cap * 2),
                )

            return self._clone(result)

    def get_balance_allowance(self, *args, **kwargs):
        return self._call_with_retry(
            "get_balance_allowance",
            self._raw.get_balance_allowance,
            *args,
            **kwargs,
        )

    def cancel_order(self, *args, **kwargs):
        """Cancel while invalidating status generations on both sides of the call."""

        order_id = self._extract_order_id(args, kwargs)

        if order_id:
            # Invalidate any status task/cache that began before cancel intent.
            self._bump_order_generation(order_id)

        try:
            return self._call_with_retry(
                "cancel_order",
                self._raw.cancel_order,
                *args,
                **kwargs,
            )
        finally:
            if order_id:
                # Also invalidate reads that raced while cancel was in-flight.
                self._bump_order_generation(order_id)
            else:
                with self._cache_gate:
                    self._get_order_cache.clear()

    # ------------------------------------------------------------------
    # Local order-signing metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _order_token_id(order_args) -> str:
        try:
            if isinstance(order_args, dict):
                return str(
                    order_args.get("token_id")
                    or order_args.get("tokenID")
                    or order_args.get("asset_id")
                    or ""
                )

            return str(
                getattr(order_args, "token_id", None)
                or getattr(order_args, "tokenID", None)
                or getattr(order_args, "asset_id", None)
                or ""
            )
        except Exception:
            return ""

    @staticmethod
    def _raw_private_cache(raw, name: str):
        try:
            cache = getattr(raw, name, None)
            return cache if isinstance(cache, dict) else None
        except Exception:
            return None

    @staticmethod
    def _make_partial_create_options(
        tick_size: object,
        neg_risk: object,
    ):
        if PartialCreateOrderOptions is None:
            return None

        tick_string = str(tick_size or "")
        normalized_neg = _coerce_optional_bool(neg_risk)
        neg_bool = (
            bool(neg_risk)
            if normalized_neg is None
            else normalized_neg
        )

        variants = (
            {"tick_size": tick_string, "neg_risk": neg_bool},
            {"tick_size": tick_string},
            {"tickSize": tick_string, "negRisk": neg_bool},
            {"tickSize": tick_string},
        )

        for kwargs in variants:
            try:
                return PartialCreateOrderOptions(**kwargs)
            except TypeError:
                continue

        try:
            return PartialCreateOrderOptions(tick_string, neg_bool)
        except Exception:
            return None

    def register_market_metadata(
        self,
        token_id: str,
        *,
        tick_size: Optional[float] = None,
        neg_risk: Optional[bool] = None,
        condition_id: str = "",
    ) -> MarketOrderMetadata:
        metadata = MarketOrderMetadata(
            token_id=token_id,
            tick_size=tick_size,
            neg_risk=neg_risk,
            condition_id=condition_id,
        )

        with self._cache_gate:
            previous = self._metadata_by_token.get(metadata.token_id)

            if previous is not None:
                metadata = MarketOrderMetadata(
                    token_id=metadata.token_id,
                    tick_size=(
                        metadata.tick_size
                        if metadata.tick_size is not None
                        else previous.tick_size
                    ),
                    neg_risk=(
                        metadata.neg_risk
                        if metadata.neg_risk is not None
                        else previous.neg_risk
                    ),
                    condition_id=(
                        metadata.condition_id
                        or previous.condition_id
                    ),
                )

            self._metadata_by_token[metadata.token_id] = metadata

        self._hydrate_raw_market_metadata(metadata)
        return metadata

    def market_metadata(
        self,
        token_id: str,
    ) -> Optional[MarketOrderMetadata]:
        with self._cache_gate:
            return self._metadata_by_token.get(str(token_id or ""))

    def _hydrate_raw_market_metadata(
        self,
        metadata: MarketOrderMetadata,
    ) -> None:
        """Feature-detect and hydrate SDK-local market metadata caches.

        This vendor-specific compatibility logic is intentionally isolated here.
        If a future SDK exposes an official cache/prewarm API, only this adapter
        needs to change.
        """

        token_id = metadata.token_id

        with self._sdk_instance_gate:
            tick_cache = self._raw_private_cache(
                self._raw,
                "_ClobClient__tick_sizes",
            )
            neg_cache = self._raw_private_cache(
                self._raw,
                "_ClobClient__neg_risk",
            )
            condition_cache = self._raw_private_cache(
                self._raw,
                "_ClobClient__token_condition_map",
            )

            if tick_cache is not None and metadata.tick_size is not None:
                tick_cache[token_id] = str(metadata.tick_size)

            if neg_cache is not None and metadata.neg_risk is not None:
                neg_cache[token_id] = bool(metadata.neg_risk)

            if condition_cache is not None and metadata.condition_id:
                condition_cache[token_id] = metadata.condition_id

    def _ensure_order_version_cached(self) -> int:
        cached_attr = "_ClobClient__cached_version"

        with self._sdk_instance_gate:
            cached = getattr(self._raw, cached_attr, None)

        if cached is not None:
            return int(cached)

        with self._create_version_lock:
            with self._sdk_instance_gate:
                cached = getattr(self._raw, cached_attr, None)

            if cached is None:
                fn = getattr(self._raw, "get_version", None)
                if not callable(fn):
                    return 2

                cached = self._call_with_retry("get_version", fn)

                with self._sdk_instance_gate:
                    try:
                        setattr(self._raw, cached_attr, int(cached))
                    except Exception:
                        pass

            return int(cached or 2)

    def _ensure_fee_rate_cached(
        self,
        token_id: str,
        version: int,
    ) -> None:
        if int(version or 2) != 1:
            return

        token_id = str(token_id or "")
        if not token_id:
            return

        with self._sdk_instance_gate:
            cache = self._raw_private_cache(
                self._raw,
                "_ClobClient__fee_rates",
            )
            if cache is not None and token_id in cache:
                return

        lock = self._key_lock(
            self._create_metadata_key_locks,
            f"fee:{token_id}",
        )

        with lock:
            with self._sdk_instance_gate:
                cache = self._raw_private_cache(
                    self._raw,
                    "_ClobClient__fee_rates",
                )
                if cache is not None and token_id in cache:
                    return

            fn = getattr(self._raw, "get_fee_rate_bps", None)
            if callable(fn):
                self._call_with_retry(
                    "get_fee_rate_bps",
                    fn,
                    token_id,
                )

    def _resolve_create_metadata(
        self,
        token_id: str,
    ) -> MarketOrderMetadata:
        token_id = str(token_id or "").strip()
        if not token_id:
            raise ValueError("order token_id is required")

        lock = self._key_lock(
            self._create_metadata_key_locks,
            token_id,
        )

        with lock:
            metadata = self.market_metadata(token_id)
            tick_size = metadata.tick_size if metadata else None
            neg_risk = metadata.neg_risk if metadata else None
            condition_id = metadata.condition_id if metadata else ""

            if tick_size is None:
                fn = getattr(self._raw, "get_tick_size", None)
                if not callable(fn):
                    raise AttributeError("CLOB SDK has no get_tick_size")
                tick_size = float(
                    self._call_with_retry(
                        "get_tick_size",
                        fn,
                        token_id,
                    )
                )

            if neg_risk is None:
                fn = getattr(self._raw, "get_neg_risk", None)
                if not callable(fn):
                    raise AttributeError("CLOB SDK has no get_neg_risk")
                neg_risk = _coerce_optional_bool(
                    self._call_with_retry(
                        "get_neg_risk",
                        fn,
                        token_id,
                    )
                )

            return self.register_market_metadata(
                token_id,
                tick_size=tick_size,
                neg_risk=neg_risk,
                condition_id=condition_id,
            )

    def prewarm_create_hotpath(
        self,
        token_ids: Iterable[str],
    ) -> PrewarmResult:
        """Resolve network-capable signing prerequisites before time-sensitive use."""

        tokens = tuple(
            dict.fromkeys(
                str(token or "")
                for token in token_ids
                if str(token or "")
            )
        )

        with self._cache_gate:
            self._create_hotpath_required_tokens.update(tokens)

        if PartialCreateOrderOptions is None:
            return PrewarmResult(
                tokens=len(tokens),
                ready=0,
                resolved=0,
                failed=len(tokens),
                ready_tokens=(),
                failed_tokens=tokens,
                version=None,
            )

        version = self._ensure_order_version_cached()
        self._create_hotpath_version = version

        ready: list[str] = []
        failed: list[str] = []
        resolved_count = 0

        for token_id in tokens:
            try:
                before = self.market_metadata(token_id)
                had_all_metadata = bool(
                    before
                    and before.tick_size is not None
                    and before.neg_risk is not None
                )

                metadata = self._resolve_create_metadata(token_id)

                if not had_all_metadata:
                    resolved_count += 1

                self._ensure_fee_rate_cached(token_id, version)

                options = self._make_partial_create_options(
                    metadata.tick_size,
                    metadata.neg_risk,
                )
                if options is None:
                    raise RuntimeError(
                        "unable to construct explicit create-order options"
                    )

                # Verify that the installed SDK exposes the caches this adapter
                # relies on to keep the sign path networkless.
                with self._sdk_instance_gate:
                    tick_cache = self._raw_private_cache(
                        self._raw,
                        "_ClobClient__tick_sizes",
                    )
                    neg_cache = self._raw_private_cache(
                        self._raw,
                        "_ClobClient__neg_risk",
                    )

                    if (
                        tick_cache is None
                        or token_id not in tick_cache
                        or neg_cache is None
                        or token_id not in neg_cache
                    ):
                        raise RuntimeError(
                            "SDK local metadata cache hydration unavailable"
                        )

                with self._cache_gate:
                    self._create_hotpath_ready_tokens.add(token_id)

                ready.append(token_id)

            except Exception as exc:
                failed.append(token_id)
                runtime_print(
                    f"[clob] prewarm failed for {token_id[:8]}: "
                    f"{type(exc).__name__}: {exc}"
                )

        self._stat_inc("create_hotpath_prewarm_ready", len(ready))
        self._stat_inc(
            "create_hotpath_prewarm_resolved",
            resolved_count,
        )
        self._stat_inc("create_hotpath_prewarm_failed", len(failed))

        return PrewarmResult(
            tokens=len(tokens),
            ready=len(ready),
            resolved=resolved_count,
            failed=len(failed),
            ready_tokens=tuple(ready),
            failed_tokens=tuple(failed),
            version=version,
        )

    def create_order(self, order_args, options=None):
        """Create/sign an order while serializing access to SDK-local state."""

        token_id = self._order_token_id(order_args)

        with self._cache_gate:
            hot_required = (
                token_id in self._create_hotpath_required_tokens
            )
            hot_ready = token_id in self._create_hotpath_ready_tokens

        if hot_required and not hot_ready and options is None:
            raise RuntimeError(
                f"create-order hot path not prewarmed for token {token_id}"
            )

        if hot_ready and options is None:
            metadata = self.market_metadata(token_id)
            if (
                metadata is None
                or metadata.tick_size is None
                or metadata.neg_risk is None
            ):
                raise RuntimeError(
                    f"prewarmed metadata missing for token {token_id}"
                )

            resolved = self._make_partial_create_options(
                metadata.tick_size,
                metadata.neg_risk,
            )
            if resolved is None:
                raise RuntimeError(
                    f"explicit create options unavailable for token {token_id}"
                )

            version = self._create_hotpath_version
            if version is None:
                raise RuntimeError("prewarmed order version missing")

            self._ensure_fee_rate_cached(token_id, version)

            with self._sdk_instance_gate:
                self._hydrate_raw_market_metadata(metadata)
                self._stat_inc("create_hotpath_local")
                return self._raw.create_order(order_args, resolved)

        metadata = self._resolve_create_metadata(token_id)
        resolved = (
            options
            if options is not None
            else self._make_partial_create_options(
                metadata.tick_size,
                metadata.neg_risk,
            )
        )

        version = self._ensure_order_version_cached()
        self._ensure_fee_rate_cached(token_id, version)

        if resolved is None:
            # Compatibility path for SDK versions without explicit create options.
            # It is serialized through the coordinator because the SDK may perform
            # hidden metadata HTTP during create_order().
            return self._call_with_retry(
                "create_order_metadata",
                self._raw.create_order,
                order_args,
            )

        with self._sdk_instance_gate:
            return self._raw.create_order(order_args, resolved)

    def prune_token_state(self, active_tokens: Iterable[str]) -> int:
        """Bound token-keyed signing metadata across market rotations."""

        keep = {
            str(token)
            for token in active_tokens
            if str(token)
        }
        removed = 0

        with self._cache_gate:
            for token_id in list(self._metadata_by_token):
                if token_id not in keep:
                    self._metadata_by_token.pop(token_id, None)
                    removed += 1

            self._create_hotpath_required_tokens.intersection_update(keep)
            self._create_hotpath_ready_tokens.intersection_update(keep)

            for key, lock in list(
                self._create_metadata_key_locks.items()
            ):
                token_key = (
                    key[4:] if key.startswith("fee:") else key
                )
                if token_key not in keep and not lock.in_use:
                    self._create_metadata_key_locks.pop(key, None)

        # Vendor cache pruning is performed under the same SDK instance gate as
        # signing and network calls, preventing concurrent dictionary mutation.
        with self._sdk_instance_gate:
            for private_name in (
                "_ClobClient__tick_sizes",
                "_ClobClient__neg_risk",
                "_ClobClient__fee_rates",
            ):
                cache = self._raw_private_cache(
                    self._raw,
                    private_name,
                )
                if not isinstance(cache, dict):
                    continue

                for token_id in list(cache):
                    if str(token_id) not in keep:
                        cache.pop(token_id, None)
                        removed += 1

        return removed
