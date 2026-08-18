"""
Generic Markov transition models for the public portfolio edition.

This module demonstrates reusable quantitative infrastructure only:
- discretized price-state transition learning
- optional time decay
- cached probability-matrix generations
- vectorized first-passage Monte Carlo simulation
- deterministic/reproducible simulation when a seed is supplied
- separation of persistent diagnostic memory from contract-local evidence
- rotation-safe anchoring without a cross-contract transition

It intentionally contains no trading admission thresholds, asset-specific setup
knowledge, position sizing, or allow/block decisions.  Model outputs are data;
policy consumers decide what, if anything, to do with them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
import time
from typing import Deque, Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class MarkovModelConfig:
    """Numerical configuration for one transition model.

    Defaults are portfolio/demo values rather than production trading parameters.
    """

    states_count: int = 100
    history_maxlen: int = 256
    transition_prior: float = 1e-3
    decay_rate_per_second: float = 0.0
    probability_refresh_ticks: int = 1
    simulation_cache_max_entries: int = 64

    def __post_init__(self) -> None:
        states = int(self.states_count)
        history = int(self.history_maxlen)
        prior = float(self.transition_prior)
        decay = float(self.decay_rate_per_second)
        refresh = int(self.probability_refresh_ticks)
        cache_max = int(self.simulation_cache_max_entries)

        if states < 2:
            raise ValueError("states_count must be at least 2")
        if history < 2:
            raise ValueError("history_maxlen must be at least 2")
        if not math.isfinite(prior) or prior <= 0.0:
            raise ValueError("transition_prior must be positive")
        if not math.isfinite(decay) or decay < 0.0:
            raise ValueError("decay_rate_per_second must be non-negative")
        if refresh < 1:
            raise ValueError("probability_refresh_ticks must be positive")
        if cache_max < 1:
            raise ValueError("simulation_cache_max_entries must be positive")

        object.__setattr__(self, "states_count", states)
        object.__setattr__(self, "history_maxlen", history)
        object.__setattr__(self, "transition_prior", prior)
        object.__setattr__(self, "decay_rate_per_second", decay)
        object.__setattr__(self, "probability_refresh_ticks", refresh)
        object.__setattr__(self, "simulation_cache_max_entries", cache_max)


@dataclass(frozen=True, slots=True)
class ModelStatistics:
    tick_count: int
    transition_count: int
    probability_generation: int
    history_size: int
    last_update_time: Optional[float]


@dataclass(frozen=True, slots=True)
class FirstPassageResult:
    """Result of a target-before-stop Monte Carlo simulation."""

    target_probability: float
    representative_peak_state: int
    one_step_up_probability: float
    current_state: int
    target_state: int
    stop_state: int
    simulations: int
    max_steps: int
    probability_generation: int
    seed: Optional[int]

    def __post_init__(self) -> None:
        probability = float(self.target_probability)
        one_step = float(self.one_step_up_probability)

        if not 0.0 <= probability <= 1.0:
            raise ValueError("target_probability must be in [0, 1]")
        if not 0.0 <= one_step <= 1.0:
            raise ValueError("one_step_up_probability must be in [0, 1]")


class MarkovTransitionModel:
    """Discretized first-order Markov transition learner.

    Prices are expected in the prediction-market interval [0, 1].  The model
    learns transition counts only.  It has no knowledge of entries, hedges,
    profitability, or execution policy.
    """

    def __init__(self, config: MarkovModelConfig = MarkovModelConfig()) -> None:
        self.config = config
        self.states_count = config.states_count

        self._gate = threading.RLock()
        self.transitions = np.full(
            (self.states_count, self.states_count),
            config.transition_prior,
            dtype=np.float64,
        )

        self.last_state: Optional[int] = None
        self.last_update_time: Optional[float] = None
        self.history: Deque[float] = deque(maxlen=config.history_maxlen)

        self._tick_count = 0
        self._transition_count = 0
        self._probability_generation = 0

        self._prob_cache: Optional[np.ndarray] = None
        self._prob_cache_tick = -1

        self._simulation_cdf_cache: Optional[np.ndarray] = None
        self._simulation_cdf_generation = -1
        self._simulation_result_cache: dict[tuple, FirstPassageResult] = {}

    # ------------------------------------------------------------------
    # State mapping / observations
    # ------------------------------------------------------------------

    def state_for_price(self, price: float) -> int:
        price = float(price)
        if not math.isfinite(price) or not 0.0 <= price <= 1.0:
            raise ValueError("price must be finite and in [0, 1]")

        return min(
            max(int(price * self.states_count), 0),
            self.states_count - 1,
        )

    def _invalidate_probability_generation(self) -> None:
        self._prob_cache = None
        self._simulation_cdf_cache = None
        self._simulation_cdf_generation = -1
        self._simulation_result_cache.clear()

    def anchor(self, price: float, *, timestamp: Optional[float] = None) -> int:
        """Set current state without recording a transition.

        This is useful at a market/contract boundary: the first observation of a
        new contract can establish the current state without fabricating a
        transition from the previous contract's final state.
        """

        state = self.state_for_price(price)
        observed = float(timestamp if timestamp is not None else time.time())

        if not math.isfinite(observed) or observed <= 0.0:
            raise ValueError("timestamp must be positive")

        with self._gate:
            self.history.append(float(price))
            self.last_state = state
            self.last_update_time = observed
            self._tick_count += 1
            self._invalidate_probability_generation()

        return state

    def add_tick(self, price: float, *, timestamp: Optional[float] = None) -> int:
        """Add one observation and, when possible, one real transition."""

        state = self.state_for_price(price)
        observed = float(timestamp if timestamp is not None else time.time())

        if not math.isfinite(observed) or observed <= 0.0:
            raise ValueError("timestamp must be positive")

        with self._gate:
            self.history.append(float(price))

            if self.last_state is not None and self.last_update_time is not None:
                dt = max(0.0, observed - self.last_update_time)
                decay_rate = self.config.decay_rate_per_second

                if decay_rate > 0.0 and dt > 0.0:
                    decay = math.exp(-decay_rate * dt)
                    self.transitions *= decay
                    np.maximum(
                        self.transitions,
                        self.config.transition_prior,
                        out=self.transitions,
                    )

                self.transitions[self.last_state, state] += 1.0
                self._transition_count += 1

            self.last_state = state
            self.last_update_time = observed
            self._tick_count += 1
            self._invalidate_probability_generation()

        return state

    # ------------------------------------------------------------------
    # Probability matrix
    # ------------------------------------------------------------------

    def probability_matrix(self, *, copy: bool = True) -> np.ndarray:
        """Return the normalized transition matrix for the current generation."""

        with self._gate:
            refresh_due = bool(
                self._prob_cache is None
                or self._prob_cache_tick < 0
                or self._tick_count - self._prob_cache_tick
                >= self.config.probability_refresh_ticks
            )

            if refresh_due:
                row_sums = self.transitions.sum(axis=1, keepdims=True)
                normalized = self.transitions / np.where(
                    (row_sums <= 0.0) | ~np.isfinite(row_sums),
                    1.0,
                    row_sums,
                )

                # Numerical defense: ensure every row is a valid probability row.
                normalized = np.nan_to_num(
                    normalized,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                normalized_sums = normalized.sum(axis=1, keepdims=True)
                normalized = normalized / np.where(
                    normalized_sums <= 0.0,
                    1.0,
                    normalized_sums,
                )

                self._prob_cache = normalized
                self._prob_cache_tick = self._tick_count
                self._probability_generation += 1

                self._simulation_cdf_cache = None
                self._simulation_cdf_generation = -1
                self._simulation_result_cache.clear()

            assert self._prob_cache is not None
            return self._prob_cache.copy() if copy else self._prob_cache

    # Backwards-readable public alias without preserving private method naming.
    get_probability_matrix = probability_matrix

    def row_probability_above_current(self, current_price: float) -> float:
        """One-step probability mass strictly above the current state."""

        matrix = self.probability_matrix(copy=False)
        current_state = self.state_for_price(current_price)
        next_higher = min(current_state + 1, self.states_count)

        if next_higher >= self.states_count:
            return 0.0

        return float(matrix[current_state, next_higher:].sum())

    # ------------------------------------------------------------------
    # Vectorized first-passage simulation
    # ------------------------------------------------------------------

    def _simulation_cdf(self) -> tuple[np.ndarray, int]:
        matrix = self.probability_matrix(copy=False)

        with self._gate:
            generation = self._probability_generation

            if (
                self._simulation_cdf_cache is None
                or self._simulation_cdf_generation != generation
            ):
                cdf = np.cumsum(matrix, axis=1)
                cdf[:, -1] = 1.0
                self._simulation_cdf_cache = cdf
                self._simulation_cdf_generation = generation

            return self._simulation_cdf_cache, generation

    def simulate_first_passage(
        self,
        *,
        current_price: float,
        target_distance: float,
        stop_distance: float,
        simulations: int = 256,
        max_steps: int = 64,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> FirstPassageResult:
        """Estimate P(target state is reached before stop state).

        `target_distance` and `stop_distance` are normalized price distances in
        [0, 1], not policy thresholds.  Supplying `seed` makes a call reproducible.
        Supplying an RNG lets a caller manage a longer deterministic stream.
        """

        current_price = float(current_price)
        target_distance = float(target_distance)
        stop_distance = float(stop_distance)
        simulations = int(simulations)
        max_steps = int(max_steps)

        if not math.isfinite(target_distance) or target_distance <= 0.0:
            raise ValueError("target_distance must be positive")
        if not math.isfinite(stop_distance) or stop_distance <= 0.0:
            raise ValueError("stop_distance must be positive")
        if simulations < 1:
            raise ValueError("simulations must be positive")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if rng is not None and seed is not None:
            raise ValueError("provide either seed or rng, not both")

        current_state = self.state_for_price(current_price)
        target_steps = max(1, int(round(target_distance * self.states_count)))
        stop_steps = max(1, int(round(stop_distance * self.states_count)))

        target_state = min(
            current_state + target_steps,
            self.states_count - 1,
        )
        stop_state = max(current_state - stop_steps, 0)

        cdf, generation = self._simulation_cdf()
        one_step_up = self.row_probability_above_current(current_price)

        # Only seeded calls are memoized.  Caching an unseeded Monte Carlo result
        # would make a stochastic draw look deterministic for the rest of one model
        # generation and obscures uncertainty during analysis.
        cache_key = None
        if rng is None and seed is not None:
            cache_key = (
                generation,
                current_state,
                target_state,
                stop_state,
                simulations,
                max_steps,
                int(seed),
            )

            with self._gate:
                cached = self._simulation_result_cache.get(cache_key)
                if cached is not None:
                    return cached

        generator = rng if rng is not None else np.random.default_rng(seed)

        path_states = np.full(simulations, current_state, dtype=np.int32)
        peaks = path_states.copy()
        active = np.ones(simulations, dtype=bool)
        target_hit = np.zeros(simulations, dtype=bool)

        for _ in range(max_steps):
            active_idx = np.flatnonzero(active)
            if active_idx.size == 0:
                break

            current_rows = cdf[path_states[active_idx]]
            draws = generator.random(active_idx.size)

            # Row-wise inverse CDF, vectorized across active paths.
            next_states = np.sum(
                current_rows < draws[:, None],
                axis=1,
            ).astype(np.int32, copy=False)

            path_states[active_idx] = next_states
            peaks[active_idx] = np.maximum(peaks[active_idx], next_states)

            hit_target = next_states >= target_state
            hit_stop = next_states <= stop_state

            if np.any(hit_target):
                target_hit[active_idx[hit_target]] = True

            stopped = hit_target | hit_stop
            if np.any(stopped):
                active[active_idx[stopped]] = False

        target_probability = (
            float(np.count_nonzero(target_hit)) / float(simulations)
        )

        if np.any(target_hit):
            representative_peak = int(np.median(peaks[target_hit]))
        elif peaks.size:
            representative_peak = int(np.median(peaks))
        else:  # pragma: no cover - simulations >= 1
            representative_peak = target_state

        representative_peak = min(
            max(representative_peak, current_state),
            self.states_count - 1,
        )

        result = FirstPassageResult(
            target_probability=target_probability,
            representative_peak_state=representative_peak,
            one_step_up_probability=one_step_up,
            current_state=current_state,
            target_state=target_state,
            stop_state=stop_state,
            simulations=simulations,
            max_steps=max_steps,
            probability_generation=generation,
            seed=seed,
        )

        if cache_key is not None:
            with self._gate:
                cap = self.config.simulation_cache_max_entries
                if len(self._simulation_result_cache) >= cap:
                    try:
                        self._simulation_result_cache.pop(
                            next(iter(self._simulation_result_cache))
                        )
                    except Exception:
                        self._simulation_result_cache.clear()

                self._simulation_result_cache[cache_key] = result

        return result

    # ------------------------------------------------------------------
    # Introspection / lifecycle
    # ------------------------------------------------------------------

    def statistics(self) -> ModelStatistics:
        with self._gate:
            return ModelStatistics(
                tick_count=self._tick_count,
                transition_count=self._transition_count,
                probability_generation=self._probability_generation,
                history_size=len(self.history),
                last_update_time=self.last_update_time,
            )

    @property
    def transition_count(self) -> int:
        with self._gate:
            return self._transition_count

    @property
    def tick_count(self) -> int:
        with self._gate:
            return self._tick_count

    def reset(self) -> None:
        with self._gate:
            self.transitions.fill(self.config.transition_prior)
            self.last_state = None
            self.last_update_time = None
            self.history.clear()

            self._tick_count = 0
            self._transition_count = 0
            self._probability_generation = 0

            self._prob_cache = None
            self._prob_cache_tick = -1
            self._simulation_cdf_cache = None
            self._simulation_cdf_generation = -1
            self._simulation_result_cache.clear()


@dataclass(frozen=True, slots=True)
class MarkovViewStatistics:
    key: str
    local_tick_count: int
    local_transition_count: int
    persistent_tick_count: int
    persistent_transition_count: int
    local_first_update_time: Optional[float]
    last_update_time: Optional[float]


class PersistentMarkovView:
    """Contract-local view over a longer-lived diagnostic transition model.

    The backing model can retain historical transition structure across rotations.
    The local model starts fresh for each contract/window and is the only model
    exposed by the `local_*` methods.  This prevents historical maturity from being
    mistaken for current-contract evidence.

    No method in this class authorizes an order or produces an admission decision.
    """

    def __init__(
        self,
        backing: MarkovTransitionModel,
        key: str,
        *,
        local_history_maxlen: int = 256,
    ) -> None:
        if not isinstance(backing, MarkovTransitionModel):
            raise TypeError("backing must be a MarkovTransitionModel")

        key = str(key or "").strip()
        if not key:
            raise ValueError("key is required")

        self.backing = backing
        self.key = key

        local_config = MarkovModelConfig(
            states_count=backing.config.states_count,
            history_maxlen=int(local_history_maxlen),
            transition_prior=backing.config.transition_prior,
            decay_rate_per_second=backing.config.decay_rate_per_second,
            probability_refresh_ticks=backing.config.probability_refresh_ticks,
            simulation_cache_max_entries=(
                backing.config.simulation_cache_max_entries
            ),
        )
        self.local_model = MarkovTransitionModel(local_config)

        self.local_history: Deque[float] = deque(
            maxlen=int(local_history_maxlen)
        )
        self.local_tick_count = 0
        self.local_first_update_time: Optional[float] = None
        self.last_update_time: Optional[float] = None

    @property
    def states_count(self) -> int:
        return self.backing.states_count

    def add_tick(
        self,
        price: float,
        *,
        timestamp: Optional[float] = None,
    ) -> None:
        observed = float(timestamp if timestamp is not None else time.time())
        price = float(price)

        if self.local_tick_count == 0:
            self.local_first_update_time = observed

        self.local_history.append(price)
        self.local_model.add_tick(price, timestamp=observed)

        # The first observation in a new view anchors persistent state without
        # creating a transition from an expired/previous contract.
        if self.local_tick_count == 0:
            self.backing.anchor(price, timestamp=observed)
        else:
            self.backing.add_tick(price, timestamp=observed)

        self.local_tick_count += 1
        self.last_update_time = observed

    def local_probability_matrix(self, *, copy: bool = True) -> np.ndarray:
        return self.local_model.probability_matrix(copy=copy)

    def persistent_probability_matrix(self, *, copy: bool = True) -> np.ndarray:
        """Long-lived diagnostic prior; not current-contract evidence by itself."""
        return self.backing.probability_matrix(copy=copy)

    def simulate_local_first_passage(self, **kwargs) -> FirstPassageResult:
        return self.local_model.simulate_first_passage(**kwargs)

    def simulate_persistent_first_passage(self, **kwargs) -> FirstPassageResult:
        """Simulate the historical diagnostic prior explicitly."""
        return self.backing.simulate_first_passage(**kwargs)

    def statistics(self) -> MarkovViewStatistics:
        local = self.local_model.statistics()
        persistent = self.backing.statistics()

        return MarkovViewStatistics(
            key=self.key,
            local_tick_count=self.local_tick_count,
            local_transition_count=local.transition_count,
            persistent_tick_count=persistent.tick_count,
            persistent_transition_count=persistent.transition_count,
            local_first_update_time=self.local_first_update_time,
            last_update_time=self.last_update_time,
        )

    def reset_local(self) -> None:
        """Reset only current-contract evidence, preserving persistent memory."""

        self.local_history.clear()
        self.local_tick_count = 0
        self.local_first_update_time = None
        self.last_update_time = None
        self.local_model.reset()
