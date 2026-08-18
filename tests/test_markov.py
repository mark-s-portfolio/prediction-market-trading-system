"""
Regression tests for src.models.markov.

The suite covers the generic quantitative model only.  It intentionally contains
no admission thresholds, asset-specific setup knowledge, or trading verdicts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.models.markov import (
    FirstPassageResult,
    MarkovModelConfig,
    MarkovTransitionModel,
    PersistentMarkovView,
)


def make_model(
    *,
    states: int = 10,
    prior: float = 0.01,
    decay: float = 0.0,
    refresh_ticks: int = 1,
    simulation_cache: int = 8,
) -> MarkovTransitionModel:
    return MarkovTransitionModel(
        MarkovModelConfig(
            states_count=states,
            history_maxlen=32,
            transition_prior=prior,
            decay_rate_per_second=decay,
            probability_refresh_ticks=refresh_ticks,
            simulation_cache_max_entries=simulation_cache,
        )
    )


def teach_path(
    model: MarkovTransitionModel,
    prices,
    *,
    start_time: float = 100.0,
) -> None:
    for index, price in enumerate(prices):
        if index == 0:
            model.anchor(
                price,
                timestamp=start_time,
            )
        else:
            model.add_tick(
                price,
                timestamp=start_time + index,
            )


def test_config_rejects_invalid_numerical_domains() -> None:
    with pytest.raises(ValueError):
        MarkovModelConfig(states_count=1)

    with pytest.raises(ValueError):
        MarkovModelConfig(history_maxlen=1)

    with pytest.raises(ValueError):
        MarkovModelConfig(transition_prior=0.0)

    with pytest.raises(ValueError):
        MarkovModelConfig(decay_rate_per_second=-0.1)

    with pytest.raises(ValueError):
        MarkovModelConfig(probability_refresh_ticks=0)

    with pytest.raises(ValueError):
        MarkovModelConfig(simulation_cache_max_entries=0)


def test_price_state_mapping_covers_closed_unit_interval() -> None:
    model = make_model(states=10)

    assert model.state_for_price(0.0) == 0
    assert model.state_for_price(0.099) == 0
    assert model.state_for_price(0.10) == 1
    assert model.state_for_price(0.999) == 9
    assert model.state_for_price(1.0) == 9

    with pytest.raises(ValueError):
        model.state_for_price(-0.001)

    with pytest.raises(ValueError):
        model.state_for_price(1.001)

    with pytest.raises(ValueError):
        model.state_for_price(float("nan"))


def test_anchor_records_observation_without_transition() -> None:
    model = make_model()

    state = model.anchor(
        0.25,
        timestamp=100.0,
    )

    stats = model.statistics()

    assert state == model.state_for_price(0.25)
    assert stats.tick_count == 1
    assert stats.transition_count == 0
    assert stats.history_size == 1
    assert stats.last_update_time == pytest.approx(100.0)


def test_add_tick_after_anchor_records_exactly_one_transition() -> None:
    model = make_model(states=10, prior=0.01)

    source = model.anchor(
        0.20,
        timestamp=100.0,
    )
    target = model.add_tick(
        0.40,
        timestamp=101.0,
    )

    assert model.transition_count == 1
    assert model.tick_count == 2
    assert model.transitions[source, target] == pytest.approx(1.01)


def test_probability_rows_are_finite_and_normalized() -> None:
    model = make_model(states=8)

    teach_path(
        model,
        (0.10, 0.20, 0.30, 0.30, 0.40),
    )

    matrix = model.probability_matrix()

    assert matrix.shape == (8, 8)
    assert np.isfinite(matrix).all()
    assert (matrix >= 0.0).all()
    assert np.allclose(
        matrix.sum(axis=1),
        np.ones(8),
        atol=1e-12,
    )


def test_probability_matrix_copy_does_not_mutate_model_cache() -> None:
    model = make_model(states=6)
    matrix = model.probability_matrix(copy=True)

    matrix[0, :] = 0.0

    fresh = model.probability_matrix(copy=True)

    assert fresh[0, :].sum() == pytest.approx(1.0)


def test_probability_refresh_ticks_are_actually_respected() -> None:
    model = make_model(
        states=6,
        refresh_ticks=3,
    )

    model.anchor(
        0.20,
        timestamp=100.0,
    )
    first = model.probability_matrix()
    first_generation = model.statistics().probability_generation

    model.add_tick(
        0.40,
        timestamp=101.0,
    )
    second = model.probability_matrix()
    assert model.statistics().probability_generation == first_generation
    assert np.array_equal(second, first)

    model.add_tick(
        0.50,
        timestamp=102.0,
    )
    third = model.probability_matrix()
    assert model.statistics().probability_generation == first_generation
    assert np.array_equal(third, first)

    model.add_tick(
        0.50,
        timestamp=103.0,
    )
    refreshed = model.probability_matrix()

    assert (
        model.statistics().probability_generation
        == first_generation + 1
    )
    assert not np.array_equal(
        refreshed,
        first,
    )


def test_default_refresh_updates_after_each_tick() -> None:
    model = make_model(
        states=6,
        refresh_ticks=1,
    )

    model.anchor(
        0.20,
        timestamp=100.0,
    )
    model.probability_matrix()
    generation = model.statistics().probability_generation

    model.add_tick(
        0.40,
        timestamp=101.0,
    )
    model.probability_matrix()

    assert (
        model.statistics().probability_generation
        == generation + 1
    )


def test_time_decay_reduces_old_observation_but_preserves_prior_floor() -> None:
    model = make_model(
        states=10,
        prior=0.1,
        decay=math.log(2.0),
    )

    source = model.anchor(
        0.10,
        timestamp=100.0,
    )
    target = model.add_tick(
        0.20,
        timestamp=101.0,
    )

    assert model.transitions[source, target] == pytest.approx(1.1)

    model.add_tick(
        0.20,
        timestamp=102.0,
    )

    # One second at ln(2) decay halves the old observation mass.
    assert model.transitions[source, target] == pytest.approx(0.55)

    # Untouched pseudo-counts do not decay below the configured prior.
    assert model.transitions[9, 0] == pytest.approx(0.1)


def test_model_rejects_backward_timestamp_without_partial_mutation() -> None:
    model = make_model()

    model.anchor(
        0.20,
        timestamp=100.0,
    )
    before = model.statistics()
    before_history = tuple(model.history)
    before_transitions = model.transitions.copy()

    with pytest.raises(
        ValueError,
        match="timestamp cannot move backwards",
    ):
        model.add_tick(
            0.30,
            timestamp=99.0,
        )

    after = model.statistics()

    assert after == before
    assert tuple(model.history) == before_history
    assert np.array_equal(
        model.transitions,
        before_transitions,
    )


def test_first_passage_rejects_invalid_distance_domains() -> None:
    model = make_model()

    for invalid in (
        0.0,
        -0.1,
        1.01,
        float("inf"),
        float("nan"),
    ):
        with pytest.raises(ValueError):
            model.simulate_first_passage(
                current_price=0.50,
                target_distance=invalid,
                stop_distance=0.10,
            )

        with pytest.raises(ValueError):
            model.simulate_first_passage(
                current_price=0.50,
                target_distance=0.10,
                stop_distance=invalid,
            )


def test_first_passage_top_boundary_is_immediate_target() -> None:
    model = make_model(states=10)

    result = model.simulate_first_passage(
        current_price=1.0,
        target_distance=0.10,
        stop_distance=0.10,
        simulations=32,
        max_steps=4,
        seed=7,
    )

    assert result.current_state == 9
    assert result.target_state == 9
    assert result.target_probability == pytest.approx(1.0)
    assert result.representative_peak_state == 9


def test_first_passage_bottom_boundary_is_immediate_stop() -> None:
    model = make_model(states=10)

    result = model.simulate_first_passage(
        current_price=0.0,
        target_distance=0.10,
        stop_distance=0.10,
        simulations=32,
        max_steps=4,
        seed=7,
    )

    assert result.current_state == 0
    assert result.stop_state == 0
    assert result.target_probability == pytest.approx(0.0)
    assert result.representative_peak_state == 0


def test_seeded_first_passage_is_reproducible_and_memoized() -> None:
    model = make_model(states=12)

    teach_path(
        model,
        (
            0.40,
            0.50,
            0.60,
            0.50,
            0.60,
            0.70,
            0.60,
            0.70,
        ),
    )

    first = model.simulate_first_passage(
        current_price=0.60,
        target_distance=0.20,
        stop_distance=0.20,
        simulations=128,
        max_steps=20,
        seed=1234,
    )
    cache_size = len(
        model._simulation_result_cache
    )

    second = model.simulate_first_passage(
        current_price=0.60,
        target_distance=0.20,
        stop_distance=0.20,
        simulations=128,
        max_steps=20,
        seed=1234,
    )

    assert first == second
    assert first.seed == 1234
    assert cache_size == 1
    assert len(model._simulation_result_cache) == 1


def test_unseeded_first_passage_is_not_memoized() -> None:
    model = make_model()

    model.simulate_first_passage(
        current_price=0.50,
        target_distance=0.20,
        stop_distance=0.20,
        simulations=32,
        max_steps=5,
    )
    model.simulate_first_passage(
        current_price=0.50,
        target_distance=0.20,
        stop_distance=0.20,
        simulations=32,
        max_steps=5,
    )

    assert model._simulation_result_cache == {}


def test_explicit_rng_is_not_memoized_and_cannot_be_combined_with_seed() -> None:
    model = make_model()
    rng = np.random.default_rng(123)

    result = model.simulate_first_passage(
        current_price=0.50,
        target_distance=0.20,
        stop_distance=0.20,
        simulations=32,
        max_steps=5,
        rng=rng,
    )

    assert result.seed is None
    assert model._simulation_result_cache == {}

    with pytest.raises(ValueError):
        model.simulate_first_passage(
            current_price=0.50,
            target_distance=0.20,
            stop_distance=0.20,
            simulations=32,
            max_steps=5,
            seed=1,
            rng=np.random.default_rng(1),
        )


def test_seeded_cache_is_invalidated_when_probability_generation_refreshes() -> None:
    model = make_model(
        states=10,
        refresh_ticks=1,
    )
    teach_path(
        model,
        (0.30, 0.40, 0.50),
    )

    first = model.simulate_first_passage(
        current_price=0.40,
        target_distance=0.20,
        stop_distance=0.20,
        simulations=64,
        max_steps=10,
        seed=11,
    )
    assert len(model._simulation_result_cache) == 1

    model.add_tick(
        0.60,
        timestamp=200.0,
    )

    second = model.simulate_first_passage(
        current_price=0.40,
        target_distance=0.20,
        stop_distance=0.20,
        simulations=64,
        max_steps=10,
        seed=11,
    )

    assert (
        second.probability_generation
        > first.probability_generation
    )
    assert len(model._simulation_result_cache) == 1


def test_simulation_cache_respects_configured_bound() -> None:
    model = make_model(
        simulation_cache=2,
    )

    for seed in (1, 2, 3):
        model.simulate_first_passage(
            current_price=0.50,
            target_distance=0.20,
            stop_distance=0.20,
            simulations=16,
            max_steps=4,
            seed=seed,
        )

    assert len(model._simulation_result_cache) == 2


def test_one_step_up_probability_is_zero_at_top_state() -> None:
    model = make_model(states=10)

    assert model.row_probability_above_current(1.0) == pytest.approx(0.0)


def test_reset_clears_observational_and_simulation_state() -> None:
    model = make_model()

    teach_path(
        model,
        (0.20, 0.30, 0.40),
    )
    model.simulate_first_passage(
        current_price=0.30,
        target_distance=0.20,
        stop_distance=0.20,
        simulations=16,
        max_steps=5,
        seed=4,
    )

    assert model.tick_count > 0
    assert model.transition_count > 0
    assert model._simulation_result_cache

    model.reset()

    stats = model.statistics()

    assert stats.tick_count == 0
    assert stats.transition_count == 0
    assert stats.probability_generation == 0
    assert stats.history_size == 0
    assert stats.last_update_time is None
    assert model.last_state is None
    assert model._simulation_result_cache == {}
    assert np.allclose(
        model.transitions,
        model.config.transition_prior,
    )


def test_persistent_view_first_tick_does_not_create_cross_contract_transition() -> None:
    backing = make_model(states=10)

    backing.anchor(
        0.20,
        timestamp=100.0,
    )
    backing.add_tick(
        0.30,
        timestamp=101.0,
    )
    prior_transition_count = backing.transition_count

    view = PersistentMarkovView(
        backing,
        "contract-b",
        local_history_maxlen=16,
    )

    view.add_tick(
        0.80,
        timestamp=102.0,
    )

    assert backing.transition_count == prior_transition_count
    assert view.local_model.transition_count == 0
    assert view.local_tick_count == 1

    view.add_tick(
        0.90,
        timestamp=103.0,
    )

    assert backing.transition_count == prior_transition_count + 1
    assert view.local_model.transition_count == 1


def test_reset_local_preserves_persistent_memory_and_reanchors_next_contract() -> None:
    backing = make_model(states=10)
    view = PersistentMarkovView(
        backing,
        "contract-a",
        local_history_maxlen=16,
    )

    view.add_tick(
        0.40,
        timestamp=100.0,
    )
    view.add_tick(
        0.50,
        timestamp=101.0,
    )

    persistent_before = backing.transition_count
    persistent_ticks_before = backing.tick_count

    view.reset_local()

    assert view.local_tick_count == 0
    assert view.local_model.transition_count == 0
    assert tuple(view.local_history) == ()
    assert backing.transition_count == persistent_before
    assert backing.tick_count == persistent_ticks_before

    view.add_tick(
        0.20,
        timestamp=102.0,
    )

    # First tick after local reset anchors persistent state instead of creating a
    # transition from the previous contract's final price.
    assert backing.transition_count == persistent_before
    assert view.local_model.transition_count == 0


def test_persistent_and_local_probability_views_are_separate() -> None:
    backing = make_model(states=8)
    teach_path(
        backing,
        (0.10, 0.20, 0.30, 0.40),
    )

    view = PersistentMarkovView(
        backing,
        "new-contract",
        local_history_maxlen=16,
    )
    view.add_tick(
        0.70,
        timestamp=200.0,
    )
    view.add_tick(
        0.70,
        timestamp=201.0,
    )

    local = view.local_probability_matrix()
    persistent = view.persistent_probability_matrix()

    assert local.shape == persistent.shape
    assert not np.array_equal(
        local,
        persistent,
    )


def test_persistent_view_statistics_keep_local_and_backing_counts_distinct() -> None:
    backing = make_model()
    backing.anchor(
        0.20,
        timestamp=100.0,
    )

    view = PersistentMarkovView(
        backing,
        "contract-stats",
    )
    view.add_tick(
        0.60,
        timestamp=101.0,
    )
    view.add_tick(
        0.70,
        timestamp=102.0,
    )

    stats = view.statistics()

    assert stats.key == "contract-stats"
    assert stats.local_tick_count == 2
    assert stats.local_transition_count == 1
    assert stats.persistent_tick_count == backing.tick_count
    assert (
        stats.persistent_transition_count
        == backing.transition_count
    )
    assert stats.local_first_update_time == pytest.approx(101.0)
    assert stats.last_update_time == pytest.approx(102.0)


def test_invalid_persistent_tick_is_atomic() -> None:
    backing = make_model()
    view = PersistentMarkovView(
        backing,
        "atomic-view",
    )

    backing_before = backing.statistics()
    local_before = view.local_model.statistics()

    with pytest.raises(ValueError):
        view.add_tick(
            1.5,
            timestamp=100.0,
        )

    assert view.local_tick_count == 0
    assert tuple(view.local_history) == ()
    assert view.local_first_update_time is None
    assert view.last_update_time is None
    assert view.local_model.statistics() == local_before
    assert backing.statistics() == backing_before


def test_backward_persistent_tick_is_atomic_across_both_models() -> None:
    backing = make_model()
    view = PersistentMarkovView(
        backing,
        "time-view",
    )

    view.add_tick(
        0.40,
        timestamp=100.0,
    )

    backing_before = backing.statistics()
    local_before = view.local_model.statistics()
    history_before = tuple(view.local_history)

    with pytest.raises(
        ValueError,
        match="timestamp cannot move backwards",
    ):
        view.add_tick(
            0.50,
            timestamp=99.0,
        )

    assert backing.statistics() == backing_before
    assert view.local_model.statistics() == local_before
    assert tuple(view.local_history) == history_before
    assert view.local_tick_count == 1
    assert view.last_update_time == pytest.approx(100.0)


def test_first_passage_result_validates_probability_range() -> None:
    with pytest.raises(ValueError):
        FirstPassageResult(
            target_probability=1.1,
            representative_peak_state=1,
            one_step_up_probability=0.5,
            current_state=1,
            target_state=2,
            stop_state=0,
            simulations=10,
            max_steps=5,
            probability_generation=1,
            seed=1,
        )

    with pytest.raises(ValueError):
        FirstPassageResult(
            target_probability=0.5,
            representative_peak_state=1,
            one_step_up_probability=-0.1,
            current_state=1,
            target_state=2,
            stop_state=0,
            simulations=10,
            max_steps=5,
            probability_generation=1,
            seed=1,
        )
