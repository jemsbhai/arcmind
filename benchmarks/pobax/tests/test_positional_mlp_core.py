"""Tests for the source-audited POPGym positional MLP policy core."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.pobax.positional_mlp_core import (
    POPGYM_POSITIONAL_MLP_REFERENCE,
    PositionalMLPPolicyCore,
    match_positional_mlp_hidden_size,
    sinusoidal_position_encoding,
)


def _core() -> PositionalMLPPolicyCore:
    return PositionalMLPPolicyCore(
        input_dim=6,
        action_dim=4,
        hidden_size=7,
        max_length=31,
    )


def _numpy_encoding(positions, hidden_size, max_length):
    output = np.zeros((*positions.shape, hidden_size), dtype=np.float32)
    for output_index in np.ndindex(positions.shape):
        position = float(positions[output_index])
        for coordinate in range(0, hidden_size, 2):
            frequency = math.exp(-math.log(max_length) * coordinate / hidden_size)
            output[output_index + (coordinate,)] = math.sin(position * frequency)
            if coordinate + 1 < hidden_size:
                output[output_index + (coordinate + 1,)] = math.cos(position * frequency)
    return output


def test_sinusoidal_encoding_matches_numpy_source_equation_fixture():
    positions = np.array([[0, 1, 7], [2, 5, 10]], dtype=np.int32)
    actual = sinusoidal_position_encoding(
        jnp.asarray(positions),
        hidden_size=5,
        max_length=11,
    )
    expected = _numpy_encoding(positions, hidden_size=5, max_length=11)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual[0, 0, 0::2], 0.0)
    np.testing.assert_array_equal(actual[0, 0, 1::2], 1.0)


def test_embedding_alpha_starts_at_half_and_clips_to_closed_unit_interval():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(0))
    policy_input = jax.random.normal(
        jax.random.PRNGKey(1),
        (3, core.input_dim),
    )
    positions = jnp.array([0, 3, 9], dtype=jnp.int32)

    projected = policy_input @ params["feature_map.kernel"] + params["feature_map.bias"]
    mean = jnp.mean(projected, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(projected - mean), axis=-1, keepdims=True)
    projected = (projected - mean) * jax.lax.rsqrt(variance + 1e-5)
    encoding = sinusoidal_position_encoding(
        positions,
        hidden_size=core.hidden_size,
        max_length=core.max_length,
    )

    assert float(params["embedding.alpha"]) == 0.5
    np.testing.assert_allclose(
        core.encode_input(params, policy_input, positions),
        0.5 * projected + 0.5 * encoding,
        rtol=1e-6,
        atol=1e-6,
    )

    above_one = dict(params)
    above_one["embedding.alpha"] = jnp.asarray(7.0)
    below_zero = dict(params)
    below_zero["embedding.alpha"] = jnp.asarray(-4.0)
    np.testing.assert_allclose(
        core.encode_input(above_one, policy_input, positions),
        encoding,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        core.encode_input(below_zero, policy_input, positions),
        projected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_apply_sequence_matches_repeated_step_with_asynchronous_resets():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(2))
    initial_state = jnp.array([4, 1, 8], dtype=jnp.int32)
    policy_inputs = jax.random.normal(
        jax.random.PRNGKey(3),
        (7, 3, core.input_dim),
    )
    resets = jnp.array(
        [
            [False, True, False],
            [False, False, False],
            [True, False, False],
            [False, False, True],
            [False, True, False],
            [False, False, False],
            [True, False, False],
        ]
    )

    expected_state = initial_state
    expected_logits = []
    expected_values = []
    for policy_input, reset in zip(policy_inputs, resets, strict=True):
        expected_state, logits, values = core.step(
            params,
            expected_state,
            policy_input,
            reset,
        )
        expected_logits.append(logits)
        expected_values.append(values)

    actual_state, actual_logits, actual_values = core.apply_sequence(
        params,
        initial_state,
        policy_inputs,
        resets,
    )

    np.testing.assert_array_equal(actual_state, expected_state)
    np.testing.assert_allclose(
        actual_logits,
        jnp.stack(expected_logits),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        actual_values,
        jnp.stack(expected_values),
        rtol=1e-6,
        atol=1e-7,
    )


def test_reset_is_independent_and_applied_before_current_output():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(4))
    state = jnp.array([12, 12], dtype=jnp.int32)
    policy_input = jnp.repeat(
        jax.random.normal(jax.random.PRNGKey(5), (1, core.input_dim)),
        2,
        axis=0,
    )

    new_state, logits, values = core.step(
        params,
        state,
        policy_input,
        jnp.array([True, False]),
    )
    fresh_state, fresh_logits, fresh_values = core.step(
        params,
        core.initial_state(1),
        policy_input[:1],
        jnp.array([False]),
    )

    np.testing.assert_array_equal(new_state, jnp.array([1, 13], dtype=jnp.int32))
    np.testing.assert_array_equal(new_state[:1], fresh_state)
    np.testing.assert_allclose(logits[:1], fresh_logits, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(values[:1], fresh_values, rtol=1e-6, atol=1e-7)
    assert not np.allclose(logits[0], logits[1])


def test_identical_policy_input_is_distinguished_by_position():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(6))
    policy_input = jnp.repeat(
        jax.random.normal(jax.random.PRNGKey(7), (1, core.input_dim)),
        2,
        axis=0,
    )
    encoded = core.encode_input(
        params,
        policy_input,
        jnp.array([0, 1], dtype=jnp.int32),
    )

    assert not np.allclose(encoded[0], encoded[1])


def test_state_contains_only_int32_positions_and_no_observation_history():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(8))
    first_state = core.initial_state(3)
    second_state = core.initial_state(3)
    first_inputs = jnp.zeros((3, core.input_dim))
    second_inputs = jnp.full((3, core.input_dim), 100.0)
    reset = jnp.array([False, True, False])

    first_state, _, _ = core.step(params, first_state, first_inputs, reset)
    second_state, _, _ = core.step(params, second_state, second_inputs, reset)

    assert first_state.shape == (3,)
    assert first_state.dtype == jnp.int32
    np.testing.assert_array_equal(first_state, second_state)
    assert first_state.nbytes == 3 * np.dtype(np.int32).itemsize


def test_step_sequence_jit_and_gradients():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(9))
    state = core.initial_state(2)
    policy_inputs = jax.random.normal(
        jax.random.PRNGKey(10),
        (5, 2, core.input_dim),
    )
    resets = jnp.array(
        [
            [True, True],
            [False, False],
            [False, True],
            [False, False],
            [True, False],
        ]
    )

    step_result = jax.jit(core.step)(
        params,
        state,
        policy_inputs[0],
        resets[0],
    )
    sequence_result = jax.jit(core.apply_sequence)(
        params,
        state,
        policy_inputs,
        resets,
    )
    assert step_result[0].dtype == jnp.int32
    assert step_result[1].shape == (2, core.action_dim)
    assert sequence_result[1].shape == (5, 2, core.action_dim)

    def loss(candidate_params):
        _, logits, values = core.apply_sequence(
            candidate_params,
            state,
            policy_inputs,
            resets,
        )
        return jnp.mean(jnp.square(logits)) + jnp.mean(jnp.square(values))

    gradients = jax.jit(jax.grad(loss))(params)
    assert jnp.isfinite(loss(params))
    assert all(bool(jnp.all(jnp.isfinite(gradient))) for gradient in jax.tree.leaves(gradients))
    assert float(jnp.abs(gradients["embedding.alpha"])) > 0.0


def test_parameter_count_includes_alpha_and_both_shared_heads():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(11))
    expected = (
        core.input_dim * core.hidden_size
        + core.hidden_size
        + 1
        + core.hidden_size * core.hidden_size
        + core.hidden_size
        + core.hidden_size * core.hidden_size
        + core.hidden_size
        + core.hidden_size * core.action_dim
        + core.action_dim
        + core.hidden_size
        + 1
    )

    assert core.count_parameters(params) == expected


@pytest.mark.parametrize(
    ("input_dim", "action_dim", "target_parameters"),
    [
        (7, 3, 10_000),
        (9, 4, 28_717),
        (128, 18, 400_000),
    ],
)
def test_width_matcher_selects_globally_closest_positive_width(
    input_dim,
    action_dim,
    target_parameters,
):
    width = match_positional_mlp_hidden_size(
        target_parameters=target_parameters,
        input_dim=input_dim,
        action_dim=action_dim,
    )

    def count(candidate_width):
        candidate = PositionalMLPPolicyCore(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_size=candidate_width,
            max_length=1_000,
        )
        params = candidate.initialize(jax.random.PRNGKey(12))
        return candidate.count_parameters(params)

    selected_error = abs(count(width) - target_parameters)
    if width > 1:
        assert selected_error <= abs(count(width - 1) - target_parameters)
    assert selected_error <= abs(count(width + 1) - target_parameters)


def test_reference_metadata_pins_audited_popgym_commit():
    assert POPGYM_POSITIONAL_MLP_REFERENCE == {
        "repository": "https://github.com/proroklab/popgym",
        "audited_commit": "410d5aa626dae8024f498354d8781a0d1870c399",
        "relationship": "shared-input and shared-head policy adaptation",
    }


@pytest.mark.parametrize(
    "field",
    ["target_parameters", "input_dim", "action_dim", "maximum_width"],
)
def test_width_matcher_rejects_nonpositive_arguments(field):
    arguments = {
        "target_parameters": 10_000,
        "input_dim": 7,
        "action_dim": 3,
        "maximum_width": 64,
    }
    arguments[field] = 0

    with pytest.raises(ValueError, match=field):
        match_positional_mlp_hidden_size(**arguments)


@pytest.mark.parametrize(
    "field",
    ["input_dim", "action_dim", "hidden_size", "max_length"],
)
def test_core_rejects_nonpositive_dimensions(field):
    arguments = {
        "input_dim": 6,
        "action_dim": 4,
        "hidden_size": 7,
        "max_length": 31,
    }
    arguments[field] = 0

    with pytest.raises(ValueError, match=field):
        PositionalMLPPolicyCore(**arguments)


@pytest.mark.parametrize(
    ("hidden_size", "max_length", "message"),
    [(0, 10, "hidden_size"), (4, 0, "max_length")],
)
def test_encoding_rejects_invalid_dimensions(hidden_size, max_length, message):
    with pytest.raises(ValueError, match=message):
        sinusoidal_position_encoding(
            jnp.array([0]),
            hidden_size=hidden_size,
            max_length=max_length,
        )


def test_initial_state_rejects_empty_batch():
    with pytest.raises(ValueError, match="batch_size"):
        _core().initial_state(0)
