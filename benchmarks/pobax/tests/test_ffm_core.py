"""Equation and protocol tests for the source-audited FFM policy core."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.pobax.ffm_core import FFMPolicyCore
from benchmarks.pobax.policy_core import augment_policy_input


def _linear(params, prefix, values):
    return values @ params[f"{prefix}.kernel"] + params[f"{prefix}.bias"]


def _direct_step(core, params, state, policy_input, reset):
    """Minimal direct translation of published FFM Equations 12 and 15-18."""
    projected = _linear(params, "ffm.pre", policy_input)
    input_gate = jax.nn.sigmoid(_linear(params, "ffm.input_gate", policy_input))
    trace = projected * input_gate

    decay = jnp.minimum(params["ffm.decay"], -1e-6)
    exponent = decay[:, None] + 1j * params["ffm.frequency"][None, :]
    gamma = jnp.exp(exponent)
    retained = jnp.where(
        reset[:, None, None],
        jnp.zeros_like(state),
        state,
    )
    new_state = retained * gamma[None, :, :] + trace[:, :, None]

    memory_parts = jnp.concatenate(
        [jnp.real(new_state), jnp.imag(new_state)],
        axis=-1,
    )
    mixed = _linear(
        params,
        "ffm.mix",
        memory_parts.reshape((policy_input.shape[0], -1)),
    )
    mean = jnp.mean(mixed, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(mixed - mean), axis=-1, keepdims=True)
    normalized = (mixed - mean) * jax.lax.rsqrt(variance + 1e-5)
    output_gate = jax.nn.sigmoid(_linear(params, "ffm.output_gate", policy_input))
    skip = _linear(params, "ffm.skip", policy_input)
    features = normalized * output_gate + skip * (1.0 - output_gate)

    # Intentional harness adaptation: actor and critic share FFM features.
    logits = _linear(params, "actor", features)
    values = _linear(params, "critic", features)[..., 0]
    return new_state, logits, values


def _core() -> FFMPolicyCore:
    return FFMPolicyCore(
        input_dim=9,
        action_dim=4,
        hidden_size=7,
        memory_size=3,
        context_size=5,
        min_period=1.0,
        max_period=64.0,
    )


def test_initialization_matches_official_decay_and_frequency_schedule():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(0))

    np.testing.assert_allclose(
        params["ffm.decay"],
        jnp.linspace(-math.e, -1e-6, core.memory_size),
    )
    np.testing.assert_allclose(
        params["ffm.frequency"],
        2.0
        * jnp.pi
        / jnp.linspace(
            core.min_period,
            core.max_period,
            core.context_size,
        ),
    )


def test_step_matches_direct_translation_of_official_recurrence_and_cell():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(1))
    state = jax.random.normal(
        jax.random.PRNGKey(2),
        (3, core.memory_size, core.context_size),
    ) + 1j * jax.random.normal(
        jax.random.PRNGKey(3),
        (3, core.memory_size, core.context_size),
    )
    policy_input = jax.random.normal(jax.random.PRNGKey(4), (3, core.input_dim))
    reset = jnp.array([False, True, False])

    actual = core.step(params, state, policy_input, reset)
    expected = _direct_step(core, params, state, policy_input, reset)

    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_value, expected_value, rtol=1e-6, atol=1e-6)


def test_apply_sequence_matches_direct_recurrence_with_asynchronous_resets():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(5))
    initial_state = jax.random.normal(
        jax.random.PRNGKey(6),
        (3, core.memory_size, core.context_size),
    ) + 1j * jax.random.normal(
        jax.random.PRNGKey(7),
        (3, core.memory_size, core.context_size),
    )
    policy_inputs = jax.random.normal(
        jax.random.PRNGKey(8),
        (8, 3, core.input_dim),
    )
    resets = jnp.array(
        [
            [False, False, True],
            [False, True, False],
            [False, False, False],
            [True, False, False],
            [False, False, True],
            [False, False, False],
            [False, True, False],
            [False, False, False],
        ]
    )

    expected_state = initial_state
    expected_logits = []
    expected_values = []
    for policy_input, reset in zip(policy_inputs, resets, strict=True):
        expected_state, logits, values = _direct_step(
            core,
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
    # The parallel prefix tree reassociates complex multiply-add operations
    # relative to the direct recurrent loop. Fused GPU kernels can therefore
    # differ by roughly 1e-3 while implementing the same affine recurrence.
    np.testing.assert_allclose(
        actual_state,
        expected_state,
        rtol=5e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        actual_logits,
        jnp.stack(expected_logits),
        rtol=5e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        actual_values,
        jnp.stack(expected_values),
        rtol=5e-3,
        atol=1e-3,
    )


def test_reset_discards_only_selected_batch_states():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(9))
    state = jnp.full(
        (2, core.memory_size, core.context_size),
        2.0 + 3.0j,
        dtype=jnp.complex64,
    )
    zero_state = core.initial_state(2)
    policy_input = jax.random.normal(jax.random.PRNGKey(10), (2, core.input_dim))

    reset_state, reset_logits, reset_values = core.step(
        params,
        state,
        policy_input,
        jnp.array([True, False]),
    )
    zero_result = core.step(
        params,
        zero_state,
        policy_input,
        jnp.array([False, False]),
    )

    np.testing.assert_allclose(reset_state[0], zero_result[0][0])
    np.testing.assert_allclose(reset_logits[0], zero_result[1][0])
    np.testing.assert_allclose(reset_values[0], zero_result[2][0])
    assert not np.allclose(reset_state[1], zero_result[0][1])


def test_accepts_registered_augmented_policy_input():
    observation = jnp.arange(12, dtype=jnp.float32).reshape((3, 4))
    previous_action = jnp.array([1, 2, 0])
    previous_reward = jnp.array([0.5, -1.0, 2.0])
    reset = jnp.array([False, True, False])
    policy_input = augment_policy_input(
        observation,
        previous_action,
        previous_reward,
        reset,
        action_dim=3,
    )
    core = FFMPolicyCore(
        input_dim=policy_input.shape[-1],
        action_dim=3,
        hidden_size=6,
        memory_size=4,
        context_size=3,
    )

    state, logits, values = core.step(
        core.initialize(jax.random.PRNGKey(11)),
        core.initial_state(3),
        policy_input,
        reset,
    )

    assert state.shape == (3, 4, 3)
    assert state.dtype == jnp.complex64
    assert logits.shape == (3, 3)
    assert values.shape == (3,)


def test_step_and_sequence_are_jittable_and_differentiable():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(12))
    state = core.initial_state(2)
    policy_inputs = jax.random.normal(
        jax.random.PRNGKey(13),
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
    assert float(jnp.linalg.norm(gradients["ffm.decay"])) > 0.0
    assert float(jnp.linalg.norm(gradients["ffm.frequency"])) > 0.0


def test_parameter_count_includes_ffm_and_intentional_shared_heads():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(14))
    expected = (
        core.input_dim * core.memory_size
        + core.memory_size
        + core.input_dim * core.memory_size
        + core.memory_size
        + core.input_dim * core.hidden_size
        + core.hidden_size
        + core.input_dim * core.hidden_size
        + core.hidden_size
        + 2 * core.memory_size * core.context_size * core.hidden_size
        + core.hidden_size
        + core.memory_size
        + core.context_size
        + core.hidden_size * core.action_dim
        + core.action_dim
        + core.hidden_size
        + 1
    )

    assert "actor.kernel" in params
    assert "critic.kernel" in params
    assert core.count_parameters(params) == expected


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("input_dim", 0, "input_dim"),
        ("action_dim", 0, "action_dim"),
        ("hidden_size", 0, "hidden_size"),
        ("memory_size", 0, "memory_size"),
        ("context_size", 0, "context_size"),
        ("min_period", 0.0, "min_period"),
    ],
)
def test_rejects_invalid_configuration(field, value, message):
    kwargs = {
        "input_dim": 9,
        "action_dim": 4,
        "hidden_size": 7,
        "memory_size": 3,
        "context_size": 5,
        "min_period": 1.0,
        "max_period": 64.0,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        FFMPolicyCore(**kwargs)


def test_rejects_reversed_period_range():
    with pytest.raises(ValueError, match="max_period"):
        FFMPolicyCore(
            input_dim=9,
            action_dim=4,
            hidden_size=7,
            memory_size=3,
            context_size=5,
            min_period=10.0,
            max_period=1.0,
        )
