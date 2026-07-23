"""Equation, addressing, and protocol tests for the audited SHM JAX core."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.pobax.baseline_cores import MemorylessMLPPolicyCore
from benchmarks.pobax.policy_core import augment_policy_input
from benchmarks.pobax.shared_ppo import (
    PPOConfig,
    Rollout,
    SharedPPO,
    categorical_log_probability,
)
from benchmarks.pobax.shm_core import (
    POPGYM_SHM_MEMORY_SIZE,
    SHM_ADDRESS_ROWS,
    SHM_SOURCE_COMMIT,
    SHMPolicyCore,
    match_shm_hidden_size,
)


def _core(
    *,
    address_mode: str = "paper_uniform",
) -> SHMPolicyCore:
    return SHMPolicyCore(
        input_dim=5,
        action_dim=3,
        hidden_size=6,
        memory_size=4,
        address_mode=address_mode,
    )


def _numpy_linear(params, prefix, values, *, bias):
    result = values @ np.asarray(params[f"{prefix}.kernel"])
    if bias:
        result = result + np.asarray(params[f"{prefix}.bias"])
    return result


def _numpy_official_step(
    params,
    state,
    policy_input,
    reset,
    addresses,
):
    """Independent NumPy translation of the pinned POPGym SHM cell."""
    state = np.asarray(state)
    policy_input = np.asarray(policy_input)
    reset = np.asarray(reset)
    addresses = np.asarray(addresses)

    mean = policy_input.mean(axis=-1, keepdims=True)
    variance = np.square(policy_input - mean).mean(axis=-1, keepdims=True)
    normalized = (policy_input - mean) / np.sqrt(variance + 1e-5)
    normalized = normalized * np.asarray(params["shm.norm.scale"]) + np.asarray(
        params["shm.norm.bias"]
    )

    key = np.maximum(
        _numpy_linear(params, "shm.key", normalized, bias=False),
        0.0,
    )
    query = np.maximum(
        _numpy_linear(params, "shm.query", normalized, bias=False),
        0.0,
    )
    key = key / (1e-5 + key.sum(axis=-1, keepdims=True))
    query = query / (1e-5 + query.sum(axis=-1, keepdims=True))
    value = _numpy_linear(params, "shm.value", normalized, bias=False)
    eta = 1.0 / (1.0 + np.exp(-_numpy_linear(params, "shm.eta", normalized, bias=False)))
    calibration_value = _numpy_linear(
        params,
        "shm.calibration",
        normalized,
        bias=False,
    )
    theta = np.asarray(params["shm.theta"])[addresses]
    calibration = 1.0 + np.tanh(theta[..., :, None] * calibration_value[..., None, :])

    retained = np.where(reset[..., None, None], np.zeros_like(state), state)
    write = (eta * value)[..., :, None] * key[..., None, :]
    new_state = retained * calibration + write
    memory_read = np.einsum("...ij,...j->...i", new_state, query)
    shortcut = _numpy_linear(
        params,
        "shm.shortcut",
        normalized,
        bias=True,
    )
    features = _numpy_linear(
        params,
        "shm.out",
        memory_read + shortcut,
        bias=True,
    )
    logits = _numpy_linear(params, "actor", features, bias=True)
    values = _numpy_linear(params, "critic", features, bias=True)[..., 0]
    return new_state, logits, values


def test_source_pin_and_scientific_defaults_are_explicit():
    core = SHMPolicyCore(input_dim=7, action_dim=2, hidden_size=9)

    assert SHM_SOURCE_COMMIT == "40d73d44936e47a29e2c76a481d93c434b857ea1"
    assert core.address_mode == "paper_uniform"
    assert core.memory_size == POPGYM_SHM_MEMORY_SIZE == 16
    assert SHM_ADDRESS_ROWS == 128


def test_initialization_has_pinned_parameterization_and_source_distributions():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(0))

    assert params["shm.key.kernel"].shape == (core.input_dim, core.memory_size)
    assert params["shm.query.kernel"].shape == (core.input_dim, core.memory_size)
    assert params["shm.value.kernel"].shape == (core.input_dim, core.memory_size)
    assert params["shm.calibration.kernel"].shape == (
        core.input_dim,
        core.memory_size,
    )
    assert params["shm.eta.kernel"].shape == (core.input_dim, 1)
    assert params["shm.theta"].shape == (SHM_ADDRESS_ROWS, core.memory_size)
    assert "shm.key.bias" not in params
    assert "shm.query.bias" not in params
    assert "shm.value.bias" not in params
    assert "shm.calibration.bias" not in params
    assert "shm.eta.bias" not in params
    np.testing.assert_array_equal(params["shm.norm.scale"], 1.0)
    np.testing.assert_array_equal(params["shm.norm.bias"], 0.0)

    theta_bound = math.sqrt(6.0 / (SHM_ADDRESS_ROWS + core.memory_size))
    assert float(jnp.max(jnp.abs(params["shm.theta"]))) <= theta_bound


def test_fixed_addresses_match_independent_translation_of_pinned_equations():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(1))
    state = jax.random.normal(
        jax.random.PRNGKey(2),
        (3, core.memory_size, core.memory_size),
    )
    policy_input = jax.random.normal(
        jax.random.PRNGKey(3),
        (3, core.input_dim),
    )
    reset = jnp.array([False, True, False])
    addresses = jnp.array([0, 73, 127], dtype=jnp.int32)

    actual = core.step(
        params,
        state,
        policy_input,
        reset,
        addresses,
    )
    expected = _numpy_official_step(
        params,
        state,
        policy_input,
        reset,
        addresses,
    )

    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            actual_value,
            expected_value,
            rtol=2e-5,
            atol=2e-5,
        )


def test_apply_sequence_matches_repeated_steps_with_asynchronous_resets():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(4))
    initial_state = jax.random.normal(
        jax.random.PRNGKey(5),
        (3, core.memory_size, core.memory_size),
    )
    policy_inputs = jax.random.normal(
        jax.random.PRNGKey(6),
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
    addresses = jnp.array(
        [
            [0, 1, 2],
            [17, 31, 63],
            [127, 5, 91],
            [4, 4, 4],
            [99, 73, 11],
            [16, 32, 64],
            [126, 125, 124],
        ],
        dtype=jnp.int32,
    )

    expected_state = initial_state
    expected_logits = []
    expected_values = []
    for policy_input, reset, address in zip(
        policy_inputs,
        resets,
        addresses,
        strict=True,
    ):
        expected_state, logits, values = core.step(
            params,
            expected_state,
            policy_input,
            reset,
            address,
        )
        expected_logits.append(logits)
        expected_values.append(values)

    actual_state, actual_logits, actual_values = core.apply_sequence(
        params,
        initial_state,
        policy_inputs,
        resets,
        addresses,
    )

    # XLA fuses the scan and repeated calls differently on GPU. The largest
    # observed float32 discrepancy is below 3e-4.
    np.testing.assert_allclose(
        actual_state,
        expected_state,
        rtol=5e-3,
        atol=5e-4,
    )
    np.testing.assert_allclose(
        actual_logits,
        jnp.stack(expected_logits),
        rtol=5e-3,
        atol=5e-4,
    )
    np.testing.assert_allclose(
        actual_values,
        jnp.stack(expected_values),
        rtol=5e-3,
        atol=5e-4,
    )


def test_reset_discards_only_selected_environment_memory():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(7))
    state = jnp.full(
        (2, core.memory_size, core.memory_size),
        3.0,
    )
    zero_state = core.initial_state(2)
    policy_input = jax.random.normal(
        jax.random.PRNGKey(8),
        (2, core.input_dim),
    )
    addresses = jnp.array([3, 3], dtype=jnp.int32)

    reset_result = core.step(
        params,
        state,
        policy_input,
        jnp.array([True, False]),
        addresses,
    )
    fresh_result = core.step(
        params,
        zero_state,
        policy_input,
        jnp.array([False, False]),
        addresses,
    )

    for reset_value, fresh_value in zip(reset_result, fresh_result, strict=True):
        np.testing.assert_allclose(reset_value[0], fresh_value[0])
    assert not np.allclose(reset_result[0][1], fresh_result[0][1])


def test_paper_uniform_addresses_cover_all_rows_and_pass_distribution_check():
    core = _core(address_mode="paper_uniform")
    addresses = np.asarray(
        core.sample_addresses(
            jax.random.PRNGKey(9),
            (SHM_ADDRESS_ROWS * 2_048,),
        )
    )
    counts = np.bincount(addresses, minlength=SHM_ADDRESS_ROWS)
    expected = addresses.size / SHM_ADDRESS_ROWS
    chi_square = np.square(counts - expected).sum() / expected

    assert addresses.min() == 0
    assert addresses.max() == SHM_ADDRESS_ROWS - 1
    assert np.all(counts > 0)
    assert chi_square < 300.0


def test_v1_1_popgym_compat_always_selects_row_zero():
    core = _core(address_mode="v1_1_popgym_compat")
    first = core.sample_addresses(jax.random.PRNGKey(10), (4_096,))
    second = core.sample_addresses(jax.random.PRNGKey(11), (4_096,))

    np.testing.assert_array_equal(first, 0)
    np.testing.assert_array_equal(second, 0)

    params = core.initialize(jax.random.PRNGKey(12))
    state = core.initial_state(3)
    policy_input = jax.random.normal(
        jax.random.PRNGKey(13),
        (3, core.input_dim),
    )
    reset = jnp.array([True, False, True])
    keyed = core.step_with_key(
        params,
        state,
        policy_input,
        reset,
        jax.random.PRNGKey(14),
    )
    explicit = core.step(
        params,
        state,
        policy_input,
        reset,
        jnp.zeros((3,), dtype=jnp.int32),
    )

    np.testing.assert_array_equal(keyed[3], 0)
    for keyed_value, explicit_value in zip(keyed[:3], explicit, strict=True):
        np.testing.assert_array_equal(keyed_value, explicit_value)


def test_collection_addresses_replay_deterministically():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(15))
    state = core.initial_state(2)
    policy_inputs = jax.random.normal(
        jax.random.PRNGKey(16),
        (8, 2, core.input_dim),
    )
    resets = jnp.array(
        [
            [True, True],
            [False, False],
            [False, False],
            [False, True],
            [False, False],
            [True, False],
            [False, False],
            [False, False],
        ]
    )
    collected = core.apply_sequence_with_key(
        params,
        state,
        policy_inputs,
        resets,
        jax.random.PRNGKey(17),
    )
    replayed = core.apply_sequence(
        params,
        state,
        policy_inputs,
        resets,
        collected[3],
    )

    for collected_value, replayed_value in zip(
        collected[:3],
        replayed,
        strict=True,
    ):
        np.testing.assert_array_equal(collected_value, replayed_value)

    other_addresses = core.sample_addresses(
        jax.random.PRNGKey(18),
        resets.shape,
    )
    assert not np.array_equal(collected[3], other_addresses)
    other = core.apply_sequence(
        params,
        state,
        policy_inputs,
        resets,
        other_addresses,
    )
    assert not np.allclose(collected[0], other[0])


def test_shared_ppo_loss_replays_collection_addresses_exactly():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(181))
    batch_size = 2
    time_steps = 5
    initial_state = core.initial_state(batch_size)
    policy_inputs = jax.random.normal(
        jax.random.PRNGKey(182),
        (time_steps, batch_size, core.input_dim),
    )
    resets = jnp.asarray(
        [
            [True, True],
            [False, False],
            [False, True],
            [False, False],
            [True, False],
        ]
    )
    addresses = core.sample_addresses(
        jax.random.PRNGKey(183),
        resets.shape,
    )
    _, logits, values = core.apply_sequence(
        params,
        initial_state,
        policy_inputs,
        resets,
        addresses,
    )
    actions = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    old_log_probability = categorical_log_probability(logits, actions)
    rollout = Rollout(
        policy_input=policy_inputs,
        reset=resets,
        action=actions,
        old_log_probability=old_log_probability,
        old_value=values,
        reward=jnp.zeros_like(values),
        done=jnp.zeros_like(resets),
        episode_return=jnp.zeros_like(values),
        episode_complete=jnp.zeros_like(resets),
        action_mask=None,
        policy_auxiliary=addresses,
    )
    learner = SharedPPO(
        policy_core=core,
        environment=None,
        environment_params=None,
        action_dim=core.action_dim,
        config=PPOConfig(
            total_steps=4,
            num_envs=2,
            rollout_steps=2,
            num_minibatches=1,
        ),
    )

    _, (_, _, _, approximate_kl) = learner._loss(
        params,
        initial_state,
        rollout,
        jnp.ones_like(values),
        values,
    )
    np.testing.assert_allclose(approximate_kl, 0.0, atol=1e-7)

    deterministic_core = MemorylessMLPPolicyCore(
        input_dim=core.input_dim,
        action_dim=core.action_dim,
        hidden_size=7,
    )
    deterministic_learner = SharedPPO(
        policy_core=deterministic_core,
        environment=None,
        environment_params=None,
        action_dim=core.action_dim,
        config=learner.config,
    )
    assert not getattr(
        deterministic_learner.policy_core,
        "requires_policy_aux_replay",
        False,
    )


def test_step_sequence_and_keyed_collection_are_jittable_and_differentiable():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(19))
    state = core.initial_state(2)
    policy_inputs = jax.random.normal(
        jax.random.PRNGKey(20),
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
    addresses = core.sample_addresses(
        jax.random.PRNGKey(21),
        resets.shape,
    )

    step_result = jax.jit(core.step)(
        params,
        state,
        policy_inputs[0],
        resets[0],
        addresses[0],
    )
    sequence_result = jax.jit(core.apply_sequence)(
        params,
        state,
        policy_inputs,
        resets,
        addresses,
    )
    keyed_result = jax.jit(core.apply_sequence_with_key)(
        params,
        state,
        policy_inputs,
        resets,
        jax.random.PRNGKey(22),
    )

    assert step_result[0].shape == (2, core.memory_size, core.memory_size)
    assert step_result[1].shape == (2, core.action_dim)
    assert sequence_result[1].shape == (5, 2, core.action_dim)
    assert keyed_result[3].shape == resets.shape

    def loss(candidate_params):
        final_state, logits, values = core.apply_sequence(
            candidate_params,
            state,
            policy_inputs,
            resets,
            addresses,
        )
        return (
            1e-4 * jnp.mean(jnp.square(final_state))
            + jnp.mean(jnp.square(logits))
            + jnp.mean(jnp.square(values))
        )

    gradients = jax.jit(jax.grad(loss))(params)
    assert bool(jnp.isfinite(loss(params)))
    assert all(bool(jnp.all(jnp.isfinite(gradient))) for gradient in jax.tree.leaves(gradients))
    assert float(jnp.linalg.norm(gradients["shm.theta"])) > 0.0
    assert float(jnp.linalg.norm(gradients["shm.calibration.kernel"])) > 0.0
    assert float(jnp.linalg.norm(gradients["actor.kernel"])) > 0.0
    assert float(jnp.linalg.norm(gradients["critic.kernel"])) > 0.0


def test_accepts_the_registered_augmented_policy_input():
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
    core = SHMPolicyCore(
        input_dim=policy_input.shape[-1],
        action_dim=3,
        hidden_size=7,
        memory_size=4,
    )

    state, logits, values = core.step(
        core.initialize(jax.random.PRNGKey(23)),
        core.initial_state(3),
        policy_input,
        reset,
        jnp.array([0, 64, 127]),
    )

    assert state.shape == (3, 4, 4)
    assert logits.shape == (3, 3)
    assert values.shape == (3,)


def test_parameter_count_includes_shm_table_projection_and_shared_heads():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(24))
    expected = (
        2 * core.input_dim
        + 4 * core.input_dim * core.memory_size
        + core.input_dim
        + SHM_ADDRESS_ROWS * core.memory_size
        + core.input_dim * core.memory_size
        + core.memory_size
        + core.memory_size * core.hidden_size
        + core.hidden_size
        + core.hidden_size * core.action_dim
        + core.action_dim
        + core.hidden_size
        + 1
    )

    assert core.count_parameters(params) == expected
    assert "actor.kernel" in params
    assert "critic.kernel" in params


def test_width_matcher_selects_globally_closest_positive_width():
    target_parameters = 20_000
    input_dim = 11
    action_dim = 5
    memory_size = 16
    width = match_shm_hidden_size(
        target_parameters=target_parameters,
        input_dim=input_dim,
        action_dim=action_dim,
        memory_size=memory_size,
    )

    def count(candidate_width):
        candidate = SHMPolicyCore(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_size=candidate_width,
            memory_size=memory_size,
        )
        return candidate.count_parameters(candidate.initialize(jax.random.PRNGKey(25)))

    selected_error = abs(count(width) - target_parameters)
    if width > 1:
        assert selected_error <= abs(count(width - 1) - target_parameters)
    assert selected_error <= abs(count(width + 1) - target_parameters)


@pytest.mark.parametrize(
    "field",
    [
        "target_parameters",
        "input_dim",
        "action_dim",
        "memory_size",
        "maximum_width",
    ],
)
def test_matcher_rejects_nonpositive_arguments(field):
    arguments = {
        "target_parameters": 10_000,
        "input_dim": 7,
        "action_dim": 3,
        "memory_size": 16,
        "maximum_width": 512,
    }
    arguments[field] = 0

    with pytest.raises(ValueError, match=field):
        match_shm_hidden_size(**arguments)


@pytest.mark.parametrize(
    "field",
    ["input_dim", "action_dim", "hidden_size", "memory_size"],
)
def test_core_rejects_nonpositive_dimensions(field):
    arguments = {
        "input_dim": 5,
        "action_dim": 3,
        "hidden_size": 6,
        "memory_size": 4,
    }
    arguments[field] = 0

    with pytest.raises(ValueError, match=field):
        SHMPolicyCore(**arguments)


def test_core_rejects_unknown_address_mode():
    with pytest.raises(ValueError, match="address_mode"):
        SHMPolicyCore(
            input_dim=5,
            action_dim=3,
            hidden_size=6,
            address_mode="unregistered",
        )


def test_rejects_address_shapes_that_cannot_be_replayed():
    core = _core()
    params = core.initialize(jax.random.PRNGKey(26))
    state = core.initial_state(2)
    policy_input = jnp.zeros((2, core.input_dim))
    reset = jnp.zeros((2,), dtype=jnp.bool_)

    with pytest.raises(ValueError, match="address shape"):
        core.step(
            params,
            state,
            policy_input,
            reset,
            jnp.zeros((2, 1), dtype=jnp.int32),
        )

    with pytest.raises(ValueError, match="addresses must match resets"):
        core.apply_sequence(
            params,
            state,
            jnp.zeros((3, 2, core.input_dim)),
            jnp.zeros((3, 2), dtype=jnp.bool_),
            jnp.zeros((3, 2, 1), dtype=jnp.int32),
        )
