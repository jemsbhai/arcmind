"""Source, recurrence, architecture, and learner tests for Memory Traces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.pobax.baseline_cores import MemoryTraceSharedPolicyCore
from benchmarks.pobax.memory_trace_core import (
    OFFICIAL_MEMORY_TRACE_HIDDEN_SIZE,
    OfficialMemoryTracePolicyCore,
    official_memory_trace_parameter_count,
)
from benchmarks.pobax.model_registry import (
    FIXED_OFFICIAL_PARAMETER_CONTRACT,
    MEMORY_TRACE_AUDITED_COMMIT,
    MEMORY_TRACE_DECAY_ORIGIN,
    MEMORY_TRACE_DECAYS,
    MEMORY_TRACE_DIFFERENTIAL_FIXTURE_SHA256,
    MEMORY_TRACE_EXAMPLE_SHA256,
    MEMORY_TRACE_OFFICIAL_REFERENCE_IMPLEMENTATION,
    MEMORY_TRACE_SHARED_REFERENCE_IMPLEMENTATION,
    MEMORY_TRACE_SOURCE_SHA256,
    PARAMETER_MATCHED_CONTRACT,
    PRIMARY_COMPARISON_ROLE,
    SUPPLEMENTAL_COMPARISON_ROLE,
)
from benchmarks.pobax.run_pilot import build_policy_core
from benchmarks.pobax.shared_ppo import (
    PPOConfig,
    Rollout,
    SharedPPO,
    categorical_log_probability,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "memory_trace_official_v1.json"
FIXTURE_HASH_PATH = FIXTURE_PATH.with_suffix(".sha256")


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _fixture_params(payload: dict) -> dict[str, jax.Array]:
    return {
        name: jnp.asarray(value, dtype=jnp.float32)
        for name, value in payload["parameters"].items()
    }


def test_differential_fixture_is_immutable_and_names_exact_official_sources() -> None:
    fixture_bytes = FIXTURE_PATH.read_bytes()
    assert hashlib.sha256(fixture_bytes).hexdigest() == (
        MEMORY_TRACE_DIFFERENTIAL_FIXTURE_SHA256
    )
    assert FIXTURE_HASH_PATH.read_text(encoding="utf-8").strip() == (
        f"{MEMORY_TRACE_DIFFERENTIAL_FIXTURE_SHA256}  {FIXTURE_PATH.name}"
    )
    provenance = _fixture()["provenance"]
    assert provenance["commit"] == MEMORY_TRACE_AUDITED_COMMIT
    assert provenance["source_hashes"] == {
        "examples/ppo_tmaze.py": MEMORY_TRACE_EXAMPLE_SHA256,
        "traces/ppo.py": MEMORY_TRACE_SOURCE_SHA256,
    }
    assert provenance["jax_backend"] == "cpu"


def test_official_core_matches_official_trace_and_actor_critic_fixture() -> None:
    payload = _fixture()
    configuration = payload["configuration"]
    # The immutable official oracle was exported on CPU. Replay on that same
    # backend so GPU TF32 lowering cannot dilute the differential comparison.
    with jax.default_device(jax.devices(backend="cpu")[0]):
        core = OfficialMemoryTracePolicyCore(
            input_dim=configuration["input_dim"],
            observation_dim=configuration["observation_dim"],
            action_dim=configuration["action_dim"],
        )
        params = _fixture_params(payload)
        policy_inputs = jnp.asarray(payload["policy_inputs"], dtype=jnp.float32)
        resets = jnp.asarray(payload["resets"], dtype=jnp.bool_)
        state = core.initial_state(policy_inputs.shape[1])
        actual_traces = []
        actual_features = []
        actual_logits = []
        actual_values = []

        for policy_input, reset in zip(policy_inputs, resets, strict=True):
            state, logits, values = core.step(
                params,
                state,
                policy_input,
                reset,
            )
            actual_traces.append(state.traces)
            actual_features.append(state.traces.reshape((state.traces.shape[0], -1)))
            actual_logits.append(logits)
            actual_values.append(values)

    expected = payload["expected"]
    np.testing.assert_allclose(actual_traces, expected["traces"], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        actual_features,
        expected["features"],
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        actual_logits,
        expected["logits"],
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(actual_values, expected["values"], rtol=2e-6, atol=2e-6)


def test_official_core_is_observation_only_and_trace_major() -> None:
    payload = _fixture()
    core = OfficialMemoryTracePolicyCore(
        input_dim=7,
        observation_dim=2,
        action_dim=3,
    )
    params = _fixture_params(payload)
    policy_inputs = jnp.asarray(payload["policy_inputs"], dtype=jnp.float32)
    resets = jnp.asarray(payload["resets"], dtype=jnp.bool_)
    changed_augmented_features = policy_inputs.at[..., 2:].set(
        jax.random.normal(jax.random.PRNGKey(30), policy_inputs[..., 2:].shape)
    )

    state_a, logits_a, values_a = core.apply_sequence(
        params,
        core.initial_state(2),
        policy_inputs,
        resets,
    )
    state_b, logits_b, values_b = core.apply_sequence(
        params,
        core.initial_state(2),
        changed_augmented_features,
        resets,
    )

    np.testing.assert_array_equal(state_a.traces, state_b.traces)
    np.testing.assert_array_equal(logits_a, logits_b)
    np.testing.assert_array_equal(values_a, values_b)

    trace_state = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]]])
    np.testing.assert_array_equal(
        trace_state.reshape((1, -1)),
        [[1.0, 2.0, 3.0, 4.0]],
    )


def test_reset_is_per_worker_and_precedes_current_observation_update() -> None:
    core = OfficialMemoryTracePolicyCore(
        input_dim=4,
        observation_dim=2,
        action_dim=2,
    )
    params = core.initialize(jax.random.PRNGKey(31))
    state = core.initial_state(2)._replace(
        traces=jnp.asarray(
            [
                [[10.0, 20.0], [30.0, 40.0]],
                [[2.0, 4.0], [6.0, 8.0]],
            ]
        )
    )
    policy_input = jnp.asarray(
        [[1.0, 3.0, 99.0, 98.0], [5.0, 7.0, 97.0, 96.0]]
    )

    new_state, _, _ = core.step(
        params,
        state,
        policy_input,
        jnp.asarray([True, False]),
    )

    np.testing.assert_allclose(
        new_state.traces[0],
        [[1.0, 3.0], [0.015, 0.045]],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        new_state.traces[1],
        [[5.0, 7.0], [5.985, 7.985]],
        rtol=1e-6,
        atol=1e-6,
    )


def test_step_loop_and_sequence_execution_are_identical() -> None:
    core = OfficialMemoryTracePolicyCore(
        input_dim=7,
        observation_dim=2,
        action_dim=3,
    )
    params = core.initialize(jax.random.PRNGKey(32))
    policy_inputs = jax.random.normal(jax.random.PRNGKey(33), (6, 3, 7))
    resets = jnp.zeros((6, 3), dtype=jnp.bool_).at[0].set(True)
    resets = resets.at[2, 1].set(True).at[4, 0].set(True)
    loop_state = core.initial_state(3)
    loop_logits = []
    loop_values = []
    for policy_input, reset in zip(policy_inputs, resets, strict=True):
        loop_state, logits, values = core.step(
            params,
            loop_state,
            policy_input,
            reset,
        )
        loop_logits.append(logits)
        loop_values.append(values)

    scan_state, scan_logits, scan_values = core.apply_sequence(
        params,
        core.initial_state(3),
        policy_inputs,
        resets,
    )

    np.testing.assert_allclose(loop_state.traces, scan_state.traces, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        loop_logits,
        scan_logits,
        rtol=1e-6,
        atol=(4e-6 if jax.default_backend() == "gpu" else 1e-6),
    )
    np.testing.assert_allclose(loop_values, scan_values, rtol=1e-6, atol=1e-6)


def test_official_initialization_has_exact_orthogonal_gains_and_zero_biases() -> None:
    core = OfficialMemoryTracePolicyCore(
        input_dim=7,
        observation_dim=2,
        action_dim=3,
    )
    params = core.initialize(jax.random.PRNGKey(34))
    expected_gain_squared = {
        "actor.hidden.0.kernel": 2.0,
        "actor.hidden.1.kernel": 2.0,
        "actor.output.kernel": 0.01**2,
        "critic.hidden.0.kernel": 2.0,
        "critic.hidden.1.kernel": 2.0,
        "critic.output.kernel": 1.0,
    }
    for name, gain_squared in expected_gain_squared.items():
        kernel = np.asarray(params[name])
        gram = kernel @ kernel.T if kernel.shape[0] < kernel.shape[1] else kernel.T @ kernel
        np.testing.assert_allclose(
            gram,
            gain_squared * np.eye(gram.shape[0]),
            rtol=2e-5,
            atol=2e-5,
        )
    for name, value in params.items():
        if name.endswith(".bias"):
            np.testing.assert_array_equal(value, 0.0)


def test_official_and_shared_parameter_contracts_are_distinct() -> None:
    official, official_count, target_count = build_policy_core(
        "memory_trace_official",
        input_dim=9,
        observation_dim=4,
        action_dim=3,
        seed=35,
    )
    shared, shared_count, shared_target = build_policy_core(
        "memory_trace_shared",
        input_dim=9,
        observation_dim=4,
        action_dim=3,
        seed=35,
    )

    assert isinstance(official, OfficialMemoryTracePolicyCore)
    assert isinstance(shared, MemoryTraceSharedPolicyCore)
    assert official_count == official.expected_parameter_count()
    assert official_count == official_memory_trace_parameter_count(
        observation_dim=4,
        action_dim=3,
    )
    assert official_count / target_count < 0.9
    assert 0.9 <= shared_count / shared_target <= 1.1
    assert OFFICIAL_MEMORY_TRACE_HIDDEN_SIZE == 64

    official_reference = MEMORY_TRACE_OFFICIAL_REFERENCE_IMPLEMENTATION
    shared_reference = MEMORY_TRACE_SHARED_REFERENCE_IMPLEMENTATION
    assert official_reference["parameter_contract"] == (
        FIXED_OFFICIAL_PARAMETER_CONTRACT
    )
    assert official_reference["comparison_role"] == SUPPLEMENTAL_COMPARISON_ROLE
    assert shared_reference["parameter_contract"] == PARAMETER_MATCHED_CONTRACT
    assert shared_reference["comparison_role"] == PRIMARY_COMPARISON_ROLE
    assert official_reference["decays"] == shared_reference["decays"] == [0.0, 0.985]
    assert official_reference["decay_origin"] == MEMORY_TRACE_DECAY_ORIGIN
    assert "not_author_selected_for_pobax" in MEMORY_TRACE_DECAY_ORIGIN


def test_official_core_exactly_replays_through_shared_ppo() -> None:
    core = OfficialMemoryTracePolicyCore(
        input_dim=7,
        observation_dim=2,
        action_dim=3,
    )
    params = core.initialize(jax.random.PRNGKey(36))
    policy_inputs = jax.random.normal(jax.random.PRNGKey(37), (5, 2, 7))
    resets = jnp.asarray(
        [[True, True], [False, False], [False, True], [True, False], [False, False]]
    )
    initial_state = core.initial_state(2)
    _, logits, values = core.apply_sequence(
        params,
        initial_state,
        policy_inputs,
        resets,
    )
    actions = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    rollout = Rollout(
        policy_input=policy_inputs,
        reset=resets,
        action=actions,
        old_log_probability=categorical_log_probability(logits, actions),
        old_value=values,
        reward=jnp.zeros_like(values),
        done=jnp.zeros_like(resets),
        episode_return=jnp.zeros_like(values),
        episode_complete=jnp.zeros_like(resets),
    )
    learner = SharedPPO(
        policy_core=core,
        environment=None,
        environment_params=None,
        action_dim=3,
        config=PPOConfig(
            total_steps=4,
            num_envs=2,
            rollout_steps=2,
            num_minibatches=1,
        ),
    )

    loss, (_, _, _, approximate_kl) = learner._loss(
        params,
        initial_state,
        rollout,
        jnp.ones_like(values),
        values,
    )

    assert bool(jnp.isfinite(loss))
    np.testing.assert_allclose(approximate_kl, 0.0, atol=1e-7)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 32},
        {"decays": (0.0, 0.9)},
        {"observation_dim": 8},
    ],
)
def test_official_core_rejects_architecture_or_decay_drift(kwargs) -> None:
    arguments = {
        "input_dim": 7,
        "observation_dim": 2,
        "action_dim": 3,
        **kwargs,
    }
    with pytest.raises(ValueError):
        OfficialMemoryTracePolicyCore(**arguments)


def test_shared_lane_freezes_source_example_decays() -> None:
    core = MemoryTraceSharedPolicyCore(
        input_dim=3,
        action_dim=2,
        hidden_size=4,
    )
    assert core.decays == MEMORY_TRACE_DECAYS

    params = core.initialize(jax.random.PRNGKey(38))
    state, _, _ = core.step(
        params,
        core.initial_state(1),
        jnp.asarray([[2.0, 4.0, 6.0]]),
        jnp.asarray([True]),
    )
    np.testing.assert_allclose(
        state.traces[0],
        [[2.0, 4.0, 6.0], [0.03, 0.06, 0.09]],
        rtol=1e-6,
        atol=1e-6,
    )
