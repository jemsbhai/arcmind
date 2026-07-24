"""Differential, invariant, parameter, and learner tests for AGaLiTe."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.pobax.agalite_core import (
    AGALITE_APPROXIMATION_CHANNELS,
    AGALITE_ATTENTION_EPSILON,
    AGALITE_ETA,
    AGALITE_FEEDFORWARD_SIZE,
    AGALITE_GATE_BIAS,
    AGALITE_HEAD_SIZE,
    AGALITE_LAYER_NORM_EPSILON,
    AGALITE_MODEL_SIZE,
    AGALITE_NUM_HEADS,
    AGALITE_NUM_LAYERS,
    AGaLiTePolicyCore,
    AGaLiTeState,
    SourceCompatibleAGaLiTePolicyCore,
    _backbone_step,
    _initial_state,
    _linear,
    agalite_parameter_count,
    match_agalite_hidden_size,
)
from benchmarks.pobax.model_registry import (
    AGALITE_A2C_SHA256,
    AGALITE_ACTOR_CRITIC_SHA256,
    AGALITE_AUDITED_COMMIT,
    AGALITE_DIFFERENTIAL_FIXTURE_SHA256,
    AGALITE_FLATTEN_SHA256,
    AGALITE_HEADS_SHA256,
    AGALITE_LAYERS_SHA256,
    AGALITE_LICENSE_SHA256,
    AGALITE_MODEL_SHA256,
    AGALITE_REQUIREMENTS_SHA256,
    AGALITE_SEQUENCE_FACTORY_SHA256,
    AGALITE_SHARED_REFERENCE_IMPLEMENTATION,
    AGALITE_SOURCE_COMPAT_REFERENCE_IMPLEMENTATION,
    AGALITE_TMAZE_CONFIG_SHA256,
    FIXED_OFFICIAL_PARAMETER_CONTRACT,
    PARAMETER_MATCHED_CONTRACT,
    PRIMARY_COMPARISON_ROLE,
    SUPPLEMENTAL_COMPARISON_ROLE,
    fixed_official_parameter_count,
    policy_contract_metadata_for_model,
    validate_policy_core_contract,
)
from benchmarks.pobax.run_pilot import build_policy_core
from benchmarks.pobax.shared_ppo import (
    PPOConfig,
    Rollout,
    SharedPPO,
    categorical_log_probability,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "agalite_official_v1.json"
FIXTURE_HASH_PATH = FIXTURE_PATH.with_suffix(".sha256")

SOURCE_HASHES = {
    "LICENSE": AGALITE_LICENSE_SHA256,
    "config/tmaze/arelit.yaml": AGALITE_TMAZE_CONFIG_SHA256,
    "requirements.txt": AGALITE_REQUIREMENTS_SHA256,
    "src/agents/a2c.py": AGALITE_A2C_SHA256,
    "src/model_fns/achead_fns.py": AGALITE_HEADS_SHA256,
    "src/model_fns/repr_fns.py": AGALITE_FLATTEN_SHA256,
    "src/model_fns/seq_fns.py": AGALITE_SEQUENCE_FACTORY_SHA256,
    "src/models/actor_critic.py": AGALITE_ACTOR_CRITIC_SHA256,
    "src/models/agalite/agalite.py": AGALITE_MODEL_SHA256,
    "src/models/agalite/layers.py": AGALITE_LAYERS_SHA256,
}


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _fixture_params(payload: dict) -> dict[str, jax.Array]:
    return {
        name: jnp.asarray(value, dtype=jnp.float32)
        for name, value in payload["parameters"].items()
    }


def _fixture_replay(payload: dict):
    configuration = payload["configuration"]
    params = _fixture_params(payload)
    policy_inputs = jnp.asarray(payload["policy_inputs"], dtype=jnp.float32)
    resets = jnp.asarray(payload["resets"], dtype=jnp.bool_)
    state = _initial_state(
        policy_inputs.shape[1],
        num_layers=configuration["num_layers"],
        approximation_channels=configuration["approximation_channels"],
        num_heads=configuration["num_heads"],
        eta=configuration["eta"],
        head_dim=configuration["head_dim"],
        dtype=jnp.float32,
    )
    logits = []
    values = []
    for policy_input, reset in zip(policy_inputs, resets, strict=True):
        state, features = _backbone_step(
            params,
            state,
            policy_input[..., : configuration["observation_dim"]],
            reset,
            hidden_size=configuration["hidden_size"],
            head_dim=configuration["head_dim"],
            feedforward_size=configuration["feedforward_size"],
            num_heads=configuration["num_heads"],
            eta=configuration["eta"],
            approximation_channels=configuration["approximation_channels"],
            num_layers=configuration["num_layers"],
            attention_epsilon=configuration["attention_epsilon"],
            layer_norm_epsilon=configuration["layer_norm_epsilon"],
        )
        actor_hidden = jnp.tanh(_linear(params, "actor.hidden", features))
        critic_hidden = jnp.tanh(_linear(params, "critic.hidden", features))
        logits.append(_linear(params, "actor.output", actor_hidden))
        values.append(_linear(params, "critic.output", critic_hidden)[..., 0])
    return state, jnp.stack(logits), jnp.stack(values)


def _tiny_core() -> AGaLiTePolicyCore:
    return AGaLiTePolicyCore(
        input_dim=5,
        action_dim=3,
        hidden_size=4,
        head_dim=2,
        feedforward_size=4,
        num_heads=2,
        eta=2,
        approximation_channels=2,
        num_layers=2,
    )


def _assert_state_close(actual: AGaLiTeState, expected: AGaLiTeState) -> None:
    for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=2e-6, atol=2e-6)


def test_differential_fixture_is_immutable_and_names_exact_official_sources() -> None:
    digest = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    assert digest == AGALITE_DIFFERENTIAL_FIXTURE_SHA256
    assert FIXTURE_HASH_PATH.read_text(encoding="utf-8").strip() == (
        f"{AGALITE_DIFFERENTIAL_FIXTURE_SHA256}  {FIXTURE_PATH.name}"
    )
    provenance = _fixture()["provenance"]
    assert provenance["commit"] == AGALITE_AUDITED_COMMIT
    assert provenance["source_hashes"] == SOURCE_HASHES
    assert provenance["jax_backend"] == "cpu"
    assert provenance["flax_version"] == "0.11.2"


def test_translated_core_matches_official_flax_policy_fixture() -> None:
    payload = _fixture()
    with jax.default_device(jax.devices(backend="cpu")[0]):
        state, logits, values = _fixture_replay(payload)
    expected = payload["expected"]
    for name in state._fields:
        np.testing.assert_allclose(
            getattr(state, name),
            expected["state"][name],
            rtol=2e-6,
            atol=2e-6,
        )
    np.testing.assert_allclose(logits, expected["logits"], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(values, expected["values"], rtol=2e-6, atol=2e-6)


def test_source_contract_is_complete_tmaze_policy_and_shared_lane_is_distinct() -> None:
    source = AGALITE_SOURCE_COMPAT_REFERENCE_IMPLEMENTATION
    shared = AGALITE_SHARED_REFERENCE_IMPLEMENTATION

    assert source["source_variant"] == "released_tmaze_vector_policy"
    assert source["input_contract"] == "flattened_observation_only"
    assert source["architecture"] == {
        "num_layers": 4,
        "model_size": 128,
        "head_size": 64,
        "feedforward_size": 128,
        "num_heads": 4,
        "eta": 4,
        "approximation_channels": 2,
    }
    assert source["actor_critic"]["activation"] == "tanh"
    assert source["author_learner"] == "A2C"
    assert source["integration_learner"] == "shared_PPO"
    assert source["parameter_contract"] == FIXED_OFFICIAL_PARAMETER_CONTRACT
    assert source["comparison_role"] == SUPPLEMENTAL_COMPARISON_ROLE
    assert shared["input_contract"] == "shared_augmented_policy_input"
    assert shared["parameter_contract"] == PARAMETER_MATCHED_CONTRACT
    assert shared["comparison_role"] == PRIMARY_COMPARISON_ROLE
    assert source["executable_contract"] == shared["executable_contract"]


def test_initial_state_has_batch_axis_zero_and_source_phase_one() -> None:
    core = _tiny_core()
    state = core.initial_state(3)
    assert state.tilde_key.shape == (3, 2, 2, 2, 4)
    assert state.tilde_value.shape == (3, 2, 2, 2, 2)
    assert state.normalizer.shape == (3, 2, 2, 4)
    assert state.tick.shape == (3, 2, 1)
    np.testing.assert_array_equal(state.tick, 1.0)


def test_first_token_uses_phase_two_and_reset_never_resets_phase() -> None:
    core = _tiny_core()
    params = core.initialize(jax.random.PRNGKey(1))
    state = core.initial_state(2)._replace(
        tick=jnp.asarray([[[7.0], [7.0]], [[11.0], [11.0]]])
    )
    next_state, _, _ = core.step(
        params,
        state,
        jnp.ones((2, core.input_dim)),
        jnp.asarray([True, False]),
    )
    np.testing.assert_array_equal(next_state.tick[:, :, 0], [[8.0, 8.0], [12.0, 12.0]])

    fresh_state, _, _ = core.step(
        params,
        core.initial_state(2),
        jnp.ones((2, core.input_dim)),
        jnp.asarray([True, True]),
    )
    np.testing.assert_array_equal(fresh_state.tick, 2.0)


def test_reset_discards_only_prior_memory_then_incorporates_current_token() -> None:
    core = _tiny_core()
    params = core.initialize(jax.random.PRNGKey(2))
    fresh = core.initial_state(2)._replace(
        tick=jnp.full((2, 2, 1), 9.0),
    )
    first_old = fresh._replace(
        tilde_key=jnp.ones_like(fresh.tilde_key),
        tilde_value=2.0 * jnp.ones_like(fresh.tilde_value),
        normalizer=3.0 * jnp.ones_like(fresh.normalizer),
    )
    second_old = fresh._replace(
        tilde_key=11.0 * jnp.ones_like(fresh.tilde_key),
        tilde_value=12.0 * jnp.ones_like(fresh.tilde_value),
        normalizer=13.0 * jnp.ones_like(fresh.normalizer),
    )
    policy_input = jax.random.normal(jax.random.PRNGKey(3), (2, core.input_dim))
    reset = jnp.asarray([True, True])

    first_state, first_logits, first_values = core.step(
        params,
        first_old,
        policy_input,
        reset,
    )
    second_state, second_logits, second_values = core.step(
        params,
        second_old,
        policy_input,
        reset,
    )

    _assert_state_close(first_state, second_state)
    np.testing.assert_allclose(first_logits, second_logits, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(first_values, second_values, rtol=2e-6, atol=2e-6)
    assert bool(jnp.any(first_state.normalizer != 0.0))


def test_step_sequence_and_chunked_execution_are_equivalent() -> None:
    core = _tiny_core()
    params = core.initialize(jax.random.PRNGKey(4))
    policy_inputs = jax.random.normal(jax.random.PRNGKey(5), (7, 3, core.input_dim))
    resets = jnp.zeros((7, 3), dtype=jnp.bool_).at[0].set(True)
    resets = resets.at[2, 1].set(True).at[5, 0].set(True)

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
    first_state, first_logits, first_values = core.apply_sequence(
        params,
        core.initial_state(3),
        policy_inputs[:3],
        resets[:3],
    )
    chunk_state, second_logits, second_values = core.apply_sequence(
        params,
        first_state,
        policy_inputs[3:],
        resets[3:],
    )

    _assert_state_close(loop_state, scan_state)
    _assert_state_close(chunk_state, scan_state)
    np.testing.assert_allclose(loop_logits, scan_logits, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(loop_values, scan_values, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        jnp.concatenate([first_logits, second_logits]),
        scan_logits,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        jnp.concatenate([first_values, second_values]),
        scan_values,
        rtol=2e-6,
        atol=2e-6,
    )


def test_source_lane_is_observation_only() -> None:
    core = SourceCompatibleAGaLiTePolicyCore(
        input_dim=8,
        observation_dim=3,
        action_dim=3,
    )
    params = core.initialize(jax.random.PRNGKey(6))
    first = jax.random.normal(jax.random.PRNGKey(7), (1, 8))
    second = first.at[:, 3:].set(jnp.asarray([[101.0, 102.0, 103.0, 104.0, 105.0]]))
    state = core.initial_state(1)

    first_state, first_logits, first_values = core.step(
        params,
        state,
        first,
        jnp.asarray([True]),
    )
    second_state, second_logits, second_values = core.step(
        params,
        state,
        second,
        jnp.asarray([True]),
    )

    _assert_state_close(first_state, second_state)
    np.testing.assert_array_equal(first_logits, second_logits)
    np.testing.assert_array_equal(first_values, second_values)


def test_source_initialization_has_released_gains_biases_and_gate_vectors() -> None:
    core = SourceCompatibleAGaLiTePolicyCore(
        input_dim=8,
        observation_dim=3,
        action_dim=3,
    )
    params = core.initialize(jax.random.PRNGKey(8))
    for name, value in params.items():
        if name.endswith(".kernel"):
            kernel = np.asarray(value)
            gram = kernel @ kernel.T if kernel.shape[0] <= kernel.shape[1] else kernel.T @ kernel
            np.testing.assert_allclose(
                gram,
                2.0 * np.eye(gram.shape[0]),
                rtol=3e-5,
                atol=3e-5,
            )
        elif name.endswith(".bias"):
            np.testing.assert_array_equal(value, 0.0)
        elif name.endswith(".bg"):
            np.testing.assert_array_equal(value, AGALITE_GATE_BIAS)


def test_parameter_formulas_and_fixed_registry_count_are_exact() -> None:
    source = SourceCompatibleAGaLiTePolicyCore(
        input_dim=22,
        observation_dim=16,
        action_dim=4,
    )
    source_params = source.initialize(jax.random.PRNGKey(9))
    source_count = source.count_parameters(source_params)
    assert source_count == 1_774_277
    assert source_count == agalite_parameter_count(
        input_dim=16,
        action_dim=4,
        hidden_size=128,
        head_dim=64,
        feedforward_size=128,
        num_heads=4,
        eta=4,
        num_layers=4,
        source_actor_critic=True,
    )
    assert source_count == fixed_official_parameter_count(
        "agalite_source_compat",
        asdict(source),
    )

    shared = _tiny_core()
    shared_params = shared.initialize(jax.random.PRNGKey(10))
    assert shared.count_parameters(shared_params) == agalite_parameter_count(
        input_dim=shared.input_dim,
        action_dim=shared.action_dim,
        hidden_size=shared.hidden_size,
        head_dim=shared.head_dim,
        feedforward_size=shared.feedforward_size,
        num_heads=shared.num_heads,
        eta=shared.eta,
        num_layers=shared.num_layers,
        source_actor_critic=False,
    )


def test_parameter_count_is_independent_of_approximation_channels() -> None:
    first = _tiny_core()
    second = AGaLiTePolicyCore(
        **{
            **asdict(first),
            "approximation_channels": 5,
        }
    )
    first_count = first.count_parameters(first.initialize(jax.random.PRNGKey(11)))
    second_count = second.count_parameters(second.initialize(jax.random.PRNGKey(11)))
    assert first_count == second_count
    assert first.initial_state(1).tilde_key.shape[2] == 2
    assert second.initial_state(1).tilde_key.shape[2] == 5


def test_matcher_is_globally_closest_even_width_and_builder_is_in_tolerance() -> None:
    target = 70_000
    width = match_agalite_hidden_size(
        target_parameters=target,
        input_dim=9,
        action_dim=4,
    )
    brute_force = min(
        range(2, 202, 2),
        key=lambda candidate: (
            abs(
                agalite_parameter_count(
                    input_dim=9,
                    action_dim=4,
                    hidden_size=candidate,
                    head_dim=candidate // 2,
                    feedforward_size=candidate,
                    num_heads=4,
                    eta=4,
                    num_layers=4,
                    source_actor_critic=False,
                )
                - target
            ),
            candidate,
        ),
    )
    assert width == brute_force

    source, source_count, source_target = build_policy_core(
        "agalite_source_compat",
        input_dim=9,
        observation_dim=3,
        action_dim=4,
        seed=12,
    )
    shared, shared_count, shared_target = build_policy_core(
        "agalite_shared",
        input_dim=9,
        observation_dim=3,
        action_dim=4,
        seed=12,
    )
    assert isinstance(source, SourceCompatibleAGaLiTePolicyCore)
    assert isinstance(shared, AGaLiTePolicyCore)
    assert source_count == fixed_official_parameter_count(
        "agalite_source_compat",
        asdict(source),
    )
    assert source_count / source_target > 1.1
    assert 0.9 <= shared_count / shared_target <= 1.1


def test_outputs_are_causal_and_jit_gradients_are_finite() -> None:
    core = _tiny_core()
    params = core.initialize(jax.random.PRNGKey(13))
    first = jax.random.normal(jax.random.PRNGKey(14), (6, 2, core.input_dim))
    second = first.at[4:].set(
        jax.random.normal(jax.random.PRNGKey(15), (2, 2, core.input_dim))
    )
    resets = jnp.zeros((6, 2), dtype=jnp.bool_).at[0].set(True)
    _, first_logits, first_values = core.apply_sequence(
        params,
        core.initial_state(2),
        first,
        resets,
    )
    _, second_logits, second_values = core.apply_sequence(
        params,
        core.initial_state(2),
        second,
        resets,
    )
    np.testing.assert_array_equal(first_logits[:4], second_logits[:4])
    np.testing.assert_array_equal(first_values[:4], second_values[:4])

    def loss_fn(trainable):
        _, logits, values = core.apply_sequence(
            trainable,
            core.initial_state(2),
            first,
            resets,
        )
        return jnp.mean(jnp.square(logits)) + jnp.mean(jnp.square(values))

    loss, gradients = jax.jit(jax.value_and_grad(loss_fn))(params)
    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(gradients))
    assert any(bool(jnp.any(jnp.abs(leaf) > 0.0)) for leaf in jax.tree.leaves(gradients))


def test_shared_ppo_replay_has_zero_preupdate_kl() -> None:
    core = _tiny_core()
    params = core.initialize(jax.random.PRNGKey(16))
    policy_inputs = jax.random.normal(jax.random.PRNGKey(17), (4, 2, core.input_dim))
    resets = jnp.asarray(
        [[True, True], [False, False], [True, False], [False, True]]
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


def test_registry_core_contracts_fail_closed_on_drift() -> None:
    source = asdict(
        SourceCompatibleAGaLiTePolicyCore(
            input_dim=8,
            observation_dim=3,
            action_dim=3,
        )
    )
    shared = asdict(
        AGaLiTePolicyCore(
            input_dim=8,
            action_dim=3,
            hidden_size=4,
            head_dim=2,
            feedforward_size=4,
        )
    )
    validate_policy_core_contract(
        "agalite_source_compat",
        source,
        field="source",
    )
    validate_policy_core_contract("agalite_shared", shared, field="shared")
    assert "agalite_executable_contract" in policy_contract_metadata_for_model(
        "agalite_shared"
    )

    for field, replacement in (
        ("eta", 8),
        ("approximation_channels", 3),
        ("layer_norm_epsilon", 1e-5),
    ):
        drifted = {**source, field: replacement}
        with pytest.raises(ValueError):
            validate_policy_core_contract(
                "agalite_source_compat",
                drifted,
                field="source",
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 127},
        {"head_dim": 32},
        {"eta": 8},
        {"approximation_channels": 1},
        {"layer_norm_epsilon": 1e-5},
    ],
)
def test_source_lane_rejects_architecture_drift(kwargs) -> None:
    arguments = {
        "input_dim": 8,
        "observation_dim": 3,
        "action_dim": 3,
        **kwargs,
    }
    with pytest.raises(ValueError):
        SourceCompatibleAGaLiTePolicyCore(**arguments)


def test_executable_constants_are_frozen() -> None:
    assert AGALITE_NUM_LAYERS == 4
    assert AGALITE_MODEL_SIZE == 128
    assert AGALITE_HEAD_SIZE == 64
    assert AGALITE_FEEDFORWARD_SIZE == 128
    assert AGALITE_NUM_HEADS == 4
    assert AGALITE_ETA == 4
    assert AGALITE_APPROXIMATION_CHANNELS == 2
    assert AGALITE_GATE_BIAS == 2.0
    assert AGALITE_ATTENTION_EPSILON == 1e-5
    assert AGALITE_LAYER_NORM_EPSILON == 1e-6
