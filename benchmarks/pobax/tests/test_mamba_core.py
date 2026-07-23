"""Equation, parity, reset, and accounting tests for the Mamba-1 core."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.pobax.arcmind_reference import ReferenceConfig
from benchmarks.pobax.mamba_core import (
    MAMBA_AUDITED_COMMIT,
    MAMBA_D_CONV,
    MAMBA_D_STATE,
    MAMBA_EXPAND,
    MAMBA_SIMPLE_SHA256,
    MAMBA_VERSION,
    MambaPolicyCore,
    MambaState,
    mamba_parameter_count,
    match_mamba_hidden_size,
)
from benchmarks.pobax.policy_core import ArcMindPolicyCore

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mamba1_official_step_v1.json"
FIXTURE_SHA256 = "8bfa948c8c1fd28bcde3e7dd7eebff8bb5e54406dd1fd11f7d69317f1c6e3015"


def _core() -> MambaPolicyCore:
    return MambaPolicyCore(input_dim=7, action_dim=4, hidden_size=5)


def _assert_state_close(
    actual: MambaState,
    expected: MambaState,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> None:
    np.testing.assert_allclose(actual.convolution, expected.convolution, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual.ssm, expected.ssm, rtol=rtol, atol=atol)


def test_fixture_is_immutable_and_names_the_exact_audited_source() -> None:
    fixture_bytes = FIXTURE_PATH.read_bytes()
    assert hashlib.sha256(fixture_bytes).hexdigest() == FIXTURE_SHA256
    payload = json.loads(fixture_bytes)
    provenance = payload["provenance"]

    assert provenance["version"] == MAMBA_VERSION == "2.2.6.post3"
    assert provenance["commit"] == MAMBA_AUDITED_COMMIT
    assert provenance["source_sha256"] == MAMBA_SIMPLE_SHA256
    assert provenance["execution_path"] == "Mamba.step dependency-light PyTorch slow path"


def test_block_step_matches_official_pytorch_outputs_with_transplanted_weights() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    configuration = payload["configuration"]
    core = MambaPolicyCore(
        input_dim=configuration["hidden_size"],
        action_dim=2,
        hidden_size=configuration["hidden_size"],
    )
    params = {
        name: jnp.asarray(values, dtype=jnp.float32)
        for name, values in payload["parameters"].items()
    }
    state = MambaState(
        convolution=jnp.asarray(
            payload["initial_state"]["convolution"],
            dtype=jnp.float32,
        ),
        ssm=jnp.asarray(payload["initial_state"]["ssm"], dtype=jnp.float32),
    )
    hidden = jnp.asarray(payload["hidden"], dtype=jnp.float32)
    reset = jnp.zeros((configuration["batch_size"],), dtype=jnp.bool_)

    outputs = []
    for timestep in range(configuration["sequence_length"]):
        state, output = core.block_step(
            params,
            state,
            hidden[:, timestep, :],
            reset,
        )
        outputs.append(output)
    output_sequence = jnp.stack(outputs, axis=1)

    np.testing.assert_allclose(
        output_sequence,
        payload["expected"]["outputs"],
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        state.convolution,
        payload["expected"]["final_convolution"],
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        state.ssm,
        payload["expected"]["final_ssm"],
        rtol=2e-5,
        atol=2e-6,
    )


def test_initialization_preserves_official_mamba1_defaults_and_special_parameters() -> None:
    core = MambaPolicyCore(input_dim=11, action_dim=3, hidden_size=17)
    params = core.initialize(jax.random.PRNGKey(1))

    assert MAMBA_D_STATE == 16
    assert MAMBA_D_CONV == 4
    assert MAMBA_EXPAND == 2
    assert core.d_inner == 34
    assert core.dt_rank == math.ceil(17 / 16) == 2
    np.testing.assert_allclose(
        params["mamba.A_log"],
        jnp.broadcast_to(
            jnp.log(jnp.arange(1, MAMBA_D_STATE + 1))[None, :],
            (core.d_inner, MAMBA_D_STATE),
        ),
    )
    np.testing.assert_allclose(params["mamba.D"], jnp.ones((core.d_inner,)))
    dt = jax.nn.softplus(params["mamba.dt_proj.bias"])
    assert bool(jnp.all(dt >= core.dt_min))
    assert bool(jnp.all(dt <= core.dt_max))
    assert params["mamba.in_proj.kernel"].shape == (17, 68)
    assert params["mamba.x_proj.kernel"].shape == (34, 34)


def test_apply_sequence_matches_repeated_steps_with_asynchronous_resets() -> None:
    core = _core()
    params = core.initialize(jax.random.PRNGKey(2))
    state = core.initial_state(3)
    policy_inputs = jax.random.normal(
        jax.random.PRNGKey(3),
        (8, 3, core.input_dim),
    )
    resets = jnp.asarray(
        [
            [True, True, True],
            [False, False, False],
            [False, True, False],
            [False, False, False],
            [True, False, False],
            [False, False, True],
            [False, False, False],
            [False, True, False],
        ]
    )

    expected_state = state
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
        state,
        policy_inputs,
        resets,
    )

    _assert_state_close(actual_state, expected_state)
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


def test_chunked_sequence_is_exactly_equivalent_to_one_scan() -> None:
    core = _core()
    params = core.initialize(jax.random.PRNGKey(4))
    state = core.initial_state(2)
    policy_inputs = jax.random.normal(
        jax.random.PRNGKey(5),
        (9, 2, core.input_dim),
    )
    resets = jnp.zeros((9, 2), dtype=jnp.bool_).at[0].set(True)
    resets = resets.at[4, 1].set(True)

    full_state, full_logits, full_values = core.apply_sequence(
        params,
        state,
        policy_inputs,
        resets,
    )
    first_state, first_logits, first_values = core.apply_sequence(
        params,
        state,
        policy_inputs[:4],
        resets[:4],
    )
    chunked_state, second_logits, second_values = core.apply_sequence(
        params,
        first_state,
        policy_inputs[4:],
        resets[4:],
    )

    _assert_state_close(chunked_state, full_state)
    np.testing.assert_allclose(jnp.concatenate([first_logits, second_logits]), full_logits)
    np.testing.assert_allclose(jnp.concatenate([first_values, second_values]), full_values)


def test_reset_clears_both_caches_for_only_the_selected_environment() -> None:
    core = _core()
    params = core.initialize(jax.random.PRNGKey(6))
    random_state = MambaState(
        convolution=jax.random.normal(
            jax.random.PRNGKey(7),
            (2, core.d_inner, MAMBA_D_CONV),
        ),
        ssm=jax.random.normal(
            jax.random.PRNGKey(8),
            (2, core.d_inner, MAMBA_D_STATE),
        ),
    )
    policy_input = jax.random.normal(jax.random.PRNGKey(9), (2, core.input_dim))

    reset_state, reset_logits, reset_values = core.step(
        params,
        random_state,
        policy_input,
        jnp.asarray([True, False]),
    )
    fresh_state, fresh_logits, fresh_values = core.step(
        params,
        core.initial_state(2),
        policy_input,
        jnp.asarray([False, False]),
    )

    np.testing.assert_allclose(reset_state.convolution[0], fresh_state.convolution[0])
    np.testing.assert_allclose(reset_state.ssm[0], fresh_state.ssm[0])
    np.testing.assert_allclose(reset_logits[0], fresh_logits[0])
    np.testing.assert_allclose(reset_values[0], fresh_values[0])
    assert not np.allclose(reset_state.convolution[1], fresh_state.convolution[1])
    assert not np.allclose(reset_state.ssm[1], fresh_state.ssm[1])


def test_outputs_are_causal() -> None:
    core = _core()
    params = core.initialize(jax.random.PRNGKey(10))
    first = jax.random.normal(jax.random.PRNGKey(11), (7, 2, core.input_dim))
    second = first.at[4:].set(jax.random.normal(jax.random.PRNGKey(12), (3, 2, core.input_dim)))
    resets = jnp.zeros((7, 2), dtype=jnp.bool_).at[0].set(True)

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


def test_jitted_outputs_and_gradients_are_finite() -> None:
    core = _core()
    params = core.initialize(jax.random.PRNGKey(13))
    policy_inputs = jax.random.normal(
        jax.random.PRNGKey(14),
        (6, 2, core.input_dim),
    )
    resets = jnp.zeros((6, 2), dtype=jnp.bool_).at[0].set(True)
    resets = resets.at[3, 0].set(True)

    def loss(candidate_params):
        final_state, logits, values = core.apply_sequence(
            candidate_params,
            core.initial_state(2),
            policy_inputs,
            resets,
        )
        cache_term = 1e-4 * (
            jnp.mean(jnp.square(final_state.convolution)) + jnp.mean(jnp.square(final_state.ssm))
        )
        return jnp.mean(jnp.square(logits)) + jnp.mean(jnp.square(values)) + cache_term

    compiled_loss = jax.jit(loss)
    gradients = jax.jit(jax.grad(loss))(params)

    assert bool(jnp.isfinite(compiled_loss(params)))
    assert all(bool(jnp.all(jnp.isfinite(gradient))) for gradient in jax.tree.leaves(gradients))
    for name in (
        "mamba.in_proj.kernel",
        "mamba.conv1d.kernel",
        "mamba.x_proj.kernel",
        "mamba.dt_proj.kernel",
        "mamba.A_log",
    ):
        assert float(jnp.linalg.norm(gradients[name])) > 0.0


def test_parameter_and_cache_counts_are_exact() -> None:
    core = MambaPolicyCore(input_dim=9, action_dim=4, hidden_size=17)
    params = core.initialize(jax.random.PRNGKey(15))
    hidden = core.hidden_size
    inner = 2 * hidden
    rank = math.ceil(hidden / 16)
    manual_count = (
        core.input_dim * hidden
        + hidden
        + hidden * 2 * inner
        + inner * 4
        + inner
        + inner * (rank + 2 * 16)
        + rank * inner
        + inner
        + inner * 16
        + inner
        + inner * hidden
        + hidden * core.action_dim
        + core.action_dim
        + hidden
        + 1
    )

    assert core.count_parameters(params) == manual_count
    assert core.expected_parameter_count() == manual_count
    assert (
        mamba_parameter_count(
            input_dim=core.input_dim,
            action_dim=core.action_dim,
            hidden_size=hidden,
        )
        == manual_count
    )

    batch_size = 7
    state = core.initial_state(batch_size)
    counts = core.cache_element_count(batch_size)
    assert counts == {
        "convolution": batch_size * inner * 4,
        "ssm": batch_size * inner * 16,
        "total": batch_size * inner * 20,
    }
    assert state.convolution.size == counts["convolution"]
    assert state.ssm.size == counts["ssm"]


def test_width_matcher_finds_the_globally_closest_integer_width() -> None:
    target_parameters = 48_321
    input_dim = 13
    action_dim = 6
    width = match_mamba_hidden_size(
        target_parameters=target_parameters,
        input_dim=input_dim,
        action_dim=action_dim,
        maximum_width=256,
    )
    selected_error = abs(
        mamba_parameter_count(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_size=width,
        )
        - target_parameters
    )
    all_errors = [
        abs(
            mamba_parameter_count(
                input_dim=input_dim,
                action_dim=action_dim,
                hidden_size=candidate,
            )
            - target_parameters
        )
        for candidate in range(1, 257)
    ]

    assert selected_error == min(all_errors)


def test_width_matcher_is_within_ten_percent_of_the_arcmind_target() -> None:
    input_dim = 128
    action_dim = 18
    target_core = ArcMindPolicyCore(
        ReferenceConfig(
            num_sensor_channels=input_dim,
            d_model=32,
            num_ssm_layers=2,
            ssm_state_dim=8,
            ssm_conv_width=3,
            ssm_expand_factor=1,
            num_attn_layers=1,
            num_attn_heads=4,
            attn_window_size=8,
            num_memory_slots=16,
            memory_compress_ratio=4,
            action_dim=action_dim,
            decision_stride=1,
        )
    )
    target_params = target_core.initialize(jax.random.PRNGKey(16))
    target_count = target_core.count_parameters(target_params)
    width = match_mamba_hidden_size(
        target_parameters=target_count,
        input_dim=input_dim,
        action_dim=action_dim,
    )
    mamba_count = mamba_parameter_count(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_size=width,
    )

    assert 0.9 <= mamba_count / target_count <= 1.1


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        (
            "target_parameters",
            {"target_parameters": 0, "input_dim": 3, "action_dim": 2},
        ),
        (
            "input_dim",
            {"target_parameters": 100, "input_dim": 0, "action_dim": 2},
        ),
        (
            "action_dim",
            {"target_parameters": 100, "input_dim": 3, "action_dim": 0},
        ),
        (
            "maximum_width",
            {
                "target_parameters": 100,
                "input_dim": 3,
                "action_dim": 2,
                "maximum_width": 0,
            },
        ),
    ],
)
def test_width_matcher_rejects_invalid_arguments(name, kwargs) -> None:
    with pytest.raises(ValueError, match=name):
        match_mamba_hidden_size(**kwargs)
