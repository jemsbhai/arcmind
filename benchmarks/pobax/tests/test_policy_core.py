"""WSL/JAX regression tests for the POBAX policy and learner interfaces."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.pobax.arcmind_reference import ReferenceConfig
from benchmarks.pobax.baseline_cores import (
    ElmanRNNPolicyCore,
    FrameStackMLPPolicyCore,
    GRUPolicyCore,
    LSTMPolicyCore,
    MemorylessMLPPolicyCore,
    MemoryTraceMLPPolicyCore,
    TCNPolicyCore,
    match_baseline_width,
)
from benchmarks.pobax.policy_core import ArcMindPolicyCore, augment_policy_input
from benchmarks.pobax.sequence_cores import (
    DiagonalSSMPolicyCore,
    FullCausalTransformerPolicyCore,
    LRUPolicyCore,
    S5RLPolicyCore,
    TransformerXLPolicyCore,
    match_sequence_width,
)
from benchmarks.pobax.shared_ppo import (
    categorical_entropy,
    categorical_log_probability,
    gaussian_entropy,
    gaussian_log_probability,
)


def tiny_arcmind_config() -> ReferenceConfig:
    return ReferenceConfig(
        num_sensor_channels=7,
        d_model=8,
        num_ssm_layers=1,
        ssm_state_dim=3,
        ssm_conv_width=3,
        ssm_expand_factor=1,
        num_attn_layers=1,
        num_attn_heads=2,
        attn_window_size=3,
        num_memory_slots=4,
        memory_compress_ratio=2,
        action_dim=3,
        decision_stride=1,
    )


def pilot_arcmind_config() -> ReferenceConfig:
    return ReferenceConfig(
        num_sensor_channels=7,
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
        action_dim=3,
        decision_stride=1,
    )


def test_augmented_input_zeroes_cross_episode_features() -> None:
    observation = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    policy_input = augment_policy_input(
        observation,
        jnp.asarray([2, 1]),
        jnp.asarray([5.0, 6.0]),
        jnp.asarray([True, False]),
        action_dim=3,
    )
    np.testing.assert_allclose(policy_input[0], [1, 2, 0, 0, 0, 0, 1])
    np.testing.assert_allclose(policy_input[1], [3, 4, 0, 1, 0, 6, 0])


def test_augmented_continuous_input_zeroes_cross_episode_features() -> None:
    policy_input = augment_policy_input(
        jnp.asarray([[1.0], [2.0]]),
        jnp.asarray([[0.5, -0.5], [0.25, 0.75]]),
        jnp.asarray([3.0, 4.0]),
        jnp.asarray([True, False]),
        action_dim=2,
        continuous_action=True,
    )
    np.testing.assert_allclose(policy_input[0], [1, 0, 0, 0, 1])
    np.testing.assert_allclose(policy_input[1], [2, 0.25, 0.75, 4, 0])


def test_parameter_matching_is_within_ten_percent() -> None:
    input_dim = 7
    action_dim = 3
    target_core = ArcMindPolicyCore(tiny_arcmind_config())
    target_count = target_core.count_parameters(
        target_core.initialize(jax.random.PRNGKey(1))
    )
    for model_name, constructor in (
        ("memoryless_mlp", MemorylessMLPPolicyCore),
        ("frame_stack_mlp", FrameStackMLPPolicyCore),
        ("memory_trace_mlp", MemoryTraceMLPPolicyCore),
        ("elman_rnn", ElmanRNNPolicyCore),
        ("gru", GRUPolicyCore),
        ("lstm", LSTMPolicyCore),
        ("tcn", TCNPolicyCore),
    ):
        width = match_baseline_width(
            model_name,
            target_parameters=target_count,
            input_dim=input_dim,
            action_dim=action_dim,
        )
        core = constructor(input_dim, action_dim, width)
        count = core.count_parameters(core.initialize(jax.random.PRNGKey(2)))
        assert 0.9 <= count / target_count <= 1.1


def test_sequence_parameter_matching_is_within_ten_percent() -> None:
    input_dim = 7
    action_dim = 3
    # The smallest GTrXL width is coarse because hidden size must be divisible
    # by the head count, so exercise the actual pilot-scale matching regime.
    target_core = ArcMindPolicyCore(pilot_arcmind_config())
    target_count = target_core.count_parameters(
        target_core.initialize(jax.random.PRNGKey(21))
    )
    for model_name in (
        "s4d",
        "ms4",
        "ms4n",
        "lru",
        "s5rl",
        "causal_transformer",
        "transformer_xl",
        "gtrxl",
    ):
        width = match_sequence_width(
            model_name,
            target_parameters=target_count,
            input_dim=input_dim,
            action_dim=action_dim,
            state_size=8,
            num_layers=1,
            num_heads=2,
        )
        if model_name in {"s4d", "ms4", "ms4n"}:
            core = DiagonalSSMPolicyCore(
                input_dim=input_dim,
                action_dim=action_dim,
                hidden_size=width,
                state_size=8,
                num_layers=1,
                variant=model_name,
            )
        elif model_name == "lru":
            core = LRUPolicyCore(
                input_dim=input_dim,
                action_dim=action_dim,
                hidden_size=width,
                num_layers=1,
            )
        elif model_name == "s5rl":
            core = S5RLPolicyCore(
                input_dim=input_dim,
                action_dim=action_dim,
                hidden_size=width,
                state_size=8,
                num_layers=1,
            )
        elif model_name in {"transformer_xl", "gtrxl"}:
            core = TransformerXLPolicyCore(
                input_dim=input_dim,
                action_dim=action_dim,
                hidden_size=width,
                num_heads=2,
                num_layers=1,
                memory_length=4,
                gated=model_name == "gtrxl",
            )
        else:
            core = FullCausalTransformerPolicyCore(
                input_dim=input_dim,
                action_dim=action_dim,
                hidden_size=width,
                num_heads=2,
                num_layers=1,
                window_length=4,
            )
        count = core.count_parameters(core.initialize(jax.random.PRNGKey(22)))
        assert 0.9 <= count / target_count <= 1.1


def _assert_sequence_matches_repeated_steps(core) -> None:
    params = core.initialize(jax.random.PRNGKey(3))
    inputs = jax.random.normal(jax.random.PRNGKey(4), (9, 4, 5))
    resets = jnp.zeros((9, 4), dtype=jnp.bool_).at[0].set(True)
    resets = resets.at[5, 2].set(True)
    initial_state = core.initial_state(4)
    sequence_state, sequence_logits, sequence_values = core.apply_sequence(
        params,
        initial_state,
        inputs,
        resets,
    )

    state = initial_state
    logits = []
    values = []
    for timestep in range(inputs.shape[0]):
        state, step_logits, step_values = core.step(
            params,
            state,
            inputs[timestep],
            resets[timestep],
        )
        logits.append(step_logits)
        values.append(step_values)
    # Fused scan and eager per-step GPU kernels may select different GEMM
    # algorithms. On the release GPU, equivalent structured cores differed by
    # at most 3e-3 after earlier compiled tests had populated the autotuning
    # cache. CPU execution and isolated GPU runs are more tightly aligned.
    parity_rtol = 1e-3
    parity_atol = 5e-3
    for sequence_leaf, step_leaf in zip(
        jax.tree.leaves(sequence_state),
        jax.tree.leaves(state),
        strict=True,
    ):
        np.testing.assert_allclose(
            sequence_leaf,
            step_leaf,
            rtol=parity_rtol,
            atol=parity_atol,
        )
    np.testing.assert_allclose(
        sequence_logits,
        jnp.stack(logits),
        rtol=parity_rtol,
        atol=parity_atol,
    )
    np.testing.assert_allclose(
        sequence_values,
        jnp.stack(values),
        rtol=parity_rtol,
        atol=parity_atol,
    )


def test_baseline_sequences_match_repeated_steps() -> None:
    for core in (
        ElmanRNNPolicyCore(input_dim=5, action_dim=3, hidden_size=7),
        GRUPolicyCore(input_dim=5, action_dim=3, hidden_size=7),
        LSTMPolicyCore(input_dim=5, action_dim=3, hidden_size=7),
        FrameStackMLPPolicyCore(
            input_dim=5,
            action_dim=3,
            hidden_size=7,
            stack_size=4,
        ),
        MemoryTraceMLPPolicyCore(
            input_dim=5,
            action_dim=3,
            hidden_size=7,
        ),
        TCNPolicyCore(
            input_dim=5,
            action_dim=3,
            hidden_size=7,
            num_layers=3,
            kernel_size=3,
        ),
    ):
        _assert_sequence_matches_repeated_steps(core)


def test_structured_sequences_match_repeated_steps() -> None:
    for core in (
        DiagonalSSMPolicyCore(
            input_dim=5,
            action_dim=3,
            hidden_size=8,
            state_size=8,
            num_layers=2,
            variant="s4d",
        ),
        DiagonalSSMPolicyCore(
            input_dim=5,
            action_dim=3,
            hidden_size=8,
            state_size=8,
            num_layers=2,
            variant="ms4n",
        ),
        LRUPolicyCore(
            input_dim=5,
            action_dim=3,
            hidden_size=8,
            num_layers=2,
        ),
        S5RLPolicyCore(
            input_dim=5,
            action_dim=3,
            hidden_size=8,
            state_size=8,
            num_layers=2,
        ),
        TransformerXLPolicyCore(
            input_dim=5,
            action_dim=3,
            hidden_size=8,
            num_heads=2,
            num_layers=2,
            memory_length=4,
        ),
        TransformerXLPolicyCore(
            input_dim=5,
            action_dim=3,
            hidden_size=8,
            num_heads=2,
            num_layers=2,
            memory_length=4,
            gated=True,
        ),
        FullCausalTransformerPolicyCore(
            input_dim=5,
            action_dim=3,
            hidden_size=8,
            num_heads=2,
            num_layers=2,
            window_length=4,
        ),
    ):
        _assert_sequence_matches_repeated_steps(core)


def test_structured_core_gradients_are_finite() -> None:
    inputs = jax.random.normal(jax.random.PRNGKey(31), (6, 2, 5))
    resets = jnp.zeros((6, 2), dtype=jnp.bool_).at[0].set(True)
    for core in (
        DiagonalSSMPolicyCore(
            input_dim=5,
            action_dim=3,
            hidden_size=8,
            state_size=8,
            num_layers=1,
            variant="s4d",
        ),
        LRUPolicyCore(
            input_dim=5,
            action_dim=3,
            hidden_size=8,
            num_layers=1,
        ),
        S5RLPolicyCore(
            input_dim=5,
            action_dim=3,
            hidden_size=8,
            state_size=8,
            num_layers=1,
        ),
        TransformerXLPolicyCore(
            input_dim=5,
            action_dim=3,
            hidden_size=8,
            num_heads=2,
            num_layers=1,
            memory_length=4,
            gated=True,
        ),
    ):
        params = core.initialize(jax.random.PRNGKey(32))

        def loss(candidate_params):
            _, logits, values = core.apply_sequence(
                candidate_params,
                core.initial_state(2),
                inputs,
                resets,
            )
            return jnp.mean(jnp.square(logits)) + jnp.mean(
                jnp.square(values)
            )

        gradients = jax.grad(loss)(params)
        leaves = jax.tree.leaves(gradients)
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
        assert any(bool(jnp.any(jnp.abs(leaf) > 0)) for leaf in leaves)


def test_finite_history_baselines_clear_each_environment_on_reset() -> None:
    for core in (
        FrameStackMLPPolicyCore(
            input_dim=5,
            action_dim=3,
            hidden_size=7,
            stack_size=4,
        ),
        TCNPolicyCore(
            input_dim=5,
            action_dim=3,
            hidden_size=7,
            num_layers=3,
            kernel_size=3,
        ),
        MemoryTraceMLPPolicyCore(
            input_dim=5,
            action_dim=3,
            hidden_size=7,
        ),
    ):
        params = core.initialize(jax.random.PRNGKey(13))
        state = core.initial_state(2)
        for frame in jax.random.normal(jax.random.PRNGKey(14), (8, 2, 5)):
            state, _, _ = core.step(
                params,
                state,
                frame,
                jnp.asarray([False, False]),
            )
        next_frame = jax.random.normal(jax.random.PRNGKey(15), (2, 5))
        reset_state, reset_logits, reset_values = core.step(
            params,
            state,
            next_frame,
            jnp.asarray([True, False]),
        )
        fresh_state, fresh_logits, fresh_values = core.step(
            params,
            core.initial_state(1),
            next_frame[:1],
            jnp.asarray([True]),
        )
        np.testing.assert_allclose(
            reset_logits[:1],
            fresh_logits,
            rtol=1e-3,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            reset_values[:1],
            fresh_values,
            rtol=1e-3,
            atol=1e-3,
        )
        for reset_leaf, fresh_leaf in zip(
            jax.tree.leaves(reset_state),
            jax.tree.leaves(fresh_state),
            strict=True,
        ):
            np.testing.assert_allclose(
                reset_leaf[:1],
                fresh_leaf,
                rtol=1e-3,
                atol=1e-3,
            )


def test_memory_trace_update_matches_registered_exponential_average() -> None:
    core = MemoryTraceMLPPolicyCore(
        input_dim=1,
        action_dim=2,
        hidden_size=4,
        decays=(0.0, 0.5),
    )
    params = core.initialize(jax.random.PRNGKey(16))
    state = core.initial_state(1)
    state, _, _ = core.step(
        params,
        state,
        jnp.asarray([[2.0]]),
        jnp.asarray([True]),
    )
    np.testing.assert_allclose(state.traces[0, :, 0], [2.0, 1.0])
    state, _, _ = core.step(
        params,
        state,
        jnp.asarray([[4.0]]),
        jnp.asarray([False]),
    )
    np.testing.assert_allclose(state.traces[0, :, 0], [4.0, 2.5])


def test_arcmind_reset_is_independent_per_environment() -> None:
    core = ArcMindPolicyCore(tiny_arcmind_config())
    params = core.initialize(jax.random.PRNGKey(5))
    state = core.initial_state(2)
    history = jax.random.normal(jax.random.PRNGKey(6), (5, 2, 7))
    for frame in history:
        state, _, _ = core.step(
            params,
            state,
            frame,
            jnp.asarray([False, False]),
        )

    next_frame = jax.random.normal(jax.random.PRNGKey(7), (2, 7))
    reset_state, reset_logits, reset_values = core.step(
        params,
        state,
        next_frame,
        jnp.asarray([True, False]),
    )
    fresh_state, fresh_logits, fresh_values = core.step(
        params,
        core.initial_state(1),
        next_frame[:1],
        jnp.asarray([True]),
    )
    np.testing.assert_allclose(reset_logits[:1], fresh_logits, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(reset_values[:1], fresh_values, rtol=5e-5, atol=5e-5)
    for reset_leaf, fresh_leaf in zip(
        jax.tree.leaves(reset_state),
        jax.tree.leaves(fresh_state),
        strict=True,
    ):
        np.testing.assert_allclose(
            reset_leaf[:1],
            fresh_leaf,
            rtol=1e-3,
            atol=1e-3,
        )


def test_arcmind_effective_parameter_count_tracks_ablations() -> None:
    full_core = ArcMindPolicyCore(pilot_arcmind_config())
    full_params = full_core.initialize(jax.random.PRNGKey(41))
    full_count = full_core.count_effective_parameters(full_params)
    assert full_count == full_core.count_parameters(full_params)

    ablation_fields = (
        "ablate_ssm",
        "ablate_attention",
        "ablate_memory",
        "ablate_gating",
        "ablate_temporal_encoding",
    )
    base = pilot_arcmind_config()
    for ablation_field in ablation_fields:
        ablated_config = replace(base, **{ablation_field: True})
        ablated_core = ArcMindPolicyCore(ablated_config)
        assert (
            ablated_core.count_effective_parameters(full_params)
            < full_count
        )


def test_categorical_statistics() -> None:
    logits = jnp.zeros((3, 2))
    actions = jnp.asarray([0, 1, 0])
    np.testing.assert_allclose(
        categorical_log_probability(logits, actions),
        -np.log(2.0),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        categorical_entropy(logits),
        np.log(2.0),
        rtol=1e-6,
    )


def test_gaussian_statistics() -> None:
    means = jnp.asarray([[0.0, 1.0], [2.0, -1.0]])
    log_standard_deviation = jnp.zeros((2,))
    actions = means
    np.testing.assert_allclose(
        gaussian_log_probability(means, log_standard_deviation, actions),
        -np.log(2.0 * np.pi),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        gaussian_entropy(log_standard_deviation),
        np.log(2.0 * np.pi * np.e),
        rtol=1e-6,
    )
