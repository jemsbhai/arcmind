"""Run a development-only POBAX cell through the shared JAX PPO learner."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np
from jax import random
from pobax.envs import get_env

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
from benchmarks.pobax.policy_core import ArcMindPolicyCore
from benchmarks.pobax.sequence_cores import (
    DiagonalSSMPolicyCore,
    FullCausalTransformerPolicyCore,
    LRUPolicyCore,
    S5RLPolicyCore,
    TransformerXLPolicyCore,
    match_sequence_width,
)
from benchmarks.pobax.shared_ppo import PPOConfig, SharedPPO
from benchmarks.pobax.smoke_environment import (
    PINNED_POBAX_COMMIT,
    source_commit,
)

REFERENCE_IMPLEMENTATIONS = {
    "memory_trace_mlp": {
        "repository": "https://github.com/onnoeberhard/memory-traces",
        "audited_commit": "fcfdacc0b0a06dc181b49b9ef95893dbae7f2bcd",
        "relationship": "shared-input policy adaptation",
    },
    "s5rl": {
        "repository": "https://github.com/luchris429/popjaxrl",
        "audited_commit": "12e5d42be3a6bde81cce4234f8be4e119e4318b6",
        "relationship": "shared-input and shared-head policy adaptation",
    },
}

MAX_EPISODE_STEPS = {
    # Verified against the pinned POBAX environment parameters.
    "simple_chain": 10,
    "tmaze_10": 1_000,
    "rocksample_11_11": 1_000,
    "battleship_10": 1_000,
    "HalfCheetah-P-v0": 1_000,
    "HalfCheetah-V-v0": 1_000,
    "Navix-DMLab-Maze-01-v0": 2_000,
}


def arcmind_config(
    input_dim: int,
    action_dim: int,
    *,
    model_name: str = "arcmind",
) -> ReferenceConfig:
    """Compact pilot configuration; not a registered final hyperparameter."""
    ablations = {
        "arcmind": {},
        "arcmind_unordered": {"ablate_temporal_encoding": True},
        "arcmind_no_memory": {"ablate_memory": True},
        "arcmind_no_ssm": {"ablate_ssm": True},
        "arcmind_no_gate": {"ablate_gating": True},
        "arcmind_ssm_only": {"ablate_attention": True},
    }
    if model_name not in ablations:
        raise ValueError(f"Unsupported ArcMind variant: {model_name}")
    return ReferenceConfig(
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
        **ablations[model_name],
    )


def build_policy_core(
    model_name: str,
    *,
    input_dim: int,
    action_dim: int,
    seed: int,
):
    """Build ArcMind or a parameter-matched baseline."""
    target_core = ArcMindPolicyCore(arcmind_config(input_dim, action_dim))
    target_params = target_core.initialize(random.PRNGKey(seed))
    target_count = target_core.count_parameters(target_params)
    if model_name.startswith("arcmind"):
        core = ArcMindPolicyCore(
            arcmind_config(
                input_dim,
                action_dim,
                model_name=model_name,
            )
        )
        params = core.initialize(random.PRNGKey(seed))
        return core, core.count_parameters(params), target_count

    if model_name in {
        "s4d",
        "ms4",
        "ms4n",
        "lru",
        "s5rl",
        "causal_transformer",
        "transformer_xl",
        "gtrxl",
    }:
        width = match_sequence_width(
            model_name,
            target_parameters=target_count,
            input_dim=input_dim,
            action_dim=action_dim,
        )
    else:
        width = match_baseline_width(
            model_name,
            target_parameters=target_count,
            input_dim=input_dim,
            action_dim=action_dim,
        )
    if model_name == "memoryless_mlp":
        core = MemorylessMLPPolicyCore(input_dim, action_dim, width)
    elif model_name == "gru":
        core = GRUPolicyCore(input_dim, action_dim, width)
    elif model_name == "elman_rnn":
        core = ElmanRNNPolicyCore(input_dim, action_dim, width)
    elif model_name == "lstm":
        core = LSTMPolicyCore(input_dim, action_dim, width)
    elif model_name == "frame_stack_mlp":
        core = FrameStackMLPPolicyCore(input_dim, action_dim, width)
    elif model_name == "memory_trace_mlp":
        core = MemoryTraceMLPPolicyCore(input_dim, action_dim, width)
    elif model_name == "tcn":
        core = TCNPolicyCore(input_dim, action_dim, width)
    elif model_name in {"s4d", "ms4", "ms4n"}:
        core = DiagonalSSMPolicyCore(
            input_dim,
            action_dim,
            width,
            variant=model_name,
        )
    elif model_name == "lru":
        core = LRUPolicyCore(input_dim, action_dim, width)
    elif model_name == "s5rl":
        core = S5RLPolicyCore(input_dim, action_dim, width)
    elif model_name in {"transformer_xl", "gtrxl"}:
        core = TransformerXLPolicyCore(
            input_dim,
            action_dim,
            width,
            gated=model_name == "gtrxl",
        )
    elif model_name == "causal_transformer":
        core = FullCausalTransformerPolicyCore(
            input_dim,
            action_dim,
            width,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    params = core.initialize(random.PRNGKey(seed))
    return core, core.count_parameters(params), target_count


def finite_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    """Replace non-finite development metrics with valid JSON null values."""
    return {
        name: value if np.isfinite(value) else None
        for name, value in metrics.items()
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    """Train and evaluate one model/environment/seed cell."""
    if jax.default_backend() != "gpu" and args.require_gpu:
        raise RuntimeError(f"Expected GPU, found {jax.default_backend()!r}")
    commit = source_commit("pobax")
    if commit != PINNED_POBAX_COMMIT:
        raise RuntimeError(f"POBAX source drift: {commit}")

    total_steps = 8_192 if args.quick else args.total_steps
    num_envs = 32 if args.quick else args.num_envs
    rollout_steps = 32 if args.quick else args.rollout_steps
    update_epochs = 2 if args.quick else args.update_epochs
    num_minibatches = 4
    ppo_config = PPOConfig(
        total_steps=total_steps,
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        update_epochs=update_epochs,
        num_minibatches=num_minibatches,
        learning_rate=args.learning_rate,
    )

    environment_key = random.PRNGKey(args.seed + 10)
    environment, environment_params = get_env(
        args.environment,
        environment_key,
        num_envs=num_envs,
    )
    sample_observation, _ = environment.reset(
        random.split(random.PRNGKey(args.seed + 11), num_envs),
        environment_params,
    )
    observation_dim = int(np.prod(sample_observation.obs.shape[1:]))
    action_space = environment.action_space(environment_params)
    continuous_action = not hasattr(action_space, "n")
    if continuous_action:
        if len(action_space.shape) != 1:
            raise ValueError(
                f"Expected one-dimensional Box actions, found {action_space.shape}"
            )
        action_dim = int(action_space.shape[0])
    else:
        action_dim = int(action_space.n)
    input_dim = observation_dim + action_dim + 2
    policy_core, parameter_count, target_parameter_count = build_policy_core(
        args.model,
        input_dim=input_dim,
        action_dim=action_dim,
        seed=args.seed,
    )
    if continuous_action:
        # Every continuous policy learns the same state-independent diagonal
        # log standard deviation in addition to its mean-producing core.
        parameter_count += action_dim
        target_parameter_count += action_dim
    initialized_params = policy_core.initialize(random.PRNGKey(args.seed))
    effective_parameter_count = (
        policy_core.count_effective_parameters(initialized_params)
        if hasattr(policy_core, "count_effective_parameters")
        else parameter_count - (action_dim if continuous_action else 0)
    )
    if continuous_action:
        effective_parameter_count += action_dim
    ratio = parameter_count / target_parameter_count
    if not 0.9 <= ratio <= 1.1:
        raise RuntimeError(f"Parameter matching failed: ratio={ratio:.4f}")

    learner = SharedPPO(
        policy_core=policy_core,
        environment=environment,
        environment_params=environment_params,
        action_dim=action_dim,
        config=ppo_config,
        continuous_action=continuous_action,
    )

    def progress(update: int, metrics: dict[str, float]) -> None:
        if update == 1 or update == ppo_config.num_updates or update % 10 == 0:
            print(
                f"update={update}/{ppo_config.num_updates} "
                f"steps={int(metrics['environment_steps'])} "
                f"return={metrics['mean_recent_return']:.4f} "
                f"loss={metrics['loss']:.4f}",
                flush=True,
            )

    start = time.perf_counter()
    result = learner.train(
        parameter_key=random.PRNGKey(args.seed),
        runner_key=random.PRNGKey(args.seed + 1),
        shuffle_key=random.PRNGKey(args.seed + 2),
        progress=progress,
    )
    jax.block_until_ready(result.params)
    training_seconds = time.perf_counter() - start
    if args.evaluation_episodes_per_env < 1:
        raise ValueError("evaluation_episodes_per_env must be positive")
    maximum_episode_steps = MAX_EPISODE_STEPS[args.environment]
    evaluation_steps = (
        args.evaluation_episodes_per_env * maximum_episode_steps
    )
    evaluation = learner.evaluate(
        result.params,
        key=random.PRNGKey(args.seed + 50_000),
        episodes_per_environment=args.evaluation_episodes_per_env,
        evaluation_steps=evaluation_steps,
    )
    record: dict[str, object] = {
        "schema_version": 3,
        "status": "development_pilot_not_for_paper",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": args.environment,
        "model": args.model,
        "seed": args.seed,
        "parameter_count": parameter_count,
        "effective_parameter_count": effective_parameter_count,
        "arcmind_target_parameter_count": target_parameter_count,
        "parameter_ratio": ratio,
        "policy_core": asdict(policy_core),
        "reference_implementation": REFERENCE_IMPLEMENTATIONS.get(args.model),
        "observation_dim": observation_dim,
        "policy_input_dim": input_dim,
        "action_dim": action_dim,
        "action_space": "continuous_box" if continuous_action else "discrete",
        "ppo": asdict(ppo_config),
        "actual_environment_steps": (
            ppo_config.num_updates * ppo_config.steps_per_update
        ),
        "evaluation_episodes_per_environment": (
            args.evaluation_episodes_per_env
        ),
        "evaluation_max_episode_steps": maximum_episode_steps,
        "actual_evaluation_steps_per_environment": evaluation_steps,
        "actual_evaluation_transitions": evaluation_steps * num_envs,
        "training_seconds": training_seconds,
        "training": finite_metrics(result.final_metrics),
        "evaluation": evaluation,
        "runtime": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "pobax_commit": commit,
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        default="tmaze_10",
        choices=tuple(MAX_EPISODE_STEPS),
    )
    parser.add_argument(
        "--model",
        default="arcmind",
        choices=(
            "memoryless_mlp",
            "frame_stack_mlp",
            "memory_trace_mlp",
            "elman_rnn",
            "gru",
            "lstm",
            "tcn",
            "s4d",
            "ms4",
            "ms4n",
            "lru",
            "s5rl",
            "causal_transformer",
            "transformer_xl",
            "gtrxl",
            "arcmind_unordered",
            "arcmind_no_memory",
            "arcmind_no_ssm",
            "arcmind_no_gate",
            "arcmind_ssm_only",
            "arcmind",
        ),
    )
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--total-steps", type=int, default=131_072)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument(
        "--evaluation-episodes-per-env",
        type=int,
        default=4,
        help="fixed completed episodes retained from each vector environment",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
