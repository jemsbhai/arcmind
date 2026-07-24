"""Run a development-only POBAX cell through the shared JAX PPO learner."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
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
    MemoryTraceSharedPolicyCore,
    TCNPolicyCore,
    match_baseline_width,
)
from benchmarks.pobax.ffm_core import (
    POPGYM_CONTEXT_SIZE,
    POPGYM_MAX_PERIOD,
    POPGYM_MEMORY_SIZE,
    POPGYM_MIN_PERIOD,
    FFMPolicyCore,
    match_ffm_hidden_size,
)
from benchmarks.pobax.implementation_provenance import gather_implementation_source
from benchmarks.pobax.mamba_core import MambaPolicyCore, match_mamba_hidden_size
from benchmarks.pobax.memory_trace_core import OfficialMemoryTracePolicyCore
from benchmarks.pobax.model_registry import (
    MAMBA1_REFERENCE_IMPLEMENTATION,
    MEMORY_TRACE_COMPATIBILITY_REFERENCE_IMPLEMENTATION,
    MEMORY_TRACE_OFFICIAL_REFERENCE_IMPLEMENTATION,
    MEMORY_TRACE_SHARED_REFERENCE_IMPLEMENTATION,
    PARAMETER_MATCHED_CONTRACT,
    POLICY_MODEL_IDS,
    policy_contract_metadata_for_model,
    validate_model_environment_contract,
    validate_model_evidence_tier,
    validate_policy_model_id,
)
from benchmarks.pobax.policy_core import ArcMindPolicyCore
from benchmarks.pobax.positional_mlp_core import (
    POPGYM_POSITIONAL_MLP_REFERENCE,
    PositionalMLPPolicyCore,
    match_positional_mlp_hidden_size,
)
from benchmarks.pobax.registered_artifacts import (
    atomic_write_json,
    canonical_json_sha256,
    dependency_lock_sha256,
    gather_git_provenance,
)
from benchmarks.pobax.registration_protocol import (
    COMPARISON_PROFILES,
    PUBLISHED_PRIMARY_TRAIN_STEPS,
    step_budget_mode,
)
from benchmarks.pobax.sequence_cores import (
    DiagonalSSMPolicyCore,
    FullCausalTransformerPolicyCore,
    LRUPolicyCore,
    S5RLPolicyCore,
    TransformerXLPolicyCore,
    match_sequence_width,
)
from benchmarks.pobax.shared_ppo import PPOConfig, SharedPPO
from benchmarks.pobax.shm_core import (
    POPGYM_SHM_MEMORY_SIZE,
    SHM_ADDRESS_ROWS,
    SHM_SOURCE_COMMIT,
    SHMPolicyCore,
    match_shm_hidden_size,
)
from benchmarks.pobax.smoke_environment import (
    PINNED_NAVIX_COMMIT,
    PINNED_POBAX_COMMIT,
    source_commit,
)
from benchmarks.pobax.upper_reference_envs import (
    BATTLESHIP_PERFECT_RECALL_SOURCE_CONTRACT,
    BattleshipPerfectRecallObservationWrapper,
)
from benchmarks.pobax.upper_reference_registry import UPPER_REFERENCE_SPECS

REFERENCE_IMPLEMENTATIONS = {
    "mamba1": MAMBA1_REFERENCE_IMPLEMENTATION,
    "memory_trace_official": MEMORY_TRACE_OFFICIAL_REFERENCE_IMPLEMENTATION,
    "memory_trace_shared": MEMORY_TRACE_SHARED_REFERENCE_IMPLEMENTATION,
    "memory_trace_mlp": MEMORY_TRACE_COMPATIBILITY_REFERENCE_IMPLEMENTATION,
    "ffm": {
        "repository": "https://github.com/proroklab/ffm",
        "audited_commit": "b3f94d2a0f35ba05089faf19ab1df846057cf8b6",
        "relationship": "shared-input and shared-head policy adaptation",
    },
    "positional_mlp": POPGYM_POSITIONAL_MLP_REFERENCE,
    "shm": {
        "repository": "https://github.com/thaihungle/SHM",
        "audited_commit": SHM_SOURCE_COMMIT,
        "source_variant": "popgym_policy_cell",
        "address_mode": "paper_uniform",
        "address_rows": SHM_ADDRESS_ROWS,
        "memory_size": POPGYM_SHM_MEMORY_SIZE,
        "address_replay": True,
        "relationship": "shared-input and shared-head JAX adaptation",
    },
    "shm_v1_1_popgym_compat": {
        "repository": "https://github.com/thaihungle/SHM",
        "audited_commit": SHM_SOURCE_COMMIT,
        "source_variant": "popgym_policy_cell",
        "address_mode": "v1_1_popgym_compat",
        "address_rows": SHM_ADDRESS_ROWS,
        "memory_size": POPGYM_SHM_MEMORY_SIZE,
        "address_replay": True,
        "relationship": "shared-input and shared-head JAX adaptation",
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
    "tmaze_10-perfect-memory": 1_000,
    "rocksample_11_11": 1_000,
    "rocksample_11_11-fully-observable": 1_000,
    "battleship_10": 1_000,
    "battleship_10-perfect-recall": 1_000,
    "HalfCheetah-P-v0": 1_000,
    "HalfCheetah-V-v0": 1_000,
    "HalfCheetah-F-v0": 1_000,
    "Walker-V-v0": 1_000,
    "Walker-F-v0": 1_000,
    "Navix-DMLab-Maze-01-v0": 2_000,
    "Navix-DMLab-Maze-01-fully-observable": 2_000,
}

ENVIRONMENT_SOURCES = {
    environment: specification["environment_source"]
    for environment, specification in UPPER_REFERENCE_SPECS.items()
}

UPPER_REFERENCE_TARGETS = {
    environment: specification["environment_reference"]
    for environment, specification in UPPER_REFERENCE_SPECS.items()
}

ENVIRONMENT_CONTRACTS = {
    "battleship_10-perfect-recall": BATTLESHIP_PERFECT_RECALL_SOURCE_CONTRACT,
    "HalfCheetah-P-v0": {
        "selected_observation_dimensions": [0, 1, 2, 3, 8, 9, 10, 11, 12],
        "action_bounds": [-1.0, 1.0],
    },
    "HalfCheetah-V-v0": {
        "selected_observation_dimensions": [4, 5, 6, 7, 13, 14, 15, 16],
        "action_bounds": [-1.0, 1.0],
    },
    "HalfCheetah-F-v0": {
        "selected_observation_dimensions": list(range(17)),
        "action_bounds": [-1.0, 1.0],
    },
    "Walker-V-v0": {
        "selected_observation_dimensions": list(range(8, 17)),
        "action_bounds": [-1.0, 1.0],
    },
    "Walker-F-v0": {
        "selected_observation_dimensions": list(range(17)),
        "action_bounds": [-1.0, 1.0],
    },
}

REGISTERED_TRAIN_STEPS = {
    "tmaze_10": 1_000_000,
    "tmaze_10-perfect-memory": 1_000_000,
    "rocksample_11_11": 5_000_000,
    "rocksample_11_11-fully-observable": 5_000_000,
    "battleship_10": 10_000_000,
    "battleship_10-perfect-recall": 10_000_000,
    "Walker-V-v0": 50_000_000,
    "Walker-F-v0": 50_000_000,
    "HalfCheetah-V-v0": 50_000_000,
    "HalfCheetah-F-v0": 50_000_000,
    "Navix-DMLab-Maze-01-v0": 10_000_000,
    "Navix-DMLab-Maze-01-fully-observable": 10_000_000,
}

EVIDENCE_STATUS = {
    "smoke": "development_smoke_not_for_paper",
    "pilot": "development_pilot_not_for_paper",
    "development_tuning": "development_tuning_not_for_paper",
    "registered_final": "registered_final_complete",
}

RUNTIME_DISTRIBUTIONS = (
    "brax",
    "gymnax",
    "jax",
    "jax-cuda12-pjrt",
    "jax-cuda12-plugin",
    "jaxlib",
    "Navix",
    "numpy",
    "optax",
    "pobax",
)


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
        attn_window_size=16,
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
    observation_dim: int | None = None,
    target_input_dim: int | None = None,
    max_episode_steps: int = 1_000,
):
    """Build ArcMind, a matched baseline, or an explicit supplemental core."""
    if target_input_dim is None:
        target_input_dim = input_dim
    if observation_dim is None:
        observation_dim = input_dim
    target_core = ArcMindPolicyCore(arcmind_config(target_input_dim, action_dim))
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

    if model_name == "memory_trace_official":
        core = OfficialMemoryTracePolicyCore(
            input_dim=input_dim,
            observation_dim=observation_dim,
            action_dim=action_dim,
        )
        params = core.initialize(random.PRNGKey(seed))
        return core, core.count_parameters(params), target_count

    if model_name == "ffm":
        width = match_ffm_hidden_size(
            target_parameters=target_count,
            input_dim=input_dim,
            action_dim=action_dim,
        )
    elif model_name == "positional_mlp":
        width = match_positional_mlp_hidden_size(
            target_parameters=target_count,
            input_dim=input_dim,
            action_dim=action_dim,
        )
    elif model_name in {"shm", "shm_v1_1_popgym_compat"}:
        width = match_shm_hidden_size(
            target_parameters=target_count,
            input_dim=input_dim,
            action_dim=action_dim,
        )
    elif model_name == "mamba1":
        width = match_mamba_hidden_size(
            target_parameters=target_count,
            input_dim=input_dim,
            action_dim=action_dim,
        )
    elif model_name in {
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
    elif model_name == "memory_trace_shared":
        core = MemoryTraceSharedPolicyCore(input_dim, action_dim, width)
    elif model_name == "memory_trace_mlp":
        core = MemoryTraceMLPPolicyCore(input_dim, action_dim, width)
    elif model_name == "tcn":
        core = TCNPolicyCore(input_dim, action_dim, width)
    elif model_name == "ffm":
        core = FFMPolicyCore(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_size=width,
            memory_size=POPGYM_MEMORY_SIZE,
            context_size=POPGYM_CONTEXT_SIZE,
            min_period=POPGYM_MIN_PERIOD,
            max_period=POPGYM_MAX_PERIOD,
        )
    elif model_name == "positional_mlp":
        core = PositionalMLPPolicyCore(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_size=width,
            max_length=max_episode_steps,
        )
    elif model_name in {"shm", "shm_v1_1_popgym_compat"}:
        core = SHMPolicyCore(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_size=width,
            memory_size=POPGYM_SHM_MEMORY_SIZE,
            address_mode=("paper_uniform" if model_name == "shm" else "v1_1_popgym_compat"),
        )
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
    elif model_name == "mamba1":
        core = MambaPolicyCore(input_dim, action_dim, width)
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
    return {name: value if np.isfinite(value) else None for name, value in metrics.items()}


def runtime_contract() -> dict[str, object]:
    """Return the stable installed-runtime identity used by a result cell."""
    devices = [
        {
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
        }
        for device in jax.devices()
    ]
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": {
            distribution: importlib.metadata.version(distribution)
            for distribution in RUNTIME_DISTRIBUTIONS
        },
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "devices": devices,
    }


def environment_horizon_and_gamma(
    environment: object,
    environment_params: object,
    environment_name: str,
) -> tuple[int, float]:
    """Resolve and verify the source-defined horizon and learner discount."""
    if not hasattr(environment_params, "max_steps_in_episode"):
        raise RuntimeError(f"{environment_name!r} parameters do not expose an episode horizon")
    horizon_value = np.asarray(environment_params.max_steps_in_episode)
    if horizon_value.size != 1:
        raise RuntimeError(f"{environment_name!r} episode horizon is not scalar")
    maximum_episode_steps = int(horizon_value.item())
    expected_horizon = MAX_EPISODE_STEPS.get(environment_name)
    if expected_horizon is None:
        raise RuntimeError(f"{environment_name!r} has no audited episode-horizon contract")
    if maximum_episode_steps != expected_horizon:
        raise RuntimeError(
            f"{environment_name!r} episode horizon drift: "
            f"expected={expected_horizon}, found={maximum_episode_steps}"
        )

    if not hasattr(environment, "gamma"):
        raise RuntimeError(f"{environment_name!r} does not expose its effective discount")
    gamma_value = np.asarray(environment.gamma)
    if gamma_value.size != 1:
        raise RuntimeError(f"{environment_name!r} discount is not scalar")
    gamma = float(gamma_value.item())
    if not np.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise RuntimeError(f"{environment_name!r} has invalid discount {gamma!r}")
    return maximum_episode_steps, gamma


def make_environment(
    environment_name: str,
    key: jax.Array,
    *,
    num_envs: int,
):
    """Construct a primary task or explicitly named upper-reference variant."""
    source = ENVIRONMENT_SOURCES.get(environment_name)
    if source is None:
        return get_env(environment_name, key, num_envs=num_envs)
    environment, environment_params = get_env(
        source["source_environment"],
        key,
        num_envs=num_envs,
        perfect_memory=source["perfect_memory"],
    )
    if environment_name == "battleship_10-perfect-recall":
        environment = BattleshipPerfectRecallObservationWrapper(environment)
    return environment, environment_params


def action_space_contract(action_space: object, *, label: str) -> tuple[bool, int]:
    """Return whether an action space is continuous and its flat dimension."""
    continuous = not hasattr(action_space, "n")
    if continuous:
        shape = getattr(action_space, "shape", None)
        if shape is None or len(shape) != 1:
            raise ValueError(f"Expected one-dimensional {label} Box actions, found {shape}")
        return True, int(shape[0])
    return False, int(action_space.n)


def validate_upper_reference_task_contract(
    *,
    upper_action_space: object,
    primary_action_space: object,
    upper_horizon: int,
    primary_horizon: int,
    upper_gamma: float,
    primary_gamma: float,
) -> int:
    """Prove that an upper reference changes information, not task dynamics."""
    upper_continuous, upper_action_dim = action_space_contract(
        upper_action_space,
        label="upper-reference",
    )
    primary_continuous, primary_action_dim = action_space_contract(
        primary_action_space,
        label="primary",
    )
    if primary_continuous != upper_continuous:
        raise RuntimeError(
            "Upper reference and primary target must use the same action-space class"
        )
    if primary_action_dim != upper_action_dim:
        raise RuntimeError(
            "Upper reference and primary target action dimensions differ: "
            f"upper={upper_action_dim}, primary={primary_action_dim}"
        )
    if primary_horizon != upper_horizon:
        raise RuntimeError(
            "Upper reference and primary target horizons differ: "
            f"upper={upper_horizon}, primary={primary_horizon}"
        )
    if not np.isclose(primary_gamma, upper_gamma, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            "Upper reference and primary target discounts differ: "
            f"upper={upper_gamma}, primary={primary_gamma}"
        )
    return primary_action_dim


def run(args: argparse.Namespace) -> dict[str, object]:
    """Train and evaluate one model/environment/seed cell."""
    validate_policy_model_id(args.model, field="model")
    validate_model_evidence_tier(args.model, args.evidence_tier, field="model")
    validate_model_environment_contract(
        args.model,
        args.environment,
        field="model",
    )
    if jax.default_backend() != "gpu" and args.require_gpu:
        raise RuntimeError(f"Expected GPU, found {jax.default_backend()!r}")
    commit = source_commit("pobax")
    if commit != PINNED_POBAX_COMMIT:
        raise RuntimeError(f"POBAX source drift: {commit}")
    navix_commit = source_commit("Navix")
    if navix_commit != PINNED_NAVIX_COMMIT:
        raise RuntimeError(f"Navix source drift: {navix_commit}")
    repository_root = Path(__file__).resolve().parents[2]
    git_provenance = gather_git_provenance(repository_root)
    if args.require_clean_git and git_provenance["dirty"]:
        raise RuntimeError("A clean Git worktree is required for this run")
    lock_path = repository_root / "benchmarks" / "pobax" / "requirements-lock.txt"
    lock_sha256 = dependency_lock_sha256(lock_path)
    installed_runtime = runtime_contract()

    total_steps = 8_192 if args.quick else args.total_steps
    num_envs = 32 if args.quick else args.num_envs
    rollout_steps = 32 if args.quick else args.rollout_steps
    update_epochs = 2 if args.quick else args.update_epochs
    candidate_id = getattr(args, "candidate_id", None)
    model_family = getattr(args, "model_family", None)
    tuning_aggregate_sha256 = getattr(args, "tuning_aggregate_sha256", None)
    tuning_completion_index_sha256 = getattr(
        args,
        "tuning_completion_index_sha256",
        None,
    )
    tuning_checksum_manifest_sha256 = getattr(
        args,
        "tuning_checksum_manifest_sha256",
        None,
    )
    tuning_implementation_source_sha256 = getattr(
        args,
        "tuning_implementation_source_sha256",
        None,
    )
    if args.registration_schema_version not in {1, 2, 3, 4}:
        raise ValueError("registration_schema_version must be 1, 2, 3, or 4")
    if args.registration_schema_version == 1:
        if args.comparison_profile is not None:
            raise ValueError("schema v1 does not accept a comparison_profile")
    elif args.comparison_profile not in COMPARISON_PROFILES:
        raise ValueError("schema v2, v3, and v4 require a supported comparison_profile")
    if args.registration_schema_version in {3, 4}:
        if (
            not isinstance(candidate_id, str)
            or not isinstance(model_family, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]*", model_family) is None
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]*", candidate_id) is None
            or not candidate_id.startswith(f"{model_family}.")
        ):
            raise ValueError(
                "schema v3 and v4 require a portable candidate ID prefixed by its model family"
            )
    elif candidate_id is not None or model_family is not None:
        raise ValueError("candidate identity is supported only by schema v3 and v4")
    if args.registration_schema_version == 3 and args.evidence_tier != "development_tuning":
        raise ValueError("schema v3 is reserved for development_tuning")
    if args.registration_schema_version == 4 and args.evidence_tier != "registered_final":
        raise ValueError("schema v4 is reserved for registered_final")
    if args.registration_schema_version == 4:
        tuning_hashes = {
            "aggregate": tuning_aggregate_sha256,
            "completion index": tuning_completion_index_sha256,
            "checksum manifest": tuning_checksum_manifest_sha256,
            "implementation source": tuning_implementation_source_sha256,
        }
        for label, value in tuning_hashes.items():
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"schema v4 requires a tuning {label} SHA256")
        if args.comparison_profile != "arcmind_shared_comparison":
            raise ValueError("schema v4 requires comparison_profile 'arcmind_shared_comparison'")
    elif any(
        value is not None
        for value in (
            tuning_aggregate_sha256,
            tuning_completion_index_sha256,
            tuning_checksum_manifest_sha256,
            tuning_implementation_source_sha256,
        )
    ):
        raise ValueError("tuning aggregate identity is supported only by schema v4")
    if args.quick and args.evidence_tier != "smoke":
        raise ValueError("--quick requires --evidence-tier smoke")
    if args.evidence_tier == "development_tuning":
        if args.registration_schema_version != 3:
            raise ValueError("development_tuning requires registration schema version 3")
        if args.comparison_profile != "arcmind_shared_comparison":
            raise ValueError(
                "development_tuning requires comparison_profile 'arcmind_shared_comparison'"
            )
        if args.quick:
            raise ValueError("development_tuning cannot use --quick")
        expected_steps = PUBLISHED_PRIMARY_TRAIN_STEPS.get(args.environment)
        if expected_steps is None:
            raise ValueError("development_tuning requires a published POBAX primary environment")
        if total_steps != expected_steps:
            raise ValueError(
                "development_tuning requires the published task budget: "
                f"expected={expected_steps}, found={total_steps}"
            )
        if git_provenance["dirty"]:
            raise RuntimeError("development_tuning requires a clean Git worktree")
        if args.matrix_manifest_sha256 is None and not args.describe_only:
            raise ValueError("development_tuning requires a frozen matrix manifest SHA256")
        if args.cell_id is None and not args.describe_only:
            raise ValueError("development_tuning requires a frozen cell ID")
    if args.evidence_tier == "registered_final":
        if args.quick:
            raise ValueError("registered_final cannot use --quick")
        is_upper_reference = args.environment in UPPER_REFERENCE_TARGETS
        if is_upper_reference and args.registration_schema_version == 4:
            raise ValueError("schema v4 is only for registered primary comparisons")
        if not is_upper_reference and args.registration_schema_version != 4:
            raise ValueError(
                "registered-final primary tasks require schema version 4 "
                "and an explicit tuning selection"
            )
        expected_steps = REGISTERED_TRAIN_STEPS.get(args.environment)
        if expected_steps is None:
            raise ValueError(f"{args.environment!r} is not a registered primary task")
        if total_steps != expected_steps:
            raise ValueError(
                "registered_final must use the published task budget: "
                f"expected={expected_steps}, found={total_steps}"
            )
        if git_provenance["dirty"]:
            raise RuntimeError("registered_final requires a clean Git worktree")
        if args.matrix_manifest_sha256 is None and not args.describe_only:
            raise ValueError("registered_final requires a frozen matrix manifest SHA256")
        if args.cell_id is None and not args.describe_only:
            raise ValueError("registered_final requires a frozen cell ID")
    implementation_source = (
        gather_implementation_source(repository_root)
        if args.registration_schema_version in {3, 4}
        else None
    )
    environment_key = random.PRNGKey(args.seed + 10)
    environment, environment_params = make_environment(
        args.environment,
        environment_key,
        num_envs=num_envs,
    )
    maximum_episode_steps, environment_gamma = environment_horizon_and_gamma(
        environment,
        environment_params,
        args.environment,
    )
    ppo_config = PPOConfig(
        total_steps=total_steps,
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        update_epochs=update_epochs,
        num_minibatches=args.num_minibatches,
        learning_rate=args.learning_rate,
        gamma=environment_gamma,
        gae_lambda=args.gae_lambda,
        entropy_coefficient=args.entropy_coefficient,
        anneal_learning_rate=args.anneal_learning_rate,
        step_budget_mode=step_budget_mode(args.comparison_profile),
    )
    ppo_config.validate()

    sample_observation, _ = environment.reset(
        random.split(random.PRNGKey(args.seed + 11), num_envs),
        environment_params,
    )
    observation_shape = tuple(int(value) for value in sample_observation.obs.shape[1:])
    observation_dim = int(np.prod(observation_shape))
    action_space = environment.action_space(environment_params)
    continuous_action, action_dim = action_space_contract(action_space, label="policy")
    if args.model == "memory_trace_official" and continuous_action:
        raise ValueError(
            "memory_trace_official has the official categorical actor and cannot "
            "serve as a continuous-action policy"
        )
    input_dim = observation_dim + action_dim + 2
    reference_metadata = UPPER_REFERENCE_TARGETS.get(args.environment)
    target_input_dim = input_dim
    if reference_metadata is not None:
        if args.model != "memoryless_mlp":
            raise ValueError("Full-observation upper references must use memoryless_mlp")
        target_environment, target_params = make_environment(
            reference_metadata["primary_environment"],
            random.PRNGKey(args.seed + 12),
            num_envs=num_envs,
        )
        target_observation, _ = target_environment.reset(
            random.split(random.PRNGKey(args.seed + 13), num_envs),
            target_params,
        )
        target_action_space = target_environment.action_space(target_params)
        target_maximum_episode_steps, target_gamma = environment_horizon_and_gamma(
            target_environment,
            target_params,
            reference_metadata["primary_environment"],
        )
        target_action_dim = validate_upper_reference_task_contract(
            upper_action_space=action_space,
            primary_action_space=target_action_space,
            upper_horizon=maximum_episode_steps,
            primary_horizon=target_maximum_episode_steps,
            upper_gamma=environment_gamma,
            primary_gamma=target_gamma,
        )
        target_observation_dim = int(np.prod(target_observation.obs.shape[1:]))
        target_input_dim = target_observation_dim + target_action_dim + 2
    policy_core, parameter_count, target_parameter_count = build_policy_core(
        args.model,
        input_dim=input_dim,
        action_dim=action_dim,
        seed=args.seed,
        observation_dim=observation_dim,
        target_input_dim=target_input_dim,
        max_episode_steps=maximum_episode_steps,
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
    policy_contract_metadata = policy_contract_metadata_for_model(args.model)
    if (
        policy_contract_metadata["parameter_contract"] == PARAMETER_MATCHED_CONTRACT
        and not 0.9 <= ratio <= 1.1
    ):
        raise RuntimeError(f"Parameter matching failed: ratio={ratio:.4f}")
    if args.evaluation_episodes_per_env < 1:
        raise ValueError("evaluation_episodes_per_env must be positive")
    evaluation_steps = args.evaluation_episodes_per_env * maximum_episode_steps
    frozen_configuration = {
        "schema_version": args.registration_schema_version,
        "evidence_tier": args.evidence_tier,
        "environment": args.environment,
        "environment_source": ENVIRONMENT_SOURCES.get(
            args.environment,
            {
                "source_environment": args.environment,
                "perfect_memory": False,
            },
        ),
        "model": candidate_id if args.registration_schema_version == 3 else args.model,
        "seed": args.seed,
        "policy_core": asdict(policy_core),
        "reference_implementation": REFERENCE_IMPLEMENTATIONS.get(args.model),
        "environment_reference": reference_metadata,
        "environment_contract": ENVIRONMENT_CONTRACTS.get(args.environment),
        "observation_shape": list(observation_shape),
        "policy_input_dim": input_dim,
        "parameter_target_policy_input_dim": target_input_dim,
        "parameter_count": parameter_count,
        "effective_parameter_count": effective_parameter_count,
        "arcmind_target_parameter_count": target_parameter_count,
        "parameter_ratio": ratio,
        **policy_contract_metadata,
        "action_dim": action_dim,
        "action_space": "continuous_box" if continuous_action else "discrete",
        "ppo": asdict(ppo_config),
        "evaluation_episodes_per_environment": args.evaluation_episodes_per_env,
        "evaluation_max_episode_steps": maximum_episode_steps,
        "pobax_commit": commit,
        "navix_commit": navix_commit,
        "dependency_lock_sha256": lock_sha256,
        "runtime_contract": installed_runtime,
    }
    if args.registration_schema_version in {3, 4}:
        frozen_configuration.update(
            {
                "candidate_id": candidate_id,
                "model_family": model_family,
                "implementation_model": args.model,
                "implementation_source": implementation_source,
            }
        )
    if args.registration_schema_version == 4:
        frozen_configuration.update(
            {
                "tuning_aggregate_sha256": tuning_aggregate_sha256,
                "tuning_completion_index_sha256": tuning_completion_index_sha256,
                "tuning_checksum_manifest_sha256": tuning_checksum_manifest_sha256,
                "tuning_implementation_source_sha256": (tuning_implementation_source_sha256),
            }
        )
    if args.registration_schema_version in {2, 3, 4}:
        frozen_configuration.update(
            {
                "comparison_profile": args.comparison_profile,
                "requested_environment_steps": total_steps,
                "realized_environment_steps": ppo_config.realized_steps,
            }
        )
    configuration_sha256 = canonical_json_sha256(frozen_configuration)
    if args.describe_only:
        return {
            "schema_version": 1,
            "status": "configuration_description",
            "configuration_sha256": configuration_sha256,
            "configuration": frozen_configuration,
            "runtime": {
                "jax": jax.__version__,
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "contract": installed_runtime,
                "git": git_provenance,
            },
        }

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
    evaluation = learner.evaluate(
        result.params,
        key=random.PRNGKey(args.seed + 50_000),
        episodes_per_environment=args.evaluation_episodes_per_env,
        evaluation_steps=evaluation_steps,
    )
    record: dict[str, object] = {
        "schema_version": (
            8
            if args.registration_schema_version == 4
            else 6
            if args.registration_schema_version == 3
            else 5
            if args.registration_schema_version == 2
            else 4
        ),
        "status": EVIDENCE_STATUS[args.evidence_tier],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration_sha256": configuration_sha256,
        "configuration": frozen_configuration,
        "matrix_manifest_sha256": args.matrix_manifest_sha256,
        "cell_id": args.cell_id,
        "provenance": {
            "git": git_provenance,
            "dependency_lock_sha256": lock_sha256,
            "pobax_commit": commit,
            "navix_commit": navix_commit,
            "runtime_contract": installed_runtime,
            **(
                {"implementation_source": implementation_source}
                if implementation_source is not None
                else {}
            ),
        },
        "environment": args.environment,
        "model": candidate_id if args.registration_schema_version == 3 else args.model,
        "seed": args.seed,
        "parameter_count": parameter_count,
        "effective_parameter_count": effective_parameter_count,
        "arcmind_target_parameter_count": target_parameter_count,
        "parameter_ratio": ratio,
        **policy_contract_metadata,
        "policy_core": asdict(policy_core),
        "reference_implementation": REFERENCE_IMPLEMENTATIONS.get(args.model),
        "environment_source": frozen_configuration["environment_source"],
        "environment_reference": reference_metadata,
        "environment_contract": ENVIRONMENT_CONTRACTS.get(args.environment),
        "observation_dim": observation_dim,
        "observation_shape": list(observation_shape),
        "policy_input_dim": input_dim,
        "parameter_target_policy_input_dim": target_input_dim,
        "action_dim": action_dim,
        "action_shape": list(action_space.shape) if continuous_action else [],
        "action_bounds": (
            {
                "low": np.asarray(action_space.low).tolist(),
                "high": np.asarray(action_space.high).tolist(),
            }
            if continuous_action
            else None
        ),
        "action_space": "continuous_box" if continuous_action else "discrete",
        "ppo": asdict(ppo_config),
        "actual_environment_steps": (ppo_config.num_updates * ppo_config.steps_per_update),
        "evaluation_episodes_per_environment": (args.evaluation_episodes_per_env),
        "evaluation_max_episode_steps": maximum_episode_steps,
        "actual_evaluation_steps_per_environment": evaluation_steps,
        "actual_evaluation_transitions": evaluation_steps * num_envs,
        "training_seconds": training_seconds,
        "training": finite_metrics(result.final_metrics),
        "training_history": [finite_metrics(metrics) for metrics in result.history],
        "evaluation": evaluation,
        "runtime": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "pobax_commit": commit,
            "navix_commit": navix_commit,
            "dependency_lock_sha256": lock_sha256,
            "git": git_provenance,
            "contract": installed_runtime,
        },
    }
    if args.registration_schema_version in {3, 4}:
        record.update(
            {
                "candidate_id": candidate_id,
                "model_family": model_family,
                "implementation_model": args.model,
                "implementation_source_sha256": implementation_source["sha256"],
            }
        )
    if args.registration_schema_version == 4:
        record.update(
            {
                "tuning_aggregate_sha256": tuning_aggregate_sha256,
                "tuning_completion_index_sha256": tuning_completion_index_sha256,
                "tuning_checksum_manifest_sha256": tuning_checksum_manifest_sha256,
                "tuning_implementation_source_sha256": (tuning_implementation_source_sha256),
            }
        )
    if args.registration_schema_version in {2, 3, 4}:
        record.update(
            {
                "comparison_profile": args.comparison_profile,
                "requested_environment_steps": total_steps,
                "realized_environment_steps": ppo_config.realized_steps,
            }
        )
    if args.output is not None:
        atomic_write_json(args.output, record)
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
        choices=POLICY_MODEL_IDS,
    )
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--total-steps", type=int, default=131_072)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--anneal-learning-rate", action="store_true")
    parser.add_argument(
        "--registration-schema-version",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
    )
    parser.add_argument("--candidate-id")
    parser.add_argument("--model-family")
    parser.add_argument("--tuning-aggregate-sha256")
    parser.add_argument("--tuning-completion-index-sha256")
    parser.add_argument("--tuning-checksum-manifest-sha256")
    parser.add_argument("--tuning-implementation-source-sha256")
    parser.add_argument(
        "--comparison-profile",
        choices=tuple(COMPARISON_PROFILES),
    )
    parser.add_argument(
        "--evaluation-episodes-per-env",
        type=int,
        default=4,
        help="fixed completed episodes retained from each vector environment",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument(
        "--evidence-tier",
        choices=tuple(EVIDENCE_STATUS),
        default="pilot",
    )
    parser.add_argument("--matrix-manifest-sha256")
    parser.add_argument("--cell-id")
    parser.add_argument("--describe-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
