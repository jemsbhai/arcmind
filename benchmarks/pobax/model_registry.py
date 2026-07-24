"""Pure metadata registry for policy implementations in the POBAX harness."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

AGALITE_REPOSITORY = "https://github.com/subho406/agalite"
AGALITE_AUDITED_COMMIT = "101acbecc121a258ad8f7e58e2f782f546674979"
AGALITE_LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
AGALITE_REQUIREMENTS_SHA256 = "6f22351a257748fe8c5b9ebe1215a5186557af50647f86900b7992c4d055a01f"
AGALITE_MODEL_SHA256 = "00d921f46740e43aed9e444c51852b0e7fdb80cd489550e1c815d2c70e00a89b"
AGALITE_LAYERS_SHA256 = "610a703e3da2736fef14a2d72545a04143fcac4a9823ce707c7301fecd2e8978"
AGALITE_SEQUENCE_FACTORY_SHA256 = (
    "d4c8fff77d316a97f909886d1a12c53485ad652e49cf927ee65aaf1fdb4e01d8"
)
AGALITE_ACTOR_CRITIC_SHA256 = (
    "6fea2b0c7dd20047a187f60c25a6ed697ce079e98cde3d96d6d6c957c24be9fd"
)
AGALITE_HEADS_SHA256 = "fe02f29c68fd1f841b3998ee1df1479012a85cf97a9ae29d0d6ab45bc6700b41"
AGALITE_FLATTEN_SHA256 = "6f3304bbb7e0553faff6e500c3ac6f656c53d10e79ec6d1ba62f38f74e8c3a0e"
AGALITE_A2C_SHA256 = "7dcb9805d70f4978b0788a2460e169f4ddad1112720a9474abd46bf78f860719"
AGALITE_TMAZE_CONFIG_SHA256 = (
    "feb7a4ab63dae47c338f651b2f6aedbc4e20ac6b802670aed05b039ef3c921d9"
)
AGALITE_DIFFERENTIAL_FIXTURE_SHA256 = (
    "3ab4f79c45168cc8ac8e53dfe4cf89a343f9323d58d83aec31256422927797c1"
)
AGALITE_NUM_LAYERS = 4
AGALITE_MODEL_SIZE = 128
AGALITE_HEAD_SIZE = 64
AGALITE_FEEDFORWARD_SIZE = 128
AGALITE_NUM_HEADS = 4
AGALITE_ETA = 4
AGALITE_APPROXIMATION_CHANNELS = 2
AGALITE_GATE_BIAS = 2.0
AGALITE_ATTENTION_EPSILON = 1e-5
AGALITE_LAYER_NORM_EPSILON = 1e-6

MAMBA_REPOSITORY = "https://github.com/state-spaces/mamba"
MAMBA_VERSION = "2.2.6.post3"
MAMBA_AUDITED_COMMIT = "10b5d6358f27966f6a40e4bf0baa17a460688128"
MAMBA_SOURCE_PATH = "mamba_ssm/modules/mamba_simple.py"
MAMBA_SIMPLE_SHA256 = "a17e4c51b582dc0d4d690a649eba521cd0c1ee3dc8f0473a0967cdc9ec0874e3"
MAMBA_BLOCK_SHA256 = "b62e755195c277a027c5d9cc8d576a8ae4a1d1317143b91370b2f8ce683b4cc1"
MAMBA_MIXER_MODEL_SHA256 = "13409d7044e930ea3271e4b8ddceaf8155ec49b8e5ac299fba7bb0df6d80cb21"
MAMBA_RMSNORM_SHA256 = "006fb18f7098fc244a318c899841ad4c1a6ea0f614dfe7a1feb4e2e38185235f"
MAMBA_CONFIG_SHA256 = "2a72c1686f775b56547e39ca4406ba10148d12fd7a791c57ce2ba85126010fcd"
MAMBA_PARITY_FIXTURE_SHA256 = (
    "8bfa948c8c1fd28bcde3e7dd7eebff8bb5e54406dd1fd11f7d69317f1c6e3015"
)

MEMORY_TRACE_REPOSITORY = "https://github.com/onnoeberhard/memory-traces"
MEMORY_TRACE_AUDITED_COMMIT = "fcfdacc0b0a06dc181b49b9ef95893dbae7f2bcd"
MEMORY_TRACE_SOURCE_SHA256 = (
    "e01aa53aa5e72e6890c2942ae77892786afe3d1f9256d5c378da1076fdfee541"
)
MEMORY_TRACE_EXAMPLE_SHA256 = (
    "841bdd3d62ce4143f149b5a5fe1c18ec37c1a96def1594ba2faa04d466bae88f"
)
MEMORY_TRACE_DECAYS = (0.0, 0.985)
MEMORY_TRACE_DECAY_ORIGIN = (
    "official_tmaze64_example_only_not_author_selected_for_pobax"
)
MEMORY_TRACE_DIFFERENTIAL_FIXTURE_SHA256 = (
    "740d2ec4a27a9d89cf000d103320a26bac7e936ef793a094829f5d34385c61ab"
)

PARAMETER_MATCHED_CONTRACT = "arcmind_parameter_matched_0.9_to_1.1"
FIXED_OFFICIAL_PARAMETER_CONTRACT = "fixed_official_architecture"
PRIMARY_COMPARISON_ROLE = "parameter_matched_primary"
SUPPLEMENTAL_COMPARISON_ROLE = "supplemental_source_compatible"
DEVELOPMENT_COMPATIBILITY_ROLE = "development_compatibility_only"
_CONTINUOUS_POBAX_ENVIRONMENTS = frozenset(
    {
        "HalfCheetah-P-v0",
        "HalfCheetah-V-v0",
        "HalfCheetah-F-v0",
        "Walker-V-v0",
        "Walker-F-v0",
    }
)

POLICY_MODEL_IDS = (
    "memoryless_mlp",
    "frame_stack_mlp",
    "agalite_source_compat",
    "agalite_shared",
    "memory_trace_mlp",
    "memory_trace_official",
    "memory_trace_shared",
    "positional_mlp",
    "shm",
    "shm_v1_1_popgym_compat",
    "elman_rnn",
    "gru",
    "lstm",
    "tcn",
    "ffm",
    "s4d",
    "ms4",
    "ms4n",
    "lru",
    "s5rl",
    "mamba1",
    "causal_transformer",
    "transformer_xl",
    "gtrxl",
    "arcmind_unordered",
    "arcmind_no_memory",
    "arcmind_no_ssm",
    "arcmind_no_gate",
    "arcmind_ssm_only",
    "arcmind",
)
_POLICY_MODEL_ID_SET = frozenset(POLICY_MODEL_IDS)

MAMBA1_REFERENCE_IMPLEMENTATION: dict[str, Any] = {
    "repository": MAMBA_REPOSITORY,
    "version": MAMBA_VERSION,
    "audited_commit": MAMBA_AUDITED_COMMIT,
    "source_variant": "one_block_mamba1_slow_step",
    "source_hashes": {
        "mamba_ssm/modules/mamba_simple.py": MAMBA_SIMPLE_SHA256,
        "mamba_ssm/modules/block.py": MAMBA_BLOCK_SHA256,
        "mamba_ssm/models/mixer_seq_simple.py": MAMBA_MIXER_MODEL_SHA256,
        "mamba_ssm/models/config_mamba.py": MAMBA_CONFIG_SHA256,
        "mamba_ssm/ops/triton/layer_norm.py": MAMBA_RMSNORM_SHA256,
    },
    "parity_fixture": {
        "path": "benchmarks/pobax/tests/fixtures/mamba1_official_step_v1.json",
        "sha256": MAMBA_PARITY_FIXTURE_SHA256,
        "execution_path": "Mamba.step dependency-light PyTorch slow path",
    },
    "relationship": "shared-input and shared-head JAX adaptation",
}

_AGALITE_EXECUTABLE_CONTRACT: dict[str, Any] = {
    "finite_channel_count": AGALITE_APPROXIMATION_CHANNELS,
    "frequency_grid": "linspace(-pi,pi,R)_inclusive",
    "initial_tick": 1.0,
    "first_used_tick": 2.0,
    "phase_reset": "never",
    "reset_order": "clear_prior_memory_then_incorporate_current_token",
    "normalizer": "2*R*dot(s,q)+1e-5",
    "attention_epsilon": AGALITE_ATTENTION_EPSILON,
    "layer_norm_epsilon": AGALITE_LAYER_NORM_EPSILON,
    "feature_layout": "projection_major_eta_then_head_dimension",
    "gate_count_per_layer": 2,
    "gate_bias": AGALITE_GATE_BIAS,
}

_AGALITE_COMMON_REFERENCE: dict[str, Any] = {
    "repository": AGALITE_REPOSITORY,
    "audited_commit": AGALITE_AUDITED_COMMIT,
    "license": "Apache-2.0",
    "source_hashes": {
        "LICENSE": AGALITE_LICENSE_SHA256,
        "requirements.txt": AGALITE_REQUIREMENTS_SHA256,
        "src/models/agalite/agalite.py": AGALITE_MODEL_SHA256,
        "src/models/agalite/layers.py": AGALITE_LAYERS_SHA256,
        "src/model_fns/seq_fns.py": AGALITE_SEQUENCE_FACTORY_SHA256,
        "src/models/actor_critic.py": AGALITE_ACTOR_CRITIC_SHA256,
        "src/model_fns/achead_fns.py": AGALITE_HEADS_SHA256,
        "src/model_fns/repr_fns.py": AGALITE_FLATTEN_SHA256,
        "src/agents/a2c.py": AGALITE_A2C_SHA256,
        "config/tmaze/arelit.yaml": AGALITE_TMAZE_CONFIG_SHA256,
    },
    "executable_contract": _AGALITE_EXECUTABLE_CONTRACT,
    "differential_fixture": {
        "path": "benchmarks/pobax/tests/fixtures/agalite_official_v1.json",
        "sha256": AGALITE_DIFFERENTIAL_FIXTURE_SHA256,
        "execution_path": (
            "official Flax AGaLiTe and T-Maze heads with translated deterministic weights"
        ),
    },
    "dependency_contract": (
        "upstream requirements are unpinned; fixture records exact JAX and Flax versions"
    ),
}

AGALITE_SOURCE_COMPAT_REFERENCE_IMPLEMENTATION: dict[str, Any] = {
    **_AGALITE_COMMON_REFERENCE,
    "source_variant": "released_tmaze_vector_policy",
    "input_contract": "flattened_observation_only",
    "architecture": {
        "num_layers": AGALITE_NUM_LAYERS,
        "model_size": AGALITE_MODEL_SIZE,
        "head_size": AGALITE_HEAD_SIZE,
        "feedforward_size": AGALITE_FEEDFORWARD_SIZE,
        "num_heads": AGALITE_NUM_HEADS,
        "eta": AGALITE_ETA,
        "approximation_channels": AGALITE_APPROXIMATION_CHANNELS,
    },
    "actor_critic": {
        "shared_parameters": False,
        "actor_hidden_size": 128,
        "critic_hidden_size": 128,
        "activation": "tanh",
        "all_kernel_orthogonal_gain": "sqrt(2)",
        "categorical_actor": True,
    },
    "author_learner": "A2C",
    "integration_learner": "shared_PPO",
    "parameter_contract": FIXED_OFFICIAL_PARAMETER_CONTRACT,
    "comparison_role": SUPPLEMENTAL_COMPARISON_ROLE,
    "relationship": (
        "released T-Maze vector policy inside the shared POBAX PPO learner"
    ),
}

AGALITE_SHARED_REFERENCE_IMPLEMENTATION: dict[str, Any] = {
    **_AGALITE_COMMON_REFERENCE,
    "source_variant": "released_recurrence_shared_policy",
    "input_contract": "shared_augmented_policy_input",
    "architecture": {
        "num_layers": AGALITE_NUM_LAYERS,
        "num_heads": AGALITE_NUM_HEADS,
        "head_size": "model_size/2",
        "feedforward_size": "model_size",
        "eta": AGALITE_ETA,
        "approximation_channels": AGALITE_APPROXIMATION_CHANNELS,
    },
    "actor_critic": {
        "shared_common_heads": True,
        "categorical_or_gaussian_mean": True,
    },
    "author_learner": "A2C_for_released_T-Maze_configuration",
    "integration_learner": "shared_PPO",
    "parameter_contract": PARAMETER_MATCHED_CONTRACT,
    "comparison_role": PRIMARY_COMPARISON_ROLE,
    "relationship": "shared-augmented-input and shared-head JAX adaptation",
}

_MEMORY_TRACE_COMMON_REFERENCE: dict[str, Any] = {
    "repository": MEMORY_TRACE_REPOSITORY,
    "audited_commit": MEMORY_TRACE_AUDITED_COMMIT,
    "source_hashes": {
        "traces/ppo.py": MEMORY_TRACE_SOURCE_SHA256,
        "examples/ppo_tmaze.py": MEMORY_TRACE_EXAMPLE_SHA256,
    },
    "decays": list(MEMORY_TRACE_DECAYS),
    "decay_origin": MEMORY_TRACE_DECAY_ORIGIN,
    "decay_scope": "The official values configure TMaze64, not POBAX.",
}

MEMORY_TRACE_OFFICIAL_REFERENCE_IMPLEMENTATION: dict[str, Any] = {
    **_MEMORY_TRACE_COMMON_REFERENCE,
    "source_variant": "observation_only_non_episodic_trace",
    "trace_layout": "trace_major_flattening",
    "reset_order": "reset_worker_then_incorporate_current_observation",
    "actor_critic": {
        "shared_parameters": False,
        "hidden_sizes": [64, 64],
        "activation": "tanh",
        "hidden_orthogonal_gain": "sqrt(2)",
        "actor_output_orthogonal_gain": 0.01,
        "critic_output_orthogonal_gain": 1.0,
    },
    "parameter_contract": FIXED_OFFICIAL_PARAMETER_CONTRACT,
    "comparison_role": SUPPLEMENTAL_COMPARISON_ROLE,
    "differential_fixture": {
        "path": "benchmarks/pobax/tests/fixtures/memory_trace_official_v1.json",
        "sha256": MEMORY_TRACE_DIFFERENTIAL_FIXTURE_SHA256,
        "execution_path": (
            "official Trace and ActorCritic with translated official-initialized weights"
        ),
    },
    "relationship": "source-compatible policy core inside the shared POBAX PPO learner",
}

MEMORY_TRACE_SHARED_REFERENCE_IMPLEMENTATION: dict[str, Any] = {
    **_MEMORY_TRACE_COMMON_REFERENCE,
    "source_variant": "non_episodic_trace_recurrence",
    "trace_layout": "trace_major_flattening",
    "reset_order": "reset_worker_then_incorporate_current_augmented_policy_input",
    "parameter_contract": PARAMETER_MATCHED_CONTRACT,
    "comparison_role": PRIMARY_COMPARISON_ROLE,
    "relationship": "shared-augmented-input and shared-trunk JAX adaptation",
}

MEMORY_TRACE_COMPATIBILITY_REFERENCE_IMPLEMENTATION: dict[str, Any] = {
    **MEMORY_TRACE_SHARED_REFERENCE_IMPLEMENTATION,
    "comparison_role": DEVELOPMENT_COMPATIBILITY_ROLE,
    "relationship": "legacy development alias for memory_trace_shared",
}

_REQUIRED_REFERENCE_IMPLEMENTATIONS = {
    "agalite_source_compat": AGALITE_SOURCE_COMPAT_REFERENCE_IMPLEMENTATION,
    "agalite_shared": AGALITE_SHARED_REFERENCE_IMPLEMENTATION,
    "mamba1": MAMBA1_REFERENCE_IMPLEMENTATION,
    "memory_trace_official": MEMORY_TRACE_OFFICIAL_REFERENCE_IMPLEMENTATION,
    "memory_trace_shared": MEMORY_TRACE_SHARED_REFERENCE_IMPLEMENTATION,
    "memory_trace_mlp": MEMORY_TRACE_COMPATIBILITY_REFERENCE_IMPLEMENTATION,
}

_MODEL_PARAMETER_CONTRACTS = {
    "agalite_source_compat": FIXED_OFFICIAL_PARAMETER_CONTRACT,
    "memory_trace_official": FIXED_OFFICIAL_PARAMETER_CONTRACT,
}
_MODEL_COMPARISON_ROLES = {
    "agalite_source_compat": SUPPLEMENTAL_COMPARISON_ROLE,
    "memory_trace_official": SUPPLEMENTAL_COMPARISON_ROLE,
    "memory_trace_mlp": DEVELOPMENT_COMPATIBILITY_ROLE,
}


def validate_policy_model_id(value: object, *, field: str) -> str:
    """Return one registered implementation ID or reject it."""

    if not isinstance(value, str) or value not in _POLICY_MODEL_ID_SET:
        raise ValueError(
            f"{field} must be one of the registered policy implementations: "
            f"{list(POLICY_MODEL_IDS)}"
        )
    return value


def reference_implementation_for_model(model: str) -> dict[str, Any] | None:
    """Return an isolated copy of required source metadata for one model."""

    expected = _REQUIRED_REFERENCE_IMPLEMENTATIONS.get(model)
    return deepcopy(expected) if expected is not None else None


def parameter_contract_for_model(model: str) -> str:
    """Return the immutable parameter-accounting rule for one model."""

    return _MODEL_PARAMETER_CONTRACTS.get(model, PARAMETER_MATCHED_CONTRACT)


def comparison_role_for_model(model: str) -> str:
    """Return the reporting role that controls aggregate separation."""

    return _MODEL_COMPARISON_ROLES.get(model, PRIMARY_COMPARISON_ROLE)


def policy_contract_metadata_for_model(model: str) -> dict[str, Any]:
    """Return fields frozen into configurations and artifacts."""

    metadata: dict[str, Any] = {
        "parameter_contract": parameter_contract_for_model(model),
        "comparison_role": comparison_role_for_model(model),
    }
    if model in {
        "memory_trace_official",
        "memory_trace_shared",
        "memory_trace_mlp",
    }:
        metadata.update(
            {
                "memory_trace_decays": list(MEMORY_TRACE_DECAYS),
                "memory_trace_decay_origin": MEMORY_TRACE_DECAY_ORIGIN,
            }
        )
    if model in {"agalite_source_compat", "agalite_shared"}:
        metadata["agalite_executable_contract"] = deepcopy(
            _AGALITE_EXECUTABLE_CONTRACT
        )
    return metadata


def requires_explicit_policy_contract(model: str) -> bool:
    """Return whether legacy omission is prohibited for this model."""

    return model in {
        "agalite_source_compat",
        "agalite_shared",
        "memory_trace_official",
        "memory_trace_shared",
        "memory_trace_mlp",
    }


def validate_policy_contract_metadata(
    model: str,
    value: object,
    *,
    field: str,
) -> dict[str, Any]:
    """Validate the exact serialized comparison and parameter contract."""

    expected = policy_contract_metadata_for_model(model)
    if value != expected:
        raise ValueError(
            f"{field} does not match the registered policy contract for model {model!r}"
        )
    return deepcopy(expected)


def validate_policy_core_contract(model: str, value: object, *, field: str) -> None:
    """Validate source-audited architecture fields for registered lanes."""

    if not requires_explicit_policy_contract(model):
        return
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object for model {model!r}")
    if model in {"agalite_source_compat", "agalite_shared"}:
        common_keys = {
            "input_dim",
            "action_dim",
            "hidden_size",
            "head_dim",
            "feedforward_size",
            "num_heads",
            "eta",
            "approximation_channels",
            "num_layers",
            "gate_bias",
            "attention_epsilon",
            "layer_norm_epsilon",
        }
        expected_keys = (
            common_keys
            | {
                "observation_dim",
                "actor_hidden_size",
                "critic_hidden_size",
            }
            if model == "agalite_source_compat"
            else common_keys
        )
        if set(value) != expected_keys:
            raise ValueError(
                f"{field} has wrong fields for model {model!r}: "
                f"expected={sorted(expected_keys)}"
            )
        integer_fields = expected_keys - {
            "gate_bias",
            "attention_epsilon",
            "layer_norm_epsilon",
        }
        if any(
            isinstance(value[name], bool)
            or not isinstance(value[name], int)
            or value[name] <= 0
            for name in integer_fields
        ):
            raise ValueError(f"{field} dimensions must be positive integers")
        if (
            model == "agalite_source_compat"
            and value["observation_dim"] > value["input_dim"]
        ):
            raise ValueError(f"{field}.observation_dim exceeds input_dim")
        expected_common = {
            "num_heads": AGALITE_NUM_HEADS,
            "eta": AGALITE_ETA,
            "approximation_channels": AGALITE_APPROXIMATION_CHANNELS,
            "num_layers": AGALITE_NUM_LAYERS,
            "gate_bias": AGALITE_GATE_BIAS,
            "attention_epsilon": AGALITE_ATTENTION_EPSILON,
            "layer_norm_epsilon": AGALITE_LAYER_NORM_EPSILON,
        }
        if any(value[name] != expected for name, expected in expected_common.items()):
            raise ValueError(f"{field} drifts from the AGaLiTe executable contract")
        if model == "agalite_source_compat":
            expected_source = {
                "hidden_size": AGALITE_MODEL_SIZE,
                "head_dim": AGALITE_HEAD_SIZE,
                "feedforward_size": AGALITE_FEEDFORWARD_SIZE,
                "actor_hidden_size": 128,
                "critic_hidden_size": 128,
            }
            if any(value[name] != expected for name, expected in expected_source.items()):
                raise ValueError(
                    f"{field} drifts from the released T-Maze architecture"
                )
        elif (
            value["hidden_size"] % 2 != 0
            or value["head_dim"] != value["hidden_size"] // 2
            or value["feedforward_size"] != value["hidden_size"]
        ):
            raise ValueError(
                f"{field} must preserve even D, Q=D/2, and F=D for agalite_shared"
            )
        return

    expected_keys = (
        {
            "input_dim",
            "observation_dim",
            "action_dim",
            "hidden_size",
            "decays",
        }
        if model == "memory_trace_official"
        else {"input_dim", "action_dim", "hidden_size", "decays"}
    )
    if set(value) != expected_keys:
        raise ValueError(
            f"{field} has wrong fields for model {model!r}: "
            f"expected={sorted(expected_keys)}"
        )
    integer_fields = expected_keys - {"decays"}
    if any(
        isinstance(value[name], bool)
        or not isinstance(value[name], int)
        or value[name] <= 0
        for name in integer_fields
    ):
        raise ValueError(f"{field} dimensions must be positive integers")
    if model == "memory_trace_official":
        if value["observation_dim"] > value["input_dim"]:
            raise ValueError(f"{field}.observation_dim exceeds input_dim")
        if value["hidden_size"] != 64:
            raise ValueError(f"{field}.hidden_size must equal the official value 64")
    decays = value["decays"]
    if not isinstance(decays, (list, tuple)) or list(decays) != list(
        MEMORY_TRACE_DECAYS
    ):
        raise ValueError(
            f"{field}.decays must equal the immutable contract {list(MEMORY_TRACE_DECAYS)}"
        )


def validate_causal_transformer_horizon_contract(
    model: str,
    value: object,
    maximum_episode_steps: object,
    *,
    field: str,
) -> None:
    """Require registered full attention to cover the complete task horizon."""

    if model != "causal_transformer":
        return
    if (
        isinstance(maximum_episode_steps, bool)
        or not isinstance(maximum_episode_steps, int)
        or maximum_episode_steps <= 0
    ):
        raise ValueError(f"{field} requires a positive integer task horizon")
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object for model {model!r}")
    window_length = value.get("window_length")
    if (
        isinstance(window_length, bool)
        or not isinstance(window_length, int)
        or window_length != maximum_episode_steps
    ):
        raise ValueError(
            f"{field}.window_length must equal the task maximum episode horizon "
            f"{maximum_episode_steps}, found {window_length!r}"
        )


def fixed_official_parameter_count(model: str, policy_core: object) -> int | None:
    """Return the exact fixed-source count, or None for matched models."""

    if model not in {"agalite_source_compat", "memory_trace_official"}:
        return None
    validate_policy_core_contract(model, policy_core, field="policy_core")
    assert isinstance(policy_core, dict)
    if model == "agalite_source_compat":
        observation_dim = int(policy_core["observation_dim"])
        action_dim = int(policy_core["action_dim"])
        return 1_771_713 + 128 * observation_dim + 129 * action_dim

    feature_dim = len(MEMORY_TRACE_DECAYS) * int(policy_core["observation_dim"])
    hidden_size = 64
    action_dim = int(policy_core["action_dim"])
    actor = (
        feature_dim * hidden_size
        + hidden_size
        + hidden_size * hidden_size
        + hidden_size
        + hidden_size * action_dim
        + action_dim
    )
    critic = (
        feature_dim * hidden_size
        + hidden_size
        + hidden_size * hidden_size
        + hidden_size
        + hidden_size
        + 1
    )
    return actor + critic


def validate_model_evidence_tier(model: str, evidence_tier: str, *, field: str) -> None:
    """Keep the ambiguous legacy alias out of selection and final evidence."""

    if model == "memory_trace_mlp" and evidence_tier in {
        "development_tuning",
        "registered_final",
    }:
        raise ValueError(
            f"{field} uses development-only compatibility alias 'memory_trace_mlp'; "
            "use 'memory_trace_shared' for registered selection or final evidence"
        )


def validate_model_environment_contract(
    model: str,
    environment: str,
    *,
    field: str,
) -> None:
    """Reject the categorical official actor on continuous-action tasks."""

    fixed_categorical = {
        "agalite_source_compat": "agalite_shared",
        "memory_trace_official": "memory_trace_shared",
    }
    if model in fixed_categorical and environment in _CONTINUOUS_POBAX_ENVIRONMENTS:
        raise ValueError(
            f"{field} cannot use categorical {model!r} on continuous-action "
            f"environment {environment!r}; use {fixed_categorical[model]!r}"
        )


def validate_required_reference_implementation(
    model: str,
    value: object,
    *,
    field: str,
) -> None:
    """Fail closed when a source-audited model's metadata is absent or drifts."""

    expected = _REQUIRED_REFERENCE_IMPLEMENTATIONS.get(model)
    if expected is not None and value != expected:
        raise ValueError(
            f"{field} does not match the registered source contract for model {model!r}"
        )
