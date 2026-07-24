"""Pure metadata registry for policy implementations in the POBAX harness."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

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
    "mamba1": MAMBA1_REFERENCE_IMPLEMENTATION,
    "memory_trace_official": MEMORY_TRACE_OFFICIAL_REFERENCE_IMPLEMENTATION,
    "memory_trace_shared": MEMORY_TRACE_SHARED_REFERENCE_IMPLEMENTATION,
    "memory_trace_mlp": MEMORY_TRACE_COMPATIBILITY_REFERENCE_IMPLEMENTATION,
}

_MODEL_PARAMETER_CONTRACTS = {
    "memory_trace_official": FIXED_OFFICIAL_PARAMETER_CONTRACT,
}
_MODEL_COMPARISON_ROLES = {
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
    return metadata


def requires_explicit_policy_contract(model: str) -> bool:
    """Return whether legacy omission is prohibited for this model."""

    return model in {
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
    """Validate architecture and decays for both Memory Traces lanes."""

    if not requires_explicit_policy_contract(model):
        return
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object for model {model!r}")
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


def fixed_official_parameter_count(model: str, policy_core: object) -> int | None:
    """Return the exact fixed-source count, or None for matched models."""

    if model != "memory_trace_official":
        return None
    validate_policy_core_contract(model, policy_core, field="policy_core")
    assert isinstance(policy_core, dict)
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

    if (
        model == "memory_trace_official"
        and environment in _CONTINUOUS_POBAX_ENVIRONMENTS
    ):
        raise ValueError(
            f"{field} cannot use categorical 'memory_trace_official' on continuous-action "
            f"environment {environment!r}; use 'memory_trace_shared'"
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
