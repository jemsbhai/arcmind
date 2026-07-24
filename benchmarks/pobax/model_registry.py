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

POLICY_MODEL_IDS = (
    "memoryless_mlp",
    "frame_stack_mlp",
    "memory_trace_mlp",
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

_REQUIRED_REFERENCE_IMPLEMENTATIONS = {
    "mamba1": MAMBA1_REFERENCE_IMPLEMENTATION,
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
