"""Regression tests for the full-horizon causal Transformer contract."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict

import pytest

from benchmarks.pobax.model_registry import (
    policy_contract_metadata_for_model,
    validate_causal_transformer_horizon_contract,
)
from benchmarks.pobax.registered_artifacts import (
    ExistingArtifactMismatchError,
    canonical_json_sha256,
)
from benchmarks.pobax.run_matrix import _load_matching_artifact
from benchmarks.pobax.run_pilot import build_policy_core
from benchmarks.pobax.sequence_cores import FullCausalTransformerPolicyCore


@pytest.mark.parametrize(
    ("environment", "maximum_episode_steps"),
    [
        ("rocksample_11_11", 1_000),
        ("Navix-DMLab-Maze-01-v0", 2_000),
    ],
)
def test_registered_builder_serializes_the_complete_task_horizon(
    environment: str,
    maximum_episode_steps: int,
) -> None:
    core, parameter_count, target_parameter_count = build_policy_core(
        "causal_transformer",
        input_dim=10,
        action_dim=4,
        seed=1103,
        max_episode_steps=maximum_episode_steps,
    )
    serialized = asdict(core)

    assert environment
    assert serialized["window_length"] == maximum_episode_steps
    assert 0.9 <= parameter_count / target_parameter_count <= 1.1
    validate_causal_transformer_horizon_contract(
        "causal_transformer",
        serialized,
        maximum_episode_steps,
        field="policy_core",
    )


def test_full_causal_transformer_has_no_silent_default_window() -> None:
    with pytest.raises(TypeError, match="window_length"):
        FullCausalTransformerPolicyCore(10, 4, 32)  # type: ignore[call-arg]


def _artifact(
    *,
    registration_schema_version: int,
    window_length: int,
) -> tuple[dict[str, object], dict[str, object]]:
    seed = 1701
    provenance: dict[str, object] = {
        "git": {"commit": "a" * 40, "dirty": False, "diff_sha256": None},
        "dependency_lock_sha256": "b" * 64,
        "pobax_commit": "c" * 40,
        "navix_commit": "d" * 40,
        "runtime_contract": {"runtime": "test"},
    }
    policy_core = {
        "input_dim": 38,
        "action_dim": 5,
        "hidden_size": 64,
        "window_length": window_length,
        "num_heads": 4,
        "num_layers": 2,
    }
    model = (
        "full_attention.lr2p5e4_const"
        if registration_schema_version == 3
        else "causal_transformer"
    )
    configuration: dict[str, object] = {
        "environment": "rocksample_11_11",
        "model": model,
        "seed": seed,
        "evaluation_max_episode_steps": 1_000,
        "policy_core": deepcopy(policy_core),
        **policy_contract_metadata_for_model("causal_transformer"),
    }
    artifact: dict[str, object] = {
        "schema_version": {2: 5, 3: 6}[registration_schema_version],
        "status": (
            "development_tuning_not_for_paper"
            if registration_schema_version == 3
            else "development_pilot_not_for_paper"
        ),
        "environment": "rocksample_11_11",
        "model": model,
        "seed": seed,
        "configuration": configuration,
        "matrix_manifest_sha256": "f" * 64,
        "cell_id": "1" * 64,
        "provenance": provenance,
        "policy_core": deepcopy(policy_core),
        **policy_contract_metadata_for_model("causal_transformer"),
    }
    if registration_schema_version == 3:
        implementation_source_sha256 = "e" * 64
        configuration.update(
            {
                "candidate_id": model,
                "model_family": "full_attention",
                "implementation_model": "causal_transformer",
                "implementation_source": {
                    "sha256": implementation_source_sha256,
                },
            }
        )
        artifact.update(
            {
                "candidate_id": model,
                "model_family": "full_attention",
                "implementation_model": "causal_transformer",
                "implementation_source_sha256": implementation_source_sha256,
            }
        )
    configuration_sha256 = canonical_json_sha256(configuration)
    artifact["configuration_sha256"] = configuration_sha256
    return artifact, provenance


def _load_artifact(
    path,
    artifact: dict[str, object],
    provenance: dict[str, object],
    *,
    registration_schema_version: int,
):
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return _load_matching_artifact(
        path,
        expected_status=str(artifact["status"]),
        environment="rocksample_11_11",
        model=str(artifact["model"]),
        seed=1701,
        configuration_sha256=str(artifact["configuration_sha256"]),
        manifest_sha256="f" * 64,
        cell_id="1" * 64,
        provenance=provenance,
        registration_schema_version=registration_schema_version,
    )


def test_registered_resume_rejects_a_silent_32_step_window(tmp_path) -> None:
    path = tmp_path / "registered-causal-transformer.json"
    artifact, provenance = _artifact(
        registration_schema_version=3,
        window_length=1_000,
    )
    assert (
        _load_artifact(
            path,
            artifact,
            provenance,
            registration_schema_version=3,
        )
        == artifact
    )

    artifact["policy_core"]["window_length"] = 32  # type: ignore[index]
    artifact["configuration"]["policy_core"]["window_length"] = 32  # type: ignore[index]
    artifact["configuration_sha256"] = canonical_json_sha256(
        artifact["configuration"]
    )
    with pytest.raises(
        ExistingArtifactMismatchError,
        match="causal attention horizon",
    ):
        _load_artifact(
            path,
            artifact,
            provenance,
            registration_schema_version=3,
        )


def test_legacy_pilot_artifact_with_32_step_window_remains_readable(tmp_path) -> None:
    path = tmp_path / "legacy-causal-transformer.json"
    artifact, provenance = _artifact(
        registration_schema_version=2,
        window_length=32,
    )

    assert (
        _load_artifact(
            path,
            artifact,
            provenance,
            registration_schema_version=2,
        )
        == artifact
    )
