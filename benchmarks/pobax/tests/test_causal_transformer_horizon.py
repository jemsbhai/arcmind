"""Regression tests for the full-horizon causal Transformer contract."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict

import pytest

from benchmarks.pobax.aggregate_development import (
    DevelopmentAggregationError,
    build_development_aggregate,
)
from benchmarks.pobax.aggregate_registered import (
    RegisteredAggregationError,
)
from benchmarks.pobax.aggregate_registered import (
    _validate_artifact as _validate_registered_artifact,
)
from benchmarks.pobax.model_registry import (
    policy_contract_metadata_for_model,
    validate_causal_transformer_horizon_contract,
)
from benchmarks.pobax.registered_artifacts import (
    ExistingArtifactMismatchError,
    canonical_json_bytes,
    canonical_json_sha256,
    registered_cell_id,
)
from benchmarks.pobax.run_matrix import _load_matching_artifact
from benchmarks.pobax.run_pilot import build_policy_core
from benchmarks.pobax.sequence_cores import FullCausalTransformerPolicyCore
from benchmarks.pobax.tests import (
    test_aggregate_development as development_helpers,
)
from benchmarks.pobax.tests import (
    test_aggregate_registered as registered_helpers,
)


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


def _causal_policy_core(
    *,
    input_dim: int,
    action_dim: int,
    hidden_size: int,
    window_length: int,
) -> dict[str, object]:
    return {
        "input_dim": input_dim,
        "action_dim": action_dim,
        "hidden_size": hidden_size,
        "window_length": window_length,
        "num_heads": 4,
        "num_layers": 2,
    }


def test_development_aggregation_rejects_top_level_causal_core_drift(
    tmp_path,
    monkeypatch,
) -> None:
    learner = {
        "num_envs": 2,
        "rollout_steps": 2,
        "update_epochs": 1,
        "num_minibatches": 1,
        "gae_lambda": 0.95,
        "entropy_coefficient": 0.01,
        "anneal_learning_rate": True,
    }
    candidate_families = [
        {
            "family_id": "full_attention",
            "implementation_model": "causal_transformer",
            "candidates": [
                {
                    "candidate_id": f"full_attention.lr_{label}",
                    "learner": {**learner, "learning_rate": learning_rate},
                }
                for label, learning_rate in (("low", 0.001), ("high", 0.002))
            ],
        }
    ]
    policy_core = _causal_policy_core(
        input_dim=7,
        action_dim=2,
        hidden_size=4,
        window_length=5,
    )
    monkeypatch.setattr(
        development_helpers,
        "requires_explicit_policy_contract",
        lambda model: model == "causal_transformer",
    )
    monkeypatch.setattr(
        development_helpers,
        "_policy_core_for_model",
        lambda model: deepcopy(policy_core),
    )
    matrix_root = tmp_path / "development"
    manifest, paths = development_helpers._write_matrix(
        matrix_root,
        tier="development_tuning",
        environments={"tmaze_10": 1_000_000},
        seeds=[7, 19, 23, 31, 43],
        schema_version=3,
        comparison_profile="arcmind_shared_comparison",
        candidate_families=candidate_families,
    )
    for path in paths.values():
        artifact = json.loads(path.read_text(encoding="utf-8"))
        final_return = artifact["evaluation"]["mean_return"]
        development_helpers._rewrite(
            path,
            lambda value, final_return=final_return: value.update(
                training_history=[
                    {
                        **development_helpers.OPTIMIZER_METRICS,
                        "environment_steps": 250_000.0,
                        "mean_recent_return": None,
                    },
                    {
                        **development_helpers.OPTIMIZER_METRICS,
                        "environment_steps": 500_000.0,
                        "mean_recent_return": final_return - 2.0,
                    },
                    {
                        **development_helpers.OPTIMIZER_METRICS,
                        "environment_steps": 1_000_000.0,
                        "mean_recent_return": final_return,
                    },
                ]
            ),
        )
    development_helpers._write_integrity_indexes(matrix_root, manifest, paths)
    assert build_development_aggregate(matrix_root)

    artifact_path = paths[("tmaze_10", "full_attention.lr_low", 7)]
    development_helpers._rewrite(
        artifact_path,
        lambda value: value["policy_core"].update(window_length=32),
    )
    registered_helpers._refresh_integrity(matrix_root, paths)
    with pytest.raises(DevelopmentAggregationError, match="window_length"):
        build_development_aggregate(matrix_root)

    development_helpers._rewrite(
        artifact_path,
        lambda value: value["policy_core"].update(
            window_length=5,
            hidden_size=8,
        ),
    )
    registered_helpers._refresh_integrity(matrix_root, paths)
    with pytest.raises(
        DevelopmentAggregationError,
        match="does not match the frozen configuration",
    ):
        build_development_aggregate(matrix_root)


def test_registered_aggregation_rejects_top_level_causal_core_drift(
    tmp_path,
    monkeypatch,
) -> None:
    policy_core = _causal_policy_core(
        input_dim=9,
        action_dim=3,
        hidden_size=16,
        window_length=1_000,
    )
    monkeypatch.setattr(
        registered_helpers,
        "requires_explicit_policy_contract",
        lambda model: model == "causal_transformer",
    )
    monkeypatch.setattr(
        registered_helpers,
        "_policy_core_for_model",
        lambda model: deepcopy(policy_core),
    )
    matrix_root = tmp_path / "registered"
    _, manifest, paths = registered_helpers._write_matrix(
        matrix_root,
        models=["causal_transformer"],
        seeds=[11],
        schema_version=2,
        comparison_profile="arcmind_shared_comparison",
    )
    identity = ("tmaze_10", "causal_transformer", 11)
    artifact_path = paths[identity]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    configuration = artifact["configuration"]
    tuning_hashes = {
        "tuning_aggregate_sha256": "5" * 64,
        "tuning_completion_index_sha256": "6" * 64,
        "tuning_checksum_manifest_sha256": "7" * 64,
        "tuning_implementation_source_sha256": (
            registered_helpers.IMPLEMENTATION_SOURCE["sha256"]
        ),
    }
    configuration.update(
        schema_version=4,
        candidate_id="full_attention.lr_low",
        model_family="full_attention",
        implementation_model="causal_transformer",
        implementation_source=deepcopy(registered_helpers.IMPLEMENTATION_SOURCE),
        **tuning_hashes,
    )
    configuration_sha256 = canonical_json_sha256(configuration)
    cell_id = registered_cell_id(*identity, configuration_sha256)
    artifact.update(
        schema_version=8,
        configuration=configuration,
        configuration_sha256=configuration_sha256,
        cell_id=cell_id,
        provenance={
            **deepcopy(registered_helpers.PROVENANCE),
            "implementation_source": deepcopy(
                registered_helpers.IMPLEMENTATION_SOURCE
            ),
        },
        candidate_id="full_attention.lr_low",
        model_family="full_attention",
        implementation_model="causal_transformer",
        implementation_source_sha256=(
            registered_helpers.IMPLEMENTATION_SOURCE["sha256"]
        ),
        **tuning_hashes,
    )
    artifact_path.write_bytes(canonical_json_bytes(artifact) + b"\n")
    learner = {
        name: configuration["ppo"][name]
        for name in (
            "num_envs",
            "rollout_steps",
            "update_epochs",
            "num_minibatches",
            "learning_rate",
            "gae_lambda",
            "entropy_coefficient",
            "anneal_learning_rate",
        )
    }
    expected = {
        "cell_id": cell_id,
        "configuration_sha256": configuration_sha256,
        "candidate_id": "full_attention.lr_low",
        "model_family": "full_attention",
        "implementation_model": "causal_transformer",
        "implementation_source_sha256": (
            registered_helpers.IMPLEMENTATION_SOURCE["sha256"]
        ),
        "learner": learner,
        **tuning_hashes,
    }
    validation_arguments = {
        "identity": identity,
        "expected": expected,
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_schema_version": 4,
        "provenance": artifact["provenance"],
    }
    assert _validate_registered_artifact(
        artifact_path,
        **validation_arguments,
    )

    artifact["policy_core"]["window_length"] = 32
    artifact_path.write_bytes(canonical_json_bytes(artifact) + b"\n")
    with pytest.raises(RegisteredAggregationError, match="window_length"):
        _validate_registered_artifact(
            artifact_path,
            **validation_arguments,
        )

    artifact["policy_core"].update(window_length=1_000, hidden_size=8)
    artifact_path.write_bytes(canonical_json_bytes(artifact) + b"\n")
    with pytest.raises(
        RegisteredAggregationError,
        match="does not match the frozen configuration",
    ):
        _validate_registered_artifact(
            artifact_path,
            **validation_arguments,
        )
