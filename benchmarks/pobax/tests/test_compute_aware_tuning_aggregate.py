"""Aggregation tests for the schema-v5 compute-aware tuning panel."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from benchmarks.pobax.aggregate_development import (
    DevelopmentAggregationError,
    build_development_aggregate,
)
from benchmarks.pobax.implementation_provenance import IMPLEMENTATION_SOURCE_ALGORITHM
from benchmarks.pobax.model_registry import (
    policy_contract_metadata_for_model,
    reference_implementation_for_model,
    requires_explicit_policy_contract,
)
from benchmarks.pobax.registered_artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    canonical_json_sha256,
    registered_cell_id,
    sha256_file,
    write_checksum_manifest,
)
from benchmarks.pobax.registration_protocol import (
    COMPUTE_AWARE_LEARNER_GRID,
    COMPUTE_AWARE_TUNED_FAMILIES,
)
from benchmarks.pobax.upper_reference_registry import (
    expected_environment_reference,
    expected_environment_source,
)

ENVIRONMENTS = {
    "tmaze_10": 1_000_000,
    "rocksample_11_11": 1_000_000,
}
SEEDS = [4409, 5519, 6637]
OPTIMIZER_METRICS = {
    "loss": 0.5,
    "actor_loss": 0.1,
    "value_loss": 0.8,
    "entropy": 0.2,
    "approximate_kl": 0.01,
}
PROVENANCE = {
    "git": {"commit": "1" * 40, "dirty": False, "diff_sha256": None},
    "dependency_lock_sha256": "2" * 64,
    "pobax_commit": "3" * 40,
    "navix_commit": "4" * 40,
    "runtime_contract": {
        "python": {"implementation": "CPython", "version": "3.12.3"},
        "packages": {"jax": "0.6.2", "jaxlib": "0.6.2", "numpy": "2.2.0"},
        "jax_backend": "gpu",
        "jax_enable_x64": False,
        "devices": [{"platform": "gpu", "device_kind": "Test GPU"}],
    },
}
_IMPLEMENTATION_SOURCE_UNSIGNED = {
    "schema_version": 1,
    "algorithm": IMPLEMENTATION_SOURCE_ALGORITHM,
    "files": [{"path": "arcmind/__init__.py", "sha256": "a" * 64}],
}
IMPLEMENTATION_SOURCE = {
    **_IMPLEMENTATION_SOURCE_UNSIGNED,
    "sha256": canonical_json_sha256(_IMPLEMENTATION_SOURCE_UNSIGNED),
}


def _learner(learning_rate: float) -> dict[str, int | float | bool]:
    return {
        "num_envs": 8,
        "rollout_steps": 125,
        "update_epochs": 4,
        "num_minibatches": 4,
        "learning_rate": learning_rate,
        "gae_lambda": 0.95,
        "entropy_coefficient": 0.01,
        "anneal_learning_rate": False,
    }


def _registration() -> dict[str, Any]:
    return {
        "schema_version": 5,
        "status": "frozen",
        "evidence_tier": "development_tuning",
        "matrix_kind": "hyperparameter_selection",
        "tuned_families": [
            {"family_id": family, "implementation_model": family}
            for family in COMPUTE_AWARE_TUNED_FAMILIES
        ],
        "learner_grid": [
            {"learner_id": learner_id, "learner": _learner(learning_rate)}
            for learner_id, learning_rate in COMPUTE_AWARE_LEARNER_GRID
        ],
        "environments": [
            {"id": environment, "total_steps": total_steps}
            for environment, total_steps in ENVIRONMENTS.items()
        ],
        "seeds": SEEDS,
        "comparison_profile": "arcmind_shared_comparison",
        "evaluation_episodes_per_env": 1,
        "require_gpu": True,
        "quick": False,
    }


def _policy_core(model: str) -> dict[str, Any]:
    if model == "agalite_shared":
        return {
            "input_dim": 7,
            "action_dim": 2,
            "hidden_size": 4,
            "head_dim": 2,
            "feedforward_size": 4,
            "num_heads": 4,
            "eta": 4,
            "approximation_channels": 2,
            "num_layers": 4,
            "gate_bias": 2.0,
            "attention_epsilon": 1e-5,
            "layer_norm_epsilon": 1e-6,
        }
    return {
        "input_dim": 7,
        "action_dim": 2,
        "hidden_size": 4,
        "decays": [0.0, 0.985],
    }


def _candidate_index(registration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        f"{family['family_id']}.{grid_item['learner_id']}": {
            "model_family": family["family_id"],
            "implementation_model": family["implementation_model"],
            "learner_id": grid_item["learner_id"],
            "learner": grid_item["learner"],
        }
        for family in registration["tuned_families"]
        for grid_item in registration["learner_grid"]
    }


def _configuration(
    *,
    environment: str,
    candidate_id: str,
    seed: int,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    model = candidate["implementation_model"]
    total_steps = ENVIRONMENTS[environment]
    configuration = {
        "schema_version": 5,
        "evidence_tier": "development_tuning",
        "environment": environment,
        "environment_source": expected_environment_source(environment),
        "environment_reference": expected_environment_reference(environment),
        "model": candidate_id,
        "seed": seed,
        "candidate_id": candidate_id,
        "model_family": candidate["model_family"],
        "implementation_model": model,
        "learner_id": candidate["learner_id"],
        "implementation_source": deepcopy(IMPLEMENTATION_SOURCE),
        "parameter_count": 1_000,
        "effective_parameter_count": 1_000,
        "arcmind_target_parameter_count": 1_000,
        "parameter_ratio": 1.0,
        "ppo": {
            "total_steps": total_steps,
            **candidate["learner"],
            "step_budget_mode": "exact",
        },
        "evaluation_episodes_per_environment": 1,
        "evaluation_max_episode_steps": 5,
        "dependency_lock_sha256": PROVENANCE["dependency_lock_sha256"],
        "pobax_commit": PROVENANCE["pobax_commit"],
        "navix_commit": PROVENANCE["navix_commit"],
        "runtime_contract": deepcopy(PROVENANCE["runtime_contract"]),
        "comparison_profile": "arcmind_shared_comparison",
        "requested_environment_steps": total_steps,
        "realized_environment_steps": total_steps,
    }
    reference = reference_implementation_for_model(model)
    if reference is not None:
        configuration["reference_implementation"] = reference
    if requires_explicit_policy_contract(model):
        configuration.update(
            policy_contract_metadata_for_model(model),
            policy_core=_policy_core(model),
        )
    return configuration


def _curve_score(environment: str, family: str, learner_id: str) -> float:
    if family == COMPUTE_AWARE_TUNED_FAMILIES[-1]:
        return 4.0
    task_scores = {
        "tmaze_10": {"lr_low": 10.0, "lr_mid": 8.0, "lr_high": 0.0},
        "rocksample_11_11": {"lr_low": 0.0, "lr_mid": 8.0, "lr_high": 10.0},
    }
    return task_scores[environment][learner_id]


def _evaluation(value: float) -> dict[str, Any]:
    rows = [[value] for _ in range(8)]
    return {
        "mean_return": value,
        "median_return": value,
        "episodes": 8,
        "episodes_per_environment": 1,
        "num_environments": 8,
        "scan_steps_per_environment": 5,
        "returns_by_environment": rows,
    }


def _artifact(
    *,
    manifest_sha256: str,
    cell_id: str,
    configuration: dict[str, Any],
    configuration_sha256: str,
    history_score: float,
    delayed_prefix: bool,
) -> dict[str, Any]:
    model = configuration["implementation_model"]
    total_steps = configuration["ppo"]["total_steps"]
    learner_id = configuration["learner_id"]
    final_return = {"lr_low": 1_000.0, "lr_mid": -1_000.0, "lr_high": 0.0}[learner_id]
    artifact = {
        "schema_version": 9,
        "status": "development_tuning_not_for_paper",
        "matrix_manifest_sha256": manifest_sha256,
        "cell_id": cell_id,
        "configuration_sha256": configuration_sha256,
        "configuration": configuration,
        "environment": configuration["environment"],
        "model": configuration["model"],
        "seed": configuration["seed"],
        "candidate_id": configuration["candidate_id"],
        "model_family": configuration["model_family"],
        "implementation_model": model,
        "learner_id": learner_id,
        "implementation_source_sha256": IMPLEMENTATION_SOURCE["sha256"],
        "environment_source": deepcopy(configuration["environment_source"]),
        "environment_reference": deepcopy(configuration["environment_reference"]),
        "parameter_count": configuration["parameter_count"],
        "effective_parameter_count": configuration["effective_parameter_count"],
        "arcmind_target_parameter_count": configuration["arcmind_target_parameter_count"],
        "parameter_ratio": configuration["parameter_ratio"],
        "provenance": {
            **deepcopy(PROVENANCE),
            "implementation_source": deepcopy(IMPLEMENTATION_SOURCE),
        },
        "actual_environment_steps": total_steps,
        "comparison_profile": "arcmind_shared_comparison",
        "requested_environment_steps": total_steps,
        "realized_environment_steps": total_steps,
        "ppo": deepcopy(configuration["ppo"]),
        "evaluation_episodes_per_environment": 1,
        "evaluation_max_episode_steps": 5,
        "actual_evaluation_steps_per_environment": 5,
        "actual_evaluation_transitions": 40,
        "evaluation": _evaluation(final_return),
        "training": {**OPTIMIZER_METRICS, "mean_recent_return": history_score},
        "training_history": [
            {
                **OPTIMIZER_METRICS,
                "environment_steps": 250_000.0,
                "mean_recent_return": None if delayed_prefix else history_score,
            },
            {
                **OPTIMIZER_METRICS,
                "environment_steps": 500_000.0,
                "mean_recent_return": history_score,
            },
            {
                **OPTIMIZER_METRICS,
                "environment_steps": 1_000_000.0,
                "mean_recent_return": history_score,
            },
        ],
    }
    reference = reference_implementation_for_model(model)
    if reference is not None:
        artifact["reference_implementation"] = reference
    if requires_explicit_policy_contract(model):
        artifact["policy_core"] = deepcopy(configuration["policy_core"])
        artifact.update(policy_contract_metadata_for_model(model))
    return artifact


def _rewrite_json(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _refresh_integrity(root: Path) -> None:
    completion_path = root / "completion_index.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    for item in completion["cells"]:
        artifact_path = root / item["artifact_path"]
        item["artifact_sha256"] = sha256_file(artifact_path)
        item["log_sha256"] = sha256_file(root / item["log_path"])
    _rewrite_json(completion_path, completion)
    checksum_path = root / "checksums.sha256"
    checksum_path.unlink()
    write_checksum_manifest(root)


def _write_matrix(root: Path) -> None:
    registration = _registration()
    candidate_index = _candidate_index(registration)
    atomic_write_json(root / "registration.json", registration)
    manifest_cells = []
    configurations = {}
    for environment in ENVIRONMENTS:
        for candidate_id, candidate in candidate_index.items():
            for seed in SEEDS:
                configuration = _configuration(
                    environment=environment,
                    candidate_id=candidate_id,
                    seed=seed,
                    candidate=candidate,
                )
                configuration_sha256 = canonical_json_sha256(configuration)
                cell_id = registered_cell_id(
                    environment,
                    candidate_id,
                    seed,
                    configuration_sha256,
                )
                relative_path = f"cells/{environment}/{candidate_id}/{seed}.json"
                manifest_cells.append(
                    {
                        "cell_id": cell_id,
                        "environment": environment,
                        "model": candidate_id,
                        "seed": seed,
                        "configuration_sha256": configuration_sha256,
                        "artifact_path": relative_path,
                        "model_family": candidate["model_family"],
                        "implementation_model": candidate["implementation_model"],
                        "learner_id": candidate["learner_id"],
                        "implementation_source_sha256": IMPLEMENTATION_SOURCE["sha256"],
                    }
                )
                configurations[(environment, candidate_id, seed)] = (
                    configuration,
                    configuration_sha256,
                    cell_id,
                    relative_path,
                )
    manifest = {
        "schema_version": 5,
        "status": "frozen",
        "matrix_kind": "hyperparameter_selection",
        "models": list(candidate_index),
        "environments": list(ENVIRONMENTS),
        "seeds": SEEDS,
        "provenance": {
            **deepcopy(PROVENANCE),
            "implementation_source": deepcopy(IMPLEMENTATION_SOURCE),
        },
        "cells": manifest_cells,
        "tuned_families": registration["tuned_families"],
        "learner_grid": registration["learner_grid"],
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_write_json(root / "frozen_manifest.json", manifest)

    completed_cells = []
    delayed_identities = {
        (
            environment,
            f"{COMPUTE_AWARE_TUNED_FAMILIES[0]}.lr_low",
            SEEDS[0],
        )
        for environment in ENVIRONMENTS
    }
    for cell in manifest_cells:
        identity = (cell["environment"], cell["model"], cell["seed"])
        configuration, configuration_sha256, cell_id, relative_path = configurations[identity]
        candidate = candidate_index[cell["model"]]
        artifact_path = root / relative_path
        atomic_write_json(
            artifact_path,
            _artifact(
                manifest_sha256=manifest["manifest_sha256"],
                cell_id=cell_id,
                configuration=configuration,
                configuration_sha256=configuration_sha256,
                history_score=_curve_score(
                    cell["environment"],
                    candidate["model_family"],
                    candidate["learner_id"],
                ),
                delayed_prefix=identity in delayed_identities,
            ),
        )
        log_path = artifact_path.with_suffix(".log")
        atomic_write_bytes(log_path, b"test log\n")
        completed_cells.append(
            {
                **cell,
                "artifact_sha256": sha256_file(artifact_path),
                "log_path": log_path.relative_to(root).as_posix(),
                "log_sha256": sha256_file(log_path),
            }
        )
    atomic_write_json(
        root / "completion_index.json",
        {
            "schema_version": 1,
            "status": "complete",
            "manifest_sha256": manifest["manifest_sha256"],
            "planned_cells": len(completed_cells),
            "completed_cells": len(completed_cells),
            "cells": completed_cells,
        },
    )
    write_checksum_manifest(root)


@pytest.fixture(scope="module")
def schema_v5_matrix(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("schema-v5-tuning") / "matrix"
    _write_matrix(root)
    return root, build_development_aggregate(root)


def test_schema_v5_selects_one_panel_learner_per_family(
    schema_v5_matrix,
) -> None:
    _, aggregate = schema_v5_matrix

    assert aggregate["schema_version"] == 2
    assert len(aggregate["groups"]) == 78
    assert len(aggregate["candidate_selection"]) == 13
    assert [item["model_family"] for item in aggregate["candidate_selection"]] == list(
        COMPUTE_AWARE_TUNED_FAMILIES
    )
    assert all(
        contract["training_curve"]["curve_start_step"] == 500_000
        for contract in aggregate["environment_contracts"]
    )

    selection = aggregate["candidate_selection"][0]
    assert selection["winner_candidate_id"] == "memoryless_mlp.lr_mid"
    assert selection["winner_learner_id"] == "lr_mid"
    assert selection["winner_learner"] == _learner(0.00025)
    assert selection["task_scores"] == [
        {
            "environment": "tmaze_10",
            "selection_score": 8.0,
            "task_rank": 2,
            "task_range_regret": 0.2,
        },
        {
            "environment": "rocksample_11_11",
            "selection_score": 8.0,
            "task_rank": 2,
            "task_range_regret": 0.2,
        },
    ]
    assert [
        (item["learner_id"], item["mean_task_rank"], item["mean_task_range_regret"])
        for item in selection["ranking"]
    ] == [
        ("lr_mid", 2.0, 0.2),
        ("lr_high", 2.0, 0.5),
        ("lr_low", 2.0, 0.5),
    ]

    exact_tie = aggregate["candidate_selection"][-1]
    assert exact_tie["winner_learner_id"] == "lr_high"
    assert all(
        task["task_rank"] == 1 and task["task_range_regret"] == 0.0
        for candidate in exact_tie["ranking"]
        for task in candidate["task_scores"]
    )


def test_final_evaluation_mutation_cannot_change_panel_selection(
    schema_v5_matrix,
    tmp_path: Path,
) -> None:
    source, baseline = schema_v5_matrix
    mutated = tmp_path / "mutated"
    shutil.copytree(source, mutated)
    for artifact_path in (mutated / "cells").rglob("*.json"):
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        learner_id = artifact["learner_id"]
        final_return = {"lr_low": -9_000.0, "lr_mid": 9_000.0, "lr_high": 7_000.0}[learner_id]
        artifact["evaluation"] = _evaluation(final_return)
        _rewrite_json(artifact_path, artifact)
    _refresh_integrity(mutated)

    rebuilt = build_development_aggregate(mutated)

    assert rebuilt["candidate_selection"] == baseline["candidate_selection"]
    assert rebuilt["groups"] != baseline["groups"]


def test_schema_v5_rejects_incomplete_manifest_inventory(
    schema_v5_matrix,
    tmp_path: Path,
) -> None:
    source, _ = schema_v5_matrix
    mutated = tmp_path / "missing-cell"
    shutil.copytree(source, mutated)
    manifest_path = mutated / "frozen_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cells"].pop()
    manifest["manifest_sha256"] = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    _rewrite_json(manifest_path, manifest)

    with pytest.raises(DevelopmentAggregationError, match="missing Cartesian cells"):
        build_development_aggregate(mutated)


def test_schema_v5_rejects_noncanonical_checksum_inventory(
    schema_v5_matrix,
    tmp_path: Path,
) -> None:
    source, _ = schema_v5_matrix
    mutated = tmp_path / "extra-inventory"
    shutil.copytree(source, mutated)
    atomic_write_bytes(mutated / "unregistered.txt", b"not a raw matrix artifact\n")
    (mutated / "checksums.sha256").unlink()
    write_checksum_manifest(mutated)

    with pytest.raises(
        DevelopmentAggregationError,
        match="checksum inventory contains noncanonical raw inputs",
    ):
        build_development_aggregate(mutated)


def test_schema_v5_binds_manifest_and_configuration_canonical_hashes(
    schema_v5_matrix,
    tmp_path: Path,
) -> None:
    source, _ = schema_v5_matrix
    manifest_mutation = tmp_path / "manifest-hash"
    shutil.copytree(source, manifest_mutation)
    manifest_path = manifest_mutation / "frozen_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["learner_grid"][0]["learner"]["learning_rate"] = 0.0002
    _rewrite_json(manifest_path, manifest)
    with pytest.raises(DevelopmentAggregationError, match="manifest_sha256"):
        build_development_aggregate(manifest_mutation)

    configuration_mutation = tmp_path / "configuration-hash"
    shutil.copytree(source, configuration_mutation)
    artifact_path = next((configuration_mutation / "cells").rglob("*.json"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["configuration"]["ppo"]["learning_rate"] = 0.25
    _rewrite_json(artifact_path, artifact)
    _refresh_integrity(configuration_mutation)
    with pytest.raises(
        DevelopmentAggregationError,
        match="configuration_sha256 does not match configuration",
    ):
        build_development_aggregate(configuration_mutation)
