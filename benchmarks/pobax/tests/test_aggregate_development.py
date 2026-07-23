"""Tests for fail-closed development matrix aggregation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pytest

from benchmarks.pobax.aggregate_development import (
    BOOTSTRAP_RESAMPLES,
    DevelopmentAggregationError,
    aggregate_development,
    build_development_aggregate,
)
from benchmarks.pobax.registered_artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_sha256,
    registered_cell_id,
    sha256_file,
    write_checksum_manifest,
)
from benchmarks.pobax.upper_reference_registry import (
    expected_environment_reference,
    expected_environment_source,
)

MODELS = ["arcmind", "gru"]
ENVIRONMENTS = {"short": 8_192, "long": 16_384}
SEEDS = [7, 19, 23]
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


def _configuration(environment: str, model: str, seed: int, *, tier: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_tier": tier,
        "environment": environment,
        "model": model,
        "seed": seed,
        "environment_source": expected_environment_source(environment),
        "environment_reference": expected_environment_reference(environment),
        "parameter_count": 1_000,
        "effective_parameter_count": 1_000,
        "arcmind_target_parameter_count": 1_000,
        "parameter_ratio": 1.0,
        "ppo": {
            "total_steps": ENVIRONMENTS[environment],
            "num_envs": 2,
            "rollout_steps": 2,
            "update_epochs": 1,
            "learning_rate": 0.001,
        },
        "evaluation_episodes_per_environment": 2,
        "evaluation_max_episode_steps": 5,
        "dependency_lock_sha256": PROVENANCE["dependency_lock_sha256"],
        "pobax_commit": PROVENANCE["pobax_commit"],
        "navix_commit": PROVENANCE["navix_commit"],
        "runtime_contract": deepcopy(PROVENANCE["runtime_contract"]),
    }


def _evaluation(value: float) -> dict[str, Any]:
    rows = [[value - 1.0, value], [value, value + 1.0]]
    flat = [item for row in rows for item in row]
    return {
        "mean_return": sum(flat) / len(flat),
        "median_return": value,
        "episodes": 4,
        "episodes_per_environment": 2,
        "num_environments": 2,
        "scan_steps_per_environment": 10,
        "returns_by_environment": rows,
    }


def _write_matrix(
    root: Path,
    *,
    tier: str = "pilot",
    models: list[str] | None = None,
    environments: dict[str, int] | None = None,
    seeds: list[int] | None = None,
) -> tuple[dict[str, Any], dict[tuple[str, str, int], Path]]:
    selected_models = models or MODELS
    selected_environments = environments or ENVIRONMENTS
    selected_seeds = seeds or SEEDS
    registration = {
        "schema_version": 1,
        "status": "frozen",
        "evidence_tier": tier,
        "matrix_kind": "primary_comparison",
        "models": selected_models,
        "environments": [
            {"id": environment, "total_steps": budget}
            for environment, budget in selected_environments.items()
        ],
        "seeds": selected_seeds,
        "learner": {
            "num_envs": 2,
            "rollout_steps": 2,
            "update_epochs": 1,
            "learning_rate": 0.001,
        },
        "evaluation_episodes_per_env": 2,
        "require_gpu": True,
        "quick": False,
    }
    atomic_write_json(root / "registration.json", registration)
    cells = []
    paths: dict[tuple[str, str, int], Path] = {}
    for environment in selected_environments:
        for model in selected_models:
            for seed in selected_seeds:
                identity = (environment, model, seed)
                configuration = _configuration(environment, model, seed, tier=tier)
                configuration_sha256 = canonical_json_sha256(configuration)
                relative = f"cells/{environment}-{model}-{seed}.json"
                cells.append(
                    {
                        "cell_id": registered_cell_id(
                            environment, model, seed, configuration_sha256
                        ),
                        "environment": environment,
                        "model": model,
                        "seed": seed,
                        "configuration_sha256": configuration_sha256,
                        "artifact_path": relative,
                    }
                )
                paths[identity] = root / relative
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "frozen",
        "matrix_kind": "primary_comparison",
        "models": selected_models,
        "environments": list(selected_environments),
        "seeds": selected_seeds,
        "provenance": deepcopy(PROVENANCE),
        "cells": cells,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_write_json(root / "frozen_manifest.json", manifest)
    for identity, path in paths.items():
        environment, model, seed = identity
        configuration = _configuration(environment, model, seed, tier=tier)
        configuration_sha256 = canonical_json_sha256(configuration)
        value = float(seed + (10 if model == "arcmind" else 0))
        artifact = {
            "schema_version": 4,
            "status": f"development_{tier}_not_for_paper",
            "matrix_manifest_sha256": manifest["manifest_sha256"],
            "cell_id": registered_cell_id(environment, model, seed, configuration_sha256),
            "configuration_sha256": configuration_sha256,
            "configuration": configuration,
            "environment": environment,
            "model": model,
            "seed": seed,
            "environment_source": deepcopy(configuration["environment_source"]),
            "environment_reference": deepcopy(configuration["environment_reference"]),
            "parameter_count": configuration["parameter_count"],
            "effective_parameter_count": configuration["effective_parameter_count"],
            "arcmind_target_parameter_count": configuration["arcmind_target_parameter_count"],
            "parameter_ratio": configuration["parameter_ratio"],
            "provenance": deepcopy(PROVENANCE),
            "actual_environment_steps": selected_environments[environment],
            "ppo": deepcopy(configuration["ppo"]),
            "evaluation_episodes_per_environment": 2,
            "evaluation_max_episode_steps": 5,
            "actual_evaluation_steps_per_environment": 10,
            "actual_evaluation_transitions": 20,
            "evaluation": _evaluation(value),
            "training_history": [
                {
                    "environment_steps": selected_environments[environment] / 2,
                    "mean_recent_return": None,
                },
                {
                    "environment_steps": float(selected_environments[environment]),
                    "mean_recent_return": value,
                },
            ],
        }
        atomic_write_json(path, artifact)
    return manifest, paths


def _rewrite(path: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _write_integrity_indexes(
    root: Path,
    manifest: dict[str, Any],
    paths: dict[tuple[str, str, int], Path],
) -> None:
    completed_cells = []
    for cell in manifest["cells"]:
        identity = (cell["environment"], cell["model"], cell["seed"])
        artifact_path = paths[identity]
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


def test_complete_matrix_is_deterministic_and_explicitly_not_for_paper(
    tmp_path: Path,
) -> None:
    matrix_root = tmp_path / "matrix"
    manifest, _ = _write_matrix(matrix_root)

    first = build_development_aggregate(matrix_root)
    second = build_development_aggregate(matrix_root)
    output = tmp_path / "aggregate.json"
    written = aggregate_development(matrix_root, output)

    assert first == second == written
    assert first["status"] == "development_pilot_aggregate_not_for_paper"
    assert first["not_for_paper"] is True
    assert first["matrix_manifest_sha256"] == manifest["manifest_sha256"]
    assert first["statistics"]["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
    assert first["environment_budgets"] == ENVIRONMENTS
    assert first["frozen_semantic_contract"] == {
        "environment_source_in_every_configuration": True,
        "parameter_match_in_every_configuration": True,
        "artifact_parameter_match_validated": True,
    }
    assert len(first["groups"]) == 4
    assert len(first["paired_differences_against_arcmind"]) == 2
    group = next(
        item
        for item in first["groups"]
        if item["environment"] == "short" and item["model"] == "arcmind"
    )
    assert [item["seed"] for item in group["raw_seed_values"]] == SEEDS
    assert group["raw_seed_values"][0]["returns_by_environment"] == [
        [16.0, 17.0],
        [17.0, 18.0],
    ]
    assert set(group["final_seed_mean_return"]) == {"mean", "median", "iqm"}
    assert json.loads(output.read_text(encoding="utf-8")) == first


@pytest.mark.parametrize(
    "relative_output",
    [".", "registration.json", "frozen_manifest.json", "derived/aggregate.json"],
)
def test_aggregate_output_cannot_target_raw_matrix_root(
    tmp_path: Path,
    relative_output: str,
) -> None:
    matrix_root = tmp_path / "matrix"
    _, paths = _write_matrix(matrix_root)
    paths[("short", "gru", 19)].unlink()
    output = matrix_root / relative_output

    with pytest.raises(DevelopmentAggregationError, match="outside the raw matrix root"):
        aggregate_development(matrix_root, output)


@pytest.mark.parametrize(
    "environment",
    [
        "tmaze_10-perfect-memory",
        "rocksample_11_11-fully-observable",
        "battleship_10-perfect-recall",
        "Navix-DMLab-Maze-01-fully-observable",
        "Walker-F-v0",
        "HalfCheetah-F-v0",
    ],
)
def test_primary_matrix_rejects_upper_reference_aliases(
    tmp_path: Path,
    environment: str,
) -> None:
    _write_matrix(tmp_path)
    _rewrite(
        tmp_path / "registration.json",
        lambda value: value.update(environments=[{"id": environment, "total_steps": 8_192}]),
    )

    with pytest.raises(DevelopmentAggregationError, match="upper-reference aliases"):
        build_development_aggregate(tmp_path)


def test_upper_reference_matrix_rejects_primary_environment(tmp_path: Path) -> None:
    _write_matrix(tmp_path)
    _rewrite(
        tmp_path / "registration.json",
        lambda value: value.update(
            matrix_kind="upper_reference",
            models=["memoryless_mlp"],
        ),
    )

    with pytest.raises(DevelopmentAggregationError, match="non-reference environments"):
        build_development_aggregate(tmp_path)


def test_arbitrary_nonempty_seed_count_and_smoke_status(tmp_path: Path) -> None:
    _write_matrix(tmp_path, tier="smoke", seeds=[5])

    result = build_development_aggregate(tmp_path)

    assert result["status"] == "development_smoke_aggregate_not_for_paper"
    summary = result["groups"][0]["final_seed_mean_return"]
    assert summary["mean"]["bootstrap_95_ci"][0] == summary["mean"]["estimate"]
    assert summary["iqm"]["estimate"] == summary["mean"]["estimate"]


def test_incomplete_cartesian_cell_fails(tmp_path: Path) -> None:
    _, paths = _write_matrix(tmp_path)
    paths[("short", "gru", 19)].unlink()

    with pytest.raises(DevelopmentAggregationError, match="does not exist"):
        build_development_aggregate(tmp_path)


def test_wrong_tier_and_registered_final_registration_fail(tmp_path: Path) -> None:
    _, paths = _write_matrix(tmp_path)
    _rewrite(
        paths[("short", "gru", 19)],
        lambda value: value.update(status="development_smoke_not_for_paper"),
    )
    with pytest.raises(DevelopmentAggregationError, match="status"):
        build_development_aggregate(tmp_path)

    other = tmp_path / "registered"
    _write_matrix(other)
    _rewrite(
        other / "registration.json",
        lambda value: value.update(evidence_tier="registered_final"),
    )
    with pytest.raises(DevelopmentAggregationError, match="smoke.*pilot"):
        build_development_aggregate(other)


def test_provenance_drift_fails(tmp_path: Path) -> None:
    _, paths = _write_matrix(tmp_path)
    _rewrite(
        paths[("long", "arcmind", 23)],
        lambda value: value["provenance"].update(pobax_commit="a" * 40),
    )

    with pytest.raises(DevelopmentAggregationError, match="provenance drifted"):
        build_development_aggregate(tmp_path)


def test_configuration_hash_and_manifest_corruption_fail(tmp_path: Path) -> None:
    _, paths = _write_matrix(tmp_path)
    _rewrite(
        paths[("short", "arcmind", 7)],
        lambda value: value["configuration"].update(seed=999),
    )
    with pytest.raises(DevelopmentAggregationError, match="configuration_sha256"):
        build_development_aggregate(tmp_path)

    other = tmp_path / "manifest"
    _write_matrix(other)
    _rewrite(
        other / "frozen_manifest.json",
        lambda value: value.update(models=["arcmind"]),
    )
    with pytest.raises(DevelopmentAggregationError, match="canonical content"):
        build_development_aggregate(other)


def test_raw_return_mean_median_and_count_mismatches_fail(tmp_path: Path) -> None:
    for name, mutation, message in [
        (
            "mean",
            lambda value: value["evaluation"].update(mean_return=999.0),
            "mean_return disagrees",
        ),
        (
            "median",
            lambda value: value["evaluation"].update(median_return=999.0),
            "median_return disagrees",
        ),
        (
            "count",
            lambda value: value["evaluation"].update(episodes=99),
            "episode counts",
        ),
    ]:
        root = tmp_path / name
        _, paths = _write_matrix(root)
        _rewrite(paths[("short", "gru", 7)], mutation)
        with pytest.raises(DevelopmentAggregationError, match=message):
            build_development_aggregate(root)


def test_budget_evaluation_and_history_contracts_fail(tmp_path: Path) -> None:
    for name, mutation, message in [
        (
            "steps",
            lambda value: value.update(actual_environment_steps=4),
            "actual_environment_steps",
        ),
        (
            "history",
            lambda value: value["training_history"][-1].update(environment_steps=16_000.0),
            "final step",
        ),
        (
            "evaluation",
            lambda value: value.update(actual_evaluation_steps_per_environment=9),
            "evaluation contract",
        ),
    ]:
        root = tmp_path / name
        _, paths = _write_matrix(root)
        _rewrite(paths[("long", "arcmind", 19)], mutation)
        with pytest.raises(DevelopmentAggregationError, match=message):
            build_development_aggregate(root)


def test_current_artifact_schema_is_required(tmp_path: Path) -> None:
    _, paths = _write_matrix(tmp_path)
    _rewrite(
        paths[("short", "gru", 7)],
        lambda value: value.update(schema_version=3),
    )

    with pytest.raises(DevelopmentAggregationError, match="current schema 4"):
        build_development_aggregate(tmp_path)


def test_parameter_matching_and_environment_semantics_fail_closed(tmp_path: Path) -> None:
    for name, mutation, message in [
        (
            "ratio",
            lambda value: value.update(parameter_ratio=1.2),
            "parameter_ratio",
        ),
        (
            "count",
            lambda value: value.update(parameter_count=1_001),
            "parameter_ratio",
        ),
        (
            "source",
            lambda value: value["environment_source"].update(perfect_memory=True),
            "environment_source",
        ),
    ]:
        root = tmp_path / name
        _, paths = _write_matrix(root)
        _rewrite(paths[("short", "gru", 7)], mutation)
        with pytest.raises(DevelopmentAggregationError, match=message):
            build_development_aggregate(root)


def test_optional_completion_and_checksum_indexes_are_validated(tmp_path: Path) -> None:
    manifest, paths = _write_matrix(tmp_path)
    _write_integrity_indexes(tmp_path, manifest, paths)

    result = build_development_aggregate(tmp_path)

    assert result["integrity_indexes"] == {
        "completion_index_present_and_validated": True,
        "checksums_present_and_validated": True,
    }

    artifact_path = paths[("short", "gru", 7)]
    artifact_path.write_bytes(artifact_path.read_bytes() + b" ")
    with pytest.raises(DevelopmentAggregationError, match="artifact_sha256"):
        build_development_aggregate(tmp_path)
