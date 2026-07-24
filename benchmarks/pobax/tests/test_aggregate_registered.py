from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import benchmarks.pobax.aggregate_registered as aggregate_registered_module
from benchmarks.pobax.aggregate_registered import (
    BOOTSTRAP_RESAMPLES,
    RegisteredAggregationError,
    _validate_frozen_configuration,
    _validate_manifest_tuning_selection,
    aggregate_registered,
    build_registered_aggregate,
    interquartile_mean,
)
from benchmarks.pobax.implementation_provenance import IMPLEMENTATION_SOURCE_ALGORITHM
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
ENVIRONMENTS = ["tmaze_10"]
SEEDS = [11, 22, 33, 44, *range(100, 126)]
PROVENANCE = {
    "git": {"commit": "1" * 40, "dirty": False, "diff_sha256": None},
    "dependency_lock_sha256": "2" * 64,
    "pobax_commit": "3" * 40,
    "navix_commit": "4" * 40,
    "runtime_contract": {
        "python": {"implementation": "CPython", "version": "3.12.3"},
        "packages": {
            "brax": "0.14.2",
            "gymnax": "0.0.9",
            "jax": "0.6.2",
            "jax-cuda12-pjrt": "0.6.2",
            "jax-cuda12-plugin": "0.6.2",
            "jaxlib": "0.6.2",
            "Navix": "0.0.1",
            "numpy": "2.5.1",
            "optax": "0.2.8",
            "pobax": "0.0.1",
        },
        "jax_backend": "gpu",
        "jax_enable_x64": False,
        "devices": [{"platform": "gpu", "device_kind": "Test GPU"}],
    },
}
TRAIN_STEPS = {
    "tmaze_10": 1_000_000,
    "rocksample_11_11": 5_000_000,
    "Walker-F-v0": 50_000_000,
}
EVALUATION_EPISODES = {
    "tmaze_10": 2,
    "rocksample_11_11": 1,
    "Walker-F-v0": 2,
}
EVALUATION_HORIZON = 1_000
OPTIMIZER_METRICS = {
    "loss": 0.5,
    "actor_loss": 0.1,
    "value_loss": 0.8,
    "entropy": 0.2,
    "approximate_kl": 0.01,
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


def _configuration(
    environment: str,
    model: str,
    seed: int,
    *,
    schema_version: int = 1,
    comparison_profile: str | None = None,
    evaluation_episodes_per_env: int | None = None,
) -> dict[str, Any]:
    total_steps = TRAIN_STEPS[environment]
    evaluation_episodes = (
        EVALUATION_EPISODES[environment]
        if evaluation_episodes_per_env is None
        else evaluation_episodes_per_env
    )
    configuration = {
        "schema_version": schema_version,
        "evidence_tier": "registered_final",
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
            "total_steps": total_steps,
            "num_envs": 2,
            "rollout_steps": 2,
            "update_epochs": 1,
            "learning_rate": 0.001,
        },
        "evaluation_episodes_per_environment": evaluation_episodes,
        "evaluation_max_episode_steps": EVALUATION_HORIZON,
        "dependency_lock_sha256": PROVENANCE["dependency_lock_sha256"],
        "pobax_commit": PROVENANCE["pobax_commit"],
        "navix_commit": PROVENANCE["navix_commit"],
        "runtime_contract": deepcopy(PROVENANCE["runtime_contract"]),
    }
    if schema_version == 2:
        configuration["ppo"].update(
            rollout_steps=2,
            update_epochs=1,
            num_minibatches=1,
            learning_rate=0.001,
            gae_lambda=0.95,
            entropy_coefficient=0.01,
            anneal_learning_rate=True,
            step_budget_mode=(
                "floor" if comparison_profile == "pobax_author_semantics" else "exact"
            ),
        )
        realized_steps = total_steps // 4 * 4
        configuration.update(
            comparison_profile=comparison_profile,
            requested_environment_steps=total_steps,
            realized_environment_steps=realized_steps,
        )
    return configuration


def _evaluation(value: float, *, episodes_per_environment: int = 2) -> dict[str, Any]:
    rows = [
        [value - 0.5, value + 0.5][:episodes_per_environment],
        [value - 1.0, value + 1.0][:episodes_per_environment],
    ]
    flat = [item for row in rows for item in row]
    ordered = sorted(flat)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "mean_return": sum(flat) / len(flat),
        "median_return": median,
        "episodes": len(flat),
        "episodes_per_environment": episodes_per_environment,
        "num_environments": len(rows),
        "scan_steps_per_environment": episodes_per_environment * EVALUATION_HORIZON,
        "returns_by_environment": rows,
    }


def test_registered_aggregation_revalidates_schema_v4_tuning_binding(
    tmp_path,
    monkeypatch,
):
    learner = {
        "num_envs": 8,
        "rollout_steps": 125,
        "update_epochs": 4,
        "num_minibatches": 4,
        "learning_rate": 0.001,
        "gae_lambda": 0.95,
        "entropy_coefficient": 0.01,
        "anneal_learning_rate": False,
    }
    tuning_aggregate = {
        "schema_version": 1,
        "status": "development_tuning_selection_aggregate_not_for_paper",
        "evidence_tier": "development_tuning",
        "matrix_kind": "hyperparameter_selection",
        "not_for_paper": True,
        "registration_sha256": "1" * 64,
        "matrix_manifest_sha256": "2" * 64,
        "completion_index_sha256": "3" * 64,
        "checksum_manifest_sha256": "4" * 64,
        "provenance": {
            **deepcopy(PROVENANCE),
            "implementation_source": deepcopy(IMPLEMENTATION_SOURCE),
        },
        "environments": ["tmaze_10"],
        "seeds": [1103, 2207, 3301, 4409, 5519],
        "integrity_indexes": {
            "completion_index_present_and_validated": True,
            "checksums_present_and_validated": True,
        },
        "frozen_semantic_contract": {
            "environment_source_in_every_configuration": True,
            "parameter_match_in_every_configuration": True,
            "artifact_parameter_match_validated": True,
        },
        "selection_eligibility": {
            "eligible_for_hyperparameter_selection": True,
            "eligible_for_architecture_selection": False,
            "eligible_for_checkpoint_selection": False,
            "eligible_for_registered_final_evidence": False,
            "eligible_for_paper_performance_claims": False,
            "selection_scope": "candidate_within_model_family_and_environment",
        },
        "candidate_selection": [
            {
                "environment": "tmaze_10",
                "model_family": "ordered_memory",
                "implementation_model": "arcmind",
                "winner_candidate_id": "ordered_memory.lr_high",
                "ranking": [{"rank": 1, "candidate_id": "ordered_memory.lr_high"}],
            }
        ],
        "groups": [
            {
                "environment": "tmaze_10",
                "candidate_id": "ordered_memory.lr_high",
                "model_family": "ordered_memory",
                "implementation_model": "arcmind",
                "learner": learner,
                "implementation_source_sha256": IMPLEMENTATION_SOURCE["sha256"],
            }
        ],
    }
    raw_matrix_path = tmp_path / "raw-tuning"
    raw_matrix_path.mkdir()
    (raw_matrix_path / "completion_index.json").write_bytes(b"tuning completion\n")
    (raw_matrix_path / "checksums.sha256").write_bytes(b"tuning checksums\n")
    tuning_aggregate["completion_index_sha256"] = sha256_file(
        raw_matrix_path / "completion_index.json"
    )
    tuning_aggregate["checksum_manifest_sha256"] = sha256_file(raw_matrix_path / "checksums.sha256")
    aggregate_path = tmp_path / "tuning-selection.json"
    atomic_write_json(aggregate_path, tuning_aggregate)
    binding = {
        "raw_matrix_path": "raw-tuning",
        "aggregate_path": "tuning-selection.json",
        "aggregate_sha256": sha256_file(aggregate_path),
        "source_registration_sha256": tuning_aggregate["registration_sha256"],
        "source_manifest_sha256": tuning_aggregate["matrix_manifest_sha256"],
        "source_completion_index_sha256": tuning_aggregate["completion_index_sha256"],
        "source_checksum_manifest_sha256": tuning_aggregate["checksum_manifest_sha256"],
        "source_implementation_sha256": IMPLEMENTATION_SOURCE["sha256"],
        "selections": [
            {
                "environment": "tmaze_10",
                "model_family": "ordered_memory",
                "implementation_model": "arcmind",
                "candidate_id": "ordered_memory.lr_high",
                "learner": learner,
                "implementation_source_sha256": IMPLEMENTATION_SOURCE["sha256"],
            }
        ],
    }
    monkeypatch.setattr(
        aggregate_registered_module,
        "_REPOSITORY_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        aggregate_registered_module,
        "build_development_aggregate",
        lambda path: deepcopy(tuning_aggregate),
    )

    normalized = _validate_manifest_tuning_selection(
        binding,
        models=("arcmind",),
        environments=("tmaze_10",),
        final_seeds=tuple(range(10_000, 10_030)),
        final_provenance=tuning_aggregate["provenance"],
    )

    assert normalized["aggregate_sha256"] == sha256_file(aggregate_path)
    assert normalized["selections"][0]["learner"] == learner


def test_schema_v4_configuration_must_match_selected_winner_learner():
    configuration = _configuration(
        "tmaze_10",
        "arcmind",
        11,
        schema_version=2,
        comparison_profile="arcmind_shared_comparison",
    )
    configuration.update(
        schema_version=4,
        candidate_id="ordered_memory.lr_high",
        model_family="ordered_memory",
        implementation_model="arcmind",
        tuning_aggregate_sha256="5" * 64,
        tuning_completion_index_sha256="6" * 64,
        tuning_checksum_manifest_sha256="7" * 64,
        tuning_implementation_source_sha256=IMPLEMENTATION_SOURCE["sha256"],
        implementation_source=deepcopy(IMPLEMENTATION_SOURCE),
    )
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
    selection = {
        "candidate_id": "ordered_memory.lr_high",
        "model_family": "ordered_memory",
        "implementation_model": "arcmind",
        "tuning_aggregate_sha256": "5" * 64,
        "tuning_completion_index_sha256": "6" * 64,
        "tuning_checksum_manifest_sha256": "7" * 64,
        "tuning_implementation_source_sha256": IMPLEMENTATION_SOURCE["sha256"],
        "implementation_source_sha256": IMPLEMENTATION_SOURCE["sha256"],
        "learner": learner,
    }
    provenance = {
        **deepcopy(PROVENANCE),
        "implementation_source": deepcopy(IMPLEMENTATION_SOURCE),
    }

    assert _validate_frozen_configuration(
        configuration,
        identity=("tmaze_10", "arcmind", 11),
        provenance=provenance,
        field="configuration",
        selection=selection,
    ) == (1_000_000, 2, EVALUATION_HORIZON)

    configuration["ppo"]["learning_rate"] = 0.0005
    with pytest.raises(RegisteredAggregationError, match="learner drifts"):
        _validate_frozen_configuration(
            configuration,
            identity=("tmaze_10", "arcmind", 11),
            provenance=provenance,
            field="configuration",
            selection=selection,
        )


def _write_matrix(
    tmp_path: Path,
    *,
    models: list[str] | None = None,
    environments: list[str] | None = None,
    seeds: list[int] | None = None,
    matrix_kind: str = "primary_comparison",
    schema_version: int = 1,
    comparison_profile: str | None = None,
    evaluation_episodes_per_env: int | None = None,
) -> tuple[Path, dict[str, Any], dict[tuple[str, str, int], Path]]:
    selected_models = models or MODELS
    selected_environments = environments or ENVIRONMENTS
    selected_seeds = seeds or SEEDS
    cells = []
    paths: dict[tuple[str, str, int], Path] = {}
    for environment in selected_environments:
        for model in selected_models:
            for seed in selected_seeds:
                configuration = _configuration(
                    environment,
                    model,
                    seed,
                    schema_version=schema_version,
                    comparison_profile=comparison_profile,
                    evaluation_episodes_per_env=evaluation_episodes_per_env,
                )
                configuration_sha256 = canonical_json_sha256(configuration)
                identity = (environment, model, seed)
                relative = f"cells/{environment}-{model}-{seed}.json"
                cells.append(
                    {
                        "cell_id": registered_cell_id(
                            environment,
                            model,
                            seed,
                            configuration_sha256,
                        ),
                        "environment": environment,
                        "model": model,
                        "seed": seed,
                        "configuration_sha256": configuration_sha256,
                        "artifact_path": relative,
                    }
                )
                paths[identity] = tmp_path / relative
    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "status": "frozen",
        "matrix_kind": matrix_kind,
        "models": selected_models,
        "environments": selected_environments,
        "seeds": selected_seeds,
        "provenance": deepcopy(PROVENANCE),
        "cells": cells,
    }
    manifest_sha256 = canonical_json_sha256(manifest)
    manifest["manifest_sha256"] = manifest_sha256
    manifest_path = tmp_path / "matrix.json"
    atomic_write_json(manifest_path, manifest)
    registration = {
        "schema_version": schema_version,
        "status": "frozen",
        "evidence_tier": "registered_final",
        "matrix_kind": matrix_kind,
        "models": selected_models,
        "environments": [
            {"id": environment, "total_steps": TRAIN_STEPS[environment]}
            for environment in selected_environments
        ],
        "seeds": selected_seeds,
        "learner": {
            "num_envs": 2,
            "rollout_steps": 2,
            "update_epochs": 1,
            "learning_rate": 0.001,
        },
        "evaluation_episodes_per_env": (
            EVALUATION_EPISODES[selected_environments[0]]
            if evaluation_episodes_per_env is None
            else evaluation_episodes_per_env
        ),
        "require_gpu": True,
        "quick": False,
    }
    if schema_version == 2:
        registration.update(
            comparison_profile=comparison_profile,
            learner={
                **registration["learner"],
                "num_minibatches": 1,
                "gae_lambda": 0.95,
                "entropy_coefficient": 0.01,
                "anneal_learning_rate": True,
            },
        )
    atomic_write_json(tmp_path / "registration.json", registration)

    values = {
        ("tmaze_10", "arcmind", 11): 1.0,
        ("tmaze_10", "arcmind", 22): 2.0,
        ("tmaze_10", "arcmind", 33): 3.0,
        ("tmaze_10", "arcmind", 44): 100.0,
        ("tmaze_10", "gru", 11): 0.0,
        ("tmaze_10", "gru", 22): 1.0,
        ("tmaze_10", "gru", 33): 5.0,
        ("tmaze_10", "gru", 44): 90.0,
    }
    for identity, path in paths.items():
        environment, model, seed = identity
        configuration = _configuration(
            environment,
            model,
            seed,
            schema_version=schema_version,
            comparison_profile=comparison_profile,
            evaluation_episodes_per_env=evaluation_episodes_per_env,
        )
        total_steps = configuration.get(
            "realized_environment_steps",
            configuration["ppo"]["total_steps"],
        )
        evaluation_episodes = configuration["evaluation_episodes_per_environment"]
        evaluation_steps = evaluation_episodes * configuration["evaluation_max_episode_steps"]
        value = values.get(identity, float(seed))
        artifact = {
            "schema_version": 5 if schema_version == 2 else 4,
            "status": "registered_final_complete",
            "matrix_manifest_sha256": manifest_sha256,
            "cell_id": registered_cell_id(
                environment,
                model,
                seed,
                canonical_json_sha256(configuration),
            ),
            "configuration_sha256": canonical_json_sha256(configuration),
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
            "actual_environment_steps": total_steps,
            "ppo": deepcopy(configuration["ppo"]),
            "evaluation_episodes_per_environment": evaluation_episodes,
            "evaluation_max_episode_steps": configuration["evaluation_max_episode_steps"],
            "actual_evaluation_steps_per_environment": evaluation_steps,
            "actual_evaluation_transitions": evaluation_steps * configuration["ppo"]["num_envs"],
            "evaluation": _evaluation(
                value,
                episodes_per_environment=evaluation_episodes,
            ),
            "training": {**OPTIMIZER_METRICS, "mean_recent_return": value},
            "training_history": [
                {
                    **OPTIMIZER_METRICS,
                    "environment_steps": float(total_steps // 2),
                    "mean_recent_return": value - 1.0,
                },
                {
                    **OPTIMIZER_METRICS,
                    "environment_steps": float(total_steps),
                    "mean_recent_return": value,
                },
            ],
        }
        if schema_version == 2:
            artifact.update(
                comparison_profile=comparison_profile,
                requested_environment_steps=configuration["requested_environment_steps"],
                realized_environment_steps=configuration["realized_environment_steps"],
            )
        atomic_write_json(path, artifact)
    completed_cells = []
    for cell in cells:
        identity = (cell["environment"], cell["model"], cell["seed"])
        artifact_path = paths[identity]
        log_path = artifact_path.with_suffix(".log")
        atomic_write_bytes(log_path, b"registered test log\n")
        completed_cells.append(
            {
                **cell,
                "artifact_sha256": sha256_file(artifact_path),
                "log_path": log_path.relative_to(tmp_path).as_posix(),
                "log_sha256": sha256_file(log_path),
            }
        )
    atomic_write_json(
        tmp_path / "completion_index.json",
        {
            "schema_version": 1,
            "status": "complete",
            "manifest_sha256": manifest_sha256,
            "planned_cells": len(cells),
            "completed_cells": len(cells),
            "cells": completed_cells,
        },
    )
    write_checksum_manifest(tmp_path)
    return manifest_path, manifest, paths


def _rewrite_json(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _rewrite_manifest(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    value_without_hash = dict(value)
    value_without_hash.pop("manifest_sha256", None)
    value["manifest_sha256"] = canonical_json_sha256(value_without_hash)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _refresh_integrity(root: Path, paths: dict[tuple[str, str, int], Path]) -> None:
    completion_path = root / "completion_index.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    for cell in completion["cells"]:
        identity = (cell["environment"], cell["model"], cell["seed"])
        cell["artifact_sha256"] = sha256_file(paths[identity])
    completion_path.write_text(
        json.dumps(completion, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    checksum_path = root / "checksums.sha256"
    checksum_path.unlink()
    write_checksum_manifest(root)


def test_iqm_fractional_boundaries() -> None:
    assert interquartile_mean([1.0]) == pytest.approx(1.0)
    assert interquartile_mean([1.0, 2.0, 3.0, 100.0]) == pytest.approx(2.5)
    assert interquartile_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    with pytest.raises(RegisteredAggregationError, match="non-empty finite"):
        interquartile_mean([])
    with pytest.raises(RegisteredAggregationError, match="non-empty finite"):
        interquartile_mean([float("nan")])


def test_complete_matrix_aggregates_raw_seeds_pairs_curves_and_canonical_json(
    tmp_path: Path,
) -> None:
    matrix_root = tmp_path / "matrix"
    manifest_path, manifest, _ = _write_matrix(matrix_root)
    output = tmp_path / "aggregate.json"

    result = aggregate_registered(manifest_path, output)

    assert result["matrix_manifest_sha256"] == manifest["manifest_sha256"]
    assert result["statistics"]["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
    assert result["statistical_unit"] == "seed"
    assert result["raw_integrity"]["completion_index_validated"] is True
    assert result["raw_integrity"]["checksum_inventory_validated"] is True
    assert output.read_bytes().endswith(b"\n")
    assert json.loads(output.read_text(encoding="utf-8")) == result


@pytest.mark.parametrize(
    ("missing_path", "message"),
    [
        ("registration.json", "frozen registration"),
        ("completion_index.json", "completion index"),
        ("checksums.sha256", "checksum manifest"),
    ],
)
def test_registered_aggregate_requires_every_raw_integrity_index(
    tmp_path: Path,
    missing_path: str,
    message: str,
) -> None:
    manifest_path, _, _ = _write_matrix(tmp_path)
    (tmp_path / missing_path).unlink()

    with pytest.raises(RegisteredAggregationError, match=message):
        build_registered_aggregate(manifest_path)


def test_registered_aggregate_rejects_missing_or_stale_logs_and_hidden_files(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing-log"
    manifest_path, _, paths = _write_matrix(missing_root)
    next(iter(paths.values())).with_suffix(".log").unlink()
    with pytest.raises(RegisteredAggregationError, match="log"):
        build_registered_aggregate(manifest_path)

    stale_root = tmp_path / "stale-log"
    manifest_path, _, paths = _write_matrix(stale_root)
    next(iter(paths.values())).with_suffix(".log").write_bytes(b"drifted log\n")
    with pytest.raises(RegisteredAggregationError, match="log_sha256"):
        build_registered_aggregate(manifest_path)

    hidden_root = tmp_path / "hidden"
    manifest_path, _, _ = _write_matrix(hidden_root)
    (hidden_root / "unchecksummed.bin").write_bytes(b"hidden")
    with pytest.raises(RegisteredAggregationError, match="checksum inventory differs"):
        build_registered_aggregate(manifest_path)


def test_registered_aggregate_rejects_checksummed_attempt_evidence(
    tmp_path: Path,
) -> None:
    manifest_path, _, _ = _write_matrix(tmp_path)
    atomic_write_bytes(
        tmp_path / "cells" / ("orphan.attempt-" + "a" * 32 + ".failed.log"),
        b"failed child evidence\n",
    )
    (tmp_path / "checksums.sha256").unlink()
    write_checksum_manifest(tmp_path)

    with pytest.raises(RegisteredAggregationError, match="canonical matrix evidence"):
        build_registered_aggregate(manifest_path)


def test_registered_aggregate_rejects_noncanonical_completion_bytes(
    tmp_path: Path,
) -> None:
    manifest_path, _, _ = _write_matrix(tmp_path)
    completion_path = tmp_path / "completion_index.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion_path.write_text(
        json.dumps(completion, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "checksums.sha256").unlink()
    write_checksum_manifest(tmp_path)

    with pytest.raises(RegisteredAggregationError, match="completion_index is not canonical JSON"):
        build_registered_aggregate(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["learner"].update(learning_rate=0.123),
            "learner drifts",
        ),
        (
            lambda value: value.update(evaluation_episodes_per_env=3),
            "evaluation episodes drift",
        ),
        (
            lambda value: value["environments"][0].update(total_steps=999_999),
            "total_steps drifts",
        ),
        (
            lambda value: value.update(comparison_profile="pobax_author_semantics"),
            "comparison_profile drifts",
        ),
        (
            lambda value: value.update(quick=True),
            "registration.quick must be false",
        ),
    ],
)
def test_schema_v2_registration_semantics_are_bound_to_artifacts(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    manifest_path, _, _ = _write_matrix(
        tmp_path,
        schema_version=2,
        comparison_profile="arcmind_shared_comparison",
    )
    registration_path = tmp_path / "registration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    mutation(registration)
    registration_path.write_text(
        json.dumps(
            registration,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "checksums.sha256").unlink()
    write_checksum_manifest(tmp_path)

    with pytest.raises(RegisteredAggregationError, match=message):
        build_registered_aggregate(manifest_path)


def test_schema_v1_evaluation_registration_is_bound_to_artifacts(
    tmp_path: Path,
) -> None:
    manifest_path, _, _ = _write_matrix(tmp_path)
    registration_path = tmp_path / "registration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    registration["evaluation_episodes_per_env"] = 999
    registration_path.write_text(
        json.dumps(
            registration,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "checksums.sha256").unlink()
    write_checksum_manifest(tmp_path)

    with pytest.raises(RegisteredAggregationError, match="evaluation episodes drift"):
        build_registered_aggregate(manifest_path)


def test_schema_v2_registered_matrix_validates_explicit_step_accounting(
    tmp_path: Path,
) -> None:
    manifest_path, _, paths = _write_matrix(
        tmp_path,
        schema_version=2,
        comparison_profile="arcmind_shared_comparison",
    )

    result = build_registered_aggregate(manifest_path)

    assert result["status"] == "registered_matrix_aggregate"

    artifact_path = paths[("tmaze_10", "arcmind", SEEDS[0])]
    _rewrite_json(
        artifact_path,
        lambda value: value.update(realized_environment_steps=4),
    )
    with pytest.raises(RegisteredAggregationError, match="step accounting"):
        build_registered_aggregate(manifest_path)

    arcmind = next(
        group
        for group in result["groups"]
        if group["environment"] == "tmaze_10" and group["model"] == "arcmind"
    )
    assert [item["seed"] for item in arcmind["raw_seed_values"]] == SEEDS
    observed_returns = [item["mean_return"] for item in arcmind["raw_seed_values"]]
    assert observed_returns[:4] == pytest.approx([1.0, 2.0, 3.0, 100.0])
    assert observed_returns[4:] == pytest.approx([float(seed) for seed in SEEDS[4:]])
    summary = arcmind["final_seed_mean_return"]
    assert summary["mean"]["estimate"] == pytest.approx(
        sum(observed_returns) / len(observed_returns)
    )
    assert summary["median"]["estimate"] == pytest.approx(float(np.median(observed_returns)))
    assert summary["iqm"]["estimate"] == pytest.approx(interquartile_mean(observed_returns))
    assert arcmind["training_curve"]["environment_steps"] == [
        TRAIN_STEPS["tmaze_10"] // 2,
        TRAIN_STEPS["tmaze_10"],
    ]
    assert arcmind["training_curve"]["mean_return_by_step"] == pytest.approx(
        [float(np.mean(observed_returns)) - 1.0, float(np.mean(observed_returns))]
    )
    assert [item["auc_return_step"] for item in arcmind["training_curve"]["raw_seed_returns"]][
        :4
    ] == pytest.approx([250_000.0, 750_000.0, 1_250_000.0, 49_750_000.0])

    paired = result["paired_differences_against_arcmind"][0]
    assert paired["model"] == "gru"
    differences = [item["difference"] for item in paired["raw_seed_differences"]]
    assert differences[:4] == pytest.approx([-1.0, -1.0, 2.0, -10.0])
    assert differences[4:] == pytest.approx([0.0] * (len(SEEDS) - 4))
    assert paired["difference_summary"]["mean"]["estimate"] == pytest.approx(
        sum(differences) / len(differences)
    )


@pytest.mark.parametrize(
    "relative_output",
    [".", "matrix.json", "cells/aggregate.json"],
)
def test_aggregate_output_cannot_target_raw_matrix_root(
    tmp_path: Path,
    relative_output: str,
) -> None:
    matrix_root = tmp_path / "matrix"
    manifest_path, _, _ = _write_matrix(matrix_root)
    _rewrite_manifest(manifest_path, lambda value: value.update(status="draft"))
    output = matrix_root / relative_output

    with pytest.raises(RegisteredAggregationError, match="outside the raw matrix root"):
        aggregate_registered(manifest_path, output)


def test_bootstrap_is_deterministic_and_group_order_independent(tmp_path: Path) -> None:
    manifest_path, _, _ = _write_matrix(tmp_path)
    first = build_registered_aggregate(manifest_path)
    second = build_registered_aggregate(manifest_path)

    assert first == second
    arcmind = next(group for group in first["groups"] if group["model"] == "arcmind")
    interval = arcmind["final_seed_mean_return"]["mean"]["bootstrap_95_ci"]
    assert interval[0] <= arcmind["final_seed_mean_return"]["mean"]["estimate"] <= interval[1]


def test_upper_reference_matrix_aggregates_without_arcmind_pairs(tmp_path: Path) -> None:
    manifest_path, _, _ = _write_matrix(
        tmp_path,
        models=["memoryless_mlp"],
        environments=["Walker-F-v0"],
        matrix_kind="upper_reference",
    )

    result = build_registered_aggregate(manifest_path)

    assert result["matrix_kind"] == "upper_reference"
    assert len(result["groups"]) == 1
    assert result["paired_differences_against_arcmind"] == []


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
    manifest_path, _, _ = _write_matrix(tmp_path)
    _rewrite_manifest(
        manifest_path,
        lambda value: value.update(environments=[environment]),
    )

    with pytest.raises(RegisteredAggregationError, match="upper-reference aliases"):
        build_registered_aggregate(manifest_path)


def test_schema_v2_frozen_configuration_requires_every_learner_field() -> None:
    configuration = _configuration(
        "tmaze_10",
        "arcmind",
        SEEDS[0],
        schema_version=2,
        comparison_profile="arcmind_shared_comparison",
    )
    del configuration["ppo"]["gae_lambda"]

    with pytest.raises(
        RegisteredAggregationError,
        match="missing registered learner fields",
    ):
        _validate_frozen_configuration(
            configuration,
            identity=("tmaze_10", "arcmind", SEEDS[0]),
            provenance=PROVENANCE,
            field="configuration",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["training_history"][0].update(value_loss=None),
            r"training_history\[0\]\.value_loss must be a finite number",
        ),
        (
            lambda value: value["training"].update(entropy=None),
            r"training\.entropy must be a finite number",
        ),
    ],
)
def test_nonfinite_optimizer_metrics_fail_closed(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    manifest_path, _, paths = _write_matrix(tmp_path)
    _rewrite_json(paths[("tmaze_10", "arcmind", SEEDS[0])], mutation)

    with pytest.raises(RegisteredAggregationError, match=message):
        build_registered_aggregate(manifest_path)


def test_upper_reference_matrix_rejects_primary_environment(tmp_path: Path) -> None:
    manifest_path, _, _ = _write_matrix(
        tmp_path,
        models=["memoryless_mlp"],
        matrix_kind="upper_reference",
    )

    with pytest.raises(RegisteredAggregationError, match="non-reference environments"):
        build_registered_aggregate(manifest_path)


def test_frozen_configuration_requires_final_tier_identity_and_sources() -> None:
    configuration = _configuration("tmaze_10", "arcmind", 11)

    assert _validate_frozen_configuration(
        configuration,
        identity=("tmaze_10", "arcmind", 11),
        provenance=PROVENANCE,
        field="configuration",
    ) == (TRAIN_STEPS["tmaze_10"], 2, EVALUATION_HORIZON)

    invalid = deepcopy(configuration)
    invalid["evidence_tier"] = "pilot"
    with pytest.raises(RegisteredAggregationError, match="registered_final"):
        _validate_frozen_configuration(
            invalid,
            identity=("tmaze_10", "arcmind", 11),
            provenance=PROVENANCE,
            field="configuration",
        )


def test_frozen_configuration_requires_exact_parameter_and_environment_semantics() -> None:
    configuration = _configuration("tmaze_10", "arcmind", 11)

    for mutation, message in [
        (
            lambda value: value.update(parameter_ratio=1.2),
            "parameter_ratio",
        ),
        (
            lambda value: value.update(parameter_count=1_001),
            "parameter_ratio",
        ),
        (
            lambda value: value["environment_source"].update(perfect_memory=True),
            "environment_source",
        ),
        (
            lambda value: value.update(
                environment_reference={
                    "primary_environment": "tmaze_10",
                    "reference_class": "invalid",
                }
            ),
            "environment_reference",
        ),
    ]:
        invalid = deepcopy(configuration)
        mutation(invalid)
        with pytest.raises(RegisteredAggregationError, match=message):
            _validate_frozen_configuration(
                invalid,
                identity=("tmaze_10", "arcmind", 11),
                provenance=PROVENANCE,
                field="configuration",
            )


def test_partial_training_history_cannot_pass_as_complete(tmp_path: Path) -> None:
    manifest_path, _, paths = _write_matrix(tmp_path)
    _rewrite_json(
        paths[("tmaze_10", "gru", 22)],
        lambda value: value["training_history"][-1].__setitem__(
            "environment_steps",
            float(TRAIN_STEPS["tmaze_10"] - 1),
        ),
    )

    with pytest.raises(RegisteredAggregationError, match="final step"):
        build_registered_aggregate(manifest_path)


def test_manifest_hash_status_schema_and_unknown_fields_fail(tmp_path: Path) -> None:
    manifest_path, _, _ = _write_matrix(tmp_path)
    _rewrite_json(manifest_path, lambda value: value.__setitem__("models", ["arcmind"]))
    with pytest.raises(RegisteredAggregationError, match="manifest_sha256"):
        build_registered_aggregate(manifest_path)

    manifest_path, _, _ = _write_matrix(tmp_path / "status")
    _rewrite_manifest(manifest_path, lambda value: value.__setitem__("status", "draft"))
    with pytest.raises(RegisteredAggregationError, match="status"):
        build_registered_aggregate(manifest_path)

    manifest_path, _, _ = _write_matrix(tmp_path / "schema")
    _rewrite_manifest(manifest_path, lambda value: value.__setitem__("schema_version", 2))
    with pytest.raises(RegisteredAggregationError, match="registration identity"):
        build_registered_aggregate(manifest_path)

    manifest_path, _, _ = _write_matrix(tmp_path / "extra")
    _rewrite_manifest(manifest_path, lambda value: value.__setitem__("unexpected", True))
    with pytest.raises(RegisteredAggregationError, match="wrong fields"):
        build_registered_aggregate(manifest_path)


def test_missing_duplicate_and_out_of_matrix_cells_fail(tmp_path: Path) -> None:
    manifest_path, _, _ = _write_matrix(tmp_path / "missing")
    _rewrite_manifest(manifest_path, lambda value: value["cells"].pop())
    with pytest.raises(RegisteredAggregationError, match="missing Cartesian"):
        build_registered_aggregate(manifest_path)

    manifest_path, _, _ = _write_matrix(tmp_path / "duplicate")
    _rewrite_manifest(manifest_path, lambda value: value["cells"].append(value["cells"][0]))
    with pytest.raises(RegisteredAggregationError, match="duplicate manifest cell identity"):
        build_registered_aggregate(manifest_path)

    manifest_path, _, _ = _write_matrix(tmp_path / "outside")

    def outside(value: dict[str, Any]) -> None:
        value["cells"][0]["model"] = "lstm"

    _rewrite_manifest(manifest_path, outside)
    with pytest.raises(RegisteredAggregationError, match="outside"):
        build_registered_aggregate(manifest_path)


def test_duplicate_cell_id_and_artifact_path_fail(tmp_path: Path) -> None:
    manifest_path, _, _ = _write_matrix(tmp_path / "cell-id")

    def duplicate_id(value: dict[str, Any]) -> None:
        value["cells"][1]["cell_id"] = value["cells"][0]["cell_id"]

    _rewrite_manifest(manifest_path, duplicate_id)
    with pytest.raises(RegisteredAggregationError, match="cell_id does not match"):
        build_registered_aggregate(manifest_path)

    manifest_path, _, _ = _write_matrix(tmp_path / "path")

    def duplicate_path(value: dict[str, Any]) -> None:
        value["cells"][1]["artifact_path"] = value["cells"][0]["artifact_path"]

    _rewrite_manifest(manifest_path, duplicate_path)
    with pytest.raises(RegisteredAggregationError, match="duplicate manifest artifact path"):
        build_registered_aggregate(manifest_path)


@pytest.mark.parametrize(
    ("name", "mutate", "match"),
    [
        (
            "schema",
            lambda value: value.__setitem__("schema_version", 3),
            "current schema 4",
        ),
        (
            "status",
            lambda value: value.__setitem__("status", "development_pilot_not_for_paper"),
            "registered_final_complete",
        ),
        (
            "manifest",
            lambda value: value.__setitem__("matrix_manifest_sha256", "9" * 64),
            "different matrix manifest",
        ),
        (
            "seed",
            lambda value: value.__setitem__("seed", 999),
            "identity does not match",
        ),
        (
            "boolean-seed",
            lambda value: value.__setitem__("seed", True),
            "seed must be an integer",
        ),
        (
            "config-entry",
            lambda value: value["configuration"].__setitem__("extra", True),
            "does not match configuration",
        ),
        (
            "config-hash",
            lambda value: value.__setitem__("configuration_sha256", "8" * 64),
            "drifted from the manifest",
        ),
        (
            "provenance",
            lambda value: value["provenance"].__setitem__("pobax_commit", "a" * 40),
            "provenance drifted",
        ),
        (
            "dirty",
            lambda value: value["provenance"]["git"].__setitem__("dirty", True),
            "dirty must be false",
        ),
        (
            "runtime",
            lambda value: value["provenance"]["runtime_contract"].__setitem__(
                "jax_backend",
                "cpu",
            ),
            "provenance drifted",
        ),
        (
            "null-return",
            lambda value: value["evaluation"].__setitem__("mean_return", None),
            "finite number",
        ),
        (
            "wrong-return",
            lambda value: value["evaluation"].__setitem__("mean_return", 123.0),
            "disagrees with raw returns",
        ),
    ],
)
def test_artifact_drift_and_nonfinite_required_metrics_fail(
    tmp_path: Path,
    name: str,
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    manifest_path, _, paths = _write_matrix(tmp_path / name)
    _rewrite_json(paths[("tmaze_10", "gru", 22)], mutate)
    with pytest.raises(RegisteredAggregationError, match=match):
        build_registered_aggregate(manifest_path)


def test_unequal_episode_counts_and_step_grids_fail(tmp_path: Path) -> None:
    manifest_path, _, paths = _write_matrix(tmp_path / "episodes")

    def one_episode(value: dict[str, Any]) -> None:
        value["evaluation"] = _evaluation(1.0, episodes_per_environment=1)

    _rewrite_json(paths[("tmaze_10", "gru", 22)], one_episode)
    with pytest.raises(RegisteredAggregationError, match="frozen evaluation contract"):
        build_registered_aggregate(manifest_path)

    manifest_path, _, paths = _write_matrix(tmp_path / "grid")
    _rewrite_json(
        paths[("tmaze_10", "gru", 22)],
        lambda value: value["training_history"][0].__setitem__(
            "environment_steps",
            float(TRAIN_STEPS["tmaze_10"] // 2 + 1),
        ),
    )
    with pytest.raises(RegisteredAggregationError, match="unequal training step grids"):
        build_registered_aggregate(manifest_path)


def test_valid_null_prefix_uses_shared_complete_case_suffix(tmp_path: Path) -> None:
    manifest_path, _, paths = _write_matrix(tmp_path)
    for path in paths.values():
        _rewrite_json(
            path,
            lambda value: value["training_history"][0].__setitem__("mean_recent_return", None),
        )
    _refresh_integrity(tmp_path, paths)

    result = build_registered_aggregate(manifest_path)

    contract = result["environment_contracts"][0]["training_curve"]
    assert contract == {
        "full_environment_step_grid": [
            TRAIN_STEPS["tmaze_10"] // 2,
            TRAIN_STEPS["tmaze_10"],
        ],
        "curve_start_step": TRAIN_STEPS["tmaze_10"],
        "excluded_prefix_length": 1,
        "retained_environment_step_grid": [TRAIN_STEPS["tmaze_10"]],
    }
    for group in result["groups"]:
        assert group["training_curve"]["environment_steps"] == [TRAIN_STEPS["tmaze_10"]]
        assert all(
            seed_curve["auc_return_step"] == 0.0
            for seed_curve in group["training_curve"]["raw_seed_returns"]
        )


def test_differing_null_prefix_lengths_use_latest_first_available_step(
    tmp_path: Path,
) -> None:
    manifest_path, _, paths = _write_matrix(tmp_path)
    _rewrite_json(
        paths[("tmaze_10", "arcmind", 11)],
        lambda value: value["training_history"][0].__setitem__("mean_recent_return", None),
    )
    _refresh_integrity(tmp_path, paths)

    result = build_registered_aggregate(manifest_path)

    contract = result["environment_contracts"][0]["training_curve"]
    assert contract["curve_start_step"] == TRAIN_STEPS["tmaze_10"]
    assert contract["excluded_prefix_length"] == 1


def test_all_null_curve_fails_without_shared_finite_suffix(tmp_path: Path) -> None:
    manifest_path, _, paths = _write_matrix(tmp_path)

    def all_null(value: dict[str, Any]) -> None:
        for point in value["training_history"]:
            point["mean_recent_return"] = None

    _rewrite_json(paths[("tmaze_10", "gru", 22)], all_null)
    _refresh_integrity(tmp_path, paths)
    with pytest.raises(RegisteredAggregationError, match="no shared finite"):
        build_registered_aggregate(manifest_path)


def test_interior_null_fails_after_return_availability(tmp_path: Path) -> None:
    manifest_path, _, paths = _write_matrix(tmp_path)
    _rewrite_json(
        paths[("tmaze_10", "gru", 22)],
        lambda value: value["training_history"][1].__setitem__("mean_recent_return", None),
    )
    with pytest.raises(RegisteredAggregationError, match="missing after returns became available"):
        build_registered_aggregate(manifest_path)


def test_missing_curve_value_is_not_treated_as_a_null_prefix(tmp_path: Path) -> None:
    manifest_path, _, paths = _write_matrix(tmp_path)
    _rewrite_json(
        paths[("tmaze_10", "gru", 22)],
        lambda value: value["training_history"][0].pop("mean_recent_return"),
    )
    with pytest.raises(RegisteredAggregationError, match="missing required fields"):
        build_registered_aggregate(manifest_path)


def test_environments_may_have_different_complete_grids_and_budgets(
    tmp_path: Path,
) -> None:
    environments = ["tmaze_10", "rocksample_11_11"]
    manifest_path, _, _ = _write_matrix(
        tmp_path,
        environments=environments,
        evaluation_episodes_per_env=2,
    )

    result = build_registered_aggregate(manifest_path)

    contracts = {item["environment"]: item for item in result["environment_contracts"]}
    assert contracts["tmaze_10"]["training_curve"]["full_environment_step_grid"] == [
        TRAIN_STEPS["tmaze_10"] // 2,
        TRAIN_STEPS["tmaze_10"],
    ]
    assert contracts["rocksample_11_11"]["training_curve"]["full_environment_step_grid"] == [
        2_500_000,
        5_000_000,
    ]
    assert contracts["tmaze_10"]["evaluation"]["episodes_per_environment"] == 2
    assert contracts["rocksample_11_11"]["evaluation"]["episodes_per_environment"] == 2
    rocksample_groups = [
        group for group in result["groups"] if group["environment"] == "rocksample_11_11"
    ]
    assert all(
        group["training_curve"]["environment_steps"] == [2_500_000, 5_000_000]
        for group in rocksample_groups
    )


def test_step_grid_mismatch_within_second_environment_fails(tmp_path: Path) -> None:
    environments = ["tmaze_10", "rocksample_11_11"]
    manifest_path, _, paths = _write_matrix(tmp_path, environments=environments)
    _rewrite_json(
        paths[("rocksample_11_11", "gru", 22)],
        lambda value: value["training_history"][0].__setitem__(
            "environment_steps",
            float(TRAIN_STEPS["rocksample_11_11"] // 2 + 1),
        ),
    )
    with pytest.raises(RegisteredAggregationError, match="within environment"):
        build_registered_aggregate(manifest_path)


def test_evaluation_shape_count_and_curve_order_fail(tmp_path: Path) -> None:
    manifest_path, _, paths = _write_matrix(tmp_path / "shape")
    _rewrite_json(
        paths[("tmaze_10", "gru", 22)],
        lambda value: value["evaluation"]["returns_by_environment"][0].pop(),
    )
    with pytest.raises(RegisteredAggregationError, match="must have 2 returns"):
        build_registered_aggregate(manifest_path)

    manifest_path, _, paths = _write_matrix(tmp_path / "count")
    _rewrite_json(
        paths[("tmaze_10", "gru", 22)],
        lambda value: value["evaluation"].__setitem__("episodes", 99),
    )
    with pytest.raises(RegisteredAggregationError, match="episode counts are inconsistent"):
        build_registered_aggregate(manifest_path)

    manifest_path, _, paths = _write_matrix(tmp_path / "order")
    _rewrite_json(
        paths[("tmaze_10", "gru", 22)],
        lambda value: value["training_history"][1].__setitem__(
            "environment_steps",
            float(TRAIN_STEPS["tmaze_10"] // 2),
        ),
    )
    with pytest.raises(RegisteredAggregationError, match="strictly increasing"):
        build_registered_aggregate(manifest_path)


def test_artifact_paths_are_relative_normalized_posix_and_existing(tmp_path: Path) -> None:
    manifest_path, _, _ = _write_matrix(tmp_path / "backslash")
    _rewrite_manifest(
        manifest_path,
        lambda value: value["cells"][0].__setitem__("artifact_path", "cells\\artifact.json"),
    )
    with pytest.raises(RegisteredAggregationError, match="POSIX"):
        build_registered_aggregate(manifest_path)

    manifest_path, _, _ = _write_matrix(tmp_path / "escape")
    _rewrite_manifest(
        manifest_path,
        lambda value: value["cells"][0].__setitem__("artifact_path", "../artifact.json"),
    )
    with pytest.raises(RegisteredAggregationError, match="normalized relative"):
        build_registered_aggregate(manifest_path)

    manifest_path, _, paths = _write_matrix(tmp_path / "missing-artifact")
    paths[("tmaze_10", "gru", 22)].unlink()
    with pytest.raises(RegisteredAggregationError, match="does not exist"):
        build_registered_aggregate(manifest_path)


def test_duplicate_json_keys_and_nonfinite_json_constants_fail(tmp_path: Path) -> None:
    manifest_path, _, paths = _write_matrix(tmp_path / "duplicate-key")
    paths[("tmaze_10", "gru", 22)].write_text(
        '{"status":"registered_final_complete","status":"duplicate"}',
        encoding="utf-8",
    )
    with pytest.raises(RegisteredAggregationError, match="duplicate JSON key"):
        build_registered_aggregate(manifest_path)

    manifest_path, _, paths = _write_matrix(tmp_path / "nan")
    text = paths[("tmaze_10", "gru", 22)].read_text(encoding="utf-8")
    text = text.replace('"mean_return":1.0', '"mean_return":NaN', 1)
    paths[("tmaze_10", "gru", 22)].write_text(text, encoding="utf-8")
    with pytest.raises(RegisteredAggregationError, match="non-finite JSON constant"):
        build_registered_aggregate(manifest_path)
