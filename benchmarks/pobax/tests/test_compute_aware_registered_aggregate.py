"""Focused schema-6 registered aggregation tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import benchmarks.pobax.aggregate_registered as aggregate_module
from benchmarks.pobax.aggregate_registered import (
    RegisteredAggregationError,
    _schema6_arcmind_vs_common,
    _validate_artifact,
    _validate_manifest,
    _validate_manifest_panel_selection,
    _validate_registered_completion_and_checksums,
    build_registered_aggregate,
)
from benchmarks.pobax.implementation_provenance import (
    IMPLEMENTATION_SOURCE_ALGORITHM,
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
    COMPUTE_AWARE_FINAL_MODELS,
    COMPUTE_AWARE_FINAL_PANEL,
    COMPUTE_AWARE_FINAL_SEEDS,
    COMPUTE_AWARE_INHERITED_LEARNER_SOURCES,
    COMPUTE_AWARE_LEARNER_GRID,
    COMPUTE_AWARE_TASK_MODEL_INCIDENCE,
    COMPUTE_AWARE_TUNED_FAMILIES,
    COMPUTE_AWARE_TUNING_PANEL,
    COMPUTE_AWARE_TUNING_SEEDS,
    normalize_learner_bindings,
    normalize_panel_selection_binding,
)
from benchmarks.pobax.upper_reference_registry import (
    expected_environment_reference,
    expected_environment_source,
)


def _learner() -> dict[str, int | float | bool]:
    return {
        "num_envs": 8,
        "rollout_steps": 125,
        "update_epochs": 4,
        "num_minibatches": 4,
        "learning_rate": 0.00025,
        "gae_lambda": 0.95,
        "entropy_coefficient": 0.01,
        "anneal_learning_rate": False,
    }


_IMPLEMENTATION_SOURCE_UNSIGNED = {
    "schema_version": 1,
    "algorithm": IMPLEMENTATION_SOURCE_ALGORITHM,
    "files": [{"path": "benchmarks/pobax/run_pilot.py", "sha256": "a" * 64}],
}
IMPLEMENTATION_SOURCE = {
    **_IMPLEMENTATION_SOURCE_UNSIGNED,
    "sha256": canonical_json_sha256(_IMPLEMENTATION_SOURCE_UNSIGNED),
}
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
        "jax_backend": "cpu",
        "jax_enable_x64": False,
        "devices": [{"platform": "cpu", "device_kind": "Test CPU"}],
    },
    "implementation_source": IMPLEMENTATION_SOURCE,
}
OPTIMIZER_METRICS = {
    "loss": 0.5,
    "actor_loss": 0.1,
    "value_loss": 0.8,
    "entropy": 0.2,
    "approximate_kl": 0.01,
}


def _selection_binding() -> dict[str, Any]:
    source_hash = IMPLEMENTATION_SOURCE["sha256"]
    return {
        "raw_matrix_path": "tuning/raw",
        "aggregate_path": "tuning/aggregate.json",
        "aggregate_sha256": "0" * 64,
        "source_registration_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "source_completion_index_sha256": "3" * 64,
        "source_checksum_manifest_sha256": "4" * 64,
        "source_implementation_sha256": source_hash,
        "selections": [
            {
                "model_family": family,
                "implementation_model": family,
                "candidate_id": f"{family}.lr_mid",
                "learner_id": "lr_mid",
                "learner": _learner(),
                "implementation_source_sha256": source_hash,
            }
            for family in COMPUTE_AWARE_TUNED_FAMILIES
        ],
    }


def _tuning_aggregate(binding: dict[str, Any]) -> dict[str, Any]:
    source_hash = IMPLEMENTATION_SOURCE["sha256"]
    groups = [
        {
            "environment": environment,
            "candidate_id": f"{family}.{learner_id}",
            "model_family": family,
            "implementation_model": family,
            "learner_id": learner_id,
            "learner": _learner()
            | {
                "learning_rate": learning_rate,
            },
            "implementation_source_sha256": source_hash,
        }
        for environment, _ in COMPUTE_AWARE_TUNING_PANEL
        for family in COMPUTE_AWARE_TUNED_FAMILIES
        for learner_id, learning_rate in COMPUTE_AWARE_LEARNER_GRID
    ]
    ranking_order = (
        ("lr_mid", 0.00025),
        ("lr_low", 0.0001),
        ("lr_high", 0.0005),
    )
    task_scores = [
        {
            "environment": environment,
            "selection_score": 1.0,
            "task_rank": 1,
            "task_range_regret": 0.0,
        }
        for environment, _ in COMPUTE_AWARE_TUNING_PANEL
    ]
    candidate_selection = [
        {
            "model_family": family,
            "implementation_model": family,
            "metric": "mean_task_rank_then_mean_task_range_regret",
            "direction": "lower_is_better",
            "tie_breaker": "ascending_learner_id",
            "winner_candidate_id": f"{family}.lr_mid",
            "winner_learner_id": "lr_mid",
            "winner_learner": _learner(),
            "task_scores": task_scores,
            "ranking": [
                {
                    "rank": rank,
                    "candidate_id": f"{family}.{learner_id}",
                    "learner_id": learner_id,
                    "learner": _learner()
                    | {
                        "learning_rate": learning_rate,
                    },
                    "mean_task_rank": float(rank),
                    "mean_task_range_regret": float(rank - 1),
                    "task_scores": task_scores,
                }
                for rank, (learner_id, learning_rate) in enumerate(
                    ranking_order,
                    start=1,
                )
            ],
        }
        for family in COMPUTE_AWARE_TUNED_FAMILIES
    ]
    return {
        "schema_version": 2,
        "status": "development_tuning_selection_aggregate_not_for_paper",
        "evidence_tier": "development_tuning",
        "matrix_kind": "hyperparameter_selection",
        "not_for_paper": True,
        "registration_sha256": binding["source_registration_sha256"],
        "matrix_manifest_sha256": binding["source_manifest_sha256"],
        "completion_index_sha256": binding["source_completion_index_sha256"],
        "checksum_manifest_sha256": binding["source_checksum_manifest_sha256"],
        "provenance": deepcopy(PROVENANCE),
        "models": [
            f"{family}.{learner_id}"
            for family in COMPUTE_AWARE_TUNED_FAMILIES
            for learner_id, _ in COMPUTE_AWARE_LEARNER_GRID
        ],
        "tuned_families": [
            {
                "family_id": family,
                "implementation_model": family,
                "candidate_ids": [
                    f"{family}.{learner_id}" for learner_id, _ in COMPUTE_AWARE_LEARNER_GRID
                ],
            }
            for family in COMPUTE_AWARE_TUNED_FAMILIES
        ],
        "learner_grid": [
            {
                "learner_id": learner_id,
                "learner": _learner()
                | {
                    "learning_rate": learning_rate,
                },
            }
            for learner_id, learning_rate in COMPUTE_AWARE_LEARNER_GRID
        ],
        "environments": [environment for environment, _ in COMPUTE_AWARE_TUNING_PANEL],
        "environment_budgets": dict(COMPUTE_AWARE_TUNING_PANEL),
        "seeds": list(COMPUTE_AWARE_TUNING_SEEDS),
        "integrity_indexes": {
            "completion_index_present_and_validated": True,
            "checksums_present_and_validated": True,
        },
        "frozen_semantic_contract": {
            "environment_source_in_every_configuration": True,
            "parameter_contract_in_every_configuration": True,
            "artifact_parameter_contract_validated": True,
        },
        "selection_eligibility": {
            "eligible_for_hyperparameter_selection": True,
            "eligible_for_architecture_selection": False,
            "eligible_for_checkpoint_selection": False,
            "eligible_for_registered_final_evidence": False,
            "eligible_for_paper_performance_claims": False,
            "selection_scope": "learner_within_model_family_across_frozen_task_panel",
            "selection_metric": "mean_task_rank_then_mean_task_range_regret",
        },
        "groups": groups,
        "candidate_selection": candidate_selection,
    }


def _learner_bindings() -> list[dict[str, str]]:
    return [
        {
            "model": model,
            "mode": (
                "inherited" if model in COMPUTE_AWARE_INHERITED_LEARNER_SOURCES else "selected"
            ),
            "source_model_family": COMPUTE_AWARE_INHERITED_LEARNER_SOURCES.get(
                model,
                model,
            ),
        }
        for model in COMPUTE_AWARE_FINAL_MODELS
    ]


def _incidence() -> list[dict[str, Any]]:
    return [
        {"environment": environment, "models": list(models)}
        for environment, models in COMPUTE_AWARE_TASK_MODEL_INCIDENCE
    ]


def _manifest(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    selection = _selection_binding()
    normalized_selection = normalize_panel_selection_binding(selection)
    normalized_bindings = normalize_learner_bindings(
        _learner_bindings(),
        models=COMPUTE_AWARE_FINAL_MODELS,
    )
    contracts = aggregate_module._schema6_cell_contracts(
        models=COMPUTE_AWARE_FINAL_MODELS,
        learner_bindings=normalized_bindings,
        tuning_selection=normalized_selection,
    )
    cells = []
    cell_index = 0
    for environment, models in COMPUTE_AWARE_TASK_MODEL_INCIDENCE:
        for model in models:
            for seed in COMPUTE_AWARE_FINAL_SEEDS:
                configuration_sha256 = canonical_json_sha256(
                    {
                        "environment": environment,
                        "model": model,
                        "seed": seed,
                    }
                )
                contract = {
                    name: value for name, value in contracts[model].items() if name != "learner"
                }
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
                        "artifact_path": f"cells/{cell_index:04d}.json",
                        **contract,
                    }
                )
                cell_index += 1
    manifest = {
        "schema_version": 6,
        "status": "frozen",
        "matrix_kind": "primary_comparison",
        "models": list(COMPUTE_AWARE_FINAL_MODELS),
        "environments": [environment for environment, _ in COMPUTE_AWARE_FINAL_PANEL],
        "seeds": list(COMPUTE_AWARE_FINAL_SEEDS),
        "provenance": deepcopy(PROVENANCE),
        "registration_sha256": "9" * 64,
        "tuning_selection": selection,
        "learner_bindings": _learner_bindings(),
        "task_model_incidence": _incidence(),
        "cells": cells,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    path = tmp_path / "frozen_manifest.json"
    atomic_write_json(path, manifest)
    return path, manifest


def _rewrite_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest = deepcopy(manifest)
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda bindings: bindings.pop(),
        lambda bindings: bindings.append(deepcopy(bindings[-1])),
        lambda bindings: bindings.__setitem__(
            slice(0, 2),
            [bindings[1], bindings[0]],
        ),
        lambda bindings: bindings[-1].update(source_model_family="gru"),
    ],
)
def test_schema6_manifest_rejects_missing_extra_reordered_or_mutated_bindings(
    tmp_path: Path,
    monkeypatch,
    mutation,
) -> None:
    path, manifest = _manifest(tmp_path)
    mutation(manifest["learner_bindings"])
    _rewrite_manifest(path, manifest)
    monkeypatch.setattr(
        aggregate_module,
        "_validate_manifest_panel_selection",
        lambda value, **kwargs: normalize_panel_selection_binding(value),
    )

    with pytest.raises(RegisteredAggregationError):
        _validate_manifest(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cells: cells.pop(), "exactly 490"),
        (lambda cells: cells.append(deepcopy(cells[-1])), "exactly 490"),
        (
            lambda cells: cells.__setitem__(slice(0, 2), [cells[1], cells[0]]),
            "exact sparse incidence order",
        ),
    ],
)
def test_schema6_manifest_rejects_sparse_inventory_drift(
    tmp_path: Path,
    monkeypatch,
    mutation,
    message: str,
) -> None:
    path, manifest = _manifest(tmp_path)
    mutation(manifest["cells"])
    _rewrite_manifest(path, manifest)
    monkeypatch.setattr(
        aggregate_module,
        "_validate_manifest_panel_selection",
        lambda value, **kwargs: normalize_panel_selection_binding(value),
    )

    with pytest.raises(RegisteredAggregationError, match=message):
        _validate_manifest(path)


def _schema6_configuration(
    contract: dict[str, Any],
    *,
    environment: str = "tmaze_10",
    model: str = "memoryless_mlp",
    seed: int = COMPUTE_AWARE_FINAL_SEEDS[0],
) -> dict[str, Any]:
    total_steps = dict(COMPUTE_AWARE_FINAL_PANEL)[environment]
    return {
        "schema_version": 6,
        "evidence_tier": "registered_final",
        "environment": environment,
        "model": model,
        "seed": seed,
        **{
            name: value
            for name, value in contract.items()
            if name not in {"learner", "implementation_source_sha256"}
        },
        "implementation_source": deepcopy(IMPLEMENTATION_SOURCE),
        "environment_source": expected_environment_source(environment),
        "environment_reference": expected_environment_reference(environment),
        "parameter_count": 1_000,
        "effective_parameter_count": 1_000,
        "arcmind_target_parameter_count": 1_000,
        "parameter_ratio": 1.0,
        "ppo": {
            **contract["learner"],
            "total_steps": total_steps,
            "step_budget_mode": "exact",
        },
        "evaluation_episodes_per_environment": 1,
        "evaluation_max_episode_steps": 10,
        "dependency_lock_sha256": PROVENANCE["dependency_lock_sha256"],
        "pobax_commit": PROVENANCE["pobax_commit"],
        "navix_commit": PROVENANCE["navix_commit"],
        "runtime_contract": deepcopy(PROVENANCE["runtime_contract"]),
        "comparison_profile": "arcmind_shared_comparison",
        "requested_environment_steps": total_steps,
        "realized_environment_steps": total_steps,
    }


def _schema6_artifact(
    configuration: dict[str, Any],
    contract: dict[str, Any],
    *,
    manifest_sha256: str,
    configuration_sha256: str,
) -> dict[str, Any]:
    environment = configuration["environment"]
    model = configuration["model"]
    seed = configuration["seed"]
    horizon = configuration["evaluation_max_episode_steps"]
    returns = [[float(index)] for index in range(8)]
    flat = [row[0] for row in returns]
    total_steps = configuration["ppo"]["total_steps"]
    return {
        "schema_version": 10,
        "status": "registered_final_complete",
        "matrix_manifest_sha256": manifest_sha256,
        "cell_id": registered_cell_id(
            environment,
            model,
            seed,
            configuration_sha256,
        ),
        "configuration_sha256": configuration_sha256,
        "configuration": configuration,
        "environment": environment,
        "model": model,
        "seed": seed,
        **{name: value for name, value in contract.items() if name != "learner"},
        "environment_source": deepcopy(configuration["environment_source"]),
        "environment_reference": deepcopy(configuration["environment_reference"]),
        "parameter_count": 1_000,
        "effective_parameter_count": 1_000,
        "arcmind_target_parameter_count": 1_000,
        "parameter_ratio": 1.0,
        "provenance": deepcopy(PROVENANCE),
        "actual_environment_steps": total_steps,
        "ppo": deepcopy(configuration["ppo"]),
        "evaluation_episodes_per_environment": 1,
        "evaluation_max_episode_steps": horizon,
        "actual_evaluation_steps_per_environment": horizon,
        "actual_evaluation_transitions": horizon * 8,
        "comparison_profile": "arcmind_shared_comparison",
        "requested_environment_steps": total_steps,
        "realized_environment_steps": total_steps,
        "evaluation": {
            "mean_return": sum(flat) / len(flat),
            "median_return": 3.5,
            "episodes": 8,
            "episodes_per_environment": 1,
            "num_environments": 8,
            "scan_steps_per_environment": horizon,
            "returns_by_environment": returns,
        },
        "training": deepcopy(OPTIMIZER_METRICS),
        "training_history": [
            {
                **OPTIMIZER_METRICS,
                "environment_steps": total_steps // 2,
                "mean_recent_return": 1.0,
            },
            {
                **OPTIMIZER_METRICS,
                "environment_steps": total_steps,
                "mean_recent_return": 2.0,
            },
        ],
    }


def test_schema6_artifact_schema10_binds_ppo_source_and_horizon(
    tmp_path: Path,
) -> None:
    selection = normalize_panel_selection_binding(_selection_binding())
    bindings = normalize_learner_bindings(
        _learner_bindings(),
        models=COMPUTE_AWARE_FINAL_MODELS,
    )
    contract = aggregate_module._schema6_cell_contracts(
        models=COMPUTE_AWARE_FINAL_MODELS,
        learner_bindings=bindings,
        tuning_selection=selection,
    )["memoryless_mlp"]
    configuration = _schema6_configuration(contract)
    configuration_sha256 = canonical_json_sha256(configuration)
    manifest_sha256 = "8" * 64
    artifact = _schema6_artifact(
        configuration,
        contract,
        manifest_sha256=manifest_sha256,
        configuration_sha256=configuration_sha256,
    )
    path = tmp_path / "artifact.json"
    atomic_write_json(path, artifact)
    expected = {
        "cell_id": artifact["cell_id"],
        "configuration_sha256": configuration_sha256,
        **contract,
    }

    record = _validate_artifact(
        path,
        identity=("tmaze_10", "memoryless_mlp", COMPUTE_AWARE_FINAL_SEEDS[0]),
        expected=expected,
        manifest_sha256=manifest_sha256,
        manifest_schema_version=6,
        provenance=PROVENANCE,
    )

    assert record["registration_contract"]["ppo"]["learning_rate"] == 0.00025
    artifact["schema_version"] = 8
    path.write_bytes(canonical_json_bytes(artifact) + b"\n")
    with pytest.raises(RegisteredAggregationError, match="schema 10"):
        _validate_artifact(
            path,
            identity=(
                "tmaze_10",
                "memoryless_mlp",
                COMPUTE_AWARE_FINAL_SEEDS[0],
            ),
            expected=expected,
            manifest_sha256=manifest_sha256,
            manifest_schema_version=6,
            provenance=PROVENANCE,
        )


def test_schema6_completion_and_checksum_inventory_close_over_490_cells(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path, raw_manifest = _manifest(tmp_path)
    monkeypatch.setattr(
        aggregate_module,
        "_validate_manifest_panel_selection",
        lambda value, **kwargs: normalize_panel_selection_binding(value),
    )
    manifest, cells = _validate_manifest(path)
    atomic_write_json(tmp_path / "registration.json", {"schema_version": 6})
    manifest_cells = {
        (cell["environment"], cell["model"], cell["seed"]): cell for cell in raw_manifest["cells"]
    }
    completed = []
    for identity, cell in cells.items():
        atomic_write_json(cell["artifact_path"], {"identity": list(identity)})
        log_path = cell["artifact_path"].with_suffix(".log")
        atomic_write_bytes(log_path, b"schema-6 test log\n")
        completed.append(
            {
                **deepcopy(manifest_cells[identity]),
                "artifact_sha256": sha256_file(cell["artifact_path"]),
                "log_path": log_path.relative_to(tmp_path).as_posix(),
                "log_sha256": sha256_file(log_path),
            }
        )
    atomic_write_json(
        tmp_path / "completion_index.json",
        {
            "schema_version": 1,
            "status": "complete",
            "manifest_sha256": manifest["manifest_sha256"],
            "planned_cells": 490,
            "completed_cells": 490,
            "cells": completed,
        },
    )
    write_checksum_manifest(tmp_path)

    integrity = _validate_registered_completion_and_checksums(
        path,
        manifest,
        cells,
    )

    assert set(integrity) == {
        "completion_index_sha256",
        "checksum_manifest_sha256",
    }


def test_schema6_panel_selection_requires_canonical_schema2_aggregate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw = tmp_path / "tuning" / "raw"
    raw.mkdir(parents=True)
    atomic_write_bytes(raw / "completion_index.json", b"completion\n")
    atomic_write_bytes(raw / "checksums.sha256", b"checksums\n")
    binding = _selection_binding()
    binding["source_completion_index_sha256"] = sha256_file(raw / "completion_index.json")
    binding["source_checksum_manifest_sha256"] = sha256_file(raw / "checksums.sha256")
    aggregate = _tuning_aggregate(binding)
    aggregate_path = tmp_path / "tuning" / "aggregate.json"
    atomic_write_json(aggregate_path, aggregate)
    binding["aggregate_sha256"] = sha256_file(aggregate_path)
    monkeypatch.setattr(aggregate_module, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        aggregate_module,
        "build_development_aggregate",
        lambda path: deepcopy(aggregate),
    )

    normalized = _validate_manifest_panel_selection(
        binding,
        final_seeds=COMPUTE_AWARE_FINAL_SEEDS,
        final_provenance=PROVENANCE,
    )

    assert len(normalized["selections"]) == 13


def test_schema6_stratified_bootstrap_is_deterministic_and_scores_ties_half() -> None:
    records = {}
    common = COMPUTE_AWARE_FINAL_MODELS[:8]
    for environment, _ in COMPUTE_AWARE_FINAL_PANEL:
        for seed_index, seed in enumerate(COMPUTE_AWARE_FINAL_SEEDS):
            arcmind_return = float(seed_index)
            for model_index, model in enumerate(common):
                value = (
                    arcmind_return
                    if model in {"arcmind", "frame_stack_mlp"}
                    else arcmind_return - 1.0
                    if model_index % 2 == 0
                    else arcmind_return + 1.0
                )
                records[(environment, model, seed)] = {"evaluation": {"mean_return": value}}
    manifest = {
        "environments": tuple(environment for environment, _ in COMPUTE_AWARE_FINAL_PANEL),
        "seeds": COMPUTE_AWARE_FINAL_SEEDS,
    }

    first = _schema6_arcmind_vs_common(manifest, records)
    second = _schema6_arcmind_vs_common(manifest, records)

    assert first == second
    assert len(first) == 7
    tie = next(item for item in first if item["comparison_model"] == "frame_stack_mlp")
    assert tie["stratified_probability_of_improvement"]["estimate"] == 0.5
    assert [task["weight"] for task in tie["tasks"]] == [0.25] * 4
    assert all(len(task["raw_paired_seed_values"]) == 10 for task in tie["tasks"])
    assert "raw_paired_seed_values" not in tie


def test_legacy_registered_aggregate_remains_schema1(tmp_path: Path) -> None:
    from benchmarks.pobax.tests.test_aggregate_registered import _write_matrix

    manifest_path, _, _ = _write_matrix(tmp_path)

    result = build_registered_aggregate(manifest_path)

    assert result["schema_version"] == 1
    assert "common_models" not in result
    assert "arcmind_vs_common_comparisons" not in result
    assert len(result["paired_differences_against_arcmind"]) == 1
