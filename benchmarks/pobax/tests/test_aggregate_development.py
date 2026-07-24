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
from benchmarks.pobax.implementation_provenance import IMPLEMENTATION_SOURCE_ALGORITHM
from benchmarks.pobax.model_registry import (
    FIXED_OFFICIAL_PARAMETER_CONTRACT,
    PARAMETER_MATCHED_CONTRACT,
    PRIMARY_COMPARISON_ROLE,
    SUPPLEMENTAL_COMPARISON_ROLE,
    fixed_official_parameter_count,
    policy_contract_metadata_for_model,
    reference_implementation_for_model,
    requires_explicit_policy_contract,
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


def _policy_core_for_model(model: str) -> dict[str, Any]:
    if model == "agalite_source_compat":
        return {
            "input_dim": 7,
            "observation_dim": 2,
            "action_dim": 2,
            "hidden_size": 128,
            "head_dim": 64,
            "feedforward_size": 128,
            "num_heads": 4,
            "eta": 4,
            "approximation_channels": 2,
            "num_layers": 4,
            "actor_hidden_size": 128,
            "critic_hidden_size": 128,
            "gate_bias": 2.0,
            "attention_epsilon": 1e-5,
            "layer_norm_epsilon": 1e-6,
        }
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
    if model == "memory_trace_official":
        return {
            "input_dim": 7,
            "observation_dim": 2,
            "action_dim": 2,
            "hidden_size": 64,
            "decays": [0.0, 0.985],
        }
    return {
        "input_dim": 7,
        "action_dim": 2,
        "hidden_size": 4,
        "decays": [0.0, 0.985],
    }


def _configuration(
    environment: str,
    model: str,
    seed: int,
    *,
    tier: str,
    total_steps: int | None = None,
    schema_version: int = 1,
    comparison_profile: str | None = None,
    learner: dict[str, Any] | None = None,
    model_family: str | None = None,
    implementation_model: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    budget = ENVIRONMENTS[environment] if total_steps is None else total_steps
    selected_provenance = provenance or PROVENANCE
    selected_learner = learner or {
        "num_envs": 2,
        "rollout_steps": 2,
        "update_epochs": 1,
        "learning_rate": 0.001,
    }
    configuration = {
        "schema_version": schema_version,
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
            "total_steps": budget,
            **selected_learner,
        },
        "evaluation_episodes_per_environment": 2,
        "evaluation_max_episode_steps": 5,
        "dependency_lock_sha256": selected_provenance["dependency_lock_sha256"],
        "pobax_commit": selected_provenance["pobax_commit"],
        "navix_commit": selected_provenance["navix_commit"],
        "runtime_contract": deepcopy(selected_provenance["runtime_contract"]),
    }
    if schema_version in {2, 3}:
        configuration["ppo"]["step_budget_mode"] = (
            "floor" if comparison_profile == "pobax_author_semantics" else "exact"
        )
        requested_steps = budget
        realized_steps = (
            requested_steps
            // (int(selected_learner["num_envs"]) * int(selected_learner["rollout_steps"]))
            * int(selected_learner["num_envs"])
            * int(selected_learner["rollout_steps"])
        )
        configuration.update(
            comparison_profile=comparison_profile,
            requested_environment_steps=requested_steps,
            realized_environment_steps=realized_steps,
        )
    if schema_version == 3:
        configuration.update(
            candidate_id=model,
            model_family=model_family,
            implementation_model=implementation_model,
            implementation_source=deepcopy(IMPLEMENTATION_SOURCE),
        )
    actual_model = implementation_model or model
    reference = reference_implementation_for_model(actual_model)
    if reference is not None:
        configuration["reference_implementation"] = reference
    if requires_explicit_policy_contract(actual_model):
        policy_core = _policy_core_for_model(actual_model)
        configuration.update(
            policy_contract_metadata_for_model(actual_model),
            policy_core=policy_core,
        )
        fixed_count = fixed_official_parameter_count(actual_model, policy_core)
        if fixed_count is not None:
            target_count = 28_000
            configuration.update(
                parameter_count=fixed_count,
                effective_parameter_count=fixed_count,
                arcmind_target_parameter_count=target_count,
                parameter_ratio=fixed_count / target_count,
            )
    return configuration


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
    schema_version: int = 1,
    comparison_profile: str | None = None,
    candidate_families: list[dict[str, Any]] | None = None,
    provenance: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[tuple[str, str, int], Path]]:
    selected_provenance = provenance or PROVENANCE
    candidate_index = {
        candidate["candidate_id"]: {
            **candidate,
            "model_family": family["family_id"],
            "implementation_model": family["implementation_model"],
        }
        for family in (candidate_families or [])
        for candidate in family["candidates"]
    }
    selected_models = list(candidate_index) if schema_version == 3 else models or MODELS
    selected_environments = environments or ENVIRONMENTS
    selected_seeds = seeds or SEEDS
    registration = {
        "schema_version": schema_version,
        "status": "frozen",
        "evidence_tier": tier,
        "matrix_kind": (
            "hyperparameter_selection" if schema_version == 3 else "primary_comparison"
        ),
        "environments": [
            {"id": environment, "total_steps": budget}
            for environment, budget in selected_environments.items()
        ],
        "seeds": selected_seeds,
        "evaluation_episodes_per_env": 2,
        "require_gpu": True,
        "quick": False,
    }
    if schema_version in {1, 2}:
        registration["models"] = selected_models
        registration["learner"] = {
            "num_envs": 2,
            "rollout_steps": 2,
            "update_epochs": 1,
            "learning_rate": 0.001,
        }
    if schema_version == 2:
        registration["comparison_profile"] = comparison_profile
        registration["learner"].update(
            num_minibatches=1,
            gae_lambda=0.95,
            entropy_coefficient=0.01,
            anneal_learning_rate=True,
        )
    if schema_version == 3:
        registration["comparison_profile"] = comparison_profile
        registration["candidate_families"] = candidate_families
    atomic_write_json(root / "registration.json", registration)
    cells = []
    paths: dict[tuple[str, str, int], Path] = {}
    for environment in selected_environments:
        for model in selected_models:
            for seed in selected_seeds:
                identity = (environment, model, seed)
                configuration = _configuration(
                    environment,
                    model,
                    seed,
                    tier=tier,
                    total_steps=selected_environments[environment],
                    schema_version=schema_version,
                    comparison_profile=comparison_profile,
                    learner=(
                        candidate_index[model]["learner"]
                        if schema_version == 3
                        else registration["learner"]
                    ),
                    model_family=(
                        candidate_index[model]["model_family"] if schema_version == 3 else None
                    ),
                    implementation_model=(
                        candidate_index[model]["implementation_model"]
                        if schema_version == 3
                        else None
                    ),
                    provenance=selected_provenance,
                )
                configuration_sha256 = canonical_json_sha256(configuration)
                relative = f"cells/{environment}-{model}-{seed}.json"
                cell = {
                    "cell_id": registered_cell_id(environment, model, seed, configuration_sha256),
                    "environment": environment,
                    "model": model,
                    "seed": seed,
                    "configuration_sha256": configuration_sha256,
                    "artifact_path": relative,
                }
                if schema_version == 3:
                    cell.update(
                        model_family=candidate_index[model]["model_family"],
                        implementation_model=candidate_index[model]["implementation_model"],
                        implementation_source_sha256=IMPLEMENTATION_SOURCE["sha256"],
                    )
                cells.append(cell)
                paths[identity] = root / relative
    manifest_provenance = deepcopy(selected_provenance)
    if schema_version == 3:
        manifest_provenance["implementation_source"] = deepcopy(IMPLEMENTATION_SOURCE)
    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "status": "frozen",
        "matrix_kind": (
            "hyperparameter_selection" if schema_version == 3 else "primary_comparison"
        ),
        "models": selected_models,
        "environments": list(selected_environments),
        "seeds": selected_seeds,
        "provenance": manifest_provenance,
        "cells": cells,
    }
    if schema_version == 3:
        manifest["candidate_families"] = candidate_families
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_write_json(root / "frozen_manifest.json", manifest)
    for identity, path in paths.items():
        environment, model, seed = identity
        configuration = _configuration(
            environment,
            model,
            seed,
            tier=tier,
            total_steps=selected_environments[environment],
            schema_version=schema_version,
            comparison_profile=comparison_profile,
            learner=(
                candidate_index[model]["learner"]
                if schema_version == 3
                else registration["learner"]
            ),
            model_family=(candidate_index[model]["model_family"] if schema_version == 3 else None),
            implementation_model=(
                candidate_index[model]["implementation_model"] if schema_version == 3 else None
            ),
            provenance=selected_provenance,
        )
        configuration_sha256 = canonical_json_sha256(configuration)
        value = float(
            seed
            + (
                candidate_index[model]["learner"]["learning_rate"] * 100_000
                + (100 if candidate_index[model]["implementation_model"] == "gru" else 0)
                if schema_version == 3
                else 10
                if model == "arcmind"
                else 0
            )
        )
        artifact_provenance = deepcopy(selected_provenance)
        if schema_version == 3:
            artifact_provenance["implementation_source"] = deepcopy(IMPLEMENTATION_SOURCE)
        artifact = {
            "schema_version": 6 if schema_version == 3 else 5 if schema_version == 2 else 4,
            "status": (
                "development_tuning_not_for_paper"
                if tier == "development_tuning"
                else f"development_{tier}_not_for_paper"
            ),
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
            "provenance": artifact_provenance,
            "actual_environment_steps": selected_environments[environment],
            "ppo": deepcopy(configuration["ppo"]),
            "evaluation_episodes_per_environment": 2,
            "evaluation_max_episode_steps": 5,
            "actual_evaluation_steps_per_environment": 10,
            "actual_evaluation_transitions": 20,
            "evaluation": _evaluation(value),
            "training": {**OPTIMIZER_METRICS, "mean_recent_return": value},
            "training_history": [
                {
                    **OPTIMIZER_METRICS,
                    "environment_steps": selected_environments[environment] / 2,
                    "mean_recent_return": None,
                },
                {
                    **OPTIMIZER_METRICS,
                    "environment_steps": float(selected_environments[environment]),
                    "mean_recent_return": value,
                },
            ],
        }
        actual_model = (
            candidate_index[model]["implementation_model"]
            if schema_version == 3
            else model
        )
        reference = reference_implementation_for_model(actual_model)
        if reference is not None:
            artifact["reference_implementation"] = reference
        if requires_explicit_policy_contract(actual_model):
            artifact["policy_core"] = deepcopy(configuration["policy_core"])
            artifact.update(policy_contract_metadata_for_model(actual_model))
        if schema_version == 3:
            artifact.update(
                candidate_id=model,
                model_family=candidate_index[model]["model_family"],
                implementation_model=candidate_index[model]["implementation_model"],
                implementation_source_sha256=IMPLEMENTATION_SOURCE["sha256"],
            )
        if schema_version in {2, 3}:
            artifact.update(
                comparison_profile=comparison_profile,
                requested_environment_steps=selected_environments[environment],
                realized_environment_steps=configuration["realized_environment_steps"],
            )
        atomic_write_json(path, artifact)
    return manifest, paths


def _rewrite(path: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    path.write_text(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


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


def _write_tuning_matrix(
    root: Path,
    *,
    provenance: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[tuple[str, str, int], Path]]:
    base_learner = {
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
            "family_id": family,
            "implementation_model": implementation_model,
            "candidates": [
                {
                    "candidate_id": f"{family}.lr_{label}",
                    "learner": {
                        **base_learner,
                        "learning_rate": learning_rate,
                    },
                }
                for label, learning_rate in (("low", 0.001), ("high", 0.002))
            ],
        }
        for family, implementation_model in (
            ("ordered_memory", "arcmind"),
            ("recurrent", "gru"),
        )
    ]
    manifest, paths = _write_matrix(
        root,
        tier="development_tuning",
        environments={"tmaze_10": 1_000_000},
        seeds=[7, 19, 23, 31, 43],
        schema_version=3,
        comparison_profile="arcmind_shared_comparison",
        candidate_families=candidate_families,
        provenance=provenance,
    )
    for path in paths.values():
        artifact = json.loads(path.read_text(encoding="utf-8"))
        final_return = artifact["evaluation"]["mean_return"]
        _rewrite(
            path,
            lambda value, final_return=final_return: value.update(
                training_history=[
                    {
                        **OPTIMIZER_METRICS,
                        "environment_steps": 250_000.0,
                        "mean_recent_return": None,
                    },
                    {
                        **OPTIMIZER_METRICS,
                        "environment_steps": 500_000.0,
                        "mean_recent_return": final_return - 2.0,
                    },
                    {
                        **OPTIMIZER_METRICS,
                        "environment_steps": 1_000_000.0,
                        "mean_recent_return": final_return,
                    },
                ]
            ),
        )
    _write_integrity_indexes(root, manifest, paths)
    return manifest, paths


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
        "parameter_contract_in_every_configuration": True,
        "artifact_parameter_contract_validated": True,
        "reference_implementation_validated": True,
    }
    assert len(first["groups"]) == 4
    assert len(first["paired_differences_against_arcmind"]) == 2
    assert "selection_eligibility" not in first
    assert "environment_contracts" not in first
    assert "candidate_selection" not in first
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


def test_mamba_development_artifacts_fail_closed_on_source_drift(
    tmp_path: Path,
) -> None:
    matrix_root = tmp_path / "matrix"
    _, paths = _write_matrix(
        matrix_root,
        models=["arcmind", "mamba1"],
        environments={"short": 8_192},
        seeds=[7, 19],
    )

    result = build_development_aggregate(matrix_root)

    assert {group["model"] for group in result["groups"]} == {
        "arcmind",
        "mamba1",
    }
    assert result["frozen_semantic_contract"]["reference_implementation_validated"] is True
    assert len(result["paired_differences_against_arcmind"]) == 1

    def drift_source(artifact: dict[str, Any]) -> None:
        artifact["reference_implementation"]["audited_commit"] = "0" * 40

    _rewrite(paths[("short", "mamba1", 7)], drift_source)
    with pytest.raises(DevelopmentAggregationError, match="registered source contract"):
        build_development_aggregate(matrix_root)


def test_memory_trace_contracts_and_supplemental_results_are_separated(
    tmp_path: Path,
) -> None:
    matrix_root = tmp_path / "memory-traces"
    _, paths = _write_matrix(
        matrix_root,
        models=["arcmind", "memory_trace_official", "memory_trace_shared"],
        environments={"short": 8_192},
        seeds=[7, 19],
    )

    result = build_development_aggregate(matrix_root)
    groups = {item["model"]: item for item in result["groups"]}

    assert groups["memory_trace_official"]["parameter_contract"] == (
        FIXED_OFFICIAL_PARAMETER_CONTRACT
    )
    assert groups["memory_trace_official"]["comparison_role"] == (
        SUPPLEMENTAL_COMPARISON_ROLE
    )
    assert groups["memory_trace_shared"]["parameter_contract"] == (
        PARAMETER_MATCHED_CONTRACT
    )
    assert groups["memory_trace_shared"]["comparison_role"] == PRIMARY_COMPARISON_ROLE
    assert [
        item["model"] for item in result["paired_differences_against_arcmind"]
    ] == ["memory_trace_shared"]
    assert [
        item["model"]
        for item in result["supplemental_paired_differences_against_arcmind"]
    ] == ["memory_trace_official"]
    assert (
        result["frozen_semantic_contract"][
            "parameter_contract_in_every_configuration"
        ]
        is True
    )
    assert (
        result["frozen_semantic_contract"]["parameter_match_in_every_configuration"]
        is False
    )
    assert (
        result["frozen_semantic_contract"]["artifact_parameter_match_validated"]
        is False
    )

    official_path = paths[("short", "memory_trace_official", 7)]
    _rewrite(
        official_path,
        lambda value: value.update(memory_trace_decays=[0.0, 0.9]),
    )
    with pytest.raises(DevelopmentAggregationError, match="policy contract"):
        build_development_aggregate(matrix_root)


def test_agalite_contracts_and_supplemental_results_are_separated(
    tmp_path: Path,
) -> None:
    matrix_root = tmp_path / "agalite"
    _, paths = _write_matrix(
        matrix_root,
        models=["arcmind", "agalite_source_compat", "agalite_shared"],
        environments={"short": 8_192},
        seeds=[7, 19],
    )

    result = build_development_aggregate(matrix_root)
    groups = {item["model"]: item for item in result["groups"]}

    assert groups["agalite_source_compat"]["parameter_contract"] == (
        FIXED_OFFICIAL_PARAMETER_CONTRACT
    )
    assert groups["agalite_source_compat"]["comparison_role"] == (
        SUPPLEMENTAL_COMPARISON_ROLE
    )
    assert groups["agalite_shared"]["parameter_contract"] == (
        PARAMETER_MATCHED_CONTRACT
    )
    assert groups["agalite_shared"]["comparison_role"] == PRIMARY_COMPARISON_ROLE
    assert [
        item["model"] for item in result["paired_differences_against_arcmind"]
    ] == ["agalite_shared"]
    assert [
        item["model"]
        for item in result["supplemental_paired_differences_against_arcmind"]
    ] == ["agalite_source_compat"]

    source_path = paths[("short", "agalite_source_compat", 7)]
    _rewrite(
        source_path,
        lambda value: value["policy_core"].update(layer_norm_epsilon=1e-5),
    )
    with pytest.raises(DevelopmentAggregationError, match="policy_core"):
        build_development_aggregate(matrix_root)


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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["training_history"][0].update(loss=None),
            r"training_history\[0\]\.loss must be a finite number",
        ),
        (
            lambda value: value["training"].update(approximate_kl=None),
            r"training\.approximate_kl must be a finite number",
        ),
    ],
)
def test_nonfinite_optimizer_metrics_fail_closed(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    _, paths = _write_matrix(tmp_path)
    _rewrite(paths[("short", "gru", 7)], mutation)

    with pytest.raises(DevelopmentAggregationError, match=message):
        build_development_aggregate(tmp_path)


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


@pytest.mark.parametrize(
    ("schema_version", "comparison_profile"),
    [(1, None), (2, "arcmind_shared_comparison")],
)
def test_legacy_completion_indexes_accept_artifact_only_rows(
    tmp_path: Path,
    schema_version: int,
    comparison_profile: str | None,
) -> None:
    manifest, paths = _write_matrix(
        tmp_path,
        schema_version=schema_version,
        comparison_profile=comparison_profile,
    )
    _write_integrity_indexes(tmp_path, manifest, paths)
    completion_path = tmp_path / "completion_index.json"

    def remove_log_identity(value: dict[str, Any]) -> None:
        for cell in value["cells"]:
            cell.pop("log_path")
            cell.pop("log_sha256")

    _rewrite(completion_path, remove_log_identity)
    for log_path in (tmp_path / "cells").glob("*.log"):
        log_path.unlink()
    (tmp_path / "checksums.sha256").unlink()
    write_checksum_manifest(tmp_path)

    result = build_development_aggregate(tmp_path)

    assert result["integrity_indexes"] == {
        "completion_index_present_and_validated": True,
        "checksums_present_and_validated": True,
    }


def test_schema_v2_development_matrix_validates_explicit_step_accounting(
    tmp_path: Path,
) -> None:
    matrix_root = tmp_path / "matrix-v2"
    _write_matrix(
        matrix_root,
        schema_version=2,
        comparison_profile="arcmind_shared_comparison",
    )

    result = build_development_aggregate(matrix_root)

    assert result["status"] == "development_pilot_aggregate_not_for_paper"

    artifact = next((matrix_root / "cells").glob("*.json"))
    _rewrite(
        artifact,
        lambda value: value.update(realized_environment_steps=4),
    )
    with pytest.raises(DevelopmentAggregationError, match="step accounting"):
        build_development_aggregate(matrix_root)


def test_development_tuning_ranks_equal_cardinality_candidates_by_shared_auc(
    tmp_path: Path,
) -> None:
    manifest, paths = _write_tuning_matrix(tmp_path)

    result = build_development_aggregate(tmp_path)

    assert result["status"] == "development_tuning_selection_aggregate_not_for_paper"
    assert result["evidence_tier"] == "development_tuning"
    assert result["selection_eligibility"] == {
        "eligible_for_hyperparameter_selection": True,
        "eligible_for_architecture_selection": False,
        "eligible_for_checkpoint_selection": False,
        "eligible_for_registered_final_evidence": False,
        "eligible_for_paper_performance_claims": False,
        "selection_scope": "candidate_within_model_family_and_environment",
        "selection_metric": "mean_seed_auc_mean_return",
    }
    assert result["integrity_indexes"] == {
        "completion_index_present_and_validated": True,
        "checksums_present_and_validated": True,
    }
    assert result["candidate_families"] == [
        {
            "family_id": "ordered_memory",
            "implementation_model": "arcmind",
            "candidate_ids": [
                "ordered_memory.lr_low",
                "ordered_memory.lr_high",
            ],
        },
        {
            "family_id": "recurrent",
            "implementation_model": "gru",
            "candidate_ids": [
                "recurrent.lr_low",
                "recurrent.lr_high",
            ],
        },
    ]
    contract = result["environment_contracts"][0]
    assert contract["model_family_count"] == 2
    assert contract["candidate_count_per_family"] == 2
    assert contract["total_candidate_count"] == 4
    assert contract["seed_count_per_candidate"] == 5
    assert contract["candidate_seed_cardinality_equal"] is True
    assert contract["candidate_cardinality_equal_across_families"] is True
    assert contract["training_curve"] == {
        "full_environment_step_grid": [250_000, 500_000, 1_000_000],
        "curve_start_step": 500_000,
        "curve_end_step": 1_000_000,
        "excluded_prefix_length": 1,
        "retained_environment_step_grid": [500_000, 1_000_000],
        "integration_width_environment_steps": 500_000,
    }
    assert len(result["candidate_selection"]) == 2
    assert result["paired_differences_against_arcmind"] == []
    selections = {
        selection["model_family"]: selection for selection in result["candidate_selection"]
    }
    assert selections["ordered_memory"]["winner_candidate_id"] == "ordered_memory.lr_high"
    assert selections["recurrent"]["winner_candidate_id"] == "recurrent.lr_high"
    assert [item["candidate_id"] for item in selections["ordered_memory"]["ranking"]] == [
        "ordered_memory.lr_high",
        "ordered_memory.lr_low",
    ]
    arcmind = next(
        group for group in result["groups"] if group["candidate_id"] == "ordered_memory.lr_high"
    )
    assert arcmind["model_family"] == "ordered_memory"
    assert arcmind["implementation_model"] == "arcmind"
    seed_row = arcmind["training_curve"]["raw_seed_returns"][0]
    assert seed_row["mean_recent_return"] == [205.0, 207.0]
    assert seed_row["auc_return_step"] == 103_000_000.0
    assert seed_row["auc_mean_return"] == 206.0
    manifest_cell = next(
        cell
        for cell in manifest["cells"]
        if cell["model"] == "ordered_memory.lr_high" and cell["seed"] == 7
    )
    assert manifest_cell["model_family"] == "ordered_memory"
    assert manifest_cell["implementation_model"] == "arcmind"
    artifact = json.loads(
        paths[("tmaze_10", "ordered_memory.lr_high", 7)].read_text(encoding="utf-8")
    )
    assert artifact["candidate_id"] == "ordered_memory.lr_high"
    assert artifact["configuration"]["candidate_id"] == "ordered_memory.lr_high"
    assert artifact["cell_id"] == registered_cell_id(
        "tmaze_10",
        "ordered_memory.lr_high",
        7,
        artifact["configuration_sha256"],
    )


def _downgrade_tuning_registration(value: dict[str, Any]) -> None:
    first_learner = value["candidate_families"][0]["candidates"][0]["learner"]
    value["schema_version"] = 2
    value["matrix_kind"] = "primary_comparison"
    value["models"] = ["arcmind", "gru"]
    value["learner"] = first_learner
    value.pop("candidate_families")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            _downgrade_tuning_registration,
            "schema version 3",
        ),
        (
            lambda value: value.update(seeds=[7, 19, 23]),
            "published tuning-seed count",
        ),
        (
            lambda value: value["environments"][0].update(total_steps=250_000),
            "published task budget",
        ),
        (
            lambda value: value["candidate_families"][0].update(
                candidates=value["candidate_families"][0]["candidates"][:1]
            ),
            "at least two candidates",
        ),
    ],
)
def test_development_tuning_registration_contract_fails_closed(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    _write_tuning_matrix(tmp_path)
    _rewrite(tmp_path / "registration.json", mutation)

    with pytest.raises(DevelopmentAggregationError, match=message):
        build_development_aggregate(tmp_path)


def test_development_tuning_requires_integrity_indexes(tmp_path: Path) -> None:
    _write_tuning_matrix(tmp_path)
    (tmp_path / "completion_index.json").unlink()
    (tmp_path / "checksums.sha256").unlink()

    with pytest.raises(
        DevelopmentAggregationError,
        match="validated completion and checksum indexes",
    ):
        build_development_aggregate(tmp_path)


def test_development_tuning_requires_two_shared_finite_curve_points(
    tmp_path: Path,
) -> None:
    manifest, paths = _write_tuning_matrix(tmp_path)
    path = paths[("tmaze_10", "recurrent.lr_low", 7)]
    _rewrite(
        path,
        lambda value: value["training_history"][1].update(mean_recent_return=None),
    )
    (tmp_path / "completion_index.json").unlink()
    (tmp_path / "checksums.sha256").unlink()
    _write_integrity_indexes(tmp_path, manifest, paths)

    with pytest.raises(
        DevelopmentAggregationError,
        match="at least two shared finite learning-curve points",
    ):
        build_development_aggregate(tmp_path)


def test_development_tuning_completion_index_freezes_candidate_identity(
    tmp_path: Path,
) -> None:
    _write_tuning_matrix(tmp_path)
    completion_path = tmp_path / "completion_index.json"
    _rewrite(
        completion_path,
        lambda value: value["cells"][0].update(model_family="wrong_family"),
    )
    (tmp_path / "checksums.sha256").unlink()
    write_checksum_manifest(tmp_path)

    with pytest.raises(
        DevelopmentAggregationError,
        match="candidate identity drifts",
    ):
        build_development_aggregate(tmp_path)


def _remove_completion_log_fields(value: dict[str, Any]) -> None:
    value["cells"][0].pop("log_path")
    value["cells"][0].pop("log_sha256")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            _remove_completion_log_fields,
            "schema v3 requires log_path and log_sha256",
        ),
        (
            lambda value: value["cells"][0].update(log_path=value["cells"][1]["log_path"]),
            "immutable cell log identity",
        ),
        (
            lambda value: value["cells"][0].update(log_sha256="0" * 64),
            "log_sha256 is incorrect",
        ),
    ],
)
def test_development_tuning_completion_logs_fail_closed(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    _write_tuning_matrix(tmp_path)
    completion_path = tmp_path / "completion_index.json"
    _rewrite(completion_path, mutation)
    (tmp_path / "checksums.sha256").unlink()
    write_checksum_manifest(tmp_path)

    with pytest.raises(DevelopmentAggregationError, match=message):
        build_development_aggregate(tmp_path)


def test_development_tuning_checksum_inventory_must_cover_logs(
    tmp_path: Path,
) -> None:
    _write_tuning_matrix(tmp_path)
    checksum_path = tmp_path / "checksums.sha256"
    retained_lines = [
        line
        for line in checksum_path.read_text(encoding="utf-8").splitlines()
        if ".log" not in line
    ]
    checksum_path.write_text("\n".join(retained_lines) + "\n", encoding="utf-8")

    with pytest.raises(
        DevelopmentAggregationError,
        match="checksum inventory differs",
    ):
        build_development_aggregate(tmp_path)


def test_development_tuning_rejects_checksummed_attempt_inside_raw_root(
    tmp_path: Path,
) -> None:
    _write_tuning_matrix(tmp_path)
    attempt_path = tmp_path / "cells" / f"orphan.attempt-{'a' * 32}.failed.log"
    attempt_path.write_bytes(b"failed attempt evidence\n")
    (tmp_path / "checksums.sha256").unlink()
    write_checksum_manifest(tmp_path)

    with pytest.raises(
        DevelopmentAggregationError,
        match="noncanonical raw inputs",
    ):
        build_development_aggregate(tmp_path)


def test_development_tuning_gpu_requirement_matches_runtime_provenance(
    tmp_path: Path,
) -> None:
    cpu_provenance = deepcopy(PROVENANCE)
    cpu_provenance["runtime_contract"]["jax_backend"] = "cpu"
    cpu_provenance["runtime_contract"]["devices"] = [{"platform": "cpu", "device_kind": "Test CPU"}]
    _write_tuning_matrix(tmp_path, provenance=cpu_provenance)

    with pytest.raises(DevelopmentAggregationError, match="requires GPU"):
        build_development_aggregate(tmp_path)


def test_development_tuning_selection_never_uses_final_evaluation_return(
    tmp_path: Path,
) -> None:
    manifest, paths = _write_tuning_matrix(tmp_path)
    path = paths[("tmaze_10", "ordered_memory.lr_low", 7)]
    _rewrite(
        path,
        lambda value: value.update(evaluation=_evaluation(10_000.0)),
    )
    (tmp_path / "completion_index.json").unlink()
    (tmp_path / "checksums.sha256").unlink()
    _write_integrity_indexes(tmp_path, manifest, paths)

    result = build_development_aggregate(tmp_path)
    selection = next(
        item for item in result["candidate_selection"] if item["model_family"] == "ordered_memory"
    )

    assert selection["winner_candidate_id"] == "ordered_memory.lr_high"


def test_development_tuning_exact_tie_uses_ascending_candidate_id(
    tmp_path: Path,
) -> None:
    manifest, paths = _write_tuning_matrix(tmp_path)
    for seed in (7, 19, 23, 31, 43):
        winning_history = json.loads(
            paths[("tmaze_10", "ordered_memory.lr_high", seed)].read_text(encoding="utf-8")
        )["training_history"]
        _rewrite(
            paths[("tmaze_10", "ordered_memory.lr_low", seed)],
            lambda value, history=deepcopy(winning_history): value.update(training_history=history),
        )
    (tmp_path / "completion_index.json").unlink()
    (tmp_path / "checksums.sha256").unlink()
    _write_integrity_indexes(tmp_path, manifest, paths)

    result = build_development_aggregate(tmp_path)
    selection = next(
        item for item in result["candidate_selection"] if item["model_family"] == "ordered_memory"
    )

    assert [item["selection_score"] for item in selection["ranking"]] == [
        selection["ranking"][0]["selection_score"]
    ] * 2
    assert [item["candidate_id"] for item in selection["ranking"]] == [
        "ordered_memory.lr_high",
        "ordered_memory.lr_low",
    ]
