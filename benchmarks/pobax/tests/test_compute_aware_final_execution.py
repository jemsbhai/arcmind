"""Execution-boundary tests for schema-v6 compute-aware final matrices."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from benchmarks.pobax import run_matrix
from benchmarks.pobax.implementation_provenance import (
    IMPLEMENTATION_SOURCE_ALGORITHM,
)
from benchmarks.pobax.registered_artifacts import (
    ExistingArtifactMismatchError,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
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
    normalize_panel_selection_binding,
    validate_panel_selection_against_aggregate,
)
from benchmarks.pobax.run_matrix import (
    _cell_namespace,
    _command_for_cell,
    _load_matching_artifact,
    _load_registration,
    _matrix_cell_identities,
    execute_matrix,
)
from benchmarks.pobax.run_pilot import ARTIFACT_SCHEMA_BY_REGISTRATION


def _learner(learning_rate: float = 0.00025) -> dict[str, int | float | bool]:
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


def _implementation_source() -> dict[str, Any]:
    unsigned = {
        "schema_version": 1,
        "algorithm": IMPLEMENTATION_SOURCE_ALGORITHM,
        "files": [{"path": "arcmind/__init__.py", "sha256": "a" * 64}],
    }
    return {**unsigned, "sha256": canonical_json_sha256(unsigned)}


def _runtime_provenance() -> dict[str, Any]:
    return {
        "git": {
            "commit": "b" * 40,
            "dirty": False,
            "diff_sha256": "c" * 64,
        },
        "dependency_lock_sha256": "d" * 64,
        "pobax_commit": "e" * 40,
        "navix_commit": "f" * 40,
        "runtime_contract": {"contract": "test"},
        "implementation_source": _implementation_source(),
    }


def _candidate_ids() -> list[str]:
    return [
        f"{family}.{learner_id}"
        for family in COMPUTE_AWARE_TUNED_FAMILIES
        for learner_id, _ in COMPUTE_AWARE_LEARNER_GRID
    ]


def _aggregate() -> dict[str, Any]:
    source_sha256 = _implementation_source()["sha256"]
    winner = _learner()
    groups = [
        {
            "environment": environment,
            "candidate_id": f"{family}.{learner_id}",
            "model_family": family,
            "implementation_model": family,
            "learner_id": learner_id,
            "learner": _learner(learning_rate),
            "implementation_source_sha256": source_sha256,
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
            "winner_learner": winner,
            "task_scores": task_scores,
            "ranking": [
                {
                    "rank": rank,
                    "candidate_id": f"{family}.{learner_id}",
                    "learner_id": learner_id,
                    "learner": _learner(learning_rate),
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
    provenance = _runtime_provenance()
    return {
        "schema_version": 2,
        "status": "development_tuning_selection_aggregate_not_for_paper",
        "evidence_tier": "development_tuning",
        "matrix_kind": "hyperparameter_selection",
        "not_for_paper": True,
        "registration_sha256": "1" * 64,
        "matrix_manifest_sha256": "2" * 64,
        "completion_index_sha256": "3" * 64,
        "checksum_manifest_sha256": "4" * 64,
        "provenance": provenance,
        "models": _candidate_ids(),
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
                "learner": _learner(learning_rate),
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


def _selection_binding(
    aggregate: dict[str, Any],
    *,
    raw_matrix_path: str = "raw-tuning",
    aggregate_path: str = "aggregate.json",
    aggregate_sha256: str = "0" * 64,
) -> dict[str, Any]:
    source_sha256 = aggregate["provenance"]["implementation_source"]["sha256"]
    return {
        "raw_matrix_path": raw_matrix_path,
        "aggregate_path": aggregate_path,
        "aggregate_sha256": aggregate_sha256,
        "source_registration_sha256": aggregate["registration_sha256"],
        "source_manifest_sha256": aggregate["matrix_manifest_sha256"],
        "source_completion_index_sha256": aggregate["completion_index_sha256"],
        "source_checksum_manifest_sha256": aggregate["checksum_manifest_sha256"],
        "source_implementation_sha256": source_sha256,
        "selections": [
            {
                "model_family": family,
                "implementation_model": family,
                "candidate_id": f"{family}.lr_mid",
                "learner_id": "lr_mid",
                "learner": _learner(),
                "implementation_source_sha256": source_sha256,
            }
            for family in COMPUTE_AWARE_TUNED_FAMILIES
        ],
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


def _registration(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 6,
        "status": "frozen",
        "evidence_tier": "registered_final",
        "matrix_kind": "primary_comparison",
        "models": list(COMPUTE_AWARE_FINAL_MODELS),
        "learner_bindings": _learner_bindings(),
        "task_model_incidence": [
            {"environment": environment, "models": list(models)}
            for environment, models in COMPUTE_AWARE_TASK_MODEL_INCIDENCE
        ],
        "tuning_selection": binding,
        "environments": [
            {"id": environment, "total_steps": total_steps}
            for environment, total_steps in COMPUTE_AWARE_FINAL_PANEL
        ],
        "seeds": list(COMPUTE_AWARE_FINAL_SEEDS),
        "comparison_profile": "arcmind_shared_comparison",
        "evaluation_episodes_per_env": 4,
        "require_gpu": True,
        "quick": False,
    }


def _write_contract(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    aggregate = _aggregate()
    raw_root = root / "raw-tuning"
    raw_root.mkdir(parents=True)
    (raw_root / "completion_index.json").write_bytes(b"completion\n")
    (raw_root / "checksums.sha256").write_bytes(b"checksums\n")
    aggregate["completion_index_sha256"] = sha256_file(raw_root / "completion_index.json")
    aggregate["checksum_manifest_sha256"] = sha256_file(raw_root / "checksums.sha256")
    aggregate_path = root / "aggregate.json"
    atomic_write_json(aggregate_path, aggregate)
    binding = _selection_binding(
        aggregate,
        aggregate_sha256=sha256_file(aggregate_path),
    )
    registration = _registration(binding)
    registration_path = root / "registration.json"
    atomic_write_json(registration_path, registration)
    monkeypatch.setattr(run_matrix, "_REPOSITORY_ROOT", root)
    monkeypatch.setattr(
        run_matrix,
        "build_development_aggregate",
        lambda path: deepcopy(aggregate),
    )
    return registration_path, registration, aggregate


def test_panel_selection_binding_rejects_a_structurally_valid_invented_winner() -> None:
    aggregate = _aggregate()
    binding = _selection_binding(aggregate)
    binding["selections"][0].update(
        candidate_id="memoryless_mlp.lr_low",
        learner_id="lr_low",
        learner=_learner(0.0001),
    )
    normalized = normalize_panel_selection_binding(binding)

    with pytest.raises(ValueError, match="does not match the tuning aggregate winner"):
        validate_panel_selection_against_aggregate(
            normalized,
            aggregate,
            final_seeds=COMPUTE_AWARE_FINAL_SEEDS,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["candidate_selection"][0].update(winner_learner_id="lr_low"),
            "does not match the tuning aggregate winner",
        ),
        (
            lambda value: value.update(models=value["models"][:-1]),
            "wrong candidate inventory",
        ),
        (
            lambda value: value["selection_eligibility"].update(
                selection_scope="candidate_within_task"
            ),
            "invalid eligibility contract",
        ),
        (
            lambda value: value["groups"].pop(),
            "wrong ordered candidate group inventory",
        ),
    ],
)
def test_panel_selection_binding_fails_closed_on_aggregate_drift(
    mutation,
    message: str,
) -> None:
    aggregate = _aggregate()
    normalized = normalize_panel_selection_binding(_selection_binding(aggregate))
    mutation(aggregate)

    with pytest.raises(ValueError, match=message):
        validate_panel_selection_against_aggregate(
            normalized,
            aggregate,
            final_seeds=COMPUTE_AWARE_FINAL_SEEDS,
        )


def test_schema_v6_registration_loads_exact_sparse_490_cell_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration_path, _, _ = _write_contract(tmp_path, monkeypatch)

    registration = _load_registration(registration_path)
    identities = _matrix_cell_identities(registration)
    counts = {
        environment: sum(identity[0]["id"] == environment for identity in identities)
        for environment, _ in COMPUTE_AWARE_FINAL_PANEL
    }

    assert len(identities) == 490
    assert counts == {
        "tmaze_10": 150,
        "rocksample_11_11": 180,
        "battleship_10": 80,
        "Navix-DMLab-Maze-01-v0": 80,
    }
    identity_pairs = {(environment["id"], model) for environment, model, _ in identities}
    assert ("battleship_10", "ffm") not in identity_pairs
    assert ("rocksample_11_11", "memory_trace_official") not in identity_pairs
    assert ("tmaze_10", "memory_trace_official") in identity_pairs
    assert ("rocksample_11_11", "arcmind_no_gate") in identity_pairs


def test_schema_v5_inventory_remains_the_complete_234_cell_tuning_product() -> None:
    registration = {
        "schema_version": 5,
        "tuned_families": [
            {"family_id": family, "implementation_model": family}
            for family in COMPUTE_AWARE_TUNED_FAMILIES
        ],
        "learner_grid": [
            {
                "learner_id": learner_id,
                "learner": _learner(learning_rate),
            }
            for learner_id, learning_rate in COMPUTE_AWARE_LEARNER_GRID
        ],
        "environments": [
            {"id": environment, "total_steps": total_steps}
            for environment, total_steps in COMPUTE_AWARE_TUNING_PANEL
        ],
        "seeds": list(COMPUTE_AWARE_TUNING_SEEDS),
    }

    identities = _matrix_cell_identities(registration)

    assert len(identities) == 234
    assert identities[0][1] == "memoryless_mlp.lr_low"
    assert identities[-1][1] == "transformer_xl.lr_high"


def test_schema_v6_loader_rejects_changed_aggregate_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration_path, _, aggregate = _write_contract(tmp_path, monkeypatch)
    aggregate["status"] = "changed"
    (tmp_path / "aggregate.json").write_text(
        json.dumps(aggregate, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match aggregate bytes"):
        _load_registration(registration_path)


def test_schema_v6_loader_rejects_noncanonical_aggregate_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration_path, _, aggregate = _write_contract(tmp_path, monkeypatch)
    rebuilt = deepcopy(aggregate)
    rebuilt["candidate_selection"][0]["winner_learner_id"] = "lr_low"
    monkeypatch.setattr(
        run_matrix,
        "build_development_aggregate",
        lambda path: deepcopy(rebuilt),
    )

    with pytest.raises(ValueError, match="not the canonical rebuild"):
        _load_registration(registration_path)


def test_schema_v6_namespace_freezes_direct_and_inherited_learner_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration_path, _, _ = _write_contract(tmp_path, monkeypatch)
    registration = _load_registration(registration_path)
    rocksample = registration["environments"][1]

    direct = _cell_namespace(
        registration,
        environment=rocksample,
        model="arcmind",
        seed=10_000,
        output=tmp_path / "direct.json",
        manifest_sha256="a" * 64,
        cell_id="b" * 64,
        describe_only=False,
    )
    inherited = _cell_namespace(
        registration,
        environment=rocksample,
        model="arcmind_no_gate",
        seed=10_000,
        output=tmp_path / "inherited.json",
        manifest_sha256="a" * 64,
        cell_id="c" * 64,
        describe_only=False,
    )
    command = _command_for_cell(inherited)

    assert direct.model == "arcmind"
    assert direct.candidate_id == "arcmind.lr_mid"
    assert direct.learner_binding_mode == "selected"
    assert direct.learner_source_model_family == "arcmind"
    assert inherited.model == "arcmind_no_gate"
    assert inherited.candidate_id == "arcmind.lr_mid"
    assert inherited.model_family == "arcmind"
    assert inherited.learner_id == "lr_mid"
    assert inherited.learner_binding_mode == "inherited"
    assert inherited.learner_source_model_family == "arcmind"
    assert command[command.index("--learner-binding-mode") + 1] == "inherited"
    assert command[command.index("--learner-source-model-family") + 1] == "arcmind"


def test_schema_v6_execute_freezes_sparse_manifest_and_registration_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration_path, _, aggregate = _write_contract(tmp_path, monkeypatch)
    provenance = _runtime_provenance()

    def describe(args) -> dict[str, Any]:
        configuration = {
            "dependency_lock_sha256": provenance["dependency_lock_sha256"],
            "pobax_commit": provenance["pobax_commit"],
            "navix_commit": provenance["navix_commit"],
            "runtime_contract": provenance["runtime_contract"],
            "implementation_source": provenance["implementation_source"],
        }
        return {
            "configuration_sha256": canonical_json_sha256(configuration),
            "configuration": configuration,
            "runtime": {"git": provenance["git"]},
        }

    def load_existing(path: Path, **kwargs) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        path.with_suffix(".log").write_text("complete\n", encoding="utf-8")
        return {}

    monkeypatch.setattr(run_matrix, "run", describe)
    monkeypatch.setattr(run_matrix, "_load_matching_artifact", load_existing)
    monkeypatch.setattr(
        run_matrix,
        "build_development_aggregate",
        lambda path: deepcopy(aggregate),
    )
    output_root = tmp_path / "final"

    completion = execute_matrix(registration_path, output_root)
    manifest = json.loads((output_root / "frozen_manifest.json").read_text(encoding="utf-8"))

    assert completion["planned_cells"] == 490
    assert completion["completed_cells"] == 490
    assert manifest["models"] == list(COMPUTE_AWARE_FINAL_MODELS)
    assert manifest["learner_bindings"] == _learner_bindings()
    assert manifest["task_model_incidence"] == [
        {"environment": environment, "models": list(models)}
        for environment, models in COMPUTE_AWARE_TASK_MODEL_INCIDENCE
    ]
    assert manifest["tuning_selection"]["aggregate_sha256"] == sha256_file(
        tmp_path / "aggregate.json"
    )
    assert manifest["registration_sha256"] == sha256_file(output_root / "registration.json")
    inherited = next(
        cell
        for cell in manifest["cells"]
        if cell["environment"] == "rocksample_11_11"
        and cell["model"] == "arcmind_no_gate"
        and cell["seed"] == 10_000
    )
    assert inherited["candidate_id"] == "arcmind.lr_mid"
    assert inherited["learner_binding_mode"] == "inherited"
    assert inherited["learner_source_model_family"] == "arcmind"
    assert inherited["tuning_aggregate_sha256"] == manifest["tuning_selection"]["aggregate_sha256"]


def _resume_fixture() -> dict[str, Any]:
    implementation_source = _implementation_source()
    configuration = {
        "model": "memoryless_mlp",
        "candidate_id": "memoryless_mlp.lr_mid",
        "model_family": "memoryless_mlp",
        "learner_id": "lr_mid",
        "implementation_model": "memoryless_mlp",
        "learner_binding_mode": "selected",
        "learner_source_model_family": "memoryless_mlp",
        "tuning_aggregate_sha256": "1" * 64,
        "tuning_completion_index_sha256": "2" * 64,
        "tuning_checksum_manifest_sha256": "3" * 64,
        "tuning_implementation_source_sha256": implementation_source["sha256"],
        "implementation_source": implementation_source,
        "evaluation_max_episode_steps": 10,
    }
    provenance = _runtime_provenance()
    artifact = {
        "schema_version": 10,
        "status": "registered_final_complete",
        "environment": "tmaze_10",
        "model": "memoryless_mlp",
        "seed": 10_000,
        "configuration_sha256": canonical_json_sha256(configuration),
        "configuration": configuration,
        "matrix_manifest_sha256": "4" * 64,
        "cell_id": "5" * 64,
        "provenance": provenance,
        "candidate_id": configuration["candidate_id"],
        "model_family": configuration["model_family"],
        "learner_id": configuration["learner_id"],
        "implementation_model": configuration["implementation_model"],
        "learner_binding_mode": configuration["learner_binding_mode"],
        "learner_source_model_family": configuration["learner_source_model_family"],
        "tuning_aggregate_sha256": configuration["tuning_aggregate_sha256"],
        "tuning_completion_index_sha256": configuration["tuning_completion_index_sha256"],
        "tuning_checksum_manifest_sha256": configuration["tuning_checksum_manifest_sha256"],
        "tuning_implementation_source_sha256": configuration["tuning_implementation_source_sha256"],
        "implementation_source_sha256": implementation_source["sha256"],
    }
    return artifact


@pytest.mark.parametrize(
    "field",
    [
        "candidate_id",
        "learner_id",
        "learner_binding_mode",
        "learner_source_model_family",
        "tuning_aggregate_sha256",
        "tuning_completion_index_sha256",
        "tuning_checksum_manifest_sha256",
        "tuning_implementation_source_sha256",
        "implementation_source_sha256",
    ],
)
def test_schema_v6_resume_fails_closed_on_bound_identity_drift(
    tmp_path: Path,
    field: str,
) -> None:
    path = tmp_path / "cell.json"
    artifact = _resume_fixture()
    artifact[field] = "drift"
    atomic_write_json(path, artifact)

    with pytest.raises(ExistingArtifactMismatchError, match="identity"):
        _load_matching_artifact(
            path,
            expected_status="registered_final_complete",
            environment="tmaze_10",
            model="memoryless_mlp",
            seed=10_000,
            configuration_sha256=artifact["configuration_sha256"],
            manifest_sha256="4" * 64,
            cell_id="5" * 64,
            provenance=_runtime_provenance(),
            registration_schema_version=6,
        )


def test_schema_v6_reserves_artifact_schema_10() -> None:
    assert ARTIFACT_SCHEMA_BY_REGISTRATION[6] == 10
