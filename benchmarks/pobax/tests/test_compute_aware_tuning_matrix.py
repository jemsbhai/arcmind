"""Execution-boundary tests for schema-v5 compute-aware tuning."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.pobax import run_matrix
from benchmarks.pobax.registered_artifacts import (
    ExistingArtifactMismatchError,
    atomic_write_json,
    canonical_json_sha256,
)
from benchmarks.pobax.run_matrix import (
    _cell_namespace,
    _command_for_cell,
    _load_registration,
    execute_matrix,
)
from benchmarks.pobax.run_pilot import ARTIFACT_SCHEMA_BY_REGISTRATION


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


def _registration() -> dict[str, object]:
    return {
        "schema_version": 5,
        "status": "frozen",
        "evidence_tier": "development_tuning",
        "matrix_kind": "hyperparameter_selection",
        "tuned_families": [
            {
                "family_id": "memoryless_mlp",
                "implementation_model": "memoryless_mlp",
            },
            {"family_id": "gru", "implementation_model": "gru"},
        ],
        "learner_grid": [
            {"learner_id": "lr_low", "learner": _learner(0.0001)},
            {"learner_id": "lr_mid", "learner": _learner(0.00025)},
            {"learner_id": "lr_high", "learner": _learner(0.0005)},
        ],
        "environments": [
            {"id": "tmaze_10", "total_steps": 1_000_000},
            {"id": "rocksample_11_11", "total_steps": 1_000_000},
        ],
        "seeds": [4409, 5519, 6637],
        "comparison_profile": "arcmind_shared_comparison",
        "evaluation_episodes_per_env": 4,
        "require_gpu": True,
        "quick": False,
    }


def _write_registration(path: Path) -> dict[str, object]:
    registration = _registration()
    path.write_text(json.dumps(registration), encoding="utf-8")
    return registration


def _provenance() -> dict[str, object]:
    implementation_source = {
        "schema_version": 1,
        "algorithm": "test-source-v1",
        "files": [],
        "sha256": "e" * 64,
    }
    return {
        "git": {"commit": "a" * 40, "dirty": False, "diff_sha256": None},
        "dependency_lock_sha256": "b" * 64,
        "pobax_commit": "c" * 40,
        "navix_commit": "d" * 40,
        "runtime_contract": {"runtime": "test"},
        "implementation_source": implementation_source,
    }


def _configuration(
    *,
    environment: str,
    implementation_model: str,
    candidate_id: str,
    model_family: str,
    learner_id: str,
    seed: int,
) -> dict[str, object]:
    provenance = _provenance()
    return {
        "schema_version": 5,
        "evidence_tier": "development_tuning",
        "environment": environment,
        "model": candidate_id,
        "candidate_id": candidate_id,
        "model_family": model_family,
        "learner_id": learner_id,
        "implementation_model": implementation_model,
        "seed": seed,
        "dependency_lock_sha256": provenance["dependency_lock_sha256"],
        "pobax_commit": provenance["pobax_commit"],
        "navix_commit": provenance["navix_commit"],
        "runtime_contract": provenance["runtime_contract"],
        "implementation_source": provenance["implementation_source"],
    }


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _install_fake_execution(monkeypatch) -> list[list[str]]:
    calls: list[list[str]] = []
    provenance = _provenance()

    def fake_describe(args):
        configuration = _configuration(
            environment=args.environment,
            implementation_model=args.model,
            candidate_id=args.candidate_id,
            model_family=args.model_family,
            learner_id=args.learner_id,
            seed=args.seed,
        )
        return {
            "configuration_sha256": canonical_json_sha256(configuration),
            "configuration": configuration,
            "runtime": {"git": provenance["git"]},
        }

    def fake_subprocess(command, **kwargs):
        del kwargs
        calls.append(list(command))
        configuration = _configuration(
            environment=_argument(command, "--environment"),
            implementation_model=_argument(command, "--model"),
            candidate_id=_argument(command, "--candidate-id"),
            model_family=_argument(command, "--model-family"),
            learner_id=_argument(command, "--learner-id"),
            seed=int(_argument(command, "--seed")),
        )
        artifact = {
            "schema_version": 9,
            "status": "development_tuning_not_for_paper",
            "environment": configuration["environment"],
            "model": configuration["candidate_id"],
            "candidate_id": configuration["candidate_id"],
            "model_family": configuration["model_family"],
            "learner_id": configuration["learner_id"],
            "implementation_model": configuration["implementation_model"],
            "implementation_source_sha256": provenance["implementation_source"][
                "sha256"
            ],
            "seed": configuration["seed"],
            "configuration_sha256": canonical_json_sha256(configuration),
            "configuration": configuration,
            "matrix_manifest_sha256": _argument(
                command,
                "--matrix-manifest-sha256",
            ),
            "cell_id": _argument(command, "--cell-id"),
            "provenance": provenance,
        }
        atomic_write_json(Path(_argument(command, "--output")), artifact)
        return SimpleNamespace(returncode=0, stdout=b"schema-v5 cell complete\n")

    monkeypatch.setattr(run_matrix, "run", fake_describe)
    monkeypatch.setattr(run_matrix.subprocess, "run", fake_subprocess)
    return calls


def test_schema_v5_registration_loads_only_the_exact_validated_panel(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registration.json"
    registration = _write_registration(path)

    assert _load_registration(path) == registration

    registration["environments"][1]["total_steps"] = 5_000_000
    path.write_text(json.dumps(registration), encoding="utf-8")
    with pytest.raises(ValueError, match="exact ordered two-task panel"):
        _load_registration(path)


def test_schema_v5_derives_candidate_namespace_and_command_metadata() -> None:
    registration = _registration()
    args = _cell_namespace(
        registration,
        environment=registration["environments"][1],
        model="gru.lr_high",
        seed=5519,
        output=Path("cell.json"),
        manifest_sha256="a" * 64,
        cell_id="b" * 64,
        describe_only=False,
    )
    command = _command_for_cell(args)

    assert args.model == "gru"
    assert args.candidate_id == "gru.lr_high"
    assert args.model_family == "gru"
    assert args.learner_id == "lr_high"
    assert args.learning_rate == 0.0005
    assert command[command.index("--model") + 1] == "gru"
    assert command[command.index("--candidate-id") + 1] == "gru.lr_high"
    assert command[command.index("--model-family") + 1] == "gru"
    assert command[command.index("--learner-id") + 1] == "lr_high"
    assert command[command.index("--registration-schema-version") + 1] == "5"


def test_schema_v5_executes_complete_matrix_and_freezes_shared_grid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registration_path = tmp_path / "registration.json"
    registration = _write_registration(registration_path)
    output_root = tmp_path / "matrix"
    calls = _install_fake_execution(monkeypatch)

    completion = execute_matrix(registration_path, output_root)

    expected_models = [
        "memoryless_mlp.lr_low",
        "memoryless_mlp.lr_mid",
        "memoryless_mlp.lr_high",
        "gru.lr_low",
        "gru.lr_mid",
        "gru.lr_high",
    ]
    expected_cells = 2 * 2 * 3 * 3
    assert completion["planned_cells"] == expected_cells
    assert completion["completed_cells"] == expected_cells
    assert len(calls) == expected_cells
    manifest = json.loads(
        (output_root / "frozen_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 5
    assert manifest["models"] == expected_models
    assert manifest["tuned_families"] == registration["tuned_families"]
    assert manifest["learner_grid"] == registration["learner_grid"]
    assert len(manifest["cells"]) == expected_cells
    assert {
        (
            cell["environment"],
            cell["model"],
            cell["seed"],
        )
        for cell in manifest["cells"]
    } == {
        (environment["id"], model, seed)
        for environment in registration["environments"]
        for model in expected_models
        for seed in registration["seeds"]
    }
    assert all(
        cell["model"] == f"{cell['model_family']}.{cell['learner_id']}"
        for cell in manifest["cells"]
    )
    assert all(cell["implementation_source_sha256"] == "e" * 64 for cell in manifest["cells"])

    resumed = execute_matrix(registration_path, output_root)
    assert resumed == completion
    assert len(calls) == expected_cells


def test_schema_v5_resume_rejects_learner_identity_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registration_path = tmp_path / "registration.json"
    _write_registration(registration_path)
    output_root = tmp_path / "matrix"
    _install_fake_execution(monkeypatch)
    completion = execute_matrix(registration_path, output_root)
    artifact_path = output_root / completion["cells"][0]["artifact_path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["learner_id"] = "lr_drift"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(
        ExistingArtifactMismatchError,
        match="compute-aware tuning cell identity",
    ):
        execute_matrix(registration_path, output_root)


def test_run_pilot_reserves_artifact_schema_9_for_registration_schema_5() -> None:
    assert [ARTIFACT_SCHEMA_BY_REGISTRATION[schema] for schema in range(1, 6)] == [
        4,
        5,
        6,
        8,
        9,
    ]
