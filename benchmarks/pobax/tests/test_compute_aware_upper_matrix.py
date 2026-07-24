"""Focused schema-7 matrix execution and resume-contract tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import benchmarks.pobax.run_matrix as matrix_module
from benchmarks.pobax.registered_artifacts import (
    ExistingArtifactMismatchError,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_file,
)
from benchmarks.pobax.registration_protocol import (
    COMPUTE_AWARE_FINAL_SEEDS,
    COMPUTE_AWARE_UPPER_REFERENCE_PANEL,
)
from benchmarks.pobax.run_matrix import (
    _cell_namespace,
    _command_for_cell,
    _load_matching_artifact,
    _load_registration,
    _matrix_cell_identities,
    _validate_compute_aware_upper_primary,
    execute_matrix,
)
from benchmarks.pobax.tests.test_compute_aware_upper_registration import (
    _implementation_source,
    _memoryless_binding,
    _primary_aggregate,
    _primary_binding,
)


def _registration() -> dict[str, Any]:
    source = _implementation_source()
    return {
        "schema_version": 7,
        "status": "frozen",
        "evidence_tier": "registered_final",
        "matrix_kind": "upper_reference",
        "models": ["memoryless_mlp"],
        "environments": [
            {"id": environment, "total_steps": total_steps}
            for environment, total_steps in COMPUTE_AWARE_UPPER_REFERENCE_PANEL
        ],
        "seeds": list(COMPUTE_AWARE_FINAL_SEEDS),
        "comparison_profile": "arcmind_shared_comparison",
        "primary_matrix_binding": _primary_binding(source["sha256"]),
        "memoryless_learner_binding": _memoryless_binding(source["sha256"]),
        "evaluation_episodes_per_env": 16,
        "require_gpu": True,
        "quick": False,
    }


def _bound_primary_files(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    registration = _registration()
    primary = registration["primary_matrix_binding"]
    raw = tmp_path / "artifacts" / "primary" / "raw"
    raw.mkdir(parents=True)
    for filename, content in (
        ("registration.json", b"registration\n"),
        ("frozen_manifest.json", b"manifest\n"),
        ("completion_index.json", b"completion\n"),
        ("checksums.sha256", b"checksums\n"),
    ):
        atomic_write_bytes(raw / filename, content)
    primary.update(
        {
            "primary_registration_file_sha256": sha256_file(raw / "registration.json"),
            "primary_manifest_file_sha256": sha256_file(raw / "frozen_manifest.json"),
            "primary_completion_index_file_sha256": sha256_file(raw / "completion_index.json"),
            "primary_checksum_manifest_file_sha256": sha256_file(raw / "checksums.sha256"),
        }
    )
    aggregate = _primary_aggregate(
        primary,
        registration["memoryless_learner_binding"],
        _implementation_source(),
    )
    aggregate_path = tmp_path / "artifacts" / "primary" / "aggregate.json"
    atomic_write_json(aggregate_path, aggregate)
    primary["primary_aggregate_file_sha256"] = sha256_file(aggregate_path)
    return registration, aggregate


def test_schema7_primary_files_require_exact_canonical_rebuild(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registration, aggregate = _bound_primary_files(tmp_path)
    monkeypatch.setattr(matrix_module, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        matrix_module,
        "build_registered_aggregate",
        lambda path: deepcopy(aggregate),
    )

    primary, memoryless, rebuilt = _validate_compute_aware_upper_primary(registration)

    assert primary == registration["primary_matrix_binding"]
    assert memoryless == registration["memoryless_learner_binding"]
    assert rebuilt == aggregate

    aggregate_path = tmp_path / registration["primary_matrix_binding"]["aggregate_path"]
    aggregate_path.write_bytes(aggregate_path.read_bytes() + b"\n")
    registration["primary_matrix_binding"]["primary_aggregate_file_sha256"] = sha256_file(
        aggregate_path
    )
    with pytest.raises(ValueError, match="canonical rebuild"):
        _validate_compute_aware_upper_primary(registration)


def test_schema7_loader_freezes_exact_40_cell_inventory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registration = _registration()
    path = tmp_path / "registration.json"
    atomic_write_json(path, registration)
    monkeypatch.setattr(
        matrix_module,
        "_validate_compute_aware_upper_primary",
        lambda value: (
            value["primary_matrix_binding"],
            value["memoryless_learner_binding"],
            {},
        ),
    )

    loaded = _load_registration(path)
    inventory = _matrix_cell_identities(loaded)

    assert len(inventory) == 40
    assert [(environment["id"], model, seed) for environment, model, seed in inventory] == [
        (environment, "memoryless_mlp", seed)
        for environment, _ in COMPUTE_AWARE_UPPER_REFERENCE_PANEL
        for seed in COMPUTE_AWARE_FINAL_SEEDS
    ]


def test_schema7_cell_namespace_and_command_bind_primary_and_learner() -> None:
    registration = _registration()
    args = _cell_namespace(
        registration,
        environment=registration["environments"][0],
        model="memoryless_mlp",
        seed=10_000,
        output=Path("cell.json"),
        manifest_sha256="b" * 64,
        cell_id="c" * 64,
        describe_only=False,
    )
    command = _command_for_cell(args)

    assert args.candidate_id == "memoryless_mlp.lr_mid"
    assert args.learner_binding_mode == "selected"
    assert args.primary_manifest_internal_sha256 == "4" * 64
    assert command[command.index("--primary-manifest-internal-sha256") + 1] == "4" * 64
    assert (
        command[command.index("--primary-implementation-source-sha256") + 1]
        == registration["primary_matrix_binding"]["primary_implementation_source_sha256"]
    )


def _resume_configuration(registration: dict[str, Any]) -> dict[str, Any]:
    memoryless = registration["memoryless_learner_binding"]
    primary = registration["primary_matrix_binding"]
    return {
        "model": "memoryless_mlp",
        "implementation_model": "memoryless_mlp",
        "candidate_id": memoryless["candidate_id"],
        "model_family": memoryless["model_family"],
        "learner_id": memoryless["learner_id"],
        "learner_binding_mode": memoryless["learner_binding_mode"],
        "learner_source_model_family": memoryless["learner_source_model_family"],
        "tuning_aggregate_sha256": memoryless["tuning_aggregate_sha256"],
        "tuning_completion_index_sha256": memoryless["tuning_completion_index_sha256"],
        "tuning_checksum_manifest_sha256": memoryless["tuning_checksum_manifest_sha256"],
        "tuning_implementation_source_sha256": memoryless["tuning_implementation_source_sha256"],
        "implementation_source": {"sha256": primary["primary_implementation_source_sha256"]},
        **{field: value for field, value in primary.items() if field.endswith("_sha256")},
    }


def test_schema7_resume_rejects_any_primary_binding_drift(tmp_path: Path) -> None:
    registration = _registration()
    configuration = _resume_configuration(registration)
    provenance = {"implementation_source": _implementation_source()}
    artifact = {
        "schema_version": 11,
        "status": "registered_final_complete",
        "environment": "tmaze_10-perfect-memory",
        "model": "memoryless_mlp",
        "seed": 10_000,
        "configuration_sha256": canonical_json_sha256(configuration),
        "configuration": configuration,
        "matrix_manifest_sha256": "b" * 64,
        "cell_id": "c" * 64,
        "provenance": provenance,
        "implementation_source_sha256": registration["primary_matrix_binding"][
            "primary_implementation_source_sha256"
        ],
        **{
            field: value
            for field, value in configuration.items()
            if field
            in {
                "candidate_id",
                "model_family",
                "learner_id",
                "implementation_model",
                "learner_binding_mode",
                "learner_source_model_family",
                "tuning_aggregate_sha256",
                "tuning_completion_index_sha256",
                "tuning_checksum_manifest_sha256",
                "tuning_implementation_source_sha256",
            }
            or field.startswith("primary_")
        },
    }
    path = tmp_path / "cell.json"
    atomic_write_json(path, artifact)

    loaded = _load_matching_artifact(
        path,
        expected_status="registered_final_complete",
        environment="tmaze_10-perfect-memory",
        model="memoryless_mlp",
        seed=10_000,
        configuration_sha256=artifact["configuration_sha256"],
        manifest_sha256="b" * 64,
        cell_id="c" * 64,
        provenance=provenance,
        registration_schema_version=7,
    )
    assert loaded == artifact

    artifact["primary_manifest_internal_sha256"] = "f" * 64
    path.write_bytes(canonical_json_bytes(artifact) + b"\n")
    with pytest.raises(
        ExistingArtifactMismatchError,
        match="upper-reference cell identity",
    ):
        _load_matching_artifact(
            path,
            expected_status="registered_final_complete",
            environment="tmaze_10-perfect-memory",
            model="memoryless_mlp",
            seed=10_000,
            configuration_sha256=artifact["configuration_sha256"],
            manifest_sha256="b" * 64,
            cell_id="c" * 64,
            provenance=provenance,
            registration_schema_version=7,
        )


def test_schema7_execute_writes_bound_manifest_and_40_cells(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registration = _registration()
    source = _implementation_source()
    primary_aggregate = _primary_aggregate(
        registration["primary_matrix_binding"],
        registration["memoryless_learner_binding"],
        source,
    )
    primary_aggregate["provenance"].update(
        {
            "dependency_lock_sha256": "d" * 64,
            "pobax_commit": "1" * 40,
            "navix_commit": "2" * 40,
            "runtime_contract": {"runtime": "test"},
        }
    )
    monkeypatch.setattr(matrix_module, "_load_registration", lambda path: registration)
    monkeypatch.setattr(
        matrix_module,
        "_validate_compute_aware_upper_primary",
        lambda value: (
            value["primary_matrix_binding"],
            value["memoryless_learner_binding"],
            primary_aggregate,
        ),
    )

    def describe(args):
        configuration = {
            "dependency_lock_sha256": "d" * 64,
            "pobax_commit": "1" * 40,
            "navix_commit": "2" * 40,
            "runtime_contract": {"runtime": "test"},
            "implementation_source": source,
            "candidate_id": args.candidate_id,
            "model_family": args.model_family,
            "learner_id": args.learner_id,
            "implementation_model": args.model,
            "learner_binding_mode": args.learner_binding_mode,
            "learner_source_model_family": args.learner_source_model_family,
            "tuning_aggregate_sha256": args.tuning_aggregate_sha256,
            "tuning_completion_index_sha256": (args.tuning_completion_index_sha256),
            "tuning_checksum_manifest_sha256": (args.tuning_checksum_manifest_sha256),
            "tuning_implementation_source_sha256": (args.tuning_implementation_source_sha256),
            **{
                field: getattr(args, field)
                for field in matrix_module._SCHEMA_V7_PRIMARY_HASH_FIELDS
            },
        }
        return {
            "configuration_sha256": canonical_json_sha256(configuration),
            "configuration": configuration,
            "runtime": {"git": {"commit": "3" * 40, "dirty": False}},
        }

    monkeypatch.setattr(matrix_module, "run", describe)

    def fake_subprocess(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        atomic_write_json(output, {"completed": True})
        return SimpleNamespace(returncode=0, stdout=b"completed\n")

    monkeypatch.setattr(matrix_module.subprocess, "run", fake_subprocess)
    monkeypatch.setattr(
        matrix_module,
        "_load_matching_artifact",
        lambda path, **kwargs: {"completed": True} if path.exists() else None,
    )
    output = tmp_path / "upper"
    completion = execute_matrix(tmp_path / "registration.json", output)
    manifest = json.loads((output / "frozen_manifest.json").read_text())

    assert completion["planned_cells"] == 40
    assert manifest["primary_matrix_binding"] == registration["primary_matrix_binding"]
    assert manifest["memoryless_learner_binding"] == registration["memoryless_learner_binding"]
    assert len(manifest["cells"]) == 40
    assert all(
        cell["primary_manifest_internal_sha256"] == "4" * 64
        and cell["candidate_id"] == "memoryless_mlp.lr_mid"
        and cell["learner_binding_mode"] == "selected"
        for cell in manifest["cells"]
    )
