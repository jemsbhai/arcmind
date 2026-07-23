"""Execute a frozen POBAX matrix without overwriting completed cells."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from benchmarks.pobax.registered_artifacts import (
    ExistingArtifactMismatchError,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_sha256,
    registered_cell_id,
    registered_cell_path,
    sha256_file,
    validate_paired_seed_manifests,
    validate_unique_cell_ids,
    write_checksum_manifest,
)
from benchmarks.pobax.registration_protocol import (
    normalize_learner,
    realized_environment_steps,
    registration_fields,
    validate_comparison_profile,
)
from benchmarks.pobax.run_pilot import (
    EVIDENCE_STATUS,
    UPPER_REFERENCE_TARGETS,
    run,
)

_ENVIRONMENT_FIELDS = {"id", "total_steps"}
_QUICK_LEARNER_VALUES = {
    "num_envs": 32,
    "rollout_steps": 32,
    "update_epochs": 2,
}
_MATRIX_KINDS = {"primary_comparison", "upper_reference"}


def _load_registration(path: Path) -> dict[str, Any]:
    registration = json.loads(path.read_text(encoding="utf-8"))
    schema_version = registration.get("schema_version")
    expected_fields = registration_fields(schema_version)
    if set(registration) != expected_fields:
        missing = sorted(expected_fields - set(registration))
        extra = sorted(set(registration) - expected_fields)
        raise ValueError(f"registration has wrong fields: missing={missing}, extra={extra}")
    if registration.get("status") != "frozen":
        raise ValueError("registration status must be 'frozen'")
    comparison_profile = validate_comparison_profile(registration)
    tier = registration.get("evidence_tier")
    if tier not in EVIDENCE_STATUS:
        raise ValueError(f"unsupported evidence_tier: {tier!r}")
    matrix_kind = registration.get("matrix_kind")
    if matrix_kind not in _MATRIX_KINDS:
        raise ValueError(f"unsupported matrix_kind: {matrix_kind!r}")

    models = registration.get("models")
    if (
        not isinstance(models, list)
        or not models
        or any(not isinstance(model, str) or not model for model in models)
    ):
        raise ValueError("models must be a non-empty list of names")
    if len(set(models)) != len(models):
        raise ValueError("models must not contain duplicates")
    if matrix_kind == "primary_comparison" and "arcmind" not in models:
        raise ValueError("primary_comparison matrices must contain arcmind")
    if matrix_kind == "upper_reference" and models != ["memoryless_mlp"]:
        raise ValueError("upper_reference matrices must contain only memoryless_mlp")

    seeds = registration.get("seeds")
    validate_paired_seed_manifests({model: seeds for model in models})

    environments = registration.get("environments")
    if not isinstance(environments, list) or not environments:
        raise ValueError("environments must be a non-empty list")
    environment_ids: list[str] = []
    for environment in environments:
        if not isinstance(environment, dict):
            raise TypeError("each environment entry must be an object")
        if set(environment) != _ENVIRONMENT_FIELDS:
            raise ValueError("environment entries must contain exactly id and total_steps")
        environment_id = environment.get("id")
        total_steps = environment.get("total_steps")
        if not isinstance(environment_id, str) or not environment_id:
            raise ValueError("each environment requires a non-empty id")
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError("each environment requires positive integer total_steps")
        environment_ids.append(environment_id)
    if len(set(environment_ids)) != len(environment_ids):
        raise ValueError("environment ids must not contain duplicates")
    if matrix_kind == "upper_reference":
        unsupported = sorted(set(environment_ids) - set(UPPER_REFERENCE_TARGETS))
        if unsupported:
            raise ValueError(
                "upper_reference matrices contain environments without "
                f"registered adapters: {unsupported}"
            )

    learner = normalize_learner(
        registration.get("learner"),
        schema_version=schema_version,
    )
    for environment in environments:
        realized_environment_steps(
            environment["total_steps"],
            num_envs=int(learner["num_envs"]),
            rollout_steps=int(learner["rollout_steps"]),
            comparison_profile=comparison_profile,
        )

    evaluation_episodes = registration.get("evaluation_episodes_per_env")
    if (
        isinstance(evaluation_episodes, bool)
        or not isinstance(evaluation_episodes, int)
        or evaluation_episodes <= 0
    ):
        raise ValueError("evaluation_episodes_per_env must be a positive integer")
    if not isinstance(registration.get("require_gpu"), bool):
        raise ValueError("require_gpu must be a boolean")
    if not isinstance(registration.get("quick"), bool):
        raise ValueError("quick must be a boolean")
    if registration["quick"] and tier != "smoke":
        raise ValueError("quick is allowed only for the smoke evidence tier")
    if registration["quick"]:
        if any(environment["total_steps"] != 8_192 for environment in environments):
            raise ValueError("quick registrations must record total_steps=8192")
        for field, expected in _QUICK_LEARNER_VALUES.items():
            if learner[field] != expected:
                raise ValueError(f"quick registrations must record learner.{field}={expected}")
    if tier == "registered_final" and len(registration["seeds"]) != 30:
        raise ValueError("registered_final requires exactly 30 paired seeds")
    return registration


def _cell_namespace(
    registration: dict[str, Any],
    *,
    environment: dict[str, Any],
    model: str,
    seed: int,
    output: Path | None,
    manifest_sha256: str | None,
    cell_id: str | None,
    describe_only: bool,
) -> Namespace:
    learner = registration["learner"]
    return Namespace(
        environment=environment["id"],
        model=model,
        seed=seed,
        total_steps=environment["total_steps"],
        num_envs=learner["num_envs"],
        rollout_steps=learner["rollout_steps"],
        update_epochs=learner["update_epochs"],
        num_minibatches=learner.get("num_minibatches", 4),
        learning_rate=learner["learning_rate"],
        gae_lambda=learner.get("gae_lambda", 0.95),
        entropy_coefficient=learner.get("entropy_coefficient", 0.01),
        anneal_learning_rate=learner.get("anneal_learning_rate", False),
        registration_schema_version=registration["schema_version"],
        comparison_profile=registration.get("comparison_profile"),
        evaluation_episodes_per_env=registration["evaluation_episodes_per_env"],
        output=output,
        quick=registration["quick"],
        require_gpu=registration["require_gpu"],
        require_clean_git=True,
        evidence_tier=registration["evidence_tier"],
        matrix_manifest_sha256=manifest_sha256,
        cell_id=cell_id,
        describe_only=describe_only,
    )


def _command_for_cell(args: Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.pobax.run_pilot",
        "--environment",
        args.environment,
        "--model",
        args.model,
        "--seed",
        str(args.seed),
        "--total-steps",
        str(args.total_steps),
        "--num-envs",
        str(args.num_envs),
        "--rollout-steps",
        str(args.rollout_steps),
        "--update-epochs",
        str(args.update_epochs),
        "--num-minibatches",
        str(args.num_minibatches),
        "--learning-rate",
        str(args.learning_rate),
        "--gae-lambda",
        str(args.gae_lambda),
        "--entropy-coefficient",
        str(args.entropy_coefficient),
        "--registration-schema-version",
        str(args.registration_schema_version),
        "--evaluation-episodes-per-env",
        str(args.evaluation_episodes_per_env),
        "--evidence-tier",
        args.evidence_tier,
        "--matrix-manifest-sha256",
        args.matrix_manifest_sha256,
        "--cell-id",
        args.cell_id,
        "--output",
        str(args.output),
        "--require-clean-git",
    ]
    if args.anneal_learning_rate:
        command.append("--anneal-learning-rate")
    if args.comparison_profile is not None:
        command.extend(["--comparison-profile", args.comparison_profile])
    if args.require_gpu:
        command.append("--require-gpu")
    if args.quick:
        command.append("--quick")
    return command


def _load_matching_artifact(
    path: Path,
    *,
    expected_status: str,
    environment: str,
    model: str,
    seed: int,
    configuration_sha256: str,
    manifest_sha256: str,
    cell_id: str,
    provenance: dict[str, Any],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") not in {4, 5}:
        raise ExistingArtifactMismatchError(f"existing cell has the wrong schema: {path}")
    expected = {
        "status": expected_status,
        "environment": environment,
        "model": model,
        "seed": seed,
        "configuration_sha256": configuration_sha256,
        "matrix_manifest_sha256": manifest_sha256,
        "cell_id": cell_id,
    }
    actual = {field: artifact.get(field) for field in expected}
    if actual != expected:
        raise ExistingArtifactMismatchError(f"existing cell does not match frozen manifest: {path}")
    artifact_provenance = artifact.get("provenance")
    if artifact_provenance != provenance:
        raise ExistingArtifactMismatchError(
            f"existing cell provenance does not match frozen manifest: {path}"
        )
    configuration = artifact.get("configuration")
    if not isinstance(configuration, dict) or canonical_json_sha256(configuration) != artifact.get(
        "configuration_sha256"
    ):
        raise ExistingArtifactMismatchError(
            f"existing cell configuration content does not match its hash: {path}"
        )
    return artifact


def execute_matrix(registration_path: Path, output_root: Path) -> dict[str, Any]:
    """Describe, freeze, execute, and index one complete Cartesian matrix."""
    registration = _load_registration(registration_path.resolve())
    output_root = output_root.resolve()
    cells: list[dict[str, Any]] = []
    provenance: dict[str, Any] | None = None

    for environment in registration["environments"]:
        for model in registration["models"]:
            for seed in registration["seeds"]:
                description = run(
                    _cell_namespace(
                        registration,
                        environment=environment,
                        model=model,
                        seed=seed,
                        output=None,
                        manifest_sha256=None,
                        cell_id=None,
                        describe_only=True,
                    )
                )
                configuration_sha256 = description["configuration_sha256"]
                if provenance is None:
                    configuration = description["configuration"]
                    provenance = {
                        "git": description["runtime"]["git"],
                        "dependency_lock_sha256": configuration["dependency_lock_sha256"],
                        "pobax_commit": configuration["pobax_commit"],
                        "navix_commit": configuration["navix_commit"],
                        "runtime_contract": configuration["runtime_contract"],
                    }
                else:
                    current_provenance = {
                        "git": description["runtime"]["git"],
                        "dependency_lock_sha256": description["configuration"][
                            "dependency_lock_sha256"
                        ],
                        "pobax_commit": description["configuration"]["pobax_commit"],
                        "navix_commit": description["configuration"]["navix_commit"],
                        "runtime_contract": description["configuration"]["runtime_contract"],
                    }
                    if current_provenance != provenance:
                        raise RuntimeError("source provenance changed while describing the matrix")
                relative_path = registered_cell_path(
                    environment["id"],
                    model,
                    seed,
                    configuration_sha256,
                )
                cell_id = registered_cell_id(
                    environment["id"],
                    model,
                    seed,
                    configuration_sha256,
                )
                cells.append(
                    {
                        "cell_id": cell_id,
                        "environment": environment["id"],
                        "model": model,
                        "seed": seed,
                        "configuration_sha256": configuration_sha256,
                        "artifact_path": relative_path.as_posix(),
                    }
                )

    validate_unique_cell_ids(cell["cell_id"] for cell in cells)
    if provenance is None:
        raise AssertionError("matrix description produced no provenance")
    manifest_without_hash = {
        "schema_version": registration["schema_version"],
        "status": "frozen",
        "matrix_kind": registration["matrix_kind"],
        "models": registration["models"],
        "environments": [environment["id"] for environment in registration["environments"]],
        "seeds": registration["seeds"],
        "provenance": provenance,
        "cells": cells,
    }
    manifest_sha256 = canonical_json_sha256(manifest_without_hash)
    manifest = {
        **manifest_without_hash,
        "manifest_sha256": manifest_sha256,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "registration.json", registration)
    atomic_write_json(output_root / "frozen_manifest.json", manifest)

    expected_status = EVIDENCE_STATUS[registration["evidence_tier"]]
    completed_cells: list[dict[str, Any]] = []
    for environment in registration["environments"]:
        for model in registration["models"]:
            for seed in registration["seeds"]:
                cell = next(
                    candidate
                    for candidate in cells
                    if candidate["environment"] == environment["id"]
                    and candidate["model"] == model
                    and candidate["seed"] == seed
                )
                artifact_path = output_root / cell["artifact_path"]
                log_path = artifact_path.with_suffix(".log")
                artifact = _load_matching_artifact(
                    artifact_path,
                    expected_status=expected_status,
                    environment=environment["id"],
                    model=model,
                    seed=seed,
                    configuration_sha256=cell["configuration_sha256"],
                    manifest_sha256=manifest_sha256,
                    cell_id=cell["cell_id"],
                    provenance=provenance,
                )
                if artifact is None:
                    cell_args = _cell_namespace(
                        registration,
                        environment=environment,
                        model=model,
                        seed=seed,
                        output=artifact_path,
                        manifest_sha256=manifest_sha256,
                        cell_id=cell["cell_id"],
                        describe_only=False,
                    )
                    process = subprocess.run(
                        _command_for_cell(cell_args),
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    atomic_write_bytes(log_path, process.stdout)
                    if process.returncode != 0:
                        raise RuntimeError(
                            f"cell failed with exit code {process.returncode}: "
                            f"{cell['cell_id']}; log={log_path}"
                        )
                    artifact = _load_matching_artifact(
                        artifact_path,
                        expected_status=expected_status,
                        environment=environment["id"],
                        model=model,
                        seed=seed,
                        configuration_sha256=cell["configuration_sha256"],
                        manifest_sha256=manifest_sha256,
                        cell_id=cell["cell_id"],
                        provenance=provenance,
                    )
                    if artifact is None:
                        raise RuntimeError(f"cell completed without creating {artifact_path}")
                if not log_path.is_file():
                    raise RuntimeError(f"completed cell is missing its immutable log: {log_path}")
                completed_cells.append(
                    {
                        **cell,
                        "artifact_sha256": sha256_file(artifact_path),
                        "log_path": log_path.relative_to(output_root).as_posix(),
                        "log_sha256": sha256_file(log_path),
                    }
                )

    completion_index = {
        "schema_version": 1,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "planned_cells": len(cells),
        "completed_cells": len(completed_cells),
        "cells": completed_cells,
    }
    atomic_write_json(output_root / "completion_index.json", completion_index)
    write_checksum_manifest(output_root)
    return completion_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            execute_matrix(args.registration, args.output_root),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
