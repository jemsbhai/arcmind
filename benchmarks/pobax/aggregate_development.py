"""Strict aggregation for smoke, pilot, and tuning POBAX matrices.

This module is intentionally separate from the registered-final aggregator.
It accepts only development evidence, preserves raw evaluation returns, and
labels every output as unsuitable for paper claims. Tuning outputs rank a
complete Cartesian candidate matrix by a predeclared learning-curve metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from benchmarks.pobax.implementation_provenance import normalize_implementation_source
from benchmarks.pobax.registered_artifacts import (
    ArtifactChecksumError,
    atomic_write_json,
    canonical_json_bytes,
    canonical_json_sha256,
    registered_cell_id,
    sha256_file,
    validate_checksum_manifest,
)
from benchmarks.pobax.registration_protocol import (
    normalize_candidate_families,
    normalize_learner,
    realized_environment_steps,
    registration_fields,
    step_budget_mode,
    validate_comparison_profile,
    validate_development_tuning_contract,
)
from benchmarks.pobax.upper_reference_registry import (
    UPPER_REFERENCE_ENVIRONMENTS,
    expected_environment_reference,
    expected_environment_source,
)

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_723
CONFIDENCE_LEVEL = 0.95
_TIERS = {
    "smoke": "development_smoke_not_for_paper",
    "pilot": "development_pilot_not_for_paper",
    "development_tuning": "development_tuning_not_for_paper",
}
_AGGREGATE_STATUS = {
    "smoke": "development_smoke_aggregate_not_for_paper",
    "pilot": "development_pilot_aggregate_not_for_paper",
    "development_tuning": "development_tuning_selection_aggregate_not_for_paper",
}
_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "manifest_sha256",
    "matrix_kind",
    "models",
    "environments",
    "seeds",
    "provenance",
    "cells",
}
_MANIFEST_KEYS_V3 = _MANIFEST_KEYS | {"candidate_families"}
_CELL_KEYS = {
    "cell_id",
    "environment",
    "model",
    "seed",
    "configuration_sha256",
    "artifact_path",
}
_CELL_KEYS_V3 = _CELL_KEYS | {
    "model_family",
    "implementation_model",
    "implementation_source_sha256",
}
_PROVENANCE_KEYS = {
    "git",
    "dependency_lock_sha256",
    "pobax_commit",
    "navix_commit",
    "runtime_contract",
}
_PROVENANCE_KEYS_WITH_IMPLEMENTATION = _PROVENANCE_KEYS | {"implementation_source"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_REPORTED_RETURN_TOLERANCE = 1e-6
_PARAMETER_MATCH_KEYS = {
    "parameter_count",
    "effective_parameter_count",
    "arcmind_target_parameter_count",
    "parameter_ratio",
}
_OPTIMIZER_METRICS = (
    "loss",
    "actor_loss",
    "value_loss",
    "entropy",
    "approximate_kl",
)


class DevelopmentAggregationError(ValueError):
    """Raised when development artifacts are incomplete or inconsistent."""


def _validate_parameter_match(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    parameter_count = _integer(
        value["parameter_count"],
        field=f"{field}.parameter_count",
        positive=True,
    )
    effective_parameter_count = _integer(
        value["effective_parameter_count"],
        field=f"{field}.effective_parameter_count",
        positive=True,
    )
    target_parameter_count = _integer(
        value["arcmind_target_parameter_count"],
        field=f"{field}.arcmind_target_parameter_count",
        positive=True,
    )
    parameter_ratio = _number(value["parameter_ratio"], field=f"{field}.parameter_ratio")
    computed_ratio = parameter_count / target_parameter_count
    if effective_parameter_count > parameter_count:
        raise DevelopmentAggregationError(
            f"{field}.effective_parameter_count exceeds parameter_count"
        )
    if not math.isclose(parameter_ratio, computed_ratio, rel_tol=1e-12, abs_tol=1e-12):
        raise DevelopmentAggregationError(
            f"{field}.parameter_ratio disagrees with parameter counts"
        )
    if not 0.9 <= parameter_ratio <= 1.1:
        raise DevelopmentAggregationError(
            f"{field}.parameter_ratio violates the matching tolerance"
        )
    return {
        "parameter_count": parameter_count,
        "effective_parameter_count": effective_parameter_count,
        "arcmind_target_parameter_count": target_parameter_count,
        "parameter_ratio": parameter_ratio,
    }


def _reject_constant(value: str) -> None:
    raise DevelopmentAggregationError(f"non-finite JSON constant is prohibited: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DevelopmentAggregationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, *, field: str) -> Any:
    if not path.is_file():
        raise DevelopmentAggregationError(f"{field} does not exist or is not a file: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DevelopmentAggregationError(f"cannot read {field} {path}: {error}") from error


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DevelopmentAggregationError(f"{field} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    actual = set(value)
    if actual != expected:
        raise DevelopmentAggregationError(
            f"{field} has wrong fields: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DevelopmentAggregationError(f"{field} must be a non-empty string")
    return value


def _integer(value: Any, *, field: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DevelopmentAggregationError(f"{field} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise DevelopmentAggregationError(f"{field} must be {qualifier}")
    return value


def _number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DevelopmentAggregationError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise DevelopmentAggregationError(f"{field} must be a finite number")
    return result


def _sha256(value: Any, *, field: str) -> str:
    result = _string(value, field=field)
    if not _SHA256_PATTERN.fullmatch(result):
        raise DevelopmentAggregationError(f"{field} must be a lowercase SHA256")
    return result


def _commit(value: Any, *, field: str) -> str:
    result = _string(value, field=field)
    if not _COMMIT_PATTERN.fullmatch(result):
        raise DevelopmentAggregationError(f"{field} must be a lowercase Git commit")
    return result


def _unique_strings(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise DevelopmentAggregationError(f"{field} must be a non-empty JSON list")
    result = tuple(_string(item, field=f"{field}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise DevelopmentAggregationError(f"{field} contains duplicates")
    return result


def _unique_seeds(value: Any, *, field: str = "seeds") -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise DevelopmentAggregationError(f"{field} must be a non-empty JSON list")
    result = tuple(_integer(item, field=f"{field}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise DevelopmentAggregationError(f"{field} contains duplicates")
    return result


def _validate_runtime(value: Any, *, field: str) -> dict[str, Any]:
    runtime = _mapping(value, field=field)
    expected = {"python", "packages", "jax_backend", "jax_enable_x64", "devices"}
    _exact_keys(runtime, expected, field=field)
    python = _mapping(runtime["python"], field=f"{field}.python")
    _exact_keys(python, {"implementation", "version"}, field=f"{field}.python")
    packages = _mapping(runtime["packages"], field=f"{field}.packages")
    if not packages:
        raise DevelopmentAggregationError(f"{field}.packages must not be empty")
    normalized_packages = {
        _string(name, field=f"{field}.packages key"): _string(
            version, field=f"{field}.packages.{name}"
        )
        for name, version in packages.items()
    }
    devices = runtime["devices"]
    if not isinstance(devices, list) or not devices:
        raise DevelopmentAggregationError(f"{field}.devices must be a non-empty JSON list")
    normalized_devices: list[dict[str, str]] = []
    for index, raw_device in enumerate(devices):
        device_field = f"{field}.devices[{index}]"
        device = _mapping(raw_device, field=device_field)
        _exact_keys(device, {"platform", "device_kind"}, field=device_field)
        normalized_devices.append(
            {
                "platform": _string(device["platform"], field=f"{device_field}.platform"),
                "device_kind": _string(device["device_kind"], field=f"{device_field}.device_kind"),
            }
        )
    if not isinstance(runtime["jax_enable_x64"], bool):
        raise DevelopmentAggregationError(f"{field}.jax_enable_x64 must be a boolean")
    return {
        "python": {
            "implementation": _string(
                python["implementation"], field=f"{field}.python.implementation"
            ),
            "version": _string(python["version"], field=f"{field}.python.version"),
        },
        "packages": dict(sorted(normalized_packages.items())),
        "jax_backend": _string(runtime["jax_backend"], field=f"{field}.jax_backend"),
        "jax_enable_x64": runtime["jax_enable_x64"],
        "devices": normalized_devices,
    }


def _validate_provenance(
    value: Any,
    *,
    field: str,
    require_implementation_source: bool,
) -> dict[str, Any]:
    provenance = _mapping(value, field=field)
    expected_keys = (
        _PROVENANCE_KEYS_WITH_IMPLEMENTATION if require_implementation_source else _PROVENANCE_KEYS
    )
    _exact_keys(provenance, expected_keys, field=field)
    git = _mapping(provenance["git"], field=f"{field}.git")
    _exact_keys(git, {"commit", "dirty", "diff_sha256"}, field=f"{field}.git")
    if git["dirty"] is not False or git["diff_sha256"] is not None:
        raise DevelopmentAggregationError(f"{field}.git must describe a clean worktree")
    normalized = {
        "git": {
            "commit": _commit(git["commit"], field=f"{field}.git.commit"),
            "dirty": False,
            "diff_sha256": None,
        },
        "dependency_lock_sha256": _sha256(
            provenance["dependency_lock_sha256"],
            field=f"{field}.dependency_lock_sha256",
        ),
        "pobax_commit": _commit(provenance["pobax_commit"], field=f"{field}.pobax_commit"),
        "navix_commit": _commit(provenance["navix_commit"], field=f"{field}.navix_commit"),
        "runtime_contract": _validate_runtime(
            provenance["runtime_contract"], field=f"{field}.runtime_contract"
        ),
    }
    if require_implementation_source:
        try:
            normalized["implementation_source"] = normalize_implementation_source(
                provenance["implementation_source"]
            )
        except ValueError as error:
            raise DevelopmentAggregationError(str(error)) from error
    return normalized


def _validate_registration(value: Any) -> dict[str, Any]:
    registration = _mapping(value, field="registration")
    schema_version = registration.get("schema_version")
    try:
        expected_fields = registration_fields(schema_version)
    except ValueError as error:
        raise DevelopmentAggregationError(str(error)) from error
    _exact_keys(registration, expected_fields, field="registration")
    if registration["status"] != "frozen":
        raise DevelopmentAggregationError("registration must be frozen")
    try:
        comparison_profile = validate_comparison_profile(registration)
    except ValueError as error:
        raise DevelopmentAggregationError(str(error)) from error
    tier = _string(registration["evidence_tier"], field="registration.evidence_tier")
    if tier not in _TIERS:
        raise DevelopmentAggregationError(
            "registration.evidence_tier must be 'smoke', 'pilot', or 'development_tuning'"
        )
    matrix_kind = _string(registration["matrix_kind"], field="registration.matrix_kind")
    if matrix_kind not in {
        "primary_comparison",
        "upper_reference",
        "hyperparameter_selection",
    }:
        raise DevelopmentAggregationError("registration.matrix_kind is unsupported")
    candidate_families: tuple[dict[str, Any], ...] = ()
    candidate_specs: dict[str, dict[str, Any]] = {}
    if schema_version == 3:
        try:
            candidate_families = normalize_candidate_families(registration["candidate_families"])
        except ValueError as error:
            raise DevelopmentAggregationError(str(error)) from error
        models = tuple(
            candidate["candidate_id"]
            for family in candidate_families
            for candidate in family["candidates"]
        )
        candidate_specs = {
            candidate["candidate_id"]: {
                **candidate,
                "model_family": family["family_id"],
            }
            for family in candidate_families
            for candidate in family["candidates"]
        }
        if tier != "development_tuning":
            raise DevelopmentAggregationError(
                "registration schema version 3 is reserved for development_tuning"
            )
    else:
        models = _unique_strings(registration["models"], field="registration.models")
    if matrix_kind == "primary_comparison" and "arcmind" not in models:
        raise DevelopmentAggregationError("primary_comparison registration must contain arcmind")
    if matrix_kind == "upper_reference" and models != ("memoryless_mlp",):
        raise DevelopmentAggregationError(
            "upper_reference registration must contain only memoryless_mlp"
        )
    if matrix_kind == "hyperparameter_selection" and schema_version != 3:
        raise DevelopmentAggregationError(
            "hyperparameter_selection requires registration schema version 3"
        )
    seeds = _unique_seeds(registration["seeds"], field="registration.seeds")
    raw_environments = registration["environments"]
    if not isinstance(raw_environments, list) or not raw_environments:
        raise DevelopmentAggregationError("registration.environments must be a non-empty JSON list")
    environments: list[dict[str, Any]] = []
    for index, raw_environment in enumerate(raw_environments):
        field = f"registration.environments[{index}]"
        environment = _mapping(raw_environment, field=field)
        _exact_keys(environment, {"id", "total_steps"}, field=field)
        environments.append(
            {
                "id": _string(environment["id"], field=f"{field}.id"),
                "total_steps": _integer(
                    environment["total_steps"], field=f"{field}.total_steps", positive=True
                ),
            }
        )
    environment_ids = [environment["id"] for environment in environments]
    if len(set(environment_ids)) != len(environment_ids):
        raise DevelopmentAggregationError("registration environment ids contain duplicates")
    selected_upper_references = set(environment_ids) & UPPER_REFERENCE_ENVIRONMENTS
    if matrix_kind == "upper_reference":
        invalid_environments = set(environment_ids) - UPPER_REFERENCE_ENVIRONMENTS
        if invalid_environments:
            raise DevelopmentAggregationError(
                "upper_reference registration contains non-reference environments: "
                f"{sorted(invalid_environments)}"
            )
    elif selected_upper_references:
        raise DevelopmentAggregationError(
            "primary_comparison registration contains upper-reference aliases: "
            f"{sorted(selected_upper_references)}"
        )
    try:
        normalized_learner = (
            None
            if schema_version == 3
            else normalize_learner(
                registration["learner"],
                schema_version=schema_version,
            )
        )
        learners = (
            [candidate["learner"] for candidate in candidate_specs.values()]
            if schema_version == 3
            else [normalized_learner]
        )
        for learner in learners:
            for environment in environments:
                realized_environment_steps(
                    environment["total_steps"],
                    num_envs=int(learner["num_envs"]),
                    rollout_steps=int(learner["rollout_steps"]),
                    comparison_profile=comparison_profile,
                )
    except ValueError as error:
        raise DevelopmentAggregationError(str(error)) from error
    evaluation_episodes = _integer(
        registration["evaluation_episodes_per_env"],
        field="registration.evaluation_episodes_per_env",
        positive=True,
    )
    if not isinstance(registration["require_gpu"], bool) or not isinstance(
        registration["quick"], bool
    ):
        raise DevelopmentAggregationError("registration boolean fields must be booleans")
    if registration["quick"] and tier != "smoke":
        raise DevelopmentAggregationError("quick registrations are allowed only for smoke")
    if registration["quick"]:
        if any(environment["total_steps"] != 8_192 for environment in environments):
            raise DevelopmentAggregationError("quick registrations must use 8192 total steps")
        for name, expected in {
            "num_envs": 32,
            "rollout_steps": 32,
            "update_epochs": 2,
        }.items():
            if normalized_learner[name] != expected:
                raise DevelopmentAggregationError(
                    f"quick registration learner.{name} must equal {expected}"
                )
    if tier == "development_tuning":
        try:
            validate_development_tuning_contract(
                schema_version=schema_version,
                comparison_profile=comparison_profile,
                matrix_kind=matrix_kind,
                candidate_families=candidate_families,
                environments={
                    environment["id"]: environment["total_steps"] for environment in environments
                },
                seeds=list(seeds),
                quick=registration["quick"],
            )
        except ValueError as error:
            raise DevelopmentAggregationError(str(error)) from error
    return {
        "schema_version": schema_version,
        "comparison_profile": comparison_profile,
        "tier": tier,
        "matrix_kind": matrix_kind,
        "models": models,
        "candidate_families": candidate_families,
        "candidate_specs": candidate_specs,
        "environments": tuple(environment_ids),
        "budgets": {environment["id"]: environment["total_steps"] for environment in environments},
        "seeds": seeds,
        "learner": normalized_learner,
        "evaluation_episodes": evaluation_episodes,
        "require_gpu": registration["require_gpu"],
        "quick": registration["quick"],
    }


def _artifact_path(root: Path, value: Any, *, field: str) -> Path:
    text = _string(value, field=field)
    if "\\" in text:
        raise DevelopmentAggregationError(f"{field} must use POSIX separators")
    pure = PurePosixPath(text)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise DevelopmentAggregationError(f"{field} must be a normalized relative path")
    resolved = root.joinpath(*pure.parts).resolve()
    if not resolved.is_relative_to(root):
        raise DevelopmentAggregationError(f"{field} escapes the output root")
    return resolved


def _validate_manifest(
    value: Any,
    *,
    root: Path,
    registration: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[tuple[str, str, int], dict[str, Any]]]:
    manifest = _mapping(value, field="manifest")
    _exact_keys(
        manifest,
        _MANIFEST_KEYS_V3 if registration["schema_version"] == 3 else _MANIFEST_KEYS,
        field="manifest",
    )
    if (
        manifest["schema_version"] != registration["schema_version"]
        or manifest["status"] != "frozen"
    ):
        raise DevelopmentAggregationError(
            "manifest schema version must match its frozen registration"
        )
    manifest_sha256 = _sha256(manifest["manifest_sha256"], field="manifest.manifest_sha256")
    hash_input = dict(manifest)
    del hash_input["manifest_sha256"]
    if canonical_json_sha256(hash_input) != manifest_sha256:
        raise DevelopmentAggregationError(
            "manifest.manifest_sha256 does not match canonical content"
        )
    models = _unique_strings(manifest["models"], field="manifest.models")
    environments = _unique_strings(manifest["environments"], field="manifest.environments")
    seeds = _unique_seeds(manifest["seeds"], field="manifest.seeds")
    matrix_kind = _string(manifest["matrix_kind"], field="manifest.matrix_kind")
    if (
        models != registration["models"]
        or environments != registration["environments"]
        or seeds != registration["seeds"]
        or matrix_kind != registration["matrix_kind"]
    ):
        raise DevelopmentAggregationError("manifest identity drifts from registration")
    if registration["schema_version"] == 3:
        try:
            manifest_families = normalize_candidate_families(manifest["candidate_families"])
        except ValueError as error:
            raise DevelopmentAggregationError(str(error)) from error
        if manifest_families != registration["candidate_families"]:
            raise DevelopmentAggregationError("manifest candidate families drift from registration")
    provenance = _validate_provenance(
        manifest["provenance"],
        field="manifest.provenance",
        require_implementation_source=registration["schema_version"] == 3,
    )
    raw_cells = manifest["cells"]
    if not isinstance(raw_cells, list) or not raw_cells:
        raise DevelopmentAggregationError("manifest.cells must be a non-empty JSON list")
    expected_identities = {
        (environment, model, seed)
        for environment in environments
        for model in models
        for seed in seeds
    }
    cells: dict[tuple[str, str, int], dict[str, Any]] = {}
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for index, raw_cell in enumerate(raw_cells):
        field = f"manifest.cells[{index}]"
        cell = _mapping(raw_cell, field=field)
        _exact_keys(
            cell,
            _CELL_KEYS_V3 if registration["schema_version"] == 3 else _CELL_KEYS,
            field=field,
        )
        identity = (
            _string(cell["environment"], field=f"{field}.environment"),
            _string(cell["model"], field=f"{field}.model"),
            _integer(cell["seed"], field=f"{field}.seed"),
        )
        if identity not in expected_identities:
            raise DevelopmentAggregationError(f"{field} is outside the Cartesian matrix")
        if identity in cells:
            raise DevelopmentAggregationError(f"duplicate manifest cell identity: {identity!r}")
        configuration_sha256 = _sha256(
            cell["configuration_sha256"], field=f"{field}.configuration_sha256"
        )
        cell_id = _sha256(cell["cell_id"], field=f"{field}.cell_id")
        if cell_id != registered_cell_id(*identity, configuration_sha256):
            raise DevelopmentAggregationError(f"{field}.cell_id does not match its identity")
        path = _artifact_path(root, cell["artifact_path"], field=f"{field}.artifact_path")
        if cell_id in seen_ids or path in seen_paths:
            raise DevelopmentAggregationError("manifest cell IDs and paths must be unique")
        seen_ids.add(cell_id)
        seen_paths.add(path)
        candidate_metadata: dict[str, str] = {}
        if registration["schema_version"] == 3:
            expected_candidate = registration["candidate_specs"][identity[1]]
            model_family = _string(
                cell["model_family"],
                field=f"{field}.model_family",
            )
            implementation_model = _string(
                cell["implementation_model"],
                field=f"{field}.implementation_model",
            )
            implementation_source_sha256 = _sha256(
                cell["implementation_source_sha256"],
                field=f"{field}.implementation_source_sha256",
            )
            if (
                model_family != expected_candidate["model_family"]
                or implementation_model != expected_candidate["implementation_model"]
                or implementation_source_sha256 != provenance["implementation_source"]["sha256"]
            ):
                raise DevelopmentAggregationError(
                    f"{field} candidate identity drifts from registration"
                )
            candidate_metadata = {
                "model_family": model_family,
                "implementation_model": implementation_model,
                "implementation_source_sha256": implementation_source_sha256,
            }
        cells[identity] = {
            "cell_id": cell_id,
            "configuration_sha256": configuration_sha256,
            "artifact_path": path,
            "artifact_relative_path": PurePosixPath(cell["artifact_path"]).as_posix(),
            **candidate_metadata,
        }
    missing = sorted(expected_identities - set(cells))
    if missing or len(cells) != len(expected_identities):
        raise DevelopmentAggregationError(f"manifest is missing Cartesian cells: {missing}")
    return {
        "schema_version": manifest["schema_version"],
        "manifest_sha256": manifest_sha256,
        "provenance": provenance,
    }, cells


def _validate_configuration(
    value: Any,
    *,
    identity: tuple[str, str, int],
    registration: Mapping[str, Any],
    provenance: Mapping[str, Any],
    field: str,
) -> tuple[Mapping[str, Any], int, int, int, dict[str, bool]]:
    configuration = _mapping(value, field=field)
    required = {
        "schema_version",
        "evidence_tier",
        "environment",
        "model",
        "seed",
        "ppo",
        "evaluation_episodes_per_environment",
        "evaluation_max_episode_steps",
        "dependency_lock_sha256",
        "pobax_commit",
        "navix_commit",
        "runtime_contract",
    }
    missing = sorted(required - set(configuration))
    if missing:
        raise DevelopmentAggregationError(f"{field} is missing required fields: {missing}")
    if configuration["schema_version"] != registration["schema_version"]:
        raise DevelopmentAggregationError(f"{field}.schema_version must match the registration")
    if registration["schema_version"] == 3:
        required.update(
            {
                "candidate_id",
                "model_family",
                "implementation_model",
                "implementation_source",
            }
        )
        missing = sorted(required - set(configuration))
        if missing:
            raise DevelopmentAggregationError(f"{field} is missing required fields: {missing}")
    if registration["schema_version"] in {2, 3}:
        for name in (
            "comparison_profile",
            "requested_environment_steps",
            "realized_environment_steps",
        ):
            if name not in configuration:
                raise DevelopmentAggregationError(f"{field} is missing {name}")
        if configuration["comparison_profile"] != registration["comparison_profile"]:
            raise DevelopmentAggregationError(
                f"{field}.comparison_profile drifts from registration"
            )
    if configuration["evidence_tier"] != registration["tier"]:
        raise DevelopmentAggregationError(f"{field}.evidence_tier drifts from registration")
    configured_identity = (
        _string(configuration["environment"], field=f"{field}.environment"),
        _string(configuration["model"], field=f"{field}.model"),
        _integer(configuration["seed"], field=f"{field}.seed"),
    )
    if configured_identity != identity:
        raise DevelopmentAggregationError(f"{field} identity does not match its artifact")
    if registration["schema_version"] == 3:
        candidate = registration["candidate_specs"][identity[1]]
        if (
            configuration["candidate_id"] != identity[1]
            or configuration["model_family"] != candidate["model_family"]
            or configuration["implementation_model"] != candidate["implementation_model"]
        ):
            raise DevelopmentAggregationError(
                f"{field} candidate identity drifts from registration"
            )
    environment = identity[0]
    expected_source = expected_environment_source(environment)
    source_frozen = "environment_source" in configuration
    if source_frozen:
        configured_source = _mapping(
            configuration["environment_source"],
            field=f"{field}.environment_source",
        )
        if dict(configured_source) != expected_source:
            raise DevelopmentAggregationError(
                f"{field}.environment_source does not match the registered source invocation"
            )
    elif environment in UPPER_REFERENCE_ENVIRONMENTS:
        raise DevelopmentAggregationError(
            f"{field}.environment_source is required for upper-reference cells"
        )
    expected_reference = expected_environment_reference(environment)
    if configuration.get("environment_reference") != expected_reference:
        raise DevelopmentAggregationError(
            f"{field}.environment_reference does not match the registered reference class"
        )
    present_parameter_keys = _PARAMETER_MATCH_KEYS & set(configuration)
    if present_parameter_keys and present_parameter_keys != _PARAMETER_MATCH_KEYS:
        missing_parameter_keys = sorted(_PARAMETER_MATCH_KEYS - set(configuration))
        raise DevelopmentAggregationError(
            f"{field} has a partial parameter-match contract: missing={missing_parameter_keys}"
        )
    parameter_match_frozen = present_parameter_keys == _PARAMETER_MATCH_KEYS
    if parameter_match_frozen:
        _validate_parameter_match(configuration, field=field)
    ppo = _mapping(configuration["ppo"], field=f"{field}.ppo")
    total_steps = _integer(ppo.get("total_steps"), field=f"{field}.ppo.total_steps", positive=True)
    if total_steps != registration["budgets"][identity[0]]:
        raise DevelopmentAggregationError(f"{field}.ppo.total_steps drifts from registration")
    learner_fields = (
        "num_envs",
        "rollout_steps",
        "update_epochs",
        "learning_rate",
    )
    if registration["schema_version"] in {2, 3}:
        learner_fields += (
            "num_minibatches",
            "gae_lambda",
            "entropy_coefficient",
            "anneal_learning_rate",
        )
    for configured_name in learner_fields:
        if configured_name not in ppo:
            raise DevelopmentAggregationError(f"{field}.ppo is missing {configured_name}")
        configured = ppo[configured_name]
        expected_learner = (
            registration["candidate_specs"][identity[1]]["learner"]
            if registration["schema_version"] == 3
            else registration["learner"]
        )
        expected = expected_learner[configured_name]
        if configured != expected:
            raise DevelopmentAggregationError(
                f"{field}.ppo.{configured_name} drifts from registration"
            )
    try:
        expected_realized_steps = realized_environment_steps(
            total_steps,
            num_envs=int(expected_learner["num_envs"]),
            rollout_steps=int(expected_learner["rollout_steps"]),
            comparison_profile=registration["comparison_profile"],
        )
    except ValueError as error:  # pragma: no cover - registration validates first
        raise DevelopmentAggregationError(str(error)) from error
    if registration["schema_version"] in {
        2,
        3,
    } and ppo.get("step_budget_mode") != step_budget_mode(registration["comparison_profile"]):
        raise DevelopmentAggregationError(
            f"{field}.ppo.step_budget_mode drifts from comparison_profile"
        )
    if registration["schema_version"] in {2, 3}:
        requested_steps = _integer(
            configuration["requested_environment_steps"],
            field=f"{field}.requested_environment_steps",
            positive=True,
        )
        realized_steps = _integer(
            configuration["realized_environment_steps"],
            field=f"{field}.realized_environment_steps",
            positive=True,
        )
        if requested_steps != total_steps or realized_steps != expected_realized_steps:
            raise DevelopmentAggregationError(
                f"{field} requested and realized step counts are inconsistent"
            )
    evaluation_episodes = _integer(
        configuration["evaluation_episodes_per_environment"],
        field=f"{field}.evaluation_episodes_per_environment",
        positive=True,
    )
    if evaluation_episodes != registration["evaluation_episodes"]:
        raise DevelopmentAggregationError(
            f"{field}.evaluation_episodes_per_environment drifts from registration"
        )
    evaluation_horizon = _integer(
        configuration["evaluation_max_episode_steps"],
        field=f"{field}.evaluation_max_episode_steps",
        positive=True,
    )
    configured_provenance = {
        "dependency_lock_sha256": _sha256(
            configuration["dependency_lock_sha256"],
            field=f"{field}.dependency_lock_sha256",
        ),
        "pobax_commit": _commit(configuration["pobax_commit"], field=f"{field}.pobax_commit"),
        "navix_commit": _commit(configuration["navix_commit"], field=f"{field}.navix_commit"),
        "runtime_contract": _validate_runtime(
            configuration["runtime_contract"], field=f"{field}.runtime_contract"
        ),
    }
    if registration["schema_version"] == 3:
        try:
            configured_provenance["implementation_source"] = normalize_implementation_source(
                configuration["implementation_source"]
            )
        except ValueError as error:
            raise DevelopmentAggregationError(str(error)) from error
    provenance_fields = [
        "dependency_lock_sha256",
        "pobax_commit",
        "navix_commit",
        "runtime_contract",
    ]
    if registration["schema_version"] == 3:
        provenance_fields.append("implementation_source")
    if configured_provenance != {key: provenance[key] for key in provenance_fields}:
        raise DevelopmentAggregationError(f"{field} provenance drifts from manifest")
    return (
        configuration,
        expected_realized_steps,
        evaluation_episodes,
        evaluation_horizon,
        {
            "environment_source_frozen": source_frozen,
            "parameter_match_frozen": parameter_match_frozen,
        },
    )


def _validate_evaluation(
    artifact: Mapping[str, Any],
    *,
    field: str,
    expected_episodes: int,
    expected_horizon: int,
    expected_num_environments: int,
) -> dict[str, Any]:
    evaluation = _mapping(artifact.get("evaluation"), field=f"{field}.evaluation")
    required = {
        "mean_return",
        "median_return",
        "episodes",
        "episodes_per_environment",
        "num_environments",
        "scan_steps_per_environment",
        "returns_by_environment",
    }
    missing = sorted(required - set(evaluation))
    if missing:
        raise DevelopmentAggregationError(
            f"{field}.evaluation is missing required fields: {missing}"
        )
    episodes = _integer(evaluation["episodes"], field=f"{field}.evaluation.episodes", positive=True)
    episodes_per_environment = _integer(
        evaluation["episodes_per_environment"],
        field=f"{field}.evaluation.episodes_per_environment",
        positive=True,
    )
    num_environments = _integer(
        evaluation["num_environments"],
        field=f"{field}.evaluation.num_environments",
        positive=True,
    )
    scan_steps = _integer(
        evaluation["scan_steps_per_environment"],
        field=f"{field}.evaluation.scan_steps_per_environment",
        positive=True,
    )
    expected_scan_steps = expected_episodes * expected_horizon
    if (
        episodes_per_environment != expected_episodes
        or num_environments != expected_num_environments
        or scan_steps != expected_scan_steps
    ):
        raise DevelopmentAggregationError(
            f"{field}.evaluation does not match the frozen evaluation contract"
        )
    rows = evaluation["returns_by_environment"]
    if not isinstance(rows, list) or len(rows) != num_environments:
        raise DevelopmentAggregationError(
            f"{field}.evaluation.returns_by_environment has the wrong row count"
        )
    normalized_rows: list[list[float]] = []
    flat: list[float] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != episodes_per_environment:
            raise DevelopmentAggregationError(
                f"{field}.evaluation.returns_by_environment[{row_index}] has the wrong return count"
            )
        normalized = [
            _number(
                item,
                field=(f"{field}.evaluation.returns_by_environment[{row_index}][{column_index}]"),
            )
            for column_index, item in enumerate(row)
        ]
        normalized_rows.append(normalized)
        flat.extend(normalized)
    if episodes != len(flat) or episodes != expected_episodes * num_environments:
        raise DevelopmentAggregationError(f"{field}.evaluation episode counts are inconsistent")
    mean_return = _number(evaluation["mean_return"], field=f"{field}.evaluation.mean_return")
    median_return = _number(evaluation["median_return"], field=f"{field}.evaluation.median_return")
    computed_mean = float(np.mean(flat))
    computed_median = float(np.median(flat))
    # SharedPPO reports JAX float32 reductions while JSON preserves each
    # float32 return exactly. Recomputing from JSON in float64 can therefore
    # differ by a few float32 ULPs.
    if not math.isclose(
        mean_return,
        computed_mean,
        rel_tol=_REPORTED_RETURN_TOLERANCE,
        abs_tol=_REPORTED_RETURN_TOLERANCE,
    ):
        raise DevelopmentAggregationError(
            f"{field}.evaluation.mean_return disagrees with raw returns"
        )
    if not math.isclose(
        median_return,
        computed_median,
        rel_tol=_REPORTED_RETURN_TOLERANCE,
        abs_tol=_REPORTED_RETURN_TOLERANCE,
    ):
        raise DevelopmentAggregationError(
            f"{field}.evaluation.median_return disagrees with raw returns"
        )
    return {
        "mean_return": computed_mean,
        "median_return": computed_median,
        "returns_by_environment": normalized_rows,
        "episodes": episodes,
        "episodes_per_environment": episodes_per_environment,
        "num_environments": num_environments,
        "scan_steps_per_environment": scan_steps,
    }


def _validate_history(
    artifact: Mapping[str, Any], *, field: str, expected_final_steps: int
) -> tuple[tuple[int, ...], tuple[float | None, ...]]:
    history = artifact.get("training_history")
    if not isinstance(history, list) or not history:
        raise DevelopmentAggregationError(f"{field}.training_history must not be empty")
    steps: list[int] = []
    returns: list[float | None] = []
    return_available = False
    for index, raw_point in enumerate(history):
        point_field = f"{field}.training_history[{index}]"
        point = _mapping(raw_point, field=point_field)
        if "environment_steps" not in point or "mean_recent_return" not in point:
            raise DevelopmentAggregationError(
                f"{point_field} must contain environment_steps and mean_recent_return"
            )
        missing_optimizer_metrics = sorted(set(_OPTIMIZER_METRICS) - set(point))
        if missing_optimizer_metrics:
            raise DevelopmentAggregationError(
                f"{point_field} is missing optimizer metrics: {missing_optimizer_metrics}"
            )
        for metric in _OPTIMIZER_METRICS:
            _number(point[metric], field=f"{point_field}.{metric}")
        step = _number(point["environment_steps"], field=f"{point_field}.environment_steps")
        if not step.is_integer() or step <= 0:
            raise DevelopmentAggregationError(
                f"{point_field}.environment_steps must be a positive integer-valued number"
            )
        steps.append(int(step))
        recent_return = point["mean_recent_return"]
        if recent_return is None:
            if return_available:
                raise DevelopmentAggregationError(
                    f"{point_field}.mean_recent_return disappeared after becoming available"
                )
            returns.append(None)
        else:
            returns.append(_number(recent_return, field=f"{point_field}.mean_recent_return"))
            return_available = True
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise DevelopmentAggregationError(
            f"{field}.training_history steps must be strictly increasing"
        )
    if steps[-1] != expected_final_steps:
        raise DevelopmentAggregationError(
            f"{field}.training_history final step does not match frozen total_steps"
        )
    return tuple(steps), tuple(returns)


def _validate_final_training_metrics(artifact: Mapping[str, Any], *, field: str) -> None:
    training = _mapping(artifact.get("training"), field=f"{field}.training")
    missing = sorted(set(_OPTIMIZER_METRICS) - set(training))
    if missing:
        raise DevelopmentAggregationError(
            f"{field}.training is missing optimizer metrics: {missing}"
        )
    for metric in _OPTIMIZER_METRICS:
        _number(training[metric], field=f"{field}.training.{metric}")


def _validate_artifact(
    path: Path,
    *,
    identity: tuple[str, str, int],
    expected: Mapping[str, Any],
    registration: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    environment, model, seed = identity
    field = f"artifact[{environment},{model},{seed}]"
    artifact = _mapping(_load_json(path, field=field), field=field)
    required = {
        "schema_version",
        "status",
        "matrix_manifest_sha256",
        "cell_id",
        "configuration_sha256",
        "configuration",
        "environment",
        "model",
        "seed",
        "parameter_count",
        "effective_parameter_count",
        "arcmind_target_parameter_count",
        "parameter_ratio",
        "provenance",
        "actual_environment_steps",
        "ppo",
        "evaluation_episodes_per_environment",
        "evaluation_max_episode_steps",
        "actual_evaluation_steps_per_environment",
        "actual_evaluation_transitions",
        "evaluation",
        "training",
        "training_history",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise DevelopmentAggregationError(f"{field} is missing required fields: {missing}")
    if registration["schema_version"] == 3:
        required.update(
            {
                "candidate_id",
                "model_family",
                "implementation_model",
                "implementation_source_sha256",
            }
        )
        missing = sorted(required - set(artifact))
        if missing:
            raise DevelopmentAggregationError(f"{field} is missing required fields: {missing}")
    expected_artifact_schema = (
        6
        if registration["schema_version"] == 3
        else 5
        if registration["schema_version"] == 2
        else 4
    )
    if artifact["schema_version"] != expected_artifact_schema:
        raise DevelopmentAggregationError(
            f"{field}.schema_version must equal current schema {expected_artifact_schema}"
        )
    if artifact["status"] != _TIERS[registration["tier"]]:
        raise DevelopmentAggregationError(
            f"{field}.status does not match development tier {registration['tier']!r}"
        )
    if (
        _sha256(
            artifact["matrix_manifest_sha256"],
            field=f"{field}.matrix_manifest_sha256",
        )
        != manifest["manifest_sha256"]
    ):
        raise DevelopmentAggregationError(f"{field} belongs to another manifest")
    if _sha256(artifact["cell_id"], field=f"{field}.cell_id") != expected["cell_id"]:
        raise DevelopmentAggregationError(f"{field}.cell_id drifted from manifest")
    configuration_sha256 = _sha256(
        artifact["configuration_sha256"], field=f"{field}.configuration_sha256"
    )
    if configuration_sha256 != expected["configuration_sha256"]:
        raise DevelopmentAggregationError(f"{field}.configuration_sha256 drifted from manifest")
    configuration = _mapping(artifact["configuration"], field=f"{field}.configuration")
    if canonical_json_sha256(configuration) != configuration_sha256:
        raise DevelopmentAggregationError(
            f"{field}.configuration_sha256 does not match configuration"
        )
    artifact_identity = (
        _string(artifact["environment"], field=f"{field}.environment"),
        _string(artifact["model"], field=f"{field}.model"),
        _integer(artifact["seed"], field=f"{field}.seed"),
    )
    if artifact_identity != identity:
        raise DevelopmentAggregationError(f"{field} identity drifted from manifest")
    if registration["schema_version"] == 3:
        candidate = registration["candidate_specs"][model]
        if (
            artifact["candidate_id"] != model
            or artifact["model_family"] != candidate["model_family"]
            or artifact["implementation_model"] != candidate["implementation_model"]
            or artifact["candidate_id"] != configuration["candidate_id"]
            or artifact["model_family"] != configuration["model_family"]
            or artifact["implementation_model"] != configuration["implementation_model"]
            or artifact["implementation_source_sha256"]
            != configuration["implementation_source"]["sha256"]
        ):
            raise DevelopmentAggregationError(
                f"{field} candidate identity drifts from frozen configuration"
            )
    artifact_provenance = _validate_provenance(
        artifact["provenance"],
        field=f"{field}.provenance",
        require_implementation_source=manifest["schema_version"] == 3,
    )
    if artifact_provenance != manifest["provenance"]:
        raise DevelopmentAggregationError(f"{field}.provenance drifted from manifest")
    (
        configuration,
        total_steps,
        evaluation_episodes,
        evaluation_horizon,
        frozen_contract,
    ) = _validate_configuration(
        configuration,
        identity=identity,
        registration=registration,
        provenance=manifest["provenance"],
        field=f"{field}.configuration",
    )
    parameter_match = _validate_parameter_match(artifact, field=field)
    if frozen_contract["parameter_match_frozen"]:
        for name in _PARAMETER_MATCH_KEYS:
            if artifact[name] != configuration[name]:
                raise DevelopmentAggregationError(
                    f"{field}.{name} does not match the frozen configuration"
                )
    expected_source = expected_environment_source(environment)
    artifact_source_frozen = "environment_source" in artifact
    if artifact_source_frozen:
        artifact_source = _mapping(
            artifact["environment_source"],
            field=f"{field}.environment_source",
        )
        if dict(artifact_source) != expected_source:
            raise DevelopmentAggregationError(
                f"{field}.environment_source does not match the registered source invocation"
            )
        if frozen_contract["environment_source_frozen"] and (
            artifact["environment_source"] != configuration["environment_source"]
        ):
            raise DevelopmentAggregationError(
                f"{field}.environment_source does not match the frozen configuration"
            )
    elif environment in UPPER_REFERENCE_ENVIRONMENTS:
        raise DevelopmentAggregationError(
            f"{field}.environment_source is required for upper-reference cells"
        )
    if artifact.get("environment_reference") != expected_environment_reference(environment):
        raise DevelopmentAggregationError(
            f"{field}.environment_reference does not match the registered reference class"
        )
    actual_steps = _integer(
        artifact["actual_environment_steps"],
        field=f"{field}.actual_environment_steps",
        positive=True,
    )
    if actual_steps != total_steps:
        raise DevelopmentAggregationError(
            f"{field}.actual_environment_steps does not match frozen total_steps"
        )
    if registration["schema_version"] in {2, 3}:
        for name in (
            "comparison_profile",
            "requested_environment_steps",
            "realized_environment_steps",
        ):
            if name not in artifact:
                raise DevelopmentAggregationError(f"{field} is missing {name}")
        requested_steps = _integer(
            artifact["requested_environment_steps"],
            field=f"{field}.requested_environment_steps",
            positive=True,
        )
        realized_steps = _integer(
            artifact["realized_environment_steps"],
            field=f"{field}.realized_environment_steps",
            positive=True,
        )
        if (
            artifact["comparison_profile"] != registration["comparison_profile"]
            or requested_steps != registration["budgets"][environment]
            or realized_steps != total_steps
            or actual_steps != realized_steps
        ):
            raise DevelopmentAggregationError(
                f"{field} comparison profile or step accounting is inconsistent"
            )
    if _mapping(artifact["ppo"], field=f"{field}.ppo") != configuration["ppo"]:
        raise DevelopmentAggregationError(f"{field}.ppo drifts from frozen configuration")
    artifact_evaluation_episodes = _integer(
        artifact["evaluation_episodes_per_environment"],
        field=f"{field}.evaluation_episodes_per_environment",
        positive=True,
    )
    artifact_evaluation_horizon = _integer(
        artifact["evaluation_max_episode_steps"],
        field=f"{field}.evaluation_max_episode_steps",
        positive=True,
    )
    expected_evaluation_steps = evaluation_episodes * evaluation_horizon
    actual_evaluation_steps = _integer(
        artifact["actual_evaluation_steps_per_environment"],
        field=f"{field}.actual_evaluation_steps_per_environment",
        positive=True,
    )
    if (
        artifact_evaluation_episodes != evaluation_episodes
        or artifact_evaluation_horizon != evaluation_horizon
        or actual_evaluation_steps != expected_evaluation_steps
    ):
        raise DevelopmentAggregationError(
            f"{field} evaluation contract drifts from frozen configuration"
        )
    num_environments = _integer(
        configuration["ppo"].get("num_envs"),
        field=f"{field}.configuration.ppo.num_envs",
        positive=True,
    )
    evaluation = _validate_evaluation(
        artifact,
        field=field,
        expected_episodes=evaluation_episodes,
        expected_horizon=evaluation_horizon,
        expected_num_environments=num_environments,
    )
    expected_transitions = expected_evaluation_steps * num_environments
    if (
        _integer(
            artifact["actual_evaluation_transitions"],
            field=f"{field}.actual_evaluation_transitions",
            positive=True,
        )
        != expected_transitions
    ):
        raise DevelopmentAggregationError(f"{field}.actual_evaluation_transitions is inconsistent")
    steps, curve_returns = _validate_history(
        artifact,
        field=field,
        expected_final_steps=total_steps,
    )
    _validate_final_training_metrics(artifact, field=field)
    return {
        "evaluation": evaluation,
        "steps": steps,
        "curve_returns": curve_returns,
        "parameter_match": parameter_match,
        "environment_source_frozen": (
            frozen_contract["environment_source_frozen"] and artifact_source_frozen
        ),
        "parameter_match_frozen": frozen_contract["parameter_match_frozen"],
        "model_family": (
            registration["candidate_specs"][model]["model_family"]
            if registration["schema_version"] == 3
            else None
        ),
        "implementation_model": (
            registration["candidate_specs"][model]["implementation_model"]
            if registration["schema_version"] == 3
            else model
        ),
    }


def interquartile_mean(values: Sequence[float]) -> float:
    """Return the fractional-boundary IQM for a finite non-empty sample."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise DevelopmentAggregationError("IQM requires a non-empty finite one-dimensional sample")
    lower = np.arange(array.size, dtype=np.float64) / array.size
    upper = np.arange(1, array.size + 1, dtype=np.float64) / array.size
    weights = np.maximum(0.0, np.minimum(upper, 0.75) - np.maximum(lower, 0.25)) / 0.5
    return float(np.dot(np.sort(array), weights))


def _bootstrap_seed(stream: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}\0{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _statistic(values: np.ndarray, statistic: str) -> float:
    if statistic == "mean":
        return float(np.mean(values))
    if statistic == "median":
        return float(np.median(values))
    if statistic == "iqm":
        return interquartile_mean(values)
    raise AssertionError(f"unknown statistic: {statistic}")


def _bootstrap_interval(values: Sequence[float], *, statistic: str, stream: str) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(_bootstrap_seed(stream))
    indices = rng.integers(0, array.size, size=(BOOTSTRAP_RESAMPLES, array.size), endpoint=False)
    samples = array[indices]
    if statistic == "mean":
        estimates = np.mean(samples, axis=1)
    elif statistic == "median":
        estimates = np.median(samples, axis=1)
    elif statistic == "iqm":
        lower = np.arange(array.size, dtype=np.float64) / array.size
        upper = np.arange(1, array.size + 1, dtype=np.float64) / array.size
        weights = np.maximum(0.0, np.minimum(upper, 0.75) - np.maximum(lower, 0.25)) / 0.5
        estimates = np.sort(samples, axis=1) @ weights
    else:
        raise AssertionError(f"unknown statistic: {statistic}")
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    bounds = np.quantile(estimates, [alpha, 1.0 - alpha], method="linear")
    return [float(bounds[0]), float(bounds[1])]


def _summary(values: Sequence[float], *, stream: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        statistic: {
            "estimate": _statistic(array, statistic),
            "bootstrap_95_ci": _bootstrap_interval(
                array, statistic=statistic, stream=f"{stream}:{statistic}"
            ),
        }
        for statistic in ("mean", "median", "iqm")
    }


def _trapezoid_by_step(values: Sequence[float], steps: Sequence[int]) -> float:
    if len(values) != len(steps) or len(values) < 2:
        raise DevelopmentAggregationError(
            "learning-curve AUC requires at least two aligned observations"
        )
    y = np.asarray(values, dtype=np.float64)
    x = np.asarray(steps, dtype=np.float64)
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
        raise DevelopmentAggregationError("learning-curve AUC requires finite observations")
    if np.any(x[1:] <= x[:-1]):
        raise DevelopmentAggregationError(
            "learning-curve AUC requires strictly increasing environment steps"
        )
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


def _validate_completion_index(
    root: Path,
    *,
    manifest_sha256: str,
    manifest_schema_version: int,
    cells: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> tuple[bool, tuple[Path, ...]]:
    path = root / "completion_index.json"
    if not path.exists():
        return False, ()
    index = _mapping(_load_json(path, field="completion index"), field="completion_index")
    _exact_keys(
        index,
        {
            "schema_version",
            "status",
            "manifest_sha256",
            "planned_cells",
            "completed_cells",
            "cells",
        },
        field="completion_index",
    )
    if index["schema_version"] != 1 or index["status"] != "complete":
        raise DevelopmentAggregationError("completion_index is not complete schema version 1")
    if manifest_schema_version == 3 and path.read_bytes() != canonical_json_bytes(index) + b"\n":
        raise DevelopmentAggregationError("completion_index is not canonical JSON")
    if index["manifest_sha256"] != manifest_sha256:
        raise DevelopmentAggregationError("completion_index belongs to another manifest")
    expected_count = len(cells)
    if (
        index["planned_cells"] != expected_count
        or index["completed_cells"] != expected_count
        or not isinstance(index["cells"], list)
        or len(index["cells"]) != expected_count
    ):
        raise DevelopmentAggregationError("completion_index has an incomplete cell set")
    indexed: set[tuple[str, str, int]] = set()
    tuning_log_paths: set[Path] = set()
    for item_index, raw_item in enumerate(index["cells"]):
        field = f"completion_index.cells[{item_index}]"
        item = _mapping(raw_item, field=field)
        cell_keys = _CELL_KEYS_V3 if manifest_schema_version == 3 else _CELL_KEYS
        artifact_only_fields = cell_keys | {"artifact_sha256"}
        artifact_and_log_fields = artifact_only_fields | {"log_path", "log_sha256"}
        supported_fields = {
            frozenset(artifact_only_fields),
            frozenset(artifact_and_log_fields),
        }
        if manifest_schema_version == 3:
            supported_fields = {frozenset(artifact_and_log_fields)}
        if frozenset(item) not in supported_fields:
            raise DevelopmentAggregationError(
                f"{field} has wrong fields for a supported completion-index "
                "schema; schema v3 requires log_path and log_sha256"
            )
        identity = (
            _string(item["environment"], field=f"{field}.environment"),
            _string(item["model"], field=f"{field}.model"),
            _integer(item["seed"], field=f"{field}.seed"),
        )
        if identity not in cells or identity in indexed:
            raise DevelopmentAggregationError(f"{field} has an invalid cell identity")
        indexed.add(identity)
        expected = cells[identity]
        if (
            item["cell_id"] != expected["cell_id"]
            or item["configuration_sha256"] != expected["configuration_sha256"]
            or item["artifact_path"] != expected["artifact_relative_path"]
        ):
            raise DevelopmentAggregationError(f"{field} drifts from frozen manifest")
        if manifest_schema_version == 3 and (
            item["model_family"] != expected["model_family"]
            or item["implementation_model"] != expected["implementation_model"]
            or item["implementation_source_sha256"] != expected["implementation_source_sha256"]
        ):
            raise DevelopmentAggregationError(
                f"{field} candidate identity drifts from frozen manifest"
            )
        artifact_hash = _sha256(item["artifact_sha256"], field=f"{field}.artifact_sha256")
        if sha256_file(expected["artifact_path"]) != artifact_hash:
            raise DevelopmentAggregationError(f"{field}.artifact_sha256 is incorrect")
        if "log_path" in item:
            log_path = _artifact_path(root, item["log_path"], field=f"{field}.log_path")
            if manifest_schema_version == 3:
                expected_log_path = expected["artifact_path"].with_suffix(".log")
                if log_path != expected_log_path or log_path in tuning_log_paths:
                    raise DevelopmentAggregationError(
                        f"{field}.log_path does not match its immutable cell log identity"
                    )
                tuning_log_paths.add(log_path)
            log_hash = _sha256(item["log_sha256"], field=f"{field}.log_sha256")
            if sha256_file(log_path) != log_hash:
                raise DevelopmentAggregationError(f"{field}.log_sha256 is incorrect")
    if len(indexed) != expected_count:
        raise DevelopmentAggregationError("completion_index is missing cells")
    return True, tuple(sorted(tuning_log_paths, key=lambda item: item.as_posix().encode("utf-8")))


def _validate_checksums(
    root: Path,
    *,
    required_paths: Sequence[Path],
    require_exact_inventory: bool = False,
) -> bool:
    checksum_path = root / "checksums.sha256"
    if not checksum_path.exists():
        return False
    try:
        entries = validate_checksum_manifest(root)
    except ArtifactChecksumError as error:
        raise DevelopmentAggregationError(str(error)) from error
    indexed = {root.joinpath(*PurePosixPath(relative).parts).resolve() for relative, _ in entries}
    required = {path.resolve() for path in required_paths}
    missing = sorted(required - indexed)
    if missing:
        raise DevelopmentAggregationError(f"checksum manifest omits required inputs: {missing}")
    if require_exact_inventory:
        extra = sorted(indexed - required)
        if extra:
            raise DevelopmentAggregationError(
                "checksum inventory contains noncanonical raw inputs: "
                f"{[path.relative_to(root).as_posix() for path in extra]}"
            )
    return True


def build_development_aggregate(output_root: str | Path) -> dict[str, Any]:
    """Validate and aggregate one complete frozen development matrix."""

    root = Path(output_root).resolve()
    if not root.is_dir():
        raise DevelopmentAggregationError(f"output root does not exist: {root}")
    registration_path = root / "registration.json"
    manifest_path = root / "frozen_manifest.json"
    raw_registration = _load_json(registration_path, field="registration")
    registration = _validate_registration(raw_registration)
    registration_sha256 = canonical_json_sha256(raw_registration)
    raw_manifest = _load_json(manifest_path, field="frozen manifest")
    manifest, cells = _validate_manifest(raw_manifest, root=root, registration=registration)
    if (
        registration["require_gpu"]
        and manifest["provenance"]["runtime_contract"]["jax_backend"] != "gpu"
    ):
        raise DevelopmentAggregationError(
            "registration requires GPU but validated artifacts used another backend"
        )
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    step_grids: dict[str, tuple[int, ...]] = {}
    for environment in registration["environments"]:
        for model in registration["models"]:
            for seed in registration["seeds"]:
                identity = (environment, model, seed)
                record = _validate_artifact(
                    cells[identity]["artifact_path"],
                    identity=identity,
                    expected=cells[identity],
                    registration=registration,
                    manifest=manifest,
                )
                if environment not in step_grids:
                    step_grids[environment] = record["steps"]
                elif step_grids[environment] != record["steps"]:
                    raise DevelopmentAggregationError(
                        f"training step grids differ within environment {environment!r}"
                    )
                records[identity] = record

    completion_validated, tuning_log_paths = _validate_completion_index(
        root,
        manifest_sha256=manifest["manifest_sha256"],
        manifest_schema_version=manifest["schema_version"],
        cells=cells,
    )
    required_checksum_paths = [
        registration_path.resolve(),
        manifest_path.resolve(),
        *(cell["artifact_path"] for cell in cells.values()),
    ]
    if completion_validated:
        required_checksum_paths.append((root / "completion_index.json").resolve())
        required_checksum_paths.extend(tuning_log_paths)
    checksums_validated = _validate_checksums(
        root,
        required_paths=required_checksum_paths,
        require_exact_inventory=manifest["schema_version"] == 3,
    )
    if registration["tier"] == "development_tuning":
        if not completion_validated or not checksums_validated:
            raise DevelopmentAggregationError(
                "development_tuning requires validated completion and checksum indexes"
            )
        if not all(
            record["environment_source_frozen"] and record["parameter_match_frozen"]
            for record in records.values()
        ):
            raise DevelopmentAggregationError(
                "development_tuning requires frozen environment source and "
                "parameter-match contracts in every cell"
            )

    curve_start_by_environment: dict[str, int] = {}
    environment_contracts: list[dict[str, Any]] = []
    if registration["tier"] == "development_tuning":
        for environment in registration["environments"]:
            first_finite_indices: list[int] = []
            for model in registration["models"]:
                for seed in registration["seeds"]:
                    identity = (environment, model, seed)
                    first_finite = next(
                        (
                            index
                            for index, value in enumerate(records[identity]["curve_returns"])
                            if value is not None
                        ),
                        None,
                    )
                    if first_finite is None:
                        raise DevelopmentAggregationError(
                            "development_tuning has no shared finite learning-curve "
                            f"suffix for environment {environment!r}: "
                            f"cell {identity!r} has no finite return"
                        )
                    first_finite_indices.append(first_finite)
            curve_start_index = max(first_finite_indices)
            full_grid = step_grids[environment]
            retained_grid = full_grid[curve_start_index:]
            if len(retained_grid) < 2:
                raise DevelopmentAggregationError(
                    "development_tuning requires at least two shared finite "
                    f"learning-curve points for environment {environment!r}"
                )
            curve_start_by_environment[environment] = curve_start_index
            environment_contracts.append(
                {
                    "environment": environment,
                    "model_family_count": len(registration["candidate_families"]),
                    "candidate_count_per_family": len(
                        registration["candidate_families"][0]["candidates"]
                    ),
                    "total_candidate_count": len(registration["models"]),
                    "seed_count_per_candidate": len(registration["seeds"]),
                    "candidate_seed_cardinality_equal": True,
                    "candidate_cardinality_equal_across_families": True,
                    "training_curve": {
                        "full_environment_step_grid": list(full_grid),
                        "curve_start_step": retained_grid[0],
                        "curve_end_step": retained_grid[-1],
                        "excluded_prefix_length": curve_start_index,
                        "retained_environment_step_grid": list(retained_grid),
                        "integration_width_environment_steps": (
                            retained_grid[-1] - retained_grid[0]
                        ),
                    },
                }
            )

    groups: list[dict[str, Any]] = []
    for environment in registration["environments"]:
        for model in registration["models"]:
            raw_seed_values = []
            seed_means = []
            for seed in registration["seeds"]:
                record = records[(environment, model, seed)]
                evaluation = record["evaluation"]
                seed_means.append(evaluation["mean_return"])
                raw_seed_values.append(
                    {
                        "seed": seed,
                        "mean_return": evaluation["mean_return"],
                        "median_return": evaluation["median_return"],
                        "returns_by_environment": evaluation["returns_by_environment"],
                        "parameter_match": record["parameter_match"],
                    }
                )
            group = {
                "environment": environment,
                "model": model,
                "seeds": list(registration["seeds"]),
                "raw_seed_values": raw_seed_values,
                "final_seed_mean_return": _summary(
                    seed_means, stream=f"development:{environment}:{model}"
                ),
            }
            if registration["tier"] == "development_tuning":
                candidate = registration["candidate_specs"][model]
                group.update(
                    {
                        "candidate_id": model,
                        "model_family": candidate["model_family"],
                        "implementation_model": candidate["implementation_model"],
                        "learner": candidate["learner"],
                        "implementation_source_sha256": manifest["provenance"][
                            "implementation_source"
                        ]["sha256"],
                    }
                )
                curve_start_index = curve_start_by_environment[environment]
                retained_steps = step_grids[environment][curve_start_index:]
                width = retained_steps[-1] - retained_steps[0]
                curve_rows = [
                    records[(environment, model, seed)]["curve_returns"][curve_start_index:]
                    for seed in registration["seeds"]
                ]
                curve_matrix = np.asarray(curve_rows, dtype=np.float64)
                auc_values = [_trapezoid_by_step(row, retained_steps) for row in curve_matrix]
                mean_return_values = [value / width for value in auc_values]
                group["training_curve"] = {
                    "environment_steps": list(retained_steps),
                    "raw_seed_returns": [
                        {
                            "seed": seed,
                            "mean_recent_return": [float(value) for value in row],
                            "auc_return_step": auc,
                            "auc_mean_return": auc_mean_return,
                        }
                        for seed, row, auc, auc_mean_return in zip(
                            registration["seeds"],
                            curve_matrix,
                            auc_values,
                            mean_return_values,
                        )
                    ],
                    "mean_return_by_step": [
                        float(value) for value in np.mean(curve_matrix, axis=0)
                    ],
                    "median_return_by_step": [
                        float(value) for value in np.median(curve_matrix, axis=0)
                    ],
                    "iqm_return_by_step": [
                        interquartile_mean(curve_matrix[:, index])
                        for index in range(curve_matrix.shape[1])
                    ],
                    "auc_return_step": _summary(
                        auc_values,
                        stream=f"development-tuning-auc:{environment}:{model}",
                    ),
                    "auc_mean_return": _summary(
                        mean_return_values,
                        stream=f"development-tuning-auc-mean:{environment}:{model}",
                    ),
                }
            groups.append(group)

    candidate_selection: list[dict[str, Any]] = []
    if registration["tier"] == "development_tuning":
        for environment in registration["environments"]:
            for family in registration["candidate_families"]:
                family_candidate_ids = {
                    candidate["candidate_id"] for candidate in family["candidates"]
                }
                candidates = []
                for group in groups:
                    if (
                        group["environment"] != environment
                        or group["candidate_id"] not in family_candidate_ids
                    ):
                        continue
                    score = group["training_curve"]["auc_mean_return"]["mean"]["estimate"]
                    candidates.append(
                        {
                            "candidate_id": group["candidate_id"],
                            "selection_score": score,
                        }
                    )
                ranked = sorted(
                    candidates,
                    key=lambda candidate: (
                        -candidate["selection_score"],
                        candidate["candidate_id"],
                    ),
                )
                candidate_selection.append(
                    {
                        "environment": environment,
                        "model_family": family["family_id"],
                        "implementation_model": family["implementation_model"],
                        "metric": "mean_seed_auc_mean_return",
                        "direction": "higher_is_better",
                        "tie_breaker": "ascending_candidate_id",
                        "winner_candidate_id": ranked[0]["candidate_id"],
                        "ranking": [
                            {"rank": rank, **candidate}
                            for rank, candidate in enumerate(ranked, start=1)
                        ],
                    }
                )

    paired: list[dict[str, Any]] = []
    if "arcmind" in registration["models"]:
        for environment in registration["environments"]:
            for model in registration["models"]:
                if model == "arcmind":
                    continue
                raw_differences = []
                differences = []
                for seed in registration["seeds"]:
                    model_return = records[(environment, model, seed)]["evaluation"]["mean_return"]
                    arcmind_return = records[(environment, "arcmind", seed)]["evaluation"][
                        "mean_return"
                    ]
                    difference = model_return - arcmind_return
                    differences.append(difference)
                    raw_differences.append(
                        {
                            "seed": seed,
                            "model_mean_return": model_return,
                            "arcmind_mean_return": arcmind_return,
                            "difference": difference,
                        }
                    )
                paired.append(
                    {
                        "environment": environment,
                        "model": model,
                        "reference_model": "arcmind",
                        "seeds": list(registration["seeds"]),
                        "raw_seed_differences": raw_differences,
                        "difference_summary": _summary(
                            differences,
                            stream=f"development-paired:{environment}:{model}:arcmind",
                        ),
                    }
                )

    result = {
        "schema_version": 1,
        "status": _AGGREGATE_STATUS[registration["tier"]],
        "evidence_tier": registration["tier"],
        "registration_sha256": registration_sha256,
        "matrix_manifest_sha256": manifest["manifest_sha256"],
        "matrix_kind": registration["matrix_kind"],
        "provenance": manifest["provenance"],
        "models": list(registration["models"]),
        "environments": list(registration["environments"]),
        "seeds": list(registration["seeds"]),
        "environment_budgets": registration["budgets"],
        "statistical_unit": "seed",
        "not_for_paper": True,
        "integrity_indexes": {
            "completion_index_present_and_validated": completion_validated,
            "checksums_present_and_validated": checksums_validated,
        },
        "frozen_semantic_contract": {
            "environment_source_in_every_configuration": all(
                record["environment_source_frozen"] for record in records.values()
            ),
            "parameter_match_in_every_configuration": all(
                record["parameter_match_frozen"] for record in records.values()
            ),
            "artifact_parameter_match_validated": True,
        },
        "statistics": {
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence_level": CONFIDENCE_LEVEL,
            "bootstrap_interval": "percentile",
            "bootstrap_unit": "seed",
            "iqm": (
                "Fractional-boundary interquartile mean over per-seed mean evaluation returns."
            ),
        },
        "groups": groups,
        "paired_differences_against_arcmind": paired,
    }
    if registration["tier"] == "development_tuning":
        result.update(
            {
                "completion_index_sha256": sha256_file(root / "completion_index.json"),
                "checksum_manifest_sha256": sha256_file(root / "checksums.sha256"),
                "candidate_families": [
                    {
                        "family_id": family["family_id"],
                        "implementation_model": family["implementation_model"],
                        "candidate_ids": [
                            candidate["candidate_id"] for candidate in family["candidates"]
                        ],
                    }
                    for family in registration["candidate_families"]
                ],
                "selection_eligibility": {
                    "eligible_for_hyperparameter_selection": True,
                    "eligible_for_architecture_selection": False,
                    "eligible_for_checkpoint_selection": False,
                    "eligible_for_registered_final_evidence": False,
                    "eligible_for_paper_performance_claims": False,
                    "selection_scope": "candidate_within_model_family_and_environment",
                    "selection_metric": "mean_seed_auc_mean_return",
                },
                "curve_integration": (
                    "Trapezoidal mean recent return by environment step over the "
                    "shared complete-case suffix, with no extrapolation."
                ),
                "environment_contracts": environment_contracts,
                "candidate_selection": candidate_selection,
            }
        )
    return result


def aggregate_development(output_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Build a development aggregate and atomically create canonical JSON."""

    root = Path(output_root).resolve()
    output = Path(output_path).resolve()
    if output == root or output.is_relative_to(root):
        raise DevelopmentAggregationError("aggregate output must be outside the raw matrix root")
    aggregate = build_development_aggregate(root)
    atomic_write_json(output, aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    result = aggregate_development(arguments.output_root, arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "status": result["status"],
                "groups": len(result["groups"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONFIDENCE_LEVEL",
    "DevelopmentAggregationError",
    "aggregate_development",
    "build_development_aggregate",
    "interquartile_mean",
]
