"""Link one primary POBAX matrix to its paired upper-reference matrix.

The link is a derived artifact. It must be written outside both immutable raw
matrix roots. Every raw artifact, log, completion record, and checksum is
validated before the link is created.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from benchmarks.pobax.aggregate_development import build_development_aggregate
from benchmarks.pobax.aggregate_registered import (
    RegisteredAggregationError,
    build_registered_aggregate,
    validate_bound_compute_aware_primary_matrix,
)
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
    COMPUTE_AWARE_FINAL_EVALUATION_EPISODES_PER_ENV,
    COMPUTE_AWARE_FINAL_PANEL,
    COMPUTE_AWARE_FINAL_SEEDS,
    COMPUTE_AWARE_UPPER_REFERENCE_PANEL,
    normalize_final_selection_binding,
    normalize_learner,
    normalize_memoryless_learner_binding,
    normalize_primary_matrix_binding,
    realized_environment_steps,
    registration_fields,
    validate_comparison_profile,
)
from benchmarks.pobax.upper_reference_registry import UPPER_TO_PRIMARY_ENVIRONMENT

REGISTERED_TRAIN_STEPS = {
    "tmaze_10": 1_000_000,
    "tmaze_10-perfect-memory": 1_000_000,
    "rocksample_11_11": 5_000_000,
    "rocksample_11_11-fully-observable": 5_000_000,
    "battleship_10": 10_000_000,
    "battleship_10-perfect-recall": 10_000_000,
    "Walker-V-v0": 50_000_000,
    "Walker-F-v0": 50_000_000,
    "HalfCheetah-V-v0": 50_000_000,
    "HalfCheetah-F-v0": 50_000_000,
    "Navix-DMLab-Maze-01-v0": 10_000_000,
    "Navix-DMLab-Maze-01-fully-observable": 10_000_000,
}

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
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
_MANIFEST_KEYS_V4 = _MANIFEST_KEYS | {"registration_sha256", "tuning_selection"}
_CELL_KEYS = {
    "cell_id",
    "environment",
    "model",
    "seed",
    "configuration_sha256",
    "artifact_path",
}
_CELL_KEYS_V4 = _CELL_KEYS | {
    "candidate_id",
    "model_family",
    "implementation_model",
    "tuning_aggregate_sha256",
    "tuning_completion_index_sha256",
    "tuning_checksum_manifest_sha256",
    "tuning_implementation_source_sha256",
    "implementation_source_sha256",
}
_COMPLETION_KEYS = {
    "schema_version",
    "status",
    "manifest_sha256",
    "planned_cells",
    "completed_cells",
    "cells",
}
_COMPLETED_CELL_KEYS = _CELL_KEYS | {
    "artifact_sha256",
    "log_path",
    "log_sha256",
}
_COMPLETED_CELL_KEYS_V4 = _CELL_KEYS_V4 | {
    "artifact_sha256",
    "log_path",
    "log_sha256",
}
_SOURCE_IDENTITY_KEYS = (
    "dependency_lock_sha256",
    "pobax_commit",
    "navix_commit",
)


class UpperReferenceLinkError(ValueError):
    """Raised when two raw matrices cannot be paired safely."""


def _reject_json_constant(value: str) -> None:
    raise UpperReferenceLinkError(f"non-finite JSON constant is prohibited: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise UpperReferenceLinkError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, *, field: str) -> Any:
    if not path.is_file():
        raise UpperReferenceLinkError(f"{field} does not exist or is not a file: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UpperReferenceLinkError(f"cannot read {field} {path}: {error}") from error


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise UpperReferenceLinkError(f"{field} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    actual = set(value)
    if actual != expected:
        raise UpperReferenceLinkError(
            f"{field} has wrong fields: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise UpperReferenceLinkError(f"{field} must be a lowercase SHA256")
    return value


def _cell_identity(
    value: Mapping[str, Any],
    *,
    field: str,
) -> tuple[str, str, int]:
    environment = value.get("environment")
    model = value.get("model")
    seed = value.get("seed")
    if not isinstance(environment, str) or not environment:
        raise UpperReferenceLinkError(f"{field}.environment must be a non-empty string")
    if not isinstance(model, str) or not model:
        raise UpperReferenceLinkError(f"{field}.model must be a non-empty string")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise UpperReferenceLinkError(f"{field}.seed must be a non-negative integer")
    return environment, model, seed


def _safe_relative_path(root: Path, value: Any, *, field: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise UpperReferenceLinkError(f"{field} must be a non-empty POSIX path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise UpperReferenceLinkError(f"{field} must be a normalized relative POSIX path")
    normalized = pure.as_posix()
    resolved = root.joinpath(*pure.parts).resolve()
    if not resolved.is_relative_to(root):
        raise UpperReferenceLinkError(f"{field} escapes its raw matrix root")
    return normalized, resolved


def _validate_registration(root: Path) -> tuple[dict[str, Any], str]:
    path = root / "registration.json"
    raw = _mapping(_load_json(path, field="registration"), field="registration")
    schema_version = raw.get("schema_version")
    try:
        expected_fields = registration_fields(schema_version)
    except ValueError as error:
        raise UpperReferenceLinkError(str(error)) from error
    _exact_keys(raw, expected_fields, field="registration")
    if raw["status"] != "frozen":
        raise UpperReferenceLinkError("registration must be frozen")
    try:
        comparison_profile = validate_comparison_profile(raw)
    except ValueError as error:
        raise UpperReferenceLinkError(str(error)) from error
    tier = raw["evidence_tier"]
    if tier not in {"smoke", "pilot", "registered_final"}:
        raise UpperReferenceLinkError(f"registration has unsupported evidence tier: {tier!r}")
    matrix_kind = raw["matrix_kind"]
    if matrix_kind not in {"primary_comparison", "upper_reference"}:
        raise UpperReferenceLinkError(f"registration has unsupported matrix kind: {matrix_kind!r}")
    models = raw["models"]
    if (
        not isinstance(models, list)
        or not models
        or any(not isinstance(model, str) or not model for model in models)
        or len(set(models)) != len(models)
    ):
        raise UpperReferenceLinkError("registration.models must be unique non-empty names")
    if matrix_kind == "primary_comparison" and "arcmind" not in models:
        raise UpperReferenceLinkError("primary registration must contain arcmind")
    if matrix_kind == "upper_reference" and models != ["memoryless_mlp"]:
        raise UpperReferenceLinkError(
            "upper-reference registration must contain only memoryless_mlp"
        )
    environments = raw["environments"]
    if not isinstance(environments, list) or not environments:
        raise UpperReferenceLinkError("registration.environments must be a non-empty list")
    environment_ids: list[str] = []
    for index, environment_value in enumerate(environments):
        field = f"registration.environments[{index}]"
        environment = _mapping(environment_value, field=field)
        _exact_keys(environment, {"id", "total_steps"}, field=field)
        environment_id = environment["id"]
        total_steps = environment["total_steps"]
        if not isinstance(environment_id, str) or not environment_id:
            raise UpperReferenceLinkError(f"{field}.id must be a non-empty string")
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
            raise UpperReferenceLinkError(f"{field}.total_steps must be a positive integer")
        environment_ids.append(environment_id)
    if len(set(environment_ids)) != len(environment_ids):
        raise UpperReferenceLinkError("registration environment IDs contain duplicates")
    selected_aliases = set(environment_ids) & set(UPPER_TO_PRIMARY_ENVIRONMENT)
    if matrix_kind == "upper_reference":
        unknown_aliases = set(environment_ids) - set(UPPER_TO_PRIMARY_ENVIRONMENT)
        if unknown_aliases:
            raise UpperReferenceLinkError(
                f"upper-reference registration has unknown aliases: {sorted(unknown_aliases)}"
            )
    elif selected_aliases:
        raise UpperReferenceLinkError(
            f"primary registration contains upper-reference aliases: {sorted(selected_aliases)}"
        )
    seeds = raw["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise UpperReferenceLinkError("registration.seeds must be unique non-negative integers")
    if tier == "registered_final" and len(seeds) != 30:
        raise UpperReferenceLinkError("registered_final requires exactly 30 paired seeds")
    if tier == "registered_final" and matrix_kind == "primary_comparison" and schema_version != 4:
        raise UpperReferenceLinkError(
            "registered-final primary comparisons require schema version 4 "
            "and a frozen tuning selection"
        )
    try:
        if schema_version == 4:
            if (
                tier != "registered_final"
                or matrix_kind != "primary_comparison"
                or comparison_profile != "arcmind_shared_comparison"
            ):
                raise ValueError(
                    "schema-v4 registration requires a registered-final primary "
                    "comparison with arcmind_shared_comparison"
                )
            binding = normalize_final_selection_binding(raw["tuning_selection"])
            selections = list(binding["selections"])
            expected_selection_identities = [
                (environment, model) for environment in environment_ids for model in models
            ]
            actual_selection_identities = [
                (selection["environment"], selection["implementation_model"])
                for selection in selections
            ]
            if actual_selection_identities != expected_selection_identities:
                raise ValueError(
                    "schema-v4 tuning selections must exactly cover the ordered "
                    "environment and model product"
                )
            environment_budgets = {
                environment["id"]: environment["total_steps"] for environment in environments
            }
            for selection in selections:
                learner = selection["learner"]
                realized_environment_steps(
                    environment_budgets[selection["environment"]],
                    num_envs=int(learner["num_envs"]),
                    rollout_steps=int(learner["rollout_steps"]),
                    comparison_profile=comparison_profile,
                )
        else:
            learner = normalize_learner(
                raw["learner"],
                schema_version=schema_version,
            )
            for environment in environments:
                realized_environment_steps(
                    environment["total_steps"],
                    num_envs=int(learner["num_envs"]),
                    rollout_steps=int(learner["rollout_steps"]),
                    comparison_profile=comparison_profile,
                )
    except ValueError as error:
        raise UpperReferenceLinkError(str(error)) from error
    evaluation_episodes = raw["evaluation_episodes_per_env"]
    if (
        isinstance(evaluation_episodes, bool)
        or not isinstance(evaluation_episodes, int)
        or evaluation_episodes <= 0
    ):
        raise UpperReferenceLinkError(
            "registration.evaluation_episodes_per_env must be a positive integer"
        )
    if not isinstance(raw["require_gpu"], bool) or not isinstance(raw["quick"], bool):
        raise UpperReferenceLinkError("registration boolean fields must be booleans")
    if raw["quick"]:
        if schema_version == 4:
            raise UpperReferenceLinkError("schema-v4 registration cannot be quick")
        if tier != "smoke":
            raise UpperReferenceLinkError("quick registration is valid only for smoke")
        if any(environment["total_steps"] != 8_192 for environment in environments):
            raise UpperReferenceLinkError("quick registration must use 8192 total steps")
        expected_quick = {"num_envs": 32, "rollout_steps": 32, "update_epochs": 2}
        for key, value in expected_quick.items():
            if learner[key] != value:
                raise UpperReferenceLinkError(
                    f"quick registration learner.{key} must equal {value}"
                )
    return dict(raw), canonical_json_sha256(raw)


def _validate_manifest(
    root: Path,
    registration: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[tuple[str, str, int], dict[str, Any]]]:
    raw = _mapping(
        _load_json(root / "frozen_manifest.json", field="frozen manifest"),
        field="frozen_manifest",
    )
    schema_version = registration["schema_version"]
    _exact_keys(
        raw,
        _MANIFEST_KEYS_V4 if schema_version == 4 else _MANIFEST_KEYS,
        field="frozen_manifest",
    )
    if raw["schema_version"] != registration["schema_version"] or raw["status"] != "frozen":
        raise UpperReferenceLinkError("frozen_manifest schema must match its frozen registration")
    manifest_sha256 = _sha256(
        raw["manifest_sha256"],
        field="frozen_manifest.manifest_sha256",
    )
    hash_input = dict(raw)
    del hash_input["manifest_sha256"]
    if canonical_json_sha256(hash_input) != manifest_sha256:
        raise UpperReferenceLinkError(
            "frozen_manifest.manifest_sha256 does not match canonical content"
        )
    expected_identity = {
        "matrix_kind": registration["matrix_kind"],
        "models": registration["models"],
        "environments": [item["id"] for item in registration["environments"]],
        "seeds": registration["seeds"],
    }
    actual_identity = {key: raw[key] for key in expected_identity}
    if actual_identity != expected_identity:
        raise UpperReferenceLinkError("frozen_manifest identity drifts from registration")
    if schema_version == 4 and raw["tuning_selection"] != registration["tuning_selection"]:
        raise UpperReferenceLinkError("frozen_manifest tuning_selection drifts from registration")
    if schema_version == 4:
        registration_sha256 = _sha256(
            raw["registration_sha256"],
            field="frozen_manifest.registration_sha256",
        )
        if registration_sha256 != sha256_file(root / "registration.json"):
            raise UpperReferenceLinkError(
                "frozen_manifest registration SHA256 drifts from registration bytes"
            )

    models = registration["models"]
    environments = expected_identity["environments"]
    seeds = registration["seeds"]
    expected_cells = {
        (environment, model, seed)
        for environment in environments
        for model in models
        for seed in seeds
    }
    raw_cells = raw["cells"]
    if not isinstance(raw_cells, list) or len(raw_cells) != len(expected_cells):
        raise UpperReferenceLinkError("frozen_manifest has an incomplete cell list")
    cells: dict[tuple[str, str, int], dict[str, Any]] = {}
    seen_cell_ids: set[str] = set()
    seen_artifact_paths: set[str] = set()
    for index, raw_cell in enumerate(raw_cells):
        field = f"frozen_manifest.cells[{index}]"
        cell = _mapping(raw_cell, field=field)
        cell_keys = _CELL_KEYS_V4 if schema_version == 4 else _CELL_KEYS
        _exact_keys(cell, cell_keys, field=field)
        identity = _cell_identity(cell, field=field)
        if identity not in expected_cells or identity in cells:
            raise UpperReferenceLinkError(f"{field} has an invalid or duplicate identity")
        configuration_sha256 = _sha256(
            cell["configuration_sha256"],
            field=f"{field}.configuration_sha256",
        )
        cell_id = _sha256(cell["cell_id"], field=f"{field}.cell_id")
        if cell_id != registered_cell_id(*identity, configuration_sha256):
            raise UpperReferenceLinkError(f"{field}.cell_id does not match its identity")
        relative_path, artifact_path = _safe_relative_path(
            root,
            cell["artifact_path"],
            field=f"{field}.artifact_path",
        )
        if cell_id in seen_cell_ids or relative_path in seen_artifact_paths:
            raise UpperReferenceLinkError(f"{field} duplicates a cell ID or artifact path")
        seen_cell_ids.add(cell_id)
        seen_artifact_paths.add(relative_path)
        if schema_version == 4:
            selection = next(
                (
                    item
                    for item in registration["tuning_selection"]["selections"]
                    if item["environment"] == identity[0]
                    and item["implementation_model"] == identity[1]
                ),
                None,
            )
            if selection is None or (
                cell["candidate_id"] != selection["candidate_id"]
                or cell["model_family"] != selection["model_family"]
                or cell["implementation_model"] != identity[1]
                or cell["tuning_aggregate_sha256"]
                != registration["tuning_selection"]["aggregate_sha256"]
                or cell["tuning_completion_index_sha256"]
                != registration["tuning_selection"]["source_completion_index_sha256"]
                or cell["tuning_checksum_manifest_sha256"]
                != registration["tuning_selection"]["source_checksum_manifest_sha256"]
                or cell["tuning_implementation_source_sha256"]
                != registration["tuning_selection"]["source_implementation_sha256"]
                or cell["implementation_source_sha256"]
                != registration["tuning_selection"]["source_implementation_sha256"]
            ):
                raise UpperReferenceLinkError(
                    f"{field} tuning-selection identity drifts from registration"
                )
        cells[identity] = {
            **dict(cell),
            "artifact_path": relative_path,
            "resolved_artifact_path": artifact_path,
        }
    if set(cells) != expected_cells:
        raise UpperReferenceLinkError("frozen_manifest is missing Cartesian matrix cells")
    return dict(raw), cells


def _validate_completion_and_checksums(
    root: Path,
    manifest: Mapping[str, Any],
    cells: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> None:
    raw_index = _mapping(
        _load_json(root / "completion_index.json", field="completion index"),
        field="completion_index",
    )
    _exact_keys(raw_index, _COMPLETION_KEYS, field="completion_index")
    expected_count = len(cells)
    if (
        raw_index["schema_version"] != 1
        or raw_index["status"] != "complete"
        or raw_index["manifest_sha256"] != manifest["manifest_sha256"]
        or raw_index["planned_cells"] != expected_count
        or raw_index["completed_cells"] != expected_count
        or not isinstance(raw_index["cells"], list)
        or len(raw_index["cells"]) != expected_count
    ):
        raise UpperReferenceLinkError("completion_index is incomplete or belongs to another matrix")
    completion_path = root / "completion_index.json"
    if completion_path.read_bytes() != canonical_json_bytes(raw_index) + b"\n":
        raise UpperReferenceLinkError("completion_index is not canonical JSON")

    indexed: set[tuple[str, str, int]] = set()
    expected_checksum_paths = {
        "registration.json",
        "frozen_manifest.json",
        "completion_index.json",
    }
    for index, raw_completed in enumerate(raw_index["cells"]):
        field = f"completion_index.cells[{index}]"
        completed = _mapping(raw_completed, field=field)
        schema_version = manifest["schema_version"]
        completed_keys = _COMPLETED_CELL_KEYS_V4 if schema_version == 4 else _COMPLETED_CELL_KEYS
        frozen_cell_keys = _CELL_KEYS_V4 if schema_version == 4 else _CELL_KEYS
        _exact_keys(completed, completed_keys, field=field)
        identity = _cell_identity(completed, field=field)
        if identity not in cells or identity in indexed:
            raise UpperReferenceLinkError(f"{field} has an invalid or duplicate identity")
        indexed.add(identity)
        frozen = cells[identity]
        for key in frozen_cell_keys:
            if key == "artifact_path":
                expected = frozen["artifact_path"]
            else:
                expected = frozen[key]
            if completed[key] != expected:
                raise UpperReferenceLinkError(f"{field}.{key} drifts from frozen_manifest")
        artifact_sha256 = _sha256(
            completed["artifact_sha256"],
            field=f"{field}.artifact_sha256",
        )
        if sha256_file(frozen["resolved_artifact_path"]) != artifact_sha256:
            raise UpperReferenceLinkError(f"{field}.artifact_sha256 is incorrect")
        log_relative, log_path = _safe_relative_path(
            root,
            completed["log_path"],
            field=f"{field}.log_path",
        )
        log_sha256 = _sha256(completed["log_sha256"], field=f"{field}.log_sha256")
        if sha256_file(log_path) != log_sha256:
            raise UpperReferenceLinkError(f"{field}.log_sha256 is incorrect")
        expected_checksum_paths.add(frozen["artifact_path"])
        expected_checksum_paths.add(log_relative)
    if indexed != set(cells):
        raise UpperReferenceLinkError("completion_index is missing frozen cells")

    try:
        checksum_entries = validate_checksum_manifest(root)
    except ArtifactChecksumError as error:
        raise UpperReferenceLinkError(str(error)) from error
    parsed_paths = {relative for relative, _ in checksum_entries}
    if parsed_paths != expected_checksum_paths:
        raise UpperReferenceLinkError(
            "checksum manifest must exactly cover canonical evidence: "
            f"missing={sorted(expected_checksum_paths - parsed_paths)}, "
            f"extra={sorted(parsed_paths - expected_checksum_paths)}"
        )


def _deep_validate(
    root: Path,
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        if registration["evidence_tier"] in {"smoke", "pilot"}:
            aggregate = build_development_aggregate(root)
            integrity = aggregate["integrity_indexes"]
            if not all(integrity.values()):
                raise UpperReferenceLinkError(
                    "development aggregate did not validate completion and checksums"
                )
            return aggregate
        if registration["evidence_tier"] == "registered_final":
            return build_registered_aggregate(root / "frozen_manifest.json")
    except UpperReferenceLinkError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise UpperReferenceLinkError(f"raw matrix validation failed: {error}") from error
    raise UpperReferenceLinkError(f"unsupported evidence tier: {registration['evidence_tier']!r}")


def _task_contracts(
    cells: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for identity, cell in cells.items():
        artifact = _mapping(
            _load_json(cell["resolved_artifact_path"], field=f"artifact{identity!r}"),
            field=f"artifact{identity!r}",
        )
        configuration = _mapping(
            artifact.get("configuration"),
            field=f"artifact{identity!r}.configuration",
        )
        ppo = _mapping(configuration.get("ppo"), field=f"artifact{identity!r}.configuration.ppo")
        contract = {
            "ppo": dict(ppo),
            "evaluation_episodes_per_environment": configuration.get(
                "evaluation_episodes_per_environment"
            ),
            "evaluation_max_episode_steps": configuration.get("evaluation_max_episode_steps"),
        }
        environment = identity[0]
        if environment in contracts and contracts[environment] != contract:
            raise UpperReferenceLinkError(
                f"learner or evaluation contract differs within environment {environment!r}"
            )
        contracts[environment] = contract
    return contracts


def _task_contracts_by_model(
    cells: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    contracts: dict[tuple[str, str], dict[str, Any]] = {}
    for identity, cell in cells.items():
        artifact = _mapping(
            _load_json(cell["resolved_artifact_path"], field=f"artifact{identity!r}"),
            field=f"artifact{identity!r}",
        )
        configuration = _mapping(
            artifact.get("configuration"),
            field=f"artifact{identity!r}.configuration",
        )
        ppo = _mapping(configuration.get("ppo"), field=f"artifact{identity!r}.configuration.ppo")
        contract = {
            "ppo": dict(ppo),
            "evaluation_episodes_per_environment": configuration.get(
                "evaluation_episodes_per_environment"
            ),
            "evaluation_max_episode_steps": configuration.get("evaluation_max_episode_steps"),
        }
        environment_model = identity[:2]
        if environment_model in contracts and contracts[environment_model] != contract:
            raise UpperReferenceLinkError(
                f"learner or evaluation contract differs across seeds for {environment_model!r}"
            )
        contracts[environment_model] = contract
    return contracts


def _evaluation_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "evaluation_episodes_per_environment": contract["evaluation_episodes_per_environment"],
        "evaluation_max_episode_steps": contract["evaluation_max_episode_steps"],
    }


def _runtime_without_device(value: Any, *, field: str) -> dict[str, Any]:
    runtime = dict(_mapping(value, field=field))
    required = {"python", "packages", "jax_backend", "jax_enable_x64", "devices"}
    if set(runtime) != required:
        raise UpperReferenceLinkError(f"{field} has wrong fields")
    del runtime["jax_backend"]
    del runtime["devices"]
    return runtime


def _validate_source_pair(
    primary: Mapping[str, Any],
    upper: Mapping[str, Any],
) -> None:
    primary_git = _mapping(primary.get("git"), field="primary.provenance.git")
    upper_git = _mapping(upper.get("git"), field="upper_reference.provenance.git")
    if primary_git.get("commit") != upper_git.get("commit"):
        raise UpperReferenceLinkError("Git commits differ across paired matrices")
    if primary_git.get("dirty") is not False or upper_git.get("dirty") is not False:
        raise UpperReferenceLinkError("paired matrices must come from clean Git worktrees")
    for key in _SOURCE_IDENTITY_KEYS:
        if primary.get(key) != upper.get(key):
            raise UpperReferenceLinkError(f"source provenance differs for {key}")
    if _runtime_without_device(
        primary.get("runtime_contract"),
        field="primary.provenance.runtime_contract",
    ) != _runtime_without_device(
        upper.get("runtime_contract"),
        field="upper_reference.provenance.runtime_contract",
    ):
        raise UpperReferenceLinkError(
            "runtime contracts differ beyond the allowed backend and device fields"
        )


def _paper_status(tier: str) -> tuple[str, bool, str]:
    if tier == "registered_final":
        return (
            "registered_final_primary_upper_reference_link",
            False,
            "eligible_only_after_all_other_release_gates_pass",
        )
    return (
        f"development_{tier}_primary_upper_reference_link_not_for_paper",
        True,
        "prohibited_development_evidence",
    )


def _validate_registered_budgets(
    tier: str,
    primary_registration: Mapping[str, Any],
    upper_registration: Mapping[str, Any],
) -> None:
    if tier != "registered_final":
        return
    for registration, label in (
        (primary_registration, "primary"),
        (upper_registration, "upper_reference"),
    ):
        for environment in registration["environments"]:
            expected = REGISTERED_TRAIN_STEPS.get(environment["id"])
            if expected is None or environment["total_steps"] != expected:
                raise UpperReferenceLinkError(
                    f"{label} registered-final budget is invalid for {environment['id']!r}"
                )


def _registration_schema(root: Path, *, label: str) -> object:
    registration = _mapping(
        _load_json(root / "registration.json", field=f"{label} registration"),
        field=f"{label} registration",
    )
    return registration.get("schema_version")


def _compute_aware_contracts_by_environment(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    model: str,
) -> dict[str, dict[str, Any]]:
    environments = manifest.get("environments")
    seeds = manifest.get("seeds")
    raw_cells = manifest.get("cells")
    if (
        not isinstance(environments, list)
        or not isinstance(seeds, list)
        or not isinstance(raw_cells, list)
    ):
        raise UpperReferenceLinkError(
            "compute-aware manifest lacks its environment, seed, or cell inventory"
        )
    contracts: dict[str, dict[str, Any]] = {}
    counts = {environment: 0 for environment in environments}
    for index, raw_cell in enumerate(raw_cells):
        if not isinstance(raw_cell, Mapping) or raw_cell.get("model") != model:
            continue
        environment = raw_cell.get("environment")
        seed = raw_cell.get("seed")
        if environment not in counts or seed not in seeds:
            raise UpperReferenceLinkError(
                "compute-aware manifest contains an invalid learner-lane identity"
            )
        _, artifact_path = _safe_relative_path(
            root,
            raw_cell.get("artifact_path"),
            field=f"compute_aware_manifest.cells[{index}].artifact_path",
        )
        artifact = _mapping(
            _load_json(
                artifact_path,
                field=f"compute-aware artifact[{environment},{model},{seed}]",
            ),
            field=f"compute-aware artifact[{environment},{model},{seed}]",
        )
        configuration = _mapping(
            artifact.get("configuration"),
            field=f"artifact[{environment},{model},{seed}].configuration",
        )
        ppo = _mapping(
            configuration.get("ppo"),
            field=f"artifact[{environment},{model},{seed}].configuration.ppo",
        )
        evaluation = _mapping(
            artifact.get("evaluation"),
            field=f"artifact[{environment},{model},{seed}].evaluation",
        )
        contract = {
            "ppo": dict(ppo),
            "evaluation": {
                "evaluation_episodes_per_environment": artifact.get(
                    "evaluation_episodes_per_environment"
                ),
                "evaluation_max_episode_steps": artifact.get("evaluation_max_episode_steps"),
                "actual_evaluation_steps_per_environment": artifact.get(
                    "actual_evaluation_steps_per_environment"
                ),
                "actual_evaluation_transitions": artifact.get("actual_evaluation_transitions"),
                "episodes": evaluation.get("episodes"),
                "episodes_per_environment": evaluation.get("episodes_per_environment"),
                "num_environments": evaluation.get("num_environments"),
                "scan_steps_per_environment": evaluation.get("scan_steps_per_environment"),
            },
            "execution": {
                "comparison_profile": configuration.get("comparison_profile"),
                "requested_environment_steps": configuration.get("requested_environment_steps"),
                "realized_environment_steps": configuration.get("realized_environment_steps"),
                "actual_environment_steps": artifact.get("actual_environment_steps"),
            },
        }
        if environment in contracts and contracts[environment] != contract:
            raise UpperReferenceLinkError(
                "compute-aware PPO or evaluation contract differs across seeds "
                f"for {(environment, model)!r}"
            )
        contracts[environment] = contract
        counts[environment] += 1
    expected_count = len(seeds)
    if set(contracts) != set(environments) or any(
        count != expected_count for count in counts.values()
    ):
        raise UpperReferenceLinkError(
            "compute-aware learner lane does not contain one cell per registered seed"
        )
    return contracts


def _build_compute_aware_upper_reference_link(
    primary_path: Path,
    upper_path: Path,
) -> dict[str, Any]:
    try:
        upper_aggregate = build_registered_aggregate(upper_path / "frozen_manifest.json")
    except (OSError, TypeError, ValueError) as error:
        raise UpperReferenceLinkError(
            f"schema-7 upper-reference validation failed: {error}"
        ) from error
    if (
        upper_aggregate.get("schema_version") != 3
        or upper_aggregate.get("matrix_kind") != "upper_reference"
        or upper_aggregate.get("raw_integrity", {}).get("primary_matrix_binding_validated")
        is not True
    ):
        raise UpperReferenceLinkError(
            "schema-7 upper-reference aggregate has the wrong evidence identity"
        )
    try:
        primary_binding = normalize_primary_matrix_binding(
            upper_aggregate["primary_matrix_binding"]
        )
        learner_binding = normalize_memoryless_learner_binding(
            upper_aggregate["memoryless_learner_binding"]
        )
        (
            validated_primary_binding,
            validated_learner_binding,
            primary_aggregate,
            _,
        ) = validate_bound_compute_aware_primary_matrix(
            primary_binding,
            learner_binding,
            expected_primary_root=primary_path,
        )
    except (KeyError, RegisteredAggregationError, ValueError) as error:
        raise UpperReferenceLinkError(
            f"schema-6 primary binding validation failed: {error}"
        ) from error
    if (
        validated_primary_binding != primary_binding or validated_learner_binding != learner_binding
    ):  # pragma: no cover - normalizers are deterministic
        raise UpperReferenceLinkError(
            "normalized compute-aware primary bindings are not deterministic"
        )
    if (
        primary_aggregate.get("schema_version") != 2
        or primary_aggregate.get("matrix_kind") != "primary_comparison"
    ):
        raise UpperReferenceLinkError("supplied primary root is not a schema-6 primary aggregate")

    primary_registration = _mapping(
        _load_json(
            primary_path / "registration.json",
            field="schema-6 primary registration",
        ),
        field="schema-6 primary registration",
    )
    upper_registration = _mapping(
        _load_json(
            upper_path / "registration.json",
            field="schema-7 upper-reference registration",
        ),
        field="schema-7 upper-reference registration",
    )
    primary_manifest = _mapping(
        _load_json(
            primary_path / "frozen_manifest.json",
            field="schema-6 primary manifest",
        ),
        field="schema-6 primary manifest",
    )
    upper_manifest = _mapping(
        _load_json(
            upper_path / "frozen_manifest.json",
            field="schema-7 upper-reference manifest",
        ),
        field="schema-7 upper-reference manifest",
    )
    if (
        primary_registration.get("schema_version") != 6
        or upper_registration.get("schema_version") != 7
        or primary_registration.get("evidence_tier") != "registered_final"
        or upper_registration.get("evidence_tier") != "registered_final"
        or primary_registration.get("comparison_profile") != "arcmind_shared_comparison"
        or upper_registration.get("comparison_profile") != "arcmind_shared_comparison"
        or primary_registration.get("evaluation_episodes_per_env")
        != COMPUTE_AWARE_FINAL_EVALUATION_EPISODES_PER_ENV
        or upper_registration.get("evaluation_episodes_per_env")
        != COMPUTE_AWARE_FINAL_EVALUATION_EPISODES_PER_ENV
        or primary_registration.get("require_gpu") is not True
        or upper_registration.get("require_gpu") is not True
        or primary_registration.get("quick") is not False
        or upper_registration.get("quick") is not False
        or primary_registration.get("seeds") != list(COMPUTE_AWARE_FINAL_SEEDS)
        or upper_registration.get("seeds") != list(COMPUTE_AWARE_FINAL_SEEDS)
    ):
        raise UpperReferenceLinkError(
            "compute-aware primary and upper registrations do not share the "
            "exact final execution contract"
        )
    try:
        registration_primary_binding = normalize_primary_matrix_binding(
            upper_registration.get("primary_matrix_binding")
        )
        registration_learner_binding = normalize_memoryless_learner_binding(
            upper_registration.get("memoryless_learner_binding")
        )
    except ValueError as error:
        raise UpperReferenceLinkError(str(error)) from error
    if (
        registration_primary_binding != primary_binding
        or registration_learner_binding != learner_binding
    ):
        raise UpperReferenceLinkError(
            "schema-7 registration bindings drift from its validated aggregate"
        )

    expected_primary_environments = [environment for environment, _ in COMPUTE_AWARE_FINAL_PANEL]
    expected_upper_environments = [
        environment for environment, _ in COMPUTE_AWARE_UPPER_REFERENCE_PANEL
    ]
    if (
        primary_aggregate.get("environments") != expected_primary_environments
        or upper_aggregate.get("environments") != expected_upper_environments
        or upper_aggregate.get("models") != ["memoryless_mlp"]
        or [
            (group.get("environment"), group.get("model"))
            for group in upper_aggregate.get("groups", [])
            if isinstance(group, Mapping)
        ]
        != [(environment, "memoryless_mlp") for environment in expected_upper_environments]
        or upper_aggregate.get("paired_differences_against_arcmind") != []
        or upper_aggregate.get("supplemental_paired_differences_against_arcmind") != []
    ):
        raise UpperReferenceLinkError(
            "schema-7 aggregate must contain only four diagnostic memoryless groups"
        )
    alias_mapping = [
        {
            "upper_reference_environment": upper_environment,
            "primary_environment": UPPER_TO_PRIMARY_ENVIRONMENT[upper_environment],
        }
        for upper_environment in expected_upper_environments
    ]
    if [item["primary_environment"] for item in alias_mapping] != (
        expected_primary_environments
    ) or upper_aggregate.get("upper_reference_alias_mapping") != alias_mapping:
        raise UpperReferenceLinkError(
            "schema-7 upper-reference aliases do not map exactly to the primary panel"
        )

    primary_contracts = _compute_aware_contracts_by_environment(
        primary_path,
        primary_manifest,
        model="memoryless_mlp",
    )
    upper_contracts = _compute_aware_contracts_by_environment(
        upper_path,
        upper_manifest,
        model="memoryless_mlp",
    )
    paired_contracts: list[dict[str, Any]] = []
    for mapping in alias_mapping:
        primary_environment = mapping["primary_environment"]
        upper_environment = mapping["upper_reference_environment"]
        primary_contract = primary_contracts[primary_environment]
        upper_contract = upper_contracts[upper_environment]
        if primary_contract != upper_contract:
            raise UpperReferenceLinkError(
                "full PPO or evaluation contract differs for paired environments "
                f"{primary_environment!r} and {upper_environment!r}"
            )
        paired_contracts.append(
            {
                **mapping,
                "model": "memoryless_mlp",
                **primary_contract,
            }
        )

    primary_provenance = _mapping(
        primary_aggregate.get("provenance"),
        field="schema-6 primary aggregate provenance",
    )
    upper_provenance = _mapping(
        upper_aggregate.get("provenance"),
        field="schema-7 upper-reference aggregate provenance",
    )
    if dict(primary_provenance) != dict(upper_provenance):
        raise UpperReferenceLinkError(
            "schema-6 and schema-7 matrices require exact provenance equality"
        )
    primary_git = _mapping(
        primary_provenance.get("git"),
        field="schema-6 primary aggregate provenance.git",
    )
    if primary_git.get("dirty") is not False:
        raise UpperReferenceLinkError(
            "compute-aware linked matrices must come from clean Git worktrees"
        )

    upper_evidence_hashes = {
        "registration_file_sha256": sha256_file(upper_path / "registration.json"),
        "manifest_file_sha256": sha256_file(upper_path / "frozen_manifest.json"),
        "manifest_internal_sha256": upper_manifest["manifest_sha256"],
        "completion_index_file_sha256": sha256_file(upper_path / "completion_index.json"),
        "checksum_manifest_file_sha256": sha256_file(upper_path / "checksums.sha256"),
    }
    return {
        "schema_version": 2,
        "status": "registered_final_primary_upper_reference_link",
        "paper_status": "eligible_only_after_all_other_release_gates_pass",
        "not_for_paper": False,
        "evidence_tier": "registered_final",
        "pairing_mode": ("schema6_selected_memoryless_to_schema7_compute_aware_upper_reference"),
        "statistical_unit": "seed",
        "seeds": list(COMPUTE_AWARE_FINAL_SEEDS),
        "alias_mapping": alias_mapping,
        "primary_matrix_binding": dict(primary_binding),
        "memoryless_learner_binding": {
            **dict(learner_binding),
            "learner": dict(learner_binding["learner"]),
        },
        "contract_equality": {
            "full_ppo_and_evaluation_validated": True,
            "full_provenance_validated": True,
            "all_bound_primary_hashes_validated": True,
            "raw_returns_included": False,
            "task_pairs": paired_contracts,
        },
        "primary": {
            "registration_file_sha256": primary_binding["primary_registration_file_sha256"],
            "manifest_internal_sha256": primary_binding["primary_manifest_internal_sha256"],
            "manifest_file_sha256": primary_binding["primary_manifest_file_sha256"],
            "aggregate_file_sha256": primary_binding["primary_aggregate_file_sha256"],
            "completion_index_file_sha256": primary_binding["primary_completion_index_file_sha256"],
            "checksum_manifest_file_sha256": primary_binding[
                "primary_checksum_manifest_file_sha256"
            ],
            "provenance": dict(primary_provenance),
            "raw_integrity": primary_aggregate.get("raw_integrity"),
        },
        "upper_reference": {
            **upper_evidence_hashes,
            "provenance": dict(upper_provenance),
            "raw_integrity": upper_aggregate.get("raw_integrity"),
        },
        "source_pairing": {
            "git_commit": primary_git["commit"],
            "dependency_lock_sha256": primary_provenance["dependency_lock_sha256"],
            "pobax_commit": primary_provenance["pobax_commit"],
            "navix_commit": primary_provenance["navix_commit"],
            "provenance_equality": "exact",
        },
    }


def build_upper_reference_link(
    primary_root: str | Path,
    upper_reference_root: str | Path,
) -> dict[str, Any]:
    """Validate two complete raw matrices and return their canonical link."""

    primary_path = Path(primary_root).resolve()
    upper_path = Path(upper_reference_root).resolve()
    if not primary_path.is_dir() or not upper_path.is_dir():
        raise UpperReferenceLinkError("both raw matrix roots must be existing directories")
    if primary_path == upper_path:
        raise UpperReferenceLinkError("primary and upper-reference roots must be distinct")

    schema_pair = (
        _registration_schema(primary_path, label="primary"),
        _registration_schema(upper_path, label="upper-reference"),
    )
    if schema_pair == (6, 7):
        return _build_compute_aware_upper_reference_link(
            primary_path,
            upper_path,
        )
    if 6 in schema_pair or 7 in schema_pair:
        raise UpperReferenceLinkError(
            "compute-aware linking requires the exact registration schema pair (6, 7)"
        )

    primary_registration, primary_registration_sha256 = _validate_registration(primary_path)
    upper_registration, upper_registration_sha256 = _validate_registration(upper_path)
    if primary_registration["matrix_kind"] != "primary_comparison":
        raise UpperReferenceLinkError("primary root is not a primary_comparison matrix")
    if upper_registration["matrix_kind"] != "upper_reference":
        raise UpperReferenceLinkError("upper-reference root is not an upper_reference matrix")

    primary_manifest, primary_cells = _validate_manifest(
        primary_path,
        primary_registration,
    )
    upper_manifest, upper_cells = _validate_manifest(
        upper_path,
        upper_registration,
    )
    _validate_completion_and_checksums(primary_path, primary_manifest, primary_cells)
    _validate_completion_and_checksums(upper_path, upper_manifest, upper_cells)
    primary_aggregate = _deep_validate(primary_path, primary_registration)
    upper_aggregate = _deep_validate(upper_path, upper_registration)

    tier = primary_registration["evidence_tier"]
    if upper_registration["evidence_tier"] != tier:
        raise UpperReferenceLinkError("paired matrices have different evidence tiers")
    selected_primary_to_author_upper = (
        tier == "registered_final"
        and primary_registration["schema_version"] == 4
        and primary_registration.get("comparison_profile") == "arcmind_shared_comparison"
        and upper_registration["schema_version"] == 2
        and upper_registration.get("comparison_profile") == "pobax_author_semantics"
    )
    legacy_matching_contract = primary_registration["schema_version"] == upper_registration[
        "schema_version"
    ] and primary_registration.get("comparison_profile") == upper_registration.get(
        "comparison_profile"
    )
    if not selected_primary_to_author_upper and not legacy_matching_contract:
        raise UpperReferenceLinkError(
            "paired matrices have different registration schemas or comparison profiles"
        )
    if primary_registration["seeds"] != upper_registration["seeds"]:
        raise UpperReferenceLinkError("paired matrices must use the same ordered seed list")
    if (
        not selected_primary_to_author_upper
        and primary_registration["learner"] != upper_registration["learner"]
    ):
        raise UpperReferenceLinkError("paired matrices have different learner registrations")
    if (
        primary_registration["evaluation_episodes_per_env"]
        != upper_registration["evaluation_episodes_per_env"]
        or primary_registration["quick"] != upper_registration["quick"]
    ):
        raise UpperReferenceLinkError("paired matrices have different evaluation registrations")

    upper_environments = [item["id"] for item in upper_registration["environments"]]
    try:
        mapped_primary_environments = [
            UPPER_TO_PRIMARY_ENVIRONMENT[environment] for environment in upper_environments
        ]
    except KeyError as error:
        raise UpperReferenceLinkError(
            f"upper-reference environment has no registered primary mapping: {error.args[0]!r}"
        ) from error
    primary_environments = [item["id"] for item in primary_registration["environments"]]
    if mapped_primary_environments != primary_environments:
        raise UpperReferenceLinkError(
            "ordered upper-reference aliases do not map exactly to primary environments"
        )

    _validate_registered_budgets(tier, primary_registration, upper_registration)
    alias_mapping: list[dict[str, str]] = []
    for upper_environment, primary_environment in zip(upper_environments, primary_environments):
        alias_mapping.append(
            {
                "upper_reference_environment": upper_environment,
                "primary_environment": primary_environment,
            }
        )
    if selected_primary_to_author_upper:
        primary_model_contracts = _task_contracts_by_model(primary_cells)
        upper_model_contracts = _task_contracts_by_model(upper_cells)
        for mapping in alias_mapping:
            primary_environment = mapping["primary_environment"]
            upper_environment = mapping["upper_reference_environment"]
            upper_contract = upper_model_contracts[(upper_environment, "memoryless_mlp")]
            for model in primary_registration["models"]:
                primary_contract = primary_model_contracts[(primary_environment, model)]
                if _evaluation_contract(primary_contract) != _evaluation_contract(upper_contract):
                    raise UpperReferenceLinkError(
                        "evaluation contract differs for paired environments "
                        f"{primary_environment!r} and {upper_environment!r}"
                    )
        binding = normalize_final_selection_binding(primary_registration["tuning_selection"])
        learner_contract = {
            "pairing_mode": "selected_primary_to_pobax_author_upper_reference",
            "primary": {
                "comparison_profile": primary_registration["comparison_profile"],
                "tuning_selection_binding": {
                    "raw_matrix_path": binding["raw_matrix_path"],
                    "aggregate_path": binding["aggregate_path"],
                    "aggregate_sha256": binding["aggregate_sha256"],
                    "source_registration_sha256": binding["source_registration_sha256"],
                    "source_manifest_sha256": binding["source_manifest_sha256"],
                },
                "evaluation_episodes_per_environment": primary_registration[
                    "evaluation_episodes_per_env"
                ],
                "quick": primary_registration["quick"],
                "selections": [
                    {
                        "environment": selection["environment"],
                        "model_family": selection["model_family"],
                        "implementation_model": selection["implementation_model"],
                        "candidate_id": selection["candidate_id"],
                        "learner": dict(selection["learner"]),
                    }
                    for selection in binding["selections"]
                ],
                "task_contracts": [
                    {
                        "primary_environment": environment,
                        "model": model,
                        **primary_model_contracts[(environment, model)],
                    }
                    for environment in primary_environments
                    for model in primary_registration["models"]
                ],
            },
            "upper_reference": {
                "comparison_profile": upper_registration["comparison_profile"],
                "learner": upper_registration["learner"],
                "evaluation_episodes_per_environment": upper_registration[
                    "evaluation_episodes_per_env"
                ],
                "quick": upper_registration["quick"],
                "task_contracts": [
                    {
                        "upper_reference_environment": environment,
                        "model": "memoryless_mlp",
                        **upper_model_contracts[(environment, "memoryless_mlp")],
                    }
                    for environment in upper_environments
                ],
            },
        }
    else:
        primary_contracts = _task_contracts(primary_cells)
        upper_contracts = _task_contracts(upper_cells)
        for mapping in alias_mapping:
            primary_environment = mapping["primary_environment"]
            upper_environment = mapping["upper_reference_environment"]
            if upper_contracts[upper_environment] != primary_contracts[primary_environment]:
                raise UpperReferenceLinkError(
                    "learner or evaluation contract differs for paired environments "
                    f"{primary_environment!r} and {upper_environment!r}"
                )
        learner_contract = {
            "learner": primary_registration["learner"],
            "evaluation_episodes_per_environment": primary_registration[
                "evaluation_episodes_per_env"
            ],
            "quick": primary_registration["quick"],
            "task_contracts": [
                {
                    "primary_environment": mapping["primary_environment"],
                    **primary_contracts[mapping["primary_environment"]],
                }
                for mapping in alias_mapping
            ],
        }

    primary_provenance = _mapping(
        primary_aggregate["provenance"],
        field="primary aggregate provenance",
    )
    upper_provenance = _mapping(
        upper_aggregate["provenance"],
        field="upper-reference aggregate provenance",
    )
    _validate_source_pair(primary_provenance, upper_provenance)
    status, not_for_paper, paper_status = _paper_status(tier)
    return {
        "schema_version": 1,
        "status": status,
        "paper_status": paper_status,
        "not_for_paper": not_for_paper,
        "evidence_tier": tier,
        "statistical_unit": "seed",
        "seeds": list(primary_registration["seeds"]),
        "alias_mapping": alias_mapping,
        "learner_contract": learner_contract,
        "primary": {
            "registration_sha256": primary_registration_sha256,
            "matrix_manifest_sha256": primary_manifest["manifest_sha256"],
            "provenance": dict(primary_provenance),
            "raw_integrity": primary_aggregate.get("raw_integrity"),
        },
        "upper_reference": {
            "registration_sha256": upper_registration_sha256,
            "matrix_manifest_sha256": upper_manifest["manifest_sha256"],
            "provenance": dict(upper_provenance),
            "raw_integrity": upper_aggregate.get("raw_integrity"),
        },
        "source_pairing": {
            "git_commit": primary_provenance["git"]["commit"],
            "dependency_lock_sha256": primary_provenance["dependency_lock_sha256"],
            "pobax_commit": primary_provenance["pobax_commit"],
            "navix_commit": primary_provenance["navix_commit"],
            "runtime_backend_and_devices_may_differ": True,
        },
    }


def link_upper_reference(
    primary_root: str | Path,
    upper_reference_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate paired matrices and atomically create their derived link."""

    primary = Path(primary_root).resolve()
    upper = Path(upper_reference_root).resolve()
    output = Path(output_path).resolve()
    if (
        output == primary
        or output.is_relative_to(primary)
        or output == upper
        or output.is_relative_to(upper)
    ):
        raise UpperReferenceLinkError("link output must be outside both immutable raw matrix roots")
    result = build_upper_reference_link(primary, upper)
    atomic_write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary_root", type=Path)
    parser.add_argument("upper_reference_root", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    result = link_upper_reference(
        arguments.primary_root,
        arguments.upper_reference_root,
        arguments.output,
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "status": result["status"],
                "primary_manifest_sha256": (
                    result["primary"]["manifest_internal_sha256"]
                    if result["schema_version"] == 2
                    else result["primary"]["matrix_manifest_sha256"]
                ),
                "upper_reference_manifest_sha256": (
                    result["upper_reference"]["manifest_internal_sha256"]
                    if result["schema_version"] == 2
                    else result["upper_reference"]["matrix_manifest_sha256"]
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "REGISTERED_TRAIN_STEPS",
    "UpperReferenceLinkError",
    "build_upper_reference_link",
    "link_upper_reference",
]
