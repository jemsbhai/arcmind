"""Execute a frozen POBAX matrix without overwriting completed cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import uuid
from argparse import Namespace
from pathlib import Path
from typing import Any

from benchmarks.pobax.aggregate_development import build_development_aggregate
from benchmarks.pobax.aggregate_registered import build_registered_aggregate
from benchmarks.pobax.model_registry import (
    policy_contract_metadata_for_model,
    reference_implementation_for_model,
    requires_explicit_policy_contract,
    validate_causal_transformer_horizon_contract,
    validate_model_environment_contract,
    validate_model_evidence_tier,
    validate_policy_contract_metadata,
    validate_policy_core_contract,
    validate_policy_model_id,
    validate_required_reference_implementation,
)
from benchmarks.pobax.registered_artifacts import (
    ExistingArtifactMismatchError,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    canonical_json_sha256,
    exclusive_process_lock,
    matrix_process_lock_path,
    registered_cell_id,
    registered_cell_path,
    sha256_file,
    validate_paired_seed_manifests,
    validate_unique_cell_ids,
    write_checksum_manifest,
)
from benchmarks.pobax.registration_protocol import (
    COMPUTE_AWARE_UPPER_REFERENCE_PANEL,
    normalize_candidate_families,
    normalize_final_selection_binding,
    normalize_learner,
    normalize_learner_bindings,
    normalize_memoryless_learner_binding,
    normalize_panel_selection_binding,
    normalize_primary_matrix_binding,
    normalize_shared_learner_grid,
    normalize_task_model_incidence,
    normalize_tuned_families,
    realized_environment_steps,
    registration_fields,
    validate_comparison_profile,
    validate_compute_aware_final_contract,
    validate_compute_aware_primary_binding_against_aggregate,
    validate_compute_aware_tuning_contract,
    validate_compute_aware_upper_reference_contract,
    validate_development_tuning_contract,
    validate_final_provenance_against_tuning,
    validate_final_selection_against_aggregate,
    validate_panel_selection_against_aggregate,
)
from benchmarks.pobax.run_pilot import (
    ARTIFACT_SCHEMA_BY_REGISTRATION,
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
_MATRIX_KINDS = {
    "primary_comparison",
    "upper_reference",
    "hyperparameter_selection",
}
_SHARD_PARTITION_ALGORITHM = "manifest-order-modulo-v1"
_SCHEMA_V7_PRIMARY_HASH_FIELDS = (
    "primary_aggregate_file_sha256",
    "primary_registration_file_sha256",
    "primary_manifest_file_sha256",
    "primary_manifest_internal_sha256",
    "primary_completion_index_file_sha256",
    "primary_checksum_manifest_file_sha256",
    "primary_implementation_source_sha256",
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _schema_v5_candidates(registration: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Expand the one shared learner grid into immutable family candidates."""

    families = normalize_tuned_families(registration["tuned_families"])
    learner_grid = normalize_shared_learner_grid(registration["learner_grid"])
    return tuple(
        {
            "candidate_id": f"{family['family_id']}.{grid_item['learner_id']}",
            "model_family": family["family_id"],
            "implementation_model": family["implementation_model"],
            "learner_id": grid_item["learner_id"],
            "learner": grid_item["learner"],
        }
        for family in families
        for grid_item in learner_grid
    )


def _schema_v6_final_specs(registration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve each final model to its selected or inherited learner."""

    binding = normalize_panel_selection_binding(registration["tuning_selection"])
    selections = {selection["model_family"]: selection for selection in binding["selections"]}
    learner_bindings = normalize_learner_bindings(
        registration["learner_bindings"],
        models=registration["models"],
    )
    return {
        learner_binding["model"]: {
            "candidate_id": selections[learner_binding["source_model_family"]]["candidate_id"],
            "model_family": learner_binding["source_model_family"],
            "learner_id": selections[learner_binding["source_model_family"]]["learner_id"],
            "learner": selections[learner_binding["source_model_family"]]["learner"],
            "implementation_model": learner_binding["model"],
            "learner_binding_mode": learner_binding["mode"],
            "learner_source_model_family": learner_binding["source_model_family"],
            "tuning_aggregate_sha256": binding["aggregate_sha256"],
            "tuning_completion_index_sha256": binding["source_completion_index_sha256"],
            "tuning_checksum_manifest_sha256": binding["source_checksum_manifest_sha256"],
            "tuning_implementation_source_sha256": binding["source_implementation_sha256"],
        }
        for learner_binding in learner_bindings
    }


def _schema_v6_environment_models(
    registration: dict[str, Any],
) -> dict[str, tuple[str, ...]]:
    incidence = normalize_task_model_incidence(
        registration["task_model_incidence"],
        environments=[environment["id"] for environment in registration["environments"]],
        models=registration["models"],
    )
    return {entry["environment"]: entry["models"] for entry in incidence}


def _schema_v7_final_spec(registration: dict[str, Any]) -> dict[str, Any]:
    """Resolve the upper-reference memoryless learner and primary hashes."""

    learner = normalize_memoryless_learner_binding(registration["memoryless_learner_binding"])
    primary = normalize_primary_matrix_binding(registration["primary_matrix_binding"])
    return {**learner, **primary}


def _matrix_cell_identities(
    registration: dict[str, Any],
) -> tuple[tuple[dict[str, Any], str, int], ...]:
    """Return the exact ordered execution inventory for one registration."""

    if registration["schema_version"] == 3:
        matrix_models = tuple(
            candidate["candidate_id"]
            for family in normalize_candidate_families(registration["candidate_families"])
            for candidate in family["candidates"]
        )
    elif registration["schema_version"] == 5:
        matrix_models = tuple(
            candidate["candidate_id"] for candidate in _schema_v5_candidates(registration)
        )
    else:
        matrix_models = tuple(registration["models"])
    environment_models = (
        _schema_v6_environment_models(registration)
        if registration["schema_version"] == 6
        else {environment["id"]: matrix_models for environment in registration["environments"]}
    )
    return tuple(
        (environment, model, seed)
        for environment in registration["environments"]
        for model in environment_models[environment["id"]]
        for seed in registration["seeds"]
    )


def _bound_repository_path(relative_path: str, *, field: str) -> Path:
    path = _REPOSITORY_ROOT.joinpath(*Path(relative_path).parts).resolve()
    if not path.is_relative_to(_REPOSITORY_ROOT):
        raise ValueError(f"{field} escapes the repository root")
    return path


def _validate_final_tuning_selection(
    registration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = normalize_final_selection_binding(registration["tuning_selection"])
    aggregate_path = _bound_repository_path(
        binding["aggregate_path"],
        field="tuning_selection.aggregate_path",
    )
    try:
        aggregate_hash = sha256_file(aggregate_path)
    except OSError as error:
        raise ValueError(f"cannot read tuning selection aggregate: {aggregate_path}") from error
    if aggregate_hash != binding["aggregate_sha256"]:
        raise ValueError("tuning_selection.aggregate_sha256 does not match aggregate bytes")
    raw_matrix_path = _bound_repository_path(
        binding["raw_matrix_path"],
        field="tuning_selection.raw_matrix_path",
    )
    for filename, binding_field in (
        ("completion_index.json", "source_completion_index_sha256"),
        ("checksums.sha256", "source_checksum_manifest_sha256"),
    ):
        source_path = raw_matrix_path / filename
        try:
            source_hash = sha256_file(source_path)
        except OSError as error:
            raise ValueError(f"cannot read tuning selection source: {source_path}") from error
        if source_hash != binding[binding_field]:
            raise ValueError(f"tuning_selection.{binding_field} does not match source bytes")
    rebuilt = build_development_aggregate(raw_matrix_path)
    aggregate_bytes = aggregate_path.read_bytes()
    if aggregate_bytes != canonical_json_bytes(rebuilt) + b"\n":
        raise ValueError(
            "tuning selection aggregate is not the canonical rebuild of its raw matrix"
        )
    if (
        rebuilt["registration_sha256"] != binding["source_registration_sha256"]
        or rebuilt["matrix_manifest_sha256"] != binding["source_manifest_sha256"]
    ):
        raise ValueError("tuning selection source registration or manifest hash drifts")
    validate_final_selection_against_aggregate(
        binding,
        rebuilt,
        models=registration["models"],
        environments=[item["id"] for item in registration["environments"]],
        final_seeds=registration["seeds"],
    )
    return binding, rebuilt


def _validate_compute_aware_final_tuning_selection(
    registration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = normalize_panel_selection_binding(registration["tuning_selection"])
    aggregate_path = _bound_repository_path(
        binding["aggregate_path"],
        field="tuning_selection.aggregate_path",
    )
    try:
        aggregate_hash = sha256_file(aggregate_path)
    except OSError as error:
        raise ValueError(
            f"cannot read compute-aware tuning selection aggregate: {aggregate_path}"
        ) from error
    if aggregate_hash != binding["aggregate_sha256"]:
        raise ValueError("tuning_selection.aggregate_sha256 does not match aggregate bytes")
    raw_matrix_path = _bound_repository_path(
        binding["raw_matrix_path"],
        field="tuning_selection.raw_matrix_path",
    )
    for filename, binding_field in (
        ("completion_index.json", "source_completion_index_sha256"),
        ("checksums.sha256", "source_checksum_manifest_sha256"),
    ):
        source_path = raw_matrix_path / filename
        try:
            source_hash = sha256_file(source_path)
        except OSError as error:
            raise ValueError(
                f"cannot read compute-aware tuning selection source: {source_path}"
            ) from error
        if source_hash != binding[binding_field]:
            raise ValueError(f"tuning_selection.{binding_field} does not match source bytes")
    rebuilt = build_development_aggregate(raw_matrix_path)
    aggregate_bytes = aggregate_path.read_bytes()
    if aggregate_bytes != canonical_json_bytes(rebuilt) + b"\n":
        raise ValueError(
            "compute-aware tuning selection aggregate is not the canonical rebuild "
            "of its raw matrix"
        )
    if (
        rebuilt["registration_sha256"] != binding["source_registration_sha256"]
        or rebuilt["matrix_manifest_sha256"] != binding["source_manifest_sha256"]
    ):
        raise ValueError("compute-aware tuning source registration or manifest hash drifts")
    validate_panel_selection_against_aggregate(
        binding,
        rebuilt,
        final_seeds=registration["seeds"],
    )
    return binding, rebuilt


def _validate_compute_aware_upper_primary(
    registration: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    primary = normalize_primary_matrix_binding(registration["primary_matrix_binding"])
    learner = normalize_memoryless_learner_binding(registration["memoryless_learner_binding"])
    raw_matrix_path = _bound_repository_path(
        primary["raw_matrix_path"],
        field="primary_matrix_binding.raw_matrix_path",
    )
    aggregate_path = _bound_repository_path(
        primary["aggregate_path"],
        field="primary_matrix_binding.aggregate_path",
    )
    paths = {
        "primary_registration_file_sha256": raw_matrix_path / "registration.json",
        "primary_manifest_file_sha256": raw_matrix_path / "frozen_manifest.json",
        "primary_completion_index_file_sha256": raw_matrix_path / "completion_index.json",
        "primary_checksum_manifest_file_sha256": raw_matrix_path / "checksums.sha256",
        "primary_aggregate_file_sha256": aggregate_path,
    }
    for binding_field, path in paths.items():
        try:
            actual_hash = sha256_file(path)
        except OSError as error:
            raise ValueError(f"cannot read schema-v7 primary source: {path}") from error
        if actual_hash != primary[binding_field]:
            raise ValueError(f"primary_matrix_binding.{binding_field} does not match source bytes")
    rebuilt = build_registered_aggregate(raw_matrix_path / "frozen_manifest.json")
    try:
        aggregate_bytes = aggregate_path.read_bytes()
    except OSError as error:  # pragma: no cover - hash read above normally catches this
        raise ValueError(f"cannot read schema-v7 primary aggregate: {aggregate_path}") from error
    if aggregate_bytes != canonical_json_bytes(rebuilt) + b"\n":
        raise ValueError(
            "schema-v7 primary aggregate is not the canonical rebuild of its raw matrix"
        )
    validate_compute_aware_primary_binding_against_aggregate(
        primary,
        learner,
        rebuilt,
    )
    return primary, learner, rebuilt


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
    if schema_version == 3 and tier != "development_tuning":
        raise ValueError("registration schema version 3 is reserved for development_tuning")
    if schema_version == 4 and tier != "registered_final":
        raise ValueError("registration schema version 4 is reserved for registered_final")
    if schema_version == 5 and tier != "development_tuning":
        raise ValueError("registration schema version 5 is reserved for development_tuning")
    if schema_version == 6 and tier != "registered_final":
        raise ValueError("registration schema version 6 is reserved for registered_final")
    if schema_version == 7 and tier != "registered_final":
        raise ValueError("registration schema version 7 is reserved for registered_final")
    matrix_kind = registration.get("matrix_kind")
    if matrix_kind not in _MATRIX_KINDS:
        raise ValueError(f"unsupported matrix_kind: {matrix_kind!r}")

    candidate_families: tuple[dict[str, Any], ...] = ()
    schema_v5_candidates: tuple[dict[str, Any], ...] = ()
    if schema_version == 3:
        candidate_families = normalize_candidate_families(registration.get("candidate_families"))
        models = [
            candidate["candidate_id"]
            for family in candidate_families
            for candidate in family["candidates"]
        ]
    elif schema_version == 5:
        schema_v5_candidates = _schema_v5_candidates(registration)
        models = [candidate["candidate_id"] for candidate in schema_v5_candidates]
    else:
        models = registration.get("models")
        if (
            not isinstance(models, list)
            or not models
            or any(not isinstance(model, str) or not model for model in models)
        ):
            raise ValueError("models must be a non-empty list of names")
        if len(set(models)) != len(models):
            raise ValueError("models must not contain duplicates")
        for index, model in enumerate(models):
            validate_policy_model_id(model, field=f"models[{index}]")
            validate_model_evidence_tier(model, tier, field=f"models[{index}]")
    if matrix_kind == "primary_comparison" and "arcmind" not in models:
        raise ValueError("primary_comparison matrices must contain arcmind")
    if matrix_kind == "upper_reference" and models != ["memoryless_mlp"]:
        raise ValueError("upper_reference matrices must contain only memoryless_mlp")
    if matrix_kind == "hyperparameter_selection" and schema_version not in {3, 5}:
        raise ValueError("hyperparameter_selection requires registration schema version 3 or 5")
    if schema_version in {4, 6} and matrix_kind != "primary_comparison":
        raise ValueError(
            f"registration schema version {schema_version} requires primary_comparison"
        )
    if schema_version == 7 and matrix_kind != "upper_reference":
        raise ValueError("registration schema version 7 requires upper_reference")
    if (
        tier == "registered_final"
        and matrix_kind == "primary_comparison"
        and schema_version not in {4, 6}
    ):
        raise ValueError(
            "registered-final primary comparisons require schema version 4 or 6 "
            "and an explicit tuning selection"
        )

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
    implementation_models = (
        [family["implementation_model"] for family in candidate_families]
        if schema_version == 3
        else [
            family["implementation_model"]
            for family in normalize_tuned_families(registration["tuned_families"])
        ]
        if schema_version == 5
        else models
    )
    if schema_version == 6:
        incidence = normalize_task_model_incidence(
            registration["task_model_incidence"],
            environments=environment_ids,
            models=models,
        )
        for incidence_entry in incidence:
            for model in incidence_entry["models"]:
                validate_model_evidence_tier(model, tier, field=f"model {model!r}")
                validate_model_environment_contract(
                    model,
                    incidence_entry["environment"],
                    field=f"model {model!r}",
                )
    else:
        for model in implementation_models:
            validate_model_evidence_tier(model, tier, field=f"model {model!r}")
            for environment_id in environment_ids:
                validate_model_environment_contract(
                    model,
                    environment_id,
                    field=f"model {model!r}",
                )
    if schema_version == 7 and environment_ids != [
        environment for environment, _ in COMPUTE_AWARE_UPPER_REFERENCE_PANEL
    ]:
        raise ValueError("schema-v7 upper references require the exact ordered registered aliases")
    if matrix_kind == "upper_reference":
        unsupported = sorted(set(environment_ids) - set(UPPER_REFERENCE_TARGETS))
        if unsupported:
            raise ValueError(
                "upper_reference matrices contain environments without "
                f"registered adapters: {unsupported}"
            )

    final_selection = (
        _validate_final_tuning_selection(registration)[0] if schema_version == 4 else None
    )
    if schema_version == 6:
        _validate_compute_aware_final_tuning_selection(registration)
    if schema_version == 7:
        _validate_compute_aware_upper_primary(registration)
    schema_v6_specs = _schema_v6_final_specs(registration) if schema_version == 6 else {}
    learners = (
        [
            candidate["learner"]
            for family in candidate_families
            for candidate in family["candidates"]
        ]
        if schema_version == 3
        else [candidate["learner"] for candidate in schema_v5_candidates]
        if schema_version == 5
        else [selection["learner"] for selection in final_selection["selections"]]
        if schema_version == 4
        else [schema_v6_specs[model]["learner"] for model in models]
        if schema_version == 6
        else [
            normalize_memoryless_learner_binding(registration["memoryless_learner_binding"])[
                "learner"
            ]
        ]
        if schema_version == 7
        else [
            normalize_learner(
                registration.get("learner"),
                schema_version=schema_version,
            )
        ]
    )
    for learner in learners:
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
            if learners[0][field] != expected:
                raise ValueError(f"quick registrations must record learner.{field}={expected}")
    if tier == "development_tuning":
        environment_budgets = {
            environment["id"]: environment["total_steps"] for environment in environments
        }
        if schema_version == 3:
            validate_development_tuning_contract(
                schema_version=schema_version,
                comparison_profile=comparison_profile,
                matrix_kind=matrix_kind,
                candidate_families=candidate_families,
                environments=environment_budgets,
                seeds=seeds,
                quick=registration["quick"],
            )
        elif schema_version == 5:
            validate_compute_aware_tuning_contract(
                schema_version=schema_version,
                comparison_profile=comparison_profile,
                matrix_kind=matrix_kind,
                tuned_families=normalize_tuned_families(registration["tuned_families"]),
                learner_grid=normalize_shared_learner_grid(registration["learner_grid"]),
                environments=environment_budgets,
                seeds=seeds,
                evaluation_episodes_per_env=evaluation_episodes,
                require_gpu=registration["require_gpu"],
                quick=registration["quick"],
            )
        else:  # pragma: no cover - schema reservation checks reject this
            raise AssertionError("unsupported development-tuning registration schema")
    if schema_version == 4 and comparison_profile != "arcmind_shared_comparison":
        raise ValueError(
            "schema-v4 registered final requires comparison_profile 'arcmind_shared_comparison'"
        )
    if schema_version == 6:
        environment_budgets = {
            environment["id"]: environment["total_steps"] for environment in environments
        }
        validate_compute_aware_final_contract(
            schema_version=schema_version,
            comparison_profile=comparison_profile,
            matrix_kind=matrix_kind,
            models=models,
            learner_bindings=registration["learner_bindings"],
            task_model_incidence=registration["task_model_incidence"],
            tuning_selection=registration["tuning_selection"],
            environments=environment_budgets,
            seeds=seeds,
            evaluation_episodes_per_env=evaluation_episodes,
            require_gpu=registration["require_gpu"],
            quick=registration["quick"],
        )
    elif schema_version == 7:
        environment_budgets = {
            environment["id"]: environment["total_steps"] for environment in environments
        }
        validate_compute_aware_upper_reference_contract(
            schema_version=schema_version,
            comparison_profile=comparison_profile,
            matrix_kind=matrix_kind,
            models=models,
            primary_matrix_binding=registration["primary_matrix_binding"],
            memoryless_learner_binding=registration["memoryless_learner_binding"],
            environments=environment_budgets,
            seeds=seeds,
            evaluation_episodes_per_env=evaluation_episodes,
            require_gpu=registration["require_gpu"],
            quick=registration["quick"],
        )
    elif tier == "registered_final" and len(registration["seeds"]) != 30:
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
    candidate_id: str | None = None
    model_family: str | None = None
    learner_id: str | None = None
    tuning_aggregate_sha256: str | None = None
    tuning_completion_index_sha256: str | None = None
    tuning_checksum_manifest_sha256: str | None = None
    tuning_implementation_source_sha256: str | None = None
    learner_binding_mode: str | None = None
    learner_source_model_family: str | None = None
    primary_aggregate_file_sha256: str | None = None
    primary_registration_file_sha256: str | None = None
    primary_manifest_file_sha256: str | None = None
    primary_manifest_internal_sha256: str | None = None
    primary_completion_index_file_sha256: str | None = None
    primary_checksum_manifest_file_sha256: str | None = None
    primary_implementation_source_sha256: str | None = None
    implementation_model = model
    if registration["schema_version"] == 3:
        candidate = next(
            (
                (family["family_id"], item)
                for family in normalize_candidate_families(registration["candidate_families"])
                for item in family["candidates"]
                if item["candidate_id"] == model
            ),
            None,
        )
        if candidate is None:  # pragma: no cover - registration validates
            raise AssertionError(f"unknown tuning candidate: {model}")
        model_family, candidate_spec = candidate
        candidate_id = candidate_spec["candidate_id"]
        implementation_model = candidate_spec["implementation_model"]
        learner = candidate_spec["learner"]
    elif registration["schema_version"] == 5:
        candidate_spec = next(
            (
                candidate
                for candidate in _schema_v5_candidates(registration)
                if candidate["candidate_id"] == model
            ),
            None,
        )
        if candidate_spec is None:  # pragma: no cover - registration validates
            raise AssertionError(f"unknown compute-aware tuning candidate: {model}")
        candidate_id = candidate_spec["candidate_id"]
        model_family = candidate_spec["model_family"]
        learner_id = candidate_spec["learner_id"]
        implementation_model = candidate_spec["implementation_model"]
        learner = candidate_spec["learner"]
    elif registration["schema_version"] == 4:
        binding = normalize_final_selection_binding(registration["tuning_selection"])
        selection = next(
            (
                item
                for item in binding["selections"]
                if item["environment"] == environment["id"]
                and item["implementation_model"] == model
            ),
            None,
        )
        if selection is None:  # pragma: no cover - registration validates
            raise AssertionError(
                f"unknown registered-final selected implementation: {environment['id']}, {model}"
            )
        candidate_id = selection["candidate_id"]
        model_family = selection["model_family"]
        learner = selection["learner"]
        tuning_aggregate_sha256 = binding["aggregate_sha256"]
        tuning_completion_index_sha256 = binding["source_completion_index_sha256"]
        tuning_checksum_manifest_sha256 = binding["source_checksum_manifest_sha256"]
        tuning_implementation_source_sha256 = binding["source_implementation_sha256"]
    elif registration["schema_version"] == 6:
        spec = _schema_v6_final_specs(registration)[model]
        candidate_id = spec["candidate_id"]
        model_family = spec["model_family"]
        learner_id = spec["learner_id"]
        implementation_model = spec["implementation_model"]
        learner_binding_mode = spec["learner_binding_mode"]
        learner_source_model_family = spec["learner_source_model_family"]
        learner = spec["learner"]
        tuning_aggregate_sha256 = spec["tuning_aggregate_sha256"]
        tuning_completion_index_sha256 = spec["tuning_completion_index_sha256"]
        tuning_checksum_manifest_sha256 = spec["tuning_checksum_manifest_sha256"]
        tuning_implementation_source_sha256 = spec["tuning_implementation_source_sha256"]
    elif registration["schema_version"] == 7:
        spec = _schema_v7_final_spec(registration)
        candidate_id = spec["candidate_id"]
        model_family = spec["model_family"]
        learner_id = spec["learner_id"]
        implementation_model = spec["implementation_model"]
        learner_binding_mode = spec["learner_binding_mode"]
        learner_source_model_family = spec["learner_source_model_family"]
        learner = spec["learner"]
        tuning_aggregate_sha256 = spec["tuning_aggregate_sha256"]
        tuning_completion_index_sha256 = spec["tuning_completion_index_sha256"]
        tuning_checksum_manifest_sha256 = spec["tuning_checksum_manifest_sha256"]
        tuning_implementation_source_sha256 = spec["tuning_implementation_source_sha256"]
        primary_aggregate_file_sha256 = spec["primary_aggregate_file_sha256"]
        primary_registration_file_sha256 = spec["primary_registration_file_sha256"]
        primary_manifest_file_sha256 = spec["primary_manifest_file_sha256"]
        primary_manifest_internal_sha256 = spec["primary_manifest_internal_sha256"]
        primary_completion_index_file_sha256 = spec["primary_completion_index_file_sha256"]
        primary_checksum_manifest_file_sha256 = spec["primary_checksum_manifest_file_sha256"]
        primary_implementation_source_sha256 = spec["primary_implementation_source_sha256"]
    else:
        learner = registration["learner"]
    return Namespace(
        environment=environment["id"],
        model=implementation_model,
        candidate_id=candidate_id,
        model_family=model_family,
        learner_id=learner_id,
        tuning_aggregate_sha256=tuning_aggregate_sha256,
        tuning_completion_index_sha256=tuning_completion_index_sha256,
        tuning_checksum_manifest_sha256=tuning_checksum_manifest_sha256,
        tuning_implementation_source_sha256=tuning_implementation_source_sha256,
        learner_binding_mode=learner_binding_mode,
        learner_source_model_family=learner_source_model_family,
        primary_aggregate_file_sha256=primary_aggregate_file_sha256,
        primary_registration_file_sha256=primary_registration_file_sha256,
        primary_manifest_file_sha256=primary_manifest_file_sha256,
        primary_manifest_internal_sha256=primary_manifest_internal_sha256,
        primary_completion_index_file_sha256=primary_completion_index_file_sha256,
        primary_checksum_manifest_file_sha256=primary_checksum_manifest_file_sha256,
        primary_implementation_source_sha256=primary_implementation_source_sha256,
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
    if getattr(args, "candidate_id", None) is not None:
        command.extend(["--candidate-id", args.candidate_id])
        command.extend(["--model-family", args.model_family])
    if getattr(args, "learner_id", None) is not None:
        command.extend(["--learner-id", args.learner_id])
    if getattr(args, "learner_binding_mode", None) is not None:
        command.extend(["--learner-binding-mode", args.learner_binding_mode])
        command.extend(["--learner-source-model-family", args.learner_source_model_family])
    if getattr(args, "tuning_aggregate_sha256", None) is not None:
        command.extend(["--tuning-aggregate-sha256", args.tuning_aggregate_sha256])
        command.extend(
            [
                "--tuning-completion-index-sha256",
                args.tuning_completion_index_sha256,
                "--tuning-checksum-manifest-sha256",
                args.tuning_checksum_manifest_sha256,
                "--tuning-implementation-source-sha256",
                args.tuning_implementation_source_sha256,
            ]
        )
    if getattr(args, "primary_aggregate_file_sha256", None) is not None:
        for option, value in (
            ("--primary-aggregate-file-sha256", args.primary_aggregate_file_sha256),
            (
                "--primary-registration-file-sha256",
                args.primary_registration_file_sha256,
            ),
            ("--primary-manifest-file-sha256", args.primary_manifest_file_sha256),
            (
                "--primary-manifest-internal-sha256",
                args.primary_manifest_internal_sha256,
            ),
            (
                "--primary-completion-index-file-sha256",
                args.primary_completion_index_file_sha256,
            ),
            (
                "--primary-checksum-manifest-file-sha256",
                args.primary_checksum_manifest_file_sha256,
            ),
            (
                "--primary-implementation-source-sha256",
                args.primary_implementation_source_sha256,
            ),
        ):
            command.extend([option, value])
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
    registration_schema_version: int,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    artifact = json.loads(path.read_text(encoding="utf-8"))
    expected_artifact_schema = ARTIFACT_SCHEMA_BY_REGISTRATION.get(registration_schema_version)
    if artifact.get("schema_version") != expected_artifact_schema:
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
    implementation_model = (
        configuration.get("implementation_model")
        if registration_schema_version in {3, 4, 5, 6, 7}
        else model
    )
    if reference_implementation_for_model(implementation_model) is not None:
        try:
            validate_required_reference_implementation(
                implementation_model,
                configuration.get("reference_implementation"),
                field="configuration.reference_implementation",
            )
            validate_required_reference_implementation(
                implementation_model,
                artifact.get("reference_implementation"),
                field="artifact.reference_implementation",
            )
        except ValueError as error:
            raise ExistingArtifactMismatchError(
                f"existing cell reference implementation does not match registry: {path}"
            ) from error
    expected_policy_contract = policy_contract_metadata_for_model(implementation_model)
    if (
        requires_explicit_policy_contract(implementation_model)
        or any(name in configuration for name in expected_policy_contract)
        or any(name in artifact for name in expected_policy_contract)
    ):
        try:
            validate_policy_contract_metadata(
                implementation_model,
                {name: configuration.get(name) for name in expected_policy_contract},
                field="configuration.policy_contract",
            )
            validate_policy_contract_metadata(
                implementation_model,
                {name: artifact.get(name) for name in expected_policy_contract},
                field="artifact.policy_contract",
            )
            validate_policy_core_contract(
                implementation_model,
                configuration.get("policy_core"),
                field="configuration.policy_core",
            )
            validate_policy_core_contract(
                implementation_model,
                artifact.get("policy_core"),
                field="artifact.policy_core",
            )
            if artifact.get("policy_core") != configuration.get("policy_core"):
                raise ValueError("artifact.policy_core does not match configuration.policy_core")
        except ValueError as error:
            raise ExistingArtifactMismatchError(
                f"existing cell policy contract does not match registry: {path}"
            ) from error
    if registration_schema_version in {3, 4, 5, 6, 7}:
        try:
            maximum_episode_steps = configuration.get("evaluation_max_episode_steps")
            validate_causal_transformer_horizon_contract(
                implementation_model,
                configuration.get("policy_core"),
                maximum_episode_steps,
                field="configuration.policy_core",
            )
            validate_causal_transformer_horizon_contract(
                implementation_model,
                artifact.get("policy_core"),
                maximum_episode_steps,
                field="artifact.policy_core",
            )
        except ValueError as error:
            raise ExistingArtifactMismatchError(
                f"existing cell causal attention horizon does not match its registered task: {path}"
            ) from error
    if registration_schema_version == 3:
        candidate_identity = {
            "candidate_id": model,
            "model_family": configuration.get("model_family"),
            "implementation_model": configuration.get("implementation_model"),
        }
        if (
            configuration.get("model") != model
            or configuration.get("candidate_id") != model
            or {field: artifact.get(field) for field in candidate_identity} != candidate_identity
            or artifact.get("implementation_source_sha256")
            != configuration.get("implementation_source", {}).get("sha256")
        ):
            raise ExistingArtifactMismatchError(
                f"existing cell candidate identity does not match its configuration: {path}"
            )
    if registration_schema_version == 5:
        candidate_identity = {
            "candidate_id": model,
            "model_family": configuration.get("model_family"),
            "learner_id": configuration.get("learner_id"),
            "implementation_model": configuration.get("implementation_model"),
        }
        if (
            configuration.get("model") != model
            or configuration.get("candidate_id") != model
            or configuration.get("candidate_id")
            != f"{configuration.get('model_family')}.{configuration.get('learner_id')}"
            or {field: artifact.get(field) for field in candidate_identity} != candidate_identity
            or artifact.get("implementation_source_sha256")
            != configuration.get("implementation_source", {}).get("sha256")
        ):
            raise ExistingArtifactMismatchError(
                "existing compute-aware tuning cell identity does not match "
                f"its configuration: {path}"
            )
    if registration_schema_version == 4:
        if (
            configuration.get("model") != model
            or configuration.get("implementation_model") != model
            or artifact.get("candidate_id") != configuration.get("candidate_id")
            or artifact.get("model_family") != configuration.get("model_family")
            or artifact.get("implementation_model") != model
            or artifact.get("tuning_aggregate_sha256")
            != configuration.get("tuning_aggregate_sha256")
            or artifact.get("tuning_completion_index_sha256")
            != configuration.get("tuning_completion_index_sha256")
            or artifact.get("tuning_checksum_manifest_sha256")
            != configuration.get("tuning_checksum_manifest_sha256")
            or artifact.get("tuning_implementation_source_sha256")
            != configuration.get("tuning_implementation_source_sha256")
            or artifact.get("implementation_source_sha256")
            != configuration.get("implementation_source", {}).get("sha256")
        ):
            raise ExistingArtifactMismatchError(
                f"existing cell final-selection identity does not match its configuration: {path}"
            )
    if registration_schema_version == 6:
        candidate_identity = {
            "candidate_id": configuration.get("candidate_id"),
            "model_family": configuration.get("model_family"),
            "learner_id": configuration.get("learner_id"),
            "implementation_model": model,
            "learner_binding_mode": configuration.get("learner_binding_mode"),
            "learner_source_model_family": configuration.get("learner_source_model_family"),
            "tuning_aggregate_sha256": configuration.get("tuning_aggregate_sha256"),
            "tuning_completion_index_sha256": configuration.get("tuning_completion_index_sha256"),
            "tuning_checksum_manifest_sha256": configuration.get("tuning_checksum_manifest_sha256"),
            "tuning_implementation_source_sha256": configuration.get(
                "tuning_implementation_source_sha256"
            ),
        }
        if (
            configuration.get("model") != model
            or configuration.get("implementation_model") != model
            or configuration.get("candidate_id")
            != (
                f"{configuration.get('learner_source_model_family')}."
                f"{configuration.get('learner_id')}"
            )
            or configuration.get("model_family") != configuration.get("learner_source_model_family")
            or {field: artifact.get(field) for field in candidate_identity} != candidate_identity
            or artifact.get("implementation_source_sha256")
            != configuration.get("implementation_source", {}).get("sha256")
        ):
            raise ExistingArtifactMismatchError(
                "existing compute-aware final cell identity does not match "
                f"its configuration: {path}"
            )
    if registration_schema_version == 7:
        candidate_identity = {
            "candidate_id": configuration.get("candidate_id"),
            "model_family": configuration.get("model_family"),
            "learner_id": configuration.get("learner_id"),
            "implementation_model": model,
            "learner_binding_mode": configuration.get("learner_binding_mode"),
            "learner_source_model_family": configuration.get("learner_source_model_family"),
            "tuning_aggregate_sha256": configuration.get("tuning_aggregate_sha256"),
            "tuning_completion_index_sha256": configuration.get("tuning_completion_index_sha256"),
            "tuning_checksum_manifest_sha256": configuration.get("tuning_checksum_manifest_sha256"),
            "tuning_implementation_source_sha256": configuration.get(
                "tuning_implementation_source_sha256"
            ),
            **{field: configuration.get(field) for field in _SCHEMA_V7_PRIMARY_HASH_FIELDS},
        }
        if (
            model != "memoryless_mlp"
            or configuration.get("model") != model
            or configuration.get("implementation_model") != model
            or configuration.get("candidate_id")
            != f"memoryless_mlp.{configuration.get('learner_id')}"
            or configuration.get("model_family") != "memoryless_mlp"
            or configuration.get("learner_binding_mode") != "selected"
            or configuration.get("learner_source_model_family") != "memoryless_mlp"
            or {field: artifact.get(field) for field in candidate_identity} != candidate_identity
            or artifact.get("implementation_source_sha256")
            != configuration.get("implementation_source", {}).get("sha256")
            or configuration.get("implementation_source", {}).get("sha256")
            != configuration.get("primary_implementation_source_sha256")
        ):
            raise ExistingArtifactMismatchError(
                "existing compute-aware upper-reference cell identity does not "
                f"match its configuration: {path}"
            )
    return artifact


def _preserve_failed_attempt(
    artifact_path: Path,
    output_root: Path,
    stdout: bytes,
    *,
    label: str,
) -> Path:
    attempt_id = uuid.uuid4().hex
    relative_artifact = artifact_path.relative_to(output_root)
    attempt_directory = output_root.with_name(f"{output_root.name}.attempts")
    attempt_artifact = attempt_directory / relative_artifact
    failed_log_path = attempt_artifact.with_name(
        f"{artifact_path.stem}.attempt-{attempt_id}.{label}.log"
    )
    atomic_write_bytes(failed_log_path, stdout)
    if artifact_path.exists():
        failed_artifact_path = attempt_artifact.with_name(
            f"{artifact_path.stem}.attempt-{attempt_id}.{label}.json"
        )
        failed_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.replace(failed_artifact_path)
    return failed_log_path


def _preserve_orphaned_file(
    path: Path,
    *,
    artifact_path: Path,
    output_root: Path,
) -> Path:
    attempt_id = uuid.uuid4().hex
    relative_artifact = artifact_path.relative_to(output_root)
    attempt_directory = output_root.with_name(f"{output_root.name}.attempts")
    attempt_artifact = attempt_directory / relative_artifact
    destination = attempt_artifact.with_name(
        f"{artifact_path.stem}.attempt-{attempt_id}.orphaned{path.suffix}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    path.replace(destination)
    return destination


def _describe_matrix(registration_path: Path) -> dict[str, Any]:
    """Describe one complete matrix without creating or executing artifacts."""

    registration = _load_registration(registration_path.resolve())
    cells: list[dict[str, Any]] = []
    provenance: dict[str, Any] | None = None
    candidate_families = (
        normalize_candidate_families(registration["candidate_families"])
        if registration["schema_version"] == 3
        else ()
    )
    schema_v5_candidates = (
        _schema_v5_candidates(registration) if registration["schema_version"] == 5 else ()
    )
    final_selection = (
        normalize_final_selection_binding(registration["tuning_selection"])
        if registration["schema_version"] == 4
        else None
    )
    panel_selection = (
        normalize_panel_selection_binding(registration["tuning_selection"])
        if registration["schema_version"] == 6
        else None
    )
    schema_v6_specs = (
        _schema_v6_final_specs(registration) if registration["schema_version"] == 6 else {}
    )
    schema_v7_spec = (
        _schema_v7_final_spec(registration) if registration["schema_version"] == 7 else {}
    )
    matrix_models = (
        [
            candidate["candidate_id"]
            for family in candidate_families
            for candidate in family["candidates"]
        ]
        if candidate_families
        else [candidate["candidate_id"] for candidate in schema_v5_candidates]
        if schema_v5_candidates
        else registration["models"]
    )
    candidate_index = {
        candidate["candidate_id"]: {
            "model_family": family["family_id"],
            "implementation_model": candidate["implementation_model"],
        }
        for family in candidate_families
        for candidate in family["candidates"]
    }
    if schema_v5_candidates:
        candidate_index = {
            candidate["candidate_id"]: {
                "model_family": candidate["model_family"],
                "learner_id": candidate["learner_id"],
                "implementation_model": candidate["implementation_model"],
            }
            for candidate in schema_v5_candidates
        }
    if final_selection is not None:
        candidate_index = {
            selection["implementation_model"]: {
                "candidate_id": selection["candidate_id"],
                "model_family": selection["model_family"],
                "implementation_model": selection["implementation_model"],
                "tuning_aggregate_sha256": final_selection["aggregate_sha256"],
                "tuning_completion_index_sha256": final_selection["source_completion_index_sha256"],
                "tuning_checksum_manifest_sha256": final_selection[
                    "source_checksum_manifest_sha256"
                ],
                "tuning_implementation_source_sha256": final_selection[
                    "source_implementation_sha256"
                ],
            }
            for selection in final_selection["selections"]
        }
    if panel_selection is not None:
        candidate_index = {
            model: {name: value for name, value in spec.items() if name != "learner"}
            for model, spec in schema_v6_specs.items()
        }
    if schema_v7_spec:
        candidate_index = {
            "memoryless_mlp": {
                name: value
                for name, value in schema_v7_spec.items()
                if name not in {"learner", "raw_matrix_path", "aggregate_path"}
            }
        }

    matrix_inventory = _matrix_cell_identities(registration)
    for environment, model, seed in matrix_inventory:
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
                **(
                    {"implementation_source": configuration["implementation_source"]}
                    if registration["schema_version"] in {3, 4, 5, 6, 7}
                    else {}
                ),
            }
        else:
            current_provenance = {
                "git": description["runtime"]["git"],
                "dependency_lock_sha256": description["configuration"]["dependency_lock_sha256"],
                "pobax_commit": description["configuration"]["pobax_commit"],
                "navix_commit": description["configuration"]["navix_commit"],
                "runtime_contract": description["configuration"]["runtime_contract"],
                **(
                    {"implementation_source": description["configuration"]["implementation_source"]}
                    if registration["schema_version"] in {3, 4, 5, 6, 7}
                    else {}
                ),
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
        cell = {
            "cell_id": cell_id,
            "environment": environment["id"],
            "model": model,
            "seed": seed,
            "configuration_sha256": configuration_sha256,
            "artifact_path": relative_path.as_posix(),
        }
        if registration["schema_version"] in {3, 4, 5, 6, 7}:
            cell.update(candidate_index[model])
            cell["implementation_source_sha256"] = description["configuration"][
                "implementation_source"
            ]["sha256"]
        cells.append(cell)

    validate_unique_cell_ids(cell["cell_id"] for cell in cells)
    if provenance is None:
        raise AssertionError("matrix description produced no provenance")
    if registration["schema_version"] in {4, 6}:
        validated_binding, tuning_aggregate = (
            _validate_final_tuning_selection(registration)
            if registration["schema_version"] == 4
            else _validate_compute_aware_final_tuning_selection(registration)
        )
        validate_final_provenance_against_tuning(
            binding=validated_binding,
            tuning_provenance=tuning_aggregate["provenance"],
            final_provenance=provenance,
        )
    if registration["schema_version"] == 7:
        primary_binding, _, primary_aggregate = _validate_compute_aware_upper_primary(registration)
        validate_final_provenance_against_tuning(
            binding={
                "source_implementation_sha256": primary_binding[
                    "primary_implementation_source_sha256"
                ]
            },
            tuning_provenance=primary_aggregate["provenance"],
            final_provenance=provenance,
        )
    manifest_without_hash = {
        "schema_version": registration["schema_version"],
        "status": "frozen",
        "matrix_kind": registration["matrix_kind"],
        "models": matrix_models,
        "environments": [environment["id"] for environment in registration["environments"]],
        "seeds": registration["seeds"],
        "provenance": provenance,
        "cells": cells,
    }
    if registration["schema_version"] == 3:
        manifest_without_hash["candidate_families"] = registration["candidate_families"]
    if registration["schema_version"] == 5:
        manifest_without_hash["tuned_families"] = registration["tuned_families"]
        manifest_without_hash["learner_grid"] = registration["learner_grid"]
    if registration["schema_version"] == 4:
        manifest_without_hash["tuning_selection"] = registration["tuning_selection"]
    if registration["schema_version"] == 6:
        manifest_without_hash["tuning_selection"] = registration["tuning_selection"]
        manifest_without_hash["learner_bindings"] = registration["learner_bindings"]
        manifest_without_hash["task_model_incidence"] = registration["task_model_incidence"]
    if registration["schema_version"] == 7:
        manifest_without_hash["primary_matrix_binding"] = registration["primary_matrix_binding"]
        manifest_without_hash["memoryless_learner_binding"] = registration[
            "memoryless_learner_binding"
        ]
    if registration["schema_version"] in {4, 6, 7}:
        manifest_without_hash["registration_sha256"] = hashlib.sha256(
            canonical_json_bytes(registration) + b"\n"
        ).hexdigest()
    manifest_sha256 = canonical_json_sha256(manifest_without_hash)
    manifest = {
        **manifest_without_hash,
        "manifest_sha256": manifest_sha256,
    }

    return {
        "registration": registration,
        "matrix_inventory": matrix_inventory,
        "cells": cells,
        "provenance": provenance,
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "expected_status": EVIDENCE_STATUS[registration["evidence_tier"]],
    }


def _validate_shard_selection(*, shard_count: int, shard_index: int) -> None:
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count <= 0:
        raise ValueError("shard_count must be a positive integer")
    if (
        isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or shard_index < 0
        or shard_index >= shard_count
    ):
        raise ValueError("shard_index must be an integer in [0, shard_count)")


def _shard_cells(
    cells: list[dict[str, Any]],
    *,
    shard_count: int,
    shard_index: int,
) -> tuple[dict[str, Any], ...]:
    """Select one stable, disjoint index-modulo shard from the frozen cell order."""

    _validate_shard_selection(shard_count=shard_count, shard_index=shard_index)
    return tuple(cell for index, cell in enumerate(cells) if index % shard_count == shard_index)


def _validate_sharded_runtime(description: dict[str, Any], *, shard_count: int) -> None:
    if shard_count == 1 or not description["registration"]["require_gpu"]:
        return
    runtime = description["provenance"].get("runtime_contract")
    if not isinstance(runtime, dict):
        raise RuntimeError("sharded GPU execution requires a structured runtime contract")
    devices = runtime.get("devices")
    if (
        runtime.get("jax_backend") != "gpu"
        or not isinstance(devices, list)
        or len(devices) != 1
        or not isinstance(devices[0], dict)
        or devices[0].get("platform") != "gpu"
    ):
        raise RuntimeError("sharded GPU execution requires exactly one visible JAX GPU per process")


def _validate_matrix_root_inventory(
    output_root: Path,
    cells: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    require_all_cells: bool,
    allow_final_files: bool,
    allow_shard_files: bool = False,
    require_role_files: bool = False,
) -> None:
    """Reject files outside the exact raw-root role for this execution."""

    if allow_final_files and allow_shard_files:
        raise ValueError("a matrix root cannot have canonical and shard finalization files")

    allowed = {"registration.json", "frozen_manifest.json"}
    for cell in cells:
        artifact = Path(cell["artifact_path"])
        allowed.add(artifact.as_posix())
        allowed.add(artifact.with_suffix(".log").as_posix())
    if allow_final_files:
        allowed.update({"completion_index.json", "checksums.sha256"})
    if allow_shard_files:
        allowed.update({"shard_completion.json", "shard_checksums.sha256"})

    actual = set()
    for path in output_root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"matrix roots must not contain symbolic links: {path}")
        if path.is_file():
            actual.add(path.relative_to(output_root).as_posix())
    unexpected = sorted(actual - allowed)
    if unexpected:
        raise RuntimeError(f"matrix root contains files outside its frozen role: {unexpected}")
    if require_all_cells:
        missing = sorted(allowed - actual)
        if not require_role_files:
            missing = [
                path
                for path in missing
                if path
                not in {
                    "completion_index.json",
                    "checksums.sha256",
                    "shard_completion.json",
                    "shard_checksums.sha256",
                }
            ]
        if missing:
            raise RuntimeError(f"matrix root is missing required frozen files: {missing}")


def _cell_lookup(description: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {
        (cell["environment"], cell["model"], cell["seed"]): cell for cell in description["cells"]
    }


def _validate_completed_cell(
    *,
    cell: dict[str, Any],
    artifact_path: Path,
    log_path: Path,
    output_root: Path,
    description: dict[str, Any],
) -> dict[str, Any]:
    artifact = _load_matching_artifact(
        artifact_path,
        expected_status=description["expected_status"],
        environment=cell["environment"],
        model=cell["model"],
        seed=cell["seed"],
        configuration_sha256=cell["configuration_sha256"],
        manifest_sha256=description["manifest_sha256"],
        cell_id=cell["cell_id"],
        provenance=description["provenance"],
        registration_schema_version=description["registration"]["schema_version"],
    )
    if artifact is None:
        raise RuntimeError(f"completed cell artifact is missing: {artifact_path}")
    if not log_path.is_file():
        raise RuntimeError(f"completed cell is missing its immutable log: {log_path}")
    return {
        **cell,
        "artifact_sha256": sha256_file(artifact_path),
        "log_path": log_path.relative_to(output_root).as_posix(),
        "log_sha256": sha256_file(log_path),
    }


def _finalize_complete_matrix(
    description: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Validate every frozen cell before creating the immutable final indexes."""

    _validate_matrix_root_inventory(
        output_root,
        description["cells"],
        require_all_cells=True,
        allow_final_files=True,
    )
    completion_path = output_root / "completion_index.json"
    checksum_path = output_root / "checksums.sha256"
    if checksum_path.exists() and not completion_path.is_file():
        raise RuntimeError("canonical checksum exists without its completion index")

    completed_cells: list[dict[str, Any]] = []
    for cell in description["cells"]:
        artifact_path = output_root / cell["artifact_path"]
        log_path = artifact_path.with_suffix(".log")
        completed = _validate_completed_cell(
            cell=cell,
            artifact_path=artifact_path,
            log_path=log_path,
            output_root=output_root,
            description=description,
        )
        completed_cells.append(completed)

    completion_index = {
        "schema_version": 1,
        "status": "complete",
        "manifest_sha256": description["manifest_sha256"],
        "planned_cells": len(description["cells"]),
        "completed_cells": len(completed_cells),
        "cells": completed_cells,
    }
    atomic_write_json(completion_path, completion_index)
    write_checksum_manifest(output_root)
    _validate_matrix_root_inventory(
        output_root,
        description["cells"],
        require_all_cells=True,
        allow_final_files=True,
        require_role_files=True,
    )
    return completion_index


def _execute_matrix_locked(
    registration_path: Path,
    output_root: Path,
    *,
    shard_count: int = 1,
    shard_index: int = 0,
) -> dict[str, Any]:
    """Describe, freeze, and execute a complete matrix or one isolated shard."""

    description = _describe_matrix(registration_path)
    _validate_sharded_runtime(description, shard_count=shard_count)
    registration = description["registration"]
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    assigned_cells = _shard_cells(
        description["cells"],
        shard_count=shard_count,
        shard_index=shard_index,
    )
    _validate_matrix_root_inventory(
        output_root,
        assigned_cells,
        require_all_cells=False,
        allow_final_files=shard_count == 1,
        allow_shard_files=shard_count > 1,
    )
    atomic_write_json(output_root / "registration.json", registration)
    atomic_write_json(output_root / "frozen_manifest.json", description["manifest"])

    cell_lookup = _cell_lookup(description)
    completed_cells: list[dict[str, Any]] = []
    assigned_ids = {cell["cell_id"] for cell in assigned_cells}
    for environment, model, seed in description["matrix_inventory"]:
        cell = cell_lookup[(environment["id"], model, seed)]
        if cell["cell_id"] not in assigned_ids:
            continue
        artifact_path = output_root / cell["artifact_path"]
        log_path = artifact_path.with_suffix(".log")
        if artifact_path.exists() and not log_path.exists():
            _preserve_orphaned_file(
                artifact_path,
                artifact_path=artifact_path,
                output_root=output_root,
            )
        if log_path.exists() and not artifact_path.exists():
            _preserve_orphaned_file(
                log_path,
                artifact_path=artifact_path,
                output_root=output_root,
            )
        artifact = _load_matching_artifact(
            artifact_path,
            expected_status=description["expected_status"],
            environment=environment["id"],
            model=model,
            seed=seed,
            configuration_sha256=cell["configuration_sha256"],
            manifest_sha256=description["manifest_sha256"],
            cell_id=cell["cell_id"],
            provenance=description["provenance"],
            registration_schema_version=registration["schema_version"],
        )
        if artifact is None:
            cell_args = _cell_namespace(
                registration,
                environment=environment,
                model=model,
                seed=seed,
                output=artifact_path,
                manifest_sha256=description["manifest_sha256"],
                cell_id=cell["cell_id"],
                describe_only=False,
            )
            process = subprocess.run(
                _command_for_cell(cell_args),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if process.returncode != 0:
                failed_log_path = _preserve_failed_attempt(
                    artifact_path,
                    output_root,
                    process.stdout,
                    label="failed",
                )
                raise RuntimeError(
                    f"cell failed with exit code {process.returncode}: "
                    f"{cell['cell_id']}; log={failed_log_path}"
                )
            try:
                artifact = _load_matching_artifact(
                    artifact_path,
                    expected_status=description["expected_status"],
                    environment=environment["id"],
                    model=model,
                    seed=seed,
                    configuration_sha256=cell["configuration_sha256"],
                    manifest_sha256=description["manifest_sha256"],
                    cell_id=cell["cell_id"],
                    provenance=description["provenance"],
                    registration_schema_version=registration["schema_version"],
                )
            except (
                ExistingArtifactMismatchError,
                OSError,
                TypeError,
                UnicodeError,
                ValueError,
            ) as error:
                failed_log_path = _preserve_failed_attempt(
                    artifact_path,
                    output_root,
                    process.stdout,
                    label="failed",
                )
                raise RuntimeError(
                    f"cell returned success with an invalid artifact: "
                    f"{cell['cell_id']}; log={failed_log_path}"
                ) from error
            if artifact is None:
                failed_log_path = _preserve_failed_attempt(
                    artifact_path,
                    output_root,
                    process.stdout,
                    label="failed",
                )
                raise RuntimeError(
                    f"cell completed without creating {artifact_path}; log={failed_log_path}"
                )
            atomic_write_bytes(log_path, process.stdout)
        if not log_path.is_file():
            raise RuntimeError(f"completed cell is missing its immutable log: {log_path}")
        completed = _validate_completed_cell(
            cell=cell,
            artifact_path=artifact_path,
            log_path=log_path,
            output_root=output_root,
            description=description,
        )
        completed_cells.append(completed)

    if shard_count > 1:
        _validate_matrix_root_inventory(
            output_root,
            assigned_cells,
            require_all_cells=True,
            allow_final_files=False,
            allow_shard_files=True,
        )
        completion_path = output_root / "shard_completion.json"
        checksum_path = output_root / "shard_checksums.sha256"
        if checksum_path.exists() and not completion_path.is_file():
            raise RuntimeError("shard checksum exists without its completion index")
        shard_completion = {
            "schema_version": 1,
            "status": "shard_complete",
            "partition_algorithm": _SHARD_PARTITION_ALGORITHM,
            "registration_file_sha256": sha256_file(output_root / "registration.json"),
            "manifest_file_sha256": sha256_file(output_root / "frozen_manifest.json"),
            "manifest_sha256": description["manifest_sha256"],
            "shard_count": shard_count,
            "shard_index": shard_index,
            "planned_cells": len(description["cells"]),
            "assigned_cells": len(assigned_cells),
            "completed_cells": len(completed_cells),
            "cells": completed_cells,
        }
        atomic_write_json(completion_path, shard_completion)
        write_checksum_manifest(output_root, filename="shard_checksums.sha256")
        _validate_matrix_root_inventory(
            output_root,
            assigned_cells,
            require_all_cells=True,
            allow_final_files=False,
            allow_shard_files=True,
            require_role_files=True,
        )
        return shard_completion

    return _finalize_complete_matrix(description, output_root)


def execute_matrix(
    registration_path: Path,
    output_root: Path,
    *,
    shard_count: int = 1,
    shard_index: int = 0,
) -> dict[str, Any]:
    """Execute one matrix role while excluding duplicate writers to that root."""

    _validate_shard_selection(shard_count=shard_count, shard_index=shard_index)
    resolved_output = output_root.resolve()
    lock_path = matrix_process_lock_path(resolved_output)
    with exclusive_process_lock(lock_path):
        return _execute_matrix_locked(
            registration_path,
            resolved_output,
            shard_count=shard_count,
            shard_index=shard_index,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--shard-index", type=int)
    args = parser.parse_args()
    if (args.shard_count is None) != (args.shard_index is None):
        parser.error("--shard-count and --shard-index must be supplied together")
    shard_count = 1 if args.shard_count is None else args.shard_count
    shard_index = 0 if args.shard_index is None else args.shard_index
    print(
        json.dumps(
            execute_matrix(
                args.registration,
                args.output_root,
                shard_count=shard_count,
                shard_index=shard_index,
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
