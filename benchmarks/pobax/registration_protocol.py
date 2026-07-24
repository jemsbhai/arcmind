"""Schema-specific validation for frozen POBAX experiment registrations."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from benchmarks.pobax.implementation_provenance import normalize_implementation_source
from benchmarks.pobax.model_registry import (
    validate_model_environment_contract,
    validate_model_evidence_tier,
    validate_policy_model_id,
)

REGISTRATION_FIELDS_V1 = {
    "schema_version",
    "status",
    "evidence_tier",
    "matrix_kind",
    "models",
    "environments",
    "seeds",
    "learner",
    "evaluation_episodes_per_env",
    "require_gpu",
    "quick",
}
REGISTRATION_FIELDS_V2 = REGISTRATION_FIELDS_V1 | {"comparison_profile"}
REGISTRATION_FIELDS_V3 = (REGISTRATION_FIELDS_V2 - {"models", "learner"}) | {"candidate_families"}
REGISTRATION_FIELDS_V4 = (REGISTRATION_FIELDS_V2 - {"learner"}) | {"tuning_selection"}
REGISTRATION_FIELDS_V5 = (REGISTRATION_FIELDS_V2 - {"models", "learner"}) | {
    "tuned_families",
    "learner_grid",
}
REGISTRATION_FIELDS_V6 = (REGISTRATION_FIELDS_V2 - {"learner"}) | {
    "learner_bindings",
    "task_model_incidence",
    "tuning_selection",
}
LEARNER_FIELDS_V1 = {
    "num_envs",
    "rollout_steps",
    "update_epochs",
    "learning_rate",
}
LEARNER_FIELDS_V2 = LEARNER_FIELDS_V1 | {
    "num_minibatches",
    "gae_lambda",
    "entropy_coefficient",
    "anneal_learning_rate",
}
COMPARISON_PROFILES = {
    "arcmind_shared_comparison": "exact",
    "pobax_author_semantics": "floor",
}
_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FINAL_SELECTION_FIELDS = {
    "environment",
    "model_family",
    "implementation_model",
    "candidate_id",
    "learner",
    "implementation_source_sha256",
}
_PANEL_SELECTION_FIELDS = {
    "model_family",
    "implementation_model",
    "candidate_id",
    "learner_id",
    "learner",
    "implementation_source_sha256",
}
COMPUTE_AWARE_TUNING_PANEL = (
    ("tmaze_10", 1_000_000),
    ("rocksample_11_11", 1_000_000),
)
COMPUTE_AWARE_TUNING_SEEDS = (4409, 5519, 6637)
COMPUTE_AWARE_FINAL_SEEDS = tuple(range(10_000, 10_010))
PUBLISHED_PRIMARY_TRAIN_STEPS = {
    "tmaze_10": 1_000_000,
    "rocksample_11_11": 5_000_000,
    "battleship_10": 10_000_000,
    "Walker-V-v0": 50_000_000,
    "HalfCheetah-V-v0": 50_000_000,
    "Navix-DMLab-Maze-01-v0": 10_000_000,
}
PUBLISHED_TUNING_SEED_COUNTS = {
    "tmaze_10": 5,
    "rocksample_11_11": 5,
    "battleship_10": 10,
    "Walker-V-v0": 5,
    "HalfCheetah-V-v0": 5,
    "Navix-DMLab-Maze-01-v0": 5,
}


def registration_fields(schema_version: object) -> set[str]:
    """Return the exact top-level field set for a supported registration."""

    if isinstance(schema_version, bool):
        raise ValueError("registration schema_version must be 1, 2, 3, 4, 5, or 6")
    if schema_version == 1:
        return REGISTRATION_FIELDS_V1
    if schema_version == 2:
        return REGISTRATION_FIELDS_V2
    if schema_version == 3:
        return REGISTRATION_FIELDS_V3
    if schema_version == 4:
        return REGISTRATION_FIELDS_V4
    if schema_version == 5:
        return REGISTRATION_FIELDS_V5
    if schema_version == 6:
        return REGISTRATION_FIELDS_V6
    raise ValueError("registration schema_version must be 1, 2, 3, 4, 5, or 6")


def learner_fields(schema_version: int) -> set[str]:
    """Return the exact learner field set for a supported registration."""

    if schema_version == 1:
        return LEARNER_FIELDS_V1
    if schema_version in {2, 3, 4, 5, 6}:
        return LEARNER_FIELDS_V2
    raise ValueError("registration schema_version must be 1, 2, 3, 4, 5, or 6")


def validate_comparison_profile(registration: Mapping[str, Any]) -> str | None:
    """Validate and return the v2 comparison profile, or None for v1."""

    schema_version = registration.get("schema_version")
    if schema_version == 1:
        return None
    profile = registration.get("comparison_profile")
    if profile not in COMPARISON_PROFILES:
        raise ValueError(
            "comparison_profile must be 'pobax_author_semantics' or 'arcmind_shared_comparison'"
        )
    return str(profile)


def normalize_learner(
    learner: object,
    *,
    schema_version: int,
) -> dict[str, int | float | bool]:
    """Fail closed on one schema-specific learner configuration."""

    expected_fields = learner_fields(schema_version)
    if not isinstance(learner, Mapping) or set(learner) != expected_fields:
        actual_fields = set(learner) if isinstance(learner, Mapping) else set()
        raise ValueError(
            "learner has wrong fields: "
            f"missing={sorted(expected_fields - actual_fields)}, "
            f"extra={sorted(actual_fields - expected_fields)}"
        )
    normalized: dict[str, int | float | bool] = {}
    integer_fields = ["num_envs", "rollout_steps", "update_epochs"]
    if schema_version in {2, 3, 4, 5, 6}:
        integer_fields.append("num_minibatches")
    for field in integer_fields:
        value = learner[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"learner.{field} must be a positive integer")
        normalized[field] = value
    if schema_version in {2, 3, 4, 5, 6} and (
        normalized["num_envs"] % normalized["num_minibatches"] != 0
    ):
        raise ValueError("learner.num_envs must be divisible by learner.num_minibatches")

    learning_rate = learner["learning_rate"]
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise ValueError("learner.learning_rate must be positive and finite")
    normalized["learning_rate"] = float(learning_rate)

    if schema_version in {2, 3, 4, 5, 6}:
        gae_lambda = learner["gae_lambda"]
        if (
            isinstance(gae_lambda, bool)
            or not isinstance(gae_lambda, (int, float))
            or not math.isfinite(float(gae_lambda))
            or not 0.0 <= float(gae_lambda) <= 1.0
        ):
            raise ValueError("learner.gae_lambda must be finite and in [0, 1]")
        normalized["gae_lambda"] = float(gae_lambda)
        entropy = learner["entropy_coefficient"]
        if (
            isinstance(entropy, bool)
            or not isinstance(entropy, (int, float))
            or not math.isfinite(float(entropy))
            or entropy < 0
        ):
            raise ValueError("learner.entropy_coefficient must be non-negative and finite")
        normalized["entropy_coefficient"] = float(entropy)
        anneal = learner["anneal_learning_rate"]
        if not isinstance(anneal, bool):
            raise ValueError("learner.anneal_learning_rate must be a boolean")
        normalized["anneal_learning_rate"] = anneal
    return normalized


def _normalized_repository_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or "\\" in value:
        raise ValueError(f"{field} must be a repository-relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field} must be a normalized relative path")
    return path.as_posix()


def normalize_final_selection_binding(value: object) -> dict[str, Any]:
    """Validate one schema-v4 tuning-selection binding."""

    expected_fields = {
        "raw_matrix_path",
        "aggregate_path",
        "aggregate_sha256",
        "source_registration_sha256",
        "source_manifest_sha256",
        "source_completion_index_sha256",
        "source_checksum_manifest_sha256",
        "source_implementation_sha256",
        "selections",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError(f"tuning_selection has wrong fields: expected={sorted(expected_fields)}")
    hashes: dict[str, str] = {}
    for name in (
        "aggregate_sha256",
        "source_registration_sha256",
        "source_manifest_sha256",
        "source_completion_index_sha256",
        "source_checksum_manifest_sha256",
        "source_implementation_sha256",
    ):
        hash_value = value[name]
        if not isinstance(hash_value, str) or not _SHA256_PATTERN.fullmatch(hash_value):
            raise ValueError(f"tuning_selection.{name} must be a lowercase SHA256")
        hashes[name] = hash_value
    raw_selections = value["selections"]
    if not isinstance(raw_selections, list) or not raw_selections:
        raise ValueError("tuning_selection.selections must be a non-empty list")
    selections: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    implementations: set[tuple[str, str]] = set()
    for index, raw_selection in enumerate(raw_selections):
        field = f"tuning_selection.selections[{index}]"
        if not isinstance(raw_selection, Mapping) or set(raw_selection) != _FINAL_SELECTION_FIELDS:
            raise ValueError(
                f"{field} must contain exactly environment, model_family, "
                "implementation_model, candidate_id, learner, and "
                "implementation_source_sha256"
            )
        environment = raw_selection["environment"]
        if not isinstance(environment, str) or not environment:
            raise ValueError(f"{field}.environment must be a non-empty string")
        model_family = raw_selection["model_family"]
        implementation_model = raw_selection["implementation_model"]
        candidate_id = raw_selection["candidate_id"]
        if (
            not isinstance(model_family, str)
            or not _IDENTIFIER_PATTERN.fullmatch(model_family)
            or not isinstance(implementation_model, str)
            or not _IDENTIFIER_PATTERN.fullmatch(implementation_model)
            or not isinstance(candidate_id, str)
            or not _IDENTIFIER_PATTERN.fullmatch(candidate_id)
            or not candidate_id.startswith(f"{model_family}.")
        ):
            raise ValueError(f"{field} contains an invalid candidate identity")
        validate_policy_model_id(
            implementation_model,
            field=f"{field}.implementation_model",
        )
        validate_model_evidence_tier(
            implementation_model,
            "registered_final",
            field=f"{field}.implementation_model",
        )
        validate_model_environment_contract(
            implementation_model,
            environment,
            field=f"{field}.implementation_model",
        )
        identity = (environment, model_family)
        implementation_identity = (environment, implementation_model)
        if identity in identities:
            raise ValueError(f"{field} duplicates a model-family selection")
        if implementation_identity in implementations:
            raise ValueError(f"{field} duplicates an implementation-model selection")
        identities.add(identity)
        implementations.add(implementation_identity)
        implementation_source_sha256 = raw_selection["implementation_source_sha256"]
        if not isinstance(implementation_source_sha256, str) or not _SHA256_PATTERN.fullmatch(
            implementation_source_sha256
        ):
            raise ValueError(f"{field}.implementation_source_sha256 must be a lowercase SHA256")
        selections.append(
            {
                "environment": environment,
                "model_family": model_family,
                "implementation_model": implementation_model,
                "candidate_id": candidate_id,
                "learner": normalize_learner(raw_selection["learner"], schema_version=4),
                "implementation_source_sha256": implementation_source_sha256,
            }
        )
    return {
        "raw_matrix_path": _normalized_repository_path(
            value["raw_matrix_path"],
            field="tuning_selection.raw_matrix_path",
        ),
        "aggregate_path": _normalized_repository_path(
            value["aggregate_path"],
            field="tuning_selection.aggregate_path",
        ),
        **hashes,
        "selections": tuple(selections),
    }


def validate_final_selection_against_aggregate(
    binding: Mapping[str, Any],
    aggregate: object,
    *,
    models: tuple[str, ...] | list[str],
    environments: tuple[str, ...] | list[str],
    final_seeds: tuple[int, ...] | list[int],
) -> None:
    """Verify schema-v4 selections against one immutable tuning aggregate."""

    if not isinstance(aggregate, Mapping):
        raise ValueError("tuning selection aggregate must be a JSON object")
    if (
        aggregate.get("schema_version") != 1
        or aggregate.get("status") != "development_tuning_selection_aggregate_not_for_paper"
        or aggregate.get("evidence_tier") != "development_tuning"
        or aggregate.get("matrix_kind") != "hyperparameter_selection"
        or aggregate.get("not_for_paper") is not True
    ):
        raise ValueError("tuning selection aggregate has the wrong evidence identity")
    integrity = aggregate.get("integrity_indexes")
    semantics = aggregate.get("frozen_semantic_contract")
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("completion_index_present_and_validated") is not True
        or integrity.get("checksums_present_and_validated") is not True
        or not isinstance(semantics, Mapping)
        or semantics.get("environment_source_in_every_configuration") is not True
        or semantics.get("parameter_contract_in_every_configuration") is not True
        or semantics.get("artifact_parameter_contract_validated") is not True
    ):
        raise ValueError("tuning selection aggregate has an invalid integrity contract")
    eligibility = aggregate.get("selection_eligibility")
    if not isinstance(eligibility, Mapping) or (
        eligibility.get("eligible_for_hyperparameter_selection") is not True
        or eligibility.get("eligible_for_architecture_selection") is not False
        or eligibility.get("eligible_for_checkpoint_selection") is not False
        or eligibility.get("eligible_for_registered_final_evidence") is not False
        or eligibility.get("eligible_for_paper_performance_claims") is not False
        or eligibility.get("selection_scope") != "candidate_within_model_family_and_environment"
    ):
        raise ValueError("tuning selection aggregate has an invalid eligibility contract")
    for binding_field, aggregate_field in (
        ("source_registration_sha256", "registration_sha256"),
        ("source_manifest_sha256", "matrix_manifest_sha256"),
        ("source_completion_index_sha256", "completion_index_sha256"),
        ("source_checksum_manifest_sha256", "checksum_manifest_sha256"),
    ):
        if aggregate.get(aggregate_field) != binding[binding_field]:
            raise ValueError(
                f"tuning selection {binding_field} drifts from aggregate {aggregate_field}"
            )
    tuning_provenance = aggregate.get("provenance")
    if not isinstance(tuning_provenance, Mapping):
        raise ValueError("tuning selection aggregate is missing provenance")
    try:
        implementation_source = normalize_implementation_source(
            tuning_provenance.get("implementation_source")
        )
    except ValueError as error:
        raise ValueError(
            "tuning selection aggregate has invalid implementation provenance"
        ) from error
    if implementation_source["sha256"] != binding["source_implementation_sha256"]:
        raise ValueError("tuning selection implementation source hash drifts from aggregate")
    if aggregate.get("environments") != list(environments) or len(environments) != 1:
        raise ValueError(
            "schema-v4 registered final requires the exact single environment "
            "from its tuning aggregate"
        )
    tuning_seeds = aggregate.get("seeds")
    if (
        not isinstance(tuning_seeds, list)
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in tuning_seeds)
        or len(set(tuning_seeds)) != len(tuning_seeds)
    ):
        raise ValueError("tuning selection aggregate has an invalid seed manifest")
    overlap = sorted(set(tuning_seeds) & set(final_seeds))
    if overlap:
        raise ValueError(
            f"registered-final seeds must be disjoint from tuning seeds: overlap={overlap}"
        )
    candidate_selection = aggregate.get("candidate_selection")
    groups = aggregate.get("groups")
    if not isinstance(candidate_selection, list) or not isinstance(groups, list):
        raise ValueError("tuning selection aggregate is missing candidate selections or groups")
    selections = tuple(binding["selections"])
    if [selection["implementation_model"] for selection in selections] != list(models):
        raise ValueError(
            "registered-final models must exactly match bound implementation models in order"
        )
    expected_identities = {
        (selection["environment"], selection["model_family"]) for selection in selections
    }
    aggregate_identities: set[tuple[object, object]] = set()
    aggregate_identity_order: list[tuple[object, object]] = []
    for raw_selection in candidate_selection:
        if not isinstance(raw_selection, Mapping):
            raise ValueError("tuning aggregate candidate selections must be objects")
        identity = (
            raw_selection.get("environment"),
            raw_selection.get("model_family"),
        )
        if identity in aggregate_identities:
            raise ValueError("tuning aggregate contains duplicate model-family selections")
        aggregate_identities.add(identity)
        aggregate_identity_order.append(identity)
    if aggregate_identities != expected_identities:
        raise ValueError(
            "registered-final selection set must exactly match tuning aggregate winners"
        )
    if aggregate_identity_order != [
        (selection["environment"], selection["model_family"]) for selection in selections
    ]:
        raise ValueError("registered-final selections must preserve tuning aggregate family order")
    for selection in selections:
        identity = (selection["environment"], selection["model_family"])
        winner = next(
            item
            for item in candidate_selection
            if (item.get("environment"), item.get("model_family")) == identity
        )
        ranking = winner.get("ranking")
        if (
            winner.get("winner_candidate_id") != selection["candidate_id"]
            or winner.get("implementation_model") != selection["implementation_model"]
            or not isinstance(ranking, list)
            or not ranking
            or ranking[0].get("rank") != 1
            or ranking[0].get("candidate_id") != selection["candidate_id"]
        ):
            raise ValueError(
                "registered-final candidate does not match the tuning aggregate winner: "
                f"environment={identity[0]!r}, model_family={identity[1]!r}"
            )
        matching_groups = [
            group
            for group in groups
            if isinstance(group, Mapping)
            and group.get("environment") == selection["environment"]
            and group.get("candidate_id") == selection["candidate_id"]
        ]
        if len(matching_groups) != 1:
            raise ValueError("tuning aggregate winner must have exactly one candidate group")
        winner_group = matching_groups[0]
        try:
            aggregate_learner = normalize_learner(
                winner_group.get("learner"),
                schema_version=4,
            )
        except ValueError as error:
            raise ValueError("tuning aggregate winner has an invalid learner") from error
        if (
            winner_group.get("model_family") != selection["model_family"]
            or winner_group.get("implementation_model") != selection["implementation_model"]
            or aggregate_learner != selection["learner"]
        ):
            raise ValueError("registered-final learner drifts from the tuning aggregate winner")
        if (
            winner_group.get("implementation_source_sha256")
            != selection["implementation_source_sha256"]
            or selection["implementation_source_sha256"] != binding["source_implementation_sha256"]
        ):
            raise ValueError(
                "registered-final implementation source drifts from the tuning aggregate winner"
            )


def validate_final_provenance_against_tuning(
    *,
    binding: Mapping[str, Any],
    tuning_provenance: object,
    final_provenance: object,
) -> None:
    """Require runtime and implementation equivalence without equal Git commits."""

    if not isinstance(tuning_provenance, Mapping) or not isinstance(final_provenance, Mapping):
        raise ValueError("tuning and final provenance must be objects")
    try:
        tuning_source = normalize_implementation_source(
            tuning_provenance.get("implementation_source")
        )
        final_source = normalize_implementation_source(
            final_provenance.get("implementation_source")
        )
    except ValueError as error:
        raise ValueError("tuning or final implementation provenance is invalid") from error
    if (
        tuning_source != final_source
        or tuning_source["sha256"] != binding["source_implementation_sha256"]
    ):
        raise ValueError("final implementation source drifts from tuning selection source")
    for field in (
        "dependency_lock_sha256",
        "pobax_commit",
        "navix_commit",
        "runtime_contract",
    ):
        if tuning_provenance.get(field) != final_provenance.get(field):
            raise ValueError(f"final provenance drifts from tuning provenance: {field}")


def realized_environment_steps(
    requested_steps: int,
    *,
    num_envs: int,
    rollout_steps: int,
    comparison_profile: str | None,
) -> int:
    """Resolve the realized step count under one registered budget profile."""

    if isinstance(requested_steps, bool) or not isinstance(requested_steps, int):
        raise ValueError("requested environment steps must be an integer")
    steps_per_update = num_envs * rollout_steps
    if requested_steps < steps_per_update:
        raise ValueError("requested environment steps must cover at least one rollout")
    remainder = requested_steps % steps_per_update
    if comparison_profile in {None, "arcmind_shared_comparison"} and remainder:
        raise ValueError(
            "requested environment steps must be exactly divisible by "
            "learner.num_envs * learner.rollout_steps"
        )
    if comparison_profile is not None and comparison_profile not in COMPARISON_PROFILES:
        raise ValueError(f"unsupported comparison_profile: {comparison_profile!r}")
    return (requested_steps // steps_per_update) * steps_per_update


def step_budget_mode(comparison_profile: str | None) -> str:
    """Return the PPO step-budget behavior for one registration profile."""

    if comparison_profile is None:
        return "exact"
    try:
        return COMPARISON_PROFILES[comparison_profile]
    except KeyError as error:
        raise ValueError(f"unsupported comparison_profile: {comparison_profile!r}") from error


def normalize_candidate_families(value: object) -> tuple[dict[str, Any], ...]:
    """Validate schema-v3 tuning candidates and preserve declared order."""

    if not isinstance(value, list) or not value:
        raise ValueError("candidate_families must be a non-empty list")
    families: list[dict[str, Any]] = []
    family_ids: set[str] = set()
    implementation_models: set[str] = set()
    candidate_ids: set[str] = set()
    cardinalities: set[int] = set()
    structural_learner_signatures: set[tuple[int, int, int, int]] = set()
    learner_grids: list[frozenset[tuple[tuple[str, int | float | bool], ...]]] = []
    for family_index, raw_family in enumerate(value):
        field = f"candidate_families[{family_index}]"
        if not isinstance(raw_family, Mapping) or set(raw_family) != {
            "family_id",
            "implementation_model",
            "candidates",
        }:
            raise ValueError(
                f"{field} must contain exactly family_id, implementation_model, and candidates"
            )
        family_id = raw_family["family_id"]
        if (
            not isinstance(family_id, str)
            or not _IDENTIFIER_PATTERN.fullmatch(family_id)
            or family_id in family_ids
        ):
            raise ValueError(f"{field}.family_id must be a unique portable identifier")
        implementation_model = raw_family["implementation_model"]
        if (
            not isinstance(implementation_model, str)
            or not _IDENTIFIER_PATTERN.fullmatch(implementation_model)
            or implementation_model in implementation_models
        ):
            raise ValueError(
                f"{field}.implementation_model must be a unique portable model identifier"
            )
        validate_policy_model_id(
            implementation_model,
            field=f"{field}.implementation_model",
        )
        validate_model_evidence_tier(
            implementation_model,
            "development_tuning",
            field=f"{field}.implementation_model",
        )
        raw_candidates = raw_family["candidates"]
        if not isinstance(raw_candidates, list) or len(raw_candidates) < 2:
            raise ValueError(f"{field}.candidates must contain at least two candidates")
        candidates: list[dict[str, Any]] = []
        learner_signatures: set[tuple[tuple[str, int | float | bool], ...]] = set()
        for candidate_index, raw_candidate in enumerate(raw_candidates):
            candidate_field = f"{field}.candidates[{candidate_index}]"
            if not isinstance(raw_candidate, Mapping) or set(raw_candidate) != {
                "candidate_id",
                "learner",
            }:
                raise ValueError(f"{candidate_field} must contain exactly candidate_id and learner")
            candidate_id = raw_candidate["candidate_id"]
            if (
                not isinstance(candidate_id, str)
                or not _IDENTIFIER_PATTERN.fullmatch(candidate_id)
                or candidate_id in candidate_ids
                or not candidate_id.startswith(f"{family_id}.")
            ):
                raise ValueError(
                    f"{candidate_field}.candidate_id must be globally unique and "
                    f"start with {family_id!r} followed by a dot"
                )
            learner = normalize_learner(raw_candidate["learner"], schema_version=3)
            learner_signature = tuple(sorted(learner.items()))
            if learner_signature in learner_signatures:
                raise ValueError(f"{field} contains duplicate normalized learner configurations")
            learner_signatures.add(learner_signature)
            structural_learner_signatures.add(
                (
                    int(learner["num_envs"]),
                    int(learner["rollout_steps"]),
                    int(learner["update_epochs"]),
                    int(learner["num_minibatches"]),
                )
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "implementation_model": implementation_model,
                    "learner": learner,
                }
            )
            candidate_ids.add(candidate_id)
        learner_grids.append(frozenset(learner_signatures))
        family_ids.add(family_id)
        implementation_models.add(implementation_model)
        cardinalities.add(len(candidates))
        families.append(
            {
                "family_id": family_id,
                "implementation_model": implementation_model,
                "candidates": tuple(candidates),
            }
        )
    if len(cardinalities) != 1:
        raise ValueError(
            "development_tuning requires equal candidate cardinality across model families"
        )
    if len(structural_learner_signatures) != 1:
        raise ValueError(
            "development_tuning requires identical num_envs, rollout_steps, "
            "update_epochs, and num_minibatches across every candidate"
        )
    if any(grid != learner_grids[0] for grid in learner_grids[1:]):
        raise ValueError(
            "development_tuning requires the exact same normalized learner "
            "configuration grid across model families"
        )
    return tuple(families)


def normalize_tuned_families(value: object) -> tuple[dict[str, str], ...]:
    """Validate schema-v5 model families for the shared tuning panel."""

    if not isinstance(value, list) or not value:
        raise ValueError("tuned_families must be a non-empty list")
    families: list[dict[str, str]] = []
    family_ids: set[str] = set()
    implementation_models: set[str] = set()
    for index, raw_family in enumerate(value):
        field = f"tuned_families[{index}]"
        if not isinstance(raw_family, Mapping) or set(raw_family) != {
            "family_id",
            "implementation_model",
        }:
            raise ValueError(
                f"{field} must contain exactly family_id and implementation_model"
            )
        family_id = raw_family["family_id"]
        if (
            not isinstance(family_id, str)
            or not _IDENTIFIER_PATTERN.fullmatch(family_id)
            or family_id in family_ids
        ):
            raise ValueError(f"{field}.family_id must be a unique portable identifier")
        implementation_model = raw_family["implementation_model"]
        if (
            not isinstance(implementation_model, str)
            or not _IDENTIFIER_PATTERN.fullmatch(implementation_model)
            or implementation_model in implementation_models
        ):
            raise ValueError(
                f"{field}.implementation_model must be a unique portable model identifier"
            )
        validate_policy_model_id(
            implementation_model,
            field=f"{field}.implementation_model",
        )
        validate_model_evidence_tier(
            implementation_model,
            "development_tuning",
            field=f"{field}.implementation_model",
        )
        for environment, _ in COMPUTE_AWARE_TUNING_PANEL:
            validate_model_environment_contract(
                implementation_model,
                environment,
                field=f"{field}.implementation_model",
            )
        family_ids.add(family_id)
        implementation_models.add(implementation_model)
        families.append(
            {
                "family_id": family_id,
                "implementation_model": implementation_model,
            }
        )
    return tuple(families)


def normalize_shared_learner_grid(value: object) -> tuple[dict[str, Any], ...]:
    """Validate the one literal learner grid shared by all schema-v5 families."""

    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("learner_grid must contain at least two learners")
    grid: list[dict[str, Any]] = []
    learner_ids: set[str] = set()
    learner_signatures: set[tuple[tuple[str, int | float | bool], ...]] = set()
    structural_signatures: set[tuple[int, int, int, int]] = set()
    for index, raw_candidate in enumerate(value):
        field = f"learner_grid[{index}]"
        if not isinstance(raw_candidate, Mapping) or set(raw_candidate) != {
            "learner_id",
            "learner",
        }:
            raise ValueError(f"{field} must contain exactly learner_id and learner")
        learner_id = raw_candidate["learner_id"]
        if (
            not isinstance(learner_id, str)
            or not _IDENTIFIER_PATTERN.fullmatch(learner_id)
            or learner_id in learner_ids
        ):
            raise ValueError(f"{field}.learner_id must be a unique portable identifier")
        learner = normalize_learner(raw_candidate["learner"], schema_version=5)
        learner_signature = tuple(sorted(learner.items()))
        if learner_signature in learner_signatures:
            raise ValueError("learner_grid contains duplicate normalized learner configurations")
        structural_signatures.add(
            (
                int(learner["num_envs"]),
                int(learner["rollout_steps"]),
                int(learner["update_epochs"]),
                int(learner["num_minibatches"]),
            )
        )
        learner_ids.add(learner_id)
        learner_signatures.add(learner_signature)
        grid.append({"learner_id": learner_id, "learner": learner})
    if len(structural_signatures) != 1:
        raise ValueError(
            "learner_grid requires identical num_envs, rollout_steps, "
            "update_epochs, and num_minibatches across every learner"
        )
    return tuple(grid)


def _normalize_seed_manifest(value: object, *, field: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be an ordered sequence of integer seeds")
    seeds = tuple(value)
    if (
        not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(f"{field} must contain unique non-negative integer seeds")
    return seeds


def validate_compute_aware_tuning_contract(
    *,
    schema_version: int,
    comparison_profile: str | None,
    matrix_kind: str,
    tuned_families: tuple[dict[str, str], ...],
    learner_grid: tuple[dict[str, Any], ...],
    environments: Mapping[str, int],
    seeds: Sequence[int],
    quick: bool,
) -> None:
    """Fail closed on the fixed compute-aware two-task tuning panel."""

    if schema_version != 5:
        raise ValueError("compute-aware tuning requires registration schema version 5")
    if comparison_profile != "arcmind_shared_comparison":
        raise ValueError(
            "compute-aware tuning requires comparison_profile "
            "'arcmind_shared_comparison'"
        )
    if matrix_kind != "hyperparameter_selection":
        raise ValueError(
            "compute-aware tuning requires matrix_kind 'hyperparameter_selection'"
        )
    if quick:
        raise ValueError("compute-aware tuning cannot use quick execution")
    if not tuned_families:
        raise ValueError("compute-aware tuning requires explicit tuned families")
    if not learner_grid:
        raise ValueError("compute-aware tuning requires an explicit shared learner grid")
    actual_panel = tuple(environments.items())
    if actual_panel != COMPUTE_AWARE_TUNING_PANEL:
        raise ValueError(
            "compute-aware tuning requires the exact ordered two-task panel: "
            f"expected={COMPUTE_AWARE_TUNING_PANEL}, found={actual_panel}"
        )
    normalized_seeds = _normalize_seed_manifest(seeds, field="seeds")
    if normalized_seeds != COMPUTE_AWARE_TUNING_SEEDS:
        raise ValueError(
            "compute-aware tuning requires the exact ordered seed manifest: "
            f"expected={COMPUTE_AWARE_TUNING_SEEDS}, found={normalized_seeds}"
        )
    for candidate in learner_grid:
        learner = candidate["learner"]
        for environment, total_steps in COMPUTE_AWARE_TUNING_PANEL:
            realized_environment_steps(
                total_steps,
                num_envs=int(learner["num_envs"]),
                rollout_steps=int(learner["rollout_steps"]),
                comparison_profile=comparison_profile,
            )


def normalize_panel_selection_binding(value: object) -> dict[str, Any]:
    """Validate the immutable schema-v6 binding to schema-v5 panel winners."""

    expected_fields = {
        "raw_matrix_path",
        "aggregate_path",
        "aggregate_sha256",
        "source_registration_sha256",
        "source_manifest_sha256",
        "source_completion_index_sha256",
        "source_checksum_manifest_sha256",
        "source_implementation_sha256",
        "selections",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError(
            "tuning_selection has wrong fields: "
            f"expected={sorted(expected_fields)}"
        )
    hashes: dict[str, str] = {}
    for name in (
        "aggregate_sha256",
        "source_registration_sha256",
        "source_manifest_sha256",
        "source_completion_index_sha256",
        "source_checksum_manifest_sha256",
        "source_implementation_sha256",
    ):
        hash_value = value[name]
        if not isinstance(hash_value, str) or not _SHA256_PATTERN.fullmatch(hash_value):
            raise ValueError(f"tuning_selection.{name} must be a lowercase SHA256")
        hashes[name] = hash_value
    raw_selections = value["selections"]
    if not isinstance(raw_selections, list) or not raw_selections:
        raise ValueError("tuning_selection.selections must be a non-empty list")
    selections: list[dict[str, Any]] = []
    model_families: set[str] = set()
    implementation_models: set[str] = set()
    candidate_ids: set[str] = set()
    for index, raw_selection in enumerate(raw_selections):
        field = f"tuning_selection.selections[{index}]"
        if not isinstance(raw_selection, Mapping) or set(raw_selection) != _PANEL_SELECTION_FIELDS:
            raise ValueError(
                f"{field} must contain exactly model_family, implementation_model, "
                "candidate_id, learner_id, learner, and implementation_source_sha256"
            )
        model_family = raw_selection["model_family"]
        implementation_model = raw_selection["implementation_model"]
        learner_id = raw_selection["learner_id"]
        candidate_id = raw_selection["candidate_id"]
        if (
            not isinstance(model_family, str)
            or not _IDENTIFIER_PATTERN.fullmatch(model_family)
            or model_family in model_families
        ):
            raise ValueError(f"{field}.model_family must be a unique portable identifier")
        if (
            not isinstance(implementation_model, str)
            or not _IDENTIFIER_PATTERN.fullmatch(implementation_model)
            or implementation_model in implementation_models
        ):
            raise ValueError(
                f"{field}.implementation_model must be a unique portable model identifier"
            )
        if not isinstance(learner_id, str) or not _IDENTIFIER_PATTERN.fullmatch(learner_id):
            raise ValueError(f"{field}.learner_id must be a portable identifier")
        if (
            not isinstance(candidate_id, str)
            or not _IDENTIFIER_PATTERN.fullmatch(candidate_id)
            or candidate_id != f"{model_family}.{learner_id}"
            or candidate_id in candidate_ids
        ):
            raise ValueError(
                f"{field}.candidate_id must be unique and equal "
                "model_family + '.' + learner_id"
            )
        validate_policy_model_id(
            implementation_model,
            field=f"{field}.implementation_model",
        )
        validate_model_evidence_tier(
            implementation_model,
            "registered_final",
            field=f"{field}.implementation_model",
        )
        implementation_source_sha256 = raw_selection["implementation_source_sha256"]
        if (
            not isinstance(implementation_source_sha256, str)
            or not _SHA256_PATTERN.fullmatch(implementation_source_sha256)
            or implementation_source_sha256 != hashes["source_implementation_sha256"]
        ):
            raise ValueError(
                f"{field}.implementation_source_sha256 must equal "
                "tuning_selection.source_implementation_sha256"
            )
        selections.append(
            {
                "model_family": model_family,
                "implementation_model": implementation_model,
                "candidate_id": candidate_id,
                "learner_id": learner_id,
                "learner": normalize_learner(raw_selection["learner"], schema_version=6),
                "implementation_source_sha256": implementation_source_sha256,
            }
        )
        model_families.add(model_family)
        implementation_models.add(implementation_model)
        candidate_ids.add(candidate_id)
    return {
        "raw_matrix_path": _normalized_repository_path(
            value["raw_matrix_path"],
            field="tuning_selection.raw_matrix_path",
        ),
        "aggregate_path": _normalized_repository_path(
            value["aggregate_path"],
            field="tuning_selection.aggregate_path",
        ),
        **hashes,
        "selections": tuple(selections),
    }


def normalize_learner_bindings(
    value: object,
    *,
    models: Sequence[str],
) -> tuple[dict[str, str], ...]:
    """Validate ordered schema-v6 selected and inherited learner bindings."""

    if isinstance(models, (str, bytes)) or not isinstance(models, Sequence):
        raise ValueError("models must be an ordered sequence")
    normalized_models = tuple(models)
    if (
        not normalized_models
        or any(not isinstance(model, str) or not model for model in normalized_models)
        or len(set(normalized_models)) != len(normalized_models)
    ):
        raise ValueError("models must contain unique non-empty model names")
    for index, model in enumerate(normalized_models):
        validate_policy_model_id(model, field=f"models[{index}]")
        validate_model_evidence_tier(model, "registered_final", field=f"models[{index}]")
    if not isinstance(value, list) or len(value) != len(normalized_models):
        raise ValueError("learner_bindings must contain exactly one entry per model")
    bindings: list[dict[str, str]] = []
    for index, raw_binding in enumerate(value):
        field = f"learner_bindings[{index}]"
        if not isinstance(raw_binding, Mapping) or set(raw_binding) != {
            "model",
            "mode",
            "source_model_family",
        }:
            raise ValueError(
                f"{field} must contain exactly model, mode, and source_model_family"
            )
        model = raw_binding["model"]
        if model != normalized_models[index]:
            raise ValueError("learner_bindings must preserve the exact models order")
        mode = raw_binding["mode"]
        if mode not in {"selected", "inherited"}:
            raise ValueError(f"{field}.mode must be 'selected' or 'inherited'")
        source_model_family = raw_binding["source_model_family"]
        if (
            not isinstance(source_model_family, str)
            or not _IDENTIFIER_PATTERN.fullmatch(source_model_family)
        ):
            raise ValueError(f"{field}.source_model_family must be a portable identifier")
        bindings.append(
            {
                "model": model,
                "mode": mode,
                "source_model_family": source_model_family,
            }
        )
    return tuple(bindings)


def normalize_task_model_incidence(
    value: object,
    *,
    environments: Sequence[str],
    models: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Validate the exact non-Cartesian task-by-model design for schema v6."""

    environment_ids = tuple(environments)
    model_ids = tuple(models)
    if (
        not environment_ids
        or any(
            not isinstance(environment, str) or not environment
            for environment in environment_ids
        )
        or len(set(environment_ids)) != len(environment_ids)
    ):
        raise ValueError("environments must contain unique non-empty identifiers")
    if (
        not model_ids
        or any(not isinstance(model, str) or not model for model in model_ids)
        or len(set(model_ids)) != len(model_ids)
    ):
        raise ValueError("models must contain unique non-empty identifiers")
    if not isinstance(value, list) or len(value) != len(environment_ids):
        raise ValueError("task_model_incidence must contain exactly one entry per environment")
    model_positions = {model: index for index, model in enumerate(model_ids)}
    used_models: set[str] = set()
    common_models = set(model_ids)
    normalized: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(value):
        field = f"task_model_incidence[{index}]"
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != {"environment", "models"}:
            raise ValueError(f"{field} must contain exactly environment and models")
        environment = raw_entry["environment"]
        if environment != environment_ids[index]:
            raise ValueError(
                "task_model_incidence must preserve the exact environments order"
            )
        raw_models = raw_entry["models"]
        if not isinstance(raw_models, list) or not raw_models:
            raise ValueError(f"{field}.models must be a non-empty list")
        if (
            any(not isinstance(model, str) or model not in model_positions for model in raw_models)
            or len(set(raw_models)) != len(raw_models)
        ):
            raise ValueError(f"{field}.models must be unique members of the global models list")
        positions = [model_positions[model] for model in raw_models]
        if positions != sorted(positions):
            raise ValueError(
                f"{field}.models must preserve the exact global models order"
            )
        if "arcmind" not in raw_models:
            raise ValueError(f"{field}.models must contain arcmind")
        for model in raw_models:
            validate_model_environment_contract(
                model,
                environment,
                field=f"{field}.models",
            )
        used_models.update(raw_models)
        common_models.intersection_update(raw_models)
        normalized.append({"environment": environment, "models": tuple(raw_models)})
    missing_models = sorted(set(model_ids) - used_models)
    if missing_models:
        raise ValueError(
            f"every global model must occur in the incidence matrix: missing={missing_models}"
        )
    if len(common_models) < 2:
        raise ValueError(
            "task_model_incidence must retain at least two all-task common models"
        )
    return tuple(normalized)


def validate_compute_aware_final_contract(
    *,
    schema_version: int,
    comparison_profile: str | None,
    matrix_kind: str,
    models: Sequence[str],
    learner_bindings: object,
    task_model_incidence: object,
    tuning_selection: object,
    environments: Mapping[str, int],
    seeds: Sequence[int],
    quick: bool,
) -> None:
    """Fail closed on the compute-aware registered-final design."""

    if schema_version != 6:
        raise ValueError("compute-aware final requires registration schema version 6")
    if comparison_profile != "arcmind_shared_comparison":
        raise ValueError(
            "compute-aware final requires comparison_profile "
            "'arcmind_shared_comparison'"
        )
    if matrix_kind != "primary_comparison":
        raise ValueError("compute-aware final requires matrix_kind 'primary_comparison'")
    if quick:
        raise ValueError("compute-aware final cannot use quick execution")
    normalized_seeds = _normalize_seed_manifest(seeds, field="seeds")
    if normalized_seeds != COMPUTE_AWARE_FINAL_SEEDS:
        raise ValueError(
            "compute-aware final requires the exact ordered seed manifest: "
            f"expected={COMPUTE_AWARE_FINAL_SEEDS}, found={normalized_seeds}"
        )
    if set(normalized_seeds) & set(COMPUTE_AWARE_TUNING_SEEDS):
        raise ValueError("compute-aware final seeds must be disjoint from tuning seeds")
    actual_budgets = tuple(environments.items())
    if not actual_budgets:
        raise ValueError("compute-aware final requires at least one environment")
    for environment, total_steps in actual_budgets:
        expected_steps = PUBLISHED_PRIMARY_TRAIN_STEPS.get(environment)
        if expected_steps is None:
            raise ValueError(
                "compute-aware final requires published POBAX primary environments"
            )
        if total_steps != expected_steps:
            raise ValueError(
                "compute-aware final requires each published task budget: "
                f"environment={environment!r}, expected={expected_steps}, "
                f"found={total_steps}"
            )
    normalized_bindings = normalize_learner_bindings(
        learner_bindings,
        models=models,
    )
    normalized_incidence = normalize_task_model_incidence(
        task_model_incidence,
        environments=[environment for environment, _ in actual_budgets],
        models=models,
    )
    binding = normalize_panel_selection_binding(tuning_selection)
    selections_by_family = {
        selection["model_family"]: selection for selection in binding["selections"]
    }
    selected_sources: set[str] = set()
    bindings_by_model: dict[str, dict[str, str]] = {}
    for learner_binding in normalized_bindings:
        model = learner_binding["model"]
        source_family = learner_binding["source_model_family"]
        selection = selections_by_family.get(source_family)
        if selection is None:
            raise ValueError(
                f"learner binding for {model!r} names unknown source family "
                f"{source_family!r}"
            )
        if learner_binding["mode"] == "selected":
            if model != selection["implementation_model"]:
                raise ValueError(
                    f"selected learner binding for {model!r} must execute its "
                    "source family's implementation model"
                )
            if source_family in selected_sources:
                raise ValueError(
                    f"source family {source_family!r} has duplicate selected bindings"
                )
            selected_sources.add(source_family)
        elif model == selection["implementation_model"]:
            raise ValueError(
                f"inherited learner binding for {model!r} must execute a different model"
            )
        bindings_by_model[model] = learner_binding
    missing_selected_sources = sorted(set(selections_by_family) - selected_sources)
    if missing_selected_sources:
        raise ValueError(
            "every tuning selection must have one direct selected binding: "
            f"missing={missing_selected_sources}"
        )
    for incidence_entry in normalized_incidence:
        environment = incidence_entry["environment"]
        total_steps = environments[environment]
        for model in incidence_entry["models"]:
            source_family = bindings_by_model[model]["source_model_family"]
            learner = selections_by_family[source_family]["learner"]
            realized_environment_steps(
                total_steps,
                num_envs=int(learner["num_envs"]),
                rollout_steps=int(learner["rollout_steps"]),
                comparison_profile=comparison_profile,
            )


def validate_development_tuning_contract(
    *,
    schema_version: int,
    comparison_profile: str | None,
    matrix_kind: str,
    candidate_families: tuple[dict[str, Any], ...],
    environments: Mapping[str, int],
    seeds: tuple[int, ...] | list[int],
    quick: bool,
) -> None:
    """Fail closed on the matrix contract used for hyperparameter selection."""

    if schema_version != 3:
        raise ValueError("development_tuning requires registration schema version 3")
    if comparison_profile != "arcmind_shared_comparison":
        raise ValueError(
            "development_tuning requires comparison_profile 'arcmind_shared_comparison'"
        )
    if matrix_kind != "hyperparameter_selection":
        raise ValueError("development_tuning requires matrix_kind 'hyperparameter_selection'")
    if quick:
        raise ValueError("development_tuning cannot use quick execution")
    if not candidate_families:
        raise ValueError("development_tuning requires explicit candidate families")
    if len(environments) != 1:
        raise ValueError("development_tuning requires exactly one primary environment")
    environment, total_steps = next(iter(environments.items()))
    expected_steps = PUBLISHED_PRIMARY_TRAIN_STEPS.get(environment)
    if expected_steps is None:
        raise ValueError("development_tuning requires one published POBAX primary environment")
    if total_steps != expected_steps:
        raise ValueError(
            "development_tuning requires the published task budget: "
            f"environment={environment!r}, expected={expected_steps}, found={total_steps}"
        )
    expected_seed_count = PUBLISHED_TUNING_SEED_COUNTS[environment]
    if len(seeds) != expected_seed_count:
        raise ValueError(
            "development_tuning requires the published tuning-seed count: "
            f"environment={environment!r}, expected={expected_seed_count}, "
            f"found={len(seeds)}"
        )
