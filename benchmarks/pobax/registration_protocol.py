"""Schema-specific validation for frozen POBAX experiment registrations."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

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
        raise ValueError("registration schema_version must be 1, 2, or 3")
    if schema_version == 1:
        return REGISTRATION_FIELDS_V1
    if schema_version == 2:
        return REGISTRATION_FIELDS_V2
    if schema_version == 3:
        return REGISTRATION_FIELDS_V3
    raise ValueError("registration schema_version must be 1, 2, or 3")


def learner_fields(schema_version: int) -> set[str]:
    """Return the exact learner field set for a supported registration."""

    if schema_version == 1:
        return LEARNER_FIELDS_V1
    if schema_version in {2, 3}:
        return LEARNER_FIELDS_V2
    raise ValueError("registration schema_version must be 1, 2, or 3")


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
    if schema_version in {2, 3}:
        integer_fields.append("num_minibatches")
    for field in integer_fields:
        value = learner[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"learner.{field} must be a positive integer")
        normalized[field] = value
    if schema_version in {2, 3} and normalized["num_envs"] % normalized["num_minibatches"] != 0:
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

    if schema_version in {2, 3}:
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
    return tuple(families)


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
