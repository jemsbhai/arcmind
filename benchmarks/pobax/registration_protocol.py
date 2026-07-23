"""Schema-specific validation for frozen POBAX experiment registrations."""

from __future__ import annotations

import math
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


def registration_fields(schema_version: object) -> set[str]:
    """Return the exact top-level field set for a supported registration."""

    if isinstance(schema_version, bool):
        raise ValueError("registration schema_version must be 1 or 2")
    if schema_version == 1:
        return REGISTRATION_FIELDS_V1
    if schema_version == 2:
        return REGISTRATION_FIELDS_V2
    raise ValueError("registration schema_version must be 1 or 2")


def learner_fields(schema_version: int) -> set[str]:
    """Return the exact learner field set for a supported registration."""

    if schema_version == 1:
        return LEARNER_FIELDS_V1
    if schema_version == 2:
        return LEARNER_FIELDS_V2
    raise ValueError("registration schema_version must be 1 or 2")


def validate_comparison_profile(registration: Mapping[str, Any]) -> str | None:
    """Validate and return the v2 comparison profile, or None for v1."""

    schema_version = registration.get("schema_version")
    if schema_version == 1:
        return None
    profile = registration.get("comparison_profile")
    if profile not in COMPARISON_PROFILES:
        raise ValueError(
            "comparison_profile must be 'pobax_author_semantics' or "
            "'arcmind_shared_comparison'"
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
    if schema_version == 2:
        integer_fields.append("num_minibatches")
    for field in integer_fields:
        value = learner[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"learner.{field} must be a positive integer")
        normalized[field] = value
    if schema_version == 2 and normalized["num_envs"] % normalized["num_minibatches"] != 0:
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

    if schema_version == 2:
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
