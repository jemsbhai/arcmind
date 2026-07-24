"""Fail-closed aggregation for registered POBAX experiment matrices.

The frozen matrix manifest is a JSON object with these exact fields:

``schema_version``
    Integer 1.
``status``
    The string ``"frozen"``.
``manifest_sha256``
    SHA256 of the canonical manifest after this field is removed.
``matrix_kind``
    Either ``"primary_comparison"`` or ``"upper_reference"``.
``models``, ``environments``, ``seeds``
    Non-empty unique lists defining the complete Cartesian matrix.
``provenance``
    Clean Git and pinned dependency provenance shared by every cell.
``cells``
    One exact entry for every environment, model, and seed combination.

Each cell entry contains ``cell_id``, ``environment``, ``model``, ``seed``,
``configuration_sha256``, and ``artifact_path``. Artifact paths must use
POSIX separators and are resolved relative to the manifest directory.

Seeds, not evaluation episodes, are the independent statistical units. For
each model and environment, final-return statistics are computed from the
per-seed mean evaluation returns. The interquartile mean (IQM) assigns each
sorted observation its overlap with the central empirical probability mass
[0.25, 0.75], then divides the weighted sum by 0.5. This fractional-boundary
definition is exact for every positive seed count.

Training step grids must match across every model and seed within an
environment. A leading JSON null is a truthful marker that no episode had
completed yet. Curves are aggregated only from the first step at which every
cell in that environment has a finite return. A null after a finite return is
invalid because availability must not disappear.
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

from benchmarks.pobax.aggregate_development import build_development_aggregate
from benchmarks.pobax.implementation_provenance import normalize_implementation_source
from benchmarks.pobax.model_registry import (
    PARAMETER_MATCHED_CONTRACT,
    SUPPLEMENTAL_COMPARISON_ROLE,
    comparison_role_for_model,
    fixed_official_parameter_count,
    parameter_contract_for_model,
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
    ArtifactChecksumError,
    atomic_write_json,
    canonical_json_bytes,
    canonical_json_sha256,
    registered_cell_id,
    sha256_file,
    validate_checksum_manifest,
)
from benchmarks.pobax.registration_protocol import (
    LEARNER_FIELDS_V2,
    normalize_final_selection_binding,
    normalize_learner,
    realized_environment_steps,
    registration_fields,
    step_budget_mode,
    validate_comparison_profile,
    validate_final_provenance_against_tuning,
    validate_final_selection_against_aggregate,
)
from benchmarks.pobax.upper_reference_registry import (
    UPPER_REFERENCE_ENVIRONMENTS,
    expected_environment_reference,
    expected_environment_source,
)

BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SEED = 20_260_723
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
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
_CELL_MANIFEST_KEYS = {
    "cell_id",
    "environment",
    "model",
    "seed",
    "configuration_sha256",
    "artifact_path",
}
_CELL_MANIFEST_KEYS_V4 = _CELL_MANIFEST_KEYS | {
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
_COMPLETED_CELL_KEYS = _CELL_MANIFEST_KEYS | {
    "artifact_sha256",
    "log_path",
    "log_sha256",
}
_COMPLETED_CELL_KEYS_V4 = _CELL_MANIFEST_KEYS_V4 | {
    "artifact_sha256",
    "log_path",
    "log_sha256",
}
_PROVENANCE_KEYS = {
    "git",
    "dependency_lock_sha256",
    "pobax_commit",
    "navix_commit",
    "runtime_contract",
}
_PROVENANCE_KEYS_WITH_IMPLEMENTATION = _PROVENANCE_KEYS | {"implementation_source"}
_GIT_KEYS = {"commit", "dirty", "diff_sha256"}
_RUNTIME_KEYS = {
    "python",
    "packages",
    "jax_backend",
    "jax_enable_x64",
    "devices",
}
_PYTHON_KEYS = {"implementation", "version"}
_DEVICE_KEYS = {"platform", "device_kind"}
_RUNTIME_PACKAGE_KEYS = {
    "brax",
    "gymnax",
    "jax",
    "jax-cuda12-pjrt",
    "jax-cuda12-plugin",
    "jaxlib",
    "Navix",
    "numpy",
    "optax",
    "pobax",
}
_MATRIX_KINDS = {"primary_comparison", "upper_reference"}
_REGISTERED_FINAL_SEED_COUNT = 30
_OPTIMIZER_METRICS = (
    "loss",
    "actor_loss",
    "value_loss",
    "entropy",
    "approximate_kl",
)
_REGISTERED_TRAIN_STEPS = {
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
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class RegisteredAggregationError(ValueError):
    """Raised when a registered matrix cannot be aggregated safely."""


def _reject_json_constant(value: str) -> None:
    raise RegisteredAggregationError(f"non-finite JSON constant is prohibited: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RegisteredAggregationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, *, kind: str) -> Any:
    if not path.is_file():
        raise RegisteredAggregationError(f"{kind} does not exist or is not a file: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RegisteredAggregationError(f"cannot read {kind} {path}: {error}") from error


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegisteredAggregationError(f"{field} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RegisteredAggregationError(
            f"{field} has wrong fields: missing={missing}, extra={extra}"
        )


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegisteredAggregationError(f"{field} must be a non-empty string")
    return value


def _integer(value: Any, *, field: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RegisteredAggregationError(f"{field} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise RegisteredAggregationError(f"{field} must be {qualifier}")
    return value


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegisteredAggregationError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RegisteredAggregationError(f"{field} must be a finite number")
    return result


def _sha256(value: Any, *, field: str) -> str:
    result = _string(value, field=field)
    if not _SHA256_PATTERN.fullmatch(result):
        raise RegisteredAggregationError(f"{field} must be a lowercase SHA256")
    return result


def _commit(value: Any, *, field: str) -> str:
    result = _string(value, field=field)
    if not _GIT_COMMIT_PATTERN.fullmatch(result):
        raise RegisteredAggregationError(f"{field} must be a lowercase Git commit")
    return result


def _unique_strings(value: Any, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list) or not value:
        raise RegisteredAggregationError(f"{field} must be a non-empty JSON list")
    result = tuple(_string(item, field=f"{field}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise RegisteredAggregationError(f"{field} contains duplicates")
    return result


def _unique_seeds(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise RegisteredAggregationError("seeds must be a non-empty JSON list")
    result = tuple(_integer(item, field=f"seeds[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise RegisteredAggregationError("seeds contains duplicates")
    return result


def _validate_runtime_contract(value: Any, *, field: str) -> dict[str, Any]:
    runtime = _mapping(value, field=field)
    _exact_keys(runtime, _RUNTIME_KEYS, field=field)
    python = _mapping(runtime["python"], field=f"{field}.python")
    _exact_keys(python, _PYTHON_KEYS, field=f"{field}.python")
    packages = _mapping(runtime["packages"], field=f"{field}.packages")
    _exact_keys(packages, _RUNTIME_PACKAGE_KEYS, field=f"{field}.packages")
    devices = runtime["devices"]
    if not isinstance(devices, list) or not devices:
        raise RegisteredAggregationError(f"{field}.devices must be a non-empty JSON list")
    normalized_devices = []
    for index, raw_device in enumerate(devices):
        device_field = f"{field}.devices[{index}]"
        device = _mapping(raw_device, field=device_field)
        _exact_keys(device, _DEVICE_KEYS, field=device_field)
        normalized_devices.append(
            {
                "platform": _string(device["platform"], field=f"{device_field}.platform"),
                "device_kind": _string(
                    device["device_kind"],
                    field=f"{device_field}.device_kind",
                ),
            }
        )
    if not isinstance(runtime["jax_enable_x64"], bool):
        raise RegisteredAggregationError(f"{field}.jax_enable_x64 must be a boolean")
    return {
        "python": {
            "implementation": _string(
                python["implementation"],
                field=f"{field}.python.implementation",
            ),
            "version": _string(
                python["version"],
                field=f"{field}.python.version",
            ),
        },
        "packages": {
            package: _string(
                packages[package],
                field=f"{field}.packages.{package}",
            )
            for package in sorted(_RUNTIME_PACKAGE_KEYS)
        },
        "jax_backend": _string(
            runtime["jax_backend"],
            field=f"{field}.jax_backend",
        ),
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
    _exact_keys(
        provenance,
        (
            _PROVENANCE_KEYS_WITH_IMPLEMENTATION
            if require_implementation_source
            else _PROVENANCE_KEYS
        ),
        field=field,
    )
    git = _mapping(provenance["git"], field=f"{field}.git")
    _exact_keys(git, _GIT_KEYS, field=f"{field}.git")
    commit = _commit(git["commit"], field=f"{field}.git.commit")
    if git["dirty"] is not False:
        raise RegisteredAggregationError(f"{field}.git.dirty must be false")
    if git["diff_sha256"] is not None:
        raise RegisteredAggregationError(
            f"{field}.git.diff_sha256 must be null for a clean registered artifact"
        )
    normalized_runtime = _validate_runtime_contract(
        provenance["runtime_contract"],
        field=f"{field}.runtime_contract",
    )
    normalized = {
        "git": {"commit": commit, "dirty": False, "diff_sha256": None},
        "dependency_lock_sha256": _sha256(
            provenance["dependency_lock_sha256"],
            field=f"{field}.dependency_lock_sha256",
        ),
        "pobax_commit": _commit(provenance["pobax_commit"], field=f"{field}.pobax_commit"),
        "navix_commit": _commit(provenance["navix_commit"], field=f"{field}.navix_commit"),
        "runtime_contract": normalized_runtime,
    }
    if require_implementation_source:
        try:
            normalized["implementation_source"] = normalize_implementation_source(
                provenance["implementation_source"]
            )
        except ValueError as error:
            raise RegisteredAggregationError(str(error)) from error
    return normalized


def _validate_frozen_configuration(
    value: Any,
    *,
    identity: tuple[str, str, int],
    provenance: Mapping[str, Any],
    field: str,
    selection: Mapping[str, Any] | None = None,
) -> tuple[int, int, int]:
    configuration = _mapping(value, field=field)
    required = {
        "schema_version",
        "evidence_tier",
        "environment",
        "model",
        "seed",
        "environment_source",
        "environment_reference",
        "parameter_count",
        "effective_parameter_count",
        "arcmind_target_parameter_count",
        "parameter_ratio",
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
        raise RegisteredAggregationError(f"{field} is missing required fields: {missing}")
    configuration_schema = configuration["schema_version"]
    if configuration_schema not in {1, 2, 4}:
        raise RegisteredAggregationError(f"{field}.schema_version must equal 1, 2, or 4")
    if configuration["evidence_tier"] != "registered_final":
        raise RegisteredAggregationError(f"{field}.evidence_tier must equal 'registered_final'")

    environment = identity[0]
    configured_identity = (
        _string(configuration["environment"], field=f"{field}.environment"),
        _string(configuration["model"], field=f"{field}.model"),
        _integer(configuration["seed"], field=f"{field}.seed"),
    )
    if configured_identity != identity:
        raise RegisteredAggregationError(f"{field} identity does not match its artifact")
    try:
        validate_required_reference_implementation(
            identity[1],
            configuration.get("reference_implementation"),
            field=f"{field}.reference_implementation",
        )
    except ValueError as error:
        raise RegisteredAggregationError(str(error)) from error
    expected_policy_contract = policy_contract_metadata_for_model(identity[1])
    if (
        requires_explicit_policy_contract(identity[1])
        or any(name in configuration for name in expected_policy_contract)
    ):
        try:
            validate_policy_contract_metadata(
                identity[1],
                {
                    name: configuration.get(name)
                    for name in expected_policy_contract
                },
                field=f"{field}.policy_contract",
            )
            validate_policy_core_contract(
                identity[1],
                configuration.get("policy_core"),
                field=f"{field}.policy_core",
            )
        except ValueError as error:
            raise RegisteredAggregationError(str(error)) from error
    if configuration_schema == 4:
        for name in (
            "candidate_id",
            "model_family",
            "implementation_model",
            "tuning_aggregate_sha256",
            "tuning_completion_index_sha256",
            "tuning_checksum_manifest_sha256",
            "tuning_implementation_source_sha256",
            "implementation_source",
        ):
            if name not in configuration:
                raise RegisteredAggregationError(f"{field} is missing {name}")
        if selection is None or (
            configuration["candidate_id"] != selection["candidate_id"]
            or configuration["model_family"] != selection["model_family"]
            or configuration["implementation_model"] != identity[1]
            or configuration["tuning_aggregate_sha256"] != selection["tuning_aggregate_sha256"]
            or configuration["tuning_completion_index_sha256"]
            != selection["tuning_completion_index_sha256"]
            or configuration["tuning_checksum_manifest_sha256"]
            != selection["tuning_checksum_manifest_sha256"]
            or configuration["tuning_implementation_source_sha256"]
            != selection["tuning_implementation_source_sha256"]
        ):
            raise RegisteredAggregationError(
                f"{field} tuning-selection identity drifts from the manifest"
            )
        try:
            implementation_source = normalize_implementation_source(
                configuration["implementation_source"]
            )
        except ValueError as error:
            raise RegisteredAggregationError(str(error)) from error
        if implementation_source["sha256"] != selection["implementation_source_sha256"]:
            raise RegisteredAggregationError(
                f"{field} implementation source drifts from the manifest"
            )
    elif selection is not None:
        raise RegisteredAggregationError(f"{field} has an unexpected tuning selection")
    environment_source = _mapping(
        configuration["environment_source"],
        field=f"{field}.environment_source",
    )
    if dict(environment_source) != expected_environment_source(environment):
        raise RegisteredAggregationError(
            f"{field}.environment_source does not match the registered source invocation"
        )
    expected_reference = expected_environment_reference(environment)
    environment_reference = configuration["environment_reference"]
    if environment_reference != expected_reference:
        raise RegisteredAggregationError(
            f"{field}.environment_reference does not match the registered reference class"
        )
    parameter_count = _integer(
        configuration["parameter_count"],
        field=f"{field}.parameter_count",
        positive=True,
    )
    effective_parameter_count = _integer(
        configuration["effective_parameter_count"],
        field=f"{field}.effective_parameter_count",
        positive=True,
    )
    target_parameter_count = _integer(
        configuration["arcmind_target_parameter_count"],
        field=f"{field}.arcmind_target_parameter_count",
        positive=True,
    )
    parameter_ratio = _finite_number(
        configuration["parameter_ratio"],
        field=f"{field}.parameter_ratio",
    )
    computed_ratio = parameter_count / target_parameter_count
    if effective_parameter_count > parameter_count:
        raise RegisteredAggregationError(
            f"{field}.effective_parameter_count exceeds parameter_count"
        )
    if not math.isclose(parameter_ratio, computed_ratio, rel_tol=1e-12, abs_tol=1e-12):
        raise RegisteredAggregationError(
            f"{field}.parameter_ratio disagrees with frozen parameter counts"
        )
    expected_fixed_count = fixed_official_parameter_count(
        identity[1],
        configuration.get("policy_core"),
    )
    if expected_fixed_count is not None and (
        parameter_count != expected_fixed_count
        or effective_parameter_count != expected_fixed_count
    ):
        raise RegisteredAggregationError(
            f"{field}.parameter_count does not match the fixed official architecture"
        )
    parameter_contract = parameter_contract_for_model(identity[1])
    if (
        parameter_contract == PARAMETER_MATCHED_CONTRACT
        and not 0.9 <= parameter_ratio <= 1.1
    ):
        raise RegisteredAggregationError(
            f"{field}.parameter_ratio violates the registered matching tolerance"
        )

    expected_budget = _REGISTERED_TRAIN_STEPS.get(environment)
    if expected_budget is None:
        raise RegisteredAggregationError(
            f"{field}.environment has no registered-final budget: {environment!r}"
        )
    ppo = _mapping(configuration["ppo"], field=f"{field}.ppo")
    if "total_steps" not in ppo:
        raise RegisteredAggregationError(f"{field}.ppo is missing total_steps")
    total_steps = _integer(
        ppo["total_steps"],
        field=f"{field}.ppo.total_steps",
        positive=True,
    )
    if total_steps != expected_budget:
        raise RegisteredAggregationError(
            f"{field}.ppo.total_steps does not match the registered budget: "
            f"expected={expected_budget}, found={total_steps}"
        )
    realized_steps = total_steps
    if configuration_schema in {2, 4}:
        for name in (
            "comparison_profile",
            "requested_environment_steps",
            "realized_environment_steps",
        ):
            if name not in configuration:
                raise RegisteredAggregationError(f"{field} is missing {name}")
        profile = configuration["comparison_profile"]
        missing_learner_fields = sorted(LEARNER_FIELDS_V2 - set(ppo))
        if missing_learner_fields:
            raise RegisteredAggregationError(
                f"{field}.ppo is missing registered learner fields: {missing_learner_fields}"
            )
        try:
            normalized_learner = normalize_learner(
                {name: ppo[name] for name in LEARNER_FIELDS_V2},
                schema_version=configuration_schema,
            )
        except ValueError as error:
            raise RegisteredAggregationError(str(error)) from error
        if configuration_schema == 4 and (
            selection is None or normalized_learner != selection["learner"]
        ):
            raise RegisteredAggregationError(
                f"{field}.ppo learner drifts from the tuning aggregate winner"
            )
        ppo_num_envs = _integer(
            ppo.get("num_envs"),
            field=f"{field}.ppo.num_envs",
            positive=True,
        )
        ppo_rollout_steps = _integer(
            ppo.get("rollout_steps"),
            field=f"{field}.ppo.rollout_steps",
            positive=True,
        )
        try:
            expected_realized_steps = realized_environment_steps(
                total_steps,
                num_envs=ppo_num_envs,
                rollout_steps=ppo_rollout_steps,
                comparison_profile=profile,
            )
            expected_step_mode = step_budget_mode(profile)
        except ValueError as error:
            raise RegisteredAggregationError(str(error)) from error
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
        if (
            requested_steps != total_steps
            or realized_steps != expected_realized_steps
            or ppo.get("step_budget_mode") != expected_step_mode
        ):
            raise RegisteredAggregationError(
                f"{field} requested and realized step accounting is inconsistent"
            )

    evaluation_episodes = _integer(
        configuration["evaluation_episodes_per_environment"],
        field=f"{field}.evaluation_episodes_per_environment",
        positive=True,
    )
    evaluation_horizon = _integer(
        configuration["evaluation_max_episode_steps"],
        field=f"{field}.evaluation_max_episode_steps",
        positive=True,
    )
    if configuration_schema == 4:
        try:
            validate_causal_transformer_horizon_contract(
                identity[1],
                configuration.get("policy_core"),
                evaluation_horizon,
                field=f"{field}.policy_core",
            )
        except ValueError as error:
            raise RegisteredAggregationError(str(error)) from error
    configured_sources = {
        "dependency_lock_sha256": _sha256(
            configuration["dependency_lock_sha256"],
            field=f"{field}.dependency_lock_sha256",
        ),
        "pobax_commit": _commit(
            configuration["pobax_commit"],
            field=f"{field}.pobax_commit",
        ),
        "navix_commit": _commit(
            configuration["navix_commit"],
            field=f"{field}.navix_commit",
        ),
        "runtime_contract": _validate_runtime_contract(
            configuration["runtime_contract"],
            field=f"{field}.runtime_contract",
        ),
    }
    source_fields = [
        "dependency_lock_sha256",
        "pobax_commit",
        "navix_commit",
        "runtime_contract",
    ]
    if configuration_schema == 4:
        configured_sources["implementation_source"] = implementation_source
        source_fields.append("implementation_source")
    expected_sources = {key: provenance[key] for key in source_fields}
    if configured_sources != expected_sources:
        raise RegisteredAggregationError(f"{field} source provenance does not match the manifest")
    return realized_steps, evaluation_episodes, evaluation_horizon


def _artifact_path(manifest_path: Path, value: Any, *, field: str) -> Path:
    text = _string(value, field=field)
    if "\\" in text:
        raise RegisteredAggregationError(f"{field} must use POSIX separators")
    pure_path = PurePosixPath(text)
    if pure_path.is_absolute() or any(part in {"", ".", ".."} for part in pure_path.parts):
        raise RegisteredAggregationError(f"{field} must be a normalized relative POSIX path")
    root = manifest_path.parent.resolve()
    path = root.joinpath(*pure_path.parts).resolve()
    if not path.is_relative_to(root):
        raise RegisteredAggregationError(f"{field} escapes the manifest directory")
    return path


def _iqm_weights(size: int) -> np.ndarray:
    lower = np.arange(size, dtype=np.float64) / size
    upper = np.arange(1, size + 1, dtype=np.float64) / size
    return np.maximum(0.0, np.minimum(upper, 0.75) - np.maximum(lower, 0.25)) / 0.5


def interquartile_mean(values: Sequence[float]) -> float:
    """Return the fractional-boundary IQM of a non-empty finite sample."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise RegisteredAggregationError("IQM requires a non-empty finite one-dimensional sample")
    return float(np.dot(np.sort(array), _iqm_weights(array.size)))


def _statistic(values: np.ndarray, name: str) -> float:
    if name == "mean":
        return float(np.mean(values))
    if name == "median":
        return float(np.median(values))
    if name == "iqm":
        return interquartile_mean(values)
    raise AssertionError(f"unsupported statistic: {name}")


def _bootstrap_stream_seed(stream: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}\0{stream}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _bootstrap_interval(values: Sequence[float], *, statistic: str, stream: str) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(_bootstrap_stream_seed(stream))
    indices = rng.integers(
        0,
        array.size,
        size=(BOOTSTRAP_RESAMPLES, array.size),
        endpoint=False,
    )
    samples = array[indices]
    if statistic == "mean":
        estimates = np.mean(samples, axis=1)
    elif statistic == "median":
        estimates = np.median(samples, axis=1)
    elif statistic == "iqm":
        estimates = np.sort(samples, axis=1) @ _iqm_weights(array.size)
    else:  # pragma: no cover - callers use the closed statistic set
        raise AssertionError(f"unsupported statistic: {statistic}")
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    bounds = np.quantile(estimates, [alpha, 1.0 - alpha], method="linear")
    return [float(bounds[0]), float(bounds[1])]


def _summary(values: Sequence[float], *, stream: str) -> dict[str, dict[str, Any]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        name: {
            "estimate": _statistic(array, name),
            "bootstrap_95_ci": _bootstrap_interval(
                array,
                statistic=name,
                stream=f"{stream}:{name}",
            ),
        }
        for name in ("mean", "median", "iqm")
    }


def _validate_evaluation(
    artifact: Mapping[str, Any],
    *,
    field: str,
) -> tuple[dict[str, Any], tuple[int, int, int]]:
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
        raise RegisteredAggregationError(
            f"{field}.evaluation is missing required fields: {missing}"
        )
    mean_return = _finite_number(
        evaluation["mean_return"],
        field=f"{field}.evaluation.mean_return",
    )
    median_return = _finite_number(
        evaluation["median_return"],
        field=f"{field}.evaluation.median_return",
    )
    episodes = _integer(
        evaluation["episodes"],
        field=f"{field}.evaluation.episodes",
        positive=True,
    )
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
    scan_steps_per_environment = _integer(
        evaluation["scan_steps_per_environment"],
        field=f"{field}.evaluation.scan_steps_per_environment",
        positive=True,
    )
    rows = evaluation["returns_by_environment"]
    if not isinstance(rows, list) or len(rows) != num_environments:
        raise RegisteredAggregationError(
            f"{field}.evaluation.returns_by_environment must have {num_environments} rows"
        )
    raw_returns: list[float] = []
    normalized_rows: list[list[float]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != episodes_per_environment:
            raise RegisteredAggregationError(
                f"{field}.evaluation.returns_by_environment[{row_index}] must have "
                f"{episodes_per_environment} returns"
            )
        normalized = [
            _finite_number(
                value,
                field=(f"{field}.evaluation.returns_by_environment[{row_index}][{column_index}]"),
            )
            for column_index, value in enumerate(row)
        ]
        normalized_rows.append(normalized)
        raw_returns.extend(normalized)
    if episodes != len(raw_returns) or episodes != episodes_per_environment * num_environments:
        raise RegisteredAggregationError(f"{field}.evaluation episode counts are inconsistent")
    computed_mean = float(np.mean(raw_returns))
    computed_median = float(np.median(raw_returns))
    if not math.isclose(mean_return, computed_mean, rel_tol=1e-12, abs_tol=1e-12):
        raise RegisteredAggregationError(
            f"{field}.evaluation.mean_return disagrees with raw returns"
        )
    if not math.isclose(median_return, computed_median, rel_tol=1e-12, abs_tol=1e-12):
        raise RegisteredAggregationError(
            f"{field}.evaluation.median_return disagrees with raw returns"
        )
    return (
        {
            "mean_return": computed_mean,
            "median_return": computed_median,
            "scan_steps_per_environment": scan_steps_per_environment,
            "returns_by_environment": normalized_rows,
        },
        (episodes, episodes_per_environment, num_environments),
    )


def _validate_training_history(
    artifact: Mapping[str, Any],
    *,
    field: str,
    expected_final_steps: int,
) -> tuple[tuple[int, ...], tuple[float | None, ...]]:
    history = artifact.get("training_history")
    if not isinstance(history, list) or not history:
        raise RegisteredAggregationError(f"{field}.training_history must be a non-empty list")
    steps: list[int] = []
    returns: list[float | None] = []
    return_available = False
    for index, item in enumerate(history):
        point = _mapping(item, field=f"{field}.training_history[{index}]")
        required = {"environment_steps", "mean_recent_return"}
        missing = sorted(required - set(point))
        if missing:
            raise RegisteredAggregationError(
                f"{field}.training_history[{index}] is missing required fields: {missing}"
            )
        missing_optimizer_metrics = sorted(set(_OPTIMIZER_METRICS) - set(point))
        if missing_optimizer_metrics:
            raise RegisteredAggregationError(
                f"{field}.training_history[{index}] is missing optimizer metrics: "
                f"{missing_optimizer_metrics}"
            )
        for metric in _OPTIMIZER_METRICS:
            _finite_number(
                point[metric],
                field=f"{field}.training_history[{index}].{metric}",
            )
        step_value = _finite_number(
            point["environment_steps"],
            field=f"{field}.training_history[{index}].environment_steps",
        )
        if not step_value.is_integer() or step_value <= 0:
            raise RegisteredAggregationError(
                f"{field}.training_history[{index}].environment_steps "
                "must be a positive integer-valued number"
            )
        steps.append(int(step_value))
        raw_return = point["mean_recent_return"]
        if raw_return is None:
            if return_available:
                raise RegisteredAggregationError(
                    f"{field}.training_history[{index}].mean_recent_return "
                    "is missing after returns became available"
                )
            returns.append(None)
        else:
            returns.append(
                _finite_number(
                    raw_return,
                    field=f"{field}.training_history[{index}].mean_recent_return",
                )
            )
            return_available = True
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise RegisteredAggregationError(
            f"{field}.training_history.environment_steps must be strictly increasing"
        )
    if steps[-1] != expected_final_steps:
        raise RegisteredAggregationError(
            f"{field}.training_history final step does not match the registered budget: "
            f"expected={expected_final_steps}, found={steps[-1]}"
        )
    return tuple(steps), tuple(returns)


def _validate_final_training_metrics(artifact: Mapping[str, Any], *, field: str) -> None:
    training = _mapping(artifact.get("training"), field=f"{field}.training")
    missing = sorted(set(_OPTIMIZER_METRICS) - set(training))
    if missing:
        raise RegisteredAggregationError(
            f"{field}.training is missing optimizer metrics: {missing}"
        )
    for metric in _OPTIMIZER_METRICS:
        _finite_number(training[metric], field=f"{field}.training.{metric}")


def _trapezoid_by_step(values: Sequence[float], steps: Sequence[int]) -> float:
    if len(values) == 1:
        return 0.0
    y = np.asarray(values, dtype=np.float64)
    x = np.asarray(steps, dtype=np.float64)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


def _bound_repository_path(relative_path: str, *, field: str) -> Path:
    path = _REPOSITORY_ROOT.joinpath(*Path(relative_path).parts).resolve()
    if not path.is_relative_to(_REPOSITORY_ROOT):
        raise RegisteredAggregationError(f"{field} escapes the repository root")
    return path


def _validate_manifest_tuning_selection(
    value: object,
    *,
    models: tuple[str, ...],
    environments: tuple[str, ...],
    final_seeds: tuple[int, ...],
    final_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        binding = normalize_final_selection_binding(value)
    except ValueError as error:
        raise RegisteredAggregationError(str(error)) from error
    aggregate_path = _bound_repository_path(
        binding["aggregate_path"],
        field="manifest.tuning_selection.aggregate_path",
    )
    try:
        aggregate_hash = sha256_file(aggregate_path)
    except OSError as error:
        raise RegisteredAggregationError(
            f"cannot read bound tuning aggregate: {aggregate_path}"
        ) from error
    if aggregate_hash != binding["aggregate_sha256"]:
        raise RegisteredAggregationError(
            "manifest tuning aggregate SHA256 does not match aggregate bytes"
        )
    raw_matrix_path = _bound_repository_path(
        binding["raw_matrix_path"],
        field="manifest.tuning_selection.raw_matrix_path",
    )
    for filename, binding_field in (
        ("completion_index.json", "source_completion_index_sha256"),
        ("checksums.sha256", "source_checksum_manifest_sha256"),
    ):
        source_path = raw_matrix_path / filename
        try:
            source_hash = sha256_file(source_path)
        except OSError as error:
            raise RegisteredAggregationError(
                f"cannot read bound tuning source: {source_path}"
            ) from error
        if source_hash != binding[binding_field]:
            raise RegisteredAggregationError(
                f"manifest tuning {binding_field} does not match source bytes"
            )
    try:
        rebuilt = build_development_aggregate(raw_matrix_path)
        aggregate_bytes = aggregate_path.read_bytes()
    except (OSError, ValueError) as error:
        raise RegisteredAggregationError(
            "cannot rebuild the bound development-tuning aggregate"
        ) from error
    if aggregate_bytes != canonical_json_bytes(rebuilt) + b"\n":
        raise RegisteredAggregationError(
            "bound tuning aggregate is not the canonical rebuild of its raw matrix"
        )
    if (
        rebuilt["registration_sha256"] != binding["source_registration_sha256"]
        or rebuilt["matrix_manifest_sha256"] != binding["source_manifest_sha256"]
    ):
        raise RegisteredAggregationError("bound tuning source registration or manifest hash drifts")
    try:
        validate_final_selection_against_aggregate(
            binding,
            rebuilt,
            models=models,
            environments=environments,
            final_seeds=final_seeds,
        )
        validate_final_provenance_against_tuning(
            binding=binding,
            tuning_provenance=rebuilt["provenance"],
            final_provenance=final_provenance,
        )
    except ValueError as error:
        raise RegisteredAggregationError(str(error)) from error
    return binding


def _validate_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[tuple[str, str, int], Any]]:
    manifest = _mapping(_load_json(manifest_path, kind="matrix manifest"), field="manifest")
    schema_version = manifest.get("schema_version")
    if schema_version not in {1, 2, 4}:
        raise RegisteredAggregationError("manifest.schema_version must equal 1, 2, or 4")
    _exact_keys(
        manifest,
        _MANIFEST_KEYS_V4 if schema_version == 4 else _MANIFEST_KEYS,
        field="manifest",
    )
    if manifest["status"] != "frozen":
        raise RegisteredAggregationError("manifest.status must equal 'frozen'")
    manifest_sha256 = _sha256(manifest["manifest_sha256"], field="manifest.manifest_sha256")
    hash_input = dict(manifest)
    del hash_input["manifest_sha256"]
    actual_manifest_sha256 = canonical_json_sha256(hash_input)
    if manifest_sha256 != actual_manifest_sha256:
        raise RegisteredAggregationError(
            "manifest.manifest_sha256 does not match the canonical manifest content"
        )

    models = _unique_strings(manifest["models"], field="manifest.models")
    for index, model in enumerate(models):
        try:
            validate_policy_model_id(model, field=f"manifest.models[{index}]")
        except ValueError as error:
            raise RegisteredAggregationError(str(error)) from error
    environments = _unique_strings(manifest["environments"], field="manifest.environments")
    for model in models:
        try:
            validate_model_evidence_tier(
                model,
                "registered_final",
                field=f"manifest model {model!r}",
            )
            for environment in environments:
                validate_model_environment_contract(
                    model,
                    environment,
                    field=f"manifest model {model!r}",
                )
        except ValueError as error:
            raise RegisteredAggregationError(str(error)) from error
    seeds = _unique_seeds(manifest["seeds"])
    if len(seeds) != _REGISTERED_FINAL_SEED_COUNT:
        raise RegisteredAggregationError(
            f"registered-final manifest must contain exactly {_REGISTERED_FINAL_SEED_COUNT} seeds"
        )
    matrix_kind = _string(manifest["matrix_kind"], field="manifest.matrix_kind")
    if matrix_kind not in _MATRIX_KINDS:
        raise RegisteredAggregationError(
            f"manifest.matrix_kind must be one of {sorted(_MATRIX_KINDS)}"
        )
    if matrix_kind == "primary_comparison" and "arcmind" not in models:
        raise RegisteredAggregationError(
            "primary_comparison manifest.models must contain 'arcmind'"
        )
    if matrix_kind == "upper_reference" and models != ("memoryless_mlp",):
        raise RegisteredAggregationError(
            "upper_reference manifest.models must contain only 'memoryless_mlp'"
        )
    selected_upper_references = set(environments) & UPPER_REFERENCE_ENVIRONMENTS
    if matrix_kind == "upper_reference":
        invalid_environments = set(environments) - UPPER_REFERENCE_ENVIRONMENTS
        if invalid_environments:
            raise RegisteredAggregationError(
                "upper_reference manifest contains non-reference environments: "
                f"{sorted(invalid_environments)}"
            )
    elif selected_upper_references:
        raise RegisteredAggregationError(
            "primary_comparison manifest contains upper-reference aliases: "
            f"{sorted(selected_upper_references)}"
        )
    if schema_version == 4 and matrix_kind != "primary_comparison":
        raise RegisteredAggregationError("schema-v4 manifest must be a primary comparison")
    provenance = _validate_provenance(
        manifest["provenance"],
        field="manifest.provenance",
        require_implementation_source=schema_version == 4,
    )
    tuning_selection = (
        _validate_manifest_tuning_selection(
            manifest["tuning_selection"],
            models=models,
            environments=environments,
            final_seeds=seeds,
            final_provenance=provenance,
        )
        if schema_version == 4
        else None
    )
    registration_sha256 = (
        _sha256(
            manifest["registration_sha256"],
            field="manifest.registration_sha256",
        )
        if schema_version == 4
        else None
    )
    cells = manifest["cells"]
    if not isinstance(cells, list) or not cells:
        raise RegisteredAggregationError("manifest.cells must be a non-empty list")

    expected = {
        (environment, model, seed)
        for environment in environments
        for model in models
        for seed in seeds
    }
    indexed: dict[tuple[str, str, int], Any] = {}
    seen_cell_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for index, raw_entry in enumerate(cells):
        field = f"manifest.cells[{index}]"
        entry = _mapping(raw_entry, field=field)
        _exact_keys(
            entry,
            _CELL_MANIFEST_KEYS_V4 if schema_version == 4 else _CELL_MANIFEST_KEYS,
            field=field,
        )
        environment = _string(entry["environment"], field=f"{field}.environment")
        model = _string(entry["model"], field=f"{field}.model")
        seed = _integer(entry["seed"], field=f"{field}.seed")
        identity = (environment, model, seed)
        if identity not in expected:
            raise RegisteredAggregationError(f"{field} is outside the frozen Cartesian matrix")
        if identity in indexed:
            raise RegisteredAggregationError(f"duplicate manifest cell identity: {identity!r}")
        configuration_sha256 = _sha256(
            entry["configuration_sha256"],
            field=f"{field}.configuration_sha256",
        )
        cell_id = _sha256(entry["cell_id"], field=f"{field}.cell_id")
        expected_cell_id = registered_cell_id(
            environment,
            model,
            seed,
            configuration_sha256,
        )
        if cell_id != expected_cell_id:
            raise RegisteredAggregationError(f"{field}.cell_id does not match its identity")
        if cell_id in seen_cell_ids:
            raise RegisteredAggregationError(f"duplicate manifest cell_id: {cell_id}")
        path = _artifact_path(
            manifest_path,
            entry["artifact_path"],
            field=f"{field}.artifact_path",
        )
        if path in seen_paths:
            raise RegisteredAggregationError(f"duplicate manifest artifact path: {path}")
        seen_cell_ids.add(cell_id)
        seen_paths.add(path)
        selection_metadata: dict[str, Any] = {}
        if tuning_selection is not None:
            selection = next(
                item
                for item in tuning_selection["selections"]
                if item["environment"] == environment and item["implementation_model"] == model
            )
            if (
                entry["candidate_id"] != selection["candidate_id"]
                or entry["model_family"] != selection["model_family"]
                or entry["implementation_model"] != model
                or entry["tuning_aggregate_sha256"] != tuning_selection["aggregate_sha256"]
                or entry["tuning_completion_index_sha256"]
                != tuning_selection["source_completion_index_sha256"]
                or entry["tuning_checksum_manifest_sha256"]
                != tuning_selection["source_checksum_manifest_sha256"]
                or entry["tuning_implementation_source_sha256"]
                != tuning_selection["source_implementation_sha256"]
                or entry["implementation_source_sha256"]
                != provenance["implementation_source"]["sha256"]
            ):
                raise RegisteredAggregationError(
                    f"{field} tuning-selection identity drifts from the manifest binding"
                )
            selection_metadata = {
                "candidate_id": selection["candidate_id"],
                "model_family": selection["model_family"],
                "implementation_model": model,
                "tuning_aggregate_sha256": tuning_selection["aggregate_sha256"],
                "tuning_completion_index_sha256": tuning_selection[
                    "source_completion_index_sha256"
                ],
                "tuning_checksum_manifest_sha256": tuning_selection[
                    "source_checksum_manifest_sha256"
                ],
                "tuning_implementation_source_sha256": tuning_selection[
                    "source_implementation_sha256"
                ],
                "implementation_source_sha256": provenance["implementation_source"]["sha256"],
                "learner": selection["learner"],
            }
        indexed[identity] = {
            "cell_id": cell_id,
            "configuration_sha256": configuration_sha256,
            "artifact_path": path,
            "artifact_relative_path": PurePosixPath(entry["artifact_path"]).as_posix(),
            **selection_metadata,
        }
    missing = sorted(expected - set(indexed))
    if missing:
        raise RegisteredAggregationError(f"manifest is missing Cartesian cells: {missing}")
    if len(indexed) != len(expected):
        raise RegisteredAggregationError("manifest cell count does not match Cartesian matrix")

    normalized = {
        "schema_version": manifest["schema_version"],
        "manifest_sha256": manifest_sha256,
        "matrix_kind": matrix_kind,
        "models": models,
        "environments": environments,
        "seeds": seeds,
        "provenance": provenance,
        "registration_sha256": registration_sha256,
        "tuning_selection": tuning_selection,
    }
    return normalized, indexed


def _validate_artifact(
    path: Path,
    *,
    identity: tuple[str, str, int],
    expected: Mapping[str, Any],
    manifest_sha256: str,
    manifest_schema_version: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    environment, model, seed = identity
    field = f"artifact[{environment},{model},{seed}]"
    artifact = _mapping(_load_json(path, kind=field), field=field)
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
        "environment_source",
        "environment_reference",
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
        raise RegisteredAggregationError(f"{field} is missing required fields: {missing}")
    if reference_implementation_for_model(model) is not None:
        try:
            validate_required_reference_implementation(
                model,
                artifact.get("reference_implementation"),
                field=f"{field}.reference_implementation",
            )
        except ValueError as error:
            raise RegisteredAggregationError(str(error)) from error
    expected_policy_contract = policy_contract_metadata_for_model(model)
    if (
        requires_explicit_policy_contract(model)
        or any(name in artifact for name in expected_policy_contract)
    ):
        try:
            validate_policy_contract_metadata(
                model,
                {name: artifact.get(name) for name in expected_policy_contract},
                field=f"{field}.policy_contract",
            )
            validate_policy_core_contract(
                model,
                artifact.get("policy_core"),
                field=f"{field}.policy_core",
            )
        except ValueError as error:
            raise RegisteredAggregationError(str(error)) from error
    if manifest_schema_version == 4:
        for name in (
            "candidate_id",
            "model_family",
            "implementation_model",
            "tuning_aggregate_sha256",
            "tuning_completion_index_sha256",
            "tuning_checksum_manifest_sha256",
            "tuning_implementation_source_sha256",
            "implementation_source_sha256",
        ):
            if name not in artifact:
                raise RegisteredAggregationError(f"{field} is missing {name}")
    expected_artifact_schema = (
        8 if manifest_schema_version == 4 else 5 if manifest_schema_version == 2 else 4
    )
    if artifact["schema_version"] != expected_artifact_schema:
        raise RegisteredAggregationError(
            f"{field}.schema_version must equal current schema {expected_artifact_schema}"
        )
    if artifact["status"] != "registered_final_complete":
        raise RegisteredAggregationError(f"{field}.status must equal 'registered_final_complete'")
    if (
        _sha256(
            artifact["matrix_manifest_sha256"],
            field=f"{field}.matrix_manifest_sha256",
        )
        != manifest_sha256
    ):
        raise RegisteredAggregationError(f"{field} belongs to a different matrix manifest")
    if _sha256(artifact["cell_id"], field=f"{field}.cell_id") != expected["cell_id"]:
        raise RegisteredAggregationError(f"{field}.cell_id drifted from the manifest")
    artifact_configuration_sha256 = _sha256(
        artifact["configuration_sha256"],
        field=f"{field}.configuration_sha256",
    )
    if artifact_configuration_sha256 != expected["configuration_sha256"]:
        raise RegisteredAggregationError(f"{field}.configuration_sha256 drifted from the manifest")
    configuration = _mapping(artifact["configuration"], field=f"{field}.configuration")
    if configuration.get("schema_version") != manifest_schema_version:
        raise RegisteredAggregationError(
            f"{field}.configuration schema does not match the manifest"
        )
    if canonical_json_sha256(configuration) != artifact_configuration_sha256:
        raise RegisteredAggregationError(
            f"{field}.configuration_sha256 does not match configuration"
        )
    artifact_environment = _string(
        artifact["environment"],
        field=f"{field}.environment",
    )
    artifact_model = _string(artifact["model"], field=f"{field}.model")
    artifact_seed = _integer(artifact["seed"], field=f"{field}.seed")
    if artifact_environment != environment or artifact_model != model or artifact_seed != seed:
        raise RegisteredAggregationError(f"{field} identity does not match the manifest")
    if manifest_schema_version == 4 and (
        artifact["candidate_id"] != expected["candidate_id"]
        or artifact["model_family"] != expected["model_family"]
        or artifact["implementation_model"] != expected["implementation_model"]
        or artifact["tuning_aggregate_sha256"] != expected["tuning_aggregate_sha256"]
        or artifact["tuning_completion_index_sha256"] != expected["tuning_completion_index_sha256"]
        or artifact["tuning_checksum_manifest_sha256"]
        != expected["tuning_checksum_manifest_sha256"]
        or artifact["tuning_implementation_source_sha256"]
        != expected["tuning_implementation_source_sha256"]
        or artifact["implementation_source_sha256"] != expected["implementation_source_sha256"]
    ):
        raise RegisteredAggregationError(
            f"{field} tuning-selection identity drifts from the manifest"
        )
    artifact_provenance = _validate_provenance(
        artifact["provenance"],
        field=f"{field}.provenance",
        require_implementation_source=manifest_schema_version == 4,
    )
    if artifact_provenance != provenance:
        raise RegisteredAggregationError(f"{field}.provenance drifted from the manifest")
    total_steps, evaluation_episodes, evaluation_horizon = _validate_frozen_configuration(
        configuration,
        identity=identity,
        provenance=provenance,
        selection=expected if manifest_schema_version == 4 else None,
        field=f"{field}.configuration",
    )
    for name in (
        "environment_source",
        "environment_reference",
        "parameter_count",
        "effective_parameter_count",
        "arcmind_target_parameter_count",
        "parameter_ratio",
    ):
        if artifact[name] != configuration[name]:
            raise RegisteredAggregationError(
                f"{field}.{name} does not match the frozen configuration"
            )
    for name in policy_contract_metadata_for_model(model):
        if name in configuration or name in artifact:
            if artifact.get(name) != configuration.get(name):
                raise RegisteredAggregationError(
                    f"{field}.{name} does not match the frozen configuration"
                )
    if requires_explicit_policy_contract(model) and artifact.get(
        "policy_core"
    ) != configuration.get("policy_core"):
        raise RegisteredAggregationError(
            f"{field}.policy_core does not match the frozen configuration"
        )
    actual_environment_steps = _integer(
        artifact["actual_environment_steps"],
        field=f"{field}.actual_environment_steps",
        positive=True,
    )
    if actual_environment_steps != total_steps:
        raise RegisteredAggregationError(
            f"{field}.actual_environment_steps does not match the registered budget"
        )
    if manifest_schema_version in {2, 4}:
        for name in (
            "comparison_profile",
            "requested_environment_steps",
            "realized_environment_steps",
        ):
            if name not in artifact:
                raise RegisteredAggregationError(f"{field} is missing {name}")
        if (
            artifact["comparison_profile"] != configuration["comparison_profile"]
            or artifact["requested_environment_steps"]
            != configuration["requested_environment_steps"]
            or artifact["realized_environment_steps"] != configuration["realized_environment_steps"]
            or artifact["realized_environment_steps"] != actual_environment_steps
        ):
            raise RegisteredAggregationError(
                f"{field} comparison profile or step accounting is inconsistent"
            )
    artifact_ppo = _mapping(artifact["ppo"], field=f"{field}.ppo")
    if artifact_ppo != configuration["ppo"]:
        raise RegisteredAggregationError(f"{field}.ppo does not match the frozen configuration")
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
    actual_evaluation_steps = _integer(
        artifact["actual_evaluation_steps_per_environment"],
        field=f"{field}.actual_evaluation_steps_per_environment",
        positive=True,
    )
    expected_evaluation_steps = evaluation_episodes * evaluation_horizon
    if (
        artifact_evaluation_episodes != evaluation_episodes
        or artifact_evaluation_horizon != evaluation_horizon
        or actual_evaluation_steps != expected_evaluation_steps
    ):
        raise RegisteredAggregationError(
            f"{field} evaluation contract does not match the frozen configuration"
        )
    evaluation, evaluation_counts = _validate_evaluation(artifact, field=field)
    _, episodes_per_environment, num_environments = evaluation_counts
    if (
        episodes_per_environment != evaluation_episodes
        or evaluation["scan_steps_per_environment"] != expected_evaluation_steps
    ):
        raise RegisteredAggregationError(
            f"{field}.evaluation does not match the frozen evaluation contract"
        )
    expected_transitions = expected_evaluation_steps * num_environments
    actual_evaluation_transitions = _integer(
        artifact["actual_evaluation_transitions"],
        field=f"{field}.actual_evaluation_transitions",
        positive=True,
    )
    if actual_evaluation_transitions != expected_transitions:
        raise RegisteredAggregationError(f"{field}.actual_evaluation_transitions is inconsistent")
    ppo = _mapping(configuration["ppo"], field=f"{field}.configuration.ppo")
    configured_num_environments = _integer(
        ppo.get("num_envs"),
        field=f"{field}.configuration.ppo.num_envs",
        positive=True,
    )
    if configured_num_environments != num_environments:
        raise RegisteredAggregationError(f"{field}.evaluation.num_environments does not match PPO")
    steps, curve_returns = _validate_training_history(
        artifact,
        field=field,
        expected_final_steps=total_steps,
    )
    _validate_final_training_metrics(artifact, field=field)
    return {
        "seed": seed,
        "evaluation": evaluation,
        "evaluation_counts": evaluation_counts,
        "steps": steps,
        "curve_returns": curve_returns,
        "registration_contract": {
            "ppo": dict(ppo),
            "evaluation_episodes_per_environment": evaluation_episodes,
            "comparison_profile": configuration.get("comparison_profile"),
        },
        "parameter_contract": parameter_contract_for_model(model),
        "comparison_role": comparison_role_for_model(model),
    }


def _group_record(
    environment: str,
    model: str,
    seeds: tuple[int, ...],
    records: Mapping[tuple[str, str, int], Mapping[str, Any]],
    steps: tuple[int, ...],
    curve_start_index: int,
) -> dict[str, Any]:
    cells = [records[(environment, model, seed)] for seed in seeds]
    mean_returns = [cell["evaluation"]["mean_return"] for cell in cells]
    median_returns = [cell["evaluation"]["median_return"] for cell in cells]
    raw_seed_values = [
        {
            "seed": seed,
            "mean_return": cell["evaluation"]["mean_return"],
            "median_return": cell["evaluation"]["median_return"],
            "returns_by_environment": cell["evaluation"]["returns_by_environment"],
        }
        for seed, cell in zip(seeds, cells)
    ]
    curve_matrix = np.asarray(
        [cell["curve_returns"][curve_start_index:] for cell in cells],
        dtype=np.float64,
    )
    retained_steps = steps[curve_start_index:]
    auc_values = [_trapezoid_by_step(row, retained_steps) for row in curve_matrix]
    return {
        "environment": environment,
        "model": model,
        "parameter_contract": cells[0]["parameter_contract"],
        "comparison_role": cells[0]["comparison_role"],
        "seeds": list(seeds),
        "raw_seed_values": raw_seed_values,
        "final_seed_mean_return": _summary(
            mean_returns,
            stream=f"final:{environment}:{model}",
        ),
        "raw_seed_median_returns": median_returns,
        "training_curve": {
            "environment_steps": list(retained_steps),
            "raw_seed_returns": [
                {
                    "seed": seed,
                    "mean_recent_return": [float(value) for value in row],
                    "auc_return_step": auc,
                }
                for seed, row, auc in zip(seeds, curve_matrix, auc_values)
            ],
            "mean_return_by_step": [float(value) for value in np.mean(curve_matrix, axis=0)],
            "median_return_by_step": [float(value) for value in np.median(curve_matrix, axis=0)],
            "iqm_return_by_step": [
                interquartile_mean(curve_matrix[:, index]) for index in range(curve_matrix.shape[1])
            ],
            "auc_return_step": _summary(
                auc_values,
                stream=f"curve_auc:{environment}:{model}",
            ),
        },
    }


def _paired_record(
    environment: str,
    model: str,
    seeds: tuple[int, ...],
    records: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    raw: list[dict[str, Any]] = []
    differences: list[float] = []
    for seed in seeds:
        model_return = records[(environment, model, seed)]["evaluation"]["mean_return"]
        arcmind_return = records[(environment, "arcmind", seed)]["evaluation"]["mean_return"]
        difference = model_return - arcmind_return
        differences.append(difference)
        raw.append(
            {
                "seed": seed,
                "model_mean_return": model_return,
                "arcmind_mean_return": arcmind_return,
                "difference": difference,
            }
        )
    return {
        "environment": environment,
        "model": model,
        "comparison_role": comparison_role_for_model(model),
        "reference_model": "arcmind",
        "seeds": list(seeds),
        "raw_seed_differences": raw,
        "difference_summary": _summary(
            differences,
            stream=f"paired:{environment}:{model}:arcmind",
        ),
    }


def _validate_registered_registration(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    registration_path = manifest_path.parent / "registration.json"
    raw = _mapping(
        _load_json(registration_path, kind="frozen registration"),
        field="registration",
    )
    try:
        expected_fields = registration_fields(raw.get("schema_version"))
    except ValueError as error:
        raise RegisteredAggregationError(str(error)) from error
    _exact_keys(raw, expected_fields, field="registration")
    environment_ids = [
        item.get("id") if isinstance(item, Mapping) else None for item in raw["environments"]
    ]
    if (
        raw["schema_version"] != manifest["schema_version"]
        or raw["status"] != "frozen"
        or raw["evidence_tier"] != "registered_final"
        or raw["matrix_kind"] != manifest["matrix_kind"]
        or tuple(raw["models"]) != manifest["models"]
        or tuple(environment_ids) != manifest["environments"]
        or tuple(raw["seeds"]) != manifest["seeds"]
    ):
        raise RegisteredAggregationError("frozen registration identity drifts from manifest")
    if manifest["schema_version"] == 4:
        try:
            registration_binding = normalize_final_selection_binding(raw["tuning_selection"])
        except ValueError as error:
            raise RegisteredAggregationError(str(error)) from error
        if registration_binding != manifest["tuning_selection"]:
            raise RegisteredAggregationError(
                "frozen registration tuning selection drifts from manifest"
            )
    try:
        comparison_profile = validate_comparison_profile(raw)
    except ValueError as error:
        raise RegisteredAggregationError(str(error)) from error
    normalized_environments: list[dict[str, Any]] = []
    for index, value in enumerate(raw["environments"]):
        field = f"registration.environments[{index}]"
        environment = _mapping(value, field=field)
        _exact_keys(environment, {"id", "total_steps"}, field=field)
        normalized_environments.append(
            {
                "id": _string(environment["id"], field=f"{field}.id"),
                "total_steps": _integer(
                    environment["total_steps"],
                    field=f"{field}.total_steps",
                    positive=True,
                ),
            }
        )
    evaluation_episodes = _integer(
        raw["evaluation_episodes_per_env"],
        field="registration.evaluation_episodes_per_env",
        positive=True,
    )
    require_gpu = raw["require_gpu"]
    quick = raw["quick"]
    if not isinstance(require_gpu, bool):
        raise RegisteredAggregationError("registration.require_gpu must be a boolean")
    if not isinstance(quick, bool):
        raise RegisteredAggregationError("registration.quick must be a boolean")
    if quick:
        raise RegisteredAggregationError("registered-final registration.quick must be false")
    normalized_learner: dict[str, int | float | bool] | None = None
    if manifest["schema_version"] in {1, 2}:
        try:
            normalized_learner = normalize_learner(
                raw["learner"],
                schema_version=manifest["schema_version"],
            )
        except ValueError as error:
            raise RegisteredAggregationError(str(error)) from error
    if registration_path.read_bytes() != canonical_json_bytes(raw) + b"\n":
        raise RegisteredAggregationError("frozen registration is not canonical JSON")
    registration_file_sha256 = sha256_file(registration_path)
    if (
        manifest["schema_version"] == 4
        and registration_file_sha256 != manifest["registration_sha256"]
    ):
        raise RegisteredAggregationError(
            "frozen registration SHA256 drifts from the frozen manifest"
        )
    return registration_file_sha256, {
        **dict(raw),
        "comparison_profile": comparison_profile,
        "environments": tuple(normalized_environments),
        "evaluation_episodes_per_env": evaluation_episodes,
        "require_gpu": require_gpu,
        "quick": quick,
        "learner": normalized_learner,
    }


def _validate_registration_against_artifacts(
    registration: Mapping[str, Any],
    manifest: Mapping[str, Any],
    records: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> None:
    """Bind every executable registration field to validated cell artifacts."""

    environment_budgets = {
        environment["id"]: environment["total_steps"]
        for environment in registration["environments"]
    }
    if set(environment_budgets) != set(manifest["environments"]):
        raise RegisteredAggregationError(
            "frozen registration environment budgets drift from the manifest"
        )
    schema_version = manifest["schema_version"]
    for identity, record in records.items():
        field = f"artifact[{identity[0]},{identity[1]},{identity[2]}].configuration"
        contract = _mapping(
            record["registration_contract"],
            field=f"{field}.registration_contract",
        )
        ppo = _mapping(contract["ppo"], field=f"{field}.ppo")
        configured_steps = _integer(
            ppo.get("total_steps"),
            field=f"{field}.ppo.total_steps",
            positive=True,
        )
        if configured_steps != environment_budgets[identity[0]]:
            raise RegisteredAggregationError(
                f"{field}.ppo.total_steps drifts from the frozen registration"
            )
        if (
            contract["evaluation_episodes_per_environment"]
            != registration["evaluation_episodes_per_env"]
        ):
            raise RegisteredAggregationError(
                f"{field} evaluation episodes drift from the frozen registration"
            )
        if schema_version in {1, 2}:
            expected_fields = set(registration["learner"])
            missing_fields = sorted(expected_fields - set(ppo))
            if missing_fields:
                raise RegisteredAggregationError(
                    f"{field}.ppo is missing frozen registration learner fields: {missing_fields}"
                )
            try:
                artifact_learner = normalize_learner(
                    {name: ppo[name] for name in expected_fields},
                    schema_version=schema_version,
                )
            except ValueError as error:
                raise RegisteredAggregationError(str(error)) from error
            if artifact_learner != registration["learner"]:
                raise RegisteredAggregationError(
                    f"{field}.ppo learner drifts from the frozen registration"
                )
        if (
            schema_version in {2, 4}
            and contract["comparison_profile"] != registration["comparison_profile"]
        ):
            raise RegisteredAggregationError(
                f"{field}.comparison_profile drifts from the frozen registration"
            )
    if (
        registration["require_gpu"]
        and manifest["provenance"]["runtime_contract"]["jax_backend"] != "gpu"
    ):
        raise RegisteredAggregationError(
            "frozen registration requires GPU but validated artifacts used another backend"
        )


def _validate_registered_completion_and_checksums(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    cells: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> dict[str, str]:
    root = manifest_path.parent
    completion_path = root / "completion_index.json"
    completion = _mapping(
        _load_json(completion_path, kind="completion index"),
        field="completion_index",
    )
    _exact_keys(completion, _COMPLETION_KEYS, field="completion_index")
    expected_count = len(cells)
    if (
        completion["schema_version"] != 1
        or completion["status"] != "complete"
        or completion["manifest_sha256"] != manifest["manifest_sha256"]
        or completion["planned_cells"] != expected_count
        or completion["completed_cells"] != expected_count
        or not isinstance(completion["cells"], list)
        or len(completion["cells"]) != expected_count
    ):
        raise RegisteredAggregationError(
            "completion_index is incomplete or belongs to another manifest"
        )
    if completion_path.read_bytes() != canonical_json_bytes(completion) + b"\n":
        raise RegisteredAggregationError("completion_index is not canonical JSON")
    indexed: set[tuple[str, str, int]] = set()
    canonical_paths = {
        (manifest_path.relative_to(root)).as_posix(),
        "registration.json",
        "completion_index.json",
    }
    cell_keys = _CELL_MANIFEST_KEYS_V4 if manifest["schema_version"] == 4 else _CELL_MANIFEST_KEYS
    completed_keys = (
        _COMPLETED_CELL_KEYS_V4 if manifest["schema_version"] == 4 else _COMPLETED_CELL_KEYS
    )
    for index, raw_cell in enumerate(completion["cells"]):
        field = f"completion_index.cells[{index}]"
        cell = _mapping(raw_cell, field=field)
        _exact_keys(cell, completed_keys, field=field)
        identity = (
            _string(cell["environment"], field=f"{field}.environment"),
            _string(cell["model"], field=f"{field}.model"),
            _integer(cell["seed"], field=f"{field}.seed"),
        )
        if identity not in cells or identity in indexed:
            raise RegisteredAggregationError(f"{field} has an invalid or duplicate identity")
        indexed.add(identity)
        expected = cells[identity]
        for key in cell_keys:
            if key == "artifact_path":
                expected_value = expected["artifact_relative_path"]
            elif key == "environment":
                expected_value = identity[0]
            elif key == "model":
                expected_value = identity[1]
            elif key == "seed":
                expected_value = identity[2]
            else:
                expected_value = expected[key]
            if cell[key] != expected_value:
                raise RegisteredAggregationError(f"{field}.{key} drifts from manifest")
        if sha256_file(expected["artifact_path"]) != _sha256(
            cell["artifact_sha256"],
            field=f"{field}.artifact_sha256",
        ):
            raise RegisteredAggregationError(f"{field}.artifact_sha256 is incorrect")
        log_path = _artifact_path(
            manifest_path,
            cell["log_path"],
            field=f"{field}.log_path",
        )
        if log_path != expected["artifact_path"].with_suffix(".log"):
            raise RegisteredAggregationError(f"{field}.log_path is not canonical")
        try:
            actual_log_sha256 = sha256_file(log_path)
        except OSError as error:
            raise RegisteredAggregationError(
                f"{field}.log_path does not exist: {log_path}"
            ) from error
        if actual_log_sha256 != _sha256(cell["log_sha256"], field=f"{field}.log_sha256"):
            raise RegisteredAggregationError(f"{field}.log_sha256 is incorrect")
        canonical_paths.add(expected["artifact_relative_path"])
        canonical_paths.add(PurePosixPath(cell["log_path"]).as_posix())
    if indexed != set(cells):
        raise RegisteredAggregationError("completion_index is missing manifest cells")
    try:
        checksum_entries = validate_checksum_manifest(root)
    except ArtifactChecksumError as error:
        raise RegisteredAggregationError(str(error)) from error
    checksum_paths = {relative for relative, _ in checksum_entries}
    if checksum_paths != canonical_paths:
        raise RegisteredAggregationError(
            "checksum inventory does not match canonical matrix evidence: "
            f"missing={sorted(canonical_paths - checksum_paths)}, "
            f"extra={sorted(checksum_paths - canonical_paths)}"
        )
    return {
        "completion_index_sha256": sha256_file(completion_path),
        "checksum_manifest_sha256": sha256_file(root / "checksums.sha256"),
    }


def build_registered_aggregate(manifest_path: str | Path) -> dict[str, Any]:
    """Validate and aggregate one complete frozen registered matrix."""

    path = Path(manifest_path).resolve()
    manifest, cell_index = _validate_manifest(path)
    registration_file_sha256, registration = _validate_registered_registration(path, manifest)
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    evaluation_counts_by_environment: dict[str, tuple[int, int, int]] = {}
    step_grid_by_environment: dict[str, tuple[int, ...]] = {}
    for environment in manifest["environments"]:
        for model in manifest["models"]:
            for seed in manifest["seeds"]:
                identity = (environment, model, seed)
                record = _validate_artifact(
                    cell_index[identity]["artifact_path"],
                    identity=identity,
                    expected=cell_index[identity],
                    manifest_sha256=manifest["manifest_sha256"],
                    manifest_schema_version=manifest["schema_version"],
                    provenance=manifest["provenance"],
                )
                if environment not in evaluation_counts_by_environment:
                    evaluation_counts_by_environment[environment] = record["evaluation_counts"]
                elif record["evaluation_counts"] != evaluation_counts_by_environment[environment]:
                    raise RegisteredAggregationError(
                        "registered cells have unequal evaluation episode counts "
                        f"within environment {environment!r}"
                    )
                if environment not in step_grid_by_environment:
                    step_grid_by_environment[environment] = record["steps"]
                elif record["steps"] != step_grid_by_environment[environment]:
                    raise RegisteredAggregationError(
                        "registered cells have unequal training step grids "
                        f"within environment {environment!r}"
                    )
                records[identity] = record
    _validate_registration_against_artifacts(registration, manifest, records)
    raw_integrity = _validate_registered_completion_and_checksums(
        path,
        manifest,
        cell_index,
    )
    if not evaluation_counts_by_environment or not step_grid_by_environment:  # pragma: no cover
        raise AssertionError("non-empty matrix did not produce aggregation metadata")

    curve_start_by_environment: dict[str, int] = {}
    for environment in manifest["environments"]:
        first_finite_indices: list[int] = []
        for model in manifest["models"]:
            for seed in manifest["seeds"]:
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
                    raise RegisteredAggregationError(
                        "registered matrix has no shared finite training-curve suffix "
                        f"for environment {environment!r}: cell {identity!r} "
                        "has no finite return"
                    )
                first_finite_indices.append(first_finite)
        curve_start_index = max(first_finite_indices)
        retained_step_grid = step_grid_by_environment[environment][curve_start_index:]
        if not retained_step_grid:  # pragma: no cover - selected from a finite value
            raise RegisteredAggregationError(
                "registered matrix has no shared finite training-curve suffix "
                f"for environment {environment!r}"
            )
        curve_start_by_environment[environment] = curve_start_index

    groups = [
        _group_record(
            environment,
            model,
            manifest["seeds"],
            records,
            step_grid_by_environment[environment],
            curve_start_by_environment[environment],
        )
        for environment in manifest["environments"]
        for model in manifest["models"]
    ]
    paired = (
        [
            _paired_record(
                environment,
                model,
                manifest["seeds"],
                records,
            )
            for environment in manifest["environments"]
            for model in manifest["models"]
            if model != "arcmind"
            and comparison_role_for_model(model) != SUPPLEMENTAL_COMPARISON_ROLE
        ]
        if manifest["matrix_kind"] == "primary_comparison"
        else []
    )
    supplemental_paired = (
        [
            {
                **_paired_record(
                    environment,
                    model,
                    manifest["seeds"],
                    records,
                ),
                "comparison_role": SUPPLEMENTAL_COMPARISON_ROLE,
            }
            for environment in manifest["environments"]
            for model in manifest["models"]
            if comparison_role_for_model(model) == SUPPLEMENTAL_COMPARISON_ROLE
        ]
        if manifest["matrix_kind"] == "primary_comparison"
        else []
    )
    environment_contracts = []
    for environment in manifest["environments"]:
        episodes, episodes_per_environment, num_environments = evaluation_counts_by_environment[
            environment
        ]
        full_grid = step_grid_by_environment[environment]
        curve_start_index = curve_start_by_environment[environment]
        retained_grid = full_grid[curve_start_index:]
        environment_contracts.append(
            {
                "environment": environment,
                "evaluation": {
                    "episodes_per_cell": episodes,
                    "episodes_per_environment": episodes_per_environment,
                    "vector_environments": num_environments,
                },
                "training_curve": {
                    "full_environment_step_grid": list(full_grid),
                    "curve_start_step": retained_grid[0],
                    "excluded_prefix_length": curve_start_index,
                    "retained_environment_step_grid": list(retained_grid),
                },
            }
        )
    result = {
        "schema_version": 1,
        "status": "registered_matrix_aggregate",
        "matrix_manifest_sha256": manifest["manifest_sha256"],
        "raw_integrity": {
            "registration_file_sha256": registration_file_sha256,
            **raw_integrity,
            "completion_index_validated": True,
            "checksum_inventory_validated": True,
            "reference_implementation_validated": True,
            "parameter_contract_validated": True,
        },
        "matrix_kind": manifest["matrix_kind"],
        "provenance": manifest["provenance"],
        "models": list(manifest["models"]),
        "environments": list(manifest["environments"]),
        "seeds": list(manifest["seeds"]),
        "statistical_unit": "seed",
        "environment_contracts": environment_contracts,
        "statistics": {
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence_level": CONFIDENCE_LEVEL,
            "bootstrap_interval": "percentile",
            "bootstrap_unit": "seed",
            "iqm": (
                "Sort seed values, weight each empirical interval by its overlap "
                "with probability mass [0.25, 0.75], and divide by 0.5."
            ),
            "curve_integration": (
                "Trapezoidal return by environment step over the shared complete-case "
                "suffix, with no extrapolation before or after the retained grid."
            ),
        },
        "groups": groups,
        "paired_differences_against_arcmind": paired,
        "supplemental_paired_differences_against_arcmind": supplemental_paired,
    }
    if manifest["tuning_selection"] is not None:
        result["tuning_selection_binding"] = {
            "raw_matrix_path": manifest["tuning_selection"]["raw_matrix_path"],
            "aggregate_path": manifest["tuning_selection"]["aggregate_path"],
            "aggregate_sha256": manifest["tuning_selection"]["aggregate_sha256"],
            "source_registration_sha256": manifest["tuning_selection"][
                "source_registration_sha256"
            ],
            "source_manifest_sha256": manifest["tuning_selection"]["source_manifest_sha256"],
            "source_completion_index_sha256": manifest["tuning_selection"][
                "source_completion_index_sha256"
            ],
            "source_checksum_manifest_sha256": manifest["tuning_selection"][
                "source_checksum_manifest_sha256"
            ],
            "source_implementation_sha256": manifest["tuning_selection"][
                "source_implementation_sha256"
            ],
            "validated": True,
            "final_seeds_disjoint_from_tuning": True,
            "selections": [
                {
                    "environment": selection["environment"],
                    "model_family": selection["model_family"],
                    "implementation_model": selection["implementation_model"],
                    "candidate_id": selection["candidate_id"],
                    "learner": selection["learner"],
                    "implementation_source_sha256": selection["implementation_source_sha256"],
                }
                for selection in manifest["tuning_selection"]["selections"]
            ],
        }
    return result


def aggregate_registered(
    manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build an aggregate and atomically create its canonical JSON artifact."""

    manifest = Path(manifest_path).resolve()
    raw_matrix_root = manifest.parent
    output = Path(output_path).resolve()
    if output == raw_matrix_root or output.is_relative_to(raw_matrix_root):
        raise RegisteredAggregationError("aggregate output must be outside the raw matrix root")
    aggregate = build_registered_aggregate(manifest)
    atomic_write_json(output, aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    result = aggregate_registered(arguments.manifest, arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "matrix_manifest_sha256": result["matrix_manifest_sha256"],
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
    "RegisteredAggregationError",
    "aggregate_registered",
    "build_registered_aggregate",
    "interquartile_mean",
]
