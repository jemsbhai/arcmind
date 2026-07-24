"""Focused schema-7 registered aggregation tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import benchmarks.pobax.aggregate_registered as aggregate_module
from benchmarks.pobax.aggregate_registered import (
    RegisteredAggregationError,
    _schema7_cell_contract,
    _validate_artifact,
    _validate_manifest,
    _validate_registered_completion_and_checksums,
    build_registered_aggregate,
    validate_bound_compute_aware_primary_matrix,
)
from benchmarks.pobax.model_registry import (
    comparison_role_for_model,
    parameter_contract_for_model,
)
from benchmarks.pobax.registered_artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    canonical_json_sha256,
    registered_cell_id,
    sha256_file,
    write_checksum_manifest,
)
from benchmarks.pobax.registration_protocol import (
    COMPUTE_AWARE_FINAL_SEEDS,
    COMPUTE_AWARE_UPPER_REFERENCE_PANEL,
    normalize_memoryless_learner_binding,
    normalize_primary_matrix_binding,
)
from benchmarks.pobax.tests.test_compute_aware_upper_registration import (
    _implementation_source,
    _memoryless_binding,
    _primary_aggregate,
    _primary_binding,
)
from benchmarks.pobax.upper_reference_registry import (
    expected_environment_reference,
    expected_environment_source,
)

OPTIMIZER_METRICS = {
    "loss": 0.5,
    "actor_loss": 0.1,
    "value_loss": 0.8,
    "entropy": 0.2,
    "approximate_kl": 0.01,
}


def _provenance() -> dict[str, Any]:
    return {
        "git": {
            "commit": "1" * 40,
            "dirty": False,
            "diff_sha256": None,
        },
        "dependency_lock_sha256": "2" * 64,
        "pobax_commit": "3" * 40,
        "navix_commit": "4" * 40,
        "runtime_contract": {
            "python": {"implementation": "CPython", "version": "3.12.3"},
            "packages": {
                "brax": "0.14.2",
                "gymnax": "0.0.9",
                "jax": "0.6.2",
                "jax-cuda12-pjrt": "0.6.2",
                "jax-cuda12-plugin": "0.6.2",
                "jaxlib": "0.6.2",
                "Navix": "0.0.1",
                "numpy": "2.5.1",
                "optax": "0.2.8",
                "pobax": "0.0.1",
            },
            "jax_backend": "gpu",
            "jax_enable_x64": False,
            "devices": [
                {
                    "platform": "gpu",
                    "device_kind": "Test GPU",
                }
            ],
        },
        "implementation_source": _implementation_source(),
    }


def _bindings() -> tuple[dict[str, str], dict[str, Any]]:
    source_hash = _implementation_source()["sha256"]
    return (
        _primary_binding(source_hash),
        _memoryless_binding(source_hash),
    )


def _manifest(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    primary, memoryless = _bindings()
    contract = _schema7_cell_contract(
        primary_matrix_binding=normalize_primary_matrix_binding(primary),
        memoryless_learner_binding=normalize_memoryless_learner_binding(memoryless),
    )
    cells = []
    for cell_index, (environment, _) in enumerate(COMPUTE_AWARE_UPPER_REFERENCE_PANEL):
        for seed in COMPUTE_AWARE_FINAL_SEEDS:
            configuration_sha256 = canonical_json_sha256(
                {
                    "environment": environment,
                    "model": "memoryless_mlp",
                    "seed": seed,
                }
            )
            cells.append(
                {
                    "cell_id": registered_cell_id(
                        environment,
                        "memoryless_mlp",
                        seed,
                        configuration_sha256,
                    ),
                    "environment": environment,
                    "model": "memoryless_mlp",
                    "seed": seed,
                    "configuration_sha256": configuration_sha256,
                    "artifact_path": (f"cells/{cell_index:02d}-{seed}.json"),
                    **{name: value for name, value in contract.items() if name != "learner"},
                }
            )
    manifest = {
        "schema_version": 7,
        "status": "frozen",
        "matrix_kind": "upper_reference",
        "models": ["memoryless_mlp"],
        "environments": [environment for environment, _ in COMPUTE_AWARE_UPPER_REFERENCE_PANEL],
        "seeds": list(COMPUTE_AWARE_FINAL_SEEDS),
        "provenance": _provenance(),
        "registration_sha256": "5" * 64,
        "primary_matrix_binding": primary,
        "memoryless_learner_binding": memoryless,
        "cells": cells,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    root = tmp_path / "upper"
    path = root / "frozen_manifest.json"
    atomic_write_json(path, manifest)
    return path, manifest


def _patch_bound_primary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provenance = _provenance()

    def validate(primary, learner, **kwargs):
        return (
            normalize_primary_matrix_binding(primary),
            normalize_memoryless_learner_binding(learner),
            {"provenance": deepcopy(provenance)},
            (tmp_path / "primary").resolve(),
        )

    monkeypatch.setattr(
        aggregate_module,
        "validate_bound_compute_aware_primary_matrix",
        validate,
    )


def _rewrite_manifest(path: Path, value: dict[str, Any]) -> None:
    rewritten = deepcopy(value)
    rewritten.pop("manifest_sha256", None)
    rewritten["manifest_sha256"] = canonical_json_sha256(rewritten)
    path.write_bytes(canonical_json_bytes(rewritten) + b"\n")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cells: cells.pop(), "exactly 40"),
        (lambda cells: cells.append(deepcopy(cells[-1])), "exactly 40"),
        (
            lambda cells: cells.__setitem__(
                slice(0, 2),
                [cells[1], cells[0]],
            ),
            "exact upper-reference order",
        ),
    ],
)
def test_schema7_manifest_rejects_missing_extra_or_reordered_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    path, manifest = _manifest(tmp_path)
    _patch_bound_primary(monkeypatch, tmp_path)
    mutation(manifest["cells"])
    _rewrite_manifest(path, manifest)

    with pytest.raises(RegisteredAggregationError, match=message):
        _validate_manifest(path)


def _configuration(
    environment: str,
    seed: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    total_steps = dict(COMPUTE_AWARE_UPPER_REFERENCE_PANEL)[environment]
    return {
        "schema_version": 7,
        "evidence_tier": "registered_final",
        "environment": environment,
        "model": "memoryless_mlp",
        "seed": seed,
        **{
            name: value
            for name, value in contract.items()
            if name
            not in {
                "learner",
                "implementation_source_sha256",
            }
        },
        "implementation_source": _implementation_source(),
        "environment_source": expected_environment_source(environment),
        "environment_reference": expected_environment_reference(environment),
        "parameter_count": 100,
        "effective_parameter_count": 100,
        "arcmind_target_parameter_count": 100,
        "parameter_ratio": 1.0,
        "ppo": {
            **contract["learner"],
            "total_steps": total_steps,
            "step_budget_mode": "exact",
        },
        "evaluation_episodes_per_environment": 16,
        "evaluation_max_episode_steps": 10,
        "dependency_lock_sha256": _provenance()["dependency_lock_sha256"],
        "pobax_commit": _provenance()["pobax_commit"],
        "navix_commit": _provenance()["navix_commit"],
        "runtime_contract": _provenance()["runtime_contract"],
        "comparison_profile": "arcmind_shared_comparison",
        "requested_environment_steps": total_steps,
        "realized_environment_steps": total_steps,
    }


def _artifact(
    configuration: dict[str, Any],
    contract: dict[str, Any],
    *,
    manifest_sha256: str,
) -> dict[str, Any]:
    environment = configuration["environment"]
    seed = configuration["seed"]
    configuration_sha256 = canonical_json_sha256(configuration)
    horizon = configuration["evaluation_max_episode_steps"]
    total_steps = configuration["ppo"]["total_steps"]
    returns = [[float(environment_index)] * 16 for environment_index in range(8)]
    flat = [value for row in returns for value in row]
    return {
        "schema_version": 11,
        "status": "registered_final_complete",
        "matrix_manifest_sha256": manifest_sha256,
        "cell_id": registered_cell_id(
            environment,
            "memoryless_mlp",
            seed,
            configuration_sha256,
        ),
        "configuration_sha256": configuration_sha256,
        "configuration": configuration,
        "environment": environment,
        "model": "memoryless_mlp",
        "seed": seed,
        **{name: value for name, value in contract.items() if name != "learner"},
        "environment_source": configuration["environment_source"],
        "environment_reference": configuration["environment_reference"],
        "parameter_count": 100,
        "effective_parameter_count": 100,
        "arcmind_target_parameter_count": 100,
        "parameter_ratio": 1.0,
        "provenance": _provenance(),
        "actual_environment_steps": total_steps,
        "ppo": configuration["ppo"],
        "evaluation_episodes_per_environment": 16,
        "evaluation_max_episode_steps": horizon,
        "actual_evaluation_steps_per_environment": 16 * horizon,
        "actual_evaluation_transitions": 16 * horizon * 8,
        "comparison_profile": "arcmind_shared_comparison",
        "requested_environment_steps": total_steps,
        "realized_environment_steps": total_steps,
        "evaluation": {
            "mean_return": sum(flat) / len(flat),
            "median_return": 3.5,
            "episodes": 128,
            "episodes_per_environment": 16,
            "num_environments": 8,
            "scan_steps_per_environment": 16 * horizon,
            "returns_by_environment": returns,
        },
        "training": OPTIMIZER_METRICS,
        "training_history": [
            {
                **OPTIMIZER_METRICS,
                "environment_steps": total_steps // 2,
                "mean_recent_return": 1.0,
            },
            {
                **OPTIMIZER_METRICS,
                "environment_steps": total_steps,
                "mean_recent_return": 2.0,
            },
        ],
    }


def test_schema7_artifact_schema11_binds_primary_learner_and_environment(
    tmp_path: Path,
) -> None:
    primary, memoryless = _bindings()
    contract = _schema7_cell_contract(
        primary_matrix_binding=normalize_primary_matrix_binding(primary),
        memoryless_learner_binding=normalize_memoryless_learner_binding(memoryless),
    )
    configuration = _configuration(
        "tmaze_10-perfect-memory",
        COMPUTE_AWARE_FINAL_SEEDS[0],
        contract,
    )
    artifact = _artifact(
        configuration,
        contract,
        manifest_sha256="a" * 64,
    )
    path = tmp_path / "cell.json"
    atomic_write_json(path, artifact)
    expected = {
        "cell_id": artifact["cell_id"],
        "configuration_sha256": artifact["configuration_sha256"],
        **contract,
    }

    record = _validate_artifact(
        path,
        identity=(
            "tmaze_10-perfect-memory",
            "memoryless_mlp",
            COMPUTE_AWARE_FINAL_SEEDS[0],
        ),
        expected=expected,
        manifest_sha256="a" * 64,
        manifest_schema_version=7,
        provenance=_provenance(),
    )
    assert record["registration_contract"]["ppo"]["learning_rate"] == 0.00025

    artifact["primary_manifest_internal_sha256"] = "f" * 64
    path.write_bytes(canonical_json_bytes(artifact) + b"\n")
    with pytest.raises(RegisteredAggregationError, match="drifts from"):
        _validate_artifact(
            path,
            identity=(
                "tmaze_10-perfect-memory",
                "memoryless_mlp",
                COMPUTE_AWARE_FINAL_SEEDS[0],
            ),
            expected=expected,
            manifest_sha256="a" * 64,
            manifest_schema_version=7,
            provenance=_provenance(),
        )


def test_schema7_completion_and_checksum_inventory_close_over_40_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, raw_manifest = _manifest(tmp_path)
    _patch_bound_primary(monkeypatch, tmp_path)
    manifest, cells = _validate_manifest(path)
    atomic_write_json(path.parent / "registration.json", {"schema_version": 7})
    raw_cells = {
        (cell["environment"], cell["model"], cell["seed"]): cell for cell in raw_manifest["cells"]
    }
    completed = []
    for identity, cell in cells.items():
        atomic_write_json(cell["artifact_path"], {"identity": list(identity)})
        log_path = cell["artifact_path"].with_suffix(".log")
        atomic_write_bytes(log_path, b"schema-7 test log\n")
        completed.append(
            {
                **deepcopy(raw_cells[identity]),
                "artifact_sha256": sha256_file(cell["artifact_path"]),
                "log_path": log_path.relative_to(path.parent).as_posix(),
                "log_sha256": sha256_file(log_path),
            }
        )
    atomic_write_json(
        path.parent / "completion_index.json",
        {
            "schema_version": 1,
            "status": "complete",
            "manifest_sha256": manifest["manifest_sha256"],
            "planned_cells": 40,
            "completed_cells": 40,
            "cells": completed,
        },
    )
    write_checksum_manifest(path.parent)

    integrity = _validate_registered_completion_and_checksums(
        path,
        manifest,
        cells,
    )
    assert set(integrity) == {
        "completion_index_sha256",
        "checksum_manifest_sha256",
    }


def test_schema7_build_emits_schema3_without_primary_comparisons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = _manifest(tmp_path)
    _patch_bound_primary(monkeypatch, tmp_path)
    manifest, cells = _validate_manifest(path)
    memoryless = manifest["memoryless_learner_binding"]
    budgets = dict(COMPUTE_AWARE_UPPER_REFERENCE_PANEL)
    registration = {
        "environments": tuple(
            {"id": environment, "total_steps": total_steps}
            for environment, total_steps in COMPUTE_AWARE_UPPER_REFERENCE_PANEL
        ),
        "evaluation_episodes_per_env": 16,
        "require_gpu": True,
        "quick": False,
        "comparison_profile": "arcmind_shared_comparison",
        "memoryless_learner_binding": memoryless,
    }

    monkeypatch.setattr(
        aggregate_module,
        "_validate_manifest",
        lambda value: (manifest, cells),
    )
    monkeypatch.setattr(
        aggregate_module,
        "_validate_registered_registration",
        lambda *args: ("b" * 64, registration),
    )

    def validate_artifact(path, *, identity, **kwargs):
        environment, model, seed = identity
        total_steps = budgets[environment]
        value = float(seed - COMPUTE_AWARE_FINAL_SEEDS[0])
        ppo = {
            **memoryless["learner"],
            "total_steps": total_steps,
            "step_budget_mode": "exact",
        }
        return {
            "seed": seed,
            "evaluation": {
                "mean_return": value,
                "median_return": value,
                "returns_by_environment": [[value] * 16 for _ in range(8)],
            },
            "evaluation_counts": (128, 16, 8),
            "steps": (total_steps // 2, total_steps),
            "curve_returns": (0.0, value),
            "registration_contract": {
                "ppo": ppo,
                "evaluation_episodes_per_environment": 16,
                "comparison_profile": "arcmind_shared_comparison",
            },
            "parameter_contract": parameter_contract_for_model(model),
            "comparison_role": comparison_role_for_model(model),
        }

    monkeypatch.setattr(
        aggregate_module,
        "_validate_artifact",
        validate_artifact,
    )
    monkeypatch.setattr(
        aggregate_module,
        "_validate_registered_completion_and_checksums",
        lambda *args: {
            "completion_index_sha256": "c" * 64,
            "checksum_manifest_sha256": "d" * 64,
        },
    )

    result = build_registered_aggregate(path)

    assert result["schema_version"] == 3
    assert len(result["groups"]) == 4
    assert result["paired_differences_against_arcmind"] == []
    assert result["supplemental_paired_differences_against_arcmind"] == []
    assert result["raw_integrity"]["primary_matrix_binding_validated"] is True
    assert result["primary_matrix_binding"] == manifest["primary_matrix_binding"]
    assert result["upper_reference_alias_mapping"] == [
        {
            "upper_reference_environment": upper,
            "primary_environment": expected_environment_reference(upper)["primary_environment"],
        }
        for upper, _ in COMPUTE_AWARE_UPPER_REFERENCE_PANEL
    ]


def test_bound_primary_proves_bytes_and_rejects_root_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary, memoryless = _bindings()
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
        memoryless,
        _implementation_source(),
    )
    aggregate_path = tmp_path / "artifacts" / "primary" / "aggregate.json"
    atomic_write_json(aggregate_path, aggregate)
    primary["primary_aggregate_file_sha256"] = sha256_file(aggregate_path)
    monkeypatch.setattr(aggregate_module, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        aggregate_module,
        "build_registered_aggregate",
        lambda path: deepcopy(aggregate),
    )

    _, _, rebuilt, bound_root = validate_bound_compute_aware_primary_matrix(
        primary,
        memoryless,
        expected_primary_root=raw,
    )
    assert rebuilt == aggregate
    assert bound_root == raw.resolve()

    with pytest.raises(RegisteredAggregationError, match="supplied primary"):
        validate_bound_compute_aware_primary_matrix(
            primary,
            memoryless,
            expected_primary_root=tmp_path / "substituted-primary",
        )

    (raw / "completion_index.json").write_bytes(b"mutated\n")
    with pytest.raises(
        RegisteredAggregationError,
        match="primary_completion_index_file_sha256",
    ):
        validate_bound_compute_aware_primary_matrix(
            primary,
            memoryless,
            expected_primary_root=raw,
        )
