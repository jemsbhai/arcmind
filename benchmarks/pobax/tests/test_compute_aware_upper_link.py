"""Focused schema-6 to schema-7 upper-reference link tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import benchmarks.pobax.link_upper_reference as linker
from benchmarks.pobax.aggregate_registered import RegisteredAggregationError
from benchmarks.pobax.link_upper_reference import (
    UpperReferenceLinkError,
    build_upper_reference_link,
)
from benchmarks.pobax.registered_artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
)
from benchmarks.pobax.registration_protocol import (
    COMPUTE_AWARE_FINAL_PANEL,
    COMPUTE_AWARE_FINAL_SEEDS,
    COMPUTE_AWARE_UPPER_REFERENCE_PANEL,
    normalize_memoryless_learner_binding,
    normalize_primary_matrix_binding,
)
from benchmarks.pobax.tests.test_compute_aware_upper_aggregate import (
    _bindings,
    _provenance,
)
from benchmarks.pobax.tests.test_compute_aware_upper_registration import (
    _primary_aggregate,
)
from benchmarks.pobax.upper_reference_registry import (
    UPPER_TO_PRIMARY_ENVIRONMENT,
)


def _artifact_contract(
    *,
    total_steps: int,
    learning_rate: float = 0.00025,
) -> dict[str, Any]:
    learner = _bindings()[1]["learner"]
    ppo = {
        **learner,
        "learning_rate": learning_rate,
        "total_steps": total_steps,
        "step_budget_mode": "exact",
    }
    horizon = 10
    return {
        "configuration": {
            "ppo": ppo,
            "comparison_profile": "arcmind_shared_comparison",
            "requested_environment_steps": total_steps,
            "realized_environment_steps": total_steps,
        },
        "ppo": ppo,
        "actual_environment_steps": total_steps,
        "evaluation_episodes_per_environment": 16,
        "evaluation_max_episode_steps": horizon,
        "actual_evaluation_steps_per_environment": 16 * horizon,
        "actual_evaluation_transitions": 16 * horizon * 8,
        "evaluation": {
            "episodes": 128,
            "episodes_per_environment": 16,
            "num_environments": 8,
            "scan_steps_per_environment": 16 * horizon,
        },
    }


def _write_memoryless_lane(
    root: Path,
    panel: tuple[tuple[str, int], ...],
) -> dict[str, Any]:
    cells = []
    for environment, total_steps in panel:
        for seed in COMPUTE_AWARE_FINAL_SEEDS:
            relative = f"cells/{environment}-{seed}.json"
            atomic_write_json(
                root / relative,
                _artifact_contract(total_steps=total_steps),
            )
            cells.append(
                {
                    "environment": environment,
                    "model": "memoryless_mlp",
                    "seed": seed,
                    "artifact_path": relative,
                }
            )
    manifest = {
        "manifest_sha256": "a" * 64,
        "environments": [environment for environment, _ in panel],
        "seeds": list(COMPUTE_AWARE_FINAL_SEEDS),
        "cells": cells,
    }
    atomic_write_json(root / "frozen_manifest.json", manifest)
    atomic_write_bytes(root / "completion_index.json", b"completion\n")
    atomic_write_bytes(root / "checksums.sha256", b"checksums\n")
    return manifest


def _link_fixture(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
]:
    primary_root = tmp_path / "primary"
    upper_root = tmp_path / "upper"
    primary_binding, memoryless_binding = _bindings()
    primary_binding["raw_matrix_path"] = "primary"
    primary_binding["aggregate_path"] = "primary-aggregate.json"
    primary_registration = {
        "schema_version": 6,
        "evidence_tier": "registered_final",
        "comparison_profile": "arcmind_shared_comparison",
        "evaluation_episodes_per_env": 16,
        "require_gpu": True,
        "quick": False,
        "seeds": list(COMPUTE_AWARE_FINAL_SEEDS),
    }
    upper_registration = {
        "schema_version": 7,
        "evidence_tier": "registered_final",
        "comparison_profile": "arcmind_shared_comparison",
        "evaluation_episodes_per_env": 16,
        "require_gpu": True,
        "quick": False,
        "seeds": list(COMPUTE_AWARE_FINAL_SEEDS),
        "primary_matrix_binding": primary_binding,
        "memoryless_learner_binding": memoryless_binding,
    }
    atomic_write_json(primary_root / "registration.json", primary_registration)
    atomic_write_json(upper_root / "registration.json", upper_registration)
    _write_memoryless_lane(
        primary_root,
        COMPUTE_AWARE_FINAL_PANEL,
    )
    _write_memoryless_lane(
        upper_root,
        COMPUTE_AWARE_UPPER_REFERENCE_PANEL,
    )
    provenance = _provenance()
    primary_aggregate = _primary_aggregate(
        primary_binding,
        memoryless_binding,
        provenance["implementation_source"],
    )
    primary_aggregate["provenance"] = deepcopy(provenance)
    alias_mapping = [
        {
            "upper_reference_environment": environment,
            "primary_environment": UPPER_TO_PRIMARY_ENVIRONMENT[environment],
        }
        for environment, _ in COMPUTE_AWARE_UPPER_REFERENCE_PANEL
    ]
    upper_aggregate = {
        "schema_version": 3,
        "matrix_kind": "upper_reference",
        "raw_integrity": {"primary_matrix_binding_validated": True},
        "primary_matrix_binding": deepcopy(primary_binding),
        "memoryless_learner_binding": deepcopy(memoryless_binding),
        "environments": [environment for environment, _ in COMPUTE_AWARE_UPPER_REFERENCE_PANEL],
        "models": ["memoryless_mlp"],
        "groups": [
            {
                "environment": environment,
                "model": "memoryless_mlp",
            }
            for environment, _ in COMPUTE_AWARE_UPPER_REFERENCE_PANEL
        ],
        "paired_differences_against_arcmind": [],
        "supplemental_paired_differences_against_arcmind": [],
        "upper_reference_alias_mapping": alias_mapping,
        "provenance": deepcopy(provenance),
    }
    return (
        primary_root,
        upper_root,
        primary_aggregate,
        upper_aggregate,
        primary_binding,
        memoryless_binding,
    )


def _patch_aggregates(
    monkeypatch: pytest.MonkeyPatch,
    *,
    primary_root: Path,
    primary_aggregate: dict[str, Any],
    upper_aggregate: dict[str, Any],
    primary_binding: dict[str, str],
    memoryless_binding: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        linker,
        "build_registered_aggregate",
        lambda path: deepcopy(upper_aggregate),
    )

    def validate(primary, learner, *, expected_primary_root=None):
        if Path(expected_primary_root).resolve() != primary_root.resolve():
            raise RegisteredAggregationError(
                "primary_matrix_binding.raw_matrix_path does not resolve to the "
                "supplied primary matrix root"
            )
        return (
            normalize_primary_matrix_binding(primary_binding),
            normalize_memoryless_learner_binding(memoryless_binding),
            deepcopy(primary_aggregate),
            primary_root.resolve(),
        )

    monkeypatch.setattr(
        linker,
        "validate_bound_compute_aware_primary_matrix",
        validate,
    )


def test_schema6_schema7_link_is_deterministic_and_contains_no_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        primary,
        upper,
        primary_aggregate,
        upper_aggregate,
        primary_binding,
        memoryless_binding,
    ) = _link_fixture(tmp_path)
    _patch_aggregates(
        monkeypatch,
        primary_root=primary,
        primary_aggregate=primary_aggregate,
        upper_aggregate=upper_aggregate,
        primary_binding=primary_binding,
        memoryless_binding=memoryless_binding,
    )

    first = build_upper_reference_link(primary, upper)
    second = build_upper_reference_link(primary, upper)

    assert first == second
    assert first["schema_version"] == 2
    assert first["pairing_mode"] == (
        "schema6_selected_memoryless_to_schema7_compute_aware_upper_reference"
    )
    assert len(first["contract_equality"]["task_pairs"]) == 4
    assert first["contract_equality"]["raw_returns_included"] is False
    assert all("returns" not in repr(pair) for pair in first["contract_equality"]["task_pairs"])
    assert first["contract_equality"]["full_ppo_and_evaluation_validated"] is True
    assert first["contract_equality"]["full_provenance_validated"] is True


def test_schema6_schema7_link_rejects_bound_primary_root_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        primary,
        upper,
        primary_aggregate,
        upper_aggregate,
        primary_binding,
        memoryless_binding,
    ) = _link_fixture(tmp_path)
    _patch_aggregates(
        monkeypatch,
        primary_root=primary,
        primary_aggregate=primary_aggregate,
        upper_aggregate=upper_aggregate,
        primary_binding=primary_binding,
        memoryless_binding=memoryless_binding,
    )
    substituted = tmp_path / "substituted"
    atomic_write_json(
        substituted / "registration.json",
        {"schema_version": 6},
    )

    with pytest.raises(UpperReferenceLinkError, match="supplied primary"):
        build_upper_reference_link(substituted, upper)


def test_schema6_schema7_link_rejects_full_ppo_or_evaluation_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        primary,
        upper,
        primary_aggregate,
        upper_aggregate,
        primary_binding,
        memoryless_binding,
    ) = _link_fixture(tmp_path)
    _patch_aggregates(
        monkeypatch,
        primary_root=primary,
        primary_aggregate=primary_aggregate,
        upper_aggregate=upper_aggregate,
        primary_binding=primary_binding,
        memoryless_binding=memoryless_binding,
    )
    first_cell = next((upper / "cells").glob("*.json"))
    artifact = deepcopy(linker._load_json(first_cell, field="test artifact"))
    artifact["configuration"]["ppo"]["learning_rate"] = 0.0001
    first_cell.write_bytes(canonical_json_bytes(artifact) + b"\n")

    with pytest.raises(UpperReferenceLinkError, match="PPO or evaluation"):
        build_upper_reference_link(primary, upper)


def test_schema6_schema7_link_requires_exact_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        primary,
        upper,
        primary_aggregate,
        upper_aggregate,
        primary_binding,
        memoryless_binding,
    ) = _link_fixture(tmp_path)
    upper_aggregate["provenance"]["git"]["commit"] = "f" * 40
    _patch_aggregates(
        monkeypatch,
        primary_root=primary,
        primary_aggregate=primary_aggregate,
        upper_aggregate=upper_aggregate,
        primary_binding=primary_binding,
        memoryless_binding=memoryless_binding,
    )

    with pytest.raises(UpperReferenceLinkError, match="provenance equality"):
        build_upper_reference_link(primary, upper)


@pytest.mark.parametrize("schema_pair", [(6, 2), (4, 7)])
def test_compute_aware_link_rejects_mismatched_schema_pairs(
    tmp_path: Path,
    schema_pair: tuple[int, int],
) -> None:
    primary = tmp_path / "primary"
    upper = tmp_path / "upper"
    atomic_write_json(
        primary / "registration.json",
        {"schema_version": schema_pair[0]},
    )
    atomic_write_json(
        upper / "registration.json",
        {"schema_version": schema_pair[1]},
    )

    with pytest.raises(UpperReferenceLinkError, match=r"schema pair \(6, 7\)"):
        build_upper_reference_link(primary, upper)
