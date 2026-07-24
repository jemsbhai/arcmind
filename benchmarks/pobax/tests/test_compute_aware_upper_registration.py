"""Focused schema-7 upper-reference registration tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from benchmarks.pobax.implementation_provenance import (
    IMPLEMENTATION_SOURCE_ALGORITHM,
)
from benchmarks.pobax.registered_artifacts import canonical_json_sha256
from benchmarks.pobax.registration_protocol import (
    COMPUTE_AWARE_FINAL_MODELS,
    COMPUTE_AWARE_FINAL_PANEL,
    COMPUTE_AWARE_FINAL_SEEDS,
    COMPUTE_AWARE_TASK_MODEL_INCIDENCE,
    COMPUTE_AWARE_UPPER_REFERENCE_PANEL,
    REGISTRATION_FIELDS_V2,
    REGISTRATION_FIELDS_V7,
    normalize_memoryless_learner_binding,
    normalize_primary_matrix_binding,
    registration_fields,
    validate_compute_aware_primary_binding_against_aggregate,
    validate_compute_aware_upper_reference_contract,
)


def _learner() -> dict[str, int | float | bool]:
    return {
        "num_envs": 8,
        "rollout_steps": 125,
        "update_epochs": 4,
        "num_minibatches": 4,
        "learning_rate": 0.00025,
        "gae_lambda": 0.95,
        "entropy_coefficient": 0.01,
        "anneal_learning_rate": False,
    }


def _implementation_source() -> dict[str, Any]:
    unsigned = {
        "schema_version": 1,
        "algorithm": IMPLEMENTATION_SOURCE_ALGORITHM,
        "files": [
            {
                "path": "benchmarks/pobax/run_pilot.py",
                "sha256": "a" * 64,
            }
        ],
    }
    return {**unsigned, "sha256": canonical_json_sha256(unsigned)}


def _primary_binding(source_hash: str) -> dict[str, str]:
    return {
        "raw_matrix_path": "artifacts/primary/raw",
        "aggregate_path": "artifacts/primary/aggregate.json",
        "primary_aggregate_file_sha256": "1" * 64,
        "primary_registration_file_sha256": "2" * 64,
        "primary_manifest_file_sha256": "3" * 64,
        "primary_manifest_internal_sha256": "4" * 64,
        "primary_completion_index_file_sha256": "5" * 64,
        "primary_checksum_manifest_file_sha256": "6" * 64,
        "primary_implementation_source_sha256": source_hash,
    }


def _memoryless_binding(source_hash: str) -> dict[str, Any]:
    return {
        "candidate_id": "memoryless_mlp.lr_mid",
        "model_family": "memoryless_mlp",
        "learner_id": "lr_mid",
        "learner": _learner(),
        "implementation_model": "memoryless_mlp",
        "learner_binding_mode": "selected",
        "learner_source_model_family": "memoryless_mlp",
        "tuning_aggregate_sha256": "7" * 64,
        "tuning_completion_index_sha256": "8" * 64,
        "tuning_checksum_manifest_sha256": "9" * 64,
        "tuning_implementation_source_sha256": source_hash,
    }


def _primary_aggregate(
    primary: dict[str, str],
    memoryless: dict[str, Any],
    implementation_source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": "registered_matrix_aggregate",
        "matrix_kind": "primary_comparison",
        "matrix_manifest_sha256": primary["primary_manifest_internal_sha256"],
        "raw_integrity": {
            "registration_file_sha256": primary["primary_registration_file_sha256"],
            "completion_index_sha256": primary["primary_completion_index_file_sha256"],
            "checksum_manifest_sha256": primary["primary_checksum_manifest_file_sha256"],
            "completion_index_validated": True,
            "checksum_inventory_validated": True,
            "reference_implementation_validated": True,
            "parameter_contract_validated": True,
        },
        "provenance": {"implementation_source": implementation_source},
        "models": list(COMPUTE_AWARE_FINAL_MODELS),
        "environments": [environment for environment, _ in COMPUTE_AWARE_FINAL_PANEL],
        "seeds": list(COMPUTE_AWARE_FINAL_SEEDS),
        "statistical_unit": "seed",
        "common_models": list(COMPUTE_AWARE_FINAL_MODELS[:8]),
        "task_model_incidence": [
            {"environment": environment, "models": list(models)}
            for environment, models in COMPUTE_AWARE_TASK_MODEL_INCIDENCE
        ],
        "groups": [
            {"environment": environment, "model": model}
            for environment, models in COMPUTE_AWARE_TASK_MODEL_INCIDENCE
            for model in models
        ],
        "learner_bindings": [
            {
                "model": "memoryless_mlp",
                "mode": "selected",
                "source_model_family": "memoryless_mlp",
            }
        ],
        "tuning_selection_binding": {
            "aggregate_sha256": memoryless["tuning_aggregate_sha256"],
            "source_completion_index_sha256": memoryless["tuning_completion_index_sha256"],
            "source_checksum_manifest_sha256": memoryless["tuning_checksum_manifest_sha256"],
            "source_implementation_sha256": memoryless["tuning_implementation_source_sha256"],
            "validated": True,
            "final_seeds_disjoint_from_tuning": True,
            "selections": [
                {
                    "model_family": "memoryless_mlp",
                    "implementation_model": "memoryless_mlp",
                    "candidate_id": memoryless["candidate_id"],
                    "learner_id": memoryless["learner_id"],
                    "learner": memoryless["learner"],
                    "implementation_source_sha256": implementation_source["sha256"],
                }
            ],
        },
    }


def test_schema7_field_set_and_exact_upper_contract() -> None:
    source = _implementation_source()
    primary = _primary_binding(source["sha256"])
    memoryless = _memoryless_binding(source["sha256"])

    assert registration_fields(7) == REGISTRATION_FIELDS_V7
    assert REGISTRATION_FIELDS_V7 == (REGISTRATION_FIELDS_V2 - {"learner"}) | {
        "primary_matrix_binding",
        "memoryless_learner_binding",
    }
    validate_compute_aware_upper_reference_contract(
        schema_version=7,
        comparison_profile="arcmind_shared_comparison",
        matrix_kind="upper_reference",
        models=["memoryless_mlp"],
        primary_matrix_binding=primary,
        memoryless_learner_binding=memoryless,
        environments=dict(COMPUTE_AWARE_UPPER_REFERENCE_PANEL),
        seeds=COMPUTE_AWARE_FINAL_SEEDS,
        evaluation_episodes_per_env=16,
        require_gpu=True,
        quick=False,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("raw_matrix_path", "../primary", "normalized relative path"),
        ("aggregate_path", r"artifacts\primary.json", "POSIX path"),
        ("primary_manifest_internal_sha256", "A" * 64, "lowercase SHA256"),
    ],
)
def test_schema7_primary_paths_and_hashes_fail_closed(
    field: str,
    value: str,
    message: str,
) -> None:
    source = _implementation_source()
    primary = _primary_binding(source["sha256"])
    primary[field] = value

    with pytest.raises(ValueError, match=message):
        normalize_primary_matrix_binding(primary)


def test_schema7_memoryless_binding_must_be_direct_selected_winner() -> None:
    source = _implementation_source()
    binding = _memoryless_binding(source["sha256"])
    binding["learner_binding_mode"] = "inherited"

    with pytest.raises(ValueError, match="direct selected"):
        normalize_memoryless_learner_binding(binding)


def test_schema7_primary_verifier_proves_selected_memoryless_binding() -> None:
    source = _implementation_source()
    primary = _primary_binding(source["sha256"])
    memoryless = _memoryless_binding(source["sha256"])
    aggregate = _primary_aggregate(primary, memoryless, source)

    validate_compute_aware_primary_binding_against_aggregate(
        primary,
        memoryless,
        aggregate,
    )

    drifted = deepcopy(aggregate)
    drifted["tuning_selection_binding"]["selections"][0]["candidate_id"] = "memoryless_mlp.lr_low"
    with pytest.raises(ValueError, match="drifts from the primary aggregate"):
        validate_compute_aware_primary_binding_against_aggregate(
            primary,
            memoryless,
            drifted,
        )


@pytest.mark.parametrize(
    ("evaluation_episodes", "require_gpu", "message"),
    [
        (1, True, "16 evaluation"),
        (16, False, "GPU"),
    ],
)
def test_schema7_freezes_primary_evaluation_and_device_contract(
    evaluation_episodes: int,
    require_gpu: bool,
    message: str,
) -> None:
    source = _implementation_source()
    with pytest.raises(ValueError, match=message):
        validate_compute_aware_upper_reference_contract(
            schema_version=7,
            comparison_profile="arcmind_shared_comparison",
            matrix_kind="upper_reference",
            models=["memoryless_mlp"],
            primary_matrix_binding=_primary_binding(source["sha256"]),
            memoryless_learner_binding=_memoryless_binding(source["sha256"]),
            environments=dict(COMPUTE_AWARE_UPPER_REFERENCE_PANEL),
            seeds=COMPUTE_AWARE_FINAL_SEEDS,
            evaluation_episodes_per_env=evaluation_episodes,
            require_gpu=require_gpu,
            quick=False,
        )
