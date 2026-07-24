"""Execution-boundary tests for schema-v5 compute-aware tuning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.pobax.registration_protocol import (
    COMPUTE_AWARE_TUNED_FAMILIES,
)
from benchmarks.pobax.run_matrix import (
    _cell_namespace,
    _load_registration,
    _schema_v5_candidates,
)
from benchmarks.pobax.run_pilot import ARTIFACT_SCHEMA_BY_REGISTRATION


def _learner(learning_rate: float) -> dict[str, int | float | bool]:
    return {
        "num_envs": 8,
        "rollout_steps": 125,
        "update_epochs": 4,
        "num_minibatches": 4,
        "learning_rate": learning_rate,
        "gae_lambda": 0.95,
        "entropy_coefficient": 0.01,
        "anneal_learning_rate": False,
    }


def _registration() -> dict[str, object]:
    return {
        "schema_version": 5,
        "status": "frozen",
        "evidence_tier": "development_tuning",
        "matrix_kind": "hyperparameter_selection",
        "tuned_families": [
            {"family_id": family, "implementation_model": family}
            for family in COMPUTE_AWARE_TUNED_FAMILIES
        ],
        "learner_grid": [
            {"learner_id": "lr_low", "learner": _learner(0.0001)},
            {"learner_id": "lr_mid", "learner": _learner(0.00025)},
            {"learner_id": "lr_high", "learner": _learner(0.0005)},
        ],
        "environments": [
            {"id": "tmaze_10", "total_steps": 1_000_000},
            {"id": "rocksample_11_11", "total_steps": 1_000_000},
        ],
        "seeds": [4409, 5519, 6637],
        "comparison_profile": "arcmind_shared_comparison",
        "evaluation_episodes_per_env": 4,
        "require_gpu": True,
        "quick": False,
    }


def _write_registration(path: Path) -> dict[str, object]:
    registration = _registration()
    path.write_text(json.dumps(registration), encoding="utf-8")
    return registration


def test_schema_v5_registration_loads_only_the_exact_validated_panel(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registration.json"
    registration = _write_registration(path)

    assert _load_registration(path) == registration

    registration["environments"][1]["total_steps"] = 5_000_000
    path.write_text(json.dumps(registration), encoding="utf-8")
    with pytest.raises(ValueError, match="exact ordered two-task panel"):
        _load_registration(path)


def test_schema_v5_derives_describe_only_candidate_namespace() -> None:
    registration = _registration()
    args = _cell_namespace(
        registration,
        environment=registration["environments"][1],
        model="gru.lr_high",
        seed=5519,
        output=Path("cell.json"),
        manifest_sha256="a" * 64,
        cell_id="b" * 64,
        describe_only=True,
    )

    assert args.model == "gru"
    assert args.candidate_id == "gru.lr_high"
    assert args.model_family == "gru"
    assert args.learner_id == "lr_high"
    assert args.learning_rate == 0.0005
    assert args.registration_schema_version == 5
    assert args.describe_only is True


def test_schema_v5_describes_full_registered_candidate_namespace() -> None:
    registration = _registration()
    candidates = _schema_v5_candidates(registration)
    expected_models = [
        f"{family}.{learner_id}"
        for family in COMPUTE_AWARE_TUNED_FAMILIES
        for learner_id in ("lr_low", "lr_mid", "lr_high")
    ]
    assert [candidate["candidate_id"] for candidate in candidates] == expected_models
    assert all(
        candidate["model_family"] == candidate["implementation_model"] for candidate in candidates
    )


def test_run_pilot_reserves_artifact_schema_9_for_registration_schema_5() -> None:
    assert [ARTIFACT_SCHEMA_BY_REGISTRATION[schema] for schema in range(1, 6)] == [
        4,
        5,
        6,
        8,
        9,
    ]
