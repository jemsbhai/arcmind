"""Focused schema-7 direct-runner boundary tests."""

from __future__ import annotations

from argparse import Namespace

import pytest

import benchmarks.pobax.run_pilot as pilot
from benchmarks.pobax.registration_protocol import (
    COMPUTE_AWARE_UPPER_REFERENCE_PANEL,
)
from benchmarks.pobax.tests.test_compute_aware_upper_registration import (
    _implementation_source,
    _memoryless_binding,
    _primary_binding,
)


def _args() -> Namespace:
    source = _implementation_source()
    primary = _primary_binding(source["sha256"])
    learner = _memoryless_binding(source["sha256"])
    environment, total_steps = COMPUTE_AWARE_UPPER_REFERENCE_PANEL[0]
    return Namespace(
        environment=environment,
        model="memoryless_mlp",
        candidate_id=learner["candidate_id"],
        model_family=learner["model_family"],
        learner_id=learner["learner_id"],
        learner_binding_mode=learner["learner_binding_mode"],
        learner_source_model_family=learner["learner_source_model_family"],
        tuning_aggregate_sha256=learner["tuning_aggregate_sha256"],
        tuning_completion_index_sha256=learner["tuning_completion_index_sha256"],
        tuning_checksum_manifest_sha256=learner["tuning_checksum_manifest_sha256"],
        tuning_implementation_source_sha256=learner["tuning_implementation_source_sha256"],
        **{field: value for field, value in primary.items() if field.endswith("_sha256")},
        seed=10_000,
        total_steps=total_steps,
        num_envs=8,
        rollout_steps=125,
        update_epochs=4,
        num_minibatches=4,
        learning_rate=0.00025,
        gae_lambda=0.95,
        entropy_coefficient=0.01,
        anneal_learning_rate=False,
        registration_schema_version=7,
        comparison_profile="arcmind_shared_comparison",
        evaluation_episodes_per_env=16,
        output=None,
        quick=False,
        require_gpu=True,
        require_clean_git=True,
        evidence_tier="registered_final",
        matrix_manifest_sha256="b" * 64,
        cell_id="c" * 64,
        describe_only=True,
    )


def _patch_runtime_prelude(monkeypatch) -> None:
    monkeypatch.setattr(pilot.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        pilot,
        "source_commit",
        lambda package: (
            pilot.PINNED_POBAX_COMMIT if package == "pobax" else pilot.PINNED_NAVIX_COMMIT
        ),
    )
    monkeypatch.setattr(
        pilot,
        "gather_git_provenance",
        lambda root: {"commit": "1" * 40, "dirty": False},
    )
    monkeypatch.setattr(pilot, "dependency_lock_sha256", lambda path: "2" * 64)
    monkeypatch.setattr(pilot, "runtime_contract", lambda: {"runtime": "test"})


def test_schema7_uses_artifact_schema11() -> None:
    assert pilot.ARTIFACT_SCHEMA_BY_REGISTRATION[7] == 11


@pytest.mark.parametrize(
    ("evaluation_episodes", "require_gpu"),
    [(1, True), (16, False)],
)
def test_schema7_direct_runner_rejects_evaluation_or_device_drift(
    evaluation_episodes: int,
    require_gpu: bool,
) -> None:
    args = _args()
    args.evaluation_episodes_per_env = evaluation_episodes
    args.require_gpu = require_gpu

    with pytest.raises(ValueError, match="evaluation_episodes_per_env=16"):
        pilot.run(args)


def test_schema7_direct_runner_rejects_seed_outside_primary_manifest(
    monkeypatch,
) -> None:
    args = _args()
    args.seed = 10_010
    _patch_runtime_prelude(monkeypatch)

    with pytest.raises(ValueError, match="exact compute-aware upper-reference cell"):
        pilot.run(args)


def test_schema7_direct_runner_rejects_current_implementation_drift(
    monkeypatch,
) -> None:
    args = _args()
    _patch_runtime_prelude(monkeypatch)
    drifted = _implementation_source()
    drifted["sha256"] = "f" * 64
    monkeypatch.setattr(pilot, "gather_implementation_source", lambda root: drifted)

    with pytest.raises(ValueError, match="current implementation source drifts"):
        pilot.run(args)
