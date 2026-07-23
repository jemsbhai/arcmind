"""Fail-closed tests for the frozen POBAX matrix launcher."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from benchmarks.pobax.registered_artifacts import (
    ExistingArtifactMismatchError,
    canonical_json_sha256,
)
from benchmarks.pobax.run_matrix import (
    _command_for_cell,
    _load_matching_artifact,
    _load_registration,
)
from benchmarks.pobax.run_pilot import environment_horizon_and_gamma


def _registration() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "frozen",
        "evidence_tier": "pilot",
        "matrix_kind": "primary_comparison",
        "models": ["arcmind", "gru"],
        "environments": [{"id": "tmaze_10", "total_steps": 131_072}],
        "seeds": [1103, 2207],
        "learner": {
            "num_envs": 64,
            "rollout_steps": 64,
            "update_epochs": 4,
            "learning_rate": 0.00025,
        },
        "evaluation_episodes_per_env": 4,
        "require_gpu": True,
        "quick": False,
    }


def test_registration_accepts_a_complete_paired_matrix(tmp_path):
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(_registration()), encoding="utf-8")

    assert _load_registration(path) == _registration()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(status="draft"), "status"),
        (
            lambda value: value.update(models=["arcmind", "arcmind"]),
            "duplicates",
        ),
        (lambda value: value.update(seeds=[1103, 1103]), "duplicate seeds"),
        (
            lambda value: value.update(
                environments=[
                    {"id": "tmaze_10", "total_steps": 131_072},
                    {"id": "tmaze_10", "total_steps": 131_072},
                ]
            ),
            "environment ids",
        ),
        (
            lambda value: value.update(
                evidence_tier="pilot",
                quick=True,
            ),
            "smoke",
        ),
        (
            lambda value: value.update(unexpected=True),
            "wrong fields",
        ),
        (
            lambda value: value["environments"][0].update(unexpected=True),
            "exactly id and total_steps",
        ),
        (
            lambda value: value.update(
                matrix_kind="primary_comparison",
                models=["gru"],
            ),
            "must contain arcmind",
        ),
    ],
)
def test_registration_rejects_unsafe_matrix_definitions(
    tmp_path,
    mutation,
    message,
):
    registration = _registration()
    mutation(registration)
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        _load_registration(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["environments"][0].update(total_steps=131_072),
            "total_steps=8192",
        ),
        (
            lambda value: value["learner"].update(num_envs=64),
            "learner.num_envs=32",
        ),
    ],
)
def test_quick_registration_cannot_misstate_overridden_values(
    tmp_path,
    mutation,
    message,
):
    registration = _registration()
    registration.update(evidence_tier="smoke", quick=True)
    registration["environments"][0]["total_steps"] = 8_192
    registration["learner"].update(
        num_envs=32,
        rollout_steps=32,
        update_epochs=2,
    )
    mutation(registration)
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_registration(path)


def test_registered_final_requires_full_seed_cardinality(tmp_path):
    registration = _registration()
    registration.update(evidence_tier="registered_final")
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 30 paired seeds"):
        _load_registration(path)


def test_upper_reference_registration_has_an_explicit_role(tmp_path):
    registration = _registration()
    registration.update(
        matrix_kind="upper_reference",
        models=["memoryless_mlp"],
        environments=[{"id": "Walker-F-v0", "total_steps": 131_072}],
    )
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")

    assert _load_registration(path)["matrix_kind"] == "upper_reference"


def test_existing_artifact_must_match_identity_and_provenance(tmp_path):
    path = tmp_path / "cell.json"
    provenance = {
        "git": {"commit": "a" * 40, "dirty": False, "diff_sha256": None},
        "dependency_lock_sha256": "b" * 64,
        "pobax_commit": "c" * 40,
        "navix_commit": "d" * 40,
        "runtime_contract": {
            "python": {"implementation": "CPython", "version": "3.12.3"},
            "packages": {"jax": "0.6.2", "jaxlib": "0.6.2"},
            "jax_backend": "gpu",
            "jax_enable_x64": False,
            "devices": [{"platform": "gpu", "device_kind": "Test GPU"}],
        },
    }
    configuration = {"environment": "tmaze_10", "model": "arcmind", "seed": 1103}
    configuration_sha256 = canonical_json_sha256(configuration)
    artifact = {
        "schema_version": 4,
        "status": "registered_final_complete",
        "environment": "tmaze_10",
        "model": "arcmind",
        "seed": 1103,
        "configuration_sha256": configuration_sha256,
        "configuration": configuration,
        "matrix_manifest_sha256": "f" * 64,
        "cell_id": "1" * 64,
        "provenance": provenance,
        "runtime": provenance,
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")

    assert (
        _load_matching_artifact(
            path,
            expected_status="registered_final_complete",
            environment="tmaze_10",
            model="arcmind",
            seed=1103,
            configuration_sha256=configuration_sha256,
            manifest_sha256="f" * 64,
            cell_id="1" * 64,
            provenance=provenance,
        )
        == artifact
    )

    altered = {**provenance, "pobax_commit": "0" * 40}
    with pytest.raises(ExistingArtifactMismatchError, match="provenance"):
        _load_matching_artifact(
            path,
            expected_status="registered_final_complete",
            environment="tmaze_10",
            model="arcmind",
            seed=1103,
            configuration_sha256=configuration_sha256,
            manifest_sha256="f" * 64,
            cell_id="1" * 64,
            provenance=altered,
        )

    artifact["configuration"]["seed"] = 999
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ExistingArtifactMismatchError, match="configuration content"):
        _load_matching_artifact(
            path,
            expected_status="registered_final_complete",
            environment="tmaze_10",
            model="arcmind",
            seed=1103,
            configuration_sha256=configuration_sha256,
            manifest_sha256="f" * 64,
            cell_id="1" * 64,
            provenance=provenance,
        )


def test_cell_command_carries_frozen_identity_and_safety_flags(tmp_path):
    args = Namespace(
        environment="tmaze_10",
        model="shm",
        seed=1103,
        total_steps=1_000_000,
        num_envs=8,
        rollout_steps=125,
        update_epochs=4,
        learning_rate=0.00025,
        evaluation_episodes_per_env=4,
        evidence_tier="registered_final",
        matrix_manifest_sha256="a" * 64,
        cell_id="b" * 64,
        output=tmp_path / "cell.json",
        require_gpu=True,
        quick=False,
    )

    command = _command_for_cell(args)

    assert "--require-clean-git" in command
    assert "--require-gpu" in command
    assert "--quick" not in command
    assert command[command.index("--matrix-manifest-sha256") + 1] == "a" * 64
    assert command[command.index("--cell-id") + 1] == "b" * 64


def test_environment_horizon_and_gamma_uses_source_contract():
    environment = Namespace(gamma=1.0)
    environment_params = Namespace(max_steps_in_episode=1_000)

    assert environment_horizon_and_gamma(
        environment,
        environment_params,
        "battleship_10",
    ) == (1_000, 1.0)


def test_environment_horizon_and_gamma_rejects_source_drift():
    environment = Namespace(gamma=0.99)
    environment_params = Namespace(max_steps_in_episode=20)

    with pytest.raises(RuntimeError, match="episode horizon drift"):
        environment_horizon_and_gamma(
            environment,
            environment_params,
            "tmaze_10",
        )
