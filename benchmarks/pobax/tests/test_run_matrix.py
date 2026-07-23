"""Fail-closed tests for the frozen POBAX matrix launcher."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from benchmarks.pobax import run_pilot
from benchmarks.pobax.registered_artifacts import (
    ExistingArtifactMismatchError,
    canonical_json_sha256,
)
from benchmarks.pobax.run_matrix import (
    _command_for_cell,
    _load_matching_artifact,
    _load_registration,
)
from benchmarks.pobax.run_pilot import (
    environment_horizon_and_gamma,
    make_environment,
    validate_upper_reference_task_contract,
)


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


def _registration_v2(
    *,
    comparison_profile: str = "arcmind_shared_comparison",
) -> dict[str, object]:
    registration = _registration()
    registration.update(
        schema_version=2,
        comparison_profile=comparison_profile,
        learner={
            "num_envs": 64,
            "rollout_steps": 64,
            "update_epochs": 4,
            "num_minibatches": 4,
            "learning_rate": 0.00025,
            "gae_lambda": 0.95,
            "entropy_coefficient": 0.01,
            "anneal_learning_rate": True,
        },
    )
    return registration


@pytest.mark.parametrize(
    "filename",
    [
        "smoke_controls_v1.json",
        "tmaze_pilot_v1.json",
        "tmaze_shm_repair_v2.json",
    ],
)
def test_repository_registrations_are_valid(filename: str):
    manifest_path = Path(__file__).resolve().parents[1] / "manifests" / filename

    assert _load_registration(manifest_path)["status"] == "frozen"


def test_tmaze_shm_repair_registration_preserves_pilot_contract():
    manifest_path = Path(__file__).resolve().parents[1] / "manifests" / "tmaze_shm_repair_v2.json"

    assert _load_registration(manifest_path) == {
        "schema_version": 2,
        "status": "frozen",
        "evidence_tier": "pilot",
        "matrix_kind": "primary_comparison",
        "comparison_profile": "arcmind_shared_comparison",
        "models": ["shm", "arcmind"],
        "environments": [{"id": "tmaze_10", "total_steps": 250_000}],
        "seeds": [1103, 2207, 3301],
        "learner": {
            "num_envs": 8,
            "rollout_steps": 125,
            "update_epochs": 4,
            "num_minibatches": 4,
            "learning_rate": 0.00025,
            "gae_lambda": 0.95,
            "entropy_coefficient": 0.01,
            "anneal_learning_rate": False,
        },
        "evaluation_episodes_per_env": 16,
        "require_gpu": True,
        "quick": False,
    }


def test_registration_accepts_a_complete_paired_matrix(tmp_path):
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(_registration()), encoding="utf-8")

    assert _load_registration(path) == _registration()


@pytest.mark.parametrize(
    "comparison_profile",
    ["pobax_author_semantics", "arcmind_shared_comparison"],
)
def test_registration_v2_requires_a_complete_explicit_learner(
    tmp_path,
    comparison_profile,
):
    registration = _registration_v2(comparison_profile=comparison_profile)
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")

    assert _load_registration(path) == registration


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("comparison_profile"), "wrong fields"),
        (
            lambda value: value.update(comparison_profile="unspecified"),
            "comparison_profile",
        ),
        (
            lambda value: value["learner"].pop("gae_lambda"),
            "learner has wrong fields",
        ),
        (
            lambda value: value["learner"].update(anneal_learning_rate=1),
            "anneal_learning_rate",
        ),
    ],
)
def test_registration_v2_fails_closed(tmp_path, mutation, message):
    registration = _registration_v2()
    mutation(registration)
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_registration(path)


def test_shared_profile_requires_exact_steps_but_source_profile_floors(tmp_path):
    registration = _registration_v2()
    registration["environments"][0]["total_steps"] = 131_073
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly divisible"):
        _load_registration(path)

    registration["comparison_profile"] = "pobax_author_semantics"
    path.write_text(json.dumps(registration), encoding="utf-8")
    assert _load_registration(path) == registration


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
        num_minibatches=4,
        learning_rate=0.00025,
        gae_lambda=0.95,
        entropy_coefficient=0.01,
        anneal_learning_rate=True,
        registration_schema_version=2,
        comparison_profile="arcmind_shared_comparison",
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
    assert command[command.index("--num-minibatches") + 1] == "4"
    assert command[command.index("--gae-lambda") + 1] == "0.95"
    assert "--anneal-learning-rate" in command
    assert command[command.index("--comparison-profile") + 1] == "arcmind_shared_comparison"


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


@pytest.mark.parametrize(
    ("alias", "source", "perfect_memory"),
    [
        ("tmaze_10-perfect-memory", "tmaze_10", True),
        ("rocksample_11_11-fully-observable", "rocksample_11_11", True),
        (
            "Navix-DMLab-Maze-01-fully-observable",
            "Navix-DMLab-Maze-F-01-v0",
            False,
        ),
    ],
)
def test_upper_reference_aliases_request_source_variant(
    monkeypatch,
    alias,
    source,
    perfect_memory,
):
    calls = []

    def fake_get_env(environment_name, key, **kwargs):
        calls.append((environment_name, key, kwargs))
        return "environment", "params"

    monkeypatch.setattr(run_pilot, "get_env", fake_get_env)

    assert make_environment(alias, "key", num_envs=8) == ("environment", "params")
    assert calls == [
        (
            source,
            "key",
            {"num_envs": 8, "perfect_memory": perfect_memory},
        )
    ]


def test_battleship_upper_reference_adds_the_audited_observation_adapter(
    monkeypatch,
):
    source_environment = object()
    calls = []

    def fake_get_env(environment_name, key, **kwargs):
        calls.append((environment_name, key, kwargs))
        return source_environment, "params"

    monkeypatch.setattr(run_pilot, "get_env", fake_get_env)

    environment, params = make_environment(
        "battleship_10-perfect-recall",
        "key",
        num_envs=8,
    )

    assert environment._env is source_environment
    assert params == "params"
    assert calls == [
        (
            "battleship_10",
            "key",
            {"num_envs": 8, "perfect_memory": True},
        )
    ]


class _DiscreteActionSpace:
    def __init__(self, n):
        self.n = n


class _ContinuousActionSpace:
    def __init__(self, shape):
        self.shape = shape


def test_upper_reference_task_contract_accepts_only_equivalent_dynamics():
    assert (
        validate_upper_reference_task_contract(
            upper_action_space=_DiscreteActionSpace(4),
            primary_action_space=_DiscreteActionSpace(4),
            upper_horizon=1_000,
            primary_horizon=1_000,
            upper_gamma=0.99,
            primary_gamma=0.99,
        )
        == 4
    )

    mismatches = [
        (
            {"primary_action_space": _ContinuousActionSpace((4,))},
            "action-space class",
        ),
        ({"primary_action_space": _DiscreteActionSpace(3)}, "action dimensions"),
        ({"primary_horizon": 999}, "horizons"),
        ({"primary_gamma": 1.0}, "discounts"),
    ]
    defaults = {
        "upper_action_space": _DiscreteActionSpace(4),
        "primary_action_space": _DiscreteActionSpace(4),
        "upper_horizon": 1_000,
        "primary_horizon": 1_000,
        "upper_gamma": 0.99,
        "primary_gamma": 0.99,
    }
    for override, message in mismatches:
        with pytest.raises(RuntimeError, match=message):
            validate_upper_reference_task_contract(**(defaults | override))
