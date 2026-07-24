"""Fail-closed tests for the frozen POBAX matrix launcher."""

from __future__ import annotations

import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.pobax import run_matrix, run_pilot
from benchmarks.pobax.implementation_provenance import (
    IMPLEMENTATION_SOURCE_ALGORITHM,
    gather_implementation_source,
)
from benchmarks.pobax.link_upper_reference import _validate_completion_and_checksums
from benchmarks.pobax.model_registry import (
    MAMBA1_REFERENCE_IMPLEMENTATION,
    MEMORY_TRACE_OFFICIAL_REFERENCE_IMPLEMENTATION,
    policy_contract_metadata_for_model,
)
from benchmarks.pobax.registered_artifacts import (
    ExistingArtifactMismatchError,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)
from benchmarks.pobax.registration_protocol import (
    validate_final_provenance_against_tuning,
)
from benchmarks.pobax.run_matrix import (
    _cell_namespace,
    _command_for_cell,
    _load_matching_artifact,
    _load_registration,
    execute_matrix,
)
from benchmarks.pobax.run_pilot import (
    environment_horizon_and_gamma,
    make_environment,
    validate_upper_reference_task_contract,
)


def _implementation_source() -> dict[str, object]:
    unsigned = {
        "schema_version": 1,
        "algorithm": IMPLEMENTATION_SOURCE_ALGORITHM,
        "files": [{"path": "arcmind/__init__.py", "sha256": "a" * 64}],
    }
    return {**unsigned, "sha256": canonical_json_sha256(unsigned)}


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


def test_implementation_source_inventory_covers_shared_and_model_runtime():
    repository_root = Path(__file__).resolve().parents[3]
    source = gather_implementation_source(repository_root)
    paths = {item["path"] for item in source["files"]}

    assert {
        "arcmind/models/arcmind_model.py",
        "benchmarks/pobax/run_pilot.py",
        "benchmarks/pobax/shared_ppo.py",
        "benchmarks/pobax/policy_core.py",
        "benchmarks/pobax/mamba_core.py",
        "benchmarks/pobax/memory_trace_core.py",
        "benchmarks/pobax/upper_reference_envs.py",
    }.issubset(paths)
    assert not any(path.startswith("benchmarks/pobax/tests/") for path in paths)


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


def _registration_v3() -> dict[str, object]:
    registration = _registration_v2()
    learner = registration.pop("learner")
    registration.pop("models")
    registration.update(
        schema_version=3,
        evidence_tier="development_tuning",
        matrix_kind="hyperparameter_selection",
        seeds=[1103, 2207, 3301, 4409, 5519],
        environments=[{"id": "tmaze_10", "total_steps": 1_000_000}],
        candidate_families=[
            {
                "family_id": family,
                "implementation_model": implementation_model,
                "candidates": [
                    {
                        "candidate_id": f"{family}.lr_{label}",
                        "learner": {
                            **learner,
                            "num_envs": 8,
                            "rollout_steps": 125,
                            "learning_rate": learning_rate,
                        },
                    }
                    for label, learning_rate in (("low", 0.00025), ("high", 0.001))
                ],
            }
            for family, implementation_model in (
                ("ordered_memory", "arcmind"),
                ("recurrent", "gru"),
            )
        ],
    )
    return registration


def _tuning_aggregate() -> dict[str, object]:
    implementation_source = _implementation_source()
    learners = {
        "ordered_memory": {
            "num_envs": 8,
            "rollout_steps": 125,
            "update_epochs": 4,
            "num_minibatches": 4,
            "learning_rate": 0.001,
            "gae_lambda": 0.95,
            "entropy_coefficient": 0.01,
            "anneal_learning_rate": False,
        },
        "recurrent": {
            "num_envs": 8,
            "rollout_steps": 125,
            "update_epochs": 4,
            "num_minibatches": 4,
            "learning_rate": 0.001,
            "gae_lambda": 0.95,
            "entropy_coefficient": 0.01,
            "anneal_learning_rate": False,
        },
    }
    return {
        "schema_version": 1,
        "status": "development_tuning_selection_aggregate_not_for_paper",
        "evidence_tier": "development_tuning",
        "matrix_kind": "hyperparameter_selection",
        "not_for_paper": True,
        "registration_sha256": "1" * 64,
        "matrix_manifest_sha256": "2" * 64,
        "completion_index_sha256": "3" * 64,
        "checksum_manifest_sha256": "4" * 64,
        "provenance": {
            "git": {"commit": "a" * 40, "dirty": False, "diff_sha256": None},
            "dependency_lock_sha256": "b" * 64,
            "pobax_commit": "c" * 40,
            "navix_commit": "d" * 40,
            "runtime_contract": {"runtime": "test"},
            "implementation_source": implementation_source,
        },
        "environments": ["tmaze_10"],
        "seeds": [1103, 2207, 3301, 4409, 5519],
        "integrity_indexes": {
            "completion_index_present_and_validated": True,
            "checksums_present_and_validated": True,
        },
        "frozen_semantic_contract": {
            "environment_source_in_every_configuration": True,
            "parameter_match_in_every_configuration": False,
            "artifact_parameter_match_validated": False,
            "parameter_contract_in_every_configuration": True,
            "artifact_parameter_contract_validated": True,
        },
        "selection_eligibility": {
            "eligible_for_hyperparameter_selection": True,
            "eligible_for_architecture_selection": False,
            "eligible_for_checkpoint_selection": False,
            "eligible_for_registered_final_evidence": False,
            "eligible_for_paper_performance_claims": False,
            "selection_scope": "candidate_within_model_family_and_environment",
        },
        "candidate_selection": [
            {
                "environment": "tmaze_10",
                "model_family": family,
                "implementation_model": implementation,
                "winner_candidate_id": f"{family}.lr_high",
                "ranking": [
                    {"rank": 1, "candidate_id": f"{family}.lr_high"},
                    {"rank": 2, "candidate_id": f"{family}.lr_low"},
                ],
            }
            for family, implementation in (
                ("ordered_memory", "arcmind"),
                ("recurrent", "gru"),
            )
        ],
        "groups": [
            {
                "environment": "tmaze_10",
                "candidate_id": f"{family}.lr_high",
                "model_family": family,
                "implementation_model": implementation,
                "learner": learners[family],
                "implementation_source_sha256": implementation_source["sha256"],
            }
            for family, implementation in (
                ("ordered_memory", "arcmind"),
                ("recurrent", "gru"),
            )
        ],
    }


def _registration_v4(tmp_path: Path, monkeypatch) -> tuple[dict[str, object], Path]:
    aggregate = _tuning_aggregate()
    raw_matrix = tmp_path / "raw-tuning"
    raw_matrix.mkdir()
    (raw_matrix / "completion_index.json").write_bytes(b"tuning completion\n")
    (raw_matrix / "checksums.sha256").write_bytes(b"tuning checksums\n")
    aggregate["completion_index_sha256"] = sha256_file(raw_matrix / "completion_index.json")
    aggregate["checksum_manifest_sha256"] = sha256_file(raw_matrix / "checksums.sha256")
    aggregate_path = tmp_path / "tuning-selection.json"
    atomic_write_json(aggregate_path, aggregate)
    monkeypatch.setattr(run_matrix, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        run_matrix,
        "build_development_aggregate",
        lambda path: deepcopy(aggregate),
    )
    selections = [
        {
            "environment": "tmaze_10",
            "model_family": group["model_family"],
            "implementation_model": group["implementation_model"],
            "candidate_id": group["candidate_id"],
            "learner": deepcopy(group["learner"]),
            "implementation_source_sha256": group["implementation_source_sha256"],
        }
        for group in aggregate["groups"]
    ]
    registration = {
        "schema_version": 4,
        "status": "frozen",
        "evidence_tier": "registered_final",
        "matrix_kind": "primary_comparison",
        "models": ["arcmind", "gru"],
        "environments": [{"id": "tmaze_10", "total_steps": 1_000_000}],
        "seeds": list(range(10_000, 10_030)),
        "comparison_profile": "arcmind_shared_comparison",
        "tuning_selection": {
            "raw_matrix_path": "raw-tuning",
            "aggregate_path": "tuning-selection.json",
            "aggregate_sha256": sha256_file(aggregate_path),
            "source_registration_sha256": aggregate["registration_sha256"],
            "source_manifest_sha256": aggregate["matrix_manifest_sha256"],
            "source_completion_index_sha256": sha256_file(raw_matrix / "completion_index.json"),
            "source_checksum_manifest_sha256": sha256_file(raw_matrix / "checksums.sha256"),
            "source_implementation_sha256": aggregate["provenance"]["implementation_source"][
                "sha256"
            ],
            "selections": selections,
        },
        "evaluation_episodes_per_env": 128,
        "require_gpu": True,
        "quick": False,
    }
    path = tmp_path / "registration-v4.json"
    return registration, path


@pytest.mark.parametrize(
    "filename",
    [
        "smoke_controls_v1.json",
        "tmaze_pilot_v1.json",
        "tmaze_attention_horizon_repair_v3.json",
        "tmaze_coverage_ablation_v2.json",
        "tmaze_shm_repair_v2.json",
    ],
)
def test_repository_registrations_are_valid(filename: str):
    manifest_path = Path(__file__).resolve().parents[1] / "manifests" / filename

    assert _load_registration(manifest_path)["status"] == "frozen"


def test_tmaze_attention_horizon_repair_preserves_pilot_contract():
    manifest_path = (
        Path(__file__).resolve().parents[1] / "manifests" / "tmaze_attention_horizon_repair_v3.json"
    )

    assert _load_registration(manifest_path) == {
        "schema_version": 2,
        "status": "frozen",
        "evidence_tier": "pilot",
        "matrix_kind": "primary_comparison",
        "comparison_profile": "arcmind_shared_comparison",
        "models": ["arcmind"],
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


def test_tmaze_coverage_ablation_registration_preserves_pilot_contract():
    manifest_path = (
        Path(__file__).resolve().parents[1] / "manifests" / "tmaze_coverage_ablation_v2.json"
    )

    assert _load_registration(manifest_path) == {
        "schema_version": 2,
        "status": "frozen",
        "evidence_tier": "pilot",
        "matrix_kind": "primary_comparison",
        "comparison_profile": "arcmind_shared_comparison",
        "models": [
            "frame_stack_mlp",
            "lru",
            "s4d",
            "arcmind_unordered",
            "arcmind_no_memory",
            "arcmind_no_ssm",
            "arcmind_no_gate",
            "arcmind",
        ],
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


def test_registration_rejects_unknown_policy_implementation(tmp_path):
    registration = _registration()
    registration["models"] = ["arcmind", "unregistered_core"]
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")

    with pytest.raises(ValueError, match="registered policy implementations"):
        _load_registration(path)


def test_tuning_registration_rejects_unknown_policy_implementation(tmp_path):
    registration = _registration_v3()
    registration["candidate_families"][0]["implementation_model"] = "unregistered_core"
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")

    with pytest.raises(ValueError, match="registered policy implementations"):
        _load_registration(path)


def test_memory_trace_alias_is_rejected_for_tuning_and_final_selection(
    tmp_path,
    monkeypatch,
):
    tuning = _registration_v3()
    tuning["candidate_families"][0]["implementation_model"] = "memory_trace_mlp"
    tuning_path = tmp_path / "tuning.json"
    tuning_path.write_text(json.dumps(tuning), encoding="utf-8")

    with pytest.raises(ValueError, match="development-only compatibility alias"):
        _load_registration(tuning_path)

    final_root = tmp_path / "final"
    final_root.mkdir()
    final, final_path = _registration_v4(final_root, monkeypatch)
    final["tuning_selection"]["selections"][0][
        "implementation_model"
    ] = "memory_trace_mlp"
    final_path.write_text(json.dumps(final), encoding="utf-8")

    with pytest.raises(ValueError, match="development-only compatibility alias"):
        _load_registration(final_path)


def test_official_memory_trace_registration_rejects_continuous_tasks(tmp_path):
    registration = _registration()
    registration["models"] = ["arcmind", "memory_trace_official"]
    registration["environments"] = [
        {"id": "Walker-V-v0", "total_steps": 131_072}
    ]
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")

    with pytest.raises(ValueError, match="official.*continuous-action"):
        _load_registration(path)


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
    registration = _registration_v2(comparison_profile="pobax_author_semantics")
    registration.update(
        evidence_tier="registered_final",
        matrix_kind="upper_reference",
        models=["memoryless_mlp"],
        environments=[{"id": "Walker-F-v0", "total_steps": 50_000_000}],
    )
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 30 paired seeds"):
        _load_registration(path)


def test_schema_v2_author_semantics_upper_reference_needs_no_tuning_binding(
    tmp_path,
):
    registration = _registration_v2(comparison_profile="pobax_author_semantics")
    registration.update(
        evidence_tier="registered_final",
        matrix_kind="upper_reference",
        models=["memoryless_mlp"],
        environments=[{"id": "Walker-F-v0", "total_steps": 50_000_000}],
        seeds=list(range(10_000, 10_030)),
    )
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")

    assert _load_registration(path) == registration


def test_registered_primary_comparison_requires_schema_v4_binding(tmp_path):
    registration = _registration_v2()
    registration.update(
        evidence_tier="registered_final",
        seeds=list(range(10_000, 10_030)),
        environments=[{"id": "tmaze_10", "total_steps": 1_000_000}],
    )
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")

    with pytest.raises(ValueError, match="schema version 4"):
        _load_registration(path)


def test_schema_v4_registered_final_binds_exact_tuning_winners(
    tmp_path,
    monkeypatch,
):
    registration, path = _registration_v4(tmp_path, monkeypatch)
    path.write_text(json.dumps(registration), encoding="utf-8")

    assert _load_registration(path) == registration


def _overlap_tuning_and_final_seeds(registration):
    registration["seeds"] = [1103, *range(10_000, 10_029)]


def _select_nonwinner(registration):
    registration["tuning_selection"]["selections"][0]["candidate_id"] = "ordered_memory.lr_low"


def _drift_selected_learner(registration):
    registration["tuning_selection"]["selections"][0]["learner"]["learning_rate"] = 0.0005


def _refresh_mutated_tuning_log(registration, tmp_path):
    raw_root = tmp_path / registration["tuning_selection"]["raw_matrix_path"]
    (raw_root / "mutated.log").write_bytes(b"mutated tuning log\n")
    (raw_root / "completion_index.json").write_bytes(b"refreshed completion\n")
    (raw_root / "checksums.sha256").write_bytes(b"refreshed checksums\n")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            _overlap_tuning_and_final_seeds,
            "disjoint from tuning seeds",
        ),
        (
            _select_nonwinner,
            "does not match the tuning aggregate winner",
        ),
        (
            lambda value: value["tuning_selection"].update(aggregate_sha256="0" * 64),
            "does not match aggregate bytes",
        ),
        (
            _drift_selected_learner,
            "learner drifts from the tuning aggregate winner",
        ),
    ],
)
def test_schema_v4_registered_final_binding_fails_closed(
    tmp_path,
    monkeypatch,
    mutation,
    message,
):
    registration, path = _registration_v4(tmp_path, monkeypatch)
    mutation(registration)
    path.write_text(json.dumps(registration), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_registration(path)


def test_schema_v4_binding_rejects_refreshed_indexes_after_log_mutation(
    tmp_path,
    monkeypatch,
):
    registration, path = _registration_v4(tmp_path, monkeypatch)
    _refresh_mutated_tuning_log(registration, tmp_path)
    path.write_text(json.dumps(registration), encoding="utf-8")

    with pytest.raises(ValueError, match="source_completion_index_sha256"):
        _load_registration(path)


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("completion_index.json", "source_completion_index_sha256"),
        ("checksums.sha256", "source_checksum_manifest_sha256"),
    ],
)
def test_schema_v4_binding_freezes_tuning_integrity_file_bytes(
    tmp_path,
    monkeypatch,
    filename,
    message,
):
    registration, path = _registration_v4(tmp_path, monkeypatch)
    source = tmp_path / registration["tuning_selection"]["raw_matrix_path"] / filename
    source.write_bytes(source.read_bytes() + b"drift")
    path.write_text(json.dumps(registration), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_registration(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dependency_lock_sha256", "0" * 64),
        ("pobax_commit", "0" * 40),
        ("navix_commit", "0" * 40),
        ("runtime_contract", {"runtime": "drifted"}),
    ],
)
def test_final_provenance_must_match_tuning_provenance(field, value):
    tuning = _tuning_aggregate()["provenance"]
    final = deepcopy(tuning)
    final["git"]["commit"] = "f" * 40
    final[field] = value
    binding = {"source_implementation_sha256": tuning["implementation_source"]["sha256"]}

    with pytest.raises(ValueError, match="final provenance drifts"):
        validate_final_provenance_against_tuning(
            binding=binding,
            tuning_provenance=tuning,
            final_provenance=final,
        )


def test_final_provenance_allows_git_only_drift_but_rejects_implementation_drift():
    tuning = _tuning_aggregate()["provenance"]
    final = deepcopy(tuning)
    final["git"]["commit"] = "f" * 40
    binding = {"source_implementation_sha256": tuning["implementation_source"]["sha256"]}

    validate_final_provenance_against_tuning(
        binding=binding,
        tuning_provenance=tuning,
        final_provenance=final,
    )

    unsigned = deepcopy(final["implementation_source"])
    unsigned["files"][0]["sha256"] = "0" * 64
    unsigned.pop("sha256")
    final["implementation_source"] = {
        **unsigned,
        "sha256": canonical_json_sha256(unsigned),
    }
    with pytest.raises(ValueError, match="implementation source drifts"):
        validate_final_provenance_against_tuning(
            binding=binding,
            tuning_provenance=tuning,
            final_provenance=final,
        )


def test_development_tuning_registration_requires_published_selection_contract(
    tmp_path,
):
    registration = _registration_v3()
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")

    assert _load_registration(path) == registration


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.update(comparison_profile="pobax_author_semantics"),
            "arcmind_shared_comparison",
        ),
        (
            lambda value: value.update(seeds=[1103, 2207, 3301]),
            "published tuning-seed count",
        ),
        (
            lambda value: value["environments"][0].update(total_steps=250_000),
            "published task budget",
        ),
        (
            lambda value: value["candidate_families"][0].update(
                candidates=value["candidate_families"][0]["candidates"][:1]
            ),
            "at least two candidates",
        ),
    ],
)
def test_development_tuning_registration_fails_closed(
    tmp_path,
    mutation,
    message,
):
    registration = _registration_v3()
    mutation(registration)
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_registration(path)


def _duplicate_candidate_learner(registration):
    candidates = registration["candidate_families"][0]["candidates"]
    candidates[1]["learner"] = dict(candidates[0]["learner"])


def _unequal_candidate_cardinality(registration):
    family = registration["candidate_families"][0]
    candidate = dict(family["candidates"][0])
    candidate["candidate_id"] = f"{family['family_id']}.lr_extra"
    candidate["learner"] = {
        **candidate["learner"],
        "learning_rate": 0.002,
    }
    family["candidates"].append(candidate)


def _drift_structural_learner(registration):
    learner = registration["candidate_families"][1]["candidates"][1]["learner"]
    learner.update(num_envs=4, rollout_steps=250)


def _drift_cross_family_tuning_grid(registration):
    learner = registration["candidate_families"][1]["candidates"][1]["learner"]
    learner["learning_rate"] = 0.002


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["candidate_families"][1].update(family_id="ordered_memory"),
            "unique portable identifier",
        ),
        (
            lambda value: value["candidate_families"][0]["candidates"][1].update(
                candidate_id="ordered_memory.lr_low"
            ),
            "globally unique",
        ),
        (
            lambda value: value["candidate_families"][1].update(implementation_model="arcmind"),
            "unique portable model identifier",
        ),
        (
            _duplicate_candidate_learner,
            "duplicate normalized learner",
        ),
        (
            _unequal_candidate_cardinality,
            "equal candidate cardinality",
        ),
        (
            _drift_structural_learner,
            "identical num_envs",
        ),
        (
            _drift_cross_family_tuning_grid,
            "exact same normalized learner configuration grid",
        ),
    ],
)
def test_development_tuning_candidate_families_fail_closed(
    tmp_path,
    mutation,
    message,
):
    registration = _registration_v3()
    mutation(registration)
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
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
            registration_schema_version=1,
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
            registration_schema_version=1,
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
            registration_schema_version=1,
        )


def test_existing_mamba_artifact_must_match_audited_source_contract(tmp_path):
    path = tmp_path / "mamba-cell.json"
    provenance = {
        "git": {"commit": "a" * 40, "dirty": False, "diff_sha256": None},
        "dependency_lock_sha256": "b" * 64,
        "pobax_commit": "c" * 40,
        "navix_commit": "d" * 40,
        "runtime_contract": {"runtime": "test"},
    }
    configuration = {
        "environment": "tmaze_10",
        "model": "mamba1",
        "seed": 1103,
        "reference_implementation": deepcopy(MAMBA1_REFERENCE_IMPLEMENTATION),
    }
    configuration_sha256 = canonical_json_sha256(configuration)
    artifact = {
        "schema_version": 4,
        "status": "registered_final_complete",
        "environment": "tmaze_10",
        "model": "mamba1",
        "seed": 1103,
        "configuration_sha256": configuration_sha256,
        "configuration": configuration,
        "reference_implementation": deepcopy(MAMBA1_REFERENCE_IMPLEMENTATION),
        "matrix_manifest_sha256": "f" * 64,
        "cell_id": "1" * 64,
        "provenance": provenance,
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    arguments = {
        "expected_status": "registered_final_complete",
        "environment": "tmaze_10",
        "model": "mamba1",
        "seed": 1103,
        "configuration_sha256": configuration_sha256,
        "manifest_sha256": "f" * 64,
        "cell_id": "1" * 64,
        "provenance": provenance,
        "registration_schema_version": 1,
    }

    assert _load_matching_artifact(path, **arguments) == artifact

    artifact["reference_implementation"]["audited_commit"] = "0" * 40
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ExistingArtifactMismatchError, match="reference implementation"):
        _load_matching_artifact(path, **arguments)

    artifact["reference_implementation"] = deepcopy(MAMBA1_REFERENCE_IMPLEMENTATION)
    artifact["configuration"]["reference_implementation"]["version"] = "drifted"
    drifted_configuration_sha256 = canonical_json_sha256(artifact["configuration"])
    artifact["configuration_sha256"] = drifted_configuration_sha256
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ExistingArtifactMismatchError, match="reference implementation"):
        _load_matching_artifact(
            path,
            **{
                **arguments,
                "configuration_sha256": drifted_configuration_sha256,
            },
        )


def test_existing_official_memory_trace_artifact_freezes_policy_contract(tmp_path):
    path = tmp_path / "memory-trace-cell.json"
    provenance = {
        "git": {"commit": "a" * 40, "dirty": False, "diff_sha256": None},
        "dependency_lock_sha256": "b" * 64,
        "pobax_commit": "c" * 40,
        "navix_commit": "d" * 40,
        "runtime_contract": {"runtime": "test"},
    }
    policy_core = {
        "input_dim": 7,
        "observation_dim": 2,
        "action_dim": 3,
        "hidden_size": 64,
        "decays": [0.0, 0.985],
    }
    policy_contract = policy_contract_metadata_for_model("memory_trace_official")
    configuration = {
        "environment": "tmaze_10",
        "model": "memory_trace_official",
        "seed": 1103,
        "reference_implementation": deepcopy(
            MEMORY_TRACE_OFFICIAL_REFERENCE_IMPLEMENTATION
        ),
        "policy_core": deepcopy(policy_core),
        **deepcopy(policy_contract),
    }
    configuration_sha256 = canonical_json_sha256(configuration)
    artifact = {
        "schema_version": 4,
        "status": "registered_final_complete",
        "environment": "tmaze_10",
        "model": "memory_trace_official",
        "seed": 1103,
        "configuration_sha256": configuration_sha256,
        "configuration": configuration,
        "reference_implementation": deepcopy(
            MEMORY_TRACE_OFFICIAL_REFERENCE_IMPLEMENTATION
        ),
        "policy_core": deepcopy(policy_core),
        **deepcopy(policy_contract),
        "matrix_manifest_sha256": "f" * 64,
        "cell_id": "1" * 64,
        "provenance": provenance,
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    arguments = {
        "expected_status": "registered_final_complete",
        "environment": "tmaze_10",
        "model": "memory_trace_official",
        "seed": 1103,
        "configuration_sha256": configuration_sha256,
        "manifest_sha256": "f" * 64,
        "cell_id": "1" * 64,
        "provenance": provenance,
        "registration_schema_version": 1,
    }

    assert _load_matching_artifact(path, **arguments) == artifact

    artifact["policy_core"]["decays"] = [0.0, 0.9]
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ExistingArtifactMismatchError, match="policy contract"):
        _load_matching_artifact(path, **arguments)

    artifact["policy_core"] = deepcopy(policy_core)
    artifact["comparison_role"] = "parameter_matched_primary"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ExistingArtifactMismatchError, match="policy contract"):
        _load_matching_artifact(path, **arguments)

    artifact["comparison_role"] = policy_contract["comparison_role"]
    artifact["policy_core"]["input_dim"] = 8
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ExistingArtifactMismatchError, match="policy contract"):
        _load_matching_artifact(path, **arguments)


def test_existing_tuning_artifact_must_match_candidate_identity(tmp_path):
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
    configuration = {
        "environment": "tmaze_10",
        "model": "ordered_memory.lr_low",
        "candidate_id": "ordered_memory.lr_low",
        "model_family": "ordered_memory",
        "implementation_model": "arcmind",
        "seed": 1103,
    }
    configuration_sha256 = canonical_json_sha256(configuration)
    artifact = {
        "schema_version": 6,
        "status": "development_tuning_not_for_paper",
        "environment": "tmaze_10",
        "model": "ordered_memory.lr_low",
        "candidate_id": "ordered_memory.lr_low",
        "model_family": "ordered_memory",
        "implementation_model": "arcmind",
        "seed": 1103,
        "configuration_sha256": configuration_sha256,
        "configuration": configuration,
        "matrix_manifest_sha256": "f" * 64,
        "cell_id": "1" * 64,
        "provenance": provenance,
        "registration_schema_version": 3,
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    arguments = {
        "expected_status": "development_tuning_not_for_paper",
        "environment": "tmaze_10",
        "model": "ordered_memory.lr_low",
        "seed": 1103,
        "configuration_sha256": configuration_sha256,
        "manifest_sha256": "f" * 64,
        "cell_id": "1" * 64,
        "provenance": provenance,
        "registration_schema_version": 3,
    }

    assert _load_matching_artifact(path, **arguments) == artifact

    artifact["schema_version"] = 5
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ExistingArtifactMismatchError, match="wrong schema"):
        _load_matching_artifact(path, **arguments)

    artifact["schema_version"] = 6
    artifact["model_family"] = "recurrent"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ExistingArtifactMismatchError, match="candidate identity"):
        _load_matching_artifact(path, **arguments)


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


def test_tuning_cell_command_carries_family_and_candidate_identity(tmp_path):
    args = Namespace(
        environment="tmaze_10",
        model="arcmind",
        candidate_id="ordered_memory.lr_low",
        model_family="ordered_memory",
        seed=1103,
        total_steps=1_000_000,
        num_envs=8,
        rollout_steps=125,
        update_epochs=4,
        num_minibatches=4,
        learning_rate=0.00025,
        gae_lambda=0.95,
        entropy_coefficient=0.01,
        anneal_learning_rate=False,
        registration_schema_version=3,
        comparison_profile="arcmind_shared_comparison",
        evaluation_episodes_per_env=4,
        evidence_tier="development_tuning",
        matrix_manifest_sha256="a" * 64,
        cell_id="b" * 64,
        output=tmp_path / "cell.json",
        require_gpu=True,
        quick=False,
    )

    command = _command_for_cell(args)

    assert command[command.index("--model") + 1] == "arcmind"
    assert command[command.index("--candidate-id") + 1] == "ordered_memory.lr_low"
    assert command[command.index("--model-family") + 1] == "ordered_memory"
    assert command[command.index("--registration-schema-version") + 1] == "3"


def test_tuning_cell_namespace_separates_family_candidate_and_implementation():
    registration = _registration_v3()

    args = _cell_namespace(
        registration,
        environment=registration["environments"][0],
        model="ordered_memory.lr_high",
        seed=1103,
        output=None,
        manifest_sha256=None,
        cell_id=None,
        describe_only=True,
    )

    assert args.model == "arcmind"
    assert args.model_family == "ordered_memory"
    assert args.candidate_id == "ordered_memory.lr_high"
    assert args.learning_rate == 0.001


def test_mamba_cell_namespace_and_command_use_registered_implementation():
    registration = _registration()
    registration["models"] = ["arcmind", "mamba1"]

    args = _cell_namespace(
        registration,
        environment=registration["environments"][0],
        model="mamba1",
        seed=1103,
        output=None,
        manifest_sha256=None,
        cell_id=None,
        describe_only=True,
    )
    command = _command_for_cell(args)

    assert args.model == "mamba1"
    assert command[command.index("--model") + 1] == "mamba1"


def test_memory_trace_shared_cell_namespace_uses_explicit_registered_identifier():
    registration = _registration()
    registration["models"] = ["arcmind", "memory_trace_shared"]

    args = _cell_namespace(
        registration,
        environment=registration["environments"][0],
        model="memory_trace_shared",
        seed=1103,
        output=None,
        manifest_sha256=None,
        cell_id=None,
        describe_only=True,
    )
    command = _command_for_cell(args)

    assert args.model == "memory_trace_shared"
    assert command[command.index("--model") + 1] == "memory_trace_shared"


def test_schema_v4_cell_namespace_carries_bound_final_selection(
    tmp_path,
    monkeypatch,
):
    registration, _ = _registration_v4(tmp_path, monkeypatch)

    args = _cell_namespace(
        registration,
        environment=registration["environments"][0],
        model="arcmind",
        seed=10_000,
        output=tmp_path / "cell.json",
        manifest_sha256="a" * 64,
        cell_id="b" * 64,
        describe_only=False,
    )
    command = _command_for_cell(args)

    assert args.model == "arcmind"
    assert args.model_family == "ordered_memory"
    assert args.candidate_id == "ordered_memory.lr_high"
    assert args.learning_rate == 0.001
    assert args.tuning_aggregate_sha256 == registration["tuning_selection"]["aggregate_sha256"]
    assert (
        command[command.index("--tuning-aggregate-sha256") + 1]
        == (registration["tuning_selection"]["aggregate_sha256"])
    )


def test_failed_attempt_log_is_never_certified_by_later_success(
    tmp_path,
    monkeypatch,
):
    registration = _registration()
    registration.update(models=["arcmind"], seeds=[1103])
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps(registration), encoding="utf-8")
    output_root = tmp_path / "matrix"
    provenance = {
        "git": {"commit": "a" * 40, "dirty": False, "diff_sha256": None},
        "dependency_lock_sha256": "b" * 64,
        "pobax_commit": "c" * 40,
        "navix_commit": "d" * 40,
        "runtime_contract": {"runtime": "test"},
    }

    def configuration(environment, model, seed):
        return {
            "schema_version": 1,
            "evidence_tier": "pilot",
            "environment": environment,
            "model": model,
            "seed": seed,
            "dependency_lock_sha256": provenance["dependency_lock_sha256"],
            "pobax_commit": provenance["pobax_commit"],
            "navix_commit": provenance["navix_commit"],
            "runtime_contract": provenance["runtime_contract"],
        }

    def fake_describe(args):
        frozen = configuration(args.environment, args.model, args.seed)
        return {
            "configuration_sha256": canonical_json_sha256(frozen),
            "configuration": frozen,
            "runtime": {"git": provenance["git"]},
        }

    attempts = []

    def argument(command, name):
        return command[command.index(name) + 1]

    def fake_subprocess(command, **kwargs):
        del kwargs
        attempts.append(list(command))
        environment = argument(command, "--environment")
        model = argument(command, "--model")
        seed = int(argument(command, "--seed"))
        frozen = configuration(environment, model, seed)
        artifact_path = Path(argument(command, "--output"))
        artifact = {
            "schema_version": 4,
            "status": "development_pilot_not_for_paper",
            "environment": environment,
            "model": model,
            "seed": seed,
            "configuration_sha256": canonical_json_sha256(frozen),
            "configuration": frozen,
            "matrix_manifest_sha256": argument(command, "--matrix-manifest-sha256"),
            "cell_id": argument(command, "--cell-id"),
            "provenance": provenance,
        }
        if len(attempts) == 2:
            artifact["status"] = "invalid_success_artifact"
        atomic_write_json(artifact_path, artifact)
        if len(attempts) == 1:
            return SimpleNamespace(returncode=1, stdout=b"failed attempt\n")
        if len(attempts) == 2:
            return SimpleNamespace(returncode=0, stdout=b"invalid success attempt\n")
        return SimpleNamespace(returncode=0, stdout=b"successful attempt\n")

    monkeypatch.setattr(run_matrix, "run", fake_describe)
    monkeypatch.setattr(run_matrix.subprocess, "run", fake_subprocess)

    with pytest.raises(RuntimeError, match="cell failed"):
        execute_matrix(registration_path, output_root)

    manifest = json.loads((output_root / "frozen_manifest.json").read_text(encoding="utf-8"))
    canonical_log_before_success = (
        output_root / manifest["cells"][0]["artifact_path"]
    ).with_suffix(".log")
    assert not canonical_log_before_success.exists()
    attempt_root = output_root.with_name(f"{output_root.name}.attempts")
    failed_logs = list(attempt_root.rglob("*.failed.log"))
    failed_artifacts = list(attempt_root.rglob("*.failed.json"))
    assert len(failed_logs) == 1
    assert failed_logs[0].read_bytes() == b"failed attempt\n"
    assert len(failed_artifacts) == 1

    with pytest.raises(RuntimeError, match="returned success with an invalid artifact"):
        execute_matrix(registration_path, output_root)

    assert not canonical_log_before_success.exists()
    failed_logs = sorted(attempt_root.rglob("*.failed.log"))
    failed_artifacts = sorted(attempt_root.rglob("*.failed.json"))
    assert len(failed_logs) == 2
    assert {path.read_bytes() for path in failed_logs} == {
        b"failed attempt\n",
        b"invalid success attempt\n",
    }
    assert len(failed_artifacts) == 2

    completion = execute_matrix(registration_path, output_root)
    canonical_log = output_root / completion["cells"][0]["log_path"]
    assert canonical_log.read_bytes() == b"successful attempt\n"
    assert sha256_file(canonical_log) == completion["cells"][0]["log_sha256"]
    assert len(attempts) == 3

    resumed = execute_matrix(registration_path, output_root)
    assert resumed == completion
    assert len(attempts) == 3
    assert canonical_log.read_bytes() == b"successful attempt\n"
    assert not list(output_root.rglob("*.attempt-*"))
    link_cells = {
        (cell["environment"], cell["model"], cell["seed"]): {
            **cell,
            "resolved_artifact_path": output_root / cell["artifact_path"],
        }
        for cell in manifest["cells"]
    }
    _validate_completion_and_checksums(output_root, manifest, link_cells)


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
