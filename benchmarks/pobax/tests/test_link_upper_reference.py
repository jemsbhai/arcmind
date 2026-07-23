"""Tests for strict primary and upper-reference matrix linking."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import benchmarks.pobax.link_upper_reference as linker
from benchmarks.pobax.link_upper_reference import (
    REGISTERED_TRAIN_STEPS,
    UpperReferenceLinkError,
    _validate_registered_budgets,
    build_upper_reference_link,
    link_upper_reference,
)
from benchmarks.pobax.registered_artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_sha256,
    registered_cell_id,
    sha256_file,
    write_checksum_manifest,
)
from benchmarks.pobax.upper_reference_registry import (
    UPPER_TO_PRIMARY_ENVIRONMENT,
    expected_environment_reference,
    expected_environment_source,
)

SEEDS = [1103, 2207]
UPPER_ENVIRONMENTS = list(UPPER_TO_PRIMARY_ENVIRONMENT)
PRIMARY_ENVIRONMENTS = [
    UPPER_TO_PRIMARY_ENVIRONMENT[environment] for environment in UPPER_ENVIRONMENTS
]
PILOT_BUDGETS = {
    environment: (index + 1) * 1_000 for index, environment in enumerate(PRIMARY_ENVIRONMENTS)
}
PILOT_BUDGETS.update(
    {upper: PILOT_BUDGETS[primary] for upper, primary in UPPER_TO_PRIMARY_ENVIRONMENT.items()}
)
HORIZONS = {environment: (index + 1) * 10 for index, environment in enumerate(PRIMARY_ENVIRONMENTS)}
HORIZONS.update(
    {upper: HORIZONS[primary] for upper, primary in UPPER_TO_PRIMARY_ENVIRONMENT.items()}
)
BASE_PROVENANCE = {
    "git": {"commit": "1" * 40, "dirty": False, "diff_sha256": None},
    "dependency_lock_sha256": "2" * 64,
    "pobax_commit": "3" * 40,
    "navix_commit": "4" * 40,
    "runtime_contract": {
        "python": {"implementation": "CPython", "version": "3.12.3"},
        "packages": {"jax": "0.6.2", "jaxlib": "0.6.2", "numpy": "2.2.0"},
        "jax_backend": "gpu",
        "jax_enable_x64": False,
        "devices": [{"platform": "gpu", "device_kind": "Test GPU"}],
    },
}
OPTIMIZER_METRICS = {
    "loss": 0.5,
    "actor_loss": 0.1,
    "value_loss": 0.8,
    "entropy": 0.2,
    "approximate_kl": 0.01,
}


def _configuration(
    environment: str,
    model: str,
    seed: int,
    *,
    tier: str,
    learner: dict[str, Any],
    evaluation_episodes: int,
    provenance: dict[str, Any],
    schema_version: int = 1,
    comparison_profile: str | None = None,
    total_steps: int | None = None,
) -> dict[str, Any]:
    selected_total_steps = PILOT_BUDGETS[environment] if total_steps is None else total_steps
    configuration = {
        "schema_version": schema_version,
        "evidence_tier": tier,
        "environment": environment,
        "environment_source": expected_environment_source(environment),
        "environment_reference": expected_environment_reference(environment),
        "model": model,
        "seed": seed,
        "parameter_count": 100,
        "effective_parameter_count": 100,
        "arcmind_target_parameter_count": 100,
        "parameter_ratio": 1.0,
        "ppo": {
            "total_steps": selected_total_steps,
            **learner,
            "num_minibatches": 1,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "value_coefficient": 0.5,
            "entropy_coefficient": 0.01,
            "max_gradient_norm": 0.5,
        },
        "evaluation_episodes_per_environment": evaluation_episodes,
        "evaluation_max_episode_steps": HORIZONS[environment],
        "dependency_lock_sha256": provenance["dependency_lock_sha256"],
        "pobax_commit": provenance["pobax_commit"],
        "navix_commit": provenance["navix_commit"],
        "runtime_contract": deepcopy(provenance["runtime_contract"]),
    }
    if schema_version in {2, 4}:
        requested_steps = selected_total_steps
        realized_steps = requested_steps // 4 * 4
        configuration["ppo"].update(
            anneal_learning_rate=learner["anneal_learning_rate"],
            step_budget_mode=(
                "floor" if comparison_profile == "pobax_author_semantics" else "exact"
            ),
        )
        configuration.update(
            comparison_profile=comparison_profile,
            requested_environment_steps=requested_steps,
            realized_environment_steps=realized_steps,
        )
    return configuration


def _write_matrix(
    root: Path,
    *,
    matrix_kind: str,
    tier: str = "pilot",
    seeds: list[int] | None = None,
    environments: list[str] | None = None,
    learning_rate: float = 0.001,
    evaluation_episodes: int = 2,
    provenance: dict[str, Any] | None = None,
    schema_version: int = 1,
    comparison_profile: str | None = None,
    models: list[str] | None = None,
    learners: dict[str, dict[str, Any]] | None = None,
    tuning_selection: dict[str, Any] | None = None,
    budgets: dict[str, int] | None = None,
) -> None:
    selected_seeds = seeds or SEEDS
    selected_environments = environments or (
        PRIMARY_ENVIRONMENTS if matrix_kind == "primary_comparison" else UPPER_ENVIRONMENTS
    )
    selected_models = models or (
        ["arcmind"] if matrix_kind == "primary_comparison" else ["memoryless_mlp"]
    )
    selected_provenance = deepcopy(provenance or BASE_PROVENANCE)
    default_learner = {
        "num_envs": 2,
        "rollout_steps": 2,
        "update_epochs": 1,
        "learning_rate": learning_rate,
    }
    if schema_version in {2, 4}:
        default_learner.update(
            num_minibatches=1,
            gae_lambda=0.95,
            entropy_coefficient=0.01,
            anneal_learning_rate=True,
        )
    selected_learners = learners or {model: deepcopy(default_learner) for model in selected_models}
    selected_budgets = budgets or PILOT_BUDGETS
    registration = {
        "schema_version": schema_version,
        "status": "frozen",
        "evidence_tier": tier,
        "matrix_kind": matrix_kind,
        "models": selected_models,
        "environments": [
            {"id": environment, "total_steps": selected_budgets[environment]}
            for environment in selected_environments
        ],
        "seeds": selected_seeds,
        "evaluation_episodes_per_env": evaluation_episodes,
        "require_gpu": selected_provenance["runtime_contract"]["jax_backend"] == "gpu",
        "quick": False,
    }
    if schema_version != 4:
        registration["learner"] = default_learner
    else:
        registration["tuning_selection"] = tuning_selection
    if schema_version in {2, 4}:
        registration["comparison_profile"] = comparison_profile
    atomic_write_json(root / "registration.json", registration)

    manifest_cells: list[dict[str, Any]] = []
    artifacts: dict[tuple[str, str, int], tuple[Path, dict[str, Any]]] = {}
    for environment in selected_environments:
        for model in selected_models:
            for seed in selected_seeds:
                learner = selected_learners[model]
                configuration = _configuration(
                    environment,
                    model,
                    seed,
                    tier=tier,
                    learner=learner,
                    evaluation_episodes=evaluation_episodes,
                    provenance=selected_provenance,
                    schema_version=schema_version,
                    comparison_profile=comparison_profile,
                    total_steps=selected_budgets[environment],
                )
                configuration_sha256 = canonical_json_sha256(configuration)
                identity = (environment, model, seed)
                relative_path = f"cells/{environment}-{model}-{seed}.json"
                manifest_cells.append(
                    {
                        "cell_id": registered_cell_id(
                            environment,
                            model,
                            seed,
                            configuration_sha256,
                        ),
                        "environment": environment,
                        "model": model,
                        "seed": seed,
                        "configuration_sha256": configuration_sha256,
                        "artifact_path": relative_path,
                    }
                )
                if schema_version == 4:
                    selection = next(
                        item
                        for item in tuning_selection["selections"]
                        if item["environment"] == environment
                        and item["implementation_model"] == model
                    )
                    manifest_cells[-1].update(
                        candidate_id=selection["candidate_id"],
                        model_family=selection["model_family"],
                        implementation_model=model,
                        tuning_aggregate_sha256=tuning_selection["aggregate_sha256"],
                    )
                artifacts[identity] = (root / relative_path, configuration)
    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "status": "frozen",
        "matrix_kind": matrix_kind,
        "models": selected_models,
        "environments": selected_environments,
        "seeds": selected_seeds,
        "provenance": selected_provenance,
        "cells": manifest_cells,
    }
    if schema_version == 4:
        manifest["tuning_selection"] = tuning_selection
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_write_json(root / "frozen_manifest.json", manifest)

    for identity, (path, configuration) in artifacts.items():
        environment, model, seed = identity
        configuration_sha256 = canonical_json_sha256(configuration)
        horizon = HORIZONS[environment]
        value = float(seed % 100)
        rows = [[value] * evaluation_episodes for _ in range(2)]
        artifact = {
            "schema_version": 7 if schema_version == 4 else 5 if schema_version == 2 else 4,
            "status": f"development_{tier}_not_for_paper",
            "matrix_manifest_sha256": manifest["manifest_sha256"],
            "cell_id": registered_cell_id(
                environment,
                model,
                seed,
                configuration_sha256,
            ),
            "configuration_sha256": configuration_sha256,
            "configuration": configuration,
            "environment": environment,
            "model": model,
            "seed": seed,
            "parameter_count": 100,
            "effective_parameter_count": 100,
            "arcmind_target_parameter_count": 100,
            "parameter_ratio": 1.0,
            "environment_source": configuration["environment_source"],
            "environment_reference": configuration["environment_reference"],
            "provenance": selected_provenance,
            "actual_environment_steps": selected_budgets[environment],
            "ppo": configuration["ppo"],
            "evaluation_episodes_per_environment": evaluation_episodes,
            "evaluation_max_episode_steps": horizon,
            "actual_evaluation_steps_per_environment": evaluation_episodes * horizon,
            "actual_evaluation_transitions": evaluation_episodes * horizon * 2,
            "evaluation": {
                "mean_return": value,
                "median_return": value,
                "episodes": evaluation_episodes * 2,
                "episodes_per_environment": evaluation_episodes,
                "num_environments": 2,
                "scan_steps_per_environment": evaluation_episodes * horizon,
                "returns_by_environment": rows,
            },
            "training": {**OPTIMIZER_METRICS, "mean_recent_return": value},
            "training_history": [
                {
                    **OPTIMIZER_METRICS,
                    "environment_steps": float(selected_budgets[environment]),
                    "mean_recent_return": value,
                }
            ],
        }
        if schema_version in {2, 4}:
            artifact.update(
                comparison_profile=comparison_profile,
                requested_environment_steps=configuration["requested_environment_steps"],
                realized_environment_steps=configuration["realized_environment_steps"],
            )
        if schema_version == 4:
            selection = next(
                item
                for item in tuning_selection["selections"]
                if item["environment"] == environment and item["implementation_model"] == model
            )
            artifact.update(
                candidate_id=selection["candidate_id"],
                model_family=selection["model_family"],
                implementation_model=model,
                tuning_aggregate_sha256=tuning_selection["aggregate_sha256"],
            )
        atomic_write_json(path, artifact)

    completed_cells = []
    for cell in manifest_cells:
        identity = (cell["environment"], cell["model"], cell["seed"])
        artifact_path = artifacts[identity][0]
        log_path = artifact_path.with_suffix(".log")
        atomic_write_bytes(log_path, b"test log\n")
        completed_cells.append(
            {
                **cell,
                "artifact_sha256": sha256_file(artifact_path),
                "log_path": log_path.relative_to(root).as_posix(),
                "log_sha256": sha256_file(log_path),
            }
        )
    atomic_write_json(
        root / "completion_index.json",
        {
            "schema_version": 1,
            "status": "complete",
            "manifest_sha256": manifest["manifest_sha256"],
            "planned_cells": len(completed_cells),
            "completed_cells": len(completed_cells),
            "cells": completed_cells,
        },
    )
    write_checksum_manifest(root)


def _paired_roots(
    tmp_path: Path,
    **upper_options: Any,
) -> tuple[Path, Path]:
    primary = tmp_path / "primary"
    upper = tmp_path / "upper"
    schema_version = upper_options.get("schema_version", 1)
    comparison_profile = upper_options.get("comparison_profile")
    _write_matrix(
        primary,
        matrix_kind="primary_comparison",
        schema_version=schema_version,
        comparison_profile=comparison_profile,
    )
    upper_provenance = deepcopy(BASE_PROVENANCE)
    upper_provenance["runtime_contract"]["jax_backend"] = "cpu"
    upper_provenance["runtime_contract"]["devices"] = [
        {"platform": "cpu", "device_kind": "Test CPU"}
    ]
    upper_options.setdefault("provenance", upper_provenance)
    _write_matrix(upper, matrix_kind="upper_reference", **upper_options)
    return primary, upper


def _schema_v4_registered_pair(tmp_path: Path) -> tuple[Path, Path]:
    primary = tmp_path / "primary-v4"
    upper = tmp_path / "upper-v2"
    seeds = list(range(10_000, 10_030))
    primary_environment = "tmaze_10"
    upper_environment = "tmaze_10-perfect-memory"
    arcmind_learner = {
        "num_envs": 2,
        "rollout_steps": 2,
        "update_epochs": 1,
        "num_minibatches": 1,
        "learning_rate": 0.001,
        "gae_lambda": 0.95,
        "entropy_coefficient": 0.01,
        "anneal_learning_rate": True,
    }
    gru_learner = {**arcmind_learner, "learning_rate": 0.002}
    tuning_selection = {
        "raw_matrix_path": "benchmark_results/pobax/tuning/raw",
        "aggregate_path": "benchmark_results/pobax/tuning/aggregate.json",
        "aggregate_sha256": "5" * 64,
        "source_registration_sha256": "6" * 64,
        "source_manifest_sha256": "7" * 64,
        "selections": [
            {
                "environment": primary_environment,
                "model_family": "ordered_memory",
                "implementation_model": "arcmind",
                "candidate_id": "ordered_memory.lr_selected",
                "learner": arcmind_learner,
            },
            {
                "environment": primary_environment,
                "model_family": "recurrent",
                "implementation_model": "gru",
                "candidate_id": "recurrent.lr_selected",
                "learner": gru_learner,
            },
        ],
    }
    budgets = {
        primary_environment: REGISTERED_TRAIN_STEPS[primary_environment],
        upper_environment: REGISTERED_TRAIN_STEPS[upper_environment],
    }
    _write_matrix(
        primary,
        matrix_kind="primary_comparison",
        tier="registered_final",
        schema_version=4,
        comparison_profile="arcmind_shared_comparison",
        seeds=seeds,
        environments=[primary_environment],
        models=["arcmind", "gru"],
        learners={"arcmind": arcmind_learner, "gru": gru_learner},
        tuning_selection=tuning_selection,
        evaluation_episodes=128,
        budgets=budgets,
    )
    upper_provenance = deepcopy(BASE_PROVENANCE)
    upper_provenance["runtime_contract"]["jax_backend"] = "cpu"
    upper_provenance["runtime_contract"]["devices"] = [
        {"platform": "cpu", "device_kind": "Test CPU"}
    ]
    _write_matrix(
        upper,
        matrix_kind="upper_reference",
        tier="registered_final",
        schema_version=2,
        comparison_profile="pobax_author_semantics",
        seeds=seeds,
        environments=[upper_environment],
        evaluation_episodes=128,
        learning_rate=0.003,
        provenance=upper_provenance,
        budgets=budgets,
    )
    return primary, upper


def _stub_registered_aggregation(monkeypatch) -> list[Path]:
    validated: list[Path] = []

    def build_registered(manifest_path: Path) -> dict[str, Any]:
        path = Path(manifest_path).resolve()
        validated.append(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return {"provenance": manifest["provenance"]}

    monkeypatch.setattr(linker, "build_registered_aggregate", build_registered)
    return validated


def _rewrite_json(path: Path, mutation: Any) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _refresh_checksums(root: Path) -> None:
    (root / "checksums.sha256").unlink()
    write_checksum_manifest(root)


def test_all_six_registered_aliases_link_deterministically(tmp_path: Path) -> None:
    primary, upper = _paired_roots(tmp_path)
    first = build_upper_reference_link(primary, upper)
    second = build_upper_reference_link(primary, upper)
    output = tmp_path / "derived" / "upper-reference-link.json"
    written = link_upper_reference(primary, upper, output)

    assert first == second == written
    assert first["status"] == "development_pilot_primary_upper_reference_link_not_for_paper"
    assert first["paper_status"] == "prohibited_development_evidence"
    assert first["not_for_paper"] is True
    assert first["seeds"] == SEEDS
    assert first["alias_mapping"] == [
        {
            "upper_reference_environment": upper_environment,
            "primary_environment": primary_environment,
        }
        for upper_environment, primary_environment in UPPER_TO_PRIMARY_ENVIRONMENT.items()
    ]
    assert first["primary"]["provenance"]["runtime_contract"]["jax_backend"] == "gpu"
    assert first["upper_reference"]["provenance"]["runtime_contract"]["jax_backend"] == "cpu"
    assert json.loads(output.read_text(encoding="utf-8")) == first


def test_schema_v2_matrices_link_only_with_the_same_comparison_profile(
    tmp_path: Path,
) -> None:
    primary, upper = _paired_roots(
        tmp_path,
        schema_version=2,
        comparison_profile="arcmind_shared_comparison",
    )

    result = build_upper_reference_link(primary, upper)

    assert result["status"] == ("development_pilot_primary_upper_reference_link_not_for_paper")

    mismatched_primary = tmp_path / "mismatched-primary"
    mismatched_upper = tmp_path / "mismatched-upper"
    _write_matrix(
        mismatched_primary,
        matrix_kind="primary_comparison",
        schema_version=2,
        comparison_profile="arcmind_shared_comparison",
    )
    _write_matrix(
        mismatched_upper,
        matrix_kind="upper_reference",
        schema_version=2,
        comparison_profile="pobax_author_semantics",
    )
    with pytest.raises(UpperReferenceLinkError, match="comparison profiles"):
        build_upper_reference_link(mismatched_primary, mismatched_upper)


def test_registered_schema_v4_primary_links_to_schema_v2_author_upper_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    primary, upper = _schema_v4_registered_pair(tmp_path)
    validated = _stub_registered_aggregation(monkeypatch)

    result = build_upper_reference_link(primary, upper)

    assert validated == [
        (primary / "frozen_manifest.json").resolve(),
        (upper / "frozen_manifest.json").resolve(),
    ]
    assert result["status"] == "registered_final_primary_upper_reference_link"
    assert result["paper_status"] == "eligible_only_after_all_other_release_gates_pass"
    assert result["seeds"] == list(range(10_000, 10_030))
    contract = result["learner_contract"]
    assert contract["pairing_mode"] == "selected_primary_to_pobax_author_upper_reference"
    assert contract["primary"]["comparison_profile"] == "arcmind_shared_comparison"
    assert contract["upper_reference"]["comparison_profile"] == "pobax_author_semantics"
    assert contract["primary"]["tuning_selection_binding"]["aggregate_sha256"] == "5" * 64
    assert [
        selection["learner"]["learning_rate"] for selection in contract["primary"]["selections"]
    ] == [0.001, 0.002]
    assert contract["upper_reference"]["learner"]["learning_rate"] == 0.003
    assert [
        (task["primary_environment"], task["model"])
        for task in contract["primary"]["task_contracts"]
    ] == [("tmaze_10", "arcmind"), ("tmaze_10", "gru")]


def test_registered_schema_v4_completion_requires_exact_selection_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    primary, upper = _schema_v4_registered_pair(tmp_path)
    _stub_registered_aggregation(monkeypatch)
    _rewrite_json(
        primary / "completion_index.json",
        lambda value: value["cells"][0].update(candidate_id="ordered_memory.drifted"),
    )
    _refresh_checksums(primary)

    with pytest.raises(UpperReferenceLinkError, match="candidate_id drifts"):
        build_upper_reference_link(primary, upper)


def test_registered_schema_v4_pair_requires_equal_evaluation_counts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    primary, upper = _schema_v4_registered_pair(tmp_path)
    _stub_registered_aggregation(monkeypatch)
    _rewrite_json(
        upper / "registration.json",
        lambda value: value.update(evaluation_episodes_per_env=64),
    )
    _refresh_checksums(upper)

    with pytest.raises(UpperReferenceLinkError, match="evaluation registrations"):
        build_upper_reference_link(primary, upper)


@pytest.mark.parametrize("raw_root_name", ["primary", "upper"])
def test_output_must_be_outside_both_raw_roots(
    tmp_path: Path,
    raw_root_name: str,
) -> None:
    primary = tmp_path / "primary"
    upper = tmp_path / "upper"
    primary.mkdir()
    upper.mkdir()
    selected = primary if raw_root_name == "primary" else upper
    with pytest.raises(UpperReferenceLinkError, match="outside both"):
        link_upper_reference(primary, upper, selected / "derived.json")


def test_matrix_kinds_are_not_inferred_from_paths(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    upper = tmp_path / "upper"
    _write_matrix(primary, matrix_kind="upper_reference")
    _write_matrix(upper, matrix_kind="upper_reference")
    with pytest.raises(UpperReferenceLinkError, match="not a primary_comparison"):
        build_upper_reference_link(primary, upper)


@pytest.mark.parametrize(
    ("upper_options", "message"),
    [
        ({"tier": "smoke"}, "different evidence tiers"),
        ({"seeds": list(reversed(SEEDS))}, "same ordered seed list"),
        ({"learning_rate": 0.002}, "different learner registrations"),
        ({"evaluation_episodes": 3}, "different evaluation registrations"),
        (
            {"environments": list(reversed(UPPER_ENVIRONMENTS))},
            "do not map exactly",
        ),
    ],
)
def test_pairing_contract_drift_is_rejected(
    tmp_path: Path,
    upper_options: dict[str, Any],
    message: str,
) -> None:
    primary, upper = _paired_roots(tmp_path, **upper_options)
    with pytest.raises(UpperReferenceLinkError, match=message):
        build_upper_reference_link(primary, upper)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("dependency_lock_sha256", "a" * 64, "dependency_lock_sha256"),
        ("pobax_commit", "b" * 40, "pobax_commit"),
        ("navix_commit", "c" * 40, "navix_commit"),
    ],
)
def test_source_identity_must_match(
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
) -> None:
    provenance = deepcopy(BASE_PROVENANCE)
    provenance[field] = replacement
    primary = tmp_path / "primary"
    upper = tmp_path / "upper"
    _write_matrix(primary, matrix_kind="primary_comparison")
    _write_matrix(upper, matrix_kind="upper_reference", provenance=provenance)
    with pytest.raises(UpperReferenceLinkError, match=message):
        build_upper_reference_link(primary, upper)


def test_git_commit_and_non_device_runtime_drift_are_rejected(tmp_path: Path) -> None:
    git_provenance = deepcopy(BASE_PROVENANCE)
    git_provenance["git"]["commit"] = "f" * 40
    primary = tmp_path / "git-primary"
    upper = tmp_path / "git-upper"
    _write_matrix(primary, matrix_kind="primary_comparison")
    _write_matrix(upper, matrix_kind="upper_reference", provenance=git_provenance)
    with pytest.raises(UpperReferenceLinkError, match="Git commits differ"):
        build_upper_reference_link(primary, upper)

    runtime_provenance = deepcopy(BASE_PROVENANCE)
    runtime_provenance["runtime_contract"]["python"]["version"] = "3.13.0"
    primary = tmp_path / "runtime-primary"
    upper = tmp_path / "runtime-upper"
    _write_matrix(primary, matrix_kind="primary_comparison")
    _write_matrix(upper, matrix_kind="upper_reference", provenance=runtime_provenance)
    with pytest.raises(UpperReferenceLinkError, match="beyond the allowed"):
        build_upper_reference_link(primary, upper)


def test_canonical_manifest_completion_and_checksum_tampering_fail(
    tmp_path: Path,
) -> None:
    primary, upper = _paired_roots(tmp_path / "manifest")
    _rewrite_json(
        upper / "frozen_manifest.json",
        lambda value: value.update(manifest_sha256="0" * 64),
    )
    _refresh_checksums(upper)
    with pytest.raises(UpperReferenceLinkError, match="canonical content"):
        build_upper_reference_link(primary, upper)

    primary, upper = _paired_roots(tmp_path / "completion")
    _rewrite_json(
        upper / "completion_index.json",
        lambda value: value["cells"][0].update(artifact_sha256="0" * 64),
    )
    _refresh_checksums(upper)
    with pytest.raises(UpperReferenceLinkError, match="artifact_sha256 is incorrect"):
        build_upper_reference_link(primary, upper)

    primary, upper = _paired_roots(tmp_path / "checksum")
    lines = (upper / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    (upper / "checksums.sha256").write_text(
        "\n".join(lines[1:]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(UpperReferenceLinkError, match="exactly cover"):
        build_upper_reference_link(primary, upper)


def test_registered_final_budget_table_covers_all_six_pairs() -> None:
    primary_registration = {
        "environments": [
            {
                "id": primary,
                "total_steps": REGISTERED_TRAIN_STEPS[primary],
            }
            for primary in PRIMARY_ENVIRONMENTS
        ]
    }
    upper_registration = {
        "environments": [
            {
                "id": upper,
                "total_steps": REGISTERED_TRAIN_STEPS[upper],
            }
            for upper in UPPER_ENVIRONMENTS
        ]
    }
    _validate_registered_budgets(
        "registered_final",
        primary_registration,
        upper_registration,
    )
    upper_registration["environments"][0]["total_steps"] += 1
    with pytest.raises(UpperReferenceLinkError, match="budget is invalid"):
        _validate_registered_budgets(
            "registered_final",
            primary_registration,
            upper_registration,
        )
