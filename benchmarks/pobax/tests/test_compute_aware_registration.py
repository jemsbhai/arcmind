"""Focused tests for compute-aware POBAX registration schemas."""

from __future__ import annotations

from copy import deepcopy

import pytest

from benchmarks.pobax.registration_protocol import (
    COMPUTE_AWARE_FINAL_SEEDS,
    COMPUTE_AWARE_TUNING_PANEL,
    COMPUTE_AWARE_TUNING_SEEDS,
    LEARNER_FIELDS_V2,
    REGISTRATION_FIELDS_V1,
    REGISTRATION_FIELDS_V2,
    REGISTRATION_FIELDS_V3,
    REGISTRATION_FIELDS_V4,
    REGISTRATION_FIELDS_V5,
    REGISTRATION_FIELDS_V6,
    normalize_learner,
    normalize_learner_bindings,
    normalize_panel_selection_binding,
    normalize_shared_learner_grid,
    normalize_task_model_incidence,
    normalize_tuned_families,
    registration_fields,
    validate_compute_aware_final_contract,
    validate_compute_aware_tuning_contract,
)


def _learner(learning_rate: float = 0.00025) -> dict[str, int | float | bool]:
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


def _families() -> list[dict[str, str]]:
    return [
        {"family_id": "memoryless_mlp", "implementation_model": "memoryless_mlp"},
        {"family_id": "gru", "implementation_model": "gru"},
        {"family_id": "arcmind", "implementation_model": "arcmind"},
    ]


def _grid() -> list[dict[str, object]]:
    return [
        {"learner_id": "lr_low", "learner": _learner(0.0001)},
        {"learner_id": "lr_mid", "learner": _learner(0.00025)},
        {"learner_id": "lr_high", "learner": _learner(0.0005)},
    ]


def _selection_binding() -> dict[str, object]:
    source_hash = "6" * 64
    return {
        "raw_matrix_path": "benchmark_results/pobax/tuning-panel-v1",
        "aggregate_path": "benchmark_results/pobax/aggregates/tuning-panel-v1.json",
        "aggregate_sha256": "0" * 64,
        "source_registration_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "source_completion_index_sha256": "3" * 64,
        "source_checksum_manifest_sha256": "4" * 64,
        "source_implementation_sha256": source_hash,
        "selections": [
            {
                "model_family": family["family_id"],
                "implementation_model": family["implementation_model"],
                "candidate_id": f"{family['family_id']}.lr_mid",
                "learner_id": "lr_mid",
                "learner": _learner(),
                "implementation_source_sha256": source_hash,
            }
            for family in _families()
        ],
    }


def _models() -> list[str]:
    return ["memoryless_mlp", "gru", "arcmind", "arcmind_no_memory"]


def _learner_bindings() -> list[dict[str, str]]:
    return [
        {
            "model": "memoryless_mlp",
            "mode": "selected",
            "source_model_family": "memoryless_mlp",
        },
        {"model": "gru", "mode": "selected", "source_model_family": "gru"},
        {"model": "arcmind", "mode": "selected", "source_model_family": "arcmind"},
        {
            "model": "arcmind_no_memory",
            "mode": "inherited",
            "source_model_family": "arcmind",
        },
    ]


def _final_environments() -> dict[str, int]:
    return {
        "tmaze_10": 1_000_000,
        "rocksample_11_11": 5_000_000,
        "battleship_10": 10_000_000,
        "Navix-DMLab-Maze-01-v0": 10_000_000,
    }


def _incidence() -> list[dict[str, object]]:
    common = ["memoryless_mlp", "gru", "arcmind"]
    return [
        {"environment": "tmaze_10", "models": common},
        {
            "environment": "rocksample_11_11",
            "models": [*common, "arcmind_no_memory"],
        },
        {"environment": "battleship_10", "models": common},
        {"environment": "Navix-DMLab-Maze-01-v0", "models": common},
    ]


def _validate_tuning(
    *,
    families: object | None = None,
    grid: object | None = None,
    environments: dict[str, int] | None = None,
    seeds: tuple[int, ...] = COMPUTE_AWARE_TUNING_SEEDS,
    comparison_profile: str = "arcmind_shared_comparison",
    matrix_kind: str = "hyperparameter_selection",
    quick: bool = False,
) -> None:
    normalized_families = normalize_tuned_families(
        _families() if families is None else families
    )
    normalized_grid = normalize_shared_learner_grid(_grid() if grid is None else grid)
    validate_compute_aware_tuning_contract(
        schema_version=5,
        comparison_profile=comparison_profile,
        matrix_kind=matrix_kind,
        tuned_families=normalized_families,
        learner_grid=normalized_grid,
        environments=(
            dict(COMPUTE_AWARE_TUNING_PANEL)
            if environments is None
            else environments
        ),
        seeds=seeds,
        quick=quick,
    )


def _validate_final(
    *,
    models: object | None = None,
    learner_bindings: object | None = None,
    incidence: object | None = None,
    tuning_selection: object | None = None,
    environments: dict[str, int] | None = None,
    seeds: tuple[int, ...] = COMPUTE_AWARE_FINAL_SEEDS,
) -> None:
    validate_compute_aware_final_contract(
        schema_version=6,
        comparison_profile="arcmind_shared_comparison",
        matrix_kind="primary_comparison",
        models=_models() if models is None else models,
        learner_bindings=(
            _learner_bindings() if learner_bindings is None else learner_bindings
        ),
        task_model_incidence=_incidence() if incidence is None else incidence,
        tuning_selection=(
            _selection_binding() if tuning_selection is None else tuning_selection
        ),
        environments=(
            _final_environments() if environments is None else environments
        ),
        seeds=seeds,
        quick=False,
    )


def test_registration_field_sets_preserve_v1_to_v4_and_add_v5_v6() -> None:
    assert registration_fields(1) == REGISTRATION_FIELDS_V1
    assert registration_fields(2) == REGISTRATION_FIELDS_V2
    assert registration_fields(3) == REGISTRATION_FIELDS_V3
    assert registration_fields(4) == REGISTRATION_FIELDS_V4
    assert registration_fields(5) == REGISTRATION_FIELDS_V5
    assert registration_fields(6) == REGISTRATION_FIELDS_V6
    assert REGISTRATION_FIELDS_V5 == (
        REGISTRATION_FIELDS_V2 - {"models", "learner"}
    ) | {"tuned_families", "learner_grid"}
    assert REGISTRATION_FIELDS_V6 == (REGISTRATION_FIELDS_V2 - {"learner"}) | {
        "learner_bindings",
        "task_model_incidence",
        "tuning_selection",
    }


@pytest.mark.parametrize("schema_version", [5, 6])
def test_compute_aware_schemas_use_the_complete_v2_learner(
    schema_version: int,
) -> None:
    assert set(normalize_learner(_learner(), schema_version=schema_version)) == (
        LEARNER_FIELDS_V2
    )


def test_valid_compute_aware_tuning_contract_normalizes_exact_design() -> None:
    families = normalize_tuned_families(_families())
    grid = normalize_shared_learner_grid(_grid())

    assert [item["family_id"] for item in families] == [
        "memoryless_mlp",
        "gru",
        "arcmind",
    ]
    assert [item["learner_id"] for item in grid] == [
        "lr_low",
        "lr_mid",
        "lr_high",
    ]
    _validate_tuning()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.append(deepcopy(value[0])),
            "unique portable identifier",
        ),
        (
            lambda value: value[1].update(implementation_model="memoryless_mlp"),
            "unique portable model identifier",
        ),
        (
            lambda value: value[0].update(extra=True),
            "exactly family_id and implementation_model",
        ),
        (
            lambda value: value[0].update(implementation_model="not_registered"),
            "registered policy implementations",
        ),
    ],
)
def test_tuned_families_fail_closed(mutation, message: str) -> None:
    families = _families()
    mutation(families)

    with pytest.raises(ValueError, match=message):
        normalize_tuned_families(families)


def _duplicate_grid_learner(grid: list[dict[str, object]]) -> None:
    grid[1]["learner"] = deepcopy(grid[0]["learner"])


def _drift_grid_structure(grid: list[dict[str, object]]) -> None:
    grid[1]["learner"]["num_envs"] = 4
    grid[1]["learner"]["rollout_steps"] = 250


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__(slice(1, None), []), "at least two"),
        (
            lambda value: value[1].update(learner_id="lr_low"),
            "unique portable identifier",
        ),
        (_duplicate_grid_learner, "duplicate normalized learner"),
        (_drift_grid_structure, "identical num_envs"),
        (lambda value: value[0].update(extra=True), "exactly learner_id and learner"),
    ],
)
def test_shared_learner_grid_fails_closed(mutation, message: str) -> None:
    grid = _grid()
    mutation(grid)

    with pytest.raises(ValueError, match=message):
        normalize_shared_learner_grid(grid)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "environments": {
                    "rocksample_11_11": 1_000_000,
                    "tmaze_10": 1_000_000,
                }
            },
            "exact ordered two-task panel",
        ),
        (
            {
                "environments": {
                    "tmaze_10": 1_000_000,
                    "rocksample_11_11": 5_000_000,
                }
            },
            "exact ordered two-task panel",
        ),
        ({"seeds": (4409, 5519, 6638)}, "exact ordered seed manifest"),
        ({"comparison_profile": "pobax_author_semantics"}, "shared_comparison"),
        ({"matrix_kind": "primary_comparison"}, "hyperparameter_selection"),
        ({"quick": True}, "cannot use quick"),
    ],
)
def test_compute_aware_tuning_contract_fails_closed(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_tuning(**kwargs)


def test_panel_selection_binding_normalizes_winners_and_paths() -> None:
    binding = normalize_panel_selection_binding(_selection_binding())

    assert binding["raw_matrix_path"] == (
        "benchmark_results/pobax/tuning-panel-v1"
    )
    assert [item["candidate_id"] for item in binding["selections"]] == [
        "memoryless_mlp.lr_mid",
        "gru.lr_mid",
        "arcmind.lr_mid",
    ]


def _duplicate_selection_family(binding: dict[str, object]) -> None:
    binding["selections"][1]["model_family"] = "memoryless_mlp"
    binding["selections"][1]["candidate_id"] = "memoryless_mlp.lr_mid"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["selections"][0].update(
                candidate_id="memoryless_mlp.lr_other"
            ),
            "model_family",
        ),
        (
            lambda value: value["selections"][0].update(
                implementation_source_sha256="7" * 64
            ),
            "must equal",
        ),
        (
            lambda value: value.update(aggregate_sha256="A" * 64),
            "lowercase SHA256",
        ),
        (
            lambda value: value.update(raw_matrix_path="../escape"),
            "normalized relative path",
        ),
        (_duplicate_selection_family, "unique portable identifier"),
        (
            lambda value: value["selections"][0].update(extra=True),
            "must contain exactly",
        ),
    ],
)
def test_panel_selection_binding_fails_closed(mutation, message: str) -> None:
    binding = _selection_binding()
    mutation(binding)

    with pytest.raises(ValueError, match=message):
        normalize_panel_selection_binding(binding)


def test_valid_learner_bindings_and_sparse_incidence_normalize() -> None:
    bindings = normalize_learner_bindings(_learner_bindings(), models=_models())
    incidence = normalize_task_model_incidence(
        _incidence(),
        environments=list(_final_environments()),
        models=_models(),
    )

    assert bindings[-1] == {
        "model": "arcmind_no_memory",
        "mode": "inherited",
        "source_model_family": "arcmind",
    }
    assert incidence[0]["models"] == ("memoryless_mlp", "gru", "arcmind")
    assert incidence[1]["models"][-1] == "arcmind_no_memory"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value[0].update(model="gru"),
            "exact models order",
        ),
        (
            lambda value: value[0].update(mode="copied"),
            "selected.*inherited",
        ),
        (
            lambda value: value.pop(),
            "exactly one entry per model",
        ),
        (
            lambda value: value[0].update(extra=True),
            "must contain exactly",
        ),
    ],
)
def test_learner_bindings_fail_closed(mutation, message: str) -> None:
    bindings = _learner_bindings()
    mutation(bindings)

    with pytest.raises(ValueError, match=message):
        normalize_learner_bindings(bindings, models=_models())


def _remove_arcmind_from_one_task(incidence: list[dict[str, object]]) -> None:
    incidence[0]["models"] = ["memoryless_mlp", "gru"]


def _leave_only_arcmind_common(incidence: list[dict[str, object]]) -> None:
    incidence[0]["models"] = ["memoryless_mlp", "arcmind"]
    incidence[2]["models"] = ["gru", "arcmind"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop(), "exactly one entry per environment"),
        (
            lambda value: value[0].update(environment="rocksample_11_11"),
            "exact environments order",
        ),
        (
            lambda value: value[0].update(models=["gru", "memoryless_mlp", "arcmind"]),
            "global models order",
        ),
        (
            lambda value: value[0].update(
                models=["memoryless_mlp", "gru", "not_registered", "arcmind"]
            ),
            "members of the global models",
        ),
        (_remove_arcmind_from_one_task, "must contain arcmind"),
        (
            lambda value: value[1].update(
                models=["memoryless_mlp", "gru", "arcmind"]
            ),
            "every global model",
        ),
        (_leave_only_arcmind_common, "at least two all-task common models"),
    ],
)
def test_task_model_incidence_fails_closed(mutation, message: str) -> None:
    incidence = _incidence()
    mutation(incidence)

    with pytest.raises(ValueError, match=message):
        normalize_task_model_incidence(
            incidence,
            environments=list(_final_environments()),
            models=_models(),
        )


def test_valid_compute_aware_final_contract_supports_inherited_ablation() -> None:
    _validate_final()


def _unknown_binding_source(bindings: list[dict[str, str]]) -> None:
    bindings[-1]["source_model_family"] = "unknown"


def _selected_model_drift(bindings: list[dict[str, str]]) -> None:
    bindings[-1].update(mode="selected", source_model_family="arcmind")


def _inherited_model_matches_source(bindings: list[dict[str, str]]) -> None:
    bindings[2]["mode"] = "inherited"


def _missing_direct_selection(bindings: list[dict[str, str]]) -> None:
    bindings[1].update(mode="inherited", source_model_family="arcmind")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "environments": {
                    **_final_environments(),
                    "rocksample_11_11": 1_000_000,
                }
            },
            "published task budget",
        ),
        (
            {"seeds": tuple(range(10_000, 10_009))},
            "exact ordered seed manifest",
        ),
        (
            {"learner_bindings": _unknown_binding_source},
            "unknown source family",
        ),
        (
            {"learner_bindings": _selected_model_drift},
            "must execute its source family's implementation",
        ),
        (
            {"learner_bindings": _inherited_model_matches_source},
            "must execute a different model",
        ),
        (
            {"learner_bindings": _missing_direct_selection},
            "one direct selected binding",
        ),
    ],
)
def test_compute_aware_final_contract_fails_closed(
    kwargs: dict[str, object],
    message: str,
) -> None:
    learner_binding_mutation = kwargs.pop("learner_bindings", None)
    if callable(learner_binding_mutation):
        bindings = _learner_bindings()
        learner_binding_mutation(bindings)
        kwargs["learner_bindings"] = bindings

    with pytest.raises(ValueError, match=message):
        _validate_final(**kwargs)
