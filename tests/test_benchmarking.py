"""Tests for the local-first research benchmark infrastructure."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from benchmarks.aggregate import aggregate_results
from benchmarks.delayed_recall import (
    DelayedRecallConfig,
    DelayedRecallDataset,
    latest_value_oracle,
)
from benchmarks.models import (
    build_arcmind,
    build_parameter_matched_baseline,
    count_parameters,
)
from benchmarks.run_delayed_recall import evaluate, train_one_run
from benchmarks.statistics import bootstrap_interval, interquartile_mean


def test_delayed_recall_is_deterministic():
    first = DelayedRecallDataset(8, seed=17)
    second = DelayedRecallDataset(8, seed=17)
    different = DelayedRecallDataset(8, seed=18)

    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(first.targets, second.targets)
    assert not torch.equal(first.inputs, different.inputs)


def test_queries_have_valid_targets_and_positive_lags():
    dataset = DelayedRecallDataset(32, seed=21)
    query_mask = dataset.targets != dataset.ignore_index

    assert query_mask.any()
    assert dataset.targets[query_mask].min() >= 0
    assert dataset.targets[query_mask].max() < dataset.config.num_values
    assert dataset.query_lags[query_mask].min() >= 1
    assert torch.equal(
        dataset.query_target_ages[query_mask],
        dataset.query_lags[query_mask] - 1,
    )
    assert dataset.query_write_counts[query_mask].min() >= 1


def test_latest_value_oracle_and_window_coverage_are_exact():
    dataset = DelayedRecallDataset(32, seed=22)
    query_mask = dataset.targets != dataset.ignore_index
    oracle = latest_value_oracle(dataset.inputs, dataset.config)

    assert torch.equal(oracle, dataset.targets)
    target_is_visible = (
        dataset.query_target_ages[query_mask]
        < dataset.config.exact_recall_window
    )
    assert torch.equal(
        target_is_visible,
        (
            dataset.query_lags[query_mask]
            <= dataset.config.exact_recall_window
        ),
    )


def test_events_occur_only_at_decision_boundaries():
    dataset = DelayedRecallDataset(4, seed=23)
    event_flags = dataset.inputs[:, :, :2].sum(dim=-1)
    non_boundaries = torch.ones(dataset.config.sequence_length, dtype=torch.bool)
    non_boundaries[:: dataset.config.sensor_stride] = False

    assert event_flags[:, non_boundaries].sum() == 0


@pytest.mark.parametrize(
    "baseline",
    ["memoryless_mlp", "gru", "lstm", "causal_transformer"],
)
def test_baselines_are_parameter_matched(baseline):
    config = replace(
        DelayedRecallConfig(),
        num_decisions=8,
        sensor_stride=2,
    )
    reference = build_arcmind(
        config.input_dim,
        config.num_values,
        sensor_stride=config.sensor_stride,
        exact_recall_window=config.exact_recall_window,
        variant="arcmind",
    )
    target = count_parameters(reference)
    model = build_parameter_matched_baseline(
        baseline,
        input_dim=config.input_dim,
        output_dim=config.num_values,
        sequence_length=config.sequence_length,
        target_parameters=target,
    )

    assert abs(count_parameters(model) / target - 1.0) < 0.1
    output = model(torch.randn(2, config.sequence_length, config.input_dim))
    assert output.shape == (2, config.sequence_length, config.num_values)


def test_interquartile_mean_uses_fractional_boundaries():
    assert interquartile_mean([0.0, 1.0, 2.0, 100.0]) == pytest.approx(1.5)
    assert interquartile_mean([3.0]) == pytest.approx(3.0)


def test_bootstrap_interval_is_deterministic():
    values = [0.1, 0.2, 0.3, 0.4]
    first = bootstrap_interval(values, num_resamples=100, seed=7)
    second = bootstrap_interval(values, num_resamples=100, seed=7)

    assert first == second
    assert first[0] <= np.mean(values) <= first[1]


def test_aggregation_preserves_seeds_and_raw_files(tmp_path):
    paths = []
    for seed, accuracy in [(1, 0.5), (2, 0.75)]:
        path = tmp_path / f"arcmind_seed-{seed}.json"
        record = {
            "schema_version": 1,
            "model": "arcmind",
            "seed": seed,
            "parameter_count": 100,
            "parameter_ratio": 1.0,
            "test": {
                "accuracy": accuracy,
                "nll": 1.0 - accuracy,
                "short_lag_accuracy": accuracy,
                "long_lag_accuracy": accuracy,
                "examples_per_second": 10.0,
            },
        }
        path.write_text(json.dumps(record), encoding="utf-8")
        paths.append(path)

    summary = aggregate_results(paths)

    assert summary["num_raw_records"] == 2
    assert summary["models"]["arcmind"]["seeds"] == [1, 2]
    assert summary["models"]["arcmind"]["metrics"]["accuracy"]["mean"] == pytest.approx(
        0.625
    )
    assert len(summary["models"]["arcmind"]["raw_files"]) == 2


def test_training_contract_selects_on_validation_and_tests_once():
    config = replace(
        DelayedRecallConfig(),
        num_decisions=8,
        sensor_stride=2,
        exact_recall_window=4,
    )
    result, _ = train_one_run(
        "memoryless_mlp",
        seed=31,
        task_config=config,
        train_examples=32,
        validation_examples=16,
        test_examples=16,
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=0.0,
        device=torch.device("cpu"),
    )

    assert result["selection"]["metric"] == "validation_nll"
    assert result["selection"]["best_epoch"] == 1
    assert result["selection"]["test_evaluations"] == 1
    assert result["test"]["queries"] > 0
    assert "test" not in result["history"][0]


def test_arcmind_evaluation_records_exact_lag_and_fusion_diagnostics():
    config = replace(
        DelayedRecallConfig(),
        num_decisions=8,
        sensor_stride=2,
        exact_recall_window=4,
    )
    dataset = DelayedRecallDataset(16, config=config, seed=37)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model = build_arcmind(
        config.input_dim,
        config.num_values,
        sensor_stride=config.sensor_stride,
        exact_recall_window=config.exact_recall_window,
        variant="arcmind",
    )

    metrics = evaluate(
        model,
        loader,
        device=torch.device("cpu"),
        short_lag_limit=config.exact_recall_window,
        collect_arcmind_diagnostics=True,
    )

    query_counts = metrics["query_count_by_lag"]
    accuracy_by_lag = metrics["accuracy_by_lag"]
    diagnostics = metrics["arcmind_diagnostics"]
    assert sum(query_counts.values()) == metrics["queries"]
    assert set(query_counts) == set(accuracy_by_lag)
    assert set(query_counts) == set(
        diagnostics["counterfactual_fast_path_accuracy_by_lag"]
    )
    assert set(query_counts) == set(diagnostics["mean_slow_gate_by_lag"])
    assert set(query_counts) == set(diagnostics["mean_slow_delta_norm_by_lag"])
    assert 0.0 <= diagnostics["counterfactual_fast_path_accuracy"] <= 1.0
    assert 0.0 <= diagnostics["mean_slow_gate"] <= 1.0
    assert diagnostics["mean_slow_delta_norm"] >= 0.0


@pytest.mark.parametrize(
    ("variant", "flag"),
    [
        ("arcmind_no_memory", "ablate_memory"),
        ("arcmind_no_gate", "ablate_gating"),
    ],
)
def test_exploratory_fusion_variants_toggle_one_mechanism(variant, flag):
    config = replace(DelayedRecallConfig(), num_decisions=8, sensor_stride=2)
    reference = build_arcmind(
        config.input_dim,
        config.num_values,
        sensor_stride=config.sensor_stride,
        exact_recall_window=config.exact_recall_window,
        variant="arcmind",
    )
    ablation = build_arcmind(
        config.input_dim,
        config.num_values,
        sensor_stride=config.sensor_stride,
        exact_recall_window=config.exact_recall_window,
        variant=variant,
    )

    assert getattr(ablation.config, flag)
    assert count_parameters(ablation) == count_parameters(reference)


def test_fast_start_variant_only_changes_gate_initialization():
    config = replace(DelayedRecallConfig(), num_decisions=8, sensor_stride=2)
    torch.manual_seed(41)
    reference = build_arcmind(
        config.input_dim,
        config.num_values,
        sensor_stride=config.sensor_stride,
        exact_recall_window=config.exact_recall_window,
        variant="arcmind",
    )
    torch.manual_seed(41)
    fast_start = build_arcmind(
        config.input_dim,
        config.num_values,
        sensor_stride=config.sensor_stride,
        exact_recall_window=config.exact_recall_window,
        variant="arcmind_fast_start",
    )

    reference_state = reference.state_dict()
    fast_start_state = fast_start.state_dict()
    assert reference_state.keys() == fast_start_state.keys()
    for name in reference_state:
        if name == "gate.0.bias":
            assert torch.allclose(
                fast_start_state[name],
                torch.full_like(fast_start_state[name], -2.944439),
            )
            assert not torch.equal(reference_state[name], fast_start_state[name])
        elif name == "gate.0.weight":
            assert torch.count_nonzero(fast_start_state[name]) == 0
            assert not torch.equal(reference_state[name], fast_start_state[name])
        else:
            assert torch.equal(reference_state[name], fast_start_state[name])
    sample_gate = fast_start.gate(
        torch.randn(3, 2 * fast_start.config.d_model)
    )
    assert sample_gate.mean().item() == pytest.approx(0.05, abs=1e-6)


def test_fast_aux_variant_has_identical_initial_parameters():
    config = replace(DelayedRecallConfig(), num_decisions=8, sensor_stride=2)
    torch.manual_seed(43)
    reference = build_arcmind(
        config.input_dim,
        config.num_values,
        sensor_stride=config.sensor_stride,
        exact_recall_window=config.exact_recall_window,
        variant="arcmind",
    )
    torch.manual_seed(43)
    fast_aux = build_arcmind(
        config.input_dim,
        config.num_values,
        sensor_stride=config.sensor_stride,
        exact_recall_window=config.exact_recall_window,
        variant="arcmind_fast_aux",
    )

    for name, reference_value in reference.state_dict().items():
        assert torch.equal(reference_value, fast_aux.state_dict()[name])


def test_fast_auxiliary_variant_trains_with_declared_loss_weight():
    config = replace(
        DelayedRecallConfig(),
        num_decisions=6,
        sensor_stride=2,
        exact_recall_window=3,
    )
    result, _ = train_one_run(
        "arcmind_fast_aux",
        seed=47,
        task_config=config,
        train_examples=8,
        validation_examples=8,
        test_examples=8,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        device=torch.device("cpu"),
    )

    assert result["training_config"]["fast_path_auxiliary_weight"] == 1.0
    assert "arcmind_diagnostics" in result["test"]


def test_match_abstention_routes_to_fast_path_without_visible_write():
    config = replace(
        DelayedRecallConfig(),
        num_decisions=12,
        sensor_stride=2,
        exact_recall_window=3,
    )
    dataset = DelayedRecallDataset(64, config=config, seed=53)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = build_arcmind(
        config.input_dim,
        config.num_values,
        sensor_stride=config.sensor_stride,
        exact_recall_window=config.exact_recall_window,
        variant="arcmind_match_abstention",
        num_keys=config.num_keys,
    )
    reference = build_arcmind(
        config.input_dim,
        config.num_values,
        sensor_stride=config.sensor_stride,
        exact_recall_window=config.exact_recall_window,
        variant="arcmind",
        num_keys=config.num_keys,
    )

    metrics = evaluate(
        model,
        loader,
        device=torch.device("cpu"),
        short_lag_limit=config.exact_recall_window,
        collect_arcmind_diagnostics=True,
    )
    gate_by_lag = metrics["arcmind_diagnostics"]["mean_slow_gate_by_lag"]
    long_lags = [
        lag for lag in gate_by_lag if int(lag) > config.exact_recall_window
    ]

    assert count_parameters(model) == count_parameters(reference)
    assert long_lags
    assert all(gate_by_lag[lag] == 0.0 for lag in long_lags)
    assert any(
        value > 0.0
        for lag, value in gate_by_lag.items()
        if int(lag) <= config.exact_recall_window
    )
