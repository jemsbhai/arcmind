"""Tests for the local-first research benchmark infrastructure."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from benchmarks.aggregate import aggregate_results
from benchmarks.delayed_recall import DelayedRecallConfig, DelayedRecallDataset
from benchmarks.models import (
    build_arcmind,
    build_parameter_matched_baseline,
    count_parameters,
)
from benchmarks.run_delayed_recall import train_one_run
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
