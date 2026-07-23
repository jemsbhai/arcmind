"""Train and evaluate delayed-recall models with validation-only selection."""

import argparse
import copy
import json
import math
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from arcmind import ArcMindModel
from benchmarks.delayed_recall import DelayedRecallConfig, DelayedRecallDataset
from benchmarks.models import (
    build_arcmind,
    build_parameter_matched_baseline,
    count_parameters,
)

REGISTERED_MODELS = (
    "memoryless_mlp",
    "gru",
    "lstm",
    "causal_transformer",
    "arcmind_ssm_only",
    "arcmind_unordered",
    "arcmind",
)

EXPLORATORY_MODELS = (
    "arcmind_no_memory",
    "arcmind_no_gate",
    "arcmind_fast_start",
    "arcmind_fast_aux",
    "arcmind_match_abstention",
)


def set_seed(seed: int) -> None:
    """Seed model initialization and stochastic training operations."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git_metadata(repository: Path) -> dict[str, str | bool | None]:
    def run(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    revision = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"revision": revision, "dirty": bool(status) if status is not None else None}


def _forward(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    if isinstance(model, ArcMindModel):
        model.reset_memory(batch_size=inputs.shape[0])
    return model(inputs)


def _forward_with_arcmind_diagnostics(
    model: torch.nn.Module,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    """Capture fast, slow, and gate tensors without changing the model API."""
    if (
        not isinstance(model, ArcMindModel)
        or model.config.ablate_ssm
        or model.config.ablate_attention
        or model.config.ablate_gating
    ):
        return _forward(model, inputs), None

    fast_outputs: list[torch.Tensor] = []
    slow_outputs: list[torch.Tensor] = []
    gate_outputs: list[torch.Tensor] = []
    handles = [
        model.ssm_core.register_forward_hook(
            lambda _module, _arguments, output: fast_outputs.append(output)
        ),
        model.slow_attention.register_forward_hook(
            lambda _module, _arguments, output: slow_outputs.append(output)
        ),
        model.gate.register_forward_hook(
            lambda _module, _arguments, output: gate_outputs.append(output)
        ),
    ]
    try:
        logits = _forward(model, inputs)
    finally:
        for handle in handles:
            handle.remove()

    expected_decisions = (
        inputs.shape[1] + model.decision_stride - 1
    ) // model.decision_stride
    if (
        len(fast_outputs) != 1
        or len(slow_outputs) != expected_decisions
        or len(gate_outputs) != expected_decisions
    ):
        raise RuntimeError(
            "ArcMind diagnostic capture did not match the forward schedule: "
            f"fast={len(fast_outputs)}, slow={len(slow_outputs)}, "
            f"gate={len(gate_outputs)}, expected_decisions={expected_decisions}"
        )
    actual_gate_values = getattr(
        model,
        "_last_actual_gate_values",
        None,
    )
    return logits, {
        "fast": fast_outputs[0],
        "slow": torch.cat(slow_outputs, dim=1),
        "gate": (
            actual_gate_values
            if actual_gate_values is not None
            else torch.cat(gate_outputs, dim=1)
        ),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    short_lag_limit: int,
    collect_arcmind_diagnostics: bool = False,
) -> dict[str, object]:
    """Evaluate all query labels in a split."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    queries = 0
    short_correct = 0
    short_queries = 0
    long_correct = 0
    long_queries = 0
    max_lag = loader.dataset.config.num_decisions
    lag_queries = torch.zeros(max_lag + 1, dtype=torch.long, device=device)
    lag_correct = torch.zeros_like(lag_queries)
    diagnostic_queries = torch.zeros_like(lag_queries)
    diagnostic_fast_correct = torch.zeros_like(lag_queries)
    diagnostic_gate_sum = torch.zeros(
        max_lag + 1,
        dtype=torch.float64,
        device=device,
    )
    diagnostic_delta_sum = torch.zeros_like(diagnostic_gate_sum)
    _synchronize(device)
    started = time.perf_counter()

    for inputs, targets, query_lags in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        query_lags = query_lags.to(device)
        if collect_arcmind_diagnostics:
            logits, diagnostic_capture = _forward_with_arcmind_diagnostics(
                model,
                inputs,
            )
        else:
            logits = _forward(model, inputs)
            diagnostic_capture = None
        mask = targets != DelayedRecallDataset.ignore_index
        query_logits = logits[mask]
        query_targets = targets[mask]
        query_lag_values = query_lags[mask]

        loss_sum += F.cross_entropy(
            query_logits,
            query_targets,
            reduction="sum",
        ).item()
        predictions = query_logits.argmax(dim=-1)
        is_correct = predictions == query_targets
        correct += is_correct.sum().item()
        queries += query_targets.numel()
        lag_queries += torch.bincount(
            query_lag_values,
            minlength=max_lag + 1,
        )
        lag_correct += torch.bincount(
            query_lag_values[is_correct],
            minlength=max_lag + 1,
        )

        short_mask = query_lag_values <= short_lag_limit
        long_mask = ~short_mask
        short_correct += is_correct[short_mask].sum().item()
        short_queries += short_mask.sum().item()
        long_correct += is_correct[long_mask].sum().item()
        long_queries += long_mask.sum().item()

        if diagnostic_capture is not None:
            query_locations = mask.nonzero(as_tuple=False)
            query_batches = query_locations[:, 0]
            query_timesteps = query_locations[:, 1]
            query_decisions = query_timesteps // model.decision_stride
            fast_query = diagnostic_capture["fast"][mask]
            slow_query = diagnostic_capture["slow"][
                query_batches,
                query_decisions,
            ]
            gate_query = diagnostic_capture["gate"][mask]
            counterfactual_fast_logits = model.action_head(fast_query)
            counterfactual_fast_correct = (
                counterfactual_fast_logits.argmax(dim=-1) == query_targets
            )
            slow_delta_norm = torch.linalg.vector_norm(
                slow_query - fast_query,
                dim=-1,
            ) / math.sqrt(model.config.d_model)
            mean_slow_gate = gate_query.mean(dim=-1)

            diagnostic_queries += torch.bincount(
                query_lag_values,
                minlength=max_lag + 1,
            )
            diagnostic_fast_correct += torch.bincount(
                query_lag_values[counterfactual_fast_correct],
                minlength=max_lag + 1,
            )
            diagnostic_gate_sum.index_add_(
                0,
                query_lag_values,
                mean_slow_gate.to(torch.float64),
            )
            diagnostic_delta_sum.index_add_(
                0,
                query_lag_values,
                slow_delta_norm.to(torch.float64),
            )

    _synchronize(device)
    elapsed = time.perf_counter() - started
    populated_lags = torch.nonzero(lag_queries, as_tuple=False).flatten().tolist()
    result: dict[str, object] = {
        "nll": loss_sum / queries,
        "accuracy": correct / queries,
        "short_lag_accuracy": short_correct / short_queries,
        "long_lag_accuracy": (
            long_correct / long_queries if long_queries else float("nan")
        ),
        "queries": queries,
        "examples_per_second": len(loader.dataset) / elapsed,
        "evaluation_seconds": elapsed,
        "query_count_by_lag": {
            str(lag): int(lag_queries[lag].item())
            for lag in populated_lags
        },
        "accuracy_by_lag": {
            str(lag): lag_correct[lag].item() / lag_queries[lag].item()
            for lag in populated_lags
        },
    }
    diagnostic_lags = torch.nonzero(
        diagnostic_queries,
        as_tuple=False,
    ).flatten().tolist()
    if diagnostic_lags:
        total_diagnostic_queries = diagnostic_queries.sum().item()
        result["arcmind_diagnostics"] = {
            "counterfactual_fast_path_accuracy": (
                diagnostic_fast_correct.sum().item()
                / total_diagnostic_queries
            ),
            "counterfactual_fast_path_accuracy_by_lag": {
                str(lag): (
                    diagnostic_fast_correct[lag].item()
                    / diagnostic_queries[lag].item()
                )
                for lag in diagnostic_lags
            },
            "mean_slow_gate": (
                diagnostic_gate_sum.sum().item()
                / total_diagnostic_queries
            ),
            "mean_slow_gate_by_lag": {
                str(lag): (
                    diagnostic_gate_sum[lag].item()
                    / diagnostic_queries[lag].item()
                )
                for lag in diagnostic_lags
            },
            "mean_slow_delta_norm": (
                diagnostic_delta_sum.sum().item()
                / total_diagnostic_queries
            ),
            "mean_slow_delta_norm_by_lag": {
                str(lag): (
                    diagnostic_delta_sum[lag].item()
                    / diagnostic_queries[lag].item()
                )
                for lag in diagnostic_lags
            },
        }
    return result


def train_one_run(
    model_name: str,
    *,
    seed: int,
    task_config: DelayedRecallConfig,
    train_examples: int,
    validation_examples: int,
    test_examples: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    progress: bool = False,
) -> tuple[dict, torch.nn.Module]:
    """Train one seed, select on validation NLL, and evaluate test exactly once."""
    set_seed(seed)
    datasets = {
        "train": DelayedRecallDataset(train_examples, config=task_config, seed=7001),
        "validation": DelayedRecallDataset(
            validation_examples,
            config=task_config,
            seed=7002,
        ),
        "test": DelayedRecallDataset(test_examples, config=task_config, seed=7003),
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
        ),
    }

    reference_model = build_arcmind(
        task_config.input_dim,
        task_config.num_values,
        sensor_stride=task_config.sensor_stride,
        exact_recall_window=task_config.exact_recall_window,
        variant="arcmind",
        num_keys=task_config.num_keys,
    )
    target_parameters = count_parameters(reference_model)
    set_seed(seed)
    if model_name.startswith("arcmind"):
        model = build_arcmind(
            task_config.input_dim,
            task_config.num_values,
            sensor_stride=task_config.sensor_stride,
            exact_recall_window=task_config.exact_recall_window,
            variant=model_name,
            num_keys=task_config.num_keys,
        )
    else:
        model = build_parameter_matched_baseline(
            model_name,
            input_dim=task_config.input_dim,
            output_dim=task_config.num_values,
            sequence_length=task_config.sequence_length,
            target_parameters=target_parameters,
        )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    best_validation_nll = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    _synchronize(device)
    training_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        query_count = 0
        for inputs, targets, _ in loaders["train"]:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if model_name == "arcmind_fast_aux":
                logits, diagnostic_capture = _forward_with_arcmind_diagnostics(
                    model,
                    inputs,
                )
            else:
                logits = _forward(model, inputs)
                diagnostic_capture = None
            mask = targets != DelayedRecallDataset.ignore_index
            loss = F.cross_entropy(logits[mask], targets[mask])
            if diagnostic_capture is not None:
                fast_logits = model.action_head(diagnostic_capture["fast"])
                loss = loss + F.cross_entropy(
                    fast_logits[mask],
                    targets[mask],
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            queries = mask.sum().item()
            loss_sum += loss.item() * queries
            query_count += queries

        validation = evaluate(
            model,
            loaders["validation"],
            device=device,
            short_lag_limit=task_config.exact_recall_window,
        )
        history.append(
            {
                "epoch": epoch,
                "train_nll": loss_sum / query_count,
                "validation_nll": validation["nll"],
                "validation_accuracy": validation["accuracy"],
            }
        )
        if validation["nll"] < best_validation_nll:
            best_validation_nll = float(validation["nll"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if progress:
            print(
                f"{model_name} seed={seed} epoch={epoch}/{epochs}: "
                f"train_nll={loss_sum / query_count:.4f}, "
                f"validation_nll={validation['nll']:.4f}, "
                f"validation_accuracy={validation['accuracy']:.4f}",
                flush=True,
            )

    _synchronize(device)
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("training produced no selected checkpoint")
    model.load_state_dict(best_state)

    # The test split is intentionally first touched after checkpoint selection.
    test_metrics = evaluate(
        model,
        loaders["test"],
        device=device,
        short_lag_limit=task_config.exact_recall_window,
        collect_arcmind_diagnostics=True,
    )
    result = {
        "schema_version": 1,
        "benchmark": "delayed_sensor_recall",
        "model": model_name,
        "seed": seed,
        "parameter_count": count_parameters(model),
        "target_parameter_count": target_parameters,
        "parameter_ratio": count_parameters(model) / target_parameters,
        "selection": {
            "metric": "validation_nll",
            "best_value": best_validation_nll,
            "best_epoch": best_epoch,
            "test_evaluations": 1,
        },
        "test": test_metrics,
        "training_seconds": training_seconds,
        "history": history,
        "task_config": asdict(task_config),
        "training_config": {
            "train_examples": train_examples,
            "validation_examples": validation_examples,
            "test_examples": test_examples,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "fast_path_auxiliary_weight": (
                1.0 if model_name == "arcmind_fast_aux" else 0.0
            ),
        },
    }
    return result, model


def _load_protocol() -> dict:
    path = Path(__file__).with_name("protocol.yaml")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _device_from_argument(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    protocol = _load_protocol()
    stage = protocol["stages"]["local_diagnostic"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=REGISTERED_MODELS + EXPLORATORY_MODELS,
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-examples", type=int)
    parser.add_argument("--validation-examples", type=int)
    parser.add_argument("--test-examples", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    models = args.models or stage["models"]
    seeds = args.seeds or stage["seeds"]
    sizes = {
        "train": args.train_examples or stage["train_examples"],
        "validation": (
            args.validation_examples or stage["validation_examples"]
        ),
        "test": args.test_examples or stage["test_examples"],
    }
    epochs = args.epochs
    if args.quick:
        sizes = {"train": 256, "validation": 128, "test": 256}
        epochs = min(epochs, 2)
        seeds = seeds[:1]

    device = _device_from_argument(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    task_config = DelayedRecallConfig()

    for model_name in models:
        for seed in seeds:
            print(
                f"Starting {model_name} seed={seed} on {device} "
                f"({sizes['train']}/{sizes['validation']}/{sizes['test']} examples)",
                flush=True,
            )
            result, _ = train_one_run(
                model_name,
                seed=seed,
                task_config=task_config,
                train_examples=sizes["train"],
                validation_examples=sizes["validation"],
                test_examples=sizes["test"],
                epochs=epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                device=device,
                progress=True,
            )
            result["created_at"] = datetime.now(timezone.utc).isoformat()
            result["code"] = _git_metadata(repository)
            result["runtime"] = {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": str(device),
                "gpu": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
            }
            output_path = args.output_dir / f"{model_name}_seed-{seed}.json"
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, allow_nan=False)
            print(
                f"{model_name} seed={seed}: "
                f"test_accuracy={result['test']['accuracy']:.4f}, "
                f"parameters={result['parameter_count']:,}, "
                f"saved={output_path}",
                flush=True,
            )


if __name__ == "__main__":
    main()
