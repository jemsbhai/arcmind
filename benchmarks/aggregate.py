"""Aggregate raw delayed-recall results without discarding per-seed records."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from benchmarks.statistics import summarize


def aggregate_results(paths: list[Path]) -> dict:
    """Aggregate result files by model and test metric."""
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("schema_version") != 1:
            raise ValueError(f"unsupported result schema in {path}")
        records.append(record)

    grouped = defaultdict(list)
    for record in records:
        grouped[record["model"]].append(record)

    models = {}
    for model, model_records in sorted(grouped.items()):
        metrics = {}
        for metric in (
            "accuracy",
            "nll",
            "short_lag_accuracy",
            "long_lag_accuracy",
            "examples_per_second",
        ):
            values = [float(record["test"][metric]) for record in model_records]
            metrics[metric] = summarize(values, seed=2026)
        models[model] = {
            "metrics": metrics,
            "seeds": sorted(record["seed"] for record in model_records),
            "parameter_count": model_records[0]["parameter_count"],
            "parameter_ratio": model_records[0]["parameter_ratio"],
            "raw_files": [
                str(path)
                for path, record in zip(paths, records)
                if record["model"] == model
            ],
        }

    return {
        "schema_version": 1,
        "benchmark": "delayed_sensor_recall",
        "num_raw_records": len(records),
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = sorted(args.result_dir.glob("*_seed-*.json"))
    if not paths:
        raise SystemExit(f"no raw result files found in {args.result_dir}")
    summary = aggregate_results(paths)
    output = args.output or args.result_dir / "summary.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(f"Aggregated {len(paths)} raw records into {output}")


if __name__ == "__main__":
    main()
