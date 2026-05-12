"""
Ablation study on UCI HAR.

Trains ArcMind with each component systematically removed to isolate
the contribution of the SSM core, attention, episodic memory, and
learned gating.

Usage:
    python examples/run_ablations.py --epochs 30
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from arcmind import ArcMindConfig, ArcMindModel
from arcmind.data.uci_har import UCIHARDataset, ACTIVITY_LABELS


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_har_config(**ablation_flags) -> ArcMindConfig:
    """Create iot_tiny config overridden for UCI HAR with optional ablation flags."""
    config = ArcMindConfig.iot_tiny()
    config.num_sensor_channels = 6
    config.sensor_freq_hz = 50.0
    config.action_dim = 6
    for k, v in ablation_flags.items():
        setattr(config, k, v)
    return config


ABLATION_VARIANTS = {
    "full": {},
    "no_memory": {"ablate_memory": True},
    "no_attention": {"ablate_attention": True},
    "no_ssm": {"ablate_ssm": True},
    "no_gating": {"ablate_gating": True},
}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for sensor_window, labels in loader:
        sensor_window = sensor_window.to(device)
        labels = labels.to(device)

        model.reset_memory(batch_size=sensor_window.shape[0])
        logits = model(sensor_window)
        logits_pooled = logits.mean(dim=1)

        loss = criterion(logits_pooled, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits_pooled.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for sensor_window, labels in loader:
        sensor_window = sensor_window.to(device)
        labels = labels.to(device)

        model.reset_memory(batch_size=sensor_window.shape[0])
        logits = model(sensor_window)
        logits_pooled = logits.mean(dim=1)

        loss = criterion(logits_pooled, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits_pooled.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    per_class = {}
    for cls_idx, cls_name in ACTIVITY_LABELS.items():
        cls_label = cls_idx - 1
        cls_mask = [i for i, l in enumerate(all_labels) if l == cls_label]
        if cls_mask:
            cls_correct = sum(1 for i in cls_mask if all_preds[i] == cls_label)
            per_class[cls_name] = cls_correct / len(cls_mask)

    return {"loss": total_loss / total, "accuracy": correct / total, "per_class": per_class}


def run_ablation(
    variant_name: str,
    ablation_flags: dict,
    train_loader: DataLoader,
    test_loader: DataLoader,
    args,
    device: torch.device,
) -> dict:
    """Train one ablation variant and return results."""
    set_seed(args.seed)

    config = make_har_config(**ablation_flags)
    model = ArcMindModel(config).to(device)
    n_params = model.count_parameters()["total"]

    # Describe what's ablated
    if ablation_flags:
        desc = ", ".join(f"{k}={v}" for k, v in ablation_flags.items())
    else:
        desc = "all components active"

    print(f"\n{'=' * 70}")
    print(f"  Ablation: {variant_name}")
    print(f"  Config:   {desc}")
    print(f"  Params:   {n_params:,}")
    print(f"{'=' * 70}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    best_epoch = 0
    best_per_class = {}
    train_history = []

    print(f"\n{'Epoch':>5s}  {'Train Loss':>10s}  {'Train Acc':>9s}  {'Test Loss':>9s}  {'Test Acc':>8s}  {'Time':>6s}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.perf_counter() - t0

        print(
            f"{epoch:5d}  "
            f"{train_metrics['loss']:10.4f}  "
            f"{train_metrics['accuracy']:9.4f}  "
            f"{test_metrics['loss']:9.4f}  "
            f"{test_metrics['accuracy']:8.4f}  "
            f"{elapsed:5.1f}s"
        )

        train_history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["accuracy"],
        })

        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy = test_metrics["accuracy"]
            best_epoch = epoch
            best_per_class = test_metrics["per_class"]

    print(f"\nBest: {best_accuracy:.4f} (epoch {best_epoch})")

    return {
        "variant": variant_name,
        "ablation_flags": ablation_flags,
        "params": n_params,
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "per_class": best_per_class,
        "history": train_history,
    }


def main():
    parser = argparse.ArgumentParser(description="ArcMind ablation study on UCI HAR")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--variants", type=str, nargs="+", default=list(ABLATION_VARIANTS.keys()),
                        choices=list(ABLATION_VARIANTS.keys()))
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Data
    print("\nLoading UCI HAR dataset...")
    train_dataset = UCIHARDataset(data_dir=args.data_dir, split="train")
    test_dataset = UCIHARDataset(data_dir=args.data_dir, split="test")
    print(f"  Train: {train_dataset}")
    print(f"  Test:  {test_dataset}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Run ablations
    all_results = []
    for variant_name in args.variants:
        flags = ABLATION_VARIANTS[variant_name]
        result = run_ablation(variant_name, flags, train_loader, test_loader, args, device)
        all_results.append(result)

    # Summary table
    print(f"\n\n{'=' * 70}")
    print(f"  ABLATION STUDY SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Variant':20s}  {'Params':>10s}  {'Best Acc':>8s}  {'Δ vs Full':>9s}")
    print("-" * 52)

    full_acc = None
    for r in all_results:
        if r["variant"] == "full":
            full_acc = r["best_accuracy"]

    for r in all_results:
        delta = ""
        if full_acc is not None and r["variant"] != "full":
            diff = r["best_accuracy"] - full_acc
            delta = f"{diff:+.4f}"
        print(
            f"  {r['variant']:18s}  "
            f"{r['params']:>10,}  "
            f"{r['best_accuracy']:8.4f}  "
            f"{delta:>9s}"
        )

    # Per-class breakdown
    print(f"\n{'':20s}", end="")
    for cls_name in ACTIVITY_LABELS.values():
        print(f"  {cls_name[:6]:>6s}", end="")
    print()
    print("-" * (20 + 8 * len(ACTIVITY_LABELS)))

    for r in all_results:
        print(f"  {r['variant']:18s}", end="")
        for cls_name in ACTIVITY_LABELS.values():
            acc = r["per_class"].get(cls_name, 0.0)
            print(f"  {acc:6.3f}", end="")
        print()

    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Strip history for the summary file (keep a separate file for curves)
    summary = [{k: v for k, v in r.items() if k != "history"} for r in all_results]
    with open(save_dir / "ablation_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(save_dir / "ablation_history.json", "w") as f:
        json.dump([{"variant": r["variant"], "history": r["history"]} for r in all_results], f, indent=2)

    print(f"\nResults saved to {save_dir / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
