"""
Train ArcMind on UCI HAR (Human Activity Recognition).

Classifies 6 activities from raw 6-axis IMU data (50 Hz, 128-sample windows).
Validates the sensor-native tokenization thesis: no vocabulary table, no
pre-computed features — raw sensor frames projected directly into model space.

Usage:
    python examples/train_uci_har.py
    python examples/train_uci_har.py --preset iot_tiny --epochs 50
    python examples/train_uci_har.py --preset robotics_small --lr 1e-3
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from arcmind import ArcMindConfig, ArcMindModel
from arcmind.data.uci_har import UCIHARDataset, ACTIVITY_LABELS


def set_seed(seed: int) -> None:
    """Pin all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_config(preset: str) -> ArcMindConfig:
    """Build config from preset name, overriding for UCI HAR."""
    presets = {
        "iot_tiny": ArcMindConfig.iot_tiny,
        "robotics_small": ArcMindConfig.robotics_small,
        "robotics_medium": ArcMindConfig.robotics_medium,
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")

    config = presets[preset]()

    # Override for UCI HAR dataset
    config.num_sensor_channels = UCIHARDataset.NUM_CHANNELS  # 6
    config.sensor_freq_hz = UCIHARDataset.SAMPLE_RATE_HZ      # 50.0
    config.action_dim = UCIHARDataset.NUM_CLASSES              # 6

    return config


def train_one_epoch(
    model: ArcMindModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for sensor_window, labels in loader:
        sensor_window = sensor_window.to(device)
        labels = labels.to(device)

        model.reset_memory(batch_size=sensor_window.shape[0])
        logits = model(sensor_window, use_memory=True)  # (batch, 128, 6)

        # Mean-pool over time dimension for classification
        logits_pooled = logits.mean(dim=1)  # (batch, 6)

        loss = criterion(logits_pooled, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits_pooled.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def evaluate(
    model: ArcMindModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
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
        logits = model(sensor_window, use_memory=True)
        logits_pooled = logits.mean(dim=1)

        loss = criterion(logits_pooled, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits_pooled.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Per-class accuracy
    per_class = {}
    for cls_idx, cls_name in ACTIVITY_LABELS.items():
        cls_label = cls_idx - 1  # 0-indexed
        cls_mask = [i for i, l in enumerate(all_labels) if l == cls_label]
        if cls_mask:
            cls_correct = sum(1 for i in cls_mask if all_preds[i] == cls_label)
            per_class[cls_name] = cls_correct / len(cls_mask)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser(description="Train ArcMind on UCI HAR")
    parser.add_argument("--preset", type=str, default="iot_tiny", choices=["iot_tiny", "robotics_small", "robotics_medium"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    # Reproducibility
    set_seed(args.seed)
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

    # Model
    config = build_config(args.preset)
    model = ArcMindModel(config).to(device)
    param_counts = model.count_parameters()
    print(f"\nModel: ArcMind ({args.preset})")
    for component, count in param_counts.items():
        print(f"  {component:20s}: {count:>10,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss()

    # Checkpointing
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = 0.0
    best_epoch = 0

    # Log hyperparameters
    hparams = {
        "preset": args.preset,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "total_params": param_counts["total"],
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
    }
    hparams_path = save_dir / f"hparams_{args.preset}.json"
    with open(hparams_path, "w") as f:
        json.dump(hparams, f, indent=2)
    print(f"\nHyperparameters saved to {hparams_path}")

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"{'Epoch':>5s}  {'Train Loss':>10s}  {'Train Acc':>9s}  {'Test Loss':>9s}  {'Test Acc':>8s}  {'LR':>10s}  {'Time':>6s}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:5d}  "
            f"{train_metrics['loss']:10.4f}  "
            f"{train_metrics['accuracy']:9.4f}  "
            f"{test_metrics['loss']:9.4f}  "
            f"{test_metrics['accuracy']:8.4f}  "
            f"{lr:10.2e}  "
            f"{elapsed:5.1f}s"
        )

        # Save best model
        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy = test_metrics["accuracy"]
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_accuracy": best_accuracy,
                "config": vars(config),
                "hparams": hparams,
            }
            ckpt_path = save_dir / f"best_{args.preset}.pt"
            torch.save(checkpoint, ckpt_path)

    # Final report
    print(f"\n{'=' * 70}")
    print(f"Training complete.")
    print(f"Best test accuracy: {best_accuracy:.4f} (epoch {best_epoch})")
    print(f"Checkpoint saved:   {ckpt_path}")

    # Load best model and report per-class accuracy
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\nPer-class accuracy (best model):")
    for cls_name, acc in final_metrics["per_class"].items():
        print(f"  {cls_name:25s}: {acc:.4f}")
    print(f"  {'OVERALL':25s}: {final_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
