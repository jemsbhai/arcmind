"""
Baseline models for UCI HAR comparison.

Trains MLP, LSTM, and Transformer baselines on the same raw sensor data
with matched preprocessing (no pre-computed features) for fair comparison
against ArcMind.

Usage:
    python examples/train_baselines.py --model mlp --epochs 30
    python examples/train_baselines.py --model lstm --epochs 30
    python examples/train_baselines.py --model transformer --epochs 30
    python examples/train_baselines.py --model all --epochs 30
"""

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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


# ============================================================
# Baseline architectures
# ============================================================


class MLPBaseline(nn.Module):
    """
    Simple MLP that flattens the sensor window.
    Common baseline for HAR — ignores temporal structure entirely.
    """

    def __init__(self, input_dim: int = 128 * 6, hidden_dim: int = 256, num_classes: int = 6):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.flatten(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LSTMBaseline(nn.Module):
    """
    2-layer bidirectional LSTM. Standard temporal baseline for HAR.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)  # (batch, 128, hidden*2)
        # Mean pool over time
        pooled = output.mean(dim=1)  # (batch, hidden*2)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TransformerBaseline(nn.Module):
    """
    Small Transformer encoder for sensor classification.
    Uses the same linear projection as ArcMind (no embedding table)
    for fair comparison of the attention mechanism itself.
    """

    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        num_classes: int = 6,
        dropout: float = 0.1,
        max_len: int = 128,
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.norm_in = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm_out = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x) + self.pos_embedding[:, : x.size(1), :]
        x = self.norm_in(x)
        x = self.encoder(x)
        x = self.norm_out(x.mean(dim=1))  # Mean pool
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================
# Training utilities (shared with train_uci_har.py)
# ============================================================


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for sensor_window, labels in loader:
        sensor_window = sensor_window.to(device)
        labels = labels.to(device)

        logits = model(sensor_window)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
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

        logits = model(sensor_window)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
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


def train_model(model_name, model, train_loader, test_loader, args, device):
    """Train a single model and return results."""
    n_params = model.count_parameters()
    print(f"\n{'=' * 70}")
    print(f"  {model_name}  —  {n_params:,} parameters")
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

        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy = test_metrics["accuracy"]
            best_epoch = epoch
            best_per_class = test_metrics["per_class"]

    print(f"\nBest test accuracy: {best_accuracy:.4f} (epoch {best_epoch})")
    print(f"Per-class accuracy:")
    for cls_name, acc in best_per_class.items():
        print(f"  {cls_name:25s}: {acc:.4f}")

    return {
        "model": model_name,
        "params": n_params,
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "per_class": best_per_class,
    }


def main():
    parser = argparse.ArgumentParser(description="Train baselines on UCI HAR")
    parser.add_argument("--model", type=str, default="all", choices=["mlp", "lstm", "transformer", "all"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

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

    # Build models
    models = {}
    if args.model in ("mlp", "all"):
        models["MLP (256h, 2-layer)"] = MLPBaseline(hidden_dim=256).to(device)

    if args.model in ("lstm", "all"):
        models["LSTM (128h, 2-layer BiDir)"] = LSTMBaseline(hidden_dim=128).to(device)

    if args.model in ("transformer", "all"):
        models["Transformer (d64, 4-layer)"] = TransformerBaseline(d_model=64, num_layers=4).to(device)

    # Train each
    all_results = []
    for model_name, model in models.items():
        set_seed(args.seed)  # Reset seed for each model
        result = train_model(model_name, model, train_loader, test_loader, args, device)
        all_results.append(result)

    # Summary table
    print(f"\n\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Model':40s}  {'Params':>10s}  {'Best Acc':>8s}")
    print("-" * 62)

    # Add ArcMind results if checkpoint exists
    for preset in ["iot_tiny", "robotics_small", "robotics_medium"]:
        ckpt = Path(args.save_dir) / f"best_{preset}.pt"
        if ckpt.exists():
            data = torch.load(ckpt, map_location="cpu", weights_only=False)
            all_results.append({
                "model": f"ArcMind ({preset})",
                "params": data.get("hparams", {}).get("total_params", 0),
                "best_accuracy": data["test_accuracy"],
            })

    for r in sorted(all_results, key=lambda x: x["params"]):
        print(f"  {r['model']:38s}  {r['params']:>10,}  {r['best_accuracy']:8.4f}")

    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_path = save_dir / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
