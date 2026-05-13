"""
Train ArcMind on MuJoCo locomotion via behavior cloning (short windows).

Uses overlapping short windows instead of full episodes to avoid
teacher-forcing compounding error at evaluation time.

Usage:
    python examples/train_mujoco_windowed.py --task halfcheetah --window-size 50 --epochs 30
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from arcmind import ArcMindConfig, ArcMindModel
from arcmind.data.mujoco_locomotion import MuJoCoLocomotionDataset, MUJOCO_DIMS


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class WindowedMuJoCoDataset(Dataset):
    """
    Extracts overlapping short windows from MuJoCo locomotion episodes.

    Each sample is a (obs_window, action_window) pair of fixed length.
    No padding needed — all windows are the same size.
    """

    def __init__(
        self,
        base_dataset: MuJoCoLocomotionDataset,
        episode_indices: np.ndarray | None = None,
        window_size: int = 50,
        stride: int = 25,
    ):
        self.window_size = window_size
        self.obs_dim = base_dataset.obs_dim
        self.act_dim = base_dataset.act_dim

        if episode_indices is None:
            episode_indices = range(len(base_dataset))

        # Extract all windows
        self.obs_windows = []
        self.act_windows = []

        for idx in episode_indices:
            sample = base_dataset[idx]
            T = sample["length"]
            obs = sample["observations"][:T].numpy()
            acts = sample["actions"][:T].numpy()

            for start in range(0, T - window_size + 1, stride):
                end = start + window_size
                self.obs_windows.append(obs[start:end])
                self.act_windows.append(acts[start:end])

        self.obs_windows = np.array(self.obs_windows, dtype=np.float32)
        self.act_windows = np.array(self.act_windows, dtype=np.float32)

        print(f"  Windowed dataset: {len(self)} windows "
              f"(window={window_size}, stride={stride}, from {len(episode_indices)} episodes)")

    def __len__(self) -> int:
        return len(self.obs_windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.obs_windows[idx]),
            torch.from_numpy(self.act_windows[idx]),
        )


def build_config(preset: str, task: str) -> ArcMindConfig:
    presets = {
        "iot_tiny": ArcMindConfig.iot_tiny,
        "robotics_small": ArcMindConfig.robotics_small,
        "robotics_medium": ArcMindConfig.robotics_medium,
    }
    config = presets[preset]()
    dims = MUJOCO_DIMS[task]
    config.num_sensor_channels = dims["obs"]
    config.action_dim = dims["act"]
    config.sensor_freq_hz = 50.0
    config.decision_freq_hz = 10.0
    return config


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for obs_window, act_window in loader:
        obs_window = obs_window.to(device)
        act_window = act_window.to(device)

        model.reset_memory(batch_size=obs_window.shape[0])
        pred_actions = model(obs_window)

        loss = nn.functional.mse_loss(pred_actions, act_window)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * obs_window.shape[0]
        total_samples += obs_window.shape[0]

    return {"loss": total_loss / total_samples}


@torch.no_grad()
def evaluate_mse(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for obs_window, act_window in loader:
        obs_window = obs_window.to(device)
        act_window = act_window.to(device)

        model.reset_memory(batch_size=obs_window.shape[0])
        pred_actions = model(obs_window)

        loss = nn.functional.mse_loss(pred_actions, act_window)

        total_loss += loss.item() * obs_window.shape[0]
        total_samples += obs_window.shape[0]

    return {"loss": total_loss / total_samples}


@torch.no_grad()
def evaluate_online(model, env_name, norm_stats, device, num_episodes=10, max_steps=1000):
    """Roll out using streaming inference."""
    try:
        import gymnasium
    except ImportError:
        print("  gymnasium not available, skipping online eval")
        return {"avg_return": float("nan")}

    try:
        env = gymnasium.make(env_name, max_episode_steps=max_steps)
    except Exception as e:
        print(f"  Could not create env {env_name}: {e}")
        return {"avg_return": float("nan")}

    obs_mean = torch.tensor(norm_stats["obs_mean"], dtype=torch.float32, device=device)
    obs_std = torch.tensor(norm_stats["obs_std"], dtype=torch.float32, device=device)

    model.eval()
    total_returns = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        model.init_streaming(batch_size=1)
        episode_return = 0.0

        for step in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            obs_tensor = (obs_tensor - obs_mean) / obs_std

            action = model.step(obs_tensor)
            action = action.squeeze(0).cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward

            if terminated or truncated:
                break

        total_returns.append(episode_return)
        print(f"  Episode {ep + 1:2d}: return={episode_return:.1f}, steps={step + 1}")

    env.close()

    return {
        "avg_return": float(np.mean(total_returns)),
        "std_return": float(np.std(total_returns)),
        "min_return": float(np.min(total_returns)),
        "max_return": float(np.max(total_returns)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train ArcMind on MuJoCo BC (windowed)")
    parser.add_argument("--task", type=str, default="halfcheetah",
                        choices=["halfcheetah", "hopper", "walker2d"])
    parser.add_argument("--quality", type=str, default="medium",
                        choices=["medium", "expert"])
    parser.add_argument("--preset", type=str, default="iot_tiny",
                        choices=["iot_tiny", "robotics_small", "robotics_medium"])
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--window-stride", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # Load base dataset
    base_dataset = MuJoCoLocomotionDataset(task=args.task, quality=args.quality)
    norm_stats = base_dataset.get_normalization_stats()
    env_name = base_dataset.get_env_name()

    # Split episodes first, then window
    n_episodes = len(base_dataset)
    indices = np.random.RandomState(args.seed).permutation(n_episodes)
    n_val = max(1, int(n_episodes * 0.1))

    # Window each split
    print("\nCreating training windows...")
    train_windows = WindowedMuJoCoDataset(
        base_dataset, indices[n_val:], args.window_size, args.window_stride
    )
    print("Creating validation windows...")
    val_windows = WindowedMuJoCoDataset(
        base_dataset, indices[:n_val], args.window_size, args.window_stride
    )

    train_loader = DataLoader(
        train_windows, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_windows, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Model
    config = build_config(args.preset, args.task)
    model = ArcMindModel(config).to(device)
    param_counts = model.count_parameters()

    variant_name = f"{args.task}_{args.quality}_{args.preset}_w{args.window_size}"

    print(f"\nModel: ArcMind ({variant_name})")
    for component, count in param_counts.items():
        print(f"  {component:20s}: {count:>10,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # Checkpointing
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0

    hparams = {
        "task": args.task,
        "quality": args.quality,
        "preset": args.preset,
        "variant_name": variant_name,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "total_params": param_counts["total"],
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
    }

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"{'Epoch':>5s}  {'Train MSE':>10s}  {'Val MSE':>10s}  {'LR':>10s}  {'Time':>6s}")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_mse(model, val_loader, device)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:5d}  "
            f"{train_metrics['loss']:10.6f}  "
            f"{val_metrics['loss']:10.6f}  "
            f"{lr:10.2e}  "
            f"{elapsed:5.1f}s"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "config": vars(config),
                "hparams": hparams,
                "norm_stats": {k: v.tolist() for k, v in norm_stats.items()},
            }
            ckpt_path = save_dir / f"mujoco_{variant_name}.pt"
            torch.save(checkpoint, ckpt_path)

    print(f"\n{'=' * 50}")
    print(f"Training complete.")
    print(f"Best val MSE: {best_val_loss:.6f} (epoch {best_epoch})")

    # Online eval
    print(f"\nRunning online evaluation ({args.eval_episodes} episodes in {env_name})...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    online_metrics = evaluate_online(
        model, env_name, norm_stats, device,
        num_episodes=args.eval_episodes,
    )

    print(f"\nOnline evaluation results:")
    print(f"  Avg return: {online_metrics['avg_return']:.1f} +/- {online_metrics.get('std_return', 0):.1f}")
    print(f"  Min/Max:    {online_metrics.get('min_return', 0):.1f} / {online_metrics.get('max_return', 0):.1f}")

    results = {
        "variant_name": variant_name,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "online": online_metrics,
        "params": param_counts["total"],
        "hparams": hparams,
    }
    results_path = save_dir / f"mujoco_{variant_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
