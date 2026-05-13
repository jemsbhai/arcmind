"""
Train ArcMind on MuJoCo locomotion via behavior cloning.

Standard D4RL benchmark for offline RL. BC on locomotion tasks is
near-Markovian, so this primarily tests the SSM core and parameter
efficiency rather than episodic memory.

Usage:
    python examples/train_mujoco.py --task halfcheetah --quality medium --epochs 50
    python examples/train_mujoco.py --task halfcheetah --preset robotics_small
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

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


def build_config(preset: str, task: str) -> ArcMindConfig:
    """Build config from preset, overridden for MuJoCo task."""
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


def masked_mse_loss(predictions, targets, mask):
    """MSE loss over valid (non-padded) timesteps only."""
    mask_expanded = mask.unsqueeze(-1).expand_as(predictions)
    diff = (predictions - targets) ** 2
    diff = diff * mask_expanded
    return diff.sum() / mask_expanded.sum().clamp(min=1)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_steps = 0

    for batch in loader:
        obs = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        mask = batch["mask"].to(device)

        model.reset_memory(batch_size=obs.shape[0])
        pred_actions = model(obs)

        loss = masked_mse_loss(pred_actions, actions, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        n_valid = mask.sum().item()
        total_loss += loss.item() * n_valid
        total_steps += n_valid

    return {"loss": total_loss / max(total_steps, 1)}


@torch.no_grad()
def evaluate_mse(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_steps = 0

    for batch in loader:
        obs = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        mask = batch["mask"].to(device)

        model.reset_memory(batch_size=obs.shape[0])
        pred_actions = model(obs)

        loss = masked_mse_loss(pred_actions, actions, mask)

        n_valid = mask.sum().item()
        total_loss += loss.item() * n_valid
        total_steps += n_valid

    return {"loss": total_loss / max(total_steps, 1)}


@torch.no_grad()
def evaluate_online(model, env_name, norm_stats, device, num_episodes=10, max_steps=1000):
    """
    Roll out the learned policy in the MuJoCo environment using
    streaming (recurrent) inference — SSM state persists between steps.
    """
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
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1, obs_dim)
            obs_tensor = (obs_tensor - obs_mean) / obs_std

            action = model.step(obs_tensor)  # (1, act_dim)
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
    parser = argparse.ArgumentParser(description="Train ArcMind on MuJoCo locomotion BC")
    parser.add_argument("--task", type=str, default="halfcheetah",
                        choices=["halfcheetah", "hopper", "walker2d"])
    parser.add_argument("--quality", type=str, default="medium",
                        choices=["medium", "expert"])
    parser.add_argument("--preset", type=str, default="iot_tiny",
                        choices=["iot_tiny", "robotics_small", "robotics_medium"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--eval-episodes", type=int, default=10)
    # Ablation flags
    parser.add_argument("--ablate-ssm", action="store_true")
    parser.add_argument("--ablate-attention", action="store_true")
    parser.add_argument("--ablate-memory", action="store_true")
    parser.add_argument("--ablate-gating", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # Data
    dataset = MuJoCoLocomotionDataset(task=args.task, quality=args.quality)
    norm_stats = dataset.get_normalization_stats()
    env_name = dataset.get_env_name()

    # Train/val split
    n_episodes = len(dataset)
    indices = np.random.RandomState(args.seed).permutation(n_episodes)
    n_val = max(1, int(n_episodes * args.val_split))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"  Train episodes: {len(train_dataset)}")
    print(f"  Val episodes:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Model
    config = build_config(args.preset, args.task)
    config.ablate_ssm = args.ablate_ssm
    config.ablate_attention = args.ablate_attention
    config.ablate_memory = args.ablate_memory
    config.ablate_gating = args.ablate_gating

    model = ArcMindModel(config).to(device)
    param_counts = model.count_parameters()

    ablation_desc = []
    if args.ablate_ssm: ablation_desc.append("no_ssm")
    if args.ablate_attention: ablation_desc.append("no_attn")
    if args.ablate_memory: ablation_desc.append("no_mem")
    if args.ablate_gating: ablation_desc.append("no_gate")
    variant_name = f"{args.task}_{args.quality}_{args.preset}" + (
        "_" + "_".join(ablation_desc) if ablation_desc else "_full"
    )

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
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "total_params": param_counts["total"],
        "ablation_flags": ablation_desc,
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
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": vars(config),
                "hparams": hparams,
                "norm_stats": {k: v.tolist() for k, v in norm_stats.items()},
            }
            ckpt_path = save_dir / f"mujoco_{variant_name}.pt"
            torch.save(checkpoint, ckpt_path)

    # Final report
    print(f"\n{'=' * 50}")
    print(f"Training complete.")
    print(f"Best val MSE: {best_val_loss:.6f} (epoch {best_epoch})")

    # Online evaluation with best model
    print(f"\nRunning online evaluation ({args.eval_episodes} episodes in {env_name})...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    online_metrics = evaluate_online(
        model, env_name, norm_stats, device,
        num_episodes=args.eval_episodes,
    )

    print(f"\nOnline evaluation results:")
    print(f"  Avg return: {online_metrics['avg_return']:.1f} ± {online_metrics.get('std_return', 0):.1f}")
    print(f"  Min/Max:    {online_metrics.get('min_return', 0):.1f} / {online_metrics.get('max_return', 0):.1f}")

    # Save results
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
