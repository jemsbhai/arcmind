"""
Train ArcMind on D4RL AntMaze via behavior cloning.

Predicts expert actions from observations over 700-step episodes.
This is the benchmark where episodic memory should matter — the ant
must navigate a maze, and remembering visited corridors aids planning.

Evaluation: action MSE on held-out episodes + online success rate in
the environment (roll out the learned policy and check goal reaching).

Usage:
    python examples/train_antmaze.py --variant umaze --epochs 50
    python examples/train_antmaze.py --variant umaze --preset robotics_small
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from arcmind import ArcMindConfig, ArcMindModel
from arcmind.data.antmaze import ACTION_DIM, NUM_OBS_CHANNELS, AntMazeDataset


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


def build_config(preset: str) -> ArcMindConfig:
    """Build config from preset, overridden for AntMaze."""
    presets = {
        "iot_tiny": ArcMindConfig.iot_tiny,
        "robotics_small": ArcMindConfig.robotics_small,
        "robotics_medium": ArcMindConfig.robotics_medium,
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}")

    config = presets[preset]()
    config.num_sensor_channels = NUM_OBS_CHANNELS  # 31
    config.action_dim = ACTION_DIM                  # 8
    config.sensor_freq_hz = 50.0                    # MuJoCo default ~50 Hz
    config.decision_freq_hz = 10.0
    return config


def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    MSE loss computed only over valid (non-padded) timesteps.

    Args:
        predictions: (batch, seq_len, action_dim)
        targets: (batch, seq_len, action_dim)
        mask: (batch, seq_len) bool
    """
    mask_expanded = mask.unsqueeze(-1).expand_as(predictions)  # (batch, seq_len, action_dim)
    diff = (predictions - targets) ** 2
    diff = diff * mask_expanded
    # Mean over valid entries only
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
def evaluate_online(model, config, norm_stats, device, num_episodes=20, max_steps=700):
    """
    Roll out the learned policy in the AntMaze environment.
    Returns success rate (fraction of episodes reaching the goal).
    """
    try:
        import gymnasium
        import gymnasium_robotics  # noqa: F401 — registers AntMaze envs
    except ImportError:
        print("  gymnasium/gymnasium-robotics not available, skipping online eval")
        return {"success_rate": float("nan"), "avg_return": float("nan")}

    try:
        env = gymnasium.make("AntMaze_UMaze-v4", max_episode_steps=max_steps)
    except Exception as e:
        print(f"  Could not create AntMaze env: {e}")
        return {"success_rate": float("nan"), "avg_return": float("nan")}

    model.eval()
    obs_mean = torch.tensor(norm_stats["obs_mean"], dtype=torch.float32, device=device)
    obs_std = torch.tensor(norm_stats["obs_std"], dtype=torch.float32, device=device)

    successes = 0
    total_returns = []

    for ep in range(num_episodes):
        obs_dict, _ = env.reset()
        model.init_streaming(batch_size=1)
        episode_return = 0.0
        done = False

        for step in range(max_steps):
            # Flatten observation dict
            obs_flat = np.concatenate([
                obs_dict["observation"],
                obs_dict["achieved_goal"],
                obs_dict["desired_goal"],
            ]).astype(np.float32)

            obs_tensor = torch.tensor(obs_flat, device=device).unsqueeze(0)
            obs_tensor = (obs_tensor - obs_mean) / obs_std

            with torch.inference_mode():
                pred = model.step(obs_tensor)
            action = pred.squeeze(0).cpu().numpy()
            action = np.clip(action, -1.0, 1.0)

            obs_dict, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            done = terminated or truncated

            if done:
                break

        if info.get("success", False) or episode_return > 0:
            successes += 1
        total_returns.append(episode_return)

    env.close()

    return {
        "success_rate": successes / num_episodes,
        "avg_return": np.mean(total_returns),
    }


def main():
    parser = argparse.ArgumentParser(description="Train ArcMind on D4RL AntMaze")
    parser.add_argument(
        "--variant",
        type=str,
        default="umaze",
        choices=["umaze", "medium-play", "large-diverse"],
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="iot_tiny",
        choices=["iot_tiny", "robotics_small", "robotics_medium"],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--eval-episodes", type=int, default=20)
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
    dataset = AntMazeDataset(variant=args.variant)
    norm_stats = dataset.get_normalization_stats()

    # Train/val split by episode
    n_episodes = len(dataset)
    indices = np.random.RandomState(args.seed).permutation(n_episodes)
    n_val = int(n_episodes * args.val_split)
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
    config = build_config(args.preset)
    config.ablate_ssm = args.ablate_ssm
    config.ablate_attention = args.ablate_attention
    config.ablate_memory = args.ablate_memory
    config.ablate_gating = args.ablate_gating
    model = ArcMindModel(config).to(device)

    param_counts = model.count_parameters()
    ablation_desc = []
    if args.ablate_ssm:
        ablation_desc.append("no_ssm")
    if args.ablate_attention:
        ablation_desc.append("no_attn")
    if args.ablate_memory:
        ablation_desc.append("no_mem")
    if args.ablate_gating:
        ablation_desc.append("no_gate")
    variant_name = args.preset + ("_" + "_".join(ablation_desc) if ablation_desc else "_full")

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

    # Log hyperparameters
    hparams = {
        "variant": args.variant,
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
            ckpt_path = save_dir / f"antmaze_{variant_name}.pt"
            torch.save(checkpoint, ckpt_path)

    # Final report
    print(f"\n{'=' * 50}")
    print("Training complete.")
    print(f"Best val MSE: {best_val_loss:.6f} (epoch {best_epoch})")

    # Load best model for online evaluation
    print(f"\nRunning online evaluation ({args.eval_episodes} episodes)...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    online_metrics = evaluate_online(
        model, config, norm_stats, device,
        num_episodes=args.eval_episodes,
    )

    print(f"  Success rate: {online_metrics['success_rate']:.2%}")
    print(f"  Avg return:   {online_metrics['avg_return']:.2f}")

    # Save final results
    results = {
        "variant_name": variant_name,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "success_rate": online_metrics["success_rate"],
        "avg_return": online_metrics["avg_return"],
        "params": param_counts["total"],
        "hparams": hparams,
    }
    results_path = save_dir / f"antmaze_{variant_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
