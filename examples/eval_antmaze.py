"""Quick eval of saved AntMaze checkpoint."""

import gymnasium
import gymnasium_robotics  # noqa: F401 — registers AntMaze envs
import numpy as np
import torch

from arcmind import ArcMindConfig, ArcMindModel


def evaluate_online(model, norm_stats, device, num_episodes=20, max_steps=700):
    env = gymnasium.make("AntMaze_UMaze-v4", max_episode_steps=max_steps)

    obs_mean = torch.tensor(norm_stats["obs_mean"], dtype=torch.float32, device=device)
    obs_std = torch.tensor(norm_stats["obs_std"], dtype=torch.float32, device=device)

    model.eval()
    successes = 0
    total_returns = []

    for ep in range(num_episodes):
        obs_dict, _ = env.reset()
        model.init_streaming(batch_size=1)
        episode_return = 0.0

        for step in range(max_steps):
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

            if terminated or truncated:
                break

        if info.get("success", False) or episode_return > 0:
            successes += 1
        total_returns.append(episode_return)
        print(f"  Episode {ep + 1:2d}: return={episode_return:.1f}, steps={step + 1}")

    env.close()
    return successes / num_episodes, np.mean(total_returns)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(
        "checkpoints/antmaze_iot_tiny_full.pt",
        map_location=device,
        weights_only=False,
    )

    config = ArcMindConfig(**ckpt["config"])
    model = ArcMindModel(config).to(device)
    # strict=False: older checkpoints may contain ephemeral memory.buffer/write_ptr
    # which are now non-persistent. Safe to ignore.
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.reset_memory(batch_size=1)

    norm_stats = {k: np.array(v) for k, v in ckpt["norm_stats"].items()}

    print(f"Evaluating ArcMind ({ckpt['hparams']['variant_name']}) on AntMaze_UMaze-v4...")
    print(f"Parameters: {ckpt['hparams']['total_params']:,}")
    print(f"Best val MSE: {ckpt['val_loss']:.6f} (epoch {ckpt['epoch']})\n")

    success_rate, avg_return = evaluate_online(model, norm_stats, device, num_episodes=20)

    print(f"\nSuccess rate: {success_rate:.2%}")
    print(f"Avg return:   {avg_return:.2f}")
