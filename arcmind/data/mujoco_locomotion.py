"""
MuJoCo Locomotion dataset loader via Minari.

Loads offline RL datasets for behavior cloning on standard D4RL
locomotion tasks. Observations are plain state vectors (joint angles,
velocities), making this a direct sensor-data benchmark.

Tasks:
- HalfCheetah: 17-dim obs, 6-dim action, 1000-step episodes
- Hopper:      11-dim obs, 3-dim action, 1000-step episodes
- Walker2d:    17-dim obs, 6-dim action, 1000-step episodes

Dataset qualities: expert, medium, simple
"""

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import minari
except ImportError:
    minari = None


MUJOCO_DATASETS = {
    "halfcheetah-medium": "mujoco/halfcheetah/medium-v0",
    "halfcheetah-expert": "mujoco/halfcheetah/expert-v0",
    "hopper-medium": "mujoco/hopper/medium-v0",
    "hopper-expert": "mujoco/hopper/expert-v0",
    "walker2d-medium": "mujoco/walker2d/medium-v0",
    "walker2d-expert": "mujoco/walker2d/expert-v0",
}

# Gymnasium env names for online evaluation
MUJOCO_ENV_NAMES = {
    "halfcheetah": "HalfCheetah-v4",
    "hopper": "Hopper-v4",
    "walker2d": "Walker2d-v4",
}

# Obs/action dims per task (for validation)
MUJOCO_DIMS = {
    "halfcheetah": {"obs": 17, "act": 6},
    "hopper": {"obs": 11, "act": 3},
    "walker2d": {"obs": 17, "act": 6},
}


class MuJoCoLocomotionDataset(Dataset):
    """
    PyTorch Dataset for MuJoCo locomotion BC.

    Each sample is one full episode, zero-padded to max_episode_len.
    Observations are plain state vectors (not dicts like AntMaze).

    Args:
        task: One of 'halfcheetah', 'hopper', 'walker2d'.
        quality: One of 'medium', 'expert'.
        max_episode_len: Pad/truncate episodes to this length.
        normalize: Standardize observations using dataset statistics.
    """

    def __init__(
        self,
        task: str = "halfcheetah",
        quality: str = "medium",
        max_episode_len: int = 1000,
        normalize: bool = True,
    ):
        if minari is None:
            raise ImportError("minari is required: pip install minari")

        key = f"{task}-{quality}"
        assert key in MUJOCO_DATASETS, (
            f"Unknown task/quality: {key}. Choose from {list(MUJOCO_DATASETS.keys())}"
        )

        self.task = task
        self.quality = quality
        self.max_episode_len = max_episode_len
        self.obs_dim = MUJOCO_DIMS[task]["obs"]
        self.act_dim = MUJOCO_DIMS[task]["act"]

        dataset_id = MUJOCO_DATASETS[key]
        print(f"Loading {dataset_id}...")
        dataset = minari.load_dataset(dataset_id, download=True)

        self.observations = []
        self.actions = []
        self.rewards = []
        self.episode_lengths = []

        for ep in dataset.iterate_episodes():
            obs = ep.observations.astype(np.float32)
            acts = ep.actions.astype(np.float32)
            rews = ep.rewards.astype(np.float32)

            # obs has T+1 entries, actions/rewards have T
            obs = obs[:-1]

            T = min(len(acts), max_episode_len)
            self.observations.append(obs[:T])
            self.actions.append(acts[:T])
            self.rewards.append(rews[:T])
            self.episode_lengths.append(T)

        # Normalization
        if normalize:
            all_obs = np.concatenate(self.observations, axis=0)
            self._obs_mean = all_obs.mean(axis=0)
            self._obs_std = all_obs.std(axis=0)
            self._obs_std[self._obs_std < 1e-8] = 1.0

            self.observations = [
                (obs - self._obs_mean) / self._obs_std for obs in self.observations
            ]
        else:
            self._obs_mean = np.zeros(self.obs_dim)
            self._obs_std = np.ones(self.obs_dim)

        self.total_episodes = len(self.observations)
        self.total_steps = sum(self.episode_lengths)

        print(f"Loaded {self.total_episodes} episodes, {self.total_steps} total steps")
        print(f"Obs dim: {self.obs_dim}, Action dim: {self.act_dim}")

    def __len__(self) -> int:
        return self.total_episodes

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with:
                'observations': (max_episode_len, obs_dim) float32, zero-padded
                'actions': (max_episode_len, act_dim) float32, zero-padded
                'rewards': (max_episode_len,) float32, zero-padded
                'mask': (max_episode_len,) bool, True for valid timesteps
                'length': int, actual episode length
        """
        obs = self.observations[idx]
        acts = self.actions[idx]
        rews = self.rewards[idx]
        T = self.episode_lengths[idx]

        obs_padded = np.zeros((self.max_episode_len, self.obs_dim), dtype=np.float32)
        act_padded = np.zeros((self.max_episode_len, self.act_dim), dtype=np.float32)
        rew_padded = np.zeros(self.max_episode_len, dtype=np.float32)
        mask = np.zeros(self.max_episode_len, dtype=bool)

        obs_padded[:T] = obs
        act_padded[:T] = acts
        rew_padded[:T] = rews
        mask[:T] = True

        return {
            "observations": torch.from_numpy(obs_padded),
            "actions": torch.from_numpy(act_padded),
            "rewards": torch.from_numpy(rew_padded),
            "mask": torch.from_numpy(mask),
            "length": T,
        }

    def get_config_kwargs(self) -> dict:
        """Return kwargs for ArcMindConfig matched to this dataset."""
        return {
            "num_sensor_channels": self.obs_dim,
            "action_dim": self.act_dim,
        }

    def get_normalization_stats(self) -> dict:
        return {
            "obs_mean": self._obs_mean,
            "obs_std": self._obs_std,
        }

    def get_env_name(self) -> str:
        """Return the Gymnasium env name for online evaluation."""
        return MUJOCO_ENV_NAMES[self.task]

    def __repr__(self) -> str:
        return (
            f"MuJoCoLocomotionDataset(task='{self.task}', quality='{self.quality}', "
            f"episodes={self.total_episodes}, steps={self.total_steps}, "
            f"obs_dim={self.obs_dim}, act_dim={self.act_dim})"
        )
