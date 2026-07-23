"""
D4RL AntMaze dataset loader via Minari.

Loads offline RL datasets for behavior cloning and offline RL evaluation.
The Ant navigates a maze to reach a goal — long-horizon, sparse reward,
and episodic memory should matter here (unlike short-window UCI HAR).

Datasets:
- D4RL/antmaze/umaze-v1:         U-shaped maze, fixed goal (1430 eps, 1M steps)
- D4RL/antmaze/medium-play-v1:   Medium maze, random goals
- D4RL/antmaze/large-diverse-v1: Large maze, diverse goals

Observation space (per step):
- observation: 27-dim (ant qpos/qvel)
- achieved_goal: 2-dim (ant xy position)
- desired_goal: 2-dim (target xy position)
- Total sensor input: 31-dim

Action space: 8-dim continuous (joint torques)
"""

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import minari
except ImportError:
    minari = None


ANTMAZE_DATASETS = {
    "umaze": "D4RL/antmaze/umaze-v1",
    "medium-play": "D4RL/antmaze/medium-play-v1",
    "large-diverse": "D4RL/antmaze/large-diverse-v1",
}

# Observation: 27 (ant state) + 2 (achieved goal) + 2 (desired goal)
NUM_OBS_CHANNELS = 31
ACTION_DIM = 8


def _flatten_obs(obs_dict: dict) -> np.ndarray:
    """
    Flatten the dict observation into a single array.

    Args:
        obs_dict: Dict with 'observation', 'achieved_goal', 'desired_goal'.

    Returns:
        Array of shape (seq_len, 31).
    """
    return np.concatenate(
        [obs_dict["observation"], obs_dict["achieved_goal"], obs_dict["desired_goal"]],
        axis=-1,
    ).astype(np.float32)


class AntMazeDataset(Dataset):
    """
    PyTorch Dataset for D4RL AntMaze offline RL (behavior cloning).

    Each sample is one full episode, zero-padded to max_episode_len.
    Returns (observations, actions, mask) where mask indicates valid timesteps.

    Args:
        variant: One of 'umaze', 'medium-play', 'large-diverse'.
        max_episode_len: Pad/truncate episodes to this length.
        normalize: Standardize observations using dataset statistics.
    """

    NUM_OBS_CHANNELS = NUM_OBS_CHANNELS
    ACTION_DIM = ACTION_DIM

    def __init__(
        self,
        variant: str = "umaze",
        max_episode_len: int = 700,
        normalize: bool = True,
    ):
        if minari is None:
            raise ImportError("minari is required: pip install minari")

        assert variant in ANTMAZE_DATASETS, (
            f"variant must be one of {list(ANTMAZE_DATASETS.keys())}, got '{variant}'"
        )

        self.variant = variant
        self.max_episode_len = max_episode_len
        dataset_id = ANTMAZE_DATASETS[variant]

        print(f"Loading {dataset_id}...")
        dataset = minari.load_dataset(dataset_id, download=True)

        # Extract all episodes into lists
        self.observations = []
        self.actions = []
        self.rewards = []
        self.episode_lengths = []

        for ep in dataset.iterate_episodes():
            obs = _flatten_obs(ep.observations)
            # obs has T+1 entries, actions/rewards have T entries
            # Use obs[:-1] to align with actions
            obs = obs[:-1]
            acts = ep.actions.astype(np.float32)
            rews = ep.rewards.astype(np.float32)

            T = min(len(acts), max_episode_len)
            self.observations.append(obs[:T])
            self.actions.append(acts[:T])
            self.rewards.append(rews[:T])
            self.episode_lengths.append(T)

        # Compute normalization statistics
        if normalize:
            all_obs = np.concatenate(self.observations, axis=0)
            self._obs_mean = all_obs.mean(axis=0)
            self._obs_std = all_obs.std(axis=0)
            self._obs_std[self._obs_std < 1e-8] = 1.0

            self.observations = [
                (obs - self._obs_mean) / self._obs_std for obs in self.observations
            ]

            all_acts = np.concatenate(self.actions, axis=0)
            self._act_mean = all_acts.mean(axis=0)
            self._act_std = all_acts.std(axis=0)
            self._act_std[self._act_std < 1e-8] = 1.0
        else:
            self._obs_mean = np.zeros(NUM_OBS_CHANNELS)
            self._obs_std = np.ones(NUM_OBS_CHANNELS)
            self._act_mean = np.zeros(ACTION_DIM)
            self._act_std = np.ones(ACTION_DIM)

        self.total_episodes = len(self.observations)
        self.total_steps = sum(self.episode_lengths)

        print(f"Loaded {self.total_episodes} episodes, {self.total_steps} total steps")
        print(
            f"Episode lengths: min={min(self.episode_lengths)}, "
            f"max={max(self.episode_lengths)}, "
            f"mean={np.mean(self.episode_lengths):.0f}"
        )

    def __len__(self) -> int:
        return self.total_episodes

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with:
                'observations': (max_episode_len, 31) float32, zero-padded
                'actions': (max_episode_len, 8) float32, zero-padded
                'rewards': (max_episode_len,) float32, zero-padded
                'mask': (max_episode_len,) bool, True for valid timesteps
                'length': int, actual episode length
        """
        obs = self.observations[idx]
        acts = self.actions[idx]
        rews = self.rewards[idx]
        T = self.episode_lengths[idx]

        # Zero-pad to max_episode_len
        obs_padded = np.zeros((self.max_episode_len, NUM_OBS_CHANNELS), dtype=np.float32)
        act_padded = np.zeros((self.max_episode_len, ACTION_DIM), dtype=np.float32)
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
        """Return kwargs to construct an ArcMindConfig matched to this dataset."""
        return {
            "num_sensor_channels": NUM_OBS_CHANNELS,
            "action_dim": ACTION_DIM,
        }

    def get_normalization_stats(self) -> dict:
        """Return obs/action normalization statistics for evaluation."""
        return {
            "obs_mean": self._obs_mean,
            "obs_std": self._obs_std,
            "act_mean": self._act_mean,
            "act_std": self._act_std,
        }

    def __repr__(self) -> str:
        return (
            f"AntMazeDataset(variant='{self.variant}', "
            f"episodes={self.total_episodes}, "
            f"steps={self.total_steps}, "
            f"max_len={self.max_episode_len})"
        )
