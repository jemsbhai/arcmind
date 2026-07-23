"""Delayed key-value recall with overwrites and sensor-rate distractors.

Write events associate a discrete key with a discrete value. Query events ask
for the value most recently written to a key. Events occur only at decision
boundaries, while intervening sensor frames contain nuisance channels. Labels
are emitted only at queries, and query lag is retained for stratified metrics.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DelayedRecallConfig:
    """Configuration for the deterministic delayed sensor-recall task."""

    num_keys: int = 4
    num_values: int = 4
    num_noise_channels: int = 4
    num_decisions: int = 24
    sensor_stride: int = 4
    exact_recall_window: int = 8
    query_probability: float = 0.6
    overwrite_probability: float = 0.7
    noise_std: float = 0.5

    @property
    def input_dim(self) -> int:
        return 2 + self.num_keys + self.num_values + self.num_noise_channels

    @property
    def sequence_length(self) -> int:
        return self.num_decisions * self.sensor_stride


class DelayedRecallDataset(Dataset):
    """Pre-generated task split whose contents are fixed by a split seed."""

    ignore_index = -100

    def __init__(
        self,
        num_examples: int,
        *,
        config: DelayedRecallConfig | None = None,
        seed: int,
    ):
        if num_examples < 1:
            raise ValueError("num_examples must be positive")
        self.config = config or DelayedRecallConfig()
        self.seed = seed
        self.inputs, self.targets, self.query_lags = self._generate(num_examples)

    def _generate(
        self,
        num_examples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        config = self.config
        rng = np.random.default_rng(self.seed)
        inputs = np.zeros(
            (num_examples, config.sequence_length, config.input_dim),
            dtype=np.float32,
        )
        targets = np.full(
            (num_examples, config.sequence_length),
            self.ignore_index,
            dtype=np.int64,
        )
        query_lags = np.full_like(targets, -1)

        noise_start = 2 + config.num_keys + config.num_values
        inputs[:, :, noise_start:] = rng.normal(
            0.0,
            config.noise_std,
            size=(num_examples, config.sequence_length, config.num_noise_channels),
        )

        for example in range(num_examples):
            values_by_key: dict[int, int] = {}
            last_write: dict[int, int] = {}
            initial_keys = rng.permutation(config.num_keys)

            for decision in range(config.num_decisions):
                timestep = decision * config.sensor_stride
                if decision < config.num_keys:
                    event = "write"
                    key = int(initial_keys[decision])
                else:
                    event = (
                        "query"
                        if rng.random() < config.query_probability
                        else "write"
                    )
                    key = int(rng.integers(config.num_keys))

                if event == "write":
                    inputs[example, timestep, 0] = 1.0
                    inputs[example, timestep, 2 + key] = 1.0
                    if (
                        key in values_by_key
                        and rng.random() >= config.overwrite_probability
                    ):
                        value = values_by_key[key]
                    else:
                        value = int(rng.integers(config.num_values))
                    value_offset = 2 + config.num_keys + value
                    inputs[example, timestep, value_offset] = 1.0
                    values_by_key[key] = value
                    last_write[key] = decision
                else:
                    inputs[example, timestep, 1] = 1.0
                    inputs[example, timestep, 2 + key] = 1.0
                    targets[example, timestep] = values_by_key[key]
                    query_lags[example, timestep] = decision - last_write[key]

        return (
            torch.from_numpy(inputs),
            torch.from_numpy(targets),
            torch.from_numpy(query_lags),
        )

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index], self.query_lags[index]
