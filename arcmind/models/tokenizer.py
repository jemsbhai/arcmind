"""
Sensor-native tokenizer.

Bypasses the standard vocabulary embedding table entirely.
Projects raw fixed-width sensor frames directly into model dimension
via a learned linear projection. This eliminates the ~40% parameter
overhead that embedding tables impose on sub-100M text LMs.
"""

import torch
import torch.nn as nn

from arcmind.config.defaults import ArcMindConfig


class SensorTokenizer(nn.Module):
    """Projects raw sensor channels into model-dimension token embeddings."""

    def __init__(self, config: ArcMindConfig):
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.num_sensor_channels, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw sensor input, shape (batch, seq_len, num_sensor_channels).
               Each timestep is one sensor frame (e.g., 6 floats for IMU).

        Returns:
            Tokenized embeddings, shape (batch, seq_len, d_model).
        """
        return self.norm(self.projection(x))
