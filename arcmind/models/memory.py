"""
Episodic memory ring buffer with learned compression.

Stores compressed snapshots of the SSM hidden state at regular intervals,
forming a fixed-size ring buffer of environment state "memories." The slow
attention path can attend over these slots for precise spatial/temporal recall.

Design rationale:
- Inspired by Expansion Span (AWS, Dec 2024): reserves attention context
  for tokens retrieved from arbitrarily distant past.
- Fixed-size ring buffer (not growing KV cache) ensures O(1) memory at inference.
- Learned MLP compressor reduces SSM state snapshots to memory slot dimension.
"""

import torch
import torch.nn as nn

from arcmind.config.defaults import ArcMindConfig


class MemoryCompressor(nn.Module):
    """Compresses SSM hidden states into compact memory slot representations."""

    def __init__(self, config: ArcMindConfig):
        super().__init__()
        self.config = config
        # Compress from d_model to d_model with dimensionality reduction
        self.compress = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // config.memory_compress_ratio),
            nn.GELU(),
            nn.Linear(config.d_model // config.memory_compress_ratio, config.d_model),
            nn.LayerNorm(config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: SSM output snapshot, shape (batch, d_model) or (batch, window, d_model).

        Returns:
            Compressed memory slot, shape (batch, d_model) or (batch, window, d_model).
        """
        return self.compress(x)


class EpisodicMemory(nn.Module):
    """Fixed-size ring buffer of compressed environment state snapshots."""

    def __init__(self, config: ArcMindConfig):
        super().__init__()
        self.config = config
        self.num_slots = config.num_memory_slots
        self.d_model = config.d_model
        self.compressor = MemoryCompressor(config)

        # Runtime state — NOT saved in state_dict (persistent=False).
        # These are ephemeral per-episode state, not learned parameters.
        self.register_buffer(
            "write_ptr", torch.zeros(1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "buffer",
            torch.zeros(1, config.num_memory_slots, config.d_model),
            persistent=False,
        )

    def reset(self, batch_size: int = 1, device: torch.device | None = None) -> None:
        """Reset the memory buffer for a new episode."""
        dev = device or self.buffer.device
        self.buffer = torch.zeros(batch_size, self.num_slots, self.d_model, device=dev)
        self.write_ptr = torch.zeros(1, dtype=torch.long, device=dev)

    def write(self, snapshot: torch.Tensor) -> None:
        """
        Write a compressed snapshot into the next ring buffer slot.

        Args:
            snapshot: SSM output to compress and store, shape (batch, d_model).
        """
        compressed = self.compressor(snapshot)
        idx = (self.write_ptr % self.num_slots).long().item()
        self.buffer[:, idx, :] = compressed.detach()
        self.write_ptr += 1

    def read(self) -> torch.Tensor:
        """
        Read all memory slots for attention.

        Returns:
            Memory tensor, shape (batch, num_slots, d_model).
        """
        return self.buffer

    def get_occupancy(self) -> int:
        """Return the number of slots that have been written to."""
        return min(self.write_ptr.item(), self.num_slots)
