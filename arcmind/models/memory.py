"""
Episodic memory ring buffer with learned compression.

Stores compressed snapshots of the SSM hidden state at regular intervals,
forming a fixed-size ring buffer of environment state "memories." The slow
attention path can attend over these slots for precise spatial/temporal recall.

The fixed-size ring buffer bounds inference memory. Learned compression maps
decision snapshots back into model dimension, while chronological reads support
relative-age encoding in the exact-recall path.
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
        self.buffer = torch.zeros(
            batch_size,
            self.num_slots,
            self.d_model,
            device=dev,
            dtype=self.buffer.dtype,
        )
        self.write_ptr = torch.zeros(1, dtype=torch.long, device=dev)

    def write(self, snapshot: torch.Tensor) -> torch.Tensor:
        """
        Write a compressed snapshot into the next ring buffer slot.

        Args:
            snapshot: SSM output to compress and store, shape (batch, d_model).

        Returns:
            The differentiable compressed snapshot. The runtime copy stored in
            the ring buffer is detached because it is episode state, not model
            state. Batched training can use the returned tensor to preserve
            gradient flow through the compressor.
        """
        compressed = self.compressor(snapshot)
        self.write_compressed(compressed)
        return compressed

    def write_compressed(self, compressed: torch.Tensor) -> None:
        """Store an already-compressed snapshot as detached runtime state."""
        expected_shape = (self.buffer.shape[0], self.d_model)
        if compressed.shape != expected_shape:
            raise ValueError(
                f"compressed snapshot must have shape {expected_shape}, "
                f"got {tuple(compressed.shape)}"
            )

        idx = (self.write_ptr % self.num_slots).long().item()
        with torch.no_grad():
            self.buffer[:, idx, :].copy_(compressed)
            self.write_ptr.add_(1)

    def read(self, valid_only: bool = False) -> torch.Tensor:
        """
        Read memory slots.

        Args:
            valid_only: If False, return the fixed-size physical buffer. If
                True, omit unwritten slots and return entries in chronological
                order (oldest to newest), including after ring-buffer wrap.

        Returns:
            Memory tensor with shape (batch, slots, d_model).
        """
        if not valid_only:
            return self.buffer

        occupancy = self.get_occupancy()
        if occupancy < self.num_slots:
            return self.buffer[:, :occupancy, :]

        oldest = int((self.write_ptr % self.num_slots).item())
        if oldest == 0:
            return self.buffer
        return torch.cat(
            [self.buffer[:, oldest:, :], self.buffer[:, :oldest, :]],
            dim=1,
        )

    def get_occupancy(self) -> int:
        """Return the number of slots that have been written to."""
        return min(self.write_ptr.item(), self.num_slots)
