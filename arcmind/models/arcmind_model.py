"""
ArcMind: Dual-timescale hybrid SSM+Attention model.

Assembles the fast SSM path (sensor-rate processing), slow attention path
(decision-rate reasoning with episodic memory), and action output head
into a unified architecture.

Architecture:
    Sensor Stream → SensorTokenizer → SSMCore (fast, 100-1000 Hz)
                                          ↓ periodic snapshot
                                     EpisodicMemory (ring buffer)
                                          ↓ read
    SSM output → SlowAttention (slow, 1-10 Hz) ← memory slots
                       ↓
                  ActionHead → action output
"""

import torch
import torch.nn as nn

from arcmind.config.defaults import ArcMindConfig
from arcmind.models.attention import SlowAttention
from arcmind.models.memory import EpisodicMemory
from arcmind.models.ssm_core import SSMCore
from arcmind.models.tokenizer import SensorTokenizer


class ActionHead(nn.Module):
    """Projects model output to action space."""

    def __init__(self, config: ArcMindConfig):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Model output, shape (batch, seq_len, d_model).

        Returns:
            Actions, shape (batch, seq_len, action_dim).
        """
        return self.head(x)


class ArcMindModel(nn.Module):
    """
    Dual-timescale model for robotics and IoT.

    Fast path (SSM): processes every sensor frame at sensor rate.
    Slow path (Attention): runs at decision rate over SSM output + episodic memory.
    """

    def __init__(self, config: ArcMindConfig):
        super().__init__()
        self.config = config

        # Sensor → embedding
        self.tokenizer = SensorTokenizer(config)

        # Fast path: SSM core
        self.ssm_core = SSMCore(config)

        # Episodic memory
        self.memory = EpisodicMemory(config)

        # Slow path: tiny attention
        self.slow_attention = SlowAttention(config)

        # Gating: learned blend of fast and slow paths
        self.gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.Sigmoid(),
        )

        # Action output
        self.action_head = ActionHead(config)

        # Compute decision stride (how many fast steps per slow step)
        self.decision_stride = max(1, int(config.sensor_freq_hz / config.decision_freq_hz))

    def forward(
        self,
        sensor_input: torch.Tensor,
        use_memory: bool = True,
    ) -> torch.Tensor:
        """
        Full forward pass over a sensor sequence.

        In training, processes the entire sequence at once.
        The slow path runs at strided intervals (every decision_stride steps).

        Args:
            sensor_input: Raw sensor data, shape (batch, seq_len, num_sensor_channels).
            use_memory: Whether to use episodic memory for attention context.

        Returns:
            Actions, shape (batch, seq_len, action_dim).
        """
        batch, seq_len, _ = sensor_input.shape

        # Step 1: Tokenize sensor input
        tokens = self.tokenizer(sensor_input)

        # Step 2: Fast path — SSM processes all tokens
        fast_output = self.ssm_core(tokens)

        # Step 3: Write periodic snapshots to memory
        if use_memory:
            for t in range(0, seq_len, self.decision_stride):
                snapshot = fast_output[:, t, :]  # (batch, d_model)
                self.memory.write(snapshot)

        # Step 4: Slow path — attention at decision rate with memory
        memory_slots = self.memory.read() if use_memory else None
        slow_output = self.slow_attention(fast_output, memory=memory_slots)

        # Step 5: Gate fast and slow outputs
        combined = torch.cat([fast_output, slow_output], dim=-1)
        gate_values = self.gate(combined)
        fused = gate_values * slow_output + (1 - gate_values) * fast_output

        # Step 6: Action prediction
        actions = self.action_head(fused)

        return actions

    def reset_memory(self, batch_size: int = 1) -> None:
        """Reset episodic memory for a new episode."""
        self.memory.reset(batch_size=batch_size, device=next(self.parameters()).device)

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by component."""
        components = {
            "tokenizer": self.tokenizer,
            "ssm_core": self.ssm_core,
            "memory": self.memory,
            "slow_attention": self.slow_attention,
            "gate": self.gate,
            "action_head": self.action_head,
        }
        counts = {}
        for name, module in components.items():
            counts[name] = sum(p.numel() for p in module.parameters())
        counts["total"] = sum(counts.values())
        return counts
