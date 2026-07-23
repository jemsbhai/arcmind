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

        In training, processes the entire sequence at once. Each call is
        stateless: episodic slots are local to the supplied sequence and never
        leak across independent minibatches.
        The slow path runs at strided intervals (every decision_stride steps)
        and its latest output is held between decisions. Memory is updated
        causally so a timestep can never attend to a future snapshot.
        Ablation flags in config control which components are active.

        Args:
            sensor_input: Raw sensor data, shape (batch, seq_len, num_sensor_channels).
            use_memory: Whether to use episodic memory for attention context.

        Returns:
            Actions, shape (batch, seq_len, action_dim).
        """
        _, seq_len, _ = sensor_input.shape
        cfg = self.config

        # Step 1: Tokenize sensor input
        tokens = self.tokenizer(sensor_input)

        # Step 2: Fast path — SSM processes all tokens (unless ablated)
        if cfg.ablate_ssm:
            fast_output = tokens  # skip SSM, pass raw embeddings
        else:
            fast_output = self.ssm_core(tokens)

        # Step 3-4: Slow path — attention with optional memory (unless ablated)
        if cfg.ablate_attention:
            # Skip attention entirely, use fast output for action head
            fused = fast_output
        else:
            # Recall from strictly prior decision states, then write the
            # current snapshot. This avoids duplicating the current state as
            # both query and memory key. Batch memory remains differentiable
            # and local to this call. Persistent runtime memory is reserved
            # for the explicit streaming API.
            effective_memory = use_memory and not cfg.ablate_memory
            memory_slots: list[torch.Tensor] = []

            fused_segments = []
            for start in range(0, seq_len, self.decision_stride):
                end = min(start + self.decision_stride, seq_len)
                snapshot = fast_output[:, start, :]

                if effective_memory:
                    memory_context = (
                        torch.stack(memory_slots, dim=1)
                        if memory_slots
                        else None
                    )
                else:
                    memory_context = None

                # Attention is evaluated once per decision interval, matching
                # streaming inference rather than every sensor-rate timestep.
                slow_decision = self.slow_attention(
                    snapshot.unsqueeze(1),
                    memory=memory_context,
                )

                if effective_memory:
                    compressed = self.memory.compressor(snapshot)
                    memory_slots.append(compressed)
                    memory_slots = memory_slots[-cfg.num_memory_slots :]

                segment_fast = fast_output[:, start:end, :]
                segment_slow = slow_decision.expand(-1, end - start, -1)

                if cfg.ablate_ssm:
                    fused_segment = segment_slow
                elif cfg.ablate_gating:
                    fused_segment = 0.5 * segment_fast + 0.5 * segment_slow
                else:
                    combined = torch.cat([segment_fast, segment_slow], dim=-1)
                    gate_values = self.gate(combined)
                    fused_segment = (
                        gate_values * segment_slow
                        + (1 - gate_values) * segment_fast
                    )

                fused_segments.append(fused_segment)

            fused = torch.cat(fused_segments, dim=1)

        # Step 6: Action prediction
        actions = self.action_head(fused)

        return actions

    def reset_memory(self, batch_size: int = 1) -> None:
        """Reset persistent episodic memory used by streaming inference."""
        self.memory.reset(batch_size=batch_size, device=next(self.parameters()).device)

    def init_streaming(self, batch_size: int = 1) -> None:
        """
        Initialize streaming (recurrent) inference mode.

        Must be called before the first step() call in an episode.
        Initializes SSM hidden state, resets memory, and resets the
        internal step counter.
        """
        parameter = next(self.parameters())
        device = parameter.device
        dtype = parameter.dtype
        self.ssm_core.init_state(batch_size, device, dtype=dtype)
        self.memory.reset(batch_size=batch_size, device=device)
        self._step_counter = 0
        self._last_slow_output = torch.zeros(
            batch_size,
            self.config.d_model,
            device=device,
            dtype=dtype,
        )

    def step(self, sensor_frame: torch.Tensor) -> torch.Tensor:
        """
        Process a single sensor frame in streaming mode.

        SSM state persists between calls. Episodic memory is written
        and attention is run at decision rate (every decision_stride steps).

        Must call init_streaming() before first step().

        Args:
            sensor_frame: Single sensor reading, shape (batch, num_sensor_channels).

        Returns:
            Action, shape (batch, action_dim).
        """
        cfg = self.config

        # Tokenize single frame: (batch, channels) -> (batch, d_model)
        token = self.tokenizer.projection(sensor_frame)
        token = self.tokenizer.norm(token)

        # Fast path: SSM step with persistent state
        if cfg.ablate_ssm:
            fast_output = token
        else:
            fast_output = self.ssm_core.step(token)  # (batch, d_model)

        # Slow path: runs at decision rate
        if cfg.ablate_attention:
            fused = fast_output
        else:
            if self._step_counter % self.decision_stride == 0:
                # Recall strictly prior snapshots. The current decision state
                # is written only after attention has produced its output.
                memory_slots = (
                    self.memory.read(valid_only=True)
                    if not cfg.ablate_memory
                    else None
                )
                query = fast_output.unsqueeze(1)  # (batch, 1, d_model)
                slow_out = self.slow_attention(
                    query,
                    memory=memory_slots,
                )  # (batch, 1, d_model)
                self._last_slow_output = slow_out.squeeze(1)  # (batch, d_model)

                if not cfg.ablate_memory:
                    self.memory.write(fast_output)

            # Gate fast and slow
            if cfg.ablate_ssm:
                fused = self._last_slow_output
            elif cfg.ablate_gating:
                fused = 0.5 * fast_output + 0.5 * self._last_slow_output
            else:
                combined = torch.cat([fast_output, self._last_slow_output], dim=-1)
                gate_values = self.gate(combined)  # (batch, d_model)
                fused = gate_values * self._last_slow_output + (1 - gate_values) * fast_output

        self._step_counter += 1

        # Action head: (batch, d_model) -> (batch, action_dim)
        action = self.action_head.head(fused)
        return action

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
