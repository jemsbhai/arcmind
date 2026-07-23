"""
ArcMind model configuration.

The presets are engineering starting points. Their dimensions and timescale
ratios are hypotheses to evaluate under the registered benchmark protocol, not
literature-derived optima.
"""

from dataclasses import dataclass


@dataclass
class ArcMindConfig:
    """Configuration for the ArcMind dual-timescale model."""

    # === Sensor input ===
    num_sensor_channels: int = 6
    """Number of raw sensor channels (e.g., 6 for IMU: 3 accel + 3 gyro)."""

    sensor_freq_hz: float = 100.0
    """Sensor sampling rate in Hz."""

    # === Model dimensions ===
    d_model: int = 128
    """Hidden dimension throughout the model."""

    # === Fast path: SSM core ===
    num_ssm_layers: int = 8
    """Number of selective SSM layers in the fast (sensor-rate) path."""

    ssm_state_dim: int = 16
    """State dimension per channel in the SSM recurrence."""

    ssm_conv_width: int = 4
    """Width of the causal depthwise convolution in each SSM layer."""

    ssm_expand_factor: int = 2
    """Expansion factor for the SSM inner dimension (d_inner = d_model * expand)."""

    # === Slow path: Tiny exact attention ===
    num_attn_layers: int = 1
    """Number of exact attention layers in the slow (decision-rate) path."""

    num_attn_heads: int = 2
    """Number of heads in each exact-attention layer."""

    attn_window_size: int = 64
    """Maximum exact-recall slots and local query attention width."""

    decision_freq_hz: float = 10.0
    """Decision-rate frequency in Hz (how often the slow path runs)."""

    # === Episodic memory ===
    num_memory_slots: int = 64
    """Number of slots in the episodic memory ring buffer."""

    memory_compress_ratio: int = 4
    """Compression ratio for writing SSM snapshots into memory slots."""

    # === Action output ===
    action_dim: int = 6
    """Dimensionality of the output action space."""

    # === Regularization ===
    dropout: float = 0.1
    """Dropout rate applied throughout the model."""

    # === Ablation flags ===
    ablate_ssm: bool = False
    """If True, skip SSM core — feed tokenized input directly to attention."""

    ablate_attention: bool = False
    """If True, skip slow attention — use SSM output directly for action head."""

    ablate_memory: bool = False
    """If True, disable episodic memory (attention runs without memory context)."""

    ablate_gating: bool = False
    """If True, replace learned gating with simple 0.5/0.5 average."""

    ablate_temporal_encoding: bool = False
    """If True, omit relative-age embeddings from episodic memory."""

    # === Presets ===
    @classmethod
    def iot_tiny(cls) -> "ArcMindConfig":
        """~245K params. For MCU/NPU deployment (Cortex-M7, ESP32-S3)."""
        return cls(
            num_sensor_channels=4,
            d_model=64,
            num_ssm_layers=4,
            ssm_state_dim=8,
            num_attn_layers=1,
            num_attn_heads=2,
            attn_window_size=32,
            num_memory_slots=16,
            action_dim=4,
        )

    @classmethod
    def robotics_small(cls) -> "ArcMindConfig":
        """~1.7M params. For Jetson Orin Nano, RPi 5 + Hailo."""
        return cls(
            num_sensor_channels=12,
            d_model=128,
            num_ssm_layers=8,
            ssm_state_dim=16,
            num_attn_layers=1,
            num_attn_heads=2,
            attn_window_size=64,
            num_memory_slots=64,
            action_dim=6,
        )

    @classmethod
    def robotics_medium(cls) -> "ArcMindConfig":
        """~10.3M params. For desktop GPU inference."""
        return cls(
            num_sensor_channels=24,
            d_model=256,
            num_ssm_layers=12,
            ssm_state_dim=32,
            num_attn_layers=2,
            num_attn_heads=4,
            attn_window_size=128,
            num_memory_slots=128,
            action_dim=12,
        )
