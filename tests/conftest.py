"""Shared test fixtures for arcmind tests."""

import pytest
import torch

from arcmind.config.defaults import ArcMindConfig


@pytest.fixture
def default_config():
    """Default config for testing."""
    return ArcMindConfig()


@pytest.fixture
def tiny_config():
    """Minimal config for fast unit tests."""
    return ArcMindConfig(
        num_sensor_channels=4,
        d_model=32,
        num_ssm_layers=2,
        ssm_state_dim=4,
        ssm_expand_factor=2,
        num_attn_layers=1,
        num_attn_heads=2,
        attn_window_size=8,
        num_memory_slots=4,
        memory_compress_ratio=2,
        action_dim=3,
        dropout=0.0,
        sensor_freq_hz=100.0,
        decision_freq_hz=10.0,
    )


@pytest.fixture
def device():
    """Use CPU for tests (no CUDA dependency)."""
    return torch.device("cpu")


@pytest.fixture
def batch_sensor_input(tiny_config):
    """Sample sensor input batch."""
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 20
    return torch.randn(batch_size, seq_len, tiny_config.num_sensor_channels)
