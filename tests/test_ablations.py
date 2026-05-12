"""Tests for ablation flag behavior."""

import pytest
import torch

from arcmind import ArcMindConfig, ArcMindModel


@pytest.fixture
def base_config():
    """Config matched to UCI HAR for ablation testing."""
    return ArcMindConfig(
        num_sensor_channels=6,
        d_model=32,
        num_ssm_layers=2,
        ssm_state_dim=4,
        ssm_expand_factor=2,
        num_attn_layers=1,
        num_attn_heads=2,
        attn_window_size=8,
        num_memory_slots=4,
        memory_compress_ratio=2,
        action_dim=6,
        dropout=0.0,
        sensor_freq_hz=50.0,
        decision_freq_hz=10.0,
    )


@pytest.fixture
def sensor_input():
    torch.manual_seed(42)
    return torch.randn(2, 20, 6)


class TestAblationFlags:
    def test_full_model_runs(self, base_config, sensor_input):
        model = ArcMindModel(base_config)
        model.reset_memory(batch_size=2)
        out = model(sensor_input)
        assert out.shape == (2, 20, 6)

    def test_ablate_ssm(self, base_config, sensor_input):
        """Without SSM, tokenized input goes directly to attention."""
        base_config.ablate_ssm = True
        model = ArcMindModel(base_config)
        model.reset_memory(batch_size=2)
        out = model(sensor_input)
        assert out.shape == (2, 20, 6)

    def test_ablate_attention(self, base_config, sensor_input):
        """Without attention, SSM output goes directly to action head."""
        base_config.ablate_attention = True
        model = ArcMindModel(base_config)
        model.reset_memory(batch_size=2)
        out = model(sensor_input)
        assert out.shape == (2, 20, 6)

    def test_ablate_memory(self, base_config, sensor_input):
        """Without memory, attention runs but with no memory context."""
        base_config.ablate_memory = True
        model = ArcMindModel(base_config)
        model.reset_memory(batch_size=2)
        out = model(sensor_input)
        assert out.shape == (2, 20, 6)
        # Memory should not have been written to
        assert model.memory.get_occupancy() == 0

    def test_ablate_gating(self, base_config, sensor_input):
        """Without gating, fast and slow paths are averaged 50/50."""
        base_config.ablate_gating = True
        model = ArcMindModel(base_config)
        model.reset_memory(batch_size=2)
        out = model(sensor_input)
        assert out.shape == (2, 20, 6)

    def test_ablated_outputs_differ(self, base_config, sensor_input):
        """Different ablations should produce different outputs."""
        torch.manual_seed(0)

        # Full model
        model_full = ArcMindModel(base_config)
        model_full.eval()
        model_full.reset_memory(batch_size=2)
        with torch.no_grad():
            out_full = model_full(sensor_input)

        # No attention
        cfg_no_attn = ArcMindConfig(**{**vars(base_config), "ablate_attention": True})
        torch.manual_seed(0)
        model_no_attn = ArcMindModel(cfg_no_attn)
        model_no_attn.eval()
        model_no_attn.reset_memory(batch_size=2)
        with torch.no_grad():
            out_no_attn = model_no_attn(sensor_input)

        # Outputs should differ (attention contributes something)
        assert not torch.allclose(out_full, out_no_attn, atol=1e-3)

    def test_gradient_flow_all_ablations(self, base_config, sensor_input):
        """Gradients should flow in every ablation mode."""
        ablation_flags = [
            {"ablate_ssm": True},
            {"ablate_attention": True},
            {"ablate_memory": True},
            {"ablate_gating": True},
        ]
        for flags in ablation_flags:
            cfg = ArcMindConfig(**{**vars(base_config), **flags})
            model = ArcMindModel(cfg)
            model.reset_memory(batch_size=2)
            x = sensor_input.clone().requires_grad_(True)
            out = model(x)
            loss = out.sum()
            loss.backward()
            assert x.grad is not None, f"No gradient flow with {flags}"
            assert x.grad.abs().sum() > 0, f"Zero gradients with {flags}"
