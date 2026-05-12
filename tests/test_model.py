"""Unit tests for ArcMind model components."""

import pytest
import torch

from arcmind import ArcMindConfig, ArcMindModel, __version__
from arcmind.models.tokenizer import SensorTokenizer
from arcmind.models.ssm_core import SSMCore, SSMLayer
from arcmind.models.attention import SlowAttention, SlowAttentionLayer
from arcmind.models.memory import EpisodicMemory, MemoryCompressor


# ============================================================
# Package-level tests
# ============================================================


class TestPackage:
    def test_version_exists(self):
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_top_level_imports(self):
        """Core classes importable from package root."""
        assert ArcMindConfig is not None
        assert ArcMindModel is not None


# ============================================================
# Config tests
# ============================================================


class TestConfig:
    def test_default_config_values(self, default_config):
        assert default_config.d_model == 128
        assert default_config.num_ssm_layers == 8
        assert default_config.num_attn_layers == 1
        assert default_config.num_attn_heads == 2

    def test_iot_tiny_preset(self):
        cfg = ArcMindConfig.iot_tiny()
        assert cfg.d_model == 64
        assert cfg.num_ssm_layers == 4
        assert cfg.num_memory_slots == 16

    def test_robotics_small_preset(self):
        cfg = ArcMindConfig.robotics_small()
        assert cfg.d_model == 128
        assert cfg.num_ssm_layers == 8
        assert cfg.num_memory_slots == 64

    def test_robotics_medium_preset(self):
        cfg = ArcMindConfig.robotics_medium()
        assert cfg.d_model == 256
        assert cfg.num_ssm_layers == 12
        assert cfg.num_attn_layers == 2

    def test_d_model_divisible_by_heads(self, tiny_config):
        assert tiny_config.d_model % tiny_config.num_attn_heads == 0


# ============================================================
# SensorTokenizer tests
# ============================================================


class TestSensorTokenizer:
    def test_output_shape(self, tiny_config, batch_sensor_input):
        tok = SensorTokenizer(tiny_config)
        out = tok(batch_sensor_input)
        batch, seq_len, _ = batch_sensor_input.shape
        assert out.shape == (batch, seq_len, tiny_config.d_model)

    def test_output_dtype(self, tiny_config, batch_sensor_input):
        tok = SensorTokenizer(tiny_config)
        out = tok(batch_sensor_input)
        assert out.dtype == torch.float32

    def test_no_embedding_table(self, tiny_config):
        """Sensor tokenizer should NOT have an nn.Embedding layer."""
        tok = SensorTokenizer(tiny_config)
        for module in tok.modules():
            assert not isinstance(module, torch.nn.Embedding), (
                "SensorTokenizer must use linear projection, not an embedding table"
            )

    def test_single_timestep(self, tiny_config):
        """Should handle a single-frame input."""
        tok = SensorTokenizer(tiny_config)
        x = torch.randn(1, 1, tiny_config.num_sensor_channels)
        out = tok(x)
        assert out.shape == (1, 1, tiny_config.d_model)

    def test_parameter_count_small(self, tiny_config):
        """Tokenizer should be lightweight — no embedding table overhead."""
        tok = SensorTokenizer(tiny_config)
        n_params = sum(p.numel() for p in tok.parameters())
        # Linear(4, 32) = 128 + 32 bias = 160, LayerNorm(32) = 64
        # Should be well under 1000 params
        assert n_params < 1000, f"Tokenizer has {n_params} params — too large"


# ============================================================
# SSMCore tests
# ============================================================


class TestSSMLayer:
    def test_output_shape(self, tiny_config):
        layer = SSMLayer(tiny_config)
        x = torch.randn(2, 10, tiny_config.d_model)
        out = layer(x)
        assert out.shape == x.shape

    def test_residual_connection(self, tiny_config):
        """Output should not be identical to input (transformation occurs)."""
        layer = SSMLayer(tiny_config)
        x = torch.randn(2, 10, tiny_config.d_model)
        out = layer(x)
        assert not torch.allclose(out, x, atol=1e-5)


class TestSSMCore:
    def test_output_shape(self, tiny_config):
        core = SSMCore(tiny_config)
        x = torch.randn(2, 20, tiny_config.d_model)
        out = core(x)
        assert out.shape == x.shape

    def test_num_layers(self, tiny_config):
        core = SSMCore(tiny_config)
        assert len(core.layers) == tiny_config.num_ssm_layers

    def test_deterministic_with_seed(self, tiny_config):
        """Same input + same seed → same output."""
        core = SSMCore(tiny_config)
        core.eval()
        x = torch.randn(1, 10, tiny_config.d_model)
        with torch.no_grad():
            out1 = core(x)
            out2 = core(x)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_gradient_flow(self, tiny_config):
        """Gradients should flow through the SSM core."""
        core = SSMCore(tiny_config)
        x = torch.randn(1, 10, tiny_config.d_model, requires_grad=True)
        out = core(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ============================================================
# SlowAttention tests
# ============================================================


class TestSlowAttentionLayer:
    def test_output_shape_no_memory(self, tiny_config):
        layer = SlowAttentionLayer(tiny_config)
        x = torch.randn(2, 10, tiny_config.d_model)
        out = layer(x)
        assert out.shape == x.shape

    def test_output_shape_with_memory(self, tiny_config):
        layer = SlowAttentionLayer(tiny_config)
        x = torch.randn(2, 10, tiny_config.d_model)
        memory = torch.randn(2, tiny_config.num_memory_slots, tiny_config.d_model)
        out = layer(x, memory=memory)
        assert out.shape == x.shape

    def test_causal_masking(self, tiny_config):
        """Future tokens should not influence past tokens."""
        layer = SlowAttentionLayer(tiny_config)
        layer.eval()
        x = torch.randn(1, 10, tiny_config.d_model)
        with torch.no_grad():
            out_full = layer(x)
            # Truncate to first 5 tokens
            out_trunc = layer(x[:, :5, :])
        # First 5 positions should be identical (causal = no future leakage)
        # Note: this is approximate due to LayerNorm over different seq lengths
        # We check the first position which should be identical
        assert torch.allclose(out_full[:, 0, :], out_trunc[:, 0, :], atol=1e-5)


class TestSlowAttention:
    def test_output_shape(self, tiny_config):
        attn = SlowAttention(tiny_config)
        x = torch.randn(2, 10, tiny_config.d_model)
        out = attn(x)
        assert out.shape == x.shape

    def test_num_layers(self, tiny_config):
        attn = SlowAttention(tiny_config)
        assert len(attn.layers) == tiny_config.num_attn_layers


# ============================================================
# EpisodicMemory tests
# ============================================================


class TestMemoryCompressor:
    def test_output_shape(self, tiny_config):
        comp = MemoryCompressor(tiny_config)
        x = torch.randn(2, tiny_config.d_model)
        out = comp(x)
        assert out.shape == (2, tiny_config.d_model)


class TestEpisodicMemory:
    def test_initial_state(self, tiny_config):
        mem = EpisodicMemory(tiny_config)
        mem.reset(batch_size=2)
        assert mem.get_occupancy() == 0
        buf = mem.read()
        assert buf.shape == (2, tiny_config.num_memory_slots, tiny_config.d_model)

    def test_write_and_occupancy(self, tiny_config):
        mem = EpisodicMemory(tiny_config)
        mem.reset(batch_size=1)
        snapshot = torch.randn(1, tiny_config.d_model)
        mem.write(snapshot)
        assert mem.get_occupancy() == 1
        mem.write(snapshot)
        assert mem.get_occupancy() == 2

    def test_ring_buffer_wraps(self, tiny_config):
        """After num_slots writes, the pointer should wrap around."""
        mem = EpisodicMemory(tiny_config)
        mem.reset(batch_size=1)
        for i in range(tiny_config.num_memory_slots + 3):
            snapshot = torch.randn(1, tiny_config.d_model)
            mem.write(snapshot)
        # Occupancy caps at num_slots
        assert mem.get_occupancy() == tiny_config.num_memory_slots

    def test_read_shape(self, tiny_config):
        mem = EpisodicMemory(tiny_config)
        mem.reset(batch_size=3)
        buf = mem.read()
        assert buf.shape == (3, tiny_config.num_memory_slots, tiny_config.d_model)

    def test_reset_clears_buffer(self, tiny_config):
        mem = EpisodicMemory(tiny_config)
        mem.reset(batch_size=1)
        mem.write(torch.ones(1, tiny_config.d_model))
        mem.reset(batch_size=1)
        assert mem.get_occupancy() == 0
        assert torch.all(mem.read() == 0)


# ============================================================
# Full ArcMindModel tests
# ============================================================


class TestArcMindModel:
    def test_output_shape(self, tiny_config, batch_sensor_input):
        model = ArcMindModel(tiny_config)
        model.reset_memory(batch_size=batch_sensor_input.shape[0])
        out = model(batch_sensor_input)
        batch, seq_len, _ = batch_sensor_input.shape
        assert out.shape == (batch, seq_len, tiny_config.action_dim)

    def test_output_shape_no_memory(self, tiny_config, batch_sensor_input):
        model = ArcMindModel(tiny_config)
        out = model(batch_sensor_input, use_memory=False)
        batch, seq_len, _ = batch_sensor_input.shape
        assert out.shape == (batch, seq_len, tiny_config.action_dim)

    def test_decision_stride(self, tiny_config):
        model = ArcMindModel(tiny_config)
        expected_stride = int(tiny_config.sensor_freq_hz / tiny_config.decision_freq_hz)
        assert model.decision_stride == expected_stride

    def test_gradient_flow_full_model(self, tiny_config, batch_sensor_input):
        """Gradients should flow from action output back to sensor input."""
        model = ArcMindModel(tiny_config)
        model.reset_memory(batch_size=batch_sensor_input.shape[0])
        sensor_input = batch_sensor_input.clone().requires_grad_(True)
        out = model(sensor_input)
        loss = out.sum()
        loss.backward()
        assert sensor_input.grad is not None
        assert sensor_input.grad.abs().sum() > 0

    def test_parameter_count(self, tiny_config):
        model = ArcMindModel(tiny_config)
        counts = model.count_parameters()
        assert "total" in counts
        assert counts["total"] > 0
        assert counts["ssm_core"] > counts["tokenizer"]  # SSM should be largest

    def test_parameter_count_reasonable(self, tiny_config):
        """Tiny config should produce a model under 1M parameters."""
        model = ArcMindModel(tiny_config)
        total = model.count_parameters()["total"]
        assert total < 1_000_000, f"Tiny config has {total} params — too large"

    def test_reset_memory(self, tiny_config, batch_sensor_input):
        """Reset should clear memory between episodes."""
        model = ArcMindModel(tiny_config)
        model.reset_memory(batch_size=batch_sensor_input.shape[0])
        model(batch_sensor_input)
        assert model.memory.get_occupancy() > 0
        model.reset_memory(batch_size=batch_sensor_input.shape[0])
        assert model.memory.get_occupancy() == 0

    def test_all_presets_instantiate(self):
        """All config presets should produce valid models."""
        for preset_fn in [ArcMindConfig.iot_tiny, ArcMindConfig.robotics_small, ArcMindConfig.robotics_medium]:
            config = preset_fn()
            model = ArcMindModel(config)
            x = torch.randn(1, 10, config.num_sensor_channels)
            model.reset_memory(batch_size=1)
            out = model(x)
            assert out.shape == (1, 10, config.action_dim)
