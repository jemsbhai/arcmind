"""Tests for streaming (recurrent) inference mode."""

import pytest
import torch

from arcmind import ArcMindConfig, ArcMindModel


@pytest.fixture
def streaming_config():
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
        action_dim=3,
        dropout=0.0,
        sensor_freq_hz=50.0,
        decision_freq_hz=10.0,
    )


class TestSSMLayerStep:
    def test_step_output_shape(self, streaming_config):
        from arcmind.models.ssm_core import SSMLayer
        layer = SSMLayer(streaming_config)
        layer.init_state(batch_size=2, device=torch.device("cpu"))
        x = torch.randn(2, streaming_config.d_model)
        out = layer.step(x)
        assert out.shape == (2, streaming_config.d_model)

    def test_step_state_changes(self, streaming_config):
        from arcmind.models.ssm_core import SSMLayer
        layer = SSMLayer(streaming_config)
        layer.eval()
        layer.init_state(batch_size=1, device=torch.device("cpu"))

        x = torch.randn(1, streaming_config.d_model)
        with torch.no_grad():
            out1 = layer.step(x)
            out2 = layer.step(x)
        # Same input but different state → different output
        assert not torch.allclose(out1, out2, atol=1e-5)

    def test_step_reset_restores(self, streaming_config):
        from arcmind.models.ssm_core import SSMLayer
        layer = SSMLayer(streaming_config)
        layer.eval()

        x = torch.randn(1, streaming_config.d_model)

        # First pass
        layer.init_state(batch_size=1, device=torch.device("cpu"))
        with torch.no_grad():
            out1 = layer.step(x)

        # Reset and do again — should match
        layer.init_state(batch_size=1, device=torch.device("cpu"))
        with torch.no_grad():
            out2 = layer.step(x)

        assert torch.allclose(out1, out2, atol=1e-6)


class TestSSMCoreStep:
    def test_step_through_all_layers(self, streaming_config):
        from arcmind.models.ssm_core import SSMCore
        core = SSMCore(streaming_config)
        core.init_state(batch_size=2, device=torch.device("cpu"))
        x = torch.randn(2, streaming_config.d_model)
        out = core.step(x)
        assert out.shape == (2, streaming_config.d_model)


class TestArcMindStreaming:
    def test_step_output_shape(self, streaming_config):
        model = ArcMindModel(streaming_config)
        model.eval()
        model.init_streaming(batch_size=1)
        frame = torch.randn(1, streaming_config.num_sensor_channels)
        with torch.no_grad():
            action = model.step(frame)
        assert action.shape == (1, streaming_config.action_dim)

    def test_multiple_steps(self, streaming_config):
        model = ArcMindModel(streaming_config)
        model.eval()
        model.init_streaming(batch_size=1)

        actions = []
        for t in range(20):
            frame = torch.randn(1, streaming_config.num_sensor_channels)
            with torch.no_grad():
                action = model.step(frame)
            actions.append(action)
            assert action.shape == (1, streaming_config.action_dim)

        # Actions should vary over time (not constant)
        stacked = torch.stack(actions)
        assert stacked.std() > 1e-6, "Actions are constant — SSM state not evolving"

    def test_memory_written_at_decision_rate(self, streaming_config):
        model = ArcMindModel(streaming_config)
        model.eval()
        model.init_streaming(batch_size=1)

        stride = model.decision_stride
        # Step for 2x decision_stride steps
        for t in range(stride * 2):
            frame = torch.randn(1, streaming_config.num_sensor_channels)
            with torch.no_grad():
                model.step(frame)

        # Memory should have been written at step 0 and step stride
        assert model.memory.get_occupancy() == 2

    def test_init_streaming_resets_everything(self, streaming_config):
        model = ArcMindModel(streaming_config)
        model.eval()
        model.init_streaming(batch_size=1)

        # Run some steps
        for t in range(10):
            frame = torch.randn(1, streaming_config.num_sensor_channels)
            with torch.no_grad():
                model.step(frame)

        assert model.memory.get_occupancy() > 0

        # Re-init should clear everything
        model.init_streaming(batch_size=1)
        assert model.memory.get_occupancy() == 0
        assert model._step_counter == 0

    def test_step_matches_forward_first_frame(self, streaming_config):
        """First step() output should match forward() at t=0 with attention ablated.

        We ablate attention to isolate SSM streaming equivalence, which is the
        critical correctness property. The attention path processes differently
        in batch vs streaming mode by design (batch sees full sequence, stream
        sees one token at a time with memory).
        """
        streaming_config.ablate_attention = True
        torch.manual_seed(99)
        model = ArcMindModel(streaming_config)
        model.eval()

        frame = torch.randn(1, streaming_config.num_sensor_channels)

        # Batch forward on single frame
        model.reset_memory(batch_size=1)
        with torch.no_grad():
            batch_out = model(frame.unsqueeze(1), use_memory=False)  # (1, 1, action_dim)
            batch_action = batch_out[:, 0, :]  # (1, action_dim)

        # Streaming step on same frame
        model.init_streaming(batch_size=1)
        with torch.no_grad():
            step_action = model.step(frame)  # (1, action_dim)

        assert torch.allclose(batch_action, step_action, atol=1e-5), (
            f"Batch: {batch_action}\nStep:  {step_action}"
        )
