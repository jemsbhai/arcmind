"""Parameter-matched sequence baselines for local mechanistic benchmarks."""

from collections.abc import Callable, Iterable

import torch
import torch.nn as nn

from arcmind import ArcMindConfig, ArcMindModel


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class MemorylessMLP(nn.Module):
    """Per-frame MLP with no temporal state."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class RecurrentBaseline(nn.Module):
    """GRU or LSTM sequence classifier evaluated at every timestep."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        *,
        cell: str,
    ):
        super().__init__()
        recurrent_type = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.recurrent = recurrent_type(
            hidden_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.input_projection(inputs))
        hidden, _ = self.recurrent(hidden)
        return self.output(hidden)


class CausalTransformerBaseline(nn.Module):
    """Causal full-attention baseline with learned absolute positions."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int,
        *,
        max_length: int,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.empty(1, max_length, d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        length = inputs.shape[1]
        hidden = self.input_projection(inputs) + self.position[:, :length, :]
        causal_mask = torch.triu(
            torch.ones(length, length, device=inputs.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.encoder(hidden, mask=causal_mask)
        return self.output(hidden)


def build_arcmind(
    input_dim: int,
    output_dim: int,
    *,
    sensor_stride: int,
    exact_recall_window: int,
    variant: str,
    d_model: int = 32,
) -> ArcMindModel:
    """Construct the registered compact ArcMind diagnostic variants."""
    config = ArcMindConfig(
        num_sensor_channels=input_dim,
        sensor_freq_hz=float(sensor_stride),
        decision_freq_hz=1.0,
        d_model=d_model,
        num_ssm_layers=2,
        ssm_state_dim=4,
        ssm_conv_width=4,
        ssm_expand_factor=2,
        num_attn_layers=1,
        num_attn_heads=2,
        attn_window_size=exact_recall_window,
        num_memory_slots=16,
        memory_compress_ratio=2,
        action_dim=output_dim,
        dropout=0.0,
    )
    if variant == "arcmind_ssm_only":
        config.ablate_attention = True
    elif variant == "arcmind_unordered":
        config.ablate_temporal_encoding = True
    elif variant != "arcmind":
        raise ValueError(f"unknown ArcMind variant: {variant}")
    return ArcMindModel(config)


def _best_width(
    factory: Callable[[int], nn.Module],
    target_parameters: int,
    widths: Iterable[int],
) -> int:
    candidates = []
    for width in widths:
        parameters = count_parameters(factory(width))
        candidates.append((abs(parameters - target_parameters), width))
    return min(candidates)[1]


def build_parameter_matched_baseline(
    name: str,
    *,
    input_dim: int,
    output_dim: int,
    sequence_length: int,
    target_parameters: int,
) -> nn.Module:
    """Build the nearest-width baseline to a target trainable parameter count."""
    if name == "memoryless_mlp":
        def factory(width: int) -> nn.Module:
            return MemorylessMLP(input_dim, output_dim, width)

        width = _best_width(factory, target_parameters, range(8, 513))
    elif name in {"gru", "lstm"}:
        def factory(width: int) -> nn.Module:
            return RecurrentBaseline(
                input_dim,
                output_dim,
                width,
                cell=name,
            )

        width = _best_width(factory, target_parameters, range(8, 257))
    elif name == "causal_transformer":
        def factory(width: int) -> nn.Module:
            return CausalTransformerBaseline(
                input_dim,
                output_dim,
                width,
                max_length=sequence_length,
            )

        width = _best_width(factory, target_parameters, range(8, 257, 4))
    else:
        raise ValueError(f"unknown baseline: {name}")
    return factory(width)
