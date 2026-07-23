"""Export a deterministic PyTorch streaming fixture for the JAX parity check."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from arcmind import ArcMindConfig, ArcMindModel


def make_config() -> ArcMindConfig:
    """Small configuration that still exercises every architectural component."""
    return ArcMindConfig(
        num_sensor_channels=5,
        sensor_freq_hz=4.0,
        decision_freq_hz=2.0,
        d_model=8,
        num_ssm_layers=2,
        ssm_state_dim=3,
        ssm_conv_width=3,
        ssm_expand_factor=1,
        num_attn_layers=2,
        num_attn_heads=2,
        attn_window_size=3,
        num_memory_slots=4,
        memory_compress_ratio=2,
        action_dim=3,
        dropout=0.0,
    )


def export_fixture(output: Path, *, seed: int = 240723) -> None:
    """Write inputs, parameters, and PyTorch outputs to a compressed NumPy file."""
    torch.manual_seed(seed)
    config = make_config()
    model = ArcMindModel(config).eval()
    model.init_streaming(batch_size=2)

    generator = torch.Generator().manual_seed(seed + 1)
    inputs = torch.randn(
        2,
        14,
        config.num_sensor_channels,
        generator=generator,
    )
    with torch.no_grad():
        outputs = torch.stack(
            [model.step(inputs[:, timestep, :]) for timestep in range(inputs.shape[1])],
            dim=1,
        )

    serialized_config = asdict(config)
    serialized_config["decision_stride"] = model.decision_stride
    payload: dict[str, np.ndarray] = {
        "config_json": np.asarray(json.dumps(serialized_config)),
        "inputs": inputs.numpy(),
        "expected_outputs": outputs.numpy(),
    }
    payload.update(
        {
            f"param::{name}": value.detach().cpu().numpy()
            for name, value in model.state_dict().items()
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    print(
        json.dumps(
            {
                "fixture": str(output.resolve()),
                "seed": seed,
                "batch_size": inputs.shape[0],
                "sequence_length": inputs.shape[1],
                "parameter_tensors": len(model.state_dict()),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/parity/pytorch_fixture.npz"),
    )
    parser.add_argument("--seed", type=int, default=240723)
    args = parser.parse_args()
    export_fixture(args.output, seed=args.seed)


if __name__ == "__main__":
    main()
