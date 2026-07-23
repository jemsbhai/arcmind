"""Export a fixture by executing the audited official Mamba-1 ``step`` method.

The exporter requires a local copy of the pinned upstream ``mamba_simple.py``.
It validates the source hash before execution and stubs only optional fused
operators. The official dependency-light PyTorch slow path produces every
expected output and cache in the fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch

MAMBA_VERSION = "2.2.6.post3"
MAMBA_AUDITED_COMMIT = "10b5d6358f27966f6a40e4bf0baa17a460688128"
MAMBA_SIMPLE_SHA256 = "a17e4c51b582dc0d4d690a649eba521cd0c1ee3dc8f0473a0967cdc9ec0874e3"
MAMBA_SOURCE_PATH = "mamba_ssm/modules/mamba_simple.py"
MAMBA_D_STATE = 16
MAMBA_D_CONV = 4
MAMBA_EXPAND = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_source_stubs() -> None:
    """Install the minimum modules needed to execute the official source."""

    def rearrange(values: torch.Tensor, pattern: str, **axes: int) -> torch.Tensor:
        del axes
        if pattern == "d 1 w -> d w":
            return values[:, 0, :]
        if pattern == "b d -> b d 1":
            return values[:, :, None]
        raise RuntimeError(f"Exporter stub does not implement rearrange pattern: {pattern}")

    def repeat(values: torch.Tensor, pattern: str, **axes: int) -> torch.Tensor:
        if pattern == "n -> d n":
            return values.unsqueeze(0).repeat(axes["d"], 1)
        raise RuntimeError(f"Exporter stub does not implement repeat pattern: {pattern}")

    einops = types.ModuleType("einops")
    einops.rearrange = rearrange
    einops.repeat = repeat
    sys.modules["einops"] = einops

    mamba_ssm = types.ModuleType("mamba_ssm")
    mamba_ssm.__path__ = []
    ops = types.ModuleType("mamba_ssm.ops")
    ops.__path__ = []
    triton = types.ModuleType("mamba_ssm.ops.triton")
    triton.__path__ = []
    selective_scan = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    selective_scan.selective_scan_fn = None
    selective_scan.mamba_inner_fn = None
    selective_update = types.ModuleType("mamba_ssm.ops.triton.selective_state_update")
    selective_update.selective_state_update = None
    layer_norm = types.ModuleType("mamba_ssm.ops.triton.layer_norm")
    layer_norm.RMSNorm = None
    layer_norm.layer_norm_fn = None
    layer_norm.rms_norm_fn = None
    causal_conv = types.ModuleType("causal_conv1d")
    causal_conv.causal_conv1d_fn = None
    causal_conv.causal_conv1d_update = None

    sys.modules.update(
        {
            "mamba_ssm": mamba_ssm,
            "mamba_ssm.ops": ops,
            "mamba_ssm.ops.triton": triton,
            "mamba_ssm.ops.selective_scan_interface": selective_scan,
            "mamba_ssm.ops.triton.selective_state_update": selective_update,
            "mamba_ssm.ops.triton.layer_norm": layer_norm,
            "causal_conv1d": causal_conv,
        }
    )


def _load_official_mamba(source_path: Path) -> type[torch.nn.Module]:
    actual_hash = _sha256(source_path)
    if actual_hash != MAMBA_SIMPLE_SHA256:
        raise ValueError(
            f"Official source hash mismatch: expected={MAMBA_SIMPLE_SHA256}, actual={actual_hash}"
        )
    _install_source_stubs()
    namespace: dict[str, Any] = {
        "__file__": str(source_path),
        "__name__": "audited_mamba_simple",
    }
    source = source_path.read_text(encoding="utf-8")
    exec(compile(source, str(source_path), "exec"), namespace)
    return namespace["Mamba"]


def _array(values: torch.Tensor) -> list[Any]:
    return values.detach().cpu().numpy().astype(np.float32).tolist()


def export_fixture(
    source_path: Path,
    output_path: Path,
    *,
    seed: int = 240723,
) -> None:
    """Execute the official slow step path and write a deterministic JSON fixture."""
    torch.manual_seed(seed)
    mamba_type = _load_official_mamba(source_path)
    hidden_size = 3
    batch_size = 2
    sequence_length = 5
    model = mamba_type(
        d_model=hidden_size,
        d_state=MAMBA_D_STATE,
        d_conv=MAMBA_D_CONV,
        expand=MAMBA_EXPAND,
        dt_rank="auto",
        use_fast_path=False,
    ).eval()

    generator = torch.Generator().manual_seed(seed + 1)
    hidden = torch.randn(
        batch_size,
        sequence_length,
        hidden_size,
        generator=generator,
    )
    convolution = torch.randn(
        batch_size,
        hidden_size * MAMBA_EXPAND,
        MAMBA_D_CONV,
        generator=generator,
    )
    ssm = torch.randn(
        batch_size,
        hidden_size * MAMBA_EXPAND,
        MAMBA_D_STATE,
        generator=generator,
    )
    initial_convolution = convolution.clone()
    initial_ssm = ssm.clone()
    outputs = []
    with torch.no_grad():
        for timestep in range(sequence_length):
            output, convolution, ssm = model.step(
                hidden[:, timestep : timestep + 1, :],
                convolution,
                ssm,
            )
            outputs.append(output[:, 0, :])
    output_sequence = torch.stack(outputs, dim=1)

    state_dict = model.state_dict()
    parameters = {
        "mamba.in_proj.kernel": _array(state_dict["in_proj.weight"].T),
        "mamba.conv1d.kernel": _array(state_dict["conv1d.weight"][:, 0, :]),
        "mamba.conv1d.bias": _array(state_dict["conv1d.bias"]),
        "mamba.x_proj.kernel": _array(state_dict["x_proj.weight"].T),
        "mamba.dt_proj.kernel": _array(state_dict["dt_proj.weight"].T),
        "mamba.dt_proj.bias": _array(state_dict["dt_proj.bias"]),
        "mamba.A_log": _array(state_dict["A_log"]),
        "mamba.D": _array(state_dict["D"]),
        "mamba.out_proj.kernel": _array(state_dict["out_proj.weight"].T),
    }
    payload = {
        "schema_version": 1,
        "provenance": {
            "repository": "https://github.com/state-spaces/mamba",
            "version": MAMBA_VERSION,
            "commit": MAMBA_AUDITED_COMMIT,
            "source_path": MAMBA_SOURCE_PATH,
            "source_sha256": MAMBA_SIMPLE_SHA256,
            "execution_path": "Mamba.step dependency-light PyTorch slow path",
        },
        "seed": seed,
        "configuration": {
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "d_state": MAMBA_D_STATE,
            "d_conv": MAMBA_D_CONV,
            "expand": MAMBA_EXPAND,
            "dt_rank": 1,
        },
        "parameters": parameters,
        "hidden": _array(hidden),
        "initial_state": {
            "convolution": _array(initial_convolution),
            "ssm": _array(initial_ssm),
        },
        "expected": {
            "outputs": _array(output_sequence),
            "final_convolution": _array(convolution),
            "final_ssm": _array(ssm),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "fixture": str(output_path.resolve()),
                "fixture_sha256": _sha256(output_path),
                "source_sha256": MAMBA_SIMPLE_SHA256,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/pobax/tests/fixtures/mamba1_official_step_v1.json"),
    )
    parser.add_argument("--seed", type=int, default=240723)
    args = parser.parse_args()
    export_fixture(args.source, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
