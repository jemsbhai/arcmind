"""Check JIT-compiled JAX streaming inference against a PyTorch fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.pobax.arcmind_reference import (
    ReferenceConfig,
    arcmind_step,
    init_stream_state,
)


def check_fixture(
    fixture_path: Path,
    *,
    absolute_tolerance: float = 5e-5,
    relative_tolerance: float = 5e-5,
) -> dict[str, float | int | str]:
    """Run the reference on GPU and fail if outputs exceed the tolerance."""
    with np.load(fixture_path, allow_pickle=False) as fixture:
        config = ReferenceConfig.from_mapping(
            json.loads(str(fixture["config_json"].item()))
        )
        inputs = jnp.asarray(fixture["inputs"])
        expected = np.asarray(fixture["expected_outputs"])
        params = {
            name.removeprefix("param::"): jnp.asarray(fixture[name])
            for name in fixture.files
            if name.startswith("param::")
        }

    initial_state = init_stream_state(config, inputs.shape[0], dtype=inputs.dtype)

    @jax.jit
    def run_sequence(sequence):
        def scan_step(state, sensor_frame):
            return arcmind_step(params, state, sensor_frame, config)

        _, outputs = jax.lax.scan(
            scan_step,
            initial_state,
            jnp.swapaxes(sequence, 0, 1),
        )
        return jnp.swapaxes(outputs, 0, 1)

    actual_device = run_sequence(inputs)
    actual_device.block_until_ready()
    actual = np.asarray(actual_device)
    difference = np.abs(actual - expected)
    max_absolute_error = float(difference.max())
    mean_absolute_error = float(difference.mean())
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    return {
        "fixture": str(fixture_path.resolve()),
        "backend": jax.default_backend(),
        "devices": len(jax.devices()),
        "values_compared": int(actual.size),
        "max_absolute_error": max_absolute_error,
        "mean_absolute_error": mean_absolute_error,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--atol", type=float, default=5e-5)
    parser.add_argument("--rtol", type=float, default=5e-5)
    args = parser.parse_args()
    print(
        json.dumps(
            check_fixture(
                args.fixture,
                absolute_tolerance=args.atol,
                relative_tolerance=args.rtol,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
