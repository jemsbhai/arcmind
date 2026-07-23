"""Validate the pinned POBAX source, environments, and JAX accelerator."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pobax
from jax import random
from pobax.envs import get_env

PINNED_POBAX_COMMIT = "a5e1d62d14e4efe783885b9d4f19cffa2a568eec"


def source_commit(distribution_name: str) -> str:
    """Read the immutable VCS revision recorded by a direct-reference install."""
    distribution = importlib.metadata.distribution(distribution_name)
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError(
            f"{distribution_name} was not installed from the pinned VCS reference"
        )
    direct_url = json.loads(direct_url_text)
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if not commit:
        raise RuntimeError(
            f"{distribution_name} installation does not record a VCS commit"
        )
    return str(commit)


def smoke_environment(*, require_gpu: bool) -> dict[str, object]:
    """Reset and step the two pilot environments."""
    backend = jax.default_backend()
    if require_gpu and backend != "gpu":
        raise RuntimeError(f"Expected JAX GPU backend, found {backend!r}")

    commit = source_commit("pobax")
    if commit != PINNED_POBAX_COMMIT:
        raise RuntimeError(
            f"POBAX commit drift: expected {PINNED_POBAX_COMMIT}, found {commit}"
        )

    environments: dict[str, object] = {}
    for index, environment_name in enumerate(("simple_chain", "tmaze_10")):
        key = random.PRNGKey(700 + index)
        environment, parameters = get_env(
            environment_name,
            key,
            num_envs=2,
        )
        observation, state = environment.reset(random.split(key, 2), parameters)
        step_key = random.PRNGKey(800 + index)
        next_observation, _, reward, done, _ = environment.step(
            random.split(step_key, 2),
            state,
            jnp.zeros((2,), dtype=jnp.int32),
            parameters,
        )
        jax.block_until_ready((next_observation, reward, done))
        environments[environment_name] = {
            "observation_shape": list(observation.obs.shape),
            "next_observation_shape": list(next_observation.obs.shape),
            "reward": reward.tolist(),
            "done": done.tolist(),
        }

    return {
        "jax": jax.__version__,
        "jax_backend": backend,
        "jax_devices": [str(device) for device in jax.devices()],
        "pobax": importlib.metadata.version("pobax"),
        "pobax_commit": commit,
        "pobax_source": str(Path(pobax.__file__).resolve()),
        "environments": environments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(smoke_environment(require_gpu=args.require_gpu), indent=2))


if __name__ == "__main__":
    main()
