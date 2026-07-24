"""Export an immutable differential fixture from the official Memory Traces code."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.pobax.model_registry import (
    MEMORY_TRACE_AUDITED_COMMIT,
    MEMORY_TRACE_DECAYS,
    MEMORY_TRACE_EXAMPLE_SHA256,
    MEMORY_TRACE_SOURCE_SHA256,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkout(path: Path) -> None:
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != MEMORY_TRACE_AUDITED_COMMIT:
        raise RuntimeError(
            f"official checkout must equal {MEMORY_TRACE_AUDITED_COMMIT}, found {commit}"
        )
    expected_hashes = {
        "traces/ppo.py": MEMORY_TRACE_SOURCE_SHA256,
        "examples/ppo_tmaze.py": MEMORY_TRACE_EXAMPLE_SHA256,
    }
    for relative, expected in expected_hashes.items():
        actual = _sha256(path / relative)
        if actual != expected:
            raise RuntimeError(
                f"official source hash drift for {relative}: expected={expected}, found={actual}"
            )


def _tolist(value: Any) -> Any:
    return np.asarray(value).tolist()


def export_fixture(official_checkout: Path, output: Path) -> None:
    """Run official Trace and ActorCritic code and save replay inputs."""

    _verify_checkout(official_checkout)
    sys.path.insert(0, str(official_checkout))

    import flax  # pylint: disable=import-outside-toplevel
    import jax  # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    from traces.ppo import ActorCritic, Trace  # pylint: disable=import-outside-toplevel

    class IdentityEncoding:
        num_actions = 3
        obs_dims = (1, 1, 2)
        act_dims = (1,)

        @staticmethod
        def encode(observation, action):
            return observation, jnp.asarray([action], dtype=observation.dtype)

    observations = jnp.asarray(
        [
            [[1.0, -2.0], [0.5, 3.0]],
            [[2.0, 4.0], [-1.0, 2.0]],
            [[-3.0, 1.0], [4.0, -2.0]],
            [[5.0, 0.25], [1.5, -0.5]],
        ],
        dtype=jnp.float32,
    )
    resets = jnp.asarray(
        [
            [True, True],
            [False, False],
            [True, False],
            [False, True],
        ],
        dtype=jnp.bool_,
    )
    previous_actions = jnp.asarray(
        [[-1, -1], [1, 2], [0, 1], [2, 0]],
        dtype=jnp.int32,
    )
    augmented_tail = jnp.asarray(
        [
            [[101.0, 102.0, 103.0, 104.0, 105.0], [111.0, 112.0, 113.0, 114.0, 115.0]],
            [[121.0, 122.0, 123.0, 124.0, 125.0], [131.0, 132.0, 133.0, 134.0, 135.0]],
            [[141.0, 142.0, 143.0, 144.0, 145.0], [151.0, 152.0, 153.0, 154.0, 155.0]],
            [[161.0, 162.0, 163.0, 164.0, 165.0], [171.0, 172.0, 173.0, 174.0, 175.0]],
        ],
        dtype=jnp.float32,
    )
    policy_inputs = jnp.concatenate([observations, augmented_tail], axis=-1)

    memory = Trace(IdentityEncoding(), lams=list(MEMORY_TRACE_DECAYS))
    actor_critic = ActorCritic(IdentityEncoding.num_actions)
    batch_size = observations.shape[1]
    state = jax.vmap(lambda _: memory.reset())(jnp.arange(batch_size))
    initial_features = jax.vmap(memory)(state)
    official_params = actor_critic.init(jax.random.key(1701), initial_features[0])

    trace_outputs = []
    feature_outputs = []
    logit_outputs = []
    value_outputs = []
    for observation, reset, previous_action in zip(
        observations,
        resets,
        previous_actions,
        strict=True,
    ):

        def reset_worker(worker_reset, worker_state):
            return jax.tree.map(
                lambda fresh, current: jax.lax.select(worker_reset, fresh, current),
                memory.reset(),
                worker_state,
            )

        state = jax.vmap(reset_worker)(reset, state)
        encoded_observation = observation.reshape((batch_size, 1, 1, 2))
        state = jax.vmap(memory.update)(state, encoded_observation, previous_action)
        features = jax.vmap(memory)(state)
        logits = jax.vmap(
            lambda item: actor_critic.apply(
                official_params,
                item,
                method=lambda module, value: module.actor(value),
            )
        )(features)
        values = jax.vmap(
            lambda item: actor_critic.apply(
                official_params,
                item,
                method="value",
            )
        )(features)[..., 0]
        trace_outputs.append(state[0].reshape((batch_size, len(MEMORY_TRACE_DECAYS), 2)))
        feature_outputs.append(features)
        logit_outputs.append(logits)
        value_outputs.append(values)

    source_params = official_params["params"]
    translated_params = {
        "actor.hidden.0.kernel": source_params["actor"]["layers_1"]["kernel"],
        "actor.hidden.0.bias": source_params["actor"]["layers_1"]["bias"],
        "actor.hidden.1.kernel": source_params["actor"]["layers_3"]["kernel"],
        "actor.hidden.1.bias": source_params["actor"]["layers_3"]["bias"],
        "actor.output.kernel": source_params["actor"]["layers_5"]["kernel"],
        "actor.output.bias": source_params["actor"]["layers_5"]["bias"],
        "critic.hidden.0.kernel": source_params["critic"]["layers_1"]["kernel"],
        "critic.hidden.0.bias": source_params["critic"]["layers_1"]["bias"],
        "critic.hidden.1.kernel": source_params["critic"]["layers_3"]["kernel"],
        "critic.hidden.1.bias": source_params["critic"]["layers_3"]["bias"],
        "critic.output.kernel": source_params["critic"]["layers_5"]["kernel"],
        "critic.output.bias": source_params["critic"]["layers_5"]["bias"],
    }
    payload = {
        "schema_version": 1,
        "provenance": {
            "repository": "https://github.com/onnoeberhard/memory-traces",
            "commit": MEMORY_TRACE_AUDITED_COMMIT,
            "source_hashes": {
                "traces/ppo.py": MEMORY_TRACE_SOURCE_SHA256,
                "examples/ppo_tmaze.py": MEMORY_TRACE_EXAMPLE_SHA256,
            },
            "execution_path": (
                "official Trace and ActorCritic with translated official-initialized weights"
            ),
            "jax_backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "flax_version": flax.__version__,
        },
        "configuration": {
            "input_dim": 7,
            "observation_dim": 2,
            "action_dim": 3,
            "hidden_size": 64,
            "decays": list(MEMORY_TRACE_DECAYS),
            "episodic": False,
            "actions": False,
            "trace_layout": "trace_major",
        },
        "policy_inputs": _tolist(policy_inputs),
        "resets": _tolist(resets),
        "previous_actions_for_official_source": _tolist(previous_actions),
        "parameters": {name: _tolist(value) for name, value in translated_params.items()},
        "expected": {
            "traces": _tolist(jnp.stack(trace_outputs)),
            "features": _tolist(jnp.stack(feature_outputs)),
            "logits": _tolist(jnp.stack(logit_outputs)),
            "values": _tolist(jnp.stack(value_outputs)),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{output} sha256={_sha256(output)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/pobax/tests/fixtures/memory_trace_official_v1.json"),
    )
    args = parser.parse_args()
    export_fixture(args.official_checkout.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
