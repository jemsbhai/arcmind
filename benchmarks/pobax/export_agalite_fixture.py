"""Export a differential fixture from the audited official AGaLiTe source."""

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
    AGALITE_A2C_SHA256,
    AGALITE_ACTOR_CRITIC_SHA256,
    AGALITE_AUDITED_COMMIT,
    AGALITE_FLATTEN_SHA256,
    AGALITE_HEADS_SHA256,
    AGALITE_LAYERS_SHA256,
    AGALITE_LICENSE_SHA256,
    AGALITE_MODEL_SHA256,
    AGALITE_REQUIREMENTS_SHA256,
    AGALITE_SEQUENCE_FACTORY_SHA256,
    AGALITE_TMAZE_CONFIG_SHA256,
)

_SOURCE_HASHES = {
    "LICENSE": AGALITE_LICENSE_SHA256,
    "requirements.txt": AGALITE_REQUIREMENTS_SHA256,
    "src/models/agalite/agalite.py": AGALITE_MODEL_SHA256,
    "src/models/agalite/layers.py": AGALITE_LAYERS_SHA256,
    "src/model_fns/seq_fns.py": AGALITE_SEQUENCE_FACTORY_SHA256,
    "src/models/actor_critic.py": AGALITE_ACTOR_CRITIC_SHA256,
    "src/model_fns/achead_fns.py": AGALITE_HEADS_SHA256,
    "src/model_fns/repr_fns.py": AGALITE_FLATTEN_SHA256,
    "src/agents/a2c.py": AGALITE_A2C_SHA256,
    "config/tmaze/arelit.yaml": AGALITE_TMAZE_CONFIG_SHA256,
}


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
    if commit != AGALITE_AUDITED_COMMIT:
        raise RuntimeError(
            f"official checkout must equal {AGALITE_AUDITED_COMMIT}, found {commit}"
        )
    for relative, expected in _SOURCE_HASHES.items():
        actual = _sha256(path / relative)
        if actual != expected:
            raise RuntimeError(
                f"official source hash drift for {relative}: "
                f"expected={expected}, found={actual}"
            )


def _tolist(value: Any) -> Any:
    return np.asarray(value).tolist()


def _translate_parameters(source_params: dict[str, Any], *, num_layers: int) -> dict:
    translated = {}
    sequence = source_params["seq_model"]
    for layer in range(num_layers):
        source = sequence[f"layer{layer + 1}"]
        target = f"layers.{layer}"
        if layer == 0:
            translated[f"{target}.embedding.kernel"] = source["emb_layer"]["kernel"]
            translated[f"{target}.embedding.bias"] = source["emb_layer"]["bias"]

        attention = source["AttentionORLiTLayer_0"]
        translated[f"{target}.attention.combined.kernel"] = attention[
            "linear_kqvbetagammas"
        ]["kernel"]
        translated[f"{target}.attention.combined.bias"] = attention[
            "linear_kqvbetagammas"
        ]["bias"]
        translated[f"{target}.attention.projections.kernel"] = attention[
            "linear_p1p2p3"
        ]["kernel"]
        translated[f"{target}.attention.projections.bias"] = attention[
            "linear_p1p2p3"
        ]["bias"]
        translated[f"{target}.attention.output.kernel"] = attention["project"][
            "kernel"
        ]
        translated[f"{target}.attention.output.bias"] = attention["project"]["bias"]

        for gate_index, gate_name in enumerate(("gate1", "gate2")):
            gate = source[f"GRUGatingUnit_{gate_index}"]
            for matrix_name in ("Wr", "Ur", "Wz", "Uz", "Wg", "Ug"):
                translated[f"{target}.{gate_name}.{matrix_name}.kernel"] = gate[
                    matrix_name
                ]["kernel"]
            translated[f"{target}.{gate_name}.bg"] = gate["bgp"]

        for norm_index, norm_name in enumerate(("norm1", "norm2")):
            norm = source[f"LayerNorm_{norm_index}"]
            translated[f"{target}.{norm_name}.scale"] = norm["scale"]
            translated[f"{target}.{norm_name}.bias"] = norm["bias"]
        translated[f"{target}.feedforward.input.kernel"] = source["Dense_0"][
            "kernel"
        ]
        translated[f"{target}.feedforward.input.bias"] = source["Dense_0"]["bias"]
        translated[f"{target}.feedforward.output.kernel"] = source["Dense_1"][
            "kernel"
        ]
        translated[f"{target}.feedforward.output.bias"] = source["Dense_1"]["bias"]

    translated["actor.hidden.kernel"] = source_params["actor"]["layers_0"]["kernel"]
    translated["actor.hidden.bias"] = source_params["actor"]["layers_0"]["bias"]
    translated["actor.output.kernel"] = source_params["actor"]["layers_2"]["kernel"]
    translated["actor.output.bias"] = source_params["actor"]["layers_2"]["bias"]
    translated["critic.hidden.kernel"] = source_params["critic"]["layers_0"][
        "kernel"
    ]
    translated["critic.hidden.bias"] = source_params["critic"]["layers_0"]["bias"]
    translated["critic.output.kernel"] = source_params["critic"]["layers_2"][
        "kernel"
    ]
    translated["critic.output.bias"] = source_params["critic"]["layers_2"]["bias"]
    return translated


def export_fixture(official_checkout: Path, output: Path) -> None:
    """Execute a tiny shape-equivalent official T-Maze vector policy."""

    _verify_checkout(official_checkout)
    sys.path.insert(0, str(official_checkout))

    import flax  # pylint: disable=import-outside-toplevel
    import jax  # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    from src.model_fns.achead_fns import (  # pylint: disable=import-outside-toplevel
        actor_model_discete,
        critic_model,
    )
    from src.model_fns.repr_fns import (  # pylint: disable=import-outside-toplevel
        flatten_repr_model,
    )
    from src.model_fns.seq_fns import (  # pylint: disable=import-outside-toplevel
        seq_model_agalite,
    )
    from src.models.actor_critic import (  # pylint: disable=import-outside-toplevel
        ActorCriticModel,
    )

    configuration = {
        "input_dim": 6,
        "observation_dim": 3,
        "action_dim": 3,
        "hidden_size": 4,
        "head_dim": 2,
        "feedforward_size": 4,
        "num_heads": 2,
        "eta": 2,
        "approximation_channels": 2,
        "num_layers": 2,
        "actor_hidden_size": 4,
        "critic_hidden_size": 4,
        "gate_bias": 2.0,
        "attention_epsilon": 1e-5,
        "layer_norm_epsilon": 1e-6,
    }
    observations = jnp.asarray(
        [
            [[0.25, -1.0, 2.0], [1.5, 0.5, -0.25]],
            [[-0.75, 1.25, 0.5], [0.0, -1.5, 2.5]],
            [[2.0, -0.5, 1.0], [-2.0, 0.75, 0.25]],
            [[0.5, 1.5, -1.0], [1.25, -0.75, 0.0]],
            [[-1.0, 0.25, 1.75], [0.5, 2.0, -1.25]],
        ],
        dtype=jnp.float32,
    )
    resets = jnp.asarray(
        [
            [True, True],
            [False, False],
            [True, False],
            [False, True],
            [False, False],
        ],
        dtype=jnp.bool_,
    )
    augmented_tail = jnp.asarray(
        [
            [[101.0, 102.0, 103.0], [111.0, 112.0, 113.0]],
            [[121.0, 122.0, 123.0], [131.0, 132.0, 133.0]],
            [[141.0, 142.0, 143.0], [151.0, 152.0, 153.0]],
            [[161.0, 162.0, 163.0], [171.0, 172.0, 173.0]],
            [[181.0, 182.0, 183.0], [191.0, 192.0, 193.0]],
        ],
        dtype=jnp.float32,
    )
    policy_inputs = jnp.concatenate([observations, augmented_tail], axis=-1)

    sequence_factory, sequence_initializer = seq_model_agalite(
        n_layers=configuration["num_layers"],
        d_model=configuration["hidden_size"],
        d_head=configuration["head_dim"],
        d_ffc=configuration["feedforward_size"],
        n_heads=configuration["num_heads"],
        eta=configuration["eta"],
        r=configuration["approximation_channels"],
        reset_hidden_on_terminate=True,
    )
    actor_critic = ActorCriticModel(
        flatten_repr_model(),
        sequence_factory,
        actor_model_discete(
            configuration["actor_hidden_size"],
            configuration["action_dim"],
        ),
        critic_model(configuration["critic_hidden_size"]),
    )
    initial_memory = sequence_initializer()
    official_params = actor_critic.init(
        jax.random.PRNGKey(4101),
        observations[:, 0],
        resets[:, 0],
        initial_memory,
    )

    logits_by_worker = []
    values_by_worker = []
    states_by_worker = []
    for worker in range(observations.shape[1]):
        logits, values, state = actor_critic.apply(
            official_params,
            observations[:, worker],
            resets[:, worker],
            sequence_initializer(),
        )
        logits_by_worker.append(logits)
        values_by_worker.append(values)
        states_by_worker.append(state)

    expected_state = {
        "tilde_key": jnp.stack(
            [
                jnp.stack(
                    [state[f"layer_{layer + 1}"][0] for layer in range(2)]
                )
                for state in states_by_worker
            ]
        ),
        "tilde_value": jnp.stack(
            [
                jnp.stack(
                    [state[f"layer_{layer + 1}"][1] for layer in range(2)]
                )
                for state in states_by_worker
            ]
        ),
        "normalizer": jnp.stack(
            [
                jnp.stack(
                    [state[f"layer_{layer + 1}"][2] for layer in range(2)]
                )
                for state in states_by_worker
            ]
        ),
        "tick": jnp.stack(
            [
                jnp.stack(
                    [state[f"layer_{layer + 1}"][3] for layer in range(2)]
                )
                for state in states_by_worker
            ]
        ),
    }
    translated_params = _translate_parameters(
        official_params["params"],
        num_layers=configuration["num_layers"],
    )
    payload = {
        "schema_version": 1,
        "provenance": {
            "repository": "https://github.com/subho406/agalite",
            "commit": AGALITE_AUDITED_COMMIT,
            "source_hashes": _SOURCE_HASHES,
            "source_variant": "released_tmaze_vector_policy_shape_equivalent_fixture",
            "execution_path": (
                "official Flax AGaLiTe, Flatten, ActorCriticModel, and tanh heads"
            ),
            "jax_backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "flax_version": flax.__version__,
        },
        "configuration": configuration,
        "policy_inputs": _tolist(policy_inputs),
        "resets": _tolist(resets),
        "parameters": {
            name: _tolist(value) for name, value in translated_params.items()
        },
        "expected": {
            "state": {name: _tolist(value) for name, value in expected_state.items()},
            "logits": _tolist(jnp.stack(logits_by_worker, axis=1)),
            "values": _tolist(jnp.stack(values_by_worker, axis=1)),
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
        default=Path("benchmarks/pobax/tests/fixtures/agalite_official_v1.json"),
    )
    args = parser.parse_args()
    export_fixture(args.official_checkout.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
