"""Source-audited POPGym positional MLP for the shared POBAX harness.

The implementation follows ``proroklab/popgym`` commit
``410d5aa626dae8024f498354d8781a0d1870c399``:

* ``popgym/baselines/models/embeddings.py`` for the sinusoidal encoding
* ``popgym/baselines/ray_models/base_model.py`` for projection, nonparametric
  layer normalization, clipped embedding strength, and feature blending
* ``popgym/baselines/ray_models/ray_mlp.py`` for the two-layer MLP

The POPGym policy has separate actor and critic feature networks. This adapter
instead shares the positional MLP feature between the common linear actor and
critic heads, matching the policy-core interface used by the POBAX harness.
Its only state is an episode-position counter. It stores no observation
history and therefore remains a no-learned-memory control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp

Array = jax.Array
Params = Mapping[str, Array]

POPGYM_POSITIONAL_MLP_REFERENCE = {
    "repository": "https://github.com/proroklab/popgym",
    "audited_commit": "410d5aa626dae8024f498354d8781a0d1870c399",
    "relationship": "shared-input and shared-head policy adaptation",
}

_LAYER_NORM_EPSILON = 1e-5
_LEAKY_RELU_SLOPE = 0.01


def _xavier(key: Array, input_features: int, output_features: int) -> Array:
    bound = jnp.sqrt(6.0 / (input_features + output_features))
    return jax.random.uniform(
        key,
        (input_features, output_features),
        minval=-bound,
        maxval=bound,
    )


def _linear(params: Params, prefix: str, values: Array) -> Array:
    return values @ params[f"{prefix}.kernel"] + params[f"{prefix}.bias"]


def _layer_norm(values: Array) -> Array:
    """Match POPGym's non-affine feature-map layer normalization."""
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + _LAYER_NORM_EPSILON)


def sinusoidal_position_encoding(
    positions: Array,
    *,
    hidden_size: int,
    max_length: int,
    dtype: jnp.dtype = jnp.float32,
) -> Array:
    """Return POPGym's fixed hidden-width sine and cosine encoding.

    For hidden coordinate ``2i``, POPGym uses
    ``sin(t * exp(-log(L) * 2i / d))``. Coordinate ``2i + 1`` uses cosine
    with the same frequency. Here ``L`` is ``max_length`` and ``d`` is
    ``hidden_size``.

    POPGym materializes a table of length ``L``. This equivalent functional
    form avoids storing that non-trainable table. Registered callers are still
    expected to keep episode positions below ``max_length``.
    """
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    output_dtype = jnp.dtype(dtype)
    positions = jnp.asarray(positions, dtype=output_dtype)
    coordinates = jnp.arange(hidden_size, dtype=output_dtype)
    paired_even_coordinates = 2.0 * jnp.floor(coordinates / 2.0)
    frequencies = jnp.exp(
        -jnp.log(jnp.asarray(max_length, dtype=output_dtype))
        * paired_even_coordinates
        / jnp.asarray(hidden_size, dtype=output_dtype)
    )
    angles = positions[..., None] * frequencies
    is_even = jnp.remainder(jnp.arange(hidden_size), 2) == 0
    return jnp.where(is_even, jnp.sin(angles), jnp.cos(angles))


@dataclass(frozen=True)
class PositionalMLPPolicyCore:
    """Reset-aware POPGym positional MLP actor-critic policy core.

    Args:
        input_dim: Width of the shared augmented policy input.
        action_dim: Number of discrete actor logits.
        hidden_size: Width of the encoding and shared MLP feature.
        max_length: Maximum registered episode length and sinusoid period base.
    """

    input_dim: int
    action_dim: int
    hidden_size: int
    max_length: int

    def __post_init__(self) -> None:
        dimensions = {
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "hidden_size": self.hidden_size,
            "max_length": self.max_length,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    def initialize(self, key: Array) -> dict[str, Array]:
        """Initialize the source-structured MLP and common policy heads."""
        feature_key, hidden_0_key, hidden_1_key, actor_key, critic_key = jax.random.split(key, 5)
        return {
            "feature_map.kernel": _xavier(
                feature_key,
                self.input_dim,
                self.hidden_size,
            ),
            "feature_map.bias": jnp.zeros((self.hidden_size,)),
            "embedding.alpha": jnp.asarray(0.5, dtype=jnp.float32),
            "hidden.0.kernel": _xavier(
                hidden_0_key,
                self.hidden_size,
                self.hidden_size,
            ),
            "hidden.0.bias": jnp.zeros((self.hidden_size,)),
            "hidden.1.kernel": _xavier(
                hidden_1_key,
                self.hidden_size,
                self.hidden_size,
            ),
            "hidden.1.bias": jnp.zeros((self.hidden_size,)),
            "actor.kernel": 0.01 * _xavier(actor_key, self.hidden_size, self.action_dim),
            "actor.bias": jnp.zeros((self.action_dim,)),
            "critic.kernel": _xavier(critic_key, self.hidden_size, 1),
            "critic.bias": jnp.zeros((1,)),
        }

    @staticmethod
    def initial_state(
        batch_size: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> Array:
        """Return int32 episode-position counters and no observation history."""
        del dtype
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return jnp.zeros((batch_size,), dtype=jnp.int32)

    def encode_input(
        self,
        params: Params,
        policy_input: Array,
        positions: Array,
    ) -> Array:
        """Project and blend policy input with its fixed positional encoding."""
        projected = _layer_norm(_linear(params, "feature_map", policy_input))
        encoding = sinusoidal_position_encoding(
            positions,
            hidden_size=self.hidden_size,
            max_length=self.max_length,
            dtype=projected.dtype,
        )
        alpha = jnp.clip(
            jnp.asarray(params["embedding.alpha"], dtype=projected.dtype),
            0.0,
            1.0,
        )
        return (1.0 - alpha) * projected + alpha * encoding

    def _features(
        self,
        params: Params,
        policy_input: Array,
        positions: Array,
    ) -> Array:
        encoded = self.encode_input(params, policy_input, positions)
        features = jax.nn.leaky_relu(
            _linear(params, "hidden.0", encoded),
            negative_slope=_LEAKY_RELU_SLOPE,
        )
        return jax.nn.leaky_relu(
            _linear(params, "hidden.1", features),
            negative_slope=_LEAKY_RELU_SLOPE,
        )

    @staticmethod
    def _heads(params: Params, features: Array) -> tuple[Array, Array]:
        logits = _linear(params, "actor", features)
        values = _linear(params, "critic", features)[..., 0]
        return logits, values

    def step(
        self,
        params: Params,
        state: Array,
        policy_input: Array,
        reset: Array,
    ) -> tuple[Array, Array, Array]:
        """Apply one position-aware step with independent episode resets.

        A reset is applied before the current output, so the first observation
        of every episode receives position zero. The returned state is the
        next position.
        """
        state = jnp.asarray(state, dtype=jnp.int32)
        reset = jnp.asarray(reset, dtype=jnp.bool_)
        positions = jnp.where(reset, jnp.zeros_like(state), state)
        features = self._features(params, policy_input, positions)
        logits, values = self._heads(params, features)
        return positions + jnp.int32(1), logits, values

    def apply_sequence(
        self,
        params: Params,
        state: Array,
        policy_inputs: Array,
        resets: Array,
    ) -> tuple[Array, Array, Array]:
        """Apply the positional MLP over a time-major reset-aware sequence."""

        def scan_step(carry, inputs):
            policy_input, reset = inputs
            new_carry, logits, values = self.step(
                params,
                carry,
                policy_input,
                reset,
            )
            return new_carry, (logits, values)

        new_state, (logits, values) = jax.lax.scan(
            scan_step,
            state,
            (policy_inputs, resets),
        )
        return new_state, logits, values

    @staticmethod
    def count_parameters(params: Params) -> int:
        """Count trainable scalars, including alpha and both common heads."""
        return sum(int(value.size) for value in jax.tree.leaves(params))


def match_positional_mlp_hidden_size(
    *,
    target_parameters: int,
    input_dim: int,
    action_dim: int,
    maximum_width: int = 4096,
) -> int:
    """Find the globally closest positive hidden width by scalar count."""
    arguments = {
        "target_parameters": target_parameters,
        "input_dim": input_dim,
        "action_dim": action_dim,
        "maximum_width": maximum_width,
    }
    for name, value in arguments.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    def parameter_count(hidden_size: int) -> int:
        return (
            input_dim * hidden_size
            + 2 * hidden_size * hidden_size
            + hidden_size * action_dim
            + 4 * hidden_size
            + action_dim
            + 2
        )

    return min(
        range(1, maximum_width + 1),
        key=lambda width: abs(parameter_count(width) - target_parameters),
    )
