"""Source-audited AGaLiTe policy cores for the shared POBAX harness.

This module is a modified JAX port of the AGaLiTe vector-policy path released
under Apache-2.0 at commit
``101acbecc121a258ad8f7e58e2f782f546674979``:

https://github.com/subho406/agalite

The executable source, rather than the differing finite-channel equations in
the paper, defines the recurrence implemented here. Modifications include a
batched step interface, explicit parameter dictionaries, a frozen LayerNorm
epsilon, and integration with ArcMind's shared PPO policy-core contract. See
``THIRD_PARTY_NOTICES.md`` and ``licenses/AGALITE-APACHE-2.0.txt``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array
Params = Mapping[str, Array]

AGALITE_NUM_LAYERS = 4
AGALITE_MODEL_SIZE = 128
AGALITE_HEAD_SIZE = 64
AGALITE_FEEDFORWARD_SIZE = 128
AGALITE_NUM_HEADS = 4
AGALITE_ETA = 4
AGALITE_APPROXIMATION_CHANNELS = 2
AGALITE_GATE_BIAS = 2.0
AGALITE_ATTENTION_EPSILON = 1e-5
# The upstream requirements are unpinned and call flax.linen.LayerNorm()
# without an epsilon. Freeze the Flax default used by the audited operational
# fixture so future dependency upgrades cannot change benchmark semantics.
AGALITE_LAYER_NORM_EPSILON = 1e-6
AGALITE_SOURCE_ACTOR_HIDDEN_SIZE = 128
AGALITE_SOURCE_CRITIC_HIDDEN_SIZE = 128


class AGaLiTeState(NamedTuple):
    """Per-worker, per-layer finite-channel attention memory."""

    tilde_key: Array
    tilde_value: Array
    normalizer: Array
    tick: Array


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _orthogonal(
    key: Array,
    input_features: int,
    output_features: int,
    *,
    gain: float = math.sqrt(2.0),
) -> Array:
    return jax.nn.initializers.orthogonal(gain)(
        key,
        (input_features, output_features),
        jnp.float32,
    )


def _xavier(key: Array, input_features: int, output_features: int) -> Array:
    bound = jnp.sqrt(6.0 / (input_features + output_features))
    return jax.random.uniform(
        key,
        (input_features, output_features),
        minval=-bound,
        maxval=bound,
    )


def _linear(params: Params, prefix: str, values: Array) -> Array:
    result = values @ params[f"{prefix}.kernel"]
    bias_name = f"{prefix}.bias"
    if bias_name in params:
        result = result + params[bias_name]
    return result


def _layer_norm(
    values: Array,
    scale: Array,
    bias: Array,
    *,
    epsilon: float,
) -> Array:
    """Flax-compatible feature LayerNorm with an explicit frozen epsilon."""

    input_dtype = values.dtype
    promoted = values.astype(jnp.float32)
    mean = jnp.mean(promoted, axis=-1, keepdims=True)
    mean_square = jnp.mean(jnp.square(promoted), axis=-1, keepdims=True)
    variance = jnp.maximum(0.0, mean_square - jnp.square(mean))
    normalized = (promoted - mean) * jax.lax.rsqrt(variance + epsilon)
    return (normalized * scale + bias).astype(input_dtype)


def _gru_gate(params: Params, prefix: str, residual: Array, update: Array) -> Array:
    reset_gate = jax.nn.sigmoid(
        _linear(params, f"{prefix}.Wr", update)
        + _linear(params, f"{prefix}.Ur", residual)
    )
    update_gate = jax.nn.sigmoid(
        _linear(params, f"{prefix}.Wz", update)
        + _linear(params, f"{prefix}.Uz", residual)
        - params[f"{prefix}.bg"]
    )
    candidate = jnp.tanh(
        _linear(params, f"{prefix}.Wg", update)
        + _linear(params, f"{prefix}.Ug", reset_gate * residual)
    )
    return (1.0 - update_gate) * residual + update_gate * candidate


def _initialize_backbone(
    key: Array,
    *,
    input_dim: int,
    hidden_size: int,
    head_dim: int,
    feedforward_size: int,
    num_heads: int,
    eta: int,
    num_layers: int,
    gate_bias: float,
) -> tuple[dict[str, Array], Array]:
    per_layer_random_keys = 17
    random_key_count = num_layers * per_layer_random_keys + 1
    keys = iter(jax.random.split(key, random_key_count + 1))
    params: dict[str, Array] = {}

    params["layers.0.embedding.kernel"] = _orthogonal(
        next(keys),
        input_dim,
        hidden_size,
    )
    params["layers.0.embedding.bias"] = jnp.zeros((hidden_size,))

    for layer in range(num_layers):
        prefix = f"layers.{layer}"
        params[f"{prefix}.norm1.scale"] = jnp.ones((hidden_size,))
        params[f"{prefix}.norm1.bias"] = jnp.zeros((hidden_size,))
        params[f"{prefix}.attention.combined.kernel"] = _orthogonal(
            next(keys),
            hidden_size,
            5 * num_heads * head_dim,
        )
        params[f"{prefix}.attention.combined.bias"] = jnp.zeros(
            (5 * num_heads * head_dim,)
        )
        params[f"{prefix}.attention.projections.kernel"] = _orthogonal(
            next(keys),
            hidden_size,
            3 * num_heads * eta,
        )
        params[f"{prefix}.attention.projections.bias"] = jnp.zeros(
            (3 * num_heads * eta,)
        )
        params[f"{prefix}.attention.output.kernel"] = _orthogonal(
            next(keys),
            num_heads * head_dim,
            hidden_size,
        )
        params[f"{prefix}.attention.output.bias"] = jnp.zeros((hidden_size,))

        for gate_name in ("gate1", "gate2"):
            gate_prefix = f"{prefix}.{gate_name}"
            for matrix_name in ("Wr", "Ur", "Wz", "Uz", "Wg", "Ug"):
                params[f"{gate_prefix}.{matrix_name}.kernel"] = _orthogonal(
                    next(keys),
                    hidden_size,
                    hidden_size,
                )
            params[f"{gate_prefix}.bg"] = jnp.full((hidden_size,), gate_bias)

        params[f"{prefix}.norm2.scale"] = jnp.ones((hidden_size,))
        params[f"{prefix}.norm2.bias"] = jnp.zeros((hidden_size,))
        params[f"{prefix}.feedforward.input.kernel"] = _orthogonal(
            next(keys),
            hidden_size,
            feedforward_size,
        )
        params[f"{prefix}.feedforward.input.bias"] = jnp.zeros(
            (feedforward_size,)
        )
        params[f"{prefix}.feedforward.output.kernel"] = _orthogonal(
            next(keys),
            feedforward_size,
            hidden_size,
        )
        params[f"{prefix}.feedforward.output.bias"] = jnp.zeros((hidden_size,))

    return params, next(keys)


def _initial_state(
    batch_size: int,
    *,
    num_layers: int,
    approximation_channels: int,
    num_heads: int,
    eta: int,
    head_dim: int,
    dtype: jnp.dtype,
) -> AGaLiTeState:
    _positive_integer(batch_size, name="batch_size")
    feature_dim = eta * head_dim
    return AGaLiTeState(
        tilde_key=jnp.zeros(
            (
                batch_size,
                num_layers,
                approximation_channels,
                num_heads,
                feature_dim,
            ),
            dtype=dtype,
        ),
        tilde_value=jnp.zeros(
            (
                batch_size,
                num_layers,
                approximation_channels,
                num_heads,
                head_dim,
            ),
            dtype=dtype,
        ),
        normalizer=jnp.zeros(
            (batch_size, num_layers, num_heads, feature_dim),
            dtype=dtype,
        ),
        tick=jnp.ones((batch_size, num_layers, 1), dtype=dtype),
    )


def _backbone_step(
    params: Params,
    state: AGaLiTeState,
    values: Array,
    reset: Array,
    *,
    hidden_size: int,
    head_dim: int,
    feedforward_size: int,
    num_heads: int,
    eta: int,
    approximation_channels: int,
    num_layers: int,
    attention_epsilon: float,
    layer_norm_epsilon: float,
) -> tuple[AGaLiTeState, Array]:
    del feedforward_size
    reset = jnp.asarray(reset, dtype=jnp.bool_)
    layer_input = values
    new_tilde_keys = []
    new_tilde_values = []
    new_normalizers = []
    new_ticks = []
    frequencies = jnp.linspace(
        -jnp.pi,
        jnp.pi,
        approximation_channels,
        dtype=values.dtype,
    )

    for layer in range(num_layers):
        prefix = f"layers.{layer}"
        if layer == 0:
            encoded = jax.nn.relu(
                _linear(params, f"{prefix}.embedding", layer_input)
            )
        else:
            encoded = layer_input

        normalized = _layer_norm(
            encoded,
            params[f"{prefix}.norm1.scale"],
            params[f"{prefix}.norm1.bias"],
            epsilon=layer_norm_epsilon,
        )
        combined = _linear(
            params,
            f"{prefix}.attention.combined",
            normalized,
        ).reshape((normalized.shape[0], num_heads, 5 * head_dim))
        raw_key, raw_query, raw_value, raw_beta, raw_gamma = jnp.split(
            combined,
            5,
            axis=-1,
        )
        projections = _linear(
            params,
            f"{prefix}.attention.projections",
            normalized,
        ).reshape((normalized.shape[0], num_heads, 3 * eta))
        projection_key, projection_query, projection_gamma = jnp.split(
            projections,
            3,
            axis=-1,
        )

        key_features = jnp.einsum(
            "bhd,bhe->bhed",
            jax.nn.relu(raw_key),
            jax.nn.relu(projection_key),
        ).reshape((normalized.shape[0], num_heads, eta * head_dim))
        query_features = jnp.einsum(
            "bhd,bhe->bhed",
            jax.nn.relu(raw_query),
            jax.nn.relu(projection_query),
        ).reshape((normalized.shape[0], num_heads, eta * head_dim))
        gamma = jnp.einsum(
            "bhd,bhe->bhed",
            jax.nn.sigmoid(raw_gamma),
            jax.nn.sigmoid(projection_gamma),
        ).reshape((normalized.shape[0], num_heads, eta * head_dim))
        beta = jax.nn.sigmoid(raw_beta)

        tick = state.tick[:, layer]
        used_tick = tick + 1.0
        oscillation = jnp.cos(used_tick * frequencies[None, :])
        gated_key = key_features * gamma
        channel_key = (
            gated_key[:, None, :, :]
            * oscillation[:, :, None, None]
        )
        gated_value = raw_value * beta
        channel_value = (
            gated_value[:, None, :, :]
            * oscillation[:, :, None, None]
        )

        keep = 1.0 - reset.astype(values.dtype)
        key_discount = (1.0 - gamma) * keep[:, None, None]
        value_discount = (1.0 - beta) * keep[:, None, None]
        tilde_key = (
            state.tilde_key[:, layer] * key_discount[:, None, :, :]
            + channel_key
        )
        tilde_value = (
            state.tilde_value[:, layer] * value_discount[:, None, :, :]
            + channel_value
        )
        normalizer = (
            state.normalizer[:, layer] * key_discount
            + gated_key
        )

        key_query = jnp.einsum(
            "brhd,bhd->brh",
            tilde_key,
            query_features,
        )
        numerator = jnp.sum(
            tilde_value * key_query[..., None],
            axis=1,
        )
        denominator = jnp.einsum(
            "bhd,bhd->bh",
            normalizer,
            query_features,
        )
        attention = numerator / (
            2.0
            * approximation_channels
            * denominator[..., None]
            + attention_epsilon
        )
        attention = _linear(
            params,
            f"{prefix}.attention.output",
            attention.reshape((normalized.shape[0], num_heads * head_dim)),
        )
        attention = jax.nn.relu(attention)
        gated_attention = _gru_gate(
            params,
            f"{prefix}.gate1",
            encoded,
            attention,
        )
        normalized_feedforward = _layer_norm(
            gated_attention,
            params[f"{prefix}.norm2.scale"],
            params[f"{prefix}.norm2.bias"],
            epsilon=layer_norm_epsilon,
        )
        feedforward = jax.nn.relu(
            _linear(
                params,
                f"{prefix}.feedforward.input",
                normalized_feedforward,
            )
        )
        feedforward = jax.nn.relu(
            _linear(
                params,
                f"{prefix}.feedforward.output",
                feedforward,
            )
        )
        layer_input = _gru_gate(
            params,
            f"{prefix}.gate2",
            gated_attention,
            feedforward,
        )

        new_tilde_keys.append(tilde_key)
        new_tilde_values.append(tilde_value)
        new_normalizers.append(normalizer)
        new_ticks.append(used_tick)

    new_state = AGaLiTeState(
        tilde_key=jnp.stack(new_tilde_keys, axis=1),
        tilde_value=jnp.stack(new_tilde_values, axis=1),
        normalizer=jnp.stack(new_normalizers, axis=1),
        tick=jnp.stack(new_ticks, axis=1),
    )
    if layer_input.shape[-1] != hidden_size:  # pragma: no cover - constructor guards
        raise AssertionError("AGaLiTe backbone produced the wrong feature size")
    return new_state, layer_input


def agalite_parameter_count(
    *,
    input_dim: int,
    action_dim: int,
    hidden_size: int,
    head_dim: int,
    feedforward_size: int,
    num_heads: int,
    eta: int,
    num_layers: int,
    source_actor_critic: bool,
    actor_hidden_size: int = AGALITE_SOURCE_ACTOR_HIDDEN_SIZE,
    critic_hidden_size: int = AGALITE_SOURCE_CRITIC_HIDDEN_SIZE,
) -> int:
    """Return the exact trainable scalar count for either AGaLiTe lane."""

    for name, value in (
        ("input_dim", input_dim),
        ("action_dim", action_dim),
        ("hidden_size", hidden_size),
        ("head_dim", head_dim),
        ("feedforward_size", feedforward_size),
        ("num_heads", num_heads),
        ("eta", eta),
        ("num_layers", num_layers),
    ):
        _positive_integer(value, name=name)
    per_layer = (
        12 * hidden_size * hidden_size
        + 6 * hidden_size * num_heads * head_dim
        + 3 * hidden_size * num_heads * eta
        + 2 * hidden_size * feedforward_size
        + 5 * num_heads * head_dim
        + 3 * num_heads * eta
        + 8 * hidden_size
        + feedforward_size
    )
    backbone = input_dim * hidden_size + hidden_size + num_layers * per_layer
    if source_actor_critic:
        _positive_integer(actor_hidden_size, name="actor_hidden_size")
        _positive_integer(critic_hidden_size, name="critic_hidden_size")
        actor = (
            hidden_size * actor_hidden_size
            + actor_hidden_size
            + actor_hidden_size * action_dim
            + action_dim
        )
        critic = (
            hidden_size * critic_hidden_size
            + critic_hidden_size
            + critic_hidden_size
            + 1
        )
        return backbone + actor + critic
    return backbone + hidden_size * action_dim + action_dim + hidden_size + 1


def match_agalite_hidden_size(
    *,
    target_parameters: int,
    input_dim: int,
    action_dim: int,
) -> int:
    """Return the globally closest even shared-lane width."""

    _positive_integer(target_parameters, name="target_parameters")
    _positive_integer(input_dim, name="input_dim")
    _positive_integer(action_dim, name="action_dim")

    previous_width: int | None = None
    previous_count: int | None = None
    width = 2
    while True:
        count = agalite_parameter_count(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_size=width,
            head_dim=width // 2,
            feedforward_size=width,
            num_heads=AGALITE_NUM_HEADS,
            eta=AGALITE_ETA,
            num_layers=AGALITE_NUM_LAYERS,
            source_actor_critic=False,
        )
        if count >= target_parameters:
            candidates = [(width, count)]
            if previous_width is not None and previous_count is not None:
                candidates.append((previous_width, previous_count))
            return min(
                candidates,
                key=lambda item: (abs(item[1] - target_parameters), item[0]),
            )[0]
        previous_width = width
        previous_count = count
        width += 2


@dataclass(frozen=True)
class AGaLiTePolicyCore:
    """Parameter-matched AGaLiTe block over the shared augmented input."""

    input_dim: int
    action_dim: int
    hidden_size: int
    head_dim: int
    feedforward_size: int
    num_heads: int = AGALITE_NUM_HEADS
    eta: int = AGALITE_ETA
    approximation_channels: int = AGALITE_APPROXIMATION_CHANNELS
    num_layers: int = AGALITE_NUM_LAYERS
    gate_bias: float = AGALITE_GATE_BIAS
    attention_epsilon: float = AGALITE_ATTENTION_EPSILON
    layer_norm_epsilon: float = AGALITE_LAYER_NORM_EPSILON

    def __post_init__(self) -> None:
        for name in (
            "input_dim",
            "action_dim",
            "hidden_size",
            "head_dim",
            "feedforward_size",
            "num_heads",
            "eta",
            "approximation_channels",
            "num_layers",
        ):
            _positive_integer(getattr(self, name), name=name)
        if self.hidden_size % 2 != 0:
            raise ValueError("hidden_size must be even")
        if self.head_dim != self.hidden_size // 2:
            raise ValueError("head_dim must equal hidden_size // 2")
        if self.feedforward_size != self.hidden_size:
            raise ValueError("feedforward_size must equal hidden_size")
        if not math.isfinite(self.gate_bias):
            raise ValueError("gate_bias must be finite")
        if self.attention_epsilon <= 0.0 or not math.isfinite(
            self.attention_epsilon
        ):
            raise ValueError("attention_epsilon must be finite and positive")
        if self.layer_norm_epsilon <= 0.0 or not math.isfinite(
            self.layer_norm_epsilon
        ):
            raise ValueError("layer_norm_epsilon must be finite and positive")

    def initialize(self, key: Array) -> dict[str, Array]:
        backbone_key, actor_key, critic_key = jax.random.split(key, 3)
        params, _ = _initialize_backbone(
            backbone_key,
            input_dim=self.input_dim,
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            feedforward_size=self.feedforward_size,
            num_heads=self.num_heads,
            eta=self.eta,
            num_layers=self.num_layers,
            gate_bias=self.gate_bias,
        )
        params["actor.kernel"] = 0.01 * _xavier(
            actor_key,
            self.hidden_size,
            self.action_dim,
        )
        params["actor.bias"] = jnp.zeros((self.action_dim,))
        params["critic.kernel"] = _xavier(
            critic_key,
            self.hidden_size,
            1,
        )
        params["critic.bias"] = jnp.zeros((1,))
        return params

    def initial_state(
        self,
        batch_size: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> AGaLiTeState:
        return _initial_state(
            batch_size,
            num_layers=self.num_layers,
            approximation_channels=self.approximation_channels,
            num_heads=self.num_heads,
            eta=self.eta,
            head_dim=self.head_dim,
            dtype=dtype,
        )

    def step(
        self,
        params: Params,
        state: AGaLiTeState,
        policy_input: Array,
        reset: Array,
    ) -> tuple[AGaLiTeState, Array, Array]:
        new_state, features = _backbone_step(
            params,
            state,
            policy_input,
            reset,
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            feedforward_size=self.feedforward_size,
            num_heads=self.num_heads,
            eta=self.eta,
            approximation_channels=self.approximation_channels,
            num_layers=self.num_layers,
            attention_epsilon=self.attention_epsilon,
            layer_norm_epsilon=self.layer_norm_epsilon,
        )
        logits = _linear(params, "actor", features)
        values = _linear(params, "critic", features)[..., 0]
        return new_state, logits, values

    def apply_sequence(
        self,
        params: Params,
        state: AGaLiTeState,
        policy_inputs: Array,
        resets: Array,
    ) -> tuple[AGaLiTeState, Array, Array]:
        def scan_step(carry, inputs):
            policy_input, reset = inputs
            new_carry, logits, value = self.step(
                params,
                carry,
                policy_input,
                reset,
            )
            return new_carry, (logits, value)

        new_state, (logits, values) = jax.lax.scan(
            scan_step,
            state,
            (policy_inputs, resets),
        )
        return new_state, logits, values

    @staticmethod
    def count_parameters(params: Params) -> int:
        return sum(int(value.size) for value in jax.tree.leaves(params))

@dataclass(frozen=True)
class SourceCompatibleAGaLiTePolicyCore:
    """Released T-Maze vector policy inside the shared PPO learner."""

    input_dim: int
    observation_dim: int
    action_dim: int
    hidden_size: int = AGALITE_MODEL_SIZE
    head_dim: int = AGALITE_HEAD_SIZE
    feedforward_size: int = AGALITE_FEEDFORWARD_SIZE
    num_heads: int = AGALITE_NUM_HEADS
    eta: int = AGALITE_ETA
    approximation_channels: int = AGALITE_APPROXIMATION_CHANNELS
    num_layers: int = AGALITE_NUM_LAYERS
    actor_hidden_size: int = AGALITE_SOURCE_ACTOR_HIDDEN_SIZE
    critic_hidden_size: int = AGALITE_SOURCE_CRITIC_HIDDEN_SIZE
    gate_bias: float = AGALITE_GATE_BIAS
    attention_epsilon: float = AGALITE_ATTENTION_EPSILON
    layer_norm_epsilon: float = AGALITE_LAYER_NORM_EPSILON

    def __post_init__(self) -> None:
        _positive_integer(self.input_dim, name="input_dim")
        _positive_integer(self.observation_dim, name="observation_dim")
        _positive_integer(self.action_dim, name="action_dim")
        if self.observation_dim > self.input_dim:
            raise ValueError("observation_dim must not exceed input_dim")
        expected = {
            "hidden_size": AGALITE_MODEL_SIZE,
            "head_dim": AGALITE_HEAD_SIZE,
            "feedforward_size": AGALITE_FEEDFORWARD_SIZE,
            "num_heads": AGALITE_NUM_HEADS,
            "eta": AGALITE_ETA,
            "approximation_channels": AGALITE_APPROXIMATION_CHANNELS,
            "num_layers": AGALITE_NUM_LAYERS,
            "actor_hidden_size": AGALITE_SOURCE_ACTOR_HIDDEN_SIZE,
            "critic_hidden_size": AGALITE_SOURCE_CRITIC_HIDDEN_SIZE,
            "gate_bias": AGALITE_GATE_BIAS,
            "attention_epsilon": AGALITE_ATTENTION_EPSILON,
            "layer_norm_epsilon": AGALITE_LAYER_NORM_EPSILON,
        }
        for name, expected_value in expected.items():
            if getattr(self, name) != expected_value:
                raise ValueError(
                    f"source-compatible AGaLiTe {name} must equal {expected_value}"
                )

    def initialize(self, key: Array) -> dict[str, Array]:
        backbone_key, actor_hidden_key, actor_output_key, critic_hidden_key, critic_output_key = (
            jax.random.split(key, 5)
        )
        params, _ = _initialize_backbone(
            backbone_key,
            input_dim=self.observation_dim,
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            feedforward_size=self.feedforward_size,
            num_heads=self.num_heads,
            eta=self.eta,
            num_layers=self.num_layers,
            gate_bias=self.gate_bias,
        )
        params["actor.hidden.kernel"] = _orthogonal(
            actor_hidden_key,
            self.hidden_size,
            self.actor_hidden_size,
        )
        params["actor.hidden.bias"] = jnp.zeros((self.actor_hidden_size,))
        params["actor.output.kernel"] = _orthogonal(
            actor_output_key,
            self.actor_hidden_size,
            self.action_dim,
        )
        params["actor.output.bias"] = jnp.zeros((self.action_dim,))
        params["critic.hidden.kernel"] = _orthogonal(
            critic_hidden_key,
            self.hidden_size,
            self.critic_hidden_size,
        )
        params["critic.hidden.bias"] = jnp.zeros((self.critic_hidden_size,))
        params["critic.output.kernel"] = _orthogonal(
            critic_output_key,
            self.critic_hidden_size,
            1,
        )
        params["critic.output.bias"] = jnp.zeros((1,))
        return params

    def initial_state(
        self,
        batch_size: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> AGaLiTeState:
        return _initial_state(
            batch_size,
            num_layers=self.num_layers,
            approximation_channels=self.approximation_channels,
            num_heads=self.num_heads,
            eta=self.eta,
            head_dim=self.head_dim,
            dtype=dtype,
        )

    def step(
        self,
        params: Params,
        state: AGaLiTeState,
        policy_input: Array,
        reset: Array,
    ) -> tuple[AGaLiTeState, Array, Array]:
        observation = policy_input[..., : self.observation_dim]
        new_state, features = _backbone_step(
            params,
            state,
            observation,
            reset,
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            feedforward_size=self.feedforward_size,
            num_heads=self.num_heads,
            eta=self.eta,
            approximation_channels=self.approximation_channels,
            num_layers=self.num_layers,
            attention_epsilon=self.attention_epsilon,
            layer_norm_epsilon=self.layer_norm_epsilon,
        )
        actor_hidden = jnp.tanh(_linear(params, "actor.hidden", features))
        critic_hidden = jnp.tanh(_linear(params, "critic.hidden", features))
        logits = _linear(params, "actor.output", actor_hidden)
        values = _linear(params, "critic.output", critic_hidden)[..., 0]
        return new_state, logits, values

    def apply_sequence(
        self,
        params: Params,
        state: AGaLiTeState,
        policy_inputs: Array,
        resets: Array,
    ) -> tuple[AGaLiTeState, Array, Array]:
        def scan_step(carry, inputs):
            policy_input, reset = inputs
            new_carry, logits, value = self.step(
                params,
                carry,
                policy_input,
                reset,
            )
            return new_carry, (logits, value)

        new_state, (logits, values) = jax.lax.scan(
            scan_step,
            state,
            (policy_inputs, resets),
        )
        return new_state, logits, values

    @staticmethod
    def count_parameters(params: Params) -> int:
        return sum(int(value.size) for value in jax.tree.leaves(params))
