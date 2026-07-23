"""Pure-JAX streaming reference for the PyTorch ArcMind implementation.

This module deliberately avoids Flax parameter conventions. It consumes a flat
mapping with the same names and tensor layouts as ``ArcMindModel.state_dict()``,
which makes cross-framework numerical checks direct and auditable.

The reference covers deterministic inference. Training-specific dropout is not
applied; parity fixtures therefore put the PyTorch model in evaluation mode.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array
Params = Mapping[str, Array]


@dataclass(frozen=True)
class ReferenceConfig:
    """Static configuration required by the streaming JAX reference."""

    num_sensor_channels: int
    d_model: int
    num_ssm_layers: int
    ssm_state_dim: int
    ssm_conv_width: int
    ssm_expand_factor: int
    num_attn_layers: int
    num_attn_heads: int
    attn_window_size: int
    num_memory_slots: int
    memory_compress_ratio: int
    action_dim: int
    decision_stride: int
    ablate_ssm: bool = False
    ablate_attention: bool = False
    ablate_memory: bool = False
    ablate_gating: bool = False
    ablate_temporal_encoding: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ReferenceConfig:
        """Construct from a serialized PyTorch configuration."""
        known = {field.name for field in fields(cls)}
        return cls(**{name: values[name] for name in known})

    @property
    def d_inner(self) -> int:
        return self.d_model * self.ssm_expand_factor


class ArcMindState(NamedTuple):
    """Functional recurrent state, with independent state for every environment."""

    ssm: tuple[Array, ...]
    convolution: tuple[Array, ...]
    memory: Array
    memory_write_count: Array
    step_count: Array
    last_slow_output: Array


def _linear(params: Params, prefix: str, x: Array, *, bias: bool) -> Array:
    output = x @ params[f"{prefix}.weight"].T
    if bias:
        output = output + params[f"{prefix}.bias"]
    return output


def _layer_norm(params: Params, prefix: str, x: Array) -> Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    normalized = (x - mean) * jax.lax.rsqrt(variance + 1e-5)
    return normalized * params[f"{prefix}.weight"] + params[f"{prefix}.bias"]


def init_stream_state(
    config: ReferenceConfig,
    batch_size: int,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> ArcMindState:
    """Create zeroed recurrent state for a vector of environments."""
    ssm = tuple(
        jnp.zeros(
            (batch_size, config.d_inner, config.ssm_state_dim),
            dtype=dtype,
        )
        for _ in range(config.num_ssm_layers)
    )
    convolution = tuple(
        jnp.zeros(
            (batch_size, config.d_inner, config.ssm_conv_width - 1),
            dtype=dtype,
        )
        for _ in range(config.num_ssm_layers)
    )
    return ArcMindState(
        ssm=ssm,
        convolution=convolution,
        memory=jnp.zeros(
            (batch_size, config.num_memory_slots, config.d_model),
            dtype=dtype,
        ),
        memory_write_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        step_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        last_slow_output=jnp.zeros((batch_size, config.d_model), dtype=dtype),
    )


def _reset_where(state: ArcMindState, reset: Array) -> ArcMindState:
    reset = jnp.asarray(reset, dtype=jnp.bool_)
    vector_mask = reset[:, None]
    matrix_mask = reset[:, None, None]
    return ArcMindState(
        ssm=tuple(jnp.where(matrix_mask, jnp.zeros_like(value), value) for value in state.ssm),
        convolution=tuple(
            jnp.where(matrix_mask, jnp.zeros_like(value), value)
            for value in state.convolution
        ),
        memory=jnp.where(matrix_mask, jnp.zeros_like(state.memory), state.memory),
        memory_write_count=jnp.where(
            reset,
            jnp.zeros_like(state.memory_write_count),
            state.memory_write_count,
        ),
        step_count=jnp.where(reset, jnp.zeros_like(state.step_count), state.step_count),
        last_slow_output=jnp.where(
            vector_mask,
            jnp.zeros_like(state.last_slow_output),
            state.last_slow_output,
        ),
    )


def _ssm_layer_step(
    params: Params,
    prefix: str,
    x: Array,
    ssm_state: Array,
    convolution_state: Array,
) -> tuple[Array, Array, Array]:
    residual = x
    projected = _linear(params, f"{prefix}.in_proj", x, bias=False)
    x_branch, z = jnp.split(projected, 2, axis=-1)

    convolution_input = jnp.concatenate(
        [convolution_state, x_branch[..., None]],
        axis=-1,
    )
    convolution_weight = jnp.squeeze(params[f"{prefix}.conv.weight"], axis=1)
    x_convolved = jnp.sum(
        convolution_input * convolution_weight[None, ...],
        axis=-1,
    )
    x_convolved = x_convolved + params[f"{prefix}.conv.bias"]
    x_convolved = jax.nn.silu(x_convolved)
    new_convolution_state = convolution_input[..., 1:]

    transition = -jnp.exp(params[f"{prefix}.A_log"])
    delta = jax.nn.softplus(
        _linear(params, f"{prefix}.dt_proj", x_convolved, bias=True)
    )
    input_projection = _linear(
        params,
        f"{prefix}.B_proj",
        x_convolved,
        bias=False,
    )
    output_projection = _linear(
        params,
        f"{prefix}.C_proj",
        x_convolved,
        bias=False,
    )

    delta_expanded = delta[..., None]
    discretized_transition = jnp.exp(transition[None, ...] * delta_expanded)
    discretized_input = delta_expanded * input_projection[:, None, :]
    new_ssm_state = (
        discretized_transition * ssm_state
        + discretized_input * x_convolved[..., None]
    )
    y = jnp.sum(new_ssm_state * output_projection[:, None, :], axis=-1)
    y = y + x_convolved * params[f"{prefix}.D"]
    y = y * jax.nn.silu(z)
    y = _linear(params, f"{prefix}.out_proj", y, bias=False) + residual
    y = _layer_norm(params, f"{prefix}.norm", y)
    return y, new_ssm_state, new_convolution_state


def _chronological_memory(
    params: Params,
    state: ArcMindState,
    config: ReferenceConfig,
    *,
    use_memory: bool,
) -> tuple[Array, Array]:
    """Return fixed-shape chronological memory and a validity mask."""
    slots = config.num_memory_slots
    positions = jnp.arange(slots, dtype=jnp.int32)[None, :]
    occupancy = jnp.minimum(state.memory_write_count, slots)
    oldest = jnp.where(
        occupancy == slots,
        state.memory_write_count % slots,
        jnp.zeros_like(occupancy),
    )
    physical_indices = (oldest[:, None] + positions) % slots
    chronological = jnp.take_along_axis(
        state.memory,
        physical_indices[..., None],
        axis=1,
    )

    first_visible = jnp.maximum(occupancy - config.attn_window_size, 0)
    valid = (positions < occupancy[:, None]) & (positions >= first_visible[:, None])
    if not use_memory or config.ablate_memory:
        valid = jnp.zeros_like(valid)

    if not config.ablate_temporal_encoding:
        ages = jnp.clip(
            occupancy[:, None] - 1 - positions,
            0,
            slots - 1,
        )
        chronological = (
            chronological
            + params["slow_attention.memory_age_embedding.weight"][ages]
        )
    return chronological, valid


def _attention_layer(
    params: Params,
    prefix: str,
    query: Array,
    memory: Array,
    memory_valid: Array,
    config: ReferenceConfig,
) -> Array:
    residual = query
    normalized_query = _layer_norm(params, f"{prefix}.norm1", query)
    context = jnp.concatenate([memory, normalized_query], axis=1)

    query_projection = _linear(
        params,
        f"{prefix}.qkv_proj",
        normalized_query,
        bias=False,
    )[..., : config.d_model]
    context_projection = _linear(
        params,
        f"{prefix}.qkv_proj",
        context,
        bias=False,
    )
    key_projection = context_projection[
        ..., config.d_model : 2 * config.d_model
    ]
    value_projection = context_projection[..., 2 * config.d_model :]

    batch_size = query.shape[0]
    head_dim = config.d_model // config.num_attn_heads
    query_projection = query_projection.reshape(
        batch_size,
        1,
        config.num_attn_heads,
        head_dim,
    ).transpose(0, 2, 1, 3)
    key_projection = key_projection.reshape(
        batch_size,
        config.num_memory_slots + 1,
        config.num_attn_heads,
        head_dim,
    ).transpose(0, 2, 1, 3)
    value_projection = value_projection.reshape(
        batch_size,
        config.num_memory_slots + 1,
        config.num_attn_heads,
        head_dim,
    ).transpose(0, 2, 1, 3)

    logits = jnp.einsum(
        "bhqd,bhkd->bhqk",
        query_projection,
        key_projection,
    ) / jnp.sqrt(jnp.asarray(head_dim, dtype=query.dtype))
    valid_context = jnp.concatenate(
        [
            memory_valid,
            jnp.ones((batch_size, 1), dtype=jnp.bool_),
        ],
        axis=1,
    )
    logits = jnp.where(
        valid_context[:, None, None, :],
        logits,
        jnp.finfo(logits.dtype).min,
    )
    attention = jax.nn.softmax(logits, axis=-1)
    attended = jnp.einsum("bhqk,bhkd->bhqd", attention, value_projection)
    attended = attended.transpose(0, 2, 1, 3).reshape(
        batch_size,
        1,
        config.d_model,
    )
    attended = _linear(params, f"{prefix}.out_proj", attended, bias=False)

    output = residual + attended
    feed_forward = _layer_norm(params, f"{prefix}.norm2", output)
    feed_forward = _linear(
        params,
        f"{prefix}.ffn.0",
        feed_forward,
        bias=True,
    )
    feed_forward = jax.nn.gelu(feed_forward, approximate=False)
    feed_forward = _linear(
        params,
        f"{prefix}.ffn.3",
        feed_forward,
        bias=True,
    )
    return output + feed_forward


def _slow_attention(
    params: Params,
    query: Array,
    state: ArcMindState,
    config: ReferenceConfig,
    *,
    use_memory: bool,
) -> Array:
    memory, memory_valid = _chronological_memory(
        params,
        state,
        config,
        use_memory=use_memory,
    )
    output = query[:, None, :]
    for layer_index in range(config.num_attn_layers):
        output = _attention_layer(
            params,
            f"slow_attention.layers.{layer_index}",
            output,
            memory,
            memory_valid,
            config,
        )
    return output[:, 0, :]


def _compress_memory(params: Params, snapshot: Array) -> Array:
    compressed = _linear(
        params,
        "memory.compressor.compress.0",
        snapshot,
        bias=True,
    )
    compressed = jax.nn.gelu(compressed, approximate=False)
    compressed = _linear(
        params,
        "memory.compressor.compress.2",
        compressed,
        bias=True,
    )
    return _layer_norm(
        params,
        "memory.compressor.compress.3",
        compressed,
    )


def _write_memory(
    state: ArcMindState,
    compressed: Array,
    write: Array,
    config: ReferenceConfig,
) -> tuple[Array, Array]:
    batch_indices = jnp.arange(state.memory.shape[0])
    slot_indices = state.memory_write_count % config.num_memory_slots
    candidate = state.memory.at[batch_indices, slot_indices, :].set(compressed)
    memory = jnp.where(write[:, None, None], candidate, state.memory)
    write_count = state.memory_write_count + write.astype(jnp.int32)
    return memory, write_count


def _arcmind_features_step(
    params: Params,
    state: ArcMindState,
    sensor_frame: Array,
    config: ReferenceConfig,
    *,
    reset: Array | None = None,
    use_memory: bool = True,
) -> tuple[ArcMindState, Array]:
    """Run the recurrent backbone and return the fused representation."""
    if reset is not None:
        state = _reset_where(state, reset)

    token = _linear(
        params,
        "tokenizer.projection",
        sensor_frame,
        bias=True,
    )
    token = _layer_norm(params, "tokenizer.norm", token)

    fast_output = token
    new_ssm_states = list(state.ssm)
    new_convolution_states = list(state.convolution)
    if not config.ablate_ssm:
        for layer_index in range(config.num_ssm_layers):
            prefix = f"ssm_core.layers.{layer_index}"
            fast_output, new_ssm, new_convolution = _ssm_layer_step(
                params,
                prefix,
                fast_output,
                state.ssm[layer_index],
                state.convolution[layer_index],
            )
            new_ssm_states[layer_index] = new_ssm
            new_convolution_states[layer_index] = new_convolution

    state = state._replace(
        ssm=tuple(new_ssm_states),
        convolution=tuple(new_convolution_states),
    )

    memory = state.memory
    memory_write_count = state.memory_write_count
    last_slow_output = state.last_slow_output
    if config.ablate_attention:
        fused = fast_output
    else:
        decision = (state.step_count % config.decision_stride) == 0
        slow_candidate = _slow_attention(
            params,
            fast_output,
            state,
            config,
            use_memory=use_memory,
        )
        last_slow_output = jnp.where(
            decision[:, None],
            slow_candidate,
            last_slow_output,
        )

        if use_memory and not config.ablate_memory:
            compressed = _compress_memory(params, fast_output)
            memory, memory_write_count = _write_memory(
                state,
                compressed,
                decision,
                config,
            )

        if config.ablate_ssm:
            fused = last_slow_output
        elif config.ablate_gating:
            fused = 0.5 * fast_output + 0.5 * last_slow_output
        else:
            gate = _linear(
                params,
                "gate.0",
                jnp.concatenate([fast_output, last_slow_output], axis=-1),
                bias=True,
            )
            gate = jax.nn.sigmoid(gate)
            fused = gate * last_slow_output + (1.0 - gate) * fast_output

    new_state = state._replace(
        memory=memory,
        memory_write_count=memory_write_count,
        step_count=state.step_count + 1,
        last_slow_output=last_slow_output,
    )
    return new_state, fused


def arcmind_step(
    params: Params,
    state: ArcMindState,
    sensor_frame: Array,
    config: ReferenceConfig,
    *,
    reset: Array | None = None,
    use_memory: bool = True,
) -> tuple[ArcMindState, Array]:
    """Run one causal sensor-rate inference step."""
    new_state, fused = _arcmind_features_step(
        params,
        state,
        sensor_frame,
        config,
        reset=reset,
        use_memory=use_memory,
    )
    action = _linear(params, "action_head.head.0", fused, bias=True)
    action = jax.nn.gelu(action, approximate=False)
    action = _linear(params, "action_head.head.2", action, bias=True)
    return new_state, action


def arcmind_actor_critic_step(
    params: Params,
    state: ArcMindState,
    sensor_frame: Array,
    config: ReferenceConfig,
    *,
    reset: Array | None = None,
    use_memory: bool = True,
) -> tuple[ArcMindState, Array, Array]:
    """Run the shared ArcMind backbone with categorical actor and value heads."""
    new_state, fused = _arcmind_features_step(
        params,
        state,
        sensor_frame,
        config,
        reset=reset,
        use_memory=use_memory,
    )
    logits = _linear(params, "action_head.head.0", fused, bias=True)
    logits = jax.nn.gelu(logits, approximate=False)
    logits = _linear(params, "action_head.head.2", logits, bias=True)
    value = _linear(params, "critic_head.0", fused, bias=True)
    value = jax.nn.gelu(value, approximate=False)
    value = _linear(params, "critic_head.2", value, bias=True)
    return new_state, logits, value[..., 0]


class _ParameterBuilder:
    """Deterministic initializer for the flat PyTorch-compatible parameter tree."""

    def __init__(self, key: Array):
        self.key = key
        self.params: dict[str, Array] = {}

    def _next_key(self) -> Array:
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def linear(
        self,
        prefix: str,
        input_features: int,
        output_features: int,
        *,
        bias: bool,
    ) -> None:
        bound = jnp.sqrt(6.0 / (input_features + output_features))
        self.params[f"{prefix}.weight"] = jax.random.uniform(
            self._next_key(),
            (output_features, input_features),
            minval=-bound,
            maxval=bound,
        )
        if bias:
            self.params[f"{prefix}.bias"] = jnp.zeros((output_features,))

    def layer_norm(self, prefix: str, features: int) -> None:
        self.params[f"{prefix}.weight"] = jnp.ones((features,))
        self.params[f"{prefix}.bias"] = jnp.zeros((features,))


def initialize_actor_critic_params(
    key: Array,
    config: ReferenceConfig,
) -> dict[str, Array]:
    """Initialize a trainable ArcMind actor-critic in reference tensor layout."""
    builder = _ParameterBuilder(key)
    inner = config.d_inner
    model = config.d_model

    builder.linear(
        "tokenizer.projection",
        config.num_sensor_channels,
        model,
        bias=True,
    )
    builder.layer_norm("tokenizer.norm", model)

    for layer_index in range(config.num_ssm_layers):
        prefix = f"ssm_core.layers.{layer_index}"
        builder.linear(f"{prefix}.in_proj", model, inner * 2, bias=False)
        convolution_bound = 1.0 / jnp.sqrt(config.ssm_conv_width)
        builder.params[f"{prefix}.conv.weight"] = jax.random.uniform(
            builder._next_key(),
            (inner, 1, config.ssm_conv_width),
            minval=-convolution_bound,
            maxval=convolution_bound,
        )
        builder.params[f"{prefix}.conv.bias"] = jax.random.uniform(
            builder._next_key(),
            (inner,),
            minval=-convolution_bound,
            maxval=convolution_bound,
        )
        builder.linear(f"{prefix}.dt_proj", inner, inner, bias=True)
        builder.params[f"{prefix}.A_log"] = jnp.log(
            jnp.arange(1, config.ssm_state_dim + 1, dtype=jnp.float32)
        )[None, :].repeat(inner, axis=0)
        builder.linear(
            f"{prefix}.B_proj",
            inner,
            config.ssm_state_dim,
            bias=False,
        )
        builder.linear(
            f"{prefix}.C_proj",
            inner,
            config.ssm_state_dim,
            bias=False,
        )
        builder.params[f"{prefix}.D"] = jnp.ones((inner,))
        builder.linear(f"{prefix}.out_proj", inner, model, bias=False)
        builder.layer_norm(f"{prefix}.norm", model)

    compressed_features = model // config.memory_compress_ratio
    builder.linear(
        "memory.compressor.compress.0",
        model,
        compressed_features,
        bias=True,
    )
    builder.linear(
        "memory.compressor.compress.2",
        compressed_features,
        model,
        bias=True,
    )
    builder.layer_norm("memory.compressor.compress.3", model)

    builder.params["slow_attention.memory_age_embedding.weight"] = (
        jax.random.normal(
            builder._next_key(),
            (config.num_memory_slots, model),
        )
        * 0.02
    )
    for layer_index in range(config.num_attn_layers):
        prefix = f"slow_attention.layers.{layer_index}"
        builder.linear(f"{prefix}.qkv_proj", model, model * 3, bias=False)
        builder.linear(f"{prefix}.out_proj", model, model, bias=False)
        builder.layer_norm(f"{prefix}.norm1", model)
        builder.layer_norm(f"{prefix}.norm2", model)
        builder.linear(f"{prefix}.ffn.0", model, model * 4, bias=True)
        builder.linear(f"{prefix}.ffn.3", model * 4, model, bias=True)

    builder.linear("gate.0", model * 2, model, bias=True)
    builder.linear("action_head.head.0", model, model, bias=True)
    builder.linear("action_head.head.2", model, config.action_dim, bias=True)
    builder.params["action_head.head.2.weight"] *= 0.01
    builder.linear("critic_head.0", model, model, bias=True)
    builder.linear("critic_head.2", model, 1, bias=True)
    return builder.params
