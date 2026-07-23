"""Baseline policy cores that share the ArcMind PPO interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array
Params = Mapping[str, Array]


def _xavier(key: Array, input_features: int, output_features: int) -> Array:
    bound = jnp.sqrt(6.0 / (input_features + output_features))
    return jax.random.uniform(
        key,
        (input_features, output_features),
        minval=-bound,
        maxval=bound,
    )


def _linear(params: Params, prefix: str, x: Array) -> Array:
    return x @ params[f"{prefix}.kernel"] + params[f"{prefix}.bias"]


@dataclass(frozen=True)
class MemorylessMLPPolicyCore:
    """Two-layer memoryless actor-critic baseline."""

    input_dim: int
    action_dim: int
    hidden_size: int

    def initialize(self, key: Array) -> dict[str, Array]:
        keys = iter(jax.random.split(key, 6))
        return {
            "hidden.0.kernel": _xavier(next(keys), self.input_dim, self.hidden_size),
            "hidden.0.bias": jnp.zeros((self.hidden_size,)),
            "hidden.1.kernel": _xavier(
                next(keys),
                self.hidden_size,
                self.hidden_size,
            ),
            "hidden.1.bias": jnp.zeros((self.hidden_size,)),
            "actor.kernel": 0.01
            * _xavier(next(keys), self.hidden_size, self.action_dim),
            "actor.bias": jnp.zeros((self.action_dim,)),
            "critic.kernel": _xavier(next(keys), self.hidden_size, 1),
            "critic.bias": jnp.zeros((1,)),
        }

    @staticmethod
    def initial_state(
        batch_size: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> Array:
        return jnp.zeros((batch_size, 0), dtype=dtype)

    @staticmethod
    def _features(params: Params, policy_input: Array) -> Array:
        features = jax.nn.tanh(_linear(params, "hidden.0", policy_input))
        return jax.nn.tanh(_linear(params, "hidden.1", features))

    def step(
        self,
        params: Params,
        state: Array,
        policy_input: Array,
        reset: Array,
    ) -> tuple[Array, Array, Array]:
        del reset
        features = self._features(params, policy_input)
        logits = _linear(params, "actor", features)
        values = _linear(params, "critic", features)[..., 0]
        return state, logits, values

    def apply_sequence(
        self,
        params: Params,
        state: Array,
        policy_inputs: Array,
        resets: Array,
    ) -> tuple[Array, Array, Array]:
        del resets
        features = jax.vmap(self._features, in_axes=(None, 0))(params, policy_inputs)
        logits = _linear(params, "actor", features)
        values = _linear(params, "critic", features)[..., 0]
        return state, logits, values

    @staticmethod
    def count_parameters(params: Params) -> int:
        return sum(int(value.size) for value in jax.tree.leaves(params))


@dataclass(frozen=True)
class GRUPolicyCore:
    """Reset-aware GRU actor-critic baseline."""

    input_dim: int
    action_dim: int
    hidden_size: int

    def initialize(self, key: Array) -> dict[str, Array]:
        input_key, recurrent_key, actor_key, critic_key = jax.random.split(key, 4)
        return {
            "gru.input_kernel": _xavier(
                input_key,
                self.input_dim,
                self.hidden_size * 3,
            ),
            "gru.recurrent_kernel": _xavier(
                recurrent_key,
                self.hidden_size,
                self.hidden_size * 3,
            ),
            "gru.bias": jnp.zeros((self.hidden_size * 3,)),
            "actor.kernel": 0.01
            * _xavier(actor_key, self.hidden_size, self.action_dim),
            "actor.bias": jnp.zeros((self.action_dim,)),
            "critic.kernel": _xavier(critic_key, self.hidden_size, 1),
            "critic.bias": jnp.zeros((1,)),
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> Array:
        return jnp.zeros((batch_size, self.hidden_size), dtype=dtype)

    def step(
        self,
        params: Params,
        state: Array,
        policy_input: Array,
        reset: Array,
    ) -> tuple[Array, Array, Array]:
        state = jnp.where(reset[:, None], jnp.zeros_like(state), state)
        input_gates = policy_input @ params["gru.input_kernel"]
        recurrent_gates = state @ params["gru.recurrent_kernel"]
        input_reset, input_update, input_candidate = jnp.split(
            input_gates + params["gru.bias"],
            3,
            axis=-1,
        )
        recurrent_reset, recurrent_update, recurrent_candidate = jnp.split(
            recurrent_gates,
            3,
            axis=-1,
        )
        reset_gate = jax.nn.sigmoid(input_reset + recurrent_reset)
        update_gate = jax.nn.sigmoid(input_update + recurrent_update)
        candidate = jnp.tanh(input_candidate + reset_gate * recurrent_candidate)
        new_state = update_gate * state + (1.0 - update_gate) * candidate
        logits = _linear(params, "actor", new_state)
        values = _linear(params, "critic", new_state)[..., 0]
        return new_state, logits, values

    def apply_sequence(
        self,
        params: Params,
        state: Array,
        policy_inputs: Array,
        resets: Array,
    ) -> tuple[Array, Array, Array]:
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
        return sum(int(value.size) for value in jax.tree.leaves(params))


@dataclass(frozen=True)
class ElmanRNNPolicyCore:
    """Vanilla tanh RNN control for isolating the effect of gating."""

    input_dim: int
    action_dim: int
    hidden_size: int

    def initialize(self, key: Array) -> dict[str, Array]:
        input_key, recurrent_key, actor_key, critic_key = jax.random.split(key, 4)
        return {
            "rnn.input_kernel": _xavier(input_key, self.input_dim, self.hidden_size),
            "rnn.recurrent_kernel": _xavier(
                recurrent_key,
                self.hidden_size,
                self.hidden_size,
            ),
            "rnn.bias": jnp.zeros((self.hidden_size,)),
            "actor.kernel": 0.01
            * _xavier(actor_key, self.hidden_size, self.action_dim),
            "actor.bias": jnp.zeros((self.action_dim,)),
            "critic.kernel": _xavier(critic_key, self.hidden_size, 1),
            "critic.bias": jnp.zeros((1,)),
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> Array:
        return jnp.zeros((batch_size, self.hidden_size), dtype=dtype)

    def step(
        self,
        params: Params,
        state: Array,
        policy_input: Array,
        reset: Array,
    ) -> tuple[Array, Array, Array]:
        state = jnp.where(reset[:, None], jnp.zeros_like(state), state)
        new_state = jnp.tanh(
            policy_input @ params["rnn.input_kernel"]
            + state @ params["rnn.recurrent_kernel"]
            + params["rnn.bias"]
        )
        logits = _linear(params, "actor", new_state)
        values = _linear(params, "critic", new_state)[..., 0]
        return new_state, logits, values

    def apply_sequence(
        self,
        params: Params,
        state: Array,
        policy_inputs: Array,
        resets: Array,
    ) -> tuple[Array, Array, Array]:
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
        return sum(int(value.size) for value in jax.tree.leaves(params))


class LSTMState(NamedTuple):
    hidden: Array
    cell: Array


@dataclass(frozen=True)
class LSTMPolicyCore:
    """Standard reset-aware LSTM actor-critic baseline."""

    input_dim: int
    action_dim: int
    hidden_size: int

    def initialize(self, key: Array) -> dict[str, Array]:
        input_key, recurrent_key, actor_key, critic_key = jax.random.split(key, 4)
        bias = jnp.zeros((self.hidden_size * 4,))
        bias = bias.at[self.hidden_size : self.hidden_size * 2].set(1.0)
        return {
            "lstm.input_kernel": _xavier(
                input_key,
                self.input_dim,
                self.hidden_size * 4,
            ),
            "lstm.recurrent_kernel": _xavier(
                recurrent_key,
                self.hidden_size,
                self.hidden_size * 4,
            ),
            "lstm.bias": bias,
            "actor.kernel": 0.01
            * _xavier(actor_key, self.hidden_size, self.action_dim),
            "actor.bias": jnp.zeros((self.action_dim,)),
            "critic.kernel": _xavier(critic_key, self.hidden_size, 1),
            "critic.bias": jnp.zeros((1,)),
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> LSTMState:
        zeros = jnp.zeros((batch_size, self.hidden_size), dtype=dtype)
        return LSTMState(hidden=zeros, cell=zeros)

    def step(
        self,
        params: Params,
        state: LSTMState,
        policy_input: Array,
        reset: Array,
    ) -> tuple[LSTMState, Array, Array]:
        hidden = jnp.where(reset[:, None], jnp.zeros_like(state.hidden), state.hidden)
        cell = jnp.where(reset[:, None], jnp.zeros_like(state.cell), state.cell)
        gates = (
            policy_input @ params["lstm.input_kernel"]
            + hidden @ params["lstm.recurrent_kernel"]
            + params["lstm.bias"]
        )
        input_gate, forget_gate, candidate, output_gate = jnp.split(
            gates,
            4,
            axis=-1,
        )
        input_gate = jax.nn.sigmoid(input_gate)
        forget_gate = jax.nn.sigmoid(forget_gate)
        output_gate = jax.nn.sigmoid(output_gate)
        candidate = jnp.tanh(candidate)
        new_cell = forget_gate * cell + input_gate * candidate
        new_hidden = output_gate * jnp.tanh(new_cell)
        logits = _linear(params, "actor", new_hidden)
        values = _linear(params, "critic", new_hidden)[..., 0]
        return LSTMState(new_hidden, new_cell), logits, values

    def apply_sequence(
        self,
        params: Params,
        state: LSTMState,
        policy_inputs: Array,
        resets: Array,
    ) -> tuple[LSTMState, Array, Array]:
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
        return sum(int(value.size) for value in jax.tree.leaves(params))


class FrameStackState(NamedTuple):
    frames: Array


class MemoryTraceState(NamedTuple):
    """Exponential moving averages at fixed retention rates."""

    traces: Array


@dataclass(frozen=True)
class MemoryTraceMLPPolicyCore:
    """ICML 2025 memory-trace features with a parameter-matched MLP."""

    input_dim: int
    action_dim: int
    hidden_size: int
    decays: tuple[float, ...] = (0.0, 0.985)

    def __post_init__(self) -> None:
        if not self.decays:
            raise ValueError("at least one memory-trace decay is required")
        if any(not 0.0 <= decay < 1.0 for decay in self.decays):
            raise ValueError("memory-trace decays must lie in [0, 1)")

    def initialize(self, key: Array) -> dict[str, Array]:
        first_key, second_key, actor_key, critic_key = jax.random.split(key, 4)
        trace_dim = self.input_dim * len(self.decays)
        return {
            "hidden.0.kernel": _xavier(first_key, trace_dim, self.hidden_size),
            "hidden.0.bias": jnp.zeros((self.hidden_size,)),
            "hidden.1.kernel": _xavier(
                second_key,
                self.hidden_size,
                self.hidden_size,
            ),
            "hidden.1.bias": jnp.zeros((self.hidden_size,)),
            "actor.kernel": 0.01
            * _xavier(actor_key, self.hidden_size, self.action_dim),
            "actor.bias": jnp.zeros((self.action_dim,)),
            "critic.kernel": _xavier(critic_key, self.hidden_size, 1),
            "critic.bias": jnp.zeros((1,)),
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> MemoryTraceState:
        return MemoryTraceState(
            traces=jnp.zeros(
                (batch_size, len(self.decays), self.input_dim),
                dtype=dtype,
            )
        )

    def step(
        self,
        params: Params,
        state: MemoryTraceState,
        policy_input: Array,
        reset: Array,
    ) -> tuple[MemoryTraceState, Array, Array]:
        traces = jnp.where(
            reset[:, None, None],
            jnp.zeros_like(state.traces),
            state.traces,
        )
        decays = jnp.asarray(self.decays, dtype=policy_input.dtype)
        traces = (
            (1.0 - decays)[None, :, None] * policy_input[:, None, :]
            + decays[None, :, None] * traces
        )
        flattened = traces.reshape((traces.shape[0], -1))
        features = jax.nn.tanh(_linear(params, "hidden.0", flattened))
        features = jax.nn.tanh(_linear(params, "hidden.1", features))
        logits = _linear(params, "actor", features)
        values = _linear(params, "critic", features)[..., 0]
        return MemoryTraceState(traces), logits, values

    def apply_sequence(
        self,
        params: Params,
        state: MemoryTraceState,
        policy_inputs: Array,
        resets: Array,
    ) -> tuple[MemoryTraceState, Array, Array]:
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
        return sum(int(value.size) for value in jax.tree.leaves(params))


@dataclass(frozen=True)
class FrameStackMLPPolicyCore:
    """Finite-history MLP control with an explicit causal frame stack."""

    input_dim: int
    action_dim: int
    hidden_size: int
    stack_size: int = 4

    def initialize(self, key: Array) -> dict[str, Array]:
        first_key, second_key, actor_key, critic_key = jax.random.split(key, 4)
        stacked_dim = self.input_dim * self.stack_size
        return {
            "hidden.0.kernel": _xavier(first_key, stacked_dim, self.hidden_size),
            "hidden.0.bias": jnp.zeros((self.hidden_size,)),
            "hidden.1.kernel": _xavier(
                second_key,
                self.hidden_size,
                self.hidden_size,
            ),
            "hidden.1.bias": jnp.zeros((self.hidden_size,)),
            "actor.kernel": 0.01
            * _xavier(actor_key, self.hidden_size, self.action_dim),
            "actor.bias": jnp.zeros((self.action_dim,)),
            "critic.kernel": _xavier(critic_key, self.hidden_size, 1),
            "critic.bias": jnp.zeros((1,)),
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> FrameStackState:
        return FrameStackState(
            frames=jnp.zeros(
                (batch_size, self.stack_size, self.input_dim),
                dtype=dtype,
            )
        )

    def step(
        self,
        params: Params,
        state: FrameStackState,
        policy_input: Array,
        reset: Array,
    ) -> tuple[FrameStackState, Array, Array]:
        frames = jnp.where(
            reset[:, None, None],
            jnp.zeros_like(state.frames),
            state.frames,
        )
        frames = jnp.concatenate(
            [frames[:, 1:, :], policy_input[:, None, :]],
            axis=1,
        )
        flattened = frames.reshape((frames.shape[0], -1))
        features = jax.nn.tanh(_linear(params, "hidden.0", flattened))
        features = jax.nn.tanh(_linear(params, "hidden.1", features))
        logits = _linear(params, "actor", features)
        values = _linear(params, "critic", features)[..., 0]
        return FrameStackState(frames), logits, values

    def apply_sequence(
        self,
        params: Params,
        state: FrameStackState,
        policy_inputs: Array,
        resets: Array,
    ) -> tuple[FrameStackState, Array, Array]:
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
        return sum(int(value.size) for value in jax.tree.leaves(params))


class TCNState(NamedTuple):
    histories: tuple[Array, ...]


@dataclass(frozen=True)
class TCNPolicyCore:
    """Streaming causal dilated temporal-convolution baseline."""

    input_dim: int
    action_dim: int
    hidden_size: int
    num_layers: int = 3
    kernel_size: int = 3

    def initialize(self, key: Array) -> dict[str, Array]:
        key_count = self.num_layers + 3
        keys = iter(jax.random.split(key, key_count))
        params = {
            "input.kernel": _xavier(next(keys), self.input_dim, self.hidden_size),
            "input.bias": jnp.zeros((self.hidden_size,)),
        }
        for layer_index in range(self.num_layers):
            convolution_key = next(keys)
            params[f"layers.{layer_index}.kernel"] = jax.random.normal(
                convolution_key,
                (self.kernel_size, self.hidden_size, self.hidden_size),
            ) * jnp.sqrt(
                2.0 / (self.kernel_size * self.hidden_size + self.hidden_size)
            )
            params[f"layers.{layer_index}.bias"] = jnp.zeros((self.hidden_size,))
            params[f"layers.{layer_index}.norm.weight"] = jnp.ones(
                (self.hidden_size,)
            )
            params[f"layers.{layer_index}.norm.bias"] = jnp.zeros(
                (self.hidden_size,)
            )
        params["actor.kernel"] = 0.01 * _xavier(
            next(keys),
            self.hidden_size,
            self.action_dim,
        )
        params["actor.bias"] = jnp.zeros((self.action_dim,))
        params["critic.kernel"] = _xavier(next(keys), self.hidden_size, 1)
        params["critic.bias"] = jnp.zeros((1,))
        return params

    def initial_state(
        self,
        batch_size: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> TCNState:
        histories = tuple(
            jnp.zeros(
                (
                    batch_size,
                    (self.kernel_size - 1) * (2**layer_index),
                    self.hidden_size,
                ),
                dtype=dtype,
            )
            for layer_index in range(self.num_layers)
        )
        return TCNState(histories)

    @staticmethod
    def _layer_norm(
        params: Params,
        prefix: str,
        values: Array,
    ) -> Array:
        mean = jnp.mean(values, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
        normalized = (values - mean) * jax.lax.rsqrt(variance + 1e-5)
        return (
            normalized * params[f"{prefix}.weight"]
            + params[f"{prefix}.bias"]
        )

    def step(
        self,
        params: Params,
        state: TCNState,
        policy_input: Array,
        reset: Array,
    ) -> tuple[TCNState, Array, Array]:
        features = jax.nn.gelu(_linear(params, "input", policy_input))
        new_histories = []
        for layer_index in range(self.num_layers):
            history = jnp.where(
                reset[:, None, None],
                jnp.zeros_like(state.histories[layer_index]),
                state.histories[layer_index],
            )
            window = jnp.concatenate([history, features[:, None, :]], axis=1)
            dilation = 2**layer_index
            taps = window[:, ::dilation, :]
            convolved = jnp.einsum(
                "bki,kio->bo",
                taps,
                params[f"layers.{layer_index}.kernel"],
            )
            convolved = convolved + params[f"layers.{layer_index}.bias"]
            convolved = jax.nn.gelu(convolved)
            features = self._layer_norm(
                params,
                f"layers.{layer_index}.norm",
                features + convolved,
            )
            new_histories.append(window[:, 1:, :])
        logits = _linear(params, "actor", features)
        values = _linear(params, "critic", features)[..., 0]
        return TCNState(tuple(new_histories)), logits, values

    def apply_sequence(
        self,
        params: Params,
        state: TCNState,
        policy_inputs: Array,
        resets: Array,
    ) -> tuple[TCNState, Array, Array]:
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
        return sum(int(value.size) for value in jax.tree.leaves(params))


def match_baseline_width(
    model: str,
    *,
    target_parameters: int,
    input_dim: int,
    action_dim: int,
    frame_stack_size: int = 4,
    memory_trace_count: int = 2,
    tcn_layers: int = 3,
    tcn_kernel_size: int = 3,
    maximum_width: int = 1024,
) -> int:
    """Find the integer width with the closest parameter count to ArcMind."""

    def count(width: int) -> int:
        if model == "memoryless_mlp":
            return (
                input_dim * width
                + width
                + width * width
                + width
                + width * action_dim
                + action_dim
                + width
                + 1
            )
        if model == "gru":
            return (
                input_dim * 3 * width
                + width * 3 * width
                + 3 * width
                + width * action_dim
                + action_dim
                + width
                + 1
            )
        if model == "elman_rnn":
            return (
                input_dim * width
                + width * width
                + width
                + width * action_dim
                + action_dim
                + width
                + 1
            )
        if model == "lstm":
            return (
                input_dim * 4 * width
                + width * 4 * width
                + 4 * width
                + width * action_dim
                + action_dim
                + width
                + 1
            )
        if model in {"frame_stack_mlp", "memory_trace_mlp"}:
            history_size = (
                frame_stack_size
                if model == "frame_stack_mlp"
                else memory_trace_count
            )
            stacked_dim = input_dim * history_size
            return (
                stacked_dim * width
                + width
                + width * width
                + width
                + width * action_dim
                + action_dim
                + width
                + 1
            )
        if model == "tcn":
            input_projection = input_dim * width + width
            blocks = tcn_layers * (
                tcn_kernel_size * width * width
                + width
                + 2 * width
            )
            heads = width * action_dim + action_dim + width + 1
            return input_projection + blocks + heads
        raise ValueError(f"Unsupported baseline model: {model}")

    return min(
        range(1, maximum_width + 1),
        key=lambda width: abs(count(width) - target_parameters),
    )
