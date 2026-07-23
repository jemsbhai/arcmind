"""Fast and Forgetful Memory policy core for the shared POBAX PPO harness.

This is a dependency-free JAX translation of the FFM cell in Morad et al.,
"Reinforcement Learning with Fast and Forgetful Memory" (NeurIPS 2023),
Equations 11 through 18. The implementation was audited against
``proroklab/ffm`` commit ``b3f94d2a0f35ba05089faf19ab1df846057cf8b6``:

* ``standalone_jax/ffm/ffa.py`` lines 8 through 79 for the complex recurrence
* ``standalone_jax/ffm/ffm.py`` lines 54 through 83 for the FFM cell
* ``standalone/ffm/ffm.py`` lines 88 through 103 for the published output gate

The original FFM produces a Markov-state feature for a downstream policy. This
adapter intentionally shares that feature between separate actor and value
heads so it conforms to the registered actor-critic protocol used by every
baseline in this harness. The two heads are not part of the FFM memory model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp

Array = jax.Array
Params = Mapping[str, Array]

_MINIMUM_DECAY = -1e-6
_LAYER_NORM_EPSILON = 1e-5
POPGYM_MEMORY_SIZE = 32
POPGYM_CONTEXT_SIZE = 4
POPGYM_MIN_PERIOD = 1.0
POPGYM_MAX_PERIOD = 1024.0


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
    """Apply the nonparametric layer norm specified in FFM Equation 18."""
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + _LAYER_NORM_EPSILON)


@dataclass(frozen=True)
class FFMPolicyCore:
    """Reset-aware Fast and Forgetful Memory actor-critic policy core.

    Args:
        input_dim: Width of the shared augmented policy input.
        action_dim: Number of discrete actor logits.
        hidden_size: Width of the FFM output and shared actor-critic feature.
        memory_size: Trace width, denoted ``m`` in the FFM paper.
        context_size: Temporal-context width, denoted ``c`` in the FFM paper.
        min_period: Smallest initial oscillation period.
        max_period: Largest initial oscillation period.
    """

    input_dim: int
    action_dim: int
    hidden_size: int
    memory_size: int
    context_size: int
    min_period: float = 1.0
    max_period: float = 1024.0

    def __post_init__(self) -> None:
        dimensions = {
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "hidden_size": self.hidden_size,
            "memory_size": self.memory_size,
            "context_size": self.context_size,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_period <= 0.0:
            raise ValueError("min_period must be positive")
        if self.max_period < self.min_period:
            raise ValueError("max_period must be at least min_period")

    def initialize(self, key: Array) -> dict[str, Array]:
        """Initialize the published FFM cell and shared actor-critic heads."""
        (
            pre_key,
            input_gate_key,
            output_gate_key,
            skip_key,
            mix_key,
            actor_key,
            critic_key,
        ) = jax.random.split(key, 7)
        return {
            "ffm.pre.kernel": _xavier(
                pre_key,
                self.input_dim,
                self.memory_size,
            ),
            "ffm.pre.bias": jnp.zeros((self.memory_size,)),
            "ffm.input_gate.kernel": _xavier(
                input_gate_key,
                self.input_dim,
                self.memory_size,
            ),
            "ffm.input_gate.bias": jnp.zeros((self.memory_size,)),
            "ffm.output_gate.kernel": _xavier(
                output_gate_key,
                self.input_dim,
                self.hidden_size,
            ),
            "ffm.output_gate.bias": jnp.zeros((self.hidden_size,)),
            "ffm.skip.kernel": _xavier(
                skip_key,
                self.input_dim,
                self.hidden_size,
            ),
            "ffm.skip.bias": jnp.zeros((self.hidden_size,)),
            "ffm.mix.kernel": _xavier(
                mix_key,
                2 * self.memory_size * self.context_size,
                self.hidden_size,
            ),
            "ffm.mix.bias": jnp.zeros((self.hidden_size,)),
            # The pinned official JAX source initializes the real exponent from
            # -e to -1e-6 and angular frequency as 2*pi / period.
            "ffm.decay": jnp.linspace(
                -math.e,
                _MINIMUM_DECAY,
                self.memory_size,
            ),
            "ffm.frequency": 2.0
            * jnp.pi
            / jnp.linspace(
                self.min_period,
                self.max_period,
                self.context_size,
            ),
            # Intentional shared-head adaptation. FFM itself ends at `features`.
            "actor.kernel": 0.01 * _xavier(actor_key, self.hidden_size, self.action_dim),
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
        """Return the complex FFM state with shape ``[batch, m, c]``."""
        real_dtype = jnp.dtype(dtype)
        complex_dtype = jnp.complex128 if real_dtype == jnp.dtype(jnp.float64) else jnp.complex64
        return jnp.zeros(
            (batch_size, self.memory_size, self.context_size),
            dtype=complex_dtype,
        )

    @staticmethod
    def _gamma(params: Params) -> Array:
        """Return one-step complex decay and context factors, Equation 12."""
        decay = jnp.minimum(params["ffm.decay"], _MINIMUM_DECAY)
        exponent = jax.lax.complex(
            decay[:, None],
            jnp.broadcast_to(
                params["ffm.frequency"][None, :],
                (decay.shape[0], params["ffm.frequency"].shape[0]),
            ),
        )
        return jnp.exp(exponent)

    def _input_trace(self, params: Params, policy_input: Array) -> Array:
        """Apply the input projection and sigmoid gate from Equation 15."""
        projected = _linear(params, "ffm.pre", policy_input)
        gate = jax.nn.sigmoid(_linear(params, "ffm.input_gate", policy_input))
        return projected * gate

    def _features(self, params: Params, policy_input: Array, state: Array) -> Array:
        """Project complex memory and apply the output gate, Equations 17-18."""
        # Match the pinned standalone JAX layout: real and imaginary context
        # vectors are concatenated before flattening.
        state_parts = jnp.concatenate(
            [jnp.real(state), jnp.imag(state)],
            axis=-1,
        )
        flattened = state_parts.reshape((*state.shape[:-2], -1))
        mixed = _linear(params, "ffm.mix", flattened)
        output_gate = jax.nn.sigmoid(_linear(params, "ffm.output_gate", policy_input))
        skip = _linear(params, "ffm.skip", policy_input)

        # Published Equation 18 and the main official PyTorch implementation
        # normalize z before gating. The experimental standalone JAX file at
        # the audited commit instead normalizes z * gate.
        return _layer_norm(mixed) * output_gate + skip * (1.0 - output_gate)

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
        """Apply one recurrent FFM update with asynchronous batch resets."""
        reset = jnp.asarray(reset, dtype=jnp.bool_)
        trace = self._input_trace(params, policy_input)
        gamma = self._gamma(params)
        retained_state = jnp.where(
            reset[:, None, None],
            jnp.zeros_like(state),
            state,
        )
        new_state = retained_state * gamma[None, :, :] + trace[:, :, None].astype(state.dtype)
        features = self._features(params, policy_input, new_state)
        logits, values = self._heads(params, features)
        return new_state, logits, values

    def apply_sequence(
        self,
        params: Params,
        state: Array,
        policy_inputs: Array,
        resets: Array,
    ) -> tuple[Array, Array, Array]:
        """Apply FFM in parallel over a time-major reset-aware sequence.

        Each timestep is the affine recurrence ``S_t = A_t S_(t-1) + B_t``.
        Associative composition computes every prefix state while preserving
        independent reset boundaries for each batch element.
        """
        resets = jnp.asarray(resets, dtype=jnp.bool_)
        traces = self._input_trace(params, policy_inputs)
        gamma = self._gamma(params)
        multipliers = gamma[None, None, :, :] * jnp.logical_not(resets)[:, :, None, None]
        additions = jnp.broadcast_to(
            traces[:, :, :, None],
            (
                traces.shape[0],
                traces.shape[1],
                self.memory_size,
                self.context_size,
            ),
        ).astype(state.dtype)

        def compose(
            left: tuple[Array, Array],
            right: tuple[Array, Array],
        ) -> tuple[Array, Array]:
            left_multiplier, left_addition = left
            right_multiplier, right_addition = right
            return (
                right_multiplier * left_multiplier,
                right_multiplier * left_addition + right_addition,
            )

        prefix_multipliers, prefix_additions = jax.lax.associative_scan(
            compose,
            (multipliers, additions),
            axis=0,
        )
        states = prefix_multipliers * state[None, ...] + prefix_additions
        features = self._features(params, policy_inputs, states)
        logits, values = self._heads(params, features)
        return states[-1], logits, values

    @staticmethod
    def count_parameters(params: Params) -> int:
        """Count scalar trainable parameters, including both shared heads."""
        return sum(int(value.size) for value in jax.tree.leaves(params))


def match_ffm_hidden_size(
    *,
    target_parameters: int,
    input_dim: int,
    action_dim: int,
    memory_size: int = POPGYM_MEMORY_SIZE,
    context_size: int = POPGYM_CONTEXT_SIZE,
    maximum_width: int = 4096,
) -> int:
    """Find the closest FFM output width without changing memory structure.

    Appendix D.2 of the FFM paper fixes the POPGym recurrent state at
    ``m = 32`` traces and ``c = 4`` complex contexts. That is 128 complex
    values, or 256 real scalar dimensions, matching the recurrent-state size
    of the paper's controls. Parameter matching must not search over ``m`` or
    ``c`` because doing so would alter the published decay and temporal-context
    structure. Instead, this adapter searches only the Markov-state output
    width shared by the actor and critic.
    """
    if target_parameters <= 0:
        raise ValueError("target_parameters must be positive")
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    if memory_size <= 0:
        raise ValueError("memory_size must be positive")
    if context_size <= 0:
        raise ValueError("context_size must be positive")
    if maximum_width <= 0:
        raise ValueError("maximum_width must be positive")

    def parameter_count(hidden_size: int) -> int:
        fixed_parameters = (
            2 * input_dim * memory_size + 3 * memory_size + context_size + action_dim + 1
        )
        parameters_per_hidden_unit = 2 * input_dim + 2 * memory_size * context_size + action_dim + 4
        return fixed_parameters + hidden_size * parameters_per_hidden_unit

    return min(
        range(1, maximum_width + 1),
        key=lambda width: abs(parameter_count(width) - target_parameters),
    )
