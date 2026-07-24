"""Source-compatible Memory Traces policy core for the shared POBAX learner.

The recurrence, feature ordering, and actor-critic architecture follow the
official ICML 2025 repository at commit
``fcfdacc0b0a06dc181b49b9ef95893dbae7f2bcd``. The source example configures
decays ``[0.0, 0.985]`` for TMaze64. Those values are frozen here as the
source-compatible lane and are not represented as an author-selected POBAX
setting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp

from benchmarks.pobax.model_registry import MEMORY_TRACE_DECAYS

Array = jax.Array
Params = Mapping[str, Array]
OFFICIAL_MEMORY_TRACE_HIDDEN_SIZE = 64


class OfficialMemoryTraceState(NamedTuple):
    """Observation traces with trace index before observation index."""

    traces: Array


def _linear(params: Params, prefix: str, values: Array) -> Array:
    return values @ params[f"{prefix}.kernel"] + params[f"{prefix}.bias"]


def _orthogonal(
    key: Array,
    input_features: int,
    output_features: int,
    gain: float,
) -> Array:
    initializer = jax.nn.initializers.orthogonal(gain)
    return initializer(key, (input_features, output_features), jnp.float32)


@dataclass(frozen=True)
class OfficialMemoryTracePolicyCore:
    """Observation-only traces with the official separate 2x64 networks."""

    input_dim: int
    observation_dim: int
    action_dim: int
    hidden_size: int = OFFICIAL_MEMORY_TRACE_HIDDEN_SIZE
    decays: tuple[float, ...] = MEMORY_TRACE_DECAYS

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError("input_dim must be positive")
        if self.observation_dim < 1 or self.observation_dim > self.input_dim:
            raise ValueError("observation_dim must lie in [1, input_dim]")
        if self.action_dim < 1:
            raise ValueError("action_dim must be positive")
        if self.hidden_size != OFFICIAL_MEMORY_TRACE_HIDDEN_SIZE:
            raise ValueError("official Memory Traces hidden_size must equal 64")
        if tuple(self.decays) != MEMORY_TRACE_DECAYS:
            raise ValueError(
                f"official Memory Traces decays must equal {MEMORY_TRACE_DECAYS}"
            )

    @property
    def trace_feature_dim(self) -> int:
        return self.observation_dim * len(self.decays)

    def initialize(self, key: Array) -> dict[str, Array]:
        """Use the exact orthogonal gains of the official ActorCritic."""

        keys = iter(jax.random.split(key, 6))
        hidden_gain = math.sqrt(2.0)
        return {
            "actor.hidden.0.kernel": _orthogonal(
                next(keys),
                self.trace_feature_dim,
                self.hidden_size,
                hidden_gain,
            ),
            "actor.hidden.0.bias": jnp.zeros((self.hidden_size,)),
            "actor.hidden.1.kernel": _orthogonal(
                next(keys),
                self.hidden_size,
                self.hidden_size,
                hidden_gain,
            ),
            "actor.hidden.1.bias": jnp.zeros((self.hidden_size,)),
            "actor.output.kernel": _orthogonal(
                next(keys),
                self.hidden_size,
                self.action_dim,
                0.01,
            ),
            "actor.output.bias": jnp.zeros((self.action_dim,)),
            "critic.hidden.0.kernel": _orthogonal(
                next(keys),
                self.trace_feature_dim,
                self.hidden_size,
                hidden_gain,
            ),
            "critic.hidden.0.bias": jnp.zeros((self.hidden_size,)),
            "critic.hidden.1.kernel": _orthogonal(
                next(keys),
                self.hidden_size,
                self.hidden_size,
                hidden_gain,
            ),
            "critic.hidden.1.bias": jnp.zeros((self.hidden_size,)),
            "critic.output.kernel": _orthogonal(
                next(keys),
                self.hidden_size,
                1,
                1.0,
            ),
            "critic.output.bias": jnp.zeros((1,)),
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> OfficialMemoryTraceState:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return OfficialMemoryTraceState(
            traces=jnp.zeros(
                (batch_size, len(self.decays), self.observation_dim),
                dtype=dtype,
            )
        )

    @staticmethod
    def _network(params: Params, prefix: str, features: Array) -> Array:
        features = jnp.tanh(_linear(params, f"{prefix}.hidden.0", features))
        features = jnp.tanh(_linear(params, f"{prefix}.hidden.1", features))
        return _linear(params, f"{prefix}.output", features)

    def step(
        self,
        params: Params,
        state: OfficialMemoryTraceState,
        policy_input: Array,
        reset: Array,
    ) -> tuple[OfficialMemoryTraceState, Array, Array]:
        """Reset each worker, then incorporate its current raw observation."""

        observation = policy_input[..., : self.observation_dim]
        traces = jnp.where(
            jnp.asarray(reset, dtype=jnp.bool_)[:, None, None],
            jnp.zeros_like(state.traces),
            state.traces,
        )
        decays = jnp.asarray(self.decays, dtype=observation.dtype)
        traces = (
            (1.0 - decays)[None, :, None] * observation[:, None, :]
            + decays[None, :, None] * traces
        )
        flattened = traces.reshape((traces.shape[0], self.trace_feature_dim))
        logits = self._network(params, "actor", flattened)
        values = self._network(params, "critic", flattened)[..., 0]
        return OfficialMemoryTraceState(traces), logits, values

    def apply_sequence(
        self,
        params: Params,
        state: OfficialMemoryTraceState,
        policy_inputs: Array,
        resets: Array,
    ) -> tuple[OfficialMemoryTraceState, Array, Array]:
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

    def expected_parameter_count(self) -> int:
        return official_memory_trace_parameter_count(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
        )


def official_memory_trace_parameter_count(
    *,
    observation_dim: int,
    action_dim: int,
) -> int:
    """Return the exact count of the two independent official networks."""

    if observation_dim < 1:
        raise ValueError("observation_dim must be positive")
    if action_dim < 1:
        raise ValueError("action_dim must be positive")
    feature_dim = len(MEMORY_TRACE_DECAYS) * observation_dim
    actor = (
        feature_dim * OFFICIAL_MEMORY_TRACE_HIDDEN_SIZE
        + OFFICIAL_MEMORY_TRACE_HIDDEN_SIZE
        + OFFICIAL_MEMORY_TRACE_HIDDEN_SIZE**2
        + OFFICIAL_MEMORY_TRACE_HIDDEN_SIZE
        + OFFICIAL_MEMORY_TRACE_HIDDEN_SIZE * action_dim
        + action_dim
    )
    critic = (
        feature_dim * OFFICIAL_MEMORY_TRACE_HIDDEN_SIZE
        + OFFICIAL_MEMORY_TRACE_HIDDEN_SIZE
        + OFFICIAL_MEMORY_TRACE_HIDDEN_SIZE**2
        + OFFICIAL_MEMORY_TRACE_HIDDEN_SIZE
        + OFFICIAL_MEMORY_TRACE_HIDDEN_SIZE
        + 1
    )
    return actor + critic
