"""Shared recurrent policy-core interface for the JAX PPO learner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp

from benchmarks.pobax.arcmind_reference import (
    ArcMindState,
    ReferenceConfig,
    arcmind_actor_critic_step,
    init_stream_state,
    initialize_actor_critic_params,
)

Array = jax.Array


def augment_policy_input(
    observation: Array,
    previous_action: Array,
    previous_reward: Array,
    reset: Array,
    *,
    action_dim: int,
    continuous_action: bool = False,
) -> Array:
    """Build the registered causal policy input.

    Previous-action and previous-reward features are zeroed at episode
    boundaries. The reset indicator remains explicit, so the input contract is
    deterministic and cannot leak the terminal transition into the new episode.
    """
    reset = jnp.asarray(reset, dtype=jnp.bool_)
    observation = observation.reshape((observation.shape[0], -1))
    if continuous_action:
        previous_action = previous_action.reshape(
            (previous_action.shape[0], action_dim)
        )
    else:
        previous_action = jax.nn.one_hot(previous_action, action_dim)
    previous_action = jnp.where(
        reset[:, None],
        jnp.zeros_like(previous_action),
        previous_action,
    )
    previous_reward = jnp.where(
        reset,
        jnp.zeros_like(previous_reward),
        previous_reward,
    )
    return jnp.concatenate(
        [
            observation,
            previous_action,
            previous_reward[:, None],
            reset[:, None].astype(observation.dtype),
        ],
        axis=-1,
    )


@dataclass(frozen=True)
class ArcMindPolicyCore:
    """Trainable ArcMind adapter with time-major sequence semantics."""

    config: ReferenceConfig

    def initialize(self, key: Array) -> dict[str, Array]:
        """Initialize trainable actor-critic parameters."""
        return initialize_actor_critic_params(key, self.config)

    def initial_state(
        self,
        batch_size: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> ArcMindState:
        """Initialize recurrent policy state."""
        return init_stream_state(self.config, batch_size, dtype=dtype)

    def step(
        self,
        params: Mapping[str, Array],
        state: ArcMindState,
        policy_input: Array,
        reset: Array,
    ) -> tuple[ArcMindState, Array, Array]:
        """Apply one environment step."""
        return arcmind_actor_critic_step(
            params,
            state,
            policy_input,
            self.config,
            reset=reset,
        )

    def apply_sequence(
        self,
        params: Mapping[str, Array],
        state: ArcMindState,
        policy_inputs: Array,
        resets: Array,
    ) -> tuple[ArcMindState, Array, Array]:
        """Apply a time-major sequence with reset-aware recurrent state."""

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
    def count_parameters(params: Mapping[str, Array]) -> int:
        """Count scalar trainable parameters."""
        return sum(int(value.size) for value in jax.tree.leaves(params))

    def count_effective_parameters(self, params: Mapping[str, Array]) -> int:
        """Count parameters reached by the configured ablation's forward pass."""

        def active(name: str) -> bool:
            if name.startswith("ssm_core.") and self.config.ablate_ssm:
                return False
            slow_disabled = self.config.ablate_attention
            if name.startswith("slow_attention.layers.") and slow_disabled:
                return False
            if name.startswith("slow_attention.memory_age_embedding."):
                return not (
                    slow_disabled
                    or self.config.ablate_memory
                    or self.config.ablate_temporal_encoding
                )
            if name.startswith("memory.compressor."):
                return not (slow_disabled or self.config.ablate_memory)
            if name.startswith("gate."):
                return not (
                    slow_disabled
                    or self.config.ablate_ssm
                    or self.config.ablate_gating
                )
            return True

        return sum(
            int(value.size)
            for name, value in params.items()
            if active(name)
        )
