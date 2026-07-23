"""Audited observation adapters for privileged POBAX reference environments."""

from __future__ import annotations

from typing import Any, Final, NamedTuple

import jax
import jax.numpy as jnp
from gymnax.environments import spaces

PINNED_POBAX_COMMIT: Final = "a5e1d62d14e4efe783885b9d4f19cffa2a568eec"

BATTLESHIP_PERFECT_RECALL_SOURCE_CONTRACT: Final = {
    "repository": "https://github.com/taodav/pobax",
    "audited_commit": PINNED_POBAX_COMMIT,
    "source_environment": "battleship_10",
    "source_perfect_memory": True,
    "source_observation_shape": [10, 10],
    "source_observation_encoding": {
        "miss": -1,
        "unvisited": 0,
        "hit": 1,
    },
    "adapter_observation": "source observation unchanged",
    "adapter_action_mask": "row-major flatten(source observation == 0)",
    "reference_class": "perfect_recall_history",
}


class BattleshipPerfectRecallObservation(NamedTuple):
    """A raw perfect-recall board paired with its legal-action mask."""

    obs: jax.Array
    action_mask: jax.Array


class BattleshipPerfectRecallObservationWrapper:
    """Add legal-action masks to POBAX's vectorized perfect-recall board.

    The wrapped object must be the final vectorized environment returned by
    ``get_env("battleship_10", ..., perfect_memory=True)``. POBAX deliberately
    leaves Battleship observations unnamed, so this adapter supplies the
    observation interface expected by the shared learner without changing the
    source observation, environment state, reward, termination, or information.
    """

    source_contract: Final = BATTLESHIP_PERFECT_RECALL_SOURCE_CONTRACT

    def __init__(self, environment: Any) -> None:
        self._env = environment

    def __getattr__(self, name: str) -> Any:
        """Delegate the source environment API and audited runtime attributes."""
        return getattr(self._env, name)

    @property
    def gamma(self) -> Any:
        """Return the effective discount exposed by POBAX's final wrapper."""
        return self._env.gamma

    def action_space(self, params: Any = None) -> Any:
        """Delegate the action-space query without changing its result."""
        return self._env.action_space(params)

    def observation_space(self, params: Any = None) -> spaces.Dict:
        """Declare the exact named observation emitted by this adapter."""
        del params
        return spaces.Dict(
            {
                "obs": spaces.Box(
                    low=-1,
                    high=1,
                    shape=(10, 10),
                    dtype=jnp.int32,
                ),
                "action_mask": spaces.Box(
                    low=False,
                    high=True,
                    shape=(100,),
                    dtype=jnp.bool_,
                ),
            }
        )

    def dummy_observation(
        self,
        num_envs: int,
        params: Any = None,
    ) -> BattleshipPerfectRecallObservation:
        """Return the sequence-first dummy batch expected by POBAX tooling."""
        del params
        return BattleshipPerfectRecallObservation(
            obs=jnp.zeros((1, num_envs, 10, 10), dtype=jnp.int32),
            action_mask=jnp.ones((1, num_envs, 100), dtype=jnp.bool_),
        )

    @staticmethod
    def _adapt(raw_observation: jax.Array) -> BattleshipPerfectRecallObservation:
        if raw_observation.ndim < 2 or raw_observation.shape[-2:] != (10, 10):
            raise ValueError(
                "Battleship perfect-recall observations must end in a 10x10 board; "
                f"found shape {raw_observation.shape}"
            )
        action_mask = jnp.equal(raw_observation, 0).reshape(raw_observation.shape[:-2] + (100,))
        return BattleshipPerfectRecallObservation(
            obs=raw_observation,
            action_mask=action_mask,
        )

    def reset(
        self,
        key: jax.Array,
        params: Any = None,
    ) -> tuple[BattleshipPerfectRecallObservation, Any]:
        """Reset the source environment and add the legal-action mask."""
        raw_observation, state = self._env.reset(key, params)
        return self._adapt(raw_observation), state

    def step(
        self,
        key: jax.Array,
        state: Any,
        action: jax.Array,
        params: Any = None,
    ) -> tuple[
        BattleshipPerfectRecallObservation,
        Any,
        jax.Array,
        jax.Array,
        Any,
    ]:
        """Step the source environment and add the updated legal-action mask."""
        raw_observation, next_state, reward, done, info = self._env.step(
            key,
            state,
            action,
            params,
        )
        return self._adapt(raw_observation), next_state, reward, done, info
