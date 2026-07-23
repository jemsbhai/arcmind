"""Regression tests for privileged-observation environment adapters."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.pobax.upper_reference_envs import (
    BATTLESHIP_PERFECT_RECALL_SOURCE_CONTRACT,
    PINNED_POBAX_COMMIT,
    BattleshipPerfectRecallObservation,
    BattleshipPerfectRecallObservationWrapper,
)


class FakeBattleshipVectorEnvironment:
    """Small vector contract double with deterministic Battleship transitions."""

    gamma = 0.97
    delegated_marker = object()

    def __init__(self) -> None:
        self.action_space_result = object()

    @staticmethod
    def _initial_board(batch_size: int) -> jax.Array:
        board = jnp.zeros((batch_size, 10, 10), dtype=jnp.float32)
        row = jnp.arange(batch_size) % 10
        miss_col = jnp.arange(batch_size) % 5
        hit_col = miss_col + 5
        board = board.at[jnp.arange(batch_size), row, miss_col].set(-1.0)
        return board.at[jnp.arange(batch_size), row, hit_col].set(1.0)

    def reset(self, keys: jax.Array, params: Any = None):
        del params
        board = self._initial_board(keys.shape[0])
        state = {
            "board": board,
            "turn": jnp.zeros((keys.shape[0],), dtype=jnp.int32),
        }
        return board, state

    def step(
        self,
        keys: jax.Array,
        state: dict[str, jax.Array],
        action: jax.Array,
        params: Any = None,
    ):
        del keys, params
        batch = jnp.arange(action.shape[0])
        row = action // 10
        column = action % 10
        next_board = state["board"].at[batch, row, column].set(1.0)
        next_state = {
            "board": next_board,
            "turn": state["turn"] + 1,
        }
        reward = action.astype(jnp.float32) / 100.0
        done = action == 99
        info = {"selected_action": action, "turn": next_state["turn"]}
        return next_board, next_state, reward, done, info

    def action_space(self, params: Any = None) -> object:
        del params
        return self.action_space_result


def _reset(
    *,
    batch_size: int = 3,
) -> tuple[
    FakeBattleshipVectorEnvironment,
    BattleshipPerfectRecallObservationWrapper,
    BattleshipPerfectRecallObservation,
    dict[str, jax.Array],
]:
    source = FakeBattleshipVectorEnvironment()
    wrapper = BattleshipPerfectRecallObservationWrapper(source)
    keys = jax.random.split(jax.random.PRNGKey(41), batch_size)
    observation, state = wrapper.reset(keys, params={"ignored": True})
    return source, wrapper, observation, state


def test_reset_preserves_raw_board_and_builds_exact_flat_mask() -> None:
    source, _, observation, _ = _reset()
    expected = source._initial_board(3)

    assert isinstance(observation, BattleshipPerfectRecallObservation)
    assert observation.obs is expected or np.array_equal(observation.obs, expected)
    assert observation.obs.dtype == jnp.float32
    assert observation.obs.shape == (3, 10, 10)
    assert observation.action_mask.dtype == jnp.bool_
    assert observation.action_mask.shape == (3, 100)
    np.testing.assert_array_equal(
        observation.action_mask,
        np.asarray(expected == 0).reshape(3, 100),
    )


def test_action_masks_are_independent_across_vector_batches() -> None:
    _, _, observation, _ = _reset(batch_size=4)

    for batch_index in range(4):
        false_indices = np.flatnonzero(~np.asarray(observation.action_mask[batch_index]))
        expected = np.asarray(
            [
                batch_index * 10 + batch_index,
                batch_index * 10 + batch_index + 5,
            ]
        )
        np.testing.assert_array_equal(false_indices, expected)


def test_step_changes_only_selected_action_mask_and_preserves_outputs() -> None:
    _, wrapper, observation, state = _reset()
    actions = jnp.asarray([1, 12, 23], dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(42), 3)

    expected_raw, expected_state, expected_reward, expected_done, expected_info = wrapper._env.step(
        keys, state, actions, params=None
    )
    next_observation, next_state, reward, done, info = wrapper.step(
        keys,
        state,
        actions,
        params=None,
    )

    np.testing.assert_array_equal(next_observation.obs, expected_raw)
    np.testing.assert_array_equal(next_observation.action_mask, expected_raw.reshape(3, 100) == 0)
    np.testing.assert_array_equal(next_state["board"], expected_state["board"])
    np.testing.assert_array_equal(next_state["turn"], expected_state["turn"])
    np.testing.assert_array_equal(reward, expected_reward)
    np.testing.assert_array_equal(done, expected_done)
    np.testing.assert_array_equal(info["selected_action"], expected_info["selected_action"])
    np.testing.assert_array_equal(info["turn"], expected_info["turn"])

    for batch_index, action in enumerate(np.asarray(actions)):
        assert bool(observation.action_mask[batch_index, action])
        assert not bool(next_observation.action_mask[batch_index, action])
        unchanged = np.ones((100,), dtype=bool)
        unchanged[action] = False
        np.testing.assert_array_equal(
            np.asarray(next_observation.action_mask[batch_index])[unchanged],
            np.asarray(observation.action_mask[batch_index])[unchanged],
        )


def test_delegates_action_space_gamma_and_arbitrary_attributes() -> None:
    source = FakeBattleshipVectorEnvironment()
    wrapper = BattleshipPerfectRecallObservationWrapper(source)

    assert wrapper.action_space({"ignored": True}) is source.action_space_result
    assert wrapper.gamma == source.gamma
    assert wrapper.delegated_marker is source.delegated_marker
    assert wrapper._env is source


def test_declared_space_and_dummy_observation_match_adapter_output() -> None:
    _, wrapper, observation, _ = _reset(batch_size=3)

    observation_space = wrapper.observation_space(params={"ignored": True})
    assert set(observation_space.spaces) == {"obs", "action_mask"}
    assert observation_space.spaces["obs"].shape == observation.obs.shape[1:]
    assert observation_space.spaces["obs"].dtype == jnp.int32
    assert observation_space.spaces["action_mask"].shape == observation.action_mask.shape[1:]
    assert observation_space.spaces["action_mask"].dtype == jnp.bool_

    dummy = wrapper.dummy_observation(3, params={"ignored": True})
    assert isinstance(dummy, BattleshipPerfectRecallObservation)
    assert dummy.obs.shape == (1, 3, 10, 10)
    assert dummy.obs.dtype == jnp.int32
    assert dummy.action_mask.shape == (1, 3, 100)
    assert dummy.action_mask.dtype == jnp.bool_
    assert bool(jnp.all(dummy.action_mask))


def test_reset_and_step_return_source_outputs_by_identity() -> None:
    raw_reset = jnp.zeros((2, 10, 10), dtype=jnp.float32)
    raw_step = raw_reset.at[0, 3, 4].set(-1.0)
    reset_state = object()
    next_state = object()
    reward = object()
    done = object()
    info = object()

    class IdentityEnvironment:
        gamma = 1.0

        @staticmethod
        def reset(key, params=None):
            del key, params
            return raw_reset, reset_state

        @staticmethod
        def step(key, state, action, params=None):
            del key, action, params
            assert state is reset_state
            return raw_step, next_state, reward, done, info

        @staticmethod
        def action_space(params=None):
            del params
            return object()

    wrapper = BattleshipPerfectRecallObservationWrapper(IdentityEnvironment())
    observation, returned_reset_state = wrapper.reset(jax.random.PRNGKey(51))
    (
        next_observation,
        returned_next_state,
        returned_reward,
        returned_done,
        returned_info,
    ) = wrapper.step(
        jax.random.PRNGKey(52),
        returned_reset_state,
        jnp.asarray([34, 35], dtype=jnp.int32),
    )

    assert observation.obs is raw_reset
    assert returned_reset_state is reset_state
    assert next_observation.obs is raw_step
    assert returned_next_state is next_state
    assert returned_reward is reward
    assert returned_done is done
    assert returned_info is info


def test_observation_named_tuple_is_a_jax_pytree() -> None:
    _, _, observation, _ = _reset()

    leaves, treedef = jax.tree_util.tree_flatten(observation)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert len(leaves) == 2
    assert isinstance(rebuilt, BattleshipPerfectRecallObservation)
    np.testing.assert_array_equal(rebuilt.obs, observation.obs)
    np.testing.assert_array_equal(rebuilt.action_mask, observation.action_mask)


def test_reset_and_step_are_safe_under_vectorized_jit() -> None:
    _, wrapper, _, _ = _reset()
    reset_jit = jax.jit(wrapper.reset)
    step_jit = jax.jit(wrapper.step)
    keys = jax.random.split(jax.random.PRNGKey(43), 5)

    observation, state = reset_jit(keys, None)
    actions = jnp.asarray([1, 12, 23, 34, 45], dtype=jnp.int32)
    next_observation, next_state, reward, done, info = step_jit(
        keys,
        state,
        actions,
        None,
    )
    jax.block_until_ready((next_observation, next_state, reward, done, info))

    assert observation.obs.shape == (5, 10, 10)
    assert observation.action_mask.shape == (5, 100)
    assert next_observation.obs.shape == (5, 10, 10)
    assert next_observation.action_mask.shape == (5, 100)
    np.testing.assert_array_equal(
        next_observation.action_mask,
        next_observation.obs.reshape(5, 100) == 0,
    )


def test_static_shape_contract_rejects_source_drift_under_jit() -> None:
    wrapper = BattleshipPerfectRecallObservationWrapper(FakeBattleshipVectorEnvironment())

    with pytest.raises(ValueError, match="10x10"):
        jax.jit(wrapper._adapt)(jnp.zeros((2, 9, 10), dtype=jnp.float32))


def test_source_contract_records_exact_pinned_semantics() -> None:
    assert PINNED_POBAX_COMMIT == "a5e1d62d14e4efe783885b9d4f19cffa2a568eec"
    assert BATTLESHIP_PERFECT_RECALL_SOURCE_CONTRACT == {
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


def test_pinned_pobax_vector_environment_contract_under_jit() -> None:
    pobax_envs = pytest.importorskip("pobax.envs")
    source, params = pobax_envs.get_env(
        "battleship_10",
        jax.random.PRNGKey(61),
        num_envs=2,
        perfect_memory=True,
    )
    wrapper = BattleshipPerfectRecallObservationWrapper(source)
    keys = jax.random.split(jax.random.PRNGKey(62), 2)
    raw_observation, _ = source.reset(keys, params)
    compiled_reset = jax.jit(lambda reset_keys: wrapper.reset(reset_keys, params))
    compiled_step = jax.jit(
        lambda step_keys, state, actions: wrapper.step(
            step_keys,
            state,
            actions,
            params,
        )
    )

    observation, state = compiled_reset(keys)
    observation_space = wrapper.observation_space(params)
    dummy = wrapper.dummy_observation(2, params)
    actions = jnp.asarray([0, 99], dtype=jnp.int32)
    next_observation, next_state, reward, done, info = compiled_step(
        keys,
        state,
        actions,
    )
    jax.block_until_ready((next_observation, next_state, reward, done, info))

    np.testing.assert_array_equal(observation.obs, raw_observation)
    assert observation.obs.dtype == raw_observation.dtype
    assert observation.obs.shape == (2, 10, 10)
    assert observation.action_mask.shape == (2, 100)
    assert observation_space.spaces["obs"].shape == observation.obs.shape[1:]
    assert observation_space.spaces["obs"].dtype == observation.obs.dtype
    assert observation_space.spaces["action_mask"].shape == observation.action_mask.shape[1:]
    assert observation_space.spaces["action_mask"].dtype == observation.action_mask.dtype
    assert dummy.obs.shape == (1, 2, 10, 10)
    assert dummy.action_mask.shape == (1, 2, 100)
    np.testing.assert_array_equal(
        observation.action_mask,
        observation.obs.reshape(2, 100) == 0,
    )
    np.testing.assert_array_equal(
        next_observation.action_mask,
        next_observation.obs.reshape(2, 100) == 0,
    )
    assert not bool(next_observation.action_mask[0, 0])
    assert not bool(next_observation.action_mask[1, 99])
