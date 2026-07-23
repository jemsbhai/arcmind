"""Architecture-agnostic recurrent PPO for controlled POBAX comparisons."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from benchmarks.pobax.policy_core import augment_policy_input

Array = jax.Array
Params = Mapping[str, Array]


@dataclass(frozen=True)
class PPOConfig:
    """Common optimization and collection settings for every policy core."""

    total_steps: int = 131_072
    num_envs: int = 64
    rollout_steps: int = 64
    update_epochs: int = 4
    num_minibatches: int = 4
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5

    @property
    def steps_per_update(self) -> int:
        return self.num_envs * self.rollout_steps

    @property
    def num_updates(self) -> int:
        return self.total_steps // self.steps_per_update

    def validate(self) -> None:
        if self.total_steps < self.steps_per_update:
            raise ValueError("total_steps must cover at least one rollout")
        if self.num_envs % self.num_minibatches != 0:
            raise ValueError("num_envs must be divisible by num_minibatches")


class RunnerState(NamedTuple):
    observation: Any
    environment_state: Any
    policy_state: Any
    previous_action: Array
    previous_reward: Array
    done: Array
    random_key: Array


class Rollout(NamedTuple):
    policy_input: Array
    reset: Array
    action: Array
    old_log_probability: Array
    old_value: Array
    reward: Array
    done: Array
    episode_return: Array
    episode_complete: Array
    action_mask: Array | None = None


class PPOTrainResult(NamedTuple):
    params: Params
    runner: RunnerState
    recent_episode_returns: tuple[float, ...]
    final_metrics: dict[str, float]


def categorical_log_probability(logits: Array, actions: Array) -> Array:
    """Log probability of selected categorical actions."""
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(
        log_probabilities,
        actions[..., None],
        axis=-1,
    )[..., 0]


def categorical_entropy(logits: Array) -> Array:
    """Entropy of a categorical distribution parameterized by logits."""
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    probabilities = jnp.exp(log_probabilities)
    return -jnp.sum(probabilities * log_probabilities, axis=-1)


def gaussian_log_probability(
    means: Array,
    log_standard_deviation: Array,
    actions: Array,
) -> Array:
    """Log probability under a diagonal Gaussian policy."""
    standardized = (actions - means) * jnp.exp(-log_standard_deviation)
    per_dimension = (
        -0.5 * jnp.square(standardized)
        - log_standard_deviation
        - 0.5 * jnp.log(2.0 * jnp.pi)
    )
    return jnp.sum(per_dimension, axis=-1)


def gaussian_entropy(log_standard_deviation: Array) -> Array:
    """Entropy of a diagonal Gaussian, summed over action dimensions."""
    return jnp.sum(
        log_standard_deviation + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e),
        axis=-1,
    )


class SharedPPO:
    """One collector and PPO update path shared by all policy cores."""

    def __init__(
        self,
        *,
        policy_core: Any,
        environment: Any,
        environment_params: Any,
        action_dim: int,
        config: PPOConfig,
        continuous_action: bool = False,
    ):
        config.validate()
        self.policy_core = policy_core
        self.environment = environment
        self.environment_params = environment_params
        self.action_dim = action_dim
        self.continuous_action = continuous_action
        self.config = config
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_gradient_norm),
            optax.adam(config.learning_rate, eps=1e-5),
        )
        self._collect_jit = jax.jit(self._collect)
        self._advantages_jit = jax.jit(self._advantages)
        self._update_minibatch_jit = jax.jit(self._update_minibatch)
        self._evaluate_jit = jax.jit(self._evaluate_steps, static_argnums=(2,))

    def _mask_logits(self, observation: Any, logits: Array) -> Array:
        if self.continuous_action:
            return logits
        action_mask = observation.action_mask
        if action_mask is None:
            return logits
        return jnp.where(action_mask.astype(jnp.bool_), logits, -1e9)

    def _mask_rollout_logits(
        self,
        action_mask: Array | None,
        logits: Array,
    ) -> Array:
        """Restore the collection-time categorical support during PPO updates."""
        if self.continuous_action or action_mask is None:
            return logits
        return jnp.where(action_mask.astype(jnp.bool_), logits, -1e9)

    def initialize_runner(self, key: Array) -> RunnerState:
        """Reset vector environments and the policy recurrence."""
        reset_key, runner_key = jax.random.split(key)
        observation, environment_state = self.environment.reset(
            jax.random.split(reset_key, self.config.num_envs),
            self.environment_params,
        )
        return RunnerState(
            observation=observation,
            environment_state=environment_state,
            policy_state=self.policy_core.initial_state(self.config.num_envs),
            previous_action=(
                jnp.zeros(
                    (self.config.num_envs, self.action_dim),
                    dtype=jnp.float32,
                )
                if self.continuous_action
                else jnp.zeros((self.config.num_envs,), dtype=jnp.int32)
            ),
            previous_reward=jnp.zeros((self.config.num_envs,), dtype=jnp.float32),
            done=jnp.ones((self.config.num_envs,), dtype=jnp.bool_),
            random_key=runner_key,
        )

    def _policy_input(self, runner: RunnerState) -> Array:
        return augment_policy_input(
            runner.observation.obs,
            runner.previous_action,
            runner.previous_reward,
            runner.done,
            action_dim=self.action_dim,
            continuous_action=self.continuous_action,
        )

    def _collect(
        self,
        params: Params,
        runner: RunnerState,
    ) -> tuple[RunnerState, Any, Rollout, Array]:
        initial_policy_state = runner.policy_state

        def environment_step(carry: RunnerState, unused):
            del unused
            random_key, action_key, environment_key = jax.random.split(
                carry.random_key,
                3,
            )
            policy_input = self._policy_input(carry)
            policy_state, logits, value = self.policy_core.step(
                params,
                carry.policy_state,
                policy_input,
                carry.done,
            )
            logits = self._mask_logits(carry.observation, logits)
            if self.continuous_action:
                log_standard_deviation = params["distribution.log_std"]
                action = logits + jnp.exp(
                    log_standard_deviation
                ) * jax.random.normal(action_key, logits.shape)
                log_probability = gaussian_log_probability(
                    logits,
                    log_standard_deviation,
                    action,
                )
            else:
                action = jax.random.categorical(action_key, logits).astype(
                    jnp.int32
                )
                log_probability = categorical_log_probability(logits, action)
            (
                observation,
                environment_state,
                reward,
                done,
                info,
            ) = self.environment.step(
                jax.random.split(environment_key, self.config.num_envs),
                carry.environment_state,
                action,
                self.environment_params,
            )
            reward = jnp.asarray(reward, dtype=jnp.float32)
            transition = Rollout(
                policy_input=policy_input,
                reset=carry.done,
                action=action,
                old_log_probability=log_probability,
                old_value=value,
                reward=reward,
                done=done,
                episode_return=info["returned_episode_returns"],
                episode_complete=info["returned_episode"],
                action_mask=(
                    None
                    if self.continuous_action
                    else carry.observation.action_mask
                ),
            )
            new_carry = RunnerState(
                observation=observation,
                environment_state=environment_state,
                policy_state=policy_state,
                previous_action=action,
                previous_reward=reward,
                done=done,
                random_key=random_key,
            )
            return new_carry, transition

        runner, rollout = jax.lax.scan(
            environment_step,
            runner,
            None,
            length=self.config.rollout_steps,
        )
        bootstrap_input = self._policy_input(runner)
        _, bootstrap_logits, bootstrap_value = self.policy_core.step(
            params,
            runner.policy_state,
            bootstrap_input,
            runner.done,
        )
        del bootstrap_logits
        return runner, initial_policy_state, rollout, bootstrap_value

    def _advantages(
        self,
        rollout: Rollout,
        bootstrap_value: Array,
    ) -> tuple[Array, Array]:
        def reverse_step(carry, transition):
            advantage, next_value = carry
            reward, done, value = transition
            delta = reward + self.config.gamma * next_value * (1.0 - done) - value
            advantage = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * (1.0 - done)
                * advantage
            )
            return (advantage, value), advantage

        (_, _), advantages = jax.lax.scan(
            reverse_step,
            (jnp.zeros_like(bootstrap_value), bootstrap_value),
            (rollout.reward, rollout.done, rollout.old_value),
            reverse=True,
        )
        return advantages, advantages + rollout.old_value

    def _loss(
        self,
        params: Params,
        initial_policy_state: Any,
        rollout: Rollout,
        advantages: Array,
        targets: Array,
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        _, logits, values = self.policy_core.apply_sequence(
            params,
            initial_policy_state,
            rollout.policy_input,
            rollout.reset,
        )
        logits = self._mask_rollout_logits(rollout.action_mask, logits)
        if self.continuous_action:
            log_standard_deviation = params["distribution.log_std"]
            new_log_probability = gaussian_log_probability(
                logits,
                log_standard_deviation,
                rollout.action,
            )
        else:
            new_log_probability = categorical_log_probability(
                logits,
                rollout.action,
            )
        probability_ratio = jnp.exp(
            new_log_probability - rollout.old_log_probability
        )
        unclipped_actor = probability_ratio * advantages
        clipped_actor = jnp.clip(
            probability_ratio,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        ) * advantages
        actor_loss = -jnp.mean(jnp.minimum(unclipped_actor, clipped_actor))

        clipped_value = rollout.old_value + jnp.clip(
            values - rollout.old_value,
            -self.config.clip_epsilon,
            self.config.clip_epsilon,
        )
        value_loss = 0.5 * jnp.mean(
            jnp.maximum(
                jnp.square(values - targets),
                jnp.square(clipped_value - targets),
            )
        )
        if self.continuous_action:
            entropy = jnp.mean(
                gaussian_entropy(params["distribution.log_std"])
            )
        else:
            entropy = jnp.mean(categorical_entropy(logits))
        approximate_kl = jnp.mean(
            (probability_ratio - 1.0)
            - (new_log_probability - rollout.old_log_probability)
        )
        total_loss = (
            actor_loss
            + self.config.value_coefficient * value_loss
            - self.config.entropy_coefficient * entropy
        )
        return total_loss, (actor_loss, value_loss, entropy, approximate_kl)

    def _update_minibatch(
        self,
        params: Params,
        optimizer_state: optax.OptState,
        initial_policy_state: Any,
        rollout: Rollout,
        advantages: Array,
        targets: Array,
    ) -> tuple[Params, optax.OptState, tuple[Array, Array, Array, Array, Array]]:
        (loss, auxiliary), gradients = jax.value_and_grad(
            self._loss,
            has_aux=True,
        )(
            params,
            initial_policy_state,
            rollout,
            advantages,
            targets,
        )
        updates, optimizer_state = self.optimizer.update(
            gradients,
            optimizer_state,
            params,
        )
        params = optax.apply_updates(params, updates)
        return params, optimizer_state, (loss, *auxiliary)

    @staticmethod
    def _select_environments(tree: Any, indices: Array, *, time_major: bool) -> Any:
        axis = 1 if time_major else 0
        return jax.tree.map(lambda value: jnp.take(value, indices, axis=axis), tree)

    def train(
        self,
        *,
        parameter_key: Array,
        runner_key: Array,
        shuffle_key: Array,
        progress: Callable[[int, dict[str, float]], None] | None = None,
    ) -> PPOTrainResult:
        """Train for the fixed registered interaction budget."""
        params = self.policy_core.initialize(parameter_key)
        if self.continuous_action:
            params = {
                **params,
                "distribution.log_std": jnp.zeros((self.action_dim,)),
            }
        optimizer_state = self.optimizer.init(params)
        runner = self.initialize_runner(runner_key)
        recent_returns: deque[float] = deque(maxlen=1_000)
        final_metrics: dict[str, float] = {}
        minibatch_size = self.config.num_envs // self.config.num_minibatches

        for update_index in range(self.config.num_updates):
            (
                runner,
                initial_policy_state,
                rollout,
                bootstrap_value,
            ) = self._collect_jit(params, runner)
            advantages, targets = self._advantages_jit(rollout, bootstrap_value)
            advantages = (advantages - jnp.mean(advantages)) / (
                jnp.std(advantages) + 1e-8
            )

            completed = np.asarray(rollout.episode_complete, dtype=bool)
            returns = np.asarray(rollout.episode_return)
            recent_returns.extend(float(value) for value in returns[completed])

            update_metrics = []
            for _ in range(self.config.update_epochs):
                shuffle_key, permutation_key = jax.random.split(shuffle_key)
                permutation = jax.random.permutation(
                    permutation_key,
                    self.config.num_envs,
                )
                for minibatch_index in range(self.config.num_minibatches):
                    start = minibatch_index * minibatch_size
                    indices = permutation[start : start + minibatch_size]
                    minibatch_initial_state = self._select_environments(
                        initial_policy_state,
                        indices,
                        time_major=False,
                    )
                    minibatch_rollout = self._select_environments(
                        rollout,
                        indices,
                        time_major=True,
                    )
                    minibatch_advantages = jnp.take(advantages, indices, axis=1)
                    minibatch_targets = jnp.take(targets, indices, axis=1)
                    (
                        params,
                        optimizer_state,
                        metrics,
                    ) = self._update_minibatch_jit(
                        params,
                        optimizer_state,
                        minibatch_initial_state,
                        minibatch_rollout,
                        minibatch_advantages,
                        minibatch_targets,
                    )
                    update_metrics.append(metrics)

            stacked_metrics = jnp.stack(
                [jnp.stack(metrics) for metrics in update_metrics]
            )
            means = np.asarray(jnp.mean(stacked_metrics, axis=0))
            final_metrics = {
                "loss": float(means[0]),
                "actor_loss": float(means[1]),
                "value_loss": float(means[2]),
                "entropy": float(means[3]),
                "approximate_kl": float(means[4]),
                "mean_recent_return": (
                    float(np.mean(recent_returns)) if recent_returns else float("nan")
                ),
                "completed_episodes": float(len(recent_returns)),
                "environment_steps": float(
                    (update_index + 1) * self.config.steps_per_update
                ),
            }
            if progress is not None:
                progress(update_index + 1, final_metrics)

        return PPOTrainResult(
            params=params,
            runner=runner,
            recent_episode_returns=tuple(recent_returns),
            final_metrics=final_metrics,
        )

    def _evaluate_steps(
        self,
        params: Params,
        key: Array,
        evaluation_steps: int,
    ) -> tuple[Array, Array]:
        runner = self.initialize_runner(key)

        def evaluation_step(carry: RunnerState, unused):
            del unused
            random_key, environment_key = jax.random.split(carry.random_key)
            policy_input = self._policy_input(carry)
            policy_state, logits, _ = self.policy_core.step(
                params,
                carry.policy_state,
                policy_input,
                carry.done,
            )
            logits = self._mask_logits(carry.observation, logits)
            if self.continuous_action:
                action = logits
            else:
                action = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            observation, environment_state, reward, done, info = (
                self.environment.step(
                    jax.random.split(environment_key, self.config.num_envs),
                    carry.environment_state,
                    action,
                    self.environment_params,
                )
            )
            reward = jnp.asarray(reward, dtype=jnp.float32)
            new_carry = RunnerState(
                observation=observation,
                environment_state=environment_state,
                policy_state=policy_state,
                previous_action=action,
                previous_reward=reward,
                done=done,
                random_key=random_key,
            )
            return new_carry, (
                info["returned_episode_returns"],
                info["returned_episode"],
            )

        _, (returns, completed) = jax.lax.scan(
            evaluation_step,
            runner,
            None,
            length=evaluation_steps,
        )
        return returns, completed

    def evaluate(
        self,
        params: Params,
        *,
        key: Array,
        episodes_per_environment: int,
        evaluation_steps: int,
    ) -> dict[str, object]:
        """Evaluate the first fixed number of episodes from every environment."""
        if episodes_per_environment < 1:
            raise ValueError("episodes_per_environment must be positive")
        if evaluation_steps < 1:
            raise ValueError("evaluation_steps must be positive")
        returns, completed = self._evaluate_jit(params, key, evaluation_steps)
        completed_host = np.asarray(completed, dtype=bool)
        returns_host = np.asarray(returns)
        completed_counts = np.sum(completed_host, axis=0)
        if np.any(completed_counts < episodes_per_environment):
            raise RuntimeError(
                "Evaluation scan did not complete the required episodes in "
                "every vector environment: "
                f"required={episodes_per_environment}, "
                f"completed={completed_counts.tolist()}, "
                f"scan_steps={evaluation_steps}"
            )

        selected_returns = np.stack(
            [
                returns_host[
                    np.flatnonzero(completed_host[:, environment_index])[
                        :episodes_per_environment
                    ],
                    environment_index,
                ]
                for environment_index in range(self.config.num_envs)
            ],
            axis=0,
        )
        flat_returns = selected_returns.reshape(-1)
        return {
            "mean_return": float(np.mean(flat_returns)),
            "median_return": float(np.median(flat_returns)),
            "episodes": int(flat_returns.size),
            "episodes_per_environment": episodes_per_environment,
            "num_environments": self.config.num_envs,
            "scan_steps_per_environment": evaluation_steps,
            "returns_by_environment": selected_returns.tolist(),
        }
