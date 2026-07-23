"""Structured state-space and recurrent-Transformer policy controls.

The implementations are dependency-light JAX translations intended for the
shared POBAX PPO harness. S4D uses the recurrent form of the same zero-order
hold discretization used by its convolutional training form. MS4/MS4N replace
their sequence-classification head with the common actor-critic heads. The
Transformer-XL core uses segment memory and the GTrXL variant replaces both
residual connections with the GRU-type gate from Parisotto et al. (2020).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

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


def _linear(params: Params, prefix: str, values: Array) -> Array:
    return values @ params[f"{prefix}.kernel"] + params[f"{prefix}.bias"]


def _layer_norm(params: Params, prefix: str, values: Array) -> Array:
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
    normalized = (values - mean) * jax.lax.rsqrt(variance + 1e-5)
    return (
        normalized * params[f"{prefix}.weight"]
        + params[f"{prefix}.bias"]
    )


def _glu(values: Array) -> Array:
    content, gate = jnp.split(values, 2, axis=-1)
    return content * jax.nn.sigmoid(gate)


class DiagonalSSMState(NamedTuple):
    """Real representation of the conjugate-symmetric complex S4D state."""

    real: Array
    imag: Array


class LRUState(NamedTuple):
    """Real representation of the complex diagonal LRU recurrence."""

    real: Array
    imag: Array


class S5State(NamedTuple):
    """Real representation of the conjugate-symmetric S5RL state."""

    real: Array
    imag: Array


def _hippo_dplr(state_size: int) -> tuple[Array, Array, Array]:
    """Return the conjugate-symmetric half of the HiPPO-LegS DPLR basis."""
    indices = np.arange(state_size, dtype=np.float64)
    hippo_scale = np.sqrt(1.0 + 2.0 * indices)
    hippo = hippo_scale[:, None] * hippo_scale[None, :]
    hippo = np.tril(hippo) - np.diag(indices)
    hippo = -hippo
    low_rank = np.sqrt(indices + 0.5)
    normal = hippo + low_rank[:, None] * low_rank[None, :]
    eigenvalues_imag, eigenvectors = np.linalg.eigh(normal * -1j)
    modes = state_size // 2
    eigenvalues_real = np.full(
        (modes,),
        np.mean(np.diag(normal)),
        dtype=np.float64,
    )
    return (
        jnp.asarray(eigenvalues_real, dtype=jnp.float32),
        jnp.asarray(eigenvalues_imag[:modes], dtype=jnp.float32),
        jnp.asarray(eigenvectors[:, :modes], dtype=jnp.complex64),
    )


@dataclass(frozen=True)
class S5RLPolicyCore:
    """Reset-aware S5 policy backbone following the NeurIPS 2023 adapter."""

    input_dim: int
    action_dim: int
    hidden_size: int
    state_size: int = 16
    num_layers: int = 2
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    pre_norm: bool = True
    post_ssm_norm: bool = True

    def __post_init__(self) -> None:
        if self.state_size < 2 or self.state_size % 2:
            raise ValueError("state_size must be a positive even integer")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")

    @property
    def complex_state_size(self) -> int:
        return self.state_size // 2

    def initialize(self, key: Array) -> dict[str, Array]:
        keys = iter(jax.random.split(key, 3 + self.num_layers * 6))
        params = {
            "encoder.kernel": _xavier(
                next(keys),
                self.input_dim,
                self.hidden_size,
            ),
            "encoder.bias": jnp.zeros((self.hidden_size,)),
        }
        lambda_real, lambda_imag, eigenvectors = _hippo_dplr(
            self.state_size
        )
        inverse_eigenvectors = jnp.conjugate(eigenvectors).T
        for layer_index in range(self.num_layers):
            prefix = f"layers.{layer_index}"
            input_key = next(keys)
            output_key = next(keys)
            step_key = next(keys)
            feedthrough_key = next(keys)
            output_1_key = next(keys)
            output_2_key = next(keys)
            input_base = jax.random.normal(
                input_key,
                (self.state_size, self.hidden_size),
            ) / jnp.sqrt(self.state_size)
            input_tilde = inverse_eigenvectors @ input_base
            output_real_key, output_imag_key = jax.random.split(output_key)
            output_base = (
                jax.random.normal(
                    output_real_key,
                    (self.hidden_size, self.state_size),
                )
                + 1j
                * jax.random.normal(
                    output_imag_key,
                    (self.hidden_size, self.state_size),
                )
            ) / jnp.sqrt(self.state_size)
            output_tilde = output_base @ eigenvectors
            params[f"{prefix}.Lambda_real"] = lambda_real
            params[f"{prefix}.Lambda_imag"] = lambda_imag
            params[f"{prefix}.B_real"] = jnp.real(input_tilde)
            params[f"{prefix}.B_imag"] = jnp.imag(input_tilde)
            params[f"{prefix}.C_real"] = jnp.real(output_tilde)
            params[f"{prefix}.C_imag"] = jnp.imag(output_tilde)
            params[f"{prefix}.D"] = jax.random.normal(
                feedthrough_key,
                (self.hidden_size,),
            )
            params[f"{prefix}.log_step"] = jax.random.uniform(
                step_key,
                (self.complex_state_size,),
                minval=math.log(self.dt_min),
                maxval=math.log(self.dt_max),
            )
            params[f"{prefix}.output_1.kernel"] = _xavier(
                output_1_key,
                self.hidden_size,
                self.hidden_size,
            )
            params[f"{prefix}.output_1.bias"] = jnp.zeros(
                (self.hidden_size,)
            )
            params[f"{prefix}.output_2.kernel"] = _xavier(
                output_2_key,
                self.hidden_size,
                self.hidden_size,
            )
            params[f"{prefix}.output_2.bias"] = jnp.zeros(
                (self.hidden_size,)
            )
            params[f"{prefix}.norm.weight"] = jnp.ones(
                (self.hidden_size,)
            )
            params[f"{prefix}.norm.bias"] = jnp.zeros(
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
    ) -> S5State:
        shape = (
            batch_size,
            self.num_layers,
            self.complex_state_size,
        )
        zeros = jnp.zeros(shape, dtype=dtype)
        return S5State(real=zeros, imag=zeros)

    @staticmethod
    def _discretize(
        lambda_real: Array,
        lambda_imag: Array,
        log_step: Array,
        B_real: Array,
        B_imag: Array,
    ) -> tuple[Array, Array, Array, Array]:
        step = jnp.exp(log_step)
        magnitude = jnp.exp(step * lambda_real)
        angle = step * lambda_imag
        discrete_real = magnitude * jnp.cos(angle)
        discrete_imag = magnitude * jnp.sin(angle)
        numerator_real = discrete_real - 1.0
        numerator_imag = discrete_imag
        denominator = jnp.square(lambda_real) + jnp.square(lambda_imag)
        scale_real = (
            numerator_real * lambda_real
            + numerator_imag * lambda_imag
        ) / denominator
        scale_imag = (
            numerator_imag * lambda_real
            - numerator_real * lambda_imag
        ) / denominator
        discrete_B_real = (
            scale_real[:, None] * B_real
            - scale_imag[:, None] * B_imag
        )
        discrete_B_imag = (
            scale_real[:, None] * B_imag
            + scale_imag[:, None] * B_real
        )
        return (
            discrete_real,
            discrete_imag,
            discrete_B_real,
            discrete_B_imag,
        )

    def step(
        self,
        params: Params,
        state: S5State,
        policy_input: Array,
        reset: Array,
    ) -> tuple[S5State, Array, Array]:
        state_real = jnp.where(
            reset[:, None, None],
            jnp.zeros_like(state.real),
            state.real,
        )
        state_imag = jnp.where(
            reset[:, None, None],
            jnp.zeros_like(state.imag),
            state.imag,
        )
        features = _linear(params, "encoder", policy_input)
        new_reals = []
        new_imags = []
        for layer_index in range(self.num_layers):
            prefix = f"layers.{layer_index}"
            skip = features
            if self.pre_norm:
                features = _layer_norm(
                    params,
                    f"{prefix}.norm",
                    features,
                )
            (
                lambda_real,
                lambda_imag,
                B_real,
                B_imag,
            ) = self._discretize(
                params[f"{prefix}.Lambda_real"],
                params[f"{prefix}.Lambda_imag"],
                params[f"{prefix}.log_step"],
                params[f"{prefix}.B_real"],
                params[f"{prefix}.B_imag"],
            )
            previous_real = state_real[:, layer_index]
            previous_imag = state_imag[:, layer_index]
            projected_real = features @ B_real.T
            projected_imag = features @ B_imag.T
            new_real = (
                lambda_real * previous_real
                - lambda_imag * previous_imag
                + projected_real
            )
            new_imag = (
                lambda_real * previous_imag
                + lambda_imag * previous_real
                + projected_imag
            )
            ssm_output = 2.0 * (
                new_real @ params[f"{prefix}.C_real"].T
                - new_imag @ params[f"{prefix}.C_imag"].T
            )
            ssm_output = (
                ssm_output + params[f"{prefix}.D"] * features
            )
            if self.post_ssm_norm:
                ssm_output = _layer_norm(
                    params,
                    f"{prefix}.norm",
                    ssm_output,
                )
            activated = jax.nn.gelu(ssm_output)
            transformed = _linear(
                params,
                f"{prefix}.output_1",
                activated,
            ) * jax.nn.sigmoid(
                _linear(params, f"{prefix}.output_2", activated)
            )
            features = skip + transformed
            if not self.pre_norm:
                features = _layer_norm(
                    params,
                    f"{prefix}.norm",
                    features,
                )
            new_reals.append(new_real)
            new_imags.append(new_imag)

        new_state = S5State(
            real=jnp.stack(new_reals, axis=1),
            imag=jnp.stack(new_imags, axis=1),
        )
        logits = _linear(params, "actor", features)
        values = _linear(params, "critic", features)[..., 0]
        return new_state, logits, values

    def apply_sequence(
        self,
        params: Params,
        state: S5State,
        policy_inputs: Array,
        resets: Array,
    ) -> tuple[S5State, Array, Array]:
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
class LRUPolicyCore:
    """Deep Linear Recurrent Unit control for partially observable RL.

    The recurrence and stable ring initialization follow Orvieto et al.
    (ICML 2023). Each recurrent layer is wrapped in a pre-normalized residual
    GLU block, matching the deep sequence-model use rather than treating the
    bare linear recurrence as an entire policy.
    """

    input_dim: int
    action_dim: int
    hidden_size: int
    num_layers: int = 2
    minimum_radius: float = 0.0
    maximum_radius: float = 1.0
    maximum_phase: float = 2.0 * math.pi

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= self.minimum_radius < self.maximum_radius <= 1.0:
            raise ValueError("LRU radii must satisfy 0 <= min < max <= 1")

    def initialize(self, key: Array) -> dict[str, Array]:
        keys = iter(jax.random.split(key, 3 + self.num_layers * 6))
        params = {
            "encoder.kernel": _xavier(
                next(keys),
                self.input_dim,
                self.hidden_size,
            ),
            "encoder.bias": jnp.zeros((self.hidden_size,)),
        }
        for layer_index in range(self.num_layers):
            prefix = f"layers.{layer_index}"
            radius_key = next(keys)
            phase_key = next(keys)
            input_key = next(keys)
            output_key = next(keys)
            mixer_key = next(keys)
            radius_uniform = jnp.clip(
                jax.random.uniform(radius_key, (self.hidden_size,)),
                1e-5,
                1.0 - 1e-5,
            )
            radius_squared = (
                radius_uniform
                * (
                    self.maximum_radius**2
                    - self.minimum_radius**2
                )
                + self.minimum_radius**2
            )
            log_nu = jnp.log(-0.5 * jnp.log(radius_squared))
            phase = jnp.clip(
                self.maximum_phase
                * jax.random.uniform(phase_key, (self.hidden_size,)),
                1e-5,
            )
            log_theta = jnp.log(phase)
            eigenvalue_radius = jnp.exp(-jnp.exp(log_nu))
            params[f"{prefix}.log_nu"] = log_nu
            params[f"{prefix}.log_theta"] = log_theta
            params[f"{prefix}.log_gamma"] = jnp.log(
                jnp.sqrt(1.0 - jnp.square(eigenvalue_radius))
            )
            input_real_key, input_imag_key = jax.random.split(input_key)
            output_real_key, output_imag_key = jax.random.split(output_key)
            input_scale = 1.0 / jnp.sqrt(2.0 * self.hidden_size)
            output_scale = 1.0 / jnp.sqrt(self.hidden_size)
            params[f"{prefix}.B_real"] = (
                jax.random.normal(
                    input_real_key,
                    (self.hidden_size, self.hidden_size),
                )
                * input_scale
            )
            params[f"{prefix}.B_imag"] = (
                jax.random.normal(
                    input_imag_key,
                    (self.hidden_size, self.hidden_size),
                )
                * input_scale
            )
            params[f"{prefix}.C_real"] = (
                jax.random.normal(
                    output_real_key,
                    (self.hidden_size, self.hidden_size),
                )
                * output_scale
            )
            params[f"{prefix}.C_imag"] = (
                jax.random.normal(
                    output_imag_key,
                    (self.hidden_size, self.hidden_size),
                )
                * output_scale
            )
            params[f"{prefix}.D"] = jax.random.normal(
                next(keys),
                (self.hidden_size,),
            )
            params[f"{prefix}.mixer.kernel"] = _xavier(
                mixer_key,
                self.hidden_size,
                self.hidden_size * 2,
            )
            params[f"{prefix}.mixer.bias"] = jnp.zeros(
                (self.hidden_size * 2,)
            )
            params[f"{prefix}.norm.weight"] = jnp.ones(
                (self.hidden_size,)
            )
            params[f"{prefix}.norm.bias"] = jnp.zeros(
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
    ) -> LRUState:
        shape = (batch_size, self.num_layers, self.hidden_size)
        zeros = jnp.zeros(shape, dtype=dtype)
        return LRUState(real=zeros, imag=zeros)

    def step(
        self,
        params: Params,
        state: LRUState,
        policy_input: Array,
        reset: Array,
    ) -> tuple[LRUState, Array, Array]:
        state_real = jnp.where(
            reset[:, None, None],
            jnp.zeros_like(state.real),
            state.real,
        )
        state_imag = jnp.where(
            reset[:, None, None],
            jnp.zeros_like(state.imag),
            state.imag,
        )
        features = _linear(params, "encoder", policy_input)
        new_reals = []
        new_imags = []
        for layer_index in range(self.num_layers):
            prefix = f"layers.{layer_index}"
            recurrent_input = _layer_norm(
                params,
                f"{prefix}.norm",
                features,
            )
            radius = jnp.exp(-jnp.exp(params[f"{prefix}.log_nu"]))
            phase = jnp.exp(params[f"{prefix}.log_theta"])
            eigenvalue_real = radius * jnp.cos(phase)
            eigenvalue_imag = radius * jnp.sin(phase)
            input_normalization = jnp.exp(params[f"{prefix}.log_gamma"])
            projected_real = (
                recurrent_input @ params[f"{prefix}.B_real"].T
            ) * input_normalization
            projected_imag = (
                recurrent_input @ params[f"{prefix}.B_imag"].T
            ) * input_normalization
            previous_real = state_real[:, layer_index]
            previous_imag = state_imag[:, layer_index]
            new_real = (
                eigenvalue_real * previous_real
                - eigenvalue_imag * previous_imag
                + projected_real
            )
            new_imag = (
                eigenvalue_real * previous_imag
                + eigenvalue_imag * previous_real
                + projected_imag
            )
            recurrent_output = (
                new_real @ params[f"{prefix}.C_real"].T
                - new_imag @ params[f"{prefix}.C_imag"].T
                + params[f"{prefix}.D"] * recurrent_input
            )
            mixed = _glu(
                _linear(
                    params,
                    f"{prefix}.mixer",
                    jax.nn.gelu(recurrent_output),
                )
            )
            features = features + mixed
            new_reals.append(new_real)
            new_imags.append(new_imag)

        new_state = LRUState(
            real=jnp.stack(new_reals, axis=1),
            imag=jnp.stack(new_imags, axis=1),
        )
        logits = _linear(params, "actor", features)
        values = _linear(params, "critic", features)[..., 0]
        return new_state, logits, values

    def apply_sequence(
        self,
        params: Params,
        state: LRUState,
        policy_inputs: Array,
        resets: Array,
    ) -> tuple[LRUState, Array, Array]:
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
class DiagonalSSMPolicyCore:
    """Streaming S4D, MS4, or MS4N actor-critic policy core.

    ``variant="s4d"`` follows the minimal official S4D block (GELU, GLU
    channel mixer, residual, LayerNorm). ``ms4`` and ``ms4n`` follow the MS4
    projection -> S4D feedthrough -> GLU channel-mixing path, with LayerNorm
    enabled only for MS4N. Classification pooling is intentionally absent
    because PPO requires one causal decision per input.
    """

    input_dim: int
    action_dim: int
    hidden_size: int
    state_size: int = 16
    num_layers: int = 2
    variant: str = "s4d"
    dt_min: float = 1e-3
    dt_max: float = 1e-1

    def __post_init__(self) -> None:
        if self.variant not in {"s4d", "ms4", "ms4n"}:
            raise ValueError(f"Unsupported diagonal SSM variant: {self.variant}")
        if self.state_size < 2 or self.state_size % 2:
            raise ValueError("state_size must be a positive even integer")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")

    @property
    def complex_state_size(self) -> int:
        return self.state_size // 2

    def initialize(self, key: Array) -> dict[str, Array]:
        keys = iter(jax.random.split(key, 3 + self.num_layers * 2))
        params = {
            "encoder.kernel": _xavier(
                next(keys),
                self.input_dim,
                self.hidden_size,
            ),
            "encoder.bias": jnp.zeros((self.hidden_size,)),
        }
        mode_count = self.complex_state_size
        imaginary_initialization = jnp.pi * jnp.arange(
            mode_count,
            dtype=jnp.float32,
        )
        for layer_index in range(self.num_layers):
            dynamics_key = next(keys)
            mixer_key = next(keys)
            dt_key, c_real_key, c_imag_key = jax.random.split(dynamics_key, 3)
            prefix = f"layers.{layer_index}"
            params[f"{prefix}.log_dt"] = jax.random.uniform(
                dt_key,
                (self.hidden_size,),
                minval=math.log(self.dt_min),
                maxval=math.log(self.dt_max),
            )
            params[f"{prefix}.log_A_real"] = jnp.full(
                (self.hidden_size, mode_count),
                math.log(0.5),
            )
            params[f"{prefix}.A_imag"] = jnp.broadcast_to(
                imaginary_initialization,
                (self.hidden_size, mode_count),
            )
            scale = jnp.sqrt(0.5 / mode_count)
            params[f"{prefix}.C_real"] = (
                jax.random.normal(
                    c_real_key,
                    (self.hidden_size, mode_count),
                )
                * scale
            )
            params[f"{prefix}.C_imag"] = (
                jax.random.normal(
                    c_imag_key,
                    (self.hidden_size, mode_count),
                )
                * scale
            )
            params[f"{prefix}.D"] = jnp.ones((self.hidden_size,))
            params[f"{prefix}.mixer.kernel"] = _xavier(
                mixer_key,
                self.hidden_size,
                self.hidden_size * 2,
            )
            params[f"{prefix}.mixer.bias"] = jnp.zeros(
                (self.hidden_size * 2,)
            )
            if self.variant in {"s4d", "ms4n"}:
                params[f"{prefix}.norm.weight"] = jnp.ones(
                    (self.hidden_size,)
                )
                params[f"{prefix}.norm.bias"] = jnp.zeros(
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
    ) -> DiagonalSSMState:
        shape = (
            batch_size,
            self.num_layers,
            self.hidden_size,
            self.complex_state_size,
        )
        zeros = jnp.zeros(shape, dtype=dtype)
        return DiagonalSSMState(real=zeros, imag=zeros)

    @staticmethod
    def _discretize(
        log_dt: Array,
        log_A_real: Array,
        A_imag: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Return real/imaginary parts of ZOH-discretized A and B."""
        dt = jnp.exp(log_dt)[:, None]
        A_real = -jnp.exp(log_A_real)
        magnitude = jnp.exp(dt * A_real)
        angle = dt * A_imag
        discrete_A_real = magnitude * jnp.cos(angle)
        discrete_A_imag = magnitude * jnp.sin(angle)

        numerator_real = discrete_A_real - 1.0
        numerator_imag = discrete_A_imag
        denominator = jnp.square(A_real) + jnp.square(A_imag)
        discrete_B_real = (
            numerator_real * A_real + numerator_imag * A_imag
        ) / denominator
        discrete_B_imag = (
            numerator_imag * A_real - numerator_real * A_imag
        ) / denominator
        return (
            discrete_A_real,
            discrete_A_imag,
            discrete_B_real,
            discrete_B_imag,
        )

    def step(
        self,
        params: Params,
        state: DiagonalSSMState,
        policy_input: Array,
        reset: Array,
    ) -> tuple[DiagonalSSMState, Array, Array]:
        state_real = jnp.where(
            reset[:, None, None, None],
            jnp.zeros_like(state.real),
            state.real,
        )
        state_imag = jnp.where(
            reset[:, None, None, None],
            jnp.zeros_like(state.imag),
            state.imag,
        )
        features = _linear(params, "encoder", policy_input)
        new_reals = []
        new_imags = []
        for layer_index in range(self.num_layers):
            prefix = f"layers.{layer_index}"
            (
                A_real,
                A_imag,
                B_real,
                B_imag,
            ) = self._discretize(
                params[f"{prefix}.log_dt"],
                params[f"{prefix}.log_A_real"],
                params[f"{prefix}.A_imag"],
            )
            previous_real = state_real[:, layer_index]
            previous_imag = state_imag[:, layer_index]
            input_expanded = features[..., None]
            new_real = (
                A_real * previous_real
                - A_imag * previous_imag
                + B_real * input_expanded
            )
            new_imag = (
                A_real * previous_imag
                + A_imag * previous_real
                + B_imag * input_expanded
            )
            temporal = 2.0 * jnp.sum(
                params[f"{prefix}.C_real"] * new_real
                - params[f"{prefix}.C_imag"] * new_imag,
                axis=-1,
            )
            temporal = temporal + params[f"{prefix}.D"] * features
            if self.variant == "s4d":
                mixed = _glu(
                    _linear(
                        params,
                        f"{prefix}.mixer",
                        jax.nn.gelu(temporal),
                    )
                )
                features = _layer_norm(
                    params,
                    f"{prefix}.norm",
                    features + mixed,
                )
            else:
                features = _glu(
                    _linear(params, f"{prefix}.mixer", temporal)
                )
                if self.variant == "ms4n":
                    features = _layer_norm(
                        params,
                        f"{prefix}.norm",
                        features,
                    )
            new_reals.append(new_real)
            new_imags.append(new_imag)

        new_state = DiagonalSSMState(
            real=jnp.stack(new_reals, axis=1),
            imag=jnp.stack(new_imags, axis=1),
        )
        logits = _linear(params, "actor", features)
        values = _linear(params, "critic", features)[..., 0]
        return new_state, logits, values

    def apply_sequence(
        self,
        params: Params,
        state: DiagonalSSMState,
        policy_inputs: Array,
        resets: Array,
    ) -> tuple[DiagonalSSMState, Array, Array]:
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


class TransformerXLState(NamedTuple):
    """Per-layer Transformer-XL memories and per-environment valid counts."""

    memories: Array
    valid: Array


class FullTransformerState(NamedTuple):
    """Raw causal input window used for exact full-window recomputation."""

    tokens: Array
    valid: Array


def _sinusoidal_encoding(positions: Array, hidden_size: int) -> Array:
    frequencies = 1.0 / (
        10000.0
        ** (
            jnp.arange(0, hidden_size, 2, dtype=positions.dtype)
            / hidden_size
        )
    )
    angles = positions[..., None] * frequencies
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


@dataclass(frozen=True)
class TransformerXLPolicyCore:
    """Causal Transformer-XL or gated Transformer-XL actor-critic core."""

    input_dim: int
    action_dim: int
    hidden_size: int
    num_heads: int = 4
    num_layers: int = 2
    memory_length: int = 32
    gated: bool = False
    gating_bias: float = 2.0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.hidden_size % 2:
            raise ValueError("hidden_size must be even for sinusoidal positions")
        if self.memory_length < 1:
            raise ValueError("memory_length must be positive")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")

    def initialize(self, key: Array) -> dict[str, Array]:
        parameters_per_layer = 7 + (12 if self.gated else 0)
        keys = iter(
            jax.random.split(
                key,
                3 + self.num_layers * parameters_per_layer,
            )
        )
        params = {
            "encoder.kernel": _xavier(
                next(keys),
                self.input_dim,
                self.hidden_size,
            ),
            "encoder.bias": jnp.zeros((self.hidden_size,)),
        }
        for layer_index in range(self.num_layers):
            prefix = f"layers.{layer_index}"
            for projection in ("query", "key", "value"):
                params[f"{prefix}.attention.{projection}.kernel"] = _xavier(
                    next(keys),
                    self.hidden_size,
                    self.hidden_size,
                )
                params[f"{prefix}.attention.{projection}.bias"] = jnp.zeros(
                    (self.hidden_size,)
                )
            params[f"{prefix}.attention.relative.kernel"] = _xavier(
                next(keys),
                self.hidden_size,
                self.hidden_size,
            )
            params[f"{prefix}.attention.content_bias"] = jnp.zeros(
                (self.num_heads, self.hidden_size // self.num_heads)
            )
            params[f"{prefix}.attention.position_bias"] = jnp.zeros(
                (self.num_heads, self.hidden_size // self.num_heads)
            )
            params[f"{prefix}.attention.output.kernel"] = _xavier(
                next(keys),
                self.hidden_size,
                self.hidden_size,
            )
            params[f"{prefix}.attention.output.bias"] = jnp.zeros(
                (self.hidden_size,)
            )
            params[f"{prefix}.attention_norm.weight"] = jnp.ones(
                (self.hidden_size,)
            )
            params[f"{prefix}.attention_norm.bias"] = jnp.zeros(
                (self.hidden_size,)
            )
            params[f"{prefix}.ffn.0.kernel"] = _xavier(
                next(keys),
                self.hidden_size,
                self.hidden_size,
            )
            params[f"{prefix}.ffn.0.bias"] = jnp.zeros((self.hidden_size,))
            params[f"{prefix}.ffn.1.kernel"] = _xavier(
                next(keys),
                self.hidden_size,
                self.hidden_size,
            )
            params[f"{prefix}.ffn.1.bias"] = jnp.zeros((self.hidden_size,))
            params[f"{prefix}.ffn_norm.weight"] = jnp.ones(
                (self.hidden_size,)
            )
            params[f"{prefix}.ffn_norm.bias"] = jnp.zeros(
                (self.hidden_size,)
            )
            if self.gated:
                for gate_name in ("attention_gate", "ffn_gate"):
                    for matrix_name in (
                        "reset_y",
                        "reset_x",
                        "update_y",
                        "update_x",
                        "candidate_y",
                        "candidate_x",
                    ):
                        params[
                            f"{prefix}.{gate_name}.{matrix_name}"
                        ] = _xavier(
                            next(keys),
                            self.hidden_size,
                            self.hidden_size,
                        )
                    params[f"{prefix}.{gate_name}.bias"] = jnp.full(
                        (self.hidden_size,),
                        self.gating_bias,
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
    ) -> TransformerXLState:
        return TransformerXLState(
            memories=jnp.zeros(
                (
                    batch_size,
                    self.num_layers,
                    self.memory_length,
                    self.hidden_size,
                ),
                dtype=dtype,
            ),
            valid=jnp.zeros((batch_size,), dtype=jnp.int32),
        )

    @staticmethod
    def _sinusoidal_positions(
        length: int,
        hidden_size: int,
        dtype: jnp.dtype,
    ) -> Array:
        positions = jnp.arange(length - 1, -1, -1, dtype=dtype)
        return _sinusoidal_encoding(positions, hidden_size)

    def _attention(
        self,
        params: Params,
        prefix: str,
        query_input: Array,
        memory: Array,
        valid: Array,
    ) -> Array:
        normalized_query = _layer_norm(
            params,
            f"{prefix}.attention_norm",
            query_input,
        )
        normalized_memory = _layer_norm(
            params,
            f"{prefix}.attention_norm",
            memory,
        )
        key_values = jnp.concatenate(
            [normalized_memory, normalized_query[:, None, :]],
            axis=1,
        )
        head_size = self.hidden_size // self.num_heads
        query = _linear(
            params,
            f"{prefix}.attention.query",
            normalized_query,
        ).reshape((-1, self.num_heads, head_size))
        key = _linear(
            params,
            f"{prefix}.attention.key",
            key_values,
        ).reshape((-1, self.memory_length + 1, self.num_heads, head_size))
        value = _linear(
            params,
            f"{prefix}.attention.value",
            key_values,
        ).reshape((-1, self.memory_length + 1, self.num_heads, head_size))

        positions = self._sinusoidal_positions(
            self.memory_length + 1,
            self.hidden_size,
            query.dtype,
        )
        relative = (
            positions @ params[f"{prefix}.attention.relative.kernel"]
        ).reshape((self.memory_length + 1, self.num_heads, head_size))
        content_scores = jnp.einsum(
            "bhd,bkhd->bhk",
            query + params[f"{prefix}.attention.content_bias"],
            key,
        )
        position_scores = jnp.einsum(
            "bhd,khd->bhk",
            query + params[f"{prefix}.attention.position_bias"],
            relative,
        )
        scores = (content_scores + position_scores) / jnp.sqrt(
            jnp.asarray(head_size, dtype=query.dtype)
        )
        memory_indices = jnp.arange(self.memory_length)
        memory_mask = memory_indices[None, :] >= (
            self.memory_length - valid[:, None]
        )
        mask = jnp.concatenate(
            [
                memory_mask,
                jnp.ones((valid.shape[0], 1), dtype=jnp.bool_),
            ],
            axis=1,
        )
        scores = jnp.where(mask[:, None, :], scores, -1e30)
        weights = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("bhk,bkhd->bhd", weights, value)
        context = context.reshape((-1, self.hidden_size))
        return _linear(params, f"{prefix}.attention.output", context)

    @staticmethod
    def _gate(
        params: Params,
        prefix: str,
        skip: Array,
        update: Array,
    ) -> Array:
        reset = jax.nn.sigmoid(
            update @ params[f"{prefix}.reset_y"]
            + skip @ params[f"{prefix}.reset_x"]
        )
        interpolation = jax.nn.sigmoid(
            update @ params[f"{prefix}.update_y"]
            + skip @ params[f"{prefix}.update_x"]
            - params[f"{prefix}.bias"]
        )
        candidate = jnp.tanh(
            update @ params[f"{prefix}.candidate_y"]
            + (reset * skip) @ params[f"{prefix}.candidate_x"]
        )
        return (1.0 - interpolation) * skip + interpolation * candidate

    def step(
        self,
        params: Params,
        state: TransformerXLState,
        policy_input: Array,
        reset: Array,
    ) -> tuple[TransformerXLState, Array, Array]:
        memories = jnp.where(
            reset[:, None, None, None],
            jnp.zeros_like(state.memories),
            state.memories,
        )
        valid = jnp.where(reset, 0, state.valid)
        features = _linear(params, "encoder", policy_input)
        next_memories = []
        for layer_index in range(self.num_layers):
            prefix = f"layers.{layer_index}"
            layer_input = features
            attention = self._attention(
                params,
                prefix,
                layer_input,
                memories[:, layer_index],
                valid,
            )
            if self.gated:
                features = self._gate(
                    params,
                    f"{prefix}.attention_gate",
                    layer_input,
                    jax.nn.relu(attention),
                )
            else:
                features = layer_input + attention
            feed_forward_input = _layer_norm(
                params,
                f"{prefix}.ffn_norm",
                features,
            )
            feed_forward = _linear(
                params,
                f"{prefix}.ffn.1",
                jax.nn.gelu(
                    _linear(params, f"{prefix}.ffn.0", feed_forward_input)
                ),
            )
            if self.gated:
                features = self._gate(
                    params,
                    f"{prefix}.ffn_gate",
                    features,
                    jax.nn.relu(feed_forward),
                )
            else:
                features = features + feed_forward
            next_memories.append(
                jnp.concatenate(
                    [
                        memories[:, layer_index, 1:, :],
                        layer_input[:, None, :],
                    ],
                    axis=1,
                )
            )

        new_state = TransformerXLState(
            memories=jnp.stack(next_memories, axis=1),
            valid=jnp.minimum(valid + 1, self.memory_length),
        )
        logits = _linear(params, "actor", features)
        values = _linear(params, "critic", features)[..., 0]
        return new_state, logits, values

    def apply_sequence(
        self,
        params: Params,
        state: TransformerXLState,
        policy_inputs: Array,
        resets: Array,
    ) -> tuple[TransformerXLState, Array, Array]:
        # Transformer-XL treats memory from the preceding segment as constant,
        # while preserving gradients through tokens generated in this segment.
        state = jax.tree.map(jax.lax.stop_gradient, state)

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
class FullCausalTransformerPolicyCore:
    """Full self-attention over a bounded causal input window.

    Unlike Transformer-XL, this control retains raw policy inputs and
    recomputes every layer for the entire visible window on each decision.
    This is deliberately more expensive and serves as the exact-attention
    reference at the same configured context length.
    """

    input_dim: int
    action_dim: int
    hidden_size: int
    num_heads: int = 4
    num_layers: int = 2
    window_length: int = 32

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.hidden_size % 2:
            raise ValueError("hidden_size must be even for sinusoidal positions")
        if self.window_length < 1:
            raise ValueError("window_length must be positive")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")

    def initialize(self, key: Array) -> dict[str, Array]:
        keys = iter(jax.random.split(key, 3 + self.num_layers * 6))
        params = {
            "encoder.kernel": _xavier(
                next(keys),
                self.input_dim,
                self.hidden_size,
            ),
            "encoder.bias": jnp.zeros((self.hidden_size,)),
        }
        for layer_index in range(self.num_layers):
            prefix = f"layers.{layer_index}"
            for projection in ("query", "key", "value"):
                params[f"{prefix}.attention.{projection}.kernel"] = _xavier(
                    next(keys),
                    self.hidden_size,
                    self.hidden_size,
                )
                params[f"{prefix}.attention.{projection}.bias"] = jnp.zeros(
                    (self.hidden_size,)
                )
            params[f"{prefix}.attention.output.kernel"] = _xavier(
                next(keys),
                self.hidden_size,
                self.hidden_size,
            )
            params[f"{prefix}.attention.output.bias"] = jnp.zeros(
                (self.hidden_size,)
            )
            params[f"{prefix}.attention_norm.weight"] = jnp.ones(
                (self.hidden_size,)
            )
            params[f"{prefix}.attention_norm.bias"] = jnp.zeros(
                (self.hidden_size,)
            )
            params[f"{prefix}.ffn.0.kernel"] = _xavier(
                next(keys),
                self.hidden_size,
                self.hidden_size,
            )
            params[f"{prefix}.ffn.0.bias"] = jnp.zeros((self.hidden_size,))
            params[f"{prefix}.ffn.1.kernel"] = _xavier(
                next(keys),
                self.hidden_size,
                self.hidden_size,
            )
            params[f"{prefix}.ffn.1.bias"] = jnp.zeros((self.hidden_size,))
            params[f"{prefix}.ffn_norm.weight"] = jnp.ones(
                (self.hidden_size,)
            )
            params[f"{prefix}.ffn_norm.bias"] = jnp.zeros(
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
    ) -> FullTransformerState:
        return FullTransformerState(
            tokens=jnp.zeros(
                (batch_size, self.window_length, self.input_dim),
                dtype=dtype,
            ),
            valid=jnp.zeros((batch_size,), dtype=jnp.int32),
        )

    def _self_attention(
        self,
        params: Params,
        prefix: str,
        features: Array,
        valid_mask: Array,
    ) -> Array:
        normalized = _layer_norm(
            params,
            f"{prefix}.attention_norm",
            features,
        )
        head_size = self.hidden_size // self.num_heads
        batch_size = features.shape[0]
        projection_shape = (
            batch_size,
            self.window_length,
            self.num_heads,
            head_size,
        )
        query = _linear(
            params,
            f"{prefix}.attention.query",
            normalized,
        ).reshape(projection_shape)
        key = _linear(
            params,
            f"{prefix}.attention.key",
            normalized,
        ).reshape(projection_shape)
        value = _linear(
            params,
            f"{prefix}.attention.value",
            normalized,
        ).reshape(projection_shape)
        scores = jnp.einsum("bqhd,bkhd->bhqk", query, key)
        scores = scores / jnp.sqrt(
            jnp.asarray(head_size, dtype=scores.dtype)
        )
        causal_mask = (
            jnp.arange(self.window_length)[:, None]
            >= jnp.arange(self.window_length)[None, :]
        )
        attention_mask = (
            valid_mask[:, None, None, :]
            & causal_mask[None, None, :, :]
        )
        scores = jnp.where(attention_mask, scores, -1e30)
        weights = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("bhqk,bkhd->bqhd", weights, value)
        context = context.reshape(
            (batch_size, self.window_length, self.hidden_size)
        )
        return _linear(params, f"{prefix}.attention.output", context)

    def step(
        self,
        params: Params,
        state: FullTransformerState,
        policy_input: Array,
        reset: Array,
    ) -> tuple[FullTransformerState, Array, Array]:
        tokens = jnp.where(
            reset[:, None, None],
            jnp.zeros_like(state.tokens),
            state.tokens,
        )
        valid = jnp.where(reset, 0, state.valid)
        tokens = jnp.concatenate(
            [tokens[:, 1:, :], policy_input[:, None, :]],
            axis=1,
        )
        valid = jnp.minimum(valid + 1, self.window_length)
        indices = jnp.arange(self.window_length)
        valid_mask = indices[None, :] >= (
            self.window_length - valid[:, None]
        )
        relative_positions = jnp.maximum(
            indices[None, :] - (self.window_length - valid[:, None]),
            0,
        ).astype(tokens.dtype)
        features = _linear(params, "encoder", tokens)
        features = features + _sinusoidal_encoding(
            relative_positions,
            self.hidden_size,
        )
        features = jnp.where(
            valid_mask[:, :, None],
            features,
            jnp.zeros_like(features),
        )
        for layer_index in range(self.num_layers):
            prefix = f"layers.{layer_index}"
            features = features + self._self_attention(
                params,
                prefix,
                features,
                valid_mask,
            )
            normalized = _layer_norm(
                params,
                f"{prefix}.ffn_norm",
                features,
            )
            feed_forward = _linear(
                params,
                f"{prefix}.ffn.1",
                jax.nn.gelu(
                    _linear(params, f"{prefix}.ffn.0", normalized)
                ),
            )
            features = features + feed_forward
            features = jnp.where(
                valid_mask[:, :, None],
                features,
                jnp.zeros_like(features),
            )
        current = features[:, -1, :]
        logits = _linear(params, "actor", current)
        values = _linear(params, "critic", current)[..., 0]
        return FullTransformerState(tokens, valid), logits, values

    def apply_sequence(
        self,
        params: Params,
        state: FullTransformerState,
        policy_inputs: Array,
        resets: Array,
    ) -> tuple[FullTransformerState, Array, Array]:
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


def match_sequence_width(
    model: str,
    *,
    target_parameters: int,
    input_dim: int,
    action_dim: int,
    state_size: int = 16,
    num_layers: int = 2,
    num_heads: int = 4,
    maximum_width: int = 1024,
) -> int:
    """Find the closest legal width for an SSM or Transformer control."""

    if model == "s5rl":
        if state_size < 2 or state_size % 2:
            raise ValueError("state_size must be a positive even integer")
        complex_modes = state_size // 2

        def count(width: int) -> int:
            encoder = input_dim * width + width
            layer = (
                4 * width * complex_modes
                + 3 * complex_modes
                + 2 * width * width
                + 5 * width
            )
            heads = width * action_dim + action_dim + width + 1
            return encoder + num_layers * layer + heads

        candidates = range(1, maximum_width + 1)
    elif model == "lru":

        def count(width: int) -> int:
            encoder = input_dim * width + width
            layer = 6 * width * width + 8 * width
            heads = width * action_dim + action_dim + width + 1
            return encoder + num_layers * layer + heads

        candidates = range(1, maximum_width + 1)
    elif model in {"s4d", "ms4", "ms4n"}:
        if state_size < 2 or state_size % 2:
            raise ValueError("state_size must be a positive even integer")
        complex_modes = state_size // 2
        has_norm = model in {"s4d", "ms4n"}

        def count(width: int) -> int:
            encoder = input_dim * width + width
            dynamics = 4 * width * complex_modes + 2 * width
            mixer = 2 * width * width + 2 * width
            normalization = 2 * width if has_norm else 0
            heads = width * action_dim + action_dim + width + 1
            return (
                encoder
                + num_layers * (dynamics + mixer + normalization)
                + heads
            )

        candidates = range(1, maximum_width + 1)
    elif model in {"causal_transformer", "transformer_xl", "gtrxl"}:
        gated = model == "gtrxl"

        def count(width: int) -> int:
            encoder = input_dim * width + width
            if model == "causal_transformer":
                layer = 6 * width * width + 10 * width
            else:
                layer = 7 * width * width + 12 * width
            if gated:
                layer += 12 * width * width + 2 * width
            heads = width * action_dim + action_dim + width + 1
            return encoder + num_layers * layer + heads

        step = math.lcm(num_heads, 2)
        candidates = range(step, maximum_width + 1, step)
    else:
        raise ValueError(f"Unsupported sequence model: {model}")

    return min(
        candidates,
        key=lambda width: abs(count(width) - target_parameters),
    )
