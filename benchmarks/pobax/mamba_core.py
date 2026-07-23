"""Source-audited Mamba-1 policy core for the shared POBAX PPO harness.

The recurrent update is a JAX translation of the official Mamba-1 slow
``step`` path. The policy adaptation adds only an input encoder and the common
actor and critic heads. There is one Mamba block, with the official default
``expand=2``, ``d_state=16``, ``d_conv=4``, and automatic
``dt_rank=ceil(d_model / 16)``.

Audited source:
https://github.com/state-spaces/mamba/blob/10b5d6358f27966f6a40e4bf0baa17a460688128/mamba_ssm/modules/mamba_simple.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array
Params = Mapping[str, Array]

MAMBA_REPOSITORY = "https://github.com/state-spaces/mamba"
MAMBA_VERSION = "2.2.6.post3"
MAMBA_AUDITED_COMMIT = "10b5d6358f27966f6a40e4bf0baa17a460688128"
MAMBA_SIMPLE_SHA256 = "a17e4c51b582dc0d4d690a649eba521cd0c1ee3dc8f0473a0967cdc9ec0874e3"
MAMBA_SOURCE_PATH = "mamba_ssm/modules/mamba_simple.py"
MAMBA_D_STATE = 16
MAMBA_D_CONV = 4
MAMBA_EXPAND = 2


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
    bias_key = f"{prefix}.bias"
    if bias_key in params:
        result = result + params[bias_key]
    return result


def _inverse_softplus(values: Array) -> Array:
    """Stable inverse of softplus, matching the audited PyTorch initializer."""
    return values + jnp.log(-jnp.expm1(-values))


class MambaState(NamedTuple):
    """Streaming convolution and selective SSM caches."""

    convolution: Array
    ssm: Array


@dataclass(frozen=True)
class MambaPolicyCore:
    """One-block, reset-aware Mamba-1 actor-critic policy core."""

    input_dim: int
    action_dim: int
    hidden_size: int
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    dt_init_floor: float = 1e-4

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError("input_dim must be positive")
        if self.action_dim < 1:
            raise ValueError("action_dim must be positive")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if not 0.0 < self.dt_min < self.dt_max:
            raise ValueError("dt bounds must satisfy 0 < dt_min < dt_max")
        if not 0.0 < self.dt_init_floor <= self.dt_max:
            raise ValueError("dt_init_floor must satisfy 0 < floor <= dt_max")

    @property
    def d_inner(self) -> int:
        return MAMBA_EXPAND * self.hidden_size

    @property
    def dt_rank(self) -> int:
        return math.ceil(self.hidden_size / 16)

    def initialize(self, key: Array) -> dict[str, Array]:
        """Initialize the policy and preserve official Mamba-1 distributions."""
        (
            encoder_key,
            in_proj_key,
            conv_weight_key,
            conv_bias_key,
            x_proj_key,
            dt_weight_key,
            dt_bias_key,
            out_proj_key,
            actor_key,
            critic_key,
        ) = jax.random.split(key, 10)

        in_projection_bound = 1.0 / math.sqrt(self.hidden_size)
        inner_projection_bound = 1.0 / math.sqrt(self.d_inner)
        conv_bound = 1.0 / math.sqrt(MAMBA_D_CONV)
        dt_weight_bound = self.dt_rank**-0.5
        dt = jnp.exp(
            jax.random.uniform(
                dt_bias_key,
                (self.d_inner,),
                minval=math.log(self.dt_min),
                maxval=math.log(self.dt_max),
            )
        )
        dt = jnp.maximum(dt, self.dt_init_floor)
        state_indices = jnp.arange(1, MAMBA_D_STATE + 1, dtype=jnp.float32)

        return {
            "encoder.kernel": _xavier(encoder_key, self.input_dim, self.hidden_size),
            "encoder.bias": jnp.zeros((self.hidden_size,)),
            "mamba.in_proj.kernel": jax.random.uniform(
                in_proj_key,
                (self.hidden_size, 2 * self.d_inner),
                minval=-in_projection_bound,
                maxval=in_projection_bound,
            ),
            "mamba.conv1d.kernel": jax.random.uniform(
                conv_weight_key,
                (self.d_inner, MAMBA_D_CONV),
                minval=-conv_bound,
                maxval=conv_bound,
            ),
            "mamba.conv1d.bias": jax.random.uniform(
                conv_bias_key,
                (self.d_inner,),
                minval=-conv_bound,
                maxval=conv_bound,
            ),
            "mamba.x_proj.kernel": jax.random.uniform(
                x_proj_key,
                (self.d_inner, self.dt_rank + 2 * MAMBA_D_STATE),
                minval=-inner_projection_bound,
                maxval=inner_projection_bound,
            ),
            "mamba.dt_proj.kernel": jax.random.uniform(
                dt_weight_key,
                (self.dt_rank, self.d_inner),
                minval=-dt_weight_bound,
                maxval=dt_weight_bound,
            ),
            "mamba.dt_proj.bias": _inverse_softplus(dt),
            "mamba.A_log": jnp.broadcast_to(
                jnp.log(state_indices),
                (self.d_inner, MAMBA_D_STATE),
            ),
            "mamba.D": jnp.ones((self.d_inner,)),
            "mamba.out_proj.kernel": jax.random.uniform(
                out_proj_key,
                (self.d_inner, self.hidden_size),
                minval=-inner_projection_bound,
                maxval=inner_projection_bound,
            ),
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
    ) -> MambaState:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return MambaState(
            convolution=jnp.zeros(
                (batch_size, self.d_inner, MAMBA_D_CONV),
                dtype=dtype,
            ),
            ssm=jnp.zeros(
                (batch_size, self.d_inner, MAMBA_D_STATE),
                dtype=dtype,
            ),
        )

    def block_step(
        self,
        params: Params,
        state: MambaState,
        hidden: Array,
        reset: Array,
    ) -> tuple[MambaState, Array]:
        """Apply the audited Mamba-1 slow-path update to one hidden token."""
        reset = jnp.asarray(reset, dtype=jnp.bool_)
        convolution = jnp.where(
            reset[:, None, None],
            jnp.zeros_like(state.convolution),
            state.convolution,
        )
        ssm = jnp.where(
            reset[:, None, None],
            jnp.zeros_like(state.ssm),
            state.ssm,
        )

        xz = hidden @ params["mamba.in_proj.kernel"]
        x, z = jnp.split(xz, 2, axis=-1)

        new_convolution = jnp.concatenate(
            [convolution[:, :, 1:], x[:, :, None]],
            axis=-1,
        )
        x = jnp.sum(
            new_convolution * params["mamba.conv1d.kernel"][None, :, :],
            axis=-1,
        )
        x = jax.nn.silu(x + params["mamba.conv1d.bias"])

        projected = x @ params["mamba.x_proj.kernel"]
        dt_low_rank, B, C = jnp.split(
            projected,
            (self.dt_rank, self.dt_rank + MAMBA_D_STATE),
            axis=-1,
        )
        dt = dt_low_rank @ params["mamba.dt_proj.kernel"]
        dt = jax.nn.softplus(dt + params["mamba.dt_proj.bias"])
        A = -jnp.exp(params["mamba.A_log"].astype(jnp.float32))

        discrete_A = jnp.exp(dt[:, :, None] * A[None, :, :])
        discrete_B = dt[:, :, None] * B[:, None, :]
        new_ssm = ssm * discrete_A + x[:, :, None] * discrete_B
        y = jnp.einsum("bdn,bn->bd", new_ssm, C)
        y = y + params["mamba.D"] * x
        y = y * jax.nn.silu(z)
        output = y @ params["mamba.out_proj.kernel"]

        return MambaState(new_convolution, new_ssm), output

    def step(
        self,
        params: Params,
        state: MambaState,
        policy_input: Array,
        reset: Array,
    ) -> tuple[MambaState, Array, Array]:
        hidden = _linear(params, "encoder", policy_input)
        new_state, features = self.block_step(params, state, hidden, reset)
        logits = _linear(params, "actor", features)
        values = _linear(params, "critic", features)[..., 0]
        return new_state, logits, values

    def apply_sequence(
        self,
        params: Params,
        state: MambaState,
        policy_inputs: Array,
        resets: Array,
    ) -> tuple[MambaState, Array, Array]:
        def scan_step(carry: MambaState, inputs: tuple[Array, Array]):
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
        return mamba_parameter_count(
            input_dim=self.input_dim,
            action_dim=self.action_dim,
            hidden_size=self.hidden_size,
        )

    def cache_element_count(self, batch_size: int = 1) -> dict[str, int]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        convolution = batch_size * self.d_inner * MAMBA_D_CONV
        ssm = batch_size * self.d_inner * MAMBA_D_STATE
        return {
            "convolution": convolution,
            "ssm": ssm,
            "total": convolution + ssm,
        }


def mamba_parameter_count(
    *,
    input_dim: int,
    action_dim: int,
    hidden_size: int,
) -> int:
    """Return the exact trainable parameter count for the policy core."""
    if input_dim < 1:
        raise ValueError("input_dim must be positive")
    if action_dim < 1:
        raise ValueError("action_dim must be positive")
    if hidden_size < 1:
        raise ValueError("hidden_size must be positive")

    d_inner = MAMBA_EXPAND * hidden_size
    dt_rank = math.ceil(hidden_size / 16)
    encoder = input_dim * hidden_size + hidden_size
    in_projection = hidden_size * 2 * d_inner
    convolution = d_inner * MAMBA_D_CONV + d_inner
    selective_projection = d_inner * (dt_rank + 2 * MAMBA_D_STATE)
    dt_projection = dt_rank * d_inner + d_inner
    dynamics = d_inner * MAMBA_D_STATE + d_inner
    output_projection = d_inner * hidden_size
    heads = hidden_size * action_dim + action_dim + hidden_size + 1
    return (
        encoder
        + in_projection
        + convolution
        + selective_projection
        + dt_projection
        + dynamics
        + output_projection
        + heads
    )


def match_mamba_hidden_size(
    *,
    target_parameters: int,
    input_dim: int,
    action_dim: int,
    maximum_width: int = 1024,
) -> int:
    """Find the positive integer width closest to the ArcMind target."""
    if target_parameters < 1:
        raise ValueError("target_parameters must be positive")
    if input_dim < 1:
        raise ValueError("input_dim must be positive")
    if action_dim < 1:
        raise ValueError("action_dim must be positive")
    if maximum_width < 1:
        raise ValueError("maximum_width must be positive")
    return min(
        range(1, maximum_width + 1),
        key=lambda width: abs(
            mamba_parameter_count(
                input_dim=input_dim,
                action_dim=action_dim,
                hidden_size=width,
            )
            - target_parameters
        ),
    )
