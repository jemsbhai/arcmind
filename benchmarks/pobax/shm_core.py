"""Source-audited Stable Hadamard Memory policy core.

This module translates the SHM POPGym cell released in ``thaihungle/SHM`` at
commit ``40d73d44936e47a29e2c76a481d93c434b857ea1``. The translated memory
equations are:

```
x_hat = LayerNorm(x)
K = normalize_sum(relu(W_K x_hat))
Q = normalize_sum(relu(W_Q x_hat))
V = W_V x_hat
eta = sigmoid(W_eta x_hat)
C = 1 + tanh(theta[address] outer (W_C x_hat))
M_t = M_(t-1) hadamard C + (eta V) outer K
y = M_t Q + W_shortcut x_hat + b_shortcut
features = W_out y + b_out
```

The actor and critic heads are an intentional shared-learner adaptation. They
are not part of SHM itself. Likewise, this port consumes the harness's common
augmented policy input rather than the positional preprocessing in the
official RLlib agent.

The released sources disagree about address sampling. The paper and POMDP
adapter sample uniformly from all 128 rows. The standalone and POPGym v1.1
paths use ``uniform_(0, 1).long()``, which always selects row zero. Both
behaviors are named here, with the paper mechanism as the scientific default.

Addresses are explicit inputs to ``step`` and ``apply_sequence``. Convenience
methods can sample them from a key and return the sampled values, but callers
must store and replay those values when recomputing a stochastic-policy loss.
The shared PPO stores the collection trace and replays it exactly during every
loss recomputation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Literal, Mapping, Sequence

import jax
import jax.numpy as jnp

Array = jax.Array
Params = Mapping[str, Array]
AddressMode = Literal["paper_uniform", "v1_1_popgym_compat"]

SHM_SOURCE_COMMIT = "40d73d44936e47a29e2c76a481d93c434b857ea1"
SHM_ADDRESS_BITS = 7
SHM_ADDRESS_ROWS = 2**SHM_ADDRESS_BITS
POPGYM_SHM_MEMORY_SIZE = 16

_LAYER_NORM_EPSILON = 1e-5
_SUM_NORMALIZATION_EPSILON = 1e-5
_VALID_ADDRESS_MODES = frozenset(("paper_uniform", "v1_1_popgym_compat"))


def _pytorch_linear_kernel(
    key: Array,
    input_features: int,
    output_features: int,
) -> Array:
    """Match the distribution of ``torch.nn.Linear``'s default kernel."""
    bound = 1.0 / math.sqrt(input_features)
    return jax.random.uniform(
        key,
        (input_features, output_features),
        minval=-bound,
        maxval=bound,
    )


def _pytorch_linear_bias(key: Array, input_features: int, output_features: int) -> Array:
    """Match the distribution of ``torch.nn.Linear``'s default bias."""
    bound = 1.0 / math.sqrt(input_features)
    return jax.random.uniform(
        key,
        (output_features,),
        minval=-bound,
        maxval=bound,
    )


def _xavier(key: Array, input_features: int, output_features: int) -> Array:
    bound = math.sqrt(6.0 / (input_features + output_features))
    return jax.random.uniform(
        key,
        (input_features, output_features),
        minval=-bound,
        maxval=bound,
    )


def _linear_without_bias(params: Params, prefix: str, values: Array) -> Array:
    return values @ params[f"{prefix}.kernel"]


def _linear(params: Params, prefix: str, values: Array) -> Array:
    return values @ params[f"{prefix}.kernel"] + params[f"{prefix}.bias"]


def _layer_norm(params: Params, values: Array) -> Array:
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
    normalized = (values - mean) * jax.lax.rsqrt(variance + _LAYER_NORM_EPSILON)
    return normalized * params["shm.norm.scale"] + params["shm.norm.bias"]


def _sum_normalize(values: Array) -> Array:
    return values / (_SUM_NORMALIZATION_EPSILON + jnp.sum(values, axis=-1, keepdims=True))


@dataclass(frozen=True)
class SHMPolicyCore:
    """Reset-aware SHM cell with shared actor and critic heads.

    ``step`` and ``apply_sequence`` deliberately require explicit addresses.
    Use ``step_with_key`` or ``apply_sequence_with_key`` only at collection
    time, then replay the returned addresses through the explicit methods.
    """

    input_dim: int
    action_dim: int
    hidden_size: int
    memory_size: int = POPGYM_SHM_MEMORY_SIZE
    address_mode: AddressMode = "paper_uniform"
    requires_policy_aux_replay: ClassVar[bool] = True

    def __post_init__(self) -> None:
        for name in ("input_dim", "action_dim", "hidden_size", "memory_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.address_mode not in _VALID_ADDRESS_MODES:
            raise ValueError("address_mode must be 'paper_uniform' or 'v1_1_popgym_compat'")

    def initialize(self, key: Array) -> dict[str, Array]:
        """Initialize the pinned POPGym cell and shared policy heads."""
        keys = iter(jax.random.split(key, 14))
        params = {
            "shm.norm.scale": jnp.ones((self.input_dim,)),
            "shm.norm.bias": jnp.zeros((self.input_dim,)),
            "shm.key.kernel": _pytorch_linear_kernel(next(keys), self.input_dim, self.memory_size),
            "shm.query.kernel": _pytorch_linear_kernel(
                next(keys), self.input_dim, self.memory_size
            ),
            "shm.value.kernel": _pytorch_linear_kernel(
                next(keys), self.input_dim, self.memory_size
            ),
            "shm.calibration.kernel": _pytorch_linear_kernel(
                next(keys), self.input_dim, self.memory_size
            ),
            "shm.eta.kernel": _pytorch_linear_kernel(next(keys), self.input_dim, 1),
            # The only explicitly initialized upstream tensor is theta_matrix.
            "shm.theta": _xavier(next(keys), SHM_ADDRESS_ROWS, self.memory_size),
            "shm.shortcut.kernel": _pytorch_linear_kernel(
                next(keys), self.input_dim, self.memory_size
            ),
            "shm.shortcut.bias": _pytorch_linear_bias(next(keys), self.input_dim, self.memory_size),
            "shm.out.kernel": _pytorch_linear_kernel(
                next(keys), self.memory_size, self.hidden_size
            ),
            "shm.out.bias": _pytorch_linear_bias(next(keys), self.memory_size, self.hidden_size),
            # Intentional shared-head adaptation. SHM ends at `features`.
            "actor.kernel": 0.01 * _xavier(next(keys), self.hidden_size, self.action_dim),
            "actor.bias": jnp.zeros((self.action_dim,)),
            "critic.kernel": _xavier(next(keys), self.hidden_size, 1),
            "critic.bias": jnp.zeros((1,)),
        }
        return params

    def initial_state(
        self,
        batch_size: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> Array:
        """Return one ``H x H`` memory matrix per vector environment."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return jnp.zeros(
            (batch_size, self.memory_size, self.memory_size),
            dtype=dtype,
        )

    def sample_addresses(
        self,
        key: Array,
        shape: Sequence[int],
    ) -> Array:
        """Sample addresses for collection according to the named mode."""
        shape_tuple = tuple(int(dimension) for dimension in shape)
        if any(dimension < 0 for dimension in shape_tuple):
            raise ValueError("address shape dimensions must be nonnegative")
        if self.address_mode == "v1_1_popgym_compat":
            return jnp.zeros(shape_tuple, dtype=jnp.int32)
        return jax.random.randint(
            key,
            shape_tuple,
            minval=0,
            maxval=SHM_ADDRESS_ROWS,
            dtype=jnp.int32,
        )

    @staticmethod
    def _check_step_shapes(
        state: Array,
        policy_input: Array,
        reset: Array,
        addresses: Array,
    ) -> None:
        batch_shape = policy_input.shape[:-1]
        if reset.shape != batch_shape:
            raise ValueError(f"reset shape {reset.shape} does not match batch shape {batch_shape}")
        if addresses.shape != batch_shape:
            raise ValueError(
                f"address shape {addresses.shape} does not match batch shape {batch_shape}"
            )
        if state.shape[0] != policy_input.shape[0]:
            raise ValueError("state and policy_input batch dimensions must match")

    def _memory_step(
        self,
        params: Params,
        state: Array,
        policy_input: Array,
        reset: Array,
        addresses: Array,
    ) -> tuple[Array, Array]:
        """Apply the pinned POPGym SHM equations at fixed addresses."""
        normalized_input = _layer_norm(params, policy_input)
        key = jax.nn.relu(_linear_without_bias(params, "shm.key", normalized_input))
        query = jax.nn.relu(_linear_without_bias(params, "shm.query", normalized_input))
        key = _sum_normalize(key)
        query = _sum_normalize(query)

        value = _linear_without_bias(params, "shm.value", normalized_input)
        eta = jax.nn.sigmoid(_linear_without_bias(params, "shm.eta", normalized_input))
        calibration_value = _linear_without_bias(params, "shm.calibration", normalized_input)
        theta = params["shm.theta"][addresses]
        calibration = 1.0 + jnp.tanh(theta[..., :, None] * calibration_value[..., None, :])

        retained_state = jnp.where(
            reset[..., None, None],
            jnp.zeros_like(state),
            state,
        )
        write = (eta * value)[..., :, None] * key[..., None, :]
        new_state = retained_state * calibration + write
        memory_read = jnp.einsum(
            "...ij,...j->...i",
            new_state,
            query,
        )
        shortcut = _linear(params, "shm.shortcut", normalized_input)
        features = _linear(params, "shm.out", memory_read + shortcut)
        return new_state, features

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
        addresses: Array,
    ) -> tuple[Array, Array, Array]:
        """Apply one update using collection-time addresses."""
        reset = jnp.asarray(reset, dtype=jnp.bool_)
        addresses = jnp.asarray(addresses, dtype=jnp.int32)
        self._check_step_shapes(state, policy_input, reset, addresses)
        new_state, features = self._memory_step(
            params,
            state,
            policy_input,
            reset,
            addresses,
        )
        logits, values = self._heads(params, features)
        return new_state, logits, values

    def step_with_key(
        self,
        params: Params,
        state: Array,
        policy_input: Array,
        reset: Array,
        address_key: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Sample one address per environment and return it for replay."""
        addresses = self.sample_addresses(address_key, reset.shape)
        new_state, logits, values = self.step(
            params,
            state,
            policy_input,
            reset,
            addresses,
        )
        return new_state, logits, values, addresses

    def apply_sequence(
        self,
        params: Params,
        state: Array,
        policy_inputs: Array,
        resets: Array,
        addresses: Array,
    ) -> tuple[Array, Array, Array]:
        """Scan a time-major sequence using replayed addresses."""
        resets = jnp.asarray(resets, dtype=jnp.bool_)
        addresses = jnp.asarray(addresses, dtype=jnp.int32)
        if policy_inputs.ndim < 3:
            raise ValueError("policy_inputs must have time, batch, and feature axes")
        if resets.shape != policy_inputs.shape[:-1]:
            raise ValueError("resets must match policy_inputs' time and batch axes")
        if addresses.shape != resets.shape:
            raise ValueError("addresses must match resets")

        def scan_step(
            carry: Array,
            inputs: tuple[Array, Array, Array],
        ) -> tuple[Array, Array]:
            policy_input, reset, address = inputs
            new_carry, features = self._memory_step(
                params,
                carry,
                policy_input,
                reset,
                address,
            )
            return new_carry, features

        new_state, features = jax.lax.scan(
            scan_step,
            state,
            (policy_inputs, resets, addresses),
        )
        logits, values = self._heads(params, features)
        return new_state, logits, values

    def apply_sequence_with_key(
        self,
        params: Params,
        state: Array,
        policy_inputs: Array,
        resets: Array,
        address_key: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Sample a complete address trace and return it for replay."""
        addresses = self.sample_addresses(address_key, resets.shape)
        new_state, logits, values = self.apply_sequence(
            params,
            state,
            policy_inputs,
            resets,
            addresses,
        )
        return new_state, logits, values, addresses

    @staticmethod
    def count_parameters(params: Params) -> int:
        """Count SHM and shared-head scalar trainable parameters."""
        return sum(int(value.size) for value in jax.tree.leaves(params))


def match_shm_hidden_size(
    *,
    target_parameters: int,
    input_dim: int,
    action_dim: int,
    memory_size: int = POPGYM_SHM_MEMORY_SIZE,
    maximum_width: int = 4096,
) -> int:
    """Find the closest output width while preserving the SHM memory.

    The official POPGym setting fixes a 16 by 16 matrix state and 128 address
    rows. Parameter matching therefore searches only the downstream feature
    width, leaving the SHM recurrence and address table unchanged.
    """
    arguments = {
        "target_parameters": target_parameters,
        "input_dim": input_dim,
        "action_dim": action_dim,
        "memory_size": memory_size,
        "maximum_width": maximum_width,
    }
    for name, value in arguments.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    fixed_parameters = (
        3 * input_dim
        + 5 * input_dim * memory_size
        + SHM_ADDRESS_ROWS * memory_size
        + memory_size
        + action_dim
        + 1
    )
    parameters_per_hidden_unit = memory_size + action_dim + 2

    def parameter_count(hidden_size: int) -> int:
        return fixed_parameters + hidden_size * parameters_per_hidden_unit

    return min(
        range(1, maximum_width + 1),
        key=lambda width: abs(parameter_count(width) - target_parameters),
    )
