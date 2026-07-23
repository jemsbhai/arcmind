"""JIT and differentiate the ArcMind policy adapter on the JAX GPU."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from benchmarks.pobax.arcmind_reference import ReferenceConfig
from benchmarks.pobax.policy_core import ArcMindPolicyCore


def run_smoke() -> dict[str, object]:
    """Exercise reset semantics, recurrent scans, JIT, and reverse-mode gradients."""
    config = ReferenceConfig(
        num_sensor_channels=12,
        d_model=16,
        num_ssm_layers=2,
        ssm_state_dim=4,
        ssm_conv_width=3,
        ssm_expand_factor=1,
        num_attn_layers=1,
        num_attn_heads=2,
        attn_window_size=4,
        num_memory_slots=6,
        memory_compress_ratio=2,
        action_dim=4,
        decision_stride=1,
    )
    core = ArcMindPolicyCore(config)
    params = core.initialize(jax.random.PRNGKey(91))
    initial_state = core.initial_state(batch_size=8)
    policy_inputs = jax.random.normal(
        jax.random.PRNGKey(92),
        (16, 8, config.num_sensor_channels),
    )
    resets = jnp.zeros((16, 8), dtype=jnp.bool_)
    resets = resets.at[0, :].set(True)
    resets = resets.at[7, 1::2].set(True)

    @jax.jit
    def loss_and_outputs(current_params):
        _, logits, values = core.apply_sequence(
            current_params,
            initial_state,
            policy_inputs,
            resets,
        )
        loss = jnp.mean(jnp.square(logits)) + jnp.mean(jnp.square(values))
        return loss, (logits, values)

    (loss, (logits, values)), gradients = jax.value_and_grad(
        loss_and_outputs,
        has_aux=True,
    )(params)
    jax.block_until_ready((loss, logits, values, gradients))
    gradient_leaves = jax.tree.leaves(gradients)
    gradients_finite = all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in gradient_leaves)
    gradient_norm = float(
        jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in gradient_leaves))
    )
    if not gradients_finite or gradient_norm == 0.0:
        raise RuntimeError(
            f"Invalid gradient smoke result: finite={gradients_finite}, norm={gradient_norm}"
        )

    return {
        "backend": jax.default_backend(),
        "parameters": core.count_parameters(params),
        "loss": float(loss),
        "gradient_norm": gradient_norm,
        "logits_shape": list(logits.shape),
        "values_shape": list(values.shape),
        "resets_exercised": int(resets.sum()),
    }


def main() -> None:
    print(json.dumps(run_smoke(), indent=2))


if __name__ == "__main__":
    main()
