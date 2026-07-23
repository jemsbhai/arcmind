# POBAX integration

This directory isolates the JAX benchmark stack from the PyTorch package. The
benchmark source is pinned to exact POBAX and Navix commits in
`requirements.in`; published runs must use the generated lock as well.

The PyPI `pobax==0.0.1` wheel is not used. It omits the source layout expected
by its environment imports and fails when `pobax.envs` is imported. The pinned
Git revision packages `pobax.definitions` correctly.

## WSL2 environment

```bash
python -m venv ~/.venvs/arcmind-pobax
~/.venvs/arcmind-pobax/bin/python -m pip install --upgrade pip
~/.venvs/arcmind-pobax/bin/python -m pip install \
  -r benchmarks/pobax/requirements-lock.txt
```

Validate the accelerator, source revision, and pilot environments:

```bash
python -m benchmarks.pobax.smoke_environment --require-gpu
```

## Cross-framework inference parity

Export a deterministic fixture with the ordinary PyTorch development
environment:

```bash
python -m benchmarks.pobax.export_parity_fixture
```

Then check it in WSL2 with JAX:

```bash
python -m benchmarks.pobax.check_parity \
  benchmark_results/parity/pytorch_fixture.npz
```

The fixture executes 14 sensor steps with decision stride 2, two SSM and two
attention layers, bounded temporal encoding, and four memory slots. Seven
memory writes force ring-buffer wraparound. Passing this check is a release
gate for the JAX learner, not evidence of benchmark quality by itself.

The trainable policy-core adapter uses the same functional implementation.
Exercise its time-major scan, asynchronous resets, JIT compilation, and
reverse-mode gradients with:

```bash
python -m benchmarks.pobax.smoke_policy_core
```

Run the JAX-specific regression suite with:

```bash
python -m pytest -q benchmarks/pobax/tests
```

## Learner contract

The RL comparison will use one parameterized JAX PPO implementation for
ArcMind and every baseline. Environment collection, reset semantics,
advantage estimation, minibatching, optimizer, fixed interaction budget,
final-policy selection, and evaluation are shared. Return-based checkpoint
selection is prohibited. Only the recurrent policy core may vary.

The development learner currently supports these parameter-matched policy
cores through that common path:

- memoryless MLP, four-frame MLP, and fixed exponential memory traces;
- Elman RNN, GRU, and LSTM;
- a three-layer causal dilated TCN;
- LRU, S5RL, recurrent S4D, MS4, and MS4N;
- full-window causal Transformer, Transformer-XL, and GTrXL; and
- ArcMind.

S4D uses the recurrent zero-order-hold form of the diagonal SSM described by
the [S4D paper](https://arxiv.org/abs/2206.11893) and initialized consistently
with the [official minimal implementation](https://github.com/state-spaces/s4).
The MS4/MS4N policy cores translate the projection, direct S4D feedthrough,
GLU channel mixing, and optional LayerNorm equations from the
[2026 preprint](https://arxiv.org/abs/2605.27406). The original model is a
sequence classifier and no official implementation was located, so these are
explicit causal actor-critic adaptations rather than claims of author-code
reproduction.

The Transformer-XL control uses bounded per-layer segment memory and relative
sinusoidal attention. GTrXL uses identity-map pre-normalization and replaces
both residual connections with the GRU-type gates from the
[GTrXL paper](https://proceedings.mlr.press/v119/parisotto20a.html). The
full-window control instead retains raw policy inputs and recomputes the whole
causal window at every decision, making it a deliberately expensive
exact-attention reference.

The memory-trace control uses the two decay rates in the official
[ICML 2025 example](https://github.com/onnoeberhard/memory-traces), and the LRU
uses the stable ring initialization and normalized input projection from
[Orvieto et al.](https://proceedings.mlr.press/v202/orvieto23a.html).
S5RL preserves the authors' HiPPO-LegS initialization, zero-order-hold
discretization, reset-aware recurrence, and full-GLU residual block while
using the shared input and actor-critic heads.

```bash
python -m benchmarks.pobax.run_pilot \
  --environment tmaze_10 \
  --model arcmind \
  --quick \
  --require-gpu
```

`--quick` results are explicitly marked `development_pilot_not_for_paper`.
They validate plumbing and learnability; they are not registered evidence.
Each JSON artifact records the complete policy-core configuration, parameter
match, PPO configuration, accelerator, and exact POBAX commit.

The discrete development runner accepts Simple Chain, T-Maze-10, RockSample
11x11, Battleship-10, and Navix DMLab Maze-01. The same learner also supports
position-only and velocity-only HalfCheetah through a learned
state-independent diagonal-Gaussian action distribution. For these two
1,000-step-horizon environments, evaluation is automatically raised to at
least 1,024 transitions so every functioning rollout can complete an episode.
