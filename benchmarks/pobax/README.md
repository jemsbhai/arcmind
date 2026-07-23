# POBAX integration

This directory isolates the JAX benchmark stack from the PyTorch package. The
benchmark source is pinned to exact POBAX and Navix commits in
`requirements.in`; published runs must use the generated lock as well.

The primary source is the
[Reinforcement Learning Journal paper](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_153.pdf).
The pinned POBAX source is
[commit `a5e1d62d14e4efe783885b9d4f19cffa2a568eec`](https://github.com/taodav/pobax/tree/a5e1d62d14e4efe783885b9d4f19cffa2a568eec).

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

Three required controls are not yet supported by the shared learner:

- positional MLP, which gives the memoryless policy an explicit timestep or
  position signal;
- [Fast and Forgetful Memory](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e3bf2f0f10774c474de22a12cb060e2c-Abstract-Conference.html);
  and
- [Stable Hadamard Memory](https://proceedings.iclr.cc/paper_files/paper/2025/file/b6446566965fa38e183650728ab70318-Paper-Conference.pdf).

The [official FFM source at commit
`b3f94d2a0f35ba05089faf19ab1df846057cf8b6`](https://github.com/proroklab/ffm/tree/b3f94d2a0f35ba05089faf19ab1df846057cf8b6/standalone_jax)
contains a JAX recurrence with explicit recurrent state and episode-done
inputs. It can be adapted as a policy core without changing PPO, after
initialization, reset, sequence, recurrent-state, and gradient parity tests.

The [official SHM `v1.1` source at commit
`40d73d44936e47a29e2c76a481d93c434b857ea1`](https://github.com/thaihungle/SHM/tree/40d73d44936e47a29e2c76a481d93c434b857ea1)
is PyTorch. SHM can use the shared PPO loss, collector, and optimizer, but it
needs a JAX policy-core port. The official repository has stateful PyTorch
benchmark adapters but no JAX streaming adapter. The port must expose initial
memory, handle asynchronous reset masks and explicit random-address keys, and
use a time-major scan. It must match the released equations under a fixed
address sequence and test stochastic addressing distributionally before it is
eligible for registered evaluation.

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
exact-attention reference. The POBAX paper names its attention agent
Transformer-XL, while the pinned repository enables GRU-style transformer
gating by default. Compatibility reports must record the gate and
normalization settings. They must use the name Transformer-XL unless those
settings verify the exact GTrXL architecture.

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

The discrete development runner accepts `simple_chain`, `tmaze_10`,
`rocksample_11_11`, `battleship_10`, and
`Navix-DMLab-Maze-01-v0`. Simple Chain has one action, so it validates
infrastructure rather than policy quality. The same learner supports
`HalfCheetah-V-v0` through a learned state-independent diagonal-Gaussian
action distribution. `Walker-V-v0` remains a required registered task after a
runner adapter is added. `HalfCheetah-P-v0` is exposed by the library and may
remain useful for development, but it is not a registered task from the
published POBAX benchmark.

For the 1,000-step-horizon masked MuJoCo environments, evaluation is
automatically raised to at least 1,024 transitions so every functioning
rollout can complete an episode.

## Evidence tiers and published reference budgets

A smoke run checks imports, shapes, reset behavior, JIT, gradients, and
artifacts. Quick or 131,072-step runs are smoke evidence only. A pilot run may
use shortened budgets and three to five seeds to check learnability and freeze
choices. It is development evidence only. A registered final run uses a frozen
configuration, paired seed manifest, and the full task budget below.

| Environment identifier | Published steps | Tuning seeds | Final seeds |
|---|---:|---:|---:|
| `tmaze_10` | 1,000,000 | 5 | 30 |
| `rocksample_11_11` | 5,000,000 | 5 | 30 |
| `battleship_10` | 10,000,000 | 10 | 30 |
| `Walker-V-v0` | 50,000,000 | 5 | 30 |
| `HalfCheetah-V-v0` | 50,000,000 | 5 | 30 |
| `Navix-DMLab-Maze-01-v0` | 10,000,000 | 5 | 30 |

These counts reproduce the published POBAX interaction budgets and seed
counts. ArcMind uses equal-cardinality tuning grids and paired seeds across
policy cores. A result with 10 final seeds is preliminary. Five final seeds is
a pilot and must not be described as paper-grade or as a reproduction of the
published POBAX result.
