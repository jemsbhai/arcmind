# ArcMind benchmark and baseline audit

Status: living primary-source audit, updated 2026-07-23.

This document separates like-for-like backbone controls from methods that
change the learner, supervision, observation privileges, or task. It is not a
claim that ArcMind exceeds any listed method.

## Benchmark selection

### Primary: POBAX

[POBAX](https://openreview.net/forum?id=HUTCbYOW5E) is the primary RL suite.
Its selection criterion is directly aligned with the ArcMind hypothesis:
environments should span distinct forms of partial observability and exhibit a
measurable gap between memoryless and better-informed agents. The source is
pinned to commit `a5e1d62d14e4efe783885b9d4f19cffa2a568eec`.

The initial discrete set covers retention, localization, active information
gathering, and action masking: T-Maze-10, RockSample 11x11, Battleship-10, and
Navix DMLab Maze-01. Position-only and velocity-only HalfCheetah exercise the
same shared PPO learner with diagonal-Gaussian actions. Simple Chain remains
an infrastructure check because its single action cannot discriminate policy
quality.

### Secondary and diagnostic suites

- [POPGym](https://arxiv.org/abs/2303.01859) supplies controlled memory
  primitives and scalable memory-length sweeps.
- [Memory Maze](https://arxiv.org/abs/2210.13383) is a stretch embodied-memory
  test after the low-dimensional policy-backbone claim is established.
- The 2026 [memory-rewriting benchmark](https://arxiv.org/abs/2601.15086)
  tests whether a memory can replace stale information rather than merely
  retain it. It is a useful stress test, but remains a preprint and should not
  displace POBAX as the primary suite.
- State-based [RoboMimic](https://proceedings.mlr.press/v164/mandlekar22a.html)
  evaluates the architecture as a behavior-cloning backbone without adding a
  vision-encoder confound.
- [MONSTER](https://arxiv.org/abs/2502.15122) and the official
  [UEA archive](https://timeseriesclassification.com/) evaluate multivariate
  sensor classification as a separate diagnostic track.

## Like-for-like policy-backbone controls

Every implemented control below receives the same causal input (current
observation, previous action, previous reward, and reset flag), actor-critic
heads, collection path, PPO loss, optimizer, update schedule, environment
seeds, and evaluation path. Width is chosen by exact scalar parameter count.

| Family | Required control | Implementation status | Reason |
|---|---|---|---|
| no learned memory | MLP, four-frame MLP | implemented | Establishes observation and short-window floors. |
| conventional recurrence | Elman RNN, GRU, LSTM | implemented | Strong recurrent RL is mandatory: careful recurrent model-free RL matched or exceeded specialized methods on 18/21 environments in the [ICML 2022 study](https://proceedings.mlr.press/v162/ni22a.html). |
| finite convolution | causal dilated TCN | implemented | Separates finite receptive field from indefinite recurrence. |
| fixed fading memory | memory traces | implemented | The [ICML 2025 method](https://proceedings.mlr.press/v267/eberhard25a.html) uses compact exponential moving averages and reports improved sample efficiency in some partially observable tasks. The official T-Maze example uses decays 0 and 0.985 and a 20M-step budget; ArcMind must not compare against an undertrained trace model. |
| stable linear recurrence | LRU | implemented | [ICML 2024](https://proceedings.mlr.press/v235/lu24h.html) identifies the Deep Linear Recurrent Unit as a strong alternative to Transformers in POMDPs; its stable parameterization comes from the [LRU paper](https://proceedings.mlr.press/v202/orvieto23a.html). |
| structured SSM | S4D, MS4, MS4N | implemented | Tests stable input-invariant diagonal dynamics. S4D follows the [paper](https://arxiv.org/abs/2206.11893) and [official implementation](https://github.com/state-spaces/s4); MS4/MS4N are causal policy adaptations of the [2026 preprint](https://arxiv.org/abs/2605.27406). |
| RL-specific structured SSM | S5RL | implemented | S5 is a multi-input/multi-output SSM with an [official JAX implementation](https://github.com/lindermanlab/S5); [S5RL](https://proceedings.neurips.cc/paper_files/paper/2023/hash/92d3d2a9801211ca3693ccb2faa1316f-Abstract-Conference.html) specifically adapts it to reset-aware RL. |
| exact attention | full-window causal Transformer | implemented | Provides the high-cost exact-context reference. |
| recurrent attention | Transformer-XL, GTrXL | implemented | Tests segment recurrence and the GRU-gated, identity-map architecture introduced by [GTrXL](https://proceedings.mlr.press/v119/parisotto20a.html). |
| proposed hybrid | ArcMind plus five registered ablations | implemented | Isolates the fast path, exact memory, temporal ordering, fusion gate, and full hybrid. |

The implemented memory-trace adapter applies the fixed exponential averages to
the common causal policy input rather than the original paper's
environment-specific one-hot observation encoder. The implemented MS4/MS4N
replaces sequence-classification pooling with causal actor-critic heads. The
S5RL core preserves the authors' HiPPO-LegS DPLR initialization, ZOH
discretization, conjugate-symmetric recurrence, reset behavior, and full-GLU
residual block, but uses the common direct input projection and actor-critic
heads. These differences must be stated in the paper and checked against
author-code results where possible.

## Algorithm-level and contextual comparisons

These methods cannot be placed in the main parameter-matched backbone table
without changing the experimental question:

- POBAX's lambda-discrepancy agent and native tuned PPO results change the
  critic/learning algorithm; report them in a separate algorithm-level table.
- [Memory Traces](https://proceedings.mlr.press/v267/eberhard25a.html) can be
  used as a fixed policy representation in the shared learner, while its
  author-code result remains a separate compatibility reproduction.
- [Recall to Imagine](https://arxiv.org/abs/2403.04253) adds a world model and
  imagined rollouts.
- [Memo](https://proceedings.neurips.cc/paper_files/paper/2025/hash/96889893231d651898b0de42fdbee3a6-Abstract-Conference.html)
  targets learned memory management for embodied agents at a different system
  scale.
- Privileged-state critics, expert distillation, and asymmetric methods use
  information unavailable to the deployed policy. They are upper references
  or training-regime comparisons, not pure memory-backbone baselines.

The 2026 preprint
[Why Linear Recurrent Memory Works in Partially Observable RL](https://arxiv.org/abs/2605.31261)
provides additional motivation for testing stable linear filters, but it is not
peer-reviewed evidence and does not replace the LRU, S5RL, GRU, or LSTM
comparisons.

## Internal research-folder review

The four internal reports on subquadratic attention are language-model
analyses, not sensor-control studies. They are retained as hypothesis-generating
material only. Three ideas transfer plausibly to ArcMind and therefore receive
direct controls: fixed-state recurrence trades exact retrieval for compression,
a small exact path may repair recall failures, and regular bounded access
patterns are more likely to yield practical speedups than irregular dynamic
sparsity.

The reports do **not** justify importing language-model context thresholds,
FlashAttention or KV-cache speedups, million-token claims, mobile deployment
claims, or absolute statements about Mamba/Transformer recall into this paper.
They also do not establish that the hybrid principle transfers to POMDP
control. That transfer is the falsifiable question tested by the SSM-only,
attention-only, full-hybrid, and memory-ordering ablations. Primary papers and
author code, rather than the internal reports, determine every baseline
equation and empirical comparison.

## Registration implications

There is no defensible single “state of the art” number across these suites:
methods differ in observations, learner, privileged information, interaction
budget, and evaluation protocol. The paper must therefore lead with paired
within-protocol comparisons and uncertainty, then place published author-code
numbers in explicitly non-comparable context tables.

Before registered POBAX runs:

1. reproduce a small author-code S5RL and memory-trace reference cell;
2. freeze one small tuning grid with the same trial count per architecture;
3. give memory traces a budget capable of reproducing their slow-learning
   behavior, or label the comparison budget-limited;
4. run fully observable/privileged references where POBAX supplies them;
5. freeze seed manifests and raw-result schemas before evaluating test seeds;
6. retain throughput, latency, state size, and peak-memory measurements; and
7. reserve the USD 10 cloud allowance for an independent environment or
   reproduction check, using the local RTX 4090 Laptop GPU for development.
