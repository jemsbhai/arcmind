# ArcMind benchmark and baseline audit

Status: living primary-source audit, updated 2026-07-23.

This document separates like-for-like backbone controls from methods that
change the learner, supervision, observation privileges, or task. It is not a
claim that ArcMind exceeds any listed method.

## Benchmark selection

### Primary: POBAX

The published
[POBAX paper](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_153.pdf)
defines the primary RL suite. Its selection criterion is directly aligned with
the ArcMind hypothesis: environments should span distinct forms of partial
observability and exhibit a measurable gap between memoryless and
better-informed agents. The official source is
[pinned to commit
`a5e1d62d14e4efe783885b9d4f19cffa2a568eec`](https://github.com/taodav/pobax/tree/a5e1d62d14e4efe783885b9d4f19cffa2a568eec).

The registered subset and published reference protocol are:

| Source environment identifier | Partial-observability category | Published training steps | Tuning seeds | Final seeds |
|---|---|---:|---:|---:|
| `tmaze_10` | object uncertainty and tracking | 1,000,000 | 5 | 30 |
| `rocksample_11_11` | object uncertainty | 5,000,000 | 5 | 30 |
| `battleship_10` | spatial uncertainty and episode nonstationarity | 10,000,000 | 10 | 30 |
| `Walker-V-v0` | moment features | 50,000,000 | 5 | 30 |
| `HalfCheetah-V-v0` | moment features | 50,000,000 | 5 | 30 |
| `Navix-DMLab-Maze-01-v0` | spatial uncertainty and episode nonstationarity | 10,000,000 | 5 | 30 |

`HalfCheetah-P-v0` exists in the library but is not one of the masked MuJoCo
tasks validated for the published POBAX benchmark. Battleship uses an action
mask, but action masking is not a category of partial observability in the
paper. Simple Chain remains an infrastructure check because its single action
cannot discriminate policy quality.

### Secondary and diagnostic suites

- [POPGym](https://arxiv.org/abs/2303.01859), with
  [official source at commit
  `410d5aa626dae8024f498354d8781a0d1870c399`](https://github.com/proroklab/popgym/tree/410d5aa626dae8024f498354d8781a0d1870c399),
  supplies controlled memory primitives and scalable memory-length sweeps.
  Its positional MLP control is required because adding timestep information
  often improves a memoryless policy and guards against crediting memory for
  position inference.
- [Memory Gym](https://www.jmlr.org/papers/v26/24-0043.html) supplies finite
  and endless 2D memory tasks with official GRU and Transformer-XL references.
  Its [official source is commit
  `a94f2b60d1769ea44df3226561488768e1dff9f4`](https://github.com/MarcoMeter/endless-memory-gym/tree/a94f2b60d1769ea44df3226561488768e1dff9f4).
  It is the preferred first external stress suite.
- [Memory Maze](https://openreview.net/pdf?id=yHLvIlE9RGN) is a 3D
  embodied-memory stretch test after the low-dimensional policy-backbone claim
  is established. Its [official source is commit
  `4030901cef3b7d4e7f92e62099e8be378303dc0a`](https://github.com/jurgisp/memory-maze/tree/4030901cef3b7d4e7f92e62099e8be378303dc0a).
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
| no learned memory | MLP, positional MLP, four-frame MLP | MLP and frame stack implemented; positional MLP required | Establishes observation, timestep-aware, and short-window floors. POPGym found the positional control important enough to recommend in future memory comparisons. |
| conventional recurrence | Elman RNN, GRU, LSTM | implemented | Strong recurrent RL is mandatory: careful recurrent model-free RL matched or exceeded specialized methods on 18/21 environments in the [ICML 2022 study](https://proceedings.mlr.press/v162/ni22a.html). |
| RL-specific fading memory | Fast and Forgetful Memory | required | [FFM, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e3bf2f0f10774c474de22a12cb060e2c-Abstract-Conference.html) is a direct drop-in RL memory baseline evaluated on POPGym. |
| learned matrix memory | Stable Hadamard Memory | required | [SHM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/b6446566965fa38e183650728ab70318-Paper-Conference.pdf) is a direct learned write, calibration, and read baseline evaluated against GRU and FFM on POPGym. |
| finite convolution | causal dilated TCN | implemented | Separates finite receptive field from indefinite recurrence. |
| fixed fading memory | memory traces | implemented | The [ICML 2025 method](https://proceedings.mlr.press/v267/eberhard25a.html) uses compact exponential moving averages and reports improved sample efficiency in some partially observable tasks. The official T-Maze example uses decays 0 and 0.985 and a 20M-step budget; ArcMind must not compare against an undertrained trace model. |
| stable linear recurrence | LRU | implemented | [ICML 2024](https://proceedings.mlr.press/v235/lu24h.html) identifies the Deep Linear Recurrent Unit as a strong alternative to Transformers in POMDPs; its stable parameterization comes from the [LRU paper](https://proceedings.mlr.press/v202/orvieto23a.html). |
| structured SSM | S4D, MS4, MS4N | implemented | Tests stable input-invariant diagonal dynamics. S4D follows the [paper](https://arxiv.org/abs/2206.11893) and [official implementation](https://github.com/state-spaces/s4); MS4/MS4N are causal policy adaptations of the [2026 preprint](https://arxiv.org/abs/2605.27406). |
| RL-specific structured SSM | S5RL | implemented | S5 is a multi-input/multi-output SSM with an [official JAX implementation](https://github.com/lindermanlab/S5); [S5RL](https://proceedings.neurips.cc/paper_files/paper/2023/hash/92d3d2a9801211ca3693ccb2faa1316f-Abstract-Conference.html) specifically adapts it to reset-aware RL. |
| exact attention | full-window causal Transformer | implemented | Provides the high-cost exact-context reference. |
| recurrent attention | Transformer-XL, GTrXL | implemented | Tests segment recurrence and the GRU-gated, identity-map architecture introduced by [GTrXL](https://proceedings.mlr.press/v119/parisotto20a.html). |
| proposed hybrid | ArcMind plus five registered ablations | implemented | Isolates the fast path, exact memory, temporal ordering, fusion gate, and full hybrid. |

### FFM and SHM shared-learner compatibility

The [official FFM repository at commit
`b3f94d2a0f35ba05089faf19ab1df846057cf8b6`](https://github.com/proroklab/ffm/tree/b3f94d2a0f35ba05089faf19ab1df846057cf8b6)
contains a JAX implementation whose call accepts the input sequence, recurrent
state, and episode-done flags. FFM can therefore be adapted as a policy core in
the shared JAX PPO without changing the learner, loss, collector, or optimizer.
Eligibility still requires parity tests for parameters, initialization,
asynchronous resets, sequence outputs, recurrent state, and gradients.

The [official SHM `v1.1` source at commit
`40d73d44936e47a29e2c76a481d93c434b857ea1`](https://github.com/thaihungle/SHM/tree/40d73d44936e47a29e2c76a481d93c434b857ea1)
is PyTorch. The released memory module uses ordinary differentiable recurrence
and does not require an auxiliary RL loss, so its equations can be placed
inside the shared PPO without changing the learner. It is not yet a
like-for-like implementation: the repository includes stateful PyTorch
benchmark adapters, but no official JAX streaming adapter is provided. The
required port must expose initial memory, accept asynchronous reset masks and
explicit random-address keys, and run a time-major scan. It must match the
released PyTorch equations under a fixed address sequence and validate
stochastic addressing distributionally before registered use.

The SHM audit found a source-level incompatibility that must remain explicit.
The paper and the official POMDP implementation sample uniformly from all 128
address rows, but the standalone and POPGym `v1.1` implementations call
`uniform_(0, 1).long()`. Values in that interval truncate to zero, so those
paths always select address row zero. The JAX port therefore requires two
named modes: `paper_uniform`, which follows the paper and POMDP code, and
`v1_1_popgym_compat`, which reproduces the released POPGym behavior. The main
shared-learner baseline should use `paper_uniform` and must not be described
as an exact reproduction of the published POPGym agent.

The implemented memory-trace adapter applies the fixed exponential averages to
the common causal policy input rather than the original paper's
environment-specific one-hot observation encoder. The implemented MS4/MS4N
replaces sequence-classification pooling with causal actor-critic heads. The
S5RL core preserves the authors' HiPPO-LegS DPLR initialization, ZOH
discretization, conjugate-symmetric recurrence, reset behavior, and full-GLU
residual block, but uses the common direct input projection and actor-critic
heads. These differences must be stated in the paper and checked against
author-code results where possible.

POBAX labels its attention agent Transformer-XL. The pinned repository enables
GRU-style gates in the transformer configuration by default, while GTrXL is a
specific architecture from the
[ICML 2020 paper](https://proceedings.mlr.press/v119/parisotto20a.html).
ArcMind must record the actual gating and normalization settings. The main text
should use Transformer-XL for a POBAX author configuration unless those
settings verify that it is exactly GTrXL.

## Algorithm-level and contextual comparisons

These methods cannot be placed in the main parameter-matched backbone table
without changing the experimental question:

- [Lambda discrepancy](https://proceedings.neurips.cc/paper_files/paper/2024/hash/73073ccb3bc559fd001e66b9079d6d5e-Abstract-Conference.html)
  adds a second critic and an auxiliary discrepancy objective. Report the
  POBAX author result in a separate algorithm-level table.
- [Memory Traces](https://proceedings.mlr.press/v267/eberhard25a.html) can be
  used as a fixed policy representation in the shared learner, while its
  author-code result remains a separate compatibility reproduction.
- [Memoroids](https://proceedings.neurips.cc/paper_files/paper/2024/hash/19f7f755908372efb25826d61959cdf9-Abstract-Conference.html)
  changes batching and the training formulation for associative recurrences.
  It is a training-method comparison, not a pure backbone.
- [Recall to Imagine](https://recall2imagine.github.io/) adds a world model
  and imagined rollouts.
- [Memo](https://proceedings.neurips.cc/paper_files/paper/2025/hash/96889893231d651898b0de42fdbee3a6-Abstract-Conference.html)
  targets learned memory management for embodied agents at a different system
  scale.
- [RATE](https://openreview.net/forum?id=kByN4v0M3e) evaluates recurrent
  attention in offline RL over stored trajectories, not the shared online PPO
  regime.
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

There is no defensible single "state of the art" number across these suites:
methods differ in observations, learner, privileged information, interaction
budget, and evaluation protocol. The paper must therefore lead with paired
within-protocol comparisons and uncertainty, then place published author-code
numbers in explicitly non-comparable context tables.

Before registered POBAX runs:

1. implement and parity-test the positional MLP, FFM, SHM, and privileged
   observation adapters;
2. reproduce a small author-code FFM or SHM reference cell and an S5RL or
   memory-trace reference cell;
3. classify every artifact as smoke, pilot, preliminary, or registered final;
4. treat quick and 131,072-step cells as infrastructure evidence only;
5. use the published interaction budget for every method in each registered
   task cell;
6. target 30 paired final seeds, matching the published POBAX result count,
   and label any 10-seed table preliminary;
7. freeze one small tuning grid with the same trial count per architecture;
8. give memory traces a budget capable of reproducing their slow-learning
   behavior, or label the comparison budget-limited;
9. run fully observable or privileged references where POBAX supplies them;
10. freeze seed manifests and raw-result schemas before evaluating final
    seeds;
11. retain throughput, latency, state size, and peak-memory measurements; and
12. reserve the USD 10 cloud allowance for an independent environment or
    author-code reproduction check, using the local RTX 4090 Laptop GPU for
    development and registered runs.
