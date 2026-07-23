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
- the source-verified POPGym positional MLP;
- Elman RNN, GRU, and LSTM;
- a three-layer causal dilated TCN;
- Fast and Forgetful Memory (FFM);
- Stable Hadamard Memory (SHM);
- LRU, S5RL, recurrent S4D, MS4, and MS4N;
- full-window causal Transformer, Transformer-XL, and GTrXL; and
- ArcMind.

The positional MLP follows the sinusoidal feature encoding, learned clipped
blend, and reset-aware episode counter in pinned POPGym commit
`410d5aa626dae8024f498354d8781a0d1870c399`. It is a shared-head JAX
adaptation, not the unrelated Brax `backend="positional"` setting and not a
scalar normalized timestep.

The [official FFM source at commit
`b3f94d2a0f35ba05089faf19ab1df846057cf8b6`](https://github.com/proroklab/ffm/tree/b3f94d2a0f35ba05089faf19ab1df846057cf8b6/standalone_jax)
contains a JAX recurrence with explicit recurrent state and episode-done
inputs. The shared core preserves the paper's POPGym setting of 32 decay
traces and four complex temporal contexts. This is a 128-value complex state,
equivalent to 256 real scalar dimensions. It also retains the official 1 to
1024 period schedule. Parameter matching searches only the FFM output width,
leaving the decay and context structure fixed. The FFM feature is intentionally
shared by the common actor and critic heads. Initialization, reset, sequence,
recurrent-state, equation, JIT, gradient, and parameter-count tests cover this
adaptation.

The [official SHM `v1.1` source at commit
`40d73d44936e47a29e2c76a481d93c434b857ea1`](https://github.com/thaihungle/SHM/tree/40d73d44936e47a29e2c76a481d93c434b857ea1)
is PyTorch. The JAX core follows the pinned POPGym policy cell and exposes
initial memory, asynchronous resets, explicit addresses, and a time-major
scan. The paper and POMDP source sample uniformly from all 128 address rows,
while the released standalone and POPGym source always select row zero.
`shm` therefore names the scientific `paper_uniform` mode and
`shm_v1_1_popgym_compat` names the source-compatible row-zero check. The
collector stores every sampled address and PPO replays the same trace during
loss recomputation. Unit tests require zero pre-update KL under that replay.

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
  --evidence-tier smoke \
  --require-gpu
```

`--quick` results are explicitly marked `development_smoke_not_for_paper`.
They validate plumbing and learnability; they are not registered evidence.
Each JSON artifact and its hashed frozen configuration record the complete
policy core, parameter counts and ratio, environment source and reference
class, PPO configuration, accelerator, and exact POBAX commit.

Frozen Cartesian matrices use the fail-closed launcher:

```bash
python -m benchmarks.pobax.run_matrix \
  --registration benchmarks/pobax/manifests/smoke_controls_v1.json \
  --output-root benchmark_results/pobax/smoke-controls-v1
```

The first predeclared multi-seed pilot uses
`benchmarks/pobax/manifests/tmaze_pilot_v1.json`. It trains ten controls for
250,000 exact environment transitions on the three development seeds. Its
pilot status remains ineligible for paper performance claims.

The numerical repair replay uses
`benchmarks/pobax/manifests/tmaze_shm_repair_v2.json`. It reruns SHM and
ArcMind on the same T-Maze pilot seeds, budget, learner settings, and
evaluation contract. ArcMind is the required comparison anchor. This replay
is pilot evidence and remains ineligible for paper performance claims.

The next coverage and ablation pilot uses
`benchmarks/pobax/manifests/tmaze_coverage_ablation_v2.json`. It compares the
four-frame MLP, LRU, S4D, four ArcMind ablations, and complete ArcMind under the
same T-Maze pilot seeds, exact 250,000-step budget, learner settings, and
evaluation contract. The manifest is independent of the SHM repair outcome.
It is development evidence and remains ineligible for paper performance
claims.

Registration schema v1 remains readable for existing development matrices.
New experiment registrations use schema v2 and must name one of two comparison
profiles:

- `pobax_author_semantics` follows the pinned author's optimizer, update, and
  step-budget semantics inside ArcMind's shared runner. If the requested
  interaction budget is not divisible by one vector rollout, the number of
  PPO updates is floored, as in the author code. This profile does not execute
  the author's learner and must not be labeled an author-code reproduction.
- `arcmind_shared_comparison` requires exact divisibility and gives every
  policy core the same realized interaction count.

Both profiles explicitly register `num_envs`, `rollout_steps`,
`update_epochs`, `num_minibatches`, `learning_rate`, `gae_lambda`,
`entropy_coefficient`, and `anneal_learning_rate`. The launcher rejects
missing or additional learner fields. Schema v2 configurations and result
artifacts store the requested and realized environment step counts separately.
The source learning rate schedule is constant within a PPO update. For
optimizer step `s`, it is
`lr * (1 - floor(s / (num_minibatches * update_epochs)) / num_updates)`.
The schedule reaches zero at the optimizer boundary and is clamped to zero
after that boundary.

An exact author-code reproduction requires a separate pinned author-code
runner and separately labeled artifacts. Results from that lane cannot be
silently combined with either shared-runner profile.

Every PPO update must report finite loss, actor loss, value loss, entropy, and
approximate KL metrics. Training stops immediately on the first invalid value
and reports the update index and realized environment-step count. A recent
return may be unavailable only until the first episode completes. Both
development and registered aggregators enforce the same rule on every history
point and on the final training metrics, including legacy v1 artifacts.

The launcher first describes every cell, hashes the expanded manifest, and
then starts each model in a fresh process. It requires clean Git provenance,
skips only identity-, configuration-, and provenance-compatible completed
cells, refuses collisions, records the dependency lock, external source
commits, and actual Python, package, JAX, backend, and accelerator identities,
then writes a stable completion index plus checksum manifest.
Every registration declares `matrix_kind`. A `primary_comparison` matrix must
include ArcMind. An `upper_reference` matrix contains only the memoryless
policy on its separately labeled privileged or full-observation task. A
registered-final matrix is accepted only with exactly 30 paired seeds.
Each child process writes an immutable per-cell log beside its JSON artifact.
The completion index records both hashes, and the directory checksum covers
the logs as well as the structured results.

A completed registered-final matrix is aggregated with:

```bash
python -m benchmarks.pobax.aggregate_registered \
  benchmark_results/pobax/registered-final/frozen_manifest.json \
  benchmark_results/pobax/registered-final-analysis/aggregate.json
```

The aggregator accepts registered-final artifacts only. It verifies the
Cartesian matrix, configuration hashes, clean provenance, raw evaluation
returns, episode counts, and within-task learning-curve grids before computing
seed-level summaries and paired differences against ArcMind. Early curve
entries with no completed episode remain JSON `null`; aggregation starts at
the first step where every cell in that task has a finite return.
Derived aggregates live outside the immutable raw-matrix directory so its
checksum manifest continues to cover every file it was created to protect.
The write commands reject any output path inside that directory.

Smoke and pilot matrices use the separately labeled development aggregator:

```bash
python -m benchmarks.pobax.aggregate_development \
  benchmark_results/pobax/tmaze-pilot-v1 \
  benchmark_results/pobax/aggregates/tmaze-pilot-v1.json
```

Development aggregation preserves raw seed returns, validates the recorded
parameter match and environment semantics, and stamps the result as not for
paper. Legacy primary pilots whose older frozen configuration omitted those
fields are accepted only when the completed artifact itself passes the same
checks, and the aggregate records that limitation explicitly.

Primary and upper-reference matrices are paired only through a validated link:

```bash
python -m benchmarks.pobax.link_upper_reference \
  benchmark_results/pobax/registered-primary \
  benchmark_results/pobax/registered-upper \
  benchmark_results/pobax/aggregates/primary-upper-link.json
```

The linker requires identical ordered seeds, exact alias-to-primary mappings,
learner and evaluation contracts, source commits, dependency locks, and
registered budgets. It validates both raw checksum inventories and permits
backend or device differences only when all non-device runtime fields match.

The discrete development runner accepts `simple_chain`, `tmaze_10`,
`rocksample_11_11`, `battleship_10`, and
`Navix-DMLab-Maze-01-v0`. Simple Chain has one action, so it validates
infrastructure rather than policy quality. The same learner supports
`HalfCheetah-V-v0` and `Walker-V-v0` through a learned state-independent
diagonal-Gaussian action distribution. Separately labeled `Walker-F-v0` and
`HalfCheetah-F-v0` full-observation references use the common runner too.
The discrete upper-reference aliases `tmaze_10-perfect-memory`,
`rocksample_11_11-fully-observable`, and
`Navix-DMLab-Maze-01-fully-observable` invoke an explicit pinned source variant
and accept only the parameter-matched memoryless policy.
`battleship_10-perfect-recall` additionally restores the legal-action mask
omitted by the public perfect-recall wrapper. The Navix reference flattens the
full `(21, 41, 6)` source observation into a narrow learned MLP whose total
parameter count is matched to the primary ArcMind target.
`HalfCheetah-P-v0` is exposed by the library and may remain useful for
development, but it is not a registered task from the published POBAX
benchmark.

For the 1,000-step masked MuJoCo horizon, each requested evaluation episode
receives exactly 1,000 scan steps. Every vector worker must complete the
requested count or the cell fails.

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
