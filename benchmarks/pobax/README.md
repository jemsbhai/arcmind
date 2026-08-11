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

The development learner supports these policy cores through that common path.
Unless a lane is explicitly marked supplemental below, learned controls use
the ArcMind parameter-matching contract:

- memoryless MLP, four-frame MLP, and shared-input memory adaptations;
- the parameter-matched AGaLiTe shared lane;
- the fixed source-compatible AGaLiTe policy for discrete tasks;
- the source-compatible supplemental Memory Traces policy for discrete tasks;
- the source-verified POPGym positional MLP;
- Elman RNN, GRU, and LSTM;
- a three-layer causal dilated TCN;
- Fast and Forgetful Memory (FFM);
- Stable Hadamard Memory (SHM);
- LRU, S5RL, recurrent S4D, MS4, and MS4N;
- the source-audited Mamba-1 baseline;
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
exact-attention reference. Its window is the task's complete maximum episode
horizon: 1,000 steps for RockSample and the other bounded POBAX tasks, and
2,000 steps for Navix. New tuning and registered configurations serialize
this value as `policy_core.window_length` and fail validation or resume when
it differs from `evaluation_max_episode_steps`. Earlier schema-v2 development
artifacts with the former 32-step default remain readable, but they are not
full-horizon attention measurements and are ineligible for paper claims. The
POBAX paper names its attention agent Transformer-XL, while the pinned
repository enables GRU-style transformer gating by default. Compatibility
reports must record the gate and normalization settings. They must use the
name Transformer-XL unless those settings verify the exact GTrXL
architecture.

Memory Traces has two explicit evidence lanes. `memory_trace_official` follows
the [official source at commit
`fcfdacc0b0a06dc181b49b9ef95893dbae7f2bcd`](https://github.com/onnoeberhard/memory-traces/tree/fcfdacc0b0a06dc181b49b9ef95893dbae7f2bcd).
It traces observations only, uses non-episodic traces with trace-major
flattening, resets each completed worker before incorporating the current
observation, and retains separate actor and critic networks. Each network has
two 64-unit tanh layers. Hidden kernels use orthogonal gain `sqrt(2)`, while
the actor and critic output gains are `0.01` and `1.0`. This fixed official
architecture is not parameter matched, uses the official categorical actor,
and is reported only as supplemental source-compatible evidence on discrete
tasks.

`memory_trace_shared` retains the same trace recurrence on ArcMind's augmented
policy input and uses the common parameter-matched shared trunk and heads. It
is the like-for-like main-table adaptation. The legacy `memory_trace_mlp`
identifier is a development compatibility alias and is rejected for tuning
and registered-final evidence.

Both lanes currently freeze decay rates `[0.0, 0.985]`. Those values come from
the official TMaze64 example. They are not claimed to be author-selected
POBAX hyperparameters. The source contract records SHA256 hashes for
`traces/ppo.py` and `examples/ppo_tmaze.py`, plus an immutable CPU differential
fixture generated by executing the official `Trace` and `ActorCritic` code.
Matrix resume and both aggregators reject source, decay, architecture,
parameter-contract, and comparison-role drift. Supplemental official results
are kept out of the parameter-matched paired comparison.

The LRU uses the stable ring initialization and normalized input projection from
[Orvieto et al.](https://proceedings.mlr.press/v202/orvieto23a.html).
S5RL preserves the authors' HiPPO-LegS initialization, zero-order-hold
discretization, reset-aware recurrence, and full-GLU residual block while
using the shared input and actor-critic heads.

The Mamba-1 baseline follows the
[official Mamba source at commit
`10b5d6358f27966f6a40e4bf0baa17a460688128`](https://github.com/state-spaces/mamba/tree/10b5d6358f27966f6a40e4bf0baa17a460688128).
It uses one Mamba-1 block with expansion factor 2, state size 16, convolution
width 4, automatic time-step rank, and RMSNorm with epsilon `1e-5`.
ArcMind's common policy input and actor-critic heads replace the language
model embedding and output head, while the selective recurrence and residual
normalization structure remain source aligned. Each environment carries
independent convolution and SSM caches, and reset masks clear only the
completed environments. Parameter matching searches only the hidden width.

The source contract pins Mamba package version `2.2.6.post3`, the audited
commit, SHA256 hashes for `mamba_simple.py`, `block.py`,
`mixer_seq_simple.py`, `config_mamba.py`, and `layer_norm.py`, plus an
immutable output fixture from the official dependency-light `Mamba.step`
path. Frozen configurations and result artifacts record this metadata.
Matrix resume and both aggregators reject missing or changed Mamba source
metadata.

AGaLiTe has two source-audited evidence lanes. Both implement the executable
finite-channel recurrence from
[official commit
`101acbecc121a258ad8f7e58e2f782f546674979`](https://github.com/subho406/agalite/tree/101acbecc121a258ad8f7e58e2f782f546674979).
The executable stores exactly `R` channels at
`linspace(-pi, pi, R)`, initializes its phase counter at one, uses phase two
for the first token, never resets phase, and normalizes by
`2 * R * dot(s, q) + 1e-5`. This differs from the finite-channel equations in
the paper and is frozen as an executable-source contract.

`agalite_source_compat` is the complete released T-Maze vector policy:
observation-only input, four 128-wide layers, four heads of width 64,
feedforward width 128, `eta=4`, `R=2`, and separate 128-wide tanh actor and
critic heads. It uses a categorical actor, is rejected on continuous tasks,
and is reported as fixed-architecture supplemental evidence. The released
T-Maze configuration used A2C. Running this policy in the shared POBAX PPO
learner is an explicit learner adaptation, not an author-code performance
reproduction.

`agalite_shared` applies the same released recurrence and GTrXL-style block to
the full augmented policy input. It uses the common heads and shared PPO.
Parameter matching varies only an even model width while preserving four
layers, four heads, head width `D/2`, feedforward width `D`, `eta=4`, and
`R=2`. The registry freezes upstream source hashes, a CPU Flax differential
fixture, the LayerNorm epsilon used by that fixture, the parameter contract,
and the comparison role. The upstream requirements are unpinned, so the
fixture records exact JAX and Flax versions.

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

Large frozen matrices can run as deterministic isolated shards. Each worker
must receive its own output root and, for GPU-required registrations, exactly
one JAX-visible GPU:

```bash
python -m benchmarks.pobax.run_matrix \
  --registration benchmarks/pobax/manifests/compute_aware_tuning_v1.json \
  --output-root "/shared/arcmind/tuning/shards/shard-${SHARD_INDEX}" \
  --shard-count 4 \
  --shard-index "${SHARD_INDEX}"
```

The shard count and zero-based index select cells by their frozen manifest
ordinal modulo the shard count. They do not change the scientific
registration, full manifest, configuration hashes, cell IDs, or artifact
paths. A shard contains the identical full `registration.json` and
`frozen_manifest.json`, only its assigned artifacts and logs, and the
logistical `shard_completion.json` and `shard_checksums.sha256`. It never
creates the canonical completion or checksum files.

After every worker exits successfully, merge from one process running under
the same runtime contract. For a GPU-required matrix, the finalizer must also
see exactly one identical GPU because it reconstructs the frozen manifest:

```bash
python -m benchmarks.pobax.merge_matrix_shards \
  --registration benchmarks/pobax/manifests/compute_aware_tuning_v1.json \
  --output-root benchmark_results/pobax/compute-aware-tuning-v1 \
  --shard-root /shared/arcmind/tuning/shards/shard-0 \
  --shard-root /shared/arcmind/tuning/shards/shard-1 \
  --shard-root /shared/arcmind/tuning/shards/shard-2 \
  --shard-root /shared/arcmind/tuning/shards/shard-3
```

Shard arguments are self-identifying and may be supplied in any order. The
merger locks every source and the canonical target, validates exact partition
coverage and source checksums before publication, copies without replacement,
and emits the existing canonical `completion_index.json` and
`checksums.sha256` in full manifest order. Keep scheduler logs, lock files,
and retry metadata outside all raw roots. The complete four-A100 Slurm
workflow is in [`cluster/README.md`](cluster/README.md).

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

The attention-horizon repair pilot uses
`benchmarks/pobax/manifests/tmaze_attention_horizon_repair_v3.json`. It reruns
ArcMind after increasing the benchmark attention window to cover the T-Maze
start cue. The frozen three-seed contract retains the earlier pilot budget,
learner settings, and evaluation count. The launcher requires clean Git
provenance and a GPU. This pilot remains ineligible for paper performance
claims.

Registration schema v1 remains readable for existing development matrices.
New pilot registrations and separately labeled registered upper references
use schema v2 and must name one of two comparison profiles:

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

Legacy development tuning uses schema v3 and `matrix_kind:
hyperparameter_selection`. Its `candidate_families` list groups immutable
candidate IDs under one implementation model per family. Every candidate
records a complete learner configuration. Candidate IDs are used in frozen
configurations, cell IDs, artifact paths, manifests, completion indexes, and
aggregate rows, while `model_family` and `implementation_model` remain
explicit. Family IDs, implementation models, and candidate IDs are unique.
Every family has the same number of candidates, with no duplicate normalized
learner configuration inside a family. The exact normalized learner
configuration set must also be identical across families.

Legacy 30-seed registered-final comparisons use schema v4. They contain no free
global learner block. Instead, `tuning_selection` binds one task to the raw
schema-v3 tuning matrix, its canonical aggregate and SHA256, its source
registration, manifest, completion-index, checksum-inventory, and
implementation-source hashes, and the exact winning candidate, learner, and
implementation-source hash for every model family. Loading the registration
rebuilds the tuning aggregate from the raw matrix, validates the exact raw
file inventory and checksums, and requires byte identity with the bound
aggregate. The final matrix must use the same deterministic implementation
source manifest, dependency lock, external POBAX and Navix commits, and
complete runtime contract, including backend and ordered device list, as
tuning. The repository commit itself may
differ so that an audited tuning selection can be frozen in a later clean
commit. The schema-v4 frozen manifest also binds the SHA256 of the complete
canonical final registration file, so rechecksumming cannot authorize a
registration change. The final 30-seed manifest must be disjoint from all tuning seeds.
Schema-v2 registered upper references remain valid without this binding,
including the `pobax_author_semantics` lane.

The compute-aware paper study uses three separate fail-closed schemas:

- Schema v5 tunes 13 ordered model families on T-Maze and RockSample at
  1,000,000 steps each. It uses learning rates `0.0001`, `0.00025`, and
  `0.0005`, seeds `4409`, `5519`, and `6637`, one evaluation episode per
  vector environment, and mandatory GPU provenance. Its exact 234-cell
  schema-v9 artifact matrix selects one learner per family by mean task rank,
  mean range-normalized task regret, then learner ID.
- Schema v6 rebuilds and proves the canonical schema-v5 aggregate, then runs
  the exact sparse 490-cell primary matrix. It uses four tasks, 49 task-model
  groups, seeds `10000` through `10009`, 16 evaluation episodes per vector
  environment, and schema-v10 artifacts. Seven source-compatible or ablation
  models inherit a named family learner. Every configuration, artifact,
  manifest cell, and resume check freezes that inheritance.
- Schema v7 binds a completed schema-v6 primary matrix and the actual selected
  memoryless learner. It runs exactly four privileged aliases, one model, and
  the same ten seeds, for 40 schema-v11 artifacts. Its aggregate and the
  schema-v6/schema-v7 link independently rebuild the primary evidence and
  verify every file hash, internal manifest identity, learner, evaluation,
  alias, and provenance field.

Schemas v3 and v4 remain supported for legacy 30-seed evidence. They are not
the first-paper compute-aware contract.

The implementation-source manifest is a versioned canonical manifest over
every tracked Python file under `arcmind/` and every tracked non-test Python
file under `benchmarks/pobax/`. It records each repository-relative path and
file SHA256, then hashes the canonical manifest. This scope includes shared
training, aggregation, environment, model-core, and reference code while
excluding tests.

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
The canonical per-cell log is written only after the child process exits
successfully. A failed attempt keeps its log and any partial artifact under
the sibling `<raw-matrix>.attempts` tree, preserving the cell-relative path
and assigning a unique attempt identity. Attempt evidence never enters the
immutable raw matrix, its completion index, or its checksum inventory.
Every registration declares `matrix_kind`. A `primary_comparison` matrix must
include ArcMind. An `upper_reference` matrix contains only the memoryless
policy on its separately labeled privileged or full-observation task. A
`hyperparameter_selection` matrix uses schema v3 legacy candidate families or
the exact schema v5 family by learner grid and cannot be aggregated by the
registered-final path. Legacy registered-final matrices require exactly 30
paired seeds. Schema v6 and schema v7 require the exact ordered ten-seed
manifest and their respective selection or primary bindings.
Each child process writes an immutable per-cell log beside its JSON artifact.
The completion index records both hashes, and the directory checksum covers
the logs as well as the structured results.

A completed registered-final matrix is aggregated with:

```bash
python -m benchmarks.pobax.aggregate_registered \
  benchmark_results/pobax/registered-final/frozen_manifest.json \
  benchmark_results/pobax/registered-final-analysis/aggregate.json
```

The aggregator accepts registered-final artifacts only. It verifies the exact
Cartesian legacy inventory or sparse schema-v6 inventory, configuration
hashes, clean provenance, raw evaluation returns, episode counts, and
within-task learning-curve grids before computing seed-level summaries and
paired differences against ArcMind. Early curve
entries with no completed episode remain JSON `null`; aggregation starts at
the first step where every cell in that task has a finite return.
Standalone aggregation also requires the frozen sibling `registration.json`,
`completion_index.json`, canonical per-cell logs, and `checksums.sha256`.
It checks exact canonical registration bytes, completion-row identities and
hashes, every artifact and log hash, and exact checksum-inventory equality
with all regular files under the raw root. Completion indexes must also have
exact canonical JSON bytes. No failed-attempt or other extra file is accepted
inside the raw root, even when checksummed. The aggregate records the three
validated raw-index hashes and explicit integrity flags.
For schema-v2 upper references, it also checks each validated cell against
the frozen learner, environment budget, evaluation episode count, comparison
profile, and quick-run contract. A declared GPU requirement must agree with
the validated runtime provenance.
Schema-v1 registered artifacts retain legacy aggregation support, but their
frozen evaluation episode count must match every validated cell. A
legacy registered-final primary-to-upper link requires schema v4 and its
frozen tuning selection. The compute-aware link accepts only the exact schema
v6 and schema v7 pair.
Derived aggregates live outside the immutable raw-matrix directory so its
checksum manifest continues to cover every file it was created to protect.
The write commands reject any output path inside that directory.

The compute-aware study uses these canonical repository-relative paths:

```text
benchmark_results/pobax/compute-aware-tuning-v1
benchmark_results/pobax/aggregates/compute-aware-tuning-v1.json
benchmarks/pobax/manifests/compute_aware_final_v1.json
benchmark_results/pobax/compute-aware-primary-v1
benchmark_results/pobax/aggregates/compute-aware-primary-v1.json
benchmark_results/pobax/registrations/compute-aware-upper-v1.json
benchmark_results/pobax/compute-aware-upper-v1
benchmark_results/pobax/aggregates/compute-aware-upper-v1.json
benchmark_results/pobax/aggregates/compute-aware-primary-upper-v1.json
```

The schema 6 registration is committed before primary execution. The
hash-bearing schema 7 registration is generated under the ignored
`benchmark_results` tree after the primary matrix completes. Schema 7 then
runs from the same clean commit and runtime as schema 6 because the final
linker requires their complete provenance objects, including Git provenance,
to match.

Schema v6 keeps raw returns inside task groups. Its cross-task analysis is
limited to the ordered eight-model all-task intersection. It resamples the ten
paired seed indices independently within each task, reuses one 10,000-draw
matrix across methods in that task, fixes task weights at 0.25, and assigns
0.5 to an exact return tie. Tasks are not resampled. FFM, SHM, LRU, S4D,
Transformer-XL, the two source-compatible lanes, and the five ArcMind
ablations receive task-local paired comparisons against ArcMind. Their raw
returns and differences are never pooled across tasks, and the source lanes
remain separately labeled supplemental evidence.

Smoke, pilot, and tuning matrices use the separately labeled development
aggregator:

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

The `development_tuning` tier is selection evidence only. Legacy schema v3
requires one published primary task, equal candidate cardinality, and the
published task-specific tuning-seed count. Schema v5 instead requires the
exact compute-aware 234-cell design described above. Every candidate uses its
schema's frozen seed set, budget, evaluation contract, and training-step grid.
The structural learner fields `num_envs`,
`rollout_steps`, `update_epochs`, and `num_minibatches` are identical across
the entire matrix. Candidates may vary only `learning_rate`, `gae_lambda`,
`entropy_coefficient`, and `anneal_learning_rate`. Completion and checksum
indexes, frozen environment semantics, and parameter-match contracts are
mandatory. Every schema-v3 completion row must carry the immutable cell log
path and hash, and the checksum inventory must cover every log.
Equal cardinality is not sufficient: every family must use the exact same
normalized set of complete learner configurations. Schema v5 further fixes
the exact family order, learner IDs, learning rates, tasks, and seeds.
The schema-v3 checksum inventory must equal the exact canonical registration,
manifest, completion, artifact, and log paths. Additional files inside the
raw tuning root are rejected even when checksummed. A declared GPU
requirement must agree with the frozen manifest and artifact runtime.

For tuning only, the aggregator removes the leading prefix through the latest
first finite `mean_recent_return` across all candidate and seed cells, then
requires at least two shared finite curve points. It integrates each seed
curve by the trapezoidal rule without extrapolation and divides by the common
integration width.
Legacy schema-v3 candidates are ranked separately within each model family by
mean seed `auc_mean_return`, with candidate ID as the exact-score tie breaker.
Schema v5 first computes task scores, shared ranks, and range-normalized
regret, then selects by mean rank, mean regret, and learner ID. Neither schema
ranks model families. Both select a configuration, never a checkpoint. Final
evaluation return is preserved for audit but cannot affect selection. The
aggregate status is
`development_tuning_selection_aggregate_not_for_paper`, and its eligibility
block prohibits registered-final evidence and paper performance claims. A
legacy selection enters schema v4. A compute-aware selection enters schema
v6. Both are rerun on disjoint final seeds and revalidated during registered
aggregation.

Primary and upper-reference matrices are paired only through a validated link:

```bash
python -m benchmarks.pobax.link_upper_reference \
  benchmark_results/pobax/registered-primary \
  benchmark_results/pobax/registered-upper \
  benchmark_results/pobax/aggregates/primary-upper-link.json
```

The linker requires identical ordered seeds, exact alias-to-primary mappings,
learner and evaluation contracts, source commits, dependency locks, and
registered budgets. It validates exact raw file inventory and checksums for
both matrices. The legacy path permits backend or device differences only when
all non-device runtime fields match. The schema-v6/schema-v7 path additionally
requires the upper registration to prove the supplied completed primary
matrix and its exact selected memoryless learner.

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
choices. It is development evidence only. A development tuning run uses the
exact registration for its evidence lineage and remains ineligible for paper
performance claims. A registered final run uses the selected frozen
configuration, a disjoint paired seed manifest, and its registered task
budget.

| Environment identifier | Published steps | Tuning seeds | Final seeds |
|---|---:|---:|---:|
| `tmaze_10` | 1,000,000 | 5 | 30 |
| `rocksample_11_11` | 5,000,000 | 5 | 30 |
| `battleship_10` | 10,000,000 | 10 | 30 |
| `Walker-V-v0` | 50,000,000 | 5 | 30 |
| `HalfCheetah-V-v0` | 50,000,000 | 5 | 30 |
| `Navix-DMLab-Maze-01-v0` | 10,000,000 | 5 | 30 |

Those counts describe the published POBAX reference protocol. The ArcMind
compute-aware study uses this separate frozen contract:

| Environment identifier | Tuning steps | Tuning seeds | Final steps | Final seeds |
|---|---:|---:|---:|---:|
| `tmaze_10` | 1,000,000 | 3 | 1,000,000 | 10 |
| `rocksample_11_11` | 1,000,000 | 3 | 5,000,000 | 10 |
| `battleship_10` | not a tuning task | 0 | 10,000,000 | 10 |
| `Navix-DMLab-Maze-01-v0` | not a tuning task | 0 | 10,000,000 | 10 |

The compute-aware result may be paper evidence after all gates pass, but it
must state that it does not match the published POBAX tuning counts, masked
continuous-control panel, or 30-seed final protocol.
