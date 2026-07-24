# ArcMind Research Protocol

Status: pre-registration draft for the core architecture, dated 2026-07-23.
The intended venue is ICLR. The schedule is deadline-agnostic; external compute
is capped at USD 10, while feasible local compute is the primary resource.

This document is the research contract for ArcMind. It defines the falsifiable
claim, architecture revision, benchmark families, baselines, statistical
protocol, and evidence required before a paper makes performance claims.

## 1. Scope and central claim

ArcMind is a **causal, streaming policy backbone for low-dimensional sensor
observations under partial observability**. It is not a language model, a
vision-language-action model, or an offline-RL algorithm by itself.

The proposed contribution is a dual-rate memory architecture:

1. a recurrent state-space fast path updates on every sensor frame;
2. a bounded exact-recall path runs at a lower decision rate over compressed
   snapshots of strictly prior decision states; and
3. a learned gate combines the recurrent and recalled representations.

For control experiments, the policy input at environment step \(t\) is the
current observation together with the previous action, previous reward, and an
episode-boundary flag. These quantities are available causally at deployment
and are required to distinguish observation histories that would otherwise be
aliased. Sensor-classification experiments use observations alone.

The paper's central hypothesis is:

> Under a matched parameter, training-compute, and inference-memory budget,
> combining recurrent fading memory with bounded exact recall improves
> performance on memory-improvable sensor-control tasks relative to either
> mechanism alone, while retaining bounded streaming state and lower
> decision-time latency than full-history attention.

This is a comparative hypothesis, not an assumed property. It must be rejected
or narrowed if the registered evaluation does not support it.

## 2. Accuracy revision

The initial implementation treated episodic slots as an unordered set and
included the current snapshot in both the attention query and memory keys. It
also exposed `attn_window_size` without enforcing the window. Those choices
prevent a defensible claim of temporal recall.

The revised exact-recall contract is:

- memory reads are chronological, oldest to newest;
- only the most recent `attn_window_size` valid slots are visible;
- each slot receives a learned relative-age embedding, where age zero is the
  newest prior decision state;
- the current decision state is the query and is written only **after** the
  recall operation, so episodic memory contains strictly prior states; and
- batch and recurrent inference implement the same recurrence.

The age encoding is deliberately ablatable. The paper must compare ordered
recall against an otherwise identical unordered-memory variant.

The fast path remains a pure-PyTorch, input-dependent selective SSM. It must not
be described as Mamba or Mamba-2: it does not use the published Mamba block or
its optimized selective-scan implementation. A stable diagonal SSM
(S4D/MS4-style) is a mandatory baseline and a candidate future backend, not
something to add without a controlled comparison.

## 3. Evaluation tracks

### Track A: memory mechanism (primary)

Use the published
[POBAX benchmark](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_153.pdf)
as the primary suite, with the
[official source pinned to commit
`a5e1d62d14e4efe783885b9d4f19cffa2a568eec`](https://github.com/taodav/pobax/tree/a5e1d62d14e4efe783885b9d4f19cffa2a568eec).
POBAX was designed around two requirements ArcMind needs: coverage of distinct
forms of partial observability and a measurable gap between memoryless and
more-informed agents. Prefer low-dimensional observations for the first paper;
a vision encoder would confound the backbone comparison.

The registered low-dimensional task set is:

- `tmaze_10`, which tests object uncertainty and tracking;
- `rocksample_11_11`, which tests object uncertainty;
- `battleship_10`, which tests spatial uncertainty and episode
  nonstationarity;
- `Walker-V-v0` and `HalfCheetah-V-v0`, which retain only velocity features,
  require history to infer missing position information, and test the moment
  features category. At the pinned POBAX commit, Walker-V selects dimensions
  8 through 16 for a 9-value observation, while Walker-F selects all 17
  dimensions. Both use a 6-value bounded continuous action and a 1,000-step
  episode limit. The corresponding upper-reference pairing is Walker-V
  against Walker-F; HalfCheetah-V is similarly paired with HalfCheetah-F,
  whose full observation also has 17 values
  ([environment map](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/__init__.py#L49-L86),
  [Brax wrapper](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/wrappers/gymnax.py#L242-L273)); and
- `Navix-DMLab-Maze-01-v0`, which tests spatial uncertainty and episode
  nonstationarity.

The environment identifiers above are the names used by the pinned source.
`HalfCheetah-P-v0` is exposed by the library but is not one of the two masked
MuJoCo tasks validated for the published POBAX benchmark. Simple Chain has one
action and remains an infrastructure check only. Pixel control and
no-inventory Crafter are held out for a later encoder study.

Use [POPGym](https://arxiv.org/abs/2303.01859) as the controlled memory suite
and [Memory Gym](https://www.jmlr.org/papers/v26/24-0043.html) as the first
external stress suite. Memory Gym supplies finite and endless 2D tasks with
official GRU and Transformer-XL references. Use
[Memory Maze](https://openreview.net/pdf?id=yHLvIlE9RGN) only as a later 3D
stretch evaluation after the low-dimensional claim is established.

Mandatory controls and baselines:

- memoryless MLP, positional MLP, and fixed frame stack;
- Elman RNN, GRU, and LSTM;
- TCN;
- Fast and Forgetful Memory, Stable Hadamard Memory, memory traces, LRU, S4D,
  S5/S5RL, and a modern stable diagonal SSM;
- the repository's selective SSM without exact recall;
- a full-window causal Transformer and the published POBAX Transformer-XL
  configuration at matched parameter count;
- ArcMind without age encoding, without episodic memory, without gating, and
  without the SSM path;
- a fully observable policy or privileged-state upper reference where the
  environment supplies one.

POBAX does not implement a positional MLP. The
[`backend="positional"` setting](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/wrappers/gymnax.py#L242-L273)
selects the Brax simulation backend and is unrelated to policy position
features. The registered positional control is a shared-head JAX
adaptation of the pinned POPGym implementation, not a scalar `t / horizon`
feature. It adds POPGym's fixed sinusoidal encoding at the first hidden
representation, uses a learned scalar blend initialized to 0.5, and carries only
a reset-aware per-environment episode counter
([MLP definition](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/baselines/ray_models/ray_mlp.py#L10-L55),
[encoding](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/baselines/models/embeddings.py#L7-L17),
[blend and counter](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/baselines/ray_models/base_model.py#L240-L317)).
The common actor, critic, causal policy input, learner, and parameter-matching
procedure remain unchanged, and the learned blend scalar counts toward the
parameter total.

Upper references are classified by the information they expose:

| Primary task | Upper invocation | Reference class |
|---|---|---|
| `Walker-V-v0` | `Walker-F-v0` | full Markov observation |
| `HalfCheetah-V-v0` | `HalfCheetah-F-v0` | full Markov observation |
| `rocksample_11_11` | same identifier with `perfect_memory=True` | full Markov observation, including true rock morality |
| `Navix-DMLab-Maze-01-v0` | `Navix-DMLab-Maze-F-01-v0` | full Markov map, position, and direction |
| `tmaze_10` | same identifier with `perfect_memory=True` | persistent-cue upper reference |
| `battleship_10` | same identifier with `perfect_memory=True` | perfect-recall history |
| `simple_chain` | same identifier with `perfect_memory=True` | broken upstream reference, excluded |

This taxonomy follows the
[pinned environment construction](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/__init__.py#L171-L319).
RockSample exposes position and true rock morality
([wrapper](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/rocksample.py#L93-L125)).
Navix-F-01 exposes a full position-encoded array of shape `(21, 41, 6)` and
uses the same 2,000-step maze specification as the partial task
([representation](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/navix_mazes.py#L295-L355),
[registrations](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/navix_mazes.py#L413-L464)).
T-Maze keeps the goal cue visible but does not expose the full latent state
([source](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/tmaze.py#L12-L65)).
Battleship exposes the complete hit and miss history, not the hidden ship
board
([state and wrapper](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/battleship.py#L15-L99)).
The Simple Chain full wrapper refers to nonexistent instance state and is not
eligible for evaluation
([source](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/simple_chain.py#L77-L89)).

These rows are diagnostic upper references, not like-for-like memory-backbone
baselines and not guaranteed empirical ceilings. They change the deployed
policy's information and, under finite optimization, may perform below a
partial-observation method. Report them separately from the parameter-matched
main table.

The shared runner implements the public T-Maze, RockSample, Battleship, and Navix
references as the explicit aliases `tmaze_10-perfect-memory`,
`rocksample_11_11-fully-observable`, and
`Navix-DMLab-Maze-01-fully-observable`, plus
`battleship_10-perfect-recall`. Each alias records the exact source invocation
in its hashed configuration, permits only the parameter-matched memoryless
policy, and inherits the primary task's registered interaction budget.
T-Maze retains the same four-value shape while keeping the cue visible.
RockSample retains the same 33-value shape while replacing uncertain rock
beliefs with true rock morality, so observation width alone does not identify
the information advantage.

Battleship's public perfect-memory path
returns a raw `(10, 10)` history array and omits the ordinary action-mask
dictionary. The implemented adapter preserves the source history and restores
the row-major legal-action mask from unvisited cells. The unregistered state
wrapper is not substituted because its returned axis order disagrees with its
declared space
([wrappers](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/battleship.py#L67-L205)).
Navix expands from a partial `(3, 3, 2)` runtime observation, or 18 flattened
values, to 5,166 full-observation values. After adding the previous action and
two boundary flags, the respective policy input widths are 23 and 5,171. The
implemented upper reference flattens the full input into a memoryless MLP with
hidden width six. It targets the parameter budget of the primary partial-task
ArcMind cell, and all learned parameters are counted, rather than allowing
privileged input width to define a larger budget
([partial wrapper](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/wrappers/nx.py#L59-L132),
[full registration](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/navix_mazes.py#L413-L464)).

POBAX is JAX-native. All accuracy comparisons must use one PPO
implementation, optimizer, rollout collector, and update schedule. The ArcMind
recurrence therefore requires a JAX benchmark implementation with numerical
parity tests against this PyTorch package; comparing separate PyTorch and JAX
learners would introduce an unacceptable framework confound.

[Fast and Forgetful Memory](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e3bf2f0f10774c474de22a12cb060e2c-Abstract-Conference.html)
has an
[official JAX recurrence at commit
`b3f94d2a0f35ba05089faf19ab1df846057cf8b6`](https://github.com/proroklab/ffm/tree/b3f94d2a0f35ba05089faf19ab1df846057cf8b6/standalone_jax).
It accepts recurrent state and episode-done flags. The implemented policy core
uses that recurrence without changing PPO and has parameter, reset, sequence,
equation, JIT, and gradient tests.

[Stable Hadamard Memory](https://proceedings.iclr.cc/paper_files/paper/2025/file/b6446566965fa38e183650728ab70318-Paper-Conference.pdf)
has [official PyTorch code at release `v1.1`, commit
`40d73d44936e47a29e2c76a481d93c434b857ea1`](https://github.com/thaihungle/SHM/tree/40d73d44936e47a29e2c76a481d93c434b857ea1).
Its memory equations serve as a policy core without adding an auxiliary loss
or changing PPO. The repository includes PyTorch benchmark adapters with
explicit recurrent state but no JAX streaming interface. The implemented JAX
core supplies initial state, asynchronous reset masks, explicit random
addresses, and a time-major scan. It tests fixed-address equations and address
sampling, and the shared learner replays collection-time addresses exactly.
The released POMDP path also clamps the recurrent matrix to `[-100, 100]`
after each update, while the standalone and POPGym paths do not. The
registered scientific baseline uses `paper_uniform`, including the POMDP
clamp. The released row-zero, unclamped behavior remains a separately named
compatibility check.

POBAX reports recurrent PPO, lambda discrepancy with recurrent PPO, and
Transformer-XL with PPO. Its repository enables GRU-style gates in the
Transformer implementation by default. The paper must call a result
Transformer-XL unless the exact gating configuration is recorded and verified
as the GTrXL architecture. Published author results remain compatibility
references unless ArcMind reproduces their complete tuning protocol.

Contextual reference methods, when their code and task protocol can be
reproduced, are
[memory traces](https://proceedings.mlr.press/v267/eberhard25a.html) and
[Recall to Imagine](https://arxiv.org/abs/2403.04253). They should not be
presented as like-for-like backbone baselines if they change the learning
algorithm or add a world model.

The detailed inclusion rationale, implementation status, and compatibility
caveats are maintained in
[the benchmark and baseline audit](literature_and_baselines.md).

Primary metrics:

- episode return or task success, normalized per environment;
- interquartile mean (IQM), median, mean, and optimality-gap profiles across
  tasks;
- sample efficiency as area under the learning curve;
- paired bootstrap 95% confidence intervals;
- parameter count, peak accelerator memory, environment steps per second, and
  single-environment streaming latency.

Classify every run before execution:

- A smoke run checks imports, shapes, resets, compilation, gradients, and
  artifact creation. Quick or 131,072-step cells are smoke evidence only.
- A pilot run checks learnability and freezes architecture and tuning choices.
  It may use shortened budgets and three to five development seeds, but it is
  not paper evidence.
- A development tuning run selects separately within each architecture from a
  frozen, equal-cardinality candidate matrix. It uses one task at its full
  published budget, the published task-specific tuning-seed count, and mean
  seed learning-curve AUC on the shared complete-case suffix. It cannot rank
  architectures, select a checkpoint, or serve as paper performance evidence.
- A registered final run uses frozen choices, paired seed manifests, and the
  published interaction budget for every method in a task cell. A primary
  comparison must use schema v4 and bind every learner to the exact winner of
  one verified tuning aggregate on the same task. Final seeds must be
  disjoint from tuning seeds. Separately labeled schema-v2 upper references
  do not require a tuning binding.

Primary and upper-reference results require a cross-matrix link artifact
before comparison. The linker validates the complete raw checksum inventories,
requires the same ordered seed list and exact alias-to-primary mapping, and
checks learner, evaluation, source, dependency, and registered-budget
contracts. A shared seed count without this link is insufficient evidence.

The published POBAX budgets are 1 million steps for T-Maze-10, 5 million for
RockSample11, 10 million for Battleship-10, 50 million each for Walker-V and
HalfCheetah-V, and 10 million for Navix-01. The paper uses 5 tuning seeds for
T-Maze, RockSample, masked MuJoCo, and Navix, 10 tuning seeds for Battleship,
then 30 final seeds for every reported environment. ArcMind targets 30 paired
final seeds per main method-task cell. A preliminary table may use 10 paired
seeds, but it must not be described as matching the published POBAX final
protocol.

Use identical environment seeds and training budgets across models. Give every
architecture the same number of tuning trials, even when its search space
differs. The reporting procedure follows the uncertainty-aware recommendations
in
[Deep Reinforcement Learning at the Edge of the Statistical
Precipice](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).

The tuning aggregate is mechanically separated from registered-final
aggregation. It requires schema v3, explicit immutable candidate IDs grouped
by implementation model, the shared exact-step comparison profile, one
complete Cartesian candidate matrix, checksum and completion indexes, frozen
environment semantics, and frozen parameter matching. Every model family has
the same candidate count and the exact same normalized learner configuration
set. All candidates share `num_envs`, `rollout_steps`, `update_epochs`, and
`num_minibatches`; only learning rate, GAE lambda, entropy coefficient, and
learning-rate annealing may vary. The common curve
starts at the latest first finite return across all candidate and seed cells
and must retain at least two observations. Per-seed AUC uses trapezoidal
integration over environment steps with no extrapolation. Dividing by the
common interval width yields `auc_mean_return`, and candidates are ranked by
its seed mean only within their model family. The selected configurations must
be frozen in a new manifest and rerun on disjoint registered-final seeds.
Every tuning completion row must identify its immutable cell log and SHA256,
and the matrix checksum inventory must cover all of those logs.

Schema v4 closes the selection-to-final boundary. Its registration names the
raw tuning matrix and canonical tuning aggregate by repository-relative path,
binds both source hashes and the aggregate file SHA256, and copies only the
winning candidate identity and complete learner for each model family. The
loader rebuilds the tuning aggregate, requires exact canonical bytes, checks
every declared winner and learner, and rejects any overlap between tuning and
final seeds. The same binding is frozen in the final matrix manifest and
revalidated during registered aggregation.

Canonical cell logs identify successful executions only. Output from a
failed child is retained under a unique failed-attempt path, together with any
partial artifact, and cannot be reused by a later successful completion or
resume.

### Track B: state-based robot imitation (primary application)

Use the low-dimensional datasets and tasks from
[RoboMimic](https://proceedings.mlr.press/v164/mandlekar22a.html), beginning
with Lift, Can, Square, Transport, and Tool Hang where available. Evaluate
proficient-human and multi-human data separately; demonstration quality is an
experimental factor, not a nuisance to average away.

Mandatory baselines:

- BC-MLP;
- BC-RNN with GRU and LSTM;
- causal Transformer BC;
- S4D/selective-SSM BC;
- ArcMind and its registered ablations;
- BC-GMM for multimodal action distributions;
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) as an application-level
  reference, with its inference cost reported.

Checkpoint selection must use a held-out demonstration validation set and a
predeclared selection metric. Test-environment rollouts must not choose the
epoch. Report the selected checkpoint over 100 rollouts for each of 5 training
seeds (500 rollouts per method-task cell), with a binomial interval and a
hierarchical bootstrap across seeds. The original RoboMimic best-checkpoint
protocol may be reported in a separate compatibility table, clearly labeled.

ArcMind's current state-to-action objective is behavior cloning. It is not
directly comparable to return-conditioned sequence models. If return-to-go,
reward, and prior-action conditioning are later added, compare against
[Decision Transformer](https://proceedings.neurips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html),
[Decision S4](https://arxiv.org/abs/2306.05167),
[Decision Mamba](https://proceedings.neurips.cc/paper_files/paper/2024/hash/850e8063d902e0825d3c5504d183bafe-Abstract-Conference.html),
and recent efficient sequence-policy baselines such as
[Decision HiFormer](https://proceedings.neurips.cc/paper_files/paper/2025/hash/03e81b3bdb6c0a61093defd319d12203-Abstract-Conference.html).

### Track C: multivariate sensor classification (diagnostic)

UCI HAR and Opportunity remain smoke tests and continuity checks, not headline
benchmarks. The paper-grade evaluation uses the official splits from the
[UEA archive](https://timeseriesclassification.com/) and the larger
[MONSTER](https://arxiv.org/abs/2502.15122) repository. MONSTER is important
because the median UEA/UCR datasets are too small to characterize scalable
sequence models reliably.

Mandatory baselines:

- linear/logistic and 1D-CNN references;
- InceptionTime;
- MiniROCKET, MultiROCKET, HYDRA, and HYDRA-MultiROCKET;
- HIVE-COTE 2.0 where computationally feasible;
- S4D, Mamba, and the selective SSM used by ArcMind;
- MS4 and MS4N, subject to public-code reproduction;
- recent peer-reviewed multivariate classifiers such as TimeMIL and FIC-TSC.

Primary sources include
[HIVE-COTE 2.0](https://link.springer.com/article/10.1007/s10994-021-06057-9),
[MultiROCKET](https://arxiv.org/abs/2102.00457),
[HYDRA](https://arxiv.org/abs/2203.13652),
[InceptionTime](https://arxiv.org/abs/1909.04939),
[TimeMIL](https://proceedings.mlr.press/v235/chen24af.html),
[FIC-TSC](https://proceedings.mlr.press/v267/chen25cq.html), and the 2026
preprint on [MS4/MS4N](https://arxiv.org/abs/2605.27406).

Use official train/test splits. Hyperparameters and early stopping are selected
only within the training partition, using nested validation or cross-validation.
For neural methods, run 10 initialization seeds per dataset where feasible.
Report per-dataset metrics, average rank, wins/ties/losses, and paired
Wilcoxon tests with Holm correction. Critical-difference diagrams are
descriptive complements, not substitutes for effect sizes and uncertainty.
Measure training time and inference throughput because the accuracy-only
leader, HIVE-COTE 2.0, is not intended as the efficiency reference.

## 4. Efficiency protocol

All latency comparisons use the same machine, framework version, precision,
batch size, and compilation setting. Publish the complete hardware and software
manifest.

For each model:

1. warm up for at least 100 calls;
2. time at least 1,000 synchronized calls;
3. report median, p90, and p99 latency;
4. report sensor-step and decision-step latency separately;
5. report resident recurrent state and peak allocated memory;
6. report parameters and multiply-accumulate estimates; and
7. verify numerical agreement between batch and streaming execution.

The asymptotic statement allowed for ArcMind is bounded recurrent and episodic
state at inference for fixed configuration. No hardware-speed claim is allowed
without the measurements above.

## 5. Registered ablations

The minimum ablation matrix is:

| Variant | Fast SSM | Exact recall | Age encoding | Learned gate |
|---|---:|---:|---:|---:|
| ArcMind | yes | yes | yes | yes |
| unordered memory | yes | yes | no | yes |
| no memory | yes | no | n/a | yes |
| no SSM | no | yes | yes | n/a |
| no gate | yes | yes | yes | no |
| recurrent baseline | yes | no | n/a | n/a |
| attention baseline | no | full causal | positional | n/a |

Sweep the decision stride, memory window, memory compression ratio, and total
parameter budget. At least one experiment must match parameter counts and one
must match measured decision latency; otherwise an accuracy gain cannot be
attributed to the hybrid design.

## 6. Claims that are currently prohibited

Until the registered results exist, do not claim:

- state of the art on robotics, IoT, time-series classification, or RL;
- smoother or more physically plausible controls than Transformers;
- precise landmark, obstacle, or temporal recall;
- deployment suitability for a named MCU, NPU, or robot computer;
- Mamba or Mamba-2 compatibility/performance;
- superiority to VLA systems, which solve a materially different problem; or
- parameter counts copied from an earlier checkpoint after the architecture
  changes.

Existing checkpoints and result files predate the causal memory correction and
must not enter the paper. Every reported result must be regenerated from a
versioned configuration, seed manifest, immutable dataset identifier, and
machine-readable output artifact. Derived aggregates and primary-to-upper
links must live outside immutable raw-matrix roots. Their writers reject
in-root paths before reading or writing any result.

## 7. Release gates

Before a paper repository is created:

- the core package passes unit, causality, gradient, and batch/streaming parity
  tests;
- benchmark configurations and seed lists are committed before full runs;
- all baselines run through the same evaluator;
- raw per-seed results and aggregation scripts reproduce every table and plot;
- negative and null results are retained;
- the package API and model description match the implemented architecture;
- at least one independent environment reproduces installation and a small
  benchmark from a clean checkout; and
- the paper distinguishes peer-reviewed results from preprints and submitted
  work.
