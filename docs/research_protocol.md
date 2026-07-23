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

Use [POBAX](https://openreview.net/forum?id=HUTCbYOW5E) as the primary suite.
POBAX was designed around two requirements ArcMind needs: coverage of distinct
forms of partial observability and a measurable gap between memoryless and
more-informed agents. Use representative tasks from localization/mapping,
visual or state control, games, and information omission. Prefer
low-dimensional observations for the first paper; a vision encoder would
confound the backbone comparison. The initial task set is Simple Chain,
T-Maze-10, RockSample 11x11, Battleship-10, the position-only and
velocity-only masked MuJoCo tasks, and DMLab MiniGrid Maze-01. Pixel control
and no-inventory Crafter are held out for a later encoder study.

Use [POPGym](https://arxiv.org/abs/2303.01859) as the controlled memory suite
and [Memory Maze](https://arxiv.org/abs/2210.13383) only as a stretch
evaluation after the low-dimensional claim is established.

Mandatory controls and baselines:

- memoryless MLP and fixed frame stack;
- Elman RNN, GRU, and LSTM;
- TCN;
- memory traces, LRU, S4D, S5/S5RL, and a modern stable diagonal SSM;
- the repository's selective SSM without exact recall;
- causal Transformer or GTrXL at matched parameter count;
- ArcMind without age encoding, without episodic memory, without gating, and
  without the SSM path;
- a fully observable policy or privileged-state upper reference where the
  environment supplies one.

POBAX is JAX-native. All accuracy comparisons must use one PPO
implementation, optimizer, rollout collector, and update schedule. The ArcMind
recurrence therefore requires a JAX benchmark implementation with numerical
parity tests against this PyTorch package; comparing separate PyTorch and JAX
learners would introduce an unacceptable framework confound. Include POBAX's
native recurrent PPO, memoryless PPO, GTrXL, and lambda-discrepancy results
under the same tuning budget.

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

Run at least 10 paired training seeds per method-task cell for inexpensive
POPGym tasks and at least 5 for the more expensive POBAX tasks. Use identical
environment seeds and training budgets across models. The reporting procedure
follows the uncertainty-aware recommendations in
[Deep Reinforcement Learning at the Edge of the Statistical
Precipice](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).

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
machine-readable output artifact.

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
