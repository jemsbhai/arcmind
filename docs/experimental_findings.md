# ArcMind experimental findings

Status: living findings ledger, initialized 2026-07-23.

This document is the single narrative record for experimental observations,
failed hypotheses, null results, and registered findings. Raw artifacts remain
the source of truth. A sentence in this file is not sufficient evidence for a
paper claim unless its entry is marked `registered evidence` and points to the
complete raw result set, frozen protocol, aggregation output, and reference
verification record.

## Evidence classes

Every finding must have exactly one of the following classes.

| Class | Permitted purpose | Permitted paper use |
|---|---|---|
| `development smoke` | Verify installation, execution, shapes, gradients, reset behavior, action masking, or basic learnability. | Methods or implementation validation only. Never an accuracy comparison. |
| `diagnostic evidence` | Test a mechanism with a controlled local task before registration. | Motivation, failure analysis, or an explicitly labeled diagnostic result. Never a headline result. |
| `registered evidence` | Evaluate a frozen hypothesis with frozen configurations, seeds, budgets, selection rules, and aggregation. | Main tables, figures, abstract, and conclusion, subject to uncertainty and multiplicity checks. |
| `null or negative result` | Preserve a failed prediction, no detectable effect, regression, instability, incomplete evaluation, or contradictory outcome. | Must be reported wherever its omission would distort the conclusion. It may also carry one of the first three provenance classes. |

Evidence classes are immutable. A smoke artifact cannot become registered
evidence after inspection. The experiment must be rerun under the registered
protocol. A diagnostic result can motivate a registered hypothesis, but it
cannot select a test seed, test checkpoint, or favorable reporting metric.

## Unit of record

An experimental cell is one benchmark task, model, training seed,
hyperparameter configuration, code revision, and interaction or sample budget.
Each cell must write one independent machine-readable artifact. An aggregate
must list every included cell and every excluded cell, with a reason for each
exclusion.

Each cell must record:

1. a stable experiment identifier and finding identifier;
2. evidence class and whether the run was planned, exploratory, or a rerun;
3. benchmark, environment, dataset split, task version, and immutable source
   revision;
4. model name, complete model configuration, parameter count, effective
   parameter count, recurrent-state size, and comparison target;
5. optimizer, learner, rollout, batch, precision, compilation, and stopping
   configuration;
6. training seed, environment seed, data seed, initialization seed, and the
   frozen seed-manifest identifier;
7. code commit, dirty-tree flag, patch or diff digest when dirty, package
   version, lockfile digest, and dependency source commits;
8. hardware, accelerator backend, operating system, language and framework
   versions, wall time, environment steps, accelerator time, and estimated
   external cost;
9. model-selection rule, selected checkpoint or final-policy marker, validation
   observations, and the count of test evaluations;
10. raw metrics, episode count, missing values, numerical warnings, crashes,
    retries, and any manual intervention;
11. raw artifact path and checksum, log path, checkpoint path when retained,
    and aggregation artifact path;
12. hypothesis stated before outcome inspection, result summary, uncertainty,
    interpretation, alternative explanations, and disposition; and
13. reference-verification records for every external method, benchmark,
    metric, or protocol claim used to interpret the cell.

Aggregate records must additionally include paired seed coverage, missing-cell
handling, bootstrap or interval method, resample count, multiplicity
correction, task normalization, and the exact script invocation.

## Reference verification contract

No reference may enter a finding or the paper from memory alone. Before using
an external claim, search for and inspect a primary source. Record:

- canonical title, complete author list, venue, year, and publication status;
- DOI, proceedings page, publisher page, or canonical preprint URL;
- date accessed and search query or discovery path;
- exact source location supporting the claim, such as section, equation,
  appendix, table, or repository file;
- whether the cited result is peer reviewed, a preprint, author code, or a
  third-party reproduction;
- benchmark observations, training budget, evaluation protocol, seeds, metric
  definition, and any privileged information relevant to comparability; and
- official code repository URL, release or commit, license, and local
  modifications when code is used.

Search snippets, abstracts alone, secondary surveys, and repository README
claims are discovery aids, not sufficient verification for an equation or
numerical comparison. Published numbers must not be copied into a comparison
table until their task version, observation space, learner, budget, and
evaluation procedure have been checked against the ArcMind cell. If those
conditions differ, label the number contextual and non-comparable.

The living baseline audit is
[literature_and_baselines.md](literature_and_baselines.md). That document
guides source selection, while this ledger records the verification associated
with each actual experiment.

## Current findings

### F-ENG-001: Cross-framework recurrence agreement

- Class: `development smoke`
- Status: observed, not independently archived as a registered cell
- Scope: deterministic PyTorch and JAX ArcMind streaming fixture
- Observation: repository validation reports a maximum absolute discrepancy of
  `5.561113e-05` across 84 compared values. The fixture includes ring-buffer
  wraparound and a decision stride of two.
- Interpretation: the JAX benchmark adapter is sufficiently close to the
  PyTorch implementation for development. This does not establish training
  equivalence, benchmark quality, or task performance.
- Required follow-up: preserve the fixture, command, environment manifest, and
  output as a checksummed release artifact before package publication.
- Protocol:
  [benchmarks/pobax/README.md](../benchmarks/pobax/README.md)

### F-ENG-002: Environment discount and runtime identity must be frozen

- Class: `development smoke`
- Status: protocol defect found and corrected before any registered run
- Date: 2026-07-23
- Source audit: the pinned POBAX PPO replaces the command-line discount with
  the environment discount after construction
  ([learner source](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/algos/ppo.py#L169-L179)).
  Battleship defines that discount as `1.0`
  ([environment source](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/battleship.py#L139-L166)).
- Defect: the shared runner previously retained PPO's `0.99` default for every
  task. This matched most current tasks but disagreed with Battleship. It also
  recorded a dependency-lock hash without freezing the actual Python, JAX,
  JAXlib, package, backend, and accelerator identities.
- Correction:
  - derive gamma from the constructed environment and include it in the
    hashed PPO configuration;
  - derive the evaluation horizon from
    `environment_params.max_steps_in_episode`, assert it against the audited
    task contract, and reuse it for positional encoding and evaluation;
  - call PPO configuration validation before a matrix description can be
    frozen; and
  - include a stable installed-runtime contract in the configuration,
    manifest provenance, cell provenance, resume checks, and registered
    aggregator validation.
- Verification: configuration-only execution resolves T-Maze to horizon
  `1000` and gamma `0.99`, and Battleship to horizon `1000` and gamma `1.0`.
- Interpretation: dependency source commits and a lockfile checksum identify
  intent, while the installed-runtime fingerprint identifies execution.
  Both are required to prevent silent mixing of incompatible cells.
- Evidence restriction: no registered cell existed before the correction.
  Any future artifact whose configuration omits the environment-derived
  discount or runtime contract is ineligible for aggregation.

### F-ENG-003: Registered labels require complete seed and matrix roles

- Class: `development smoke`
- Status: protocol defect found and corrected before any registered run
- Date: 2026-07-23
- Defect: the initial artifact schema allowed a clean full-budget cell to use
  `registered_final_complete` even when its matrix contained fewer than the
  published 30 final seeds. It also assumed every aggregate contained ArcMind,
  which made separately labeled full-observation upper references impossible
  to aggregate under the registered path.
- Correction:
  - a `registered_final` registration now requires exactly 30 unique paired
    seeds, and the registered aggregator independently enforces the same
    cardinality;
  - every matrix declares either `primary_comparison` or `upper_reference`;
  - primary matrices must include ArcMind and retain paired differences
    against it;
  - upper-reference matrices contain only the parameter-matched memoryless
    policy, emit no misleading ArcMind pair, and accept only implemented
    upper-reference environment adapters; and
  - Walker-F and HalfCheetah-F inherit the 50 million interaction budget of
    their corresponding primary tasks.
- Interpretation: budget compliance alone is not final-evidence compliance.
  Seed cardinality and the information role of each environment are part of
  the evidence label and must fail closed.
- Evidence restriction: no registered result existed before this gate.

### F-ENG-004: A valid hash does not prove a completed experiment

- Class: `development smoke`
- Status: artifact-integrity defect found and corrected before any registered
  run
- Date: 2026-07-23
- Defect: the initial registered aggregator verified that a configuration
  matched its hash but did not interpret the configuration. A crafted or
  partial artifact could therefore retain a self-consistent hash while using a
  development evidence tier, mismatched cell identity or source provenance,
  or a learning curve that stopped before the registered interaction budget.
- Correction: aggregation now independently requires:
  - `registered_final` inside the frozen cell configuration;
  - exact agreement among configuration, artifact, and manifest environment,
    model, seed, source commits, dependency lock, and runtime contract;
  - the published task interaction budget in PPO configuration, top-level
    actual steps, and the last learning-curve point;
  - exact agreement between top-level and frozen PPO configurations; and
  - consistent evaluation episodes, horizon, scan steps, vector-worker count,
    and total evaluation transitions.
- Verification: focused tests reject a pilot-tier frozen configuration and a
  history ending one transition before the T-Maze registered budget.
- Interpretation: canonical hashing establishes immutability, not scientific
  completeness. Registered aggregation must validate both bytes and meaning.
- Evidence restriction: no registered result existed before this gate.

### F-DIAG-001: Exact recall helps the diagnostic learn, but current long-lag recall is weak

- Class: `diagnostic evidence` and `null or negative result`
- Status: exploratory, one seed, dirty working tree, not for a paper claim
- Task: delayed sensor recall, seed `1103`, 1,024 training examples, 512 test
  examples, parameter-matched models
- Observation after 10 epochs: ArcMind reached `0.4982` test accuracy, while
  the SSM-only ablation reached `0.2550`. ArcMind short-lag accuracy was
  `0.6083`, but long-lag accuracy was `0.2343`, close to the four-class chance
  level and the SSM-only value of `0.2475`.
- Interpretation: under this small diagnostic run, the exact-recall path
  contributed to learnability, but the result gives no evidence that ArcMind
  recovered long-lag associations. The long-lag outcome contradicts a broad
  exact-recall claim and is retained as a negative result.
- Alternative explanations: insufficient training data, optimization,
  compression, memory addressing, write timing, or an exact-recall window
  shorter than the evaluated lag.
- Raw artifacts:
  `benchmark_results/pilot-1/arcmind_seed-1103.json` and
  `benchmark_results/pilot-1/arcmind_ssm_only_seed-1103.json`
- Disposition: diagnose lag coverage and addressing before freezing the
  registered diagnostic.

### F-DIAG-002: The current diagnostic favors full causal attention

- Class: `diagnostic evidence` and `null or negative result`
- Status: exploratory, one seed, dirty working tree, not for a paper claim
- Task: same cell family as F-DIAG-001
- Observation: the causal Transformer reached `0.9205` test accuracy and
  `0.9365` long-lag accuracy after 10 epochs. ArcMind reached `0.4982` overall
  and `0.2343` long-lag accuracy. The Transformer had `44,044` parameters,
  `1.0723` times the ArcMind target, and therefore remained within the current
  ten percent width-matching tolerance.
- Interpretation: the diagnostic and training budget strongly favor full
  causal attention. ArcMind does not currently support an accuracy advantage
  on this task. Future comparisons must report attention's compute and memory
  costs rather than using efficiency as an unmeasured explanation.
- Raw artifacts:
  `benchmark_results/pilot-1/causal_transformer_seed-1103.json` and
  `benchmark_results/pilot-1/arcmind_seed-1103.json`
- Disposition: retain as a falsification pressure test. Do not weaken or remove
  this baseline.

### F-DIAG-003: Relative age encoding has no established benefit yet

- Class: `diagnostic evidence` and `null or negative result`
- Status: exploratory, one seed, dirty working tree, not for a paper claim
- Task: delayed sensor recall, seed `1103`, 20 epochs, 1,024 training examples,
  512 test examples
- Observation: ordered ArcMind reached `0.7321` overall accuracy and the
  unordered variant reached `0.7209`. Ordered short-lag accuracy was `0.9336`
  versus `0.9166`, while ordered long-lag accuracy was `0.2492` versus
  `0.2519`.
- Interpretation: the one-seed overall difference is too small and too
  uncertain to establish an ordering benefit. The long-lag metric does not
  favor age encoding. Any claim that relative age improves recall remains
  unsupported.
- Raw artifacts:
  `benchmark_results/pilot-20/arcmind_seed-1103.json` and
  `benchmark_results/pilot-20/arcmind_unordered_seed-1103.json`
- Disposition: keep the registered ordering ablation, add paired seeds, and
  inspect effects by lag and overwrite count.

### F-SMOKE-001: The shared PPO path executes diverse policy and action families

- Class: `development smoke`
- Status: infrastructure validation only
- Observation: development artifacts show completed execution for recurrent,
  convolutional, structured-state, attention, and ArcMind cores. Discrete
  RockSample exercised action masking. HalfCheetah-P exercised a diagonal
  Gaussian policy and, after raising evaluation to 1,024 transitions,
  completed 32 evaluation episodes.
- Interpretation: these runs validate plumbing across model families and
  discrete or continuous action spaces. Their returns are not comparative
  evidence.
- Representative artifacts:
  `benchmark_results/pobax/smoke/rocksample_11_11_gru_seed1103.json` and
  `benchmark_results/halfcheetah_p_gru_eval_horizon_smoke.json`
- Disposition: convert infrastructure requirements into regression tests and
  rerun them from the release candidate package.

### F-SMOKE-002: Simple Chain cannot discriminate policy quality

- Class: `development smoke`
- Status: expected task limitation
- Observation: multiple policy cores reached return `1` on the single-action
  Simple Chain smoke task.
- Interpretation: this environment can validate execution and reset handling,
  but it cannot support architecture ranking or learnability claims.
- Disposition: prohibit Simple Chain results from comparative tables.

### F-SMOKE-003: The short T-Maze S5RL evaluation produced no completed episode

- Class: `development smoke` and `null or negative result`
- Status: incomplete deterministic evaluation
- Task: T-Maze-10, S5RL, seed `1103`, 8,192 environment steps
- Observation: training completed 21 episodes, but deterministic evaluation
  completed zero episodes. Mean and median evaluation return are therefore
  null. A memory-trace smoke cell at the same interaction budget completed
  evaluation episodes, but this is not a controlled performance comparison.
- Interpretation: the S5RL cell validates execution but not policy quality.
  Zero completed episodes must remain explicit rather than being converted to
  zero return or omitted.
- Raw artifacts:
  `benchmark_results/pobax/smoke/tmaze_10_s5rl_seed1103.json` and
  `benchmark_results/pobax/smoke/tmaze_10_memory_trace_seed1103.json`
- Disposition: add a registered rule for episode completion, horizon handling,
  and missing evaluation metrics before model comparisons.

### F-SMOKE-004: Required memory controls execute through the shared learner

- Class: `development smoke`
- Status: infrastructure validation only
- Date: 2026-07-23
- Task: T-Maze-10, seed `1103`, 8,192 environment transitions, 32 vector
  environments, and exactly one retained evaluation episode per environment
- Controls:
  - FFM with the published POPGym structure of 32 traces and four complex
    contexts;
  - the pinned POPGym sinusoidal positional MLP with a reset-aware episode
    counter; and
  - SHM in `paper_uniform` mode with 128 address rows, 16 by 16 memory, and
    exact replay of collection-time addresses during PPO recomputation.
- Observation:

  | Control | Parameters | ArcMind target | Ratio | Training episodes | Evaluation episodes | Mean return |
  |---|---:|---:|---:|---:|---:|---:|
  | FFM | 28,577 | 28,717 | 0.9951 | 30 | 32 | 0.0000 |
  | Positional MLP | 28,526 | 28,717 | 0.9933 | 31 | 32 | 0.0000 |
  | SHM | 28,727 | 28,717 | 1.0003 | 143 | 32 | 2.0781 |
- Replay check: the SHM unit test reconstructs collection logits with stored
  addresses and obtains zero approximate KL before an update. The real smoke
  completed with finite loss and final approximate KL `0.00092`.
- Test coverage: the unified POBAX suite passed all 148 tests on CUDA and CPU.
  These include source-equation fixtures, asynchronous reset, step-scan
  agreement, JIT, gradients, parameter matching, SHM address distribution,
  source-compatible row-zero addressing, and exact address replay.
- Interpretation: all three controls now satisfy the common learner interface,
  parameter tolerance, and fixed evaluation episode-count contract. The
  different smoke returns are not comparative evidence because this run is
  short, single-seed, and executed during development.
- Raw artifacts and SHA256:
  - `benchmark_results/pobax/smoke-ffm-20260723.json`,
    `93320e4d9acf543bb98d08686d0352830e16dc06a054dab5bfec47ffd0548d1e`;
  - `benchmark_results/pobax/smoke-positional-20260723.json`,
    `53b5d6f55ea5060beb93092ab3101344649c98f0672f8e0ee50157ec486ded8b`;
    and
  - `benchmark_results/pobax/smoke-shm-replay-20260723.json`,
    `5bfb800c4178caf265833329f21461d2cd6af969978e6f51c2798a1b0af85068`.
- Disposition: retain FFM, positional MLP, and SHM `paper_uniform` in the
  pilot matrix. Keep `v1_1_popgym_compat` as a source-compatibility check, not
  the scientific SHM baseline.

### F-SMOKE-005: Walker partial and full observation adapters execute

- Class: `development smoke`
- Status: infrastructure validation only
- Date: 2026-07-23
- Task: Walker2d, seed `1103`, 8,192 environment transitions, and exactly one
  retained evaluation episode for each of 32 vector environments
- Observation:
  - `Walker-V-v0` exposed nine observation values and a six-value continuous
    action, giving the common ArcMind policy input width 17;
  - `Walker-F-v0` exposed all 17 observation values and the same action,
    giving the memoryless policy input width 25;
  - the full-observation memoryless policy was matched to the partial-task
    ArcMind target, not to a larger full-input ArcMind model. It used 28,965
    parameters against the 29,013 target, a ratio of `0.9983`;
  - both cells completed finite PPO updates and exactly 32 evaluation
    episodes; and
  - the short-run mean returns were `276.49` for partial-observation ArcMind
    and `243.01` for the full-observation memoryless reference.
- Interpretation: the result validates observation slicing, continuous action
  handling, exact evaluation counts, and the primary-task parameter target.
  The return ordering reinforces why a full-observation trained policy is not
  a guaranteed empirical ceiling. It is not performance evidence because the
  runs are short, single-seed, and use different policy classes by design.
- Raw artifacts and SHA256:
  - `benchmark_results/pobax/smoke-walker-v-arcmind-20260723.json`,
    `025a0a7c3c59ced88d8ee7e770a578f799e6fb4519190e51711422fba256f3d8`;
    and
  - `benchmark_results/pobax/smoke-walker-f-memoryless-20260723.json`,
    `1c871aabe995f4a2c32bce5c7ecd230c3c40b69c75d52f352c0518eec93172b8`.
- Disposition: include Walker-V in the pilot task set. Report Walker-F only in
  a separately labeled upper-reference table.

### F-SMOKE-006: The frozen matrix launcher completes and resumes cleanly

- Class: `development smoke`
- Status: infrastructure validation only
- Date: 2026-07-23
- Source commit:
  `de799c334072e270bce9102a04765e75f5943020`, clean detached worktree
- Task: T-Maze-10, seed `1103`, ten policy cores, 8,192 environment
  transitions per cell, and exactly one retained evaluation episode for each
  of 32 vector environments
- First attempt: a worktree created by Windows Git embedded a Windows absolute
  Git-directory pointer that WSL Git could not resolve. The launcher stopped
  during provenance discovery before creating an output directory or running
  a cell. Recreating the same detached worktree with WSL Git resolved the
  platform-path mismatch.
- Completed matrix:

  | Model | Parameters | Ratio | Training episodes | Evaluation episodes | Mean return |
  |---|---:|---:|---:|---:|---:|
  | ArcMind | 28,717 | 1.0000 | 10 | 32 | 0.0000 |
  | ArcMind SSM only | 28,717 | 1.0000 | 13 | 32 | 0.0000 |
  | FFM | 28,577 | 0.9951 | 30 | 32 | 0.0000 |
  | GRU | 28,893 | 1.0061 | 30 | 32 | 0.0000 |
  | LSTM | 28,840 | 1.0043 | 20 | 32 | 0.0000 |
  | Memoryless MLP | 28,663 | 0.9981 | 29 | 32 | 2.0781 |
  | Positional MLP | 28,526 | 0.9933 | 31 | 32 | 0.0000 |
  | S5RL | 28,617 | 0.9965 | 21 | 32 | 0.0000 |
  | SHM | 28,727 | 1.0003 | 143 | 32 | 2.0781 |
  | Transformer-XL | 28,869 | 1.0053 | 8 | 32 | 0.0000 |
- Execution: the complete launcher used 240.5 wall-clock seconds, while the
  ten recorded training intervals summed to 107.2 local GPU seconds. A second
  invocation resumed all ten cells without retraining in 28.6 seconds.
- Integrity:
  - completion status was `complete` with 10 planned and 10 completed cells;
  - canonical matrix-manifest SHA256 was
    `07845dfff52ba337bcc7027301dd8a9ace2fd57693f7a1f93e22de3069f6257a`;
  - frozen-manifest file SHA256 was
    `1a0cd352663bd98d395ebb92b5d7324c44f4647e0a90cab903d953c4b9039f77`;
  - checksum-manifest SHA256 was
    `22a71ce3ed04a50dd311f247c41f966c2251800de382e7bf1673f0a588f0b1a3`;
    and
  - `sha256sum -c` verified the registration, frozen manifest, completion
    index, and all ten cell artifacts before and after resume.
- Runtime notes: optional Warp imports were unavailable, and XLA emitted
  nonfatal autotuning fallback warnings during ArcMind compilation. Every
  selected cell still completed with finite training and evaluation metrics.
- Interpretation: this validates clean provenance discovery, frozen Cartesian
  expansion, fresh-process isolation, complete runtime fingerprints,
  collision-safe artifact creation, checksum generation, and idempotent
  resume. The return differences are not comparative evidence because this is
  a one-seed development smoke at less than one percent of the registered
  T-Maze interaction budget.
- Raw artifact directory:
  `benchmark_results/pobax/matrix-smoke-controls-v1-de799c3`
- Disposition: use the launcher for a predeclared multi-seed T-Maze pilot.
  Create WSL worktrees with WSL Git whenever WSL executes the experiment.

### F-PILOT-001: T-Maze multi-seed viability screen

- Class: `development smoke`
- Status: planned and frozen before outcome inspection
- Date planned: 2026-07-23
- Question: do the ten required low-dimensional controls train stably and
  complete independent evaluation under a shared configuration long enough to
  justify full-budget registered execution?
- Prediction: ArcMind, its SSM-only ablation, and every required source-audited
  control will complete all three paired seeds with finite losses and exactly
  128 evaluation episodes per seed. No directional performance claim is
  preregistered for this development screen.
- Cells: T-Maze-10 crossed with memoryless MLP, positional MLP, FFM, SHM
  `paper_uniform`, GRU, LSTM, S5RL, Transformer-XL, ArcMind SSM-only, and
  ArcMind on seeds `1103`, `2207`, and `3301`, for 30 cells total.
- Training contract:
  - 250,000 exact environment transitions per cell;
  - eight vector environments and 125 rollout steps, giving exactly 1,000
    transitions per update and 250 updates;
  - four PPO update epochs, four environment-axis minibatches, learning rate
    `0.00025`, environment gamma `0.99`, no checkpoint selection, and the final
    policy only; and
  - parameter matching to the ArcMind cell within ten percent, with the
    source-defined FFM and SHM memory structures held fixed.
- Evaluation contract: 16 deterministic episodes from each of eight fresh
  vector environments, or 128 episodes per model and seed, with the full
  source-defined 1,000-step horizon.
- Frozen registration:
  `benchmarks/pobax/manifests/tmaze_pilot_v1.json`
- Decision rules:
  - any crash, nonfinite required metric, incomplete evaluation count,
    provenance drift, or missing artifact fails that cell;
  - do not delete or replace a low-return method;
  - if a method fails for an implementation reason, repair it and rerun all
    three cells for that method under a new manifest rather than cherry-pick a
    seed;
  - retain every required baseline for final registration regardless of pilot
    rank; and
  - treat all return differences as development evidence that cannot enter a
    paper table, abstract, or conclusion.
- Compute: local GPU only, expected external cost `$0`.

### F-REG-000: No registered performance finding exists

- Class: `registered evidence`
- Status: empty by design
- Observation: no current artifact satisfies the frozen configuration, seed
  manifest, complete baseline, and aggregation requirements in
  [research_protocol.md](research_protocol.md).
- Interpretation: the abstract, results, and conclusion must contain no
  ArcMind performance claim yet.
- Disposition: keep this entry until the first complete registered result set
  passes audit. Replace it only by adding new entries, never by deleting the
  historical absence.

### F-DIAG-004: Clean-release delayed-recall replication

- Class: `diagnostic evidence`
- Status: completed, one seed, not for a paper claim
- Date planned: 2026-07-23, before outcome inspection
- Code: clean `v0.2.0` checkout at commit
  `302e70f373ad308a7c65946d07003df8d7aeda0d`
- Task: delayed sensor recall with 4,096 training examples, 1,024 validation
  examples, 2,048 test examples, 20 epochs, and seed `2207`
- Models: memoryless MLP, GRU, LSTM, causal Transformer, ArcMind SSM-only,
  unordered ArcMind, and ordered ArcMind
- Selection: minimum validation NLL, followed by exactly one test evaluation
- Prediction: full causal attention will remain strongest. ArcMind will
  improve short-lag accuracy relative to its SSM-only ablation, but long-lag
  accuracy will remain near chance. The ordered and unordered variants will
  have no practically clear separation in this single seed.
- Decision rule: if exact recall does not improve short-lag accuracy over the
  SSM-only ablation, inspect addressing and optimization before any registered
  diagnostic. If it improves short-lag but not long-lag accuracy, narrow the
  mechanism claim to bounded recall. Do not use one seed to infer an ordering
  effect.
- Raw artifact directory:
  `benchmark_results/delayed_recall/dev-v0.2.0-seed2207-full`
- Observation: the causal Transformer reached `0.9994` test accuracy,
  ArcMind reached `0.7745`, unordered ArcMind reached `0.7292`, the SSM-only
  ablation reached `0.6481`, GRU reached `0.4821`, LSTM reached `0.4765`, and
  the memoryless MLP reached `0.2518`. ArcMind short-lag accuracy was `0.9991`,
  compared with `0.6925` for SSM-only and `0.9350` for unordered ArcMind.
  ArcMind long-lag accuracy was `0.2519`, compared with `0.5446` for SSM-only,
  `0.2504` for unordered ArcMind, and `0.9996` for the Transformer.
- Prediction check: the prediction was partly supported. Full causal
  attention remained strongest, and ArcMind improved short-lag recall while
  remaining at chance on long lags. Contrary to the implicit expectation that
  adding recall would not damage the recurrent path, SSM-only was `0.2927`
  more accurate than ArcMind on long lags. Ordered ArcMind was `0.0453` more
  accurate overall and `0.0641` more accurate on short lags than unordered
  ArcMind, which is a visible one-seed difference but not evidence of a
  general ordering effect.
- Interpretation: the exact-recall path solves queries whose relevant writes
  remain inside its bounded window, but the current fusion does not preserve
  the SSM-only model's longer-lag behavior. The result points to interference
  in recall or gating, not a broad memory advantage. The Transformer result
  remains a strong falsification pressure test.
- Required follow-up: stratify by exact decision lag and overwrite count,
  verify the latest-value oracle, add a gate diagnostic that reports fast and
  slow path contributions at query time, and test whether recall can abstain
  outside its valid window. Do not expand to registered seeds before this
  interference is understood.
- Provenance: all seven raw records identify the clean release commit and
  `dirty: false`. File hashes are recorded in `checksums.sha256`; the aggregate
  is `summary.json`. The seven cells used 1,032.47 local GPU training seconds.
- Evidence restriction: this run is development evidence and is not eligible
  for a paper performance table.

### F-REF-001: SHM paper and POPGym source use different addressing

- Class: `development smoke` and `null or negative result`
- Status: verified source audit, no performance claim
- Date: 2026-07-23
- Sources: the ICLR 2025 SHM paper and official `v1.1` commit
  `40d73d44936e47a29e2c76a481d93c434b857ea1`
- Observation: the paper specifies uniform sampling from 128 trainable address
  rows. The official standalone and POPGym implementations sample with
  `uniform_(0, 1).long()`, which always produces integer zero. The official
  POMDP implementation instead uses `randint(0, L)` and can reach every row.
  A direct execution of the pinned standalone path over 40,960 positions
  selected only row zero.
- Interpretation: one JAX port cannot simultaneously reproduce the paper's
  intended mechanism and the released POPGym source. Any SHM result must name
  its compatibility mode.
- Disposition: implement `paper_uniform` as the scientific baseline and
  `v1_1_popgym_compat` as a source-compatibility check. Replay collection-time
  addresses during PPO loss recomputation so internal randomness cannot alter
  probability ratios.
- Primary source record:
  [paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/b6446566965fa38e183650728ab70318-Paper-Conference.pdf),
  [standalone source](https://github.com/thaihungle/SHM/blob/40d73d44936e47a29e2c76a481d93c434b857ea1/shm.py#L33-L38),
  [POPGym source](https://github.com/thaihungle/SHM/blob/40d73d44936e47a29e2c76a481d93c434b857ea1/popgym/baselines/ray_models/ray_shm.py#L57-L62),
  and
  [POMDP source](https://github.com/thaihungle/SHM/blob/40d73d44936e47a29e2c76a481d93c434b857ea1/pomdp-baselines/torchkit/shm.py#L39-L44)

### F-DIAG-005: Predeclared fusion-interference controls

- Class: `diagnostic evidence`
- Status: completed as planned
- Date planned: 2026-07-23, before outcome inspection
- Date completed: 2026-07-23
- Task: the F-DIAG-004 delayed-recall task, seed `2207`, 4,096 training
  examples, 1,024 validation examples, 2,048 test examples, 20 epochs, and
  validation-NLL checkpoint selection
- Cells, in fixed order:
  1. instrumented default ArcMind;
  2. SSM-only reference;
  3. fast-start ArcMind, with gate weight zero and gate bias
     `logit(0.05) = -2.944439`;
  4. fast-preserving ArcMind, with training loss
     `CE(fused) + CE(fast)` and the same shared action head; and
  5. observable match-abstention, which exposes only visible write slots whose
     observable key matches the query and forces the slow gate to zero when no
     match is visible.
- Passive measurements: exact-lag accuracy and query count, counterfactual
  shared-head accuracy on the fast representation, mean slow-path gate, and
  mean norm of the slow residual delta. Match-abstention is a task-specific
  control, not a proposed general model.
- Prediction: default ArcMind will replicate the sharp boundary after lag
  eight. Fast-start will reduce early interference but may converge toward the
  same shortcut. The auxiliary fast loss will improve long-lag fast-path
  decodability if joint optimization caused representation specialization.
  Match-abstention will improve fused long-lag accuracy only if irrelevant
  retrieval is a major remaining cause.
- Descriptive decision rules:
  - a change of less than `0.10` long-lag accuracy is not treated as a
    practically clear one-seed effect;
  - fast-start must lose no more than `0.02` short-lag accuracy to count as a
    useful initialization change;
  - counterfactual fast accuracy at least `0.45` with fused accuracy at most
    `0.30` indicates destructive fusion;
  - fast-path accuracy below `0.30` that improves by at least `0.10` with the
    auxiliary loss supports representation specialization; and
  - no one-seed outcome is an inferential or paper-ready claim.
- Raw artifact directory:
  `benchmark_results/delayed_recall/dev-fusion-seed2207`
- Provenance: all five cells ran from clean commit
  `f49fb789d788da90c4ba83c21cdf013581730ae2` on an NVIDIA GeForce RTX
  4090 Laptop GPU. The five training runs used 1,802.10 local GPU seconds in
  total.
- Completed results:

  | Cell | Test accuracy | Short lag | Long lag | Test NLL | Best epoch |
  |---|---:|---:|---:|---:|---:|
  | ArcMind | 0.7745 | 0.9991 | 0.2519 | 0.4215 | 18 |
  | SSM only | 0.6481 | 0.6925 | 0.5446 | 0.7795 | 20 |
  | Fast start | 0.7746 | 0.9988 | 0.2531 | 0.4215 | 18 |
  | Fast auxiliary | 0.7739 | 0.9992 | 0.2499 | 0.4237 | 18 |
  | Match abstention | 0.7744 | 0.9999 | 0.2497 | 0.4177 | 18 |
- Passive diagnostics:
  - default counterfactual fast-path accuracy was `0.2531` at short lags
    and `0.2480` at long lags, while the fused outputs scored `0.9991` and
    `0.2519`;
  - the fast auxiliary changed counterfactual long-lag fast-path accuracy
    from `0.2480` to `0.2587`, a gain of `0.0107`, below the predeclared
    `0.10` threshold;
  - default mean slow gate was `0.7670`. Its exact-lag mean fell from
    `0.8120` at lag 8 to `0.6058` at lag 9;
  - match abstention set the mean slow gate to exactly zero for every lag
    greater than eight, as designed, but changed long-lag accuracy by
    `-0.0022`; and
  - SSM-only long-lag accuracy exceeded the default by `0.2927`, although its
    short-lag accuracy was lower by `0.3066`.
- Decision-rule outcomes:
  - fast-start preserved short-lag accuracy, with a change of `-0.0004`, but
    its long-lag change of `+0.0012` was not practically clear;
  - neither the default nor any intervention met the destructive-fusion
    rule because counterfactual fast-path accuracy remained below `0.30`;
  - the auxiliary result did not meet the representation-specialization
    rule; and
  - match abstention did not support irrelevant retrieval as the main
    remaining explanation.
- Supported conclusion: in this one-seed diagnostic, the default joint model
  did not preserve the SSM-only model's long-lag behavior. Fast-start
  initialization, a shared-head fast-path auxiliary loss, and task-specific
  match abstention did not recover long-lag accuracy under the predeclared
  thresholds.
- Alternative explanations: the task permits a high-accuracy bounded
  shortcut, the shared decoder and both representations may co-adapt, and the
  optimization horizon may favor the sharp lag-eight solution. This study
  does not distinguish these possibilities.
- Decision: stop tuning this synthetic diagnostic. Retain SSM-only as a
  required ablation and prioritize registered partially observed control
  tasks, where returns determine whether the joint architecture is useful.
- Artifact integrity: `checksums.sha256` verifies the five raw JSON files and
  `analysis_summary.json`. The summary SHA256 is
  `b044f2caa4fb7eaac04cffa2b8769d67397e20728442ead67b09e36da533dae9`.
- Evidence restriction: all cells remain diagnostic and cannot enter a paper
  performance table.

## Finding promotion checklist

A finding may be cited as registered evidence only when all answers are yes:

- Was its hypothesis and direction written before registered outcomes were
  inspected?
- Were model configurations, tuning grid, budgets, seed manifests, metrics,
  and exclusion rules frozen?
- Did every method use the same learner and applicable observation contract?
- Are parameter-matched and measured-latency-matched comparisons both present
  where required?
- Are all planned cells present, or are missing cells disclosed with reasons?
- Do raw artifacts identify clean code and immutable dependencies?
- Was test evaluation performed only under the declared selection rule?
- Are effect sizes, paired uncertainty, and per-task results reported?
- Are null, negative, unstable, and failed runs retained?
- Was every external reference and numerical comparison verified through a
  primary source search and recorded?
- Can a clean environment reproduce the aggregate from raw cells?
- Does the proposed sentence state no more than the result supports?

## Experiment log template

Copy this section for each new finding. Do not overwrite an earlier entry when
a rerun changes the conclusion.

```text
### F-[CLASS]-[NUMBER]: [Descriptive finding]

Finding metadata
- Date:
- Owner:
- Evidence class:
- Planned, exploratory, or rerun:
- Status:
- Related hypothesis identifier:
- Related prior finding identifiers:

Question and prediction
- Question:
- Prediction recorded before outcome inspection:
- Smallest effect of scientific interest:
- Decision rule:

Experimental cells
- Benchmark and task:
- Immutable task or dataset revision:
- Observation and action contract:
- Models and ablations:
- Parameter or latency matching rule:
- Training and evaluation budgets:
- Frozen configuration identifier:
- Frozen seed-manifest identifier:
- Selection rule:
- Planned metrics:
- Planned exclusions:

Provenance
- Code commit:
- Dirty tree:
- Patch or diff digest:
- Package version:
- Dependency lock digest:
- External source commits:
- Hardware and software manifest:
- Raw artifact directory:
- Artifact checksums:
- Exact commands:
- Estimated external cost:

Results
- Completed and planned cells:
- Primary estimate and uncertainty:
- Per-task or per-lag results:
- Efficiency measurements:
- Missing values and failed cells:
- Unexpected observations:
- Aggregate artifact:

Interpretation
- Supported conclusion:
- Claims not supported:
- Alternative explanations:
- Null or negative result:
- Sensitivity checks:
- Decision to continue, revise, narrow, or reject:

Reference verification
- Search date:
- Search query or discovery path:
- Primary source URL or DOI:
- Verified bibliographic metadata:
- Publication status:
- Exact supporting source location:
- Official code URL and commit:
- Protocol differences affecting comparability:
- Reviewer initials and verification date:

Paper mapping
- Candidate section, table, or figure:
- Exact claim sentence proposed:
- Required caveat:
- Eligible for paper use: yes or no
```

## Writing constraints

Findings must use direct, specific prose. Report what was measured, the
uncertainty, and the conditions. Avoid promotional adjectives, unsupported
mechanistic explanations, and claims that turn absence of evidence into
evidence of equivalence. Paper-bound prose must contain no em dash or en dash
characters. Before paper export, scan every generated text artifact for
Unicode dash characters and replace them with grammatically appropriate
punctuation or words.
