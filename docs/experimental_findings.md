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
