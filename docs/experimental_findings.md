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
The machine-readable `development_tuning` tier is selection-only diagnostic
evidence. It may choose one frozen candidate by its predeclared
learning-curve metric within each model family, but neither its final returns
nor its AUC values become registered evidence. It cannot rank or remove
architectures. Each selected candidate must be rerun on the disjoint
registered-final seed manifest.

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

### F-ENG-005: Discrete upper references need explicit aliases

- Class: `development smoke`
- Status: configuration and shape validation complete, performance not run
- Date: 2026-07-23
- Source contract: the pinned POBAX factory uses `perfect_memory=True` to keep
  the T-Maze cue visible and to wrap RockSample with its fully observable
  representation
  ([factory source](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/__init__.py#L171-L319)).
- Implementation: the registered runner exposes
  `tmaze_10-perfect-memory` and
  `rocksample_11_11-fully-observable`, which call the primary source with
  `perfect_memory=True`, and
  `battleship_10-perfect-recall`, which applies the same source flag and
  restores its omitted action mask, and
  `Navix-DMLab-Maze-01-fully-observable`, which calls the separately
  registered full source environment. Each hashed configuration records its
  source invocation, upper-reference class, primary parameter target, and
  inherited registered interaction budget. Upper-reference matrices accept
  only the memoryless policy.
- Shape validation:
  - T-Maze primary and persistent-cue observations both contain four values,
    giving policy input width ten with four actions. The matched policy has
    28,663 parameters against the 28,717-parameter primary target, a ratio of
    `0.9981`;
  - RockSample primary and fully observable observations both contain 33
    values, giving policy input width 51 with 16 actions. The upper wrapper
    replaces uncertain rock features with true rock morality rather than
    increasing the width
    ([wrapper source](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/rocksample.py#L93-L125)).
    The matched counts are 30,476 and 30,425, a ratio of `1.0017`;
  - Battleship perfect recall supplies a `(10, 10)` hit and miss history. The
    local adapter preserves that array and supplies the row-major mask
    `observation == 0`, giving 100 legal-action flags
    ([source wrapper](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/battleship.py#L67-L99)).
    Its policy input width is 202 against the primary width 103. The matched
    counts are 34,685 and 34,861, a ratio of `0.9950`; and
  - Navix primary runtime observations have shape `(3, 3, 2)`, or 18 values,
    while the full source observations have shape `(21, 41, 6)`, or 5,166
    values. The shared input adds the previous-action one-hot vector and two
    boundary flags, giving widths 23 and 5,171. A hidden width of six keeps
    the full-observation memoryless MLP within the primary ArcMind parameter
    budget. The matched counts are 31,102 and 29,100, a ratio of `1.0688`.
- Task-equivalence validation: each alias now fails closed unless its action
  space class, action dimension, evaluation horizon, and discount match the
  paired primary task. All four runtime descriptions passed.
- Contract correction: an independent review found that the initial
  Battleship wrapper delegated stale source `observation_space` and
  `dummy_observation` metadata. The corrected adapter declares the emitted
  named `(10, 10)` observation and 100-value Boolean mask, supplies a matching
  sequence-first dummy batch, and passes compiled tests against the pinned
  vector environment.
- Interpretation: an upper reference is defined by information, not tensor
  size or the label `perfect memory`. Explicit aliases prevent accidental
  mixing with primary cells and preserve the exact source invocation.
- Evidence restriction: configuration descriptions and adapter tests are not
  performance evidence.

### F-ENG-006: Upper-reference evidence needed four additional fail-closed gates

- Class: `engineering validation`
- Status: complete
- Date: 2026-07-23
- Trigger: independent review of the completed upper-reference paths.
- Findings:
  - the first Battleship adapter emitted the correct observation and mask but
    delegated stale `observation_space` and `dummy_observation` metadata;
  - aggregate writers could be pointed inside the immutable raw matrix and
    therefore could overwrite inputs after validating them;
  - independent aggregation checked the matrix role but did not prove the
    exact environment source, reference class, or parameter-match ratio; and
  - two separate 30-seed matrices did not by themselves prove that primary and
    upper-reference runs used the same ordered seeds.
- Corrections:
  - the Battleship adapter now declares and tests its exact named observation
    space and sequence-first dummy batch;
  - registered and development aggregate writers reject every output at or
    below the raw matrix root before input validation;
  - a pure six-alias registry supplies the runner, both aggregators, and the
    cross-matrix linker with one source of truth;
  - new frozen configurations record parameter count, effective parameter
    count, primary ArcMind target count, ratio, source invocation, and
    reference class. Registered aggregation requires these fields and checks
    the ratio range `0.9` through `1.1`;
  - the development aggregator validates the same artifact fields while
    explicitly marking whether an older primary pilot froze them inside its
    configuration; and
  - `link_upper_reference.py` validates both complete checksum inventories,
    exact ordered seeds, all alias mappings, learner and evaluation contracts,
    registered budgets, Git and dependency identity, and non-device runtime
    fields before creating a derived link outside both raw roots.
- Validation: 64 aggregate tests, 15 cross-matrix link tests, 11 adapter tests,
  and 24 runner-matrix tests passed in focused CPU runs. The runtime
  descriptions for all four discrete aliases and Walker-F passed their
  parameter and task-equivalence checks. The integrated POBAX CPU suite passed
  all 212 tests, and the Windows package suite passed all 118 selected tests.
  CI for commit `fdaa74f7be5d203f7c302cddaac2719abde8fbc6` passed on
  Python 3.10, 3.11, and 3.12
  ([run](https://github.com/jemsbhai/arcmind/actions/runs/30037446425)).
- Interpretation: equal seed cardinality is not paired evidence. Pairing
  requires a machine-verifiable link between the exact immutable manifests.
- Evidence restriction: these gates establish artifact validity, not policy
  performance.

### F-ENG-007: Tuning evidence requires a separate fail-closed selection path

- Class: `engineering validation`
- Status: protocol defect found and corrected before registered tuning
- Date: 2026-07-23
- Defect: smoke and pilot aggregation preserved development returns but did
  not define an artifact tier, common curve interval, or immutable eligibility
  label for hyperparameter selection. Using those outputs informally could
  mix shortened pilot evidence with full-budget tuning or make a selected
  development result appear registered.
- Correction:
  - add the artifact status `development_tuning_not_for_paper` and aggregate
    status `development_tuning_selection_aggregate_not_for_paper`;
  - reserve schema v3 and `matrix_kind: hyperparameter_selection` for tuning,
    with immutable candidate IDs grouped under unique model-family and
    implementation-model identities;
  - require `arcmind_shared_comparison`, exactly one published primary task at
    its full interaction budget, at least two candidates per family, equal
    candidate cardinality across families, and the published task-specific
    tuning-seed count;
  - require identical `num_envs`, `rollout_steps`, `update_epochs`, and
    `num_minibatches` across the entire matrix while allowing candidates to
    vary only learning rate, GAE lambda, entropy coefficient, and
    learning-rate annealing;
  - require a complete Cartesian candidate matrix, validated checksum and
    completion indexes, frozen environment semantics, frozen parameter
    matching, and identical within-task training-step grids;
  - require every schema-v3 completion row to identify its immutable cell log
    and hash, and require the checksum inventory to cover every log;
  - start every candidate curve at the latest first finite return across all
    candidate and seed cells, require at least two retained points, integrate
    by the trapezoidal rule without extrapolation, and divide by the common
    interval width;
  - rank candidates separately within each model family by mean seed
    `auc_mean_return`, with ascending candidate ID as the deterministic
    exact-tie rule, and never use the tuning aggregate to rank architectures;
    and
  - mark the aggregate eligible for hyperparameter selection only, while
    explicitly setting registered-final and paper-performance eligibility to
    false.
- Interpretation: equal candidate and seed cardinality are enforced by the
  frozen Cartesian matrix rather than inferred after execution. AUC selects
  one candidate per architecture over a common observed interval and never
  selects a checkpoint or removes a required baseline.
- Validation:
  - the focused registration and development-aggregation suite passed 90 of
    90 tests in the pinned Ubuntu environment;
  - the complete POBAX suite passed 269 of 269 tests in the same environment;
    and
  - rebuilding the legacy SHM repair aggregate produced 7,420 bytes identical
    to the existing artifact, with SHA256
    `c5b926b38e32b52eba45d9eaacd7b0bc9478b00c03aeb1218ae2424bb79b7a8f`.
- Evidence restriction: the tuning aggregate can freeze a choice for a new
  registered-final manifest. None of its performance measurements can enter a
  registered table or paper claim.

### F-ENG-008: Selection-to-final identity and attempt logs require stronger binding

- Class: `engineering validation`
- Status: three protocol gaps found by independent audit and corrected before
  registered tuning
- Date: 2026-07-23
- Gaps:
  - schema-v2 primary registered-final matrices could declare any global
    learner and could reuse tuning seeds;
  - equal tuning-grid cardinality did not require the same normalized learner
    configuration set across model families; and
  - a fixed canonical log path was written before the child return code was
    checked, so a failed-attempt log could survive beside a later successful
    artifact.
- Correction:
  - reserve schema v4 for primary registered-final comparisons and require a
    `tuning_selection` binding to the raw schema-v3 matrix, canonical tuning
    aggregate and file SHA256, source registration and manifest hashes, and
    the exact winner identity and complete learner for every model family;
  - rebuild the tuning aggregate during final registration loading, require
    exact canonical byte identity, revalidate the binding during registered
    aggregation, and prohibit all overlap between tuning and final seeds;
  - retain schema-v2 registered upper references without a tuning binding,
    including the author-semantics lane;
  - require the exact same normalized learner configuration set across every
    tuning family, in addition to equal cardinality and shared structural
    fields; and
  - write the canonical cell log only after a successful child exit, while
    moving failed logs and any partial artifacts to unique noncanonical
    attempt paths.
- Interpretation: a final primary cell is now mechanically derived from one
  verified tuning winner, not merely labeled as selected. Search-space
  fairness is enforced by configuration identity rather than trial count
  alone. A canonical log now certifies the same successful attempt that
  produced the canonical artifact.
- Validation:
  - the focused schema, aggregation, matrix, and linking suite passed 170 of
    170 tests in the pinned Ubuntu environment;
  - the complete POBAX suite passed 287 of 287 tests in the same environment;
    and
  - rebuilding the legacy SHM repair aggregate produced 7,420 bytes identical
    to the existing artifact, with SHA256
    `c5b926b38e32b52eba45d9eaacd7b0bc9478b00c03aeb1218ae2424bb79b7a8f`.
- Evidence restriction: this correction validates selection provenance and
  execution identity only. It introduces no performance result or paper
  claim.

### F-ENG-009: Final selection requires source-complete and raw-complete evidence

- Class: `engineering validation`
- Status: twelve protocol gaps found by four independent audit passes and corrected before
  registered tuning
- Date: 2026-07-23
- Gaps:
  - schema-v4 final selection bound a tuning aggregate and its registration
    and manifest, but did not prove that tuning and final execution used the
    same implementation source;
  - the selection binding did not freeze the tuning completion-index or
    checksum-inventory file hashes;
  - failed-attempt evidence remained inside the immutable raw root even though
    strict linking treated that root as a closed inventory; and
  - standalone registered aggregation could validate cell artifacts without
    independently requiring the registration, completion index, canonical
    logs, and exact checksum inventory;
  - a schema-v4 manifest compared only selected registration semantics, so a
    different but rechecksummed final registration could remain admissible;
  - a special-case filename allowance still admitted checksummed
    failed-attempt evidence inside the immutable raw root; and
  - completion-index content was structurally validated but its exact
    canonical bytes were not required;
  - schema-v3 tuning aggregation required canonical files to be checksummed
    but did not reject additional checksummed files inside the raw root; and
  - schema-v2 registered upper references did not bind the frozen learner and
    other executable registration semantics back to validated cell artifacts;
  - a legacy schema-v2 primary matrix could still be labeled registered-final
    and linked as paper-eligible without the mandatory tuning selection;
  - schema-v3 tuning did not enforce a declared GPU requirement against
    validated runtime provenance; and
  - schema-v1 registered aggregation did not bind the frozen evaluation
    episode count back to validated artifacts.
- Correction:
  - define a versioned canonical implementation-source manifest over every
    tracked Python file under `arcmind/` and every tracked non-test Python file
    under `benchmarks/pobax/`, recording each path and file SHA256;
  - require tuning and final execution to have the exact same implementation
    manifest, dependency-lock hash, POBAX commit, Navix commit, and non-device
    runtime contract, while allowing the repository commit to differ;
  - bind the tuning completion-index, checksum-inventory, and
    implementation-source hashes in the schema-v4 registration, final
    manifest, every selected family, and every final cell;
  - bind the complete canonical schema-v4 final registration bytes into the
    frozen manifest with a file SHA256;
  - preserve failed logs and partial artifacts only in the sibling
    `<raw-matrix>.attempts` tree, never inside the immutable raw root; and
  - make registered aggregation require canonical registration bytes, an
    exact canonical completion index, every canonical cell log, and a checksum manifest
    whose paths exactly equal all regular files below the raw root. The
    aggregate records the validated registration, completion-index, and
    checksum-inventory hashes;
  - require schema-v3 tuning checksum paths to equal the exact canonical
    registration, manifest, completion, artifact, and log inventory; and
  - require schema-v2 registered artifacts to match the frozen learner,
    environment budget, evaluation episode count, comparison profile, and
    quick-run contract. A GPU requirement must agree with validated runtime
    provenance;
  - reject every registered-final primary link unless its primary matrix uses
    schema v4 and a frozen tuning selection;
  - enforce tuning GPU requirements against the frozen manifest runtime; and
  - apply evaluation episode binding to schema-v1 registered artifacts too.
- Interpretation: a final result can no longer inherit a tuning choice after
  implementation, dependency, external-source, or runtime drift. Binding the
  two raw index files prevents later reindexing from preserving a stale
  selection claim. Attempt evidence remains auditable without weakening the
  immutable successful-run inventory. The frozen manifest also binds every
  final-registration byte, including fields that are not part of matrix
  identity. Current schema-v2 upper references can no longer acquire a
  different learner or execution contract through registration-only edits.
  Legacy primary matrices cannot enter the paper-eligible linking path.
  Registered and tuning aggregation now fail closed when any independent raw
  evidence layer is absent, stale, noncanonical, extra, or unindexed.
- Validation:
  - the focused schema, aggregation, matrix, and linking suite passed 197 of
    197 tests in the pinned Ubuntu environment;
  - targeted adversarial regressions passed 5 of 5 tests for registration
    mutation, checksummed in-root attempt evidence, and noncanonical
    completion bytes;
  - the final independent adversarial audit passed 15 of 15 focused
    reproductions and found no remaining P0, P1, or P2 integrity bypass;
  - the complete POBAX suite passed 332 of 332 tests in the same environment;
    and
  - rebuilding the legacy SHM repair aggregate produced 7,420 bytes identical
    to the existing artifact, with SHA256
    `c5b926b38e32b52eba45d9eaacd7b0bc9478b00c03aeb1218ae2424bb79b7a8f`.
- Evidence restriction: this correction strengthens provenance and artifact
  integrity only. It introduces no performance result or paper claim.

### F-ENG-010: Mamba-1 requires an immutable source contract

- Class: `engineering validation`
- Status: implementation complete, performance experiments not started
- Date: 2026-07-23
- Question: can Mamba-1 enter the POBAX comparison through the exact shared
  learner without relying on an unverified reimplementation label?
- Source audit:
  - the implementation is based on official Mamba package version
    `2.2.6.post3` at commit
    `10b5d6358f27966f6a40e4bf0baa17a460688128`;
  - the contract records SHA256 hashes for `mamba_simple.py`, `block.py`,
    `mixer_seq_simple.py`, `config_mamba.py`, and `layer_norm.py`; and
  - an immutable fixture records outputs and recurrent caches from the
    official dependency-light `Mamba.step` path.
- Implementation:
  - add one Mamba-1 residual block with expansion factor 2, state size 16,
    convolution width 4, automatic time-step rank, and RMSNorm with epsilon
    `1e-5`;
  - preserve independent convolution and SSM caches for every environment,
    including asynchronous reset behavior;
  - adapt only the common policy input, actor-critic heads, and hidden width;
  - select the integer hidden width with the globally closest parameter count
    to the ArcMind target; and
  - route Mamba-1 through the same collection, PPO replay, optimizer,
    evaluation, artifact, and aggregation paths as every other policy core.
- Evidence controls:
  - every frozen Mamba configuration and artifact contains the exact audited
    source metadata;
  - matrix resume rejects missing or changed metadata; and
  - development and registered aggregation independently revalidate the same
    source contract.
- Validation:
  - transplanted official weights reproduce the pinned official-step fixture;
  - equation, causal sequence, asynchronous reset, JIT, gradient, parameter,
    cache, and width-matching tests pass;
  - exact shared-PPO replay has zero pre-update KL; and
  - focused launcher and aggregation tests accept intact Mamba evidence and
    reject source drift;
  - the integration-focused slices passed 225 of 225 tests in the pinned
    Ubuntu environment; and
  - the complete POBAX suite passed 340 of 340 tests in that environment.
- Independent audit:
  - the five pinned upstream source hashes and the `v2.2.6.post3` tag were
    verified from an independent checkout;
  - an independent PyTorch fixture replay had maximum output error
    `1.19e-7` and exact final recurrent caches;
  - a 16,384-case analytic width grid stayed within the 10 percent parameter
    tolerance, with worst ratio `0.98558`;
  - the focused audit suite passed 200 of 200 tests and the complete POBAX
    suite passed 340 of 340 tests; and
  - no P0, P1, or P2 correctness or evidence-integrity issue was found.
- Interpretation: Mamba-1 is now an executable, source-audited comparison
  core. This engineering result establishes implementation identity and
  learner parity only.
- Evidence restriction: no Mamba-1 return, efficiency, superiority, or
  equivalence claim is supported until registered experiments are complete.

### F-REF-004: The existing Memory Traces core is an adaptation, not an official policy

- Class: `reference audit`
- Status: primary paper and official source verified, implementation revision
  required before registered use
- Date: 2026-07-23
- Primary sources:
  - ICML 2025 proceedings:
    `https://proceedings.mlr.press/v267/eberhard25a.html`;
  - official MIT repository:
    `https://github.com/onnoeberhard/memory-traces`; and
  - audited source revision:
    `fcfdacc0b0a06dc181b49b9ef95893dbae7f2bcd`.
- Verified mechanism: each fixed trace updates as
  `z_t = lambda * z_(t-1) + (1 - lambda) * y_t`, with no learned trace
  parameter, gate, bias, activation, or normalization. The PPO implementation
  concatenates traces in trace-major order.
- Official T-Maze policy:
  - traces observations only by default;
  - resets state at episode boundaries and incorporates the new initial
    observation on the reset step;
  - uses separate two-layer, width-64 tanh actor and critic networks; and
  - uses orthogonal initialization, actor output gain `0.01`, critic output
    gain `1.0`, and zero biases.
- Current ArcMind difference: `memory_trace_mlp` applies the correct recurrence
  to the complete shared causal input, including previous action, previous
  reward, and reset metadata. It uses a shared parameter-matched trunk and
  Xavier initialization. It is recurrence-faithful but not a complete
  source-faithful policy.
- Decision:
  - rename the registered shared comparison to `memory_trace_shared`;
  - add `memory_trace_official` with observation-only traces and the official
    separate actor and critic architecture; and
  - retain the old name only as a development compatibility alias.
- Configuration caveat: the paper defines the corridor decay as `(k - 1) / k`.
  The public corridor-64 example rounds it to `0.985` instead of the exact
  `0.984375`. No official POBAX decay or parameter-matching rule exists, so
  POBAX decays must be frozen as tuning choices.
- Evidence restriction: the existing core cannot be called an official
  reproduction. No performance claim is supported by this audit.

### F-REF-005: AGaLiTe is a missing matched online-PPO baseline

- Class: `reference audit`
- Status: baseline set revised after a primary-source search through
  2026-07-23
- Date: 2026-07-23
- Finding: AGaLiTe is accepted by TMLR and directly evaluates a constant-state
  approximate gated linear-attention policy in partially observable online
  RL with PureJaxRL PPO.
- Primary sources:
  - TMLR OpenReview record: `https://openreview.net/forum?id=lh6vOAHuvo`; and
  - official JAX and Flax source:
    `https://github.com/subho406/agalite` at
    `101acbecc121a258ad8f7e58e2f782f546674979`.
- Decision: add AGaLiTe to the mandatory executable matched-PPO set.
- Source incompatibility:
  - the paper defines `r + 1` cosine channels with frequencies
    `2 * pi * i / r`, while the released code stores exactly `R` channels with
    `linspace(-pi, pi, R)`;
  - the released phase counter starts at one, the first token uses phase two,
    and episode resets do not reset phase;
  - the released readout divides by
    `2 * R * dot(s, q) + 1e-5`, while two finite-r expressions in the paper
    differ by a factor of four; and
  - published experiments report `r = 1`, while released configurations use
    `R = 2`.
- Implementation decision:
  - `agalite_source_compat` will preserve the pinned executable recurrence,
    phase, frequency, denominator, and GTrXL-style block;
  - `agalite_shared` will attach that source-compatible block to the shared
    input, PPO, and parameter-matching contract; and
  - a literal paper-equation implementation may be retained only as an audit
    ablation.
- License: the official source is Apache-2.0. A close port requires preserved
  attribution, the license text, and a third-party notice identifying the
  audited revision.
- Other classification changes:
  - Memory Traces, Mamba-1, S5RL, LRU, FFM, SHM, recurrent attention, GRU,
    LSTM, memory-free controls, and ArcMind ablations remain mandatory;
  - Mamba-2, Mamba-3, RATE, ELMUR, GPO, Memo, LinOSS, the Kalman state-space
    layer, and MS4/MS4N remain contextual because their published evidence
    changes the task, learner, supervision, privilege, scale, or application;
    and
  - POPGym Arcade remains an ICLR 2026 submission. Its current release is
    0.0.7 at source commit
    `d061b611718ae55d095791b4ea7046b5266cafd4`.
- Evidence restriction: baseline relevance does not imply that ArcMind
  matches or exceeds any listed method. Comparative claims require the frozen
  shared protocol and registered results.

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
- Development aggregate: the later fail-closed development aggregator accepted
  the immutable matrix and produced
  `benchmark_results/pobax/aggregates/matrix-smoke-controls-v1-de799c3.json`,
  SHA256
  `9b7ee143ab7a50ad256f819a597c3252c5eb6859e8f81bcfeb527047560bb8e6`.
  It validated every top-level parameter match and labeled the result
  `development_smoke_aggregate_not_for_paper`. Its semantic-freeze flags are
  false because this historical smoke manifest predates parameter counts and
  environment source metadata inside the hashed configuration.
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

### F-REF-002: POBAX paper and pinned RockSample scripts define different best settings

- Class: `reference verification`
- Status: completed source audit, not a performance finding
- Date: 2026-07-23
- Sources: the primary
  [POBAX paper](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_153.pdf)
  and author repository commit
  [`a5e1d62d14e4efe783885b9d4f19cffa2a568eec`](https://github.com/taodav/pobax/tree/a5e1d62d14e4efe783885b9d4f19cffa2a568eec).
- Verified paper contract:
  - the general PPO defaults are 128 rollout steps, four update epochs, four
    minibatches, entropy coefficient `0.01`, and linear learning-rate
    annealing;
  - T-Maze, RockSample(11,11), Battleship, masked MuJoCo, and Navix-01 use 4,
    8, 32, 4, and 256 parallel environments, respectively;
  - RockSample uses entropy coefficient `0.2`, while Battleship uses `0.05`;
    and
  - learning rate and GAE lambda are selected separately by algorithm before
    30-seed final evaluation.
- Verified RockSample(11,11) discrepancy:

  | Model | Paper learning rate | Paper GAE lambda | Pinned script learning rate | Pinned script GAE lambda | Pinned script environments | Pinned script entropy |
  |---|---:|---:|---:|---:|---:|---:|
  | Memoryless | 0.0025 | 0.3 | 0.0025 | 0.7 | 8 | 0.2 |
  | RNN | 0.00025 | 0.95 | 0.0025 | 0.7 | 8 | 0.2 |
  | Transformer-XL | 0.00025 | 0.1 | 0.00025 | 0.7 | 16 | 0.1 |

- Script evidence:
  [RNN](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/scripts/hyperparams/rocksample/best/rocksample_11_11_ppo_best.py),
  [memoryless](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/scripts/hyperparams/rocksample/best/rocksample_11_11_ppo_memoryless_best.py),
  and
  [Transformer-XL](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/scripts/hyperparams/rocksample/best/rocksample_11_11_transformer_best.py).
  Downloaded file SHA256 values were
  `64fcd689f580f90043366ce5a6482a0670e885878c84cd9e6c08da2dc2a3c952`,
  `c23f48783f0e67e398cd7d6039ddbdffb5054e66280c75d3af3fa356edacc658`,
  and
  `b84e52222a614b816b5ccfbc1e1966ce1e8db20f5c4508f6de7b3e0608adc8a5`,
  respectively.
- Interpretation: there is no single unambiguous source-defined RockSample
  profile. The paper table and executable author scripts must not be silently
  combined.
- Disposition: maintain two named evidence lanes. A pinned author-code
  reproduction follows and records the executable scripts. The ArcMind shared
  comparison applies an equal-cardinality tuning grid to every architecture.
  Neither lane may be described as reproducing the other.
- Pilot restriction: `F-PILOT-001` uses one shared constant learning rate,
  GAE lambda `0.95`, entropy coefficient `0.01`, eight environments, and no
  learning-rate annealing. It is a frozen development viability screen, not a
  POBAX hyperparameter reproduction.

### F-PILOT-001: T-Maze multi-seed viability screen

- Class: `development pilot`
- Status: execution complete, numerical acceptance gate failed
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
- Execution outcome:
  - all 30 planned cells produced immutable artifacts at exactly 250,000
    environment transitions and 128 evaluation episodes;
  - all cells used the local GPU and clean detached commit
    `4b9887126678c08967f7a92e8a3c77abf65f486c`;
  - the launcher completed in 5,645.3 wall-clock seconds, while cell artifacts
    recorded 5,464.3 training seconds in total; and
  - no subprocess or artifact-writing operation failed.
- Evaluation returns, in seed order `1103`, `2207`, and `3301`:

  | Model | Seed 1103 | Seed 2207 | Seed 3301 | Across-seed mean | Parameters | ArcMind ratio |
  |---|---:|---:|---:|---:|---:|---:|
  | FFM | 4.0000 | 4.0000 | 4.0000 | 4.0000 | 28,577 | 0.9951 |
  | GRU | 4.0000 | 4.0000 | 4.0000 | 4.0000 | 28,893 | 1.0061 |
  | LSTM | 4.0000 | 4.0000 | 4.0000 | 4.0000 | 28,840 | 1.0043 |
  | S5RL | 4.0000 | 4.0000 | 4.0000 | 4.0000 | 28,617 | 0.9965 |
  | ArcMind SSM only | 4.0000 | 2.2062 | 4.0000 | 3.4021 | 28,717 | 1.0000 |
  | Transformer-XL | 1.9375 | 4.0000 | 4.0000 | 3.3125 | 28,869 | 1.0053 |
  | ArcMind | 1.6617 | 4.0000 | 4.0000 | 3.2206 | 28,717 | 1.0000 |
  | SHM | 0.0000 | 4.0000 | 4.0000 | 2.6667 | 28,727 | 1.0003 |
  | Memoryless MLP | 1.6617 | 2.2062 | 1.6297 | 1.8326 | 28,663 | 0.9981 |
  | Positional MLP | 1.5656 | 0.0000 | 2.2062 | 1.2573 | 28,526 | 0.9933 |

- Numerical failure: SHM seed `1103` had finite metrics through update 110,
  then nonfinite loss, actor loss, value loss, entropy, and approximate KL
  from environment step 111,000 through the end of training. The immutable
  JSON represents these values as null. SHM seeds `2207` and `3301` remained
  finite. Two other early null recent-return values occurred before the first
  completed training episode and are not numerical failures.
- Integrity:
  - matrix-manifest identity:
    `69a858ad9e97cfba8b68d0c330b0a8234480aa5ffcc1b6a1f4db2f8282e776c4`;
  - `checksums.sha256` SHA256:
    `545c3a0c34f38a824e3fb6b61643c30dc449edbd88f1b628d764ae907c4022bb`;
  - frozen-manifest SHA256:
    `c10323dcc36d7bc3471cda5a27c7032271df1e55216b6c43e3352f4345eb066c`;
  - completion-index SHA256:
    `1738d86512b2a56635912d94fe174a6b38567bb40fb30bc00ca9d2f99f23cbb0`;
    and
  - `sha256sum -c` independently passed for every cell artifact, log,
    registration, manifest, and completion index.
- Aggregation defect found: the first development aggregator accepted the
  matrix because it checked evaluation returns but did not reject null
  optimizer metrics. That aggregate was immediately quarantined as
  `benchmark_results/pobax/aggregates/tmaze-pilot-v1-4b98871.invalid-pre-finite-gate.json`,
  SHA256
  `ea2313b99d4fe27f6d149267df66c934250f95f47e24821ec92246250b4170b2`.
  It is invalid evidence. A new fail-closed gate must reject every required
  nonfinite optimizer metric before another aggregate is accepted.
- Decision-rule outcome: the preregistered finite-metric prediction failed.
  Diagnose SHM, make any source-faithful repair under a new code commit, and
  rerun all three SHM seeds under a new frozen manifest. Do not replace only
  seed `1103`, and do not promote the current rank table to a paper claim.
- Timing restriction: local CPU validation and source auditing overlapped with
  parts of the GPU matrix, so recorded training times are engineering
  diagnostics rather than controlled throughput comparisons.

### F-PILOT-002: Source-faithful SHM repair passes the three-seed numerical gate

- Class: `development pilot`
- Status: execution and integrity validation complete, numerical repair
  accepted, not for a paper performance claim
- Date planned and frozen: 2026-07-23
- Date completed: 2026-07-23
- Question: does restoring the official POMDP cell's `[-100, 100]`
  recurrent-state clamp eliminate the SHM nonfinite failure across every
  original pilot seed?
- Prediction stated before execution: all three repaired `paper_uniform` SHM
  cells will finish 250,000 transitions with finite optimizer metrics. No
  directional return prediction was registered.
- Frozen matrix:
  - manifest:
    `benchmarks/pobax/manifests/tmaze_shm_repair_v2.json`;
  - source commit:
    `cfe0fc3ee782b094573f22d782cf0bcd62f09978`;
  - schema: `2`;
  - comparison profile: `arcmind_shared_comparison`;
  - models: repaired SHM and ArcMind as the required comparison anchor;
  - seeds: `1103`, `2207`, and `3301`;
  - learner: eight environments, 125 rollout steps, four update epochs, four
    minibatches, constant learning rate `0.00025`, GAE lambda `0.95`, and
    entropy coefficient `0.01`;
  - budget: exactly 250,000 requested and realized transitions per cell; and
  - evaluation: 16 episodes per environment, or 128 episodes per seed.
- Numerical outcome:
  - every one of the six cells recorded 250 finite PPO updates and completed
    independent evaluation;
  - SHM mean evaluation returns were `4.0`, `4.0`, and `4.0` for seeds `1103`,
    `2207`, and `3301`;
  - ArcMind mean evaluation returns were `4.0`, `4.0`, and `0.0` in the same
    seed order; and
  - the strict development aggregate therefore reports an across-seed mean of
    `4.0` for SHM and `2.6667` for ArcMind.
- Interpretation: the source-faithful POMDP recurrence clamp passes the
  predeclared numerical acceptance gate. The historical failure in
  `F-PILOT-001` remains a failed artifact and was not replaced or reclassified.
  The three-seed return difference is development evidence only and cannot
  support a comparative claim.
- Integrity:
  - matrix-manifest identity:
    `e065b36a930079e601c4ca648bcf9df01246d8f53a198203e8305456fceed062`;
  - `checksums.sha256` SHA256:
    `fe9590afc771ea9609ffedb86391bfeb4b6ba8ebaa0a45eadd4d5a5e6a0e1aba`;
  - frozen-manifest SHA256:
    `f25c095e634062630aaad3fce375af0db96e4e37530de18d247ed1cc16bf4509`;
  - completion-index SHA256:
    `53b53490c5cf1380ff8441d70b5b5175efd47587b341a0894b3fbda705f4d082`;
  - registration SHA256:
    `1fc57b7760ae92d6528def20203f1f2d23a212c046ade2885acc962ae1001edd`;
  - independent `sha256sum -c` validation passed before and after a resume
    check; and
  - resume completed six of six cells in 26.6 seconds without retraining or
    changing any recorded hash.
- Aggregate: the strict development aggregate is
  `benchmark_results/pobax/aggregates/tmaze-shm-repair-v2-cfe0fc3.json`,
  SHA256
  `c5b926b38e32b52eba45d9eaacd7b0bc9478b00c03aeb1218ae2424bb79b7a8f`.
  Its status is `development_pilot_aggregate_not_for_paper`.
- Runtime restriction: cell artifacts record 1,677.6 training seconds in
  total. A validation process overlapped part of the matrix, so these timings
  are not controlled throughput evidence.
- Disposition: retain the repaired `paper_uniform` SHM core in later shared
  comparisons. Do not use this repair matrix in a paper table, abstract, or
  conclusion. The next T-Maze coverage and ablation manifest was frozen before
  this aggregate was inspected.

### F-PILOT-003: T-Maze coverage and ablation pilot exposes seed sensitivity

- Class: `development pilot`
- Status: execution, aggregation, integrity, and resume validation complete,
  not for a paper performance claim
- Date planned and frozen: 2026-07-23
- Date completed: 2026-07-23
- Question: do the missing short-window, stable-recurrence, and ArcMind
  ablation cells execute reproducibly enough to justify full-budget tuning?
- Prediction stated before execution: every cell will complete with finite
  optimizer metrics. No directional return or model-ranking prediction was
  registered.
- Frozen matrix:
  - manifest:
    `benchmarks/pobax/manifests/tmaze_coverage_ablation_v2.json`;
  - source commit:
    `3b947d2968d14f313db450a1f9e009123a373a75`;
  - schema: `2`;
  - comparison profile: `arcmind_shared_comparison`;
  - models: four-frame MLP, LRU, S4D, unordered ArcMind, ArcMind without
    memory, ArcMind without the SSM, ArcMind without the learned gate, and
    full ArcMind;
  - seeds: `1103`, `2207`, and `3301`;
  - learner: eight environments, 125 rollout steps, four update epochs, four
    minibatches, constant learning rate `0.00025`, GAE lambda `0.95`, and
    entropy coefficient `0.01`;
  - budget: exactly 250,000 requested and realized transitions per cell; and
  - evaluation: 16 episodes per environment, or 128 episodes per seed.
- Completed results:

  | Model | Seed 1103 | Seed 2207 | Seed 3301 | Strict mean |
  |---|---:|---:|---:|---:|
  | Four-frame MLP | 1.8539 | 2.0461 | 1.8539 | 1.9180 |
  | LRU | 4.0000 | 4.0000 | 4.0000 | 4.0000 |
  | S4D | 4.0000 | 4.0000 | 0.0000 | 2.6667 |
  | Unordered ArcMind | 4.0000 | 4.0000 | 4.0000 | 4.0000 |
  | ArcMind without memory | 1.6617 | 4.0000 | 0.0000 | 1.8872 |
  | ArcMind without SSM | 0.0000 | 4.0000 | 4.0000 | 2.6667 |
  | ArcMind without learned gate | 4.0000 | 2.2062 | 2.2703 | 2.8255 |
  | ArcMind | 4.0000 | 1.8219 | 4.0000 | 3.2740 |

- Numerical outcome: all 24 cells recorded 250 finite PPO updates, completed
  independent evaluation, and retained every low or null return. The
  predeclared numerical prediction passed.
- Interpretation: the three-seed pilot is visibly seed-sensitive. LRU and
  unordered ArcMind reached `4.0` in all three cells, while S4D, the
  no-memory ablation, and the no-SSM ablation each included both `0.0` and
  `4.0` outcomes. This matrix is too small and too short to support a ranking
  or component-importance claim.
- Architecture audit triggered by the pilot: `F-DIAG-006` proves that the
  benchmark-only ArcMind attention window cannot expose the initial T-Maze
  cue at the junction. The current result therefore cannot test the intended
  exact-recall mechanism fairly.
- Integrity:
  - source manifest SHA256:
    `fda5837ef8c177e78f73aa439c90e255fdc32ad33dce95fcf9fdaee9e6bb50bb`;
  - matrix-manifest identity:
    `c8b8d36405aa8cfd9d6a1ce459ed858e9181cadc1050a679e51441f9519d4c81`;
  - `checksums.sha256` SHA256:
    `fcb41616b3ef3925a52f19e4bd77b0b1cac0b95978f8df8ba8f9277ef251e319`;
  - frozen-manifest SHA256:
    `81ede090d7f5def0e2d247736f19655e051e5c455697201f38a6ba5018201a9a`;
  - completion-index SHA256:
    `365d05a7ed631ffed26336e5d15b8e92b748713682cecd28a0efd00311f84f06`;
  - registration SHA256:
    `eeff948367c30fe836b4630e4311a92a2f572322b31d9858232e3f29157e496f`;
  - independent `sha256sum -c` validation passed before and after the resume
    check; and
  - resume completed 24 of 24 cells in 36.6 seconds without retraining or
    changing any recorded hash.
- Aggregate: the strict development aggregate is
  `benchmark_results/pobax/aggregates/tmaze-coverage-ablation-v2-3b947d2.json`,
  SHA256
  `b35f5221ac934e0101db43130e2445915393e0c7fd5cca550c67dca161e8e0b0`.
  Its status is `development_pilot_aggregate_not_for_paper`.
- Runtime: cell artifacts record 6,450.2 local GPU training seconds in total.
  This is engineering timing, not controlled throughput evidence.
- Disposition: preserve the matrix as negative development evidence. Repair
  the benchmark attention horizon under a new commit and frozen manifest
  before hyperparameter selection. Do not use this matrix in a paper table,
  abstract, conclusion, or architecture-ranking statement.

### F-DIAG-006: The pilot attention window excludes the T-Maze start cue

- Class: `configuration diagnosis`
- Status: verified from pinned source, repair pending
- Date: 2026-07-23
- Observation: the benchmark ArcMind configuration uses decision stride one,
  16 memory slots, and an attention window of eight prior decision snapshots.
  The package presets use windows of 32 or more and are not affected.
- Pinned environment contract:
  - the source factory parses `tmaze_10` as `hallway_length=10`;
  - the start observation at grid index zero exposes the goal cue;
  - the junction is grid index `hallway_length + 1`, or 11; and
  - the policy must therefore retain the start cue through 11 transitions.
- Causal-memory consequence: ArcMind writes a snapshot after the current
  decision and reads only strictly prior snapshots. At the junction, memory
  contains the snapshots from decision times zero through ten. An
  eight-snapshot window exposes only times three through ten, so the initial
  cue at time zero is not an attention key or value. A window of at least 11
  is required for direct exact recall.
- Verified primary sources:
  - factory at pinned POBAX commit
    [`a5e1d62d14e4efe783885b9d4f19cffa2a568eec`](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/__init__.py),
    downloaded-file SHA256
    `eae213739f4ad7b7d8e1afd57de4bf334358055fa07ea0ea813080f44a345749`;
    and
  - [T-Maze transition and observation source](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/tmaze.py),
    downloaded-file SHA256
    `6102e0c974cae6e7827ad03c2e8e066538e4b616ae6d2cffb2356a4bfa80a222`.
- Supported conclusion: the pilot configuration cannot use its exact-memory
  path to retrieve the task-defining start cue at the junction. Any successful
  pilot policy must solve the task through the recurrent path, a bounded
  shortcut, or their learned interaction.
- Repair decision: set the benchmark attention window to 16, equal to the
  existing memory capacity, then rerun a new three-seed ArcMind repair pilot.
  This changes no package preset, preserves the 28,717-parameter target, and
  is recorded before inspecting any repair outcome.
- Evidence restriction: this diagnosis invalidates the pilot as a test of
  exact recall. It does not invalidate the finite-execution result and does
  not imply that the repaired model will improve.

### F-PILOT-004: T-Maze attention-horizon repair pilot

- Class: `development pilot`
- Status: completed, not for a paper performance claim
- Date planned and frozen: 2026-07-23
- Date completed: 2026-07-23
- Question: does extending the benchmark attention window from eight to 16
  prior snapshots translate the semantic T-Maze repair into reliable
  three-seed accuracy?
- Prediction and decision rule stated before execution:
  - every cell must complete with finite optimizer metrics;
  - a repaired across-seed mean evaluation return of at least `3.5`, with
    every seed strictly above `0`, counts as an accuracy repair; and
  - any other finite outcome means the horizon is semantically corrected but
    accuracy remains unresolved.
- Frozen matrix:
  - manifest:
    `benchmarks/pobax/manifests/tmaze_attention_horizon_repair_v3.json`;
  - repair implementation commit:
    `2d01d304c6fe112822d5fffe804873860f90a12b`;
  - execution source commit:
    `390d5996024b41d0506b2611f22ef33db619284b`;
  - schema: `2`;
  - comparison profile: `arcmind_shared_comparison`;
  - model: ArcMind only;
  - environment: `tmaze_10`;
  - seeds: `1103`, `2207`, and `3301`;
  - learner: eight environments, 125 rollout steps, four update epochs, four
    minibatches, constant learning rate `0.00025`, GAE lambda `0.95`, and
    entropy coefficient `0.01`;
  - budget: exactly 250,000 requested and realized transitions per cell;
  - evaluation: 16 episodes per environment, or 128 episodes per seed; and
  - accelerator: GPU required.
- Raw artifact directory:
  `benchmark_results/pobax/tmaze-attention-horizon-repair-v3-390d599`.
- Derived aggregate:
  `benchmark_results/pobax/aggregates/tmaze-attention-horizon-repair-v3-390d599.json`.
- Integrity:
  - canonical expanded-manifest identity:
    `bad6b5d00684ec2f9cbb40f2b65e018eb5e36661b46994755f9cfed14c999ea1`;
  - source manifest file SHA256:
    `36b41ba20a5da85f733c7c2d99820266f5cb93d212b85f672391d802e1993425`;
  - frozen manifest file SHA256:
    `474607c21488268355db1ab5ff31176576746793aac6d2f4888b7a270d35c4c4`;
  - registration file SHA256:
    `cad6a1066ce15559a6748b6e7cf71521b48446819415b6ee378fe0917cb866d1`;
  - completion-index file SHA256:
    `f900915470b02d7afba7029f06cbf6e821801c1cc5de3db2537ad72e05434547`;
  - checksum-inventory file SHA256:
    `124f931c93c9e78b72bc53bb2535d9dda4f3c9c58e4cd9f7c2f629dc09896784`;
  - aggregate file SHA256:
    `9929b47f8a9a1c62bd1d50f296e85adaf5e3b867efea5347269220ec72e7df01`;
  - all checksum entries passed before and after the resume check;
  - the completion and checksum indexes validated during aggregation;
  - every artifact records clean Git provenance at the execution source
    commit; and
  - a second launcher invocation completed in 17 seconds, skipped all three
    compatible cells, and preserved all listed raw-matrix hashes.
- Observation:
  - seed `1103`: mean return `4.0`;
  - seed `2207`: mean return `4.0`;
  - seed `3301`: mean return `4.0`;
  - across-seed mean, median, and interquartile mean: `4.0`;
  - every seed used 128 evaluation episodes;
  - every cell realized exactly 250,000 environment transitions and 250
    optimizer updates; and
  - all histories and final optimizer metrics were finite. Total recorded
    training time was `1172.55904429` seconds.
- Decision: the preregistered accuracy-repair criterion is satisfied. The
  repaired mean is at least `3.5`, every seed is strictly above `0`, and all
  cells are finite.
- Interpretation: window 16 restores access to the start cue and removes the
  accuracy failure observed in the window-eight pilot under this exact
  three-seed development protocol.
- Evidence restriction: this pilot can accept or reject the stated
  development repair criterion. It cannot establish comparative performance,
  support a paper result, or replace a registered-final matrix.

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

### F-REF-003: Official SHM sources use different recurrence bounds

- Class: `reference verification` and `diagnostic evidence`
- Status: diagnosis and source-faithful repair complete, registered rerun
  pending
- Date: 2026-07-23
- Sources: the ICLR 2025
  [SHM paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/b6446566965fa38e183650728ab70318-Paper-Conference.pdf)
  and official source commit
  [`40d73d44936e47a29e2c76a481d93c434b857ea1`](https://github.com/thaihungle/SHM/tree/40d73d44936e47a29e2c76a481d93c434b857ea1).
- Source discrepancy:
  - the official
    [POMDP cell](https://github.com/thaihungle/SHM/blob/40d73d44936e47a29e2c76a481d93c434b857ea1/pomdp-baselines/torchkit/shm.py#L68-L75)
    clamps the recurrent matrix to `[-100, 100]` after every calibration and
    write;
  - the official
    [POPGym cell](https://github.com/thaihungle/SHM/blob/40d73d44936e47a29e2c76a481d93c434b857ea1/popgym/baselines/ray_models/ray_shm.py#L83-L87)
    and
    [standalone cell](https://github.com/thaihungle/SHM/blob/40d73d44936e47a29e2c76a481d93c434b857ea1/shm.py#L59-L64)
    do not clamp the recurrence; and
  - the paper states that cumulative products can occasionally become large
    enough to overflow and names gradient clipping as a remedy, but its
    equations do not specify the POMDP cell's forward-state clamp.
- Downloaded source SHA256 values:
  - POMDP cell:
    `7df5a127d286434a52a8294f68b6e86ac297d010c06c6f774d4404b7b617965b`;
  - POPGym cell:
    `d45e97cd9fc606372c44c885892614c79d7f422c550277a0b7d0935807f475e4`;
    and
  - standalone cell:
    `7ba92a52e7ec4d75f2b8aab09ea463324b3b550dff597747df1e524aa75c0146`.
- Failure diagnosis: fixed-seed instrumented replays of the preregistered
  T-Maze configuration showed healthy uniform addresses. Each 1,000-step
  collection selected 127 or 128 of the 128 rows. The maximum absolute
  recurrent states for seeds `1103`, `2207`, and `3301` reached
  `20,814,988`, `40,824,856`, and `50,378.0742`, respectively. Maximum raw
  gradient norms reached `269,777,696`, `5.100072337408e12`, and
  `1,160,457.125`. These observations rule out collapsed addressing and show
  that the forward recurrence can expand by many orders of magnitude before
  learner-side gradient clipping is applied.
- Diagnostic artifact: local file
  `E:\data\code\claudecode\arcmind\tmp\shm-diagnostic.json`, SHA256
  `47aad8059e121a73467a2834dae45a9179a0012f987bf50b44fb1685844fc5d7`.
- Repair: commit
  [`bd43c6cda88d493df318c3c5ca6bf7e22da60279`](https://github.com/jemsbhai/arcmind/commit/bd43c6cda88d493df318c3c5ca6bf7e22da60279)
  applies the official POMDP cell's `[-100, 100]` forward-state clamp only in
  `paper_uniform` mode. The `v1_1_popgym_compat` mode remains unclamped and
  retains row-zero addressing. The serialized policy-core contract exposes
  this mode difference.
- Repair validation: source-equation, mode-serialization, long-sequence,
  finite-gradient, and unclamped-compatibility tests passed. A direct replay
  of seed `1103` then completed all 250,000 transitions with finite optimizer
  metrics and achieved mean return `4.0` over 128 evaluation episodes. The
  replay artifact is
  `E:\data\code\claudecode\arcmind\tmp\shm-fix-seed1103.json`, SHA256
  `43547f8a1771034957d71d3325a447a0fa8d1e887962d7176b25ff28b9089284`.
- Interpretation: the historical failure is consistent with forward
  recurrence overflow, not address collapse or optimizer-input corruption.
  The repair is source-faithful to the official POMDP implementation while
  preserving an exact POPGym compatibility mode.
- Evidence restriction: the direct replay is a diagnostic, not registered
  performance evidence. All three SHM seeds must be rerun under a new frozen
  manifest before any comparison is aggregated or used in the paper.

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
