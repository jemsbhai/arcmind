# ArcMind Compute Budget

Status: frozen pre-performance planning estimate, dated 2026-07-24, with an
outcome-independent A100 execution amendment dated 2026-08-11.

This document converts the completed local timing gates into the planned
compute for the exact compute-aware study. It is a resource plan, not
performance evidence. Calibration returns are excluded from model incidence,
learner selection, stopping, budgeting, and claims.

## 1. Fixed study inventory

The tuning matrix has 13 model families, three learning rates, three
development seeds, and two tasks. Both tuning tasks use 1,000,000 environment
steps:

`13 * 3 * 3 * 2 = 234 cells`

The registered primary matrix has exact sparse task incidence:

| Task | Models | Seeds | Cells | Steps per cell |
|---|---:|---:|---:|---:|
| T-Maze | 15 | 10 | 150 | 1,000,000 |
| RockSample | 18 | 10 | 180 | 5,000,000 |
| Battleship | 8 | 10 | 80 | 10,000,000 |
| Navix | 8 | 10 | 80 | 10,000,000 |
| Total | 49 task-model groups | | 490 | |

The separately labeled upper-reference matrix has one memoryless policy,
four task aliases, and the same ten final seeds, for 40 cells. The complete
planned inventory is therefore 764 cells:

| Stage | Cells | Environment steps |
|---|---:|---:|
| Development tuning | 234 | 234,000,000 |
| Registered primary | 490 | 2,650,000,000 |
| Upper references | 40 | 260,000,000 |
| Total | 764 | 3,144,000,000 |

No performance-based pruning, stopping, task removal, or model removal is
permitted after the study begins.

## 2. Timing basis

All timing gates used the pinned POBAX environment, JAX and `jaxlib` 0.6.2,
and an NVIDIA GeForce RTX 4090 Laptop GPU. Each measurement trained for
128,000 exact environment steps. Planning scales each measured training time
linearly by requested environment steps.

This extrapolation is conservative with respect to fixed compilation
overhead, but it is not a wall-clock guarantee. The runtime manifest does not
record accelerator clocks, power limits, temperature, or other system load.

For models without a direct measurement on a longer task, the plan multiplies
the T-Maze rate by 1.4. The largest measured RockSample to T-Maze ratio in the
representative two-task gate was 1.384162, so 1.4 is a rounded conservative
proxy. Direct task measurements replace the proxy where they exist.

ArcMind has four comparable clean T-Maze measurements:

`203.447793, 217.655288, 381.072660, 648.643372 seconds`

The central estimate uses their median, 299.363974 seconds. Three comparable
RockSample measurements are 216.140067, 226.309223, and 272.094813 seconds;
their median is 226.309223 seconds. The sensitivity estimate applies the
worst observed ArcMind T-Maze rate, 648.643372 seconds, together with the 1.4
task proxy across every ArcMind task. It does not average that observation
away.

The source-compatible and RockSample ablation rates are direct measurements:

| Block | Measured 128,000-step seconds |
|---|---:|
| Official Memory Traces lane | 43.291716 |
| AGaLiTe source-compatible lane | 470.897213 |
| SSM-only ablation | 663.392761 |
| Unordered-recall ablation | 857.056704 |
| No-memory ablation | 212.879527 |
| No-SSM ablation | 130.856833 |
| No-gate ablation | 198.080019 |

The exact integrity hashes and raw locations for every timing input are
recorded in finding F-ENG-013 in `docs/experimental_findings.md`.

## 3. Training-time estimate

The estimates below are single-device training hours. They exclude final
artifact aggregation, plotting, paper compilation, and time lost to a failed
process. The runner is resumable at exact validated cell boundaries.

| Study block | Central hours | Worst-rate sensitivity hours |
|---|---:|---:|
| Development tuning | 71.892 | 92.029 |
| Seven common non-ArcMind models, four tasks | 756.553 | 756.553 |
| ArcMind, four tasks | 144.832 | 506.753 |
| FFM, SHM, LRU, S4D, Transformer-XL, two tasks | 53.817 | 53.817 |
| Two source-compatible lanes, T-Maze | 11.159 | 11.159 |
| Five ArcMind ablations, RockSample | 223.770 | 223.770 |
| Four upper references | 9.244 | 9.244 |
| Total | 1,271.267 | 1,653.326 |

The central estimate is 52.97 uninterrupted device-days. The worst-rate
sensitivity is 68.89 uninterrupted device-days. A scheduling reserve of 20
percent gives 1,525.521 to 1,983.991 device-hours, or 63.56 to 82.67
device-days. The reserve is an operational allowance, not additional
experimental cells.

## 4. Dollar budget

Planned external compute spend remains USD 0 because the collaborator's A100
allocation is in-kind. The USD 10 cap remains unspent and reserved for a small
emergency compatibility check only. Electricity, machine depreciation,
researcher time, and in-kind cluster value are not converted to a
cloud-equivalent dollar figure.

Any paid run must be recorded here before launch with provider, instance,
region, duration, purpose, and exact cost. Paid performance expansion beyond
the frozen matrices requires a new pre-performance registration and cannot be
silently merged into the primary evidence.

## 5. 2026-08-11 A100 execution amendment

The local timing measurements and device-hour totals above remain the frozen
planning basis; they are RTX 4090 Laptop measurements, not claims about A100
speed. Before any tuning return or ranking was inspected, the laptop run was
paused at 55 of 234 cells for capacity reasons. The A100 lineage excludes
those cells and restarts the complete 234-cell tuning matrix.

Execution is reassigned to four homogeneous 40 GB A100s, one JAX-visible GPU
per worker. Dividing the original single-device estimate equally across four
workers gives an idealized wall-time range of 317.817 to 413.332 hours, or
13.24 to 17.22 uninterrupted days. Applying the existing 20 percent reserve
gives 381.380 to 495.998 hours, or 15.89 to 20.67 days. This is a scheduling
bound derived from laptop rates, not an A100 runtime forecast. The
compatibility preflight is not a timing calibration. A cluster-specific
replacement may be recorded only from a fixed, return-blinded timing procedure
declared before reading its elapsed times; it cannot use returns, rankings, or
selective stopping. Queue delay, preemption, uneven per-cell costs,
aggregation, and transfer time remain additional operational factors.

The collaborator must record the cluster/site, allocation or project ID,
Slurm job IDs, A100 device identity, calibration timing, elapsed stage times,
and whether the allocation remained in-kind. Any monetary charge must be
entered under the dollar budget before launch.
