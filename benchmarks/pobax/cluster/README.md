# Four-A100 Slurm handoff

This directory is the operational handoff for running the frozen compute-aware
study on four homogeneous 40 GB A100s. It preserves the repository's existing
provenance rules: one JAX-visible GPU per process, identical runtime contracts,
clean Git state, isolated shard outputs, and a single validated finalization
step that creates the canonical raw matrix.

The 55 laptop tuning cells are an incomplete RTX 4090 Laptop attempt. Archive
them separately and do not copy them into any A100 shard or canonical matrix.
The A100 lineage restarts all 234 tuning cells.

## Non-negotiable execution contract

- Use a clean, shared-filesystem clone at the exact 40-character commit named
  in the cluster configuration. Detached HEAD is recommended.
- Use one unchanged Python 3.12 virtual environment installed from
  `benchmarks/pobax/requirements-lock.txt` for tuning, primary, and upper
  execution. The lock pins JAX 0.6.2 plus the exact POBAX and Navix commits.
- Request one A100 per array task. Do not allocate four visible GPUs to one
  Python process. The preflight and every worker require exactly one JAX GPU.
- Use four deterministic index-modulo shards. Each shard has its own
  `shard_completion.json` and `shard_checksums.sha256` in place of the
  canonical indexes; only the separate merger may create the canonical
  `completion_index.json` and `checksums.sha256`.
- Keep the venv, personalized configuration, Slurm output, shard roots, and
  transfer archives outside the Git worktree. `benchmark_results/` is ignored
  and reserved for canonical raw matrices, registrations, and aggregates.
- Do not edit tracked Python or the dependency lock after A100 tuning starts.
  Schema 6 must match tuning's implementation source and complete runtime
  contract, including the exact one-element device list. After primary starts,
  do not change the Git commit or any runtime field through schema 7 completion
  and linking.

The pip lock is version-pinned but not hash-pinned. The runner compensates by
recording its file SHA256, the selected installed package versions, the VCS
commits, and the complete tracked Python implementation manifest. Do not
substitute a cluster module's JAX build if the lock cannot be installed. A
driver incompatibility requires resolving and freezing a new environment
before performance execution begins.

## 1. Receive and pin the source

Prefer a pushed reviewable commit. For an offline transfer, the maintainer can
send a Git bundle plus its checksum:

```bash
git bundle create arcmind-cluster.bundle HEAD
sha256sum arcmind-cluster.bundle > arcmind-cluster.bundle.sha256
```

On the cluster:

```bash
sha256sum --check arcmind-cluster.bundle.sha256
git clone arcmind-cluster.bundle /shared/projects/arcmind/repo
git -C /shared/projects/arcmind/repo checkout --detach EXPECTED_FULL_COMMIT
git -C /shared/projects/arcmind/repo status --porcelain=v1 --untracked-files=all
```

The final command must print nothing. Copy `cluster.env.example` to a location
outside the clone, fill in the absolute shared paths and Slurm routing, and set
`ARCMIND_EXPECTED_COMMIT` to the detached commit. Do not put a personalized
configuration in the repository; any untracked file makes registered launch
fail closed.

## 2. Build the locked environment

Run this once on a node that has network access to PyPI and GitHub, or against
your site's populated pip cache:

```bash
bash benchmarks/pobax/cluster/setup_env.sh /shared/config/arcmind-cluster.env
```

`ARCMIND_VENV` and `ARCMIND_CLUSTER_ROOT` must be shared with every compute
node and must be outside the clone. The setup records `pip freeze`, the Python
patch version, and the lock checksum under the external cluster root. It also
runs `pip check`. Reuse this venv unchanged for every registered stage.

## 3. Prove the four worker runtimes

Submit a four-way GPU preflight:

```bash
bash benchmarks/pobax/cluster/submit_preflight.sh /shared/config/arcmind-cluster.env
```

After all four array tasks succeed, approve their common contract:

```bash
bash benchmarks/pobax/cluster/check_preflight.sh /shared/config/arcmind-cluster.env
```

Each task runs the repository GPU/environment smoke test and records the exact
Git, implementation-source, dependency, POBAX, Navix, Python, package, JAX,
backend, and device contract. Approval requires byte-equivalent JSON across
all four allocations and a `device_kind` matching the configured 40 GB A100
expression. Every later worker recomputes and compares this contract before it
describes or trains a cell.

Run a fresh preflight after checking out the schema-6 commit. Primary and upper
must then reuse that same approved commit/runtime.

## 4. Execute one stage

The stage names, registrations, and canonical outputs are fixed:

| Stage | Registration | Cells | Canonical raw root |
|---|---|---:|---|
| `tuning` | `benchmarks/pobax/manifests/compute_aware_tuning_v1.json` | 234 | `benchmark_results/pobax/compute-aware-tuning-v1` |
| `primary` | `benchmarks/pobax/manifests/compute_aware_final_v1.json` | 490 | `benchmark_results/pobax/compute-aware-primary-v1` |
| `upper` | `benchmark_results/pobax/registrations/compute-aware-upper-v1.json` | 40 | `benchmark_results/pobax/compute-aware-upper-v1` |

Submit, for example, tuning:

```bash
bash benchmarks/pobax/cluster/submit_stage.sh /shared/config/arcmind-cluster.env tuning
```

The Slurm array is `0-3%4`; every task requests one A100 and writes only to
`$ARCMIND_CLUSTER_ROOT/$COMMIT/tuning/shards/shard-$INDEX`. File locks reject
overlapping submissions for the same shard. If a task is preempted, times out,
or fails, inspect its external Slurm log and resubmit the identical command.
The launcher resumes only identity-, configuration-, manifest-, and
provenance-compatible completed cells. It does not checkpoint within a cell.
The deterministic shard counts are 59/59/58/58 for tuning,
123/123/122/122 for primary, and 10/10/10/10 for upper references.
Each root self-identifies its shard count, shard index, and partition algorithm
inside `shard_completion.json`; root names and the order of repeated
`--shard-root` arguments do not establish identity.

Do not submit a second copy while an earlier array is still running. Helpful
read-only checks are:

```bash
squeue --me --name arcmind-tuning
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
```

Once all four shards exit successfully, submit the single-GPU merge and
canonical aggregation job from the same clean clone and venv:

```bash
bash benchmarks/pobax/cluster/submit_finalize.sh /shared/config/arcmind-cluster.env tuning
```

The merger rejects missing/overlapping cells, different frozen manifests,
different complete runtime provenance, duplicate/missing self-identified shard
indexes, unexpected files, or invalid artifacts/logs. Before copying anything,
it validates each source root's canonical `shard_completion.json` and
`shard_checksums.sha256`. It then creates the one complete
`completion_index.json` and `checksums.sha256` under the canonical raw root.
The aggregate is outside that immutable raw tree. The finalizer requests one
A100 because rebuilding the exact frozen manifest must observe the same
approved runtime, including its device list; a CPU-only login-node
finalization is not valid.

## 5. Hand back at each frozen boundary

Create a minimal stage archive outside the clone:

```bash
bash benchmarks/pobax/cluster/package_stage.sh \
  /shared/config/arcmind-cluster.env tuning /shared/handoff
```

Send all four emitted sidecars: `.tar.gz`, `.commit`, `.runtime.json`, and
`.sha256`. The receiver first checks the transport hashes in the handoff
directory:

```bash
sha256sum --check arcmind-tuning-COMMIT.sha256
test "$(cat arcmind-tuning-COMMIT.commit)" = "$(git rev-parse HEAD)"
tar --list --gzip --file arcmind-tuning-COMMIT.tar.gz
```

Review the listing before extraction. Extract into a clean clone that does not
already contain that stage; `--keep-old-files` prevents replacement:

```bash
tar --extract --gzip --keep-old-files \
  --directory /path/to/clean/arcmind \
  --file arcmind-tuning-COMMIT.tar.gz
```

Now prove that the approved worker sidecar is exactly the provenance embedded
in the returned frozen manifest. Substitute the matching stage filename and
raw path for primary or upper handoffs:

```bash
python - arcmind-tuning-COMMIT.runtime.json \
  benchmark_results/pobax/compute-aware-tuning-v1/frozen_manifest.json <<'PY'
import json
from pathlib import Path
import sys

approved = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if approved.pop("schema_version", None) != 1:
    raise SystemExit("unsupported approved-runtime schema")
if approved != manifest.get("provenance"):
    raise SystemExit("handoff runtime does not match frozen-manifest provenance")
print("runtime-provenance:verified")
PY
```

Then rerun the relevant canonical validator. Atomic writers reuse only
byte-identical aggregates and reject drift:

```bash
# Tuning
python -m benchmarks.pobax.aggregate_development \
  benchmark_results/pobax/compute-aware-tuning-v1 \
  benchmark_results/pobax/aggregates/compute-aware-tuning-v1.json

# Primary
python -m benchmarks.pobax.aggregate_registered \
  benchmark_results/pobax/compute-aware-primary-v1/frozen_manifest.json \
  benchmark_results/pobax/aggregates/compute-aware-primary-v1.json

# Upper (after retaining the validated primary inputs)
python -m benchmarks.pobax.aggregate_registered \
  benchmark_results/pobax/compute-aware-upper-v1/frozen_manifest.json \
  benchmark_results/pobax/aggregates/compute-aware-upper-v1.json
python -m benchmarks.pobax.link_upper_reference \
  benchmark_results/pobax/compute-aware-primary-v1 \
  benchmark_results/pobax/compute-aware-upper-v1 \
  benchmark_results/pobax/aggregates/compute-aware-primary-upper-v1.json
```

Do not inspect cell returns or candidate rankings during this handoff. The
maintainer performs the mechanical aggregate/materialization boundary:

1. After `tuning`, validate the raw tree and aggregate, materialize the actual
   13 winners into schema 6, commit only the canonical schema-6 registration,
   and send that new clean commit to the collaborator. The collaborator keeps
   the canonical tuning raw tree/aggregate, checks out the new commit, updates
   `ARCMIND_EXPECTED_COMMIT`, reruns the four-GPU preflight, and launches
   `primary`.
2. After `primary`, validate its raw tree/aggregate and materialize schema 7 at
   `benchmark_results/pobax/registrations/compute-aware-upper-v1.json`. Send
   that exact file and a SHA256 back to the collaborator. Schema 7 is ignored
   data, so both primary and upper remain on the identical clean schema-6 Git
   commit and approved runtime.
3. Launch and finalize `upper`. Finalization also creates the validated
   primary/upper link. Package the upper stage and return it; the receiver must
   already retain the tuning and primary handoffs because schema 6, schema 7,
   and the linker intentionally rebuild their upstream evidence.

The laptop partial remains a separately labeled incomplete attempt throughout.
Never merge, rename, or copy its 55 artifacts into this lineage.

For the schema-7 return trip, verify before placing the ignored registration:

```bash
sha256sum --check compute-aware-upper-v1.json.sha256
mkdir -p benchmark_results/pobax/registrations
cp --no-clobber compute-aware-upper-v1.json \
  benchmark_results/pobax/registrations/compute-aware-upper-v1.json
git status --porcelain=v1 --untracked-files=all  # must still print nothing
```
