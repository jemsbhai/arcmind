#!/usr/bin/env bash
set -euo pipefail

: "${ARCMIND_CLUSTER_CONFIG:?submit through submit_stage.sh}"
: "${ARCMIND_STAGE:?submit through submit_stage.sh}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"
cluster_load_config "$ARCMIND_CLUSTER_CONFIG"
cluster_set_stage "$ARCMIND_STAGE"
cluster_assert_clean_repo
cluster_assert_registration
cluster_require_command flock

[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-3]$ ]] || cluster_die \
    "SLURM_ARRAY_TASK_ID must be one of 0, 1, 2, or 3"
shard_index="$SLURM_ARRAY_TASK_ID"
python="$(cluster_python)"
[[ -s "$ARCMIND_APPROVED_RUNTIME" ]] || cluster_die \
    "approved runtime is missing; complete check_preflight.sh first"

mkdir -p -- "$ARCMIND_SHARDS_ROOT" "${ARCMIND_STAGE_ROOT}/locks"
lock_path="${ARCMIND_STAGE_ROOT}/locks/shard-${shard_index}.lock"
exec 9>"$lock_path"
flock -n 9 || cluster_die "matrix shard ${shard_index} is already running"

cd -- "$ARCMIND_REPO_ROOT"
"$python" - "$ARCMIND_APPROVED_RUNTIME" <<'PY'
import json
import os
from pathlib import Path
import sys

from benchmarks.pobax.implementation_provenance import gather_implementation_source
from benchmarks.pobax.registered_artifacts import dependency_lock_sha256, gather_git_provenance
from benchmarks.pobax.run_pilot import runtime_contract
from benchmarks.pobax.smoke_environment import source_commit

repo = Path(os.environ["ARCMIND_REPO_ROOT"]).resolve()
actual = {
    "schema_version": 1,
    "git": gather_git_provenance(repo),
    "dependency_lock_sha256": dependency_lock_sha256(
        repo / "benchmarks" / "pobax" / "requirements-lock.txt"
    ),
    "pobax_commit": source_commit("pobax"),
    "navix_commit": source_commit("Navix"),
    "implementation_source": gather_implementation_source(repo),
    "runtime_contract": runtime_contract(),
}
approved = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if actual != approved:
    raise SystemExit("worker provenance/runtime differs from the approved four-A100 preflight")
PY

shard_root="${ARCMIND_SHARDS_ROOT}/shard-${shard_index}"
exec "$python" -m benchmarks.pobax.run_matrix \
    --registration "$ARCMIND_REGISTRATION" \
    --output-root "$shard_root" \
    --shard-count "$ARCMIND_SHARD_COUNT" \
    --shard-index "$shard_index"
