#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

[[ $# -eq 2 ]] || cluster_die "usage: $0 /outside/repo/cluster.env tuning|primary|upper"
cluster_require_command git
cluster_load_config "$1"
cluster_set_stage "$2"
cluster_assert_clean_repo
cluster_assert_registration
python="$(cluster_python)"
[[ -s "$ARCMIND_APPROVED_RUNTIME" ]] || cluster_die \
    "approved runtime is missing; complete check_preflight.sh first"
cluster_require_command flock
mkdir -p -- "${ARCMIND_STAGE_ROOT}/locks"
exec 9>"${ARCMIND_STAGE_ROOT}/locks/finalize.lock"
flock -n 9 || cluster_die "${ARCMIND_STAGE} finalization is already running"

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
    raise SystemExit("finalizer provenance/runtime differs from the approved four-A100 preflight")
PY

merge_args=(
    --registration "$ARCMIND_REGISTRATION"
    --output-root "$ARCMIND_CANONICAL_ROOT"
)
for shard_index in 0 1 2 3; do
    shard_root="${ARCMIND_SHARDS_ROOT}/shard-${shard_index}"
    [[ -d "$shard_root" ]] || cluster_die "missing shard root: ${shard_root}"
    merge_args+=(--shard-root "$shard_root")
done

"$python" -m benchmarks.pobax.merge_matrix_shards "${merge_args[@]}"
mkdir -p -- "$(dirname -- "$ARCMIND_AGGREGATE")"

case "$ARCMIND_STAGE" in
    tuning)
        "$python" -m benchmarks.pobax.aggregate_development \
            "$ARCMIND_CANONICAL_ROOT" "$ARCMIND_AGGREGATE"
        ;;
    primary|upper)
        "$python" -m benchmarks.pobax.aggregate_registered \
            "${ARCMIND_CANONICAL_ROOT}/frozen_manifest.json" "$ARCMIND_AGGREGATE"
        ;;
esac

if [[ "$ARCMIND_STAGE" == upper ]]; then
    "$python" -m benchmarks.pobax.link_upper_reference \
        "${ARCMIND_REPO_ROOT}/benchmark_results/pobax/compute-aware-primary-v1" \
        "$ARCMIND_CANONICAL_ROOT" \
        "${ARCMIND_REPO_ROOT}/benchmark_results/pobax/aggregates/compute-aware-primary-upper-v1.json"
fi

printf 'finalized-%s:%s\n' "$ARCMIND_STAGE" "$ARCMIND_CANONICAL_ROOT"
printf 'aggregate:%s\n' "$ARCMIND_AGGREGATE"
