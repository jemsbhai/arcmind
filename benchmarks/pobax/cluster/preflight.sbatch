#!/usr/bin/env bash
set -euo pipefail

: "${ARCMIND_CLUSTER_CONFIG:?submit through submit_preflight.sh}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"
cluster_load_config "$ARCMIND_CLUSTER_CONFIG"
cluster_assert_clean_repo
cluster_require_command flock

[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-3]$ ]] || cluster_die \
    "SLURM_ARRAY_TASK_ID must be one of 0, 1, 2, or 3"
shard_index="$SLURM_ARRAY_TASK_ID"
python="$(cluster_python)"
mkdir -p -- "$ARCMIND_PREFLIGHT_ROOT"

lock_path="${ARCMIND_PREFLIGHT_ROOT}/shard-${shard_index}.lock"
exec 9>"$lock_path"
flock -n 9 || cluster_die "preflight shard ${shard_index} is already running"

cd -- "$ARCMIND_REPO_ROOT"
smoke_tmp="${ARCMIND_PREFLIGHT_ROOT}/.shard-${shard_index}.smoke.$$.tmp"
record_tmp="${ARCMIND_PREFLIGHT_ROOT}/.shard-${shard_index}.record.$$.tmp"
trap 'rm -f -- "$smoke_tmp" "$record_tmp"' EXIT

"$python" -m benchmarks.pobax.smoke_environment --require-gpu > "$smoke_tmp"
"$python" - "$record_tmp" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

from benchmarks.pobax.implementation_provenance import gather_implementation_source
from benchmarks.pobax.registered_artifacts import dependency_lock_sha256, gather_git_provenance
from benchmarks.pobax.run_pilot import runtime_contract
from benchmarks.pobax.smoke_environment import source_commit

repo = Path(os.environ["ARCMIND_REPO_ROOT"]).resolve()
runtime = runtime_contract()
devices = runtime["devices"]
if runtime["jax_backend"] != "gpu" or len(devices) != 1:
    raise SystemExit(f"expected exactly one JAX GPU, found {runtime!r}")
device_kind = devices[0]["device_kind"]
pattern = os.environ["ARCMIND_DEVICE_KIND_REGEX"]
if re.search(pattern, device_kind) is None:
    raise SystemExit(
        f"JAX device_kind {device_kind!r} does not match required expression {pattern!r}"
    )

git = gather_git_provenance(repo)
if git["dirty"] or git["commit"] != os.environ["ARCMIND_EXPECTED_COMMIT"]:
    raise SystemExit(f"Git provenance drifted: {git!r}")

record = {
    "schema_version": 1,
    "git": git,
    "dependency_lock_sha256": dependency_lock_sha256(
        repo / "benchmarks" / "pobax" / "requirements-lock.txt"
    ),
    "pobax_commit": source_commit("pobax"),
    "navix_commit": source_commit("Navix"),
    "implementation_source": gather_implementation_source(repo),
    "runtime_contract": runtime,
}
Path(sys.argv[1]).write_text(
    json.dumps(record, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    + "\n",
    encoding="utf-8",
)
PY

mv -f -- "$smoke_tmp" "${ARCMIND_PREFLIGHT_ROOT}/shard-${shard_index}.smoke.json"
mv -f -- "$record_tmp" "${ARCMIND_PREFLIGHT_ROOT}/shard-${shard_index}.json"
trap - EXIT
printf 'preflight-complete:%s\n' "$shard_index"
