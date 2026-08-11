#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

[[ $# -eq 3 ]] || cluster_die \
    "usage: $0 /outside/repo/cluster.env tuning|primary|upper /handoff/directory"
cluster_require_command git
cluster_require_command sha256sum
cluster_require_command tar
cluster_load_config "$1"
cluster_set_stage "$2"
cluster_assert_clean_repo
python="$(cluster_python)"

destination="$(cluster_realpath "$3")"
cluster_assert_outside_repo "handoff destination" "$destination"
[[ -f "${ARCMIND_CANONICAL_ROOT}/completion_index.json" ]] || cluster_die \
    "canonical matrix is not finalized: ${ARCMIND_CANONICAL_ROOT}"
[[ -f "$ARCMIND_AGGREGATE" ]] || cluster_die \
    "canonical aggregate is missing: ${ARCMIND_AGGREGATE}"
[[ -f "$ARCMIND_APPROVED_RUNTIME" ]] || cluster_die "approved runtime is missing"

"$python" - "$ARCMIND_APPROVED_RUNTIME" \
    "${ARCMIND_CANONICAL_ROOT}/frozen_manifest.json" <<'PY'
import json
from pathlib import Path
import sys

approved = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if approved.pop("schema_version", None) != 1:
    raise SystemExit("unsupported approved-runtime schema")
if approved != manifest.get("provenance"):
    raise SystemExit("approved runtime does not match frozen-manifest provenance")
PY

cd -- "$ARCMIND_REPO_ROOT"
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
    upper_registration="${ARCMIND_REPO_ROOT}/benchmark_results/pobax/registrations/compute-aware-upper-v1.json"
    upper_link="${ARCMIND_REPO_ROOT}/benchmark_results/pobax/aggregates/compute-aware-primary-upper-v1.json"
    [[ -f "$upper_registration" ]] || cluster_die \
        "schema-7 registration is missing: ${upper_registration}"
    [[ -f "$upper_link" ]] || cluster_die "primary/upper link is missing: ${upper_link}"
    "$python" -m benchmarks.pobax.link_upper_reference \
        "${ARCMIND_REPO_ROOT}/benchmark_results/pobax/compute-aware-primary-v1" \
        "$ARCMIND_CANONICAL_ROOT" \
        "$upper_link"
fi

mkdir -p -- "$destination"
prefix="arcmind-${ARCMIND_STAGE}-${ARCMIND_EXPECTED_COMMIT}"
archive="${destination}/${prefix}.tar.gz"
commit_file="${destination}/${prefix}.commit"
runtime_file="${destination}/${prefix}.runtime.json"
checksum_file="${destination}/${prefix}.sha256"
for output in "$archive" "$commit_file" "$runtime_file" "$checksum_file"; do
    [[ ! -e "$output" ]] || cluster_die "refusing to replace existing handoff file: ${output}"
done

paths=("$ARCMIND_RAW_REL" "$ARCMIND_AGGREGATE_REL")
if [[ "$ARCMIND_STAGE" == upper ]]; then
    paths+=(
        "benchmark_results/pobax/registrations/compute-aware-upper-v1.json"
        "benchmark_results/pobax/aggregates/compute-aware-primary-upper-v1.json"
    )
fi

tar --create --gzip --file "$archive" --directory "$ARCMIND_REPO_ROOT" -- "${paths[@]}"
printf '%s\n' "$ARCMIND_EXPECTED_COMMIT" > "$commit_file"
cp -- "$ARCMIND_APPROVED_RUNTIME" "$runtime_file"
(
    cd -- "$destination"
    sha256sum "${prefix}.tar.gz" "${prefix}.commit" "${prefix}.runtime.json" \
        > "${prefix}.sha256"
    sha256sum --check "${prefix}.sha256"
)
printf 'handoff:%s\n' "$archive"
printf 'checksums:%s\n' "$checksum_file"
