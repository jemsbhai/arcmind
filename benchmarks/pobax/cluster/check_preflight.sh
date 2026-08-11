#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

[[ $# -eq 1 ]] || cluster_die "usage: $0 /outside/repo/cluster.env"
cluster_require_command sha256sum
cluster_load_config "$1"
cluster_assert_clean_repo
python="$(cluster_python)"

for shard_index in 0 1 2 3; do
    [[ -s "${ARCMIND_PREFLIGHT_ROOT}/shard-${shard_index}.json" ]] || cluster_die \
        "missing preflight record for shard ${shard_index}"
    [[ -s "${ARCMIND_PREFLIGHT_ROOT}/shard-${shard_index}.smoke.json" ]] || cluster_die \
        "missing smoke output for shard ${shard_index}"
done

"$python" - "$ARCMIND_PREFLIGHT_ROOT" "$ARCMIND_APPROVED_RUNTIME" <<'PY'
import json
import os
import tempfile
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
records = [json.loads((root / f"shard-{index}.json").read_text()) for index in range(4)]
reference = records[0]
for index, record in enumerate(records[1:], start=1):
    if record != reference:
        raise SystemExit(f"preflight provenance/runtime differs between shard 0 and shard {index}")

content = (
    json.dumps(reference, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    + "\n"
).encode()
if output.exists():
    if output.read_bytes() != content:
        raise SystemExit(f"refusing to replace a different approved runtime: {output}")
else:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
print(reference["runtime_contract"]["devices"][0]["device_kind"])
PY

printf 'approved-runtime:'
sha256sum "$ARCMIND_APPROVED_RUNTIME" | cut -d ' ' -f 1
