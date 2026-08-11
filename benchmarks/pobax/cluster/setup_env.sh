#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

[[ $# -eq 1 ]] || cluster_die "usage: $0 /outside/repo/cluster.env"
cluster_require_command git
cluster_require_command realpath
cluster_require_command sha256sum
ARCMIND_ALLOW_MISSING_VENV=1
cluster_load_config "$1"

[[ -d "$ARCMIND_REPO_ROOT" ]] || cluster_die "repository does not exist"
cluster_assert_clean_repo

bootstrap="${ARCMIND_PYTHON_BOOTSTRAP:-python3.12}"
cluster_require_command "$bootstrap"
mkdir -p -- "$(dirname -- "$ARCMIND_VENV")" "$ARCMIND_CLUSTER_ROOT/environment"

if [[ ! -x "$ARCMIND_VENV/bin/python" ]]; then
    "$bootstrap" -m venv "$ARCMIND_VENV"
fi

python="$(cluster_python)"
lock="${ARCMIND_REPO_ROOT}/benchmarks/pobax/requirements-lock.txt"
"$python" -m pip install -r "$lock"
"$python" -m pip check
"$python" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

record_root="${ARCMIND_CLUSTER_ROOT}/environment"
"$python" -m pip freeze --all > "${record_root}/pip-freeze.txt"
"$python" --version > "${record_root}/python-version.txt" 2>&1
sha256sum "$lock" > "${record_root}/requirements-lock.sha256"

printf 'environment-ready:%s\n' "$ARCMIND_VENV"
printf 'requirements-lock:'
sha256sum "$lock" | cut -d ' ' -f 1
