#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

[[ $# -eq 1 ]] || cluster_die "usage: $0 /outside/repo/cluster.env"
cluster_require_command sbatch
cluster_require_command git
cluster_load_config "$1"
cluster_assert_clean_repo

mkdir -p -- "$ARCMIND_PREFLIGHT_ROOT/slurm"
export ARCMIND_CLUSTER_CONFIG

sbatch_args=(
    --parsable
    --array="0-3%4"
    --nodes=1
    --ntasks=1
    --gres="${ARCMIND_SLURM_GRES:-gpu:a100:1}"
    --cpus-per-task="${ARCMIND_SLURM_CPUS:-8}"
    --mem="${ARCMIND_SLURM_MEMORY:-64G}"
    --time="${ARCMIND_SLURM_TIME_PREFLIGHT:-00:30:00}"
    --job-name=arcmind-preflight
    --chdir="$ARCMIND_REPO_ROOT"
    --output="${ARCMIND_PREFLIGHT_ROOT}/slurm/%A_%a.out"
    --error="${ARCMIND_PREFLIGHT_ROOT}/slurm/%A_%a.err"
    --export=ALL
)
[[ -n "${ARCMIND_SLURM_ACCOUNT:-}" ]] && sbatch_args+=(--account="$ARCMIND_SLURM_ACCOUNT")
[[ -n "${ARCMIND_SLURM_PARTITION:-}" ]] && sbatch_args+=(--partition="$ARCMIND_SLURM_PARTITION")
[[ -n "${ARCMIND_SLURM_QOS:-}" ]] && sbatch_args+=(--qos="$ARCMIND_SLURM_QOS")
[[ -n "${ARCMIND_SLURM_CONSTRAINT:-}" ]] && sbatch_args+=(--constraint="$ARCMIND_SLURM_CONSTRAINT")

job_id="$(sbatch "${sbatch_args[@]}" "${script_dir}/preflight.sbatch")"
printf 'submitted-preflight:%s\n' "$job_id"
printf 'after all four tasks finish, run: %s %q\n' "${script_dir}/check_preflight.sh" "$ARCMIND_CLUSTER_CONFIG"
