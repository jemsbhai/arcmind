#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

[[ $# -eq 2 ]] || cluster_die "usage: $0 /outside/repo/cluster.env tuning|primary|upper"
cluster_require_command sbatch
cluster_require_command git
cluster_load_config "$1"
cluster_set_stage "$2"
cluster_assert_clean_repo
cluster_assert_registration

bash "${script_dir}/check_preflight.sh" "$ARCMIND_CLUSTER_CONFIG"
if [[ -f "${ARCMIND_CANONICAL_ROOT}/completion_index.json" ]]; then
    cluster_die "canonical ${ARCMIND_STAGE} matrix is already complete"
fi

mkdir -p -- "$ARCMIND_SHARDS_ROOT" "$ARCMIND_SLURM_LOG_ROOT"
export ARCMIND_CLUSTER_CONFIG ARCMIND_STAGE

time_variable="ARCMIND_SLURM_TIME_${ARCMIND_STAGE^^}"
stage_time="${!time_variable:-2-00:00:00}"
sbatch_args=(
    --parsable
    --array="0-3%4"
    --nodes=1
    --ntasks=1
    --gres="${ARCMIND_SLURM_GRES:-gpu:a100:1}"
    --cpus-per-task="${ARCMIND_SLURM_CPUS:-8}"
    --mem="${ARCMIND_SLURM_MEMORY:-64G}"
    --time="$stage_time"
    --job-name="arcmind-${ARCMIND_STAGE}"
    --chdir="$ARCMIND_REPO_ROOT"
    --output="${ARCMIND_SLURM_LOG_ROOT}/%A_%a.out"
    --error="${ARCMIND_SLURM_LOG_ROOT}/%A_%a.err"
    --export=ALL
)
[[ -n "${ARCMIND_SLURM_ACCOUNT:-}" ]] && sbatch_args+=(--account="$ARCMIND_SLURM_ACCOUNT")
[[ -n "${ARCMIND_SLURM_PARTITION:-}" ]] && sbatch_args+=(--partition="$ARCMIND_SLURM_PARTITION")
[[ -n "${ARCMIND_SLURM_QOS:-}" ]] && sbatch_args+=(--qos="$ARCMIND_SLURM_QOS")
[[ -n "${ARCMIND_SLURM_CONSTRAINT:-}" ]] && sbatch_args+=(--constraint="$ARCMIND_SLURM_CONSTRAINT")

job_id="$(sbatch "${sbatch_args[@]}" "${script_dir}/matrix_shard.sbatch")"
printf 'submitted-%s:%s\n' "$ARCMIND_STAGE" "$job_id"
printf 'shards:%s\n' "$ARCMIND_SHARDS_ROOT"
printf 'resubmit the same command after timeout or preemption; validated cells are reused\n'
