#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

[[ $# -eq 2 ]] || cluster_die "usage: $0 /outside/repo/cluster.env tuning|primary|upper"
cluster_require_command sbatch
cluster_load_config "$1"
cluster_set_stage "$2"
cluster_assert_clean_repo
cluster_assert_registration

bash "${script_dir}/check_preflight.sh" "$ARCMIND_CLUSTER_CONFIG"
for shard_index in 0 1 2 3; do
    [[ -d "${ARCMIND_SHARDS_ROOT}/shard-${shard_index}" ]] || cluster_die \
        "missing shard root: ${ARCMIND_SHARDS_ROOT}/shard-${shard_index}"
done

mkdir -p -- "$ARCMIND_SLURM_LOG_ROOT"
export ARCMIND_CLUSTER_CONFIG ARCMIND_STAGE
sbatch_args=(
    --parsable
    --nodes=1
    --ntasks=1
    --gres="${ARCMIND_SLURM_GRES:-gpu:a100:1}"
    --cpus-per-task="${ARCMIND_SLURM_CPUS:-8}"
    --mem="${ARCMIND_SLURM_MEMORY:-64G}"
    --time="${ARCMIND_SLURM_TIME_FINALIZE:-04:00:00}"
    --job-name="arcmind-${ARCMIND_STAGE}-finalize"
    --chdir="$ARCMIND_REPO_ROOT"
    --output="${ARCMIND_SLURM_LOG_ROOT}/finalize-%j.out"
    --error="${ARCMIND_SLURM_LOG_ROOT}/finalize-%j.err"
    --export=ALL
)
[[ -n "${ARCMIND_SLURM_ACCOUNT:-}" ]] && sbatch_args+=(--account="$ARCMIND_SLURM_ACCOUNT")
[[ -n "${ARCMIND_SLURM_PARTITION:-}" ]] && sbatch_args+=(--partition="$ARCMIND_SLURM_PARTITION")
[[ -n "${ARCMIND_SLURM_QOS:-}" ]] && sbatch_args+=(--qos="$ARCMIND_SLURM_QOS")
[[ -n "${ARCMIND_SLURM_CONSTRAINT:-}" ]] && sbatch_args+=(--constraint="$ARCMIND_SLURM_CONSTRAINT")

job_id="$(sbatch "${sbatch_args[@]}" "${script_dir}/finalize_stage.sh" \
    "$ARCMIND_CLUSTER_CONFIG" "$ARCMIND_STAGE")"
printf 'submitted-%s-finalizer:%s\n' "$ARCMIND_STAGE" "$job_id"
