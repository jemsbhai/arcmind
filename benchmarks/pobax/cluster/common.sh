#!/usr/bin/env bash
# Shared fail-closed helpers for the ArcMind Slurm handoff.

set -euo pipefail

cluster_die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

cluster_require_command() {
    command -v "$1" >/dev/null 2>&1 || cluster_die "required command is unavailable: $1"
}

cluster_realpath() {
    realpath -m -- "$1"
}

cluster_assert_outside_repo() {
    local label="$1"
    local candidate
    candidate="$(cluster_realpath "$2")"
    case "${candidate}/" in
        "${ARCMIND_REPO_ROOT}/"*)
            cluster_die "${label} must be outside the Git worktree: ${candidate}"
            ;;
    esac
}

cluster_load_config() {
    if [[ $# -ne 1 ]]; then
        cluster_die "cluster_load_config expects one configuration path"
    fi
    cluster_require_command realpath
    local config_path
    config_path="$(cluster_realpath "$1")"
    [[ -f "$config_path" ]] || cluster_die "configuration file does not exist: ${config_path}"

    # shellcheck source=/dev/null
    source "$config_path"

    : "${ARCMIND_REPO_ROOT:?set ARCMIND_REPO_ROOT in the cluster configuration}"
    : "${ARCMIND_VENV:?set ARCMIND_VENV in the cluster configuration}"
    : "${ARCMIND_CLUSTER_ROOT:?set ARCMIND_CLUSTER_ROOT in the cluster configuration}"
    : "${ARCMIND_EXPECTED_COMMIT:?set ARCMIND_EXPECTED_COMMIT in the cluster configuration}"

    ARCMIND_REPO_ROOT="$(cluster_realpath "$ARCMIND_REPO_ROOT")"
    ARCMIND_VENV="$(cluster_realpath "$ARCMIND_VENV")"
    ARCMIND_CLUSTER_ROOT="$(cluster_realpath "$ARCMIND_CLUSTER_ROOT")"
    ARCMIND_CLUSTER_CONFIG="$config_path"
    ARCMIND_SHARD_COUNT=4
    ARCMIND_DEVICE_KIND_REGEX="${ARCMIND_DEVICE_KIND_REGEX:-A100.*40GB}"

    [[ -e "$ARCMIND_REPO_ROOT/.git" ]] || cluster_die \
        "ARCMIND_REPO_ROOT is not a Git worktree root: ${ARCMIND_REPO_ROOT}"
    if [[ "${ARCMIND_ALLOW_MISSING_VENV:-0}" != 1 ]]; then
        [[ -x "$ARCMIND_VENV/bin/python" ]] || cluster_die \
            "benchmark Python is unavailable: ${ARCMIND_VENV}/bin/python"
    fi
    [[ "$ARCMIND_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || cluster_die \
        "ARCMIND_EXPECTED_COMMIT must be a full lowercase 40-character commit"

    cluster_assert_outside_repo "ARCMIND_VENV" "$ARCMIND_VENV"
    cluster_assert_outside_repo "ARCMIND_CLUSTER_ROOT" "$ARCMIND_CLUSTER_ROOT"
    cluster_assert_outside_repo "cluster configuration" "$config_path"

    ARCMIND_PREFLIGHT_ROOT="${ARCMIND_CLUSTER_ROOT}/${ARCMIND_EXPECTED_COMMIT}/preflight"
    ARCMIND_APPROVED_RUNTIME="${ARCMIND_PREFLIGHT_ROOT}/approved-runtime.json"

    export ARCMIND_REPO_ROOT ARCMIND_VENV ARCMIND_CLUSTER_ROOT
    export ARCMIND_EXPECTED_COMMIT ARCMIND_CLUSTER_CONFIG ARCMIND_SHARD_COUNT
    export ARCMIND_DEVICE_KIND_REGEX ARCMIND_PREFLIGHT_ROOT ARCMIND_APPROVED_RUNTIME
}

cluster_assert_clean_repo() {
    local actual_commit status
    cluster_require_command git
    actual_commit="$(git -C "$ARCMIND_REPO_ROOT" rev-parse --verify HEAD)"
    [[ "$actual_commit" == "$ARCMIND_EXPECTED_COMMIT" ]] || cluster_die \
        "wrong ArcMind commit: expected ${ARCMIND_EXPECTED_COMMIT}, found ${actual_commit}"
    status="$(git -C "$ARCMIND_REPO_ROOT" status --porcelain=v1 --untracked-files=all)"
    [[ -z "$status" ]] || cluster_die \
        "ArcMind worktree is dirty; registered execution requires an exact clean commit"
}

cluster_set_stage() {
    if [[ $# -ne 1 ]]; then
        cluster_die "cluster_set_stage expects tuning, primary, or upper"
    fi
    ARCMIND_STAGE="$1"
    case "$ARCMIND_STAGE" in
        tuning)
            ARCMIND_REGISTRATION_REL="benchmarks/pobax/manifests/compute_aware_tuning_v1.json"
            ARCMIND_RAW_REL="benchmark_results/pobax/compute-aware-tuning-v1"
            ARCMIND_AGGREGATE_REL="benchmark_results/pobax/aggregates/compute-aware-tuning-v1.json"
            ;;
        primary)
            ARCMIND_REGISTRATION_REL="benchmarks/pobax/manifests/compute_aware_final_v1.json"
            ARCMIND_RAW_REL="benchmark_results/pobax/compute-aware-primary-v1"
            ARCMIND_AGGREGATE_REL="benchmark_results/pobax/aggregates/compute-aware-primary-v1.json"
            ;;
        upper)
            ARCMIND_REGISTRATION_REL="benchmark_results/pobax/registrations/compute-aware-upper-v1.json"
            ARCMIND_RAW_REL="benchmark_results/pobax/compute-aware-upper-v1"
            ARCMIND_AGGREGATE_REL="benchmark_results/pobax/aggregates/compute-aware-upper-v1.json"
            ;;
        *)
            cluster_die "unknown stage ${ARCMIND_STAGE@Q}; expected tuning, primary, or upper"
            ;;
    esac

    ARCMIND_REGISTRATION="${ARCMIND_REPO_ROOT}/${ARCMIND_REGISTRATION_REL}"
    ARCMIND_CANONICAL_ROOT="${ARCMIND_REPO_ROOT}/${ARCMIND_RAW_REL}"
    ARCMIND_AGGREGATE="${ARCMIND_REPO_ROOT}/${ARCMIND_AGGREGATE_REL}"
    ARCMIND_STAGE_ROOT="${ARCMIND_CLUSTER_ROOT}/${ARCMIND_EXPECTED_COMMIT}/${ARCMIND_STAGE}"
    ARCMIND_SHARDS_ROOT="${ARCMIND_STAGE_ROOT}/shards"
    ARCMIND_SLURM_LOG_ROOT="${ARCMIND_STAGE_ROOT}/slurm"
    export ARCMIND_STAGE ARCMIND_REGISTRATION_REL ARCMIND_RAW_REL ARCMIND_AGGREGATE_REL
    export ARCMIND_REGISTRATION ARCMIND_CANONICAL_ROOT ARCMIND_AGGREGATE
    export ARCMIND_STAGE_ROOT ARCMIND_SHARDS_ROOT ARCMIND_SLURM_LOG_ROOT
    export ARCMIND_PREFLIGHT_ROOT ARCMIND_APPROVED_RUNTIME
}

cluster_assert_registration() {
    [[ -f "$ARCMIND_REGISTRATION" ]] || cluster_die \
        "stage registration is unavailable: ${ARCMIND_REGISTRATION}"
}

cluster_python() {
    printf '%s/bin/python\n' "$ARCMIND_VENV"
}
