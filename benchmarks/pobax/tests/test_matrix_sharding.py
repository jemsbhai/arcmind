"""Fail-closed tests for isolated matrix shards and canonical merging."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.pobax import merge_matrix_shards as merge_module
from benchmarks.pobax import run_matrix
from benchmarks.pobax.merge_matrix_shards import merge_matrix_shards
from benchmarks.pobax.registered_artifacts import (
    ExistingArtifactMismatchError,
    RegisteredArtifactError,
    atomic_write_json,
    canonical_json_sha256,
    exclusive_process_lock,
    matrix_process_lock_path,
)
from benchmarks.pobax.run_matrix import (
    _shard_cells,
    _validate_sharded_runtime,
    execute_matrix,
)


def _registration() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "frozen",
        "evidence_tier": "pilot",
        "matrix_kind": "primary_comparison",
        "models": ["arcmind", "gru"],
        "environments": [{"id": "tmaze_10", "total_steps": 131_072}],
        "seeds": [1103, 2207],
        "learner": {
            "num_envs": 64,
            "rollout_steps": 64,
            "update_epochs": 4,
            "learning_rate": 0.00025,
        },
        "evaluation_episodes_per_env": 4,
        "require_gpu": True,
        "quick": False,
    }


def _write_registration(path: Path) -> None:
    path.write_text(json.dumps(_registration()), encoding="utf-8")


def _install_fake_execution(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    provenance = {
        "git": {"commit": "a" * 40, "dirty": False, "diff_sha256": None},
        "dependency_lock_sha256": "b" * 64,
        "pobax_commit": "c" * 40,
        "navix_commit": "d" * 40,
        "runtime_contract": {
            "jax_backend": "gpu",
            "devices": [{"platform": "gpu", "device_kind": "NVIDIA A100-SXM4-40GB"}],
        },
    }

    def configuration(environment: str, model: str, seed: int) -> dict[str, object]:
        return {
            "schema_version": 1,
            "evidence_tier": "pilot",
            "environment": environment,
            "model": model,
            "seed": seed,
            "dependency_lock_sha256": provenance["dependency_lock_sha256"],
            "pobax_commit": provenance["pobax_commit"],
            "navix_commit": provenance["navix_commit"],
            "runtime_contract": provenance["runtime_contract"],
        }

    def fake_describe(args):
        frozen = configuration(args.environment, args.model, args.seed)
        return {
            "configuration_sha256": canonical_json_sha256(frozen),
            "configuration": frozen,
            "runtime": {"git": provenance["git"]},
        }

    attempts: list[list[str]] = []

    def argument(command: list[str], name: str) -> str:
        return command[command.index(name) + 1]

    def fake_subprocess(command, **kwargs):
        del kwargs
        command = list(command)
        attempts.append(command)
        environment = argument(command, "--environment")
        model = argument(command, "--model")
        seed = int(argument(command, "--seed"))
        frozen = configuration(environment, model, seed)
        artifact = {
            "schema_version": 4,
            "status": "development_pilot_not_for_paper",
            "environment": environment,
            "model": model,
            "seed": seed,
            "configuration_sha256": canonical_json_sha256(frozen),
            "configuration": frozen,
            "matrix_manifest_sha256": argument(command, "--matrix-manifest-sha256"),
            "cell_id": argument(command, "--cell-id"),
            "provenance": provenance,
        }
        atomic_write_json(Path(argument(command, "--output")), artifact)
        return SimpleNamespace(returncode=0, stdout=f"completed {model} {seed}\n".encode())

    monkeypatch.setattr(run_matrix, "run", fake_describe)
    monkeypatch.setattr(run_matrix.subprocess, "run", fake_subprocess)
    return attempts


def test_shard_selection_is_disjoint_complete_and_balanced() -> None:
    cells = [{"cell_id": str(index)} for index in range(234)]
    shards = [_shard_cells(cells, shard_count=4, shard_index=index) for index in range(4)]

    assert [len(shard) for shard in shards] == [59, 59, 58, 58]
    assert {cell["cell_id"] for shard in shards for cell in shard} == {
        str(index) for index in range(234)
    }
    assert sum(len(shard) for shard in shards) == 234


@pytest.mark.parametrize(
    ("cell_count", "expected_counts"),
    [
        (490, [123, 123, 122, 122]),
        (40, [10, 10, 10, 10]),
    ],
)
def test_four_way_final_stage_shard_counts(
    cell_count: int,
    expected_counts: list[int],
) -> None:
    cells = [{"cell_id": str(index)} for index in range(cell_count)]

    assert [
        len(_shard_cells(cells, shard_count=4, shard_index=index)) for index in range(4)
    ] == expected_counts


def test_sharded_gpu_execution_requires_exactly_one_visible_device() -> None:
    description = {
        "registration": {"require_gpu": True},
        "provenance": {
            "runtime_contract": {
                "jax_backend": "gpu",
                "devices": [
                    {"platform": "gpu", "device_kind": "A100"},
                    {"platform": "gpu", "device_kind": "A100"},
                ],
            }
        },
    }

    with pytest.raises(RuntimeError, match="exactly one visible JAX GPU"):
        _validate_sharded_runtime(description, shard_count=4)

    description["provenance"]["runtime_contract"]["devices"] = [
        {"platform": "gpu", "device_kind": "A100"}
    ]
    _validate_sharded_runtime(description, shard_count=4)


@pytest.mark.parametrize(
    ("shard_count", "shard_index", "message"),
    [
        (0, 0, "shard_count"),
        (True, 0, "shard_count"),
        (4, -1, "shard_index"),
        (4, 4, "shard_index"),
        (4, True, "shard_index"),
    ],
)
def test_invalid_shard_selection_fails_closed(
    tmp_path: Path,
    shard_count: int,
    shard_index: int,
    message: str,
) -> None:
    registration_path = tmp_path / "registration.json"
    _write_registration(registration_path)

    with pytest.raises(ValueError, match=message):
        execute_matrix(
            registration_path,
            tmp_path / "shard",
            shard_count=shard_count,
            shard_index=shard_index,
        )


def test_isolated_shards_resume_and_merge_into_one_certified_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration_path = tmp_path / "registration.json"
    _write_registration(registration_path)
    attempts = _install_fake_execution(monkeypatch)
    shard_roots = [tmp_path / "shard-0", tmp_path / "shard-1"]

    summaries = [
        execute_matrix(
            registration_path,
            shard_root,
            shard_count=2,
            shard_index=shard_index,
        )
        for shard_index, shard_root in enumerate(shard_roots)
    ]

    assert [summary["status"] for summary in summaries] == ["shard_complete"] * 2
    assert [summary["assigned_cells"] for summary in summaries] == [2, 2]
    assert len(attempts) == 4
    for shard_root in shard_roots:
        assert not (shard_root / "completion_index.json").exists()
        assert not (shard_root / "checksums.sha256").exists()
        assert (shard_root / "shard_completion.json").is_file()
        assert (shard_root / "shard_checksums.sha256").is_file()

    resumed = execute_matrix(
        registration_path,
        shard_roots[0],
        shard_count=2,
        shard_index=0,
    )
    assert resumed == summaries[0]
    assert len(attempts) == 4

    canonical_root = tmp_path / "canonical"
    completion = merge_matrix_shards(
        registration_path,
        canonical_root,
        list(reversed(shard_roots)),
    )
    assert completion["status"] == "complete"
    assert completion["planned_cells"] == 4
    assert completion["completed_cells"] == 4
    assert (canonical_root / "completion_index.json").is_file()
    assert (canonical_root / "checksums.sha256").is_file()
    assert len(attempts) == 4

    repeated = merge_matrix_shards(registration_path, canonical_root, shard_roots)
    assert repeated == completion


@pytest.mark.parametrize("failure", ["missing", "extra", "manifest", "checksum"])
def test_merge_rejects_incomplete_or_contaminated_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    registration_path = tmp_path / "registration.json"
    _write_registration(registration_path)
    _install_fake_execution(monkeypatch)
    shard_roots = [tmp_path / "shard-0", tmp_path / "shard-1"]
    for shard_index, shard_root in enumerate(shard_roots):
        execute_matrix(
            registration_path,
            shard_root,
            shard_count=2,
            shard_index=shard_index,
        )

    if failure == "missing":
        next(shard_roots[0].rglob("*.log")).unlink()
        message = "missing required frozen files"
    elif failure == "extra":
        (shard_roots[0] / "scheduler.log").write_text("must remain outside", encoding="utf-8")
        message = "outside its frozen role"
    elif failure == "manifest":
        manifest_path = shard_roots[0] / "frozen_manifest.json"
        manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
        message = "differs from the current frozen matrix"
    else:
        checksum_path = shard_roots[0] / "shard_checksums.sha256"
        checksum_lines = checksum_path.read_text(encoding="utf-8").splitlines()
        checksum_lines[0] = f"{'0' * 64}{checksum_lines[0][64:]}"
        checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        message = "checksum mismatch"

    with pytest.raises(RuntimeError, match=message):
        merge_matrix_shards(registration_path, tmp_path / "canonical", shard_roots)


def test_merge_rejects_overlapping_roots(
    tmp_path: Path,
) -> None:
    registration_path = tmp_path / "registration.json"
    _write_registration(registration_path)
    shard_roots = [tmp_path / "shard-0", tmp_path / "shard-1"]
    for shard_root in shard_roots:
        shard_root.mkdir()

    with pytest.raises(ValueError, match="must not contain one another"):
        merge_matrix_shards(
            registration_path,
            shard_roots[0] / "canonical",
            shard_roots,
        )


def test_merge_prevalidates_target_collisions_before_copying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration_path = tmp_path / "registration.json"
    _write_registration(registration_path)
    _install_fake_execution(monkeypatch)
    shard_roots = [tmp_path / "shard-0", tmp_path / "shard-1"]
    for shard_index, shard_root in enumerate(shard_roots):
        execute_matrix(
            registration_path,
            shard_root,
            shard_count=2,
            shard_index=shard_index,
        )

    manifest = json.loads((shard_roots[0] / "frozen_manifest.json").read_text(encoding="utf-8"))
    conflicting_path = tmp_path / "canonical" / manifest["cells"][0]["artifact_path"]
    conflicting_path.parent.mkdir(parents=True)
    conflicting_path.write_bytes(b"different existing evidence\n")

    with pytest.raises(ExistingArtifactMismatchError, match="different existing file"):
        merge_matrix_shards(registration_path, tmp_path / "canonical", shard_roots)

    assert not (tmp_path / "canonical" / "registration.json").exists()
    assert conflicting_path.read_bytes() == b"different existing evidence\n"


@pytest.mark.parametrize(
    ("shard_count", "shard_index", "completion_name", "checksum_name"),
    [
        (1, 0, "completion_index.json", "checksums.sha256"),
        (2, 0, "shard_completion.json", "shard_checksums.sha256"),
    ],
)
def test_runner_rejects_extra_file_before_writing_indexes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shard_count: int,
    shard_index: int,
    completion_name: str,
    checksum_name: str,
) -> None:
    registration_path = tmp_path / "registration.json"
    _write_registration(registration_path)
    _install_fake_execution(monkeypatch)
    fake_subprocess = run_matrix.subprocess.run
    shard_root = tmp_path / "shard-0"

    def inject_extra_file(*args, **kwargs):
        result = fake_subprocess(*args, **kwargs)
        (shard_root / "scheduler.log").write_text("external log", encoding="utf-8")
        return result

    monkeypatch.setattr(run_matrix.subprocess, "run", inject_extra_file)

    with pytest.raises(RuntimeError, match="outside its frozen role"):
        execute_matrix(
            registration_path,
            shard_root,
            shard_count=shard_count,
            shard_index=shard_index,
        )

    assert not (shard_root / completion_name).exists()
    assert not (shard_root / checksum_name).exists()


def test_merge_and_direct_execution_share_the_canonical_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration_path = tmp_path / "registration.json"
    _write_registration(registration_path)
    _install_fake_execution(monkeypatch)
    canonical_root = tmp_path / "canonical"
    shard_roots = [tmp_path / "shard-0", tmp_path / "shard-1"]
    for shard_index, shard_root in enumerate(shard_roots):
        execute_matrix(
            registration_path,
            shard_root,
            shard_count=2,
            shard_index=shard_index,
        )

    with exclusive_process_lock(matrix_process_lock_path(canonical_root)):
        with pytest.raises(RegisteredArtifactError, match="another process holds"):
            execute_matrix(registration_path, canonical_root)
        with pytest.raises(RegisteredArtifactError, match="another process holds"):
            merge_matrix_shards(registration_path, canonical_root, shard_roots)


def test_merge_locks_every_source_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration_path = tmp_path / "registration.json"
    _write_registration(registration_path)
    _install_fake_execution(monkeypatch)
    shard_roots = [tmp_path / "shard-0", tmp_path / "shard-1"]
    for shard_index, shard_root in enumerate(shard_roots):
        execute_matrix(
            registration_path,
            shard_root,
            shard_count=2,
            shard_index=shard_index,
        )

    with exclusive_process_lock(matrix_process_lock_path(shard_roots[0])):
        with pytest.raises(RegisteredArtifactError, match="another process holds"):
            merge_matrix_shards(registration_path, tmp_path / "canonical", shard_roots)


def test_merge_rejects_source_mutation_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration_path = tmp_path / "registration.json"
    _write_registration(registration_path)
    _install_fake_execution(monkeypatch)
    shard_roots = [tmp_path / "shard-0", tmp_path / "shard-1"]
    for shard_index, shard_root in enumerate(shard_roots):
        execute_matrix(
            registration_path,
            shard_root,
            shard_count=2,
            shard_index=shard_index,
        )

    original_prevalidate = merge_module._prevalidate_canonical_collisions

    def mutate_then_prevalidate(output_root, **kwargs):
        source = kwargs["copy_plan"][0][0]
        source.write_bytes(b"mutated after shard validation\n")
        return original_prevalidate(output_root, **kwargs)

    monkeypatch.setattr(
        merge_module,
        "_prevalidate_canonical_collisions",
        mutate_then_prevalidate,
    )

    with pytest.raises(RuntimeError, match="shard source changed after validation"):
        merge_matrix_shards(registration_path, tmp_path / "canonical", shard_roots)

    assert not (tmp_path / "canonical").exists()
