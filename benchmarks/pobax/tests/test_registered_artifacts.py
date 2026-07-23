from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from benchmarks.pobax.registered_artifacts import (
    ArtifactChecksumError,
    ExistingArtifactMismatchError,
    RegisteredArtifactError,
    atomic_write_json,
    canonical_json_bytes,
    canonical_json_sha256,
    dependency_lock_sha256,
    gather_git_provenance,
    registered_cell_id,
    registered_cell_path,
    validate_paired_seed_manifests,
    validate_unique_cell_ids,
    verify_artifact_checksum,
    write_checksum_manifest,
)


def test_canonical_json_is_order_independent_and_strict() -> None:
    left = {"model": "ArcMind", "nested": {"z": 2, "a": [1, True, None]}, "seed": 7}
    right = {"seed": 7, "nested": {"a": [1, True, None], "z": 2}, "model": "ArcMind"}

    expected = b'{"model":"ArcMind","nested":{"a":[1,true,null],"z":2},"seed":7}'
    assert canonical_json_bytes(left) == expected
    assert canonical_json_bytes(right) == expected
    assert canonical_json_sha256(left) == hashlib.sha256(expected).hexdigest()

    with pytest.raises(ValueError, match="canonical JSON"):
        canonical_json_bytes({"invalid": float("nan")})
    with pytest.raises(ValueError, match="canonical JSON"):
        canonical_json_bytes({"invalid": {1, 2}})
    with pytest.raises(ValueError, match="mapping key"):
        canonical_json_bytes({1: "ambiguous"})


def test_validate_paired_seed_manifests() -> None:
    manifests = {
        "arcmind": [1103, 2207, 3301],
        "gru": (1103, 2207, 3301),
    }
    assert validate_paired_seed_manifests(manifests) == (1103, 2207, 3301)

    with pytest.raises(ValueError, match="not paired"):
        validate_paired_seed_manifests({"arcmind": [1, 2], "gru": [2, 1]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_paired_seed_manifests({"arcmind": [1, 1]})
    with pytest.raises(TypeError, match="integer"):
        validate_paired_seed_manifests({"arcmind": [True]})
    with pytest.raises(ValueError, match="at least one"):
        validate_paired_seed_manifests({})


def test_validate_unique_cell_ids() -> None:
    assert validate_unique_cell_ids(["cell-a", "cell-b"]) == ("cell-a", "cell-b")
    with pytest.raises(ValueError, match="duplicate"):
        validate_unique_cell_ids(["cell-a", "cell-a"])
    with pytest.raises(ValueError, match="non-empty"):
        validate_unique_cell_ids([""])


def test_registered_cell_path_is_stable_safe_and_collision_resistant() -> None:
    config_hash = canonical_json_sha256({"learning_rate": 0.0003})
    first = registered_cell_path("T-Maze/10", "ArcMind:base", 2207, config_hash)
    repeated = registered_cell_path("T-Maze/10", "ArcMind:base", 2207, config_hash.upper())
    nearby = registered_cell_path("T-Maze_10", "ArcMind:base", 2207, config_hash)

    assert first == repeated
    assert first != nearby
    assert first.suffix == ".json"
    assert not any(character in str(first) for character in '<>:"|?*')
    assert registered_cell_id("T-Maze/10", "ArcMind:base", 2207, config_hash) not in first.name
    assert len(registered_cell_id("T-Maze/10", "ArcMind:base", 2207, config_hash)) == 64


def test_atomic_json_write_skips_identical_and_refuses_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "artifact.json"
    first = atomic_write_json(target, {"seed": 2207, "result": [1, 2]})

    assert first.written is True
    assert first.path == target.resolve()
    assert target.read_bytes() == b'{"result":[1,2],"seed":2207}\n'
    assert first.sha256 == hashlib.sha256(target.read_bytes()).hexdigest()

    repeated = atomic_write_json(target, {"result": [1, 2], "seed": 2207})
    assert repeated.written is False
    assert repeated.sha256 == first.sha256

    with pytest.raises(ExistingArtifactMismatchError, match="refusing to replace"):
        atomic_write_json(target, {"seed": 2207, "result": [9]})
    assert target.read_bytes() == b'{"result":[1,2],"seed":2207}\n'
    assert not list(target.parent.glob("*.tmp"))


def test_dependency_and_artifact_checksums(tmp_path: Path) -> None:
    lock = tmp_path / "requirements-lock.txt"
    lock.write_bytes(b"jax==1.2.3\n")
    expected = hashlib.sha256(lock.read_bytes()).hexdigest()

    assert dependency_lock_sha256(lock) == expected
    assert verify_artifact_checksum(lock, expected.upper()) == expected
    with pytest.raises(ArtifactChecksumError, match="checksum mismatch"):
        verify_artifact_checksum(lock, "0" * 64)


def test_checksum_manifest_is_stable_and_excludes_itself(tmp_path: Path) -> None:
    (tmp_path / "z.json").write_bytes(b"z\n")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.json").write_bytes(b"a\n")

    result = write_checksum_manifest(tmp_path)
    a_hash = hashlib.sha256(b"a\n").hexdigest()
    z_hash = hashlib.sha256(b"z\n").hexdigest()
    expected_lines = [
        f"{a_hash}  nested/a.json",
        f"{z_hash}  z.json",
    ]
    assert result.written is True
    assert (tmp_path / "checksums.sha256").read_text(encoding="utf-8").splitlines() == (
        expected_lines
    )

    repeated = write_checksum_manifest(tmp_path)
    assert repeated.written is False
    assert repeated.sha256 == result.sha256

    (tmp_path / "new.json").write_bytes(b"new\n")
    with pytest.raises(ExistingArtifactMismatchError, match="refusing to replace"):
        write_checksum_manifest(tmp_path)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is unavailable")
def test_git_provenance_uses_only_a_temporary_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*arguments: str) -> None:
        subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    git("init", "--quiet")
    tracked = repo / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    git("add", "tracked.txt")
    git(
        "-c",
        "user.name=ArcMind Tests",
        "-c",
        "user.email=tests@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "initial",
    )

    clean = gather_git_provenance(repo)
    assert len(clean["commit"]) in {40, 64}
    assert clean["dirty"] is False
    assert clean["diff_sha256"] is None

    tracked.write_text("dirty one\n", encoding="utf-8")
    dirty_one = gather_git_provenance(repo)
    assert dirty_one["commit"] == clean["commit"]
    assert dirty_one["dirty"] is True
    assert len(dirty_one["diff_sha256"]) == 64

    tracked.write_text("dirty two\n", encoding="utf-8")
    dirty_two = gather_git_provenance(repo)
    assert dirty_two["diff_sha256"] != dirty_one["diff_sha256"]

    (repo / "untracked.bin").write_bytes(b"\x00first")
    with_untracked = gather_git_provenance(repo)
    (repo / "untracked.bin").write_bytes(b"\x00second")
    changed_untracked = gather_git_provenance(repo)
    assert changed_untracked["diff_sha256"] != with_untracked["diff_sha256"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is unavailable")
def test_git_provenance_requires_the_specified_git_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    child = repo / "child"
    child.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    with pytest.raises(RegisteredArtifactError, match="must be its Git root"):
        gather_git_provenance(child)
