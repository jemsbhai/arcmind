"""Reproducible artifact utilities for registered POBAX experiments.

The helpers in this module use only the Python standard library. Artifact
creation is fail closed: an existing identical file is reused, while an
existing file with different bytes is never replaced.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_UNSAFE_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


class RegisteredArtifactError(RuntimeError):
    """Base error for invalid or unsafe registered artifacts."""


class ExistingArtifactMismatchError(RegisteredArtifactError):
    """Raised when a write would replace a different existing artifact."""


class ArtifactChecksumError(RegisteredArtifactError):
    """Raised when an artifact does not have its expected checksum."""


@dataclass(frozen=True)
class ArtifactWriteResult:
    """Result of an idempotent artifact creation."""

    path: Path
    sha256: str
    written: bool


def _validate_json_tree(value: Any, *, location: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON mapping key at {location} is not a string")
            _validate_json_tree(child, location=f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_tree(child, location=f"{location}[{index}]")
        return
    raise TypeError(f"value at {location} is outside the JSON data model")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value to stable UTF-8 bytes.

    Mapping keys must be strings. Non-finite floats and values outside the JSON
    data model are rejected instead of receiving platform-specific encodings.
    """

    try:
        _validate_json_tree(value)
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"value is not canonical JSON: {error}") from error
    return encoded.encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Return the SHA256 of a canonical JSON configuration or manifest."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_paired_seed_manifests(
    manifests: Mapping[str, Sequence[int]],
) -> tuple[int, ...]:
    """Validate that every named method uses one identical ordered seed list.

    The returned tuple is the shared frozen seed manifest.
    """

    if not manifests:
        raise ValueError("at least one seed manifest is required")

    shared: tuple[int, ...] | None = None
    for name, seeds in manifests.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("seed manifest names must be non-empty strings")
        if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
            raise TypeError(f"seed manifest {name!r} must be a sequence")

        frozen = tuple(seeds)
        if not frozen:
            raise ValueError(f"seed manifest {name!r} must not be empty")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in frozen):
            raise TypeError(f"seed manifest {name!r} must contain only integer seeds")
        if any(seed < 0 for seed in frozen):
            raise ValueError(f"seed manifest {name!r} contains a negative seed")
        if len(set(frozen)) != len(frozen):
            raise ValueError(f"seed manifest {name!r} contains duplicate seeds")

        if shared is None:
            shared = frozen
        elif frozen != shared:
            raise ValueError(f"seed manifest {name!r} is not paired with the shared ordered seeds")

    if shared is None:  # pragma: no cover - guarded by the non-empty mapping check
        raise AssertionError("seed manifest validation did not select shared seeds")
    return shared


def validate_unique_cell_ids(cell_ids: Iterable[str]) -> tuple[str, ...]:
    """Validate and freeze a non-empty collection of unique cell identifiers."""

    frozen = tuple(cell_ids)
    if not frozen:
        raise ValueError("at least one cell ID is required")

    seen: set[str] = set()
    for cell_id in frozen:
        if not isinstance(cell_id, str) or not cell_id.strip():
            raise ValueError("cell IDs must be non-empty strings")
        if cell_id in seen:
            raise ValueError(f"duplicate cell ID: {cell_id!r}")
        seen.add(cell_id)
    return frozen


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA256 of a regular file without loading it all into memory."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"not a regular file: {file_path}")

    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dependency_lock_sha256(path: str | os.PathLike[str]) -> str:
    """Return the content checksum that identifies a dependency lock file."""

    return sha256_file(path)


def _run_git(repo: Path, arguments: Sequence[str]) -> bytes:
    try:
        process = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise RegisteredArtifactError("git is required to gather provenance") from error

    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise RegisteredArtifactError(f"git {' '.join(arguments)} failed for {repo}: {detail}")
    return process.stdout


def _untracked_paths(status: bytes) -> tuple[bytes, ...]:
    return tuple(
        entry[3:] for entry in status.split(b"\0") if entry.startswith(b"?? ") and len(entry) > 3
    )


def _dirty_state_sha256(repo: Path, status: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"arcmind-git-dirty-state-v1\0")
    digest.update(len(status).to_bytes(8, "big"))
    digest.update(status)

    diff = _run_git(repo, ["diff", "--binary", "--no-ext-diff", "HEAD", "--"])
    digest.update(len(diff).to_bytes(8, "big"))
    digest.update(diff)

    for encoded_path in sorted(_untracked_paths(status)):
        relative_path = os.fsdecode(encoded_path)
        path = repo / relative_path
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        if path.is_symlink():
            target = os.fsencode(os.readlink(path))
            digest.update(b"L")
            digest.update(len(target).to_bytes(8, "big"))
            digest.update(target)
        elif path.is_file():
            digest.update(b"F")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        else:
            digest.update(b"M")
    return digest.hexdigest()


def gather_git_provenance(repo: str | os.PathLike[str]) -> dict[str, Any]:
    """Record commit, dirty status, and a dirty-state checksum for a Git repo."""

    repo_path = Path(repo).resolve()
    if not repo_path.is_dir():
        raise FileNotFoundError(f"repository directory does not exist: {repo_path}")

    root = Path(
        os.fsdecode(_run_git(repo_path, ["rev-parse", "--show-toplevel"])).strip()
    ).resolve()
    if root != repo_path:
        raise RegisteredArtifactError(
            f"specified repository must be its Git root: {repo_path} != {root}"
        )

    commit = _run_git(repo_path, ["rev-parse", "--verify", "HEAD"]).decode("ascii").strip()
    status = _run_git(
        repo_path,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )
    dirty = bool(status)
    return {
        "commit": commit,
        "dirty": dirty,
        "diff_sha256": _dirty_state_sha256(repo_path, status) if dirty else None,
    }


def _validate_sha256(value: str, *, field: str) -> str:
    normalized = value.lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA256")
    return normalized


def _safe_component(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    cleaned = _UNSAFE_COMPONENT_PATTERN.sub("-", value.strip()).strip(" ._-").lower()
    if not cleaned:
        cleaned = "item"
    cleaned = cleaned[:40].rstrip(" ._-") or "item"
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned}-{suffix}"


def registered_cell_id(
    environment: str,
    model: str,
    seed: int,
    config_sha256: str,
) -> str:
    """Return a deterministic full SHA256 identifier for one registered cell."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    normalized_hash = _validate_sha256(config_sha256, field="config_sha256")
    return canonical_json_sha256(
        {
            "config_sha256": normalized_hash,
            "environment": environment,
            "model": model,
            "seed": seed,
        }
    )


def registered_cell_path(
    environment: str,
    model: str,
    seed: int,
    config_sha256: str,
) -> Path:
    """Build a readable, Windows-safe, collision-resistant relative JSON path."""

    normalized_hash = _validate_sha256(config_sha256, field="config_sha256")
    cell_id = registered_cell_id(environment, model, seed, normalized_hash)
    environment_part = f"environment-{_safe_component(environment, field='environment')}"
    model_part = f"model-{_safe_component(model, field='model')}"
    return (
        Path(environment_part)
        / model_part
        / f"seed-{seed}"
        / f"config-{normalized_hash[:16]}-cell-{cell_id[:20]}.json"
    )


def _atomic_create(path: Path, content: bytes) -> ArtifactWriteResult:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = path.read_bytes()
        if existing == content:
            return ArtifactWriteResult(path, hashlib.sha256(content).hexdigest(), False)
        raise ExistingArtifactMismatchError(
            f"refusing to replace different existing artifact: {path}"
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())

        try:
            os.link(temporary_path, path)
        except FileExistsError:
            existing = path.read_bytes()
            if existing == content:
                return ArtifactWriteResult(
                    path,
                    hashlib.sha256(content).hexdigest(),
                    False,
                )
            raise ExistingArtifactMismatchError(
                f"refusing to replace concurrently created artifact: {path}"
            ) from None
    finally:
        temporary_path.unlink(missing_ok=True)

    return ArtifactWriteResult(path, hashlib.sha256(content).hexdigest(), True)


def atomic_write_json(
    path: str | os.PathLike[str],
    value: Any,
) -> ArtifactWriteResult:
    """Atomically create canonical JSON, reuse identical bytes, and never replace."""

    return _atomic_create(Path(path), canonical_json_bytes(value) + b"\n")


def atomic_write_bytes(
    path: str | os.PathLike[str],
    content: bytes,
) -> ArtifactWriteResult:
    """Atomically create an arbitrary byte artifact without replacement."""
    if not isinstance(content, bytes):
        raise TypeError("content must be bytes")
    return _atomic_create(Path(path), content)


def verify_artifact_checksum(
    path: str | os.PathLike[str],
    expected_sha256: str,
) -> str:
    """Verify an artifact checksum and return the normalized actual checksum."""

    expected = _validate_sha256(expected_sha256, field="expected_sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise ArtifactChecksumError(
            f"checksum mismatch for {Path(path)}: expected {expected}, got {actual}"
        )
    return actual


def write_checksum_manifest(
    directory: str | os.PathLike[str],
    *,
    filename: str = "checksums.sha256",
) -> ArtifactWriteResult:
    """Create a stable SHA256 manifest for every file below a directory.

    The manifest itself is excluded. Relative paths use forward slashes and are
    sorted by their UTF-8 encoded representation.
    """

    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {root}")
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise ValueError("checksum manifest filename must be a plain file name")

    output_path = root / filename
    entries: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.resolve() == output_path.resolve():
            continue
        relative = path.relative_to(root).as_posix()
        entries.append((relative, path))
    entries.sort(key=lambda item: item[0].encode("utf-8"))

    content = "".join(f"{sha256_file(path)}  {relative}\n" for relative, path in entries).encode(
        "utf-8"
    )
    return _atomic_create(output_path, content)


def validate_checksum_manifest(
    directory: str | os.PathLike[str],
    *,
    filename: str = "checksums.sha256",
) -> tuple[tuple[str, str], ...]:
    """Validate a checksum manifest against the exact regular-file inventory."""

    root = Path(directory).resolve()
    manifest_path = (root / filename).resolve()
    if not manifest_path.is_file():
        raise ArtifactChecksumError(f"checksum manifest does not exist: {manifest_path}")
    try:
        content = manifest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactChecksumError(f"cannot read checksum manifest: {error}") from error
    lines = content.splitlines()
    if not lines:
        raise ArtifactChecksumError("checksum manifest must not be empty")

    actual_files = []
    for path in root.rglob("*"):
        resolved = path.resolve()
        if path.is_file() and resolved != manifest_path:
            if not resolved.is_relative_to(root):
                raise ArtifactChecksumError(f"artifact file escapes checksum root: {path}")
            actual_files.append(path.relative_to(root).as_posix())
    actual_files.sort(key=lambda item: item.encode("utf-8"))

    parsed: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ArtifactChecksumError(f"invalid checksum line {index + 1}")
        relative = match.group(2)
        pure = PurePosixPath(relative)
        if (
            "\\" in relative
            or pure.is_absolute()
            or relative != pure.as_posix()
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise ArtifactChecksumError(f"unsafe checksum path: {relative!r}")
        if relative == filename:
            raise ArtifactChecksumError("checksum manifest must not contain itself")
        if parsed and relative.encode("utf-8") <= parsed[-1][0].encode("utf-8"):
            raise ArtifactChecksumError("checksum paths must be uniquely sorted")
        parsed.append((relative, match.group(1)))

    parsed_paths = [relative for relative, _ in parsed]
    if parsed_paths != actual_files:
        missing = sorted(set(actual_files) - set(parsed_paths))
        extra = sorted(set(parsed_paths) - set(actual_files))
        raise ArtifactChecksumError(
            f"checksum inventory differs from regular files: missing={missing}, extra={extra}"
        )
    for relative, expected_sha256 in parsed:
        path = root.joinpath(*PurePosixPath(relative).parts)
        if sha256_file(path) != expected_sha256:
            raise ArtifactChecksumError(f"checksum mismatch for {relative}")
    return tuple(parsed)


__all__ = [
    "ArtifactChecksumError",
    "ArtifactWriteResult",
    "ExistingArtifactMismatchError",
    "RegisteredArtifactError",
    "atomic_write_bytes",
    "atomic_write_json",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "dependency_lock_sha256",
    "gather_git_provenance",
    "registered_cell_id",
    "registered_cell_path",
    "sha256_file",
    "validate_paired_seed_manifests",
    "validate_unique_cell_ids",
    "verify_artifact_checksum",
    "validate_checksum_manifest",
    "write_checksum_manifest",
]
