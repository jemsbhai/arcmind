"""Deterministic provenance for the code that can affect POBAX execution."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

IMPLEMENTATION_SOURCE_SCHEMA_VERSION = 1
IMPLEMENTATION_SOURCE_ALGORITHM = "canonical-path-file-sha256-manifest-v1"
_SOURCE_FIELDS = {"schema_version", "algorithm", "files", "sha256"}
_FILE_FIELDS = {"path", "sha256"}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tracked_runtime_paths(repository_root: Path) -> tuple[str, ...]:
    process = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "ls-files",
            "-z",
            "--",
            "arcmind",
            "benchmarks/pobax",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    paths = []
    for raw_path in process.stdout.decode("utf-8").split("\0"):
        if not raw_path.endswith(".py"):
            continue
        if raw_path.startswith("arcmind/") or (
            raw_path.startswith("benchmarks/pobax/")
            and not raw_path.startswith("benchmarks/pobax/tests/")
        ):
            paths.append(raw_path)
    paths.sort(key=lambda item: item.encode("utf-8"))
    if not paths:
        raise RuntimeError("implementation source inventory is empty")
    return tuple(paths)


def gather_implementation_source(repository_root: str | Path) -> dict[str, Any]:
    """Hash every tracked Python source file in the conservative runtime surface."""

    root = Path(repository_root).resolve()
    files = []
    for relative in _tracked_runtime_paths(root):
        path = root.joinpath(*relative.split("/")).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f"tracked implementation source is unavailable: {relative}")
        files.append({"path": relative, "sha256": _sha256_bytes(path.read_bytes())})
    unsigned = {
        "schema_version": IMPLEMENTATION_SOURCE_SCHEMA_VERSION,
        "algorithm": IMPLEMENTATION_SOURCE_ALGORITHM,
        "files": files,
    }
    return {**unsigned, "sha256": _sha256_bytes(_canonical_json_bytes(unsigned))}


def normalize_implementation_source(value: object) -> dict[str, Any]:
    """Validate one implementation-source manifest and its digest."""

    if not isinstance(value, Mapping) or set(value) != _SOURCE_FIELDS:
        raise ValueError("implementation_source has the wrong fields")
    if (
        value["schema_version"] != IMPLEMENTATION_SOURCE_SCHEMA_VERSION
        or value["algorithm"] != IMPLEMENTATION_SOURCE_ALGORITHM
    ):
        raise ValueError("implementation_source has an unsupported schema or algorithm")
    raw_files = value["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("implementation_source.files must be a non-empty list")
    files = []
    previous_path: str | None = None
    for index, raw_file in enumerate(raw_files):
        if not isinstance(raw_file, Mapping) or set(raw_file) != _FILE_FIELDS:
            raise ValueError(f"implementation_source.files[{index}] has the wrong fields")
        path = raw_file["path"]
        sha256 = raw_file["sha256"]
        if (
            not isinstance(path, str)
            or "\\" in path
            or not path.endswith(".py")
            or not (
                path.startswith("arcmind/")
                or (
                    path.startswith("benchmarks/pobax/")
                    and not path.startswith("benchmarks/pobax/tests/")
                )
            )
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(f"implementation_source.files[{index}] is invalid")
        if previous_path is not None and path.encode("utf-8") <= previous_path.encode("utf-8"):
            raise ValueError("implementation_source.files must be uniquely sorted")
        previous_path = path
        files.append({"path": path, "sha256": sha256})
    unsigned = {
        "schema_version": IMPLEMENTATION_SOURCE_SCHEMA_VERSION,
        "algorithm": IMPLEMENTATION_SOURCE_ALGORITHM,
        "files": files,
    }
    expected = _sha256_bytes(_canonical_json_bytes(unsigned))
    if value["sha256"] != expected:
        raise ValueError("implementation_source.sha256 does not match its manifest")
    return {**unsigned, "sha256": expected}


__all__ = [
    "IMPLEMENTATION_SOURCE_ALGORITHM",
    "IMPLEMENTATION_SOURCE_SCHEMA_VERSION",
    "gather_implementation_source",
    "normalize_implementation_source",
]
