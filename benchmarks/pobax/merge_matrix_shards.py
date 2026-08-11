"""Validate isolated matrix shards and finalize one canonical raw matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from benchmarks.pobax.registered_artifacts import (
    ExistingArtifactMismatchError,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    exclusive_process_lock,
    matrix_process_lock_path,
    sha256_file,
    validate_checksum_manifest,
)
from benchmarks.pobax.run_matrix import (
    _SHARD_PARTITION_ALGORITHM,
    _describe_matrix,
    _finalize_complete_matrix,
    _shard_cells,
    _validate_completed_cell,
    _validate_matrix_root_inventory,
)


def _reject_overlapping_roots(output_root: Path, shard_roots: tuple[Path, ...]) -> None:
    all_roots = (output_root, *shard_roots)
    if len(set(all_roots)) != len(all_roots):
        raise ValueError("canonical output and shard roots must be distinct")
    for index, left in enumerate(all_roots):
        for right in all_roots[index + 1 :]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("canonical output and shard roots must not contain one another")


def _require_identical_control_files(
    shard_root: Path,
    *,
    registration: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    expected = {
        "registration.json": canonical_json_bytes(registration) + b"\n",
        "frozen_manifest.json": canonical_json_bytes(manifest) + b"\n",
    }
    for filename, content in expected.items():
        path = shard_root / filename
        if not path.is_file():
            raise RuntimeError(f"shard is missing {filename}: {shard_root}")
        if path.read_bytes() != content:
            raise RuntimeError(f"shard {filename} differs from the current frozen matrix: {path}")


def _prevalidate_canonical_collisions(
    output_root: Path,
    *,
    registration: dict[str, Any],
    manifest: dict[str, Any],
    copy_plan: list[tuple[Path, Path, str]],
) -> list[tuple[Path, bytes]]:
    expected = {
        output_root / "registration.json": canonical_json_bytes(registration) + b"\n",
        output_root / "frozen_manifest.json": canonical_json_bytes(manifest) + b"\n",
    }
    copy_payloads: list[tuple[Path, bytes]] = []
    for source, destination, expected_sha256 in copy_plan:
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise RuntimeError(f"shard source changed after validation: {source}")
        expected[destination] = content
        copy_payloads.append((destination, content))
    for destination, content in expected.items():
        if destination.exists() and destination.read_bytes() != content:
            raise ExistingArtifactMismatchError(
                f"canonical matrix contains a different existing file: {destination}"
            )
    return copy_payloads


def _merge_matrix_shards_locked(
    registration_path: Path,
    output_root: Path,
    shard_roots: list[Path] | tuple[Path, ...],
) -> dict[str, Any]:
    """Merge one complete set of self-identifying isolated shards."""

    resolved_output = output_root.resolve()
    resolved_shards = tuple(path.resolve() for path in shard_roots)
    if len(resolved_shards) < 2:
        raise ValueError("at least two shard roots are required")
    if any(not path.is_dir() for path in resolved_shards):
        raise FileNotFoundError("every shard root must be an existing directory")
    _reject_overlapping_roots(resolved_output, resolved_shards)

    description = _describe_matrix(registration_path)
    indexed_shards: dict[int, tuple[Path, dict[str, Any]]] = {}
    for shard_root in resolved_shards:
        _require_identical_control_files(
            shard_root,
            registration=description["registration"],
            manifest=description["manifest"],
        )
        completion_path = shard_root / "shard_completion.json"
        try:
            shard_completion = json.loads(completion_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot read shard completion index: {completion_path}") from error
        if not isinstance(shard_completion, dict):
            raise RuntimeError(f"shard completion index must be an object: {completion_path}")
        shard_count = shard_completion.get("shard_count")
        shard_index = shard_completion.get("shard_index")
        if (
            shard_count != len(resolved_shards)
            or isinstance(shard_index, bool)
            or not isinstance(shard_index, int)
            or shard_index < 0
            or shard_index >= len(resolved_shards)
            or shard_completion.get("partition_algorithm") != _SHARD_PARTITION_ALGORITHM
        ):
            raise RuntimeError(f"shard completion has an invalid partition: {completion_path}")
        if shard_index in indexed_shards:
            raise RuntimeError(f"duplicate shard index: {shard_index}")
        indexed_shards[shard_index] = (shard_root, shard_completion)
    if set(indexed_shards) != set(range(len(resolved_shards))):
        raise RuntimeError("shard indexes do not cover the complete partition")

    copy_plan: list[tuple[Path, Path, str]] = []
    for shard_index in range(len(resolved_shards)):
        shard_root, shard_completion = indexed_shards[shard_index]
        assigned_cells = _shard_cells(
            description["cells"],
            shard_count=len(resolved_shards),
            shard_index=shard_index,
        )
        _validate_matrix_root_inventory(
            shard_root,
            assigned_cells,
            require_all_cells=True,
            allow_final_files=False,
            allow_shard_files=True,
            require_role_files=True,
        )
        completed_cells = []
        for cell in assigned_cells:
            artifact_path = shard_root / cell["artifact_path"]
            log_path = artifact_path.with_suffix(".log")
            completed_cells.append(
                _validate_completed_cell(
                    cell=cell,
                    artifact_path=artifact_path,
                    log_path=log_path,
                    output_root=shard_root,
                    description=description,
                )
            )
            copy_plan.extend(
                (
                    (
                        artifact_path,
                        resolved_output / cell["artifact_path"],
                        completed_cells[-1]["artifact_sha256"],
                    ),
                    (
                        log_path,
                        (resolved_output / cell["artifact_path"]).with_suffix(".log"),
                        completed_cells[-1]["log_sha256"],
                    ),
                )
            )
        expected_shard_completion = {
            "schema_version": 1,
            "status": "shard_complete",
            "partition_algorithm": _SHARD_PARTITION_ALGORITHM,
            "registration_file_sha256": sha256_file(shard_root / "registration.json"),
            "manifest_file_sha256": sha256_file(shard_root / "frozen_manifest.json"),
            "manifest_sha256": description["manifest_sha256"],
            "shard_count": len(resolved_shards),
            "shard_index": shard_index,
            "planned_cells": len(description["cells"]),
            "assigned_cells": len(assigned_cells),
            "completed_cells": len(completed_cells),
            "cells": completed_cells,
        }
        completion_path = shard_root / "shard_completion.json"
        if (
            shard_completion != expected_shard_completion
            or completion_path.read_bytes()
            != canonical_json_bytes(expected_shard_completion) + b"\n"
        ):
            raise RuntimeError(
                f"shard completion index is not canonical or exact: {completion_path}"
            )
        validate_checksum_manifest(shard_root, filename="shard_checksums.sha256")

    if resolved_output.exists():
        _validate_matrix_root_inventory(
            resolved_output,
            description["cells"],
            require_all_cells=False,
            allow_final_files=True,
        )
        completion_path = resolved_output / "completion_index.json"
        checksum_path = resolved_output / "checksums.sha256"
        if checksum_path.exists() and not completion_path.is_file():
            raise RuntimeError("canonical checksum exists without its completion index")
        if completion_path.exists():
            _finalize_complete_matrix(description, resolved_output)
    copy_payloads = _prevalidate_canonical_collisions(
        resolved_output,
        registration=description["registration"],
        manifest=description["manifest"],
        copy_plan=copy_plan,
    )

    resolved_output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(resolved_output / "registration.json", description["registration"])
    atomic_write_json(resolved_output / "frozen_manifest.json", description["manifest"])
    _validate_matrix_root_inventory(
        resolved_output,
        description["cells"],
        require_all_cells=False,
        allow_final_files=True,
    )
    for destination, content in copy_payloads:
        atomic_write_bytes(destination, content)

    _validate_matrix_root_inventory(
        resolved_output,
        description["cells"],
        require_all_cells=True,
        allow_final_files=True,
    )
    return _finalize_complete_matrix(description, resolved_output)


def merge_matrix_shards(
    registration_path: Path,
    output_root: Path,
    shard_roots: list[Path] | tuple[Path, ...],
) -> dict[str, Any]:
    """Merge shards while excluding concurrent finalizers for the canonical root."""

    resolved_output = output_root.resolve()
    resolved_shards = tuple(path.resolve() for path in shard_roots)
    _reject_overlapping_roots(resolved_output, resolved_shards)
    lock_paths = {matrix_process_lock_path(root) for root in (resolved_output, *resolved_shards)}
    with ExitStack() as stack:
        for lock_path in sorted(lock_paths, key=lambda path: str(path).encode()):
            stack.enter_context(exclusive_process_lock(lock_path))
        return _merge_matrix_shards_locked(
            registration_path,
            resolved_output,
            resolved_shards,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--shard-root",
        type=Path,
        action="append",
        required=True,
        help="Shard stage root; repeat once per shard. Order is irrelevant.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            merge_matrix_shards(args.registration, args.output_root, args.shard_root),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
