"""Safe JSONL persistence for versioned wandering trajectory samples."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator

from elderly_monitoring.modules.mental_health.wandering.schemas import (
    TrajectorySample,
    TrajectorySchemaError,
)


class TrajectoryDatasetError(ValueError):
    """A contextual error while reading or writing a trajectory dataset."""


def iter_trajectory_jsonl(path: str | Path) -> Iterator[TrajectorySample]:
    """Yield validated samples from a safe JSONL file.

    Pickle and other executable or ambiguous serialization formats are
    intentionally not supported in the application environment.
    """

    dataset_path = _validated_jsonl_path(path)
    if not dataset_path.is_file():
        raise TrajectoryDatasetError(f"trajectory dataset not found: {dataset_path}")

    sample_ids: set[str] = set()
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise TrajectoryDatasetError(
                    f"{dataset_path}:{line_number}: blank JSONL line"
                )
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise TrajectoryDatasetError(
                    f"{dataset_path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(record, dict):
                raise TrajectoryDatasetError(
                    f"{dataset_path}:{line_number}: JSONL row must be an object"
                )
            try:
                sample = TrajectorySample.from_dict(record)
            except TrajectorySchemaError as exc:
                raise TrajectoryDatasetError(
                    f"{dataset_path}:{line_number}: {exc}"
                ) from exc
            if sample.sample_id in sample_ids:
                raise TrajectoryDatasetError(
                    f"{dataset_path}:{line_number}: duplicate sample_id "
                    f"{sample.sample_id!r}"
                )
            sample_ids.add(sample.sample_id)
            yield sample


def load_trajectory_jsonl(path: str | Path) -> tuple[TrajectorySample, ...]:
    """Load a complete, validated trajectory dataset."""

    return tuple(iter_trajectory_jsonl(path))


def write_trajectory_jsonl(
    path: str | Path,
    samples: Iterable[TrajectorySample],
) -> None:
    """Atomically write deterministic JSONL without exposing partial output."""

    output_path = _validated_jsonl_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    sample_ids: set[str] = set()
    sample_count = 0

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for sample in samples:
                if not isinstance(sample, TrajectorySample):
                    raise TrajectoryDatasetError(
                        "write_trajectory_jsonl expects TrajectorySample objects"
                    )
                if sample.sample_id in sample_ids:
                    raise TrajectoryDatasetError(
                        f"duplicate sample_id {sample.sample_id!r}"
                    )
                sample_ids.add(sample.sample_id)
                sample_count += 1
                serialized = json.dumps(
                    sample.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.write(serialized)
                handle.write("\n")
            if sample_count == 0:
                raise TrajectoryDatasetError("trajectory dataset must not be empty")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _validated_jsonl_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise TrajectoryDatasetError("trajectory dataset path must be a string or Path")
    dataset_path = Path(path)
    if dataset_path.suffix.lower() != ".jsonl":
        raise TrajectoryDatasetError(
            "only safe .jsonl trajectory datasets are supported; "
            "convert Pickle in an isolated environment first"
        )
    return dataset_path
