#!/usr/bin/env python3
"""One-shot, allowlisted WanderingPatterns pickle to inert JSON extractor.

This helper is intentionally standalone.  Production invocation must place it
inside the no-network/read-only mount namespace created by
``_run_isolated_pickle_extract.sh``.  The application converter never imports
this file and never accepts pickle input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import sys
import tempfile
from numbers import Integral, Real
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd


EXTRACT_SCHEMA_VERSION = "wandering-patterns-safe-extract-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_COLUMNS = frozenset(
    {
        "CartesianX",
        "CartesianY",
        "Coords",
        "Slope",
        "Path_Efficiency",
        "Coords_Slope",
        "pattern",
    }
)
_GROUP_FIELDS = frozenset(
    {
        "group",
        "group_id",
        "participant",
        "participant_id",
        "person_id",
        "session",
        "session_id",
        "subject",
        "subject_id",
        "user_id",
    }
)
_PATTERNS = frozenset({"direct", "pacing", "lapping", "random"})

# Exact constructors observed in the fixed upstream Pandas DataFrame pickle,
# plus NumPy 2 aliases needed by trusted, locally generated regression fixtures.
_ALLOWED_GLOBALS = frozenset(
    {
        ("builtins", "slice"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.numeric", "_frombuffer"),
        ("pandas._libs.internals", "_unpickle_block"),
        ("pandas", "DataFrame"),
        ("pandas", "Index"),
        ("pandas", "RangeIndex"),
        ("pandas.core.frame", "DataFrame"),
        ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.range", "RangeIndex"),
        ("pandas.core.internals.managers", "BlockManager"),
    }
)


class ExtractError(ValueError):
    pass


class _RestrictedUnpickler(pickle.Unpickler):
    def __init__(self, handle: BinaryIO) -> None:
        super().__init__(handle)
        self.loaded_globals: set[str] = set()

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED_GLOBALS:
            raise ExtractError(f"forbidden pickle global: {module}.{name}")
        self.loaded_globals.add(f"{module}.{name}")
        return super().find_class(module, name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a fixed-hash WanderingPatterns DataFrame to inert JSONL."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-metadata", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _extract(
            input_path=args.input,
            output_jsonl=args.output_jsonl,
            output_metadata=args.output_metadata,
            expected_sha256=args.expected_sha256,
        )
    except (ExtractError, OSError, pickle.UnpicklingError, ValueError) as exc:
        print(f"WanderingPatterns extraction failed: {exc}", file=sys.stderr)
        return 2
    return 0


def _extract(
    *,
    input_path: Path,
    output_jsonl: Path,
    output_metadata: Path,
    expected_sha256: str,
) -> None:
    if os.environ.get("WANDERING_PICKLE_ISOLATED") != "1":
        raise ExtractError(
            "isolation marker missing; run through _run_isolated_pickle_extract.sh"
        )
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ExtractError("expected SHA-256 must be lowercase 64-character hex")
    if input_path.suffix.lower() not in {".pkl", ".pickle"}:
        raise ExtractError("input must be an explicitly isolated .pkl or .pickle file")
    if not input_path.is_file():
        raise ExtractError(f"input not found: {input_path}")
    if output_jsonl.suffix.lower() != ".jsonl":
        raise ExtractError("output-jsonl must end with .jsonl")
    if output_metadata.suffix.lower() != ".json":
        raise ExtractError("output-metadata must end with .json")
    if output_jsonl == output_metadata:
        raise ExtractError("output paths must differ")
    if output_jsonl.exists() or output_metadata.exists():
        raise ExtractError("output paths must not already exist")

    actual_sha256 = _sha256_file(input_path)
    if actual_sha256 != expected_sha256:
        raise ExtractError(
            f"SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    with input_path.open("rb") as handle:
        unpickler = _RestrictedUnpickler(handle)
        value = unpickler.load()
        if handle.read(1):
            raise ExtractError("pickle contains trailing data")
    if type(value) is not pd.DataFrame:
        raise ExtractError("pickle root must be exactly pandas.DataFrame")

    columns = [str(column) for column in value.columns.tolist()]
    if len(set(columns)) != len(columns):
        raise ExtractError("DataFrame column names must be unique strings")
    missing_columns = sorted(_REQUIRED_COLUMNS - set(columns))
    if missing_columns:
        raise ExtractError(f"required DataFrame columns missing: {missing_columns}")
    group_fields = sorted(set(columns) & _GROUP_FIELDS)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    json_descriptor, json_temp_name = tempfile.mkstemp(
        dir=output_jsonl.parent,
        prefix=f".{output_jsonl.name}.",
        suffix=".tmp",
    )
    metadata_descriptor, metadata_temp_name = tempfile.mkstemp(
        dir=output_metadata.parent,
        prefix=f".{output_metadata.name}.",
        suffix=".tmp",
    )
    json_temp = Path(json_temp_name)
    metadata_temp = Path(metadata_temp_name)
    json_committed = False
    try:
        with os.fdopen(json_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record_index, (_, row) in enumerate(value.iterrows()):
                points = _validated_points(row, record_index)
                pattern = row["pattern"]
                if not isinstance(pattern, str) or pattern not in _PATTERNS:
                    raise ExtractError(
                        f"row {record_index}: pattern must be one of {sorted(_PATTERNS)}"
                    )
                group_values = {
                    field: _group_value(row[field], field, record_index)
                    for field in group_fields
                }
                payload = {
                    "record_index": record_index,
                    "pattern": pattern,
                    "points": [[x, y] for x, y in points],
                    "group_values": group_values,
                }
                handle.write(_canonical_json(payload))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        metadata = {
            "schema_version": EXTRACT_SCHEMA_VERSION,
            "source_sha256": actual_sha256,
            "record_count": len(value),
            "columns": columns,
            "group_fields_found": group_fields,
            "pickle_globals_loaded": sorted(unpickler.loaded_globals),
        }
        with os.fdopen(metadata_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_json(metadata))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(json_temp, output_jsonl)
        json_committed = True
        try:
            os.replace(metadata_temp, output_metadata)
        except Exception:
            output_jsonl.unlink(missing_ok=True)
            raise
    finally:
        if not json_committed:
            try:
                os.close(json_descriptor)
            except OSError:
                pass
        try:
            os.close(metadata_descriptor)
        except OSError:
            pass
        json_temp.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)


def _validated_points(row: pd.Series, record_index: int) -> list[tuple[float, float]]:
    raw_x = np.asarray(row["CartesianX"])
    raw_y = np.asarray(row["CartesianY"])
    x_values = _finite_vector(raw_x, f"row {record_index} CartesianX")
    y_values = _finite_vector(raw_y, f"row {record_index} CartesianY")
    coords_value = row["Coords"]
    if isinstance(coords_value, (str, bytes, bytearray, dict)):
        raise ExtractError(f"row {record_index}: Coords must be a point sequence")
    try:
        coordinate_count = len(coords_value)
    except TypeError as exc:
        raise ExtractError(
            f"row {record_index}: Coords must be a point sequence"
        ) from exc
    if len(x_values) != len(y_values) or len(x_values) != coordinate_count:
        raise ExtractError(
            f"row {record_index}: CartesianX/CartesianY/Coords lengths differ"
        )
    if len(x_values) < 2:
        raise ExtractError(f"row {record_index}: trajectory has fewer than two points")
    points: list[tuple[float, float]] = []
    for point_index, point in enumerate(coords_value):
        point_array = np.asarray(point)
        if point_array.ndim != 1 or point_array.shape[0] != 2:
            raise ExtractError(
                f"row {record_index}: Coords[{point_index}] must contain x and y"
            )
        x = _finite_float(
            point_array[0], f"row {record_index} Coords[{point_index}].x"
        )
        y = _finite_float(
            point_array[1], f"row {record_index} Coords[{point_index}].y"
        )
        if not math.isclose(x, x_values[point_index], rel_tol=0.0, abs_tol=1e-12):
            raise ExtractError(
                f"row {record_index}: Coords x differs from CartesianX at {point_index}"
            )
        if not math.isclose(y, y_values[point_index], rel_tol=0.0, abs_tol=1e-12):
            raise ExtractError(
                f"row {record_index}: Coords y differs from CartesianY at {point_index}"
            )
        points.append((x, y))
    return points


def _finite_vector(value: Any, field: str) -> list[float]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ExtractError(f"{field} must be one-dimensional")
    return [_finite_float(item, f"{field}[{index}]") for index, item in enumerate(array)]


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ExtractError(f"{field} must be a finite real number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ExtractError(f"{field} must be a finite real number")
    return parsed


def _group_value(value: Any, field: str, record_index: int) -> str | int:
    if isinstance(value, (bool, np.bool_)):
        raise ExtractError(f"row {record_index}: group field {field} cannot be boolean")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and value and value == value.strip():
        return value
    raise ExtractError(
        f"row {record_index}: group field {field} must be integer or trimmed string"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
