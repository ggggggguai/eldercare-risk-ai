"""Deterministic, source-isolated converters for public wandering datasets.

SmartCare is parsed as inert JSON.  WanderingPatterns is deliberately accepted
only as a safe JSONL extract produced by the separate no-network pickle helper;
this application module never imports :mod:`pickle` and never deserializes a
pickle payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from elderly_monitoring.modules.mental_health.wandering.datasets import (
    load_trajectory_jsonl,
    write_trajectory_jsonl,
)
from elderly_monitoring.modules.mental_health.wandering.manifests import (
    CONVERSION_REPORT_SCHEMA_VERSION,
    SOURCE_MANIFEST_SCHEMA_VERSION,
    ConversionManifest,
    ConversionReport,
    ConversionWarning,
    OutputArtifact,
    SourceFile,
    derive_sample_id,
    describe_source_file,
    validate_conversion_bundle,
    write_conversion_manifest,
    write_conversion_report,
)
from elderly_monitoring.modules.mental_health.wandering.schemas import (
    TRAJECTORY_SAMPLE_SCHEMA_VERSION,
    CoordinateSystem,
    PatternLabel,
    TrajectorySample,
    TrajectorySchemaError,
)


SMARTCARE_CONVERTER_VERSION = "wandering-smartcare-converter-v1"
WANDERING_PATTERNS_CONVERTER_VERSION = "wandering-patterns-converter-v1"
WANDERING_PATTERNS_EXTRACT_SCHEMA_VERSION = "wandering-patterns-safe-extract-v1"
CONVERSION_REJECTIONS_SCHEMA_VERSION = "wandering-conversion-rejections-v1"

_SMARTCARE_FIELDS = frozenset({"date", "x", "y", "stress"})
_SAFE_EXTRACT_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "source_sha256",
        "record_count",
        "columns",
        "group_fields_found",
        "pickle_globals_loaded",
    }
)
_SAFE_EXTRACT_ROW_FIELDS = frozenset(
    {"record_index", "pattern", "points", "group_values"}
)
_WANDERING_REQUIRED_COLUMNS = frozenset(
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
_GROUP_FIELD_CANDIDATES = frozenset(
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
_PATTERN_TO_BINARY = {
    PatternLabel.DIRECT: 0,
    PatternLabel.PACING: 1,
    PatternLabel.LAPPING: 1,
    PatternLabel.RANDOM: 1,
}


class WanderingConversionError(ValueError):
    """A contextual, fail-closed public trajectory conversion error."""


@dataclass(frozen=True)
class ConversionBundle:
    """A committed conversion bundle and its validated in-memory contracts."""

    output_dir: Path
    manifest: ConversionManifest
    report: ConversionReport


@dataclass(frozen=True)
class _Rejection:
    sample_id: str
    source_role: str
    source_path: str
    source_record_id: str
    codes: tuple[str, ...]
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_role": self.source_role,
            "source_path": self.source_path,
            "source_record_id": self.source_record_id,
            "codes": list(self.codes),
            "details": dict(self.details),
        }


def convert_smartcare(
    source_root: str | Path,
    output_dir: str | Path,
    *,
    main_relative_path: str = "raw/dataset.json",
    validation_relative_path: str = "raw/dataset-validacao.json",
    minimum_points: int = 4,
    canvas_width: int = 602,
    canvas_height: int = 369,
) -> ConversionBundle:
    """Convert both SmartCare source files without guessing time or subtype.

    Groups with fewer than ``minimum_points`` or any point outside the stated
    602 x 369 source canvas are rejected as complete trajectories.  They are
    never clipped, padded, or silently omitted.
    """

    source_path, final_output = _validated_roots(source_root, output_dir)
    if isinstance(minimum_points, bool) or not isinstance(minimum_points, int):
        raise WanderingConversionError("minimum_points must be an integer")
    if minimum_points < 2:
        raise WanderingConversionError("minimum_points must be at least 2")
    if canvas_width < 2 or canvas_height < 2:
        raise WanderingConversionError("canvas dimensions must be at least 2")

    source_specs = (
        ("train_pool", main_relative_path),
        ("official_validation", validation_relative_path),
    )
    source_files = tuple(
        describe_source_file(
            source_path / relative_path,
            source_root=source_path,
            role=role,
        )
        for role, relative_path in source_specs
    )
    source_by_role = {item.role: item for item in source_files}

    samples_by_role: dict[str, list[TrajectorySample]] = defaultdict(list)
    rejections: list[_Rejection] = []
    raw_finite_points: list[tuple[float, float]] = []
    written_points: list[tuple[float, float]] = []
    out_of_bounds_point_count = 0
    invalid_coordinate_count = 0
    sample_count_read = 0

    for role, relative_path in source_specs:
        source_file = source_by_role[role]
        rows = _load_smartcare_rows(source_path / relative_path)
        grouped_rows = _group_smartcare_rows(rows, source_path / relative_path)
        sample_count_read += len(grouped_rows)
        for trajectory_id, trajectory_rows in grouped_rows:
            sample_id = derive_sample_id(
                source_dataset="smartcare",
                source_role=role,
                source_sha256=source_file.sha256,
                source_record_id=trajectory_id,
            )
            codes: set[str] = set()
            details: dict[str, Any] = {"point_count": len(trajectory_rows)}
            labels = [row["stress"] for row in trajectory_rows]
            if any(not isinstance(label, bool) for label in labels) or len(set(labels)) != 1:
                codes.add("inconsistent_binary_label")

            parsed_points: list[tuple[float, float]] = []
            invalid_indices: list[int] = []
            out_of_bounds: list[dict[str, float | int]] = []
            for point_index, row in enumerate(trajectory_rows):
                point = _optional_finite_point(row["x"], row["y"])
                if point is None:
                    invalid_indices.append(point_index)
                    invalid_coordinate_count += 1
                    continue
                x, y = point
                parsed_points.append(point)
                raw_finite_points.append(point)
                if not (0.0 <= x <= canvas_width - 1 and 0.0 <= y <= canvas_height - 1):
                    out_of_bounds.append(
                        {"point_index": point_index, "x": x, "y": y}
                    )
                    out_of_bounds_point_count += 1
            if invalid_indices:
                codes.add("invalid_coordinate")
                details["invalid_coordinate_indices"] = invalid_indices
            if out_of_bounds:
                codes.add("coordinate_out_of_bounds")
                details["out_of_bounds_points"] = out_of_bounds
            if len(trajectory_rows) < minimum_points:
                codes.add("trajectory_too_short")

            if codes:
                rejections.append(
                    _Rejection(
                        sample_id=sample_id,
                        source_role=role,
                        source_path=source_file.path,
                        source_record_id=trajectory_id,
                        codes=tuple(sorted(codes)),
                        details=details,
                    )
                )
                continue

            binary_label = int(labels[0])
            normalized_points = tuple(
                (x / (canvas_width - 1), y / (canvas_height - 1))
                for x, y in parsed_points
            )
            written_points.extend(normalized_points)
            samples_by_role[role].append(
                TrajectorySample(
                    schema_version=TRAJECTORY_SAMPLE_SCHEMA_VERSION,
                    sample_id=sample_id,
                    source_dataset="smartcare",
                    source_group_id=trajectory_id,
                    coordinate_system=CoordinateSystem.IMAGE_NORMALIZED,
                    points=normalized_points,
                    point_times_sec=None,
                    point_mask=(1,) * len(normalized_points),
                    binary_label=binary_label,
                    pattern_label=PatternLabel.UNKNOWN,
                    quality_flags=("time_unavailable",),
                    source_path=source_file.path,
                    source_sha256=source_file.sha256,
                )
            )

    for samples in samples_by_role.values():
        samples.sort(key=lambda sample: sample.sample_id)
    all_samples = tuple(
        sample
        for role in ("train_pool", "official_validation")
        for sample in samples_by_role[role]
    )
    if not all_samples:
        raise WanderingConversionError("SmartCare conversion produced no valid samples")

    warnings = _conversion_warnings(
        rejections,
        extra=(
            ConversionWarning(
                code="point_time_unavailable",
                count=len(all_samples),
                sample_ids=(),
            ),
        ),
    )
    class_counts = Counter(
        "wandering_like" if sample.binary_label == 1 else "normal"
        for sample in all_samples
    )
    length_summary = _length_summary(all_samples)
    coordinate_summary = {
        **_coordinate_extrema(raw_finite_points, prefix="source"),
        **_coordinate_extrema(written_points, prefix="normalized"),
        "invalid_coordinate_count": invalid_coordinate_count,
        "out_of_bounds_point_count": out_of_bounds_point_count,
    }

    def build(stage: Path) -> tuple[ConversionManifest, ConversionReport]:
        write_trajectory_jsonl(stage / "train_pool.jsonl", samples_by_role["train_pool"])
        write_trajectory_jsonl(
            stage / "official_validation.jsonl",
            samples_by_role["official_validation"],
        )
        _write_rejections(stage / "rejections.json", "smartcare", rejections)
        _verify_sources_unchanged(source_path, source_files)
        return _write_bundle_contracts(
            stage=stage,
            source_name="smartcare",
            converter_version=SMARTCARE_CONVERTER_VERSION,
            source_files=source_files,
            sample_count_read=sample_count_read,
            samples=all_samples,
            samples_by_output={
                "train_pool": samples_by_role["train_pool"],
                "official_validation": samples_by_role["official_validation"],
            },
            rejections=rejections,
            class_counts=class_counts,
            length_summary=length_summary,
            coordinate_summary=coordinate_summary,
            group_fields_found=("date",),
            warnings=warnings,
        )

    bundle = _commit_bundle(final_output, build)
    _verify_sources_unchanged(source_path, source_files)
    return bundle


def convert_wandering_patterns_safe_extract(
    source_root: str | Path,
    output_dir: str | Path,
    *,
    extracted_jsonl: str | Path,
    extraction_metadata: str | Path,
    primary_relative_path: str = "raw/patterns_dataset.pkl",
    duplicate_relative_path: str | None = None,
    minimum_points: int = 4,
) -> ConversionBundle:
    """Convert an inert WanderingPatterns extract; never accept a pickle here."""

    source_path, final_output = _validated_roots(source_root, output_dir)
    extract_path = Path(extracted_jsonl)
    metadata_path = Path(extraction_metadata)
    if extract_path.suffix.lower() != ".jsonl":
        raise WanderingConversionError(
            "WanderingPatterns conversion requires a safe .jsonl extract; pickle is refused"
        )
    if metadata_path.suffix.lower() != ".json":
        raise WanderingConversionError("extraction_metadata must be a .json file")
    if not extract_path.is_file():
        raise WanderingConversionError(f"safe extract not found: {extract_path}")
    if not metadata_path.is_file():
        raise WanderingConversionError(f"extraction metadata not found: {metadata_path}")

    source_specs: list[tuple[str, str]] = [("primary_pickle", primary_relative_path)]
    if duplicate_relative_path is not None:
        source_specs.append(("duplicate_pickle", duplicate_relative_path))
    source_files = tuple(
        describe_source_file(
            source_path / relative_path,
            source_root=source_path,
            role=role,
        )
        for role, relative_path in source_specs
    )
    source_by_role = {item.role: item for item in source_files}
    primary_source = source_by_role["primary_pickle"]
    if "duplicate_pickle" in source_by_role and (
        source_by_role["duplicate_pickle"].sha256 != primary_source.sha256
    ):
        raise WanderingConversionError(
            "WanderingPatterns primary and duplicate pickle SHA-256 values differ"
        )

    metadata = _load_extract_metadata(metadata_path)
    if metadata["source_sha256"] != primary_source.sha256:
        raise WanderingConversionError(
            "safe extract source_sha256 does not match the primary pickle"
        )
    group_fields = tuple(sorted(metadata["group_fields_found"]))
    rows = _load_extract_rows(extract_path)
    if len(rows) != metadata["record_count"]:
        raise WanderingConversionError(
            "safe extract record_count does not match JSONL row count"
        )

    samples: list[TrajectorySample] = []
    rejections: list[_Rejection] = []
    finite_points: list[tuple[float, float]] = []
    for row in rows:
        record_index = row["record_index"]
        source_record_id = str(record_index)
        sample_id = derive_sample_id(
            source_dataset="wandering_patterns",
            source_role="primary_pickle",
            source_sha256=primary_source.sha256,
            source_record_id=source_record_id,
        )
        codes: set[str] = set()
        details: dict[str, Any] = {}
        try:
            pattern_label = PatternLabel(row["pattern"])
        except (TypeError, ValueError):
            pattern_label = PatternLabel.UNKNOWN
            codes.add("unsupported_pattern_label")
        if pattern_label not in _PATTERN_TO_BINARY:
            codes.add("unsupported_pattern_label")

        points, invalid_indices = _parse_inert_points(row["points"])
        details["point_count"] = len(row["points"]) if isinstance(row["points"], list) else 0
        if invalid_indices:
            codes.add("invalid_coordinate")
            details["invalid_coordinate_indices"] = invalid_indices
        if len(points) < minimum_points:
            codes.add("trajectory_too_short")
        finite_points.extend(points)

        group_values = row["group_values"]
        if not isinstance(group_values, dict) or set(group_values) != set(group_fields):
            raise WanderingConversionError(
                f"safe extract row {record_index} group_values do not match metadata"
            )
        source_group_id = _source_group_id(group_fields, group_values)
        if codes:
            rejections.append(
                _Rejection(
                    sample_id=sample_id,
                    source_role="primary_pickle",
                    source_path=primary_source.path,
                    source_record_id=source_record_id,
                    codes=tuple(sorted(codes)),
                    details=details,
                )
            )
            continue

        try:
            samples.append(
                TrajectorySample(
                    schema_version=TRAJECTORY_SAMPLE_SCHEMA_VERSION,
                    sample_id=sample_id,
                    source_dataset="wandering_patterns",
                    source_group_id=source_group_id,
                    coordinate_system=CoordinateSystem.SOURCE_NATIVE,
                    points=tuple(points),
                    point_times_sec=None,
                    point_mask=(1,) * len(points),
                    binary_label=_PATTERN_TO_BINARY[pattern_label],
                    pattern_label=pattern_label,
                    quality_flags=("time_unavailable",),
                    source_path=primary_source.path,
                    source_sha256=primary_source.sha256,
                )
            )
        except TrajectorySchemaError as exc:
            raise WanderingConversionError(
                f"safe extract row {record_index} violates TrajectorySample v1: {exc}"
            ) from exc

    samples.sort(key=lambda sample: sample.sample_id)
    if not samples:
        raise WanderingConversionError(
            "WanderingPatterns safe extract produced no valid samples"
        )
    extra_warnings = [
        ConversionWarning(
            code="point_time_unavailable",
            count=len(samples),
            sample_ids=(),
        )
    ]
    if not group_fields:
        extra_warnings.append(
            ConversionWarning(
                code="group_field_unavailable",
                count=len(samples),
                sample_ids=(),
            )
        )
    warnings = _conversion_warnings(rejections, extra=tuple(extra_warnings))
    class_counts = Counter(sample.pattern_label.value for sample in samples)
    coordinate_summary = {
        **_coordinate_extrema(finite_points, prefix="source"),
        "invalid_coordinate_count": sum(
            len(item.details.get("invalid_coordinate_indices", ()))
            for item in rejections
        ),
        "out_of_bounds_point_count": 0,
    }

    def build(stage: Path) -> tuple[ConversionManifest, ConversionReport]:
        write_trajectory_jsonl(stage / "samples.jsonl", samples)
        _write_rejections(
            stage / "rejections.json", "wandering_patterns", rejections
        )
        _verify_sources_unchanged(source_path, source_files)
        return _write_bundle_contracts(
            stage=stage,
            source_name="wandering_patterns",
            converter_version=WANDERING_PATTERNS_CONVERTER_VERSION,
            source_files=source_files,
            sample_count_read=len(rows),
            samples=tuple(samples),
            samples_by_output={"samples": tuple(samples)},
            rejections=rejections,
            class_counts=class_counts,
            length_summary=_length_summary(samples),
            coordinate_summary=coordinate_summary,
            group_fields_found=group_fields,
            warnings=warnings,
        )

    bundle = _commit_bundle(final_output, build)
    _verify_sources_unchanged(source_path, source_files)
    return bundle


def _validated_roots(
    source_root: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    source_path = Path(source_root).resolve(strict=True)
    if not source_path.is_dir():
        raise WanderingConversionError(f"source_root is not a directory: {source_path}")
    final_output = Path(output_dir).resolve(strict=False)
    if final_output == source_path or source_path in final_output.parents:
        raise WanderingConversionError("output_dir must remain outside source_root")
    if final_output.exists():
        raise WanderingConversionError(f"output_dir already exists: {final_output}")
    return source_path, final_output


def _load_smartcare_rows(path: Path) -> list[dict[str, Any]]:
    try:
        outer = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WanderingConversionError(f"{path}: invalid outer JSON") from exc
    if not isinstance(outer, str):
        raise WanderingConversionError(f"{path}: outer JSON must be a string")
    try:
        rows = json.loads(outer)
    except json.JSONDecodeError as exc:
        raise WanderingConversionError(f"{path}: invalid inner JSON") from exc
    if not isinstance(rows, list):
        raise WanderingConversionError(f"{path}: inner JSON must be an array")
    parsed: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise WanderingConversionError(f"{path}: row {index} must be an object")
        if set(row) != _SMARTCARE_FIELDS:
            unknown = sorted(set(row) - _SMARTCARE_FIELDS)
            missing = sorted(_SMARTCARE_FIELDS - set(row))
            raise WanderingConversionError(
                f"{path}: row {index} fields differ; missing={missing}, unknown={unknown}"
            )
        trajectory_id = row["date"]
        if (
            not isinstance(trajectory_id, str)
            or not trajectory_id
            or trajectory_id != trajectory_id.strip()
        ):
            raise WanderingConversionError(
                f"{path}: row {index} date must be a non-empty trimmed string"
            )
        parsed.append(row)
    return parsed


def _group_smartcare_rows(
    rows: Sequence[dict[str, Any]],
    path: Path,
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        trajectory_id = row["date"]
        grouped.setdefault(trajectory_id, []).append(row)
    if not grouped:
        raise WanderingConversionError(f"{path}: no trajectory groups")
    return list(grouped.items())


def _optional_finite_point(x_value: Any, y_value: Any) -> tuple[float, float] | None:
    values = (x_value, y_value)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        return None
    x, y = (float(value) for value in values)
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _load_extract_metadata(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WanderingConversionError(f"{path}: invalid extraction metadata JSON") from exc
    if not isinstance(record, dict) or set(record) != _SAFE_EXTRACT_METADATA_FIELDS:
        raise WanderingConversionError(f"{path}: extraction metadata fields differ")
    if record["schema_version"] != WANDERING_PATTERNS_EXTRACT_SCHEMA_VERSION:
        raise WanderingConversionError(f"{path}: unsupported extraction schema_version")
    digest = record["source_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise WanderingConversionError(f"{path}: invalid source_sha256")
    count = record["record_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise WanderingConversionError(f"{path}: invalid record_count")
    columns = record["columns"]
    if (
        not isinstance(columns, list)
        or any(not isinstance(item, str) or not item for item in columns)
        or len(set(columns)) != len(columns)
    ):
        raise WanderingConversionError(f"{path}: invalid columns")
    missing = sorted(_WANDERING_REQUIRED_COLUMNS - set(columns))
    if missing:
        raise WanderingConversionError(
            f"{path}: required WanderingPatterns columns missing: {missing}"
        )
    group_fields = record["group_fields_found"]
    if (
        not isinstance(group_fields, list)
        or any(not isinstance(item, str) for item in group_fields)
        or len(set(group_fields)) != len(group_fields)
        or not set(group_fields).issubset(set(columns))
        or not set(group_fields).issubset(_GROUP_FIELD_CANDIDATES)
    ):
        raise WanderingConversionError(f"{path}: invalid group_fields_found")
    globals_loaded = record["pickle_globals_loaded"]
    if not isinstance(globals_loaded, list) or any(
        not isinstance(item, str) or not item for item in globals_loaded
    ):
        raise WanderingConversionError(f"{path}: invalid pickle_globals_loaded")
    return record


def _load_extract_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    indexes: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise WanderingConversionError(f"{path}:{line_number}: blank JSONL line")
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise WanderingConversionError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(row, dict) or set(row) != _SAFE_EXTRACT_ROW_FIELDS:
                raise WanderingConversionError(
                    f"{path}:{line_number}: safe extract row fields differ"
                )
            index = row["record_index"]
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise WanderingConversionError(
                    f"{path}:{line_number}: record_index must be non-negative integer"
                )
            if index in indexes:
                raise WanderingConversionError(
                    f"{path}:{line_number}: duplicate record_index {index}"
                )
            indexes.add(index)
            rows.append(row)
    if indexes != set(range(len(rows))):
        raise WanderingConversionError(
            "safe extract record_index values must be contiguous from zero"
        )
    rows.sort(key=lambda row: row["record_index"])
    return rows


def _parse_inert_points(value: Any) -> tuple[list[tuple[float, float]], list[int]]:
    if not isinstance(value, list):
        return [], [0]
    points: list[tuple[float, float]] = []
    invalid_indices: list[int] = []
    for index, point in enumerate(value):
        if not isinstance(point, list) or len(point) != 2:
            invalid_indices.append(index)
            continue
        parsed = _optional_finite_point(point[0], point[1])
        if parsed is None:
            invalid_indices.append(index)
            continue
        points.append(parsed)
    return points, invalid_indices


def _source_group_id(
    group_fields: Sequence[str],
    group_values: Mapping[str, Any],
) -> str | None:
    if not group_fields:
        return None
    parsed: list[tuple[str, str]] = []
    for field in group_fields:
        value = group_values[field]
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise WanderingConversionError(
                f"group field {field!r} must be a non-empty string or integer"
            )
        text = str(value)
        if not text or text != text.strip():
            raise WanderingConversionError(
                f"group field {field!r} must be a non-empty trimmed value"
            )
        parsed.append((field, text))
    if len(parsed) == 1:
        return parsed[0][1]
    return "|".join(f"{field}={value}" for field, value in parsed)


def _conversion_warnings(
    rejections: Iterable[_Rejection],
    *,
    extra: Sequence[ConversionWarning],
) -> tuple[ConversionWarning, ...]:
    sample_ids_by_code: dict[str, list[str]] = defaultdict(list)
    for rejection in rejections:
        for code in rejection.codes:
            sample_ids_by_code[code].append(rejection.sample_id)
    warnings = [
        ConversionWarning(
            code=code,
            count=len(sample_ids),
            sample_ids=tuple(sample_ids),
        )
        for code, sample_ids in sample_ids_by_code.items()
    ]
    warnings.extend(extra)
    return tuple(warnings)


def _length_summary(samples: Sequence[TrajectorySample]) -> dict[str, int | float]:
    lengths = [len(sample.points) for sample in samples]
    if not lengths:
        return {"count": 0, "max": 0, "mean": 0.0, "median": 0.0, "min": 0, "total_points": 0}
    return {
        "count": len(lengths),
        "max": max(lengths),
        "mean": statistics.fmean(lengths),
        "median": float(statistics.median(lengths)),
        "min": min(lengths),
        "total_points": sum(lengths),
    }


def _coordinate_extrema(
    points: Sequence[tuple[float, float]],
    *,
    prefix: str,
) -> dict[str, int | float]:
    if not points:
        return {
            f"{prefix}_point_count": 0,
            f"{prefix}_x_max": 0.0,
            f"{prefix}_x_min": 0.0,
            f"{prefix}_y_max": 0.0,
            f"{prefix}_y_min": 0.0,
        }
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    return {
        f"{prefix}_point_count": len(points),
        f"{prefix}_x_max": max(x_values),
        f"{prefix}_x_min": min(x_values),
        f"{prefix}_y_max": max(y_values),
        f"{prefix}_y_min": min(y_values),
    }


def _write_bundle_contracts(
    *,
    stage: Path,
    source_name: str,
    converter_version: str,
    source_files: Sequence[SourceFile],
    sample_count_read: int,
    samples: Sequence[TrajectorySample],
    samples_by_output: Mapping[str, Sequence[TrajectorySample]],
    rejections: Sequence[_Rejection],
    class_counts: Mapping[str, int],
    length_summary: Mapping[str, int | float],
    coordinate_summary: Mapping[str, int | float],
    group_fields_found: Sequence[str],
    warnings: Sequence[ConversionWarning],
) -> tuple[ConversionManifest, ConversionReport]:
    output_artifacts = [
        OutputArtifact(
            role=role,
            path=f"{role}.jsonl",
            byte_count=(stage / f"{role}.jsonl").stat().st_size,
            sha256=_sha256_file(stage / f"{role}.jsonl"),
            sample_count=len(output_samples),
        )
        for role, output_samples in samples_by_output.items()
    ]
    rejection_path = stage / "rejections.json"
    output_artifacts.append(
        OutputArtifact(
            role="rejections",
            path="rejections.json",
            byte_count=rejection_path.stat().st_size,
            sha256=_sha256_file(rejection_path),
            sample_count=0,
        )
    )
    manifest = ConversionManifest(
        schema_version=SOURCE_MANIFEST_SCHEMA_VERSION,
        source_name=source_name,
        converter_version=converter_version,
        source_files=tuple(source_files),
        outputs=tuple(output_artifacts),
        sample_count_read=sample_count_read,
        sample_count_written=len(samples),
        sample_count_rejected=len(rejections),
        sample_ids=tuple(sample.sample_id for sample in samples),
        warnings=tuple(warnings),
    )
    report = ConversionReport(
        schema_version=CONVERSION_REPORT_SCHEMA_VERSION,
        source_name=source_name,
        converter_version=converter_version,
        source_sha256=tuple(item.sha256 for item in source_files),
        sample_count_read=sample_count_read,
        sample_count_written=len(samples),
        sample_count_rejected=len(rejections),
        class_counts=dict(class_counts),
        length_summary=dict(length_summary),
        coordinate_summary=dict(coordinate_summary),
        group_fields_found=tuple(group_fields_found),
        warnings=tuple(warnings),
        output_sha256={item.role: item.sha256 for item in output_artifacts},
    )
    validate_conversion_bundle(manifest, report)
    for artifact in output_artifacts:
        if artifact.path.endswith(".jsonl"):
            loaded = load_trajectory_jsonl(stage / artifact.path)
            if len(loaded) != artifact.sample_count:
                raise WanderingConversionError(
                    f"strict reader count mismatch for {artifact.path}"
                )
    write_conversion_manifest(stage / "manifest.json", manifest)
    write_conversion_report(stage / "conversion_report.json", report)
    return manifest, report


def _write_rejections(
    path: Path,
    source_name: str,
    rejections: Sequence[_Rejection],
) -> None:
    payload = {
        "schema_version": CONVERSION_REJECTIONS_SCHEMA_VERSION,
        "source_name": source_name,
        "records": [
            item.to_dict() for item in sorted(rejections, key=lambda item: item.sample_id)
        ],
    }
    _atomic_write_bytes(path, _canonical_json_bytes(payload))


def _commit_bundle(
    final_output: Path,
    builder: Any,
) -> ConversionBundle:
    final_output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            dir=final_output.parent,
            prefix=f".{final_output.name}.",
            suffix=".tmp",
        )
    )
    try:
        manifest, report = builder(stage)
        os.replace(stage, final_output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return ConversionBundle(
        output_dir=final_output,
        manifest=manifest,
        report=report,
    )


def _verify_sources_unchanged(
    source_root: Path,
    source_files: Sequence[SourceFile],
) -> None:
    for expected in source_files:
        actual = describe_source_file(
            source_root / expected.path,
            source_root=source_root,
            role=expected.role,
        )
        if actual != expected:
            raise WanderingConversionError(
                f"raw source changed during conversion: {expected.path}"
            )


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
