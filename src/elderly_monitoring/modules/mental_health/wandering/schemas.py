"""Versioned, dependency-light contracts for wandering trajectory data."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


TRAJECTORY_SAMPLE_SCHEMA_VERSION = "wandering-trajectory-sample-v1"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_DATASET_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_QUALITY_FLAG_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(r"^[A-Za-z]:[/\\]")
_TRAJECTORY_SAMPLE_FIELDS = frozenset(
    {
        "schema_version",
        "sample_id",
        "source_dataset",
        "source_group_id",
        "coordinate_system",
        "points",
        "point_times_sec",
        "point_mask",
        "binary_label",
        "pattern_label",
        "quality_flags",
        "source_path",
        "source_sha256",
    }
)


class TrajectorySchemaError(ValueError):
    """A field-level error in the internal wandering trajectory contract."""


class CoordinateSystem(str, Enum):
    """Coordinate spaces that must never be silently mixed."""

    SOURCE_NATIVE = "source_native"
    SHAPE_NORMALIZED = "shape_normalized"
    IMAGE_NORMALIZED = "image_normalized"


class PatternLabel(str, Enum):
    """Supported four-class labels plus an explicit unknown value."""

    DIRECT = "direct"
    PACING = "pacing"
    LAPPING = "lapping"
    RANDOM = "random"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TrajectorySample:
    """One immutable trajectory sample used by wandering research code.

    Missing or padded positions must use finite placeholders in ``points`` and
    a zero in ``point_mask``. Consumers must apply the mask before computing
    trajectory features.
    """

    schema_version: str
    sample_id: str
    source_dataset: str
    source_group_id: str | None
    coordinate_system: CoordinateSystem
    points: tuple[tuple[float, float], ...]
    point_times_sec: tuple[float, ...] | None
    point_mask: tuple[int, ...]
    binary_label: int | None
    pattern_label: PatternLabel
    quality_flags: tuple[str, ...]
    source_path: str
    source_sha256: str

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        _validate_identifier(self.sample_id, "sample_id")
        _validate_source_dataset(self.source_dataset)
        if self.source_group_id is not None:
            _validate_identifier(self.source_group_id, "source_group_id")
        if not isinstance(self.coordinate_system, CoordinateSystem):
            raise TrajectorySchemaError("coordinate_system must be a supported enum value")
        _validate_points(self.points, self.coordinate_system)
        _validate_point_mask(self.point_mask, len(self.points))
        _validate_point_times(self.point_times_sec, len(self.points))
        _validate_binary_label(self.binary_label)
        if not isinstance(self.pattern_label, PatternLabel):
            raise TrajectorySchemaError("pattern_label must be a supported enum value")
        _validate_label_consistency(self.pattern_label, self.binary_label)
        _validate_quality_flags(self.quality_flags)
        _validate_source_path(self.source_path)
        _validate_source_sha256(self.source_sha256)

    @property
    def valid_point_count(self) -> int:
        return sum(self.point_mask)

    @property
    def is_wandering(self) -> bool | None:
        if self.binary_label is None:
            return None
        return bool(self.binary_label)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the exact v1 schema."""

        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "source_dataset": self.source_dataset,
            "source_group_id": self.source_group_id,
            "coordinate_system": self.coordinate_system.value,
            "points": [[x, y] for x, y in self.points],
            "point_times_sec": (
                list(self.point_times_sec) if self.point_times_sec is not None else None
            ),
            "point_mask": list(self.point_mask),
            "binary_label": self.binary_label,
            "pattern_label": self.pattern_label.value,
            "quality_flags": list(self.quality_flags),
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "TrajectorySample":
        """Parse an untrusted mapping without accepting silent schema drift."""

        if not isinstance(record, Mapping):
            raise TrajectorySchemaError("trajectory sample must be an object")
        unknown_fields = sorted(set(record) - _TRAJECTORY_SAMPLE_FIELDS)
        missing_fields = sorted(_TRAJECTORY_SAMPLE_FIELDS - set(record))
        if unknown_fields:
            raise TrajectorySchemaError(
                f"trajectory sample has unknown fields: {', '.join(unknown_fields)}"
            )
        if missing_fields:
            raise TrajectorySchemaError(
                f"trajectory sample is missing fields: {', '.join(missing_fields)}"
            )

        return cls(
            schema_version=record["schema_version"],
            sample_id=record["sample_id"],
            source_dataset=record["source_dataset"],
            source_group_id=record["source_group_id"],
            coordinate_system=_parse_coordinate_system(record["coordinate_system"]),
            points=_parse_points(record["points"]),
            point_times_sec=_parse_point_times(record["point_times_sec"]),
            point_mask=_parse_point_mask(record["point_mask"]),
            binary_label=record["binary_label"],
            pattern_label=_parse_pattern_label(record["pattern_label"]),
            quality_flags=_parse_quality_flags(record["quality_flags"]),
            source_path=record["source_path"],
            source_sha256=record["source_sha256"],
        )


def _validate_schema_version(value: Any) -> None:
    if value != TRAJECTORY_SAMPLE_SCHEMA_VERSION:
        raise TrajectorySchemaError(
            "schema_version must be "
            f"{TRAJECTORY_SAMPLE_SCHEMA_VERSION!r}, got {value!r}"
        )


def _validate_identifier(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TrajectorySchemaError(f"{field} must be a non-empty trimmed string")


def _validate_source_dataset(value: Any) -> None:
    if not isinstance(value, str) or _SOURCE_DATASET_PATTERN.fullmatch(value) is None:
        raise TrajectorySchemaError(
            "source_dataset must contain only lowercase letters, digits, '_' or '-'"
        )


def _parse_coordinate_system(value: Any) -> CoordinateSystem:
    try:
        return CoordinateSystem(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in CoordinateSystem)
        raise TrajectorySchemaError(
            f"coordinate_system must be one of: {allowed}"
        ) from exc


def _parse_pattern_label(value: Any) -> PatternLabel:
    try:
        return PatternLabel(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in PatternLabel)
        raise TrajectorySchemaError(f"pattern_label must be one of: {allowed}") from exc


def _parse_points(value: Any) -> tuple[tuple[float, float], ...]:
    if not _is_sequence(value) or not value:
        raise TrajectorySchemaError("points must be a non-empty sequence of [x, y]")

    parsed: list[tuple[float, float]] = []
    for index, point in enumerate(value):
        if not _is_sequence(point) or len(point) != 2:
            raise TrajectorySchemaError(f"points[{index}] must contain exactly x and y")
        x = _finite_float(point[0], f"points[{index}][0]")
        y = _finite_float(point[1], f"points[{index}][1]")
        parsed.append((x, y))
    return tuple(parsed)


def _validate_points(
    points: tuple[tuple[float, float], ...],
    coordinate_system: CoordinateSystem,
) -> None:
    if not isinstance(points, tuple) or not points:
        raise TrajectorySchemaError("points must be a non-empty tuple")
    for index, point in enumerate(points):
        if not isinstance(point, tuple) or len(point) != 2:
            raise TrajectorySchemaError(f"points[{index}] must be an immutable x/y pair")
        for axis, value in zip(("x", "y"), point):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise TrajectorySchemaError(f"points[{index}].{axis} must be finite")
            if coordinate_system is CoordinateSystem.IMAGE_NORMALIZED and not 0.0 <= value <= 1.0:
                raise TrajectorySchemaError(
                    f"image_normalized points[{index}].{axis} must be within [0, 1]"
                )


def _parse_point_mask(value: Any) -> tuple[int, ...]:
    if not _is_sequence(value):
        raise TrajectorySchemaError("point_mask must be a sequence of 0 or 1")
    return tuple(value)


def _validate_point_mask(mask: tuple[int, ...], point_count: int) -> None:
    if not isinstance(mask, tuple) or len(mask) != point_count:
        raise TrajectorySchemaError("point_mask length must match points")
    if any(isinstance(item, bool) or not isinstance(item, int) or item not in (0, 1) for item in mask):
        raise TrajectorySchemaError("point_mask values must be integer 0 or 1")
    if sum(mask) < 2:
        raise TrajectorySchemaError("point_mask must contain at least two valid points")


def _parse_point_times(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not _is_sequence(value):
        raise TrajectorySchemaError("point_times_sec must be null or a numeric sequence")
    return tuple(
        _finite_float(item, f"point_times_sec[{index}]")
        for index, item in enumerate(value)
    )


def _validate_point_times(times: tuple[float, ...] | None, point_count: int) -> None:
    if times is None:
        return
    if not isinstance(times, tuple) or len(times) != point_count:
        raise TrajectorySchemaError("point_times_sec length must match points")
    if any(value < 0.0 for value in times):
        raise TrajectorySchemaError("point_times_sec values must be non-negative")
    if any(current <= previous for previous, current in zip(times, times[1:])):
        raise TrajectorySchemaError("point_times_sec must be strictly increasing")


def _validate_binary_label(value: Any) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1)
    ):
        raise TrajectorySchemaError("binary_label must be integer 0, integer 1, or null")


def _validate_label_consistency(
    pattern_label: PatternLabel,
    binary_label: int | None,
) -> None:
    expected_binary = {
        PatternLabel.DIRECT: 0,
        PatternLabel.PACING: 1,
        PatternLabel.LAPPING: 1,
        PatternLabel.RANDOM: 1,
    }.get(pattern_label)
    if expected_binary is not None and binary_label != expected_binary:
        raise TrajectorySchemaError(
            f"binary_label must be {expected_binary} when pattern_label is "
            f"{pattern_label.value!r}"
        )


def _parse_quality_flags(value: Any) -> tuple[str, ...]:
    if not _is_sequence(value):
        raise TrajectorySchemaError("quality_flags must be a sequence of strings")
    return tuple(value)


def _validate_quality_flags(flags: tuple[str, ...]) -> None:
    if not isinstance(flags, tuple):
        raise TrajectorySchemaError("quality_flags must be an immutable tuple")
    if any(
        not isinstance(flag, str) or _QUALITY_FLAG_PATTERN.fullmatch(flag) is None
        for flag in flags
    ):
        raise TrajectorySchemaError(
            "quality_flags values must be non-empty lowercase snake_case strings"
        )
    if len(set(flags)) != len(flags):
        raise TrajectorySchemaError("quality_flags must not contain duplicates")


def _validate_source_path(value: Any) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TrajectorySchemaError("source_path must be a non-empty trimmed string")
    if (
        "\\" in value
        or _WINDOWS_ABSOLUTE_PATH_PATTERN.match(value) is not None
        or PurePosixPath(value).is_absolute()
        or ".." in PurePosixPath(value).parts
    ):
        raise TrajectorySchemaError(
            "source_path must be a portable relative POSIX path without '..'"
        )


def _validate_source_sha256(value: Any) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise TrajectorySchemaError(
            "source_sha256 must be a lowercase 64-character SHA-256 digest"
        )


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TrajectorySchemaError(f"{field} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise TrajectorySchemaError(f"{field} must be a finite number")
    return parsed


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
