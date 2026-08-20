from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from numbers import Real
from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.mental_health.config import AggregationConfig, load_aggregation_config


_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_REJECTED_QUALITY_STATES = {"invalid", "low_quality", "low_quality_run", "rejected"}
_REJECTING_QUALITY_FLAGS = {
    "insufficient_pose_quality",
    "invalid_pose",
    "pose_quality_rejected",
    "rejected_by_pose_quality",
}


class MentalHealthDataError(ValueError):
    """A field-level validation error in mental-health source data."""


@dataclass(frozen=True)
class BehaviorObservation:
    person_id: str
    device_id: str | None
    frame_id: int
    observed_at: datetime | None
    timestamp_sec: float | None
    scene_region: str
    keypoint_quality: float | None
    core_coordinates: tuple[tuple[str, float, float], ...]
    usable_for_daily_aggregation: bool
    usable_for_valid_interval: bool
    data_quality_flags: tuple[str, ...]
    record_number: int

    @property
    def coordinate_map(self) -> dict[str, tuple[float, float]]:
        return {name: (x, y) for name, x, y in self.core_coordinates}


def adapt_behavior_record(
    record: Mapping[str, Any],
    *,
    config: AggregationConfig | None = None,
    record_number: int = 1,
) -> BehaviorObservation:
    aggregation_config = config or load_aggregation_config()
    if not isinstance(record, Mapping):
        raise MentalHealthDataError(f"behavior record {record_number}: expected an object")

    person_id = _required_nonempty_string(record, "person_id", "behavior", record_number)
    device_id = _optional_nonempty_string(record, "device_id", "behavior", record_number)
    frame_id = _required_nonnegative_integer(record, "frame_id", "behavior", record_number)

    flags: list[str] = []
    timestamp_sec = _optional_finite_number(record.get("timestamp_sec"))
    if record.get("timestamp_sec") is not None and timestamp_sec is None:
        flags.append("invalid_timestamp_sec")
    if timestamp_sec is not None and timestamp_sec < 0:
        flags.append("invalid_timestamp_sec")
        timestamp_sec = None

    observed_at, time_flags = _resolve_observed_at(record, timestamp_sec, aggregation_config)
    flags.extend(time_flags)

    quality_key = "core_keypoint_quality" if record.get("core_keypoint_quality") is not None else "keypoint_quality"
    keypoint_quality = _optional_finite_number(record.get(quality_key))
    quality_usable = True
    if keypoint_quality is None or not 0.0 <= keypoint_quality <= 1.0:
        flags.append("invalid_keypoint_quality")
        quality_usable = False
    elif keypoint_quality < aggregation_config.min_keypoint_quality:
        flags.append("insufficient_pose_quality")
        quality_usable = False

    quality_state = str(record.get("quality_state", "")).strip().lower()
    if quality_state in _REJECTED_QUALITY_STATES:
        flags.append("insufficient_pose_quality")
        quality_usable = False
    if record.get("quality_rejected") is True or record.get("rejected") is True:
        flags.append("pose_quality_rejected")
        quality_usable = False
    if record.get("usable") is False:
        flags.append("pose_quality_rejected")
        quality_usable = False

    source_quality_flags = _string_flags(record.get("quality_flags")) + _string_flags(
        record.get("pose_quality_flags")
    )
    flags.extend(source_quality_flags)
    if any(flag in _REJECTING_QUALITY_FLAGS for flag in source_quality_flags):
        flags.append("pose_quality_rejected")
        quality_usable = False

    core_coordinates, coordinate_flags = _core_coordinates(record.get("keypoints"), aggregation_config)
    flags.extend(coordinate_flags)
    if "missing_keypoints" in coordinate_flags or "invalid_normalized_keypoints" in coordinate_flags:
        quality_usable = False

    scene_region = record.get("scene_region")
    if not isinstance(scene_region, str) or not scene_region.strip():
        scene = "unknown"
        flags.append("unknown_scene_region")
    else:
        scene = scene_region.strip()

    timeline_usable = observed_at is not None
    valid_interval_usable = (
        timeline_usable
        and quality_usable
        and "timestamp_conflict" not in flags
        and "invalid_session_time_source" not in flags
        and "invalid_timestamp_sec" not in flags
    )
    return BehaviorObservation(
        person_id=person_id,
        device_id=device_id,
        frame_id=frame_id,
        observed_at=observed_at,
        timestamp_sec=timestamp_sec,
        scene_region=scene,
        keypoint_quality=keypoint_quality,
        core_coordinates=core_coordinates,
        usable_for_daily_aggregation=timeline_usable,
        usable_for_valid_interval=valid_interval_usable,
        data_quality_flags=tuple(_dedupe(flags)),
        record_number=record_number,
    )


def adapt_sleep_record(
    record: Mapping[str, Any],
    *,
    config: AggregationConfig | None = None,
    record_number: int = 1,
) -> dict[str, Any]:
    aggregation_config = config or load_aggregation_config()
    if not isinstance(record, Mapping):
        raise MentalHealthDataError(f"sleep record {record_number}: expected an object")

    person_id = _required_nonempty_string(record, "person_id", "sleep", record_number)
    local_date = _sleep_date(record, aggregation_config, record_number)
    sleep_onset_latency = _optional_bounded_number(
        record,
        "sleep_onset_latency",
        minimum=0.0,
        maximum=720.0,
        source="sleep",
        record_number=record_number,
    )
    night_awakenings = _optional_bounded_integer(
        record,
        "night_awakenings",
        minimum=0,
        maximum=100,
        source="sleep",
        record_number=record_number,
    )
    sleep_efficiency = _optional_bounded_number(
        record,
        "sleep_efficiency",
        minimum=0.0,
        maximum=1.0,
        source="sleep",
        record_number=record_number,
    )
    quality_score = _optional_bounded_number(
        record,
        "quality_score",
        minimum=0.0,
        maximum=1.0,
        source="sleep",
        record_number=record_number,
    )
    device_source = _optional_nonempty_string(record, "device_source", "sleep", record_number)

    raw_flags = record.get("quality_flags")
    if raw_flags is None:
        quality_flags: list[str] = []
    elif not isinstance(raw_flags, list) or any(
        not isinstance(flag, str) or not flag.strip() for flag in raw_flags
    ):
        raise _field_error("sleep", record_number, "quality_flags", "must be a list of non-empty strings")
    else:
        quality_flags = [flag.strip() for flag in raw_flags]

    for field_name, value in (
        ("sleep_onset_latency", sleep_onset_latency),
        ("night_awakenings", night_awakenings),
        ("sleep_efficiency", sleep_efficiency),
    ):
        if value is None:
            quality_flags.append(f"missing_{field_name}")

    return {
        "person_id": person_id,
        "date": local_date.isoformat(),
        "sleep_onset_latency": sleep_onset_latency,
        "night_awakenings": night_awakenings,
        "sleep_efficiency": sleep_efficiency,
        "device_source": device_source,
        "quality_score": quality_score,
        "quality_flags": _dedupe(quality_flags),
    }


def adapt_sleep_records(
    records: Iterable[Mapping[str, Any]],
    *,
    config: AggregationConfig | None = None,
) -> list[dict[str, Any]]:
    aggregation_config = config or load_aggregation_config()
    adapted = [
        adapt_sleep_record(record, config=aggregation_config, record_number=index)
        for index, record in enumerate(records, start=1)
    ]
    return sorted(adapted, key=lambda item: (item["date"], item["person_id"], item["device_source"] or ""))


def _resolve_observed_at(
    record: Mapping[str, Any],
    timestamp_sec: float | None,
    config: AggregationConfig,
) -> tuple[datetime | None, list[str]]:
    flags: list[str] = []
    has_observed_at = record.get("observed_at") is not None
    direct_time = _parse_aware_datetime(record.get("observed_at"), "observed_at", flags) if has_observed_at else None

    has_session_start = record.get("session_start_time") is not None
    session_start = (
        _parse_aware_datetime(record.get("session_start_time"), "session_start_time", flags)
        if has_session_start
        else None
    )
    derived_time = None
    if session_start is not None:
        if timestamp_sec is None:
            flags.append("invalid_timestamp_sec")
        else:
            try:
                derived_time = session_start + timedelta(seconds=timestamp_sec)
            except (OverflowError, ValueError):
                flags.append("invalid_timestamp_sec")

    if has_observed_at:
        selected = direct_time
        if direct_time is not None and has_session_start:
            if derived_time is None:
                flags.append("invalid_session_time_source")
            elif abs((direct_time - derived_time).total_seconds()) > config.timestamp_conflict_tolerance_seconds:
                flags.append("timestamp_conflict")
    else:
        selected = derived_time

    if selected is None and not any(flag in {"timezone_missing", "invalid_observed_at"} for flag in flags):
        flags.append("missing_absolute_time")
    if selected is None:
        return None, _dedupe(flags)
    try:
        localized = selected.astimezone(config.timezone_info)
    except (OverflowError, ValueError):
        flags.append("invalid_absolute_time")
        localized = None
    return localized, _dedupe(flags)


def _parse_aware_datetime(value: Any, field: str, flags: list[str]) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        flags.append(f"invalid_{field}")
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        flags.append(f"invalid_{field}")
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        flags.append("timezone_missing")
        return None
    return parsed


def _core_coordinates(
    raw_keypoints: Any,
    config: AggregationConfig,
) -> tuple[tuple[tuple[str, float, float], ...], list[str]]:
    if not isinstance(raw_keypoints, (list, tuple)) or not raw_keypoints:
        return (), ["missing_keypoints"]

    allowed = set(config.core_keypoints)
    coordinates: dict[str, tuple[float, float]] = {}
    invalid_coordinate = False
    for point in raw_keypoints:
        if not isinstance(point, Mapping):
            continue
        name = point.get("name")
        if not isinstance(name, str) or name not in allowed or name in coordinates:
            continue
        if point.get("valid") is False or point.get("is_jump_outlier") is True:
            continue
        if point.get("valid") is not True:
            score = _optional_finite_number(point.get("score"))
            if score is None or score < 0.30:
                continue
        x = _preferred_coordinate(point, "x_smooth", "x")
        y = _preferred_coordinate(point, "y_smooth", "y")
        if x is None or y is None or not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            invalid_coordinate = True
            continue
        coordinates[name] = (x, y)

    flags = ["invalid_normalized_keypoints"] if invalid_coordinate else []
    ordered = tuple(
        (name, coordinates[name][0], coordinates[name][1])
        for name in config.core_keypoints
        if name in coordinates
    )
    return ordered, flags


def _preferred_coordinate(point: Mapping[str, Any], preferred: str, fallback: str) -> float | None:
    preferred_value = _optional_finite_number(point.get(preferred))
    if preferred_value is not None:
        return preferred_value
    return _optional_finite_number(point.get(fallback))


def _sleep_date(record: Mapping[str, Any], config: AggregationConfig, record_number: int) -> date:
    raw_date = record.get("date")
    if raw_date is not None:
        if not isinstance(raw_date, str) or _DATE_PATTERN.fullmatch(raw_date) is None:
            raise _field_error("sleep", record_number, "date", "must use YYYY-MM-DD format")
        try:
            return date.fromisoformat(raw_date)
        except ValueError as exc:
            raise _field_error("sleep", record_number, "date", "must be a valid calendar date") from exc

    timestamp_field = "observed_at" if record.get("observed_at") is not None else "timestamp"
    raw_timestamp = record.get(timestamp_field)
    if raw_timestamp is None:
        raise _field_error(
            "sleep",
            record_number,
            "date",
            "requires YYYY-MM-DD date or a timezone-aware observed_at/timestamp",
        )
    flags: list[str] = []
    parsed = _parse_aware_datetime(raw_timestamp, timestamp_field, flags)
    if parsed is None:
        reason = ", ".join(flags) or "invalid timestamp"
        raise _field_error("sleep", record_number, timestamp_field, reason)
    try:
        return parsed.astimezone(config.timezone_info).date()
    except (OverflowError, ValueError) as exc:
        raise _field_error(
            "sleep",
            record_number,
            timestamp_field,
            "cannot be represented in the configured timezone",
        ) from exc


def _required_nonempty_string(
    record: Mapping[str, Any],
    field: str,
    source: str,
    record_number: int,
) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        detail = "must be a non-empty stable business person ID"
        if source == "behavior" and field == "person_id" and record.get("track_id") is not None:
            detail += "; track_id cannot substitute for person_id"
        raise _field_error(source, record_number, field, detail)
    return value.strip()


def _optional_nonempty_string(
    record: Mapping[str, Any],
    field: str,
    source: str,
    record_number: int,
) -> str | None:
    value = record.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _field_error(source, record_number, field, "must be a non-empty string when provided")
    return value.strip()


def _required_nonnegative_integer(
    record: Mapping[str, Any],
    field: str,
    source: str,
    record_number: int,
) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _field_error(source, record_number, field, "must be a non-negative integer")
    return value


def _optional_bounded_number(
    record: Mapping[str, Any],
    field: str,
    *,
    minimum: float,
    maximum: float,
    source: str,
    record_number: int,
) -> float | None:
    value = record.get(field)
    if value is None:
        return None
    number = _optional_finite_number(value)
    if number is None or not minimum <= number <= maximum:
        raise _field_error(
            source,
            record_number,
            field,
            f"must be a finite number in [{minimum}, {maximum}]",
        )
    return number


def _optional_bounded_integer(
    record: Mapping[str, Any],
    field: str,
    *,
    minimum: int,
    maximum: int,
    source: str,
    record_number: int,
) -> int | None:
    value = record.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise _field_error(
            source,
            record_number,
            field,
            f"must be an integer in [{minimum}, {maximum}]",
        )
    return value


def _optional_finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _string_flags(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(flag).strip() for flag in value if isinstance(flag, str) and flag.strip()]


def _field_error(source: str, record_number: int, field: str, detail: str) -> MentalHealthDataError:
    return MentalHealthDataError(f"{source} record {record_number}, field '{field}': {detail}")


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
