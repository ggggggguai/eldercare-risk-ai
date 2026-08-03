"""Independently validate DATA-003 inputs and RESILIENT artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path, PurePosixPath
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from elderly_monitoring.modules.mental_health.mood_social import (
    feature_schema_manifest,
)


DATASET_ID = "resilient"
SOURCE_RELATIVE = Path("数据集/心理/RESILIENT")
SENSOR_RELATIVE = Path("Sleepmat_Watch_Data/Sleepmat_Watch_Data")
OUTPUT_RELATIVE = Path("data/processed/mental_health/mood_social/v3.3.3")
MANIFEST_SHA256 = "59257b21fee44bd405e0be6f299d49ad945c6e6289002717f8769389dce32bd5"
COLLECTION_SHA256 = "77278afd8d443cf53608ada9cf576b46b1799951ab79149c39af4e5ef50f53c2"
SCHEMA_SHA256 = "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
TIMEZONE = "Europe/London"
WINDOW_DAYS = 14
MIN_STEP_HOURS = 20
MAX_EPISODE_GAP = 60.0
MIN_SLEEP_MINUTES = 60.0
MAX_SLEEP_MINUTES = 960.0
EXPECTED_ROWS = 73
EXPECTED_SOURCE_FILES = 589
EXPECTED_SUMMARY_ROWS = 292
EXPECTED_POSITIVES = 10
EXPECTED_GDS_POSITIVES = 23
EXPECTED_GAD_POSITIVES = 6
EXPECTED_CATEGORIES = {0: 37, 1: 26, 2: 5, 3: 4, 4: 1}
EXPECTED_SENSOR_ROWS = {
    "ScanWatch_HR.csv": 991_325,
    "ScanWatch_Steps.csv": 241_505,
    "Sleep_physio.csv": 1_115_748,
    "Sleep_state.csv": 1_132_263,
}
EXPECTED_ARTIFACTS = {
    "mappings/resilient_v3_3_3_mapping.json",
    "resilient/adapter_metadata.json",
    "resilient/canonical_resilient.parquet",
    "resilient/data_quality_report.json",
}
SENSOR_COLUMNS = {
    "ScanWatch_HR.csv": ("HR Timestamp", "Heart Rate"),
    "ScanWatch_Steps.csv": ("Steps Timestamp", "Steps"),
    "Sleep_physio.csv": (
        "Heart Rate",
        "Timestamp",
        "Respiration Rate",
        "Snoring",
        "SDNN_1",
    ),
    "Sleep_state.csv": ("Start time", "End time", "Sleep state"),
}
SLEEP_STATES = {"REM", "deep", "light", "wakeup"}
OFFSET_PATTERN = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")
EPSILON = 1e-12

IDENTITY_COLUMNS = (
    "dataset_id",
    "global_participant_id",
    "participant_id",
    "timescale_semantics",
    "source_timezone",
    "window_start_date",
    "window_end_date",
    "window_calendar_days",
)
TARGET_COLUMNS = (
    "phq_assessment_date",
    "phq9_score",
    "phq9_category",
    "binary_target",
    "gds_assessment_date",
    "gds15_score",
    "gds15_binary_target",
    "gad_assessment_date",
    "gad7_score",
    "gad7_binary_target",
)
X_COLUMNS = ("x_source_name", "x_source_value", "x_source_mask")
SOURCE_TYPES: dict[str, pa.DataType] = {
    "profile_sex": pa.large_string(),
    "profile_age_group": pa.large_string(),
    "profile_essential_hypertension": pa.int8(),
    "profile_osteoarthritis": pa.int8(),
    "scanwatch_steps_daily_mean": pa.float64(),
    "scanwatch_steps_valid_days": pa.int16(),
    "scanwatch_relative_amplitude": pa.float64(),
    "scanwatch_interdaily_stability": pa.float64(),
    "scanwatch_intradaily_variability": pa.float64(),
    "scanwatch_activity_variability": pa.float64(),
    "scanwatch_hr_mean_bpm": pa.float64(),
    "scanwatch_hr_min_bpm": pa.float64(),
    "scanwatch_hr_sd_bpm": pa.float64(),
    "scanwatch_hr_valid_days": pa.int16(),
    "sleep_valid_nights": pa.int16(),
    "sleep_duration_minutes_mean": pa.float64(),
    "time_in_bed_minutes_mean": pa.float64(),
    "sleep_efficiency": pa.float64(),
    "sleep_onset_sin": pa.float64(),
    "sleep_onset_cos": pa.float64(),
    "wake_time_sin": pa.float64(),
    "wake_time_cos": pa.float64(),
    "sleep_midpoint_sin": pa.float64(),
    "sleep_midpoint_cos": pa.float64(),
    "sleep_fragmentation": pa.float64(),
    "sleep_regularity": pa.float64(),
    "sleep_awakening_count_mean": pa.float64(),
    "sleep_physio_valid_nights": pa.int16(),
    "sleep_hr_mean_bpm": pa.float64(),
    "sleep_hr_min_bpm": pa.float64(),
    "sleep_hr_sd_bpm": pa.float64(),
    "sleep_hrv_sdnn_ms": pa.float64(),
    "sleep_respiration_rate_mean_bpm": pa.float64(),
    "sleep_respiration_rate_sd_bpm": pa.float64(),
    "sleep_snoring_minutes_mean": pa.float64(),
}
SOURCE_FIELDS = tuple(SOURCE_TYPES)
DIRECT_TARGETS = {
    "activity.relative_amplitude": "scanwatch_relative_amplitude",
    "activity.interdaily_stability": "scanwatch_interdaily_stability",
    "activity.intradaily_variability": "scanwatch_intradaily_variability",
    "activity.activity_variability": "scanwatch_activity_variability",
    "sleep.sleep_duration_norm": "sleep_duration_minutes_mean",
    "sleep.time_in_bed_norm": "time_in_bed_minutes_mean",
    "sleep.sleep_efficiency": "sleep_efficiency",
    "sleep.sleep_onset_sin": "sleep_onset_sin",
    "sleep.sleep_onset_cos": "sleep_onset_cos",
    "sleep.wake_time_sin": "wake_time_sin",
    "sleep.wake_time_cos": "wake_time_cos",
    "sleep.sleep_midpoint_sin": "sleep_midpoint_sin",
    "sleep.sleep_midpoint_cos": "sleep_midpoint_cos",
    "sleep.sleep_fragmentation": "sleep_fragmentation",
    "sleep.sleep_regularity": "sleep_regularity",
    "physiology.heart_rate_mean_bpm": "sleep_hr_mean_bpm",
    "physiology.heart_rate_min_bpm": "sleep_hr_min_bpm",
    "physiology.heart_rate_sd_bpm": "sleep_hr_sd_bpm",
    "physiology.hrv_sdnn_ms": "sleep_hrv_sdnn_ms",
    "physiology.respiration_rate_mean_bpm": "sleep_respiration_rate_mean_bpm",
    "physiology.respiration_rate_sd_bpm": "sleep_respiration_rate_sd_bpm",
    "physiology.snoring_minutes_norm": "sleep_snoring_minutes_mean",
    "social_context.sex": "profile_sex",
    "social_context.age_group": "profile_age_group",
}
EXPECTED_FORMULAS = {
    "activity.relative_amplitude": (
        "publisher hourly step increments; M10/L5 relative amplitude"
    ),
    "activity.interdaily_stability": (
        "coverage-weighted hour-of-day variance / total variance"
    ),
    "activity.intradaily_variability": (
        "mean adjacent-hour squared difference / total variance"
    ),
    "activity.activity_variability": (
        "population coefficient of variation of valid daily step totals"
    ),
    "sleep.sleep_duration_norm": "sleep_duration_minutes_mean / 1440",
    "sleep.time_in_bed_norm": "time_in_bed_minutes_mean / 1440",
    "sleep.sleep_efficiency": ("sum(asleep minutes) / sum(observed in-bed minutes)"),
    "sleep.sleep_onset_sin": "circular mean component of local sleep onset",
    "sleep.sleep_onset_cos": "circular mean component of local sleep onset",
    "sleep.wake_time_sin": "circular mean component of local wake time",
    "sleep.wake_time_cos": "circular mean component of local wake time",
    "sleep.sleep_midpoint_sin": (
        "circular mean component of forward local sleep midpoint"
    ),
    "sleep.sleep_midpoint_cos": (
        "circular mean component of forward local sleep midpoint"
    ),
    "sleep.sleep_fragmentation": (
        "mean(awake/in_bed + awakenings/(asleep_minutes/60))"
    ),
    "sleep.sleep_regularity": (
        "mean circular resultant length for onset, wake, midpoint"
    ),
    "physiology.heart_rate_mean_bpm": (
        "mean of valid nightly means after timestamp deduplication"
    ),
    "physiology.heart_rate_min_bpm": (
        "mean of valid nightly minima after timestamp deduplication"
    ),
    "physiology.heart_rate_sd_bpm": (
        "mean of valid nightly population standard deviations"
    ),
    "physiology.hrv_sdnn_ms": "mean of valid nightly SDNN_1 means",
    "physiology.respiration_rate_mean_bpm": ("mean of valid nightly respiration means"),
    "physiology.respiration_rate_sd_bpm": (
        "mean of valid nightly population standard deviations"
    ),
    "physiology.snoring_minutes_norm": ("mean nightly sum(source seconds) / 60 / 1440"),
    "social_context.sex": "Female->female; Male->male",
    "social_context.age_group": (
        "map only source intervals wholly contained in a target interval"
    ),
}
LEAKAGE_EXCLUSIONS = [
    "phq_date",
    "phq1-phq9",
    "phq_extraq",
    "phq_total",
    "gad_date",
    "gad1-gad7",
    "gad_7_additional_question",
    "gad_total",
    "gds_date",
    "gds1-gds15",
    "gds_total",
    "all ACE fields",
]


class ValidationError(RuntimeError):
    """Raised when independently observed DATA-003 facts do not match."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _collection_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(
        rows, key=lambda item: str(item["relative_path"]).encode("utf-8")
    ):
        digest.update(
            f"{row['relative_path']}\0{row['bytes']}\0{row['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _find_workspace_root(start: Path) -> Path:
    resolved = start.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / SOURCE_RELATIVE).is_dir() and (
            candidate / "algorithm" / "eldercare-risk-ai-main"
        ).is_dir():
            return candidate
    raise ValidationError("workspace root was not found")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"JSON artifact is not readable: {path.name}") from exc


def _check(condition: bool, name: str, checks: list[str]) -> None:
    if not condition:
        raise ValidationError(f"independent DATA-003 check failed: {name}")
    checks.append(name)


def _target_specs() -> list[dict[str, Any]]:
    schema = feature_schema_manifest()
    return [
        {"group": group, **dict(spec)}
        for group in schema["group_order"]
        for spec in schema["groups"][group]
    ]


def _column_order() -> tuple[str, ...]:
    features = tuple(f"{spec['group']}.{spec['name']}" for spec in _target_specs())
    masks = tuple(f"feature_mask.{name}" for name in features)
    return (
        *IDENTITY_COLUMNS,
        *TARGET_COLUMNS,
        *X_COLUMNS,
        *(f"source__{name}" for name in SOURCE_FIELDS),
        *features,
        *masks,
    )


def _arrow_schema() -> pa.Schema:
    types: dict[str, pa.DataType] = {
        "dataset_id": pa.large_string(),
        "global_participant_id": pa.large_string(),
        "participant_id": pa.large_string(),
        "timescale_semantics": pa.large_string(),
        "source_timezone": pa.large_string(),
        "window_start_date": pa.date32(),
        "window_end_date": pa.date32(),
        "window_calendar_days": pa.int8(),
        "phq_assessment_date": pa.date32(),
        "phq9_score": pa.int8(),
        "phq9_category": pa.int8(),
        "binary_target": pa.int8(),
        "gds_assessment_date": pa.date32(),
        "gds15_score": pa.int8(),
        "gds15_binary_target": pa.int8(),
        "gad_assessment_date": pa.date32(),
        "gad7_score": pa.int8(),
        "gad7_binary_target": pa.int8(),
        "x_source_name": pa.large_string(),
        "x_source_value": pa.float64(),
        "x_source_mask": pa.int8(),
    }
    types.update({f"source__{name}": value for name, value in SOURCE_TYPES.items()})
    feature_types = {
        "float": pa.float64(),
        "integer": pa.int64(),
        "category": pa.large_string(),
    }
    for spec in _target_specs():
        name = f"{spec['group']}.{spec['name']}"
        types[name] = feature_types[str(spec["value_type"])]
        types[f"feature_mask.{name}"] = pa.int8()
    return pa.schema([pa.field(name, types[name]) for name in _column_order()])


@dataclass(frozen=True)
class FrozenInputs:
    workspace_root: Path
    source_root: Path
    manifests_root: Path
    sensor_directories: Mapping[str, Path]


@dataclass(frozen=True)
class ObservedEpisode:
    start: pd.Timestamp
    end: pd.Timestamp
    wake_date: date
    in_bed_minutes: float
    asleep_minutes: float
    awake_minutes: float
    onset_minute: float
    wake_minute: float
    midpoint_minute: float
    awakening_count: int


@dataclass(frozen=True)
class ObservedSource:
    canonical: pd.DataFrame
    quality_rows: tuple[Mapping[str, Any], ...]
    sensor_row_totals: Mapping[str, int]
    source_participant_ids: frozenset[str]
    demographics_columns: tuple[str, ...]


def _read_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValidationError("manifest row is not an object")
                    rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError("DATA-001 manifest is not readable") from exc
    return rows


def _sensor_directories(source_root: Path) -> dict[str, Path]:
    sensor_root = source_root / SENSOR_RELATIVE
    if not sensor_root.is_dir():
        raise ValidationError("real sensor directory is missing")
    directories = sorted(
        (path for path in sensor_root.iterdir() if path.is_dir()),
        key=lambda path: path.name.encode("utf-8"),
    )
    if len(directories) != EXPECTED_ROWS:
        raise ValidationError("real sensor directory count differs from frozen input")
    return {path.name: path for path in directories}


def _public_relative_path(
    workspace_root: Path,
    source_root: Path,
    source_path: Path,
    aliases: Mapping[str, str],
) -> str:
    parts = list(source_path.relative_to(source_root).parts)
    for position, part in enumerate(parts):
        prefix = "._" if part.startswith("._") else ""
        candidate = part[2:] if prefix else part
        if candidate in aliases:
            parts[position] = f"{prefix}{aliases[candidate]}"
    logical = workspace_root / SOURCE_RELATIVE / Path(*parts)
    return logical.relative_to(workspace_root).as_posix()


def _source_file_role(path: Path, source_root: Path) -> tuple[str, str]:
    parts = path.relative_to(source_root).parts
    if path.name == ".DS_Store":
        return "os_metadata", "ds_store"
    if "__MACOSX" in parts:
        return "os_metadata", "appledouble"
    if path.suffix.lower() == ".md":
        return "source_documentation", "markdown"
    if path.suffix.lower() == ".csv":
        role = "source_metadata" if "Resilient_metadata" in parts else "source_data"
        return role, "csv"
    raise ValidationError("RESILIENT source contains an unsupported file type")


def _expected_binding() -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "source_relative": SOURCE_RELATIVE.as_posix(),
        "source_collection_sha256": COLLECTION_SHA256,
        "file_manifest_sha256": MANIFEST_SHA256,
        "feature_schema_sha256": SCHEMA_SHA256,
        "source_file_count": EXPECTED_SOURCE_FILES,
        "all_bindings_validated": True,
    }


def _validate_frozen_inputs(
    workspace_root: Path | None,
    manifests_root: Path | None,
    checks: list[str],
) -> FrozenInputs:
    root = _find_workspace_root(workspace_root or Path.cwd())
    algorithm_root = root / "algorithm" / "eldercare-risk-ai-main"
    audit_root = (
        manifests_root or algorithm_root / OUTPUT_RELATIVE / "manifests"
    ).resolve()
    manifest_path = audit_root / "dataset_file_manifest.jsonl"
    schema_path = audit_root / "feature_schema_manifest.json"
    _check(
        _sha256_file(manifest_path) == MANIFEST_SHA256,
        "data001_manifest_sha256",
        checks,
    )
    schema = feature_schema_manifest()
    _check(
        hashlib.sha256(_canonical_json_bytes(schema)).hexdigest() == SCHEMA_SHA256,
        "live_mh003_schema_sha256",
        checks,
    )
    _check(
        _sha256_file(schema_path) == SCHEMA_SHA256,
        "data001_schema_snapshot_sha256",
        checks,
    )
    _check(_load_json(schema_path) == schema, "schema_snapshot_matches_live", checks)

    manifest_rows = _read_manifest_rows(manifest_path)
    selected = [row for row in manifest_rows if row.get("dataset_id") == DATASET_ID]
    expected_keys = {
        "bytes",
        "dataset_id",
        "file_role",
        "format",
        "relative_path",
        "sha256",
    }
    _check(len(selected) == EXPECTED_SOURCE_FILES, "source_manifest_file_count", checks)
    _check(
        all(set(row) == expected_keys for row in selected),
        "source_manifest_row_contract",
        checks,
    )

    source_root = root / SOURCE_RELATIVE
    directories = _sensor_directories(source_root)
    aliases = {
        source_id: f"participant_{position:04d}"
        for position, source_id in enumerate(
            sorted(directories, key=lambda value: value.encode("utf-8")), 1
        )
    }
    actual_files = sorted(
        (path for path in source_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix().encode("utf-8"),
    )
    _check(len(actual_files) == EXPECTED_SOURCE_FILES, "live_source_file_count", checks)
    public_rows: list[dict[str, Any]] = []
    collection_rows: list[dict[str, Any]] = []
    for path in actual_files:
        _check(not path.is_symlink(), "source_file_not_symlink", checks)
        before = path.stat()
        digest = _sha256_file(path)
        after = path.stat()
        _check(
            before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns,
            "source_stable_while_hashed",
            checks,
        )
        role, file_format = _source_file_role(path, source_root)
        public_rows.append(
            {
                "bytes": int(after.st_size),
                "dataset_id": DATASET_ID,
                "file_role": role,
                "format": file_format,
                "relative_path": _public_relative_path(
                    root, source_root, path, aliases
                ),
                "sha256": digest,
            }
        )
        collection_rows.append(
            {
                "bytes": int(after.st_size),
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": digest,
            }
        )

    def sort_key(row: Mapping[str, Any]) -> bytes:
        return str(row["relative_path"]).encode("utf-8")

    _check(
        sorted(public_rows, key=sort_key) == sorted(selected, key=sort_key),
        "source_files_match_data001_manifest",
        checks,
    )
    _check(
        _collection_sha256(collection_rows) == COLLECTION_SHA256,
        "resilient_collection_sha256",
        checks,
    )
    public_parts = {
        part
        for row in selected
        for part in PurePosixPath(str(row["relative_path"])).parts
    }
    _check(
        not set(directories).intersection(public_parts),
        "source_participant_ids_absent_from_manifest",
        checks,
    )
    _check(
        any(path.name == ".DS_Store" for path in actual_files)
        and any(
            "__MACOSX" in path.relative_to(source_root).parts for path in actual_files
        ),
        "os_metadata_included_in_collection",
        checks,
    )
    return FrozenInputs(root, source_root, audit_root, directories)


def _parse_utc(series: pd.Series, field_name: str) -> pd.Series:
    raw = series.astype("string")
    present = raw.notna() & raw.str.strip().ne("")
    if (
        not bool(present.all())
        or not raw[present].str.contains(OFFSET_PATTERN, regex=True).all()
    ):
        raise ValidationError(f"{field_name} lacks an explicit UTC offset")
    try:
        return pd.to_datetime(raw, errors="raise", utc=True)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} contains invalid timestamps") from exc


def _numeric(
    series: pd.Series,
    field_name: str,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise").astype("Float64")
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} contains non-numeric data") from exc
    values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    if np.isinf(values).any():
        raise ValidationError(f"{field_name} contains infinity")
    finite = values[~np.isnan(values)]
    if minimum is not None:
        invalid = finite <= minimum if strict_minimum else finite < minimum
        if invalid.any():
            raise ValidationError(f"{field_name} contains out-of-range data")
    return numeric


def _read_demographics(source_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        source_root / "Demographics.csv",
        dtype={"user_id": "string"},
        keep_default_na=True,
    )
    required = {
        "user_id",
        "Sex",
        "Age group",
        "Essential hypertension",
        "Osteoarthritis",
        "phq_date",
        "phq_total",
        "gds_date",
        "gds_total",
        "gad_date",
        "gad_total",
    }
    if len(frame) != EXPECTED_ROWS or len(frame.columns) != 231:
        raise ValidationError("demographics shape differs from frozen input")
    if not required.issubset(frame.columns):
        raise ValidationError("demographics required columns are missing")
    frame["user_id"] = frame["user_id"].str.strip()
    if (
        frame["user_id"].isna().any()
        or frame["user_id"].eq("").any()
        or not frame["user_id"].is_unique
    ):
        raise ValidationError("demographics join key is invalid")
    return frame


def _read_summary(source_root: Path) -> pd.DataFrame:
    expected = (
        "participant",
        "file",
        "num_days",
        "avg_records_per_day",
        "earliest_date",
        "latest_date",
        "total_records",
    )
    frame = pd.read_csv(
        source_root / "summary_stats_per_participant.csv",
        dtype={"participant": "string", "file": "string"},
    )
    if tuple(frame.columns) != expected or len(frame) != EXPECTED_SUMMARY_ROWS:
        raise ValidationError("summary table contract differs from frozen input")
    frame["participant"] = frame["participant"].str.strip()
    frame["file"] = frame["file"].str.strip()
    if frame[["participant", "file"]].isna().any(axis=None):
        raise ValidationError("summary join key is missing")
    if frame.duplicated(["participant", "file"]).any():
        raise ValidationError("summary join key is duplicated")
    if set(frame["file"]) != set(SENSOR_COLUMNS):
        raise ValidationError("summary sensor file set differs from contract")
    return frame


def _read_sensor_frame(directory: Path, file_name: str) -> pd.DataFrame:
    frame = pd.read_csv(directory / file_name)
    if tuple(frame.columns) != tuple(SENSOR_COLUMNS[file_name]):
        raise ValidationError(f"{file_name} columns differ from contract")
    return frame


def _reconcile_summary(
    frame: pd.DataFrame, file_name: str, summary_row: pd.Series
) -> None:
    if int(summary_row["total_records"]) != len(frame):
        raise ValidationError("summary source record count does not match")
    if frame.empty:
        if int(summary_row["num_days"]) != 0:
            raise ValidationError("summary empty-file day count does not match")
        return
    if file_name == "Sleep_state.csv":
        timestamps = _parse_utc(frame["Start time"], "Start time")
        latest = _parse_utc(frame["End time"], "End time")
    else:
        field = {
            "ScanWatch_HR.csv": "HR Timestamp",
            "ScanWatch_Steps.csv": "Steps Timestamp",
            "Sleep_physio.csv": "Timestamp",
        }[file_name]
        timestamps = _parse_utc(frame[field], field)
        latest = timestamps
    day_count = int(timestamps.dt.date.nunique())
    if int(summary_row["num_days"]) != day_count:
        raise ValidationError("summary UTC day count does not match")
    earliest_date = pd.to_datetime(summary_row["earliest_date"], errors="raise").date()
    latest_date = pd.to_datetime(summary_row["latest_date"], errors="raise").date()
    if timestamps.min().date() != earliest_date or latest.max().date() != latest_date:
        raise ValidationError("summary UTC date bounds do not match")
    if not math.isclose(
        float(summary_row["avg_records_per_day"]),
        round(len(frame) / day_count, 2),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValidationError("summary average records per day does not match")


def _minute_of_local_day(timestamp: pd.Timestamp) -> float:
    local = timestamp.tz_convert(TIMEZONE)
    return (
        float(local.hour * 60 + local.minute)
        + local.second / 60.0
        + local.microsecond / 60_000_000.0
    )


def _circular_components(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    angles = [2.0 * math.pi * value / 1440.0 for value in values]
    sine = fmean(math.sin(angle) for angle in angles)
    cosine = fmean(math.cos(angle) for angle in angles)
    length = math.hypot(sine, cosine)
    if length <= EPSILON:
        return None, None
    return sine / length, cosine / length


def _circular_regularity(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    angles = [2.0 * math.pi * value / 1440.0 for value in values]
    return min(
        max(
            math.hypot(
                fmean(math.sin(angle) for angle in angles),
                fmean(math.cos(angle) for angle in angles),
            ),
            0.0,
        ),
        1.0,
    )


def _coefficient_of_variation(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean_value = fmean(values)
    if abs(mean_value) <= EPSILON:
        return 0.0 if all(abs(value) <= EPSILON for value in values) else None
    return pstdev(values) / abs(mean_value)


def _relative_amplitude(hour_profile: Mapping[int, float]) -> float | None:
    def circular_means(width: int) -> list[float]:
        output: list[float] = []
        for start in range(24):
            hours = [(start + offset) % 24 for offset in range(width)]
            if all(hour in hour_profile for hour in hours):
                output.append(fmean(hour_profile[hour] for hour in hours))
        return output

    most_active_windows = circular_means(10)
    least_active_windows = circular_means(5)
    if not most_active_windows or not least_active_windows:
        return None
    most_active = max(most_active_windows)
    least_active = min(least_active_windows)
    denominator = most_active + least_active
    if denominator <= EPSILON:
        return 0.0
    return min(max((most_active - least_active) / denominator, 0.0), 1.0)


def _interdaily_stability(cells: pd.DataFrame) -> float | None:
    if cells.empty or cells["date"].nunique() < 2 or cells["hour"].nunique() < 2:
        return None
    values = cells["steps"].to_numpy(dtype="float64")
    overall = float(np.mean(values))
    total = float(np.sum(np.square(values - overall)))
    if total <= EPSILON:
        return 1.0
    profile = cells.groupby("hour", sort=True)["steps"].mean().to_dict()
    between = math.fsum(
        (float(profile[int(row.hour)]) - overall) ** 2
        for row in cells.itertuples(index=False)
    )
    return min(max(between / total, 0.0), 1.0)


def _intradaily_variability(cells: pd.DataFrame) -> float | None:
    if len(cells) < 2:
        return None
    values = cells["steps"].to_numpy(dtype="float64")
    variance = float(np.mean(np.square(values - float(np.mean(values)))))
    differences: list[float] = []
    for _, daily in cells.groupby("date", sort=False):
        by_hour = daily.set_index("hour")["steps"]
        for hour in sorted(int(value) for value in by_hour.index):
            if hour + 1 in by_hour.index:
                differences.append(
                    (float(by_hour.loc[hour + 1]) - float(by_hour.loc[hour])) ** 2
                )
    if not differences:
        return None
    if variance <= EPSILON:
        return 0.0
    return fmean(differences) / variance


def _aggregate_steps(
    frame: pd.DataFrame, start: date, end: date
) -> tuple[dict[str, Any], dict[str, Any]]:
    if frame.empty:
        raise ValidationError("ScanWatch steps file is empty")
    timestamps = _parse_utc(frame["Steps Timestamp"], "Steps Timestamp")
    if timestamps.duplicated().any():
        raise ValidationError("ScanWatch step timestamps are duplicated")
    steps = _numeric(frame["Steps"], "Steps", minimum=0.0)
    if steps.isna().any():
        raise ValidationError("ScanWatch steps contain missing values")
    local = timestamps.dt.tz_convert(TIMEZONE)
    observed = pd.DataFrame(
        {
            "date": local.dt.date,
            "hour": local.dt.hour.astype("int16"),
            "steps": steps.astype("float64"),
        }
    )
    window = observed[observed["date"].map(lambda value: start <= value <= end)].copy()
    daily_totals: list[float] = []
    valid_cells: list[pd.DataFrame] = []
    for day, daily in window.groupby("date", sort=True):
        hourly = daily.groupby("hour", sort=True)["steps"].sum()
        if len(hourly) < MIN_STEP_HOURS:
            continue
        daily_totals.append(float(hourly.sum()))
        valid_cells.append(
            pd.DataFrame(
                {
                    "date": [day] * len(hourly),
                    "hour": hourly.index.astype("int16"),
                    "steps": hourly.to_numpy(dtype="float64"),
                }
            )
        )
    cells = (
        pd.concat(valid_cells, ignore_index=True)
        if valid_cells
        else pd.DataFrame(columns=["date", "hour", "steps"])
    )
    profile = (
        cells.groupby("hour", sort=True)["steps"].mean().to_dict()
        if not cells.empty
        else {}
    )
    values = {
        "scanwatch_steps_daily_mean": (fmean(daily_totals) if daily_totals else None),
        "scanwatch_steps_valid_days": len(valid_cells),
        "scanwatch_relative_amplitude": _relative_amplitude(profile),
        "scanwatch_interdaily_stability": _interdaily_stability(cells),
        "scanwatch_intradaily_variability": _intradaily_variability(cells),
        "scanwatch_activity_variability": _coefficient_of_variation(daily_totals),
    }
    quality = {
        "source_rows": len(frame),
        "window_rows": len(window),
        "window_observed_days": int(window["date"].nunique()),
        "valid_days": len(valid_cells),
        "minimum_distinct_hours_per_valid_day": MIN_STEP_HOURS,
    }
    return values, quality


def _aggregate_scanwatch_hr(
    frame: pd.DataFrame, start: date, end: date
) -> tuple[dict[str, Any], dict[str, Any]]:
    timestamps = _parse_utc(frame["HR Timestamp"], "HR Timestamp")
    if timestamps.duplicated().any():
        raise ValidationError("ScanWatch HR timestamps are duplicated")
    heart_rate = _numeric(
        frame["Heart Rate"], "ScanWatch Heart Rate", minimum=0.0, strict_minimum=True
    )
    if heart_rate.isna().any():
        raise ValidationError("ScanWatch HR contains missing values")
    dates = timestamps.dt.tz_convert(TIMEZONE).dt.date
    observed = pd.DataFrame({"date": dates, "heart_rate": heart_rate})
    window = observed[observed["date"].map(lambda value: start <= value <= end)]
    grouped = window.groupby("date", sort=True)["heart_rate"]
    daily_means = [float(value) for value in grouped.mean().tolist()]
    daily_mins = [float(value) for value in grouped.min().tolist()]
    daily_sds = [
        float(values.to_numpy(dtype="float64").std(ddof=0)) for _, values in grouped
    ]
    return (
        {
            "scanwatch_hr_mean_bpm": fmean(daily_means) if daily_means else None,
            "scanwatch_hr_min_bpm": fmean(daily_mins) if daily_mins else None,
            "scanwatch_hr_sd_bpm": fmean(daily_sds) if daily_sds else None,
            "scanwatch_hr_valid_days": len(daily_means),
        },
        {
            "source_rows": len(frame),
            "window_rows": len(window),
            "valid_days": len(daily_means),
        },
    )


def _episode_from_group(rows: pd.DataFrame) -> ObservedEpisode | None:
    clipped: list[dict[str, Any]] = []
    covered_until: pd.Timestamp | None = None
    for row in rows.itertuples(index=False):
        effective_start = (
            row.start
            if covered_until is None or row.start >= covered_until
            else covered_until
        )
        if row.end > effective_start:
            clipped.append(
                {"start": effective_start, "end": row.end, "state": row.state}
            )
        covered_until = (
            row.end if covered_until is None else max(covered_until, row.end)
        )
    if not clipped:
        return None
    effective = pd.DataFrame(clipped)
    first = effective["start"].min()
    last = effective["end"].max()
    span_minutes = (last - first).total_seconds() / 60.0
    durations = (effective["end"] - effective["start"]).dt.total_seconds() / 60.0
    sleeping = effective["state"].ne("wakeup")
    in_bed = float(durations.sum())
    asleep = float(durations[sleeping].sum())
    awake = float(durations[~sleeping].sum())
    if (
        in_bed < MIN_SLEEP_MINUTES
        or in_bed > MAX_SLEEP_MINUTES
        or span_minutes > MAX_SLEEP_MINUTES
        or asleep <= 0.0
    ):
        return None
    onset = effective.loc[sleeping, "start"].min()
    onset_minute = _minute_of_local_day(onset)
    wake_minute = _minute_of_local_day(last)
    midpoint_minute = (
        onset_minute + ((wake_minute - onset_minute) % 1440.0) / 2.0
    ) % 1440.0
    states = effective["state"].tolist()
    awakenings = sum(
        current == "wakeup" and previous != "wakeup"
        for previous, current in zip(states, states[1:], strict=False)
    )
    return ObservedEpisode(
        start=first,
        end=last,
        wake_date=last.tz_convert(TIMEZONE).date(),
        in_bed_minutes=in_bed,
        asleep_minutes=asleep,
        awake_minutes=awake,
        onset_minute=onset_minute,
        wake_minute=wake_minute,
        midpoint_minute=midpoint_minute,
        awakening_count=awakenings,
    )


def _main_sleep_episodes(
    frame: pd.DataFrame,
) -> tuple[list[ObservedEpisode], dict[str, Any]]:
    starts = _parse_utc(frame["Start time"], "Start time")
    ends = _parse_utc(frame["End time"], "End time")
    states = frame["Sleep state"].astype("string").str.strip()
    if states.isna().any() or not set(states).issubset(SLEEP_STATES):
        raise ValidationError("Sleep_state contains an unsupported state")
    if not bool((ends > starts).all()):
        raise ValidationError("Sleep_state interval duration is invalid")
    records = pd.DataFrame({"start": starts, "end": ends, "state": states}).sort_values(
        ["start", "end", "state"], kind="stable"
    )
    if records.duplicated().any():
        raise ValidationError("Sleep_state contains duplicate intervals")
    groups: list[pd.DataFrame] = []
    group_start = 0
    running_end: pd.Timestamp | None = None
    for position, row in enumerate(records.itertuples(index=False)):
        gap = (
            None
            if running_end is None
            else (row.start - running_end).total_seconds() / 60.0
        )
        if position > group_start and gap is not None and gap > MAX_EPISODE_GAP:
            groups.append(records.iloc[group_start:position])
            group_start = position
            running_end = None
        running_end = row.end if running_end is None else max(running_end, row.end)
    if len(records):
        groups.append(records.iloc[group_start:])
    candidates = [
        episode for group in groups if (episode := _episode_from_group(group))
    ]
    main_by_wake_date: dict[date, ObservedEpisode] = {}
    for episode in candidates:
        current = main_by_wake_date.get(episode.wake_date)
        if current is None or (
            episode.asleep_minutes,
            episode.in_bed_minutes,
            episode.end,
        ) > (current.asleep_minutes, current.in_bed_minutes, current.end):
            main_by_wake_date[episode.wake_date] = episode
    main = [main_by_wake_date[key] for key in sorted(main_by_wake_date)]
    return main, {
        "source_rows": len(frame),
        "episode_count_before_main_selection": len(candidates),
        "main_sleep_night_count": len(main),
        "discarded_invalid_episode_count": len(groups) - len(candidates),
    }


def _aggregate_sleep_state(
    frame: pd.DataFrame, start: date, end: date
) -> tuple[dict[str, Any], dict[str, Any], list[ObservedEpisode]]:
    episodes, quality = _main_sleep_episodes(frame)
    window = [episode for episode in episodes if start <= episode.wake_date <= end]
    durations = [episode.asleep_minutes for episode in window]
    in_bed = [episode.in_bed_minutes for episode in window]
    onset = [episode.onset_minute for episode in window]
    wake = [episode.wake_minute for episode in window]
    midpoint = [episode.midpoint_minute for episode in window]
    onset_sin, onset_cos = _circular_components(onset)
    wake_sin, wake_cos = _circular_components(wake)
    midpoint_sin, midpoint_cos = _circular_components(midpoint)
    regularity = [
        score
        for sequence in (onset, wake, midpoint)
        if (score := _circular_regularity(sequence)) is not None
    ]
    fragmentation = [
        episode.awake_minutes / episode.in_bed_minutes
        + episode.awakening_count / (episode.asleep_minutes / 60.0)
        for episode in window
    ]
    return (
        {
            "sleep_valid_nights": len(window),
            "sleep_duration_minutes_mean": fmean(durations) if durations else None,
            "time_in_bed_minutes_mean": fmean(in_bed) if in_bed else None,
            "sleep_efficiency": (
                math.fsum(durations) / math.fsum(in_bed) if in_bed else None
            ),
            "sleep_onset_sin": onset_sin,
            "sleep_onset_cos": onset_cos,
            "wake_time_sin": wake_sin,
            "wake_time_cos": wake_cos,
            "sleep_midpoint_sin": midpoint_sin,
            "sleep_midpoint_cos": midpoint_cos,
            "sleep_fragmentation": fmean(fragmentation) if fragmentation else None,
            "sleep_regularity": fmean(regularity) if regularity else None,
            "sleep_awakening_count_mean": (
                fmean(episode.awakening_count for episode in window) if window else None
            ),
        },
        {**quality, "window_main_sleep_night_count": len(window)},
        window,
    )


def _episode_indices(
    timestamps: pd.Series, episodes: Sequence[ObservedEpisode]
) -> np.ndarray:
    assigned = np.full(len(timestamps), -1, dtype=np.int32)
    if not episodes or timestamps.empty:
        return assigned
    ordered = sorted(enumerate(episodes), key=lambda item: item[1].start)
    starts = np.asarray(
        [
            episode.start.to_datetime64().astype("datetime64[ns]").astype(np.int64)
            for _, episode in ordered
        ],
        dtype=np.int64,
    )
    ends = np.asarray(
        [
            episode.end.to_datetime64().astype("datetime64[ns]").astype(np.int64)
            for _, episode in ordered
        ],
        dtype=np.int64,
    )
    values = timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    positions = np.searchsorted(starts, values, side="right") - 1
    candidate = positions >= 0
    safe = np.maximum(positions, 0)
    candidate &= values <= ends[safe]
    original = np.asarray([index for index, _ in ordered], dtype=np.int32)
    assigned[candidate] = original[safe[candidate]]
    return assigned


def _nightly_values(grouped: Any, field: str, operation: str) -> list[float]:
    output: list[float] = []
    for _, night in grouped:
        values = night[field].dropna().to_numpy(dtype="float64")
        if not len(values):
            continue
        operations = {
            "mean": lambda array: float(np.mean(array)),
            "min": lambda array: float(np.min(array)),
            "sd": lambda array: float(np.std(array, ddof=0)),
            "sum": lambda array: float(np.sum(array)),
        }
        output.append(operations[operation](values))
    return output


def _aggregate_sleep_physio(
    frame: pd.DataFrame, episodes: Sequence[ObservedEpisode]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if frame.empty:
        return (
            {
                "sleep_physio_valid_nights": 0,
                "sleep_hr_mean_bpm": None,
                "sleep_hr_min_bpm": None,
                "sleep_hr_sd_bpm": None,
                "sleep_hrv_sdnn_ms": None,
                "sleep_respiration_rate_mean_bpm": None,
                "sleep_respiration_rate_sd_bpm": None,
                "sleep_snoring_minutes_mean": None,
            },
            {
                "source_rows": 0,
                "unique_timestamp_rows": 0,
                "duplicate_timestamp_rows_removed": 0,
                "aligned_rows": 0,
                "valid_nights": 0,
            },
        )
    timestamps = _parse_utc(frame["Timestamp"], "Timestamp")
    numeric = pd.DataFrame(
        {
            "timestamp": timestamps,
            "heart_rate": _numeric(
                frame["Heart Rate"],
                "Sleep Heart Rate",
                minimum=0.0,
                strict_minimum=True,
            ),
            "respiration_rate": _numeric(
                frame["Respiration Rate"],
                "Respiration Rate",
                minimum=0.0,
                strict_minimum=True,
            ),
            "snoring_seconds": _numeric(frame["Snoring"], "Snoring", minimum=0.0),
            "sdnn_ms": _numeric(frame["SDNN_1"], "SDNN_1", minimum=0.0),
        }
    )
    deduplicated = numeric.groupby("timestamp", sort=True, as_index=False).mean()
    assigned = _episode_indices(deduplicated["timestamp"], episodes)
    aligned = deduplicated[assigned >= 0].copy()
    aligned["episode_index"] = assigned[assigned >= 0]
    grouped = aligned.groupby("episode_index", sort=True)
    heart_mean = _nightly_values(grouped, "heart_rate", "mean")
    heart_min = _nightly_values(grouped, "heart_rate", "min")
    heart_sd = _nightly_values(grouped, "heart_rate", "sd")
    sdnn = _nightly_values(grouped, "sdnn_ms", "mean")
    respiration_mean = _nightly_values(grouped, "respiration_rate", "mean")
    respiration_sd = _nightly_values(grouped, "respiration_rate", "sd")
    snoring = _nightly_values(grouped, "snoring_seconds", "sum")
    valid_nights = int(aligned["episode_index"].nunique())
    return (
        {
            "sleep_physio_valid_nights": valid_nights,
            "sleep_hr_mean_bpm": fmean(heart_mean) if heart_mean else None,
            "sleep_hr_min_bpm": fmean(heart_min) if heart_min else None,
            "sleep_hr_sd_bpm": fmean(heart_sd) if heart_sd else None,
            "sleep_hrv_sdnn_ms": fmean(sdnn) if sdnn else None,
            "sleep_respiration_rate_mean_bpm": (
                fmean(respiration_mean) if respiration_mean else None
            ),
            "sleep_respiration_rate_sd_bpm": (
                fmean(respiration_sd) if respiration_sd else None
            ),
            "sleep_snoring_minutes_mean": (
                fmean(value / 60.0 for value in snoring) if snoring else None
            ),
        },
        {
            "source_rows": len(frame),
            "unique_timestamp_rows": len(deduplicated),
            "duplicate_timestamp_rows_removed": len(frame) - len(deduplicated),
            "aligned_rows": len(aligned),
            "valid_nights": valid_nights,
        },
    )


def _first_valid_instant(frames: Mapping[str, pd.DataFrame]) -> pd.Timestamp:
    candidates: list[pd.Timestamp] = []
    scan_hr = frames["ScanWatch_HR.csv"]
    if not scan_hr.empty:
        timestamps = _parse_utc(scan_hr["HR Timestamp"], "HR Timestamp")
        values = _numeric(
            scan_hr["Heart Rate"],
            "ScanWatch Heart Rate",
            minimum=0.0,
            strict_minimum=True,
        )
        if values.notna().any():
            candidates.append(timestamps[values.notna()].min())
    steps = frames["ScanWatch_Steps.csv"]
    if not steps.empty:
        timestamps = _parse_utc(steps["Steps Timestamp"], "Steps Timestamp")
        values = _numeric(steps["Steps"], "Steps", minimum=0.0)
        if values.notna().any():
            candidates.append(timestamps[values.notna()].min())
    physio = frames["Sleep_physio.csv"]
    if not physio.empty:
        timestamps = _parse_utc(physio["Timestamp"], "Timestamp")
        present = pd.concat(
            [
                _numeric(physio[field], field, minimum=0.0).notna()
                for field in ("Heart Rate", "Respiration Rate", "Snoring", "SDNN_1")
            ],
            axis=1,
        ).any(axis=1)
        if present.any():
            candidates.append(timestamps[present].min())
    sleep_state = frames["Sleep_state.csv"]
    if not sleep_state.empty:
        starts = _parse_utc(sleep_state["Start time"], "Start time")
        ends = _parse_utc(sleep_state["End time"], "End time")
        states = sleep_state["Sleep state"].astype("string").str.strip()
        valid = (ends > starts) & states.isin(SLEEP_STATES)
        if valid.any():
            candidates.append(starts[valid].min())
    if not candidates:
        raise ValidationError("participant has no valid sensor timestamp")
    return min(candidates)


def _aggregate_participant(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[date, date, dict[str, Any], dict[str, Any]]:
    if set(frames) != set(SENSOR_COLUMNS):
        raise ValidationError("participant sensor file set differs from contract")
    first = _first_valid_instant(frames)
    start = first.tz_convert(TIMEZONE).date()
    end = start + timedelta(days=WINDOW_DAYS - 1)
    step_values, step_quality = _aggregate_steps(
        frames["ScanWatch_Steps.csv"], start, end
    )
    scan_hr_values, scan_hr_quality = _aggregate_scanwatch_hr(
        frames["ScanWatch_HR.csv"], start, end
    )
    sleep_values, sleep_quality, episodes = _aggregate_sleep_state(
        frames["Sleep_state.csv"], start, end
    )
    physio_values, physio_quality = _aggregate_sleep_physio(
        frames["Sleep_physio.csv"], episodes
    )
    values = {**step_values, **scan_hr_values, **sleep_values, **physio_values}
    if set(values) != set(SOURCE_FIELDS) - {
        "profile_sex",
        "profile_age_group",
        "profile_essential_hypertension",
        "profile_osteoarthritis",
    }:
        raise ValidationError("independent sensor source fields are incomplete")
    return (
        start,
        end,
        values,
        {
            "ScanWatch_Steps.csv": step_quality,
            "ScanWatch_HR.csv": scan_hr_quality,
            "Sleep_state.csv": sleep_quality,
            "Sleep_physio.csv": physio_quality,
        },
    )


def _profile_bool(value: Any) -> int | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    normalized = str(value).strip().casefold()
    if normalized == "true":
        return 1
    if normalized == "false":
        return 0
    raise ValidationError("profile boolean value differs from contract")


def _profile_values(row: pd.Series) -> dict[str, Any]:
    raw_sex = None if pd.isna(row["Sex"]) else str(row["Sex"]).strip()
    if raw_sex not in {None, "Female", "Male"}:
        raise ValidationError("profile sex value differs from contract")
    raw_age = None if pd.isna(row["Age group"]) else str(row["Age group"]).strip()
    if raw_age not in {None, "[72, 75]", "[76, 87]", "[88, 99]"}:
        raise ValidationError("profile age interval differs from contract")
    return {
        "profile_sex": {None: None, "Female": "female", "Male": "male"}[raw_sex],
        "profile_age_group": {
            None: None,
            "[72, 75]": "70_79",
            "[76, 87]": None,
            "[88, 99]": "80_plus",
        }[raw_age],
        "profile_essential_hypertension": _profile_bool(row["Essential hypertension"]),
        "profile_osteoarthritis": _profile_bool(row["Osteoarthritis"]),
    }


def _label_score(value: Any, name: str, maximum: int) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} label is invalid") from exc
    if not math.isfinite(numeric) or numeric != math.floor(numeric):
        raise ValidationError(f"{name} label is not an integer")
    result = int(numeric)
    if result < 0 or result > maximum:
        raise ValidationError(f"{name} label is outside range")
    return result


def _assessment_date(value: Any, name: str, *, allow_missing: bool) -> date | None:
    if pd.isna(value) or str(value).strip() == "":
        if allow_missing:
            return None
        raise ValidationError(f"{name} assessment date is missing")
    try:
        return pd.to_datetime(str(value), format="%d/%m/%Y", errors="raise").date()
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} assessment date is invalid") from exc


def _phq_category(score: int) -> int:
    if score <= 4:
        return 0
    if score <= 9:
        return 1
    if score <= 14:
        return 2
    if score <= 19:
        return 3
    return 4


def _canonical_target(source_values: Mapping[str, Any], target: str) -> Any:
    value = source_values[DIRECT_TARGETS[target]]
    if value is None or pd.isna(value):
        return None
    if target in {
        "sleep.sleep_duration_norm",
        "sleep.time_in_bed_norm",
        "physiology.snoring_minutes_norm",
    }:
        normalized = float(value) / 1440.0
        if normalized < 0.0 or normalized > 1.0:
            raise ValidationError("duration-derived canonical value is outside range")
        return normalized
    return value


def _load_observed_source(frozen: FrozenInputs, checks: list[str]) -> ObservedSource:
    demographics = _read_demographics(frozen.source_root)
    summary = _read_summary(frozen.source_root)
    demographic_ids = set(demographics["user_id"].astype(str))
    summary_ids = set(summary["participant"].astype(str))
    directory_ids = set(frozen.sensor_directories)
    _check(
        demographic_ids == summary_ids == directory_ids,
        "publisher_user_id_join_sets_equal",
        checks,
    )
    _check(len(demographic_ids) == EXPECTED_ROWS, "joined_participant_count", checks)
    aliases = {
        source_id: f"participant_{position:04d}"
        for position, source_id in enumerate(
            sorted(demographic_ids, key=lambda value: value.encode("utf-8")), 1
        )
    }
    demographics_by_id = demographics.set_index("user_id")
    summary_by_key = summary.set_index(["participant", "file"])
    target_names = [f"{spec['group']}.{spec['name']}" for spec in _target_specs()]
    rows: list[dict[str, Any]] = []
    quality_rows: list[Mapping[str, Any]] = []
    sensor_totals: defaultdict[str, int] = defaultdict(int)
    business_files = set(SENSOR_COLUMNS)
    for source_id in sorted(demographic_ids, key=lambda value: value.encode("utf-8")):
        directory = frozen.sensor_directories[source_id]
        file_names = {path.name for path in directory.iterdir() if path.is_file()}
        if file_names != business_files:
            raise ValidationError(
                "real participant directory does not contain four CSVs"
            )
        frames: dict[str, pd.DataFrame] = {}
        for file_name in SENSOR_COLUMNS:
            frame = _read_sensor_frame(directory, file_name)
            _reconcile_summary(
                frame, file_name, summary_by_key.loc[(source_id, file_name)]
            )
            frames[file_name] = frame
            sensor_totals[file_name] += len(frame)
        start, end, sensor_values, quality = _aggregate_participant(frames)
        quality_rows.append(quality)
        demographics_row = demographics_by_id.loc[source_id]
        source_values = {**_profile_values(demographics_row), **sensor_values}
        if set(source_values) != set(SOURCE_FIELDS):
            raise ValidationError("independent source field set is incomplete")
        phq = _label_score(demographics_row["phq_total"], "PHQ-9", 27)
        gds = _label_score(demographics_row["gds_total"], "GDS-15", 15)
        gad = _label_score(demographics_row["gad_total"], "GAD-7", 21)
        alias = aliases[source_id]
        row: dict[str, Any] = {
            "dataset_id": DATASET_ID,
            "global_participant_id": f"{DATASET_ID}::{alias}",
            "participant_id": alias,
            "timescale_semantics": "assessment_nearby_state_association",
            "source_timezone": TIMEZONE,
            "window_start_date": start,
            "window_end_date": end,
            "window_calendar_days": WINDOW_DAYS,
            "phq_assessment_date": _assessment_date(
                demographics_row["phq_date"], "PHQ-9", allow_missing=False
            ),
            "phq9_score": phq,
            "phq9_category": _phq_category(phq),
            "binary_target": int(phq >= 10),
            "gds_assessment_date": _assessment_date(
                demographics_row["gds_date"], "GDS-15", allow_missing=False
            ),
            "gds15_score": gds,
            "gds15_binary_target": int(gds >= 5),
            "gad_assessment_date": _assessment_date(
                demographics_row["gad_date"], "GAD-7", allow_missing=True
            ),
            "gad7_score": gad,
            "gad7_binary_target": int(gad >= 10),
            "x_source_name": "scanwatch_steps_daily_mean",
            "x_source_value": source_values["scanwatch_steps_daily_mean"],
            "x_source_mask": int(
                source_values["scanwatch_steps_daily_mean"] is not None
            ),
        }
        row.update({f"source__{name}": source_values[name] for name in SOURCE_FIELDS})
        for target in target_names:
            value = (
                _canonical_target(source_values, target)
                if target in DIRECT_TARGETS
                else None
            )
            row[target] = value
            row[f"feature_mask.{target}"] = int(value is not None)
        rows.append(row)
    _check(dict(sensor_totals) == EXPECTED_SENSOR_ROWS, "sensor_row_totals", checks)
    canonical = (
        pd.DataFrame(rows)
        .sort_values("participant_id", kind="stable")
        .reset_index(drop=True)
    )
    canonical = canonical.loc[:, list(_column_order())]
    return ObservedSource(
        canonical=canonical,
        quality_rows=tuple(quality_rows),
        sensor_row_totals=dict(sensor_totals),
        source_participant_ids=frozenset(demographic_ids),
        demographics_columns=tuple(str(name) for name in demographics.columns),
    )


def _series_equal(
    actual: pd.Series, expected: pd.Series, arrow_type: pa.DataType
) -> bool:
    if pa.types.is_large_string(arrow_type):

        def normalize(value: Any) -> str | None:
            return None if pd.isna(value) else str(value)

        return [normalize(value) for value in actual] == [
            normalize(value) for value in expected
        ]
    if pa.types.is_date32(arrow_type):

        def normalize_date(value: Any) -> date | None:
            if pd.isna(value):
                return None
            if isinstance(value, date):
                return value
            return pd.Timestamp(value).date()

        return [normalize_date(value) for value in actual] == [
            normalize_date(value) for value in expected
        ]
    try:
        actual_values = pd.to_numeric(actual, errors="raise").to_numpy(
            dtype="float64", na_value=np.nan
        )
        expected_values = pd.to_numeric(expected, errors="raise").to_numpy(
            dtype="float64", na_value=np.nan
        )
    except (TypeError, ValueError):
        return False
    return bool(
        np.allclose(
            actual_values,
            expected_values,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
    )


def _canonical_frame_sha256(frame: pd.DataFrame) -> str:
    schema = _arrow_schema()
    try:
        arrays = [
            pa.array(
                frame[field.name].tolist(),
                type=field.type,
                from_pandas=True,
                safe=True,
            )
            for field in schema
        ]
        table = pa.Table.from_arrays(arrays, schema=schema).combine_chunks()
    except (KeyError, TypeError, ValueError, pa.ArrowException) as exc:
        raise ValidationError(
            "canonical values do not match independent Arrow schema"
        ) from exc
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table, max_chunksize=max(len(table), 1))
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _missing_summary(series: pd.Series) -> dict[str, Any]:
    missing = int(series.isna().sum())
    total = len(series)
    return {
        "present_count": total - missing,
        "missing_count": missing,
        "missing_fraction": 0.0 if total == 0 else missing / total,
    }


def _integer_distribution(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    array = np.asarray(values, dtype=np.int64)
    return {
        "count": len(values),
        "minimum": int(np.min(array)),
        "median": float(np.median(array)),
        "maximum": int(np.max(array)),
    }


def _validate_mapping(
    mapping: Mapping[str, Any], target_names: Sequence[str], checks: list[str]
) -> None:
    _check(mapping.get("dataset_id") == DATASET_ID, "mapping_dataset_id", checks)
    _check(
        mapping.get("source_collection_sha256") == COLLECTION_SHA256
        and mapping.get("file_manifest_sha256") == MANIFEST_SHA256
        and mapping.get("feature_schema_sha256") == SCHEMA_SHA256,
        "mapping_frozen_input_bindings",
        checks,
    )
    source_rows = mapping.get("source_fields")
    _check(
        isinstance(source_rows, list)
        and [row.get("source_field") for row in source_rows] == list(SOURCE_FIELDS)
        and [row.get("canonical_source_column") for row in source_rows]
        == [f"source__{name}" for name in SOURCE_FIELDS],
        "mapping_source_field_coverage_and_order",
        checks,
    )
    target_rows = mapping.get("target_fields")
    _check(
        isinstance(target_rows, list)
        and [row.get("canonical_value_column") for row in target_rows]
        == list(target_names),
        "mapping_target_field_coverage_and_order",
        checks,
    )
    rows_by_target = {str(row["canonical_value_column"]): row for row in target_rows}
    for target in target_names:
        row = rows_by_target[target]
        if target == "activity.activity_volume_norm":
            _check(
                row.get("mapping_status") == "fold_derived"
                and row.get("source_fields") == ["scanwatch_steps_daily_mean"]
                and row.get("formula") == "right_continuous_ecdf(count(train <= x) / n)"
                and row.get("base_canonical_policy")
                == "null_with_mask_0_until_explicit_training_fold_fit",
                "mapping_fold_only_activity_volume",
                checks,
            )
        elif target in DIRECT_TARGETS:
            _check(
                row.get("mapping_status") == "mapped"
                and row.get("source_fields") == [DIRECT_TARGETS[target]]
                and row.get("formula") == EXPECTED_FORMULAS[target]
                and row.get("base_canonical_policy")
                == "mapped_when_source_is_valid_else_null_with_mask_0",
                f"mapping_strict_semantics_{target}",
                checks,
            )
        else:
            _check(
                row.get("mapping_status") == "unsupported"
                and row.get("source_fields") == []
                and row.get("formula") is None
                and row.get("base_canonical_policy") == "null_with_mask_0",
                f"mapping_unsupported_{target}",
                checks,
            )
    unsupported = [name for name in target_names if name not in DIRECT_TARGETS]
    _check(
        mapping.get("base_supported_feature_set") == sorted(DIRECT_TARGETS)
        and mapping.get("fold_supported_feature_set")
        == ["activity.activity_volume_norm"]
        and mapping.get("unsupported_feature_set")
        == [name for name in unsupported if name != "activity.activity_volume_norm"],
        "mapping_supported_and_unsupported_sets",
        checks,
    )
    labels = mapping.get("labels", {})
    _check(
        labels.get("binary_target") == "phq_total >= 10"
        and labels.get("gds15_binary_target") == "gds_total >= 5"
        and labels.get("gad7_binary_target") == "gad_total >= 10"
        and labels.get("input_feature_use") == "prohibited"
        and labels.get("questionnaire_items_and_dates_as_model_features") is False,
        "mapping_label_and_leakage_contract",
        checks,
    )
    _check(
        mapping.get("target_leakage_exclusions") == LEAKAGE_EXCLUSIONS,
        "mapping_target_leakage_exclusions",
        checks,
    )
    timezone_contract = mapping.get("timezone_contract", {})
    window_contract = mapping.get("window_contract", {})
    _check(
        timezone_contract.get("source_timestamps") == "UTC"
        and timezone_contract.get("calendar_timezone") == TIMEZONE
        and window_contract.get("length_calendar_days") == WINDOW_DAYS
        and window_contract.get("future_prediction_claim") is False,
        "mapping_timezone_and_window_contract",
        checks,
    )
    sensors = mapping.get("sensor_contract", {})
    _check(
        sensors.get("business_csvs") == list(SENSOR_COLUMNS)
        and sensors.get("os_metadata_parsed") is False
        and "increments"
        in sensors.get("ScanWatch_Steps.csv", {}).get("daily_total", "")
        and "wakeup"
        not in sensors.get("Sleep_state.csv", {}).get("wakeup_semantics", "").casefold()
        and sensors.get("Sleep_state.csv", {}).get("wakeup_semantics") == "awake state"
        and sensors.get("Sleep_physio.csv", {}).get("respiratory_abnormal_ratio")
        == "unsupported: no frozen clinical range",
        "mapping_sensor_contract",
        checks,
    )
    ecdf = mapping.get("ecdf_contract", {})
    _check(
        ecdf.get("fit_scope") == "explicit outer-training participants only"
        and ecdf.get("ties") == "right_continuous"
        and ecdf.get("formal_instances_before_data007") is False,
        "mapping_ecdf_deferred_until_data007",
        checks,
    )


def _validate_quality(
    quality: Mapping[str, Any], observed: ObservedSource, checks: list[str]
) -> None:
    expected = observed.canonical
    _check(
        quality.get("input_binding") == _expected_binding(), "quality_binding", checks
    )
    _check(
        quality.get("timescale_semantics") == "assessment_nearby_state_association"
        and quality.get("future_prediction_claim") is False,
        "quality_timescale_semantics",
        checks,
    )
    join = quality.get("join_summary", {})
    _check(
        join.get("demographics_rows") == EXPECTED_ROWS
        and join.get("summary_rows") == EXPECTED_SUMMARY_ROWS
        and join.get("real_sensor_directories") == EXPECTED_ROWS
        and join.get("joined_participants") == EXPECTED_ROWS
        and join.get("publisher_user_id_sets_equal") is True
        and join.get("source_participant_values_written_to_report") is False,
        "quality_join_summary",
        checks,
    )
    _check(
        quality.get("os_metadata")
        == {
            "included_in_source_collection_hash": True,
            "parsed_as_business_csv": False,
        },
        "quality_os_metadata_boundary",
        checks,
    )
    row_summary = quality.get("sensor_row_summary", {})
    _check(
        row_summary
        == {
            name: {"source_rows": count, "file_count": EXPECTED_ROWS}
            for name, count in EXPECTED_SENSOR_ROWS.items()
        },
        "quality_sensor_row_summary",
        checks,
    )
    step_days = [
        int(row["ScanWatch_Steps.csv"]["valid_days"]) for row in observed.quality_rows
    ]
    hr_days = [
        int(row["ScanWatch_HR.csv"]["valid_days"]) for row in observed.quality_rows
    ]
    sleep_nights = [
        int(row["Sleep_state.csv"]["window_main_sleep_night_count"])
        for row in observed.quality_rows
    ]
    physio_nights = [
        int(row["Sleep_physio.csv"]["valid_nights"]) for row in observed.quality_rows
    ]
    window = quality.get("window_summary", {})
    _check(
        window.get("source_timestamp_timezone") == "UTC"
        and window.get("calendar_timezone") == TIMEZONE
        and window.get("calendar_days") == WINDOW_DAYS
        and window.get("step_valid_days") == _integer_distribution(step_days)
        and window.get("scanwatch_hr_valid_days") == _integer_distribution(hr_days)
        and window.get("main_sleep_valid_nights") == _integer_distribution(sleep_nights)
        and window.get("sleep_physio_valid_nights")
        == _integer_distribution(physio_nights),
        "quality_window_distributions",
        checks,
    )
    sensor_quality = quality.get("sensor_quality_summary", {})
    _check(
        sensor_quality.get("sleep_physio_duplicate_timestamp_rows_removed")
        == sum(
            int(row["Sleep_physio.csv"]["duplicate_timestamp_rows_removed"])
            for row in observed.quality_rows
        )
        and sensor_quality.get("sleep_physio_rows_aligned_to_main_sleep")
        == sum(
            int(row["Sleep_physio.csv"]["aligned_rows"])
            for row in observed.quality_rows
        )
        and sensor_quality.get("sleep_episode_invalid_count")
        == sum(
            int(row["Sleep_state.csv"]["discarded_invalid_episode_count"])
            for row in observed.quality_rows
        ),
        "quality_independent_sensor_aggregates",
        checks,
    )
    _check(
        quality.get("source_field_missingness")
        == {
            name: _missing_summary(expected[f"source__{name}"])
            for name in SOURCE_FIELDS
        },
        "quality_source_missingness",
        checks,
    )
    target_names = [f"{spec['group']}.{spec['name']}" for spec in _target_specs()]
    expected_target_missingness = {}
    for target in target_names:
        mask = f"feature_mask.{target}"
        expected_target_missingness[target] = {
            **_missing_summary(expected[target]),
            "mask_1_count": int(expected[mask].sum()),
            "mask_0_count": int((expected[mask] == 0).sum()),
        }
    _check(
        quality.get("canonical_target_missingness_and_masks")
        == expected_target_missingness,
        "quality_target_missingness_and_masks",
        checks,
    )
    categories = expected["phq9_category"].value_counts().sort_index()
    labels = quality.get("label_summary", {})
    _check(
        labels.get("binary_definition") == "phq_total >= 10"
        and labels.get("positive_count") == EXPECTED_POSITIVES
        and labels.get("negative_count") == EXPECTED_ROWS - EXPECTED_POSITIVES
        and labels.get("phq9_category_counts")
        == {str(int(key)): int(value) for key, value in categories.items()}
        and labels.get("gds15_at_or_above_5_count") == EXPECTED_GDS_POSITIVES
        and labels.get("gad7_at_or_above_10_count") == EXPECTED_GAD_POSITIVES
        and labels.get("label_fields_used_as_input_features") is False,
        "quality_label_summary",
        checks,
    )
    _check(
        quality.get("x_source")
        == {
            "field": "scanwatch_steps_daily_mean",
            **_missing_summary(expected["x_source_value"]),
            "base_canonical_ecdf_fitted": False,
        },
        "quality_x_source_without_ecdf",
        checks,
    )


def validate(
    *,
    workspace_root: Path | None = None,
    manifests_root: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Recompute DATA-003 source semantics and validate every published artifact."""

    checks: list[str] = []
    frozen = _validate_frozen_inputs(workspace_root, manifests_root, checks)
    destination = (
        output_root
        or frozen.workspace_root
        / "algorithm"
        / "eldercare-risk-ai-main"
        / OUTPUT_RELATIVE
    ).resolve()
    manifest_path = destination / "resilient" / "artifact_manifest.json"
    artifact_manifest = _load_json(manifest_path)
    _check(
        artifact_manifest.get("dataset_id") == DATASET_ID
        and artifact_manifest.get("complete") is True
        and artifact_manifest.get("artifact_count") == len(EXPECTED_ARTIFACTS)
        and artifact_manifest.get("input_binding") == _expected_binding(),
        "artifact_manifest_contract",
        checks,
    )
    artifacts = artifact_manifest.get("artifacts")
    _check(
        isinstance(artifacts, dict) and set(artifacts) == EXPECTED_ARTIFACTS,
        "artifact_manifest_exact_path_set",
        checks,
    )
    for relative, expected_artifact in artifacts.items():
        path = destination / Path(relative)
        _check(path.is_file(), f"artifact_exists_{relative}", checks)
        _check(
            path.stat().st_size == int(expected_artifact["bytes"]),
            f"artifact_size_{relative}",
            checks,
        )
        _check(
            _sha256_file(path) == expected_artifact["sha256"],
            f"artifact_sha256_{relative}",
            checks,
        )
    resilient_files = {
        path.name for path in (destination / "resilient").iterdir() if path.is_file()
    }
    _check(
        resilient_files
        == {
            "adapter_metadata.json",
            "artifact_manifest.json",
            "canonical_resilient.parquet",
            "data_quality_report.json",
        },
        "no_unfrozen_or_ecdf_resilient_artifacts",
        checks,
    )

    observed = _load_observed_source(frozen, checks)
    canonical_path = destination / "resilient" / "canonical_resilient.parquet"
    parquet_schema = pq.read_schema(canonical_path)
    canonical = pd.read_parquet(canonical_path)
    schema = _arrow_schema()
    _check(parquet_schema == schema, "canonical_arrow_schema", checks)
    _check(
        tuple(canonical.columns) == _column_order(), "canonical_column_order", checks
    )
    _check(len(canonical) == EXPECTED_ROWS, "canonical_row_count", checks)
    _check(
        canonical["participant_id"].nunique() == EXPECTED_ROWS,
        "canonical_participant_count",
        checks,
    )
    for field in schema:
        _check(
            _series_equal(
                canonical[field.name], observed.canonical[field.name], field.type
            ),
            f"canonical_independent_value_{field.name}",
            checks,
        )
    _check(
        int(canonical["binary_target"].sum()) == EXPECTED_POSITIVES,
        "phq_binary_target_count",
        checks,
    )
    _check(
        int(canonical["gds15_binary_target"].sum()) == EXPECTED_GDS_POSITIVES,
        "gds_binary_target_count",
        checks,
    )
    _check(
        int(canonical["gad7_binary_target"].sum()) == EXPECTED_GAD_POSITIVES,
        "gad_binary_target_count",
        checks,
    )
    categories = {
        int(key): int(value)
        for key, value in canonical["phq9_category"].value_counts().items()
    }
    _check(categories == EXPECTED_CATEGORIES, "phq_category_distribution", checks)
    _check(
        int(canonical["social_context.sex"].notna().sum()) == 72
        and int(canonical["social_context.age_group"].notna().sum()) == 25,
        "strict_profile_mapping_counts",
        checks,
    )
    _check(
        canonical["activity.activity_volume_norm"].isna().all()
        and int(canonical["feature_mask.activity.activity_volume_norm"].sum()) == 0,
        "base_activity_ecdf_not_fitted",
        checks,
    )
    _check(
        canonical["physiology.respiratory_abnormal_ratio"].isna().all()
        and int(canonical["feature_mask.physiology.respiratory_abnormal_ratio"].sum())
        == 0,
        "respiratory_threshold_not_invented",
        checks,
    )
    _check(
        not {
            "scanwatch_hr_mean_bpm",
            "scanwatch_hr_min_bpm",
            "scanwatch_hr_sd_bpm",
        }.intersection(DIRECT_TARGETS.values()),
        "scanwatch_hr_not_mapped_as_sleep_physiology",
        checks,
    )
    non_label_columns = set(canonical.columns) - set(TARGET_COLUMNS)
    _check(
        not any(
            token in name.casefold()
            for name in non_label_columns
            for token in ("phq", "gds", "gad", "ace")
        ),
        "questionnaire_and_ace_fields_excluded_from_inputs",
        checks,
    )

    mapping = _load_json(destination / "mappings" / "resilient_v3_3_3_mapping.json")
    metadata = _load_json(destination / "resilient" / "adapter_metadata.json")
    quality = _load_json(destination / "resilient" / "data_quality_report.json")
    target_names = [f"{spec['group']}.{spec['name']}" for spec in _target_specs()]
    _validate_mapping(mapping, target_names, checks)
    _validate_quality(quality, observed, checks)
    _check(
        metadata.get("input_binding")
        == quality.get("input_binding")
        == artifact_manifest.get("input_binding")
        == _expected_binding(),
        "all_json_input_bindings_match",
        checks,
    )
    preliminary_paths = {
        "resilient/canonical_resilient.parquet",
        "resilient/data_quality_report.json",
        "mappings/resilient_v3_3_3_mapping.json",
    }
    _check(
        metadata.get("canonical_column_order") == list(_column_order())
        and metadata.get("canonical_row_count") == EXPECTED_ROWS
        and metadata.get("canonical_participant_count") == EXPECTED_ROWS
        and metadata.get("canonical_frame_sha256") == _canonical_frame_sha256(canonical)
        and metadata.get("artifact_sha256_before_metadata")
        == {relative: artifacts[relative]["sha256"] for relative in preliminary_paths},
        "metadata_canonical_and_artifact_hash_bindings",
        checks,
    )
    _check(
        metadata.get("base_canonical_ecdf_fitted") is False
        and metadata.get("formal_ecdf_instance_created") is False
        and metadata.get("calendar_timezone") == TIMEZONE
        and metadata.get("source_timestamp_timezone") == "UTC"
        and metadata.get("window_calendar_days") == WINDOW_DAYS,
        "metadata_window_and_deferred_ecdf_contract",
        checks,
    )
    json_paths = (
        destination / "resilient" / "adapter_metadata.json",
        destination / "resilient" / "artifact_manifest.json",
        destination / "resilient" / "data_quality_report.json",
        destination / "mappings" / "resilient_v3_3_3_mapping.json",
    )
    report_text = "\n".join(path.read_text(encoding="utf-8") for path in json_paths)
    _check(
        re.search(r"(?:resilient::)?participant_\d{4}", report_text) is None
        and not any(
            source_id in report_text for source_id in observed.source_participant_ids
        ),
        "json_reports_exclude_all_participant_values",
        checks,
    )

    artifact_hashes = {
        relative: _sha256_file(destination / Path(relative))
        for relative in sorted(EXPECTED_ARTIFACTS)
    }
    artifact_hashes["resilient/artifact_manifest.json"] = _sha256_file(manifest_path)
    return {
        "validation_version": "resilient-independent-validation-v1",
        "dataset_id": DATASET_ID,
        "status": "pass",
        "check_count": len(checks),
        "checks": checks,
        "canonical": {
            "row_count": len(canonical),
            "participant_count": int(canonical["participant_id"].nunique()),
            "phq9_positive_count": int(canonical["binary_target"].sum()),
            "gds15_positive_count": int(canonical["gds15_binary_target"].sum()),
            "gad7_positive_count": int(canonical["gad7_binary_target"].sum()),
            "phq9_category_counts": {
                str(key): value for key, value in sorted(categories.items())
            },
        },
        "sensor_row_totals": dict(observed.sensor_row_totals),
        "artifact_sha256": artifact_hashes,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--manifests-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = validate(
        workspace_root=args.workspace_root,
        manifests_root=args.manifests_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
