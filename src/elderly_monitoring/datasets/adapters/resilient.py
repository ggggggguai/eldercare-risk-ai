"""RESILIENT V3.3.3 canonical adapter and fold-bound activity ECDF.

The publisher timestamps are UTC. Calendar windows are evaluated in
Europe/London so that wearable days and cross-midnight sleep keep their local
meaning across daylight-saving transitions. The base canonical table never
contains a fitted activity ECDF.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
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
ADAPTER_VERSION = "resilient-adapter-v3.3.3-data003"
CANONICAL_COLUMN_ORDER_VERSION = "resilient-canonical-columns-v1"
MAPPING_VERSION = "resilient-field-mapping-v3.3.3-data003"
QUALITY_REPORT_VERSION = "resilient-quality-report-v1"
ARTIFACT_MANIFEST_VERSION = "resilient-artifact-manifest-v1"
ECDF_VERSION = "training-fold-right-continuous-ecdf-v1"

SOURCE_RELATIVE = Path("数据集/心理/RESILIENT")
DEFAULT_OUTPUT_RELATIVE = Path("data/processed/mental_health/mood_social/v3.3.3-r2")
SENSOR_RELATIVE = Path("Sleepmat_Watch_Data/Sleepmat_Watch_Data")
DEMOGRAPHICS_NAME = "Demographics.csv"
SUMMARY_NAME = "summary_stats_per_participant.csv"
SOURCE_TIMEZONE = "Europe/London"
SOURCE_TIMESTAMP_TIMEZONE = "UTC"
WINDOW_CALENDAR_DAYS = 14
MIN_STEP_HOURLY_BINS = 20
SLEEP_EPISODE_MAX_GAP_MINUTES = 60.0
MIN_MAIN_SLEEP_MINUTES = 60.0
MAX_MAIN_SLEEP_MINUTES = 960.0

RESILIENT_COLLECTION_SHA256 = (
    "95abce53d3c0a2303fd8f6a0de52c76e5552c51e8d1f90b921ac88b2a3dfaed3"
)
DATA001_FILE_MANIFEST_SHA256 = (
    "6591374bb4273c073126c1ed750c75bd852e08bddb2104797bcf893aec8f11ff"
)
MH003_FEATURE_SCHEMA_SHA256 = (
    "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
)

EXPECTED_SOURCE_FILE_COUNT = 589
EXPECTED_PARTICIPANTS = 73
EXPECTED_SUMMARY_ROWS = 292
EXPECTED_SENSOR_ROWS = {
    "ScanWatch_HR.csv": 991_325,
    "ScanWatch_Steps.csv": 241_505,
    "Sleep_physio.csv": 1_115_748,
    "Sleep_state.csv": 1_132_263,
}
EXPECTED_POSITIVES = 10
EXPECTED_GDS_POSITIVES = 23
EXPECTED_GAD_POSITIVES = 6
EXPECTED_PHQ_CATEGORY_COUNTS = {0: 37, 1: 26, 2: 5, 3: 4, 4: 1}

SENSOR_FILES = (
    "ScanWatch_HR.csv",
    "ScanWatch_Steps.csv",
    "Sleep_physio.csv",
    "Sleep_state.csv",
)
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
SLEEP_STATES = frozenset({"REM", "deep", "light", "wakeup"})
_TIMESTAMP_OFFSET = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")
_EPSILON = 1e-12

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
X_SOURCE_FIELD = "scanwatch_steps_daily_mean"
X_SOURCE_COLUMNS = ("x_source_name", "x_source_value", "x_source_mask")
X_SOURCE_CANONICAL_COLUMN = f"source__{X_SOURCE_FIELD}"
ACTIVITY_VOLUME_COLUMN = "activity.activity_volume_norm"
ACTIVITY_VOLUME_MASK_COLUMN = "feature_mask.activity.activity_volume_norm"
TIMESCALE_SEMANTICS = "assessment_nearby_state_association"

SOURCE_FIELD_TYPES: dict[str, pa.DataType] = {
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
SOURCE_FIELDS = tuple(SOURCE_FIELD_TYPES)

_DIRECT_TARGET_RULES: dict[str, dict[str, Any]] = {
    "activity.relative_amplitude": {
        "source_fields": ["scanwatch_relative_amplitude"],
        "formula": "publisher hourly step increments; M10/L5 relative amplitude",
        "mapping_kind": "derived_same_semantics",
    },
    "activity.interdaily_stability": {
        "source_fields": ["scanwatch_interdaily_stability"],
        "formula": "coverage-weighted hour-of-day variance / total variance",
        "mapping_kind": "derived_same_semantics",
    },
    "activity.intradaily_variability": {
        "source_fields": ["scanwatch_intradaily_variability"],
        "formula": "mean adjacent-hour squared difference / total variance",
        "mapping_kind": "derived_same_semantics",
    },
    "activity.activity_variability": {
        "source_fields": ["scanwatch_activity_variability"],
        "formula": "population coefficient of variation of valid daily step totals",
        "mapping_kind": "derived_same_semantics",
    },
    "sleep.sleep_duration_norm": {
        "source_fields": ["sleep_duration_minutes_mean"],
        "formula": "sleep_duration_minutes_mean / 1440",
        "mapping_kind": "derived_same_semantics",
    },
    "sleep.time_in_bed_norm": {
        "source_fields": ["time_in_bed_minutes_mean"],
        "formula": "time_in_bed_minutes_mean / 1440",
        "mapping_kind": "derived_same_semantics",
    },
    "sleep.sleep_efficiency": {
        "source_fields": ["sleep_efficiency"],
        "formula": "sum(asleep minutes) / sum(observed in-bed minutes)",
        "mapping_kind": "derived_same_semantics",
    },
    "sleep.sleep_onset_sin": {
        "source_fields": ["sleep_onset_sin"],
        "formula": "circular mean component of local sleep onset",
        "mapping_kind": "derived_circular_same_semantics",
    },
    "sleep.sleep_onset_cos": {
        "source_fields": ["sleep_onset_cos"],
        "formula": "circular mean component of local sleep onset",
        "mapping_kind": "derived_circular_same_semantics",
    },
    "sleep.wake_time_sin": {
        "source_fields": ["wake_time_sin"],
        "formula": "circular mean component of local wake time",
        "mapping_kind": "derived_circular_same_semantics",
    },
    "sleep.wake_time_cos": {
        "source_fields": ["wake_time_cos"],
        "formula": "circular mean component of local wake time",
        "mapping_kind": "derived_circular_same_semantics",
    },
    "sleep.sleep_midpoint_sin": {
        "source_fields": ["sleep_midpoint_sin"],
        "formula": "circular mean component of forward local sleep midpoint",
        "mapping_kind": "derived_circular_same_semantics",
    },
    "sleep.sleep_midpoint_cos": {
        "source_fields": ["sleep_midpoint_cos"],
        "formula": "circular mean component of forward local sleep midpoint",
        "mapping_kind": "derived_circular_same_semantics",
    },
    "sleep.sleep_fragmentation": {
        "source_fields": ["sleep_fragmentation"],
        "formula": "mean(awake/in_bed + awakenings/(asleep_minutes/60))",
        "mapping_kind": "derived_same_semantics",
    },
    "sleep.sleep_regularity": {
        "source_fields": ["sleep_regularity"],
        "formula": "mean circular resultant length for onset, wake, midpoint",
        "mapping_kind": "derived_same_semantics",
    },
    "physiology.heart_rate_mean_bpm": {
        "source_fields": ["sleep_hr_mean_bpm"],
        "formula": "mean of valid nightly means after timestamp deduplication",
        "mapping_kind": "derived_same_semantics",
    },
    "physiology.heart_rate_min_bpm": {
        "source_fields": ["sleep_hr_min_bpm"],
        "formula": "mean of valid nightly minima after timestamp deduplication",
        "mapping_kind": "derived_same_semantics",
    },
    "physiology.heart_rate_sd_bpm": {
        "source_fields": ["sleep_hr_sd_bpm"],
        "formula": "mean of valid nightly population standard deviations",
        "mapping_kind": "derived_same_semantics",
    },
    "physiology.hrv_sdnn_ms": {
        "source_fields": ["sleep_hrv_sdnn_ms"],
        "formula": "mean of valid nightly SDNN_1 means",
        "mapping_kind": "direct_same_semantics",
    },
    "physiology.respiration_rate_mean_bpm": {
        "source_fields": ["sleep_respiration_rate_mean_bpm"],
        "formula": "mean of valid nightly respiration means",
        "mapping_kind": "derived_same_semantics",
    },
    "physiology.respiration_rate_sd_bpm": {
        "source_fields": ["sleep_respiration_rate_sd_bpm"],
        "formula": "mean of valid nightly population standard deviations",
        "mapping_kind": "derived_same_semantics",
    },
    "physiology.snoring_minutes_norm": {
        "source_fields": ["sleep_snoring_minutes_mean"],
        "formula": "mean nightly sum(source seconds) / 60 / 1440",
        "mapping_kind": "derived_same_semantics",
    },
    "social_context.sex": {
        "source_fields": ["profile_sex"],
        "formula": "Female->female; Male->male",
        "mapping_kind": "category_same_semantics",
    },
    "social_context.age_group": {
        "source_fields": ["profile_age_group"],
        "formula": "map only source intervals wholly contained in a target interval",
        "mapping_kind": "partial_category_same_semantics",
    },
}


class ResilientAdapterError(ValueError):
    """Raised when frozen source, schema, or canonical contracts do not match."""


@dataclass(frozen=True)
class SourceBinding:
    workspace_root: Path
    source_root: Path
    manifests_root: Path
    source_collection_sha256: str
    file_manifest_sha256: str
    feature_schema_sha256: str
    source_file_count: int

    def report_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": DATASET_ID,
            "source_relative": SOURCE_RELATIVE.as_posix(),
            "source_collection_sha256": self.source_collection_sha256,
            "file_manifest_sha256": self.file_manifest_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "source_file_count": self.source_file_count,
            "all_bindings_validated": True,
        }


@dataclass(frozen=True)
class SleepEpisode:
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
class AggregatedSensorWindow:
    window_start_date: date
    window_end_date: date
    source_values: Mapping[str, Any]
    quality: Mapping[str, Any]


@dataclass(frozen=True)
class BuildResult:
    output_root: Path
    artifact_sha256: Mapping[str, str]
    row_count: int
    participant_count: int
    positive_row_count: int


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


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collection_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(
        rows,
        key=lambda item: str(item["relative_path"]).encode("utf-8"),
    ):
        digest.update(
            (f"{row['relative_path']}\0{row['bytes']}\0{row['sha256']}\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _find_workspace_root(start: Path) -> Path:
    resolved = start.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / SOURCE_RELATIVE).is_dir() and (
            candidate / "algorithm" / "eldercare-risk-ai-main"
        ).is_dir():
            return candidate
    raise ResilientAdapterError(
        "workspace root containing the frozen dataset was not found"
    )


def _algorithm_root(workspace_root: Path) -> Path:
    path = workspace_root / "algorithm" / "eldercare-risk-ai-main"
    if not path.is_dir():
        raise ResilientAdapterError("algorithm repository was not found")
    return path


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResilientAdapterError(
            f"frozen JSON input is not readable: {path.name}"
        ) from exc


def _read_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ResilientAdapterError(
                        "DATA-001 file manifest contains a non-object row"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResilientAdapterError("DATA-001 file manifest is not readable") from exc
    return rows


def _sensor_directories(source_root: Path) -> list[Path]:
    root = source_root / SENSOR_RELATIVE
    if not root.is_dir():
        raise ResilientAdapterError("real RESILIENT sensor root is missing")
    directories = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name.encode("utf-8"),
    )
    if len(directories) != EXPECTED_PARTICIPANTS:
        raise ResilientAdapterError("RESILIENT sensor directory count does not match")
    return directories


def _participant_aliases(source_root: Path) -> dict[str, str]:
    return {
        directory.name: f"participant_{index:04d}"
        for index, directory in enumerate(_sensor_directories(source_root), 1)
    }


def _classify_source_file(path: Path, source_root: Path) -> tuple[str, str]:
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
    raise ResilientAdapterError("RESILIENT source contains an unsupported file")


def _public_relative_path(
    workspace_root: Path,
    source_root: Path,
    source_path: Path,
    aliases: Mapping[str, str],
) -> str:
    parts = list(source_path.relative_to(source_root).parts)
    for index, part in enumerate(parts):
        prefix = "._" if part.startswith("._") else ""
        candidate = part[2:] if prefix else part
        if candidate in aliases:
            parts[index] = f"{prefix}{aliases[candidate]}"
    logical = workspace_root / SOURCE_RELATIVE / Path(*parts)
    return logical.relative_to(workspace_root).as_posix()


def validate_frozen_inputs(
    workspace_root: Path | None = None,
    manifests_root: Path | None = None,
) -> SourceBinding:
    """Revalidate the complete DATA-001 collection and live MH-003 schema."""

    root = _find_workspace_root(workspace_root or Path.cwd())
    algorithm = _algorithm_root(root)
    audit_root = (
        manifests_root or algorithm / DEFAULT_OUTPUT_RELATIVE / "manifests"
    ).resolve()
    manifest_path = audit_root / "dataset_file_manifest.jsonl"
    schema_path = audit_root / "feature_schema_manifest.json"
    if _sha256_file(manifest_path) != DATA001_FILE_MANIFEST_SHA256:
        raise ResilientAdapterError("DATA-001 file manifest SHA-256 does not match")
    live_schema = feature_schema_manifest()
    live_schema_hash = _sha256_bytes(_canonical_json_bytes(live_schema))
    if live_schema_hash != MH003_FEATURE_SCHEMA_SHA256:
        raise ResilientAdapterError("live MH-003 feature schema SHA-256 does not match")
    if _sha256_file(schema_path) != MH003_FEATURE_SCHEMA_SHA256:
        raise ResilientAdapterError(
            "DATA-001 feature schema snapshot SHA-256 does not match"
        )
    if _read_json(schema_path) != live_schema:
        raise ResilientAdapterError(
            "DATA-001 schema snapshot differs from live MH-003 schema"
        )

    manifest_rows = _read_manifest_rows(manifest_path)
    selected = [row for row in manifest_rows if row.get("dataset_id") == DATASET_ID]
    if len(selected) != EXPECTED_SOURCE_FILE_COUNT:
        raise ResilientAdapterError("DATA-001 RESILIENT file count does not match")
    expected_keys = {
        "bytes",
        "dataset_id",
        "file_role",
        "format",
        "relative_path",
        "sha256",
    }
    if any(set(row) != expected_keys for row in selected):
        raise ResilientAdapterError("DATA-001 RESILIENT manifest fields do not match")

    source_root = root / SOURCE_RELATIVE
    aliases = _participant_aliases(source_root)
    actual_files = sorted(
        (path for path in source_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix().encode("utf-8"),
    )
    if len(actual_files) != EXPECTED_SOURCE_FILE_COUNT:
        raise ResilientAdapterError("current RESILIENT file set differs from DATA-001")
    current_public_rows: list[dict[str, Any]] = []
    collection_rows: list[dict[str, Any]] = []
    for path in actual_files:
        if path.is_symlink():
            raise ResilientAdapterError("RESILIENT source files must not be symlinks")
        before = path.stat()
        file_hash = _sha256_file(path)
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise ResilientAdapterError("RESILIENT source changed during validation")
        role, file_format = _classify_source_file(path, source_root)
        current_public_rows.append(
            {
                "bytes": int(after.st_size),
                "dataset_id": DATASET_ID,
                "file_role": role,
                "format": file_format,
                "relative_path": _public_relative_path(
                    root, source_root, path, aliases
                ),
                "sha256": file_hash,
            }
        )
        collection_rows.append(
            {
                "bytes": int(after.st_size),
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": file_hash,
            }
        )

    def row_key(row: Mapping[str, Any]) -> bytes:
        return str(row["relative_path"]).encode("utf-8")

    if sorted(current_public_rows, key=row_key) != sorted(selected, key=row_key):
        raise ResilientAdapterError(
            "current RESILIENT file path/hash bindings differ from DATA-001"
        )
    public_parts = {
        part
        for row in selected
        for part in PurePosixPath(str(row["relative_path"])).parts
    }
    if set(aliases) & public_parts:
        raise ResilientAdapterError("DATA-001 manifest exposes source participant keys")
    collection_hash = _collection_sha256(collection_rows)
    if collection_hash != RESILIENT_COLLECTION_SHA256:
        raise ResilientAdapterError("RESILIENT collection SHA-256 does not match")
    return SourceBinding(
        workspace_root=root,
        source_root=source_root,
        manifests_root=audit_root,
        source_collection_sha256=collection_hash,
        file_manifest_sha256=DATA001_FILE_MANIFEST_SHA256,
        feature_schema_sha256=live_schema_hash,
        source_file_count=len(actual_files),
    )


def _target_specs(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in schema["group_order"]:
        for spec in schema["groups"][group]:
            rows.append({"group": group, **dict(spec)})
    if len(rows) != 54:
        raise ResilientAdapterError("MH-003 windowed feature count does not match")
    return rows


def _feature_column(group: str, name: str) -> str:
    return f"{group}.{name}"


def _mask_column(group: str, name: str) -> str:
    return f"feature_mask.{group}.{name}"


def _source_column(name: str) -> str:
    return f"source__{name}"


def build_field_mapping() -> dict[str, Any]:
    """Build the complete source and MH-003 target mapping contract."""

    schema = feature_schema_manifest()
    targets: list[dict[str, Any]] = []
    source_uses: dict[str, list[str]] = {name: [] for name in SOURCE_FIELDS}
    source_uses[X_SOURCE_FIELD].append("x_source_value")
    for target, rule in _DIRECT_TARGET_RULES.items():
        for source_field in rule["source_fields"]:
            source_uses[source_field].append(target)
    source_fields = [
        {
            "source_field": name,
            "canonical_source_column": _source_column(name),
            "arrow_type": str(SOURCE_FIELD_TYPES[name]),
            "canonical_uses": source_uses[name],
            "mapping_status": "used" if source_uses[name] else "audit_only",
        }
        for name in SOURCE_FIELDS
    ]
    for spec in _target_specs(schema):
        group = str(spec["group"])
        name = str(spec["name"])
        target = _feature_column(group, name)
        if target == ACTIVITY_VOLUME_COLUMN:
            mapping_status = "fold_derived"
            source_fields_for_target = [X_SOURCE_FIELD]
            formula = "right_continuous_ecdf(count(train <= x) / n)"
            mapping_kind = "source_relative_rank_only"
            base_policy = "null_with_mask_0_until_explicit_training_fold_fit"
        elif target in _DIRECT_TARGET_RULES:
            rule = _DIRECT_TARGET_RULES[target]
            mapping_status = "mapped"
            source_fields_for_target = list(rule["source_fields"])
            formula = str(rule["formula"])
            mapping_kind = str(rule["mapping_kind"])
            base_policy = "mapped_when_source_is_valid_else_null_with_mask_0"
        else:
            mapping_status = "unsupported"
            source_fields_for_target = []
            formula = None
            mapping_kind = "no_strict_same_semantics_source"
            base_policy = "null_with_mask_0"
        targets.append(
            {
                "group": group,
                "target_field": name,
                "canonical_value_column": target,
                "canonical_mask_column": _mask_column(group, name),
                "schema_spec": {
                    key: value for key, value in spec.items() if key != "group"
                },
                "mapping_status": mapping_status,
                "mapping_kind": mapping_kind,
                "source_fields": source_fields_for_target,
                "formula": formula,
                "base_canonical_policy": base_policy,
            }
        )
    return {
        "mapping_version": MAPPING_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "dataset_id": DATASET_ID,
        "feature_schema_version": schema["schema_version"],
        "feature_schema_sha256": MH003_FEATURE_SCHEMA_SHA256,
        "source_collection_sha256": RESILIENT_COLLECTION_SHA256,
        "file_manifest_sha256": DATA001_FILE_MANIFEST_SHA256,
        "timescale_semantics": TIMESCALE_SEMANTICS,
        "timezone_contract": {
            "source_timestamps": SOURCE_TIMESTAMP_TIMEZONE,
            "calendar_timezone": SOURCE_TIMEZONE,
            "reason": "publisher UTC timestamps converted to UK local calendar days",
        },
        "window_contract": {
            "start": "local calendar date of earliest valid row in any of four sensor files",
            "length_calendar_days": WINDOW_CALENDAR_DAYS,
            "end_inclusive": "start + 13 calendar days",
            "future_prediction_claim": False,
        },
        "labels": {
            "phq9_score": "publisher phq_total",
            "binary_target": "phq_total >= 10",
            "phq9_category": "0-4/5-9/10-14/15-19/20-27",
            "gds15_score": "publisher gds_total",
            "gds15_binary_target": "gds_total >= 5",
            "gad7_score": "publisher gad_total",
            "gad7_binary_target": "gad_total >= 10",
            "input_feature_use": "prohibited",
            "questionnaire_items_and_dates_as_model_features": False,
        },
        "sensor_contract": {
            "business_csvs": list(SENSOR_FILES),
            "os_metadata_parsed": False,
            "summary_role": "join and reconciliation only; never a feature source",
            "ScanWatch_Steps.csv": {
                "valid_row": "finite non-negative Steps with explicit UTC timestamp",
                "daily_total": "sum hourly step increments by Europe/London date",
                "valid_day": f"at least {MIN_STEP_HOURLY_BINS} distinct local hours",
            },
            "ScanWatch_HR.csv": {
                "valid_row": "finite positive Heart Rate with explicit UTC timestamp",
                "valid_day": "at least one valid row",
            },
            "Sleep_state.csv": {
                "valid_row": "end > start, explicit UTC timestamps, known sleep state",
                "episode": f"adjacent rows separated by at most {SLEEP_EPISODE_MAX_GAP_MINUTES:g} minutes",
                "main_sleep": (
                    "per local wake date choose greatest asleep minutes, then observed "
                    "minutes, then later end; require 60-960 observed minutes and sleep"
                ),
                "wakeup_semantics": "awake state",
            },
            "Sleep_physio.csv": {
                "deduplication": "per timestamp and field arithmetic mean",
                "night_alignment": "only rows inside selected main-sleep interval",
                "valid_night": "at least one aligned finite physiology value",
                "respiratory_abnormal_ratio": "unsupported: no frozen clinical range",
            },
        },
        "x_source": {
            "source_field": X_SOURCE_FIELD,
            "canonical_value_column": "x_source_value",
            "canonical_mask_column": "x_source_mask",
            "definition": "mean valid local-day step total in the fixed 14-day window",
            "production_camera_equivalence": False,
            "base_activity_volume_policy": "not_fitted",
        },
        "source_fields": source_fields,
        "target_fields": targets,
        "base_supported_feature_set": sorted(_DIRECT_TARGET_RULES),
        "fold_supported_feature_set": [ACTIVITY_VOLUME_COLUMN],
        "unsupported_feature_set": [
            row["canonical_value_column"]
            for row in targets
            if row["mapping_status"] == "unsupported"
        ],
        "target_leakage_exclusions": [
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
        ],
        "ecdf_contract": {
            "version": ECDF_VERSION,
            "fit_scope": "explicit outer-training participants only",
            "weighting_unit": "finite participant fixed-window value",
            "ties": "right_continuous",
            "formula": "count(training_values <= x) / finite_training_value_count",
            "formal_instances_before_data007": False,
            "required_binding": [
                "adapter_version",
                "dataset_id",
                "source_collection_sha256",
                "feature_schema_sha256",
                "source_field",
                "split_id",
                "canonical_artifact_sha256",
                "canonical_frame_sha256",
                "training_participant_sha256",
            ],
        },
    }


def _require_exact_columns(
    frame: pd.DataFrame, expected: Sequence[str], *, table_name: str
) -> None:
    if tuple(frame.columns) != tuple(expected):
        raise ResilientAdapterError(f"{table_name} columns do not match the contract")


def _parse_utc(series: pd.Series, *, field_name: str) -> pd.Series:
    raw = series.astype("string")
    present = raw.notna() & raw.str.strip().ne("")
    if (
        not bool(present.all())
        or not raw[present].str.contains(_TIMESTAMP_OFFSET, regex=True).all()
    ):
        raise ResilientAdapterError(f"{field_name} must contain explicit UTC offsets")
    try:
        parsed = pd.to_datetime(raw, errors="raise", utc=True)
    except (TypeError, ValueError) as exc:
        raise ResilientAdapterError(f"{field_name} is not valid UTC time") from exc
    return parsed


def _numeric(
    series: pd.Series,
    *,
    field_name: str,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise").astype("Float64")
    except (TypeError, ValueError) as exc:
        raise ResilientAdapterError(f"{field_name} is not numeric") from exc
    values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    if np.isinf(values).any():
        raise ResilientAdapterError(f"{field_name} contains infinity")
    finite = values[~np.isnan(values)]
    if minimum is not None:
        invalid = finite <= minimum if strict_minimum else finite < minimum
        if invalid.any():
            raise ResilientAdapterError(f"{field_name} is outside its valid range")
    return numeric


def _read_demographics(source_root: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            source_root / DEMOGRAPHICS_NAME,
            dtype={"user_id": "string"},
            keep_default_na=True,
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ResilientAdapterError("RESILIENT demographics are not readable") from exc
    if len(frame) != EXPECTED_PARTICIPANTS or len(frame.columns) != 231:
        raise ResilientAdapterError("RESILIENT demographics shape does not match")
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
    if not required.issubset(frame.columns):
        raise ResilientAdapterError("RESILIENT demographics fields are missing")
    frame["user_id"] = frame["user_id"].str.strip()
    if frame["user_id"].isna().any() or frame["user_id"].eq("").any():
        raise ResilientAdapterError("RESILIENT user_id is missing")
    if not frame["user_id"].is_unique:
        raise ResilientAdapterError("RESILIENT user_id must be unique")
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
    try:
        frame = pd.read_csv(
            source_root / SUMMARY_NAME,
            dtype={"participant": "string", "file": "string"},
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ResilientAdapterError("RESILIENT summary is not readable") from exc
    _require_exact_columns(frame, expected, table_name="summary")
    if len(frame) != EXPECTED_SUMMARY_ROWS:
        raise ResilientAdapterError("RESILIENT summary row count does not match")
    frame["participant"] = frame["participant"].str.strip()
    frame["file"] = frame["file"].str.strip()
    if frame[["participant", "file"]].isna().any(axis=None):
        raise ResilientAdapterError("RESILIENT summary key is missing")
    if frame.duplicated(["participant", "file"]).any():
        raise ResilientAdapterError("RESILIENT summary key is not unique")
    if set(frame["file"]) != set(SENSOR_FILES):
        raise ResilientAdapterError("RESILIENT summary sensor set does not match")
    return frame


def _read_sensor_frame(directory: Path, file_name: str) -> pd.DataFrame:
    if file_name not in SENSOR_COLUMNS:
        raise ResilientAdapterError("unknown RESILIENT sensor file")
    path = directory / file_name
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ResilientAdapterError(f"RESILIENT {file_name} is not readable") from exc
    _require_exact_columns(frame, SENSOR_COLUMNS[file_name], table_name=file_name)
    return frame


def _reconcile_summary_row(
    frame: pd.DataFrame,
    file_name: str,
    summary_row: pd.Series,
) -> None:
    if int(summary_row["total_records"]) != len(frame):
        raise ResilientAdapterError("RESILIENT summary record count does not match")
    if frame.empty:
        if int(summary_row["num_days"]) != 0:
            raise ResilientAdapterError("RESILIENT empty sensor summary does not match")
        return
    if file_name == "Sleep_state.csv":
        timestamp = _parse_utc(frame["Start time"], field_name="Start time")
        latest_timestamp = _parse_utc(frame["End time"], field_name="End time")
    else:
        timestamp_field = {
            "ScanWatch_HR.csv": "HR Timestamp",
            "ScanWatch_Steps.csv": "Steps Timestamp",
            "Sleep_physio.csv": "Timestamp",
        }[file_name]
        timestamp = _parse_utc(frame[timestamp_field], field_name=timestamp_field)
        latest_timestamp = timestamp
    if int(summary_row["num_days"]) != timestamp.dt.date.nunique():
        raise ResilientAdapterError("RESILIENT summary UTC day count does not match")
    expected_earliest = pd.to_datetime(summary_row["earliest_date"], errors="raise")
    expected_latest = pd.to_datetime(summary_row["latest_date"], errors="raise")
    if timestamp.min().date() != expected_earliest.date():
        raise ResilientAdapterError("RESILIENT summary earliest date does not match")
    if latest_timestamp.max().date() != expected_latest.date():
        raise ResilientAdapterError("RESILIENT summary latest date does not match")
    expected_average = round(len(frame) / timestamp.dt.date.nunique(), 2)
    if not math.isclose(
        float(summary_row["avg_records_per_day"]),
        expected_average,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ResilientAdapterError("RESILIENT summary average does not match")


def _date_in_window(value: date, start: date, end: date) -> bool:
    return start <= value <= end


def _minute_of_day(timestamp: pd.Timestamp) -> float:
    local = timestamp.tz_convert(SOURCE_TIMEZONE)
    return (
        float(local.hour * 60 + local.minute)
        + local.second / 60.0
        + local.microsecond / 60_000_000.0
    )


def _circular_mean_components(
    minutes: Sequence[float],
) -> tuple[float | None, float | None]:
    if not minutes:
        return None, None
    angles = [2.0 * math.pi * value / 1440.0 for value in minutes]
    mean_sin = fmean(math.sin(angle) for angle in angles)
    mean_cos = fmean(math.cos(angle) for angle in angles)
    length = math.hypot(mean_sin, mean_cos)
    if length <= _EPSILON:
        return None, None
    return mean_sin / length, mean_cos / length


def _circular_regularity(minutes: Sequence[float]) -> float | None:
    if len(minutes) < 2:
        return None
    angles = [2.0 * math.pi * value / 1440.0 for value in minutes]
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
    if abs(mean_value) <= _EPSILON:
        return 0.0 if all(abs(value) <= _EPSILON for value in values) else None
    return pstdev(values) / abs(mean_value)


def _circular_window_means(
    hourly_profile: Mapping[int, float], size: int
) -> list[float]:
    means: list[float] = []
    for start in range(24):
        hours = tuple((start + offset) % 24 for offset in range(size))
        if all(hour in hourly_profile for hour in hours):
            means.append(fmean(hourly_profile[hour] for hour in hours))
    return means


def _relative_amplitude(hourly_profile: Mapping[int, float]) -> float | None:
    ten_hour_means = _circular_window_means(hourly_profile, 10)
    five_hour_means = _circular_window_means(hourly_profile, 5)
    if not ten_hour_means or not five_hour_means:
        return None
    most_active = max(ten_hour_means)
    least_active = min(five_hour_means)
    denominator = most_active + least_active
    if denominator <= _EPSILON:
        return 0.0
    return min(max((most_active - least_active) / denominator, 0.0), 1.0)


def _interdaily_stability(cells: pd.DataFrame) -> float | None:
    if cells.empty or cells["date"].nunique() < 2 or cells["hour"].nunique() < 2:
        return None
    values = cells["steps"].to_numpy(dtype="float64")
    overall = float(np.mean(values))
    total_variation = float(np.sum((values - overall) ** 2))
    if total_variation <= _EPSILON:
        return 1.0
    profile = cells.groupby("hour", sort=True)["steps"].mean()
    between = math.fsum(
        (float(profile.loc[int(row.hour)]) - overall) ** 2
        for row in cells.itertuples(index=False)
    )
    return min(max(between / total_variation, 0.0), 1.0)


def _intradaily_variability(cells: pd.DataFrame) -> float | None:
    if len(cells) < 2:
        return None
    values = cells["steps"].to_numpy(dtype="float64")
    variance = float(np.mean((values - float(np.mean(values))) ** 2))
    squared_differences: list[float] = []
    for _, day_frame in cells.groupby("date", sort=False):
        by_hour = day_frame.set_index("hour")["steps"]
        for hour in sorted(int(value) for value in by_hour.index):
            if hour + 1 in by_hour.index:
                squared_differences.append(
                    (float(by_hour.loc[hour + 1]) - float(by_hour.loc[hour])) ** 2
                )
    if not squared_differences:
        return None
    if variance <= _EPSILON:
        return 0.0
    return fmean(squared_differences) / variance


def _aggregate_steps(
    frame: pd.DataFrame, start: date, end: date
) -> tuple[dict[str, Any], dict[str, Any], pd.Timestamp | None]:
    if frame.empty:
        raise ResilientAdapterError("ScanWatch_Steps.csv must not be empty")
    timestamps = _parse_utc(frame["Steps Timestamp"], field_name="Steps Timestamp")
    if timestamps.duplicated().any():
        raise ResilientAdapterError("ScanWatch step timestamps must be unique")
    steps = _numeric(frame["Steps"], field_name="Steps", minimum=0.0)
    if steps.isna().any():
        raise ResilientAdapterError("ScanWatch steps contain missing values")
    local = timestamps.dt.tz_convert(SOURCE_TIMEZONE)
    observed = pd.DataFrame(
        {
            "timestamp": timestamps,
            "date": local.dt.date,
            "hour": local.dt.hour.astype("int16"),
            "steps": steps.astype("float64"),
        }
    )
    window = observed[
        observed["date"].map(lambda value: _date_in_window(value, start, end))
    ].copy()
    daily_totals: list[float] = []
    valid_cells: list[pd.DataFrame] = []
    valid_dates: list[date] = []
    for day, day_frame in window.groupby("date", sort=True):
        hourly = day_frame.groupby("hour", sort=True)["steps"].sum()
        if len(hourly) < MIN_STEP_HOURLY_BINS:
            continue
        valid_dates.append(day)
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
        "scanwatch_steps_valid_days": len(valid_dates),
        "scanwatch_relative_amplitude": _relative_amplitude(profile),
        "scanwatch_interdaily_stability": _interdaily_stability(cells),
        "scanwatch_intradaily_variability": _intradaily_variability(cells),
        "scanwatch_activity_variability": _coefficient_of_variation(daily_totals),
    }
    quality = {
        "source_rows": len(frame),
        "window_rows": len(window),
        "window_observed_days": int(window["date"].nunique()),
        "valid_days": len(valid_dates),
        "minimum_distinct_hours_per_valid_day": MIN_STEP_HOURLY_BINS,
    }
    return values, quality, timestamps.min()


def _aggregate_scanwatch_hr(
    frame: pd.DataFrame, start: date, end: date
) -> tuple[dict[str, Any], dict[str, Any], pd.Timestamp | None]:
    timestamps = _parse_utc(frame["HR Timestamp"], field_name="HR Timestamp")
    if timestamps.duplicated().any():
        raise ResilientAdapterError("ScanWatch HR timestamps must be unique")
    heart_rate = _numeric(
        frame["Heart Rate"],
        field_name="ScanWatch Heart Rate",
        minimum=0.0,
        strict_minimum=True,
    )
    if heart_rate.isna().any():
        raise ResilientAdapterError("ScanWatch HR contains missing values")
    local_dates = timestamps.dt.tz_convert(SOURCE_TIMEZONE).dt.date
    observed = pd.DataFrame({"date": local_dates, "heart_rate": heart_rate})
    window = observed[
        observed["date"].map(lambda value: _date_in_window(value, start, end))
    ]
    daily = window.groupby("date", sort=True)["heart_rate"]
    daily_means = [float(value) for value in daily.mean().tolist()]
    daily_mins = [float(value) for value in daily.min().tolist()]
    daily_sds = [
        float(values.to_numpy(dtype="float64").std(ddof=0)) for _, values in daily
    ]
    values = {
        "scanwatch_hr_mean_bpm": fmean(daily_means) if daily_means else None,
        "scanwatch_hr_min_bpm": fmean(daily_mins) if daily_mins else None,
        "scanwatch_hr_sd_bpm": fmean(daily_sds) if daily_sds else None,
        "scanwatch_hr_valid_days": len(daily_means),
    }
    quality = {
        "source_rows": len(frame),
        "window_rows": len(window),
        "valid_days": len(daily_means),
    }
    return values, quality, (timestamps.min() if len(timestamps) else None)


def _episode_from_rows(rows: pd.DataFrame) -> SleepEpisode | None:
    effective_rows: list[dict[str, Any]] = []
    covered_until: pd.Timestamp | None = None
    for row in rows.itertuples(index=False):
        effective_start = (
            row.start
            if covered_until is None or row.start >= covered_until
            else covered_until
        )
        if row.end > effective_start:
            effective_rows.append(
                {"start": effective_start, "end": row.end, "state": row.state}
            )
        covered_until = (
            row.end if covered_until is None else max(covered_until, row.end)
        )
    if not effective_rows:
        return None
    effective = pd.DataFrame(effective_rows)
    start = effective["start"].min()
    end = effective["end"].max()
    span_minutes = (end - start).total_seconds() / 60.0
    durations = (effective["end"] - effective["start"]).dt.total_seconds() / 60.0
    in_bed = float(durations.sum())
    asleep_mask = effective["state"].ne("wakeup")
    asleep = float(durations[asleep_mask].sum())
    awake = float(durations[~asleep_mask].sum())
    if (
        in_bed < MIN_MAIN_SLEEP_MINUTES
        or in_bed > MAX_MAIN_SLEEP_MINUTES
        or span_minutes > MAX_MAIN_SLEEP_MINUTES
        or asleep <= 0.0
    ):
        return None
    wake_date = end.tz_convert(SOURCE_TIMEZONE).date()
    onset_timestamp = effective.loc[asleep_mask, "start"].min()
    onset_minute = _minute_of_day(onset_timestamp)
    wake_minute = _minute_of_day(end)
    forward = (wake_minute - onset_minute) % 1440.0
    midpoint = (onset_minute + forward / 2.0) % 1440.0
    states = effective["state"].tolist()
    awakening_count = sum(
        current == "wakeup" and previous != "wakeup"
        for previous, current in zip(states, states[1:], strict=False)
    )
    return SleepEpisode(
        start=start,
        end=end,
        wake_date=wake_date,
        in_bed_minutes=in_bed,
        asleep_minutes=asleep,
        awake_minutes=awake,
        onset_minute=onset_minute,
        wake_minute=wake_minute,
        midpoint_minute=midpoint,
        awakening_count=awakening_count,
    )


def _main_sleep_episodes(
    frame: pd.DataFrame,
) -> tuple[list[SleepEpisode], dict[str, Any]]:
    starts = _parse_utc(frame["Start time"], field_name="Start time")
    ends = _parse_utc(frame["End time"], field_name="End time")
    states = frame["Sleep state"].astype("string").str.strip()
    if states.isna().any() or not set(states).issubset(SLEEP_STATES):
        raise ResilientAdapterError("Sleep_state contains an unsupported state")
    if not bool((ends > starts).all()):
        raise ResilientAdapterError("Sleep_state intervals must have positive duration")
    records = pd.DataFrame({"start": starts, "end": ends, "state": states}).sort_values(
        ["start", "end", "state"], kind="stable"
    )
    if records.duplicated().any():
        raise ResilientAdapterError("Sleep_state contains duplicate intervals")
    groups: list[pd.DataFrame] = []
    group_start = 0
    running_end: pd.Timestamp | None = None
    for position, row in enumerate(records.itertuples(index=False)):
        gap = (
            None
            if running_end is None
            else (row.start - running_end).total_seconds() / 60.0
        )
        if (
            position > group_start
            and gap is not None
            and gap > SLEEP_EPISODE_MAX_GAP_MINUTES
        ):
            groups.append(records.iloc[group_start:position])
            group_start = position
            running_end = None
        running_end = row.end if running_end is None else max(running_end, row.end)
    if len(records):
        groups.append(records.iloc[group_start:])
    candidates = [episode for group in groups if (episode := _episode_from_rows(group))]
    main_by_date: dict[date, SleepEpisode] = {}
    for episode in candidates:
        existing = main_by_date.get(episode.wake_date)
        if existing is None or (
            episode.asleep_minutes,
            episode.in_bed_minutes,
            episode.end,
        ) > (
            existing.asleep_minutes,
            existing.in_bed_minutes,
            existing.end,
        ):
            main_by_date[episode.wake_date] = episode
    main = [main_by_date[key] for key in sorted(main_by_date)]
    return main, {
        "source_rows": len(frame),
        "episode_count_before_main_selection": len(candidates),
        "main_sleep_night_count": len(main),
        "discarded_invalid_episode_count": len(groups) - len(candidates),
    }


def _aggregate_sleep_state(
    frame: pd.DataFrame, start: date, end: date
) -> tuple[dict[str, Any], dict[str, Any], list[SleepEpisode], pd.Timestamp | None]:
    episodes, quality = _main_sleep_episodes(frame)
    window = [
        episode
        for episode in episodes
        if _date_in_window(episode.wake_date, start, end)
    ]
    durations = [episode.asleep_minutes for episode in window]
    in_bed = [episode.in_bed_minutes for episode in window]
    onset = [episode.onset_minute for episode in window]
    wake = [episode.wake_minute for episode in window]
    midpoint = [episode.midpoint_minute for episode in window]
    onset_sin, onset_cos = _circular_mean_components(onset)
    wake_sin, wake_cos = _circular_mean_components(wake)
    midpoint_sin, midpoint_cos = _circular_mean_components(midpoint)
    regularity_parts = [
        value
        for values in (onset, wake, midpoint)
        if (value := _circular_regularity(values)) is not None
    ]
    fragmentations = [
        episode.awake_minutes / episode.in_bed_minutes
        + episode.awakening_count / (episode.asleep_minutes / 60.0)
        for episode in window
    ]
    values = {
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
        "sleep_fragmentation": (fmean(fragmentations) if fragmentations else None),
        "sleep_regularity": (fmean(regularity_parts) if regularity_parts else None),
        "sleep_awakening_count_mean": (
            fmean(episode.awakening_count for episode in window) if window else None
        ),
    }
    starts = _parse_utc(frame["Start time"], field_name="Start time")
    return (
        values,
        {**quality, "window_main_sleep_night_count": len(window)},
        window,
        starts.min() if len(starts) else None,
    )


def _assign_episode_indices(
    timestamps: pd.Series, episodes: Sequence[SleepEpisode]
) -> np.ndarray:
    output = np.full(len(timestamps), -1, dtype=np.int32)
    if not episodes or timestamps.empty:
        return output
    ordered = sorted(enumerate(episodes), key=lambda item: item[1].start)
    starts = np.asarray(
        [
            item[1].start.to_datetime64().astype("datetime64[ns]").astype(np.int64)
            for item in ordered
        ],
        dtype=np.int64,
    )
    ends = np.asarray(
        [
            item[1].end.to_datetime64().astype("datetime64[ns]").astype(np.int64)
            for item in ordered
        ],
        dtype=np.int64,
    )
    values = timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    positions = np.searchsorted(starts, values, side="right") - 1
    candidate = positions >= 0
    safe_positions = np.maximum(positions, 0)
    candidate &= values <= ends[safe_positions]
    original_indices = np.asarray([item[0] for item in ordered], dtype=np.int32)
    output[candidate] = original_indices[safe_positions[candidate]]
    return output


def _field_nightly_aggregate(
    grouped: pd.core.groupby.DataFrameGroupBy,
    field: str,
    operation: str,
) -> list[float]:
    values: list[float] = []
    for _, night in grouped:
        finite = night[field].dropna().to_numpy(dtype="float64")
        if not len(finite):
            continue
        if operation == "mean":
            values.append(float(np.mean(finite)))
        elif operation == "min":
            values.append(float(np.min(finite)))
        elif operation == "sd":
            values.append(float(np.std(finite, ddof=0)))
        elif operation == "sum":
            values.append(float(np.sum(finite)))
        else:
            raise ResilientAdapterError("unknown physiology aggregation")
    return values


def _aggregate_sleep_physio(
    frame: pd.DataFrame,
    episodes: Sequence[SleepEpisode],
) -> tuple[dict[str, Any], dict[str, Any], pd.Timestamp | None]:
    if frame.empty:
        values = {
            "sleep_physio_valid_nights": 0,
            "sleep_hr_mean_bpm": None,
            "sleep_hr_min_bpm": None,
            "sleep_hr_sd_bpm": None,
            "sleep_hrv_sdnn_ms": None,
            "sleep_respiration_rate_mean_bpm": None,
            "sleep_respiration_rate_sd_bpm": None,
            "sleep_snoring_minutes_mean": None,
        }
        return (
            values,
            {
                "source_rows": 0,
                "unique_timestamp_rows": 0,
                "duplicate_timestamp_rows_removed": 0,
                "aligned_rows": 0,
                "valid_nights": 0,
            },
            None,
        )
    timestamps = _parse_utc(frame["Timestamp"], field_name="Timestamp")
    numeric = pd.DataFrame(
        {
            "heart_rate": _numeric(
                frame["Heart Rate"],
                field_name="Sleep Heart Rate",
                minimum=0.0,
                strict_minimum=True,
            ),
            "respiration_rate": _numeric(
                frame["Respiration Rate"],
                field_name="Respiration Rate",
                minimum=0.0,
                strict_minimum=True,
            ),
            "snoring_seconds": _numeric(
                frame["Snoring"], field_name="Snoring", minimum=0.0
            ),
            "sdnn_ms": _numeric(frame["SDNN_1"], field_name="SDNN_1", minimum=0.0),
        }
    )
    numeric.insert(0, "timestamp", timestamps)
    deduplicated = numeric.groupby("timestamp", sort=True, as_index=False).mean()
    episode_index = _assign_episode_indices(deduplicated["timestamp"], episodes)
    aligned = deduplicated[episode_index >= 0].copy()
    aligned["episode_index"] = episode_index[episode_index >= 0]
    grouped = aligned.groupby("episode_index", sort=True)
    heart_mean = _field_nightly_aggregate(grouped, "heart_rate", "mean")
    heart_min = _field_nightly_aggregate(grouped, "heart_rate", "min")
    heart_sd = _field_nightly_aggregate(grouped, "heart_rate", "sd")
    sdnn = _field_nightly_aggregate(grouped, "sdnn_ms", "mean")
    respiration_mean = _field_nightly_aggregate(grouped, "respiration_rate", "mean")
    respiration_sd = _field_nightly_aggregate(grouped, "respiration_rate", "sd")
    snoring_seconds = _field_nightly_aggregate(grouped, "snoring_seconds", "sum")
    valid_nights = int(aligned["episode_index"].nunique())
    values = {
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
            fmean(value / 60.0 for value in snoring_seconds)
            if snoring_seconds
            else None
        ),
    }
    quality = {
        "source_rows": len(frame),
        "unique_timestamp_rows": len(deduplicated),
        "duplicate_timestamp_rows_removed": len(frame) - len(deduplicated),
        "aligned_rows": len(aligned),
        "valid_nights": valid_nights,
    }
    return values, quality, timestamps.min()


def _first_valid_sensor_instant(frames: Mapping[str, pd.DataFrame]) -> pd.Timestamp:
    candidates: list[pd.Timestamp] = []
    hr = frames["ScanWatch_HR.csv"]
    if not hr.empty:
        timestamp = _parse_utc(hr["HR Timestamp"], field_name="HR Timestamp")
        values = _numeric(
            hr["Heart Rate"],
            field_name="ScanWatch Heart Rate",
            minimum=0.0,
            strict_minimum=True,
        )
        if values.notna().any():
            candidates.append(timestamp[values.notna()].min())
    steps = frames["ScanWatch_Steps.csv"]
    if not steps.empty:
        timestamp = _parse_utc(steps["Steps Timestamp"], field_name="Steps Timestamp")
        values = _numeric(steps["Steps"], field_name="Steps", minimum=0.0)
        if values.notna().any():
            candidates.append(timestamp[values.notna()].min())
    physio = frames["Sleep_physio.csv"]
    if not physio.empty:
        timestamp = _parse_utc(physio["Timestamp"], field_name="Timestamp")
        any_value = pd.concat(
            [
                _numeric(physio[field], field_name=field, minimum=0.0).notna()
                for field in ("Heart Rate", "Respiration Rate", "Snoring", "SDNN_1")
            ],
            axis=1,
        ).any(axis=1)
        if any_value.any():
            candidates.append(timestamp[any_value].min())
    state = frames["Sleep_state.csv"]
    if not state.empty:
        starts = _parse_utc(state["Start time"], field_name="Start time")
        ends = _parse_utc(state["End time"], field_name="End time")
        states = state["Sleep state"].astype("string").str.strip()
        valid = (ends > starts) & states.isin(SLEEP_STATES)
        if valid.any():
            candidates.append(starts[valid].min())
    if not candidates:
        raise ResilientAdapterError("participant has no valid sensor timestamp")
    return min(candidates)


def aggregate_sensor_window(
    frames: Mapping[str, pd.DataFrame],
) -> AggregatedSensorWindow:
    """Aggregate one participant's four real sensor files into 14 local days."""

    if set(frames) != set(SENSOR_FILES):
        raise ResilientAdapterError("sensor frame set does not contain four files")
    for file_name in SENSOR_FILES:
        _require_exact_columns(
            frames[file_name], SENSOR_COLUMNS[file_name], table_name=file_name
        )
    first = _first_valid_sensor_instant(frames)
    start = first.tz_convert(SOURCE_TIMEZONE).date()
    end = start + timedelta(days=WINDOW_CALENDAR_DAYS - 1)
    steps_values, steps_quality, _ = _aggregate_steps(
        frames["ScanWatch_Steps.csv"], start, end
    )
    scan_hr_values, scan_hr_quality, _ = _aggregate_scanwatch_hr(
        frames["ScanWatch_HR.csv"], start, end
    )
    sleep_values, sleep_quality, episodes, _ = _aggregate_sleep_state(
        frames["Sleep_state.csv"], start, end
    )
    physio_values, physio_quality, _ = _aggregate_sleep_physio(
        frames["Sleep_physio.csv"], episodes
    )
    values = {
        **steps_values,
        **scan_hr_values,
        **sleep_values,
        **physio_values,
    }
    quality = {
        "window_start_source": "earliest valid timestamp across four sensor files",
        "window_calendar_timezone": SOURCE_TIMEZONE,
        "window_calendar_days": WINDOW_CALENDAR_DAYS,
        "ScanWatch_Steps.csv": steps_quality,
        "ScanWatch_HR.csv": scan_hr_quality,
        "Sleep_state.csv": sleep_quality,
        "Sleep_physio.csv": physio_quality,
    }
    return AggregatedSensorWindow(
        window_start_date=start,
        window_end_date=end,
        source_values=values,
        quality=quality,
    )


def _empty_target_series(length: int, value_type: str) -> pd.Series:
    if value_type == "float":
        return pd.Series(pd.array([pd.NA] * length, dtype="Float64"))
    if value_type == "integer":
        return pd.Series(pd.array([pd.NA] * length, dtype="Int64"))
    if value_type == "category":
        return pd.Series(pd.array([pd.NA] * length, dtype="string"))
    raise ResilientAdapterError("MH-003 feature type is unsupported")


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


def _parse_label_score(value: Any, *, name: str, maximum: int) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ResilientAdapterError(f"RESILIENT {name} label is invalid") from exc
    if not math.isfinite(numeric) or numeric != math.floor(numeric):
        raise ResilientAdapterError(f"RESILIENT {name} label is invalid")
    score = int(numeric)
    if score < 0 or score > maximum:
        raise ResilientAdapterError(f"RESILIENT {name} label is outside range")
    return score


def _parse_assessment_date(
    value: Any, *, name: str, allow_missing: bool
) -> date | None:
    if pd.isna(value) or str(value).strip() == "":
        if allow_missing:
            return None
        raise ResilientAdapterError(f"RESILIENT {name} date is missing")
    try:
        parsed = pd.to_datetime(str(value), format="%d/%m/%Y", errors="raise")
    except (TypeError, ValueError) as exc:
        raise ResilientAdapterError(f"RESILIENT {name} date is invalid") from exc
    return parsed.date()


def _parse_profile_bool(value: Any) -> int | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    normalized = str(value).strip().casefold()
    if normalized == "true":
        return 1
    if normalized == "false":
        return 0
    raise ResilientAdapterError("RESILIENT chronic condition encoding is invalid")


def _profile_values(row: pd.Series) -> dict[str, Any]:
    raw_sex = None if pd.isna(row["Sex"]) else str(row["Sex"]).strip()
    if raw_sex not in {None, "Female", "Male"}:
        raise ResilientAdapterError("RESILIENT Sex encoding is invalid")
    sex = {None: None, "Female": "female", "Male": "male"}[raw_sex]
    raw_age = None if pd.isna(row["Age group"]) else str(row["Age group"]).strip()
    allowed_age = {None, "[72, 75]", "[76, 87]", "[88, 99]"}
    if raw_age not in allowed_age:
        raise ResilientAdapterError("RESILIENT Age group encoding is invalid")
    age_group = {
        None: None,
        "[72, 75]": "70_79",
        "[76, 87]": None,
        "[88, 99]": "80_plus",
    }[raw_age]
    return {
        "profile_sex": sex,
        "profile_age_group": age_group,
        "profile_essential_hypertension": _parse_profile_bool(
            row["Essential hypertension"]
        ),
        "profile_osteoarthritis": _parse_profile_bool(row["Osteoarthritis"]),
    }


def _target_value_from_source(source_values: Mapping[str, Any], target: str) -> Any:
    source_name = str(_DIRECT_TARGET_RULES[target]["source_fields"][0])
    value = source_values[source_name]
    if value is None or pd.isna(value):
        return None
    if target in {
        "sleep.sleep_duration_norm",
        "sleep.time_in_bed_norm",
        "physiology.snoring_minutes_norm",
    }:
        result = float(value) / 1440.0
        if result < 0.0 or result > 1.0:
            raise ResilientAdapterError("duration-derived target is outside MH-003")
        return result
    return value


def canonical_column_order() -> tuple[str, ...]:
    target_specs = _target_specs(feature_schema_manifest())
    features = tuple(
        _feature_column(str(spec["group"]), str(spec["name"])) for spec in target_specs
    )
    masks = tuple(
        _mask_column(str(spec["group"]), str(spec["name"])) for spec in target_specs
    )
    return (
        *IDENTITY_COLUMNS,
        *TARGET_COLUMNS,
        *X_SOURCE_COLUMNS,
        *(_source_column(name) for name in SOURCE_FIELDS),
        *features,
        *masks,
    )


def canonical_arrow_schema() -> pa.Schema:
    field_types: dict[str, pa.DataType] = {
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
    field_types.update(
        {_source_column(name): value for name, value in SOURCE_FIELD_TYPES.items()}
    )
    value_types = {
        "float": pa.float64(),
        "integer": pa.int64(),
        "category": pa.large_string(),
    }
    for spec in _target_specs(feature_schema_manifest()):
        target = _feature_column(str(spec["group"]), str(spec["name"]))
        try:
            field_types[target] = value_types[str(spec["value_type"])]
        except KeyError as exc:
            raise ResilientAdapterError(
                "unsupported MH-003 Arrow feature type"
            ) from exc
        field_types[f"feature_mask.{target}"] = pa.int8()
    order = canonical_column_order()
    if set(field_types) != set(order):
        raise ResilientAdapterError("canonical Arrow schema columns do not match")
    return pa.schema([pa.field(name, field_types[name]) for name in order])


def _canonical_arrow_table(frame: pd.DataFrame) -> pa.Table:
    if tuple(frame.columns) != canonical_column_order():
        raise ResilientAdapterError("canonical frame columns do not match")
    schema = canonical_arrow_schema()
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
        return pa.Table.from_arrays(arrays, schema=schema).combine_chunks()
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise ResilientAdapterError(
            "canonical values do not match the RESILIENT Arrow schema"
        ) from exc


def canonical_frame_sha256(frame: pd.DataFrame) -> str:
    table = _canonical_arrow_table(frame)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table, max_chunksize=max(len(table), 1))
    return _sha256_bytes(sink.getvalue().to_pybytes())


def build_canonical_frame(
    demographics: pd.DataFrame,
    sensor_windows: Mapping[str, AggregatedSensorWindow],
    *,
    enforce_frozen_statistics: bool = False,
) -> pd.DataFrame:
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
    if not required.issubset(demographics.columns):
        raise ResilientAdapterError("demographics are missing required fields")
    participants = demographics["user_id"].astype("string").str.strip()
    if participants.isna().any() or not participants.is_unique:
        raise ResilientAdapterError("canonical demographics user_id is invalid")
    if set(participants.astype(str)) != set(sensor_windows):
        raise ResilientAdapterError("demographics and sensor window keys differ")

    rows: list[dict[str, Any]] = []
    target_specs = _target_specs(feature_schema_manifest())
    for position, (_, source_row) in enumerate(demographics.iterrows()):
        participant_id = str(participants.iloc[position])
        window = sensor_windows[participant_id]
        phq = _parse_label_score(source_row["phq_total"], name="PHQ-9", maximum=27)
        gds = _parse_label_score(source_row["gds_total"], name="GDS-15", maximum=15)
        gad = _parse_label_score(source_row["gad_total"], name="GAD-7", maximum=21)
        source_values = {**_profile_values(source_row), **dict(window.source_values)}
        if set(source_values) != set(SOURCE_FIELDS):
            raise ResilientAdapterError(
                "aggregated RESILIENT source fields do not match"
            )
        row: dict[str, Any] = {
            "dataset_id": DATASET_ID,
            "global_participant_id": f"{DATASET_ID}::{participant_id}",
            "participant_id": participant_id,
            "timescale_semantics": TIMESCALE_SEMANTICS,
            "source_timezone": SOURCE_TIMEZONE,
            "window_start_date": window.window_start_date,
            "window_end_date": window.window_end_date,
            "window_calendar_days": WINDOW_CALENDAR_DAYS,
            "phq_assessment_date": _parse_assessment_date(
                source_row["phq_date"], name="PHQ-9", allow_missing=False
            ),
            "phq9_score": phq,
            "phq9_category": _phq_category(phq),
            "binary_target": int(phq >= 10),
            "gds_assessment_date": _parse_assessment_date(
                source_row["gds_date"], name="GDS-15", allow_missing=False
            ),
            "gds15_score": gds,
            "gds15_binary_target": int(gds >= 5),
            "gad_assessment_date": _parse_assessment_date(
                source_row["gad_date"], name="GAD-7", allow_missing=True
            ),
            "gad7_score": gad,
            "gad7_binary_target": int(gad >= 10),
            "x_source_name": X_SOURCE_FIELD,
            "x_source_value": source_values[X_SOURCE_FIELD],
            "x_source_mask": int(source_values[X_SOURCE_FIELD] is not None),
        }
        row.update(
            {_source_column(name): source_values[name] for name in SOURCE_FIELDS}
        )
        for spec in target_specs:
            target = _feature_column(str(spec["group"]), str(spec["name"]))
            if target in _DIRECT_TARGET_RULES:
                value = _target_value_from_source(source_values, target)
            else:
                value = None
            row[target] = value
            row[f"feature_mask.{target}"] = int(value is not None)
        rows.append(row)
    canonical = pd.DataFrame(rows)
    canonical = canonical.sort_values("participant_id", kind="stable").reset_index(
        drop=True
    )
    for spec in target_specs:
        target = _feature_column(str(spec["group"]), str(spec["name"]))
        if str(spec["value_type"]) == "float":
            canonical[target] = pd.array(canonical[target], dtype="Float64")
        elif str(spec["value_type"]) == "integer":
            canonical[target] = pd.array(canonical[target], dtype="Int64")
        else:
            canonical[target] = pd.array(canonical[target], dtype="string")
        canonical[f"feature_mask.{target}"] = pd.array(
            canonical[f"feature_mask.{target}"], dtype="Int8"
        )
    canonical = canonical.loc[:, list(canonical_column_order())]
    if canonical[ACTIVITY_VOLUME_COLUMN].notna().any() or int(
        canonical[ACTIVITY_VOLUME_MASK_COLUMN].sum()
    ):
        raise ResilientAdapterError("base canonical table must not contain ECDF")
    if enforce_frozen_statistics:
        _validate_frozen_statistics(canonical)
    _canonical_arrow_table(canonical)
    return canonical


def _validate_frozen_statistics(canonical: pd.DataFrame) -> None:
    observed_categories = {
        int(key): int(value)
        for key, value in canonical["phq9_category"].value_counts().items()
    }
    if (
        len(canonical) != EXPECTED_PARTICIPANTS
        or canonical["participant_id"].nunique() != EXPECTED_PARTICIPANTS
        or int(canonical["binary_target"].sum()) != EXPECTED_POSITIVES
        or int(canonical["gds15_binary_target"].sum()) != EXPECTED_GDS_POSITIVES
        or int(canonical["gad7_binary_target"].sum()) != EXPECTED_GAD_POSITIVES
        or observed_categories != EXPECTED_PHQ_CATEGORY_COUNTS
    ):
        raise ResilientAdapterError(
            "RESILIENT frozen aggregate statistics do not match"
        )


def _load_and_aggregate_sources(
    binding: SourceBinding,
) -> tuple[pd.DataFrame, dict[str, AggregatedSensorWindow], list[dict[str, Any]]]:
    demographics = _read_demographics(binding.source_root)
    summary = _read_summary(binding.source_root)
    directories = _sensor_directories(binding.source_root)
    directory_by_id = {directory.name: directory for directory in directories}
    demographic_ids = set(demographics["user_id"].astype(str))
    summary_ids = set(summary["participant"].astype(str))
    if demographic_ids != set(directory_by_id) or demographic_ids != summary_ids:
        raise ResilientAdapterError(
            "publisher user_id join differs across demographics, summary, and sensors"
        )
    summary_groups = summary.set_index(["participant", "file"])
    aliases = {
        source_id: f"participant_{index:04d}"
        for index, source_id in enumerate(
            sorted(demographic_ids, key=lambda value: value.encode("utf-8")), 1
        )
    }
    sensor_windows: dict[str, AggregatedSensorWindow] = {}
    quality_rows: list[dict[str, Any]] = []
    total_rows = defaultdict(int)
    for source_id in sorted(demographic_ids, key=lambda value: value.encode("utf-8")):
        directory = directory_by_id[source_id]
        actual_names = {path.name for path in directory.iterdir() if path.is_file()}
        if actual_names != set(SENSOR_FILES):
            raise ResilientAdapterError(
                "real participant sensor directory does not contain four CSV files"
            )
        frames: dict[str, pd.DataFrame] = {}
        for file_name in SENSOR_FILES:
            frame = _read_sensor_frame(directory, file_name)
            summary_row = summary_groups.loc[(source_id, file_name)]
            _reconcile_summary_row(frame, file_name, summary_row)
            frames[file_name] = frame
            total_rows[file_name] += len(frame)
        alias = aliases[source_id]
        window = aggregate_sensor_window(frames)
        sensor_windows[alias] = window
        quality_rows.append(dict(window.quality))
    if dict(total_rows) != EXPECTED_SENSOR_ROWS:
        raise ResilientAdapterError("RESILIENT sensor row totals do not match")
    canonical_demographics = demographics.copy(deep=True)
    canonical_demographics["user_id"] = canonical_demographics["user_id"].map(aliases)
    canonical_demographics = canonical_demographics.sort_values(
        "user_id", kind="stable"
    ).reset_index(drop=True)
    if canonical_demographics["user_id"].isna().any():
        raise ResilientAdapterError("RESILIENT alias mapping is incomplete")
    return canonical_demographics, sensor_windows, quality_rows


def _missing_summary(series: pd.Series) -> dict[str, Any]:
    missing = int(series.isna().sum())
    total = int(len(series))
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
        "count": int(len(array)),
        "minimum": int(np.min(array)),
        "median": float(np.median(array)),
        "maximum": int(np.max(array)),
    }


def _quality_report(
    canonical: pd.DataFrame,
    quality_rows: Sequence[Mapping[str, Any]],
    binding: SourceBinding,
) -> dict[str, Any]:
    source_missingness = {
        name: _missing_summary(canonical[_source_column(name)])
        for name in SOURCE_FIELDS
    }
    target_missingness: dict[str, Any] = {}
    for spec in _target_specs(feature_schema_manifest()):
        target = _feature_column(str(spec["group"]), str(spec["name"]))
        mask = f"feature_mask.{target}"
        target_missingness[target] = {
            **_missing_summary(canonical[target]),
            "mask_1_count": int(canonical[mask].sum()),
            "mask_0_count": int((canonical[mask] == 0).sum()),
        }
    steps_days = [int(row["ScanWatch_Steps.csv"]["valid_days"]) for row in quality_rows]
    hr_days = [int(row["ScanWatch_HR.csv"]["valid_days"]) for row in quality_rows]
    sleep_nights = [
        int(row["Sleep_state.csv"]["window_main_sleep_night_count"])
        for row in quality_rows
    ]
    physio_nights = [
        int(row["Sleep_physio.csv"]["valid_nights"]) for row in quality_rows
    ]
    category_counts = canonical["phq9_category"].value_counts().sort_index()
    return {
        "report_version": QUALITY_REPORT_VERSION,
        "dataset_id": DATASET_ID,
        "input_binding": binding.report_dict(),
        "timescale_semantics": TIMESCALE_SEMANTICS,
        "future_prediction_claim": False,
        "join_summary": {
            "demographics_rows": EXPECTED_PARTICIPANTS,
            "summary_rows": EXPECTED_SUMMARY_ROWS,
            "real_sensor_directories": EXPECTED_PARTICIPANTS,
            "joined_participants": len(canonical),
            "publisher_user_id_sets_equal": True,
            "source_participant_values_written_to_report": False,
            "canonical_participant_ids_are_collection_local_aliases": True,
        },
        "os_metadata": {
            "included_in_source_collection_hash": True,
            "parsed_as_business_csv": False,
        },
        "window_summary": {
            "source_timestamp_timezone": SOURCE_TIMESTAMP_TIMEZONE,
            "calendar_timezone": SOURCE_TIMEZONE,
            "calendar_days": WINDOW_CALENDAR_DAYS,
            "step_valid_days": _integer_distribution(steps_days),
            "scanwatch_hr_valid_days": _integer_distribution(hr_days),
            "main_sleep_valid_nights": _integer_distribution(sleep_nights),
            "sleep_physio_valid_nights": _integer_distribution(physio_nights),
        },
        "sensor_row_summary": {
            file_name: {
                "source_rows": EXPECTED_SENSOR_ROWS[file_name],
                "file_count": EXPECTED_PARTICIPANTS,
            }
            for file_name in SENSOR_FILES
        },
        "sensor_quality_summary": {
            "sleep_physio_duplicate_timestamp_rows_removed": sum(
                int(row["Sleep_physio.csv"]["duplicate_timestamp_rows_removed"])
                for row in quality_rows
            ),
            "sleep_physio_rows_aligned_to_main_sleep": sum(
                int(row["Sleep_physio.csv"]["aligned_rows"]) for row in quality_rows
            ),
            "sleep_episode_invalid_count": sum(
                int(row["Sleep_state.csv"]["discarded_invalid_episode_count"])
                for row in quality_rows
            ),
        },
        "participant_summary": {
            "participant_count": int(canonical["participant_id"].nunique()),
            "participant_values_written_to_report": False,
        },
        "label_summary": {
            "binary_definition": "phq_total >= 10",
            "positive_count": int(canonical["binary_target"].sum()),
            "negative_count": int((canonical["binary_target"] == 0).sum()),
            "phq9_category_counts": {
                str(int(key)): int(value) for key, value in category_counts.items()
            },
            "gds15_at_or_above_5_count": int(canonical["gds15_binary_target"].sum()),
            "gad7_at_or_above_10_count": int(canonical["gad7_binary_target"].sum()),
            "label_fields_used_as_input_features": False,
        },
        "x_source": {
            "field": X_SOURCE_FIELD,
            **_missing_summary(canonical["x_source_value"]),
            "base_canonical_ecdf_fitted": False,
        },
        "source_field_missingness": source_missingness,
        "canonical_target_missingness_and_masks": target_missingness,
    }


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    table = _canonical_arrow_table(frame)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        version="2.6",
        data_page_version="1.0",
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        row_group_size=128,
    )
    return sink.getvalue().to_pybytes()


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second or first.is_relative_to(second) or second.is_relative_to(first)
    )


def _validated_output_root(destination: Path, binding: SourceBinding) -> Path:
    resolved = destination.resolve(strict=False)
    source_data_root = (binding.workspace_root / "数据集").resolve()
    if _paths_overlap(resolved, source_data_root):
        raise ResilientAdapterError("output root must not overlap source data")
    return resolved


def _stage_content(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".stage", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _reserve_backup_path(path: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".backup", dir=path.parent
    )
    os.close(descriptor)
    backup = Path(temporary_name)
    backup.unlink()
    return backup


def _publish_artifacts(
    destination: Path,
    payloads: Mapping[str, bytes],
    *,
    overwrite: bool,
) -> None:
    manifest_relative = "resilient/artifact_manifest.json"
    if manifest_relative not in payloads:
        raise ResilientAdapterError("artifact payload has no completion manifest")
    for relative in payloads:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ResilientAdapterError("artifact relative path is unsafe")
    targets = {relative: destination / Path(relative) for relative in payloads}
    changes: dict[str, bytes] = {}
    original_exists: dict[str, bool] = {}
    for relative, target in targets.items():
        exists = target.is_file()
        original_exists[relative] = exists
        if exists and target.read_bytes() == payloads[relative]:
            continue
        if target.exists() and not target.is_file():
            raise ResilientAdapterError("artifact target is not a file")
        if exists and not overwrite:
            raise ResilientAdapterError(
                f"refusing to overwrite differing artifact: {target.name}"
            )
        changes[relative] = payloads[relative]
    if not changes:
        return
    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    backup_order = [manifest_relative, *sorted(set(changes) - {manifest_relative})]
    publish_order = [*sorted(set(changes) - {manifest_relative}), manifest_relative]
    try:
        for relative, content in changes.items():
            staged[relative] = _stage_content(targets[relative], content)
        for relative in backup_order:
            target = targets[relative]
            if relative in changes and target.is_file():
                backup = _reserve_backup_path(target)
                os.replace(target, backup)
                backups[relative] = backup
        for relative in publish_order:
            if relative in changes:
                os.replace(staged[relative], targets[relative])
        for backup in backups.values():
            backup.unlink(missing_ok=True)
    except BaseException:
        for relative in reversed(publish_order):
            if relative not in changes:
                continue
            target = targets[relative]
            backup = backups.get(relative)
            if backup is not None and backup.exists():
                target.unlink(missing_ok=True)
                os.replace(backup, target)
            elif not original_exists[relative]:
                target.unlink(missing_ok=True)
        raise
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)
        for path in backups.values():
            path.unlink(missing_ok=True)


def build_resilient_artifacts(
    *,
    workspace_root: Path | None = None,
    manifests_root: Path | None = None,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> BuildResult:
    """Validate frozen inputs and write deterministic DATA-003 artifacts."""

    binding = validate_frozen_inputs(workspace_root, manifests_root)
    destination = _validated_output_root(
        output_root
        or _algorithm_root(binding.workspace_root) / DEFAULT_OUTPUT_RELATIVE,
        binding,
    )
    demographics, sensor_windows, quality_rows = _load_and_aggregate_sources(binding)
    canonical = build_canonical_frame(
        demographics,
        sensor_windows,
        enforce_frozen_statistics=True,
    )
    mapping = build_field_mapping()
    quality = _quality_report(canonical, quality_rows, binding)
    revalidated_binding = validate_frozen_inputs(
        binding.workspace_root, binding.manifests_root
    )
    if revalidated_binding != binding:
        raise ResilientAdapterError("frozen source binding changed while being read")

    canonical_bytes = _parquet_bytes(canonical)
    mapping_bytes = _canonical_json_bytes(mapping)
    quality_bytes = _canonical_json_bytes(quality)
    preliminary_hashes = {
        "resilient/canonical_resilient.parquet": _sha256_bytes(canonical_bytes),
        "resilient/data_quality_report.json": _sha256_bytes(quality_bytes),
        "mappings/resilient_v3_3_3_mapping.json": _sha256_bytes(mapping_bytes),
    }
    metadata = {
        "adapter_version": ADAPTER_VERSION,
        "canonical_column_order_version": CANONICAL_COLUMN_ORDER_VERSION,
        "dataset_id": DATASET_ID,
        "feature_schema_version": feature_schema_manifest()["schema_version"],
        "input_binding": binding.report_dict(),
        "timescale_semantics": TIMESCALE_SEMANTICS,
        "source_timestamp_timezone": SOURCE_TIMESTAMP_TIMEZONE,
        "calendar_timezone": SOURCE_TIMEZONE,
        "window_calendar_days": WINDOW_CALENDAR_DAYS,
        "canonical_sort": ["participant_id"],
        "canonical_identity_policy": (
            "collection-local deterministic aliases; source user_id is used only "
            "for in-memory joins"
        ),
        "canonical_column_order": list(canonical.columns),
        "canonical_dtypes": {
            name: str(dtype) for name, dtype in canonical.dtypes.items()
        },
        "canonical_row_count": len(canonical),
        "canonical_participant_count": int(canonical["participant_id"].nunique()),
        "canonical_frame_sha256": canonical_frame_sha256(canonical),
        "base_canonical_ecdf_fitted": False,
        "formal_ecdf_instance_created": False,
        "artifact_sha256_before_metadata": preliminary_hashes,
    }
    metadata_bytes = _canonical_json_bytes(metadata)
    payloads = {
        "resilient/canonical_resilient.parquet": canonical_bytes,
        "resilient/data_quality_report.json": quality_bytes,
        "resilient/adapter_metadata.json": metadata_bytes,
        "mappings/resilient_v3_3_3_mapping.json": mapping_bytes,
    }
    artifact_rows = {
        relative: {"bytes": len(content), "sha256": _sha256_bytes(content)}
        for relative, content in sorted(payloads.items())
    }
    artifact_manifest = {
        "manifest_version": ARTIFACT_MANIFEST_VERSION,
        "dataset_id": DATASET_ID,
        "input_binding": binding.report_dict(),
        "artifacts": artifact_rows,
        "artifact_count": len(artifact_rows),
        "complete": True,
    }
    payloads["resilient/artifact_manifest.json"] = _canonical_json_bytes(
        artifact_manifest
    )
    _publish_artifacts(destination, payloads, overwrite=overwrite)
    all_hashes = {
        relative: _sha256_bytes(content)
        for relative, content in sorted(payloads.items())
    }
    return BuildResult(
        output_root=destination,
        artifact_sha256=all_hashes,
        row_count=len(canonical),
        participant_count=int(canonical["participant_id"].nunique()),
        positive_row_count=int(canonical["binary_target"].sum()),
    )


def _participant_set_sha256(participant_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for participant_id in sorted(
        participant_ids, key=lambda value: value.encode("utf-8")
    ):
        digest.update(participant_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _numeric_array(series: pd.Series, *, message: str) -> np.ndarray:
    try:
        values = pd.to_numeric(series, errors="raise").to_numpy(
            dtype="float64", na_value=np.nan
        )
    except (TypeError, ValueError) as exc:
        raise ResilientAdapterError(message) from exc
    if np.isinf(values).any():
        raise ResilientAdapterError(message)
    return values


def _validate_ecdf_canonical_frame(
    frame: pd.DataFrame,
    *,
    require_base_activity: bool,
    require_sorted: bool,
) -> np.ndarray:
    if tuple(frame.columns) != canonical_column_order() or frame.empty:
        raise ResilientAdapterError("ECDF input is not a complete canonical frame")
    if frame["dataset_id"].isna().any() or not frame["dataset_id"].eq(DATASET_ID).all():
        raise ResilientAdapterError("ECDF input frame has the wrong dataset ID")
    identities = frame[["global_participant_id", "participant_id"]]
    if identities.isna().any(axis=None) or frame["participant_id"].duplicated().any():
        raise ResilientAdapterError("ECDF canonical identity keys are invalid")
    participants = frame["participant_id"].astype(str)
    expected_global = DATASET_ID + "::" + participants
    if not frame["global_participant_id"].astype(str).equals(expected_global):
        raise ResilientAdapterError("ECDF global participant keys do not match")
    if require_sorted and participants.tolist() != sorted(participants.tolist()):
        raise ResilientAdapterError("ECDF canonical rows are not sorted")
    phq = _numeric_array(frame["phq9_score"], message="ECDF labels are invalid")
    binary = _numeric_array(
        frame["binary_target"], message="ECDF binary target is invalid"
    )
    if not np.array_equal(binary, (phq >= 10).astype("float64")):
        raise ResilientAdapterError("ECDF binary target does not match PHQ-9")
    if not frame["x_source_name"].eq(X_SOURCE_FIELD).all():
        raise ResilientAdapterError("ECDF x_source name does not match")
    source = _numeric_array(
        frame[X_SOURCE_CANONICAL_COLUMN], message="ECDF source values are invalid"
    )
    x_source = _numeric_array(
        frame["x_source_value"], message="ECDF source values are invalid"
    )
    same = (np.isnan(source) & np.isnan(x_source)) | (source == x_source)
    if not bool(same.all()):
        raise ResilientAdapterError("ECDF canonical source columns do not match")
    source_mask = _numeric_array(
        frame["x_source_mask"], message="ECDF source mask is invalid"
    )
    if not np.array_equal(source_mask, (~np.isnan(source)).astype("float64")):
        raise ResilientAdapterError("ECDF x_source mask does not match")
    activity = _numeric_array(
        frame[ACTIVITY_VOLUME_COLUMN], message="ECDF activity values are invalid"
    )
    activity_mask = _numeric_array(
        frame[ACTIVITY_VOLUME_MASK_COLUMN],
        message="ECDF activity mask is invalid",
    )
    if require_base_activity:
        if not np.isnan(activity).all() or bool(activity_mask.any()):
            raise ResilientAdapterError("ECDF fit requires base activity to be null")
    else:
        finite = activity[~np.isnan(activity)]
        if ((finite < 0.0) | (finite > 1.0)).any() or not np.array_equal(
            activity_mask, (~np.isnan(activity)).astype("float64")
        ):
            raise ResilientAdapterError("ECDF activity values are invalid")
    return source


@dataclass(frozen=True)
class TrainingFoldECDF:
    """Right-continuous ECDF fitted on explicit outer-training participants."""

    split_id: str
    canonical_artifact_sha256: str
    canonical_frame_sha256: str
    training_participant_sha256: str
    training_participant_count: int
    sorted_training_values: tuple[float, ...]
    version: str = ECDF_VERSION
    adapter_version: str = ADAPTER_VERSION
    dataset_id: str = DATASET_ID
    source_collection_sha256: str = RESILIENT_COLLECTION_SHA256
    feature_schema_sha256: str = MH003_FEATURE_SCHEMA_SHA256
    source_field: str = X_SOURCE_CANONICAL_COLUMN
    target_field: str = ACTIVITY_VOLUME_COLUMN
    target_mask_field: str = ACTIVITY_VOLUME_MASK_COLUMN

    def __post_init__(self) -> None:
        if not self.split_id.strip() or "\0" in self.split_id:
            raise ResilientAdapterError("ECDF split_id must be non-empty")
        if self.version != ECDF_VERSION or self.dataset_id != DATASET_ID:
            raise ResilientAdapterError("ECDF identity binding does not match DATA-003")
        if self.adapter_version != ADAPTER_VERSION:
            raise ResilientAdapterError("ECDF adapter binding does not match DATA-003")
        if self.source_collection_sha256 != RESILIENT_COLLECTION_SHA256:
            raise ResilientAdapterError("ECDF source collection binding does not match")
        if self.feature_schema_sha256 != MH003_FEATURE_SCHEMA_SHA256:
            raise ResilientAdapterError("ECDF feature schema binding does not match")
        if self.source_field != X_SOURCE_CANONICAL_COLUMN:
            raise ResilientAdapterError("ECDF source field binding does not match")
        if self.target_field != ACTIVITY_VOLUME_COLUMN:
            raise ResilientAdapterError("ECDF target field binding does not match")
        if self.target_mask_field != ACTIVITY_VOLUME_MASK_COLUMN:
            raise ResilientAdapterError("ECDF target mask binding does not match")
        if self.training_participant_count <= 0:
            raise ResilientAdapterError("ECDF needs training participants")
        if not _is_sha256(self.canonical_artifact_sha256) or not _is_sha256(
            self.canonical_frame_sha256
        ):
            raise ResilientAdapterError("ECDF canonical hash binding is invalid")
        if not _is_sha256(self.training_participant_sha256):
            raise ResilientAdapterError("ECDF participant hash binding is invalid")
        if not self.sorted_training_values or any(
            not math.isfinite(value) for value in self.sorted_training_values
        ):
            raise ResilientAdapterError("ECDF training values must be finite")
        if tuple(sorted(self.sorted_training_values)) != self.sorted_training_values:
            raise ResilientAdapterError("ECDF training values must be sorted")

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        training_participant_ids: Sequence[str],
        split_id: str,
        canonical_artifact_sha256: str,
        expected_canonical_frame_sha256: str,
    ) -> TrainingFoldECDF:
        if not isinstance(split_id, str) or not split_id.strip():
            raise ResilientAdapterError("ECDF fit requires a non-empty split_id")
        participants = list(training_participant_ids)
        if not participants or any(
            not isinstance(value, str)
            or not value.startswith(f"{DATASET_ID}::participant_")
            or "\0" in value
            for value in participants
        ):
            raise ResilientAdapterError(
                "ECDF fit requires namespaced RESILIENT participant IDs"
            )
        if len(participants) != len(set(participants)):
            raise ResilientAdapterError("ECDF training participant IDs must be unique")
        if not _is_sha256(canonical_artifact_sha256):
            raise ResilientAdapterError("ECDF fit requires a canonical SHA-256")
        source = _validate_ecdf_canonical_frame(
            frame, require_base_activity=True, require_sorted=True
        )
        observed_frame_hash = canonical_frame_sha256(frame)
        if (
            not _is_sha256(expected_canonical_frame_sha256)
            or observed_frame_hash != expected_canonical_frame_sha256
        ):
            raise ResilientAdapterError("ECDF canonical frame SHA-256 does not match")
        present = set(frame["global_participant_id"].astype(str))
        if not set(participants).issubset(present):
            raise ResilientAdapterError("ECDF training participants are absent")
        selected = frame["global_participant_id"].isin(participants).to_numpy()
        finite = source[selected]
        finite = finite[~np.isnan(finite)]
        if not len(finite):
            raise ResilientAdapterError("ECDF training fold has no finite values")
        return cls(
            split_id=split_id.strip(),
            canonical_artifact_sha256=canonical_artifact_sha256,
            canonical_frame_sha256=observed_frame_hash,
            training_participant_sha256=_participant_set_sha256(participants),
            training_participant_count=len(participants),
            sorted_training_values=tuple(float(value) for value in np.sort(finite)),
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        source = _validate_ecdf_canonical_frame(
            frame, require_base_activity=False, require_sorted=False
        )
        result = frame.copy(deep=True)
        output = np.full(len(result), np.nan, dtype="float64")
        present = ~np.isnan(source)
        training = np.asarray(self.sorted_training_values, dtype="float64")
        output[present] = np.searchsorted(
            training, source[present], side="right"
        ) / len(training)
        result[self.target_field] = pd.array(output, dtype="Float64")
        result[self.target_mask_field] = present.astype("int8")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "adapter_version": self.adapter_version,
            "dataset_id": self.dataset_id,
            "source_collection_sha256": self.source_collection_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "source_field": self.source_field,
            "target_field": self.target_field,
            "target_mask_field": self.target_mask_field,
            "split_id": self.split_id,
            "canonical_artifact_sha256": self.canonical_artifact_sha256,
            "canonical_frame_sha256": self.canonical_frame_sha256,
            "training_participant_sha256": self.training_participant_sha256,
            "training_participant_count": self.training_participant_count,
            "training_value_count": len(self.sorted_training_values),
            "sorted_training_values": list(self.sorted_training_values),
            "definition": "count(training_values <= x) / training_value_count",
            "ties": "right_continuous",
        }

    def to_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return _sha256_bytes(self.to_bytes())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrainingFoldECDF:
        expected = {
            "version",
            "adapter_version",
            "dataset_id",
            "source_collection_sha256",
            "feature_schema_sha256",
            "source_field",
            "target_field",
            "target_mask_field",
            "split_id",
            "canonical_artifact_sha256",
            "canonical_frame_sha256",
            "training_participant_sha256",
            "training_participant_count",
            "training_value_count",
            "sorted_training_values",
            "definition",
            "ties",
        }
        if set(payload) != expected:
            raise ResilientAdapterError("serialized ECDF fields do not match")
        values = tuple(float(value) for value in payload["sorted_training_values"])
        if int(payload["training_value_count"]) != len(values):
            raise ResilientAdapterError("serialized ECDF value count does not match")
        if (
            payload["definition"]
            != "count(training_values <= x) / training_value_count"
        ):
            raise ResilientAdapterError("serialized ECDF definition does not match")
        if payload["ties"] != "right_continuous":
            raise ResilientAdapterError("serialized ECDF tie policy does not match")
        return cls(
            version=str(payload["version"]),
            adapter_version=str(payload["adapter_version"]),
            dataset_id=str(payload["dataset_id"]),
            source_collection_sha256=str(payload["source_collection_sha256"]),
            feature_schema_sha256=str(payload["feature_schema_sha256"]),
            source_field=str(payload["source_field"]),
            target_field=str(payload["target_field"]),
            target_mask_field=str(payload["target_mask_field"]),
            split_id=str(payload["split_id"]),
            canonical_artifact_sha256=str(payload["canonical_artifact_sha256"]),
            canonical_frame_sha256=str(payload["canonical_frame_sha256"]),
            training_participant_sha256=str(payload["training_participant_sha256"]),
            training_participant_count=int(payload["training_participant_count"]),
            sorted_training_values=values,
        )


__all__ = [
    "ACTIVITY_VOLUME_COLUMN",
    "ACTIVITY_VOLUME_MASK_COLUMN",
    "ADAPTER_VERSION",
    "AggregatedSensorWindow",
    "BuildResult",
    "CANONICAL_COLUMN_ORDER_VERSION",
    "DATA001_FILE_MANIFEST_SHA256",
    "DATASET_ID",
    "ECDF_VERSION",
    "EXPECTED_GAD_POSITIVES",
    "EXPECTED_GDS_POSITIVES",
    "EXPECTED_PARTICIPANTS",
    "EXPECTED_PHQ_CATEGORY_COUNTS",
    "EXPECTED_POSITIVES",
    "MH003_FEATURE_SCHEMA_SHA256",
    "RESILIENT_COLLECTION_SHA256",
    "ResilientAdapterError",
    "SOURCE_FIELDS",
    "SOURCE_TIMEZONE",
    "SourceBinding",
    "TIMESCALE_SEMANTICS",
    "TrainingFoldECDF",
    "WINDOW_CALENDAR_DAYS",
    "aggregate_sensor_window",
    "build_canonical_frame",
    "build_field_mapping",
    "build_resilient_artifacts",
    "canonical_arrow_schema",
    "canonical_column_order",
    "canonical_frame_sha256",
    "validate_frozen_inputs",
]
