"""NHANES V3.3.3 canonical adapter and fold-bound activity ECDF.

NHANES DPQ is collected before the de-identified PAM windows.  The adapter
therefore marks every row as a post-assessment association, namespaces SEQN by
survey cycle, and never fabricates an assessment date.  The base canonical
table preserves the mean M10VALUE only as ``x_source_value``; activity volume
remains unavailable until an explicit outer-training fold fits an ECDF.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    feature_schema_manifest,
)


DATASET_ID = "nhanes"
ADAPTER_VERSION = "nhanes-adapter-v3.3.3-data004"
CANONICAL_COLUMN_ORDER_VERSION = "nhanes-canonical-columns-v1"
MAPPING_VERSION = "nhanes-field-mapping-v3.3.3-data004"
QUALITY_REPORT_VERSION = "nhanes-quality-report-v1"
ARTIFACT_MANIFEST_VERSION = "nhanes-artifact-manifest-v1"
ECDF_VERSION = "training-fold-right-continuous-ecdf-v1"

SOURCE_RELATIVE = Path("数据集/心理/NHANES")
PAM_FILE_NAME = "NHANES Preliminary Day Level Output.csv"
DEFAULT_OUTPUT_RELATIVE = Path("data/processed/mental_health/mood_social/v3.3.3")

NHANES_COLLECTION_SHA256 = (
    "893a069a74ef8e515c49ef95b96182e03f611ae65abac66fd93cfc375ec17c1e"
)
DATA001_FILE_MANIFEST_SHA256 = (
    "59257b21fee44bd405e0be6f299d49ad945c6e6289002717f8769389dce32bd5"
)
MH003_FEATURE_SCHEMA_SHA256 = (
    "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
)

CYCLES: tuple[str, ...] = ("G", "H")
SURVEY_CYCLES = {"G": "2011-2012", "H": "2013-2014"}
TIMESCALE_SEMANTICS = "post_assessment_association"
WINDOW_SEMANTICS = (
    "excluded_equals_0; cycle_plus_seqn_plus_window_number; "
    "window_number_ascending; first_7; calendar_date_not_deduplicated"
)

DPQ_FIELDS = tuple(f"DPQ0{index}0" for index in range(1, 10))
DPQ_FUNCTION_FIELD = "DPQ100"
PAM_REQUIRED_FIELDS = (
    "SEQN",
    "calendar_date",
    "window_number",
    "excluded",
    "M10VALUE",
    "L5VALUE",
    "dur_spt_sleep_min",
    "dur_spt_min",
    "sleep_efficiency",
    "sleeponset",
    "wakeup",
    "dur_spt_wake_IN_min",
)
DEMOGRAPHIC_FIELDS = (
    "SEQN",
    "RIDAGEYR",
    "RIAGENDR",
    "DMDMARTL",
    "DMDEDUC2",
)

IDENTITY_COLUMNS = (
    "dataset_id",
    "global_participant_id",
    "participant_id",
    "cycle",
    "survey_cycle",
    "seqn",
    "timescale_semantics",
    "window_semantics",
    "window_number_values_json",
    "pam_valid_day_count",
)
TARGET_COLUMNS = (*DPQ_FIELDS, "phq9_total", "phq9_severity", "binary_target")
X_SOURCE_COLUMNS = ("x_source_name", "x_source_value", "x_source_mask")
X_SOURCE_NAME = "M10VALUE_mean_valid_pam_days"
ACTIVITY_VOLUME_COLUMN = "activity.activity_volume_norm"
ACTIVITY_VOLUME_MASK_COLUMN = "feature_mask.activity.activity_volume_norm"

EXPECTED_SOURCE_FILE_COUNT = 22
EXPECTED_PAM_ROWS = 89_104
EXPECTED_PRE_QUALITY_PARTICIPANTS = 2_827
EXPECTED_CANONICAL_PARTICIPANTS = 2_775
EXPECTED_CYCLE_COUNTS = {"G": 1_373, "H": 1_402}
EXPECTED_POSITIVE_ROWS = 254
EXPECTED_SEVERITY_COUNTS = {0: 2_079, 1: 442, 2: 165, 3: 65, 4: 24}
EXPECTED_VALID_DAY_COUNTS = {1: 61, 2: 115, 3: 165, 4: 244, 5: 427, 6: 672, 7: 1_091}
EXPECTED_SAME_DATE_MULTIWINDOW_GROUPS = 702
EXPECTED_SAME_DATE_MULTIWINDOW_RECORDS = 1_404

_MINUTES_PER_DAY = 1_440.0
_EPSILON = 1e-12

_DIRECT_TARGET_RULES: dict[str, dict[str, Any]] = {
    "activity.relative_amplitude": {
        "source_fields": ["M10VALUE", "L5VALUE"],
        "formula": "(mean(M10VALUE)-mean(L5VALUE))/(mean(M10VALUE)+mean(L5VALUE))",
        "mapping_kind": "derived_same_semantics",
    },
    "activity.activity_variability": {
        "source_fields": ["M10VALUE"],
        "formula": "population_sd(M10VALUE)/abs(mean(M10VALUE)); requires >=2 days",
        "mapping_kind": "derived_same_semantics",
    },
    "activity.valid_days": {
        "source_fields": ["excluded", "window_number", "M10VALUE", "L5VALUE"],
        "formula": "count(selected PAM windows with a finite activity measure)",
        "mapping_kind": "derived_coverage",
    },
    "activity.feature_coverage": {
        "source_fields": ["excluded", "M10VALUE", "L5VALUE"],
        "formula": "(valid_activity_days/7)*(available_supported_semantics/7)",
        "mapping_kind": "derived_coverage_only",
    },
    "sleep.sleep_duration_norm": {
        "source_fields": ["dur_spt_sleep_min"],
        "formula": "mean(dur_spt_sleep_min)/1440",
        "mapping_kind": "derived_same_semantics",
    },
    "sleep.time_in_bed_norm": {
        "source_fields": ["dur_spt_min"],
        "formula": "mean(dur_spt_min)/1440",
        "mapping_kind": "derived_same_semantics_sleep_period",
    },
    "sleep.sleep_efficiency": {
        "source_fields": ["dur_spt_sleep_min", "dur_spt_min"],
        "formula": "sum(dur_spt_sleep_min)/sum(dur_spt_min)",
        "mapping_kind": "derived_same_semantics",
    },
    "sleep.sleep_onset_sin": {
        "source_fields": ["sleeponset"],
        "formula": "unit_circular_mean_sin(sleeponset mod 24)",
        "mapping_kind": "derived_circular_same_semantics",
    },
    "sleep.sleep_onset_cos": {
        "source_fields": ["sleeponset"],
        "formula": "unit_circular_mean_cos(sleeponset mod 24)",
        "mapping_kind": "derived_circular_same_semantics",
    },
    "sleep.wake_time_sin": {
        "source_fields": ["wakeup"],
        "formula": "unit_circular_mean_sin(wakeup mod 24)",
        "mapping_kind": "derived_circular_same_semantics",
    },
    "sleep.wake_time_cos": {
        "source_fields": ["wakeup"],
        "formula": "unit_circular_mean_cos(wakeup mod 24)",
        "mapping_kind": "derived_circular_same_semantics",
    },
    "sleep.sleep_midpoint_sin": {
        "source_fields": ["sleeponset", "wakeup"],
        "formula": "unit_circular_mean_sin(forward sleep midpoint)",
        "mapping_kind": "derived_circular_same_semantics",
    },
    "sleep.sleep_midpoint_cos": {
        "source_fields": ["sleeponset", "wakeup"],
        "formula": "unit_circular_mean_cos(forward sleep midpoint)",
        "mapping_kind": "derived_circular_same_semantics",
    },
    "sleep.sleep_fragmentation": {
        "source_fields": ["dur_spt_wake_IN_min", "dur_spt_min"],
        "formula": "mean(dur_spt_wake_IN_min/dur_spt_min)",
        "mapping_kind": "derived_same_semantics_awake_component",
    },
    "sleep.sleep_regularity": {
        "source_fields": ["sleeponset", "wakeup"],
        "formula": "mean(resultant_length(onset),resultant_length(wake),resultant_length(midpoint)); each requires >=2 nights",
        "mapping_kind": "derived_same_semantics",
    },
    "sleep.valid_nights": {
        "source_fields": ["excluded", "window_number", "dur_spt_sleep_min"],
        "formula": "count(selected PAM windows with a finite sleep measure)",
        "mapping_kind": "derived_coverage",
    },
    "sleep.feature_coverage": {
        "source_fields": list(PAM_REQUIRED_FIELDS),
        "formula": "observed semantic-window cells/(7*9)",
        "mapping_kind": "derived_coverage_only",
    },
    "social_context.age_group": {
        "source_fields": ["RIDAGEYR"],
        "formula": "60-69=>60_69; 70-79=>70_79; >=80=>80_plus",
        "mapping_kind": "derived_same_semantics",
    },
    "social_context.sex": {
        "source_fields": ["RIAGENDR"],
        "formula": "1=>male; 2=>female; other=>null",
        "mapping_kind": "coded_same_semantics",
    },
    "social_context.marital_status": {
        "source_fields": ["DMDMARTL"],
        "formula": "1 or 6=>partnered; 2,3,4,5=>not_partnered; 77/99/missing=>null",
        "mapping_kind": "coded_same_semantics",
    },
    "social_context.education_level": {
        "source_fields": ["DMDEDUC2"],
        "formula": "1=>primary_or_less; 2 or 3=>middle; 4 or 5=>high_or_above; 7/9/missing=>null",
        "mapping_kind": "coded_same_semantics",
    },
}


class NhanesAdapterError(ValueError):
    """Raised when a frozen DATA-004 source or output contract is violated."""


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
class BuildResult:
    output_root: Path
    artifact_sha256: Mapping[str, str]
    row_count: int
    participant_count: int
    positive_row_count: int


@dataclass(frozen=True)
class _PamAudit:
    source_rows: int
    unique_window_keys: int
    same_date_multiwindow_groups: int
    same_date_multiwindow_records: int
    invalid_excluded_rows: int
    valid_rows_before_limit: int
    selected_rows_all_participants: int
    truncated_valid_rows: int
    selected_same_date_multiwindow_groups: int


@dataclass(frozen=True)
class _Assembly:
    canonical: pd.DataFrame
    flow_by_cycle: Mapping[str, Mapping[str, int]]
    dpq_invalid_counts: Mapping[str, int]
    pam_audit: _PamAudit
    final_window_rows: int


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
    ordered = sorted(rows, key=lambda row: str(row["relative_path"]).encode("utf-8"))
    for row in ordered:
        digest.update(
            (f"{row['relative_path']}\0{row['bytes']}\0{row['sha256']}\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _find_workspace_root(start: Path) -> Path:
    resolved = start.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "数据集" / "心理" / "NHANES").is_dir() and (
            candidate / "algorithm" / "eldercare-risk-ai-main"
        ).is_dir():
            return candidate
    raise NhanesAdapterError(
        "workspace root containing the NHANES source was not found"
    )


def _algorithm_root(workspace_root: Path) -> Path:
    path = workspace_root / "algorithm" / "eldercare-risk-ai-main"
    if not path.is_dir():
        raise NhanesAdapterError("algorithm repository root was not found")
    return path


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NhanesAdapterError("frozen JSON input is not readable") from exc


def _read_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise NhanesAdapterError(
                        "DATA-001 file manifest contains a non-object row"
                    )
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NhanesAdapterError("DATA-001 file manifest is not readable") from exc
    return rows


def validate_frozen_inputs(
    workspace_root: Path | None = None,
    manifests_root: Path | None = None,
) -> SourceBinding:
    """Revalidate DATA-001 files and the live MH-003 schema before reading data."""

    root = _find_workspace_root(workspace_root or Path.cwd())
    algorithm_root = _algorithm_root(root)
    audit_root = (
        manifests_root or algorithm_root / DEFAULT_OUTPUT_RELATIVE / "manifests"
    )
    manifest_path = audit_root / "dataset_file_manifest.jsonl"
    schema_path = audit_root / "feature_schema_manifest.json"

    if _sha256_file(manifest_path) != DATA001_FILE_MANIFEST_SHA256:
        raise NhanesAdapterError("DATA-001 file manifest SHA-256 does not match")

    live_schema = feature_schema_manifest()
    live_schema_sha256 = _sha256_bytes(_canonical_json_bytes(live_schema))
    if live_schema_sha256 != MH003_FEATURE_SCHEMA_SHA256:
        raise NhanesAdapterError("live MH-003 feature schema SHA-256 does not match")
    if _sha256_file(schema_path) != MH003_FEATURE_SCHEMA_SHA256:
        raise NhanesAdapterError(
            "DATA-001 feature schema snapshot SHA-256 does not match"
        )
    if _read_json(schema_path) != live_schema:
        raise NhanesAdapterError(
            "DATA-001 schema snapshot differs from live MH-003 schema"
        )

    rows = _read_manifest_rows(manifest_path)
    selected = [row for row in rows if row.get("dataset_id") == DATASET_ID]
    if len(selected) != EXPECTED_SOURCE_FILE_COUNT:
        raise NhanesAdapterError("DATA-001 NHANES file count does not match")

    source_root = root / SOURCE_RELATIVE
    actual_paths = sorted(
        (path for path in source_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(source_root).as_posix().encode("utf-8"),
    )
    expected_paths = {str(row.get("relative_path", "")) for row in selected}
    actual_relative_paths = {
        (SOURCE_RELATIVE / path.relative_to(source_root)).as_posix()
        for path in actual_paths
    }
    if expected_paths != actual_relative_paths:
        raise NhanesAdapterError("current NHANES file set differs from DATA-001")

    current_rows: list[dict[str, Any]] = []
    for row in selected:
        relative = str(row["relative_path"])
        path = root / Path(relative)
        if path.is_symlink():
            raise NhanesAdapterError("NHANES source files must not be symbolic links")
        before = path.stat()
        file_sha256 = _sha256_file(path)
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise NhanesAdapterError("NHANES source changed during validation")
        if int(row.get("bytes", -1)) != after.st_size:
            raise NhanesAdapterError("NHANES source size differs from DATA-001")
        if str(row.get("sha256", "")) != file_sha256:
            raise NhanesAdapterError("NHANES source SHA-256 differs from DATA-001")
        current_rows.append(
            {
                "relative_path": relative,
                "bytes": int(after.st_size),
                "sha256": file_sha256,
            }
        )

    collection_sha256 = _collection_sha256(current_rows)
    if collection_sha256 != NHANES_COLLECTION_SHA256:
        raise NhanesAdapterError("NHANES collection SHA-256 does not match")
    return SourceBinding(
        workspace_root=root,
        source_root=source_root,
        manifests_root=audit_root,
        source_collection_sha256=collection_sha256,
        file_manifest_sha256=DATA001_FILE_MANIFEST_SHA256,
        feature_schema_sha256=live_schema_sha256,
        source_file_count=len(current_rows),
    )


def _target_specs(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in schema["group_order"]:
        for spec in schema["groups"][group]:
            rows.append({"group": group, **dict(spec)})
    if len(rows) != 54:
        raise NhanesAdapterError("MH-003 windowed feature count does not match")
    return rows


def _feature_column(group: str, name: str) -> str:
    return f"{group}.{name}"


def _mask_column(group: str, name: str) -> str:
    return f"feature_mask.{group}.{name}"


def canonical_column_order() -> tuple[str, ...]:
    target_specs = _target_specs(feature_schema_manifest())
    feature_columns = tuple(
        _feature_column(str(spec["group"]), str(spec["name"])) for spec in target_specs
    )
    mask_columns = tuple(
        _mask_column(str(spec["group"]), str(spec["name"])) for spec in target_specs
    )
    return (
        *IDENTITY_COLUMNS,
        *TARGET_COLUMNS,
        *X_SOURCE_COLUMNS,
        *feature_columns,
        *mask_columns,
    )


def canonical_arrow_schema() -> pa.Schema:
    field_types: dict[str, pa.DataType] = {
        "dataset_id": pa.large_string(),
        "global_participant_id": pa.large_string(),
        "participant_id": pa.large_string(),
        "cycle": pa.large_string(),
        "survey_cycle": pa.large_string(),
        "seqn": pa.int64(),
        "timescale_semantics": pa.large_string(),
        "window_semantics": pa.large_string(),
        "window_number_values_json": pa.large_string(),
        "pam_valid_day_count": pa.int8(),
        **{field: pa.int8() for field in TARGET_COLUMNS},
        "x_source_name": pa.large_string(),
        "x_source_value": pa.float64(),
        "x_source_mask": pa.int8(),
    }
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
            raise NhanesAdapterError("unsupported MH-003 feature type") from exc
        field_types[f"feature_mask.{target}"] = pa.int8()
    order = canonical_column_order()
    if set(field_types) != set(order):
        raise NhanesAdapterError("canonical Arrow schema columns do not match DATA-004")
    return pa.schema([pa.field(name, field_types[name]) for name in order])


def _canonical_arrow_table(frame: pd.DataFrame) -> pa.Table:
    if tuple(frame.columns) != canonical_column_order():
        raise NhanesAdapterError("canonical frame columns do not match DATA-004")
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
        raise NhanesAdapterError(
            "canonical frame values do not match the DATA-004 Arrow schema"
        ) from exc


def _coerce_canonical_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[:, list(canonical_column_order())].copy()
    for field in canonical_arrow_schema():
        if pa.types.is_large_string(field.type):
            result[field.name] = pd.array(result[field.name], dtype="string")
        elif pa.types.is_int8(field.type):
            result[field.name] = pd.array(result[field.name], dtype="Int8")
        elif pa.types.is_int64(field.type):
            result[field.name] = pd.array(result[field.name], dtype="Int64")
        elif pa.types.is_float64(field.type):
            result[field.name] = pd.array(result[field.name], dtype="Float64")
        else:
            raise NhanesAdapterError("canonical Arrow schema contains an unknown type")
    return result


def canonical_frame_sha256(frame: pd.DataFrame) -> str:
    table = _canonical_arrow_table(frame)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table, max_chunksize=max(len(table), 1))
    return _sha256_bytes(sink.getvalue().to_pybytes())


def _numeric_series(series: pd.Series, *, field: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise").astype("Float64")
    except (TypeError, ValueError) as exc:
        raise NhanesAdapterError(f"NHANES {field} is not numeric") from exc
    values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    if np.isinf(values).any():
        raise NhanesAdapterError(f"NHANES {field} contains infinity")
    return numeric


def _integral_series(
    series: pd.Series,
    *,
    field: str,
    allow_missing: bool,
) -> pd.Series:
    numeric = _numeric_series(series, field=field)
    values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    near_zero = np.isfinite(values) & (np.abs(values) < _EPSILON)
    values[near_zero] = 0.0
    finite = values[~np.isnan(values)]
    if (finite != np.floor(finite)).any():
        raise NhanesAdapterError(f"NHANES {field} must contain integer codes")
    if not allow_missing and np.isnan(values).any():
        raise NhanesAdapterError(f"NHANES {field} contains a missing key")
    return pd.Series(pd.array(values, dtype="Int64"), index=series.index)


def _prepare_participant_table(
    frame: pd.DataFrame,
    *,
    required: Sequence[str],
    table_name: str,
) -> pd.DataFrame:
    if not set(required).issubset(frame.columns):
        raise NhanesAdapterError(f"NHANES {table_name} is missing required fields")
    result = frame.loc[:, list(required)].copy()
    result["SEQN"] = _integral_series(
        result["SEQN"], field=f"{table_name}.SEQN", allow_missing=False
    )
    if bool(result["SEQN"].duplicated().any()):
        raise NhanesAdapterError(f"NHANES {table_name} SEQN must be unique")
    return result


def _normalise_dpq(
    dpq: pd.DataFrame, *, cycle: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    prepared = _prepare_participant_table(
        dpq,
        required=("SEQN", *DPQ_FIELDS, DPQ_FUNCTION_FIELD),
        table_name=f"DPQ_{cycle}",
    )
    output = prepared[["SEQN"]].copy()
    invalid_counts: dict[str, int] = {}
    for field in DPQ_FIELDS:
        numeric = _numeric_series(prepared[field], field=f"DPQ_{cycle}.{field}")
        values = numeric.to_numpy(dtype="float64", na_value=np.nan)
        near_zero = np.isfinite(values) & (np.abs(values) < _EPSILON)
        values[near_zero] = 0.0
        finite = values[np.isfinite(values)]
        if (finite != np.floor(finite)).any():
            raise NhanesAdapterError("NHANES DPQ items must contain integer codes")
        valid = np.isin(values, (0.0, 1.0, 2.0, 3.0))
        invalid_counts[field] = int((~valid).sum())
        values[~valid] = np.nan
        output[field] = pd.array(values, dtype="Int8")
    return output, invalid_counts


def _prepare_pam(
    pam: pd.DataFrame,
    *,
    cycle_members: Mapping[str, set[int]],
) -> tuple[pd.DataFrame, pd.DataFrame, _PamAudit]:
    if not set(PAM_REQUIRED_FIELDS).issubset(pam.columns):
        raise NhanesAdapterError("NHANES PAM table is missing required fields")
    result = pam.loc[:, list(PAM_REQUIRED_FIELDS)].copy()
    result["SEQN"] = _integral_series(
        result["SEQN"], field="PAM.SEQN", allow_missing=False
    )

    membership: dict[int, str] = {}
    for cycle in CYCLES:
        for seqn in cycle_members[cycle]:
            if seqn in membership:
                raise NhanesAdapterError(
                    "NHANES SEQN occurs in more than one survey cycle"
                )
            membership[seqn] = cycle
    result["cycle"] = pd.array(
        [membership.get(int(value)) for value in result["SEQN"]], dtype="string"
    )
    if result["cycle"].isna().any():
        raise NhanesAdapterError("NHANES PAM SEQN cannot be assigned to one cycle")

    result["window_number"] = _integral_series(
        result["window_number"], field="PAM.window_number", allow_missing=False
    )
    if bool((result["window_number"] <= 0).any()):
        raise NhanesAdapterError("NHANES PAM window_number must be positive")
    result["excluded"] = _integral_series(
        result["excluded"], field="PAM.excluded", allow_missing=False
    )
    if not result["excluded"].isin((0, 1)).all():
        raise NhanesAdapterError("NHANES PAM excluded must be 0 or 1")
    result["calendar_date"] = result["calendar_date"].astype("string")
    if result["calendar_date"].isna().any() or result["calendar_date"].eq("").any():
        raise NhanesAdapterError("NHANES PAM calendar_date is missing")
    for field in PAM_REQUIRED_FIELDS[4:]:
        result[field] = _numeric_series(result[field], field=f"PAM.{field}")

    window_key = ["cycle", "SEQN", "window_number"]
    if bool(result.duplicated(window_key).any()):
        raise NhanesAdapterError("NHANES cycle/SEQN/window_number key is not unique")

    same_date_sizes = result.groupby(
        ["cycle", "SEQN", "calendar_date"], dropna=False, sort=False
    ).size()
    duplicate_date_sizes = same_date_sizes[same_date_sizes > 1]
    valid = result.loc[result["excluded"].eq(0)].copy()
    valid = valid.sort_values(window_key, kind="stable")
    selected = valid.groupby(["cycle", "SEQN"], sort=False, group_keys=False).head(7)
    selected = selected.reset_index(drop=True)
    selected_date_sizes = selected.groupby(
        ["cycle", "SEQN", "calendar_date"], dropna=False, sort=False
    ).size()
    return (
        result,
        selected,
        _PamAudit(
            source_rows=len(result),
            unique_window_keys=int(result[window_key].drop_duplicates().shape[0]),
            same_date_multiwindow_groups=int(len(duplicate_date_sizes)),
            same_date_multiwindow_records=int(duplicate_date_sizes.sum()),
            invalid_excluded_rows=int(result["excluded"].eq(1).sum()),
            valid_rows_before_limit=len(valid),
            selected_rows_all_participants=len(selected),
            truncated_valid_rows=len(valid) - len(selected),
            selected_same_date_multiwindow_groups=int((selected_date_sizes > 1).sum()),
        ),
    )


def _finite_values(series: pd.Series) -> list[float]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    return [float(value) for value in values if math.isfinite(float(value))]


def _circular_components(hours: Sequence[float]) -> tuple[float | None, float | None]:
    if not hours:
        return None, None
    angles = [2.0 * math.pi * (float(hour) % 24.0) / 24.0 for hour in hours]
    mean_sin = fmean(math.sin(angle) for angle in angles)
    mean_cos = fmean(math.cos(angle) for angle in angles)
    length = math.hypot(mean_sin, mean_cos)
    if length <= _EPSILON:
        return None, None
    return mean_sin / length, mean_cos / length


def _circular_regularity(hours: Sequence[float]) -> float | None:
    if len(hours) < 2:
        return None
    angles = [2.0 * math.pi * (float(hour) % 24.0) / 24.0 for hour in hours]
    mean_sin = fmean(math.sin(angle) for angle in angles)
    mean_cos = fmean(math.cos(angle) for angle in angles)
    return min(max(math.hypot(mean_sin, mean_cos), 0.0), 1.0)


def _sleep_midpoint_hour(onset: float, wake: float) -> float:
    forward = (float(wake) - float(onset)) % 24.0
    return (float(onset) + forward / 2.0) % 24.0


def _bounded(value: float | None, minimum: float, maximum: float) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if (
        not math.isfinite(numeric)
        or numeric < minimum - 1e-12
        or numeric > maximum + 1e-12
    ):
        raise NhanesAdapterError("derived NHANES feature is outside its frozen range")
    return min(max(numeric, minimum), maximum)


def _code(value: Any) -> int | None:
    if value is None or bool(pd.isna(value)):
        return None
    numeric = float(value)
    if abs(numeric) < _EPSILON:
        numeric = 0.0
    if not math.isfinite(numeric) or numeric != math.floor(numeric):
        raise NhanesAdapterError("NHANES demographic code is not an integer")
    return int(numeric)


def _set_feature(record: dict[str, Any], target: str, value: Any) -> None:
    record[target] = value
    record[f"feature_mask.{target}"] = int(value is not None)


def _derive_activity(record: dict[str, Any], windows: pd.DataFrame) -> None:
    m10 = _finite_values(windows["M10VALUE"])
    activity_rows = windows.loc[windows[["M10VALUE", "L5VALUE"]].notna().any(axis=1)]
    paired = windows.loc[windows[["M10VALUE", "L5VALUE"]].notna().all(axis=1)]
    relative_amplitude: float | None = None
    if not paired.empty:
        m10_mean = fmean(_finite_values(paired["M10VALUE"]))
        l5_mean = fmean(_finite_values(paired["L5VALUE"]))
        denominator = m10_mean + l5_mean
        if denominator > _EPSILON:
            relative_amplitude = _bounded((m10_mean - l5_mean) / denominator, 0.0, 1.0)
    variability = (
        pstdev(m10) / abs(fmean(m10))
        if len(m10) >= 2 and abs(fmean(m10)) > _EPSILON
        else None
    )
    valid_days = len(activity_rows)
    _set_feature(record, "activity.relative_amplitude", relative_amplitude)
    _set_feature(record, "activity.activity_variability", variability)
    _set_feature(record, "activity.valid_days", valid_days if valid_days else None)
    available_semantics = sum(
        value is not None for value in (relative_amplitude, variability)
    )
    coverage = (
        min(max((valid_days / 7.0) * (available_semantics / 7.0), 0.0), 1.0)
        if valid_days
        else None
    )
    _set_feature(record, "activity.feature_coverage", coverage)


def _derive_sleep(record: dict[str, Any], windows: pd.DataFrame) -> None:
    sleep_fields = (
        "dur_spt_sleep_min",
        "dur_spt_min",
        "sleep_efficiency",
        "sleeponset",
        "wakeup",
        "dur_spt_wake_IN_min",
    )
    sleep_rows = windows.loc[windows[list(sleep_fields)].notna().any(axis=1)]
    sleep_minutes = _finite_values(sleep_rows["dur_spt_sleep_min"])
    period_minutes = _finite_values(sleep_rows["dur_spt_min"])
    if any(value < 0.0 or value > _MINUTES_PER_DAY for value in sleep_minutes):
        raise NhanesAdapterError("NHANES sleep duration is outside 0..1440 minutes")
    if any(value <= 0.0 or value > _MINUTES_PER_DAY for value in period_minutes):
        raise NhanesAdapterError("NHANES sleep period is outside (0,1440] minutes")
    duration_norm = fmean(sleep_minutes) / _MINUTES_PER_DAY if sleep_minutes else None
    period_norm = fmean(period_minutes) / _MINUTES_PER_DAY if period_minutes else None

    pooled = sleep_rows.loc[
        sleep_rows[["dur_spt_sleep_min", "dur_spt_min"]].notna().all(axis=1)
        & sleep_rows["dur_spt_min"].gt(0)
    ]
    efficiency: float | None = None
    if not pooled.empty:
        efficiency = float(pooled["dur_spt_sleep_min"].sum()) / float(
            pooled["dur_spt_min"].sum()
        )
    else:
        direct_efficiency = _finite_values(sleep_rows["sleep_efficiency"])
        if direct_efficiency:
            efficiency = fmean(direct_efficiency)
    efficiency = _bounded(efficiency, 0.0, 1.0)

    onset = _finite_values(sleep_rows["sleeponset"])
    wake = _finite_values(sleep_rows["wakeup"])
    if any(value < 0.0 for value in (*onset, *wake)):
        raise NhanesAdapterError("NHANES sleep clock hour must be non-negative")
    paired_clock = sleep_rows.loc[
        sleep_rows[["sleeponset", "wakeup"]].notna().all(axis=1)
    ]
    midpoint = [
        _sleep_midpoint_hour(float(row.sleeponset), float(row.wakeup))
        for row in paired_clock.itertuples(index=False)
    ]
    onset_sin, onset_cos = _circular_components(onset)
    wake_sin, wake_cos = _circular_components(wake)
    midpoint_sin, midpoint_cos = _circular_components(midpoint)

    fragmentation_rows = sleep_rows.loc[
        sleep_rows[["dur_spt_wake_IN_min", "dur_spt_min"]].notna().all(axis=1)
        & sleep_rows["dur_spt_min"].gt(0)
    ]
    fragmentation_values = (
        fragmentation_rows["dur_spt_wake_IN_min"] / fragmentation_rows["dur_spt_min"]
    ).tolist()
    fragmentation = (
        fmean(float(value) for value in fragmentation_values)
        if fragmentation_values
        else None
    )
    if fragmentation is not None and fragmentation < 0.0:
        raise NhanesAdapterError("NHANES sleep fragmentation must be non-negative")

    regularities = [
        value
        for value in (
            _circular_regularity(onset),
            _circular_regularity(wake),
            _circular_regularity(midpoint),
        )
        if value is not None
    ]
    regularity = fmean(regularities) if regularities else None

    direct_values = {
        "sleep.sleep_duration_norm": _bounded(duration_norm, 0.0, 1.0),
        "sleep.time_in_bed_norm": _bounded(period_norm, 0.0, 1.0),
        "sleep.sleep_efficiency": efficiency,
        "sleep.sleep_onset_sin": onset_sin,
        "sleep.sleep_onset_cos": onset_cos,
        "sleep.wake_time_sin": wake_sin,
        "sleep.wake_time_cos": wake_cos,
        "sleep.sleep_midpoint_sin": midpoint_sin,
        "sleep.sleep_midpoint_cos": midpoint_cos,
        "sleep.sleep_fragmentation": fragmentation,
        "sleep.sleep_regularity": regularity,
    }
    for target, value in direct_values.items():
        _set_feature(record, target, value)

    valid_nights = len(sleep_rows)
    _set_feature(record, "sleep.valid_nights", valid_nights if valid_nights else None)
    regularity_support = valid_nights if regularities else 0
    coverage_cells = (
        int(sleep_rows["dur_spt_sleep_min"].notna().sum()),
        int(sleep_rows["dur_spt_min"].notna().sum()),
        int(sleep_rows["sleep_efficiency"].notna().sum()),
        len(onset),
        len(wake),
        len(midpoint),
        len(fragmentation_values),
        regularity_support,
        0,
    )
    coverage = (
        min(max(sum(coverage_cells) / (7.0 * len(coverage_cells)), 0.0), 1.0)
        if valid_nights
        else None
    )
    _set_feature(record, "sleep.feature_coverage", coverage)


def _derive_social_context(
    record: dict[str, Any], demographic: Mapping[str, Any]
) -> None:
    age = _code(demographic["RIDAGEYR"])
    if age is None or age < 60:
        raise NhanesAdapterError("canonical NHANES participant is younger than 60")
    age_group = "60_69" if age < 70 else "70_79" if age < 80 else "80_plus"
    sex = {1: "male", 2: "female"}.get(_code(demographic["RIAGENDR"]))
    marital_code = _code(demographic["DMDMARTL"])
    marital = (
        "partnered"
        if marital_code in {1, 6}
        else "not_partnered"
        if marital_code in {2, 3, 4, 5}
        else None
    )
    education_code = _code(demographic["DMDEDUC2"])
    education = (
        "primary_or_less"
        if education_code == 1
        else "middle"
        if education_code in {2, 3}
        else "high_or_above"
        if education_code in {4, 5}
        else None
    )
    for target, value in {
        "social_context.age_group": age_group,
        "social_context.sex": sex,
        "social_context.marital_status": marital,
        "social_context.education_level": education,
    }.items():
        _set_feature(record, target, value)


def _empty_feature_record() -> dict[str, Any]:
    record: dict[str, Any] = {}
    for spec in _target_specs(feature_schema_manifest()):
        target = _feature_column(str(spec["group"]), str(spec["name"]))
        record[target] = None
        record[f"feature_mask.{target}"] = 0
    return record


def _severity(total: int) -> int:
    if total <= 4:
        return 0
    if total <= 9:
        return 1
    if total <= 14:
        return 2
    if total <= 19:
        return 3
    return 4


def _assemble_canonical(
    demographics_by_cycle: Mapping[str, pd.DataFrame],
    dpq_by_cycle: Mapping[str, pd.DataFrame],
    pam: pd.DataFrame,
) -> _Assembly:
    if set(demographics_by_cycle) != set(CYCLES) or set(dpq_by_cycle) != set(CYCLES):
        raise NhanesAdapterError("NHANES adapter requires separate G and H tables")

    demographics: dict[str, pd.DataFrame] = {}
    dpq_tables: dict[str, pd.DataFrame] = {}
    cycle_members: dict[str, set[int]] = {}
    invalid_counts = {field: 0 for field in DPQ_FIELDS}
    for cycle in CYCLES:
        demo = _prepare_participant_table(
            demographics_by_cycle[cycle],
            required=DEMOGRAPHIC_FIELDS,
            table_name=f"DEMO_{cycle}",
        )
        dpq, cycle_invalid = _normalise_dpq(dpq_by_cycle[cycle], cycle=cycle)
        demographics[cycle] = demo
        dpq_tables[cycle] = dpq
        cycle_members[cycle] = set(demo["SEQN"].astype(int))
        for field, count in cycle_invalid.items():
            invalid_counts[field] += count

    full_pam, selected_pam, pam_audit = _prepare_pam(pam, cycle_members=cycle_members)
    full_ids = {
        (str(row.cycle), int(row.SEQN))
        for row in full_pam[["cycle", "SEQN"]].drop_duplicates().itertuples(index=False)
    }
    selected_groups = {
        (str(cycle), int(seqn)): group.copy()
        for (cycle, seqn), group in selected_pam.groupby(["cycle", "SEQN"], sort=False)
    }

    records: list[dict[str, Any]] = []
    flow_by_cycle: dict[str, dict[str, int]] = {}
    for cycle in CYCLES:
        demo = demographics[cycle].copy()
        age = _integral_series(
            demo["RIDAGEYR"], field=f"DEMO_{cycle}.RIDAGEYR", allow_missing=True
        )
        demo = demo.loc[age.ge(60).fillna(False)].copy()
        dpq = dpq_tables[cycle]
        complete_dpq = dpq.loc[dpq[list(DPQ_FIELDS)].notna().all(axis=1)].copy()
        eligible = demo.merge(
            complete_dpq, on="SEQN", how="inner", validate="one_to_one"
        )
        any_pam = eligible.loc[
            eligible["SEQN"].map(lambda value: (cycle, int(value)) in full_ids)
        ].copy()
        final = any_pam.loc[
            any_pam["SEQN"].map(lambda value: (cycle, int(value)) in selected_groups)
        ].copy()
        flow_by_cycle[cycle] = {
            "age_60_plus": len(demo),
            "age_60_plus_complete_dpq": len(eligible),
            "age_60_plus_complete_dpq_any_pam": len(any_pam),
            "age_60_plus_complete_dpq_valid_pam": len(final),
        }

        for row in final.itertuples(index=False):
            seqn = int(row.SEQN)
            participant_id = f"{cycle}:{seqn}"
            windows = selected_groups[(cycle, seqn)].sort_values(
                "window_number", kind="stable"
            )
            window_numbers = [int(value) for value in windows["window_number"]]
            record = {
                **_empty_feature_record(),
                "dataset_id": DATASET_ID,
                "global_participant_id": f"{DATASET_ID}::{participant_id}",
                "participant_id": participant_id,
                "cycle": cycle,
                "survey_cycle": SURVEY_CYCLES[cycle],
                "seqn": seqn,
                "timescale_semantics": TIMESCALE_SEMANTICS,
                "window_semantics": WINDOW_SEMANTICS,
                "window_number_values_json": json.dumps(
                    window_numbers, separators=(",", ":")
                ),
                "pam_valid_day_count": len(windows),
                "x_source_name": X_SOURCE_NAME,
            }
            total = 0
            for field in DPQ_FIELDS:
                value = int(getattr(row, field))
                record[field] = value
                total += value
            record["phq9_total"] = total
            record["phq9_severity"] = _severity(total)
            record["binary_target"] = int(total >= 10)
            m10 = _finite_values(windows["M10VALUE"])
            record["x_source_value"] = fmean(m10) if m10 else None
            record["x_source_mask"] = int(bool(m10))
            _derive_activity(record, windows)
            _derive_sleep(record, windows)
            _derive_social_context(record, row._asdict())
            records.append(record)

    canonical = pd.DataFrame.from_records(records)
    if canonical.empty:
        raise NhanesAdapterError("NHANES canonical population is empty")
    canonical = canonical.sort_values(["cycle", "seqn"], kind="stable").reset_index(
        drop=True
    )
    canonical = _coerce_canonical_dtypes(canonical)
    if bool(canonical.duplicated(["cycle", "seqn"]).any()):
        raise NhanesAdapterError("NHANES canonical cycle/SEQN key is not unique")
    if bool(canonical["global_participant_id"].duplicated().any()):
        raise NhanesAdapterError("NHANES global participant key is not unique")
    if canonical[ACTIVITY_VOLUME_COLUMN].notna().any() or bool(
        canonical[ACTIVITY_VOLUME_MASK_COLUMN].any()
    ):
        raise NhanesAdapterError("base NHANES canonical must not contain a fitted ECDF")
    _validate_feature_contract(canonical)
    final_keys = set(
        zip(canonical["cycle"].astype(str), canonical["seqn"].astype(int), strict=True)
    )
    final_window_rows = sum(len(selected_groups[key]) for key in final_keys)
    return _Assembly(
        canonical=canonical,
        flow_by_cycle=flow_by_cycle,
        dpq_invalid_counts=invalid_counts,
        pam_audit=pam_audit,
        final_window_rows=final_window_rows,
    )


def _validate_feature_contract(frame: pd.DataFrame) -> None:
    for spec in _target_specs(feature_schema_manifest()):
        target = _feature_column(str(spec["group"]), str(spec["name"]))
        mask = f"feature_mask.{target}"
        expected_mask = frame[target].notna().astype("int8")
        actual_mask = pd.to_numeric(frame[mask], errors="raise").astype("int8")
        if not actual_mask.equals(expected_mask):
            raise NhanesAdapterError(
                "NHANES feature mask does not match value missingness"
            )
        present = frame[target].dropna()
        if str(spec["value_type"]) == "category":
            if not set(present.astype(str)).issubset(set(spec["categories"])):
                raise NhanesAdapterError("NHANES category is outside MH-003 schema")
            continue
        numeric = pd.to_numeric(present, errors="raise").astype(float)
        minimum = spec.get("minimum")
        maximum = spec.get("maximum")
        if minimum is not None and bool((numeric < float(minimum)).any()):
            raise NhanesAdapterError("NHANES feature is below MH-003 minimum")
        if maximum is not None and bool((numeric > float(maximum)).any()):
            raise NhanesAdapterError("NHANES feature is above MH-003 maximum")


def _validate_frozen_statistics(assembly: _Assembly) -> None:
    canonical = assembly.canonical
    cycle_counts = {
        str(key): int(value)
        for key, value in canonical["cycle"].value_counts().sort_index().items()
    }
    severity_counts = {
        int(key): int(value)
        for key, value in canonical["phq9_severity"].value_counts().sort_index().items()
    }
    valid_day_counts = {
        int(key): int(value)
        for key, value in canonical["pam_valid_day_count"]
        .value_counts()
        .sort_index()
        .items()
    }
    observed = {
        "pam_rows": assembly.pam_audit.source_rows,
        "pre_quality_participants": sum(
            row["age_60_plus_complete_dpq_any_pam"]
            for row in assembly.flow_by_cycle.values()
        ),
        "canonical_participants": len(canonical),
        "cycle_counts": cycle_counts,
        "positive_rows": int(canonical["binary_target"].sum()),
        "severity_counts": severity_counts,
        "valid_day_counts": valid_day_counts,
        "same_date_groups": assembly.pam_audit.same_date_multiwindow_groups,
        "same_date_records": assembly.pam_audit.same_date_multiwindow_records,
    }
    expected = {
        "pam_rows": EXPECTED_PAM_ROWS,
        "pre_quality_participants": EXPECTED_PRE_QUALITY_PARTICIPANTS,
        "canonical_participants": EXPECTED_CANONICAL_PARTICIPANTS,
        "cycle_counts": EXPECTED_CYCLE_COUNTS,
        "positive_rows": EXPECTED_POSITIVE_ROWS,
        "severity_counts": EXPECTED_SEVERITY_COUNTS,
        "valid_day_counts": EXPECTED_VALID_DAY_COUNTS,
        "same_date_groups": EXPECTED_SAME_DATE_MULTIWINDOW_GROUPS,
        "same_date_records": EXPECTED_SAME_DATE_MULTIWINDOW_RECORDS,
    }
    if observed != expected:
        raise NhanesAdapterError("NHANES frozen aggregate statistics do not match")


def build_canonical_frame(
    demographics_by_cycle: Mapping[str, pd.DataFrame],
    dpq_by_cycle: Mapping[str, pd.DataFrame],
    pam: pd.DataFrame,
    *,
    enforce_frozen_statistics: bool = False,
) -> pd.DataFrame:
    """Build one cycle-namespaced row per 60+ participant with valid DPQ/PAM."""

    assembly = _assemble_canonical(demographics_by_cycle, dpq_by_cycle, pam)
    if enforce_frozen_statistics:
        _validate_frozen_statistics(assembly)
    return assembly.canonical


def _read_source_tables(
    source_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    demographics: dict[str, pd.DataFrame] = {}
    dpq: dict[str, pd.DataFrame] = {}
    try:
        for cycle in CYCLES:
            demographics[cycle] = pd.read_sas(
                source_root / f"DEMO_{cycle}.xpt", format="xport"
            )
            dpq[cycle] = pd.read_sas(source_root / f"DPQ_{cycle}.xpt", format="xport")
        pam = pd.read_csv(source_root / PAM_FILE_NAME)
    except (OSError, ValueError, TypeError) as exc:
        raise NhanesAdapterError("NHANES source tables are not readable") from exc
    return demographics, dpq, pam


def build_field_mapping() -> dict[str, Any]:
    schema = feature_schema_manifest()
    targets: list[dict[str, Any]] = []
    for spec in _target_specs(schema):
        group = str(spec["group"])
        name = str(spec["name"])
        target = _feature_column(group, name)
        if target == ACTIVITY_VOLUME_COLUMN:
            status = "fold_derived"
            source_fields = ["M10VALUE"]
            formula = "right_continuous_ecdf(count(train <= x) / n)"
            kind = "source_relative_rank_only"
            base_policy = "null_with_mask_0_until_explicit_training_fold_fit"
        elif target in _DIRECT_TARGET_RULES:
            rule = _DIRECT_TARGET_RULES[target]
            status = "mapped"
            source_fields = list(rule["source_fields"])
            formula = str(rule["formula"])
            kind = str(rule["mapping_kind"])
            base_policy = "mapped_when_required_source_is_valid_else_null_with_mask_0"
        else:
            status = "unsupported"
            source_fields = []
            formula = None
            kind = "no_strict_same_semantics_source"
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
                "mapping_status": status,
                "mapping_kind": kind,
                "source_fields": source_fields,
                "formula": formula,
                "base_canonical_policy": base_policy,
            }
        )

    source_contract = [
        {
            "table": "DEMO_G/H",
            "source_field": field,
            "role": "join_key" if field == "SEQN" else "production_mapping_source",
        }
        for field in DEMOGRAPHIC_FIELDS
    ]
    source_contract.extend(
        {
            "table": "DPQ_G/H",
            "source_field": field,
            "role": "label_only_prohibited_as_input",
        }
        for field in (*DPQ_FIELDS, DPQ_FUNCTION_FIELD)
    )
    source_contract.extend(
        {
            "table": PAM_FILE_NAME,
            "source_field": field,
            "role": (
                "join_or_window_audit"
                if field in {"SEQN", "calendar_date", "window_number", "excluded"}
                else "production_mapping_source"
            ),
        }
        for field in PAM_REQUIRED_FIELDS
    )
    return {
        "mapping_version": MAPPING_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "dataset_id": DATASET_ID,
        "feature_schema_version": schema["schema_version"],
        "feature_schema_sha256": MH003_FEATURE_SCHEMA_SHA256,
        "source_collection_sha256": NHANES_COLLECTION_SHA256,
        "file_manifest_sha256": DATA001_FILE_MANIFEST_SHA256,
        "timescale_semantics": TIMESCALE_SEMANTICS,
        "natural_dates_available": False,
        "participant_contract": {
            "key": ["cycle", "SEQN"],
            "global_key": "nhanes::{cycle}:{integer_seqn}",
            "cycle_values": SURVEY_CYCLES,
            "join_policy": "join participant and PAM tables inside each cycle only",
            "minimum_age_years": 60,
        },
        "pam_window_contract": {
            "valid_day": "excluded == 0",
            "record_key": ["cycle", "SEQN", "window_number"],
            "order": ["window_number ascending"],
            "maximum_selected_windows": 7,
            "calendar_date_role": "synthetic audit field; never used to order or deduplicate",
            "same_date_different_window_policy": "preserve as distinct records",
            "window_semantics": WINDOW_SEMANTICS,
        },
        "labels": {
            "required_complete_fields": list(DPQ_FIELDS),
            "valid_item_codes": [0, 1, 2, 3],
            "missing_item_codes": [7, 9],
            "total": "sum(DPQ010..DPQ090) only when all nine are valid",
            "excluded_from_total": DPQ_FUNCTION_FIELD,
            "binary_target": "phq9_total >= 10",
            "severity_bands": ["0-4", "5-9", "10-14", "15-19", "20-27"],
            "input_feature_use": "prohibited",
        },
        "x_source": {
            "source_field": "M10VALUE",
            "canonical_value_column": "x_source_value",
            "canonical_mask_column": "x_source_mask",
            "definition": "mean finite M10VALUE across selected valid PAM windows",
            "only_canonical_storage_of_participant_mean": True,
            "production_camera_equivalence": False,
            "base_activity_volume_policy": "not_fitted",
        },
        "source_fields": source_contract,
        "target_fields": targets,
        "base_supported_feature_set": sorted(_DIRECT_TARGET_RULES),
        "fold_supported_feature_set": [ACTIVITY_VOLUME_COLUMN],
        "unsupported_feature_set": [
            row["canonical_value_column"]
            for row in targets
            if row["mapping_status"] == "unsupported"
        ],
        "production_input_policy": {
            "allowed_columns": [
                row["canonical_value_column"]
                for row in targets
                if row["mapping_status"] in {"mapped", "fold_derived"}
            ],
            "cycle_is_input": False,
            "window_number_is_input": False,
            "dpq_and_targets_are_inputs": False,
            "survey_weights_are_inputs": False,
            "research_only_fields_are_inputs": False,
            "unmapped_target_policy": "null_with_feature_mask_0",
        },
        "explicit_research_exclusions": [
            "DPQ100",
            "calendar_date",
            "guider",
            "mec4yr",
            "SDMVPSU",
            "SDMVSTRA",
            "CFQ fields",
            "other questionnaire outcomes",
        ],
        "ecdf_contract": {
            "version": ECDF_VERSION,
            "fit_scope": "explicit outer-training participants only",
            "weighting_unit": "one finite participant x_source value",
            "ties": "right_continuous",
            "formula": "count(training_values <= x) / finite_training_value_count",
            "split_id_required": True,
            "formal_instance_before_DATA_007": False,
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


def _missing_summary(series: pd.Series) -> dict[str, Any]:
    missing = int(series.isna().sum())
    total = len(series)
    return {
        "present_count": total - missing,
        "missing_count": missing,
        "missing_fraction": 0.0 if total == 0 else missing / total,
    }


def _quality_report(assembly: _Assembly, binding: SourceBinding) -> dict[str, Any]:
    canonical = assembly.canonical
    target_fields: dict[str, Any] = {}
    for spec in _target_specs(feature_schema_manifest()):
        target = _feature_column(str(spec["group"]), str(spec["name"]))
        mask = f"feature_mask.{target}"
        target_fields[target] = {
            **_missing_summary(canonical[target]),
            "mask_1_count": int(canonical[mask].sum()),
            "mask_0_count": int(canonical[mask].eq(0).sum()),
        }
    return {
        "report_version": QUALITY_REPORT_VERSION,
        "dataset_id": DATASET_ID,
        "input_binding": binding.report_dict(),
        "timescale_semantics": TIMESCALE_SEMANTICS,
        "natural_dates_available": False,
        "row_filter": {
            "cycle_flows": {
                cycle: dict(assembly.flow_by_cycle[cycle]) for cycle in CYCLES
            },
            "pre_quality_reference_count": sum(
                row["age_60_plus_complete_dpq_any_pam"]
                for row in assembly.flow_by_cycle.values()
            ),
            "canonical_participant_count": len(canonical),
            "rule": "age>=60; nine DPQ items in 0..3; at least one selected PAM window with excluded==0",
            "reference_count_used_as_filter_target": False,
        },
        "participant_summary": {
            "participant_count": len(canonical),
            "cycle_counts": {
                str(key): int(value)
                for key, value in canonical["cycle"].value_counts().sort_index().items()
            },
            "participant_values_written_to_report": False,
        },
        "pam_window_audit": {
            "source_rows": assembly.pam_audit.source_rows,
            "unique_cycle_seqn_window_keys": assembly.pam_audit.unique_window_keys,
            "same_calendar_date_multiwindow_groups": assembly.pam_audit.same_date_multiwindow_groups,
            "same_calendar_date_multiwindow_records": assembly.pam_audit.same_date_multiwindow_records,
            "same_calendar_date_records_silently_deduplicated": 0,
            "invalid_excluded_rows": assembly.pam_audit.invalid_excluded_rows,
            "valid_rows_before_seven_window_limit": assembly.pam_audit.valid_rows_before_limit,
            "selected_rows_all_participants": assembly.pam_audit.selected_rows_all_participants,
            "truncated_valid_rows": assembly.pam_audit.truncated_valid_rows,
            "selected_same_calendar_date_multiwindow_groups": assembly.pam_audit.selected_same_date_multiwindow_groups,
            "final_population_selected_window_rows": assembly.final_window_rows,
            "window_rule": WINDOW_SEMANTICS,
            "window_number_values_saved_per_participant": True,
        },
        "dpq": {
            "valid_item_codes": [0, 1, 2, 3],
            "invalid_or_missing_counts_by_item": dict(assembly.dpq_invalid_counts),
            "all_nine_required": True,
            "DPQ100_in_total": False,
            "dpq_or_label_fields_used_as_input_features": False,
        },
        "labels": {
            "binary_negative_count": int(canonical["binary_target"].eq(0).sum()),
            "binary_positive_count": int(canonical["binary_target"].eq(1).sum()),
            "severity_counts": {
                str(int(key)): int(value)
                for key, value in canonical["phq9_severity"]
                .value_counts()
                .sort_index()
                .items()
            },
        },
        "valid_day_distribution": {
            str(int(key)): int(value)
            for key, value in canonical["pam_valid_day_count"]
            .value_counts()
            .sort_index()
            .items()
        },
        "x_source": {
            "field": X_SOURCE_NAME,
            **_missing_summary(canonical["x_source_value"]),
            "participant_mean_stored_in_other_canonical_column": False,
            "base_canonical_ecdf_fitted": False,
        },
        "production_input_policy": {
            "cycle_window_and_identifiers_used_as_features": False,
            "survey_weights_or_research_fields_used_as_features": False,
            "unmapped_fields_are_null_with_mask_0": True,
        },
        "canonical_target_missingness_and_masks": target_fields,
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
        row_group_size=4096,
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
        raise NhanesAdapterError("output root must not overlap the source data tree")
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
    manifest_relative = "nhanes/artifact_manifest.json"
    if manifest_relative not in payloads:
        raise NhanesAdapterError("artifact payload set has no completion manifest")
    for relative in payloads:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise NhanesAdapterError("artifact relative path is unsafe")
    targets = {relative: destination / Path(relative) for relative in payloads}
    changes: dict[str, bytes] = {}
    original_exists: dict[str, bool] = {}
    for relative, path in targets.items():
        exists = path.is_file()
        original_exists[relative] = exists
        if exists and path.read_bytes() == payloads[relative]:
            continue
        if path.exists() and not path.is_file():
            raise NhanesAdapterError("artifact target exists but is not a file")
        if exists and not overwrite:
            raise NhanesAdapterError(
                f"refusing to overwrite differing artifact: {path.name}"
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


def build_nhanes_artifacts(
    *,
    workspace_root: Path | None = None,
    manifests_root: Path | None = None,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> BuildResult:
    """Validate frozen sources and publish deterministic DATA-004 artifacts."""

    binding = validate_frozen_inputs(workspace_root, manifests_root)
    destination = _validated_output_root(
        output_root
        or _algorithm_root(binding.workspace_root) / DEFAULT_OUTPUT_RELATIVE,
        binding,
    )
    demographics, dpq, pam = _read_source_tables(binding.source_root)
    assembly = _assemble_canonical(demographics, dpq, pam)
    _validate_frozen_statistics(assembly)
    mapping = build_field_mapping()
    quality = _quality_report(assembly, binding)
    revalidated = validate_frozen_inputs(binding.workspace_root, binding.manifests_root)
    if revalidated != binding:
        raise NhanesAdapterError(
            "frozen NHANES source binding changed while being read"
        )

    canonical = assembly.canonical
    canonical_bytes = _parquet_bytes(canonical)
    mapping_bytes = _canonical_json_bytes(mapping)
    quality_bytes = _canonical_json_bytes(quality)
    preliminary_hashes = {
        "nhanes/canonical_nhanes.parquet": _sha256_bytes(canonical_bytes),
        "nhanes/data_quality_report.json": _sha256_bytes(quality_bytes),
        "mappings/nhanes_v3_3_3_mapping.json": _sha256_bytes(mapping_bytes),
    }
    metadata = {
        "adapter_version": ADAPTER_VERSION,
        "canonical_column_order_version": CANONICAL_COLUMN_ORDER_VERSION,
        "dataset_id": DATASET_ID,
        "feature_schema_version": feature_schema_manifest()["schema_version"],
        "input_binding": binding.report_dict(),
        "timescale_semantics": TIMESCALE_SEMANTICS,
        "natural_dates_fabricated": False,
        "canonical_sort": ["cycle", "seqn"],
        "canonical_column_order": list(canonical.columns),
        "canonical_dtypes": {
            name: str(dtype) for name, dtype in canonical.dtypes.items()
        },
        "canonical_row_count": len(canonical),
        "canonical_participant_count": int(
            canonical["global_participant_id"].nunique()
        ),
        "canonical_frame_sha256": canonical_frame_sha256(canonical),
        "base_canonical_ecdf_fitted": False,
        "formal_ecdf_instance_created": False,
        "artifact_sha256_before_metadata": preliminary_hashes,
    }
    metadata_bytes = _canonical_json_bytes(metadata)
    artifact_payloads = {
        "nhanes/canonical_nhanes.parquet": canonical_bytes,
        "nhanes/data_quality_report.json": quality_bytes,
        "nhanes/adapter_metadata.json": metadata_bytes,
        "mappings/nhanes_v3_3_3_mapping.json": mapping_bytes,
    }
    artifact_rows = {
        relative: {"bytes": len(content), "sha256": _sha256_bytes(content)}
        for relative, content in sorted(artifact_payloads.items())
    }
    artifact_manifest = {
        "manifest_version": ARTIFACT_MANIFEST_VERSION,
        "dataset_id": DATASET_ID,
        "input_binding": binding.report_dict(),
        "artifacts": artifact_rows,
        "artifact_count": len(artifact_rows),
        "complete": True,
    }
    artifact_payloads["nhanes/artifact_manifest.json"] = _canonical_json_bytes(
        artifact_manifest
    )
    _publish_artifacts(destination, artifact_payloads, overwrite=overwrite)
    hashes = {
        relative: _sha256_bytes(content)
        for relative, content in sorted(artifact_payloads.items())
    }
    return BuildResult(
        output_root=destination,
        artifact_sha256=hashes,
        row_count=len(canonical),
        participant_count=int(canonical["global_participant_id"].nunique()),
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
        raise NhanesAdapterError(message) from exc
    if np.isinf(values).any():
        raise NhanesAdapterError(message)
    return values


def _validate_ecdf_frame(
    frame: pd.DataFrame,
    *,
    require_base_activity: bool,
    require_sorted: bool,
) -> np.ndarray:
    if tuple(frame.columns) != canonical_column_order() or frame.empty:
        raise NhanesAdapterError(
            "ECDF input is not a complete DATA-004 canonical frame"
        )
    if frame["dataset_id"].isna().any() or not frame["dataset_id"].eq(DATASET_ID).all():
        raise NhanesAdapterError("ECDF input frame has the wrong dataset ID")
    if (
        frame[["global_participant_id", "participant_id", "cycle", "seqn"]]
        .isna()
        .any(axis=None)
    ):
        raise NhanesAdapterError("ECDF input frame has a missing identity key")
    expected_participant = frame["cycle"].astype(str) + ":" + frame["seqn"].astype(str)
    if not frame["participant_id"].astype(str).equals(expected_participant):
        raise NhanesAdapterError("ECDF participant keys do not match cycle/SEQN")
    expected_global = DATASET_ID + "::" + expected_participant
    if not frame["global_participant_id"].astype(str).equals(expected_global):
        raise NhanesAdapterError("ECDF global participant keys do not match")
    keys = list(zip(frame["cycle"].astype(str), frame["seqn"].astype(int), strict=True))
    if len(keys) != len(set(keys)):
        raise NhanesAdapterError("ECDF cycle/SEQN keys must be unique")
    if require_sorted and keys != sorted(keys):
        raise NhanesAdapterError("ECDF canonical rows are not in frozen key order")
    if frame[list(DPQ_FIELDS)].isna().any(axis=None):
        raise NhanesAdapterError("ECDF DPQ labels must be complete")
    item_values = frame[list(DPQ_FIELDS)].astype(int)
    total = item_values.sum(axis=1).to_numpy(dtype="float64")
    if not np.array_equal(
        _numeric_array(frame["phq9_total"], message="ECDF PHQ-9 is invalid"), total
    ):
        raise NhanesAdapterError("ECDF PHQ-9 total does not match nine items")
    if not np.array_equal(
        _numeric_array(frame["binary_target"], message="ECDF target is invalid"),
        (total >= 10).astype("float64"),
    ):
        raise NhanesAdapterError("ECDF binary target does not match PHQ-9")
    if not frame["x_source_name"].eq(X_SOURCE_NAME).all():
        raise NhanesAdapterError("ECDF x_source name does not match")
    source = _numeric_array(frame["x_source_value"], message="ECDF x_source is invalid")
    source_mask = _numeric_array(
        frame["x_source_mask"], message="ECDF x_source mask is invalid"
    )
    if not np.array_equal(source_mask, (~np.isnan(source)).astype("float64")):
        raise NhanesAdapterError("ECDF x_source mask does not match")
    activity = _numeric_array(
        frame[ACTIVITY_VOLUME_COLUMN], message="ECDF activity volume is invalid"
    )
    activity_mask = _numeric_array(
        frame[ACTIVITY_VOLUME_MASK_COLUMN], message="ECDF activity mask is invalid"
    )
    if require_base_activity:
        if not np.isnan(activity).all() or bool(activity_mask.any()):
            raise NhanesAdapterError(
                "ECDF fit requires an unfitted base canonical frame"
            )
    else:
        finite = activity[~np.isnan(activity)]
        if ((finite < 0.0) | (finite > 1.0)).any() or not np.array_equal(
            activity_mask, (~np.isnan(activity)).astype("float64")
        ):
            raise NhanesAdapterError("ECDF activity volume or mask is invalid")
    _validate_feature_contract(frame)
    return source


@dataclass(frozen=True)
class TrainingFoldECDF:
    """Right-continuous ECDF fitted only on explicit outer-training participants."""

    split_id: str
    canonical_artifact_sha256: str
    canonical_frame_sha256: str
    training_participant_sha256: str
    training_participant_count: int
    sorted_training_values: tuple[float, ...]
    version: str = ECDF_VERSION
    adapter_version: str = ADAPTER_VERSION
    dataset_id: str = DATASET_ID
    source_collection_sha256: str = NHANES_COLLECTION_SHA256
    feature_schema_sha256: str = MH003_FEATURE_SCHEMA_SHA256
    source_field: str = "x_source_value"
    target_field: str = ACTIVITY_VOLUME_COLUMN
    target_mask_field: str = ACTIVITY_VOLUME_MASK_COLUMN

    def __post_init__(self) -> None:
        if not self.split_id.strip() or "\0" in self.split_id:
            raise NhanesAdapterError("ECDF split_id must be non-empty")
        if self.version != ECDF_VERSION or self.dataset_id != DATASET_ID:
            raise NhanesAdapterError("ECDF identity binding does not match DATA-004")
        if self.adapter_version != ADAPTER_VERSION:
            raise NhanesAdapterError("ECDF adapter binding does not match DATA-004")
        if self.source_collection_sha256 != NHANES_COLLECTION_SHA256:
            raise NhanesAdapterError("ECDF source collection binding does not match")
        if self.feature_schema_sha256 != MH003_FEATURE_SCHEMA_SHA256:
            raise NhanesAdapterError("ECDF feature schema binding does not match")
        if self.source_field != "x_source_value":
            raise NhanesAdapterError("ECDF source field binding does not match")
        if (
            self.target_field != ACTIVITY_VOLUME_COLUMN
            or self.target_mask_field != ACTIVITY_VOLUME_MASK_COLUMN
        ):
            raise NhanesAdapterError("ECDF target binding does not match")
        if self.training_participant_count <= 0:
            raise NhanesAdapterError("ECDF needs at least one training participant")
        if not _is_sha256(self.canonical_artifact_sha256) or not _is_sha256(
            self.canonical_frame_sha256
        ):
            raise NhanesAdapterError("ECDF canonical hash binding is invalid")
        if not _is_sha256(self.training_participant_sha256):
            raise NhanesAdapterError("ECDF training participant hash is invalid")
        if not self.sorted_training_values:
            raise NhanesAdapterError("ECDF needs at least one finite training value")
        if any(not math.isfinite(value) for value in self.sorted_training_values):
            raise NhanesAdapterError("ECDF training values must be finite")
        if tuple(sorted(self.sorted_training_values)) != self.sorted_training_values:
            raise NhanesAdapterError("ECDF training values must be sorted")

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
            raise NhanesAdapterError("ECDF fit requires a non-empty split_id")
        participant_ids = list(training_participant_ids)
        if not participant_ids or any(
            not isinstance(value, str)
            or not value.startswith(f"{DATASET_ID}::")
            or not value.removeprefix(f"{DATASET_ID}::")
            or "\0" in value
            for value in participant_ids
        ):
            raise NhanesAdapterError(
                "ECDF fit requires namespaced NHANES training participant IDs"
            )
        if len(participant_ids) != len(set(participant_ids)):
            raise NhanesAdapterError("ECDF training participant IDs must be unique")
        if not _is_sha256(canonical_artifact_sha256):
            raise NhanesAdapterError("ECDF fit requires a canonical artifact SHA-256")
        source = _validate_ecdf_frame(
            frame, require_base_activity=True, require_sorted=True
        )
        observed_frame_sha256 = canonical_frame_sha256(frame)
        if (
            not _is_sha256(expected_canonical_frame_sha256)
            or observed_frame_sha256 != expected_canonical_frame_sha256
        ):
            raise NhanesAdapterError("ECDF canonical frame SHA-256 does not match")
        present_ids = set(frame["global_participant_id"].astype(str))
        if not set(participant_ids).issubset(present_ids):
            raise NhanesAdapterError("ECDF training participants are absent from frame")
        selected = frame["global_participant_id"].isin(participant_ids).to_numpy()
        finite = source[selected]
        finite = finite[~np.isnan(finite)]
        if not len(finite):
            raise NhanesAdapterError("ECDF training fold has no finite source values")
        return cls(
            split_id=split_id.strip(),
            canonical_artifact_sha256=canonical_artifact_sha256,
            canonical_frame_sha256=observed_frame_sha256,
            training_participant_sha256=_participant_set_sha256(participant_ids),
            training_participant_count=len(participant_ids),
            sorted_training_values=tuple(float(value) for value in np.sort(finite)),
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        source = _validate_ecdf_frame(
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
        result[self.target_mask_field] = pd.array(present.astype("int8"), dtype="Int8")
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
        expected_keys = {
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
        if set(payload) != expected_keys:
            raise NhanesAdapterError("serialized ECDF fields do not match")
        values = tuple(float(value) for value in payload["sorted_training_values"])
        if int(payload["training_value_count"]) != len(values):
            raise NhanesAdapterError("serialized ECDF value count does not match")
        if (
            payload["definition"]
            != "count(training_values <= x) / training_value_count"
        ):
            raise NhanesAdapterError("serialized ECDF definition does not match")
        if payload["ties"] != "right_continuous":
            raise NhanesAdapterError("serialized ECDF tie policy does not match")
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
    "BuildResult",
    "CANONICAL_COLUMN_ORDER_VERSION",
    "DATA001_FILE_MANIFEST_SHA256",
    "DATASET_ID",
    "DPQ_FIELDS",
    "EXPECTED_CANONICAL_PARTICIPANTS",
    "EXPECTED_POSITIVE_ROWS",
    "MH003_FEATURE_SCHEMA_SHA256",
    "NHANES_COLLECTION_SHA256",
    "NhanesAdapterError",
    "SourceBinding",
    "TIMESCALE_SEMANTICS",
    "TrainingFoldECDF",
    "WINDOW_SEMANTICS",
    "build_canonical_frame",
    "build_field_mapping",
    "build_nhanes_artifacts",
    "canonical_arrow_schema",
    "canonical_column_order",
    "canonical_frame_sha256",
    "validate_frozen_inputs",
]
