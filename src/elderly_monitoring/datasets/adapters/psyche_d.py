"""PSYCHE-D V3.3.3 canonical adapter and fold-bound activity ECDF.

The source has publisher-defined nominal-month windows rather than row-level
natural dates. The base canonical table therefore preserves the longitudinal
proxy semantics and leaves activity volume unset until a training fold fits a
source-specific empirical distribution.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from elderly_monitoring.modules.mental_health.mood_social import (
    feature_schema_manifest,
)


DATASET_ID = "psyche_d"
ADAPTER_VERSION = "psyche-d-adapter-v3.3.3-data002"
CANONICAL_COLUMN_ORDER_VERSION = "psyche-d-canonical-columns-v1"
MAPPING_VERSION = "psyche-d-field-mapping-v3.3.3-data002"
QUALITY_REPORT_VERSION = "psyche-d-quality-report-v1"
ARTIFACT_MANIFEST_VERSION = "psyche-d-artifact-manifest-v1"
ECDF_VERSION = "training-fold-right-continuous-ecdf-v1"

SOURCE_RELATIVE = Path("数据集/心理/PSYCHE-D")
SOURCE_PARQUET_NAME = "anon_processed_df_parquet"
SOURCE_DICTIONARY_NAME = "Feature explanations - anonymized data - Sheet1.tsv"
SOURCE_WHITELIST_NAME = "p0_feature_mapping_v1.yaml"
DEFAULT_OUTPUT_RELATIVE = Path("data/processed/mental_health/mood_social/v3.3.3-r2")

PSYCHE_D_COLLECTION_SHA256 = (
    "c82371846e17ab879ed89fac556f1614aa9382a232c8fa986430d9559756c480"
)
DATA001_FILE_MANIFEST_SHA256 = (
    "6591374bb4273c073126c1ed750c75bd852e08bddb2104797bcf893aec8f11ff"
)
MH003_FEATURE_SCHEMA_SHA256 = (
    "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
)

EXPECTED_SOURCE_ROWS = 35_694
EXPECTED_SOURCE_COLUMNS = 154
EXPECTED_CANONICAL_ROWS = 10_866
EXPECTED_CANONICAL_PARTICIPANTS = 4_036
EXPECTED_POSITIVE_ROWS = 3_444
EXPECTED_POSITIVE_PARTICIPANTS = 1_790
EXPECTED_END_CATEGORY_COUNTS = {0: 4_202, 1: 3_220, 2: 1_941, 3: 981, 4: 522}

LABEL_FIELDS = (
    "phq9_score_start",
    "phq9_score_end",
    "phq9_cat_start",
    "phq9_cat_end",
)

PSYCHE_D_SOURCE_FIELDS = (
    "steps_awake_mean",
    "steps_mvpa_iqr",
    "steps_lpa_iqr",
    "steps_awake_sum_iqr",
    "steps__active_day_count_",
    "steps__sedentary_day_count_",
    "steps_mvpa_sum_recent",
    "steps_lpa_sum_recent",
    "steps_rolling_6_median_recent",
    "steps_rolling_6_max_recent",
    "sleep_asleep_weekday_mean",
    "sleep_asleep_weekend_mean",
    "sleep_in_bed_weekday_mean",
    "sleep_in_bed_weekend_mean",
    "sleep_ratio_asleep_in_bed_weekday_mean",
    "sleep_ratio_asleep_in_bed_weekend_mean",
    "sleep_in_bed_iqr",
    "sleep_asleep_iqr",
    "sleep_ratio_asleep_in_bed_iqr",
    "sleep_main_start_hour_adj_median",
    "sleep_main_start_hour_adj_iqr",
    "sleep_main_start_hour_adj_range",
    "sleep__hypersomnia_count_",
    "sleep__hyposomnia_count_",
    "sleep_asleep_mean_recent",
    "sleep_in_bed_mean_recent",
    "sleep_ratio_asleep_in_bed_mean_recent",
)

IDENTITY_COLUMNS = (
    "dataset_id",
    "global_participant_id",
    "participant_id",
    "nominal_month",
    "timescale_semantics",
)
TARGET_COLUMNS = (*LABEL_FIELDS, "binary_target")
X_SOURCE_COLUMNS = ("x_source_name", "x_source_value", "x_source_mask")
X_SOURCE_FIELD = "steps_awake_mean"
X_SOURCE_CANONICAL_COLUMN = f"source__{X_SOURCE_FIELD}"
ACTIVITY_VOLUME_COLUMN = "activity.activity_volume_norm"
ACTIVITY_VOLUME_MASK_COLUMN = "feature_mask.activity.activity_volume_norm"
TIMESCALE_SEMANTICS = "timescale_proxy_transfer"

_DIRECT_TARGET_RULES: dict[str, dict[str, Any]] = {
    "sleep.sleep_duration_norm": {
        "source_fields": ["sleep_asleep_mean_recent"],
        "formula": "sleep_asleep_mean_recent / 1440",
        "mapping_kind": "derived_same_semantics",
    },
    "sleep.time_in_bed_norm": {
        "source_fields": ["sleep_in_bed_mean_recent"],
        "formula": "sleep_in_bed_mean_recent / 1440",
        "mapping_kind": "derived_same_semantics",
    },
    "sleep.sleep_efficiency": {
        "source_fields": ["sleep_ratio_asleep_in_bed_mean_recent"],
        "formula": "sleep_ratio_asleep_in_bed_mean_recent",
        "mapping_kind": "direct_same_semantics",
    },
    "sleep.sleep_onset_sin": {
        "source_fields": ["sleep_main_start_hour_adj_median"],
        "formula": "sin(2*pi*((adjusted_hour mod 24)*60)/1440)",
        "mapping_kind": "derived_circular_same_semantics",
    },
    "sleep.sleep_onset_cos": {
        "source_fields": ["sleep_main_start_hour_adj_median"],
        "formula": "cos(2*pi*((adjusted_hour mod 24)*60)/1440)",
        "mapping_kind": "derived_circular_same_semantics",
    },
}


class PsycheDAdapterError(ValueError):
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
    ordered = sorted(
        rows,
        key=lambda row: str(row["relative_path"]).encode("utf-8"),
    )
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
        if (candidate / SOURCE_RELATIVE).is_dir() and (
            candidate / "algorithm" / "eldercare-risk-ai-main"
        ).is_dir():
            return candidate
    raise PsycheDAdapterError(
        "workspace root containing the frozen dataset was not found"
    )


def _algorithm_root(workspace_root: Path) -> Path:
    path = workspace_root / "algorithm" / "eldercare-risk-ai-main"
    if not path.is_dir():
        raise PsycheDAdapterError("algorithm repository was not found in the workspace")
    return path


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PsycheDAdapterError(
            f"frozen JSON input is not readable: {path.name}"
        ) from exc


def _read_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise PsycheDAdapterError(
                            "DATA-001 file manifest contains a non-object row"
                        )
                    rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PsycheDAdapterError("DATA-001 file manifest is not readable") from exc
    return rows


def validate_frozen_inputs(
    workspace_root: Path | None = None,
    manifests_root: Path | None = None,
) -> SourceBinding:
    """Independently revalidate DATA-001 files and the live MH-003 schema."""

    root = _find_workspace_root(workspace_root or Path.cwd())
    algorithm_root = _algorithm_root(root)
    audit_root = (
        manifests_root or algorithm_root / DEFAULT_OUTPUT_RELATIVE / "manifests"
    )
    manifest_path = audit_root / "dataset_file_manifest.jsonl"
    schema_snapshot_path = audit_root / "feature_schema_manifest.json"

    if _sha256_file(manifest_path) != DATA001_FILE_MANIFEST_SHA256:
        raise PsycheDAdapterError("DATA-001 file manifest SHA-256 does not match")

    live_schema = feature_schema_manifest()
    live_schema_sha256 = _sha256_bytes(_canonical_json_bytes(live_schema))
    if live_schema_sha256 != MH003_FEATURE_SCHEMA_SHA256:
        raise PsycheDAdapterError("live MH-003 feature schema SHA-256 does not match")
    if _sha256_file(schema_snapshot_path) != MH003_FEATURE_SCHEMA_SHA256:
        raise PsycheDAdapterError(
            "DATA-001 feature schema snapshot SHA-256 does not match"
        )
    if _read_json(schema_snapshot_path) != live_schema:
        raise PsycheDAdapterError(
            "DATA-001 schema snapshot differs from live MH-003 schema"
        )

    manifest_rows = _read_manifest_rows(manifest_path)
    selected = [row for row in manifest_rows if row.get("dataset_id") == DATASET_ID]
    if len(selected) != 4:
        raise PsycheDAdapterError("DATA-001 PSYCHE-D file count does not match")

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
        raise PsycheDAdapterError("current PSYCHE-D file set differs from DATA-001")

    current_rows: list[dict[str, Any]] = []
    for row in selected:
        relative = str(row["relative_path"])
        path = root / Path(relative)
        if path.is_symlink():
            raise PsycheDAdapterError(
                "PSYCHE-D source files must not be symbolic links"
            )
        before = path.stat()
        file_sha256 = _sha256_file(path)
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise PsycheDAdapterError("PSYCHE-D source changed during validation")
        if int(row.get("bytes", -1)) != after.st_size:
            raise PsycheDAdapterError("PSYCHE-D source size differs from DATA-001")
        if str(row.get("sha256", "")) != file_sha256:
            raise PsycheDAdapterError("PSYCHE-D source SHA-256 differs from DATA-001")
        current_rows.append(
            {
                "relative_path": relative,
                "bytes": int(after.st_size),
                "sha256": file_sha256,
            }
        )

    collection_sha256 = _collection_sha256(current_rows)
    if collection_sha256 != PSYCHE_D_COLLECTION_SHA256:
        raise PsycheDAdapterError("PSYCHE-D collection SHA-256 does not match")

    return SourceBinding(
        workspace_root=root,
        source_root=source_root,
        manifests_root=audit_root,
        source_collection_sha256=collection_sha256,
        file_manifest_sha256=DATA001_FILE_MANIFEST_SHA256,
        feature_schema_sha256=live_schema_sha256,
        source_file_count=len(current_rows),
    )


def parse_sample_index(index: pd.Index) -> pd.DataFrame:
    """Split sample IDs at the final underscore without exposing values in errors."""

    participants: list[str] = []
    nominal_months: list[int] = []
    for value in index.tolist():
        if not isinstance(value, str):
            raise PsycheDAdapterError("PSYCHE-D sample index contains a non-string key")
        participant_id, separator, month_text = value.rpartition("_")
        if not separator or not participant_id or not month_text.isdigit():
            raise PsycheDAdapterError("PSYCHE-D sample index contains a malformed key")
        participants.append(participant_id)
        nominal_months.append(int(month_text))
    return pd.DataFrame(
        {
            "participant_id": pd.array(participants, dtype="string"),
            "nominal_month": pd.array(nominal_months, dtype="Int32"),
        },
        index=index,
    )


def _load_whitelist(source_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = source_root / SOURCE_WHITELIST_NAME
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PsycheDAdapterError("PSYCHE-D source whitelist is not readable") from exc
    if not isinstance(payload, dict):
        raise PsycheDAdapterError("PSYCHE-D source whitelist has an invalid root")
    features = payload.get("features")
    if not isinstance(features, dict):
        raise PsycheDAdapterError("PSYCHE-D source whitelist has no feature groups")
    records: list[dict[str, Any]] = []
    for group in ("steps", "sleep"):
        group_rows = features.get(group)
        if not isinstance(group_rows, list):
            raise PsycheDAdapterError("PSYCHE-D source whitelist group is invalid")
        for row in group_rows:
            if not isinstance(row, dict) or not isinstance(
                row.get("source_field"), str
            ):
                raise PsycheDAdapterError("PSYCHE-D source whitelist row is invalid")
            records.append(
                {
                    "source_field": row["source_field"],
                    "legacy_semantic_group": row.get("semantic_group"),
                }
            )
    if tuple(row["source_field"] for row in records) != PSYCHE_D_SOURCE_FIELDS:
        raise PsycheDAdapterError("PSYCHE-D 27-field source whitelist does not match")
    return records, payload


def _load_dictionary(source_root: Path) -> dict[str, dict[str, str]]:
    path = source_root / SOURCE_DICTIONARY_NAME
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PsycheDAdapterError("PSYCHE-D field dictionary is not readable") from exc
    dictionary: dict[str, dict[str, str]] = {}
    for row in rows:
        name = str(row.get("Feature name", ""))
        if not name or name in dictionary:
            raise PsycheDAdapterError(
                "PSYCHE-D field dictionary contains invalid names"
            )
        dictionary[name] = {
            "category": str(row.get("Category", "")),
            "subcategory": str(row.get("Subcategory", "")),
            "description": str(row.get("Description", "")),
            "notes": str(row.get("Notes", "")),
        }
    for name in (*PSYCHE_D_SOURCE_FIELDS, *LABEL_FIELDS):
        if name not in dictionary:
            raise PsycheDAdapterError("PSYCHE-D field dictionary is incomplete")
    return dictionary


def _target_specs(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in schema["group_order"]:
        for spec in schema["groups"][group]:
            rows.append({"group": group, **dict(spec)})
    if len(rows) != 54:
        raise PsycheDAdapterError("MH-003 windowed feature count does not match")
    return rows


def _feature_column(group: str, name: str) -> str:
    return f"{group}.{name}"


def _mask_column(group: str, name: str) -> str:
    return f"feature_mask.{group}.{name}"


def build_field_mapping(source_root: Path) -> dict[str, Any]:
    """Build the complete source and MH-003 target mapping contract."""

    whitelist, _ = _load_whitelist(source_root)
    dictionary = _load_dictionary(source_root)
    schema = feature_schema_manifest()
    target_specs = _target_specs(schema)

    source_uses: dict[str, list[str]] = {name: [] for name in PSYCHE_D_SOURCE_FIELDS}
    source_uses[X_SOURCE_FIELD].append("x_source_value")
    for target, rule in _DIRECT_TARGET_RULES.items():
        for name in rule["source_fields"]:
            source_uses[name].append(target)

    source_fields: list[dict[str, Any]] = []
    for whitelist_row in whitelist:
        name = whitelist_row["source_field"]
        uses = source_uses[name]
        source_fields.append(
            {
                **whitelist_row,
                **dictionary[name],
                "canonical_source_column": f"source__{name}",
                "canonical_uses": uses,
                "mapping_status": "used" if uses else "audit_only",
            }
        )

    targets: list[dict[str, Any]] = []
    for spec in target_specs:
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
            base_policy = "mapped_when_source_is_finite_else_null_with_mask_0"
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
        "source_collection_sha256": PSYCHE_D_COLLECTION_SHA256,
        "file_manifest_sha256": DATA001_FILE_MANIFEST_SHA256,
        "timescale_semantics": TIMESCALE_SEMANTICS,
        "natural_dates_available": False,
        "source_whitelist_role": (
            "27-field source whitelist only; legacy permission language is not authoritative"
        ),
        "labels": {
            "required_complete_fields": list(LABEL_FIELDS),
            "binary_target": "phq9_score_end >= 10",
            "input_feature_use": "prohibited",
        },
        "x_source": {
            "source_field": X_SOURCE_FIELD,
            "canonical_value_column": "x_source_value",
            "canonical_mask_column": "x_source_mask",
            "definition": "publisher mean number of steps taken while awake",
            "production_camera_equivalence": False,
            "base_activity_volume_policy": "not_fitted",
        },
        "source_fields": source_fields,
        "target_fields": targets,
        "base_supported_feature_set": sorted(target for target in _DIRECT_TARGET_RULES),
        "fold_supported_feature_set": [ACTIVITY_VOLUME_COLUMN],
        "unsupported_feature_set": [
            row["canonical_value_column"]
            for row in targets
            if row["mapping_status"] == "unsupported"
        ],
        "ecdf_contract": {
            "version": ECDF_VERSION,
            "fit_scope": "explicit outer-training participants only",
            "weighting_unit": "finite training window value",
            "ties": "right_continuous",
            "formula": "count(training_values <= x) / finite_training_value_count",
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


def _numeric_series(frame: pd.DataFrame, name: str) -> pd.Series:
    try:
        result = pd.to_numeric(frame[name], errors="raise").astype("Float64")
    except (KeyError, TypeError, ValueError) as exc:
        raise PsycheDAdapterError("PSYCHE-D required numeric field is invalid") from exc
    values = result.to_numpy(dtype="float64", na_value=np.nan)
    if np.isinf(values).any():
        raise PsycheDAdapterError("PSYCHE-D numeric fields must not contain infinity")
    return result


def _validate_label_series(series: pd.Series, *, maximum: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise").astype("Float64")
    values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    finite = values[~np.isnan(values)]
    if np.isinf(finite).any() or (finite != np.floor(finite)).any():
        raise PsycheDAdapterError("PSYCHE-D label fields must be finite integers")
    if ((finite < 0) | (finite > maximum)).any():
        raise PsycheDAdapterError("PSYCHE-D label field is outside its frozen range")
    return numeric


def _empty_target_series(length: int, value_type: str) -> pd.Series:
    if value_type == "float":
        return pd.Series(pd.array([pd.NA] * length, dtype="Float64"))
    if value_type == "integer":
        return pd.Series(pd.array([pd.NA] * length, dtype="Int64"))
    if value_type == "category":
        return pd.Series(pd.array([pd.NA] * length, dtype="string"))
    raise PsycheDAdapterError("MH-003 feature type is unsupported by the adapter")


def _validate_range(values: pd.Series, minimum: float, maximum: float) -> None:
    array = values.to_numpy(dtype="float64", na_value=np.nan)
    finite = array[~np.isnan(array)]
    if ((finite < minimum) | (finite > maximum)).any():
        raise PsycheDAdapterError("derived PSYCHE-D feature is outside MH-003 range")


def _derive_direct_targets(canonical: pd.DataFrame) -> None:
    duration = canonical["source__sleep_asleep_mean_recent"] / 1440.0
    time_in_bed = canonical["source__sleep_in_bed_mean_recent"] / 1440.0
    efficiency = canonical["source__sleep_ratio_asleep_in_bed_mean_recent"].copy()
    for values in (duration, time_in_bed, efficiency):
        _validate_range(values, 0.0, 1.0)

    direct = {
        "sleep.sleep_duration_norm": duration,
        "sleep.time_in_bed_norm": time_in_bed,
        "sleep.sleep_efficiency": efficiency,
    }
    for target, values in direct.items():
        canonical[target] = values.astype("Float64")
        canonical[f"feature_mask.{target}"] = values.notna().astype("int8")

    adjusted = canonical["source__sleep_main_start_hour_adj_median"]
    adjusted_array = adjusted.to_numpy(dtype="float64", na_value=np.nan)
    finite = adjusted_array[~np.isnan(adjusted_array)]
    if (finite < 0).any():
        raise PsycheDAdapterError("adjusted sleep onset hour must be non-negative")
    angle = 2.0 * math.pi * np.mod(adjusted_array, 24.0) / 24.0
    for target, values in (
        ("sleep.sleep_onset_sin", np.sin(angle)),
        ("sleep.sleep_onset_cos", np.cos(angle)),
    ):
        series = pd.Series(pd.array(values, dtype="Float64"))
        canonical[target] = series
        canonical[f"feature_mask.{target}"] = series.notna().astype("int8")


def canonical_column_order() -> tuple[str, ...]:
    schema = feature_schema_manifest()
    target_specs = _target_specs(schema)
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
        *(f"source__{name}" for name in PSYCHE_D_SOURCE_FIELDS),
        *feature_columns,
        *mask_columns,
    )


def canonical_arrow_schema() -> pa.Schema:
    """Return the fixed Arrow schema used for storage and content hashing."""

    field_types: dict[str, pa.DataType] = {
        "dataset_id": pa.large_string(),
        "global_participant_id": pa.large_string(),
        "participant_id": pa.large_string(),
        "nominal_month": pa.int32(),
        "timescale_semantics": pa.large_string(),
        "phq9_score_start": pa.int16(),
        "phq9_score_end": pa.int16(),
        "phq9_cat_start": pa.int8(),
        "phq9_cat_end": pa.int8(),
        "binary_target": pa.int8(),
        "x_source_name": pa.large_string(),
        "x_source_value": pa.float64(),
        "x_source_mask": pa.int8(),
    }
    field_types.update(
        {f"source__{name}": pa.float64() for name in PSYCHE_D_SOURCE_FIELDS}
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
            raise PsycheDAdapterError(
                "MH-003 feature type is unsupported by the canonical Arrow schema"
            ) from exc
        field_types[f"feature_mask.{target}"] = pa.int8()

    order = canonical_column_order()
    if set(field_types) != set(order):
        raise PsycheDAdapterError(
            "canonical Arrow schema columns do not match DATA-002"
        )
    return pa.schema([pa.field(name, field_types[name]) for name in order])


def _canonical_arrow_table(frame: pd.DataFrame) -> pa.Table:
    if tuple(frame.columns) != canonical_column_order():
        raise PsycheDAdapterError("canonical frame columns do not match DATA-002")
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
        raise PsycheDAdapterError(
            "canonical frame values do not match the DATA-002 Arrow schema"
        ) from exc


def build_canonical_frame(
    source: pd.DataFrame,
    *,
    enforce_frozen_statistics: bool = False,
) -> pd.DataFrame:
    """Create the deterministic base canonical table without fitting ECDF."""

    required = set(PSYCHE_D_SOURCE_FIELDS) | set(LABEL_FIELDS)
    if not required.issubset(source.columns):
        raise PsycheDAdapterError("PSYCHE-D source matrix is missing required fields")
    if not source.index.is_unique:
        raise PsycheDAdapterError("PSYCHE-D sample index must be unique")

    parsed = parse_sample_index(source.index)
    numeric_sources = {
        name: _numeric_series(source, name) for name in PSYCHE_D_SOURCE_FIELDS
    }
    label_numeric = {
        "phq9_score_start": _validate_label_series(
            source["phq9_score_start"], maximum=27
        ),
        "phq9_score_end": _validate_label_series(source["phq9_score_end"], maximum=27),
        "phq9_cat_start": _validate_label_series(source["phq9_cat_start"], maximum=4),
        "phq9_cat_end": _validate_label_series(source["phq9_cat_end"], maximum=4),
    }
    complete = pd.DataFrame(label_numeric).notna().all(axis=1)
    keep_positions = np.flatnonzero(complete.to_numpy())
    parsed_kept = parsed.iloc[keep_positions].reset_index(drop=True)

    canonical = pd.DataFrame(
        {
            "dataset_id": pd.array([DATASET_ID] * len(keep_positions), dtype="string"),
            "global_participant_id": pd.array(
                [
                    f"{DATASET_ID}::{participant}"
                    for participant in parsed_kept["participant_id"]
                ],
                dtype="string",
            ),
            "participant_id": parsed_kept["participant_id"].array,
            "nominal_month": parsed_kept["nominal_month"].array,
            "timescale_semantics": pd.array(
                [TIMESCALE_SEMANTICS] * len(keep_positions), dtype="string"
            ),
        }
    )
    for name in LABEL_FIELDS:
        values = label_numeric[name].iloc[keep_positions]
        dtype = "Int16" if "score" in name else "Int8"
        canonical[name] = pd.array(values, dtype=dtype)
    canonical["binary_target"] = (canonical["phq9_score_end"] >= 10).astype("int8")

    for name in PSYCHE_D_SOURCE_FIELDS:
        canonical[f"source__{name}"] = pd.array(
            numeric_sources[name].iloc[keep_positions], dtype="Float64"
        )
    canonical["x_source_name"] = pd.array(
        [X_SOURCE_FIELD] * len(canonical), dtype="string"
    )
    canonical["x_source_value"] = canonical[X_SOURCE_CANONICAL_COLUMN].copy()
    canonical["x_source_mask"] = canonical["x_source_value"].notna().astype("int8")

    schema = feature_schema_manifest()
    target_specs = _target_specs(schema)
    for spec in target_specs:
        group = str(spec["group"])
        name = str(spec["name"])
        canonical[_feature_column(group, name)] = _empty_target_series(
            len(canonical), str(spec["value_type"])
        )
        canonical[_mask_column(group, name)] = pd.Series(
            np.zeros(len(canonical), dtype=np.int8)
        )
    _derive_direct_targets(canonical)

    duplicate_key = canonical.duplicated(["participant_id", "nominal_month"])
    if bool(duplicate_key.any()):
        raise PsycheDAdapterError("parsed participant/nominal-month key is not unique")
    canonical = canonical.sort_values(
        ["participant_id", "nominal_month"], kind="stable"
    ).reset_index(drop=True)
    canonical = canonical.loc[:, list(canonical_column_order())]

    if canonical[ACTIVITY_VOLUME_COLUMN].notna().any() or int(
        canonical[ACTIVITY_VOLUME_MASK_COLUMN].sum()
    ):
        raise PsycheDAdapterError("base canonical table must not contain a fitted ECDF")

    if enforce_frozen_statistics:
        _validate_frozen_statistics(source, canonical)
    return canonical


def _validate_frozen_statistics(source: pd.DataFrame, canonical: pd.DataFrame) -> None:
    observed = {
        "source_rows": len(source),
        "canonical_rows": len(canonical),
        "participants": int(canonical["participant_id"].nunique()),
        "positive_rows": int(canonical["binary_target"].sum()),
        "positive_participants": int(
            canonical.groupby("participant_id", sort=False)["binary_target"].max().sum()
        ),
    }
    expected = {
        "source_rows": EXPECTED_SOURCE_ROWS,
        "canonical_rows": EXPECTED_CANONICAL_ROWS,
        "participants": EXPECTED_CANONICAL_PARTICIPANTS,
        "positive_rows": EXPECTED_POSITIVE_ROWS,
        "positive_participants": EXPECTED_POSITIVE_PARTICIPANTS,
    }
    if observed != expected:
        raise PsycheDAdapterError("PSYCHE-D frozen aggregate statistics do not match")
    category_counts = {
        int(key): int(value)
        for key, value in canonical["phq9_cat_end"].value_counts().items()
    }
    if category_counts != EXPECTED_END_CATEGORY_COUNTS:
        raise PsycheDAdapterError("PSYCHE-D end-category distribution does not match")


def _read_source_frame(source_root: Path) -> tuple[pd.DataFrame, int]:
    parquet_path = source_root / SOURCE_PARQUET_NAME
    try:
        parquet = pq.ParquetFile(parquet_path)
        pandas_metadata = json.loads(parquet.schema_arrow.metadata[b"pandas"])
        physical_index_columns = {
            value
            for value in pandas_metadata["index_columns"]
            if isinstance(value, str)
        }
        source_column_count = sum(
            1
            for column in pandas_metadata["columns"]
            if column["field_name"] not in physical_index_columns
        )
        frame = pd.read_parquet(
            parquet_path,
            columns=[*PSYCHE_D_SOURCE_FIELDS, *LABEL_FIELDS],
        )
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        pa.ArrowException,
    ) as exc:
        raise PsycheDAdapterError("PSYCHE-D source Parquet is not readable") from exc
    if source_column_count != EXPECTED_SOURCE_COLUMNS:
        raise PsycheDAdapterError("PSYCHE-D source column count does not match")
    return frame, source_column_count


def _missing_summary(series: pd.Series) -> dict[str, Any]:
    missing = int(series.isna().sum())
    total = int(len(series))
    return {
        "present_count": total - missing,
        "missing_count": missing,
        "missing_fraction": 0.0 if total == 0 else missing / total,
    }


def _quality_report(
    source: pd.DataFrame,
    source_column_count: int,
    canonical: pd.DataFrame,
    binding: SourceBinding,
) -> dict[str, Any]:
    source_fields = {
        name: {
            "full_source_population": _missing_summary(source[name]),
            "complete_label_population": _missing_summary(canonical[f"source__{name}"]),
        }
        for name in PSYCHE_D_SOURCE_FIELDS
    }
    target_fields: dict[str, Any] = {}
    schema = feature_schema_manifest()
    for spec in _target_specs(schema):
        target = _feature_column(str(spec["group"]), str(spec["name"]))
        mask = f"feature_mask.{target}"
        target_fields[target] = {
            **_missing_summary(canonical[target]),
            "mask_1_count": int(canonical[mask].sum()),
            "mask_0_count": int((canonical[mask] == 0).sum()),
        }

    start_counts = canonical["phq9_cat_start"].value_counts().sort_index()
    end_counts = canonical["phq9_cat_end"].value_counts().sort_index()
    return {
        "report_version": QUALITY_REPORT_VERSION,
        "dataset_id": DATASET_ID,
        "input_binding": binding.report_dict(),
        "timescale_semantics": TIMESCALE_SEMANTICS,
        "natural_dates_available": False,
        "row_filter": {
            "input_rows": int(len(source)),
            "complete_four_label_rows": int(len(canonical)),
            "excluded_incomplete_label_rows": int(len(source) - len(canonical)),
            "filter": "all four frozen PHQ-9 label fields are non-null",
        },
        "source_structure": {
            "row_count": int(len(source)),
            "column_count": int(source_column_count),
            "whitelisted_source_field_count": len(PSYCHE_D_SOURCE_FIELDS),
        },
        "participant_summary": {
            "participant_count": int(canonical["participant_id"].nunique()),
            "participants_with_at_least_one_positive_window": int(
                canonical.groupby("participant_id", sort=False)["binary_target"]
                .max()
                .sum()
            ),
            "participant_values_written_to_report": False,
        },
        "nominal_month_summary": {
            "minimum": int(canonical["nominal_month"].min()),
            "maximum": int(canonical["nominal_month"].max()),
            "unique_count": int(canonical["nominal_month"].nunique()),
            "natural_date_interpretation": "prohibited",
        },
        "label_summary": {
            "binary_definition": "phq9_score_end >= 10",
            "positive_window_count": int(canonical["binary_target"].sum()),
            "negative_window_count": int((canonical["binary_target"] == 0).sum()),
            "phq9_cat_start_counts": {
                str(int(key)): int(value) for key, value in start_counts.items()
            },
            "phq9_cat_end_counts": {
                str(int(key)): int(value) for key, value in end_counts.items()
            },
            "label_fields_used_as_input_features": False,
            "full_source_missingness": {
                name: _missing_summary(source[name]) for name in LABEL_FIELDS
            },
        },
        "x_source": {
            "field": X_SOURCE_FIELD,
            **_missing_summary(canonical["x_source_value"]),
            "base_canonical_ecdf_fitted": False,
        },
        "source_field_missingness": source_fields,
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
        raise PsycheDAdapterError("output root must not overlap the source data tree")
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
    manifest_relative = "psyche_d/artifact_manifest.json"
    if manifest_relative not in payloads:
        raise PsycheDAdapterError("artifact payload set has no completion manifest")
    for relative in payloads:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise PsycheDAdapterError("artifact relative path is unsafe")

    targets = {relative: destination / Path(relative) for relative in payloads}
    changes: dict[str, bytes] = {}
    original_exists: dict[str, bool] = {}
    for relative, path in targets.items():
        exists = path.is_file()
        original_exists[relative] = exists
        if exists and path.read_bytes() == payloads[relative]:
            continue
        if path.exists() and not path.is_file():
            raise PsycheDAdapterError("artifact target exists but is not a file")
        if exists and not overwrite:
            raise PsycheDAdapterError(
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
        for temporary_path in staged.values():
            temporary_path.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def build_psyche_d_artifacts(
    *,
    workspace_root: Path | None = None,
    manifests_root: Path | None = None,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> BuildResult:
    """Validate frozen inputs and write deterministic DATA-002 artifacts."""

    binding = validate_frozen_inputs(workspace_root, manifests_root)
    destination = _validated_output_root(
        output_root
        or _algorithm_root(binding.workspace_root) / DEFAULT_OUTPUT_RELATIVE,
        binding,
    )
    source, source_column_count = _read_source_frame(binding.source_root)
    _load_whitelist(binding.source_root)
    _load_dictionary(binding.source_root)
    canonical = build_canonical_frame(source, enforce_frozen_statistics=True)
    mapping = build_field_mapping(binding.source_root)
    quality = _quality_report(source, source_column_count, canonical, binding)
    revalidated_binding = validate_frozen_inputs(
        binding.workspace_root, binding.manifests_root
    )
    if revalidated_binding != binding:
        raise PsycheDAdapterError("frozen source binding changed while being read")

    canonical_bytes = _parquet_bytes(canonical)
    mapping_bytes = _canonical_json_bytes(mapping)
    quality_bytes = _canonical_json_bytes(quality)
    preliminary_hashes = {
        "psyche_d/canonical_psyche_d.parquet": _sha256_bytes(canonical_bytes),
        "psyche_d/data_quality_report.json": _sha256_bytes(quality_bytes),
        "mappings/psyche_d_v3_3_3_mapping.json": _sha256_bytes(mapping_bytes),
    }
    metadata = {
        "adapter_version": ADAPTER_VERSION,
        "canonical_column_order_version": CANONICAL_COLUMN_ORDER_VERSION,
        "dataset_id": DATASET_ID,
        "feature_schema_version": feature_schema_manifest()["schema_version"],
        "input_binding": binding.report_dict(),
        "timescale_semantics": TIMESCALE_SEMANTICS,
        "natural_dates_fabricated": False,
        "canonical_sort": ["participant_id", "nominal_month"],
        "canonical_column_order": list(canonical.columns),
        "canonical_dtypes": {
            name: str(dtype) for name, dtype in canonical.dtypes.items()
        },
        "canonical_row_count": len(canonical),
        "canonical_participant_count": int(canonical["participant_id"].nunique()),
        "canonical_frame_sha256": canonical_frame_sha256(canonical),
        "base_canonical_ecdf_fitted": False,
        "artifact_sha256_before_metadata": preliminary_hashes,
    }
    metadata_bytes = _canonical_json_bytes(metadata)
    artifact_payloads = {
        "psyche_d/canonical_psyche_d.parquet": canonical_bytes,
        "psyche_d/data_quality_report.json": quality_bytes,
        "psyche_d/adapter_metadata.json": metadata_bytes,
        "mappings/psyche_d_v3_3_3_mapping.json": mapping_bytes,
    }
    artifact_rows = {
        relative: {
            "bytes": len(content),
            "sha256": _sha256_bytes(content),
        }
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
    artifact_manifest_bytes = _canonical_json_bytes(artifact_manifest)
    artifact_payloads["psyche_d/artifact_manifest.json"] = artifact_manifest_bytes

    _publish_artifacts(destination, artifact_payloads, overwrite=overwrite)

    all_hashes = {
        relative: _sha256_bytes(content)
        for relative, content in sorted(artifact_payloads.items())
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


def canonical_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash a canonical frame's ordered Arrow representation."""

    table = _canonical_arrow_table(frame)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table, max_chunksize=max(len(table), 1))
    return _sha256_bytes(sink.getvalue().to_pybytes())


def _numeric_array(series: pd.Series, *, message: str) -> np.ndarray:
    try:
        values = pd.to_numeric(series, errors="raise").to_numpy(
            dtype="float64", na_value=np.nan
        )
    except (TypeError, ValueError) as exc:
        raise PsycheDAdapterError(message) from exc
    if np.isinf(values).any():
        raise PsycheDAdapterError(message)
    return values


def _require_values_and_mask(
    frame: pd.DataFrame,
    target: str,
    expected: np.ndarray,
) -> None:
    actual = _numeric_array(
        frame[target], message="ECDF canonical mapped values are invalid"
    )
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-12, equal_nan=True):
        raise PsycheDAdapterError("ECDF canonical mapped values do not match")
    mask_column = f"feature_mask.{target}"
    actual_mask = _numeric_array(
        frame[mask_column], message="ECDF canonical feature mask is invalid"
    )
    expected_mask = (~np.isnan(expected)).astype("float64")
    if not np.array_equal(actual_mask, expected_mask):
        raise PsycheDAdapterError("ECDF canonical feature mask does not match")


def _validate_ecdf_canonical_frame(
    frame: pd.DataFrame,
    *,
    require_base_activity: bool,
    require_sorted: bool,
) -> np.ndarray:
    if tuple(frame.columns) != canonical_column_order() or frame.empty:
        raise PsycheDAdapterError(
            "ECDF input is not a complete DATA-002 canonical frame"
        )
    if frame["dataset_id"].isna().any() or not frame["dataset_id"].eq(DATASET_ID).all():
        raise PsycheDAdapterError("ECDF input frame has the wrong dataset ID")
    if (
        frame[["global_participant_id", "participant_id", "nominal_month"]]
        .isna()
        .any(axis=None)
    ):
        raise PsycheDAdapterError("ECDF input frame has a missing identity key")
    participants = frame["participant_id"].astype(str)
    expected_global = DATASET_ID + "::" + participants
    if not frame["global_participant_id"].astype(str).equals(expected_global):
        raise PsycheDAdapterError("ECDF global participant keys do not match")
    months = _numeric_array(
        frame["nominal_month"], message="ECDF nominal months are invalid"
    )
    if np.isnan(months).any() or (months != np.floor(months)).any():
        raise PsycheDAdapterError("ECDF nominal months are invalid")
    keys = list(zip(participants.tolist(), months.astype(int).tolist(), strict=True))
    if len(keys) != len(set(keys)):
        raise PsycheDAdapterError("ECDF participant-month keys must be unique")
    if require_sorted and keys != sorted(keys):
        raise PsycheDAdapterError("ECDF canonical rows are not in frozen key order")

    if frame[list(LABEL_FIELDS)].isna().any(axis=None):
        raise PsycheDAdapterError("ECDF canonical labels must be complete")
    score_end = _numeric_array(
        frame["phq9_score_end"], message="ECDF canonical labels are invalid"
    )
    binary_target = _numeric_array(
        frame["binary_target"], message="ECDF binary target is invalid"
    )
    if not np.array_equal(binary_target, (score_end >= 10).astype("float64")):
        raise PsycheDAdapterError("ECDF binary target does not match PHQ-9")

    if not frame["x_source_name"].eq(X_SOURCE_FIELD).all():
        raise PsycheDAdapterError("ECDF x_source name does not match")
    raw_source = _numeric_array(
        frame[X_SOURCE_CANONICAL_COLUMN], message="ECDF source values are invalid"
    )
    x_source = _numeric_array(
        frame["x_source_value"], message="ECDF source values are invalid"
    )
    same_source = (np.isnan(raw_source) & np.isnan(x_source)) | (raw_source == x_source)
    if not bool(same_source.all()):
        raise PsycheDAdapterError("ECDF canonical source columns do not match")
    source_mask = _numeric_array(
        frame["x_source_mask"], message="ECDF x_source mask is invalid"
    )
    if not np.array_equal(source_mask, (~np.isnan(raw_source)).astype("float64")):
        raise PsycheDAdapterError("ECDF x_source mask does not match")

    asleep = _numeric_array(
        frame["source__sleep_asleep_mean_recent"],
        message="ECDF sleep source values are invalid",
    )
    in_bed = _numeric_array(
        frame["source__sleep_in_bed_mean_recent"],
        message="ECDF sleep source values are invalid",
    )
    efficiency = _numeric_array(
        frame["source__sleep_ratio_asleep_in_bed_mean_recent"],
        message="ECDF sleep source values are invalid",
    )
    adjusted = _numeric_array(
        frame["source__sleep_main_start_hour_adj_median"],
        message="ECDF sleep source values are invalid",
    )
    angle = 2.0 * math.pi * np.mod(adjusted, 24.0) / 24.0
    expected_direct = {
        "sleep.sleep_duration_norm": asleep / 1440.0,
        "sleep.time_in_bed_norm": in_bed / 1440.0,
        "sleep.sleep_efficiency": efficiency,
        "sleep.sleep_onset_sin": np.sin(angle),
        "sleep.sleep_onset_cos": np.cos(angle),
    }
    for target, expected in expected_direct.items():
        _require_values_and_mask(frame, target, expected)

    for spec in _target_specs(feature_schema_manifest()):
        target = _feature_column(str(spec["group"]), str(spec["name"]))
        if target in expected_direct:
            continue
        values = _numeric_array(
            frame[target], message="ECDF canonical feature values are invalid"
        )
        mask = _numeric_array(
            frame[f"feature_mask.{target}"],
            message="ECDF canonical feature mask is invalid",
        )
        if target == ACTIVITY_VOLUME_COLUMN and not require_base_activity:
            finite = values[~np.isnan(values)]
            if ((finite < 0.0) | (finite > 1.0)).any() or not np.array_equal(
                mask, (~np.isnan(values)).astype("float64")
            ):
                raise PsycheDAdapterError("ECDF activity volume values are invalid")
        elif not np.isnan(values).all() or bool(mask.any()):
            raise PsycheDAdapterError("ECDF unsupported canonical fields must be null")
    return raw_source


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
    source_collection_sha256: str = PSYCHE_D_COLLECTION_SHA256
    feature_schema_sha256: str = MH003_FEATURE_SCHEMA_SHA256
    source_field: str = X_SOURCE_CANONICAL_COLUMN
    target_field: str = ACTIVITY_VOLUME_COLUMN
    target_mask_field: str = ACTIVITY_VOLUME_MASK_COLUMN

    def __post_init__(self) -> None:
        if not self.split_id.strip() or "\0" in self.split_id:
            raise PsycheDAdapterError("ECDF split_id must be non-empty")
        if self.version != ECDF_VERSION or self.dataset_id != DATASET_ID:
            raise PsycheDAdapterError("ECDF identity binding does not match DATA-002")
        if self.adapter_version != ADAPTER_VERSION:
            raise PsycheDAdapterError("ECDF adapter binding does not match DATA-002")
        if self.source_collection_sha256 != PSYCHE_D_COLLECTION_SHA256:
            raise PsycheDAdapterError("ECDF source collection binding does not match")
        if self.feature_schema_sha256 != MH003_FEATURE_SCHEMA_SHA256:
            raise PsycheDAdapterError("ECDF feature schema binding does not match")
        if self.source_field != X_SOURCE_CANONICAL_COLUMN:
            raise PsycheDAdapterError("ECDF source field binding does not match")
        if self.target_field != ACTIVITY_VOLUME_COLUMN:
            raise PsycheDAdapterError("ECDF target field binding does not match")
        if self.target_mask_field != ACTIVITY_VOLUME_MASK_COLUMN:
            raise PsycheDAdapterError("ECDF target mask binding does not match")
        if self.training_participant_count <= 0:
            raise PsycheDAdapterError("ECDF needs at least one training participant")
        if not _is_sha256(self.canonical_artifact_sha256) or not _is_sha256(
            self.canonical_frame_sha256
        ):
            raise PsycheDAdapterError("ECDF canonical hash binding is invalid")
        if not _is_sha256(self.training_participant_sha256):
            raise PsycheDAdapterError("ECDF training participant hash is invalid")
        if not self.sorted_training_values:
            raise PsycheDAdapterError("ECDF needs at least one finite training value")
        if any(not math.isfinite(value) for value in self.sorted_training_values):
            raise PsycheDAdapterError("ECDF training values must be finite")
        if tuple(sorted(self.sorted_training_values)) != self.sorted_training_values:
            raise PsycheDAdapterError("ECDF training values must be sorted")

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
            raise PsycheDAdapterError("ECDF fit requires a non-empty split_id")
        participant_ids = list(training_participant_ids)
        if not participant_ids or any(
            not isinstance(value, str)
            or not value.startswith(f"{DATASET_ID}::")
            or not value.removeprefix(f"{DATASET_ID}::")
            or "\0" in value
            for value in participant_ids
        ):
            raise PsycheDAdapterError(
                "ECDF fit requires namespaced PSYCHE-D training participant IDs"
            )
        if len(participant_ids) != len(set(participant_ids)):
            raise PsycheDAdapterError("ECDF training participant IDs must be unique")
        if not _is_sha256(canonical_artifact_sha256):
            raise PsycheDAdapterError("ECDF fit requires a canonical artifact SHA-256")
        raw_source = _validate_ecdf_canonical_frame(
            frame, require_base_activity=True, require_sorted=True
        )
        observed_frame_sha256 = canonical_frame_sha256(frame)
        if (
            not _is_sha256(expected_canonical_frame_sha256)
            or observed_frame_sha256 != expected_canonical_frame_sha256
        ):
            raise PsycheDAdapterError("ECDF canonical frame SHA-256 does not match")
        present_ids = set(frame["global_participant_id"].astype(str))
        if not set(participant_ids).issubset(present_ids):
            raise PsycheDAdapterError(
                "ECDF training participants are absent from the frame"
            )
        selected = frame["global_participant_id"].isin(participant_ids)
        source_values = raw_source[selected.to_numpy()]
        finite_values = source_values[~np.isnan(source_values)]
        if not len(finite_values):
            raise PsycheDAdapterError("ECDF training fold has no finite source values")
        return cls(
            split_id=split_id.strip(),
            canonical_artifact_sha256=canonical_artifact_sha256,
            canonical_frame_sha256=observed_frame_sha256,
            training_participant_sha256=_participant_set_sha256(participant_ids),
            training_participant_count=len(participant_ids),
            sorted_training_values=tuple(
                float(value) for value in np.sort(finite_values)
            ),
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
            raise PsycheDAdapterError("serialized ECDF fields do not match")
        values = tuple(float(value) for value in payload["sorted_training_values"])
        if int(payload["training_value_count"]) != len(values):
            raise PsycheDAdapterError("serialized ECDF value count does not match")
        if (
            payload["definition"]
            != "count(training_values <= x) / training_value_count"
        ):
            raise PsycheDAdapterError("serialized ECDF definition does not match")
        if payload["ties"] != "right_continuous":
            raise PsycheDAdapterError("serialized ECDF tie policy does not match")
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
    "DATASET_ID",
    "DATA001_FILE_MANIFEST_SHA256",
    "EXPECTED_CANONICAL_PARTICIPANTS",
    "EXPECTED_CANONICAL_ROWS",
    "EXPECTED_END_CATEGORY_COUNTS",
    "EXPECTED_POSITIVE_ROWS",
    "EXPECTED_SOURCE_COLUMNS",
    "EXPECTED_SOURCE_ROWS",
    "LABEL_FIELDS",
    "MH003_FEATURE_SCHEMA_SHA256",
    "PSYCHE_D_COLLECTION_SHA256",
    "PSYCHE_D_SOURCE_FIELDS",
    "PsycheDAdapterError",
    "SourceBinding",
    "TIMESCALE_SEMANTICS",
    "TrainingFoldECDF",
    "build_canonical_frame",
    "build_field_mapping",
    "build_psyche_d_artifacts",
    "canonical_column_order",
    "canonical_frame_sha256",
    "parse_sample_index",
    "validate_frozen_inputs",
]
