"""DATA-006 NHANES 2005-2008 DPQ+SSQ V3.3.3 canonical adapter.

The source is a cross-sectional questionnaire. SSQ values are retained as
controlled source audit columns, but they are not treated as S10 call logs and
do not map to ``social_contact`` or PersonalTrend inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    feature_schema_manifest,
)


DATASET_ID = "nhanes_ssq_2005_2008"
ADAPTER_VERSION = "nhanes-ssq-adapter-v3.3.3-data006"
CANONICAL_COLUMN_ORDER_VERSION = "nhanes-ssq-canonical-columns-v1"
MAPPING_VERSION = "nhanes-ssq-field-mapping-v3.3.3-data006"
QUALITY_REPORT_VERSION = "nhanes-ssq-quality-report-v1"
ARTIFACT_MANIFEST_VERSION = "nhanes-ssq-artifact-manifest-v1"

SOURCE_RELATIVE = Path("数据集/心理/NHANES-SSQ-2005-2008")
SOURCE_MANIFEST_NAME = "source_manifest.json"
DEFAULT_OUTPUT_RELATIVE = Path("data/processed/mental_health/mood_social/v3.3.3")

SOURCE_MANIFEST_SHA256 = (
    "5d95ffd6e86bd83e2221d30a91fe727deac7eed1e6ed7744e1bf624569577baa"
)
SOURCE_COLLECTION_SHA256 = (
    "48b47b9904cbe91e679046e10ede1340b4401c1cb674fcaf1ff047cccfd31608"
)
DATA001_FILE_MANIFEST_SHA256 = (
    "59257b21fee44bd405e0be6f299d49ad945c6e6289002717f8769389dce32bd5"
)
MH003_FEATURE_SCHEMA_SHA256 = (
    "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
)

CYCLES: tuple[str, ...] = ("D", "E")
SURVEY_CYCLES = {"D": "2005-2006", "E": "2007-2008"}
TIMESCALE_SEMANTICS = "cross_sectional_concurrent_association"
PARTICIPANT_KEY_POLICY = "cycle_plus_integer_seqn_v1"

DPQ_FIELDS = tuple(f"DPQ0{index}0" for index in range(1, 10))
DPQ_FUNCTION_FIELD = "DPQ100"
SSQ_SUPPORT_SOURCE_FIELDS = tuple(f"SSQ021{letter}" for letter in "ABCDEFGHIJKLMN")
SSQ_FIELDS = (
    "SSQ011",
    *SSQ_SUPPORT_SOURCE_FIELDS,
    "SSQ031",
    "SSQ041",
    "SSD044",
    "SSQ051",
    "SSQ061",
)
DEMOGRAPHIC_COMMON_FIELDS = (
    "SEQN",
    "RIDAGEYR",
    "RIAGENDR",
    "DMDMARTL",
    "DMDEDUC2",
    "INDFMPIR",
)
DEMOGRAPHIC_SOURCE_COLUMNS = (
    "SEQN",
    "RIDAGEYR",
    "RIAGENDR",
    "DMDMARTL",
    "DMDEDUC2",
    "INDHHINC",
    "INDHHIN2",
    "INDFMPIR",
)
DEMOGRAPHIC_AUDIT_FIELDS = DEMOGRAPHIC_SOURCE_COLUMNS[1:]
SOURCE_FIELDS = (*DEMOGRAPHIC_AUDIT_FIELDS, DPQ_FUNCTION_FIELD, *SSQ_FIELDS)
INTEGER_SOURCE_FIELDS = tuple(field for field in SOURCE_FIELDS if field != "INDFMPIR")

IDENTITY_COLUMNS = (
    "dataset_id",
    "global_participant_id",
    "participant_id",
    "cycle",
    "survey_cycle",
    "seqn",
    "timescale_semantics",
    "participant_key_policy",
)
TARGET_COLUMNS = (*DPQ_FIELDS, "phq9_total", "phq9_severity", "binary_target")

EXPECTED_SOURCE_FILE_COUNT = 12
EXPECTED_SOURCE_TOTAL_BYTES = 9_762_095
EXPECTED_SOURCE_ROWS = {
    "D": {"DEMO": 10_348, "DPQ": 5_334, "SSQ": 3_056},
    "E": {"DEMO": 10_149, "DPQ": 5_995, "SSQ": 4_025},
}
EXPECTED_AGE_60_PLUS = {"D": 1_570, "E": 2_154}
EXPECTED_COMPLETE_DPQ = {"D": 4_799, "E": 5_415}
EXPECTED_CYCLE_COUNTS = {"D": 1_311, "E": 1_839}
EXPECTED_POSITIVE_BY_CYCLE = {"D": 59, "E": 126}
EXPECTED_POSITIVE_ROWS = 185
EXPECTED_SEVERITY_COUNTS = {0: 2_521, 1: 444, 2: 120, 3: 50, 4: 15}

BASE_SUPPORTED_FEATURE_SET = (
    "social_context.age_group",
    "social_context.sex",
    "social_context.marital_status",
    "social_context.education_level",
)

_EPSILON = 1e-12
_MAPPED_TARGETS: dict[str, dict[str, Any]] = {
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
        "formula": ("1 or 6=>partnered; 2,3,4,5=>not_partnered; 77/99/missing=>null"),
        "mapping_kind": "coded_same_semantics",
    },
    "social_context.education_level": {
        "source_fields": ["DMDEDUC2"],
        "formula": (
            "1=>primary_or_less; 2 or 3=>middle; "
            "4 or 5=>high_or_above; 7/9/missing=>null"
        ),
        "mapping_kind": "coded_same_semantics",
    },
}


class NhanesSsqAdapterError(ValueError):
    """Raised when frozen DATA-006 inputs or outputs are invalid."""


@dataclass(frozen=True)
class SourceBinding:
    workspace_root: Path
    source_root: Path
    manifests_root: Path
    source_manifest_sha256: str
    source_collection_sha256: str
    data001_file_manifest_sha256: str
    feature_schema_sha256: str
    source_file_count: int
    source_total_bytes: int

    def report_dict(self) -> dict[str, Any]:
        return {
            "all_bindings_validated": True,
            "data001_file_manifest_sha256": self.data001_file_manifest_sha256,
            "dataset_id": DATASET_ID,
            "feature_schema_sha256": self.feature_schema_sha256,
            "source_collection_sha256": self.source_collection_sha256,
            "source_file_count": self.source_file_count,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_relative": SOURCE_RELATIVE.as_posix(),
            "source_total_bytes": self.source_total_bytes,
        }


@dataclass(frozen=True)
class BuildResult:
    output_root: Path
    artifact_sha256: Mapping[str, str]
    row_count: int
    participant_count: int
    positive_row_count: int


@dataclass(frozen=True)
class _Assembly:
    canonical: pd.DataFrame
    flow_by_cycle: Mapping[str, Mapping[str, int]]
    dpq_invalid_counts: Mapping[str, Mapping[str, int]]


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
        rows, key=lambda value: str(value["relative_path"]).encode("utf-8")
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
    raise NhanesSsqAdapterError("workspace root was not found")


def _algorithm_root(workspace_root: Path) -> Path:
    root = workspace_root / "algorithm" / "eldercare-risk-ai-main"
    if not root.is_dir():
        raise NhanesSsqAdapterError("algorithm repository root was not found")
    return root


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NhanesSsqAdapterError(f"JSON input is not readable: {path.name}") from exc


def validate_frozen_inputs(
    workspace_root: Path | None = None,
    manifests_root: Path | None = None,
) -> SourceBinding:
    """Validate the new source manifest and unchanged DATA-001/MH-003 bindings."""

    workspace = _find_workspace_root(workspace_root or Path.cwd())
    algorithm_root = _algorithm_root(workspace)
    manifests = (
        manifests_root or algorithm_root / DEFAULT_OUTPUT_RELATIVE / "manifests"
    ).resolve()
    data001_manifest = manifests / "dataset_file_manifest.jsonl"
    schema_snapshot = manifests / "feature_schema_manifest.json"
    if _sha256_file(data001_manifest) != DATA001_FILE_MANIFEST_SHA256:
        raise NhanesSsqAdapterError("DATA-001 file manifest SHA-256 changed")

    live_schema = feature_schema_manifest()
    live_schema_sha256 = _sha256_bytes(_canonical_json_bytes(live_schema))
    if live_schema_sha256 != MH003_FEATURE_SCHEMA_SHA256:
        raise NhanesSsqAdapterError("live MH-003 feature schema SHA-256 changed")
    if _sha256_file(schema_snapshot) != MH003_FEATURE_SCHEMA_SHA256:
        raise NhanesSsqAdapterError("MH-003 schema snapshot SHA-256 changed")
    if _read_json(schema_snapshot) != live_schema:
        raise NhanesSsqAdapterError("MH-003 schema snapshot content changed")

    source_root = (workspace / SOURCE_RELATIVE).resolve()
    source_manifest_path = source_root / SOURCE_MANIFEST_NAME
    if _sha256_file(source_manifest_path) != SOURCE_MANIFEST_SHA256:
        raise NhanesSsqAdapterError("DATA-006 source manifest SHA-256 changed")
    source_manifest = _read_json(source_manifest_path)
    if not isinstance(source_manifest, dict):
        raise NhanesSsqAdapterError("DATA-006 source manifest is not an object")
    expected_header = {
        "dataset_id": DATASET_ID,
        "downloaded_on": "2026-07-31",
        "file_count": EXPECTED_SOURCE_FILE_COUNT,
        "manifest_version": "nhanes-ssq-source-manifest-v1",
        "participant_values_written": False,
        "source_collection_sha256": SOURCE_COLLECTION_SHA256,
        "source_relative": SOURCE_RELATIVE.as_posix(),
        "total_bytes": EXPECTED_SOURCE_TOTAL_BYTES,
    }
    for name, expected in expected_header.items():
        if source_manifest.get(name) != expected:
            raise NhanesSsqAdapterError("DATA-006 source manifest content changed")
    files = source_manifest.get("files")
    if not isinstance(files, list) or len(files) != EXPECTED_SOURCE_FILE_COUNT:
        raise NhanesSsqAdapterError("DATA-006 source file manifest is incomplete")

    current_rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    total_bytes = 0
    for row in files:
        if not isinstance(row, dict):
            raise NhanesSsqAdapterError("DATA-006 source file row is invalid")
        try:
            name = str(row["file_name"])
            relative = str(row["relative_path"])
            expected_bytes = int(row["bytes"])
            expected_sha256 = str(row["sha256"])
            url = str(row["url"])
            fields = row["fields"]
        except (KeyError, TypeError, ValueError) as exc:
            raise NhanesSsqAdapterError(
                "DATA-006 source file row is incomplete"
            ) from exc
        if name in seen_names:
            raise NhanesSsqAdapterError("DATA-006 source file name is duplicated")
        seen_names.add(name)
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.parent.as_posix() != SOURCE_RELATIVE.as_posix()
            or relative_path.name != name
        ):
            raise NhanesSsqAdapterError("DATA-006 source relative path is unsafe")
        if not url.startswith("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/"):
            raise NhanesSsqAdapterError("DATA-006 source URL is not official")
        if (
            not isinstance(fields, list)
            or int(row.get("field_count", -1)) != len(fields)
            or not all(isinstance(field, str) and field for field in fields)
        ):
            raise NhanesSsqAdapterError("DATA-006 source field audit is invalid")
        path = workspace / relative_path
        if not path.is_file():
            raise NhanesSsqAdapterError("DATA-006 official source file is missing")
        actual_bytes = path.stat().st_size
        actual_sha256 = _sha256_file(path)
        if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
            raise NhanesSsqAdapterError("DATA-006 official source file changed")
        total_bytes += actual_bytes
        current_rows.append(
            {
                "relative_path": relative,
                "bytes": actual_bytes,
                "sha256": actual_sha256,
            }
        )
    if total_bytes != EXPECTED_SOURCE_TOTAL_BYTES:
        raise NhanesSsqAdapterError("DATA-006 source byte total changed")
    if _collection_sha256(current_rows) != SOURCE_COLLECTION_SHA256:
        raise NhanesSsqAdapterError("DATA-006 source collection SHA-256 changed")

    return SourceBinding(
        workspace_root=workspace,
        source_root=source_root,
        manifests_root=manifests,
        source_manifest_sha256=SOURCE_MANIFEST_SHA256,
        source_collection_sha256=SOURCE_COLLECTION_SHA256,
        data001_file_manifest_sha256=DATA001_FILE_MANIFEST_SHA256,
        feature_schema_sha256=live_schema_sha256,
        source_file_count=len(current_rows),
        source_total_bytes=total_bytes,
    )


def _target_specs(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {"group": group, **dict(spec)}
        for group in schema["group_order"]
        for spec in schema["groups"][group]
    ]
    if len(rows) != 54:
        raise NhanesSsqAdapterError("MH-003 feature count does not match")
    return rows


def _feature_column(group: str, name: str) -> str:
    return f"{group}.{name}"


def _mask_column(group: str, name: str) -> str:
    return f"feature_mask.{group}.{name}"


def feature_columns() -> tuple[str, ...]:
    return tuple(
        _feature_column(str(spec["group"]), str(spec["name"]))
        for spec in _target_specs(feature_schema_manifest())
    )


def canonical_column_order() -> tuple[str, ...]:
    features = feature_columns()
    return (
        *IDENTITY_COLUMNS,
        *TARGET_COLUMNS,
        *(f"source__{name}" for name in SOURCE_FIELDS),
        *features,
        *(f"feature_mask.{name}" for name in features),
    )


def canonical_arrow_schema() -> pa.Schema:
    types: dict[str, pa.DataType] = {
        name: pa.large_string() for name in IDENTITY_COLUMNS
    }
    types["seqn"] = pa.int64()
    types.update({name: pa.int8() for name in TARGET_COLUMNS})
    types.update(
        {
            f"source__{name}": (
                pa.int64() if name in INTEGER_SOURCE_FIELDS else pa.float64()
            )
            for name in SOURCE_FIELDS
        }
    )
    value_types = {
        "float": pa.float64(),
        "integer": pa.int64(),
        "category": pa.large_string(),
    }
    for spec in _target_specs(feature_schema_manifest()):
        target = _feature_column(str(spec["group"]), str(spec["name"]))
        try:
            types[target] = value_types[str(spec["value_type"])]
        except KeyError as exc:
            raise NhanesSsqAdapterError("unsupported MH-003 feature type") from exc
        types[f"feature_mask.{target}"] = pa.int8()
    order = canonical_column_order()
    if set(order) != set(types):
        raise NhanesSsqAdapterError("canonical Arrow schema columns do not match")
    return pa.schema([pa.field(name, types[name]) for name in order])


def _canonical_arrow_table(frame: pd.DataFrame) -> pa.Table:
    if tuple(frame.columns) != canonical_column_order():
        raise NhanesSsqAdapterError("canonical frame columns do not match DATA-006")
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
        raise NhanesSsqAdapterError(
            "canonical values do not match the DATA-006 Arrow schema"
        ) from exc


def _coerce_canonical_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[:, list(canonical_column_order())].copy()
    for field in canonical_arrow_schema():
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            result[field.name] = pd.array(result[field.name], dtype="string")
        elif pa.types.is_int8(field.type):
            result[field.name] = pd.array(result[field.name], dtype="Int8")
        elif pa.types.is_int64(field.type):
            result[field.name] = pd.array(result[field.name], dtype="Int64")
        elif pa.types.is_float64(field.type):
            result[field.name] = pd.array(result[field.name], dtype="Float64")
        else:
            raise NhanesSsqAdapterError("canonical Arrow schema type is unsupported")
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
        raise NhanesSsqAdapterError(f"NHANES SSQ {field} is not numeric") from exc
    values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    if np.isinf(values).any():
        raise NhanesSsqAdapterError(f"NHANES SSQ {field} contains infinity")
    near_zero = np.isfinite(values) & (np.abs(values) < _EPSILON)
    values[near_zero] = 0.0
    return pd.Series(pd.array(values, dtype="Float64"), index=series.index)


def _integral_series(
    series: pd.Series,
    *,
    field: str,
    allow_missing: bool,
) -> pd.Series:
    numeric = _numeric_series(series, field=field)
    values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    finite = values[np.isfinite(values)]
    if (finite != np.floor(finite)).any():
        raise NhanesSsqAdapterError(f"NHANES SSQ {field} must contain integer codes")
    if not allow_missing and np.isnan(values).any():
        raise NhanesSsqAdapterError(f"NHANES SSQ {field} contains a missing key")
    return pd.Series(pd.array(values, dtype="Int64"), index=series.index)


def _prepare_participant_table(
    frame: pd.DataFrame,
    *,
    required: Sequence[str],
    table_name: str,
) -> pd.DataFrame:
    if not set(required).issubset(frame.columns):
        raise NhanesSsqAdapterError(
            f"NHANES SSQ {table_name} is missing required fields"
        )
    result = frame.loc[:, list(required)].copy()
    result["SEQN"] = _integral_series(
        result["SEQN"], field=f"{table_name}.SEQN", allow_missing=False
    )
    if bool(result["SEQN"].duplicated().any()):
        raise NhanesSsqAdapterError(f"NHANES SSQ {table_name} SEQN is not unique")
    return result


def _prepare_demographics(frame: pd.DataFrame, *, cycle: str) -> pd.DataFrame:
    income_field = "INDHHINC" if cycle == "D" else "INDHHIN2"
    prepared = _prepare_participant_table(
        frame,
        required=(*DEMOGRAPHIC_COMMON_FIELDS, income_field),
        table_name=f"DEMO_{cycle}",
    )
    output = prepared[["SEQN"]].copy()
    for field in ("RIDAGEYR", "RIAGENDR", "DMDMARTL", "DMDEDUC2"):
        output[field] = _integral_series(
            prepared[field], field=f"DEMO_{cycle}.{field}", allow_missing=True
        )
    output["INDHHINC"] = pd.array([pd.NA] * len(output), dtype="Int64")
    output["INDHHIN2"] = pd.array([pd.NA] * len(output), dtype="Int64")
    output[income_field] = _integral_series(
        prepared[income_field],
        field=f"DEMO_{cycle}.{income_field}",
        allow_missing=True,
    )
    output["INDFMPIR"] = _numeric_series(
        prepared["INDFMPIR"], field=f"DEMO_{cycle}.INDFMPIR"
    )
    return output.loc[:, list(DEMOGRAPHIC_SOURCE_COLUMNS)]


def _normalise_dpq(
    frame: pd.DataFrame, *, cycle: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    prepared = _prepare_participant_table(
        frame,
        required=("SEQN", *DPQ_FIELDS, DPQ_FUNCTION_FIELD),
        table_name=f"DPQ_{cycle}",
    )
    output = prepared[["SEQN"]].copy()
    invalid_counts: dict[str, int] = {}
    for field in DPQ_FIELDS:
        numeric = _numeric_series(prepared[field], field=f"DPQ_{cycle}.{field}")
        values = numeric.to_numpy(dtype="float64", na_value=np.nan)
        finite = values[np.isfinite(values)]
        if (finite != np.floor(finite)).any():
            raise NhanesSsqAdapterError("DPQ items must contain integer codes")
        valid = np.isin(values, (0.0, 1.0, 2.0, 3.0))
        invalid_counts[field] = int((~valid).sum())
        values[~valid] = np.nan
        output[field] = pd.array(values, dtype="Int8")
    output[DPQ_FUNCTION_FIELD] = _integral_series(
        prepared[DPQ_FUNCTION_FIELD],
        field=f"DPQ_{cycle}.{DPQ_FUNCTION_FIELD}",
        allow_missing=True,
    )
    return output, invalid_counts


def _prepare_ssq(frame: pd.DataFrame, *, cycle: str) -> pd.DataFrame:
    prepared = _prepare_participant_table(
        frame,
        required=("SEQN", *SSQ_FIELDS),
        table_name=f"SSQ_{cycle}",
    )
    output = prepared[["SEQN"]].copy()
    for field in SSQ_FIELDS:
        output[field] = _integral_series(
            prepared[field], field=f"SSQ_{cycle}.{field}", allow_missing=True
        )
    return output


def _empty_feature_record() -> dict[str, Any]:
    record: dict[str, Any] = {}
    for target in feature_columns():
        record[target] = None
        record[f"feature_mask.{target}"] = 0
    return record


def _value_or_none(value: Any) -> Any:
    return None if pd.isna(value) else value


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


def _set_feature(record: dict[str, Any], target: str, value: Any) -> None:
    record[target] = value
    record[f"feature_mask.{target}"] = int(value is not None)


def _derive_profile_features(
    record: dict[str, Any], demographic: Mapping[str, Any]
) -> None:
    age_value = _value_or_none(demographic["RIDAGEYR"])
    if age_value is None or int(age_value) < 60:
        raise NhanesSsqAdapterError("canonical participant is younger than 60")
    age = int(age_value)
    age_group = "60_69" if age < 70 else "70_79" if age < 80 else "80_plus"
    sex_code = _value_or_none(demographic["RIAGENDR"])
    sex = {1: "male", 2: "female"}.get(None if sex_code is None else int(sex_code))
    marital_code_value = _value_or_none(demographic["DMDMARTL"])
    marital_code = None if marital_code_value is None else int(marital_code_value)
    marital = (
        "partnered"
        if marital_code in {1, 6}
        else "not_partnered"
        if marital_code in {2, 3, 4, 5}
        else None
    )
    education_code_value = _value_or_none(demographic["DMDEDUC2"])
    education_code = None if education_code_value is None else int(education_code_value)
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


def _assemble_canonical(
    demographics_by_cycle: Mapping[str, pd.DataFrame],
    dpq_by_cycle: Mapping[str, pd.DataFrame],
    ssq_by_cycle: Mapping[str, pd.DataFrame],
) -> _Assembly:
    if (
        set(demographics_by_cycle) != set(CYCLES)
        or set(dpq_by_cycle) != set(CYCLES)
        or set(ssq_by_cycle) != set(CYCLES)
    ):
        raise NhanesSsqAdapterError("DATA-006 requires separate D and E tables")

    records: list[dict[str, Any]] = []
    flow_by_cycle: dict[str, dict[str, int]] = {}
    dpq_invalid_counts: dict[str, dict[str, int]] = {}
    for cycle in CYCLES:
        demographics = _prepare_demographics(demographics_by_cycle[cycle], cycle=cycle)
        dpq, invalid_counts = _normalise_dpq(dpq_by_cycle[cycle], cycle=cycle)
        ssq = _prepare_ssq(ssq_by_cycle[cycle], cycle=cycle)
        dpq_invalid_counts[cycle] = invalid_counts

        age_60_plus = demographics["RIDAGEYR"].ge(60).fillna(False)
        complete_dpq = dpq.loc[:, list(DPQ_FIELDS)].notna().all(axis=1)
        eligible_demo = demographics.loc[age_60_plus].copy()
        eligible_dpq = dpq.loc[complete_dpq].copy()
        age_dpq = eligible_demo.merge(
            eligible_dpq, on="SEQN", how="inner", validate="one_to_one"
        )
        final = age_dpq.merge(
            ssq, on="SEQN", how="inner", validate="one_to_one"
        ).sort_values("SEQN", kind="stable")
        flow_by_cycle[cycle] = {
            "demo_source_rows": len(demographics),
            "dpq_source_rows": len(dpq),
            "ssq_source_rows": len(ssq),
            "age_60_plus": int(age_60_plus.sum()),
            "complete_dpq_all_ages": int(complete_dpq.sum()),
            "age_60_plus_complete_dpq": len(age_dpq),
            "age_60_plus_complete_dpq_joined_ssq": len(final),
            "dropped_for_missing_ssq_join": len(age_dpq) - len(final),
        }

        for row in final.itertuples(index=False):
            seqn = int(row.SEQN)
            participant_id = f"{cycle}:{seqn}"
            record = {
                **_empty_feature_record(),
                "dataset_id": DATASET_ID,
                "global_participant_id": f"{DATASET_ID}::{participant_id}",
                "participant_id": participant_id,
                "cycle": cycle,
                "survey_cycle": SURVEY_CYCLES[cycle],
                "seqn": seqn,
                "timescale_semantics": TIMESCALE_SEMANTICS,
                "participant_key_policy": PARTICIPANT_KEY_POLICY,
            }
            total = 0
            for field in DPQ_FIELDS:
                value = int(getattr(row, field))
                record[field] = value
                total += value
            record["phq9_total"] = total
            record["phq9_severity"] = _severity(total)
            record["binary_target"] = int(total >= 10)
            row_values = row._asdict()
            for field in SOURCE_FIELDS:
                record[f"source__{field}"] = _value_or_none(row_values[field])
            _derive_profile_features(record, row_values)
            records.append(record)

    if not records:
        canonical = pd.DataFrame(columns=canonical_column_order())
        canonical = _coerce_canonical_dtypes(canonical)
    else:
        canonical = pd.DataFrame.from_records(records)
        canonical = canonical.sort_values(["cycle", "seqn"], kind="stable").reset_index(
            drop=True
        )
        canonical = _coerce_canonical_dtypes(canonical)
    if bool(canonical.duplicated(["cycle", "seqn"]).any()):
        raise NhanesSsqAdapterError("canonical cycle/SEQN key is not unique")
    if bool(canonical["global_participant_id"].duplicated().any()):
        raise NhanesSsqAdapterError("canonical global participant key is not unique")
    _validate_feature_contract(canonical)
    return _Assembly(
        canonical=canonical,
        flow_by_cycle=flow_by_cycle,
        dpq_invalid_counts=dpq_invalid_counts,
    )


def _validate_feature_contract(frame: pd.DataFrame) -> None:
    supported = set(BASE_SUPPORTED_FEATURE_SET)
    for spec in _target_specs(feature_schema_manifest()):
        target = _feature_column(str(spec["group"]), str(spec["name"]))
        mask = f"feature_mask.{target}"
        expected_mask = frame[target].notna().astype("int8")
        actual_mask = pd.to_numeric(frame[mask], errors="raise").astype("int8")
        if not actual_mask.equals(expected_mask):
            raise NhanesSsqAdapterError("feature mask does not match missingness")
        if target not in supported and (
            frame[target].notna().any() or bool(actual_mask.any())
        ):
            raise NhanesSsqAdapterError("unsupported feature is not null/mask 0")
        present = frame[target].dropna()
        if str(spec["value_type"]) == "category":
            if not set(present.astype(str)).issubset(set(spec["categories"])):
                raise NhanesSsqAdapterError(
                    "canonical category is outside MH-003 schema"
                )
            continue
        numeric = pd.to_numeric(present, errors="raise").astype(float)
        minimum = spec.get("minimum")
        maximum = spec.get("maximum")
        if minimum is not None and bool((numeric < float(minimum)).any()):
            raise NhanesSsqAdapterError("canonical feature is below schema minimum")
        if maximum is not None and bool((numeric > float(maximum)).any()):
            raise NhanesSsqAdapterError("canonical feature is above schema maximum")


def _validate_frozen_statistics(assembly: _Assembly) -> None:
    canonical = assembly.canonical
    observed_cycle_counts = {
        str(key): int(value)
        for key, value in canonical["cycle"].value_counts().sort_index().items()
    }
    observed_positive_by_cycle = {
        cycle: int(canonical.loc[canonical["cycle"].eq(cycle), "binary_target"].sum())
        for cycle in CYCLES
    }
    observed_severity = {
        int(key): int(value)
        for key, value in canonical["phq9_severity"].value_counts().sort_index().items()
    }
    observed_source_rows = {
        cycle: {
            "DEMO": assembly.flow_by_cycle[cycle]["demo_source_rows"],
            "DPQ": assembly.flow_by_cycle[cycle]["dpq_source_rows"],
            "SSQ": assembly.flow_by_cycle[cycle]["ssq_source_rows"],
        }
        for cycle in CYCLES
    }
    observed = {
        "source_rows": observed_source_rows,
        "age_60_plus": {
            cycle: assembly.flow_by_cycle[cycle]["age_60_plus"] for cycle in CYCLES
        },
        "complete_dpq": {
            cycle: assembly.flow_by_cycle[cycle]["complete_dpq_all_ages"]
            for cycle in CYCLES
        },
        "cycle_counts": observed_cycle_counts,
        "positive_by_cycle": observed_positive_by_cycle,
        "positive_rows": int(canonical["binary_target"].sum()),
        "severity_counts": observed_severity,
    }
    expected = {
        "source_rows": EXPECTED_SOURCE_ROWS,
        "age_60_plus": EXPECTED_AGE_60_PLUS,
        "complete_dpq": EXPECTED_COMPLETE_DPQ,
        "cycle_counts": EXPECTED_CYCLE_COUNTS,
        "positive_by_cycle": EXPECTED_POSITIVE_BY_CYCLE,
        "positive_rows": EXPECTED_POSITIVE_ROWS,
        "severity_counts": EXPECTED_SEVERITY_COUNTS,
    }
    if observed != expected:
        raise NhanesSsqAdapterError("DATA-006 frozen aggregate statistics do not match")
    if any(
        assembly.flow_by_cycle[cycle]["dropped_for_missing_ssq_join"] != 0
        for cycle in CYCLES
    ):
        raise NhanesSsqAdapterError("DATA-006 SSQ join coverage changed")


def build_canonical_frame(
    demographics_by_cycle: Mapping[str, pd.DataFrame],
    dpq_by_cycle: Mapping[str, pd.DataFrame],
    ssq_by_cycle: Mapping[str, pd.DataFrame],
    *,
    enforce_frozen_statistics: bool = False,
) -> pd.DataFrame:
    """Build one cycle-namespaced row per eligible DPQ+SSQ participant."""

    assembly = _assemble_canonical(demographics_by_cycle, dpq_by_cycle, ssq_by_cycle)
    if enforce_frozen_statistics:
        _validate_frozen_statistics(assembly)
    return assembly.canonical


def _read_source_tables(
    source_root: Path,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    demographics: dict[str, pd.DataFrame] = {}
    dpq: dict[str, pd.DataFrame] = {}
    ssq: dict[str, pd.DataFrame] = {}
    try:
        for cycle in CYCLES:
            demographics[cycle] = pd.read_sas(
                source_root / f"DEMO_{cycle}.xpt",
                format="xport",
                encoding="utf-8",
            )
            dpq[cycle] = pd.read_sas(
                source_root / f"DPQ_{cycle}.xpt",
                format="xport",
                encoding="utf-8",
            )
            ssq[cycle] = pd.read_sas(
                source_root / f"SSQ_{cycle}.xpt",
                format="xport",
                encoding="utf-8",
            )
    except (OSError, TypeError, ValueError) as exc:
        raise NhanesSsqAdapterError("DATA-006 source tables are not readable") from exc
    return demographics, dpq, ssq


def build_field_mapping() -> dict[str, Any]:
    schema = feature_schema_manifest()
    targets: list[dict[str, Any]] = []
    for spec in _target_specs(schema):
        group = str(spec["group"])
        name = str(spec["name"])
        target = _feature_column(group, name)
        rule = _MAPPED_TARGETS.get(target)
        targets.append(
            {
                "base_canonical_policy": (
                    "mapped_when_source_is_valid_else_null_with_mask_0"
                    if rule
                    else "null_with_mask_0"
                ),
                "canonical_mask_column": _mask_column(group, name),
                "canonical_value_column": target,
                "formula": rule["formula"] if rule else None,
                "group": group,
                "mapping_kind": (
                    rule["mapping_kind"] if rule else "no_strict_same_semantics_source"
                ),
                "mapping_status": "mapped" if rule else "unsupported",
                "schema_spec": {
                    key: value for key, value in spec.items() if key != "group"
                },
                "source_fields": list(rule["source_fields"]) if rule else [],
                "target_field": name,
            }
        )

    source_contract: list[dict[str, Any]] = []
    for field in DEMOGRAPHIC_AUDIT_FIELDS:
        source_contract.append(
            {
                "role": (
                    "production_mapping_source"
                    if field in {"RIDAGEYR", "RIAGENDR", "DMDMARTL", "DMDEDUC2"}
                    else "controlled_source_audit_only"
                ),
                "source_field": field,
                "table": "DEMO_D/E",
            }
        )
    source_contract.extend(
        {
            "role": "label_only_prohibited_as_input",
            "source_field": field,
            "table": "DPQ_D/E",
        }
        for field in (*DPQ_FIELDS, DPQ_FUNCTION_FIELD)
    )
    source_contract.extend(
        {
            "role": "controlled_source_audit_only_no_production_mapping",
            "source_field": field,
            "table": "SSQ_D/E",
        }
        for field in SSQ_FIELDS
    )
    return {
        "adapter_version": ADAPTER_VERSION,
        "base_supported_feature_set": list(BASE_SUPPORTED_FEATURE_SET),
        "dataset_id": DATASET_ID,
        "ecdf_contract": {
            "applicable": False,
            "formal_instance_created": False,
            "reason": "cross-sectional SocialContext source without x_source",
        },
        "explicit_research_exclusions": [
            "DPQ100",
            "SSQ fields as production features",
            "survey weights",
            "research identifiers",
        ],
        "feature_schema_sha256": MH003_FEATURE_SCHEMA_SHA256,
        "feature_schema_version": schema["schema_version"],
        "fold_supported_feature_set": [],
        "labels": {
            "binary_target": "phq9_total >= 10",
            "excluded_from_total": DPQ_FUNCTION_FIELD,
            "input_feature_use": "prohibited",
            "missing_item_codes": [7, 9],
            "required_complete_fields": list(DPQ_FIELDS),
            "severity_bands": ["0-4", "5-9", "10-14", "15-19", "20-27"],
            "total": "sum(DPQ010..DPQ090) only when all nine are valid",
            "valid_item_codes": [0, 1, 2, 3],
        },
        "mapping_version": MAPPING_VERSION,
        "natural_dates_available": False,
        "participant_contract": {
            "cycle_values": SURVEY_CYCLES,
            "global_key": f"{DATASET_ID}::{{cycle}}:{{integer_seqn}}",
            "join_policy": "join DEMO, DPQ and SSQ inside each cycle only",
            "key": ["cycle", "SEQN"],
            "minimum_age_years": 60,
        },
        "production_input_policy": {
            "allowed_columns": list(BASE_SUPPORTED_FEATURE_SET),
            "dpq_and_targets_are_inputs": False,
            "income_codes_are_mapped_to_economic_status": False,
            "ssq_fields_are_inputs": False,
            "survey_weights_are_inputs": False,
            "unmapped_target_policy": "null_with_feature_mask_0",
        },
        "source_collection_sha256": SOURCE_COLLECTION_SHA256,
        "source_fields": source_contract,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "ssq_boundary": {
            "SSQ061_maps_to_active_contact_count": False,
            "emotional_or_financial_support_maps_to_call_fields": False,
            "religious_attendance_maps_to_social_participation": False,
            "s10_social_contact_mapping_allowed": False,
            "source_fields": list(SSQ_FIELDS),
            "storage": "controlled source audit columns only",
            "trains_personal_trend": False,
        },
        "target_fields": targets,
        "timescale_semantics": TIMESCALE_SEMANTICS,
        "unsupported_feature_set": [
            row["canonical_value_column"]
            for row in targets
            if row["mapping_status"] == "unsupported"
        ],
        "x_source": {
            "applicable": False,
            "canonical_columns_created": False,
            "reason": "no comparable activity-volume source",
        },
    }


def _missing_summary(series: pd.Series) -> dict[str, Any]:
    missing = int(series.isna().sum())
    total = int(len(series))
    return {
        "missing_count": missing,
        "missing_fraction": 0.0 if total == 0 else missing / total,
        "present_count": total - missing,
    }


def _quality_report(assembly: _Assembly, binding: SourceBinding) -> dict[str, Any]:
    canonical = assembly.canonical
    target_fields: dict[str, Any] = {}
    for target in feature_columns():
        mask = f"feature_mask.{target}"
        target_fields[target] = {
            **_missing_summary(canonical[target]),
            "mask_0_count": int(canonical[mask].eq(0).sum()),
            "mask_1_count": int(canonical[mask].sum()),
        }
    source_fields = {
        field: {
            **_missing_summary(canonical[f"source__{field}"]),
            "distinct_present_value_count": int(
                canonical[f"source__{field}"].dropna().nunique()
            ),
        }
        for field in SOURCE_FIELDS
    }
    return {
        "canonical_target_missingness_and_masks": target_fields,
        "dataset_id": DATASET_ID,
        "dpq": {
            "DPQ100_in_total": False,
            "all_nine_required": True,
            "dpq_or_label_fields_used_as_input_features": False,
            "invalid_or_missing_counts_by_cycle": {
                cycle: dict(assembly.dpq_invalid_counts[cycle]) for cycle in CYCLES
            },
            "sas_xport_near_zero_normalised_to_zero": True,
            "valid_item_codes": [0, 1, 2, 3],
        },
        "input_binding": binding.report_dict(),
        "labels": {
            "binary_negative_count": int(canonical["binary_target"].eq(0).sum()),
            "binary_positive_count": int(canonical["binary_target"].eq(1).sum()),
            "positive_by_cycle": {
                cycle: int(
                    canonical.loc[canonical["cycle"].eq(cycle), "binary_target"].sum()
                )
                for cycle in CYCLES
            },
            "severity_counts": {
                str(int(key)): int(value)
                for key, value in canonical["phq9_severity"]
                .value_counts()
                .sort_index()
                .items()
            },
        },
        "natural_dates_available": False,
        "participant_summary": {
            "cycle_counts": {
                str(key): int(value)
                for key, value in canonical["cycle"].value_counts().sort_index().items()
            },
            "participant_count": len(canonical),
            "participant_values_written_to_report": False,
        },
        "production_input_policy": {
            "income_codes_mapped_to_economic_status": False,
            "ssq_fields_mapped_to_s10_or_profile_features": False,
            "unmapped_fields_are_null_with_mask_0": True,
        },
        "report_version": QUALITY_REPORT_VERSION,
        "row_filter": {
            "cycle_flows": {
                cycle: dict(assembly.flow_by_cycle[cycle]) for cycle in CYCLES
            },
            "rule": (
                "cycle-local joins; RIDAGEYR>=60; nine DPQ items in 0..3; "
                "successful SSQ join"
            ),
        },
        "source_field_missingness": source_fields,
        "ssq_audit": {
            "all_fields_controlled_source_only": True,
            "fields": list(SSQ_FIELDS),
            "s10_social_contact_mapping_allowed": False,
            "SSQ061_maps_to_active_contact_count": False,
        },
        "timescale_semantics": TIMESCALE_SEMANTICS,
        "x_source": {
            "applicable": False,
            "formal_ecdf_instance_created": False,
        },
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
        raise NhanesSsqAdapterError("output root must not overlap the source data tree")
    return resolved


def _stage_content(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".stage", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


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
    manifest_relative = f"{DATASET_ID}/artifact_manifest.json"
    if manifest_relative not in payloads:
        raise NhanesSsqAdapterError("artifact payload set has no completion manifest")
    for relative in payloads:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise NhanesSsqAdapterError("artifact relative path is unsafe")
    targets = {relative: destination / Path(relative) for relative in payloads}
    changes: dict[str, bytes] = {}
    original_exists: dict[str, bool] = {}
    for relative, path in targets.items():
        exists = path.is_file()
        original_exists[relative] = exists
        if exists and path.read_bytes() == payloads[relative]:
            continue
        if path.exists() and not path.is_file():
            raise NhanesSsqAdapterError("artifact target exists but is not a file")
        if exists and not overwrite:
            raise NhanesSsqAdapterError(
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


def build_nhanes_ssq_artifacts(
    *,
    workspace_root: Path | None = None,
    manifests_root: Path | None = None,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> BuildResult:
    """Validate frozen sources and publish deterministic DATA-006 artifacts."""

    binding = validate_frozen_inputs(workspace_root, manifests_root)
    destination = _validated_output_root(
        output_root
        or _algorithm_root(binding.workspace_root) / DEFAULT_OUTPUT_RELATIVE,
        binding,
    )
    demographics, dpq, ssq = _read_source_tables(binding.source_root)
    assembly = _assemble_canonical(demographics, dpq, ssq)
    _validate_frozen_statistics(assembly)
    mapping = build_field_mapping()
    quality = _quality_report(assembly, binding)
    revalidated = validate_frozen_inputs(binding.workspace_root, binding.manifests_root)
    if revalidated != binding:
        raise NhanesSsqAdapterError(
            "frozen DATA-006 source binding changed while being read"
        )

    canonical = assembly.canonical
    canonical_bytes = _parquet_bytes(canonical)
    mapping_bytes = _canonical_json_bytes(mapping)
    quality_bytes = _canonical_json_bytes(quality)
    canonical_relative = f"{DATASET_ID}/canonical_nhanes_ssq_2005_2008.parquet"
    quality_relative = f"{DATASET_ID}/data_quality_report.json"
    metadata_relative = f"{DATASET_ID}/adapter_metadata.json"
    mapping_relative = "mappings/nhanes_ssq_2005_2008_v3_3_3_mapping.json"
    preliminary_hashes = {
        canonical_relative: _sha256_bytes(canonical_bytes),
        mapping_relative: _sha256_bytes(mapping_bytes),
        quality_relative: _sha256_bytes(quality_bytes),
    }
    metadata = {
        "adapter_version": ADAPTER_VERSION,
        "artifact_sha256_before_metadata": preliminary_hashes,
        "canonical_column_order": list(canonical.columns),
        "canonical_column_order_version": CANONICAL_COLUMN_ORDER_VERSION,
        "canonical_dtypes": {
            name: str(dtype) for name, dtype in canonical.dtypes.items()
        },
        "canonical_frame_sha256": canonical_frame_sha256(canonical),
        "canonical_participant_count": int(
            canonical["global_participant_id"].nunique()
        ),
        "canonical_row_count": len(canonical),
        "canonical_sort": ["cycle", "seqn"],
        "dataset_id": DATASET_ID,
        "feature_schema_version": feature_schema_manifest()["schema_version"],
        "formal_ecdf_instance_created": False,
        "input_binding": binding.report_dict(),
        "natural_dates_fabricated": False,
        "participant_values_written_to_report": False,
        "s10_social_contact_mapping_allowed": False,
        "timescale_semantics": TIMESCALE_SEMANTICS,
        "x_source_created": False,
    }
    metadata_bytes = _canonical_json_bytes(metadata)
    artifact_payloads = {
        canonical_relative: canonical_bytes,
        quality_relative: quality_bytes,
        metadata_relative: metadata_bytes,
        mapping_relative: mapping_bytes,
    }
    artifact_rows = {
        relative: {"bytes": len(content), "sha256": _sha256_bytes(content)}
        for relative, content in sorted(artifact_payloads.items())
    }
    artifact_manifest = {
        "artifact_count": len(artifact_rows),
        "artifacts": artifact_rows,
        "complete": True,
        "dataset_id": DATASET_ID,
        "input_binding": binding.report_dict(),
        "manifest_version": ARTIFACT_MANIFEST_VERSION,
    }
    artifact_payloads[f"{DATASET_ID}/artifact_manifest.json"] = _canonical_json_bytes(
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
        participant_count=int(canonical["global_participant_id"].nunique()),
        positive_row_count=int(canonical["binary_target"].sum()),
        row_count=len(canonical),
    )


__all__ = [
    "ADAPTER_VERSION",
    "BASE_SUPPORTED_FEATURE_SET",
    "BuildResult",
    "CANONICAL_COLUMN_ORDER_VERSION",
    "DATASET_ID",
    "DEMOGRAPHIC_SOURCE_COLUMNS",
    "DPQ_FIELDS",
    "MH003_FEATURE_SCHEMA_SHA256",
    "NhanesSsqAdapterError",
    "SOURCE_COLLECTION_SHA256",
    "SOURCE_MANIFEST_NAME",
    "SOURCE_MANIFEST_SHA256",
    "SOURCE_RELATIVE",
    "SSQ_FIELDS",
    "SSQ_SUPPORT_SOURCE_FIELDS",
    "build_canonical_frame",
    "build_field_mapping",
    "build_nhanes_ssq_artifacts",
    "canonical_arrow_schema",
    "canonical_column_order",
    "canonical_frame_sha256",
    "feature_columns",
    "validate_frozen_inputs",
]
