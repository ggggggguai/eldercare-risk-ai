"""DATA-005 Shenzhen community-elderly V3.3.3 canonical adapter.

The Dryad table is a cross-sectional questionnaire.  The publisher's
``code`` column has two non-identical collision groups.  DEC-046 therefore
excludes every row in a conflicting group instead of guessing an identifier,
merging fields, or assigning a synthetic participant id.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    feature_schema_manifest,
)


DATASET_ID = "shenzhen_elderly"
ADAPTER_VERSION = "shenzhen-adapter-v3.3.3-data005"
CANONICAL_COLUMN_ORDER_VERSION = "shenzhen-canonical-columns-v1"
MAPPING_VERSION = "shenzhen-field-mapping-v3.3.3-data005"
QUALITY_REPORT_VERSION = "shenzhen-quality-report-v1"
ARTIFACT_MANIFEST_VERSION = "shenzhen-artifact-manifest-v1"

SOURCE_RELATIVE = Path("数据集/心理/DRYAD深证社区老年心理健康")
SOURCE_CSV_NAME = "Mental_Health_Survey_of_the_Elderly.csv"
DEFAULT_OUTPUT_RELATIVE = Path("data/processed/mental_health/mood_social/v3.3.3")

SHENZHEN_COLLECTION_SHA256 = (
    "b9d3d115369b06348e0318bd640e60b027d01ce0486029ae4afeb1ed3cfcbddc"
)
DATA001_FILE_MANIFEST_SHA256 = (
    "59257b21fee44bd405e0be6f299d49ad945c6e6289002717f8769389dce32bd5"
)
MH003_FEATURE_SCHEMA_SHA256 = (
    "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
)

TIMESCALE_SEMANTICS = "cross_sectional_concurrent_association"
PARTICIPANT_KEY_POLICY = "exclude_all_rows_in_conflicting_normalised_code_groups_v1"
CONFLICT_NORMALISATION = "NFKC_then_trim_then_casefold"

SOURCE_FIELDS = (
    "educationcat",
    "marriagecat",
    "chronicdiseases",
    "Monthlypersonalincome",
    "drinking",
    "Smoking",
    "Healthstatus",
    "PHQ9score",
    "GAD7score",
    "ISIscore",
    "Sleepduration",
    "AD8score",
    "CSIDscore",
    "ULSscore",
    "Depressivesymptoms",
    "Anxietysymptoms",
    "Mildcognitiveimpairment",
    "Earlydementia",
    "Insomnia",
)
SOURCE_COLUMNS = ("code", *SOURCE_FIELDS)
TARGET_COLUMNS = ("phq9_score", "phq9_severity", "binary_target")
IDENTITY_COLUMNS = (
    "dataset_id",
    "global_participant_id",
    "participant_id",
    "timescale_semantics",
    "participant_key_policy",
)

EXPECTED_SOURCE_FILE_COUNT = 3
EXPECTED_SOURCE_ROWS = 5_331
EXPECTED_SOURCE_COLUMNS = 20
EXPECTED_CONFLICT_GROUPS = 2
EXPECTED_CONFLICT_ROWS = 4
EXPECTED_CANONICAL_ROWS = 5_327
EXPECTED_CANONICAL_PARTICIPANTS = 5_327
EXPECTED_POSITIVE_ROWS = 186
EXPECTED_SEVERITY_COUNTS = {0: 4_773, 1: 368, 2: 117, 3: 46, 4: 23}

INTEGER_SOURCE_FIELDS = (
    "educationcat",
    "marriagecat",
    "chronicdiseases",
    "Monthlypersonalincome",
    "drinking",
    "Smoking",
    "Healthstatus",
    "PHQ9score",
    "GAD7score",
    "ISIscore",
    "AD8score",
    "CSIDscore",
    "ULSscore",
    "Depressivesymptoms",
    "Anxietysymptoms",
    "Mildcognitiveimpairment",
    "Earlydementia",
    "Insomnia",
)
FLOAT_SOURCE_FIELDS = ("Sleepduration",)

_SOURCE_RANGES: dict[str, tuple[float, float]] = {
    "educationcat": (1, 5),
    "marriagecat": (1, 2),
    "chronicdiseases": (1, 2),
    "Monthlypersonalincome": (1, 5),
    "drinking": (1, 3),
    "Smoking": (1, 3),
    "Healthstatus": (1, 5),
    "PHQ9score": (0, 27),
    "GAD7score": (0, 21),
    # The official ISI instrument is 0..28; this remains source-only.
    "ISIscore": (0, 28),
    "Sleepduration": (0, 24),
    "AD8score": (0, 8),
    # The downloaded source contains values up to 12; it is not a production input.
    "CSIDscore": (0, 12),
    "ULSscore": (6, 24),
    "Depressivesymptoms": (0, 1),
    "Anxietysymptoms": (0, 1),
    "Mildcognitiveimpairment": (0, 1),
    "Earlydementia": (0, 1),
    "Insomnia": (0, 1),
}

_MAPPED_TARGETS: dict[str, dict[str, Any]] = {
    "social_context.marital_status": {
        "source_fields": ["marriagecat"],
        "formula": "1=>not_partnered; 2=>partnered",
        "mapping_kind": "coded_same_semantics",
    },
    "social_context.self_rated_health": {
        "source_fields": ["Healthstatus"],
        "formula": "Healthstatus 1..5 copied; larger means worse self-rated health",
        "mapping_kind": "coded_same_semantics",
    },
    "social_context.education_level": {
        "source_fields": ["educationcat"],
        "formula": "1=>primary_or_less; 2=>middle; 3..5=>high_or_above",
        "mapping_kind": "coded_same_semantics",
    },
    "social_context.economic_status": {
        "source_fields": ["Monthlypersonalincome"],
        "formula": "1..2=>low; 3=>middle; 4..5=>high",
        "mapping_kind": "ordered_category_grouping",
    },
}


class ShenzhenAdapterError(ValueError):
    """Raised when frozen DATA-005 inputs or outputs are invalid."""


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
    for row in sorted(rows, key=lambda value: str(value["relative_path"]).encode()):
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
    raise ShenzhenAdapterError(
        "workspace root containing the Shenzhen source was not found"
    )


def _algorithm_root(workspace_root: Path) -> Path:
    path = workspace_root / "algorithm" / "eldercare-risk-ai-main"
    if not path.is_dir():
        raise ShenzhenAdapterError(
            "algorithm repository was not found in the workspace"
        )
    return path


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ShenzhenAdapterError(
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
                        raise ShenzhenAdapterError(
                            "DATA-001 manifest contains a non-object row"
                        )
                    rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ShenzhenAdapterError("DATA-001 file manifest is not readable") from exc
    return rows


def validate_frozen_inputs(
    workspace_root: Path | None = None,
    manifests_root: Path | None = None,
) -> SourceBinding:
    """Revalidate DATA-001 files, collection hash, and the live MH-003 schema."""

    root = _find_workspace_root(workspace_root or Path.cwd())
    algorithm_root = _algorithm_root(root)
    audit_root = (
        manifests_root or algorithm_root / DEFAULT_OUTPUT_RELATIVE / "manifests"
    )
    manifest_path = audit_root / "dataset_file_manifest.jsonl"
    schema_snapshot_path = audit_root / "feature_schema_manifest.json"
    if _sha256_file(manifest_path) != DATA001_FILE_MANIFEST_SHA256:
        raise ShenzhenAdapterError("DATA-001 file manifest SHA-256 does not match")
    live_schema = feature_schema_manifest()
    live_schema_sha256 = _sha256_bytes(_canonical_json_bytes(live_schema))
    if live_schema_sha256 != MH003_FEATURE_SCHEMA_SHA256:
        raise ShenzhenAdapterError("live MH-003 feature schema SHA-256 does not match")
    if _sha256_file(schema_snapshot_path) != MH003_FEATURE_SCHEMA_SHA256:
        raise ShenzhenAdapterError(
            "DATA-001 feature schema snapshot SHA-256 does not match"
        )
    if _read_json(schema_snapshot_path) != live_schema:
        raise ShenzhenAdapterError(
            "DATA-001 schema snapshot differs from live MH-003 schema"
        )

    manifest_rows = _read_manifest_rows(manifest_path)
    selected = [row for row in manifest_rows if row.get("dataset_id") == DATASET_ID]
    if len(selected) != EXPECTED_SOURCE_FILE_COUNT:
        raise ShenzhenAdapterError("DATA-001 Shenzhen file count does not match")
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
        raise ShenzhenAdapterError("current Shenzhen file set differs from DATA-001")
    current_rows: list[dict[str, Any]] = []
    for row in selected:
        relative = str(row["relative_path"])
        path = root / Path(relative)
        if path.is_symlink():
            raise ShenzhenAdapterError(
                "Shenzhen source files must not be symbolic links"
            )
        before = path.stat()
        file_sha256 = _sha256_file(path)
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise ShenzhenAdapterError("Shenzhen source changed during validation")
        if int(row.get("bytes", -1)) != after.st_size:
            raise ShenzhenAdapterError("Shenzhen source size differs from DATA-001")
        if str(row.get("sha256", "")) != file_sha256:
            raise ShenzhenAdapterError("Shenzhen source SHA-256 differs from DATA-001")
        current_rows.append(
            {
                "relative_path": relative,
                "bytes": int(after.st_size),
                "sha256": file_sha256,
            }
        )
    collection_sha256 = _collection_sha256(current_rows)
    if collection_sha256 != SHENZHEN_COLLECTION_SHA256:
        raise ShenzhenAdapterError("Shenzhen collection SHA-256 does not match")
    return SourceBinding(
        workspace_root=root,
        source_root=source_root,
        manifests_root=audit_root,
        source_collection_sha256=collection_sha256,
        file_manifest_sha256=DATA001_FILE_MANIFEST_SHA256,
        feature_schema_sha256=live_schema_sha256,
        source_file_count=len(current_rows),
    )


def _normalise_key(value: Any) -> tuple[str | None, str | None]:
    if value is None or pd.isna(value):
        return None, None
    raw = unicodedata.normalize("NFKC", str(value).strip())
    if not raw:
        return None, None
    return raw.casefold(), raw


def _target_specs(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in schema["group_order"]:
        for spec in schema["groups"][group]:
            rows.append({"group": group, **dict(spec)})
    if len(rows) != 54:
        raise ShenzhenAdapterError("MH-003 feature count does not match")
    return rows


def _feature_column(group: str, name: str) -> str:
    return f"{group}.{name}"


def _mask_column(group: str, name: str) -> str:
    return f"feature_mask.{group}.{name}"


def canonical_column_order() -> tuple[str, ...]:
    features = tuple(
        _feature_column(str(spec["group"]), str(spec["name"]))
        for spec in _target_specs(feature_schema_manifest())
    )
    masks = tuple(
        _mask_column(str(spec["group"]), str(spec["name"]))
        for spec in _target_specs(feature_schema_manifest())
    )
    return (
        *IDENTITY_COLUMNS,
        *TARGET_COLUMNS,
        *(f"source__{name}" for name in SOURCE_FIELDS),
        *features,
        *masks,
    )


def canonical_arrow_schema() -> pa.Schema:
    field_types: dict[str, pa.DataType] = {
        name: pa.large_string() for name in IDENTITY_COLUMNS
    }
    field_types.update(
        {
            "phq9_score": pa.int8(),
            "phq9_severity": pa.int8(),
            "binary_target": pa.int8(),
        }
    )
    field_types.update(
        {
            f"source__{name}": pa.float64()
            if name in FLOAT_SOURCE_FIELDS
            else pa.int8()
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
            field_types[target] = value_types[str(spec["value_type"])]
        except KeyError as exc:
            raise ShenzhenAdapterError("unsupported MH-003 feature type") from exc
        field_types[f"feature_mask.{target}"] = pa.int8()
    order = canonical_column_order()
    if set(field_types) != set(order):
        raise ShenzhenAdapterError(
            "canonical Arrow schema columns do not match DATA-005"
        )
    return pa.schema([pa.field(name, field_types[name]) for name in order])


def _canonical_arrow_table(frame: pd.DataFrame) -> pa.Table:
    if tuple(frame.columns) != canonical_column_order():
        raise ShenzhenAdapterError("canonical frame columns do not match DATA-005")
    schema = canonical_arrow_schema()
    try:
        arrays = [
            pa.array(
                frame[field.name].tolist(), type=field.type, from_pandas=True, safe=True
            )
            for field in schema
        ]
        return pa.Table.from_arrays(arrays, schema=schema).combine_chunks()
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise ShenzhenAdapterError(
            "canonical frame values do not match the DATA-005 Arrow schema"
        ) from exc


def _empty_feature_record(length: int) -> dict[str, pd.Series]:
    record: dict[str, pd.Series] = {}
    for spec in _target_specs(feature_schema_manifest()):
        target = _feature_column(str(spec["group"]), str(spec["name"]))
        value_type = str(spec["value_type"])
        dtype = (
            "string"
            if value_type == "category"
            else "Int64"
            if value_type == "integer"
            else "Float64"
        )
        record[target] = pd.Series(pd.array([pd.NA] * length, dtype=dtype))
        record[f"feature_mask.{target}"] = pd.Series(np.zeros(length, dtype=np.int8))
    return record


def _numeric_source(source: pd.DataFrame, field: str) -> pd.Series:
    try:
        values = pd.to_numeric(source[field], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ShenzhenAdapterError(f"Shenzhen {field} is not numeric") from exc
    numeric = values.astype("Float64")
    array = numeric.to_numpy(dtype="float64", na_value=np.nan)
    if np.isnan(array).any() or np.isinf(array).any():
        raise ShenzhenAdapterError(
            f"Shenzhen {field} contains missing or non-finite values"
        )
    minimum, maximum = _SOURCE_RANGES[field]
    if (array < minimum - 1e-12).any() or (array > maximum + 1e-12).any():
        raise ShenzhenAdapterError(
            f"Shenzhen {field} is outside its frozen source range"
        )
    if field in INTEGER_SOURCE_FIELDS and (array != np.floor(array)).any():
        raise ShenzhenAdapterError(f"Shenzhen {field} must contain integer codes")
    if field in INTEGER_SOURCE_FIELDS:
        return pd.Series(
            pd.array(array.astype(np.int8), dtype="Int8"), index=source.index
        )
    return pd.Series(pd.array(array, dtype="Float64"), index=source.index)


def _conflict_profile(source: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    normalised = source["code"].map(lambda value: _normalise_key(value)[0])
    if normalised.isna().any():
        raise ShenzhenAdapterError("Shenzhen code contains missing values")
    counts = normalised.value_counts(sort=False)
    duplicate_keys = set(counts[counts > 1].index)
    excluded_mask = normalised.isin(duplicate_keys)
    differing_groups = 0
    for key in duplicate_keys:
        group = source.loc[normalised == key].drop(columns=["code"])
        if len(group.astype("string").drop_duplicates()) <= 1:
            raise ShenzhenAdapterError("Shenzhen conflict group is an exact duplicate")
        differing_groups += 1
    profile = {
        "normalised_key": CONFLICT_NORMALISATION,
        "source_rows": int(len(source)),
        "missing_code_count": int(normalised.isna().sum()),
        "unique_code_count": int(normalised.nunique()),
        "conflict_group_count": int(len(duplicate_keys)),
        "conflict_row_count": int(excluded_mask.sum()),
        "non_identical_conflict_group_count": int(differing_groups),
        "excluded_row_count": int(excluded_mask.sum()),
        "canonical_row_count": int((~excluded_mask).sum()),
    }
    return excluded_mask, profile


def _severity(score: int) -> int:
    if score <= 4:
        return 0
    if score <= 9:
        return 1
    if score <= 14:
        return 2
    if score <= 19:
        return 3
    return 4


def _set_feature(record: dict[str, pd.Series], target: str, values: pd.Series) -> None:
    record[target] = values
    record[f"feature_mask.{target}"] = values.notna().astype("int8")


def _derive_profile_features(
    source: pd.DataFrame, record: dict[str, pd.Series]
) -> None:
    marriage = (
        source["marriagecat"].map({1: "not_partnered", 2: "partnered"}).astype("string")
    )
    health = source["Healthstatus"].astype("Int64")
    education = (
        source["educationcat"]
        .map(
            {
                1: "primary_or_less",
                2: "middle",
                3: "high_or_above",
                4: "high_or_above",
                5: "high_or_above",
            }
        )
        .astype("string")
    )
    economic = (
        source["Monthlypersonalincome"]
        .map({1: "low", 2: "low", 3: "middle", 4: "high", 5: "high"})
        .astype("string")
    )
    _set_feature(record, "social_context.marital_status", marriage)
    _set_feature(record, "social_context.self_rated_health", health)
    _set_feature(record, "social_context.education_level", education)
    _set_feature(record, "social_context.economic_status", economic)


def build_canonical_frame(
    source: pd.DataFrame,
    *,
    enforce_frozen_statistics: bool = False,
) -> pd.DataFrame:
    """Build the base canonical table and apply DEC-046 conflict filtering."""

    missing = set(SOURCE_COLUMNS) - set(source.columns)
    if missing:
        raise ShenzhenAdapterError("Shenzhen source matrix is missing required fields")
    source = source.loc[:, list(SOURCE_COLUMNS)].copy()
    source["code"] = (
        source["code"].map(lambda value: _normalise_key(value)[1]).astype("string")
    )
    if source["code"].isna().any():
        raise ShenzhenAdapterError("Shenzhen code contains missing values")
    excluded_mask, conflict_profile = _conflict_profile(source)
    numeric = {field: _numeric_source(source, field) for field in SOURCE_FIELDS}
    source_typed = pd.DataFrame(numeric, index=source.index)
    source_typed["code"] = source["code"]
    kept_positions = np.flatnonzero((~excluded_mask).to_numpy())
    kept_source = source_typed.iloc[kept_positions].reset_index(drop=True)
    canonical = pd.DataFrame(
        {
            "dataset_id": pd.array([DATASET_ID] * len(kept_source), dtype="string"),
            "global_participant_id": pd.array(
                [f"{DATASET_ID}::{value}" for value in kept_source["code"]],
                dtype="string",
            ),
            "participant_id": kept_source["code"].array,
            "timescale_semantics": pd.array(
                [TIMESCALE_SEMANTICS] * len(kept_source), dtype="string"
            ),
            "participant_key_policy": pd.array(
                [PARTICIPANT_KEY_POLICY] * len(kept_source), dtype="string"
            ),
        }
    )
    scores = kept_source["PHQ9score"].astype("Int8")
    canonical["phq9_score"] = scores
    canonical["phq9_severity"] = pd.array(
        [_severity(int(value)) for value in scores], dtype="Int8"
    )
    canonical["binary_target"] = (scores >= 10).astype("int8")
    for field in SOURCE_FIELDS:
        canonical[f"source__{field}"] = kept_source[field].array

    feature_record = _empty_feature_record(len(canonical))
    _derive_profile_features(kept_source, feature_record)
    for name, values in feature_record.items():
        canonical[name] = values
    if canonical["participant_id"].duplicated().any():
        raise ShenzhenAdapterError("DEC-046 canonical participant key is not unique")
    canonical = canonical.sort_values("participant_id", kind="stable").reset_index(
        drop=True
    )
    canonical = canonical.loc[:, list(canonical_column_order())]
    if enforce_frozen_statistics:
        _validate_frozen_statistics(source, canonical, conflict_profile)
    return canonical


def _validate_frozen_statistics(
    source: pd.DataFrame, canonical: pd.DataFrame, conflict_profile: Mapping[str, Any]
) -> None:
    observed = {
        "source_rows": len(source),
        "source_columns": len(source.columns),
        "conflict_groups": int(conflict_profile["conflict_group_count"]),
        "conflict_rows": int(conflict_profile["conflict_row_count"]),
        "canonical_rows": len(canonical),
        "participants": int(canonical["participant_id"].nunique()),
        "positive_rows": int(canonical["binary_target"].sum()),
        "severity_counts": {
            int(key): int(value)
            for key, value in canonical["phq9_severity"]
            .value_counts()
            .sort_index()
            .items()
        },
    }
    expected = {
        "source_rows": EXPECTED_SOURCE_ROWS,
        "source_columns": EXPECTED_SOURCE_COLUMNS,
        "conflict_groups": EXPECTED_CONFLICT_GROUPS,
        "conflict_rows": EXPECTED_CONFLICT_ROWS,
        "canonical_rows": EXPECTED_CANONICAL_ROWS,
        "participants": EXPECTED_CANONICAL_PARTICIPANTS,
        "positive_rows": EXPECTED_POSITIVE_ROWS,
        "severity_counts": EXPECTED_SEVERITY_COUNTS,
    }
    if observed != expected:
        raise ShenzhenAdapterError("Shenzhen frozen aggregate statistics do not match")


def _read_source_frame(source_root: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            source_root / SOURCE_CSV_NAME,
            dtype="string",
            encoding="utf-8-sig",
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ShenzhenAdapterError("Shenzhen CSV is not readable") from exc
    if tuple(frame.columns) != SOURCE_COLUMNS:
        raise ShenzhenAdapterError("Shenzhen CSV columns do not match DATA-001")
    return frame


def _missing_summary(series: pd.Series) -> dict[str, Any]:
    missing = int(series.isna().sum())
    total = int(len(series))
    return {
        "present_count": total - missing,
        "missing_count": missing,
        "missing_fraction": 0.0 if total == 0 else missing / total,
    }


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
                "group": group,
                "target_field": name,
                "canonical_value_column": target,
                "canonical_mask_column": _mask_column(group, name),
                "schema_spec": {
                    key: value for key, value in spec.items() if key != "group"
                },
                "mapping_status": "mapped" if rule else "unsupported",
                "mapping_kind": rule["mapping_kind"]
                if rule
                else "no_strict_same_semantics_source",
                "source_fields": list(rule["source_fields"]) if rule else [],
                "formula": rule["formula"] if rule else None,
                "base_canonical_policy": (
                    "mapped_when_required_source_is_valid_else_null_with_mask_0"
                    if rule
                    else "null_with_mask_0"
                ),
            }
        )
    source_roles = {
        "code": "participant_key_only",
        "marriagecat": "production_mapping_source",
        "Healthstatus": "production_mapping_source",
        "educationcat": "production_mapping_source",
        "Monthlypersonalincome": "production_mapping_source",
        "chronicdiseases": "source_only_binary_status_not_chronic_count",
        "drinking": "source_only_not_in_v3_profile_schema",
        "Smoking": "source_only_not_in_v3_profile_schema",
        "Sleepduration": "source_only_self_report_sleep_prohibited_from_production",
        "PHQ9score": "label_only_prohibited_as_input",
        "GAD7score": "neighbouring_scale_label_only_prohibited_as_input",
        "ISIscore": "neighbouring_scale_label_only_prohibited_as_input",
        "AD8score": "neighbouring_scale_label_only_prohibited_as_input",
        "CSIDscore": "neighbouring_scale_label_only_prohibited_as_input",
        "ULSscore": "neighbouring_scale_label_only_prohibited_as_input",
        "Depressivesymptoms": "published_label_only_prohibited_as_input",
        "Anxietysymptoms": "published_label_only_prohibited_as_input",
        "Mildcognitiveimpairment": "published_label_only_prohibited_as_input",
        "Earlydementia": "published_label_only_prohibited_as_input",
        "Insomnia": "published_label_only_prohibited_as_input",
    }
    return {
        "mapping_version": MAPPING_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "dataset_id": DATASET_ID,
        "feature_schema_version": schema["schema_version"],
        "feature_schema_sha256": MH003_FEATURE_SCHEMA_SHA256,
        "source_collection_sha256": SHENZHEN_COLLECTION_SHA256,
        "file_manifest_sha256": DATA001_FILE_MANIFEST_SHA256,
        "timescale_semantics": TIMESCALE_SEMANTICS,
        "natural_dates_available": False,
        "participant_contract": {
            "candidate_key": "code",
            "global_key": "shenzhen_elderly::participant_id",
            "normalisation": CONFLICT_NORMALISATION,
            "conflict_policy": PARTICIPANT_KEY_POLICY,
            "source_rows": EXPECTED_SOURCE_ROWS,
            "conflict_groups": EXPECTED_CONFLICT_GROUPS,
            "conflict_rows_excluded": EXPECTED_CONFLICT_ROWS,
            "canonical_rows": EXPECTED_CANONICAL_ROWS,
            "participant_values_written_to_reports": False,
        },
        "labels": {
            "source_field": "PHQ9score",
            "binary_target": "phq9_score >= 10",
            "severity_bands": ["0-4", "5-9", "10-14", "15-19", "20-27"],
            "input_feature_use": "prohibited",
        },
        "source_fields": [
            {
                "source_field": field,
                "canonical_column": f"source__{field}",
                "role": source_roles[field],
            }
            for field in SOURCE_COLUMNS
        ],
        "target_fields": targets,
        "base_supported_feature_set": sorted(_MAPPED_TARGETS),
        "fold_supported_feature_set": [],
        "unsupported_feature_set": [
            row["canonical_value_column"]
            for row in targets
            if row["mapping_status"] == "unsupported"
        ],
        "production_input_policy": {
            "allowed_columns": sorted(_MAPPED_TARGETS),
            "source_label_and_neighbouring_scale_columns_are_inputs": False,
            "self_report_sleep_is_production_input": False,
            "chronic_binary_is_mapped_to_count": False,
            "unmapped_target_policy": "null_with_feature_mask_0",
            "categorical_encoding_and_imputation": "fit_inside_training_fold_only",
        },
        "exclusion_policy": {
            "rule": PARTICIPANT_KEY_POLICY,
            "normalisation": CONFLICT_NORMALISATION,
            "source_rows_excluded": EXPECTED_CONFLICT_ROWS,
            "row_values_or_keys_persisted": False,
        },
    }


def _quality_report(
    source: pd.DataFrame,
    canonical: pd.DataFrame,
    binding: SourceBinding,
    conflict_profile: Mapping[str, Any],
) -> dict[str, Any]:
    target_fields: dict[str, Any] = {}
    for spec in _target_specs(feature_schema_manifest()):
        target = _feature_column(str(spec["group"]), str(spec["name"]))
        mask = _mask_column(str(spec["group"]), str(spec["name"]))
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
        "source_structure": {
            "row_count": int(len(source)),
            "column_count": int(len(source.columns)),
            "source_file": SOURCE_CSV_NAME,
        },
        "row_filter": {
            "rule": PARTICIPANT_KEY_POLICY,
            "normalisation": CONFLICT_NORMALISATION,
            "source_rows": int(len(source)),
            "conflict_group_count": int(conflict_profile["conflict_group_count"]),
            "conflict_row_count": int(conflict_profile["conflict_row_count"]),
            "canonical_rows": int(len(canonical)),
            "arbitrary_rows_kept": 0,
            "merged_rows": 0,
        },
        "participant_summary": {
            "participant_count": int(canonical["participant_id"].nunique()),
            "unique_global_participant_count": int(
                canonical["global_participant_id"].nunique()
            ),
            "participant_values_written_to_report": False,
        },
        "source_field_missingness": {
            field: _missing_summary(canonical[f"source__{field}"])
            for field in SOURCE_FIELDS
        },
        "labels": {
            "binary_definition": "phq9_score >= 10",
            "positive_count": int(canonical["binary_target"].sum()),
            "negative_count": int(canonical["binary_target"].eq(0).sum()),
            "severity_counts": {
                str(int(key)): int(value)
                for key, value in canonical["phq9_severity"]
                .value_counts()
                .sort_index()
                .items()
            },
            "source_label_columns_used_as_input_features": False,
        },
        "production_mapping": {
            "mapped_feature_count": len(_MAPPED_TARGETS),
            "mapped_feature_set": sorted(_MAPPED_TARGETS),
            "unsupported_feature_count": 54 - len(_MAPPED_TARGETS),
            "unmapped_features_are_null_with_mask_0": True,
            "sleep_and_neighbouring_scale_fields_excluded": True,
        },
        "canonical_target_missingness_and_masks": target_fields,
    }


def canonical_frame_sha256(frame: pd.DataFrame) -> str:
    table = _canonical_arrow_table(frame)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table, max_chunksize=max(len(frame), 1))
    return _sha256_bytes(sink.getvalue().to_pybytes())


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
        raise ShenzhenAdapterError("output root must not overlap the source data tree")
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
    destination: Path, payloads: Mapping[str, bytes], *, overwrite: bool
) -> None:
    manifest_relative = "shenzhen/artifact_manifest.json"
    if manifest_relative not in payloads:
        raise ShenzhenAdapterError("artifact payload set has no completion manifest")
    for relative in payloads:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ShenzhenAdapterError("artifact relative path is unsafe")
    targets = {relative: destination / Path(relative) for relative in payloads}
    changes: dict[str, bytes] = {}
    original_exists: dict[str, bool] = {}
    for relative, path in targets.items():
        exists = path.is_file()
        original_exists[relative] = exists
        if exists and path.read_bytes() == payloads[relative]:
            continue
        if path.exists() and not path.is_file():
            raise ShenzhenAdapterError("artifact target exists but is not a file")
        if exists and not overwrite:
            raise ShenzhenAdapterError(
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


def build_shenzhen_artifacts(
    *,
    workspace_root: Path | None = None,
    manifests_root: Path | None = None,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> BuildResult:
    """Validate frozen sources and publish deterministic DATA-005 artifacts."""

    binding = validate_frozen_inputs(workspace_root, manifests_root)
    destination = _validated_output_root(
        output_root
        or _algorithm_root(binding.workspace_root) / DEFAULT_OUTPUT_RELATIVE,
        binding,
    )
    source = _read_source_frame(binding.source_root)
    excluded_mask, conflict_profile = _conflict_profile(source)
    canonical = build_canonical_frame(source, enforce_frozen_statistics=True)
    mapping = build_field_mapping()
    quality = _quality_report(source, canonical, binding, conflict_profile)
    revalidated = validate_frozen_inputs(binding.workspace_root, binding.manifests_root)
    if revalidated != binding:
        raise ShenzhenAdapterError(
            "frozen Shenzhen source binding changed while being read"
        )
    canonical_bytes = _parquet_bytes(canonical)
    mapping_bytes = _canonical_json_bytes(mapping)
    quality_bytes = _canonical_json_bytes(quality)
    preliminary_hashes = {
        "shenzhen/canonical_shenzhen.parquet": _sha256_bytes(canonical_bytes),
        "shenzhen/data_quality_report.json": _sha256_bytes(quality_bytes),
        "mappings/shenzhen_v3_3_3_mapping.json": _sha256_bytes(mapping_bytes),
    }
    metadata = {
        "adapter_version": ADAPTER_VERSION,
        "canonical_column_order_version": CANONICAL_COLUMN_ORDER_VERSION,
        "dataset_id": DATASET_ID,
        "feature_schema_version": feature_schema_manifest()["schema_version"],
        "input_binding": binding.report_dict(),
        "timescale_semantics": TIMESCALE_SEMANTICS,
        "natural_dates_fabricated": False,
        "participant_key_policy": PARTICIPANT_KEY_POLICY,
        "canonical_sort": ["participant_id"],
        "canonical_column_order": list(canonical.columns),
        "canonical_dtypes": {
            name: str(dtype) for name, dtype in canonical.dtypes.items()
        },
        "canonical_row_count": len(canonical),
        "canonical_participant_count": int(canonical["participant_id"].nunique()),
        "canonical_frame_sha256": canonical_frame_sha256(canonical),
        "excluded_conflict_group_count": int(conflict_profile["conflict_group_count"]),
        "excluded_conflict_row_count": int(conflict_profile["conflict_row_count"]),
        "base_canonical_ecdf_fitted": False,
        "formal_ecdf_instance_created": False,
        "artifact_sha256_before_metadata": preliminary_hashes,
    }
    metadata_bytes = _canonical_json_bytes(metadata)
    artifact_payloads = {
        "shenzhen/canonical_shenzhen.parquet": canonical_bytes,
        "shenzhen/data_quality_report.json": quality_bytes,
        "shenzhen/adapter_metadata.json": metadata_bytes,
        "mappings/shenzhen_v3_3_3_mapping.json": mapping_bytes,
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
    artifact_payloads["shenzhen/artifact_manifest.json"] = _canonical_json_bytes(
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
        participant_count=int(canonical["participant_id"].nunique()),
        positive_row_count=int(canonical["binary_target"].sum()),
    )


__all__ = [
    "ADAPTER_VERSION",
    "BuildResult",
    "CANONICAL_COLUMN_ORDER_VERSION",
    "DATASET_ID",
    "MH003_FEATURE_SCHEMA_SHA256",
    "SHENZHEN_COLLECTION_SHA256",
    "ShenzhenAdapterError",
    "build_canonical_frame",
    "build_field_mapping",
    "build_shenzhen_artifacts",
    "canonical_arrow_schema",
    "canonical_column_order",
    "canonical_frame_sha256",
    "validate_frozen_inputs",
]
