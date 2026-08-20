"""DATA-006 Hefei community-elderly V3.3.3 canonical adapter.

The legacy workbook is read-only.  Publisher feedback confirms that ``a1`` is
the participant code, later duplicate rows are revisions, ``h1`` .. ``h9``
are PHQ-9 items stored as 1 .. 4 for questionnaire scores 0 .. 3, and
``b1`` .. ``b7`` form the Teo social-frailty index.  The one frozen collision
is therefore resolved by retaining the later corrected row, with row hashes
and changed fields recorded without persisting the participant value.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
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


DATASET_ID = "hefei_elderly"
ADAPTER_VERSION = "hefei-adapter-v3.3.3-data006"
CANONICAL_COLUMN_ORDER_VERSION = "hefei-canonical-columns-v1"
MAPPING_VERSION = "hefei-field-mapping-v3.3.3-data006"
QUALITY_REPORT_VERSION = "hefei-quality-report-v1"
ARTIFACT_MANIFEST_VERSION = "hefei-artifact-manifest-v1"

SOURCE_RELATIVE = Path("数据集/心理/合肥社区老年衰弱数据")
SOURCE_XLS_NAME = "社区老年人社会衰弱与抑郁症状研究原始数据.xls"
SOURCE_SHEET_NAME = "1-511"
DEFAULT_OUTPUT_RELATIVE = Path("data/processed/mental_health/mood_social/v3.3.3-r2")

HEFEI_COLLECTION_SHA256 = (
    "c02bce6dbb2e137fc9fbc6b6a8e1c51aeb48c5a048fc10226420ff87d074bd1d"
)
DATA001_FILE_MANIFEST_SHA256 = (
    "6591374bb4273c073126c1ed750c75bd852e08bddb2104797bcf893aec8f11ff"
)
MH003_FEATURE_SCHEMA_SHA256 = (
    "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
)
SOURCE_XLS_SHA256 = (
    "5fc1769ef5e4032cf50386be2c1c48c8ea2833ca903b2d5d79f64cf3124cfb90"
)

TIMESCALE_SEMANTICS = "cross_sectional_concurrent_association"
PARTICIPANT_KEY_POLICY = "publisher_confirmed_later_revision_keep_last_v1"
PARTICIPANT_KEY_NORMALISATION = "integer_code_to_canonical_decimal_v1"
PHQ9_ITEM_POLICY = "publisher_confirmed_source_1_to_4_minus_1_v1"
SFI_POLICY = "teo_2017_items_1_2_6_7_yes_items_3_4_5_no_score_1_v1"

EXPECTED_SOURCE_FILE_COUNT = 2
EXPECTED_SOURCE_ROWS = 511
EXPECTED_SOURCE_COLUMNS = 31
EXPECTED_CANONICAL_ROWS = 510
EXPECTED_CANONICAL_PARTICIPANTS = 510
EXPECTED_DUPLICATE_GROUPS = 1
EXPECTED_DUPLICATE_ROWS = 2
EXPECTED_CORRECTION_EXCEL_ROWS = [258, 358]
EXPECTED_CHANGED_COLUMNS = [
    "a3",
    "a4",
    "a5",
    "b2",
    "b5",
    "h1",
    "h2",
    "h3",
    "h4",
    "i10",
]
EXPECTED_ROW_SHA256 = [
    "f2cb150bc8ee9b18b3389343cfce0b089b5151a12ed7eff5c6b5f7664b60e5f6",
    "924c643c5fb628b88d1f2a6b98ddfacb37606d2f01ec38241655e6d66920b9c7",
]
EXPECTED_KEY_SHA256_12 = "4c970004b067"
EXPECTED_POSITIVE_ROWS = 31
EXPECTED_GE5_ROWS = 140
EXPECTED_GE15_ROWS = 5
EXPECTED_SEVERITY_COUNTS = {0: 370, 1: 109, 2: 26, 3: 3, 4: 2}
EXPECTED_SFI_COUNTS = {1: 19, 2: 64, 3: 243, 4: 156, 5: 25, 6: 2, 7: 1}
EXPECTED_SFI_CATEGORY_COUNTS = {"early": 19, "severe": 491}
EXPECTED_WILLINGNESS_CODE_COUNTS = {1: 315, 2: 195}

PHQ_SOURCE_FIELDS = tuple(f"h{index}" for index in range(1, 10))
SFI_SOURCE_FIELDS = tuple(f"b{index}" for index in range(1, 8))
UNCONFIRMED_I_FIELDS = tuple(f"i{index}" for index in range(1, 11))
SOURCE_COLUMNS = (
    "a1",
    "a3",
    "a4",
    "a5",
    "a18",
    *SFI_SOURCE_FIELDS,
    *PHQ_SOURCE_FIELDS,
    *UNCONFIRMED_I_FIELDS,
)
CANONICAL_SOURCE_FIELDS = ("a18", *SFI_SOURCE_FIELDS, *PHQ_SOURCE_FIELDS, *UNCONFIRMED_I_FIELDS)
IDENTITY_COLUMNS = (
    "dataset_id",
    "global_participant_id",
    "participant_id",
    "timescale_semantics",
    "participant_key_policy",
)
TARGET_COLUMNS = (
    "phq9_score",
    "phq9_severity",
    "binary_target",
    "phq9_target_ge5",
    "phq9_target_ge15",
)
DERIVED_SOURCE_COLUMNS = ("source__sfi_score", "source__sfi_category")


class HefeiAdapterError(ValueError):
    """Raised when frozen DATA-006 inputs or outputs are invalid."""


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
    auxiliary_ge5_row_count: int


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
            f"{row['relative_path']}\0{row['bytes']}\0{row['sha256']}\n".encode(
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
    raise HefeiAdapterError("workspace root containing the Hefei source was not found")


def _algorithm_root(workspace_root: Path) -> Path:
    path = workspace_root / "algorithm" / "eldercare-risk-ai-main"
    if not path.is_dir():
        raise HefeiAdapterError("algorithm repository was not found in the workspace")
    return path


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HefeiAdapterError(f"frozen JSON input is not readable: {path.name}") from exc


def _read_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise HefeiAdapterError("DATA-001 manifest contains a non-object row")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HefeiAdapterError("DATA-001 file manifest is not readable") from exc
    return rows


def validate_frozen_inputs(
    workspace_root: Path | None = None,
    manifests_root: Path | None = None,
) -> SourceBinding:
    """Revalidate DATA-001 files, collection hash, and live MH-003 schema."""

    root = _find_workspace_root(workspace_root or Path.cwd())
    algorithm_root = _algorithm_root(root)
    audit_root = manifests_root or algorithm_root / DEFAULT_OUTPUT_RELATIVE / "manifests"
    manifest_path = audit_root / "dataset_file_manifest.jsonl"
    schema_snapshot_path = audit_root / "feature_schema_manifest.json"
    if _sha256_file(manifest_path) != DATA001_FILE_MANIFEST_SHA256:
        raise HefeiAdapterError("DATA-001 file manifest SHA-256 does not match")
    live_schema = feature_schema_manifest()
    live_schema_sha256 = _sha256_bytes(_canonical_json_bytes(live_schema))
    if live_schema_sha256 != MH003_FEATURE_SCHEMA_SHA256:
        raise HefeiAdapterError("live MH-003 feature schema SHA-256 does not match")
    if _sha256_file(schema_snapshot_path) != MH003_FEATURE_SCHEMA_SHA256:
        raise HefeiAdapterError("DATA-001 feature schema snapshot SHA-256 does not match")
    if _read_json(schema_snapshot_path) != live_schema:
        raise HefeiAdapterError("DATA-001 schema snapshot differs from live MH-003 schema")

    selected = [
        row
        for row in _read_manifest_rows(manifest_path)
        if row.get("dataset_id") == DATASET_ID
    ]
    if len(selected) != EXPECTED_SOURCE_FILE_COUNT:
        raise HefeiAdapterError("DATA-001 Hefei file count does not match")
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
        raise HefeiAdapterError("current Hefei file set differs from DATA-001")
    current_rows: list[dict[str, Any]] = []
    for row in selected:
        relative = str(row["relative_path"])
        path = root / Path(relative)
        if path.is_symlink():
            raise HefeiAdapterError("Hefei source files must not be symbolic links")
        before = path.stat()
        file_sha256 = _sha256_file(path)
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise HefeiAdapterError("Hefei source changed during validation")
        if int(row.get("bytes", -1)) != after.st_size:
            raise HefeiAdapterError("Hefei source size differs from DATA-001")
        if str(row.get("sha256", "")) != file_sha256:
            raise HefeiAdapterError("Hefei source SHA-256 differs from DATA-001")
        current_rows.append(
            {"relative_path": relative, "bytes": int(after.st_size), "sha256": file_sha256}
        )
    if _sha256_file(source_root / SOURCE_XLS_NAME) != SOURCE_XLS_SHA256:
        raise HefeiAdapterError("Hefei XLS SHA-256 does not match DATA-006")
    collection_sha256 = _collection_sha256(current_rows)
    if collection_sha256 != HEFEI_COLLECTION_SHA256:
        raise HefeiAdapterError("Hefei collection SHA-256 does not match")
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
    rows = [
        {"group": group, **dict(spec)}
        for group in schema["group_order"]
        for spec in schema["groups"][group]
    ]
    if len(rows) != 54:
        raise HefeiAdapterError("MH-003 feature count does not match")
    return rows


def _feature_column(group: str, name: str) -> str:
    return f"{group}.{name}"


def _mask_column(group: str, name: str) -> str:
    return f"feature_mask.{group}.{name}"


def canonical_column_order() -> tuple[str, ...]:
    specs = _target_specs(feature_schema_manifest())
    features = tuple(_feature_column(str(row["group"]), str(row["name"])) for row in specs)
    masks = tuple(_mask_column(str(row["group"]), str(row["name"])) for row in specs)
    return (
        *IDENTITY_COLUMNS,
        *TARGET_COLUMNS,
        *(f"source__{name}" for name in CANONICAL_SOURCE_FIELDS),
        *DERIVED_SOURCE_COLUMNS,
        *features,
        *masks,
    )


def canonical_arrow_schema() -> pa.Schema:
    types: dict[str, pa.DataType] = {name: pa.large_string() for name in IDENTITY_COLUMNS}
    types.update({name: pa.int8() for name in TARGET_COLUMNS})
    types.update({f"source__{name}": pa.int8() for name in CANONICAL_SOURCE_FIELDS})
    types["source__sfi_score"] = pa.int8()
    types["source__sfi_category"] = pa.large_string()
    feature_types = {"float": pa.float64(), "integer": pa.int64(), "category": pa.large_string()}
    for spec in _target_specs(feature_schema_manifest()):
        target = _feature_column(str(spec["group"]), str(spec["name"]))
        try:
            types[target] = feature_types[str(spec["value_type"])]
        except KeyError as exc:
            raise HefeiAdapterError("unsupported MH-003 feature type") from exc
        types[f"feature_mask.{target}"] = pa.int8()
    order = canonical_column_order()
    if set(types) != set(order):
        raise HefeiAdapterError("canonical Arrow schema columns do not match DATA-006")
    return pa.schema([pa.field(name, types[name]) for name in order])


def _canonical_arrow_table(frame: pd.DataFrame) -> pa.Table:
    if tuple(frame.columns) != canonical_column_order():
        raise HefeiAdapterError("canonical frame columns do not match DATA-006")
    schema = canonical_arrow_schema()
    try:
        arrays = [
            pa.array(frame[field.name].tolist(), type=field.type, from_pandas=True, safe=True)
            for field in schema
        ]
        return pa.Table.from_arrays(arrays, schema=schema).combine_chunks()
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise HefeiAdapterError("canonical values do not match DATA-006 Arrow schema") from exc


def _empty_feature_record(length: int) -> dict[str, pd.Series]:
    record: dict[str, pd.Series] = {}
    for spec in _target_specs(feature_schema_manifest()):
        target = _feature_column(str(spec["group"]), str(spec["name"]))
        value_type = str(spec["value_type"])
        dtype = "string" if value_type == "category" else "Int64" if value_type == "integer" else "Float64"
        record[target] = pd.Series(pd.array([pd.NA] * length, dtype=dtype))
        record[f"feature_mask.{target}"] = pd.Series(np.zeros(length, dtype=np.int8))
    return record


def _normalise_participant_id(value: Any) -> str:
    if value is None or pd.isna(value) or isinstance(value, (bool, np.bool_)):
        raise HefeiAdapterError("Hefei a1 contains a missing or invalid value")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HefeiAdapterError("Hefei a1 is not an integer participant code") from exc
    if not math.isfinite(numeric) or numeric != math.floor(numeric):
        raise HefeiAdapterError("Hefei a1 is not an integer participant code")
    return str(int(numeric))


def _stable_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _row_sha256(row: pd.Series) -> str:
    payload = json.dumps(
        [_stable_value(value) for value in row.tolist()],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _revision_profile(source: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    normalised = source["a1"].map(_normalise_participant_id)
    duplicate_keys = list(normalised.value_counts(sort=False).loc[lambda s: s > 1].index)
    keep = pd.Series(True, index=source.index, dtype=bool)
    groups: list[dict[str, Any]] = []
    for key in duplicate_keys:
        positions = [
            int(position)
            for position in np.flatnonzero(normalised.eq(key).to_numpy())
        ]
        if len(positions) != 2:
            raise HefeiAdapterError("Hefei revision group must contain exactly two rows")
        rows = source.iloc[positions]
        comparison = rows.drop(columns=["a1"]).astype("string").fillna("<NA>")
        if len(comparison.drop_duplicates()) == 1:
            raise HefeiAdapterError("Hefei duplicate group is not a corrected revision")
        changed = [
            column
            for column in source.columns
            if rows[column].astype("string").fillna("<NA>").nunique(dropna=False) > 1
        ]
        hashes = [_row_sha256(source.iloc[position]) for position in positions]
        keep.iloc[positions[0]] = False
        groups.append(
            {
                "participant_key_sha256_12": _sha256_bytes(key.encode("utf-8"))[:12],
                "excel_rows_in_source_order": [position + 2 for position in positions],
                "row_sha256_in_source_order": hashes,
                "changed_columns": changed,
                "retained_excel_row": positions[-1] + 2,
                "discarded_excel_row": positions[0] + 2,
            }
        )
    profile = {
        "policy": PARTICIPANT_KEY_POLICY,
        "normalisation": PARTICIPANT_KEY_NORMALISATION,
        "source_rows": int(len(source)),
        "missing_key_count": 0,
        "unique_key_count": int(normalised.nunique()),
        "duplicate_group_count": len(groups),
        "duplicate_row_count": int(sum(len(group["excel_rows_in_source_order"]) for group in groups)),
        "discarded_revision_row_count": int((~keep).sum()),
        "canonical_row_count": int(keep.sum()),
        "groups": groups,
        "participant_values_written_to_report": False,
    }
    return keep, profile


def _numeric_integer_source(source: pd.DataFrame, field: str) -> pd.Series:
    try:
        values = pd.to_numeric(source[field], errors="raise")
    except (TypeError, ValueError) as exc:
        raise HefeiAdapterError(f"Hefei {field} is not numeric") from exc
    array = values.to_numpy(dtype="float64", na_value=np.nan)
    if np.isnan(array).any() or np.isinf(array).any() or (array != np.floor(array)).any():
        raise HefeiAdapterError(f"Hefei {field} contains missing or non-integer values")
    if field == "a18" and not np.isin(array, [1, 2]).all():
        raise HefeiAdapterError("Hefei a18 is outside its observed binary code set")
    if field in SFI_SOURCE_FIELDS and not np.isin(array, [0, 1]).all():
        raise HefeiAdapterError(f"Hefei {field} is outside the confirmed 0/1 code set")
    if field in PHQ_SOURCE_FIELDS and not np.isin(array, [1, 2, 3, 4]).all():
        raise HefeiAdapterError(f"Hefei {field} is outside the confirmed 1..4 storage range")
    if field in UNCONFIRMED_I_FIELDS and (array.min() < 0 or array.max() > 10):
        raise HefeiAdapterError(f"Hefei {field} is outside its frozen raw range")
    return pd.Series(pd.array(array.astype(np.int8), dtype="Int8"), index=source.index)


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


def _sfi_category(score: int) -> str:
    if score == 0:
        return "none"
    if score == 1:
        return "early"
    return "severe"


def build_canonical_frame(
    source: pd.DataFrame,
    *,
    enforce_frozen_statistics: bool = False,
) -> pd.DataFrame:
    """Build the Hefei canonical table using publisher-confirmed rules."""

    if tuple(str(column).strip() for column in source.columns) != SOURCE_COLUMNS:
        raise HefeiAdapterError("Hefei source columns do not match DATA-006")
    source = source.copy()
    source.columns = list(SOURCE_COLUMNS)
    keep_mask, revision_profile = _revision_profile(source)
    participant_ids = source["a1"].map(_normalise_participant_id)
    numeric = {
        field: _numeric_integer_source(source, field)
        for field in CANONICAL_SOURCE_FIELDS
    }
    typed = pd.DataFrame(numeric, index=source.index)
    typed["participant_id"] = participant_ids
    kept = typed.loc[keep_mask].reset_index(drop=True)

    phq_items = pd.DataFrame(
        {field: kept[field].astype("Int8") - 1 for field in PHQ_SOURCE_FIELDS}
    )
    phq9 = phq_items.sum(axis=1).astype("Int8")
    sfi_items = pd.DataFrame(
        {
            "b1": kept["b1"],
            "b2": kept["b2"],
            "b3": 1 - kept["b3"],
            "b4": 1 - kept["b4"],
            "b5": 1 - kept["b5"],
            "b6": kept["b6"],
            "b7": kept["b7"],
        }
    )
    sfi = sfi_items.sum(axis=1).astype("Int8")
    canonical = pd.DataFrame(
        {
            "dataset_id": pd.array([DATASET_ID] * len(kept), dtype="string"),
            "global_participant_id": pd.array(
                [f"{DATASET_ID}::{value}" for value in kept["participant_id"]],
                dtype="string",
            ),
            "participant_id": kept["participant_id"].astype("string").array,
            "timescale_semantics": pd.array([TIMESCALE_SEMANTICS] * len(kept), dtype="string"),
            "participant_key_policy": pd.array([PARTICIPANT_KEY_POLICY] * len(kept), dtype="string"),
            "phq9_score": phq9.array,
            "phq9_severity": pd.array([_severity(int(value)) for value in phq9], dtype="Int8"),
            "binary_target": (phq9 >= 10).astype("int8"),
            "phq9_target_ge5": (phq9 >= 5).astype("int8"),
            "phq9_target_ge15": (phq9 >= 15).astype("int8"),
        }
    )
    for field in CANONICAL_SOURCE_FIELDS:
        canonical[f"source__{field}"] = kept[field].array
    canonical["source__sfi_score"] = sfi.array
    canonical["source__sfi_category"] = pd.array(
        [_sfi_category(int(value)) for value in sfi], dtype="string"
    )
    for name, values in _empty_feature_record(len(canonical)).items():
        canonical[name] = values
    if canonical["participant_id"].duplicated().any():
        raise HefeiAdapterError("DATA-006 canonical participant key is not unique")
    canonical["_sort_key"] = canonical["participant_id"].astype("int64")
    canonical = canonical.sort_values("_sort_key", kind="stable").drop(columns=["_sort_key"])
    canonical = canonical.reset_index(drop=True).loc[:, list(canonical_column_order())]
    if enforce_frozen_statistics:
        _validate_frozen_statistics(source, canonical, revision_profile)
    return canonical


def _validate_frozen_statistics(
    source: pd.DataFrame,
    canonical: pd.DataFrame,
    revision_profile: Mapping[str, Any],
) -> None:
    groups = list(revision_profile["groups"])
    if len(groups) != 1:
        raise HefeiAdapterError("Hefei frozen revision group count does not match")
    group = groups[0]
    correction_observed = {
        "participant_key_sha256_12": group["participant_key_sha256_12"],
        "excel_rows_in_source_order": group["excel_rows_in_source_order"],
        "row_sha256_in_source_order": group["row_sha256_in_source_order"],
        "changed_columns": group["changed_columns"],
        "retained_excel_row": group["retained_excel_row"],
    }
    correction_expected = {
        "participant_key_sha256_12": EXPECTED_KEY_SHA256_12,
        "excel_rows_in_source_order": EXPECTED_CORRECTION_EXCEL_ROWS,
        "row_sha256_in_source_order": EXPECTED_ROW_SHA256,
        "changed_columns": EXPECTED_CHANGED_COLUMNS,
        "retained_excel_row": EXPECTED_CORRECTION_EXCEL_ROWS[-1],
    }
    if correction_observed != correction_expected:
        raise HefeiAdapterError("Hefei frozen publisher revision fingerprint does not match")
    severity = {
        int(key): int(value)
        for key, value in canonical["phq9_severity"].value_counts().sort_index().items()
    }
    sfi_counts = {
        int(key): int(value)
        for key, value in canonical["source__sfi_score"].value_counts().sort_index().items()
    }
    sfi_categories = {
        str(key): int(value)
        for key, value in canonical["source__sfi_category"].value_counts().sort_index().items()
    }
    willingness = {
        int(key): int(value)
        for key, value in canonical["source__a18"].value_counts().sort_index().items()
    }
    observed = {
        "source_rows": len(source),
        "source_columns": len(source.columns),
        "duplicate_groups": revision_profile["duplicate_group_count"],
        "duplicate_rows": revision_profile["duplicate_row_count"],
        "canonical_rows": len(canonical),
        "participants": int(canonical["participant_id"].nunique()),
        "positive_rows": int(canonical["binary_target"].sum()),
        "ge5_rows": int(canonical["phq9_target_ge5"].sum()),
        "ge15_rows": int(canonical["phq9_target_ge15"].sum()),
        "severity": severity,
        "sfi": sfi_counts,
        "sfi_categories": sfi_categories,
        "willingness": willingness,
    }
    expected = {
        "source_rows": EXPECTED_SOURCE_ROWS,
        "source_columns": EXPECTED_SOURCE_COLUMNS,
        "duplicate_groups": EXPECTED_DUPLICATE_GROUPS,
        "duplicate_rows": EXPECTED_DUPLICATE_ROWS,
        "canonical_rows": EXPECTED_CANONICAL_ROWS,
        "participants": EXPECTED_CANONICAL_PARTICIPANTS,
        "positive_rows": EXPECTED_POSITIVE_ROWS,
        "ge5_rows": EXPECTED_GE5_ROWS,
        "ge15_rows": EXPECTED_GE15_ROWS,
        "severity": EXPECTED_SEVERITY_COUNTS,
        "sfi": EXPECTED_SFI_COUNTS,
        "sfi_categories": EXPECTED_SFI_CATEGORY_COUNTS,
        "willingness": EXPECTED_WILLINGNESS_CODE_COUNTS,
    }
    if observed != expected:
        raise HefeiAdapterError("Hefei frozen aggregate statistics do not match")


def _read_source_frame(source_root: Path) -> pd.DataFrame:
    path = source_root / SOURCE_XLS_NAME
    try:
        workbook = pd.ExcelFile(path, engine="xlrd")
        if workbook.sheet_names != [SOURCE_SHEET_NAME]:
            raise HefeiAdapterError("Hefei workbook sheet set does not match DATA-006")
        frame = pd.read_excel(workbook, sheet_name=SOURCE_SHEET_NAME)
    except HefeiAdapterError:
        raise
    except Exception as exc:
        raise HefeiAdapterError(f"Hefei XLS is not readable ({type(exc).__name__})") from exc
    frame.columns = [str(column).strip() for column in frame.columns]
    if tuple(frame.columns) != SOURCE_COLUMNS:
        raise HefeiAdapterError("Hefei XLS columns do not match DATA-006")
    return frame


def build_field_mapping() -> dict[str, Any]:
    schema = feature_schema_manifest()
    targets = [
        {
            "group": str(spec["group"]),
            "target_field": str(spec["name"]),
            "canonical_value_column": _feature_column(str(spec["group"]), str(spec["name"])),
            "canonical_mask_column": _mask_column(str(spec["group"]), str(spec["name"])),
            "schema_spec": {key: value for key, value in spec.items() if key != "group"},
            "mapping_status": "unsupported",
            "mapping_kind": "no_publisher_confirmed_strict_same_semantics_source",
            "source_fields": [],
            "formula": None,
            "base_canonical_policy": "null_with_mask_0",
        }
        for spec in _target_specs(schema)
    ]
    source_roles = {
        "a1": "participant_key_only",
        "a3": "excluded_birth_date_sensitive_missing_and_two_pre_1900_anomalies_no_age_derived",
        "a4": "excluded_unconfirmed_semantics",
        "a5": "excluded_unconfirmed_semantics",
        "a18": "source_only_yes_no_willingness_direction_of_1_2_unconfirmed",
        **{field: "offline_sfi_item_not_production_input" for field in SFI_SOURCE_FIELDS},
        **{field: "label_item_only_prohibited_as_input" for field in PHQ_SOURCE_FIELDS},
        **{field: "source_only_unconfirmed_semantics" for field in UNCONFIRMED_I_FIELDS},
    }
    return {
        "mapping_version": MAPPING_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "dataset_id": DATASET_ID,
        "feature_schema_version": schema["schema_version"],
        "feature_schema_sha256": MH003_FEATURE_SCHEMA_SHA256,
        "source_collection_sha256": HEFEI_COLLECTION_SHA256,
        "file_manifest_sha256": DATA001_FILE_MANIFEST_SHA256,
        "publisher_feedback_status": "confirmed_for_a1_phq9_sfi_and_single_yes_no_willingness",
        "timescale_semantics": TIMESCALE_SEMANTICS,
        "natural_dates_available": False,
        "participant_contract": {
            "candidate_key": "a1",
            "global_key": "hefei_elderly::participant_id",
            "normalisation": PARTICIPANT_KEY_NORMALISATION,
            "revision_policy": PARTICIPANT_KEY_POLICY,
            "source_rows": EXPECTED_SOURCE_ROWS,
            "duplicate_groups": EXPECTED_DUPLICATE_GROUPS,
            "discarded_superseded_rows": 1,
            "canonical_rows": EXPECTED_CANONICAL_ROWS,
            "participant_values_written_to_reports": False,
        },
        "labels": {
            "source_fields": list(PHQ_SOURCE_FIELDS),
            "source_storage": "1..4",
            "questionnaire_item_score": "source_code - 1 gives 0..3",
            "phq9_score": "sum(h1..h9 after source_code - 1)",
            "primary_binary_target": "phq9_score >= 10",
            "publisher_study_auxiliary_target": "phq9_score >= 5",
            "additional_auxiliary_target": "phq9_score >= 15",
            "severity_bands": ["0-4", "5-9", "10-14", "15-19", "20-27"],
            "input_feature_use": "prohibited",
        },
        "social_frailty": {
            "policy": SFI_POLICY,
            "formula": "b1+b2+(1-b3)+(1-b4)+(1-b5)+b6+b7",
            "categories": {"0": "none", "1": "early", "2-7": "severe"},
            "production_input_use": "prohibited_without_runtime_equivalent",
            "offline_research_use": "allowed_within_training_fold_and_never_with_phq_items",
        },
        "digital_health_willingness": {
            "source_field": "a18",
            "publisher_semantics": "single yes/no item",
            "observed_codes": [1, 2],
            "code_direction": "unconfirmed",
            "canonical_policy": "retain_raw_code_source_only_not_a_production_feature",
        },
        "source_fields": [
            {
                "source_field": field,
                "canonical_column": f"source__{field}" if field in CANONICAL_SOURCE_FIELDS else None,
                "role": source_roles[field],
            }
            for field in SOURCE_COLUMNS
        ],
        "derived_source_fields": [
            {"canonical_column": "source__sfi_score", "role": "offline_source_specific_feature"},
            {"canonical_column": "source__sfi_category", "role": "offline_source_specific_category"},
        ],
        "target_fields": targets,
        "base_supported_feature_set": [],
        "fold_supported_feature_set": [],
        "unsupported_feature_set": [row["canonical_value_column"] for row in targets],
        "production_input_policy": {
            "allowed_columns": [],
            "phq9_items_scores_and_targets_are_inputs": False,
            "sfi_items_score_and_category_are_production_inputs": False,
            "a18_is_production_input_before_code_direction_confirmation": False,
            "unconfirmed_i_fields_are_inputs": False,
            "unmapped_target_policy": "null_with_feature_mask_0",
        },
    }


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
    canonical: pd.DataFrame,
    binding: SourceBinding,
    revision_profile: Mapping[str, Any],
) -> dict[str, Any]:
    birth_dates = pd.to_datetime(source["a3"], errors="coerce")
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
        "source_structure": {
            "row_count": len(source),
            "column_count": len(source.columns),
            "source_file": SOURCE_XLS_NAME,
            "sheet_name": SOURCE_SHEET_NAME,
        },
        "publisher_revision_resolution": dict(revision_profile),
        "participant_summary": {
            "participant_count": int(canonical["participant_id"].nunique()),
            "unique_global_participant_count": int(canonical["global_participant_id"].nunique()),
            "participant_values_written_to_report": False,
        },
        "labels": {
            "phq9_item_storage": "1..4",
            "phq9_item_score_formula": "source_code - 1",
            "primary_binary_definition": "phq9_score >= 10",
            "primary_positive_count": int(canonical["binary_target"].sum()),
            "publisher_auxiliary_definition": "phq9_score >= 5",
            "publisher_auxiliary_positive_count": int(canonical["phq9_target_ge5"].sum()),
            "ge15_positive_count": int(canonical["phq9_target_ge15"].sum()),
            "severity_counts": {
                str(int(key)): int(value)
                for key, value in canonical["phq9_severity"].value_counts().sort_index().items()
            },
            "phq9_columns_used_as_input_features": False,
        },
        "social_frailty": {
            "formula": "b1+b2+(1-b3)+(1-b4)+(1-b5)+b6+b7",
            "score_counts": {
                str(int(key)): int(value)
                for key, value in canonical["source__sfi_score"].value_counts().sort_index().items()
            },
            "category_counts": {
                str(key): int(value)
                for key, value in canonical["source__sfi_category"].value_counts().sort_index().items()
            },
            "used_as_production_feature": False,
        },
        "digital_health_willingness": {
            "source_field": "a18",
            "observed_code_counts": {
                str(int(key)): int(value)
                for key, value in canonical["source__a18"].value_counts().sort_index().items()
            },
            "code_direction_confirmed": False,
            "used_as_production_feature": False,
        },
        "excluded_source_quality": {
            "birth_date_missing_count": int(source["a3"].isna().sum()),
            "birth_year_before_1900_count": int((birth_dates.dt.year < 1900).sum()),
            "age_derived": False,
            "a4_a5_semantics_confirmed": False,
            "i1_i10_semantics_confirmed": False,
        },
        "production_mapping": {
            "mapped_feature_count": 0,
            "mapped_feature_set": [],
            "unsupported_feature_count": 54,
            "unmapped_features_are_null_with_mask_0": True,
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
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _validated_output_root(destination: Path, binding: SourceBinding) -> Path:
    resolved = destination.resolve(strict=False)
    if _paths_overlap(resolved, (binding.workspace_root / "数据集").resolve()):
        raise HefeiAdapterError("output root must not overlap the source data tree")
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


def _publish_artifacts(destination: Path, payloads: Mapping[str, bytes], *, overwrite: bool) -> None:
    manifest_relative = "hefei/artifact_manifest.json"
    if manifest_relative not in payloads:
        raise HefeiAdapterError("artifact payload set has no completion manifest")
    for relative in payloads:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise HefeiAdapterError("artifact relative path is unsafe")
    targets = {relative: destination / Path(relative) for relative in payloads}
    changes: dict[str, bytes] = {}
    original_exists: dict[str, bool] = {}
    for relative, path in targets.items():
        exists = path.is_file()
        original_exists[relative] = exists
        if exists and path.read_bytes() == payloads[relative]:
            continue
        if path.exists() and not path.is_file():
            raise HefeiAdapterError("artifact target exists but is not a file")
        if exists and not overwrite:
            raise HefeiAdapterError(f"refusing to overwrite differing artifact: {path.name}")
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


def build_hefei_artifacts(
    *,
    workspace_root: Path | None = None,
    manifests_root: Path | None = None,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> BuildResult:
    """Validate frozen sources and publish deterministic DATA-006 artifacts."""

    binding = validate_frozen_inputs(workspace_root, manifests_root)
    destination = _validated_output_root(
        output_root or _algorithm_root(binding.workspace_root) / DEFAULT_OUTPUT_RELATIVE,
        binding,
    )
    source = _read_source_frame(binding.source_root)
    _, revision_profile = _revision_profile(source)
    canonical = build_canonical_frame(source, enforce_frozen_statistics=True)
    mapping = build_field_mapping()
    quality = _quality_report(source, canonical, binding, revision_profile)
    if validate_frozen_inputs(binding.workspace_root, binding.manifests_root) != binding:
        raise HefeiAdapterError("frozen Hefei source binding changed while being read")
    canonical_bytes = _parquet_bytes(canonical)
    mapping_bytes = _canonical_json_bytes(mapping)
    quality_bytes = _canonical_json_bytes(quality)
    preliminary_hashes = {
        "hefei/canonical_hefei.parquet": _sha256_bytes(canonical_bytes),
        "hefei/data_quality_report.json": _sha256_bytes(quality_bytes),
        "mappings/hefei_v3_3_3_mapping.json": _sha256_bytes(mapping_bytes),
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
        "publisher_revision_retained_excel_row": EXPECTED_CORRECTION_EXCEL_ROWS[-1],
        "participant_values_written_to_metadata": False,
        "canonical_sort": ["participant_id_as_integer"],
        "canonical_column_order": list(canonical.columns),
        "canonical_dtypes": {name: str(dtype) for name, dtype in canonical.dtypes.items()},
        "canonical_row_count": len(canonical),
        "canonical_participant_count": int(canonical["participant_id"].nunique()),
        "canonical_frame_sha256": canonical_frame_sha256(canonical),
        "primary_positive_row_count": int(canonical["binary_target"].sum()),
        "publisher_auxiliary_ge5_row_count": int(canonical["phq9_target_ge5"].sum()),
        "base_canonical_ecdf_fitted": False,
        "formal_ecdf_instance_created": False,
        "artifact_sha256_before_metadata": preliminary_hashes,
    }
    metadata_bytes = _canonical_json_bytes(metadata)
    artifact_payloads = {
        "hefei/canonical_hefei.parquet": canonical_bytes,
        "hefei/data_quality_report.json": quality_bytes,
        "hefei/adapter_metadata.json": metadata_bytes,
        "mappings/hefei_v3_3_3_mapping.json": mapping_bytes,
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
    artifact_payloads["hefei/artifact_manifest.json"] = _canonical_json_bytes(artifact_manifest)
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
        auxiliary_ge5_row_count=int(canonical["phq9_target_ge5"].sum()),
    )


__all__ = [
    "ADAPTER_VERSION",
    "BuildResult",
    "CANONICAL_COLUMN_ORDER_VERSION",
    "CANONICAL_SOURCE_FIELDS",
    "DATASET_ID",
    "HEFEI_COLLECTION_SHA256",
    "HefeiAdapterError",
    "PARTICIPANT_KEY_POLICY",
    "SOURCE_COLUMNS",
    "build_canonical_frame",
    "build_field_mapping",
    "build_hefei_artifacts",
    "canonical_arrow_schema",
    "canonical_column_order",
    "canonical_frame_sha256",
    "validate_frozen_inputs",
]
