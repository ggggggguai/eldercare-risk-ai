"""Independently validate DATA-005 Shenzhen V3.3.3 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    feature_schema_manifest,
)


DATASET_ID = "shenzhen_elderly"
SOURCE_RELATIVE = Path("数据集/心理/DRYAD深证社区老年心理健康")
SOURCE_CSV_NAME = "Mental_Health_Survey_of_the_Elderly.csv"
DEFAULT_OUTPUT_RELATIVE = Path("data/processed/mental_health/mood_social/v3.3.3")
COLLECTION_SHA256 = "b9d3d115369b06348e0318bd640e60b027d01ce0486029ae4afeb1ed3cfcbddc"
FILE_MANIFEST_SHA256 = (
    "59257b21fee44bd405e0be6f299d49ad945c6e6289002717f8769389dce32bd5"
)
FEATURE_SCHEMA_SHA256 = (
    "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
)
SOURCE_CSV_SHA256 = "14a9d12f1bffb61442d1fdbfc2961b8e24f9f8f405d33ef57d3dc6a09fb2eb3d"
KEY_POLICY = "exclude_all_rows_in_conflicting_normalised_code_groups_v1"
TIMESCALE = "cross_sectional_concurrent_association"

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
FLOAT_SOURCE_FIELDS = {"Sleepduration"}
IDENTITY_COLUMNS = (
    "dataset_id",
    "global_participant_id",
    "participant_id",
    "timescale_semantics",
    "participant_key_policy",
)
TARGET_COLUMNS = ("phq9_score", "phq9_severity", "binary_target")
MAPPED_TARGETS = {
    "social_context.marital_status",
    "social_context.self_rated_health",
    "social_context.education_level",
    "social_context.economic_status",
}
EXPECTED_SEVERITY = {0: 4_773, 1: 368, 2: 117, 3: 46, 4: 23}


class ValidationError(RuntimeError):
    """Raised without including participant values in the message."""


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _collection_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(
        rows, key=lambda value: str(value["relative_path"]).encode("utf-8")
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


def _check(condition: bool, name: str, checks: list[dict[str, Any]]) -> None:
    if not condition:
        raise ValidationError(f"validation failed: {name}")
    checks.append({"check": name, "status": "pass"})


def _target_specs(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"group": group, **dict(spec)}
        for group in schema["group_order"]
        for spec in schema["groups"][group]
    ]


def _feature_columns(schema: Mapping[str, Any]) -> list[str]:
    return [f"{spec['group']}.{spec['name']}" for spec in _target_specs(schema)]


def _expected_columns(schema: Mapping[str, Any]) -> list[str]:
    features = _feature_columns(schema)
    return [
        *IDENTITY_COLUMNS,
        *TARGET_COLUMNS,
        *(f"source__{name}" for name in SOURCE_FIELDS),
        *features,
        *(f"feature_mask.{name}" for name in features),
    ]


def _expected_arrow_schema(schema: Mapping[str, Any]) -> pa.Schema:
    types: dict[str, pa.DataType] = {
        name: pa.large_string() for name in IDENTITY_COLUMNS
    }
    types.update({name: pa.int8() for name in TARGET_COLUMNS})
    types.update(
        {
            f"source__{name}": pa.float64()
            if name in FLOAT_SOURCE_FIELDS
            else pa.int8()
            for name in SOURCE_FIELDS
        }
    )
    feature_types = {
        "float": pa.float64(),
        "integer": pa.int64(),
        "category": pa.large_string(),
    }
    for spec in _target_specs(schema):
        target = f"{spec['group']}.{spec['name']}"
        types[target] = feature_types[str(spec["value_type"])]
        types[f"feature_mask.{target}"] = pa.int8()
    return pa.schema(
        [pa.field(name, types[name]) for name in _expected_columns(schema)]
    )


def _normalised_arrow_table(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    try:
        arrays = [
            pa.array(
                frame[field.name].tolist(), type=field.type, from_pandas=True, safe=True
            )
            for field in schema
        ]
        return pa.Table.from_arrays(arrays, schema=schema).combine_chunks()
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise ValidationError(
            "canonical values do not match expected Arrow schema"
        ) from exc


def _frame_sha256(frame: pd.DataFrame, schema: pa.Schema) -> str:
    table = _normalised_arrow_table(frame, schema)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table, max_chunksize=max(len(frame), 1))
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _normalise_code(value: Any) -> tuple[str, str]:
    if value is None or pd.isna(value):
        raise ValidationError("source code is missing")
    raw = unicodedata.normalize("NFKC", str(value).strip())
    if not raw:
        raise ValidationError("source code is missing")
    return raw.casefold(), raw


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


def _read_manifest_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError("DATA-001 file manifest is not readable") from exc


def _verify_inputs(
    workspace_root: Path,
    checks: list[dict[str, Any]],
) -> pd.DataFrame:
    manifests = (
        workspace_root
        / "algorithm"
        / "eldercare-risk-ai-main"
        / DEFAULT_OUTPUT_RELATIVE
        / "manifests"
    )
    manifest_path = manifests / "dataset_file_manifest.jsonl"
    schema_path = manifests / "feature_schema_manifest.json"
    _check(
        _sha256_file(manifest_path) == FILE_MANIFEST_SHA256,
        "file_manifest_sha256",
        checks,
    )
    schema = feature_schema_manifest()
    _check(
        hashlib.sha256(_canonical_json_bytes(schema)).hexdigest()
        == FEATURE_SCHEMA_SHA256,
        "live_schema_sha256",
        checks,
    )
    _check(
        _sha256_file(schema_path) == FEATURE_SCHEMA_SHA256,
        "schema_snapshot_sha256",
        checks,
    )
    _check(_load_json(schema_path) == schema, "schema_snapshot_content", checks)
    selected = [
        row
        for row in _read_manifest_rows(manifest_path)
        if row.get("dataset_id") == DATASET_ID
    ]
    _check(len(selected) == 3, "source_file_count", checks)
    current_rows: list[dict[str, Any]] = []
    for row in selected:
        relative = str(row["relative_path"])
        path = workspace_root / Path(relative)
        digest = _sha256_file(path)
        _check(
            path.stat().st_size == int(row["bytes"]), f"source_size_{path.name}", checks
        )
        _check(digest == str(row["sha256"]), f"source_sha256_{path.name}", checks)
        current_rows.append(
            {"relative_path": relative, "bytes": path.stat().st_size, "sha256": digest}
        )
    _check(
        _collection_sha256(current_rows) == COLLECTION_SHA256,
        "source_collection_sha256",
        checks,
    )
    csv_path = workspace_root / SOURCE_RELATIVE / SOURCE_CSV_NAME
    _check(_sha256_file(csv_path) == SOURCE_CSV_SHA256, "official_csv_sha256", checks)
    try:
        source = pd.read_csv(csv_path, dtype="string", encoding="utf-8-sig")
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValidationError("source CSV is not readable") from exc
    _check(tuple(source.columns) == SOURCE_COLUMNS, "source_columns", checks)
    _check(len(source) == 5_331, "source_row_count", checks)
    return source


def _kept_source(source: pd.DataFrame, checks: list[dict[str, Any]]) -> pd.DataFrame:
    normalised = source["code"].map(lambda value: _normalise_code(value)[0])
    raw = source["code"].map(lambda value: _normalise_code(value)[1])
    counts = normalised.value_counts(sort=False)
    conflicts = set(counts[counts > 1].index)
    excluded = normalised.isin(conflicts)
    _check(len(conflicts) == 2, "conflict_group_count", checks)
    _check(int(excluded.sum()) == 4, "conflict_row_count", checks)
    for key in conflicts:
        group = source.loc[normalised == key].drop(columns=["code"])
        _check(
            len(group.astype("string").drop_duplicates()) > 1,
            "conflict_group_non_identical",
            checks,
        )
    kept = source.loc[~excluded].copy()
    kept["code"] = raw.loc[~excluded]
    kept = kept.sort_values("code", kind="stable").reset_index(drop=True)
    _check(len(kept) == 5_327, "canonical_source_row_count", checks)
    _check(kept["code"].nunique() == 5_327, "canonical_source_key_unique", checks)
    return kept


def _series_equal(
    actual: pd.Series, expected: pd.Series, *, numeric: bool = False
) -> bool:
    if numeric:
        left = pd.to_numeric(actual, errors="coerce").to_numpy(
            dtype="float64", na_value=np.nan
        )
        right = pd.to_numeric(expected, errors="coerce").to_numpy(
            dtype="float64", na_value=np.nan
        )
        return bool(np.array_equal(left, right, equal_nan=True))
    return (
        actual.astype("string")
        .reset_index(drop=True)
        .equals(expected.astype("string").reset_index(drop=True))
    )


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        if any(
            str(key) in {"participant_ids", "participant_values", "conflict_codes"}
            for key in value
        ):
            return True
        return any(_contains_forbidden_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def validate(workspace_root: Path, output_root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    source = _verify_inputs(workspace_root, checks)
    kept = _kept_source(source, checks)
    canonical_path = output_root / "shenzhen" / "canonical_shenzhen.parquet"
    quality_path = output_root / "shenzhen" / "data_quality_report.json"
    metadata_path = output_root / "shenzhen" / "adapter_metadata.json"
    mapping_path = output_root / "mappings" / "shenzhen_v3_3_3_mapping.json"
    artifact_manifest_path = output_root / "shenzhen" / "artifact_manifest.json"
    expected_shenzhen_files = {
        "canonical_shenzhen.parquet",
        "data_quality_report.json",
        "adapter_metadata.json",
        "artifact_manifest.json",
    }
    _check(
        {path.name for path in (output_root / "shenzhen").iterdir() if path.is_file()}
        == expected_shenzhen_files,
        "shenzhen_artifact_file_set",
        checks,
    )
    table = pq.read_table(canonical_path)
    canonical = table.to_pandas()
    schema = feature_schema_manifest()
    expected_schema = _expected_arrow_schema(schema)
    _check(table.schema == expected_schema, "canonical_arrow_schema", checks)
    _check(
        list(canonical.columns) == _expected_columns(schema),
        "canonical_column_order",
        checks,
    )
    _check(len(canonical) == 5_327, "canonical_row_count", checks)
    _check(
        canonical["participant_id"].nunique() == 5_327,
        "canonical_participant_count",
        checks,
    )
    _check(
        not canonical["participant_id"].duplicated().any(),
        "canonical_participant_unique",
        checks,
    )
    _check(
        _series_equal(canonical["participant_id"], kept["code"]),
        "participant_id_reconstruction",
        checks,
    )
    _check(
        _series_equal(
            canonical["global_participant_id"],
            kept["code"].map(lambda value: f"{DATASET_ID}::{value}"),
        ),
        "global_participant_id_reconstruction",
        checks,
    )
    _check(canonical["dataset_id"].eq(DATASET_ID).all(), "dataset_namespace", checks)
    _check(
        canonical["timescale_semantics"].eq(TIMESCALE).all(),
        "timescale_semantics",
        checks,
    )
    _check(
        canonical["participant_key_policy"].eq(KEY_POLICY).all(),
        "participant_key_policy",
        checks,
    )

    scores = pd.to_numeric(kept["PHQ9score"], errors="raise").astype(int)
    severity = scores.map(_severity)
    binary = (scores >= 10).astype(int)
    _check(
        _series_equal(canonical["phq9_score"], scores, numeric=True),
        "phq9_score_reconstruction",
        checks,
    )
    _check(
        _series_equal(canonical["phq9_severity"], severity, numeric=True),
        "severity_reconstruction",
        checks,
    )
    _check(
        _series_equal(canonical["binary_target"], binary, numeric=True),
        "binary_target_reconstruction",
        checks,
    )
    _check(int(canonical["binary_target"].sum()) == 186, "positive_count", checks)
    observed_severity = {
        int(key): int(value)
        for key, value in canonical["phq9_severity"].value_counts().sort_index().items()
    }
    _check(observed_severity == EXPECTED_SEVERITY, "severity_distribution", checks)
    for field in SOURCE_FIELDS:
        _check(
            _series_equal(canonical[f"source__{field}"], kept[field], numeric=True),
            f"source_field_{field}",
            checks,
        )

    expected_mapped: dict[str, pd.Series] = {
        "social_context.marital_status": pd.to_numeric(kept["marriagecat"]).map(
            {1: "not_partnered", 2: "partnered"}
        ),
        "social_context.self_rated_health": pd.to_numeric(kept["Healthstatus"]),
        "social_context.education_level": pd.to_numeric(kept["educationcat"]).map(
            {
                1: "primary_or_less",
                2: "middle",
                3: "high_or_above",
                4: "high_or_above",
                5: "high_or_above",
            }
        ),
        "social_context.economic_status": pd.to_numeric(
            kept["Monthlypersonalincome"]
        ).map({1: "low", 2: "low", 3: "middle", 4: "high", 5: "high"}),
    }
    for target in _feature_columns(schema):
        mask = f"feature_mask.{target}"
        if target in expected_mapped:
            _check(
                _series_equal(
                    canonical[target],
                    expected_mapped[target],
                    numeric=target.endswith("self_rated_health"),
                ),
                f"mapped_{target}",
                checks,
            )
            _check(canonical[mask].eq(1).all(), f"mapped_mask_{target}", checks)
        else:
            _check(canonical[target].isna().all(), f"unsupported_null_{target}", checks)
            _check(canonical[mask].eq(0).all(), f"unsupported_mask_{target}", checks)

    quality = _load_json(quality_path)
    metadata = _load_json(metadata_path)
    mapping = _load_json(mapping_path)
    artifact_manifest = _load_json(artifact_manifest_path)
    for payload, name in (
        (quality, "quality"),
        (metadata, "metadata"),
        (mapping, "mapping"),
        (artifact_manifest, "artifact_manifest"),
    ):
        _check(payload.get("dataset_id") == DATASET_ID, f"{name}_dataset_id", checks)
        _check(
            not _contains_forbidden_key(payload), f"{name}_aggregate_privacy", checks
        )
    _check(
        quality["row_filter"]["conflict_group_count"] == 2,
        "quality_conflict_groups",
        checks,
    )
    _check(
        quality["row_filter"]["conflict_row_count"] == 4,
        "quality_conflict_rows",
        checks,
    )
    _check(
        quality["row_filter"]["canonical_rows"] == 5_327,
        "quality_canonical_rows",
        checks,
    )
    _check(
        quality["row_filter"]["arbitrary_rows_kept"] == 0,
        "quality_no_arbitrary_keep",
        checks,
    )
    _check(quality["labels"]["positive_count"] == 186, "quality_positive_count", checks)
    _check(
        set(mapping["base_supported_feature_set"]) == MAPPED_TARGETS,
        "mapping_supported_set",
        checks,
    )
    _check(
        mapping["fold_supported_feature_set"] == [], "mapping_no_fold_features", checks
    )
    _check(
        mapping["production_input_policy"]["self_report_sleep_is_production_input"]
        is False,
        "mapping_sleep_excluded",
        checks,
    )
    _check(
        mapping["production_input_policy"]["chronic_binary_is_mapped_to_count"]
        is False,
        "mapping_chronic_semantics",
        checks,
    )
    _check(
        metadata["canonical_frame_sha256"] == _frame_sha256(canonical, expected_schema),
        "metadata_frame_sha256",
        checks,
    )
    _check(
        metadata["excluded_conflict_group_count"] == 2,
        "metadata_conflict_groups",
        checks,
    )
    _check(
        metadata["excluded_conflict_row_count"] == 4, "metadata_conflict_rows", checks
    )
    _check(
        metadata["formal_ecdf_instance_created"] is False, "metadata_no_ecdf", checks
    )

    manifest_artifacts = artifact_manifest["artifacts"]
    expected_artifacts = {
        "shenzhen/canonical_shenzhen.parquet": canonical_path,
        "shenzhen/data_quality_report.json": quality_path,
        "shenzhen/adapter_metadata.json": metadata_path,
        "mappings/shenzhen_v3_3_3_mapping.json": mapping_path,
    }
    _check(
        set(manifest_artifacts) == set(expected_artifacts),
        "artifact_manifest_paths",
        checks,
    )
    _check(artifact_manifest["artifact_count"] == 4, "artifact_manifest_count", checks)
    _check(artifact_manifest["complete"] is True, "artifact_manifest_complete", checks)
    for relative, path in expected_artifacts.items():
        row = manifest_artifacts[relative]
        _check(
            path.stat().st_size == row["bytes"], f"artifact_size_{path.name}", checks
        )
        _check(
            _sha256_file(path) == row["sha256"], f"artifact_sha256_{path.name}", checks
        )
    return {
        "check_count": len(checks),
        "checks": checks,
        "dataset_id": DATASET_ID,
        "participant_count": 5_327,
        "positive_count": 186,
        "status": "pass",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    workspace_root = _find_workspace_root(args.workspace_root or Path.cwd())
    output_root = (
        args.output_root
        or workspace_root
        / "algorithm"
        / "eldercare-risk-ai-main"
        / DEFAULT_OUTPUT_RELATIVE
    )
    print(
        json.dumps(
            validate(workspace_root, output_root),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
