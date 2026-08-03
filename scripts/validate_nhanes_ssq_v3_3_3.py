"""Independently validate DATA-006 NHANES DPQ+SSQ V3.3.3 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
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
FEATURE_SCHEMA_SHA256 = (
    "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
)
SOURCE_TOTAL_BYTES = 9_762_095
TIMESCALE = "cross_sectional_concurrent_association"
KEY_POLICY = "cycle_plus_integer_seqn_v1"
CYCLES = ("D", "E")
SURVEY_CYCLES = {"D": "2005-2006", "E": "2007-2008"}
DPQ_FIELDS = tuple(f"DPQ0{index}0" for index in range(1, 10))
SSQ_SUPPORT_FIELDS = tuple(f"SSQ021{letter}" for letter in "ABCDEFGHIJKLMN")
SSQ_FIELDS = (
    "SSQ011",
    *SSQ_SUPPORT_FIELDS,
    "SSQ031",
    "SSQ041",
    "SSD044",
    "SSQ051",
    "SSQ061",
)
DEMO_FIELDS = (
    "RIDAGEYR",
    "RIAGENDR",
    "DMDMARTL",
    "DMDEDUC2",
    "INDHHINC",
    "INDHHIN2",
    "INDFMPIR",
)
SOURCE_FIELDS = (*DEMO_FIELDS, "DPQ100", *SSQ_FIELDS)
INTEGER_SOURCE_FIELDS = {field for field in SOURCE_FIELDS if field != "INDFMPIR"}
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
MAPPED_TARGETS = {
    "social_context.age_group",
    "social_context.sex",
    "social_context.marital_status",
    "social_context.education_level",
}
EXPECTED_CYCLE_COUNTS = {"D": 1_311, "E": 1_839}
EXPECTED_POSITIVE_BY_CYCLE = {"D": 59, "E": 126}
EXPECTED_SEVERITY = {0: 2_521, 1: 444, 2: 120, 3: 50, 4: 15}
_EPSILON = 1e-12


class ValidationError(RuntimeError):
    """Raised without including participant values in the message."""


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
    rows = [
        {"group": group, **dict(spec)}
        for group in schema["group_order"]
        for spec in schema["groups"][group]
    ]
    if len(rows) != 54:
        raise ValidationError("MH-003 feature count changed")
    return rows


def _feature_columns(schema: Mapping[str, Any]) -> list[str]:
    return [f"{spec['group']}.{spec['name']}" for spec in _target_specs(schema)]


def _expected_columns(schema: Mapping[str, Any]) -> list[str]:
    features = _feature_columns(schema)
    return [
        *IDENTITY_COLUMNS,
        *TARGET_COLUMNS,
        *(f"source__{field}" for field in SOURCE_FIELDS),
        *features,
        *(f"feature_mask.{field}" for field in features),
    ]


def _expected_arrow_schema(schema: Mapping[str, Any]) -> pa.Schema:
    types: dict[str, pa.DataType] = {
        name: pa.large_string() for name in IDENTITY_COLUMNS
    }
    types["seqn"] = pa.int64()
    types.update({name: pa.int8() for name in TARGET_COLUMNS})
    types.update(
        {
            f"source__{field}": (
                pa.int64() if field in INTEGER_SOURCE_FIELDS else pa.float64()
            )
            for field in SOURCE_FIELDS
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
                frame[field.name].tolist(),
                type=field.type,
                from_pandas=True,
                safe=True,
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
        writer.write_table(table, max_chunksize=max(len(table), 1))
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _normalise_numeric(series: pd.Series, field: str) -> pd.Series:
    try:
        values = pd.to_numeric(series, errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"source field is not numeric: {field}") from exc
    finite = np.isfinite(values.to_numpy())
    if bool((~finite & ~values.isna().to_numpy()).any()):
        raise ValidationError(f"source field contains infinity: {field}")
    values.loc[values.abs().lt(_EPSILON)] = 0.0
    return values


def _normalise_integer(
    series: pd.Series, field: str, *, allow_missing: bool
) -> pd.Series:
    values = _normalise_numeric(series, field)
    finite = values.dropna()
    if bool((finite != np.floor(finite)).any()):
        raise ValidationError(f"source field is not integral: {field}")
    if not allow_missing and bool(values.isna().any()):
        raise ValidationError(f"source key is missing: {field}")
    return pd.Series(pd.array(values, dtype="Int64"), index=series.index)


def _prepare_table(
    frame: pd.DataFrame,
    required: Sequence[str],
    table_name: str,
) -> pd.DataFrame:
    if not set(required).issubset(frame.columns):
        raise ValidationError(f"source columns changed: {table_name}")
    output = frame.loc[:, list(required)].copy()
    output["SEQN"] = _normalise_integer(
        output["SEQN"], f"{table_name}.SEQN", allow_missing=False
    )
    if bool(output["SEQN"].duplicated().any()):
        raise ValidationError(f"source key is duplicated: {table_name}")
    return output


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


def _profile_values(row: Mapping[str, Any]) -> dict[str, Any]:
    age = int(row["RIDAGEYR"])
    sex_code = None if pd.isna(row["RIAGENDR"]) else int(row["RIAGENDR"])
    marital_code = None if pd.isna(row["DMDMARTL"]) else int(row["DMDMARTL"])
    education_code = None if pd.isna(row["DMDEDUC2"]) else int(row["DMDEDUC2"])
    return {
        "social_context.age_group": (
            "60_69" if age < 70 else "70_79" if age < 80 else "80_plus"
        ),
        "social_context.sex": {1: "male", 2: "female"}.get(sex_code),
        "social_context.marital_status": (
            "partnered"
            if marital_code in {1, 6}
            else "not_partnered"
            if marital_code in {2, 3, 4, 5}
            else None
        ),
        "social_context.education_level": (
            "primary_or_less"
            if education_code == 1
            else "middle"
            if education_code in {2, 3}
            else "high_or_above"
            if education_code in {4, 5}
            else None
        ),
    }


def _read_and_rebuild(
    workspace_root: Path,
    schema_manifest: Mapping[str, Any],
    checks: list[dict[str, Any]],
) -> pd.DataFrame:
    source_root = workspace_root / SOURCE_RELATIVE
    records: list[dict[str, Any]] = []
    source_rows: dict[str, dict[str, int]] = {}
    for cycle in CYCLES:
        try:
            demo_raw = pd.read_sas(
                source_root / f"DEMO_{cycle}.xpt",
                format="xport",
                encoding="utf-8",
            )
            dpq_raw = pd.read_sas(
                source_root / f"DPQ_{cycle}.xpt",
                format="xport",
                encoding="utf-8",
            )
            ssq_raw = pd.read_sas(
                source_root / f"SSQ_{cycle}.xpt",
                format="xport",
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            raise ValidationError("official XPT is not readable") from exc
        source_rows[cycle] = {
            "DEMO": len(demo_raw),
            "DPQ": len(dpq_raw),
            "SSQ": len(ssq_raw),
        }
        income = "INDHHINC" if cycle == "D" else "INDHHIN2"
        demo = _prepare_table(
            demo_raw,
            (
                "SEQN",
                "RIDAGEYR",
                "RIAGENDR",
                "DMDMARTL",
                "DMDEDUC2",
                income,
                "INDFMPIR",
            ),
            f"DEMO_{cycle}",
        )
        for field in ("RIDAGEYR", "RIAGENDR", "DMDMARTL", "DMDEDUC2", income):
            demo[field] = _normalise_integer(
                demo[field], f"DEMO_{cycle}.{field}", allow_missing=True
            )
        demo["INDFMPIR"] = pd.array(
            _normalise_numeric(demo["INDFMPIR"], f"DEMO_{cycle}.INDFMPIR"),
            dtype="Float64",
        )
        other_income = "INDHHIN2" if cycle == "D" else "INDHHINC"
        demo[other_income] = pd.array([pd.NA] * len(demo), dtype="Int64")
        demo = demo.loc[:, ["SEQN", *DEMO_FIELDS]]

        dpq = _prepare_table(dpq_raw, ("SEQN", *DPQ_FIELDS, "DPQ100"), f"DPQ_{cycle}")
        complete = pd.Series(True, index=dpq.index)
        for field in DPQ_FIELDS:
            numeric = _normalise_numeric(dpq[field], f"DPQ_{cycle}.{field}")
            finite = numeric.dropna()
            _check(
                bool((finite == np.floor(finite)).all()),
                f"source_{cycle}_{field}_integral",
                checks,
            )
            valid = numeric.isin([0.0, 1.0, 2.0, 3.0])
            dpq[field] = pd.array(numeric.where(valid), dtype="Int8")
            complete &= valid
        dpq["DPQ100"] = _normalise_integer(
            dpq["DPQ100"], f"DPQ_{cycle}.DPQ100", allow_missing=True
        )
        dpq = dpq.loc[complete].copy()

        ssq = _prepare_table(ssq_raw, ("SEQN", *SSQ_FIELDS), f"SSQ_{cycle}")
        for field in SSQ_FIELDS:
            ssq[field] = _normalise_integer(
                ssq[field], f"SSQ_{cycle}.{field}", allow_missing=True
            )

        joined = (
            demo.loc[demo["RIDAGEYR"].ge(60).fillna(False)]
            .merge(dpq, on="SEQN", how="inner", validate="one_to_one")
            .merge(ssq, on="SEQN", how="inner", validate="one_to_one")
            .sort_values("SEQN", kind="stable")
        )
        _check(
            len(joined) == EXPECTED_CYCLE_COUNTS[cycle],
            f"source_{cycle}_canonical_count",
            checks,
        )
        feature_names = _feature_columns(schema_manifest)
        for row in joined.to_dict(orient="records"):
            seqn = int(row["SEQN"])
            participant_id = f"{cycle}:{seqn}"
            record: dict[str, Any] = {
                "dataset_id": DATASET_ID,
                "global_participant_id": f"{DATASET_ID}::{participant_id}",
                "participant_id": participant_id,
                "cycle": cycle,
                "survey_cycle": SURVEY_CYCLES[cycle],
                "seqn": seqn,
                "timescale_semantics": TIMESCALE,
                "participant_key_policy": KEY_POLICY,
            }
            total = 0
            for field in DPQ_FIELDS:
                value = int(row[field])
                record[field] = value
                total += value
            record["phq9_total"] = total
            record["phq9_severity"] = _severity(total)
            record["binary_target"] = int(total >= 10)
            for field in SOURCE_FIELDS:
                record[f"source__{field}"] = None if pd.isna(row[field]) else row[field]
            profile = _profile_values(row)
            for target in feature_names:
                value = profile.get(target)
                record[target] = value
                record[f"feature_mask.{target}"] = int(value is not None)
            records.append(record)
    _check(
        source_rows
        == {
            "D": {"DEMO": 10_348, "DPQ": 5_334, "SSQ": 3_056},
            "E": {"DEMO": 10_149, "DPQ": 5_995, "SSQ": 4_025},
        },
        "source_row_counts",
        checks,
    )
    frame = pd.DataFrame.from_records(records)
    frame = frame.sort_values(["cycle", "seqn"], kind="stable").reset_index(drop=True)
    return frame.loc[:, _expected_columns(schema_manifest)]


def _validate_source_binding(
    workspace_root: Path, checks: list[dict[str, Any]]
) -> Mapping[str, Any]:
    algorithm_root = workspace_root / "algorithm" / "eldercare-risk-ai-main"
    manifests = algorithm_root / DEFAULT_OUTPUT_RELATIVE / "manifests"
    data001_manifest = manifests / "dataset_file_manifest.jsonl"
    schema_snapshot = manifests / "feature_schema_manifest.json"
    _check(
        _sha256_file(data001_manifest) == DATA001_FILE_MANIFEST_SHA256,
        "data001_manifest_sha256",
        checks,
    )
    live_schema = feature_schema_manifest()
    _check(
        hashlib.sha256(_canonical_json_bytes(live_schema)).hexdigest()
        == FEATURE_SCHEMA_SHA256,
        "live_schema_sha256",
        checks,
    )
    _check(
        _sha256_file(schema_snapshot) == FEATURE_SCHEMA_SHA256,
        "schema_snapshot_sha256",
        checks,
    )
    _check(
        _load_json(schema_snapshot) == live_schema,
        "schema_snapshot_content",
        checks,
    )

    source_root = workspace_root / SOURCE_RELATIVE
    manifest_path = source_root / SOURCE_MANIFEST_NAME
    _check(
        _sha256_file(manifest_path) == SOURCE_MANIFEST_SHA256,
        "source_manifest_sha256",
        checks,
    )
    manifest = _load_json(manifest_path)
    _check(manifest["dataset_id"] == DATASET_ID, "source_manifest_dataset", checks)
    _check(manifest["file_count"] == 12, "source_manifest_file_count", checks)
    _check(
        manifest["total_bytes"] == SOURCE_TOTAL_BYTES,
        "source_manifest_total_bytes",
        checks,
    )
    _check(
        manifest["source_collection_sha256"] == SOURCE_COLLECTION_SHA256,
        "source_manifest_collection_sha256",
        checks,
    )
    current_rows: list[dict[str, Any]] = []
    for row in manifest["files"]:
        path = workspace_root / Path(row["relative_path"])
        digest = _sha256_file(path)
        _check(
            path.stat().st_size == int(row["bytes"]),
            f"source_size_{path.name}",
            checks,
        )
        _check(
            digest == str(row["sha256"]),
            f"source_sha256_{path.name}",
            checks,
        )
        _check(
            str(row["url"]).startswith("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/"),
            f"source_url_{path.name}",
            checks,
        )
        _check(
            int(row["field_count"]) == len(row["fields"]),
            f"source_field_count_{path.name}",
            checks,
        )
        current_rows.append(
            {
                "relative_path": row["relative_path"],
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    _check(
        _collection_sha256(current_rows) == SOURCE_COLLECTION_SHA256,
        "live_source_collection_sha256",
        checks,
    )
    by_name = {row["file_name"]: row for row in manifest["files"]}
    _check(
        by_name["DPQ_D.xpt"]["field_count"] == 11,
        "dpq_d_xpt_field_count",
        checks,
    )
    _check(
        by_name["DPQ_D.htm"]["field_count"] == 13,
        "dpq_d_codebook_field_count",
        checks,
    )
    _check(
        {"DPQ001", "DPQ095"}.isdisjoint(by_name["DPQ_D.xpt"]["fields"]),
        "dpq_d_xpt_excludes_documented_only_fields",
        checks,
    )
    return live_schema


def validate(workspace_root: Path, output_root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    schema_manifest = _validate_source_binding(workspace_root, checks)
    expected = _read_and_rebuild(workspace_root, schema_manifest, checks)
    expected_schema = _expected_arrow_schema(schema_manifest)
    dataset_root = output_root / DATASET_ID
    canonical_path = dataset_root / "canonical_nhanes_ssq_2005_2008.parquet"
    quality_path = dataset_root / "data_quality_report.json"
    metadata_path = dataset_root / "adapter_metadata.json"
    mapping_path = output_root / "mappings" / "nhanes_ssq_2005_2008_v3_3_3_mapping.json"
    artifact_manifest_path = dataset_root / "artifact_manifest.json"
    try:
        actual_table = pq.read_table(canonical_path)
        actual = actual_table.to_pandas()
    except (OSError, pa.ArrowException) as exc:
        raise ValidationError("canonical Parquet is not readable") from exc
    _check(actual_table.schema == expected_schema, "canonical_arrow_schema", checks)
    _check(
        list(actual.columns) == _expected_columns(schema_manifest),
        "canonical_column_order",
        checks,
    )
    _check(len(actual) == 3_150, "canonical_row_count", checks)
    _check(actual["global_participant_id"].is_unique, "global_key_unique", checks)
    _check(
        not any("x_source" in name for name in actual.columns),
        "canonical_has_no_x_source",
        checks,
    )

    expected_table = _normalised_arrow_table(expected, expected_schema)
    normalised_actual = _normalised_arrow_table(actual, expected_schema)
    for name in _expected_columns(schema_manifest):
        _check(
            expected_table[name].equals(normalised_actual[name]),
            f"canonical_column_{name}",
            checks,
        )

    severity = {
        int(key): int(value)
        for key, value in actual["phq9_severity"].value_counts().sort_index().items()
    }
    _check(severity == EXPECTED_SEVERITY, "severity_distribution", checks)
    cycle_counts = {
        str(key): int(value)
        for key, value in actual["cycle"].value_counts().sort_index().items()
    }
    _check(cycle_counts == EXPECTED_CYCLE_COUNTS, "cycle_distribution", checks)
    _check(
        int(actual["binary_target"].sum()) == 185,
        "positive_count",
        checks,
    )
    for target in _feature_columns(schema_manifest):
        mask = f"feature_mask.{target}"
        _check(
            actual[mask].astype("int8").equals(actual[target].notna().astype("int8")),
            f"mask_matches_{target}",
            checks,
        )
        if target not in MAPPED_TARGETS:
            _check(
                actual[target].isna().all() and actual[mask].eq(0).all(),
                f"unsupported_null_{target}",
                checks,
            )

    quality = _load_json(quality_path)
    metadata = _load_json(metadata_path)
    mapping = _load_json(mapping_path)
    artifact_manifest = _load_json(artifact_manifest_path)
    _check(quality["dataset_id"] == DATASET_ID, "quality_dataset", checks)
    _check(
        quality["participant_summary"]["participant_count"] == 3_150,
        "quality_participant_count",
        checks,
    )
    _check(
        quality["labels"]["binary_positive_count"] == 185,
        "quality_positive_count",
        checks,
    )
    _check(
        quality["ssq_audit"]["s10_social_contact_mapping_allowed"] is False,
        "quality_ssq_s10_boundary",
        checks,
    )
    _check(
        quality["x_source"]["formal_ecdf_instance_created"] is False,
        "quality_no_ecdf",
        checks,
    )
    _check(metadata["dataset_id"] == DATASET_ID, "metadata_dataset", checks)
    _check(
        metadata["canonical_frame_sha256"] == _frame_sha256(actual, expected_schema),
        "metadata_frame_sha256",
        checks,
    )
    _check(metadata["x_source_created"] is False, "metadata_no_x_source", checks)
    _check(
        metadata["formal_ecdf_instance_created"] is False,
        "metadata_no_ecdf",
        checks,
    )
    _check(mapping["dataset_id"] == DATASET_ID, "mapping_dataset", checks)
    _check(len(mapping["target_fields"]) == 54, "mapping_target_count", checks)
    mapped = {
        row["canonical_value_column"]
        for row in mapping["target_fields"]
        if row["mapping_status"] == "mapped"
    }
    _check(mapped == MAPPED_TARGETS, "mapping_supported_targets", checks)
    _check(
        all(
            not row["source_fields"]
            for row in mapping["target_fields"]
            if row["group"] == "social_contact"
        ),
        "mapping_social_contact_has_no_sources",
        checks,
    )
    _check(
        mapping["ssq_boundary"]["SSQ061_maps_to_active_contact_count"] is False,
        "mapping_ssq061_boundary",
        checks,
    )
    _check(
        mapping["ecdf_contract"]["applicable"] is False,
        "mapping_no_ecdf",
        checks,
    )

    expected_artifacts = {
        f"{DATASET_ID}/canonical_nhanes_ssq_2005_2008.parquet": canonical_path,
        f"{DATASET_ID}/data_quality_report.json": quality_path,
        f"{DATASET_ID}/adapter_metadata.json": metadata_path,
        "mappings/nhanes_ssq_2005_2008_v3_3_3_mapping.json": mapping_path,
    }
    _check(
        set(artifact_manifest["artifacts"]) == set(expected_artifacts),
        "artifact_manifest_paths",
        checks,
    )
    _check(
        artifact_manifest["artifact_count"] == 4,
        "artifact_manifest_count",
        checks,
    )
    _check(
        artifact_manifest["complete"] is True,
        "artifact_manifest_complete",
        checks,
    )
    for relative, path in expected_artifacts.items():
        row = artifact_manifest["artifacts"][relative]
        _check(
            path.stat().st_size == int(row["bytes"]),
            f"artifact_size_{path.name}",
            checks,
        )
        _check(
            _sha256_file(path) == str(row["sha256"]),
            f"artifact_sha256_{path.name}",
            checks,
        )
    _check(
        len(checks) >= 200,
        "independent_check_count_minimum",
        checks,
    )
    return {
        "check_count": len(checks),
        "checks": checks,
        "dataset_id": DATASET_ID,
        "participant_count": 3_150,
        "positive_count": 185,
        "status": "pass",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
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
