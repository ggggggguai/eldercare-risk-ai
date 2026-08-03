"""Independently validate DATA-002 inputs and generated PSYCHE-D artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
SOURCE_RELATIVE = Path("数据集/心理/PSYCHE-D")
OUTPUT_RELATIVE = Path("data/processed/mental_health/mood_social/v3.3.3")
MANIFEST_SHA256 = "59257b21fee44bd405e0be6f299d49ad945c6e6289002717f8769389dce32bd5"
COLLECTION_SHA256 = "a29ba036dcc0dad646e8817a5bd6892d2cb25ef29f7db388bc74910f91145a59"
SCHEMA_SHA256 = "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
EXPECTED_ROWS = 10_866
EXPECTED_PARTICIPANTS = 4_036
EXPECTED_POSITIVES = 3_444
EXPECTED_END_CATEGORIES = {0: 4_202, 1: 3_220, 2: 1_941, 3: 981, 4: 522}
LABEL_FIELDS = (
    "phq9_score_start",
    "phq9_score_end",
    "phq9_cat_start",
    "phq9_cat_end",
)
EXPECTED_ARTIFACTS = {
    "mappings/psyche_d_v3_3_3_mapping.json",
    "psyche_d/adapter_metadata.json",
    "psyche_d/canonical_psyche_d.parquet",
    "psyche_d/data_quality_report.json",
}


class ValidationError(RuntimeError):
    """Raised when an independently observed DATA-002 contract does not match."""


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


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
        raise ValidationError(f"independent DATA-002 check failed: {name}")
    checks.append({"check": name, "passed": True})


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValidationError("file manifest has a non-object row")
                    rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError("file manifest is not readable") from exc
    return rows


def _source_fields(source_root: Path) -> list[str]:
    try:
        payload = yaml.safe_load(
            (source_root / "p0_feature_mapping_v1.yaml").read_text(encoding="utf-8")
        )
        return [
            str(row["source_field"])
            for group in ("steps", "sleep")
            for row in payload["features"][group]
        ]
    except (OSError, UnicodeError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise ValidationError("source whitelist is not readable") from exc


def _expected_columns(
    schema: Mapping[str, Any], source_fields: Sequence[str]
) -> list[str]:
    feature_columns = [
        f"{group}.{spec['name']}"
        for group in schema["group_order"]
        for spec in schema["groups"][group]
    ]
    mask_columns = [f"feature_mask.{name}" for name in feature_columns]
    return [
        "dataset_id",
        "global_participant_id",
        "participant_id",
        "nominal_month",
        "timescale_semantics",
        *LABEL_FIELDS,
        "binary_target",
        "x_source_name",
        "x_source_value",
        "x_source_mask",
        *(f"source__{name}" for name in source_fields),
        *feature_columns,
        *mask_columns,
    ]


def _expected_binding() -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "source_relative": SOURCE_RELATIVE.as_posix(),
        "source_collection_sha256": COLLECTION_SHA256,
        "file_manifest_sha256": MANIFEST_SHA256,
        "feature_schema_sha256": SCHEMA_SHA256,
        "source_file_count": 4,
        "all_bindings_validated": True,
    }


def _numeric_values(series: pd.Series) -> np.ndarray:
    try:
        values = pd.to_numeric(series, errors="raise").to_numpy(
            dtype="float64", na_value=np.nan
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError("canonical numeric values are invalid") from exc
    if np.isinf(values).any():
        raise ValidationError("canonical numeric values contain infinity")
    return values


def _numeric_equal(actual: pd.Series, expected: Sequence[Any] | np.ndarray) -> bool:
    actual_values = _numeric_values(actual)
    expected_values = pd.to_numeric(pd.Series(expected), errors="raise").to_numpy(
        dtype="float64", na_value=np.nan
    )
    return bool(
        np.allclose(
            actual_values,
            expected_values,
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
    )


def _parse_source_population(source: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    participants: list[str] = []
    months: list[int] = []
    for value in source.index.tolist():
        if not isinstance(value, str):
            raise ValidationError("source sample index contains a non-string key")
        participant, separator, month_text = value.rpartition("_")
        if not separator or not participant or not month_text.isdigit():
            raise ValidationError("source sample index contains a malformed key")
        participants.append(participant)
        months.append(int(month_text))
    complete = source[list(LABEL_FIELDS)].notna().all(axis=1).to_numpy()
    positions = np.flatnonzero(complete)
    population = pd.DataFrame(
        {
            "participant_id": np.asarray(participants, dtype=object)[positions],
            "nominal_month": np.asarray(months, dtype=np.int64)[positions],
            "source_position": positions,
        }
    ).sort_values(["participant_id", "nominal_month"], kind="stable")
    return population.reset_index(drop=True), population["source_position"].to_numpy()


def _expected_arrow_types(
    schema: Mapping[str, Any], source_fields: Sequence[str]
) -> dict[str, pa.DataType]:
    expected: dict[str, pa.DataType] = {
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
    expected.update({f"source__{name}": pa.float64() for name in source_fields})
    value_types = {
        "float": pa.float64(),
        "integer": pa.int64(),
        "category": pa.large_string(),
    }
    for group in schema["group_order"]:
        for spec in schema["groups"][group]:
            target = f"{group}.{spec['name']}"
            expected[target] = value_types[spec["value_type"]]
            expected[f"feature_mask.{target}"] = pa.int8()
    return expected


def _expected_arrow_schema(
    schema: Mapping[str, Any], source_fields: Sequence[str]
) -> pa.Schema:
    field_types = _expected_arrow_types(schema, source_fields)
    return pa.schema(
        [
            pa.field(name, field_types[name])
            for name in _expected_columns(schema, source_fields)
        ]
    )


def _canonical_frame_sha256(
    frame: pd.DataFrame,
    schema: Mapping[str, Any],
    source_fields: Sequence[str],
) -> str:
    expected_columns = _expected_columns(schema, source_fields)
    if list(frame.columns) != expected_columns:
        raise ValidationError("canonical frame columns do not match the frozen schema")
    expected_schema = _expected_arrow_schema(schema, source_fields)
    try:
        arrays = [
            pa.array(
                frame[field.name].tolist(),
                type=field.type,
                from_pandas=True,
                safe=True,
            )
            for field in expected_schema
        ]
        table = pa.Table.from_arrays(arrays, schema=expected_schema).combine_chunks()
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise ValidationError(
            "canonical frame values do not match the frozen Arrow schema"
        ) from exc
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table, max_chunksize=max(len(table), 1))
    return _sha256_bytes(sink.getvalue().to_pybytes())


def validate(
    *,
    workspace_root: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    root = _find_workspace_root(workspace_root or Path.cwd())
    repository_root = root / "algorithm" / "eldercare-risk-ai-main"
    destination = output_root or repository_root / OUTPUT_RELATIVE
    manifests_root = repository_root / OUTPUT_RELATIVE / "manifests"
    source_root = root / SOURCE_RELATIVE
    checks: list[dict[str, Any]] = []

    manifest_path = manifests_root / "dataset_file_manifest.jsonl"
    _check(
        _sha256_file(manifest_path) == MANIFEST_SHA256,
        "data001_file_manifest_sha256",
        checks,
    )
    schema = feature_schema_manifest()
    _check(
        _sha256_bytes(_canonical_json_bytes(schema)) == SCHEMA_SHA256,
        "live_mh003_schema_sha256",
        checks,
    )
    _check(
        _sha256_file(manifests_root / "feature_schema_manifest.json") == SCHEMA_SHA256,
        "data001_schema_snapshot_sha256",
        checks,
    )

    source_manifest_rows = [
        row
        for row in _manifest_rows(manifest_path)
        if row.get("dataset_id") == DATASET_ID
    ]
    _check(len(source_manifest_rows) == 4, "psyche_d_source_file_count", checks)
    current_rows: list[dict[str, Any]] = []
    for row in source_manifest_rows:
        path = root / Path(str(row["relative_path"]))
        size = path.stat().st_size
        file_sha256 = _sha256_file(path)
        _check(size == int(row["bytes"]), "source_file_size", checks)
        _check(file_sha256 == row["sha256"], "source_file_sha256", checks)
        current_rows.append(
            {
                "relative_path": row["relative_path"],
                "bytes": size,
                "sha256": file_sha256,
            }
        )
    _check(
        _collection_sha256(current_rows) == COLLECTION_SHA256,
        "psyche_d_collection_sha256",
        checks,
    )

    artifact_manifest_path = destination / "psyche_d" / "artifact_manifest.json"
    artifact_manifest = _load_json(artifact_manifest_path)
    _check(
        artifact_manifest.get("complete") is True, "artifact_manifest_complete", checks
    )
    _check(
        artifact_manifest.get("dataset_id") == DATASET_ID,
        "artifact_manifest_dataset_id",
        checks,
    )
    _check(
        artifact_manifest.get("input_binding") == _expected_binding(),
        "artifact_manifest_input_binding",
        checks,
    )
    artifacts = artifact_manifest.get("artifacts")
    _check(
        isinstance(artifacts, dict) and set(artifacts) == EXPECTED_ARTIFACTS,
        "artifact_exact_path_set",
        checks,
    )
    for relative, expected in artifacts.items():
        path = destination / Path(relative)
        _check(path.is_file(), "artifact_exists", checks)
        _check(path.stat().st_size == int(expected["bytes"]), "artifact_size", checks)
        _check(_sha256_file(path) == expected["sha256"], "artifact_sha256", checks)

    source_fields = _source_fields(source_root)
    _check(len(source_fields) == 27, "source_whitelist_field_count", checks)
    source = pd.read_parquet(
        source_root / "anon_processed_df_parquet",
        columns=[*source_fields, *LABEL_FIELDS],
    )
    population, source_positions = _parse_source_population(source)
    _check(len(source) == 35_694, "source_row_count", checks)
    _check(len(population) == EXPECTED_ROWS, "source_complete_label_count", checks)
    canonical_path = destination / "psyche_d" / "canonical_psyche_d.parquet"
    canonical = pd.read_parquet(canonical_path)
    parquet_schema = pq.read_schema(canonical_path)
    _check(len(canonical) == EXPECTED_ROWS, "canonical_row_count", checks)
    _check(
        int(canonical["participant_id"].nunique()) == EXPECTED_PARTICIPANTS,
        "canonical_participant_count",
        checks,
    )
    _check(
        int(canonical["binary_target"].sum()) == EXPECTED_POSITIVES,
        "canonical_positive_count",
        checks,
    )
    category_counts = {
        int(key): int(value)
        for key, value in canonical["phq9_cat_end"].value_counts().items()
    }
    _check(
        category_counts == EXPECTED_END_CATEGORIES,
        "canonical_end_category_distribution",
        checks,
    )
    _check(
        parquet_schema.names == _expected_columns(schema, source_fields),
        "canonical_column_order",
        checks,
    )
    expected_types = _expected_arrow_types(schema, source_fields)
    _check(
        all(
            parquet_schema.field(name).type == expected_type
            for name, expected_type in expected_types.items()
        ),
        "canonical_arrow_dtypes",
        checks,
    )
    _check(
        not any("date" in name for name in canonical.columns),
        "no_fabricated_natural_date",
        checks,
    )
    _check(
        set(canonical["timescale_semantics"].astype(str))
        == {"timescale_proxy_transfer"},
        "timescale_proxy_transfer_marked",
        checks,
    )
    _check(
        canonical["activity.activity_volume_norm"].isna().all()
        and int(canonical["feature_mask.activity.activity_volume_norm"].sum()) == 0,
        "base_canonical_ecdf_not_fitted",
        checks,
    )
    _check(
        canonical[list(LABEL_FIELDS)].notna().all(axis=None),
        "four_labels_complete",
        checks,
    )

    expected_participants = population["participant_id"].astype(str)
    expected_months = population["nominal_month"].to_numpy()
    _check(
        canonical["dataset_id"].eq(DATASET_ID).all()
        and canonical["participant_id"].astype(str).tolist()
        == expected_participants.tolist()
        and _numeric_equal(canonical["nominal_month"], expected_months),
        "canonical_identity_and_row_order",
        checks,
    )
    _check(
        canonical["global_participant_id"].astype(str).tolist()
        == (DATASET_ID + "::" + expected_participants).tolist(),
        "canonical_global_participant_keys",
        checks,
    )
    for label in LABEL_FIELDS:
        _check(
            _numeric_equal(canonical[label], source[label].iloc[source_positions]),
            f"canonical_label_values_{label}",
            checks,
        )
    expected_binary = (
        pd.to_numeric(source["phq9_score_end"].iloc[source_positions]).to_numpy() >= 10
    ).astype("float64")
    _check(
        _numeric_equal(canonical["binary_target"], expected_binary),
        "canonical_binary_target_formula",
        checks,
    )
    for source_field in source_fields:
        _check(
            _numeric_equal(
                canonical[f"source__{source_field}"],
                source[source_field].iloc[source_positions],
            ),
            f"canonical_source_values_{source_field}",
            checks,
        )
    expected_x_source = pd.to_numeric(
        source["steps_awake_mean"].iloc[source_positions]
    ).to_numpy(dtype="float64", na_value=np.nan)
    _check(
        canonical["x_source_name"].eq("steps_awake_mean").all()
        and _numeric_equal(canonical["x_source_value"], expected_x_source)
        and _numeric_equal(
            canonical["x_source_mask"], (~np.isnan(expected_x_source)).astype("int8")
        ),
        "canonical_x_source_values_and_mask",
        checks,
    )

    asleep = pd.to_numeric(
        source["sleep_asleep_mean_recent"].iloc[source_positions]
    ).to_numpy(dtype="float64", na_value=np.nan)
    in_bed = pd.to_numeric(
        source["sleep_in_bed_mean_recent"].iloc[source_positions]
    ).to_numpy(dtype="float64", na_value=np.nan)
    efficiency = pd.to_numeric(
        source["sleep_ratio_asleep_in_bed_mean_recent"].iloc[source_positions]
    ).to_numpy(dtype="float64", na_value=np.nan)
    adjusted = pd.to_numeric(
        source["sleep_main_start_hour_adj_median"].iloc[source_positions]
    ).to_numpy(dtype="float64", na_value=np.nan)
    angle = 2.0 * math.pi * np.mod(adjusted, 24.0) / 24.0
    expected_direct = {
        "sleep.sleep_duration_norm": asleep / 1440.0,
        "sleep.time_in_bed_norm": in_bed / 1440.0,
        "sleep.sleep_efficiency": efficiency,
        "sleep.sleep_onset_sin": np.sin(angle),
        "sleep.sleep_onset_cos": np.cos(angle),
    }
    for target, expected in expected_direct.items():
        _check(
            _numeric_equal(canonical[target], expected)
            and _numeric_equal(
                canonical[f"feature_mask.{target}"],
                (~np.isnan(expected)).astype("int8"),
            ),
            f"canonical_mapped_values_and_mask_{target}",
            checks,
        )
    expected_targets = [
        f"{group}.{spec['name']}"
        for group in schema["group_order"]
        for spec in schema["groups"][group]
    ]
    unsupported_targets = set(expected_targets) - set(expected_direct)
    _check(
        all(
            canonical[target].isna().all()
            and int(canonical[f"feature_mask.{target}"].sum()) == 0
            for target in unsupported_targets
        ),
        "canonical_unsupported_targets_null_and_masked",
        checks,
    )

    mapping = _load_json(destination / "mappings" / "psyche_d_v3_3_3_mapping.json")
    metadata = _load_json(destination / "psyche_d" / "adapter_metadata.json")
    quality = _load_json(destination / "psyche_d" / "data_quality_report.json")
    _check(
        artifact_manifest["input_binding"]
        == metadata.get("input_binding")
        == quality.get("input_binding")
        == _expected_binding(),
        "json_input_bindings_match",
        checks,
    )
    _check(
        mapping.get("source_collection_sha256") == COLLECTION_SHA256
        and mapping.get("file_manifest_sha256") == MANIFEST_SHA256
        and mapping.get("feature_schema_sha256") == SCHEMA_SHA256,
        "mapping_input_bindings",
        checks,
    )
    _check(
        [row["source_field"] for row in mapping["source_fields"]] == source_fields,
        "mapping_source_coverage",
        checks,
    )
    _check(
        [row["canonical_value_column"] for row in mapping["target_fields"]]
        == expected_targets,
        "mapping_target_coverage_and_order",
        checks,
    )
    target_mapping = {
        row["canonical_value_column"]: row for row in mapping["target_fields"]
    }
    _check(
        all(
            target_mapping[target]["mapping_status"] == "mapped"
            for target in expected_direct
        )
        and target_mapping["activity.activity_volume_norm"]["mapping_status"]
        == "fold_derived"
        and all(
            target_mapping[target]["mapping_status"] == "unsupported"
            for target in unsupported_targets - {"activity.activity_volume_norm"}
        ),
        "mapping_status_contract",
        checks,
    )
    expected_formulas = {
        "sleep.sleep_duration_norm": "sleep_asleep_mean_recent / 1440",
        "sleep.time_in_bed_norm": "sleep_in_bed_mean_recent / 1440",
        "sleep.sleep_efficiency": "sleep_ratio_asleep_in_bed_mean_recent",
        "sleep.sleep_onset_sin": ("sin(2*pi*((adjusted_hour mod 24)*60)/1440)"),
        "sleep.sleep_onset_cos": ("cos(2*pi*((adjusted_hour mod 24)*60)/1440)"),
        "activity.activity_volume_norm": (
            "right_continuous_ecdf(count(train <= x) / n)"
        ),
    }
    _check(
        all(
            target_mapping[target]["formula"] == formula
            for target, formula in expected_formulas.items()
        ),
        "mapping_formula_contract",
        checks,
    )
    _check(
        not set(LABEL_FIELDS).intersection(source_fields),
        "labels_excluded_from_source_features",
        checks,
    )
    complete_source = source.iloc[source_positions]
    _check(
        quality["row_filter"]["input_rows"] == len(source)
        and quality["row_filter"]["complete_four_label_rows"] == len(canonical)
        and quality["row_filter"]["excluded_incomplete_label_rows"]
        == len(source) - len(canonical),
        "quality_row_filter_counts",
        checks,
    )
    _check(
        all(
            quality["source_field_missingness"][name]["full_source_population"][
                "missing_count"
            ]
            == int(source[name].isna().sum())
            and quality["source_field_missingness"][name]["complete_label_population"][
                "missing_count"
            ]
            == int(complete_source[name].isna().sum())
            for name in source_fields
        ),
        "quality_source_missingness_both_populations",
        checks,
    )
    _check(
        all(
            quality["label_summary"]["full_source_missingness"][name]["missing_count"]
            == int(source[name].isna().sum())
            for name in LABEL_FIELDS
        ),
        "quality_label_missingness_full_population",
        checks,
    )
    _check(
        metadata["canonical_column_order"] == parquet_schema.names
        and metadata["canonical_frame_sha256"]
        == _canonical_frame_sha256(canonical, schema, source_fields)
        and metadata["artifact_sha256_before_metadata"]
        == {
            relative: expected["sha256"]
            for relative, expected in artifacts.items()
            if relative
            in {
                "psyche_d/canonical_psyche_d.parquet",
                "psyche_d/data_quality_report.json",
                "mappings/psyche_d_v3_3_3_mapping.json",
            }
        },
        "metadata_columns_and_artifact_hashes",
        checks,
    )
    report_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            destination / "psyche_d" / "data_quality_report.json",
            destination / "psyche_d" / "adapter_metadata.json",
            destination / "psyche_d" / "artifact_manifest.json",
            destination / "mappings" / "psyche_d_v3_3_3_mapping.json",
        )
    )
    _check(
        "psyche_d::" not in report_text, "reports_exclude_participant_values", checks
    )

    return {
        "validation_version": "psyche-d-independent-validation-v1",
        "dataset_id": DATASET_ID,
        "status": "pass",
        "checks": checks,
        "check_count": len(checks),
        "canonical": {
            "row_count": len(canonical),
            "participant_count": int(canonical["participant_id"].nunique()),
            "positive_count": int(canonical["binary_target"].sum()),
            "sha256": _sha256_file(canonical_path),
        },
        "artifact_manifest_sha256": _sha256_file(artifact_manifest_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = validate(
        workspace_root=args.workspace_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
