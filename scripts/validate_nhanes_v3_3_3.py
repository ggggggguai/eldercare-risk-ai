"""Independently validate DATA-004 NHANES V3.3.3 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
SOURCE_RELATIVE = Path("数据集/心理/NHANES")
OUTPUT_RELATIVE = Path("data/processed/mental_health/mood_social/v3.3.3")
PAM_FILE_NAME = "NHANES Preliminary Day Level Output.csv"
MANIFEST_SHA256 = "59257b21fee44bd405e0be6f299d49ad945c6e6289002717f8769389dce32bd5"
COLLECTION_SHA256 = "893a069a74ef8e515c49ef95b96182e03f611ae65abac66fd93cfc375ec17c1e"
SCHEMA_SHA256 = "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
DPQ_FIELDS = tuple(f"DPQ0{index}0" for index in range(1, 10))
CYCLES = ("G", "H")
SURVEY_CYCLES = {"G": "2011-2012", "H": "2013-2014"}
TIMESCALE = "post_assessment_association"
WINDOW_SEMANTICS = (
    "excluded_equals_0; cycle_plus_seqn_plus_window_number; "
    "window_number_ascending; first_7; calendar_date_not_deduplicated"
)
X_SOURCE_NAME = "M10VALUE_mean_valid_pam_days"
EXPECTED_FILES = {
    "nhanes/canonical_nhanes.parquet",
    "nhanes/data_quality_report.json",
    "nhanes/adapter_metadata.json",
    "mappings/nhanes_v3_3_3_mapping.json",
}
EXPECTED_ROWS = 2_775
EXPECTED_POSITIVES = 254
EXPECTED_CYCLES = {"G": 1_373, "H": 1_402}
EXPECTED_SEVERITY = {0: 2_079, 1: 442, 2: 165, 3: 65, 4: 24}
EXPECTED_VALID_DAYS = {1: 61, 2: 115, 3: 165, 4: 244, 5: 427, 6: 672, 7: 1_091}
MAPPED_TARGETS = {
    "activity.relative_amplitude",
    "activity.activity_variability",
    "activity.valid_days",
    "activity.feature_coverage",
    "sleep.sleep_duration_norm",
    "sleep.time_in_bed_norm",
    "sleep.sleep_efficiency",
    "sleep.sleep_onset_sin",
    "sleep.sleep_onset_cos",
    "sleep.wake_time_sin",
    "sleep.wake_time_cos",
    "sleep.sleep_midpoint_sin",
    "sleep.sleep_midpoint_cos",
    "sleep.sleep_fragmentation",
    "sleep.sleep_regularity",
    "sleep.valid_nights",
    "sleep.feature_coverage",
    "social_context.age_group",
    "social_context.sex",
    "social_context.marital_status",
    "social_context.education_level",
}
ACTIVITY_VOLUME = "activity.activity_volume_norm"


class ValidationError(RuntimeError):
    """Raised when independent reconstruction differs from an artifact."""


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
        raise ValidationError("JSON artifact is not readable") from exc


def _check(condition: bool, name: str, checks: list[dict[str, Any]]) -> None:
    if not condition:
        raise ValidationError(f"independent validation failed: {name}")
    checks.append({"check": name, "passed": True})


def _target_specs(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"group": group, **dict(spec)}
        for group in schema["group_order"]
        for spec in schema["groups"][group]
    ]


def _feature_columns(schema: Mapping[str, Any]) -> list[str]:
    return [f"{row['group']}.{row['name']}" for row in _target_specs(schema)]


def _expected_columns(schema: Mapping[str, Any]) -> list[str]:
    identity = [
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
    ]
    targets = [*DPQ_FIELDS, "phq9_total", "phq9_severity", "binary_target"]
    x_source = ["x_source_name", "x_source_value", "x_source_mask"]
    features = _feature_columns(schema)
    return [
        *identity,
        *targets,
        *x_source,
        *features,
        *(f"feature_mask.{x}" for x in features),
    ]


def _expected_arrow_schema(schema: Mapping[str, Any]) -> pa.Schema:
    types: dict[str, pa.DataType] = {
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
        **{
            field: pa.int8()
            for field in (*DPQ_FIELDS, "phq9_total", "phq9_severity", "binary_target")
        },
        "x_source_name": pa.large_string(),
        "x_source_value": pa.float64(),
        "x_source_mask": pa.int8(),
    }
    value_types = {
        "float": pa.float64(),
        "integer": pa.int64(),
        "category": pa.large_string(),
    }
    for spec in _target_specs(schema):
        name = f"{spec['group']}.{spec['name']}"
        types[name] = value_types[str(spec["value_type"])]
        types[f"feature_mask.{name}"] = pa.int8()
    return pa.schema(
        [pa.field(name, types[name]) for name in _expected_columns(schema)]
    )


def _normalised_arrow_table(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    arrays = [
        pa.array(
            frame[field.name].tolist(), type=field.type, from_pandas=True, safe=True
        )
        for field in schema
    ]
    return pa.Table.from_arrays(arrays, schema=schema).combine_chunks()


def _frame_sha256(frame: pd.DataFrame, schema: pa.Schema) -> str:
    table = _normalised_arrow_table(frame, schema)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table, max_chunksize=max(len(table), 1))
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _integer(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64").copy()
    values[np.isfinite(values) & (np.abs(values) < 1e-12)] = 0.0
    if (values[np.isfinite(values)] != np.floor(values[np.isfinite(values)])).any():
        raise ValidationError("source integer field contains a non-integer")
    return pd.Series(pd.array(values, dtype="Int64"), index=series.index)


def _dpq_items(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["SEQN"]].copy()
    result["SEQN"] = _integer(result["SEQN"])
    for field in DPQ_FIELDS:
        values = (
            pd.to_numeric(frame[field], errors="coerce")
            .to_numpy(dtype="float64")
            .copy()
        )
        values[np.isfinite(values) & (np.abs(values) < 1e-12)] = 0.0
        valid = np.isin(values, (0.0, 1.0, 2.0, 3.0))
        values[~valid] = np.nan
        result[field] = pd.array(values, dtype="Int8")
    return result


def _finite(series: pd.Series) -> list[float]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    return [float(value) for value in values if math.isfinite(float(value))]


def _components(hours: Sequence[float]) -> tuple[float | None, float | None]:
    if not hours:
        return None, None
    angles = [2.0 * math.pi * (value % 24.0) / 24.0 for value in hours]
    sine = fmean(math.sin(value) for value in angles)
    cosine = fmean(math.cos(value) for value in angles)
    length = math.hypot(sine, cosine)
    return (None, None) if length <= 1e-12 else (sine / length, cosine / length)


def _regularity(hours: Sequence[float]) -> float | None:
    if len(hours) < 2:
        return None
    angles = [2.0 * math.pi * (value % 24.0) / 24.0 for value in hours]
    return math.hypot(
        fmean(math.sin(value) for value in angles),
        fmean(math.cos(value) for value in angles),
    )


def _severity(total: int) -> int:
    return (
        0
        if total <= 4
        else 1
        if total <= 9
        else 2
        if total <= 14
        else 3
        if total <= 19
        else 4
    )


def _code(value: Any) -> int | None:
    if value is None or bool(pd.isna(value)):
        return None
    numeric = float(value)
    if abs(numeric) < 1e-12:
        numeric = 0.0
    if not math.isfinite(numeric) or numeric != math.floor(numeric):
        raise ValidationError("demographic code is invalid")
    return int(numeric)


def _expected_record(
    cycle: str,
    participant: Any,
    windows: pd.DataFrame,
) -> dict[str, Any]:
    seqn = int(participant.SEQN)
    record: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "global_participant_id": f"nhanes::{cycle}:{seqn}",
        "participant_id": f"{cycle}:{seqn}",
        "cycle": cycle,
        "survey_cycle": SURVEY_CYCLES[cycle],
        "seqn": seqn,
        "timescale_semantics": TIMESCALE,
        "window_semantics": WINDOW_SEMANTICS,
        "window_number_values_json": json.dumps(
            [int(value) for value in windows["window_number"]], separators=(",", ":")
        ),
        "pam_valid_day_count": len(windows),
        "x_source_name": X_SOURCE_NAME,
    }
    total = 0
    for field in DPQ_FIELDS:
        value = int(getattr(participant, field))
        record[field] = value
        total += value
    record["phq9_total"] = total
    record["phq9_severity"] = _severity(total)
    record["binary_target"] = int(total >= 10)

    m10 = _finite(windows["M10VALUE"])
    record["x_source_value"] = fmean(m10) if m10 else None
    record["x_source_mask"] = int(bool(m10))
    l5_pair = windows.loc[windows[["M10VALUE", "L5VALUE"]].notna().all(axis=1)]
    if l5_pair.empty:
        record["activity.relative_amplitude"] = None
    else:
        m10_mean = fmean(_finite(l5_pair["M10VALUE"]))
        l5_mean = fmean(_finite(l5_pair["L5VALUE"]))
        record["activity.relative_amplitude"] = (
            (m10_mean - l5_mean) / (m10_mean + l5_mean)
            if m10_mean + l5_mean > 1e-12
            else None
        )
    record["activity.activity_variability"] = (
        pstdev(m10) / abs(fmean(m10))
        if len(m10) >= 2 and abs(fmean(m10)) > 1e-12
        else None
    )
    activity_days = int(windows[["M10VALUE", "L5VALUE"]].notna().any(axis=1).sum())
    record["activity.valid_days"] = activity_days or None
    activity_available = sum(
        record[name] is not None
        for name in ("activity.relative_amplitude", "activity.activity_variability")
    )
    record["activity.feature_coverage"] = (
        (activity_days / 7.0) * (activity_available / 7.0) if activity_days else None
    )

    sleep_fields = [
        "dur_spt_sleep_min",
        "dur_spt_min",
        "sleep_efficiency",
        "sleeponset",
        "wakeup",
        "dur_spt_wake_IN_min",
    ]
    sleep = windows.loc[windows[sleep_fields].notna().any(axis=1)]
    duration = _finite(sleep["dur_spt_sleep_min"])
    period = _finite(sleep["dur_spt_min"])
    record["sleep.sleep_duration_norm"] = fmean(duration) / 1440.0 if duration else None
    record["sleep.time_in_bed_norm"] = fmean(period) / 1440.0 if period else None
    pooled = sleep.loc[
        sleep[["dur_spt_sleep_min", "dur_spt_min"]].notna().all(axis=1)
        & sleep["dur_spt_min"].gt(0)
    ]
    record["sleep.sleep_efficiency"] = (
        float(pooled["dur_spt_sleep_min"].sum()) / float(pooled["dur_spt_min"].sum())
        if not pooled.empty
        else None
    )
    onset = _finite(sleep["sleeponset"])
    wake = _finite(sleep["wakeup"])
    paired = sleep.loc[sleep[["sleeponset", "wakeup"]].notna().all(axis=1)]
    midpoint = [
        (
            float(row.sleeponset)
            + ((float(row.wakeup) - float(row.sleeponset)) % 24.0) / 2.0
        )
        % 24.0
        for row in paired.itertuples(index=False)
    ]
    record["sleep.sleep_onset_sin"], record["sleep.sleep_onset_cos"] = _components(
        onset
    )
    record["sleep.wake_time_sin"], record["sleep.wake_time_cos"] = _components(wake)
    record["sleep.sleep_midpoint_sin"], record["sleep.sleep_midpoint_cos"] = (
        _components(midpoint)
    )
    fragment_rows = sleep.loc[
        sleep[["dur_spt_wake_IN_min", "dur_spt_min"]].notna().all(axis=1)
        & sleep["dur_spt_min"].gt(0)
    ]
    fragments = (
        fragment_rows["dur_spt_wake_IN_min"] / fragment_rows["dur_spt_min"]
    ).tolist()
    record["sleep.sleep_fragmentation"] = (
        fmean(float(value) for value in fragments) if fragments else None
    )
    regularities = [
        value
        for value in (_regularity(onset), _regularity(wake), _regularity(midpoint))
        if value is not None
    ]
    record["sleep.sleep_regularity"] = fmean(regularities) if regularities else None
    valid_nights = len(sleep)
    record["sleep.valid_nights"] = valid_nights or None
    cells = (
        int(sleep["dur_spt_sleep_min"].notna().sum()),
        int(sleep["dur_spt_min"].notna().sum()),
        int(sleep["sleep_efficiency"].notna().sum()),
        len(onset),
        len(wake),
        len(midpoint),
        len(fragments),
        valid_nights if regularities else 0,
        0,
    )
    record["sleep.feature_coverage"] = sum(cells) / 63.0 if valid_nights else None

    age = _code(participant.RIDAGEYR)
    record["social_context.age_group"] = (
        "60_69"
        if age is not None and age < 70
        else "70_79"
        if age is not None and age < 80
        else "80_plus"
    )
    record["social_context.sex"] = {1: "male", 2: "female"}.get(
        _code(participant.RIAGENDR)
    )
    marital = _code(participant.DMDMARTL)
    record["social_context.marital_status"] = (
        "partnered"
        if marital in {1, 6}
        else "not_partnered"
        if marital in {2, 3, 4, 5}
        else None
    )
    education = _code(participant.DMDEDUC2)
    record["social_context.education_level"] = (
        "primary_or_less"
        if education == 1
        else "middle"
        if education in {2, 3}
        else "high_or_above"
        if education in {4, 5}
        else None
    )
    return record


def _reconstruct(source_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pam = pd.read_csv(source_root / PAM_FILE_NAME)
    demographics: dict[str, pd.DataFrame] = {}
    dpq: dict[str, pd.DataFrame] = {}
    members: dict[int, str] = {}
    for cycle in CYCLES:
        demo = pd.read_sas(source_root / f"DEMO_{cycle}.xpt", format="xport")
        depression = pd.read_sas(source_root / f"DPQ_{cycle}.xpt", format="xport")
        demo["SEQN"] = _integer(demo["SEQN"])
        depression = _dpq_items(depression)
        demographics[cycle] = demo
        dpq[cycle] = depression
        for value in demo["SEQN"].dropna().astype(int):
            if value in members:
                raise ValidationError("SEQN has ambiguous cycle membership")
            members[value] = cycle

    pam = pam.copy()
    pam["SEQN"] = _integer(pam["SEQN"])
    pam["window_number"] = _integer(pam["window_number"])
    pam["excluded"] = _integer(pam["excluded"])
    pam["cycle"] = pam["SEQN"].astype(int).map(members).astype("string")
    if pam["cycle"].isna().any():
        raise ValidationError("PAM has an unassigned cycle")
    for field in (
        "M10VALUE",
        "L5VALUE",
        "dur_spt_sleep_min",
        "dur_spt_min",
        "sleep_efficiency",
        "sleeponset",
        "wakeup",
        "dur_spt_wake_IN_min",
    ):
        pam[field] = pd.to_numeric(pam[field], errors="coerce")
    if pam.duplicated(["cycle", "SEQN", "window_number"]).any():
        raise ValidationError("PAM window key is duplicated")
    same_date = pam.groupby(["cycle", "SEQN", "calendar_date"], sort=False).size()
    selected = pam.loc[pam["excluded"].eq(0)].sort_values(
        ["cycle", "SEQN", "window_number"], kind="stable"
    )
    selected = selected.groupby(["cycle", "SEQN"], sort=False, group_keys=False).head(7)
    groups = {
        (str(cycle), int(seqn)): group
        for (cycle, seqn), group in selected.groupby(["cycle", "SEQN"], sort=False)
    }
    all_pam_keys = set(
        zip(pam["cycle"].astype(str), pam["SEQN"].astype(int), strict=True)
    )

    records: list[dict[str, Any]] = []
    flow: dict[str, dict[str, int]] = {}
    for cycle in CYCLES:
        demo = (
            demographics[cycle]
            .loc[:, ["SEQN", "RIDAGEYR", "RIAGENDR", "DMDMARTL", "DMDEDUC2"]]
            .copy()
        )
        demo["RIDAGEYR"] = _integer(demo["RIDAGEYR"])
        demo = demo.loc[demo["RIDAGEYR"].ge(60).fillna(False)]
        complete = dpq[cycle].loc[dpq[cycle][list(DPQ_FIELDS)].notna().all(axis=1)]
        eligible = demo.merge(complete, on="SEQN", how="inner", validate="one_to_one")
        any_pam = eligible.loc[
            eligible["SEQN"].map(lambda value: (cycle, int(value)) in all_pam_keys)
        ]
        final = any_pam.loc[
            any_pam["SEQN"].map(lambda value: (cycle, int(value)) in groups)
        ]
        flow[cycle] = {"pre_quality": len(any_pam), "final": len(final)}
        records.extend(
            _expected_record(
                cycle,
                row,
                groups[(cycle, int(row.SEQN))].sort_values(
                    "window_number", kind="stable"
                ),
            )
            for row in final.itertuples(index=False)
        )
    records.sort(key=lambda row: (str(row["cycle"]), int(row["seqn"])))
    audit = {
        "source_rows": len(pam),
        "same_date_groups": int((same_date > 1).sum()),
        "same_date_records": int(same_date[same_date > 1].sum()),
        "pre_quality": sum(row["pre_quality"] for row in flow.values()),
        "flow": flow,
    }
    return records, audit


def _equal(actual: Any, expected: Any) -> bool:
    if expected is None:
        return bool(pd.isna(actual))
    if isinstance(expected, float):
        return not bool(pd.isna(actual)) and math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-12
        )
    return actual == expected


def validate(workspace_root: Path, output_root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    algorithm_root = workspace_root / "algorithm" / "eldercare-risk-ai-main"
    manifests_root = algorithm_root / OUTPUT_RELATIVE / "manifests"
    manifest_path = manifests_root / "dataset_file_manifest.jsonl"
    schema_path = manifests_root / "feature_schema_manifest.json"
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
    _check(_sha256_file(schema_path) == SCHEMA_SHA256, "data001_schema_sha256", checks)
    _check(_load_json(schema_path) == schema, "data001_schema_equals_live", checks)

    manifest_rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected_rows = [
        row for row in manifest_rows if row.get("dataset_id") == DATASET_ID
    ]
    _check(len(selected_rows) == 22, "source_file_count", checks)
    source_root = workspace_root / SOURCE_RELATIVE
    actual_paths = {
        (SOURCE_RELATIVE / path.relative_to(source_root)).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    _check(
        actual_paths == {str(row["relative_path"]) for row in selected_rows},
        "source_file_set",
        checks,
    )
    current_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        path = workspace_root / Path(str(row["relative_path"]))
        digest = _sha256_file(path)
        _check(path.stat().st_size == int(row["bytes"]), "source_size", checks)
        _check(digest == str(row["sha256"]), "source_file_sha256", checks)
        current_rows.append(
            {
                "relative_path": row["relative_path"],
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    _check(
        _collection_sha256(current_rows) == COLLECTION_SHA256,
        "source_collection_sha256",
        checks,
    )

    artifact_manifest_path = output_root / "nhanes" / "artifact_manifest.json"
    artifact_manifest = _load_json(artifact_manifest_path)
    _check(
        artifact_manifest.get("complete") is True, "artifact_manifest_complete", checks
    )
    _check(
        set(artifact_manifest.get("artifacts", {})) == EXPECTED_FILES,
        "artifact_manifest_file_set",
        checks,
    )
    artifact_hashes: dict[str, str] = {}
    for relative in sorted(EXPECTED_FILES):
        path = output_root / Path(relative)
        digest = _sha256_file(path)
        row = artifact_manifest["artifacts"][relative]
        _check(path.stat().st_size == int(row["bytes"]), "artifact_size", checks)
        _check(digest == str(row["sha256"]), "artifact_sha256", checks)
        artifact_hashes[relative] = digest
    artifact_hashes["nhanes/artifact_manifest.json"] = _sha256_file(
        artifact_manifest_path
    )
    _check(
        {path.name for path in (output_root / "nhanes").iterdir() if path.is_file()}
        == {
            "canonical_nhanes.parquet",
            "data_quality_report.json",
            "adapter_metadata.json",
            "artifact_manifest.json",
        },
        "no_formal_ecdf_or_extra_nhanes_artifact",
        checks,
    )

    canonical_path = output_root / "nhanes" / "canonical_nhanes.parquet"
    expected_schema = _expected_arrow_schema(schema)
    _check(
        pq.read_schema(canonical_path).equals(expected_schema),
        "canonical_arrow_schema",
        checks,
    )
    canonical = pq.read_table(canonical_path).to_pandas()
    _check(
        list(canonical.columns) == _expected_columns(schema),
        "canonical_column_order",
        checks,
    )
    _check(len(canonical) == EXPECTED_ROWS, "canonical_row_count", checks)
    _check(
        canonical["global_participant_id"].nunique() == EXPECTED_ROWS,
        "canonical_unique_participants",
        checks,
    )
    _check(
        not canonical.duplicated(["cycle", "seqn"]).any(),
        "canonical_unique_cycle_seqn",
        checks,
    )

    expected_records, raw_audit = _reconstruct(source_root)
    _check(
        len(expected_records) == len(canonical),
        "independent_reconstruction_row_count",
        checks,
    )
    keys = list(
        zip(canonical["cycle"].astype(str), canonical["seqn"].astype(int), strict=True)
    )
    expected_keys = [(str(row["cycle"]), int(row["seqn"])) for row in expected_records]
    _check(keys == expected_keys, "independent_cycle_seqn_sort_and_set", checks)
    for column in [
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
        *DPQ_FIELDS,
        "phq9_total",
        "phq9_severity",
        "binary_target",
        "x_source_name",
        "x_source_value",
        "x_source_mask",
        *sorted(MAPPED_TARGETS),
    ]:
        _check(
            all(
                _equal(actual, expected[column])
                for actual, expected in zip(
                    canonical[column], expected_records, strict=True
                )
            ),
            f"independent_values::{column}",
            checks,
        )

    feature_columns = _feature_columns(schema)
    unsupported = set(feature_columns) - MAPPED_TARGETS - {ACTIVITY_VOLUME}
    _check(canonical[ACTIVITY_VOLUME].isna().all(), "base_activity_volume_null", checks)
    _check(
        canonical[f"feature_mask.{ACTIVITY_VOLUME}"].eq(0).all(),
        "base_activity_volume_mask_0",
        checks,
    )
    for target in sorted(MAPPED_TARGETS):
        _check(
            canonical[f"feature_mask.{target}"]
            .astype(int)
            .equals(canonical[target].notna().astype(int)),
            f"mapped_mask::{target}",
            checks,
        )
    for target in sorted(unsupported):
        _check(canonical[target].isna().all(), f"unsupported_null::{target}", checks)
        _check(
            canonical[f"feature_mask.{target}"].eq(0).all(),
            f"unsupported_mask_0::{target}",
            checks,
        )

    _check(
        int(canonical["binary_target"].sum()) == EXPECTED_POSITIVES,
        "positive_count",
        checks,
    )
    cycle_counts = {
        str(k): int(v)
        for k, v in canonical["cycle"].value_counts().sort_index().items()
    }
    severity_counts = {
        int(k): int(v)
        for k, v in canonical["phq9_severity"].value_counts().sort_index().items()
    }
    valid_day_counts = {
        int(k): int(v)
        for k, v in canonical["pam_valid_day_count"].value_counts().sort_index().items()
    }
    _check(cycle_counts == EXPECTED_CYCLES, "cycle_counts", checks)
    _check(severity_counts == EXPECTED_SEVERITY, "severity_counts", checks)
    _check(valid_day_counts == EXPECTED_VALID_DAYS, "valid_day_counts", checks)
    _check(
        raw_audit["pre_quality"] == 2_827,
        "reference_count_reconstructed_not_targeted",
        checks,
    )
    _check(
        raw_audit["same_date_groups"] == 702
        and raw_audit["same_date_records"] == 1_404,
        "same_date_multiwindow_audit",
        checks,
    )

    mapping = _load_json(output_root / "mappings" / "nhanes_v3_3_3_mapping.json")
    target_rows = mapping["target_fields"]
    _check(len(target_rows) == 54, "mapping_target_count", checks)
    _check(
        [row["canonical_value_column"] for row in target_rows] == feature_columns,
        "mapping_target_order",
        checks,
    )
    mapped = {
        row["canonical_value_column"]
        for row in target_rows
        if row["mapping_status"] == "mapped"
    }
    folded = {
        row["canonical_value_column"]
        for row in target_rows
        if row["mapping_status"] == "fold_derived"
    }
    _check(mapped == MAPPED_TARGETS, "mapping_strict_supported_set", checks)
    _check(folded == {ACTIVITY_VOLUME}, "mapping_fold_only_activity_volume", checks)
    _check(
        mapping["labels"]["excluded_from_total"] == "DPQ100",
        "mapping_dpq100_excluded",
        checks,
    )
    _check(
        mapping["labels"]["input_feature_use"] == "prohibited",
        "mapping_labels_not_input",
        checks,
    )
    _check(
        mapping["x_source"]["only_canonical_storage_of_participant_mean"] is True,
        "mapping_m10_mean_x_source_only",
        checks,
    )
    _check(
        mapping["ecdf_contract"]["split_id_required"] is True,
        "mapping_ecdf_split_required",
        checks,
    )
    _check(
        mapping["ecdf_contract"]["formal_instance_before_DATA_007"] is False,
        "mapping_no_pre_data007_ecdf",
        checks,
    )
    _check(
        mapping["production_input_policy"]["cycle_is_input"] is False,
        "mapping_cycle_not_input",
        checks,
    )
    _check(
        mapping["production_input_policy"]["survey_weights_are_inputs"] is False,
        "mapping_research_fields_not_input",
        checks,
    )

    quality = _load_json(output_root / "nhanes" / "data_quality_report.json")
    _check(
        quality["row_filter"]["pre_quality_reference_count"]
        == raw_audit["pre_quality"],
        "quality_reference_count",
        checks,
    )
    _check(
        quality["row_filter"]["reference_count_used_as_filter_target"] is False,
        "quality_reference_not_target",
        checks,
    )
    _check(
        quality["pam_window_audit"]["same_calendar_date_multiwindow_groups"]
        == raw_audit["same_date_groups"],
        "quality_same_date_groups",
        checks,
    )
    _check(
        quality["pam_window_audit"]["same_calendar_date_records_silently_deduplicated"]
        == 0,
        "quality_no_silent_date_dedup",
        checks,
    )
    _check(
        quality["dpq"]["DPQ100_in_total"] is False, "quality_dpq100_not_total", checks
    )
    _check(
        quality["dpq"]["dpq_or_label_fields_used_as_input_features"] is False,
        "quality_labels_not_input",
        checks,
    )
    _check(
        quality["x_source"]["base_canonical_ecdf_fitted"] is False,
        "quality_ecdf_not_fitted",
        checks,
    )
    _check(
        quality["participant_summary"]["participant_values_written_to_report"] is False,
        "quality_no_participant_values",
        checks,
    )

    metadata = _load_json(output_root / "nhanes" / "adapter_metadata.json")
    frame_sha256 = _frame_sha256(canonical, expected_schema)
    _check(
        metadata["canonical_frame_sha256"] == frame_sha256,
        "canonical_frame_sha256",
        checks,
    )
    _check(
        metadata["base_canonical_ecdf_fitted"] is False,
        "metadata_ecdf_not_fitted",
        checks,
    )
    _check(
        metadata["formal_ecdf_instance_created"] is False,
        "metadata_no_formal_ecdf",
        checks,
    )
    _check(
        metadata["canonical_row_count"] == EXPECTED_ROWS, "metadata_row_count", checks
    )

    return {
        "artifact_sha256": artifact_hashes,
        "canonical_frame_sha256": frame_sha256,
        "dataset_id": DATASET_ID,
        "participant_count": len(canonical),
        "positive_row_count": int(canonical["binary_target"].sum()),
        "status": "pass",
        "validation_check_count": len(checks),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace = _find_workspace_root(args.workspace_root or Path.cwd())
    algorithm_root = workspace / "algorithm" / "eldercare-risk-ai-main"
    output = (args.output_root or algorithm_root / OUTPUT_RELATIVE).resolve()
    result = validate(workspace, output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
