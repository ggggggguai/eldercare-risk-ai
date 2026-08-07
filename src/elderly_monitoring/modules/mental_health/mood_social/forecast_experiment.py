"""PSYCHE-D nominal-month forecast experiment context.

V3.4 is deliberately separate from the V3.3 state-attention pipeline.  This
module only constructs auditable ``forecast_1m``/``forecast_2m`` samples from
the raw PSYCHE-D matrix.  It does not fit a model, alter the production
package, or expose an HTTP result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from elderly_monitoring.datasets.adapters.psyche_d import (
    LABEL_FIELDS,
    PSYCHE_D_SOURCE_FIELDS,
    parse_sample_index,
)


FORECAST_TASKS: Mapping[str, int] = {"forecast_1m": 1, "forecast_2m": 2}
LABEL_MONTHS = frozenset({3, 6, 9, 12})
TIMING_EVIDENCE = "design_level"
RAW_DATE_AVAILABLE = False
EXPERIMENT_VERSION = "mood-social-forecast-v3.4"
CONTEXT_SCHEMA_VERSION = "mood-social-forecast-context-v3.4-v1"
FORECAST_VERSION = CONTEXT_SCHEMA_VERSION
SPLIT_MANIFEST_VERSION = "mood-social-nested-participant-split-manifest-v1"
GLOBAL_PARTICIPANT_PREFIX = "psyche_d::"
AUDIT_TIME_COLUMNS = ("feature_start_time", "feature_end_time", "assessment_time")


class ForecastExperimentError(ValueError):
    """Raised when the frozen forecast context contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_names(mapping_path: Path) -> tuple[str, ...]:
    """Read the frozen P0 whitelist and return the exact 27-field order."""

    try:
        payload = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ForecastExperimentError("forecast feature mapping is unreadable") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("features"), dict):
        raise ForecastExperimentError("forecast feature mapping has an invalid root")
    fields: list[str] = []
    for group in ("steps", "sleep"):
        rows = payload["features"].get(group)
        if not isinstance(rows, list):
            raise ForecastExperimentError(f"forecast feature group is invalid: {group}")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(
                row.get("source_field"), str
            ):
                raise ForecastExperimentError("forecast feature mapping row is invalid")
            fields.append(row["source_field"])
    expected = tuple(PSYCHE_D_SOURCE_FIELDS)
    if tuple(fields) != expected or len(fields) != 27:
        raise ForecastExperimentError(
            "forecast feature whitelist must contain the frozen 27 fields"
        )
    return expected


def load_split_assignments(split_path: Path) -> dict[str, dict[str, Any]]:
    """Load the V3.3 participant assignment without rebuilding another split."""

    try:
        payload = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ForecastExperimentError("V3.3 split manifest is unreadable") from exc
    if not isinstance(payload, dict):
        raise ForecastExperimentError("V3.3 split manifest root is invalid")
    if payload.get("manifest_version") != SPLIT_MANIFEST_VERSION:
        raise ForecastExperimentError("V3.3 split manifest version is not frozen")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("random_seed") != 20260728:
        raise ForecastExperimentError("V3.3 split manifest random seed is not frozen")
    if protocol.get("global_participant_key_format") != "dataset_id::participant_id":
        raise ForecastExperimentError(
            "V3.3 split manifest participant key format is invalid"
        )
    rows = payload.get("participant_assignments")
    if not isinstance(rows, list) or not rows:
        raise ForecastExperimentError(
            "V3.3 split manifest has no participant assignments"
        )
    assignments: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ForecastExperimentError("V3.3 participant assignment is invalid")
        key = row.get("global_participant_id")
        fold = row.get("outer_fold")
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(fold, int)
            or fold not in range(5)
        ):
            raise ForecastExperimentError(
                "V3.3 participant assignment has an invalid key or fold"
            )
        dataset_id = row.get("dataset_id")
        key_dataset, separator, participant_id = key.partition("::")
        if (
            not isinstance(dataset_id, str)
            or not dataset_id
            or not separator
            or key_dataset != dataset_id
            or not participant_id
        ):
            raise ForecastExperimentError(
                "V3.3 participant assignment key is not canonical"
            )
        if key in assignments:
            raise ForecastExperimentError(
                "V3.3 participant assignment contains a duplicate participant"
            )
        assignments[key] = {
            "dataset_id": dataset_id,
            "global_participant_id": key,
            "outer_fold": fold,
            "inner_validation_fold_by_outer_fold": row.get(
                "inner_validation_fold_by_outer_fold"
            ),
        }
    return assignments


def _finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    return values.mask(~np.isfinite(values), np.nan)


def _validate_source_frame(
    source: pd.DataFrame, feature_names: Sequence[str]
) -> pd.DataFrame:
    required = set(feature_names) | set(LABEL_FIELDS)
    if not required.issubset(source.columns):
        missing = sorted(required.difference(source.columns))
        raise ForecastExperimentError(
            f"PSYCHE-D source is missing required fields: {missing}"
        )
    if not source.index.is_unique:
        raise ForecastExperimentError("PSYCHE-D sample index must be unique")
    parsed = parse_sample_index(source.index)
    keys = pd.MultiIndex.from_arrays(
        [parsed["participant_id"].astype(str), parsed["nominal_month"].astype(int)],
        names=["participant_id", "nominal_month"],
    )
    if keys.has_duplicates:
        raise ForecastExperimentError("(participant_id, nominal_month) must be unique")
    frame = source.copy()
    for name in (*feature_names, *LABEL_FIELDS):
        frame[name] = _finite_numeric(frame[name])
    _validate_label_ranges(frame)
    frame["__participant_id"] = parsed["participant_id"].astype(str).to_numpy()
    frame["__nominal_month"] = parsed["nominal_month"].astype(int).to_numpy()
    return frame


def _validate_label_ranges(frame: pd.DataFrame) -> None:
    """Reject non-integer or out-of-range PHQ-9 values before target selection."""

    ranges = {
        "phq9_score_start": (0, 27),
        "phq9_score_end": (0, 27),
        "phq9_cat_start": (0, 4),
        "phq9_cat_end": (0, 4),
    }
    for name, (lower, upper) in ranges.items():
        values = frame[name].dropna()
        invalid = (values % 1 != 0) | (values < lower) | (values > upper)
        if invalid.any():
            raise ForecastExperimentError(f"{name} contains an invalid PHQ-9 value")


def _target_rows(frame: pd.DataFrame) -> pd.DataFrame:
    complete_labels = frame.loc[:, list(LABEL_FIELDS)].notna().all(axis=1)
    valid_month = frame["__nominal_month"].isin(LABEL_MONTHS)
    if (complete_labels & ~valid_month).any():
        raise ForecastExperimentError(
            "complete PHQ-9 label rows must use nominal months 3, 6, 9 or 12"
        )
    targets = frame.loc[complete_labels & valid_month].copy()
    targets["__target"] = (targets["phq9_score_end"] >= 10).astype("int8")
    return targets


def build_forecast_samples(
    source: pd.DataFrame,
    *,
    task_id: str,
    mapping_path: Path,
    split_path: Path,
    expected_count: int | None = None,
) -> pd.DataFrame:
    """Build one horizon using an exact nominal-month key join.

    The label row contributes only the target and audit metadata.  All model
    fields are copied from the same participant's exact ``t-gap`` row.
    """

    if task_id not in FORECAST_TASKS:
        raise ForecastExperimentError(f"unsupported forecast task: {task_id}")
    feature_names = load_feature_names(mapping_path)
    assignments = load_split_assignments(split_path)
    frame = _validate_source_frame(source, feature_names)
    targets = _target_rows(frame)
    gap = FORECAST_TASKS[task_id]
    lookup = frame.set_index(["__participant_id", "__nominal_month"], drop=False)
    rows: list[dict[str, Any]] = []
    for target_index, target in targets.iterrows():
        participant = str(target["__participant_id"])
        target_month = int(target["__nominal_month"])
        feature_month = target_month - gap
        feature_key = (participant, feature_month)
        if feature_key not in lookup.index:
            continue
        feature = lookup.loc[feature_key]
        if isinstance(feature, pd.DataFrame):
            raise ForecastExperimentError(
                "forecast feature lookup returned duplicate rows"
            )
        values = {name: feature[name] for name in feature_names}
        nonmissing = int(pd.Series(values, dtype="float64").notna().sum())
        if nonmissing == 0:
            raise ForecastExperimentError(
                "forecast feature row has all 27 fields missing"
            )
        global_id = f"psyche_d::{participant}"
        assignment = assignments.get(global_id)
        if assignment is None:
            raise ForecastExperimentError(
                f"participant is absent from V3.3 split: {global_id}"
            )
        if assignment["dataset_id"] != "psyche_d":
            raise ForecastExperimentError(
                f"participant has a non-PSYCHE-D split assignment: {global_id}"
            )
        target_window_id = f"psyche_d::{participant}::m{target_month}"
        feature_window_id = f"psyche_d::{participant}::m{feature_month}"
        rows.append(
            {
                "participant_id": participant,
                "global_participant_id": global_id,
                "target_window_id": target_window_id,
                "feature_window_id": feature_window_id,
                "label_month_slot": target_month,
                "feature_month_slot": feature_month,
                "task_id": task_id,
                "nominal_gap_months": gap,
                "timing_evidence": TIMING_EVIDENCE,
                "natural_dates_available": RAW_DATE_AVAILABLE,
                "feature_start_time": None,
                "feature_end_time": None,
                "assessment_time": None,
                "future_binary_target": int(target["__target"]),
                "feature_nonmissing_count": nonmissing,
                "outer_fold_id": int(assignment["outer_fold"]),
                **values,
            }
        )
    result = pd.DataFrame(
        rows,
        columns=[
            "participant_id",
            "global_participant_id",
            "target_window_id",
            "feature_window_id",
            "label_month_slot",
            "feature_month_slot",
            "task_id",
            "nominal_gap_months",
            "timing_evidence",
            "natural_dates_available",
            *AUDIT_TIME_COLUMNS,
            "future_binary_target",
            "feature_nonmissing_count",
            "outer_fold_id",
            *feature_names,
        ],
    )
    if result.empty:
        raise ForecastExperimentError(f"no eligible rows were built for {task_id}")
    if result["target_window_id"].duplicated().any():
        raise ForecastExperimentError("forecast target window IDs must be unique")
    if (result["feature_month_slot"] != result["label_month_slot"] - gap).any():
        raise ForecastExperimentError("forecast month join relation is invalid")
    if expected_count is not None and len(result) != expected_count:
        raise ForecastExperimentError(
            f"{task_id} row count {len(result)} does not match expected {expected_count}"
        )
    return result


def build_forecast_pair(
    source: pd.DataFrame,
    *,
    mapping_path: Path,
    split_path: Path,
    expected_1m: int | None = None,
    expected_2m: int | None = None,
    expected_common: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build both horizons and return the auditable common target cohort."""

    one = build_forecast_samples(
        source,
        task_id="forecast_1m",
        mapping_path=mapping_path,
        split_path=split_path,
        expected_count=expected_1m,
    )
    two = build_forecast_samples(
        source,
        task_id="forecast_2m",
        mapping_path=mapping_path,
        split_path=split_path,
        expected_count=expected_2m,
    )
    pair_key = ["global_participant_id", "target_window_id"]
    one_keys = one[
        pair_key + ["label_month_slot", "future_binary_target", "outer_fold_id"]
    ]
    two_keys = two[
        pair_key + ["label_month_slot", "future_binary_target", "outer_fold_id"]
    ]
    paired = one_keys.merge(
        two_keys,
        on=pair_key,
        how="inner",
        validate="one_to_one",
        suffixes=("_1m", "_2m"),
    )
    for field in ("label_month_slot", "future_binary_target", "outer_fold_id"):
        if (paired[f"{field}_1m"] != paired[f"{field}_2m"]).any():
            raise ForecastExperimentError(f"common horizon {field} values do not match")
    common = paired[
        [
            *pair_key,
            "label_month_slot_1m",
            "future_binary_target_1m",
            "outer_fold_id_1m",
        ]
    ].rename(
        columns={
            "label_month_slot_1m": "label_month_slot",
            "future_binary_target_1m": "future_binary_target",
            "outer_fold_id_1m": "outer_fold_id",
        }
    )
    if expected_common is not None and len(common) != expected_common:
        raise ForecastExperimentError(
            f"common target count {len(common)} does not match expected {expected_common}"
        )
    if not set(one["global_participant_id"]).issubset(
        set(two["global_participant_id"])
    ):
        # This is not an error: horizons can have different eligible participants.
        pass
    return one, two, common


__all__ = [
    "FORECAST_TASKS",
    "FORECAST_VERSION",
    "ForecastExperimentError",
    "build_forecast_pair",
    "build_forecast_samples",
    "load_feature_names",
    "load_split_assignments",
    "sha256_file",
]
