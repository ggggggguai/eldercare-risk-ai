"""Strict horizon-specific context construction for FORECAST-OPT-002B.

The module is deliberately offline-only.  It builds reproducible feature
tables from the frozen PSYCHE-D nominal-month samples and never fits a model
or changes the V3.3 runtime.  PHQ-9 fields are retained only by the source
adapter for target construction; they are never returned in ``feature_names``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from elderly_monitoring.datasets.adapters.psyche_d import (
    LABEL_FIELDS,
    PSYCHE_D_SOURCE_FIELDS,
    parse_sample_index,
)
from elderly_monitoring.modules.mental_health.mood_social.forecast_experiment import (
    FORECAST_TASKS,
    build_forecast_samples,
    load_feature_names,
    load_split_assignments,
)


HORIZON_FEATURE_VERSION = "mood-social-forecast-v3.4-opt002b-context-v1"
EPSILON = 1e-6
MIN_PERSONAL_BASELINE_COUNT = 2
PROHIBITED_INPUTS = frozenset(
    {
        *LABEL_FIELDS,
        "future_binary_target",
        "label_month_slot",
        "target_month_features",
        "future_month_features",
        "dataset_id",
        "participant_identity",
        "84_undefined_window_fields",
    }
)


class HorizonOptimizationError(ValueError):
    """Raised when the 002B context contract is violated."""


@dataclass(frozen=True)
class HorizonContext:
    """A context table and its auditable feature groups."""

    task_id: str
    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    feature_groups: Mapping[str, tuple[str, ...]]
    audit: Mapping[str, Any]


def _numeric(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _safe_median(values: Sequence[float]) -> float:
    values_array = np.asarray(values, dtype="float64")
    values_array = values_array[np.isfinite(values_array)]
    return float(np.median(values_array)) if values_array.size else float("nan")


def _safe_mad(values: Sequence[float], center: float | None = None) -> float:
    values_array = np.asarray(values, dtype="float64")
    values_array = values_array[np.isfinite(values_array)]
    if not values_array.size:
        return float("nan")
    location = float(np.median(values_array) if center is None else center)
    return float(np.median(np.abs(values_array - location)))


def _slope(points: Sequence[tuple[int, float]]) -> float:
    if len(points) < 2:
        return float("nan")
    x = np.asarray([point[0] for point in points], dtype="float64")
    y = np.asarray([point[1] for point in points], dtype="float64")
    centered = x - x.mean()
    denominator = float(np.sum(centered * centered))
    if denominator <= EPSILON:
        return float("nan")
    return float(np.sum(centered * (y - y.mean())) / denominator)


def _direction(delta: float) -> float:
    if not np.isfinite(delta):
        return float("nan")
    if abs(delta) <= EPSILON:
        return 0.0
    return float(np.sign(delta))


def _consecutive_direction(
    values_by_month: Mapping[int, float], cutoff: int, direction: float
) -> float:
    if not np.isfinite(direction):
        return float("nan")
    if direction == 0:
        return 1.0
    count = 1
    current = cutoff
    while True:
        newer = _numeric(values_by_month.get(current))
        older = _numeric(values_by_month.get(current - 1))
        if not (np.isfinite(newer) and np.isfinite(older)):
            break
        if _direction(newer - older) != direction:
            break
        count += 1
        current -= 1
    return float(count)


def _change_point_features(
    values_by_month: Mapping[int, float], cutoff: int
) -> tuple[float, float]:
    points = [
        (month, _numeric(value))
        for month, value in values_by_month.items()
        if month <= cutoff and np.isfinite(_numeric(value))
    ]
    points.sort()
    if len(points) < 3:
        return float("nan"), float("nan")
    deltas = np.diff(np.asarray([value for _, value in points], dtype="float64"))
    delta_center = float(np.median(deltas))
    delta_scale = max(1.4826 * _safe_mad(deltas, delta_center), EPSILON)
    candidates = [
        index
        for index, delta in enumerate(deltas)
        if abs(float(delta) - delta_center) > 2.5 * delta_scale
    ]
    if not candidates:
        return float("nan"), float("nan")
    index = candidates[-1]
    change_month = points[index + 1][0]
    before = np.asarray([value for _, value in points[: index + 1]], dtype="float64")
    after = np.asarray([value for _, value in points[index + 1 :]], dtype="float64")
    if not before.size or not after.size:
        return float("nan"), float("nan")
    return float(cutoff - change_month), float(after.mean() - before.mean())


def _consecutive_missing(
    values_by_month: Mapping[int, float], cutoff: int
) -> tuple[int, float]:
    observed_months = [int(month) for month in values_by_month if month <= cutoff]
    if not observed_months:
        return 0, float("nan")
    lower_bound = min(observed_months)
    count = 0
    for month in range(cutoff, lower_bound - 1, -1):
        value = _numeric(values_by_month.get(month))
        if np.isfinite(value):
            break
        count += 1
    valid = [
        month
        for month, value in values_by_month.items()
        if month <= cutoff and np.isfinite(_numeric(value))
    ]
    if not valid:
        return count, float("nan")
    return count, float(cutoff - max(valid))


def _personal_stats(
    values_by_month: Mapping[int, float], cutoff: int
) -> tuple[float, float, float, int, int]:
    prior = [
        _numeric(value)
        for month, value in values_by_month.items()
        if month < cutoff and np.isfinite(_numeric(value))
    ]
    median = _safe_median(prior)
    mad = _safe_mad(prior, median)
    available = int(len(prior) >= MIN_PERSONAL_BASELINE_COUNT)
    anchor = _numeric(values_by_month.get(cutoff))
    if not available or not np.isfinite(anchor):
        z = float("nan")
    else:
        z = (anchor - median) / max(1.4826 * mad, EPSILON)
    return median, mad, z, len(prior), available


def _field_features(
    values_by_month: Mapping[int, float], cutoff: int, *, long_term: bool
) -> dict[str, float | int]:
    anchor = _numeric(values_by_month.get(cutoff))
    previous = _numeric(values_by_month.get(cutoff - 1))
    previous_two = _numeric(values_by_month.get(cutoff - 2))
    local_points = [
        (month, _numeric(values_by_month.get(month)))
        for month in (cutoff - 2, cutoff - 1, cutoff)
        if np.isfinite(_numeric(values_by_month.get(month)))
    ]
    local_delta_1 = (
        anchor - previous
        if np.isfinite(anchor) and np.isfinite(previous)
        else float("nan")
    )
    local_delta_2 = (
        anchor - previous_two
        if np.isfinite(anchor) and np.isfinite(previous_two)
        else float("nan")
    )
    slope = _slope(local_points)
    direction = _direction(local_delta_1)
    missing_count, months_since = _consecutive_missing(values_by_month, cutoff)
    valid_history = [
        _numeric(value)
        for month, value in values_by_month.items()
        if month <= cutoff and np.isfinite(_numeric(value))
    ]
    median, mad, personal_z, prior_count, baseline_available = _personal_stats(
        values_by_month, cutoff
    )
    change_distance, change_delta = _change_point_features(values_by_month, cutoff)
    output: dict[str, float | int] = {
        "anchor": anchor,
        "anchor_missing": int(not np.isfinite(anchor)),
        "delta_1": local_delta_1,
        "delta_1_missing": int(not np.isfinite(local_delta_1)),
        "delta_2": local_delta_2,
        "delta_2_missing": int(not np.isfinite(local_delta_2)),
        "local_slope": slope,
        "local_slope_missing": int(not np.isfinite(slope)),
        "direction": direction,
        "direction_missing": int(not np.isfinite(direction)),
        "direction_persistence": _consecutive_direction(
            values_by_month, cutoff, direction
        ),
        "change_point_distance": change_distance,
        "change_point_distance_missing": int(not np.isfinite(change_distance)),
        "change_point_mean_delta": change_delta,
        "change_point_mean_delta_missing": int(not np.isfinite(change_delta)),
        "history_count": len(valid_history),
        "history_count_prior": prior_count,
        "consecutive_missing_count": missing_count,
        "months_since_last_valid": months_since,
        "months_since_last_valid_missing": int(not np.isfinite(months_since)),
        "personal_median": median,
        "personal_mad": mad,
        "personal_z": personal_z,
        "personal_z_missing": int(not np.isfinite(personal_z)),
        "baseline_available": baseline_available,
        "cold_start": int(not baseline_available),
    }
    if long_term:
        values = np.asarray(valid_history, dtype="float64")
        if values.size:
            q10, q90 = np.quantile(values, [0.1, 0.9])
            std = float(np.std(values, ddof=0))
            mean = float(np.mean(values))
            median_long = float(np.median(values))
            mad_long = _safe_mad(values, median_long)
            scale = max(1.4826 * mad_long, EPSILON)
            z_values = (values - median_long) / scale
            anomaly_burden = float(np.mean(np.abs(z_values) >= 2.0))
            persistence = float(np.sum(np.abs(z_values) >= 1.0) / values.size)
            recovery = (
                float(values[-1] - values[-2]) if values.size >= 2 else float("nan")
            )
        else:
            q10 = q90 = std = mean = median_long = mad_long = float("nan")
            anomaly_burden = persistence = recovery = float("nan")
        all_points = [
            (month, _numeric(value))
            for month, value in values_by_month.items()
            if month <= cutoff and np.isfinite(_numeric(value))
        ]
        all_points.sort()
        long_slope = _slope(all_points)
        output.update(
            {
                "historical_mean": mean,
                "historical_median": median_long,
                "historical_mad": mad_long,
                "historical_std": std,
                "historical_q10": float(q10),
                "historical_q90": float(q90),
                "long_term_slope": long_slope,
                "cumulative_anomaly_burden": anomaly_burden,
                "recovery_speed": recovery,
                "personal_deviation_persistence": persistence,
                "history_coverage": float(
                    len(valid_history)
                    / max(1, len([m for m in values_by_month if m <= cutoff]))
                ),
                "observation_density": float(
                    len(valid_history) / max(1, cutoff - min(values_by_month) + 1)
                ),
            }
        )
    return output


def _domain_series(
    participant_frame: pd.DataFrame,
    fields: Sequence[str],
    cutoff: int,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for month, group in participant_frame.groupby("__nominal_month", sort=False):
        if int(month) > cutoff:
            continue
        numeric = pd.to_numeric(group[list(fields)].iloc[0], errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        if not finite.empty:
            values[int(month)] = float(finite.mean())
    return values


def _synchrony_features(
    participant_frame: pd.DataFrame, cutoff: int, *, long_term: bool
) -> dict[str, float | int]:
    activity_fields = tuple(
        field for field in PSYCHE_D_SOURCE_FIELDS if field.startswith("steps")
    )
    sleep_fields = tuple(
        field for field in PSYCHE_D_SOURCE_FIELDS if field.startswith("sleep")
    )
    activity = _domain_series(participant_frame, activity_fields, cutoff)
    sleep = _domain_series(participant_frame, sleep_fields, cutoff)
    common = sorted(set(activity).intersection(sleep))
    if len(common) >= 2:
        activity_values = np.asarray([activity[m] for m in common], dtype="float64")
        sleep_values = np.asarray([sleep[m] for m in common], dtype="float64")
        activity_deltas = np.diff(activity_values)
        sleep_deltas = np.diff(sleep_values)
        synchrony_rate = (
            float(np.mean(np.sign(activity_deltas) == np.sign(sleep_deltas)))
            if activity_deltas.size
            else float("nan")
        )
        if np.std(activity_values) > EPSILON and np.std(sleep_values) > EPSILON:
            correlation = float(np.corrcoef(activity_values, sleep_values)[0, 1])
        else:
            correlation = float("nan")
        slope_gap = _slope(list(zip(common, activity_values, strict=True))) - _slope(
            list(zip(common, sleep_values, strict=True))
        )
    else:
        synchrony_rate = correlation = slope_gap = float("nan")
    current_activity = _numeric(activity.get(cutoff))
    prior_activity = _numeric(activity.get(cutoff - 1))
    current_sleep = _numeric(sleep.get(cutoff))
    prior_sleep = _numeric(sleep.get(cutoff - 1))
    activity_delta = (
        current_activity - prior_activity
        if np.isfinite(current_activity) and np.isfinite(prior_activity)
        else float("nan")
    )
    sleep_delta = (
        current_sleep - prior_sleep
        if np.isfinite(current_sleep) and np.isfinite(prior_sleep)
        else float("nan")
    )
    return {
        "activity_sleep__synchronous_change": (
            float(_direction(activity_delta) == _direction(sleep_delta))
            if np.isfinite(activity_delta) and np.isfinite(sleep_delta)
            else float("nan")
        ),
        "activity_sleep__delta_gap": (
            activity_delta - sleep_delta
            if np.isfinite(activity_delta) and np.isfinite(sleep_delta)
            else float("nan")
        ),
        "activity_sleep__long_term_corr": correlation if long_term else float("nan"),
        "activity_sleep__co_movement_slope_gap": slope_gap
        if long_term
        else float("nan"),
        "activity_sleep__synchronous_rate": synchrony_rate
        if long_term
        else float("nan"),
        "activity_sleep__history_month_count": len(common),
        "activity_sleep__history_missing": int(len(common) == 0),
    }


def _prepare_source(
    source: pd.DataFrame, source_fields: Sequence[str], split_path: Any
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    required = set(source_fields) | set(LABEL_FIELDS)
    missing = sorted(required.difference(source.columns))
    if missing:
        raise HorizonOptimizationError(f"source fields are missing: {missing}")
    parsed = parse_sample_index(source.index)
    frame = source.copy()
    frame["__participant_id"] = parsed["participant_id"].astype(str).to_numpy()
    frame["__nominal_month"] = parsed["nominal_month"].astype(int).to_numpy()
    for field in (*source_fields, *LABEL_FIELDS):
        values = pd.to_numeric(frame[field], errors="coerce").astype("float64")
        frame[field] = values.mask(~np.isfinite(values), np.nan)
    assignments = load_split_assignments(split_path)
    frame["__global_participant_id"] = frame["__participant_id"].map(
        lambda value: f"psyche_d::{value}"
    )
    frame["__outer_fold"] = frame["__global_participant_id"].map(
        lambda value: assignments.get(value, {}).get("outer_fold")
    )
    return frame, assignments


def _group_reference_stats(
    frame: pd.DataFrame, source_fields: Sequence[str]
) -> dict[tuple[int, int], dict[str, tuple[float, float, int]]]:
    stats: dict[tuple[int, int], dict[str, tuple[float, float, int]]] = {}
    for heldout_fold in range(5):
        train = frame.loc[
            frame["__outer_fold"].notna() & frame["__outer_fold"].ne(heldout_fold)
        ]
        for cutoff in range(
            int(frame["__nominal_month"].min()), int(frame["__nominal_month"].max()) + 1
        ):
            eligible = train.loc[train["__nominal_month"] < cutoff]
            by_field: dict[str, tuple[float, float, int]] = {}
            for field in source_fields:
                values = pd.to_numeric(eligible[field], errors="coerce").to_numpy(
                    dtype="float64"
                )
                values = values[np.isfinite(values)]
                median = float(np.median(values)) if values.size else float("nan")
                mad = _safe_mad(values, median) if values.size else float("nan")
                by_field[field] = (median, mad, int(values.size))
            stats[(heldout_fold, cutoff)] = by_field
    return stats


def _feature_names(source_fields: Sequence[str], *, long_term: bool) -> tuple[str, ...]:
    names: list[str] = []
    for field in source_fields:
        base = (
            "anchor",
            "anchor_missing",
            "delta_1",
            "delta_1_missing",
            "delta_2",
            "delta_2_missing",
            "local_slope",
            "local_slope_missing",
            "direction",
            "direction_missing",
            "direction_persistence",
            "change_point_distance",
            "change_point_distance_missing",
            "change_point_mean_delta",
            "change_point_mean_delta_missing",
            "history_count",
            "history_count_prior",
            "consecutive_missing_count",
            "months_since_last_valid",
            "months_since_last_valid_missing",
            "personal_median",
            "personal_mad",
            "personal_z",
            "personal_z_missing",
            "baseline_available",
            "cold_start",
            "group_reference_median",
            "group_reference_mad",
            "group_reference_z",
            "group_reference_available",
            "group_reference_count",
        )
        if long_term:
            base += (
                "historical_mean",
                "historical_median",
                "historical_mad",
                "historical_std",
                "historical_q10",
                "historical_q90",
                "long_term_slope",
                "cumulative_anomaly_burden",
                "recovery_speed",
                "personal_deviation_persistence",
                "history_coverage",
                "observation_density",
            )
        names.extend(f"{field}__{suffix}" for suffix in base)
    names.extend(
        (
            "activity_sleep__synchronous_change",
            "activity_sleep__delta_gap",
            "activity_sleep__long_term_corr",
            "activity_sleep__co_movement_slope_gap",
            "activity_sleep__synchronous_rate",
            "activity_sleep__history_month_count",
            "activity_sleep__history_missing",
        )
    )
    return tuple(names)


def build_horizon_context(
    source: pd.DataFrame,
    *,
    task_id: str,
    mapping_path: Any,
    split_path: Any,
    expected_count: int | None = None,
) -> HorizonContext:
    """Build one strict horizon without fitting any model or using labels as input."""

    if task_id not in FORECAST_TASKS:
        raise HorizonOptimizationError(f"unsupported task: {task_id}")
    source_fields = load_feature_names(mapping_path)
    frame, _assignments = _prepare_source(source, source_fields, split_path)
    baseline = build_forecast_samples(
        source,
        task_id=task_id,
        mapping_path=mapping_path,
        split_path=split_path,
        expected_count=expected_count,
    )
    long_term = task_id == "forecast_2m"
    references = _group_reference_stats(frame, source_fields)
    participant_frames = {
        participant: group.sort_values("__nominal_month", kind="mergesort")
        for participant, group in frame.groupby("__participant_id", sort=False)
    }
    output_rows: list[dict[str, Any]] = []
    for sample in baseline.to_dict(orient="records"):
        participant = str(sample["participant_id"])
        cutoff = int(sample["feature_month_slot"])
        heldout_fold = int(sample["outer_fold_id"])
        participant_frame = participant_frames[participant]
        values_by_field = {
            field: dict(
                zip(
                    participant_frame["__nominal_month"].astype(int),
                    participant_frame[field],
                    strict=True,
                )
            )
            for field in source_fields
        }
        row: dict[str, Any] = {
            name: sample[name]
            for name in (
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
                "feature_start_time",
                "feature_end_time",
                "assessment_time",
                "future_binary_target",
                "feature_nonmissing_count",
                "outer_fold_id",
            )
        }
        row["history_observed_month_count"] = int(
            participant_frame.loc[
                participant_frame["__nominal_month"].le(cutoff), "__nominal_month"
            ].nunique()
        )
        row["history_month_span"] = int(
            cutoff - int(participant_frame["__nominal_month"].min()) + 1
        )
        row["history_months_since_last_any_record"] = float(
            cutoff
            - int(
                participant_frame.loc[
                    participant_frame["__nominal_month"].le(cutoff), "__nominal_month"
                ].max()
            )
        )
        reference = references[(heldout_fold, cutoff)]
        personal_available_count = 0
        cold_start_count = 0
        group_available_count = 0
        for field in source_fields:
            field_values = _field_features(
                values_by_field[field], cutoff, long_term=long_term
            )
            group_median, group_mad, group_count = reference[field]
            anchor = _numeric(field_values["anchor"])
            group_available = int(np.isfinite(group_median) and group_count > 0)
            group_z = (
                (anchor - group_median) / max(1.4826 * group_mad, EPSILON)
                if group_available and np.isfinite(anchor)
                else float("nan")
            )
            field_values["group_reference_median"] = group_median
            field_values["group_reference_mad"] = group_mad
            field_values["group_reference_z"] = group_z
            field_values["group_reference_available"] = group_available
            field_values["group_reference_count"] = group_count
            personal_available_count += int(field_values["baseline_available"])
            cold_start_count += int(field_values["cold_start"])
            group_available_count += group_available
            row.update(
                {f"{field}__{key}": value for key, value in field_values.items()}
            )
        row.update(_synchrony_features(participant_frame, cutoff, long_term=long_term))
        row["baseline_available_count"] = personal_available_count
        row["cold_start_field_count"] = cold_start_count
        row["group_reference_available_count"] = group_available_count
        row["cold_start_flag"] = int(personal_available_count < len(source_fields))
        output_rows.append(row)
    feature_names = _feature_names(source_fields, long_term=long_term)
    metadata = [
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
        "feature_start_time",
        "feature_end_time",
        "assessment_time",
        "future_binary_target",
        "feature_nonmissing_count",
        "outer_fold_id",
        "history_observed_month_count",
        "history_month_span",
        "history_months_since_last_any_record",
        "baseline_available_count",
        "cold_start_field_count",
        "group_reference_available_count",
        "cold_start_flag",
    ]
    feature_frame = pd.DataFrame(output_rows).reindex(
        columns=metadata + list(feature_names)
    )
    _validate_context(feature_frame, task_id, feature_names, expected_count)
    groups = {
        "anchor": tuple(name for name in feature_names if "__anchor" in name),
        "delta": tuple(
            name
            for name in feature_names
            if "__delta_" in name or "__local_slope" in name
        ),
        "personal_baseline": tuple(
            name for name in feature_names if "personal_" in name or "baseline_" in name
        ),
        "missingness": tuple(
            name for name in feature_names if "missing" in name or "history_" in name
        ),
        "synchrony": tuple(
            name for name in feature_names if name.startswith("activity_sleep__")
        ),
    }
    if long_term:
        groups["long_term"] = tuple(
            name
            for name in feature_names
            if any(
                token in name
                for token in (
                    "historical_",
                    "long_term_",
                    "recovery_",
                    "anomaly_",
                    "observation_",
                )
            )
        )
    audit = {
        "task_id": task_id,
        "schema_version": HORIZON_FEATURE_VERSION,
        "sample_count": int(len(feature_frame)),
        "participant_count": int(feature_frame["global_participant_id"].nunique()),
        "outer_fold_counts": {
            str(int(fold)): int(count)
            for fold, count in feature_frame["outer_fold_id"]
            .value_counts()
            .sort_index()
            .items()
        },
        "feature_count": len(feature_names),
        "feature_groups": {key: len(value) for key, value in groups.items()},
        "cutoff_rule": "c=t-h; all derived inputs use nominal_month <= c",
        "personal_baseline_rule": "participant-local strict history months < c; no cross-participant fit",
        "group_reference_rule": "outer-train participants only; month < c",
        "missing_policy": "preserve nulls and explicit masks; no cross-cutoff fill",
        "prohibited_inputs": sorted(PROHIBITED_INPUTS),
        "training_started": False,
        "outer_oof_generated": False,
    }
    return HorizonContext(task_id, feature_frame, feature_names, groups, audit)


def _validate_context(
    frame: pd.DataFrame,
    task_id: str,
    feature_names: Sequence[str],
    expected_count: int | None,
) -> None:
    if frame.empty:
        raise HorizonOptimizationError("context is empty")
    if expected_count is not None and len(frame) != expected_count:
        raise HorizonOptimizationError(f"{task_id} context row count changed")
    if frame["target_window_id"].duplicated().any():
        raise HorizonOptimizationError("context target windows are duplicated")
    gap = FORECAST_TASKS[task_id]
    if (frame["feature_month_slot"] != frame["label_month_slot"] - gap).any():
        raise HorizonOptimizationError("context cutoff is not c=t-h")
    if set(feature_names).intersection(PROHIBITED_INPUTS):
        raise HorizonOptimizationError("prohibited target field entered feature names")
    if any("phq" in name.lower() for name in feature_names):
        raise HorizonOptimizationError("PHQ field entered model features")
    if frame["timing_evidence"].ne("design_level").any():
        raise HorizonOptimizationError("unexpected timing evidence")


def feature_manifest(contexts: Mapping[str, HorizonContext]) -> dict[str, Any]:
    return {
        "schema_version": "mood-social-forecast-v3.4-opt002b-feature-manifest-v1",
        "source_feature_count": 27,
        "horizons": {
            task: {
                "feature_count": len(context.feature_names),
                "feature_names": list(context.feature_names),
                "feature_groups": {
                    name: list(values)
                    for name, values in context.feature_groups.items()
                },
                "audit": dict(context.audit),
            }
            for task, context in contexts.items()
        },
        "prohibited_inputs": sorted(PROHIBITED_INPUTS),
        "input_policy": "Only nominal_month <= c=t-h; PHQ and future fields are target-only",
        "training_started": False,
    }


__all__ = [
    "EPSILON",
    "HORIZON_FEATURE_VERSION",
    "HorizonContext",
    "HorizonOptimizationError",
    "PROHIBITED_INPUTS",
    "build_horizon_context",
    "feature_manifest",
]
