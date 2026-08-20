"""Leakage-safe auxiliary targets and training-weight audits for 002C."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from elderly_monitoring.datasets.adapters.psyche_d import (
    LABEL_FIELDS,
    parse_sample_index,
)

TARGET_WEIGHT_VERSION = "mood-social-forecast-v3.4-opt002c-target-weight-v1"
PHQ9_THRESHOLD = 10
PHQ9_NEAR_RADIUS = 1
RECENCY_DECAY = 0.15


class TargetWeightError(ValueError):
    """Raised when a target or weight violates the 002C contract."""


def _finite(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _validate_source(source: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(LABEL_FIELDS).difference(source.columns))
    if missing:
        raise TargetWeightError(f"source is missing target fields: {missing}")
    parsed = parse_sample_index(source.index)
    frame = source.copy()
    frame["__participant_id"] = parsed["participant_id"].astype(str).to_numpy()
    frame["__nominal_month"] = parsed["nominal_month"].astype(int).to_numpy()
    for field in LABEL_FIELDS:
        values = pd.to_numeric(frame[field], errors="coerce").astype("float64")
        frame[field] = values.mask(~np.isfinite(values), np.nan)
    return frame


def phq9_ordinal(score: float | int) -> int:
    """Map PHQ-9 total to the frozen five-level ordinal severity."""

    value = _finite(score)
    if not np.isfinite(value) or value < 0 or value > 27 or value % 1:
        raise TargetWeightError("PHQ-9 score must be an integer in [0, 27]")
    if value <= 4:
        return 0
    if value <= 9:
        return 1
    if value <= 14:
        return 2
    if value <= 19:
        return 3
    return 4


def build_auxiliary_targets(
    source: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    """Attach target-only PHQ-9 fields using only rows at the label month.

    The previous valid score is looked up strictly before the label month and
    is used solely to construct the direction target.  It is never returned as
    a feature column.
    """

    frame = _validate_source(source)
    if "target_window_id" not in context.columns or "label_month_slot" not in context:
        raise TargetWeightError("context lacks target identity columns")
    participant_frames = {
        participant: rows.sort_values("__nominal_month", kind="mergesort")
        for participant, rows in frame.groupby("__participant_id", sort=False)
    }
    lookup = frame.set_index(["__participant_id", "__nominal_month"], drop=False)
    rows: list[dict[str, Any]] = []
    for sample in context[
        [
            "target_window_id",
            "global_participant_id",
            "label_month_slot",
            "task_id",
            "future_binary_target",
        ]
    ].to_dict(orient="records"):
        participant = str(sample["global_participant_id"]).split("::", 1)[-1]
        label_month = int(sample["label_month_slot"])
        key = (participant, label_month)
        if key not in lookup.index:
            raise TargetWeightError("label row disappeared during target construction")
        target = lookup.loc[key]
        if isinstance(target, pd.DataFrame):
            raise TargetWeightError("target key is duplicated")
        score = _finite(target["phq9_score_end"])
        ordinal = _finite(target["phq9_cat_end"])
        if np.isfinite(score) and (score < 0 or score > 27 or score % 1):
            raise TargetWeightError("target PHQ-9 score is invalid")
        if np.isfinite(ordinal) and (ordinal < 0 or ordinal > 4 or ordinal % 1):
            raise TargetWeightError("target PHQ-9 ordinal is invalid")
        if np.isfinite(score) and np.isfinite(ordinal):
            if int(ordinal) != phq9_ordinal(score):
                raise TargetWeightError("PHQ-9 score and ordinal target disagree")
        participant_frame = participant_frames[participant]
        previous = participant_frame.loc[
            participant_frame["__nominal_month"].lt(label_month)
            & participant_frame["phq9_score_end"].notna()
        ]
        previous_score = (
            _finite(previous.iloc[-1]["phq9_score_end"])
            if not previous.empty
            else float("nan")
        )
        direction = float("nan")
        if np.isfinite(score) and np.isfinite(previous_score):
            direction = float(np.sign(score - previous_score))
        rows.append(
            {
                "target_window_id": sample["target_window_id"],
                "global_participant_id": sample["global_participant_id"],
                "task_id": sample["task_id"],
                "label_month_slot": label_month,
                "future_binary_target": int(sample["future_binary_target"]),
                "aux_phq9_score": score,
                "aux_phq9_score_available": int(np.isfinite(score)),
                "aux_phq9_ordinal": ordinal,
                "aux_phq9_ordinal_available": int(np.isfinite(ordinal)),
                "previous_phq9_score_target_only": previous_score,
                "previous_phq9_available": int(np.isfinite(previous_score)),
                "aux_phq9_direction": direction,
                "aux_phq9_direction_available": int(np.isfinite(direction)),
                "threshold_near_flag": int(
                    np.isfinite(score)
                    and abs(score - PHQ9_THRESHOLD) <= PHQ9_NEAR_RADIUS
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result["target_window_id"].duplicated().any():
        raise TargetWeightError("target windows are duplicated")
    _validate_targets(result)
    return result


def _validate_targets(targets: pd.DataFrame) -> None:
    if any("phq" in name.lower() for name in targets.columns if name == "feature"):
        raise TargetWeightError("PHQ field was classified as a feature")
    for field in ("aux_phq9_score", "previous_phq9_score_target_only"):
        values = pd.to_numeric(targets[field], errors="coerce").dropna()
        if ((values < 0) | (values > 27)).any():
            raise TargetWeightError(f"{field} is outside [0, 27]")
    if not set(targets["aux_phq9_direction"].dropna().unique()).issubset(
        {-1.0, 0.0, 1.0}
    ):
        raise TargetWeightError("direction target has an invalid value")


def participant_equal_weight(frame: pd.DataFrame) -> pd.Series:
    """Give each participant the same total weight in the supplied fit set."""

    counts = frame.groupby("global_participant_id", sort=False)[
        "target_window_id"
    ].transform("count")
    return pd.Series(1.0 / counts.astype("float64"), index=frame.index, name="weight")


def inner_class_balanced_weight(frame: pd.DataFrame) -> pd.Series:
    """Class balance computed only from the supplied inner-train rows."""

    labels = pd.to_numeric(frame["future_binary_target"], errors="raise").astype("int8")
    counts = labels.value_counts().to_dict()
    if set(counts) != {0, 1}:
        raise TargetWeightError("inner training set must contain both binary classes")
    total = len(labels)
    class_count = {int(key): int(value) for key, value in counts.items()}
    values = labels.map(
        {key: total / (2.0 * count) for key, count in class_count.items()}
    )
    return pd.Series(values.to_numpy(dtype="float64"), index=frame.index, name="weight")


def strict_history_recency_weight(
    frame: pd.DataFrame, decay: float = RECENCY_DECAY
) -> pd.Series:
    """Apply finite decay from the latest cutoff in this fit set only."""

    if decay <= 0 or not np.isfinite(decay):
        raise TargetWeightError("recency decay must be positive and finite")
    latest = int(pd.to_numeric(frame["feature_month_slot"], errors="raise").max())
    interval = latest - pd.to_numeric(
        frame["feature_month_slot"], errors="raise"
    ).astype("float64")
    values = np.exp(-decay * interval.to_numpy(dtype="float64"))
    values = values / max(float(values.mean()), 1e-12)
    return pd.Series(values, index=frame.index, name="weight")


def threshold_near_weight(frame: pd.DataFrame, near_weight: float = 0.5) -> pd.Series:
    """Downweight only threshold-near rows; hard labels remain unchanged."""

    if not 0 < near_weight <= 1:
        raise TargetWeightError("near threshold weight must be in (0, 1]")
    score = pd.to_numeric(frame["aux_phq9_score"], errors="coerce")
    values = np.where(
        score.notna() & (score.sub(PHQ9_THRESHOLD).abs() <= PHQ9_NEAR_RADIUS),
        near_weight,
        1.0,
    )
    return pd.Series(values.astype("float64"), index=frame.index, name="weight")


def _inner_fold_map(
    assignments: Mapping[str, Mapping[str, Any]], outer_fold: int
) -> dict[str, int]:
    values: dict[str, int] = {}
    for participant, row in assignments.items():
        if int(row["outer_fold"]) == outer_fold:
            continue
        mapping = row.get("inner_validation_fold_by_outer_fold") or {}
        inner = mapping.get(str(outer_fold), mapping.get(outer_fold))
        if inner is None:
            raise TargetWeightError("inner fold assignment is missing")
        values[participant] = int(inner)
    return values


def build_weight_audit(
    targets: pd.DataFrame,
    assignments: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build auditable weight rows for every outer/inner training scope.

    The returned long table never contains an outer-test row in a fitting
    scope.  It is an audit artifact, not a model-training result.
    """

    rows: list[pd.DataFrame] = []
    scheme_names = (
        "participant_equal",
        "inner_fold_class_balanced",
        "strict_history_recency",
        "threshold_near_downweight",
    )
    for outer_fold in range(5):
        outer_train = targets.loc[targets["outer_fold_id"].ne(outer_fold)].copy()
        if outer_train.empty:
            raise TargetWeightError("outer training partition is empty")
        fold_map = _inner_fold_map(assignments, outer_fold)
        outer_train["participant_inner_fold_id"] = outer_train[
            "global_participant_id"
        ].map(fold_map)
        if outer_train["participant_inner_fold_id"].isna().any():
            raise TargetWeightError("outer-train participant lacks inner assignment")
        for inner_fold in range(5):
            inner_train = outer_train.loc[
                outer_train["participant_inner_fold_id"].ne(inner_fold)
            ].copy()
            if inner_train.empty:
                raise TargetWeightError("inner training partition is empty")
            values_by_scheme = {
                "participant_equal": participant_equal_weight(inner_train),
                "inner_fold_class_balanced": inner_class_balanced_weight(inner_train),
                "strict_history_recency": strict_history_recency_weight(inner_train),
                "threshold_near_downweight": threshold_near_weight(inner_train),
            }
            for scheme in scheme_names:
                part = inner_train[
                    [
                        "target_window_id",
                        "global_participant_id",
                        "task_id",
                        "outer_fold_id",
                        "participant_inner_fold_id",
                    ]
                ].copy()
                part["inner_fold_id"] = inner_fold
                part["evaluation_outer_fold"] = outer_fold
                part["fit_scope"] = "outer_train_inner_train"
                part["weight_scheme"] = scheme
                part["weight"] = values_by_scheme[scheme].to_numpy(dtype="float64")
                part["target_in_fit"] = True
                rows.append(part)
        # A participant-equal audit for the complete outer-train scope makes
        # it possible to verify the formal evaluation denominator separately.
        part = outer_train[
            [
                "target_window_id",
                "global_participant_id",
                "task_id",
                "outer_fold_id",
                "participant_inner_fold_id",
            ]
        ].copy()
        part["inner_fold_id"] = -1
        part["evaluation_outer_fold"] = outer_fold
        part["fit_scope"] = "outer_train"
        part["weight_scheme"] = "participant_equal"
        part["weight"] = participant_equal_weight(outer_train).to_numpy(dtype="float64")
        part["target_in_fit"] = True
        rows.append(part)
    audit_table = pd.concat(rows, ignore_index=True)
    if (
        not np.isfinite(audit_table["weight"]).all()
        or (audit_table["weight"] <= 0).any()
    ):
        raise TargetWeightError("weight audit contains invalid values")
    target_ids = set(targets["target_window_id"])
    if not set(audit_table["target_window_id"]).issubset(target_ids):
        raise TargetWeightError("weight audit references unknown target windows")
    per_participant = (
        audit_table.loc[audit_table["weight_scheme"].eq("participant_equal")]
        .groupby(
            [
                "fit_scope",
                "evaluation_outer_fold",
                "inner_fold_id",
                "global_participant_id",
            ]
        )["weight"]
        .sum()
    )
    participant_equal_ok = bool(np.allclose(per_participant.to_numpy(), 1.0))
    outer_test_in_fit = bool(
        (audit_table["outer_fold_id"] == audit_table["evaluation_outer_fold"]).any()
    )
    audit = {
        "schema_version": TARGET_WEIGHT_VERSION,
        "scheme_names": list(scheme_names),
        "outer_fold_count": 5,
        "inner_fold_count": 5,
        "participant_equal_total_is_one": participant_equal_ok,
        "outer_test_rows_in_fit": outer_test_in_fit,
        "ordinary_smote": False,
        "cross_participant_synthesis": False,
        "cross_month_random_concat": False,
        "fit_scope": "outer_train then inner_train only",
        "formal_evaluation_weights": "original participant-equal hard-label protocol",
        "training_started": False,
    }
    if outer_test_in_fit:
        raise TargetWeightError("outer-test row entered a fitting scope")
    return audit_table, audit


def target_manifest(targets: pd.DataFrame) -> dict[str, Any]:
    available = {
        field: int(targets[field].notna().sum())
        for field in (
            "aux_phq9_score",
            "aux_phq9_ordinal",
            "aux_phq9_direction",
            "previous_phq9_score_target_only",
        )
    }
    score = pd.to_numeric(targets["aux_phq9_score"], errors="coerce")
    direction_counts = {
        str(int(value)): int(count)
        for value, count in targets["aux_phq9_direction"]
        .dropna()
        .value_counts()
        .sort_index()
        .items()
    }
    return {
        "schema_version": "mood-social-forecast-v3.4-opt002c-target-manifest-v1",
        "target_roles": {
            "future_binary_target": "primary_v3_4_hard_label",
            "aux_phq9_score": "auxiliary_regression_target_only",
            "aux_phq9_ordinal": "auxiliary_ordinal_target_only",
            "aux_phq9_direction": "auxiliary_worsening_improvement_target_only",
            "previous_phq9_score_target_only": "target_construction_only",
        },
        "input_feature_policy": "PHQ-9 scores, categories and directions are prohibited from inference inputs",
        "label_availability": {
            field: {
                "available_count": count,
                "missing_count": int(len(targets) - count),
                "available_rate": float(count / len(targets)),
            }
            for field, count in available.items()
        },
        "direction_distribution": direction_counts,
        "threshold": {
            "phq9_binary_threshold": PHQ9_THRESHOLD,
            "near_radius": PHQ9_NEAR_RADIUS,
            "near_count": int(targets["threshold_near_flag"].sum()),
            "near_score_distribution": {
                str(int(value)): int(count)
                for value, count in score.dropna().value_counts().sort_index().items()
                if abs(float(value) - PHQ9_THRESHOLD) <= PHQ9_NEAR_RADIUS
            },
        },
        "direction_definition": "sign(current valid PHQ-9 end minus previous valid PHQ-9 end strictly before label month)",
        "evaluation_label": "future_binary_target with original participant-equal protocol",
        "training_started": False,
        "outer_oof_generated": False,
    }


__all__ = [
    "PHQ9_NEAR_RADIUS",
    "PHQ9_THRESHOLD",
    "RECENCY_DECAY",
    "TARGET_WEIGHT_VERSION",
    "TargetWeightError",
    "build_auxiliary_targets",
    "build_weight_audit",
    "inner_class_balanced_weight",
    "participant_equal_weight",
    "phq9_ordinal",
    "strict_history_recency_weight",
    "target_manifest",
    "threshold_near_weight",
]
