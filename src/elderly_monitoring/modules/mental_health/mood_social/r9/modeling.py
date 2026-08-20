"""Participant-safe fold-local modeling primitives for R9."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.calibration import (
    FittedCalibration,
    fit_calibration,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    CandidateSpec,
    fit_model,
    participant_equal_weights,
    predict_model,
)
from elderly_monitoring.modules.mental_health.mood_social.r7.modeling import (
    binary_metrics,
    project_heads,
    select_specificity_threshold,
)


HEAD_TARGETS = {"ge5": "phq9_ge5_target", "ge10": "phq9_ge10_target"}


def inner_folds_for_outer(frame: pd.DataFrame, outer_fold: int) -> pd.Series:
    key = str(int(outer_fold))

    def extract(value: object) -> float:
        parsed = json.loads(value) if isinstance(value, str) else dict(value)  # type: ignore[arg-type]
        result = parsed.get(key)
        return np.nan if result is None else float(result)

    folds = frame["inner_validation_fold_by_outer_fold"].map(extract)
    is_test = frame["outer_fold"].eq(int(outer_fold))
    if folds[is_test].notna().any() or folds[~is_test].isna().any():
        raise ValueError(f"invalid R9 inner-fold coverage for outer {outer_fold}")
    participant_count = (
        frame.loc[~is_test]
        .assign(_fold=folds.loc[~is_test].astype(int))
        .groupby("global_participant_id")["_fold"]
        .nunique()
    )
    if not participant_count.eq(1).all():
        raise ValueError(f"participant crosses R9 inner folds for outer {outer_fold}")
    return folds


def training_weights(frame: pd.DataFrame, mode: str) -> np.ndarray:
    participant = participant_equal_weights(frame)
    if mode == "participant_equal":
        return participant
    if mode != "participant_source_balanced":
        raise ValueError(f"unknown R9 training weight mode: {mode}")
    counts = frame["dataset_id"].value_counts()
    factor = frame["dataset_id"].map(
        {source: np.sqrt(len(frame) / max(int(count), 1)) for source, count in counts.items()}
    ).to_numpy(float)
    factor = np.clip(factor, 0.5, 3.0)
    weight = participant * factor
    return weight * len(weight) / weight.sum()


def fit_inner_oof(
    train: pd.DataFrame,
    features: Sequence[str],
    target: str,
    folds: pd.Series,
    spec: CandidateSpec,
    *,
    weight_mode: str = "participant_equal",
) -> np.ndarray:
    output = pd.Series(np.nan, index=train.index, dtype=float)
    for validation_fold in sorted(folds.astype(int).unique()):
        fit = train.loc[folds.ne(validation_fold)].copy()
        validation = train.loc[folds.eq(validation_fold)].copy()
        fit["binary_target"] = fit[target].astype(int)
        model = fit_model(
            fit,
            features,
            spec,
            participant_equal=False,
            sample_weight=training_weights(fit, weight_mode),
        )
        output.loc[validation.index] = predict_model(model, validation, features)
    if output.isna().any():
        raise ValueError("R9 inner OOF is incomplete")
    return output.loc[train.index].to_numpy(float)


def fit_outer_fixed(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    target: str,
    folds: pd.Series,
    spec: CandidateSpec,
    *,
    weight_mode: str = "participant_equal",
) -> tuple[np.ndarray, np.ndarray, FittedCalibration, dict[str, Any]]:
    inner_raw = fit_inner_oof(
        train, features, target, folds, spec, weight_mode=weight_mode
    )
    calibration_frame = train.copy()
    calibration_frame["binary_target"] = calibration_frame[target].astype(int)
    calibrator = fit_calibration(calibration_frame, inner_raw, "platt")
    inner_probability = calibrator.predict(inner_raw)
    fit = train.copy()
    fit["binary_target"] = fit[target].astype(int)
    model = fit_model(
        fit,
        features,
        spec,
        participant_equal=False,
        sample_weight=training_weights(fit, weight_mode),
    )
    outer_probability = calibrator.predict(predict_model(model, test, features))
    audit = {
        "spec": asdict(spec),
        "weight_mode": weight_mode,
        "calibration": "platt-fit-on-outer-train-inner-oof",
        "workpoints": {
            "specificity_080": select_specificity_threshold(
                train[target].astype(int), inner_probability, 0.80
            ),
            "specificity_090": select_specificity_threshold(
                train[target].astype(int), inner_probability, 0.90
            ),
        },
    }
    return (
        np.asarray(outer_probability, dtype=float),
        np.asarray(inner_probability, dtype=float),
        calibrator,
        audit,
    )


def sensitivity_at_specificity(
    target: Iterable[int], probability: Iterable[float], specificity: float = 0.80
) -> float:
    y = np.asarray(list(target), dtype=int)
    p = np.asarray(list(probability), dtype=float)
    threshold = select_specificity_threshold(y, p, specificity)
    positive = y == 1
    if not positive.any():
        return 0.0
    return float((p[positive] >= threshold).mean())


def metric_block(
    frame: pd.DataFrame,
    *,
    target: str,
    probability: str,
    participant_equal: bool,
) -> dict[str, float]:
    weight = participant_equal_weights(frame) if participant_equal else None
    metrics = binary_metrics(
        frame[target].astype(int), frame[probability].astype(float), weight=weight
    )
    metrics["sensitivity_at_specificity_080"] = sensitivity_at_specificity(
        frame[target].astype(int), frame[probability].astype(float), 0.80
    )
    metrics["rows"] = float(len(frame))
    metrics["participants"] = float(frame["global_participant_id"].nunique())
    metrics["positive"] = float(frame[target].sum())
    return metrics


def candidate_score(
    frame: pd.DataFrame, target: str, probability: np.ndarray
) -> tuple[float, dict[str, float]]:
    work = frame.copy()
    work["_candidate_probability"] = probability
    metrics = metric_block(
        work,
        target=target,
        probability="_candidate_probability",
        participant_equal=True,
    )
    by_source: list[float] = []
    for _, part in work.groupby("dataset_id", sort=True):
        if part[target].nunique() == 2:
            by_source.append(
                metric_block(
                    part,
                    target=target,
                    probability="_candidate_probability",
                    participant_equal=True,
                )["normalized_ap"]
            )
    worst_source = min(by_source) if by_source else metrics["normalized_ap"]
    score = 0.8 * metrics["normalized_ap"] + 0.2 * worst_source
    metrics["worst_source_normalized_ap"] = float(worst_source)
    metrics["selection_score"] = float(score)
    return float(score), metrics


__all__ = [
    "HEAD_TARGETS",
    "candidate_score",
    "fit_inner_oof",
    "fit_outer_fixed",
    "inner_folds_for_outer",
    "metric_block",
    "project_heads",
    "sensitivity_at_specificity",
    "training_weights",
]
