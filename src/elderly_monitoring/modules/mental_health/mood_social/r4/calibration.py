"""Inner-OOF-only calibration selection for r4 outer predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import (
    ap_context_metrics,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    participant_equal_weights,
)


EPSILON = 1.0e-6
CalibrationMethod = Literal["none", "platt", "beta", "isotonic"]


def _clip(probability: Sequence[float]) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)


def _matrix(probability: Sequence[float], method: str) -> np.ndarray:
    value = _clip(probability)
    if method == "platt":
        return np.log(value / (1.0 - value))[:, None]
    if method == "beta":
        return np.column_stack([np.log(value), -np.log1p(-value)])
    raise ValueError(f"unsupported calibration matrix: {method}")


@dataclass(frozen=True)
class FittedCalibration:
    method: CalibrationMethod
    estimator: Any | None
    constant: float | None = None

    def predict(self, probability: Sequence[float]) -> np.ndarray:
        value = _clip(probability)
        if self.constant is not None:
            return np.full(len(value), self.constant, dtype=float)
        if self.method == "none":
            return value
        if self.method == "isotonic":
            return _clip(self.estimator.predict(value))
        return _clip(self.estimator.predict_proba(_matrix(value, self.method))[:, 1])


def fit_calibration(
    frame: pd.DataFrame,
    probability: Sequence[float],
    method: CalibrationMethod,
) -> FittedCalibration:
    value = _clip(probability)
    target = frame["binary_target"].to_numpy(int)
    weight = participant_equal_weights(frame)
    if method == "none":
        return FittedCalibration("none", None)
    if np.unique(target).size < 2:
        return FittedCalibration(method, None, float(np.clip(target.mean(), EPSILON, 1 - EPSILON)))
    if method == "isotonic":
        if len(frame) < 200 or np.unique(value).size < 20:
            return FittedCalibration("none", None)
        estimator = IsotonicRegression(out_of_bounds="clip", y_min=EPSILON, y_max=1.0 - EPSILON)
        estimator.fit(value, target, sample_weight=weight)
        return FittedCalibration("isotonic", estimator)
    estimator = LogisticRegression(
        solver="liblinear", C=1.0, max_iter=3000, random_state=20260812
    )
    estimator.fit(_matrix(value, method), target, sample_weight=weight)
    return FittedCalibration(method, estimator)


def _crossfit(
    frame: pd.DataFrame,
    raw_probability: np.ndarray,
    method: CalibrationMethod,
) -> np.ndarray:
    fold = frame["inner_fold"].astype(int)
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    raw = pd.Series(np.asarray(raw_probability, dtype=float), index=frame.index)
    for validation_fold in sorted(fold.unique()):
        train_mask = fold.ne(validation_fold)
        validation_mask = fold.eq(validation_fold)
        fitted = fit_calibration(
            frame.loc[train_mask], raw.loc[train_mask].to_numpy(float), method
        )
        result.loc[validation_mask] = fitted.predict(
            raw.loc[validation_mask].to_numpy(float)
        )
    if result.isna().any():
        raise ValueError("r4 calibration crossfit is incomplete")
    return result.loc[frame.index].to_numpy(float)


def calibrate_outer_prediction(
    inner_frame: pd.DataFrame,
    raw_inner_probability: np.ndarray,
    raw_outer_probability: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    methods: tuple[CalibrationMethod, ...] = ("none", "platt", "beta", "isotonic")
    candidates: dict[str, dict[str, float]] = {}
    for method in methods:
        calibrated = _crossfit(inner_frame, raw_inner_probability, method)
        metric = ap_context_metrics(
            inner_frame["binary_target"].to_numpy(int),
            calibrated,
            sample_weight=participant_equal_weights(inner_frame),
        )
        candidates[method] = metric
    selected = min(
        methods,
        key=lambda method: (
            candidates[method]["brier"] + 0.25 * candidates[method]["ece"],
            methods.index(method),
        ),
    )
    fitted = fit_calibration(inner_frame, raw_inner_probability, selected)
    outer = fitted.predict(raw_outer_probability)
    return outer, {
        "selected_method": selected,
        "selection_metric": "participant_equal_brier_plus_0.25_ece",
        "candidate_metrics": candidates,
        "ranking_metric_note": "calibration does not select or claim AUPRC gain",
    }


__all__ = ["calibrate_outer_prediction", "fit_calibration"]
