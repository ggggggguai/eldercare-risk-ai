"""Fail-closed LODO primitives and all-route compatibility smoke checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


EPSILON = 1.0e-6


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


@dataclass(frozen=True)
class ConstantCalibrator:
    probability: float
    reason: str = "single_class_or_empty_calibration_partition"

    def predict(self, probability: Sequence[float]) -> np.ndarray:
        return np.full(len(np.asarray(probability)), self.probability, dtype=float)


@dataclass(frozen=True)
class PlattCalibrator:
    estimator: LogisticRegression

    def predict(self, probability: Sequence[float]) -> np.ndarray:
        values = _logit(np.asarray(probability, dtype=float))[:, None]
        return np.clip(self.estimator.predict_proba(values)[:, 1], EPSILON, 1.0 - EPSILON)


def fit_safe_calibrator(
    probability: np.ndarray, target: np.ndarray
) -> ConstantCalibrator | PlattCalibrator:
    """Fit Platt when possible; use an audited constant for degenerate folds."""

    probability = np.asarray(probability, dtype=float)
    target = np.asarray(target, dtype=int)
    selected = np.isfinite(probability)
    if not selected.any():
        return ConstantCalibrator(0.5, "empty_calibration_partition")
    probability = probability[selected]
    target = target[selected]
    if np.unique(target).size < 2:
        return ConstantCalibrator(
            float(np.clip(target.mean(), EPSILON, 1.0 - EPSILON)),
            "single_class_calibration_partition",
        )
    estimator = LogisticRegression(
        solver="liblinear", C=1.0, max_iter=3000, random_state=20260812
    )
    estimator.fit(_logit(probability)[:, None], target)
    return PlattCalibrator(estimator)


def safe_predict_available(
    estimator: Any,
    frame: pd.DataFrame,
    features: Sequence[str],
    available: Sequence[bool],
) -> np.ndarray:
    """Predict only supported rows, preserving unsupported routes as NaN."""

    mask = np.asarray(available, dtype=bool)
    if len(mask) != len(frame):
        raise ValueError("availability mask length mismatch")
    result = np.full(len(frame), np.nan, dtype=float)
    if not mask.any():
        return result
    values = frame.loc[mask, list(features)]
    if hasattr(estimator, "predict_proba"):
        predicted = np.asarray(estimator.predict_proba(values), dtype=float)[:, 1]
    else:
        predicted = np.asarray(estimator.predict(values), dtype=float)
    if len(predicted) != int(mask.sum()) or not np.isfinite(predicted).all():
        raise ValueError("invalid supported-route prediction")
    result[mask] = np.clip(predicted, EPSILON, 1.0 - EPSILON)
    return result


def _synthetic_lodo_frame() -> pd.DataFrame:
    specifications = {
        "nhanes": "111",
        "nhanes_ssq_2005_2008": "001",
        "psyche_d": "010",
        "shenzhen_elderly": "001",
        # Real r4 training keeps the small Resilient source; it supplies a
        # 111 route after NHANES is held out, albeit with low reliability.
        "resilient": "111",
    }
    rows: list[dict[str, Any]] = []
    for source_index, (source, route) in enumerate(specifications.items()):
        for index in range(12):
            rows.append(
                {
                    "dataset_id": source,
                    "global_participant_id": f"{source}::p{index}",
                    "route_pattern": route,
                    "x": float(index) / 11.0 + source_index * 0.01,
                    "binary_target": int(index >= 6),
                }
            )
    return pd.DataFrame(rows)


def validate_lodo_route_compatibility() -> dict[str, dict[str, Any]]:
    """Exercise every major source, including absent-route abstention paths."""

    frame = _synthetic_lodo_frame()
    major_sources = (
        "nhanes",
        "nhanes_ssq_2005_2008",
        "psyche_d",
        "shenzhen_elderly",
    )
    result: dict[str, dict[str, Any]] = {}
    for source in major_sources:
        train = frame.loc[frame["dataset_id"].ne(source)].copy()
        test = frame.loc[frame["dataset_id"].eq(source)].copy()
        overlap = set(train["global_participant_id"]) & set(test["global_participant_id"])
        supported_routes = set(train["route_pattern"].astype(str))
        available = test["route_pattern"].astype(str).isin(supported_routes).to_numpy()
        model = LogisticRegression(solver="liblinear", random_state=20260812)
        model.fit(train[["x"]], train["binary_target"])
        raw = safe_predict_available(model, test, ["x"], available)
        if np.isfinite(raw).any():
            calibrator = fit_safe_calibrator(
                model.predict_proba(train[["x"]])[:, 1],
                train["binary_target"].to_numpy(int),
            )
            raw[np.isfinite(raw)] = calibrator.predict(raw[np.isfinite(raw)])
        result[source] = {
            "status": "pass",
            "held_out_route_patterns": sorted(test["route_pattern"].astype(str).unique()),
            "train_supported_route_patterns": sorted(supported_routes),
            "row_count": int(len(test)),
            "prediction_coverage": float(np.isfinite(raw).mean()),
            "unsupported_rows_abstained": int(np.isnan(raw).sum()),
            "participant_overlap": int(len(overlap)),
        }
    return result


__all__ = [
    "ConstantCalibrator",
    "fit_safe_calibrator",
    "safe_predict_available",
    "validate_lodo_route_compatibility",
]
