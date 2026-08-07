"""Strict inner-OOF fusion and calibration for FORECAST-OPT-001D."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from elderly_monitoring.modules.mental_health.mood_social.forecast_modeling import (
    expected_calibration_error,
    participant_equal_weights,
    select_threshold,
)
from elderly_monitoring.modules.mental_health.mood_social.forecast_optimization import (
    ForecastOptimizationError,
)


EPSILON = 1.0e-6
CALIBRATION_METHODS = ("none", "platt", "isotonic", "beta")
COMPONENT_FAMILIES = ("elasticnet_logistic", "lightgbm", "catboost")


def simplex_weight_grid(step: float = 0.1) -> list[tuple[float, float, float]]:
    units = round(1.0 / step)
    if step <= 0.0 or not np.isclose(units * step, 1.0):
        raise ForecastOptimizationError("weight step must exactly divide one")
    return [
        (first / units, second / units, (units - first - second) / units)
        for first in range(units + 1)
        for second in range(units - first + 1)
    ]


def select_family_components(search: pd.DataFrame) -> pd.DataFrame:
    passed = search.loc[search["status"].eq("passed")].copy()
    if set(passed["model_family"]) != set(COMPONENT_FAMILIES):
        raise ForecastOptimizationError("a fusion component family is unavailable")
    selected = (
        passed.sort_values(
            ["model_family", "auprc", "brier", "candidate_index"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .groupby("model_family", sort=False, as_index=False)
        .head(1)
        .set_index("model_family")
        .loc[list(COMPONENT_FAMILIES)]
        .reset_index()
    )
    return selected


def component_oof_table(all_oof: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "global_participant_id",
        "target_window_id",
        "future_binary_target",
        "outer_fold_id",
        "inner_fold_id",
    ]
    parts: list[pd.DataFrame] = []
    for row in selected.to_dict(orient="records"):
        part = all_oof.loc[
            all_oof["candidate_id"].eq(row["candidate_id"]), identifiers + ["p_raw"]
        ].copy()
        if part["target_window_id"].duplicated().any():
            raise ForecastOptimizationError("component inner OOF has duplicate windows")
        part = part.rename(columns={"p_raw": f"p_{row['model_family']}"})
        parts.append(part)
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on=identifiers, how="inner", validate="one_to_one")
    expected = all_oof.loc[all_oof["candidate_id"].eq(selected.iloc[0]["candidate_id"])]
    if len(merged) != len(expected):
        raise ForecastOptimizationError("component inner OOF coverage is incomplete")
    return merged.sort_values("target_window_id", kind="mergesort").reset_index(
        drop=True
    )


def search_blend_weights(
    frame: pd.DataFrame, step: float = 0.1
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    weights = participant_equal_weights(frame)
    target = frame["future_binary_target"].to_numpy(dtype="int8")
    matrix = frame[[f"p_{name}" for name in COMPONENT_FAMILIES]].to_numpy(
        dtype="float64"
    )
    records: list[dict[str, Any]] = []
    best_key: tuple[float, float, int] | None = None
    best_index = -1
    best_probability: np.ndarray | None = None
    for grid_index, values in enumerate(simplex_weight_grid(step)):
        probability = matrix @ np.asarray(values, dtype="float64")
        auprc = float(
            average_precision_score(target, probability, sample_weight=weights)
        )
        brier = float(np.average((probability - target) ** 2, weights=weights))
        records.append(
            {
                "grid_index": grid_index,
                **{
                    f"weight_{name}": value
                    for name, value in zip(COMPONENT_FAMILIES, values, strict=True)
                },
                "auprc": auprc,
                "brier": brier,
            }
        )
        key = (-auprc, brier, grid_index)
        if best_key is None or key < best_key:
            best_key = key
            best_index = grid_index
            best_probability = probability
    if best_probability is None:
        raise ForecastOptimizationError("blend weight search produced no candidate")
    selected = records[best_index]
    selected_weights = {
        name: float(selected[f"weight_{name}"]) for name in COMPONENT_FAMILIES
    }
    return selected_weights, pd.DataFrame(records), best_probability


@dataclass
class ProbabilityCalibrator:
    method: str
    model: Any = None

    @classmethod
    def fit(
        cls,
        method: str,
        probability: Sequence[float],
        target: Sequence[int],
        weights: Sequence[float],
    ) -> "ProbabilityCalibrator":
        if method not in CALIBRATION_METHODS:
            raise ForecastOptimizationError(f"unsupported calibrator: {method}")
        if method == "none":
            return cls(method)
        p = _clip(probability)
        y = np.asarray(target, dtype="int8")
        w = np.asarray(weights, dtype="float64")
        if np.unique(y).size != 2:
            raise ForecastOptimizationError("calibrator fit requires both classes")
        if method == "isotonic":
            model = IsotonicRegression(
                y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
            )
            model.fit(p, y, sample_weight=w)
        else:
            model = LogisticRegression(
                C=np.inf, solver="lbfgs", max_iter=5000, random_state=20260728
            )
            model.fit(_calibration_matrix(p, method), y, sample_weight=w)
        return cls(method, model)

    def predict(self, probability: Sequence[float]) -> np.ndarray:
        p = _clip(probability)
        if self.method == "none":
            return p
        if self.method == "isotonic":
            output = self.model.predict(p)
        else:
            output = self.model.predict_proba(_calibration_matrix(p, self.method))[:, 1]
        return _clip(output)

    def audit(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"method": self.method}
        if self.method == "isotonic":
            payload["x_thresholds"] = self.model.X_thresholds_.tolist()
            payload["y_thresholds"] = self.model.y_thresholds_.tolist()
        elif self.method != "none":
            payload["coef"] = self.model.coef_.ravel().tolist()
            payload["intercept"] = self.model.intercept_.ravel().tolist()
        return payload


def _clip(values: Sequence[float]) -> np.ndarray:
    return np.clip(np.asarray(values, dtype="float64"), EPSILON, 1.0 - EPSILON)


def _calibration_matrix(probability: np.ndarray, method: str) -> np.ndarray:
    if method == "platt":
        return np.log(probability / (1.0 - probability))[:, None]
    if method == "beta":
        return np.column_stack([np.log(probability), np.log1p(-probability)])
    raise ForecastOptimizationError(f"calibration matrix is undefined: {method}")


def compare_calibrators(
    frame: pd.DataFrame,
    blended_probability: Sequence[float],
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    source = frame.copy()
    source["p_blended"] = np.asarray(blended_probability, dtype="float64")
    weights = participant_equal_weights(source)
    target = source["future_binary_target"].to_numpy(dtype="int8")
    output = source[
        [
            "global_participant_id",
            "target_window_id",
            "future_binary_target",
            "outer_fold_id",
            "inner_fold_id",
            "p_blended",
        ]
    ].copy()
    records: list[dict[str, Any]] = []
    best_key: tuple[float, float, int] | None = None
    selected_method = ""
    for method_index, method in enumerate(CALIBRATION_METHODS):
        calibrated = np.full(len(source), np.nan, dtype="float64")
        audits: list[dict[str, Any]] = []
        for inner_fold in sorted(source["inner_fold_id"].astype(int).unique()):
            validation = source["inner_fold_id"].astype(int).eq(inner_fold).to_numpy()
            train = ~validation
            calibrator = ProbabilityCalibrator.fit(
                method,
                source.loc[train, "p_blended"],
                target[train],
                weights[train],
            )
            calibrated[validation] = calibrator.predict(
                source.loc[validation, "p_blended"]
            )
            audits.append(
                {
                    "inner_fold_id": int(inner_fold),
                    "fit_row_count": int(train.sum()),
                    "validation_row_count": int(validation.sum()),
                }
            )
        if not np.isfinite(calibrated).all():
            raise ForecastOptimizationError(f"calibration OOF is incomplete: {method}")
        output[f"p_calibrated_{method}"] = calibrated
        brier = float(np.average((calibrated - target) ** 2, weights=weights))
        ece = float(expected_calibration_error(target, calibrated, weights))
        records.append(
            {
                "method_index": method_index,
                "method": method,
                "brier": brier,
                "ece": ece,
                "fold_audit_json": json.dumps(audits, sort_keys=True),
            }
        )
        key = (brier, ece, method_index)
        if best_key is None or key < best_key:
            best_key = key
            selected_method = method
    output["p_calibrated_selected"] = output[f"p_calibrated_{selected_method}"]
    return selected_method, pd.DataFrame(records), output


def freeze_calibration_and_threshold(
    frame: pd.DataFrame,
    blended_probability: Sequence[float],
    selected_method: str,
    selected_crossfit_probability: Sequence[float],
) -> tuple[ProbabilityCalibrator, float, dict[str, Any]]:
    weights = participant_equal_weights(frame)
    target = frame["future_binary_target"].to_numpy(dtype="int8")
    calibrator = ProbabilityCalibrator.fit(
        selected_method, blended_probability, target, weights
    )
    threshold, audit = select_threshold(target, selected_crossfit_probability, weights)
    return calibrator, threshold, audit


__all__ = [
    "CALIBRATION_METHODS",
    "COMPONENT_FAMILIES",
    "ProbabilityCalibrator",
    "compare_calibrators",
    "component_oof_table",
    "freeze_calibration_and_threshold",
    "search_blend_weights",
    "select_family_components",
    "simplex_weight_grid",
]
