"""Strict monotonic calibrators used by FORECAST-OPT-005B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize

CLIP = 1e-6


def _clip(values: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(values), dtype="float64")
    return np.clip(result, CLIP, 1.0 - CLIP)


def _logit(values: Iterable[float]) -> np.ndarray:
    probability = _clip(values)
    return np.log(probability) - np.log1p(-probability)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype="float64")
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return np.clip(result, CLIP, 1.0 - CLIP)


def _weighted_log_loss(
    probability: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> float:
    p = np.clip(probability, CLIP, 1.0 - CLIP)
    normalized = weights / weights.sum()
    return float(
        -np.sum(normalized * (labels * np.log(p) + (1 - labels) * np.log1p(-p)))
    )


@dataclass(frozen=True)
class MonotonicCalibrator:
    """Serializable monotonic probability mapping."""

    method: str
    parameters: tuple[float, ...]

    def predict(self, score: Iterable[float]) -> np.ndarray:
        probability = _clip(score)
        logit = _logit(probability)
        if self.method == "temperature":
            (temperature,) = self.parameters
            result = _sigmoid(logit / temperature)
        elif self.method == "platt_positive":
            slope, intercept = self.parameters
            result = _sigmoid(slope * logit + intercept)
        elif self.method == "beta_monotonic":
            positive, negative, intercept = self.parameters
            result = _sigmoid(
                positive * np.log(probability)
                - negative * np.log1p(-probability)
                + intercept
            )
        else:
            raise ValueError(f"unsupported calibrator: {self.method}")
        if not np.isfinite(result).all():
            raise ValueError("calibrator produced a non-finite probability")
        return result

    def to_dict(self) -> dict[str, Any]:
        names = {
            "temperature": ("temperature",),
            "platt_positive": ("slope", "intercept"),
            "beta_monotonic": ("positive", "negative", "intercept"),
        }[self.method]
        return {
            "method": self.method,
            "parameters": dict(zip(names, self.parameters, strict=True)),
            "probability_clip": [CLIP, 1.0 - CLIP],
            "monotonic": True,
        }


def fit_calibrator(
    method: str,
    score: Iterable[float],
    labels: Iterable[int],
    weights: Iterable[float],
) -> MonotonicCalibrator:
    """Fit one frozen positive-slope calibrator with weighted log loss."""

    probability = _clip(score)
    y = np.asarray(list(labels), dtype="float64")
    w = np.asarray(list(weights), dtype="float64")
    if len(probability) != len(y) or len(y) != len(w) or len(y) == 0:
        raise ValueError("calibration inputs are empty or misaligned")
    if set(np.unique(y).tolist()) != {0.0, 1.0}:
        raise ValueError("calibration labels must contain both classes")
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("calibration weights are invalid")
    logit = _logit(probability)

    if method == "temperature":

        def objective(raw: np.ndarray) -> float:
            temperature = float(np.exp(np.clip(raw[0], -10.0, 10.0)))
            return _weighted_log_loss(_sigmoid(logit / temperature), y, w)

        fitted = minimize(objective, np.array([0.0]), method="L-BFGS-B")
        parameters = (float(np.exp(np.clip(fitted.x[0], -10.0, 10.0))),)
    elif method == "platt_positive":

        def objective(raw: np.ndarray) -> float:
            slope = float(np.exp(np.clip(raw[0], -10.0, 10.0)))
            return _weighted_log_loss(_sigmoid(slope * logit + raw[1]), y, w)

        fitted = minimize(objective, np.array([0.0, 0.0]), method="L-BFGS-B")
        parameters = (
            float(np.exp(np.clip(fitted.x[0], -10.0, 10.0))),
            float(fitted.x[1]),
        )
    elif method == "beta_monotonic":
        log_p = np.log(probability)
        log_one_minus_p = np.log1p(-probability)

        def objective(raw: np.ndarray) -> float:
            positive = float(np.exp(np.clip(raw[0], -10.0, 10.0)))
            negative = float(np.exp(np.clip(raw[1], -10.0, 10.0)))
            prediction = _sigmoid(
                positive * log_p - negative * log_one_minus_p + raw[2]
            )
            return _weighted_log_loss(prediction, y, w)

        fitted = minimize(objective, np.array([0.0, 0.0, 0.0]), method="L-BFGS-B")
        parameters = (
            float(np.exp(np.clip(fitted.x[0], -10.0, 10.0))),
            float(np.exp(np.clip(fitted.x[1], -10.0, 10.0))),
            float(fitted.x[2]),
        )
    else:
        raise ValueError(f"unsupported calibrator: {method}")
    if not fitted.success:
        raise RuntimeError(f"{method} optimization failed: {fitted.message}")
    calibrator = MonotonicCalibrator(method=method, parameters=parameters)
    audit_monotonic(calibrator)
    return calibrator


def audit_monotonic(calibrator: MonotonicCalibrator) -> dict[str, Any]:
    grid = np.linspace(CLIP, 1.0 - CLIP, 10_001)
    prediction = calibrator.predict(grid)
    differences = np.diff(prediction)
    return {
        "finite": bool(np.isfinite(prediction).all()),
        "within_probability_range": bool(((prediction >= 0) & (prediction <= 1)).all()),
        "nondecreasing": bool((differences >= -1e-12).all()),
        "minimum_difference": float(differences.min()),
    }


def calibration_slope_intercept(
    probability: Iterable[float], labels: Iterable[int], weights: Iterable[float]
) -> dict[str, float]:
    """Estimate weighted calibration slope/intercept without regularization."""

    calibrator = fit_calibrator("platt_positive", probability, labels, weights)
    slope, intercept = calibrator.parameters
    return {"slope": float(slope), "intercept": float(intercept)}


__all__ = [
    "MonotonicCalibrator",
    "audit_monotonic",
    "calibration_slope_intercept",
    "fit_calibrator",
]
