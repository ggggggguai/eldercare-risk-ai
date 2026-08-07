"""Frozen V3.4 binary metrics, aggregation, calibration and workpoint rules."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score


class V34MetricError(RuntimeError):
    pass


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def expected_calibration_error(
    labels: Sequence[int], probabilities: Sequence[float], *, bins: int = 10
) -> float:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if len(y) != len(p) or not len(y):
        raise V34MetricError("ECE requires equally sized nonempty labels and probabilities")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise V34MetricError("ECE probabilities must be finite and inside [0, 1]")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = float(len(y))
    ece = 0.0
    for index in range(int(bins)):
        if index + 1 == int(bins):
            mask = (p >= edges[index]) & (p <= edges[index + 1])
        else:
            mask = (p >= edges[index]) & (p < edges[index + 1])
        if not bool(mask.any()):
            continue
        ece += float(mask.sum()) / total * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def binary_metrics(
    labels: Sequence[int], probabilities: Sequence[float], *, threshold: float = 0.5
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if len(y) != len(p) or not len(y):
        raise V34MetricError("binary metrics require equally sized nonempty inputs")
    if set(y.tolist()) != {0, 1}:
        raise V34MetricError("binary metrics require both classes")
    predicted = (p >= float(threshold)).astype(np.int64)
    tp = int(((predicted == 1) & (y == 1)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "count": len(y),
        "positive_count": int(y.sum()),
        "negative_count": int((1 - y).sum()),
        "roc_auc": float(roc_auc_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "brier": float(brier_score_loss(y, p)),
        "ece_10_equal_width": expected_calibration_error(y, p, bins=10),
        "threshold": float(threshold),
        "confusion_matrix": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
    }


def fit_platt(raw_logits: Sequence[float], labels: Sequence[int]) -> dict[str, float]:
    x = np.asarray(raw_logits, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(labels, dtype=np.int64)
    if len(x) != len(y) or not len(y) or set(y.tolist()) != {0, 1}:
        raise V34MetricError("Platt calibration requires nonempty two-class inputs")
    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=1e6,
        max_iter=1000,
        fit_intercept=True,
    )
    model.fit(x, y)
    return {"a": float(model.coef_[0, 0]), "b": float(model.intercept_[0])}


def apply_platt(raw_logits: Sequence[float], parameters: Mapping[str, float]) -> list[float]:
    a = float(parameters["a"])
    b = float(parameters["b"])
    return [sigmoid(a * float(value) + b) for value in raw_logits]


def fit_temperature(raw_logits: Sequence[float], labels: Sequence[int]) -> dict[str, float]:
    z = np.asarray(raw_logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if len(z) != len(y) or not len(y) or set(y.tolist()) != {0, 1}:
        raise V34MetricError("temperature calibration requires nonempty two-class inputs")

    def objective(temperature: float) -> float:
        scaled = z / float(temperature)
        return float(np.mean(np.logaddexp(0.0, scaled) - y * scaled))

    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=(0.05, 10.0),
        options={"xatol": 1e-8},
    )
    if not result.success or not math.isfinite(float(result.x)):
        raise V34MetricError("temperature optimization failed")
    return {"temperature": float(result.x)}


def apply_temperature(
    raw_logits: Sequence[float], parameters: Mapping[str, float]
) -> list[float]:
    temperature = float(parameters["temperature"])
    if not 0.05 <= temperature <= 10.0:
        raise V34MetricError("temperature is outside the frozen bounds")
    return [sigmoid(float(value) / temperature) for value in raw_logits]


def fit_isotonic(raw_logits: Sequence[float], labels: Sequence[int]) -> dict[str, list[float]]:
    z = np.asarray(raw_logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if len(z) != len(y) or not len(y) or set(y.tolist()) != {0, 1}:
        raise V34MetricError("isotonic calibration requires nonempty two-class inputs")
    model = IsotonicRegression(
        increasing=True,
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )
    model.fit(z, y)
    return {
        "x_thresholds": [float(value) for value in model.X_thresholds_],
        "y_thresholds": [float(value) for value in model.y_thresholds_],
    }


def apply_isotonic(
    raw_logits: Sequence[float], parameters: Mapping[str, Sequence[float]]
) -> list[float]:
    x = np.asarray(parameters["x_thresholds"], dtype=np.float64)
    y = np.asarray(parameters["y_thresholds"], dtype=np.float64)
    if len(x) != len(y) or not len(x) or np.any(np.diff(x) < 0.0):
        raise V34MetricError("invalid isotonic thresholds")
    values = np.asarray(raw_logits, dtype=np.float64)
    return np.interp(values, x, y, left=y[0], right=y[-1]).astype(float).tolist()


def fit_calibrator(
    method: str, raw_logits: Sequence[float], labels: Sequence[int]
) -> dict[str, Any]:
    if method == "platt":
        return fit_platt(raw_logits, labels)
    if method == "temperature":
        return fit_temperature(raw_logits, labels)
    if method == "isotonic":
        return fit_isotonic(raw_logits, labels)
    raise V34MetricError(f"unsupported calibration method: {method}")


def apply_calibrator(
    method: str, raw_logits: Sequence[float], parameters: Mapping[str, Any]
) -> list[float]:
    if method == "platt":
        return apply_platt(raw_logits, parameters)
    if method == "temperature":
        return apply_temperature(raw_logits, parameters)
    if method == "isotonic":
        return apply_isotonic(raw_logits, parameters)
    raise V34MetricError(f"unsupported calibration method: {method}")


def aggregate_subjects(
    rows: Iterable[Mapping[str, Any]],
    *,
    probability_field: str | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("raw_logit") is None:
            continue
        grouped[str(row["subject_id"])].append(row)
    result: list[dict[str, Any]] = []
    for subject_id, subject_rows in sorted(grouped.items()):
        labels = {int(row["label_hc_vs_non_hc"]) for row in subject_rows}
        diagnoses = {str(row["diagnosis"]) for row in subject_rows}
        if len(labels) != 1 or len(diagnoses) != 1:
            raise V34MetricError(f"subject label conflict: {subject_id}")
        mean_logit = float(np.mean([float(row["raw_logit"]) for row in subject_rows]))
        if probability_field is None:
            probability = sigmoid(mean_logit)
            aggregation = "mean_task_raw_logit_then_sigmoid"
        else:
            values = [row.get(probability_field) for row in subject_rows]
            if any(value is None for value in values):
                raise V34MetricError(f"missing calibrated probability for {subject_id}")
            probability = float(np.mean([float(value) for value in values]))
            aggregation = "mean_task_calibrated_probability"
        result.append(
            {
                "subject_id": subject_id,
                "diagnosis": next(iter(diagnoses)),
                "label_hc_vs_non_hc": next(iter(labels)),
                "task_count": len(subject_rows),
                "mean_raw_logit": mean_logit,
                "probability": probability,
                "aggregation": aggregation,
            }
        )
    if not result:
        raise V34MetricError("subject aggregation has no scored rows")
    return result


def select_workpoint(
    subject_rows: Sequence[Mapping[str, Any]], *, target_sensitivity: float = 0.80
) -> dict[str, Any]:
    labels = [int(row["label_hc_vs_non_hc"]) for row in subject_rows]
    probabilities = [float(row["probability"]) for row in subject_rows]
    candidates: list[dict[str, Any]] = []
    for step in range(1001):
        threshold = step / 1000.0
        metrics = binary_metrics(labels, probabilities, threshold=threshold)
        if metrics["sensitivity"] + 1e-12 >= float(target_sensitivity):
            candidates.append(metrics)
    if not candidates:
        raise V34MetricError("no threshold satisfies target sensitivity")
    candidates.sort(
        key=lambda row: (
            float(row["specificity"]),
            float(row["threshold"]),
        ),
        reverse=True,
    )
    return candidates[0]


def temporary_platt_evaluation(
    calibration_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    calibration_valid = [row for row in calibration_rows if row.get("raw_logit") is not None]
    evaluation_valid = [row for row in evaluation_rows if row.get("raw_logit") is not None]
    parameters = fit_platt(
        [float(row["raw_logit"]) for row in calibration_valid],
        [int(row["label_hc_vs_non_hc"]) for row in calibration_valid],
    )
    calibration_probabilities = apply_platt(
        [float(row["raw_logit"]) for row in calibration_valid], parameters
    )
    evaluation_probabilities = apply_platt(
        [float(row["raw_logit"]) for row in evaluation_valid], parameters
    )
    calibration_enriched = [
        {**row, "temporary_calibrated_probability": probability}
        for row, probability in zip(calibration_valid, calibration_probabilities)
    ]
    evaluation_enriched = [
        {**row, "temporary_calibrated_probability": probability}
        for row, probability in zip(evaluation_valid, evaluation_probabilities)
    ]
    calibration_subjects = aggregate_subjects(
        calibration_enriched, probability_field="temporary_calibrated_probability"
    )
    workpoint = select_workpoint(calibration_subjects)
    evaluation_subjects = aggregate_subjects(
        evaluation_enriched, probability_field="temporary_calibrated_probability"
    )
    task_metrics = binary_metrics(
        [int(row["label_hc_vs_non_hc"]) for row in evaluation_enriched],
        [float(row["temporary_calibrated_probability"]) for row in evaluation_enriched],
        threshold=float(workpoint["threshold"]),
    )
    subject_metrics = binary_metrics(
        [int(row["label_hc_vs_non_hc"]) for row in evaluation_subjects],
        [float(row["probability"]) for row in evaluation_subjects],
        threshold=float(workpoint["threshold"]),
    )
    false_positive_subjects = sorted(
        row["subject_id"]
        for row in evaluation_subjects
        if int(row["label_hc_vs_non_hc"]) == 0
        and float(row["probability"]) >= float(workpoint["threshold"])
    )
    return {
        "method": "platt",
        "parameters": parameters,
        "fit_scope": "inner_calibration_task_raw_logits",
        "subject_probability_aggregation": "calibrate_each_task_then_mean_probability",
        "workpoint_selection_scope": "inner_calibration_subject_probabilities",
        "workpoint": workpoint,
        "outer_evaluation_task_metrics": task_metrics,
        "outer_evaluation_subject_metrics": subject_metrics,
        "false_positive_subjects": false_positive_subjects,
    }


__all__ = [
    "V34MetricError",
    "aggregate_subjects",
    "apply_calibrator",
    "apply_isotonic",
    "apply_platt",
    "apply_temperature",
    "binary_metrics",
    "expected_calibration_error",
    "fit_calibrator",
    "fit_isotonic",
    "fit_platt",
    "fit_temperature",
    "select_workpoint",
    "sigmoid",
    "temporary_platt_evaluation",
]
