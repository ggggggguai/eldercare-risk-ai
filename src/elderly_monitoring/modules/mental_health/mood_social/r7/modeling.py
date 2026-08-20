"""Shared participant-safe modeling primitives for independent R7 experts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from elderly_monitoring.modules.mental_health.mood_social.r4.calibration import (
    FittedCalibration,
    calibrate_outer_prediction,
    fit_calibration,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    CandidateSpec,
    fit_model,
    participant_equal_weights,
    predict_model,
)


HEAD_TARGETS = {"ge5": "phq9_ge5_target", "ge10": "phq9_ge10_target"}


def candidate_specs(seed: int) -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec("logistic_c01", "elasticnet", {"C": 0.1, "l1_ratio": 0.2}, seed),
        CandidateSpec("logistic_c05", "elasticnet", {"C": 0.5, "l1_ratio": 0.5}, seed),
        CandidateSpec("hist_leaf15", "hist_gradient", {"max_leaf_nodes": 15, "l2_regularization": 4.0, "max_iter": 200, "learning_rate": 0.04}, seed),
        CandidateSpec("lightgbm_leaf7", "lightgbm", {"n_estimators": 250, "learning_rate": 0.025, "num_leaves": 7, "min_child_samples": 30, "reg_lambda": 5.0, "subsample": 0.9, "colsample_bytree": 0.9}, seed),
        CandidateSpec("catboost_d3", "catboost", {"iterations": 250, "depth": 3, "learning_rate": 0.035, "l2_leaf_reg": 8.0}, seed),
    )


def inner_fold_series(frame: pd.DataFrame, outer_fold: int) -> pd.Series:
    key = str(int(outer_fold))
    def extract(value: object) -> float:
        parsed = json.loads(value) if isinstance(value, str) else dict(value)  # type: ignore[arg-type]
        result = parsed.get(key)
        return np.nan if result is None else float(result)
    result = frame["inner_validation_fold_by_outer_fold"].map(extract)
    is_test = frame["outer_fold"].eq(int(outer_fold))
    if result[is_test].notna().any() or result[~is_test].isna().any():
        raise ValueError(f"invalid R7 inner folds for outer {outer_fold}")
    return result


def _weighted_ece(target: np.ndarray, probability: np.ndarray, weight: np.ndarray | None = None) -> float:
    weights = np.ones(len(target), dtype=float) if weight is None else np.asarray(weight, float)
    groups = np.clip(np.digitize(probability, np.linspace(0.0, 1.0, 11)[1:-1]), 0, 9)
    total = float(weights.sum())
    value = 0.0
    for group in range(10):
        selected = groups == group
        if selected.any():
            mass = float(weights[selected].sum())
            value += mass / total * abs(float(np.average(target[selected], weights=weights[selected])) - float(np.average(probability[selected], weights=weights[selected])))
    return float(value)


def binary_metrics(target: Iterable[int], probability: Iterable[float], *, weight: Iterable[float] | None = None) -> dict[str, float]:
    y = np.asarray(list(target), int)
    p = np.asarray(list(probability), float)
    w = None if weight is None else np.asarray(list(weight), float)
    prevalence = float(np.average(y, weights=w))
    ap = float(average_precision_score(y, p, sample_weight=w))
    return {
        "auprc": ap,
        "prevalence": prevalence,
        "ap_lift": ap / max(prevalence, np.finfo(float).eps),
        "ap_gain": ap - prevalence,
        "normalized_ap": (ap - prevalence) / max(1.0 - prevalence, np.finfo(float).eps),
        "auroc": float(roc_auc_score(y, p, sample_weight=w)),
        "brier": float(np.average(np.square(p - y), weights=w)),
        "ece": _weighted_ece(y, p, w),
    }


def select_specificity_threshold(target: Sequence[int], probability: Sequence[float], specificity: float) -> float:
    y = np.asarray(target, int)
    p = np.asarray(probability, float)
    negative = p[y == 0]
    if not len(negative):
        return 1.0
    return float(np.clip(np.quantile(negative, specificity, method="higher"), 0.0, 1.0))


def _inner_oof(frame: pd.DataFrame, features: Sequence[str], folds: pd.Series, spec: CandidateSpec) -> np.ndarray:
    probability = pd.Series(np.nan, index=frame.index, dtype=float)
    for validation_fold in sorted(folds.astype(int).unique()):
        train = frame.loc[folds.ne(validation_fold)]
        validation = frame.loc[folds.eq(validation_fold)]
        model = fit_model(train, features, spec, participant_equal=True)
        probability.loc[validation.index] = predict_model(model, validation, features)
    if probability.isna().any():
        raise ValueError("R7 inner OOF is incomplete")
    return probability.loc[frame.index].to_numpy(float)


def fit_outer_head(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    folds: pd.Series,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    specs = candidate_specs(seed)
    scored: list[tuple[float, float, CandidateSpec, np.ndarray]] = []
    rows: list[dict[str, Any]] = []
    weight = participant_equal_weights(train)
    for spec in specs:
        raw = _inner_oof(train, features, folds, spec)
        metric = binary_metrics(train["binary_target"], raw, weight=weight)
        score = 0.6 * metric["auprc"] + 0.4 * metric["normalized_ap"]
        scored.append((score, metric["auprc"], spec, raw))
        rows.append({"spec": asdict(spec), "metrics": metric, "selection_score": score})
    scored.sort(key=lambda value: (value[0], value[1], value[2].candidate_id), reverse=True)
    _score, _ap, selected, selected_inner = scored[0]
    model = fit_model(train, features, selected, participant_equal=True)
    raw_outer = predict_model(model, test, features)
    calibrator_frame = train.copy()
    calibrator_frame["inner_fold"] = folds.astype(int)
    calibrated_outer, calibration = calibrate_outer_prediction(calibrator_frame, selected_inner, raw_outer)

    baseline = specs[0]
    baseline_inner = next(value[3] for value in scored if value[2].candidate_id == baseline.candidate_id)
    baseline_model = fit_model(train, features, baseline, participant_equal=True)
    baseline_raw_outer = predict_model(baseline_model, test, features)
    baseline_outer, baseline_calibration = calibrate_outer_prediction(calibrator_frame, baseline_inner, baseline_raw_outer)
    return calibrated_outer, baseline_outer, {
        "selected": asdict(selected),
        "candidates": rows,
        "calibration": calibration,
        "baseline": asdict(baseline),
        "baseline_calibration": baseline_calibration,
        "workpoints": {
            "specificity_080": select_specificity_threshold(train["binary_target"], selected_inner, 0.80),
            "specificity_090": select_specificity_threshold(train["binary_target"], selected_inner, 0.90),
        },
    }


def project_heads(probability_ge5: Sequence[float], probability_ge10: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    p5 = np.asarray(probability_ge5, float).copy()
    p10 = np.asarray(probability_ge10, float).copy()
    violated = p10 > p5
    midpoint = 0.5 * (p5[violated] + p10[violated])
    p5[violated] = midpoint
    p10[violated] = midpoint
    return np.clip(p5, 1e-6, 1 - 1e-6), np.clip(p10, 1e-6, 1 - 1e-6)


@dataclass
class FittedExpertBundle:
    domain: str
    features: tuple[str, ...]
    model_ge5: Any
    model_ge10: Any
    calibration_ge5: FittedCalibration
    calibration_ge10: FittedCalibration
    threshold_ge5: float
    threshold_ge10: float
    evidence_grade: str
    limitations: tuple[str, ...]
    model_version: str

    def predict_frame(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p5 = self.calibration_ge5.predict(predict_model(self.model_ge5, frame, self.features))
        p10 = self.calibration_ge10.predict(predict_model(self.model_ge10, frame, self.features))
        return project_heads(p5, p10)


def fit_full_bundle(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    domain: str,
    selected_specs: dict[str, CandidateSpec],
    seed: int,
    evidence_grade: str,
    limitations: Sequence[str],
) -> FittedExpertBundle:
    models: dict[str, Any] = {}
    calibrators: dict[str, FittedCalibration] = {}
    thresholds: dict[str, float] = {}
    folds = pd.Series(np.arange(len(frame)) % 5, index=frame.index)
    for head, target in HEAD_TARGETS.items():
        work = frame.copy()
        work["binary_target"] = work[target].astype(int)
        spec = selected_specs[head]
        oof = _inner_oof(work, features, folds, spec)
        calibrator = fit_calibration(work, oof, "platt")
        models[head] = fit_model(work, features, spec, participant_equal=True)
        calibrators[head] = calibrator
        thresholds[head] = select_specificity_threshold(work["binary_target"], calibrator.predict(oof), 0.80)
    return FittedExpertBundle(
        domain=domain,
        features=tuple(features),
        model_ge5=models["ge5"],
        model_ge10=models["ge10"],
        calibration_ge5=calibrators["ge5"],
        calibration_ge10=calibrators["ge10"],
        threshold_ge5=thresholds["ge5"],
        threshold_ge10=thresholds["ge10"],
        evidence_grade=evidence_grade,
        limitations=tuple(limitations),
        model_version=f"{domain}-expert-r7",
    )


def write_json(path: Path, payload: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


__all__ = [
    "FittedExpertBundle", "HEAD_TARGETS", "binary_metrics", "candidate_specs",
    "fit_full_bundle", "fit_outer_head", "inner_fold_series", "participant_equal_weights",
    "project_heads", "select_specificity_threshold", "write_json",
]
