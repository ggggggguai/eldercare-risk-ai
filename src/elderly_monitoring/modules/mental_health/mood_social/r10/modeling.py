"""Finite R10 local-joint models and fold-local calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from elderly_monitoring.modules.mental_health.mood_social.r4.calibration import (
    FittedCalibration,
    fit_calibration,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    CandidateSpec,
    fit_model,
    predict_model,
)
from elderly_monitoring.modules.mental_health.mood_social.r7.modeling import binary_metrics
from elderly_monitoring.modules.mental_health.mood_social.r9.contract import (
    ACTIVITY_FEATURES,
    R9_HISTORY_FEATURES,
    SLEEP_FEATURES,
)
from elderly_monitoring.modules.mental_health.mood_social.r9.modeling import training_weights

from .features import (
    ACTIVITY_SLEEP_BASE_FEATURES,
    ACTIVITY_SLEEP_INTERACTION_FEATURES,
    SLEEP_HISTORY_FEATURES,
)


EPS = 1e-6
HEAD_TARGETS = {"ge5": "phq9_ge5_target", "ge10": "phq9_ge10_target"}
CALIBRATION_METHODS = ("platt", "beta", "restricted_isotonic")
A_S_CANDIDATES = (
    "r9_elastic_replay",
    "interaction_elastic",
    "spline_additive",
    "crossfit_logit_fusion",
    "ordinal_auxiliary",
)
S_H_CANDIDATES = (
    "r9_joint_elastic_replay",
    "residual_platt",
    "residual_beta",
    "residual_restricted_isotonic",
)


def logit(probability: Sequence[float]) -> np.ndarray:
    value = np.clip(np.asarray(probability, float), EPS, 1 - EPS)
    return np.log(value / (1.0 - value))


def expit(value: Sequence[float]) -> np.ndarray:
    raw = np.asarray(value, float)
    return np.clip(1.0 / (1.0 + np.exp(-raw)), EPS, 1 - EPS)


def participant_weight(frame: pd.DataFrame) -> np.ndarray:
    count = frame.groupby("global_participant_id")["global_participant_id"].transform("size")
    value = 1.0 / count.to_numpy(float)
    return value * len(value) / value.sum()


def _elastic(frame: pd.DataFrame, features: Sequence[str], target: str, seed: int) -> Any:
    work = frame.copy()
    work["binary_target"] = work[target].astype(int)
    spec = CandidateSpec(
        "r10_elastic", "elasticnet", {"C": 0.1, "l1_ratio": 0.2}, seed
    )
    return fit_model(
        work,
        features,
        spec,
        participant_equal=False,
        sample_weight=training_weights(work, "participant_equal"),
    )


def _spline(frame: pd.DataFrame, features: Sequence[str], target: str, seed: int) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("spline", SplineTransformer(n_knots=4, degree=2, include_bias=False)),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(features),
            )
        ],
        sparse_threshold=0.0,
    )
    model = Pipeline(
        [
            ("preprocess", preprocessing),
            ("model", LogisticRegression(C=0.05, max_iter=3000, random_state=seed)),
        ]
    )
    model.fit(
        frame[list(features)],
        frame[target].astype(int),
        model__sample_weight=participant_weight(frame),
    )
    return model


def _ordinal(frame: pd.DataFrame, features: Sequence[str], seed: int) -> Pipeline:
    category = frame["phq9_ge5_target"].astype(int) + frame["phq9_ge10_target"].astype(int)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=3000, random_state=seed)),
        ]
    )
    model.fit(frame[list(features)], category, model__sample_weight=participant_weight(frame))
    return model


def _ordinal_predict(model: Pipeline, frame: pd.DataFrame, features: Sequence[str], head: str) -> np.ndarray:
    probability = model.predict_proba(frame[list(features)])
    classes = model.named_steps["model"].classes_
    keep = classes >= (1 if head == "ge5" else 2)
    return np.clip(probability[:, keep].sum(axis=1), EPS, 1 - EPS)


@dataclass
class FittedCandidate:
    candidate_id: str
    head: str
    models: dict[str, Any]
    features: dict[str, tuple[str, ...]]
    meta: Any | None = None
    residual_intercept: float = 0.0
    residual_slope: float = 0.0
    residual_lambda: float = 0.0
    components: tuple["FittedCandidate", ...] = field(default_factory=tuple)
    blend_weight: float = 0.5

    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        if self.candidate_id in {"r9_elastic_replay", "interaction_elastic", "r9_joint_elastic_replay"}:
            return predict_model(self.models["main"], frame, self.features["main"])
        if self.candidate_id == "spline_additive":
            return np.clip(
                self.models["main"].predict_proba(frame[list(self.features["main"])])[:, 1],
                EPS,
                1 - EPS,
            )
        if self.candidate_id == "ordinal_auxiliary":
            return _ordinal_predict(self.models["main"], frame, self.features["main"], self.head)
        if self.candidate_id == "crossfit_logit_fusion":
            activity = predict_model(self.models["activity"], frame, self.features["activity"])
            sleep = predict_model(self.models["sleep"], frame, self.features["sleep"])
            return np.clip(
                self.meta.predict_proba(np.column_stack([logit(activity), logit(sleep)]))[:, 1],
                EPS,
                1 - EPS,
            )
        if self.candidate_id.startswith("residual_"):
            history = predict_model(self.models["history"], frame, self.features["history"])
            sleep = predict_model(self.models["sleep"], frame, self.features["sleep"])
            residual = logit(sleep) - (
                self.residual_intercept + self.residual_slope * logit(history)
            )
            return expit(logit(history) + self.residual_lambda * residual)
        if self.candidate_id.startswith("top2_blend"):
            first = self.components[0].predict_raw(frame)
            second = self.components[1].predict_raw(frame)
            return np.clip(
                self.blend_weight * first + (1.0 - self.blend_weight) * second,
                EPS,
                1 - EPS,
            )
        raise ValueError(f"unknown fitted R10 candidate: {self.candidate_id}")


def _fit_simple(
    frame: pd.DataFrame, candidate_id: str, head: str, seed: int
) -> FittedCandidate:
    target = HEAD_TARGETS[head]
    if candidate_id == "r9_elastic_replay":
        features = tuple(ACTIVITY_SLEEP_BASE_FEATURES)
        model = _elastic(frame, features, target, seed)
    elif candidate_id == "interaction_elastic":
        features = tuple(ACTIVITY_SLEEP_INTERACTION_FEATURES)
        model = _elastic(frame, features, target, seed)
    elif candidate_id == "spline_additive":
        features = tuple(ACTIVITY_SLEEP_BASE_FEATURES)
        model = _spline(frame, features, target, seed)
    elif candidate_id == "ordinal_auxiliary":
        features = tuple(ACTIVITY_SLEEP_INTERACTION_FEATURES)
        model = _ordinal(frame, features, seed)
    elif candidate_id == "r9_joint_elastic_replay":
        features = tuple(SLEEP_HISTORY_FEATURES)
        model = _elastic(frame, features, target, seed)
    else:
        raise ValueError(candidate_id)
    return FittedCandidate(candidate_id, head, {"main": model}, {"main": features})


def _fold_oof_simple(
    frame: pd.DataFrame,
    folds: pd.Series,
    candidate_id: str,
    head: str,
    seed: int,
) -> np.ndarray:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold in sorted(folds.astype(int).unique()):
        train = frame.loc[folds.ne(fold)]
        validation = frame.loc[folds.eq(fold)]
        fitted = _fit_simple(train, candidate_id, head, seed + int(fold))
        result.loc[validation.index] = fitted.predict_raw(validation)
    if result.isna().any():
        raise ValueError(f"incomplete R10 OOF for {candidate_id}/{head}")
    return result.loc[frame.index].to_numpy(float)


def _crossfit_logit_oof(
    frame: pd.DataFrame, folds: pd.Series, head: str, seed: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    target = HEAD_TARGETS[head]
    activity = pd.Series(np.nan, index=frame.index, dtype=float)
    sleep = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold in sorted(folds.astype(int).unique()):
        train = frame.loc[folds.ne(fold)]
        validation = frame.loc[folds.eq(fold)]
        a = _elastic(train, ACTIVITY_FEATURES, target, seed + int(fold))
        s = _elastic(train, SLEEP_FEATURES, target, seed + 20 + int(fold))
        activity.loc[validation.index] = predict_model(a, validation, ACTIVITY_FEATURES)
        sleep.loc[validation.index] = predict_model(s, validation, SLEEP_FEATURES)
    matrix = np.column_stack([logit(activity.loc[frame.index]), logit(sleep.loc[frame.index])])
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold in sorted(folds.astype(int).unique()):
        fit_mask = folds.ne(fold).to_numpy()
        val_mask = folds.eq(fold).to_numpy()
        meta = LogisticRegression(C=0.1, max_iter=3000, random_state=seed + 40 + int(fold))
        meta.fit(matrix[fit_mask], frame.loc[folds.ne(fold), target], sample_weight=participant_weight(frame.loc[folds.ne(fold)]))
        result.loc[folds.eq(fold)] = meta.predict_proba(matrix[val_mask])[:, 1]
    return result.loc[frame.index].to_numpy(float), {
        "activity": activity.loc[frame.index].to_numpy(float),
        "sleep": sleep.loc[frame.index].to_numpy(float),
    }


def _fit_crossfit_logit(
    frame: pd.DataFrame,
    folds: pd.Series,
    head: str,
    seed: int,
    base_probability: dict[str, np.ndarray] | None = None,
) -> FittedCandidate:
    target = HEAD_TARGETS[head]
    if base_probability is None:
        _, base_probability = _crossfit_logit_oof(frame, folds, head, seed)
    matrix = np.column_stack(
        [logit(base_probability["activity"]), logit(base_probability["sleep"])]
    )
    meta = LogisticRegression(C=0.1, max_iter=3000, random_state=seed + 80)
    meta.fit(matrix, frame[target], sample_weight=participant_weight(frame))
    return FittedCandidate(
        "crossfit_logit_fusion",
        head,
        {
            "activity": _elastic(frame, ACTIVITY_FEATURES, target, seed + 81),
            "sleep": _elastic(frame, SLEEP_FEATURES, target, seed + 82),
        },
        {"activity": tuple(ACTIVITY_FEATURES), "sleep": tuple(SLEEP_FEATURES)},
        meta=meta,
    )


def _residual_oof(
    frame: pd.DataFrame, folds: pd.Series, head: str, seed: int
) -> tuple[dict[float, np.ndarray], dict[str, Any]]:
    target = HEAD_TARGETS[head]
    history = pd.Series(np.nan, index=frame.index, dtype=float)
    sleep = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold in sorted(folds.astype(int).unique()):
        train = frame.loc[folds.ne(fold)]
        validation = frame.loc[folds.eq(fold)]
        h = _elastic(train, R9_HISTORY_FEATURES, target, seed + int(fold))
        s = _elastic(train, SLEEP_FEATURES, target, seed + 20 + int(fold))
        history.loc[validation.index] = predict_model(h, validation, R9_HISTORY_FEATURES)
        sleep.loc[validation.index] = predict_model(s, validation, SLEEP_FEATURES)
    regression = LinearRegression().fit(
        logit(history.loc[frame.index])[:, None], logit(sleep.loc[frame.index])
    )
    residual = logit(sleep.loc[frame.index]) - regression.predict(
        logit(history.loc[frame.index])[:, None]
    )
    candidates = {
        value: expit(logit(history.loc[frame.index]) + value * residual)
        for value in (0.0, 0.15, 0.30, 0.50, 0.75, 1.0)
    }
    return candidates, {
        "history": history.loc[frame.index].to_numpy(float),
        "sleep": sleep.loc[frame.index].to_numpy(float),
        "intercept": float(regression.intercept_),
        "slope": float(regression.coef_[0]),
    }


def _fit_residual(
    frame: pd.DataFrame,
    head: str,
    seed: int,
    residual_lambda: float,
    residual_parameters: dict[str, Any],
    candidate_id: str,
) -> FittedCandidate:
    target = HEAD_TARGETS[head]
    return FittedCandidate(
        candidate_id,
        head,
        {
            "history": _elastic(frame, R9_HISTORY_FEATURES, target, seed + 1),
            "sleep": _elastic(frame, SLEEP_FEATURES, target, seed + 2),
        },
        {"history": tuple(R9_HISTORY_FEATURES), "sleep": tuple(SLEEP_FEATURES)},
        residual_intercept=float(residual_parameters["intercept"]),
        residual_slope=float(residual_parameters["slope"]),
        residual_lambda=float(residual_lambda),
    )


def _crossfit_calibration(
    frame: pd.DataFrame,
    raw: np.ndarray,
    folds: pd.Series,
    method: str,
) -> np.ndarray:
    actual = "isotonic" if method == "restricted_isotonic" else method
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    raw_series = pd.Series(raw, index=frame.index)
    for fold in sorted(folds.astype(int).unique()):
        fit = frame.loc[folds.ne(fold)].copy()
        fit["binary_target"] = fit["_selection_target"].astype(int)
        calibration = fit_calibration(
            fit, raw_series.loc[folds.ne(fold)].to_numpy(float), actual
        )
        value = calibration.predict(raw_series.loc[folds.eq(fold)].to_numpy(float))
        if method == "restricted_isotonic":
            value = 0.85 * value + 0.15 * raw_series.loc[folds.eq(fold)].to_numpy(float)
        result.loc[folds.eq(fold)] = value
    if result.isna().any():
        raise ValueError("R10 cross-fit calibration incomplete")
    return result.loc[frame.index].to_numpy(float)


def select_calibration(
    frame: pd.DataFrame, raw: np.ndarray, folds: pd.Series, target: str
) -> tuple[str, dict[str, Any]]:
    work = frame.copy()
    work["_selection_target"] = work[target].astype(int)
    metrics: dict[str, Any] = {}
    for method in CALIBRATION_METHODS:
        probability = _crossfit_calibration(work, raw, folds, method)
        block = binary_metrics(work[target], probability, weight=participant_weight(work))
        metrics[method] = block
    selected = min(
        CALIBRATION_METHODS,
        key=lambda value: metrics[value]["brier"] + 0.25 * metrics[value]["ece"],
    )
    return selected, metrics


def fit_final_calibration(
    frame: pd.DataFrame, raw: np.ndarray, target: str, method: str
) -> FittedCalibration:
    work = frame.copy()
    work["binary_target"] = work[target].astype(int)
    actual = "isotonic" if method == "restricted_isotonic" else method
    return fit_calibration(work, raw, actual)


def apply_calibration(
    calibration: FittedCalibration, raw: np.ndarray, method: str
) -> np.ndarray:
    value = calibration.predict(raw)
    if method == "restricted_isotonic":
        value = 0.85 * value + 0.15 * np.asarray(raw, float)
    return np.clip(value, EPS, 1 - EPS)


def candidate_inner_predictions(
    node: str, frame: pd.DataFrame, folds: pd.Series, head: str, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    target = HEAD_TARGETS[head]
    probability: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    if node == "activity_sleep":
        for candidate in ("r9_elastic_replay", "interaction_elastic", "spline_additive", "ordinal_auxiliary"):
            probability[candidate] = _fold_oof_simple(frame, folds, candidate, head, seed)
            metadata[candidate] = {}
        logit_probability, base = _crossfit_logit_oof(frame, folds, head, seed + 100)
        probability["crossfit_logit_fusion"] = logit_probability
        metadata["crossfit_logit_fusion"] = {"base_probability": base}
        ranked = sorted(
            probability,
            key=lambda value: binary_metrics(
                frame[target], probability[value], weight=participant_weight(frame)
            )["auprc"],
            reverse=True,
        )
        first, second = ranked[:2]
        residual_correlation = float(
            np.corrcoef(frame[target] - probability[first], frame[target] - probability[second])[0, 1]
        )
        best_ap = binary_metrics(frame[target], probability[first], weight=participant_weight(frame))["auprc"]
        best_blend: tuple[float, float, np.ndarray] | None = None
        for weight in (0.25, 0.5, 0.75):
            blended = weight * probability[first] + (1 - weight) * probability[second]
            ap = binary_metrics(frame[target], blended, weight=participant_weight(frame))["auprc"]
            if best_blend is None or ap > best_blend[0]:
                best_blend = (ap, weight, blended)
        assert best_blend is not None
        if residual_correlation < 0.97 and best_blend[0] - best_ap >= 0.0015:
            candidate_id = f"top2_blend__{first}__{second}"
            probability[candidate_id] = best_blend[2]
            metadata[candidate_id] = {
                "components": [first, second],
                "weight_first": best_blend[1],
                "residual_correlation": residual_correlation,
                "inner_delta_auprc": best_blend[0] - best_ap,
            }
    elif node == "sleep_history":
        probability["r9_joint_elastic_replay"] = _fold_oof_simple(
            frame, folds, "r9_joint_elastic_replay", head, seed
        )
        metadata["r9_joint_elastic_replay"] = {}
        residual_candidates, residual_meta = _residual_oof(frame, folds, head, seed + 100)
        best_lambda = max(
            residual_candidates,
            key=lambda value: binary_metrics(
                frame[target], residual_candidates[value], weight=participant_weight(frame)
            )["auprc"],
        )
        for candidate, calibration in (
            ("residual_platt", "platt"),
            ("residual_beta", "beta"),
            ("residual_restricted_isotonic", "restricted_isotonic"),
        ):
            probability[candidate] = residual_candidates[best_lambda]
            metadata[candidate] = {
                "lambda": best_lambda,
                "residual_parameters": residual_meta,
                "forced_calibration": calibration,
            }
    else:
        raise ValueError(node)
    return probability, metadata


def fit_selected_candidate(
    node: str,
    frame: pd.DataFrame,
    folds: pd.Series,
    head: str,
    seed: int,
    selection: dict[str, Any],
    cached_metadata: dict[str, dict[str, Any]] | None = None,
) -> FittedCandidate:
    candidate = selection["candidate_id"]
    if candidate.startswith("top2_blend"):
        components = selection["metadata"]["components"]
        fitted = tuple(
            fit_selected_candidate(
                node,
                frame,
                folds,
                head,
                seed + index * 10,
                {"candidate_id": value, "metadata": {}},
                cached_metadata,
            )
            for index, value in enumerate(components)
        )
        return FittedCandidate(
            candidate,
            head,
            {},
            {},
            components=fitted,
            blend_weight=float(selection["metadata"]["weight_first"]),
        )
    if candidate == "crossfit_logit_fusion":
        base = None if cached_metadata is None else cached_metadata.get(candidate, {}).get("base_probability")
        return _fit_crossfit_logit(frame, folds, head, seed, base)
    if candidate.startswith("residual_"):
        metadata = selection["metadata"]
        return _fit_residual(
            frame,
            head,
            seed,
            float(metadata["lambda"]),
            metadata["residual_parameters"],
            candidate,
        )
    return _fit_simple(frame, candidate, head, seed)


__all__ = [
    "A_S_CANDIDATES",
    "CALIBRATION_METHODS",
    "FittedCandidate",
    "HEAD_TARGETS",
    "S_H_CANDIDATES",
    "candidate_inner_predictions",
    "apply_calibration",
    "fit_final_calibration",
    "fit_selected_candidate",
    "participant_weight",
    "select_calibration",
]
