"""Inner-only finite model competition and constrained convex blending."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import (
    ap_context_metrics,
)


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: Literal["elasticnet", "hist_gradient", "lightgbm", "catboost"]
    params: dict[str, Any]
    seed: int = 20260812

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def screening_candidates(seed: int = 20260812) -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec("elasticnet_c01_l020", "elasticnet", {"C": 0.1, "l1_ratio": 0.2}, seed),
        CandidateSpec("elasticnet_c05_l050", "elasticnet", {"C": 0.5, "l1_ratio": 0.5}, seed),
        CandidateSpec("hist_leaf15_l2", "hist_gradient", {"max_leaf_nodes": 15, "l2_regularization": 2.0, "max_iter": 250, "learning_rate": 0.05}, seed),
        CandidateSpec("hist_leaf31_l5", "hist_gradient", {"max_leaf_nodes": 31, "l2_regularization": 5.0, "max_iter": 250, "learning_rate": 0.04}, seed),
        CandidateSpec("lightgbm_leaf7", "lightgbm", {"n_estimators": 300, "learning_rate": 0.025, "num_leaves": 7, "min_child_samples": 30, "reg_lambda": 3.0, "subsample": 0.9, "colsample_bytree": 0.9}, seed),
        CandidateSpec("lightgbm_leaf15", "lightgbm", {"n_estimators": 350, "learning_rate": 0.025, "num_leaves": 15, "min_child_samples": 35, "reg_lambda": 5.0, "subsample": 0.9, "colsample_bytree": 0.9}, seed),
        CandidateSpec("catboost_d3", "catboost", {"iterations": 300, "depth": 3, "learning_rate": 0.035, "l2_leaf_reg": 8.0}, seed),
        CandidateSpec("catboost_d5", "catboost", {"iterations": 350, "depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 12.0}, seed),
    )


def participant_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("global_participant_id")["global_participant_id"].transform("size")
    weight = 1.0 / counts.to_numpy(float)
    return weight * len(weight) / weight.sum()


def _column_groups(frame: pd.DataFrame, features: Sequence[str]) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for feature in features:
        if pd.api.types.is_numeric_dtype(frame[feature].dtype):
            numeric.append(feature)
        else:
            categorical.append(feature)
    return numeric, categorical


def _pipeline(frame: pd.DataFrame, features: Sequence[str], spec: CandidateSpec) -> Pipeline:
    numeric, categorical = _column_groups(frame, features)
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        numeric_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
        ]
        if spec.family == "elasticnet":
            numeric_steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    preprocess = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    if spec.family == "elasticnet":
        model: Any = LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            max_iter=5000,
            random_state=spec.seed,
            n_jobs=-1,
            **spec.params,
        )
    elif spec.family == "hist_gradient":
        model = HistGradientBoostingClassifier(random_state=spec.seed, **spec.params)
    elif spec.family == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            objective="binary",
            random_state=spec.seed,
            n_jobs=-1,
            verbosity=-1,
            **spec.params,
        )
    elif spec.family == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            loss_function="Logloss",
            random_seed=spec.seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            **spec.params,
        )
    else:  # pragma: no cover - Literal protects callers
        raise ValueError(f"unknown r4 model family: {spec.family}")
    return Pipeline([("preprocess", preprocess), ("model", model)])


def fit_model(
    frame: pd.DataFrame,
    features: Sequence[str],
    spec: CandidateSpec,
    *,
    participant_equal: bool,
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
    if frame["binary_target"].nunique() < 2:
        raise ValueError("r4 model fit requires both classes")
    model = _pipeline(frame, features, spec)
    if sample_weight is not None:
        weight = np.asarray(sample_weight, dtype=float)
        if weight.shape != (len(frame),) or not np.isfinite(weight).all() or (weight <= 0).any():
            raise ValueError("r4 custom sample weights must be finite, positive and row-aligned")
        weight = weight * len(weight) / weight.sum()
    else:
        weight = participant_equal_weights(frame) if participant_equal else np.ones(len(frame))
    model.fit(frame[list(features)], frame["binary_target"].astype(int), model__sample_weight=weight)
    return model


def predict_model(model: Pipeline, frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    return np.clip(model.predict_proba(frame[list(features)])[:, 1], 1.0e-6, 1.0 - 1.0e-6)


def _selection_metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    target = frame["binary_target"].to_numpy(int)
    natural = ap_context_metrics(target, probability)
    participant = ap_context_metrics(
        target, probability, sample_weight=participant_equal_weights(frame)
    )
    return {
        "natural_auprc": natural["auprc"],
        "natural_auroc": natural["auroc"],
        "natural_brier": natural["brier"],
        "participant_auprc": participant["auprc"],
        "participant_auroc": participant["auroc"],
        "participant_brier": participant["brier"],
        "selection_score": 0.6 * natural["auprc"] + 0.4 * participant["auprc"],
    }


def candidate_inner_oof(
    frame: pd.DataFrame,
    features: Sequence[str],
    inner_fold: pd.Series,
    spec: CandidateSpec,
    *,
    participant_equal: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    fold = pd.Series(inner_fold, index=frame.index).astype(int)
    probability = pd.Series(np.nan, index=frame.index, dtype=float)
    for validation_fold in sorted(fold.unique()):
        train = frame.loc[fold.ne(validation_fold)]
        validation = frame.loc[fold.eq(validation_fold)]
        model = fit_model(train, features, spec, participant_equal=participant_equal)
        probability.loc[validation.index] = predict_model(model, validation, features)
    if probability.isna().any():
        raise ValueError("r4 candidate inner OOF is incomplete")
    values = probability.loc[frame.index].to_numpy(float)
    return values, _selection_metrics(frame, values)


def select_convex_blend(
    frame: pd.DataFrame,
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        probability = weight * np.asarray(first) + (1.0 - weight) * np.asarray(second)
        metrics = _selection_metrics(frame, probability)
        candidate = {"weight_first": weight, "probability": probability, **metrics}
        if best is None or (
            metrics["selection_score"], -abs(weight - 0.5)
        ) > (best["selection_score"], -abs(best["weight_first"] - 0.5)):
            best = candidate
    assert best is not None
    return best


def select_and_fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    inner_fold: pd.Series,
    *,
    participant_equal: bool,
    seed: int = 20260812,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    predictions: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    specs = screening_candidates(seed)
    for spec in specs:
        probability, metrics = candidate_inner_oof(
            train,
            features,
            inner_fold,
            spec,
            participant_equal=participant_equal,
        )
        predictions[spec.candidate_id] = probability
        rows.append({**spec.to_dict(), **metrics, "status": "pass"})
    ranked = sorted(rows, key=lambda row: (row["selection_score"], row["candidate_id"]), reverse=True)
    first_row = ranked[0]
    second_row = next(row for row in ranked[1:] if row["family"] != first_row["family"])
    first_spec = next(spec for spec in specs if spec.candidate_id == first_row["candidate_id"])
    second_spec = next(spec for spec in specs if spec.candidate_id == second_row["candidate_id"])
    blend = select_convex_blend(
        train,
        predictions[first_spec.candidate_id],
        predictions[second_spec.candidate_id],
    )
    model_first = fit_model(train, features, first_spec, participant_equal=participant_equal)
    model_second = fit_model(train, features, second_spec, participant_equal=participant_equal)
    first_test = predict_model(model_first, test, features)
    second_test = predict_model(model_second, test, features)
    weight = float(blend["weight_first"])
    test_probability = weight * first_test + (1.0 - weight) * second_test
    selection = {
        "first": first_spec.to_dict(),
        "second": second_spec.to_dict(),
        "weight_first": weight,
        "inner_selection_score": float(blend["selection_score"]),
        "feature_count": len(features),
    }
    rows.append(
        {
            "candidate_id": f"convex__{first_spec.candidate_id}__{second_spec.candidate_id}",
            "family": "convex_blend",
            "params": {"weight_first": weight},
            "seed": seed,
            **{key: value for key, value in blend.items() if key != "probability"},
            "status": "selected",
        }
    )
    return (
        np.clip(test_probability, 1.0e-6, 1.0 - 1.0e-6),
        np.asarray(blend["probability"], dtype=float),
        rows,
        selection,
    )


def enforce_dual_head_monotonicity(
    probability_ge5: np.ndarray, probability_ge10: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    early = np.asarray(probability_ge5, dtype=float).copy()
    elevated = np.asarray(probability_ge10, dtype=float).copy()
    if early.shape != elevated.shape:
        raise ValueError("dual-head probability shapes differ")
    violated = elevated > early
    midpoint = 0.5 * (early[violated] + elevated[violated])
    early[violated] = midpoint
    elevated[violated] = midpoint
    return np.clip(early, 1.0e-6, 1.0 - 1.0e-6), np.clip(
        elevated, 1.0e-6, 1.0 - 1.0e-6
    )


__all__ = [
    "CandidateSpec",
    "candidate_inner_oof",
    "enforce_dual_head_monotonicity",
    "fit_model",
    "participant_equal_weights",
    "predict_model",
    "screening_candidates",
    "select_and_fit_predict",
    "select_convex_blend",
]
