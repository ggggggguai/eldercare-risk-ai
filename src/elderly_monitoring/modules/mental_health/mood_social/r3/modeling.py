"""Deterministic tabular model and sample-weight factories for r3."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from elderly_monitoring.modules.mental_health.mood_social.r3.contract import (
    assert_deployable_feature_names,
)


class CandidateUnavailable(RuntimeError):
    """Raised when a preregistered optional model family is unavailable."""


@dataclass(frozen=True)
class ModelSpec:
    candidate_id: str
    family: Literal[
        "logistic",
        "elasticnet_logistic",
        "hist_gradient",
        "lightgbm",
        "catboost",
        "xgboost",
    ]
    params: dict[str, Any]
    seed: int = 20260728

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WeightSpec:
    weight_id: str
    participant_equal: bool
    source_alpha: float = 0.0
    class_mode: Literal["none", "sqrt_balanced", "capped_balanced"] = "none"
    relative_cap: float = 10.0

    def __post_init__(self) -> None:
        if self.source_alpha not in {0.0, 0.25, 0.5}:
            raise ValueError("source_alpha must be 0, 0.25 or 0.5")
        if self.relative_cap not in {5.0, 10.0}:
            raise ValueError("relative_cap must be 5 or 10")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_WEIGHT_SPECS: tuple[WeightSpec, ...] = (
    WeightSpec("natural", participant_equal=False),
    WeightSpec("participant", participant_equal=True),
    WeightSpec(
        "participant_source025", participant_equal=True, source_alpha=0.25
    ),
    WeightSpec(
        "participant_source050", participant_equal=True, source_alpha=0.5
    ),
    WeightSpec(
        "participant_sqrt_class",
        participant_equal=True,
        class_mode="sqrt_balanced",
    ),
    WeightSpec(
        "participant_source025_sqrt_class",
        participant_equal=True,
        source_alpha=0.25,
        class_mode="sqrt_balanced",
    ),
    WeightSpec(
        "participant_source050_capped_class",
        participant_equal=True,
        source_alpha=0.5,
        class_mode="capped_balanced",
    ),
)


def default_model_specs() -> tuple[ModelSpec, ...]:
    seeds = (20260728, 20260729, 20260730)
    specs: list[ModelSpec] = [
        ModelSpec("logistic_c01", "logistic", {"C": 0.1}),
        ModelSpec("logistic_c1", "logistic", {"C": 1.0}),
        ModelSpec("logistic_c10", "logistic", {"C": 10.0}),
        ModelSpec(
            "elasticnet_c1_l020",
            "elasticnet_logistic",
            {"C": 1.0, "l1_ratio": 0.2},
        ),
        ModelSpec(
            "hist_leaf15_l2",
            "hist_gradient",
            {"max_leaf_nodes": 15, "l2_regularization": 1.0},
        ),
        ModelSpec(
            "hist_leaf31_l5",
            "hist_gradient",
            {"max_leaf_nodes": 31, "l2_regularization": 5.0},
        ),
    ]
    for seed in seeds:
        specs.extend(
            [
                ModelSpec(
                    f"lightgbm_leaf15_s{seed}",
                    "lightgbm",
                    {
                        "n_estimators": 350,
                        "learning_rate": 0.03,
                        "num_leaves": 15,
                        "min_child_samples": 30,
                        "reg_lambda": 1.0,
                    },
                    seed,
                ),
                ModelSpec(
                    f"catboost_d4_s{seed}",
                    "catboost",
                    {
                        "iterations": 350,
                        "depth": 4,
                        "learning_rate": 0.035,
                        "l2_leaf_reg": 10.0,
                    },
                    seed,
                ),
                ModelSpec(
                    f"xgboost_d3_s{seed}",
                    "xgboost",
                    {
                        "n_estimators": 350,
                        "max_depth": 3,
                        "learning_rate": 0.03,
                        "min_child_weight": 5.0,
                        "subsample": 0.8,
                        "colsample_bytree": 0.9,
                        "reg_lambda": 5.0,
                    },
                    seed,
                ),
            ]
        )
    return tuple(specs)


LEGACY_ACTIVITY_SPEC = ModelSpec(
    "legacy_lightgbm_activity",
    "lightgbm",
    {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 20,
        "reg_alpha": 0.0,
        "reg_lambda": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
    },
)
LEGACY_SLEEP_SPEC = ModelSpec(
    "legacy_lightgbm_sleep", "lightgbm", dict(LEGACY_ACTIVITY_SPEC.params)
)
LEGACY_JOINT_SPEC = ModelSpec(
    "legacy_lightgbm_joint", "lightgbm", dict(LEGACY_ACTIVITY_SPEC.params)
)
LEGACY_PROFILE_SPEC = ModelSpec(
    "legacy_catboost_profile",
    "catboost",
    {"iterations": 300, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 10.0},
)


def sample_weights(frame: pd.DataFrame, spec: WeightSpec) -> np.ndarray:
    """Build independent participant/source/class weights with a hard cap."""

    if not len(frame):
        return np.empty(0, dtype=float)
    natural = np.ones(len(frame), dtype=float)
    weight = natural.copy()
    if spec.participant_equal:
        counts = frame.groupby("global_participant_id")["global_participant_id"].transform(
            "size"
        )
        weight /= counts.to_numpy(dtype=float)
        weight *= len(frame) / weight.sum()
    if spec.source_alpha > 0.0:
        sizes = frame.groupby("dataset_id")["dataset_id"].transform("size").to_numpy(
            dtype=float
        )
        source_factor = np.power(sizes, -spec.source_alpha)
        source_factor /= np.average(source_factor, weights=weight)
        weight *= source_factor
    target = frame["binary_target"].to_numpy(dtype=int)
    if spec.class_mode != "none":
        counts = np.bincount(target, minlength=2).astype(float)
        if np.any(counts == 0):
            raise ValueError("class weighting requires both classes")
        balanced = len(target) / (2.0 * counts)
        class_factor = (
            np.sqrt(balanced)
            if spec.class_mode == "sqrt_balanced"
            else np.minimum(balanced, 5.0)
        )
        weight *= class_factor[target]
    reference = natural if not spec.participant_equal else np.maximum(
        natural / frame.groupby("global_participant_id")["global_participant_id"]
        .transform("size")
        .to_numpy(dtype=float),
        np.finfo(float).eps,
    )
    ratio = weight / reference
    weight = np.where(ratio > spec.relative_cap, reference * spec.relative_cap, weight)
    weight *= len(weight) / weight.sum()
    if not np.isfinite(weight).all() or np.any(weight <= 0.0):
        raise ValueError("invalid r3 sample weights")
    return weight


def _column_groups(frame: pd.DataFrame, features: Sequence[str]) -> tuple[list[str], list[str]]:
    categorical: list[str] = []
    numeric: list[str] = []
    for feature in features:
        if pd.api.types.is_numeric_dtype(frame[feature].dtype):
            numeric.append(feature)
        else:
            categorical.append(feature)
    return numeric, categorical


def _preprocessor(frame: pd.DataFrame, features: Sequence[str]) -> ColumnTransformer:
    numeric, categorical = _column_groups(frame, features)
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent", keep_empty_features=True
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def _classifier(spec: ModelSpec) -> Any:
    params = dict(spec.params)
    if spec.family == "logistic":
        return LogisticRegression(
            solver="liblinear", max_iter=3000, random_state=spec.seed, **params
        )
    if spec.family == "elasticnet_logistic":
        return LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            max_iter=4000,
            random_state=spec.seed,
            **params,
        )
    if spec.family == "hist_gradient":
        return HistGradientBoostingClassifier(random_state=spec.seed, **params)
    if spec.family == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            random_state=spec.seed,
            n_jobs=-1,
            verbosity=-1,
            objective="binary",
            **params,
        )
    if spec.family == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            random_seed=spec.seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            loss_function="Logloss",
            **params,
        )
    if spec.family == "xgboost":
        if importlib.util.find_spec("xgboost") is None:
            raise CandidateUnavailable("xgboost is not installed in the formal environment")
        from xgboost import XGBClassifier

        return XGBClassifier(
            random_state=spec.seed,
            n_jobs=-1,
            objective="binary:logistic",
            eval_metric="logloss",
            **params,
        )
    raise ValueError(f"unsupported model family: {spec.family}")


def build_classifier_pipeline(
    frame: pd.DataFrame,
    features: Sequence[str],
    spec: ModelSpec,
) -> Pipeline:
    names = assert_deployable_feature_names(features)
    missing = sorted(set(names).difference(frame.columns))
    if missing:
        raise ValueError(f"training frame missing model features: {missing}")
    return Pipeline(
        [
            ("preprocess", _preprocessor(frame, names)),
            ("model", _classifier(spec)),
        ]
    )


def fit_classifier(
    train: pd.DataFrame,
    features: Sequence[str],
    spec: ModelSpec,
    weight_spec: WeightSpec,
) -> Pipeline:
    model = build_classifier_pipeline(train, features, spec)
    weight = sample_weights(train, weight_spec)
    model.fit(train[list(features)], train["binary_target"].astype(int), model__sample_weight=weight)
    return model


def fit_classifier_target(
    train: pd.DataFrame,
    features: Sequence[str],
    spec: ModelSpec,
    weight_spec: WeightSpec,
    target_column: str,
) -> Pipeline:
    if target_column not in train.columns:
        raise ValueError(f"training frame missing target column: {target_column}")
    work = train.copy()
    work["binary_target"] = pd.to_numeric(
        work[target_column], errors="raise"
    ).astype(int)
    if not work["binary_target"].isin([0, 1]).all():
        raise ValueError(f"auxiliary target is not binary: {target_column}")
    return fit_classifier(work, features, spec, weight_spec)


def _regressor(spec: ModelSpec) -> Any:
    params = dict(spec.params)
    if spec.family == "hist_gradient":
        return HistGradientBoostingRegressor(random_state=spec.seed, **params)
    if spec.family == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            random_state=spec.seed,
            n_jobs=-1,
            verbosity=-1,
            objective="regression_l2",
            **params,
        )
    if spec.family == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            random_seed=spec.seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            loss_function="RMSE",
            **params,
        )
    if spec.family == "xgboost":
        if importlib.util.find_spec("xgboost") is None:
            raise CandidateUnavailable("xgboost is not installed in the formal environment")
        from xgboost import XGBRegressor

        return XGBRegressor(
            random_state=spec.seed,
            n_jobs=-1,
            objective="reg:squarederror",
            **params,
        )
    raise CandidateUnavailable(
        f"model family {spec.family} has no preregistered PHQ regression learner"
    )


def fit_regressor(
    train: pd.DataFrame,
    features: Sequence[str],
    spec: ModelSpec,
    weight_spec: WeightSpec,
    target_column: str = "phq9_score_r3_target",
) -> Pipeline:
    names = assert_deployable_feature_names(features)
    if target_column not in train.columns:
        raise ValueError(f"training frame missing regression target: {target_column}")
    model = Pipeline(
        [
            ("preprocess", _preprocessor(train, names)),
            ("model", _regressor(spec)),
        ]
    )
    weight = sample_weights(train, weight_spec)
    target = pd.to_numeric(train[target_column], errors="raise").to_numpy(float)
    model.fit(train[list(names)], target, model__sample_weight=weight)
    return model


def predict_regression(
    model: Pipeline, frame: pd.DataFrame, features: Sequence[str]
) -> np.ndarray:
    return np.clip(
        np.asarray(model.predict(frame[list(features)]), dtype=float),
        0.0,
        27.0,
    )


def predict_probability(
    model: Pipeline, frame: pd.DataFrame, features: Sequence[str]
) -> np.ndarray:
    probability = np.asarray(model.predict_proba(frame[list(features)])[:, 1], dtype=float)
    return np.clip(probability, 1e-6, 1.0 - 1e-6)


def binary_metrics(target: Sequence[int], probability: Sequence[float]) -> dict[str, float]:
    y = np.asarray(target, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(y) == 0 or np.unique(y).size < 2:
        return {"auprc": float("nan"), "auroc": float("nan"), "brier": float("nan")}
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def weighted_binary_metrics(
    target: Sequence[int], probability: Sequence[float], sample_weight: Sequence[float]
) -> dict[str, float]:
    y = np.asarray(target, dtype=int)
    p = np.asarray(probability, dtype=float)
    weight = np.asarray(sample_weight, dtype=float)
    if len(y) == 0 or np.unique(y).size < 2:
        return {"auprc": float("nan"), "auroc": float("nan"), "brier": float("nan")}
    return {
        "auprc": float(average_precision_score(y, p, sample_weight=weight)),
        "auroc": float(roc_auc_score(y, p, sample_weight=weight)),
        "brier": float(brier_score_loss(y, p, sample_weight=weight)),
    }


__all__ = [
    "CandidateUnavailable",
    "DEFAULT_WEIGHT_SPECS",
    "LEGACY_ACTIVITY_SPEC",
    "LEGACY_JOINT_SPEC",
    "LEGACY_PROFILE_SPEC",
    "LEGACY_SLEEP_SPEC",
    "ModelSpec",
    "WeightSpec",
    "binary_metrics",
    "build_classifier_pipeline",
    "default_model_specs",
    "fit_classifier",
    "fit_classifier_target",
    "fit_regressor",
    "predict_probability",
    "predict_regression",
    "sample_weights",
    "weighted_binary_metrics",
]
