"""Finite r5 candidate registry and participant-safe model fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import (
    ap_context_metrics,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.evaluation import (
    ap_at_prevalence,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    participant_equal_weights,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.contract import (
    R5_CANDIDATE_FAMILY_BUDGET,
)


Family = Literal["catboost", "lightgbm", "extra_trees", "elasticnet_spline", "small_mlp"]


@dataclass(frozen=True)
class R5CandidateSpec:
    candidate_id: str
    family: Family
    params: dict[str, Any]
    seed: int


def r5_candidate_space(seed: int) -> tuple[R5CandidateSpec, ...]:
    """Return the exact frozen maximum budget: 12/12/6/6/4."""

    specs: list[R5CandidateSpec] = []
    for depth in (3, 4, 5, 6):
        for learning_rate, l2 in ((0.02, 12.0), (0.035, 8.0), (0.05, 5.0)):
            specs.append(
                R5CandidateSpec(
                    f"cat_d{depth}_lr{str(learning_rate).replace('.', '')}_l2{int(l2)}",
                    "catboost",
                    {
                        "iterations": 350 if learning_rate <= 0.035 else 250,
                        "depth": depth,
                        "learning_rate": learning_rate,
                        "l2_leaf_reg": l2,
                        "random_strength": 0.5,
                        "rsm": 0.9,
                    },
                    int(seed),
                )
            )
    for leaves in (7, 15, 31, 47):
        for minimum, regularization in ((20, 3.0), (40, 7.0), (80, 12.0)):
            specs.append(
                R5CandidateSpec(
                    f"lgb_leaf{leaves}_min{minimum}_l2{int(regularization)}",
                    "lightgbm",
                    {
                        "n_estimators": 400,
                        "learning_rate": 0.025,
                        "num_leaves": leaves,
                        "min_child_samples": minimum,
                        "reg_lambda": regularization,
                        "reg_alpha": regularization / 10.0,
                        "subsample": 0.85,
                        "colsample_bytree": 0.85,
                    },
                    int(seed),
                )
            )
    for leaf, features in ((2, 0.5), (4, 0.7), (8, 1.0)):
        for estimators in (300, 600):
            specs.append(
                R5CandidateSpec(
                    f"extra_n{estimators}_leaf{leaf}_f{int(features * 10)}",
                    "extra_trees",
                    {
                        "n_estimators": estimators,
                        "min_samples_leaf": leaf,
                        "max_features": features,
                        "class_weight": "balanced_subsample",
                    },
                    int(seed),
                )
            )
    for c, ratio, spline in (
        (0.05, 0.1, False),
        (0.1, 0.2, False),
        (0.5, 0.5, False),
        (1.0, 0.8, False),
        (0.1, 0.2, True),
        (0.5, 0.5, True),
    ):
        specs.append(
            R5CandidateSpec(
                f"elastic_c{str(c).replace('.', '')}_l1{int(ratio * 10)}{'_spline' if spline else ''}",
                "elasticnet_spline",
                {"C": c, "l1_ratio": ratio, "spline": spline},
                int(seed),
            )
        )
    for width, alpha in ((16, 0.01), (32, 0.01), (32, 0.1), (64, 0.1)):
        specs.append(
            R5CandidateSpec(
                f"mlp_w{width}_a{str(alpha).replace('.', '')}",
                "small_mlp",
                {"hidden_layer_sizes": (width,), "alpha": alpha},
                int(seed),
            )
        )
    observed = {
        family: len([spec for spec in specs if spec.family == family])
        for family in R5_CANDIDATE_FAMILY_BUDGET
    }
    if observed != R5_CANDIDATE_FAMILY_BUDGET:
        raise AssertionError(f"r5 candidate registry drift: {observed}")
    return tuple(specs)


def _column_groups(frame: pd.DataFrame, features: Sequence[str]) -> tuple[list[str], list[str]]:
    numeric = [name for name in features if pd.api.types.is_numeric_dtype(frame[name].dtype)]
    categorical = [name for name in features if name not in numeric]
    return numeric, categorical


def _preprocessor(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    scaled: bool,
    spline: bool = False,
) -> ColumnTransformer:
    numeric, categorical = _column_groups(frame, features)
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if spline:
        numeric_steps.append(
            (
                "spline",
                SplineTransformer(
                    n_knots=4,
                    degree=2,
                    knots="quantile",
                    extrapolation="linear",
                    include_bias=False,
                ),
            )
        )
    if scaled:
        numeric_steps.append(("scale", StandardScaler()))
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
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
    return ColumnTransformer(transformers, remainder="drop")


def build_r5_pipeline(
    frame: pd.DataFrame, features: Sequence[str], spec: R5CandidateSpec
) -> Pipeline:
    scaled = spec.family in {"elasticnet_spline", "small_mlp"}
    spline = spec.family == "elasticnet_spline" and bool(spec.params.get("spline"))
    preprocessor = _preprocessor(frame, features, scaled=scaled, spline=spline)
    if spec.family == "catboost":
        from catboost import CatBoostClassifier

        model: Any = CatBoostClassifier(
            loss_function="Logloss",
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
            random_seed=spec.seed,
            **spec.params,
        )
    elif spec.family == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            objective="binary",
            verbosity=-1,
            n_jobs=1,
            random_state=spec.seed,
            **spec.params,
        )
    elif spec.family == "extra_trees":
        model = ExtraTreesClassifier(n_jobs=1, random_state=spec.seed, **spec.params)
    elif spec.family == "elasticnet_spline":
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=float(spec.params["C"]),
            l1_ratio=float(spec.params["l1_ratio"]),
            max_iter=3000,
            random_state=spec.seed,
        )
    elif spec.family == "small_mlp":
        model = MLPClassifier(
            activation="relu",
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=250,
            batch_size=128,
            random_state=spec.seed,
            **spec.params,
        )
    else:  # pragma: no cover
        raise ValueError(f"unknown r5 model family: {spec.family}")
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def fit_r5_model(
    frame: pd.DataFrame,
    features: Sequence[str],
    spec: R5CandidateSpec,
    *,
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
    model = build_r5_pipeline(frame, features, spec)
    weight = participant_equal_weights(frame) if sample_weight is None else np.asarray(sample_weight, float)
    if spec.family == "small_mlp":
        # sklearn's MLP sample_weight support varies by release; the frozen
        # environment supports it, and tests cover the current binding.
        model.fit(frame[list(features)], frame["binary_target"].astype(int), model__sample_weight=weight)
    else:
        model.fit(frame[list(features)], frame["binary_target"].astype(int), model__sample_weight=weight)
    return model


def predict_r5_model(model: Pipeline, frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    return np.clip(model.predict_proba(frame[list(features)])[:, 1], 1.0e-6, 1.0 - 1.0e-6)


def candidate_inner_oof(
    frame: pd.DataFrame,
    features: Sequence[str],
    inner_fold: pd.Series,
    spec: R5CandidateSpec,
) -> tuple[np.ndarray, dict[str, float]]:
    probability = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold in sorted(inner_fold.astype(int).unique()):
        train = frame.loc[inner_fold.ne(fold)]
        validation = frame.loc[inner_fold.eq(fold)]
        model = fit_r5_model(train, features, spec)
        probability.loc[validation.index] = predict_r5_model(model, validation, features)
    if probability.isna().any():
        raise ValueError("r5 candidate inner OOF is incomplete")
    target = frame["binary_target"].to_numpy(int)
    natural = ap_context_metrics(target, probability.to_numpy(float))
    participant = ap_context_metrics(
        target,
        probability.to_numpy(float),
        sample_weight=participant_equal_weights(frame),
    )
    source_ap_at_10pct: list[float] = []
    probability_array = probability.to_numpy(float)
    if "dataset_id" in frame.columns:
        for _, positions in frame.groupby("dataset_id", sort=True).indices.items():
            position_array = np.asarray(positions, dtype=int)
            source_target = target[position_array]
            if np.unique(source_target).size < 2:
                continue
            source_ap_at_10pct.append(
                ap_at_prevalence(
                    source_target,
                    probability_array[position_array],
                    reference_prevalence=0.10,
                )
            )
    macro_ap_at_10pct = (
        float(np.mean(source_ap_at_10pct))
        if source_ap_at_10pct
        else ap_at_prevalence(target, probability_array, reference_prevalence=0.10)
    )
    metrics = {
        "natural_auprc": natural["auprc"],
        "participant_auprc": participant["auprc"],
        "normalized_ap": natural["normalized_ap"],
        "auroc": natural["auroc"],
        "brier": natural["brier"],
        "macro_ap_at_10pct": macro_ap_at_10pct,
        "selection_score_multisource": 0.55 * natural["auprc"] + 0.25 * participant["auprc"] + 0.20 * macro_ap_at_10pct,
        "selection_score_psyche": 0.65 * natural["auprc"] + 0.35 * participant["auprc"],
    }
    return probability.to_numpy(float), metrics


def serialize_spec(spec: R5CandidateSpec) -> dict[str, Any]:
    return asdict(spec)


__all__ = [
    "R5CandidateSpec",
    "build_r5_pipeline",
    "candidate_inner_oof",
    "fit_r5_model",
    "predict_r5_model",
    "r5_candidate_space",
    "serialize_spec",
]
