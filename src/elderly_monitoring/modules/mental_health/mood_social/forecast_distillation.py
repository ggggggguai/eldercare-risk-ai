"""Passive CatBoost student helpers for FORECAST-OPT-005C."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore[assignment,misc]
    CatBoostRegressor = None  # type: ignore[assignment,misc]

CLIP = (1e-6, 1.0 - 1e-6)


def participant_class_weights(frame: pd.DataFrame) -> np.ndarray:
    labels = frame["future_binary_target"].to_numpy(dtype="int8")
    work = frame[["global_participant_id"]].copy()
    work["label"] = labels
    participant_count = work.groupby("label")["global_participant_id"].nunique()
    row_count = (
        work.groupby(["global_participant_id", "label"], sort=False)["label"]
        .transform("size")
        .to_numpy(dtype="float64")
    )
    class_count = np.asarray(
        [participant_count[int(label)] for label in labels], dtype="float64"
    )
    weights = 0.5 / (class_count * row_count)
    return weights * (len(frame) / weights.sum())


def teacher_reliability(
    available: Iterable[float], count: Iterable[float], age_months: Iterable[float]
) -> np.ndarray:
    availability = np.nan_to_num(np.asarray(list(available), dtype="float64"))
    history_count = np.nan_to_num(np.asarray(list(count), dtype="float64"))
    age = np.nan_to_num(np.asarray(list(age_months), dtype="float64"), nan=np.inf)
    result = availability * np.minimum(history_count / 3.0, 1.0) * np.exp(-age / 3.0)
    return np.clip(np.nan_to_num(result), 0.0, 1.0)


def soft_target(
    hard_label: Iterable[int],
    teacher_probability: Iterable[float],
    reliability: Iterable[float],
    alpha: float,
) -> np.ndarray:
    y = np.asarray(list(hard_label), dtype="float64")
    teacher = np.clip(
        np.nan_to_num(np.asarray(list(teacher_probability), dtype="float64"), nan=0.5),
        *CLIP,
    )
    confidence = np.clip(np.asarray(list(reliability), dtype="float64"), 0.0, 1.0)
    target = (1.0 - alpha * confidence) * y + alpha * confidence * teacher
    return np.clip(target, *CLIP)


def _matrix(frame: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    return frame.loc[:, list(feature_names)].apply(pd.to_numeric, errors="coerce")


def fit_predict_hard(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: Sequence[str],
    seeds: Sequence[int],
    *,
    iterations: int = 100,
) -> np.ndarray:
    if CatBoostClassifier is None:
        raise RuntimeError("CatBoost is required; run in eldercare-ai")
    x_train = _matrix(train, feature_names)
    x_validation = _matrix(validation, feature_names)

    def train_seed(seed: int) -> np.ndarray:
        model = CatBoostClassifier(
            loss_function="Logloss",
            iterations=iterations,
            depth=4,
            learning_rate=0.045,
            l2_leaf_reg=6.0,
            random_seed=int(seed),
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            x_train,
            train["future_binary_target"].to_numpy(dtype="int8"),
            sample_weight=participant_class_weights(train),
        )
        return model.predict_proba(x_validation)[:, 1]

    with ThreadPoolExecutor(max_workers=len(seeds)) as executor:
        predictions = list(executor.map(train_seed, seeds))
    return np.clip(np.mean(predictions, axis=0), *CLIP)


def fit_predict_soft(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: Sequence[str],
    seeds: Sequence[int],
    *,
    alpha: float,
    iterations: int = 100,
) -> np.ndarray:
    if CatBoostRegressor is None:
        raise RuntimeError("CatBoost is required; run in eldercare-ai")
    target = soft_target(
        train["future_binary_target"],
        train["teacher_probability"],
        train["teacher_reliability"],
        alpha,
    )
    x_train = _matrix(train, feature_names)
    x_validation = _matrix(validation, feature_names)

    def train_seed(seed: int) -> np.ndarray:
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=iterations,
            depth=4,
            learning_rate=0.045,
            l2_leaf_reg=6.0,
            random_seed=int(seed),
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            x_train,
            target,
            sample_weight=participant_class_weights(train),
        )
        return model.predict(x_validation)

    with ThreadPoolExecutor(max_workers=len(seeds)) as executor:
        predictions = list(executor.map(train_seed, seeds))
    return np.clip(np.mean(predictions, axis=0), *CLIP)


__all__ = [
    "fit_predict_hard",
    "fit_predict_soft",
    "participant_class_weights",
    "soft_target",
    "teacher_reliability",
]
