"""Historical-assessment state-aware helpers for FORECAST-OPT-004F."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore[assignment,misc]


HISTORY_FEATURES = (
    "historical_phq9_last_score",
    "historical_phq9_age_months",
    "historical_phq9_count",
    "historical_phq9_mean",
    "historical_phq9_median",
    "historical_phq9_std",
    "historical_phq9_min",
    "historical_phq9_max",
    "historical_phq9_slope",
    "historical_phq9_last_delta",
)


def prepare_source_index(source: pd.DataFrame) -> pd.DataFrame:
    if {"__participant_id", "__nominal_month"}.issubset(source.columns):
        return source
    participants = []
    nominal_months = []
    for value in source.index.tolist():
        if not isinstance(value, str):
            raise ValueError("PSYCHE-D source index contains a non-string key")
        participant, separator, month_text = value.rpartition("_")
        if not separator or not participant or not month_text.isdigit():
            raise ValueError("PSYCHE-D source index contains a malformed key")
        participants.append(participant)
        nominal_months.append(int(month_text))
    result = source.copy()
    result["__participant_id"] = participants
    result["__nominal_month"] = nominal_months
    if result[["__participant_id", "__nominal_month"]].duplicated().any():
        raise ValueError("PSYCHE-D participant-month key is duplicated")
    return result


def build_history_features(samples: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    source = prepare_source_index(source)
    assessments = source.loc[
        source["phq9_score_end"].notna(),
        ["__participant_id", "__nominal_month", "phq9_score_end"],
    ].copy()
    assessments["__participant_id"] = assessments["__participant_id"].astype(str)
    assessments["__nominal_month"] = pd.to_numeric(
        assessments["__nominal_month"], errors="raise"
    ).astype(int)
    assessments["phq9_score_end"] = pd.to_numeric(
        assessments["phq9_score_end"], errors="raise"
    ).astype(float)
    history_map = {
        participant: rows.sort_values("__nominal_month")
        for participant, rows in assessments.groupby("__participant_id", sort=False)
    }
    rows = []
    for sample in samples.itertuples(index=False):
        participant = str(sample.global_participant_id).split("::", 1)[-1]
        cutoff = int(sample.feature_month_slot)
        history = history_map.get(participant)
        if history is None:
            eligible = pd.DataFrame(columns=assessments.columns)
        else:
            eligible = history[history["__nominal_month"] < cutoff]
        output: dict[str, float | int | str] = {
            "global_participant_id": sample.global_participant_id,
            "target_window_id": sample.target_window_id,
            "historical_assessment_available": int(len(eligible) > 0),
            "historical_assessment_slot": float("nan"),
        }
        if len(eligible) == 0:
            output.update({name: float("nan") for name in HISTORY_FEATURES})
        else:
            slots = eligible["__nominal_month"].to_numpy(dtype="float64")
            scores = eligible["phq9_score_end"].to_numpy(dtype="float64")
            output["historical_assessment_slot"] = int(slots[-1])
            slope = (
                float(np.polyfit(slots, scores, 1)[0])
                if len(scores) >= 2 and np.unique(slots).size >= 2
                else 0.0
            )
            output.update(
                {
                    "historical_phq9_last_score": float(scores[-1]),
                    "historical_phq9_age_months": float(cutoff - slots[-1]),
                    "historical_phq9_count": int(len(scores)),
                    "historical_phq9_mean": float(scores.mean()),
                    "historical_phq9_median": float(np.median(scores)),
                    "historical_phq9_std": float(scores.std(ddof=0)),
                    "historical_phq9_min": float(scores.min()),
                    "historical_phq9_max": float(scores.max()),
                    "historical_phq9_slope": slope,
                    "historical_phq9_last_delta": (
                        float(scores[-1] - scores[-2]) if len(scores) >= 2 else 0.0
                    ),
                }
            )
        rows.append(output)
    return pd.DataFrame(rows)


def select_passive_features(columns: Sequence[str]) -> tuple[str, ...]:
    direct = {
        "feature_nonmissing_count",
        "history_observed_month_count",
        "history_month_span",
        "history_months_since_last_any_record",
        "baseline_available_count",
        "cold_start_field_count",
    }
    suffixes = ("__anchor", "__delta_1", "__local_slope", "__personal_z")
    result = [
        name
        for name in columns
        if name in direct
        or (name.startswith("short__") and name.endswith(suffixes))
        or name.startswith("opt003__")
    ]
    prohibited = ("phq", "target", "future", "participant_id", "fold_id")
    result = [
        name
        for name in result
        if not any(token in name.lower() for token in prohibited)
    ]
    if not result:
        raise ValueError("no passive state-aware control features selected")
    return tuple(result)


def participant_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = (
        frame.groupby("global_participant_id", sort=False)["global_participant_id"]
        .transform("size")
        .to_numpy(dtype="float64")
    )
    return 1.0 / (float(frame["global_participant_id"].nunique()) * counts)


def participant_class_weights(frame: pd.DataFrame) -> np.ndarray:
    labels = frame["future_binary_target"].to_numpy(dtype="int8")
    work = frame[["global_participant_id"]].copy()
    work["label"] = labels
    class_participants = work.groupby("label")["global_participant_id"].nunique()
    row_counts = (
        work.groupby(["global_participant_id", "label"], sort=False)["label"]
        .transform("size")
        .to_numpy(dtype="float64")
    )
    participant_counts = np.asarray(
        [class_participants[int(label)] for label in labels], dtype="float64"
    )
    weights = 0.5 / (participant_counts * row_counts)
    return weights * (len(frame) / weights.sum())


def fit_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: Sequence[str],
    seeds: Sequence[int],
    iterations: int = 100,
) -> np.ndarray:
    if CatBoostClassifier is None:
        raise RuntimeError("CatBoost is required; run in eldercare-ai")
    x_train = train.loc[:, list(feature_names)].apply(pd.to_numeric, errors="coerce")
    x_validation = validation.loc[:, list(feature_names)].apply(
        pd.to_numeric, errors="coerce"
    )
    predictions = []
    for seed in seeds:
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
        predictions.append(model.predict_proba(x_validation)[:, 1])
    return np.mean(predictions, axis=0)


def metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    y = frame["future_binary_target"].to_numpy(dtype="int8")
    weights = participant_equal_weights(frame)
    weights /= weights.sum()
    return {
        "auprc": float(average_precision_score(y, probability, sample_weight=weights)),
        "auroc": float(roc_auc_score(y, probability, sample_weight=weights)),
        "brier": float(np.sum(weights * (probability - y) ** 2)),
    }
