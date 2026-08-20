"""Shrunk personal baselines and Cold/Warm/Recent experts for OPT-004C."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - guarded at runtime
    CatBoostClassifier = None  # type: ignore[assignment,misc]


EXPERTS = ("cold_start", "warm_history", "recent_trend")
GATES = ("single_model", "hard_gate", "soft_gate")
LAMBDA_CANDIDATES = (1.0, 3.0, 6.0, 12.0)


def participant_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = (
        frame.groupby("global_participant_id", sort=False)["global_participant_id"]
        .transform("size")
        .to_numpy(dtype="float64")
    )
    return 1.0 / (float(frame["global_participant_id"].nunique()) * counts)


def participant_class_weights(frame: pd.DataFrame) -> np.ndarray:
    labels = frame["future_binary_target"].to_numpy(dtype="int8")
    if np.unique(labels).size != 2:
        raise ValueError("expert training subset must contain both classes")
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
    # Tree libraries use absolute hessian mass for split constraints.
    return weights * (len(frame) / weights.sum())


def assign_expert(frame: pd.DataFrame) -> np.ndarray:
    """Frozen, mutually exclusive eligibility based only on history quality."""

    history = frame["history_observed_month_count"].to_numpy(dtype="float64")
    baseline = frame["baseline_available_count"].to_numpy(dtype="float64")
    completeness = frame["feature_nonmissing_count"].to_numpy(dtype="float64")
    cold = (history <= 2.0) | (baseline < 9.0)
    recent = (~cold) & (history >= 6.0) & (completeness >= 20.0)
    result = np.full(len(frame), "warm_history", dtype=object)
    result[cold] = "cold_start"
    result[recent] = "recent_trend"
    return result.astype(str)


def soft_gate_weights(frame: pd.DataFrame) -> np.ndarray:
    """Continuous expert weights derived only from history and completeness."""

    history = frame["history_observed_month_count"].to_numpy(dtype="float64")
    baseline = frame["baseline_available_count"].to_numpy(dtype="float64")
    completeness = frame["feature_nonmissing_count"].to_numpy(dtype="float64")
    history_strength = np.clip((history - 1.0) / 8.0, 0.0, 1.0)
    baseline_strength = np.clip(baseline / 27.0, 0.0, 1.0)
    complete_strength = np.clip((completeness - 10.0) / 17.0, 0.0, 1.0)
    quality = history_strength * np.sqrt(baseline_strength * complete_strength)
    cold = np.clip(1.0 - 1.5 * quality, 0.0, 1.0)
    recent = np.clip((quality - 0.35) / 0.65, 0.0, 1.0)
    warm = np.maximum(0.15, 1.0 - cold - recent)
    weights = np.column_stack([cold, warm, recent])
    return weights / weights.sum(axis=1, keepdims=True)


def _base_fields(columns: Sequence[str]) -> tuple[str, ...]:
    fields = []
    for name in columns:
        if name.startswith("short__") and name.endswith("__anchor"):
            fields.append(name[len("short__") : -len("__anchor")])
    return tuple(sorted(fields))


def select_expert_base_features(columns: Sequence[str]) -> tuple[str, ...]:
    direct = {
        "feature_nonmissing_count",
        "history_observed_month_count",
        "history_month_span",
        "history_months_since_last_any_record",
        "baseline_available_count",
        "cold_start_field_count",
        "group_reference_available_count",
        "cold_start_flag",
    }
    suffixes = (
        "__anchor",
        "__anchor_missing",
        "__delta_1",
        "__delta_1_missing",
        "__local_slope",
        "__local_slope_missing",
        "__personal_z",
        "__personal_z_missing",
        "__history_count",
        "__months_since_last_valid",
        "__months_since_last_valid_missing",
    )
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
        raise ValueError("no expert base features selected")
    return tuple(result)


@dataclass
class ShrunkBaselineTransformer:
    """Fold-local cohort references with strict feature-slot precedence."""

    lambda_value: float
    base_features: tuple[str, ...]
    fields: tuple[str, ...] = field(default_factory=tuple)
    cohort_by_field_slot: dict[str, dict[int, float]] = field(default_factory=dict)
    reference_member_counts: dict[str, dict[int, int]] = field(default_factory=dict)

    def fit(self, train: pd.DataFrame) -> "ShrunkBaselineTransformer":
        if self.lambda_value not in LAMBDA_CANDIDATES:
            raise ValueError("lambda is outside the frozen candidate set")
        self.fields = _base_fields(train.columns)
        slots = sorted(int(value) for value in train["feature_month_slot"].unique())
        for field_name in self.fields:
            anchor_name = f"short__{field_name}__anchor"
            refs: dict[int, float] = {}
            counts: dict[int, int] = {}
            for slot in slots:
                eligible = train[train["feature_month_slot"] < slot][
                    ["global_participant_id", anchor_name]
                ].copy()
                eligible[anchor_name] = pd.to_numeric(
                    eligible[anchor_name], errors="coerce"
                )
                eligible = eligible[np.isfinite(eligible[anchor_name])]
                participant_values = eligible.groupby("global_participant_id")[
                    anchor_name
                ].median()
                refs[slot] = (
                    float(participant_values.median())
                    if len(participant_values)
                    else float("nan")
                )
                counts[slot] = int(len(participant_values))
            self.cohort_by_field_slot[field_name] = refs
            self.reference_member_counts[field_name] = counts
        return self

    @property
    def derived_feature_names(self) -> tuple[str, ...]:
        names = []
        for field_name in self.fields:
            names.extend(
                [
                    f"shrunk__{field_name}__baseline",
                    f"shrunk__{field_name}__anchor_delta",
                    f"shrunk__{field_name}__reference_available",
                ]
            )
        return tuple(names)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fields:
            raise RuntimeError("shrunk baseline transformer is not fitted")
        output = frame.loc[:, list(self.base_features)].apply(
            pd.to_numeric, errors="coerce"
        )
        slot_values = frame["feature_month_slot"].to_numpy(dtype="int64")
        for field_name in self.fields:
            personal = pd.to_numeric(
                frame[f"short__{field_name}__personal_median"], errors="coerce"
            ).to_numpy(dtype="float64")
            count = (
                pd.to_numeric(
                    frame[f"short__{field_name}__history_count"], errors="coerce"
                )
                .fillna(0.0)
                .to_numpy(dtype="float64")
            )
            anchor = pd.to_numeric(
                frame[f"short__{field_name}__anchor"], errors="coerce"
            ).to_numpy(dtype="float64")
            reference = np.asarray(
                [
                    self.cohort_by_field_slot[field_name].get(int(slot), np.nan)
                    for slot in slot_values
                ],
                dtype="float64",
            )
            available = np.isfinite(personal) & np.isfinite(reference)
            shrunk = np.full(len(frame), np.nan, dtype="float64")
            weight = count / (count + self.lambda_value)
            shrunk[available] = (
                weight[available] * personal[available]
                + (1.0 - weight[available]) * reference[available]
            )
            output[f"shrunk__{field_name}__baseline"] = shrunk
            output[f"shrunk__{field_name}__anchor_delta"] = anchor - shrunk
            output[f"shrunk__{field_name}__reference_available"] = available.astype(
                "float64"
            )
        output = output.replace([np.inf, -np.inf], np.nan)
        return output

    def audit(self) -> dict[str, Any]:
        return {
            "lambda": self.lambda_value,
            "formula": "n/(n+lambda)*personal + lambda/(n+lambda)*cohort_reference",
            "cohort_scope": "inner_train_participants_with_feature_month_slot_strictly_before_row_cutoff",
            "field_count": len(self.fields),
            "reference_member_counts": self.reference_member_counts,
        }


def fit_expert_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    transformer: ShrunkBaselineTransformer,
    seeds: Sequence[int],
    iterations: int = 80,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if CatBoostClassifier is None:
        raise RuntimeError("CatBoost is required; run in eldercare-ai")
    x_train = transformer.transform(train)
    x_validation = transformer.transform(validation)
    train_expert = assign_expert(train)
    predictions: dict[str, np.ndarray] = {}
    audit: dict[str, Any] = {"models": {}}
    scopes = {"single_model": np.ones(len(train), dtype=bool)}
    scopes.update({name: train_expert == name for name in EXPERTS})
    for name, mask in scopes.items():
        subset = train.loc[mask]
        if len(subset) == 0 or subset["future_binary_target"].nunique() != 2:
            raise ValueError(f"{name} expert has insufficient class coverage")
        seed_predictions = []
        for seed in seeds:
            model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
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
                x_train.loc[mask],
                subset["future_binary_target"].to_numpy(dtype="int8"),
                sample_weight=participant_class_weights(subset),
            )
            seed_predictions.append(model.predict_proba(x_validation)[:, 1])
        predictions[name] = np.mean(seed_predictions, axis=0)
        audit["models"][name] = {
            "train_rows": int(mask.sum()),
            "train_participants": int(subset["global_participant_id"].nunique()),
            "positive_rate": float(subset["future_binary_target"].mean()),
            "seed_members": list(map(int, seeds)),
        }
    return predictions, audit


def apply_gate(
    gate: str, validation: pd.DataFrame, predictions: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    if gate == "single_model":
        weights = np.zeros((len(validation), 3), dtype="float64")
        return predictions["single_model"], weights
    if gate == "hard_gate":
        assigned = assign_expert(validation)
        weights = np.column_stack([assigned == name for name in EXPERTS]).astype(
            "float64"
        )
    elif gate == "soft_gate":
        weights = soft_gate_weights(validation)
    else:
        raise ValueError(f"unsupported gate: {gate}")
    matrix = np.column_stack([predictions[name] for name in EXPERTS])
    return np.sum(weights * matrix, axis=1), weights


def weighted_ece(y: np.ndarray, probability: np.ndarray, weights: np.ndarray) -> float:
    weights = weights / weights.sum()
    bins = np.minimum((np.clip(probability, 0.0, 1.0) * 10).astype(int), 9)
    result = 0.0
    for bin_id in range(10):
        mask = bins == bin_id
        if mask.any():
            result += float(weights[mask].sum()) * abs(
                float(np.average(probability[mask], weights=weights[mask]))
                - float(np.average(y[mask], weights=weights[mask]))
            )
    return float(result)


def binary_metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    y = frame["future_binary_target"].to_numpy(dtype="int8")
    weights = participant_equal_weights(frame)
    weights /= weights.sum()
    return {
        "auprc": float(average_precision_score(y, probability, sample_weight=weights)),
        "auroc": float(roc_auc_score(y, probability, sample_weight=weights)),
        "brier": float(np.sum(weights * (probability - y) ** 2)),
        "ece": weighted_ece(y, probability, weights),
    }
