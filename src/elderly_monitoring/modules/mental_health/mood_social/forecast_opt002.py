"""Leakage-safe model selection primitives for FORECAST-OPT-002D-F."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score

from elderly_monitoring.modules.mental_health.mood_social.forecast_modeling import (
    participant_equal_weights,
)
from elderly_monitoring.modules.mental_health.mood_social.forecast_targets_weights import (
    inner_class_balanced_weight,
    participant_equal_weight,
    strict_history_recency_weight,
    threshold_near_weight,
)

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore[assignment,misc]

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None  # type: ignore[assignment,misc]

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


SEED = 20260728
TASKS = ("forecast_1m", "forecast_2m")
FAMILIES = ("elasticnet_logistic", "lightgbm", "catboost")
WEIGHT_SCHEMES = (
    "participant_equal",
    "inner_fold_class_balanced",
    "strict_history_recency",
    "threshold_near_downweight",
)
AUXILIARY_MODES = ("none", "phq9_score", "phq9_ordinal", "phq9_direction")


class Opt002Error(ValueError):
    """Raised when a frozen nested-CV contract is violated."""


@dataclass
class NumericPreprocessor:
    feature_names: tuple[str, ...]
    medians: np.ndarray | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "NumericPreprocessor":
        values = numeric_matrix(frame, self.feature_names)
        medians = np.zeros(values.shape[1], dtype="float64")
        for index in range(values.shape[1]):
            finite = values[:, index][np.isfinite(values[:, index])]
            medians[index] = float(np.median(finite)) if finite.size else 0.0
        filled = np.where(np.isfinite(values), values, medians)
        means = filled.mean(axis=0)
        scales = filled.std(axis=0)
        scales[~np.isfinite(scales) | (scales == 0.0)] = 1.0
        self.medians, self.means, self.scales = medians, means, scales
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians is None or self.means is None or self.scales is None:
            raise RuntimeError("preprocessor has not been fitted")
        values = numeric_matrix(frame, self.feature_names)
        filled = np.where(np.isfinite(values), values, self.medians)
        return (filled - self.means) / self.scales


@dataclass(frozen=True)
class Candidate:
    feature_group: str
    model_family: str
    weight_scheme: str = "participant_equal"
    auxiliary_mode: str = "none"
    stage: str = "base"

    @property
    def candidate_id(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, str]:
        return {
            "feature_group": self.feature_group,
            "model_family": self.model_family,
            "weight_scheme": self.weight_scheme,
            "auxiliary_mode": self.auxiliary_mode,
            "stage": self.stage,
        }


@dataclass
class AuxiliaryBundle:
    mode: str
    preprocessor: NumericPreprocessor
    estimator: Any

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = self.estimator.predict(self.preprocessor.transform(frame))
        return np.clip(np.asarray(values, dtype="float64"), 0.0, 1.0)


@dataclass
class BinaryBundle:
    candidate: Candidate
    feature_names: tuple[str, ...]
    preprocessor: NumericPreprocessor | None
    estimator: Any
    auxiliary: AuxiliaryBundle | None = None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        work = frame.copy()
        features = list(self.feature_names)
        if self.auxiliary is not None:
            work["__auxiliary_prediction"] = self.auxiliary.predict(work)
            features.append("__auxiliary_prediction")
        if self.preprocessor is not None:
            matrix = self.preprocessor.transform(work)
        else:
            matrix = numeric_matrix(work, features)
        return np.asarray(self.estimator.predict_proba(matrix)[:, 1], dtype="float64")


def numeric_matrix(frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise Opt002Error(f"model features are missing: {missing[:10]}")
    values = (
        frame.loc[:, list(features)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype="float64", copy=True)
    )
    values[~np.isfinite(values)] = np.nan
    return values


def feature_groups(manifest: Mapping[str, Any], task: str) -> dict[str, tuple[str, ...]]:
    groups = manifest["horizons"][task]["feature_groups"]
    anchor = tuple(groups["anchor"])
    core_names = [*groups["anchor"], *groups["delta"], *groups["synchrony"]]
    if task == "forecast_2m":
        core_names.extend(groups["long_term"])
    core = tuple(dict.fromkeys(core_names))
    full = tuple(manifest["horizons"][task]["feature_names"])
    result = {"anchor": anchor, "horizon_core": core, "full": full}
    expected = int(manifest["horizons"][task]["feature_count"])
    if len(full) != expected:
        raise Opt002Error(f"{task} feature manifest count changed")
    return result


def training_weights(frame: pd.DataFrame, scheme: str) -> np.ndarray:
    if scheme == "participant_equal":
        values = participant_equal_weight(frame)
    elif scheme == "inner_fold_class_balanced":
        values = inner_class_balanced_weight(frame)
    elif scheme == "strict_history_recency":
        values = strict_history_recency_weight(frame)
    elif scheme == "threshold_near_downweight":
        values = threshold_near_weight(frame)
    else:
        raise Opt002Error(f"unknown weight scheme: {scheme}")
    output = values.to_numpy(dtype="float64")
    if not np.isfinite(output).all() or (output <= 0).any():
        raise Opt002Error("training weights are invalid")
    return output


def verify_weight_audit(
    train: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    task: str,
    outer_fold: int,
    inner_fold: int,
    scheme: str,
) -> np.ndarray:
    expected = audit.loc[
        audit["task_id"].eq(task)
        & audit["evaluation_outer_fold"].eq(outer_fold)
        & audit["inner_fold_id"].eq(inner_fold)
        & audit["weight_scheme"].eq(scheme)
        & audit["fit_scope"].eq("outer_train_inner_train"),
        ["target_window_id", "weight"],
    ]
    if len(expected) != len(train):
        raise Opt002Error("weight audit row count does not match inner-train")
    merged = train[["target_window_id"]].merge(
        expected, on="target_window_id", how="left", validate="one_to_one"
    )
    if merged["weight"].isna().any():
        raise Opt002Error("weight audit does not cover inner-train")
    actual = training_weights(train, scheme)
    if not np.allclose(actual, merged["weight"].to_numpy(dtype="float64")):
        raise Opt002Error("recomputed training weights differ from frozen audit")
    return actual


def _estimator(family: str) -> Any:
    if family == "elasticnet_logistic":
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=0.08,
            l1_ratio=0.15,
            max_iter=180,
            tol=5e-3,
            random_state=SEED,
        )
    if family == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("LightGBM is unavailable")
        return LGBMClassifier(
            objective="binary",
            n_estimators=80,
            learning_rate=0.035,
            num_leaves=15,
            max_depth=5,
            min_child_samples=30,
            subsample=0.9,
            colsample_bytree=0.75,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
    if family == "catboost":
        if CatBoostClassifier is None:
            raise RuntimeError("CatBoost is unavailable")
        return CatBoostClassifier(
            iterations=80,
            depth=5,
            learning_rate=0.035,
            l2_leaf_reg=3.0,
            loss_function="Logloss",
            random_seed=SEED,
            thread_count=1,
            allow_writing_files=False,
            verbose=False,
        )
    raise Opt002Error(f"unsupported family: {family}")


def _auxiliary_target(frame: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "phq9_score":
        target = frame["aux_phq9_score"].to_numpy(dtype="float64") / 27.0
    elif mode == "phq9_ordinal":
        target = frame["aux_phq9_ordinal"].to_numpy(dtype="float64") / 4.0
    elif mode == "phq9_direction":
        target = (frame["aux_phq9_direction"].to_numpy(dtype="float64") + 1.0) / 2.0
    else:
        raise Opt002Error(f"unsupported auxiliary mode: {mode}")
    available = np.isfinite(target)
    return target, available


def fit_auxiliary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
    mode: str,
) -> tuple[np.ndarray, AuxiliaryBundle]:
    target, available = _auxiliary_target(train, mode)
    if available.sum() < 20:
        raise Opt002Error(f"insufficient available rows for {mode}")
    fit = train.loc[available]
    preprocessor = NumericPreprocessor(tuple(features)).fit(fit)
    estimator = Ridge(alpha=10.0, solver="lsqr", tol=1e-4)
    estimator.fit(
        preprocessor.transform(fit),
        target[available],
        sample_weight=participant_equal_weights(fit),
    )
    bundle = AuxiliaryBundle(mode, preprocessor, estimator)
    return bundle.predict(validation), bundle


def strict_auxiliary_views(
    frame: pd.DataFrame,
    assignments: Mapping[str, int],
    features: Sequence[str],
    mode: str,
    validation_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = frame["global_participant_id"].map(assignments)
    if folds.isna().any():
        raise Opt002Error("inner split does not cover all participants")
    train = frame.loc[folds.ne(validation_fold)].copy()
    validation = frame.loc[folds.eq(validation_fold)].copy()
    train["__auxiliary_prediction"] = np.nan
    for aux_fold in sorted(folds.loc[train.index].astype(int).unique()):
        aux_valid = train["global_participant_id"].map(assignments).eq(aux_fold)
        prediction, _ = fit_auxiliary(
            train.loc[~aux_valid], train.loc[aux_valid], features, mode
        )
        train.loc[aux_valid, "__auxiliary_prediction"] = prediction
    prediction, _ = fit_auxiliary(train, validation, features, mode)
    validation["__auxiliary_prediction"] = prediction
    if train["__auxiliary_prediction"].isna().any():
        raise Opt002Error("nested auxiliary OOF is incomplete")
    return train, validation


def fit_binary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
    family: str,
    weights: np.ndarray,
) -> tuple[np.ndarray, NumericPreprocessor | None, Any]:
    if family == "elasticnet_logistic":
        preprocessor = NumericPreprocessor(tuple(features)).fit(train)
        x_train = preprocessor.transform(train)
        x_validation = preprocessor.transform(validation)
    else:
        preprocessor = None
        x_train = numeric_matrix(train, features)
        x_validation = numeric_matrix(validation, features)
    estimator = _estimator(family)
    estimator.fit(
        x_train,
        train["future_binary_target"].to_numpy(dtype="int8"),
        sample_weight=weights,
    )
    probability = estimator.predict_proba(x_validation)[:, 1]
    return np.asarray(probability, dtype="float64"), preprocessor, estimator


def evaluate_candidate(
    frame: pd.DataFrame,
    assignments: Mapping[str, int],
    groups: Mapping[str, tuple[str, ...]],
    candidate: Candidate,
    weight_audit: pd.DataFrame,
    task: str,
    outer_fold: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    parts: list[pd.DataFrame] = []
    base_features = groups[candidate.feature_group]
    auxiliary_features = groups["horizon_core"]
    for inner_fold in range(5):
        folds = frame["global_participant_id"].map(assignments)
        if folds.isna().any():
            raise Opt002Error("inner split is incomplete")
        if candidate.auxiliary_mode == "none":
            train = frame.loc[folds.ne(inner_fold)].copy()
            validation = frame.loc[folds.eq(inner_fold)].copy()
            model_features = base_features
        else:
            train, validation = strict_auxiliary_views(
                frame,
                assignments,
                auxiliary_features,
                candidate.auxiliary_mode,
                inner_fold,
            )
            model_features = (*base_features, "__auxiliary_prediction")
        weights = verify_weight_audit(
            train,
            weight_audit,
            task=task,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            scheme=candidate.weight_scheme,
        )
        probability, _, _ = fit_binary(
            train, validation, model_features, candidate.model_family, weights
        )
        part = validation[
            [
                "global_participant_id",
                "target_window_id",
                "future_binary_target",
                "outer_fold_id",
            ]
        ].copy()
        part["inner_fold_id"] = inner_fold
        part["p_raw"] = probability
        parts.append(part)
    output = pd.concat(parts, ignore_index=True).sort_values(
        "target_window_id", kind="mergesort"
    )
    if len(output) != len(frame) or output["target_window_id"].duplicated().any():
        raise Opt002Error("candidate inner OOF coverage is invalid")
    weights = participant_equal_weights(output)
    target = output["future_binary_target"].to_numpy(dtype="int8")
    probability = output["p_raw"].to_numpy(dtype="float64")
    metrics = {
        "auprc": float(average_precision_score(target, probability, sample_weight=weights)),
        "brier": float(np.average((probability - target) ** 2, weights=weights)),
    }
    return output.reset_index(drop=True), metrics


def fit_final_binary_bundle(
    frame: pd.DataFrame,
    assignments: Mapping[str, int],
    groups: Mapping[str, tuple[str, ...]],
    candidate: Candidate,
) -> BinaryBundle:
    work = frame.copy()
    features = list(groups[candidate.feature_group])
    auxiliary_bundle = None
    if candidate.auxiliary_mode != "none":
        work["__auxiliary_prediction"] = np.nan
        folds = work["global_participant_id"].map(assignments)
        for inner_fold in range(5):
            validation = folds.eq(inner_fold)
            prediction, _ = fit_auxiliary(
                work.loc[~validation],
                work.loc[validation],
                groups["horizon_core"],
                candidate.auxiliary_mode,
            )
            work.loc[validation, "__auxiliary_prediction"] = prediction
        _, auxiliary_bundle = fit_auxiliary(
            work, work.iloc[:1], groups["horizon_core"], candidate.auxiliary_mode
        )
        features.append("__auxiliary_prediction")
    weights = training_weights(work, candidate.weight_scheme)
    _, preprocessor, estimator = fit_binary(
        work, work.iloc[:1], features, candidate.model_family, weights
    )
    return BinaryBundle(
        candidate=candidate,
        feature_names=tuple(groups[candidate.feature_group]),
        preprocessor=preprocessor,
        estimator=estimator,
        auxiliary=auxiliary_bundle,
    )


if nn is not None:

    class SharedTwoHeadNet(nn.Module):
        def __init__(self, input_dim: int, hidden_units: int) -> None:
            super().__init__()
            self.shared = nn.Sequential(nn.Linear(input_dim, hidden_units), nn.ReLU())
            self.heads = nn.ModuleList([nn.Linear(hidden_units, 1), nn.Linear(hidden_units, 1)])

        def forward(self, values: Any, task_index: Any) -> Any:
            hidden = self.shared(values)
            logits = torch.cat([head(hidden) for head in self.heads], dim=1)
            return logits.gather(1, task_index[:, None]).squeeze(1)


@dataclass
class SharedBundle:
    feature_names: tuple[str, ...]
    preprocessor: NumericPreprocessor
    hidden_units: int
    state_dict: dict[str, Any]

    def predict(self, frame: pd.DataFrame, task: str) -> np.ndarray:
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is unavailable")
        model = SharedTwoHeadNet(len(self.feature_names), self.hidden_units)
        model.load_state_dict(self.state_dict)
        model.eval()
        values = torch.tensor(
            self.preprocessor.transform(frame), dtype=torch.float32
        )
        tasks = torch.full((len(frame),), TASKS.index(task), dtype=torch.long)
        with torch.no_grad():
            return torch.sigmoid(model(values, tasks)).numpy().astype("float64")


def fit_shared_bundle(
    frames: Mapping[str, pd.DataFrame],
    features: Sequence[str],
    *,
    hidden_units: int = 16,
    epochs: int = 32,
    learning_rate: float = 0.01,
) -> SharedBundle:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is unavailable")
    combined = pd.concat(
        [frame.assign(__task_index=TASKS.index(task)) for task, frame in frames.items()],
        ignore_index=True,
    )
    preprocessor = NumericPreprocessor(tuple(features)).fit(combined)
    x = torch.tensor(preprocessor.transform(combined), dtype=torch.float32)
    y = torch.tensor(combined["future_binary_target"].to_numpy(), dtype=torch.float32)
    task_index = torch.tensor(combined["__task_index"].to_numpy(), dtype=torch.long)
    weights = []
    for task in TASKS:
        part = combined.loc[combined["__task_index"].eq(TASKS.index(task))]
        task_weights = participant_equal_weights(part)
        weights.extend(zip(part.index.tolist(), task_weights.tolist(), strict=True))
    weight_array = np.zeros(len(combined), dtype="float64")
    for index, value in weights:
        weight_array[int(index)] = value
    weight_array /= max(float(weight_array.mean()), 1e-12)
    weight = torch.tensor(weight_array, dtype=torch.float32)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    model = SharedTwoHeadNet(len(features), hidden_units)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = (loss_fn(model(x, task_index), y) * weight).mean()
        loss.backward()
        optimizer.step()
    return SharedBundle(tuple(features), preprocessor, hidden_units, model.state_dict())


def evaluate_shared_candidate(
    frames: Mapping[str, pd.DataFrame],
    assignments: Mapping[str, Mapping[str, int]],
    features: Sequence[str],
    *,
    hidden_units: int = 16,
    epochs: int = 32,
    learning_rate: float = 0.01,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    parts = {task: [] for task in TASKS}
    for inner_fold in range(5):
        train: dict[str, pd.DataFrame] = {}
        validation: dict[str, pd.DataFrame] = {}
        for task in TASKS:
            fold = frames[task]["global_participant_id"].map(assignments[task])
            if fold.isna().any():
                raise Opt002Error("shared candidate inner split is incomplete")
            train[task] = frames[task].loc[fold.ne(inner_fold)].copy()
            validation[task] = frames[task].loc[fold.eq(inner_fold)].copy()
        bundle = fit_shared_bundle(
            train,
            features,
            hidden_units=hidden_units,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        for task in TASKS:
            part = validation[task][
                [
                    "global_participant_id",
                    "target_window_id",
                    "future_binary_target",
                    "outer_fold_id",
                ]
            ].copy()
            part["inner_fold_id"] = inner_fold
            part["p_raw"] = bundle.predict(validation[task], task)
            parts[task].append(part)
    outputs: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict[str, float]] = {}
    for task in TASKS:
        output = pd.concat(parts[task], ignore_index=True).sort_values(
            "target_window_id", kind="mergesort"
        )
        weights = participant_equal_weights(output)
        target = output["future_binary_target"].to_numpy(dtype="int8")
        probability = output["p_raw"].to_numpy(dtype="float64")
        outputs[task] = output
        metrics[task] = {
            "auprc": float(
                average_precision_score(target, probability, sample_weight=weights)
            ),
            "brier": float(np.average((probability - target) ** 2, weights=weights)),
        }
    return outputs, metrics


def select_best(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    passed = [row for row in records if row.get("status") == "passed"]
    if not passed:
        raise Opt002Error("candidate stage has no passing result")
    return min(
        passed,
        key=lambda row: (-float(row["auprc"]), float(row["brier"]), str(row["candidate_id"])),
    )


def fast_select_threshold(
    target: Sequence[int], probability: Sequence[float], weights: Sequence[float]
) -> tuple[float, dict[str, Any]]:
    """Select the frozen Macro-F1 threshold with a cumulative O(n log n) scan."""

    y = np.asarray(target, dtype="int8")
    p = np.asarray(probability, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    if len(y) == 0 or not np.isfinite(p).all() or not np.isfinite(w).all():
        raise Opt002Error("threshold inputs are empty or non-finite")
    order = np.argsort(-p, kind="mergesort")
    y, p, w = y[order], p[order], w[order]
    total_positive = float(w[y == 1].sum())
    total_negative = float(w[y == 0].sum())
    tp = fp = 0.0
    candidates: list[tuple[float, float, float]] = []

    def add(threshold: float) -> None:
        fn = total_positive - tp
        tn = total_negative - fp
        positive_denominator = 2.0 * tp + fp + fn
        negative_denominator = 2.0 * tn + fp + fn
        positive_f1 = 2.0 * tp / positive_denominator if positive_denominator else 0.0
        negative_f1 = 2.0 * tn / negative_denominator if negative_denominator else 0.0
        macro = 0.5 * (positive_f1 + negative_f1)
        sensitivity = tp / total_positive if total_positive else 0.0
        candidates.append((float(threshold), float(macro), float(sensitivity)))

    if not np.isclose(p[0], 1.0):
        add(1.0)
    index = 0
    while index < len(y):
        end = index + 1
        while end < len(y) and p[end] == p[index]:
            end += 1
        group_y, group_w = y[index:end], w[index:end]
        tp += float(group_w[group_y == 1].sum())
        fp += float(group_w[group_y == 0].sum())
        add(float(p[index]))
        index = end
    if not np.isclose(p[-1], 0.0):
        add(0.0)
    selected = min(candidates, key=lambda row: (-row[1], -row[2], row[0]))
    return selected[0], {
        "metric": "participant_equal_macro_f1",
        "operator": "greater_than_or_equal",
        "selected_threshold": selected[0],
        "selected_macro_f1": selected[1],
        "selected_sensitivity": selected[2],
        "candidate_count": len(candidates),
        "algorithm": "tie_preserving_descending_cumulative_scan",
    }


__all__ = [
    "AUXILIARY_MODES",
    "Candidate",
    "FAMILIES",
    "Opt002Error",
    "SharedBundle",
    "TASKS",
    "WEIGHT_SCHEMES",
    "evaluate_candidate",
    "evaluate_shared_candidate",
    "feature_groups",
    "fit_final_binary_bundle",
    "fit_shared_bundle",
    "fast_select_threshold",
    "select_best",
]
