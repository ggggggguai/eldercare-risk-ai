"""Frozen shared contract utilities for FORECAST-OPT-005."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EPSILON = 1e-12
PROBABILITY_CLIP = (1e-6, 1.0 - 1e-6)


class Opt005ContractError(ValueError):
    """Raised when a frozen OPT005 invariant is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, default=_json_default
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def write_sha256s(root: Path) -> None:
    rows = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (root / "SHA256SUMS").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )


def verify_sha256s(root: Path) -> dict[str, Any]:
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        raise Opt005ContractError(f"SHA256SUMS missing: {root}")
    checked = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise Opt005ContractError(f"checksum mismatch: {path}")
        checked += 1
    return {"root": str(root), "checked": checked, "all_passed": True}


def participant_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    if "global_participant_id" not in frame:
        raise Opt005ContractError("global_participant_id is required")
    counts = frame.groupby("global_participant_id", sort=False)[
        "global_participant_id"
    ].transform("size")
    weights = 1.0 / counts.to_numpy(dtype="float64")
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise Opt005ContractError("participant weights are invalid")
    return weights / total


def validate_probability(probability: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(probability), dtype="float64")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise Opt005ContractError("probability must be a finite vector")
    if ((values < 0) | (values > 1)).any():
        raise Opt005ContractError("probability is outside [0,1]")
    return values


def weighted_ece(
    labels: Iterable[int], probability: Iterable[float], weights: Iterable[float]
) -> float:
    y = np.asarray(list(labels), dtype="int8")
    p = validate_probability(probability)
    w = np.asarray(list(weights), dtype="float64")
    if len(y) != len(p) or len(p) != len(w) or not np.isfinite(w).all():
        raise Opt005ContractError("ECE vectors are misaligned")
    w = w / w.sum()
    bins = np.minimum((p * 10).astype("int64"), 9)
    result = 0.0
    for bin_id in range(10):
        mask = bins == bin_id
        if not mask.any():
            continue
        mass = float(w[mask].sum())
        confidence = float(np.average(p[mask], weights=w[mask]))
        observed = float(np.average(y[mask], weights=w[mask]))
        result += mass * abs(confidence - observed)
    return float(result)


def metric_bundle(
    frame: pd.DataFrame,
    probability: Iterable[float],
    *,
    ranking_score: Iterable[float] | None = None,
) -> dict[str, float]:
    y = frame["future_binary_target"].to_numpy(dtype="int8")
    p = validate_probability(probability)
    score = (
        p if ranking_score is None else np.asarray(list(ranking_score), dtype="float64")
    )
    if len(y) != len(p) or len(score) != len(y) or not np.isfinite(score).all():
        raise Opt005ContractError("metric vectors are misaligned")
    weights = participant_equal_weights(frame)
    return {
        "auprc": float(average_precision_score(y, score, sample_weight=weights)),
        "auroc": float(roc_auc_score(y, score, sample_weight=weights)),
        "brier": float(np.sum(weights * np.square(p - y))),
        "ece": weighted_ece(y, p, weights),
    }


def select_macro_f1_threshold(
    frame: pd.DataFrame, probability: Iterable[float]
) -> float:
    p = validate_probability(probability)
    y = frame["future_binary_target"].to_numpy(dtype="int8")
    w = participant_equal_weights(frame)
    candidates = np.unique(np.concatenate(([0.0], p, [1.0])))
    best_value = -1.0
    best_threshold = 1.0
    for threshold in candidates:
        pred = p >= threshold
        f1s: list[float] = []
        for cls in (0, 1):
            tp = float(w[(pred == cls) & (y == cls)].sum())
            fp = float(w[(pred == cls) & (y != cls)].sum())
            fn = float(w[(pred != cls) & (y == cls)].sum())
            denom = 2 * tp + fp + fn
            f1s.append(0.0 if denom <= 0 else 2 * tp / denom)
        value = float(np.mean(f1s))
        if value > best_value + EPSILON or (
            abs(value - best_value) <= EPSILON and threshold > best_threshold
        ):
            best_value = value
            best_threshold = float(threshold)
    return best_threshold


def classification_metrics(
    frame: pd.DataFrame, probability: Iterable[float], threshold: float
) -> dict[str, float]:
    p = validate_probability(probability)
    y = frame["future_binary_target"].to_numpy(dtype="int8")
    pred = p >= float(threshold)
    w = participant_equal_weights(frame)
    sensitivity = float(w[(pred == 1) & (y == 1)].sum() / w[y == 1].sum())
    specificity = float(w[(pred == 0) & (y == 0)].sum() / w[y == 0].sum())
    f1s = []
    for cls in (0, 1):
        tp = float(w[(pred == cls) & (y == cls)].sum())
        fp = float(w[(pred == cls) & (y != cls)].sum())
        fn = float(w[(pred != cls) & (y == cls)].sum())
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom <= 0 else 2 * tp / denom)
    return {
        "threshold": float(threshold),
        "macro_f1": float(np.mean(f1s)),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def assert_unique_keys(frame: pd.DataFrame, keys: list[str]) -> None:
    missing = [name for name in keys if name not in frame]
    if missing:
        raise Opt005ContractError(f"missing key columns: {missing}")
    if frame.duplicated(keys).any():
        raise Opt005ContractError(f"duplicate keys: {keys}")


def promotion_checks(
    candidate: dict[str, float],
    baseline: dict[str, float],
    *,
    fold_deltas: Iterable[float],
    bootstrap_delta_median: float,
) -> dict[str, bool]:
    deltas = np.asarray(list(fold_deltas), dtype="float64")
    return {
        "auprc_delta_ge_0_005": candidate["auprc"] - baseline["auprc"]
        >= 0.005 - EPSILON,
        "auroc_drop_within_0_005": baseline["auroc"] - candidate["auroc"]
        <= 0.005 + EPSILON,
        "brier_increase_within_0_002": candidate["brier"] - baseline["brier"]
        <= 0.002 + EPSILON,
        "ece_absolute_le_0_03": candidate["ece"] <= 0.03 + EPSILON,
        "ece_increase_within_0_005": candidate["ece"] - baseline["ece"]
        <= 0.005 + EPSILON,
        "folds_non_decrease_ge_3": int((deltas >= -EPSILON).sum()) >= 3,
        "worst_fold_drop_within_0_02": float(deltas.min()) >= -0.02 - EPSILON,
        "bootstrap_delta_median_positive": float(bootstrap_delta_median) > 0,
    }


__all__ = [
    "EPSILON",
    "Opt005ContractError",
    "PROBABILITY_CLIP",
    "assert_unique_keys",
    "classification_metrics",
    "metric_bundle",
    "participant_equal_weights",
    "promotion_checks",
    "select_macro_f1_threshold",
    "sha256_file",
    "validate_probability",
    "verify_sha256s",
    "weighted_ece",
    "write_json",
    "write_sha256s",
]
