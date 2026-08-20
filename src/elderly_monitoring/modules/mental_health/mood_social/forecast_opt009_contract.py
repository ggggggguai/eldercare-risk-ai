"""Frozen contracts for the final FORECAST-OPT-009 development round.

The module intentionally exposes no outer-evaluation entry point.  Candidate
selection is strict-inner only; historical outer artifacts may be hashed and
quoted as retained evidence, but are rejected by the selection-path guard.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score

from .forecast_opt007_contract import safe_logit, sigmoid


RUN_IDS = {
    "FORECAST-OPT-009A": "MH-20260810-FOPT-078",
    "FORECAST-OPT-009B": "MH-20260810-FOPT-079",
    "FORECAST-OPT-009C": "MH-20260810-FOPT-080",
    "FORECAST-OPT-009D": "MH-20260810-FOPT-081",
    "FORECAST-OPT-009E": "MH-20260810-FOPT-082",
    "FORECAST-OPT-009F": "MH-20260810-FOPT-083",
    "FORECAST-OPT-009G": "MH-20260810-FOPT-084",
}

KEYS = [
    "global_participant_id",
    "target_window_id",
    "task_id",
    "evaluation_outer_fold",
]

GROUP_FAMILIES = (
    "sex",
    "birthyear_quartile",
    "educ",
    "label_month_slot",
    "short_history",
    "low_completeness",
)

SPENT_OUTER_RUNS = {
    "MH-20260808-FOPT-013",
    "MH-20260809-FOPT-025",
    "MH-20260809-FOPT-026",
    "MH-20260809-FOPT-044",
    "MH-20260809-FOPT-053",
    "MH-20260809-FOPT-059",
}

OUTER_FILE_NAMES = {
    "outer_oof_predictions.parquet",
    "state_aware_outer_oof_predictions.parquet",
    "source_outer_independent_predictions.parquet",
    "final_nested_oof_predictions.parquet",
}


class Opt009ContractError(ValueError):
    """Raised when a frozen OPT009 invariant is violated."""


def initialize_run_layout(output: Path, task_id: str) -> None:
    """Create the immutable minimum run layout."""
    if output.exists():
        raise FileExistsError(f"immutable run exists: {output}")
    output.mkdir(parents=True)
    for name in ("logs", "metrics", "predictions", "audit", "reports"):
        (output / name).mkdir()
    (output / "logs" / "run.log").write_text(
        f"{task_id} initialized\n", encoding="utf-8", newline="\n"
    )


def tree_fingerprint(path: Path, *, include: Iterable[Path] | None = None) -> dict[str, Any]:
    """Return a deterministic path-and-content fingerprint."""
    if not path.exists():
        raise FileNotFoundError(path)
    candidates = [path] if path.is_file() else sorted(path.rglob("*"))
    if include is not None:
        candidates = sorted(include)
    digest = hashlib.sha256()
    count = 0
    total = 0
    for item in candidates:
        if not item.is_file() or ".git" in item.parts or "__pycache__" in item.parts:
            continue
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                total += len(chunk)
        count += 1
    return {"path": str(path), "files": count, "bytes": total, "sha256": digest.hexdigest()}


def assert_selection_paths_are_inner_only(paths: Sequence[Path]) -> None:
    """Reject every known spent outer source from B--F selection paths."""
    for path in paths:
        lower_name = path.name.lower()
        if SPENT_OUTER_RUNS.intersection(path.parts):
            raise Opt009ContractError(f"spent outer run is forbidden for selection: {path}")
        if path.name in OUTER_FILE_NAMES or "outer_independent" in lower_name:
            raise Opt009ContractError(f"spent outer evidence is forbidden for selection: {path}")


def validate_exact_keys(frame: pd.DataFrame, expected: pd.DataFrame | None = None) -> None:
    """Require the frozen four-column unique key and exact optional coverage."""
    missing = sorted(set(KEYS) - set(frame.columns))
    if missing:
        raise Opt009ContractError(f"missing exact keys: {missing}")
    if frame[KEYS].isna().any().any():
        raise Opt009ContractError("null OPT009 prediction key")
    if frame.duplicated(KEYS).any():
        raise Opt009ContractError("duplicate OPT009 prediction key")
    if expected is not None:
        left = frame[KEYS].sort_values(KEYS).reset_index(drop=True)
        right = expected[KEYS].sort_values(KEYS).reset_index(drop=True)
        if not left.equals(right):
            raise Opt009ContractError("prediction keys or coverage changed")


def assert_participant_isolation(train: pd.DataFrame, valid: pd.DataFrame) -> None:
    overlap = set(train["global_participant_id"]) & set(valid["global_participant_id"])
    if overlap:
        raise Opt009ContractError(f"participant leakage: {len(overlap)} participants")


def participant_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    """Return weights summing to one with equal total mass per participant."""
    if frame.empty:
        return np.zeros(0, dtype="float64")
    counts = frame.groupby("global_participant_id", sort=False)["global_participant_id"].transform("size")
    participants = frame["global_participant_id"].nunique()
    return 1.0 / (float(participants) * counts.to_numpy(dtype="float64"))


def participant_normalized_training_weights(
    frame: pd.DataFrame, multipliers: Sequence[float]
) -> np.ndarray:
    """Apply row multipliers while retaining exactly 1/P mass per participant."""
    multiplier = np.asarray(multipliers, dtype="float64")
    if len(multiplier) != len(frame) or not np.isfinite(multiplier).all() or (multiplier <= 0).any():
        raise Opt009ContractError("invalid training multiplier")
    work = pd.DataFrame(
        {
            "participant": frame["global_participant_id"].to_numpy(),
            "mass": participant_equal_weights(frame) * multiplier,
        },
        index=frame.index,
    )
    within = work.groupby("participant", sort=False)["mass"].transform("sum").to_numpy()
    participants = frame["global_participant_id"].nunique()
    weights = work["mass"].to_numpy() / within / float(participants)
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise Opt009ContractError("training weights do not sum to one")
    totals = pd.Series(weights, index=frame.index).groupby(frame["global_participant_id"]).sum()
    if not np.allclose(totals.to_numpy(), 1.0 / participants, atol=1e-12):
        raise Opt009ContractError("participant totals are not equal")
    return weights


def weighted_average_precision(frame: pd.DataFrame, probability: Sequence[float]) -> float:
    return float(
        average_precision_score(
            frame["future_binary_target"].to_numpy(dtype="int8"),
            np.asarray(probability, dtype="float64"),
            sample_weight=participant_equal_weights(frame),
        )
    )


def weighted_brier(frame: pd.DataFrame, probability: Sequence[float]) -> float:
    p = np.asarray(probability, dtype="float64")
    y = frame["future_binary_target"].to_numpy(dtype="float64")
    return float(np.sum(participant_equal_weights(frame) * (p - y) ** 2))


def weighted_log_loss(frame: pd.DataFrame, probability: Sequence[float]) -> float:
    p = np.clip(np.asarray(probability, dtype="float64"), 1e-6, 1 - 1e-6)
    y = frame["future_binary_target"].to_numpy(dtype="float64")
    w = participant_equal_weights(frame)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log1p(-p))))


def weighted_ece(frame: pd.DataFrame, probability: Sequence[float]) -> float:
    """Participant-equal ECE with ten fixed equal-width probability bins."""
    p = np.asarray(probability, dtype="float64")
    y = frame["future_binary_target"].to_numpy(dtype="float64")
    w = participant_equal_weights(frame)
    bins = np.minimum((np.clip(p, 0.0, 1.0) * 10).astype(int), 9)
    value = 0.0
    for index in range(10):
        mask = bins == index
        mass = float(w[mask].sum())
        if mass == 0:
            continue
        value += mass * abs(float(np.sum(w[mask] * p[mask]) / mass) - float(np.sum(w[mask] * y[mask]) / mass))
    return float(value)


def train_only_birthyear_quartiles(
    train_birthyear: Sequence[float], values: Sequence[float]
) -> tuple[np.ndarray, list[float]]:
    """Apply frozen linear quantiles and left-sided searchsorted encoding."""
    train = np.asarray(train_birthyear, dtype="float64")
    finite = train[np.isfinite(train)]
    if not len(finite):
        raise Opt009ContractError("birthyear training split has no finite values")
    boundaries = np.quantile(finite, [0.25, 0.5, 0.75], method="linear")
    raw = np.asarray(values, dtype="float64")
    result = np.full(len(raw), "__MISSING__", dtype=object)
    present = np.isfinite(raw)
    result[present] = [f"Q{item + 1}" for item in np.searchsorted(boundaries, raw[present], side="left")]
    return result, boundaries.astype(float).tolist()


def fit_scale_corrected_platt(frame: pd.DataFrame, probability: Sequence[float]) -> dict[str, Any]:
    """Fit the one frozen monotone scale-corrected Platt map."""
    x = safe_logit(np.clip(np.asarray(probability, dtype="float64"), 1e-6, 1 - 1e-6))
    y = frame["future_binary_target"].to_numpy(dtype="float64")
    w = participant_equal_weights(frame)
    n_eff = 1.0 / float(np.sum(w**2))

    def objective(theta: np.ndarray) -> float:
        alpha, gamma = theta
        beta = np.exp(gamma)
        p = sigmoid(alpha + beta * x)
        loss = -np.sum(w * (y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))
        return float(loss + (beta - 1.0) ** 2 / n_eff)

    fitted = minimize(
        objective,
        np.array([0.0, 0.0], dtype="float64"),
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise Opt009ContractError(f"scale-corrected Platt failed: {fitted.message}")
    return {
        "kind": "scale_corrected_platt",
        "alpha": float(fitted.x[0]),
        "gamma": float(fitted.x[1]),
        "beta": float(np.exp(fitted.x[1])),
        "n_eff": n_eff,
        "optimizer_success": True,
    }


def apply_scale_corrected_platt(probability: Sequence[float], state: dict[str, Any]) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype="float64"), 1e-6, 1 - 1e-6)
    return sigmoid(float(state["alpha"]) + float(state["beta"]) * safe_logit(p))


def batch_invariance(probability: Sequence[float], state: dict[str, Any]) -> float:
    values = np.asarray(probability, dtype="float64")
    batch = apply_scale_corrected_platt(values, state)
    single = np.array([apply_scale_corrected_platt([value], state)[0] for value in values])
    return float(np.max(np.abs(batch - single))) if len(values) else 0.0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "GROUP_FAMILIES",
    "KEYS",
    "OUTER_FILE_NAMES",
    "RUN_IDS",
    "SPENT_OUTER_RUNS",
    "Opt009ContractError",
    "apply_scale_corrected_platt",
    "assert_participant_isolation",
    "assert_selection_paths_are_inner_only",
    "batch_invariance",
    "fit_scale_corrected_platt",
    "initialize_run_layout",
    "load_json",
    "participant_equal_weights",
    "participant_normalized_training_weights",
    "train_only_birthyear_quartiles",
    "tree_fingerprint",
    "validate_exact_keys",
    "weighted_average_precision",
    "weighted_brier",
    "weighted_ece",
    "weighted_log_loss",
]
