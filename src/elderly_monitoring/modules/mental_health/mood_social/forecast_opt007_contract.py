"""Frozen helpers for the FORECAST-OPT-007 post-selection development round.

The module intentionally contains no outer-evaluation entry point.  It provides
the deterministic profile definitions, deployment-safe residual transforms and
the shared run/evidence guards used by OPT007A--G.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .forecast_opt005_contract import participant_equal_weights


PROFILE_CORE = ("birthyear", "sex", "educ")
PROFILE_CONDITIONAL = ("bmi",)
COMORBIDITY_FIELDS = (
    "comorbid_cancer",
    "comorbid_diabetes_typ1",
    "comorbid_diabetes_typ2",
    "comorbid_gout",
    "comorbid_migraines",
    "comorbid_ms",
    "comorbid_osteoporosis",
    "comorbid_neuropathic",
    "comorbid_arthritis",
)

# Small, theory-led list.  Every item is deployable and belongs to the frozen
# 156-field passive contract; no automatic Cartesian expansion is allowed.
INTERACTION_SPECS: tuple[dict[str, str], ...] = (
    {"name": "sex_x_sleep_duration", "left": "sex", "right": "short__sleep_asleep_mean_recent__anchor", "reason": "sex-specific sleep duration association"},
    {"name": "sex_x_sleep_efficiency", "left": "sex", "right": "short__sleep_ratio_asleep_in_bed_mean_recent__anchor", "reason": "sex-specific sleep efficiency association"},
    {"name": "sex_x_sleep_disruption", "left": "sex", "right": "opt003__sleep__delta_1__abs_mean", "reason": "sex-specific response to sleep disruption"},
    {"name": "birthyear_x_activity", "left": "birthyear", "right": "short__steps_awake_mean__anchor", "reason": "age-related activity baseline"},
    {"name": "birthyear_x_activity_drop", "left": "birthyear", "right": "opt003__activity__delta_1__mean", "reason": "age-related activity decline"},
    {"name": "birthyear_x_activity_slope", "left": "birthyear", "right": "opt003__activity__local_slope__mean", "reason": "age-related activity trajectory"},
    {"name": "educ_x_missingness", "left": "educ", "right": "opt003__anchor__missing_fraction", "reason": "profile-dependent sensor completeness"},
    {"name": "educ_x_history", "left": "educ", "right": "history_observed_month_count", "reason": "profile-dependent history coverage"},
    {"name": "educ_x_sleep_irregularity", "left": "educ", "right": "short__sleep_main_start_hour_adj_iqr__anchor", "reason": "sleep regularity interaction"},
    {"name": "sex_x_activity_drop", "left": "sex", "right": "opt003__activity__delta_1__mean", "reason": "sex-specific activity decline"},
    {"name": "bmi_x_activity", "left": "bmi", "right": "short__steps_awake_mean__anchor", "reason": "mobility burden conditional on BMI"},
    {"name": "bmi_x_sleep_duration", "left": "bmi", "right": "short__sleep_asleep_mean_recent__anchor", "reason": "sleep duration conditional on BMI"},
)

RUN_IDS = {
    "FORECAST-OPT-007A": "MH-20260809-FOPT-062",
    "FORECAST-OPT-007B": "MH-20260809-FOPT-063",
    "FORECAST-OPT-007C": "MH-20260809-FOPT-064",
    "FORECAST-OPT-007D": "MH-20260809-FOPT-065",
    "FORECAST-OPT-007E": "MH-20260809-FOPT-066",
    "FORECAST-OPT-007F": "MH-20260809-FOPT-067",
    "FORECAST-OPT-007G": "MH-20260809-FOPT-070",
}


class Opt007ContractError(ValueError):
    """Raised when a frozen OPT007 invariant is violated."""


def safe_logit(probability: Sequence[float]) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def sigmoid(values: Sequence[float]) -> np.ndarray:
    z = np.clip(np.asarray(values, dtype="float64"), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def add_profile_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Add only the pre-frozen profile/passive interactions that are available."""
    result = frame.copy()
    for spec in INTERACTION_SPECS:
        left, right = spec["left"], spec["right"]
        if left not in result or right not in result:
            continue
        result[spec["name"]] = pd.to_numeric(result[left], errors="coerce") * pd.to_numeric(result[right], errors="coerce")
    return result


def comorbidity_burden(frame: pd.DataFrame) -> pd.Series:
    """Count diagnosed categories, not raw severity sums, across nine fields."""
    missing = sorted(set(COMORBIDITY_FIELDS) - set(frame.columns))
    if missing:
        raise Opt007ContractError(f"comorbidity fields missing: {missing}")
    numeric = frame.loc[:, list(COMORBIDITY_FIELDS)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise Opt007ContractError("comorbidity burden has unresolved missing values")
    return numeric.gt(0).sum(axis=1).astype("float64")


def fit_numeric_transform(train: pd.DataFrame, features: Sequence[str]) -> dict[str, Any]:
    x = train.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    medians = x.median(axis=0).fillna(0.0)
    filled = x.fillna(medians)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    return {
        "features": list(features),
        "medians": medians.to_numpy(dtype="float64"),
        "means": means.to_numpy(dtype="float64"),
        "scales": scales.to_numpy(dtype="float64"),
    }


def apply_numeric_transform(frame: pd.DataFrame, state: dict[str, Any]) -> np.ndarray:
    features = list(state["features"])
    x = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    medians = np.asarray(state["medians"], dtype="float64")
    means = np.asarray(state["means"], dtype="float64")
    scales = np.asarray(state["scales"], dtype="float64")
    x = np.where(np.isfinite(x), x, medians)
    return (x - means) / scales


def fit_offset_logistic(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: Sequence[str],
    train_reference_probability: Sequence[float],
    valid_reference_probability: Sequence[float],
    *,
    l2: float = 4.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a fixed-coefficient logit offset plus a strongly shrunk residual."""
    state = fit_numeric_transform(train, features)
    x_train = apply_numeric_transform(train, state)
    x_valid = apply_numeric_transform(valid, state)
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_valid = np.column_stack([np.ones(len(x_valid)), x_valid])
    y = train["future_binary_target"].to_numpy(dtype="float64")
    weights = participant_equal_weights(train)
    offset = safe_logit(train_reference_probability)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = offset + x_train @ beta
        p = sigmoid(eta)
        loss = -np.sum(weights * (y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))
        penalty = 0.5 * l2 * float(np.dot(beta[1:], beta[1:]))
        gradient = x_train.T @ (weights * (p - y))
        gradient[1:] += l2 * beta[1:]
        return float(loss + penalty), gradient

    initial = np.zeros(x_train.shape[1], dtype="float64")
    fitted = minimize(objective, initial, jac=True, method="L-BFGS-B", options={"maxiter": 300, "ftol": 1e-10})
    if not fitted.success:
        raise Opt007ContractError(f"offset logistic failed: {fitted.message}")
    state["coef"] = fitted.x
    state["l2"] = float(l2)
    state["optimizer_success"] = True
    probability = sigmoid(safe_logit(valid_reference_probability) + x_valid @ fitted.x)
    return probability, state


def predict_offset_logistic(frame: pd.DataFrame, reference_probability: Sequence[float], state: dict[str, Any]) -> np.ndarray:
    x = apply_numeric_transform(frame, state)
    x = np.column_stack([np.ones(len(x)), x])
    return sigmoid(safe_logit(reference_probability) + x @ np.asarray(state["coef"], dtype="float64"))


def serializable_offset_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in state.items()
    }


def assert_no_existing_outer_access(paths: Sequence[Path]) -> None:
    """Refuse the two spent OPT006 outer sources in candidate-selection code."""
    forbidden = {"MH-20260809-FOPT-053", "MH-20260809-FOPT-059"}
    for path in paths:
        if forbidden.intersection(path.parts):
            raise Opt007ContractError(f"spent outer path is forbidden for selection: {path}")


def initialize_run_layout(output: Path, task_id: str) -> None:
    if output.exists():
        raise FileExistsError(f"immutable run exists: {output}")
    output.mkdir(parents=True)
    for name in ("logs", "metrics", "predictions", "audit"):
        (output / name).mkdir()
    (output / "logs" / "run.log").write_text(f"{task_id} initialized\n", encoding="utf-8", newline="\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "COMORBIDITY_FIELDS",
    "INTERACTION_SPECS",
    "Opt007ContractError",
    "PROFILE_CONDITIONAL",
    "PROFILE_CORE",
    "RUN_IDS",
    "add_profile_interactions",
    "apply_numeric_transform",
    "assert_no_existing_outer_access",
    "comorbidity_burden",
    "fit_numeric_transform",
    "fit_offset_logistic",
    "initialize_run_layout",
    "load_json",
    "predict_offset_logistic",
    "safe_logit",
    "serializable_offset_state",
    "sigmoid",
]
