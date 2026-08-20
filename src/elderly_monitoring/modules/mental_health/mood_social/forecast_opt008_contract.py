"""Frozen contracts for FORECAST-OPT-008.

OPT008 is a development-only, strict-inner round.  This module deliberately
contains no outer-evaluation helper.  All selection code must pass paths
through :func:`assert_selection_paths_are_inner_only`.
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

from .forecast_opt005_contract import participant_equal_weights
from .forecast_opt007_contract import safe_logit, sigmoid


RUN_IDS = {
    "FORECAST-OPT-008A": "MH-20260810-FOPT-071",
    "FORECAST-OPT-008B": "MH-20260810-FOPT-072",
    "FORECAST-OPT-008C": "MH-20260810-FOPT-073",
    "FORECAST-OPT-008D": "MH-20260810-FOPT-074",
    "FORECAST-OPT-008E": "MH-20260810-FOPT-075",
    "FORECAST-OPT-008F": "MH-20260810-FOPT-076",
    "FORECAST-OPT-008G": "MH-20260810-FOPT-077",
}

KEYS = [
    "task_id",
    "evaluation_outer_fold",
    "global_participant_id",
    "target_window_id",
]

SPENT_OUTER_RUNS = {
    "MH-20260808-FOPT-013",
    "MH-20260809-FOPT-026",
    "MH-20260809-FOPT-044",
    "MH-20260809-FOPT-053",
    "MH-20260809-FOPT-059",
}


class Opt008ContractError(ValueError):
    """Raised when a frozen OPT008 invariant is violated."""


def initialize_run_layout(output: Path, task_id: str) -> None:
    """Create an immutable run layout; existing run IDs are never overwritten."""
    if output.exists():
        raise FileExistsError(f"immutable run exists: {output}")
    output.mkdir(parents=True)
    for name in ("logs", "metrics", "predictions", "audit", "reports"):
        (output / name).mkdir()
    (output / "logs" / "run.log").write_text(
        f"{task_id} initialized\n", encoding="utf-8", newline="\n"
    )


def tree_fingerprint(path: Path, *, include: Iterable[Path] | None = None) -> dict[str, Any]:
    """Return a deterministic content fingerprint for a directory or file set."""
    if include is None:
        candidates = [path] if path.is_file() else sorted(path.rglob("*"))
    else:
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
    return {
        "path": str(path),
        "files": count,
        "bytes": total,
        "sha256": digest.hexdigest(),
    }


def assert_selection_paths_are_inner_only(paths: Sequence[Path]) -> None:
    """Reject spent outer artifacts in any B--F selection path."""
    forbidden_names = {
        "outer_oof_predictions.parquet",
        "state_aware_outer_oof_predictions.parquet",
        "final_nested_oof_predictions.parquet",
    }
    for path in paths:
        if SPENT_OUTER_RUNS.intersection(path.parts):
            raise Opt008ContractError(f"spent outer run is forbidden for selection: {path}")
        if path.name in forbidden_names or "outer_independent" in path.name:
            raise Opt008ContractError(f"spent outer evidence is forbidden for selection: {path}")


def validate_exact_keys(frame: pd.DataFrame, expected: pd.DataFrame | None = None) -> None:
    missing = sorted(set(KEYS) - set(frame.columns))
    if missing:
        raise Opt008ContractError(f"missing exact keys: {missing}")
    if frame.duplicated(KEYS).any():
        raise Opt008ContractError("duplicate OPT008 prediction keys")
    if expected is not None:
        left = frame[KEYS].sort_values(KEYS).reset_index(drop=True)
        right = expected[KEYS].sort_values(KEYS).reset_index(drop=True)
        if not left.equals(right):
            raise Opt008ContractError("prediction keys or coverage changed")


def assert_participant_isolation(train: pd.DataFrame, valid: pd.DataFrame) -> None:
    overlap = set(train["global_participant_id"]) & set(valid["global_participant_id"])
    if overlap:
        raise Opt008ContractError(f"participant leakage: {len(overlap)} participants")


def fit_platt(
    frame: pd.DataFrame, probability: Sequence[float], *, l2: float = 1.0
) -> dict[str, Any]:
    """Fit a monotone Platt map with participant-equal weighting."""
    x = safe_logit(probability)
    y = frame["future_binary_target"].to_numpy(dtype="float64")
    weights = participant_equal_weights(frame)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = beta[0] + beta[1] * x
        p = sigmoid(eta)
        loss = -np.sum(
            weights
            * (y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))
        )
        penalty = 0.5 * l2 * float(beta[1] ** 2)
        gradient = np.array(
            [np.sum(weights * (p - y)), np.sum(weights * (p - y) * x) + l2 * beta[1]],
            dtype="float64",
        )
        return float(loss + penalty), gradient

    fitted = minimize(
        objective,
        np.array([0.0, 1.0]),
        jac=True,
        method="L-BFGS-B",
        bounds=[(None, None), (1e-8, None)],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not fitted.success:
        raise Opt008ContractError(f"Platt optimization failed: {fitted.message}")
    return {"kind": "platt", "intercept": float(fitted.x[0]), "slope": float(fitted.x[1]), "l2": l2}


def fit_beta(
    frame: pd.DataFrame, probability: Sequence[float], *, l2: float = 4.0
) -> dict[str, Any]:
    """Fit a monotone L2 Beta calibration map.

    eta = c + a*log(p) + b*(-log(1-p)); non-negative a,b guarantee
    monotonicity and therefore preserve the exact AUPRC ordering.
    """
    p0 = np.clip(np.asarray(probability, dtype="float64"), 1e-6, 1 - 1e-6)
    x = np.column_stack([np.ones(len(p0)), np.log(p0), -np.log1p(-p0)])
    y = frame["future_binary_target"].to_numpy(dtype="float64")
    weights = participant_equal_weights(frame)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        p = sigmoid(x @ beta)
        loss = -np.sum(
            weights
            * (y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))
        )
        penalty = 0.5 * l2 * float(np.dot(beta[1:] - 1.0, beta[1:] - 1.0))
        gradient = x.T @ (weights * (p - y))
        gradient[1:] += l2 * (beta[1:] - 1.0)
        return float(loss + penalty), gradient

    fitted = minimize(
        objective,
        np.array([0.0, 1.0, 1.0]),
        jac=True,
        method="L-BFGS-B",
        bounds=[(None, None), (1e-8, None), (1e-8, None)],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not fitted.success:
        raise Opt008ContractError(f"Beta calibration failed: {fitted.message}")
    return {
        "kind": "beta_l2",
        "intercept": float(fitted.x[0]),
        "log_p_coef": float(fitted.x[1]),
        "log_one_minus_p_coef": float(fitted.x[2]),
        "l2": l2,
    }


def apply_calibrator(probability: Sequence[float], state: dict[str, Any]) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype="float64"), 1e-6, 1 - 1e-6)
    if state["kind"] == "platt":
        return sigmoid(state["intercept"] + state["slope"] * safe_logit(p))
    if state["kind"] == "beta_l2":
        eta = (
            state["intercept"]
            + state["log_p_coef"] * np.log(p)
            + state["log_one_minus_p_coef"] * (-np.log1p(-p))
        )
        return sigmoid(eta)
    raise Opt008ContractError(f"unknown calibrator: {state.get('kind')}")


def batch_invariance(probability: Sequence[float], state: dict[str, Any]) -> float:
    values = np.asarray(probability, dtype="float64")
    batch = apply_calibrator(values, state)
    single = np.array([apply_calibrator([value], state)[0] for value in values])
    return float(np.max(np.abs(batch - single))) if len(values) else 0.0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "KEYS",
    "RUN_IDS",
    "SPENT_OUTER_RUNS",
    "Opt008ContractError",
    "apply_calibrator",
    "assert_participant_isolation",
    "assert_selection_paths_are_inner_only",
    "batch_invariance",
    "fit_beta",
    "fit_platt",
    "initialize_run_layout",
    "load_json",
    "tree_fingerprint",
    "validate_exact_keys",
]
