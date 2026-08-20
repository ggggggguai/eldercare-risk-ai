"""Locked small-budget S+History transition research for R11."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.calibration import FittedCalibration
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import predict_model
from elderly_monitoring.modules.mental_health.mood_social.r7.modeling import binary_metrics
from elderly_monitoring.modules.mental_health.mood_social.r9.contract import SLEEP_FEATURES
from elderly_monitoring.modules.mental_health.mood_social.r9.modeling import (
    inner_folds_for_outer,
    project_heads,
)
from elderly_monitoring.modules.mental_health.mood_social.r10.modeling import (
    FittedCandidate,
    _crossfit_calibration,
    _elastic,
    _fit_residual,
    _fit_simple,
    _fold_oof_simple,
    _residual_oof,
    apply_calibration,
    fit_final_calibration,
    participant_weight,
)

from .contract import (
    DEFAULT_DATA_RELATIVE,
    DEFAULT_REPORT_RELATIVE,
    R11_PROTOCOL_VERSION,
    repository_root,
    sha256_file,
    write_json,
)


TARGETS = {"ge5": "phq9_ge5_target", "ge10": "phq9_ge10_target"}
CANDIDATES = (
    "r9_joint_replay",
    "r10_unconditioned_residual",
    "prior_state_conditioned_residual",
)
CALIBRATION_METHODS = ("platt", "beta")
SELECTION_RELATIVE = DEFAULT_REPORT_RELATIVE / "selection"


def load_sleep_history(root: Path) -> pd.DataFrame:
    return pd.read_parquet(
        root / DEFAULT_DATA_RELATIVE / "signatures/sleep_history_repeated.parquet"
    )


def _conditioned_feature_names(head: str) -> tuple[str, ...]:
    return tuple(f"r11.{head}.state_x.{name}" for name in SLEEP_FEATURES)


def add_conditioned_features(frame: pd.DataFrame, head: str) -> pd.DataFrame:
    result = frame.copy()
    prior = pd.to_numeric(result[f"history.last_{head}"], errors="raise").astype(float)
    for source, target in zip(SLEEP_FEATURES, _conditioned_feature_names(head)):
        result[target] = pd.to_numeric(result[source], errors="coerce").astype(float) * prior
    return result


def conditioned_features(head: str) -> tuple[str, ...]:
    from elderly_monitoring.modules.mental_health.mood_social.r10.features import (
        SLEEP_HISTORY_FEATURES,
    )

    return tuple(SLEEP_HISTORY_FEATURES) + _conditioned_feature_names(head)


def _conditioned_oof(
    frame: pd.DataFrame, folds: pd.Series, head: str, seed: int
) -> np.ndarray:
    work = add_conditioned_features(frame, head)
    target = TARGETS[head]
    features = conditioned_features(head)
    result = pd.Series(np.nan, index=work.index, dtype=float)
    for fold in sorted(folds.astype(int).unique()):
        train = work.loc[folds.ne(fold)]
        validation = work.loc[folds.eq(fold)]
        model = _elastic(train, features, target, seed + int(fold))
        result.loc[validation.index] = predict_model(model, validation, features)
    if result.isna().any():
        raise ValueError(f"incomplete R11 conditioned OOF: {head}")
    return result.loc[work.index].to_numpy(float)


def candidate_inner_predictions(
    frame: pd.DataFrame, folds: pd.Series, head: str, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    target = TARGETS[head]
    baseline = _fold_oof_simple(frame, folds, "r9_joint_elastic_replay", head, seed)
    residual_candidates, residual_meta = _residual_oof(frame, folds, head, seed + 100)
    best_lambda = max(
        residual_candidates,
        key=lambda value: binary_metrics(
            frame[target], residual_candidates[value], weight=participant_weight(frame)
        )["auprc"],
    )
    conditioned = _conditioned_oof(frame, folds, head, seed + 200)
    return (
        {
            "r9_joint_replay": baseline,
            "r10_unconditioned_residual": residual_candidates[best_lambda],
            "prior_state_conditioned_residual": conditioned,
        },
        {
            "r9_joint_replay": {},
            "r10_unconditioned_residual": {
                "lambda": float(best_lambda),
                "residual_parameters": {
                    "intercept": float(residual_meta["intercept"]),
                    "slope": float(residual_meta["slope"]),
                },
            },
            "prior_state_conditioned_residual": {
                "prior_state": f"history.last_{head}",
                "interaction_count": len(SLEEP_FEATURES),
            },
        },
    )


def _calibration_selection(
    frame: pd.DataFrame, raw: np.ndarray, folds: pd.Series, target: str
) -> tuple[str, dict[str, Any]]:
    work = frame.copy()
    work["_selection_target"] = work[target].astype(int)
    metrics: dict[str, Any] = {}
    for method in CALIBRATION_METHODS:
        value = _crossfit_calibration(work, raw, folds, method)
        metrics[method] = binary_metrics(
            work[target], value, weight=participant_weight(work)
        )
    selected = min(
        CALIBRATION_METHODS,
        key=lambda method: metrics[method]["brier"] + 0.25 * metrics[method]["ece"],
    )
    return selected, metrics


def _safe_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(value)
    if "residual_parameters" in output:
        output["residual_parameters"] = {
            **output["residual_parameters"],
            "history_oof_crossfit": True,
            "sleep_oof_crossfit": True,
        }
    return output


def run_r11_transition_selection(
    *, repository_root_value: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    protocol_path = root / DEFAULT_DATA_RELATIVE / "protocol/r11_protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "runtime-feature-audit-complete-awaiting-sleep-history-selection":
        raise ValueError("R11 transition selection requires completed runtime audit")
    frame = load_sleep_history(root)
    lock: dict[str, Any] = {
        "protocol_version": R11_PROTOCOL_VERSION,
        "status": "locked",
        "selection_scope": "outer-train-inner-oof-only",
        "candidate_outer_opened": False,
        "candidates": list(CANDIDATES),
        "cancelled_candidates": ["four_state_causal_sleep_delta_conditional"],
        "nodes": {"sleep_history": {}},
    }
    ledger: list[dict[str, Any]] = []
    winner_deltas: list[float] = []
    for repeat, repeat_frame in frame.groupby("repeat", sort=True):
        repeat_lock: dict[str, Any] = {}
        for outer_fold in range(5):
            train = repeat_frame.loc[repeat_frame["outer_fold"].ne(outer_fold)].copy()
            folds = inner_folds_for_outer(repeat_frame, outer_fold).loc[train.index].astype(int)
            head_lock: dict[str, Any] = {}
            for head, target in TARGETS.items():
                probabilities, metadata = candidate_inner_predictions(
                    train, folds, head, 20261020 + int(repeat) * 1000 + outer_fold * 100
                )
                baseline_metrics = binary_metrics(
                    train[target], probabilities["r9_joint_replay"], weight=participant_weight(train)
                )
                ranked: list[tuple[float, str, dict[str, Any]]] = []
                for candidate in CANDIDATES:
                    raw_metrics = binary_metrics(
                        train[target], probabilities[candidate], weight=participant_weight(train)
                    )
                    method, calibration_metrics = _calibration_selection(
                        train, probabilities[candidate], folds, target
                    )
                    row = {
                        "node": "sleep_history",
                        "repeat": int(repeat),
                        "outer_fold": outer_fold,
                        "head": head,
                        "candidate_id": candidate,
                        "calibration": method,
                        "raw_metrics": raw_metrics,
                        "calibrated_metrics": calibration_metrics[method],
                        "calibration_candidates": calibration_metrics,
                        "inner_delta_auprc_vs_r9": raw_metrics["auprc"] - baseline_metrics["auprc"],
                        "metadata": _safe_metadata(metadata[candidate]),
                        "selection_data": "outer-train-inner-oof-only",
                    }
                    ledger.append(row)
                    ranked.append((float(raw_metrics["auprc"]), candidate, row))
                ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
                winner = ranked[0][2]
                winner_deltas.append(float(winner["inner_delta_auprc_vs_r9"]))
                head_lock[head] = {
                    "candidate_id": winner["candidate_id"],
                    "calibration": winner["calibration"],
                    "metadata": winner["metadata"],
                    "inner_metrics": winner["calibrated_metrics"],
                    "inner_raw_metrics": winner["raw_metrics"],
                    "inner_delta_auprc_vs_r9": winner["inner_delta_auprc_vs_r9"],
                    "candidate_count": len(ranked),
                    "runner_up": ranked[1][1],
                }
            repeat_lock[str(outer_fold)] = head_lock
        lock["nodes"]["sleep_history"][str(repeat)] = repeat_lock
    lock["mean_winner_inner_delta_auprc"] = float(np.mean(winner_deltas))
    lock["selection_lock_rule"] = "no candidates, features, seeds, folds or thresholds may be added"
    ledger_path = root / SELECTION_RELATIVE / "candidate_ledger.json"
    lock_path = root / SELECTION_RELATIVE / "selection_lock.json"
    write_json(ledger_path, ledger, overwrite=overwrite)
    write_json(lock_path, lock, overwrite=overwrite)
    contract_path = root / DEFAULT_REPORT_RELATIVE / "selection_lock_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["lock_status"] = "selection-locked-r11-outer-unopened"
    contract["selection_lock_sha256"] = sha256_file(lock_path)
    contract["candidate_ledger_sha256"] = sha256_file(ledger_path)
    write_json(contract_path, contract, overwrite=True)
    protocol["status"] = "selection-locked-r11-outer-unopened"
    protocol["selection_locked"] = True
    protocol["selection_lock_sha256"] = sha256_file(lock_path)
    protocol["tasks_completed"] = list(protocol["tasks_completed"]) + ["OPT-V333-R11-002-selection"]
    write_json(protocol_path, protocol, overwrite=True)
    return lock


@dataclass
class FittedR11Candidate:
    candidate_id: str
    head: str
    fitted: FittedCandidate | Any
    features: tuple[str, ...] = ()

    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        if self.candidate_id == "prior_state_conditioned_residual":
            work = add_conditioned_features(frame, self.head)
            return predict_model(self.fitted, work, self.features)
        return self.fitted.predict_raw(frame)


def fit_candidate(
    frame: pd.DataFrame,
    folds: pd.Series,
    head: str,
    selection: Mapping[str, Any],
    seed: int,
) -> tuple[FittedR11Candidate, np.ndarray]:
    candidate = str(selection["candidate_id"])
    probabilities, metadata = candidate_inner_predictions(frame, folds, head, seed)
    raw = probabilities[candidate]
    if candidate == "r9_joint_replay":
        fitted = _fit_simple(frame, "r9_joint_elastic_replay", head, seed + 500)
        return FittedR11Candidate(candidate, head, fitted), raw
    if candidate == "r10_unconditioned_residual":
        value = metadata[candidate]
        fitted = _fit_residual(
            frame,
            head,
            seed + 500,
            float(selection.get("metadata", value).get("lambda", value["lambda"])),
            value["residual_parameters"],
            "residual_platt",
        )
        return FittedR11Candidate(candidate, head, fitted), raw
    if candidate == "prior_state_conditioned_residual":
        work = add_conditioned_features(frame, head)
        features = conditioned_features(head)
        fitted = _elastic(work, features, TARGETS[head], seed + 500)
        return FittedR11Candidate(candidate, head, fitted, features), raw
    raise ValueError(f"unknown R11 candidate: {candidate}")


def fit_calibrator(
    frame: pd.DataFrame, raw: np.ndarray, head: str, method: str
) -> FittedCalibration:
    return fit_final_calibration(frame, raw, TARGETS[head], method)


def predict_calibrated(
    fitted: FittedR11Candidate,
    calibrator: FittedCalibration,
    method: str,
    frame: pd.DataFrame,
) -> np.ndarray:
    return apply_calibration(calibrator, fitted.predict_raw(frame), method)


__all__ = [
    "CALIBRATION_METHODS",
    "CANDIDATES",
    "SELECTION_RELATIVE",
    "TARGETS",
    "FittedR11Candidate",
    "add_conditioned_features",
    "candidate_inner_predictions",
    "conditioned_features",
    "fit_calibrator",
    "fit_candidate",
    "load_sleep_history",
    "predict_calibrated",
    "run_r11_transition_selection",
]

