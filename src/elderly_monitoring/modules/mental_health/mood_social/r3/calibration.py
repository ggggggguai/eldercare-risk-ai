"""Inner-only calibration and workpoint selection for OPT-V333-006."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from elderly_monitoring.modules.mental_health.mood_social.r3.contract import (
    R3_PROTOCOL_VERSION,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.modeling import (
    binary_metrics,
)


CALIBRATION_VERSION = "mood-social-v3.3.3-r3-calibration-v1"
CALIBRATION_METHODS = ("none", "platt", "beta", "temperature", "hierarchical")
DEFAULT_FUSION_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-005-fusion/selected_fusion_inner_oof.parquet"
)
DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-006-calibration"
)
EPSILON = 1.0e-6


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _logit(probability: Sequence[float]) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    return np.log(value / (1.0 - value))


def _sigmoid(value: Sequence[float]) -> np.ndarray:
    array = np.clip(np.asarray(value, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-array))


def _ece(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability < edges[index + 1]
            if index < bins - 1
            else probability <= edges[index + 1]
        )
        if selected.any():
            result += float(selected.mean()) * abs(
                float(probability[selected].mean()) - float(target[selected].mean())
            )
    return float(result)


def _calibration_metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    target = frame["binary_target"].to_numpy(int)
    ranking = binary_metrics(target, probability)
    return {
        **ranking,
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "ece": _ece(target, probability),
    }


@dataclass
class FittedCalibrator:
    method: str
    estimator: Any = None
    temperature: float = 1.0
    route_offsets: Mapping[str, float] | None = None
    eligible_routes: tuple[str, ...] = ()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        probability = frame["probability"].to_numpy(float)
        if self.method == "none":
            result = probability
        elif self.method == "platt":
            result = self.estimator.predict_proba(_logit(probability)[:, None])[:, 1]
        elif self.method == "beta":
            matrix = np.column_stack(
                [np.log(np.clip(probability, EPSILON, 1.0)), np.log(np.clip(1.0 - probability, EPSILON, 1.0))]
            )
            result = self.estimator.predict_proba(matrix)[:, 1]
        elif self.method == "temperature":
            result = _sigmoid(_logit(probability) / self.temperature)
        elif self.method == "hierarchical":
            result = self.estimator.predict_proba(_logit(probability)[:, None])[:, 1]
            base_logit = _logit(result)
            route = frame["route_pattern"].astype(str).to_numpy()
            offsets = self.route_offsets or {}
            result = _sigmoid(
                base_logit + np.asarray([float(offsets.get(value, 0.0)) for value in route])
            )
        else:
            raise ValueError(f"unknown calibrator: {self.method}")
        return np.clip(result, EPSILON, 1.0 - EPSILON)


def _fit_logistic(matrix: np.ndarray, target: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        solver="liblinear", C=1.0, max_iter=3000, random_state=20260728
    )
    model.fit(matrix, target)
    return model


def _temperature(frame: pd.DataFrame) -> float:
    target = frame["binary_target"].to_numpy(int)
    logit = _logit(frame["probability"])
    candidates = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
    return min(
        candidates,
        key=lambda value: (
            log_loss(target, _sigmoid(logit / value), labels=[0, 1]),
            value,
        ),
    )


def _route_eligibility(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    audit: dict[str, dict[str, Any]] = {}
    for route, route_frame in frame.groupby("route_pattern", sort=True):
        source_rows = route_frame.groupby("dataset_id").size().to_dict()
        source_positive_participants = (
            route_frame.loc[route_frame["binary_target"].eq(1)]
            .groupby("dataset_id")["global_participant_id"]
            .nunique()
            .to_dict()
        )
        sources = sorted(str(value) for value in route_frame["dataset_id"].unique())
        eligible = (
            len(sources) >= 2
            and all(int(source_rows.get(source, 0)) >= 500 for source in sources)
            and all(int(source_positive_participants.get(source, 0)) >= 50 for source in sources)
        )
        audit[str(route)] = {
            "sources": sources,
            "source_rows": {str(key): int(value) for key, value in source_rows.items()},
            "source_positive_participants": {
                str(key): int(value) for key, value in source_positive_participants.items()
            },
            "minimums_pass": bool(eligible),
        }
    return audit


def _fit_hierarchical(frame: pd.DataFrame) -> FittedCalibrator:
    target = frame["binary_target"].to_numpy(int)
    global_model = _fit_logistic(_logit(frame["probability"])[:, None], target)
    eligibility = _route_eligibility(frame)
    offsets: dict[str, float] = {}
    eligible_routes: list[str] = []
    global_prevalence = float(target.mean())
    for route, detail in eligibility.items():
        if not detail["minimums_pass"]:
            continue
        selected = frame["route_pattern"].astype(str).eq(route).to_numpy()
        route_prevalence = float(target[selected].mean())
        raw_offset = float(_logit([route_prevalence])[0] - _logit([global_prevalence])[0])
        shrinkage = float(selected.sum() / (selected.sum() + 500.0))
        offset = shrinkage * raw_offset
        # True source-held-out stability: estimate both the global calibrator
        # and the route offset without the held source, then score only that
        # source.  Merely applying the all-source offset to each source would
        # let the held source's prevalence leak into this gate.
        stable = True
        for source in detail["sources"]:
            source_mask = frame["dataset_id"].astype(str).eq(source).to_numpy()
            heldout = selected & source_mask
            if not heldout.any():
                continue
            source_train = frame.loc[~source_mask]
            source_target = source_train["binary_target"].to_numpy(int)
            source_global_model = _fit_logistic(
                _logit(source_train["probability"])[:, None], source_target
            )
            heldout_global = source_global_model.predict_proba(
                _logit(frame.loc[heldout, "probability"])[:, None]
            )[:, 1]
            source_route = source_train["route_pattern"].astype(str).eq(route).to_numpy()
            if not source_route.any() or source_target[source_route].sum() == 0:
                stable = False
                break
            source_global_prevalence = float(source_target.mean())
            source_route_prevalence = float(source_target[source_route].mean())
            heldout_offset = float(
                _logit([source_route_prevalence])[0]
                - _logit([source_global_prevalence])[0]
            )
            heldout_shrinkage = float(
                source_route.sum() / (source_route.sum() + 500.0)
            )
            shifted = _sigmoid(
                _logit(heldout_global) + heldout_shrinkage * heldout_offset
            )
            y = target[heldout]
            delta = float(
                np.mean(np.square(shifted - y))
                - np.mean(np.square(heldout_global - y))
            )
            if delta > 0.005:
                stable = False
                break
        if stable:
            offsets[route] = offset
            eligible_routes.append(route)
    return FittedCalibrator(
        "hierarchical",
        estimator=global_model,
        route_offsets=offsets,
        eligible_routes=tuple(sorted(eligible_routes)),
    )


def fit_calibrator(frame: pd.DataFrame, method: str) -> FittedCalibrator:
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"unknown calibration method: {method}")
    target = frame["binary_target"].to_numpy(int)
    probability = frame["probability"].to_numpy(float)
    if method == "none":
        return FittedCalibrator("none")
    if method == "platt":
        return FittedCalibrator("platt", _fit_logistic(_logit(probability)[:, None], target))
    if method == "beta":
        matrix = np.column_stack(
            [np.log(np.clip(probability, EPSILON, 1.0)), np.log(np.clip(1.0 - probability, EPSILON, 1.0))]
        )
        return FittedCalibrator("beta", _fit_logistic(matrix, target))
    if method == "temperature":
        return FittedCalibrator("temperature", temperature=_temperature(frame))
    return _fit_hierarchical(frame)


def crossfit_calibrator(frame: pd.DataFrame, method: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    result = np.full(len(frame), np.nan, dtype=float)
    audits: list[dict[str, Any]] = []
    for validation_fold in sorted(frame["fusion_validation_fold"].unique()):
        train = frame.loc[frame["fusion_validation_fold"].ne(validation_fold)]
        selected = frame["fusion_validation_fold"].eq(validation_fold).to_numpy()
        fitted = fit_calibrator(train, method)
        result[selected] = fitted.predict(frame.loc[selected])
        audits.append(
            {
                "validation_fold": int(validation_fold),
                "train_rows": int(len(train)),
                "validation_rows": int(selected.sum()),
                "eligible_routes": list(fitted.eligible_routes),
            }
        )
    if not np.isfinite(result).all():
        raise ValueError("calibration cross-fitting is incomplete")
    return result, audits


def _threshold_metrics(target: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = probability >= threshold
    tp = int(np.sum(predicted & (target == 1)))
    fp = int(np.sum(predicted & (target == 0)))
    fn = int(np.sum(~predicted & (target == 1)))
    tn = int(np.sum(~predicted & (target == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "alert_rate": float(predicted.mean()),
    }


def select_workpoints(target: Sequence[int], probability: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(target, dtype=int)
    p = np.asarray(probability, dtype=float)
    thresholds = np.unique(np.concatenate(([0.0], p, [1.0 + EPSILON])))
    rows = [_threshold_metrics(y, p, threshold) for threshold in thresholds]
    competition = max(rows, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
    safety_candidates = [row for row in rows if row["sensitivity"] >= 0.80]
    safety = max(
        safety_candidates,
        key=lambda row: (row["precision"], row["f1"], row["threshold"]),
    )

    def at_specificity(value: float) -> dict[str, float]:
        candidates = [row for row in rows if row["specificity"] >= value]
        return max(candidates, key=lambda row: (row["sensitivity"], row["precision"], -row["threshold"]))

    return {
        "competition": competition,
        "safety": safety,
        "at_specificity_0_80": at_specificity(0.80),
        "at_specificity_0_90": at_specificity(0.90),
        "threshold_candidate_count": int(len(thresholds)),
    }


def run_calibration_selection(
    *,
    repository_root: Path,
    fusion_oof_path: Path | None = None,
    report_directory: Path | None = None,
    outer_folds: Iterable[int] = range(5),
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    source_path = fusion_oof_path or root / DEFAULT_FUSION_RELATIVE
    source = pd.read_parquet(source_path)
    output = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    search_rows: list[dict[str, Any]] = []
    calibrated_parts: list[pd.DataFrame] = []
    choices: dict[str, Any] = {}
    for outer_context in tuple(int(value) for value in outer_folds):
        frame = source.loc[source["outer_context"].eq(outer_context)].copy().reset_index(drop=True)
        candidates: dict[str, np.ndarray] = {}
        audits: dict[str, Any] = {}
        for method in CALIBRATION_METHODS:
            probability, audit = crossfit_calibrator(frame, method)
            candidates[method] = probability
            audits[method] = audit
            metrics = _calibration_metrics(frame, probability)
            search_rows.append(
                {
                    "outer_context": outer_context,
                    "method": method,
                    **metrics,
                }
            )
        outer_search = [row for row in search_rows if row["outer_context"] == outer_context]
        selected = min(
            outer_search,
            key=lambda row: (row["brier"], row["log_loss"], row["ece"], row["method"]),
        )
        method = str(selected["method"])
        frame["uncalibrated_probability"] = frame["probability"].to_numpy(float)
        frame["probability"] = candidates[method]
        frame["calibration_method"] = method
        calibrated_parts.append(frame)
        choices[str(outer_context)] = {
            "selected_method": method,
            "selection_rule": "lowest cross-fitted Brier then log-loss then ECE",
            "selected_metrics": selected,
            "crossfit_audit": audits[method],
            "route_eligibility": _route_eligibility(frame),
            "workpoints": select_workpoints(frame["binary_target"], frame["probability"]),
            "outer_results_opened": False,
        }
    search = pd.DataFrame(search_rows)
    calibrated = pd.concat(calibrated_parts, ignore_index=True)
    search_path = output / "calibration_search.parquet"
    oof_path = output / "calibrated_fusion_inner_oof.parquet"
    choices_path = output / "calibration_and_workpoints.json"
    for path in (search_path, oof_path, choices_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite calibration artifact: {path}")
    search.to_parquet(search_path, index=False)
    calibrated.to_parquet(oof_path, index=False)
    choices_path.write_text(
        json.dumps(
            {
                "protocol_version": R3_PROTOCOL_VERSION,
                "calibration_version": CALIBRATION_VERSION,
                "outer_results_opened": False,
                "outer_choices": choices,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol_version": R3_PROTOCOL_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "task_id": "OPT-V333-006",
        "status": "pass",
        "outer_results_opened": False,
        "source_path": source_path.relative_to(root).as_posix(),
        "source_sha256": _sha256_file(source_path),
        "search_path": search_path.relative_to(root).as_posix(),
        "search_sha256": _sha256_file(search_path),
        "calibrated_oof_path": oof_path.relative_to(root).as_posix(),
        "calibrated_oof_sha256": _sha256_file(oof_path),
        "choices_path": choices_path.relative_to(root).as_posix(),
        "choices_sha256": _sha256_file(choices_path),
    }
    manifest_path = output / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "CALIBRATION_METHODS",
    "CALIBRATION_VERSION",
    "FittedCalibrator",
    "crossfit_calibrator",
    "fit_calibrator",
    "run_calibration_selection",
    "select_workpoints",
]
