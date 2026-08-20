"""Offline-only optimization for OPT-FUSION-001.

This module consumes the frozen FUSION-001 strict OOF table and never mutates
the active FUSION-002 bundle.  All candidate selection happens inside the
DATA-007 participant-level outer folds; published metrics are outer-test only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from .masked_stacking import (
    INPUT_FEATURES,
    build_fusion_features,
    effective_evidence,
    masked_stacking_sample_weights,
)


TASK_ID = "OPT-FUSION-001"
RUN_ID = "MH-20260802-011"
TRAINING_VERSION = "mood-social-fusion-optimization-v3.3.3-v1"
REPORT_VERSION = "mood-social-fusion-optimization-report-v1"
MODEL_VERSION = "mood-social-fusion-v2-candidate-v1"
RANDOM_SEED = 20260728
OUTER_FOLDS = 5
INNER_FOLDS = 5
EPSILON = 1.0e-6
CALIBRATION_METHODS = ("platt", "isotonic", "beta", "temperature")
V2_VARIANTS = ("interactions", "source_calibrated", "reliability_gated")
BRANCH_NAMES = (
    "activity",
    "sleep",
    "joint",
    "physiology",
    "social_context",
    "trend_activity",
    "trend_sleep",
    "trend_social",
)
WEAK_BRANCHES = ("physiology", "social_context", "trend_social")


class FusionOptimizationError(RuntimeError):
    """Raised when the frozen optimization boundary is violated."""


@dataclass(frozen=True)
class OptimizationConfig:
    repository_root: Path
    payload: Mapping[str, Any]
    config_path: Path

    @property
    def report_directory(self) -> Path:
        return _resolve(self.repository_root, self.payload["output"]["report_directory"])

    @property
    def model_path(self) -> Path:
        return _resolve(self.repository_root, self.payload["output"]["model_path"])

    @property
    def manifest_path(self) -> Path:
        return _resolve(self.repository_root, self.payload["output"]["manifest_path"])

    @property
    def table_path(self) -> Path:
        return _resolve(self.repository_root, self.payload["input"]["fusion_oof_table_path"])


@dataclass(frozen=True)
class FusionV2Model:
    task_id: str
    run_id: str
    variant: str
    feature_names: tuple[str, ...]
    classifier: LogisticRegression
    source_calibrators: Mapping[str, Mapping[str, tuple[float, float]]]
    branch_scales: Mapping[str, float]

    def validate(self) -> None:
        if self.task_id != TASK_ID or self.run_id != RUN_ID:
            raise FusionOptimizationError("candidate model identity changed")
        if self.variant not in V2_VARIANTS:
            raise FusionOptimizationError("candidate variant is invalid")
        if not isinstance(self.classifier, LogisticRegression):
            raise FusionOptimizationError("candidate classifier type changed")
        if self.classifier.coef_.shape != (1, len(self.feature_names)):
            raise FusionOptimizationError("candidate feature shape changed")


def load_optimization_config(path: str | Path, *, repository_root: str | Path | None = None) -> OptimizationConfig:
    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise FusionOptimizationError("OPT-FUSION-001 config is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise FusionOptimizationError("OPT-FUSION-001 config must be a mapping")
    if payload.get("task_id") != TASK_ID or payload.get("run_id") != RUN_ID:
        raise FusionOptimizationError("OPT-FUSION-001 task or run ID changed")
    if int(payload.get("random_seed", -1)) != RANDOM_SEED:
        raise FusionOptimizationError("OPT-FUSION-001 random seed changed")
    if tuple(payload.get("calibration", {}).get("methods", ())) != CALIBRATION_METHODS:
        raise FusionOptimizationError("calibration method grid changed")
    if tuple(payload.get("fusion_v2", {}).get("variants", ())) != V2_VARIANTS:
        raise FusionOptimizationError("Fusion v2 variant grid changed")
    root = Path(repository_root or config_path.parents[2]).resolve()
    return OptimizationConfig(repository_root=root, payload=payload, config_path=config_path)


def load_optimization_inputs(config: OptimizationConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Read and verify frozen FUSION-001/002 inputs without fitting experts."""

    table = pd.read_parquet(config.table_path).reset_index(drop=True)
    expected_sha = str(config.payload["input"]["fusion_oof_table_sha256"])
    if _sha256_file(config.table_path) != expected_sha:
        raise FusionOptimizationError("FUSION-001 table hash drifted")
    if len(table) != 22191 or table["global_participant_id"].nunique() != 15361:
        raise FusionOptimizationError("FUSION-001 row or participant count changed")
    if table["outer_fold"].isna().any() or not table["outer_fold"].isin(range(OUTER_FOLDS)).all():
        raise FusionOptimizationError("DATA-007 outer folds are invalid")
    split_path = _resolve(config.repository_root, config.payload["input"]["split"]["path"])
    if _sha256_file(split_path) != str(config.payload["input"]["split"]["sha256"]):
        raise FusionOptimizationError("DATA-007 split hash drifted")
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    assignments = pd.DataFrame(split_payload.get("participant_assignments", []))
    if assignments.empty or assignments["global_participant_id"].duplicated().any():
        raise FusionOptimizationError("DATA-007 participant assignments are invalid")
    if set(assignments["global_participant_id"].astype(str)) != set(table["global_participant_id"].astype(str)):
        raise FusionOptimizationError("FUSION-001 participants do not match DATA-007")
    baseline_manifest = _resolve(config.repository_root, "models/mental_health/mood_social/v3.3.3/mood_fusion_manifest.json")
    baseline_model = _resolve(config.repository_root, "models/mental_health/mood_social/v3.3.3/mood_fusion.joblib")
    manifest = json.loads(baseline_manifest.read_text(encoding="utf-8"))
    if manifest.get("strict_oof") is not True or manifest.get("posthoc_calibrator_selected") is not False:
        raise FusionOptimizationError("FUSION-002 baseline boundary changed")
    if _sha256_file(baseline_model) != manifest.get("model_sha256"):
        raise FusionOptimizationError("FUSION-002 baseline model hash drifted")
    protection = {
        "fusion_oof_table_sha256": expected_sha,
        "fusion_oof_report_core_sha256": str(config.payload["input"]["fusion_oof_report_core_sha256"]),
        "split_sha256": str(config.payload["input"]["split"]["sha256"]),
        "baseline_model_sha256": manifest["model_sha256"],
        "baseline_manifest_sha256": _sha256_file(baseline_manifest),
        "model006_enters_fusion": False,
    }
    return table, assignments, protection


def optimize_fusion(config: OptimizationConfig, *, overwrite: bool = False, command: Sequence[str] | None = None) -> dict[str, Any]:
    table, assignments, protection = load_optimization_inputs(config)
    _validate_output_paths(config, overwrite=overwrite)
    base_oof = _load_baseline_oof(config)
    frame = table.copy()
    frame["baseline_probability"] = base_oof.set_index(["dataset_id", "canonical_row_index"])["fusion_probability"].reindex(
        pd.MultiIndex.from_frame(frame[["dataset_id", "canonical_row_index"]])
    ).to_numpy()
    available = effective_evidence(frame) > 0
    frame = frame.loc[available].reset_index(drop=True)
    assignments_by_id = assignments.copy()
    assignments_by_id["global_participant_id"] = assignments_by_id["global_participant_id"].astype(str)
    assignments_by_id = assignments_by_id.set_index("global_participant_id")

    calibration_rows, calibration_payload = _calibration_audit(frame, assignments_by_id)
    v2_results, best_variant, best_model, v2_oof = _v2_audit(frame, assignments_by_id)
    sensitivity = _sensitivity_payload(frame, base_oof, v2_oof)
    source_robustness = _source_robustness(frame, assignments_by_id)
    ablation = _weak_branch_ablation(frame, assignments_by_id)
    summary = _summary_payload(base_oof, v2_oof, best_variant, calibration_payload, source_robustness, ablation)
    _publish(
        config,
        frame,
        base_oof,
        calibration_rows,
        calibration_payload,
        v2_results,
        v2_oof,
        sensitivity,
        source_robustness,
        ablation,
        summary,
        best_model,
        protection,
        command or (),
    )
    return {
        "status": "pass",
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "row_count": int(len(table)),
        "available_row_count": int(len(frame)),
        "selected_candidate_variant": best_variant,
        "model_path": str(config.model_path),
        "report_directory": str(config.report_directory),
    }


def _calibration_audit(frame: pd.DataFrame, assignments: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for method in CALIBRATION_METHODS:
        prediction, audits = _cross_fit_calibrator(frame, assignments, method)
        predictions[method] = prediction
        for variant in ("natural", "frozen_four_level"):
            weight = np.ones(len(frame)) if variant == "natural" else masked_stacking_sample_weights(frame)
            metrics = _metrics(frame["binary_target"].to_numpy(), prediction, weight)
            rows.append({"method": method, "weight_variant": variant, **metrics})
        for audit in audits:
            audit["method"] = method
    table = pd.DataFrame(rows)
    selected = (
        table.loc[table["weight_variant"].eq("frozen_four_level")]
        .sort_values(["brier", "ece", "method"], kind="stable")
        .iloc[0]["method"]
    )
    return table, {
        "version": "mood-social-optimization-calibration-v1",
        "methods": list(CALIBRATION_METHODS),
        "selection": "diagnostic_only_brier_then_ece",
        "diagnostic_best_method": str(selected),
        "predictions": {name: values.tolist() for name, values in predictions.items()},
    }


def _cross_fit_calibrator(frame: pd.DataFrame, assignments: pd.DataFrame, method: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    output = np.full(len(frame), np.nan, dtype=float)
    audits: list[dict[str, Any]] = []
    outer = frame["outer_fold"].astype(int).to_numpy()
    for fold in range(OUTER_FOLDS):
        test = outer == fold
        train = ~test
        train_values = frame.loc[train]
        test_values = frame.loc[test]
        train_p = _fit_calibrator(train_values, method)
        output[test] = _predict_calibrator(train_p, test_values["baseline_probability"].to_numpy(), method)
        audits.append({"outer_fold": fold, "fit_rows": int(train.sum()), "test_rows": int(test.sum())})
    if not np.isfinite(output).all():
        raise FusionOptimizationError(f"calibration coverage incomplete: {method}")
    return output, audits


def _fit_calibrator(frame: pd.DataFrame, method: str) -> Any:
    p = np.clip(frame["baseline_probability"].to_numpy(dtype=float), EPSILON, 1 - EPSILON)
    y = frame["binary_target"].to_numpy(dtype=int)
    w = masked_stacking_sample_weights(frame)
    if method == "platt":
        model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=5000, random_state=RANDOM_SEED)
        model.fit(_logit(p)[:, None], y, sample_weight=w)
        return model
    if method == "isotonic":
        model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        model.fit(p, y, sample_weight=w)
        return model
    if method == "beta":
        model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=5000, random_state=RANDOM_SEED)
        model.fit(np.column_stack([np.log(p), np.log1p(-p)]), y, sample_weight=w)
        return model
    if method == "temperature":
        z = _logit(p)
        def objective(log_t: float) -> float:
            t = float(np.exp(log_t))
            return float(log_loss(y, _sigmoid(z / t), sample_weight=w, labels=[0, 1]))
        result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded", options={"xatol": 1e-6})
        return float(np.exp(result.x))
    raise FusionOptimizationError(f"unsupported calibration method: {method}")


def _predict_calibrator(model: Any, p: np.ndarray, method: str) -> np.ndarray:
    p = np.clip(np.nan_to_num(p, nan=0.5), EPSILON, 1 - EPSILON)
    if method == "platt":
        out = model.predict_proba(_logit(p)[:, None])[:, 1]
    elif method == "isotonic":
        out = model.predict(p)
    elif method == "beta":
        out = model.predict_proba(np.column_stack([np.log(p), np.log1p(-p)]))[:, 1]
    else:
        out = _sigmoid(_logit(p) / float(model))
    return np.clip(np.asarray(out, dtype=float), EPSILON, 1 - EPSILON)


def _v2_audit(frame: pd.DataFrame, assignments: pd.DataFrame) -> tuple[pd.DataFrame, str, FusionV2Model, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    selected_by_fold: list[str] = []
    outer_predictions = np.full(len(frame), np.nan)
    for fold in range(OUTER_FOLDS):
        train_mask = frame["outer_fold"].astype(int).to_numpy() != fold
        test_mask = ~train_mask
        scores: list[dict[str, Any]] = []
        for variant in V2_VARIANTS:
            inner_scores: list[dict[str, Any]] = []
            outer_train = frame.loc[train_mask]
            for inner_fold in range(INNER_FOLDS):
                inner_test = outer_train["outer_fold"].astype(int).map(
                    lambda value: int(value == inner_fold)
                ).to_numpy(dtype=bool)
                # DATA-007 stores the inner fold per participant in the
                # assignment manifest, so derive it without using labels.
                inner_test = np.array([
                    _inner_fold_for(assignments, participant, fold) == inner_fold
                    for participant in outer_train["global_participant_id"].astype(str)
                ])
                if not inner_test.any() or inner_test.all():
                    continue
                inner_model = _fit_v2(outer_train.loc[~inner_test], variant)
                inner_probability = _predict_v2(inner_model, outer_train.loc[inner_test])
                inner_metrics = _metrics(
                    outer_train.loc[inner_test, "binary_target"].to_numpy(dtype=int),
                    inner_probability,
                    masked_stacking_sample_weights(outer_train.loc[inner_test]),
                )
                inner_scores.append(inner_metrics)
            if not inner_scores:
                raise FusionOptimizationError(f"no inner folds for Fusion v2 variant {variant}")
            scores.append({
                "variant": variant,
                "pooled_inner_auprc": float(np.mean([item["auprc"] for item in inner_scores])),
                "pooled_inner_brier": float(np.mean([item["brier"] for item in inner_scores])),
            })
        selected = sorted(scores, key=lambda row: (-row["pooled_inner_auprc"], row["pooled_inner_brier"], row["variant"]))[0]["variant"]
        selected_by_fold.append(selected)
        model = _fit_v2(frame.loc[train_mask], selected)
        outer_predictions[test_mask] = _predict_v2(model, frame.loc[test_mask])
        rows.extend({"outer_fold": fold, **row, "selected": row["variant"] == selected} for row in scores)
    counts = pd.Series(selected_by_fold).value_counts()
    best_variant = sorted(V2_VARIANTS, key=lambda value: (-int(counts.get(value, 0)), value))[0]
    best_model = _fit_v2(frame, best_variant)
    result = pd.DataFrame(rows)
    oof = frame[["dataset_id", "canonical_row_index", "global_participant_id", "binary_target", "outer_fold"]].copy()
    oof["fusion_v2_probability"] = outer_predictions
    oof["variant"] = best_variant
    return result, best_variant, best_model, oof


def _fit_v2(frame: pd.DataFrame, variant: str) -> FusionV2Model:
    base = build_fusion_features(frame)
    calibrated: dict[str, Mapping[str, tuple[float, float]]] = {}
    if variant == "source_calibrated":
        base = _source_calibrated_features(frame, base, calibrated, fit=True)
    scales = {name: 1.0 for name in BRANCH_NAMES}
    if variant == "reliability_gated":
        scales.update({"physiology": 0.25, "social_context": 0.50, "trend_social": 0.25})
        base = _apply_branch_scales(base, scales)
    features = _v2_features(base)
    y = frame["binary_target"].to_numpy(dtype=int)
    weights = masked_stacking_sample_weights(frame)
    classifier = LogisticRegression(C=10.0, penalty="l2", solver="lbfgs", max_iter=5000, tol=1e-10, random_state=RANDOM_SEED)
    classifier.fit(features, y, sample_weight=weights)
    return FusionV2Model(TASK_ID, RUN_ID, variant, tuple(_v2_feature_names()), classifier, calibrated, scales)


def _predict_v2(model: FusionV2Model, frame: pd.DataFrame) -> np.ndarray:
    base = build_fusion_features(frame)
    if model.variant == "source_calibrated":
        base = _source_calibrated_features(frame, base, dict(model.source_calibrators), fit=False)
    base = _apply_branch_scales(base, model.branch_scales)
    return _sigmoid(model.classifier.decision_function(_v2_features(base)))


def _source_calibrated_features(frame: pd.DataFrame, base: np.ndarray, state: dict[str, Mapping[str, tuple[float, float]]], *, fit: bool) -> np.ndarray:
    result = base.copy()
    for index, branch in enumerate(("activity", "sleep", "joint", "physiology", "social_context")):
        p = np.clip(frame[f"expert_{branch}_current_probability"].fillna(0.5).to_numpy(dtype=float), EPSILON, 1 - EPSILON)
        for source in sorted(frame["dataset_id"].astype(str).unique(), key=lambda value: value.encode("utf-8")):
            selected = frame["dataset_id"].astype(str).eq(source).to_numpy()
            if fit:
                if selected.sum() < 20 or frame.loc[selected, "binary_target"].nunique() < 2:
                    continue
                logit_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=RANDOM_SEED)
                logit_model.fit(_logit(p[selected])[:, None], frame.loc[selected, "binary_target"].to_numpy(dtype=int), sample_weight=masked_stacking_sample_weights(frame.loc[selected]))
                a, b = float(logit_model.coef_[0, 0]), float(logit_model.intercept_[0])
                state.setdefault(source, {})[branch] = (a, b)
            else:
                a, b = state.get(source, {}).get(branch, (1.0, 0.0))
            result[selected, index] = np.asarray(frame.loc[selected, f"expert_{branch}_expert_mask"], dtype=float) * np.asarray(frame.loc[selected, f"expert_{branch}_confidence"], dtype=float) * _logit(_sigmoid(a * _logit(p[selected]) + b))
    return result


def _apply_branch_scales(base: np.ndarray, scales: Mapping[str, float]) -> np.ndarray:
    result = base.copy()
    for index, name in enumerate(BRANCH_NAMES):
        result[:, index] *= float(scales.get(name, 1.0))
    return result


def _v2_features(base: np.ndarray) -> np.ndarray:
    interactions = np.column_stack([
        base[:, 0] * base[:, 1],
        base[:, 0] * base[:, 2],
        base[:, 1] * base[:, 2],
        base[:, 0] * base[:, 5],
        base[:, 1] * base[:, 6],
        base[:, 4] * base[:, 7],
    ])
    return np.column_stack([base, interactions])


def _v2_feature_names() -> list[str]:
    return list(INPUT_FEATURES) + [
        "interaction_activity_sleep",
        "interaction_activity_joint",
        "interaction_sleep_joint",
        "interaction_activity_trend_activity",
        "interaction_sleep_trend_sleep",
        "interaction_social_context_trend_social",
    ]


def _source_robustness(frame: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in sorted(frame["dataset_id"].astype(str).unique(), key=lambda value: value.encode("utf-8")):
        for fold in range(OUTER_FOLDS):
            train = frame[frame["outer_fold"].astype(int).ne(fold) & frame["dataset_id"].astype(str).ne(source)]
            test = frame[frame["outer_fold"].astype(int).eq(fold) & frame["dataset_id"].astype(str).eq(source)]
            if train.empty or test.empty or train["binary_target"].nunique() < 2 or test["binary_target"].nunique() < 2:
                continue
            features = build_fusion_features(train)
            clf = LogisticRegression(C=10.0, penalty="l2", solver="lbfgs", max_iter=5000, random_state=RANDOM_SEED).fit(features, train["binary_target"], sample_weight=masked_stacking_sample_weights(train))
            probability = _sigmoid(clf.decision_function(build_fusion_features(test)))
            metrics = _metrics(test["binary_target"].to_numpy(dtype=int), probability, masked_stacking_sample_weights(test))
            rows.append({"held_out_source": source, "outer_fold": fold, "training_source_count": int(train["dataset_id"].nunique()), **metrics})
    return pd.DataFrame(rows)


def _weak_branch_ablation(frame: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for branch in ("none", *WEAK_BRANCHES, "all_weak"):
        predictions = np.full(len(frame), np.nan)
        for fold in range(OUTER_FOLDS):
            train_mask = frame["outer_fold"].astype(int).to_numpy() != fold
            test_mask = ~train_mask
            train_features = build_fusion_features(frame.loc[train_mask])
            test_features = build_fusion_features(frame.loc[test_mask])
            if branch == "all_weak":
                indexes = [BRANCH_NAMES.index(name) for name in WEAK_BRANCHES]
            elif branch == "none":
                indexes = []
            else:
                indexes = [BRANCH_NAMES.index(branch)]
            train_features[:, indexes] = 0.0
            test_features[:, indexes] = 0.0
            clf = LogisticRegression(C=10.0, penalty="l2", solver="lbfgs", max_iter=5000, random_state=RANDOM_SEED).fit(train_features, frame.loc[train_mask, "binary_target"], sample_weight=masked_stacking_sample_weights(frame.loc[train_mask]))
            predictions[test_mask] = _sigmoid(clf.decision_function(test_features))
        metrics = _metrics(frame["binary_target"].to_numpy(dtype=int), predictions, masked_stacking_sample_weights(frame))
        rows.append({"ablation": branch, **metrics})
    return pd.DataFrame(rows)


def _inner_fold_for(assignments: pd.DataFrame, participant: str, outer_fold: int) -> int:
    try:
        value = assignments.loc[str(participant), "inner_validation_fold_by_outer_fold"]
    except KeyError as exc:
        raise FusionOptimizationError(f"participant missing from DATA-007 assignments: {participant}") from exc
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    if isinstance(value, Mapping):
        value = value.get(str(outer_fold), value.get(outer_fold))
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise FusionOptimizationError("DATA-007 inner fold value is invalid") from exc
    if value not in range(INNER_FOLDS):
        raise FusionOptimizationError("DATA-007 inner fold is outside 0..4")
    return value


def _sensitivity_payload(frame: pd.DataFrame, baseline_oof: pd.DataFrame, v2_oof: pd.DataFrame) -> dict[str, Any]:
    base = baseline_oof.set_index(["dataset_id", "canonical_row_index"])["fusion_probability"].reindex(pd.MultiIndex.from_frame(frame[["dataset_id", "canonical_row_index"]])).to_numpy()
    candidate = v2_oof.set_index(["dataset_id", "canonical_row_index"])["fusion_v2_probability"].reindex(pd.MultiIndex.from_frame(frame[["dataset_id", "canonical_row_index"]])).to_numpy()
    target = frame["binary_target"].to_numpy(dtype=int)
    priors = [float(value) for value in (0.05, 0.10, 0.20, 0.30)]
    prior_rows = []
    observed = float(target.mean())
    for model_name, probability in (("baseline", base), ("fusion_v2", candidate)):
        for prior in priors:
            adjusted = _prior_adjust(probability, observed, prior)
            prior_rows.append({"model": model_name, "target_prior": prior, "label": "offline_sensitivity", **_metrics(target, adjusted, masked_stacking_sample_weights(frame))})
    cost_rows = []
    for model_name, probability in (("baseline", base), ("fusion_v2", candidate)):
        for fn_cost in (1.0, 2.0, 5.0, 10.0):
            for fp_cost in (1.0, 2.0, 5.0):
                thresholds = np.linspace(0.1, 0.9, 9)
                utilities = []
                for threshold in thresholds:
                    pred = probability >= threshold
                    weight = masked_stacking_sample_weights(frame)
                    utilities.append(float(np.sum(weight[(target == 1) & ~pred]) * -fn_cost + np.sum(weight[(target == 0) & pred]) * -fp_cost))
                index = int(np.argmax(utilities))
                cost_rows.append({"model": model_name, "false_negative_cost": fn_cost, "false_positive_cost": fp_cost, "descriptive_threshold": float(thresholds[index]), "utility": utilities[index], "label": "offline_sensitivity"})
    return {"version": "mood-social-optimization-sensitivity-v1", "interpretation": "offline_sensitivity_only_no_production_prior_or_threshold", "target_prior": prior_rows, "cost_work_points": cost_rows}


def _summary_payload(base_oof: pd.DataFrame, v2_oof: pd.DataFrame, best_variant: str, calibration: Mapping[str, Any], source: pd.DataFrame, ablation: pd.DataFrame) -> dict[str, Any]:
    base = base_oof.dropna(subset=["fusion_probability"])
    candidate = v2_oof.dropna(subset=["fusion_v2_probability"])
    base_metrics = _metrics(base["binary_target"].to_numpy(dtype=int), base["fusion_probability"].to_numpy(dtype=float), np.ones(len(base)))
    cand_metrics = _metrics(candidate["binary_target"].to_numpy(dtype=int), candidate["fusion_v2_probability"].to_numpy(dtype=float), np.ones(len(candidate)))
    return {"version": REPORT_VERSION, "baseline": base_metrics, "fusion_v2": cand_metrics, "best_variant": best_variant, "diagnostic_best_calibration": calibration["diagnostic_best_method"], "source_robustness_rows": int(len(source)), "ablation_rows": int(len(ablation)), "promotion_status": "review_required_art_candidate_not_production"}


def _metrics(target: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), EPSILON, 1 - EPSILON)
    weight = np.asarray(weight, dtype=float)
    predicted = probability >= 0.5
    return {"row_count": int(len(target)), "positive_rows": int(target.sum()), "prevalence": float(np.average(target, weights=weight)), "auprc": float(average_precision_score(target, probability, sample_weight=weight)), "auroc": float(roc_auc_score(target, probability, sample_weight=weight)) if len(np.unique(target)) == 2 else None, "brier": float(np.average((probability - target) ** 2, weights=weight)), "log_loss": float(log_loss(target, probability, sample_weight=weight, labels=[0, 1])), "ece": _ece(target, probability, weight), "descriptive_threshold": 0.5, "sensitivity": float(np.sum(weight[(target == 1) & predicted]) / np.sum(weight[target == 1])) if np.any(target == 1) else None, "specificity": float(np.sum(weight[(target == 0) & ~predicted]) / np.sum(weight[target == 0])) if np.any(target == 0) else None}


def _ece(target: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> float:
    total = float(weight.sum())
    result = 0.0
    for index in range(10):
        lower, upper = index / 10.0, (index + 1) / 10.0
        selected = (probability >= lower) & ((probability < upper) if index < 9 else (probability <= upper))
        if selected.any():
            result += float(weight[selected].sum() / total) * abs(float(np.average(probability[selected], weights=weight[selected])) - float(np.average(target[selected], weights=weight[selected])))
    return float(result)


def _prior_adjust(probability: np.ndarray, observed: float, prior: float) -> np.ndarray:
    probability = np.clip(probability, EPSILON, 1 - EPSILON)
    odds = probability / (1 - probability)
    adjusted_odds = odds * (prior / (1 - prior)) / (observed / (1 - observed))
    return adjusted_odds / (1 + adjusted_odds)


def _load_baseline_oof(config: OptimizationConfig) -> pd.DataFrame:
    path = _resolve(config.repository_root, config.payload["input"]["baseline_oof_path"])
    if _sha256_file(path) != str(config.payload["input"]["baseline_oof_sha256"]):
        raise FusionOptimizationError("FUSION-002 baseline OOF hash drifted")
    return pd.read_parquet(path)


def _publish(config: OptimizationConfig, frame: pd.DataFrame, base_oof: pd.DataFrame, calibration: pd.DataFrame, calibration_payload: Mapping[str, Any], v2_results: pd.DataFrame, v2_oof: pd.DataFrame, sensitivity: Mapping[str, Any], source: pd.DataFrame, ablation: pd.DataFrame, summary: Mapping[str, Any], model: FusionV2Model, protection: Mapping[str, Any], command: Sequence[str]) -> None:
    report = config.report_directory
    report.mkdir(parents=True, exist_ok=True)
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.model_path)
    _write_parquet(calibration, report / "calibration_comparison.parquet")
    _write_parquet(v2_results, report / "fusion_v2_candidates.parquet")
    _write_parquet(v2_oof, report / "fusion_v2_oof_predictions.parquet")
    _write_parquet(source, report / "source_robustness.parquet")
    _write_parquet(ablation, report / "weak_branch_ablation.parquet")
    _write_json(report / "sensitivity.json", sensitivity)
    _write_json(report / "summary.json", summary)
    _write_json(report / "calibration_payload.json", {k: v for k, v in calibration_payload.items() if k != "predictions"})
    _write_json(report / "upstream_protection.json", dict(protection))
    _write_json(report / "config.json", dict(config.payload))
    _write_json(report / "run.json", {"task_id": TASK_ID, "run_id": RUN_ID, "training_version": TRAINING_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(), "command": list(command), "strict_oof": True})
    _write_json(report / "model_card.json", {"task_id": TASK_ID, "run_id": RUN_ID, "model_version": MODEL_VERSION, "variant": model.variant, "offline_only": True, "promotion_status": "review_required", "model_path": str(config.model_path)})
    manifest = {"version": "mood-social-fusion-optimization-manifest-v1", "task_id": TASK_ID, "run_id": RUN_ID, "training_version": TRAINING_VERSION, "strict_oof": True, "input_table_sha256": str(config.payload["input"]["fusion_oof_table_sha256"]), "baseline_oof_sha256": str(config.payload["input"]["baseline_oof_sha256"]), "oof_sha256": _sha256_file(report / "fusion_v2_oof_predictions.parquet"), "model_sha256": _sha256_file(config.model_path), "report_core_sha256": _report_core_sha256(report), "offline_only": True, "production_boundary": {"select_threshold": False, "change_active_experts": False, "change_http_behavior": False, "include_model006_predictions": False, "overwrite_active_artifacts": False}}
    _write_json(config.manifest_path, manifest)
    artifacts = []
    for path in sorted(report.rglob("*"), key=lambda item: item.relative_to(report).as_posix()):
        if path.is_file() and path.name not in {"artifacts.json", "run.json"}:
            artifacts.append({"path": path.relative_to(report).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    _write_json(report / "artifacts.json", {"version": "mood-social-fusion-optimization-artifacts-v1", "artifacts": artifacts})


def _validate_output_paths(config: OptimizationConfig, *, overwrite: bool) -> None:
    if not overwrite and (config.report_directory.exists() or config.model_path.exists() or config.manifest_path.exists()):
        raise FusionOptimizationError("OPT-FUSION-001 output exists; use --overwrite")


def _v2_feature_names_for_test() -> tuple[str, ...]:
    return tuple(_v2_feature_names())


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(values, dtype=float), -700, 700)))


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.log(values / (1 - values))


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_core_sha256(report: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((item for item in report.rglob("*") if item.is_file() and item.name not in {"artifacts.json", "run.json"}), key=lambda item: item.relative_to(report).as_posix()):
        digest.update(path.relative_to(report).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False, engine="pyarrow", compression="snappy")


__all__ = [
    "CALIBRATION_METHODS",
    "FusionOptimizationError",
    "OptimizationConfig",
    "RUN_ID",
    "TASK_ID",
    "V2_VARIANTS",
    "_v2_feature_names_for_test",
    "load_optimization_config",
    "load_optimization_inputs",
    "optimize_fusion",
]
