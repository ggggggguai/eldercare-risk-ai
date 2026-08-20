"""Read-only calibration, threshold and source-transfer audit for V3.3.3 experts.

EVAL-001 consumes the five already-published strict outer-fold OOF tables.  It
never writes to ``models/`` or to an earlier run directory.  Diagnostic
calibrators and leave-one-source-out classifiers are intentionally report-only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

EVALUATION_VERSION = "mood-social-expert-audit-v3.3.3-v1"
METRICS_VERSION = "mood-social-expert-audit-metrics-v1"
PREDICTION_VERSION = "mood-social-calibration-candidate-oof-v1"
LOSO_VERSION = "mood-social-leave-one-source-out-v1"
RUN_ID = "MH-20260801-006"
TASK_ID = "EVAL-001"
SPLIT_ID = "mood-social-v3.3.3-participant-nested-5x5-seed-20260728-v1"
SPLIT_SHA256 = "e9915dbc590a6ea454c34d26558ac5866c57ee8a8e78f44cdcc79c859df77ee3"
FEATURE_SCHEMA_SHA256 = (
    "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
)
EXPERT_ORDER = (
    "activity",
    "sleep",
    "joint",
    "physiology",
    "social_context",
)
LOSO_EXPERT_ORDER = ("activity", "sleep", "joint", "social_context")
DISPLAY_NAMES = {
    "activity": "ActivityExpert",
    "sleep": "SleepExpert",
    "joint": "ActivitySleepJointExpert",
    "physiology": "PhysiologyExpert",
    "social_context": "SocialContextExpert",
}
PROBABILITY_CLIP_EPSILON = 1.0e-6


class MoodSocialEvaluationError(RuntimeError):
    """Raised when a frozen input or EVAL-001 invariant is violated."""


@dataclass(frozen=True)
class CalibrationCandidate:
    """One report-only cross-fitted calibration candidate."""

    variant: str
    method: str
    weight_scheme: str
    target_prior: float | None = None


def load_evaluation_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the frozen EVAL-001 configuration."""

    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise MoodSocialEvaluationError("EVAL-001 configuration is unreadable") from exc
    if not isinstance(payload, dict):
        raise MoodSocialEvaluationError("EVAL-001 configuration must be a mapping")
    if payload.get("evaluation_version") != EVALUATION_VERSION:
        raise MoodSocialEvaluationError("EVAL-001 evaluation version changed")
    if payload.get("task_id") != TASK_ID or payload.get("run_id") != RUN_ID:
        raise MoodSocialEvaluationError("EVAL-001 task or run ID changed")
    if int(payload.get("random_seed", -1)) != 20260728:
        raise MoodSocialEvaluationError("EVAL-001 random seed changed")
    split = payload.get("split")
    if not isinstance(split, dict) or (
        split.get("split_id") != SPLIT_ID
        or split.get("sha256") != SPLIT_SHA256
        or split.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256
    ):
        raise MoodSocialEvaluationError("EVAL-001 frozen split binding changed")
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("methods") != [
        "platt",
        "isotonic",
    ]:
        raise MoodSocialEvaluationError("EVAL-001 calibration methods changed")
    if calibration.get("weight_schemes") != [
        "frozen_four_level",
        "natural_sample",
    ]:
        raise MoodSocialEvaluationError("EVAL-001 calibration weight schemes changed")
    priors = [float(value) for value in calibration.get("target_prior_sensitivity", [])]
    if priors != [0.05, 0.1, 0.2, 0.3]:
        raise MoodSocialEvaluationError("EVAL-001 target-prior grid changed")
    protection = payload.get("baseline_protection")
    if not isinstance(protection, dict) or tuple(protection) != EXPERT_ORDER:
        raise MoodSocialEvaluationError("EVAL-001 expert protection order changed")
    for expert, binding in protection.items():
        if not isinstance(binding, dict):
            raise MoodSocialEvaluationError("EVAL-001 expert binding is malformed")
        for field in (
            "model_sha256",
            "manifest_sha256",
            "predictions_sha256",
            "metrics_sha256",
            "search_results_sha256",
            "training_config_sha256",
        ):
            if not _is_sha256(str(binding.get(field, ""))):
                raise MoodSocialEvaluationError(
                    f"EVAL-001 {expert} protection hash is malformed"
                )
    loso = payload.get("leave_one_source_out")
    if (
        not isinstance(loso, dict)
        or tuple(loso.get("experts", ())) != LOSO_EXPERT_ORDER
    ):
        raise MoodSocialEvaluationError("EVAL-001 LOSO expert set changed")
    reporting = payload.get("reporting")
    if not isinstance(reporting, dict) or any(
        bool(reporting.get(field))
        for field in (
            "choose_production_threshold",
            "modify_attention_level_boundaries",
            "publish_diagnostic_models",
            "overwrite_active_artifacts",
        )
    ):
        raise MoodSocialEvaluationError("EVAL-001 read-only reporting boundary changed")
    return payload


def calibration_sample_weights(
    frame: pd.DataFrame,
    scheme: str,
    *,
    target_prior: float | None = None,
) -> np.ndarray:
    """Compute one calibration-only weight vector inside the supplied fold."""

    required = {"dataset_id", "global_participant_id", "binary_target"}
    if frame.empty or not required.issubset(frame.columns):
        raise MoodSocialEvaluationError("calibration weight frame is incomplete")
    target = pd.to_numeric(frame["binary_target"], errors="raise").to_numpy(
        dtype="int64"
    )
    if set(np.unique(target)) != {0, 1}:
        raise MoodSocialEvaluationError("calibration fold must contain both classes")
    if scheme == "natural_sample":
        weights = np.ones(len(frame), dtype="float64")
    elif scheme == "frozen_four_level":
        datasets = frame["dataset_id"].astype(str)
        participants = frame["global_participant_id"].astype(str)
        weights = np.zeros(len(frame), dtype="float64")
        for dataset_id in sorted(
            set(datasets), key=lambda value: value.encode("utf-8")
        ):
            dataset_mask = datasets.eq(dataset_id).to_numpy()
            classes = sorted(set(target[dataset_mask].tolist()))
            class_mass = 1.0 / len(classes)
            for class_value in classes:
                class_mask = dataset_mask & (target == class_value)
                unit_counts = participants[class_mask].value_counts(sort=False)
                unit_mass = class_mass / len(unit_counts)
                for row_index in np.flatnonzero(class_mask):
                    weights[row_index] = unit_mass / int(
                        unit_counts[participants.iloc[row_index]]
                    )
    elif scheme == "target_prior":
        if target_prior is None or not 0.0 < target_prior < 1.0:
            raise MoodSocialEvaluationError(
                "target-prior calibration requires 0 < p < 1"
            )
        observed = float(target.mean())
        if not 0.0 < observed < 1.0:
            raise MoodSocialEvaluationError(
                "target-prior calibration needs both classes"
            )
        weights = np.where(
            target == 1,
            target_prior / observed,
            (1.0 - target_prior) / (1.0 - observed),
        ).astype("float64")
    else:
        raise MoodSocialEvaluationError("unknown calibration weight scheme")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise MoodSocialEvaluationError("calibration weight calculation failed")
    weights *= len(weights) / weights.sum()
    return weights


def cross_fit_calibrator(
    frame: pd.DataFrame,
    *,
    method: str,
    weight_scheme: str,
    target_prior: float | None = None,
    random_seed: int = 20260728,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit one calibrator per frozen outer-training partition and predict its test fold."""

    required = {
        "global_participant_id",
        "dataset_id",
        "binary_target",
        "outer_fold",
        "raw_probability",
    }
    if frame.empty or not required.issubset(frame.columns):
        raise MoodSocialEvaluationError("calibration OOF frame is incomplete")
    if method not in {"platt", "isotonic"}:
        raise MoodSocialEvaluationError("unsupported diagnostic calibrator")
    participant_folds = frame.groupby("global_participant_id", sort=False)[
        "outer_fold"
    ].nunique()
    if not participant_folds.eq(1).all():
        raise MoodSocialEvaluationError(
            "one participant appears in multiple outer folds"
        )
    output = np.full(len(frame), np.nan, dtype="float64")
    audits: list[dict[str, Any]] = []
    folds = sorted(pd.to_numeric(frame["outer_fold"], errors="raise").unique())
    if folds != [0, 1, 2, 3, 4]:
        raise MoodSocialEvaluationError(
            "diagnostic calibration requires frozen five folds"
        )
    for fold in folds:
        test_mask = pd.to_numeric(frame["outer_fold"]).eq(fold).to_numpy()
        train_mask = ~test_mask
        train = frame.loc[train_mask].reset_index(drop=True)
        test = frame.loc[test_mask]
        train_participants = set(train["global_participant_id"].astype(str))
        test_participants = set(test["global_participant_id"].astype(str))
        if train_participants & test_participants:
            raise MoodSocialEvaluationError("calibrator participant leakage detected")
        weights = calibration_sample_weights(
            train,
            weight_scheme,
            target_prior=target_prior,
        )
        raw_train = pd.to_numeric(train["raw_probability"]).to_numpy(dtype="float64")
        raw_test = pd.to_numeric(test["raw_probability"]).to_numpy(dtype="float64")
        target = pd.to_numeric(train["binary_target"]).to_numpy(dtype="int64")
        estimator: LogisticRegression | IsotonicRegression
        if method == "platt":
            estimator = LogisticRegression(
                C=np.inf,
                solver="lbfgs",
                max_iter=5000,
                random_state=random_seed,
            )
            estimator.fit(
                _probability_logit(raw_train)[:, None],
                target,
                sample_weight=weights,
            )
            predicted = estimator.predict_proba(_probability_logit(raw_test)[:, None])[
                :, 1
            ]
            state = {
                "coefficient": float(estimator.coef_[0, 0]),
                "intercept": float(estimator.intercept_[0]),
                "iterations": int(estimator.n_iter_[0]),
            }
        else:
            estimator = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                increasing=True,
                out_of_bounds="clip",
            )
            estimator.fit(raw_train, target, sample_weight=weights)
            predicted = estimator.predict(raw_test)
            state = {
                "x_thresholds": [float(value) for value in estimator.X_thresholds_],
                "y_thresholds": [float(value) for value in estimator.y_thresholds_],
            }
        output[test_mask] = np.clip(predicted, 0.0, 1.0)
        audits.append(
            {
                "outer_fold": int(fold),
                "method": method,
                "weight_scheme": weight_scheme,
                "target_prior": target_prior,
                "fit_row_count": int(len(train)),
                "fit_participant_count": int(len(train_participants)),
                "fit_positive_row_count": int(target.sum()),
                "fit_positive_participant_count": int(
                    train.loc[train["binary_target"].eq(1), "global_participant_id"]
                    .astype(str)
                    .nunique()
                ),
                "test_row_count": int(len(test)),
                "test_participant_count": int(len(test_participants)),
                "natural_fit_prevalence": float(target.mean()),
                "weighted_fit_prevalence": float(np.average(target, weights=weights)),
                "train_participant_sha256": _string_set_sha256(train_participants),
                "test_participant_sha256": _string_set_sha256(test_participants),
                "state": state,
            }
        )
    if not np.isfinite(output).all():
        raise MoodSocialEvaluationError(
            "cross-fitted calibration coverage is incomplete"
        )
    return output, audits


def binary_metric_summary(
    target: Sequence[int],
    probability: Sequence[float],
    *,
    sample_weight: Sequence[float] | None = None,
    decision_threshold: float = 0.5,
    ece_bin_count: int = 10,
) -> dict[str, Any]:
    """Compute weighted or natural-row binary metrics with an explicit threshold."""

    y = np.asarray(target, dtype="int64")
    score = np.asarray(probability, dtype="float64")
    weight = (
        np.ones(len(y), dtype="float64")
        if sample_weight is None
        else np.asarray(sample_weight, dtype="float64")
    )
    if (
        y.ndim != 1
        or score.ndim != 1
        or weight.ndim != 1
        or len(y) == 0
        or len(y) != len(score)
        or len(y) != len(weight)
        or not np.isfinite(score).all()
        or not np.isfinite(weight).all()
        or np.any(weight <= 0)
        or not np.all((score >= 0.0) & (score <= 1.0))
        or not set(np.unique(y)).issubset({0, 1})
    ):
        raise MoodSocialEvaluationError("binary metric inputs are invalid")
    predicted = (score >= decision_threshold).astype("int64")
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    curve, ece = calibration_curve_rows(
        y,
        score,
        sample_weight=weight,
        bin_count=ece_bin_count,
    )
    both_classes = len(np.unique(y)) == 2
    return {
        "row_count": int(len(y)),
        "positive_row_count": int(y.sum()),
        "negative_row_count": int(len(y) - y.sum()),
        "natural_prevalence": float(y.mean()),
        "weighted_prevalence": float(np.average(y, weights=weight)),
        "mean_probability": float(np.average(score, weights=weight)),
        "auprc": (
            float(average_precision_score(y, score, sample_weight=weight))
            if np.any(y == 1)
            else None
        ),
        "auroc": (
            float(roc_auc_score(y, score, sample_weight=weight))
            if both_classes
            else None
        ),
        "brier_score": float(np.average((score - y) ** 2, weights=weight)),
        "log_loss": float(log_loss(y, score, sample_weight=weight, labels=[0, 1])),
        "ece": ece,
        "decision_threshold": float(decision_threshold),
        "macro_f1": float(f1_score(y, predicted, average="macro", labels=[0, 1])),
        "f1_positive": float(f1_score(y, predicted, pos_label=1, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "confusion": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "calibration_curve": curve,
    }


def calibration_curve_rows(
    target: Sequence[int],
    probability: Sequence[float],
    *,
    sample_weight: Sequence[float] | None = None,
    bin_count: int = 10,
) -> tuple[list[dict[str, Any]], float]:
    """Return equal-width calibration bins and weighted ECE."""

    y = np.asarray(target, dtype="int64")
    score = np.asarray(probability, dtype="float64")
    weight = (
        np.ones(len(y), dtype="float64")
        if sample_weight is None
        else np.asarray(sample_weight, dtype="float64")
    )
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    indices = np.minimum(
        np.searchsorted(edges, score, side="right") - 1,
        bin_count - 1,
    )
    indices = np.maximum(indices, 0)
    total_weight = float(weight.sum())
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for bin_index in range(bin_count):
        selected = indices == bin_index
        if not selected.any():
            continue
        selected_weight = weight[selected]
        bin_weight = float(selected_weight.sum())
        mean_probability = float(np.average(score[selected], weights=selected_weight))
        observed = float(np.average(y[selected], weights=selected_weight))
        ece += (bin_weight / total_weight) * abs(mean_probability - observed)
        rows.append(
            {
                "bin_index": int(bin_index),
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "row_count": int(selected.sum()),
                "weight_sum": bin_weight,
                "mean_probability": mean_probability,
                "observed_positive_rate": observed,
            }
        )
    return rows, float(ece)


def threshold_curve(
    target: Sequence[int],
    probability: Sequence[float],
) -> pd.DataFrame:
    """Compute the full relationship at every unique observed probability."""

    y = np.asarray(target, dtype="int64")
    score = np.asarray(probability, dtype="float64")
    if len(y) == 0 or len(y) != len(score) or not np.isfinite(score).all():
        raise MoodSocialEvaluationError("threshold curve inputs are invalid")
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    sorted_target = y[order]
    total_positive = int(y.sum())
    total_negative = int(len(y) - total_positive)
    rows: list[dict[str, Any]] = []
    cumulative_tp = 0
    cumulative_fp = 0
    index = 0
    if sorted_score[0] < 1.0:
        rows.append(
            _threshold_row(
                1.0,
                tp=0,
                fp=0,
                total_positive=total_positive,
                total_negative=total_negative,
            )
        )
    while index < len(sorted_score):
        threshold = float(sorted_score[index])
        end = index
        while end < len(sorted_score) and sorted_score[end] == sorted_score[index]:
            cumulative_tp += int(sorted_target[end] == 1)
            cumulative_fp += int(sorted_target[end] == 0)
            end += 1
        rows.append(
            _threshold_row(
                threshold,
                tp=cumulative_tp,
                fp=cumulative_fp,
                total_positive=total_positive,
                total_negative=total_negative,
            )
        )
        index = end
    if sorted_score[-1] > 0.0:
        rows.append(
            _threshold_row(
                0.0,
                tp=total_positive,
                fp=total_negative,
                total_positive=total_positive,
                total_negative=total_negative,
            )
        )
    return pd.DataFrame(rows)


def validate_baseline_protection(
    repository_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-hash all five active baselines and every listed report artifact."""

    root = Path(repository_root).resolve()
    split_binding = config["split"]
    split_path = root / str(split_binding["relative_path"])
    if _sha256_file(split_path) != SPLIT_SHA256:
        raise MoodSocialEvaluationError("frozen DATA-007 split hash drifted")
    try:
        split = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MoodSocialEvaluationError("frozen DATA-007 split is unreadable") from exc
    if split.get("split_id") != SPLIT_ID:
        raise MoodSocialEvaluationError("frozen DATA-007 split ID drifted")
    schema = split.get("feature_schema_binding", {})
    schema_path = root / str(schema.get("snapshot_relative_path", ""))
    if _sha256_file(schema_path) != FEATURE_SCHEMA_SHA256:
        raise MoodSocialEvaluationError("MH-003 feature Schema hash drifted")
    input_audit: list[dict[str, Any]] = []
    for binding in split.get("inputs", []):
        item: dict[str, Any] = {"dataset_id": str(binding["dataset_id"])}
        for prefix in ("canonical", "mapping", "adapter_metadata", "artifact_manifest"):
            path_field = f"{prefix}_relative_path"
            hash_field = f"{prefix}_sha256"
            path = root / str(binding[path_field])
            actual = _sha256_file(path)
            if actual != binding[hash_field]:
                raise MoodSocialEvaluationError(
                    f"{binding['dataset_id']} {prefix} hash drifted"
                )
            item[prefix] = {
                "path": path.relative_to(root).as_posix(),
                "sha256": actual,
                "bytes": path.stat().st_size,
            }
        input_audit.append(item)
    experts: dict[str, Any] = {}
    for expert in EXPERT_ORDER:
        binding = config["baseline_protection"][expert]
        model_path = root / str(binding["model_path"])
        manifest_path = root / str(binding["manifest_path"])
        _assert_file_hash(model_path, str(binding["model_sha256"]), expert)
        _assert_file_hash(manifest_path, str(binding["manifest_sha256"]), expert)
        report_dir = root / str(binding["report_path"])
        expected_report_files = {
            "predictions.parquet": str(binding["predictions_sha256"]),
            "metrics.json": str(binding["metrics_sha256"]),
            "search_results.json": str(binding["search_results_sha256"]),
            "training_config.yaml": str(binding["training_config_sha256"]),
        }
        for relative, expected in expected_report_files.items():
            _assert_file_hash(report_dir / relative, expected, expert)
        artifact_manifest_path = report_dir / "artifacts.json"
        try:
            artifact_manifest = json.loads(
                artifact_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MoodSocialEvaluationError(
                f"{expert} report artifact manifest is unreadable"
            ) from exc
        listed = artifact_manifest.get("artifacts")
        if not isinstance(listed, list) or artifact_manifest.get(
            "artifact_count"
        ) != len(listed):
            raise MoodSocialEvaluationError(
                f"{expert} report artifact manifest is invalid"
            )
        for item in listed:
            path = report_dir / str(item["path"])
            if path.stat().st_size != int(item["bytes"]) or _sha256_file(path) != str(
                item["sha256"]
            ):
                raise MoodSocialEvaluationError(f"{expert} report artifact drifted")
        report_rows = _tree_hash_rows(report_dir)
        experts[expert] = {
            "model_id": binding["model_id"],
            "run_id": binding["run_id"],
            "model": _file_audit(root, model_path),
            "manifest": _file_audit(root, manifest_path),
            "report_file_count": len(report_rows),
            "report_tree_sha256": _rows_sha256(report_rows),
            "report_artifacts_manifest_sha256": _sha256_file(artifact_manifest_path),
            "protected_report_core": {
                name: expected for name, expected in expected_report_files.items()
            },
        }
    return {
        "protection_version": "mood-social-eval-baseline-protection-v1",
        "split": _file_audit(root, split_path),
        "feature_schema": _file_audit(root, schema_path),
        "inputs": input_audit,
        "experts": experts,
    }


def run_expert_evaluation(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    report_dir: str | Path | None = None,
    overwrite: bool = False,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run EVAL-001 and atomically publish a report-only artifact directory."""

    started = datetime.now(timezone.utc)
    root = Path(repository_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_evaluation_config(config_file)
    configured_report = root / str(config["reporting"]["report_directory"])
    destination = (
        Path(report_dir).resolve() if report_dir is not None else configured_report
    )
    _validate_report_destination(root, destination)
    if destination.exists() and not overwrite:
        raise MoodSocialEvaluationError("EVAL-001 report directory already exists")
    protection_before = validate_baseline_protection(root, config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    try:
        (stage / "figures").mkdir(parents=True, exist_ok=True)
        _write_bytes(stage / "evaluation_config.yaml", config_file.read_bytes())
        _write_json(stage / "baseline_protection.json", protection_before)
        predictions: list[pd.DataFrame] = []
        metric_rows: list[dict[str, Any]] = []
        source_metric_rows: list[dict[str, Any]] = []
        threshold_rows: list[pd.DataFrame] = []
        calibration_rows: list[dict[str, Any]] = []
        calibration_fits: dict[str, Any] = {}
        source_frames: dict[str, pd.DataFrame] = {}
        for expert in EXPERT_ORDER:
            binding = config["baseline_protection"][expert]
            source = pd.read_parquet(
                root / binding["report_path"] / "predictions.parquet"
            )
            available = source.loc[source["expert_mask"].eq(1)].reset_index(drop=True)
            _validate_published_oof(available, expert)
            evaluated, fits = _evaluate_calibration_candidates(
                available,
                config=config,
                expert=expert,
            )
            source_frames[expert] = available
            predictions.append(_published_candidate_frame(evaluated, expert))
            calibration_fits[expert] = fits
            variants = _variant_columns(evaluated)
            for variant in variants:
                fit_scheme, target_prior = _variant_weight_definition(variant)
                natural = binary_metric_summary(
                    evaluated["binary_target"],
                    evaluated[variant],
                    decision_threshold=float(
                        config["threshold_audit"]["reference_threshold"]
                    ),
                    ece_bin_count=int(config["calibration"]["ece_bin_count"]),
                )
                fit_weight = calibration_sample_weights(
                    evaluated,
                    fit_scheme,
                    target_prior=target_prior,
                )
                weighted = binary_metric_summary(
                    evaluated["binary_target"],
                    evaluated[variant],
                    sample_weight=fit_weight,
                    decision_threshold=float(
                        config["threshold_audit"]["reference_threshold"]
                    ),
                    ece_bin_count=int(config["calibration"]["ece_bin_count"]),
                )
                metric_rows.append(
                    _flatten_metric_row(
                        expert,
                        variant,
                        fit_scheme,
                        target_prior,
                        natural,
                        weighted,
                    )
                )
                for bin_row in natural["calibration_curve"]:
                    calibration_rows.append(
                        {
                            "expert": expert,
                            "probability_variant": variant,
                            "dataset_id": "__pooled__",
                            **bin_row,
                        }
                    )
                curve = threshold_curve(evaluated["binary_target"], evaluated[variant])
                curve.insert(0, "probability_variant", variant)
                curve.insert(0, "expert", expert)
                threshold_rows.append(curve)
                for dataset_id, group in evaluated.groupby("dataset_id", sort=True):
                    metrics = binary_metric_summary(
                        group["binary_target"],
                        group[variant],
                        decision_threshold=float(
                            config["threshold_audit"]["reference_threshold"]
                        ),
                        ece_bin_count=int(config["calibration"]["ece_bin_count"]),
                    )
                    source_metric_rows.append(
                        _flatten_source_metric_row(
                            expert,
                            variant,
                            str(dataset_id),
                            int(group["global_participant_id"].astype(str).nunique()),
                            metrics,
                        )
                    )
        candidate_predictions = pd.concat(predictions, ignore_index=True)
        metrics_frame = pd.DataFrame(metric_rows)
        source_metrics_frame = pd.DataFrame(source_metric_rows)
        threshold_frame = pd.concat(threshold_rows, ignore_index=True)
        calibration_frame = pd.DataFrame(calibration_rows)
        _write_parquet(stage / "calibration_candidates.parquet", candidate_predictions)
        _write_parquet(stage / "metric_summary.parquet", metrics_frame)
        _write_parquet(stage / "source_metrics.parquet", source_metrics_frame)
        _write_parquet(stage / "threshold_curves.parquet", threshold_frame)
        _write_parquet(stage / "calibration_curves.parquet", calibration_frame)
        _write_json(stage / "calibration_fits.json", calibration_fits)
        loso_predictions, loso_metrics, loso_audit = _run_leave_one_source_out(
            root,
            config,
            source_frames,
        )
        _write_parquet(stage / "leave_one_source_predictions.parquet", loso_predictions)
        _write_json(stage / "leave_one_source_metrics.json", loso_metrics)
        _write_json(stage / "leave_one_source_preprocessing.json", loso_audit)
        physiology_uncertainty = _physiology_uncertainty(
            source_frames["physiology"],
            seed=int(config["random_seed"]),
        )
        _write_json(stage / "physiology_uncertainty.json", physiology_uncertainty)
        _make_figures(
            stage / "figures",
            metrics_frame,
            source_metrics_frame,
            threshold_frame,
            loso_metrics,
        )
        metrics_payload = _metrics_payload(
            config,
            metrics_frame,
            source_metrics_frame,
            loso_metrics,
            physiology_uncertainty,
        )
        _write_json(stage / "metrics.json", metrics_payload)
        _write_bytes(
            stage / "report.md",
            _technical_report_markdown(
                config,
                metrics_frame,
                source_metrics_frame,
                loso_metrics,
                physiology_uncertainty,
            ).encode("utf-8"),
        )
        _write_json(
            stage / "artifact.json",
            _report_artifact_payload(
                config,
                metrics_frame,
                loso_metrics,
                physiology_uncertainty,
            ),
        )
        protection_after = validate_baseline_protection(root, config)
        if protection_after != protection_before:
            raise MoodSocialEvaluationError("active baseline changed during EVAL-001")
        ended = datetime.now(timezone.utc)
        run_payload = {
            "run_manifest_version": "mood-social-evaluation-run-v1",
            "run_id": config["run_id"],
            "task_id": config["task_id"],
            "status": "completed",
            "started_at_utc": started.isoformat(),
            "ended_at_utc": ended.isoformat(),
            "duration_seconds": (ended - started).total_seconds(),
            "command": list(command or ()),
            "evaluation_version": config["evaluation_version"],
            "frozen_document_version": config["frozen_document_version"],
            "split": config["split"],
            "read_only_boundary": config["reporting"],
            "baseline_protection_before_sha256": _canonical_payload_sha256(
                protection_before
            ),
            "baseline_protection_after_sha256": _canonical_payload_sha256(
                protection_after
            ),
            "calibration_candidate_count_per_expert": 12,
            "probability_variant_count_per_expert": 14,
            "leave_one_source_out": config["leave_one_source_out"],
            "physiology_uncertainty_policy": (
                "single_source_participant_bootstrap_no_loso"
            ),
            "production_threshold_selected": False,
            "attention_level_boundaries_modified": False,
            "diagnostic_models_published": False,
            "code_version": _git_state(root),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "code_bindings": {
                "evaluation": _sha256_file(Path(__file__)),
                "config": _sha256_file(config_file),
            },
            "warnings": metrics_payload["warnings"],
            "conclusion": (
                "Read-only diagnostics completed; active experts remain unchanged and "
                "no production calibration or threshold was selected."
            ),
        }
        _write_json(stage / "run.json", run_payload)
        _write_json(stage / "artifacts.json", _artifact_manifest(stage))
        _publish_stage(stage, destination, overwrite=overwrite)
    except Exception:
        _remove_stage(stage)
        raise
    return {
        "status": "completed",
        "run_id": config["run_id"],
        "report_dir": destination.as_posix(),
        "metric_row_count": int(len(metrics_frame)),
        "source_metric_row_count": int(len(source_metrics_frame)),
        "threshold_row_count": int(len(threshold_frame)),
        "loso_prediction_row_count": int(len(loso_predictions)),
        "baseline_protection_sha256": _canonical_payload_sha256(protection_after),
    }


def _evaluate_calibration_candidates(
    frame: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    expert: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = frame.copy()
    result = result.rename(columns={"calibrated_probability": "active_calibrated"})
    result["raw"] = pd.to_numeric(result["raw_probability"]).to_numpy(dtype="float64")
    result = result.drop(columns="raw_probability")
    fits: dict[str, Any] = {}
    candidates: list[CalibrationCandidate] = []
    for method in config["calibration"]["methods"]:
        for scheme in config["calibration"]["weight_schemes"]:
            candidates.append(
                CalibrationCandidate(
                    variant=f"cf_{method}__{scheme}",
                    method=method,
                    weight_scheme=scheme,
                )
            )
        for prior in config["calibration"]["target_prior_sensitivity"]:
            prior_value = float(prior)
            candidates.append(
                CalibrationCandidate(
                    variant=f"cf_{method}__target_prior_{_prior_token(prior_value)}",
                    method=method,
                    weight_scheme="target_prior",
                    target_prior=prior_value,
                )
            )
    for candidate in candidates:
        predicted, audit = cross_fit_calibrator(
            result.rename(columns={"raw": "raw_probability"}),
            method=candidate.method,
            weight_scheme=candidate.weight_scheme,
            target_prior=candidate.target_prior,
            random_seed=int(config["random_seed"]),
        )
        result[candidate.variant] = predicted
        fits[candidate.variant] = {
            "expert": expert,
            "method": candidate.method,
            "weight_scheme": candidate.weight_scheme,
            "target_prior": candidate.target_prior,
            "folds": audit,
        }
    return result, fits


def _variant_columns(frame: pd.DataFrame) -> list[str]:
    return [
        "raw",
        "active_calibrated",
        *sorted(
            [column for column in frame.columns if column.startswith("cf_")],
            key=lambda value: value.encode("utf-8"),
        ),
    ]


def _published_candidate_frame(frame: pd.DataFrame, expert: str) -> pd.DataFrame:
    columns = [
        "prediction_id",
        "dataset_id",
        "canonical_row_index",
        "binary_target",
        "outer_fold",
        *_variant_columns(frame),
    ]
    result = frame[columns].copy()
    result.insert(0, "expert", expert)
    result.insert(0, "prediction_schema_version", PREDICTION_VERSION)
    return result


def _variant_weight_definition(variant: str) -> tuple[str, float | None]:
    if variant in {"raw", "active_calibrated"}:
        return "natural_sample", None
    if variant.endswith("__frozen_four_level"):
        return "frozen_four_level", None
    if variant.endswith("__natural_sample"):
        return "natural_sample", None
    marker = "__target_prior_"
    if marker in variant:
        return "target_prior", float(variant.split(marker, 1)[1].replace("p", "."))
    raise MoodSocialEvaluationError("unknown probability variant")


def _flatten_metric_row(
    expert: str,
    variant: str,
    fit_scheme: str,
    target_prior: float | None,
    natural: Mapping[str, Any],
    weighted: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "metrics_version": METRICS_VERSION,
        "expert": expert,
        "probability_variant": variant,
        "calibration_fit_weight_scheme": fit_scheme,
        "target_prior": target_prior,
        "row_count": natural["row_count"],
        "positive_row_count": natural["positive_row_count"],
        "natural_prevalence": natural["natural_prevalence"],
    }
    for field in (
        "mean_probability",
        "auprc",
        "auroc",
        "brier_score",
        "log_loss",
        "ece",
        "macro_f1",
        "f1_positive",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
    ):
        row[f"natural_{field}"] = natural[field]
        row[f"fit_weighted_{field}"] = weighted[field]
    row["fit_weighted_prevalence"] = weighted["weighted_prevalence"]
    row.update(
        {f"natural_{name}": value for name, value in natural["confusion"].items()}
    )
    return row


def _flatten_source_metric_row(
    expert: str,
    variant: str,
    dataset_id: str,
    participant_count: int,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "metrics_version": METRICS_VERSION,
        "expert": expert,
        "probability_variant": variant,
        "dataset_id": dataset_id,
        "participant_count": participant_count,
        "row_count": metrics["row_count"],
        "positive_row_count": metrics["positive_row_count"],
        "prevalence": metrics["natural_prevalence"],
        "mean_probability": metrics["mean_probability"],
        "auprc": metrics["auprc"],
        "auroc": metrics["auroc"],
        "brier_score": metrics["brier_score"],
        "log_loss": metrics["log_loss"],
        "ece": metrics["ece"],
        "macro_f1": metrics["macro_f1"],
        "f1_positive": metrics["f1_positive"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        **metrics["confusion"],
    }


def _run_leave_one_source_out(
    root: Path,
    config: Mapping[str, Any],
    published_oof: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    from elderly_monitoring.modules.mental_health.mood_social.experts import (
        activity,
        joint,
        sleep,
        social_context,
    )

    modules = {
        "activity": activity,
        "sleep": sleep,
        "joint": joint,
        "social_context": social_context,
    }
    prediction_parts: list[pd.DataFrame] = []
    metrics: dict[str, Any] = {"version": LOSO_VERSION, "experts": {}}
    audit: dict[str, Any] = {
        "version": LOSO_VERSION,
        "policy": config["leave_one_source_out"],
        "experts": {},
    }
    for expert in LOSO_EXPERT_ORDER:
        module = modules[expert]
        inputs, training_config = _load_loso_inputs_and_config(
            root, config, expert, module
        )
        artifact_path = (
            root
            / config["baseline_protection"][expert]["report_path"]
            / "model_artifact.json"
        )
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        hyperparameters = artifact["hyperparameters"]
        dataset_ids = _expert_dataset_ids(module, expert)
        expert_audit: list[dict[str, Any]] = []
        source_results: dict[str, Any] = {}
        assignments = inputs.assignments.copy()
        assignments["outer_fold"] = pd.to_numeric(
            assignments["outer_fold"], errors="raise"
        ).astype("int64")
        for heldout in dataset_ids:
            source_parts: list[pd.DataFrame] = []
            fold_audits: list[dict[str, Any]] = []
            for outer_fold in range(5):
                outer_train = assignments.loc[assignments["outer_fold"].ne(outer_fold)]
                training_keys = set(
                    outer_train.loc[
                        outer_train["dataset_id"].astype(str).ne(heldout),
                        "global_participant_id",
                    ].astype(str)
                )
                all_outer_train_keys = set(
                    outer_train["global_participant_id"].astype(str)
                )
                test_keys = set(
                    assignments.loc[
                        assignments["dataset_id"].astype(str).eq(heldout)
                        & assignments["outer_fold"].eq(outer_fold),
                        "global_participant_id",
                    ].astype(str)
                )
                if not training_keys or not test_keys or training_keys & test_keys:
                    raise MoodSocialEvaluationError("LOSO split construction failed")
                train_table, test_table, adaptation_audit = _loso_tables(
                    expert,
                    module,
                    inputs,
                    training_keys,
                    test_keys,
                    all_outer_train_keys,
                    heldout,
                )
                available_train = _available_rows(module, expert, train_table)
                available_test = _available_rows(module, expert, test_table)
                preprocessor = _preprocessor_class(module, expert).fit(available_train)
                train_matrix = preprocessor.transform(available_train)
                test_matrix = preprocessor.transform(available_test)
                weights = module.compute_training_sample_weights(available_train)
                classifier = _fit_loso_classifier(
                    module,
                    expert,
                    train_matrix,
                    pd.to_numeric(available_train["binary_target"]).to_numpy(
                        dtype="int64"
                    ),
                    weights,
                    hyperparameters,
                    training_config,
                    preprocessor,
                    training_identity=(
                        f"eval001:{expert}:heldout={heldout}:outer={outer_fold}"
                    ),
                )
                raw = np.asarray(
                    classifier.predict_proba(test_matrix)[:, 1], dtype="float64"
                )
                fold_output = available_test[
                    [
                        "prediction_id",
                        "dataset_id",
                        "canonical_row_index",
                        "binary_target",
                    ]
                ].copy()
                fold_output.insert(0, "loso_version", LOSO_VERSION)
                fold_output.insert(1, "expert", expert)
                fold_output.insert(2, "heldout_source", heldout)
                fold_output["outer_fold"] = outer_fold
                fold_output["raw_probability"] = raw
                source_parts.append(fold_output)
                fold_audits.append(
                    {
                        "heldout_source": heldout,
                        "outer_fold": outer_fold,
                        "training_source_ids": sorted(
                            set(available_train["dataset_id"].astype(str))
                        ),
                        "training_row_count": int(len(available_train)),
                        "training_participant_count": int(
                            available_train["global_participant_id"]
                            .astype(str)
                            .nunique()
                        ),
                        "test_row_count": int(len(available_test)),
                        "test_participant_count": int(
                            available_test["global_participant_id"]
                            .astype(str)
                            .nunique()
                        ),
                        "selected_features": list(preprocessor.selected_features),
                        "training_participant_sha256": _string_set_sha256(
                            set(available_train["global_participant_id"].astype(str))
                        ),
                        "test_participant_sha256": _string_set_sha256(
                            set(available_test["global_participant_id"].astype(str))
                        ),
                        "target_source_adaptation": adaptation_audit,
                    }
                )
            source_output = pd.concat(source_parts, ignore_index=True).sort_values(
                ["outer_fold", "prediction_id"], kind="stable"
            )
            prediction_parts.append(source_output)
            loso_metric = binary_metric_summary(
                source_output["binary_target"], source_output["raw_probability"]
            )
            current = published_oof[expert].loc[
                published_oof[expert]["dataset_id"].astype(str).eq(heldout)
            ]
            current_metric = binary_metric_summary(
                current["binary_target"], current["raw_probability"]
            )
            source_results[heldout] = {
                "heldout_source": heldout,
                "raw_loso": _compact_metrics(loso_metric),
                "published_raw_oof": _compact_metrics(current_metric),
                "delta_loso_minus_published": {
                    field: _safe_difference(loso_metric[field], current_metric[field])
                    for field in ("auprc", "auroc", "brier_score", "ece")
                },
                "fold_count": 5,
                "hyperparameter_policy": (
                    "fixed_active_hyperparameters_frozen_before_audit"
                ),
                "calibration": "none_raw_probability_only",
            }
            expert_audit.extend(fold_audits)
        metrics["experts"][expert] = source_results
        audit["experts"][expert] = expert_audit
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["expert", "heldout_source", "outer_fold", "prediction_id"],
        kind="stable",
    )
    return predictions.reset_index(drop=True), metrics, audit


def _load_loso_inputs_and_config(
    root: Path,
    config: Mapping[str, Any],
    expert: str,
    module: Any,
) -> tuple[Any, Any]:
    report = root / config["baseline_protection"][expert]["report_path"]
    config_path = report / "training_config.yaml"
    if expert == "activity":
        return module.load_activity_training_inputs(
            root
        ), module.load_activity_training_config(config_path)
    if expert == "sleep":
        return module.load_sleep_training_inputs(
            root
        ), module.load_sleep_training_config(config_path)
    if expert == "joint":
        return module.load_joint_training_inputs(
            root
        ), module.load_joint_training_config(config_path)
    return (
        module.load_social_context_training_inputs(root),
        module.load_social_context_training_config(config_path),
    )


def _expert_dataset_ids(module: Any, expert: str) -> tuple[str, ...]:
    if expert == "activity":
        return tuple(module.ACTIVITY_DATASET_IDS)
    if expert == "sleep":
        return tuple(module.SLEEP_DATASET_IDS)
    if expert == "joint":
        return tuple(module.JOINT_DATASET_IDS)
    return tuple(module.SOCIAL_CONTEXT_DATASET_IDS)


def _loso_tables(
    expert: str,
    module: Any,
    inputs: Any,
    training_keys: set[str],
    test_keys: set[str],
    all_outer_train_keys: set[str],
    heldout: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dataset_ids = _expert_dataset_ids(module, expert)
    training_sources = tuple(value for value in dataset_ids if value != heldout)
    if expert in {"activity", "joint"}:
        ecdfs = module._fit_ecdfs(inputs, all_outer_train_keys)
        train = _ecdf_partition(
            module,
            expert,
            inputs,
            ecdfs,
            training_keys,
            training_sources,
        )
        test = _ecdf_partition(
            module,
            expert,
            inputs,
            ecdfs,
            test_keys,
            (heldout,),
        )
        target_ecdf = ecdfs[heldout]
        adaptation = {
            "policy": "unlabeled_outer_training_participants_only",
            "heldout_source": heldout,
            "fit_participant_count": int(target_ecdf.training_participant_count),
            "fit_participant_sha256": str(target_ecdf.training_participant_sha256),
            "test_participants_excluded": True,
            "labels_used": False,
        }
        return train, test, adaptation
    train = _identity_partition(
        module,
        expert,
        inputs,
        training_keys,
        training_sources,
    )
    test = _identity_partition(
        module,
        expert,
        inputs,
        test_keys,
        (heldout,),
    )
    return train, test, {"policy": "none_identity_features", "labels_used": False}


def _ecdf_partition(
    module: Any,
    expert: str,
    inputs: Any,
    ecdfs: Mapping[str, Any],
    participant_keys: set[str],
    dataset_ids: Sequence[str],
) -> pd.DataFrame:
    feature_columns, mask_columns = _feature_and_mask_columns(module, expert)
    parts: list[pd.DataFrame] = []
    for order, dataset_id in enumerate(dataset_ids):
        base = inputs.frames[dataset_id]
        selected = base[
            base["global_participant_id"].astype(str).isin(participant_keys)
        ]
        if selected.empty:
            continue
        transformed = ecdfs[dataset_id].transform(selected)
        part = transformed[
            [
                "dataset_id",
                "global_participant_id",
                "binary_target",
                *feature_columns,
                *mask_columns,
            ]
        ].copy()
        part["canonical_row_index"] = selected.index.to_numpy(dtype="int64")
        part["prediction_id"] = (
            part["dataset_id"].astype(str)
            + "::row="
            + part["canonical_row_index"].astype(str)
        )
        part["_dataset_order"] = order
        parts.append(part)
    if not parts:
        raise MoodSocialEvaluationError("LOSO ECDF partition is empty")
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["_dataset_order", "canonical_row_index"], kind="stable")
        .drop(columns="_dataset_order")
        .reset_index(drop=True)
    )


def _identity_partition(
    module: Any,
    expert: str,
    inputs: Any,
    participant_keys: set[str],
    dataset_ids: Sequence[str],
) -> pd.DataFrame:
    feature_columns, mask_columns = _feature_and_mask_columns(module, expert)
    parts: list[pd.DataFrame] = []
    for order, dataset_id in enumerate(dataset_ids):
        base = inputs.frames[dataset_id]
        selected = base[
            base["global_participant_id"].astype(str).isin(participant_keys)
        ]
        if selected.empty:
            continue
        part = selected[
            [
                "dataset_id",
                "global_participant_id",
                "binary_target",
                *feature_columns,
                *mask_columns,
            ]
        ].copy()
        part["canonical_row_index"] = selected.index.to_numpy(dtype="int64")
        part["prediction_id"] = (
            part["dataset_id"].astype(str)
            + "::row="
            + part["canonical_row_index"].astype(str)
        )
        part["_dataset_order"] = order
        parts.append(part)
    if not parts:
        raise MoodSocialEvaluationError("LOSO identity partition is empty")
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["_dataset_order", "canonical_row_index"], kind="stable")
        .drop(columns="_dataset_order")
        .reset_index(drop=True)
    )


def _feature_and_mask_columns(
    module: Any, expert: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if expert == "activity":
        return tuple(module.ACTIVITY_FEATURE_COLUMNS), tuple(
            module.ACTIVITY_MASK_COLUMNS
        )
    if expert == "sleep":
        return tuple(module.SLEEP_FEATURE_COLUMNS), tuple(module.SLEEP_MASK_COLUMNS)
    if expert == "joint":
        return tuple(module.JOINT_FEATURE_COLUMNS), tuple(module.JOINT_MASK_COLUMNS)
    return (
        tuple(module.SOCIAL_CONTEXT_FEATURE_COLUMNS),
        tuple(module.SOCIAL_CONTEXT_MASK_COLUMNS),
    )


def _available_rows(module: Any, expert: str, frame: pd.DataFrame) -> pd.DataFrame:
    if expert == "activity":
        return module._available_activity_rows(frame)
    if expert == "sleep":
        return module._available_sleep_rows(frame)
    if expert == "joint":
        return module._available_joint_rows(frame)
    return module._available_social_rows(frame)


def _preprocessor_class(module: Any, expert: str) -> Any:
    if expert == "activity":
        return module.NumericFoldPreprocessor
    if expert == "sleep":
        return module.SleepNumericFoldPreprocessor
    if expert == "joint":
        return module.JointNumericFoldPreprocessor
    return module.SocialContextFoldPreprocessor


def _fit_loso_classifier(
    module: Any,
    expert: str,
    matrix: pd.DataFrame,
    target: np.ndarray,
    sample_weight: np.ndarray,
    hyperparameters: Mapping[str, Any],
    training_config: Any,
    preprocessor: Any,
    *,
    training_identity: str,
) -> Any:
    if expert == "social_context":
        return module._fit_classifier(
            matrix,
            target,
            sample_weight,
            hyperparameters,
            training_config,
            preprocessor,
            training_identity=training_identity,
        )
    return module._fit_classifier(
        matrix,
        target,
        sample_weight,
        hyperparameters,
        training_config,
    )


def _physiology_uncertainty(frame: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    variants = {"raw": "raw_probability", "active_calibrated": "calibrated_probability"}
    rng = np.random.default_rng(seed)
    participants = sorted(
        set(frame["global_participant_id"].astype(str)),
        key=lambda value: value.encode("utf-8"),
    )
    groups = {
        participant: frame.loc[
            frame["global_participant_id"].astype(str).eq(participant)
        ].reset_index(drop=True)
        for participant in participants
    }
    result: dict[str, Any] = {
        "policy": "participant_bootstrap_percentile_95",
        "bootstrap_replicates": 1000,
        "seed": seed,
        "source_count": 1,
        "dataset_id": "resilient",
        "participant_count": len(participants),
        "positive_row_count": int(frame["binary_target"].sum()),
        "variants": {},
        "warning": (
            "Single-source small-sample evidence; intervals are descriptive and do not "
            "establish transportability."
        ),
    }
    for name, column in variants.items():
        values = {metric: [] for metric in ("auprc", "auroc", "brier_score")}
        valid_replicates = 0
        for _ in range(1000):
            sampled = rng.choice(participants, size=len(participants), replace=True)
            bootstrap = pd.concat(
                [groups[value] for value in sampled], ignore_index=True
            )
            if bootstrap["binary_target"].nunique() < 2:
                continue
            metric = binary_metric_summary(
                bootstrap["binary_target"], bootstrap[column]
            )
            valid_replicates += 1
            for field in values:
                values[field].append(float(metric[field]))
        point = binary_metric_summary(frame["binary_target"], frame[column])
        result["variants"][name] = {
            "point": _compact_metrics(point),
            "valid_bootstrap_replicates": valid_replicates,
            "intervals_95": {
                field: {
                    "lower": float(np.quantile(series, 0.025)),
                    "upper": float(np.quantile(series, 0.975)),
                }
                for field, series in values.items()
            },
        }
    return result


def _make_figures(
    directory: Path,
    metrics: pd.DataFrame,
    source_metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    loso_metrics: Mapping[str, Any],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
            "axes.edgecolor": "#334155",
            "axes.grid": True,
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.7,
        }
    )
    variants = [
        "raw",
        "active_calibrated",
        "cf_platt__natural_sample",
        "cf_isotonic__natural_sample",
    ]
    colors = ["#2563eb", "#d97706", "#64748b", "#db2777"]
    subset = metrics.loc[metrics["probability_variant"].isin(variants)].copy()
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    x = np.arange(len(EXPERT_ORDER), dtype="float64")
    width = 0.19
    for index, variant in enumerate(variants):
        rows = subset.loc[subset["probability_variant"].eq(variant)].set_index("expert")
        values = [
            float(rows.loc[expert, "natural_brier_score"]) for expert in EXPERT_ORDER
        ]
        ax.bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=variant,
            color=colors[index],
            edgecolor="#1f2937",
            linewidth=0.5,
        )
    ax.set_xticks(x, [DISPLAY_NAMES[value] for value in EXPERT_ORDER], rotation=15)
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Natural-sample OOF Brier score by probability representation")
    ax.set_ylim(bottom=0)
    ax.legend(ncol=2, frameon=False)
    _save_figure(fig, directory / "calibration_brier_comparison.png")

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    for axis, expert in zip(axes.flat, EXPERT_ORDER, strict=False):
        rows = thresholds.loc[
            thresholds["expert"].eq(expert)
            & thresholds["probability_variant"].eq("active_calibrated")
        ].sort_values("threshold")
        axis.plot(
            rows["threshold"], rows["sensitivity"], label="Sensitivity", color="#2563eb"
        )
        axis.plot(
            rows["threshold"],
            rows["specificity"],
            label="Specificity",
            color="#d97706",
            linestyle="--",
        )
        axis.plot(
            rows["threshold"],
            rows["f1_positive"],
            label="F1 positive",
            color="#64748b",
            linestyle=":",
        )
        axis.axvline(0.5, color="#111827", linewidth=0.8, alpha=0.65)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_title(DISPLAY_NAMES[expert])
        axis.set_xlabel("Diagnostic threshold")
        axis.set_ylabel("Metric")
    axes.flat[-1].axis("off")
    axes.flat[0].legend(ncol=3, frameon=False, loc="lower center")
    fig.suptitle(
        "Published calibrated OOF threshold relationships (0.5 is diagnostic only)"
    )
    _save_figure(fig, directory / "threshold_tradeoff.png")

    active = source_metrics.loc[
        source_metrics["probability_variant"].eq("active_calibrated")
    ].copy()
    active["label"] = active["expert"] + "\n" + active["dataset_id"]
    active = active.sort_values(["expert", "dataset_id"], kind="stable")
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    y = np.arange(len(active))
    ax.barh(y, active["auprc"], color="#2563eb", edgecolor="#1f2937", linewidth=0.4)
    ax.set_yticks(y, active["label"])
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("AUPRC")
    ax.set_title("Published calibrated OOF AUPRC by expert and source")
    _save_figure(fig, directory / "source_auprc.png")

    rows: list[dict[str, Any]] = []
    for expert, source_map in loso_metrics["experts"].items():
        for source, item in source_map.items():
            rows.append(
                {
                    "label": f"{expert}\n{source}",
                    "published": item["published_raw_oof"]["auprc"],
                    "loso": item["raw_loso"]["auprc"],
                }
            )
    loso_frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    x = np.arange(len(loso_frame))
    ax.bar(
        x - 0.2,
        loso_frame["published"],
        0.4,
        label="Published raw OOF",
        color="#2563eb",
    )
    ax.bar(
        x + 0.2,
        loso_frame["loso"],
        0.4,
        label="Label-source-held-out raw",
        color="#d97706",
    )
    ax.set_xticks(x, loso_frame["label"], rotation=55, ha="right")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("AUPRC")
    ax.set_title("Fixed-configuration leave-one-source-out AUPRC stress test")
    ax.legend(frameon=False)
    _save_figure(fig, directory / "leave_one_source_auprc.png")


def _metrics_payload(
    config: Mapping[str, Any],
    metrics: pd.DataFrame,
    source_metrics: pd.DataFrame,
    loso_metrics: Mapping[str, Any],
    physiology_uncertainty: Mapping[str, Any],
) -> dict[str, Any]:
    core_variants = {
        expert: {
            variant: _json_safe_record(
                metrics.loc[
                    metrics["expert"].eq(expert)
                    & metrics["probability_variant"].eq(variant)
                ].iloc[0]
            )
            for variant in (
                "raw",
                "active_calibrated",
                "cf_platt__natural_sample",
                "cf_isotonic__natural_sample",
            )
        }
        for expert in EXPERT_ORDER
    }
    return {
        "metrics_version": METRICS_VERSION,
        "run_id": config["run_id"],
        "task_id": config["task_id"],
        "split_id": config["split"]["split_id"],
        "split_sha256": config["split"]["sha256"],
        "probability_interpretation": (
            "calibration sensitivity only; no deployment prior is available"
        ),
        "reference_threshold": config["threshold_audit"]["reference_threshold"],
        "reference_threshold_is_production_choice": False,
        "core_probability_variants": core_variants,
        "source_metric_row_count": int(len(source_metrics)),
        "leave_one_source_out": loso_metrics,
        "physiology_uncertainty": physiology_uncertainty,
        "warnings": [
            "No project deployment prevalence is available; none of the candidate probabilities is an absolute risk estimate.",
            "Threshold 0.5 is retained only as a diagnostic reference and is not a production threshold decision.",
            "The frozen four-level weights imply a weighted 0.5 class prior and are separated from natural-sample calibration.",
            "LOSO uses hyperparameters frozen before this audit; it is a stress test, not source-independent model selection.",
            "Activity and Joint LOSO use label-free heldout-source ECDFs fitted only on outer-training participants; test participants are excluded.",
            "Physiology has one source, 71 evaluable rows and 9 evaluable positives; no LOSO result is fabricated.",
            "All five standalone active experts and their prior reports remain byte-for-byte protected.",
        ],
    }


def _legacy_report_markdown(
    config: Mapping[str, Any],
    metrics: pd.DataFrame,
    source_metrics: pd.DataFrame,
    loso_metrics: Mapping[str, Any],
    physiology_uncertainty: Mapping[str, Any],
) -> str:
    lines = [
        "# EVAL-001 五专家校准、阈值与跨来源泛化审计",
        "",
        "## 技术摘要",
        "",
        "本次审计基于五个专家已经发布的严格参与者级外层 OOF；所有新校准器均按 DATA-007 外层折交叉拟合，外层测试参与者不进入对应校准器。五个 active Joblib、manifest、既有 OOF、指标、搜索结果和报告均保持只读，审计未选择生产阈值，也未改变 0.25/0.45/0.65 的接口等级边界。",
        "",
        "训练时的四级等权会把校准拟合分布的加权阳性率推到 0.5；自然样本与目标先验敏感性给出了明显不同的概率尺度。由于当前没有项目部署阳性先验，本报告中的概率只能视为筛查证据和敏感性结果，不能解释为绝对患病概率。",
        "",
        "固定 0.5 阈值在若干专家上表现失衡；完整阈值曲线已保存，但本任务只展示权衡，不做生产选择。跨来源结果差异明显，说明后续融合应保留来源、置信度和 mask 审计，并避免把单一总体指标当作部署保证。",
        "",
        "## 原始、现行校准与自然样本交叉校准",
        "",
        "下表全部按自然 OOF 行评价。Platt/Isotonic 行只改变校准权重与映射，不重训基础专家。",
        "",
        "| 专家 | 表示 | AUPRC | AUROC | Brier | ECE | 0.5 Sens. | 0.5 Spec. |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    core_variants = (
        ("raw", "原始"),
        ("active_calibrated", "现行校准"),
        ("cf_platt__natural_sample", "自然样本 Platt"),
        ("cf_isotonic__natural_sample", "自然样本 Isotonic"),
    )
    for expert in EXPERT_ORDER:
        for variant, label in core_variants:
            row = metrics.loc[
                metrics["expert"].eq(expert)
                & metrics["probability_variant"].eq(variant)
            ].iloc[0]
            lines.append(
                "| {expert} | {label} | {auprc} | {auroc} | {brier} | {ece} | {sens} | {spec} |".format(
                    expert=DISPLAY_NAMES[expert],
                    label=label,
                    auprc=_fmt(row["natural_auprc"]),
                    auroc=_fmt(row["natural_auroc"]),
                    brier=_fmt(row["natural_brier_score"]),
                    ece=_fmt(row["natural_ece"]),
                    sens=_fmt(row["natural_sensitivity"]),
                    spec=_fmt(row["natural_specificity"]),
                )
            )
    lines.extend(
        [
            "",
            "![校准 Brier 对比](figures/calibration_brier_comparison.png)",
            "",
            "图中比较相同严格 OOF 上的四种概率表示；较低 Brier 表示概率尺度与自然样本标签更接近，但不构成生产采用决定。",
            "",
            "## 阈值关系只用于诊断",
            "",
            "`threshold_curves.parquet` 对每个专家、每种概率表示保存所有观测唯一阈值及 Sensitivity、Specificity、Precision、Recall、阳性 F1、Macro-F1、FP、FN、TP 和 TN。下图仅展示现行校准概率；0.5 竖线是历史诊断参考，不是本次选择的工作阈值。",
            "",
            "![阈值权衡](figures/threshold_tradeoff.png)",
            "",
            "## 来源差异与标签来源留出压力测试",
            "",
            "来源级指标使用同一严格 OOF 逐来源计算。Activity、Sleep、Joint 和 SocialContext 另外进行了五个外层折的标签来源留出重拟合：每折只使用其他来源且属于外层训练折的标签行，预处理只在这些训练行拟合；活动 ECDF 对目标来源只使用其外层训练参与者的无标签数值，测试参与者完全排除。模型超参数固定为审计前 active 配置，因此这是运输压力测试，不是来源独立的重新选模结果。",
            "",
            "![来源 AUPRC](figures/source_auprc.png)",
            "",
            "![留一来源 AUPRC](figures/leave_one_source_auprc.png)",
            "",
            "| 专家 | 留出来源 | 现有 raw OOF AUPRC | LOSO raw AUPRC | 差值 | LOSO Brier |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for expert in LOSO_EXPERT_ORDER:
        for source, item in loso_metrics["experts"][expert].items():
            lines.append(
                "| {expert} | {source} | {published} | {loso} | {delta} | {brier} |".format(
                    expert=DISPLAY_NAMES[expert],
                    source=source,
                    published=_fmt(item["published_raw_oof"]["auprc"]),
                    loso=_fmt(item["raw_loso"]["auprc"]),
                    delta=_fmt(item["delta_loso_minus_published"]["auprc"]),
                    brier=_fmt(item["raw_loso"]["brier_score"]),
                )
            )
    phys = physiology_uncertainty["variants"]["active_calibrated"]
    lines.extend(
        [
            "",
            "## Physiology 只报告单来源小样本不确定性",
            "",
            f"Physiology 只有 RESILIENT：{physiology_uncertainty['participant_count']} 名可审计参与者范围内，正式可评价样本为 71 行、9 个阳性。现行校准 AUPRC 为 {_fmt(phys['point']['auprc'])}，参与者 bootstrap 95% 区间为 [{_fmt(phys['intervals_95']['auprc']['lower'])}, {_fmt(phys['intervals_95']['auprc']['upper'])}]；AUROC 区间为 [{_fmt(phys['intervals_95']['auroc']['lower'])}, {_fmt(phys['intervals_95']['auroc']['upper'])}]。该区间仅描述当前单来源样本波动，不证明跨设备或跨人群泛化。",
            "",
            "## 范围、数据与方法定义",
            "",
            f"- 评估运行：`{config['run_id']}`；任务：`{config['task_id']}`；数据截至 2026-08-01。",
            f"- Split：`{config['split']['split_id']}`，SHA-256 `{config['split']['sha256']}`。",
            "- 原始概率和现行校准概率直接读取五次正式严格 OOF；无证据行继续排除在概率评价之外。",
            "- 新 Platt/Isotonic 使用 outer_fold 作为五折参与者分组交叉拟合；校准训练折重新计算四级权重、自然行权重或目标先验权重。",
            "- 自然样本指标按原 OOF 行计算；同一人的重复窗口不会在训练权重中放大，但自然部署敏感性仍保留观测行分布。",
            "- 完整阈值表使用每种概率表示的全部唯一观测概率，不用测试折选择阈值。",
            "",
            "## 限制、稳健性与后续建议",
            "",
            "1. 当前没有真实部署人群先验，目标先验 0.05/0.10/0.20/0.30 只用于敏感性；后续不得把其中任一结果直接写成绝对风险。",
            "2. 来源标签率、字段覆盖和代理设备语义不同；留一来源结果应作为融合设计和置信度衰减证据，而不是对某来源做医学优劣判断。",
            "3. LOSO 固定了此前选定的 active 超参数；若未来要形成可发布的来源独立模型比较，必须另立版本化任务并在剩余来源内重新搜索。",
            "4. MODEL-006 可继续按冻结顺序实施；连续分数和五级等级仍只作离线辅助标签，不进入生产专家、PersonalTrend 或融合输入。",
            "5. FUSION-001/002 应保留 raw、现行校准和自然样本候选的可追溯比较，但采用哪一种表示必须由融合严格 OOF 证据和新决定确定，EVAL-001 不提前选择。",
            "",
            "## 进一步问题",
            "",
            "- 真实部署阳性先验和实际误报/漏报代价仍未知；只能在未来真实运营定义和非诊断产品目标明确后版本化选择工作阈值。",
            "- 睡眠仪和 S10 尚无真实联调数据；当前来源迁移结果不能替代设备接入后的工程稳定性与漂移监测。",
        ]
    )
    return "\n".join(lines) + "\n"


def _technical_report_markdown(
    config: Mapping[str, Any],
    metrics: pd.DataFrame,
    source_metrics: pd.DataFrame,
    loso_metrics: Mapping[str, Any],
    physiology_uncertainty: Mapping[str, Any],
) -> str:
    """Build the reader-facing technical report without changing active artifacts."""

    del source_metrics
    lines = [
        "# EVAL-001 Five-expert calibration, threshold, and source-transfer audit",
        "",
        "## Technical summary",
        "",
        "This audit uses the already-published strict participant-level outer-fold "
        "OOF predictions from five experts. Every new calibrator is cross-fitted by "
        "the frozen DATA-007 outer fold, so participants in a test fold never enter "
        "that fold's calibrator. The five active model bundles and prior reports "
        "remain read-only.",
        "",
        "Four-level training weights imply a calibration-fit prevalence of 0.5. "
        "Natural-sample and target-prior sensitivity fits therefore produce different "
        "probability scales. Because no deployment prevalence is available, none of "
        "these values should be interpreted as absolute clinical risk.",
        "",
        "The full threshold relationships and source-specific results show material "
        "trade-offs. EVAL-001 does not select a production threshold, change the "
        "frozen 0.25/0.45/0.65 attention boundaries, or publish a diagnostic model.",
        "",
        "## Raw, active-calibrated, and cross-fitted calibration results",
        "",
        "All rows use natural OOF row frequency. Diagnostic Platt and Isotonic "
        "variants alter calibration only; they do not retrain the base experts.",
        "",
        "| Expert | Probability representation | AUPRC | AUROC | Brier | ECE | 0.5 sensitivity | 0.5 specificity |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    variants = (
        ("raw", "Raw"),
        ("active_calibrated", "Published calibrated"),
        ("cf_platt__natural_sample", "Natural-sample Platt"),
        ("cf_isotonic__natural_sample", "Natural-sample Isotonic"),
    )
    for expert in EXPERT_ORDER:
        for variant, label in variants:
            row = metrics.loc[
                metrics["expert"].eq(expert)
                & metrics["probability_variant"].eq(variant)
            ].iloc[0]
            lines.append(
                "| {expert} | {label} | {auprc} | {auroc} | {brier} | "
                "{ece} | {sens} | {spec} |".format(
                    expert=DISPLAY_NAMES[expert],
                    label=label,
                    auprc=_fmt(row["natural_auprc"]),
                    auroc=_fmt(row["natural_auroc"]),
                    brier=_fmt(row["natural_brier_score"]),
                    ece=_fmt(row["natural_ece"]),
                    sens=_fmt(row["natural_sensitivity"]),
                    spec=_fmt(row["natural_specificity"]),
                )
            )
    lines.extend(
        [
            "",
            "![Calibration Brier comparison](figures/calibration_brier_comparison.png)",
            "",
            "Lower Brier score means closer agreement with natural-sample labels; it "
            "is not, by itself, a production adoption decision.",
            "",
            "## Threshold relationships are diagnostic only",
            "",
            "`threshold_curves.parquet` stores every unique observed threshold with "
            "sensitivity, specificity, precision, recall, positive F1, macro F1, and "
            "full confusion counts. The 0.5 line below is a historical diagnostic "
            "reference, not a selected operating threshold.",
            "",
            "![Threshold trade-off](figures/threshold_tradeoff.png)",
            "",
            "## Source differences and label-source-held-out stress tests",
            "",
            "Activity, Sleep, Joint, and SocialContext are refit in five outer folds "
            "while holding out one label source. Each fit uses labels only from other "
            "sources in that fold's outer-training participants. Preprocessing is fit "
            "inside that partition. Activity ECDF adaptation may use unlabeled held-"
            "out-source outer-training participants, but never test participants or "
            "their labels. Fixed active hyperparameters make this a transport stress "
            "test rather than independent model selection.",
            "",
            "![Source AUPRC](figures/source_auprc.png)",
            "",
            "![Leave-one-source-out AUPRC](figures/leave_one_source_auprc.png)",
            "",
            "| Expert | Held-out source | Published raw OOF AUPRC | LOSO raw AUPRC | Delta | LOSO Brier |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for expert in LOSO_EXPERT_ORDER:
        for source, item in loso_metrics["experts"][expert].items():
            lines.append(
                "| {expert} | {source} | {published} | {loso} | {delta} | "
                "{brier} |".format(
                    expert=DISPLAY_NAMES[expert],
                    source=source,
                    published=_fmt(item["published_raw_oof"]["auprc"]),
                    loso=_fmt(item["raw_loso"]["auprc"]),
                    delta=_fmt(item["delta_loso_minus_published"]["auprc"]),
                    brier=_fmt(item["raw_loso"]["brier_score"]),
                )
            )
    physiology = physiology_uncertainty["variants"]["active_calibrated"]
    lines.extend(
        [
            "",
            "## Physiology: single-source small-sample uncertainty",
            "",
            "Physiology has only RESILIENT and "
            f"{physiology_uncertainty['participant_count']} evaluable participants. "
            f"Published calibrated AUPRC is {_fmt(physiology['point']['auprc'])}; "
            "its participant-bootstrap 95% interval is "
            f"[{_fmt(physiology['intervals_95']['auprc']['lower'])}, "
            f"{_fmt(physiology['intervals_95']['auprc']['upper'])}]. The AUROC "
            "interval is "
            f"[{_fmt(physiology['intervals_95']['auroc']['lower'])}, "
            f"{_fmt(physiology['intervals_95']['auroc']['upper'])}]. These intervals "
            "describe one-source sampling variation, not transportability.",
            "",
            "## Scope, data, and method definitions",
            "",
            f"- Run `{config['run_id']}`; task `{config['task_id']}`; cutoff 2026-08-01.",
            f"- Split `{config['split']['split_id']}`; SHA-256 `{config['split']['sha256']}`.",
            "- Raw and published-calibrated probabilities come from the five formal "
            "strict OOF tables; unavailable expert rows remain excluded.",
            "- New mappings are participant-group cross-fitted by outer fold with "
            "fold-local four-level, natural-row, or target-prior weights.",
            "- Threshold tables cover every unique probability and do not use a test "
            "fold to choose an operating point.",
            "",
            "## Limitations, robustness, and next steps",
            "",
            "1. Target priors 0.05/0.10/0.20/0.30 are sensitivity scenarios only; no "
            "candidate is absolute risk without deployment evidence.",
            "2. Sources differ in prevalence, field coverage, and proxy meaning. LOSO "
            "evidence should inform fusion confidence and masking, not medical ranking.",
            "3. A publishable source-independent comparison needs a separate versioned "
            "task with model selection repeated inside the remaining sources.",
            "4. MODEL-006 may proceed under the frozen sequence. Continuous scores and "
            "five-level grades remain offline auxiliary labels only.",
            "5. FUSION-001/002 should retain traceable calibration alternatives; adoption "
            "requires strict fusion OOF evidence and a new versioned decision.",
            "",
            "## Further questions",
            "",
            "- Deployment prevalence and operational false-alarm/miss costs remain "
            "unknown; a working threshold requires those definitions.",
            "- Real joint device data from the sleep sensor and S10 remains absent; "
            "source-transfer tests do not replace post-integration drift monitoring.",
        ]
    )
    return "\n".join(lines) + "\n"


def _report_artifact_payload(
    config: Mapping[str, Any],
    metrics: pd.DataFrame,
    loso_metrics: Mapping[str, Any],
    physiology_uncertainty: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a bounded native report artifact from reviewed aggregate results."""

    core_variants = (
        "raw",
        "active_calibrated",
        "cf_platt__natural_sample",
        "cf_isotonic__natural_sample",
    )
    calibration = metrics.loc[
        metrics["probability_variant"].isin(core_variants),
        [
            "expert",
            "probability_variant",
            "row_count",
            "positive_row_count",
            "natural_prevalence",
            "natural_auprc",
            "natural_auroc",
            "natural_brier_score",
            "natural_ece",
            "natural_sensitivity",
            "natural_specificity",
        ],
    ].copy()
    calibration["expert_display"] = calibration["expert"].map(DISPLAY_NAMES)
    variant_labels = {
        "raw": "Raw",
        "active_calibrated": "Published calibrated",
        "cf_platt__natural_sample": "Natural Platt",
        "cf_isotonic__natural_sample": "Natural Isotonic",
    }
    calibration["variant_display"] = calibration["probability_variant"].map(
        variant_labels
    )
    current = calibration.loc[
        calibration["probability_variant"].eq("active_calibrated")
    ].copy()
    loso_rows: list[dict[str, Any]] = []
    for expert in LOSO_EXPERT_ORDER:
        for source, item in loso_metrics["experts"][expert].items():
            published = item["published_raw_oof"]
            loso = item["raw_loso"]
            loso_rows.append(
                {
                    "expert": expert,
                    "expert_display": DISPLAY_NAMES[expert],
                    "heldout_source": source,
                    "published_auprc": published["auprc"],
                    "loso_auprc": loso["auprc"],
                    "delta_auprc": item["delta_loso_minus_published"]["auprc"],
                    "published_brier": published["brier_score"],
                    "loso_brier": loso["brier_score"],
                    "test_rows": loso["row_count"],
                    "positive_rows": loso["positive_row_count"],
                }
            )
    physiology = physiology_uncertainty["variants"]["active_calibrated"]
    headline = [
        {
            "expert_count": len(EXPERT_ORDER),
            "probability_variants_per_expert": int(
                metrics.groupby("expert")["probability_variant"].nunique().min()
            ),
            "loso_source_tests": len(loso_rows),
            "protected_active_experts": len(config["baseline_protection"]),
            "physiology_evaluable_participants": physiology_uncertainty[
                "participant_count"
            ],
            "physiology_auprc": physiology["point"]["auprc"],
        }
    ]
    report_path = str(config["reporting"]["report_directory"]).replace("\\", "/")
    metric_path = f"{report_path}/metric_summary.parquet"
    loso_path = f"{report_path}/leave_one_source_metrics.json"
    metric_source = {
        "id": "eval_metrics",
        "label": "EVAL-001 aggregate OOF metrics",
        "path": metric_path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": f"SELECT * FROM read_parquet('{metric_path}')",
            "description": (
                "Reads reviewed aggregate strict-OOF metric rows for all five experts "
                "and fourteen probability representations per expert."
            ),
            "tables_used": [metric_path],
            "filters": [
                "expert_mask = 1 in the source OOF tables",
                "strict DATA-007 participant outer folds",
                "natural-row metrics unless explicitly prefixed fit_weighted",
            ],
            "metric_definitions": [
                "AUPRC is average precision on strict OOF probabilities.",
                "Brier is mean squared probability error on natural OOF rows.",
                "ECE uses ten equal-width probability bins.",
                "Reference-threshold sensitivity and specificity use 0.5 only for diagnostics.",
            ],
        },
    }
    loso_source = {
        "id": "eval_loso",
        "label": "EVAL-001 leave-one-label-source-out metrics",
        "path": loso_path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": f"SELECT * FROM read_json_auto('{loso_path}')",
            "description": (
                "Reads fixed-hyperparameter label-source-held-out transport stress-test "
                "metrics produced within the frozen outer folds."
            ),
            "tables_used": [loso_path],
            "filters": [
                "held-out source labels excluded from every fit",
                "outer-test participants excluded from preprocessing",
                "raw probability only; no LOSO calibrator",
            ],
            "metric_definitions": [
                "Delta AUPRC equals LOSO raw AUPRC minus published raw OOF AUPRC.",
                "LOSO Brier is mean squared raw-probability error on held-out-source outer-test rows.",
            ],
        },
    }
    sources = [metric_source, loso_source]
    title = "V3.3.3 five-expert calibration and source-transfer audit"
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": (
            "Read-only EVAL-001 audit of calibration, thresholds, source variation, "
            "and label-source-held-out transport."
        ),
        "sources": sources,
        "cards": [
            {
                "id": "experts_audited",
                "dataset": "headline",
                "sourceId": "eval_metrics",
                "description": "Active standalone experts covered by the strict OOF audit.",
                "metrics": [
                    {
                        "label": "Experts audited",
                        "field": "expert_count",
                        "format": "number",
                    }
                ],
            },
            {
                "id": "probability_variants",
                "dataset": "headline",
                "sourceId": "eval_metrics",
                "description": "Raw, published, cross-fitted, and target-prior variants per expert.",
                "metrics": [
                    {
                        "label": "Variants per expert",
                        "field": "probability_variants_per_expert",
                        "format": "number",
                    }
                ],
            },
            {
                "id": "loso_tests",
                "dataset": "headline",
                "sourceId": "eval_loso",
                "description": "Fixed-configuration held-out label-source stress tests.",
                "metrics": [
                    {
                        "label": "LOSO source tests",
                        "field": "loso_source_tests",
                        "format": "number",
                    }
                ],
            },
        ],
        "charts": [
            {
                "id": "calibration_brier",
                "title": "Natural-sample OOF Brier score by expert and probability representation",
                "subtitle": "Lower values indicate closer probability-label agreement; no candidate is adopted here.",
                "type": "bar",
                "dataset": "calibration",
                "sourceId": "eval_metrics",
                "encodings": {
                    "x": {
                        "field": "expert_display",
                        "type": "nominal",
                        "label": "Expert",
                    },
                    "y": {
                        "field": "natural_brier_score",
                        "type": "quantitative",
                        "label": "Brier score",
                    },
                    "color": {
                        "field": "variant_display",
                        "type": "nominal",
                        "label": "Probability representation",
                    },
                    "tooltip": [
                        {
                            "field": "natural_auprc",
                            "type": "quantitative",
                            "label": "AUPRC",
                        },
                        {
                            "field": "natural_ece",
                            "type": "quantitative",
                            "label": "ECE",
                        },
                        {"field": "row_count", "type": "quantitative", "label": "Rows"},
                    ],
                },
            },
            {
                "id": "loso_auprc_delta",
                "title": "LOSO minus published raw AUPRC by held-out source",
                "subtitle": "Negative values indicate transport degradation under label-source exclusion.",
                "type": "bar",
                "dataset": "loso",
                "sourceId": "eval_loso",
                "encodings": {
                    "x": {
                        "field": "heldout_source",
                        "type": "nominal",
                        "label": "Held-out source",
                    },
                    "y": {
                        "field": "delta_auprc",
                        "type": "quantitative",
                        "label": "AUPRC delta",
                    },
                    "color": {
                        "field": "expert_display",
                        "type": "nominal",
                        "label": "Expert",
                    },
                    "tooltip": [
                        {
                            "field": "published_auprc",
                            "type": "quantitative",
                            "label": "Published AUPRC",
                        },
                        {
                            "field": "loso_auprc",
                            "type": "quantitative",
                            "label": "LOSO AUPRC",
                        },
                        {
                            "field": "test_rows",
                            "type": "quantitative",
                            "label": "Test rows",
                        },
                    ],
                },
            },
        ],
        "tables": [
            {
                "id": "published_metrics",
                "title": "Published calibrated strict-OOF metrics",
                "dataset": "published",
                "sourceId": "eval_metrics",
                "defaultSort": {"field": "natural_auprc", "direction": "desc"},
                "columns": [
                    {"field": "expert_display", "label": "Expert", "type": "text"},
                    {"field": "row_count", "label": "Rows", "format": "number"},
                    {
                        "field": "positive_row_count",
                        "label": "Positive rows",
                        "format": "number",
                    },
                    {"field": "natural_auprc", "label": "AUPRC", "format": "number"},
                    {"field": "natural_auroc", "label": "AUROC", "format": "number"},
                    {
                        "field": "natural_brier_score",
                        "label": "Brier",
                        "format": "number",
                    },
                    {"field": "natural_ece", "label": "ECE", "format": "number"},
                ],
            },
            {
                "id": "loso_metrics",
                "title": "Leave-one-label-source-out transport metrics",
                "dataset": "loso",
                "sourceId": "eval_loso",
                "defaultSort": {"field": "delta_auprc", "direction": "asc"},
                "columns": [
                    {"field": "expert_display", "label": "Expert", "type": "text"},
                    {
                        "field": "heldout_source",
                        "label": "Held-out source",
                        "type": "text",
                    },
                    {
                        "field": "published_auprc",
                        "label": "Published AUPRC",
                        "format": "number",
                    },
                    {"field": "loso_auprc", "label": "LOSO AUPRC", "format": "number"},
                    {
                        "field": "delta_auprc",
                        "label": "Delta AUPRC",
                        "format": "number",
                    },
                    {"field": "loso_brier", "label": "LOSO Brier", "format": "number"},
                    {"field": "test_rows", "label": "Test rows", "format": "number"},
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {title}"},
            {
                "id": "executive_summary",
                "type": "markdown",
                "body": (
                    "## Executive Summary\n\nFive active experts were audited without changing "
                    "their artifacts. Fourteen probability representations per expert "
                    "and all applicable label-source-held-out comparisons were retained. "
                    "The evidence supports continuing development, but not selecting a "
                    "production probability mapping or threshold."
                ),
                "sourceId": "eval_metrics",
            },
            {
                "id": "headline_metrics",
                "type": "metric-strip",
                "cardIds": ["experts_audited", "probability_variants", "loso_tests"],
            },
            {
                "id": "calibration_section",
                "type": "markdown",
                "body": (
                    "## Calibration findings\n\nCompare Brier, AUPRC, ECE, and sample "
                    "counts together. Calibration improvement does not establish an "
                    "operational threshold or an absolute clinical-risk scale."
                ),
            },
            {
                "id": "calibration_chart",
                "type": "chart",
                "chartId": "calibration_brier",
            },
            {"id": "published_table", "type": "table", "tableId": "published_metrics"},
            {
                "id": "transport_section",
                "type": "markdown",
                "body": (
                    "## Source-transfer findings\n\nLOSO is a fixed-hyperparameter stress "
                    "test. It is not a source-independent model-selection experiment, "
                    "and Physiology is excluded because it has only one source."
                ),
            },
            {"id": "loso_chart", "type": "chart", "chartId": "loso_auprc_delta"},
            {"id": "loso_table", "type": "table", "tableId": "loso_metrics"},
            {
                "id": "methods",
                "type": "markdown",
                "body": (
                    "## Scope and methods\n\nStrict DATA-007 participant outer folds, "
                    "fold-local calibration and preprocessing, natural-row metrics, and "
                    "participant bootstrap uncertainty are used throughout."
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## Limitations and next steps\n\nDeployment prevalence and operational "
                    "false-alarm costs are unknown. MODEL-006 can proceed, while "
                    "FUSION-001/002 must make any later calibration choice under a new "
                    "versioned decision."
                ),
            },
        ],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "calibration": [
                    _json_safe_record(row) for _, row in calibration.iterrows()
                ],
                "published": [_json_safe_record(row) for _, row in current.iterrows()],
                "loso": loso_rows,
            },
        },
        "sources": sources,
        "package_info": {
            "run_id": config["run_id"],
            "task_id": config["task_id"],
            "snapshot_kind": "reviewed_offline_evaluation",
        },
    }


def _physiology_bootstrap_value(
    physiology_uncertainty: Mapping[str, Any],
    variant: str,
    metric: str,
    bound: str,
) -> float:
    return float(
        physiology_uncertainty["variants"][variant]["intervals_95"][metric][bound]
    )


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: metrics[field]
        for field in (
            "row_count",
            "positive_row_count",
            "natural_prevalence",
            "mean_probability",
            "auprc",
            "auroc",
            "brier_score",
            "log_loss",
            "ece",
            "macro_f1",
            "f1_positive",
            "sensitivity",
            "specificity",
            "precision",
            "recall",
        )
    }


def _threshold_row(
    threshold: float,
    *,
    tp: int,
    fp: int,
    total_positive: int,
    total_negative: int,
) -> dict[str, Any]:
    fn = total_positive - tp
    tn = total_negative - fp
    sensitivity = float(tp / total_positive) if total_positive else None
    specificity = float(tn / total_negative) if total_negative else None
    precision = float(tp / (tp + fp)) if tp + fp else None
    f1_positive = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
    negative_f1 = float(2 * tn / (2 * tn + fp + fn)) if 2 * tn + fp + fn else 0.0
    return {
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": sensitivity,
        "f1_positive": f1_positive,
        "macro_f1": float((f1_positive + negative_f1) / 2.0),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "predicted_positive_count": int(tp + fp),
    }


def _validate_published_oof(frame: pd.DataFrame, expert: str) -> None:
    required = {
        "prediction_id",
        "dataset_id",
        "global_participant_id",
        "canonical_row_index",
        "binary_target",
        "outer_fold",
        "raw_probability",
        "calibrated_probability",
        "expert_mask",
    }
    if frame.empty or not required.issubset(frame.columns):
        raise MoodSocialEvaluationError(f"{expert} published OOF is incomplete")
    if frame["prediction_id"].duplicated().any():
        raise MoodSocialEvaluationError(f"{expert} published OOF IDs are not unique")
    if not frame["expert_mask"].eq(1).all():
        raise MoodSocialEvaluationError(f"{expert} available OOF mask is invalid")
    for column in ("raw_probability", "calibrated_probability"):
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype="float64")
        if not np.isfinite(values).all() or not np.all((values >= 0) & (values <= 1)):
            raise MoodSocialEvaluationError(
                f"{expert} published probability is invalid"
            )
    participant_folds = frame.groupby("global_participant_id")["outer_fold"].nunique()
    if not participant_folds.eq(1).all():
        raise MoodSocialEvaluationError(f"{expert} participant OOF leakage detected")


def _prior_token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _probability_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        np.asarray(probability, dtype="float64"),
        PROBABILITY_CLIP_EPSILON,
        1.0 - PROBABILITY_CLIP_EPSILON,
    )
    return np.log(clipped / (1.0 - clipped))


def _safe_difference(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _file_audit(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _assert_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or _sha256_file(path) != expected:
        raise MoodSocialEvaluationError(f"{label} protected file hash drifted")


def _tree_hash_rows(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(
            (item for item in root.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(root).as_posix().encode("utf-8"),
        )
    ]


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(
        f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n" for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _string_set_sha256(values: set[str] | Sequence[str]) -> str:
    ordered = sorted(
        set(str(value) for value in values), key=lambda value: value.encode("utf-8")
    )
    return hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()


def _canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _artifact_manifest(stage: Path) -> dict[str, Any]:
    rows = [row for row in _tree_hash_rows(stage) if row["path"] != "artifacts.json"]
    return {
        "artifact_manifest_version": "mood-social-evaluation-artifacts-v1",
        "artifact_count": len(rows),
        "artifacts": rows,
        "core_sha256": _rows_sha256([row for row in rows if row["path"] != "run.json"]),
    }


def _write_json(path: Path, payload: Any) -> None:
    _write_bytes(
        path,
        (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(
        temporary,
        engine="pyarrow",
        compression="zstd",
        index=False,
        use_dictionary=False,
        row_group_size=4096,
    )
    os.replace(temporary, path)


def _save_figure(fig: Any, path: Path) -> None:
    fig.savefig(
        path,
        dpi=160,
        facecolor="white",
        metadata={"Software": "EVAL-001 matplotlib 3.11.0"},
    )
    plt.close(fig)


def _validate_report_destination(root: Path, destination: Path) -> None:
    reports_root = (root / "reports" / "mental_health" / "mood_social").resolve()
    try:
        destination.relative_to(reports_root)
    except ValueError as exc:
        raise MoodSocialEvaluationError("EVAL-001 output must stay in reports") from exc
    if destination.name in {
        "MH-20260731-001",
        "MH-20260731-002",
        "MH-20260731-003",
        "MH-20260731-004",
        "MH-20260801-005",
    }:
        raise MoodSocialEvaluationError(
            "EVAL-001 cannot target an active expert report"
        )


def _publish_stage(stage: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise MoodSocialEvaluationError("EVAL-001 destination already exists")
        backup = destination.with_name(f".{destination.name}.backup-{os.getpid()}")
        if backup.exists():
            raise MoodSocialEvaluationError("EVAL-001 backup path already exists")
        destination.rename(backup)
        try:
            stage.rename(destination)
        except Exception:
            backup.rename(destination)
            raise
        _remove_stage(backup)
    else:
        stage.rename(destination)


def _remove_stage(path: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    if not (
        resolved.name.startswith(".")
        and (".stage-" in resolved.name or ".backup-" in resolved.name)
    ):
        raise MoodSocialEvaluationError("refusing to remove an unsafe path")
    for item in sorted(
        resolved.rglob("*"), key=lambda value: len(value.parts), reverse=True
    ):
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            item.rmdir()
    resolved.rmdir()


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _json_safe_record(row: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            result[str(key)] = None
        elif isinstance(value, (np.integer, int)):
            result[str(key)] = int(value)
        elif isinstance(value, (np.floating, float)):
            result[str(key)] = float(value)
        else:
            result[str(key)] = value
    return result


def _legacy_fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


__all__ = [
    "EVALUATION_VERSION",
    "LOSO_VERSION",
    "METRICS_VERSION",
    "MoodSocialEvaluationError",
    "binary_metric_summary",
    "calibration_curve_rows",
    "calibration_sample_weights",
    "cross_fit_calibrator",
    "load_evaluation_config",
    "run_expert_evaluation",
    "threshold_curve",
    "validate_baseline_protection",
]
