"""Dual-label legacy replay and shortcut diagnostics for OPT-V333-R4-001."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from elderly_monitoring.modules.mental_health.mood_social.r3.baseline import (
    _ece,
    replay_outer_fold,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.modeling import (
    WeightSpec,
    sample_weights,
    weighted_binary_metrics,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.contract import (
    R4_PROTOCOL_VERSION,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.data import (
    load_r4_development_frame,
    select_label_task,
)


DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r4/OPT-V333-R4-001-baseline"
)
R3_GE10_OOF_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-002-baseline-replay/"
    "strict_baseline_oof.parquet"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ap_context_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(target) == 0 or len(target) != len(probability):
        raise ValueError("metric arrays must be non-empty and aligned")
    prevalence = float(np.average(target, weights=sample_weight))
    auprc = float(average_precision_score(target, probability, sample_weight=sample_weight))
    auroc = float(roc_auc_score(target, probability, sample_weight=sample_weight))
    brier = float(
        np.average(np.square(probability - target), weights=sample_weight)
        if sample_weight is not None
        else brier_score_loss(target, probability)
    )
    denominator = max(1.0 - prevalence, np.finfo(float).eps)
    return {
        "auprc": auprc,
        "prevalence": prevalence,
        "ap_lift": auprc / max(prevalence, np.finfo(float).eps),
        "normalized_ap": (auprc - prevalence) / denominator,
        "auroc": auroc,
        "brier": brier,
        "ece": _ece(target, probability),
    }


def diagnostic_prior_oof(
    frame: pd.DataFrame,
    *,
    group_column: str,
    smoothing: float = 20.0,
) -> pd.DataFrame:
    """Outer-safe shortcut diagnostic; never a deployable candidate."""

    required = {
        "r4_row_id",
        "global_participant_id",
        "outer_fold",
        "binary_target",
        group_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prior diagnostic missing columns: {missing}")
    parts: list[pd.DataFrame] = []
    for outer_fold in sorted(frame["outer_fold"].astype(int).unique()):
        train = frame.loc[frame["outer_fold"].ne(outer_fold)]
        test = frame.loc[frame["outer_fold"].eq(outer_fold)]
        global_positive = float(train["binary_target"].sum())
        global_rows = float(len(train))
        global_prior = global_positive / global_rows
        grouped = train.groupby(group_column, dropna=False)["binary_target"].agg(["sum", "size"])
        probability: list[float] = []
        fit_rows: list[int] = []
        fit_positive: list[int] = []
        for value in test[group_column]:
            if value in grouped.index:
                positives = int(grouped.loc[value, "sum"])
                rows = int(grouped.loc[value, "size"])
            else:
                positives = 0
                rows = 0
            probability.append((positives + smoothing * global_prior) / (rows + smoothing))
            fit_rows.append(rows)
            fit_positive.append(positives)
        part = test[["r4_row_id", "global_participant_id", "outer_fold", "binary_target"]].copy()
        part["diagnostic"] = f"{group_column}_prior_only"
        part["probability"] = np.asarray(probability, dtype=float)
        part["fit_rows"] = fit_rows
        part["fit_positive_rows"] = fit_positive
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    if len(result) != len(frame) or result["r4_row_id"].duplicated().any():
        raise ValueError("prior diagnostic OOF does not cover every row once")
    return result


def _load_sealed_ge10_oof(root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    path = root / R3_GE10_OOF_RELATIVE
    sealed = pd.read_parquet(path)
    expected = set(frame["r3_row_id"].astype(str))
    if set(sealed["r3_row_id"].astype(str)) != expected:
        raise ValueError("sealed r3 ge10 baseline row keys drifted")
    result = frame[
        [
            "r3_row_id",
            "r4_row_id",
            "dataset_id",
            "global_participant_id",
            "route_pattern",
            "outer_fold",
            "binary_target",
        ]
    ].merge(
        sealed[["r3_row_id", "binary_target", "baseline_raw_probability", "baseline_probability"]],
        on="r3_row_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_sealed"),
    )
    if not result["binary_target"].eq(result.pop("binary_target_sealed")).all():
        raise ValueError("sealed r3 ge10 baseline target drifted")
    result["r4_task_id"] = "phq_ge10_current"
    result["baseline_provenance"] = "sealed_r3_same_graph_same_rows"
    return result


def _replay_ge5(root: Path, frame: pd.DataFrame, checkpoints: Path, overwrite: bool) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    checkpoints.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        path = checkpoints / f"phq_ge5_outer_fold_{fold}.parquet"
        if path.exists() and not overwrite:
            prediction = pd.read_parquet(path)
        else:
            prediction, _audit = replay_outer_fold(frame, fold)
            prediction.to_parquet(path, index=False)
        parts.append(prediction)
    result = pd.concat(parts, ignore_index=True)
    if len(result) != len(frame) or result["r3_row_id"].duplicated().any():
        raise ValueError("r4 ge5 baseline OOF coverage is invalid")
    result = result.merge(
        frame[["r3_row_id", "r4_row_id"]], on="r3_row_id", how="left", validate="one_to_one"
    )
    result["r4_task_id"] = "phq_ge5_current"
    result["baseline_provenance"] = "r4_strict_replay_legacy_recipe"
    return result


def _metrics(frame: pd.DataFrame, probability_column: str) -> dict[str, Any]:
    target = frame["binary_target"].to_numpy(int)
    probability = frame[probability_column].to_numpy(float)
    participant_weight = sample_weights(
        frame, WeightSpec("participant_eval", participant_equal=True)
    )
    return {
        "rows": int(len(frame)),
        "participants": int(frame["global_participant_id"].nunique()),
        "positive_rows": int(target.sum()),
        "natural": ap_context_metrics(target, probability),
        "participant_equal": ap_context_metrics(
            target, probability, sample_weight=participant_weight
        ),
    }


def run_dual_label_baseline(
    *, repository_root: Path, report_directory: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    base = load_r4_development_frame(repository_root=root)
    ge10_frame = select_label_task(base, "phq_ge10_current")
    ge5_frame = select_label_task(base, "phq_ge5_current")
    ge10 = _load_sealed_ge10_oof(root, ge10_frame)
    ge5 = _replay_ge5(root, ge5_frame, output / "_checkpoints", overwrite)
    combined = pd.concat([ge5, ge10], ignore_index=True, sort=False)

    diagnostic_parts: list[pd.DataFrame] = []
    diagnostic_metrics: dict[str, Any] = {}
    for task_id, task_frame in (("phq_ge5_current", ge5_frame), ("phq_ge10_current", ge10_frame)):
        for group_column in ("dataset_id", "route_pattern"):
            prediction = diagnostic_prior_oof(task_frame, group_column=group_column)
            prediction["r4_task_id"] = task_id
            diagnostic_parts.append(prediction)
            diagnostic_metrics[f"{task_id}:{group_column}_prior_only"] = _metrics(
                prediction, "probability"
            )
    diagnostics = pd.concat(diagnostic_parts, ignore_index=True)

    oof_path = output / "strict_dual_label_baseline_oof.parquet"
    diagnostics_path = output / "shortcut_diagnostic_oof.parquet"
    metrics_path = output / "baseline_and_shortcut_metrics.json"
    manifest_path = output / "artifact_manifest.json"
    for path in (oof_path, diagnostics_path, metrics_path, manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite r4 baseline artifact: {path}")
    combined.to_parquet(oof_path, index=False)
    diagnostics.to_parquet(diagnostics_path, index=False)
    metrics = {
        "protocol_version": R4_PROTOCOL_VERSION,
        "evidence_level": "adaptive-development/reused-benchmark",
        "phq_ge5_current": _metrics(ge5, "baseline_probability"),
        "phq_ge10_current": _metrics(ge10, "baseline_probability"),
        "shortcut_diagnostics_not_deployable": diagnostic_metrics,
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol_version": R4_PROTOCOL_VERSION,
        "status": "pass",
        "ge10_reused_sealed_same_graph": True,
        "ge10_source_sha256": _sha256_file(root / R3_GE10_OOF_RELATIVE),
        "artifacts": {
            "oof": {"path": oof_path.relative_to(root).as_posix(), "sha256": _sha256_file(oof_path)},
            "diagnostics": {"path": diagnostics_path.relative_to(root).as_posix(), "sha256": _sha256_file(diagnostics_path)},
            "metrics": {"path": metrics_path.relative_to(root).as_posix(), "sha256": _sha256_file(metrics_path)},
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "ap_context_metrics",
    "diagnostic_prior_oof",
    "run_dual_label_baseline",
]
