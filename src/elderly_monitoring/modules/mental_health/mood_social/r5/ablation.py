"""Inner-only causal feature ablations for OPT-V333-R5-002."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import (
    ap_context_metrics,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    participant_equal_weights,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.contract import (
    R5_DEVELOPMENT_SEEDS,
    R5_LABEL_TASKS,
    R5_PROTOCOL_VERSION,
    R5_PSYCHE_RESEARCH_SENSOR_FEATURES,
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.data import (
    load_r5_development_frame,
    r5_inner_fold_series,
    select_r5_label_task,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.features import (
    add_psyche_personal_change_features,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.selection import (
    R5CandidateSpec,
    candidate_inner_oof,
    r5_candidate_space,
)


DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/OPT-V333-R5-002-feature-ablation"
)
SCREENING_OUTER_FOLD = 0
SCREENING_SEED = R5_DEVELOPMENT_SEEDS[0]
SCREENING_CANDIDATE_ID = "lgb_leaf15_min40_l27"


def _write_json(path: Path, value: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite r5 ablation artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _spec() -> R5CandidateSpec:
    return next(
        candidate
        for candidate in r5_candidate_space(SCREENING_SEED)
        if candidate.candidate_id == SCREENING_CANDIDATE_ID
    )


def _metric(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    natural = ap_context_metrics(frame["binary_target"].to_numpy(int), probability)
    participant = ap_context_metrics(
        frame["binary_target"].to_numpy(int),
        probability,
        sample_weight=participant_equal_weights(frame),
    )
    return {
        "auprc": natural["auprc"],
        "normalized_ap": natural["normalized_ap"],
        "auroc": natural["auroc"],
        "brier": natural["brier"],
        "participant_auprc": participant["auprc"],
    }


def _fold_metrics(
    frame: pd.DataFrame, probability: np.ndarray, inner_fold: pd.Series
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for fold in sorted(inner_fold.astype(int).unique()):
        mask = inner_fold.eq(fold).to_numpy(bool)
        result[str(int(fold))] = _metric(frame.loc[mask], probability[mask])
    return result


def _variant_features(
    *,
    batch: int,
    static: tuple[str, ...],
    batch1: tuple[str, ...],
    batch2: tuple[str, ...],
    batch1_retained: bool,
) -> dict[str, tuple[str, ...]]:
    if batch == 1:
        return {
            "static_only": static,
            "longitudinal_batch1_only": batch1,
            "static_plus_longitudinal_batch1": (*static, *batch1),
        }
    base = (*static, *batch1) if batch1_retained else static
    return {
        "retained_baseline_without_batch2": base,
        "interpretable_batch2_only": batch2,
        "retained_baseline_plus_batch2": (*base, *batch2),
    }


def _delta_table(
    metrics: dict[str, dict[str, Any]], *, baseline: str, candidate: str
) -> dict[str, Any]:
    by_task: dict[str, Any] = {}
    for task_id in R5_LABEL_TASKS:
        base = metrics[task_id][baseline]
        new = metrics[task_id][candidate]
        fold_delta = {
            fold: new["by_inner_fold"][fold]["auprc"]
            - base["by_inner_fold"][fold]["auprc"]
            for fold in base["by_inner_fold"]
        }
        by_task[task_id] = {
            "delta_auprc": new["overall"]["auprc"] - base["overall"]["auprc"],
            "delta_participant_auprc": new["overall"]["participant_auprc"]
            - base["overall"]["participant_auprc"],
            "delta_normalized_ap": new["overall"]["normalized_ap"]
            - base["overall"]["normalized_ap"],
            "delta_auroc": new["overall"]["auroc"] - base["overall"]["auroc"],
            "delta_brier": new["overall"]["brier"] - base["overall"]["brier"],
            "nonnegative_inner_folds": int(
                sum(delta >= 0.0 for delta in fold_delta.values())
            ),
            "fold_delta_auprc": fold_delta,
        }
    deltas = [row["delta_auprc"] for row in by_task.values()]
    retained = max(deltas) >= 0.001 and min(deltas) >= -0.002
    return {
        "baseline": baseline,
        "candidate": candidate,
        "by_task": by_task,
        "screening_rule": (
            "at least one head delta AUPRC >= +0.001 and the other >= -0.002"
        ),
        "retained": bool(retained),
    }


def run_feature_ablation(
    *,
    repository_root: Path,
    batch: int,
    report_directory: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one frozen feature batch without reading outer-test labels."""

    if int(batch) not in {1, 2}:
        raise ValueError("r5 feature ablation batch must be 1 or 2")
    root = Path(repository_root).resolve()
    report_root = (
        Path(report_directory)
        if report_directory is not None
        else root / DEFAULT_REPORT_RELATIVE
    )
    if not report_root.is_absolute():
        report_root = root / report_root
    output = report_root / f"batch-{int(batch)}"
    base = load_r5_development_frame(repository_root=root, seed=SCREENING_SEED)
    psyche = base.loc[
        base["dataset_id"].eq("psyche_d")
        & base["outer_fold"].ne(SCREENING_OUTER_FOLD)
    ].copy()
    enriched, batch1_features, batch2_features = add_psyche_personal_change_features(
        psyche, include_batch2=True
    )
    batch1_manifest_path = report_root / "batch-1" / "decision.json"
    if int(batch) == 2:
        if not batch1_manifest_path.is_file():
            raise FileNotFoundError("r5 batch 1 decision is required before batch 2")
        batch1_retained = bool(
            json.loads(batch1_manifest_path.read_text(encoding="utf-8"))["retained"]
        )
    else:
        batch1_retained = False
    variants = _variant_features(
        batch=int(batch),
        static=tuple(R5_PSYCHE_RESEARCH_SENSOR_FEATURES),
        batch1=batch1_features,
        batch2=batch2_features,
        batch1_retained=batch1_retained,
    )
    if any(not names for names in variants.values()):
        raise ValueError("r5 feature ablation cannot evaluate an empty feature batch")
    inner = r5_inner_fold_series(enriched, SCREENING_OUTER_FOLD).astype(int)
    prediction_rows: list[pd.DataFrame] = []
    metrics: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for task_id in R5_LABEL_TASKS:
        task_frame = select_r5_label_task(enriched, task_id)
        metrics[task_id] = {}
        for variant, features in variants.items():
            try:
                probability, selection_metrics = candidate_inner_oof(
                    task_frame,
                    features,
                    inner,
                    _spec(),
                )
            except Exception as error:
                failures.append(
                    {
                        "task_id": task_id,
                        "variant": variant,
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "predictions_opened": False,
                    }
                )
                continue
            part = task_frame[
                [
                    "r5_row_id",
                    "global_participant_id",
                    "nominal_month",
                    "binary_target",
                ]
            ].copy()
            part["task_id"] = task_id
            part["variant"] = variant
            part["inner_fold"] = inner.to_numpy(int)
            part["probability"] = probability
            prediction_rows.append(part)
            metrics[task_id][variant] = {
                "feature_count": len(features),
                "features": list(features),
                "selection": selection_metrics,
                "overall": _metric(task_frame, probability),
                "by_inner_fold": _fold_metrics(task_frame, probability, inner),
            }
    required = len(R5_LABEL_TASKS) * len(variants)
    if len(prediction_rows) != required:
        _write_json(output / "failed_runs.json", failures, overwrite=overwrite)
        raise RuntimeError(
            f"r5 feature ablation completed {len(prediction_rows)}/{required} required runs"
        )
    if int(batch) == 1:
        baseline_name = "static_only"
        candidate_name = "static_plus_longitudinal_batch1"
    else:
        baseline_name = "retained_baseline_without_batch2"
        candidate_name = "retained_baseline_plus_batch2"
    decision = _delta_table(
        metrics, baseline=baseline_name, candidate=candidate_name
    )
    prediction_path = output / "inner_oof.parquet"
    metrics_path = output / "metrics.json"
    decision_path = output / "decision.json"
    manifest_path = output / "artifact_manifest.json"
    if prediction_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite r5 ablation artifact: {prediction_path}")
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(prediction_rows, ignore_index=True).to_parquet(prediction_path, index=False)
    _write_json(metrics_path, metrics, overwrite=overwrite)
    _write_json(decision_path, decision, overwrite=overwrite)
    _write_json(output / "failed_runs.json", failures, overwrite=overwrite)
    manifest = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "status": "pass",
        "evidence_level": "adaptive-development inner-only",
        "seed": SCREENING_SEED,
        "screening_outer_fold_excluded": SCREENING_OUTER_FOLD,
        "outer_test_labels_read": False,
        "confirmation_opened": False,
        "candidate_id": SCREENING_CANDIDATE_ID,
        "batch": int(batch),
        "retained": decision["retained"],
        "artifacts": {
            path.name: {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in (prediction_path, metrics_path, decision_path)
        },
    }
    _write_json(manifest_path, manifest, overwrite=overwrite)
    return manifest


def finalize_feature_ablation(
    *, repository_root: Path, report_directory: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    """Seal the retained R5-002 feature list from both completed decisions."""

    root = Path(repository_root).resolve()
    report_root = (
        Path(report_directory)
        if report_directory is not None
        else root / DEFAULT_REPORT_RELATIVE
    )
    if not report_root.is_absolute():
        report_root = root / report_root
    decision_paths = {
        batch: report_root / f"batch-{batch}" / "decision.json" for batch in (1, 2)
    }
    metric_paths = {
        batch: report_root / f"batch-{batch}" / "metrics.json" for batch in (1, 2)
    }
    if any(not path.is_file() for path in (*decision_paths.values(), *metric_paths.values())):
        raise FileNotFoundError("both r5 feature batches must complete before finalization")
    decisions = {
        batch: json.loads(path.read_text(encoding="utf-8"))
        for batch, path in decision_paths.items()
    }
    if not decisions[1]["retained"]:
        retained_variant = "static_only"
        retention_reason = "batch1_failed_screening_gate"
    elif decisions[2]["retained"]:
        retained_variant = "retained_baseline_plus_batch2"
        retention_reason = "batch1_and_batch2_passed_screening_gate"
    else:
        retained_variant = "static_plus_longitudinal_batch1"
        retention_reason = "batch1_passed_batch2_failed_screening_gate"
    metric_batch = 2 if retained_variant == "retained_baseline_plus_batch2" else 1
    metrics = json.loads(metric_paths[metric_batch].read_text(encoding="utf-8"))
    features = metrics["phq_ge10_current"][retained_variant]["features"]
    if not features or any("phq" in name.lower() for name in features):
        raise ValueError("r5 retained feature list is empty or leaked")
    payload = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "status": "pass",
        "evidence_level": "adaptive-development inner-only",
        "retained_variant": retained_variant,
        "retention_reason": retention_reason,
        "retained_features": features,
        "retained_feature_count": len(features),
        "batch1_retained": bool(decisions[1]["retained"]),
        "batch2_retained": bool(decisions[2]["retained"]),
        "batch2_excluded_from_candidates": not bool(decisions[2]["retained"]),
        "confirmation_opened": False,
        "inputs": {
            f"batch_{batch}_decision_sha256": sha256_file(path)
            for batch, path in decision_paths.items()
        },
    }
    output_path = report_root / "retained_feature_set.json"
    _write_json(output_path, payload, overwrite=overwrite)
    return {**payload, "sha256": sha256_file(output_path)}


__all__ = ["finalize_feature_ablation", "run_feature_ablation"]
