"""Locked three-seed evaluation, participant bootstrap and moderate gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import (
    ap_context_metrics,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.contract import (
    R4_PROTOCOL_VERSION,
    R4_REPEAT_SEEDS,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    participant_equal_weights,
)


SCREENING_DIRECTORY = Path(
    "reports/mental_health/mood_social/v3.3.3-r4/OPT-V333-R4-003-007-selection"
)
EVALUATION_DIRECTORY = Path(
    "reports/mental_health/mood_social/v3.3.3-r4/OPT-V333-R4-008-evaluation"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def ap_at_prevalence(
    target: Sequence[int],
    probability: Sequence[float],
    *,
    reference_prevalence: float = 0.10,
) -> float:
    target_array = np.asarray(target, dtype=int)
    probability_array = np.asarray(probability, dtype=float)
    prevalence = float(target_array.mean())
    if not 0.0 < prevalence < 1.0:
        raise ValueError("AP reference-prior reweighting requires both classes")
    weight = np.where(
        target_array == 1,
        reference_prevalence / prevalence,
        (1.0 - reference_prevalence) / (1.0 - prevalence),
    )
    return float(average_precision_score(target_array, probability_array, sample_weight=weight))


def _classification_point(
    target: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, float]:
    prediction = probability >= float(threshold)
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    specificity = tn / max(tn + fp, 1)
    return {
        "threshold": float(threshold),
        "sensitivity": float(recall_score(target, prediction, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "alert_rate": float(prediction.mean()),
    }


def operating_points(
    target: Sequence[int], probability: Sequence[float]
) -> dict[str, Any]:
    target_array = np.asarray(target, dtype=int)
    probability_array = np.asarray(probability, dtype=float)
    thresholds = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, 1.0]),
                probability_array,
                np.quantile(probability_array, np.linspace(0.0, 1.0, 501)),
            ]
        )
    )
    points = [_classification_point(target_array, probability_array, threshold) for threshold in thresholds]
    competition = max(points, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
    safety_candidates = [row for row in points if row["sensitivity"] >= 0.80]
    safety = max(
        safety_candidates,
        key=lambda row: (row["precision"], row["specificity"], row["threshold"]),
    )
    sensitivity_at_specificity: dict[str, Any] = {}
    for target_specificity in (0.80, 0.90):
        eligible = [row for row in points if row["specificity"] >= target_specificity]
        sensitivity_at_specificity[f"{target_specificity:.2f}"] = max(
            eligible,
            key=lambda row: (row["sensitivity"], row["precision"], -row["threshold"]),
        )
    return {
        "competition": competition,
        "safety": safety,
        "sensitivity_at_specificity": sensitivity_at_specificity,
    }


def _metric_block(frame: pd.DataFrame) -> dict[str, Any]:
    target = frame["binary_target"].to_numpy(int)
    candidate = frame["candidate_probability"].to_numpy(float)
    baseline = frame["baseline_probability"].to_numpy(float)
    weight = participant_equal_weights(frame)
    candidate_metric = ap_context_metrics(target, candidate)
    baseline_metric = ap_context_metrics(target, baseline)
    candidate_participant = ap_context_metrics(target, candidate, sample_weight=weight)
    baseline_participant = ap_context_metrics(target, baseline, sample_weight=weight)
    no_skill_brier = float(target.mean() * (1.0 - target.mean()))
    return {
        "rows": int(len(frame)),
        "participants": int(frame["global_participant_id"].nunique()),
        "positive_rows": int(target.sum()),
        "coverage": float(np.isfinite(candidate).mean()),
        "candidate": {
            **candidate_metric,
            "ap_at_10pct": ap_at_prevalence(target, candidate),
            "brier_skill": 1.0 - candidate_metric["brier"] / no_skill_brier,
        },
        "baseline": {
            **baseline_metric,
            "ap_at_10pct": ap_at_prevalence(target, baseline),
            "brier_skill": 1.0 - baseline_metric["brier"] / no_skill_brier,
        },
        "delta": {
            key: float(candidate_metric[key] - baseline_metric[key])
            for key in ("auprc", "auroc", "brier", "ece", "normalized_ap")
        },
        "delta_ap_at_10pct": float(
            ap_at_prevalence(target, candidate) - ap_at_prevalence(target, baseline)
        ),
        "participant_equal": {
            "candidate": candidate_participant,
            "baseline": baseline_participant,
            "delta_auprc": float(candidate_participant["auprc"] - baseline_participant["auprc"]),
            "delta_auroc": float(candidate_participant["auroc"] - baseline_participant["auroc"]),
        },
        "operating_points": operating_points(target, candidate),
    }


def _participant_aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    aggregate = (
        frame.groupby("global_participant_id", sort=False)
        .agg(
            binary_target=("binary_target", "max"),
            candidate_probability=("candidate_probability", "mean"),
            baseline_probability=("baseline_probability", "mean"),
            window_count=("r4_row_id", "size"),
        )
        .reset_index()
    )
    result = _metric_block(aggregate)
    result["aggregation_rule"] = "descriptive ever-positive target plus mean window probability"
    result["not_primary_current_state_metric"] = True
    result["maximum_windows"] = int(aggregate["window_count"].max())
    return result


def _macro_source_metric(frame: pd.DataFrame) -> dict[str, Any]:
    source_rows: dict[str, Any] = {}
    for source, part in frame.groupby("dataset_id", sort=True):
        source_rows[str(source)] = _metric_block(part)
    return {
        "reference_prevalence": 0.10,
        "source_count": len(source_rows),
        "macro_candidate_ap_at_10pct": float(
            np.mean([row["candidate"]["ap_at_10pct"] for row in source_rows.values()])
        ),
        "macro_baseline_ap_at_10pct": float(
            np.mean([row["baseline"]["ap_at_10pct"] for row in source_rows.values()])
        ),
        "macro_delta_ap_at_10pct": float(
            np.mean([row["delta_ap_at_10pct"] for row in source_rows.values()])
        ),
        "sources": source_rows,
    }


def _bootstrap(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    source_groups: dict[str, list[np.ndarray]] = {}
    for (source, _participant), indices in frame.groupby(
        ["dataset_id", "global_participant_id"], sort=False
    ).indices.items():
        source_groups.setdefault(str(source), []).append(np.asarray(indices, dtype=int))
    target = frame["binary_target"].to_numpy(int)
    candidate = frame["candidate_probability"].to_numpy(float)
    baseline = frame["baseline_probability"].to_numpy(float)
    values = np.full((replicates, 4), np.nan, dtype=float)
    successful = 0
    for replicate in range(replicates):
        sampled_parts: list[np.ndarray] = []
        for source in sorted(source_groups):
            arrays = source_groups[source]
            choices = rng.integers(0, len(arrays), size=len(arrays))
            sampled_parts.extend(arrays[index] for index in choices)
        indices = np.concatenate(sampled_parts)
        sampled_target = target[indices]
        if np.unique(sampled_target).size < 2:
            continue
        candidate_ap = average_precision_score(sampled_target, candidate[indices])
        baseline_ap = average_precision_score(sampled_target, baseline[indices])
        candidate_auc = roc_auc_score(sampled_target, candidate[indices])
        baseline_auc = roc_auc_score(sampled_target, baseline[indices])
        values[successful] = [candidate_ap, baseline_ap, candidate_ap - baseline_ap, candidate_auc - baseline_auc]
        successful += 1
    values = values[:successful]
    if successful < int(0.99 * replicates):
        raise ValueError("too many invalid participant bootstrap replicates")

    def interval(column: int) -> dict[str, float]:
        return {
            "mean": float(values[:, column].mean()),
            "lower_95": float(np.quantile(values[:, column], 0.025)),
            "upper_95": float(np.quantile(values[:, column], 0.975)),
        }

    return {
        "requested_replicates": replicates,
        "successful_replicates": successful,
        "seed": seed,
        "unit": "global_participant_id",
        "stratification": "dataset_id",
        "participant_count": int(frame["global_participant_id"].nunique()),
        "candidate_auprc": interval(0),
        "baseline_auprc": interval(1),
        "delta_auprc": interval(2),
        "delta_auroc": interval(3),
    }


def _load_seed_frames(root: Path) -> list[tuple[int, pd.DataFrame, Path]]:
    directories = {
        20260812: root / SCREENING_DIRECTORY,
        20260813: root / EVALUATION_DIRECTORY / "seed-20260813",
        20260814: root / EVALUATION_DIRECTORY / "seed-20260814",
    }
    result: list[tuple[int, pd.DataFrame, Path]] = []
    for seed in R4_REPEAT_SEEDS:
        directory = directories[seed]
        path = directory / "screening_outer_oof.parquet"
        result.append((seed, pd.read_parquet(path), path))
    return result


def _combine_seed_frames(seed_frames: list[tuple[int, pd.DataFrame, Path]]) -> pd.DataFrame:
    keys = ["track", "r4_task_id", "r4_row_id"]
    first_seed, first, _ = seed_frames[0]
    stable_columns = [
        "track",
        "r4_task_id",
        "r4_row_id",
        "r3_row_id",
        "dataset_id",
        "global_participant_id",
        "route_pattern",
        "outer_fold",
        "binary_target",
        "feature_signature",
        "baseline_probability",
    ]
    combined = first[stable_columns].copy()
    combined[f"candidate_probability_seed_{first_seed}"] = first["candidate_probability"].to_numpy(float)
    for seed, frame, _ in seed_frames[1:]:
        selected = frame[keys + ["candidate_probability"]].rename(
            columns={"candidate_probability": f"candidate_probability_seed_{seed}"}
        )
        combined = combined.merge(selected, on=keys, how="inner", validate="one_to_one")
    probability_columns = [f"candidate_probability_seed_{seed}" for seed in R4_REPEAT_SEEDS]
    combined["candidate_probability"] = combined[probability_columns].mean(axis=1)
    if len(combined) != len(first):
        raise ValueError("three-seed row keys differ")
    return combined


def run_locked_evaluation(
    *,
    repository_root: Path,
    output_directory: Path | None = None,
    replicates: int = 2000,
    seed: int = 20260812,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_directory) if output_directory else root / EVALUATION_DIRECTORY / "locked"
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    seed_frames = _load_seed_frames(root)
    combined = _combine_seed_frames(seed_frames)
    seed_metrics: dict[str, Any] = {}
    probability_columns = [f"candidate_probability_seed_{value}" for value in R4_REPEAT_SEEDS]
    for seed_value, _seed_frame, _path in seed_frames:
        column = f"candidate_probability_seed_{seed_value}"
        seed_metrics[str(seed_value)] = {}
        for (track, task_id), part in combined.groupby(["track", "r4_task_id"], sort=True):
            work = part.copy()
            work["candidate_probability"] = work[column]
            seed_metrics[str(seed_value)][f"{track}:{task_id}"] = _metric_block(work)

    metrics: dict[str, Any] = {}
    bootstrap: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for (track, task_id), part in combined.groupby(["track", "r4_task_id"], sort=True):
        key = f"{track}:{task_id}"
        overall = _metric_block(part)
        by_source = {
            str(source): _metric_block(source_part)
            for source, source_part in part.groupby("dataset_id", sort=True)
        }
        by_route = {
            str(route): _metric_block(route_part)
            for route, route_part in part.groupby("route_pattern", sort=True)
        }
        by_fold = {
            str(fold): _metric_block(fold_part)
            for fold, fold_part in part.groupby("outer_fold", sort=True)
        }
        metrics[key] = {
            "overall": overall,
            "participant_aggregate_descriptive": _participant_aggregate(part),
            "macro_source": _macro_source_metric(part),
            "by_source": by_source,
            "by_route": by_route,
            "by_outer_fold": by_fold,
        }
        bootstrap[key] = _bootstrap(part.reset_index(drop=True), replicates=replicates, seed=seed)
        seed_deltas = [seed_metrics[str(value)][key]["delta"]["auprc"] for value in R4_REPEAT_SEEDS]
        fold_deltas = [row["delta"]["auprc"] for row in by_fold.values()]
        source_deltas = [row["delta"]["auprc"] for row in by_source.values()]
        gates[key] = {
            "delta_auprc_ge_0_005": overall["delta"]["auprc"] >= 0.005,
            "participant_delta_auprc_ge_minus_0_005": overall["participant_equal"]["delta_auprc"] >= -0.005,
            "delta_auroc_ge_minus_0_01": overall["delta"]["auroc"] >= -0.01,
            "brier_not_worse_by_0_01": overall["delta"]["brier"] <= 0.01,
            "ece_not_worse_by_0_02": overall["delta"]["ece"] <= 0.02,
            "coverage_ge_0_93": overall["coverage"] >= 0.93,
            "at_least_2_of_3_seeds_nonnegative": sum(value >= 0.0 for value in seed_deltas) >= 2,
            "at_least_3_of_5_folds_nonnegative": sum(value >= 0.0 for value in fold_deltas) >= 3,
            "no_major_source_delta_below_minus_0_03": min(source_deltas) >= -0.03,
            "bootstrap_lower_ge_minus_0_01": bootstrap[key]["delta_auprc"]["lower_95"] >= -0.01,
            "seed_delta_auprc": seed_deltas,
            "fold_delta_auprc": fold_deltas,
            "worst_source_delta_auprc": min(source_deltas),
        }
        gates[key]["development_pass"] = all(
            value
            for name, value in gates[key].items()
            if name not in {"seed_delta_auprc", "fold_delta_auprc", "worst_source_delta_auprc", "bootstrap_lower_ge_minus_0_01"}
        )
        gates[key]["offline_competition_pass"] = bool(
            overall["delta"]["auprc"] >= 0.01
            and overall["participant_equal"]["delta_auprc"] >= -0.005
            and overall["delta"]["auroc"] >= -0.01
            and overall["coverage"] >= 0.93
            and (
                sum(value >= 0.0 for value in fold_deltas) >= 3
                or bootstrap[key]["delta_auprc"]["lower_95"] >= -0.01
            )
        )

    # Verify the final projection, not just each seed, obeys the dual-head order.
    monotonic: dict[str, Any] = {}
    for track, part in combined.groupby("track", sort=True):
        early = part.loc[part["r4_task_id"].eq("phq_ge5_current")].sort_values("r4_row_id")
        elevated = part.loc[part["r4_task_id"].eq("phq_ge10_current")].sort_values("r4_row_id")
        violations = elevated["candidate_probability"].to_numpy(float) > early["candidate_probability"].to_numpy(float) + 1.0e-12
        monotonic[track] = {
            "row_count": int(len(early)),
            "violation_count": int(violations.sum()),
            "pass": not bool(violations.any()),
        }

    oof_path = output / "three_seed_mean_outer_oof.parquet"
    metrics_path = output / "locked_metrics.json"
    seed_metrics_path = output / "seed_stability_metrics.json"
    bootstrap_path = output / "participant_bootstrap_2000.json"
    gates_path = output / "promotion_gates.json"
    manifest_path = output / "artifact_manifest.json"
    for path in (oof_path, metrics_path, seed_metrics_path, bootstrap_path, gates_path, manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite locked evaluation artifact: {path}")
    combined.to_parquet(oof_path, index=False)
    _write_json(metrics_path, metrics)
    _write_json(seed_metrics_path, seed_metrics)
    _write_json(bootstrap_path, bootstrap)
    _write_json(gates_path, {"tables": gates, "dual_head_monotonicity": monotonic})
    manifest = {
        "protocol_version": R4_PROTOCOL_VERSION,
        "status": "pass",
        "evidence_level": "adaptive-development/reused-benchmark_not_independent_blind",
        "repeat_seeds": list(R4_REPEAT_SEEDS),
        "bootstrap_replicates": replicates,
        "historical_phq_used": False,
        "artifacts": {
            "oof": {"path": oof_path.relative_to(root).as_posix(), "sha256": _sha256(oof_path)},
            "metrics": {"path": metrics_path.relative_to(root).as_posix(), "sha256": _sha256(metrics_path)},
            "seed_metrics": {"path": seed_metrics_path.relative_to(root).as_posix(), "sha256": _sha256(seed_metrics_path)},
            "bootstrap": {"path": bootstrap_path.relative_to(root).as_posix(), "sha256": _sha256(bootstrap_path)},
            "gates": {"path": gates_path.relative_to(root).as_posix(), "sha256": _sha256(gates_path)},
        },
        "source_oof_sha256": {
            str(seed_value): _sha256(path) for seed_value, _frame, path in seed_frames
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


__all__ = ["ap_at_prevalence", "operating_points", "run_locked_evaluation"]
