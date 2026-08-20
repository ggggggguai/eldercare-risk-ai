"""Sealed confirmation and complete route-aware LODO evaluation for r5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from elderly_monitoring.modules.mental_health.mood_social.r3.baseline import (
    legacy_train_and_predict,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.features import (
    deployable_features_for_signature,
    route_signature,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    participant_equal_weights,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.competition import (
    _load_track_frame,
    _selected_structure,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.confirmation import (
    confirmation_state,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.contract import (
    R5_CONFIRMATION_SEED,
    R5_PROTOCOL_VERSION,
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.fusion import (
    nested_fusion_predictions,
    select_stable_single_specs,
)


DEFAULT_CONFIRMATION_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/OPT-V333-R5-006-confirmation"
)
DEFAULT_BASELINE_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/OPT-V333-R5-000/"
    "r4-paired-baseline"
)
MAJOR_SOURCES = (
    "nhanes",
    "nhanes_ssq_2005_2008",
    "psyche_d",
    "shenzhen_elderly",
)
HEADS = {
    "ge5": "phq9_ge5_r3_target",
    "ge10": "phq9_ge10_r3_target",
}


def _write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite r5 confirmation artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _weighted_ece(
    target: np.ndarray,
    probability: np.ndarray,
    sample_weight: np.ndarray | None = None,
    bins: int = 10,
) -> float:
    weight = np.ones(len(target), dtype=float) if sample_weight is None else np.asarray(sample_weight, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignment = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, bins - 1)
    total = float(weight.sum())
    result = 0.0
    for index in range(bins):
        selected = assignment == index
        if not selected.any():
            continue
        mass = float(weight[selected].sum())
        result += mass / total * abs(
            float(np.average(target[selected], weights=weight[selected]))
            - float(np.average(probability[selected], weights=weight[selected]))
        )
    return float(result)


def binary_metrics(
    target: Iterable[int],
    probability: Iterable[float],
    *,
    sample_weight: Iterable[float] | None = None,
) -> dict[str, float]:
    y = np.asarray(list(target), dtype=int)
    p = np.asarray(list(probability), dtype=float)
    weight = None if sample_weight is None else np.asarray(list(sample_weight), dtype=float)
    if not len(y) or len(y) != len(p) or np.unique(y).size != 2:
        raise ValueError("binary metrics require aligned non-empty two-class arrays")
    prevalence = float(np.average(y, weights=weight))
    ap = float(average_precision_score(y, p, sample_weight=weight))
    return {
        "auprc": ap,
        "prevalence": prevalence,
        "ap_lift": ap / max(prevalence, np.finfo(float).eps),
        "normalized_ap": (ap - prevalence) / max(1.0 - prevalence, np.finfo(float).eps),
        "auroc": float(roc_auc_score(y, p, sample_weight=weight)),
        "brier": float(np.average(np.square(p - y), weights=weight)),
        "ece": _weighted_ece(y, p, weight),
    }


def _metric_delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        key: float(candidate[key] - baseline[key])
        for key in ("auprc", "normalized_ap", "auroc", "brier", "ece")
    }


def _workpoints(frame: pd.DataFrame, head: str) -> dict[str, Any]:
    target = frame["binary_target"].to_numpy(int)
    result: dict[str, Any] = {}
    for point in ("competition", "safety"):
        column = f"{head}_{point}_positive"
        if column not in frame:
            continue
        positive = frame[column].to_numpy(int)
        tp = int(np.sum((target == 1) & (positive == 1)))
        tn = int(np.sum((target == 0) & (positive == 0)))
        fp = int(np.sum((target == 0) & (positive == 1)))
        fn = int(np.sum((target == 1) & (positive == 0)))
        result[point] = {
            "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
            "precision": float(tp / max(tp + fp, 1)),
            "alert_rate": float(positive.mean()),
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        }
    return result


def paired_participant_bootstrap(
    frame: pd.DataFrame,
    *,
    repetitions: int = 2000,
    seed: int = R5_CONFIRMATION_SEED,
) -> dict[str, Any]:
    """Paired participant bootstrap, stratified by source and participant label."""

    required = {
        "global_participant_id",
        "dataset_id",
        "binary_target",
        "candidate_probability",
        "baseline_probability",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"r5 bootstrap frame missing columns: {missing}")
    work = frame.reset_index(drop=True).copy()
    participant = (
        work.groupby(["global_participant_id", "dataset_id"], sort=True)["binary_target"]
        .max()
        .rename("participant_target")
        .reset_index()
    )
    participant["stratum"] = (
        participant["dataset_id"].astype(str)
        + "::"
        + participant["participant_target"].astype(str)
    )
    participant_index = {value: index for index, value in enumerate(participant["global_participant_id"])}
    row_participant = work["global_participant_id"].map(participant_index).to_numpy(int)
    row_count = work.groupby("global_participant_id", sort=False)["global_participant_id"].transform("size").to_numpy(float)
    strata = [
        np.asarray(values, dtype=int)
        for values in participant.groupby("stratum", sort=True).indices.values()
    ]
    y = work["binary_target"].to_numpy(int)
    candidate = work["candidate_probability"].to_numpy(float)
    baseline = work["baseline_probability"].to_numpy(float)
    rng = np.random.default_rng(int(seed))
    natural: dict[str, list[float]] = {"auprc": [], "auroc": []}
    participant_equal: dict[str, list[float]] = {"auprc": [], "auroc": []}
    rankings = {
        name: _prepare_rank_metric(y, probability)
        for name, probability in (("candidate", candidate), ("baseline", baseline))
    }
    for _ in range(int(repetitions)):
        counts = np.zeros(len(participant), dtype=float)
        for positions in strata:
            sampled = rng.choice(positions, size=len(positions), replace=True)
            counts += np.bincount(sampled, minlength=len(participant))
        natural_weight = counts[row_participant]
        equal_weight = natural_weight / row_count
        if natural_weight.sum() == 0 or np.unique(y[natural_weight > 0]).size != 2:
            continue
        for name, weight in (("natural", natural_weight), ("participant_equal", equal_weight)):
            target_store = natural if name == "natural" else participant_equal
            candidate_ap, candidate_auc = _weighted_rank_metrics(rankings["candidate"], weight)
            baseline_ap, baseline_auc = _weighted_rank_metrics(rankings["baseline"], weight)
            target_store["auprc"].append(candidate_ap - baseline_ap)
            target_store["auroc"].append(candidate_auc - baseline_auc)
    def summarize(values: dict[str, list[float]]) -> dict[str, Any]:
        return {
            metric: {
                "mean": float(np.mean(samples)),
                "lower_95": float(np.quantile(samples, 0.025)),
                "upper_95": float(np.quantile(samples, 0.975)),
            }
            for metric, samples in values.items()
        }
    return {
        "unit": "participant",
        "stratification": "dataset_id x participant_max_target",
        "requested_repetitions": int(repetitions),
        "valid_repetitions": len(natural["auprc"]),
        "seed": int(seed),
        "natural_delta": summarize(natural),
        "participant_equal_delta": summarize(participant_equal),
    }


def _prepare_rank_metric(target: np.ndarray, probability: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(np.asarray(probability, float), kind="mergesort")[::-1]
    score = np.asarray(probability, float)[order]
    endpoint = np.r_[np.where(np.diff(score))[0], len(score) - 1].astype(int)
    return {"target": np.asarray(target, int)[order], "order": order, "endpoint": endpoint}


def _weighted_rank_metrics(
    prepared: dict[str, np.ndarray], sample_weight: np.ndarray
) -> tuple[float, float]:
    weight = np.asarray(sample_weight, float)[prepared["order"]]
    target = prepared["target"]
    endpoint = prepared["endpoint"]
    true_positive = np.cumsum(weight * target)[endpoint]
    false_positive = np.cumsum(weight * (1 - target))[endpoint]
    total_positive = float(true_positive[-1])
    total_negative = float(false_positive[-1])
    if total_positive <= 0.0 or total_negative <= 0.0:
        raise ValueError("weighted rank metrics require both weighted classes")
    previous_true = np.r_[0.0, true_positive[:-1]]
    precision = true_positive / np.maximum(true_positive + false_positive, np.finfo(float).eps)
    average_precision = float(np.sum((true_positive - previous_true) / total_positive * precision))
    tpr = np.r_[0.0, true_positive / total_positive]
    fpr = np.r_[0.0, false_positive / total_negative]
    auroc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) * 0.5))
    return average_precision, auroc


def paired_table(
    frame: pd.DataFrame,
    *,
    head: str,
    repetitions: int = 2000,
    bootstrap_seed: int = R5_CONFIRMATION_SEED,
) -> dict[str, Any]:
    finite = np.isfinite(frame["candidate_probability"]) & np.isfinite(frame["baseline_probability"])
    scored = frame.loc[finite].copy()
    if scored.empty or scored["binary_target"].nunique() != 2:
        return {
            "status": "no_common_predictions",
            "rows": int(len(frame)),
            "coverage": float(finite.mean()) if len(frame) else 0.0,
        }
    natural_candidate = binary_metrics(scored["binary_target"], scored["candidate_probability"])
    natural_baseline = binary_metrics(scored["binary_target"], scored["baseline_probability"])
    participant_weight = participant_equal_weights(scored)
    participant_candidate = binary_metrics(
        scored["binary_target"], scored["candidate_probability"], sample_weight=participant_weight
    )
    participant_baseline = binary_metrics(
        scored["binary_target"], scored["baseline_probability"], sample_weight=participant_weight
    )
    by_fold: dict[str, Any] = {}
    for fold, part in scored.groupby("outer_fold", sort=True):
        if part["binary_target"].nunique() != 2:
            continue
        candidate_metric = binary_metrics(part["binary_target"], part["candidate_probability"])
        baseline_metric = binary_metrics(part["binary_target"], part["baseline_probability"])
        by_fold[str(int(fold))] = _metric_delta(candidate_metric, baseline_metric)
    return {
        "status": "pass",
        "rows": int(len(frame)),
        "scored_rows": int(len(scored)),
        "participants": int(frame["global_participant_id"].nunique()),
        "positive_rows": int(frame["binary_target"].sum()),
        "candidate_coverage": float(np.isfinite(frame["candidate_probability"]).mean()),
        "baseline_coverage": float(np.isfinite(frame["baseline_probability"]).mean()),
        "common_coverage": float(finite.mean()),
        "natural": {
            "candidate": natural_candidate,
            "baseline": natural_baseline,
            "delta": _metric_delta(natural_candidate, natural_baseline),
        },
        "participant_equal": {
            "candidate": participant_candidate,
            "baseline": participant_baseline,
            "delta": _metric_delta(participant_candidate, participant_baseline),
        },
        "by_outer_fold_delta": by_fold,
        "nonnegative_auprc_folds": int(
            sum(row["auprc"] >= 0.0 for row in by_fold.values())
        ),
        "candidate_workpoints": _workpoints(scored, head),
        "paired_participant_bootstrap": (
            paired_participant_bootstrap(
                scored, repetitions=repetitions, seed=bootstrap_seed
            )
            if repetitions > 0
            else None
        ),
    }


def _aligned_confirmation_frame(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    head: str,
) -> pd.DataFrame:
    task_id = f"phq_{head}_current"
    target_column = HEADS[head]
    baseline_part = baseline.loc[baseline["r5_task_id"].eq(task_id)].copy()
    selected = [
        "r5_row_id",
        "baseline_probability",
        "binary_target",
        "outer_fold",
    ]
    result = candidate.merge(
        baseline_part[selected], on="r5_row_id", how="left", validate="one_to_one",
        suffixes=("", "_baseline"),
    )
    if result["baseline_probability"].isna().any():
        raise ValueError(f"r5 confirmation baseline alignment is incomplete for {task_id}")
    if not result[target_column].astype(int).eq(result["binary_target"].astype(int)).all():
        raise ValueError(f"r5 confirmation target drift for {task_id}")
    if not result["outer_fold"].astype(int).eq(result["outer_fold_baseline"].astype(int)).all():
        raise ValueError(f"r5 confirmation outer-fold drift for {task_id}")
    result["binary_target"] = result[target_column].astype(int)
    result["candidate_probability"] = result[f"probability_{head}"].astype(float)
    return result


def evaluate_confirmation(
    *,
    repository_root: Path,
    output_directory: Path | None = None,
    repetitions: int = 2000,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    state = confirmation_state(root)
    if not state["opened"]:
        raise PermissionError("r5 confirmation cannot be evaluated before one-time opening")
    output = Path(output_directory) if output_directory else root / DEFAULT_CONFIRMATION_RELATIVE
    if not output.is_absolute():
        output = root / output
    baseline_path = root / DEFAULT_BASELINE_RELATIVE / f"seed-{R5_CONFIRMATION_SEED}" / "r4_recipe_paired_baseline_oof.parquet"
    candidate_root = output / "candidate"
    paths = {
        "multisource_deployable": candidate_root / "multisource_deployable" / f"seed-{R5_CONFIRMATION_SEED}.parquet",
        "psyche_d_single_source_research": candidate_root / "psyche_d_single_source_research" / f"seed-{R5_CONFIRMATION_SEED}.parquet",
    }
    for path in (baseline_path, *paths.values()):
        if not path.is_file():
            raise FileNotFoundError(f"r5 confirmation prediction is missing: {path}")
    baseline = pd.read_parquet(baseline_path)
    candidates = {track: pd.read_parquet(path) for track, path in paths.items()}
    aligned = {
        (track, head): _aligned_confirmation_frame(
            candidate,
            baseline.loc[baseline["track"].eq(track)],
            head=head,
        )
        for track, candidate in candidates.items()
        for head in HEADS
    }
    tables = {
        "multisource_phq_ge10": (aligned[("multisource_deployable", "ge10")], "ge10"),
        "psyche_d_phq_ge10": (aligned[("psyche_d_single_source_research", "ge10")], "ge10"),
        "psyche_d_phq_ge5": (aligned[("psyche_d_single_source_research", "ge5")], "ge5"),
        "joint_101_111_phq_ge10": (
            aligned[("multisource_deployable", "ge10")].loc[
                aligned[("multisource_deployable", "ge10")]["route_pattern"].isin(["101", "111"])
            ],
            "ge10",
        ),
        "multisource_phq_ge5_noninferiority": (aligned[("multisource_deployable", "ge5")], "ge5"),
    }
    table_results = {
        name: paired_table(
            table,
            head=head,
            repetitions=repetitions,
            bootstrap_seed=R5_CONFIRMATION_SEED + index * 101,
        )
        for index, (name, (table, head)) in enumerate(tables.items())
    }
    multisource = {head: aligned[("multisource_deployable", head)] for head in HEADS}
    by_source: dict[str, Any] = {}
    by_route: dict[str, Any] = {}
    for head, frame in multisource.items():
        for source, part in frame.groupby("dataset_id", sort=True):
            by_source[f"{head}:{source}"] = paired_table(
                part, head=head, repetitions=0,
                bootstrap_seed=R5_CONFIRMATION_SEED + int(hashlib.sha256(f"source:{head}:{source}".encode()).hexdigest()[:8], 16),
            )
        for route, part in frame.groupby("route_pattern", sort=True):
            by_route[f"{head}:{route}"] = paired_table(
                part, head=head, repetitions=0,
                bootstrap_seed=R5_CONFIRMATION_SEED + int(hashlib.sha256(f"route:{head}:{route}".encode()).hexdigest()[:8], 16),
            )
    monotonic_violations = {
        track: int(np.sum(frame["probability_ge10"] > frame["probability_ge5"] + 1.0e-12))
        for track, frame in candidates.items()
    }
    primary_names = (
        "multisource_phq_ge10",
        "psyche_d_phq_ge10",
        "psyche_d_phq_ge5",
        "joint_101_111_phq_ge10",
    )
    primary_deltas = [table_results[name]["natural"]["delta"]["auprc"] for name in primary_names]
    other_online = [
        table_results["multisource_phq_ge5_noninferiority"]["natural"]["delta"]["auprc"]
    ]
    stability = all(
        row["nonnegative_auprc_folds"] >= 3
        or row["paired_participant_bootstrap"]["natural_delta"]["auprc"]["lower_95"] >= -0.01
        for row in table_results.values()
    )
    offline_gate = {
        "any_primary_delta_auprc_ge_0_005": bool(max(primary_deltas) >= 0.005),
        "other_online_table_delta_auprc_ge_minus_0_005": bool(min(other_online) >= -0.005),
        "coverage_ge_0_93": bool(
            min(row["candidate_coverage"] for row in table_results.values()) >= 0.93
        ),
        "stability_rule": bool(stability),
        "dual_head_monotonicity": not any(monotonic_violations.values()),
    }
    offline_gate["pass"] = all(offline_gate.values())
    report = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "status": "offline_candidate_pass" if offline_gate["pass"] else "offline_candidate_not_promoted",
        "evidence_level": "sealed reused-cohort confirmation; not a new-subject blind test",
        "confirmation_seed": R5_CONFIRMATION_SEED,
        "bootstrap_repetitions": int(repetitions),
        "tables": table_results,
        "by_source": by_source,
        "by_route": by_route,
        "monotonic_violation_count": monotonic_violations,
        "offline_gate": offline_gate,
    }
    report_path = output / "confirmation_evaluation.json"
    _write_json(report_path, report, overwrite=overwrite)
    manifest = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "status": report["status"],
        "evidence_level": report["evidence_level"],
        "artifacts": {
            "baseline_oof": {"path": baseline_path.relative_to(root).as_posix(), "sha256": sha256_file(baseline_path)},
            "candidate_oof": {
                track: {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
                for track, path in paths.items()
            },
            "evaluation": {"path": report_path.relative_to(root).as_posix(), "sha256": sha256_file(report_path)},
        },
    }
    _write_json(output / "confirmation_artifact_manifest.json", manifest, overwrite=overwrite)
    return report


def compatible_training_mask(frame: pd.DataFrame, signature: str) -> pd.Series:
    route = frame["route_pattern"].astype(str).str.pad(3, side="left", fillchar="0")
    if signature == "profile":
        return route.str[2].eq("1")
    if signature == "sleep":
        return route.str[1].eq("1")
    if signature == "activity":
        return route.str[0].eq("1")
    if signature == "joint":
        return route.eq("111")
    raise ValueError(f"unknown r5 LODO signature: {signature}")


def _lodo_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    common = np.isfinite(frame["candidate_probability"]) & np.isfinite(frame["baseline_probability"])
    if not common.any():
        return {
            "status": "no_common_predictions",
            "rows": int(len(frame)),
            "coverage": 0.0,
        }
    scored = frame.loc[common]
    if scored["binary_target"].nunique() != 2:
        return {"status": "single_class", "rows": int(len(frame)), "coverage": float(common.mean())}
    candidate = binary_metrics(scored["binary_target"], scored["candidate_probability"])
    baseline = binary_metrics(scored["binary_target"], scored["baseline_probability"])
    weight = participant_equal_weights(scored)
    participant_candidate = binary_metrics(scored["binary_target"], scored["candidate_probability"], sample_weight=weight)
    participant_baseline = binary_metrics(scored["binary_target"], scored["baseline_probability"], sample_weight=weight)
    return {
        "status": "pass",
        "rows": int(len(frame)),
        "scored_rows": int(len(scored)),
        "participants": int(frame["global_participant_id"].nunique()),
        "coverage": float(common.mean()),
        "candidate": candidate,
        "baseline": baseline,
        "delta": _metric_delta(candidate, baseline),
        "participant_equal_delta": _metric_delta(participant_candidate, participant_baseline),
    }


def _run_lodo_source(
    frame: pd.DataFrame,
    *,
    source: str,
    seed: int,
    spec: Any,
    structure: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    test = frame.loc[frame["dataset_id"].eq(source)].copy()
    if test.empty:
        raise ValueError(f"r5 LODO held source is absent: {source}")
    signatures = sorted({route_signature(value) for value in test["route_pattern"]})
    if "unsupported" in signatures or len(signatures) != 1:
        raise ValueError(f"r5 LODO held source has ambiguous runtime signature: {source} {signatures}")
    signature = signatures[0]
    train_all = frame.loc[frame["dataset_id"].ne(source)].copy()
    train = train_all.loc[compatible_training_mask(train_all, signature)].copy()
    train["feature_signature"] = signature
    test["feature_signature"] = signature
    audit: dict[str, Any] = {
        "held_out_source": source,
        "feature_signature": signature,
        "compatible_train_rows": int(len(train)),
        "compatible_train_participants": int(train["global_participant_id"].nunique()),
        "train_sources": sorted(train["dataset_id"].astype(str).unique()),
        "test_rows": int(len(test)),
        "participant_overlap_count": int(
            len(set(train["global_participant_id"].astype(str)) & set(test["global_participant_id"].astype(str)))
        ),
        "held_source_labels_used_in_fit_calibration_or_workpoints": False,
    }
    base_columns = [
        "r5_row_id", "global_participant_id", "dataset_id", "route_pattern", "outer_fold",
        "phq9_ge5_r3_target", "phq9_ge10_r3_target",
    ]
    output = test[base_columns].copy()
    try:
        if len(train) < 20:
            raise ValueError("fewer than 20 compatible training rows")
        inner = train["outer_fold"].astype(int)
        candidate, nested_audit = nested_fusion_predictions(
            train, test, None, inner, [spec], "multisource_deployable", structure
        )
        output = output.merge(
            candidate[["r5_row_id", "probability_ge5", "probability_ge10"]],
            on="r5_row_id", how="left", validate="one_to_one",
        )
        audit["candidate_status"] = "pass"
        audit["candidate_nested_audit"] = nested_audit
    except Exception as exc:
        output["probability_ge5"] = np.nan
        output["probability_ge10"] = np.nan
        audit["candidate_status"] = "failed_abstain"
        audit["candidate_failure"] = f"{type(exc).__name__}: {exc}"
    for head, target_column in HEADS.items():
        train_task = train_all.copy()
        train_task["binary_target"] = train_task[target_column].astype(int)
        test_task = test.copy()
        test_task["binary_target"] = test_task[target_column].astype(int)
        try:
            _oof, _raw, probability = legacy_train_and_predict(
                train_task, train_task["outer_fold"].astype(int), test_task
            )
            output[f"baseline_probability_{head}"] = probability
            audit[f"baseline_{head}_status"] = "pass"
        except Exception as exc:
            output[f"baseline_probability_{head}"] = np.nan
            audit[f"baseline_{head}_status"] = "failed_abstain"
            audit[f"baseline_{head}_failure"] = f"{type(exc).__name__}: {exc}"
    head_metrics: dict[str, Any] = {}
    for head, target_column in HEADS.items():
        metric_frame = output.copy()
        metric_frame["binary_target"] = metric_frame[target_column].astype(int)
        metric_frame["candidate_probability"] = metric_frame[f"probability_{head}"]
        metric_frame["baseline_probability"] = metric_frame[f"baseline_probability_{head}"]
        head_metrics[head] = _lodo_metrics(metric_frame)
    audit["heads"] = head_metrics
    audit["route_warning"] = "none"
    for metric in head_metrics.values():
        if (
            metric.get("status") != "pass"
            or metric.get("coverage", 0.0) < 0.93
            or metric.get("delta", {}).get("auprc", 0.0) < -0.03
            or metric.get("delta", {}).get("auroc", 0.0) < -0.05
        ):
            audit["route_warning"] = "block_related_source_route"
    return output, audit


def run_lodo_evaluation(
    *,
    repository_root: Path,
    seed: int = R5_CONFIRMATION_SEED,
    output_directory: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if seed == R5_CONFIRMATION_SEED and not confirmation_state(root)["opened"]:
        raise PermissionError("r5 confirmation LODO cannot run before one-time opening")
    output = Path(output_directory) if output_directory else root / DEFAULT_CONFIRMATION_RELATIVE / "lodo"
    if not output.is_absolute():
        output = root / output
    existing_report = output / "lodo_report.json"
    existing_manifest = output / "artifact_manifest.json"
    if existing_report.is_file() and existing_manifest.is_file() and not overwrite:
        return json.loads(existing_report.read_text(encoding="utf-8"))
    frame, _features = _load_track_frame(root, int(seed), "multisource_deployable")
    spec = select_stable_single_specs(root)["multisource_deployable"]
    structure = _selected_structure(root)
    checkpoints = output / "_checkpoints"
    predictions: list[pd.DataFrame] = []
    audits: dict[str, Any] = {}
    for source in MAJOR_SOURCES:
        prediction_path = checkpoints / f"{source}.parquet"
        audit_path = checkpoints / f"{source}.json"
        if prediction_path.is_file() and audit_path.is_file() and not overwrite:
            prediction = pd.read_parquet(prediction_path)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        else:
            prediction, audit = _run_lodo_source(
                frame, source=source, seed=int(seed), spec=spec, structure=structure
            )
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            prediction.to_parquet(prediction_path, index=False)
            _write_json(audit_path, audit, overwrite=True)
        predictions.append(prediction)
        audits[source] = audit
    combined = pd.concat(predictions, ignore_index=True)
    prediction_path = output / "lodo_predictions.parquet"
    report_path = output / "lodo_report.json"
    combined.to_parquet(prediction_path, index=False)
    warning_count = sum(row["route_warning"] != "none" for row in audits.values())
    report = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "status": "complete_with_route_local_warnings" if warning_count else "pass",
        "evidence_level": (
            "sealed reused-cohort confirmation LODO"
            if seed == R5_CONFIRMATION_SEED
            else "adaptive-development LODO branch preflight"
        ),
        "seed": int(seed),
        "policy": "diagnostic and source-route-local blocking; not a global veto",
        "sources": audits,
        "route_local_warning_count": int(warning_count),
    }
    _write_json(report_path, report, overwrite=overwrite)
    manifest = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "status": report["status"],
        "artifacts": {
            "predictions": {"path": prediction_path.relative_to(root).as_posix(), "sha256": sha256_file(prediction_path)},
            "report": {"path": report_path.relative_to(root).as_posix(), "sha256": sha256_file(report_path)},
        },
    }
    _write_json(output / "artifact_manifest.json", manifest, overwrite=overwrite)
    return report


__all__ = [
    "MAJOR_SOURCES",
    "binary_metrics",
    "compatible_training_mask",
    "evaluate_confirmation",
    "paired_participant_bootstrap",
    "paired_table",
    "run_lodo_evaluation",
]
