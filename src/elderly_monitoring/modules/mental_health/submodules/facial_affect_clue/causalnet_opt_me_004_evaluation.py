from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .causalnet_opt_me_003_evaluation import (
    apply_calibrator,
    average_probability_rows,
    fit_calibrator,
    paired_subject_bootstrap,
    probability_metric_bundle,
)
from .causalnet_opt_me_004 import CANDIDATE_REGISTRY


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, *, bins: int = 10
) -> float:
    matrix = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if matrix.shape != (len(target), 3) or bins <= 0:
        raise ValueError("ECE expects [n,3] probabilities and positive bin count")
    confidence = matrix.max(axis=1)
    correct = matrix.argmax(axis=1) == target
    total = len(target)
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if not np.any(mask):
            continue
        value += float(np.sum(mask)) / total * abs(
            float(np.mean(correct[mask])) - float(np.mean(confidence[mask]))
        )
    return float(value)


def metric_bundle(
    rows: Sequence[Mapping[str, Any]],
    *,
    ece_bins: int = 10,
    high_confidence_threshold: float = 0.70,
) -> dict[str, Any]:
    base = probability_metric_bundle(rows)
    probabilities = np.asarray([row["probabilities"] for row in rows], dtype=np.float64)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    high_mask = confidence >= high_confidence_threshold
    high_error_count = int(np.sum(high_mask & (predictions != labels)))
    high_error_rate = float(high_error_count / len(rows))
    subject_scores = {
        subject: min(float(values["uf1"]), float(values["uar"]))
        for subject, values in base["per_subject"].items()
    }
    quartile_count = max(1, int(np.ceil(len(subject_scores) * 0.25)))
    ordered = sorted(subject_scores.items(), key=lambda item: (item[1], item[0]))
    worst = ordered[:quartile_count]
    return {
        **base,
        "ece": expected_calibration_error(probabilities, labels, bins=ece_bins),
        "ece_bins": ece_bins,
        "high_confidence_threshold": high_confidence_threshold,
        "high_confidence_error_count": high_error_count,
        "high_confidence_error_rate": high_error_rate,
        "subject_selection_scores": subject_scores,
        "worst_quartile_subjects": [subject for subject, _ in worst],
        "worst_quartile_subject_score": float(np.mean([score for _, score in worst])),
    }


def crossfit_temperature_calibration(
    rows: Sequence[Mapping[str, Any]],
    *,
    ece_bins: int = 10,
    high_confidence_threshold: float = 0.70,
) -> dict[str, Any]:
    folds = sorted({str(row["fold_id"]) for row in rows})
    if len(folds) < 2:
        raise ValueError("cross-fitted calibration requires at least two folds")
    identity_rows = [{**dict(row), "calibration_id": "identity"} for row in rows]
    output: list[dict[str, Any]] = []
    parameters: dict[str, list[float]] = {}
    for fold_id in folds:
        fit_rows = [row for row in rows if str(row["fold_id"]) != fold_id]
        held_rows = [row for row in rows if str(row["fold_id"]) == fold_id]
        fitted = fit_calibrator(
            np.asarray([row["probabilities"] for row in fit_rows], dtype=np.float64),
            np.asarray([int(row["label"]) for row in fit_rows], dtype=np.int64),
            calibration_id="temperature",
        )
        calibrated = apply_calibrator(
            np.asarray([row["probabilities"] for row in held_rows], dtype=np.float64),
            fitted,
        )
        parameters[fold_id] = list(fitted.parameters)
        output.extend(
            {
                **dict(row),
                "probabilities": probability.tolist(),
                "prediction": int(np.argmax(probability)),
                "calibration_id": "temperature",
            }
            for row, probability in zip(held_rows, calibrated, strict=True)
        )
    return {
        "selected_calibration_id": "temperature",
        "selected_rows": output,
        "temperature_parameters_by_fold": parameters,
        "audit_control": {
            "identity": metric_bundle(
                identity_rows,
                ece_bins=ece_bins,
                high_confidence_threshold=high_confidence_threshold,
            ),
            "temperature": metric_bundle(
                output,
                ece_bins=ece_bins,
                high_confidence_threshold=high_confidence_threshold,
            ),
        },
        "vector_scaling_evaluated": False,
    }


def average_rows_with_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    ece_bins: int = 10,
    high_confidence_threshold: float = 0.70,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    averaged = average_probability_rows(rows)
    return averaged, metric_bundle(
        averaged,
        ece_bins=ece_bins,
        high_confidence_threshold=high_confidence_threshold,
    )


def seed_mean_std(seed_metrics: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"ddof": 0}
    for metric in ("uf1", "uar", "accuracy", "selection_score"):
        values = [float(seed_metrics[seed][metric]) for seed in sorted(seed_metrics)]
        output[metric] = {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
        }
    return output


def persistent_zero_subjects(seed_metrics: Mapping[int, Mapping[str, Any]]) -> list[str]:
    sets = [set(map(str, metrics["zero_score_subjects"])) for metrics in seed_metrics.values()]
    return sorted(set.intersection(*sets)) if sets else []


def leave_one_subject_out_influence(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = {str(row["sample_id"]): dict(row) for row in baseline_rows}
    candidate = {str(row["sample_id"]): dict(row) for row in candidate_rows}
    if baseline.keys() != candidate.keys():
        raise ValueError("LOSO influence requires paired sample coverage")
    subjects = sorted({str(row["subject_id"]) for row in baseline.values()})
    full_delta = metric_bundle(list(candidate.values()))["selection_score"] - metric_bundle(list(baseline.values()))["selection_score"]
    deltas: dict[str, float] = {}
    for subject in subjects:
        base_rows = [row for row in baseline.values() if str(row["subject_id"]) != subject]
        cand_rows = [row for row in candidate.values() if str(row["subject_id"]) != subject]
        deltas[subject] = float(
            metric_bundle(cand_rows)["selection_score"]
            - metric_bundle(base_rows)["selection_score"]
        )
    return {
        "full_delta": float(full_delta),
        "deltas_by_excluded_subject": deltas,
        "positive_count": sum(value > 0.0 for value in deltas.values()),
        "subject_count": len(subjects),
        "maximum_absolute_delta_shift": max(abs(value - full_delta) for value in deltas.values()),
    }


def screen_gate(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    delta = {
        "uf1": float(candidate["uf1"]) - float(baseline["uf1"]),
        "uar": float(candidate["uar"]) - float(baseline["uar"]),
        "positive_f1": float(candidate["per_class"]["positive"]["f1"])
        - float(baseline["per_class"]["positive"]["f1"]),
        "positive_recall": float(candidate["per_class"]["positive"]["recall"])
        - float(baseline["per_class"]["positive"]["recall"]),
        "zero_score_subject_count": int(candidate["zero_score_subject_count"])
        - int(baseline["zero_score_subject_count"]),
    }
    checks = {
        "uf1_decline": delta["uf1"] >= -0.005,
        "uar_decline": delta["uar"] >= -0.005,
        "one_primary_improvement": max(delta["uf1"], delta["uar"]) >= 0.01,
        "positive_f1_non_decline": delta["positive_f1"] >= 0.0,
        "positive_recall_non_decline": delta["positive_recall"] >= 0.0,
        "zero_subject_non_increase": delta["zero_score_subject_count"] <= 0,
    }
    return {"passed": all(checks.values()), "checks": checks, "delta": delta}


def freeze_gate(
    *,
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    candidate_seed_metrics: Mapping[int, Mapping[str, Any]],
    persistent_zero: Sequence[str],
    paired_bootstrap: Mapping[str, Any],
    influence: Mapping[str, Any],
    protocol_audit_passed: bool,
    independent_audit_passed: bool,
) -> dict[str, Any]:
    delta = {
        "uf1": float(candidate_metrics["uf1"]) - float(baseline_metrics["uf1"]),
        "uar": float(candidate_metrics["uar"]) - float(baseline_metrics["uar"]),
        "positive_f1": float(candidate_metrics["per_class"]["positive"]["f1"])
        - float(baseline_metrics["per_class"]["positive"]["f1"]),
        "positive_recall": float(candidate_metrics["per_class"]["positive"]["recall"])
        - float(baseline_metrics["per_class"]["positive"]["recall"]),
        "worst_quartile_subject_score": float(candidate_metrics["worst_quartile_subject_score"])
        - float(baseline_metrics["worst_quartile_subject_score"]),
        "brier": float(candidate_metrics["brier"]) - float(baseline_metrics["brier"]),
        "nll": float(candidate_metrics["nll"]) - float(baseline_metrics["nll"]),
        "ece": float(candidate_metrics["ece"]) - float(baseline_metrics["ece"]),
        "high_confidence_error_rate": float(candidate_metrics["high_confidence_error_rate"])
        - float(baseline_metrics["high_confidence_error_rate"]),
    }
    mean_std = seed_mean_std(candidate_seed_metrics)
    paired_lower = float(paired_bootstrap["delta_95_ci"]["selection_score"][0])
    checks = {
        "uf1_improvement": delta["uf1"] >= 0.01,
        "uar_improvement": delta["uar"] >= 0.01,
        "positive_f1_improvement": delta["positive_f1"] >= 0.02,
        "positive_recall_improvement": delta["positive_recall"] >= 0.02,
        "uf1_seed_std": float(mean_std["uf1"]["std"]) <= 0.03,
        "uar_seed_std": float(mean_std["uar"]["std"]) <= 0.03,
        "no_persistent_zero_subject": len(persistent_zero) == 0,
        "paired_ci_lower": paired_lower >= -0.01,
        "worst_quartile_non_decline": delta["worst_quartile_subject_score"] >= -0.005,
        "leave_one_subject_positive_count": int(influence["positive_count"]) >= 20,
        "single_subject_delta_shift": float(influence["maximum_absolute_delta_shift"]) <= 0.01,
        "brier_guard": delta["brier"] <= 0.005,
        "nll_guard": delta["nll"] <= 0.01,
        "ece_guard": delta["ece"] <= 0.01,
        "high_confidence_error_guard": delta["high_confidence_error_rate"] <= 0.01,
        "protocol_audit": bool(protocol_audit_passed),
        "independent_audit": bool(independent_audit_passed),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "delta": delta,
        "candidate_mean_std": mean_std,
        "paired_selection_score_ci_lower": paired_lower,
        "persistent_zero_subjects": list(persistent_zero),
        "leave_one_subject_out": dict(influence),
    }


def candidate_rank(
    candidate_id: str,
    metrics: Mapping[str, Any],
    *,
    paired_ci_width: float = float("inf"),
    seed_std: float = float("inf"),
) -> tuple[Any, ...]:
    return (
        -float(metrics["selection_score"]),
        -float(metrics["per_class"]["positive"]["f1"]),
        -float(metrics["per_class"]["positive"]["recall"]),
        -float(metrics["worst_quartile_subject_score"]),
        float(paired_ci_width),
        float(seed_std),
        CANDIDATE_REGISTRY[candidate_id].complexity_rank,
        candidate_id,
    )


def subject_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, iterations: int = 2000, seed: int = 20260810
) -> dict[str, Any]:
    subjects = sorted({str(row["subject_id"]) for row in rows})
    by_subject = {subject: [dict(row) for row in rows if str(row["subject_id"]) == subject] for subject in subjects}
    generator = np.random.default_rng(seed)
    values = {"uf1": [], "uar": [], "accuracy": []}
    for _ in range(iterations):
        sampled = generator.choice(subjects, size=len(subjects), replace=True)
        sample: list[dict[str, Any]] = []
        for copy_index, subject in enumerate(sampled):
            sample.extend({**row, "sample_id": f"{copy_index}:{row['sample_id']}"} for row in by_subject[str(subject)])
        metrics = metric_bundle(sample)
        for metric in values:
            values[metric].append(float(metrics[metric]))
    return {
        "iterations": iterations,
        "seed": seed,
        "ci_95": {metric: [float(np.quantile(data, 0.025)), float(np.quantile(data, 0.975))] for metric, data in values.items()},
        "mean": {metric: float(np.mean(data)) for metric, data in values.items()},
    }


__all__ = [
    "average_rows_with_metrics",
    "candidate_rank",
    "crossfit_temperature_calibration",
    "expected_calibration_error",
    "freeze_gate",
    "leave_one_subject_out_influence",
    "metric_bundle",
    "paired_subject_bootstrap",
    "persistent_zero_subjects",
    "screen_gate",
    "seed_mean_std",
    "subject_cluster_bootstrap",
]
