from __future__ import annotations

from typing import Any, Mapping

from .causalnet_opt_me_004_evaluation import (
    average_rows_with_metrics,
    crossfit_temperature_calibration,
    expected_calibration_error,
    leave_one_subject_out_influence,
    metric_bundle,
    paired_subject_bootstrap,
    persistent_zero_subjects,
    seed_mean_std,
    subject_cluster_bootstrap,
)
from .causalnet_opt_me_006 import CANDIDATE_REGISTRY


def metric_deltas(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    return {
        "uf1": float(candidate["uf1"]) - float(baseline["uf1"]),
        "uar": float(candidate["uar"]) - float(baseline["uar"]),
        "accuracy": float(candidate["accuracy"]) - float(baseline["accuracy"]),
        "positive_f1": float(candidate["per_class"]["positive"]["f1"])
        - float(baseline["per_class"]["positive"]["f1"]),
        "positive_recall": float(candidate["per_class"]["positive"]["recall"])
        - float(baseline["per_class"]["positive"]["recall"]),
        "worst_quartile_subject_score": float(candidate["worst_quartile_subject_score"])
        - float(baseline["worst_quartile_subject_score"]),
        "zero_score_subject_count": float(candidate["zero_score_subject_count"])
        - float(baseline["zero_score_subject_count"]),
        "brier": float(candidate["brier"]) - float(baseline["brier"]),
        "nll": float(candidate["nll"]) - float(baseline["nll"]),
        "ece": float(candidate["ece"]) - float(baseline["ece"]),
        "high_confidence_error_rate": float(candidate["high_confidence_error_rate"])
        - float(baseline["high_confidence_error_rate"]),
    }


def screen_gate(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    delta = metric_deltas(baseline, candidate)
    primary_min = min(delta["uf1"], delta["uar"])
    basic = {
        "uf1_guard": delta["uf1"] >= -0.005,
        "uar_guard": delta["uar"] >= -0.005,
        "positive_f1_guard": delta["positive_f1"] >= -0.010,
        "zero_subject_guard": delta["zero_score_subject_count"] <= 0,
        "worst_quartile_guard": delta["worst_quartile_subject_score"] >= -0.010,
        "ece_guard": delta["ece"] <= 0.015,
    }
    route_a = primary_min >= 0.015 and delta["positive_recall"] >= -0.035
    route_b = delta["positive_recall"] >= 0.030 and primary_min >= -0.005
    return {
        "passed": all(basic.values()) and (route_a or route_b),
        "basic_checks": basic,
        "route_a_primary_gain_with_recall_guard": route_a,
        "route_b_recall_gain_with_primary_guard": route_b,
        "delta": delta,
        "minimum_primary_delta": primary_min,
    }


def candidate_rank(candidate_id: str, metrics: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -min(float(metrics["uf1"]), float(metrics["uar"])),
        -float(metrics["per_class"]["positive"]["recall"]),
        -float(metrics["worst_quartile_subject_score"]),
        -float(metrics["per_class"]["positive"]["f1"]),
        float(metrics["ece"]),
        CANDIDATE_REGISTRY[candidate_id].complexity_rank,
        candidate_id,
    )


def freeze_gate(
    *,
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    baseline_seed_metrics: Mapping[int, Mapping[str, Any]],
    candidate_seed_metrics: Mapping[int, Mapping[str, Any]],
    paired_bootstrap: Mapping[str, Any],
    influence: Mapping[str, Any],
    protocol_audit_passed: bool,
    independent_audit_passed: bool,
) -> dict[str, Any]:
    delta = metric_deltas(baseline_metrics, candidate_metrics)
    paired_lower = float(paired_bootstrap["delta_95_ci"]["selection_score"][0])
    candidate_variability = seed_mean_std(candidate_seed_metrics)
    baseline_persistent = set(persistent_zero_subjects(baseline_seed_metrics))
    candidate_persistent = set(persistent_zero_subjects(candidate_seed_metrics))
    seed_wq_deltas = {
        int(seed): float(candidate_seed_metrics[seed]["worst_quartile_subject_score"])
        - float(baseline_seed_metrics[seed]["worst_quartile_subject_score"])
        for seed in sorted(set(baseline_seed_metrics) & set(candidate_seed_metrics))
    }
    seed_wq_guard_count = sum(value >= -0.010 for value in seed_wq_deltas.values())
    checks = {
        "one_primary_improvement": max(delta["uf1"], delta["uar"]) >= 0.010,
        "other_primary_guard": min(delta["uf1"], delta["uar"]) >= -0.005,
        "positive_f1_non_decline": delta["positive_f1"] >= 0.0,
        "positive_recall_non_decline": delta["positive_recall"] >= 0.0,
        "paired_ci_lower": paired_lower >= -0.005,
        "uf1_seed_std": float(candidate_variability["uf1"]["std"]) <= 0.025,
        "uar_seed_std": float(candidate_variability["uar"]["std"]) <= 0.025,
        "ensemble_worst_quartile_guard": delta["worst_quartile_subject_score"] >= -0.005,
        "two_of_three_seed_worst_quartile_guard": len(seed_wq_deltas) == 3 and seed_wq_guard_count >= 2,
        "zero_subject_non_increase": int(candidate_metrics["zero_score_subject_count"])
        <= int(baseline_metrics["zero_score_subject_count"]),
        "persistent_zero_set_non_expansion": candidate_persistent <= baseline_persistent,
        "leave_one_subject_positive_count": int(influence["positive_count"]) >= 18,
        "single_subject_delta_shift": float(influence["maximum_absolute_delta_shift"]) <= 0.015,
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
        "candidate_mean_std": candidate_variability,
        "paired_selection_score_ci_lower": paired_lower,
        "baseline_persistent_zero_subjects": sorted(baseline_persistent),
        "candidate_persistent_zero_subjects": sorted(candidate_persistent),
        "seed_worst_quartile_deltas": seed_wq_deltas,
        "seed_worst_quartile_guard_count": seed_wq_guard_count,
        "leave_one_subject_out": dict(influence),
    }


__all__ = [
    "average_rows_with_metrics", "candidate_rank", "crossfit_temperature_calibration",
    "expected_calibration_error", "freeze_gate", "leave_one_subject_out_influence",
    "metric_bundle", "metric_deltas", "paired_subject_bootstrap",
    "persistent_zero_subjects", "screen_gate", "seed_mean_std", "subject_cluster_bootstrap",
]
