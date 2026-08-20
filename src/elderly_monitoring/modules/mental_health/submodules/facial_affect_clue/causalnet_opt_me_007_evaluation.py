from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .causalnet_opt_me_004_evaluation import (
    leave_one_subject_out_influence,
    metric_bundle,
    paired_subject_bootstrap,
    seed_mean_std,
    subject_cluster_bootstrap,
)
from .causalnet_opt_me_006_evaluation import freeze_gate as me6_freeze_gate


CANDIDATE_COMPLEXITY = {"C1": 0, "C2": 0, "C3": 1}


def hard_predictions_identical(
    reference_rows: Sequence[Mapping[str, Any]], candidate_rows: Sequence[Mapping[str, Any]]
) -> bool:
    reference = {str(row["sample_id"]): int(np.argmax(row["probabilities"])) for row in reference_rows}
    candidate = {str(row["sample_id"]): int(np.argmax(row["probabilities"])) for row in candidate_rows}
    return reference == candidate and len(reference) == len(reference_rows) == len(candidate_rows)


def screen_gate(
    *,
    baseline: Mapping[str, Any],
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    hard_identity: bool,
) -> dict[str, Any]:
    hard_keys = ("uf1", "uar", "accuracy", "worst_quartile_subject_score")
    hard_equal = all(float(candidate[key]) == float(control[key]) for key in hard_keys)
    hard_equal = hard_equal and all(
        float(candidate["per_class"]["positive"][key])
        == float(control["per_class"]["positive"][key])
        for key in ("f1", "recall")
    )
    hard_equal = hard_equal and int(candidate["zero_score_subject_count"]) == int(
        control["zero_score_subject_count"]
    )
    delta_control_ece = float(candidate["ece"]) - float(control["ece"])
    delta_baseline = {
        key: float(candidate[key]) - float(baseline[key])
        for key in ("ece", "brier", "nll", "high_confidence_error_rate")
    }
    checks = {
        "sample_hard_prediction_identity": bool(hard_identity),
        "hard_metric_identity": bool(hard_equal),
        "ece_repair_vs_control": delta_control_ece <= -0.008,
        "ece_guard_vs_baseline": delta_baseline["ece"] <= 0.010,
        "brier_guard_vs_baseline": delta_baseline["brier"] <= 0.005,
        "nll_guard_vs_baseline": delta_baseline["nll"] <= 0.010,
        "high_confidence_error_guard_vs_baseline": delta_baseline[
            "high_confidence_error_rate"
        ]
        <= 0.010,
        "zero_subject_identity": int(candidate["zero_score_subject_count"])
        == int(control["zero_score_subject_count"]),
        "zero_subject_set_identity": set(candidate["zero_score_subjects"])
        == set(control["zero_score_subjects"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "delta_ece_vs_control": delta_control_ece,
        "delta_vs_baseline": delta_baseline,
    }


def candidate_rank(candidate_id: str, metrics: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        float(metrics["ece"]),
        float(metrics["nll"]),
        float(metrics["brier"]),
        float(metrics["high_confidence_error_rate"]),
        CANDIDATE_COMPLEXITY[candidate_id],
        candidate_id,
    )


def ece_improvement_subject_bootstrap(
    control_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    control = {str(row["sample_id"]): dict(row) for row in control_rows}
    candidate = {str(row["sample_id"]): dict(row) for row in candidate_rows}
    if set(control) != set(candidate):
        raise ValueError("ECE bootstrap requires aligned samples")
    subjects = sorted({str(row["subject_id"]) for row in control.values()})
    by_subject = {
        subject: [sample for sample, row in control.items() if str(row["subject_id"]) == subject]
        for subject in subjects
    }
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        control_sample: list[dict[str, Any]] = []
        candidate_sample: list[dict[str, Any]] = []
        for draw_index, subject in enumerate(sampled):
            for sample in by_subject[str(subject)]:
                control_sample.append({**control[sample], "sample_id": f"{draw_index}:{sample}"})
                candidate_sample.append({**candidate[sample], "sample_id": f"{draw_index}:{sample}"})
        deltas.append(
            float(metric_bundle(candidate_sample)["ece"])
            - float(metric_bundle(control_sample)["ece"])
        )
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "iterations": iterations,
        "seed": seed,
        "probability_delta_ece_below_zero": float(np.mean(values < 0.0)),
        "delta_ece_mean": float(values.mean()),
        "delta_ece_95_ci": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
    }


def freeze_gate(
    *,
    baseline_metrics: Mapping[str, Any],
    control_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    baseline_seed_metrics: Mapping[int, Mapping[str, Any]],
    control_seed_metrics: Mapping[int, Mapping[str, Any]],
    candidate_seed_metrics: Mapping[int, Mapping[str, Any]],
    paired_bootstrap: Mapping[str, Any],
    influence: Mapping[str, Any],
    ece_bootstrap: Mapping[str, Any],
    hard_identity: bool,
    leakage_audit_passed: bool,
    protocol_audit_passed: bool,
    independent_audit_passed: bool,
) -> dict[str, Any]:
    inherited = me6_freeze_gate(
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        baseline_seed_metrics=baseline_seed_metrics,
        candidate_seed_metrics=candidate_seed_metrics,
        paired_bootstrap=paired_bootstrap,
        influence=influence,
        protocol_audit_passed=protocol_audit_passed,
        independent_audit_passed=independent_audit_passed,
    )
    shared_seeds = sorted(set(control_seed_metrics) & set(candidate_seed_metrics))
    seed_ece_deltas = {
        seed: float(candidate_seed_metrics[seed]["ece"])
        - float(control_seed_metrics[seed]["ece"])
        for seed in shared_seeds
    }
    repair_checks = {
        "sample_hard_prediction_identity": bool(hard_identity),
        "ensemble_ece_repair_vs_control": float(candidate_metrics["ece"])
        <= float(control_metrics["ece"]) - 0.010,
        "two_of_three_seed_ece_improvement": len(seed_ece_deltas) == 3
        and sum(delta < 0.0 for delta in seed_ece_deltas.values()) >= 2,
        "subject_bootstrap_ece_improvement_probability": float(
            ece_bootstrap["probability_delta_ece_below_zero"]
        )
        >= 0.80,
        "held_subject_leakage_audit": bool(leakage_audit_passed),
    }
    checks = {**inherited["checks"], **repair_checks}
    return {
        **inherited,
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "repair_checks": repair_checks,
        "seed_ece_deltas_vs_control": seed_ece_deltas,
        "ece_improvement_subject_bootstrap": dict(ece_bootstrap),
        "control_ensemble_ece": float(control_metrics["ece"]),
    }


__all__ = [
    "CANDIDATE_COMPLEXITY",
    "candidate_rank",
    "ece_improvement_subject_bootstrap",
    "freeze_gate",
    "hard_predictions_identical",
    "leave_one_subject_out_influence",
    "metric_bundle",
    "paired_subject_bootstrap",
    "screen_gate",
    "seed_mean_std",
    "subject_cluster_bootstrap",
]
