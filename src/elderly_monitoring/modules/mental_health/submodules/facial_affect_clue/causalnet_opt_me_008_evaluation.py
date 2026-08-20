from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .causalnet_opt_me_004_evaluation import (
    average_rows_with_metrics,
    leave_one_subject_out_influence,
    metric_bundle,
    paired_subject_bootstrap,
    persistent_zero_subjects,
    seed_mean_std,
    subject_cluster_bootstrap,
)


def metric_deltas(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
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


def _keyed(metrics: Mapping[int | str, Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(seed): value for seed, value in metrics.items()}


def _primary_mean_std(
    metrics: Mapping[int, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for metric in ("uf1", "uar"):
        values = [float(metrics[seed][metric]) for seed in sorted(metrics)]
        output[metric] = {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
        }
    return output


def screening_gate(
    *,
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    baseline_seed_metrics: Mapping[int | str, Mapping[str, Any]],
    candidate_seed_metrics: Mapping[int | str, Mapping[str, Any]],
    hard_identity: bool,
    audits_passed: bool,
) -> dict[str, Any]:
    delta = metric_deltas(baseline_metrics, candidate_metrics)
    base_seed, cand_seed = _keyed(baseline_seed_metrics), _keyed(candidate_seed_metrics)
    if base_seed.keys() != cand_seed.keys() or len(base_seed) != 2:
        raise ValueError("screening gate requires the same two seeds")
    seed_primary = {
        seed: min(
            float(cand_seed[seed]["uf1"]) - float(base_seed[seed]["uf1"]),
            float(cand_seed[seed]["uar"]) - float(base_seed[seed]["uar"]),
        )
        for seed in sorted(base_seed)
    }
    checks = {
        "audits_passed": bool(audits_passed),
        "zero_subject_non_increase": delta["zero_score_subject_count"] <= 0.0,
        "worst_quartile_guard": delta["worst_quartile_subject_score"] >= -0.005,
        "positive_f1_guard": delta["positive_f1"] >= -0.010,
        "positive_recall_guard": delta["positive_recall"] >= -0.010,
        "each_seed_primary_guard": all(value >= -0.005 for value in seed_primary.values()),
        "positive_seed_count": sum(value > 0.0 for value in seed_primary.values()) >= 1,
        "ensemble_primary_improvement": min(delta["uf1"], delta["uar"]) >= 0.005,
        "brier_improvement": delta["brier"] <= -0.002,
        "nll_improvement": delta["nll"] <= -0.003,
        "ece_guard": delta["ece"] <= 0.005,
        "high_confidence_error_guard": delta["high_confidence_error_rate"] <= 0.005,
        "hard_identity": bool(hard_identity),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "delta_vs_B0": delta,
        "seed_min_primary_deltas": seed_primary,
    }


def paired_proper_score_bootstrap(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = 2000,
    seed: int = 20260811,
) -> dict[str, Any]:
    baseline = {str(row["sample_id"]): dict(row) for row in baseline_rows}
    candidate = {str(row["sample_id"]): dict(row) for row in candidate_rows}
    if baseline.keys() != candidate.keys() or not baseline:
        raise ValueError("proper-score bootstrap requires paired sample coverage")
    subjects = sorted({str(row["subject_id"]) for row in baseline.values()})
    by_subject = {
        subject: [sample for sample, row in baseline.items() if str(row["subject_id"]) == subject]
        for subject in subjects
    }
    generator = np.random.default_rng(seed)
    deltas = {"brier": [], "nll": []}
    for _ in range(iterations):
        sampled = generator.choice(subjects, size=len(subjects), replace=True)
        ids = [sample for subject in sampled for sample in by_subject[str(subject)]]
        labels = np.asarray([int(baseline[sample]["label"]) for sample in ids], dtype=np.int64)
        bprob = np.asarray([baseline[sample]["probabilities"] for sample in ids], dtype=np.float64)
        cprob = np.asarray([candidate[sample]["probabilities"] for sample in ids], dtype=np.float64)
        target = np.eye(3, dtype=np.float64)[labels]
        deltas["brier"].append(
            float(np.mean(np.sum((cprob - target) ** 2, axis=1)))
            - float(np.mean(np.sum((bprob - target) ** 2, axis=1)))
        )
        deltas["nll"].append(
            float(-np.mean(np.log(np.clip(cprob[np.arange(len(labels)), labels], 1e-12, 1.0))))
            - float(-np.mean(np.log(np.clip(bprob[np.arange(len(labels)), labels], 1e-12, 1.0))))
        )
    return {
        "iterations": iterations,
        "seed": seed,
        "delta_95_ci": {
            key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            for key, values in deltas.items()
        },
        "delta_mean": {key: float(np.mean(values)) for key, values in deltas.items()},
        "probability_delta_below_zero": {
            key: float(np.mean(np.asarray(values) < 0.0)) for key, values in deltas.items()
        },
    }


def final_gate(
    *,
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    baseline_seed_metrics: Mapping[int | str, Mapping[str, Any]],
    candidate_seed_metrics: Mapping[int | str, Mapping[str, Any]],
    paired_bootstrap: Mapping[str, Any],
    proper_bootstrap: Mapping[str, Any],
    protocol_audit_passed: bool,
    independent_audit_passed: bool,
    hard_identity: bool,
) -> dict[str, Any]:
    delta = metric_deltas(baseline_metrics, candidate_metrics)
    base_seed, cand_seed = _keyed(baseline_seed_metrics), _keyed(candidate_seed_metrics)
    if base_seed.keys() != cand_seed.keys() or len(base_seed) != 3:
        raise ValueError("final gate requires the same three seeds")
    seed_primary = {
        seed: min(
            float(cand_seed[seed]["uf1"]) - float(base_seed[seed]["uf1"]),
            float(cand_seed[seed]["uar"]) - float(base_seed[seed]["uar"]),
        )
        for seed in sorted(base_seed)
    }
    seed_wq = {
        seed: float(cand_seed[seed]["worst_quartile_subject_score"])
        - float(base_seed[seed]["worst_quartile_subject_score"])
        for seed in sorted(base_seed)
    }
    proper_seed_count = sum(
        float(cand_seed[seed]["brier"]) < float(base_seed[seed]["brier"])
        and float(cand_seed[seed]["nll"]) < float(base_seed[seed]["nll"])
        for seed in sorted(base_seed)
    )
    cand_std = _primary_mean_std(cand_seed)
    base_std = _primary_mean_std(base_seed)
    baseline_persistent = set(persistent_zero_subjects(base_seed))
    candidate_persistent = set(persistent_zero_subjects(cand_seed))
    proper_probability = proper_bootstrap["probability_delta_below_zero"]
    checks = {
        "uf1_improvement": delta["uf1"] >= 0.005,
        "uar_improvement": delta["uar"] >= 0.005,
        "paired_ci_lower": float(paired_bootstrap["delta_95_ci"]["selection_score"][0]) >= -0.005,
        "positive_seed_count": sum(value > 0.0 for value in seed_primary.values()) >= 2,
        "no_seed_primary_failure": all(value >= -0.005 for value in seed_primary.values()),
        "uf1_seed_std_absolute": float(cand_std["uf1"]["std"]) <= 0.025,
        "uar_seed_std_absolute": float(cand_std["uar"]["std"]) <= 0.025,
        "uf1_seed_std_vs_B0": float(cand_std["uf1"]["std"]) <= float(base_std["uf1"]["std"]) + 0.003,
        "uar_seed_std_vs_B0": float(cand_std["uar"]["std"]) <= float(base_std["uar"]["std"]) + 0.003,
        "positive_f1_non_decline": delta["positive_f1"] >= 0.0,
        "positive_recall_non_decline": delta["positive_recall"] >= 0.0,
        "ensemble_worst_quartile_guard": delta["worst_quartile_subject_score"] >= -0.005,
        "two_seed_worst_quartile_guard": sum(value >= -0.010 for value in seed_wq.values()) >= 2,
        "zero_subject_non_increase": delta["zero_score_subject_count"] <= 0.0,
        "persistent_zero_non_expansion": candidate_persistent <= baseline_persistent,
        "brier_improvement": delta["brier"] <= -0.003,
        "nll_improvement": delta["nll"] <= -0.005,
        "ece_non_increase": delta["ece"] <= 0.0,
        "high_confidence_error_non_increase": delta["high_confidence_error_rate"] <= 0.0,
        "proper_seed_count": proper_seed_count >= 2,
        "brier_bootstrap_probability": float(proper_probability["brier"]) >= 0.80,
        "nll_bootstrap_probability": float(proper_probability["nll"]) >= 0.80,
        "protocol_audit": bool(protocol_audit_passed),
        "independent_audit": bool(independent_audit_passed),
        "hard_identity": bool(hard_identity),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "delta_vs_B0": delta,
        "seed_min_primary_deltas": seed_primary,
        "seed_worst_quartile_deltas": seed_wq,
        "proper_score_seed_improvement_count": proper_seed_count,
        "candidate_mean_std": cand_std,
        "baseline_mean_std": base_std,
        "baseline_persistent_zero_subjects": sorted(baseline_persistent),
        "candidate_persistent_zero_subjects": sorted(candidate_persistent),
    }


def candidate_rank(candidate_id: str, gate: Mapping[str, Any], metrics: Mapping[str, Any]) -> tuple[Any, ...]:
    delta = gate["delta_vs_B0"]
    return (
        -min(float(delta["uf1"]), float(delta["uar"])),
        float(metrics["brier"]),
        float(metrics["nll"]),
        -float(metrics["worst_quartile_subject_score"]),
        0 if candidate_id == "A1" else 1,
        candidate_id,
    )


__all__ = [
    "average_rows_with_metrics",
    "candidate_rank",
    "final_gate",
    "leave_one_subject_out_influence",
    "metric_bundle",
    "metric_deltas",
    "paired_proper_score_bootstrap",
    "paired_subject_bootstrap",
    "persistent_zero_subjects",
    "screening_gate",
    "seed_mean_std",
    "subject_cluster_bootstrap",
]
