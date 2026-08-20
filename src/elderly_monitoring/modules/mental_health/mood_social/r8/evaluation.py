"""R8 adaptive expert screening and honest integration evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    CandidateSpec,
    fit_model,
    participant_equal_weights,
    predict_model,
)
from elderly_monitoring.modules.mental_health.mood_social.r7.contract import (
    R7_EXISTING_EXPERT_SEEDS,
    repository_root,
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.r7.independent_experts import (
    FEATURES,
    FIXED_SPECS,
    load_domain_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.r7.modeling import (
    binary_metrics,
    project_heads,
)

from .contract import DEFAULT_DATA_RELATIVE, write_json
from .rule_fusion import enumerate_truth_table, rule_contract_sha256, validate_truth_table


R8_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r8/OPT-V333-R8-006-evaluation"
)
R7_REPORT_ROOT = Path("reports/mental_health/mood_social/v3.3.3-r7")
ACTIVITY_BLEND_WEIGHTS = {"ge5": 0.30, "ge10": 0.15}
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260819


def _metric_pair(
    frame: pd.DataFrame, head: str, candidate: np.ndarray, baseline: np.ndarray
) -> dict[str, Any]:
    weights = participant_equal_weights(frame)
    target = frame[f"phq9_{head}_target"].astype(int)
    candidate_metrics = binary_metrics(target, candidate, weight=weights)
    baseline_metrics = binary_metrics(target, baseline, weight=weights)
    fold_delta: dict[str, float] = {}
    for fold, part in frame.groupby("outer_fold", sort=True):
        indices = part.index.to_numpy()
        fold_weights = participant_equal_weights(part)
        fold_delta[str(int(fold))] = (
            binary_metrics(
                part[f"phq9_{head}_target"], candidate[indices], weight=fold_weights
            )["auprc"]
            - binary_metrics(
                part[f"phq9_{head}_target"], baseline[indices], weight=fold_weights
            )["auprc"]
        )
    return {
        "candidate": candidate_metrics,
        "baseline": baseline_metrics,
        "delta": {
            key: candidate_metrics[key] - baseline_metrics[key]
            for key in ("auprc", "normalized_ap", "auroc", "brier", "ece")
        },
        "fold_delta_auprc": fold_delta,
        "nonnegative_folds": sum(value >= 0.0 for value in fold_delta.values()),
    }


def _participant_bootstrap(
    frame: pd.DataFrame,
    *,
    head: str,
    candidate: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, Any]:
    participants = frame["global_participant_id"].astype(str).drop_duplicates().to_numpy()
    index_by_participant = {
        key: group.index.to_numpy()
        for key, group in frame.groupby(frame["global_participant_id"].astype(str), sort=False)
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED + (5 if head == "ge5" else 10))
    deltas: list[float] = []
    target_all = frame[f"phq9_{head}_target"].to_numpy(int)
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        indices = np.concatenate([index_by_participant[key] for key in sampled])
        target = target_all[indices]
        if target.min() == target.max():
            continue
        deltas.append(
            binary_metrics(target, candidate[indices])["auprc"]
            - binary_metrics(target, baseline[indices])["auprc"]
        )
    values = np.asarray(deltas, float)
    return {
        "unit": "participant",
        "seed": BOOTSTRAP_SEED + (5 if head == "ge5" else 10),
        "requested_repetitions": BOOTSTRAP_RESAMPLES,
        "valid_repetitions": int(len(values)),
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _activity_screen(root: Path, report: Path) -> dict[str, Any]:
    source = root / R7_REPORT_ROOT / "OPT-V333-R7-002-independent-experts/activity/three_seed_mean_oof.parquet"
    frame = pd.read_parquet(source).reset_index(drop=True)
    p5 = (
        ACTIVITY_BLEND_WEIGHTS["ge5"] * frame["probability_ge5"].to_numpy(float)
        + (1.0 - ACTIVITY_BLEND_WEIGHTS["ge5"])
        * frame["baseline_probability_ge5"].to_numpy(float)
    )
    p10 = (
        ACTIVITY_BLEND_WEIGHTS["ge10"] * frame["probability_ge10"].to_numpy(float)
        + (1.0 - ACTIVITY_BLEND_WEIGHTS["ge10"])
        * frame["baseline_probability_ge10"].to_numpy(float)
    )
    p5, p10 = project_heads(p5, p10)
    baseline5, baseline10 = project_heads(
        frame["baseline_probability_ge5"], frame["baseline_probability_ge10"]
    )
    result = {
        "status": "adaptive-candidate-not-promoted",
        "selection": "fixed 0.05 grid on already-exposed R7 OOF",
        "candidate_weights": ACTIVITY_BLEND_WEIGHTS,
        "metrics": {
            "ge5": _metric_pair(frame, "ge5", p5, baseline5),
            "ge10": _metric_pair(frame, "ge10", p10, baseline10),
        },
        "bootstrap": {
            "ge5": _participant_bootstrap(
                frame, head="ge5", candidate=p5, baseline=baseline5
            ),
            "ge10": _participant_bootstrap(
                frame, head="ge10", candidate=p10, baseline=baseline10
            ),
        },
        "monotonic_violations": int((p10 > p5).sum()),
        "decision": "retain immutable R7 activity expert in R8 core",
        "decision_reason": (
            "exploratory blend improves aggregate AP but R8 hard no-facial equivalence "
            "requires byte-compatible R7 core and no new confirmation boundary exists"
        ),
    }
    oof = frame.copy()
    oof["r8_probability_ge5"] = p5
    oof["r8_probability_ge10"] = p10
    path = report / "activity_adaptive_blend_oof.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(path, index=False)
    result["oof_sha256"] = sha256_file(path)
    return result


def _r7_metric(root: Path, relative: str) -> dict[str, Any]:
    return json.loads((root / R7_REPORT_ROOT / relative).read_text(encoding="utf-8"))


def _lodo(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for domain in ("activity", "sleep"):
        frame, features = load_domain_frame(root, domain, R7_EXISTING_EXPERT_SEEDS[0])
        domain_result: dict[str, Any] = {}
        for source in sorted(frame["dataset_id"].astype(str).unique()):
            train = frame.loc[frame["dataset_id"].astype(str).ne(source)].copy()
            test = frame.loc[frame["dataset_id"].astype(str).eq(source)].copy()
            source_result: dict[str, Any] = {
                "rows": int(len(test)),
                "participants": int(test["global_participant_id"].nunique()),
            }
            if len(train) == 0 or len(test) == 0:
                source_result["status"] = "not_applicable_empty_train_or_test"
                domain_result[source] = source_result
                continue
            for head in ("ge5", "ge10"):
                target = f"phq9_{head}_target"
                if train[target].nunique() < 2 or test[target].nunique() < 2:
                    source_result[head] = {"status": "not_estimable_single_class"}
                    continue
                train["binary_target"] = train[target].astype(int)
                family, params = (
                    ("elasticnet", {"C": 0.1, "l1_ratio": 0.2})
                    if domain == "activity"
                    else FIXED_SPECS[domain]
                )
                selected = CandidateSpec(
                    f"{domain}_r7_frozen", family, params, R7_EXISTING_EXPERT_SEEDS[0]
                )
                baseline = CandidateSpec(
                    "logistic_c01",
                    "elasticnet",
                    {"C": 0.1, "l1_ratio": 0.2},
                    R7_EXISTING_EXPERT_SEEDS[0],
                )
                selected_model = fit_model(train, features, selected, participant_equal=True)
                baseline_model = fit_model(train, features, baseline, participant_equal=True)
                selected_probability = predict_model(selected_model, test, features)
                baseline_probability = predict_model(baseline_model, test, features)
                selected_metrics = binary_metrics(test[target], selected_probability)
                baseline_metrics = binary_metrics(test[target], baseline_probability)
                source_result[head] = {
                    "status": "success",
                    "candidate": selected_metrics,
                    "baseline": baseline_metrics,
                    "delta_auprc": selected_metrics["auprc"] - baseline_metrics["auprc"],
                    "delta_auroc": selected_metrics["auroc"] - baseline_metrics["auroc"],
                }
            source_result["status"] = "success"
            domain_result[source] = source_result
        result[domain] = domain_result
    result["social"] = {
        "status": "not_applicable_single_dataset",
        "reason": "DepreST-CAT is the only supervised social-contact source",
    }
    result["phq_history"] = {
        "status": "not_applicable_single_longitudinal_source",
        "reason": "only PSYCHE-D has repeated PHQ suitable for current-state history evaluation",
    }
    return result


def run_r8_evaluation(
    *, repository_root_value: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    protocol = root / DEFAULT_DATA_RELATIVE / "r8_protocol_manifest.json"
    if not protocol.is_file():
        raise FileNotFoundError("R8-000 must complete before R8 evaluation")
    report = root / R8_REPORT_RELATIVE
    expected = [
        report / "evaluation_report.json",
        report / "rule_truth_table.json",
        report / "rule_truth_table_audit.json",
        report / "lodo_report.json",
    ]
    if not overwrite and any(path.exists() for path in expected):
        raise FileExistsError("refusing to overwrite frozen R8 evaluation artifacts")

    activity = _activity_screen(root, report)
    sleep = _r7_metric(
        root, "OPT-V333-R7-002-independent-experts/sleep/metrics.json"
    )
    social = _r7_metric(root, "OPT-V333-R7-003-social/confirmation_report.json")
    history = _r7_metric(root, "OPT-V333-R7-004-history/metrics.json")
    r7_evaluation = _r7_metric(root, "OPT-V333-R7-006-evaluation/evaluation_report.json")
    truth = enumerate_truth_table()
    truth_audit = validate_truth_table(truth)
    if truth_audit["status"] != "pass":
        raise ValueError("R8 truth table failed")
    write_json(report / "rule_truth_table.json", truth, overwrite=overwrite)
    write_json(report / "rule_truth_table_audit.json", truth_audit, overwrite=overwrite)
    lodo = _lodo(root)
    write_json(report / "lodo_report.json", lodo, overwrite=overwrite)
    facial = {
        "model_version": "facial-affect-causalnet-b0-opt-me-008-deploy-v1",
        "casme_ii_exposed_development": {
            "uf1": 0.8625816993464053,
            "uar": 0.8616569767441861,
            "accuracy": 0.8671328671328671,
        },
        "smic_frozen_zero_shot_cross_source": {
            "uf1": 0.31342310106716886,
            "uar": 0.3378264307975769,
            "accuracy": 0.3475609756097561,
        },
        "real_device_sessions": 0,
        "paired_phq_video_labels": 0,
        "phq_auprc_reported": False,
        "complete_r8_auprc_gain_claimed": False,
        "role": "engineering-validated bounded support evidence only",
    }
    expert_decisions = {
        "activity": {
            "selected": "R7 logistic unchanged",
            "adaptive_blend_promoted": False,
        },
        "sleep": {"selected": "R7 promoted LightGBM unchanged", "promoted": True},
        "social": {"selected": "anomaly-only", "phq_probability_enabled": False},
        "phq_history": {"selected": "R7 causal history expert unchanged", "promoted": True},
        "profile": {"selected": "background-only"},
        "physiology": {"selected": "support-only"},
    }
    result = {
        "protocol": "mood-social-v3.3.3-r8",
        "status": "pass",
        "tasks_completed": [
            "OPT-V333-R8-001",
            "OPT-V333-R8-002",
            "OPT-V333-R8-003",
            "OPT-V333-R8-004",
            "OPT-V333-R8-005",
            "OPT-V333-R8-006",
        ],
        "expert_decisions": expert_decisions,
        "single_experts": {
            "activity": activity,
            "sleep": sleep,
            "social": social,
            "phq_history": history,
        },
        "facial_affect": facial,
        "rule_contract_sha256": rule_contract_sha256(),
        "rule_truth_table": truth_audit,
        "lodo": lodo,
        "partial_scenario_rule_fusion": r7_evaluation.get(
            "partial_scenario_rule_fusion"
        ),
        "partial_scenario_contains_facial": False,
        "complete_five_domain_auprc_reported": False,
        "reason": (
            "no participant has contemporaneous full R7 modalities, longitudinal PHQ, "
            "and frozen-B0 real-device observations"
        ),
        "release_ceiling": "offline-validated/integration-ready/device-validation-pending",
    }
    write_json(report / "evaluation_report.json", result, overwrite=overwrite)
    artifact_manifest = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(report.iterdir(), key=lambda value: value.name)
        if path.is_file()
    }
    write_json(report / "artifact_manifest.json", artifact_manifest, overwrite=overwrite)
    return result


__all__ = [
    "ACTIVITY_BLEND_WEIGHTS",
    "BOOTSTRAP_RESAMPLES",
    "R8_REPORT_RELATIVE",
    "run_r8_evaluation",
]
