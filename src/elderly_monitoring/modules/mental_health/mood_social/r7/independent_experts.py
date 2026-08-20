"""R7 Activity, Sleep and Profile independent dual-head experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.data import load_r4_development_frame
from elderly_monitoring.modules.mental_health.mood_social.r4.features import (
    DEPLOYABLE_ACTIVITY_FEATURES,
    DEPLOYABLE_PROFILE_FEATURES,
    DEPLOYABLE_SLEEP_FEATURES,
)
from elderly_monitoring.modules.mental_health.mood_social.r7.contract import (
    DEFAULT_DATA_RELATIVE,
    R7_EXISTING_EXPERT_SEEDS,
    repository_root,
)
from elderly_monitoring.modules.mental_health.mood_social.r7.modeling import (
    binary_metrics,
    participant_equal_weights,
    project_heads,
    write_json,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import CandidateSpec, fit_model, predict_model


DEFAULT_REPORT_RELATIVE = Path("reports/mental_health/mood_social/v3.3.3-r7/OPT-V333-R7-002-independent-experts")
FEATURES = {
    "activity": tuple(DEPLOYABLE_ACTIVITY_FEATURES),
    "sleep": tuple(DEPLOYABLE_SLEEP_FEATURES),
    "profile": tuple(DEPLOYABLE_PROFILE_FEATURES),
}
PROMOTED_FAMILY = {"activity": "baseline_logistic", "sleep": "candidate", "profile": "background_baseline"}
FIXED_SPECS = {
    "activity": ("hist_gradient", {"max_leaf_nodes": 15, "l2_regularization": 4.0, "max_iter": 160, "learning_rate": 0.04}),
    "sleep": ("lightgbm", {"n_estimators": 160, "learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 40, "reg_lambda": 6.0, "subsample": 0.9, "colsample_bytree": 0.9}),
    "profile": ("elasticnet", {"C": 0.1, "l1_ratio": 0.2}),
}


def _available(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.Series:
    masks = [f"feature_mask.{name}" for name in features]
    return frame[masks].fillna(0).max(axis=1).gt(0)


def load_domain_frame(root: Path, domain: str, seed: int) -> tuple[pd.DataFrame, tuple[str, ...]]:
    features = FEATURES[domain]
    base = load_r4_development_frame(repository_root=root).copy()
    split = pd.read_parquet(root / DEFAULT_DATA_RELATIVE / "protocol/existing_splits" / f"seed-{seed}.parquet")
    base = base.drop(columns=["outer_fold", "inner_validation_fold_by_outer_fold"], errors="ignore").merge(
        split[["r4_row_id", "outer_fold", "inner_validation_fold_by_outer_fold"]],
        on="r4_row_id", how="left", validate="one_to_one",
    )
    result = base.loc[_available(base, features)].copy()
    result = result.rename(columns={"phq9_ge5_r3_target": "phq9_ge5_target", "phq9_ge10_r3_target": "phq9_ge10_target"})
    if result.empty or result["outer_fold"].isna().any():
        raise ValueError(f"R7 {domain} eligible frame is empty or unsplit")
    return result, features


def _metric_block(frame: pd.DataFrame, head: str) -> dict[str, Any]:
    target = frame[f"phq9_{head}_target"].astype(int)
    candidate = binary_metrics(target, frame[f"probability_{head}"], weight=participant_equal_weights(frame))
    baseline = binary_metrics(target, frame[f"baseline_probability_{head}"], weight=participant_equal_weights(frame))
    return {
        "candidate": candidate,
        "baseline": baseline,
        "delta": {key: candidate[key] - baseline[key] for key in ("auprc", "normalized_ap", "auroc", "brier", "ece")},
        "by_source": {
            str(source): binary_metrics(part[f"phq9_{head}_target"], part[f"probability_{head}"])
            for source, part in frame.groupby("dataset_id", sort=True)
            if part[f"phq9_{head}_target"].nunique() == 2
        },
    }


def run_independent_experts(*, repository_root_value: Path | None = None, overwrite: bool = False) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    report = root / DEFAULT_REPORT_RELATIVE
    all_reports: dict[str, Any] = {}
    for domain in FEATURES:
        seed_predictions: list[pd.DataFrame] = []
        seed_audits: dict[str, Any] = {}
        for seed in R7_EXISTING_EXPERT_SEEDS:
            frame, features = load_domain_frame(root, domain, seed)
            predictions = frame[["r4_row_id", "dataset_id", "global_participant_id", "outer_fold", "phq9_ge5_target", "phq9_ge10_target"]].copy()
            audit: dict[str, Any] = {}
            for head in ("ge5", "ge10"):
                candidate = pd.Series(np.nan, index=frame.index, dtype=float)
                baseline = pd.Series(np.nan, index=frame.index, dtype=float)
                audit[head] = {}
                for outer_fold in range(5):
                    train = frame.loc[frame["outer_fold"].ne(outer_fold)].copy()
                    test = frame.loc[frame["outer_fold"].eq(outer_fold)].copy()
                    train["binary_target"] = train[f"phq9_{head}_target"].astype(int)
                    test["binary_target"] = test[f"phq9_{head}_target"].astype(int)
                    family, params = FIXED_SPECS[domain]
                    selected_spec = CandidateSpec(f"{domain}_fixed", family, params, seed + outer_fold)
                    baseline_spec = CandidateSpec("logistic_c01", "elasticnet", {"C": 0.1, "l1_ratio": 0.2}, seed + outer_fold)
                    selected_model = fit_model(train, features, selected_spec, participant_equal=True)
                    baseline_model = fit_model(train, features, baseline_spec, participant_equal=True)
                    candidate.loc[test.index] = predict_model(selected_model, test, features)
                    baseline.loc[test.index] = predict_model(baseline_model, test, features)
                    audit[head][str(outer_fold)] = {"selected": selected_spec.to_dict(), "baseline": baseline_spec.to_dict(), "selection": "pre_registered_fixed_family_no_outer_label_selection"}
                predictions[f"probability_{head}"] = candidate.loc[frame.index].to_numpy(float)
                predictions[f"baseline_probability_{head}"] = baseline.loc[frame.index].to_numpy(float)
            predictions["probability_ge5"], predictions["probability_ge10"] = project_heads(predictions["probability_ge5"], predictions["probability_ge10"])
            predictions["baseline_probability_ge5"], predictions["baseline_probability_ge10"] = project_heads(predictions["baseline_probability_ge5"], predictions["baseline_probability_ge10"])
            predictions["seed"] = seed
            seed_predictions.append(predictions)
            seed_audits[str(seed)] = audit
            path = report / domain / f"outer_oof_seed-{seed}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_parquet(path, index=False)
        combined = seed_predictions[0].copy()
        for head in ("ge5", "ge10"):
            combined[f"probability_{head}"] = np.mean([part[f"probability_{head}"].to_numpy(float) for part in seed_predictions], axis=0)
            combined[f"baseline_probability_{head}"] = np.mean([part[f"baseline_probability_{head}"].to_numpy(float) for part in seed_predictions], axis=0)
        combined["seed"] = -1
        metrics = {head: _metric_block(combined, head) for head in ("ge5", "ge10")}
        report_entry = {
            "domain": domain,
            "rows": int(len(combined)),
            "participants": int(combined["global_participant_id"].nunique()),
            "features": list(FEATURES[domain]),
            "metrics": metrics,
            "monotonic_violations": int((combined["probability_ge10"] > combined["probability_ge5"]).sum()),
            "evidence": "adaptive-development/reused-benchmark independent expert",
        }
        write_json(report / domain / "selection_audit.json", seed_audits, overwrite=overwrite)
        write_json(report / domain / "metrics.json", report_entry, overwrite=overwrite)
        combined.to_parquet(report / domain / "three_seed_mean_oof.parquet", index=False)
        all_reports[domain] = report_entry
    write_json(report / "summary.json", all_reports, overwrite=overwrite)
    return all_reports


__all__ = ["FEATURES", "load_domain_frame", "run_independent_experts"]
