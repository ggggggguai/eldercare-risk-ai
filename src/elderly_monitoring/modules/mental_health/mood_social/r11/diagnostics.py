"""Post-lock diagnostic ablations and engineering stress tables for R11.

These diagnostics cannot select or promote a model and never rewrite sealed OOF.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.selection import predict_model
from elderly_monitoring.modules.mental_health.mood_social.r7.modeling import (
    binary_metrics,
    select_specificity_threshold,
)
from elderly_monitoring.modules.mental_health.mood_social.r9.contract import (
    R9_HISTORY_FEATURES,
    SLEEP_FEATURES,
)
from elderly_monitoring.modules.mental_health.mood_social.r9.modeling import inner_folds_for_outer
from elderly_monitoring.modules.mental_health.mood_social.r10.modeling import (
    _elastic,
    apply_calibration,
    fit_final_calibration,
    participant_weight,
)

from .contract import DEFAULT_REPORT_RELATIVE, repository_root, sha256_file, write_json
from .evaluation import EVALUATION_RELATIVE
from .reliability_gate import evaluate_reliability
from .transition_modeling import TARGETS, load_sleep_history


DIAGNOSTIC_RELATIVE = DEFAULT_REPORT_RELATIVE / "diagnostics"


def _oof_elastic(
    train: pd.DataFrame,
    folds: pd.Series,
    features: Sequence[str],
    target: str,
    seed: int,
) -> np.ndarray:
    result = pd.Series(np.nan, index=train.index, dtype=float)
    for fold in sorted(folds.unique()):
        fit = train.loc[folds.ne(fold)]
        validation = train.loc[folds.eq(fold)]
        model = _elastic(fit, features, target, seed + int(fold))
        result.loc[validation.index] = predict_model(model, validation, features)
    if result.isna().any():
        raise ValueError("R11 diagnostic OOF incomplete")
    return result.loc[train.index].to_numpy(float)


def _metric(frame: pd.DataFrame, target: str, probability: str) -> dict[str, float]:
    return binary_metrics(
        frame[target].astype(int),
        frame[probability].astype(float),
        weight=participant_weight(frame),
    )


def _ablation_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    definitions = {
        "history_only": tuple(R9_HISTORY_FEATURES),
        "sleep_only": tuple(SLEEP_FEATURES),
    }
    for repeat, repeat_frame in frame.groupby("repeat", sort=True):
        for outer_fold in range(5):
            train = repeat_frame.loc[repeat_frame["outer_fold"].ne(outer_fold)].copy()
            test = repeat_frame.loc[repeat_frame["outer_fold"].eq(outer_fold)].copy()
            inner = inner_folds_for_outer(repeat_frame, outer_fold).loc[train.index].astype(int)
            current = test[
                [
                    "r4_row_id",
                    "global_participant_id",
                    "repeat",
                    "outer_fold",
                    "phq9_ge5_target",
                    "phq9_ge10_target",
                    "history.age_days",
                ]
            ].copy()
            for head, target in TARGETS.items():
                for index, (name, features) in enumerate(definitions.items()):
                    seed = 20261200 + int(repeat) * 1000 + outer_fold * 100 + index * 10
                    raw = _oof_elastic(train, inner, features, target, seed)
                    calibrator = fit_final_calibration(train, raw, target, "platt")
                    model = _elastic(train, features, target, seed + 50)
                    probability = predict_model(model, test, features)
                    current[f"{name}_probability_{head}"] = apply_calibration(
                        calibrator, probability, "platt"
                    )
            outputs.append(current)
    return pd.concat(outputs, ignore_index=True)


def _ablation_report(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"role": "diagnostic-only/not-eligible-for-promotion", "heads": {}}
    for head, target in TARGETS.items():
        result["heads"][head] = {
            name: _metric(frame, target, f"{name}_probability_{head}")
            for name in ("history_only", "sleep_only")
        }
    return result


def _sealed_diagnostics(sealed: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    freshness: dict[str, Any] = {}
    for lower, upper in ((76, 85), (86, 92), (93, 100)):
        part = sealed.loc[sealed["history.age_days"].between(lower, upper)].copy()
        entry: dict[str, Any] = {
            "rows": int(len(part)),
            "participants": int(part["global_participant_id"].nunique()),
            "heads": {},
        }
        for head, target in TARGETS.items():
            if part[target].nunique() == 2:
                candidate = _metric(part, target, f"candidate_probability_{head}")
                baseline = _metric(part, target, f"baseline_probability_{head}")
                entry["heads"][head] = {
                    "candidate": candidate,
                    "baseline": baseline,
                    "delta_auprc": candidate["auprc"] - baseline["auprc"],
                }
        freshness[f"{lower}_{upper}"] = entry
    workpoints: dict[str, Any] = {"definition": "fixed probability thresholds; descriptive only", "heads": {}}
    for head, target in TARGETS.items():
        block: dict[str, Any] = {}
        for threshold in (0.5,):
            y = sealed[target].to_numpy(int)
            p = sealed[f"candidate_probability_{head}"].to_numpy(float)
            predicted = p >= threshold
            positive = y == 1
            negative = ~positive
            block[str(threshold)] = {
                "sensitivity": float(predicted[positive].mean()),
                "specificity": float((~predicted[negative]).mean()),
                "alert_rate": float(predicted.mean()),
            }
        # A full-OOF threshold is diagnostic only and explicitly never packaged.
        for specificity in (0.80, 0.90):
            threshold = select_specificity_threshold(
                sealed[target].astype(int),
                sealed[f"candidate_probability_{head}"].astype(float),
                specificity,
            )
            block[f"descriptive_specificity_{specificity:.2f}"] = {
                "threshold": float(threshold),
                "packaged": False,
                "reason": "uses complete sealed OOF labels; report-only",
            }
        workpoints["heads"][head] = block
    return freshness, workpoints


def _stress_report() -> dict[str, Any]:
    cases = {
        "usable": evaluate_reliability(
            "activity", {"available": True, "domain_level": 2, "confidence": 0.9, "valid_days": 14}
        ),
        "degraded_quality": evaluate_reliability(
            "activity", {"available": True, "domain_level": 3, "confidence": 0.5, "valid_days": 6}
        ),
        "low_coverage": evaluate_reliability(
            "activity",
            {"available": True, "domain_level": 3, "confidence": 0.9, "valid_days": 14, "coverage": 0.25},
        ),
        "missing": evaluate_reliability("activity", {"available": False}),
        "ood": evaluate_reliability(
            "activity", {"available": True, "domain_level": 3, "ood": True}
        ),
        "stale_history": evaluate_reliability(
            "phq_history", {"available": True, "domain_level": 3}, history_age_days=101
        ),
        "invalid_known_at": evaluate_reliability(
            "phq_history",
            {"available": True, "domain_level": 3, "known_at_valid": False},
            history_age_days=90,
        ),
    }
    violations = [
        name
        for name, value in cases.items()
        if value.effective_level is not None
        and value.original_level is not None
        and value.effective_level > value.original_level
    ]
    expected_status = {
        "usable": "usable",
        "degraded_quality": "degraded",
        "low_coverage": "abstain",
        "missing": "abstain",
        "ood": "abstain",
        "stale_history": "abstain",
        "invalid_known_at": "abstain",
    }
    violations.extend(
        f"{name}:expected_{expected}_got_{cases[name].reliability_status}"
        for name, expected in expected_status.items()
        if cases[name].reliability_status != expected
    )
    return {
        "status": "pass" if not violations else "fail",
        "cases": {name: value.to_dict() for name, value in cases.items()},
        "quality_degradation_upgrades": len(violations),
        "violations": violations,
    }


def run_r11_diagnostics(
    *, repository_root_value: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    sealed_path = root / EVALUATION_RELATIVE / "sleep_history/repeated_outer_oof.parquet"
    sealed_sha_before = sha256_file(sealed_path)
    ablation = _ablation_predictions(load_sleep_history(root))
    ablation_path = root / DIAGNOSTIC_RELATIVE / "sleep_history_ablation_oof.parquet"
    ablation_path.parent.mkdir(parents=True, exist_ok=True)
    if ablation_path.exists() and not overwrite:
        raise FileExistsError(ablation_path)
    ablation.to_parquet(ablation_path, index=False)
    sealed = pd.read_parquet(sealed_path)
    freshness, workpoints = _sealed_diagnostics(sealed)
    report = {
        "status": "pass",
        "sealed_candidate_oof_unchanged": sha256_file(sealed_path) == sealed_sha_before,
        "post_lock_diagnostics_used_for_selection_or_promotion": False,
        "ablation": _ablation_report(ablation),
        "freshness_slices": freshness,
        "fixed_workpoints": workpoints,
        "engineering_stress": _stress_report(),
        "source_slice": "not estimable: 76-100 day Sleep+History is PSYCHE-D only",
        "ablation_oof_sha256": sha256_file(ablation_path),
    }
    if report["engineering_stress"]["status"] != "pass":
        report["status"] = "fail"
    write_json(root / DIAGNOSTIC_RELATIVE / "diagnostic_report.json", report, overwrite=overwrite)
    return report


__all__ = ["DIAGNOSTIC_RELATIVE", "run_r11_diagnostics"]
