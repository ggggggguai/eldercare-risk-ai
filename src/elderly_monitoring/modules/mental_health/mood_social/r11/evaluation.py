"""One-time R11 repeated outer evaluation on the locked S+History candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from elderly_monitoring.modules.mental_health.mood_social.r7.modeling import (
    binary_metrics,
    project_heads,
)
from elderly_monitoring.modules.mental_health.mood_social.r9.modeling import (
    inner_folds_for_outer,
)
from elderly_monitoring.modules.mental_health.mood_social.r10.modeling import (
    participant_weight,
)

from .contract import (
    DEFAULT_DATA_RELATIVE,
    DEFAULT_REPORT_RELATIVE,
    R11_BOOTSTRAP_RESAMPLES,
    R11_BOOTSTRAP_SEED,
    R11_PROTOCOL_VERSION,
    repository_root,
    sha256_file,
    write_json,
)
from .transition_modeling import (
    SELECTION_RELATIVE,
    TARGETS,
    fit_calibrator,
    fit_candidate,
    load_sleep_history,
    predict_calibrated,
)


EVALUATION_RELATIVE = DEFAULT_REPORT_RELATIVE / "evaluation"


def _metric_set(
    frame: pd.DataFrame, target: str, probability: str, *, participant_equal: bool
) -> dict[str, float]:
    weight = participant_weight(frame) if participant_equal else None
    result = binary_metrics(frame[target].astype(int), frame[probability].astype(float), weight=weight)
    result.update(
        {
            "rows": float(len(frame)),
            "participants": float(frame["global_participant_id"].nunique()),
            "positive": float(frame[target].sum()),
            "coverage": 1.0,
        }
    )
    return result


def _outer_predictions(
    frame: pd.DataFrame, lock: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    outputs: list[pd.DataFrame] = []
    fit_audit: list[dict[str, Any]] = []
    perturbation: list[dict[str, Any]] = []
    for repeat, repeat_frame in frame.groupby("repeat", sort=True):
        for outer_fold in range(5):
            train = repeat_frame.loc[repeat_frame["outer_fold"].ne(outer_fold)].copy()
            test = repeat_frame.loc[repeat_frame["outer_fold"].eq(outer_fold)].copy()
            inner = inner_folds_for_outer(repeat_frame, outer_fold).loc[train.index].astype(int)
            columns = [
                "r4_row_id",
                "global_participant_id",
                "dataset_id",
                "repeat",
                "outer_fold",
                "phq9_ge5_target",
                "phq9_ge10_target",
                "history.last_ge5",
                "history.last_ge10",
                "history.age_days",
            ]
            current = test[columns].copy()
            fitted_models: dict[str, tuple[Any, Any, str]] = {}
            baseline_models: dict[str, tuple[Any, Any, str]] = {}
            for head in TARGETS:
                selected = lock["nodes"]["sleep_history"][str(int(repeat))][str(outer_fold)][head]
                seed = 20261100 + int(repeat) * 1000 + outer_fold * 100 + (0 if head == "ge5" else 10)
                fitted, inner_raw = fit_candidate(train, inner, head, selected, seed)
                calibrator = fit_calibrator(train, inner_raw, head, selected["calibration"])
                current[f"candidate_probability_{head}"] = predict_calibrated(
                    fitted, calibrator, selected["calibration"], test
                )
                baseline_selection = {
                    "candidate_id": "r9_joint_replay",
                    "calibration": "platt",
                    "metadata": {},
                }
                baseline_fitted, baseline_inner_raw = fit_candidate(
                    train, inner, head, baseline_selection, seed + 30
                )
                baseline_calibrator = fit_calibrator(
                    train, baseline_inner_raw, head, "platt"
                )
                current[f"baseline_probability_{head}"] = predict_calibrated(
                    baseline_fitted, baseline_calibrator, "platt", test
                )
                fitted_models[head] = (fitted, calibrator, selected["calibration"])
                baseline_models[head] = (baseline_fitted, baseline_calibrator, "platt")
                fit_audit.append(
                    {
                        "repeat": int(repeat),
                        "outer_fold": outer_fold,
                        "head": head,
                        "candidate_id": selected["candidate_id"],
                        "calibration": selected["calibration"],
                        "baseline": "r9_joint_replay+platt",
                        "outer_test_excluded_from_fit_select_calibrate_threshold": True,
                        "inner_oof_rows": int(len(train)),
                        "participant_overlap": int(
                            len(
                                set(train["global_participant_id"].astype(str))
                                & set(test["global_participant_id"].astype(str))
                            )
                        ),
                    }
                )
            candidate5, candidate10 = project_heads(
                current["candidate_probability_ge5"], current["candidate_probability_ge10"]
            )
            baseline5, baseline10 = project_heads(
                current["baseline_probability_ge5"], current["baseline_probability_ge10"]
            )
            current["candidate_probability_ge5"] = candidate5
            current["candidate_probability_ge10"] = candidate10
            current["baseline_probability_ge5"] = baseline5
            current["baseline_probability_ge10"] = baseline10
            shifted = test.copy()
            shifted["history.age_days"] = shifted["history.age_days"].astype(float) + 7.0
            for head in TARGETS:
                fitted, calibrator, method = fitted_models[head]
                value = predict_calibrated(fitted, calibrator, method, shifted)
                perturbation.append(
                    {
                        "repeat": int(repeat),
                        "outer_fold": outer_fold,
                        "head": head,
                        "perturbation": "history_age_plus_7_days",
                        "finite": bool(np.isfinite(value).all()),
                        "mean_absolute_probability_change": float(
                            np.mean(np.abs(value - current[f"candidate_probability_{head}"].to_numpy(float)))
                        ),
                    }
                )
            outputs.append(current)
    return pd.concat(outputs, ignore_index=True), fit_audit, perturbation


def _bootstrap(frame: pd.DataFrame, head: str) -> dict[str, Any]:
    participant_values = frame["global_participant_id"].astype(str).unique()
    participant_index = {value: index for index, value in enumerate(participant_values)}
    row_participant = np.array(
        [participant_index[value] for value in frame["global_participant_id"].astype(str)], int
    )
    row_count = np.bincount(row_participant)
    base_weight = 1.0 / row_count[row_participant]
    y = frame[TARGETS[head]].to_numpy(int)
    candidate = frame[f"candidate_probability_{head}"].to_numpy(float)
    baseline = frame[f"baseline_probability_{head}"].to_numpy(float)
    rng = np.random.default_rng(R11_BOOTSTRAP_SEED + (0 if head == "ge5" else 1))
    values = np.empty(R11_BOOTSTRAP_RESAMPLES, float)
    for index in range(R11_BOOTSTRAP_RESAMPLES):
        sampled = rng.integers(0, len(participant_values), len(participant_values))
        multiplicity = np.bincount(sampled, minlength=len(participant_values))
        weight = base_weight * multiplicity[row_participant]
        selected = weight > 0
        values[index] = average_precision_score(
            y[selected], candidate[selected], sample_weight=weight[selected]
        ) - average_precision_score(
            y[selected], baseline[selected], sample_weight=weight[selected]
        )
    return {
        "resamples": R11_BOOTSTRAP_RESAMPLES,
        "seed": R11_BOOTSTRAP_SEED + (0 if head == "ge5" else 1),
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
        "positive_fraction": float(np.mean(values > 0)),
    }


def _transition(frame: pd.DataFrame, head: str) -> dict[str, Any]:
    prior = f"history.last_{head}"
    target = TARGETS[head]
    definitions = {
        "new_onset": (frame[prior].eq(0), False),
        "recovery": (frame[prior].eq(1), True),
        "persistent_high": (frame[prior].eq(1), False),
        "persistent_low": (frame[prior].eq(0), True),
    }
    result: dict[str, Any] = {}
    for name, (mask, invert) in definitions.items():
        part = frame.loc[mask].copy()
        y = part[target].astype(int)
        candidate = part[f"candidate_probability_{head}"].astype(float)
        baseline = part[f"baseline_probability_{head}"].astype(float)
        if invert:
            y, candidate, baseline = 1 - y, 1 - candidate, 1 - baseline
        if len(part) and y.nunique() == 2:
            part["_target"] = y
            part["_candidate"] = candidate
            part["_baseline"] = baseline
            cm = _metric_set(part, "_target", "_candidate", participant_equal=True)
            bm = _metric_set(part, "_target", "_baseline", participant_equal=True)
            result[name] = {
                "rows": int(len(part)),
                "participants": int(part["global_participant_id"].nunique()),
                "candidate": cm,
                "baseline": bm,
                "delta": {key: cm[key] - bm[key] for key in ("auprc", "normalized_ap", "auroc", "brier", "ece")},
            }
        else:
            result[name] = {
                "rows": int(len(part)),
                "participants": int(part["global_participant_id"].nunique()),
                "metric_available": False,
            }
    return result


def _head_report(frame: pd.DataFrame, head: str) -> dict[str, Any]:
    target = TARGETS[head]
    candidate = _metric_set(frame, target, f"candidate_probability_{head}", participant_equal=True)
    baseline = _metric_set(frame, target, f"baseline_probability_{head}", participant_equal=True)
    natural_candidate = _metric_set(frame, target, f"candidate_probability_{head}", participant_equal=False)
    natural_baseline = _metric_set(frame, target, f"baseline_probability_{head}", participant_equal=False)
    keys = ("auprc", "normalized_ap", "auroc", "brier", "ece")
    folds: list[dict[str, Any]] = []
    for (repeat, outer_fold), part in frame.groupby(["repeat", "outer_fold"], sort=True):
        cm = _metric_set(part, target, f"candidate_probability_{head}", participant_equal=True)
        bm = _metric_set(part, target, f"baseline_probability_{head}", participant_equal=True)
        folds.append(
            {
                "repeat": int(repeat),
                "outer_fold": int(outer_fold),
                "delta_auprc": cm["auprc"] - bm["auprc"],
                "delta_auroc": cm["auroc"] - bm["auroc"],
            }
        )
    nonnegative = {
        str(repeat): int(sum(row["delta_auprc"] >= 0 for row in folds if row["repeat"] == repeat))
        for repeat in range(3)
    }
    return {
        "candidate": candidate,
        "baseline": baseline,
        "delta": {key: candidate[key] - baseline[key] for key in keys},
        "natural_row_weighted": {
            "candidate": natural_candidate,
            "baseline": natural_baseline,
            "delta": {key: natural_candidate[key] - natural_baseline[key] for key in keys},
        },
        "fold_delta": folds,
        "nonnegative_folds_by_repeat": nonnegative,
        "paired_participant_bootstrap_delta_auprc": _bootstrap(frame, head),
        "transitions": _transition(frame, head),
    }


def _promotion(heads: dict[str, Any]) -> dict[str, Any]:
    ge5, ge10 = heads["ge5"], heads["ge10"]
    d5, d10 = ge5["delta"], ge10["delta"]
    repeat_stability = sum(
        ge5["nonnegative_folds_by_repeat"][str(repeat)] >= 3
        and ge10["nonnegative_folds_by_repeat"][str(repeat)] >= 3
        for repeat in range(3)
    ) >= 2
    calibration = (
        (d5["brier"] < 0 or d5["ece"] < 0 or d10["brier"] < 0 or d10["ece"] < 0)
        and max(d5["brier"], d10["brier"]) <= 0.005
        and max(d5["ece"], d10["ece"]) <= 0.010
    )
    overall = (
        (d5["auprc"] >= 0.0005 or d10["auprc"] >= 0.0010)
        and (d10["auprc"] >= -0.0010 if d5["auprc"] >= 0.0005 else d5["auprc"] >= -0.0010)
        and repeat_stability
        and calibration
    )
    new_onset = ge10["transitions"]["new_onset"].get("delta", {}).get("auprc", -1.0)
    recovery = ge10["transitions"]["recovery"].get("delta", {}).get("auprc", -1.0)
    transition = (
        new_onset >= 0.004
        and recovery >= 0.001
        and d5["auprc"] >= -0.001
        and d10["auprc"] >= -0.001
    )
    safety = (
        d5["auroc"] >= -0.010
        and d10["auroc"] >= -0.010
        and d5["brier"] <= 0.005
        and d10["brier"] <= 0.005
        and d5["ece"] <= 0.010
        and d10["ece"] <= 0.010
    )
    return {
        "promoted": bool((overall or transition) and safety),
        "path": "overall" if overall and safety else "transition" if transition and safety else "no-go",
        "overall_path_passed": bool(overall),
        "transition_path_passed": bool(transition),
        "hard_safety_passed": bool(safety),
        "repeat_stability_passed": bool(repeat_stability),
        "calibration_condition_passed": bool(calibration),
        "new_onset_ge10_delta_auprc": float(new_onset),
        "recovery_ge10_delta_auprc": float(recovery),
    }


def run_r11_evaluation(
    *, repository_root_value: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    protocol_path = root / DEFAULT_DATA_RELATIVE / "protocol/r11_protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock_path = root / SELECTION_RELATIVE / "selection_lock.json"
    if protocol.get("status") != "selection-locked-r11-outer-unopened":
        raise ValueError("R11 outer evaluation requires the locked unopened protocol")
    if protocol.get("candidate_outer_opened"):
        raise ValueError("R11 outer has already been opened")
    if sha256_file(lock_path) != protocol.get("selection_lock_sha256"):
        raise ValueError("R11 selection lock drift")
    event = {
        "event": "r11-candidate-outer-opened-once",
        "protocol_version": R11_PROTOCOL_VERSION,
        "selection_lock_sha256": sha256_file(lock_path),
        "r10_outer_reopened": False,
        "repeats": 3,
        "outer_folds": 5,
        "burn_after_open": True,
    }
    write_json(root / EVALUATION_RELATIVE / "outer_open_event.json", event, overwrite=overwrite)
    protocol["candidate_outer_opened"] = True
    protocol["status"] = "r11-candidate-outer-opened-burned"
    write_json(protocol_path, protocol, overwrite=True)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    predictions, fit_audit, perturbation = _outer_predictions(load_sleep_history(root), lock)
    oof_path = root / EVALUATION_RELATIVE / "sleep_history/repeated_outer_oof.parquet"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    if oof_path.exists() and not overwrite:
        raise FileExistsError(oof_path)
    predictions.to_parquet(oof_path, index=False)
    heads = {head: _head_report(predictions, head) for head in TARGETS}
    promotion = _promotion(heads)
    report = {
        "protocol_version": R11_PROTOCOL_VERSION,
        "status": "pass",
        "evidence_status": "adaptive-development/reused-benchmark/post-selection",
        "candidate_outer_opened_once": True,
        "r10_outer_reopened": False,
        "complete_r11_multimodal_auprc_reported": False,
        "activity_sleep": {
            "status": "cancelled",
            "reason": "no new runtime-equivalent information beyond R10",
            "metrics_reported": False,
        },
        "sleep_history": {
            "heads": heads,
            "promotion": promotion,
            "oof_sha256": sha256_file(oof_path),
            "probability_time_band_days": [76, 100],
            "source_generalization": "not estimable: PSYCHE-D only",
        },
        "promoted_nodes": ["sleep_history"] if promotion["promoted"] else [],
        "stop_rule": {
            "mean_winner_inner_delta_auprc": lock["mean_winner_inner_delta_auprc"],
            "threshold": 0.0004,
            "triggered": bool(lock["mean_winner_inner_delta_auprc"] < 0.0004),
            "no_post_outer_retuning": True,
        },
    }
    write_json(root / EVALUATION_RELATIVE / "fold_fit_audit.json", fit_audit, overwrite=overwrite)
    write_json(root / EVALUATION_RELATIVE / "perturbation_report.json", perturbation, overwrite=overwrite)
    write_json(root / EVALUATION_RELATIVE / "evaluation_report.json", report, overwrite=overwrite)
    protocol["status"] = "evaluation-complete"
    protocol["promoted_nodes"] = report["promoted_nodes"]
    protocol["evaluation_report_sha256"] = sha256_file(
        root / EVALUATION_RELATIVE / "evaluation_report.json"
    )
    protocol["tasks_completed"] = list(dict.fromkeys(list(protocol["tasks_completed"]) + [
        "OPT-V333-R11-002-evaluation",
        "OPT-V333-R11-003-cancelled-no-new-runtime-information",
        "OPT-V333-R11-006-supervised-evaluation",
    ]))
    write_json(protocol_path, protocol, overwrite=True)
    return report


__all__ = ["EVALUATION_RELATIVE", "run_r11_evaluation"]
