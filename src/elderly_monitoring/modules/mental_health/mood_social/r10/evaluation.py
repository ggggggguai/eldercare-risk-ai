"""One-time repeated nested outer evaluation and robustness audit for R10."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from elderly_monitoring.modules.mental_health.mood_social.r7.modeling import (
    binary_metrics,
    project_heads,
    select_specificity_threshold,
)
from elderly_monitoring.modules.mental_health.mood_social.r9.modeling import inner_folds_for_outer

from .baseline import BASELINE_REPORT_RELATIVE, load_r10_frames
from .contract import (
    DEFAULT_DATA_RELATIVE,
    DEFAULT_REPORT_RELATIVE,
    R10_BOOTSTRAP_RESAMPLES,
    R10_BOOTSTRAP_SEED,
    R10_PROTOCOL_VERSION,
    repository_root,
    sha256_file,
    write_json,
)
from .evidence_graph import enumerate_signature_routes, validate_signature_routes
from .features import (
    ACTIVITY_SLEEP_BASE_FEATURES,
    ACTIVITY_SLEEP_INTERACTION_FEATURES,
    SLEEP_HISTORY_FEATURES,
)
from .modeling import (
    HEAD_TARGETS,
    apply_calibration,
    candidate_inner_predictions,
    fit_final_calibration,
    fit_selected_candidate,
    participant_weight,
    _fit_simple,
    _fold_oof_simple,
)
from .rule_fusion import enumerate_truth_table, validate_truth_table
from .selection import SELECTION_REPORT_RELATIVE


EVALUATION_REPORT_RELATIVE = DEFAULT_REPORT_RELATIVE / "evaluation"


def _metrics(frame: pd.DataFrame, target: str, probability: str) -> dict[str, float]:
    result = binary_metrics(
        frame[target].astype(int),
        frame[probability].astype(float),
        weight=participant_weight(frame),
    )
    threshold_80 = select_specificity_threshold(
        frame[target].astype(int), frame[probability].astype(float), 0.80
    )
    threshold_90 = select_specificity_threshold(
        frame[target].astype(int), frame[probability].astype(float), 0.90
    )
    positive = frame[target].astype(int).eq(1)
    result["sensitivity_at_specificity_080"] = float(
        frame.loc[positive, probability].ge(threshold_80).mean()
    )
    result["sensitivity_at_specificity_090"] = float(
        frame.loc[positive, probability].ge(threshold_90).mean()
    )
    result["rows"] = float(len(frame))
    result["participants"] = float(frame["global_participant_id"].nunique())
    result["positive"] = float(frame[target].sum())
    result["coverage"] = 1.0
    return result


def _outer_predictions(
    node: str, frame: pd.DataFrame, lock: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    perturbations: list[dict[str, Any]] = []
    for repeat, repeat_frame in frame.groupby("repeat", sort=True):
        for outer_fold in range(5):
            train = repeat_frame.loc[repeat_frame["outer_fold"].ne(outer_fold)].copy()
            test = repeat_frame.loc[repeat_frame["outer_fold"].eq(outer_fold)].copy()
            folds = inner_folds_for_outer(repeat_frame, outer_fold).loc[train.index].astype(int)
            current = test[
                [
                    "r4_row_id",
                    "global_participant_id",
                    "dataset_id",
                    "repeat",
                    "outer_fold",
                    "phq9_ge5_target",
                    "phq9_ge10_target",
                    *(
                        ["history.last_ge5", "history.last_ge10", "history.age_days"]
                        if node == "sleep_history"
                        else []
                    ),
                    *(
                        ["social_context.age_group", "social_context.sex"]
                        if node == "activity_sleep"
                        else []
                    ),
                ]
            ].copy()
            selected_models: dict[str, Any] = {}
            for head, target in HEAD_TARGETS.items():
                selection = lock["nodes"][node][str(int(repeat))][str(outer_fold)][head]
                inner_probability, inner_metadata = candidate_inner_predictions(
                    node,
                    train,
                    folds,
                    head,
                    20260821 + int(repeat) * 1000 + outer_fold * 100,
                )
                candidate_id = selection["candidate_id"]
                if candidate_id not in inner_probability:
                    raise ValueError(f"locked R10 candidate disappeared: {candidate_id}")
                fit_selection = dict(selection)
                fit_selection["metadata"] = dict(inner_metadata[candidate_id])
                # Lambda is part of the lock, regression coefficients are recomputed fold-locally.
                if candidate_id.startswith("residual_"):
                    fit_selection["metadata"]["lambda"] = selection["metadata"]["lambda"]
                fitted = fit_selected_candidate(
                    node,
                    train,
                    folds,
                    head,
                    20260831 + int(repeat) * 1000 + outer_fold * 100,
                    fit_selection,
                    inner_metadata,
                )
                raw_outer = fitted.predict_raw(test)
                calibration = fit_final_calibration(
                    train, inner_probability[candidate_id], target, selection["calibration"]
                )
                probability = apply_calibration(calibration, raw_outer, selection["calibration"])
                current[f"candidate_probability_{head}"] = probability
                selected_models[head] = (fitted, calibration, selection["calibration"])
                audits.append(
                    {
                        "node": node,
                        "repeat": int(repeat),
                        "outer_fold": outer_fold,
                        "head": head,
                        "candidate_id": candidate_id,
                        "calibration": selection["calibration"],
                        "outer_test_excluded_from_all_fit_select_calibrate_threshold": True,
                        "inner_oof_rows": int(len(train)),
                    }
                )
            p5, p10 = project_heads(
                current["candidate_probability_ge5"], current["candidate_probability_ge10"]
            )
            current["candidate_probability_ge5"] = p5
            current["candidate_probability_ge10"] = p10
            rng = np.random.default_rng(20260901 + int(repeat) * 100 + outer_fold)
            variants: dict[str, pd.DataFrame] = {}
            missing = test.copy()
            risk_features = list(
                ACTIVITY_SLEEP_INTERACTION_FEATURES
                if node == "activity_sleep"
                else SLEEP_HISTORY_FEATURES
            )
            mask = rng.random((len(missing), len(risk_features))) < 0.10
            for index, feature in enumerate(risk_features):
                missing.loc[mask[:, index], feature] = np.nan
            variants["ten_percent_field_dropout"] = missing
            noisy = test.copy()
            for feature in risk_features:
                numeric = pd.to_numeric(noisy[feature], errors="coerce")
                scale = float(numeric.std())
                if np.isfinite(scale) and scale > 0:
                    noisy[feature] = numeric + rng.normal(0.0, 0.05 * scale, len(noisy))
            variants["five_percent_feature_noise"] = noisy
            if node == "sleep_history":
                shifted = test.copy()
                shifted["history.age_days"] = shifted["history.age_days"] + 7.0
                shifted["history.freshness_weight"] = np.exp(
                    -(shifted["history.age_days"] - 30.0) / 90.0
                )
                shifted["history.decayed_last_score"] = (
                    shifted["history.last_score"] * shifted["history.freshness_weight"]
                )
                variants["history_time_shift_plus_7_days"] = shifted
            for name, variant in variants.items():
                row: dict[str, Any] = {
                    "node": node,
                    "repeat": int(repeat),
                    "outer_fold": outer_fold,
                    "perturbation": name,
                }
                for head in HEAD_TARGETS:
                    fitted, calibration, method = selected_models[head]
                    shifted_probability = apply_calibration(
                        calibration, fitted.predict_raw(variant), method
                    )
                    row[f"mean_absolute_probability_change_{head}"] = float(
                        np.mean(
                            np.abs(
                                shifted_probability
                                - current[f"candidate_probability_{head}"].to_numpy(float)
                            )
                        )
                    )
                    row[f"finite_{head}"] = bool(np.isfinite(shifted_probability).all())
                perturbations.append(row)
            outputs.append(current)
    result = pd.concat(outputs, ignore_index=True)
    return result, audits, perturbations


def _align_baseline(root: Path, node: str, frame: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.read_parquet(
        root
        / BASELINE_REPORT_RELATIVE
        / node
        / "r9_recipe_repeated_outer_oof.parquet"
    )
    columns = [
        "r4_row_id",
        "repeat",
        "baseline_probability_ge5",
        "baseline_probability_ge10",
    ]
    result = frame.merge(
        baseline[columns], on=["r4_row_id", "repeat"], how="left", validate="one_to_one"
    )
    if result[["baseline_probability_ge5", "baseline_probability_ge10"]].isna().any().any():
        raise ValueError(f"R10 baseline alignment incomplete: {node}")
    return result


def _bootstrap(frame: pd.DataFrame, target: str, seed: int) -> dict[str, Any]:
    participants = frame["global_participant_id"].astype(str).unique()
    groups = {
        participant: np.flatnonzero(
            frame["global_participant_id"].astype(str).to_numpy() == participant
        )
        for participant in participants
    }
    base_count = np.array(
        [len(groups[value]) for value in frame["global_participant_id"].astype(str)], float
    )
    rng = np.random.default_rng(seed)
    delta = np.empty(R10_BOOTSTRAP_RESAMPLES, float)
    y = frame[target].to_numpy(int)
    candidate = frame[f"candidate_probability_{target.split('_')[1]}"] if False else None
    del candidate
    head = "ge5" if target.endswith("ge5_target") else "ge10"
    p_candidate = frame[f"candidate_probability_{head}"].to_numpy(float)
    p_baseline = frame[f"baseline_probability_{head}"].to_numpy(float)
    participant_index = {participant: index for index, participant in enumerate(participants)}
    row_participant = np.array(
        [participant_index[value] for value in frame["global_participant_id"].astype(str)], int
    )
    row_base = 1.0 / base_count
    for index in range(R10_BOOTSTRAP_RESAMPLES):
        sampled = rng.integers(0, len(participants), len(participants))
        multiplicity = np.bincount(sampled, minlength=len(participants))
        weight = row_base * multiplicity[row_participant]
        selected = weight > 0
        delta[index] = average_precision_score(
            y[selected], p_candidate[selected], sample_weight=weight[selected]
        ) - average_precision_score(
            y[selected], p_baseline[selected], sample_weight=weight[selected]
        )
    return {
        "resamples": R10_BOOTSTRAP_RESAMPLES,
        "seed": seed,
        "lower_95": float(np.quantile(delta, 0.025)),
        "median": float(np.quantile(delta, 0.5)),
        "upper_95": float(np.quantile(delta, 0.975)),
        "positive_fraction": float((delta > 0).mean()),
    }


def _transition_report(frame: pd.DataFrame, head: str) -> dict[str, Any]:
    target = f"phq9_{head}_target"
    prior = f"history.last_{head}"
    result: dict[str, Any] = {}
    definitions = {
        "new_onset": (
            frame[prior].eq(0), frame[target], frame[f"candidate_probability_{head}"], frame[f"baseline_probability_{head}"]
        ),
        "recovery": (
            frame[prior].eq(1), 1 - frame[target], 1 - frame[f"candidate_probability_{head}"], 1 - frame[f"baseline_probability_{head}"]
        ),
        "persistent_high": (
            frame[prior].eq(1), frame[target], frame[f"candidate_probability_{head}"], frame[f"baseline_probability_{head}"]
        ),
        "persistent_low": (
            frame[prior].eq(0), 1 - frame[target], 1 - frame[f"candidate_probability_{head}"], 1 - frame[f"baseline_probability_{head}"]
        ),
    }
    for name, (mask, y, candidate_p, baseline_p) in definitions.items():
        subset = frame.loc[mask].copy()
        yy = y.loc[mask].astype(int)
        pp = candidate_p.loc[mask].astype(float)
        bb = baseline_p.loc[mask].astype(float)
        if len(subset) and yy.nunique() == 2:
            candidate_metrics = binary_metrics(yy, pp, weight=participant_weight(subset))
            baseline_metrics = binary_metrics(yy, bb, weight=participant_weight(subset))
            result[name] = {
                "candidate": candidate_metrics,
                "baseline": baseline_metrics,
                "delta": {
                    key: candidate_metrics[key] - baseline_metrics[key]
                    for key in ("auprc", "normalized_ap", "auroc", "brier", "ece")
                },
                "rows": int(len(subset)),
                "participants": int(subset["global_participant_id"].nunique()),
            }
        else:
            result[name] = {
                "rows": int(len(subset)),
                "participants": int(subset["global_participant_id"].nunique()),
                "metric_available": False,
            }
    discordant = frame[
        frame[prior].astype(int).ne(frame[target].astype(int))
    ]
    result["discordant"] = {
        "rows": int(len(discordant)),
        "participants": int(discordant["global_participant_id"].nunique()),
    }
    return result


def _slice_report(frame: pd.DataFrame, head: str) -> dict[str, Any]:
    target = f"phq9_{head}_target"
    output: dict[str, Any] = {"source": {}}
    for source, part in frame.groupby("dataset_id", sort=True):
        if part[target].nunique() == 2:
            output["source"][str(source)] = {
                "candidate": _metrics(part, target, f"candidate_probability_{head}"),
                "baseline": _metrics(part, target, f"baseline_probability_{head}"),
            }
    for column, name in (
        ("social_context.age_group", "age"),
        ("social_context.sex", "sex"),
    ):
        output[name] = {}
        if column in frame:
            for value, part in frame.groupby(column, dropna=False):
                if len(part) >= 20 and part[target].nunique() == 2:
                    output[name][str(value)] = {
                        "candidate": _metrics(part, target, f"candidate_probability_{head}"),
                        "baseline": _metrics(part, target, f"baseline_probability_{head}"),
                    }
    return output


def _node_report(node: str, frame: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {"heads": {}}
    for head, target in HEAD_TARGETS.items():
        candidate = _metrics(frame, target, f"candidate_probability_{head}")
        baseline = _metrics(frame, target, f"baseline_probability_{head}")
        delta = {key: candidate[key] - baseline[key] for key in ("auprc", "normalized_ap", "auroc", "brier", "ece", "sensitivity_at_specificity_080")}
        fold_delta = []
        for (repeat, fold), part in frame.groupby(["repeat", "outer_fold"], sort=True):
            current_candidate = _metrics(part, target, f"candidate_probability_{head}")
            current_baseline = _metrics(part, target, f"baseline_probability_{head}")
            fold_delta.append(
                {
                    "repeat": int(repeat),
                    "outer_fold": int(fold),
                    "delta_auprc": current_candidate["auprc"] - current_baseline["auprc"],
                    "delta_auroc": current_candidate["auroc"] - current_baseline["auroc"],
                }
            )
        repeat_nonnegative = {
            str(repeat): int(sum(row["delta_auprc"] >= 0 for row in fold_delta if row["repeat"] == repeat))
            for repeat in sorted(frame["repeat"].unique())
        }
        report["heads"][head] = {
            "candidate": candidate,
            "baseline": baseline,
            "delta": delta,
            "fold_delta": fold_delta,
            "nonnegative_folds_by_repeat": repeat_nonnegative,
            "paired_participant_bootstrap_delta_auprc": _bootstrap(
                frame, target, R10_BOOTSTRAP_SEED + (0 if head == "ge5" else 1)
            ),
            "slices": _slice_report(frame, head),
        }
        if node == "sleep_history":
            report["heads"][head]["transitions"] = _transition_report(frame, head)
    return report


def _promotion(node: str, report: dict[str, Any]) -> dict[str, Any]:
    ge5 = report["heads"]["ge5"]
    ge10 = report["heads"]["ge10"]
    stability5 = all(value >= 3 for value in ge5["nonnegative_folds_by_repeat"].values()) or ge5["paired_participant_bootstrap_delta_auprc"]["lower_95"] >= -0.010
    stability10 = all(value >= 3 for value in ge10["nonnegative_folds_by_repeat"].values()) or ge10["paired_participant_bootstrap_delta_auprc"]["lower_95"] >= -0.010
    safety = all(
        block["delta"]["auroc"] >= -0.015
        and block["delta"]["brier"] <= 0.010
        and block["delta"]["ece"] <= 0.020
        for block in (ge5, ge10)
    )
    if node == "activity_sleep":
        standard = (
            ge5["delta"]["auprc"] >= 0.003
            and ge10["delta"]["auprc"] >= 0.005
            and stability5
            and stability10
            and safety
        )
        robust = (
            ge5["delta"]["auprc"] >= 0.001
            and ge10["delta"]["auprc"] >= -0.005
            and safety
            and stability5
            and sum(
                (
                    ge5["delta"]["brier"] < 0,
                    ge5["delta"]["ece"] < 0,
                    ge5["delta"]["normalized_ap"] > 0,
                    min(
                        value["candidate"]["normalized_ap"] - value["baseline"]["normalized_ap"]
                        for value in ge5["slices"]["source"].values()
                    ) > 0 if ge5["slices"]["source"] else False,
                )
            )
            >= 2
        )
        return {
            "promoted": bool(standard or robust),
            "path": "standard" if standard else "robustness-promoted" if robust else "no-go",
            "safety": safety,
        }
    new_onset = ge10.get("transitions", {}).get("new_onset", {})
    recovery = ge10.get("transitions", {}).get("recovery", {})
    transition_available = "delta" in new_onset or "delta" in recovery
    ranking = ge10["delta"]["auprc"] >= 0.002 and ge5["delta"]["auprc"] >= -0.002
    transition_path = (
        ge10["delta"]["auprc"] >= -0.001
        and transition_available
        and max(
            new_onset.get("delta", {}).get("auprc", -1),
            recovery.get("delta", {}).get("auprc", -1),
        ) >= 0.005
    )
    calibration_path = (
        ge10["delta"]["auprc"] >= -0.001
        and ge10["delta"]["brier"] < 0
        and ge10["delta"]["ece"] < 0
        and ge10["delta"]["sensitivity_at_specificity_080"] >= 0.02
    )
    return {
        "promoted": bool((ranking or transition_path or calibration_path) and safety),
        "path": "ranking" if ranking else "transition" if transition_path else "calibration" if calibration_path else "no-go",
        "safety": safety,
    }


def run_r10_evaluation(
    *, repository_root_value: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    protocol_path = root / DEFAULT_DATA_RELATIVE / "protocol/r10_protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock_path = root / SELECTION_REPORT_RELATIVE / "selection_lock.json"
    if protocol.get("status") != "selection-locked-candidate-outer-unopened":
        raise ValueError("R10 evaluation requires a locked and unopened outer protocol")
    if sha256_file(lock_path) != protocol.get("selection_lock_sha256"):
        raise ValueError("R10 selection lock drift")
    event = {
        "event": "candidate-outer-opened-once",
        "protocol_version": R10_PROTOCOL_VERSION,
        "selection_lock_sha256": sha256_file(lock_path),
        "repeats": 3,
        "outer_folds": 5,
        "burn_after_open": True,
    }
    write_json(root / EVALUATION_REPORT_RELATIVE / "outer_open_event.json", event, overwrite=overwrite)
    protocol["candidate_outer_opened"] = True
    protocol["status"] = "candidate-outer-opened-burned"
    write_json(protocol_path, protocol, overwrite=True)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "protocol_version": R10_PROTOCOL_VERSION,
        "status": "pass",
        "evidence_status": "adaptive-development/reused-benchmark/locked-procedure-estimate",
        "candidate_outer_opened_once": True,
        "nodes": {},
        "promoted_nodes": [],
        "complete_r10_auprc_reported": False,
    }
    all_audits: list[dict[str, Any]] = []
    all_perturbations: list[dict[str, Any]] = []
    for node, frame in load_r10_frames(root).items():
        prediction, audits, perturbations = _outer_predictions(node, frame, lock)
        prediction = _align_baseline(root, node, prediction)
        path = root / EVALUATION_REPORT_RELATIVE / node / "repeated_outer_oof.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        prediction.to_parquet(path, index=False)
        node_report = _node_report(node, prediction)
        node_report["promotion"] = _promotion(node, node_report)
        node_report["oof_sha256"] = sha256_file(path)
        report["nodes"][node] = node_report
        if node_report["promotion"]["promoted"]:
            report["promoted_nodes"].append(node)
        all_audits.extend(audits)
        all_perturbations.extend(perturbations)
    graph = validate_signature_routes()
    truth = validate_truth_table()
    write_json(root / EVALUATION_REPORT_RELATIVE / "evidence_graph_routes.json", enumerate_signature_routes(), overwrite=overwrite)
    write_json(root / EVALUATION_REPORT_RELATIVE / "evidence_graph_audit.json", graph, overwrite=overwrite)
    write_json(root / EVALUATION_REPORT_RELATIVE / "rule_truth_table.json", enumerate_truth_table(), overwrite=overwrite)
    write_json(root / EVALUATION_REPORT_RELATIVE / "rule_truth_table_audit.json", truth, overwrite=overwrite)
    write_json(root / EVALUATION_REPORT_RELATIVE / "fold_fit_audit.json", all_audits, overwrite=overwrite)
    write_json(root / EVALUATION_REPORT_RELATIVE / "perturbation_report.json", all_perturbations, overwrite=overwrite)
    write_json(
        root / EVALUATION_REPORT_RELATIVE / "lodo_report.json",
        {
            "activity_sleep": "two-source diagnostic reported in source slices; RESILIENT n=71 is descriptive",
            "sleep_history": "not estimable: PSYCHE-D is the only source",
            "route_policy": "source-specific degradation blocks only that signature",
        },
        overwrite=overwrite,
    )
    if graph["status"] != "pass" or truth["status"] != "pass":
        report["status"] = "fail"
    write_json(root / EVALUATION_REPORT_RELATIVE / "evaluation_report.json", report, overwrite=overwrite)
    protocol["status"] = "evaluation-complete" if report["status"] == "pass" else "evaluation-failed"
    protocol["tasks_completed"] = list(protocol["tasks_completed"]) + [
        "OPT-V333-R10-005",
        "OPT-V333-R10-006",
        "OPT-V333-R10-007",
    ]
    protocol["promoted_nodes"] = report["promoted_nodes"]
    protocol["evaluation_report_sha256"] = sha256_file(
        root / EVALUATION_REPORT_RELATIVE / "evaluation_report.json"
    )
    write_json(protocol_path, protocol, overwrite=True)
    return report


def repair_r10_transition_reporting(
    *, repository_root_value: Path | None = None
) -> dict[str, Any]:
    """Repair a reporting-only transition delta bug without refitting or reopening outer data."""

    root = repository_root(repository_root_value)
    protocol_path = root / DEFAULT_DATA_RELATIVE / "protocol/r10_protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("candidate_outer_opened") is not True:
        raise ValueError("R10 transition report repair requires the sealed outer predictions")
    report_path = root / EVALUATION_REPORT_RELATIVE / "evaluation_report.json"
    prior = json.loads(report_path.read_text(encoding="utf-8"))
    repaired = {
        **prior,
        "nodes": {},
        "promoted_nodes": [],
        "reporting_repairs": list(prior.get("reporting_repairs", []))
        + [
            {
                "issue": "transition gate compared candidate absolute AP instead of paired delta AP",
                "scope": "reporting-and-promotion-decision-only",
                "models_refit": False,
                "outer_predictions_changed": False,
            }
        ],
    }
    for node in ("activity_sleep", "sleep_history"):
        frame = pd.read_parquet(
            root / EVALUATION_REPORT_RELATIVE / node / "repeated_outer_oof.parquet"
        )
        node_report = _node_report(node, frame)
        node_report["promotion"] = _promotion(node, node_report)
        node_report["oof_sha256"] = sha256_file(
            root / EVALUATION_REPORT_RELATIVE / node / "repeated_outer_oof.parquet"
        )
        repaired["nodes"][node] = node_report
        if node_report["promotion"]["promoted"]:
            repaired["promoted_nodes"].append(node)
    write_json(report_path, repaired, overwrite=True)
    protocol["promoted_nodes"] = repaired["promoted_nodes"]
    protocol["evaluation_report_sha256"] = sha256_file(report_path)
    protocol["reporting_repair_applied"] = True
    protocol["status"] = "evaluation-complete"
    write_json(protocol_path, protocol, overwrite=True)
    failure_path = root / DEFAULT_REPORT_RELATIVE / "failure_ledger.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    failure["failures"].append(
        {
            "stage": "OPT-V333-R10-007-reporting",
            "failure": "transition promotion used absolute AP instead of paired delta AP",
            "resolution": "recomputed report from immutable sealed OOF; no model refit or retuning",
            "recovered": True,
        }
    )
    write_json(failure_path, failure, overwrite=True)
    return repaired


def run_r10_lodo(*, repository_root_value: Path | None = None) -> dict[str, Any]:
    """Evaluate the frozen modal A+S recipes while holding out each whole source."""

    root = repository_root(repository_root_value)
    lock = json.loads(
        (root / SELECTION_REPORT_RELATIVE / "selection_lock.json").read_text(encoding="utf-8")
    )
    frame = load_r10_frames(root)["activity_sleep"]
    frame = frame.loc[frame["repeat"].eq(0)].copy().reset_index(drop=True)
    result: dict[str, Any] = {
        "protocol": "whole-source-held-out; modal recipes frozen before LODO",
        "sources": {},
        "sleep_history": "not-estimable-single-source-psyche_d",
    }
    for held_source in sorted(frame["dataset_id"].unique()):
        train = frame.loc[frame["dataset_id"].ne(held_source)].copy()
        test = frame.loc[frame["dataset_id"].eq(held_source)].copy()
        source_result: dict[str, Any] = {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "test_participants": int(test["global_participant_id"].nunique()),
            "heads": {},
            "failures": [],
        }
        for head, target in HEAD_TARGETS.items():
            try:
                folds = pd.Series(-1, index=train.index, dtype=int)
                split = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260930)
                for fold, (_, validation_position) in enumerate(
                    split.split(train, train[target].astype(int))
                ):
                    folds.iloc[validation_position] = fold
                rows = [
                    lock["nodes"]["activity_sleep"][str(repeat)][str(outer)][head]
                    for repeat in range(3)
                    for outer in range(5)
                ]
                candidate_id = Counter(row["candidate_id"] for row in rows).most_common(1)[0][0]
                candidate_method = Counter(
                    row["calibration"] for row in rows if row["candidate_id"] == candidate_id
                ).most_common(1)[0][0]
                baseline_raw = _fold_oof_simple(
                    train, folds, "r9_elastic_replay", head, 20260931
                )
                candidate_raw = _fold_oof_simple(
                    train, folds, candidate_id, head, 20260941
                )
                baseline_model = _fit_simple(train, "r9_elastic_replay", head, 20260951)
                candidate_model = _fit_simple(train, candidate_id, head, 20260961)
                baseline_calibration = fit_final_calibration(
                    train, baseline_raw, target, "platt"
                )
                candidate_calibration = fit_final_calibration(
                    train, candidate_raw, target, candidate_method
                )
                baseline_probability = apply_calibration(
                    baseline_calibration, baseline_model.predict_raw(test), "platt"
                )
                candidate_probability = apply_calibration(
                    candidate_calibration,
                    candidate_model.predict_raw(test),
                    candidate_method,
                )
                work = test.copy()
                work["_baseline"] = baseline_probability
                work["_candidate"] = candidate_probability
                baseline_metrics = _metrics(work, target, "_baseline")
                candidate_metrics = _metrics(work, target, "_candidate")
                source_result["heads"][head] = {
                    "candidate_id": candidate_id,
                    "candidate": candidate_metrics,
                    "baseline": baseline_metrics,
                    "delta": {
                        key: candidate_metrics[key] - baseline_metrics[key]
                        for key in ("auprc", "auroc", "brier", "ece")
                    },
                }
            except Exception as exc:  # LODO failures are evidence, never a silent fallback.
                source_result["failures"].append(
                    {"head": head, "type": type(exc).__name__, "message": str(exc)}
                )
        source_result["signature_blocked"] = any(
            value["delta"]["auprc"] < -0.03 and value["delta"]["auroc"] < -0.03
            for value in source_result["heads"].values()
        )
        result["sources"][str(held_source)] = source_result
    result["status"] = (
        "pass" if all(not value["failures"] for value in result["sources"].values()) else "partial"
    )
    write_json(root / EVALUATION_REPORT_RELATIVE / "lodo_report.json", result, overwrite=True)
    return result


__all__ = [
    "EVALUATION_REPORT_RELATIVE",
    "repair_r10_transition_reporting",
    "run_r10_lodo",
    "run_r10_evaluation",
]
