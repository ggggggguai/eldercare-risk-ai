"""One-time locked outer evaluation for R9 independent and local joint experts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from elderly_monitoring.modules.mental_health.mood_social.r4.selection import CandidateSpec
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import fit_model, predict_model
from elderly_monitoring.modules.mental_health.mood_social.r7.modeling import binary_metrics

from .baseline import BASELINE_REPORT_RELATIVE, BASELINE_SPECS
from .contract import (
    DEFAULT_DATA_RELATIVE,
    DEFAULT_REPORT_RELATIVE,
    R9_PROTOCOL_VERSION,
    repository_root,
    sha256_file,
    write_json,
)
from .evidence_graph import (
    enumerate_signature_routes,
    graph_contract_sha256,
    validate_signature_routes,
)
from .modeling import HEAD_TARGETS, fit_outer_fixed, inner_folds_for_outer, project_heads
from .rule_fusion import (
    enumerate_truth_table,
    rule_contract_sha256,
    validate_truth_table,
)
from .selection import FEATURES, SELECTION_REPORT_RELATIVE, WEIGHT_MODE, load_selection_frames


EVALUATION_REPORT_RELATIVE = DEFAULT_REPORT_RELATIVE / "evaluation"
BOOTSTRAP_SEED = 20260820
BOOTSTRAP_RESAMPLES = 2000
MAIN_HEAD = "ge5"
BASELINE_FILE = {
    "activity": "activity/baseline_outer_oof.parquet",
    "sleep": "sleep/baseline_outer_oof.parquet",
    "phq_history": "phq_history/baseline_outer_oof.parquet",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _participant_weight(frame: pd.DataFrame) -> np.ndarray:
    count = frame.groupby("global_participant_id")["global_participant_id"].transform("size").to_numpy(float)
    weight = 1.0 / count
    return weight * len(weight) / weight.sum()


def _metrics(frame: pd.DataFrame, target: str, probability: str) -> dict[str, float]:
    return binary_metrics(frame[target].astype(int), frame[probability].astype(float), weight=_participant_weight(frame))


def _align_probability(frame: pd.DataFrame, source: pd.DataFrame, head: str) -> np.ndarray:
    aligned = frame[["r4_row_id"]].merge(
        source[["r4_row_id", f"probability_{head}"]],
        on="r4_row_id",
        how="left",
        validate="one_to_one",
    )
    if aligned[f"probability_{head}"].isna().any():
        raise ValueError("R9 paired baseline alignment is incomplete")
    return aligned[f"probability_{head}"].to_numpy(float)


def _seal_joint_baseline_contract(root: Path, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    path = root / EVALUATION_REPORT_RELATIVE / "joint_baseline_contract.json"
    if path.is_file():
        return _load_json(path)
    protocol = _load_json(root / DEFAULT_DATA_RELATIVE / "protocol/r9_protocol_manifest.json")
    if protocol.get("candidate_outer_metrics_opened") is not False:
        raise ValueError("joint baseline contract must be sealed before candidate outer opening")
    baseline = {
        node: pd.read_parquet(root / BASELINE_REPORT_RELATIVE / relative)
        for node, relative in BASELINE_FILE.items()
    }
    result: dict[str, Any] = {
        "status": "sealed-before-candidate-outer",
        "selection_uses_candidate_outer": False,
        "joint_nodes": {},
    }
    for joint, candidates in {
        "activity_sleep": ("activity", "sleep"),
        "sleep_history": ("sleep", "phq_history"),
    }.items():
        node: dict[str, Any] = {}
        for head, target in HEAD_TARGETS.items():
            rows: list[dict[str, Any]] = []
            for candidate in candidates:
                work = frames[joint].copy()
                work["_probability"] = _align_probability(work, baseline[candidate], head)
                rows.append({"node": candidate, "metrics": _metrics(work, target, "_probability")})
            rows.sort(key=lambda row: (row["metrics"]["auprc"], row["node"]), reverse=True)
            node[head] = {"selected_best_single": rows[0], "all_single_baselines": rows}
        result["joint_nodes"][joint] = node
    write_json(path, result)
    result["sha256"] = sha256_file(path)
    return result


def _open_candidate_outer(root: Path, selection_sha: str, joint_baseline_sha: str) -> None:
    protocol_path = root / DEFAULT_DATA_RELATIVE / "protocol/r9_protocol_manifest.json"
    protocol = _load_json(protocol_path)
    if protocol.get("candidate_outer_metrics_opened") is not False:
        raise ValueError("R9 candidate outer has already been opened")
    if protocol.get("selection_lock_sha256") != selection_sha:
        raise ValueError("R9 selection lock changed before outer opening")
    event = {
        "protocol_version": R9_PROTOCOL_VERSION,
        "event": "candidate-outer-opened-once",
        "selection_lock_sha256": selection_sha,
        "joint_baseline_contract_sha256": joint_baseline_sha,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "candidate_outer_metrics_burn_after_event": True,
    }
    event_path = root / EVALUATION_REPORT_RELATIVE / "candidate_outer_open_event.json"
    write_json(event_path, event)
    protocol["candidate_outer_metrics_opened"] = True
    protocol["candidate_outer_open_event"] = str(event_path.relative_to(root)).replace("\\", "/")
    protocol["candidate_outer_open_event_sha256"] = sha256_file(event_path)
    protocol["status"] = "candidate-outer-opened-burned"
    write_json(protocol_path, protocol, overwrite=True)


def _fit_selected_recipe(
    node: str,
    head: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    folds: pd.Series,
    selected: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    payload = selected["candidate"]
    spec = CandidateSpec(payload["candidate_id"], payload["family"], payload["params"], payload["seed"])
    if spec.candidate_id != "activity_r8_fixed_70logistic_30tree":
        outer, _inner, _calibrator, audit = fit_outer_fixed(
            train,
            test,
            FEATURES[node],
            HEAD_TARGETS[head],
            folds,
            spec,
            weight_mode=WEIGHT_MODE[node],
        )
        return outer, audit
    logistic = CandidateSpec("activity_elastic_c01", "elasticnet", {"C": 0.1, "l1_ratio": 0.2}, spec.seed)
    logistic_outer, _a, _b, logistic_audit = fit_outer_fixed(train, test, FEATURES[node], HEAD_TARGETS[head], folds, logistic, weight_mode=WEIGHT_MODE[node])
    if head == "ge10":
        return logistic_outer, {"recipe": "R8 fixed dual-head; ge10 logistic only", "logistic": logistic_audit}
    tree = CandidateSpec("activity_hist_leaf7", "hist_gradient", {"max_leaf_nodes": 7, "l2_regularization": 5.0, "max_iter": 180, "learning_rate": 0.04}, spec.seed)
    tree_outer, _c, _d, tree_audit = fit_outer_fixed(train, test, FEATURES[node], HEAD_TARGETS[head], folds, tree, weight_mode=WEIGHT_MODE[node])
    return 0.7 * logistic_outer + 0.3 * tree_outer, {"recipe": "R8 fixed 70% logistic + 30% shallow tree", "logistic": logistic_audit, "tree": tree_audit}


def _candidate_oof(node: str, frame: pd.DataFrame, lock: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    keep = ["r4_row_id", "dataset_id", "global_participant_id", "outer_fold", "phq9_ge5_target", "phq9_ge10_target"]
    output = frame[keep].copy()
    audit: dict[str, Any] = {}
    for head in HEAD_TARGETS:
        probability = pd.Series(np.nan, index=frame.index, dtype=float)
        audit[head] = {}
        for outer_fold in range(5):
            train = frame.loc[frame["outer_fold"].ne(outer_fold)].copy()
            test = frame.loc[frame["outer_fold"].eq(outer_fold)].copy()
            folds = inner_folds_for_outer(frame, outer_fold).loc[train.index].astype(int)
            selected = lock["nodes"][node]["outer_fold_selection"][str(outer_fold)][head]
            predicted, fold_audit = _fit_selected_recipe(node, head, train, test, folds, selected)
            probability.loc[test.index] = predicted
            audit[head][str(outer_fold)] = {"selection": selected, "fit": fold_audit}
        if probability.isna().any():
            raise ValueError(f"R9 {node}/{head} outer OOF is incomplete")
        output[f"candidate_probability_{head}"] = probability.loc[frame.index].to_numpy(float)
    output["candidate_probability_ge5"], output["candidate_probability_ge10"] = project_heads(output["candidate_probability_ge5"], output["candidate_probability_ge10"])
    return output, audit


def _baseline_for_node(
    root: Path,
    node: str,
    frame: pd.DataFrame,
    joint_contract: dict[str, Any],
) -> pd.DataFrame:
    output = frame[["r4_row_id"]].copy()
    for head in HEAD_TARGETS:
        baseline_node = node
        if node in joint_contract["joint_nodes"]:
            baseline_node = joint_contract["joint_nodes"][node][head]["selected_best_single"]["node"]
        source = pd.read_parquet(root / BASELINE_REPORT_RELATIVE / BASELINE_FILE[baseline_node])
        output[f"baseline_probability_{head}"] = _align_probability(frame, source, head)
    output["baseline_probability_ge5"], output["baseline_probability_ge10"] = project_heads(output["baseline_probability_ge5"], output["baseline_probability_ge10"])
    return output


def _bootstrap(frame: pd.DataFrame, head: str) -> dict[str, Any]:
    participants = frame["global_participant_id"].astype(str).drop_duplicates().to_numpy()
    grouped = {key: group.index.to_numpy() for key, group in frame.groupby(frame["global_participant_id"].astype(str), sort=False)}
    rng = np.random.default_rng(BOOTSTRAP_SEED + (5 if head == "ge5" else 10))
    target_all = frame[HEAD_TARGETS[head]].to_numpy(int)
    candidate_all = frame[f"candidate_probability_{head}"].to_numpy(float)
    baseline_all = frame[f"baseline_probability_{head}"].to_numpy(float)
    deltas: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled = rng.choice(participants, len(participants), replace=True)
        indices: list[int] = []
        weights: list[float] = []
        for participant in sampled:
            current = grouped[participant]
            indices.extend(current.tolist())
            weights.extend([1.0 / len(current)] * len(current))
        idx = np.asarray(indices, dtype=int)
        y = target_all[idx]
        if y.min() == y.max():
            continue
        weight = np.asarray(weights, dtype=float)
        deltas.append(float(average_precision_score(y, candidate_all[idx], sample_weight=weight) - average_precision_score(y, baseline_all[idx], sample_weight=weight)))
    value = np.asarray(deltas, dtype=float)
    return {"unit": "participant", "seed": BOOTSTRAP_SEED + (5 if head == "ge5" else 10), "requested_repetitions": BOOTSTRAP_RESAMPLES, "valid_repetitions": int(len(value)), "mean": float(value.mean()), "lower_95": float(np.quantile(value, 0.025)), "upper_95": float(np.quantile(value, 0.975))}


def _paired_report(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"heads": {}}
    for head, target in HEAD_TARGETS.items():
        candidate = _metrics(frame, target, f"candidate_probability_{head}")
        baseline = _metrics(frame, target, f"baseline_probability_{head}")
        delta = {key: candidate[key] - baseline[key] for key in ("auprc", "normalized_ap", "auroc", "brier", "ece")}
        fold_delta = {}
        source_delta = {}
        for fold, part in frame.groupby("outer_fold", sort=True):
            fold_delta[str(int(fold))] = _metrics(part, target, f"candidate_probability_{head}")["auprc"] - _metrics(part, target, f"baseline_probability_{head}")["auprc"]
        for source, part in frame.groupby("dataset_id", sort=True):
            if part[target].nunique() == 2:
                source_delta[str(source)] = {
                    "delta_auprc": _metrics(part, target, f"candidate_probability_{head}")["auprc"] - _metrics(part, target, f"baseline_probability_{head}")["auprc"],
                    "delta_auroc": _metrics(part, target, f"candidate_probability_{head}")["auroc"] - _metrics(part, target, f"baseline_probability_{head}")["auroc"],
                    "rows": int(len(part)),
                    "participants": int(part["global_participant_id"].nunique()),
                }
        result["heads"][head] = {
            "candidate": candidate,
            "baseline": baseline,
            "delta": delta,
            "fold_delta_auprc": fold_delta,
            "nonnegative_folds": sum(value >= 0 for value in fold_delta.values()),
            "source_slices": source_delta,
            "paired_bootstrap_delta_auprc": _bootstrap(frame.reset_index(drop=True), head),
        }
    result["probability_order_violations"] = int((frame["candidate_probability_ge10"] > frame["candidate_probability_ge5"]).sum())
    return result


def _promotion(node: str, report: dict[str, Any]) -> dict[str, Any]:
    main = report["heads"][MAIN_HEAD]
    other = report["heads"]["ge10"]
    stability = main["nonnegative_folds"] >= 3 or main["paired_bootstrap_delta_auprc"]["lower_95"] >= -0.010
    standard = (
        main["delta"]["auprc"] >= 0.002
        and other["delta"]["auprc"] >= -0.005
        and stability
        and main["delta"]["auroc"] >= -0.015
        and main["delta"]["brier"] <= 0.010
        and main["delta"]["ece"] <= 0.020
        and report["probability_order_violations"] == 0
    )
    history_alternative = False
    if node in {"phq_history", "sleep_history"}:
        history_alternative = (
            main["delta"]["auprc"] >= -0.003
            and other["delta"]["auprc"] >= -0.005
            and stability
            and (main["delta"]["brier"] < 0 or main["delta"]["ece"] < 0)
            and report["probability_order_violations"] == 0
        )
    return {
        "promoted": bool(standard or history_alternative),
        "route": "standard" if standard else "history-calibration-alternative" if history_alternative else "fallback",
        "main_head": MAIN_HEAD,
        "standard_gate_pass": bool(standard),
        "history_alternative_pass": bool(history_alternative),
        "stability_pass": bool(stability),
    }


def _role_audit() -> dict[str, Any]:
    return {
        "social": {"role": "anomaly-only", "supervised_probability_promoted": False, "reason": "DepreST-CAT confirmation is exposed, non-elder and insufficiently discriminative"},
        "facial_affect": {"role": "bounded-support-only", "phq_probability_available": False, "real_s10_sessions": 0, "paired_phq_video_labels": 0},
        "profile": {"role": "background-only", "dynamic_vote": False},
        "physiology": {"role": "support-only", "dynamic_vote": False, "phq_probability_available": False},
        "complete_seven_domain": {"training_status": "data-blocked", "rows": 0, "complete_r9_auprc_reported": False},
    }


def _modal_spec(lock: dict[str, Any], node: str, head: str) -> CandidateSpec:
    selections = [
        lock["nodes"][node]["outer_fold_selection"][str(fold)][head]["candidate"]
        for fold in range(5)
    ]
    candidate_id = Counter(item["candidate_id"] for item in selections).most_common(1)[0][0]
    payload = next(item for item in selections if item["candidate_id"] == candidate_id)
    return CandidateSpec(payload["candidate_id"], payload["family"], payload["params"], payload["seed"])


def _direct_probability(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
    spec: CandidateSpec,
    *,
    head: str,
) -> np.ndarray:
    train = train.copy()
    train["binary_target"] = train[target].astype(int)
    if spec.candidate_id != "activity_r8_fixed_70logistic_30tree":
        return predict_model(fit_model(train, features, spec, participant_equal=True), test, features)
    logistic = CandidateSpec("activity_elastic_c01", "elasticnet", {"C": 0.1, "l1_ratio": 0.2}, spec.seed)
    first = predict_model(fit_model(train, features, logistic, participant_equal=True), test, features)
    if head == "ge10":
        return first
    tree = CandidateSpec("activity_hist_leaf7", "hist_gradient", {"max_leaf_nodes": 7, "l2_regularization": 5.0, "max_iter": 180, "learning_rate": 0.04}, spec.seed)
    second = predict_model(fit_model(train, features, tree, participant_equal=True), test, features)
    return 0.7 * first + 0.3 * second


def _lodo(
    frames: dict[str, pd.DataFrame],
    lock: dict[str, Any],
    joint_contract: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {"semantics": "locked-modal-recipe source-held-out ranking diagnostic; no held-out-source calibration", "nodes": {}, "route_blocks": []}
    for node in ("activity", "sleep", "activity_sleep"):
        frame = frames[node]
        node_result: dict[str, Any] = {}
        for source in sorted(frame["dataset_id"].astype(str).unique()):
            train = frame.loc[frame["dataset_id"].astype(str).ne(source)].copy()
            test = frame.loc[frame["dataset_id"].astype(str).eq(source)].copy()
            source_result: dict[str, Any] = {"rows": int(len(test)), "participants": int(test["global_participant_id"].nunique()), "heads": {}}
            for head, target in HEAD_TARGETS.items():
                if train[target].nunique() < 2 or test[target].nunique() < 2:
                    source_result["heads"][head] = {"status": "not-estimable-single-class"}
                    continue
                candidate_spec = _modal_spec(lock, node, head)
                candidate = _direct_probability(train, test, FEATURES[node], target, candidate_spec, head=head)
                baseline_node = node
                if node == "activity_sleep":
                    baseline_node = joint_contract["joint_nodes"][node][head]["selected_best_single"]["node"]
                baseline_spec = BASELINE_SPECS[baseline_node]
                baseline = _direct_probability(train, test, FEATURES[baseline_node], target, baseline_spec, head=head)
                work = test[["global_participant_id", target]].copy()
                work["candidate"] = candidate
                work["baseline"] = baseline
                candidate_metrics = _metrics(work, target, "candidate")
                baseline_metrics = _metrics(work, target, "baseline")
                delta_ap = candidate_metrics["auprc"] - baseline_metrics["auprc"]
                delta_auc = candidate_metrics["auroc"] - baseline_metrics["auroc"]
                severe = delta_ap < -0.03 and delta_auc < -0.03
                source_result["heads"][head] = {
                    "status": "success",
                    "candidate": candidate_metrics,
                    "baseline": baseline_metrics,
                    "delta_auprc": delta_ap,
                    "delta_auroc": delta_auc,
                    "route_blocked": severe,
                }
                if severe:
                    result["route_blocks"].append({"node": node, "dataset_id": source, "head": head, "reason": "LODO delta AUPRC and AUROC both below -0.03"})
            node_result[source] = source_result
        result["nodes"][node] = node_result
    result["nodes"]["phq_history"] = {"status": "not-applicable-single-longitudinal-source"}
    result["nodes"]["sleep_history"] = {"status": "not-applicable-single-longitudinal-source"}
    return result


def run_r9_evaluation(
    *, repository_root_value: Path | None = None
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    final_path = root / EVALUATION_REPORT_RELATIVE / "evaluation_report.json"
    protocol_path = root / DEFAULT_DATA_RELATIVE / "protocol/r9_protocol_manifest.json"
    protocol = _load_json(protocol_path)
    if protocol.get("status") == "evaluation-complete" and final_path.is_file() and protocol.get("evaluation_report_sha256") == sha256_file(final_path):
        return _load_json(final_path)
    if protocol.get("candidate_outer_metrics_opened") is not False:
        raise ValueError("R9 candidate outer is burned; automatic rerun is forbidden")
    lock_path = root / SELECTION_REPORT_RELATIVE / "selection_lock.json"
    lock = _load_json(lock_path)
    lock_sha = sha256_file(lock_path)
    if protocol.get("selection_lock_sha256") != lock_sha:
        raise ValueError("R9 selection lock integrity failure")
    frames = load_selection_frames(root)
    joint_contract = _seal_joint_baseline_contract(root, frames)
    joint_path = root / EVALUATION_REPORT_RELATIVE / "joint_baseline_contract.json"
    _open_candidate_outer(root, lock_sha, sha256_file(joint_path))
    result: dict[str, Any] = {
        "status": "pass",
        "protocol_version": R9_PROTOCOL_VERSION,
        "evidence_status": "adaptive-development/reused-benchmark/locked-procedure-estimate",
        "candidate_outer_opened_once": True,
        "selection_lock_sha256": lock_sha,
        "joint_baseline_contract_sha256": sha256_file(joint_path),
        "main_head": MAIN_HEAD,
        "nodes": {},
        "complete_r9_auprc_reported": False,
    }
    for node in ("activity", "sleep", "phq_history", "activity_sleep", "sleep_history"):
        candidate, audit = _candidate_oof(node, frames[node], lock)
        baseline = _baseline_for_node(root, node, frames[node], joint_contract)
        paired = candidate.merge(baseline, on="r4_row_id", validate="one_to_one")
        output_path = root / EVALUATION_REPORT_RELATIVE / node / "candidate_outer_oof.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        paired.to_parquet(output_path, index=False)
        write_json(root / EVALUATION_REPORT_RELATIVE / node / "fold_audit.json", audit)
        report = _paired_report(paired)
        report.update({
            "rows": int(len(paired)),
            "participants": int(paired["global_participant_id"].nunique()),
            "features": list(FEATURES[node]),
            "oof_sha256": sha256_file(output_path),
            "baseline_semantics": "same-fold R7/R8 expert" if node not in {"activity_sleep", "sleep_history"} else "best sealed same-signature single expert",
        })
        report["promotion"] = _promotion(node, report)
        write_json(root / EVALUATION_REPORT_RELATIVE / node / "metrics.json", report)
        result["nodes"][node] = report
    role_audit = _role_audit()
    lodo = _lodo(frames, lock, joint_contract)
    graph_rows = enumerate_signature_routes()
    graph_audit = validate_signature_routes()
    truth_rows = enumerate_truth_table()
    truth_audit = validate_truth_table(truth_rows)
    if graph_audit["status"] != "pass" or truth_audit["status"] != "pass":
        raise ValueError("R9 evidence graph or rule truth table failed")
    write_json(root / EVALUATION_REPORT_RELATIVE / "role_audit.json", role_audit)
    write_json(root / EVALUATION_REPORT_RELATIVE / "lodo_report.json", lodo)
    write_json(root / EVALUATION_REPORT_RELATIVE / "evidence_graph_routes.json", graph_rows)
    write_json(root / EVALUATION_REPORT_RELATIVE / "evidence_graph_audit.json", graph_audit)
    write_json(root / EVALUATION_REPORT_RELATIVE / "rule_truth_table.json", truth_rows)
    write_json(root / EVALUATION_REPORT_RELATIVE / "rule_truth_table_audit.json", truth_audit)
    result["role_audit"] = role_audit
    result["lodo"] = lodo
    result["evidence_graph"] = {"contract_sha256": graph_contract_sha256(), "audit": graph_audit}
    result["rule_fusion_v3"] = {"contract_sha256": rule_contract_sha256(), "audit": truth_audit}
    result["promoted_nodes"] = [node for node, report in result["nodes"].items() if report["promotion"]["promoted"]]
    result["release_candidate_allowed"] = bool(result["promoted_nodes"])
    write_json(final_path, result)
    protocol = _load_json(protocol_path)
    protocol["status"] = "evaluation-complete"
    protocol["evaluation_report"] = str(final_path.relative_to(root)).replace("\\", "/")
    protocol["evaluation_report_sha256"] = sha256_file(final_path)
    protocol["promoted_nodes"] = result["promoted_nodes"]
    protocol["tasks_completed"] = list(protocol.get("tasks_completed", [])) + [
        "OPT-V333-R9-005", "OPT-V333-R9-006", "OPT-V333-R9-007", "OPT-V333-R9-008"
    ]
    write_json(protocol_path, protocol, overwrite=True)
    return result


__all__ = ["BOOTSTRAP_RESAMPLES", "EVALUATION_REPORT_RELATIVE", "run_r9_evaluation"]
