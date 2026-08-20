"""Inner-only finite candidate selection for the sealed R9 protocol."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.selection import CandidateSpec
from elderly_monitoring.modules.mental_health.mood_social.r7.phq_history_expert import HISTORY_FEATURES

from .baseline import load_r9_frames
from .contract import (
    ACTIVITY_FEATURES,
    DEFAULT_DATA_RELATIVE,
    DEFAULT_REPORT_RELATIVE,
    R9_PROTOCOL_VERSION,
    SLEEP_FEATURES,
    repository_root,
    sha256_file,
    write_json,
)
from .modeling import HEAD_TARGETS, candidate_score, fit_inner_oof, inner_folds_for_outer


SELECTION_REPORT_RELATIVE = DEFAULT_REPORT_RELATIVE / "selection"
PRIMARY_HISTORY_MIN_DAYS = 14.0
PRIMARY_HISTORY_MAX_DAYS = 100.0


def _spec(candidate_id: str, family: str, params: dict[str, Any]) -> CandidateSpec:
    return CandidateSpec(candidate_id, family, params, 20260819)  # type: ignore[arg-type]


DOMAIN_CANDIDATES: dict[str, tuple[CandidateSpec, ...]] = {
    "activity": (
        _spec("activity_elastic_c01", "elasticnet", {"C": 0.1, "l1_ratio": 0.2}),
        _spec("activity_elastic_c05", "elasticnet", {"C": 0.5, "l1_ratio": 0.5}),
        _spec("activity_hist_leaf7", "hist_gradient", {"max_leaf_nodes": 7, "l2_regularization": 5.0, "max_iter": 180, "learning_rate": 0.04}),
        _spec("activity_hist_leaf15", "hist_gradient", {"max_leaf_nodes": 15, "l2_regularization": 8.0, "max_iter": 180, "learning_rate": 0.035}),
        _spec("activity_lgb_leaf7", "lightgbm", {"n_estimators": 160, "learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 40, "reg_lambda": 6.0}),
        _spec("activity_r8_fixed_70logistic_30tree", "elasticnet", {"C": 0.1, "l1_ratio": 0.2}),
    ),
    "sleep": (
        _spec("sleep_r7_lgb_leaf7", "lightgbm", {"n_estimators": 160, "learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 40, "reg_lambda": 6.0, "subsample": 0.9, "colsample_bytree": 0.9}),
        _spec("sleep_elastic_c01", "elasticnet", {"C": 0.1, "l1_ratio": 0.2}),
        _spec("sleep_hist_leaf7", "hist_gradient", {"max_leaf_nodes": 7, "l2_regularization": 5.0, "max_iter": 180, "learning_rate": 0.04}),
        _spec("sleep_hist_leaf15", "hist_gradient", {"max_leaf_nodes": 15, "l2_regularization": 8.0, "max_iter": 180, "learning_rate": 0.035}),
        _spec("sleep_lgb_leaf5", "lightgbm", {"n_estimators": 140, "learning_rate": 0.03, "num_leaves": 5, "min_child_samples": 50, "reg_lambda": 8.0}),
        _spec("sleep_lgb_leaf11", "lightgbm", {"n_estimators": 180, "learning_rate": 0.025, "num_leaves": 11, "min_child_samples": 55, "reg_lambda": 10.0}),
    ),
    "phq_history": (
        _spec("history_elastic_c01", "elasticnet", {"C": 0.1, "l1_ratio": 0.2}),
        _spec("history_elastic_c05", "elasticnet", {"C": 0.5, "l1_ratio": 0.5}),
        _spec("history_hist_leaf7", "hist_gradient", {"max_leaf_nodes": 7, "l2_regularization": 8.0, "max_iter": 160, "learning_rate": 0.035}),
        _spec("history_lgb_leaf5", "lightgbm", {"n_estimators": 120, "learning_rate": 0.025, "num_leaves": 5, "min_child_samples": 60, "reg_lambda": 10.0}),
    ),
    "activity_sleep": (
        _spec("activity_sleep_elastic", "elasticnet", {"C": 0.1, "l1_ratio": 0.2}),
        _spec("activity_sleep_hist7", "hist_gradient", {"max_leaf_nodes": 7, "l2_regularization": 6.0, "max_iter": 180, "learning_rate": 0.04}),
        _spec("activity_sleep_lgb5", "lightgbm", {"n_estimators": 140, "learning_rate": 0.03, "num_leaves": 5, "min_child_samples": 40, "reg_lambda": 7.0}),
        _spec("activity_sleep_lgb9", "lightgbm", {"n_estimators": 180, "learning_rate": 0.025, "num_leaves": 9, "min_child_samples": 45, "reg_lambda": 10.0}),
    ),
    "sleep_history": (
        _spec("sleep_history_elastic", "elasticnet", {"C": 0.1, "l1_ratio": 0.2}),
        _spec("sleep_history_hist7", "hist_gradient", {"max_leaf_nodes": 7, "l2_regularization": 8.0, "max_iter": 160, "learning_rate": 0.035}),
        _spec("sleep_history_lgb5", "lightgbm", {"n_estimators": 140, "learning_rate": 0.025, "num_leaves": 5, "min_child_samples": 60, "reg_lambda": 10.0}),
    ),
}


FEATURES: dict[str, tuple[str, ...]] = {
    "activity": tuple(ACTIVITY_FEATURES),
    "sleep": tuple(SLEEP_FEATURES),
    "phq_history": tuple(HISTORY_FEATURES) + ("history.freshness_weight", "history.decayed_last_score"),
    "activity_sleep": tuple(ACTIVITY_FEATURES) + tuple(SLEEP_FEATURES),
    "sleep_history": tuple(SLEEP_FEATURES) + tuple(HISTORY_FEATURES) + ("history.freshness_weight", "history.decayed_last_score"),
}


WEIGHT_MODE = {
    "activity": "participant_equal",
    "sleep": "participant_source_balanced",
    "phq_history": "participant_equal",
    "activity_sleep": "participant_source_balanced",
    "sleep_history": "participant_equal",
}


def load_selection_frames(root: Path) -> dict[str, pd.DataFrame]:
    frames = load_r9_frames(root)
    frames["activity_sleep"] = pd.read_parquet(root / DEFAULT_DATA_RELATIVE / "signatures/activity_sleep.parquet")
    frames["sleep_history"] = pd.read_parquet(root / DEFAULT_DATA_RELATIVE / "signatures/sleep_history.parquet")
    for node in ("phq_history", "sleep_history"):
        frames[node] = frames[node].loc[
            frames[node]["history.age_days"].between(PRIMARY_HISTORY_MIN_DAYS, PRIMARY_HISTORY_MAX_DAYS, inclusive="both")
        ].copy()
    return frames


def _select_node_outer(
    node: str,
    frame: pd.DataFrame,
    outer_fold: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train = frame.loc[frame["outer_fold"].ne(outer_fold)].copy()
    folds = inner_folds_for_outer(frame, outer_fold).loc[train.index].astype(int)
    selected: dict[str, Any] = {}
    ledger: list[dict[str, Any]] = []
    for head, target in HEAD_TARGETS.items():
        ranked: list[tuple[float, str, CandidateSpec, dict[str, float]]] = []
        cached_probability: dict[str, Any] = {}
        for template in DOMAIN_CANDIDATES[node]:
            spec = CandidateSpec(template.candidate_id, template.family, template.params, template.seed + outer_fold)
            if spec.candidate_id == "activity_r8_fixed_70logistic_30tree":
                logistic = next(item for item in DOMAIN_CANDIDATES["activity"] if item.candidate_id == "activity_elastic_c01")
                tree = next(item for item in DOMAIN_CANDIDATES["activity"] if item.candidate_id == "activity_hist_leaf7")
                for base in (logistic, tree):
                    if base.candidate_id not in cached_probability:
                        current = CandidateSpec(base.candidate_id, base.family, base.params, base.seed + outer_fold)
                        cached_probability[base.candidate_id] = fit_inner_oof(train, FEATURES[node], target, folds, current, weight_mode=WEIGHT_MODE[node])
                probability = (
                    0.7 * cached_probability[logistic.candidate_id]
                    + 0.3 * cached_probability[tree.candidate_id]
                    if head == "ge5"
                    else cached_probability[logistic.candidate_id]
                )
            else:
                if spec.candidate_id not in cached_probability:
                    cached_probability[spec.candidate_id] = fit_inner_oof(
                        train,
                        FEATURES[node],
                        target,
                        folds,
                        spec,
                        weight_mode=WEIGHT_MODE[node],
                    )
                probability = cached_probability[spec.candidate_id]
            score, metrics = candidate_score(train, target, probability)
            ledger.append({
                "node": node,
                "outer_fold": outer_fold,
                "head": head,
                "candidate": spec.to_dict(),
                "metrics": metrics,
                "selection_data": "outer-train-inner-oof-only",
                "candidate_outer_metrics_opened": False,
            })
            ranked.append((score, spec.candidate_id, spec, metrics))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        winner = ranked[0]
        selected[head] = {
            "candidate": winner[2].to_dict(),
            "inner_metrics": winner[3],
            "runner_up_candidate_id": ranked[1][2].candidate_id,
            "candidate_count": len(ranked),
        }
    return selected, ledger


def run_r9_selection(
    *, repository_root_value: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    protocol_path = root / DEFAULT_DATA_RELATIVE / "protocol/r9_protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report_path = root / SELECTION_REPORT_RELATIVE / "selection_lock.json"
    if protocol.get("selection_locked") is True:
        expected = protocol.get("selection_lock_sha256")
        if report_path.is_file() and sha256_file(report_path) == expected:
            return json.loads(report_path.read_text(encoding="utf-8"))
        raise ValueError("R9 selection is marked locked but its artifact is invalid")
    if protocol.get("status") != "pass" or protocol.get("candidate_outer_metrics_opened") is not False:
        raise ValueError("R9 selection requires sealed baselines and unopened candidate outer metrics")
    frames = load_selection_frames(root)
    result: dict[str, Any] = {
        "status": "locked",
        "protocol_version": R9_PROTOCOL_VERSION,
        "candidate_outer_metrics_opened": False,
        "selection_scope": "outer-train-inner-oof-only",
        "history_primary_interval_days": [14, 100],
        "nodes": {},
    }
    ledger: list[dict[str, Any]] = []
    for node in ("activity", "sleep", "phq_history", "activity_sleep", "sleep_history"):
        node_selection: dict[str, Any] = {}
        for outer_fold in range(5):
            selected, rows = _select_node_outer(node, frames[node], outer_fold)
            node_selection[str(outer_fold)] = selected
            ledger.extend(rows)
        result["nodes"][node] = {
            "rows": int(len(frames[node])),
            "participants": int(frames[node]["global_participant_id"].nunique()),
            "features": list(FEATURES[node]),
            "weight_mode": WEIGHT_MODE[node],
            "candidate_budget": len(DOMAIN_CANDIDATES[node]),
            "outer_fold_selection": node_selection,
            "modal_candidates": {
                head: Counter(
                    node_selection[str(fold)][head]["candidate"]["candidate_id"]
                    for fold in range(5)
                ).most_common()
                for head in HEAD_TARGETS
            },
        }
    write_json(root / SELECTION_REPORT_RELATIVE / "candidate_ledger.json", ledger, overwrite=overwrite)
    write_json(report_path, result, overwrite=overwrite)
    protocol["selection_locked"] = True
    protocol["selection_lock"] = str(report_path.relative_to(root)).replace("\\", "/")
    protocol["selection_lock_sha256"] = sha256_file(report_path)
    protocol["status"] = "selection-locked-candidate-outer-unopened"
    protocol["tasks_completed"] = list(protocol.get("tasks_completed", [])) + [
        "OPT-V333-R9-002",
        "OPT-V333-R9-003",
        "OPT-V333-R9-004",
    ]
    write_json(protocol_path, protocol, overwrite=True)
    return result


def repair_pre_outer_selection_lock(
    *, repository_root_value: Path | None = None
) -> dict[str, Any]:
    """Repair a selection implementation mismatch before any candidate outer opening."""

    root = repository_root(repository_root_value)
    protocol_path = root / DEFAULT_DATA_RELATIVE / "protocol/r9_protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("candidate_outer_metrics_opened") is not False:
        raise ValueError("cannot repair R9 selection after candidate outer opening")
    lock_path = root / SELECTION_REPORT_RELATIVE / "selection_lock.json"
    if not lock_path.is_file() or sha256_file(lock_path) != protocol.get("selection_lock_sha256"):
        raise ValueError("existing R9 selection lock is not valid")
    protocol["selection_locked"] = False
    protocol["status"] = "pass"
    protocol.pop("selection_lock", None)
    protocol.pop("selection_lock_sha256", None)
    protocol["tasks_completed"] = [
        task for task in protocol.get("tasks_completed", [])
        if task not in {"OPT-V333-R9-002", "OPT-V333-R9-003", "OPT-V333-R9-004"}
    ]
    write_json(protocol_path, protocol, overwrite=True)
    return run_r9_selection(repository_root_value=root, overwrite=True)


__all__ = [
    "DOMAIN_CANDIDATES",
    "FEATURES",
    "PRIMARY_HISTORY_MAX_DAYS",
    "PRIMARY_HISTORY_MIN_DAYS",
    "SELECTION_REPORT_RELATIVE",
    "WEIGHT_MODE",
    "load_selection_frames",
    "repair_pre_outer_selection_lock",
    "run_r9_selection",
]
