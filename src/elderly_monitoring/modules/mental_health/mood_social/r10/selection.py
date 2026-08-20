"""Inner-only candidate and calibration selection for R10."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r7.modeling import binary_metrics
from elderly_monitoring.modules.mental_health.mood_social.r9.modeling import inner_folds_for_outer

from .baseline import load_r10_frames
from .contract import (
    DEFAULT_DATA_RELATIVE,
    DEFAULT_REPORT_RELATIVE,
    R10_PROTOCOL_VERSION,
    repository_root,
    sha256_file,
    write_json,
)
from .modeling import (
    HEAD_TARGETS,
    _crossfit_calibration,
    candidate_inner_predictions,
    participant_weight,
    select_calibration,
)


SELECTION_REPORT_RELATIVE = DEFAULT_REPORT_RELATIVE / "selection"


def _safe_metadata(value: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, item in value.items():
        if key == "base_probability":
            output["base_oof_crossfit"] = True
        elif key == "residual_parameters":
            output[key] = {
                "intercept": float(item["intercept"]),
                "slope": float(item["slope"]),
                "history_oof_crossfit": True,
                "sleep_oof_crossfit": True,
            }
        else:
            output[key] = item
    return output


def run_r10_selection(
    *, repository_root_value: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    protocol_path = root / DEFAULT_DATA_RELATIVE / "protocol/r10_protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "paired-baseline-sealed-candidate-outer-unopened":
        raise ValueError("R10 selection requires sealed baseline and unopened outer results")
    result: dict[str, Any] = {
        "protocol_version": R10_PROTOCOL_VERSION,
        "status": "locked",
        "selection_scope": "outer-train-inner-oof-only",
        "candidate_outer_opened": False,
        "nodes": {},
    }
    ledger: list[dict[str, Any]] = []
    for node, frame in load_r10_frames(root).items():
        node_lock: dict[str, Any] = {}
        for repeat, repeat_frame in frame.groupby("repeat", sort=True):
            repeat_lock: dict[str, Any] = {}
            for outer_fold in range(5):
                train = repeat_frame.loc[repeat_frame["outer_fold"].ne(outer_fold)].copy()
                folds = inner_folds_for_outer(repeat_frame, outer_fold).loc[train.index].astype(int)
                head_lock: dict[str, Any] = {}
                for head, target in HEAD_TARGETS.items():
                    probability, metadata = candidate_inner_predictions(
                        node,
                        train,
                        folds,
                        head,
                        20260821 + int(repeat) * 1000 + outer_fold * 100,
                    )
                    ranked: list[tuple[float, float, str, dict[str, Any]]] = []
                    for candidate, raw in probability.items():
                        forced = metadata[candidate].get("forced_calibration")
                        if forced:
                            method = str(forced)
                            work = train.copy()
                            work["_selection_target"] = work[target].astype(int)
                            calibrated = _crossfit_calibration(work, raw, folds, method)
                            calibration_metrics = {
                                method: binary_metrics(
                                    train[target], calibrated, weight=participant_weight(train)
                                )
                            }
                        else:
                            method, calibration_metrics = select_calibration(
                                train, raw, folds, target
                            )
                        raw_metrics = binary_metrics(
                            train[target], raw, weight=participant_weight(train)
                        )
                        selected_metrics = calibration_metrics[method]
                        score = float(
                            0.8 * raw_metrics["normalized_ap"]
                            + 0.2 * selected_metrics["normalized_ap"]
                        )
                        row = {
                            "node": node,
                            "repeat": int(repeat),
                            "outer_fold": outer_fold,
                            "head": head,
                            "candidate_id": candidate,
                            "calibration": method,
                            "raw_metrics": raw_metrics,
                            "calibrated_metrics": selected_metrics,
                            "calibration_candidates": calibration_metrics,
                            "selection_score": score,
                            "metadata": _safe_metadata(metadata[candidate]),
                            "selection_data": "outer-train-inner-oof-only",
                        }
                        ledger.append(row)
                        ranked.append((score, -selected_metrics["brier"], candidate, row))
                    ranked.sort(reverse=True)
                    winner = ranked[0][3]
                    head_lock[head] = {
                        "candidate_id": winner["candidate_id"],
                        "calibration": winner["calibration"],
                        "metadata": winner["metadata"],
                        "inner_metrics": winner["calibrated_metrics"],
                        "candidate_count": len(ranked),
                        "runner_up": ranked[1][2],
                    }
                repeat_lock[str(outer_fold)] = head_lock
            node_lock[str(repeat)] = repeat_lock
        result["nodes"][node] = node_lock
    write_json(root / SELECTION_REPORT_RELATIVE / "candidate_ledger.json", ledger, overwrite=overwrite)
    lock_path = root / SELECTION_REPORT_RELATIVE / "selection_lock.json"
    write_json(lock_path, result, overwrite=overwrite)
    protocol["selection_locked"] = True
    protocol["selection_lock_sha256"] = sha256_file(lock_path)
    protocol["status"] = "selection-locked-candidate-outer-unopened"
    protocol["tasks_completed"] = list(protocol["tasks_completed"]) + [
        "OPT-V333-R10-003",
        "OPT-V333-R10-004",
    ]
    write_json(protocol_path, protocol, overwrite=True)
    return result


__all__ = ["SELECTION_REPORT_RELATIVE", "run_r10_selection"]
