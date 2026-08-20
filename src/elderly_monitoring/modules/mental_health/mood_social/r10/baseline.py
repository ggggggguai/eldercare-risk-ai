"""Same-row, same-fold replay of the frozen R9 local-joint recipes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.selection import CandidateSpec
from elderly_monitoring.modules.mental_health.mood_social.r7.modeling import binary_metrics
from elderly_monitoring.modules.mental_health.mood_social.r9.modeling import (
    fit_outer_fixed,
    inner_folds_for_outer,
    project_heads,
)

from .contract import (
    DEFAULT_DATA_RELATIVE,
    DEFAULT_REPORT_RELATIVE,
    R10_PROTOCOL_VERSION,
    repository_root,
    sha256_file,
    write_json,
)
from .features import ACTIVITY_SLEEP_BASE_FEATURES, SLEEP_HISTORY_FEATURES


BASELINE_REPORT_RELATIVE = DEFAULT_REPORT_RELATIVE / "paired-baseline"
TARGETS = {"ge5": "phq9_ge5_target", "ge10": "phq9_ge10_target"}
R9_RECIPE = CandidateSpec(
    "r9_elastic_replay", "elasticnet", {"C": 0.1, "l1_ratio": 0.2}, 20260819
)


def load_r10_frames(root: Path) -> dict[str, pd.DataFrame]:
    base = root / DEFAULT_DATA_RELATIVE / "signatures"
    return {
        "activity_sleep": pd.read_parquet(base / "activity_sleep_repeated.parquet"),
        "sleep_history": pd.read_parquet(base / "sleep_history_repeated.parquet"),
    }


def _weights(frame: pd.DataFrame) -> np.ndarray:
    count = frame.groupby("global_participant_id")["global_participant_id"].transform("size")
    value = 1.0 / count.to_numpy(float)
    return value * len(value) / value.sum()


def _report(frame: pd.DataFrame) -> dict[str, Any]:
    heads: dict[str, Any] = {}
    for head, target in TARGETS.items():
        probability = frame[f"baseline_probability_{head}"].to_numpy(float)
        heads[head] = {
            "natural": binary_metrics(frame[target].astype(int), probability),
            "participant_equal": binary_metrics(
                frame[target].astype(int), probability, weight=_weights(frame)
            ),
            "by_repeat": {
                str(repeat): binary_metrics(
                    part[target].astype(int), part[f"baseline_probability_{head}"].astype(float),
                    weight=_weights(part),
                )
                for repeat, part in frame.groupby("repeat")
            },
        }
    unique = frame.loc[frame["repeat"].eq(0)]
    return {
        "rows_per_repeat": int(len(unique)),
        "participants": int(unique["global_participant_id"].nunique()),
        "repeats": int(frame["repeat"].nunique()),
        "heads": heads,
    }


def _node_oof(node: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    features = (
        ACTIVITY_SLEEP_BASE_FEATURES if node == "activity_sleep" else SLEEP_HISTORY_FEATURES
    )
    weight_mode = "participant_source_balanced" if node == "activity_sleep" else "participant_equal"
    output: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for repeat, repeat_frame in frame.groupby("repeat", sort=True):
        current = repeat_frame.copy()
        for head, target in TARGETS.items():
            current[f"baseline_probability_{head}"] = np.nan
            for outer_fold in range(5):
                train = repeat_frame.loc[repeat_frame["outer_fold"].ne(outer_fold)].copy()
                test = repeat_frame.loc[repeat_frame["outer_fold"].eq(outer_fold)].copy()
                folds = inner_folds_for_outer(repeat_frame, outer_fold).loc[train.index].astype(int)
                spec = CandidateSpec(
                    R9_RECIPE.candidate_id,
                    R9_RECIPE.family,
                    R9_RECIPE.params,
                    R9_RECIPE.seed + int(repeat) * 100 + outer_fold,
                )
                probability, _, _, audit = fit_outer_fixed(
                    train,
                    test,
                    features,
                    target,
                    folds,
                    spec,
                    weight_mode=weight_mode,
                )
                current.loc[test.index, f"baseline_probability_{head}"] = probability
                audits.append(
                    {
                        "node": node,
                        "repeat": int(repeat),
                        "outer_fold": outer_fold,
                        "head": head,
                        **audit,
                    }
                )
        p5, p10 = project_heads(
            current["baseline_probability_ge5"], current["baseline_probability_ge10"]
        )
        current["baseline_probability_ge5"] = p5
        current["baseline_probability_ge10"] = p10
        output.append(current)
    result = pd.concat(output).sort_values(["repeat", "r4_row_id"], kind="stable")
    if result[["baseline_probability_ge5", "baseline_probability_ge10"]].isna().any().any():
        raise ValueError(f"R10 incomplete baseline OOF: {node}")
    if (result["baseline_probability_ge10"] > result["baseline_probability_ge5"] + 1e-12).any():
        raise ValueError(f"R10 baseline monotonicity violation: {node}")
    return result.reset_index(drop=True), audits


def run_r10_baseline(
    *, repository_root_value: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    protocol_path = root / DEFAULT_DATA_RELATIVE / "protocol/r10_protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "prepared-awaiting-paired-baseline":
        raise ValueError("R10 paired baseline requires a prepared, unopened protocol")
    reports: dict[str, Any] = {}
    frames = load_r10_frames(root)
    for node, frame in frames.items():
        oof, audits = _node_oof(node, frame)
        node_dir = root / BASELINE_REPORT_RELATIVE / node
        output = node_dir / "r9_recipe_repeated_outer_oof.parquet"
        if output.exists() and not overwrite:
            raise FileExistsError(output)
        node_dir.mkdir(parents=True, exist_ok=True)
        oof.to_parquet(output, index=False)
        report = {
            "protocol_version": R10_PROTOCOL_VERSION,
            "node": node,
            "recipe": R9_RECIPE.to_dict(),
            "paired_same_rows_folds_weights": True,
            "metrics": _report(oof),
            "oof_sha256": sha256_file(output),
            "fold_audits": audits,
        }
        write_json(node_dir / "baseline_report.json", report, overwrite=overwrite)
        reports[node] = report
    contract = json.loads(
        (root / DEFAULT_REPORT_RELATIVE / "selection_lock_contract.json").read_text(encoding="utf-8")
    )
    contract["lock_status"] = "baseline-sealed-awaiting-inner-selection"
    contract["baseline_report_sha256"] = {
        node: sha256_file(root / BASELINE_REPORT_RELATIVE / node / "baseline_report.json")
        for node in reports
    }
    write_json(root / DEFAULT_REPORT_RELATIVE / "selection_lock_contract.json", contract, overwrite=True)
    protocol["status"] = "paired-baseline-sealed-candidate-outer-unopened"
    protocol["paired_baseline_sealed"] = True
    protocol["tasks_completed"] = list(protocol["tasks_completed"]) + ["OPT-V333-R10-000-baseline"]
    write_json(protocol_path, protocol, overwrite=True)
    return {"status": "pass", "nodes": {key: value["metrics"] for key, value in reports.items()}}


__all__ = ["BASELINE_REPORT_RELATIVE", "load_r10_frames", "run_r10_baseline"]
