"""R9-000 same-fold replay of frozen R7/R8 expert recipes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.data import (
    load_r4_development_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    CandidateSpec,
)
from elderly_monitoring.modules.mental_health.mood_social.r7.phq_history_expert import (
    HISTORY_FEATURES,
)

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
from .modeling import (
    HEAD_TARGETS,
    fit_outer_fixed,
    inner_folds_for_outer,
    metric_block,
    project_heads,
)


BASELINE_REPORT_RELATIVE = DEFAULT_REPORT_RELATIVE / "paired-baseline"

BASELINE_SPECS = {
    "activity": CandidateSpec(
        "r7_activity_logistic",
        "elasticnet",
        {"C": 0.1, "l1_ratio": 0.2},
        20260819,
    ),
    "sleep": CandidateSpec(
        "r7_sleep_promoted",
        "lightgbm",
        {
            "n_estimators": 160,
            "learning_rate": 0.03,
            "num_leaves": 7,
            "min_child_samples": 40,
            "reg_lambda": 6.0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        20260819,
    ),
    "phq_history": CandidateSpec(
        "r7_history_logistic",
        "elasticnet",
        {"C": 0.1, "l1_ratio": 0.2},
        20260819,
    ),
}


def _available(frame: pd.DataFrame, features: Sequence[str]) -> pd.Series:
    return frame[[f"feature_mask.{name}" for name in features]].fillna(0).max(axis=1).gt(0)


def load_r9_frames(root: Path) -> dict[str, pd.DataFrame]:
    base = load_r4_development_frame(repository_root=root).copy()
    split = pd.read_parquet(root / DEFAULT_DATA_RELATIVE / "protocol/r9_split.parquet")
    base = base.drop(
        columns=["outer_fold", "inner_validation_fold_by_outer_fold"],
        errors="ignore",
    ).merge(
        split[["r4_row_id", "outer_fold", "inner_validation_fold_by_outer_fold"]],
        on="r4_row_id",
        validate="one_to_one",
    )
    base = base.rename(
        columns={
            "phq9_ge5_r3_target": "phq9_ge5_target",
            "phq9_ge10_r3_target": "phq9_ge10_target",
        }
    )
    history = pd.read_parquet(
        root / DEFAULT_DATA_RELATIVE / "signatures/sleep_history.parquet"
    )
    return {
        "activity": base.loc[_available(base, ACTIVITY_FEATURES)].copy(),
        "sleep": base.loc[_available(base, SLEEP_FEATURES)].copy(),
        "phq_history": history.copy(),
    }


def _domain_oof(
    frame: pd.DataFrame,
    features: Sequence[str],
    spec: CandidateSpec,
    *,
    weight_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keep = [
        "r4_row_id",
        "dataset_id",
        "global_participant_id",
        "outer_fold",
        "phq9_ge5_target",
        "phq9_ge10_target",
    ]
    prediction = frame[keep].copy()
    audit: dict[str, Any] = {}
    for head, target in HEAD_TARGETS.items():
        probability = pd.Series(np.nan, index=frame.index, dtype=float)
        audit[head] = {}
        for outer_fold in range(5):
            train = frame.loc[frame["outer_fold"].ne(outer_fold)].copy()
            test = frame.loc[frame["outer_fold"].eq(outer_fold)].copy()
            folds = inner_folds_for_outer(frame, outer_fold).loc[train.index].astype(int)
            current = CandidateSpec(
                spec.candidate_id, spec.family, spec.params, spec.seed + outer_fold
            )
            outer, _inner, _calibrator, fold_audit = fit_outer_fixed(
                train,
                test,
                features,
                target,
                folds,
                current,
                weight_mode=weight_mode,
            )
            probability.loc[test.index] = outer
            audit[head][str(outer_fold)] = fold_audit
        prediction[f"probability_{head}"] = probability.loc[frame.index].to_numpy(float)
    prediction["probability_ge5"], prediction["probability_ge10"] = project_heads(
        prediction["probability_ge5"], prediction["probability_ge10"]
    )
    return prediction, audit


def _report(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rows": int(len(frame)),
        "participants": int(frame["global_participant_id"].nunique()),
        "probability_order_violations": int(
            (frame["probability_ge10"] > frame["probability_ge5"]).sum()
        ),
        "heads": {},
    }
    for head, target in HEAD_TARGETS.items():
        natural = metric_block(
            frame,
            target=target,
            probability=f"probability_{head}",
            participant_equal=False,
        )
        participant = metric_block(
            frame,
            target=target,
            probability=f"probability_{head}",
            participant_equal=True,
        )
        by_fold = {
            str(fold): metric_block(
                part,
                target=target,
                probability=f"probability_{head}",
                participant_equal=True,
            )
            for fold, part in frame.groupby("outer_fold", sort=True)
        }
        by_source = {
            str(source): metric_block(
                part,
                target=target,
                probability=f"probability_{head}",
                participant_equal=True,
            )
            for source, part in frame.groupby("dataset_id", sort=True)
            if part[target].nunique() == 2
        }
        result["heads"][head] = {
            "natural": natural,
            "participant_equal": participant,
            "by_fold": by_fold,
            "by_source": by_source,
        }
    return result


def run_r9_baseline_replay(
    *, repository_root_value: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    protocol_path = root / DEFAULT_DATA_RELATIVE / "protocol/r9_protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    completed_report = root / BASELINE_REPORT_RELATIVE / "baseline_replay_report.json"
    if protocol.get("status") == "pass":
        expected_sha = protocol.get("paired_baseline_report_sha256")
        if (
            protocol.get("candidate_outer_metrics_opened") is False
            and {"OPT-V333-R9-000", "OPT-V333-R9-001"}.issubset(
                set(protocol.get("tasks_completed", []))
            )
            and completed_report.is_file()
            and expected_sha == sha256_file(completed_report)
        ):
            return json.loads(completed_report.read_text(encoding="utf-8"))
        raise ValueError("R9 baseline is marked complete but its sealed report is invalid")
    if protocol.get("status") != "prepared-awaiting-paired-baseline":
        raise ValueError("R9 protocol is not ready for its one baseline replay")
    report = root / BASELINE_REPORT_RELATIVE
    frames = load_r9_frames(root)
    outputs: dict[str, pd.DataFrame] = {}
    result: dict[str, Any] = {
        "status": "pass",
        "protocol_version": R9_PROTOCOL_VERSION,
        "candidate_outer_metrics_opened": False,
        "domains": {},
    }
    features = {
        "activity": ACTIVITY_FEATURES,
        "sleep": SLEEP_FEATURES,
        "phq_history": HISTORY_FEATURES,
    }
    modes = {
        "activity": "participant_equal",
        "sleep": "participant_source_balanced",
        "phq_history": "participant_equal",
    }
    for domain in ("activity", "sleep", "phq_history"):
        oof, audit = _domain_oof(
            frames[domain],
            features[domain],
            BASELINE_SPECS[domain],
            weight_mode=modes[domain],
        )
        output_path = report / domain / "baseline_outer_oof.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            raise FileExistsError(output_path)
        oof.to_parquet(output_path, index=False)
        write_json(report / domain / "fold_audit.json", audit, overwrite=overwrite)
        domain_report = _report(oof)
        domain_report.update(
            {
                "spec": BASELINE_SPECS[domain].to_dict(),
                "features": list(features[domain]),
                "weight_mode": modes[domain],
                "oof_sha256": sha256_file(output_path),
            }
        )
        write_json(report / domain / "metrics.json", domain_report, overwrite=overwrite)
        result["domains"][domain] = domain_report
        outputs[domain] = oof

    activity_sleep = outputs["activity"].merge(
        outputs["sleep"],
        on=[
            "r4_row_id",
            "dataset_id",
            "global_participant_id",
            "outer_fold",
            "phq9_ge5_target",
            "phq9_ge10_target",
        ],
        suffixes=("_activity", "_sleep"),
        validate="one_to_one",
    )
    activity_sleep = activity_sleep.loc[
        activity_sleep["dataset_id"].isin(["nhanes", "resilient"])
    ].copy()
    paired_as = activity_sleep[[
        "r4_row_id", "dataset_id", "global_participant_id", "outer_fold",
        "phq9_ge5_target", "phq9_ge10_target",
    ]].copy()
    for head in ("ge5", "ge10"):
        paired_as[f"probability_{head}"] = 0.5 * (
            activity_sleep[f"probability_{head}_activity"]
            + activity_sleep[f"probability_{head}_sleep"]
        )
    paired_as["probability_ge5"], paired_as["probability_ge10"] = project_heads(
        paired_as["probability_ge5"], paired_as["probability_ge10"]
    )

    sleep_history = outputs["phq_history"].merge(
        outputs["sleep"],
        on=[
            "r4_row_id",
            "dataset_id",
            "global_participant_id",
            "outer_fold",
            "phq9_ge5_target",
            "phq9_ge10_target",
        ],
        suffixes=("_history", "_sleep"),
        validate="one_to_one",
    )
    paired_sh = sleep_history[[
        "r4_row_id", "dataset_id", "global_participant_id", "outer_fold",
        "phq9_ge5_target", "phq9_ge10_target",
    ]].copy()
    for head in ("ge5", "ge10"):
        paired_sh[f"probability_{head}"] = (
            0.9 * sleep_history[f"probability_{head}_history"]
            + 0.1 * sleep_history[f"probability_{head}_sleep"]
        )
    paired_sh["probability_ge5"], paired_sh["probability_ge10"] = project_heads(
        paired_sh["probability_ge5"], paired_sh["probability_ge10"]
    )
    paired_reports = {}
    for name, frame in (("activity_sleep_50_50", paired_as), ("sleep_history_10_90", paired_sh)):
        path = report / "paired" / f"{name}_baseline_outer_oof.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        paired_reports[name] = {**_report(frame), "oof_sha256": sha256_file(path)}
    result["paired_signatures"] = paired_reports
    write_json(report / "baseline_replay_report.json", result, overwrite=overwrite)

    protocol["status"] = "pass"
    protocol["tasks_completed"] = [
        "OPT-V333-R9-000",
        "OPT-V333-R9-001",
    ]
    protocol["paired_baseline_report"] = (
        BASELINE_REPORT_RELATIVE / "baseline_replay_report.json"
    ).as_posix()
    protocol["paired_baseline_report_sha256"] = sha256_file(
        report / "baseline_replay_report.json"
    )
    protocol["candidate_outer_metrics_opened"] = False
    write_json(protocol_path, protocol, overwrite=True)
    return result


__all__ = [
    "BASELINE_REPORT_RELATIVE",
    "BASELINE_SPECS",
    "load_r9_frames",
    "run_r9_baseline_replay",
]
