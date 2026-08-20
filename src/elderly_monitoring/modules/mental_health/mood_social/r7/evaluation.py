"""R7 single-expert, bootstrap and real-overlap partial-scenario evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r5.evaluation import paired_participant_bootstrap
from elderly_monitoring.modules.mental_health.mood_social.r7.contract import R7_BOOTSTRAP_RESAMPLES, R7_BOOTSTRAP_SEED, repository_root, sha256_file
from elderly_monitoring.modules.mental_health.mood_social.r7.modeling import binary_metrics, write_json
from elderly_monitoring.modules.mental_health.mood_social.r7.rule_fusion import DomainEvidence, HistoryEvidence, enumerate_truth_table, fuse_current_state, rule_contract_sha256


REPORT_RELATIVE = Path("reports/mental_health/mood_social/v3.3.3-r7/OPT-V333-R7-006-evaluation")
INDEPENDENT_RELATIVE = Path("reports/mental_health/mood_social/v3.3.3-r7/OPT-V333-R7-002-independent-experts")
HISTORY_RELATIVE = Path("reports/mental_health/mood_social/v3.3.3-r7/OPT-V333-R7-004-history")
SOCIAL_RELATIVE = Path("reports/mental_health/mood_social/v3.3.3-r7/OPT-V333-R7-003-social")


def _bootstrap(frame: pd.DataFrame, head: str, dataset: str) -> dict[str, Any]:
    work = frame[["global_participant_id", f"phq9_{head}_target", f"probability_{head}", f"baseline_probability_{head}"]].copy()
    work["dataset_id"] = dataset
    work["binary_target"] = work[f"phq9_{head}_target"].astype(int)
    work["candidate_probability"] = work[f"probability_{head}"].astype(float)
    work["baseline_probability"] = work[f"baseline_probability_{head}"].astype(float)
    return paired_participant_bootstrap(work, repetitions=R7_BOOTSTRAP_RESAMPLES, seed=R7_BOOTSTRAP_SEED + (5 if head == "ge5" else 10))


def _level(p5: float, p10: float) -> int:
    return 2 if p10 >= 0.5 else 1 if p5 >= 0.5 else 0


def _partial_scenario(root: Path) -> dict[str, Any]:
    frames = {domain: pd.read_parquet(root / INDEPENDENT_RELATIVE / domain / "three_seed_mean_oof.parquet") for domain in ("activity", "sleep", "profile")}
    merged = frames["activity"].merge(frames["sleep"], on=["r4_row_id", "dataset_id", "global_participant_id", "outer_fold", "phq9_ge5_target", "phq9_ge10_target"], suffixes=("_activity", "_sleep"), validate="one_to_one")
    merged = merged.loc[merged["dataset_id"].isin(["nhanes", "resilient"])].copy()
    scores: list[float] = []
    for row in merged.itertuples(index=False):
        result = fuse_current_state([
            DomainEvidence("activity", True, _level(row.probability_ge5_activity, row.probability_ge10_activity), 1.0),
            DomainEvidence("sleep", True, _level(row.probability_ge5_sleep, row.probability_ge10_sleep), 1.0),
        ])
        scores.append(float(result["passive_current_state"]["attention_index"]) / 100.0)
    merged["rule_score"] = scores
    return {
        "scenario": "real-person activity+sleep overlap; profile evaluated separately as non-voting background",
        "rows": int(len(merged)),
        "participants": int(merged["global_participant_id"].nunique()),
        "sources": {str(k): int(v) for k, v in merged["dataset_id"].value_counts().items()},
        "ge10": binary_metrics(merged["phq9_ge10_target"], merged["rule_score"]),
        "ge5": binary_metrics(merged["phq9_ge5_target"], merged["rule_score"]),
        "limitations": "partial-scenario only; not complete five-domain fusion and not real-device validation",
    }


def run_r7_evaluation(*, repository_root_value: Path | None = None, overwrite: bool = False) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    experts: dict[str, Any] = {}
    bootstrap: dict[str, Any] = {}
    for domain in ("activity", "sleep", "profile"):
        frame = pd.read_parquet(root / INDEPENDENT_RELATIVE / domain / "three_seed_mean_oof.parquet")
        metrics = json.loads((root / INDEPENDENT_RELATIVE / domain / "metrics.json").read_text(encoding="utf-8"))
        experts[domain] = metrics
        bootstrap[domain] = {head: _bootstrap(frame, head, domain) for head in ("ge5", "ge10")}
    history = json.loads((root / HISTORY_RELATIVE / "metrics.json").read_text(encoding="utf-8"))
    social = json.loads((root / SOCIAL_RELATIVE / "confirmation_report.json").read_text(encoding="utf-8"))
    result = {
        "status": "pass",
        "protocol": "mood-social-v3.3.3-r7",
        "single_experts": {**experts, "history_phq": history, "social": social},
        "participant_bootstrap_2000": bootstrap,
        "partial_scenario_rule_fusion": _partial_scenario(root),
        "rule_contract_sha256": rule_contract_sha256(),
        "complete_five_domain_auprc_reported": False,
        "reason": "no real participant has all five modalities with contemporaneous PHQ labels",
        "evidence_limits": ["existing five-source cohort is adaptive/reused", "DepreST confirmation is model-unseen but not elderly", "no real device data"],
    }
    output = root / REPORT_RELATIVE
    write_json(output / "evaluation_report.json", result, overwrite=overwrite)
    write_json(output / "rule_truth_table_contract.json", {"sha256": rule_contract_sha256(), "anchors": {"L0": 15, "L1": 40, "L2": 65, "L3": 90}, "trained": False, "label_tuned": False}, overwrite=overwrite)
    write_json(output / "rule_truth_table.json", enumerate_truth_table(), overwrite=overwrite)
    write_json(output / "artifact_manifest.json", {"evaluation_sha256": sha256_file(output / "evaluation_report.json"), "rule_sha256": sha256_file(output / "rule_truth_table_contract.json")}, overwrite=overwrite)
    return result


__all__ = ["REPORT_RELATIVE", "run_r7_evaluation"]
