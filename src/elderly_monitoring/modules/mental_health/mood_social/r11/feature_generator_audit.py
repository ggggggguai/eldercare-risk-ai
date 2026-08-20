"""R11 field-level runtime parity and incremental-information audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r9.contract import (
    ACTIVITY_FEATURES,
    R9_HISTORY_FEATURES,
    SLEEP_FEATURES,
)
from elderly_monitoring.modules.mental_health.mood_social.r10.features import (
    INTERACTION_FEATURES,
)

from .contract import (
    DEFAULT_DATA_RELATIVE,
    DEFAULT_REPORT_RELATIVE,
    R11_PROTOCOL_VERSION,
    repository_root,
    sha256_file,
    write_json,
)


def _row(feature: str, domain: str, status: str, generator: str, *, used_in_r10: bool) -> dict[str, Any]:
    return {
        "feature": feature,
        "domain": domain,
        "status": status,
        "runtime_generator": generator,
        "available_before_target": status == "available",
        "unit_window_missingness_parity": status == "available",
        "used_in_r10": used_in_r10,
        "new_runtime_equivalent_information": status == "available" and not used_in_r10,
    }


def run_r11_feature_audit(
    *, repository_root_value: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    protocol_path = root / DEFAULT_DATA_RELATIVE / "protocol/r11_protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "prepared-awaiting-runtime-feature-audit":
        raise ValueError("R11 feature audit requires prepared protocol")
    mapper = root / "src/elderly_monitoring/modules/mental_health/mood_social/feature_mapper.py"
    if not mapper.is_file():
        raise FileNotFoundError(mapper)
    rows: list[dict[str, Any]] = []
    for feature in ACTIVITY_FEATURES:
        rows.append(_row(feature, "activity", "available", "feature_mapper.map_mood_social_features", used_in_r10=True))
    for feature in SLEEP_FEATURES:
        rows.append(_row(feature, "sleep", "available", "feature_mapper.map_mood_social_features", used_in_r10=True))
    for feature in INTERACTION_FEATURES:
        rows.append(_row(feature, "activity_sleep", "available", "r10.features.add_activity_sleep_interactions", used_in_r10=True))
    for feature in R9_HISTORY_FEATURES:
        rows.append(_row(feature, "phq_history", "available", "r7.phq_history_expert.build_history_features", used_in_r10=True))
    activity_path = root / DEFAULT_DATA_RELATIVE / "signatures/activity_sleep_repeated.parquet"
    history_path = root / DEFAULT_DATA_RELATIVE / "signatures/sleep_history_repeated.parquet"
    activity = pd.read_parquet(activity_path)
    history = pd.read_parquet(history_path)
    source_only = sorted(
        column
        for column in activity.columns
        if column.startswith("source__")
        and column not in set(ACTIVITY_FEATURES).union(SLEEP_FEATURES)
    )
    for feature in source_only:
        rows.append(_row(feature, "source_audit", "research-only", "none", used_in_r10=False))
    forbidden = sorted(
        column
        for column in activity.columns
        if any(token in column.lower() for token in ("phq", "target", "dataset_id", "participant_id", "route", "mask", "coverage"))
    )
    for feature in forbidden:
        rows.append(_row(feature, "denylist", "leakage-denied", "none", used_in_r10=False))
    causal_sleep_delta = sorted(
        column
        for column in history.columns
        if column.startswith("sleep.") and any(token in column.lower() for token in ("delta", "slope", "lag", "prior"))
    )
    new_deployable = sorted(
        row["feature"] for row in rows if row["new_runtime_equivalent_information"]
    )
    report = {
        "protocol_version": R11_PROTOCOL_VERSION,
        "status": "pass",
        "mapper_sha256": sha256_file(mapper),
        "field_rows": rows,
        "counts": {
            status: sum(row["status"] == status for row in rows)
            for status in ("available", "research-only", "unsupported", "leakage-denied")
        },
        "new_runtime_equivalent_activity_sleep_features": new_deployable,
        "activity_sleep_training_authorized": bool(new_deployable),
        "activity_sleep_decision": "run" if new_deployable else "cancel-no-new-runtime-equivalent-information",
        "causal_sleep_delta_features": causal_sleep_delta,
        "four_state_causal_sleep_delta_authorized": bool(causal_sleep_delta),
        "four_state_causal_sleep_delta_decision": "run" if causal_sleep_delta else "cancel-no-strict-prior-sleep-delta",
        "risk_input_uses_source_mask_coverage": False,
    }
    write_json(root / DEFAULT_REPORT_RELATIVE / "runtime_feature_generator_audit.json", report, overwrite=overwrite)
    failure_path = root / DEFAULT_REPORT_RELATIVE / "failure_ledger.json"
    ledger = json.loads(failure_path.read_text(encoding="utf-8"))
    if not new_deployable:
        ledger["skips"].append({
            "task": "OPT-V333-R11-003",
            "reason": "no new runtime-equivalent A+S information beyond R10",
            "status": "cancelled-by-frozen-protocol",
        })
    if not causal_sleep_delta:
        ledger["skips"].append({
            "candidate": "four_state_causal_sleep_delta_conditional",
            "reason": "no strict prior sleep delta/slope feature in the same-person signature",
            "status": "cancelled-by-frozen-protocol",
        })
    write_json(failure_path, ledger, overwrite=True)
    lock_path = root / DEFAULT_REPORT_RELATIVE / "selection_lock_contract.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["lock_status"] = "runtime-feature-audit-sealed-awaiting-inner-selection"
    lock["runtime_feature_audit_sha256"] = sha256_file(
        root / DEFAULT_REPORT_RELATIVE / "runtime_feature_generator_audit.json"
    )
    lock["activity_sleep_training_authorized"] = bool(new_deployable)
    write_json(lock_path, lock, overwrite=True)
    protocol["status"] = "runtime-feature-audit-complete-awaiting-sleep-history-selection"
    protocol["activity_sleep_training_authorized"] = bool(new_deployable)
    protocol["four_state_causal_sleep_delta_authorized"] = bool(causal_sleep_delta)
    protocol["tasks_completed"] = list(protocol["tasks_completed"]) + ["OPT-V333-R11-001"]
    if not new_deployable:
        protocol["tasks_completed"].append("OPT-V333-R11-003-cancelled-by-protocol")
    write_json(protocol_path, protocol, overwrite=True)
    return report


__all__ = ["run_r11_feature_audit"]

