"""Deterministic R9 rule fusion v3 with history freshness and evidence de-duplication."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from itertools import product
from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.mental_health.mood_social.r7.rule_fusion import (
    ATTENTION_ANCHORS,
    DomainEvidence,
    fuse_current_state,
)
from elderly_monitoring.modules.mental_health.mood_social.r8.rule_fusion import apply_facial_support

from .evidence_graph import route_evidence


R9_RULE_VERSION = "mood-social-r9-rule-fusion-v3"


def _state(level: int | None, reason: str | None = None) -> dict[str, object]:
    if level is None:
        return {"available": False, "level": None, "attention_index": None, "attention_index_semantics": "rule-derived-not-PHQ-probability", "reason": reason or "insufficient_dynamic_evidence"}
    return {"available": True, "level": int(level), "attention_index": ATTENTION_ANCHORS[int(level)], "attention_index_semantics": "rule-derived-not-PHQ-probability", "reason": reason}


def _history_adjustment(
    current_level: int | None,
    history: Mapping[str, Any] | None,
) -> tuple[int | None, bool, list[str], list[str]]:
    if not history or not history.get("available"):
        return current_level, False, [], []
    age = history.get("last_age_days", history.get("age_days"))
    level = history.get("level")
    high = bool(history.get("high", level is not None and int(level) >= 2))
    low = bool(history.get("low", level == 0))
    flags: list[str] = []
    reasons: list[str] = []
    if age is None:
        return current_level, False, ["history_age_unknown_abstain"], reasons
    age = float(age)
    if age < 14:
        return current_level, False, ["history_0_13_day_overlap_sensitivity_only"], reasons
    if age > 180:
        return current_level, False, ["history_over_180_day_abstain"], reasons
    if age > 100:
        return current_level, False, ["history_101_180_day_background_retest"], reasons
    reasons.append("history_14_30_strong" if age <= 30 else "history_31_100_time_decayed")
    if low and current_level is not None and current_level >= 1:
        flags.append("possible_new_onset_change")
        return current_level, True, flags, reasons
    if not high:
        return current_level, True, flags, reasons
    if current_level is None:
        return min(int(level or 1), 2), True, flags, reasons
    if current_level == 0:
        flags.append("historical_high_dynamic_normal_retest")
        return 1, True, flags, reasons
    return min(int(current_level) + 1, 3), True, flags, reasons


def fuse_r9_current_state(
    dynamic: Iterable[DomainEvidence],
    *,
    history: Mapping[str, Any] | None = None,
    facial: Mapping[str, Any] | None = None,
    profile_background: Mapping[str, Any] | None = None,
    physiology_support: Mapping[str, Any] | None = None,
    local_joint_assessments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dynamic_list = list(dynamic)
    available = {item.domain for item in dynamic_list if item.available}
    if history and history.get("available"):
        available.add("phq_history")
    if facial and facial.get("available"):
        available.add("facial_affect")
    if profile_background:
        available.add("profile")
    if physiology_support:
        available.add("physiology")
    route = route_evidence(available)
    core = fuse_current_state(
        dynamic_list,
        history=None,
        profile_background=profile_background,
        physiology_support=physiology_support,
    )
    core["expert_assessments"] = {"phq_history": dict(history or {"available": False})}
    affect = dict(facial or {"available": False, "evidence_label": "unavailable", "supports_rule_vote": False, "phq_probability_available": False})
    facial_result = apply_facial_support(core, affect)
    interaction = deepcopy(facial_result["interaction_enhanced_current_state"])
    interaction_level = interaction.get("level") if interaction.get("available") else None
    adjusted, history_used, history_flags, history_reasons = _history_adjustment(interaction_level, history)
    history_state = _state(adjusted, "history_freshness_rule" if history_used else interaction.get("reason"))
    operational = deepcopy(history_state if history_used else interaction)
    reasons = list(facial_result["operational_attention_state"].get("decision_reasons", [])) + history_reasons
    flags = list(facial_result["operational_attention_state"].get("discordance_flags", [])) + history_flags
    operational.update({
        "source_state": "history_informed_integrated_state" if history_used else "interaction_enhanced_current_state",
        "decision_reasons": reasons,
        "discordance_flags": flags,
        "rule_version": R9_RULE_VERSION,
        "rule_contract_sha256": rule_contract_sha256(),
    })
    return {
        "passive_current_state": deepcopy(core["passive_current_state"]),
        "facial_affect_assessment": affect,
        "interaction_enhanced_current_state": interaction,
        "history_informed_integrated_state": history_state,
        "operational_attention_state": operational,
        "evidence_graph": route.to_dict(),
        "local_joint_assessments": deepcopy(dict(local_joint_assessments or {})),
        "rule_fusion": {
            "rule_version": R9_RULE_VERSION,
            "rule_contract_sha256": rule_contract_sha256(),
            "history_vote_used": history_used,
            "decision_reasons": reasons,
            "discordance_flags": flags,
            "joint_nodes_add_votes": False,
            "phq_probability_semantics": False,
        },
    }


def enumerate_truth_table() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    levels: tuple[int | None, ...] = (None, 0, 1, 2, 3)
    facial_options = (None, "no_clue", "observation_only", "F1", "F2")
    history_options = (None, "low", "high_fresh", "high_decayed", "high_old")
    for activity, sleep, social, facial_option, history_option in product(levels, levels, levels, facial_options, history_options):
        dynamic = [DomainEvidence(domain, level is not None, level, 1.0) for domain, level in (("activity", activity), ("sleep", sleep), ("social", social)) if level is not None]
        facial = {"available": facial_option is not None, "evidence_label": facial_option or "unavailable", "supports_rule_vote": facial_option in {"F1", "F2"}, "phq_probability_available": False}
        history = {
            "available": history_option is not None,
            "level": 0 if history_option == "low" else 2,
            "low": history_option == "low",
            "high": history_option not in {None, "low"},
            "last_age_days": {None: None, "low": 20, "high_fresh": 20, "high_decayed": 60, "high_old": 120}[history_option],
        }
        result = fuse_r9_current_state(dynamic, history=history, facial=facial)
        rows.append({
            "activity": activity,
            "sleep": sleep,
            "social": social,
            "facial": facial_option,
            "history": history_option,
            "passive_level": result["passive_current_state"]["level"],
            "interaction_level": result["interaction_enhanced_current_state"]["level"],
            "operational_level": result["operational_attention_state"]["level"],
            "history_vote_used": result["rule_fusion"]["history_vote_used"],
        })
    return rows


def validate_truth_table(rows: list[dict[str, object]] | None = None) -> dict[str, object]:
    rows = rows or enumerate_truth_table()
    violations: list[str] = []
    for row in rows:
        passive, interaction, operational = row["passive_level"], row["interaction_level"], row["operational_level"]
        if row["facial"] in {None, "no_clue", "observation_only"} and interaction != passive:
            violations.append("ineligible facial changed passive result")
        if isinstance(passive, int) and isinstance(interaction, int) and interaction < passive:
            violations.append("facial lowered passive result")
        if row["history"] in {None, "high_old"} and operational != interaction:
            violations.append("ineligible or old history changed result")
        if row["history"] == "low" and isinstance(interaction, int) and operational < interaction:
            violations.append("low history lowered new dynamic evidence")
    return {"status": "pass" if not violations else "fail", "row_count": len(rows), "violation_count": len(violations), "violations": violations[:100]}


def rule_contract_sha256() -> str:
    payload = {
        "version": R9_RULE_VERSION,
        "history": {"0_13": "sensitivity-only", "14_30": "strong", "31_100": "decayed", "101_180": "background", "over_180": "abstain"},
        "joint_nodes_add_votes": False,
        "history_low_never_lowers": True,
        "facial_role": "bounded-support-only",
        "profile_role": "background-only",
        "physiology_role": "support-only",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


__all__ = ["R9_RULE_VERSION", "enumerate_truth_table", "fuse_r9_current_state", "rule_contract_sha256", "validate_truth_table"]
