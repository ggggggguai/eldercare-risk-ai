"""Deterministic R10 rule fusion v4; joint probability nodes affect the decision."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from itertools import product
import json
from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.mental_health.mood_social.r7.rule_fusion import (
    ATTENTION_ANCHORS,
    DomainEvidence,
    HistoryEvidence,
    fuse_current_state,
)

from .evidence_graph import graph_contract_sha256, route_evidence_v2


R10_RULE_VERSION = "mood-social-r10-rule-fusion-v4"


def _level_from_probability(assessment: Mapping[str, Any]) -> int | None:
    if not assessment.get("available"):
        return None
    if assessment.get("domain_level") is not None:
        return int(assessment["domain_level"])
    p5 = assessment.get("probability_phq_ge5")
    p10 = assessment.get("probability_phq_ge10")
    if p5 is None or p10 is None:
        return None
    if float(p10) >= float(assessment.get("threshold_ge10", 0.5)):
        return 3
    if float(p10) >= 0.35:
        return 2
    if float(p5) >= float(assessment.get("threshold_ge5", 0.5)):
        return 1
    return 0


def _state(level: int | None, reason: str) -> dict[str, Any]:
    if level is None:
        return {
            "available": False,
            "level": None,
            "attention_index": None,
            "attention_index_semantics": "rule-derived-not-PHQ-probability",
            "reason": reason,
        }
    return {
        "available": True,
        "level": int(level),
        "attention_index": ATTENTION_ANCHORS[int(level)],
        "attention_index_semantics": "rule-derived-not-PHQ-probability",
        "reason": reason,
    }


def _support_level(value: Mapping[str, Any] | None) -> int:
    if not value or not value.get("available"):
        return 0
    return int(value.get("domain_level", value.get("level", 0)) or 0)


def fuse_r10_current_state(
    dynamic: Iterable[DomainEvidence],
    *,
    activity_sleep_joint: Mapping[str, Any] | None = None,
    sleep_history_joint: Mapping[str, Any] | None = None,
    history: Mapping[str, Any] | None = None,
    facial: Mapping[str, Any] | None = None,
    profile_background: Mapping[str, Any] | None = None,
    physiology_support: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dynamic_list = [item for item in dynamic if item.available]
    by_domain = {item.domain: item for item in dynamic_list}
    available = set(by_domain)
    if history and history.get("available"):
        available.add("phq_history")
    if facial and facial.get("available"):
        available.add("facial_affect")
    if profile_background and profile_background.get("available", True):
        available.add("profile")
    if physiology_support and physiology_support.get("available", True):
        available.add("physiology")
    as_available = bool(activity_sleep_joint and activity_sleep_joint.get("available"))
    sh_available = bool(sleep_history_joint and sleep_history_joint.get("available"))
    route = route_evidence_v2(
        available,
        activity_sleep_joint=as_available,
        sleep_history_joint=sh_available,
    )
    selected = route.selected_probability_node
    reasons: list[str] = []
    flags: list[str] = []
    if selected == "sleep_history_v2":
        base_level = _level_from_probability(sleep_history_joint or {})
        reasons.append("sleep_history_v2_selected_probability_node")
        activity_support = by_domain.get("activity")
        if activity_support and int(activity_support.level or 0) >= 2 and base_level is not None:
            base_level = min(base_level + 1, 3)
            reasons.append("independent_activity_support_confirmed")
    elif selected == "activity_sleep_v2":
        base_level = _level_from_probability(activity_sleep_joint or {})
        reasons.append("activity_sleep_v2_selected_probability_node")
    else:
        independent = [
            item for item in dynamic_list if item.domain in {"activity", "sleep", "social"}
        ]
        history_evidence = None
        if history and history.get("available"):
            history_evidence = HistoryEvidence(
                available=True,
                level=history.get("domain_level", history.get("level")),
                age_days=int(history.get("last_age_days", history.get("age_days", 999))),
                high=bool(history.get("high")),
                low=bool(history.get("low")),
            )
        base = fuse_current_state(
            independent,
            history=history_evidence,
            profile_background=profile_background,
            physiology_support=physiology_support,
        )
        base_level = base["history_informed_current_state"].get("level")
        reasons.append("independent_expert_fallback")
    # Social and facial from the same S10 interaction form one support cluster.
    social_level = _support_level(
        {
            "available": "social" in by_domain,
            "domain_level": by_domain.get("social").level if "social" in by_domain else 0,
        }
    )
    facial_votes = bool(
        facial
        and facial.get("available")
        and facial.get("supports_rule_vote")
        and facial.get("evidence_label") in {"F1", "F2"}
    )
    s10_cluster = max(social_level, 1 if facial_votes else 0)
    if base_level is not None and base_level > 0 and s10_cluster >= 2 and base_level < 3:
        base_level += 1
        reasons.append("s10_correlated_support_cluster_confirmed")
    if base_level is None and facial_votes:
        base_level = 1
        reasons.append("facial_support_only_capped_l1")
    if facial and facial.get("prediction") in {"positive", "surprise"}:
        flags.append("facial_positive_or_surprise_never_downranks")
    if history and history.get("available") and history.get("low") and base_level and base_level >= 1:
        flags.append("possible_new_onset_history_low_did_not_downrank")
    operational = _state(base_level, reasons[-1] if reasons else "insufficient_evidence")
    operational.update(
        {
            "source_state": "joint_probability_rule_v4" if selected else "independent_rule_v4",
            "decision_reasons": reasons,
            "discordance_flags": flags,
            "rule_version": R10_RULE_VERSION,
            "rule_contract_sha256": rule_contract_sha256(),
        }
    )
    return {
        "operational_attention_state": operational,
        "evidence_graph": route.to_dict(),
        "rule_fusion": {
            "rule_version": R10_RULE_VERSION,
            "rule_contract_sha256": rule_contract_sha256(),
            "graph_contract_sha256": graph_contract_sha256(),
            "selected_probability_node": selected,
            "joint_probability_used_in_decision": selected is not None,
            "consumed_probability_nodes": list(route.consumed_probability_nodes),
            "decision_reasons": reasons,
            "discordance_flags": flags,
            "attention_index_is_phq_probability": False,
        },
    }


def enumerate_truth_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    levels: tuple[int | None, ...] = (None, 0, 1, 2, 3)
    for activity, sleep, social, history, facial_vote in product(
        levels, levels, levels, levels, (False, True)
    ):
        dynamic = [
            DomainEvidence(name, level is not None, level, 1.0)
            for name, level in (("activity", activity), ("sleep", sleep), ("social", social))
            if level is not None
        ]
        hist = {
            "available": history is not None,
            "domain_level": history,
            "low": history == 0,
        }
        facial = {
            "available": facial_vote,
            "supports_rule_vote": facial_vote,
            "evidence_label": "F1" if facial_vote else "unavailable",
            "prediction": "negative" if facial_vote else None,
        }
        for as_level, sh_level in ((None, None), (1, None), (2, None), (None, 1), (None, 2), (2, 2)):
            as_joint = {"available": as_level is not None, "domain_level": as_level}
            sh_joint = {"available": sh_level is not None, "domain_level": sh_level}
            result = fuse_r10_current_state(
                dynamic,
                activity_sleep_joint=as_joint,
                sleep_history_joint=sh_joint,
                history=hist,
                facial=facial,
            )
            rows.append(
                {
                    "activity": activity,
                    "sleep": sleep,
                    "social": social,
                    "history": history,
                    "facial_vote": facial_vote,
                    "activity_sleep_joint": as_level,
                    "sleep_history_joint": sh_level,
                    "selected": result["rule_fusion"]["selected_probability_node"],
                    "level": result["operational_attention_state"]["level"],
                }
            )
    return rows


def validate_truth_table() -> dict[str, Any]:
    rows = enumerate_truth_table()
    violations: list[str] = []
    for row in rows:
        if row["sleep_history_joint"] is not None and row["sleep"] is not None and row["history"] is not None:
            if row["selected"] != "sleep_history_v2":
                violations.append("eligible S+History was not selected")
        elif row["activity_sleep_joint"] is not None and row["activity"] is not None and row["sleep"] is not None:
            if row["selected"] != "activity_sleep_v2":
                violations.append("eligible A+S was not selected")
        if row["facial_vote"] and all(row[name] is None for name in ("activity", "sleep", "social", "history")):
            if row["level"] not in {1, None}:
                violations.append("facial alone exceeded L1")
        if (
            row["sleep_history_joint"] is not None
            and row["activity_sleep_joint"] is not None
            and row["activity"] is not None
            and row["sleep"] is not None
            and row["history"] is not None
        ):
            if row["selected"] == "activity_sleep_v2":
                violations.append("shared sleep was consumed twice")
    return {
        "status": "pass" if not violations else "fail",
        "row_count": len(rows),
        "violation_count": len(violations),
        "violations": violations[:100],
    }


def rule_contract_sha256() -> str:
    payload = {
        "version": R10_RULE_VERSION,
        "joint_probability_drives_operational_decision": True,
        "sleep_consumed_once": True,
        "social_facial_cluster_max_one": True,
        "facial_alone_max_level": 1,
        "profile_or_physiology_alone_high": False,
        "positive_or_surprise_downrank": False,
        "history_low_downrank": False,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "R10_RULE_VERSION",
    "enumerate_truth_table",
    "fuse_r10_current_state",
    "rule_contract_sha256",
    "validate_truth_table",
]
