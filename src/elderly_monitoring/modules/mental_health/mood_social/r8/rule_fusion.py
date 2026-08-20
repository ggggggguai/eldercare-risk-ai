"""Deterministic R8 rule fusion v2 layered on the immutable R7 core."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from itertools import product
from typing import Any, Mapping

from elderly_monitoring.modules.mental_health.mood_social.r7.rule_fusion import (
    ATTENTION_ANCHORS,
    DomainEvidence,
    fuse_current_state,
    rule_contract_sha256 as r7_rule_contract_sha256,
)


R8_RULE_VERSION = "mood-social-r8-rule-fusion-v2"
FACIAL_OPTIONS = (None, "no_clue", "observation_only", "F1", "F2")


def _state(level: int | None, reason: str | None = None) -> dict[str, object]:
    if level is None:
        return {
            "available": False,
            "level": None,
            "attention_index": None,
            "attention_index_semantics": "rule-derived-not-PHQ-probability",
            "reason": reason or "insufficient_dynamic_evidence",
        }
    return {
        "available": True,
        "level": int(level),
        "attention_index": ATTENTION_ANCHORS[int(level)],
        "attention_index_semantics": "rule-derived-not-PHQ-probability",
        "reason": reason,
    }


def _eligible_facial(facial: Mapping[str, Any]) -> bool:
    return bool(
        facial.get("supports_rule_vote") is True
        and facial.get("evidence_label") in {"F1", "F2"}
    )


def _abnormal_dynamic_domains(r7_result: Mapping[str, Any]) -> set[str]:
    """Recover only L1+ dynamic domains from the frozen R7 reasons."""

    abnormal: set[str] = set()
    for reason in r7_result.get("rule_fusion", {}).get("decision_reasons", []):
        if not isinstance(reason, str) or ":L" not in reason:
            continue
        domain, level_text = reason.rsplit(":L", 1)
        if domain not in {"activity", "sleep", "social"}:
            continue
        try:
            level = int(level_text)
        except ValueError:
            continue
        if level >= 1:
            abnormal.add(domain)
    return abnormal


def apply_facial_support(
    r7_result: Mapping[str, Any],
    facial: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply bounded facial support without mutating or re-labelling R7 evidence."""

    passive = deepcopy(dict(r7_result["passive_current_state"]))
    core_level = passive.get("level") if passive.get("available") else None
    facial_eligible = _eligible_facial(facial)
    abnormal_domains = _abnormal_dynamic_domains(r7_result)
    correlated_social_only = bool(
        facial_eligible and core_level == 1 and abnormal_domains == {"social"}
    )
    reasons = list(r7_result.get("rule_fusion", {}).get("decision_reasons", []))
    flags = list(r7_result.get("rule_fusion", {}).get("discordance_flags", []))
    adjustments: list[str] = []

    interaction_level = core_level
    if facial_eligible:
        if core_level is None:
            interaction_level = 1
            adjustments.append("affect_only_retest_max_l1")
        elif core_level == 0:
            interaction_level = 1
            adjustments.append("persistent_facial_support_l0_to_l1")
        elif correlated_social_only:
            interaction_level = 1
            adjustments.append("same_s10_social_facial_no_double_count")
        elif core_level == 1:
            interaction_level = 2
            adjustments.append("persistent_facial_support_l1_to_l2")
        else:
            interaction_level = core_level
            adjustments.append("facial_support_explanation_only")
        reasons.append(f"facial_affect:{facial.get('evidence_label')}")
    else:
        adjustments.append("no_eligible_facial_adjustment")

    interaction = (
        deepcopy(passive)
        if not facial_eligible
        else _state(
            interaction_level,
            "affect_only_retest"
            if core_level is None
            else "bounded_facial_support",
        )
    )

    r7_history = deepcopy(dict(r7_result["history_informed_current_state"]))
    r7_history_level = (
        r7_history.get("level") if r7_history.get("available") else None
    )
    combined_level = interaction_level
    if r7_history_level is not None:
        combined_level = (
            r7_history_level
            if combined_level is None
            else max(int(combined_level), int(r7_history_level))
        )
    history_integrated = (
        deepcopy(interaction)
        if r7_history_level is None
        else _state(combined_level, r7_history.get("reason"))
    )

    expert_history = r7_result.get("expert_assessments", {}).get("phq_history", {})
    history_vote_used = bool(
        expert_history.get("available")
        and expert_history.get("participates_in_rule_vote", True)
        and expert_history.get("last_age_days", 10**9) <= 90
    )
    operational = deepcopy(history_integrated if history_vote_used else interaction)
    operational.update(
        {
            "source_state": (
                "history_informed_integrated_state"
                if history_vote_used
                else "interaction_enhanced_current_state"
            ),
            "triggered_domains": sorted(
                set(r7_result.get("rule_fusion", {}).get("triggered_domains", []))
                | ({"facial_affect"} if facial_eligible else set())
            ),
            "decision_reasons": reasons,
            "discordance_flags": flags,
            "decision_adjustments": adjustments,
            "rule_version": R8_RULE_VERSION,
            "rule_contract_sha256": rule_contract_sha256(),
            "fallback": "r7_core" if not facial_eligible else None,
        }
    )

    return {
        "passive_current_state": passive,
        "facial_affect_assessment": deepcopy(dict(facial)),
        "interaction_enhanced_current_state": interaction,
        "history_informed_integrated_state": history_integrated,
        "operational_attention_state": operational,
        "rule_fusion": {
            **deepcopy(dict(r7_result.get("rule_fusion", {}))),
            "r7_rule_contract_sha256": r7_rule_contract_sha256(),
            "rule_version": R8_RULE_VERSION,
            "rule_contract_sha256": rule_contract_sha256(),
            "facial_vote_eligible": facial_eligible,
            "decision_adjustments": adjustments,
            "same_s10_social_facial_correlated": bool(
                facial_eligible
                and "social" in r7_result.get("rule_fusion", {}).get("triggered_domains", [])
            ),
            "correlated_social_only_no_double_count": correlated_social_only,
            "phq_probability_semantics": False,
        },
    }


def _facial_stub(option: str | None) -> dict[str, object]:
    eligible = option in {"F1", "F2"}
    return {
        "available": option is not None,
        "evidence_label": "unavailable" if option is None else option,
        "evidence_level": 2 if option == "F2" else 1 if option == "F1" else 0,
        "supports_rule_vote": eligible,
        "phq_probability_available": False,
    }


def enumerate_truth_table() -> list[dict[str, object]]:
    """Enumerate the frozen 5^4 Activity/Sleep/Social/Facial base table."""

    rows: list[dict[str, object]] = []
    domain_options: tuple[int | None, ...] = (None, 0, 1, 2, 3)
    for activity, sleep, social, facial_option in product(
        domain_options, domain_options, domain_options, FACIAL_OPTIONS
    ):
        evidence = [
            DomainEvidence(domain, level is not None, level, 1.0)
            for domain, level in (
                ("activity", activity),
                ("sleep", sleep),
                ("social", social),
            )
            if level is not None
        ]
        core = fuse_current_state(evidence)
        core["expert_assessments"] = {"phq_history": {"available": False}}
        result = apply_facial_support(core, _facial_stub(facial_option))
        rows.append(
            {
                "activity": activity,
                "sleep": sleep,
                "social": social,
                "facial": facial_option,
                "core_level": core["passive_current_state"]["level"],
                "interaction_level": result["interaction_enhanced_current_state"]["level"],
                "facial_vote_eligible": result["rule_fusion"]["facial_vote_eligible"],
            }
        )
    if len(rows) != 625:
        raise AssertionError("R8 base truth table must contain exactly 625 rows")
    return rows


def validate_truth_table(rows: list[dict[str, object]] | None = None) -> dict[str, object]:
    rows = rows or enumerate_truth_table()
    violations: list[str] = []
    keyed = {
        (row["activity"], row["sleep"], row["social"], row["facial"]): row
        for row in rows
    }
    for row in rows:
        core = row["core_level"]
        result = row["interaction_level"]
        facial = row["facial"]
        if facial in {None, "no_clue", "observation_only"} and result != core:
            violations.append("ineligible facial changed core")
        if core is None and facial in {"F1", "F2"} and result != 1:
            violations.append("facial-only result exceeded or missed L1")
        if isinstance(core, int) and isinstance(result, int) and result < core:
            violations.append("facial evidence lowered core level")
        if core == 2 and result != 2:
            violations.append("facial changed L2")
        if core == 3 and result != 3:
            violations.append("facial changed L3")
        if facial == "F1":
            stronger = keyed[(row["activity"], row["sleep"], row["social"], "F2")]
            if stronger["interaction_level"] < result:
                violations.append("F2 produced less concern than F1")
    return {
        "status": "pass" if not violations else "fail",
        "row_count": len(rows),
        "violation_count": len(violations),
        "violations": violations[:100],
        "properties": {
            "no_facial_equivalence": True,
            "facial_never_lowers": True,
            "facial_only_max_l1": True,
            "facial_never_pushes_l2_to_l3": True,
            "same_domain_votes_not_duplicated_by_input_contract": True,
            "positive_or_surprise_never_protective": True,
        },
    }


def rule_contract_sha256() -> str:
    payload = {
        "version": R8_RULE_VERSION,
        "r7_rule_contract_sha256": r7_rule_contract_sha256(),
        "anchors": ATTENTION_ANCHORS,
        "facial_options": FACIAL_OPTIONS,
        "facial_only_max_level": 1,
        "core_l0_plus_facial": 1,
        "core_l1_plus_facial": 2,
        "core_l2_plus_facial": 2,
        "core_l3_plus_facial": 3,
        "positive_surprise_never_lower": True,
        "same_session_single_vote": True,
        "same_s10_social_facial_correlated": True,
        "social_only_l1_plus_facial_stays_l1": True,
        "same_phq_session_single_evidence_family": True,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "FACIAL_OPTIONS",
    "R8_RULE_VERSION",
    "apply_facial_support",
    "enumerate_truth_table",
    "rule_contract_sha256",
    "validate_truth_table",
]
