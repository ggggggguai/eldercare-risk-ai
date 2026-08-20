"""Deterministic R11 rule fusion v5 and fixed one-step hysteresis."""

from __future__ import annotations

import hashlib
from itertools import product
import json
from typing import Any, Mapping, Sequence

from elderly_monitoring.modules.mental_health.mood_social.r7.rule_fusion import (
    ATTENTION_ANCHORS,
)

from .evidence_orchestrator import OrchestrationResult, graph_contract_sha256


R11_RULE_VERSION = "mood-social-r11-rule-fusion-v5"


def _level(value: Mapping[str, Any] | None, reliability: Mapping[str, Any]) -> int | None:
    if not value or not value.get("available") or reliability.get("reliability_status") == "abstain":
        return None
    raw = reliability.get("effective_level")
    if raw is None:
        raw = value.get("domain_level", value.get("level"))
    return None if raw is None else min(max(int(raw), 0), 3)


def _prior_level(history_attention_indices: Sequence[Mapping[str, Any]]) -> int | None:
    if not history_attention_indices:
        return None
    raw = float(history_attention_indices[-1]["attention_index"])
    value = raw * 100.0 if raw <= 1.0 else raw
    return min(ATTENTION_ANCHORS, key=lambda level: abs(ATTENTION_ANCHORS[level] - value))


def _stabilize(level: int | None, previous: int | None) -> tuple[int | None, str]:
    if level is None or previous is None:
        return level, "no_prior_hysteresis"
    if level > previous + 1:
        return previous + 1, "upward_step_capped_one_level"
    if level < previous - 1:
        return previous - 1, "downward_step_capped_one_level"
    return level, "within_hysteresis_band"


def fuse_r11_current_state(
    experts: Mapping[str, Mapping[str, Any]],
    joints: Mapping[str, Mapping[str, Any]],
    route: OrchestrationResult,
    *,
    history_attention_indices: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    reliability = route.reliability
    selected = route.selected_probability_node
    reasons: list[str] = []
    discordance: list[str] = []
    levels = {
        node: _level(experts.get(node), reliability[node])
        for node in ("activity", "sleep", "phq_history")
    }
    usable = [value for value in levels.values() if value is not None]
    high = [value for value in usable if value >= 2]
    abnormal = [value for value in usable if value >= 1]
    if not usable:
        independent_base = None
    elif len(high) >= 2 or (high and len(abnormal) >= 2):
        independent_base = 3
    elif high or len(abnormal) >= 2:
        independent_base = 2
    elif abnormal:
        independent_base = 1
    else:
        independent_base = 0
    # A joint node is operational evidence only when both of its members are
    # present.  This keeps route changes monotonic: deleting a member can
    # remove a joint floor, but can never expose a higher score.
    activity_sleep_level = (
        _level(joints.get("activity_sleep"), reliability["activity_sleep"])
        if levels["activity"] is not None and levels["sleep"] is not None
        else None
    )
    sleep_history_level = (
        _level(joints.get("sleep_history"), reliability["sleep_history"])
        if levels["sleep"] is not None and levels["phq_history"] is not None
        else None
    )
    joint_floor = max(
        [
            value
            for value in (sleep_history_level, activity_sleep_level)
            if value is not None
        ],
        default=None,
    )
    # The unique selected node owns the PHQ probability output.  A max-only
    # safety floor across already observed evidence prevents deletion from
    # increasing operational attention without adding a second vote/weight.
    base = max(
        [value for value in (independent_base, joint_floor) if value is not None],
        default=None,
    )
    if selected == "sleep_history_r9":
        reasons.append("selected_sleep_history_r9_probability_path")
    elif selected == "activity_sleep_r9":
        reasons.append("selected_activity_sleep_r9_probability_path")
    else:
        reasons.append("independent_probability_fallback")
    if joint_floor is not None:
        reasons.append("max_only_correlated_safety_floor_no_double_vote")
    activity_level = levels["activity"]
    social_level = _level(experts.get("social"), reliability["social"])
    facial = experts.get("facial_affect", {})
    facial_level = _level(facial, reliability["facial_affect"])
    facial_votes = bool(
        facial_level is not None
        and facial.get("supports_rule_vote")
        and facial.get("evidence_label") in {"F1", "F2"}
    )
    s10_cluster = max(social_level or 0, 1 if facial_votes else 0)
    if base is not None and base >= 1 and s10_cluster >= 2 and base < 3:
        base += 1
        reasons.append("correlated_s10_support_cluster_max_plus_one")
    elif s10_cluster and (base is None or base == 0):
        base = 1
        reasons.append("support_only_capped_l1")
    history_level = _level(experts.get("phq_history"), reliability["phq_history"])
    dynamic_level = max(
        [value for value in (activity_level, _level(experts.get("sleep"), reliability["sleep"]), social_level) if value is not None],
        default=None,
    )
    if history_level == 0 and dynamic_level is not None and dynamic_level >= 1:
        discordance.append("possible_new_onset_history_low_did_not_downrank")
    if history_level is not None and history_level >= 2 and (dynamic_level or 0) == 0:
        discordance.append("historical_high_dynamic_normal_retest_recommended")
    if facial.get("prediction") in {"positive", "surprise"}:
        discordance.append("facial_positive_or_surprise_never_downranks")
    previous = _prior_level(history_attention_indices)
    stabilized, hysteresis_reason = _stabilize(base, previous)
    if hysteresis_reason != "within_hysteresis_band":
        reasons.append(hysteresis_reason)
    persistent = False
    if stabilized is not None and stabilized >= 1 and len(history_attention_indices) >= 2:
        recent = [_prior_level([value]) for value in history_attention_indices[-2:]]
        persistent = all(value is not None and value >= stabilized for value in recent)
    state = {
        "available": stabilized is not None,
        "level": stabilized,
        "attention_index": None if stabilized is None else ATTENTION_ANCHORS[stabilized],
        "attention_index_semantics": "rule-derived-not-PHQ-probability",
        "observation_status": "persistent_concern" if persistent else "observation",
        "decision_reasons": reasons,
        "discordance_flags": discordance,
    }
    return {
        "operational_attention_state": state,
        "rule_fusion": {
            "rule_version": R11_RULE_VERSION,
            "rule_contract_sha256": rule_contract_sha256(),
            "graph_contract_sha256": graph_contract_sha256(),
            "selected_probability_node": selected,
            "consumed_nodes": list(route.consumed_nodes),
            "supporting_nodes": list(route.supporting_nodes),
            "suppressed_nodes": list(route.suppressed_nodes),
            "decision_reasons": reasons,
            "discordance_flags": discordance,
            "attention_index_is_phq_probability": False,
            "support_only_can_create_l2_l3": False,
        },
    }


def enumerate_truth_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    options: tuple[int | None, ...] = (None, 0, 1, 2, 3)
    from .evidence_orchestrator import orchestrate_evidence

    for activity, sleep, social, history, facial in product(options, options, options, options, (False, True)):
        experts = {
            name: {"available": level is not None, "domain_level": level, "confidence": 1.0}
            for name, level in (
                ("activity", activity),
                ("sleep", sleep),
                ("social", social),
                ("phq_history", history),
            )
        }
        experts.update(
            {
                "profile": {"available": False},
                "physiology": {"available": False},
                "facial_affect": {
                    "available": facial,
                    "domain_level": 1 if facial else None,
                    "supports_rule_vote": facial,
                    "evidence_label": "F1" if facial else "unavailable",
                    "prediction": "negative" if facial else None,
                },
            }
        )
        for activity_sleep, sleep_history in product(options, options):
            joints = {
                "activity_sleep": {"available": activity_sleep is not None, "domain_level": activity_sleep},
                "sleep_history": {"available": sleep_history is not None, "domain_level": sleep_history},
            }
            route = orchestrate_evidence(experts, joints, history_age_days=90)
            result = fuse_r11_current_state(experts, joints, route)
            rows.append(
                {
                    "activity": activity,
                    "sleep": sleep,
                    "social": social,
                    "history": history,
                    "facial": facial,
                    "activity_sleep": activity_sleep,
                    "sleep_history": sleep_history,
                    "selected": route.selected_probability_node,
                    "level": result["operational_attention_state"]["level"],
                    "consumed": list(route.consumed_nodes),
                }
            )
    return rows


def validate_truth_table() -> dict[str, Any]:
    rows = enumerate_truth_table()
    violations: list[str] = []
    names = (
        "activity",
        "sleep",
        "social",
        "history",
        "facial",
        "activity_sleep",
        "sleep_history",
    )
    lookup = {tuple(row[name] for name in names): row for row in rows}
    for row in rows:
        if row["facial"] and all(
            row[name] is None
            for name in (
                "activity",
                "sleep",
                "social",
                "history",
                "activity_sleep",
                "sleep_history",
            )
        ):
            if row["level"] not in {None, 1}:
                violations.append("facial support alone exceeded L1")
        if row["selected"] == "sleep_history_r9" and set(row["consumed"]) != {"sleep", "phq_history"}:
            violations.append("sleep/history was not consumed exactly once")
        if row["selected"] == "activity_sleep_r9" and set(row["consumed"]) != {"activity", "sleep"}:
            violations.append("activity/sleep was not consumed exactly once")
        current_level = -1 if row["level"] is None else int(row["level"])
        key = [row[name] for name in names]
        for index, name in enumerate(names):
            if name == "facial":
                continue
            if key[index] is not None:
                removed = list(key)
                removed[index] = None
                removed_level = lookup[tuple(removed)]["level"]
                if (-1 if removed_level is None else int(removed_level)) > current_level:
                    violations.append(f"removing {name} upgraded attention")
            else:
                # Only abnormal evidence (L1+) is "concordant" support.
                for added_level in (1, 2, 3):
                    added = list(key)
                    added[index] = added_level
                    added_value = lookup[tuple(added)]["level"]
                    if (-1 if added_value is None else int(added_value)) < current_level:
                        violations.append(f"adding concordant {name} downgraded attention")
        if not row["facial"]:
            added_facial = list(key)
            added_facial[4] = True
            added_level = lookup[tuple(added_facial)]["level"]
            if (-1 if added_level is None else int(added_level)) < current_level:
                violations.append("adding facial support downgraded attention")
    return {
        "status": "pass" if not violations else "fail",
        "row_count": len(rows),
        "violation_count": len(violations),
        "violations": violations[:100],
    }


def rule_contract_sha256() -> str:
    payload = {
        "version": R11_RULE_VERSION,
        "hysteresis_max_daily_level_change": 1,
        "support_max_increment": 1,
        "operational_safety_floor": "max-only/no-double-vote",
        "removing_evidence_can_upgrade": False,
        "adding_concordant_evidence_can_downgrade": False,
        "support_only_max_level": 1,
        "positive_or_surprise_downrank": False,
        "history_low_downrank": False,
        "attention_semantics": "rule-derived-not-PHQ-probability",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "R11_RULE_VERSION",
    "enumerate_truth_table",
    "fuse_r11_current_state",
    "rule_contract_sha256",
    "validate_truth_table",
]
