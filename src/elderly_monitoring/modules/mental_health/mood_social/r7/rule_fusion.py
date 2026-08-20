"""Frozen, deterministic, non-trained R7 current-state rule fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable, Literal, Mapping


DYNAMIC_DOMAINS = ("activity", "sleep", "social")
ATTENTION_ANCHORS = {0: 15, 1: 40, 2: 65, 3: 90}


@dataclass(frozen=True)
class DomainEvidence:
    domain: Literal["activity", "sleep", "social"]
    available: bool
    level: int | None
    confidence: float
    persistent: bool = False
    personal_change_level: int | None = None

    def effective_level(self) -> int | None:
        if not self.available or self.level is None:
            return None
        base = min(max(int(self.level), 0), 3)
        personal = 0 if self.personal_change_level is None else min(max(int(self.personal_change_level), 0), 2)
        if self.persistent and personal > base:
            return min(base + 1, 3)
        return base


@dataclass(frozen=True)
class HistoryEvidence:
    available: bool
    level: int | None
    age_days: int | None
    high: bool = False
    low: bool = False

    @property
    def vote_eligible(self) -> bool:
        return bool(self.available and self.age_days is not None and self.age_days <= 90)


def _state(level: int | None, reason: str | None = None) -> dict[str, object]:
    if level is None:
        return {"available": False, "level": None, "attention_index": None, "attention_index_semantics": "rule-derived-not-PHQ-probability", "reason": reason or "insufficient_dynamic_evidence"}
    return {"available": True, "level": int(level), "attention_index": ATTENTION_ANCHORS[int(level)], "attention_index_semantics": "rule-derived-not-PHQ-probability", "reason": reason}


def fuse_current_state(
    dynamic: Iterable[DomainEvidence],
    *,
    history: HistoryEvidence | None = None,
    profile_background: Mapping[str, object] | None = None,
    physiology_support: Mapping[str, object] | None = None,
) -> dict[str, object]:
    evidence = list(dynamic)
    if len({item.domain for item in evidence}) != len(evidence):
        raise ValueError("each dynamic domain may vote at most once")
    levels = {item.domain: item.effective_level() for item in evidence if item.effective_level() is not None}
    if not levels:
        passive_level: int | None = None
    else:
        values = list(levels.values())
        abnormal = [value for value in values if value >= 1]
        high = [value for value in values if value >= 2]
        if len(abnormal) >= 2 and high:
            passive_level = 3
        elif high or sum(value >= 1 for value in values) >= 2:
            passive_level = 2
        elif abnormal:
            passive_level = 1
        else:
            passive_level = 0
    passive = _state(passive_level)
    history_level = passive_level
    flags: list[str] = []
    reasons = [f"{domain}:L{level}" for domain, level in sorted(levels.items())]
    if history and history.available:
        if history.age_days is not None and history.age_days <= 13 and history.level is not None:
            history_level = max(history_level or 0, min(int(history.level), 3))
            reasons.append("recent_screening_continuity")
        elif history.vote_eligible and history.high:
            if passive_level is None:
                history_level = min(int(history.level or 1), 2)
            elif passive_level == 0:
                history_level = 1
                flags.append("historical_high_dynamic_normal_retest")
            else:
                history_level = min(passive_level + 1, 3)
        if history.low and passive_level is not None and passive_level >= 1:
            flags.append("possible_new_onset_change")
            history_level = passive_level
        if history.age_days is not None and history.age_days > 90:
            history_level = passive_level
            flags.append("historical_phq_too_old_for_vote")
    history_state = _state(history_level, "history_only_current_state" if passive_level is None and history_level is not None else None)
    return {
        "passive_current_state": passive,
        "history_informed_current_state": history_state,
        "rule_fusion": {
            "triggered_domains": sorted(levels),
            "decision_reasons": reasons,
            "discordance_flags": flags,
            "profile_background": dict(profile_background or {}),
            "physiology_support": dict(physiology_support or {}),
            "rule_version": "mood-social-r7-rule-fusion-v1",
        },
    }


def enumerate_truth_table() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    options: tuple[int | None, ...] = (None, 0, 1, 2, 3)
    for activity in options:
        for sleep in options:
            for social in options:
                evidence = [
                    DomainEvidence(domain, level is not None, level, 1.0)
                    for domain, level in (("activity", activity), ("sleep", sleep), ("social", social))
                    if level is not None
                ]
                result = fuse_current_state(evidence)
                rows.append({
                    "activity": activity,
                    "sleep": sleep,
                    "social": social,
                    "passive_level": result["passive_current_state"]["level"],
                    "attention_index": result["passive_current_state"]["attention_index"],
                })
    return rows


def rule_contract_sha256() -> str:
    payload = {
        "dynamic_domains": DYNAMIC_DOMAINS,
        "anchors": ATTENTION_ANCHORS,
        "single_high_max": 2,
        "two_abnormal_one_high": 3,
        "history_max_age_days": 90,
        "profile_physiology_never_vote": True,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


__all__ = ["ATTENTION_ANCHORS", "DYNAMIC_DOMAINS", "DomainEvidence", "HistoryEvidence", "enumerate_truth_table", "fuse_current_state", "rule_contract_sha256"]
