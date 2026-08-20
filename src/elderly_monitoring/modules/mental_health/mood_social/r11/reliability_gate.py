"""Label-free Reliability Gate v1 for R11 evidence nodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Literal, Mapping


R11_RELIABILITY_VERSION = "mood-social-r11-reliability-gate-v1"
ReliabilityStatus = Literal["usable", "degraded", "abstain"]


@dataclass(frozen=True)
class ReliabilityDecision:
    node: str
    reliability_status: ReliabilityStatus
    domain_compatibility: bool
    quality_reasons: tuple[str, ...]
    original_level: int | None
    effective_level: int | None
    confidence_multiplier: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _level(value: Mapping[str, Any]) -> int | None:
    raw = value.get("domain_level", value.get("level"))
    return None if raw is None else min(max(int(raw), 0), 3)


def evaluate_reliability(
    node: str,
    assessment: Mapping[str, Any] | None,
    *,
    history_age_days: float | None = None,
) -> ReliabilityDecision:
    value = dict(assessment or {})
    original = _level(value)
    reasons: list[str] = []
    status: ReliabilityStatus = "usable"
    compatible = True
    if not value.get("available"):
        return ReliabilityDecision(
            node, "abstain", False, ("node_unavailable",), original, None, 0.0
        )
    if value.get("ood") is True or value.get("ood_status") in {"out_of_distribution", "reject"}:
        reasons.append("ood_rejected")
        status, compatible = "abstain", False
    if value.get("contract_valid") is False or value.get("checksum_valid") is False:
        reasons.append("feature_or_checksum_contract_mismatch")
        status, compatible = "abstain", False
    p5, p10 = value.get("probability_phq_ge5"), value.get("probability_phq_ge10")
    if p5 is not None and p10 is not None and float(p10) > float(p5):
        reasons.append("dual_head_monotonicity_violation")
        status, compatible = "abstain", False
    if node in {"phq_history", "sleep_history"}:
        if history_age_days is None or not 76 <= float(history_age_days) <= 100:
            reasons.append("history_outside_validated_probability_band_76_100")
            status, compatible = "abstain", False
        if value.get("known_at_valid") is False:
            reasons.append("history_known_at_invalid")
            status, compatible = "abstain", False
    confidence = value.get("confidence")
    if status != "abstain" and confidence is not None:
        if float(confidence) < 0.35:
            reasons.append("confidence_below_abstain_threshold")
            status = "abstain"
        elif float(confidence) < 0.60:
            reasons.append("confidence_degraded")
            status = "degraded"
    quality = value.get("quality_score")
    if status != "abstain" and quality is not None:
        if float(quality) < 0.40:
            reasons.append("quality_below_abstain_threshold")
            status = "abstain"
        elif float(quality) < 0.65:
            reasons.append("quality_degraded")
            status = "degraded"
    coverage = value.get("coverage", value.get("feature_coverage"))
    if status != "abstain" and coverage is not None:
        if float(coverage) < 0.40:
            reasons.append("coverage_below_abstain_threshold")
            status = "abstain"
        elif float(coverage) < 0.70:
            reasons.append("coverage_degraded")
            status = "degraded"
    required = {"activity": 7, "sleep": 7, "social": 7}
    valid_count = value.get("valid_days", value.get("valid_nights", value.get("valid_sessions")))
    if status != "abstain" and node in required and valid_count is not None:
        if int(valid_count) < max(3, required[node] // 2):
            reasons.append("insufficient_valid_observations")
            status = "abstain"
        elif int(valid_count) < required[node]:
            reasons.append("limited_valid_observations")
            status = "degraded"
    if status == "usable":
        effective, multiplier = original, 1.0
    elif status == "degraded":
        effective, multiplier = (None if original is None else min(original, 2)), 0.60
    else:
        effective, multiplier = None, 0.0
    if not reasons:
        reasons.append("quality_contract_passed")
    return ReliabilityDecision(
        node, status, compatible, tuple(reasons), original, effective, multiplier
    )


def reliability_contract_sha256() -> str:
    payload = {
        "version": R11_RELIABILITY_VERSION,
        "history_probability_band_days": [76, 100],
        "confidence": {"abstain_below": 0.35, "degraded_below": 0.60},
        "quality": {"abstain_below": 0.40, "degraded_below": 0.65},
        "coverage": {"abstain_below": 0.40, "degraded_below": 0.70},
        "minimum_observations": {"activity": 7, "sleep": 7, "social": 7},
        "can_upgrade_level": False,
        "risk_labels_used_for_gate": False,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "R11_RELIABILITY_VERSION",
    "ReliabilityDecision",
    "evaluate_reliability",
    "reliability_contract_sha256",
]
