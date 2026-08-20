"""R11 EvidenceOrchestrator v3 with mutually-exclusive probability paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json
from typing import Any, Mapping

from .reliability_gate import ReliabilityDecision, evaluate_reliability


R11_GRAPH_VERSION = "mood-social-r11-evidence-orchestrator-v3"
BASE_NODES = (
    "activity",
    "sleep",
    "social",
    "profile",
    "physiology",
    "phq_history",
    "facial_affect",
)


@dataclass(frozen=True)
class OrchestrationResult:
    evidence_signature: str
    selected_probability_node: str | None
    probability_used_sources: tuple[str, ...]
    consumed_nodes: tuple[str, ...]
    supporting_nodes: tuple[str, ...]
    suppressed_nodes: tuple[str, ...]
    correlation_clusters: tuple[tuple[str, ...], ...]
    fusion_mode: str
    reliability: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["graph_version"] = R11_GRAPH_VERSION
        value["graph_contract_sha256"] = graph_contract_sha256()
        return value


def orchestrate_evidence(
    experts: Mapping[str, Mapping[str, Any]],
    joints: Mapping[str, Mapping[str, Any]],
    *,
    history_age_days: float | None,
) -> OrchestrationResult:
    decisions: dict[str, ReliabilityDecision] = {}
    for node in BASE_NODES:
        decisions[node] = evaluate_reliability(
            node, experts.get(node), history_age_days=history_age_days
        )
    decisions["activity_sleep"] = evaluate_reliability(
        "activity_sleep", joints.get("activity_sleep"), history_age_days=history_age_days
    )
    decisions["sleep_history"] = evaluate_reliability(
        "sleep_history", joints.get("sleep_history"), history_age_days=history_age_days
    )
    usable = {name for name, value in decisions.items() if value.reliability_status != "abstain"}
    selected: str | None = None
    consumed: tuple[str, ...] = ()
    suppressed: list[str] = []
    probability: list[str] = []
    # R9's S+History node is allowed only in the R11 validated 76-100 day band.
    if {"sleep", "phq_history", "sleep_history"}.issubset(usable):
        selected = "sleep_history_r9"
        consumed = ("sleep", "phq_history")
        probability = [selected]
        if {"activity", "activity_sleep"}.issubset(usable):
            suppressed.append("activity_sleep_r9_shared_sleep")
    elif {"activity", "sleep", "activity_sleep"}.issubset(usable):
        selected = "activity_sleep_r9"
        consumed = ("activity", "sleep")
        probability = [selected]
    else:
        probability = [node for node in ("activity", "sleep", "phq_history") if node in usable]
    support = [node for node in ("activity", "social", "facial_affect", "physiology") if node in usable and node not in consumed]
    clusters: list[tuple[str, ...]] = []
    if "social" in support and "facial_affect" in support:
        clusters.append(("s10_interaction", "social", "facial_affect"))
    signature = "+".join(node for node in BASE_NODES if node in usable) or "none"
    mode = (
        "mutually_exclusive_sleep_history"
        if selected == "sleep_history_r9"
        else "mutually_exclusive_activity_sleep"
        if selected == "activity_sleep_r9"
        else "independent_fallback"
    )
    return OrchestrationResult(
        signature,
        selected,
        tuple(probability),
        consumed,
        tuple(support),
        tuple(suppressed),
        tuple(clusters),
        mode,
        {name: value.to_dict() for name, value in decisions.items()},
    )


def enumerate_routes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bits in product((False, True), repeat=len(BASE_NODES)):
        experts = {
            node: {"available": flag, "domain_level": 1, "confidence": 1.0}
            for node, flag in zip(BASE_NODES, bits)
        }
        for activity_sleep, sleep_history, age_band in product((False, True), (False, True), (50, 90)):
            joints = {
                "activity_sleep": {"available": activity_sleep, "domain_level": 1},
                "sleep_history": {"available": sleep_history, "domain_level": 1},
            }
            rows.append(
                orchestrate_evidence(experts, joints, history_age_days=age_band).to_dict()
            )
    return rows


def validate_routes() -> dict[str, Any]:
    rows = enumerate_routes()
    violations: list[str] = []
    for row in rows:
        consumed = set(row["consumed_nodes"])
        probability = set(row["probability_used_sources"])
        if consumed & probability:
            violations.append("member node consumed twice")
        if row["selected_probability_node"] == "sleep_history_r9" and consumed != {"sleep", "phq_history"}:
            violations.append("S+History failed to consume exactly sleep and history")
        if row["selected_probability_node"] == "activity_sleep_r9" and consumed != {"activity", "sleep"}:
            violations.append("A+S failed to consume exactly activity and sleep")
        if "sleep_history_r9" in probability and "activity_sleep_r9" in probability:
            violations.append("correlated joint nodes selected together")
    return {
        "status": "pass" if not violations else "fail",
        "row_count": len(rows),
        "violation_count": len(violations),
        "violations": violations[:100],
    }


def graph_contract_sha256() -> str:
    payload = {
        "version": R11_GRAPH_VERSION,
        "priority": ["sleep_history_r9", "activity_sleep_r9", "independent_fallback"],
        "sleep_consumed_once": True,
        "history_probability_band": [76, 100],
        "social_facial_cluster_max_one": True,
        "profile": "background-only",
        "physiology": "support-only",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "R11_GRAPH_VERSION",
    "OrchestrationResult",
    "enumerate_routes",
    "graph_contract_sha256",
    "orchestrate_evidence",
    "validate_routes",
]
