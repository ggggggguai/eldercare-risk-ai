"""R10 mutually-exclusive evidence graph router."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import json
from typing import Iterable


R10_GRAPH_VERSION = "mood-social-r10-evidence-graph-v2"
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
class EvidenceRouteV2:
    signature: str
    selected_probability_node: str | None
    consumed_probability_nodes: tuple[str, ...]
    independent_probability_nodes: tuple[str, ...]
    supporting_nodes: tuple[str, ...]
    background_nodes: tuple[str, ...]
    correlation_clusters: tuple[tuple[str, ...], ...]
    fallback_reasons: tuple[str, ...]
    fusion_mode: str

    def to_dict(self) -> dict[str, object]:
        return {
            "signature": self.signature,
            "selected_probability_node": self.selected_probability_node,
            "consumed_probability_nodes": list(self.consumed_probability_nodes),
            "independent_probability_nodes": list(self.independent_probability_nodes),
            "supporting_nodes": list(self.supporting_nodes),
            "background_nodes": list(self.background_nodes),
            "correlation_clusters": [list(group) for group in self.correlation_clusters],
            "fallback_reasons": list(self.fallback_reasons),
            "fusion_mode": self.fusion_mode,
            "graph_version": R10_GRAPH_VERSION,
            "graph_contract_sha256": graph_contract_sha256(),
        }


def route_evidence_v2(
    available: Iterable[str], *, activity_sleep_joint: bool, sleep_history_joint: bool
) -> EvidenceRouteV2:
    nodes = frozenset(str(value) for value in available)
    unknown = nodes.difference(BASE_NODES)
    if unknown:
        raise ValueError(f"unknown R10 evidence nodes: {sorted(unknown)}")
    as_eligible = activity_sleep_joint and {"activity", "sleep"}.issubset(nodes)
    sh_eligible = sleep_history_joint and {"sleep", "phq_history"}.issubset(nodes)
    selected: str | None = None
    consumed: tuple[str, ...] = ()
    support: list[str] = []
    independent: list[str] = []
    fallback: list[str] = []
    correlations: list[tuple[str, ...]] = []
    if sh_eligible:
        selected = "sleep_history_v2"
        consumed = ("sleep", "phq_history")
        if "activity" in nodes:
            support.append("activity")
        if as_eligible:
            fallback.append("activity_sleep_suppressed_shared_sleep")
        correlations.append(("sleep_history_v2", "sleep", "phq_history"))
    elif as_eligible:
        selected = "activity_sleep_v2"
        consumed = ("activity", "sleep")
        correlations.append(("activity_sleep_v2", "activity", "sleep"))
    else:
        independent.extend(node for node in ("activity", "sleep", "phq_history") if node in nodes)
        if activity_sleep_joint:
            fallback.append("activity_sleep_signature_incomplete")
        if sleep_history_joint:
            fallback.append("sleep_history_signature_incomplete")
    if "social" in nodes:
        support.append("social")
    if "facial_affect" in nodes:
        support.append("facial_affect")
    if "physiology" in nodes:
        support.append("physiology")
    if "social" in nodes and "facial_affect" in nodes:
        correlations.append(("s10_interaction", "social", "facial_affect"))
    background = ("profile",) if "profile" in nodes else ()
    mode = (
        "mutually_exclusive_sleep_history"
        if selected == "sleep_history_v2"
        else "mutually_exclusive_activity_sleep"
        if selected == "activity_sleep_v2"
        else "independent_fallback"
    )
    return EvidenceRouteV2(
        signature="+".join(node for node in BASE_NODES if node in nodes) or "none",
        selected_probability_node=selected,
        consumed_probability_nodes=consumed,
        independent_probability_nodes=tuple(independent),
        supporting_nodes=tuple(dict.fromkeys(support)),
        background_nodes=background,
        correlation_clusters=tuple(correlations),
        fallback_reasons=tuple(fallback),
        fusion_mode=mode,
    )


def enumerate_signature_routes() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bits in product((False, True), repeat=len(BASE_NODES)):
        available = [node for node, flag in zip(BASE_NODES, bits) if flag]
        for as_joint, sh_joint in product((False, True), repeat=2):
            rows.append(
                route_evidence_v2(
                    available,
                    activity_sleep_joint=as_joint,
                    sleep_history_joint=sh_joint,
                ).to_dict()
            )
    return rows


def validate_signature_routes() -> dict[str, object]:
    violations: list[str] = []
    for row in enumerate_signature_routes():
        selected = row["selected_probability_node"]
        consumed = set(row["consumed_probability_nodes"])
        independent = set(row["independent_probability_nodes"])
        if consumed & independent:
            violations.append("consumed probability node remained independent")
        if selected == "sleep_history_v2" and "sleep" not in consumed:
            violations.append("S+History failed to consume sleep")
        if selected == "sleep_history_v2" and row["fusion_mode"] != "mutually_exclusive_sleep_history":
            violations.append("S+History route mode drift")
        if selected == "activity_sleep_v2" and consumed != {"activity", "sleep"}:
            violations.append("A+S did not replace both component nodes")
        if {"social", "facial_affect"}.issubset(set(row["supporting_nodes"])):
            clusters = [set(value) for value in row["correlation_clusters"]]
            if not any({"social", "facial_affect"}.issubset(group) for group in clusters):
                violations.append("S10 social/facial correlation cluster absent")
    return {
        "status": "pass" if not violations else "fail",
        "row_count": len(enumerate_signature_routes()),
        "violation_count": len(violations),
        "violations": violations[:100],
    }


def graph_contract_sha256() -> str:
    payload = {
        "version": R10_GRAPH_VERSION,
        "priority": ["sleep_history_v2", "activity_sleep_v2", "independent_fallback"],
        "sleep_consumed_once": True,
        "activity_support_allowed_with_sleep_history": True,
        "social_facial_same_s10_cluster_max_one": True,
        "profile": "background-only",
        "physiology": "support-only",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "BASE_NODES",
    "EvidenceRouteV2",
    "enumerate_signature_routes",
    "graph_contract_sha256",
    "route_evidence_v2",
    "validate_signature_routes",
]
