"""R9 evidence-signature router with explicit correlation and de-duplication."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from itertools import product
from typing import Iterable


R9_GRAPH_VERSION = "mood-social-r9-evidence-graph-v1"
KNOWN_NODES = (
    "activity",
    "sleep",
    "social",
    "profile",
    "physiology",
    "phq_history",
    "facial_affect",
)


@dataclass(frozen=True)
class EvidenceRoute:
    signature: str
    probability_nodes: tuple[str, ...]
    dynamic_vote_nodes: tuple[str, ...]
    support_nodes: tuple[str, ...]
    background_nodes: tuple[str, ...]
    replaced_nodes: tuple[str, ...]
    correlation_groups: tuple[tuple[str, ...], ...]
    fusion_mode: str

    def to_dict(self) -> dict[str, object]:
        return {
            "signature": self.signature,
            "probability_nodes": list(self.probability_nodes),
            "dynamic_vote_nodes": list(self.dynamic_vote_nodes),
            "support_nodes": list(self.support_nodes),
            "background_nodes": list(self.background_nodes),
            "replaced_nodes": list(self.replaced_nodes),
            "correlation_groups": [list(group) for group in self.correlation_groups],
            "fusion_mode": self.fusion_mode,
            "graph_version": R9_GRAPH_VERSION,
            "graph_contract_sha256": graph_contract_sha256(),
        }


def route_evidence(available: Iterable[str]) -> EvidenceRoute:
    nodes = frozenset(str(item) for item in available)
    unknown = nodes.difference(KNOWN_NODES)
    if unknown:
        raise ValueError(f"unknown R9 evidence nodes: {sorted(unknown)}")
    signature = "+".join(node for node in KNOWN_NODES if node in nodes) or "none"
    probability: list[str] = []
    replaced: list[str] = []
    correlations: list[tuple[str, ...]] = []
    if {"activity", "sleep"}.issubset(nodes):
        probability.append("activity_sleep")
        replaced.extend(("activity", "sleep"))
        correlations.append(("activity_sleep", "activity", "sleep"))
    else:
        probability.extend(node for node in ("activity", "sleep") if node in nodes)
    if {"sleep", "phq_history"}.issubset(nodes):
        probability.append("sleep_history")
        replaced.append("phq_history")
        correlations.append(("sleep_history", "sleep", "phq_history"))
    elif "phq_history" in nodes:
        probability.append("phq_history")
    dynamic = tuple(node for node in ("activity", "sleep", "social") if node in nodes)
    support = tuple(node for node in ("physiology", "facial_affect") if node in nodes)
    background = ("profile",) if "profile" in nodes else ()
    if "facial_affect" in nodes and "social" in nodes:
        correlations.append(("s10_interaction", "social", "facial_affect"))
    if "activity_sleep" in probability and "sleep_history" in probability:
        correlations.append(("shared_sleep", "activity_sleep", "sleep_history"))
    mode = (
        "local_activity_sleep_and_sleep_history"
        if {"activity_sleep", "sleep_history"}.issubset(probability)
        else "local_activity_sleep"
        if "activity_sleep" in probability
        else "local_sleep_history"
        if "sleep_history" in probability
        else "independent_experts"
    )
    return EvidenceRoute(
        signature=signature,
        probability_nodes=tuple(probability),
        dynamic_vote_nodes=dynamic,
        support_nodes=support,
        background_nodes=background,
        replaced_nodes=tuple(dict.fromkeys(replaced)),
        correlation_groups=tuple(correlations),
        fusion_mode=mode,
    )


def enumerate_signature_routes() -> list[dict[str, object]]:
    return [
        route_evidence(node for node, flag in zip(KNOWN_NODES, bits) if flag).to_dict()
        for bits in product((False, True), repeat=len(KNOWN_NODES))
    ]


def validate_signature_routes() -> dict[str, object]:
    violations: list[str] = []
    rows = enumerate_signature_routes()
    for row in rows:
        available = set(str(row["signature"]).split("+"))
        probability = set(row["probability_nodes"])
        dynamic = list(row["dynamic_vote_nodes"])
        if "activity_sleep" in probability and ({"activity", "sleep"} & probability):
            violations.append("A+S joint node duplicated an individual probability node")
        if "sleep_history" in probability and "phq_history" in probability:
            violations.append("S+H joint node duplicated the history probability node")
        if len(dynamic) != len(set(dynamic)):
            violations.append("dynamic vote duplicated")
        if "none" not in available and "profile" in available and "profile" in dynamic:
            violations.append("profile became a dynamic vote")
        if "facial_affect" in dynamic or "physiology" in dynamic:
            violations.append("support evidence became a dynamic vote")
    return {
        "status": "pass" if not violations else "fail",
        "row_count": len(rows),
        "violation_count": len(violations),
        "violations": violations[:100],
    }


def graph_contract_sha256() -> str:
    payload = {
        "version": R9_GRAPH_VERSION,
        "known_nodes": KNOWN_NODES,
        "activity_sleep_replaces_probability_nodes": ["activity", "sleep"],
        "sleep_history_replaces_probability_node": "phq_history",
        "joint_nodes_never_add_dynamic_votes": True,
        "profile_role": "background-only",
        "physiology_role": "support-only",
        "facial_role": "support-only",
        "same_s10_social_facial_correlated": True,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


__all__ = [
    "EvidenceRoute",
    "KNOWN_NODES",
    "R9_GRAPH_VERSION",
    "enumerate_signature_routes",
    "graph_contract_sha256",
    "route_evidence",
    "validate_signature_routes",
]
