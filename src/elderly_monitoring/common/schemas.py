from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ModuleName = Literal["fall_risk", "mental_health", "fusion"]
RecommendedAction = Literal[
    "record_only",
    "observe",
    "remind_user",
    "notify_guardian",
    "emergency_alert",
    "manual_review",
]


@dataclass(frozen=True)
class EvidenceWindow:
    start_time: float | None = None
    end_time: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass(frozen=True)
class AlgorithmEvent:
    module: ModuleName
    person_id: str
    timestamp: str
    risk_level: int
    risk_score: float
    confidence: float
    trigger_event: str
    risk_factors: list[str]
    recommended_action: RecommendedAction
    model_version: str
    device_id: str | None = None
    scene_region: str | None = None
    evidence_window: EvidenceWindow | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.evidence_window is None:
            payload["evidence_window"] = None
        else:
            payload["evidence_window"] = self.evidence_window.to_dict()
        return payload


def action_for_level(risk_level: int, *, mental_health: bool = False) -> RecommendedAction:
    if risk_level <= 0:
        return "record_only"
    if risk_level == 1:
        return "observe"
    if risk_level == 2:
        return "remind_user"
    if risk_level == 3:
        return "manual_review" if mental_health else "notify_guardian"
    return "manual_review" if mental_health else "emergency_alert"
