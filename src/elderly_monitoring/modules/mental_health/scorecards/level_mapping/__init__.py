from __future__ import annotations


def level_from_score(score: float, thresholds: tuple[float, float, float]) -> int:
    level_1, level_2, level_3 = thresholds
    if score >= level_3:
        return 3
    if score >= level_2:
        return 2
    if score >= level_1:
        return 1
    return 0


def trigger_event_for_level(risk_level: int, score_status: str) -> str:
    if score_status == "unavailable":
        return "insufficient_data"
    return {
        0: "normal",
        1: "mild_behavioral_change",
        2: "behavioral_rhythm_deviation",
        3: "persistent_behavioral_risk",
        4: "urgent_manual_review",
    }[risk_level]
