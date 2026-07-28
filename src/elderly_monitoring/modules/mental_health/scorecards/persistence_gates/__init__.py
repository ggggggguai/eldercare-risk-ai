from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PersistenceCaps:
    coverage_below_0_40: int
    coverage_below_0_60: int
    initial_baseline_not_ready: int
    stable_baseline_not_ready: int
    persistent_days_below_minimum: int


def applicable_level_caps(
    *,
    feature_coverage: float,
    initial_ready: bool,
    stable_ready: bool,
    persistent_days: int,
    min_persistent_days_for_level_3: int,
    caps: Any,
) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    if feature_coverage < 0.40:
        applied.append({"reason": "coverage_below_0_40", "max_level": caps.coverage_below_0_40})
    elif feature_coverage < 0.60:
        applied.append({"reason": "coverage_below_0_60", "max_level": caps.coverage_below_0_60})
    if not initial_ready:
        applied.append({"reason": "initial_baseline_not_ready", "max_level": caps.initial_baseline_not_ready})
    elif not stable_ready:
        applied.append({"reason": "stable_baseline_not_ready", "max_level": caps.stable_baseline_not_ready})
    if persistent_days < min_persistent_days_for_level_3:
        applied.append(
            {
                "reason": "persistent_days_below_minimum",
                "max_level": caps.persistent_days_below_minimum,
            }
        )
    return applied


def cap_level(candidate_level: int, *, passive_max_level: int, applied_caps: list[dict[str, Any]]) -> int:
    return min(
        candidate_level,
        passive_max_level,
        *(item["max_level"] for item in applied_caps),
    )


def baseline_confidence(
    *,
    baseline_quality: float,
    initial_ready: bool,
    stable_ready: bool,
) -> str:
    if not initial_ready or baseline_quality < 0.35:
        return "low"
    if not stable_ready or baseline_quality < 0.75:
        return "medium"
    return "high"
