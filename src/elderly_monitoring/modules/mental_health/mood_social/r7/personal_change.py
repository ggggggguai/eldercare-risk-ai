"""Deterministic within-domain change evidence for R7 runtime."""

from __future__ import annotations

from statistics import median
from typing import Callable, Sequence

import numpy as np

from elderly_monitoring.modules.mental_health.mood_social.schemas import MoodSocialDailyFeatures, MoodSocialInferRequest


def _robust_change(recent: Sequence[float], baseline: Sequence[float], *, lower_is_worse: bool) -> dict[str, object]:
    if len(recent) < 7 or len(baseline) < 14:
        return {"available": False, "level": None, "persistent": False, "reason": "stable_28_day_baseline_unavailable"}
    recent_value = float(median(recent))
    baseline_value = float(median(baseline))
    mad = float(median(abs(value - baseline_value) for value in baseline))
    scale = max(1.4826 * mad, abs(baseline_value) * 0.10, 1e-6)
    signed = (baseline_value - recent_value) if lower_is_worse else (recent_value - baseline_value)
    robust_z = signed / scale
    relative = signed / max(abs(baseline_value), 1e-6)
    level = 2 if robust_z >= 3.0 and relative >= 0.30 else 1 if robust_z >= 2.0 and relative >= 0.20 else 0
    return {"available": True, "level": level, "persistent": level > 0, "recent_median": recent_value, "baseline_median": baseline_value, "robust_z": float(robust_z), "relative_change": float(relative)}


def _values(records: Sequence[MoodSocialDailyFeatures], getter: Callable[[MoodSocialDailyFeatures], float | None]) -> list[float]:
    return [float(value) for item in records if (value := getter(item)) is not None]


def runtime_personal_changes(request: MoodSocialInferRequest) -> dict[str, dict[str, object]]:
    records = [*request.history_daily_features, request.current_daily_features]
    recent_records = records[-7:]
    baseline_records = records[:-7][-21:]
    activity_recent = _values(recent_records, lambda item: None if item.activity is None or not item.activity.valid_daytime_detection_minutes else item.activity.daytime_active_minutes)
    activity_base = _values(baseline_records, lambda item: None if item.activity is None or not item.activity.valid_daytime_detection_minutes else item.activity.daytime_active_minutes)
    sleep_recent = _values(recent_records, lambda item: None if item.sleep is None else item.sleep.sleep_efficiency)
    sleep_base = _values(baseline_records, lambda item: None if item.sleep is None else item.sleep.sleep_efficiency)
    social_recent = _values(recent_records, lambda item: None if item.social is None or not item.social.call_log_observed else item.social.connected_duration_minutes)
    social_base = _values(baseline_records, lambda item: None if item.social is None or not item.social.call_log_observed else item.social.connected_duration_minutes)
    return {
        "activity": _robust_change(activity_recent, activity_base, lower_is_worse=True),
        "sleep": _robust_change(sleep_recent, sleep_base, lower_is_worse=True),
        "social": _robust_change(social_recent, social_base, lower_is_worse=True),
    }


def physiology_support(request: MoodSocialInferRequest) -> dict[str, object]:
    records = [*request.history_daily_features, request.current_daily_features]
    recent = _values(records[-7:], lambda item: None if item.physiology is None else item.physiology.heart_rate_mean_bpm)
    baseline = _values(records[:-7][-21:], lambda item: None if item.physiology is None else item.physiology.heart_rate_mean_bpm)
    result = _robust_change(recent, baseline, lower_is_worse=False)
    result["role"] = "support-only-never-independent-vote"
    return result


__all__ = ["physiology_support", "runtime_personal_changes"]
