"""Runtime-identical 1/3/7/14/28-day feature generation for r3.

The public canonicals do not contain raw day histories for every source.  This
module therefore keeps two contracts explicit: exact runtime window features,
and a parity map which only promotes fields whose training-side semantics can
be proven.  It never manufactures shorter or longer windows from a canonical
seven-day snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
from typing import Any, Mapping, Sequence

import numpy as np

from elderly_monitoring.modules.mental_health.mood_social.config import (
    MoodSocialConfig,
    load_mood_social_config,
)
from elderly_monitoring.modules.mental_health.mood_social.feature_mapper import (
    _aggregate_activity,
    _aggregate_sleep,
    map_mood_social_trend_days,
)
from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    DailyMappedFeatures,
)
from elderly_monitoring.modules.mental_health.mood_social.schemas import (
    MoodSocialInferRequest,
)


WINDOW_DAYS = (1, 3, 7, 14, 28)
TEMPORAL_SCHEMA_VERSION = "mood-social-v3.3.3-r3-temporal-v1"
DAILY_DOMAINS = ("activity", "sleep")
SUMMARY_STATISTICS = (
    "current",
    "mean",
    "median",
    "std",
    "iqr",
    "range",
    "theil_sen_slope",
    "change_point_strength",
    "recent_long_term_delta",
    "baseline_z",
    "baseline_relative_change",
    "baseline_percentile",
)


@dataclass(frozen=True)
class MultiWindowFeatures:
    schema_version: str
    target_date: date
    values: Mapping[str, float | int | None]
    feature_mask: Mapping[str, int]
    training_parity: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.schema_version != TEMPORAL_SCHEMA_VERSION:
            raise ValueError("unsupported r3 temporal schema")
        if set(self.values) != set(self.feature_mask):
            raise ValueError("r3 temporal values and masks differ")
        for name, mask in self.feature_mask.items():
            if mask not in (0, 1):
                raise ValueError(f"invalid temporal mask for {name}")
            value = self.values[name]
            if mask == 1 and (value is None or not math.isfinite(float(value))):
                raise ValueError(f"available temporal value is invalid: {name}")
            if mask == 0 and value is not None:
                raise ValueError(f"unavailable temporal value must be null: {name}")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_date": self.target_date.isoformat(),
            "window_days": list(WINDOW_DAYS),
            "values": dict(self.values),
            "feature_mask": dict(self.feature_mask),
            "training_parity": dict(self.training_parity),
        }


def _observations(
    daily: Sequence[DailyMappedFeatures], domain: str, feature: str
) -> tuple[np.ndarray, np.ndarray]:
    offsets: list[float] = []
    values: list[float] = []
    first = daily[0].date
    for item in daily:
        vector = getattr(item, domain)
        value = vector.value(feature)
        if vector.mask(feature) and value is not None:
            offsets.append(float((item.date - first).days))
            values.append(float(value))
    return np.asarray(offsets, dtype=float), np.asarray(values, dtype=float)


def _theil_sen(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(y) < 3:
        return None
    slopes = [
        (y[right] - y[left]) / (x[right] - x[left])
        for left in range(len(y))
        for right in range(left + 1, len(y))
        if x[right] != x[left]
    ]
    return float(np.median(slopes)) if slopes else None


def _change_point_strength(y: np.ndarray) -> float | None:
    if len(y) < 4:
        return None
    scale = float(np.std(y, ddof=0))
    if scale <= 1.0e-12:
        return 0.0
    candidates = [
        abs(float(y[:split].mean() - y[split:].mean())) / scale
        for split in range(2, len(y) - 1)
    ]
    return float(max(candidates)) if candidates else None


def _baseline_statistics(
    current: np.ndarray, baseline: np.ndarray
) -> tuple[float | None, float | None, float | None]:
    if len(current) == 0 or len(baseline) < 3:
        return None, None, None
    current_level = float(np.median(current))
    baseline_level = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - baseline_level)))
    robust_scale = max(1.4826 * mad, 1.0e-6)
    z = (current_level - baseline_level) / robust_scale
    relative = (current_level - baseline_level) / max(abs(baseline_level), 1.0e-6)
    percentile = (float(np.sum(baseline <= current_level)) + 0.5) / (
        float(len(baseline)) + 1.0
    )
    return float(z), float(relative), float(percentile)


def _summary(
    x: np.ndarray,
    y: np.ndarray,
    baseline_y: np.ndarray,
) -> dict[str, float | None]:
    if len(y) == 0:
        return {name: None for name in SUMMARY_STATISTICS}
    q25, q75 = np.quantile(y, [0.25, 0.75])
    prior = baseline_y[-len(y) :] if len(baseline_y) >= len(y) else baseline_y
    recent_long_term = (
        float(y.mean() - prior.mean()) if len(prior) >= max(1, len(y) // 2) else None
    )
    baseline_z, relative, percentile = _baseline_statistics(y, baseline_y)
    return {
        "current": float(y[-1]),
        "mean": float(y.mean()),
        "median": float(np.median(y)),
        "std": float(np.std(y, ddof=0)),
        "iqr": float(q75 - q25),
        "range": float(y.max() - y.min()),
        "theil_sen_slope": _theil_sen(x, y),
        "change_point_strength": _change_point_strength(y),
        "recent_long_term_delta": recent_long_term,
        "baseline_z": baseline_z,
        "baseline_relative_change": relative,
        "baseline_percentile": percentile,
    }


def _put(
    values: dict[str, float | int | None],
    masks: dict[str, int],
    name: str,
    value: float | int | None,
) -> None:
    if value is None or not math.isfinite(float(value)):
        values[name] = None
        masks[name] = 0
    else:
        values[name] = float(value)
        masks[name] = 1


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return None
    x = left[valid]
    y = right[valid]
    if np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _joint_features(
    daily: Sequence[DailyMappedFeatures],
    baseline: Sequence[DailyMappedFeatures],
) -> dict[str, float | None]:
    activity = np.asarray(
        [
            float(item.activity.value("observed_activity_intensity"))
            if item.activity.mask("observed_activity_intensity")
            else np.nan
            for item in daily
        ],
        dtype=float,
    )
    sleep = np.asarray(
        [
            float(item.sleep.value("sleep_efficiency"))
            if item.sleep.mask("sleep_efficiency")
            else np.nan
            for item in daily
        ],
        dtype=float,
    )
    synchronous = _pearson(activity, sleep)
    lag_activity_to_sleep = _pearson(activity[:-1], sleep[1:]) if len(daily) > 1 else None
    lag_sleep_to_activity = _pearson(sleep[:-1], activity[1:]) if len(daily) > 1 else None
    valid_activity = np.isfinite(activity)
    valid_sleep = np.isfinite(sleep)
    quality_consistency = 1.0 - abs(valid_activity.mean() - valid_sleep.mean())

    baseline_activity = np.asarray(
        [
            float(item.activity.value("observed_activity_intensity"))
            for item in baseline
            if item.activity.mask("observed_activity_intensity")
        ],
        dtype=float,
    )
    baseline_sleep = np.asarray(
        [
            float(item.sleep.value("sleep_efficiency"))
            for item in baseline
            if item.sleep.mask("sleep_efficiency")
        ],
        dtype=float,
    )
    common_worsening: float | None = None
    compensation: float | None = None
    if valid_activity.any() and valid_sleep.any() and len(baseline_activity) >= 3 and len(baseline_sleep) >= 3:
        activity_delta = float(np.nanmedian(activity) - np.median(baseline_activity))
        sleep_delta = float(np.nanmedian(sleep) - np.median(baseline_sleep))
        activity_scale = max(float(np.std(baseline_activity)), 1.0e-6)
        sleep_scale = max(float(np.std(baseline_sleep)), 1.0e-6)
        activity_z = activity_delta / activity_scale
        sleep_z = sleep_delta / sleep_scale
        common_worsening = float(max(0.0, -activity_z) * max(0.0, -sleep_z))
        compensation = float(max(0.0, -activity_z) * max(0.0, sleep_z))
    return {
        "activity_sleep_sync": synchronous,
        "activity_to_next_sleep_lag": lag_activity_to_sleep,
        "sleep_to_next_activity_lag": lag_sleep_to_activity,
        "cross_modal_quality_consistency": float(quality_consistency),
        "common_worsening": common_worsening,
        "sleep_compensation": compensation,
    }


def build_multiwindow_features(
    request: MoodSocialInferRequest | Mapping[str, Any],
    *,
    config: MoodSocialConfig | None = None,
) -> MultiWindowFeatures:
    parsed = (
        request
        if isinstance(request, MoodSocialInferRequest)
        else MoodSocialInferRequest.model_validate(request)
    )
    active_config = config or load_mood_social_config()
    all_daily = map_mood_social_trend_days(parsed, config=active_config)
    records = {
        item.date: item
        for item in (*parsed.history_daily_features, parsed.current_daily_features)
    }
    values: dict[str, float | int | None] = {}
    masks: dict[str, int] = {}
    parity: dict[str, str] = {}

    for days in WINDOW_DAYS:
        start = parsed.target_date - timedelta(days=days - 1)
        window = tuple(item for item in all_daily if start <= item.date <= parsed.target_date)
        baseline = tuple(item for item in all_daily if item.date < start)
        if len(window) != days:
            raise ValueError(f"r3 temporal window {days}d is not calendar-complete")
        dates = tuple(item.date for item in window)
        # The legacy aggregate schema encodes valid_days/valid_nights with a
        # hard maximum of seven.  Reuse it only where mathematically valid;
        # longer windows are represented by the daily-statistic contract below
        # instead of clipping their quality counts.
        aggregates: tuple[tuple[str, Any], ...] = ()
        if days <= 7:
            aggregates = (
                (
                    "activity",
                    _aggregate_activity(
                        window,
                        records,
                        dates,
                        daytime_minutes=float(active_config.camera.daytime_minutes),
                    ),
                ),
                ("sleep", _aggregate_sleep(window, records, dates)),
            )
        for domain, aggregate in aggregates:
            for feature, value in aggregate.values_dict().items():
                name = f"{domain}.w{days}.{feature}"
                _put(values, masks, name, value if aggregate.mask(feature) else None)
                if days == 7:
                    parity[name] = f"{domain}.{feature}"
        for domain in DAILY_DOMAINS:
            daily_names = getattr(window[0], domain).feature_names
            for feature in daily_names:
                x, y = _observations(window, domain, feature)
                _, baseline_y = _observations(baseline, domain, feature) if baseline else (
                    np.empty(0),
                    np.empty(0),
                )
                for statistic, value in _summary(x, y, baseline_y).items():
                    _put(
                        values,
                        masks,
                        f"{domain}.w{days}.{feature}.{statistic}",
                        value,
                    )
        for feature, value in _joint_features(window, baseline).items():
            _put(values, masks, f"joint.w{days}.{feature}", value)
        _put(
            values,
            masks,
            f"quality.w{days}.valid_activity_days",
            sum(item.day_mask.activity for item in window),
        )
        _put(
            values,
            masks,
            f"quality.w{days}.valid_sleep_nights",
            sum(item.day_mask.sleep for item in window),
        )
    return MultiWindowFeatures(
        schema_version=TEMPORAL_SCHEMA_VERSION,
        target_date=parsed.target_date,
        values=values,
        feature_mask=masks,
        training_parity=parity,
    )


__all__ = [
    "MultiWindowFeatures",
    "SUMMARY_STATISTICS",
    "TEMPORAL_SCHEMA_VERSION",
    "WINDOW_DAYS",
    "build_multiwindow_features",
]
