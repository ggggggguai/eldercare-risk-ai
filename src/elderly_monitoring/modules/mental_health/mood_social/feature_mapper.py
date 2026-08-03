"""Deterministic V3.3.3 daily-to-seven-day feature mapping."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, timedelta
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence

from elderly_monitoring.modules.mental_health.mood_social.config import (
    CURRENT_CONFIG_VERSION,
    MoodSocialConfig,
    load_mood_social_config,
)
from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    ACTIVITY_DAILY_FEATURE_SPECS,
    ACTIVITY_FEATURE_SPECS,
    CAMERA_DAYTIME_END,
    CAMERA_DAYTIME_MINUTES,
    CAMERA_DAYTIME_START,
    FEATURE_SCHEMA_VERSION,
    HISTORY_LOOKBACK_DAYS,
    PHYSIOLOGY_DAILY_FEATURE_SPECS,
    PHYSIOLOGY_FEATURE_SPECS,
    SLEEP_DAILY_FEATURE_SPECS,
    SLEEP_FEATURE_SPECS,
    SOCIAL_CONTACT_DAILY_FEATURE_SPECS,
    SOCIAL_CONTACT_FEATURE_SPECS,
    SOCIAL_CONTEXT_FEATURE_SPECS,
    STATE_WINDOW_DAYS,
    CameraGaitDay,
    DailyMappedFeatures,
    DayMask,
    DomainFeatureVector,
    FeatureSpec,
    FeatureValue,
    MappedMoodSocialFeatures,
    TrendContext,
)
from elderly_monitoring.modules.mental_health.mood_social.schemas import (
    MoodSocialActivity,
    MoodSocialDailyFeatures,
    MoodSocialInferRequest,
    MoodSocialPhysiology,
    MoodSocialProfile,
    MoodSocialSleep,
    MoodSocialSocial,
)


_MINUTES_PER_DAY = 1440.0
_CIRCULAR_EPSILON = 1e-12


def map_mood_social_features(
    request: MoodSocialInferRequest | Mapping[str, Any],
    *,
    config: MoodSocialConfig | None = None,
) -> MappedMoodSocialFeatures:
    """Map one strict V3 request into frozen daily and seven-day features.

    The state vector uses exactly D-6 through D.  D-28 through D is visible
    only to explicitly long-horizon derivations and the internal trend
    context; sparse history is never replaced or forward-filled.
    """

    parsed = (
        request
        if isinstance(request, MoodSocialInferRequest)
        else MoodSocialInferRequest.model_validate(request)
    )
    active_config = config or load_mood_social_config()
    _validate_config(active_config)

    records = {
        item.date: item
        for item in (
            *parsed.history_daily_features,
            parsed.current_daily_features,
        )
    }
    window_dates = tuple(
        parsed.target_date - timedelta(days=offset)
        for offset in range(STATE_WINDOW_DAYS - 1, -1, -1)
    )

    daily_features = tuple(
        _map_daily_features(
            day,
            records.get(day),
            target_date=parsed.target_date,
            records=records,
            daytime_minutes=float(active_config.camera.daytime_minutes),
        )
        for day in window_dates
    )

    return MappedMoodSocialFeatures(
        schema_version=FEATURE_SCHEMA_VERSION,
        target_date=parsed.target_date,
        window_start_date=window_dates[0],
        window_end_date=window_dates[-1],
        daily_features=daily_features,
        activity=_aggregate_activity(
            daily_features,
            records,
            window_dates,
            daytime_minutes=float(active_config.camera.daytime_minutes),
        ),
        sleep=_aggregate_sleep(daily_features, records, window_dates),
        physiology=_aggregate_physiology(daily_features),
        social_context=_map_profile(parsed.profile),
        social_contact=_aggregate_social(
            records,
            window_dates,
            parsed.target_date,
        ),
        trend_context=_build_trend_context(records),
    )


def map_mood_social_trend_days(
    request: MoodSocialInferRequest | Mapping[str, Any],
    *,
    config: MoodSocialConfig | None = None,
) -> tuple[DailyMappedFeatures, ...]:
    """Map the exact D-28 through D natural-day sequence for PersonalTrend.

    Unlike :func:`map_mood_social_features`, this function does not aggregate a
    seven-day state vector.  Missing calendar days remain explicit mask-zero
    records so a gap can break persistence without being forward-filled.
    """

    parsed = (
        request
        if isinstance(request, MoodSocialInferRequest)
        else MoodSocialInferRequest.model_validate(request)
    )
    active_config = config or load_mood_social_config()
    _validate_config(active_config)
    records = {
        item.date: item
        for item in (
            *parsed.history_daily_features,
            parsed.current_daily_features,
        )
    }
    days = tuple(
        parsed.target_date - timedelta(days=offset)
        for offset in range(HISTORY_LOOKBACK_DAYS, -1, -1)
    )
    return tuple(
        _map_daily_features(
            day,
            records.get(day),
            target_date=parsed.target_date,
            records=records,
            daytime_minutes=float(active_config.camera.daytime_minutes),
        )
        for day in days
    )


def _validate_config(config: MoodSocialConfig) -> None:
    if config.module != "mood_social" or config.version != CURRENT_CONFIG_VERSION:
        raise ValueError(
            "V3.3.3 feature mapping requires the current mood-social config"
        )
    if config.runtime.timezone != "Asia/Shanghai":
        raise ValueError("V3.3.3 feature mapping requires Asia/Shanghai")
    if config.runtime.inference_window_days != STATE_WINDOW_DAYS:
        raise ValueError("V3.3.3 feature mapping requires a seven-day state window")
    if config.baseline.history_lookback_calendar_days != HISTORY_LOOKBACK_DAYS:
        raise ValueError("V3.3.3 feature mapping requires a 28-day history window")
    if (
        config.camera.daytime_start != CAMERA_DAYTIME_START
        or config.camera.daytime_end != CAMERA_DAYTIME_END
        or config.camera.daytime_minutes != CAMERA_DAYTIME_MINUTES
    ):
        raise ValueError("V3.3.3 camera mapping requires the frozen 06:00-18:00 window")


def _map_daily_features(
    day: date,
    record: MoodSocialDailyFeatures | None,
    *,
    target_date: date,
    records: Mapping[date, MoodSocialDailyFeatures],
    daytime_minutes: float,
) -> DailyMappedFeatures:
    activity, activity_day_mask = _map_activity_day(
        record.activity if record is not None else None,
        daytime_minutes=daytime_minutes,
    )
    sleep, sleep_day_mask = _map_sleep_day(record.sleep if record is not None else None)
    physiology, physiology_day_mask = _map_physiology_day(
        record.physiology if record is not None else None
    )
    social, social_day_mask = _map_social_day(
        record.social if record is not None else None,
        no_effective_contact_days=(
            _no_effective_contact_days(records, target_date)
            if day == target_date
            else None
        ),
    )
    day_mask = DayMask(
        date=day,
        activity=activity_day_mask,
        sleep=sleep_day_mask,
        physiology=physiology_day_mask,
        social=social_day_mask,
    )
    return DailyMappedFeatures(
        date=day,
        activity=activity,
        sleep=sleep,
        physiology=physiology,
        social=social,
        day_mask=day_mask,
    )


def _map_activity_day(
    activity: MoodSocialActivity | None,
    *,
    daytime_minutes: float,
) -> tuple[DomainFeatureVector, int]:
    values: dict[str, FeatureValue] = {}
    if activity is None or activity.valid_daytime_detection_minutes <= 0.0:
        return _vector("activity", ACTIVITY_DAILY_FEATURE_SPECS, values), 0

    valid_minutes = activity.valid_daytime_detection_minutes
    weighted = activity.weighted_daytime_activity
    if weighted is None:
        weighted = _hourly_weighted_activity(activity)
    values["observed_activity_intensity"] = _safe_ratio(weighted, valid_minutes)
    values["active_ratio"] = _safe_ratio(
        activity.daytime_active_minutes,
        valid_minutes,
    )
    values["sedentary_ratio"] = _safe_ratio(
        activity.low_activity_minutes,
        valid_minutes,
    )
    values["longest_inactive_bout_norm"] = _safe_ratio(
        activity.longest_sedentary_bout_minutes,
        daytime_minutes,
    )
    return _vector("activity", ACTIVITY_DAILY_FEATURE_SPECS, values), 1


def _map_sleep_day(
    sleep: MoodSocialSleep | None,
) -> tuple[DomainFeatureVector, int]:
    values: dict[str, FeatureValue] = {}
    if sleep is None:
        return _vector("sleep", SLEEP_DAILY_FEATURE_SPECS, values), 0

    values["sleep_duration_norm"] = _normalize_minutes(sleep.sleep_minutes)
    values["time_in_bed_norm"] = _normalize_minutes(sleep.in_bed_minutes)
    efficiency = sleep.sleep_efficiency
    if efficiency is None:
        efficiency = _safe_ratio(sleep.sleep_minutes, sleep.in_bed_minutes)
    values["sleep_efficiency"] = efficiency
    values["sleep_onset_minute_of_day"] = sleep.sleep_onset_minute_of_day
    values["wake_time_minute_of_day"] = sleep.wake_time_minute_of_day
    values["sleep_midpoint_minute_of_day"] = _sleep_midpoint(
        sleep.sleep_onset_minute_of_day,
        sleep.wake_time_minute_of_day,
    )
    values["sleep_fragmentation"] = _sleep_fragmentation(sleep)
    values["night_exit_count"] = sleep.night_exit_count
    values["night_exit_minutes_norm"] = _normalize_minutes(sleep.night_exit_minutes)
    return _vector("sleep", SLEEP_DAILY_FEATURE_SPECS, values), 1


def _map_physiology_day(
    physiology: MoodSocialPhysiology | None,
) -> tuple[DomainFeatureVector, int]:
    values: dict[str, FeatureValue] = {}
    if physiology is None:
        return _vector("physiology", PHYSIOLOGY_DAILY_FEATURE_SPECS, values), 0

    values.update(
        {
            "heart_rate_mean_bpm": physiology.heart_rate_mean_bpm,
            "heart_rate_min_bpm": physiology.heart_rate_min_bpm,
            "heart_rate_sd_bpm": physiology.heart_rate_sd_bpm,
            "hrv_sdnn_ms": physiology.hrv_sdnn_ms,
            "respiration_rate_mean_bpm": physiology.respiration_rate_mean_bpm,
            "respiration_rate_sd_bpm": physiology.respiration_rate_sd_bpm,
            "respiratory_abnormal_ratio": physiology.respiratory_abnormal_ratio,
            "snoring_minutes_norm": _normalize_minutes(physiology.snoring_minutes),
        }
    )
    return _vector("physiology", PHYSIOLOGY_DAILY_FEATURE_SPECS, values), 1


def _map_social_day(
    social: MoodSocialSocial | None,
    *,
    no_effective_contact_days: int | None,
) -> tuple[DomainFeatureVector, int]:
    values: dict[str, FeatureValue] = {}
    if social is None or not social.call_log_observed:
        return _vector("social", SOCIAL_CONTACT_DAILY_FEATURE_SPECS, values), 0

    incoming = social.incoming_call_opportunities
    outgoing = social.outgoing_call_count
    answered = social.answered_call_count
    missed = social.missed_call_count
    duration = social.connected_duration_minutes
    contacts = social.active_contact_count
    assert incoming is not None
    assert outgoing is not None
    assert answered is not None
    assert missed is not None
    assert duration is not None
    assert contacts is not None
    connected = answered + outgoing
    values.update(
        {
            "incoming_call_opportunities": incoming,
            "outgoing_call_count": outgoing,
            "answered_call_count": answered,
            "missed_call_count": missed,
            "answer_rate": _safe_ratio(answered, incoming),
            "connected_call_count": connected,
            "connected_duration_minutes": duration,
            "mean_connected_duration_minutes": _safe_ratio(
                duration,
                connected,
            ),
            "active_contact_count": contacts,
            "no_effective_contact_days": no_effective_contact_days,
        }
    )
    return _vector("social", SOCIAL_CONTACT_DAILY_FEATURE_SPECS, values), 1


def _aggregate_activity(
    daily: Sequence[DailyMappedFeatures],
    records: Mapping[date, MoodSocialDailyFeatures],
    window_dates: Sequence[date],
    *,
    daytime_minutes: float,
) -> DomainFeatureVector:
    values: dict[str, FeatureValue] = {"activity_volume_norm": None}
    window_activities = [
        records[day].activity
        for day in window_dates
        if day in records and records[day].activity is not None
    ]
    observed_activities = [
        item
        for item in window_activities
        if item is not None and item.valid_daytime_detection_minutes > 0.0
    ]

    values["active_ratio"] = _minute_weighted_ratio(
        observed_activities,
        "daytime_active_minutes",
    )
    values["sedentary_ratio"] = _minute_weighted_ratio(
        observed_activities,
        "low_activity_minutes",
    )
    longest_values = [
        item.longest_sedentary_bout_minutes
        for item in observed_activities
        if item.longest_sedentary_bout_minutes is not None
    ]
    values["longest_inactive_bout_norm"] = (
        max(longest_values) / daytime_minutes if longest_values else None
    )

    hourly_cells = _activity_hourly_cells(records, window_dates)
    hourly_profile = _pooled_hourly_profile(hourly_cells)
    values["relative_amplitude"] = _relative_amplitude(hourly_profile)
    values["interdaily_stability"] = _interdaily_stability(hourly_cells)
    values["intradaily_variability"] = _intradaily_variability(hourly_cells)
    daily_intensities = [
        item.activity.value("observed_activity_intensity")
        for item in daily
        if item.activity.mask("observed_activity_intensity") == 1
    ]
    values["activity_variability"] = _coefficient_of_variation(
        [float(value) for value in daily_intensities if value is not None]
    )

    valid_days = sum(item.day_mask.activity for item in daily)
    values["valid_days"] = valid_days
    supported_semantics = (
        "active_ratio",
        "sedentary_ratio",
        "longest_inactive_bout_norm",
        "relative_amplitude",
        "interdaily_stability",
        "intradaily_variability",
        "activity_variability",
    )
    available_semantics = sum(
        values.get(name) is not None for name in supported_semantics
    )
    observed_minutes = math.fsum(
        item.valid_daytime_detection_minutes for item in observed_activities
    )
    time_coverage = _clip01(observed_minutes / (daytime_minutes * STATE_WINDOW_DAYS))
    values["feature_coverage"] = _clip01(
        time_coverage * available_semantics / len(supported_semantics)
    )
    masks = {
        name: int(value is not None)
        for name, value in values.items()
        if name not in {"valid_days", "feature_coverage"}
    }
    masks["valid_days"] = int(valid_days > 0)
    masks["feature_coverage"] = int(valid_days > 0)
    return _vector("activity", ACTIVITY_FEATURE_SPECS, values, masks=masks)


def _aggregate_sleep(
    daily: Sequence[DailyMappedFeatures],
    records: Mapping[date, MoodSocialDailyFeatures],
    window_dates: Sequence[date],
) -> DomainFeatureVector:
    values: dict[str, FeatureValue] = {
        "sleep_duration_norm": _daily_mean(daily, "sleep", "sleep_duration_norm"),
        "time_in_bed_norm": _daily_mean(daily, "sleep", "time_in_bed_norm"),
        "sleep_efficiency": _pooled_sleep_efficiency(
            daily,
            records,
            window_dates,
        ),
        "sleep_fragmentation": _daily_mean(
            daily,
            "sleep",
            "sleep_fragmentation",
        ),
        "night_exit_count_mean": _daily_mean(
            daily,
            "sleep",
            "night_exit_count",
        ),
    }
    onset_minutes: list[float] = []
    wake_minutes: list[float] = []
    midpoint_minutes: list[float] = []
    onset_days: set[date] = set()
    wake_days: set[date] = set()
    midpoint_days: set[date] = set()
    for day in window_dates:
        record = records.get(day)
        sleep = record.sleep if record is not None else None
        if sleep is None:
            continue
        if sleep.sleep_onset_minute_of_day is not None:
            onset_minutes.append(float(sleep.sleep_onset_minute_of_day))
            onset_days.add(day)
        if sleep.wake_time_minute_of_day is not None:
            wake_minutes.append(float(sleep.wake_time_minute_of_day))
            wake_days.add(day)
        midpoint = _sleep_midpoint(
            sleep.sleep_onset_minute_of_day,
            sleep.wake_time_minute_of_day,
        )
        if midpoint is not None:
            midpoint_minutes.append(midpoint)
            midpoint_days.add(day)

    values["sleep_onset_sin"], values["sleep_onset_cos"] = _circular_mean_components(
        onset_minutes
    )
    values["wake_time_sin"], values["wake_time_cos"] = _circular_mean_components(
        wake_minutes
    )
    values["sleep_midpoint_sin"], values["sleep_midpoint_cos"] = (
        _circular_mean_components(midpoint_minutes)
    )
    regularities: list[float] = []
    regularity_days: set[date] = set()
    for minutes, supporting_days in (
        (onset_minutes, onset_days),
        (wake_minutes, wake_days),
        (midpoint_minutes, midpoint_days),
    ):
        regularity = _circular_regularity(minutes)
        if regularity is not None:
            regularities.append(regularity)
            regularity_days.update(supporting_days)
    values["sleep_regularity"] = fmean(regularities) if regularities else None

    valid_nights = sum(item.day_mask.sleep for item in daily)
    values["valid_nights"] = valid_nights
    regularity_support = len(regularity_days)
    coverage_cell_counts = (
        _masked_cell_count(daily, "sleep", "sleep_duration_norm"),
        _masked_cell_count(daily, "sleep", "time_in_bed_norm"),
        _masked_cell_count(daily, "sleep", "sleep_efficiency"),
        _masked_cell_count(daily, "sleep", "sleep_onset_minute_of_day"),
        _masked_cell_count(daily, "sleep", "wake_time_minute_of_day"),
        _masked_cell_count(daily, "sleep", "sleep_midpoint_minute_of_day"),
        _masked_cell_count(daily, "sleep", "sleep_fragmentation"),
        regularity_support,
        _masked_cell_count(daily, "sleep", "night_exit_count"),
    )
    values["feature_coverage"] = _clip01(
        sum(coverage_cell_counts) / (STATE_WINDOW_DAYS * len(coverage_cell_counts))
    )
    masks = {
        name: int(value is not None)
        for name, value in values.items()
        if name not in {"valid_nights", "feature_coverage"}
    }
    masks["valid_nights"] = int(valid_nights > 0)
    masks["feature_coverage"] = int(valid_nights > 0)
    return _vector("sleep", SLEEP_FEATURE_SPECS, values, masks=masks)


def _aggregate_physiology(
    daily: Sequence[DailyMappedFeatures],
) -> DomainFeatureVector:
    semantic_names = tuple(spec.name for spec in PHYSIOLOGY_DAILY_FEATURE_SPECS)
    values: dict[str, FeatureValue] = {
        name: _daily_mean(daily, "physiology", name) for name in semantic_names
    }
    valid_nights = sum(item.day_mask.physiology for item in daily)
    values["valid_nights"] = valid_nights
    values["feature_coverage"] = _clip01(
        sum(_masked_cell_count(daily, "physiology", name) for name in semantic_names)
        / (STATE_WINDOW_DAYS * len(semantic_names))
    )
    masks = {
        name: int(value is not None)
        for name, value in values.items()
        if name not in {"valid_nights", "feature_coverage"}
    }
    masks["valid_nights"] = int(valid_nights > 0)
    masks["feature_coverage"] = int(valid_nights > 0)
    return _vector(
        "physiology",
        PHYSIOLOGY_FEATURE_SPECS,
        values,
        masks=masks,
    )


def _map_profile(profile: MoodSocialProfile) -> DomainFeatureVector:
    values = {
        spec.name: getattr(profile, spec.name) for spec in SOCIAL_CONTEXT_FEATURE_SPECS
    }
    return _vector("social_context", SOCIAL_CONTEXT_FEATURE_SPECS, values)


def _aggregate_social(
    records: Mapping[date, MoodSocialDailyFeatures],
    window_dates: Sequence[date],
    target_date: date,
) -> DomainFeatureVector:
    observed = []
    for day in window_dates:
        record = records.get(day)
        social = record.social if record is not None else None
        if social is not None and social.call_log_observed:
            observed.append(social)

    values: dict[str, FeatureValue] = {}
    if observed:
        incoming = [
            int(item.incoming_call_opportunities)
            for item in observed
            if item.incoming_call_opportunities is not None
        ]
        outgoing = [
            int(item.outgoing_call_count)
            for item in observed
            if item.outgoing_call_count is not None
        ]
        answered = [
            int(item.answered_call_count)
            for item in observed
            if item.answered_call_count is not None
        ]
        missed = [
            int(item.missed_call_count)
            for item in observed
            if item.missed_call_count is not None
        ]
        durations = [
            float(item.connected_duration_minutes)
            for item in observed
            if item.connected_duration_minutes is not None
        ]
        contacts = [
            int(item.active_contact_count)
            for item in observed
            if item.active_contact_count is not None
        ]
        connected = [
            answered_count + outgoing_count
            for answered_count, outgoing_count in zip(
                answered,
                outgoing,
                strict=True,
            )
        ]
        values.update(
            {
                "incoming_call_opportunities": fmean(incoming),
                "outgoing_call_count": fmean(outgoing),
                "answered_call_count": fmean(answered),
                "missed_call_count": fmean(missed),
                "answer_rate": _safe_ratio(sum(answered), sum(incoming)),
                "connected_call_count": fmean(connected),
                "connected_duration_minutes": fmean(durations),
                "mean_connected_duration_minutes": _safe_ratio(
                    math.fsum(durations),
                    sum(connected),
                ),
                "active_contact_count": fmean(contacts),
            }
        )
    values["no_effective_contact_days"] = _no_effective_contact_days(
        records,
        target_date,
    )
    return _vector("social_contact", SOCIAL_CONTACT_FEATURE_SPECS, values)


def _build_trend_context(
    records: Mapping[date, MoodSocialDailyFeatures],
) -> TrendContext:
    gait_days: list[CameraGaitDay] = []
    for day, record in records.items():
        if record.activity is None:
            continue
        for metric in record.activity.camera_gait_metrics:
            gait_days.append(
                CameraGaitDay(
                    date=day,
                    camera_id=metric.camera_id,
                    scene_version=metric.scene_version,
                    gait_speed_image_norm_per_sec=(
                        metric.gait_speed_image_norm_per_sec
                    ),
                    sit_to_stand_duration_seconds=(
                        metric.sit_to_stand_duration_seconds
                    ),
                    turn_duration_seconds=metric.turn_duration_seconds,
                    postural_stability=metric.postural_stability,
                )
            )
    gait_days.sort(key=lambda item: (item.date, item.camera_id, item.scene_version))
    return TrendContext(camera_gait_days=tuple(gait_days))


def _no_effective_contact_days(
    records: Mapping[date, MoodSocialDailyFeatures],
    target_date: date,
) -> int | None:
    current = records.get(target_date)
    current_social = current.social if current is not None else None
    if current_social is None or not current_social.call_log_observed:
        return None

    consecutive_days = 0
    for offset in range(HISTORY_LOOKBACK_DAYS + 1):
        day = target_date - timedelta(days=offset)
        record = records.get(day)
        social = record.social if record is not None else None
        if social is None or not social.call_log_observed:
            break
        answered = social.answered_call_count
        outgoing = social.outgoing_call_count
        assert answered is not None
        assert outgoing is not None
        if answered + outgoing > 0:
            break
        consecutive_days += 1
    return consecutive_days


def _activity_hourly_cells(
    records: Mapping[date, MoodSocialDailyFeatures],
    window_dates: Sequence[date],
) -> list[tuple[date, int, float, float]]:
    cells: list[tuple[date, int, float, float]] = []
    for day in window_dates:
        record = records.get(day)
        activity = record.activity if record is not None else None
        if activity is None or activity.valid_daytime_detection_minutes <= 0.0:
            continue
        for hour, (intensity, coverage) in enumerate(
            zip(
                activity.hourly_activity_intensity,
                activity.hourly_valid_detection_minutes,
                strict=True,
            )
        ):
            if intensity is None or coverage <= 0.0:
                continue
            cells.append((day, hour, float(intensity), float(coverage)))
    return cells


def _pooled_hourly_profile(
    cells: Iterable[tuple[date, int, float, float]],
) -> dict[int, float]:
    weighted_sum: dict[int, float] = defaultdict(float)
    coverage_sum: dict[int, float] = defaultdict(float)
    for _, hour, intensity, coverage in cells:
        weighted_sum[hour] += intensity * coverage
        coverage_sum[hour] += coverage
    return {
        hour: weighted_sum[hour] / coverage_sum[hour]
        for hour in sorted(weighted_sum)
        if coverage_sum[hour] > 0.0
    }


def _relative_amplitude(hourly_profile: Mapping[int, float]) -> float | None:
    ten_hour_means = _circular_window_means(hourly_profile, 10)
    five_hour_means = _circular_window_means(hourly_profile, 5)
    if not ten_hour_means or not five_hour_means:
        return None
    most_active = max(ten_hour_means)
    least_active = min(five_hour_means)
    denominator = most_active + least_active
    if denominator <= _CIRCULAR_EPSILON:
        return 0.0
    return _clip01((most_active - least_active) / denominator)


def _circular_window_means(
    hourly_profile: Mapping[int, float],
    size: int,
) -> list[float]:
    means: list[float] = []
    for start in range(24):
        hours = tuple((start + offset) % 24 for offset in range(size))
        if all(hour in hourly_profile for hour in hours):
            means.append(fmean(hourly_profile[hour] for hour in hours))
    return means


def _interdaily_stability(
    cells: Sequence[tuple[date, int, float, float]],
) -> float | None:
    if len({day for day, _, _, _ in cells}) < 2:
        return None
    if len({hour for _, hour, _, _ in cells}) < 2:
        return None
    total_weight = math.fsum(coverage for _, _, _, coverage in cells)
    if total_weight <= 0.0:
        return None
    overall = (
        math.fsum(intensity * coverage for _, _, intensity, coverage in cells)
        / total_weight
    )
    profile = _pooled_hourly_profile(cells)
    total_variation = math.fsum(
        coverage * (intensity - overall) ** 2 for _, _, intensity, coverage in cells
    )
    between_hour_variation = math.fsum(
        coverage * (profile[hour] - overall) ** 2 for _, hour, _, coverage in cells
    )
    if total_variation <= _CIRCULAR_EPSILON:
        return 1.0
    return _clip01(between_hour_variation / total_variation)


def _intradaily_variability(
    cells: Sequence[tuple[date, int, float, float]],
) -> float | None:
    values_by_day: dict[date, dict[int, float]] = defaultdict(dict)
    values: list[float] = []
    for day, hour, intensity, _ in cells:
        values_by_day[day][hour] = intensity
        values.append(intensity)
    if len(values) < 2:
        return None
    squared_differences = [
        (hours[hour + 1] - hours[hour]) ** 2
        for hours in values_by_day.values()
        for hour in sorted(hours)
        if hour + 1 in hours
    ]
    if not squared_differences:
        return None
    mean_value = fmean(values)
    variance = fmean((value - mean_value) ** 2 for value in values)
    if variance <= _CIRCULAR_EPSILON:
        return 0.0
    return fmean(squared_differences) / variance


def _coefficient_of_variation(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean_value = fmean(values)
    if abs(mean_value) <= _CIRCULAR_EPSILON:
        return 0.0 if all(abs(value) <= _CIRCULAR_EPSILON for value in values) else None
    return pstdev(values) / abs(mean_value)


def _minute_weighted_ratio(
    activities: Iterable[MoodSocialActivity],
    numerator_field: str,
) -> float | None:
    eligible = [
        item
        for item in activities
        if item.valid_daytime_detection_minutes > 0.0
        and getattr(item, numerator_field) is not None
    ]
    denominator = math.fsum(item.valid_daytime_detection_minutes for item in eligible)
    if not eligible or denominator <= 0.0:
        return None
    numerator = math.fsum(float(getattr(item, numerator_field)) for item in eligible)
    return _clip01(numerator / denominator)


def _hourly_weighted_activity(activity: MoodSocialActivity) -> float | None:
    values = [
        float(intensity) * float(coverage)
        for intensity, coverage in zip(
            activity.hourly_activity_intensity,
            activity.hourly_valid_detection_minutes,
            strict=True,
        )
        if intensity is not None and coverage > 0.0
    ]
    return math.fsum(values) if values else None


def _sleep_fragmentation(sleep: MoodSocialSleep) -> float | None:
    components: list[float] = []
    awake_ratio = _safe_ratio(sleep.awake_minutes, sleep.in_bed_minutes)
    if awake_ratio is not None:
        components.append(awake_ratio)
    if (
        sleep.awakening_count is not None
        and sleep.sleep_minutes is not None
        and sleep.sleep_minutes > 0.0
    ):
        components.append(float(sleep.awakening_count) / (sleep.sleep_minutes / 60.0))
    return math.fsum(components) if components else None


def _pooled_sleep_efficiency(
    daily: Sequence[DailyMappedFeatures],
    records: Mapping[date, MoodSocialDailyFeatures],
    window_dates: Sequence[date],
) -> float | None:
    eligible: list[MoodSocialSleep] = []
    for day in window_dates:
        record = records.get(day)
        sleep = record.sleep if record is not None else None
        if (
            sleep is not None
            and sleep.sleep_minutes is not None
            and sleep.in_bed_minutes is not None
            and sleep.in_bed_minutes > 0.0
        ):
            eligible.append(sleep)
    if eligible:
        return _clip01(
            math.fsum(float(item.sleep_minutes) for item in eligible)
            / math.fsum(float(item.in_bed_minutes) for item in eligible)
        )
    return _daily_mean(daily, "sleep", "sleep_efficiency")


def _sleep_midpoint(
    onset_minute: int | None,
    wake_minute: int | None,
) -> float | None:
    if onset_minute is None or wake_minute is None:
        return None
    forward_duration = (wake_minute - onset_minute) % int(_MINUTES_PER_DAY)
    return (onset_minute + forward_duration / 2.0) % _MINUTES_PER_DAY


def _circular_mean_components(
    minutes: Sequence[float],
) -> tuple[float | None, float | None]:
    if not minutes:
        return None, None
    angles = [2.0 * math.pi * value / _MINUTES_PER_DAY for value in minutes]
    mean_sin = fmean(math.sin(angle) for angle in angles)
    mean_cos = fmean(math.cos(angle) for angle in angles)
    length = math.hypot(mean_sin, mean_cos)
    if length <= _CIRCULAR_EPSILON:
        return None, None
    return mean_sin / length, mean_cos / length


def _circular_regularity(minutes: Sequence[float]) -> float | None:
    if len(minutes) < 2:
        return None
    angles = [2.0 * math.pi * value / _MINUTES_PER_DAY for value in minutes]
    mean_sin = fmean(math.sin(angle) for angle in angles)
    mean_cos = fmean(math.cos(angle) for angle in angles)
    return _clip01(math.hypot(mean_sin, mean_cos))


def _daily_mean(
    daily: Sequence[DailyMappedFeatures],
    domain: str,
    feature: str,
) -> float | None:
    values: list[float] = []
    for item in daily:
        vector = getattr(item, domain)
        value = vector.value(feature)
        if vector.mask(feature) == 1 and value is not None:
            values.append(float(value))
    return fmean(values) if values else None


def _masked_cell_count(
    daily: Sequence[DailyMappedFeatures],
    domain: str,
    feature: str,
) -> int:
    return sum(getattr(item, domain).mask(feature) for item in daily)


def _normalize_minutes(value: float | None) -> float | None:
    if value is None:
        return None
    return _clip01(float(value) / _MINUTES_PER_DAY)


def _safe_ratio(
    numerator: float | int | None,
    denominator: float | int | None,
) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _vector(
    domain: str,
    specs: tuple[FeatureSpec, ...],
    values: Mapping[str, FeatureValue],
    *,
    masks: Mapping[str, int] | None = None,
) -> DomainFeatureVector:
    unknown = set(values) - {spec.name for spec in specs}
    if unknown:
        raise ValueError(f"unknown {domain} feature values: {sorted(unknown)!r}")
    ordered_values = tuple(values.get(spec.name) for spec in specs)
    ordered_masks = tuple(
        (
            masks.get(spec.name, int(values.get(spec.name) is not None))
            if masks is not None
            else int(values.get(spec.name) is not None)
        )
        for spec in specs
    )
    return DomainFeatureVector(
        domain=domain,
        specs=specs,
        values=ordered_values,
        feature_mask=ordered_masks,
    )


__all__ = ["map_mood_social_features"]
