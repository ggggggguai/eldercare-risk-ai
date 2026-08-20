from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from numbers import Real
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.mental_health.config import (
    AggregationConfig,
    load_aggregation_config,
)


EP_TIME_TYPE = {
    1: "bed_time",
    2: "sleep_time",
    3: "wake_time",
    4: "get_up_time",
    5: "in_bed_seconds",
    6: "awake_seconds",
    7: "light_sleep_seconds",
    8: "deep_sleep_seconds",
    9: "sleep_seconds",
    10: "leave_bed_seconds",
    11: "time_record_end_time",
    12: "time_record_start_time",
}
QUALITY_BY_RESULT_CODE = {0: "valid", 4001: "low_confidence", 4002: "invalid"}
BASELINE_REJECTING_QUALITIES = {"invalid", "low_confidence", "missing_core_sleep_metrics"}


@dataclass(frozen=True)
class SleepRhythmScore:
    sleep_disturbance_score: float | None
    sleep_rhythm_domain_score: float | None
    baseline_confidence: str
    baseline_quality: float
    persistent_abnormal_days: int
    factors: tuple[str, ...]
    risk_factor_details: dict[str, Any]
    initial_baseline_ready: bool
    stable_baseline_ready: bool


def normalize_ep_sleep_reports(
    reports: Iterable[Mapping[str, Any]],
    *,
    person_id: str,
    device_id: str | None = None,
    device_serial: str | None = None,
    body_detect_messages: Iterable[Mapping[str, Any]] = (),
    config: AggregationConfig | None = None,
) -> list[dict[str, Any]]:
    """Convert Ezviz EP sleep reports into daily sleep rhythm features."""
    aggregation_config = config or load_aggregation_config()
    normalized = [
        _normalize_one_ep_report(
            report,
            person_id=person_id,
            device_id=device_id,
            device_serial=device_serial,
            config=aggregation_config,
        )
        for report in reports
    ]
    event_counts = _body_detect_event_counts(
        body_detect_messages,
        config=aggregation_config,
    )
    for item in normalized:
        events = event_counts.get(item["date"])
        if not events:
            continue
        item["body_detect_leave_bed_count"] = events["leave_bed_count"]
        if item.get("leave_bed_count") is None or events["leave_bed_count"] > item["leave_bed_count"]:
            item["leave_bed_count"] = events["leave_bed_count"]
            item["night_leave_bed_count"] = events["leave_bed_count"]
        item["body_detect_first_message_time"] = events["first_message_time"]
        item["body_detect_last_message_time"] = events["last_message_time"]
    return sorted(normalized, key=lambda row: (row["date"], row.get("device_serial") or ""))


def score_sleep_rhythm_day(
    current: Mapping[str, Any],
    history: Iterable[Mapping[str, Any]],
    *,
    config: AggregationConfig | None = None,
) -> SleepRhythmScore:
    """Score one daily sleep feature record against qualified personal history."""
    aggregation_config = config or load_aggregation_config()
    person_id = str(current.get("person_id") or "").strip()
    current_day = _parse_date(current.get("date"), "current.date")
    history_days = [
        item
        for item in history
        if str(item.get("person_id") or "").strip() == person_id
        and _parse_date(item.get("date"), "history.date") < current_day
        and _baseline_eligible(item)
    ][-14:]
    baseline_quality = _baseline_quality(history_days)
    baseline_confidence = _baseline_confidence(history_days)
    initial_ready = len(history_days) >= 3
    stable_ready = len(history_days) >= 7
    if not initial_ready:
        return SleepRhythmScore(
            sleep_disturbance_score=None,
            sleep_rhythm_domain_score=None,
            baseline_confidence=baseline_confidence,
            baseline_quality=baseline_quality,
            persistent_abnormal_days=0,
            factors=("insufficient_sleep_baseline",),
            risk_factor_details={},
            initial_baseline_ready=False,
            stable_baseline_ready=False,
        )

    metric_specs = {
        "sleep_efficiency": "decrease",
        "sleep_latency_minutes": "increase",
        "night_awakenings": "increase",
        "night_leave_bed_count": "increase",
        "awake_ratio": "increase",
        "sleep_midpoint_minute_of_day": "two_sided",
    }
    details: dict[str, Any] = {}
    for metric, direction in metric_specs.items():
        current_value = _optional_number(current.get(metric))
        values = [_optional_number(row.get(metric)) for row in history_days]
        values = [value for value in values if value is not None]
        if current_value is None or len(values) < 3:
            continue
        detail = _metric_deviation(current_value, values, direction)
        if detail["score"] > 0:
            details[metric] = detail

    if not details:
        score = 0.0
    else:
        score = max(detail["score"] for detail in details.values())
    factors = _sleep_factors(details)
    if not factors:
        factors = ["sleep_rhythm_within_personal_range"]

    persistence = 1 if score >= 0.6 else 0
    if persistence:
        previous = sorted(history_days, key=lambda item: str(item.get("date") or ""), reverse=True)
        for row in previous:
            previous_score = score_sleep_rhythm_day(
                row,
                [item for item in history_days if _parse_date(item.get("date"), "history.date") < _parse_date(row.get("date"), "history.date")],
                config=aggregation_config,
            )
            if previous_score.sleep_disturbance_score is None or previous_score.sleep_disturbance_score < 0.6:
                break
            persistence += 1

    return SleepRhythmScore(
        sleep_disturbance_score=round(score, 4),
        sleep_rhythm_domain_score=round(score * 100.0, 1),
        baseline_confidence=baseline_confidence,
        baseline_quality=baseline_quality,
        persistent_abnormal_days=persistence,
        factors=tuple(factors),
        risk_factor_details=details,
        initial_baseline_ready=initial_ready,
        stable_baseline_ready=stable_ready,
    )


def build_sleep_rhythm_result(
    *,
    person_id: str,
    reports: Iterable[Mapping[str, Any]] = (),
    daily_features: Iterable[Mapping[str, Any]] = (),
    body_detect_messages: Iterable[Mapping[str, Any]] = (),
    history_daily_features: Iterable[Mapping[str, Any]] = (),
    device_id: str | None = None,
    device_serial: str | None = None,
    requested_date: date | None = None,
    config: AggregationConfig | None = None,
) -> dict[str, Any]:
    aggregation_config = config or load_aggregation_config()
    generated = normalize_ep_sleep_reports(
        reports,
        person_id=person_id,
        device_id=device_id,
        device_serial=device_serial,
        body_detect_messages=body_detect_messages,
        config=aggregation_config,
    )
    provided = [
        _normalize_standard_daily_feature(
            record,
            person_id=person_id,
            device_id=device_id,
            device_serial=device_serial,
            config=aggregation_config,
        )
        for record in daily_features
    ]
    daily = sorted([*generated, *provided], key=lambda item: item["date"])
    if requested_date is not None:
        daily = [item for item in daily if item["date"] == requested_date.isoformat()]

    history = [
        _normalize_standard_daily_feature(
            record,
            person_id=str(record.get("person_id") or person_id),
            device_id=device_id,
            device_serial=device_serial,
            config=aggregation_config,
        )
        for record in history_daily_features
    ]
    scored: list[dict[str, Any]] = []
    for item in daily:
        score = score_sleep_rhythm_day(item, history, config=aggregation_config)
        enriched = dict(item)
        enriched.update(
            {
                "sleep_disturbance_score": score.sleep_disturbance_score,
                "sleep_rhythm_domain_score": score.sleep_rhythm_domain_score,
                "baseline_confidence": score.baseline_confidence,
                "baseline_quality": score.baseline_quality,
                "persistent_abnormal_days": score.persistent_abnormal_days,
                "sleep_rhythm_factors": list(score.factors),
                "sleep_rhythm_details": score.risk_factor_details,
                "initial_baseline_ready": score.initial_baseline_ready,
                "stable_baseline_ready": score.stable_baseline_ready,
            }
        )
        scored.append(enriched)
        history.append(enriched)

    return {
        "schema_version": "sleep_rhythm_service_v1",
        "model_version": "sleep-rhythm-rulecard-v1",
        "person_id": person_id,
        "requested_date": requested_date.isoformat() if requested_date else None,
        "daily_features": scored,
        "quality_flags": _quality_flags(scored),
        "medical_disclaimer": "behavioral sleep rhythm trend only; not a medical diagnosis",
    }


def _normalize_one_ep_report(
    report: Mapping[str, Any],
    *,
    person_id: str,
    device_id: str | None,
    device_serial: str | None,
    config: AggregationConfig,
) -> dict[str, Any]:
    analysis = report.get("sleepAnalysis")
    if not isinstance(analysis, Mapping):
        analysis = report.get("sleep_analysis") if isinstance(report.get("sleep_analysis"), Mapping) else {}
    time_values = _time_output_values(analysis.get("timeOutput") or analysis.get("time_output"), config)
    local_date = _report_date(report, time_values, config)
    result_code = _optional_int(analysis.get("resultCode") if analysis.get("resultCode") is not None else report.get("resultCode"))
    data_quality = QUALITY_BY_RESULT_CODE.get(result_code, "valid" if result_code is None else "unknown_result_code")
    flags = [] if data_quality == "valid" else [data_quality]

    sleep_seconds = _optional_number(time_values.get("sleep_seconds"))
    in_bed_seconds = _optional_number(time_values.get("in_bed_seconds"))
    awake_seconds = _optional_number(time_values.get("awake_seconds"))
    light_seconds = _optional_number(time_values.get("light_sleep_seconds"))
    deep_seconds = _optional_number(time_values.get("deep_sleep_seconds"))
    leave_bed_seconds = _optional_number(time_values.get("leave_bed_seconds"))
    sleep_time = time_values.get("sleep_time")
    bed_time = time_values.get("bed_time")
    wake_time = time_values.get("wake_time")

    efficiency = _safe_ratio(sleep_seconds, in_bed_seconds)
    latency_minutes = _minutes_between(bed_time, sleep_time)
    awake_ratio = _safe_ratio(awake_seconds, in_bed_seconds)
    deep_ratio = _safe_ratio(deep_seconds, sleep_seconds)
    light_ratio = _safe_ratio(light_seconds, sleep_seconds)
    midpoint = _midpoint(sleep_time, wake_time, config)
    night_awakenings = _night_awakenings(analysis)
    leave_bed_count = _optional_int(analysis.get("leaveBedCount") if analysis.get("leaveBedCount") is not None else analysis.get("leave_bed_count"))
    sleep_score = _sleep_score(analysis)
    move_output = analysis.get("moveOutPut") or analysis.get("moveOutput") or analysis.get("move_output")
    move_times = _nested_number(move_output, "moveTimes", "move_times")
    move_per_hour = _nested_number(move_output, "moveTimePerHour", "move_time_per_hour")
    freq_stats = _freq_record_stats(
        analysis.get("freqRecordOutput")
        or analysis.get("freqRecordOutPut")
        or analysis.get("freq_record_output")
    )

    if efficiency is None or latency_minutes is None:
        flags.append("missing_core_sleep_metrics")
    baseline_eligible = data_quality == "valid" and "missing_core_sleep_metrics" not in flags
    quality_score = 1.0 if data_quality == "valid" else 0.5 if data_quality == "low_confidence" else 0.0
    return {
        "person_id": person_id,
        "date": local_date.isoformat(),
        "device_id": device_id,
        "device_serial": device_serial or _optional_string(report.get("deviceSerial") or report.get("device_serial")),
        "report_start_time": _datetime_iso(report.get("reportStartTime") or report.get("report_start_time"), config),
        "report_end_time": _datetime_iso(report.get("reportEndTime") or report.get("report_end_time"), config),
        "bed_time": _iso_or_none(bed_time),
        "sleep_time": _iso_or_none(sleep_time),
        "wake_time": _iso_or_none(wake_time),
        "get_up_time": _iso_or_none(time_values.get("get_up_time")),
        "in_bed_seconds": _round_or_none(in_bed_seconds),
        "sleep_seconds": _round_or_none(sleep_seconds),
        "awake_seconds": _round_or_none(awake_seconds),
        "light_sleep_seconds": _round_or_none(light_seconds),
        "deep_sleep_seconds": _round_or_none(deep_seconds),
        "leave_bed_seconds": _round_or_none(leave_bed_seconds),
        "leave_bed_count": leave_bed_count,
        "night_leave_bed_count": leave_bed_count,
        "night_leave_bed_minutes": _round_or_none(leave_bed_seconds / 60.0 if leave_bed_seconds is not None else None),
        "mean_heart_rate": _nested_number(analysis, "meanHeartFreqOutPut", "meanHeartFreqOutput", "mean_heart_rate"),
        "mean_breath_rate": _nested_number(analysis, "meanBreathFreqOutPut", "meanBreathFreqOutput", "mean_breath_rate"),
        "heart_rate_std": _round_or_none(freq_stats.get("heart_rate_std")),
        "breath_rate_std": _round_or_none(freq_stats.get("breath_rate_std")),
        "heart_rate_min": _round_or_none(freq_stats.get("heart_rate_min")),
        "heart_rate_max": _round_or_none(freq_stats.get("heart_rate_max")),
        "breath_rate_min": _round_or_none(freq_stats.get("breath_rate_min")),
        "breath_rate_max": _round_or_none(freq_stats.get("breath_rate_max")),
        "heart_rate_range": _round_or_none(freq_stats.get("heart_rate_range")),
        "breath_rate_range": _round_or_none(freq_stats.get("breath_rate_range")),
        "heart_rate_measurement_count": _round_or_none(freq_stats.get("heart_rate_measurement_count")),
        "breath_rate_measurement_count": _round_or_none(freq_stats.get("breath_rate_measurement_count")),
        "move_times": _optional_int(move_times),
        "move_time_per_hour": _round_or_none(move_per_hour),
        "motion_per_hour": _round_or_none(move_per_hour),
        "sleep_score": _round_or_none(sleep_score),
        "result_code": result_code,
        "data_quality": data_quality,
        "quality_score": quality_score,
        "quality_flags": _dedupe(flags),
        "baseline_eligible": baseline_eligible,
        "sleep_efficiency": _round_or_none(efficiency),
        "efficiency": _round_or_none(efficiency),
        "sleep_latency_minutes": _round_or_none(latency_minutes),
        "sleep_onset_latency": _round_or_none(latency_minutes),
        "night_awakenings": night_awakenings,
        "awake_ratio": _round_or_none(awake_ratio),
        "deep_sleep_ratio": _round_or_none(deep_ratio),
        "light_sleep_ratio": _round_or_none(light_ratio),
        "sleep_midpoint": _iso_or_none(midpoint),
        "sleep_midpoint_minute_of_day": _minute_of_day(midpoint),
    }


def _normalize_standard_daily_feature(
    record: Mapping[str, Any],
    *,
    person_id: str,
    device_id: str | None,
    device_serial: str | None,
    config: AggregationConfig,
) -> dict[str, Any]:
    item = dict(record)
    item["person_id"] = str(item.get("person_id") or person_id)
    item["date"] = _parse_date(item.get("date"), "daily.date").isoformat()
    item.setdefault("device_id", device_id)
    item.setdefault("device_serial", device_serial)
    if item.get("sleep_efficiency") is None and item.get("efficiency") is not None:
        item["sleep_efficiency"] = item["efficiency"]
    if item.get("efficiency") is None and item.get("sleep_efficiency") is not None:
        item["efficiency"] = item["sleep_efficiency"]
    if item.get("sleep_latency_minutes") is None and item.get("sleep_onset_latency") is not None:
        item["sleep_latency_minutes"] = item["sleep_onset_latency"]
    if item.get("sleep_onset_latency") is None and item.get("sleep_latency_minutes") is not None:
        item["sleep_onset_latency"] = item["sleep_latency_minutes"]
    if item.get("night_leave_bed_count") is None and item.get("leave_bed_count") is not None:
        item["night_leave_bed_count"] = item["leave_bed_count"]
    if item.get("sleep_midpoint_minute_of_day") is None and item.get("sleep_midpoint") is not None:
        midpoint = _parse_datetime(item.get("sleep_midpoint"), config)
        item["sleep_midpoint_minute_of_day"] = _minute_of_day(midpoint)
    item.setdefault("data_quality", "valid")
    item.setdefault("quality_flags", [])
    item.setdefault("baseline_eligible", _baseline_eligible(item))
    return item


def _time_output_values(raw: Any, config: AggregationConfig) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if not isinstance(raw, list):
        return values
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        raw_type = item.get("type")
        if isinstance(raw_type, str) and raw_type.isdigit():
            raw_type = int(raw_type)
        if raw_type not in EP_TIME_TYPE:
            continue
        key = EP_TIME_TYPE[int(raw_type)]
        value = _first_present(item, "value", "time", "timestamp", "duration", "output", "data", "text")
        if key.endswith("_time"):
            parsed = _parse_datetime(value, config)
            if parsed is not None:
                values[key] = parsed
        else:
            parsed_duration = _duration_seconds(value, item)
            if parsed_duration is not None:
                values[key] = parsed_duration
    return values


def _body_detect_event_counts(
    messages: Iterable[Mapping[str, Any]],
    *,
    config: AggregationConfig,
) -> dict[str, dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for message in messages:
        message_type = _optional_int(message.get("messageType") if message.get("messageType") is not None else message.get("message_type"))
        if message_type != 2:
            continue
        timestamp = _parse_datetime(message.get("messageTime") or message.get("message_time") or message.get("timestamp"), config)
        if timestamp is None:
            continue
        if not _is_night_time(timestamp.time(), config):
            continue
        day = timestamp.date().isoformat()
        bucket = counts.setdefault(day, {"leave_bed_count": 0, "first_message_time": timestamp.isoformat(), "last_message_time": timestamp.isoformat()})
        bucket["leave_bed_count"] += 1
        if timestamp.isoformat() < bucket["first_message_time"]:
            bucket["first_message_time"] = timestamp.isoformat()
        if timestamp.isoformat() > bucket["last_message_time"]:
            bucket["last_message_time"] = timestamp.isoformat()
    return counts


def _metric_deviation(current: float, values: list[float], direction: str) -> dict[str, Any]:
    mean = fmean(values)
    std = pstdev(values)
    scale = max(std, abs(mean) * 0.05, 0.05)
    if direction == "decrease":
        delta = max(0.0, mean - current)
    elif direction == "increase":
        delta = max(0.0, current - mean)
    else:
        delta = abs(current - mean)
        if direction == "two_sided":
            delta = min(delta, 1440.0 - delta) if delta > 720.0 else delta
    standardized = min(delta / (scale * 2.0), 1.0)
    relative = min((delta / max(abs(mean), 0.05)) / 0.50, 1.0)
    return {
        "direction": direction,
        "current_value": round(current, 4),
        "baseline_count": len(values),
        "baseline_mean": round(mean, 4),
        "baseline_std": round(std, 4),
        "risk_delta": round(delta, 4),
        "score": round(max(standardized, relative), 4),
    }


def _sleep_factors(details: Mapping[str, Mapping[str, Any]]) -> list[str]:
    labels = {
        "sleep_efficiency": "sleep_efficiency_decline",
        "sleep_latency_minutes": "sleep_latency_increase",
        "night_awakenings": "night_awakenings_increase",
        "night_leave_bed_count": "night_leave_bed_increase",
        "awake_ratio": "awake_ratio_increase",
        "sleep_midpoint_minute_of_day": "sleep_timing_shift",
    }
    return [
        labels[name]
        for name, detail in sorted(details.items(), key=lambda item: item[1]["score"], reverse=True)
        if detail["score"] >= 0.35
    ]


def _baseline_eligible(record: Mapping[str, Any]) -> bool:
    if record.get("baseline_eligible") is False:
        return False
    quality = str(record.get("data_quality") or "valid")
    flags = set(str(flag) for flag in record.get("quality_flags") or [])
    return quality not in BASELINE_REJECTING_QUALITIES and not flags.intersection(BASELINE_REJECTING_QUALITIES)


def _baseline_quality(history: list[Mapping[str, Any]]) -> float:
    if not history:
        return 0.0
    qualities = [_optional_number(item.get("quality_score")) for item in history]
    quality = fmean([value for value in qualities if value is not None] or [1.0])
    return round(min(len(history) / 7.0, 1.0) * quality, 4)


def _baseline_confidence(history: list[Mapping[str, Any]]) -> str:
    if len(history) >= 14:
        return "high"
    if len(history) >= 7:
        return "medium"
    if len(history) >= 3:
        return "low"
    return "insufficient"


def _quality_flags(records: Iterable[Mapping[str, Any]]) -> list[str]:
    flags: list[str] = []
    for record in records:
        flags.extend(str(flag) for flag in record.get("quality_flags") or [])
        if record.get("sleep_disturbance_score") is None:
            flags.append("sleep_rhythm_score_unavailable")
    return _dedupe(flags)


def _report_date(report: Mapping[str, Any], time_values: Mapping[str, Any], config: AggregationConfig) -> date:
    raw = report.get("date")
    if isinstance(raw, str) and len(raw) == 10:
        return date.fromisoformat(raw)
    for key in ("wake_time", "get_up_time", "sleep_time", "reportEndTime", "report_end_time"):
        value = time_values.get(key) if key in time_values else report.get(key)
        parsed = value if isinstance(value, datetime) else _parse_datetime(value, config)
        if parsed is not None:
            return parsed.date()
    raise ValueError("sleep report requires date or a usable report/sleep timestamp")


def _parse_date(value: Any, path: str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError(f"{path} must use YYYY-MM-DD format")
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError(f"{path} must use YYYY-MM-DD format")
    return parsed


def _parse_datetime(value: Any, config: AggregationConfig) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            return None
        parsed = datetime.fromtimestamp(number / 1000.0 if number > 1e12 else number, tz=config.timezone_info)
    elif isinstance(value, str) and value.strip():
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=config.timezone_info)
    return parsed.astimezone(config.timezone_info)


def _duration_seconds(value: Any, item: Mapping[str, Any]) -> float | None:
    number = _optional_number(value)
    if number is not None:
        keys = " ".join(str(key).lower() for key in item)
        if "minute" in keys or keys.endswith("min"):
            return number * 60.0
        return number
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if ":" in text:
            parts = text.split(":")
            try:
                numbers = [float(part) for part in parts]
            except ValueError:
                return None
            if len(numbers) == 3:
                return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
            if len(numbers) == 2:
                return numbers[0] * 3600 + numbers[1] * 60
        if text.endswith("分钟"):
            try:
                return float(text[:-2]) * 60.0
            except ValueError:
                return None
        if text.endswith("秒"):
            try:
                return float(text[:-1])
            except ValueError:
                return None
    return None


def _first_present(item: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def _nested_number(raw: Any, *keys: str) -> float | None:
    if isinstance(raw, Real) and not isinstance(raw, bool):
        return _optional_number(raw)
    if not isinstance(raw, Mapping):
        return None
    for key in keys:
        value = raw.get(key)
        if isinstance(value, Mapping):
            nested = _first_present(value, "value", "avg", "mean")
            number = _optional_number(nested)
        else:
            number = _optional_number(value)
        if number is not None:
            return number
    return None


def _freq_record_stats(raw: Any) -> dict[str, float]:
    if not isinstance(raw, list):
        return {}
    heart_values: list[float] = []
    breath_values: list[float] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        heart = _record_number(
            item,
            "heartFreq",
            "heartFreqOutput",
            "heartRate",
            "heart_rate",
            "heart",
        )
        breath = _record_number(
            item,
            "breathFreq",
            "breathFreqOutput",
            "breathRate",
            "breath_rate",
            "breath",
        )
        if heart is not None:
            heart_values.append(heart)
        if breath is not None:
            breath_values.append(breath)
    stats: dict[str, float] = {}
    stats.update(_series_stats("heart_rate", heart_values))
    stats.update(_series_stats("breath_rate", breath_values))
    return stats


def _record_number(item: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in item:
            continue
        value = item[key]
        if isinstance(value, Mapping):
            value = _first_present(value, "value", "avg", "mean", "data")
        number = _optional_number(value)
        if number is not None:
            return number
    return None


def _series_stats(prefix: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    minimum = min(values)
    maximum = max(values)
    return {
        f"{prefix}_std": pstdev(values) if len(values) > 1 else 0.0,
        f"{prefix}_min": minimum,
        f"{prefix}_max": maximum,
        f"{prefix}_range": maximum - minimum,
        f"{prefix}_measurement_count": float(len(values)),
    }


def _sleep_score(analysis: Mapping[str, Any]) -> float | None:
    percentage = analysis.get("percentageOutPut") or analysis.get("percentageOutput") or analysis.get("percentage_output")
    score = _nested_number(percentage, "sleepPoint", "sleep_point", "score")
    return score if score is not None else _nested_number(analysis, "score")


def _night_awakenings(analysis: Mapping[str, Any]) -> int | None:
    direct = _optional_int(_first_present(analysis, "nightAwakenings", "night_awakenings", "awakeCount", "awake_count"))
    if direct is not None:
        return direct
    problem = analysis.get("sleepProblem") or analysis.get("sleep_problem")
    if isinstance(problem, Mapping):
        return _optional_int(_first_present(problem, "wakeCount", "wake_count", "awakeCount", "awake_count"))
    return None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return max(0.0, min(numerator / denominator, 1.0))


def _minutes_between(start: Any, end: Any) -> float | None:
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        return None
    delta = (end - start).total_seconds() / 60.0
    if delta < 0:
        delta += 24.0 * 60.0
    return delta


def _midpoint(start: Any, end: Any, config: AggregationConfig) -> datetime | None:
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        return None
    if end < start:
        end += timedelta(days=1)
    return (start + (end - start) / 2).astimezone(config.timezone_info)


def _minute_of_day(value: datetime | None) -> float | None:
    if value is None:
        return None
    return float(value.hour * 60 + value.minute + value.second / 60.0)


def _datetime_iso(value: Any, config: AggregationConfig) -> str | None:
    parsed = _parse_datetime(value, config)
    return _iso_or_none(parsed)


def _iso_or_none(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _optional_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _round_or_none(value: float | None) -> float | None:
    return round(float(value), 4) if value is not None else None


def _is_night_time(value: time, config: AggregationConfig) -> bool:
    start = config.night_start_time
    end = config.night_end_time
    if start < end:
        return start <= value < end
    return value >= start or value < end


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
