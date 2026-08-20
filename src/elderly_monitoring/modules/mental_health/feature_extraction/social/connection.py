from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from numbers import Real
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.mental_health.config import (
    AggregationConfig,
    load_aggregation_config,
)


ANSWER_ACTIONS = {"answer", "answered", "accept", "accepted", "connect", "connected", "complete", "completed"}
REJECT_ACTIONS = {"reject", "rejected"}
BUSY_ACTIONS = {"busy"}
TIMEOUT_ACTIONS = {"belltimeout", "bell_timeout", "timeout", "ring_timeout"}
CANCEL_ACTIONS = {"cancel", "canceled", "cancelled"}
REQUEST_ACTIONS = {"request", "invite", "call", "start"}
BASELINE_REJECTING_QUALITIES = {"invalid", "low_quality", "data_missing"}


@dataclass(frozen=True)
class SocialConnectionScore:
    social_withdrawal_score: float | None
    social_connection_domain_score: float | None
    baseline_confidence: str
    baseline_quality: float
    persistent_abnormal_days: int
    factors: tuple[str, ...]
    risk_factor_details: dict[str, Any]
    initial_baseline_ready: bool
    stable_baseline_ready: bool


@dataclass(frozen=True)
class _CallEvent:
    person_id: str
    occurred_at: datetime
    call_id: str
    device_id: str | None
    contact_id: str | None
    action: str | None
    direction: str | None
    duration_seconds: float | None
    quality_flags: tuple[str, ...]


@dataclass
class _CallAttempt:
    person_id: str
    call_id: str
    started_at: datetime
    ended_at: datetime
    device_id: str | None = None
    contact_id: str | None = None
    direction: str | None = None
    answered: bool = False
    rejected: bool = False
    busy: bool = False
    bell_timeout: bool = False
    canceled: bool = False
    duration_seconds: float = 0.0
    quality_flags: set[str] | None = None


def normalize_call_events(
    records: Iterable[Mapping[str, Any]],
    *,
    person_id: str | None = None,
    device_id: str | None = None,
    config: AggregationConfig | None = None,
) -> list[dict[str, Any]]:
    """Normalize ERTC/webhook/call-log records without inspecting call content."""
    aggregation_config = config or load_aggregation_config()
    events = [
        _adapt_call_event(
            record,
            person_id=person_id,
            device_id=device_id,
            config=aggregation_config,
            record_number=index,
        )
        for index, record in enumerate(records, start=1)
    ]
    return [
        _event_to_dict(event)
        for event in sorted(events, key=lambda item: (item.occurred_at, item.person_id, item.call_id))
    ]


def aggregate_social_connection_daily(
    records: Iterable[Mapping[str, Any]],
    *,
    person_id: str | None = None,
    requested_date: date | None = None,
    device_id: str | None = None,
    config: AggregationConfig | None = None,
) -> list[dict[str, Any]]:
    """Aggregate structured call logs into daily and rolling 7-day social features."""
    aggregation_config = config or load_aggregation_config()
    normalized = [
        _adapt_call_event(
            record,
            person_id=person_id,
            device_id=device_id,
            config=aggregation_config,
            record_number=index,
        )
        for index, record in enumerate(records, start=1)
    ]
    if not normalized:
        raise ValueError("No social connection call records were provided")
    attempts = _build_call_attempts(normalized)
    if not attempts:
        raise ValueError("No usable social connection call attempts were identified")
    return aggregate_social_connection_attempts(
        attempts,
        requested_date=requested_date,
        config=aggregation_config,
    )


def aggregate_social_connection_attempts(
    attempts: Iterable[_CallAttempt],
    *,
    requested_date: date | None = None,
    config: AggregationConfig | None = None,
) -> list[dict[str, Any]]:
    aggregation_config = config or load_aggregation_config()
    attempts_by_person: dict[str, list[_CallAttempt]] = defaultdict(list)
    for attempt in attempts:
        attempts_by_person[attempt.person_id].append(attempt)
    if not attempts_by_person:
        raise ValueError("No social connection call attempts were provided")

    outputs: list[dict[str, Any]] = []
    for person, person_attempts in sorted(attempts_by_person.items()):
        days = sorted({attempt.started_at.date() for attempt in person_attempts})
        if requested_date is not None:
            days = [requested_date]
        for day in days:
            outputs.append(
                _daily_features_for_day(
                    person,
                    day,
                    person_attempts,
                    config=aggregation_config,
                )
            )
    return sorted(outputs, key=lambda item: (str(item["date"]), str(item["person_id"])))


def score_social_connection_day(
    current: Mapping[str, Any],
    history: Iterable[Mapping[str, Any]],
    *,
    config: AggregationConfig | None = None,
) -> SocialConnectionScore:
    """Score one social-connection feature row against qualified personal history."""
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
        return SocialConnectionScore(
            social_withdrawal_score=None,
            social_connection_domain_score=None,
            baseline_confidence=baseline_confidence,
            baseline_quality=baseline_quality,
            persistent_abnormal_days=0,
            factors=("insufficient_social_baseline",),
            risk_factor_details={},
            initial_baseline_ready=False,
            stable_baseline_ready=False,
        )

    metric_specs = {
        "call_count_7d": "decrease",
        "answered_call_count_7d": "decrease",
        "call_answer_rate_7d": "decrease",
        "call_duration_minutes_7d": "decrease",
        "active_call_count_7d": "decrease",
        "missed_call_count_7d": "increase",
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

    score = max((detail["score"] for detail in details.values()), default=0.0)
    factors = _social_factors(details)
    if not factors:
        factors = ["social_connection_within_personal_range"]

    persistence = 1 if score >= 0.6 else 0
    if persistence:
        previous = sorted(history_days, key=lambda item: str(item.get("date") or ""), reverse=True)
        for row in previous:
            row_day = _parse_date(row.get("date"), "history.date")
            previous_history = [
                item
                for item in history_days
                if _parse_date(item.get("date"), "history.date") < row_day
            ]
            previous_score = score_social_connection_day(
                row,
                previous_history,
                config=aggregation_config,
            )
            if previous_score.social_withdrawal_score is None or previous_score.social_withdrawal_score < 0.6:
                break
            persistence += 1

    return SocialConnectionScore(
        social_withdrawal_score=round(score, 4),
        social_connection_domain_score=round(score * 100.0, 1),
        baseline_confidence=baseline_confidence,
        baseline_quality=baseline_quality,
        persistent_abnormal_days=persistence,
        factors=tuple(factors),
        risk_factor_details=details,
        initial_baseline_ready=initial_ready,
        stable_baseline_ready=stable_ready,
    )


def build_social_connection_result(
    *,
    person_id: str,
    call_events: Iterable[Mapping[str, Any]] = (),
    daily_features: Iterable[Mapping[str, Any]] = (),
    history_daily_features: Iterable[Mapping[str, Any]] = (),
    requested_date: date | None = None,
    device_id: str | None = None,
    config: AggregationConfig | None = None,
) -> dict[str, Any]:
    aggregation_config = config or load_aggregation_config()
    generated = []
    if call_events:
        generated = aggregate_social_connection_daily(
            call_events,
            person_id=person_id,
            requested_date=requested_date,
            device_id=device_id,
            config=aggregation_config,
        )
    provided = [
        _normalize_daily_feature(
            record,
            person_id=person_id,
            device_id=device_id,
        )
        for record in daily_features
    ]
    daily = sorted([*generated, *provided], key=lambda item: item["date"])
    if requested_date is not None:
        daily = [item for item in daily if item["date"] == requested_date.isoformat()]

    history = [
        _normalize_daily_feature(
            record,
            person_id=str(record.get("person_id") or person_id),
            device_id=device_id,
        )
        for record in history_daily_features
    ]
    scored: list[dict[str, Any]] = []
    for item in daily:
        score = score_social_connection_day(item, history, config=aggregation_config)
        enriched = dict(item)
        enriched.update(
            {
                "social_withdrawal_score": score.social_withdrawal_score,
                "social_connection_domain_score": score.social_connection_domain_score,
                "baseline_confidence": score.baseline_confidence,
                "baseline_quality": score.baseline_quality,
                "persistent_abnormal_days": score.persistent_abnormal_days,
                "social_connection_factors": list(score.factors),
                "social_connection_details": score.risk_factor_details,
                "initial_baseline_ready": score.initial_baseline_ready,
                "stable_baseline_ready": score.stable_baseline_ready,
            }
        )
        scored.append(enriched)
        history.append(enriched)

    return {
        "schema_version": "social_connection_service_v1",
        "model_version": "social-connection-rulecard-v1",
        "person_id": person_id,
        "requested_date": requested_date.isoformat() if requested_date else None,
        "daily_features": scored,
        "quality_flags": _quality_flags(scored),
        "privacy_boundary": "structured call metadata only; call content is not analyzed",
        "medical_disclaimer": "social connection trend only; not a medical diagnosis",
    }


def _adapt_call_event(
    record: Mapping[str, Any],
    *,
    person_id: str | None,
    device_id: str | None,
    config: AggregationConfig,
    record_number: int,
) -> _CallEvent:
    if not isinstance(record, Mapping):
        raise ValueError(f"social call record {record_number}: expected an object")
    body = record.get("body") if isinstance(record.get("body"), Mapping) else {}
    header = record.get("header") if isinstance(record.get("header"), Mapping) else {}
    result = record.get("result") if isinstance(record.get("result"), Mapping) else {}

    resolved_person = _first_string(record, body, keys=("person_id", "elder_id")) or person_id
    if not resolved_person:
        raise ValueError(f"social call record {record_number}: field 'person_id' must be provided")
    resolved_device = (
        _first_string(record, body, header, keys=("device_id", "deviceId", "device_serial", "deviceSerial"))
        or device_id
    )
    occurred_at = _event_time(record, body, header, config, record_number)
    call_id = (
        _first_string(record, body, result, keys=("call_id", "callId", "requestId", "request_id", "roomId", "room_id", "strRoomId"))
        or f"{resolved_person}:{occurred_at.isoformat()}:{record_number}"
    )
    action = _normalize_action(
        _first_string(record, body, result, keys=("action", "status", "msg", "message"))
    )
    duration = _duration_seconds(
        _first_present(record, body, result, keys=("duration_seconds", "durationSeconds", "duration", "call_duration_seconds", "callDurationSeconds"))
    )
    flags = _string_list(record.get("quality_flags")) + _string_list(record.get("data_quality_flags"))
    if not action and duration is None:
        flags.append("missing_call_action")
    return _CallEvent(
        person_id=resolved_person,
        occurred_at=occurred_at,
        call_id=call_id,
        device_id=resolved_device,
        contact_id=_first_string(record, body, keys=("contact_id", "contactId", "account")),
        action=action,
        direction=_normalize_direction(_first_string(record, body, keys=("direction", "initiator", "source"))),
        duration_seconds=duration,
        quality_flags=tuple(_dedupe(flags)),
    )


def _build_call_attempts(events: list[_CallEvent]) -> list[_CallAttempt]:
    grouped: dict[tuple[str, str], list[_CallEvent]] = defaultdict(list)
    for event in events:
        grouped[(event.person_id, event.call_id)].append(event)

    attempts: list[_CallAttempt] = []
    for (person_id, call_id), call_events in sorted(grouped.items()):
        ordered = sorted(call_events, key=lambda item: item.occurred_at)
        first = ordered[0]
        last = ordered[-1]
        attempt = _CallAttempt(
            person_id=person_id,
            call_id=call_id,
            started_at=first.occurred_at,
            ended_at=last.occurred_at,
            device_id=first.device_id,
            contact_id=first.contact_id,
            direction=first.direction,
            quality_flags=set(),
        )
        for event in ordered:
            if event.device_id and not attempt.device_id:
                attempt.device_id = event.device_id
            if event.contact_id and not attempt.contact_id:
                attempt.contact_id = event.contact_id
            if event.direction and not attempt.direction:
                attempt.direction = event.direction
            if event.quality_flags and attempt.quality_flags is not None:
                attempt.quality_flags.update(event.quality_flags)
            action = event.action or ""
            if action in ANSWER_ACTIONS:
                attempt.answered = True
            if action in REJECT_ACTIONS:
                attempt.rejected = True
            if action in BUSY_ACTIONS:
                attempt.busy = True
            if action in TIMEOUT_ACTIONS:
                attempt.bell_timeout = True
            if action in CANCEL_ACTIONS:
                attempt.canceled = True
            if event.duration_seconds is not None and event.duration_seconds > 0:
                attempt.duration_seconds = max(attempt.duration_seconds, event.duration_seconds)
                attempt.answered = True
        attempts.append(attempt)
    return attempts


def _daily_features_for_day(
    person_id: str,
    day: date,
    attempts: list[_CallAttempt],
    *,
    config: AggregationConfig,
) -> dict[str, Any]:
    day_start = datetime.combine(day, time.min, tzinfo=config.timezone_info)
    day_end = datetime.combine(day, time.max, tzinfo=config.timezone_info)
    window_start = day_start - timedelta(days=6)
    daily_attempts = [item for item in attempts if day_start <= item.started_at <= day_end]
    rolling = [item for item in attempts if window_start <= item.started_at <= day_end]
    flags = _dedupe(flag for item in rolling for flag in (item.quality_flags or set()))
    call_count = len(rolling)
    answered_count = sum(1 for item in rolling if item.answered)
    duration_minutes = sum(item.duration_seconds for item in rolling) / 60.0
    active_count = sum(1 for item in rolling if item.direction == "elder_to_family")
    missed_count = sum(
        1
        for item in rolling
        if not item.answered and (item.rejected or item.busy or item.bell_timeout)
    )
    data_quality = "valid"
    if not rolling:
        data_quality = "data_missing"
        flags.append("no_call_records_in_7d")

    return {
        "person_id": person_id,
        "date": day.isoformat(),
        "window_start_date": window_start.date().isoformat(),
        "window_end_date": day.isoformat(),
        "start_time": (min((item.started_at for item in daily_attempts), default=day_start)).isoformat(),
        "end_time": (max((item.ended_at for item in daily_attempts), default=day_end)).isoformat(),
        "device_id": _first_nonempty(item.device_id for item in daily_attempts + rolling),
        "daily_call_count": len(daily_attempts),
        "daily_answered_call_count": sum(1 for item in daily_attempts if item.answered),
        "daily_call_duration_minutes": _round(sum(item.duration_seconds for item in daily_attempts) / 60.0),
        "call_count_7d": call_count,
        "answered_call_count_7d": answered_count,
        "call_answer_rate_7d": _round(answered_count / call_count) if call_count else None,
        "call_duration_minutes_7d": _round(duration_minutes),
        "active_call_count_7d": active_count,
        "rejected_call_count_7d": sum(1 for item in rolling if item.rejected),
        "busy_call_count_7d": sum(1 for item in rolling if item.busy),
        "bell_timeout_count_7d": sum(1 for item in rolling if item.bell_timeout),
        "canceled_call_count_7d": sum(1 for item in rolling if item.canceled),
        "missed_call_count_7d": missed_count,
        "data_quality": data_quality,
        "quality_score": 1.0 if data_quality == "valid" else 0.0,
        "quality_flags": flags,
        "baseline_eligible": data_quality == "valid",
    }


def _normalize_daily_feature(
    record: Mapping[str, Any],
    *,
    person_id: str,
    device_id: str | None,
) -> dict[str, Any]:
    item = dict(record)
    item["person_id"] = str(item.get("person_id") or person_id)
    item["date"] = _parse_date(item.get("date"), "daily.date").isoformat()
    item.setdefault("device_id", device_id)
    item.setdefault("data_quality", "valid")
    item.setdefault("quality_flags", [])
    item.setdefault("baseline_eligible", _baseline_eligible(item))
    if item.get("call_answer_rate_7d") is None:
        call_count = _optional_number(item.get("call_count_7d"))
        answered = _optional_number(item.get("answered_call_count_7d"))
        if call_count and answered is not None:
            item["call_answer_rate_7d"] = _round(answered / call_count)
    return item


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


def _social_factors(details: Mapping[str, Mapping[str, Any]]) -> list[str]:
    labels = {
        "call_count_7d": "call_frequency_decline",
        "answered_call_count_7d": "answered_call_decline",
        "call_answer_rate_7d": "call_answer_rate_decline",
        "call_duration_minutes_7d": "call_duration_decline",
        "active_call_count_7d": "active_call_decline",
        "missed_call_count_7d": "missed_call_increase",
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
        if record.get("social_withdrawal_score") is None:
            flags.append("social_connection_score_unavailable")
    return _dedupe(flags)


def _event_time(
    record: Mapping[str, Any],
    body: Mapping[str, Any],
    header: Mapping[str, Any],
    config: AggregationConfig,
    record_number: int,
) -> datetime:
    value = _first_present(
        record,
        body,
        header,
        keys=("observed_at", "timestamp", "timestamp_ms", "messageTime", "message_time", "event_time", "occurred_at"),
    )
    parsed = _parse_datetime(value, config)
    if parsed is None:
        raise ValueError(f"social call record {record_number}: requires a usable event timestamp")
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
        normalized = value.strip().replace(" ", "T")
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        if normalized.isdigit():
            return _parse_datetime(float(normalized), config)
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=config.timezone_info)
    return parsed.astimezone(config.timezone_info)


def _duration_seconds(value: Any) -> float | None:
    number = _optional_number(value)
    if number is not None:
        return number
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if ":" not in text:
        return None
    try:
        parts = [float(part) for part in text.split(":")]
    except ValueError:
        return None
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    return None


def _normalize_action(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().lower()
    return text.replace("-", "_").replace(" ", "_")


def _normalize_direction(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().lower()
    if text in {"device", "s10", "elder", "elderly", "elder_to_family", "outbound"}:
        return "elder_to_family"
    if text in {"client", "family", "family_to_elder", "inbound"}:
        return "family_to_elder"
    return text


def _event_to_dict(event: _CallEvent) -> dict[str, Any]:
    return {
        "person_id": event.person_id,
        "occurred_at": event.occurred_at.isoformat(),
        "call_id": event.call_id,
        "device_id": event.device_id,
        "contact_id": event.contact_id,
        "action": event.action,
        "direction": event.direction,
        "duration_seconds": event.duration_seconds,
        "quality_flags": list(event.quality_flags),
    }


def _first_present(*mappings: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for mapping in mappings:
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
    return None


def _first_string(*mappings: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    value = _first_present(*mappings, keys=keys)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_nonempty(values: Iterable[str | None]) -> str | None:
    for value in values:
        if value:
            return value
    return None


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _parse_date(value: Any, path: str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError(f"{path} must use YYYY-MM-DD format")
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError(f"{path} must use YYYY-MM-DD format")
    return parsed


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _round(value: float) -> float:
    return round(float(value), 4)
