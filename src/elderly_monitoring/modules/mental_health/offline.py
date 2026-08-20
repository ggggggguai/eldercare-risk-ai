from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.mental_health.adapters import (
    BehaviorObservation,
    MentalHealthDataError,
    adapt_behavior_record,
    adapt_sleep_record,
)
from elderly_monitoring.modules.mental_health.baseline import score_daily_mental_health
from elderly_monitoring.modules.mental_health.config import MentalHealthConfig, load_mental_health_config
from elderly_monitoring.modules.mental_health.daily_aggregation import aggregate_daily_behavior
from elderly_monitoring.modules.mental_health.pipeline import MentalHealthRiskPipeline


_SELF_REPORT_SCORES = (
    "social_withdrawal_score",
    "negative_affect_score",
    "self_report_risk_score",
)


@dataclass(frozen=True)
class LoadedRecords:
    path: Path
    records: tuple[dict[str, Any], ...]
    line_numbers: tuple[int, ...]


def load_jsonl_records(path: Path) -> LoadedRecords:
    records: list[dict[str, Any]] = []
    line_numbers: list[int] = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}: line {line_number}: invalid JSON at column {exc.colno}: {exc.msg}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}: line {line_number}: record must be a JSON object")
                records.append(value)
                line_numbers.append(line_number)
    except OSError as exc:
        raise OSError(f"{path}: unable to read input: {exc}") from exc
    return LoadedRecords(path, tuple(records), tuple(line_numbers))


def load_json_or_jsonl_records(path: Path) -> LoadedRecords:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl_records(path)
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}: line {exc.lineno}: invalid JSON at column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise OSError(f"{path}: unable to read input: {exc}") from exc

    values = value if isinstance(value, list) else [value]
    records: list[dict[str, Any]] = []
    for index, item in enumerate(values, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: line 1, record {index}: expected a JSON object")
        records.append(item)
    return LoadedRecords(path, tuple(records), tuple(1 for _ in records))


def run_daily_mental_health(
    *,
    history_behavior: LoadedRecords,
    current_behavior: LoadedRecords,
    sleep: LoadedRecords | None = None,
    self_report: LoadedRecords | None = None,
    evaluation_time: str | None = None,
    config: MentalHealthConfig | None = None,
) -> list[dict[str, Any]]:
    mental_config = config or load_mental_health_config()
    evaluation_at = _parse_evaluation_time(evaluation_time, mental_config)
    sleep_source = sleep or LoadedRecords(Path("<sleep>"), (), ())
    self_report_source = self_report or LoadedRecords(Path("<self-report>"), (), ())

    history_observations = _adapt_behavior_source(history_behavior, mental_config)
    current_observations = _adapt_behavior_source(current_behavior, mental_config)
    history_daily = _aggregate_history_behavior(
        history_behavior,
        history_observations,
        mental_config,
    )
    current_daily, event_times = _aggregate_current_behavior(
        current_behavior,
        current_observations,
        evaluation_at,
        mental_config,
    )

    sleep_records, sleep_times = _adapt_sleep_source(sleep_source, mental_config)
    self_report_records, self_report_times = _adapt_self_report_source(
        self_report_source,
        mental_config,
    )
    event_times = _merge_latest_times(event_times, sleep_times, self_report_times)

    current_by_key = {
        (str(record["person_id"]), str(record["date"])): dict(record)
        for record in current_daily
    }
    optional_records = [*sleep_records, *self_report_records]
    if evaluation_at is not None:
        evaluation_date = evaluation_at.date().isoformat()
        for record in optional_records:
            key = (str(record["person_id"]), str(record["date"]))
            if key[1] == evaluation_date and key not in current_by_key:
                current_by_key[key] = _empty_daily_record(*key)

    if not current_by_key:
        if evaluation_at is None:
            raise ValueError(
                "current behavior has no usable absolute event time; evaluation_time is required"
            )
        raise ValueError("no current person-day could be identified from the provided inputs")

    current_keys = set(current_by_key)
    history_records: list[Mapping[str, Any]] = list(history_daily)
    current_records_by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {
        key: [record] for key, record in current_by_key.items()
    }
    for record in optional_records:
        key = (str(record["person_id"]), str(record["date"]))
        if key in current_keys:
            current_records_by_key[key].append(record)
        else:
            history_records.append(record)

    merged_current = [
        _merge_daily_records(key, current_records_by_key[key])
        for key in sorted(current_records_by_key, key=lambda item: (item[1], item[0]))
    ]
    baseline_results = score_daily_mental_health(
        history_records,
        merged_current,
        config=mental_config,
    )
    baseline_by_key = {
        (str(record["person_id"]), str(record["date"])): record
        for record in baseline_results
    }
    pipeline = MentalHealthRiskPipeline(mental_config)

    outputs: list[dict[str, Any]] = []
    for daily in merged_current:
        key = (str(daily["person_id"]), str(daily["date"]))
        baseline = dict(baseline_by_key[key])
        observed_at = event_times.get(key)
        if observed_at is not None:
            baseline["timestamp"] = observed_at.isoformat()
        elif evaluation_at is not None:
            baseline["timestamp"] = evaluation_at.isoformat()
            baseline["evidence_window"] = {
                "start_date": key[1],
                "end_date": key[1],
                "start_time": round(evaluation_at.timestamp(), 4),
                "end_time": round(evaluation_at.timestamp(), 4),
            }
        else:
            raise ValueError(
                f"person {key[0]!r} date {key[1]} has no valid observed_at; "
                "evaluation_time is required"
            )

        event = pipeline.predict_from_features(baseline)
        if observed_at is None and event.trigger_event != "insufficient_data":
            raise ValueError(
                f"person {key[0]!r} date {key[1]} has scoreable data but no valid observed_at; "
                "evaluation_time may only timestamp an insufficient_data event"
            )
        outputs.append(
            {
                "person_id": key[0],
                "date": key[1],
                "daily_features": daily,
                "baseline_features": baseline,
                "event": event.to_dict(),
            }
        )
    return sorted(outputs, key=lambda item: (item["date"], item["person_id"]))


def render_jsonl(records: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(
            dict(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for record in records
    )


def write_jsonl_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except OSError as exc:
        raise OSError(f"{path}: unable to write output: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _adapt_behavior_source(
    source: LoadedRecords,
    config: MentalHealthConfig,
) -> list[BehaviorObservation]:
    observations: list[BehaviorObservation] = []
    for record, line_number in zip(source.records, source.line_numbers, strict=True):
        try:
            observations.append(
                adapt_behavior_record(
                    record,
                    config=config.aggregation,
                    record_number=line_number,
                )
            )
        except MentalHealthDataError as exc:
            raise ValueError(
                f"{source.path}: line {line_number}: "
                f"{_strip_record_prefix(str(exc), 'behavior', line_number)}"
            ) from exc
    return observations


def _aggregate_history_behavior(
    source: LoadedRecords,
    observations: list[BehaviorObservation],
    config: MentalHealthConfig,
) -> list[dict[str, Any]]:
    if not source.records:
        return []
    by_person: dict[str, list[tuple[dict[str, Any], BehaviorObservation, int]]] = defaultdict(list)
    for record, observation, line_number in zip(
        source.records,
        observations,
        source.line_numbers,
        strict=True,
    ):
        by_person[observation.person_id].append((record, observation, line_number))

    outputs: list[dict[str, Any]] = []
    for person_id in sorted(by_person):
        items = by_person[person_id]
        if not any(observation.observed_at is not None for _, observation, _ in items):
            line_number = items[0][2]
            raise ValueError(
                f"{source.path}: line {line_number}: field 'observed_at': "
                "history person has no usable absolute event time"
            )
        try:
            outputs.extend(
                aggregate_daily_behavior(
                    [record for record, _, _ in items],
                    config=config.aggregation,
                )
            )
        except (MentalHealthDataError, ValueError) as exc:
            raise ValueError(f"{source.path}: {exc}") from exc
    return sorted(outputs, key=lambda item: (item["date"], item["person_id"]))


def _aggregate_current_behavior(
    source: LoadedRecords,
    observations: list[BehaviorObservation],
    evaluation_at: datetime | None,
    config: MentalHealthConfig,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], datetime]]:
    by_person: dict[str, list[tuple[dict[str, Any], BehaviorObservation, int]]] = defaultdict(list)
    for record, observation, line_number in zip(
        source.records,
        observations,
        source.line_numbers,
        strict=True,
    ):
        by_person[observation.person_id].append((record, observation, line_number))

    outputs: list[dict[str, Any]] = []
    event_times: dict[tuple[str, str], datetime] = {}
    for person_id in sorted(by_person):
        items = by_person[person_id]
        valid_times = [
            observation.observed_at
            for _, observation, _ in items
            if observation.observed_at is not None
        ]
        if valid_times:
            try:
                outputs.extend(
                    aggregate_daily_behavior(
                        [record for record, _, _ in items],
                        config=config.aggregation,
                    )
                )
            except (MentalHealthDataError, ValueError) as exc:
                raise ValueError(f"{source.path}: {exc}") from exc
            for observed_at in valid_times:
                if observed_at is None:
                    continue
                key = (person_id, observed_at.date().isoformat())
                if key not in event_times or observed_at > event_times[key]:
                    event_times[key] = observed_at
            continue

        if evaluation_at is None:
            line_number = items[0][2]
            raise ValueError(
                f"{source.path}: line {line_number}: field 'observed_at': "
                "no usable absolute event time; evaluation_time is required"
            )
        daily = _empty_daily_record(person_id, evaluation_at.date().isoformat())
        daily["data_quality_flags"] = sorted(
            {
                *daily["data_quality_flags"],
                *(
                    flag
                    for _, observation, _ in items
                    for flag in observation.data_quality_flags
                ),
            }
        )
        outputs.append(daily)
    return sorted(outputs, key=lambda item: (item["date"], item["person_id"])), event_times


def _adapt_sleep_source(
    source: LoadedRecords,
    config: MentalHealthConfig,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], datetime]]:
    outputs: list[dict[str, Any]] = []
    event_times: dict[tuple[str, str], datetime] = {}
    for record, line_number in zip(source.records, source.line_numbers, strict=True):
        try:
            adapted = adapt_sleep_record(
                record,
                config=config.aggregation,
                record_number=line_number,
            )
            observed_at = _source_event_time(record, config, "sleep")
        except (MentalHealthDataError, ValueError) as exc:
            raise ValueError(
                f"{source.path}: line {line_number}: "
                f"{_strip_record_prefix(str(exc), 'sleep', line_number)}"
            ) from exc
        outputs.append(adapted)
        if observed_at is not None:
            key = (adapted["person_id"], adapted["date"])
            event_times[key] = max(observed_at, event_times.get(key, observed_at))
    return sorted(outputs, key=_daily_sort_key), event_times


def _adapt_self_report_source(
    source: LoadedRecords,
    config: MentalHealthConfig,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], datetime]]:
    outputs: list[dict[str, Any]] = []
    event_times: dict[tuple[str, str], datetime] = {}
    for record, line_number in zip(source.records, source.line_numbers, strict=True):
        try:
            person_id = _required_person_id(record, "self-report")
            observed_at = _source_event_time(record, config, "self-report")
            local_date = _record_date(record, observed_at, "self-report")
            adapted: dict[str, Any] = {
                "person_id": person_id,
                "date": local_date.isoformat(),
            }
            for field in _SELF_REPORT_SCORES:
                adapted[field] = _optional_unit_score(record, field, "self-report")
            manual_flag = record.get("manual_emergency_flag")
            if manual_flag is not None and not isinstance(manual_flag, bool):
                raise ValueError("field 'manual_emergency_flag' must be a boolean")
            adapted["manual_emergency_flag"] = manual_flag
        except ValueError as exc:
            raise ValueError(f"{source.path}: line {line_number}: {exc}") from exc
        outputs.append(adapted)
        if observed_at is not None:
            key = (person_id, local_date.isoformat())
            event_times[key] = max(observed_at, event_times.get(key, observed_at))
    return sorted(outputs, key=_daily_sort_key), event_times


def _parse_evaluation_time(
    value: str | None,
    config: MentalHealthConfig,
) -> datetime | None:
    if value is None:
        return None
    parsed = _parse_aware_datetime(value, "evaluation_time")
    return parsed.astimezone(config.aggregation.timezone_info)


def _source_event_time(
    record: Mapping[str, Any],
    config: MentalHealthConfig,
    source: str,
) -> datetime | None:
    field = "observed_at" if record.get("observed_at") is not None else "timestamp"
    value = record.get(field)
    if value is None:
        return None
    return _parse_aware_datetime(value, f"field '{field}'").astimezone(
        config.aggregation.timezone_info
    )


def _parse_aware_datetime(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a timezone-aware ISO-8601 timestamp")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field} must be a timezone-aware ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def _record_date(
    record: Mapping[str, Any],
    observed_at: datetime | None,
    source: str,
) -> date:
    value = record.get("date")
    if value is None:
        if observed_at is None:
            raise ValueError(
                f"field 'date' requires YYYY-MM-DD or a timezone-aware observed_at/timestamp"
            )
        return observed_at.date()
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError("field 'date' must use YYYY-MM-DD format")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("field 'date' must be a valid YYYY-MM-DD date") from exc
    if parsed.isoformat() != value:
        raise ValueError("field 'date' must use YYYY-MM-DD format")
    return parsed


def _required_person_id(record: Mapping[str, Any], source: str) -> str:
    value = record.get("person_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"field 'person_id' must be a non-empty stable business ID for {source}")
    return value.strip()


def _optional_unit_score(record: Mapping[str, Any], field: str, source: str) -> float | None:
    value = record.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"field '{field}' must be a finite number in [0, 1] for {source}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"field '{field}' must be a finite number in [0, 1] for {source}")
    return number


def _empty_daily_record(person_id: str, day: str) -> dict[str, Any]:
    return {
        "person_id": person_id,
        "date": day,
        "start_time": None,
        "end_time": None,
        "observation_seconds": 0.0,
        "valid_observation_seconds": 0.0,
        "activity_volume": None,
        "active_ratio": None,
        "nighttime_activity_ratio": None,
        "scene_region_distribution": {},
        "scene_transition_count": 0,
        "observation_coverage": None,
        "data_quality_flags": [
            "active_ratio_unavailable",
            "missing_absolute_time",
            "nighttime_activity_ratio_unavailable",
            "no_observation_seconds",
            "no_valid_observation_seconds",
        ],
    }


def _merge_daily_records(
    key: tuple[str, str],
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    merged: dict[str, Any] = {"person_id": key[0], "date": key[1]}
    data_quality_flags: set[str] = set()
    source_quality_flags: set[str] = set()
    ordered = sorted(records, key=_canonical_record)
    for record in ordered:
        data_quality_flags.update(_string_list(record.get("data_quality_flags")))
        source_quality_flags.update(_string_list(record.get("quality_flags")))
        for field, value in record.items():
            if field in {"person_id", "date", "data_quality_flags", "quality_flags"}:
                continue
            if field not in merged:
                merged[field] = value
            elif value is not None and merged[field] is None:
                merged[field] = value
    merged["data_quality_flags"] = sorted(data_quality_flags)
    merged["quality_flags"] = sorted(source_quality_flags)
    return merged


def _merge_latest_times(
    *mappings: Mapping[tuple[str, str], datetime],
) -> dict[tuple[str, str], datetime]:
    merged: dict[tuple[str, str], datetime] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if key not in merged or value > merged[key]:
                merged[key] = value
    return merged


def _canonical_record(record: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(record),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _daily_sort_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("date", "")),
        str(record.get("person_id", "")),
        _canonical_record(record),
    )


def _strip_record_prefix(message: str, source: str, record_number: int) -> str:
    prefix = f"{source} record {record_number}, "
    return message[len(prefix) :] if message.startswith(prefix) else message


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]
