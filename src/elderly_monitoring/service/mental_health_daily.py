from __future__ import annotations

from typing import Any, Mapping

from elderly_monitoring.modules.mental_health.baseline import score_daily_mental_health
from elderly_monitoring.modules.mental_health.config import load_mental_health_config
from elderly_monitoring.modules.mental_health.pipeline import MentalHealthRiskPipeline
from elderly_monitoring.service.schemas import MentalHealthDailyRiskRequest


def build_mental_health_daily_risk_result(request: MentalHealthDailyRiskRequest) -> dict[str, Any]:
    """Score daily mental-safety trends from already aggregated day-level features."""
    config = load_mental_health_config()
    person_id = request.person_id.strip()
    history = [_with_person_id(record, person_id) for record in request.history_daily_features]
    current = [_with_person_id(record, person_id) for record in request.current_daily_features]
    if not current:
        raise ValueError("current_daily_features must contain at least one daily record")

    baseline_results = score_daily_mental_health(history, current, config=config)
    pipeline = MentalHealthRiskPipeline(config)
    results: list[dict[str, Any]] = []
    current_by_key = {
        (str(record["person_id"]), str(record["date"])): record
        for record in current
    }
    for baseline in baseline_results:
        event = pipeline.predict_mental_safety(baseline)
        key = (str(baseline["person_id"]), str(baseline["date"]))
        results.append(
            {
                "date": key[1],
                "daily_features": current_by_key.get(key, {}),
                "baseline_features": baseline,
                "event": event.to_dict(),
            }
        )
    return {
        "schema_version": "mental_health_daily_risk_service_v1",
        "model_version": config.version,
        "person_id": person_id,
        "results": sorted(results, key=lambda item: item["date"]),
        "medical_disclaimer": "该结果为行为趋势提示，不构成医学诊断。",
    }


def _with_person_id(record: Mapping[str, Any], person_id: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError("daily feature records must be objects")
    copied = dict(record)
    copied.setdefault("person_id", person_id)
    if str(copied["person_id"]).strip() != person_id:
        raise ValueError("daily feature records must use the request person_id")
    return copied
