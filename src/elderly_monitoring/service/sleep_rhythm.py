from __future__ import annotations

from typing import Any

from elderly_monitoring.modules.mental_health.feature_extraction.sleep import (
    build_sleep_rhythm_result,
)
from elderly_monitoring.service.schemas import SleepRhythmRequest


def build_sleep_rhythm_service_result(request: SleepRhythmRequest) -> dict[str, Any]:
    return build_sleep_rhythm_result(
        person_id=request.person_id,
        reports=request.reports,
        daily_features=request.daily_features,
        body_detect_messages=request.body_detect_messages,
        history_daily_features=request.history_daily_features,
        device_id=request.device_id,
        device_serial=request.device_serial,
        requested_date=request.date,
    )
