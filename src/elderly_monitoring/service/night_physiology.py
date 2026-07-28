from __future__ import annotations

from typing import Any

from elderly_monitoring.modules.mental_health.feature_extraction.physiology import (
    build_night_physiology_result,
)
from elderly_monitoring.service.schemas import NightPhysiologyRequest


def build_night_physiology_service_result(request: NightPhysiologyRequest) -> dict[str, Any]:
    return build_night_physiology_result(
        person_id=request.person_id,
        daily_features=request.daily_features,
        history_daily_features=request.history_daily_features,
        requested_date=request.date,
        device_id=request.device_id,
        device_serial=request.device_serial,
    )
