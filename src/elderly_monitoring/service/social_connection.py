from __future__ import annotations

from typing import Any

from elderly_monitoring.modules.mental_health.feature_extraction.social import (
    build_social_connection_result,
)
from elderly_monitoring.service.schemas import SocialConnectionRequest


def build_social_connection_service_result(request: SocialConnectionRequest) -> dict[str, Any]:
    return build_social_connection_result(
        person_id=request.person_id,
        call_events=request.call_events,
        daily_features=request.daily_features,
        history_daily_features=request.history_daily_features,
        requested_date=request.date,
        device_id=request.device_id,
    )
