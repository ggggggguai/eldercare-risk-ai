"""Dedicated V3.4 forecast API used beside the R11 current-state primary."""

from __future__ import annotations

from typing import Any

from .forecast_runtime import predict_mood_forecast, unavailable_forecast
from .schemas import MoodSocialForecastResult, MoodSocialInferRequest


MOOD_SOCIAL_FORECAST_INFER_PATH = "/v1/mental-health/mood-social/forecast/infer"


def infer_mood_social_forecast(
    request: MoodSocialInferRequest | dict[str, Any],
) -> MoodSocialForecastResult:
    parsed = (
        request
        if isinstance(request, MoodSocialInferRequest)
        else MoodSocialInferRequest.model_validate(request)
    )
    try:
        result = predict_mood_forecast(parsed)
    except Exception as exc:
        result = unavailable_forecast(
            f"forecast runtime failed safely: {type(exc).__name__}"
        )
    return MoodSocialForecastResult.model_validate(result)


__all__ = ["MOOD_SOCIAL_FORECAST_INFER_PATH", "infer_mood_social_forecast"]
