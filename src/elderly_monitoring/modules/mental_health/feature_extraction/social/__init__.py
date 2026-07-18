"""Call count, answer rate, and call-duration social connection features."""

from elderly_monitoring.modules.mental_health.feature_extraction.social.connection import (
    SocialConnectionScore,
    aggregate_social_connection_daily,
    build_social_connection_result,
    normalize_call_events,
    score_social_connection_day,
)

__all__ = [
    "SocialConnectionScore",
    "aggregate_social_connection_daily",
    "build_social_connection_result",
    "normalize_call_events",
    "score_social_connection_day",
]
