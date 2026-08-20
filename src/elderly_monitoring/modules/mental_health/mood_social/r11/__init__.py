"""V3.3.3-R11 adaptive transition research and deterministic evidence orchestration."""

from .contract import R11_FROZEN_DOCUMENT_SHA256, R11_PROTOCOL_VERSION
from .release import (
    MOOD_SOCIAL_R11_CANDIDATE_INFER_PATH,
    R11_PACKAGE_RUN_ID,
    infer_mood_social_r11_candidate,
)
from .production import (
    MOOD_SOCIAL_R11_PRODUCTION_INFER_PATH,
    infer_mood_social_r11_production,
)

__all__ = [
    "MOOD_SOCIAL_R11_CANDIDATE_INFER_PATH",
    "MOOD_SOCIAL_R11_PRODUCTION_INFER_PATH",
    "R11_FROZEN_DOCUMENT_SHA256",
    "R11_PACKAGE_RUN_ID",
    "R11_PROTOCOL_VERSION",
    "infer_mood_social_r11_candidate",
    "infer_mood_social_r11_production",
]
