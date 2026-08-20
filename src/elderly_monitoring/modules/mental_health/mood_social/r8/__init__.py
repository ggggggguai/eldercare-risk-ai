"""R8 current-state integration: R7 core plus bounded facial-affect support."""

from .facial_affect import (
    FacialAffectSupportAdapter,
    FacialEvidenceLevel,
    aggregate_facial_affect,
)
from .release import (
    MOOD_SOCIAL_R8_CANDIDATE_INFER_PATH,
    R8CandidateRequest,
    R8CandidateResponse,
    R8PackageError,
    infer_mood_social_r8_candidate,
)

__all__ = [
    "FacialAffectSupportAdapter",
    "FacialEvidenceLevel",
    "MOOD_SOCIAL_R8_CANDIDATE_INFER_PATH",
    "R8CandidateRequest",
    "R8CandidateResponse",
    "R8PackageError",
    "aggregate_facial_affect",
    "infer_mood_social_r8_candidate",
]
