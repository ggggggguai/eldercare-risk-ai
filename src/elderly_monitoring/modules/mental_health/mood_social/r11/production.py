"""Production alias for the frozen R11 evidence orchestrator.

R11 did not promote new supervised weights.  This endpoint promotes the
engineering-approved orchestrator around the strongest frozen R9/R7/R8 nodes
without changing their artifacts or their evaluation claims.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .release import infer_mood_social_r11_candidate
from .schemas import R11CandidateRequest, R11CandidateResponse


MOOD_SOCIAL_R11_PRODUCTION_INFER_PATH = (
    "/v1/mental-health/mood-social/r11-production/infer"
)


def infer_mood_social_r11_production(
    request: R11CandidateRequest | dict[str, Any],
    *,
    package_directory: Path | None = None,
) -> dict[str, Any]:
    """Run the frozen R11 graph as the production current-state primary.

    Only runtime routing metadata changes.  Model identity, package identity,
    probabilities, rules and evidence remain byte-for-byte attributable to the
    frozen R11 package and its R9/R7/R8 model nodes.
    """

    response = infer_mood_social_r11_candidate(
        request,
        package_directory=package_directory,
    )
    response["fallback_chain"] = {
        **response["fallback_chain"],
        "production": "R11 primary; fail-closed; R5 manual rollback only",
    }
    return R11CandidateResponse.model_validate(response).model_dump(mode="json")


__all__ = [
    "MOOD_SOCIAL_R11_PRODUCTION_INFER_PATH",
    "infer_mood_social_r11_production",
]
