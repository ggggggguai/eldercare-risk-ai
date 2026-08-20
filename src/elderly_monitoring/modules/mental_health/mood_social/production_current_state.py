"""Production adapter for the user-authorized r5 current-state replacement.

The sealed r5 package remains immutable.  This module records the later
production decision, exposes r5 dual-head probabilities on signatures that
passed its integration checks, and keeps MH-20260802-013 as a versioned safety
fallback for blocked/unsupported signatures or artifact failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from elderly_monitoring.modules.mental_health.mood_social.config import (
    MoodSocialConfig,
)
from elderly_monitoring.modules.mental_health.mood_social.pipeline import (
    _attention_level,
    _attention_trend,
    _score_half_up,
)
from elderly_monitoring.modules.mental_health.mood_social.schemas import (
    MOOD_SOCIAL_MODEL_VERSION,
    MOOD_SOCIAL_PRODUCTION_MODEL_VERSION,
    MoodSocialInferRequest,
    MoodSocialInferResponse,
    _summary_for_contributions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[5]
R5_MODEL_VERSION = "mood-social-current-state-v3.3.3-r5"
R5_PACKAGE_RUN_ID = "MH-20260812-R5-001"
R5_PACKAGE_RELATIVE = Path(
    "models/mental_health/mood_social/v3.3.3-r5/packages/MH-20260812-R5-001"
)
PRODUCTION_SELECTION_RELATIVE = Path(
    "configs/runtime/mood_social_current_state_production.json"
)
LEGACY_PACKAGE_RUN_ID = "MH-20260802-013"
PRIMARY_LIMITATION = "r5_reused_cohort_confirmation_without_real_device_parity"


class ProductionSelectionError(RuntimeError):
    """Raised when the explicit production selection is absent or inconsistent."""


@dataclass(frozen=True)
class CurrentStateProductionSelection:
    primary_model_version: str
    primary_package_run_id: str
    fallback_model_version: str
    fallback_package_run_id: str
    primary_signatures: tuple[str, ...]
    fallback_signatures: tuple[str, ...]


def load_current_state_production_selection(
    path: Path | None = None,
) -> CurrentStateProductionSelection:
    selection_path = path or PROJECT_ROOT / PRODUCTION_SELECTION_RELATIVE
    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProductionSelectionError("current-state production selection unavailable") from exc
    expected = {
        "schema_version": "mood_social_current_state_production_v1",
        "active": True,
        "decision": "user_authorized_best_deployable_model_replacement",
        "primary_model_version": R5_MODEL_VERSION,
        "primary_package_run_id": R5_PACKAGE_RUN_ID,
        "fallback_model_version": MOOD_SOCIAL_MODEL_VERSION,
        "fallback_package_run_id": LEGACY_PACKAGE_RUN_ID,
        "primary_signatures": ["profile", "sleep"],
        "fallback_signatures": ["joint", "unsupported"],
        "mixed_signature_policy": (
            "project_to_sleep_else_profile_without_relabeling_camera"
        ),
        "phq9_input_allowed": False,
        "forecast_model_changed": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ProductionSelectionError("current-state production selection changed")
    return CurrentStateProductionSelection(
        primary_model_version=payload["primary_model_version"],
        primary_package_run_id=payload["primary_package_run_id"],
        fallback_model_version=payload["fallback_model_version"],
        fallback_package_run_id=payload["fallback_package_run_id"],
        primary_signatures=tuple(payload["primary_signatures"]),
        fallback_signatures=tuple(payload["fallback_signatures"]),
    )


@lru_cache(maxsize=2)
def _load_r5_runner(package_directory: str) -> Any:
    # Keep the heavy sklearn/joblib graph out of API import and validation paths.
    from elderly_monitoring.modules.mental_health.mood_social.r5_release import (
        R5IntegrationRunner,
    )

    return R5IntegrationRunner.from_package(Path(package_directory))


def predict_primary_current_state(
    request: MoodSocialInferRequest,
) -> dict[str, Any]:
    """Score the immutable r5 package after validating the production decision."""

    selection = load_current_state_production_selection()
    runner = _load_r5_runner(str((PROJECT_ROOT / R5_PACKAGE_RELATIVE).resolve()))
    result = runner.predict(request)
    # Real backend requests normally contain a profile alongside device evidence,
    # while the sealed r5 package only approved exact profile-only and sleep-only
    # signatures.  Prefer the higher-signal sleep signature when present; otherwise
    # project to profile-only.  Camera evidence is never relabelled as sleep/profile.
    if not result["available"]:
        if request.current_daily_features.sleep is not None:
            empty_profile = request.profile.model_copy(
                update={name: None for name in request.profile.model_fields}
            )
            sleep_only = request.model_copy(
                update={
                    "profile": empty_profile,
                    "current_daily_features": request.current_daily_features.model_copy(
                        update={"activity": None, "physiology": None, "social": None}
                    ),
                    "history_daily_features": [],
                }
            )
            result = runner.predict(sleep_only)
            result["production_projection"] = "sleep_only"
        else:
            profile_only = request.model_copy(
                update={
                    "current_daily_features": request.current_daily_features.model_copy(
                        update={
                            "activity": None,
                            "sleep": None,
                            "physiology": None,
                            "social": None,
                        }
                    ),
                    "history_daily_features": [],
                }
            )
            projected = runner.predict(profile_only)
            if projected["available"]:
                result = projected
                result["production_projection"] = "profile_only"
    if result["package_run_id"] != selection.primary_package_run_id:
        raise ProductionSelectionError("r5 response package does not match selection")
    return result


def _legacy_with_metadata(
    response: MoodSocialInferResponse,
    *,
    reason: str,
    signature: str,
) -> MoodSocialInferResponse:
    probability = response.attention_index if response.available else None
    return response.model_copy(
        update={
            "model_version": MOOD_SOCIAL_PRODUCTION_MODEL_VERSION,
            "current_state_model_version": MOOD_SOCIAL_MODEL_VERSION,
            "current_state_package_run_id": LEGACY_PACKAGE_RUN_ID,
            "current_state_fallback_used": True,
            "current_state_fallback_reason": reason,
            "current_state_evidence_signature": signature,
            "current_state_projection": "none",
            "probability_phq_ge5": None,
            "probability_phq_ge10": probability,
        }
    )


def _minimal_r5_scaffold(
    request: MoodSocialInferRequest,
    *,
    signature: str,
    probability: float,
    config: MoodSocialConfig,
) -> MoodSocialInferResponse:
    if signature == "profile":
        used_sources = ["profile"]
        covered_domains = ["social_context"]
        domain_scores = {
            "activity": None,
            "sleep": None,
            "physiology": None,
            "activity_sleep_joint": None,
            "social_context": probability,
            "personal_change": {"activity": None, "sleep": None, "social": None},
        }
    elif signature == "sleep":
        used_sources = ["sleep_device"]
        covered_domains = ["sleep"]
        domain_scores = {
            "activity": None,
            "sleep": probability,
            "physiology": None,
            "activity_sleep_joint": None,
            "social_context": None,
            "personal_change": {"activity": None, "sleep": None, "social": None},
        }
    else:  # guarded by the r5 production selection
        raise ProductionSelectionError("r5 returned an unapproved primary signature")
    return MoodSocialInferResponse.model_validate(
        {
            "schema_version": "mood_social_infer_response_v3",
            "request_id": request.request_id,
            "person_id": request.person_id,
            "target_date": request.target_date,
            "module": "mood_social_attention",
            "model_version": MOOD_SOCIAL_PRODUCTION_MODEL_VERSION,
            "available": True,
            "evidence_scope": "proxy_label_supported",
            "attention_index": probability,
            "attention_score": _score_half_up(probability),
            "attention_level": _attention_level(probability, config),
            "confidence": _signature_confidence(request, signature),
            "used_sources": used_sources,
            "covered_domains": covered_domains,
            "domain_scores": domain_scores,
            "model_contributions": [],
            "trend": _attention_trend(request, probability, config),
            "summary": _summary_for_contributions([]),
            "limitations": [PRIMARY_LIMITATION, "legacy_explanation_scaffold_unavailable"],
            "diagnosis": False,
        }
    )


def _signature_confidence(request: MoodSocialInferRequest, signature: str) -> float:
    if signature == "profile":
        values = request.profile.model_dump().values()
    elif signature == "sleep" and request.current_daily_features.sleep is not None:
        values = request.current_daily_features.sleep.model_dump().values()
    else:
        return 0.0
    values = list(values)
    if not values:
        return 0.0
    return round(sum(value is not None for value in values) / len(values), 6)


def apply_primary_current_state(
    request: MoodSocialInferRequest,
    *,
    config: MoodSocialConfig,
    legacy_response: MoodSocialInferResponse | None,
    primary_predictor: Callable[[MoodSocialInferRequest], Mapping[str, Any]] = (
        predict_primary_current_state
    ),
) -> MoodSocialInferResponse:
    """Choose r5 on approved signatures and otherwise return the legacy fallback."""

    try:
        candidate = dict(primary_predictor(request))
    except Exception as exc:
        if legacy_response is None:
            raise
        return _legacy_with_metadata(
            legacy_response,
            reason=f"primary_artifact_unavailable:{type(exc).__name__}",
            signature="legacy",
        )

    signature = str(candidate.get("evidence_signature", "unsupported"))
    if not candidate.get("available"):
        if legacy_response is None:
            raise ProductionSelectionError(
                "primary route requires fallback, but fallback is unavailable"
            )
        return _legacy_with_metadata(
            legacy_response,
            reason=str(candidate.get("reason") or "primary_signature_unavailable"),
            signature=signature,
        )

    probability5 = float(candidate["probability_ge5"])
    probability10 = float(candidate["probability_ge10"])
    if not 0.0 <= probability10 <= probability5 <= 1.0:
        raise ProductionSelectionError("r5 dual-head probability ordering failed")
    # Do not reuse legacy branch scores or explanations: they were produced by
    # a different ranking model and could falsely claim that unused evidence
    # contributed to the r5 probability.
    scaffold = _minimal_r5_scaffold(
        request,
        signature=signature,
        probability=probability10,
        config=config,
    )
    limitations = list(scaffold.limitations)
    if PRIMARY_LIMITATION not in limitations:
        limitations.append(PRIMARY_LIMITATION)
    return scaffold.model_copy(
        update={
            "model_version": MOOD_SOCIAL_PRODUCTION_MODEL_VERSION,
            "attention_index": probability10,
            "attention_score": _score_half_up(probability10),
            "attention_level": _attention_level(probability10, config),
            "trend": _attention_trend(request, probability10, config),
            "limitations": limitations,
            "current_state_model_version": R5_MODEL_VERSION,
            "current_state_package_run_id": R5_PACKAGE_RUN_ID,
            "current_state_fallback_used": False,
            "current_state_fallback_reason": None,
            "current_state_evidence_signature": signature,
            "current_state_projection": str(candidate.get("production_projection", "none")),
            "probability_phq_ge5": probability5,
            "probability_phq_ge10": probability10,
        }
    )


__all__ = [
    "CurrentStateProductionSelection",
    "LEGACY_PACKAGE_RUN_ID",
    "PRODUCTION_SELECTION_RELATIVE",
    "ProductionSelectionError",
    "apply_primary_current_state",
    "load_current_state_production_selection",
    "predict_primary_current_state",
]
