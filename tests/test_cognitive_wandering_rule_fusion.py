from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.v35_schemas import (
    CognitiveV35InferResponse,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_wandering_fusion import (
    COGNITIVE_WANDERING_FUSION_PATH,
    CognitiveWanderingFusionRequest,
    create_cognitive_wandering_fusion_router,
    fuse_cognitive_wandering_attention,
    load_fusion_policy,
    validate_rule_truth_table,
)


TZ = timezone(timedelta(hours=8))
EVALUATED_AT = datetime(2026, 8, 20, 12, 0, tzinfo=TZ)
IDENTITY = {
    "model_id": "topowander-fixture",
    "model_sha256": "1" * 64,
    "config_id": "wandering-fixture-config",
    "config_sha256": "2" * 64,
    "policy_id": "wandering-fixture-policy",
    "policy_sha256": "3" * 64,
}
SOURCE_REF = {
    "ref_type": "fixture",
    "ref_id": "fixture-source",
    "artifact_path": None,
    "sha256": "4" * 64,
}
METRICS = (
    "tracking_coverage",
    "presence_hours",
    "direct_count_per_presence_hour",
    "pacing_count_per_presence_hour",
    "lapping_count_per_presence_hour",
    "random_count_per_presence_hour",
    "wandering_like_count_per_presence_hour",
    "wandering_like_duration_seconds_per_presence_hour",
    "uncertain_wandering_like_count_per_presence_hour",
    "night_wandering_like_ratio",
)
ELEVATION_METRICS = {
    "wandering_like_count_per_presence_hour",
    "wandering_like_duration_seconds_per_presence_hour",
    "night_wandering_like_ratio",
}


def _cognitive_response(
    level: str = "attention",
    *,
    status: str = "completed",
    adaptation_mode: str = "standard_picture_description",
) -> CognitiveV35InferResponse:
    score = {"normal": 25.0, "attention": 55.0, "high_attention": 82.0}[level]
    voice = adaptation_mode == "voice_prompt_adaptation"
    task_status = "degraded" if status == "degraded" else "completed"
    task_quality = [
        {
            "task_slot": slot,
            "status": task_status,
            "used_modalities": ["audio", "text"] if task_status == "degraded" else ["audio", "text", "face"],
            "modality_quality": {"audio": 0.9, "text": 0.8, "face": 0.0 if task_status == "degraded" else 0.7},
            "asr_status": "completed",
            "warnings": [],
        }
        for slot in (1, 2, 3)
    ]
    return CognitiveV35InferResponse.model_validate(
        {
            "schema_version": "cognitive_subject_infer_response_v1",
            "request_id": "cog-source-001",
            "task_protocol": (
                "cogvoice_scene_narration_3task_v1"
                if voice
                else "cogpic_picture_description_3task_v1"
            ),
            "stimulus_mode": "audio_prompt" if voice else "visual_image",
            "adaptation_mode": adaptation_mode,
            "status": status,
            "cognitive_clue_score": score,
            "cognitive_clue_level": level,
            "confidence": 0.6 if voice else 0.82,
            "research_outputs": None,
            "used_modalities": ["audio", "text"] if status == "degraded" else ["audio", "text", "face"],
            "modality_quality": {"audio": 0.9, "text": 0.8, "face": 0.0 if status == "degraded" else 0.7},
            "task_quality": task_quality,
            "asr_status": "completed",
            "model_version": "cognitive-mm-v3.5.0",
            "warnings": (
                [
                    {
                        "code": "voice_prompt_adaptation",
                        "modality": "model",
                        "message": "picture-protocol metrics do not transfer",
                    }
                ]
                if voice
                else []
            ),
        }
    )


def _daily(day: date, *, elevated: bool, context: str | None = None) -> dict[str, object]:
    count = 1 if elevated else 0
    contexts = {} if context is None else {context: count}
    return {
        "schema_version": "wandering-handoff-daily-report-v2",
        "module": "mental_health",
        "record_id": f"daily-{day.isoformat()}",
        "person_id": "elder-001",
        "session_id": None,
        "source_video_id": None,
        "local_date": day,
        "timezone": "Asia/Shanghai",
        "status": "ready",
        "presence_seconds": 3600.0,
        "tracking_coverage_seconds": 3600.0,
        "tracking_coverage": 1.0,
        "qc_coverage": 1.0 if elevated else None,
        "episode_counts": {
            "direct": 1,
            "pacing": count,
            "lapping": 0,
            "random": 0,
            "wandering_like": count,
        },
        "episode_duration_seconds": {
            "direct": 10.0,
            "pacing": float(count * 60),
            "lapping": 0.0,
            "random": 0.0,
            "wandering_like": float(count * 60),
        },
        "confidence_tiers": {
            "high_confidence": {"count": count, "duration_seconds": float(count * 60)},
            "uncertain": {"count": 0, "duration_seconds": 0.0},
            "unavailable": {"count": 0, "duration_seconds": 0.0},
            "error": {"count": 0, "duration_seconds": 0.0},
        },
        "rates_per_presence_hour": {
            "wandering_like_count": float(count),
            "wandering_like_duration_seconds": float(count * 60),
        },
        "night_wandering_like_ratio": 0.2 if elevated else None,
        "context_counts": contexts,
        "unavailable_count": 0,
        "error_count": 0,
        "baseline_readiness": "stable_ready",
        "quality_flags": [],
        "identity": dict(IDENTITY),
        "source_refs": [dict(SOURCE_REF)],
    }


def _deviation(day: date, *, elevated: bool) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for name in METRICS:
        is_target = name in ELEVATION_METRICS
        delta = 0.5 if elevated and is_target else -0.5
        metrics[name] = {
            "status": "ready",
            "observation_value": 1.0 if elevated and is_target else 0.0,
            "reference_count": 7,
            "reference_median": 0.25,
            "reference_p90": 0.5,
            "delta_from_median": 0.75 if elevated and is_target else -0.25,
            "delta_from_p90": delta,
        }
    return {
        "schema_version": "wandering-handoff-baseline-deviation-v1",
        "module": "mental_health",
        "record_id": f"deviation-{day.isoformat()}",
        "deviation_id": f"deviation-{day.isoformat()}",
        "profile_id": f"profile-{day.isoformat()}",
        "person_id": "elder-001",
        "session_id": None,
        "source_video_id": None,
        "local_date": day,
        "timezone": "Asia/Shanghai",
        "status": "ready",
        "readiness_status": "stable_ready",
        "metrics": metrics,
        "quality_flags": [],
        "identity": dict(IDENTITY),
        "source_refs": [dict(SOURCE_REF)],
    }


def _request(
    *,
    cognitive_level: str | None = "attention",
    elevated_days: int = 0,
    context: str | None = None,
    cognitive_observed_at: datetime | None = None,
) -> CognitiveWanderingFusionRequest:
    days = [EVALUATED_AT.date() - timedelta(days=offset) for offset in range(3)]
    daily = [
        _daily(day, elevated=index < elevated_days, context=context)
        for index, day in enumerate(days)
    ]
    deviations = [
        _deviation(day, elevated=index < elevated_days)
        for index, day in enumerate(days)
    ]
    cognitive = (
        {
            "observed_at": cognitive_observed_at or EVALUATED_AT - timedelta(days=1),
            "result": _cognitive_response(cognitive_level),
            "unavailable_reason": None,
        }
        if cognitive_level is not None
        else {
            "observed_at": None,
            "result": None,
            "unavailable_reason": "s10_not_completed",
        }
    )
    return CognitiveWanderingFusionRequest.model_validate(
        {
            "schema_version": "cognitive_wandering_fusion_request_v1",
            "request_id": "fusion-001",
            "person_id": "elder-001",
            "evaluated_at": EVALUATED_AT,
            "cognitive": cognitive,
            "wandering_daily_reports": daily,
            "wandering_baseline_deviations": deviations,
            "policy_version": "cognitive-wandering-rule-fusion-v1",
        }
    )


def test_policy_and_truth_table_are_deterministic_and_monotonic() -> None:
    policy = load_fusion_policy()
    assert policy.policy_version == "cognitive-wandering-rule-fusion-v1"
    audit = validate_rule_truth_table(policy)
    assert audit == {"status": "passed", "row_count": 12, "violations": []}


def test_attention_cognitive_plus_persistent_wandering_advances_one_level() -> None:
    result = fuse_cognitive_wandering_attention(_request(elevated_days=3))
    assert result.attention_level == "high_attention"
    assert result.attention_score == 85.0
    assert result.wandering_component.support_level == 2
    assert "persistent_wandering_support_incremented_one_level" in result.decision_reasons
    assert result.diagnosis is False
    assert result.attention_score_semantics == "rule-derived-not-diagnostic-probability"


def test_normal_cognitive_can_advance_only_to_attention() -> None:
    result = fuse_cognitive_wandering_attention(
        _request(cognitive_level="normal", elevated_days=3)
    )
    assert result.attention_level == "attention"
    assert result.cognitive_component.level == "normal"


def test_wandering_never_downranks_cognitive_and_no_support_is_explained() -> None:
    result = fuse_cognitive_wandering_attention(
        _request(cognitive_level="high_attention", elevated_days=0)
    )
    assert result.attention_level == "high_attention"
    assert "cognitive_clue_not_supported_by_wandering_no_downrank" in result.discordance_flags


def test_wandering_only_is_capped_at_attention() -> None:
    result = fuse_cognitive_wandering_attention(
        _request(cognitive_level=None, elevated_days=3)
    )
    assert result.status == "degraded"
    assert result.attention_level == "attention"
    assert "wandering_support_only_capped_at_attention" in result.decision_reasons


def test_no_cognitive_and_no_persistent_wandering_is_insufficient() -> None:
    result = fuse_cognitive_wandering_attention(
        _request(cognitive_level=None, elevated_days=0)
    )
    assert result.status == "insufficient_evidence"
    assert result.attention_level is None
    assert result.attention_score is None


def test_context_is_explanatory_only_and_never_silently_downranks() -> None:
    result = fuse_cognitive_wandering_attention(
        _request(cognitive_level="normal", elevated_days=3, context="exercise")
    )
    assert result.attention_level == "attention"
    assert "purpose_context_observed_not_used_for_downranking" in result.discordance_flags


def test_stale_cognitive_is_not_used_as_current_evidence() -> None:
    result = fuse_cognitive_wandering_attention(
        _request(
            cognitive_level="high_attention",
            elevated_days=0,
            cognitive_observed_at=EVALUATED_AT - timedelta(days=31),
        )
    )
    assert result.status == "insufficient_evidence"
    assert result.attention_level is None
    assert "cognitive_evidence_stale" in result.warnings


def test_identity_mismatch_fails_closed() -> None:
    payload = _request().model_dump(mode="python")
    payload["wandering_baseline_deviations"][0]["person_id"] = "elder-002"
    with pytest.raises(ValidationError):
        CognitiveWanderingFusionRequest.model_validate(payload)


def test_candidate_api_requires_authentication_and_returns_contract() -> None:
    app = FastAPI()
    app.include_router(create_cognitive_wandering_fusion_router(api_token="secret"))
    client = TestClient(app)
    payload = _request(elevated_days=2).model_dump(mode="json")
    assert client.post(COGNITIVE_WANDERING_FUSION_PATH, json=payload).status_code == 401
    response = client.post(
        COGNITIVE_WANDERING_FUSION_PATH,
        json=payload,
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "cognitive_wandering_fusion_response_v1"
    assert body["attention_level"] == "high_attention"
    assert body["deployment_status"] == "local_candidate_pending_home_validation"
