"""Deterministic, non-diagnostic fusion of V3.5 and wandering evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from .schemas import (
    FUSION_DEPLOYMENT_STATUS,
    FUSION_POLICY_VERSION,
    CognitiveComponentSummary,
    CognitiveWanderingFusionRequest,
    CognitiveWanderingFusionResponse,
    WanderingComponentSummary,
)


DEFAULT_FUSION_POLICY_PATH = (
    Path(__file__).resolve().parents[6]
    / "configs"
    / "modules"
    / "cognitive_wandering_rule_fusion_v1.yaml"
)
EXPECTED_FUSION_POLICY_SHA256 = (
    "909ade571f87f12a1159c6019d17d55b19b7f013e0b743ac82938856fa3f028d"
)
LEVELS = ("normal", "attention", "high_attention")
PURPOSE_CONTEXTS = frozenset({"phone_call", "searching", "cleaning", "exercise", "social"})
NON_HOME_FLAGS = frozenset(
    {
        "real_development_camera_evidence",
        "capture_clock_unavailable_declared_schedule",
        "deterministic_replay_not_real_longitudinal_observation",
    }
)


class CognitiveWanderingFusionError(ValueError):
    """The fusion policy or one of its bound evidence inputs failed closed."""


@dataclass(frozen=True)
class FusionPolicy:
    policy_version: str
    deployment_status: str
    maximum_cognitive_age_days: int
    wandering_window_days: int
    minimum_tracking_coverage: float
    minimum_baseline_ready_days: int
    baseline_ready_states: tuple[str, ...]
    usable_statuses: tuple[str, ...]
    elevation_metrics: tuple[str, ...]
    minimum_elevated_days_for_support: int
    minimum_elevated_days_for_strong_support: int
    strong_support_requires_stable_baseline: bool
    require_high_confidence_wandering_episode: bool
    purpose_context_is_explanatory_only: bool
    support_maximum_increment_levels: int
    support_only_maximum_level: str
    wandering_can_downrank_cognitive: bool
    anchors: Mapping[str, float]
    config_sha256: str


def load_fusion_policy(
    path: str | Path = DEFAULT_FUSION_POLICY_PATH,
) -> FusionPolicy:
    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CognitiveWanderingFusionError(
            f"cannot load cognitive/wandering fusion policy: {config_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CognitiveWanderingFusionError("fusion policy root must be a mapping")
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != EXPECTED_FUSION_POLICY_SHA256:
        raise CognitiveWanderingFusionError("fusion policy SHA-256 drifted")
    _exact(payload, {"schema_version", "policy_version", "deployment_status", "time", "wandering_quality", "wandering_support", "fusion"}, "policy")
    if payload["schema_version"] != "cognitive-wandering-rule-fusion-config-v1":
        raise CognitiveWanderingFusionError("fusion policy schema version drifted")
    if payload["policy_version"] != FUSION_POLICY_VERSION:
        raise CognitiveWanderingFusionError("fusion policy version drifted")
    if payload["deployment_status"] != FUSION_DEPLOYMENT_STATUS:
        raise CognitiveWanderingFusionError("fusion deployment status drifted")
    time = _mapping(payload["time"], "time")
    quality = _mapping(payload["wandering_quality"], "wandering_quality")
    support = _mapping(payload["wandering_support"], "wandering_support")
    fusion = _mapping(payload["fusion"], "fusion")
    _exact(time, {"maximum_cognitive_age_days", "wandering_window_days"}, "time")
    _exact(quality, {"minimum_tracking_coverage", "minimum_baseline_ready_days", "baseline_ready_states", "usable_statuses"}, "wandering_quality")
    _exact(support, {"elevation_metrics", "minimum_elevated_days_for_support", "minimum_elevated_days_for_strong_support", "strong_support_requires_stable_baseline", "require_high_confidence_wandering_episode", "purpose_context_is_explanatory_only"}, "wandering_support")
    _exact(fusion, {"support_maximum_increment_levels", "support_only_maximum_level", "wandering_can_downrank_cognitive", "anchors"}, "fusion")
    anchors = _mapping(fusion["anchors"], "fusion.anchors")
    _exact(anchors, set(LEVELS), "fusion.anchors")
    policy = FusionPolicy(
        policy_version=str(payload["policy_version"]),
        deployment_status=str(payload["deployment_status"]),
        maximum_cognitive_age_days=_positive_int(time["maximum_cognitive_age_days"], "maximum_cognitive_age_days"),
        wandering_window_days=_positive_int(time["wandering_window_days"], "wandering_window_days"),
        minimum_tracking_coverage=_bounded_float(quality["minimum_tracking_coverage"], "minimum_tracking_coverage"),
        minimum_baseline_ready_days=_positive_int(quality["minimum_baseline_ready_days"], "minimum_baseline_ready_days"),
        baseline_ready_states=_string_tuple(quality["baseline_ready_states"], "baseline_ready_states"),
        usable_statuses=_string_tuple(quality["usable_statuses"], "usable_statuses"),
        elevation_metrics=_string_tuple(support["elevation_metrics"], "elevation_metrics"),
        minimum_elevated_days_for_support=_positive_int(support["minimum_elevated_days_for_support"], "minimum_elevated_days_for_support"),
        minimum_elevated_days_for_strong_support=_positive_int(support["minimum_elevated_days_for_strong_support"], "minimum_elevated_days_for_strong_support"),
        strong_support_requires_stable_baseline=_boolean(support["strong_support_requires_stable_baseline"], "strong_support_requires_stable_baseline"),
        require_high_confidence_wandering_episode=_boolean(support["require_high_confidence_wandering_episode"], "require_high_confidence_wandering_episode"),
        purpose_context_is_explanatory_only=_boolean(support["purpose_context_is_explanatory_only"], "purpose_context_is_explanatory_only"),
        support_maximum_increment_levels=_positive_int(fusion["support_maximum_increment_levels"], "support_maximum_increment_levels"),
        support_only_maximum_level=str(fusion["support_only_maximum_level"]),
        wandering_can_downrank_cognitive=_boolean(fusion["wandering_can_downrank_cognitive"], "wandering_can_downrank_cognitive"),
        anchors={name: _bounded_score(anchors[name], f"anchors.{name}") for name in LEVELS},
        config_sha256=observed_sha256,
    )
    _validate_policy_semantics(policy)
    return policy


def fuse_cognitive_wandering_attention(
    request: CognitiveWanderingFusionRequest,
    *,
    policy: FusionPolicy | None = None,
) -> CognitiveWanderingFusionResponse:
    active = policy or load_fusion_policy()
    if request.policy_version != active.policy_version:
        raise CognitiveWanderingFusionError("request fusion policy version drifted")

    warnings: list[str] = []
    decision_reasons: list[str] = []
    discordance: list[str] = []
    cognitive, cognitive_level, cognitive_quality = _summarize_cognitive(
        request, active, warnings
    )
    wandering, purpose_context_seen, non_home_seen = _summarize_wandering(
        request, active, warnings
    )
    if purpose_context_seen and active.purpose_context_is_explanatory_only:
        discordance.append("purpose_context_observed_not_used_for_downranking")
    if non_home_seen:
        warnings.append("wandering_evidence_not_home_validated")

    fused_level = _fuse_level(cognitive_level, wandering.support_level)
    if cognitive_level is not None:
        if wandering.support_level is not None and wandering.support_level >= 1:
            if fused_level > cognitive_level:
                decision_reasons.append("persistent_wandering_support_incremented_one_level")
            else:
                decision_reasons.append("cognitive_already_at_highest_attention_level")
        else:
            decision_reasons.append("cognitive_level_preserved_without_wandering_increment")
        if cognitive_level >= 1 and wandering.status == "ready" and wandering.support_level == 0:
            discordance.append("cognitive_clue_not_supported_by_wandering_no_downrank")
        if cognitive_level == 0 and (wandering.support_level or 0) >= 1:
            discordance.append("wandering_support_cognitive_normal")
    elif wandering.support_level is not None and wandering.support_level >= 1:
        decision_reasons.append("wandering_support_only_capped_at_attention")
    else:
        decision_reasons.append("insufficient_current_cognitive_and_persistent_wandering_evidence")

    if cognitive_level is not None and wandering.status == "ready":
        evidence_status = "cognitive_and_wandering"
    elif cognitive_level is not None:
        evidence_status = "cognitive_only"
    elif wandering.support_level is not None and wandering.support_level >= 1:
        evidence_status = "wandering_only"
    else:
        evidence_status = "insufficient"

    if fused_level is None:
        status = "insufficient_evidence"
    elif (
        evidence_status == "cognitive_and_wandering"
        and cognitive_quality == "complete"
    ):
        status = "completed"
    else:
        status = "degraded"
    level_name = None if fused_level is None else LEVELS[fused_level]
    score = None if level_name is None else float(active.anchors[level_name])
    recommendation = _recommendation(level_name)

    return CognitiveWanderingFusionResponse.model_validate(
        {
            "schema_version": "cognitive_wandering_fusion_response_v1",
            "request_id": request.request_id,
            "person_id": request.person_id,
            "evaluated_at": request.evaluated_at,
            "status": status,
            "evidence_status": evidence_status,
            "attention_level": level_name,
            "attention_score": score,
            "attention_score_semantics": "rule-derived-not-diagnostic-probability",
            "cognitive_component": cognitive,
            "wandering_component": wandering,
            "decision_reasons": _unique(decision_reasons),
            "discordance_flags": _unique(discordance),
            "warnings": _unique(warnings),
            "recommendation": recommendation,
            "diagnosis": False,
            "research_outputs": None,
            "cognitive_model_version": cognitive.model_version,
            "fusion_policy_version": active.policy_version,
            "fusion_policy_config_sha256": active.config_sha256,
            "rule_contract_sha256": rule_contract_sha256(active),
            "deployment_status": active.deployment_status,
        }
    )


def _summarize_cognitive(
    request: CognitiveWanderingFusionRequest,
    policy: FusionPolicy,
    warnings: list[str],
) -> tuple[CognitiveComponentSummary, int | None, str]:
    evidence = request.cognitive
    result = evidence.result
    if result is None:
        return (
            CognitiveComponentSummary(
                usable=False,
                source_status="unavailable",
                source_request_id=None,
                observed_at=None,
                age_days=None,
                score=None,
                level=None,
                confidence=None,
                adaptation_mode=None,
                model_version=None,
            ),
            None,
            "unavailable",
        )
    assert evidence.observed_at is not None
    age_seconds = (request.evaluated_at - evidence.observed_at).total_seconds()
    if age_seconds < 0:
        raise CognitiveWanderingFusionError("cognitive observed_at is in the future")
    age_days = age_seconds / 86400.0
    stale = age_days > float(policy.maximum_cognitive_age_days)
    available = result.status in {"completed", "degraded"} and not stale
    if stale:
        warnings.append("cognitive_evidence_stale")
    if result.status == "insufficient_input":
        warnings.append("cognitive_source_insufficient_input")
    level = LEVELS.index(result.cognitive_clue_level) if available else None
    quality = (
        "complete"
        if available
        and result.status == "completed"
        and result.adaptation_mode == "standard_picture_description"
        else ("degraded" if available else "unavailable")
    )
    if available and result.adaptation_mode == "voice_prompt_adaptation":
        warnings.append("cognitive_voice_prompt_adaptation")
    return (
        CognitiveComponentSummary(
            usable=available,
            source_status=result.status,
            source_request_id=result.request_id,
            observed_at=evidence.observed_at,
            age_days=round(age_days, 6),
            score=result.cognitive_clue_score,
            level=result.cognitive_clue_level,
            confidence=result.confidence,
            adaptation_mode=result.adaptation_mode,
            model_version=result.model_version,
        ),
        level,
        quality,
    )


def _summarize_wandering(
    request: CognitiveWanderingFusionRequest,
    policy: FusionPolicy,
    warnings: list[str],
) -> tuple[WanderingComponentSummary, bool, bool]:
    deviations = {row.local_date: row for row in request.wandering_baseline_deviations}
    timezone_name = (
        request.wandering_daily_reports[0].timezone
        if request.wandering_daily_reports
        else None
    )
    if timezone_name is None:
        evaluation_date = request.evaluated_at.date()
    else:
        try:
            evaluation_date = request.evaluated_at.astimezone(ZoneInfo(timezone_name)).date()
        except ZoneInfoNotFoundError as exc:
            raise CognitiveWanderingFusionError("wandering timezone is unknown") from exc

    ignored_dates: list[date] = []
    usable_dates: list[date] = []
    baseline_ready_dates: list[date] = []
    elevated_dates: list[date] = []
    stable_elevated = False
    purpose_context_seen = False
    non_home_seen = False
    model_ids: set[str] = set()
    for daily in sorted(request.wandering_daily_reports, key=lambda row: row.local_date):
        age_days = (evaluation_date - daily.local_date).days
        if age_days < 0:
            raise CognitiveWanderingFusionError("wandering daily report is in the future")
        if age_days >= policy.wandering_window_days:
            ignored_dates.append(daily.local_date)
            continue
        non_home_seen = non_home_seen or bool(set(daily.quality_flags) & NON_HOME_FLAGS)
        purpose_context_seen = purpose_context_seen or any(
            daily.context_counts.get(name, 0) > 0 for name in PURPOSE_CONTEXTS
        )
        if daily.identity.model_id:
            model_ids.add(daily.identity.model_id)
        tracking_ok = (
            daily.tracking_coverage is not None
            and daily.tracking_coverage >= policy.minimum_tracking_coverage
        )
        daily_usable = daily.status in policy.usable_statuses and tracking_ok
        if not daily_usable:
            ignored_dates.append(daily.local_date)
            continue
        usable_dates.append(daily.local_date)
        deviation = deviations.get(daily.local_date)
        deviation_ready = (
            deviation is not None
            and deviation.status == "ready"
            and deviation.readiness_status in policy.baseline_ready_states
        )
        if not deviation_ready:
            ignored_dates.append(daily.local_date)
            continue
        assert deviation is not None
        baseline_ready_dates.append(daily.local_date)
        high_confidence_ok = (
            daily.confidence_tiers.high_confidence.count > 0
            and daily.episode_counts.wandering_like > 0
        )
        metric_elevated = any(
            deviation.metrics[name].status == "ready"
            and deviation.metrics[name].delta_from_p90 is not None
            and float(deviation.metrics[name].delta_from_p90) > 0.0
            for name in policy.elevation_metrics
        )
        elevated = metric_elevated and (
            high_confidence_ok
            if policy.require_high_confidence_wandering_episode
            else True
        )
        if elevated:
            elevated_dates.append(daily.local_date)
            stable_elevated = stable_elevated or deviation.readiness_status == "stable_ready"

    if ignored_dates:
        warnings.append("wandering_days_ignored_by_freshness_or_quality_gate")
    if len(baseline_ready_dates) < policy.minimum_baseline_ready_days:
        status = "warming_up" if usable_dates else "unavailable"
        support_level: int | None = None
        warnings.append(
            "wandering_personal_baseline_not_ready"
            if status == "warming_up"
            else "wandering_evidence_unavailable"
        )
    else:
        status = "ready"
        elevated_count = len(elevated_dates)
        strong = elevated_count >= policy.minimum_elevated_days_for_strong_support
        if policy.strong_support_requires_stable_baseline:
            strong = strong and stable_elevated
        if strong:
            support_level = 2
        elif elevated_count >= policy.minimum_elevated_days_for_support:
            support_level = 1
        else:
            support_level = 0
    return (
        WanderingComponentSummary(
            status=status,
            support_level=support_level,
            usable_day_count=len(usable_dates),
            baseline_ready_day_count=len(baseline_ready_dates),
            elevated_day_count=len(elevated_dates),
            elevated_dates=elevated_dates,
            ignored_dates=sorted(set(ignored_dates)),
            latest_usable_date=max(usable_dates, default=None),
            source_model_ids=sorted(model_ids),
        ),
        purpose_context_seen,
        non_home_seen,
    )


def _fuse_level(cognitive_level: int | None, wandering_support: int | None) -> int | None:
    if cognitive_level is None:
        return 1 if wandering_support is not None and wandering_support >= 1 else None
    if wandering_support is not None and wandering_support >= 1:
        return min(2, cognitive_level + 1)
    return cognitive_level


def enumerate_rule_truth_table() -> list[dict[str, int | None]]:
    return [
        {
            "cognitive_level": cognitive,
            "wandering_support": support,
            "fused_level": _fuse_level(cognitive, support),
        }
        for cognitive in (None, 0, 1, 2)
        for support in (0, 1, 2)
    ]


def validate_rule_truth_table(policy: FusionPolicy | None = None) -> dict[str, Any]:
    active = policy or load_fusion_policy()
    rows = enumerate_rule_truth_table()
    violations: list[str] = []
    for cognitive in (None, 0, 1, 2):
        outputs = [_fuse_level(cognitive, support) for support in (0, 1, 2)]
        numeric = [-1 if value is None else value for value in outputs]
        if numeric != sorted(numeric):
            violations.append(f"wandering support downranked cognitive={cognitive}")
        for result in outputs:
            if cognitive is not None and result is not None:
                if result < cognitive:
                    violations.append(f"cognitive level was downranked: {cognitive}->{result}")
                if result - cognitive > active.support_maximum_increment_levels:
                    violations.append(f"support increment exceeded cap: {cognitive}->{result}")
            if cognitive is None and result is not None and result > 1:
                violations.append("wandering-only result exceeded attention")
    return {
        "status": "passed" if not violations else "failed",
        "row_count": len(rows),
        "violations": violations,
    }


def rule_contract_sha256(policy: FusionPolicy | None = None) -> str:
    active = policy or load_fusion_policy()
    payload = {
        "policy_version": active.policy_version,
        "config_sha256": active.config_sha256,
        "cognitive_levels": LEVELS,
        "wandering_support_levels": (0, 1, 2),
        "support_maximum_increment_levels": active.support_maximum_increment_levels,
        "support_only_maximum_level": active.support_only_maximum_level,
        "wandering_can_downrank_cognitive": active.wandering_can_downrank_cognitive,
        "context_role": "explanatory_only_never_downrank",
        "score_semantics": "rule-derived-not-diagnostic-probability",
        "truth_table": enumerate_rule_truth_table(),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _recommendation(level: str | None) -> str:
    return {
        None: "当前证据不足，请补充有效的 S10 三任务小测或足够的徘徊个人基线。",
        "normal": "保持常规观察，并按既定周期重复评估。",
        "attention": "建议近期复测认知任务，并结合日常行为变化持续关注。",
        "high_attention": "建议家属重点关注并安排专业评估；本结果不构成医学诊断。",
    }[level]


def _validate_policy_semantics(policy: FusionPolicy) -> None:
    if policy.support_maximum_increment_levels != 1:
        raise CognitiveWanderingFusionError("support increment must remain exactly one level")
    if policy.support_only_maximum_level != "attention":
        raise CognitiveWanderingFusionError("wandering-only cap must remain attention")
    if policy.wandering_can_downrank_cognitive:
        raise CognitiveWanderingFusionError("wandering must never downrank cognitive evidence")
    if policy.minimum_elevated_days_for_strong_support < policy.minimum_elevated_days_for_support:
        raise CognitiveWanderingFusionError("strong support cannot require fewer days")
    values = [policy.anchors[name] for name in LEVELS]
    if values != sorted(values) or len(set(values)) != len(values):
        raise CognitiveWanderingFusionError("fusion attention anchors must be strictly increasing")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CognitiveWanderingFusionError(f"{name} must be a mapping")
    return value


def _exact(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise CognitiveWanderingFusionError(f"{name} fields drifted")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CognitiveWanderingFusionError(f"{name} must be a positive integer")
    return value


def _bounded_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CognitiveWanderingFusionError(f"{name} must be numeric")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise CognitiveWanderingFusionError(f"{name} must be within [0, 1]")
    return parsed


def _bounded_score(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CognitiveWanderingFusionError(f"{name} must be numeric")
    parsed = float(value)
    if not 0.0 <= parsed <= 100.0:
        raise CognitiveWanderingFusionError(f"{name} must be within [0, 100]")
    return parsed


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise CognitiveWanderingFusionError(f"{name} must be boolean")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise CognitiveWanderingFusionError(f"{name} must be a non-empty list")
    parsed = tuple(str(item) for item in value)
    if any(not item for item in parsed) or len(parsed) != len(set(parsed)):
        raise CognitiveWanderingFusionError(f"{name} values must be unique non-empty strings")
    return parsed


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


__all__ = [
    "DEFAULT_FUSION_POLICY_PATH",
    "EXPECTED_FUSION_POLICY_SHA256",
    "CognitiveWanderingFusionError",
    "FusionPolicy",
    "enumerate_rule_truth_table",
    "fuse_cognitive_wandering_attention",
    "load_fusion_policy",
    "rule_contract_sha256",
    "validate_rule_truth_table",
]
