"""Algorithm-side V3.3.3 mood-social inference orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.config import (
    MoodSocialConfig,
    load_mood_social_config,
)
from elderly_monitoring.modules.mental_health.mood_social.feature_mapper import (
    map_mood_social_features,
)
from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    DomainFeatureVector,
    MappedMoodSocialFeatures,
)
from elderly_monitoring.modules.mental_health.mood_social.model_package import (
    ModelPackageError,
    MoodSocialModelPackage,
    load_mood_social_model_package,
)
from elderly_monitoring.modules.mental_health.mood_social.package_selection import (
    PackageSelectionError,
    load_active_package_selection,
)
from elderly_monitoring.modules.mental_health.mood_social.schemas import (
    MOOD_SOCIAL_MODEL_VERSION,
    MOOD_SOCIAL_PRODUCTION_MODEL_VERSION,
    MoodSocialInferRequest,
    MoodSocialInferResponse,
)


PACKAGE_ENVIRONMENT_VARIABLE = "MOOD_SOCIAL_MODEL_PACKAGE_PATH"
DEFAULT_PACKAGE_DIRECTORY = (
    Path(__file__).resolve().parents[5]
    / "models/mental_health/mood_social/v3.3.3/packages/MH-20260802-013"
)
EXPERT_MODEL_NAMES = {
    "activity": "activity_expert",
    "sleep": "sleep_expert",
    "joint": "activity_sleep_joint_expert",
    "physiology": "physiology_expert",
    "social_context": "social_context_expert",
}
COVERED_DOMAIN_ORDER = (
    "activity",
    "sleep",
    "physiology",
    "activity_sleep_joint",
    "social_context",
    "personal_change_activity",
    "personal_change_sleep",
    "personal_change_social",
)
SOURCE_BY_DOMAIN = {
    "activity": "camera",
    "sleep": "sleep_device",
    "physiology": "sleep_device",
    "activity_sleep_joint": ("camera", "sleep_device"),
    "social_context": "profile",
    "personal_change_activity": "camera",
    "personal_change_sleep": "sleep_device",
    "personal_change_social": "s10",
}
USED_SOURCE_ORDER = ("camera", "sleep_device", "s10", "profile")


class MoodSocialPipelineError(RuntimeError):
    """Raised when valid inference input cannot be processed."""


class MoodSocialPipelineUnavailableError(MoodSocialPipelineError):
    """Raised when the versioned model package is unavailable or invalid."""


@dataclass(frozen=True)
class MoodSocialPipeline:
    package: MoodSocialModelPackage
    config: MoodSocialConfig

    @classmethod
    def from_package(
        cls,
        package_directory: str | Path | None = None,
        *,
        config: MoodSocialConfig | None = None,
    ) -> "MoodSocialPipeline":
        configured_path = package_directory or os.environ.get(PACKAGE_ENVIRONMENT_VARIABLE)
        if configured_path is None:
            try:
                configured_path = load_active_package_selection().package_directory
            except PackageSelectionError as exc:
                raise MoodSocialPipelineUnavailableError(
                    "mood-social package selection is unavailable"
                ) from exc
        try:
            package = load_mood_social_model_package(configured_path)
        except (ModelPackageError, OSError, ValueError) as exc:
            raise MoodSocialPipelineUnavailableError(
                "mood-social model package is unavailable"
            ) from exc
        if package.offline_models:
            raise MoodSocialPipelineUnavailableError(
                "offline auxiliary models cannot enter online inference"
            )
        return cls(package=package, config=config or load_mood_social_config())

    def predict(
        self,
        request: MoodSocialInferRequest | Mapping[str, Any],
    ) -> MoodSocialInferResponse:
        parsed = (
            request
            if isinstance(request, MoodSocialInferRequest)
            else MoodSocialInferRequest.model_validate(request)
        )
        mapped = map_mood_social_features(parsed, config=self.config)
        expert_frame = _build_expert_frame(mapped)
        fusion_frame, expert_scores, trend_scores = self._build_fusion_frame(
            parsed,
            expert_frame,
        )
        fused = self.package.fusion.predict_frame(fusion_frame).iloc[0]
        if not bool(fused["available"]):
            return _unavailable_response(parsed, _limitations(parsed, mapped, {}, {}))

        probability = float(fused["fusion_probability"])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise MoodSocialPipelineError("fusion produced an invalid probability")
        covered_domains = _covered_domains(expert_scores, trend_scores)
        if not covered_domains:
            raise MoodSocialPipelineError(
                "fusion availability disagrees with effective branch evidence"
            )
        used_sources = _used_sources(covered_domains)
        confidence = _response_confidence(expert_scores, trend_scores)
        response = {
            "schema_version": "mood_social_infer_response_v3",
            "request_id": parsed.request_id,
            "person_id": parsed.person_id,
            "target_date": parsed.target_date,
            "module": "mood_social_attention",
            "model_version": MOOD_SOCIAL_MODEL_VERSION,
            "available": True,
            "evidence_scope": (
                "engineering_proxy" if "s10" in used_sources else "proxy_label_supported"
            ),
            "attention_index": probability,
            "attention_score": _score_half_up(probability),
            "attention_level": _attention_level(probability, self.config),
            "confidence": confidence,
            "used_sources": used_sources,
            "covered_domains": covered_domains,
            "domain_scores": _domain_scores(expert_scores, trend_scores),
            "model_contributions": [],
            "trend": _attention_trend(parsed, probability, self.config),
            "summary": (
                "当前模型结果可用，暂未识别出明确的主要关注因素，"
                "建议继续观察近期趋势。"
            ),
            "limitations": _limitations(
                parsed,
                mapped,
                expert_scores,
                trend_scores,
            ),
            "diagnosis": False,
        }
        return MoodSocialInferResponse.model_validate(response)

    def _build_fusion_frame(
        self,
        request: MoodSocialInferRequest,
        expert_frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        runtime = self.package.manifest.get("runtime_confidence")
        if not isinstance(runtime, Mapping):
            raise MoodSocialPipelineUnavailableError(
                "model package runtime confidence metadata is missing"
            )
        reliability = runtime.get("expert_reliability")
        if not isinstance(reliability, Mapping) or set(reliability) != set(
            EXPERT_MODEL_NAMES
        ):
            raise MoodSocialPipelineUnavailableError(
                "model package runtime confidence metadata is invalid"
            )

        columns: dict[str, list[float | int]] = {}
        expert_scores: dict[str, dict[str, float]] = {}
        for branch, model_name in EXPERT_MODEL_NAMES.items():
            model = self.package.online_models[model_name]
            predicted = model.predict(expert_frame).iloc[0]
            mask = int(predicted["expert_mask"])
            coverage = float(predicted["observed_feature_fraction"])
            branch_reliability = float(reliability[branch])
            confidence = (
                float(np.clip(branch_reliability * coverage, 0.0, 1.0))
                if mask
                else 0.0
            )
            probability = (
                float(predicted["calibrated_probability"]) if mask else math.nan
            )
            expert_scores[branch] = {
                "probability": probability,
                "mask": float(mask),
                "confidence": confidence,
            }
            columns[f"expert_{branch}_current_probability"] = [probability]
            columns[f"expert_{branch}_expert_mask"] = [mask]
            columns[f"expert_{branch}_confidence"] = [confidence]

        trend_model = self.package.online_models["personal_trend"]
        trend_result = trend_model.predict_request(request)
        trend_scores: dict[str, dict[str, float]] = {}
        for branch in ("activity", "sleep", "social"):
            result = trend_result[branch]
            mask = int(result.personal_change_mask)
            reliability_value = float(result.reliability) if mask else 0.0
            probability = float(result.probability) if mask else math.nan
            trend_scores[branch] = {
                "probability": probability,
                "mask": float(mask),
                "reliability": reliability_value,
            }
            columns[f"trend_{branch}_personal_change_evidence"] = [probability]
            columns[f"trend_{branch}_personal_change_mask"] = [mask]
            columns[f"trend_{branch}_reliability"] = [reliability_value]
        return pd.DataFrame(columns), expert_scores, trend_scores


def _build_expert_frame(mapped: MappedMoodSocialFeatures) -> pd.DataFrame:
    vectors = {
        "activity": mapped.activity,
        "sleep": mapped.sleep,
        "physiology": mapped.physiology,
        "social_context": mapped.social_context,
    }
    row: dict[str, Any] = {}
    for vector in vectors.values():
        _append_vector(row, vector)
    return pd.DataFrame([row])


def _append_vector(row: dict[str, Any], vector: DomainFeatureVector) -> None:
    for name, value, mask in zip(
        vector.feature_names,
        vector.values,
        vector.feature_mask,
        strict=True,
    ):
        feature = f"{vector.domain}.{name}"
        row[feature] = value if mask else None
        row[f"feature_mask.{feature}"] = mask


def _covered_domains(
    expert_scores: Mapping[str, Mapping[str, float]],
    trend_scores: Mapping[str, Mapping[str, float]],
) -> list[str]:
    domains: set[str] = set()
    expert_domain = {
        "activity": "activity",
        "sleep": "sleep",
        "joint": "activity_sleep_joint",
        "physiology": "physiology",
        "social_context": "social_context",
    }
    for branch, score in expert_scores.items():
        if score["mask"] > 0 and score["confidence"] > 0:
            domains.add(expert_domain[branch])
    for branch, score in trend_scores.items():
        if score["mask"] > 0 and score["reliability"] > 0:
            domains.add(f"personal_change_{branch}")
    return [domain for domain in COVERED_DOMAIN_ORDER if domain in domains]


def _used_sources(covered_domains: list[str]) -> list[str]:
    sources: set[str] = set()
    for domain in covered_domains:
        value = SOURCE_BY_DOMAIN[domain]
        if isinstance(value, tuple):
            sources.update(value)
        else:
            sources.add(value)
    return [source for source in USED_SOURCE_ORDER if source in sources]


def _domain_scores(
    expert_scores: Mapping[str, Mapping[str, float]],
    trend_scores: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    def expert(branch: str) -> float | None:
        score = expert_scores[branch]
        return (
            float(score["probability"])
            if score["mask"] > 0 and score["confidence"] > 0
            else None
        )

    def trend(branch: str) -> float | None:
        score = trend_scores[branch]
        return (
            float(score["probability"])
            if score["mask"] > 0 and score["reliability"] > 0
            else None
        )

    return {
        "activity": expert("activity"),
        "sleep": expert("sleep"),
        "physiology": expert("physiology"),
        "activity_sleep_joint": expert("joint"),
        "social_context": expert("social_context"),
        "personal_change": {
            "activity": trend("activity"),
            "sleep": trend("sleep"),
            "social": trend("social"),
        },
    }


def _response_confidence(
    expert_scores: Mapping[str, Mapping[str, float]],
    trend_scores: Mapping[str, Mapping[str, float]],
) -> float:
    effective = [
        score["confidence"]
        for score in expert_scores.values()
        if score["mask"] > 0 and score["confidence"] > 0
    ]
    effective.extend(
        score["reliability"]
        for score in trend_scores.values()
        if score["mask"] > 0 and score["reliability"] > 0
    )
    if not effective:
        return 0.0
    return float(np.clip(np.mean(effective), 0.0, 1.0))


def _score_half_up(probability: float) -> int:
    return int(
        (Decimal(str(probability)) * Decimal("100")).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )


def _attention_level(probability: float, config: MoodSocialConfig) -> int:
    first, second, third = config.attention.thresholds
    if probability < first:
        return 0
    if probability < second:
        return 1
    if probability < third:
        return 2
    return 3


def _attention_trend(
    request: MoodSocialInferRequest,
    probability: float,
    config: MoodSocialConfig,
) -> str:
    values = [
        (item.date, float(item.attention_index))
        for item in request.history_attention_indices
        if item.model_version
        in {MOOD_SOCIAL_MODEL_VERSION, MOOD_SOCIAL_PRODUCTION_MODEL_VERSION}
    ]
    values.append((request.target_date, probability))
    values = values[-(config.trend.max_history_results + 1) :]
    if len(values) < config.trend.minimum_points:
        return "unknown"
    origin = values[0][0]
    x = np.asarray([(day - origin).days for day, _ in values], dtype=float)
    y = np.asarray([value for _, value in values], dtype=float)
    if len(set(x.tolist())) < 2:
        return "unknown"
    slope = float(np.polyfit(x, y, 1)[0])
    if slope >= config.trend.rising_threshold_per_day:
        return "rising"
    if slope <= config.trend.falling_threshold_per_day:
        return "falling"
    return "stable"


def _limitations(
    request: MoodSocialInferRequest,
    mapped: MappedMoodSocialFeatures,
    expert_scores: Mapping[str, Mapping[str, float]],
    trend_scores: Mapping[str, Mapping[str, float]],
) -> list[str]:
    covered = set(_covered_domains(expert_scores, trend_scores))
    values: list[str] = []
    if covered & {"activity", "activity_sleep_joint", "personal_change_activity"}:
        values.extend(
            (
                "camera_activity_uses_proxy_transfer",
                "camera_activity_limited_to_valid_observation_periods",
            )
        )
    if covered & {
        "sleep",
        "physiology",
        "activity_sleep_joint",
        "personal_change_sleep",
    }:
        values.append("sleep_uses_proxy_transfer")
    if covered & {"personal_change_activity", "personal_change_sleep"}:
        values.append("timescale_proxy_transfer")
    if "personal_change_social" in covered:
        values.append("s10_social_uses_engineering_proxy")
    if any(mask == 0 for mask in mapped.social_context.feature_mask):
        values.append("profile_fields_incomplete")
    if not any(
        score["mask"] > 0 and score["reliability"] > 0
        for score in trend_scores.values()
    ):
        values.append("personal_history_insufficient")

    current_mask = mapped.daily_features[-1].day_mask
    declared = set(request.available_sources)
    if "camera" in declared and current_mask.activity == 0:
        values.append("current_activity_missing")
    if "sleep_device" in declared and current_mask.sleep == 0:
        values.append("current_sleep_missing")
    if "sleep_device" in declared and current_mask.physiology == 0:
        values.append("current_physiology_missing")
    if "s10" in declared and current_mask.social == 0:
        values.append("current_social_missing")
    if not covered:
        values.append("insufficient_data")
    return values


def _unavailable_response(
    request: MoodSocialInferRequest,
    limitations: list[str],
) -> MoodSocialInferResponse:
    return MoodSocialInferResponse.model_validate(
        {
            "schema_version": "mood_social_infer_response_v3",
            "request_id": request.request_id,
            "person_id": request.person_id,
            "target_date": request.target_date,
            "module": "mood_social_attention",
            "model_version": MOOD_SOCIAL_MODEL_VERSION,
            "available": False,
            "evidence_scope": "insufficient_data",
            "attention_index": None,
            "attention_score": None,
            "attention_level": None,
            "confidence": 0.0,
            "used_sources": [],
            "covered_domains": [],
            "domain_scores": None,
            "model_contributions": [],
            "trend": "unknown",
            "summary": "当前未获得可用的情绪与社交关注证据。",
            "limitations": limitations,
            "diagnosis": False,
        }
    )


__all__ = [
    "DEFAULT_PACKAGE_DIRECTORY",
    "PACKAGE_ENVIRONMENT_VARIABLE",
    "MoodSocialPipeline",
    "MoodSocialPipelineError",
    "MoodSocialPipelineUnavailableError",
]
