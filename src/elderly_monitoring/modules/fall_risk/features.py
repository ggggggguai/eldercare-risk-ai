"""跌倒风险融合层的特征契约。

本文件不直接从视频中提取特征，而是定义上游模块需要产出的 0-1
归一化风险特征字段，供最终规则评分卡融合使用。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class FallRiskFeatureSpec:
    name: str
    weight: float
    description: str


# 第一版可解释权重。这里是工程 baseline，不是经过临床标定的系数。
FALL_RISK_FEATURE_SPECS = (
    FallRiskFeatureSpec("gait_risk_score", 0.22, "步态不稳、步速/步幅异常等风险特征"),
    FallRiskFeatureSpec("sit_stand_risk_score", 0.18, "坐站转换困难、起身失败或耗时增加"),
    FallRiskFeatureSpec("near_fall_event_score", 0.28, "近跌倒、踉跄恢复或快速扶物等前置事件"),
    FallRiskFeatureSpec("baseline_deviation_score", 0.16, "相对个体行为基线的异常偏离"),
    FallRiskFeatureSpec("scene_risk_score", 0.08, "夜间、床边、浴室等场景环境风险"),
    FallRiskFeatureSpec("activity_rhythm_score", 0.08, "活动节律下降或昼夜活动模式改变"),
)

FALL_RISK_FEATURES = tuple(spec.name for spec in FALL_RISK_FEATURE_SPECS)
EVENT_RISK_FEATURES = (
    "fall_event_score",
    "long_static_score",
)

ALL_FALL_RISK_INPUT_FEATURES = (
    "gait_risk_score",
    "sit_stand_risk_score",
    "near_fall_event_score",
    "baseline_deviation_score",
    "scene_risk_score",
    "activity_rhythm_score",
    "fall_event_score",
    "long_static_score",
)

ENVIRONMENT_FEATURES = (
    "low_light_score",
    "water_exposure_score",
    "behavior_anchor",
    "light_interaction_score",
    "water_interaction_score",
    "environment_interaction_score",
)


def environment_assist_contribution(
    features: Mapping[str, Any],
    *,
    mode: str,
    weight: float | None,
    min_behavior_anchor: float | None,
) -> tuple[float, dict[str, Any]]:
    """Return the bounded assist contribution without touching base features."""
    diagnostics = {
        "mode": mode,
        "contribution": 0.0,
        "status": "disabled" if mode != "assist" else "unavailable",
        "reason": None,
    }
    if mode != "assist":
        return 0.0, diagnostics
    mask = features.get("environment_mask")
    environment_score = features.get("environment_interaction_score")
    anchor = features.get("behavior_anchor")
    if not isinstance(mask, Mapping) or mask.get("environment_interaction_score") is not True:
        diagnostics["reason"] = "environment_interaction_unavailable"
        return 0.0, diagnostics
    if anchor is None or float(anchor) < float(min_behavior_anchor or 1.0):
        diagnostics["reason"] = "behavior_anchor_below_threshold"
        return 0.0, diagnostics
    if environment_score is None or weight is None:
        diagnostics["reason"] = "assist_policy_unconfigured"
        return 0.0, diagnostics
    contribution = clamp_score(float(weight) * clamp_score(environment_score))
    diagnostics.update({
        "status": "valid",
        "contribution": round(contribution, 4),
        "environment_interaction_score": clamp_score(environment_score),
        "behavior_anchor": clamp_score(anchor),
    })
    return round(contribution, 4), diagnostics


def clamp_score(value: float | int | None) -> float:
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def weighted_fall_risk_score(features: Mapping[str, Any]) -> float:
    """Fuse available normalized inputs without treating missing branches as zero risk."""
    mask = features.get("fusion_mask")
    explicit_mask = mask if isinstance(mask, Mapping) else None
    score = 0.0
    available_weight = 0.0
    for spec in FALL_RISK_FEATURE_SPECS:
        if explicit_mask is not None and explicit_mask.get(spec.name) is not True:
            continue
        effective_weight = spec.weight
        if spec.name == "baseline_deviation_score":
            configured_weight = features.get("baseline_fusion_weight")
            if configured_weight is not None:
                effective_weight *= clamp_score(configured_weight)
        score += clamp_score(features.get(spec.name)) * effective_weight
        available_weight += effective_weight
    if available_weight <= 0:
        return 0.0
    if explicit_mask is not None:
        score /= available_weight
    return round(score, 4)


def feature_coverage(features: Mapping[str, object]) -> float:
    """估计核心融合输入的覆盖率，用于置信度计算。"""
    mask = features.get("fusion_mask")
    if isinstance(mask, Mapping):
        available = sum(mask.get(name) is True for name in FALL_RISK_FEATURES)
        return round(available / len(FALL_RISK_FEATURES), 4)
    available = sum(1 for name in FALL_RISK_FEATURES if features.get(name) is not None)
    return round(available / len(FALL_RISK_FEATURES), 4)
