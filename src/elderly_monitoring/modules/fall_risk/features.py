"""跌倒风险融合层的特征契约。

本文件不直接从视频中提取特征，而是定义上游模块需要产出的 0-1
归一化风险特征字段，供最终规则评分卡融合使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


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


def clamp_score(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def weighted_fall_risk_score(features: Mapping[str, float | int]) -> float:
    """把多个归一化风险特征融合成一个 baseline 风险分。"""
    score = 0.0
    for spec in FALL_RISK_FEATURE_SPECS:
        score += clamp_score(features.get(spec.name)) * spec.weight
    return round(score, 4)


def feature_coverage(features: Mapping[str, object]) -> float:
    """估计核心融合输入的覆盖率，用于置信度计算。"""
    available = sum(1 for name in FALL_RISK_FEATURES if features.get(name) is not None)
    return round(available / len(FALL_RISK_FEATURES), 4)
