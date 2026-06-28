from __future__ import annotations

from typing import Any, Mapping

from elderly_monitoring.common.schemas import AlgorithmEvent, EvidenceWindow, action_for_level
from elderly_monitoring.modules.fall_risk.features import (
    feature_coverage,
    clamp_score,
    weighted_fall_risk_score,
)


class FallRiskPipeline:
    """轻量跌倒风险预警融合管线，不是医疗诊断模型。

    这个类只做最终融合，期望上游已经提供步态风险、近跌倒风险等
    归一化特征分；它本身不做人检、姿态估计或步态窗口提取。
    """

    model_version = "fall-risk-v0.1"

    def predict_from_features(self, sample: Mapping[str, Any]) -> AlgorithmEvent:
        risk_score = weighted_fall_risk_score(sample)
        near_fall_score = clamp_score(sample.get("near_fall_event_score"))
        fall_event_score = clamp_score(sample.get("fall_event_score"))
        long_static_score = clamp_score(sample.get("long_static_score"))

        # 跌倒和长时间静止属于紧急强触发信号：即使其他长窗口特征缺失，
        # 也应直接升到 4 级，方便下游响应逻辑立即处理。
        if max(fall_event_score, long_static_score) >= 0.8:
            risk_level = 4
            trigger_event = "fall_or_long_static"
        elif near_fall_score >= 0.7 or risk_score >= 0.65:
            risk_level = 3
            trigger_event = "near_fall" if near_fall_score >= 0.7 else "combined_high_risk"
        elif risk_score >= 0.45:
            risk_level = 2
            trigger_event = "mobility_risk"
        elif risk_score >= 0.25:
            risk_level = 1
            trigger_event = "mild_deviation"
        else:
            risk_level = 0
            trigger_event = "normal"

        factors = self._risk_factors(sample, risk_level)
        confidence = self._confidence(sample, risk_score)

        return AlgorithmEvent(
            module="fall_risk",
            device_id=sample.get("device_id"),
            person_id=str(sample.get("person_id", "unknown")),
            timestamp=str(sample.get("timestamp", "")),
            scene_region=sample.get("scene_region"),
            risk_level=risk_level,
            risk_score=risk_score,
            confidence=confidence,
            trigger_event=trigger_event,
            risk_factors=factors,
            recommended_action=action_for_level(risk_level),
            evidence_window=EvidenceWindow(
                start_time=sample.get("start_time"),
                end_time=sample.get("end_time"),
            ),
            model_version=self.model_version,
        )

    def _risk_factors(self, sample: Mapping[str, Any], risk_level: int) -> list[str]:
        # 解释因子保持机器可读。展示层可以把这些 code 映射成中文文案，
        # 不需要改动算法事件 schema。
        factors: list[str] = []
        if clamp_score(sample.get("fall_event_score")) >= 0.8:
            factors.append("suspected_fall_event")
        if clamp_score(sample.get("long_static_score")) >= 0.8:
            factors.append("long_static_after_fall_risk")
        if clamp_score(sample.get("near_fall_event_score")) >= 0.7:
            factors.append("near_fall_event")
        if clamp_score(sample.get("gait_risk_score")) >= 0.5:
            factors.append("gait_instability")
        if clamp_score(sample.get("sit_stand_risk_score")) >= 0.5:
            factors.append("sit_stand_difficulty")
        if clamp_score(sample.get("baseline_deviation_score")) >= 0.5:
            factors.append("personal_baseline_deviation")
        if clamp_score(sample.get("activity_rhythm_score")) >= 0.5:
            factors.append("activity_rhythm_change")
        if clamp_score(sample.get("scene_risk_score")) >= 0.5:
            factors.append("high_risk_scene")
        if not factors and risk_level == 0:
            factors.append("no_obvious_risk")
        return factors

    def _confidence(self, sample: Mapping[str, Any], risk_score: float) -> float:
        keypoint_quality = clamp_score(sample.get("keypoint_quality", 0.8))
        coverage = clamp_score(sample.get("feature_coverage", feature_coverage(sample)))
        # 置信度主要反映输入可靠性，不代表医学确定性。这里综合姿态质量、
        # 特征覆盖率和融合信号强度，避免稀疏输入看起来过于“确定”。
        confidence = 0.45 * keypoint_quality + 0.35 * coverage + 0.20 * min(1.0, risk_score + 0.2)
        return round(confidence, 4)
