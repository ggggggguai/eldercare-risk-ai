from __future__ import annotations

import ast
import copy
import inspect
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient
from pydantic import ValidationError

from elderly_monitoring.modules.mental_health.mood_social import (
    DEFAULT_PACKAGE_DIRECTORY,
    CameraGaitMetric,
    MoodSocialActivity,
    MoodSocialDailyFeatures,
    MoodSocialDomainScores,
    MoodSocialErrorDetail,
    MoodSocialErrorItem,
    MoodSocialErrorResponse,
    MoodSocialHistoryAttentionIndex,
    MoodSocialInferRequest,
    MoodSocialInferResponse,
    MoodSocialModelContribution,
    MoodSocialPersonalChangeScores,
    MoodSocialPhysiology,
    MoodSocialProfile,
    MoodSocialSleep,
    MoodSocialSocial,
    load_mood_social_config,
)
from elderly_monitoring.service.app import create_app
from elderly_monitoring.service.settings import ServiceSettings


def _target_date() -> str:
    current = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    return (current - timedelta(days=1)).isoformat()


def _date_before(value: str, days: int) -> str:
    return (
        datetime.fromisoformat(value).date() - timedelta(days=days)
    ).isoformat()


def _minimal_request_payload() -> dict[str, object]:
    target = _target_date()
    return {
        "schema_version": "mood_social_infer_request_v3",
        "request_id": "req_mh_002",
        "person_id": "elder-001",
        "target_date": target,
        "timezone": "Asia/Shanghai",
        "available_sources": ["camera"],
        "profile": {},
        "current_daily_features": {
            "date": target,
            "activity": None,
            "sleep": None,
            "physiology": None,
            "social": None,
        },
        "history_daily_features": [],
        "history_attention_indices": [],
    }


def _observed_activity() -> dict[str, object]:
    coverage = [0.0] * 24
    intensity: list[float | None] = [None] * 24
    coverage[6] = 60.0
    intensity[6] = 0.5
    return {
        "daytime_active_minutes": 10.0,
        "weighted_daytime_activity": 30.0,
        "valid_daytime_detection_minutes": 60.0,
        "low_activity_minutes": 40.0,
        "sedentary_bout_total_minutes": 30.0,
        "longest_sedentary_bout_minutes": 30.0,
        "activity_peak_minute_of_day": 390,
        "hourly_activity_intensity": intensity,
        "hourly_valid_detection_minutes": coverage,
        "camera_gait_metrics": [],
    }


def _zero_coverage_activity() -> dict[str, object]:
    return {
        "daytime_active_minutes": None,
        "weighted_daytime_activity": None,
        "valid_daytime_detection_minutes": 0.0,
        "low_activity_minutes": None,
        "sedentary_bout_total_minutes": None,
        "longest_sedentary_bout_minutes": None,
        "activity_peak_minute_of_day": None,
        "hourly_activity_intensity": [None] * 24,
        "hourly_valid_detection_minutes": [0.0] * 24,
        "camera_gait_metrics": [],
    }


def _observed_social() -> dict[str, object]:
    return {
        "call_log_observed": True,
        "incoming_call_opportunities": 0,
        "answered_call_count": 0,
        "missed_call_count": 0,
        "outgoing_call_count": 0,
        "connected_duration_minutes": 0.0,
        "active_contact_count": 0,
    }


def _unavailable_response_payload() -> dict[str, object]:
    return {
        "schema_version": "mood_social_infer_response_v3",
        "request_id": "req_unavailable",
        "person_id": "elder-001",
        "target_date": _target_date(),
        "module": "mood_social_attention",
        "model_version": "mood-fusion-v3.3.3",
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
        "limitations": ["insufficient_data"],
        "diagnosis": False,
    }


def _available_response_payload(index: float = 0.455) -> dict[str, object]:
    if index < 0.25:
        level = 0
    elif index < 0.45:
        level = 1
    elif index < 0.65:
        level = 2
    else:
        level = 3
    return {
        "schema_version": "mood_social_infer_response_v3",
        "request_id": "req_available",
        "person_id": "elder-001",
        "target_date": _target_date(),
        "module": "mood_social_attention",
        "model_version": "mood-fusion-v3.3.3",
        "available": True,
        "evidence_scope": "proxy_label_supported",
        "attention_index": index,
        "attention_score": int(
            (Decimal(str(index)) * Decimal("100")).quantize(
                Decimal("1"),
                rounding=ROUND_HALF_UP,
            )
        ),
        "attention_level": level,
        "confidence": 0.8,
        "used_sources": ["camera"],
        "covered_domains": ["activity"],
        "domain_scores": {
            "activity": 0.6,
            "sleep": None,
            "physiology": None,
            "activity_sleep_joint": None,
            "social_context": None,
            "personal_change": {
                "activity": None,
                "sleep": None,
                "social": None,
            },
        },
        "model_contributions": [],
        "trend": "unknown",
        "summary": (
            "当前模型结果可用，暂未识别出明确的主要关注因素，"
            "建议继续观察近期趋势。"
        ),
        "limitations": [],
        "diagnosis": False,
    }


class MoodSocialRequestSchemaTest(unittest.TestCase):
    def test_minimal_v3_request_and_empty_profile_are_valid(self) -> None:
        request = MoodSocialInferRequest.model_validate(
            _minimal_request_payload()
        )

        self.assertEqual(
            request.schema_version,
            "mood_social_infer_request_v3",
        )
        self.assertEqual(request.profile.model_dump(exclude_none=True), {})
        self.assertFalse(request.current_daily_features.has_domain_record)

    def test_every_http_model_forbids_extra_fields(self) -> None:
        model_types = (
            MoodSocialProfile,
            CameraGaitMetric,
            MoodSocialActivity,
            MoodSocialSleep,
            MoodSocialPhysiology,
            MoodSocialSocial,
            MoodSocialDailyFeatures,
            MoodSocialHistoryAttentionIndex,
            MoodSocialInferRequest,
            MoodSocialPersonalChangeScores,
            MoodSocialDomainScores,
            MoodSocialModelContribution,
            MoodSocialInferResponse,
            MoodSocialErrorItem,
            MoodSocialErrorDetail,
            MoodSocialErrorResponse,
        )

        for model_type in model_types:
            with self.subTest(model=model_type.__name__):
                self.assertEqual(model_type.model_config["extra"], "forbid")

        payload = _minimal_request_payload()
        payload["profile"] = {"unknown_field": 1}
        with self.assertRaises(ValidationError):
            MoodSocialInferRequest.model_validate(payload)

    def test_rejects_old_schema_string_numbers_bools_and_non_finite_numbers(
        self,
    ) -> None:
        for version in (
            "mood_social_infer_request_v1",
            "mood_social_infer_request_v2",
        ):
            payload = _minimal_request_payload()
            payload["schema_version"] = version
            with self.subTest(version=version), self.assertRaises(
                ValidationError
            ):
                MoodSocialInferRequest.model_validate(payload)

        valid = _zero_coverage_activity()
        for field, value in (
            ("valid_daytime_detection_minutes", "0"),
            ("valid_daytime_detection_minutes", True),
            ("valid_daytime_detection_minutes", math.nan),
            ("valid_daytime_detection_minutes", math.inf),
        ):
            payload = copy.deepcopy(valid)
            payload[field] = value
            with self.subTest(value=value), self.assertRaises(ValidationError):
                MoodSocialActivity.model_validate(payload)

    def test_activity_zero_coverage_has_a_single_strict_shape(self) -> None:
        activity = MoodSocialActivity.model_validate(
            _zero_coverage_activity()
        )
        self.assertEqual(activity.valid_daytime_detection_minutes, 0.0)

        invalid = _zero_coverage_activity()
        invalid["daytime_active_minutes"] = 0.0
        with self.assertRaises(ValidationError):
            MoodSocialActivity.model_validate(invalid)

        invalid = _zero_coverage_activity()
        invalid["hourly_activity_intensity"][6] = 0.0
        with self.assertRaises(ValidationError):
            MoodSocialActivity.model_validate(invalid)

    def test_activity_validates_hourly_totals_low_activity_and_scene_keys(
        self,
    ) -> None:
        activity = MoodSocialActivity.model_validate(_observed_activity())
        self.assertEqual(activity.weighted_daytime_activity, 30.0)

        invalid = _observed_activity()
        invalid["hourly_valid_detection_minutes"][6] = 59.0
        with self.assertRaises(ValidationError):
            MoodSocialActivity.model_validate(invalid)

        boundary = _observed_activity()
        boundary["hourly_valid_detection_minutes"][6] = 59.9
        MoodSocialActivity.model_validate(boundary)

        invalid = _observed_activity()
        invalid["hourly_valid_detection_minutes"][6] = 59.89
        with self.assertRaises(ValidationError):
            MoodSocialActivity.model_validate(invalid)

        boundary = _observed_activity()
        boundary["weighted_daytime_activity"] = 30.1
        MoodSocialActivity.model_validate(boundary)

        invalid = _observed_activity()
        invalid["weighted_daytime_activity"] = 30.1001
        with self.assertRaises(ValidationError):
            MoodSocialActivity.model_validate(invalid)

        invalid = _observed_activity()
        invalid["sedentary_bout_total_minutes"] = 21.0
        with self.assertRaises(ValidationError):
            MoodSocialActivity.model_validate(invalid)

        for field in (
            "sedentary_bout_total_minutes",
            "longest_sedentary_bout_minutes",
        ):
            invalid = _observed_activity()
            invalid[field] = 29.9
            with self.subTest(field=field), self.assertRaises(ValidationError):
                MoodSocialActivity.model_validate(invalid)

            no_bout = _observed_activity()
            no_bout["sedentary_bout_total_minutes"] = 0.0
            no_bout["longest_sedentary_bout_minutes"] = 0.0
            MoodSocialActivity.model_validate(no_bout)

        invalid = _observed_activity()
        invalid["low_activity_minutes"] = None
        invalid["sedentary_bout_total_minutes"] = 61.0
        invalid["longest_sedentary_bout_minutes"] = 61.0
        with self.assertRaises(ValidationError):
            MoodSocialActivity.model_validate(invalid)

        metric = {
            "camera_id": "cam-1",
            "scene_version": "scene-1",
            "gait_speed_image_norm_per_sec": 1.5,
        }
        invalid = _observed_activity()
        invalid["camera_gait_metrics"] = [metric, metric]
        with self.assertRaises(ValidationError):
            MoodSocialActivity.model_validate(invalid)

    def test_sleep_physiology_and_social_cross_field_rules(self) -> None:
        with self.assertRaises(ValidationError):
            MoodSocialSleep.model_validate({})
        with self.assertRaises(ValidationError):
            MoodSocialSleep.model_validate(
                {
                    "in_bed_minutes": 100.0,
                    "sleep_minutes": 80.0,
                    "sleep_efficiency": 0.5,
                }
            )
        MoodSocialSleep.model_validate(
            {
                "in_bed_minutes": 100.0,
                "sleep_minutes": 80.0,
                "sleep_efficiency": 0.81,
            }
        )
        with self.assertRaises(ValidationError):
            MoodSocialSleep.model_validate(
                {
                    "in_bed_minutes": 100.0,
                    "sleep_minutes": 80.0,
                    "sleep_efficiency": 0.8101,
                }
            )
        with self.assertRaises(ValidationError):
            MoodSocialPhysiology.model_validate({})

        unobserved = MoodSocialSocial.model_validate(
            {"call_log_observed": False}
        )
        self.assertFalse(unobserved.call_log_observed)
        with self.assertRaises(ValidationError):
            MoodSocialSocial.model_validate(
                {
                    "call_log_observed": False,
                    "answered_call_count": 0,
                }
            )

        observed = MoodSocialSocial.model_validate(_observed_social())
        self.assertTrue(observed.call_log_observed)
        invalid = _observed_social()
        invalid["incoming_call_opportunities"] = 1
        with self.assertRaises(ValidationError):
            MoodSocialSocial.model_validate(invalid)

    def test_rejects_derived_social_and_legacy_camera_fields(self) -> None:
        social = _observed_social()
        social["answer_rate"] = 0.0
        with self.assertRaises(ValidationError):
            MoodSocialSocial.model_validate(social)

        activity = _observed_activity()
        activity["daily_step_count"] = 100
        with self.assertRaises(ValidationError):
            MoodSocialActivity.model_validate(activity)

        metric = {
            "camera_id": "cam-1",
            "scene_version": "scene-1",
            "gait_speed_image_norm_per_sec": 2.0,
            "gait_speed_mps": 1.0,
        }
        with self.assertRaises(ValidationError):
            CameraGaitMetric.model_validate(metric)

    def test_history_range_order_content_and_attention_capacity(self) -> None:
        payload = _minimal_request_payload()
        target = str(payload["target_date"])
        history_record = {
            "date": _date_before(target, 28),
            "activity": _zero_coverage_activity(),
            "sleep": None,
            "physiology": None,
            "social": None,
        }
        payload["history_daily_features"] = [history_record]
        MoodSocialInferRequest.model_validate(payload)

        invalid = copy.deepcopy(payload)
        invalid["history_daily_features"][0]["date"] = _date_before(
            target,
            29,
        )
        with self.assertRaises(ValidationError):
            MoodSocialInferRequest.model_validate(invalid)

        preserved = _minimal_request_payload()
        preserved["history_attention_indices"] = [
            {
                "date": _date_before(target, 1),
                "attention_index": 0.2,
                "model_version": " mood-fusion-v3.3.3 ",
            }
        ]
        request = MoodSocialInferRequest.model_validate(preserved)
        self.assertEqual(
            request.history_attention_indices[0].model_version,
            " mood-fusion-v3.3.3 ",
        )

        historical_model = _minimal_request_payload()
        historical_model["history_attention_indices"] = [
            {
                "date": _date_before(target, 1),
                "attention_index": 0.2,
                "model_version": "mood-fusion-v3.3.2",
            }
        ]
        request = MoodSocialInferRequest.model_validate(historical_model)
        self.assertEqual(
            request.history_attention_indices[0].model_version,
            "mood-fusion-v3.3.2",
        )

        invalid = _minimal_request_payload()
        invalid["history_daily_features"] = [
            {
                "date": _date_before(target, 1),
                "activity": None,
                "sleep": None,
                "physiology": None,
                "social": None,
            }
        ]
        with self.assertRaises(ValidationError):
            MoodSocialInferRequest.model_validate(invalid)

        invalid = _minimal_request_payload()
        invalid["history_attention_indices"] = [
            {
                "date": _date_before(target, day),
                "attention_index": 0.2,
                "model_version": "mood-fusion-v3.3.3",
            }
            for day in range(7, 0, -1)
        ]
        with self.assertRaises(ValidationError):
            MoodSocialInferRequest.model_validate(invalid)

    def test_source_declaration_covers_current_and_history_domains(self) -> None:
        payload = _minimal_request_payload()
        payload["current_daily_features"]["social"] = {
            "call_log_observed": False
        }
        with self.assertRaises(ValidationError) as context:
            MoodSocialInferRequest.model_validate(payload)

        self.assertIn(
            "source_declaration_mismatch",
            {error["type"] for error in context.exception.errors()},
        )

    def test_date_collection_and_attention_history_edges(self) -> None:
        payload = _minimal_request_payload()
        target = str(payload["target_date"])

        mismatched_current = copy.deepcopy(payload)
        mismatched_current["current_daily_features"]["date"] = _date_before(
            target,
            1,
        )
        with self.assertRaises(ValidationError):
            MoodSocialInferRequest.model_validate(mismatched_current)

        too_many = copy.deepcopy(payload)
        too_many["history_daily_features"] = [
            {
                "date": _date_before(target, days_before),
                "activity": _zero_coverage_activity(),
                "sleep": None,
                "physiology": None,
                "social": None,
            }
            for days_before in range(29, 0, -1)
        ]
        with self.assertRaises(ValidationError):
            MoodSocialInferRequest.model_validate(too_many)

        ordered = copy.deepcopy(payload)
        ordered["history_daily_features"] = [
            {
                "date": _date_before(target, days_before),
                "activity": _zero_coverage_activity(),
                "sleep": None,
                "physiology": None,
                "social": None,
            }
            for days_before in (2, 1)
        ]
        reversed_history = copy.deepcopy(ordered)
        reversed_history["history_daily_features"].reverse()
        with self.assertRaises(ValidationError):
            MoodSocialInferRequest.model_validate(reversed_history)

        duplicate_history = copy.deepcopy(ordered)
        duplicate_history["history_daily_features"][1]["date"] = (
            duplicate_history["history_daily_features"][0]["date"]
        )
        with self.assertRaises(ValidationError):
            MoodSocialInferRequest.model_validate(duplicate_history)

        missing_model_version = copy.deepcopy(payload)
        missing_model_version["history_attention_indices"] = [
            {
                "date": _date_before(target, 1),
                "attention_index": 0.2,
            }
        ]
        with self.assertRaises(ValidationError):
            MoodSocialInferRequest.model_validate(missing_model_version)


class MoodSocialResponseSchemaTest(unittest.TestCase):
    def test_validates_frozen_unavailable_shape(self) -> None:
        response = MoodSocialInferResponse.model_validate(
            _unavailable_response_payload()
        )
        self.assertFalse(response.available)

        forward_compatible = _unavailable_response_payload()
        forward_compatible["limitations"] = [
            "vendor.future-code",
            "insufficient_data",
        ]
        MoodSocialInferResponse.model_validate(forward_compatible)

        invalid = _unavailable_response_payload()
        invalid["attention_index"] = 0.0
        with self.assertRaises(ValidationError):
            MoodSocialInferResponse.model_validate(invalid)

        for invalid_diagnosis in (True, 0):
            invalid = _unavailable_response_payload()
            invalid["diagnosis"] = invalid_diagnosis
            with self.subTest(
                diagnosis=invalid_diagnosis
            ), self.assertRaises(ValidationError):
                MoodSocialInferResponse.model_validate(invalid)

        invalid = _available_response_payload()
        invalid["used_sources"] = []
        invalid["covered_domains"] = []
        invalid["domain_scores"] = {
            "activity": None,
            "sleep": None,
            "physiology": None,
            "activity_sleep_joint": None,
            "social_context": None,
            "personal_change": {
                "activity": None,
                "sleep": None,
                "social": None,
            },
        }
        with self.assertRaises(ValidationError):
            MoodSocialInferResponse.model_validate(invalid)

        invalid = _available_response_payload()
        invalid["limitations"] = ["insufficient_data"]
        with self.assertRaises(ValidationError):
            MoodSocialInferResponse.model_validate(invalid)

    def test_validates_half_up_score_and_level_boundaries(self) -> None:
        cases = (
            (0.005, 1, 0),
            (0.145, 15, 0),
            (0.25, 25, 1),
            (0.285, 29, 1),
            (0.45, 45, 2),
            (0.65, 65, 3),
            (0.995, 100, 3),
        )
        for index, score, level in cases:
            with self.subTest(index=index):
                payload = _available_response_payload(index)
                response = MoodSocialInferResponse.model_validate(payload)
                self.assertEqual(response.attention_score, score)
                self.assertEqual(response.attention_level, level)

    def test_validates_source_domain_alignment_contributions_and_summary(
        self,
    ) -> None:
        payload = _available_response_payload()
        payload["model_contributions"] = [
            {
                "factor": "observed_daytime_activity_low",
                "label": "有效观测时段内白天活动偏低",
                "direction": "up",
                "contribution": 0.3,
            },
            {
                "factor": "activity_rhythm_irregular",
                "label": "日间活动节律不规则",
                "direction": "up",
                "contribution": 0.2,
            },
        ]
        payload["summary"] = (
            "有效观测时段内白天活动偏低、日间活动节律不规则"
            "，建议家属主动沟通并继续观察。"
        )
        MoodSocialInferResponse.model_validate(payload)

        invalid = copy.deepcopy(payload)
        invalid["model_contributions"].reverse()
        with self.assertRaises(ValidationError):
            MoodSocialInferResponse.model_validate(invalid)

        invalid = copy.deepcopy(payload)
        invalid["used_sources"] = ["sleep_device", "camera"]
        with self.assertRaises(ValidationError):
            MoodSocialInferResponse.model_validate(invalid)

        invalid = copy.deepcopy(payload)
        invalid["summary"] = "自由生成的总结"
        with self.assertRaises(ValidationError):
            MoodSocialInferResponse.model_validate(invalid)

        invalid = copy.deepcopy(payload)
        invalid["model_contributions"][0]["label"] = "错误标签"
        with self.assertRaises(ValidationError):
            MoodSocialInferResponse.model_validate(invalid)

        invalid = copy.deepcopy(payload)
        invalid["model_contributions"][0]["label"] = (
            f" {invalid['model_contributions'][0]['label']} "
        )
        invalid["summary"] = (
            f" {invalid['model_contributions'][0]['label']}，"
            "建议家属主动沟通并继续观察。"
        )
        with self.assertRaises(ValidationError):
            MoodSocialInferResponse.model_validate(invalid)

        invalid = _available_response_payload()
        invalid["summary"] = f" {invalid['summary']} "
        with self.assertRaises(ValidationError):
            MoodSocialInferResponse.model_validate(invalid)

        invalid = _available_response_payload()
        invalid["model_contributions"] = [
            {
                "factor": "sleep_efficiency_low",
                "label": "睡眠效率偏低",
                "direction": "up",
                "contribution": 0.3,
            }
        ]
        invalid["summary"] = "睡眠效率偏低，建议家属主动沟通并继续观察。"
        with self.assertRaises(ValidationError):
            MoodSocialInferResponse.model_validate(invalid)

    def test_error_model_is_independent_and_strict(self) -> None:
        response = MoodSocialErrorResponse.model_validate(
            {
                "detail": {
                    "schema_version": "mood_social_error_v1",
                    "request_id": None,
                    "code": "FIELD_VALIDATION_ERROR",
                    "message": "请求字段校验失败",
                    "errors": [
                        {"field": "profile", "reason": "unknown field"}
                    ],
                }
            }
        )
        self.assertEqual(response.detail.schema_version, "mood_social_error_v1")

        invalid = response.model_dump()
        invalid["detail"]["extra"] = True
        with self.assertRaises(ValidationError):
            MoodSocialErrorResponse.model_validate(invalid)

    def test_attribution_rejects_change_without_evidence_and_prohibited_output(
        self,
    ) -> None:
        missing_change_branch = _available_response_payload()
        missing_change_branch["model_contributions"] = [
            {
                "factor": "personal_activity_deviation",
                "label": "活动状态偏离个人历史",
                "direction": "up",
                "contribution": 0.2,
            }
        ]
        missing_change_branch["summary"] = (
            "活动状态偏离个人历史，建议家属主动沟通并继续观察。"
        )
        with self.assertRaises(ValidationError):
            MoodSocialInferResponse.model_validate(missing_change_branch)

        four_contributions = _available_response_payload()
        four_contributions["model_contributions"] = [
            {
                "factor": factor,
                "label": label,
                "direction": "up",
                "contribution": contribution,
            }
            for factor, label, contribution in (
                ("observed_daytime_activity_low", "有效观测时段内白天活动偏低", 0.4),
                ("observed_low_activity_time_high", "有效观测时段内低活动时间偏高", 0.3),
                ("activity_rhythm_irregular", "日间活动节律不规则", 0.2),
                ("bias", "偏置", 0.1),
            )
        ]
        with self.assertRaises(ValidationError):
            MoodSocialInferResponse.model_validate(four_contributions)

        prohibited_factor = _available_response_payload()
        prohibited_factor["model_contributions"] = [
            {
                "factor": "mask",
                "label": "掩码",
                "direction": "up",
                "contribution": 0.2,
            }
        ]
        with self.assertRaises(ValidationError):
            MoodSocialInferResponse.model_validate(prohibited_factor)


class MoodSocialRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        settings = ServiceSettings(
            api_token="api",
            model_path="missing-fall-model.pt",
        )
        self.app = create_app(settings=settings)
        self.client = TestClient(
            self.app,
            raise_server_exceptions=False,
        )

    @staticmethod
    def headers() -> dict[str, str]:
        return {"Authorization": "Bearer api"}

    def test_legal_request_without_evidence_returns_available_false_200(
        self,
    ) -> None:
        response = self.client.post(
            "/v1/mental-health/mood-social/infer",
            json=_minimal_request_payload(),
            headers=self.headers(),
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["available"])
        self.assertEqual(response.json()["evidence_scope"], "insufficient_data")
        self.assertIn("insufficient_data", response.json()["limitations"])
        self.assertNotIn("results", response.text)
        self.assertNotIn("mental_safety", response.text)

    def test_complete_legacy_placeholder_bundle_cannot_bypass_package_gate(
        self,
    ) -> None:
        current = load_mood_social_config()
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_directory = Path(temporary_directory)
            for filename in current.models.artifacts.values():
                (artifact_directory / filename).write_bytes(b"placeholder")
            configured = replace(
                current,
                models=replace(
                    current.models,
                    directory=artifact_directory,
                ),
            )
            with patch(
                "elderly_monitoring.modules.mental_health.mood_social.api."
                "load_mood_social_config",
                return_value=configured,
            ):
                response = self.client.post(
                    "/v1/mental-health/mood-social/infer",
                    json=_minimal_request_payload(),
                    headers=self.headers(),
                )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"]["code"],
            "MODEL_ARTIFACT_UNAVAILABLE",
        )

    def test_hash_invalid_versioned_package_returns_structured_503(self) -> None:
        current = load_mood_social_config()
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_directory = Path(temporary_directory) / "v3.3.3"
            package_directory = model_directory / "packages" / "MH-20260802-013"
            shutil.copytree(DEFAULT_PACKAGE_DIRECTORY, package_directory)
            (package_directory / "selection.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            configured = replace(
                current,
                models=replace(current.models, directory=model_directory),
            )
            with patch(
                "elderly_monitoring.modules.mental_health.mood_social.api."
                "load_mood_social_config",
                return_value=configured,
            ):
                response = self.client.post(
                    "/v1/mental-health/mood-social/infer",
                    json=_minimal_request_payload(),
                    headers=self.headers(),
                )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"]["code"],
            "MODEL_ARTIFACT_UNAVAILABLE",
        )

    def test_activity_evidence_returns_real_v3_pipeline_result(self) -> None:
        payload = _minimal_request_payload()
        payload["request_id"] = "req_api002_activity"
        payload["current_daily_features"]["activity"] = _observed_activity()
        response = self.client.post(
            "/v1/mental-health/mood-social/infer",
            json=payload,
            headers=self.headers(),
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["available"])
        self.assertEqual(body["used_sources"], ["camera"])
        self.assertEqual(body["covered_domains"], ["activity"])
        self.assertIsInstance(body["attention_index"], float)
        self.assertFalse(body["diagnosis"])

    def test_authentication_validation_and_internal_errors_use_error_v1(
        self,
    ) -> None:
        unauthorized = self.client.post(
            "/v1/mental-health/mood-social/infer",
            json=_minimal_request_payload(),
        )
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(
            unauthorized.json()["detail"]["code"],
            "AUTHENTICATION_FAILED",
        )

        malformed = self.client.post(
            "/v1/mental-health/mood-social/infer",
            content="{",
            headers={
                **self.headers(),
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(malformed.status_code, 422)
        self.assertEqual(
            malformed.json()["detail"]["schema_version"],
            "mood_social_error_v1",
        )

        with patch(
            "elderly_monitoring.service.app.infer_mood_social",
            side_effect=RuntimeError("do not expose this"),
        ):
            internal = self.client.post(
                "/v1/mental-health/mood-social/infer",
                json=_minimal_request_payload(),
                headers=self.headers(),
            )
        self.assertEqual(internal.status_code, 500)
        self.assertEqual(internal.json()["detail"]["code"], "INTERNAL_ERROR")
        self.assertNotIn("do not expose this", internal.text)

    def test_old_and_unknown_schema_versions_use_specific_422_code(self) -> None:
        for version in (
            "mood_social_infer_request_v1",
            "mood_social_infer_request_v2",
            "mood_social_infer_request_v99",
        ):
            payload = _minimal_request_payload()
            payload["schema_version"] = version
            response = self.client.post(
                "/v1/mental-health/mood-social/infer",
                json=payload,
                headers=self.headers(),
            )
            with self.subTest(version=version):
                self.assertEqual(response.status_code, 422)
                self.assertEqual(
                    response.json()["detail"]["code"],
                    "UNSUPPORTED_SCHEMA_VERSION",
                )

    def test_date_source_and_nested_field_errors_are_classified(self) -> None:
        current_day = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
        invalid_date = _minimal_request_payload()
        invalid_date["target_date"] = current_day
        invalid_date["current_daily_features"]["date"] = current_day
        response = self.client.post(
            "/v1/mental-health/mood-social/infer",
            json=invalid_date,
            headers=self.headers(),
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["detail"]["code"],
            "DATE_VALIDATION_ERROR",
        )

        invalid_date_and_social = _minimal_request_payload()
        invalid_date_and_social["target_date"] = current_day
        invalid_date_and_social["current_daily_features"]["date"] = current_day
        invalid_date_and_social["available_sources"] = ["s10"]
        invalid_date_and_social["current_daily_features"]["social"] = {
            "call_log_observed": True
        }
        response = self.client.post(
            "/v1/mental-health/mood-social/infer",
            json=invalid_date_and_social,
            headers=self.headers(),
        )
        self.assertEqual(
            response.json()["detail"]["code"],
            "DATE_VALIDATION_ERROR",
        )

        source_mismatch = _minimal_request_payload()
        source_mismatch["current_daily_features"]["social"] = {
            "call_log_observed": False
        }
        response = self.client.post(
            "/v1/mental-health/mood-social/infer",
            json=source_mismatch,
            headers=self.headers(),
        )
        self.assertEqual(
            response.json()["detail"]["code"],
            "SOURCE_DECLARATION_MISMATCH",
        )

        source_and_social_mismatch = _minimal_request_payload()
        source_and_social_mismatch["current_daily_features"]["social"] = {
            "call_log_observed": True
        }
        response = self.client.post(
            "/v1/mental-health/mood-social/infer",
            json=source_and_social_mismatch,
            headers=self.headers(),
        )
        self.assertEqual(
            response.json()["detail"]["code"],
            "SOURCE_DECLARATION_MISMATCH",
        )

        nested_activity = _minimal_request_payload()
        nested_activity["current_daily_features"]["activity"] = (
            _observed_activity()
        )
        nested_activity["current_daily_features"]["activity"][
            "weighted_daytime_activity"
        ] = 29.0
        response = self.client.post(
            "/v1/mental-health/mood-social/infer",
            json=nested_activity,
            headers=self.headers(),
        )
        self.assertEqual(
            response.json()["detail"]["errors"][0]["field"],
            "current_daily_features.activity.weighted_daytime_activity",
        )

        invalid_gait = _minimal_request_payload()
        invalid_gait["current_daily_features"]["activity"] = (
            _observed_activity()
        )
        invalid_gait["current_daily_features"]["activity"][
            "camera_gait_metrics"
        ] = [
            {
                "camera_id": "cam-1",
                "scene_version": "scene-1",
                "gait_speed_image_norm_per_sec": None,
                "sit_to_stand_duration_seconds": None,
                "turn_duration_seconds": None,
                "postural_stability": None,
            }
        ]
        response = self.client.post(
            "/v1/mental-health/mood-social/infer",
            json=invalid_gait,
            headers=self.headers(),
        )
        self.assertEqual(
            response.json()["detail"]["errors"][0]["field"],
            "current_daily_features.activity.camera_gait_metrics[0]",
        )

        history_social = _minimal_request_payload()
        history_social["available_sources"] = ["camera", "s10"]
        history_social["history_daily_features"] = [
            {
                "date": _date_before(str(history_social["target_date"]), 1),
                "activity": None,
                "sleep": None,
                "physiology": None,
                "social": {"call_log_observed": True},
            }
        ]
        response = self.client.post(
            "/v1/mental-health/mood-social/infer",
            json=history_social,
            headers=self.headers(),
        )
        self.assertEqual(
            response.json()["detail"]["errors"][0]["field"],
            "history_daily_features[0].social.call_log_observed",
        )

        unknown = _minimal_request_payload()
        unknown["profile"]["unknown"] = True
        response = self.client.post(
            "/v1/mental-health/mood-social/infer",
            json=unknown,
            headers=self.headers(),
        )
        self.assertEqual(
            response.json()["detail"]["code"],
            "FIELD_VALIDATION_ERROR",
        )

    def test_service_schema_module_reexports_the_v3_types(self) -> None:
        from elderly_monitoring.service import schemas

        self.assertIs(schemas.MoodSocialInferRequest, MoodSocialInferRequest)
        self.assertIs(schemas.MoodSocialInferResponse, MoodSocialInferResponse)
        self.assertIs(schemas.MoodSocialErrorResponse, MoodSocialErrorResponse)

    def test_openapi_freezes_security_and_all_route_models(self) -> None:
        operation = self.app.openapi()["paths"][
            "/v1/mental-health/mood-social/infer"
        ]["post"]

        self.assertTrue(operation["security"])
        self.assertTrue(
            operation["requestBody"]["content"]["application/json"]["schema"][
                "$ref"
            ].endswith("/MoodSocialInferRequest")
        )
        expected_models = {
            "200": "MoodSocialInferResponse",
            "401": "MoodSocialErrorResponse",
            "422": "MoodSocialErrorResponse",
            "500": "MoodSocialErrorResponse",
            "503": "MoodSocialErrorResponse",
        }
        for status, model_name in expected_models.items():
            with self.subTest(status=status):
                schema = operation["responses"][status]["content"][
                    "application/json"
                ]["schema"]
                self.assertTrue(schema["$ref"].endswith(f"/{model_name}"))

    def test_route_specific_handlers_do_not_change_legacy_validation(self) -> None:
        response = self.client.post(
            "/v1/mental-health/daily-risk",
            json={},
            headers=self.headers(),
        )

        self.assertEqual(response.status_code, 422)
        self.assertIsInstance(response.json()["detail"], list)

    def test_route_specific_handlers_do_not_change_legacy_internal_errors(
        self,
    ) -> None:
        with patch(
            "elderly_monitoring.service.mental_health_daily."
            "build_mental_health_daily_risk_result",
            side_effect=RuntimeError("legacy failure"),
        ):
            response = self.client.post(
                "/v1/mental-health/daily-risk",
                json={
                    "person_id": "elder-001",
                    "history_daily_features": [],
                    "current_daily_features": [],
                },
                headers=self.headers(),
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.text, "Internal Server Error")

    def test_route_endpoint_only_calls_the_new_mood_social_shell(self) -> None:
        route = next(
            route
            for route in self.app.routes
            if getattr(route, "path", None)
            == "/v1/mental-health/mood-social/infer"
        )
        source = inspect.getsource(route.endpoint)
        tree = ast.parse(inspect.cleandoc(source))
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }

        self.assertIn("infer_mood_social", names)
        self.assertNotIn("MentalHealthRiskPipeline", names)
        self.assertNotIn("build_mental_health_daily_risk_result", names)
        self.assertNotIn("scorecards", source)

    def test_fresh_import_does_not_load_legacy_pipeline_or_scorecards(
        self,
    ) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        script = (
            f"import sys; sys.path.insert(0, {str(source_root)!r}); "
            "import elderly_monitoring.service.app; "
            "print(int('elderly_monitoring.modules.mental_health.pipeline' "
            "in sys.modules), "
            "int(any(name.startswith("
            "'elderly_monitoring.modules.mental_health.scorecards') "
            "for name in sys.modules)))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "0 0")


if __name__ == "__main__":
    unittest.main()
