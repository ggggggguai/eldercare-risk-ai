from __future__ import annotations

import json
import unittest
from collections import Counter
from dataclasses import FrozenInstanceError
from datetime import date

import elderly_monitoring.modules.mental_health.mood_social as mood_social
from elderly_monitoring.modules.mental_health.mood_social import feature_schema
from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    ACTIVITY_EXPERT_FEATURE_ORDER,
    ACTIVITY_FEATURE_ORDER,
    ACTIVITY_FEATURE_SPECS,
    CAMERA_GAIT_FEATURE_ORDER,
    CAMERA_GAIT_FEATURE_SPECS,
    CAMERA_GAIT_KEY_ORDER,
    DAILY_FEATURE_GROUP_SPECS,
    FEATURE_GROUP_ORDER,
    FEATURE_GROUP_SPECS,
    FEATURE_SCHEMA_VERSION,
    HISTORY_LOOKBACK_DAYS,
    PHYSIOLOGY_FEATURE_ORDER,
    SLEEP_FEATURE_ORDER,
    SLEEP_FEATURE_SPECS,
    SOCIAL_CONTACT_FEATURE_ORDER,
    SOCIAL_CONTEXT_FEATURE_ORDER,
    SOCIAL_CONTEXT_FEATURE_SPECS,
    STATE_WINDOW_DAYS,
    SUPERVISED_EXPERT_FEATURE_ORDER,
    CameraGaitDay,
    DomainFeatureVector,
    FeatureRole,
    FeatureSpec,
    FeatureValueType,
    RiskDirection,
    feature_schema_manifest,
)


WINDOWED_METADATA = {
    "activity": (
        ("activity_volume_norm", "float", "decrease", "supervised"),
        ("active_ratio", "float", "decrease", "supervised"),
        ("sedentary_ratio", "float", "increase", "supervised"),
        ("longest_inactive_bout_norm", "float", "increase", "supervised"),
        ("relative_amplitude", "float", "decrease", "supervised"),
        ("interdaily_stability", "float", "decrease", "supervised"),
        ("intradaily_variability", "float", "increase", "supervised"),
        ("activity_variability", "float", "increase", "supervised"),
        ("valid_days", "integer", "coverage_only", "coverage"),
        ("feature_coverage", "float", "coverage_only", "coverage"),
    ),
    "sleep": (
        ("sleep_duration_norm", "float", "two_sided", "supervised"),
        ("time_in_bed_norm", "float", "two_sided", "supervised"),
        ("sleep_efficiency", "float", "decrease", "supervised"),
        (
            "sleep_onset_sin",
            "float",
            "circular_two_sided",
            "supervised",
        ),
        (
            "sleep_onset_cos",
            "float",
            "circular_two_sided",
            "supervised",
        ),
        ("wake_time_sin", "float", "circular_two_sided", "supervised"),
        ("wake_time_cos", "float", "circular_two_sided", "supervised"),
        (
            "sleep_midpoint_sin",
            "float",
            "circular_two_sided",
            "supervised",
        ),
        (
            "sleep_midpoint_cos",
            "float",
            "circular_two_sided",
            "supervised",
        ),
        ("sleep_fragmentation", "float", "increase", "supervised"),
        ("sleep_regularity", "float", "decrease", "supervised"),
        ("night_exit_count_mean", "float", "increase", "supervised"),
        ("valid_nights", "integer", "coverage_only", "coverage"),
        ("feature_coverage", "float", "coverage_only", "coverage"),
    ),
    "physiology": (
        ("heart_rate_mean_bpm", "float", "model_learned", "supervised"),
        ("heart_rate_min_bpm", "float", "model_learned", "supervised"),
        ("heart_rate_sd_bpm", "float", "model_learned", "supervised"),
        ("hrv_sdnn_ms", "float", "model_learned", "supervised"),
        (
            "respiration_rate_mean_bpm",
            "float",
            "model_learned",
            "supervised",
        ),
        (
            "respiration_rate_sd_bpm",
            "float",
            "model_learned",
            "supervised",
        ),
        (
            "respiratory_abnormal_ratio",
            "float",
            "model_learned",
            "supervised",
        ),
        ("snoring_minutes_norm", "float", "model_learned", "supervised"),
        ("valid_nights", "integer", "coverage_only", "coverage"),
        ("feature_coverage", "float", "coverage_only", "coverage"),
    ),
    "social_context": (
        ("age_group", "category", "model_learned", "supervised"),
        ("sex", "category", "model_learned", "supervised"),
        (
            "living_arrangement",
            "category",
            "model_learned",
            "supervised",
        ),
        ("marital_status", "category", "model_learned", "supervised"),
        (
            "chronic_disease_count",
            "integer",
            "model_learned",
            "supervised",
        ),
        ("self_rated_health", "integer", "model_learned", "supervised"),
        (
            "functional_limitation",
            "integer",
            "model_learned",
            "supervised",
        ),
        (
            "social_participation_days_per_week",
            "integer",
            "model_learned",
            "supervised",
        ),
        ("education_level", "category", "model_learned", "supervised"),
        ("economic_status", "category", "model_learned", "supervised"),
    ),
    "social_contact": (
        (
            "incoming_call_opportunities",
            "float",
            "model_learned",
            "personal_trend",
        ),
        ("outgoing_call_count", "float", "decrease", "personal_trend"),
        ("answered_call_count", "float", "decrease", "personal_trend"),
        ("missed_call_count", "float", "increase", "personal_trend"),
        ("answer_rate", "float", "decrease", "personal_trend"),
        ("connected_call_count", "float", "decrease", "personal_trend"),
        (
            "connected_duration_minutes",
            "float",
            "decrease",
            "personal_trend",
        ),
        (
            "mean_connected_duration_minutes",
            "float",
            "decrease",
            "personal_trend",
        ),
        ("active_contact_count", "float", "decrease", "personal_trend"),
        (
            "no_effective_contact_days",
            "integer",
            "increase",
            "personal_trend",
        ),
    ),
}


DAILY_METADATA = {
    "activity": (
        (
            "observed_activity_intensity",
            "float",
            "decrease",
            "personal_trend",
        ),
        ("active_ratio", "float", "decrease", "personal_trend"),
        ("sedentary_ratio", "float", "increase", "personal_trend"),
        (
            "longest_inactive_bout_norm",
            "float",
            "increase",
            "personal_trend",
        ),
    ),
    "sleep": (
        ("sleep_duration_norm", "float", "two_sided", "personal_trend"),
        ("time_in_bed_norm", "float", "two_sided", "personal_trend"),
        ("sleep_efficiency", "float", "decrease", "personal_trend"),
        (
            "sleep_onset_minute_of_day",
            "integer",
            "circular_two_sided",
            "personal_trend",
        ),
        (
            "wake_time_minute_of_day",
            "integer",
            "circular_two_sided",
            "personal_trend",
        ),
        (
            "sleep_midpoint_minute_of_day",
            "float",
            "circular_two_sided",
            "personal_trend",
        ),
        ("sleep_fragmentation", "float", "increase", "personal_trend"),
        ("night_exit_count", "integer", "increase", "personal_trend"),
        (
            "night_exit_minutes_norm",
            "float",
            "increase",
            "personal_trend",
        ),
    ),
    "physiology": (
        ("heart_rate_mean_bpm", "float", "model_learned", "supervised"),
        ("heart_rate_min_bpm", "float", "model_learned", "supervised"),
        ("heart_rate_sd_bpm", "float", "model_learned", "supervised"),
        ("hrv_sdnn_ms", "float", "model_learned", "supervised"),
        (
            "respiration_rate_mean_bpm",
            "float",
            "model_learned",
            "supervised",
        ),
        (
            "respiration_rate_sd_bpm",
            "float",
            "model_learned",
            "supervised",
        ),
        (
            "respiratory_abnormal_ratio",
            "float",
            "model_learned",
            "supervised",
        ),
        ("snoring_minutes_norm", "float", "model_learned", "supervised"),
    ),
    "social": (
        (
            "incoming_call_opportunities",
            "integer",
            "model_learned",
            "personal_trend",
        ),
        ("outgoing_call_count", "integer", "decrease", "personal_trend"),
        ("answered_call_count", "integer", "decrease", "personal_trend"),
        ("missed_call_count", "integer", "increase", "personal_trend"),
        ("answer_rate", "float", "decrease", "personal_trend"),
        ("connected_call_count", "integer", "decrease", "personal_trend"),
        (
            "connected_duration_minutes",
            "float",
            "decrease",
            "personal_trend",
        ),
        (
            "mean_connected_duration_minutes",
            "float",
            "decrease",
            "personal_trend",
        ),
        ("active_contact_count", "integer", "decrease", "personal_trend"),
        (
            "no_effective_contact_days",
            "integer",
            "increase",
            "personal_trend",
        ),
    ),
}


def _metadata(specs: tuple[FeatureSpec, ...]) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (
            spec.name,
            spec.value_type.value,
            spec.risk_direction.value,
            spec.role.value,
        )
        for spec in specs
    )


class MoodSocialFeatureSchemaTest(unittest.TestCase):
    def test_windowed_group_and_field_orders_are_exact(self) -> None:
        self.assertEqual(FEATURE_GROUP_ORDER, tuple(WINDOWED_METADATA))
        self.assertEqual(tuple(FEATURE_GROUP_SPECS), FEATURE_GROUP_ORDER)

        exported_orders = {
            "activity": ACTIVITY_FEATURE_ORDER,
            "sleep": SLEEP_FEATURE_ORDER,
            "physiology": PHYSIOLOGY_FEATURE_ORDER,
            "social_context": SOCIAL_CONTEXT_FEATURE_ORDER,
            "social_contact": SOCIAL_CONTACT_FEATURE_ORDER,
        }
        for group in FEATURE_GROUP_ORDER:
            with self.subTest(group=group):
                expected_names = tuple(item[0] for item in WINDOWED_METADATA[group])
                actual_names = tuple(spec.name for spec in FEATURE_GROUP_SPECS[group])
                self.assertEqual(actual_names, expected_names)
                self.assertEqual(exported_orders[group], expected_names)

    def test_daily_group_and_field_orders_are_exact(self) -> None:
        self.assertEqual(
            tuple(DAILY_FEATURE_GROUP_SPECS),
            ("activity", "sleep", "physiology", "social"),
        )
        for group, expected in DAILY_METADATA.items():
            with self.subTest(group=group):
                self.assertEqual(
                    tuple(spec.name for spec in DAILY_FEATURE_GROUP_SPECS[group]),
                    tuple(item[0] for item in expected),
                )

    def test_every_feature_has_the_frozen_type_direction_and_role(self) -> None:
        for group, expected in WINDOWED_METADATA.items():
            with self.subTest(scope="windowed", group=group):
                specs = FEATURE_GROUP_SPECS[group]
                self.assertEqual(_metadata(specs), expected)
                for spec in specs:
                    self.assertIsInstance(spec.value_type, FeatureValueType)
                    self.assertIsInstance(spec.risk_direction, RiskDirection)
                    self.assertIsInstance(spec.role, FeatureRole)

        for group, expected in DAILY_METADATA.items():
            with self.subTest(scope="daily", group=group):
                self.assertEqual(
                    _metadata(DAILY_FEATURE_GROUP_SPECS[group]),
                    expected,
                )

    def test_categories_and_circular_pairs_preserve_their_semantics(self) -> None:
        context_specs = {spec.name: spec for spec in SOCIAL_CONTEXT_FEATURE_SPECS}
        self.assertEqual(
            context_specs["age_group"].categories,
            ("60_69", "70_79", "80_plus"),
        )
        self.assertEqual(context_specs["sex"].categories, ("female", "male"))
        self.assertEqual(
            context_specs["living_arrangement"].categories,
            ("alone", "with_family", "institution", "other"),
        )
        self.assertEqual(
            context_specs["marital_status"].categories,
            ("partnered", "not_partnered"),
        )
        self.assertEqual(
            context_specs["education_level"].categories,
            ("primary_or_less", "middle", "high_or_above"),
        )
        self.assertEqual(
            context_specs["economic_status"].categories,
            ("low", "middle", "high"),
        )

        circular_groups: dict[str, list[str]] = {}
        for spec in SLEEP_FEATURE_SPECS:
            if spec.circular_group is not None:
                circular_groups.setdefault(spec.circular_group, []).append(spec.name)
        self.assertEqual(
            circular_groups,
            {
                "sleep_onset": ["sleep_onset_sin", "sleep_onset_cos"],
                "wake_time": ["wake_time_sin", "wake_time_cos"],
                "sleep_midpoint": [
                    "sleep_midpoint_sin",
                    "sleep_midpoint_cos",
                ],
            },
        )

    def test_exclusive_numeric_bounds_match_the_http_contract(self) -> None:
        physiology_specs = {
            spec.name: spec for spec in FEATURE_GROUP_SPECS["physiology"]
        }
        for name in (
            "heart_rate_mean_bpm",
            "heart_rate_min_bpm",
            "respiration_rate_mean_bpm",
        ):
            with self.subTest(feature=name):
                spec = physiology_specs[name]
                self.assertEqual(spec.minimum, 0.0)
                self.assertTrue(spec.exclusive_minimum)
                with self.assertRaisesRegex(ValueError, "below its minimum"):
                    DomainFeatureVector(
                        domain="test",
                        specs=(spec,),
                        values=(0.0,),
                        feature_mask=(1,),
                    )

        midpoint_spec = next(
            spec
            for spec in DAILY_FEATURE_GROUP_SPECS["sleep"]
            if spec.name == "sleep_midpoint_minute_of_day"
        )
        self.assertEqual(midpoint_spec.maximum, 1440.0)
        self.assertTrue(midpoint_spec.exclusive_maximum)
        with self.assertRaisesRegex(ValueError, "above its maximum"):
            DomainFeatureVector(
                domain="test",
                specs=(midpoint_spec,),
                values=(1440.0,),
                feature_mask=(1,),
            )

        for scope, specs in (
            ("windowed", FEATURE_GROUP_SPECS["social_contact"]),
            ("daily", DAILY_FEATURE_GROUP_SPECS["social"]),
        ):
            social_specs = {spec.name: spec for spec in specs}
            for name in (
                "connected_duration_minutes",
                "mean_connected_duration_minutes",
            ):
                with self.subTest(scope=scope, feature=name):
                    self.assertEqual(social_specs[name].maximum, 1440.0)

    def test_cross_group_duplicates_require_a_namespace(self) -> None:
        unqualified_names = [
            spec.name
            for group in FEATURE_GROUP_ORDER
            for spec in FEATURE_GROUP_SPECS[group]
        ]
        duplicate_counts = {
            name: count
            for name, count in Counter(unqualified_names).items()
            if count > 1
        }
        self.assertEqual(
            duplicate_counts,
            {"valid_nights": 2, "feature_coverage": 3},
        )

        for group in FEATURE_GROUP_ORDER:
            names = tuple(spec.name for spec in FEATURE_GROUP_SPECS[group])
            self.assertEqual(len(names), len(set(names)), group)

        namespaced_names = tuple(
            f"{group}.{spec.name}"
            for group in FEATURE_GROUP_ORDER
            for spec in FEATURE_GROUP_SPECS[group]
        )
        self.assertEqual(len(namespaced_names), len(set(namespaced_names)))

        expected_joint = (
            *(
                f"activity.{name}"
                for name in SUPERVISED_EXPERT_FEATURE_ORDER["activity"]
            ),
            *(f"sleep.{name}" for name in SUPERVISED_EXPERT_FEATURE_ORDER["sleep"]),
        )
        self.assertEqual(
            SUPERVISED_EXPERT_FEATURE_ORDER["activity_sleep_joint"],
            expected_joint,
        )
        self.assertEqual(len(expected_joint), len(set(expected_joint)))

    def test_expert_orders_keep_valid_counts_but_exclude_feature_coverage(
        self,
    ) -> None:
        self.assertEqual(
            SUPERVISED_EXPERT_FEATURE_ORDER["activity"],
            ACTIVITY_EXPERT_FEATURE_ORDER,
        )
        expected_valid_count = {
            "activity": "valid_days",
            "sleep": "valid_nights",
            "physiology": "valid_nights",
        }
        for expert, valid_count in expected_valid_count.items():
            with self.subTest(expert=expert):
                order = SUPERVISED_EXPERT_FEATURE_ORDER[expert]
                self.assertIn(valid_count, order)
                self.assertNotIn("feature_coverage", order)

    def test_camera_gait_trend_schema_is_frozen_and_rejects_empty_rows(self) -> None:
        self.assertEqual(
            CAMERA_GAIT_KEY_ORDER,
            ("date", "camera_id", "scene_version"),
        )
        self.assertEqual(
            CAMERA_GAIT_FEATURE_ORDER,
            (
                "gait_speed_image_norm_per_sec",
                "sit_to_stand_duration_seconds",
                "turn_duration_seconds",
                "postural_stability",
            ),
        )
        self.assertEqual(
            _metadata(CAMERA_GAIT_FEATURE_SPECS),
            (
                (
                    "gait_speed_image_norm_per_sec",
                    "float",
                    "decrease",
                    "personal_trend",
                ),
                (
                    "sit_to_stand_duration_seconds",
                    "float",
                    "increase",
                    "personal_trend",
                ),
                (
                    "turn_duration_seconds",
                    "float",
                    "increase",
                    "personal_trend",
                ),
                (
                    "postural_stability",
                    "float",
                    "decrease",
                    "personal_trend",
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "observed motion value"):
            CameraGaitDay(
                date=date(2026, 7, 28),
                camera_id="camera-a",
                scene_version="scene-1",
                gait_speed_image_norm_per_sec=None,
                sit_to_stand_duration_seconds=None,
                turn_duration_seconds=None,
                postural_stability=None,
            )

    def test_schema_constants_and_specs_are_immutable(self) -> None:
        with self.assertRaises(TypeError):
            FEATURE_GROUP_SPECS["activity"] = ()
        with self.assertRaises(TypeError):
            DAILY_FEATURE_GROUP_SPECS["activity"] = ()
        with self.assertRaises(TypeError):
            SUPERVISED_EXPERT_FEATURE_ORDER["activity"] = ()
        with self.assertRaises(TypeError):
            ACTIVITY_FEATURE_SPECS[0] = ACTIVITY_FEATURE_SPECS[1]
        with self.assertRaises(FrozenInstanceError):
            ACTIVITY_FEATURE_SPECS[0].name = "changed"

        category_spec = SOCIAL_CONTEXT_FEATURE_SPECS[0]
        serialized = category_spec.to_dict()
        serialized["categories"].append("unknown")
        self.assertEqual(category_spec.categories, ("60_69", "70_79", "80_plus"))

    def test_package_facade_exports_the_complete_feature_schema_api(self) -> None:
        self.assertTrue(set(feature_schema.__all__).issubset(mood_social.__all__))
        for name in feature_schema.__all__:
            with self.subTest(name=name):
                self.assertIs(getattr(mood_social, name), getattr(feature_schema, name))

    def test_manifest_is_deterministic_json_ready_and_isolated_per_call(self) -> None:
        first = feature_schema_manifest()
        second = feature_schema_manifest()
        canonical = json.dumps(first, ensure_ascii=True, separators=(",", ":"))

        self.assertEqual(first, second)
        self.assertEqual(
            canonical,
            json.dumps(second, ensure_ascii=True, separators=(",", ":")),
        )
        self.assertIsNot(first, second)
        self.assertIsNot(first["groups"], second["groups"])
        self.assertIsNot(
            first["groups"]["activity"],
            second["groups"]["activity"],
        )
        self.assertEqual(
            list(first),
            [
                "schema_version",
                "state_window_days",
                "history_lookback_days",
                "group_order",
                "groups",
                "daily_groups",
                "trend_context",
                "supervised_expert_feature_order",
            ],
        )
        self.assertEqual(first["schema_version"], FEATURE_SCHEMA_VERSION)
        self.assertEqual(first["state_window_days"], STATE_WINDOW_DAYS)
        self.assertEqual(first["history_lookback_days"], HISTORY_LOOKBACK_DAYS)
        self.assertEqual(tuple(first["groups"]), FEATURE_GROUP_ORDER)
        self.assertEqual(
            tuple(first["daily_groups"]),
            ("activity", "sleep", "physiology", "social"),
        )
        self.assertEqual(
            first["trend_context"]["camera_gait"]["key_order"],
            list(CAMERA_GAIT_KEY_ORDER),
        )
        self.assertEqual(
            tuple(
                item["name"]
                for item in first["trend_context"]["camera_gait"]["features"]
            ),
            CAMERA_GAIT_FEATURE_ORDER,
        )
        self.assertEqual(
            tuple(first["supervised_expert_feature_order"]),
            (
                "activity",
                "sleep",
                "activity_sleep_joint",
                "physiology",
                "social_context",
            ),
        )

        first["group_order"].reverse()
        first["groups"]["activity"][0]["name"] = "changed"
        first["groups"]["social_context"][0]["categories"].append("unknown")
        first["daily_groups"]["activity"].clear()
        first["trend_context"]["camera_gait"]["features"].clear()
        first["supervised_expert_feature_order"]["activity"].append("changed")

        third = feature_schema_manifest()
        self.assertEqual(
            json.dumps(third, ensure_ascii=True, separators=(",", ":")),
            canonical,
        )


if __name__ == "__main__":
    unittest.main()
