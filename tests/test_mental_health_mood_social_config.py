from __future__ import annotations

import ast
import copy
import importlib
import os
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from elderly_monitoring.common.config import load_yaml
from elderly_monitoring.modules.mental_health.mood_social import config as config_module
from elderly_monitoring.modules.mental_health.mood_social import (
    CURRENT_CONFIG_VERSION,
    DEFAULT_CONFIG_PATH,
    HISTORICAL_V3_3_2_CONFIG_PATH,
    load_mood_social_config,
    mood_social_config_from_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = (
    PROJECT_ROOT
    / "src"
    / "elderly_monitoring"
    / "modules"
    / "mental_health"
    / "mood_social"
)
PACKAGE_NAME = "elderly_monitoring.modules.mental_health.mood_social"
FORBIDDEN_LEGACY_MODULES = {
    "elderly_monitoring.modules.mental_health.pipeline",
    "elderly_monitoring.modules.mental_health.scorecards",
}


def _legacy_import_violations(source: str, source_name: str) -> list[str]:
    tree = ast.parse(source, filename=source_name)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules = [alias.name for alias in node.names]
            imported_symbols = [
                alias.name.rsplit(".", 1)[-1]
                for alias in node.names
            ]
        elif isinstance(node, ast.ImportFrom):
            relative_name = f"{'.' * node.level}{node.module or ''}"
            imported_base = (
                importlib.util.resolve_name(relative_name, PACKAGE_NAME)
                if node.level
                else (node.module or "")
            )
            imported_modules = [imported_base]
            imported_modules.extend(
                f"{imported_base}.{alias.name}"
                for alias in node.names
                if imported_base
            )
            imported_symbols = [alias.name for alias in node.names]
        else:
            continue

        if "MentalHealthRiskPipeline" in imported_symbols:
            violations.append(f"{source_name}: MentalHealthRiskPipeline")
        for module_name in imported_modules:
            if any(
                module_name == prefix
                or module_name.startswith(f"{prefix}.")
                for prefix in FORBIDDEN_LEGACY_MODULES
            ):
                violations.append(f"{source_name}: {module_name}")
    return violations


class MoodSocialPackageTest(unittest.TestCase):
    def test_package_and_reserved_subpackages_import(self) -> None:
        package = importlib.import_module(
            "elderly_monitoring.modules.mental_health.mood_social"
        )
        experts = importlib.import_module(
            "elderly_monitoring.modules.mental_health.mood_social.experts"
        )
        fusion = importlib.import_module(
            "elderly_monitoring.modules.mental_health.mood_social.fusion"
        )

        self.assertTrue(callable(package.load_mood_social_config))
        self.assertEqual(package.CURRENT_CONFIG_VERSION, "3.3.3")
        self.assertEqual(experts.__name__.rsplit(".", 1)[-1], "experts")
        self.assertEqual(fusion.__name__.rsplit(".", 1)[-1], "fusion")

    def test_new_package_has_no_legacy_pipeline_or_scorecard_imports(self) -> None:
        violations: list[str] = []
        for source_path in sorted(PACKAGE_ROOT.rglob("*.py")):
            violations.extend(
                _legacy_import_violations(
                    source_path.read_text(encoding="utf-8"),
                    str(source_path),
                )
            )
        self.assertEqual(violations, [])

    def test_legacy_import_scan_catches_absolute_and_relative_forms(self) -> None:
        cases = {
            "absolute pipeline": (
                "from elderly_monitoring.modules.mental_health.pipeline "
                "import MentalHealthRiskPipeline"
            ),
            "absolute parent alias": (
                "from elderly_monitoring.modules.mental_health import scorecards"
            ),
            "relative module": "from ..pipeline import MentalHealthRiskPipeline",
            "relative parent alias": "from .. import scorecards",
            "deep relative alias": "from ...mental_health import pipeline",
        }

        for label, source in cases.items():
            with self.subTest(label=label):
                self.assertTrue(_legacy_import_violations(source, label))
        self.assertEqual(
            _legacy_import_violations("from .config import RuntimeConfig", "safe"),
            [],
        )


class MoodSocialConfigTest(unittest.TestCase):
    def test_installed_layout_discovers_external_config_from_working_tree(
        self,
    ) -> None:
        installed_module = (
            Path("C:/temporary/site-packages")
            / "elderly_monitoring"
            / "modules"
            / "mental_health"
            / "mood_social"
            / "config.py"
        )
        with (
            patch.dict(
                os.environ,
                {
                    config_module.PROJECT_ROOT_ENV: "",
                    config_module.CONFIG_PATH_ENV: "",
                },
            ),
            patch.object(config_module, "__file__", str(installed_module)),
            patch.object(config_module.Path, "cwd", return_value=PROJECT_ROOT),
        ):
            discovered = config_module._discover_project_root()

        self.assertEqual(discovered, PROJECT_ROOT.resolve())

    def test_default_configuration_matches_v3_3_3_freeze(self) -> None:
        config = load_mood_social_config()

        self.assertEqual(CURRENT_CONFIG_VERSION, "3.3.3")
        self.assertEqual(DEFAULT_CONFIG_PATH.name, "mood_social_v3_3_3.yaml")
        self.assertEqual(config.module, "mood_social")
        self.assertEqual(config.version, "3.3.3")
        self.assertEqual(config.schema.request, "mood_social_infer_request_v3")
        self.assertEqual(config.schema.response, "mood_social_infer_response_v3")
        self.assertEqual(config.schema.error, "mood_social_error_v1")
        self.assertEqual(config.schema.response_module, "mood_social_attention")
        self.assertEqual(config.runtime.timezone, "Asia/Shanghai")
        self.assertEqual(config.runtime.timezone_info.key, "Asia/Shanghai")
        self.assertEqual(config.runtime.update_frequency, "daily")
        self.assertEqual(config.runtime.inference_window_days, 7)
        self.assertEqual(config.runtime.fallback_strategy, "unavailable")
        self.assertEqual(config.runtime.random_seed, 20260728)

        self.assertEqual(
            (
                config.baseline.history_lookback_calendar_days,
                config.baseline.initial_days,
                config.baseline.stable_days,
                config.baseline.max_valid_days,
            ),
            (28, 3, 7, 14),
        )
        self.assertEqual(config.baseline.center_method, "median")
        self.assertEqual(config.baseline.quantile_interpolation, "linear")
        self.assertEqual(
            (
                config.baseline.lower_quantile,
                config.baseline.upper_quantile,
                config.baseline.mad_scale_factor,
                config.baseline.quantile_scale_divisor,
            ),
            (0.10, 0.90, 1.4826, 2.563),
        )
        self.assertEqual(
            (
                config.baseline.relative_scale_fraction,
                config.baseline.absolute_scale_floor,
                config.baseline.standardized_component_divisor,
                config.baseline.relative_denominator_floor,
                config.baseline.relative_change_scale,
            ),
            (0.05, 0.05, 2.0, 0.05, 0.50),
        )
        self.assertEqual(config.baseline.deviation_aggregation, "max")
        self.assertEqual(config.baseline.abnormal_score_threshold, 0.60)
        self.assertTrue(config.baseline.exclude_current_day)
        self.assertTrue(config.baseline.exclude_imputed_values)
        self.assertTrue(
            config.baseline.exclude_abnormal_domain_days_after_initialization
        )

        self.assertEqual(
            (config.camera.daytime_start, config.camera.daytime_end),
            ("06:00", "18:00"),
        )
        self.assertEqual(config.camera.daytime_minutes, 720)
        self.assertEqual(config.camera.activity_window_seconds, 10)
        self.assertEqual(
            config.camera.active_score_weights.as_dict(),
            {
                "center_motion_score": 0.55,
                "pose_motion_score": 0.30,
                "zone_transition_score": 0.10,
                "posture_change_score": 0.05,
            },
        )
        self.assertEqual(config.camera.active_score_threshold, 0.40)
        self.assertEqual(config.camera.low_activity_score_threshold, 0.20)
        self.assertEqual(config.camera.sedentary_min_minutes, 30)
        self.assertEqual(
            (
                config.camera.walking_speed.history_lookback_calendar_days,
                config.camera.walking_speed.min_history_days,
                config.camera.walking_speed.max_valid_days,
            ),
            (28, 3, 14),
        )
        self.assertEqual(
            (
                config.camera.walking_speed.lower_quantile,
                config.camera.walking_speed.upper_quantile,
                config.camera.walking_speed.denominator_epsilon,
            ),
            (0.10, 0.90, 1e-6),
        )

        self.assertEqual(
            config.masks.layers,
            (
                "feature_mask",
                "day_mask",
                "expert_mask",
                "personal_change_mask",
            ),
        )
        self.assertFalse(config.masks.imputed_values_activate_masks)
        self.assertFalse(config.masks.mask_values_are_risk_features)

        self.assertEqual(
            config.personal_trend.domains,
            ("activity", "sleep", "social"),
        )
        self.assertEqual(config.personal_trend.slope.method, "theil_sen")
        self.assertEqual(
            config.personal_trend.slope.minimum_total_points,
            3,
        )
        self.assertTrue(
            config.personal_trend.slope.use_calendar_day_spacing
        )
        self.assertTrue(config.personal_trend.missing_day_breaks_sequence)
        self.assertEqual(
            (
                config.personal_trend.isolation_forest.stable_baseline_only,
                config.personal_trend.isolation_forest.n_estimators,
                config.personal_trend.isolation_forest.contamination,
                config.personal_trend.isolation_forest.random_seed,
                config.personal_trend.isolation_forest.minimum_common_features,
            ),
            (True, 100, 0.15, 20260728, 2),
        )
        self.assertEqual(
            (
                config.personal_trend.change_point.minimum_valid_points,
                config.personal_trend.change_point.recent_window_points,
            ),
            (6, 3),
        )

        self.assertEqual(config.fusion.method, "masked_logistic_stacking")
        self.assertEqual(config.fusion.probability_clip_epsilon, 1e-6)
        self.assertEqual(config.fusion.confidence_mode, "decay_only")
        self.assertFalse(config.fusion.independent_confidence_risk_term)
        self.assertFalse(config.fusion.mask_values_are_risk_features)
        self.assertEqual(
            config.fusion.effective_evidence.supervised_term,
            "expert_mask_times_confidence",
        )
        self.assertEqual(
            config.fusion.effective_evidence.personal_change_term,
            "personal_change_mask_times_reliability",
        )
        self.assertEqual(
            config.fusion.effective_evidence.unavailable_at_or_below,
            0.0,
        )
        self.assertFalse(
            config.fusion.effective_evidence.run_intercept_when_unavailable
        )

        self.assertEqual(config.thresholds, (0.25, 0.45, 0.65))
        self.assertEqual(config.attention.score_rounding, "half_up")
        self.assertEqual(
            (
                config.trend.max_history_results,
                config.trend.include_current_result,
                config.trend.same_model_version_only,
                config.trend.minimum_points,
                config.trend.method,
                config.trend.use_calendar_day_spacing,
                config.trend.rising_threshold_per_day,
                config.trend.falling_threshold_per_day,
            ),
            (
                6,
                True,
                True,
                3,
                "ordinary_least_squares",
                True,
                0.03,
                -0.03,
            ),
        )

        self.assertEqual(
            config.devices.source_codes,
            {"C": "camera", "S": "sleep_device", "T": "s10"},
        )
        self.assertEqual(len(config.devices.supported_combinations), 7)
        self.assertEqual(
            config.devices.supported_combinations[-1],
            ("camera", "sleep_device", "s10"),
        )
        self.assertFalse(config.devices.profile_is_device)

        self.assertEqual(config.models.version, "mood-fusion-v3.3.3")
        self.assertEqual(
            config.models.directory.relative_to(PROJECT_ROOT).as_posix(),
            "models/mental_health/mood_social/v3.3.3",
        )
        self.assertEqual(
            config.models.path_for("mood_fusion").name,
            "mood_fusion.joblib",
        )

    def test_historical_v3_3_2_config_is_retained_but_not_current(self) -> None:
        self.assertTrue(HISTORICAL_V3_3_2_CONFIG_PATH.is_file())
        historical = load_yaml(HISTORICAL_V3_3_2_CONFIG_PATH)

        self.assertEqual(historical["version"], "3.3.2")
        self.assertEqual(
            historical["schema"]["request"],
            "mood_social_infer_request_v2",
        )
        self.assertNotEqual(DEFAULT_CONFIG_PATH, HISTORICAL_V3_3_2_CONFIG_PATH)
        with self.assertRaisesRegex(ValueError, "version"):
            load_mood_social_config(HISTORICAL_V3_3_2_CONFIG_PATH)

    def test_loaded_mapping_fields_are_read_only(self) -> None:
        config = load_mood_social_config()

        with self.assertRaises(TypeError):
            config.devices.source_codes["C"] = "other"  # type: ignore[index]
        with self.assertRaises(TypeError):
            config.models.artifacts["mood_fusion"] = "other.joblib"  # type: ignore[index]

    def test_yaml_clock_and_epsilon_are_loaded_with_expected_types(self) -> None:
        values = load_yaml(DEFAULT_CONFIG_PATH)

        self.assertIsInstance(values["camera"]["daytime_start"], str)
        self.assertIsInstance(values["camera"]["daytime_end"], str)
        self.assertIsInstance(
            values["camera"]["walking_speed"]["denominator_epsilon"],
            float,
        )
        self.assertIsInstance(
            values["fusion"]["probability_clip_epsilon"],
            float,
        )
        self.assertIsInstance(values["trend"]["include_current_result"], bool)

    def test_configuration_rejects_legacy_or_drifted_values(self) -> None:
        valid = load_yaml(DEFAULT_CONFIG_PATH)
        cases: list[tuple[str, dict[str, object], str]] = []

        old_version = copy.deepcopy(valid)
        old_version["version"] = "3.3.2"
        cases.append(("old config version", old_version, "version"))

        old_schema = copy.deepcopy(valid)
        old_schema["schema"]["request"] = "mood_social_infer_request_v2"
        cases.append(("old schema", old_schema, "schema"))

        old_response = copy.deepcopy(valid)
        old_response["schema"]["response"] = "mood_social_infer_response_v2"
        cases.append(("old response", old_response, "schema"))

        wrong_error_schema = copy.deepcopy(valid)
        wrong_error_schema["schema"]["error"] = "mood_social_error_v2"
        cases.append(("error schema", wrong_error_schema, "schema"))

        wrong_response_module = copy.deepcopy(valid)
        wrong_response_module["schema"]["response_module"] = "mood_social"
        cases.append(("response module", wrong_response_module, "schema"))

        legacy_fallback = copy.deepcopy(valid)
        legacy_fallback["runtime"]["fallback_strategy"] = "rule_scorecard"
        cases.append(("legacy fallback", legacy_fallback, "runtime"))

        invalid_lookback = copy.deepcopy(valid)
        invalid_lookback["baseline"]["history_lookback_calendar_days"] = 14
        cases.append(("baseline lookback", invalid_lookback, "28/3/7/14"))

        legacy_baseline_key = copy.deepcopy(valid)
        legacy_baseline_key["baseline"]["max_window_days"] = (
            legacy_baseline_key["baseline"].pop("max_valid_days")
        )
        cases.append(("legacy baseline key", legacy_baseline_key, "keys mismatch"))

        invalid_robust_scale = copy.deepcopy(valid)
        invalid_robust_scale["baseline"]["mad_scale_factor"] = 1.0
        cases.append(("robust scale", invalid_robust_scale, "robust deviation"))

        invalid_abnormal_update = copy.deepcopy(valid)
        invalid_abnormal_update["baseline"][
            "exclude_abnormal_domain_days_after_initialization"
        ] = False
        cases.append(("abnormal update", invalid_abnormal_update, "exclude"))

        invalid_weights = copy.deepcopy(valid)
        invalid_weights["camera"]["active_score_weights"][
            "center_motion_score"
        ] = 0.50
        cases.append(("activity weights", invalid_weights, "active_score_weights"))

        invalid_daytime = copy.deepcopy(valid)
        invalid_daytime["camera"]["daytime_minutes"] = 1440
        cases.append(("daytime", invalid_daytime, "camera"))

        invalid_sedentary = copy.deepcopy(valid)
        invalid_sedentary["camera"]["sedentary_min_minutes"] = 20
        cases.append(("sedentary", invalid_sedentary, "camera"))

        invalid_walking_history = copy.deepcopy(valid)
        invalid_walking_history["camera"]["walking_speed"][
            "history_lookback_calendar_days"
        ] = 14
        cases.append(("walking history", invalid_walking_history, "walking_speed"))

        three_masks = copy.deepcopy(valid)
        three_masks["masks"]["layers"] = three_masks["masks"]["layers"][:3]
        cases.append(("three masks", three_masks, "four V3.3.3 masks"))

        imputed_activates_mask = copy.deepcopy(valid)
        imputed_activates_mask["masks"]["imputed_values_activate_masks"] = True
        cases.append(("imputed mask", imputed_activates_mask, "imputed"))

        mask_as_risk = copy.deepcopy(valid)
        mask_as_risk["masks"]["mask_values_are_risk_features"] = True
        cases.append(("mask as risk", mask_as_risk, "risk features"))

        wrong_personal_slope = copy.deepcopy(valid)
        wrong_personal_slope["personal_trend"]["slope"]["method"] = "ols"
        cases.append(("personal slope", wrong_personal_slope, "Theil-Sen"))

        invalid_isolation_forest = copy.deepcopy(valid)
        invalid_isolation_forest["personal_trend"]["isolation_forest"][
            "n_estimators"
        ] = 50
        cases.append(
            ("isolation forest", invalid_isolation_forest, "isolation_forest")
        )

        invalid_change_point = copy.deepcopy(valid)
        invalid_change_point["personal_trend"]["change_point"][
            "minimum_valid_points"
        ] = 5
        cases.append(("change point", invalid_change_point, "change_point"))

        invalid_probability_clip = copy.deepcopy(valid)
        invalid_probability_clip["fusion"]["probability_clip_epsilon"] = 0.0
        cases.append(("probability clip", invalid_probability_clip, "fusion"))

        legacy_confidence_mode = copy.deepcopy(valid)
        legacy_confidence_mode["fusion"]["confidence_mode"] = "independent_term"
        cases.append(("confidence mode", legacy_confidence_mode, "fusion"))

        independent_confidence = copy.deepcopy(valid)
        independent_confidence["fusion"]["independent_confidence_risk_term"] = True
        cases.append(("independent confidence", independent_confidence, "fusion"))

        invalid_evidence_formula = copy.deepcopy(valid)
        invalid_evidence_formula["fusion"]["effective_evidence"][
            "supervised_term"
        ] = "expert_mask_only"
        cases.append(("evidence formula", invalid_evidence_formula, "fusion"))

        invalid_evidence_gate = copy.deepcopy(valid)
        invalid_evidence_gate["fusion"]["effective_evidence"][
            "unavailable_at_or_below"
        ] = -1.0
        cases.append(("evidence threshold", invalid_evidence_gate, "fusion"))

        invalid_intercept = copy.deepcopy(valid)
        invalid_intercept["fusion"]["effective_evidence"][
            "run_intercept_when_unavailable"
        ] = True
        cases.append(("unavailable intercept", invalid_intercept, "fusion"))

        invalid_rounding = copy.deepcopy(valid)
        invalid_rounding["attention"]["score_rounding"] = "bankers"
        cases.append(("score rounding", invalid_rounding, "attention"))

        invalid_trend_history = copy.deepcopy(valid)
        invalid_trend_history["trend"]["max_history_results"] = 7
        cases.append(("trend history", invalid_trend_history, "trend"))

        invalid_trend_current = copy.deepcopy(valid)
        invalid_trend_current["trend"]["include_current_result"] = False
        cases.append(("trend current", invalid_trend_current, "trend"))

        invalid_trend_version = copy.deepcopy(valid)
        invalid_trend_version["trend"]["same_model_version_only"] = False
        cases.append(("trend version", invalid_trend_version, "trend"))

        invalid_trend_method = copy.deepcopy(valid)
        invalid_trend_method["trend"]["method"] = "theil_sen"
        cases.append(("output trend method", invalid_trend_method, "trend"))

        invalid_trend_threshold = copy.deepcopy(valid)
        invalid_trend_threshold["trend"]["rising_threshold_per_day"] = 0.04
        cases.append(("trend threshold", invalid_trend_threshold, "trend"))

        invalid_model = copy.deepcopy(valid)
        invalid_model["models"]["version"] = "mood-fusion-v3.3.2"
        cases.append(("model version", invalid_model, "models.version"))

        invalid_model_directory = copy.deepcopy(valid)
        invalid_model_directory["models"]["directory"] = (
            "models/mental_health/mood_social/v3.3.2"
        )
        cases.append(
            ("model directory", invalid_model_directory, "models.directory")
        )

        invalid_artifact = copy.deepcopy(valid)
        invalid_artifact["models"]["artifacts"]["mood_fusion"] = "fusion.joblib"
        cases.append(("model artifact", invalid_artifact, "manifest"))

        invalid_device_code = copy.deepcopy(valid)
        invalid_device_code["devices"]["source_codes"]["T"] = "telephone"
        cases.append(("device code", invalid_device_code, "source_codes"))

        missing_device_combination = copy.deepcopy(valid)
        missing_device_combination["devices"]["supported_combinations"].pop()
        cases.append(
            (
                "device combinations",
                missing_device_combination,
                "seven C/S/T combinations",
            )
        )

        invalid_seed = copy.deepcopy(valid)
        invalid_seed["runtime"]["random_seed"] = True
        cases.append(("random seed", invalid_seed, "random_seed"))

        extra_key = copy.deepcopy(valid)
        extra_key["runtime"]["passive_max_level"] = 3
        cases.append(("legacy extra key", extra_key, "keys mismatch"))

        for label, values, pattern in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, pattern):
                    mood_social_config_from_mapping(values)

    def test_configuration_root_must_be_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "root must be a mapping"):
            mood_social_config_from_mapping([])  # type: ignore[arg-type]

    def test_yaml_excludes_legacy_rule_or_v2_production_configuration(self) -> None:
        source = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")

        self.assertNotIn("rule_scorecard", source)
        self.assertNotIn("strong_rule", source)
        self.assertNotIn("passive_max_level", source)
        self.assertNotIn("mood_social_infer_request_v2", source)
        self.assertNotIn("mood_social_infer_response_v2", source)
        self.assertNotIn("mood-fusion-v3.3.2", source)
        self.assertNotIn("beta", source.lower())
        self.assertNotIn("masked_gated_mlp", source.lower())

    def test_ml_extra_declares_v3_3_3_data_and_model_dependencies(self) -> None:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
            project = tomllib.load(file)
        requirements = project["project"]["optional-dependencies"]["ml"]
        names = {item.split("==", 1)[0] for item in requirements}

        self.assertTrue(
            {
                "pandas",
                "pyarrow",
                "joblib",
                "catboost",
                "shap",
                "openpyxl",
                "xlrd",
            }.issubset(names)
        )


if __name__ == "__main__":
    unittest.main()
