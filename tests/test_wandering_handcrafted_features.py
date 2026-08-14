from __future__ import annotations

import copy
import math
import unittest

import numpy as np

from elderly_monitoring.modules.mental_health.wandering.anchorless_normalization import (
    robust_isotropic_normalize,
)
from elderly_monitoring.modules.mental_health.wandering.handcrafted_features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    HandcraftedFeatureError,
    extract_features_from_points,
    feature_schema_document,
)


def _transform(points: np.ndarray, matrix: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ matrix.T + offset


class WanderingHandcraftedFeaturesTest(unittest.TestCase):
    def test_schema_has_the_exact_frozen_26_feature_order(self) -> None:
        self.assertEqual(FEATURE_SCHEMA_VERSION, "wandering-handcrafted-features-v1")
        self.assertEqual(
            FEATURE_NAMES,
            (
                "path_length", "net_displacement", "path_efficiency",
                "step_length_mean", "step_length_std", "step_length_max",
                "heading_resultant_length", "abs_turn_mean", "abs_turn_std",
                "abs_turn_max", "abs_turn_p90", "cumulative_abs_turn",
                "turn_direction_coherence", "absolute_winding", "reversal_count",
                "reversal_rate", "reversal_angle_mean_deg", "revisit_pair_count",
                "revisit_pair_density", "nearest_far_distance_clipped",
                "convex_hull_area", "radius_of_gyration", "spatial_anisotropy",
                "valid_point_ratio", "mean_quality", "min_quality",
            ),
        )
        schema = feature_schema_document()
        self.assertEqual(schema["schema_version"], FEATURE_SCHEMA_VERSION)
        self.assertEqual([row["name"] for row in schema["features"]], list(FEATURE_NAMES))
        forbidden = {"source_dataset", "sample_id", "input_point_count", "model_features", "normalized_dt", "time_available"}
        self.assertTrue(forbidden.isdisjoint(FEATURE_NAMES))

    def test_straight_line_formula_is_exact_and_finite(self) -> None:
        points = np.column_stack((np.arange(10, dtype=np.float64), np.zeros(10)))
        values = extract_features_from_points(points, quality=np.ones(10))
        by_name = dict(zip(FEATURE_NAMES, values, strict=True))

        self.assertEqual(values.shape, (26,))
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(by_name["path_length"], 9.0)
        self.assertAlmostEqual(by_name["net_displacement"], 9.0)
        self.assertAlmostEqual(by_name["path_efficiency"], 1.0)
        self.assertAlmostEqual(by_name["step_length_mean"], 1.0)
        self.assertAlmostEqual(by_name["step_length_std"], 0.0)
        self.assertAlmostEqual(by_name["heading_resultant_length"], 1.0)
        self.assertAlmostEqual(by_name["cumulative_abs_turn"], 0.0)
        self.assertAlmostEqual(by_name["convex_hull_area"], 0.0)
        self.assertAlmostEqual(by_name["spatial_anisotropy"], 1.0)
        self.assertAlmostEqual(by_name["valid_point_ratio"], 1.0)
        self.assertAlmostEqual(by_name["mean_quality"], 1.0)
        self.assertAlmostEqual(by_name["min_quality"], 1.0)

    def test_pacing_closed_loop_and_revisit_fixtures_cover_topology_formulas(self) -> None:
        pacing = np.column_stack(
            (
                np.asarray([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0], dtype=np.float64),
                np.zeros(17, dtype=np.float64),
            )
        )
        loop = np.asarray([[math.cos(t), math.sin(t)] for t in np.linspace(0, 2 * math.pi, 17)], dtype=np.float64)
        revisit = np.asarray([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [0, 1], [0.01, 0.01]], dtype=np.float64)

        pacing_f = dict(zip(FEATURE_NAMES, extract_features_from_points(pacing), strict=True))
        loop_f = dict(zip(FEATURE_NAMES, extract_features_from_points(loop), strict=True))
        revisit_f = dict(zip(FEATURE_NAMES, extract_features_from_points(revisit), strict=True))

        self.assertGreaterEqual(pacing_f["reversal_count"], 2)
        self.assertGreater(pacing_f["reversal_angle_mean_deg"], 170.0)
        self.assertAlmostEqual(loop_f["net_displacement"], 0.0, places=12)
        self.assertGreater(loop_f["absolute_winding"], 0.8)
        self.assertGreater(loop_f["convex_hull_area"], 2.5)
        self.assertEqual(revisit_f["revisit_pair_count"], 1.0)
        self.assertAlmostEqual(revisit_f["revisit_pair_density"], 1.0)
        self.assertLess(revisit_f["nearest_far_distance_clipped"], 0.02)

    def test_direct_extractor_is_translation_rotation_reflection_and_time_reverse_invariant(self) -> None:
        points = np.asarray([[0.0, 0.0], [0.3, 0.1], [0.5, 0.7], [-0.2, 0.9], [-0.5, 0.2], [0.1, -0.3], [0.4, 0.2], [0.0, 0.0], [0.2, 0.6]], dtype=np.float64)
        angle = 0.731
        rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        reflection = np.asarray([[-1.0, 0.0], [0.0, 1.0]])
        baseline = extract_features_from_points(points)
        variants = (
            points + np.asarray([17.0, -9.0]),
            _transform(points, rotation, np.asarray([3.0, 4.0])),
            _transform(points, reflection, np.asarray([-2.0, 8.0])),
            points[::-1],
        )
        for variant in variants:
            with self.subTest(variant=variant[0].tolist()):
                np.testing.assert_allclose(extract_features_from_points(variant), baseline, atol=1e-6, rtol=0.0)

    def test_source_to_shape_chain_is_positive_uniform_scale_invariant(self) -> None:
        source = np.asarray([[0.0, 0.0], [0.4, 0.1], [0.8, 0.6], [0.2, 1.0], [-0.3, 0.4], [0.0, 0.0], [0.7, -0.2], [0.9, 0.5], [0.1, 0.8]], dtype=np.float64)
        base_shape = robust_isotropic_normalize(source).points
        scaled_shape = robust_isotropic_normalize(source * 123.5).points
        np.testing.assert_allclose(
            extract_features_from_points(base_shape),
            extract_features_from_points(scaled_shape),
            atol=1e-6,
            rtol=0.0,
        )

    def test_nonfinite_or_malformed_ready_data_fails_without_imputation(self) -> None:
        points = np.column_stack((np.arange(9, dtype=np.float64), np.zeros(9)))
        with self.assertRaises(HandcraftedFeatureError):
            extract_features_from_points(np.asarray([[0.0, 0.0], [float("nan"), 1.0]]))
        with self.assertRaises(HandcraftedFeatureError):
            extract_features_from_points(points, quality=np.asarray([1.0] * 8 + [float("inf")]))
        with self.assertRaises(HandcraftedFeatureError):
            extract_features_from_points(points, point_mask=np.ones(8))


if __name__ == "__main__":
    unittest.main()
