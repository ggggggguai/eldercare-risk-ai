from __future__ import annotations

import unittest

import numpy as np

from elderly_monitoring.modules.mental_health.wandering.anchorless_normalization import (
    AnchorlessNormalizationError,
    robust_isotropic_normalize,
)


class WanderingAnchorlessNormalizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.points = np.asarray(
            [
                [-3.0, -1.0],
                [-1.0, 0.5],
                [0.0, 2.0],
                [2.5, 1.0],
                [5.0, -2.0],
                [7.0, 3.0],
            ],
            dtype=np.float64,
        )

    def test_center_scale_and_linear_quantiles_follow_the_frozen_formula(self) -> None:
        result = robust_isotropic_normalize(
            self.points,
            quantile_low=0.05,
            quantile_high=0.95,
            quantile_method="linear",
            scale_epsilon=1e-8,
        )

        expected_center = np.median(self.points, axis=0)
        span = np.quantile(
            self.points,
            [0.05, 0.95],
            axis=0,
            method="linear",
        )
        expected_scale = float(np.linalg.norm(span[1] - span[0]))
        np.testing.assert_array_equal(result.center, expected_center)
        self.assertEqual(result.scale, expected_scale)
        np.testing.assert_array_equal(
            result.points,
            (self.points - expected_center) / expected_scale,
        )
        self.assertEqual(result.points.dtype, np.float64)

    def test_translation_and_positive_uniform_scale_are_invariant(self) -> None:
        baseline = robust_isotropic_normalize(self.points).points
        translated = robust_isotropic_normalize(
            self.points + np.asarray([123.0, -87.5])
        ).points
        uniformly_scaled = robust_isotropic_normalize(self.points * 7.25).points

        self.assertLessEqual(float(np.max(np.abs(baseline - translated))), 1e-6)
        self.assertLessEqual(
            float(np.max(np.abs(baseline - uniformly_scaled))),
            1e-6,
        )

    def test_independent_axis_stretch_is_not_erased(self) -> None:
        baseline = robust_isotropic_normalize(self.points).points
        stretched_points = self.points.copy()
        stretched_points[:, 0] *= 2.0
        stretched = robust_isotropic_normalize(stretched_points).points

        self.assertGreaterEqual(float(np.max(np.abs(baseline - stretched))), 1e-3)

    def test_degenerate_or_non_finite_input_fails_closed(self) -> None:
        fixtures = (
            np.zeros((8, 2), dtype=np.float64),
            np.full((8, 2), [2.0, -3.0], dtype=np.float64),
            np.asarray([[0.0, 0.0], [np.nan, 1.0]], dtype=np.float64),
        )
        for points in fixtures:
            with self.subTest(points=points.tolist()):
                with self.assertRaises(AnchorlessNormalizationError):
                    robust_isotropic_normalize(points, scale_epsilon=1e-8)


if __name__ == "__main__":
    unittest.main()
