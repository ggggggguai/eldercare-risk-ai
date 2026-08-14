from __future__ import annotations

import math
import unittest

import numpy as np

from elderly_monitoring.modules.mental_health.wandering.topology import (
    compute_topology,
    compute_turn_geometry,
)


class WanderingTopologyTest(unittest.TestCase):
    def test_straight_line_has_zero_curvature_reversal_and_winding(self) -> None:
        points = np.column_stack(
            (np.linspace(-1.0, 1.0, 80), np.zeros(80, dtype=np.float64))
        )
        mask = np.ones(80, dtype=np.int8)

        geometry = compute_turn_geometry(points, mask, step_epsilon=1e-8)
        topology = compute_topology(points, mask)

        self.assertLessEqual(float(np.max(geometry.abs_curvature)), 1e-7)
        self.assertEqual(topology["reversal_events"], [])
        self.assertLessEqual(topology["absolute_winding"], 1e-7)

    def test_pacing_fixture_localizes_the_prearranged_reversal(self) -> None:
        x = np.concatenate(
            (
                np.arange(41, dtype=np.float64),
                np.arange(39, 0, -1, dtype=np.float64),
            )
        )
        points = np.column_stack((x, np.zeros_like(x)))
        topology = compute_topology(points, np.ones(80, dtype=np.int8))

        self.assertEqual(
            [event["index"] for event in topology["reversal_events"]],
            [40],
        )
        self.assertAlmostEqual(topology["reversal_events"][0]["angle_deg"], 180.0)

    def test_single_closed_loop_winding_error_is_within_point_zero_five_turns(self) -> None:
        theta = np.linspace(0.0, 2.0 * math.pi, 80, dtype=np.float64)
        points = np.column_stack((np.cos(theta), np.sin(theta)))
        topology = compute_topology(points, np.ones(80, dtype=np.int8))

        self.assertLessEqual(abs(topology["absolute_winding"] - 1.0), 0.05)

    def test_revisit_count_and_selected_links_follow_fixed_order_and_endpoint_rule(self) -> None:
        points = np.column_stack(
            (
                np.concatenate((np.arange(10), np.arange(10), np.arange(10))),
                np.zeros(30),
            )
        ).astype(np.float64)
        topology = compute_topology(
            points,
            np.ones(30, dtype=np.int8),
            revisit_index_gap=8,
            revisit_radius=0.10,
            max_revisit_links=5,
        )

        self.assertGreater(topology["revisit_pair_count"], 0)
        links = topology["selected_revisit_links"]
        self.assertLessEqual(len(links), 5)
        endpoints = [index for link in links for index in (link["i"], link["j"])]
        self.assertEqual(len(endpoints), len(set(endpoints)))
        self.assertEqual(
            links,
            sorted(links, key=lambda link: (link["distance"], link["i"], link["j"])),
        )

    def test_missing_neighbors_and_zero_length_segments_have_zero_geometry(self) -> None:
        points = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]],
            dtype=np.float64,
        )
        mask = np.asarray([1, 1, 1, 0, 1], dtype=np.int8)
        geometry = compute_turn_geometry(points, mask, step_epsilon=1e-8)

        self.assertEqual(geometry.valid_displacement.tolist(), [False, True, False, False, False])
        self.assertTrue(np.all(geometry.turn_angles == 0.0))
        self.assertTrue(np.all(geometry.abs_curvature == 0.0))


if __name__ == "__main__":
    unittest.main()
