from __future__ import annotations

import unittest

import numpy as np

from elderly_monitoring.modules.fall_risk.sit_stand_tabular import (
    aggregate_event_scores,
    select_f1_threshold,
    split_validation_groups,
)


class SitStandTabularTest(unittest.TestCase):
    def test_validation_group_split_is_deterministic_and_disjoint(self) -> None:
        groups = np.asarray([f"group-{index}" for index in range(20) for _ in range(2)])
        labels = np.asarray([index % 2 for index in range(20) for _ in range(2)])

        selection_a, confirmation_a = split_validation_groups(groups, labels)
        selection_b, confirmation_b = split_validation_groups(groups, labels)

        np.testing.assert_array_equal(selection_a, selection_b)
        np.testing.assert_array_equal(confirmation_a, confirmation_b)
        self.assertFalse(set(groups[selection_a]) & set(groups[confirmation_a]))

    def test_event_aggregation_uses_mean_and_preserves_context(self) -> None:
        rows = aggregate_event_scores(
            [0, 1, 2],
            [0.2, 0.8, 0.1],
            labels=[1, 1, 0],
            event_ids=["positive", "positive", "negative"],
            split_group_ids=["g1", "g1", "g2"],
            datasets=["d1", "d1", "d2"],
            action_ids=["A04", "A04", "A05"],
        )

        self.assertEqual([row["event_id"] for row in rows], ["negative", "positive"])
        self.assertAlmostEqual(rows[1]["score"], 0.5)
        self.assertEqual(rows[1]["window_count"], 2)

    def test_threshold_selection_maximizes_event_f1(self) -> None:
        rows = [
            {"label": 1, "score": 0.9},
            {"label": 1, "score": 0.6},
            {"label": 0, "score": 0.7},
            {"label": 0, "score": 0.1},
        ]

        threshold, metrics = select_f1_threshold(rows)

        self.assertGreater(threshold, 0.1)
        self.assertLessEqual(threshold, 0.6)
        self.assertAlmostEqual(metrics["f1"], 0.8)


if __name__ == "__main__":
    unittest.main()
