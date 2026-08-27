from __future__ import annotations

import unittest

from scripts.prepare.prepare_fall_self_near_fall_dataset import (
    _balanced_loss_weights,
    build_candidates,
    partition_for_scene,
)

import numpy as np


class FallSelfNearFallDatasetTest(unittest.TestCase):
    def test_partition_for_scene(self) -> None:
        self.assertEqual(partition_for_scene("base"), "train")
        self.assertEqual(partition_for_scene("dark"), "validation")
        self.assertEqual(partition_for_scene("hall"), "challenge")
        self.assertIsNone(partition_for_scene("unknown"))

    def test_build_candidates_keeps_near_fall_and_reviewed_negatives(self) -> None:
        manifest = {
            "positive": {"video_id": "positive", "scene_region": "dark"},
            "negative": {"video_id": "negative", "scene_region": "base"},
            "fall": {"video_id": "fall", "scene_region": "hall"},
        }
        actions = [
            {
                "label_id": "p1",
                "video_id": "positive",
                "action_id": "C03",
                "action_name": "stumble_recovery",
                "scene": "dark",
                "start_frame": 10,
                "end_frame": 20,
            },
            {
                "label_id": "n1",
                "video_id": "negative",
                "action_id": "A03",
                "action_name": "controlled_sit_down",
                "scene": "base",
                "start_frame": 5,
                "end_frame": 15,
            },
            {
                "label_id": "f1",
                "video_id": "fall",
                "action_id": "D01",
                "action_name": "forward_fall",
                "scene": "hall",
                "start_frame": 5,
                "end_frame": 15,
            },
        ]

        candidates = build_candidates(actions, manifest)

        self.assertEqual(
            [row["candidate_role"] for row in candidates],
            ["hard_negative_candidate", "positive_candidate"],
        )
        self.assertEqual(
            [row["partition"] for row in candidates],
            ["train", "validation"],
        )

    def test_balanced_loss_weights_equalize_class_mass_per_partition(self) -> None:
        samples = [
            {"event_id": "n1", "partition": "train", "label": 0},
            {"event_id": "n2", "partition": "train", "label": 0},
            {"event_id": "p1", "partition": "train", "label": 1},
            {"event_id": "vn1", "partition": "validation", "label": 0},
            {"event_id": "vp1", "partition": "validation", "label": 1},
        ]

        weights = _balanced_loss_weights(samples, np.ones(5, dtype=np.float32))

        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2]))
        self.assertAlmostEqual(float(weights[3]), float(weights[4]))
