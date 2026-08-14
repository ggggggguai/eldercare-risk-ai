from __future__ import annotations

import unittest

from elderly_monitoring.modules.fall_risk.sit_stand_event_evaluation import (
    evaluate_sit_stand_events,
)


CONFIG = {
    "protocol_version": "sit-stand-event-eval-v1",
    "protocol_status": "development_provisional",
    "score_threshold": 0.5,
    "iou_threshold": 0.3,
    "onset_tolerance_sec": 0.75,
    "offset_tolerance_sec": 0.75,
    "bootstrap_iterations": 20,
    "bootstrap_seed": 42,
}


def _truth(
    label_id: str,
    *,
    video_id: str = "video_1",
    transition: str | None = "sit_to_stand",
    onset: float = 1.0,
    offset: float = 3.0,
    interval_type: str = "event",
    hard_negative_type: str | None = None,
) -> dict:
    return {
        "label_id": label_id,
        "event_id": label_id,
        "video_id": video_id,
        "interval_type": interval_type,
        "transition_type": transition,
        "onset_time": onset,
        "offset_time": offset,
        "eligibility": "eligible" if interval_type != "ignore" else "ignore",
        "split_group_id": f"group_{video_id}",
        "dataset": "fixture",
        "scene_region": "home",
        "hard_negative_type": hard_negative_type,
    }


def _prediction(
    prediction_id: str,
    *,
    video_id: str = "video_1",
    transition: str = "sit_to_stand",
    onset: float = 1.0,
    offset: float = 3.0,
    score: float = 0.9,
) -> dict:
    return {
        "prediction_id": prediction_id,
        "video_id": video_id,
        "transition_type": transition,
        "onset_time": onset,
        "offset_time": offset,
        "score": score,
        "quality_state": "valid",
        "model_version": "fixture-v1",
    }


class SitStandEventEvaluationTest(unittest.TestCase):
    def test_exact_match_reports_event_direction_and_boundary_metrics(self) -> None:
        result = evaluate_sit_stand_events(
            [_truth("event_1")], [_prediction("prediction_1")], config=CONFIG
        )

        self.assertEqual(result["metrics"]["event_f1"], 1.0)
        self.assertEqual(result["metrics"]["direction_macro_f1"], 1.0)
        self.assertIn("source_group_id", result["metrics"]["stratified"])
        self.assertEqual(result["metrics"]["onset_error_sec"]["median"], 0.0)
        self.assertEqual(result["metrics"]["boundary_iou"]["median"], 1.0)
        self.assertEqual(result["metrics"]["fp_per_hour"]["status"], "unavailable")

    def test_wrong_direction_is_not_a_true_positive(self) -> None:
        result = evaluate_sit_stand_events(
            [_truth("event_1")],
            [_prediction("prediction_1", transition="stand_to_sit")],
            config=CONFIG,
        )

        self.assertEqual(result["metrics"]["true_positive"], 0)
        self.assertEqual(result["metrics"]["false_positive"], 1)
        self.assertEqual(result["metrics"]["false_negative"], 1)
        self.assertEqual(result["metrics"]["direction_macro_f1"], 0.0)

    def test_duplicate_prediction_is_a_false_positive(self) -> None:
        result = evaluate_sit_stand_events(
            [_truth("event_1")],
            [_prediction("prediction_1"), _prediction("prediction_2", onset=1.1)],
            config=CONFIG,
        )

        self.assertEqual(result["metrics"]["true_positive"], 1)
        self.assertEqual(result["metrics"]["false_positive"], 1)

    def test_cross_video_prediction_cannot_match(self) -> None:
        result = evaluate_sit_stand_events(
            [_truth("event_1")],
            [_prediction("prediction_1", video_id="video_2")],
            config=CONFIG,
        )

        self.assertEqual(result["metrics"]["true_positive"], 0)
        self.assertEqual(result["metrics"]["false_positive"], 0)
        self.assertEqual(
            result["excluded"][0]["exclusion_reason"], "outside_reviewed_intervals"
        )

    def test_unlabelled_gap_is_not_inferred_as_negative(self) -> None:
        result = evaluate_sit_stand_events(
            [
                _truth("event_1", onset=1.0, offset=2.0),
                _truth(
                    "background_1",
                    onset=4.0,
                    offset=6.0,
                    interval_type="explicit_background",
                    transition=None,
                ),
            ],
            [_prediction("prediction_1", onset=7.0, offset=8.0)],
            config=CONFIG,
        )

        self.assertEqual(result["metrics"]["false_positive"], 0)
        self.assertEqual(result["metrics"]["fp_per_hour"]["false_positive_count"], 0)
        self.assertEqual(
            result["excluded"][0]["exclusion_reason"], "outside_reviewed_intervals"
        )

    def test_ignore_overlap_excludes_prediction(self) -> None:
        result = evaluate_sit_stand_events(
            [
                _truth("event_1", video_id="video_1", onset=1.0, offset=2.0),
                _truth(
                    "ignore_1",
                    video_id="video_1",
                    onset=4.0,
                    offset=6.0,
                    interval_type="ignore",
                    transition=None,
                ),
            ],
            [_prediction("prediction_1", onset=4.5, offset=5.5)],
            config=CONFIG,
        )

        self.assertEqual(result["metrics"]["false_positive"], 0)
        self.assertEqual(len(result["excluded"]), 1)

    def test_explicit_background_enables_fp_per_hour(self) -> None:
        result = evaluate_sit_stand_events(
            [
                _truth("event_1"),
                _truth(
                    "background_1",
                    video_id="video_2",
                    onset=0.0,
                    offset=3600.0,
                    interval_type="explicit_background",
                    transition=None,
                    hard_negative_type="controlled_bend",
                ),
            ],
            [_prediction("prediction_1", video_id="video_2", onset=10.0, offset=11.0)],
            config=CONFIG,
        )

        self.assertEqual(result["metrics"]["fp_per_hour"]["status"], "available")
        self.assertEqual(result["metrics"]["fp_per_hour"]["value"], 1.0)
        self.assertEqual(
            result["metrics"]["hard_negative_false_positive_rate"]["controlled_bend"],
            1.0,
        )

    def test_bootstrap_includes_background_only_groups(self) -> None:
        config = {**CONFIG, "bootstrap_iterations": 100}
        result = evaluate_sit_stand_events(
            [
                _truth("event_1", video_id="event_video"),
                _truth(
                    "background_1",
                    video_id="background_video",
                    onset=0.0,
                    offset=10.0,
                    interval_type="explicit_background",
                    transition=None,
                ),
            ],
            [
                _prediction("event_prediction", video_id="event_video"),
                _prediction(
                    "background_prediction",
                    video_id="background_video",
                    onset=1.0,
                    offset=2.0,
                ),
            ],
            config=config,
        )

        self.assertEqual(result["confidence_intervals_95"]["group_count"], 2)

    def test_threshold_curve_rematches_instead_of_reusing_default_matches(self) -> None:
        result = evaluate_sit_stand_events(
            [_truth("event_1")],
            [
                _prediction("high_wrong", transition="stand_to_sit", score=0.9),
                _prediction("low_correct", score=0.4),
            ],
            config=CONFIG,
        )

        by_threshold = {row["threshold"]: row for row in result["threshold_curve"]}
        self.assertEqual(by_threshold[0.5]["true_positive"], 0)
        self.assertEqual(by_threshold[0.4]["true_positive"], 1)


if __name__ == "__main__":
    unittest.main()
