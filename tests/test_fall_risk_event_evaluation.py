from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import elderly_monitoring.modules.fall_risk.evaluation as evaluation_module
import scripts.evaluate.evaluate_fall_events as evaluation_cli_module
from elderly_monitoring.modules.fall_risk.evaluation import (
    evaluate_event_predictions,
    write_evaluation_bundle,
)
from scripts.evaluate.evaluate_fall_events import (
    _evaluation_implementation_sha256,
    _implementation_files_sha256,
    _load_jsonl_snapshot,
    _load_yaml_mapping_snapshot,
    _validate_split_metadata,
    _validate_split_root,
    _validate_test_partition_access,
    _validate_validation_report_binding,
    main as evaluation_main,
)
from scripts.evaluate.build_synthetic_fall_event_fixture import build_fixture


def _config(**overrides):
    config = {
        "protocol_version": "fall-event-eval-v1",
        "protocol_status": "development_provisional",
        "score_threshold": 0.5,
        "iou_threshold": 0.5,
        "onset_tolerance_sec": 0.25,
        "search_window_before_sec": 1.0,
        "search_window_after_sec": 1.0,
        "matching_operator": "iou_or_onset",
        "one_to_one_assignment": "maximum_cardinality_deterministic",
        "ground_truth_event_types": ["fall", "near_fall", "recovery"],
        "pr_auc_method": "average_precision_step",
        "reset_rule_application": "audit_only_postprocessed_predictions",
        "event_merge_enabled": False,
        "event_merge_gap_sec": 0.5,
        "event_reset_gap_sec": 2.0,
        "duplicate_alert_policy": "unmatched_is_false_positive",
        "event_type_compatibility": {
            "fall": ["fall"],
            "near_fall": ["near_fall"],
            "recovery": ["recovery"],
        },
        "bootstrap_iterations": 50,
        "bootstrap_seed": 17,
        "minimum_clusters_for_inference": 5,
    }
    config.update(overrides)
    return config


def _truth(
    label_id="gt-1",
    *,
    event_type="fall",
    task_type="fall_event",
    start_time=10.0,
    end_time=12.0,
    video_id="video-1",
    source_group_id="group-1",
    **extra,
):
    row = {
        "label_id": label_id,
        "video_id": video_id,
        "task_type": task_type,
        "event_type": event_type,
        "start_time": start_time,
        "end_time": end_time,
        "onset_time": start_time,
        "review_status": "final",
        "eligibility": True,
        "source_group_id": source_group_id,
        "subject_id": "unknown",
        "split_id": "split-1",
    }
    row.update(extra)
    return row


def _prediction(
    prediction_id="pred-1",
    *,
    event_type="fall",
    task_type="fall_event",
    start_time=10.0,
    end_time=12.0,
    onset_time=10.1,
    score=0.9,
    video_id="video-1",
    **extra,
):
    row = {
        "prediction_id": prediction_id,
        "video_id": video_id,
        "task_type": task_type,
        "event_type": event_type,
        "score": score,
        "start_time": start_time,
        "end_time": end_time,
        "onset_time": onset_time,
        "status": "emitted",
        "quality_state": "usable",
        "model_version": "test-model-v1",
        "config_hash": "a" * 64,
        "split_id": "split-1",
    }
    row.update(extra)
    return row


class FallRiskEventEvaluationTest(unittest.TestCase):
    def test_synthetic_fixture_builder_is_complete_and_no_overwrite(self) -> None:
        repo = Path(__file__).parents[1]
        config_path = repo / "configs/evaluation/fall_event_v1.provisional.yaml"
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "fixture"

            result = build_fixture(output, config_path)

            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "assignments.jsonl",
                    "ground_truth.jsonl",
                    "manifest.jsonl",
                    "predictions.jsonl",
                    "split.json",
                },
            )
            split = json.loads((output / "split.json").read_text(encoding="utf-8"))
            self.assertEqual(split["split_id"], result["split_id"])
            with self.assertRaises(FileExistsError):
                build_fixture(output, config_path)

    def test_perfect_match_reports_event_metrics_and_latency(self) -> None:
        result = evaluate_event_predictions(
            [_truth()],
            [_prediction()],
            manifest=[],
            config=_config(),
        )

        self.assertEqual(result.metrics["tp"], 1)
        self.assertEqual(result.metrics["fp"], 0)
        self.assertEqual(result.metrics["fn"], 0)
        self.assertEqual(result.metrics["precision"], 1.0)
        self.assertEqual(result.metrics["recall"], 1.0)
        self.assertEqual(result.metrics["f1"], 1.0)
        self.assertEqual(result.metrics["event_pr_auc"], 1.0)
        self.assertEqual(result.metrics["mean_boundary_iou"], 1.0)
        self.assertAlmostEqual(result.metrics["mean_detection_latency_sec"], 0.1)

    def test_no_predictions_and_no_truth_use_null_for_undefined_rates(self) -> None:
        missed = evaluate_event_predictions(
            [_truth()], [], manifest=[], config=_config()
        )
        empty = evaluate_event_predictions([], [], manifest=[], config=_config())

        self.assertEqual(missed.metrics["fn"], 1)
        self.assertIsNone(missed.metrics["precision"])
        self.assertEqual(missed.metrics["recall"], 0.0)
        self.assertEqual(missed.metrics["event_pr_auc"], 0.0)
        self.assertIsNone(empty.metrics["precision"])
        self.assertIsNone(empty.metrics["recall"])
        self.assertIsNone(empty.metrics["event_pr_auc"])

    def test_prediction_without_truth_is_false_positive(self) -> None:
        result = evaluate_event_predictions(
            [], [_prediction()], manifest=[], config=_config()
        )

        self.assertEqual(result.metrics["tp"], 0)
        self.assertEqual(result.metrics["fp"], 1)
        self.assertEqual(result.metrics["precision"], 0.0)
        self.assertIsNone(result.metrics["recall"])

    def test_duplicate_alert_and_many_to_one_are_counted_as_false_positives(self) -> None:
        predictions = [
            _prediction("pred-best", score=0.9),
            _prediction("pred-duplicate", score=0.8, start_time=10.1, end_time=11.9),
        ]

        result = evaluate_event_predictions(
            [_truth()], predictions, manifest=[], config=_config()
        )

        self.assertEqual(result.metrics["tp"], 1)
        self.assertEqual(result.metrics["fp"], 1)
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches[0]["prediction_id"], "pred-best")
        self.assertEqual(
            {row["prediction_id"] for row in result.false_positives},
            {"pred-duplicate"},
        )

    def test_one_prediction_cannot_match_two_truth_events(self) -> None:
        truths = [
            _truth("gt-1", start_time=10.0, end_time=11.0),
            _truth("gt-2", start_time=11.0, end_time=12.0),
        ]
        prediction = _prediction(start_time=10.0, end_time=12.0, onset_time=10.0)

        result = evaluate_event_predictions(
            truths,
            [prediction],
            manifest=[],
            config=_config(iou_threshold=0.5),
        )

        self.assertEqual(result.metrics["tp"], 1)
        self.assertEqual(result.metrics["fn"], 1)

    def test_matching_maximizes_valid_one_to_one_pairs(self) -> None:
        truths = [
            _truth("gt-flexible", start_time=0.0, end_time=10.0),
            _truth("gt-constrained", start_time=0.0, end_time=4.0),
        ]
        predictions = [
            _prediction("pred-shared", start_time=0.0, end_time=10.0, onset_time=0.0),
            _prediction("pred-flexible-only", start_time=6.0, end_time=10.0, onset_time=6.0),
        ]

        result = evaluate_event_predictions(
            truths,
            predictions,
            manifest=[],
            config=_config(
                iou_threshold=0.4,
                onset_tolerance_sec=0.0,
                search_window_before_sec=0.0,
                search_window_after_sec=0.0,
            ),
        )

        self.assertEqual(result.metrics["tp"], 2)
        self.assertEqual(result.metrics["fp"], 0)
        self.assertEqual(result.metrics["fn"], 0)

    def test_iou_and_onset_tolerance_boundaries_are_inclusive(self) -> None:
        iou_boundary = _prediction(start_time=10.0, end_time=11.0, onset_time=10.8)
        onset_boundary = _prediction(
            "pred-onset",
            start_time=9.75,
            end_time=9.9,
            onset_time=9.75,
        )
        outside = _prediction(
            "pred-outside",
            start_time=9.749,
            end_time=9.9,
            onset_time=9.749,
        )

        by_iou = evaluate_event_predictions(
            [_truth()], [iou_boundary], manifest=[], config=_config()
        )
        by_onset = evaluate_event_predictions(
            [_truth()], [onset_boundary], manifest=[], config=_config()
        )
        rejected = evaluate_event_predictions(
            [_truth()], [outside], manifest=[], config=_config()
        )

        self.assertEqual(by_iou.metrics["tp"], 1)
        self.assertEqual(by_onset.metrics["tp"], 1)
        self.assertEqual(rejected.metrics["tp"], 0)

    def test_event_onset_must_be_inside_its_declared_interval(self) -> None:
        malformed = _prediction(start_time=100.0, end_time=101.0, onset_time=10.0)

        with self.assertRaisesRegex(ValueError, "onset_time"):
            evaluate_event_predictions(
                [_truth()], [malformed], manifest=[], config=_config()
            )

    def test_cross_task_and_incompatible_event_types_do_not_match(self) -> None:
        cross_task = _prediction("pred-cross-task", task_type="near_fall_event")
        cross_type = _prediction("pred-cross-type", event_type="near_fall")

        result = evaluate_event_predictions(
            [_truth()], [cross_task, cross_type], manifest=[], config=_config()
        )

        self.assertEqual(result.metrics["tp"], 0)
        self.assertEqual(result.metrics["fp"], 2)
        self.assertEqual(result.metrics["fn"], 1)

    def test_uncertain_and_pending_truth_are_excluded(self) -> None:
        truths = [
            _truth("uncertain", event_type="uncertain", exclusion_reason="occluded"),
            _truth("pending", review_status="pending"),
        ]

        result = evaluate_event_predictions(
            truths, [], manifest=[], config=_config()
        )

        self.assertEqual(result.metrics["ground_truth_count"], 0)
        self.assertEqual(len(result.excluded_samples), 2)
        self.assertEqual(
            {row["exclusion_reason"] for row in result.excluded_samples},
            {"uncertain_ground_truth", "ineligible_review_status"},
        )

    def test_ground_truth_requires_explicit_true_eligibility(self) -> None:
        missing_eligibility = _truth("missing-eligibility")
        missing_eligibility.pop("eligibility")
        truths = [
            _truth("eligible"),
            missing_eligibility,
            _truth("ineligible", eligibility=False),
        ]

        result = evaluate_event_predictions(
            truths, [], manifest=[], config=_config()
        )

        self.assertEqual(result.metrics["ground_truth_count"], 1)
        self.assertEqual(result.metrics["fn"], 1)
        self.assertEqual(
            {row["label_id"] for row in result.excluded_samples},
            {"missing-eligibility", "ineligible"},
        )
        self.assertEqual(
            {row["exclusion_reason"] for row in result.excluded_samples},
            {"ineligible_ground_truth"},
        )

    def test_lead_time_uses_reference_start_minus_alert_time(self) -> None:
        truths = [
            _truth("gt-early", video_id="early", reference_event_start=20.0),
            _truth("gt-late", video_id="late", reference_event_start=8.0),
        ]
        predictions = [
            _prediction("pred-early", video_id="early", onset_time=10.0, risk_level=3),
            _prediction("pred-late", video_id="late", onset_time=10.0, risk_level=4),
        ]

        result = evaluate_event_predictions(
            truths, predictions, manifest=[], config=_config()
        )

        self.assertEqual(sorted(result.metrics["lead_times_sec"]), [-2.0, 10.0])

    def test_lead_time_uses_first_legal_level_three_alert(self) -> None:
        truth = _truth(reference_event_start=20.0)
        predictions = [
            _prediction(
                "early",
                start_time=9.5,
                end_time=10.5,
                onset_time=10.0,
                alert_time=5.0,
                risk_level=3,
            ),
            _prediction(
                "late",
                start_time=10.0,
                end_time=12.0,
                onset_time=10.0,
                alert_time=15.0,
                risk_level=4,
            ),
        ]

        result = evaluate_event_predictions(
            [truth],
            predictions,
            manifest=[],
            config=_config(iou_threshold=0.1),
        )

        self.assertEqual(result.matches[0]["prediction_id"], "early")
        self.assertEqual(result.matches[0]["lead_time_sec"], 15.0)
        self.assertEqual(result.matches[0]["first_level_3_alert_time"], 5.0)

    def test_latency_prefers_alert_time_over_predicted_event_onset(self) -> None:
        result = evaluate_event_predictions(
            [_truth()],
            [_prediction(onset_time=10.0, alert_time=10.75)],
            manifest=[],
            config=_config(),
        )

        self.assertEqual(result.metrics["mean_detection_latency_sec"], 0.75)

    def test_event_specific_latency_metrics_and_bootstrap_do_not_mix_types(self) -> None:
        truths = [
            _truth("fall", video_id="fall-video", source_group_id="fall-group"),
            _truth(
                "near-fall",
                event_type="near_fall",
                video_id="near-fall-video",
                source_group_id="near-fall-group",
                start_time=20.0,
                end_time=22.0,
            ),
            _truth(
                "recovery",
                event_type="recovery",
                video_id="recovery-video",
                source_group_id="recovery-group",
                start_time=30.0,
                end_time=32.0,
            ),
        ]
        predictions = [
            _prediction(
                "fall-prediction",
                video_id="fall-video",
                onset_time=10.0,
                alert_time=11.0,
            ),
            _prediction(
                "near-fall-prediction",
                event_type="near_fall",
                video_id="near-fall-video",
                start_time=20.0,
                end_time=22.0,
                onset_time=20.0,
                alert_time=23.0,
            ),
            _prediction(
                "recovery-prediction",
                event_type="recovery",
                video_id="recovery-video",
                start_time=30.0,
                end_time=32.0,
                onset_time=30.0,
                alert_time=39.0,
            ),
        ]

        result = evaluate_event_predictions(
            truths, predictions, manifest=[], config=_config()
        )

        self.assertEqual(result.metrics["tp"], 3)
        self.assertEqual(result.metrics["mean_detection_latency_sec"], 1.0)
        self.assertEqual(result.metrics["mean_onset_detection_latency_sec"], 3.0)
        self.assertEqual(result.metrics["recovery_recall"], 1.0)
        self.assertEqual(result.metrics["erroneous_recovery_rate"], 0.0)
        self.assertEqual(
            result.confidence_intervals["mean_detection_latency_sec"],
            {"lower": 1.0, "upper": 1.0},
        )
        self.assertEqual(
            result.confidence_intervals["mean_onset_detection_latency_sec"],
            {"lower": 3.0, "upper": 3.0},
        )
        self.assertEqual(
            result.confidence_intervals["recovery_recall"],
            {"lower": 1.0, "upper": 1.0},
        )

    def test_frozen_protocol_requires_explicit_alert_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "alert_time"):
            evaluate_event_predictions(
                [_truth()],
                [_prediction()],
                manifest=[],
                config=_config(
                    protocol_status="frozen", bootstrap_iterations=10_000
                ),
            )

    def test_alert_time_and_risk_level_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "alert_time must be non-negative"):
            evaluate_event_predictions(
                [_truth()],
                [_prediction(alert_time=-1_000_000.0, risk_level=3)],
                manifest=[],
                config=_config(),
            )
        with self.assertRaisesRegex(ValueError, "risk_level must be an integer"):
            evaluate_event_predictions(
                [_truth()],
                [_prediction(alert_time=10.0, risk_level=3.9)],
                manifest=[],
                config=_config(),
            )

    def test_frozen_protocol_requires_formal_bootstrap_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "10,000"):
            evaluate_event_predictions(
                [_truth()],
                [_prediction(alert_time=10.1)],
                manifest=[],
                config=_config(
                    protocol_status="frozen",
                    bootstrap_iterations=9999,
                ),
            )

    def test_event_merge_rule_is_applied_from_config(self) -> None:
        predictions = [
            _prediction("pred-a", start_time=10.0, end_time=10.8, onset_time=10.0),
            _prediction("pred-b", start_time=10.9, end_time=12.0, onset_time=10.9),
        ]

        result = evaluate_event_predictions(
            [_truth()],
            predictions,
            manifest=[],
            config=_config(event_merge_enabled=True, event_merge_gap_sec=0.2),
        )

        self.assertEqual(result.metrics["tp"], 1)
        self.assertEqual(result.metrics["fp"], 0)
        self.assertEqual(result.metrics["prediction_count"], 1)
        self.assertTrue(result.matches[0]["prediction_id"].startswith("merged_"))

    def test_reset_gap_rule_marks_repeated_events_from_config(self) -> None:
        predictions = [
            _prediction("pred-a", start_time=10.0, end_time=11.0),
            _prediction("pred-b", start_time=12.0, end_time=13.0, onset_time=12.0),
        ]

        result = evaluate_event_predictions(
            [_truth(end_time=11.0)],
            predictions,
            manifest=[],
            config=_config(event_reset_gap_sec=2.0),
        )

        self.assertEqual(result.metrics["reset_gap_violation_count"], 1)
        self.assertTrue(result.false_positives[0]["reset_gap_violation"])

    def test_matching_is_independent_of_input_order(self) -> None:
        predictions = [
            _prediction("pred-best", score=0.9),
            _prediction("pred-other", score=0.8, start_time=10.1, end_time=11.9),
        ]

        forward = evaluate_event_predictions(
            [_truth()], predictions, manifest=[], config=_config()
        )
        reverse = evaluate_event_predictions(
            [_truth()], list(reversed(predictions)), manifest=[], config=_config()
        )

        self.assertEqual(forward.matches, reverse.matches)
        self.assertEqual(forward.false_positives, reverse.false_positives)

    def test_fp_per_camera_hour_uses_only_explicit_continuous_duration(self) -> None:
        manifest = [
            {
                "asset_id": "video-1",
                "duration_sec": 1800.0,
                "continuous_monitoring_eligible": True,
                "eligibility": True,
            },
            {
                "asset_id": "ignored-short-clip",
                "duration_sec": 1800.0,
                "continuous_monitoring_eligible": False,
                "eligibility": True,
            },
        ]
        predictions = [
            _prediction("fp-1", video_id="video-1"),
            _prediction(
                "fp-2",
                video_id="video-1",
                start_time=20.0,
                end_time=21.0,
                onset_time=20.0,
            ),
        ]

        result = evaluate_event_predictions(
            [], predictions, manifest=manifest, config=_config()
        )

        self.assertEqual(result.metrics["continuous_camera_hours"], 0.5)
        self.assertEqual(result.metrics["fp_per_camera_hour"], 4.0)

    def test_fp_per_household_day_uses_household_scoped_continuous_assets(self) -> None:
        manifest = [
            {
                "asset_id": "video-1",
                "eligibility": True,
                "continuous_monitoring_eligible": True,
                "duration_sec": 3600.0,
                "household_id": "household-1",
                "household_monitoring_period_id": "period-1",
                "continuous_household_days": 0.5,
            }
        ]

        result = evaluate_event_predictions(
            [], [_prediction()], manifest=manifest, config=_config()
        )

        self.assertEqual(result.metrics["continuous_household_days"], 0.5)
        self.assertEqual(result.metrics["household_false_positive_count"], 1)
        self.assertEqual(result.metrics["fp_per_household_day"], 2.0)

    def test_short_clip_cannot_create_household_day_or_traditional_fpr(self) -> None:
        manifest = [
            {
                "asset_id": "video-1",
                "eligibility": True,
                "continuous_monitoring_eligible": False,
                "duration_sec": 10.0,
                "continuous_household_days": 1.0,
                "negative_observation_count": 1,
            }
        ]

        result = evaluate_event_predictions(
            [],
            [
                _prediction("fp-1"),
                _prediction(
                    "fp-2", start_time=20.0, end_time=21.0, onset_time=20.0
                ),
            ],
            manifest=manifest,
            config=_config(),
        )

        self.assertIsNone(result.metrics["continuous_household_days"])
        self.assertIsNone(result.metrics["fp_per_household_day"])
        self.assertIsNone(result.metrics["traditional_fpr"])
        self.assertEqual(
            result.metrics["traditional_fpr_unavailable_reason"],
            "registered_true_negative_observation_units_required",
        )

    def test_bootstrap_is_deterministic_and_marked_exploratory_when_small(self) -> None:
        truths = [
            _truth("gt-1", video_id="v1", source_group_id="g1"),
            _truth("gt-2", video_id="v2", source_group_id="g2"),
        ]
        predictions = [_prediction("pred-1", video_id="v1")]

        first = evaluate_event_predictions(
            truths, predictions, manifest=[], config=_config()
        )
        second = evaluate_event_predictions(
            truths, predictions, manifest=[], config=_config()
        )

        self.assertEqual(first.confidence_intervals, second.confidence_intervals)
        self.assertTrue(first.confidence_intervals["exploratory"])
        self.assertEqual(first.confidence_intervals["cluster_count"], 2)
        self.assertIsNotNone(first.confidence_intervals["event_pr_auc"])
        self.assertIsNotNone(first.confidence_intervals["mean_boundary_iou"])
        self.assertIsNotNone(
            first.confidence_intervals["mean_detection_latency_sec"]
        )

    def test_known_truth_subject_overrides_manifest_source_group_for_bootstrap(self) -> None:
        truth = _truth(subject_id="subject-1", source_group_id="source-1")
        predictions = [
            _prediction("matched"),
            _prediction("duplicate", start_time=10.1, end_time=11.9),
        ]
        manifest = [
            {
                "asset_id": "video-1",
                "video_id": "video-1",
                "eligibility": True,
                "subject_id": "unknown",
                "source_group_id": "source-1",
            }
        ]

        result = evaluate_event_predictions(
            [truth], predictions, manifest=manifest, config=_config()
        )

        self.assertEqual(result.confidence_intervals["cluster_count"], 1)
        self.assertEqual(result.matches[0]["cluster_id"], "subject:subject-1")
        self.assertEqual(
            result.false_positives[0]["cluster_id"], "subject:subject-1"
        )

    def test_split_id_mismatch_fails_closed(self) -> None:
        prediction = _prediction(split_id="different-split")

        with self.assertRaisesRegex(ValueError, "split_id"):
            evaluate_event_predictions(
                [_truth()], [prediction], manifest=[], config=_config()
            )

    def test_bundle_contains_required_evidence_files(self) -> None:
        result = evaluate_event_predictions(
            [_truth()], [_prediction()], manifest=[], config=_config()
        )
        metadata = {
            "code_version": "dirty-test-tree",
            "manifest_hash": "b" * 64,
            "split_id": "split-1",
            "label_version": "labels-v1",
            "config_hash": "c" * 64,
            "prediction_hash": "d" * 64,
            "reproduction_command": "conda run -n eldercare-ai python scripts/evaluate/evaluate_fall_events.py ...",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "evaluation"
            write_evaluation_bundle(result, output_dir, metadata=metadata)
            names = {path.name for path in output_dir.iterdir()}
            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(
            names,
            {
                "metrics.json",
                "matches.jsonl",
                "false_positives.jsonl",
                "false_negatives.jsonl",
                "excluded_samples.jsonl",
                "threshold_curve.csv",
                "report.md",
            },
        )
        self.assertEqual(metrics["protocol_status"], "development_provisional")
        self.assertEqual(metrics["reproducibility"]["split_id"], "split-1")

    def test_bundle_failure_leaves_no_partial_directory(self) -> None:
        result = evaluate_event_predictions(
            [_truth()], [_prediction()], manifest=[], config=_config()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir)
            output_dir = parent / "evaluation"

            with self.assertRaises(TypeError):
                write_evaluation_bundle(
                    result,
                    output_dir,
                    metadata={"not_json_serializable": object()},
                )

            self.assertFalse(output_dir.exists())
            self.assertEqual(list(parent.iterdir()), [])

    def test_bundle_no_overwrite_has_one_concurrent_winner(self) -> None:
        result = evaluate_event_predictions(
            [_truth()], [_prediction()], manifest=[], config=_config()
        )
        barrier = threading.Barrier(2)
        original_write_contents = evaluation_module._write_bundle_contents

        def synchronized_write_contents(*args, **kwargs):
            original_write_contents(*args, **kwargs)
            barrier.wait(timeout=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir)
            output_dir = parent / "evaluation"
            with mock.patch.object(
                evaluation_module,
                "_write_bundle_contents",
                side_effect=synchronized_write_contents,
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(
                            write_evaluation_bundle,
                            result,
                            output_dir,
                            metadata={"writer": writer},
                        )
                        for writer in ("first", "second")
                    ]
                    outcomes = []
                    for future in futures:
                        try:
                            future.result(timeout=10)
                        except BaseException as exc:  # capture the losing commit
                            outcomes.append(exc)
                        else:
                            outcomes.append(None)

            self.assertEqual(sum(value is None for value in outcomes), 1)
            self.assertEqual(
                sum(isinstance(value, FileExistsError) for value in outcomes), 1
            )
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "metrics.json",
                    "matches.jsonl",
                    "false_positives.jsonl",
                    "false_negatives.jsonl",
                    "excluded_samples.jsonl",
                    "threshold_curve.csv",
                    "report.md",
                },
            )
            self.assertEqual(
                [path for path in parent.iterdir() if path != output_dir], []
            )

    def test_bundle_overwrite_commit_failure_restores_previous_bundle(self) -> None:
        result = evaluate_event_predictions(
            [_truth()], [_prediction()], manifest=[], config=_config()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir)
            output_dir = parent / "evaluation"
            write_evaluation_bundle(
                result, output_dir, metadata={"generation": "original"}
            )
            original_metrics = (output_dir / "metrics.json").read_bytes()
            real_replace = os.replace

            def fail_staging_commit(source, destination):
                source_path = Path(source)
                if (
                    source_path.is_dir()
                    and source_path.name.startswith(".evaluation.staging.")
                    and Path(destination) == output_dir
                ):
                    raise OSError("simulated commit failure")
                return real_replace(source, destination)

            with mock.patch.object(
                evaluation_module.os, "replace", side_effect=fail_staging_commit
            ):
                with self.assertRaisesRegex(OSError, "simulated commit failure"):
                    write_evaluation_bundle(
                        result,
                        output_dir,
                        metadata={"generation": "replacement"},
                        overwrite=True,
                    )

            self.assertEqual(
                (output_dir / "metrics.json").read_bytes(), original_metrics
            )
            write_evaluation_bundle(
                result,
                output_dir,
                metadata={"generation": "replacement"},
                overwrite=True,
            )
            replacement_metrics = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                replacement_metrics["reproducibility"]["generation"],
                "replacement",
            )
            self.assertEqual(
                [path for path in parent.iterdir() if path != output_dir], []
            )

    def test_input_snapshot_loaders_reject_duplicate_keys_and_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            duplicate_jsonl = root / "duplicate.jsonl"
            duplicate_yaml = root / "duplicate.yaml"
            invalid_utf8 = root / "invalid.jsonl"
            duplicate_jsonl.write_bytes(b'{"video_id":"a","video_id":"b"}\n')
            duplicate_yaml.write_bytes(b"protocol_status: frozen\nprotocol_status: provisional\n")
            invalid_utf8.write_bytes(b'{"video_id":"\xff"}\n')

            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                _load_jsonl_snapshot(duplicate_jsonl)
            with self.assertRaisesRegex(ValueError, "duplicate YAML key"):
                _load_yaml_mapping_snapshot(duplicate_yaml)
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                _load_jsonl_snapshot(invalid_utf8)

    def test_evaluation_implementation_hash_binds_relative_paths_and_bytes(
        self,
    ) -> None:
        repo = Path(__file__).parents[1]
        files = {
            "scripts/evaluate/evaluate_fall_events.py": (
                repo / "scripts/evaluate/evaluate_fall_events.py"
            ).read_bytes(),
            "src/elderly_monitoring/modules/fall_risk/evaluation.py": (
                repo / "src/elderly_monitoring/modules/fall_risk/evaluation.py"
            ).read_bytes(),
        }

        self.assertEqual(
            _evaluation_implementation_sha256(),
            _implementation_files_sha256(dict(reversed(list(files.items())))),
        )
        changed_path = {
            **files,
            "renamed/evaluation.py": files[
                "src/elderly_monitoring/modules/fall_risk/evaluation.py"
            ],
        }
        changed_path.pop("src/elderly_monitoring/modules/fall_risk/evaluation.py")
        changed_content = {
            **files,
            "src/elderly_monitoring/modules/fall_risk/evaluation.py": (
                files["src/elderly_monitoring/modules/fall_risk/evaluation.py"]
                + b"\n"
            ),
        }
        self.assertNotEqual(
            _implementation_files_sha256(files),
            _implementation_files_sha256(changed_path),
        )
        self.assertNotEqual(
            _implementation_files_sha256(files),
            _implementation_files_sha256(changed_content),
        )

    def test_frozen_evaluation_requires_frozen_protocol_metadata_and_validation_hash(
        self,
    ) -> None:
        config = _config(
            protocol_status="frozen",
            bootstrap_iterations=10_000,
            task_type="fall_event",
        )
        split_sha256 = "a" * 64
        split_metadata = {
            "status": "frozen",
            "protocol_status": "frozen",
            "split_name": "fall_event_v1",
            "split_id": f"fall_event_v1:sha256:{split_sha256}",
            "split_sha256": split_sha256,
            "task_type": "fall_event",
            "validation_report_sha256": "b" * 64,
        }

        self.assertEqual(
            _validate_split_metadata(split_metadata, config),
            split_metadata["split_id"],
        )
        for field, invalid_value in (
            ("protocol_status", "provisional"),
            ("validation_report_sha256", None),
            ("validation_report_sha256", "not-a-sha256"),
        ):
            invalid = {**split_metadata, field: invalid_value}
            with self.subTest(field=field, invalid_value=invalid_value):
                with self.assertRaises(ValueError):
                    _validate_split_metadata(invalid, config)

    def test_split_root_binds_validation_report_hash(self) -> None:
        assignment = {
            "sample_id": "video-1",
            "asset_id": "video-1",
            "video_id": "video-1",
            "partition": "validation",
            "task_type": "fall_event",
        }
        payload = {
            "schema_version": "1.0",
            "split_name": "fall_event_v1",
            "task_type": "fall_event",
            "protocol_status": "frozen",
            "status": "frozen",
            "manifest_sha256": "a" * 64,
            "manifest_canonical_sha256": "b" * 64,
            "validation_report_sha256": "c" * 64,
            "labels_sha256": "d" * 64,
            "config_sha256": "e" * 64,
            "assignments": [assignment],
        }
        split_sha256 = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        split_metadata = {
            key: value for key, value in payload.items() if key != "assignments"
        }
        split_metadata["split_sha256"] = split_sha256
        assignment_with_id = {
            **assignment,
            "split_id": f"fall_event_v1:sha256:{split_sha256}",
        }

        _validate_split_root(split_metadata, [assignment_with_id])
        with self.assertRaisesRegex(ValueError, "split root"):
            _validate_split_root(
                {**split_metadata, "validation_report_sha256": "f" * 64},
                [assignment_with_id],
            )

    def test_cli_runs_a_partition_scoped_provisional_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            truth_path = root / "truth.jsonl"
            predictions_path = root / "predictions.jsonl"
            manifest_path = root / "manifest.jsonl"
            split_path = root / "split.json"
            assignments_path = root / "assignments.jsonl"
            validation_report_path = root / "validation-report.json"
            output_dir = root / "output"
            truth_row = _truth()
            truth_row.pop("split_id")
            truth_path.write_text(json.dumps(truth_row) + "\n", encoding="utf-8")
            manifest_row = {
                "asset_id": "video-1",
                "video_id": "video-1",
                "eligibility": True,
                "duration_sec": 10.0,
                "continuous_monitoring_eligible": False,
            }
            manifest_path.write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")
            base_assignment = {
                "sample_id": "video-1",
                "asset_id": "video-1",
                "video_id": "video-1",
                "partition": "validation",
                "task_type": "fall_event",
            }
            manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            manifest_canonical_hash = hashlib.sha256(
                (
                    json.dumps(
                        manifest_row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            ).hexdigest()
            labels_hash = hashlib.sha256(
                (
                    json.dumps(
                        truth_row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            ).hexdigest()
            validation_report_path.write_text(
                json.dumps(
                    {
                        "schema_version": "fall-risk-label-validation-report-v1",
                        "mode": "audit",
                        "valid": True,
                    }
                ),
                encoding="utf-8",
            )
            validation_report_hash = hashlib.sha256(
                validation_report_path.read_bytes()
            ).hexdigest()
            config_sha256 = "c" * 64
            split_payload = {
                "schema_version": "1.0",
                "split_name": "fall_event_v1",
                "task_type": "fall_event",
                "protocol_status": "provisional",
                "status": "ready",
                "manifest_sha256": manifest_hash,
                "manifest_canonical_sha256": manifest_canonical_hash,
                "validation_report_sha256": validation_report_hash,
                "labels_sha256": labels_hash,
                "config_sha256": config_sha256,
                "assignments": [base_assignment],
            }
            split_sha256 = hashlib.sha256(
                json.dumps(
                    split_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            split_id = f"fall_event_v1:sha256:{split_sha256}"
            assignment_row = {**base_assignment, "split_id": split_id}
            assignments_path.write_text(
                json.dumps(assignment_row) + "\n", encoding="utf-8"
            )
            assignments_hash = hashlib.sha256(assignments_path.read_bytes()).hexdigest()
            prediction_row = _prediction(split_id=split_id)
            predictions_path.write_text(
                json.dumps(prediction_row) + "\n", encoding="utf-8"
            )
            split_path.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "split_name": "fall_event_v1",
                        "task_type": "fall_event",
                        "split_id": split_id,
                        "split_sha256": split_sha256,
                        "manifest_sha256": manifest_hash,
                        "manifest_canonical_sha256": manifest_canonical_hash,
                        "validation_report_sha256": validation_report_hash,
                        "assignments_sha256": assignments_hash,
                        "labels_sha256": labels_hash,
                        "schema_version": "1.0",
                        "protocol_status": "provisional",
                        "config_sha256": config_sha256,
                    }
                ),
                encoding="utf-8",
            )
            arguments = [
                "--ground-truth",
                str(truth_path),
                "--predictions",
                str(predictions_path),
                "--manifest",
                str(manifest_path),
                "--split",
                str(split_path),
                "--assignments",
                str(assignments_path),
                "--config",
                "configs/evaluation/fall_event_v1.provisional.yaml",
                "--validation-report",
                str(validation_report_path),
                "--output-dir",
                str(output_dir),
                "--label-version",
                "synthetic-labels-v1",
            ]
            with self.assertRaises(SystemExit):
                evaluation_main(arguments)
            config_path = Path(
                "configs/evaluation/fall_event_v1.provisional.yaml"
            ).resolve()
            monitored_paths = {
                path.resolve()
                for path in (
                    truth_path,
                    predictions_path,
                    manifest_path,
                    split_path,
                    assignments_path,
                    validation_report_path,
                    config_path,
                )
            }
            read_counts = {path: 0 for path in monitored_paths}
            original_inputs = {
                path: path.read_bytes() for path in monitored_paths
            }
            original_read_bytes = Path.read_bytes

            def tracked_read_bytes(path):
                resolved = path.resolve()
                if resolved in read_counts:
                    read_counts[resolved] += 1
                content = original_read_bytes(path)
                if resolved in read_counts:
                    path.write_bytes(b"\xff")
                return content

            with mock.patch.object(Path, "read_bytes", tracked_read_bytes):
                result = evaluation_main([*arguments, "--allow-provisional"])
            for path, content in original_inputs.items():
                path.write_bytes(content)

            metrics = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )

            assignments_path.write_text(
                assignments_path.read_text(encoding="utf-8")
                + json.dumps({**assignment_row, "video_id": "tampered"})
                + "\n",
                encoding="utf-8",
            )
            tampered_arguments = list(arguments)
            tampered_arguments[tampered_arguments.index("--output-dir") + 1] = str(
                root / "tampered-output"
            )
            with self.assertRaises(SystemExit):
                evaluation_main([*tampered_arguments, "--allow-provisional"])

        self.assertEqual(result, 0)
        self.assertEqual(set(read_counts.values()), {1})
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["reproducibility"]["partition"], "validation")
        self.assertEqual(
            metrics["reproducibility"]["validation_report_hash"],
            validation_report_hash,
        )
        self.assertEqual(
            metrics["reproducibility"]["evaluation_implementation_sha256"],
            _evaluation_implementation_sha256(),
        )
        self.assertEqual(
            metrics["reproducibility"]["prediction_hash"],
            hashlib.sha256(original_inputs[predictions_path.resolve()]).hexdigest(),
        )
        self.assertEqual(
            metrics["reproducibility"]["package_source"],
            "src/elderly_monitoring/__init__.py",
        )
        self.assertFalse(
            Path(metrics["reproducibility"]["python_executable"]).is_absolute()
        )
        self.assertNotIn("python_prefix", metrics["reproducibility"])

    def test_test_partition_requires_frozen_governance_acknowledgement(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen"):
            _validate_test_partition_access(
                partition="test",
                protocol_status="development_provisional",
                protocol_version="fall-event-eval-v1",
                split_metadata={"status": "ready", "split_id": "split-1"},
                acknowledgement_path=None,
                evaluation_config_sha256="a" * 64,
                prediction_sha256="b" * 64,
                labels_sha256="c" * 64,
                code_commit="d" * 40,
            )

    def test_frozen_validation_report_file_is_required_and_must_match_split(
        self,
    ) -> None:
        split_metadata = {"validation_report_sha256": "a" * 64}
        with self.assertRaisesRegex(ValueError, "--validation-report"):
            _validate_validation_report_binding(
                protocol_status="frozen",
                split_metadata=split_metadata,
                validation_report_snapshot=None,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "validation-report.json"
            report_path.write_text(json.dumps({"valid": True}), encoding="utf-8")
            snapshot = evaluation_cli_module._load_json_object_snapshot(report_path)

        with self.assertRaisesRegex(ValueError, "does not match"):
            _validate_validation_report_binding(
                protocol_status="frozen",
                split_metadata=split_metadata,
                validation_report_snapshot=snapshot,
            )

        matching_metadata = {"validation_report_sha256": snapshot.sha256}
        self.assertEqual(
            _validate_validation_report_binding(
                protocol_status="frozen",
                split_metadata=matching_metadata,
                validation_report_snapshot=snapshot,
            ),
            snapshot.sha256,
        )

    def test_test_partition_acknowledgement_is_read_once_for_parse_and_hash(
        self,
    ) -> None:
        acknowledgement = {
            "partition": "test",
            "split_id": "split-1",
            "protocol_version": "fall-event-eval-v1",
            "blind_test_governance_confirmed": True,
            "evaluation_config_sha256": "a" * 64,
            "prediction_sha256": "b" * 64,
            "labels_sha256": "c" * 64,
            "code_commit": "d" * 40,
            "authorization_id": "authorization-1",
            "authorized_by_role": "data-governance-owner",
            "evaluation_run_id": "evaluation-run-1",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test-release-ack.json"
            path.write_text(json.dumps(acknowledgement), encoding="utf-8")
            original_read_bytes = Path.read_bytes
            read_count = 0

            def tracked_read_bytes(candidate):
                nonlocal read_count
                if candidate == path:
                    read_count += 1
                return original_read_bytes(candidate)

            with mock.patch.object(Path, "read_bytes", tracked_read_bytes):
                metadata = _validate_test_partition_access(
                    partition="test",
                    protocol_status="frozen",
                    protocol_version="fall-event-eval-v1",
                    split_metadata={
                        "status": "frozen",
                        "protocol_status": "frozen",
                        "split_id": "split-1",
                    },
                    acknowledgement_path=path,
                    evaluation_config_sha256="a" * 64,
                    prediction_sha256="b" * 64,
                    labels_sha256="c" * 64,
                    code_commit="d" * 40,
                )

        self.assertEqual(read_count, 1)
        self.assertEqual(
            metadata["test_release_ack_sha256"],
            hashlib.sha256(json.dumps(acknowledgement).encode("utf-8")).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
