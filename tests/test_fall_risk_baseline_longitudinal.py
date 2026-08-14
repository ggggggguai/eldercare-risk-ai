from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from elderly_monitoring.modules.fall_risk.baseline_longitudinal import (
    LongitudinalDataError,
    audit_longitudinal_split,
    build_longitudinal_split,
    evaluate_longitudinal_ablation,
    generate_longitudinal_ablation_predictions,
    write_longitudinal_evaluation_bundle,
)
from elderly_monitoring.modules.fall_risk.baseline import BaselineModelConfig

from test_fall_risk_baseline import make_daily_record


REQUIRED_VARIANTS = (
    "no_personal_baseline",
    "mean_std_personal_baseline",
    "median_mad_personal_baseline",
    "robust_ewma_cusum_guarded",
)


def protocol(
    *,
    status: str = "provisional",
    partitions: dict[str, float] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "fall-baseline-longitudinal-protocol-v1",
        "protocol_version": "longitudinal-baseline-v1-test",
        "protocol_status": status,
        "seed": "longitudinal-test-seed",
        "aggregation_period": "day",
        "outer_partitions": partitions
        or {"train": 0.34, "validation": 0.33, "test": 0.33},
        "min_reference_periods": 3,
        "min_scoring_periods_per_identity": 2,
        "min_subjects_per_partition": 0,
        "min_total_subjects": 1,
        "min_valid_monitoring_hours": 1.0,
        "prediction_horizon_hours": {"min": 0.0, "max": 72.0},
        "high_risk_level_threshold": 3,
        "score_threshold": 0.5,
        "bootstrap_iterations": 40,
        "bootstrap_seed": 20260812,
        "required_variants": list(REQUIRED_VARIANTS),
        "review": {
            "accepted_decision": "accepted",
            "manual_consensus_min_reviewers": 2,
            "clinical_proxy_min_reviewers": 1,
        },
        "test_governance": {
            "custodian": "test-custodian" if status == "frozen" else None,
            "frozen_at": "2026-08-12T00:00:00+08:00" if status == "frozen" else None,
        },
    }


def profile(person_id: str, source_group_id: str) -> dict[str, object]:
    return {
        "subject_id": person_id,
        "profile_version": "profile-v1",
        "profile_source": "project_collection",
        "consent_id": f"consent-{person_id}",
        "features": {
            "source_group_id": source_group_id,
            "authorized_device_ids": ["device-home-1"],
            "authorized_camera_profile_ids": ["camera-home-1"],
        },
    }


def observation(
    person_index: int,
    day: int,
    *,
    source_group_id: str | None = None,
    labelled: bool = True,
    risk_level: int = 0,
) -> tuple[dict[str, object], dict[str, object] | None, list[dict[str, object]]]:
    person_id = f"person-{person_index:02d}"
    source_group = source_group_id or f"home-{person_index:02d}"
    row = make_daily_record(day, person_id=person_id)
    row.update(
        {
            "observation_id": f"obs-{person_index:02d}-{day:02d}",
            "asset_id": f"longitudinal-asset-{person_index:02d}",
            "source_group_id": source_group,
            "quality_state": "valid",
            "no_personal_baseline_score": 0.15,
            "no_personal_baseline_available_weight": 0.68,
            "fusion_config_version": "fall-risk-fusion-test-v1",
            "outcome_label_id": f"risk-{person_index:02d}-{day:02d}" if labelled else None,
        }
    )
    if not labelled:
        return row, None, []

    period_start = datetime(2026, 6, day, tzinfo=timezone(timedelta(hours=8)))
    period_end = period_start + timedelta(days=1) - timedelta(seconds=1)
    period_end_epoch = period_end.timestamp()
    row["period_start"] = period_start.isoformat()
    row["period_end"] = period_end.isoformat()
    row["start_time"] = row["period_start"]
    row["end_time"] = row["period_end"]
    label = {
        "label_id": row["outcome_label_id"],
        "asset_id": row["asset_id"],
        "task_type": "longitudinal_baseline",
        "subject_id": person_id,
        "start_time": period_end_epoch + 3600.0,
        "end_time": period_end_epoch + 7200.0,
        "risk_level": risk_level,
        "risk_factors": ["reviewed_state_change"] if risk_level >= 3 else [],
        "label_source": "manual_consensus",
    }
    reviews = [
        {
            "review_id": f"review-{person_index:02d}-{day:02d}-{reviewer}",
            "label_id": label["label_id"],
            "reviewer_id": f"reviewer-{reviewer}",
            "decision": "accepted",
            "reviewed_at": "2026-08-11T12:00:00+08:00",
            "evidence_reference": f"evidence-{person_index:02d}-{day:02d}",
        }
        for reviewer in (1, 2)
    ]
    return row, label, reviews


def dataset(
    *,
    people: int = 6,
    periods: int = 6,
) -> tuple[list[dict], list[dict], dict, list[dict]]:
    observations: list[dict] = []
    labels: list[dict] = []
    reviews: list[dict] = []
    profiles = {"schema_version": "fall-risk-subject-profiles-v2", "subjects": []}
    for person_index in range(1, people + 1):
        profiles["subjects"].append(profile(f"person-{person_index:02d}", f"home-{person_index:02d}"))
        for day in range(1, periods + 1):
            row, label, label_reviews = observation(
                person_index,
                day,
                labelled=day > 3,
                risk_level=3 if day == periods and person_index % 2 == 0 else 0,
            )
            observations.append(row)
            if label is not None:
                labels.append(label)
                reviews.extend(label_reviews)
    return observations, labels, profiles, reviews


class LongitudinalSplitTest(unittest.TestCase):
    def test_protocol_rejects_extra_variant_and_out_of_range_threshold(self) -> None:
        extra_variant = protocol()
        extra_variant["required_variants"].append("unregistered_candidate")
        invalid_threshold = protocol()
        invalid_threshold["score_threshold"] = 1.1

        with self.assertRaisesRegex(ValueError, "four baseline ablations"):
            build_longitudinal_split([], [], {}, [], extra_variant)
        with self.assertRaisesRegex(ValueError, "score_threshold"):
            build_longitudinal_split([], [], {}, [], invalid_threshold)

    def test_build_is_deterministic_label_free_and_forward_only(self) -> None:
        observations, labels, profiles, reviews = dataset()

        first = build_longitudinal_split(
            observations, labels, profiles, reviews, protocol()
        )
        second = build_longitudinal_split(
            reversed(observations), reversed(labels), profiles, reversed(reviews), protocol()
        )

        self.assertEqual(first, second)
        self.assertEqual(first["metadata"]["status"], "ready")
        self.assertEqual(audit_longitudinal_split(first["assignments"]), [])
        self.assertEqual(len(first["assignments"]), len(observations))

        partitions_by_person: dict[str, set[str]] = {}
        by_identity: dict[tuple[str, str], list[dict]] = {}
        for assignment in first["assignments"]:
            partitions_by_person.setdefault(assignment["person_id"], set()).add(
                assignment["partition"]
            )
            by_identity.setdefault(
                (assignment["person_id"], assignment["camera_profile_id"]), []
            ).append(assignment)
            for forbidden in ("risk_level", "risk_factors", "outcome_label_id", "label_id"):
                self.assertNotIn(forbidden, assignment)

        self.assertTrue(all(len(values) == 1 for values in partitions_by_person.values()))
        for rows in by_identity.values():
            ordered = sorted(rows, key=lambda row: row["period_start_epoch"])
            self.assertEqual([row["role"] for row in ordered[:3]], ["reference"] * 3)
            self.assertEqual([row["role"] for row in ordered[3:]], ["scoring"] * 3)
            first_scoring = ordered[3]["period_start_epoch"]
            self.assertTrue(
                all(row["period_end_epoch"] < first_scoring for row in ordered[:3])
            )

    def test_source_group_is_outer_isolation_unit(self) -> None:
        observations, labels, profiles, reviews = dataset(people=2)
        for subject in profiles["subjects"]:
            subject["features"]["source_group_id"] = "shared-home"
        for row in observations:
            row["source_group_id"] = "shared-home"

        artifact = build_longitudinal_split(
            observations, labels, profiles, reviews, protocol()
        )

        self.assertEqual(
            len({row["partition"] for row in artifact["assignments"]}), 1
        )

    def test_missing_real_inputs_fail_closed_with_machine_blockers(self) -> None:
        artifact = build_longitudinal_split(
            [],
            [],
            {"schema_version": "fall-risk-subject-profiles-v2", "subjects": []},
            [],
            protocol(),
        )

        self.assertEqual(artifact["metadata"]["status"], "blocked")
        self.assertIsNone(artifact["metadata"]["split_id"] )
        self.assertEqual(artifact["assignments"], [])
        codes = {item["code"] for item in artifact["metadata"]["blockers"]}
        self.assertEqual(
            codes,
            {"no_longitudinal_observations", "no_risk_labels", "no_subject_profiles"},
        )

    def test_invalid_future_outcome_and_review_are_rejected(self) -> None:
        observations, labels, profiles, reviews = dataset(people=1)
        labels[0]["start_time"] = (
            datetime.fromisoformat(str(observations[3]["period_end"])).timestamp() - 1.0
        )
        reviews[:] = [row for row in reviews if row["label_id"] != labels[1]["label_id"]]

        artifact = build_longitudinal_split(
            observations, labels, profiles, reviews, protocol()
        )

        self.assertEqual(artifact["metadata"]["status"], "blocked")
        codes = {item["code"] for item in artifact["metadata"]["blockers"]}
        self.assertIn("outcome_not_after_observation", codes)
        self.assertIn("insufficient_label_reviews", codes)

    def test_overlapping_periods_for_one_identity_are_rejected(self) -> None:
        observations, labels, profiles, reviews = dataset(people=1)
        observations[1]["period_start"] = observations[0]["period_end"]

        artifact = build_longitudinal_split(
            observations, labels, profiles, reviews, protocol()
        )

        codes = {item["code"] for item in artifact["metadata"]["blockers"]}
        self.assertIn("overlapping_identity_periods", codes)

    def test_functional_proxy_labels_are_ignored_by_longitudinal_builder(self) -> None:
        observations, labels, profiles, reviews = dataset(people=1)
        labels.append(
            {
                "label_id": "functional-proxy-only",
                "asset_id": "other-asset",
                "task_type": "functional_proxy",
                "subject_id": "person-01",
                "start_time": 1.0,
                "end_time": 2.0,
                "risk_level": 1,
                "risk_factors": [],
                "label_source": "clinical_proxy",
            }
        )

        artifact = build_longitudinal_split(
            observations, labels, profiles, reviews, protocol()
        )

        self.assertEqual(artifact["metadata"]["status"], "ready")

    def test_frozen_split_requires_formal_v2_validation_attestation(self) -> None:
        observations, labels, profiles, reviews = dataset(people=1)

        frozen_protocol = protocol(status="frozen")
        frozen_protocol["bootstrap_iterations"] = 10_000
        artifact = build_longitudinal_split(
            observations, labels, profiles, reviews, frozen_protocol
        )

        codes = {item["code"] for item in artifact["metadata"]["blockers"]}
        self.assertIn("missing_formal_validation_attestation", codes)

    def test_frozen_protocol_requires_production_bootstrap_budget(self) -> None:
        frozen_protocol = protocol(status="frozen")

        with self.assertRaisesRegex(
            ValueError, "frozen protocol requires at least 10000 bootstrap iterations"
        ):
            build_longitudinal_split([], [], {}, [], frozen_protocol)

    def test_review_for_unknown_longitudinal_label_is_rejected(self) -> None:
        observations, labels, profiles, reviews = dataset(people=1)
        reviews.append(
            {
                "review_id": "orphan-review",
                "label_id": "missing-label",
                "reviewer_id": "reviewer-1",
                "decision": "accepted",
                "reviewed_at": "2026-08-12T00:00:00+08:00",
                "evidence_reference": "missing-evidence",
            }
        )

        artifact = build_longitudinal_split(
            observations, labels, profiles, reviews, protocol()
        )

        codes = {item["code"] for item in artifact["metadata"]["blockers"]}
        self.assertIn("review_for_unknown_label", codes)

    def test_review_requires_known_reviewer_decision_and_evidence(self) -> None:
        observations, labels, profiles, reviews = dataset(people=1)
        reviews[0]["reviewer_id"] = "unknown"
        reviews[1]["decision"] = "auto_accepted"
        reviews[2]["evidence_reference"] = ""

        artifact = build_longitudinal_split(
            observations, labels, profiles, reviews, protocol()
        )

        codes = {item["code"] for item in artifact["metadata"]["blockers"]}
        self.assertIn("invalid_label_review", codes)


class LongitudinalEvaluationTest(unittest.TestCase):
    def test_causal_replay_generates_four_variants_without_outcomes(self) -> None:
        observations, labels, profiles, reviews = dataset(people=1, periods=6)
        development_protocol = protocol(
            partitions={"train": 0.0, "validation": 1.0, "test": 0.0}
        )
        artifact = build_longitudinal_split(
            observations, labels, profiles, reviews, development_protocol
        )
        partition_observations = list(observations)
        model_config = BaselineModelConfig(
            min_history_days=3,
            stable_history_days=3,
            min_history_records=3,
            max_history_days=14,
        )

        predictions = generate_longitudinal_ablation_predictions(
            partition_observations,
            artifact["assignments"],
            artifact["metadata"],
            development_protocol,
            partition="validation",
            baseline_config=model_config,
        )
        changed_labels = [{**row, "risk_level": 4 - row["risk_level"]} for row in labels]
        repeated = generate_longitudinal_ablation_predictions(
            partition_observations,
            artifact["assignments"],
            artifact["metadata"],
            development_protocol,
            partition="validation",
            baseline_config=model_config,
        )

        self.assertEqual(predictions, repeated)
        self.assertTrue(changed_labels)  # Labels are deliberately not passed to replay.
        by_variant: dict[str, set[str]] = {}
        for row in predictions:
            by_variant.setdefault(row["variant"], set()).add(row["observation_id"])
        self.assertEqual(set(by_variant), set(REQUIRED_VARIANTS))
        self.assertEqual(len({frozenset(values) for values in by_variant.values()}), 1)
        no_baseline = [
            row for row in predictions if row["variant"] == "no_personal_baseline"
        ]
        self.assertTrue(all(row["score"] == 0.15 for row in no_baseline))

    def _validation_fixture(
        self,
    ) -> tuple[list[dict], list[dict], dict, list[dict], list[dict]]:
        observations, labels, profiles, reviews = dataset(people=2)
        artifact = build_longitudinal_split(
            observations,
            labels,
            profiles,
            reviews,
            protocol(partitions={"train": 0.0, "validation": 1.0, "test": 0.0}),
        )
        assignments = artifact["assignments"]
        partition_ids = {
            row["observation_id"]
            for row in assignments
            if row["partition"] == "validation"
        }
        scoring_ids = {
            row["observation_id"]
            for row in assignments
            if row["partition"] == "validation" and row["role"] == "scoring"
        }
        selected_observations = [
            row for row in observations if row["observation_id"] in partition_ids
        ]
        selected_label_ids = {
            row["outcome_label_id"]
            for row in selected_observations
            if row["observation_id"] in scoring_ids
        }
        selected_labels = [
            row for row in labels if row["label_id"] in selected_label_ids
        ]
        predictions = []
        for variant_index, variant in enumerate(REQUIRED_VARIANTS):
            for row in observations:
                if row["observation_id"] not in scoring_ids:
                    continue
                label = next(
                    item for item in labels if item["label_id"] == row["outcome_label_id"]
                )
                score = 0.85 if label["risk_level"] >= 3 else 0.10
                if variant_index == 0 and row["period_id"].endswith("05"):
                    score = 0.75
                predictions.append(
                    {
                        "observation_id": row["observation_id"],
                        "variant": variant,
                        "score": score,
                        "baseline_state": "stable",
                        "quality_state": row["quality_state"],
                        "model_version": f"{variant}-test-v1",
                        "config_hash": "a" * 64,
                        "split_id": artifact["metadata"]["split_id"],
                    }
                )
        return (
            selected_observations,
            selected_labels,
            artifact["metadata"],
            assignments,
            predictions,
        )

    def test_evaluates_required_variants_with_strata_ci_and_failures(self) -> None:
        observations, labels, split, assignments, predictions = self._validation_fixture()

        result = evaluate_longitudinal_ablation(
            observations,
            labels,
            assignments,
            predictions,
            protocol(partitions={"train": 0.0, "validation": 1.0, "test": 0.0}),
            partition="validation",
            split_metadata=split,
        )

        self.assertEqual(set(result.metrics_by_variant), set(REQUIRED_VARIANTS))
        robust = result.metrics_by_variant["robust_ewma_cusum_guarded"]
        self.assertEqual(robust["f1"], 1.0)
        self.assertEqual(robust["false_positives_per_observed_day"], 0.0)
        self.assertIn("f1", result.confidence_intervals["robust_ewma_cusum_guarded"])
        self.assertIn("person_id", result.stratified_metrics["robust_ewma_cusum_guarded"])
        self.assertTrue(
            any(
                row["variant"] == "no_personal_baseline"
                and row["failure_type"] == "false_positive"
                for row in result.failure_cases
            )
        )

        reversed_result = evaluate_longitudinal_ablation(
            list(reversed(observations)),
            list(reversed(labels)),
            list(reversed(assignments)),
            list(reversed(predictions)),
            protocol(partitions={"train": 0.0, "validation": 1.0, "test": 0.0}),
            partition="validation",
            split_metadata=split,
        )
        self.assertEqual(result.metrics_by_variant, reversed_result.metrics_by_variant)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "bundle"
            write_longitudinal_evaluation_bundle(
                result, output, metadata={"data_status": "synthetic"}
            )
            report = (output / "report.md").read_text(encoding="utf-8")
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        self.assertIn("synthetic", report.lower())
        self.assertEqual(metrics["partition"], "validation")

    def test_false_positive_denominators_deduplicate_household_and_camera_periods(self) -> None:
        observations, labels, profiles, reviews = dataset(people=2)
        for subject in profiles["subjects"]:
            subject["features"]["source_group_id"] = "shared-home"
        for row in observations:
            row["source_group_id"] = "shared-home"
        development_protocol = protocol(
            partitions={"train": 0.0, "validation": 1.0, "test": 0.0}
        )
        artifact = build_longitudinal_split(
            observations, labels, profiles, reviews, development_protocol
        )
        scoring_ids = {
            row["observation_id"]
            for row in artifact["assignments"]
            if row["role"] == "scoring"
        }
        scoring_label_ids = {
            row["outcome_label_id"]
            for row in observations
            if row["observation_id"] in scoring_ids
        }
        predictions = [
            {
                "observation_id": observation_id,
                "variant": variant,
                "score": 0.75 if observation_id.endswith("-05") else 0.1,
                "baseline_state": "stable",
                "quality_state": "valid",
                "model_version": f"{variant}-test-v1",
                "config_hash": "a" * 64,
                "split_id": artifact["metadata"]["split_id"],
            }
            for variant in REQUIRED_VARIANTS
            for observation_id in sorted(scoring_ids)
        ]

        result = evaluate_longitudinal_ablation(
            observations,
            [row for row in labels if row["label_id"] in scoring_label_ids],
            artifact["assignments"],
            predictions,
            development_protocol,
            partition="validation",
            split_metadata=artifact["metadata"],
        )

        metrics = result.metrics_by_variant["no_personal_baseline"]
        self.assertEqual(metrics["household_day_count"], 3)
        self.assertEqual(metrics["camera_hour_count"], 6.0)
        self.assertAlmostEqual(
            metrics["false_positives_per_household_day"], 2 / 3, places=6
        )
        self.assertAlmostEqual(
            metrics["false_positives_per_camera_hour"], 2 / 6, places=6
        )

    def test_variant_coverage_must_be_identical(self) -> None:
        observations, labels, split, assignments, predictions = self._validation_fixture()
        predictions.pop()

        with self.assertRaisesRegex(LongitudinalDataError, "prediction coverage"):
            evaluate_longitudinal_ablation(
                observations,
                labels,
                assignments,
                predictions,
                protocol(partitions={"train": 0.0, "validation": 1.0, "test": 0.0}),
                partition="validation",
                split_metadata=split,
            )

    def test_validation_rejects_unsealed_observations_from_other_partitions(self) -> None:
        observations, labels, profiles, reviews = dataset(people=12)
        partitioned = build_longitudinal_split(
            observations, labels, profiles, reviews, protocol()
        )
        validation_ids = {
            row["observation_id"]
            for row in partitioned["assignments"]
            if row["partition"] == "validation"
        }
        scoring_ids = {
            row["observation_id"]
            for row in partitioned["assignments"]
            if row["partition"] == "validation" and row["role"] == "scoring"
        }
        validation_labels = [
            row
            for row in labels
            if any(
                observation["outcome_label_id"] == row["label_id"]
                and observation["observation_id"] in scoring_ids
                for observation in observations
            )
        ]
        predictions = [
            {
                "observation_id": observation_id,
                "variant": variant,
                "score": 0.1,
                "baseline_state": "stable",
                "quality_state": "valid",
                "model_version": f"{variant}-test-v1",
                "config_hash": "a" * 64,
                "split_id": partitioned["metadata"]["split_id"],
            }
            for variant in REQUIRED_VARIANTS
            for observation_id in sorted(scoring_ids)
        ]
        self.assertTrue(validation_ids)
        self.assertNotEqual(validation_ids, {row["observation_id"] for row in observations})

        with self.assertRaisesRegex(LongitudinalDataError, "exactly one partition"):
            evaluate_longitudinal_ablation(
                observations,
                validation_labels,
                partitioned["assignments"],
                predictions,
                protocol(),
                partition="validation",
                split_metadata=partitioned["metadata"],
            )

    def test_partition_input_hash_rejects_mutated_validation_features(self) -> None:
        observations, labels, split, assignments, predictions = self._validation_fixture()
        observations[0]["valid_monitoring_hours"] = 7.5

        with self.assertRaisesRegex(LongitudinalDataError, "observation hash"):
            evaluate_longitudinal_ablation(
                observations,
                labels,
                assignments,
                predictions,
                protocol(partitions={"train": 0.0, "validation": 1.0, "test": 0.0}),
                partition="validation",
                split_metadata=split,
            )

    def test_split_root_rejects_tampered_partition_hash_metadata(self) -> None:
        observations, labels, split, assignments, predictions = self._validation_fixture()
        tampered = json.loads(json.dumps(split))
        tampered["partition_input_sha256"]["validation"]["observations"] = "f" * 64

        with self.assertRaisesRegex(LongitudinalDataError, "split root hash"):
            evaluate_longitudinal_ablation(
                observations,
                labels,
                assignments,
                predictions,
                protocol(partitions={"train": 0.0, "validation": 1.0, "test": 0.0}),
                partition="validation",
                split_metadata=tampered,
            )

    def test_frozen_protocol_rejects_non_frozen_split_status(self) -> None:
        observations, labels, profiles, reviews = dataset(people=1)
        frozen = protocol(
            status="frozen",
            partitions={"train": 0.0, "validation": 1.0, "test": 0.0},
        )
        frozen["bootstrap_iterations"] = 10_000
        input_hashes = {"risk_labels": "a" * 64, "subject_profiles": "b" * 64}
        formal = {
            "schema_version": "fall-risk-label-validation-report-v2",
            "mode": "formal",
            "valid": True,
            "formal_ready": True,
            "input_sha256": input_hashes,
        }
        artifact = build_longitudinal_split(
            observations,
            labels,
            profiles,
            reviews,
            frozen,
            formal_validation_report=formal,
            formal_input_sha256=input_hashes,
        )
        tampered = json.loads(json.dumps(artifact["metadata"]))
        tampered["status"] = "ready"

        with self.assertRaisesRegex(LongitudinalDataError, "split metadata"):
            generate_longitudinal_ablation_predictions(
                observations,
                artifact["assignments"],
                tampered,
                frozen,
                partition="validation",
                baseline_config=BaselineModelConfig(
                    min_history_days=3,
                    stable_history_days=3,
                    min_history_records=3,
                    max_history_days=14,
                ),
            )

    def test_test_partition_requires_frozen_protocol_and_explicit_release(self) -> None:
        observations, labels, profiles, reviews = dataset(people=1)
        provisional = protocol(
            partitions={"train": 0.0, "validation": 0.0, "test": 1.0}
        )
        artifact = build_longitudinal_split(
            observations, labels, profiles, reviews, provisional
        )
        predictions = []
        for variant in REQUIRED_VARIANTS:
            for assignment in artifact["assignments"]:
                if assignment["role"] != "scoring":
                    continue
                predictions.append(
                    {
                        "observation_id": assignment["observation_id"],
                        "variant": variant,
                        "score": 0.1,
                        "baseline_state": "stable",
                        "quality_state": "valid",
                        "model_version": f"{variant}-test-v1",
                        "config_hash": "a" * 64,
                        "split_id": artifact["metadata"]["split_id"],
                    }
                )

        with self.assertRaisesRegex(LongitudinalDataError, "frozen protocol"):
            evaluate_longitudinal_ablation(
                observations,
                labels,
                artifact["assignments"],
                predictions,
                provisional,
                partition="test",
                split_metadata=artifact["metadata"],
                test_release_ack={},
            )

        frozen = protocol(
            status="frozen",
            partitions={"train": 0.0, "validation": 0.0, "test": 1.0},
        )
        frozen["bootstrap_iterations"] = 10_000
        with self.assertRaisesRegex(LongitudinalDataError, "explicit test release"):
            evaluate_longitudinal_ablation(
                observations,
                labels,
                artifact["assignments"],
                predictions,
                frozen,
                partition="test",
                split_metadata=artifact["metadata"],
            )


class LongitudinalCliTest(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    def test_real_empty_inputs_write_blocked_artifact_and_exit_three(self) -> None:
        repo = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profiles = root / "profiles.json"
            profiles.write_text(
                json.dumps(
                    {
                        "schema_version": "fall-risk-subject-profiles-v2",
                        "subjects": [],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "split"
            command = [
                "conda",
                "run",
                "-n",
                "eldercare-ai",
                "python",
                "scripts/split/build_fall_baseline_longitudinal_split.py",
                "--observations",
                str(root / "missing-observations.jsonl"),
                "--risk-labels",
                str(root / "missing-labels.jsonl"),
                "--subject-profiles",
                str(profiles),
                "--review-log",
                str(root / "missing-review.jsonl"),
                "--output-dir",
                str(output),
            ]
            completed = subprocess.run(
                command,
                cwd=repo,
                check=False,
                capture_output=True,
                text=True,
            )
            metadata = json.loads((output / "split.json").read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 3, completed.stderr)
        self.assertEqual(metadata["status"], "blocked")
        self.assertEqual(
            {row["code"] for row in metadata["blockers"]},
            {"no_longitudinal_observations", "no_risk_labels", "no_subject_profiles"},
        )

    def test_frozen_split_cli_binds_formal_report_to_input_file_bytes(self) -> None:
        repo = Path(__file__).parents[1]
        observations, labels, profiles_payload, reviews = dataset(people=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            observations_path = root / "observations.jsonl"
            labels_path = root / "risk-labels.jsonl"
            profiles_path = root / "profiles.json"
            reviews_path = root / "reviews.jsonl"
            config_path = root / "protocol.yaml"
            report_path = root / "formal-report.json"
            self._write_jsonl(observations_path, observations)
            self._write_jsonl(labels_path, labels)
            profiles_path.write_text(
                json.dumps(profiles_payload, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._write_jsonl(reviews_path, reviews)
            frozen_protocol = protocol(
                status="frozen",
                partitions={"train": 0.0, "validation": 1.0, "test": 0.0},
            )
            frozen_protocol["bootstrap_iterations"] = 10_000
            config_path.write_text(
                json.dumps(frozen_protocol, ensure_ascii=False), encoding="utf-8"
            )
            input_hashes = {
                "risk_labels": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
                "subject_profiles": hashlib.sha256(profiles_path.read_bytes()).hexdigest(),
            }
            report = {
                "schema_version": "fall-risk-label-validation-report-v2",
                "mode": "formal",
                "valid": True,
                "formal_ready": True,
                "input_sha256": input_hashes,
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")

            base_command = [
                "conda",
                "run",
                "-n",
                "eldercare-ai",
                "python",
                "scripts/split/build_fall_baseline_longitudinal_split.py",
                "--observations",
                str(observations_path),
                "--risk-labels",
                str(labels_path),
                "--subject-profiles",
                str(profiles_path),
                "--review-log",
                str(reviews_path),
                "--config",
                str(config_path),
                "--formal-validation-report",
                str(report_path),
            ]
            frozen_output = root / "frozen-split"
            accepted = subprocess.run(
                [*base_command, "--output-dir", str(frozen_output)],
                cwd=repo,
                check=False,
                capture_output=True,
                text=True,
            )
            frozen_metadata = json.loads(
                (frozen_output / "split.json").read_text(encoding="utf-8")
            )

            report["input_sha256"]["risk_labels"] = "f" * 64
            report_path.write_text(json.dumps(report), encoding="utf-8")
            blocked_output = root / "blocked-split"
            rejected = subprocess.run(
                [*base_command, "--output-dir", str(blocked_output)],
                cwd=repo,
                check=False,
                capture_output=True,
                text=True,
            )
            blocked_metadata = json.loads(
                (blocked_output / "split.json").read_text(encoding="utf-8")
            )

        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertEqual(frozen_metadata["status"], "frozen")
        self.assertEqual(rejected.returncode, 3, rejected.stderr)
        self.assertIn(
            "formal_validation_input_mismatch",
            {row["code"] for row in blocked_metadata["blockers"]},
        )

    def test_evaluation_cli_rejects_inconsistent_data_status_claims(self) -> None:
        repo = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "protocol.yaml"
            split_path = root / "split.json"
            config_path.write_text(json.dumps(protocol()), encoding="utf-8")
            split_path.write_text(json.dumps({"status": "ready"}), encoding="utf-8")
            base_command = [
                "conda",
                "run",
                "-n",
                "eldercare-ai",
                "python",
                "scripts/evaluate/evaluate_fall_baseline_longitudinal.py",
                "--observations",
                str(root / "unopened-observations.jsonl"),
                "--risk-labels",
                str(root / "unopened-labels.jsonl"),
                "--assignments",
                str(root / "unopened-assignments.jsonl"),
                "--split",
                str(split_path),
                "--predictions",
                str(root / "unopened-predictions.jsonl"),
                "--config",
                str(config_path),
                "--output-dir",
                str(root / "unwritten-bundle"),
            ]
            false_formal = subprocess.run(
                [*base_command, "--partition", "validation", "--data-status", "formal"],
                cwd=repo,
                check=False,
                capture_output=True,
                text=True,
            )
            false_test = subprocess.run(
                [*base_command, "--partition", "test", "--data-status", "synthetic"],
                cwd=repo,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(false_formal.returncode, 2)
        self.assertIn("formal data status requires", false_formal.stderr)
        self.assertEqual(false_test.returncode, 2)
        self.assertIn("test partition requires --data-status formal", false_test.stderr)


if __name__ == "__main__":
    unittest.main()
