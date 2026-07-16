import hashlib
import json
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import yaml

from elderly_monitoring.modules.fall_risk.annotation_validation import DEFAULT_CONFIG
from elderly_monitoring.modules.fall_risk.splits import (
    FrozenSplitError,
    SplitConfigError,
    SplitDataError,
    audit_split_leakage,
    build_fall_risk_splits,
    build_splits_from_files,
    write_split_artifacts,
)


TASK_TYPES = {
    "fall_event_v1": "fall_event",
    "near_fall_event_v1": "near_fall_event",
    "functional_proxy_v1": "functional_proxy",
    "longitudinal_baseline_v1": "longitudinal_baseline",
}


def _config(
    *,
    seed: str = "split-test-seed",
    frozen: bool = False,
    partitions: dict[str, float] | None = None,
    protocol_status: str | None = None,
) -> dict:
    label_settings = {
        "fall_event_v1": {
            "label_file": "event_labels.jsonl",
            "event_types": ["fall"],
        },
        "near_fall_event_v1": {
            "label_file": "event_labels.jsonl",
            "event_types": ["near_fall"],
        },
        "functional_proxy_v1": {
            "label_file": "risk_labels.jsonl",
            "label_task_types": ["functional_proxy"],
        },
        "longitudinal_baseline_v1": {
            "label_file": "risk_labels.jsonl",
            "label_task_types": ["longitudinal_baseline"],
        },
    }
    tasks = {}
    for split_name, task_type in TASK_TYPES.items():
        tasks[split_name] = {
            "task_type": task_type,
            "frozen": frozen,
            "stratify_by": [],
            **label_settings[split_name],
        }
    return {
        "schema_version": "1.0",
        "protocol_status": protocol_status or ("frozen" if frozen else "provisional"),
        "seed": seed,
        "partitions": partitions
        or {"train": 1.0, "validation": 0.0, "test": 0.0},
        "tasks": tasks,
    }


def _manifest(
    asset_id: str,
    *,
    subject_id: str = "subject-1",
    source_group_id: str | None = None,
    original_event_id: str | None = None,
    eligibility: bool = True,
    content_sha256: str | None = None,
    **extra: object,
) -> dict:
    row = {
        "asset_id": asset_id,
        "video_id": asset_id,
        "dataset": "synthetic",
        "eligibility": eligibility,
        "source_uri": "https://example.test/dataset",
        "license_id": "CC-BY-4.0",
        "exclusion_reasons": [],
        "subject_id": subject_id,
        "source_group_id": source_group_id or f"group-{asset_id}",
        "original_event_id": original_event_id or f"event-{asset_id}",
        "sha256": content_sha256
        or hashlib.sha256(asset_id.encode("utf-8")).hexdigest(),
    }
    row.update(extra)
    return row


def _label(
    asset_id: str,
    *,
    label_id: str | None = None,
    review_status: str = "final",
    event_type: str = "fall",
    task_type: str | None = None,
    eligibility: bool | None = True,
    **extra: object,
) -> dict:
    row = {
        "label_id": label_id or f"label-{asset_id}",
        "asset_id": asset_id,
        "video_id": asset_id,
        "review_status": review_status,
        "event_type": event_type,
        "start_time": 1.0,
        "end_time": 2.0,
        "review_evidence_ids": [f"review-{asset_id}"],
    }
    if task_type is not None:
        row["task_type"] = task_type
    if eligibility is not None:
        row["eligibility"] = eligibility
    row.update(extra)
    return row


def _empty_labels() -> dict[str, list[dict]]:
    return {split_name: [] for split_name in TASK_TYPES}


def _labels_for_all_tasks() -> tuple[list[dict], dict[str, list[dict]]]:
    manifest = [
        _manifest("fall-ok", subject_id="fall-subject"),
        _manifest("near-ok", subject_id="near-subject"),
        _manifest("functional-ok", subject_id="functional-subject"),
        _manifest("longitudinal-ok", subject_id="longitudinal-subject"),
        _manifest("manifest-ineligible", eligibility=False),
        _manifest("pending-label"),
        _manifest("label-ineligible"),
    ]
    labels = {
        "fall_event_v1": [
            _label("fall-ok"),
            _label("manifest-ineligible"),
            _label("pending-label", review_status="pending"),
            _label("label-ineligible", eligibility=False),
        ],
        "near_fall_event_v1": [_label("near-ok", event_type="near_fall")],
        "functional_proxy_v1": [
            _label("functional-ok", task_type="functional_proxy")
        ],
        "longitudinal_baseline_v1": [
            _label("longitudinal-ok", task_type="longitudinal_baseline")
        ],
    }
    return manifest, labels


class FallRiskSplitBuildTest(unittest.TestCase):
    def test_builds_four_independent_schemas_and_filters_ineligible_rows(self) -> None:
        manifest, labels = _labels_for_all_tasks()

        artifacts = build_fall_risk_splits(manifest, labels, _config())

        self.assertEqual(set(artifacts), set(TASK_TYPES))
        for split_name, task_type in TASK_TYPES.items():
            artifact = artifacts[split_name]
            metadata = artifact["metadata"]
            self.assertEqual(metadata["schema_version"], "1.0")
            self.assertEqual(metadata["split_name"], split_name)
            self.assertEqual(metadata["task_type"], task_type)
            self.assertEqual(metadata["status"], "ready")
            self.assertEqual(metadata["protocol_status"], "provisional")
            self.assertTrue(metadata["split_id"].startswith(f"{split_name}:sha256:"))
            self.assertEqual(len(metadata["split_sha256"]), 64)
            self.assertEqual(len(artifact["assignments"]), 1)

            assignment = artifact["assignments"][0]
            self.assertEqual(assignment["split_id"], metadata["split_id"])
            self.assertNotIn("event_type", assignment)
            self.assertNotIn("start_time", assignment)
            self.assertNotIn("end_time", assignment)
            self.assertNotIn("label_id", assignment)

        fall_metadata = artifacts["fall_event_v1"]["metadata"]
        self.assertEqual(fall_metadata["excluded_counts"]["manifest_ineligible"], 1)
        self.assertEqual(fall_metadata["excluded_counts"]["review_status"], 1)
        self.assertEqual(fall_metadata["excluded_counts"]["label_eligibility"], 1)

    def test_unknown_subjects_use_source_group_and_missing_group_is_excluded(self) -> None:
        manifest = [
            _manifest("unknown-a", subject_id="unknown", source_group_id="shared"),
            _manifest("unknown-b", subject_id="unknown", source_group_id="shared"),
            _manifest("unknown-missing", subject_id="unknown", source_group_id="unknown"),
        ]
        labels = _empty_labels()
        labels["fall_event_v1"] = [_label(row["asset_id"]) for row in manifest]

        artifact = build_fall_risk_splits(manifest, labels, _config())["fall_event_v1"]

        self.assertEqual(len(artifact["assignments"]), 2)
        self.assertEqual(
            {row["leakage_component_id"] for row in artifact["assignments"]},
            {artifact["assignments"][0]["leakage_component_id"]},
        )
        self.assertEqual(
            artifact["metadata"]["excluded_counts"]["missing_conservative_group"],
            1,
        )

    def test_union_find_closes_all_leakage_relations_transitively(self) -> None:
        shared_hash = "f" * 64
        manifest = [
            _manifest("a", subject_id="p1", source_group_id="g1"),
            _manifest("b", subject_id="p1", source_group_id="g2"),
            _manifest("c", subject_id="p2", source_group_id="g2", original_event_id="e3"),
            _manifest(
                "d",
                subject_id="p3",
                source_group_id="g3",
                original_event_id="e3",
                content_sha256=shared_hash,
            ),
            _manifest("e", subject_id="p4", source_group_id="g4", content_sha256=shared_hash),
            _manifest("f", subject_id="p5", source_group_id="g5", content_sha256=shared_hash),
            _manifest(
                "derived",
                subject_id="p6",
                source_group_id="g6",
                derived_from_asset_id="f",
            ),
            _manifest(
                "adjacent",
                subject_id="p7",
                source_group_id="g7",
                adjacent_asset_ids=["derived"],
            ),
        ]
        labels = _empty_labels()
        labels["fall_event_v1"] = [_label(row["asset_id"]) for row in manifest]

        assignments = build_fall_risk_splits(manifest, labels, _config())[
            "fall_event_v1"
        ]["assignments"]

        self.assertEqual(len(assignments), len(manifest))
        self.assertEqual(len({row["leakage_component_id"] for row in assignments}), 1)
        self.assertEqual(len({row["partition"] for row in assignments}), 1)

    def test_input_order_does_not_change_assignments_or_hashes(self) -> None:
        manifest = [_manifest(f"asset-{index}", subject_id=f"p-{index}") for index in range(8)]
        labels = _empty_labels()
        labels["fall_event_v1"] = [_label(row["asset_id"]) for row in manifest]
        config = _config(partitions={"train": 0.5, "validation": 0.25, "test": 0.25})

        first = build_fall_risk_splits(manifest, labels, config)
        reversed_labels = _empty_labels()
        reversed_labels["fall_event_v1"] = list(reversed(labels["fall_event_v1"]))
        second = build_fall_risk_splits(list(reversed(manifest)), reversed_labels, config)

        self.assertEqual(first, second)

    def test_seed_changes_split_identity(self) -> None:
        manifest = [_manifest("asset-a")]
        labels = _empty_labels()
        labels["fall_event_v1"] = [_label("asset-a")]

        first = build_fall_risk_splits(manifest, labels, _config(seed="one"))
        second = build_fall_risk_splits(manifest, labels, _config(seed="two"))

        self.assertNotEqual(
            first["fall_event_v1"]["metadata"]["split_id"],
            second["fall_event_v1"]["metadata"]["split_id"],
        )

    def test_identifiers_and_content_hashes_are_normalized_before_grouping(self) -> None:
        manifest = [
            _manifest(
                "normalized-a",
                subject_id=" subject-1 ",
                source_group_id=" group-a ",
                content_sha256="A" * 64,
            ),
            _manifest(
                "normalized-b",
                subject_id="subject-1",
                source_group_id="group-b",
                content_sha256="a" * 64,
            ),
        ]
        labels = _empty_labels()
        labels["fall_event_v1"] = [_label(row["asset_id"]) for row in manifest]

        assignments = build_fall_risk_splits(manifest, labels, _config())[
            "fall_event_v1"
        ]["assignments"]

        self.assertEqual({row["subject_id"] for row in assignments}, {"subject-1"})
        self.assertEqual({row["content_sha256"] for row in assignments}, {"a" * 64})
        self.assertEqual(len({row["leakage_component_id"] for row in assignments}), 1)

    def test_invalid_content_hash_is_excluded_from_formal_split(self) -> None:
        manifest = [_manifest("bad-hash", content_sha256="not-a-sha256")]
        labels = _empty_labels()
        labels["fall_event_v1"] = [_label("bad-hash")]

        artifact = build_fall_risk_splits(manifest, labels, _config())["fall_event_v1"]

        self.assertEqual(artifact["metadata"]["status"], "blocked")
        self.assertEqual(artifact["metadata"]["excluded_counts"]["invalid_content_hash"], 1)

    def test_no_eligible_data_produces_blocked_artifacts_without_split_id(self) -> None:
        manifest = [_manifest("pending-only")]
        labels = _empty_labels()
        labels["fall_event_v1"] = [_label("pending-only", review_status="pending")]

        artifacts = build_fall_risk_splits(manifest, labels, _config())

        for artifact in artifacts.values():
            self.assertEqual(artifact["metadata"]["status"], "blocked")
            self.assertIsNone(artifact["metadata"]["split_id"])
            self.assertIsNone(artifact["metadata"]["split_sha256"])
            self.assertEqual(artifact["metadata"]["blockers"][0]["code"], "no_eligible_samples")
            self.assertEqual(artifact["assignments"], [])

    def test_leakage_audit_reports_cross_partition_keys(self) -> None:
        assignments = [
            {
                "sample_id": "a",
                "partition": "train",
                "subject_id": "p1",
                "source_group_id": "g1",
                "original_event_id": "e1",
                "content_sha256": "a" * 64,
                "leakage_component_id": "component-a",
            },
            {
                "sample_id": "b",
                "partition": "test",
                "subject_id": "p2",
                "source_group_id": "g1",
                "original_event_id": "e2",
                "content_sha256": "b" * 64,
                "leakage_component_id": "component-b",
            },
        ]

        issues = audit_split_leakage(assignments)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["key"], "source_group_id")
        self.assertEqual(issues[0]["partitions"], ["test", "train"])

    def test_invalid_config_is_rejected_before_build(self) -> None:
        config = _config()
        del config["tasks"]["near_fall_event_v1"]

        with self.assertRaises(SplitConfigError):
            build_fall_risk_splits([], _empty_labels(), config)

    def test_labels_require_explicit_eligibility_and_review_evidence(self) -> None:
        manifest = [_manifest("missing-eligibility"), _manifest("missing-review")]
        labels = _empty_labels()
        labels["fall_event_v1"] = [
            _label("missing-eligibility", eligibility=None),
            _label("missing-review", review_evidence_ids=[]),
        ]

        artifact = build_fall_risk_splits(manifest, labels, _config())["fall_event_v1"]

        self.assertEqual(artifact["metadata"]["status"], "blocked")
        self.assertEqual(artifact["metadata"]["excluded_counts"]["label_eligibility"], 1)
        self.assertEqual(
            artifact["metadata"]["excluded_counts"]["missing_review_evidence"], 1
        )

    def test_manifest_source_license_and_exclusions_fail_closed(self) -> None:
        manifest = [
            _manifest("missing-source", source_uri=None),
            _manifest("missing-license", license_id=None),
            _manifest("excluded", exclusion_reasons=["license_unknown"]),
        ]
        labels = _empty_labels()
        labels["fall_event_v1"] = [_label(row["asset_id"]) for row in manifest]

        artifact = build_fall_risk_splits(manifest, labels, _config())["fall_event_v1"]

        self.assertEqual(artifact["metadata"]["status"], "blocked")
        self.assertEqual(
            artifact["metadata"]["excluded_counts"]["manifest_provenance"], 3
        )

    def test_provisional_protocol_cannot_create_frozen_task(self) -> None:
        config = _config(protocol_status="provisional")
        config["tasks"]["fall_event_v1"]["frozen"] = True

        with self.assertRaises(SplitConfigError):
            build_fall_risk_splits([], _empty_labels(), config)

    def test_nonempty_stratification_is_rejected_until_implemented(self) -> None:
        config = _config()
        config["tasks"]["fall_event_v1"]["stratify_by"] = ["dataset"]

        with self.assertRaisesRegex(SplitConfigError, "stratify"):
            build_fall_risk_splits([], _empty_labels(), config)

    def test_frozen_direct_api_cannot_use_an_unverified_report_hash(self) -> None:
        manifest, labels = _labels_for_all_tasks()

        with self.assertRaisesRegex(SplitDataError, "verified formal validation report"):
            build_fall_risk_splits(manifest, labels, _config(frozen=True))
        with self.assertRaises(TypeError):
            build_fall_risk_splits(
                manifest,
                labels,
                _config(frozen=True),
                validation_report_sha256="f" * 64,
            )

    def test_non_video_functional_proxy_asset_is_eligible(self) -> None:
        functional_asset = _manifest(
            "gstride-functional-table",
            subject_id="gstride-subject",
            asset_type="tabular_measurements",
        )
        functional_asset.pop("video_id")
        functional_label = _label(
            "gstride-functional-table", task_type="functional_proxy"
        )
        functional_label.pop("video_id")
        labels = _empty_labels()
        labels["functional_proxy_v1"] = [functional_label]

        artifact = build_fall_risk_splits(
            [functional_asset], labels, _config()
        )["functional_proxy_v1"]

        self.assertEqual(artifact["metadata"]["status"], "ready")
        self.assertEqual(
            [row["asset_id"] for row in artifact["assignments"]],
            ["gstride-functional-table"],
        )
        self.assertIsNone(artifact["assignments"][0]["video_id"])


class FallRiskSplitWriteTest(unittest.TestCase):
    def test_writes_stable_files_and_refuses_default_overwrite(self) -> None:
        manifest, labels = _labels_for_all_tasks()
        artifacts = build_fall_risk_splits(manifest, labels, _config())

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first"
            second = root / "second"
            write_split_artifacts(artifacts, first)
            write_split_artifacts(artifacts, second)

            first_files = {
                path.relative_to(first): path.read_bytes()
                for path in first.rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(second): path.read_bytes()
                for path in second.rglob("*")
                if path.is_file()
            }
            self.assertEqual(first_files, second_files)

            with self.assertRaises(FileExistsError):
                write_split_artifacts(artifacts, first)

    def test_frozen_split_cannot_be_rewritten_even_with_development_override(self) -> None:
        manifest, labels = _labels_for_all_tasks()
        frozen = build_fall_risk_splits(manifest, labels, _config())
        for artifact in frozen.values():
            artifact["metadata"]["status"] = "frozen"
        changed = build_fall_risk_splits(manifest, labels, _config(seed="changed"))

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "splits"
            write_split_artifacts(frozen, output)

            with self.assertRaises(FrozenSplitError):
                write_split_artifacts(changed, output, overwrite_development=True)

    def test_pair_write_failure_leaves_no_partial_task_directory(self) -> None:
        manifest, labels = _labels_for_all_tasks()
        artifacts = build_fall_risk_splits(manifest, labels, _config())

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "splits"
            with mock.patch(
                "elderly_monitoring.modules.fall_risk.splits._write_atomic",
                side_effect=[None, OSError("simulated second-file failure")],
            ):
                with self.assertRaisesRegex(OSError, "second-file"):
                    write_split_artifacts(artifacts, output)

            self.assertFalse((output / "fall_event_v1").exists())

    def test_second_task_commit_failure_removes_every_new_task(self) -> None:
        manifest, labels = _labels_for_all_tasks()
        artifacts = build_fall_risk_splits(manifest, labels, _config())

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "splits"
            real_replace = os.replace
            task_commit_count = 0

            def fail_second_task_commit(source: object, destination: object) -> None:
                nonlocal task_commit_count
                destination_path = Path(destination)
                if (
                    destination_path.parent == output
                    and destination_path.name in TASK_TYPES
                ):
                    task_commit_count += 1
                    if task_commit_count == 2:
                        raise OSError("simulated second-task commit failure")
                real_replace(source, destination)

            with mock.patch(
                "elderly_monitoring.modules.fall_risk.splits.os.replace",
                side_effect=fail_second_task_commit,
            ):
                with self.assertRaisesRegex(OSError, "second-task"):
                    write_split_artifacts(artifacts, output)

            self.assertEqual(task_commit_count, 2)
            self.assertEqual(list(output.iterdir()), [])

    def test_third_task_commit_failure_restores_every_previous_task(self) -> None:
        manifest, labels = _labels_for_all_tasks()
        original = build_fall_risk_splits(manifest, labels, _config(seed="original"))
        replacement = build_fall_risk_splits(
            manifest, labels, _config(seed="replacement")
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "splits"
            write_split_artifacts(original, output)
            before = {
                path.relative_to(output): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }
            real_replace = os.replace
            task_commit_count = 0

            def fail_third_task_commit(source: object, destination: object) -> None:
                nonlocal task_commit_count
                destination_path = Path(destination)
                if (
                    destination_path.parent == output
                    and destination_path.name in TASK_TYPES
                ):
                    task_commit_count += 1
                    if task_commit_count == 3:
                        raise OSError("simulated third-task commit failure")
                real_replace(source, destination)

            with mock.patch(
                "elderly_monitoring.modules.fall_risk.splits.os.replace",
                side_effect=fail_third_task_commit,
            ):
                with self.assertRaisesRegex(OSError, "third-task"):
                    write_split_artifacts(
                        replacement, output, overwrite_development=True
                    )

            after = {
                path.relative_to(output): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }
            self.assertEqual(task_commit_count, 6)
            self.assertEqual(after, before)
            self.assertEqual(
                {path.name for path in output.iterdir()}, set(TASK_TYPES)
            )

    def test_existing_write_lock_is_preserved_and_blocks_all_writes(self) -> None:
        manifest, labels = _labels_for_all_tasks()
        artifacts = build_fall_risk_splits(manifest, labels, _config())

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "splits"
            output.mkdir()
            lock_path = output / ".fall-risk-splits.write.lock"
            lock_payload = b'{"pid":999999,"owner":"stale-or-active"}\n'
            lock_path.write_bytes(lock_payload)

            with self.assertRaisesRegex(SplitDataError, "write lock already exists"):
                write_split_artifacts(artifacts, output)

            self.assertEqual(lock_path.read_bytes(), lock_payload)
            self.assertFalse((output / "fall_event_v1").exists())

    def test_builds_from_files_and_keeps_task_label_filters_independent(self) -> None:
        manifest, _ = _labels_for_all_tasks()
        event_labels = [
            _label("fall-ok", event_type="fall"),
            _label("near-ok", event_type="near_fall"),
        ]
        risk_labels = [
            _label("functional-ok", task_type="functional_proxy"),
            _label("longitudinal-ok", task_type="longitudinal_baseline"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.jsonl"
            annotation_dir = root / "annotations"
            annotation_dir.mkdir()
            config_path = root / "splits.yaml"
            output_dir = root / "output"
            _write_jsonl(manifest_path, manifest)
            _write_jsonl(annotation_dir / "event_labels.jsonl", event_labels)
            _write_jsonl(annotation_dir / "risk_labels.jsonl", risk_labels)
            config_path.write_text(
                yaml.safe_dump(_config(), sort_keys=False),
                encoding="utf-8",
            )

            artifacts = build_splits_from_files(
                manifest_path=manifest_path,
                annotations_dir=annotation_dir,
                config_path=config_path,
                output_dir=output_dir,
            )

            self.assertTrue((output_dir / "fall_event_v1" / "split.json").is_file())
            self.assertEqual(
                artifacts["fall_event_v1"]["metadata"]["manifest_sha256"],
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                {
                    row["sample_id"]
                    for row in artifacts["fall_event_v1"]["assignments"]
                },
                {"fall-ok"},
            )
            self.assertEqual(
                {
                    row["sample_id"]
                    for row in artifacts["near_fall_event_v1"]["assignments"]
                },
                {"near-ok"},
            )
            self.assertEqual(
                {
                    row["sample_id"]
                    for row in artifacts["functional_proxy_v1"]["assignments"]
                },
                {"functional-ok"},
            )
            self.assertEqual(
                {
                    row["sample_id"]
                    for row in artifacts["longitudinal_baseline_v1"]["assignments"]
                },
                {"longitudinal-ok"},
            )

    def test_missing_label_file_has_a_specific_blocker(self) -> None:
        manifest = [_manifest("fall-ok")]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.jsonl"
            annotation_dir = root / "annotations"
            annotation_dir.mkdir()
            config_path = root / "splits.yaml"
            output_dir = root / "output"
            _write_jsonl(manifest_path, manifest)
            _write_jsonl(annotation_dir / "event_labels.jsonl", [_label("fall-ok")])
            config_path.write_text(
                yaml.safe_dump(_config(), sort_keys=False), encoding="utf-8"
            )

            artifacts = build_splits_from_files(
                manifest_path=manifest_path,
                annotations_dir=annotation_dir,
                config_path=config_path,
                output_dir=output_dir,
            )

        for split_name in ("functional_proxy_v1", "longitudinal_baseline_v1"):
            self.assertEqual(artifacts[split_name]["metadata"]["status"], "blocked")
            self.assertEqual(
                artifacts[split_name]["metadata"]["blockers"][0]["code"],
                "missing_label_file",
            )

    def test_frozen_file_build_requires_a_clean_bound_formal_report(self) -> None:
        manifest, labels = _labels_for_all_tasks()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, report = _write_formal_split_fixture(
                Path(tmpdir), manifest=manifest, labels=labels
            )

            artifacts = build_splits_from_files(**paths)

            report_sha256 = hashlib.sha256(
                paths["validation_report_path"].read_bytes()
            ).hexdigest()
            for artifact in artifacts.values():
                self.assertEqual(artifact["metadata"]["status"], "frozen")
                self.assertEqual(
                    artifact["metadata"]["validation_report_sha256"],
                    report_sha256,
                )
            self.assertEqual(report["counts"]["errors"], 0)
            self.assertEqual(report["counts"]["blockers"], 0)

    def test_frozen_report_binds_every_validator_input_hash(self) -> None:
        manifest, labels = _labels_for_all_tasks()
        fields = (
            "manifest",
            "action_labels",
            "event_labels",
            "risk_labels",
            "subject_profiles",
            "review_log",
            "validation_config",
        )
        for field in fields:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmpdir:
                paths, report = _write_formal_split_fixture(
                    Path(tmpdir), manifest=manifest, labels=labels
                )
                report["input_sha256"][field] = "0" * 64
                _write_json(paths["validation_report_path"], report)

                with self.assertRaisesRegex(SplitDataError, field):
                    build_splits_from_files(**paths)

    def test_frozen_report_rejects_invalid_or_blocked_results(self) -> None:
        manifest, labels = _labels_for_all_tasks()
        mutations = {
            "audit mode": lambda report: report.update(mode="audit"),
            "invalid": lambda report: report.update(valid=False),
            "errors": lambda report: report["counts"].update(errors=1),
            "blockers": lambda report: report["counts"].update(blockers=1),
            "blocking issue": lambda report: report["issues"].append(
                {"severity": "blocker", "code": "manual_blocker"}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmpdir:
                paths, report = _write_formal_split_fixture(
                    Path(tmpdir), manifest=manifest, labels=labels
                )
                mutate(report)
                _write_json(paths["validation_report_path"], report)

                with self.assertRaisesRegex(
                    SplitDataError, "valid formal report|error or blocker"
                ):
                    build_splits_from_files(**paths)

    def test_frozen_split_rejects_a_hash_bound_but_weakened_validator_config(self) -> None:
        manifest, labels = _labels_for_all_tasks()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, report = _write_formal_split_fixture(
                Path(tmpdir), manifest=manifest, labels=labels
            )
            unsafe = dict(DEFAULT_CONFIG)
            unsafe["minimum_independent_reviewers"] = 1
            paths["validation_config_path"].write_text(
                yaml.safe_dump(unsafe, sort_keys=False), encoding="utf-8"
            )
            report["input_sha256"]["validation_config"] = hashlib.sha256(
                paths["validation_config_path"].read_bytes()
            ).hexdigest()
            _write_json(paths["validation_report_path"], report)

            with self.assertRaisesRegex(SplitDataError, "unsafe"):
                build_splits_from_files(**paths)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _write_formal_split_fixture(
    root: Path,
    *,
    manifest: list[dict],
    labels: dict[str, list[dict]],
) -> tuple[dict[str, Path], dict]:
    annotation_dir = root / "annotations"
    annotation_dir.mkdir()
    manifest_path = root / "manifest.jsonl"
    split_config_path = root / "splits.yaml"
    validation_config_path = root / "validation.yaml"
    validation_report_path = root / "validation-report.json"
    output_dir = root / "output"
    event_labels = labels["fall_event_v1"] + labels["near_fall_event_v1"]
    risk_labels = labels["functional_proxy_v1"] + labels["longitudinal_baseline_v1"]
    input_paths = {
        "manifest": manifest_path,
        "action_labels": annotation_dir / "action_labels.jsonl",
        "event_labels": annotation_dir / "event_labels.jsonl",
        "risk_labels": annotation_dir / "risk_labels.jsonl",
        "subject_profiles": annotation_dir / "subject_profiles.json",
        "review_log": annotation_dir / "annotation_review_log.jsonl",
        "validation_config": validation_config_path,
    }
    _write_jsonl(manifest_path, manifest)
    _write_jsonl(input_paths["action_labels"], [])
    _write_jsonl(input_paths["event_labels"], event_labels)
    _write_jsonl(input_paths["risk_labels"], risk_labels)
    _write_json(input_paths["subject_profiles"], {"schema_version": "1.0", "subjects": []})
    _write_jsonl(input_paths["review_log"], [])
    validation_config_path.write_text(
        yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False), encoding="utf-8"
    )
    split_config_path.write_text(
        yaml.safe_dump(_config(frozen=True), sort_keys=False), encoding="utf-8"
    )
    report = {
        "schema_version": "fall-risk-label-validation-report-v1",
        "mode": "formal",
        "valid": True,
        "input_sha256": {
            field: hashlib.sha256(path.read_bytes()).hexdigest()
            for field, path in input_paths.items()
        },
        "counts": {"errors": 0, "blockers": 0, "warnings": 0},
        "issues": [],
    }
    _write_json(validation_report_path, report)
    paths = {
        "manifest_path": manifest_path,
        "annotations_dir": annotation_dir,
        "config_path": split_config_path,
        "output_dir": output_dir,
        "validation_report_path": validation_report_path,
        "validation_config_path": validation_config_path,
    }
    return paths, report


if __name__ == "__main__":
    unittest.main()
