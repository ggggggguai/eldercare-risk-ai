from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import yaml

from elderly_monitoring.modules.mental_health.wandering.preprocessing_bundle import (
    BUNDLE_MODE_DEVELOPMENT,
    BUNDLE_MODE_FROZEN_WP_TEST,
    BundleAccessError,
    BundleIntegrityError,
    _commit_new_output_directory,
    _load_canonical_json,
    _load_canonical_jsonl,
    _validate_assignments_and_records,
    _validate_preprocessed_record,
    load_preprocessing_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/modules/wandering_rf_v1.yaml"


class WanderingPreprocessingBundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.development = load_preprocessing_bundle(
            rf_config_path=CONFIG_PATH,
            project_root=PROJECT_ROOT,
            mode=BUNDLE_MODE_DEVELOPMENT,
        )

    def test_frozen_hashes_counts_and_development_partition_view(self) -> None:
        bundle = self.development

        self.assertEqual(bundle.total_record_count, 1790)
        self.assertEqual(bundle.ready_count, 1775)
        self.assertEqual(bundle.unavailable_count, 15)
        self.assertEqual(len(bundle.records_for_features()), 1535)
        self.assertEqual(
            Counter(row["split"] for row in bundle.records_for_features()),
            Counter({"train": 1257, "validation": 278}),
        )
        self.assertTrue(all(row["preprocess_status"] == "ready" for row in bundle.records_for_features()))
        self.assertTrue(all(row["split"] != "test" for row in bundle.records_for_features()))
        self.assertEqual(len(bundle.unavailable_records()), 15)
        self.assertTrue(all(row["split"] == "train" for row in bundle.unavailable_records()))

        with self.assertRaises(BundleAccessError):
            bundle.records_for_split("test")

    def test_frozen_test_mode_exposes_only_wp_test_and_never_official(self) -> None:
        bundle = load_preprocessing_bundle(
            rf_config_path=CONFIG_PATH,
            project_root=PROJECT_ROOT,
            mode=BUNDLE_MODE_FROZEN_WP_TEST,
        )

        rows = bundle.records_for_features()
        self.assertEqual(len(rows), 240)
        self.assertTrue(all(row["split"] == "test" for row in rows))
        self.assertTrue(all(row["source_dataset"] == "wandering_patterns" for row in rows))
        with self.assertRaises(BundleAccessError):
            bundle.records_for_split("train")
        with self.assertRaises(BundleAccessError):
            bundle.records_for_split("sealed_external_test")

    def test_task_cohorts_match_the_frozen_section_2_3_contract(self) -> None:
        development = self.development
        frozen = load_preprocessing_bundle(
            rf_config_path=CONFIG_PATH,
            project_root=PROJECT_ROOT,
            mode=BUNDLE_MODE_FROZEN_WP_TEST,
        )

        four_dev = development.task_records("four_class")
        four_test = frozen.task_records("four_class")
        self.assertEqual(Counter(row["split"] for row in four_dev), Counter(train=1120, validation=240))
        self.assertEqual(Counter(row["pattern_label"] for row in four_dev if row["split"] == "train"), Counter(direct=280, pacing=280, lapping=280, random=280))
        self.assertEqual(Counter(row["pattern_label"] for row in four_dev if row["split"] == "validation"), Counter(direct=60, pacing=60, lapping=60, random=60))
        self.assertEqual(len(four_test), 240)
        self.assertTrue(all(row["source_dataset"] == "wandering_patterns" for row in four_dev + four_test))

        binary_dev = development.task_records("binary")
        binary_test = frozen.task_records("binary")
        self.assertEqual(Counter(row["split"] for row in binary_dev), Counter(train=1257, validation=278))
        self.assertEqual(Counter((row["source_dataset"], row["binary_label"]) for row in binary_dev if row["split"] == "train"), Counter({("wandering_patterns", 0): 280, ("wandering_patterns", 1): 840, ("smartcare", 0): 63, ("smartcare", 1): 74}))
        self.assertEqual(Counter((row["source_dataset"], row["binary_label"]) for row in binary_dev if row["split"] == "validation"), Counter({("wandering_patterns", 0): 60, ("wandering_patterns", 1): 180, ("smartcare", 0): 17, ("smartcare", 1): 21}))
        self.assertEqual(Counter((row["source_dataset"], row["binary_label"]) for row in binary_test), Counter({("wandering_patterns", 0): 60, ("wandering_patterns", 1): 180}))

    def test_config_or_human_gate_hash_drift_fails_closed(self) -> None:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        mutations = (
            lambda value: value["inputs"]["preprocessing_manifest"].update(sha256="0" * 64),
            lambda value: value["inputs"]["near_neighbor_audit"].update(sha256="0" * 64),
            lambda value: value["inputs"]["human_review"].update(sha256="0" * 64),
            lambda value: value["inputs"]["human_review"].update(required_status="pending_human_review"),
            lambda value: value.update(split_sha256="0" * 64),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as tmpdir:
                changed = copy.deepcopy(raw)
                mutate(changed)
                changed_path = Path(tmpdir) / "changed.yaml"
                changed_path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
                with self.assertRaises(BundleIntegrityError):
                    load_preprocessing_bundle(
                        rf_config_path=changed_path,
                        project_root=PROJECT_ROOT,
                        mode=BUNDLE_MODE_DEVELOPMENT,
                    )

    def test_record_schema_sort_label_shape_and_finite_checks_fail(self) -> None:
        ready = copy.deepcopy(self.development.records_for_features()[0])
        mutations = (
            lambda row: row.update(schema_version="wrong"),
            lambda row: row.update(parent_sample_id="other"),
            lambda row: row.update(split="test"),
            lambda row: row.update(binary_label=9),
            lambda row: row.update(pattern_supervision_eligible=True, source_dataset="smartcare"),
            lambda row: row.update(shape_normalized_points=[[0.0, 0.0]]),
            lambda row: row["raw_features"][0].__setitem__(0, float("nan")),
            lambda row: row.update(split_sha256="0" * 64),
            lambda row: row.update(preprocessing_config_sha256="0" * 64),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = copy.deepcopy(ready)
                mutate(changed)
                with self.assertRaises(BundleIntegrityError):
                    _validate_preprocessed_record(
                        changed,
                        expected_split_sha256=self.development.split_sha256,
                        expected_preprocessing_config_sha256=self.development.preprocessing_config_sha256,
                    )

    def test_duplicate_unknown_or_unsorted_batch_ids_fail_closed(self) -> None:
        assignments = _load_canonical_jsonl(
            PROJECT_ROOT / "data/splits/mental_health/wandering/v1/assignments.jsonl"
        )
        split = _load_canonical_json(
            PROJECT_ROOT / "data/splits/mental_health/wandering/v1/split.json"
        )
        original = list(self.development._records)
        variants = []
        duplicate = list(original)
        duplicate[-1] = dict(duplicate[0])
        variants.append(duplicate)
        unknown = list(original)
        unknown[-1] = {**unknown[-1], "sample_id": "unknown_sample", "parent_sample_id": "unknown_sample"}
        variants.append(unknown)
        unsorted = list(original)
        unsorted[0], unsorted[1] = unsorted[1], unsorted[0]
        variants.append(unsorted)
        for rows in variants:
            with self.subTest(last_id=rows[-1]["sample_id"]), self.assertRaises(BundleIntegrityError):
                _validate_assignments_and_records(
                    assignments=assignments,
                    records=rows,
                    split=split,
                    expected_preprocessing_config_sha256=self.development.preprocessing_config_sha256,
                )

    def test_existing_output_and_failed_atomic_commit_leave_no_partial_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                _commit_new_output_directory(existing, {"a.json": b"{}\n"})

            output = root / "output"
            with self.assertRaises(TypeError):
                _commit_new_output_directory(output, {"a.json": object()})  # type: ignore[dict-item]
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".output.*.tmp")), [])

    def test_config_binds_all_step_four_machine_files_and_current_bytes(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        expected = {
            "preprocessing_config": "5b69243337cddeeec8beed4f081330b06c47eadeaee06dc3b3e4ba026f36ff45",
            "preprocessing_samples": "323ec1258a03ed2daab1e84e3291edc62644c68b1c292dc93a332c01faa08cda",
            "preprocessing_feature_stats": "249ad8383043dba1d95452533ddc6004dfecc2c9e99be43f788dcfcbfb434c8d",
            "preprocessing_report": "d600fbd16efdc8c2b6e89db8c35c965fbb8322f099c464e67600abff5d8804e8",
            "preprocessing_manifest": "242072bdfe4b969d320a091ecc499aff445c30a937dfb1e6739ad869aa938593",
            "near_neighbor_audit": "72d6f9f10f092fcadcd81f71e424d51f3b26938d6bfccbdcecced5852f851d10",
            "human_review": "f017f207053fb8948e217bd5b72d6bdf13d19a31ccd635f48b1f48915d60aa8c",
        }
        for role, digest in expected.items():
            descriptor = config["inputs"][role]
            self.assertEqual(descriptor["sha256"], digest)
            path = PROJECT_ROOT / descriptor["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
