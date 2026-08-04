from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from unittest import mock

import yaml

from elderly_monitoring.modules.mental_health.wandering.datasets import (
    load_trajectory_jsonl,
)
from elderly_monitoring.modules.mental_health.wandering.manifests import (
    SPLIT_MANIFEST_SCHEMA_VERSION,
    SplitManifest,
)
from elderly_monitoring.modules.mental_health.wandering.schemas import (
    CoordinateSystem,
    PatternLabel,
    TrajectorySample,
    TRAJECTORY_SAMPLE_SCHEMA_VERSION,
)
from elderly_monitoring.modules.mental_health.wandering.splits import (
    InputHashMismatchError,
    SplitAccessError,
    SplitConfigError,
    SplitDataError,
    assign_smartcare_development,
    assign_wandering_patterns,
    audit_exact_duplicates,
    build_wandering_split_from_files,
    derive_smartcare_day,
    extract_sealed_sample_ids,
    inherit_parent_split,
    load_development_partition_ids,
    load_split_config,
    procrustes_shape_distance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/data/wandering_split_v1.yaml"
OUTPUT_FILENAMES = (
    "split.json",
    "split.sha256",
    "assignments.jsonl",
    "near_neighbor_audit.json",
    "split_report.json",
)
EXPECTED_INPUT_HASHES = {
    "wandering_patterns_manifest": (
        "data/processed/wandering/wandering_patterns/manifest.json",
        "04bb88849eddd15ef12de7883612f3b03a20004dfbcd9684e1ef059f06875b9e",
    ),
    "wandering_patterns_samples": (
        "data/processed/wandering/wandering_patterns/samples.jsonl",
        "498a0c39af27240bd38ab82367fdc1a13373ba3a22e8fede915419fea54e6ee7",
    ),
    "smartcare_manifest": (
        "data/processed/wandering/smartcare/manifest.json",
        "c74bb729cf2c353311fef2013ae6abf72a59f70dab5a1969ffb48a37262736bc",
    ),
    "smartcare_train_pool": (
        "data/processed/wandering/smartcare/train_pool.jsonl",
        "01c5e3b8ecabd594837e5154ab64e52a1d364bcc15096a8193e8b627aa6fb780",
    ),
    "smartcare_official_validation": (
        "data/processed/wandering/smartcare/official_validation.jsonl",
        "78ac0568089ca1aa5aedd4e38f77d37bc5d0d40c3c1ba14d51ceaa4a0ead7b40",
    ),
    "step2_human_review": (
        "reports/mental_health/wandering_step2/HUMAN_REVIEW.md",
        "9577803dde1a4a9cb6b97ad09bb4fbc573817ba9d1e8954d44ad22d312cb0cd0",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def _sample(
    sample_id: str,
    *,
    source_dataset: str = "wandering_patterns",
    source_group_id: str | None = None,
    pattern_label: PatternLabel = PatternLabel.DIRECT,
    points: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
    ),
) -> TrajectorySample:
    binary_label = 0 if pattern_label is PatternLabel.DIRECT else 1
    if pattern_label is PatternLabel.UNKNOWN:
        binary_label = 1
    if source_dataset == "smartcare" and points == (
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
    ):
        points = ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0))
    return TrajectorySample(
        schema_version=TRAJECTORY_SAMPLE_SCHEMA_VERSION,
        sample_id=sample_id,
        source_dataset=source_dataset,
        source_group_id=source_group_id,
        coordinate_system=(
            CoordinateSystem.SOURCE_NATIVE
            if source_dataset == "wandering_patterns"
            else CoordinateSystem.IMAGE_NORMALIZED
        ),
        points=points,
        point_times_sec=None,
        point_mask=(1,) * len(points),
        binary_label=binary_label,
        pattern_label=pattern_label,
        quality_flags=("time_unavailable",),
        source_path="raw/source.json",
        source_sha256="a" * 64,
    )


class WanderingSplitConfigTest(unittest.TestCase):
    def test_config_freezes_the_six_inputs_seed_policy_dates_and_counts(self) -> None:
        config = load_split_config(CONFIG_PATH)

        self.assertEqual(config["schema_version"], "wandering-split-config-v1")
        self.assertEqual(config["split_version"], SPLIT_MANIFEST_SCHEMA_VERSION)
        self.assertEqual(config["seed"], 20260731)
        self.assertEqual(config["group_policy"], "source_specific_v1")
        self.assertEqual(
            config["smartcare"]["validation_dates"],
            ["2020-09-20", "2020-10-07"],
        )
        self.assertEqual(
            config["wandering_patterns"]["per_class_counts"],
            {"train": 280, "validation": 60, "test": 60},
        )
        self.assertEqual(
            config["smartcare"]["expected_counts"],
            {"train": 152, "validation": 38, "sealed_external_test": 20},
        )
        self.assertEqual(
            set(config["inputs"]),
            set(EXPECTED_INPUT_HASHES),
        )
        for role, (relative_path, expected_hash) in EXPECTED_INPUT_HASHES.items():
            with self.subTest(role=role):
                self.assertEqual(config["inputs"][role]["path"], relative_path)
                self.assertEqual(config["inputs"][role]["sha256"], expected_hash)
                self.assertEqual(_sha256(PROJECT_ROOT / relative_path), expected_hash)

    def test_seed_date_rule_and_unknown_config_changes_fail_closed(self) -> None:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        mutations = {
            "seed": lambda value: value.update(seed=1),
            "dates": lambda value: value["smartcare"].update(
                validation_dates=["2020-09-20"]
            ),
            "group policy": lambda value: value.update(group_policy="sample_random"),
            "unknown key": lambda value: value.update(training_split_ratio=0.8),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmpdir:
                changed = copy.deepcopy(raw)
                mutate(changed)
                path = Path(tmpdir) / "changed.yaml"
                path.write_text(
                    yaml.safe_dump(changed, sort_keys=False), encoding="utf-8"
                )
                with self.assertRaises(SplitConfigError):
                    load_split_config(path)


class WanderingSourceSpecificAssignmentTest(unittest.TestCase):
    def test_wandering_patterns_uses_the_fixed_per_class_hash_order(self) -> None:
        samples = []
        for label in (
            PatternLabel.DIRECT,
            PatternLabel.PACING,
            PatternLabel.LAPPING,
            PatternLabel.RANDOM,
        ):
            samples.extend(
                _sample(f"{label.value}-{index}", pattern_label=label)
                for index in range(5)
            )

        assignments = assign_wandering_patterns(
            samples,
            seed=20260731,
            per_class_counts={"train": 2, "validation": 1, "test": 2},
        )

        for label in (
            PatternLabel.DIRECT,
            PatternLabel.PACING,
            PatternLabel.LAPPING,
            PatternLabel.RANDOM,
        ):
            ids = [f"{label.value}-{index}" for index in range(5)]
            expected_order = sorted(
                ids,
                key=lambda sample_id: hashlib.sha256(
                    f"20260731:wandering_patterns:{sample_id}".encode("utf-8")
                ).hexdigest(),
            )
            self.assertEqual(
                [assignments[sample_id] for sample_id in expected_order],
                ["train", "train", "validation", "test", "test"],
            )

    def test_smartcare_uses_strict_calendar_day_proxy_groups(self) -> None:
        samples = (
            _sample(
                "smartcare-train-a",
                source_dataset="smartcare",
                source_group_id="19/09/2020 23:00",
                pattern_label=PatternLabel.UNKNOWN,
            ),
            _sample(
                "smartcare-validation-a",
                source_dataset="smartcare",
                source_group_id="20/09/2020 00:00",
                pattern_label=PatternLabel.UNKNOWN,
            ),
            _sample(
                "smartcare-validation-b",
                source_dataset="smartcare",
                source_group_id="07/10/2020 21:00",
                pattern_label=PatternLabel.UNKNOWN,
            ),
        )

        assignments = assign_smartcare_development(
            samples,
            validation_dates=("2020-09-20", "2020-10-07"),
        )

        self.assertEqual(derive_smartcare_day("20/09/2020 13:00"), "2020-09-20")
        self.assertEqual(assignments["smartcare-train-a"], "train")
        self.assertEqual(assignments["smartcare-validation-a"], "validation")
        self.assertEqual(assignments["smartcare-validation-b"], "validation")
        for invalid in (
            "2020-09-20 13:00",
            "20/09/2020",
            "31/02/2020 13:00",
            "20/09/2020 13:00 extra",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(SplitDataError):
                    derive_smartcare_day(invalid)

    def test_duplicate_reverse_and_cross_label_conflicts_fail_closed(self) -> None:
        first = _sample("first")
        exact = replace(first, sample_id="exact")
        reversed_copy = replace(
            first,
            sample_id="reverse",
            points=tuple(reversed(first.points)),
        )
        with self.assertRaisesRegex(SplitDataError, "duplicate.*partition"):
            audit_exact_duplicates(
                (first, exact, reversed_copy),
                {"first": "train", "exact": "validation", "reverse": "train"},
                source_name="wandering_patterns",
            )

        with self.assertRaisesRegex(SplitDataError, "duplicate.*partition"):
            audit_exact_duplicates(
                (first, reversed_copy),
                {"first": "train", "reverse": "validation"},
                source_name="wandering_patterns",
            )

        conflicting_label = replace(
            first,
            sample_id="conflicting-label",
            pattern_label=PatternLabel.PACING,
            binary_label=1,
        )
        with self.assertRaisesRegex(SplitDataError, "label conflict"):
            audit_exact_duplicates(
                (first, conflicting_label),
                {"first": "train", "conflicting-label": "train"},
                source_name="wandering_patterns",
            )

    def test_shape_distance_is_rotation_mirror_and_reversal_invariant(self) -> None:
        original = ((0.0, 0.0), (1.0, 0.0), (1.0, 2.0), (3.0, 2.0))
        rotated = tuple((-y + 10.0, x - 4.0) for x, y in original)
        mirrored = tuple((-x, y) for x, y in original)

        self.assertAlmostEqual(procrustes_shape_distance(original, rotated), 0.0)
        self.assertAlmostEqual(procrustes_shape_distance(original, mirrored), 0.0)
        self.assertAlmostEqual(
            procrustes_shape_distance(original, tuple(reversed(rotated))), 0.0
        )

    def test_derived_views_inherit_the_parent_partition(self) -> None:
        split = {
            "train": ["parent-train"],
            "validation": ["parent-validation"],
            "test": [],
            "sealed_external_test": ["parent-sealed"],
        }
        self.assertEqual(inherit_parent_split("parent-train", split), "train")
        self.assertEqual(
            inherit_parent_split(
                "parent-validation", split, requested_split="validation"
            ),
            "validation",
        )
        with self.assertRaisesRegex(SplitDataError, "inherit"):
            inherit_parent_split("parent-train", split, requested_split="test")
        with self.assertRaisesRegex(SplitDataError, "unknown parent"):
            inherit_parent_split("missing", split)


class WanderingSplitInputDriftTest(unittest.TestCase):
    def test_any_input_byte_drift_stops_before_output_creation(self) -> None:
        for changed_role in EXPECTED_INPUT_HASHES:
            with self.subTest(changed_role=changed_role), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir) / "repo"
                root.mkdir()
                shutil.copy2(CONFIG_PATH, root / "config.yaml")
                for relative_path, _ in EXPECTED_INPUT_HASHES.values():
                    destination = root / relative_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(PROJECT_ROOT / relative_path, destination)
                drift_path = root / EXPECTED_INPUT_HASHES[changed_role][0]
                drift_path.write_bytes(drift_path.read_bytes() + b" ")
                output = Path(tmpdir) / "output"

                with self.assertRaisesRegex(InputHashMismatchError, changed_role):
                    build_wandering_split_from_files(
                        config_path=root / "config.yaml",
                        project_root=root,
                        output_dir=output,
                    )
                self.assertFalse(output.exists())

    def test_official_sealed_reader_extracts_only_ids_after_file_hash_gate(self) -> None:
        official_path = PROJECT_ROOT / EXPECTED_INPUT_HASHES[
            "smartcare_official_validation"
        ][0]
        with mock.patch(
            "elderly_monitoring.modules.mental_health.wandering.splits.json.loads",
            side_effect=AssertionError("sealed reader must not decode coordinates/labels"),
        ):
            ids = extract_sealed_sample_ids(official_path)
        self.assertEqual(len(ids), 20)
        self.assertEqual(len(set(ids)), 20)


class WanderingSplitRealArtifactAcceptanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        root = Path(cls.tempdir.name)
        cls.first = root / "first"
        cls.second = root / "second"
        cls.input_hashes_before = {
            role: _sha256(PROJECT_ROOT / relative_path)
            for role, (relative_path, _) in EXPECTED_INPUT_HASHES.items()
        }
        build_wandering_split_from_files(
            config_path=CONFIG_PATH,
            project_root=PROJECT_ROOT,
            output_dir=cls.first,
        )
        build_wandering_split_from_files(
            config_path=CONFIG_PATH,
            project_root=PROJECT_ROOT,
            output_dir=cls.second,
        )
        cls.split_record = json.loads(
            (cls.first / "split.json").read_text(encoding="utf-8")
        )
        cls.split_manifest = SplitManifest.from_dict(cls.split_record)
        cls.assignments = [
            json.loads(line)
            for line in (cls.first / "assignments.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        cls.audit = json.loads(
            (cls.first / "near_neighbor_audit.json").read_text(encoding="utf-8")
        )
        cls.report = json.loads(
            (cls.first / "split_report.json").read_text(encoding="utf-8")
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def test_two_fresh_builds_are_byte_identical_and_existing_output_is_refused(self) -> None:
        self.assertEqual(
            {name: (self.first / name).read_bytes() for name in OUTPUT_FILENAMES},
            {name: (self.second / name).read_bytes() for name in OUTPUT_FILENAMES},
        )
        with self.assertRaises(FileExistsError):
            build_wandering_split_from_files(
                config_path=CONFIG_PATH,
                project_root=PROJECT_ROOT,
                output_dir=self.first,
            )

    def test_all_1810_samples_are_assigned_once_with_fixed_partition_counts(self) -> None:
        ids = [row["sample_id"] for row in self.assignments]
        self.assertEqual(len(ids), 1810)
        self.assertEqual(len(set(ids)), 1810)
        self.assertEqual(
            Counter(row["split"] for row in self.assignments),
            {
                "train": 1272,
                "validation": 278,
                "test": 240,
                "sealed_external_test": 20,
            },
        )
        self.assertEqual(set(ids), set(self.split_manifest.all_sample_ids))
        partitions = (
            set(self.split_manifest.train),
            set(self.split_manifest.validation),
            set(self.split_manifest.test),
            set(self.split_manifest.sealed_external_test),
        )
        for index, left in enumerate(partitions):
            for right in partitions[index + 1 :]:
                self.assertTrue(left.isdisjoint(right))

    def test_wp_and_smartcare_counts_and_group_boundaries_are_exact(self) -> None:
        wp = [row for row in self.assignments if row["source_name"] == "wandering_patterns"]
        smartcare_dev = [
            row
            for row in self.assignments
            if row["source_name"] == "smartcare"
            and row["input_role"] == "train_pool"
        ]
        for label in ("direct", "pacing", "lapping", "random"):
            self.assertEqual(
                Counter(
                    row["split"] for row in wp if row["pattern_label"] == label
                ),
                {"train": 280, "validation": 60, "test": 60},
            )
        self.assertEqual(
            Counter(row["split"] for row in smartcare_dev),
            {"train": 152, "validation": 38},
        )
        self.assertEqual(
            Counter(
                (row["split"], row["binary_label"])
                for row in smartcare_dev
            ),
            {
                ("train", 0): 73,
                ("train", 1): 79,
                ("validation", 0): 17,
                ("validation", 1): 21,
            },
        )
        day_partitions: dict[str, set[str]] = defaultdict(set)
        for row in smartcare_dev:
            day_partitions[row["calendar_day"]].add(row["split"])
            self.assertEqual(
                row["leakage_group_id"], f"smartcare_day:{row['calendar_day']}"
            )
            self.assertFalse(row["pattern_supervision_eligible"])
            self.assertEqual(row["pattern_label"], "unknown")
        self.assertTrue(all(len(values) == 1 for values in day_partitions.values()))

    def test_official_twenty_are_only_sealed_and_never_decoded_into_assignments(self) -> None:
        official_ids = set(
            extract_sealed_sample_ids(
                PROJECT_ROOT
                / EXPECTED_INPUT_HASHES["smartcare_official_validation"][0]
            )
        )
        sealed_ids = set(self.split_manifest.sealed_external_test)
        self.assertEqual(official_ids, sealed_ids)
        self.assertEqual(len(sealed_ids), 20)
        for partition in (
            self.split_manifest.train,
            self.split_manifest.validation,
            self.split_manifest.test,
        ):
            self.assertTrue(official_ids.isdisjoint(partition))
        sealed_rows = [
            row for row in self.assignments if row["split"] == "sealed_external_test"
        ]
        self.assertTrue(
            all(
                "points" not in row
                and "binary_label" not in row
                and "pattern_label" not in row
                and not row["pattern_supervision_eligible"]
                for row in sealed_rows
            )
        )

    def test_development_loader_defaults_to_train_and_rejects_sealed(self) -> None:
        default_ids = load_development_partition_ids(self.first / "split.json")
        self.assertEqual(default_ids, self.split_manifest.train)
        with self.assertRaises(SplitAccessError):
            load_development_partition_ids(
                self.first / "split.json", partition="validation"
            )
        self.assertEqual(
            load_development_partition_ids(
                self.first / "split.json",
                partition="validation",
                allow_evaluation_partition=True,
            ),
            self.split_manifest.validation,
        )
        with self.assertRaisesRegex(SplitAccessError, "sealed_external_test"):
            load_development_partition_ids(
                self.first / "split.json",
                partition="sealed_external_test",
                allow_evaluation_partition=True,
            )

    def test_near_neighbor_audit_reproduces_the_frozen_read_only_precheck(self) -> None:
        thresholds = self.audit["wandering_patterns"]["near_neighbors"]
        self.assertEqual(thresholds["lt_0_05"]["total_pairs"], 9061)
        self.assertEqual(thresholds["lt_0_05"]["same_label_pairs"], 9055)
        self.assertEqual(thresholds["lt_0_05"]["cross_label_pairs"], 6)
        self.assertGreater(thresholds["lt_0_05"]["cross_partition_pairs"], 0)
        self.assertIn("lt_0_02", thresholds)
        self.assertNotIn("sealed_external_test", json.dumps(self.audit))

    def test_report_uses_only_the_supported_benchmark_and_leakage_wording(self) -> None:
        self.assertEqual(
            self.report["evaluation_scopes"]["wandering_patterns_test"],
            "public_shape_benchmark",
        )
        self.assertEqual(
            self.report["group_guarantees"]["smartcare_development"],
            "calendar_day_proxy_groups_are_disjoint",
        )
        serialized = json.dumps(self.report, ensure_ascii=False).lower()
        for forbidden in (
            "人员级无泄漏",
            "跨人员无泄漏",
            "smartcare 无泄漏",
            "person-level leakage-free",
            "participant-level leakage-free",
            "cross-person generalization",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, serialized)

    def test_input_config_split_report_and_artifact_hashes_are_complete(self) -> None:
        expected_source_hashes = {
            role: expected_hash
            for role, (_, expected_hash) in EXPECTED_INPUT_HASHES.items()
        }
        expected_source_hashes["split_config"] = _sha256(CONFIG_PATH)
        self.assertEqual(self.split_record["source_hashes"], expected_source_hashes)
        self.assertEqual(
            (self.first / "split.sha256").read_text(encoding="ascii"),
            f"{self.split_manifest.split_sha256}\n",
        )
        artifact_hashes = self.report["artifact_hashes"]
        for name in OUTPUT_FILENAMES[:-1]:
            with self.subTest(name=name):
                self.assertEqual(artifact_hashes[name], _sha256(self.first / name))
        report_payload = dict(self.report)
        report_payload_sha256 = report_payload.pop("report_payload_sha256")
        self.assertEqual(report_payload_sha256, _canonical_sha256(report_payload))

    def test_machine_outputs_have_no_absolute_paths_or_wall_clock(self) -> None:
        for name in OUTPUT_FILENAMES:
            payload = (self.first / name).read_text(encoding="utf-8")
            with self.subTest(name=name):
                self.assertNotIn(str(PROJECT_ROOT), payload)
                self.assertNotRegex(payload, r"[A-Za-z]:[/\\]")
                self.assertNotIn("generated_at", payload)
                self.assertNotIn("created_at", payload)

    def test_step2_inputs_remain_byte_identical(self) -> None:
        after = {
            role: _sha256(PROJECT_ROOT / relative_path)
            for role, (relative_path, _) in EXPECTED_INPUT_HASHES.items()
        }
        self.assertEqual(after, self.input_hashes_before)


if __name__ == "__main__":
    unittest.main()
