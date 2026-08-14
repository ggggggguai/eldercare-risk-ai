from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from elderly_monitoring.modules.mental_health.wandering.preprocessing import (
    PREPROCESSED_SAMPLE_SCHEMA_VERSION,
    InputHashMismatchError,
    PreprocessingInput,
    _commit_new_output_directory,
    apply_feature_stats,
    arc_length_resample,
    build_raw_features,
    build_wandering_preprocessing_from_files,
    compute_train_feature_stats,
    load_preprocessing_config,
    preprocess_input,
    preprocess_public_trajectory,
    repair_public_index_gaps,
)
from elderly_monitoring.modules.mental_health.wandering.schemas import (
    TRAJECTORY_SAMPLE_SCHEMA_VERSION,
    CoordinateSystem,
    PatternLabel,
    TrajectorySample,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/data/wandering_preprocessing_v1.yaml"
MACHINE_FILES = (
    "samples.jsonl",
    "feature_stats.json",
    "preprocessing_report.json",
    "manifest.json",
)
EXPECTED_FROZEN_INPUTS = {
    "wandering_patterns_samples": (
        "data/processed/wandering/wandering_patterns/samples.jsonl",
        "498a0c39af27240bd38ab82367fdc1a13373ba3a22e8fede915419fea54e6ee7",
    ),
    "smartcare_train_pool": (
        "data/processed/wandering/smartcare/train_pool.jsonl",
        "01c5e3b8ecabd594837e5154ab64e52a1d364bcc15096a8193e8b627aa6fb780",
    ),
    "split_json": (
        "data/splits/mental_health/wandering/v1/split.json",
        "7fa934c8041f538ff033260d722c21f5a4dc0e6cd8d23fa42836aa2e24ed808f",
    ),
    "split_sha256_file": (
        "data/splits/mental_health/wandering/v1/split.sha256",
        "426175e2c5a7f822706f91445b2bd80c37704fb4b9943891a5fcfcd1b1efb96f",
    ),
    "assignments": (
        "data/splits/mental_health/wandering/v1/assignments.jsonl",
        "993307c13cb30484dba9fa1359c9c4fce714bb45b7d776e9a36ac6dcbee6eac6",
    ),
    "split_config": (
        "configs/data/wandering_split_v1.yaml",
        "579ad16e13b72f2914c5a2d14dfa73ef4a10a9a50f7ecc13bc20a67380a2e8ca",
    ),
}
EXPECTED_SPLIT_SHA256 = "4ac4a3877a056809066562cb09e4d30aa1d38baafcb4600f1f8a8a672776adbf"
FEATURE_CHANNELS = [
    "x",
    "y",
    "dx",
    "dy",
    "step_length",
    "sin_heading",
    "cos_heading",
    "sin_turn",
    "cos_turn",
    "abs_curvature",
    "normalized_dt",
    "time_available",
    "point_mask",
    "quality_score",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _input(
    points: np.ndarray,
    mask: np.ndarray,
    *,
    sample_id: str = "fixture-001",
    source_dataset: str = "wandering_patterns",
    split: str = "train",
) -> PreprocessingInput:
    return PreprocessingInput(
        sample_id=sample_id,
        source_dataset=source_dataset,
        input_role="samples" if source_dataset == "wandering_patterns" else "train_pool",
        split=split,
        binary_label=0 if source_dataset == "wandering_patterns" else 1,
        pattern_label="direct" if source_dataset == "wandering_patterns" else "unknown",
        binary_supervision_eligible=True,
        pattern_supervision_eligible=source_dataset == "wandering_patterns",
        coordinate_system=(
            "source_native" if source_dataset == "wandering_patterns" else "image_normalized"
        ),
        points=np.asarray(points, dtype=np.float64),
        point_mask=np.asarray(mask, dtype=np.int8),
        point_times_sec=None,
        point_quality=np.asarray(mask, dtype=np.float64),
    )


def _public_sample(
    points: np.ndarray,
    mask: np.ndarray,
    *,
    sample_id: str = "fixture-001",
    source_dataset: str = "wandering_patterns",
) -> tuple[TrajectorySample, dict[str, object]]:
    is_wp = source_dataset == "wandering_patterns"
    sample = TrajectorySample(
        schema_version=TRAJECTORY_SAMPLE_SCHEMA_VERSION,
        sample_id=sample_id,
        source_dataset=source_dataset,
        source_group_id=None if is_wp else "01/01/2020 00:00",
        coordinate_system=(
            CoordinateSystem.SOURCE_NATIVE if is_wp else CoordinateSystem.IMAGE_NORMALIZED
        ),
        points=tuple(tuple(float(value) for value in point) for point in points),
        point_times_sec=None,
        point_mask=tuple(int(value) for value in mask),
        binary_label=0 if is_wp else 1,
        pattern_label=PatternLabel.DIRECT if is_wp else PatternLabel.UNKNOWN,
        quality_flags=("time_unavailable",),
        source_path="raw/source.json",
        source_sha256="a" * 64,
    )
    assignment: dict[str, object] = {
        "input_role": "samples" if is_wp else "train_pool",
        "split": "train",
        "binary_supervision_eligible": True,
        "pattern_supervision_eligible": is_wp,
    }
    return sample, assignment


class WanderingPreprocessingConfigTest(unittest.TestCase):
    def test_config_freezes_every_step_two_and_step_three_input_and_all_parameters(self) -> None:
        config = load_preprocessing_config(CONFIG_PATH)

        self.assertEqual(config["schema_version"], "wandering-preprocessing-config-v1")
        self.assertEqual(config["split_sha256"], EXPECTED_SPLIT_SHA256)
        self.assertEqual(config["target_points"], 80)
        self.assertEqual(config["min_valid_points"], 8)
        self.assertEqual(config["max_index_gap"], 2)
        self.assertEqual(config["interpolated_quality"], 0.5)
        self.assertEqual(config["quantile_method"], "linear")
        self.assertEqual(config["scale_quantiles"], [0.05, 0.95])
        self.assertEqual(config["scale_epsilon"], 1e-8)
        self.assertEqual(config["step_epsilon"], 1e-8)
        self.assertEqual(config["iqr_epsilon"], 1e-8)
        self.assertEqual(config["numeric_tolerance"], 1e-6)
        self.assertEqual(config["diagnostics"], {"seed": 20260803, "samples_per_class": 8})
        self.assertEqual(
            config["feature_channels"],
            [
                "x",
                "y",
                "dx",
                "dy",
                "step_length",
                "sin_heading",
                "cos_heading",
                "sin_turn",
                "cos_turn",
                "abs_curvature",
                "normalized_dt",
                "time_available",
                "point_mask",
                "quality_score",
            ],
        )
        self.assertEqual(set(config["inputs"]), set(EXPECTED_FROZEN_INPUTS))
        for role, (relative_path, expected_hash) in EXPECTED_FROZEN_INPUTS.items():
            with self.subTest(role=role):
                self.assertEqual(config["inputs"][role]["path"], relative_path)
                self.assertEqual(config["inputs"][role]["sha256"], expected_hash)
                self.assertEqual(_sha256(PROJECT_ROOT / relative_path), expected_hash)

    def test_each_file_hash_and_canonical_split_hash_drift_fails_before_output_creation(self) -> None:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        mutations: list[tuple[str, callable]] = [
            (
                role,
                lambda value, role=role: value["inputs"][role].update(sha256="0" * 64),
            )
            for role in EXPECTED_FROZEN_INPUTS
        ]
        mutations.append(
            ("canonical split", lambda value: value.update(split_sha256="0" * 64))
        )
        for name, mutate in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmpdir:
                changed = copy.deepcopy(raw)
                mutate(changed)
                config_path = Path(tmpdir) / "changed.yaml"
                config_path.write_text(
                    yaml.safe_dump(changed, sort_keys=False),
                    encoding="utf-8",
                )
                output = Path(tmpdir) / "output"
                with self.assertRaises(InputHashMismatchError):
                    build_wandering_preprocessing_from_files(
                        config_path=config_path,
                        project_root=PROJECT_ROOT,
                        output_dir=output,
                    )
                self.assertFalse(output.exists())


class WanderingPublicQualityControlTest(unittest.TestCase):
    def test_eight_original_observations_are_required_before_interpolation(self) -> None:
        points = np.column_stack((np.arange(8, dtype=np.float64), np.zeros(8)))
        sample, assignment = _public_sample(
            points,
            np.asarray([1, 1, 1, 0, 1, 1, 1, 1]),
        )

        record = preprocess_public_trajectory(sample, assignment)

        self.assertEqual(record["preprocess_status"], "unavailable")
        self.assertEqual(record["reason_codes"], ["too_few_valid_points"])
        self.assertEqual(record["valid_input_point_count"], 7)
        self.assertIsNone(record["raw_features"])

    def test_edge_missing_values_are_trimmed_and_short_internal_gap_quality_propagates(self) -> None:
        points = np.column_stack((np.arange(13, dtype=np.float64), np.zeros(13)))
        mask = np.asarray([0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0])

        repaired = repair_public_index_gaps(
            points,
            mask,
            min_valid_points=8,
            max_index_gap=2,
            interpolated_quality=0.5,
        )

        self.assertEqual(repaired.valid_input_point_count, 8)
        self.assertEqual(repaired.trimmed_edge_point_count, 3)
        self.assertEqual(repaired.interpolated_input_point_count, 2)
        self.assertEqual(repaired.quality.tolist(), [1, 1, 1, 0.5, 0.5, 1, 1, 1, 1, 1])
        resampled = arc_length_resample(
            repaired.points,
            repaired.quality,
            target_points=80,
        )
        self.assertEqual(resampled.points.shape, (80, 2))
        self.assertTrue(np.any(resampled.quality == 0.5))
        self.assertTrue(np.all(np.isin(resampled.quality, [0.5, 1.0])))

    def test_three_point_internal_gap_is_unavailable_and_not_silently_deleted(self) -> None:
        points = np.column_stack((np.arange(14, dtype=np.float64), np.zeros(14)))
        mask = np.asarray([1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        sample, assignment = _public_sample(points, mask)
        record = preprocess_public_trajectory(sample, assignment)

        self.assertEqual(record["sample_id"], "fixture-001")
        self.assertEqual(record["parent_sample_id"], "fixture-001")
        self.assertEqual(record["preprocess_status"], "unavailable")
        self.assertEqual(record["reason_codes"], ["unresolved_internal_gap"])
        self.assertIsNone(record["resampled_source_points"])
        self.assertIsNone(record["topology"])

    def test_arc_length_resampling_removes_zero_length_segments_and_propagates_min_quality(self) -> None:
        points = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        quality = np.asarray([1.0, 0.5, 0.5, 1.0])

        result = arc_length_resample(points, quality, target_points=5)

        np.testing.assert_allclose(result.points[:, 0], [0.0, 0.5, 1.0, 1.5, 2.0])
        np.testing.assert_allclose(result.points[:, 1], 0.0)
        np.testing.assert_array_equal(result.quality, [1.0, 0.5, 0.5, 0.5, 1.0])

    def test_degenerate_motion_is_unavailable(self) -> None:
        points = np.full((8, 2), [0.25, 0.75], dtype=np.float64)
        record = preprocess_input(
            _input(points, np.ones(8, dtype=np.int8), source_dataset="smartcare")
        )

        self.assertEqual(record["preprocess_status"], "unavailable")
        self.assertEqual(record["reason_codes"], ["insufficient_motion"])


class WanderingFeatureChannelTest(unittest.TestCase):
    def test_fourteen_channels_use_the_exact_frozen_formula(self) -> None:
        points = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        features = build_raw_features(
            points,
            np.ones(3, dtype=np.int8),
            np.asarray([1.0, 0.5, 1.0]),
            point_times_sec=None,
            step_epsilon=1e-8,
            temporal_features_enabled=False,
        )

        self.assertEqual(features.shape, (3, 14))
        np.testing.assert_allclose(
            features[0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        )
        np.testing.assert_allclose(
            features[1],
            [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0.5],
        )
        np.testing.assert_allclose(
            features[2],
            [1, 1, 0, 1, 1, 1, 0, 1, 0, math.pi / 2, 0, 0, 1, 1],
            atol=1e-12,
        )

    def test_masked_positions_and_invalid_neighbor_derivatives_are_zero(self) -> None:
        points = np.asarray([[2.0, 3.0], [3.0, 3.0], [99.0, 99.0], [4.0, 4.0]])
        mask = np.asarray([1, 1, 0, 1], dtype=np.int8)
        features = build_raw_features(
            points,
            mask,
            np.asarray([1.0, 1.0, 0.0, 1.0]),
            point_times_sec=None,
            step_epsilon=1e-8,
            temporal_features_enabled=False,
        )

        np.testing.assert_array_equal(features[2], np.zeros(14))
        np.testing.assert_array_equal(features[3, 2:12], np.zeros(10))
        self.assertEqual(features[3, 12], 1.0)
        self.assertEqual(features[3, 13], 1.0)

    def test_train_statistics_use_only_ready_train_valid_positions(self) -> None:
        train = np.zeros((3, 14), dtype=np.float64)
        train[:, 0] = [1.0, 3.0, 9999.0]
        train[:, 12] = [1.0, 1.0, 0.0]
        validation = np.zeros((2, 14), dtype=np.float64)
        validation[:, 0] = [-9999.0, -9999.0]
        validation[:, 12] = 1.0
        records = [
            {"preprocess_status": "ready", "split": "train", "raw_features": train.tolist()},
            {"preprocess_status": "ready", "split": "validation", "raw_features": validation.tolist()},
            {"preprocess_status": "unavailable", "split": "train", "raw_features": None},
        ]

        stats = compute_train_feature_stats(
            records,
            feature_channels=FEATURE_CHANNELS,
            quantile_method="linear",
            iqr_epsilon=1e-8,
            binding_hashes={"split_sha256": "a" * 64, "preprocessing_config": "b" * 64},
        )

        self.assertEqual(stats["valid_position_count"], 2)
        self.assertEqual(stats["channels"][0]["q50"], 2.0)
        self.assertEqual(stats["channels"][0]["iqr"], 1.0)
        standardized = apply_feature_stats(train, stats)
        self.assertEqual(standardized[2].tolist(), [0.0] * 14)
        np.testing.assert_array_equal(standardized[:, 10:14], train[:, 10:14])


class WanderingPreprocessingBundleIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.first = cls.root / "first"
        cls.second = cls.root / "second"
        cls.step_input_hashes_before = {
            role: _sha256(PROJECT_ROOT / path)
            for role, (path, _expected) in EXPECTED_FROZEN_INPUTS.items()
        }
        build_wandering_preprocessing_from_files(
            config_path=CONFIG_PATH,
            project_root=PROJECT_ROOT,
            output_dir=cls.first,
        )
        build_wandering_preprocessing_from_files(
            config_path=CONFIG_PATH,
            project_root=PROJECT_ROOT,
            output_dir=cls.second,
        )
        cls.rows = [
            json.loads(line)
            for line in (cls.first / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        cls.report = json.loads((cls.first / "preprocessing_report.json").read_text(encoding="utf-8"))
        cls.stats = json.loads((cls.first / "feature_stats.json").read_text(encoding="utf-8"))
        cls.manifest = json.loads((cls.first / "manifest.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_every_nonsealed_id_appears_once_and_sealed_content_is_absent(self) -> None:
        ids = [row["sample_id"] for row in self.rows]
        self.assertEqual(len(ids), 1790)
        self.assertEqual(len(set(ids)), 1790)
        self.assertEqual(ids, sorted(ids))
        self.assertTrue(all(row["sample_id"] == row["parent_sample_id"] for row in self.rows))

        split = json.loads(
            (PROJECT_ROOT / EXPECTED_FROZEN_INPUTS["split_json"][0]).read_text(encoding="utf-8")
        )
        sealed_ids = split["sealed_external_test"]
        bundle_bytes = b"".join((self.first / name).read_bytes() for name in MACHINE_FILES)
        for sample_id in sealed_ids:
            self.assertNotIn(sample_id.encode("ascii"), bundle_bytes)
        self.assertEqual(self.report["counts"]["sealed_excluded"], 20)

    def test_fixed_ready_unavailable_counts_shapes_labels_and_time_channels(self) -> None:
        status_counts = Counter(row["preprocess_status"] for row in self.rows)
        self.assertEqual(status_counts, {"ready": 1775, "unavailable": 15})
        ready_split_counts = Counter(
            row["split"] for row in self.rows if row["preprocess_status"] == "ready"
        )
        self.assertEqual(ready_split_counts, {"train": 1257, "validation": 278, "test": 240})
        unavailable = [row for row in self.rows if row["preprocess_status"] == "unavailable"]
        self.assertTrue(all(row["source_dataset"] == "smartcare" for row in unavailable))
        self.assertTrue(all(row["split"] == "train" for row in unavailable))
        self.assertEqual(Counter(row["binary_label"] for row in unavailable), {0: 10, 1: 5})
        self.assertTrue(all(row["reason_codes"] == ["too_few_valid_points"] for row in unavailable))

        for row in self.rows:
            self.assertEqual(row["schema_version"], PREPROCESSED_SAMPLE_SCHEMA_VERSION)
            self.assertEqual(row["split_sha256"], EXPECTED_SPLIT_SHA256)
            if row["preprocess_status"] == "unavailable":
                for field in (
                    "resampled_source_points",
                    "image_normalized_points",
                    "shape_normalized_points",
                    "point_mask",
                    "raw_features",
                    "model_features",
                    "topology",
                ):
                    self.assertIsNone(row[field])
                continue
            self.assertEqual(np.asarray(row["resampled_source_points"]).shape, (80, 2))
            self.assertEqual(np.asarray(row["shape_normalized_points"]).shape, (80, 2))
            self.assertEqual(np.asarray(row["point_mask"]).shape, (80,))
            self.assertEqual(np.asarray(row["raw_features"]).shape, (80, 14))
            self.assertEqual(np.asarray(row["model_features"]).shape, (80, 14))
            self.assertTrue(np.isfinite(np.asarray(row["raw_features"])).all())
            self.assertTrue(np.isfinite(np.asarray(row["model_features"])).all())
            self.assertEqual(set(row["point_mask"]), {1})
            raw = np.asarray(row["raw_features"])
            np.testing.assert_array_equal(raw[:, 10:12], np.zeros((80, 2)))
            if row["source_dataset"] == "wandering_patterns":
                self.assertIsNone(row["image_normalized_points"])
            else:
                np.testing.assert_array_equal(
                    row["image_normalized_points"],
                    row["resampled_source_points"],
                )

    def test_feature_stats_are_train_only_and_bound_to_all_frozen_inputs(self) -> None:
        self.assertEqual(self.stats["schema_version"], "wandering-feature-stats-v1")
        self.assertEqual(self.stats["ready_train_sample_count"], 1257)
        self.assertEqual(self.stats["valid_position_count"], 1257 * 80)
        self.assertFalse(self.stats["temporal_features_enabled"])
        self.assertEqual(self.stats["binding_hashes"]["split_sha256"], EXPECTED_SPLIT_SHA256)
        for role, (_path, digest) in EXPECTED_FROZEN_INPUTS.items():
            self.assertEqual(self.stats["binding_hashes"][role], digest)

    def test_manifest_is_one_way_canonical_and_two_builds_are_byte_identical(self) -> None:
        for name in MACHINE_FILES:
            with self.subTest(name=name):
                self.assertEqual((self.first / name).read_bytes(), (self.second / name).read_bytes())
                payload = (self.first / name).read_bytes()
                self.assertTrue(payload.endswith(b"\n"))
                self.assertFalse(payload.endswith(b"\n\n"))
                if name == "samples.jsonl":
                    for line in payload.splitlines(keepends=True):
                        self.assertEqual(line, _canonical_bytes(json.loads(line)))
                else:
                    self.assertEqual(payload, _canonical_bytes(json.loads(payload)))

        self.assertEqual(self.manifest["schema_version"], "wandering-preprocessing-manifest-v1")
        self.assertEqual(set(self.manifest["artifacts"]), set(MACHINE_FILES[:3]))
        for name in MACHINE_FILES[:3]:
            self.assertEqual(self.manifest["artifacts"][name]["sha256"], _sha256(self.first / name))
        for name in MACHINE_FILES[:3]:
            payload = (self.first / name).read_bytes()
            self.assertNotIn(b'"manifest.json"', payload)
            self.assertNotIn(b'"manifest_sha256"', payload)

    def test_builder_rejects_existing_output_preserves_inputs_and_commits_atomically(self) -> None:
        with self.assertRaises(FileExistsError):
            build_wandering_preprocessing_from_files(
                config_path=CONFIG_PATH,
                project_root=PROJECT_ROOT,
                output_dir=self.first,
            )
        after = {
            role: _sha256(PROJECT_ROOT / path)
            for role, (path, _expected) in EXPECTED_FROZEN_INPUTS.items()
        }
        self.assertEqual(after, self.step_input_hashes_before)

        atomic_output = self.root / "atomic-failure"
        with mock.patch(
            "elderly_monitoring.modules.mental_health.wandering.preprocessing.os.replace",
            side_effect=OSError("synthetic commit failure"),
        ):
            with self.assertRaises(OSError):
                _commit_new_output_directory(
                    atomic_output,
                    {"samples.jsonl": b"{}\n", "manifest.json": b"{}\n"},
                )
        self.assertFalse(atomic_output.exists())
        self.assertEqual(list(self.root.glob(".atomic-failure.*.tmp")), [])

    def test_builder_records_fixed_train_diagnostics_and_all_short_ids(self) -> None:
        diagnostics = self.report["diagnostics"]
        self.assertEqual(diagnostics["seed"], 20260803)
        self.assertEqual(diagnostics["samples_per_class"], 8)
        ready_groups = diagnostics["ready_train_sample_ids"]
        self.assertEqual(set(ready_groups["wandering_patterns"]), {"direct", "pacing", "lapping", "random"})
        self.assertEqual(set(ready_groups["smartcare"]), {"normal", "wandering_like"})
        self.assertTrue(
            all(len(ids) == 8 for groups in ready_groups.values() for ids in groups.values())
        )
        unavailable_ids = diagnostics["unavailable_sample_ids"]
        self.assertEqual(len(unavailable_ids), 15)
        self.assertEqual(
            set(unavailable_ids),
            {row["sample_id"] for row in self.rows if row["preprocess_status"] == "unavailable"},
        )


if __name__ == "__main__":
    unittest.main()
