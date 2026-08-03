from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering import (
    TRAJECTORY_SAMPLE_SCHEMA_VERSION,
    CoordinateSystem,
    PatternLabel,
    TrajectoryDatasetError,
    TrajectorySample,
    TrajectorySchemaError,
    load_trajectory_jsonl,
    write_trajectory_jsonl,
)


def trajectory_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": TRAJECTORY_SAMPLE_SCHEMA_VERSION,
        "sample_id": "wandering_patterns_000001",
        "source_dataset": "wandering_patterns",
        "source_group_id": None,
        "coordinate_system": "shape_normalized",
        "points": [[0.0, 0.0], [0.02, -0.01], [0.04, -0.02]],
        "point_times_sec": [0.0, 0.5, 1.0],
        "point_mask": [1, 1, 1],
        "binary_label": 1,
        "pattern_label": "pacing",
        "quality_flags": [],
        "source_path": "raw/patterns_dataset.pkl",
        "source_sha256": "a" * 64,
    }
    record.update(overrides)
    return record


class TrajectorySampleSchemaTest(unittest.TestCase):
    def test_valid_mapping_round_trips_without_losing_contract_fields(self) -> None:
        sample = TrajectorySample.from_dict(trajectory_record())

        self.assertEqual(sample.schema_version, TRAJECTORY_SAMPLE_SCHEMA_VERSION)
        self.assertEqual(sample.coordinate_system, CoordinateSystem.SHAPE_NORMALIZED)
        self.assertEqual(sample.pattern_label, PatternLabel.PACING)
        self.assertEqual(sample.valid_point_count, 3)
        self.assertTrue(sample.is_wandering)
        self.assertEqual(sample.to_dict(), trajectory_record())

    def test_pattern_and_binary_labels_must_be_consistent(self) -> None:
        invalid_cases = (
            ("direct", 1),
            ("pacing", 0),
            ("lapping", None),
            ("random", 0),
        )

        for pattern_label, binary_label in invalid_cases:
            with self.subTest(pattern_label=pattern_label, binary_label=binary_label):
                with self.assertRaisesRegex(TrajectorySchemaError, "binary_label"):
                    TrajectorySample.from_dict(
                        trajectory_record(
                            pattern_label=pattern_label,
                            binary_label=binary_label,
                        )
                    )

        unknown = TrajectorySample.from_dict(
            trajectory_record(pattern_label="unknown", binary_label=0)
        )
        self.assertFalse(unknown.is_wandering)

    def test_points_masks_and_times_are_validated_together(self) -> None:
        invalid_cases = (
            (
                {"points": [[0.0, 0.0], [float("nan"), 0.1], [0.2, 0.2]]},
                "points",
            ),
            ({"point_mask": [1, 0]}, "point_mask"),
            ({"point_mask": [1, 0, 0]}, "at least two valid"),
            ({"point_mask": [1, 2, 1]}, "point_mask"),
            ({"point_times_sec": [0.0, 1.0]}, "point_times_sec"),
            ({"point_times_sec": [0.0, 1.0, 1.0]}, "strictly increasing"),
        )

        for overrides, message in invalid_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(TrajectorySchemaError, message):
                    TrajectorySample.from_dict(trajectory_record(**overrides))

    def test_image_normalized_points_must_stay_in_unit_square(self) -> None:
        with self.assertRaisesRegex(TrajectorySchemaError, "image_normalized"):
            TrajectorySample.from_dict(
                trajectory_record(
                    coordinate_system="image_normalized",
                    points=[[0.0, 0.0], [1.1, 0.5], [0.9, 1.0]],
                )
            )

    def test_schema_version_provenance_and_unknown_fields_are_strict(self) -> None:
        invalid_cases = (
            ({"schema_version": "wandering-trajectory-sample-v0"}, "schema_version"),
            ({"source_path": "D:/private/raw.pkl"}, "source_path"),
            ({"source_path": "../raw.pkl"}, "source_path"),
            ({"source_sha256": "not-a-hash"}, "source_sha256"),
            ({"quality_flags": ["occluded", "occluded"]}, "quality_flags"),
            ({"unexpected_field": "value"}, "unknown fields"),
        )

        for overrides, message in invalid_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(TrajectorySchemaError, message):
                    TrajectorySample.from_dict(trajectory_record(**overrides))


class TrajectoryJsonlTest(unittest.TestCase):
    def test_jsonl_round_trip_is_deterministic(self) -> None:
        first = TrajectorySample.from_dict(trajectory_record())
        second = TrajectorySample.from_dict(
            trajectory_record(
                sample_id="wandering_patterns_000002",
                pattern_label="direct",
                binary_label=0,
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "samples.jsonl"

            write_trajectory_jsonl(output_path, [first, second])
            first_bytes = output_path.read_bytes()
            loaded = load_trajectory_jsonl(output_path)
            write_trajectory_jsonl(output_path, loaded)

            self.assertEqual(loaded, (first, second))
            self.assertEqual(output_path.read_bytes(), first_bytes)

    def test_loader_rejects_unsafe_or_non_jsonl_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pickle_path = Path(temp_dir) / "samples.pkl"
            pickle_path.write_bytes(b"not executed")

            with self.assertRaisesRegex(TrajectoryDatasetError, "safe .jsonl"):
                load_trajectory_jsonl(pickle_path)

    def test_loader_reports_line_context_and_duplicate_sample_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_path = Path(temp_dir) / "invalid.jsonl"
            invalid_path.write_text(
                json.dumps(trajectory_record(), ensure_ascii=False)
                + "\n"
                + "{not-json}\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                TrajectoryDatasetError,
                r"invalid\.jsonl:2",
            ):
                load_trajectory_jsonl(invalid_path)

            duplicate_path = Path(temp_dir) / "duplicate.jsonl"
            row = json.dumps(trajectory_record(), ensure_ascii=False)
            duplicate_path.write_text(f"{row}\n{row}\n", encoding="utf-8")

            with self.assertRaisesRegex(
                TrajectoryDatasetError,
                r"duplicate\.jsonl:2.*duplicate sample_id",
            ):
                load_trajectory_jsonl(duplicate_path)

    def test_failed_atomic_write_does_not_replace_existing_dataset(self) -> None:
        sample = TrajectorySample.from_dict(trajectory_record())

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "samples.jsonl"
            write_trajectory_jsonl(output_path, [sample])
            original_bytes = output_path.read_bytes()

            with self.assertRaisesRegex(TrajectoryDatasetError, "duplicate sample_id"):
                write_trajectory_jsonl(output_path, [sample, sample])

            self.assertEqual(output_path.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
