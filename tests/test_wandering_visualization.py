from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.schemas import (
    TRAJECTORY_SAMPLE_SCHEMA_VERSION,
    CoordinateSystem,
    PatternLabel,
    TrajectorySample,
)
from elderly_monitoring.modules.mental_health.wandering.visualization import (
    VISUAL_REVIEW_SCHEMA_VERSION,
    WanderingVisualizationError,
    write_visual_review,
)


def _sample(index: int, label: PatternLabel) -> TrajectorySample:
    binary = 0 if label is PatternLabel.DIRECT else 1
    return TrajectorySample(
        schema_version=TRAJECTORY_SAMPLE_SCHEMA_VERSION,
        sample_id=f"wandering_patterns_{label.value}_{index:03d}",
        source_dataset="wandering_patterns",
        source_group_id=None,
        coordinate_system=CoordinateSystem.SOURCE_NATIVE,
        points=((0.0, 0.0), (1.0, float(index % 3)), (2.0, 0.0), (3.0, 1.0)),
        point_times_sec=None,
        point_mask=(1, 1, 1, 1),
        binary_label=binary,
        pattern_label=label,
        quality_flags=("time_unavailable",),
        source_path="raw/patterns_dataset.pkl",
        source_sha256="a" * 64,
    )


class WanderingVisualizationTest(unittest.TestCase):
    def test_seeded_review_selects_exactly_twenty_per_class_deterministically(self) -> None:
        samples = [
            _sample(index, label)
            for label in (PatternLabel.DIRECT, PatternLabel.PACING)
            for index in range(25)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = write_visual_review(
                samples,
                source_name="wandering_patterns",
                input_role="samples",
                input_sha256="b" * 64,
                output_dir=root / "first",
                samples_per_class=20,
                seed=20260801,
            )
            second = write_visual_review(
                samples,
                source_name="wandering_patterns",
                input_role="samples",
                input_sha256="b" * 64,
                output_dir=root / "second",
                samples_per_class=20,
                seed=20260801,
            )

            first_record = json.loads(first.selection_path.read_text(encoding="utf-8"))
            second_record = json.loads(second.selection_path.read_text(encoding="utf-8"))
            self.assertEqual(
                first_record["schema_version"], VISUAL_REVIEW_SCHEMA_VERSION
            )
            self.assertEqual(first_record, second_record)
            self.assertEqual(set(first_record["classes"]), {"direct", "pacing"})
            self.assertTrue(
                all(len(sample_ids) == 20 for sample_ids in first_record["classes"].values())
            )
            self.assertEqual(first_record["review_status"], "pending_human_review")
            self.assertEqual(len(first.plot_paths), 2)
            self.assertTrue(all(path.is_file() for path in first.plot_paths))

    def test_insufficient_class_count_fails_without_partial_review(self) -> None:
        samples = [_sample(index, PatternLabel.DIRECT) for index in range(19)]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "review"
            with self.assertRaisesRegex(
                WanderingVisualizationError, "at least 20"
            ):
                write_visual_review(
                    samples,
                    source_name="wandering_patterns",
                    input_role="samples",
                    input_sha256="b" * 64,
                    output_dir=output,
                    samples_per_class=20,
                    seed=20260801,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
