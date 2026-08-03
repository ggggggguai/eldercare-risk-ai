from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.converters import (
    CONVERSION_REJECTIONS_SCHEMA_VERSION,
    SMARTCARE_CONVERTER_VERSION,
    WANDERING_PATTERNS_EXTRACT_SCHEMA_VERSION,
    WanderingConversionError,
    convert_smartcare,
    convert_wandering_patterns_safe_extract,
)
from elderly_monitoring.modules.mental_health.wandering.datasets import (
    load_trajectory_jsonl,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _write_double_encoded_json(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    inner = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    path.write_text(json.dumps(inner, ensure_ascii=False), encoding="utf-8")


def _smartcare_points(
    trajectory_id: str,
    *,
    stress: bool,
    points: list[tuple[float, float]],
) -> list[dict[str, object]]:
    return [
        {"date": trajectory_id, "x": x, "y": y, "stress": stress}
        for x, y in points
    ]


class SmartCareConverterTest(unittest.TestCase):
    def _build_source(self, root: Path) -> tuple[Path, Path]:
        main_path = root / "raw" / "dataset.json"
        validation_path = root / "raw" / "dataset-validacao.json"
        main_rows = [
            *_smartcare_points(
                "normal-valid",
                stress=False,
                points=[(0, 0), (100, 20), (200, 40), (601, 368)],
            ),
            *_smartcare_points(
                "wandering-valid",
                stress=True,
                points=[(20, 30), (80, 80), (20, 100), (80, 130), (20, 160)],
            ),
            *_smartcare_points(
                "too-short",
                stress=True,
                points=[(10, 10)],
            ),
            *_smartcare_points(
                "out-of-bounds",
                stress=False,
                points=[(0, 0), (602, 20), (100, -1), (120, 40)],
            ),
        ]
        validation_rows = [
            *_smartcare_points(
                "sealed-normal",
                stress=False,
                points=[(5, 5), (15, 15), (25, 25), (35, 35)],
            ),
            *_smartcare_points(
                "sealed-wandering",
                stress=True,
                points=[(50, 50), (80, 50), (50, 50), (80, 50)],
            ),
        ]
        _write_double_encoded_json(main_path, main_rows)
        _write_double_encoded_json(validation_path, validation_rows)
        return main_path, validation_path

    def test_converts_safe_jsonl_and_records_short_and_out_of_bounds_trajectories(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "source"
            main_path, validation_path = self._build_source(source_root)
            original = {
                path: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in (main_path, validation_path)
            }

            bundle = convert_smartcare(source_root, root / "converted")

            train_samples = load_trajectory_jsonl(bundle.output_dir / "train_pool.jsonl")
            sealed_samples = load_trajectory_jsonl(
                bundle.output_dir / "official_validation.jsonl"
            )
            self.assertEqual(len(train_samples), 2)
            self.assertEqual(len(sealed_samples), 2)
            self.assertEqual(
                {sample.binary_label for sample in train_samples},
                {0, 1},
            )
            boundary_sample = next(
                sample
                for sample in train_samples
                if sample.source_group_id == "normal-valid"
            )
            self.assertEqual(boundary_sample.points[-1], (1.0, 1.0))
            self.assertIsNone(boundary_sample.point_times_sec)
            self.assertIn("time_unavailable", boundary_sample.quality_flags)

            self.assertEqual(bundle.report.converter_version, SMARTCARE_CONVERTER_VERSION)
            self.assertEqual(bundle.report.sample_count_read, 6)
            self.assertEqual(bundle.report.sample_count_written, 4)
            self.assertEqual(bundle.report.sample_count_rejected, 2)
            self.assertEqual(
                dict(bundle.report.class_counts),
                {"normal": 2, "wandering_like": 2},
            )
            self.assertEqual(bundle.report.group_fields_found, ("date",))
            warnings = {warning.code: warning for warning in bundle.report.warnings}
            self.assertEqual(warnings["trajectory_too_short"].count, 1)
            self.assertEqual(warnings["coordinate_out_of_bounds"].count, 1)
            self.assertEqual(warnings["point_time_unavailable"].count, 4)

            rejection_report = json.loads(
                (bundle.output_dir / "rejections.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                rejection_report["schema_version"],
                CONVERSION_REJECTIONS_SCHEMA_VERSION,
            )
            rejections = {
                item["source_record_id"]: item
                for item in rejection_report["records"]
            }
            self.assertEqual(
                rejections["too-short"]["codes"],
                ["trajectory_too_short"],
            )
            self.assertEqual(
                rejections["too-short"]["details"]["point_count"],
                1,
            )
            self.assertEqual(
                rejections["out-of-bounds"]["codes"],
                ["coordinate_out_of_bounds"],
            )
            self.assertEqual(
                len(
                    rejections["out-of-bounds"]["details"][
                        "out_of_bounds_points"
                    ]
                ),
                2,
            )

            for path, (raw_bytes, mtime_ns) in original.items():
                self.assertEqual(path.read_bytes(), raw_bytes)
                self.assertEqual(path.stat().st_mtime_ns, mtime_ns)

    def test_two_runs_are_byte_deterministic_and_output_cannot_touch_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "source"
            self._build_source(source_root)

            first = convert_smartcare(source_root, root / "first")
            second = convert_smartcare(source_root, root / "second")

            self.assertEqual(_tree_hashes(first.output_dir), _tree_hashes(second.output_dir))
            with self.assertRaisesRegex(WanderingConversionError, "outside source_root"):
                convert_smartcare(source_root, source_root / "processed")

    def test_malformed_or_mixed_label_input_fails_or_rejects_without_partial_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "source"
            main_path, _ = self._build_source(source_root)
            main_path.write_text("not-json", encoding="utf-8")
            output_dir = root / "converted"

            with self.assertRaisesRegex(WanderingConversionError, "outer JSON"):
                convert_smartcare(source_root, output_dir)

            self.assertFalse(output_dir.exists())


class WanderingPatternsSafeExtractConverterTest(unittest.TestCase):
    def _build_safe_extract(
        self,
        root: Path,
        *,
        group_field: str | None = None,
    ) -> tuple[Path, Path, Path, Path]:
        source_root = root / "source"
        primary = source_root / "raw" / "patterns_dataset.pkl"
        duplicate = source_root / "source_repository" / "model_data" / "patterns_dataset.pkl"
        primary.parent.mkdir(parents=True)
        duplicate.parent.mkdir(parents=True)
        primary.write_bytes(b"trusted-test-pickle-bytes")
        duplicate.write_bytes(primary.read_bytes())
        source_sha256 = _sha256(primary)

        extract_dir = root / "extract"
        extract_dir.mkdir()
        metadata_path = extract_dir / "metadata.json"
        rows_path = extract_dir / "rows.jsonl"
        columns = [
            "CartesianX",
            "CartesianY",
            "Coords",
            "Slope",
            "Path_Efficiency",
            "Coords_Slope",
            "pattern",
        ]
        if group_field is not None:
            columns.append(group_field)
        metadata = {
            "schema_version": WANDERING_PATTERNS_EXTRACT_SCHEMA_VERSION,
            "source_sha256": source_sha256,
            "record_count": 4,
            "columns": columns,
            "group_fields_found": [] if group_field is None else [group_field],
            "pickle_globals_loaded": ["pandas.core.frame.DataFrame"],
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        labels = ("direct", "pacing", "lapping", "random")
        with rows_path.open("w", encoding="utf-8", newline="\n") as handle:
            for index, label in enumerate(labels):
                group_values = (
                    {} if group_field is None else {group_field: f"participant-{index % 2}"}
                )
                row = {
                    "record_index": index,
                    "pattern": label,
                    "points": [
                        [0.0, 0.0],
                        [float(index + 1), 1.0],
                        [0.5, 1.5],
                        [0.0, 2.0],
                    ],
                    "group_values": group_values,
                }
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        return source_root, primary, metadata_path, rows_path

    def test_safe_extract_converts_all_four_classes_and_proves_group_unavailable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root, primary, metadata_path, rows_path = self._build_safe_extract(
                root
            )
            primary_bytes = primary.read_bytes()
            primary_mtime_ns = primary.stat().st_mtime_ns

            bundle = convert_wandering_patterns_safe_extract(
                source_root,
                root / "converted",
                extracted_jsonl=rows_path,
                extraction_metadata=metadata_path,
                duplicate_relative_path="source_repository/model_data/patterns_dataset.pkl",
            )

            samples = load_trajectory_jsonl(bundle.output_dir / "samples.jsonl")
            self.assertEqual(len(samples), 4)
            self.assertEqual(
                {sample.pattern_label.value for sample in samples},
                {"direct", "pacing", "lapping", "random"},
            )
            self.assertTrue(all(sample.source_group_id is None for sample in samples))
            self.assertEqual(bundle.report.group_fields_found, ())
            warnings = {warning.code: warning for warning in bundle.report.warnings}
            self.assertEqual(warnings["group_field_unavailable"].count, 4)
            self.assertEqual(dict(bundle.report.class_counts), {label: 1 for label in (
                "direct",
                "lapping",
                "pacing",
                "random",
            )})
            self.assertEqual(primary.read_bytes(), primary_bytes)
            self.assertEqual(primary.stat().st_mtime_ns, primary_mtime_ns)

    def test_group_field_is_preserved_only_when_extract_proves_it_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root, _, metadata_path, rows_path = self._build_safe_extract(
                root,
                group_field="participant_id",
            )

            bundle = convert_wandering_patterns_safe_extract(
                source_root,
                root / "converted",
                extracted_jsonl=rows_path,
                extraction_metadata=metadata_path,
                duplicate_relative_path="source_repository/model_data/patterns_dataset.pkl",
            )

            samples = load_trajectory_jsonl(bundle.output_dir / "samples.jsonl")
            self.assertEqual(bundle.report.group_fields_found, ("participant_id",))
            self.assertEqual(
                {sample.source_group_id for sample in samples},
                {"participant-0", "participant-1"},
            )
            self.assertNotIn(
                "group_field_unavailable",
                {warning.code for warning in bundle.report.warnings},
            )

    def test_two_runs_match_and_pickle_is_never_accepted_as_safe_extract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root, primary, metadata_path, rows_path = self._build_safe_extract(
                root
            )

            first = convert_wandering_patterns_safe_extract(
                source_root,
                root / "first",
                extracted_jsonl=rows_path,
                extraction_metadata=metadata_path,
                duplicate_relative_path="source_repository/model_data/patterns_dataset.pkl",
            )
            second = convert_wandering_patterns_safe_extract(
                source_root,
                root / "second",
                extracted_jsonl=rows_path,
                extraction_metadata=metadata_path,
                duplicate_relative_path="source_repository/model_data/patterns_dataset.pkl",
            )
            self.assertEqual(_tree_hashes(first.output_dir), _tree_hashes(second.output_dir))

            with self.assertRaisesRegex(WanderingConversionError, "safe .jsonl extract"):
                convert_wandering_patterns_safe_extract(
                    source_root,
                    root / "unsafe",
                    extracted_jsonl=primary,
                    extraction_metadata=metadata_path,
                )


if __name__ == "__main__":
    unittest.main()
