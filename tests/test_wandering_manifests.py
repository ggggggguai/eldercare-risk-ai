from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.manifests import (
    CONVERSION_REPORT_SCHEMA_VERSION,
    SOURCE_MANIFEST_SCHEMA_VERSION,
    SPLIT_MANIFEST_SCHEMA_VERSION,
    ConversionManifest,
    ConversionReport,
    ConversionWarning,
    ManifestValidationError,
    OutputArtifact,
    SourceFile,
    SplitManifest,
    build_split_manifest,
    derive_sample_id,
    describe_source_file,
    validate_conversion_bundle,
    write_conversion_manifest,
    write_conversion_report,
    write_split_manifest,
)


class WanderingManifestContractTest(unittest.TestCase):
    def _source(self) -> SourceFile:
        return SourceFile(
            role="primary",
            path="raw/source.json",
            byte_count=12,
            sha256="a" * 64,
        )

    def _output(self) -> OutputArtifact:
        return OutputArtifact(
            role="samples",
            path="samples.jsonl",
            byte_count=256,
            sha256="b" * 64,
            sample_count=2,
        )

    def _sample_ids(self) -> tuple[str, str]:
        return (
            derive_sample_id(
                source_dataset="smartcare",
                source_role="primary",
                source_sha256="a" * 64,
                source_record_id="trajectory-1",
            ),
            derive_sample_id(
                source_dataset="smartcare",
                source_role="primary",
                source_sha256="a" * 64,
                source_record_id="trajectory-2",
            ),
        )

    def _manifest(self) -> ConversionManifest:
        return ConversionManifest(
            schema_version=SOURCE_MANIFEST_SCHEMA_VERSION,
            source_name="smartcare",
            converter_version="wandering-smartcare-converter-v1",
            source_files=(self._source(),),
            outputs=(self._output(),),
            sample_count_read=2,
            sample_count_written=2,
            sample_count_rejected=0,
            sample_ids=self._sample_ids(),
            warnings=(),
        )

    def _report(self) -> ConversionReport:
        return ConversionReport(
            schema_version=CONVERSION_REPORT_SCHEMA_VERSION,
            source_name="smartcare",
            converter_version="wandering-smartcare-converter-v1",
            source_sha256=("a" * 64,),
            sample_count_read=2,
            sample_count_written=2,
            sample_count_rejected=0,
            class_counts={"direct": 1, "unknown": 1},
            length_summary={"max": 3, "mean": 3.0, "min": 3},
            coordinate_summary={
                "x_max": 1.0,
                "x_min": 0.0,
                "y_max": 1.0,
                "y_min": 0.0,
            },
            group_fields_found=(),
            warnings=(),
            output_sha256={"samples": "b" * 64},
        )

    def test_same_source_produces_same_id_manifest_and_does_not_modify_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw" / "source.json"
            raw.parent.mkdir()
            raw.write_bytes(b'{"safe":true}\n')
            original_bytes = raw.read_bytes()
            original_mtime_ns = raw.stat().st_mtime_ns

            first_source = describe_source_file(
                raw,
                source_root=root,
                role="primary",
            )
            second_source = describe_source_file(
                raw,
                source_root=root,
                role="primary",
            )
            first_id = derive_sample_id(
                source_dataset="smartcare",
                source_role=first_source.role,
                source_sha256=first_source.sha256,
                source_record_id="07/10/2020 06:00",
            )
            second_id = derive_sample_id(
                source_dataset="smartcare",
                source_role=second_source.role,
                source_sha256=second_source.sha256,
                source_record_id="07/10/2020 06:00",
            )
            output = OutputArtifact(
                role="samples",
                path="samples.jsonl",
                byte_count=1,
                sha256="c" * 64,
                sample_count=1,
            )
            first_manifest = ConversionManifest(
                schema_version=SOURCE_MANIFEST_SCHEMA_VERSION,
                source_name="smartcare",
                converter_version="wandering-smartcare-converter-v1",
                source_files=(first_source,),
                outputs=(output,),
                sample_count_read=1,
                sample_count_written=1,
                sample_count_rejected=0,
                sample_ids=(first_id,),
                warnings=(),
            )
            second_manifest = ConversionManifest(
                schema_version=SOURCE_MANIFEST_SCHEMA_VERSION,
                source_name="smartcare",
                converter_version="wandering-smartcare-converter-v1",
                source_files=(second_source,),
                outputs=(output,),
                sample_count_read=1,
                sample_count_written=1,
                sample_count_rejected=0,
                sample_ids=(second_id,),
                warnings=(),
            )

            self.assertEqual(first_id, second_id)
            self.assertEqual(first_manifest.to_bytes(), second_manifest.to_bytes())
            self.assertEqual(first_manifest.sha256, second_manifest.sha256)
            self.assertEqual(raw.read_bytes(), original_bytes)
            self.assertEqual(raw.stat().st_mtime_ns, original_mtime_ns)

    def test_duplicate_ids_and_invalid_hashes_fail_closed(self) -> None:
        duplicate_id = self._sample_ids()[0]
        with self.assertRaisesRegex(ManifestValidationError, "duplicate sample_id"):
            ConversionManifest(
                schema_version=SOURCE_MANIFEST_SCHEMA_VERSION,
                source_name="smartcare",
                converter_version="wandering-smartcare-converter-v1",
                source_files=(self._source(),),
                outputs=(self._output(),),
                sample_count_read=2,
                sample_count_written=2,
                sample_count_rejected=0,
                sample_ids=(duplicate_id, duplicate_id),
                warnings=(),
            )

        invalid_hashes = ("not-a-hash", "A" * 64, "a" * 63)
        for digest in invalid_hashes:
            with self.subTest(digest=digest):
                with self.assertRaisesRegex(ManifestValidationError, "SHA-256"):
                    SourceFile(
                        role="primary",
                        path="raw/source.json",
                        byte_count=12,
                        sha256=digest,
                    )

    def test_manifest_and_report_are_strict_and_cross_checked(self) -> None:
        manifest = self._manifest()
        report = self._report()

        validate_conversion_bundle(manifest, report)
        self.assertEqual(
            ConversionManifest.from_dict(manifest.to_dict()),
            manifest,
        )
        self.assertEqual(ConversionReport.from_dict(report.to_dict()), report)

        mismatched_report = ConversionReport(
            **{
                **report.__dict__,
                "output_sha256": {"samples": "d" * 64},
            }
        )
        with self.assertRaisesRegex(ManifestValidationError, "output SHA-256"):
            validate_conversion_bundle(manifest, mismatched_report)

        invalid_record = manifest.to_dict()
        invalid_record["unexpected"] = True
        with self.assertRaisesRegex(ManifestValidationError, "unknown fields"):
            ConversionManifest.from_dict(invalid_record)

    def test_reports_enforce_count_invariants_and_structured_warnings(self) -> None:
        warning = ConversionWarning(
            code="trajectory_too_short",
            count=1,
            sample_ids=("smartcare_rejected_1",),
        )
        with self.assertRaisesRegex(ManifestValidationError, "read.*written.*rejected"):
            ConversionReport(
                schema_version=CONVERSION_REPORT_SCHEMA_VERSION,
                source_name="smartcare",
                converter_version="wandering-smartcare-converter-v1",
                source_sha256=("a" * 64,),
                sample_count_read=3,
                sample_count_written=1,
                sample_count_rejected=1,
                class_counts={"direct": 1},
                length_summary={},
                coordinate_summary={},
                group_fields_found=(),
                warnings=(warning,),
                output_sha256={"samples": "b" * 64},
            )


class WanderingSplitManifestTest(unittest.TestCase):
    def _build(self) -> SplitManifest:
        return build_split_manifest(
            split_version=SPLIT_MANIFEST_SCHEMA_VERSION,
            seed=20260731,
            group_policy="source_specific",
            train=("sample-a",),
            validation=("sample-b",),
            test=("sample-c",),
            sealed_external_test=("sample-d",),
            source_hashes={"smartcare": "a" * 64},
            known_sample_ids=("sample-a", "sample-b", "sample-c", "sample-d"),
        )

    def test_all_partitions_are_disjoint_complete_and_hash_bound(self) -> None:
        split = self._build()

        self.assertEqual(SplitManifest.from_dict(split.to_dict()), split)
        self.assertEqual(len(split.split_sha256), 64)
        self.assertEqual(
            set(split.all_sample_ids),
            {"sample-a", "sample-b", "sample-c", "sample-d"},
        )

        tampered = split.to_dict()
        tampered["seed"] = 1
        with self.assertRaisesRegex(ManifestValidationError, "split_sha256"):
            SplitManifest.from_dict(tampered)

    def test_overlap_duplicate_unknown_and_missing_ids_fail_closed(self) -> None:
        common = {
            "split_version": SPLIT_MANIFEST_SCHEMA_VERSION,
            "seed": 20260731,
            "group_policy": "source_specific",
            "source_hashes": {"smartcare": "a" * 64},
        }

        with self.assertRaisesRegex(ManifestValidationError, "overlap"):
            build_split_manifest(
                **common,
                train=("sample-a",),
                validation=("sample-b",),
                test=("sample-a",),
                sealed_external_test=(),
                known_sample_ids=("sample-a", "sample-b"),
            )

        with self.assertRaisesRegex(ManifestValidationError, "duplicate sample_id"):
            build_split_manifest(
                **common,
                train=("sample-a", "sample-a"),
                validation=(),
                test=(),
                sealed_external_test=(),
                known_sample_ids=("sample-a",),
            )

        with self.assertRaisesRegex(ManifestValidationError, "unknown sample_id"):
            build_split_manifest(
                **common,
                train=("sample-a",),
                validation=("unknown",),
                test=(),
                sealed_external_test=(),
                known_sample_ids=("sample-a",),
            )

        with self.assertRaisesRegex(ManifestValidationError, "missing sample_id"):
            build_split_manifest(
                **common,
                train=("sample-a",),
                validation=(),
                test=(),
                sealed_external_test=(),
                known_sample_ids=("sample-a", "sample-b"),
            )

    def test_writers_are_deterministic_and_split_sidecar_matches(self) -> None:
        manifest = WanderingManifestContractTest()._manifest()
        report = WanderingManifestContractTest()._report()
        split = self._build()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_manifest = root / "first-manifest.json"
            second_manifest = root / "second-manifest.json"
            first_report = root / "first-report.json"
            second_report = root / "second-report.json"
            split_path = root / "split_v1.json"
            split_sha_path = root / "split_v1.sha256"

            write_conversion_manifest(first_manifest, manifest)
            write_conversion_manifest(second_manifest, manifest)
            write_conversion_report(first_report, report)
            write_conversion_report(second_report, report)
            write_split_manifest(
                split_path,
                split,
                checksum_path=split_sha_path,
            )

            self.assertEqual(first_manifest.read_bytes(), second_manifest.read_bytes())
            self.assertEqual(first_report.read_bytes(), second_report.read_bytes())
            self.assertEqual(
                split_sha_path.read_text(encoding="ascii"),
                f"{split.split_sha256}\n",
            )


if __name__ == "__main__":
    unittest.main()
