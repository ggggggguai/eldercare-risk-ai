from __future__ import annotations

import csv
import json
from pathlib import Path

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.manifest import (
    audit_smic_mapping,
    build_microexpression_manifest,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "sub", "filename_o", "label"])
        writer.writeheader()
        writer.writerows(rows)


def _make_fixture(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    casme_root = tmp_path / "casme"
    casme_dir = casme_root / "sub01" / "EP01"
    casme_dir.mkdir(parents=True)
    (casme_dir / "frame1.jpg").write_bytes(b"x")
    casme_csv = tmp_path / "casme.csv"
    with casme_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset", "sub", "filename_o", "label"]
        )
        writer.writeheader()
        writer.writerow(
            {"dataset": "casme2", "sub": "sub01", "filename_o": "EP01", "label": 0}
        )

    smic_root = tmp_path / "smic"
    micro_dir = smic_root / "s1" / "micro" / "negative" / "s1_ne_01"
    non_micro_dir = smic_root / "s1" / "non_micro" / "s1_n1"
    micro_dir.mkdir(parents=True)
    non_micro_dir.mkdir(parents=True)
    (micro_dir / "frame1.bmp").write_bytes(b"x")
    (non_micro_dir / "frame1.bmp").write_bytes(b"x")
    smic_csv = tmp_path / "smic.csv"
    _write_csv(
        smic_csv,
        [{"dataset": "smic", "sub": "s01", "filename_o": "s01_ne_01", "label": 0}],
    )

    records = build_microexpression_manifest(
        casme_root=casme_root,
        smic_root=smic_root,
        casme_csv=casme_csv,
        smic_csv=smic_csv,
    )
    return smic_root, smic_csv, records


def test_smic_audit_passes_canonical_to_local_mapping(tmp_path: Path) -> None:
    smic_root, smic_csv, records = _make_fixture(tmp_path)
    report = audit_smic_mapping(
        smic_root=smic_root, smic_csv=smic_csv, manifest_records=records
    )

    assert report["status"] == "pass"
    assert report["issues"] == []
    assert report["csv"]["row_count"] == 1
    assert report["manifest"]["classification_count"] == 1
    assert report["manifest"]["spotting_negative_count"] == 1
    assert report["row_checks"]["label_directory_match_rows"] == 1
    assert report["subject_summary"]["s01"]["local_subject_id"] == "s1"


def test_smic_audit_reports_label_directory_mismatch(tmp_path: Path) -> None:
    smic_root, smic_csv, records = _make_fixture(tmp_path)
    smic_csv.write_text(
        "dataset,sub,filename_o,label\nsmic,s01,s01_ne_01,1\n", encoding="utf-8"
    )
    report = audit_smic_mapping(
        smic_root=smic_root, smic_csv=smic_csv, manifest_records=records
    )

    assert report["status"] == "fail"
    assert any(issue["code"] == "missing_expected_micro_dir" for issue in report["issues"])


def test_smic_audit_reports_unlisted_micro_directory(tmp_path: Path) -> None:
    smic_root, smic_csv, records = _make_fixture(tmp_path)
    extra = smic_root / "s1" / "micro" / "negative" / "s1_ne_02"
    extra.mkdir(parents=True)
    (extra / "frame1.bmp").write_bytes(b"x")

    report = audit_smic_mapping(
        smic_root=smic_root, smic_csv=smic_csv, manifest_records=records
    )

    assert report["status"] == "fail"
    assert any(
        issue["code"] == "micro_directory_csv_set_mismatch"
        for issue in report["issues"]
    )


def test_smic_audit_reports_sequence_subject_prefix_mismatch(tmp_path: Path) -> None:
    smic_root, smic_csv, records = _make_fixture(tmp_path)
    smic_csv.write_text(
        "dataset,sub,filename_o,label\nsmic,s01,s02_ne_01,0\n", encoding="utf-8"
    )

    report = audit_smic_mapping(
        smic_root=smic_root, smic_csv=smic_csv, manifest_records=records
    )

    assert report["status"] == "fail"
    assert any(
        issue["code"] == "sequence_subject_prefix_mismatch"
        for issue in report["issues"]
    )


def test_smic_audit_reports_duplicate_csv_sample_id(tmp_path: Path) -> None:
    smic_root, smic_csv, records = _make_fixture(tmp_path)
    duplicate_row = "smic,s01,s01_ne_01,0\n"
    smic_csv.write_text(
        "dataset,sub,filename_o,label\n" + duplicate_row + duplicate_row,
        encoding="utf-8",
    )

    report = audit_smic_mapping(
        smic_root=smic_root, smic_csv=smic_csv, manifest_records=records
    )

    assert report["status"] == "fail"
    assert any(
        issue["code"] == "duplicate_csv_sample_id" for issue in report["issues"]
    )
