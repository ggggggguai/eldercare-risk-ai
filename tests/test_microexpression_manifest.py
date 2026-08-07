from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.manifest import (
    ManifestBuildError,
    build_microexpression_manifest,
    normalize_smic_subject,
    write_microexpression_manifest,
)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_frames(frame_dir: Path, extension: str, count: int = 2) -> None:
    frame_dir.mkdir(parents=True)
    for index in range(1, count + 1):
        (frame_dir / f"frame{index}{extension}").write_bytes(b"frame")


def _dataset_fixture(tmp_path: Path) -> dict[str, Path]:
    casme_root = tmp_path / "casme"
    smic_root = tmp_path / "smic"
    casme_csv = tmp_path / "casme.csv"
    smic_csv = tmp_path / "smic.csv"

    _write_frames(casme_root / "sub01" / "EP01_01", ".jpg", count=3)
    _write_frames(
        smic_root / "s1" / "micro" / "negative" / "s1_ne_01", ".bmp"
    )
    _write_frames(smic_root / "s1" / "non_micro" / "s1_n1", ".bmp")

    _write_csv(
        casme_csv,
        ["dataset", "sub", "filename_o", "label", "Action Units"],
        [
            {
                "dataset": "casme2",
                "sub": "sub01",
                "filename_o": "EP01_01",
                "label": 1,
                "Action Units": "12",
            }
        ],
    )
    _write_csv(
        smic_csv,
        ["dataset", "sub", "filename_o", "label"],
        [
            {
                "dataset": "smic",
                "sub": "s01",
                "filename_o": "s01_ne_01",
                "label": 0,
            }
        ],
    )
    return {
        "casme_root": casme_root,
        "smic_root": smic_root,
        "casme_csv": casme_csv,
        "smic_csv": smic_csv,
    }


def test_normalize_smic_subject_preserves_canonical_and_local_ids() -> None:
    assert normalize_smic_subject("s01") == ("s01", "s1")
    assert normalize_smic_subject("S11") == ("s11", "s11")
    with pytest.raises(ManifestBuildError):
        normalize_smic_subject("subject01")


def test_build_manifest_maps_both_datasets_and_keeps_non_micro_outside_labels(
    tmp_path: Path,
) -> None:
    paths = _dataset_fixture(tmp_path)
    records = build_microexpression_manifest(**paths)

    assert len(records) == 3
    by_id = {record["sample_id"]: record for record in records}

    smic = by_id["smic_hs__s01__s1_ne_01"]
    assert smic["source_subject_id"] == "s01"
    assert smic["local_subject_id"] == "s1"
    assert smic["label"] == 0
    assert smic["label_name"] == "negative"
    assert smic["frame_count"] == 2
    assert smic["frame_extension"] == ".bmp"
    assert smic["frame_annotation_source"] == "missing"

    non_micro = by_id["smic_hs__s01__s1_n1"]
    assert non_micro["sample_role"] == "spotting_negative"
    assert non_micro["label"] is None
    assert non_micro["label_name"] == "non_micro"

    casme = by_id["casme2__sub01__EP01_01"]
    assert casme["action_units"] == "12"
    assert casme["frame_count"] == 3
    assert casme["evaluation_scope"] == "awaiting_estimated_or_official_frames"


def test_label_directory_mismatch_fails_instead_of_guessing(tmp_path: Path) -> None:
    paths = _dataset_fixture(tmp_path)
    _write_csv(
        paths["smic_csv"],
        ["dataset", "sub", "filename_o", "label"],
        [
            {
                "dataset": "smic",
                "sub": "s01",
                "filename_o": "s01_ne_01",
                "label": 1,
            }
        ],
    )

    with pytest.raises(ManifestBuildError, match="Label-directory mismatch"):
        build_microexpression_manifest(**paths)


def test_duplicate_sample_id_is_rejected(tmp_path: Path) -> None:
    paths = _dataset_fixture(tmp_path)
    duplicate = {
        "dataset": "casme2",
        "sub": "sub01",
        "filename_o": "EP01_01",
        "label": 1,
        "Action Units": "12",
    }
    _write_csv(
        paths["casme_csv"],
        ["dataset", "sub", "filename_o", "label", "Action Units"],
        [duplicate, duplicate],
    )

    with pytest.raises(ManifestBuildError, match="Duplicate sample_id"):
        build_microexpression_manifest(**paths)


def test_write_manifest_outputs_auditable_files(tmp_path: Path) -> None:
    paths = _dataset_fixture(tmp_path)
    records = build_microexpression_manifest(**paths)
    output_dir = tmp_path / "output"

    summary = write_microexpression_manifest(
        records=records, output_dir=output_dir, **paths
    )

    manifest_path = output_dir / "sequence_manifest_v1.jsonl"
    mapping_path = output_dir / "path_mapping_v1.csv"
    summary_path = output_dir / "quality_summary_v1.json"
    payloads = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(payloads) == 3
    assert len(mapping_path.read_text(encoding="utf-8-sig").splitlines()) == 4
    assert summary["manifest"]["sha256"] == persisted_summary["manifest"]["sha256"]
    assert len(summary["manifest"]["sha256"]) == 64
    assert summary["manifest"]["excluded_count"] == 0
    assert summary["validation"]["non_micro_outside_class_label_space"]
