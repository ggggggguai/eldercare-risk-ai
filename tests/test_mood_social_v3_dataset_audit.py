from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from scripts.audit.audit_mood_social_v3_datasets import (
    DATASET_SPECS,
    _classify_file,
    _collection_sha256,
    _encode_jsonl,
    _key_profile,
    _public_relative_path,
    _resilient_participant_aliases,
    _sha256_file,
)


def test_data001_freezes_exactly_five_dataset_ids() -> None:
    assert [spec.dataset_id for spec in DATASET_SPECS] == [
        "psyche_d",
        "resilient",
        "nhanes",
        "shenzhen_elderly",
        "hefei_elderly",
    ]
    assert all(
        not spec.adapter_entrypoint.endswith("feature_schema_manifest")
        for spec in DATASET_SPECS
    )


def test_collection_hash_is_order_independent_and_ignores_extra_metadata() -> None:
    rows = [
        {
            "relative_path": "数据集/乙.csv",
            "bytes": 2,
            "sha256": "b" * 64,
            "mtime_ns": 1,
        },
        {
            "relative_path": "data/a.csv",
            "bytes": 1,
            "sha256": "a" * 64,
            "mtime_ns": 2,
        },
    ]
    changed_mtime = [{**row, "mtime_ns": 999} for row in rows]

    assert _collection_sha256(rows) == _collection_sha256(list(reversed(rows)))
    assert _collection_sha256(rows) == _collection_sha256(changed_mtime)


def test_manifest_encoding_is_stable_utf8_jsonl() -> None:
    rows = [{"dataset_id": "深圳", "relative_path": "数据.csv", "bytes": 1}]

    first = _encode_jsonl(rows)
    second = _encode_jsonl(rows)

    assert first == second
    assert first.endswith(b"\n")
    assert "深圳" in first.decode("utf-8")


def test_os_metadata_is_classified_before_csv_extension(tmp_path: Path) -> None:
    root = tmp_path / "RESILIENT"
    metadata = root / "__MACOSX" / "._data.csv"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(b"\x00\x05\x16\x07payload")

    assert _classify_file(metadata, root) == ("os_metadata", "appledouble")


def test_resilient_manifest_aliases_nested_participant_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "RESILIENT"
    source = (
        root
        / "Sleepmat_Watch_Data"
        / "Sleepmat_Watch_Data"
        / "private-participant-key"
        / "ScanWatch_HR.csv"
    )
    source.parent.mkdir(parents=True)
    source.write_text("timestamp,value\n", encoding="utf-8")
    aliases = _resilient_participant_aliases(root)

    public_path = _public_relative_path(DATASET_SPECS[1], source, root, aliases)

    assert "private-participant-key" not in public_path
    assert "participant_0001" in public_path


def test_key_profile_reports_counts_without_returning_values() -> None:
    profile = _key_profile(pd.Series(["A", "Ａ", " a ", None]))

    assert profile == {
        "record_count": 4,
        "missing_key_count": 1,
        "unique_key_count": 1,
        "duplicate_record_count": 2,
        "duplicate_key_group_count": 1,
        "normalisation_collision_group_count": 1,
    }
    assert "values" not in profile


def test_file_hash_does_not_change_when_only_mtime_changes(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"same-content")
    before = _sha256_file(source)

    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    assert _sha256_file(source) == before
