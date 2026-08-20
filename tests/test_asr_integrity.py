from __future__ import annotations

import hashlib

import pytest

from elderly_monitoring.modules.asr.integrity import (
    ASRPackageIntegrityError,
    render_checksum_manifest,
    verify_model_package,
)


def _freeze_package(root) -> None:
    (root / "nested").mkdir(parents=True, exist_ok=True)
    (root / "a.bin").write_bytes(b"model-a")
    (root / "nested" / "b.bin").write_bytes(b"model-b")
    (root / "sha256sums.txt").write_text(
        render_checksum_manifest(root),
        encoding="utf-8",
        newline="\n",
    )


def test_model_package_manifest_is_sorted_and_verified(tmp_path) -> None:
    _freeze_package(tmp_path)
    manifest = tmp_path / "sha256sums.txt"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert [line.split("  ", 1)[1] for line in lines] == ["a.bin", "nested/b.bin"]
    assert verify_model_package(tmp_path) == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_model_package_verification_detects_content_and_file_set_drift(tmp_path) -> None:
    _freeze_package(tmp_path)
    (tmp_path / "a.bin").write_bytes(b"changed")
    with pytest.raises(ASRPackageIntegrityError, match="checksum mismatch"):
        verify_model_package(tmp_path)

    _freeze_package(tmp_path)
    (tmp_path / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(ASRPackageIntegrityError, match="file set changed"):
        verify_model_package(tmp_path)
