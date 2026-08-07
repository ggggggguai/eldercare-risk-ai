from __future__ import annotations

import pytest

from training.cognitive_change_clue.package_v34 import (
    V34PackageError,
    _package_sums,
    fixed_epoch_median,
    verify_package,
)


def test_fixed_epoch_median_uses_five_fold_middle_value() -> None:
    assert fixed_epoch_median([7, 4, 11, 23, 3]) == 7


@pytest.mark.parametrize("values", [[], [1, 2, 3, 4], [1, 2, 0, 4, 5]])
def test_fixed_epoch_median_rejects_invalid_protocol(values: list[int]) -> None:
    with pytest.raises(V34PackageError):
        fixed_epoch_median(values)


def test_package_verifier_rejects_checksum_tampering(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "sha256sums.txt").write_text(_package_sums(tmp_path), encoding="ascii")
    (tmp_path / "manifest.json").write_text('{"tampered": true}\n', encoding="utf-8")

    with pytest.raises(V34PackageError, match="checksums"):
        verify_package(tmp_path, load_model=False)
