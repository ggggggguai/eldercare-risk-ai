"""Generate and verify the frozen ASR model package checksum manifest."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath


CHECKSUM_FILENAME = "sha256sums.txt"
_CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


class ASRPackageIntegrityError(RuntimeError):
    """Raised when the local ASR package differs from its frozen manifest."""


def render_checksum_manifest(model_root: str | Path) -> str:
    root = Path(model_root).resolve()
    files = _package_files(root)
    if not files:
        raise ASRPackageIntegrityError(f"ASR model package is empty: {root}")
    lines = [f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files]
    return "\n".join(lines) + "\n"


def verify_model_package(model_root: str | Path) -> str:
    """Verify all package files and return the checksum manifest SHA-256."""

    root = Path(model_root).resolve()
    manifest_path = root / CHECKSUM_FILENAME
    if not manifest_path.is_file():
        raise ASRPackageIntegrityError(f"ASR checksum manifest is missing: {manifest_path}")

    entries: list[tuple[str, str]] = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ASRPackageIntegrityError(
                f"invalid ASR checksum line {line_number}: {line!r}"
            )
        digest, relative = match.groups()
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ASRPackageIntegrityError(
                f"unsafe ASR checksum path on line {line_number}: {relative!r}"
            )
        entries.append((relative, digest))

    paths = [relative for relative, _ in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ASRPackageIntegrityError(
            "ASR checksum paths must be unique and sorted by Unicode code point"
        )

    actual_paths = [path.relative_to(root).as_posix() for path in _package_files(root)]
    if paths != actual_paths:
        missing = sorted(set(paths) - set(actual_paths))
        unexpected = sorted(set(actual_paths) - set(paths))
        raise ASRPackageIntegrityError(
            f"ASR package file set changed; missing={missing}, unexpected={unexpected}"
        )

    for relative, expected_digest in entries:
        actual_digest = _sha256_file(root / Path(relative))
        if actual_digest != expected_digest:
            raise ASRPackageIntegrityError(
                f"ASR package checksum mismatch: {relative}"
            )
    return _sha256_file(manifest_path)


def _package_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ASRPackageIntegrityError(f"ASR model package is unavailable: {root}")
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.name != CHECKSUM_FILENAME
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ASRPackageIntegrityError",
    "CHECKSUM_FILENAME",
    "render_checksum_manifest",
    "verify_model_package",
]
