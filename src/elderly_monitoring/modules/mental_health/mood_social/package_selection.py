"""Resolve the active MoodSocial package from the versioned ART decision."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_ART002_DECISION_PATH = (
    PROJECT_ROOT
    / "models/mental_health/mood_social/v3.3.3/candidates/MH-20260804-023/art002_decision_manifest.json"
)
EXPECTED_MODEL_VERSION = "mood-fusion-v3.3.3"
EXPECTED_ACTIVE_RUN_ID = "MH-20260802-013"


class PackageSelectionError(RuntimeError):
    """Raised when the versioned package decision is missing or inconsistent."""


@dataclass(frozen=True)
class ActivePackageSelection:
    decision_run_id: str
    promotion_passed: bool
    package_run_id: str
    package_directory: Path
    model_version: str


def load_active_package_selection(
    decision_path: str | Path = DEFAULT_ART002_DECISION_PATH,
    *,
    repository_root: str | Path = PROJECT_ROOT,
) -> ActivePackageSelection:
    root = Path(repository_root).resolve()
    path = Path(decision_path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackageSelectionError("ART-002 package decision is unavailable") from exc
    if not isinstance(payload, Mapping):
        raise PackageSelectionError("ART-002 package decision must be an object")
    if (
        payload.get("task_id") != "ART-002"
        or payload.get("run_id") != "MH-20260804-023"
    ):
        raise PackageSelectionError("ART-002 package decision identity changed")
    if (
        payload.get("promotion_passed") is not False
        or payload.get("new_online_package_created") is not False
    ):
        raise PackageSelectionError(
            "this API-003 path only permits the audited no-promotion decision"
        )
    if payload.get("active_package_run_id") != EXPECTED_ACTIVE_RUN_ID:
        raise PackageSelectionError("active fallback package changed")
    relative = Path(str(payload.get("active_package_directory", "")))
    if relative.is_absolute():
        package = relative.resolve()
    else:
        package = (root / relative).resolve()
    model_root = (root / "models/mental_health/mood_social").resolve()
    try:
        package.relative_to(model_root)
    except ValueError as exc:
        raise PackageSelectionError("active package escapes model root") from exc
    manifest = package / "manifest.json"
    checksums = package / "SHA256SUMS"
    upstream = payload.get("upstream")
    if not isinstance(upstream, Mapping):
        raise PackageSelectionError("ART-002 upstream protection is missing")
    if _sha256_file(manifest) != str(upstream.get("fallback_manifest_sha256")):
        raise PackageSelectionError("active package manifest hash changed")
    if _sha256_file(checksums) != str(upstream.get("fallback_checksums_sha256")):
        raise PackageSelectionError("active package checksum hash changed")
    return ActivePackageSelection(
        decision_run_id="MH-20260804-023",
        promotion_passed=False,
        package_run_id=EXPECTED_ACTIVE_RUN_ID,
        package_directory=package,
        model_version=EXPECTED_MODEL_VERSION,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PackageSelectionError(
            f"active package file is missing: {path.name}"
        ) from exc
    return digest.hexdigest()


__all__ = [
    "ActivePackageSelection",
    "DEFAULT_ART002_DECISION_PATH",
    "PackageSelectionError",
    "load_active_package_selection",
]
