"""Strict verification for externally mounted ASR model assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from elderly_monitoring.modules.asr.integrity import verify_model_package


ASR_ASSET_SCHEMA = "asr-external-assets-v1"


class ASRDeploymentAssetError(RuntimeError):
    """Raised when the external deployment bundle differs from its manifest."""


def load_asr_asset_manifest(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ASRDeploymentAssetError("cannot read ASR asset manifest") from exc
    required = {
        "assets",
        "manifest_version",
        "model_version",
        "package_manifest_sha256",
        "package_root",
        "schema_version",
        "storage_policy",
    }
    if set(payload) != required or payload.get("schema_version") != ASR_ASSET_SCHEMA:
        raise ASRDeploymentAssetError("ASR asset manifest schema drifted")
    package_root = _safe_relative(payload.get("package_root"))
    if package_root.as_posix() != "models/asr/asr-paraformer-zh-v1.0":
        raise ASRDeploymentAssetError("ASR package root drifted")
    expected_manifest = payload.get("package_manifest_sha256")
    if not _valid_sha256(expected_manifest):
        raise ASRDeploymentAssetError("invalid ASR package manifest SHA-256")
    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ASRDeploymentAssetError("ASR asset manifest is empty")
    seen: set[str] = set()
    for item in assets:
        if not isinstance(item, dict) or set(item) != {
            "component",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ASRDeploymentAssetError("ASR asset descriptor fields drifted")
        relative = _safe_relative(item["path"]).as_posix()
        if relative in seen:
            raise ASRDeploymentAssetError(f"duplicate ASR asset path: {relative}")
        seen.add(relative)
        if not _valid_sha256(item["sha256"]):
            raise ASRDeploymentAssetError(f"invalid ASR asset SHA-256: {relative}")
        if not isinstance(item["size_bytes"], int) or item["size_bytes"] < 1:
            raise ASRDeploymentAssetError(f"invalid ASR asset size: {relative}")
    return payload


def verify_asr_assets(
    *, project_root: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    payload = load_asr_asset_manifest(manifest_path)
    failures: list[str] = []
    component_counts: dict[str, int] = {}
    for item in payload["assets"]:
        relative = _safe_relative(item["path"])
        target = (root / relative).resolve()
        if root not in target.parents:
            raise ASRDeploymentAssetError("ASR asset escaped project root")
        component = str(item["component"])
        component_counts[component] = component_counts.get(component, 0) + 1
        if not target.is_file():
            failures.append(f"{relative.as_posix()}:missing")
            continue
        if target.stat().st_size != item["size_bytes"]:
            failures.append(f"{relative.as_posix()}:size_mismatch")
            continue
        if _sha256_file(target) != item["sha256"]:
            failures.append(f"{relative.as_posix()}:sha256_mismatch")
    if failures:
        raise ASRDeploymentAssetError(
            f"ASR asset verification failed ({len(failures)} files); first={failures[0]}"
        )
    package_root = root / _safe_relative(payload["package_root"])
    observed_manifest = verify_model_package(package_root)
    if observed_manifest != payload["package_manifest_sha256"]:
        raise ASRDeploymentAssetError("ASR internal package manifest drifted")
    return {
        "status": "passed",
        "asset_count": len(payload["assets"]),
        "components": component_counts,
        "manifest_version": payload["manifest_version"],
        "model_version": payload["model_version"],
        "package_manifest_sha256": observed_manifest,
    }


def _safe_relative(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ASRDeploymentAssetError("ASR asset path must be a non-empty string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "\\" in value or ":" in value:
        raise ASRDeploymentAssetError(f"unsafe ASR asset path: {value!r}")
    return Path(*pure.parts)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ASRDeploymentAssetError",
    "ASR_ASSET_SCHEMA",
    "load_asr_asset_manifest",
    "verify_asr_assets",
]
