"""Strict file-level verification for externally mounted psychological assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


class MentalHealthAssetError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise MentalHealthAssetError("asset path must be a non-empty string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "\\" in value or ":" in value:
        raise MentalHealthAssetError(f"unsafe asset path: {value!r}")
    return Path(*pure.parts)


def load_asset_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MentalHealthAssetError("cannot read asset manifest") from exc
    if payload.get("schema_version") != "mental-health-external-assets-v1":
        raise MentalHealthAssetError("asset manifest schema drifted")
    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise MentalHealthAssetError("asset manifest is empty")
    seen: set[str] = set()
    for item in assets:
        if not isinstance(item, dict) or set(item) != {
            "component",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise MentalHealthAssetError("asset descriptor fields drifted")
        relative = _safe_relative(item["path"]).as_posix()
        if relative in seen:
            raise MentalHealthAssetError(f"duplicate asset path: {relative}")
        seen.add(relative)
        digest = item["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise MentalHealthAssetError(f"invalid asset SHA-256: {relative}")
        if not isinstance(item["size_bytes"], int) or item["size_bytes"] < 1:
            raise MentalHealthAssetError(f"invalid asset size: {relative}")
    return payload


def verify_asset_manifest(
    *, project_root: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    payload = load_asset_manifest(manifest_path)
    failures: list[dict[str, Any]] = []
    component_counts: dict[str, int] = {}
    for item in payload["assets"]:
        relative = _safe_relative(item["path"])
        target = (root / relative).resolve()
        if root not in target.parents:
            raise MentalHealthAssetError("asset escaped project root")
        component = str(item["component"])
        component_counts[component] = component_counts.get(component, 0) + 1
        if not target.is_file():
            failures.append({"path": relative.as_posix(), "reason": "missing"})
            continue
        observed_size = target.stat().st_size
        if observed_size != item["size_bytes"]:
            failures.append(
                {
                    "path": relative.as_posix(),
                    "reason": "size_mismatch",
                    "observed_size": observed_size,
                }
            )
            continue
        observed_hash = _sha256(target)
        if observed_hash != item["sha256"]:
            failures.append(
                {
                    "path": relative.as_posix(),
                    "reason": "sha256_mismatch",
                    "observed_sha256": observed_hash,
                }
            )
    if failures:
        first = failures[0]
        raise MentalHealthAssetError(
            f"asset verification failed ({len(failures)} files); first={first['path']}:{first['reason']}"
        )
    return {
        "status": "passed",
        "asset_count": len(payload["assets"]),
        "components": component_counts,
        "manifest_version": payload.get("manifest_version"),
    }


__all__ = [
    "MentalHealthAssetError",
    "load_asset_manifest",
    "verify_asset_manifest",
]
