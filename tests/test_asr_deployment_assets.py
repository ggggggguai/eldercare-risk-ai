from __future__ import annotations

import hashlib
import json

import pytest

from elderly_monitoring.modules.asr.deployment_assets import (
    ASRDeploymentAssetError,
    load_asr_asset_manifest,
    verify_asr_assets,
)
from elderly_monitoring.modules.asr.integrity import render_checksum_manifest


def _bundle(tmp_path):
    package = tmp_path / "models" / "asr" / "asr-paraformer-zh-v1.0"
    package.mkdir(parents=True)
    model = package / "model.pt"
    model.write_bytes(b"model")
    checksum = package / "sha256sums.txt"
    checksum.write_text(render_checksum_manifest(package), encoding="utf-8", newline="\n")
    assets = []
    for path in sorted(package.iterdir()):
        raw = path.read_bytes()
        assets.append(
            {
                "component": "package_integrity" if path.name == "sha256sums.txt" else "paraformer",
                "path": path.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    payload = {
        "assets": assets,
        "manifest_version": "test",
        "model_version": "asr-paraformer-zh-v1.0",
        "package_manifest_sha256": hashlib.sha256(checksum.read_bytes()).hexdigest(),
        "package_root": "models/asr/asr-paraformer-zh-v1.0",
        "schema_version": "asr-external-assets-v1",
        "storage_policy": {
            "credentials_in_repository": False,
            "deployment_source": "test",
            "git_contains_assets": False,
        },
    }
    manifest = tmp_path / "asset_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, model


def test_external_asr_bundle_is_strictly_verified(tmp_path) -> None:
    manifest, model = _bundle(tmp_path)
    result = verify_asr_assets(project_root=tmp_path, manifest_path=manifest)
    assert result["status"] == "passed"
    model.write_bytes(b"drift")
    with pytest.raises(ASRDeploymentAssetError, match="verification failed"):
        verify_asr_assets(project_root=tmp_path, manifest_path=manifest)


def test_manifest_rejects_unsafe_asset_paths(tmp_path) -> None:
    manifest, _ = _bundle(tmp_path)
    payload = load_asr_asset_manifest(manifest)
    payload["assets"][0]["path"] = "../escape.pt"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ASRDeploymentAssetError, match="unsafe"):
        load_asr_asset_manifest(manifest)
