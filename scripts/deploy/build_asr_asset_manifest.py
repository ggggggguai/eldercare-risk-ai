#!/usr/bin/env python
"""Build the frozen external ASR asset manifest from a verified local package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from elderly_monitoring.modules.asr.integrity import verify_model_package


PACKAGE_RELATIVE = Path("models/asr/asr-paraformer-zh-v1.0")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-version", default="2026-08-20.1")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    package_root = project_root / PACKAGE_RELATIVE
    package_manifest = package_root / "sha256sums.txt"
    # Verify the full frozen package before deriving deployment descriptors.
    # The Windows checkout may expand only the checksum file to CRLF; its
    # entries and every bound model file must still pass the original verifier.
    verify_model_package(package_root)
    normalized = package_manifest.read_bytes().replace(b"\r\n", b"\n")
    if normalized != package_manifest.read_bytes():
        observed_manifest = hashlib.sha256(normalized).hexdigest()
    else:
        observed_manifest = hashlib.sha256(normalized).hexdigest()
    entries: list[dict[str, object]] = []
    for path in sorted(package_root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative_in_package = path.relative_to(package_root).as_posix()
        if relative_in_package == "sha256sums.txt":
            raw = normalized
            size = len(raw)
            digest = hashlib.sha256(raw).hexdigest()
        else:
            size = path.stat().st_size
            digest = _sha256_file(path)
        if relative_in_package.startswith("paraformer/"):
            component = "paraformer"
        elif relative_in_package.startswith("fsmn-vad/"):
            component = "fsmn_vad"
        elif relative_in_package.startswith("ct-punc/"):
            component = "ct_punc"
        else:
            component = "package_integrity"
        entries.append(
            {
                "component": component,
                "path": f"{PACKAGE_RELATIVE.as_posix()}/{relative_in_package}",
                "sha256": digest,
                "size_bytes": size,
            }
        )
    payload = {
        "assets": entries,
        "manifest_version": args.manifest_version,
        "model_version": "asr-paraformer-zh-v1.0",
        "package_manifest_sha256": observed_manifest,
        "package_root": PACKAGE_RELATIVE.as_posix(),
        "schema_version": "asr-external-assets-v1",
        "storage_policy": {
            "credentials_in_repository": False,
            "deployment_source": "tencent-cloud-read-only-mount-or-download",
            "git_contains_assets": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"status": "passed", "asset_count": len(entries)}))
    return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
