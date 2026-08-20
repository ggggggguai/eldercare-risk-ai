#!/usr/bin/env python
"""Build the file-level external asset manifest for mental-health deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PACKAGE_ROOTS = {
    "mood_social_r7": "models/mental_health/mood_social/v3.3.3-r7/packages/MH-20260813-R7-001",
    "mood_social_r8": "models/mental_health/mood_social/v3.3.3-r8/packages/MH-20260814-R8-001",
    "mood_social_r9": "models/mental_health/mood_social/v3.3.3-r9/packages/MH-20260814-R9-001",
    "mood_social_r11": "models/mental_health/mood_social/v3.3.3-r11/packages/MH-20260814-R11-001",
    "mood_social_forecast_v3_4": "models/mental_health/mood_social/v3.4.0/packages/MH-20260810-FDEP-001",
    "cognitive_encoder_v3_3": "models/mental_health/cognitive_change_clue/v3.3.0",
    "cognitive_subject_v3_5": "models/mental_health/cognitive_change_clue/v3.5.0",
    "facial_affect_b0": "models/mental_health/facial_affect/b0_opt_me_008_deployment_v1",
}

EXTRA_EXISTING = {
    "wandering_preprocessing": (
        "data/processed/wandering/preprocessing/v1/manifest.json",
        "data/processed/wandering/preprocessing/v1/feature_stats.json",
    ),
    "wandering_tracker": ("yolov8n-pose.pt",),
}

MISSING_FIXED = (
    {
        "component": "wandering_primary_candidate",
        "path": "reports/mental_health/wandering_performance/m0r_score_entry_hardening_v1/artifacts/topowander_m0r_candidate_v3/candidate_manifest.json",
        "size_bytes": 12642,
        "sha256": "3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7",
    },
    {
        "component": "wandering_primary_candidate",
        "path": "reports/mental_health/wandering_performance/m0r_score_entry_hardening_v1/artifacts/topowander_m0r_candidate_v3/model_state.npz",
        "size_bytes": 838934,
        "sha256": "94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031",
    },
    {
        "component": "wandering_primary_candidate",
        "path": "reports/mental_health/wandering_performance/m0r_score_entry_hardening_v1/artifacts/topowander_m0r_candidate_v3/forward_config.yaml",
        "size_bytes": 4469,
        "sha256": "debd3adcf0c84d266ca594fd9ba2716a5a854e074e6e2092570b1ec656ed80f7",
    },
    {
        "component": "wandering_primary_candidate",
        "path": "reports/mental_health/wandering_performance/m0r_score_entry_hardening_v1/artifacts/topowander_m0r_candidate_v3/performance_config.yaml",
        "size_bytes": 1840,
        "sha256": "ecc4c5a00dc9c30b1a4a16b943d39d9de85cce27077c8ccba1ba223c529de5f2",
    },
    {
        "component": "wandering_primary_candidate",
        "path": "reports/mental_health/wandering_performance/m0r_score_entry_hardening_v1/artifacts/topowander_m0r_candidate_v3/frozen_wp_rf_config.yaml",
        "size_bytes": 3398,
        "sha256": "d29db531f42ae5d7cf4ed540a85681ddab78e483695633ccc8b382d98e745e35",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum_entries(package: Path) -> dict[str, str]:
    for name in ("SHA256SUMS", "sha256sums.txt"):
        checksum = package / name
        if not checksum.is_file():
            continue
        entries: dict[str, str] = {}
        for line in checksum.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            digest, relative = stripped.split(maxsplit=1)
            relative = relative.lstrip("*")
            entries[Path(relative).as_posix()] = digest.lower()
        return entries
    return {}


def descriptor(
    root: Path,
    component: str,
    source: Path,
    target: str,
    *,
    frozen_sha256: str | None = None,
) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    size = len(raw)
    if frozen_sha256 is not None and digest != frozen_sha256:
        normalized = raw.replace(b"\r\n", b"\n")
        normalized_digest = hashlib.sha256(normalized).hexdigest()
        if normalized_digest != frozen_sha256:
            raise ValueError(
                f"{source} differs from its frozen checksum after LF normalization"
            )
        digest = normalized_digest
        size = len(normalized)
    return {
        "component": component,
        "path": target,
        "sha256": digest,
        "size_bytes": size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    assets: list[dict] = []
    for component, relative in PACKAGE_ROOTS.items():
        package = root / relative
        if not package.is_dir():
            raise FileNotFoundError(package)
        checksums = _checksum_entries(package)
        for source in sorted(path for path in package.rglob("*") if path.is_file()):
            target = source.relative_to(root).as_posix()
            within_package = source.relative_to(package).as_posix()
            assets.append(
                descriptor(
                    root,
                    component,
                    source,
                    target,
                    frozen_sha256=checksums.get(within_package),
                )
            )
    for component, relatives in EXTRA_EXISTING.items():
        for relative in relatives:
            source = root / relative
            target = (
                "models/shared/yolov8n-pose.pt"
                if relative == "yolov8n-pose.pt"
                else relative
            )
            assets.append(descriptor(root, component, source, target))
    assets.extend(MISSING_FIXED)
    assets.sort(key=lambda item: item["path"])
    payload = {
        "schema_version": "mental-health-external-assets-v1",
        "manifest_version": "2026-08-20.1",
        "storage_policy": {
            "git_contains_assets": False,
            "deployment_source": "tencent-cloud-read-only-mount-or-download",
            "credentials_in_repository": False,
        },
        "known_external_blockers": [
            {
                "component": "wandering_primary_candidate",
                "status": "source_files_not_present_in_received_step5_step6_bundle",
                "required_before_wandering_runtime_ready": True,
            }
        ],
        "assets": assets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"asset_count": len(assets), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
