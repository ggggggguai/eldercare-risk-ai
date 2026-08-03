"""Independently validate the ART-001 mood-social package and source bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from elderly_monitoring.modules.mental_health.mood_social.model_package import (
    RUNTIME_EXPERT_NAMES,
    MANIFEST_VERSION,
    OFFLINE_NAMES,
    ONLINE_NAMES,
    PACKAGE_VERSION,
    RUN_ID,
    TASK_ID,
    ModelPackageError,
    load_model_package_config,
)


class ValidationFailure(RuntimeError):
    pass


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        self.count += 1
        if not condition:
            raise ValidationFailure(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/artifacts/mood_social_model_package_v3_3_3.yaml"),
    )
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Verify bytes and metadata without deserializing online models.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = args.config if args.config.is_absolute() else root / args.config
    checks = Checks()
    try:
        config = load_model_package_config(path, repository_root=root)
        result = validate(config, checks, load_models=not args.skip_model_load)
    except (
        ModelPackageError,
        ValidationFailure,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {"status": "pass", "checks": checks.count, **result},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def validate(config: Any, checks: Checks, *, load_models: bool) -> dict[str, Any]:
    package = config.package_directory
    report = config.report_directory
    required_package = {
        "manifest.json",
        "runtime_confidence.json",
        "selection.json",
        "SHA256SUMS",
    }
    for name in required_package:
        checks.require((package / name).is_file(), f"missing package file: {name}")
    required_report = {
        "artifacts.json",
        "package_summary.json",
        "run.json",
        "selection.json",
        "upstream_protection.json",
    }
    for name in required_report:
        checks.require((report / name).is_file(), f"missing ART-001 report: {name}")

    manifest = _json(package / "manifest.json")
    selection = _json(package / "selection.json")
    runtime_confidence = _json(package / "runtime_confidence.json")
    checks.require(manifest["version"] == MANIFEST_VERSION, "manifest version changed")
    checks.require(manifest["task_id"] == TASK_ID, "manifest task changed")
    checks.require(manifest["run_id"] == RUN_ID, "manifest run changed")
    checks.require(
        manifest["package_version"] == PACKAGE_VERSION, "package version changed"
    )
    checks.require(
        selection["promotion_passed"] is True,
        "frozen OPT-FUSION-002 promotion did not pass",
    )
    checks.require(
        selection["selected_default"] == "opt_fusion_002_deployment_candidate",
        "packaged fusion selection changed",
    )
    checks.require(
        manifest["default_fusion_source"] == selection["selected_default"],
        "manifest selection binding changed",
    )
    checks.require(manifest["model006_online"] is False, "MODEL-006 entered online")
    checks.require(
        manifest["model006_http_output"] is False, "MODEL-006 entered HTTP output"
    )
    checks.require(
        manifest["production_threshold_selected"] is False,
        "production threshold was selected",
    )
    checks.require(
        manifest["attention_levels_changed"] is False,
        "attention levels were changed",
    )
    checks.require(
        manifest["runtime_confidence"] == runtime_confidence,
        "runtime confidence manifest binding changed",
    )
    checks.require(
        set(runtime_confidence["expert_reliability"]) == set(RUNTIME_EXPERT_NAMES),
        "runtime confidence expert set changed",
    )
    checks.require(
        all(
            0.0 <= float(runtime_confidence["expert_reliability"][name]) <= 1.0
            for name in RUNTIME_EXPERT_NAMES
        ),
        "runtime confidence value range changed",
    )
    checks.require(
        runtime_confidence["public_dataset_id_required_at_inference"] is False,
        "runtime confidence requires public dataset ID",
    )
    confidence_source = config.repository_root / runtime_confidence["source_path"]
    checks.require(
        _sha256_file(confidence_source) == runtime_confidence["source_sha256"],
        "runtime confidence source hash changed",
    )

    entries = manifest["entries"]
    checks.require(set(entries) == set((*ONLINE_NAMES, *OFFLINE_NAMES)), "entry set changed")
    checks.require(
        sum(entry["scope"] == "online" for entry in entries.values()) == 7,
        "online entry count changed",
    )
    checks.require(
        entries["model006"]["scope"] == "offline"
        and entries["model006"]["status"] == "offline_auxiliary_not_active",
        "MODEL-006 scope changed",
    )

    checksum_rows = _read_checksums(package / "SHA256SUMS")
    package_files = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    checks.require(set(checksum_rows) == package_files, "SHA256SUMS coverage changed")
    for relative, expected in checksum_rows.items():
        checks.require(
            _sha256_file(package / relative) == expected,
            f"package checksum changed: {relative}",
        )

    source_groups = {
        **config.payload["online_artifacts"],
        **config.payload["offline_artifacts"],
    }
    for name, entry in entries.items():
        source = source_groups[name]
        source_model = config.repository_root / source["model_path"]
        source_manifest = config.repository_root / source["manifest_path"]
        packaged_model = package / entry["model_path"]
        packaged_manifest = package / entry["manifest_path"]
        checks.require(
            _sha256_file(source_model) == source["model_sha256"],
            f"source model changed: {name}",
        )
        checks.require(
            _sha256_file(source_manifest) == source["manifest_sha256"],
            f"source manifest changed: {name}",
        )
        checks.require(
            _sha256_file(packaged_model) == source["model_sha256"],
            f"packaged model differs from source: {name}",
        )
        checks.require(
            _sha256_file(packaged_manifest) == source["manifest_sha256"],
            f"packaged manifest differs from source: {name}",
        )

    for name, source in config.payload["retained_artifacts"].items():
        checks.require(
            _sha256_file(config.repository_root / source["model_path"])
            == source["model_sha256"],
            f"retained model changed: {name}",
        )
        checks.require(
            _sha256_file(config.repository_root / source["manifest_path"])
            == source["manifest_sha256"],
            f"retained manifest changed: {name}",
        )

    if load_models:
        import joblib

        for name in ONLINE_NAMES:
            model = joblib.load(package / entries[name]["model_path"])
            checks.require(model is not None, f"online model failed to load: {name}")
        checks.require(
            "model006" not in ONLINE_NAMES,
            "MODEL-006 was included in online model loading",
        )

    summary = _json(report / "package_summary.json")
    checks.require(summary["entry_count"] == 8, "summary entry count changed")
    checks.require(summary["online_entry_count"] == 7, "summary online count changed")
    checks.require(summary["offline_entry_count"] == 1, "summary offline count changed")
    checks.require(
        summary["manifest_sha256"] == _sha256_file(package / "manifest.json"),
        "summary manifest hash changed",
    )
    checks.require(
        summary["checksums_sha256"] == _sha256_file(package / "SHA256SUMS"),
        "summary checksum-list hash changed",
    )
    _validate_report_artifacts(report, checks)
    return {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "selected_default": selection["selected_default"],
        "manifest_sha256": _sha256_file(package / "manifest.json"),
        "checksums_sha256": _sha256_file(package / "SHA256SUMS"),
        "package_payload_sha256": summary["package_payload_sha256"],
        "models_loaded": load_models,
    }


def _validate_report_artifacts(report: Path, checks: Checks) -> None:
    payload = _json(report / "artifacts.json")
    checks.require(isinstance(payload.get("artifacts"), list), "artifact list missing")
    for row in payload["artifacts"]:
        path = report / row["path"]
        checks.require(path.is_file(), f"report artifact missing: {row['path']}")
        checks.require(path.stat().st_size == row["bytes"], f"size changed: {row['path']}")
        checks.require(_sha256_file(path) == row["sha256"], f"hash changed: {row['path']}")


def _read_checksums(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        if relative in rows:
            raise ValidationFailure(f"duplicate checksum path: {relative}")
        rows[relative] = expected
    return rows


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationFailure(f"JSON object expected: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
