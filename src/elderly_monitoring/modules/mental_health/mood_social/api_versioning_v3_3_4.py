"""API-003 no-promotion package and backend-version audit."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .package_selection import load_active_package_selection
from .pipeline import MoodSocialPipeline


TASK_ID = "API-003"
RUN_ID = "MH-20260804-024"


class ApiV334AuditError(RuntimeError):
    """Raised when the retained API version path is inconsistent."""


def audit_api003(
    config_path: str | Path,
    *,
    repository_root: str | Path | None = None,
    command: Sequence[str] = (),
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ApiV334AuditError("API-003 config is unreadable") from exc
    if (
        not isinstance(config, Mapping)
        or config.get("task_id") != TASK_ID
        or config.get("run_id") != RUN_ID
    ):
        raise ApiV334AuditError("API-003 config identity changed")
    root = Path(repository_root or path.parents[2]).resolve()
    inputs = config["input"]
    for stem in (
        "art002_decision",
        "package_selection",
        "pipeline",
        "backend_service",
        "backend_migration",
    ):
        source = _resolve(root, inputs[f"{stem}_path"])
        if _sha256_file(source) != str(inputs[f"{stem}_sha256"]):
            raise ApiV334AuditError(f"API-003 source hash failed: {stem}")
    selection = load_active_package_selection(repository_root=root)
    pipeline = MoodSocialPipeline.from_package()
    expected = config["expected"]
    if selection.package_run_id != expected["active_package_run_id"]:
        raise ApiV334AuditError("active package run changed")
    if selection.model_version != expected["model_version"]:
        raise ApiV334AuditError("active model version changed")
    if pipeline.package.manifest.get("run_id") != selection.package_run_id:
        raise ApiV334AuditError("pipeline package disagrees with ART-002 decision")
    service_text = _resolve(root, inputs["backend_service_path"]).read_text(
        encoding="utf-8"
    )
    migration_text = _resolve(root, inputs["backend_migration_path"]).read_text(
        encoding="utf-8"
    )
    if f'CURRENT_MODEL_VERSION = "{selection.model_version}"' not in service_text:
        raise ApiV334AuditError("backend current model version changed")
    if (
        'MOOD_SOCIAL_UNIQUE_COLUMNS = ("elder_id", "target_date", "model_version")'
        not in migration_text
    ):
        raise ApiV334AuditError("backend model-version idempotency key changed")
    forbidden = ("lightgbm", "catboost", "build_fusion_features", "joblib.load")
    if any(token in service_text.lower() for token in forbidden):
        raise ApiV334AuditError("algorithm implementation entered backend service")
    result = {
        "version": "mood-social-api003-audit-v1",
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "mode": "no_promotion_keep_art001",
        "promotion_passed": False,
        "active_package_run_id": selection.package_run_id,
        "active_package_directory": str(selection.package_directory),
        "model_version": selection.model_version,
        "algorithm_loader_uses_art002_decision": True,
        "backend_version_idempotency_preserved": True,
        "backend_algorithm_implementation": False,
        "raw_probability_contract_preserved": bool(
            expected["original_probability_preserved"]
        ),
        "no_evidence_http_status": int(expected["no_evidence_http_status"]),
        "unavailable_package_http_status": int(
            expected["unavailable_package_http_status"]
        ),
        "new_package_connected": False,
    }
    _publish(config, root, result, command)
    return {"status": "pass", **result}


def _publish(
    config: Mapping[str, Any],
    root: Path,
    result: Mapping[str, Any],
    command: Sequence[str],
) -> None:
    report = _resolve(root, config["output"]["report_directory"])
    manifest = _resolve(root, config["output"]["manifest_path"])
    report.mkdir(parents=True, exist_ok=True)
    _write_json(report / "api_version_audit.json", result)
    _write_json(report / "config.json", config)
    _write_json(
        report / "run.json",
        {
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "command": list(command),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    (report / "model_card.md").write_text(
        "# API-003\n\nThe V3.3.4 candidate did not pass promotion. The algorithm loader therefore resolves the audited ART-001 fallback, while the backend retains model-version-separated storage.\n",
        encoding="utf-8",
        newline="\n",
    )
    report_core = _report_core_sha256(report)
    _write_json(manifest, {**result, "report_core_sha256": report_core})
    artifacts = [
        {"path": item.name, "bytes": item.stat().st_size, "sha256": _sha256_file(item)}
        for item in sorted(
            report.iterdir(), key=lambda value: value.name.encode("utf-8")
        )
        if item.is_file() and item.name != "artifacts.json"
    ]
    _write_json(
        report / "artifacts.json",
        {
            "version": "mood-social-api003-artifacts-v1",
            "report_core_sha256": report_core,
            "artifacts": artifacts,
        },
    )


def _report_core_sha256(report: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        (
            value
            for value in report.iterdir()
            if value.is_file() and value.name not in {"run.json", "artifacts.json"}
        ),
        key=lambda value: value.name.encode("utf-8"),
    ):
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


__all__ = ["RUN_ID", "TASK_ID", "ApiV334AuditError", "audit_api003"]
