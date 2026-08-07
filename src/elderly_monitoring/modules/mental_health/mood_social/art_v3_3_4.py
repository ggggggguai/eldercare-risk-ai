"""ART-002 package decision and fallback audit for V3.3.4."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .model_package import load_mood_social_model_package


TASK_ID = "ART-002"
RUN_ID = "MH-20260804-023"


class ArtV334Error(RuntimeError):
    """Raised when the ART-002 fallback decision is not auditable."""


def load_art_config(
    path: str | Path, *, repository_root: str | Path | None = None
) -> tuple[Path, Mapping[str, Any], Path]:
    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ArtV334Error("ART-002 config is unreadable") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("task_id") != TASK_ID
        or payload.get("run_id") != RUN_ID
    ):
        raise ArtV334Error("ART-002 task identity changed")
    root = Path(repository_root or config_path.parents[2]).resolve()
    return config_path, payload, root


def audit_art002(
    config_path: str | Path,
    *,
    repository_root: str | Path | None = None,
    command: Sequence[str] = (),
) -> dict[str, Any]:
    config_file, config, root = load_art_config(
        config_path, repository_root=repository_root
    )
    inputs = config["input"]
    promotion_path = _resolve(root, inputs["promotion_path"])
    promotion = _read_hashed_json(promotion_path, str(inputs["promotion_sha256"]))
    candidate_model = _resolve(root, inputs["candidate_model_path"])
    candidate_manifest = _resolve(root, inputs["candidate_manifest_path"])
    workpoint_manifest = _resolve(root, inputs["workpoint_manifest_path"])
    for path, key in (
        (candidate_model, "candidate_model_sha256"),
        (candidate_manifest, "candidate_manifest_sha256"),
        (workpoint_manifest, "workpoint_manifest_sha256"),
    ):
        if _sha256_file(path) != str(inputs[key]):
            raise ArtV334Error(f"ART-002 upstream hash failed: {path.name}")
    fallback = _resolve(root, inputs["fallback_package_directory"])
    fallback_manifest = fallback / "manifest.json"
    checksums = fallback / "SHA256SUMS"
    if _sha256_file(fallback_manifest) != str(inputs["fallback_manifest_sha256"]):
        raise ArtV334Error("ART-001 manifest hash changed")
    if _sha256_file(checksums) != str(inputs["fallback_checksums_sha256"]):
        raise ArtV334Error("ART-001 checksum hash changed")
    if _package_payload_sha256(fallback) != str(inputs["fallback_payload_sha256"]):
        raise ArtV334Error("ART-001 payload hash changed")
    package = load_mood_social_model_package(fallback)
    if len(package.online_models) != 7 or package.offline_models:
        raise ArtV334Error("ART-001 online/offline boundary changed")
    if package.manifest.get("model006_online") is not False:
        raise ArtV334Error("MODEL-006 entered online package")
    promotion_passed = bool(promotion.get("promotion_passed"))
    new_package = _resolve(root, config["output"]["new_package_directory"])
    if not promotion_passed and new_package.exists():
        raise ArtV334Error("a new package exists despite failed promotion")
    decision = {
        "version": "mood-social-art002-decision-v1",
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "promotion_passed": promotion_passed,
        "new_online_package_created": promotion_passed,
        "active_package_directory": str(inputs["fallback_package_directory"]),
        "active_package_run_id": "MH-20260802-013",
        "fallback_preserved": True,
        "model006_online": False,
        "production_threshold_selected": False,
        "candidate_not_promoted_reason": None
        if promotion_passed
        else "OPT-FUSION-003 failed frozen five-condition promotion gate",
        "upstream": {
            "promotion_sha256": str(inputs["promotion_sha256"]),
            "candidate_model_sha256": str(inputs["candidate_model_sha256"]),
            "candidate_manifest_sha256": str(inputs["candidate_manifest_sha256"]),
            "workpoint_manifest_sha256": str(inputs["workpoint_manifest_sha256"]),
            "fallback_manifest_sha256": str(inputs["fallback_manifest_sha256"]),
            "fallback_checksums_sha256": str(inputs["fallback_checksums_sha256"]),
            "fallback_payload_sha256": str(inputs["fallback_payload_sha256"]),
        },
    }
    _publish(config, root, decision, command)
    return {
        "status": "pass",
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "promotion_passed": promotion_passed,
        "active_package_run_id": "MH-20260802-013",
    }


def _publish(
    config: Mapping[str, Any],
    root: Path,
    decision: Mapping[str, Any],
    command: Sequence[str],
) -> None:
    report = _resolve(root, config["output"]["report_directory"])
    manifest_path = _resolve(root, config["output"]["decision_manifest_path"])
    report.mkdir(parents=True, exist_ok=True)
    _write_json(report / "decision.json", decision)
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
        "# ART-002 V3.3.4 Package Decision\n\n"
        "This audit creates a new package only when the frozen promotion gate passes.\n"
        "The current result retains ART-001 as the online fallback.\n",
        encoding="utf-8",
        newline="\n",
    )
    report_core = _report_core_sha256(report)
    decision_with_hash = {**decision, "report_core_sha256": report_core}
    _write_json(manifest_path, decision_with_hash)
    artifact_rows = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in sorted(report.iterdir(), key=lambda item: item.name.encode("utf-8"))
        if path.is_file() and path.name != "artifacts.json"
    ]
    _write_json(
        report / "artifacts.json",
        {
            "version": "mood-social-art002-artifacts-v1",
            "report_core_sha256": report_core,
            "artifacts": artifact_rows,
        },
    )


def _read_hashed_json(path: Path, expected: str) -> Mapping[str, Any]:
    if _sha256_file(path) != expected:
        raise ArtV334Error(f"hash failed: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ArtV334Error(f"JSON object expected: {path.name}")
    return value


def _package_payload_sha256(package: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (item for item in package.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(package).as_posix().encode("utf-8"),
    ):
        digest.update(path.relative_to(package).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _report_core_sha256(report: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (
            item
            for item in report.iterdir()
            if item.is_file() and item.name not in {"run.json", "artifacts.json"}
        ),
        key=lambda item: item.name.encode("utf-8"),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
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


__all__ = ["RUN_ID", "TASK_ID", "ArtV334Error", "audit_art002", "load_art_config"]
