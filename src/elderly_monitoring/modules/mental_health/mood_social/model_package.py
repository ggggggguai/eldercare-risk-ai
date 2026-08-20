"""Build and verify the versioned ART-001 mood-social model package."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import joblib
import yaml


TASK_ID = "ART-001"
RUN_ID = "MH-20260802-013"
PACKAGE_VERSION = "mood-social-model-package-v3.3.3-r1"
MANIFEST_VERSION = "mood-social-unified-package-manifest-v1"
ONLINE_NAMES = (
    "activity_expert",
    "sleep_expert",
    "activity_sleep_joint_expert",
    "physiology_expert",
    "social_context_expert",
    "personal_trend",
    "selected_fusion",
)
OFFLINE_NAMES = ("model006",)
RUNTIME_EXPERT_NAMES = (
    "activity",
    "sleep",
    "joint",
    "physiology",
    "social_context",
)
PACKAGE_MODEL_PATHS = {
    "activity_expert": "experts/activity_expert.joblib",
    "sleep_expert": "experts/sleep_expert.joblib",
    "activity_sleep_joint_expert": "experts/activity_sleep_joint_expert.joblib",
    "physiology_expert": "experts/physiology_expert.joblib",
    "social_context_expert": "experts/social_context_expert.joblib",
    "personal_trend": "personal_trend/personal_trend_expert.joblib",
    "selected_fusion": "fusion/mood_fusion.joblib",
    "model006": "offline/offline_auxiliary_models.joblib",
}


class ModelPackageError(RuntimeError):
    """Raised when a package or protected upstream artifact is invalid."""


@dataclass(frozen=True)
class ModelPackageConfig:
    repository_root: Path
    payload: Mapping[str, Any]
    config_path: Path

    @property
    def package_directory(self) -> Path:
        return _resolve(self.repository_root, self.payload["output"]["package_directory"])

    @property
    def report_directory(self) -> Path:
        return _resolve(self.repository_root, self.payload["output"]["report_directory"])


@dataclass(frozen=True)
class MoodSocialModelPackage:
    package_directory: Path
    manifest: Mapping[str, Any]
    online_models: Mapping[str, Any]
    offline_models: Mapping[str, Any]

    @property
    def fusion(self) -> Any:
        return self.online_models["selected_fusion"]


def load_model_package_config(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> ModelPackageConfig:
    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ModelPackageError("ART-001 config is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ModelPackageError("ART-001 config must be a mapping")
    if payload.get("task_id") != TASK_ID or payload.get("run_id") != RUN_ID:
        raise ModelPackageError("ART-001 task identity changed")
    if payload.get("package_version") != PACKAGE_VERSION:
        raise ModelPackageError("ART-001 package version changed")
    if tuple(payload.get("online_artifacts", {})) != ONLINE_NAMES:
        raise ModelPackageError("ART-001 online artifact order changed")
    if tuple(payload.get("offline_artifacts", {})) != OFFLINE_NAMES:
        raise ModelPackageError("ART-001 offline artifact order changed")
    confidence = payload.get("runtime_confidence", {})
    if (
        confidence.get("method")
        != "full_strict_oof_auroc_excess_times_feature_coverage"
        or tuple(confidence.get("expert_reliability", {})) != RUNTIME_EXPERT_NAMES
        or confidence.get("public_dataset_id_required_at_inference") is not False
    ):
        raise ModelPackageError("ART-001 runtime confidence policy changed")
    reliability = confidence["expert_reliability"]
    if any(
        isinstance(reliability[name], bool)
        or not 0.0 <= float(reliability[name]) <= 1.0
        for name in RUNTIME_EXPERT_NAMES
    ):
        raise ModelPackageError("ART-001 runtime expert reliability is invalid")
    boundaries = payload.get("boundaries", {})
    if any(
        boundaries.get(key) is not False
        for key in (
            "backend_code_changed",
            "model006_online",
            "overwrite_upstream",
            "select_production_threshold",
            "change_attention_levels",
        )
    ):
        raise ModelPackageError("ART-001 boundary changed")
    root = Path(repository_root or config_path.parents[2]).resolve()
    return ModelPackageConfig(root, payload, config_path)


def build_model_package(
    config: ModelPackageConfig,
    *,
    overwrite: bool = False,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    package = config.package_directory
    report = config.report_directory
    if not overwrite and (package.exists() or report.exists()):
        raise ModelPackageError("ART-001 output already exists; use --overwrite")
    if package.exists():
        shutil.rmtree(package)
    if report.exists():
        shutil.rmtree(report)
    protection = _validate_upstream(config)
    runtime_confidence = _runtime_confidence_payload(config)
    selection = _selection_payload(config)
    if selection["selected_default"] != "opt_fusion_002_deployment_candidate":
        raise ModelPackageError("ART-001 selection does not match frozen promotion result")
    package.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    for scope, names in (("online", ONLINE_NAMES), ("offline", OFFLINE_NAMES)):
        source_group = config.payload[f"{scope}_artifacts"]
        for name in names:
            source = source_group[name]
            model_source = _resolve(config.repository_root, source["model_path"])
            manifest_source = _resolve(config.repository_root, source["manifest_path"])
            model_relative = Path(PACKAGE_MODEL_PATHS[name])
            manifest_relative = model_relative.with_name(model_relative.stem + "_manifest.json")
            model_destination = package / model_relative
            manifest_destination = package / manifest_relative
            model_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(model_source, model_destination)
            shutil.copyfile(manifest_source, manifest_destination)
            status = "online" if scope == "online" else str(source["status"])
            entries[name] = {
                "scope": scope,
                "status": status,
                "model_path": model_relative.as_posix(),
                "model_sha256": _sha256_file(model_destination),
                "manifest_path": manifest_relative.as_posix(),
                "manifest_sha256": _sha256_file(manifest_destination),
                "source_model_path": str(source["model_path"]),
                "source_manifest_path": str(source["manifest_path"]),
            }
    _write_json(package / "selection.json", selection)
    _write_json(package / "runtime_confidence.json", runtime_confidence)
    manifest = {
        "version": MANIFEST_VERSION,
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "package_version": PACKAGE_VERSION,
        "default_fusion": "selected_fusion",
        "default_fusion_source": selection["selected_default"],
        "entries": entries,
        "retained_artifacts": config.payload["retained_artifacts"],
        "upstream_protection": protection,
        "runtime_confidence": runtime_confidence,
        "model006_online": False,
        "model006_http_output": False,
        "production_threshold_selected": False,
        "attention_levels_changed": False,
    }
    _write_json(package / "manifest.json", manifest)
    checksum_paths = sorted(
        (
            path
            for path in package.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        ),
        key=lambda path: path.relative_to(package).as_posix().encode("utf-8"),
    )
    checksum_text = "".join(
        f"{_sha256_file(path)}  {path.relative_to(package).as_posix()}\n"
        for path in checksum_paths
    )
    (package / "SHA256SUMS").write_text(
        checksum_text,
        encoding="ascii",
        newline="\n",
    )
    package_summary = {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "package_version": PACKAGE_VERSION,
        "package_directory": str(package),
        "selected_default": selection["selected_default"],
        "entry_count": len(entries),
        "online_entry_count": len(ONLINE_NAMES),
        "offline_entry_count": len(OFFLINE_NAMES),
        "manifest_sha256": _sha256_file(package / "manifest.json"),
        "checksums_sha256": _sha256_file(package / "SHA256SUMS"),
        "package_payload_sha256": _package_payload_sha256(package),
    }
    _write_json(report / "selection.json", selection)
    _write_json(report / "upstream_protection.json", protection)
    _write_json(report / "package_summary.json", package_summary)
    _write_json(
        report / "run.json",
        {
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": list(command or ()),
        },
    )
    artifact_rows = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(report.iterdir(), key=lambda item: item.name.encode("utf-8"))
        if path.is_file() and path.name != "artifacts.json"
    ]
    _write_json(
        report / "artifacts.json",
        {"version": "mood-social-art001-artifacts-v1", "artifacts": artifact_rows},
    )
    return {"status": "pass", **package_summary}


def load_mood_social_model_package(
    package_directory: str | Path,
    *,
    load_offline: bool = False,
) -> MoodSocialModelPackage:
    package = Path(package_directory).resolve()
    manifest_path = package / "manifest.json"
    checksums_path = package / "SHA256SUMS"
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise ModelPackageError("mood-social package manifest or SHA256SUMS is missing")
    _verify_checksums(package, checksums_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != MANIFEST_VERSION or manifest.get("task_id") != TASK_ID:
        raise ModelPackageError("mood-social package manifest identity changed")
    if manifest.get("default_fusion") != "selected_fusion":
        raise ModelPackageError("mood-social package default fusion changed")
    if manifest.get("model006_online") is not False:
        raise ModelPackageError("MODEL-006 cannot be enabled for online inference")
    entries = manifest.get("entries")
    if not isinstance(entries, Mapping) or set(entries) != set((*ONLINE_NAMES, *OFFLINE_NAMES)):
        raise ModelPackageError("mood-social package entry set changed")
    online_models = {
        name: _load_entry(package, name, entries[name]) for name in ONLINE_NAMES
    }
    offline_models: dict[str, Any] = {}
    if load_offline:
        offline_models = {
            name: _load_entry(package, name, entries[name]) for name in OFFLINE_NAMES
        }
    return MoodSocialModelPackage(package, manifest, online_models, offline_models)


def _load_entry(package: Path, name: str, entry: Mapping[str, Any]) -> Any:
    model_path = package / str(entry["model_path"])
    manifest_path = package / str(entry["manifest_path"])
    if _sha256_file(model_path) != str(entry["model_sha256"]):
        raise ModelPackageError(f"package model hash failed: {name}")
    if _sha256_file(manifest_path) != str(entry["manifest_sha256"]):
        raise ModelPackageError(f"package manifest hash failed: {name}")
    return joblib.load(model_path)


def _validate_upstream(config: ModelPackageConfig) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for group_name in ("online_artifacts", "offline_artifacts", "retained_artifacts"):
        group = config.payload[group_name]
        for name, entry in group.items():
            model_path = _resolve(config.repository_root, entry["model_path"])
            manifest_path = _resolve(config.repository_root, entry["manifest_path"])
            expected_model = str(entry["model_sha256"])
            expected_manifest = str(entry["manifest_sha256"])
            if _sha256_file(model_path) != expected_model:
                raise ModelPackageError(f"upstream model hash changed: {name}")
            if _sha256_file(manifest_path) != expected_manifest:
                raise ModelPackageError(f"upstream manifest hash changed: {name}")
            rows[name] = {
                "model_sha256": expected_model,
                "manifest_sha256": expected_manifest,
                "source_group": group_name,
            }
    confidence = config.payload["runtime_confidence"]
    confidence_source = _resolve(config.repository_root, confidence["source_path"])
    expected_confidence_source = str(confidence["source_sha256"])
    if _sha256_file(confidence_source) != expected_confidence_source:
        raise ModelPackageError("runtime confidence OOF source hash changed")
    rows["runtime_confidence_source"] = {
        "sha256": expected_confidence_source,
        "source_group": "runtime_confidence",
    }
    return rows


def _runtime_confidence_payload(config: ModelPackageConfig) -> dict[str, Any]:
    confidence = config.payload["runtime_confidence"]
    return {
        "version": "mood-social-runtime-confidence-v1",
        "method": confidence["method"],
        "formula": confidence["formula"],
        "source_path": confidence["source_path"],
        "source_sha256": confidence["source_sha256"],
        "expert_reliability": {
            name: float(confidence["expert_reliability"][name])
            for name in RUNTIME_EXPERT_NAMES
        },
        "public_dataset_id_required_at_inference": False,
    }


def _selection_payload(config: ModelPackageConfig) -> dict[str, Any]:
    selection = config.payload["selection"]
    promotion_path = _resolve(config.repository_root, selection["promotion_path"])
    report_directory = promotion_path.parent
    expected_report_core = str(selection["opt_fusion_002_report_core_sha256"])
    if _report_core_sha256(report_directory) != expected_report_core:
        raise ModelPackageError("OPT-FUSION-002 report core hash changed")
    candidate_manifest_path = _resolve(
        config.repository_root,
        selection["opt_fusion_002_manifest_path"],
    )
    if _sha256_file(candidate_manifest_path) != str(
        selection["opt_fusion_002_manifest_sha256"]
    ):
        raise ModelPackageError("OPT-FUSION-002 candidate manifest hash changed")
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if candidate_manifest.get("report_core_sha256") != expected_report_core:
        raise ModelPackageError("OPT-FUSION-002 manifest report binding changed")
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    promotion_passed = bool(promotion.get("promotion_passed"))
    selected = selection["pass_choice"] if promotion_passed else selection["fail_choice"]
    return {
        "version": "mood-social-art001-selection-v1",
        "promotion_run_id": "MH-20260802-012",
        "promotion_passed": promotion_passed,
        "promotion_checks": promotion.get("checks", {}),
        "selected_default": selected,
        "selection_rule": "OPT-FUSION-002 frozen all-conditions promotion gate",
        "fusion_002_retained": True,
        "opt_fusion_001_retained_offline": True,
        "opt_fusion_001_production_eligible": False,
        "production_threshold_selected": False,
    }


def _verify_checksums(package: Path, checksum_path: Path) -> None:
    lines = checksum_path.read_text(encoding="ascii").splitlines()
    expected_paths = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    listed_paths: set[str] = set()
    for line in lines:
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ModelPackageError("SHA256SUMS contains a malformed line") from exc
        if relative in listed_paths:
            raise ModelPackageError(f"duplicate package checksum: {relative}")
        listed_paths.add(relative)
        path = (package / relative).resolve()
        try:
            path.relative_to(package)
        except ValueError as exc:
            raise ModelPackageError("SHA256SUMS path escapes package") from exc
        if not path.is_file() or _sha256_file(path) != expected:
            raise ModelPackageError(f"package checksum failed: {relative}")
    if listed_paths != expected_paths:
        raise ModelPackageError("SHA256SUMS coverage does not match package files")


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
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
