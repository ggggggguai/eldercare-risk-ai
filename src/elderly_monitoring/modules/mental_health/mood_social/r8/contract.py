"""R8-000 frozen lineage, exposure, time-context and no-data protocol."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from elderly_monitoring.modules.mental_health.mood_social.r7.contract import (
    repository_root,
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.r7.release import (
    R7_PACKAGE_RELATIVE,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.deployment import (
    B0_DEPLOYMENT_MODEL_VERSION,
    B0_EXPECTED_RUN_LOCK_SHA256,
    canonical_manifest_sha256,
)

from .rule_fusion import rule_contract_sha256
from .schemas import B0_PACKAGE_SHA256, B0_RUN_LOCK_SHA256


R8_PROTOCOL_VERSION = "mood-social-v3.3.3-r8"
R8_FROZEN_DOCUMENT_SHA256 = (
    "8951B8CCB8D43921A206CFD14950C0C4BF057BC01B65C2C00D529F392C84B3E2"
)
FROZEN_DOCUMENT_RELATIVE = Path(
    "项目文档/开发协作/情绪与社交关注模块V3.3.3优化开发/"
    "00-V3.3.3-r8优化方案冻结说明.md"
)
DEFAULT_DATA_RELATIVE = Path(
    "data/processed/mental_health/mood_social/v3.3.3-r8/protocol"
)
DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r8/OPT-V333-R8-000"
)
B0_PACKAGE_RELATIVE = Path(
    "models/mental_health/facial_affect/b0_opt_me_008_deployment_v1"
)
B0_RUN_LOCK_RELATIVE = Path(
    "reports/microexpression/opt_me_008/development/DEV-ME8/run_lock.json"
)


def project_root(root: Path) -> Path:
    value = root.parents[1]
    if not (value / "项目文档").is_dir():
        raise FileNotFoundError("cannot locate project documentation root")
    return value


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite frozen R8 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def tree_manifest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        value = sha256_file(path)
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": value})
        digest.update(relative.encode() + b"\0" + value.encode() + b"\n")
    return digest.hexdigest(), rows


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "numpy",
        "pandas",
        "scikit-learn",
        "pyarrow",
        "pydantic",
        "fastapi",
        "lightgbm",
        "catboost",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "network_dataset_download": False,
        "new_dependency_install": False,
    }


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _verify_r7_package(package: Path) -> dict[str, Any]:
    checksums = package / "SHA256SUMS"
    if not checksums.is_file():
        raise FileNotFoundError(checksums)
    listed: set[str] = set()
    for line in checksums.read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", 1)
        path = (package / name).resolve()
        path.relative_to(package.resolve())
        if name in listed or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"R7 package checksum drift: {name}")
        listed.add(name)
    expected_files = {
        path.name
        for path in package.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if listed != expected_files:
        raise ValueError("R7 package checksum coverage incomplete")
    tree_sha, files = tree_manifest(package)
    return {"tree_sha256": tree_sha, "files": files}


def _verify_b0_package(package: Path, run_lock: Path) -> dict[str, Any]:
    manifest_path = package / "manifest.json"
    audit_path = package / "freeze_audit.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if manifest.get("model_version") != B0_DEPLOYMENT_MODEL_VERSION:
        raise ValueError("B0 model version drift")
    if manifest.get("manifest_sha256") != B0_PACKAGE_SHA256:
        raise ValueError("B0 package manifest hash drift")
    if canonical_manifest_sha256(manifest) != B0_PACKAGE_SHA256:
        raise ValueError("B0 manifest self hash drift")
    if manifest.get("lineage", {}).get("run_lock_sha256") != B0_RUN_LOCK_SHA256:
        raise ValueError("B0 run-lock lineage drift")
    if B0_EXPECTED_RUN_LOCK_SHA256 != B0_RUN_LOCK_SHA256:
        raise ValueError("B0 runtime run-lock constant drift")
    if audit.get("status") != "passed" or audit.get("error_count") != 0:
        raise ValueError("B0 freeze audit is not clean")
    if not run_lock.is_file():
        raise FileNotFoundError(run_lock)
    tree_sha, files = tree_manifest(package)
    return {
        "tree_sha256": tree_sha,
        "files": files,
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_self_sha256": manifest["manifest_sha256"],
        "freeze_audit_sha256": sha256_file(audit_path),
        "run_lock_file_sha256": sha256_file(run_lock),
        "run_lock_payload_sha256": B0_RUN_LOCK_SHA256,
    }


def build_r8_protocol_artifacts(
    *,
    repository_root_value: Path | None = None,
    output_directory: Path | None = None,
    report_directory: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    project = project_root(root)
    frozen = project / FROZEN_DOCUMENT_RELATIVE
    frozen_sha = sha256_file(frozen).upper()
    if frozen_sha != R8_FROZEN_DOCUMENT_SHA256:
        raise ValueError(f"R8 frozen document hash drift: {frozen_sha}")

    output = (Path(output_directory) if output_directory else root / DEFAULT_DATA_RELATIVE).resolve()
    report = (Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE).resolve()
    expected = [
        output / "r8_protocol_manifest.json",
        report / "phq_video_data_availability.json",
        report / "data_exposure_ledger.json",
        report / "license_manifest.json",
        report / "facial_time_context_contract.json",
        report / "risk_feature_denylist.json",
        report / "lineage_manifest.json",
        report / "environment.json",
        report / "failed_runs.jsonl",
    ]
    if not overwrite:
        existing = [str(path) for path in expected if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite R8 protocol artifacts: {existing}")

    r7_package = root / R7_PACKAGE_RELATIVE
    b0_package = root / B0_PACKAGE_RELATIVE
    b0_run_lock = root / B0_RUN_LOCK_RELATIVE
    r7 = _verify_r7_package(r7_package)
    b0 = _verify_b0_package(b0_package, b0_run_lock)
    r7_manifest = json.loads((r7_package / "manifest.json").read_text(encoding="utf-8"))

    no_data = {
        "schema_version": "mood_social_r8_phq_video_data_availability_v1",
        "capturing_feature_implemented": True,
        "participant_count": 0,
        "video_count": 0,
        "paired_label_count": 0,
        "training_authorized": False,
        "phq_session_expert_present": False,
        "synthetic_video_may_substitute_real_data": False,
        "future_change_requires_new_protocol": "R8.x/R9",
    }
    exposure = {
        "protocol_version": R8_PROTOCOL_VERSION,
        "existing_r7_population": {
            "rows": 22070,
            "participants": 15327,
            "status": "adaptive-development/reused-benchmark",
        },
        "deprest_cat": {
            "participants": 369,
            "confirmation_already_opened": True,
            "reopen_allowed": False,
            "status": "adaptive-development-only",
        },
        "facial_affect": {
            "casme_ii": "exposed-development estimate",
            "smic": "frozen zero-shot cross-source diagnostic",
            "real_s10_elderly_sessions": 0,
            "paired_current_phq_sessions": 0,
        },
        "real_device_data_available": False,
        "new_subjects_available": False,
        "full_r8_auprc_authorized": False,
        "maximum_release_state": "offline-validated/integration-ready/device-validation-pending",
    }
    license_manifest = {
        "DepreST-CAT": {
            "license": "CC BY-NC-SA academic use",
            "commercial_or_default_release": "blocked_pending_legal_review",
        },
        "CASME_II_and_SMIC": {
            "role": "facial-affect research evidence only",
            "redistribution": "follow dataset-specific terms; raw data excluded from package",
        },
        "raw_media_in_r8_package": False,
    }
    time_contract = {
        "schema_version": "mood_social_r8_facial_time_context_contract_v1",
        "allowed_contexts": ["active_task_bundle", "phq9_questionnaire_session"],
        "contexts_aggregated_together": False,
        "identity": "observation.person_id == request.person_id",
        "time": "observed_at <= inference_cutoff and known_at <= inference_cutoff",
        "lookback_days": 14,
        "deduplication_key": ["observation_context", "session_id", "task_slot", "segment_id"],
        "quality": {"status": "completed", "quality_score_min": 0.65, "confidence_min": 0.60, "multiple_face_conflicts": False},
        "approved_b0": {"model_version": B0_DEPLOYMENT_MODEL_VERSION, "manifest_self_sha256": B0_PACKAGE_SHA256, "run_lock_payload_sha256": B0_RUN_LOCK_SHA256},
        "phq_questionnaire_context_vote_enabled": False,
    }
    denylist = {
        "risk_inputs": [
            "current/target/future PHQ items, total, severity or derived labels",
            "PHQ item 9 answer as a facial or sensor model feature",
            "raw video, video URL, frames, face embeddings or audio transcript",
            "participant/dataset/source/path identifiers",
            "post-cutoff observations or backfilled records unknown at cutoff",
            "model coverage/mask/source identity as learned risk predictors",
        ],
        "facial_semantics": [
            "negative as depression/PHQ probability",
            "confidence as mental-health probability",
            "positive/surprise as protective risk-lowering evidence",
            "no facial detection as normal state",
        ],
        "phq_history": "independent R7 PHQHistory expert only",
        "forecast": "V3.4 only; never fed back into R8 current state",
    }
    lineage = {
        "protocol_version": R8_PROTOCOL_VERSION,
        "frozen_document_sha256": frozen_sha,
        "r7_package": {"path": R7_PACKAGE_RELATIVE.as_posix(), "run_id": r7_manifest.get("run_id"), **r7},
        "r7_rule_contract_sha256": r7_manifest.get("rule_fusion", {}).get("rule_contract_sha256"),
        "b0_package": {"path": B0_PACKAGE_RELATIVE.as_posix(), **b0},
        "r8_rule_contract_sha256": rule_contract_sha256(),
    }
    write_json(report / "phq_video_data_availability.json", no_data, overwrite=overwrite)
    write_json(report / "data_exposure_ledger.json", exposure, overwrite=overwrite)
    write_json(report / "license_manifest.json", license_manifest, overwrite=overwrite)
    write_json(report / "facial_time_context_contract.json", time_contract, overwrite=overwrite)
    write_json(report / "risk_feature_denylist.json", denylist, overwrite=overwrite)
    write_json(report / "lineage_manifest.json", lineage, overwrite=overwrite)
    write_json(report / "environment.json", _environment(), overwrite=overwrite)
    failed_runs = report / "failed_runs.jsonl"
    failed_runs.parent.mkdir(parents=True, exist_ok=True)
    failed_runs.write_text("", encoding="utf-8")

    manifest = {
        "protocol_version": R8_PROTOCOL_VERSION,
        "status": "pass",
        "tasks_completed": ["OPT-V333-R8-000"],
        "frozen_document_sha256": frozen_sha,
        "lineage_manifest_sha256": sha256_file(report / "lineage_manifest.json"),
        "no_phq_video_data_sha256": sha256_file(report / "phq_video_data_availability.json"),
        "time_context_contract_sha256": sha256_file(report / "facial_time_context_contract.json"),
        "denylist_sha256": sha256_file(report / "risk_feature_denylist.json"),
        "environment_sha256": sha256_file(report / "environment.json"),
        "rule_contract_sha256": rule_contract_sha256(),
        "git": _git_state(root),
        "training_performed": False,
        "network_download_performed": False,
    }
    write_json(output / "r8_protocol_manifest.json", manifest, overwrite=overwrite)
    return manifest


__all__ = [
    "B0_PACKAGE_RELATIVE",
    "DEFAULT_DATA_RELATIVE",
    "DEFAULT_REPORT_RELATIVE",
    "FROZEN_DOCUMENT_RELATIVE",
    "R8_FROZEN_DOCUMENT_SHA256",
    "R8_PROTOCOL_VERSION",
    "build_r8_protocol_artifacts",
    "project_root",
    "tree_manifest",
    "write_json",
]
