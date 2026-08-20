"""Fail-closed R4-009 release decision and active fallback verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from elderly_monitoring.modules.mental_health.mood_social.model_package import (
    load_mood_social_model_package,
)
from elderly_monitoring.modules.mental_health.mood_social.package_selection import (
    EXPECTED_ACTIVE_RUN_ID,
    load_active_package_selection,
)


DEFAULT_OUTPUT = Path(
    "reports/mental_health/mood_social/v3.3.3-r4/OPT-V333-R4-009-release"
)
LOCKED_MANIFEST = Path(
    "reports/mental_health/mood_social/v3.3.3-r4/OPT-V333-R4-008-evaluation/"
    "locked/artifact_manifest.json"
)
LODO_MANIFEST = Path(
    "reports/mental_health/mood_social/v3.3.3-r4/OPT-V333-R4-008-evaluation/"
    "lodo/artifact_manifest.json"
)
POSTLOCK_MANIFEST = Path(
    "reports/mental_health/mood_social/v3.3.3-r4/"
    "OPT-V333-R4-005-006-postlocked-audit/artifact_manifest.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def make_r4_release_decision(repository_root: Path, output: Path | None = None) -> dict[str, Any]:
    root = repository_root.resolve()
    destination = (output or root / DEFAULT_OUTPUT).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite r4 release decision: {destination}")

    locked_manifest_path = root / LOCKED_MANIFEST
    lodo_manifest_path = root / LODO_MANIFEST
    postlock_manifest_path = root / POSTLOCK_MANIFEST
    locked = _json(locked_manifest_path)
    lodo = _json(lodo_manifest_path)
    selection = load_active_package_selection(repository_root=root)
    package = load_mood_social_model_package(selection.package_directory)
    if selection.package_run_id != EXPECTED_ACTIVE_RUN_ID:
        raise ValueError("r4 release fallback package identity drifted")
    if locked.get("status") != "pass" or lodo.get("status") not in {
        "pass",
        "complete_with_route_local_warnings",
    }:
        raise ValueError("r4 locked or LODO evidence is incomplete")

    decision = {
        "version": "mood-social-v3.3.3-r4-release-decision-v1",
        "task_id": "OPT-V333-R4-009",
        "protocol_version": "mood-social-v3.3.3-r4",
        "evidence_level": "adaptive-development/reused-benchmark_not_independent_blind",
        "offline_competition_candidate": {
            "status": "pass",
            "track": "multisource_deployable",
            "historical_phq_used": False,
            "claim_boundary": "offline_current_state_only",
        },
        "single_source_competition_candidate": {
            "status": "pass_research_only",
            "track": "psyche_d_single_source_research",
            "runtime_generator_parity": False,
            "online_eligible": False,
        },
        "shadow_candidate": {
            "status": "not_created_not_promoted",
            "reason": (
                "formal offline metrics passed, but a frozen full-training dual-head package, "
                "distribution/runtime generator parity, train-only workpoints and end-to-end API "
                "regression were not available before the evaluation lock"
            ),
        },
        "active_online": {
            "package_run_id": selection.package_run_id,
            "package_directory": selection.package_directory.relative_to(root).as_posix(),
            "manifest_run_id": package.manifest.get("run_id"),
            "fallback_verified": True,
            "new_package_connected": False,
            "backend_algorithm_implementation": False,
        },
        "route_policy": {
            "nhanes_phq_ge10_joint": "block_r4_route_and_keep_existing_package",
            "reason": "LODO delta AUROC below -0.05",
            "all_other_lodo_tables": "retain_offline_candidate_evidence",
            "no_evidence": "abstain_or_existing_versioned_fallback",
        },
        "workpoint_policy": {
            "outer_oof_curve_summaries": "descriptive_only",
            "train_only_inner_oof_workpoints_frozen": False,
            "shadow_blocked_until_train_only_workpoints": True,
        },
        "upstream": {
            "locked_manifest_sha256": _sha256(locked_manifest_path),
            "lodo_manifest_sha256": _sha256(lodo_manifest_path),
            "postlocked_audit_manifest_sha256": _sha256(postlock_manifest_path),
            "active_package_manifest_sha256": _sha256(selection.package_directory / "manifest.json"),
            "active_package_checksums_sha256": _sha256(selection.package_directory / "SHA256SUMS"),
        },
        "git_policy": "commit_algorithm_then_root; do_not_push",
    }
    decision_path = destination / "r4_release_decision.json"
    decision_path.write_text(
        json.dumps(decision, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    try:
        decision_reference = decision_path.relative_to(root).as_posix()
    except ValueError:
        decision_reference = str(decision_path)
    manifest = {
        "status": "pass_with_existing_online_fallback",
        "decision": decision_reference,
        "decision_sha256": _sha256(decision_path),
    }
    manifest_path = destination / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest


__all__ = ["make_r4_release_decision"]
