"""Delayed one-time reused-cohort confirmation seal for r6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from elderly_monitoring.modules.mental_health.mood_social.r6.contract import (
    DEFAULT_DATA_RELATIVE,
    R6_CONFIRMATION_SEED,
    R6_PROTOCOL_VERSION,
    sha256_file,
    tree_hash,
    write_json,
)


LOCKED_IMPLEMENTATION_FILES = tuple(
    Path("src/elderly_monitoring/modules/mental_health/mood_social/r6") / name
    for name in ("selection.py", "competition.py", "objectives.py", "finalization.py", "confirmation.py")
)


DEFAULT_CONFIRMATION_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r6/OPT-V333-R6-006-confirmation"
)


def confirmation_state(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = root / DEFAULT_CONFIRMATION_RELATIVE
    return {
        "confirmation_seed": R6_CONFIRMATION_SEED,
        "recipe_locked": (output / "candidate_recipe_lock.json").is_file(),
        "opened": (output / "OPENED.json").is_file(),
        "sealed": (output / "SEALED.json").is_file(),
        "candidate_prediction_files": [
            path.relative_to(root).as_posix()
            for path in sorted(output.glob(f"**/seed-{R6_CONFIRMATION_SEED}.parquet"))
        ],
    }


def lock_confirmation_recipe(*, repository_root: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = root / DEFAULT_CONFIRMATION_RELATIVE
    if confirmation_state(root)["opened"]:
        raise PermissionError("r6 confirmation is already opened and cannot be relocked")
    path = output / "candidate_recipe_lock.json"
    required = {
        "features", "candidates", "objective_structure", "training_weights", "fusion",
        "calibration", "workpoints", "development_evidence_sha256", "implementation_code_tree_sha256",
    }
    missing = sorted(required.difference(recipe))
    if missing:
        raise ValueError(f"r6 confirmation recipe is incomplete: {missing}")
    payload = {
        "protocol_version": R6_PROTOCOL_VERSION,
        "confirmation_seed": R6_CONFIRMATION_SEED,
        "confirmation_open_once": True,
        "evidence_level": "sealed reused-cohort confirmation",
        "recipe": recipe,
    }
    write_json(path, payload)
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def locked_confirmation_recipe(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path = root / DEFAULT_CONFIRMATION_RELATIVE / "candidate_recipe_lock.json"
    if not path.is_file():
        raise FileNotFoundError("r6 confirmation recipe is not locked")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("confirmation_seed") != R6_CONFIRMATION_SEED:
        raise ValueError("r6 confirmation lock seed drifted")
    return payload


def open_confirmation_once(*, repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = root / DEFAULT_CONFIRMATION_RELATIVE
    lock_path = output / "candidate_recipe_lock.json"
    opened_path = output / "OPENED.json"
    if not lock_path.is_file():
        raise PermissionError("r6 confirmation recipe must be locked before opening")
    if opened_path.exists():
        raise PermissionError("r6 confirmation was already opened; re-open is forbidden")
    protocol = root / DEFAULT_DATA_RELATIVE / "r6_protocol_manifest.json"
    baseline = root / (
        "reports/mental_health/mood_social/v3.3.3-r6/OPT-V333-R6-000/"
        "r5-paired-baseline/development_paired_baseline_summary.json"
    )
    for dependency in (protocol, baseline):
        if not dependency.is_file():
            raise FileNotFoundError(f"r6 confirmation prerequisite is missing: {dependency}")
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_code_sha = lock_payload["recipe"]["implementation_code_tree_sha256"]
    observed_code_sha, _ = tree_hash(root, LOCKED_IMPLEMENTATION_FILES)
    if observed_code_sha != expected_code_sha:
        raise PermissionError("r6 confirmation implementation code tree drifted after recipe lock")
    payload = {
        "protocol_version": R6_PROTOCOL_VERSION,
        "confirmation_seed": R6_CONFIRMATION_SEED,
        "opened": True,
        "opened_after_recipe_lock": True,
        "reopen_allowed": False,
        "recipe_sha256": sha256_file(lock_path),
        "protocol_manifest_sha256": sha256_file(protocol),
        "paired_baseline_summary_sha256": sha256_file(baseline),
        "implementation_code_tree_sha256": observed_code_sha,
    }
    write_json(opened_path, payload)
    return payload


def seal_confirmation(*, repository_root: Path, artifact_paths: list[Path]) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = root / DEFAULT_CONFIRMATION_RELATIVE
    opened = output / "OPENED.json"
    sealed = output / "SEALED.json"
    if not opened.is_file():
        raise PermissionError("r6 confirmation must be opened before sealing")
    if sealed.exists():
        raise PermissionError("r6 confirmation is already sealed")
    artifacts = []
    digest = hashlib.sha256()
    for path in sorted((Path(value).resolve() for value in artifact_paths), key=str):
        if not path.is_file() or root not in path.parents:
            raise FileNotFoundError(f"invalid r6 confirmation artifact: {path}")
        relative = path.relative_to(root).as_posix()
        value = sha256_file(path)
        artifacts.append({"path": relative, "sha256": value})
        digest.update(relative.encode("utf-8") + b"\0" + value.encode("ascii") + b"\n")
    payload = {
        "protocol_version": R6_PROTOCOL_VERSION,
        "confirmation_seed": R6_CONFIRMATION_SEED,
        "sealed": True,
        "reopen_allowed": False,
        "artifacts": artifacts,
        "artifact_tree_sha256": digest.hexdigest(),
    }
    write_json(sealed, payload)
    return payload


__all__ = [
    "DEFAULT_CONFIRMATION_RELATIVE", "confirmation_state", "lock_confirmation_recipe",
    "locked_confirmation_recipe", "open_confirmation_once", "seal_confirmation",
]
