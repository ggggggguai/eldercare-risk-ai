"""One-time reused-cohort confirmation seal for OPT-V333-R5-006."""

from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path
from typing import Any

from elderly_monitoring.modules.mental_health.mood_social.r5.contract import (
    DEFAULT_DATA_RELATIVE,
    DEFAULT_REPORT_RELATIVE,
    R5_CONFIRMATION_SEED,
    R5_PROTOCOL_VERSION,
    sha256_file,
)


DEFAULT_LOCK_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/OPT-V333-R5-006-confirmation"
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def confirmation_state(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = root / DEFAULT_LOCK_RELATIVE
    opened = output / "OPENED.json"
    recipe = output / "candidate_recipe_lock.json"
    confirmation_outputs = list(output.glob(f"**/seed-{R5_CONFIRMATION_SEED}.parquet"))
    return {
        "confirmation_seed": R5_CONFIRMATION_SEED,
        "recipe_locked": recipe.is_file(),
        "opened": opened.is_file(),
        "confirmation_prediction_files": [path.relative_to(root).as_posix() for path in confirmation_outputs],
    }


def _code_tree_sha256(root: Path) -> str:
    paths = sorted(
        [
            *(
                root
                / "src/elderly_monitoring/modules/mental_health/mood_social/r5"
            ).glob("*.py"),
            *root.glob("scripts/*mood_social_v3_3_3_r5*.py"),
        ],
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_current_confirmation_recipe(repository_root: Path) -> dict[str, Any]:
    """Assemble the complete immutable recipe from development evidence."""

    root = Path(repository_root).resolve()
    paths = {
        "features": root
        / "reports/mental_health/mood_social/v3.3.3-r5/"
        "OPT-V333-R5-002-feature-ablation/retained_feature_set.json",
        "structure": root
        / "reports/mental_health/mood_social/v3.3.3-r5/"
        "OPT-V333-R5-003-structure/selected_structure.json",
        "fusion": root
        / "reports/mental_health/mood_social/v3.3.3-r5/"
        "OPT-V333-R5-005-fusion/locked_recipe.json",
        "development": root
        / "reports/mental_health/mood_social/v3.3.3-r5/"
        "OPT-V333-R5-005-fusion/development_summary.json",
        "config": root / "configs/training/mood_social_v3_3_3_r5.yaml",
        "protocol": root / DEFAULT_DATA_RELATIVE / "r5_protocol_manifest.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"r5 confirmation recipe inputs are missing: {missing}")
    feature = json.loads(paths["features"].read_text(encoding="utf-8"))
    structure = json.loads(paths["structure"].read_text(encoding="utf-8"))
    fusion = json.loads(paths["fusion"].read_text(encoding="utf-8"))
    development = json.loads(paths["development"].read_text(encoding="utf-8"))
    if feature.get("confirmation_opened") or development.get("confirmation_opened"):
        raise ValueError("r5 development evidence claims confirmation was already opened")
    if fusion.get("status") != "locked_before_confirmation":
        raise ValueError("r5 development fusion recipe is not locked")
    return {
        "features": {
            "path": paths["features"].relative_to(root).as_posix(),
            "sha256": sha256_file(paths["features"]),
            "retained_feature_count": int(feature["retained_feature_count"]),
            "retained_feature_names_sha256": hashlib.sha256(
                "\n".join(feature["retained_features"]).encode("utf-8")
            ).hexdigest(),
        },
        "candidates": {
            track: payload["stable_single_spec"]
            for track, payload in fusion["tracks"].items()
        },
        "objective_structure": {
            "path": paths["structure"].relative_to(root).as_posix(),
            "sha256": sha256_file(paths["structure"]),
            "structure": structure["structure"],
        },
        "training_weights": {"scheme": structure["weight_scheme"]},
        "fusion": {
            "path": paths["fusion"].relative_to(root).as_posix(),
            "sha256": sha256_file(paths["fusion"]),
            "member_rule": fusion["member_rule"],
            "fusion_rule": fusion["fusion_rule"],
        },
        "calibration": fusion["calibration_rule"],
        "workpoints": {"selection": fusion["workpoint_rule"]},
        "development_evidence_sha256": sha256_file(paths["development"]),
        "protocol_manifest_sha256": sha256_file(paths["protocol"]),
        "config_sha256": sha256_file(paths["config"]),
        "implementation_code_tree_sha256": _code_tree_sha256(root),
        "implementation_git_head": _git_head(root),
    }


def locked_confirmation_recipe(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path = root / DEFAULT_LOCK_RELATIVE / "candidate_recipe_lock.json"
    if not path.is_file():
        raise FileNotFoundError("r5 confirmation recipe is not locked")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("confirmation_seed") != R5_CONFIRMATION_SEED:
        raise ValueError("r5 confirmation lock seed drifted")
    return payload


def verify_locked_confirmation_recipe(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    payload = locked_confirmation_recipe(root)
    observed = build_current_confirmation_recipe(root)
    if payload["recipe"] != observed:
        raise ValueError("r5 locked confirmation recipe no longer matches development evidence/code")
    opened = root / DEFAULT_LOCK_RELATIVE / "OPENED.json"
    if opened.is_file():
        opening = json.loads(opened.read_text(encoding="utf-8"))
        lock_path = root / DEFAULT_LOCK_RELATIVE / "candidate_recipe_lock.json"
        if opening.get("recipe_sha256") != sha256_file(lock_path):
            raise ValueError("r5 confirmation opening no longer matches recipe lock")
    return payload


def lock_current_confirmation_recipe(*, repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    return lock_confirmation_recipe(
        repository_root=root,
        recipe=build_current_confirmation_recipe(root),
    )


def lock_confirmation_recipe(
    *,
    repository_root: Path,
    recipe: dict[str, Any],
) -> dict[str, Any]:
    """Lock the full candidate recipe before confirmation can be opened."""

    root = Path(repository_root).resolve()
    output = root / DEFAULT_LOCK_RELATIVE
    state = confirmation_state(root)
    if state["opened"]:
        raise PermissionError("r5 confirmation is already opened and cannot be relocked")
    recipe_path = output / "candidate_recipe_lock.json"
    if recipe_path.exists():
        raise FileExistsError("r5 confirmation recipe is already locked")
    required = {
        "features",
        "candidates",
        "objective_structure",
        "training_weights",
        "fusion",
        "calibration",
        "workpoints",
        "development_evidence_sha256",
    }
    missing = sorted(required.difference(recipe))
    if missing:
        raise ValueError(f"r5 confirmation recipe is incomplete: {missing}")
    payload = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "confirmation_seed": R5_CONFIRMATION_SEED,
        "confirmation_open_once": True,
        "evidence_level": "sealed reused-cohort confirmation",
        "recipe": recipe,
    }
    _write_json(recipe_path, payload)
    return {"path": recipe_path.relative_to(root).as_posix(), "sha256": sha256_file(recipe_path)}


def open_confirmation_once(*, repository_root: Path) -> dict[str, Any]:
    """Irreversibly record the one-time open after all development choices lock."""

    root = Path(repository_root).resolve()
    output = root / DEFAULT_LOCK_RELATIVE
    recipe_path = output / "candidate_recipe_lock.json"
    opened_path = output / "OPENED.json"
    if not recipe_path.is_file():
        raise PermissionError("r5 confirmation recipe must be locked before opening")
    if opened_path.exists():
        raise PermissionError("r5 confirmation was already opened; re-open is forbidden")
    verify_locked_confirmation_recipe(root)
    protocol_manifest = root / DEFAULT_DATA_RELATIVE / "r5_protocol_manifest.json"
    r5_000_manifest = root / DEFAULT_REPORT_RELATIVE / "r4-paired-baseline" / "development_paired_baseline_summary.json"
    for dependency in (protocol_manifest, r5_000_manifest):
        if not dependency.is_file():
            raise FileNotFoundError(f"r5 confirmation prerequisite is missing: {dependency}")
    payload = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "confirmation_seed": R5_CONFIRMATION_SEED,
        "opened": True,
        "opened_after_recipe_lock": True,
        "reopen_allowed": False,
        "recipe_sha256": sha256_file(recipe_path),
        "protocol_manifest_sha256": sha256_file(protocol_manifest),
        "paired_baseline_summary_sha256": sha256_file(r5_000_manifest),
    }
    _write_json(opened_path, payload)
    return payload


__all__ = [
    "build_current_confirmation_recipe",
    "confirmation_state",
    "lock_confirmation_recipe",
    "lock_current_confirmation_recipe",
    "locked_confirmation_recipe",
    "open_confirmation_once",
    "verify_locked_confirmation_recipe",
]
