"""Reproducible offline environment manifest for V3.3.3-r3."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


ENVIRONMENT_VERSION = "mood-social-v3.3.3-r3-environment-v1"
DEFAULT_OUTPUT = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/environment_manifest.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def build_environment_manifest(
    *, repository_root: Path, output_path: Path | None = None
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = output_path or root / DEFAULT_OUTPUT
    if not output.is_absolute():
        output = root / output
    code_paths = sorted(
        (root / "src/elderly_monitoring/modules/mental_health/mood_social/r3").glob(
            "*.py"
        ),
        key=lambda item: item.name,
    )
    code_paths.extend(
        sorted(
            root.glob("scripts/*mood_social_v3_3_3_r3*.py"),
            key=lambda item: item.name,
        )
    )
    binding_paths = [
        root / "configs/training/mood_social_v3_3_3_r3.yaml",
        root
        / "data/processed/mental_health/mood_social/v3.3.3-r2/splits/split_manifest.json",
        root
        / "data/processed/mental_health/mood_social/v3.3.3-r2/r3/artifact_manifest.json",
    ]
    payload: dict[str, Any] = {
        "environment_version": ENVIRONMENT_VERSION,
        "offline_training": True,
        "dataset_download_performed": False,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "dependencies": {
            name: _version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "lightgbm",
                "catboost",
                "xgboost",
                "joblib",
                "pyarrow",
                "pydantic",
                "fastapi",
            )
        },
        "candidate_availability": {
            "xgboost": _version("xgboost") is not None,
            "xgboost_search_status": (
                "available"
                if _version("xgboost") is not None
                else "all frozen XGBoost candidates recorded as unavailable"
            ),
        },
        "git": {
            "head": _git(root, "rev-parse", "HEAD"),
            "branch": _git(root, "branch", "--show-current"),
            "tracked_diff_sha256": hashlib.sha256(
                _git(root, "diff", "--binary").encode("utf-8")
            ).hexdigest(),
            "index_diff_sha256": hashlib.sha256(
                _git(root, "diff", "--cached", "--binary").encode("utf-8")
            ).hexdigest(),
        },
        "code": {
            path.relative_to(root).as_posix(): _sha256_file(path)
            for path in code_paths
        },
        "bindings": {
            path.relative_to(root).as_posix(): _sha256_file(path)
            for path in binding_paths
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = ["ENVIRONMENT_VERSION", "build_environment_manifest"]
