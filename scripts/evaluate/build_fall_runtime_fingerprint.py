from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and fingerprint the fall-risk runtime acceptance inputs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/modules/fall_risk_runtime_acceptance.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def build_fingerprint(config_path: Path) -> dict[str, Any]:
    raw = config_path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("runtime acceptance config must be a mapping")
    _validate_config(config)
    inputs = []
    for item in config["fixed_inputs"]:
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"fixed input is unavailable: {path}")
        inputs.append({
            "input_id": item["input_id"],
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "acceptance_id": config["acceptance_id"],
        "acceptance_status": config["status"],
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "git_revision": _git_value("rev-parse", "HEAD"),
        "git_worktree_dirty": bool(_git_value("status", "--porcelain")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "fixed_inputs": inputs,
        "external_acceptance": config["external_acceptance"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fingerprint = build_fingerprint(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fingerprint, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote runtime fingerprint: {args.output}")
    return 0


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "acceptance_id",
        "status",
        "runtime",
        "clock_contract",
        "branch_quality_contract",
        "fixed_inputs",
        "fault_injection",
        "budgets",
        "external_acceptance",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"runtime acceptance config missing fields: {', '.join(missing)}")
    if not isinstance(config["fixed_inputs"], list) or not config["fixed_inputs"]:
        raise ValueError("runtime acceptance config requires at least one fixed input")
    states = config["branch_quality_contract"].get("branch_states")
    if states != ["valid", "unavailable", "inference_error"]:
        raise ValueError(
            "branch states must freeze valid/unavailable/inference_error"
        )
    if config["external_acceptance"].get("status") != "blocked_external":
        raise ValueError("external acceptance cannot be marked complete by offline tooling")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())
