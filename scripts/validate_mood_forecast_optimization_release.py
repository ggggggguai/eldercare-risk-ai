"""Independently validate FORECAST-OPT-001G delivery and isolation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ALGORITHM_ROOT.parents[1]
sys.path.insert(0, str(ALGORITHM_ROOT / "src"))

from elderly_monitoring.modules.mental_health.mood_social.forecast_experiment import (  # noqa: E402
    sha256_file,
)


FORECAST_ROOT = (
    ALGORITHM_ROOT / "artifacts" / "mental_health" / "mood_social" / "forecast_v3.4"
)
RUN_ROOT = FORECAST_ROOT / "optimization_candidates" / "MH-20260807-FOPT-002"
RELEASE_ROOT = RUN_ROOT / "release"
EVALUATION_ROOT = RUN_ROOT / "evaluation"
V33_ROOT = ALGORITHM_ROOT / "models" / "mental_health" / "mood_social" / "v3.3.3"
BACKEND_TOKENS = ("forecast_1m", "forecast_2m", "mood_forecast", "forecast_v3.4")
REQUIRED_RELEASE_FILES = {
    "forecast_1m_001d_candidate_model_card.md",
    "forecast_1m_001d_candidate_model_card.json",
    "forecast_2m_001d_candidate_model_card.md",
    "forecast_2m_001d_candidate_model_card.json",
    "isolation_report.json",
    "rollback_decision.json",
    "final_delivery_report.md",
    "release_manifest.json",
    "validation_report.json",
}


@dataclass
class Checks:
    count: int = 0

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)
        self.count += 1


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object expected: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_sums(root: Path) -> None:
    rows = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _verify_sums(checks: Checks, root: Path) -> None:
    sums = root / "SHA256SUMS"
    listed: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checks.require(relative not in listed, f"duplicate checksum: {relative}")
        listed.add(relative)
        path = root / relative
        checks.require(path.is_file(), f"checksum target missing: {relative}")
        checks.require(sha256_file(path) == digest, f"checksum mismatch: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    checks.require(actual == listed, "SHA256SUMS does not cover the complete release")


def _backend_hits() -> list[str]:
    hits: list[str] = []
    backend = WORKSPACE_ROOT / "backend"
    for path in backend.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".py",
            ".json",
            ".yaml",
            ".yml",
            ".md",
        }:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        if any(token in content for token in BACKEND_TOKENS):
            hits.append(path.relative_to(backend).as_posix())
    return sorted(hits)


def _verify_v33(checks: Checks, isolation: dict[str, Any]) -> None:
    acceptance_path = V33_ROOT / "acceptance.json"
    acceptance = _json(acceptance_path)
    checks.require(acceptance["status"] == "passed", "V3.3 acceptance status changed")
    checks.require(
        acceptance["model_version"] == "mood-fusion-v3.3.3", "online model changed"
    )
    checks.require(
        acceptance["package_run_id"] == "MH-20260802-013", "active V3.3 package changed"
    )
    checks.require(
        sha256_file(acceptance_path) == isolation["v3_3"]["acceptance_sha256"],
        "V3.3 acceptance hash drift",
    )
    package = ALGORITHM_ROOT / acceptance["package_path"]
    checks.require(
        sha256_file(package / "manifest.json")
        == isolation["v3_3"]["package_manifest_sha256"],
        "V3.3 package manifest drift",
    )
    checks.require(
        sha256_file(package / "SHA256SUMS")
        == isolation["v3_3"]["package_checksums_sha256"],
        "V3.3 package checksum drift",
    )
    checks.require(
        sha256_file(
            ALGORITHM_ROOT
            / "src/elderly_monitoring/modules/mental_health/mood_social/pipeline.py"
        )
        == isolation["v3_3"]["pipeline_sha256"],
        "pipeline drift",
    )
    checks.require(
        sha256_file(
            ALGORITHM_ROOT
            / "src/elderly_monitoring/modules/mental_health/mood_social/package_selection.py"
        )
        == isolation["v3_3"]["package_selection_sha256"],
        "package selection drift",
    )
    checks.require(
        sha256_file(
            ALGORITHM_ROOT
            / "src/elderly_monitoring/modules/mental_health/mood_social/schemas.py"
        )
        == isolation["v3_3"]["schemas_sha256"],
        "schema drift",
    )
    checks.require(
        not any(
            "forecast" in path.name.lower()
            for path in V33_ROOT.rglob("*")
            if path.is_file()
        ),
        "forecast entered V3.3 model tree",
    )


def validate() -> dict[str, Any]:
    checks = Checks()
    manifest = _json(RELEASE_ROOT / "release_manifest.json")
    checks.require(manifest["task_id"] == "FORECAST-OPT-001G", "task identity changed")
    checks.require(manifest["run_id"] == "MH-20260807-FOPT-002", "run identity changed")
    checks.require(manifest["status"] == "release_completed", "release incomplete")
    checks.require(
        manifest["promotion_status"] == "retain_v3_4_baseline",
        "promotion status changed",
    )
    checks.require(
        manifest["recommended_fallback"] == "MH-20260805-FCAST-001", "fallback changed"
    )
    for key, expected in {
        "release_status": "experimental",
        "execution_mode": "offline_only",
        "decision_authority": "shadow_only",
        "product_integration_performed": False,
        "product_visible": False,
    }.items():
        checks.require(manifest[key] == expected, f"release boundary changed: {key}")
    for name in REQUIRED_RELEASE_FILES:
        checks.require(
            (RELEASE_ROOT / name).is_file(), f"missing release artifact: {name}"
        )
    checks.require(
        manifest["upstream_evaluation_manifest_sha256"]
        == sha256_file(EVALUATION_ROOT / "evaluation_manifest.json"),
        "evaluation manifest binding changed",
    )
    checks.require(
        manifest["upstream_promotion_decision_sha256"]
        == sha256_file(EVALUATION_ROOT / "promotion_decision.json"),
        "promotion decision binding changed",
    )
    checks.require(
        manifest["upstream_evaluation_validation_sha256"]
        == sha256_file(EVALUATION_ROOT / "validation_report.json"),
        "evaluation validation binding changed",
    )
    _verify_sums(checks, RELEASE_ROOT)
    _verify_sums(checks, RUN_ROOT)

    rollback = _json(RELEASE_ROOT / "rollback_decision.json")
    checks.require(rollback["candidate_promoted"] is False, "candidate was promoted")
    checks.require(
        rollback["online_model_version"] == "mood-fusion-v3.3.3",
        "online version changed",
    )
    checks.require(
        rollback["online_package_run_id"] == "MH-20260802-013", "online package changed"
    )
    isolation = _json(RELEASE_ROOT / "isolation_report.json")
    checks.require(
        isolation["forecast_backend_hits"] == [], "backend contains forecast references"
    )
    checks.require(
        isolation["product_integration_performed"] is False, "integration was enabled"
    )
    checks.require(
        isolation["online_model_version"] == "mood-fusion-v3.3.3",
        "isolation model version changed",
    )
    checks.require(
        isolation["online_package_run_id"] == "MH-20260802-013",
        "isolation package changed",
    )
    checks.require(_backend_hits() == [], "backend changed to reference forecast")
    _verify_v33(checks, isolation)

    decision = _json(EVALUATION_ROOT / "promotion_decision.json")
    checks.require(
        decision["status"] == "retain_v3_4_baseline", "001F decision changed"
    )
    cards = []
    for task_id in ("forecast_1m", "forecast_2m"):
        card = _json(RELEASE_ROOT / f"{task_id}_001d_candidate_model_card.json")
        checks.require(card["task_id"] == task_id, f"card task changed: {task_id}")
        checks.require(
            card["candidate_status"] == "audit_only_not_promoted",
            f"card promoted: {task_id}",
        )
        checks.require(
            card["product_visible"] is False,
            f"card product visibility changed: {task_id}",
        )
        markdown = (RELEASE_ROOT / f"{task_id}_001d_candidate_model_card.md").read_text(
            encoding="utf-8"
        )
        checks.require(
            "not a diagnosis" in markdown, f"card boundary missing: {task_id}"
        )
        checks.require("audit only" in markdown, f"card fallback missing: {task_id}")
        cards.append(task_id)
    report = (RELEASE_ROOT / "final_delivery_report.md").read_text(encoding="utf-8")
    for required in (
        "retain_v3_4_baseline",
        "MH-20260805-FCAST-001",
        "mood-fusion-v3.3.3",
        "offline_only",
    ):
        checks.require(required in report, f"delivery report missing: {required}")
    checks.require(
        not any(term in report for term in ("进入家属端", "抑郁症诊断", "已经部署")),
        "delivery report contains a prohibited claim",
    )
    validation = {
        "task_id": "FORECAST-OPT-001G",
        "run_id": manifest["run_id"],
        "status": "pass",
        "checks_passed": checks.count,
        "promotion_status": manifest["promotion_status"],
        "recommended_fallback": manifest["recommended_fallback"],
        "online_model_version": isolation["online_model_version"],
        "online_package_run_id": isolation["online_package_run_id"],
        "product_integration_performed": False,
        "release_status": manifest["release_status"],
        "execution_mode": manifest["execution_mode"],
        "decision_authority": manifest["decision_authority"],
        "product_visible": False,
        "cards": cards,
        "validation_code_sha256": sha256_file(Path(__file__).resolve()),
    }
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    if args.write_report:
        _write_json(RELEASE_ROOT / "validation_report.json", {})
        _write_sums(RELEASE_ROOT)
        _write_sums(RUN_ROOT)
        result = validate()
        _write_json(RELEASE_ROOT / "validation_report.json", result)
        _write_sums(RELEASE_ROOT)
        _write_sums(RUN_ROOT)
        result = validate()
    else:
        result = validate()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
