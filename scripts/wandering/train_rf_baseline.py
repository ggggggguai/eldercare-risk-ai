from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from elderly_monitoring.modules.mental_health.wandering.rf_baseline import (
    build_development_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the comparison_only wandering RF development artifact. "
            "Only train is fitted; validation is evaluation-only; test and official are unavailable."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/modules/wandering_rf_v1.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/mental_health/wandering_step5/development/v1"),
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("reports/mental_health/wandering_step5"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_development_artifacts(
        rf_config_path=args.config,
        project_root=args.project_root,
        output_dir=args.output,
    )
    _write_supplementary_reports(
        report_root=args.report_root,
        runtime=result.runtime_benchmark,
        metrics=result.validation_metrics,
        failures=result.failure_cases,
        phase="validation",
    )
    print(f"development_manifest_sha256={result.manifest_sha256}")
    return 0


def _write_supplementary_reports(
    *,
    report_root: Path,
    runtime: Mapping[str, Any] | None,
    metrics: Mapping[str, Any],
    failures: Mapping[str, Any],
    phase: str,
) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    if runtime is not None:
        _write_json_atomic(report_root / "runtime_benchmark.json", runtime)
    _write_json_atomic(report_root / "failure_cases" / f"{phase}.json", failures)
    matrices = {
        "schema_version": "wandering-rf-confusion-matrices-v1",
        "phase": phase,
        "tasks": {
            task: {
                seed: {
                    cohort: values["confusion_matrix"]
                    for cohort, values in seed_values["cohorts"].items()
                }
                for seed, seed_values in task_values["seeds"].items()
            }
            for task, task_values in metrics["tasks"].items()
        },
    }
    _write_json_atomic(report_root / "confusion_matrices" / f"{phase}.json", matrices)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
