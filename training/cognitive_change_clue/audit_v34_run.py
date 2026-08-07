"""Audit V3.4 frozen prediction bundles without running model inference."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .build_subject_cv_v34 import DEFAULT_AUDIT_PATH, DEFAULT_OUTPUT_PATH
    from .common import ALGORITHM_ROOT, sha256_file, workspace_relative
    from .train import write_json
    from .train_v34 import PREDICTION_FILES, REPORT_ROOT, verify_prediction_bundle
except ImportError:
    from build_subject_cv_v34 import DEFAULT_AUDIT_PATH, DEFAULT_OUTPUT_PATH  # type: ignore[no-redef]
    from common import ALGORITHM_ROOT, sha256_file, workspace_relative  # type: ignore[no-redef]
    from train import write_json  # type: ignore[no-redef]
    from train_v34 import (  # type: ignore[no-redef]
        PREDICTION_FILES,
        REPORT_ROOT,
        verify_prediction_bundle,
    )


class V34RunAuditError(RuntimeError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise V34RunAuditError(f"non-finite JSON value in {path}")
    return rows


def audit_run(
    run_dir: str | Path,
    *,
    candidate_id: str = "V34-A0",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    split = json.loads(DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))
    split_audit = json.loads(DEFAULT_AUDIT_PATH.read_text(encoding="utf-8"))
    if split_audit.get("status") != "passed":
        raise V34RunAuditError("V3.4 split audit is not passed")
    split_sha = sha256_file(DEFAULT_OUTPUT_PATH)
    outer_samples: Counter[str] = Counter()
    outer_subjects: Counter[str] = Counter()
    fold_reports: list[dict[str, Any]] = []
    violations: list[str] = []
    for fold in split["folds"]:
        outer_fold = int(fold["outer_fold"])
        candidate_dir = run_dir / f"fold_{outer_fold}" / candidate_id
        manifest = verify_prediction_bundle(
            candidate_dir, expected_split_sha256=split_sha
        )
        checkpoint_path = ALGORITHM_ROOT.parents[1] / str(manifest["checkpoint_path"])
        if not checkpoint_path.is_file():
            violations.append(f"fold {outer_fold} checkpoint is missing")
        elif sha256_file(checkpoint_path) != manifest["checkpoint_sha256"]:
            violations.append(f"fold {outer_fold} checkpoint hash mismatch")
        role_reports: dict[str, Any] = {}
        for role, file_name in PREDICTION_FILES.items():
            rows = _read_jsonl(candidate_dir / file_name)
            expected_subjects = set(str(value) for value in fold[f"{role}_subjects"])
            observed_subjects = set(str(row["subject_id"]) for row in rows)
            if observed_subjects != expected_subjects:
                violations.append(f"fold {outer_fold} {role} subject set mismatch")
            if role == "outer_evaluation":
                outer_samples.update(str(row["sample_id"]) for row in rows)
                outer_subjects.update(observed_subjects)
            role_reports[role] = {
                "row_count": len(rows),
                "subject_count": len(observed_subjects),
                "scored_row_count": sum(row.get("raw_logit") is not None for row in rows),
                "sha256": sha256_file(candidate_dir / file_name),
            }
        fold_reports.append(
            {
                "outer_fold": outer_fold,
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "roles": role_reports,
            }
        )
    duplicate_outer_samples = sorted(key for key, count in outer_samples.items() if count != 1)
    duplicate_outer_subjects = sorted(key for key, count in outer_subjects.items() if count != 1)
    if len(outer_samples) != 1377:
        violations.append(f"outer OOF task coverage is {len(outer_samples)}, expected 1377")
    if len(outer_subjects) != 459:
        violations.append(f"outer OOF subject coverage is {len(outer_subjects)}, expected 459")
    if duplicate_outer_samples:
        violations.append("one or more tasks enter outer evaluation more than once")
    if duplicate_outer_subjects:
        violations.append("one or more subjects enter outer evaluation more than once")
    summary_path = run_dir / f"{candidate_id}_cv_summary.json"
    if not summary_path.is_file():
        violations.append("candidate CV summary is missing")
    report = {
        "schema_version": "cognitive_v34_run_audit_v1",
        "status": "passed" if not violations else "failed",
        "run_id": run_dir.name,
        "candidate_id": candidate_id,
        "split_path": workspace_relative(DEFAULT_OUTPUT_PATH),
        "split_sha256": split_sha,
        "split_audit_sha256": sha256_file(DEFAULT_AUDIT_PATH),
        "official_train_subject_count": 459,
        "official_train_task_count": 1377,
        "official_test_overlap_count": 0,
        "outer_evaluation_unique_subject_count": len(outer_subjects),
        "outer_evaluation_unique_task_count": len(outer_samples),
        "outer_evaluation_duplicate_subjects": duplicate_outer_subjects,
        "outer_evaluation_duplicate_tasks": duplicate_outer_samples,
        "folds": fold_reports,
        "candidate_summary_sha256": sha256_file(summary_path) if summary_path.is_file() else None,
        "official_test_evaluated": False,
        "violations": violations,
    }
    target = Path(output_path) if output_path else REPORT_ROOT / f"{candidate_id}_run_audit.json"
    write_json(target, report)
    if violations:
        raise V34RunAuditError("; ".join(violations))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate", default="V34-A0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_run(
        args.run_dir, candidate_id=args.candidate, output_path=args.output
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["V34RunAuditError", "audit_run"]
