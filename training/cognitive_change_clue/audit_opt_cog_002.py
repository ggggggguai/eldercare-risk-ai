"""Independent artifact audit for OPT-COG-002."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .common import WORKSPACE_ROOT, sha256_file, workspace_relative
    from .train import write_json
    from .train_v34 import REPORT_ROOT, V34TrainingError, verify_prediction_bundle
except ImportError:
    from common import WORKSPACE_ROOT, sha256_file, workspace_relative  # type: ignore[no-redef]
    from train import write_json  # type: ignore[no-redef]
    from train_v34 import REPORT_ROOT, V34TrainingError, verify_prediction_bundle  # type: ignore[no-redef]


def _workspace_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def audit(selection_path: Path, output_path: Path) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "passed" or selection.get("official_test_evaluated") is not False:
        raise V34TrainingError("OPT-COG-002 selection report did not pass")
    split_sha = str(selection["split_sha256"])
    audited_candidates = []
    for candidate_id, run_record in selection["runs"].items():
        run_dir = _workspace_path(run_record["path"])
        summary_path = run_dir / f"{candidate_id}_cv_summary.json"
        if sha256_file(summary_path) != run_record["summary_sha256"]:
            raise V34TrainingError(f"summary hash mismatch: {candidate_id}")
        for outer_fold in range(5):
            candidate_dir = run_dir / f"fold_{outer_fold}" / candidate_id
            verify_prediction_bundle(candidate_dir, expected_split_sha256=split_sha)
            if candidate_id == "V34-A3":
                _audit_a3_fold(candidate_dir, split_sha)
        audited_candidates.append(candidate_id)
    face_run = _workspace_path(selection["face_run"]["path"])
    face_summary = face_run / "V34-Face_cv_summary.json"
    if sha256_file(face_summary) != selection["face_run"]["summary_sha256"]:
        raise V34TrainingError("Face summary hash mismatch")
    for outer_fold in range(5):
        verify_prediction_bundle(
            face_run / f"fold_{outer_fold}" / "V34-Face",
            expected_split_sha256=split_sha,
        )
    report = {
        "schema_version": "cognitive_v34_opt_cog_002_audit_v1",
        "task_id": "OPT-COG-002",
        "status": "passed",
        "official_test_evaluated": False,
        "selection_report": workspace_relative(selection_path),
        "selection_report_sha256": sha256_file(selection_path),
        "split_sha256": split_sha,
        "audited_candidates": sorted(audited_candidates),
        "audited_candidate_folds": len(audited_candidates) * 5,
        "audited_face_folds": 5,
        "selected_audio_text_candidate": selection["selected_audio_text_candidate"],
    }
    write_json(output_path, report)
    return report


def _audit_a3_fold(candidate_dir: Path, split_sha: str) -> None:
    bundle_path = candidate_dir / "late_fusion_bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if (
        bundle.get("schema_version") != "cognitive_v34_a3_fold_bundle_v1"
        or bundle.get("candidate_id") != "V34-A3"
        or bundle.get("split_sha256") != split_sha
    ):
        raise V34TrainingError(f"A3 bundle boundary mismatch: {candidate_dir}")
    for record in bundle.get("base_fold_metrics", {}).values():
        path = _workspace_path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise V34TrainingError("A3 base fold metrics hash mismatch")
    audit_path = candidate_dir / "inner_oof_split_audit.json"
    inner = json.loads(audit_path.read_text(encoding="utf-8"))
    if inner.get("status") != "passed" or len(inner.get("folds", ())) != 5:
        raise V34TrainingError(f"A3 inner OOF audit did not pass: {candidate_dir}")
    evaluation_subjects = []
    for fold in inner["folds"]:
        train = set(fold["train_subjects"])
        evaluation = set(fold["evaluation_subjects"])
        if train.intersection(evaluation):
            raise V34TrainingError("A3 inner OOF subject leakage")
        evaluation_subjects.extend(evaluation)
    if len(evaluation_subjects) != len(set(evaluation_subjects)) or len(evaluation_subjects) != int(inner["subject_count"]):
        raise V34TrainingError("A3 inner OOF subject coverage mismatch")
    for scope in ("inner_oof", "final"):
        records = bundle["components"][scope]
        containers = records.values() if scope == "inner_oof" else (records,)
        for container in containers:
            for record in container.values():
                checkpoint = _workspace_path(record["checkpoint"])
                if sha256_file(checkpoint) != record["checkpoint_sha256"]:
                    raise V34TrainingError("A3 component checkpoint hash mismatch")
                if "predictions" in record:
                    predictions = _workspace_path(record["predictions"])
                    if sha256_file(predictions) != record["predictions_sha256"]:
                        raise V34TrainingError("A3 component prediction hash mismatch")
    fit_path = candidate_dir / "inner_selection_oof_fusion_inputs.jsonl"
    fit_rows = [json.loads(line) for line in fit_path.read_text(encoding="utf-8").splitlines() if line]
    if len(fit_rows) != int(bundle["fusion"]["fit_task_count"]):
        raise V34TrainingError("A3 fusion input row count mismatch")
    if len({row["sample_id"] for row in fit_rows}) != len(fit_rows):
        raise V34TrainingError("A3 fusion input contains duplicate tasks")
    if len({row["subject_id"] for row in fit_rows}) != int(bundle["fusion"]["fit_subject_count"]):
        raise V34TrainingError("A3 fusion input subject count mismatch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=REPORT_ROOT / "OPT-COG-002_selection_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_ROOT / "OPT-COG-002_run_audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit(args.selection, args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit"]
