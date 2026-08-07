"""Audit and select the OPT-COG-002 Audio/Text backbone on frozen folds."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .build_subject_cv_v34 import DEFAULT_OUTPUT_PATH
    from .common import sha256_file, workspace_relative
    from .train import write_json
    from .train_v34 import PREDICTION_FILES, REPORT_ROOT, V34TrainingError, verify_prediction_bundle
    from .v34_metrics import temporary_platt_evaluation
    from .train_v34 import _raw_metric_report
except ImportError:
    from build_subject_cv_v34 import DEFAULT_OUTPUT_PATH  # type: ignore[no-redef]
    from common import sha256_file, workspace_relative  # type: ignore[no-redef]
    from train import write_json  # type: ignore[no-redef]
    from train_v34 import (  # type: ignore[no-redef]
        PREDICTION_FILES,
        REPORT_ROOT,
        V34TrainingError,
        _raw_metric_report,
        verify_prediction_bundle,
    )
    from v34_metrics import temporary_platt_evaluation  # type: ignore[no-redef]


BACKBONE_IDS = ("V34-A1", "V34-A2", "V34-A3")
ALL_COMPARISON_IDS = ("V34-A0", *BACKBONE_IDS)
COMPLEXITY_RANK = {"V34-A2": 1, "V34-A1": 2, "V34-A3": 3}
ALPHAS = (0.10, 0.25, 0.50)


def _read_rows(candidate_dir: Path, role: str) -> list[dict[str, Any]]:
    path = candidate_dir / PREDICTION_FILES[role]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _complete_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(row["sample_id"])
        for row in rows
        if row.get("raw_logit") is not None
        and all(int(row[f"{name}_missing_mask"]) == 0 for name in ("audio", "text", "face"))
    }


def _summary(run_dir: Path, candidate_id: str) -> dict[str, Any]:
    path = run_dir / f"{candidate_id}_cv_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("candidate_id") != candidate_id or payload.get("official_test_evaluated") is not False:
        raise V34TrainingError(f"candidate summary boundary mismatch: {path}")
    return payload


def _ranking(candidates: Mapping[str, Mapping[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    for candidate_id in BACKBONE_IDS:
        summary = candidates[candidate_id]
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_valid": bool(summary.get("candidate_valid", True)),
                "subject_auc_mean": float(summary["subject_raw_auc"]["mean"]),
                "subject_auc_population_std": float(summary["subject_raw_auc"]["population_std"]),
                "temporary_specificity": float(summary["pooled_temporary_workpoint"]["specificity"]),
                "task_auc_mean": float(summary["task_raw_auc"]["mean"]),
                "complexity_rank": COMPLEXITY_RANK[candidate_id],
            }
        )
    valid_rows = [row for row in rows if row["candidate_valid"]]
    if not valid_rows:
        raise V34TrainingError("OPT-COG-002 has no valid Audio/Text candidate")
    best_auc = max(row["subject_auc_mean"] for row in valid_rows)
    contenders = [
        row for row in valid_rows if best_auc - row["subject_auc_mean"] < 0.005
    ]
    contenders.sort(
        key=lambda row: (
            row["subject_auc_population_std"],
            -row["temporary_specificity"],
            -row["task_auc_mean"],
            row["complexity_rank"],
            row["candidate_id"],
        )
    )
    selected = str(contenders[0]["candidate_id"])
    ordered = sorted(
        rows,
        key=lambda row: (
            0 if row["candidate_id"] == selected else 1,
            -row["subject_auc_mean"],
            row["candidate_id"],
        ),
    )
    return selected, ordered


def _a4_rows(
    at_rows: Sequence[Mapping[str, Any]],
    face_rows: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
) -> list[dict[str, Any]]:
    faces = {str(row["sample_id"]): row for row in face_rows}
    if {str(row["sample_id"]) for row in at_rows} != set(faces):
        raise V34TrainingError("early A4 Audio/Text and Face sample sets differ")
    result = []
    for source in at_rows:
        row = copy.deepcopy(dict(source))
        face = faces[str(row["sample_id"])]
        if row["raw_logit"] is not None and face["raw_logit"] is not None:
            row["raw_logit"] = float(row["raw_logit"]) + float(alpha) * float(face["q_face"]) * math.tanh(float(face["raw_logit"]))
        row["face_logit"] = None if face["raw_logit"] is None else float(face["raw_logit"])
        result.append(row)
    return result


def _early_a4_diagnostic(
    candidate_runs: Mapping[str, Path], face_run: Path
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for candidate_id in BACKBONE_IDS:
        fold_rows = []
        for outer_fold in range(5):
            at_dir = candidate_runs[candidate_id] / f"fold_{outer_fold}" / candidate_id
            face_dir = face_run / f"fold_{outer_fold}" / "V34-Face"
            roles = {}
            for role in PREDICTION_FILES:
                at_rows = _read_rows(at_dir, role)
                face_rows = _read_rows(face_dir, role)
                roles[role] = (at_rows, face_rows)
            alpha_metrics = []
            for alpha in ALPHAS:
                validation = _a4_rows(*roles["inner_validation"], alpha=alpha)
                complete = [row for row in validation if str(row["sample_id"]) in _complete_ids(validation)]
                auc = _raw_metric_report(complete)["subject"]["roc_auc"]
                alpha_metrics.append({"alpha": alpha, "subject_auc": float(auc)})
            alpha_metrics.sort(key=lambda row: (-row["subject_auc"], row["alpha"]))
            selected_alpha = float(alpha_metrics[0]["alpha"])
            calibration = _a4_rows(*roles["inner_calibration"], alpha=selected_alpha)
            outer = _a4_rows(*roles["outer_evaluation"], alpha=selected_alpha)
            calibration_ids = _complete_ids(calibration)
            outer_ids = _complete_ids(outer)
            calibrated = temporary_platt_evaluation(
                [row for row in calibration if str(row["sample_id"]) in calibration_ids],
                [row for row in outer if str(row["sample_id"]) in outer_ids],
            )
            fold_rows.append(
                {
                    "outer_fold": outer_fold,
                    "alpha_candidates": alpha_metrics,
                    "selected_alpha": selected_alpha,
                    "outer_raw_subject_metrics": _raw_metric_report(
                        [row for row in outer if str(row["sample_id"]) in outer_ids]
                    )["subject"],
                    "temporary_outer_subject_metrics": calibrated["outer_evaluation_subject_metrics"],
                }
            )
        report[candidate_id] = {
            "purpose": "diagnostic_only_not_face_production_decision",
            "folds": fold_rows,
        }
    return report


def select(
    *,
    run_dirs: Mapping[str, Path],
    face_run: Path,
    output_path: Path,
) -> dict[str, Any]:
    split_sha = sha256_file(DEFAULT_OUTPUT_PATH)
    summaries = {candidate_id: _summary(run_dirs[candidate_id], candidate_id) for candidate_id in ALL_COMPARISON_IDS}
    alignment = []
    for outer_fold in range(5):
        role_counts = {}
        for role in PREDICTION_FILES:
            expected: set[str] | None = None
            for candidate_id in ALL_COMPARISON_IDS:
                candidate_dir = run_dirs[candidate_id] / f"fold_{outer_fold}" / candidate_id
                verify_prediction_bundle(candidate_dir, expected_split_sha256=split_sha)
                current = _complete_ids(_read_rows(candidate_dir, role))
                if expected is None:
                    expected = current
                elif current != expected:
                    raise V34TrainingError(
                        f"candidate comparison population differs fold={outer_fold} role={role}"
                    )
            role_counts[role] = len(expected or ())
        face_dir = face_run / f"fold_{outer_fold}" / "V34-Face"
        verify_prediction_bundle(face_dir, expected_split_sha256=split_sha)
        alignment.append({"outer_fold": outer_fold, "complete_task_counts": role_counts})
    selected, ranking = _ranking(summaries)
    report = {
        "schema_version": "cognitive_v34_opt_cog_002_selection_v1",
        "task_id": "OPT-COG-002",
        "status": "passed",
        "official_test_evaluated": False,
        "split_sha256": split_sha,
        "candidate_count": len(ALL_COMPARISON_IDS),
        "eligible_audio_text_candidates": list(BACKBONE_IDS),
        "selected_audio_text_candidate": selected,
        "ranking_rule": {
            "primary": "five_fold_subject_auc_mean",
            "auc_tie_tolerance": 0.005,
            "tie_breaks": [
                "subject_auc_population_std_ascending",
                "temporary_workpoint_specificity_descending",
                "task_auc_mean_descending",
                "complexity_ascending",
            ],
        },
        "ranking": ranking,
        "runs": {
            candidate_id: {
                "path": workspace_relative(path),
                "summary_sha256": sha256_file(path / f"{candidate_id}_cv_summary.json"),
            }
            for candidate_id, path in run_dirs.items()
        },
        "face_run": {
            "path": workspace_relative(face_run),
            "summary_sha256": sha256_file(face_run / "V34-Face_cv_summary.json"),
            "decision_scope": "frozen_z_face_and_early_a4_diagnostic_only",
        },
        "population_alignment": alignment,
        "early_a4_diagnostic": _early_a4_diagnostic(run_dirs, face_run),
        "limitations": [
            "All metrics are development-set five-fold evidence on CogPic official Train.",
            "The independent official Test was not read or evaluated.",
            "Face production inclusion is deferred to OPT-COG-004.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for candidate_id in ALL_COMPARISON_IDS:
        parser.add_argument(f"--{candidate_id.lower().replace('-', '_')}-run", required=True)
    parser.add_argument("--face-run", required=True)
    parser.add_argument("--output", type=Path, default=REPORT_ROOT / "OPT-COG-002_selection_report.json")
    return parser.parse_args()


def _resolve_run(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPORT_ROOT / "runs" / path


def main() -> int:
    args = parse_args()
    runs = {
        candidate_id: _resolve_run(getattr(args, f"{candidate_id.lower().replace('-', '_')}_run"))
        for candidate_id in ALL_COMPARISON_IDS
    }
    report = select(
        run_dirs=runs,
        face_run=_resolve_run(args.face_run),
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["select"]
