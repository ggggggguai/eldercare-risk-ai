"""Freeze V3.4 calibration, workpoints, and the final Face decision."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

try:
    from .common import WORKSPACE_ROOT, sha256_file, workspace_relative
    from .train import write_json, write_jsonl
    from .train_v34 import (
        PREDICTION_FILES,
        REPORT_ROOT,
        V34TrainingError,
        _raw_metric_report,
        _write_prediction_bundle,
        verify_prediction_bundle,
    )
    from .v34_metrics import (
        aggregate_subjects,
        apply_calibrator,
        binary_metrics,
        fit_calibrator,
        select_workpoint,
    )
except ImportError:
    from common import WORKSPACE_ROOT, sha256_file, workspace_relative  # type: ignore[no-redef]
    from train import write_json, write_jsonl  # type: ignore[no-redef]
    from train_v34 import (  # type: ignore[no-redef]
        PREDICTION_FILES,
        REPORT_ROOT,
        V34TrainingError,
        _raw_metric_report,
        _write_prediction_bundle,
        verify_prediction_bundle,
    )
    from v34_metrics import (  # type: ignore[no-redef]
        aggregate_subjects,
        apply_calibrator,
        binary_metrics,
        fit_calibrator,
        select_workpoint,
    )


METHODS = ("platt", "temperature", "isotonic")
METHOD_PREFERENCE = {"platt": 0, "temperature": 1, "isotonic": 2}
ALPHAS = (0.10, 0.25, 0.50)


def _read_rows(candidate_dir: Path, role: str) -> list[dict[str, Any]]:
    path = candidate_dir / PREDICTION_FILES[role]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _align(
    at_rows: Sequence[Mapping[str, Any]], face_rows: Sequence[Mapping[str, Any]]
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    faces = {str(row["sample_id"]): row for row in face_rows}
    if len(faces) != len(face_rows) or {str(row["sample_id"]) for row in at_rows} != set(faces):
        raise V34TrainingError("Audio/Text and Face predictions do not align by sample_id")
    return [(row, faces[str(row["sample_id"])]) for row in at_rows]


def _a4_rows(
    at_rows: Sequence[Mapping[str, Any]],
    face_rows: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for at, face in _align(at_rows, face_rows):
        row = copy.deepcopy(dict(at))
        row["candidate_id"] = "V34-A4"
        at_logit = at.get("raw_logit")
        face_logit = face.get("raw_logit")
        if at_logit is not None and face_logit is not None:
            row["raw_logit"] = float(at_logit) + float(alpha) * float(face["q_face"]) * math.tanh(
                float(face_logit)
            )
        row["face_logit"] = None if face_logit is None else float(face_logit)
        row["q_face"] = float(face["q_face"])
        result.append(row)
    return result


def _select_alpha(
    at_rows: Sequence[Mapping[str, Any]], face_rows: Sequence[Mapping[str, Any]]
) -> tuple[float, list[dict[str, float]]]:
    candidates = []
    for alpha in ALPHAS:
        rows = [row for row in _a4_rows(at_rows, face_rows, alpha=alpha) if row["raw_logit"] is not None]
        auc = float(_raw_metric_report(rows)["subject"]["roc_auc"])
        candidates.append({"alpha": alpha, "subject_auc": auc})
    candidates.sort(key=lambda row: (-row["subject_auc"], row["alpha"]))
    return float(candidates[0]["alpha"]), candidates


def _calibration_method_report(
    calibration_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    calibration = [dict(row) for row in calibration_rows if row.get("raw_logit") is not None]
    evaluation = [dict(row) for row in evaluation_rows if row.get("raw_logit") is not None]
    parameters = fit_calibrator(
        method,
        [float(row["raw_logit"]) for row in calibration],
        [int(row["label_hc_vs_non_hc"]) for row in calibration],
    )
    calibration_probabilities = apply_calibrator(
        method, [float(row["raw_logit"]) for row in calibration], parameters
    )
    evaluation_probabilities = apply_calibrator(
        method, [float(row["raw_logit"]) for row in evaluation], parameters
    )
    calibration_enriched = [
        {**row, "calibrated_probability": probability}
        for row, probability in zip(calibration, calibration_probabilities, strict=True)
    ]
    evaluation_enriched = [
        {**row, "calibration_method": method, "calibrated_probability": probability}
        for row, probability in zip(evaluation, evaluation_probabilities, strict=True)
    ]
    calibration_subjects = aggregate_subjects(
        calibration_enriched, probability_field="calibrated_probability"
    )
    subject_workpoint = select_workpoint(calibration_subjects)
    task_workpoint = select_workpoint(
        [
            {
                "label_hc_vs_non_hc": int(row["label_hc_vs_non_hc"]),
                "probability": float(row["calibrated_probability"]),
            }
            for row in calibration_enriched
        ]
    )
    evaluation_subjects = aggregate_subjects(
        evaluation_enriched, probability_field="calibrated_probability"
    )
    labels_task = [int(row["label_hc_vs_non_hc"]) for row in evaluation_enriched]
    probabilities_task = [float(row["calibrated_probability"]) for row in evaluation_enriched]
    labels_subject = [int(row["label_hc_vs_non_hc"]) for row in evaluation_subjects]
    probabilities_subject = [float(row["probability"]) for row in evaluation_subjects]
    report = {
        "method": method,
        "parameters": parameters,
        "fit_scope": "inner_calibration_task_raw_logits",
        "fit_task_count": len(calibration),
        "subject_probability_aggregation": "calibrate_each_task_then_mean_probability",
        "subject_workpoint": subject_workpoint,
        "task_workpoint_supplemental": task_workpoint,
        "outer_task_metrics_at_subject_workpoint": binary_metrics(
            labels_task,
            probabilities_task,
            threshold=float(subject_workpoint["threshold"]),
        ),
        "outer_task_metrics_at_task_workpoint": binary_metrics(
            labels_task,
            probabilities_task,
            threshold=float(task_workpoint["threshold"]),
        ),
        "outer_subject_metrics": binary_metrics(
            labels_subject,
            probabilities_subject,
            threshold=float(subject_workpoint["threshold"]),
        ),
    }
    return report, evaluation_enriched


def _select_calibration_type(folds: Sequence[Mapping[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    for method in METHODS:
        task = [
            fold["methods"][method]["outer_task_metrics_at_subject_workpoint"]
            for fold in folds
        ]
        rows.append(
            {
                "method": method,
                "outer_task_brier_mean": mean(float(value["brier"]) for value in task),
                "outer_task_ece_mean": mean(
                    float(value["ece_10_equal_width"]) for value in task
                ),
            }
        )
    best_brier = min(row["outer_task_brier_mean"] for row in rows)
    brier_contenders = [
        row for row in rows if row["outer_task_brier_mean"] - best_brier < 0.002
    ]
    best_ece = min(row["outer_task_ece_mean"] for row in brier_contenders)
    ece_contenders = [
        row for row in brier_contenders if row["outer_task_ece_mean"] - best_ece < 0.005
    ]
    ece_contenders.sort(key=lambda row: METHOD_PREFERENCE[row["method"]])
    return str(ece_contenders[0]["method"]), rows


def _pooled_subject_confusion(
    folds: Sequence[Mapping[str, Any]], selected_method: str
) -> dict[str, Any]:
    confusion = {key: 0 for key in ("tp", "fn", "tn", "fp")}
    for fold in folds:
        matrix = fold["methods"][selected_method]["outer_subject_metrics"]["confusion_matrix"]
        for key in confusion:
            confusion[key] += int(matrix[key])
    sensitivity = confusion["tp"] / (confusion["tp"] + confusion["fn"])
    specificity = confusion["tn"] / (confusion["tn"] + confusion["fp"])
    return {
        "confusion_matrix": confusion,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
    }


def run_opt_cog_004(
    *,
    feature_selection_path: Path,
    backbone_selection_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    feature_selection = json.loads(feature_selection_path.read_text(encoding="utf-8"))
    backbone_selection = json.loads(backbone_selection_path.read_text(encoding="utf-8"))
    if (
        feature_selection.get("status") != "passed"
        or feature_selection.get("official_test_evaluated") is not False
        or backbone_selection.get("status") != "passed"
    ):
        raise V34TrainingError("upstream V3.4 selections did not pass")
    split_sha = str(feature_selection["split_sha256"])
    at_candidate = str(feature_selection["selected_candidate_id"])
    at_run = WORKSPACE_ROOT / str(feature_selection["selected_run_path"])
    face_run = WORKSPACE_ROOT / str(backbone_selection["face_run"]["path"])
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_folds: dict[str, list[dict[str, Any]]] = {"audio_text": [], "audio_text_face": []}
    raw_auc_gains = []
    for outer_fold in range(5):
        at_dir = at_run / f"fold_{outer_fold}" / at_candidate
        face_dir = face_run / f"fold_{outer_fold}" / "V34-Face"
        verify_prediction_bundle(at_dir, expected_split_sha256=split_sha)
        verify_prediction_bundle(face_dir, expected_split_sha256=split_sha)
        at_roles = {role: _read_rows(at_dir, role) for role in PREDICTION_FILES}
        face_roles = {role: _read_rows(face_dir, role) for role in PREDICTION_FILES}
        alpha, alpha_candidates = _select_alpha(
            at_roles["inner_validation"], face_roles["inner_validation"]
        )
        a4_roles = {
            role: _a4_rows(at_roles[role], face_roles[role], alpha=alpha)
            for role in PREDICTION_FILES
        }
        a4_dir = output_root / f"fold_{outer_fold}" / "V34-A4"
        a4_bundle = {
            "schema_version": "cognitive_v34_a4_fold_bundle_v1",
            "candidate_id": "V34-A4",
            "outer_fold": outer_fold,
            "split_sha256": split_sha,
            "alpha_candidates": alpha_candidates,
            "selected_alpha": alpha,
            "audio_text_prediction_manifest_sha256": sha256_file(at_dir / "prediction_manifest.json"),
            "face_prediction_manifest_sha256": sha256_file(face_dir / "prediction_manifest.json"),
            "official_test_evaluated": False,
        }
        a4_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = a4_dir / "a4_bundle.json"
        write_json(bundle_path, a4_bundle)
        _write_prediction_bundle(
            candidate_dir=a4_dir,
            candidate_id="V34-A4",
            outer_fold=outer_fold,
            split_sha256=split_sha,
            checkpoint_path=bundle_path,
            input_hashes={
                "feature_selection": sha256_file(feature_selection_path),
                "backbone_selection": sha256_file(backbone_selection_path),
                "audio_text_prediction_manifest": sha256_file(at_dir / "prediction_manifest.json"),
                "face_prediction_manifest": sha256_file(face_dir / "prediction_manifest.json"),
            },
            prediction_rows=a4_roles,
        )
        at_auc = float(
            _raw_metric_report(
                [row for row in at_roles["outer_evaluation"] if row["raw_logit"] is not None]
            )["subject"]["roc_auc"]
        )
        a4_auc = float(
            _raw_metric_report(
                [row for row in a4_roles["outer_evaluation"] if row["raw_logit"] is not None]
            )["subject"]["roc_auc"]
        )
        raw_auc_gains.append(a4_auc - at_auc)
        for key, roles in (("audio_text", at_roles), ("audio_text_face", a4_roles)):
            methods: dict[str, Any] = {}
            prediction_rows: list[dict[str, Any]] = []
            for method in METHODS:
                method_report, method_predictions = _calibration_method_report(
                    roles["inner_calibration"],
                    roles["outer_evaluation"],
                    method=method,
                )
                methods[method] = method_report
                prediction_rows.extend(method_predictions)
            fold_report = {
                "outer_fold": outer_fold,
                "candidate": key,
                "selected_alpha": alpha if key == "audio_text_face" else None,
                "methods": methods,
            }
            candidate_folds[key].append(fold_report)
            candidate_dir = output_root / f"fold_{outer_fold}" / key
            candidate_dir.mkdir(parents=True, exist_ok=True)
            write_json(candidate_dir / "calibration_report.json", fold_report)
            write_jsonl(candidate_dir / "outer_calibrated_predictions.jsonl", prediction_rows)
    candidates: dict[str, Any] = {}
    for key, folds in candidate_folds.items():
        method, comparison = _select_calibration_type(folds)
        candidates[key] = {
            "selected_calibration_method": method,
            "calibration_method_comparison": comparison,
            "pooled_subject_workpoint": _pooled_subject_confusion(folds, method),
            "folds": folds,
        }
    at_pooled = candidates["audio_text"]["pooled_subject_workpoint"]
    a4_pooled = candidates["audio_text_face"]["pooled_subject_workpoint"]
    face_checks = {
        "positive_auc_gain_fold_count": sum(value > 0.0 for value in raw_auc_gains),
        "minimum_positive_auc_folds": 4,
        "mean_raw_subject_auc_gain": mean(raw_auc_gains),
        "audio_text_sensitivity": at_pooled["sensitivity"],
        "audio_text_face_sensitivity": a4_pooled["sensitivity"],
        "minimum_pooled_sensitivity": 0.80,
        "audio_text_specificity": at_pooled["specificity"],
        "audio_text_face_specificity": a4_pooled["specificity"],
    }
    use_face = bool(
        face_checks["positive_auc_gain_fold_count"] >= 4
        and face_checks["mean_raw_subject_auc_gain"] > 0.0
        and face_checks["audio_text_sensitivity"] >= 0.80
        and face_checks["audio_text_face_sensitivity"] >= 0.80
        and face_checks["audio_text_face_specificity"]
        >= face_checks["audio_text_specificity"]
    )
    final_key = "audio_text_face" if use_face else "audio_text"
    report = {
        "schema_version": "cognitive_v34_opt_cog_004_report_v1",
        "task_id": "OPT-COG-004",
        "status": "passed",
        "official_test_evaluated": False,
        "split_sha256": split_sha,
        "feature_selection_path": workspace_relative(feature_selection_path),
        "feature_selection_sha256": sha256_file(feature_selection_path),
        "backbone_selection_path": workspace_relative(backbone_selection_path),
        "backbone_selection_sha256": sha256_file(backbone_selection_path),
        "audio_text_candidate_id": at_candidate,
        "audio_text_run_path": workspace_relative(at_run),
        "face_run_path": workspace_relative(face_run),
        "raw_subject_auc_gain_by_fold": raw_auc_gains,
        "candidates": candidates,
        "face_decision_checks": face_checks,
        "face_in_primary_score": use_face,
        "final_candidate": final_key,
        "final_calibration_method": candidates[final_key]["selected_calibration_method"],
        "product_level_boundaries": {
            "normal_upper_exclusive": 40.0,
            "attention_upper_exclusive": 70.0,
            "high_attention_lower_inclusive": 70.0,
            "changed_by_opt_cog_004": False,
        },
        "limitations": [
            "Calibration and workpoints are CogPic official-Train development evidence.",
            "The subject workpoint is not an API product-level boundary.",
            "No official Test prediction was generated.",
        ],
    }
    output_path = output_root / "OPT-COG-004_report.json"
    write_json(output_path, report)
    write_json(
        REPORT_ROOT / "cognitive_v34_candidate_config.json",
        {
            "schema_version": "cognitive_v34_candidate_config_v1",
            "model_version": "cognitive-mm-v3.4.0",
            "final_candidate": final_key,
            "audio_text_candidate_id": at_candidate,
            "feature_variant": feature_selection["selected_variant"],
            "face_in_primary_score": use_face,
            "calibration_method": report["final_calibration_method"],
            "source_report": workspace_relative(output_path),
            "source_report_sha256": sha256_file(output_path),
            "official_test_evaluated": False,
        },
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-selection",
        type=Path,
        default=REPORT_ROOT / "OPT-COG-003_selection_report.json",
    )
    parser.add_argument(
        "--backbone-selection",
        type=Path,
        default=REPORT_ROOT / "OPT-COG-002_selection_report.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPORT_ROOT / "opt_cog_004",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_opt_cog_004(
        feature_selection_path=args.feature_selection,
        backbone_selection_path=args.backbone_selection,
        output_root=args.output_root,
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_opt_cog_004"]
