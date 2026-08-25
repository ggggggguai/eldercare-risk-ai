from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence

from sklearn.metrics import f1_score, precision_score, recall_score

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_development import (
    EPISODE_SCOPE_FIELDS,
    match_episodes_one_to_one,
)


ROOT = Path(__file__).resolve().parents[2]
CLASS_ORDER = ("direct", "pacing", "lapping", "random")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Join final automatic-proposal classifications to temporal/behavior "
            "development truth after inference."
        )
    )
    parser.add_argument("--prediction-jsonl", type=Path, required=True)
    parser.add_argument("--resolved-jsonl", type=Path)
    parser.add_argument("--boundary-eval-root", type=Path, required=True)
    parser.add_argument("--labeled-import-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prediction_path = args.prediction_jsonl.resolve(strict=True)
    boundary_root = args.boundary_eval_root.resolve(strict=True)
    labeled_root = args.labeled_import_root.resolve(strict=True)
    output = args.output_json.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"classification evaluation output exists: {output}")
    predictions = {
        str(row["proposal_id"]): row for row in _load_jsonl(prediction_path)
    }
    parent_match_by_episode = {}
    for row in _load_jsonl(boundary_root / "matches.jsonl"):
        if row["view"] == "all_locomotion_candidates":
            parent_match_by_episode[str(row["episode_id"])] = str(row["proposal_id"])
    truths = {}
    bound_truths = {}
    for task_dir in sorted((labeled_root / "tasks").iterdir()):
        if not task_dir.is_dir():
            continue
        media = json.loads(
            (task_dir / "media_sidecar.labeled.json").read_text(encoding="utf-8")
        )
        boundaries = {
            str(row["episode_id"]): row
            for row in _load_jsonl(task_dir / "episode_boundaries.jsonl")
        }
        for row in _load_jsonl(task_dir / "episode_truth.jsonl"):
            label = str(row["observable_pattern"])
            if label in CLASS_ORDER:
                episode_id = str(row["episode_id"])
                truths[episode_id] = label
                boundary = boundaries[episode_id]
                bound_truths[episode_id] = {
                    "annotation_id": episode_id,
                    "source_group_id": media["source_group_id"],
                    "source_video_id": media["source_video_id"],
                    "device_id": media["device_id"],
                    "setup_id": media["setup_id"],
                    "stream_epoch": media["stream_epoch"],
                    "track_id": boundary["target_track_id"],
                    "participant_id": "home-development-participant",
                    "session_id": "home-repair-20260822",
                    "camera_setup_id": media["setup_id"],
                    "clock_domain_id": "home-video-relative-time",
                    "start_sec": boundary["start_sec"],
                    "end_sec_exclusive": boundary["end_sec_exclusive"],
                }

    resolved_predictions = None
    final_match_by_episode = parent_match_by_episode
    prediction_id_field = "proposal_id"
    boundary_detection = None
    if args.resolved_jsonl is not None:
        resolved_predictions = {
            str(row["resolved_episode_id"]): row
            for row in _load_jsonl(args.resolved_jsonl.resolve(strict=True))
        }
        prediction_rows = [
            {
                "prediction_id": prediction_id,
                **{field: row[field] for field in EPISODE_SCOPE_FIELDS},
                "start_sec": row["start_sec"],
                "end_sec_exclusive": row["end_sec_exclusive"],
            }
            for prediction_id, row in resolved_predictions.items()
        ]
        matched = match_episodes_one_to_one(
            prediction_rows,
            list(bound_truths.values()),
            matching_policy={
                "policy_id": "m0cam-ep2a-s1b-b01-development-frozen-v1",
                "minimum_temporal_iou": 0.25,
                "maximum_onset_delta_sec": 10.0,
            },
        )
        final_match_by_episode = {
            str(row["annotation_id"]): str(row["prediction_id"])
            for row in matched["matches"]
        }
        prediction_id_field = "resolved_episode_id"
        match_count = len(matched["matches"])
        prediction_count = len(resolved_predictions)
        truth_count = len(bound_truths)
        precision = match_count / prediction_count if prediction_count else 0.0
        recall = match_count / truth_count if truth_count else 0.0
        boundary_detection = {
            "truth_support": truth_count,
            "candidate_support": prediction_count,
            "matched_count": match_count,
            "precision": precision,
            "recall": recall,
            "f1": (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
        }

    final_predicted = []
    base_predicted = []
    truth_labels = []
    result_rows = []
    for episode_id, truth in sorted(truths.items()):
        proposal_id = parent_match_by_episode.get(episode_id)
        base_prediction = predictions.get(proposal_id) if proposal_id is not None else None
        final_prediction_id = final_match_by_episode.get(episode_id)
        if resolved_predictions is not None:
            prediction = resolved_predictions.get(final_prediction_id)
        else:
            prediction = predictions.get(final_prediction_id)
        final_label = (
            str(prediction["predicted_pattern"])
            if prediction is not None and prediction["prediction_status"] == "ready"
            else "pipeline_miss"
        )
        base_label = (
            str(base_prediction["base_model_four_class"]["predicted_label"])
            if base_prediction is not None
            and base_prediction["prediction_status"] == "ready"
            and base_prediction["base_model_four_class"] is not None
            else "pipeline_miss"
        )
        truth_labels.append(truth)
        final_predicted.append(final_label)
        base_predicted.append(base_label)
        result_rows.append(
            {
                "episode_id": episode_id,
                "proposal_id": proposal_id,
                prediction_id_field: final_prediction_id,
                "truth": truth,
                "base_prediction": base_label,
                "final_prediction": final_label,
                "pipeline_miss": final_label == "pipeline_miss",
            }
        )
    report = {
        "schema_version": "wandering-camera-home-repair-classification-eval-v1",
        "validation_scope": "same_participant_home_development",
        "evaluation_population": "all_known_behavior_truth_with_pipeline_miss",
        "truth_role": "temporal_behavior_labels_only",
        "cvat_rectangle_used_as_detection_or_tracking_truth": False,
        "evaluation_role_used_as_detection_or_tracking_truth": False,
        "truth_consumed_at_inference": False,
        "truth_count": len(truth_labels),
        "matched_prediction_count": sum(
            value != "pipeline_miss" for value in final_predicted
        ),
        "pipeline_miss_count": sum(
            value == "pipeline_miss" for value in final_predicted
        ),
        "resolved_boundary_detection": boundary_detection,
        "base_model": _metrics(truth_labels, base_predicted),
        "final_geometry_policy": _metrics(truth_labels, final_predicted),
        "results": result_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report))
    print(json.dumps(report["final_geometry_policy"], sort_keys=True))
    return 0


def _metrics(truth: Sequence[str], predicted: Sequence[str]) -> dict[str, Any]:
    precision = precision_score(
        truth,
        predicted,
        labels=list(CLASS_ORDER),
        average=None,
        zero_division=0,
    )
    recall = recall_score(
        truth,
        predicted,
        labels=list(CLASS_ORDER),
        average=None,
        zero_division=0,
    )
    f1 = f1_score(
        truth,
        predicted,
        labels=list(CLASS_ORDER),
        average=None,
        zero_division=0,
    )
    confusion = {
        true_name: {
            predicted_name: sum(
                left == true_name and right == predicted_name
                for left, right in zip(truth, predicted, strict=True)
            )
            for predicted_name in (*CLASS_ORDER, "pipeline_miss")
        }
        for true_name in CLASS_ORDER
    }
    return {
        "accuracy_with_pipeline_miss": sum(
            left == right for left, right in zip(truth, predicted, strict=True)
        )
        / len(truth),
        "macro_f1_with_pipeline_miss": f1_score(
            truth,
            predicted,
            labels=list(CLASS_ORDER),
            average="macro",
            zero_division=0,
        ),
        "support_by_class": dict(Counter(truth)),
        "precision_by_class": {
            name: float(value)
            for name, value in zip(CLASS_ORDER, precision, strict=True)
        },
        "recall_by_class": {
            name: float(value)
            for name, value in zip(CLASS_ORDER, recall, strict=True)
        },
        "f1_by_class": {
            name: float(value)
            for name, value in zip(CLASS_ORDER, f1, strict=True)
        },
        "confusion": confusion,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
