from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from elderly_monitoring.modules.fall_risk.near_fall_self_collected import (
    build_scf_challenge_windows,
)
from elderly_monitoring.modules.fall_risk.near_fall_tcn import NearFallTCNPredictor
from elderly_monitoring.modules.fall_risk.near_fall_training import NearFallDatasetConfig


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def evaluate(
    *,
    candidate_dir: Path,
    pose_dir: Path,
    runs_dir: Path,
    baseline_dir: Path,
    output_path: Path,
    threshold: float = 0.5,
) -> dict[str, Any]:
    tensors, samples, rejected = build_scf_challenge_windows(
        candidates_path=candidate_dir / "near_fall_candidates.jsonl",
        manifest_path=candidate_dir / "manifest.jsonl",
        pose_dir=pose_dir,
        experiment="E3",
    )
    base_f1: dict[int, float] = {}
    for path in sorted(baseline_dir.glob("development-splitv3-e71a045-trackmap-v2-seed*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        base_f1[int(payload["seed"])] = float(payload["validation"]["event"]["f1"])
    experiments: dict[str, Any] = {}
    run_locations = {
        "E0": {
            seed: baseline_dir / f"development-splitv3-e71a045-trackmap-v2-seed{seed}"
            for seed in (42, 43, 44)
        },
        **{
            experiment: {
                seed: runs_dir / f"{experiment.lower()}-seed{seed}"
                for seed in (42, 43, 44)
            }
            for experiment in ("E1", "E2", "E3")
        },
    }
    for experiment in ("E0", "E1", "E2", "E3"):
        seed_results = []
        for seed in (42, 43, 44):
            run = run_locations[experiment][seed]
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            predictor = NearFallTCNPredictor(run / "best_model.pt", device="cpu")
            by_event: dict[str, list[float]] = defaultdict(list)
            event_sample: dict[str, Mapping[str, Any]] = {}
            unavailable = 0
            for tensor, sample in zip(tensors, samples, strict=True):
                prediction = predictor.predict_tensor(tensor)
                if prediction["status"] != "valid":
                    unavailable += 1
                    continue
                event_id = str(sample["event_id"])
                by_event[event_id].append(float(prediction["near_fall_event_score"]))
                event_sample[event_id] = sample
            strata: dict[str, Any] = {}
            grouped: dict[str, list[bool]] = defaultdict(list)
            for event_id, scores in by_event.items():
                sample = event_sample[event_id]
                grouped[str(sample["source_action_id"])].append(max(scores) >= threshold)
            for action, values in sorted(grouped.items()):
                positives = sum(values)
                strata[action] = {
                    "events": len(values),
                    "triggered": positives,
                    "trigger_rate": positives / len(values),
                    "wilson_95": _wilson(positives, len(values)),
                }
            negative_values = [
                value
                for action, values in grouped.items()
                if action.startswith("A")
                for value in values
            ]
            positive_values = [
                value
                for action, values in grouped.items()
                if action.startswith("C")
                for value in values
            ]
            negative_triggered = sum(negative_values)
            positive_detected = sum(positive_values)
            validation_f1 = float(metrics["validation"]["event"]["f1"])
            seed_results.append(
                {
                    "seed": seed,
                    "validation_event_f1": validation_f1,
                    "validation_event_f1_delta_vs_e0": validation_f1 - base_f1[seed],
                    "validation_regression_gate_passed": validation_f1 >= base_f1[seed] - 0.01,
                    "challenge_by_action": strata,
                    "challenge_negative_overall": {
                        "events": len(negative_values),
                        "triggered": negative_triggered,
                        "trigger_rate": negative_triggered / len(negative_values),
                        "wilson_95": _wilson(negative_triggered, len(negative_values)),
                    },
                    "challenge_positive_proxy_overall": {
                        "events": len(positive_values),
                        "detected": positive_detected,
                        "detection_rate": positive_detected / len(positive_values),
                        "wilson_95": _wilson(positive_detected, len(positive_values)),
                    },
                    "unavailable_window_count": unavailable,
                    "test_evaluated": False,
                }
            )
        experiments[experiment] = {"seeds": seed_results}
    for experiment in ("E1", "E2", "E3"):
        for result, baseline in zip(
            experiments[experiment]["seeds"], experiments["E0"]["seeds"], strict=True
        ):
            negative_delta = (
                result["challenge_negative_overall"]["trigger_rate"]
                - baseline["challenge_negative_overall"]["trigger_rate"]
            )
            positive_delta = (
                result["challenge_positive_proxy_overall"]["detection_rate"]
                - baseline["challenge_positive_proxy_overall"]["detection_rate"]
            )
            result["challenge_negative_trigger_rate_delta_vs_e0"] = negative_delta
            result["challenge_positive_proxy_detection_rate_delta_vs_e0"] = positive_delta
        deltas = [
            row["challenge_negative_trigger_rate_delta_vs_e0"]
            for row in experiments[experiment]["seeds"]
        ]
        experiments[experiment]["negative_improvement_direction_consistent"] = all(
            value < 0 for value in deltas
        )
    report = {
        "schema_version": "self-collected-near-fall-augmentation-evaluation-v1",
        "status": "development_provisional",
        "threshold": threshold,
        "challenge_subjects": ["P05"],
        "challenge_window_count": len(samples),
        "challenge_event_count": len({row["event_id"] for row in samples}),
        "challenge_rejected_window_counts": rejected,
        "metric_semantics": {
            "A02_A03_A05_A06_A08_A10": "undesired near-fall trigger rate; lower is better",
            "C03_C04_C05": "coarse reviewed-action-interval-end-proxy detection rate; higher is better",
        },
        "experiments": experiments,
        "fp_hour_reported": False,
        "test_pose_read": False,
        "test_evaluated": False,
        "main_path_unchanged": True,
        "limitations": [
            "P05 is a self-collected development challenge partition, not a formal validation or test set",
            "C03/C04/C05 anchors are reviewed action interval end proxies, not exact recovery timing",
            "short action-centred clips do not support FP/hour",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def evaluate_single_checkpoint(
    *,
    candidate_dir: Path,
    pose_dir: Path,
    checkpoint_path: Path,
    validation_predictions_path: Path,
    output_path: Path,
    baseline_evaluation_path: Path | None = None,
    seed: int = 42,
    min_validation_recall: float = 0.90,
) -> dict[str, Any]:
    validation_rows = [
        json.loads(line)
        for line in validation_predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validation_by_event: dict[str, dict[str, Any]] = {}
    for row in validation_rows:
        event = validation_by_event.setdefault(
            str(row["event_id"]), {"label": int(row["label"]), "scores": []}
        )
        if int(event["label"]) != int(row["label"]):
            raise ValueError(f"inconsistent validation event label: {row['event_id']}")
        event["scores"].append(float(row["probability"]))
    labels = np.asarray(
        [int(row["label"]) for row in validation_by_event.values()], dtype=np.int64
    )
    scores = np.asarray(
        [max(row["scores"]) for row in validation_by_event.values()], dtype=np.float64
    )
    candidates = np.unique(np.concatenate(([0.001, 0.5, 0.999], scores)))
    threshold_rows: list[dict[str, float]] = []
    for threshold in candidates:
        predictions = scores >= threshold
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "f1": float(f1_score(labels, predictions)),
                "precision": float(
                    precision_score(labels, predictions, zero_division=0)
                ),
                "recall": float(recall_score(labels, predictions, zero_division=0)),
            }
        )
    eligible = [
        row
        for row in threshold_rows
        if row["recall"] >= min_validation_recall - 1e-12
    ]
    if not eligible:
        raise ValueError("no validation threshold satisfies the recall constraint")
    calibrated = max(
        eligible,
        key=lambda row: (row["precision"], row["f1"], row["threshold"]),
    )

    tensors, samples, rejected = build_scf_challenge_windows(
        candidates_path=candidate_dir / "near_fall_candidates.jsonl",
        manifest_path=candidate_dir / "manifest.jsonl",
        pose_dir=pose_dir,
        experiment="E3",
        config=NearFallDatasetConfig(fallback_window_secs=(2.0,)),
    )
    predictor = NearFallTCNPredictor(checkpoint_path, device="cpu")
    challenge_scores: dict[str, list[float]] = defaultdict(list)
    event_sample: dict[str, Mapping[str, Any]] = {}
    unavailable = 0
    for tensor, sample in zip(tensors, samples, strict=True):
        prediction = predictor.predict_tensor(tensor)
        if prediction["status"] != "valid":
            unavailable += 1
            continue
        event_id = str(sample["event_id"])
        challenge_scores[event_id].append(
            float(prediction["near_fall_event_score"])
        )
        event_sample[event_id] = sample

    def summarize(threshold: float) -> dict[str, Any]:
        by_action: dict[str, list[bool]] = defaultdict(list)
        for event_id, event_scores in challenge_scores.items():
            action = str(event_sample[event_id]["source_action_id"])
            by_action[action].append(max(event_scores) >= threshold)
        strata = {
            action: {
                "events": len(values),
                "triggered": int(sum(values)),
                "trigger_rate": float(sum(values) / len(values)),
                "wilson_95": _wilson(int(sum(values)), len(values)),
            }
            for action, values in sorted(by_action.items())
        }
        negatives = [
            value
            for action, values in by_action.items()
            if action.startswith("A")
            for value in values
        ]
        positives = [
            value
            for action, values in by_action.items()
            if action.startswith("C")
            for value in values
        ]
        return {
            "threshold": threshold,
            "by_action": strata,
            "negative_overall": {
                "events": len(negatives),
                "triggered": int(sum(negatives)),
                "trigger_rate": float(sum(negatives) / len(negatives)),
                "wilson_95": _wilson(int(sum(negatives)), len(negatives)),
            },
            "positive_proxy_overall": {
                "events": len(positives),
                "detected": int(sum(positives)),
                "detection_rate": float(sum(positives) / len(positives)),
                "wilson_95": _wilson(int(sum(positives)), len(positives)),
            },
        }

    fixed = summarize(0.5)
    calibrated_challenge = summarize(float(calibrated["threshold"]))
    baseline = None
    if baseline_evaluation_path is not None:
        baseline_payload = json.loads(
            baseline_evaluation_path.read_text(encoding="utf-8")
        )
        baseline = next(
            row
            for row in baseline_payload["experiments"]["E0"]["seeds"]
            if int(row["seed"]) == seed
        )
        fixed["delta_vs_e0"] = {
            "negative_trigger_rate": fixed["negative_overall"]["trigger_rate"]
            - float(baseline["challenge_negative_overall"]["trigger_rate"]),
            "positive_proxy_detection_rate": fixed["positive_proxy_overall"][
                "detection_rate"
            ]
            - float(
                baseline["challenge_positive_proxy_overall"]["detection_rate"]
            ),
        }

    report = {
        "schema_version": "near-fall-scf-challenge-evaluation-v2",
        "status": "development_provisional",
        "checkpoint": checkpoint_path.as_posix(),
        "seed": seed,
        "validation_calibration": {
            "selection_partition": "validation",
            "event_count": len(validation_by_event),
            "minimum_recall": min_validation_recall,
            "selected": calibrated,
        },
        "challenge_subjects": ["P05"],
        "challenge_window_count": len(samples),
        "challenge_event_count": len(challenge_scores),
        "challenge_rejected_window_counts": rejected,
        "unavailable_window_count": unavailable,
        "fixed_threshold": fixed,
        "validation_calibrated_threshold": calibrated_challenge,
        "baseline_e0_seed": baseline,
        "test_pose_read": False,
        "test_evaluated": False,
        "main_path_unchanged": True,
        "decision": "no_go_main_path_replacement",
        "limitations": [
            "P05 is a development challenge subject, not the frozen test partition",
            "C03/C04/C05 are coarse positive proxies rather than exact recovery truth",
            "short action-centred clips do not support FP/hour",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate E1-E3 on locked base validation and P05 challenge.")
    parser.add_argument(
        "--candidate-dir", type=Path,
        default=Path("data/annotations/fall_risk/generated/v2/self_collected_scf_mvp_v1_candidate"),
    )
    parser.add_argument(
        "--pose-dir", type=Path,
        default=Path("reports/fall_risk/self_collected_scf_mvp_v1/baseline_replay/cleaned_pose"),
    )
    parser.add_argument(
        "--runs-dir", type=Path,
        default=Path("reports/fall_risk/self_collected_scf_mvp_v1/near_fall_augmentation"),
    )
    parser.add_argument(
        "--baseline-dir", type=Path,
        default=Path("reports/fall_risk/near_fall_event_v1"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single-checkpoint", type=Path, default=None)
    parser.add_argument("--validation-predictions", type=Path, default=None)
    parser.add_argument("--baseline-evaluation", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-validation-recall", type=float, default=0.90)
    args = parser.parse_args(argv)
    if args.single_checkpoint is not None:
        if args.validation_predictions is None:
            parser.error("--single-checkpoint requires --validation-predictions")
        report = evaluate_single_checkpoint(
            candidate_dir=args.candidate_dir,
            pose_dir=args.pose_dir,
            checkpoint_path=args.single_checkpoint,
            validation_predictions_path=args.validation_predictions,
            output_path=args.output,
            baseline_evaluation_path=args.baseline_evaluation,
            seed=args.seed,
            min_validation_recall=args.min_validation_recall,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    report = evaluate(
        candidate_dir=args.candidate_dir,
        pose_dir=args.pose_dir,
        runs_dir=args.runs_dir,
        baseline_dir=args.baseline_dir,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
