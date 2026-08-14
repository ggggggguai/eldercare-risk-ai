from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from elderly_monitoring.modules.fall_risk.near_fall_self_collected import (
    build_scf_challenge_windows,
)
from elderly_monitoring.modules.fall_risk.near_fall_tcn import NearFallTCNPredictor


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
    args = parser.parse_args(argv)
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
