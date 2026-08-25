from __future__ import annotations

import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    load_camera_inputs,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_boundary import (
    load_camera_episode_boundary_development_profile,
    propose_camera_episode_boundaries,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_boundary_evaluation import (
    evaluate_camera_episode_boundaries,
    load_camera_episode_boundary_evaluation_config,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    load_camera_config,
)


ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune home-development motion-bout hysteresis against temporal labels."
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--labeled-import-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.project_root.resolve(strict=True)
    tracking_root = args.tracking_root.resolve(strict=True)
    labeled_root = args.labeled_import_root.resolve(strict=True)
    output = args.output_json.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"tuning output already exists: {output}")
    camera_config = load_camera_config(
        root / "configs/modules/wandering_camera_v1.yaml"
    )
    base_profile = load_camera_episode_boundary_development_profile(
        root / "configs/modules/wandering_camera_episode_boundary_home_v1.yaml"
    )
    evaluation_config = load_camera_episode_boundary_evaluation_config(
        root / "configs/modules/wandering_camera_episode_boundary_eval_v1.yaml"
    )
    adapters = {}
    boundaries = []
    for task_dir in sorted((labeled_root / "tasks").iterdir()):
        if not task_dir.is_dir():
            continue
        video_id = task_dir.name
        adapter = load_camera_inputs(
            tracking_root / video_id / "inputs/tracking.jsonl",
            task_dir / "media_sidecar.labeled.json",
            camera_config,
        )
        adapters[video_id] = adapter
        media = adapter.media_sidecar
        for boundary in _load_jsonl(task_dir / "episode_boundaries.jsonl"):
            boundaries.append(
                {
                    **boundary,
                    "duration_sec": (
                        float(boundary["end_sec_exclusive"])
                        - float(boundary["start_sec"])
                    ),
                    "source_group_id": media["source_group_id"],
                    "device_id": media["device_id"],
                    "setup_id": media["setup_id"],
                    "stream_epoch": media["stream_epoch"],
                    "track_id": boundary["target_track_id"],
                    "participant_id": "home-development-participant",
                    "session_id": "home-repair-20260822",
                    "camera_setup_id": media["setup_id"],
                    "clock_domain_id": "home-video-relative-time",
                }
            )

    results = []
    for dwell, stationary, movement in itertools.product(
        (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
        (0.010, 0.015, 0.020, 0.025, 0.030),
        (0.040, 0.050, 0.060, 0.080),
    ):
        profile = deepcopy(base_profile)
        profile["state_machine"]["stationary_dwell_candidate_seconds"] = dwell
        profile["state_machine"][
            "stationary_max_displacement_body_heights"
        ] = stationary
        profile["state_machine"][
            "movement_start_min_displacement_body_heights"
        ] = movement
        proposals = []
        for adapter in adapters.values():
            rows, _diagnostics = propose_camera_episode_boundaries(
                adapter,
                proposal_config=profile,
                camera_config=camera_config,
            )
            for row in rows:
                if row["proposal_status"] == "rejected_by_qc":
                    continue
                proposals.append(
                    {
                        **row,
                        "participant_id": "home-development-participant",
                        "session_id": "home-repair-20260822",
                        "camera_setup_id": adapter.media_sidecar["setup_id"],
                        "clock_domain_id": "home-video-relative-time",
                    }
                )
        artifacts = evaluate_camera_episode_boundaries(
            proposals,
            boundaries,
            config=evaluation_config,
        )
        detection = artifacts["metrics"]["views"][
            "all_locomotion_candidates"
        ]["detection"]
        localization = artifacts["metrics"]["views"][
            "all_locomotion_candidates"
        ]["localization"]["temporal_iou"]
        results.append(
            {
                "stationary_dwell_candidate_seconds": dwell,
                "stationary_max_displacement_body_heights": stationary,
                "movement_start_min_displacement_body_heights": movement,
                "candidate_support": detection["candidate_support"],
                "matched_count": detection["matched_count"],
                "precision": detection["precision"],
                "recall": detection["recall"],
                "f1": detection["f1"],
                "matched_temporal_iou_median": localization["median"],
            }
        )
    ranked = sorted(
        results,
        key=lambda row: (
            float(row["f1"]),
            float(row["recall"]),
            float(row["precision"]),
            float(row["matched_temporal_iou_median"]),
        ),
        reverse=True,
    )
    payload = {
        "schema_version": "wandering-camera-home-boundary-tuning-v1",
        "validation_scope": "same_participant_home_development",
        "truth_role": "temporal_behavior_labels_only",
        "tracking_truth_consumed": False,
        "candidate_count": len(results),
        "best": ranked[0],
        "target_recall_085_candidates": sum(
            float(row["recall"]) >= 0.85 for row in ranked
        ),
        "top_20": ranked[:20],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))
    print(json.dumps(payload["best"], sort_keys=True))
    return 0


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
