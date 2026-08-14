from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from elderly_monitoring.modules.fall_risk.gait_tcn import GaitTCNPredictor
from elderly_monitoring.modules.fall_risk.self_collected_scf import (
    replay_fall_rule_video,
    replay_gait_video,
    replay_sit_stand_video,
)
from elderly_monitoring.modules.fall_risk.sit_stand_continuous import (
    SitStandContinuousConfig,
)
from elderly_monitoring.modules.fall_risk.sit_stand_continuous_inference import (
    load_continuous_sit_stand_tcn,
)
from elderly_monitoring.modules.fall_risk.sit_stand_continuous_tcn import (
    SitStandStreamDecoderConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay SCF_MVP_V1 G1 gait, sit-stand and fall-rule baselines from cached pose."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--action-labels", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--gait-checkpoint", type=Path, required=True)
    parser.add_argument("--gait-metrics", type=Path, required=True)
    parser.add_argument("--sit-stand-checkpoint", type=Path, required=True)
    parser.add_argument("--sit-stand-training-config", type=Path, required=True)
    parser.add_argument("--sit-stand-evaluation-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output_dir.exists():
            raise FileExistsError(f"output exists: {args.output_dir}")
        if args.batch_size < 1:
            raise ValueError("batch-size must be positive")
        manifest = _read_jsonl(args.manifest)
        actions = _read_jsonl(args.action_labels)
        planned_actions = _planned_actions(actions)
        pose_paths = {
            str(row["video_id"]): args.pose_dir / f"{row['video_id']}.jsonl"
            for row in manifest
        }
        missing = [video_id for video_id, path in pose_paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"cached pose is missing: {missing[:5]}")
        gait_predictor = GaitTCNPredictor(
            args.gait_checkpoint,
            device=args.device,
            expected_task="gait_instability_vs_normal_activity",
        )
        gait_metrics = _read_json(args.gait_metrics)
        gait_threshold = float(gait_metrics["selected_threshold"])
        sit_model, sit_device, sit_sha = load_continuous_sit_stand_tcn(
            args.sit_stand_checkpoint, device=args.device
        )
        continuous_config, decoder_config = _sit_configs(
            args.sit_stand_training_config, args.sit_stand_evaluation_config
        )
        contract = {
            "schema_version": "self-collected-g1-baseline-replay-v1",
            "manifest_sha256": _sha256(args.manifest),
            "action_labels_sha256": _sha256(args.action_labels),
            "pose_dir": args.pose_dir.as_posix(),
            "gait_checkpoint": _file_contract(args.gait_checkpoint),
            "gait_metrics": _file_contract(args.gait_metrics),
            "gait_threshold": gait_threshold,
            "sit_stand_checkpoint": _file_contract(args.sit_stand_checkpoint),
            "sit_stand_checkpoint_sha256": sit_sha,
            "device": args.device,
            "short_clip_replay": True,
            "fp_hour_reported": False,
            "test_pose_read": False,
            "test_evaluated": False,
            "main_path_unchanged": True,
        }
        args.output_dir.mkdir(parents=True)
        _write_json(args.output_dir / "contract.json", contract)
        results: list[dict[str, Any]] = []
        for index, row in enumerate(sorted(manifest, key=lambda item: str(item["video_id"])), 1):
            video_id = str(row["video_id"])
            records = _read_jsonl(pose_paths[video_id])
            result = {
                "video_id": video_id,
                "file_name": row["file_name"],
                "provisional_subject": row.get("provisional_subject"),
                "planned_action": planned_actions.get(video_id),
                "pose_record_count": len(records),
                "source_diagnostics": _source_diagnostics(row, records),
                "gait": replay_gait_video(
                    records=records,
                    model_predictor=gait_predictor,
                    model_threshold=gait_threshold,
                ),
                "sit_stand": replay_sit_stand_video(
                    records=records,
                    video_id=video_id,
                    model=sit_model,
                    device=sit_device,
                    checkpoint_sha256=sit_sha,
                    continuous_config=continuous_config,
                    decoder_config=decoder_config,
                    batch_size=args.batch_size,
                ),
                "fall_rule": replay_fall_rule_video(records=records),
            }
            _write_json(args.output_dir / "videos" / f"{video_id}.json", result)
            results.append(_result_row(result))
            print(f"[{index}/{len(manifest)}] completed {video_id}", flush=True)
        summary = _summary(results, contract)
        _write_json(args.output_dir / "summary.json", summary)
        print(json.dumps({key: summary[key] for key in ("video_count", "gait", "sit_stand", "fall_rule")}, sort_keys=True))
        return 0
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _result_row(result: Mapping[str, Any]) -> dict[str, Any]:
    gait = result["gait"]
    sit = result["sit_stand"]
    fall = result["fall_rule"]
    return {
        "video_id": result["video_id"],
        "file_name": result["file_name"],
        "provisional_subject": result.get("provisional_subject"),
        "planned_action": result.get("planned_action"),
        "pose_record_count": result["pose_record_count"],
        "source_diagnostics": result["source_diagnostics"],
        "gait_rule_video_positive": gait["rule"]["video_positive"],
        "gait_tcn_video_positive": gait["tcn"]["video_positive"],
        "gait_rule_window_count": gait["rule"]["window_count"],
        "gait_tcn_window_count": gait["tcn"]["window_count"],
        "sit_stand_rule_event_count": len(sit["rule"]["predictions"]),
        "sit_stand_tcn_event_count": len(sit["tcn"]["predictions"]),
        "sit_stand_tcn_valid_window_count": sit["tcn"]["diagnostics"]["valid_window_count"],
        "fall_rule_trigger_count": fall["trigger_count"],
        "fall_rule_video_positive": fall["video_positive"],
    }


def _summary(rows: list[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    def counts(field: str) -> dict[str, int]:
        return {
            "video_count": len(rows),
            "positive_video_count": sum(bool(row[field]) for row in rows),
        }

    strata: defaultdict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"video_count": 0, "gait_tcn_positive": 0, "sit_stand_tcn_events": 0, "fall_rule_triggers": 0}
    )
    for row in rows:
        key = (str(row.get("provisional_subject")), str(row.get("planned_action")))
        item = strata[key]
        item["video_count"] += 1
        item["gait_tcn_positive"] += int(bool(row["gait_tcn_video_positive"]))
        item["sit_stand_tcn_events"] += int(row["sit_stand_tcn_event_count"])
        item["fall_rule_triggers"] += int(row["fall_rule_trigger_count"])
    return {
        **dict(contract),
        "video_count": len(rows),
        "gait": {
            "rule": counts("gait_rule_video_positive"),
            "tcn": counts("gait_tcn_video_positive"),
            "window_count": sum(int(row["gait_tcn_window_count"]) for row in rows),
        },
        "sit_stand": {
            "rule_event_count": sum(int(row["sit_stand_rule_event_count"]) for row in rows),
            "tcn_event_count": sum(int(row["sit_stand_tcn_event_count"]) for row in rows),
            "tcn_valid_window_count": sum(int(row["sit_stand_tcn_valid_window_count"]) for row in rows),
        },
        "fall_rule": {
            "trigger_count": sum(int(row["fall_rule_trigger_count"]) for row in rows),
            "positive_video_count": sum(bool(row["fall_rule_video_positive"]) for row in rows),
        },
        "strata": [
            {"provisional_subject": key[0], "planned_action": key[1], **value}
            for key, value in sorted(strata.items())
        ],
        "source_diagnostics": _diagnostic_summary(rows),
        "status": "completed",
        "results": rows,
    }


def _planned_actions(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    output: dict[str, str] = {}
    for row in rows:
        video_id = str(row["video_id"])
        action_id = str(row.get("action_id", ""))
        filename_action = str(row["file_name"]).rsplit("_", 1)[-1].removesuffix(".mp4")
        if action_id == filename_action:
            output[video_id] = action_id
    return output


def _source_diagnostics(
    manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    filename_parts = Path(str(manifest["file_name"])).stem.split("_")
    frame_ids = {
        int(row["frame_id"])
        for row in records
        if row.get("frame_id") is not None
    }
    groups: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    qualities = []
    for row in records:
        groups[(str(row.get("person_id", "unknown")), str(row.get("track_id", "none")))].append(row)
        if row.get("core_keypoint_quality") is not None:
            qualities.append(float(row["core_keypoint_quality"]))
    timestamps = sorted(
        {
            float(row["timestamp_sec"])
            for row in records
            if row.get("timestamp_sec") is not None
        }
    )
    gaps = [later - earlier for earlier, later in zip(timestamps, timestamps[1:])]
    coverage_sec = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    source_frames = int(manifest["frame_count"])
    return {
        "session": filename_parts[1],
        "episode": filename_parts[2],
        "script": filename_parts[3],
        "rotation": int(manifest.get("rotation", 0)),
        "camera_view": "unavailable_not_annotated",
        "low_light": "unavailable_not_annotated",
        "occlusion": "unavailable_not_annotated",
        "source_frame_count": source_frames,
        "detected_frame_count": len(frame_ids),
        "detected_frame_coverage": round(len(frame_ids) / source_frames, 4),
        "pose_record_count": len(records),
        "track_count": len(groups),
        "main_track_record_count": max((len(rows) for rows in groups.values()), default=0),
        "mean_core_keypoint_quality": (
            round(sum(qualities) / len(qualities), 4) if qualities else None
        ),
        "effective_fps": round((len(timestamps) - 1) / coverage_sec, 4) if coverage_sec > 0 else 0.0,
        "max_gap_sec": round(max(gaps, default=0.0), 4),
        "source_gap_count_over_0_75_sec": sum(gap > 0.75 for gap in gaps),
    }


def _diagnostic_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    diagnostics = [row["source_diagnostics"] for row in rows]
    coverage = [float(row["detected_frame_coverage"]) for row in diagnostics]
    return {
        "video_count": len(diagnostics),
        "mean_detected_frame_coverage": (
            round(sum(coverage) / len(coverage), 4) if coverage else None
        ),
        "videos_below_0_8_detected_frame_coverage": sum(value < 0.8 for value in coverage),
        "videos_with_multiple_tracks": sum(int(row["track_count"]) > 1 for row in diagnostics),
        "videos_with_source_gap_over_0_75_sec": sum(
            int(row["source_gap_count_over_0_75_sec"]) > 0 for row in diagnostics
        ),
        "unavailable_strata": ["camera_view", "low_light", "occlusion"],
    }


def _sit_configs(training_path: Path, evaluation_path: Path) -> tuple[Any, Any]:
    training = _read_yaml(training_path)
    evaluation = _read_yaml(evaluation_path)
    values = dict(training["continuous_dataset"])
    for key in ("window_frames", "partial_context_policy", "multi_person_policy", "joints", "base_channels", "timing_channels", "negative_policy", "test_pose_allowed"):
        values.pop(key, None)
    continuous = SitStandContinuousConfig(**values)
    decoder = SitStandStreamDecoderConfig(
        presence_threshold=float(evaluation["score_threshold"]),
        state_threshold=float(evaluation["score_threshold"]),
        boundary_threshold=float(evaluation["score_threshold"]),
        confirmation_frames=int(evaluation["confirmation_frames"]),
        event_merge_gap_sec=float(evaluation["event_merge_gap_sec"]),
        max_sequence_gap_sec=continuous.max_gap_sec,
        minimum_duration_sec=1.0 / continuous.target_fps,
    )
    return continuous, decoder


def _file_contract(path: Path) -> dict[str, str]:
    return {"path": path.as_posix(), "sha256": _sha256(path)}


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML mapping required: {path}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSON object required: {path}:{number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
