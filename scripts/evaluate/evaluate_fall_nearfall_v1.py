from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from elderly_monitoring.modules.fall_risk.evaluation import evaluate_event_predictions
from elderly_monitoring.modules.fall_risk.fall_event_continuous_tcn import (
    ContinuousFallTCNShadowPredictor,
)
from elderly_monitoring.modules.fall_risk.near_fall import extract_near_fall_events
from elderly_monitoring.modules.fall_risk.near_fall_tabular import (
    NearFallTabularPredictor,
    rescore_near_fall_events,
)
from elderly_monitoring.modules.fall_risk.pose import run_yolov8_pose, write_jsonl
from elderly_monitoring.modules.fall_risk.pose_quality import (
    PoseQualityConfig,
    process_pose_records,
)
from elderly_monitoring.modules.fall_risk.self_collected_scf import replay_fall_rule_video


RELEASE_ID = "fall-risk-competition-v1-20260819"
SPLIT_ID = "fall-nearfall-v1-test-candidate"
FALL_WINDOW_SEC = 4.0
FALL_TARGET_FPS = 8.0
FALL_MAX_GAP_SEC = 0.25
FALL_STRIDE_SEC = 0.5
FALL_EVENT_RESET_GAP_SEC = 2.0
FALL_EVENT_END_CAP_SEC_DEFAULT = None
FALL_EVENT_PRE_ALERT_SEC_DEFAULT = None
FALL_MIN_OBSERVED_FRAMES = 16
FALL_THRESHOLD = 0.5
NEAR_FALL_THRESHOLD = 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen fall-event TCN and near-fall rule on the isolated home candidate set."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--fall-config", type=Path, required=True)
    parser.add_argument("--near-fall-config", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--fall-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--release-id",
        default=None,
        help="Expected release ID; required for a non-frozen development release.",
    )
    parser.add_argument(
        "--split-id",
        default=None,
        help="Prediction split ID; defaults to the frozen candidate split.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument(
        "--fall-score-threshold",
        type=float,
        default=FALL_THRESHOLD,
        help="Experimental fall score threshold; omitted keeps v1 behavior.",
    )
    parser.add_argument(
        "--fall-event-end-cap-sec",
        type=float,
        default=FALL_EVENT_END_CAP_SEC_DEFAULT,
        help="Experimental post-processing cap after the first fall alert; omitted keeps v1 behavior.",
    )
    parser.add_argument(
        "--fall-event-pre-alert-sec",
        type=float,
        default=FALL_EVENT_PRE_ALERT_SEC_DEFAULT,
        help="Experimental event start window before the first fall alert; omitted keeps v1 behavior.",
    )
    parser.add_argument(
        "--fall-event-alert-cooldown-sec",
        type=float,
        default=None,
        help=(
            "Experimental per-subject alert cooldown after a fall event; "
            "omitted keeps all independently decoded events."
        ),
    )
    parser.add_argument(
        "--near-fall-score-threshold",
        type=float,
        default=NEAR_FALL_THRESHOLD,
        help="Experimental near-fall score threshold; omitted keeps v1 behavior.",
    )
    parser.add_argument(
        "--near-fall-track-policy",
        choices=("per_frame", "longest_track"),
        default="per_frame",
        help="Experimental near-fall stream selection; omitted keeps per-frame v1 behavior.",
    )
    parser.add_argument(
        "--near-fall-suppress-on-fall",
        action="store_true",
        help="Experimental cross-branch deduplication: suppress near-fall alerts when a fall event is emitted.",
    )
    parser.add_argument(
        "--near-fall-tabular-checkpoint",
        type=Path,
        default=None,
        help="Optional development checkpoint used to rescore rule-generated near-fall candidates.",
    )
    parser.add_argument(
        "--near-fall-alert-cooldown-sec",
        type=float,
        default=None,
        help="Optional causal cooldown used to suppress repeated near-fall alerts.",
    )
    parser.add_argument(
        "--near-fall-event-hold-sec",
        type=float,
        default=0.0,
        help="Optional post-alert duration retained in the decoded near-fall event interval.",
    )
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, RuntimeError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def run(args: argparse.Namespace) -> int:
    required_files = [
        args.manifest,
        args.ground_truth,
        args.fall_config,
        args.near_fall_config,
        args.release,
        args.pose_model,
        *args.fall_checkpoint,
    ]
    if args.near_fall_tabular_checkpoint is not None:
        required_files.append(args.near_fall_tabular_checkpoint)
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing input files: " + ", ".join(missing))
    if args.max_videos is not None and args.max_videos < 1:
        raise ValueError("max-videos must be positive")
    if not 0.0 < args.fall_score_threshold < 1.0:
        raise ValueError("fall score threshold must be within (0, 1)")
    if not 0.0 < args.near_fall_score_threshold < 1.0:
        raise ValueError("near-fall score threshold must be within (0, 1)")
    if (
        args.fall_event_alert_cooldown_sec is not None
        and args.fall_event_alert_cooldown_sec <= 0
    ):
        raise ValueError("fall event alert cooldown must be positive")
    if (
        args.near_fall_alert_cooldown_sec is not None
        and args.near_fall_alert_cooldown_sec <= 0
    ):
        raise ValueError("near-fall alert cooldown must be positive")
    if args.near_fall_event_hold_sec < 0:
        raise ValueError("near-fall event hold must be non-negative")
    if len(args.fall_checkpoint) != 3:
        raise ValueError("the frozen release requires exactly three fall checkpoints")

    manifest = sorted(_read_jsonl(args.manifest), key=lambda row: str(row["video_id"]))
    if args.max_videos is not None:
        manifest = manifest[: args.max_videos]
    ground_truth = _read_jsonl(args.ground_truth)
    fall_config = _effective_evaluation_config(
        _load_yaml(args.fall_config), args.fall_score_threshold
    )
    near_fall_config = _effective_evaluation_config(
        _load_yaml(args.near_fall_config), args.near_fall_score_threshold
    )
    fall_config_hash = _canonical_hash(fall_config)
    near_fall_config_hash = _canonical_hash(near_fall_config)
    release = _load_yaml(args.release)
    _validate_release(release, args)

    output = args.output_dir
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; pass --overwrite to replace it")
    output.mkdir(parents=True, exist_ok=True)
    pose_dir = output / "pose"
    video_dir = output / "videos"
    pose_dir.mkdir(exist_ok=True)
    video_dir.mkdir(exist_ok=True)
    contract = _contract(args, manifest, release, fall_config_hash, near_fall_config_hash)
    _write_json(output / "contract.json", contract)

    predictor = ContinuousFallTCNShadowPredictor(
        args.fall_checkpoint,
        device=args.device,
        threshold=args.fall_score_threshold,
        window_sec=FALL_WINDOW_SEC,
        target_fps=FALL_TARGET_FPS,
        max_gap_sec=FALL_MAX_GAP_SEC,
        min_observed_frames=FALL_MIN_OBSERVED_FRAMES,
    )
    near_fall_predictor = (
        NearFallTabularPredictor(args.near_fall_tabular_checkpoint)
        if args.near_fall_tabular_checkpoint is not None
        else None
    )
    all_fall_predictions: list[dict[str, Any]] = []
    all_near_fall_predictions: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    from ultralytics import YOLO

    pose_model = YOLO(args.pose_model.as_posix())
    for index, manifest_row in enumerate(manifest, 1):
        video_id = str(manifest_row["video_id"])
        video_output = video_dir / f"{video_id}.json"
        pose_output = pose_dir / f"{video_id}.jsonl"
        try:
            cleaned, pose_status = _load_or_extract_pose(
                manifest_row,
                pose_output,
                args.pose_model,
                pose_model,
                args.device,
            )
            evaluation_records, track_selection = _select_single_subject_stream(
                cleaned, video_id=video_id
            )
            near_fall_records = _select_near_fall_stream(
                cleaned,
                video_id=video_id,
                policy=args.near_fall_track_policy,
            )
            fall_windows, fall_events = _fall_event_predictions(
                evaluation_records,
                video_id=video_id,
                predictor=predictor,
                threshold=args.fall_score_threshold,
                end_cap_sec=args.fall_event_end_cap_sec,
                pre_alert_sec=args.fall_event_pre_alert_sec,
            )
            fall_events = _suppress_fall_event_realerts(
                fall_events,
                cooldown_sec=args.fall_event_alert_cooldown_sec,
            )
            near_events = extract_near_fall_events(near_fall_records)
            if near_fall_predictor is not None:
                near_events = rescore_near_fall_events(
                    near_events,
                    near_fall_records,
                    predictor=near_fall_predictor,
                    fallback_to_rule=False,
                )
            fall_predictions = _fall_prediction_rows(
                fall_events,
                video_id=video_id,
                config_hash=fall_config_hash,
                model_version=predictor.model_version,
                split_id=str(args.split_id or SPLIT_ID),
            )
            near_predictions = _near_fall_predictions(
                near_events,
                video_id=video_id,
                config_hash=near_fall_config_hash,
                split_id=str(args.split_id or SPLIT_ID),
                threshold=args.near_fall_score_threshold,
                hold_sec=args.near_fall_event_hold_sec,
            )
            near_predictions = _suppress_near_fall_realerts(
                near_predictions,
                cooldown_sec=args.near_fall_alert_cooldown_sec,
            )
            near_predictions = _suppress_near_fall_on_fall(
                near_predictions,
                fall_predictions,
                enabled=args.near_fall_suppress_on_fall,
            )
            all_fall_predictions.extend(fall_predictions)
            all_near_fall_predictions.extend(near_predictions)
            fallback = replay_fall_rule_video(records=cleaned)
            row = {
                "video_id": video_id,
                "file_name": manifest_row["file_name"],
                "scene_region": manifest_row.get("scene_region"),
                "planned_action": _planned_action(manifest_row),
                "status": "completed",
                "pose": pose_status,
                "evaluation_track": track_selection,
                "fall_event": {
                    "window_count": len(fall_windows),
                    "valid_window_count": sum(row["status"] == "valid" for row in fall_windows),
                    "positive_window_count": sum(
                        row["score"] is not None
                        and row["score"] >= args.fall_score_threshold
                        for row in fall_windows
                    ),
                    "event_count": len(fall_predictions),
                    "events": fall_predictions,
                },
                "near_fall": {
                    "raw_event_count": len(near_events),
                    "positive_event_count": len(near_predictions),
                    "events": near_predictions,
                    "max_score": max(
                        (
                            float(item["near_fall_event_score"])
                            for item in near_events
                            if item.get("near_fall_event_score") is not None
                        ),
                        default=None,
                    ),
                },
                "fall_rule_fallback": {
                    "trigger_count": fallback["trigger_count"],
                    "video_positive": fallback["video_positive"],
                },
            }
        except (OSError, RuntimeError, ValueError) as exc:
            row = {
                "video_id": video_id,
                "file_name": manifest_row["file_name"],
                "scene_region": manifest_row.get("scene_region"),
                "planned_action": _planned_action(manifest_row),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        _write_json(video_output, row)
        video_rows.append(row)
        _write_json(output / "progress.json", _progress(video_rows, started))
        print(f"[{index}/{len(manifest)}] {row['status']} {video_id}", flush=True)

    evaluated_video_ids = {
        str(item["video_id"])
        for item in video_rows
        if item.get("status") == "completed"
    }
    fall_truth = [
        row
        for row in ground_truth
        if row.get("task_type") == "fall_event"
        and str(row.get("video_id")) in evaluated_video_ids
    ]
    near_truth = [
        row
        for row in ground_truth
        if row.get("task_type") == "near_fall_event"
        and str(row.get("video_id")) in evaluated_video_ids
    ]
    manifest_for_eval = [
        row for row in manifest if str(row["video_id"]) in evaluated_video_ids
    ]
    fall_result = evaluate_event_predictions(
        fall_truth,
        all_fall_predictions,
        manifest=manifest_for_eval,
        config=fall_config,
    )
    near_result = evaluate_event_predictions(
        near_truth,
        all_near_fall_predictions,
        manifest=manifest_for_eval,
        config=near_fall_config,
    )
    _write_jsonl(output / "predictions_fall_event.jsonl", all_fall_predictions)
    _write_jsonl(output / "predictions_near_fall_event.jsonl", all_near_fall_predictions)
    _write_json(output / "metrics_fall_event.json", _result_payload(fall_result))
    _write_json(output / "metrics_near_fall_event.json", _result_payload(near_result))
    failures = [row for row in video_rows if row.get("status") != "completed"]
    _write_jsonl(output / "failures.jsonl", failures)
    summary = _summary(
        contract=contract,
        video_rows=video_rows,
        fall_result=fall_result,
        near_result=near_result,
        ground_truth=ground_truth,
        predictions=(all_fall_predictions, all_near_fall_predictions),
    )
    _write_json(output / "metrics.json", summary)
    _write_json(output / "progress.json", {**_progress(video_rows, started), "status": "completed"})
    print(json.dumps(summary["headline"], ensure_ascii=False, sort_keys=True))
    return 0 if not failures else 2


def _load_or_extract_pose(
    manifest: Mapping[str, Any],
    output_path: Path,
    model_path: Path,
    model: Any,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = Path(str(manifest["path"]))
    if not source.is_file():
        raise FileNotFoundError(source)
    if _sha256(source) != str(manifest["content_sha256"]):
        raise ValueError(f"source SHA-256 mismatch: {manifest['video_id']}")
    if output_path.is_file():
        records = _read_jsonl(output_path)
    else:
        raw_path = output_path.with_suffix(".raw.jsonl")
        run_yolov8_pose(
            video_path=source,
            output_path=raw_path,
            model_name=model_path.as_posix(),
            scene_region=str(manifest.get("scene_region", "unknown")),
            person_id_prefix=str(manifest["video_id"]),
            confidence_threshold=0.25,
            iou_threshold=0.5,
            tracker_config="bytetrack.yaml",
            max_frames=None,
            normalize_coordinates=True,
            device=device,
            model=model,
            persist_tracker=False,
        )
        records = process_pose_records(_read_jsonl(raw_path), config=PoseQualityConfig())
        write_jsonl(records, output_path)
        raw_path.unlink(missing_ok=True)
    timestamps = sorted({float(row["timestamp_sec"]) for row in records if row.get("timestamp_sec") is not None})
    frame_ids = {int(row["frame_id"]) for row in records if row.get("frame_id") is not None}
    groups = defaultdict(list)
    quality: list[float] = []
    for row in records:
        groups[(str(row.get("person_id", "unknown")), str(row.get("track_id", "none")))].append(row)
        if row.get("core_keypoint_quality") is not None:
            quality.append(float(row["core_keypoint_quality"]))
    gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
    return records, {
        "record_count": len(records),
        "detected_frame_count": len(frame_ids),
        "source_frame_count": int(manifest["frame_count"]),
        "detected_frame_coverage": round(len(frame_ids) / max(int(manifest["frame_count"]), 1), 4),
        "track_count": len(groups),
        "main_track_record_count": max((len(items) for items in groups.values()), default=0),
        "mean_core_keypoint_quality": round(sum(quality) / len(quality), 4) if quality else None,
        "max_gap_sec": round(max(gaps, default=0.0), 4),
        "effective_fps": round((len(timestamps) - 1) / (timestamps[-1] - timestamps[0]), 4) if len(timestamps) > 1 and timestamps[-1] > timestamps[0] else 0.0,
    }


def _select_single_subject_stream(
    records: Sequence[Mapping[str, Any]], *, video_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collapse tracker fragments to one causal stream for this single-subject set.

    The candidate manifest is explicitly single-subject, but ByteTrack can emit
    several IDs for the same person during occlusion or a detector re-acquisition.
    Evaluating every ID separately creates duplicate events and shortens the
    causal history.  Select one observation per source frame using pose
    confidence, then keypoint quality and box area as deterministic tie-breakers.
    """
    by_frame: dict[tuple[int, float], dict[str, Any]] = {}
    source_track_ids: set[str] = set()
    for source in records:
        row = dict(source)
        frame_id = int(row.get("frame_id", -1))
        timestamp = float(row.get("timestamp_sec", 0.0))
        key = (frame_id, round(timestamp, 6))
        track_id = str(row.get("track_id", "none"))
        source_track_ids.add(track_id)
        previous = by_frame.get(key)
        if previous is None or _single_subject_rank(row) > _single_subject_rank(previous):
            by_frame[key] = row

    selected: list[dict[str, Any]] = []
    for row in sorted(
        by_frame.values(),
        key=lambda item: (float(item.get("timestamp_sec", 0.0)), int(item.get("frame_id", 0))),
    ):
        row["source_track_id"] = row.get("track_id")
        row["person_id"] = f"{video_id}_primary"
        row["track_id"] = "primary"
        selected.append(row)
    return selected, {
        "policy": "per_frame_highest_pose_confidence",
        "source_record_count": len(records),
        "selected_record_count": len(selected),
        "source_track_count": len(source_track_ids),
        "selected_track_id": "primary",
    }


def _select_near_fall_stream(
    records: Sequence[Mapping[str, Any]],
    *,
    video_id: str,
    policy: str,
) -> list[dict[str, Any]]:
    """Select a temporally coherent stream for the near-fall branch.

    The default keeps the v1 per-frame stream.  The experimental longest-track
    policy avoids stitching unrelated tracker fragments into one motion series.
    """
    if policy == "per_frame":
        selected, _ = _select_single_subject_stream(records, video_id=video_id)
        return selected
    if policy != "longest_track":
        raise ValueError(f"unsupported near-fall track policy: {policy}")
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in records:
        groups[
            (str(source.get("person_id", "unknown")), str(source.get("track_id", "none")))
        ].append(dict(source))
    if not groups:
        return []
    key = max(
        groups,
        key=lambda item: (
            len(groups[item]),
            sum(float(row.get("core_keypoint_quality", 0.0) or 0.0) for row in groups[item]),
            item,
        ),
    )
    selected = sorted(
        groups[key],
        key=lambda row: (
            float(row.get("timestamp_sec", 0.0)),
            int(row.get("frame_id", 0)),
        ),
    )
    for row in selected:
        row["source_track_id"] = row.get("track_id")
        row["person_id"] = f"{video_id}_primary"
        row["track_id"] = "primary"
    return selected


def _single_subject_rank(row: Mapping[str, Any]) -> tuple[float, float, float]:
    bbox = row.get("bbox")
    area = 0.0
    if isinstance(bbox, Sequence) and len(bbox) >= 4:
        try:
            area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(
                0.0, float(bbox[3]) - float(bbox[1])
            )
        except (TypeError, ValueError):
            area = 0.0
    return (
        float(row.get("pose_confidence") or 0.0),
        float(row.get("core_keypoint_quality") or 0.0),
        area,
    )


def _fall_event_predictions(
    records: Sequence[Mapping[str, Any]],
    *,
    video_id: str,
    predictor: ContinuousFallTCNShadowPredictor,
    threshold: float = FALL_THRESHOLD,
    end_cap_sec: float | None = FALL_EVENT_END_CAP_SEC_DEFAULT,
    pre_alert_sec: float | None = FALL_EVENT_PRE_ALERT_SEC_DEFAULT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < threshold < 1.0:
        raise ValueError("fall score threshold must be within (0, 1)")
    if end_cap_sec is not None and end_cap_sec <= 0:
        raise ValueError("fall event end cap must be positive")
    if pre_alert_sec is not None and pre_alert_sec <= 0:
        raise ValueError("fall event pre-alert window must be positive")
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record.get("person_id", "unknown")), str(record.get("track_id", "none")))].append(dict(record))
    windows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for key, source in sorted(groups.items()):
        ordered = sorted(source, key=lambda row: (float(row.get("timestamp_sec", 0.0)), int(row.get("frame_id", 0))))
        if not ordered:
            continue
        first = float(ordered[0].get("timestamp_sec", 0.0))
        last = float(ordered[-1].get("timestamp_sec", 0.0))
        if last - first < FALL_WINDOW_SEC - 1e-9:
            continue
        cutoffs = _grid(first + FALL_WINDOW_SEC, last, FALL_STRIDE_SEC)
        positive_windows: list[dict[str, Any]] = []
        for cutoff in cutoffs:
            past = [row for row in ordered if float(row.get("timestamp_sec", 0.0)) <= cutoff + 1e-9]
            item = predictor.predict_records(past)
            score = item.get("fall_event_tcn_shadow_score")
            start = max(0.0, cutoff - FALL_WINDOW_SEC)
            onset = None
            if score is not None and item.get("fall_event_tcn_shadow_onset_frame") is not None:
                onset = min(cutoff, max(start, start + float(item["fall_event_tcn_shadow_onset_frame"]) / FALL_TARGET_FPS))
            window = {
                "video_id": video_id,
                "person_id": key[0],
                "track_id": key[1],
                "cutoff_time": round(cutoff, 4),
                "window_start": round(start, 4),
                "score": round(float(score), 6) if score is not None else None,
                "onset_time": round(float(onset), 4) if onset is not None else None,
                "status": "valid" if score is not None else "unavailable",
                "reason": item.get("fall_event_tcn_shadow_reason"),
                "observed_frame_count": item.get("fall_event_tcn_shadow_observed_frame_count"),
                "valid_joint_ratio": item.get("fall_event_tcn_shadow_valid_joint_ratio"),
            }
            windows.append(window)
            if score is not None and float(score) >= threshold:
                positive_windows.append(window)
        events.extend(
            _merge_fall_windows(
                positive_windows,
                video_id=video_id,
                person_id=key[0],
                track_id=key[1],
                end_cap_sec=end_cap_sec,
                pre_alert_sec=pre_alert_sec,
            )
        )
    return windows, events


def _merge_fall_windows(
    windows: Sequence[Mapping[str, Any]],
    *,
    video_id: str,
    person_id: str,
    track_id: str,
    end_cap_sec: float | None = FALL_EVENT_END_CAP_SEC_DEFAULT,
    pre_alert_sec: float | None = FALL_EVENT_PRE_ALERT_SEC_DEFAULT,
) -> list[dict[str, Any]]:
    if not windows:
        return []
    ordered = sorted(windows, key=lambda row: float(row["cutoff_time"]))
    merged: list[dict[str, Any]] = []
    current = [dict(ordered[0])]
    for row in ordered[1:]:
        if (
            float(row["cutoff_time"]) - float(current[-1]["cutoff_time"])
            <= FALL_EVENT_RESET_GAP_SEC
        ):
            current.append(dict(row))
        else:
            merged.append(_merged_fall_event(current, video_id, person_id, track_id, end_cap_sec=end_cap_sec, pre_alert_sec=pre_alert_sec))
            current = [dict(row)]
    merged.append(_merged_fall_event(current, video_id, person_id, track_id, end_cap_sec=end_cap_sec, pre_alert_sec=pre_alert_sec))
    return merged


def _suppress_fall_event_realerts(
    events: Sequence[Mapping[str, Any]], *, cooldown_sec: float | None
) -> list[dict[str, Any]]:
    """Keep the first alert in a causal incident cooldown window.

    A fallen person often remains in a pose that keeps consecutive causal
    windows positive.  This opt-in gate prevents that one incident from
    repeatedly emitting a new alert while retaining a later incident once the
    cooldown has elapsed.
    """
    if cooldown_sec is None:
        return [dict(event) for event in events]
    if cooldown_sec <= 0:
        raise ValueError("fall event alert cooldown must be positive")
    grouped: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[
            (
                str(event["video_id"]),
                str(event["person_id"]),
                str(event["track_id"]),
            )
        ].append(dict(event))
    retained: list[dict[str, Any]] = []
    for key in sorted(grouped):
        last_alert: float | None = None
        for event in sorted(grouped[key], key=lambda row: float(row["alert_time"])):
            alert_time = float(event["alert_time"])
            if last_alert is not None and alert_time - last_alert < cooldown_sec:
                continue
            retained.append(event)
            last_alert = alert_time
    return sorted(
        retained,
        key=lambda row: (
            str(row["video_id"]),
            str(row["person_id"]),
            str(row["track_id"]),
            float(row["alert_time"]),
        ),
    )


def _merged_fall_event(
    rows: Sequence[Mapping[str, Any]],
    video_id: str,
    person_id: str,
    track_id: str,
    *,
    end_cap_sec: float | None = FALL_EVENT_END_CAP_SEC_DEFAULT,
    pre_alert_sec: float | None = FALL_EVENT_PRE_ALERT_SEC_DEFAULT,
) -> dict[str, Any]:
    onset_values = [float(row["onset_time"]) for row in rows if row.get("onset_time") is not None]
    alert_time = float(rows[0]["cutoff_time"])
    onset_time = min(onset_values) if onset_values else alert_time
    start_time = min(float(row["window_start"]) for row in rows)
    if pre_alert_sec is not None:
        start_time = max(0.0, alert_time - float(pre_alert_sec))
        start_time = min(start_time, onset_time)
    end_time = max(float(row["cutoff_time"]) for row in rows)
    if end_cap_sec is not None:
        end_time = min(end_time, alert_time + float(end_cap_sec))
    return {
        "video_id": video_id,
        "person_id": person_id,
        "track_id": track_id,
        "start_time": start_time,
        "end_time": end_time,
        "onset_time": onset_time,
        "alert_time": alert_time,
        "score": max(float(row["score"]) for row in rows),
        "window_count": len(rows),
    }


def _fall_prediction_rows(
    events: Sequence[Mapping[str, Any]], *, video_id: str, config_hash: str, model_version: str,
    split_id: str = SPLIT_ID,
) -> list[dict[str, Any]]:
    return [
        {
            "video_id": video_id,
            "task_type": "fall_event",
            "event_type": "fall",
            "prediction_id": f"pred_fall_{video_id}_{index:03d}",
            "score": round(float(event["score"]), 6),
            "start_time": round(float(event["start_time"]), 4),
            "end_time": round(float(event["end_time"]), 4),
            "onset_time": round(float(event["onset_time"]), 4),
            "alert_time": round(float(event["alert_time"]), 4),
            "status": "emitted",
            "quality_state": "valid",
            "model_version": model_version,
            "config_hash": config_hash,
            "split_id": split_id,
            "risk_level": 4,
            "window_count": int(event["window_count"]),
        }
        for index, event in enumerate(events, 1)
    ]


def _near_fall_predictions(
    events: Sequence[Mapping[str, Any]], *, video_id: str, config_hash: str,
    split_id: str = SPLIT_ID,
    threshold: float = NEAR_FALL_THRESHOLD,
    hold_sec: float = 0.0,
) -> list[dict[str, Any]]:
    if hold_sec < 0:
        raise ValueError("near-fall event hold must be non-negative")
    predictions = []
    for index, event in enumerate(events, 1):
        score = event.get("near_fall_event_score")
        if score is None or float(score) < threshold:
            continue
        start = float(event["start_time"])
        alert_time = max(start, float(event["end_time"]))
        end = alert_time + hold_sec
        predictions.append(
            {
                "video_id": video_id,
                "task_type": "near_fall_event",
                "event_type": "near_fall",
                "prediction_id": f"pred_near_fall_{video_id}_{index:03d}",
                "score": round(float(score), 6),
                "start_time": round(start, 4),
                "end_time": round(end, 4),
                "onset_time": round(start, 4),
                "alert_time": round(alert_time, 4),
                "status": "emitted",
                "quality_state": "valid",
                "model_version": str(event.get("model_version", "near-fall-rule-v0.1")),
                "config_hash": config_hash,
                "split_id": split_id,
                "risk_level": 3,
                "event_type_detail": event.get("event_type"),
            }
        )
    return predictions


def _suppress_near_fall_on_fall(
    near_predictions: Sequence[Mapping[str, Any]],
    fall_predictions: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
) -> list[dict[str, Any]]:
    if enabled and fall_predictions:
        return []
    return [dict(row) for row in near_predictions]


def _suppress_near_fall_realerts(
    predictions: Sequence[Mapping[str, Any]],
    *,
    cooldown_sec: float | None,
) -> list[dict[str, Any]]:
    if cooldown_sec is None:
        return [dict(row) for row in predictions]
    if cooldown_sec <= 0:
        raise ValueError("near-fall alert cooldown must be positive")
    retained: list[dict[str, Any]] = []
    last_alert_by_video: dict[str, float] = {}
    for row in sorted(
        (dict(item) for item in predictions),
        key=lambda item: (str(item["video_id"]), float(item["alert_time"])),
    ):
        video_id = str(row["video_id"])
        alert_time = float(row["alert_time"])
        last_alert = last_alert_by_video.get(video_id)
        if last_alert is not None and alert_time - last_alert < cooldown_sec:
            continue
        retained.append(row)
        last_alert_by_video[video_id] = alert_time
    return retained


def _result_payload(result: Any) -> dict[str, Any]:
    return {
        "metrics": result.metrics,
        "confidence_intervals_95": result.confidence_intervals,
        "matches": result.matches,
        "false_positives": result.false_positives,
        "false_negatives": result.false_negatives,
        "excluded_samples": result.excluded_samples,
        "threshold_curve": result.threshold_curve,
        "protocol_version": result.protocol_version,
        "protocol_status": result.protocol_status,
        "evaluation_config_hash": result.config_hash,
    }


def _summary(
    *,
    contract: Mapping[str, Any],
    video_rows: Sequence[Mapping[str, Any]],
    fall_result: Any,
    near_result: Any,
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    fall_metrics = fall_result.metrics
    near_metrics = near_result.metrics
    completed = [row for row in video_rows if row.get("status") == "completed"]
    fall_video_presence = _video_presence_metrics(
        ground_truth,
        predictions[0],
        evaluated_video_ids={str(row["video_id"]) for row in completed},
    )
    return {
        "schema_version": "fall-nearfall-v1-engineering-evaluation-report-v1",
        "status": "completed",
        "contract": dict(contract),
        "video_count": len(video_rows),
        "completed_video_count": len(completed),
        "failed_video_count": len(video_rows) - len(completed),
        "ground_truth_count": {
            "fall_event": sum(row.get("task_type") == "fall_event" for row in ground_truth),
            "near_fall_event": sum(row.get("task_type") == "near_fall_event" for row in ground_truth),
        },
        "prediction_count": {
            "fall_event": len(predictions[0]),
            "near_fall_event": len(predictions[1]),
        },
        "headline": {
            "fall_event": {key: fall_metrics.get(key) for key in ("tp", "fp", "fn", "precision", "recall", "f1", "event_pr_auc", "mean_onset_detection_latency_sec")},
            "near_fall_event": {key: near_metrics.get(key) for key in ("tp", "fp", "fn", "precision", "recall", "f1", "event_pr_auc", "mean_onset_detection_latency_sec")},
        },
        "fall_video_presence": fall_video_presence,
        "fall_event": fall_metrics,
        "near_fall_event": near_metrics,
        "fallback_diagnostics": {
            "videos_with_fall_rule_trigger": sum(bool(row.get("fall_rule_fallback", {}).get("video_positive")) for row in completed),
            "total_fall_rule_triggers": sum(int(row.get("fall_rule_fallback", {}).get("trigger_count", 0)) for row in completed),
        },
    }


def _video_presence_metrics(
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    evaluated_video_ids: set[str],
) -> dict[str, Any]:
    """Report the less strict question: did a video receive any fall alert?"""
    truth_by_video = {
        str(row["video_id"]): row
        for row in ground_truth
        if row.get("task_type") == "fall_event"
        and str(row.get("video_id")) in evaluated_video_ids
    }
    predicted_by_video: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        if str(row.get("video_id")) in evaluated_video_ids:
            predicted_by_video[str(row["video_id"])].append(row)
    truth_ids = set(truth_by_video)
    predicted_ids = set(predicted_by_video)
    tp = len(truth_ids & predicted_ids)
    fp = len(predicted_ids - truth_ids)
    fn = len(truth_ids - predicted_ids)
    precision, recall, f1 = _rates(tp, fp, fn)
    any_interval_hits = 0
    first_alert_interval_hits = 0
    for video_id, truth in truth_by_video.items():
        onset = float(truth.get("onset_time", truth["start_time"]))
        end = float(truth["end_time"])
        alerts = sorted(
            float(row.get("alert_time", row["onset_time"]))
            for row in predicted_by_video.get(video_id, [])
        )
        if any(
            onset <= float(row.get("alert_time", row["onset_time"])) <= end
            for row in predicted_by_video.get(video_id, [])
        ):
            any_interval_hits += 1
        if alerts and onset <= alerts[0] <= end:
            first_alert_interval_hits += 1
    duplicate_prediction_count = sum(
        max(0, len(rows) - 1) for rows in predicted_by_video.values()
    )
    videos_with_duplicate_predictions = sum(
        len(rows) > 1 for rows in predicted_by_video.values()
    )
    return {
        "definition": "one positive if any emitted fall prediction exists in a video",
        "truth_video_count": len(truth_ids),
        "predicted_video_count": len(predicted_ids),
        "negative_video_count": len(evaluated_video_ids - truth_ids),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "any_alert_inside_annotated_interval_count": any_interval_hits,
        "any_alert_inside_annotated_interval_recall": (
            any_interval_hits / len(truth_ids) if truth_ids else None
        ),
        "first_alert_inside_annotated_interval_count": first_alert_interval_hits,
        "first_alert_inside_annotated_interval_recall": (
            first_alert_interval_hits / len(truth_ids) if truth_ids else None
        ),
        "duplicate_prediction_count": duplicate_prediction_count,
        "videos_with_duplicate_predictions": videos_with_duplicate_predictions,
        "warning": "video presence is not an event-boundary metric and must not replace event F1",
    }


def _rates(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return precision, recall, f1


def _contract(
    args: argparse.Namespace,
    manifest: Sequence[Mapping[str, Any]],
    release: Mapping[str, Any],
    fall_config_hash: str,
    near_fall_config_hash: str,
) -> dict[str, Any]:
    release_id = str(args.release_id or RELEASE_ID)
    split_id = str(args.split_id or SPLIT_ID)
    return {
        "schema_version": "fall-nearfall-v1-engineering-evaluation-contract-v1",
        "release_id": release_id,
        "release_file_sha256": _sha256(args.release),
        "split_id": split_id,
        "manifest_sha256": _sha256(args.manifest),
        "ground_truth_sha256": _sha256(args.ground_truth),
        "video_count_requested": len(manifest),
        "pose_model": {"path": args.pose_model.as_posix(), "sha256": _sha256(args.pose_model)},
        "fall_checkpoints": [{"path": path.as_posix(), "sha256": _sha256(path)} for path in args.fall_checkpoint],
        "fall_config_sha256": fall_config_hash,
        "fall_config_file_sha256": _sha256(args.fall_config),
        "near_fall_config_sha256": near_fall_config_hash,
        "near_fall_config_file_sha256": _sha256(args.near_fall_config),
        "fall_window_sec": FALL_WINDOW_SEC,
        "fall_target_fps": FALL_TARGET_FPS,
        "fall_max_gap_sec": FALL_MAX_GAP_SEC,
        "fall_min_observed_frames": FALL_MIN_OBSERVED_FRAMES,
        "fall_stride_sec": FALL_STRIDE_SEC,
        "fall_event_reset_gap_sec": FALL_EVENT_RESET_GAP_SEC,
        "fall_event_end_cap_sec": args.fall_event_end_cap_sec,
        "fall_event_pre_alert_sec": args.fall_event_pre_alert_sec,
        "fall_event_alert_cooldown_sec": args.fall_event_alert_cooldown_sec,
        "fall_score_threshold": args.fall_score_threshold,
        "near_fall_score_threshold": args.near_fall_score_threshold,
        "near_fall_track_policy": args.near_fall_track_policy,
        "near_fall_suppress_on_fall": args.near_fall_suppress_on_fall,
        "near_fall_alert_cooldown_sec": args.near_fall_alert_cooldown_sec,
        "near_fall_event_hold_sec": args.near_fall_event_hold_sec,
        "near_fall_tabular_checkpoint": (
            {
                "path": args.near_fall_tabular_checkpoint.as_posix(),
                "sha256": _sha256(args.near_fall_tabular_checkpoint),
            }
            if args.near_fall_tabular_checkpoint is not None
            else None
        ),
        "device": args.device,
        "protocol_status": "development_provisional",
        "test_pose_read": False,
        "test_evaluated": False,
        "single_subject_single_household": True,
        "evaluation_only_candidate": True,
        "release_snapshot": release.get("release_id"),
    }


def _validate_release(release: Mapping[str, Any], args: argparse.Namespace) -> None:
    expected_release_id = str(args.release_id or RELEASE_ID)
    if release.get("release_id") != expected_release_id:
        raise ValueError(f"release file ID mismatch: expected {expected_release_id}")
    expected = {
        str(item.get("role")): str(item.get("sha256"))
        for item in release.get("artifacts", [])
        if isinstance(item, Mapping)
    }
    for role, path in (
        ("pose_detection_and_tracking", args.pose_model),
        ("fall_event_ensemble_seed_42", args.fall_checkpoint[0]),
        ("fall_event_ensemble_seed_43", args.fall_checkpoint[1]),
        ("fall_event_ensemble_seed_44", args.fall_checkpoint[2]),
    ):
        if expected.get(role) != _sha256(path):
            raise ValueError(f"release artifact hash mismatch: {role}")


def _planned_action(row: Mapping[str, Any]) -> str | None:
    file_name = str(row.get("file_name", ""))
    suffix = Path(file_name).stem.rsplit("_", 1)[-1]
    action = str(row.get("action_id", ""))
    return action if action == suffix else None


def _grid(start: float, end: float, stride: float) -> list[float]:
    values: list[float] = []
    value = start
    while value <= end + 1e-9:
        values.append(round(value, 6))
        value += stride
    if not values or end - values[-1] > 1e-6:
        values.append(round(end, 6))
    return values


def _progress(rows: Sequence[Mapping[str, Any]], started: float) -> dict[str, Any]:
    return {
        "status": "running",
        "video_count": len(rows),
        "status_counts": dict(Counter(str(row.get("status")) for row in rows)),
        "elapsed_sec": round(time.monotonic() - started, 3),
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return dict(value)


def _effective_evaluation_config(
    config: Mapping[str, Any], score_threshold: float
) -> dict[str, Any]:
    effective = dict(config)
    effective["score_threshold"] = float(score_threshold)
    return effective


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {path}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _write_text(path, "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
