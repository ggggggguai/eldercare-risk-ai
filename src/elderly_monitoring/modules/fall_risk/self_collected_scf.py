from __future__ import annotations

import hashlib
import json
import math
import os
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


_BATCH_ID = "SCF_MVP_V1"
_DECISION_SCHEMA = "self-collected-scf-mvp-v1-import-decision-v1"
_INVENTORY_SCHEMA = "self-collected-delivery-inventory-v1"
_FILENAME_PATTERN = re.compile(
    r"(?P<subject>P\d{2})_(?P<session>S\d{2})_(?P<episode>E\d{3})_"
    r"(?P<script>[A-Z]\d{2})_(?P<action>[A-DU]\d{2})\.mp4"
)
_LABEL_PATTERN = re.compile(r"(?P<action>[A-DU]\d{2})_(?P<name>[A-Za-z0-9_]+)")
_HARD_NEGATIVE_TYPES = {
    "A02": "normal_turn",
    "A03": "fast_but_controlled_sit",
    "A05": "controlled_squat",
    "A06": "controlled_bend",
    "A08": "routine_support_contact",
    "A10": "normal_step_adjustment",
}
_NEAR_FALL_POSITIVE_ACTIONS = {"C03", "C04", "C05"}
_PLACEHOLDERS = {"", "unknown", "none", "null"}


@dataclass(frozen=True)
class _Task:
    task_id: str
    source: str
    size: int
    width: int
    height: int
    frame_offset: int


def build_scf_mvp_v1_candidates(
    *,
    inventory_path: Path | str,
    decision_path: Path | str,
    output_dir: Path | str,
) -> dict[str, str]:
    """Build an isolated, fail-closed SCF_MVP_V1 candidate release.

    The builder never mutates the root manifest, v2/v3 labels, or an existing
    split. Human-review gates are represented in the output instead of being
    inferred from filenames or CVAT action tracks.
    """
    inventory_file = Path(inventory_path)
    decision_file = Path(decision_path)
    destination = Path(output_dir)
    if not inventory_file.is_file():
        raise FileNotFoundError(inventory_file)
    if not decision_file.is_file():
        raise FileNotFoundError(decision_file)
    if destination.exists():
        raise FileExistsError(destination)

    decision = _read_json(decision_file)
    _validate_decision(decision, inventory_file)
    inventory = _read_jsonl(inventory_file)
    _validate_inventory(inventory)
    inventory_by_name = {str(row["file_name"]): row for row in inventory}
    expected_subjects = sorted({str(row["subject"]) for row in inventory})
    exports = _validated_exports(decision, expected_subjects)

    actions: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for subject, export in sorted(exports.items()):
        root = _read_cvat_root(export)
        tasks = _parse_tasks(root)
        for track in root.findall("track"):
            task = _task_for_track(track, tasks)
            row = inventory_by_name.get(task.source)
            if row is None:
                raise ValueError(f"CVAT source is absent from inventory: {task.source}")
            if str(row["subject"]) != subject:
                raise ValueError(f"CVAT source subject mismatch: {task.source}")
            seen_sources.add(task.source)
            actions.append(
                _action_candidate(
                    track,
                    task,
                    row,
                    decision=decision,
                    export_path=export,
                )
            )
    missing_sources = sorted(set(inventory_by_name) - seen_sources)
    if missing_sources:
        raise ValueError(f"inventory videos have no CVAT track: {missing_sources[:5]}")

    manifest = [_manifest_candidate(row, decision) for row in inventory]
    near_fall = _near_fall_candidates(actions, inventory_by_name)
    manifest.sort(key=lambda row: str(row["video_id"]))
    actions.sort(key=_record_sort_key)
    near_fall.sort(key=_record_sort_key)

    g0_blockers = _g0_blockers(inventory, decision)
    report = {
        "schema_version": "self-collected-scf-mvp-v1-import-report-v1",
        "batch_id": _BATCH_ID,
        "status": "candidate_built_g0_blocked" if g0_blockers else "candidate_built_g0_ready",
        "input_hashes": {
            "inventory_sha256": _sha256(inventory_file),
            "decision_sha256": _sha256(decision_file),
            "annotation_exports": {
                subject: _sha256(path) for subject, path in sorted(exports.items())
            },
        },
        "counts": {
            "videos": len(manifest),
            "unique_content": len({str(row["content_sha256"]) for row in manifest}),
            "action_tracks": len(actions),
            "near_fall_candidates": len(near_fall),
            "near_fall_hard_negative_types": dict(
                sorted(
                    Counter(
                        str(row["hard_negative_type"])
                        for row in near_fall
                        if row.get("hard_negative_type")
                    ).items()
                )
            ),
            "loss_eligible": sum(bool(row["loss_eligible"]) for row in near_fall),
        },
        "coordinate_transforms": dict(
            sorted(Counter(str(row["coordinate_transform"]) for row in actions).items())
        ),
        "g0": {
            "status": "blocked" if g0_blockers else "ready",
            "blockers": g0_blockers,
        },
        "root_artifacts_modified": False,
        "root_manifest_modified": False,
        "root_v2_labels_modified": False,
        "root_v3_labels_modified": False,
        "existing_splits_modified": False,
        "test_pose_read": False,
        "test_evaluated": False,
    }

    destination.mkdir(parents=True, exist_ok=False)
    paths = {
        "manifest_path": destination / "manifest.jsonl",
        "action_labels_path": destination / "action_labels.jsonl",
        "near_fall_candidates_path": destination / "near_fall_candidates.jsonl",
        "report_path": destination / "import_report.json",
    }
    _write_jsonl(paths["manifest_path"], manifest)
    _write_jsonl(paths["action_labels_path"], actions)
    _write_jsonl(paths["near_fall_candidates_path"], near_fall)
    _write_json(paths["report_path"], report)
    return {name: path.as_posix() for name, path in paths.items()}


def evaluate_scf_near_fall_gate(
    *,
    import_report_path: Path | str,
    near_fall_candidates_path: Path | str,
    output_path: Path | str,
    baseline_runs: Sequence[Path | str] = (),
) -> dict[str, Any]:
    """Evaluate whether self-collected candidates may enter G2 loss."""
    report_file = Path(import_report_path)
    candidates_file = Path(near_fall_candidates_path)
    output_file = Path(output_path)
    report = _read_json(report_file)
    candidates = _read_jsonl(candidates_file)
    eligible = [row for row in candidates if bool(row.get("loss_eligible"))]
    negative_types = {
        str(row["hard_negative_type"])
        for row in eligible
        if row.get("candidate_role") == "hard_negative"
    }
    positive_actions = {
        str(row["source_action_id"])
        for row in eligible
        if row.get("candidate_role") == "positive_candidate"
    }
    baseline_evidence = []
    for path_value in baseline_runs:
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = _read_json(path)
        baseline_evidence.append(
            {
                "path": _portable_path(path),
                "sha256": _sha256(path),
                "status": payload.get("status"),
                "seed": payload.get("seed", payload.get("training", {}).get("seed")),
            }
        )
    blockers = list(report.get("g0", {}).get("blockers", []))
    missing_negatives = sorted(set(_HARD_NEGATIVE_TYPES.values()) - negative_types)
    if missing_negatives:
        blockers.append("six_hard_negative_types_not_loss_eligible")
    experiments = {
        "E0": {
            "status": "frozen_baseline_available" if baseline_evidence else "baseline_reference_only",
            "training_increment": "none",
            "baseline_runs": baseline_evidence,
        },
        "E1": {
            "status": "ready" if not blockers else "blocked",
            "training_increment": "six_self_collected_hard_negative_types",
            "eligible_hard_negative_types": sorted(negative_types),
            "missing_hard_negative_types": missing_negatives,
        },
        "E2": {
            "status": "ready" if not blockers and "C03" in positive_actions else "blocked",
            "training_increment": "E1_plus_double_reviewed_C03",
        },
        "E3": {
            "status": (
                "ready"
                if not blockers and {"C03", "C04", "C05"} <= positive_actions
                else "blocked"
            ),
            "training_increment": "E2_plus_double_reviewed_C04_C05",
        },
    }
    gate = {
        "schema_version": "self-collected-near-fall-augmentation-gate-v1",
        "batch_id": _BATCH_ID,
        "status": "ready" if experiments["E1"]["status"] == "ready" else "blocked",
        "input_hashes": {
            "import_report_sha256": _sha256(report_file),
            "near_fall_candidates_sha256": _sha256(candidates_file),
        },
        "blockers": sorted(set(blockers)),
        "experiments": experiments,
        "main_path_unchanged": True,
        "test_pose_read": False,
        "test_evaluated": False,
    }
    if output_file.exists():
        raise FileExistsError(output_file)
    _write_json(output_file, gate)
    return gate


def replay_near_fall_video(
    *,
    cleaned_pose_path: Path | str,
    checkpoint_paths: Sequence[Path | str],
) -> dict[str, Any]:
    """Replay the rule baseline and frozen recovery-confirmation checkpoints."""
    from elderly_monitoring.modules.fall_risk.near_fall import (
        extract_near_fall_events,
    )
    from elderly_monitoring.modules.fall_risk.near_fall_tcn import (
        NearFallTCNPredictor,
    )
    from elderly_monitoring.modules.fall_risk.near_fall_training import (
        NearFallDatasetConfig,
        _resample_causal_window,
        build_near_fall_tensor,
    )

    pose_path = Path(cleaned_pose_path)
    records = _read_jsonl(pose_path)
    rule_events = extract_near_fall_events(records)
    rule_scores = [
        float(row["near_fall_event_score"])
        for row in rule_events
        if row.get("near_fall_event_score") is not None
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for source in records:
        row = dict(source)
        key = (str(row.get("person_id", "unknown")), str(row.get("track_id", "none")))
        grouped.setdefault(key, []).append(row)
    config = NearFallDatasetConfig()
    windows: list[tuple[tuple[str, str], int, Any]] = []
    for key, track in grouped.items():
        ordered = sorted(track, key=lambda row: (float(row.get("timestamp_sec", 0.0)), int(row.get("frame_id", 0))))
        if not ordered:
            continue
        span_sec = (config.window_frames - 1) / config.target_fps
        eligible = [
            row
            for row in ordered
            if float(row.get("timestamp_sec", 0.0)) >= float(ordered[0].get("timestamp_sec", 0.0)) + span_sec - 1e-9
        ]
        last_anchor_time: float | None = None
        for last in eligible:
            anchor_time = float(last.get("timestamp_sec", 0.0))
            if last_anchor_time is not None and anchor_time - last_anchor_time < config.stride_sec - 1e-9:
                continue
            anchor_frame = int(last.get("frame_id", -1))
            sampled = _resample_causal_window(
                ordered,
                anchor_time_sec=anchor_time,
                anchor_frame=anchor_frame,
                config=config,
            )
            tensor = build_near_fall_tensor(
                sampled,
                window_frames=config.window_frames,
                target_fps=config.target_fps,
            )
            windows.append((key, anchor_frame, tensor))
            last_anchor_time = anchor_time
        if eligible:
            last = eligible[-1]
            last_frame = int(last.get("frame_id", -1))
            if not windows or windows[-1][0] != key or windows[-1][1] != last_frame:
                sampled = _resample_causal_window(
                    ordered,
                    anchor_time_sec=float(last.get("timestamp_sec", 0.0)),
                    anchor_frame=last_frame,
                    config=config,
                )
                windows.append(
                    (
                        key,
                        last_frame,
                        build_near_fall_tensor(
                            sampled,
                            window_frames=config.window_frames,
                            target_fps=config.target_fps,
                        ),
                    )
                )
    model_runs = []
    for checkpoint_value in checkpoint_paths:
        checkpoint = Path(checkpoint_value)
        predictor = NearFallTCNPredictor(checkpoint, device="cpu")
        predictions = []
        for key, anchor_frame, tensor in windows:
            prediction = predictor.predict_tensor(tensor)
            predictions.append(
                {
                    "person_id": key[0],
                    "track_id": key[1],
                    "anchor_frame": anchor_frame,
                    **prediction,
                }
            )
        valid_scores = [
            float(row["near_fall_event_score"])
            for row in predictions
            if row.get("status") == "valid"
        ]
        model_runs.append(
            {
                "checkpoint_path": _portable_path(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "valid_prediction_count": len(valid_scores),
                "unavailable_prediction_count": len(predictions) - len(valid_scores),
                "max_near_fall_event_score": max(valid_scores, default=None),
                "threshold": 0.5,
                "positive_window_count": sum(score >= 0.5 for score in valid_scores),
                "video_positive": any(score >= 0.5 for score in valid_scores),
                "predictions": predictions,
            }
        )
    return {
        "schema_version": "self-collected-near-fall-video-replay-v1",
        "pose_path": _portable_path(pose_path),
        "pose_sha256": _sha256(pose_path),
        "pose_record_count": len(records),
        "track_count": len(grouped),
        "causal_window_count": len(windows),
        "rule": {
            "model_version": "near-fall-rule-v0.1",
            "event_count": len(rule_events),
            "max_near_fall_event_score": max(rule_scores, default=None),
            "events": rule_events,
        },
        "tcn": model_runs,
        "short_clip_replay": True,
        "fp_hour_reported": False,
        "main_path_unchanged": True,
    }


def replay_gait_video(
    *,
    records: Sequence[Mapping[str, Any]],
    model_predictor: Any,
    model_threshold: float,
    rule_threshold: float = 0.5,
) -> dict[str, Any]:
    """Replay the gait rule and one frozen provisional TCN on a short clip."""
    from elderly_monitoring.modules.fall_risk.gait import extract_gait_windows

    if not 0.0 <= model_threshold <= 1.0:
        raise ValueError("gait model threshold must be between zero and one")
    if not 0.0 <= rule_threshold <= 1.0:
        raise ValueError("gait rule threshold must be between zero and one")
    source = [dict(row) for row in records]
    rule_windows = extract_gait_windows(source)
    model_windows = extract_gait_windows(source, model_predictor=model_predictor)
    rule_scores = [
        float(row["gait_risk_score"])
        for row in rule_windows
        if row.get("score_source") != "unavailable"
        and row.get("gait_risk_score") is not None
    ]
    model_scores = [
        float(row["model_score"])
        for row in model_windows
        if row.get("score_source") == "tcn" and row.get("model_score") is not None
    ]
    return {
        "schema_version": "self-collected-gait-video-replay-v1",
        "pose_record_count": len(source),
        "rule": {
            "model_version": "gait-risk-rule-v0.2",
            "window_count": len(rule_windows),
            "valid_window_count": len(rule_scores),
            "unavailable_window_count": len(rule_windows) - len(rule_scores),
            "max_gait_risk_score": max(rule_scores, default=None),
            "threshold": rule_threshold,
            "positive_window_count": sum(score >= rule_threshold for score in rule_scores),
            "video_positive": any(score >= rule_threshold for score in rule_scores),
            "windows": rule_windows,
        },
        "tcn": {
            "model_version": str(model_predictor.model_version),
            "window_count": len(model_windows),
            "valid_window_count": len(model_scores),
            "fallback_window_count": sum(
                row.get("score_source") == "rule_fallback" for row in model_windows
            ),
            "unavailable_window_count": sum(
                row.get("score_source") == "unavailable" for row in model_windows
            ),
            "max_gait_risk_score": max(model_scores, default=None),
            "threshold": model_threshold,
            "positive_window_count": sum(score >= model_threshold for score in model_scores),
            "video_positive": any(score >= model_threshold for score in model_scores),
            "windows": model_windows,
        },
        "short_clip_replay": True,
        "fp_hour_reported": False,
        "main_path_unchanged": True,
        "test_pose_read": False,
        "test_evaluated": False,
    }


def replay_sit_stand_video(
    *,
    records: Sequence[Mapping[str, Any]],
    video_id: str,
    model: Any,
    device: Any,
    checkpoint_sha256: str,
    continuous_config: Any,
    decoder_config: Any,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Replay sit/stand rule and frozen continuous TCN without truth scoring."""
    from elderly_monitoring.modules.fall_risk.sit_stand_continuous_inference import (
        predict_rule_sit_stand_video,
        predict_tcn_sit_stand_video,
    )

    source = [dict(row) for row in records]
    rule = predict_rule_sit_stand_video(source, video_id=video_id)
    tcn = predict_tcn_sit_stand_video(
        source,
        video_id=video_id,
        model=model,
        device=device,
        checkpoint_sha256=checkpoint_sha256,
        continuous_config=continuous_config,
        decoder_config=decoder_config,
        batch_size=batch_size,
    )
    return {
        "schema_version": "self-collected-sit-stand-video-replay-v1",
        "pose_record_count": len(source),
        "rule": rule,
        "tcn": tcn,
        "short_clip_replay": True,
        "fp_hour_reported": False,
        "main_path_unchanged": True,
        "test_pose_read": False,
        "test_evaluated": False,
    }


def replay_fall_rule_video(
    *,
    records: Sequence[Mapping[str, Any]],
    analysis_interval_sec: float = 0.5,
    max_gap_sec: float = 0.75,
) -> dict[str, Any]:
    """Replay the runtime fall-state rule per pose track on a short clip."""
    from elderly_monitoring.runtime.fall_state import FallStateDetector
    from elderly_monitoring.runtime.feature_assembly import _state_observation

    if max_gap_sec <= 0:
        raise ValueError("fall rule max_gap_sec must be positive")
    if analysis_interval_sec <= 0:
        raise ValueError("fall rule analysis_interval_sec must be positive")
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for value in records:
        row = dict(value)
        key = (str(row.get("person_id", "unknown")), str(row.get("track_id", "none")))
        groups[key].append(row)

    streams = []
    total_trigger_count = 0
    total_valid_observations = 0
    for key, rows in sorted(groups.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                float(row.get("timestamp_sec", 0.0)),
                int(row.get("frame_id", 0)),
            ),
        )
        detector = FallStateDetector()
        previous: Mapping[str, Any] | None = None
        previous_time: float | None = None
        last_analysis_time: float | None = None
        gap_reset_count = 0
        trigger_rows = []
        valid_observation_count = 0
        max_fall_score = 0.0
        max_static_score = 0.0
        for row in ordered:
            timestamp = float(row.get("timestamp_sec", 0.0))
            if previous_time is not None and timestamp - previous_time > max_gap_sec:
                detector.reset()
                previous = None
                last_analysis_time = None
                gap_reset_count += 1
            if (
                last_analysis_time is not None
                and timestamp - last_analysis_time < analysis_interval_sec - 1e-9
            ):
                previous = row
                previous_time = timestamp
                continue
            observation = _state_observation(row, previous)
            if observation["core_keypoint_quality"] >= detector.config.min_quality:
                valid_observation_count += 1
            state = detector.update(observation)
            max_fall_score = max(max_fall_score, state.fall_event_score)
            max_static_score = max(max_static_score, state.long_static_score)
            if state.triggered_now:
                trigger_rows.append(
                    {
                        "frame_id": row.get("frame_id"),
                        "timestamp_sec": timestamp,
                        "fall_event_score": state.fall_event_score,
                    }
                )
            previous = row
            previous_time = timestamp
            last_analysis_time = timestamp
        total_trigger_count += len(trigger_rows)
        total_valid_observations += valid_observation_count
        streams.append(
            {
                "person_id": key[0],
                "track_id": key[1],
                "observation_count": len(ordered),
                "valid_observation_count": valid_observation_count,
                "gap_reset_count": gap_reset_count,
                "trigger_count": len(trigger_rows),
                "max_fall_event_score": max_fall_score,
                "max_long_static_score": max_static_score,
                "triggers": trigger_rows,
            }
        )
    return {
        "schema_version": "self-collected-fall-rule-video-replay-v1",
        "model_version": "fall-state-rule-v0.1",
        "analysis_interval_sec": analysis_interval_sec,
        "pose_record_count": len(records),
        "track_count": len(groups),
        "valid_observation_count": total_valid_observations,
        "trigger_count": total_trigger_count,
        "video_positive": total_trigger_count > 0,
        "streams": streams,
        "short_clip_replay": True,
        "fp_hour_reported": False,
        "main_path_unchanged": True,
        "test_pose_read": False,
        "test_evaluated": False,
    }


def _validate_decision(decision: Mapping[str, Any], inventory_path: Path) -> None:
    if decision.get("schema_version") != _DECISION_SCHEMA:
        raise ValueError("invalid SCF import decision schema_version")
    if decision.get("batch_id") != _BATCH_ID:
        raise ValueError("SCF import decision batch_id mismatch")
    if decision.get("inventory_sha256") != _sha256(inventory_path):
        raise ValueError("inventory SHA-256 mismatch")


def _validate_inventory(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("SCF inventory is empty")
    seen: set[str] = set()
    for row in rows:
        if row.get("schema_version") != _INVENTORY_SCHEMA:
            raise ValueError("invalid SCF inventory schema_version")
        if row.get("batch_id") != _BATCH_ID:
            raise ValueError("SCF inventory batch_id mismatch")
        filename = str(row.get("file_name", ""))
        match = _FILENAME_PATTERN.fullmatch(filename)
        if match is None:
            raise ValueError(f"invalid SCF filename: {filename}")
        if filename in seen:
            raise ValueError(f"duplicate SCF inventory filename: {filename}")
        seen.add(filename)
        for field in ("subject", "session", "episode", "script", "action"):
            if str(row.get(field)) != match.group(field):
                raise ValueError(f"SCF filename metadata mismatch for {filename}: {field}")
        media_path = Path(str(row.get("canonical_relative_path", "")))
        if not media_path.is_file():
            raise FileNotFoundError(media_path)
        if row.get("sha256") != _sha256(media_path):
            raise ValueError(f"SCF media SHA-256 mismatch: {filename}")


def _validated_exports(
    decision: Mapping[str, Any], expected_subjects: Sequence[str]
) -> dict[str, Path]:
    configured = decision.get("annotation_exports")
    if not isinstance(configured, Mapping):
        raise ValueError("SCF decision annotation_exports must be an object")
    if set(configured) != set(expected_subjects):
        raise ValueError("SCF annotation export subjects do not match inventory")
    output: dict[str, Path] = {}
    for subject in expected_subjects:
        entry = configured[subject]
        if not isinstance(entry, Mapping):
            raise ValueError(f"invalid SCF annotation export entry: {subject}")
        path = Path(str(entry.get("path", "")))
        if not path.is_file():
            raise FileNotFoundError(path)
        if entry.get("sha256") != _sha256(path):
            raise ValueError(f"SCF annotation export SHA-256 mismatch: {subject}")
        output[subject] = path
    return output


def _read_cvat_root(path: Path) -> ET.Element:
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"invalid SCF CVAT archive: {path}")
            payload = archive.read("annotations.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"invalid SCF CVAT archive: {path}") from exc
    return ET.fromstring(payload)


def _parse_tasks(root: ET.Element) -> dict[str, _Task]:
    task_elements = root.findall("./meta/project/tasks/task") or root.findall("./meta/task")
    if not task_elements:
        raise ValueError("SCF CVAT export contains no tasks")
    tasks: dict[str, _Task] = {}
    offset = 0
    for element in task_elements:
        task_id = _required_text(element, "id")
        source = _required_text(element, "source")
        name = _required_text(element, "name")
        if source != name or _FILENAME_PATTERN.fullmatch(source) is None:
            raise ValueError(f"invalid SCF CVAT task source: {source}")
        size = _positive_int(_required_text(element, "size"), "task size")
        original_size = element.find("original_size")
        if original_size is None:
            raise ValueError(f"SCF CVAT task lacks original_size: {source}")
        width = _positive_int(_required_text(original_size, "width"), "task width")
        height = _positive_int(_required_text(original_size, "height"), "task height")
        tasks[task_id] = _Task(task_id, source, size, width, height, offset)
        offset += size
    return tasks


def _task_for_track(track: ET.Element, tasks: Mapping[str, _Task]) -> _Task:
    task_id = track.attrib.get("task_id")
    if task_id is None and len(tasks) == 1:
        return next(iter(tasks.values()))
    if task_id not in tasks:
        raise ValueError(f"SCF CVAT track references unknown task_id: {task_id}")
    return tasks[str(task_id)]


def _action_candidate(
    track: ET.Element,
    task: _Task,
    inventory: Mapping[str, Any],
    *,
    decision: Mapping[str, Any],
    export_path: Path,
) -> dict[str, Any]:
    label = str(track.attrib.get("label", ""))
    match = _LABEL_PATTERN.fullmatch(label)
    if match is None:
        raise ValueError(f"invalid SCF action label: {label}")
    action_id = match.group("action")
    track_id = str(track.attrib.get("id", ""))
    if not track_id:
        raise ValueError("SCF CVAT track is missing id")
    boxes = _active_boxes(track, task)
    if not boxes:
        raise ValueError(f"SCF CVAT track has no active boxes: {task.source}:{track_id}")
    start_frame, start_box = boxes[0]
    end_frame, end_box = boxes[-1]
    frame_delta = int(inventory.get("cvat_frame_delta", 0))
    mapping = _frame_mapping(decision, task.source, frame_delta)
    media_width = int(inventory["width"])
    media_height = int(inventory["height"])
    rotation = int(inventory.get("rotation", 0))
    if rotation == -90 and (task.width, task.height) == (media_height, media_width):
        transform = "display_rotation_minus_90_to_encoded"
        bbox_transform = lambda box: _rotated_bbox_minus_90(  # noqa: E731
            box, display_height=task.height, inventory=inventory
        )
    else:
        scale_x = media_width / task.width
        scale_y = media_height / task.height
        if not math.isclose(scale_x, scale_y, abs_tol=1e-9):
            raise ValueError(f"non-uniform SCF coordinate scaling is unsupported: {task.source}")
        if not (math.isclose(scale_x, 1.0) or math.isclose(scale_x, 4.0)):
            raise ValueError(f"unexpected SCF coordinate scale: {task.source}:{scale_x}")
        transform = "identity" if math.isclose(scale_x, 1.0) else "scale_xy_4_4"
        bbox_transform = lambda box: _scaled_bbox(  # noqa: E731
            box, scale_x, scale_y, inventory
        )
    subject = str(inventory["subject"])
    reviewed = _is_reviewed_file(decision, task.source)
    authorized = subject in decision.get("authorization_evidence", {})
    profiled = subject in decision.get("subject_profiles", {})
    split_frozen = bool(decision.get("frozen_split_id"))
    development_split = decision.get("development_split", {})
    train_subjects = set(
        development_split.get("auxiliary_train_subjects", [])
        if isinstance(development_split, Mapping)
        else []
    )
    challenge_subjects = set(
        development_split.get("challenge_validation_subjects", [])
        if isinstance(development_split, Mapping)
        else []
    )
    loss_eligible = bool(
        reviewed
        and authorized
        and profiled
        and split_frozen
        and mapping["resolved"]
        and subject in train_subjects
    )
    canonical_reviews = decision.get("canonical_event_reviews", {})
    canonical_reviewed = bool(
        isinstance(canonical_reviews, Mapping)
        and canonical_reviews.get(action_id, {}).get("status") in {"double_reviewed", "adjudicated"}
    )
    blockers = []
    if not authorized:
        blockers.append("authorization_evidence_missing")
    if not profiled:
        blockers.append("subject_profile_missing")
    if not reviewed:
        blockers.append("formal_label_review_missing")
    if not split_frozen:
        blockers.append("split_not_frozen")
    if not mapping["resolved"]:
        blockers.append("frame_mapping_unresolved")
    if subject in challenge_subjects:
        blockers.append("challenge_only_not_loss_eligible")
    content_sha = str(inventory["sha256"])
    video_id = _video_id(task.source)
    fps = float(inventory["fps_num"]) / float(inventory["fps_den"])
    return {
        "schema_version": "self-collected-action-candidate-v1",
        "batch_id": _BATCH_ID,
        "label_id": _stable_id("scf_action", content_sha, task.task_id, track_id, action_id),
        "asset_id": _stable_id("scf_asset", content_sha),
        "video_id": video_id,
        "file_name": task.source,
        "file_path": str(inventory["canonical_relative_path"]),
        "content_sha256": content_sha,
        "source_annotation_path": _portable_path(export_path),
        "source_annotation_sha256": _sha256(export_path),
        "cvat_task_id": task.task_id,
        "cvat_track_id": _maybe_int(track_id),
        "subject_id": str(decision.get("subject_profiles", {}).get(subject, {}).get("subject_id", "unknown")),
        "provisional_subject": subject,
        "source_group_id": f"scf_mvp_v1_{subject.lower()}_provisional",
        "sample_group_id": _stable_id("scf_samplegrp", content_sha),
        "action_id": action_id,
        "action_name": match.group("name"),
        "start_frame": start_frame,
        "end_frame_exclusive": end_frame + 1,
        "start_time": round(start_frame / fps, 4),
        "end_time_exclusive": round((end_frame + 1) / fps, 4),
        "frame_index_base": 0,
        "bbox_start": bbox_transform(start_box),
        "bbox_end": bbox_transform(end_box),
        "coordinate_transform": transform,
        "frame_mapping": mapping["strategy"],
        "training_tier": (
            "auxiliary" if loss_eligible else "challenge" if subject in challenge_subjects else "ignore"
        ),
        "loss_eligible": loss_eligible,
        "canonical_event_reviewed": canonical_reviewed,
        "recovery_anchor": (
            "reviewed_action_interval_end_proxy" if canonical_reviewed else None
        ),
        "training_blockers": blockers,
        "target_status": "confirmed" if reviewed and profiled else "unknown",
    }


def _active_boxes(track: ET.Element, task: _Task) -> list[tuple[int, ET.Element]]:
    boxes = track.findall("box")
    raw_frames = [int(box.attrib["frame"]) for box in boxes]
    local_valid = all(0 <= frame < task.size for frame in raw_frames)
    global_valid = all(task.frame_offset <= frame < task.frame_offset + task.size for frame in raw_frames)
    if task.frame_offset == 0 and local_valid:
        global_valid = False
    if local_valid == global_valid:
        raise ValueError(f"ambiguous SCF CVAT frame coordinates: {task.source}")
    output = []
    for raw_frame, box in zip(raw_frames, boxes, strict=True):
        if box.attrib.get("outside") == "1":
            continue
        frame = raw_frame - task.frame_offset if global_valid else raw_frame
        output.append((frame, box))
    return sorted(output, key=lambda item: item[0])


def _frame_mapping(
    decision: Mapping[str, Any], filename: str, frame_delta: int
) -> dict[str, Any]:
    if frame_delta == 0:
        return {"strategy": "identity", "resolved": True}
    mappings = decision.get("frame_mappings", {})
    mapping = mappings.get(filename) if isinstance(mappings, Mapping) else None
    if mapping is None:
        subject = _FILENAME_PATTERN.fullmatch(filename).group("subject")  # type: ignore[union-attr]
        policies = decision.get("frame_mapping_policies", {})
        policy = policies.get(subject) if isinstance(policies, Mapping) else None
        if isinstance(policy, Mapping):
            allowed = policy.get("allowed_frame_deltas", [])
            if frame_delta in allowed and policy.get("inventory_bound_files_only") is True:
                mapping = policy
    if not isinstance(mapping, Mapping):
        raise ValueError(f"SCF video requires explicit frame mapping: {filename}")
    strategy = mapping.get("strategy")
    if strategy == "identity_with_trailing_media_frames":
        if int(mapping.get("trailing_media_frames", -1)) != frame_delta:
            raise ValueError(f"SCF trailing frame mapping mismatch: {filename}")
        return {"strategy": strategy, "resolved": True}
    if strategy == "unresolved_ignore":
        return {"strategy": strategy, "resolved": False}
    raise ValueError(f"unsupported SCF frame mapping: {filename}")


def _manifest_candidate(row: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    subject = str(row["subject"])
    content_sha = str(row["sha256"])
    duplicate_files = sorted(str(value) for value in row.get("duplicate_content_files", []))
    return {
        "schema_version": "self-collected-candidate-manifest-v1",
        "batch_id": _BATCH_ID,
        "asset_id": _stable_id("scf_asset", content_sha),
        "video_id": _video_id(str(row["file_name"])),
        "path": str(row["canonical_relative_path"]),
        "file_name": str(row["file_name"]),
        "content_sha256": content_sha,
        "fps_num": int(row["fps_num"]),
        "fps_den": int(row["fps_den"]),
        "frame_count": int(row["frame_count"]),
        "duration_sec": float(row["duration_sec"]),
        "width": int(row["width"]),
        "height": int(row["height"]),
        "rotation": int(row.get("rotation", 0)),
        "subject_id": str(decision.get("subject_profiles", {}).get(subject, {}).get("subject_id", "unknown")),
        "provisional_subject": subject,
        "source_group_id": f"scf_mvp_v1_{subject.lower()}_provisional",
        "sample_group_id": _stable_id("scf_samplegrp", content_sha),
        "duplicate_content_files": duplicate_files,
        "disposition": str(row["disposition"]),
        "training_eligible": False,
    }


def _near_fall_candidates(
    actions: Sequence[Mapping[str, Any]], inventory_by_name: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for action in actions:
        action_id = str(action["action_id"])
        if action_id not in _HARD_NEGATIVE_TYPES and action_id not in _NEAR_FALL_POSITIVE_ACTIONS:
            continue
        role = "hard_negative" if action_id in _HARD_NEGATIVE_TYPES else "positive_candidate"
        loss_eligible = bool(action["loss_eligible"])
        canonical_reviewed = bool(action.get("canonical_event_reviewed"))
        if role == "positive_candidate" and not canonical_reviewed:
            loss_eligible = False
        output.append(
            {
                "schema_version": "self-collected-near-fall-candidate-v1",
                "batch_id": _BATCH_ID,
                "candidate_id": _stable_id("scf_nearfall", action["label_id"]),
                "source_action_label_id": action["label_id"],
                "source_action_id": action_id,
                "candidate_role": role,
                "hard_negative_type": _HARD_NEGATIVE_TYPES.get(action_id),
                "canonical_event_review_required": role == "positive_candidate",
                "canonical_event_reviewed": canonical_reviewed,
                "recovery_anchor": action.get("recovery_anchor"),
                "video_id": action["video_id"],
                "file_name": action["file_name"],
                "subject_id": action["subject_id"],
                "provisional_subject": action["provisional_subject"],
                "source_group_id": action["source_group_id"],
                "sample_group_id": action["sample_group_id"],
                "start_frame": action["start_frame"],
                "end_frame_exclusive": action["end_frame_exclusive"],
                "loss_eligible": loss_eligible,
                "training_tier": (
                    "auxiliary"
                    if loss_eligible
                    else "challenge"
                    if str(action.get("provisional_subject")) in {"P05"}
                    else "ignore"
                ),
                "training_blockers": sorted(
                    set(
                        [*action["training_blockers"]]
                        + (
                            ["canonical_event_double_review_missing"]
                            if role == "positive_candidate" and not canonical_reviewed
                            else []
                        )
                    )
                ),
                "disposition": inventory_by_name[str(action["file_name"])]["disposition"],
            }
        )
    return output


def _g0_blockers(rows: Sequence[Mapping[str, Any]], decision: Mapping[str, Any]) -> list[str]:
    blockers = set()
    subjects = {str(row["subject"]) for row in rows}
    if not subjects <= set(decision.get("authorization_evidence", {})):
        blockers.add("authorization_evidence_missing")
    if not subjects <= set(decision.get("subject_profiles", {})):
        blockers.add("subject_profile_missing")
    if any(not _is_reviewed_file(decision, str(row["file_name"])) for row in rows):
        blockers.add("formal_label_review_missing")
    if not decision.get("frozen_split_id"):
        blockers.add("split_not_frozen")
    # Quarantined duplicate-condition files are excluded from the eligible
    # development subset; they remain visible in the candidate audit.
    if any(
        int(row.get("cvat_frame_delta", 0)) != 0
        and _frame_mapping(
            decision,
            str(row["file_name"]),
            int(row.get("cvat_frame_delta", 0)),
        )["resolved"]
        is False
        for row in rows
    ):
        # P03 is intentionally excluded from the first frozen development
        # split.  Unresolved frame deltas therefore do not block P01/P02/P04.
        eligible_subjects = set(decision.get("development_split", {}).get("auxiliary_train_subjects", []))
        unresolved_subjects = {
            str(row["subject"])
            for row in rows
            if int(row.get("cvat_frame_delta", 0)) != 0
        }
        if unresolved_subjects & eligible_subjects:
            blockers.add("frame_mapping_unresolved")
    return sorted(blockers)


def _is_reviewed_file(decision: Mapping[str, Any], filename: str) -> bool:
    if filename in set(decision.get("double_reviewed_files", [])):
        return True
    review_scope = decision.get("review_scope", {})
    return bool(
        isinstance(review_scope, Mapping)
        and review_scope.get("status") in {"double_reviewed", "adjudicated"}
        and review_scope.get("scope") == "all_inventory_bound_files"
        and review_scope.get("inventory_sha256") == decision.get("inventory_sha256")
    )


def _scaled_bbox(
    box: ET.Element,
    scale_x: float,
    scale_y: float,
    inventory: Mapping[str, Any],
) -> list[float]:
    values = [float(box.attrib[name]) for name in ("xtl", "ytl", "xbr", "ybr")]
    scaled = [values[0] * scale_x, values[1] * scale_y, values[2] * scale_x, values[3] * scale_y]
    if not all(math.isfinite(value) for value in scaled):
        raise ValueError("SCF CVAT box contains non-finite coordinates")
    if scaled[0] < 0 or scaled[1] < 0 or scaled[2] > int(inventory["width"]) or scaled[3] > int(inventory["height"]):
        raise ValueError(f"SCF scaled box exceeds media bounds: {inventory['file_name']}")
    if scaled[2] < scaled[0] or scaled[3] < scaled[1]:
        raise ValueError("SCF CVAT box has reversed coordinates")
    return [round(value, 2) for value in scaled]


def _rotated_bbox_minus_90(
    box: ET.Element,
    *,
    display_height: int,
    inventory: Mapping[str, Any],
) -> list[float]:
    xtl, ytl, xbr, ybr = (
        float(box.attrib[name]) for name in ("xtl", "ytl", "xbr", "ybr")
    )
    encoded = [display_height - ybr, xtl, display_height - ytl, xbr]
    if (
        not all(math.isfinite(value) for value in encoded)
        or encoded[0] < 0
        or encoded[1] < 0
        or encoded[2] > int(inventory["width"])
        or encoded[3] > int(inventory["height"])
    ):
        raise ValueError(f"SCF rotated box exceeds media bounds: {inventory['file_name']}")
    return [round(value, 2) for value in encoded]


def _video_id(filename: str) -> str:
    return f"scf_mvp_v1_{Path(filename).stem.lower()}"


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:24]}"


def _record_sort_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(row["video_id"]),
        int(row.get("start_frame", 0)),
        str(row.get("label_id", row.get("candidate_id", ""))),
    )


def _required_text(parent: ET.Element, name: str) -> str:
    value = (parent.findtext(name) or "").strip()
    if not value:
        raise ValueError(f"SCF CVAT task is missing {name}")
    return value


def _positive_int(value: str, field: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError(f"invalid SCF {field}") from exc
    if number <= 0:
        raise ValueError(f"invalid SCF {field}")
    return number


def _maybe_int(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write(
        path,
        "".join(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()
