"""Fail-closed intake for multi-task CVAT wandering project exports."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    load_camera_inputs,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_import import (
    CameraEpisodeImportError,
    import_cvat_episode_tracks,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_boundary import (
    build_camera_episode_boundary_development_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    load_camera_config,
)


CVAT_PROJECT_IMPORT_SUMMARY_SCHEMA_VERSION = (
    "wandering-camera-cvat-project-import-summary-v1"
)
CVAT_PROJECT_TRACK_ALIGNMENT_SCHEMA_VERSION = (
    "wandering-camera-cvat-track-alignment-v1"
)
TRACK_ALIGNMENT_POLICY_ID = "single-subject-temporal-dominance-v1"


class CameraCvatProjectError(ValueError):
    """Project XML, media, tracking, or identity intake failed closed."""


@dataclass(frozen=True)
class CameraCvatProjectBuildResult:
    output_dir: Path
    task_count: int
    episode_count: int
    aligned_episode_count: int
    low_coverage_episode_count: int
    geometry_mismatch_episode_count: int


@dataclass(frozen=True)
class _ProjectTask:
    task_id: str
    source_video_id: str
    source_name: str
    frame_count: int
    frame_offset: int
    tracks: tuple[ET.Element, ...]


def build_cvat_project_import_bundle(
    *,
    project_root: str | Path,
    cvat_xml_path: str | Path,
    video_root: str | Path,
    tracking_output_root: str | Path,
    output_dir: str | Path,
    participant_id: str,
    session_id: str,
    clock_domain_id: str,
    minimum_temporal_coverage: float = 0.8,
    geometry_iou_threshold: float = 0.1,
) -> CameraCvatProjectBuildResult:
    """Split a CVAT project export and derive auditable machine-track bindings.

    The CVAT rectangles are audited but never used to choose a machine Track ID.
    Selection uses only tracking observation coverage inside each independently
    annotated temporal interval under the declared single-subject video policy.
    """

    output = Path(output_dir).resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"CVAT project import output already exists: {output}")
    root = Path(project_root).resolve(strict=True)
    xml_path = Path(cvat_xml_path).resolve(strict=True)
    videos = Path(video_root).resolve(strict=True)
    tracking_root = Path(tracking_output_root).resolve(strict=True)
    for value, name in (
        (minimum_temporal_coverage, "minimum_temporal_coverage"),
        (geometry_iou_threshold, "geometry_iou_threshold"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise CameraCvatProjectError(f"{name} must be within [0, 1]")
    participant = _token(participant_id, "participant_id")
    session = _token(session_id, "session_id")
    clock = _token(clock_domain_id, "clock_domain_id")
    try:
        xml_root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise CameraCvatProjectError("cannot parse CVAT project XML") from exc
    tasks = _load_project_tasks(xml_root)
    camera_config = load_camera_config(root / "configs/modules/wandering_camera_v1.yaml")
    profile_path = (
        root
        / "configs/modules/wandering_camera_episode_boundary_home_v1.yaml"
    ).resolve(strict=True)
    trusted_minimum_track_confidence = 0.70

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    alignments: list[dict[str, Any]] = []
    boundary_index: list[dict[str, str]] = []
    source_hashes: dict[str, str] = {}
    episode_count = 0
    try:
        for task in tasks:
            source_video_id = task.source_video_id
            video_path = (videos / task.source_name).resolve(strict=True)
            source_hash = _sha256_file(video_path)
            source_hashes[source_video_id] = source_hash
            source_bundle = tracking_root / source_video_id
            tracking_path = (source_bundle / "inputs/tracking.jsonl").resolve(
                strict=True
            )
            sidecar_path = (source_bundle / "inputs/media_sidecar.json").resolve(
                strict=True
            )
            try:
                adapter = load_camera_inputs(
                    tracking_path,
                    sidecar_path,
                    camera_config,
                )
            except CameraAdapterError as exc:
                raise CameraCvatProjectError(
                    f"invalid tracking/media pair for {source_video_id}: {exc}"
                ) from exc
            media = dict(adapter.media_sidecar)
            if media["source_video_id"] != source_video_id:
                raise CameraCvatProjectError("CVAT task and sidecar video identity differ")
            if media["source_sha256"] != source_hash:
                raise CameraCvatProjectError("video SHA-256 differs from media sidecar")
            expected_frames = int(
                round(float(media["nominal_fps"]) * float(media["duration_sec"]))
            )
            if expected_frames != task.frame_count:
                raise CameraCvatProjectError(
                    f"CVAT task size differs from media sidecar for {source_video_id}"
                )
            normalized_rows = [
                dict(row)
                for row in adapter.normalized_rows
                if float(row["track_confidence"])
                >= trusted_minimum_track_confidence
            ]
            if not normalized_rows:
                raise CameraCvatProjectError(
                    f"no tracking observation reaches trusted confidence for {source_video_id}"
                )
            task_alignments = _align_task_tracks(
                task,
                normalized_rows,
                minimum_temporal_coverage=float(minimum_temporal_coverage),
                geometry_iou_threshold=float(geometry_iou_threshold),
            )
            target_track_ids = {
                str(row["cvat_track_id"]): int(row["target_track_id"])
                for row in task_alignments
            }
            try:
                imported = import_cvat_episode_tracks(
                    task.tracks,
                    media,
                    target_track_ids=target_track_ids,
                    video_frame_count=task.frame_count,
                    frame_offset=task.frame_offset,
                )
            except CameraEpisodeImportError as exc:
                raise CameraCvatProjectError(
                    f"CVAT task import failed for {source_video_id}: {exc}"
                ) from exc
            episode_count += len(imported.boundaries)
            task_dir = stage / "tasks" / source_video_id
            final_task_dir = output / "tasks" / source_video_id
            task_dir.mkdir(parents=True)
            boundary_path = task_dir / "episode_boundaries.jsonl"
            truth_path = task_dir / "episode_truth.jsonl"
            alignment_path = task_dir / "track_alignment.jsonl"
            labeled_sidecar_path = task_dir / "media_sidecar.labeled.json"
            boundary_path.write_bytes(canonical_jsonl_bytes(imported.boundaries))
            truth_path.write_bytes(canonical_jsonl_bytes(imported.truth_records))
            alignment_path.write_bytes(canonical_jsonl_bytes(task_alignments))
            labeled_sidecar = {
                **media,
                "authorization_status": "authorized_camera_labeled_evaluation",
            }
            labeled_sidecar_path.write_bytes(canonical_json_bytes(labeled_sidecar))
            proposal_dir = task_dir / "proposals"
            build_camera_episode_boundary_development_bundle(
                project_root=root,
                development_profile_path=profile_path,
                tracking_jsonl_path=tracking_path,
                media_sidecar_path=labeled_sidecar_path,
                output_dir=proposal_dir,
            )
            task_summary = {
                "schema_version": "wandering-camera-cvat-project-task-import-v1",
                "source_video_id": source_video_id,
                "cvat_task_id": task.task_id,
                "cvat_frame_offset": task.frame_offset,
                "video_frame_count": task.frame_count,
                "episode_count": len(imported.boundaries),
                "target_track_binding_policy_id": TRACK_ALIGNMENT_POLICY_ID,
                "cvat_box_geometry_consumed_for_target_binding": False,
                "model_predictions_consumed": False,
                "annotation_authorization_status": (
                    "authorized_camera_labeled_evaluation"
                ),
                "proposal_regenerated_with_labeled_sidecar": True,
                "trusted_minimum_track_confidence": (
                    trusted_minimum_track_confidence
                ),
                "source_video_sha256": source_hash,
                "source_tracking_sha256": adapter.source_tracking_sha256,
                "source_media_sidecar_sha256": _sha256_file(sidecar_path),
            }
            (task_dir / "import_summary.json").write_bytes(
                canonical_json_bytes(task_summary)
            )
            alignments.extend(task_alignments)
            boundary_index.append(
                {
                    "bundle_id": f"home-{source_video_id}",
                    "proposal_bundle_dir": (
                        final_task_dir / "proposals"
                    ).as_posix(),
                    "human_boundary_jsonl": (
                        final_task_dir / "episode_boundaries.jsonl"
                    ).as_posix(),
                    "media_sidecar": (
                        final_task_dir / "media_sidecar.labeled.json"
                    ).as_posix(),
                    "participant_id": participant,
                    "session_id": session,
                    "camera_setup_id": str(media["setup_id"]),
                    "clock_domain_id": clock,
                    "truth_source": "independent_human",
                }
            )

        if len(alignments) != episode_count:
            raise CameraCvatProjectError("alignment and imported episode counts differ")
        alignment_payload = canonical_jsonl_bytes(alignments)
        index_payload = canonical_jsonl_bytes(boundary_index)
        (stage / "track_alignment.jsonl").write_bytes(alignment_payload)
        (stage / "boundary_evaluation_batch_index.jsonl").write_bytes(index_payload)
        geometry_values = [
            float(row["median_cvat_machine_iou"])
            for row in alignments
            if row["median_cvat_machine_iou"] is not None
        ]
        coverage_values = [float(row["temporal_coverage_ratio"]) for row in alignments]
        low_coverage_episode_count = sum(
            not bool(row["temporal_coverage_gate_satisfied"])
            for row in alignments
        )
        summary = {
            "schema_version": CVAT_PROJECT_IMPORT_SUMMARY_SCHEMA_VERSION,
            "status": (
                "labeled_development_intake_uncertain_track_alignment"
                if low_coverage_episode_count
                else "labeled_development_intake_ready_with_geometry_limitation"
            ),
            "validation_scope": "home_labeled_development",
            "task_count": len(tasks),
            "episode_count": episode_count,
            "manual_track_count": episode_count,
            "all_attributes_confirmed": True,
            "target_track_binding": {
                "policy_id": TRACK_ALIGNMENT_POLICY_ID,
                "single_subject_video_assumption": True,
                "cvat_box_geometry_consumed": False,
                "minimum_temporal_coverage": float(minimum_temporal_coverage),
                "aligned_episode_count": sum(
                    bool(row["temporal_coverage_gate_satisfied"])
                    for row in alignments
                ),
                "low_coverage_episode_count": low_coverage_episode_count,
                "coverage_min": min(coverage_values),
                "coverage_median": statistics.median(coverage_values),
                "coverage_mean": statistics.fmean(coverage_values),
            },
            "cvat_box_geometry_audit": {
                "threshold": float(geometry_iou_threshold),
                "episode_count": len(geometry_values),
                "mismatch_episode_count": sum(
                    value < float(geometry_iou_threshold)
                    for value in geometry_values
                ),
                "median_iou": statistics.median(geometry_values),
                "mean_iou": statistics.fmean(geometry_values),
                "person_bbox_alignment_claimed": False,
            },
            "annotation_label_counts": _annotation_label_counts(tasks),
            "model_predictions_consumed": False,
            "original_media_or_annotations_modified": False,
            "source_cvat_xml": {
                "path": xml_path.as_posix(),
                "sha256": _sha256_file(xml_path),
            },
            "source_video_count": len(source_hashes),
            "segmenter_profile_id": "home-recall-confidence070-v1",
            "trusted_minimum_track_confidence": trusted_minimum_track_confidence,
            "artifacts": {
                "track_alignment.jsonl": _artifact_descriptor(alignment_payload),
                "boundary_evaluation_batch_index.jsonl": _artifact_descriptor(
                    index_payload
                ),
            },
        }
        summary_payload = canonical_json_bytes(summary)
        (stage / "project_summary.json").write_bytes(summary_payload)
        _fsync_tree(stage)
        stage.replace(output)
        return CameraCvatProjectBuildResult(
            output_dir=output,
            task_count=len(tasks),
            episode_count=episode_count,
            aligned_episode_count=int(
                summary["target_track_binding"]["aligned_episode_count"]
            ),
            low_coverage_episode_count=int(
                summary["target_track_binding"]["low_coverage_episode_count"]
            ),
            geometry_mismatch_episode_count=int(
                summary["cvat_box_geometry_audit"]["mismatch_episode_count"]
            ),
        )
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _load_project_tasks(root: ET.Element) -> list[_ProjectTask]:
    if root.tag != "annotations" or root.findtext("./version") != "1.1":
        raise CameraCvatProjectError("CVAT XML must use annotations version 1.1")
    task_nodes = root.findall("./meta/project/tasks/task")
    if not task_nodes:
        raise CameraCvatProjectError("CVAT XML is not a project export")
    tracks_by_task: dict[str, list[ET.Element]] = defaultdict(list)
    for track in root.findall("./track"):
        task_id = track.get("task_id")
        if not task_id:
            raise CameraCvatProjectError("project track has no task_id")
        tracks_by_task[task_id].append(track)
    tasks: list[_ProjectTask] = []
    seen_ids: set[str] = set()
    seen_sources: set[str] = set()
    offset = 0
    for task_node in task_nodes:
        task_id = _token(task_node.findtext("./id"), "cvat task id")
        source_name = task_node.findtext("./source") or task_node.findtext("./name")
        if not source_name or Path(source_name).name != source_name:
            raise CameraCvatProjectError("CVAT task source must be a file name")
        source_video_id = _token(Path(source_name).stem, "source_video_id")
        try:
            frame_count = int(task_node.findtext("./size", ""))
        except ValueError as exc:
            raise CameraCvatProjectError("CVAT task size is invalid") from exc
        if frame_count <= 0:
            raise CameraCvatProjectError("CVAT task size must be positive")
        if task_id in seen_ids or source_video_id in seen_sources:
            raise CameraCvatProjectError("CVAT task ID or source is duplicated")
        seen_ids.add(task_id)
        seen_sources.add(source_video_id)
        tracks = tuple(tracks_by_task.pop(task_id, []))
        if not tracks:
            raise CameraCvatProjectError("every CVAT project task must contain tracks")
        if any(track.get("label") != "wandering_episode" for track in tracks):
            raise CameraCvatProjectError("CVAT project contains a non-episode track")
        if any(track.get("source") != "manual" for track in tracks):
            raise CameraCvatProjectError("CVAT project contains a non-manual track")
        tasks.append(
            _ProjectTask(
                task_id=task_id,
                source_video_id=source_video_id,
                source_name=source_name,
                frame_count=frame_count,
                frame_offset=offset,
                tracks=tracks,
            )
        )
        offset += frame_count
    if tracks_by_task:
        raise CameraCvatProjectError("CVAT track references an unknown task_id")
    return tasks


def _align_task_tracks(
    task: _ProjectTask,
    tracking_rows: Sequence[Mapping[str, Any]],
    *,
    minimum_temporal_coverage: float,
    geometry_iou_threshold: float,
) -> list[dict[str, Any]]:
    frames_by_track: dict[int, set[int]] = defaultdict(set)
    boxes_by_frame_track: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    for row in tracking_rows:
        frame = int(row["frame_id"])
        track = int(row["track_id"])
        frames_by_track[track].add(frame)
        bbox = tuple(float(value) for value in row["bbox"])
        boxes_by_frame_track[(frame, track)] = bbox  # type: ignore[assignment]
    if not frames_by_track:
        raise CameraCvatProjectError("tracking contains no machine tracks")
    output: list[dict[str, Any]] = []
    for track in task.tracks:
        cvat_track_id = _token(track.get("id"), "cvat_track_id")
        start, end = _normalized_interval(track, task)
        support = end - start
        scores = sorted(
            (
                (len(frames.intersection(range(start, end))), track_id)
                for track_id, frames in frames_by_track.items()
            ),
            key=lambda value: (-value[0], value[1]),
        )
        selected_count, selected_track = scores[0]
        runner_up_count = scores[1][0] if len(scores) > 1 else 0
        coverage = selected_count / support
        ious: list[float] = []
        for box in track.findall("./box"):
            if box.get("outside", "0") == "1":
                continue
            frame = _int_attribute(box, "frame") - task.frame_offset
            machine_bbox = boxes_by_frame_track.get((frame, selected_track))
            if machine_bbox is None:
                continue
            human_bbox = tuple(
                float(box.get(field, "nan"))
                for field in ("xtl", "ytl", "xbr", "ybr")
            )
            if all(math.isfinite(value) for value in human_bbox):
                ious.append(_bbox_iou(human_bbox, machine_bbox))
        median_iou = statistics.median(ious) if ious else None
        output.append(
            {
                "schema_version": CVAT_PROJECT_TRACK_ALIGNMENT_SCHEMA_VERSION,
                "policy_id": TRACK_ALIGNMENT_POLICY_ID,
                "source_video_id": task.source_video_id,
                "cvat_task_id": task.task_id,
                "cvat_track_id": cvat_track_id,
                "target_track_id": selected_track,
                "start_frame": start,
                "end_frame_exclusive": end,
                "interval_frame_count": support,
                "selected_tracking_observation_count": selected_count,
                "runner_up_tracking_observation_count": runner_up_count,
                "temporal_coverage_ratio": coverage,
                "temporal_coverage_gate_satisfied": (
                    coverage >= minimum_temporal_coverage
                ),
                "cvat_geometry_sample_count": len(ious),
                "median_cvat_machine_iou": median_iou,
                "cvat_geometry_consistent": (
                    median_iou is not None and median_iou >= geometry_iou_threshold
                ),
                "cvat_box_geometry_consumed_for_target_binding": False,
                "model_predictions_consumed": False,
            }
        )
    return output


def _normalized_interval(track: ET.Element, task: _ProjectTask) -> tuple[int, int]:
    boxes = list(track.findall("./box"))
    if not boxes:
        raise CameraCvatProjectError("CVAT track contains no boxes")
    parsed = sorted(
        (
            _int_attribute(box, "frame") - task.frame_offset,
            box.get("outside", "0") == "1",
        )
        for box in boxes
    )
    if parsed[0][0] < 0:
        raise CameraCvatProjectError("CVAT frame precedes cumulative task offset")
    visible = [frame for frame, outside in parsed if not outside]
    if not visible:
        raise CameraCvatProjectError("CVAT track has no visible frames")
    outside = [frame for frame, is_outside in parsed if is_outside]
    start = visible[0]
    end = min(outside) if outside else visible[-1] + 1
    if not 0 <= start < end <= task.frame_count:
        raise CameraCvatProjectError("normalized CVAT interval exceeds task size")
    if any(frame >= end for frame in visible):
        raise CameraCvatProjectError("visible CVAT frame occurs after outside")
    return start, end


def _annotation_label_counts(tasks: Sequence[_ProjectTask]) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for task in tasks:
        for track in task.tracks:
            visible = [
                box
                for box in track.findall("./box")
                if box.get("outside", "0") != "1"
                and box.findall("./attribute")
            ]
            if not visible:
                raise CameraCvatProjectError("CVAT track attributes are missing")
            attributes = {
                str(item.get("name")): item.text or ""
                for item in visible[0].findall("./attribute")
            }
            if any(value == "not_set" for value in attributes.values()):
                raise CameraCvatProjectError("CVAT track contains not_set attributes")
            for name in (
                "observable_pattern",
                "purpose_context",
                "evaluation_role",
                "script_type",
                "visibility_quality",
                "tracking_issue",
            ):
                counts[name][attributes.get(name, "<missing>")] += 1
    return {
        name: dict(sorted(counter.items()))
        for name, counter in sorted(counts.items())
    }


def _bbox_iou(
    left: Sequence[float], right: Sequence[float]
) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - overlap
    return overlap / union if union > 0.0 else 0.0


def _artifact_descriptor(payload: bytes) -> dict[str, Any]:
    return {
        "byte_count": len(payload),
        "record_count": len(payload.splitlines()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int_attribute(element: ET.Element, name: str) -> int:
    try:
        value = int(element.get(name, ""))
    except ValueError as exc:
        raise CameraCvatProjectError(f"CVAT {name} is invalid") from exc
    if value < 0:
        raise CameraCvatProjectError(f"CVAT {name} must be non-negative")
    return value


def _token(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 160
        or any(char in value for char in "\\/\r\n\0")
    ):
        raise CameraCvatProjectError(f"{field} must be a safe non-empty token")
    return value


def _fsync_tree(root: Path) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())


__all__ = [
    "CVAT_PROJECT_IMPORT_SUMMARY_SCHEMA_VERSION",
    "CVAT_PROJECT_TRACK_ALIGNMENT_SCHEMA_VERSION",
    "TRACK_ALIGNMENT_POLICY_ID",
    "CameraCvatProjectBuildResult",
    "CameraCvatProjectError",
    "build_cvat_project_import_bundle",
]
