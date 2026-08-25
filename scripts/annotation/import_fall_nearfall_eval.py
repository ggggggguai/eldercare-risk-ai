#!/usr/bin/env python3
"""Build an isolated evaluation candidate from the fall_nearfall CVAT export.

The input archive is normalized into a redacted, project-local CVAT archive and
converted through the repository's manifest-backed CVAT converter.  The output
never modifies the root fall-risk manifest, labels, or an existing split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from typing import Any

from elderly_monitoring.modules.fall_risk.annotations import convert_cvat_xml
from elderly_monitoring.modules.fall_risk.data_manifest import probe_video_metadata


BATCH_ID = "fall_nearfall_v1"
SCHEMA_VERSION = "fall-nearfall-evaluation-candidate-v1"
SUBJECT_ID = "self_collected_adult_01"
SOURCE_GROUP_ID = "fall_nearfall_v1_single_subject_session"
IDENTITY_TAGS = {"owner", "assignee", "username", "email"}
EVAL_ACTIONS = {"C03", "C04", "C05", "D01", "D02", "D03", "D05"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_source_xml(archive_path: Path) -> ET.Element:
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("annotation archive contains a corrupt member")
        xml_members = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        if len(xml_members) != 1:
            raise ValueError(f"annotation archive must contain one XML file, found {xml_members}")
        return ET.fromstring(archive.read(xml_members[0]))


def _remove_identity_nodes(parent: ET.Element) -> int:
    removed = 0
    for child in list(parent):
        if child.tag.rsplit("}", 1)[-1].lower() in IDENTITY_TAGS:
            parent.remove(child)
            removed += 1
        else:
            removed += _remove_identity_nodes(child)
    return removed


def _normalize_xml(root: ET.Element) -> tuple[bytes, dict[str, Any]]:
    removed_identity_nodes = _remove_identity_nodes(root)
    tasks = root.findall("./meta/project/tasks/task")
    if not tasks:
        raise ValueError("CVAT project contains no task metadata")
    task_names: list[str] = []
    for task in tasks:
        source = (task.findtext("source") or "").strip()
        if not source:
            raise ValueError("CVAT task is missing source")
        stem = Path(source).stem
        normalized_name = f"fall_risk__fall_nearfall__home__{stem}"
        name_element = task.find("name")
        if name_element is None:
            raise ValueError(f"CVAT task has no name: {source}")
        name_element.text = normalized_name
        task_names.append(normalized_name)

    removed_uncertain_tracks: list[dict[str, str]] = []
    for track in list(root.findall("track")):
        label = str(track.attrib.get("label", ""))
        if label.startswith("U01_"):
            removed_uncertain_tracks.append(
                {"task_id": str(track.attrib.get("task_id", "")), "track_id": str(track.attrib.get("id", "")), "label": label}
            )
            root.remove(track)

    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return payload, {
        "task_count": len(tasks),
        "removed_identity_nodes": removed_identity_nodes,
        "removed_uncertain_tracks": removed_uncertain_tracks,
        "normalized_task_names": task_names,
    }


def _write_redacted_archive(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("annotations.xml", date_time=(2020, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, payload)


def _scene_from_name(name: str) -> str:
    parts = Path(name).stem.split("_")
    return parts[2] if len(parts) >= 3 else "unknown"


def _build_manifest(root: ET.Element, media_dir: Path, annotation_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in root.findall("./meta/project/tasks/task"):
        source = (task.findtext("source") or "").strip()
        media_path = media_dir / source
        if not media_path.is_file():
            raise FileNotFoundError(media_path)
        metadata = probe_video_metadata(media_path)
        task_size = int(task.findtext("size") or "0")
        if task_size != metadata.frame_count:
            raise ValueError(
                f"frame count mismatch for {source}: CVAT={task_size} media={metadata.frame_count}"
            )
        task_original = task.find("original_size")
        if task_original is None:
            raise ValueError(f"CVAT task lacks original_size: {source}")
        task_width = int(task_original.findtext("width") or "0")
        task_height = int(task_original.findtext("height") or "0")
        if (task_width, task_height) != (metadata.width, metadata.height):
            raise ValueError(
                f"resolution mismatch for {source}: CVAT={task_width}x{task_height} "
                f"media={metadata.width}x{metadata.height}"
            )
        # The repository's CVAT converter canonicalizes identifiers to lowercase.
        video_id = Path(source).stem.lower()
        if video_id in seen:
            raise ValueError(f"duplicate video_id: {video_id}")
        seen.add(video_id)
        content_sha = _sha256(media_path)
        rows.append(
            {
                "schema_version": "fall-nearfall-evaluation-manifest-v1",
                "batch_id": BATCH_ID,
                "asset_id": f"fall_nearfall_asset_{content_sha[:24]}",
                "video_id": video_id,
                "file_name": source,
                "path": str(media_path.resolve()),
                "sha256": content_sha,
                "content_sha256": content_sha,
                "dataset": "self_collected_fall_nearfall",
                "media_type": "video",
                "modality": "rgb_video",
                "fps_num": metadata.fps_num,
                "fps_den": metadata.fps_den,
                "fps": metadata.fps,
                "frame_count": metadata.frame_count,
                "duration_sec": metadata.duration_sec,
                "width": metadata.width,
                "height": metadata.height,
                "rotation": 0,
                "subject_id": SUBJECT_ID,
                "source_group_id": SOURCE_GROUP_ID,
                "sample_group_id": f"fall_nearfall_sample_{content_sha[:24]}",
                "scene_region": _scene_from_name(source),
                "view": "fixed_camera",
                "annotation_path": annotation_path,
                "annotation_source": "cvat_manual",
                "split_partition": "test",
                "eligibility": True,
                "evaluation_only": True,
            }
        )
    return sorted(rows, key=lambda row: str(row["video_id"]))


def _ground_truth(action_labels: list[dict[str, Any]], split_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action in action_labels:
        action_id = str(action["action_id"])
        if action_id not in EVAL_ACTIONS:
            continue
        event_type = "fall" if action_id.startswith("D") else "near_fall"
        task_type = "fall_event" if event_type == "fall" else "near_fall_event"
        rows.append(
            {
                "label_id": f"eval_{action['label_id']}",
                "source_action_label_id": action["label_id"],
                "video_id": action["video_id"],
                "task_type": task_type,
                "event_type": event_type,
                "start_frame": action["start_frame"],
                "end_frame_exclusive": int(action["end_frame"]) + 1,
                "start_time": action["start_time"],
                "end_time": action["end_time"],
                "onset_time": action["start_time"],
                "split_id": split_id,
                "source": "manual_cvat_action_interval",
                "source_action_id": action_id,
                "event_subtype": action["action_name"],
                "quality": action["quality"],
                "review_status": "single_annotated",
            }
        )
    return sorted(rows, key=lambda row: (str(row["video_id"]), float(row["start_time"]), str(row["label_id"])))


def build_candidate(annotations: Path, media_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    annotations = annotations.resolve()
    media_dir = media_dir.resolve()
    original_sha = _sha256(annotations)
    source_root = _read_source_xml(annotations)
    normalized_payload, normalization = _normalize_xml(source_root)

    output_dir.mkdir(parents=True)
    redacted_path = output_dir / "source_annotations_redacted.zip"
    _write_redacted_archive(normalized_payload, redacted_path)
    try:
        annotation_rel = redacted_path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        annotation_rel = str(redacted_path)
    manifest = _build_manifest(_read_source_xml(redacted_path), media_dir, annotation_rel)
    manifest_path = output_dir / "manifest.jsonl"
    _jsonl(manifest_path, manifest)

    converted = convert_cvat_xml(
        redacted_path,
        manifest_path=manifest_path,
        fps=None,
        labeler="fall_nearfall_eval_manual_import",
    )
    action_labels = converted.action_labels
    event_labels = converted.event_labels
    split_id = "fall-nearfall-v1-test-candidate"
    ground_truth = _ground_truth(action_labels, split_id)

    action_path = output_dir / "action_labels.jsonl"
    event_path = output_dir / "event_labels.jsonl"
    truth_path = output_dir / "ground_truth_events.jsonl"
    _jsonl(action_path, action_labels)
    _jsonl(event_path, event_labels)
    _jsonl(truth_path, ground_truth)

    assignments = [
        {
            "assignment_id": f"assignment_{row['video_id']}",
            "video_id": row["video_id"],
            "asset_id": row["asset_id"],
            "partition": "test",
            "split_id": split_id,
            "subject_id": row["subject_id"],
            "source_group_id": row["source_group_id"],
            "content_sha256": row["content_sha256"],
            "sample_group_id": row["sample_group_id"],
        }
        for row in manifest
    ]
    assignments_path = output_dir / "assignments.jsonl"
    _jsonl(assignments_path, assignments)
    split = {
        "schema_version": "fall-nearfall-evaluation-split-v1",
        "split_id": split_id,
        "protocol_status": "development_provisional",
        "partition": "test",
        "assignment_count": len(assignments),
        "video_count": len(manifest),
        "manifest_sha256": _sha256(manifest_path),
        "assignments_sha256": _sha256(assignments_path),
        "ground_truth_sha256": _sha256(truth_path),
        "source_archive_sha256": original_sha,
        "redacted_annotation_sha256": _sha256(redacted_path),
        "leakage_policy": "all videos are one subject/session; no train or validation assignments",
    }
    split_path = output_dir / "split.json"
    split_path.write_text(json.dumps(split, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    report = {
        "schema_version": SCHEMA_VERSION,
        "batch_id": BATCH_ID,
        "status": "candidate_built_not_formal_frozen",
        "source": {
            "archive_path": str(annotations),
            "archive_sha256": original_sha,
            "media_dir": str(media_dir),
            "media_copied_into_repository": False,
        },
        "normalization": normalization,
        "counts": {
            "videos": len(manifest),
            "unique_video_content": len({row["content_sha256"] for row in manifest}),
            "action_tracks": len(action_labels),
            "mapped_event_rows": len(event_labels),
            "ground_truth_events": len(ground_truth),
            "fall_events": sum(row["event_type"] == "fall" for row in ground_truth),
            "near_fall_events": sum(row["event_type"] == "near_fall" for row in ground_truth),
            "negative_videos": len(manifest) - len({row["video_id"] for row in ground_truth}),
        },
        "ground_truth_policy": {
            "fall_actions": ["D01", "D02", "D03", "D05"],
            "near_fall_actions": ["C03", "C04", "C05"],
            "excluded_from_event_truth": ["U01", "D04"],
            "d04_policy": "post-fall static continuation; not a second fall event",
            "unannotated_intervals": "not inferred as negative events",
        },
        "artifacts": {
            "manifest": manifest_path.name,
            "action_labels": action_path.name,
            "event_labels": event_path.name,
            "ground_truth_events": truth_path.name,
            "assignments": assignments_path.name,
            "split": split_path.name,
            "redacted_annotations": redacted_path.name,
        },
        "formal_evaluation_allowed": False,
        "blockers": [
            "single_adult_single_home_session_cannot_establish_cross_subject_generalization",
            "labels_are_single_annotated_not_double_reviewed",
            "evaluation_protocol_is_development_provisional",
            "using_this_bundle_for_threshold_or_model_tuning_invalidates_independent_test_claim",
        ],
    }
    report_path = output_dir / "import_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_candidate(args.annotations, args.media_dir, args.output_dir)
    print(json.dumps(report["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
