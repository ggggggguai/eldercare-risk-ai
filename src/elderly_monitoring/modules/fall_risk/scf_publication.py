"""Publish the reviewed SCF_MVP_V1 training subset into the v2 root contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from elderly_monitoring.modules.fall_risk.annotations import ACTION_EVENT_MAP, ACTION_NAMES


SCF_BATCH_ID = "SCF_MVP_V1"
SCF_PUBLICATION_SCHEMA = "self-collected-scf-mvp-v1-root-publication-v1"


def publish_scf_mvp_v1_training_batch(
    *,
    candidate_dir: Path | str,
    decision_path: Path | str,
    output_dir: Path | str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert the approved SCF auxiliary-train subset to standard v2 rows.

    This intentionally publishes action labels only. The short clips contain
    proxy recovery anchors, not canonical near-fall event truth, so event labels
    remain out of this root publication until separately reviewed.
    """
    candidate = Path(candidate_dir)
    decision_file = Path(decision_path)
    output = Path(output_dir)
    if not candidate.is_dir():
        raise FileNotFoundError(candidate)
    decision = _read_json(decision_file)
    if decision.get("batch_id") != SCF_BATCH_ID:
        raise ValueError("SCF decision batch_id mismatch")
    split = decision.get("development_split")
    if not isinstance(split, Mapping):
        raise ValueError("SCF decision lacks development_split")
    train_subjects = {str(value) for value in split.get("auxiliary_train_subjects", [])}
    if not train_subjects:
        raise ValueError("SCF decision has no auxiliary train subjects")

    import_report = _read_json(candidate / "import_report.json")
    if import_report.get("batch_id") != SCF_BATCH_ID:
        raise ValueError("SCF candidate import report batch_id mismatch")
    if import_report.get("g0", {}).get("status") != "ready":
        raise ValueError("SCF candidate G0 is not ready")
    actions = _read_jsonl(candidate / "action_labels.jsonl")
    manifests = _read_jsonl(candidate / "manifest.jsonl")
    manifest_by_video = {str(row["video_id"]): row for row in manifests}
    selected = [
        row
        for row in actions
        if bool(row.get("loss_eligible"))
        and str(row.get("provisional_subject")) in train_subjects
    ]
    if not selected:
        raise ValueError("SCF publication selection is empty")
    if any(str(row.get("training_tier")) != "auxiliary" for row in selected):
        raise ValueError("SCF selected rows must be auxiliary training labels")
    if any(str(row.get("provisional_subject")) == "P03" for row in selected):
        raise ValueError("P03 labels cannot enter the SCF training publication")
    if any(str(row.get("provisional_subject")) == "P05" for row in selected):
        raise ValueError("P05 challenge labels cannot enter the SCF training publication")

    selected_video_ids = {str(row["video_id"]) for row in selected}
    missing = sorted(selected_video_ids - set(manifest_by_video))
    if missing:
        raise ValueError(f"SCF selected labels lack manifest rows: {missing[:5]}")
    standard_manifest = [
        _manifest_row(
            manifest_by_video[video_id],
            decision,
            decision_path=decision_file,
            decision_sha256=_sha256(decision_file),
        )
        for video_id in sorted(selected_video_ids)
    ]
    standard_actions = [
        _action_row(row, manifest_by_video[str(row["video_id"])]) for row in selected
    ]
    standard_actions.sort(key=lambda row: (str(row["video_id"]), int(row["start_frame"]), str(row["label_id"])))

    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "manifest.jsonl", standard_manifest)
    _write_jsonl(output / "action_labels.jsonl", standard_actions)
    (output / "event_labels.jsonl").write_text("", encoding="utf-8")
    publication = {
        "schema_version": SCF_PUBLICATION_SCHEMA,
        "batch_id": SCF_BATCH_ID,
        "publication_status": "accepted_for_v2_publication",
        "source_candidate_dir": candidate.as_posix(),
        "source_candidate_sha256": {
            "action_labels": _sha256(candidate / "action_labels.jsonl"),
            "manifest": _sha256(candidate / "manifest.jsonl"),
            "import_report": _sha256(candidate / "import_report.json"),
        },
        "selection": {
            "auxiliary_train_subjects": sorted(train_subjects),
            "excluded_subjects": sorted(
                {str(value) for value in split.get("excluded_subjects", [])}
            ),
            "challenge_subjects": sorted(
                {str(value) for value in split.get("challenge_validation_subjects", [])}
            ),
            "selected_action_labels": len(standard_actions),
            "selected_videos": len(standard_manifest),
        },
        "event_labels_published": 0,
        "root_manifest_modified": True,
        "root_v2_labels_modified": True,
        "root_v3_labels_modified": False,
    }
    _write_json(output / "import_report.json", publication)
    return publication


def _manifest_row(
    row: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    decision_path: Path,
    decision_sha256: str,
) -> dict[str, Any]:
    subject = str(row["provisional_subject"])
    annotation_path = (
        f"data/annotations/fall_risk/cvat_exports/raw/"
        f"self_collected_scf_mvp_v1/{subject}_cvat_redacted.zip"
    )
    return {
        "annotation_path": annotation_path,
        "asset_id": str(row["asset_id"]),
        "consent_id": None,
        "consent_status": "not_recorded",
        "collection_decided_at": str(decision["decided_at"]),
        "collection_decided_by": str(decision["decided_by"]),
        "collection_decision_id": str(decision["decision_id"]),
        "collection_decision_path": decision_path.as_posix(),
        "collection_decision_sha256": decision_sha256,
        "collection_status": "project_self_collected",
        "dataset": "self_collected_scf",
        "duplicate_group_id": None,
        "duplicate_of_asset_id": None,
        "duration_sec": float(row["duration_sec"]),
        "eligibility": True,
        "exclusion_reasons": [],
        "fps": float(row["fps_num"]) / float(row["fps_den"]),
        "fps_den": int(row["fps_den"]),
        "fps_num": int(row["fps_num"]),
        "frame_count": int(row["frame_count"]),
        "height": int(row["height"]),
        "label_source": "cvat_manual",
        "media_type": "video",
        "modality": "rgb_video",
        "original_event_id": str(row["video_id"]),
        "path": str(row["path"]),
        "provenance_status": "project_collected_training_authorized",
        "redistribution_use": "not_authorized_by_this_decision",
        "scene_region": "self_collected_home",
        "sha256": str(row["content_sha256"]),
        "split_partition": "train",
        "split_policy_id": str(decision["frozen_split_id"]),
        "training_tier_cap": "auxiliary",
        "gait_training_role": "action_pretraining_and_walking_gate_only",
        "source_group_id": str(row["source_group_id"]),
        "source_uri": "internal://collection/SCF_MVP_V1",
        "subject_grouping_status": "pseudonymous_project_subject",
        "subject_id": str(row["subject_id"]),
        "subset": subject,
        "training_use": "authorized",
        "video_id": str(row["video_id"]),
        "view": "fixed_camera",
        "width": int(row["width"]),
    }


def _action_row(row: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    action_id = str(row["action_id"])
    event_type, _severity = ACTION_EVENT_MAP[action_id]
    end_frame = int(row["end_frame_exclusive"]) - 1
    fps = float(manifest["fps_num"]) / float(manifest["fps_den"])
    end_time = end_frame / fps
    source_export_id = f"scf_{str(row['provisional_subject']).lower()}_{_sha256_text(str(row['source_annotation_sha256']))[:12]}"
    return {
        "action_id": action_id,
        "action_name": ACTION_NAMES[action_id],
        "asset_id": str(row["asset_id"]),
        "bbox_end": list(row["bbox_end"]),
        "bbox_start": list(row["bbox_start"]),
        "cvat_task_id": str(row["cvat_task_id"]),
        "cvat_track_id": row["cvat_track_id"],
        "end_frame": end_frame,
        "end_time": round(end_time, 6),
        "event_type": event_type,
        "file_path": str(row["file_path"]),
        "frame_index_base": 0,
        "label_id": str(row["label_id"]),
        "labeler": "scf_mvp_v1_double_reviewed",
        "note": "",
        "quality": "clear",
        "scene": "self_collected_home",
        "source": "cvat",
        "source_annotation_path": str(row["source_annotation_path"]),
        "source_annotation_sha256": str(row["source_annotation_sha256"]),
        "source_export_id": source_export_id,
        "source_record_id": f"{source_export_id}:task:{row['cvat_task_id']}:track:{row['cvat_track_id']}",
        "start_frame": int(row["start_frame"]),
        "start_time": float(row["start_time"]),
        "subject_id": str(row["subject_id"]),
        "video_id": str(row["video_id"]),
        "view": "fixed_camera",
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSON object expected: {path}:{line_number}")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
