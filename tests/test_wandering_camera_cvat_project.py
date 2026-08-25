from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from elderly_monitoring.modules.mental_health.wandering.camera_cvat_project import (
    CameraCvatProjectError,
    build_cvat_project_import_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


def _attributes(pattern: str = "direct", role: str = "ordinary_negative") -> str:
    purpose = "nonpurposeful"
    evidence = "scripted"
    return f"""
      <attribute name="observable_pattern">{pattern}</attribute>
      <attribute name="purpose_context">{purpose}</attribute>
      <attribute name="purpose_evidence">{evidence}</attribute>
      <attribute name="evaluation_role">{role}</attribute>
      <attribute name="script_type">shape_prompt</attribute>
      <attribute name="visibility_quality">good</attribute>
      <attribute name="tracking_issue">not_checked</attribute>
      <attribute name="note"></attribute>
    """


def _project_xml(*, second_start_frame: int = 4) -> str:
    attrs = _attributes()
    return f"""<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><project><tasks>
    <task><id>10</id><name>home_a.mp4</name><size>4</size><source>home_a.mp4</source></task>
    <task><id>11</id><name>home_b.mp4</name><size>3</size><source>home_b.mp4</source></task>
  </tasks></project></meta>
  <track id="cvat-a" label="wandering_episode" source="manual" task_id="10">
    <box frame="0" outside="0" xtl="50" ytl="50" xbr="60" ybr="60">{attrs}</box>
    <box frame="1" outside="0" xtl="50" ytl="50" xbr="60" ybr="60" />
    <box frame="2" outside="1" xtl="50" ytl="50" xbr="60" ybr="60" />
  </track>
  <track id="cvat-b" label="wandering_episode" source="manual" task_id="11">
    <box frame="{second_start_frame}" outside="0" xtl="1" ytl="1" xbr="11" ybr="21">{attrs}</box>
    <box frame="{second_start_frame + 1}" outside="0" xtl="1" ytl="1" xbr="11" ybr="21" />
    <box frame="{second_start_frame + 2}" outside="1" xtl="1" ytl="1" xbr="11" ybr="21" />
  </track>
</annotations>
"""


def _tracking_row(frame: int, track_id: int) -> dict[str, object]:
    return {
        "frame_id": frame,
        "person_id": f"temporary_track_{track_id:03d}",
        "track_id": track_id,
        "bbox": [1.0, 1.0, 11.0, 21.0],
        "scene_region": "unknown",
        "track_confidence": 0.9,
        "center": [6.0, 11.0],
        "speed_px_per_sec": None if frame == 0 else 0.0,
        "timestamp_sec": frame / 15.0,
    }


def _write_video_pair(root: Path, video_id: str, frame_count: int) -> None:
    video_root = root / "videos"
    output_root = root / "tracking"
    video_root.mkdir(exist_ok=True)
    video_path = video_root / f"{video_id}.mp4"
    video_path.write_bytes(f"fixture-{video_id}".encode("utf-8"))
    bundle = output_root / video_id
    inputs = bundle / "inputs"
    inputs.mkdir(parents=True)
    (bundle / "proposals").mkdir()
    rows = [_tracking_row(frame, 1) for frame in range(frame_count)]
    tracking_payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for row in rows
    )
    tracking_path = inputs / "tracking.jsonl"
    tracking_path.write_bytes(tracking_payload)
    sidecar = {
        "schema_version": "wandering-media-v1",
        "source_video_id": video_id,
        "source_group_id": "home-fixture",
        "device_id": "camera-fixture",
        "setup_id": "setup-fixture",
        "stream_epoch": "full-recording",
        "media_ref": f"external/{video_id}.mp4",
        "source_sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
        "tracking_jsonl_sha256": hashlib.sha256(tracking_payload).hexdigest(),
        "video_width": 100,
        "video_height": 100,
        "nominal_fps": 15.0,
        "duration_sec": frame_count / 15.0,
        "capture_started_at": None,
        "timezone": None,
        "coordinate_system": "pixel_xyxy_top_left",
        "detector": {"backend": "fixture", "model": "fixture", "version": "1"},
        "tracker": {"backend": "fixture", "config": "fixture", "version": "1"},
        "fixed_camera_assumed": True,
        "camera_motion_state": "not_checked",
        "authorization_status": "authorized_camera_engineering_smoke",
        "deidentification_status": "anonymous_fixture",
    }
    (inputs / "media_sidecar.json").write_text(
        json.dumps(sidecar, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, *, second_start_frame: int = 4) -> tuple[Path, Path, Path]:
    _write_video_pair(tmp_path, "home_a", 4)
    _write_video_pair(tmp_path, "home_b", 3)
    xml_path = tmp_path / "annotations.xml"
    xml_path.write_text(
        _project_xml(second_start_frame=second_start_frame), encoding="utf-8"
    )
    return xml_path, tmp_path / "videos", tmp_path / "tracking"


def test_project_import_normalizes_cumulative_frames_and_separates_truth(
    tmp_path: Path,
) -> None:
    xml_path, video_root, tracking_root = _fixture(tmp_path)
    output = tmp_path / "imported"

    result = build_cvat_project_import_bundle(
        project_root=ROOT,
        cvat_xml_path=xml_path,
        video_root=video_root,
        tracking_output_root=tracking_root,
        output_dir=output,
        participant_id="P-HOME-01",
        session_id="S-HOME-01",
        clock_domain_id="CLOCK-HOME-01",
    )

    assert result.task_count == 2
    assert result.episode_count == 2
    assert result.aligned_episode_count == 2
    assert result.geometry_mismatch_episode_count == 1
    boundary_a = json.loads(
        (output / "tasks/home_a/episode_boundaries.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    boundary_b = json.loads(
        (output / "tasks/home_b/episode_boundaries.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert (boundary_a["start_sec"], boundary_a["end_sec_exclusive"]) == (
        0.0,
        2.0 / 15.0,
    )
    assert (boundary_b["start_sec"], boundary_b["end_sec_exclusive"]) == (
        0.0,
        2.0 / 15.0,
    )
    assert boundary_a["target_track_id"] == boundary_b["target_track_id"] == 1
    assert "observable_pattern" not in boundary_a
    truth_b = json.loads(
        (output / "tasks/home_b/episode_truth.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert truth_b["observable_pattern"] == "direct"
    summary = json.loads((output / "project_summary.json").read_text(encoding="utf-8"))
    assert summary["model_predictions_consumed"] is False
    assert summary["target_track_binding"]["cvat_box_geometry_consumed"] is False
    labeled_sidecar = json.loads(
        (output / "tasks/home_a/media_sidecar.labeled.json").read_text(
            encoding="utf-8"
        )
    )
    assert labeled_sidecar["authorization_status"] == "authorized_camera_labeled_evaluation"
    task_summary = json.loads(
        (output / "tasks/home_a/import_summary.json").read_text(encoding="utf-8")
    )
    assert task_summary["annotation_authorization_status"] == (
        "authorized_camera_labeled_evaluation"
    )
    assert (output / "tasks/home_a/proposals/summary.json").is_file()
    index_rows = [
        json.loads(line)
        for line in (output / "boundary_evaluation_batch_index.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(index_rows) == 2
    assert all(Path(row["human_boundary_jsonl"]).is_file() for row in index_rows)
    assert all(
        Path(row["proposal_bundle_dir"]).parent.parent == output / "tasks"
        for row in index_rows
    )


def test_project_import_rejects_task_frames_that_do_not_match_cumulative_offset(
    tmp_path: Path,
) -> None:
    xml_path, video_root, tracking_root = _fixture(
        tmp_path, second_start_frame=0
    )
    with pytest.raises(CameraCvatProjectError, match="task offset"):
        build_cvat_project_import_bundle(
            project_root=ROOT,
            cvat_xml_path=xml_path,
            video_root=video_root,
            tracking_output_root=tracking_root,
            output_dir=tmp_path / "rejected",
            participant_id="P-HOME-01",
            session_id="S-HOME-01",
            clock_domain_id="CLOCK-HOME-01",
        )


def test_project_import_cli_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/wandering/import_wandering_cvat_project.py"),
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--tracking-output-root" in completed.stdout
