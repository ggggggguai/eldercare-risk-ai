from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from elderly_monitoring.modules.mental_health.wandering.camera_episode_import import (
    CAMERA_EPISODE_BOUNDARY_SCHEMA_VERSION,
    CAMERA_EPISODE_TRUTH_SCHEMA_VERSION,
    CameraEpisodeImportError,
    import_cvat_episode_xml,
    load_episode_boundaries,
)


ROOT = Path(__file__).resolve().parents[1]


def _media() -> dict[str, object]:
    return {
        "source_video_id": "s00-01",
        "nominal_fps": 2.0,
        "duration_sec": 15.0,
    }


def _attributes(
    *,
    pattern: str,
    role: str,
    note: str = "",
) -> str:
    purpose = "purposeful" if pattern == "direct" else "unknown"
    evidence = "participant_report" if pattern == "direct" else "unknown"
    return f"""
      <attribute name="observable_pattern">{pattern}</attribute>
      <attribute name="purpose_context">{purpose}</attribute>
      <attribute name="purpose_evidence">{evidence}</attribute>
      <attribute name="evaluation_role">{role}</attribute>
      <attribute name="script_type">tracking_smoke</attribute>
      <attribute name="visibility_quality">good</attribute>
      <attribute name="tracking_issue">not_checked</attribute>
      <attribute name="note">{note}</attribute>
    """


def _box(frame: int, *, outside: int, attributes: str = "") -> str:
    return (
        f'<box frame="{frame}" outside="{outside}" occluded="0" keyframe="1" '
        f'xtl="1" ytl="1" xbr="2" ybr="2">{attributes}</box>'
    )


def _write_xml(path: Path) -> None:
    direct = _attributes(pattern="direct", role="ordinary_negative")
    uncertain = _attributes(
        pattern="unknown",
        role="uncertain",
        note="shape boundary requires review",
    )
    excluded = _attributes(
        pattern="unknown",
        role="excluded",
        note="target out of frame",
    )
    payload = f"""<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><job><size>30</size><start_frame>0</start_frame><stop_frame>29</stop_frame></job></meta>
  <track id="cvat-0" label="wandering_episode" source="manual">
    {_box(0, outside=0, attributes=direct)}
    {_box(9, outside=0, attributes=direct)}
    {_box(10, outside=1)}
  </track>
  <track id="cvat-1" label="wandering_episode" source="manual">
    {_box(10, outside=0, attributes=uncertain)}
    {_box(19, outside=0, attributes=uncertain)}
    {_box(20, outside=1)}
  </track>
  <track id="cvat-2" label="wandering_episode" source="manual">
    {_box(20, outside=0, attributes=excluded)}
    {_box(29, outside=0, attributes=excluded)}
  </track>
</annotations>
"""
    path.write_text(payload, encoding="utf-8")


def test_cvat_outside_end_of_video_and_track_identity_are_imported_separately(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "annotations.xml"
    _write_xml(xml_path)

    result = import_cvat_episode_xml(xml_path, _media(), target_track_id=7)

    assert CAMERA_EPISODE_BOUNDARY_SCHEMA_VERSION == "wandering-camera-episode-boundary-v1"
    assert CAMERA_EPISODE_TRUTH_SCHEMA_VERSION == "wandering-camera-episode-truth-v1"
    assert [(row["start_sec"], row["end_sec_exclusive"]) for row in result.boundaries] == [
        (0.0, 5.0),
        (5.0, 10.0),
        (10.0, 15.0),
    ]
    assert [row["cvat_track_id"] for row in result.boundaries] == [
        "cvat-0",
        "cvat-1",
        "cvat-2",
    ]
    assert {row["target_track_id"] for row in result.boundaries} == {7}
    assert all(row["boundary_source"] == "cvat_xml" for row in result.boundaries)
    assert [row["annotation_status"] for row in result.truth_records] == [
        "accepted",
        "uncertain",
        "excluded",
    ]
    assert result.truth_records[0]["observable_pattern"] == "direct"
    assert result.truth_records[1]["evaluation_role"] == "uncertain"
    assert result.truth_records[2]["evaluation_role"] == "excluded"

    truth_fields = {
        "observable_pattern",
        "purpose_context",
        "purpose_evidence",
        "evaluation_role",
        "annotation_status",
    }
    assert all(not truth_fields.intersection(row) for row in result.boundaries)


def test_simplified_boundary_loader_is_exact_truth_free_and_fail_closed(
    tmp_path: Path,
) -> None:
    valid = {
        "schema_version": CAMERA_EPISODE_BOUNDARY_SCHEMA_VERSION,
        "episode_id": "episode-001",
        "source_video_id": "s00-01",
        "target_track_id": 1,
        "start_sec": 0.0,
        "end_sec_exclusive": 5.0,
        "boundary_source": "simplified_jsonl",
        "boundary_status": "ready",
        "boundary_reason_codes": [],
        "cvat_track_id": None,
    }
    path = tmp_path / "boundaries.jsonl"
    path.write_text(json.dumps(valid) + "\n", encoding="utf-8")
    assert load_episode_boundaries(path, _media()) == [valid]

    contaminated = {**valid, "observable_pattern": "direct"}
    path.write_text(json.dumps(contaminated) + "\n", encoding="utf-8")
    with pytest.raises(CameraEpisodeImportError, match="boundary fields"):
        load_episode_boundaries(path, _media())

    missing_track = dict(valid)
    missing_track.pop("target_track_id")
    path.write_text(json.dumps(missing_track) + "\n", encoding="utf-8")
    with pytest.raises(CameraEpisodeImportError, match="boundary fields"):
        load_episode_boundaries(path, _media())


def test_cvat_rejects_inconsistent_attributes_and_visible_frames_after_outside(
    tmp_path: Path,
) -> None:
    direct = _attributes(pattern="direct", role="ordinary_negative")
    pacing = _attributes(pattern="pacing", role="uncertain", note="changed")
    xml = f"""<annotations><version>1.1</version><meta><job><size>12</size></job></meta>
    <track id="0" label="wandering_episode">
      {_box(0, outside=0, attributes=direct)}
      {_box(5, outside=1)}
      {_box(6, outside=0, attributes=pacing)}
    </track></annotations>"""
    path = tmp_path / "bad.xml"
    path.write_text(xml, encoding="utf-8")
    with pytest.raises(CameraEpisodeImportError, match="outside"):
        import_cvat_episode_xml(path, _media(), target_track_id=1)


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/wandering/import_wandering_cvat_episodes.py",
        "scripts/wandering/run_camera_episode_inference.py",
    ],
)
def test_episode_cli_help(relative: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / relative), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--help" in completed.stdout
