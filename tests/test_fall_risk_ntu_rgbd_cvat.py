from __future__ import annotations

import json
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from elderly_monitoring.modules.fall_risk.ntu_rgbd_cvat import (
    NTU_RGBD_A043_BATCH_ID,
    import_ntu_rgbd_a043_cvat_labels,
)


def _box(
    track: ET.Element,
    frame: int,
    *,
    outside: bool = False,
    note: str = "",
) -> None:
    box = ET.SubElement(
        track,
        "box",
        {
            "frame": str(frame),
            "keyframe": "1",
            "outside": "1" if outside else "0",
            "occluded": "0",
            "xtl": "10",
            "ytl": "20",
            "xbr": "30",
            "ybr": "40",
            "z_order": "0",
        },
    )
    for name, value in (
        ("quality", "clear"),
        ("target_subject", "unknown"),
        ("note", note),
    ):
        attribute = ET.SubElement(box, "attribute", {"name": name})
        attribute.text = value


def _task(parent: ET.Element, task_id: str, source: str, size: int) -> ET.Element:
    task = ET.SubElement(parent, "task")
    for tag, value in (
        ("id", task_id),
        ("name", "{{nturgbd_rgb_fixture}}"),
        ("size", str(size)),
        ("mode", "interpolation"),
        ("start_frame", "0"),
        ("stop_frame", str(size - 1)),
    ):
        ET.SubElement(task, tag).text = value
    owner = ET.SubElement(task, "owner")
    ET.SubElement(owner, "username").text = "private"
    ET.SubElement(owner, "email").text = "private@example.invalid"
    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    ET.SubElement(segment, "url").text = "http://localhost:8080/private"
    size_element = ET.SubElement(task, "original_size")
    ET.SubElement(size_element, "width").text = "640"
    ET.SubElement(size_element, "height").text = "360"
    ET.SubElement(task, "source").text = source
    return task


def _write_zip(path: Path, root: ET.Element) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "annotations.xml",
            ET.tostring(root, encoding="utf-8", xml_declaration=True),
        )


def _write_project_export(path: Path) -> list[str]:
    sources = [
        "S001C001P001R001A043_rgb.mp4",
        "S001C002P001R001A043_rgb.mp4",
    ]
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    project = ET.SubElement(ET.SubElement(root, "meta"), "project")
    tasks = ET.SubElement(project, "tasks")
    _task(tasks, "1", sources[0], 3)
    _task(tasks, "2", sources[1], 3)
    first = ET.SubElement(
        root,
        "track",
        {"id": "0", "task_id": "1", "label": "D01_forward_fall"},
    )
    _box(first, 0)
    _box(first, 1)
    _box(first, 2, outside=True)
    second = ET.SubElement(
        root,
        "track",
        {"id": "1", "task_id": "2", "label": "D02_lateral_fall"},
    )
    _box(second, 3)
    _box(second, 4)
    _box(second, 5, outside=True)
    _write_zip(path, root)
    return sources


def _write_task_export(path: Path) -> str:
    source = "S002C001P002R001A043_rgb.mp4"
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    task = _task(ET.SubElement(root, "meta"), "1", source, 4)
    labels = ET.SubElement(task, "labels")
    ET.SubElement(ET.SubElement(labels, "label"), "name").text = "A01_normal_walk"
    normal = ET.SubElement(root, "track", {"id": "0", "label": "A01_normal_walk"})
    _box(normal, 0)
    _box(normal, 1, outside=True)
    fall = ET.SubElement(root, "track", {"id": "1", "label": "D01_forward_fall"})
    _box(fall, 1)
    _box(fall, 2)
    _box(fall, 3)
    _write_zip(path, root)
    return source


def _write_sequence_export(
    path: Path,
    source: str,
    size: int,
    intervals: list[tuple[str, int, int]],
    *,
    trailing_outside: bool = False,
    notes: dict[str, str] | None = None,
) -> None:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    _task(ET.SubElement(root, "meta"), "1", source, size)
    for track_id, (label, start, end) in enumerate(intervals):
        track = ET.SubElement(
            root,
            "track",
            {"id": str(track_id), "label": label},
        )
        for frame in range(start, end + 1):
            _box(track, frame, note=(notes or {}).get(label, ""))
        if trailing_outside and track_id == len(intervals) - 1:
            _box(track, size - 1, outside=True)
    _write_zip(path, root)


def _write_job_export(
    path: Path,
    size: int,
    label: str,
) -> None:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    job = ET.SubElement(meta, "job")
    for tag, value in (
        ("id", "329"),
        ("size", str(size)),
        ("mode", "interpolation"),
        ("start_frame", "0"),
        ("stop_frame", str(size - 1)),
    ):
        ET.SubElement(job, tag).text = value
    labels = ET.SubElement(job, "labels")
    ET.SubElement(ET.SubElement(labels, "label"), "name").text = label
    original_size = ET.SubElement(meta, "original_size")
    ET.SubElement(original_size, "width").text = "1920"
    ET.SubElement(original_size, "height").text = "1080"
    track = ET.SubElement(root, "track", {"id": "0", "label": label})
    for frame in range(size):
        _box(track, frame)
    _write_zip(path, root)


def _write_review_decision(
    path: Path,
    adjudications: list[dict[str, object]],
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "ntu-rgbd-a043-cvat-decision-v1",
                "decision_id": "ntu-rgbd-a043-test-review",
                "reviewed_at": "2026-07-30",
                "reviewer_id": "test-reviewer",
                "source_action_code": "A043",
                "decision": "accept_manual_cvat_labels",
                "direct_filename_import": False,
                "accepted_batch_id": NTU_RGBD_A043_BATCH_ID,
                "accepted_protocols": [
                    "segmented_normal_to_fall",
                    "segmented_normal_to_fall_with_trailing_outside",
                    "segmented_normal_to_fall_to_sit_to_stand",
                    "segmented_nonfall_controlled_lie_down",
                    "whole_clip_controlled_squat_hard_negative",
                    "whole_clip_fall_without_onset",
                    "whole_clip_uncertain",
                ],
                "v3_event_training_policy": "auxiliary_approximate",
                "adjudications": adjudications,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_empty_task_export(path: Path, source: str, size: int) -> None:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    _task(ET.SubElement(root, "meta"), "1", source, size)
    _write_zip(path, root)


def _manifest_row(source: str, frame_count: int) -> dict[str, object]:
    stem = Path(source).stem.removesuffix("_rgb")
    setup, camera, person, repetition = stem[1:4], stem[5:8], stem[9:12], stem[13:16]
    video_id = (
        f"ntu_rgbd_s{setup}_p{person}_r{repetition}_a043_c{camera}"
    )
    return {
        "asset_id": f"asset_{video_id}",
        "dataset": "ntu_rgbd",
        "video_id": video_id,
        "path": f"/unavailable/nturgb+d_rgb/{Path(source).with_suffix('.avi').name}",
        "fps_num": 30,
        "fps_den": 1,
        "fps": 30.0,
        "frame_count": frame_count,
        "duration_sec": frame_count / 30,
        "width": 1920,
        "height": 1080,
        "subject_id": f"ntu_rgbd_p{person}",
        "scene_region": "unknown",
        "view": f"c{camera}",
        "subset": f"setup_s{setup}",
        "source_group_id": f"ntu_rgbd_subject_p{person}",
        "source_action_code": "A043",
        "eligibility": True,
    }


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_import_normalizes_project_frames_scales_boxes_and_redacts() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        project_zip = root / "s001.zip"
        task_zip = root / "s002.zip"
        project_sources = _write_project_export(project_zip)
        task_source = _write_task_export(task_zip)
        manifest = root / "manifest.jsonl"
        _write_manifest(
            manifest,
            [
                _manifest_row(project_sources[0], 3),
                _manifest_row(project_sources[1], 3),
                _manifest_row(task_source, 4),
            ],
        )
        output_dir = root / NTU_RGBD_A043_BATCH_ID

        report = import_ntu_rgbd_a043_cvat_labels(
            [project_zip, task_zip],
            manifest_path=manifest,
            output_dir=output_dir,
        )

        actions = [
            json.loads(line)
            for line in (output_dir / "action_labels.jsonl").read_text().splitlines()
        ]
        events = [
            json.loads(line)
            for line in (output_dir / "event_labels.jsonl").read_text().splitlines()
        ]
        assert len(actions) == 4
        assert len(events) == 4
        assert report["source_video_count"] == 3
        assert report["protocol_counts"] == {
            "segmented_normal_to_fall": 1,
            "whole_clip_fall_without_onset": 2,
        }
        assert report["publication_status"] == "accepted_for_v2_publication"
        assert report["manual_acceptance_decision"]["decision"] == (
            "accept_manual_cvat_labels"
        )
        assert all(row["subject_id"].startswith("ntu_rgbd_p") for row in actions)
        assert all(row["source"] == "cvat" for row in actions)
        assert all("isolated" not in row["note"] for row in actions)
        assert all(
            row["source_annotation_path"].endswith("source_annotations.zip")
            for row in actions
        )
        assert actions[0]["bbox_start"] == [30.0, 60.0, 90.0, 120.0]
        second_view = next(row for row in actions if row["view"] == "c002")
        assert (second_view["start_frame"], second_view["end_frame"]) == (0, 1)
        with zipfile.ZipFile(output_dir / "source_annotations.zip") as archive:
            payload = archive.read("annotations.xml").decode("utf-8")
        assert "private@example.invalid" not in payload
        assert "localhost" not in payload
        assert "target_subject\">ntu_rgbd_p001" in payload
        assert "S001C001P001R001A043_rgb.avi" in payload

        replay_dir = root / "replayed"
        replay_report = import_ntu_rgbd_a043_cvat_labels(
            [output_dir / "source_annotations.zip"],
            manifest_path=manifest,
            output_dir=replay_dir,
        )
        assert replay_report["output_counts"] == report["output_counts"]
        assert replay_report["redacted_source_sha256"] == report["redacted_source_sha256"]


def test_import_rejects_missing_manifest_sources_unless_explicitly_allowed() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        task_zip = root / "s002.zip"
        source = _write_task_export(task_zip)
        missing_source = "S002C002P002R001A043_rgb.mp4"
        manifest = root / "manifest.jsonl"
        _write_manifest(
            manifest,
            [_manifest_row(source, 4), _manifest_row(missing_source, 4)],
        )

        with pytest.raises(ValueError, match="missing 1 expected A043 source"):
            import_ntu_rgbd_a043_cvat_labels(
                [task_zip],
                manifest_path=manifest,
                output_dir=root / "strict",
            )

        report = import_ntu_rgbd_a043_cvat_labels(
            [task_zip],
            manifest_path=manifest,
            output_dir=root / "partial",
            allow_incomplete=True,
        )
        assert report["complete_against_manifest"] is False
        assert report["missing_source_names"] == [missing_source]


def test_import_accepts_s006_s010_action_protocols_and_avi_sources() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        fixtures = [
            (
                "S010C001P013R002A043_rgb.avi",
                5,
                [("D01_forward_fall", 0, 4)],
                False,
            ),
            (
                "S010C001P015R001A043_rgb.avi",
                7,
                [
                    ("A01_normal_walk", 0, 1),
                    ("D02_lateral_fall", 2, 4),
                    ("A04_normal_sit_to_stand", 5, 6),
                ],
                False,
            ),
            (
                "S010C002P023R002A043_rgb.avi",
                6,
                [
                    ("A01_normal_walk", 0, 1),
                    ("A07_controlled_lie_down", 2, 5),
                ],
                False,
            ),
            (
                "S010C003P007R001A043_rgb.avi",
                6,
                [
                    ("A01_normal_walk", 0, 1),
                    ("D03_backward_fall", 2, 4),
                ],
                True,
            ),
        ]
        archives: list[Path] = []
        manifest_rows: list[dict[str, object]] = []
        for index, (source, size, intervals, trailing_outside) in enumerate(fixtures):
            archive = root / f"task-{index}.zip"
            _write_sequence_export(
                archive,
                source,
                size,
                intervals,
                trailing_outside=trailing_outside,
            )
            archives.append(archive)
            manifest_rows.append(_manifest_row(source, size))
        manifest = root / "manifest.jsonl"
        _write_manifest(manifest, manifest_rows)

        report = import_ntu_rgbd_a043_cvat_labels(
            archives,
            manifest_path=manifest,
            output_dir=root / "output",
        )

        assert report["protocol_counts"] == {
            "segmented_nonfall_controlled_lie_down": 1,
            "segmented_normal_to_fall_to_sit_to_stand": 1,
            "segmented_normal_to_fall_with_trailing_outside": 1,
            "whole_clip_fall_without_onset": 1,
        }
        assert report["raw_label_counts"] == {
            "A01_normal_walk": 3,
            "A04_normal_sit_to_stand": 1,
            "A07_controlled_lie_down": 1,
            "D01_forward_fall": 1,
            "D02_lateral_fall": 1,
            "D03_backward_fall": 1,
        }
        assert report["output_counts"] == {
            "action_labels": 8,
            "event_labels": 8,
        }


def test_revision_replaces_matching_base_task_including_empty_task() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = "S008C001P015R002A043_rgb.avi"
        base = root / "base.zip"
        revision = root / "revision.zip"
        _write_empty_task_export(base, source, 5)
        _write_sequence_export(
            revision,
            source,
            5,
            [
                ("A01_normal_walk", 0, 1),
                ("D03_backward_fall", 2, 4),
            ],
        )
        manifest = root / "manifest.jsonl"
        _write_manifest(manifest, [_manifest_row(source, 5)])

        report = import_ntu_rgbd_a043_cvat_labels(
            [base],
            revision_paths=[revision],
            manifest_path=manifest,
            output_dir=root / "output",
        )

        actions = [
            json.loads(line)
            for line in (root / "output" / "action_labels.jsonl").read_text().splitlines()
        ]
        assert [row["action_id"] for row in actions] == ["A01", "D03"]
        assert report["revision_overlay"] == {
            "replaced_source_count": 1,
            "replaced_source_names": ["S008C001P015R002A043_rgb.mp4"],
        }


def test_revision_must_match_a_base_task() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        base_source = "S008C001P015R001A043_rgb.avi"
        revision_source = "S008C001P015R002A043_rgb.avi"
        base = root / "base.zip"
        revision = root / "revision.zip"
        _write_sequence_export(
            base,
            base_source,
            4,
            [("A01_normal_walk", 0, 1), ("D01_forward_fall", 2, 3)],
        )
        _write_sequence_export(
            revision,
            revision_source,
            4,
            [("A01_normal_walk", 0, 1), ("D01_forward_fall", 2, 3)],
        )
        manifest = root / "manifest.jsonl"
        _write_manifest(
            manifest,
            [_manifest_row(base_source, 4), _manifest_row(revision_source, 4)],
        )

        with pytest.raises(ValueError, match="revision sources do not exist in base"):
            import_ntu_rgbd_a043_cvat_labels(
                [base],
                revision_paths=[revision],
                manifest_path=manifest,
                output_dir=root / "output",
                allow_incomplete=True,
            )


def test_job_export_revision_uses_archive_name_and_replaces_base() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = "S016C003P008R001A043_rgb.avi"
        base = root / "s016.zip"
        revision = root / "S016C003P008R001A043.zip"
        _write_sequence_export(
            base,
            source,
            5,
            [("D03_backward_fall", 0, 4)],
        )
        _write_job_export(revision, 5, "D02_lateral_fall")
        manifest = root / "manifest.jsonl"
        _write_manifest(manifest, [_manifest_row(source, 5)])

        report = import_ntu_rgbd_a043_cvat_labels(
            [base],
            revision_paths=[revision],
            manifest_path=manifest,
            output_dir=root / "output",
        )

        actions = [
            json.loads(line)
            for line in (root / "output" / "action_labels.jsonl").read_text().splitlines()
        ]
        assert [row["action_id"] for row in actions] == ["D02"]
        assert report["revision_overlay"] == {
            "replaced_source_count": 1,
            "replaced_source_names": ["S016C003P008R001A043_rgb.mp4"],
        }
        revision_archive = next(
            row for row in report["source_archives"] if row["role"] == "revision"
        )
        assert revision_archive["export_format"] == "job"
        assert revision_archive["source_names"] == [
            "S016C003P008R001A043_rgb.mp4"
        ]
        with zipfile.ZipFile(root / "output" / "source_annotations.zip") as archive:
            normalized = ET.fromstring(archive.read("annotations.xml"))
        task_sources = [
            task.findtext("source")
            for task in normalized.findall("./meta/project/tasks/task")
        ]
        assert task_sources == [source]
        assert normalized.find("./meta/project") is not None
        assert [track.get("label") for track in normalized.findall("./track")] == [
            "D02_lateral_fall"
        ]


def test_reviewed_whole_clip_labels_are_applied_and_preserved() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        override_source = "S013C001P018R001A043_rgb.avi"
        uncertain_source = "S011C001P015R001A043_rgb.avi"
        hard_negative_source = "S017C001P020R001A043_rgb.avi"
        override = root / "override.zip"
        uncertain = root / "uncertain.zip"
        hard_negative = root / "hard-negative.zip"
        _write_sequence_export(
            override,
            override_source,
            4,
            [("U01_unable_to_judge", 0, 3)],
            notes={"U01_unable_to_judge": "单视角方向不清，三视角复核为前向跌倒。"},
        )
        _write_sequence_export(
            uncertain,
            uncertain_source,
            4,
            [("U01_unable_to_judge", 0, 3)],
            notes={"U01_unable_to_judge": "三视角均无法区分跌倒和受控下蹲。"},
        )
        _write_sequence_export(
            hard_negative,
            hard_negative_source,
            4,
            [("A05_controlled_squat", 0, 3)],
        )
        decision = root / "decision.json"
        _write_review_decision(
            decision,
            [
                {
                    "type": "label_override",
                    "source_name": override_source,
                    "expected_label": "U01_unable_to_judge",
                    "accepted_label": "D01_forward_fall",
                    "evidence": "three_view_visual_review",
                    "reason": "Forward trunk descent and hand contact agree across views.",
                },
                {
                    "type": "accepted_hard_negative",
                    "source_names": [hard_negative_source],
                    "accepted_label": "A05_controlled_squat",
                    "training_use": "fall_hard_negative",
                    "evidence": "three_view_visual_review",
                    "reason": "Controlled symmetric lowering without loss of support.",
                },
            ],
        )
        manifest = root / "manifest.jsonl"
        _write_manifest(
            manifest,
            [
                _manifest_row(override_source, 4),
                _manifest_row(uncertain_source, 4),
                _manifest_row(hard_negative_source, 4),
            ],
        )

        report = import_ntu_rgbd_a043_cvat_labels(
            [override, uncertain, hard_negative],
            manifest_path=manifest,
            output_dir=root / "output",
            decision_config_path=decision,
        )

        actions = [
            json.loads(line)
            for line in (root / "output" / "action_labels.jsonl").read_text().splitlines()
        ]
        assert sorted(row["action_id"] for row in actions) == ["A05", "D01", "U01"]
        uncertain_action = next(row for row in actions if row["action_id"] == "U01")
        assert uncertain_action["note"] == "三视角均无法区分跌倒和受控下蹲。"
        assert report["protocol_counts"] == {
            "whole_clip_controlled_squat_hard_negative": 1,
            "whole_clip_fall_without_onset": 1,
            "whole_clip_uncertain": 1,
        }
        assert report["adjudication"]["label_overrides_applied"] == [
            "S013C001P018R001A043_rgb.mp4"
        ]
        assert report["adjudication"]["accepted_hard_negative_sources"] == [
            "S017C001P020R001A043_rgb.mp4"
        ]
