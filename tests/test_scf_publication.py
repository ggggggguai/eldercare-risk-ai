from __future__ import annotations

import json
from pathlib import Path

from elderly_monitoring.modules.fall_risk.scf_publication import (
    publish_scf_mvp_v1_training_batch,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_scf_publication_selects_only_auxiliary_train_subjects(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    decision = tmp_path / "decision.json"
    output = tmp_path / "published"
    _write_json(
        decision,
        {
            "batch_id": "SCF_MVP_V1",
            "decision_id": "scf-test-decision",
            "decided_at": "2026-08-17",
            "decided_by": "project_owner",
            "frozen_split_id": "scf-test-split",
            "development_split": {
                "auxiliary_train_subjects": ["P01"],
                "challenge_validation_subjects": ["P05"],
                "excluded_subjects": ["P03"],
            },
        },
    )
    _write_json(
        candidate / "import_report.json",
        {"batch_id": "SCF_MVP_V1", "g0": {"status": "ready"}},
    )
    common_manifest = {
        "asset_id": "scf_asset_aaaaaaaaaaaaaaaaaaaaaaaa",
        "content_sha256": "a" * 64,
        "duration_sec": 2.0,
        "fps_num": 30,
        "fps_den": 1,
        "frame_count": 60,
        "height": 1080,
        "path": "data/raw_videos/fall_risk/self_collected/SCF_MVP_V1/P01.mp4",
        "provisional_subject": "P01",
        "source_group_id": "scf_mvp_v1_p01_provisional",
        "subject_id": "scf_subject_p01",
        "video_id": "scf_mvp_v1_p01",
        "width": 1920,
    }
    _write_jsonl(candidate / "manifest.jsonl", [common_manifest])
    _write_jsonl(
        candidate / "action_labels.jsonl",
        [
            {
                "action_id": "A01",
                "asset_id": common_manifest["asset_id"],
                "bbox_end": [1, 2, 3, 4],
                "bbox_start": [1, 2, 3, 4],
                "cvat_task_id": "1",
                "cvat_track_id": 2,
                "end_frame_exclusive": 31,
                "end_time_exclusive": 31 / 30,
                "file_path": common_manifest["path"],
                "label_id": "scf_action_aaaaaaaaaaaaaaaaaaaaaaaa",
                "loss_eligible": True,
                "provisional_subject": "P01",
                "source_annotation_path": "data/annotations/P01.zip",
                "source_annotation_sha256": "b" * 64,
                "start_frame": 0,
                "start_time": 0.0,
                "subject_id": common_manifest["subject_id"],
                "training_tier": "auxiliary",
                "video_id": common_manifest["video_id"],
            },
            {
                "action_id": "A01",
                "asset_id": common_manifest["asset_id"],
                "bbox_end": [1, 2, 3, 4],
                "bbox_start": [1, 2, 3, 4],
                "cvat_task_id": "1",
                "cvat_track_id": 3,
                "end_frame_exclusive": 31,
                "end_time_exclusive": 31 / 30,
                "file_path": common_manifest["path"],
                "label_id": "scf_action_bbbbbbbbbbbbbbbbbbbbbbbb",
                "loss_eligible": False,
                "provisional_subject": "P05",
                "source_annotation_path": "data/annotations/P05.zip",
                "source_annotation_sha256": "c" * 64,
                "start_frame": 0,
                "start_time": 0.0,
                "subject_id": "scf_subject_p05",
                "training_tier": "challenge",
                "video_id": "scf_mvp_v1_p05",
            },
        ],
    )
    report = publish_scf_mvp_v1_training_batch(
        candidate_dir=candidate,
        decision_path=decision,
        output_dir=output,
    )
    assert report["selection"]["selected_action_labels"] == 1
    assert report["selection"]["selected_videos"] == 1
    assert len((output / "action_labels.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    [published_manifest] = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert published_manifest["split_partition"] == "train"
    assert published_manifest["training_tier_cap"] == "auxiliary"
    assert (
        published_manifest["gait_training_role"]
        == "action_pretraining_and_walking_gate_only"
    )
    assert (output / "event_labels.jsonl").read_text(encoding="utf-8") == ""
