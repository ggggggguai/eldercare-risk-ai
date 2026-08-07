from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from elderly_monitoring.modules.fall_risk.data_manifest import (
    VIDEO_METADATA_FIELDS,
    VideoMetadata,
    build_fall_risk_manifest,
    build_ntu_rgbd_clip_manifest,
    probe_video_metadata,
    write_pre_vfallp_media_inventory,
    write_fall_risk_manifest,
)


class FallRiskDataManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.repo = Path(self._tempdir.name)
        self._build_fixture()

    def tearDown(self) -> None:
        self._tempdir.cleanup()

    def test_build_manifest_covers_all_adapters_and_required_metadata(self) -> None:
        result = build_fall_risk_manifest(self.repo, probe_video=self._probe_video)
        rows = result.rows

        self.assertEqual(
            {row["dataset"] for row in rows},
            {
                "le2i_imvia",
                "fall_detection_2017",
                "ur_fall",
                "toaga",
                "gstride",
                "ltmm",
                "pre_vfallp",
                "caucafall",
            },
        )
        self.assertEqual(len(rows), 22)
        self.assertEqual(
            [row["path"] for row in rows],
            sorted(row["path"] for row in rows),
        )
        self.assertEqual(len({row["asset_id"] for row in rows}), len(rows))
        self.assertFalse(
            any(Path(row["path"]).suffix.lower() == ".zip" for row in rows)
        )

        for row in rows:
            self.assertFalse(Path(row["path"]).is_absolute())
            self.assertTrue((self.repo / row["path"]).is_file())
            self.assertEqual(
                row["sha256"],
                hashlib.sha256((self.repo / row["path"]).read_bytes()).hexdigest(),
            )
            self.assertIn("consent_id", row)
            self.assertIn("eligibility", row)
            self.assertIsInstance(row["exclusion_reasons"], list)

        videos = [row for row in rows if row["media_type"] == "video"]
        self.assertTrue(videos)
        self.assertTrue(all(row["video_id"] for row in videos))
        self.assertEqual(len({row["video_id"] for row in videos}), len(videos))
        by_name = {Path(row["path"]).name: row for row in videos}
        self.assertEqual(by_name["video (1).avi"]["fps_num"], 24_000)
        self.assertEqual(by_name["video (1).avi"]["fps_den"], 1_001)
        self.assertAlmostEqual(by_name["video (1).avi"]["fps"], 24_000 / 1_001)
        self.assertEqual(by_name["video (1).avi"]["frame_count"], 240)

        nonvideos = [row for row in rows if row["media_type"] != "video"]
        self.assertTrue(nonvideos)
        for row in nonvideos:
            self.assertIsNone(row["video_id"])
            for field in VIDEO_METADATA_FIELDS:
                self.assertIsNone(row[field])

        le2i = next(row for row in rows if row["dataset"] == "le2i_imvia")
        self.assertEqual(le2i["video_id"], "le2i_home_01_video_1")
        self.assertEqual(
            le2i["annotation_path"],
            "data/external/le2i_imvia/raw/FallDataset/Home_01/Annotation_files/video (1).txt",
        )
        self.assertEqual(le2i["source_group_id"], "le2i_home_01_unknown_subject_pool")
        lecture = next(
            row
            for row in rows
            if row["video_id"] == "le2i_lecture_room_video_1"
        )
        self.assertEqual(lecture["label_source"], "unlabeled")
        self.assertIsNone(lecture["annotation_path"])

        caucafall = [row for row in rows if row["dataset"] == "caucafall"]
        self.assertEqual(len(caucafall), 2)
        self.assertEqual(
            {row["subject_id"] for row in caucafall}, {"caucafall_s01"}
        )
        self.assertEqual(
            {row["source_group_id"] for row in caucafall}, {"caucafall_s01"}
        )
        self.assertEqual(
            {row["source_action_code"] for row in caucafall},
            {"FallForward", "Walk"},
        )
        self.assertTrue(all(row["eligibility"] is True for row in caucafall))
        self.assertTrue(all(row["label_source"] == "unlabeled" for row in caucafall))
        self.assertTrue(all(row["annotation_path"] is None for row in caucafall))

    def test_structured_event_and_subject_grouping(self) -> None:
        rows = build_fall_risk_manifest(
            self.repo, probe_video=self._probe_video
        ).rows

        fall_2017 = [row for row in rows if row["dataset"] == "fall_detection_2017"]
        self.assertEqual(len(fall_2017), 2)
        self.assertEqual({row["subject_id"] for row in fall_2017}, {"fall_detection_2017_sbj_07"})
        self.assertEqual(len({row["original_event_id"] for row in fall_2017}), 1)
        self.assertEqual(len({row["source_group_id"] for row in fall_2017}), 1)
        self.assertEqual(
            {row["annotation_path"] for row in fall_2017},
            {
                "data/external/fall_detection_2017/raw/VideoDataset/Fall/"
                "SBJ_01_LOC1/ACT1_F_1/metadata.json"
            },
        )

        ur_fall = [row for row in rows if row["dataset"] == "ur_fall"]
        event_assets = [
            row
            for row in ur_fall
            if row["original_event_id"] == "ur_fall_fall_01"
        ]
        self.assertEqual(len(event_assets), 4)
        self.assertEqual(
            {row["modality"] for row in event_assets},
            {"rgb_video", "wearable_accelerometer", "event_sync_data"},
        )
        self.assertEqual(
            {row["view"] for row in event_assets if row["media_type"] == "video"},
            {"cam0", "cam1"},
        )
        self.assertEqual(len({row["source_group_id"] for row in event_assets}), 1)
        self.assertTrue(
            all(
                row["video_id"] is None
                for row in event_assets
                if row["media_type"] != "video"
            )
        )
        indexes = [
            row for row in ur_fall if row["modality"] == "camera_event_index"
        ]
        self.assertEqual(len(indexes), 2)
        self.assertTrue(all(row["video_id"] is None for row in indexes))
        ur_videos = [row for row in ur_fall if row["media_type"] == "video"]
        self.assertEqual(
            {row["annotation_path"] for row in ur_videos},
            {"data/external/ur_fall/raw/fall-01-data.csv"},
        )

        toaga = [row for row in rows if row["dataset"] == "toaga"]
        participant_assets = [
            row for row in toaga if row["subject_id"] == "toaga_oaw01"
        ]
        self.assertEqual(len(participant_assets), 4)
        self.assertEqual(
            {row["view"] for row in participant_assets if row["media_type"] == "video"},
            {"top", "bottom"},
        )
        self.assertEqual(
            {row["modality"] for row in participant_assets},
            {"rgb_video", "pose_keypoints"},
        )
        self.assertEqual(
            len({row["source_group_id"] for row in participant_assets}), 1
        )
        self.assertEqual(
            len({row["original_event_id"] for row in participant_assets}), 1
        )
        pose_assets = [
            row
            for row in participant_assets
            if row["modality"] == "pose_keypoints"
        ]
        self.assertTrue(all(row["video_id"] is None for row in pose_assets))
        self.assertEqual(
            {row["subset"] for row in pose_assets},
            {"pose_tracking_openpose", "pose_tracking_detectron"},
        )
        toaga_table = next(
            row for row in toaga if row["modality"] == "participant_metadata"
        )
        self.assertEqual(toaga_table["subject_id"], "unknown")
        self.assertIsNone(toaga_table["video_id"])
        self.assertIsNone(toaga_table["annotation_path"])

        gstride_table = next(
            row
            for row in rows
            if row["dataset"] == "gstride"
            and row["modality"] == "participant_metadata"
        )
        self.assertEqual(gstride_table["subject_id"], "unknown")
        self.assertEqual(gstride_table["source_group_id"], "gstride_participant_table")
        self.assertIsNone(gstride_table["video_id"])
        self.assertIsNone(gstride_table["annotation_path"])

    def test_fall_tiktok_manifest_keeps_annotated_clips_without_raw_sources(self) -> None:
        entries = [
            {"sequence": 1, "original_filename": "7.mp4", "filename": "1.mp4"},
            {"sequence": 2, "original_filename": "12.mp4", "filename": "2.mp4"},
        ]
        self._write(
            "configs/data/fall_tiktok_source_map_v1.json",
            json.dumps(
                {
                    "schema_version": "fall-tiktok-source-map-v1",
                    "dataset": "fall_tiktok",
                    "source_directory": "data/external/抖音b站跌倒视频整理",
                    "entries": entries,
                }
            ).encode(),
        )
        self._write(
            "configs/data/fall_tiktok_collection_decision_v1.json",
            json.dumps(
                {
                    "schema_version": "fall-tiktok-collection-decision-v1",
                    "decision_id": "fixture-training-authorization",
                    "decided_at": "2026-07-28",
                    "decided_by": "project_owner",
                    "dataset": "fall_tiktok",
                    "collection_status": "project_collected",
                    "training_use": "authorized",
                    "redistribution_use": "not_authorized_by_this_decision",
                    "consent_status": "not_recorded",
                    "subject_grouping_status": "unknown",
                    "source_group_id": "fall_tiktok_project_collection_pool",
                    "source_uri": "internal://collection/fall_tiktok_fixture",
                    "provenance_status": "project_collected_training_authorized",
                }
            ).encode(),
        )
        for filename in ("1.mp4", "2.mp4"):
            self._write(
                f"data/external/抖音b站跌倒视频整理/annotated_clips/{filename}",
                f"clip-{filename}".encode(),
            )

        rows = [
            row
            for row in build_fall_risk_manifest(
                self.repo, probe_video=self._probe_video
            ).rows
            if row["dataset"] == "fall_tiktok"
        ]

        self.assertEqual(len(rows), 2)
        by_video = {row["video_id"]: row for row in rows}
        clip = by_video["fall_tiktok_clip_001"]
        self.assertNotIn("fall_tiktok_raw_001", by_video)
        self.assertEqual(clip["original_event_id"], "fall_tiktok_sample_001")
        self.assertEqual(clip["source_group_id"], "fall_tiktok_project_collection_pool")
        self.assertEqual(clip["original_filename"], "7.mp4")
        self.assertNotIn("derived_from_video_id", clip)
        self.assertEqual(clip["subset"], "annotated_clips")
        self.assertEqual(clip["label_source"], "cvat_manual")
        self.assertTrue(all(row["eligibility"] is True for row in rows))
        self.assertTrue(all(row["exclusion_reasons"] == [] for row in rows))
        self.assertTrue(
            all(
                row["source_uri"]
                == "internal://collection/fall_tiktok_fixture"
                for row in rows
            )
        )
        self.assertTrue(
            all(
                row["provenance_status"]
                == "project_collected_training_authorized"
                for row in rows
            )
        )
        self.assertTrue(
            all(
                row["collection_decision_id"]
                == "fixture-training-authorization"
                for row in rows
            )
        )

    def test_ntu_rgbd_external_manifest_preserves_trial_and_subject_groups(self) -> None:
        source_root = self.repo / "external-ntu"
        first_view = source_root / "part_a" / "S006C001P016R001A042_rgb.avi"
        second_view = source_root / "part_b" / "S006C002P016R001A042_rgb.avi"
        squat = source_root / "part_c" / "S017C003P040R002A080_rgb.avi"
        for path, payload in (
            (first_view, b"first-view"),
            (second_view, b"second-view"),
            (squat, b"squat"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

        result = build_ntu_rgbd_clip_manifest(
            source_root, probe_video=self._probe_video
        )

        self.assertEqual(len(result.rows), 3)
        self.assertEqual(result.summary["version"], "ntu-rgbd-clip-manifest-v1")
        by_video = {row["video_id"]: row for row in result.rows}
        first = by_video["ntu_rgbd_s006_p016_r001_a042_c001"]
        second = by_video["ntu_rgbd_s006_p016_r001_a042_c002"]
        self.assertTrue(Path(first["path"]).is_absolute())
        self.assertEqual(Path(first["path"]), first_view.resolve())
        self.assertEqual(first["dataset"], "ntu_rgbd")
        self.assertEqual(first["source_action_code"], "A042")
        self.assertEqual(first["subject_id"], "ntu_rgbd_p016")
        self.assertEqual(first["source_group_id"], "ntu_rgbd_subject_p016")
        self.assertEqual(first["original_event_id"], "ntu_rgbd_s006_p016_r001_a042")
        self.assertEqual(first["view"], "c001")
        self.assertEqual(first["original_event_id"], second["original_event_id"])
        self.assertEqual(first["source_group_id"], second["source_group_id"])
        self.assertEqual(
            by_video["ntu_rgbd_s017_p040_r002_a080_c003"]["source_action_code"],
            "A080",
        )

    def test_main_manifest_includes_only_manually_reviewed_ntu_clips(self) -> None:
        ntu_video = self.repo / "external-ntu" / "S006C001P016R001A042_rgb.avi"
        accepted_a043 = self.repo / "external-ntu" / "S006C001P016R001A043_rgb.avi"
        unlabelled_a043 = self.repo / "external-ntu" / "S006C002P016R001A043_rgb.avi"
        ntu_video.parent.mkdir(parents=True)
        ntu_video.write_bytes(b"reviewed-ntu")
        accepted_a043.write_bytes(b"accepted-a043")
        unlabelled_a043.write_bytes(b"unlabelled-a043")
        external = build_ntu_rgbd_clip_manifest(
            ntu_video.parent, probe_video=self._probe_video
        )
        manifest_path = self.repo / "data/manifests/ntu_rgbd_clip_manifest.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(external.content)
        map_path = self.repo / "configs/data/ntu_rgbd_clip_label_map_v2.json"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_path.write_text(
            json.dumps(
                {
                    "schema_version": "ntu-rgbd-clip-label-map-v2",
                    "mapping_id": "ntu-rgbd-clip-label-map-v2",
                    "boundary_review": {
                        "decision_id": "ntu-rgbd-manual-boundary-review-20260725",
                        "reviewed_at": "2026-07-25",
                        "reviewer_id": "project_owner",
                        "boundary_precision": "exact",
                        "training_tier": "primary",
                    },
                    "mappings": [
                        {
                            "source_action_code": "A042",
                            "mode": "manual_exact",
                            "action_id": "C03",
                        },
                        {
                            "source_action_code": "A043",
                            "mode": "excluded",
                            "action_id": None,
                        },
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self._write(
            "configs/data/ntu_rgbd_a043_cvat_decision_v1.json",
            json.dumps(
                {
                    "schema_version": "ntu-rgbd-a043-cvat-decision-v1",
                    "decision_id": "fixture-a043-acceptance",
                    "reviewed_at": "2026-07-28",
                    "reviewer_id": "project_owner",
                    "source_action_code": "A043",
                    "decision": "accept_manual_cvat_labels",
                    "direct_filename_import": False,
                    "accepted_batch_id": "ntu_rgbd_a043_cvat_review",
                    "accepted_protocols": [
                        "segmented_nonfall_controlled_lie_down",
                        "segmented_normal_to_fall",
                        "segmented_normal_to_fall_to_sit_to_stand",
                        "segmented_normal_to_fall_with_trailing_outside",
                        "whole_clip_controlled_squat_hard_negative",
                        "whole_clip_fall_without_onset",
                        "whole_clip_uncertain",
                    ],
                    "v3_event_training_policy": "auxiliary_approximate",
                    "adjudications": [],
                }
            ).encode(),
        )
        accepted_video_id = "ntu_rgbd_s006_p016_r001_a043_c001"
        self._write(
            "data/annotations/fall_risk/generated/v2/"
            "ntu_rgbd_a043_cvat_review/action_labels.jsonl",
            (json.dumps({"video_id": accepted_video_id, "source": "cvat"}) + "\n").encode(),
        )
        self._write(
            "data/annotations/fall_risk/generated/v2/"
            "ntu_rgbd_a043_cvat_review/source_annotations.zip",
            b"fixture-redacted-cvat",
        )

        rows = build_fall_risk_manifest(
            self.repo, probe_video=self._probe_video
        ).rows
        ntu_rows = [row for row in rows if row["dataset"] == "ntu_rgbd"]

        self.assertEqual(len(ntu_rows), 2)
        by_video = {row["video_id"]: row for row in ntu_rows}
        self.assertEqual(
            by_video["ntu_rgbd_s006_p016_r001_a042_c001"]["label_source"],
            "manual_exact_clip_boundary",
        )
        accepted = by_video[accepted_video_id]
        self.assertEqual(accepted["label_source"], "cvat_manual")
        self.assertEqual(
            accepted["annotation_path"],
            "data/annotations/fall_risk/generated/v2/"
            "ntu_rgbd_a043_cvat_review/source_annotations.zip",
        )
        self.assertEqual(accepted["label_decision_id"], "fixture-a043-acceptance")
        self.assertNotIn("ntu_rgbd_s006_p016_r001_a043_c002", by_video)

    def test_missing_source_quarantine_and_duplicate_content_are_ineligible(self) -> None:
        rows = build_fall_risk_manifest(
            self.repo, probe_video=self._probe_video
        ).rows

        for dataset in ("le2i_imvia", "toaga"):
            affected = [row for row in rows if row["dataset"] == dataset]
            self.assertTrue(affected)
            self.assertTrue(all(row["eligibility"] is True for row in affected))

        pre_vfallp = [row for row in rows if row["dataset"] == "pre_vfallp"]
        self.assertTrue(pre_vfallp)
        self.assertTrue(all(row["eligibility"] is False for row in pre_vfallp))
        self.assertTrue(
            all("source_unknown" in row["exclusion_reasons"] for row in pre_vfallp)
        )

        duplicates = [
            row
            for row in rows
            if "duplicate_content" in row["exclusion_reasons"]
        ]
        self.assertEqual(len(duplicates), 2)
        self.assertEqual(len({row["sha256"] for row in duplicates}), 1)
        self.assertEqual(
            {row["source_group_id"] for row in duplicates},
            {"gstride_v001", "gstride_v002"},
        )
        self.assertEqual(len({row["duplicate_group_id"] for row in duplicates}), 1)
        self.assertTrue(all(row["eligibility"] is False for row in duplicates))

        ur_fall = [row for row in rows if row["dataset"] == "ur_fall"]
        self.assertTrue(ur_fall)
        self.assertTrue(all(row["eligibility"] is True for row in ur_fall))

    def test_internal_authorization_override_only_enables_listed_pre_vfallp_assets(self) -> None:
        self._write(
            "data/external/Pre_VFallp/unreviewed_subset/other.mp4",
            b"unreviewed-pre-vfallp-video",
        )
        initial_rows = build_fall_risk_manifest(
            self.repo, probe_video=self._probe_video
        ).rows
        authorized = next(
            row
            for row in initial_rows
            if row["dataset"] == "pre_vfallp"
            and row["subset"] == "dizziness_fall_forward"
        )
        config = self.repo / "configs/data/fall_risk_internal_authorizations.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            "\n".join(
                [
                    "schema_version: fall-risk-internal-authorizations-v2",
                    "authorizations:",
                    "  - authorization_id: internal_pre_vfallp_fixture",
                    "    approval_reference: test_internal_authorization",
                    "    approved_at: '2026-07-22'",
                    "    source_uri: internal://authorization/internal_pre_vfallp_fixture",
                    "    evidence:",
                    "      kind: cvat_export_archive",
                    "      name: fixture.zip",
                    f"      sha256: {'a' * 64}",
                    "    video_ids:",
                    f"      - {authorized['video_id']}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        rows = build_fall_risk_manifest(self.repo, probe_video=self._probe_video).rows
        by_video = {row["video_id"]: row for row in rows if row["video_id"]}
        enabled = by_video[authorized["video_id"]]
        remaining = next(
            row
            for row in rows
            if row["dataset"] == "pre_vfallp"
            and row["subset"] == "unreviewed_subset"
        )

        self.assertTrue(enabled["eligibility"])
        self.assertEqual(enabled["exclusion_reasons"], [])
        self.assertIsNone(enabled["consent_id"])
        self.assertEqual(
            enabled["source_uri"],
            "internal://authorization/internal_pre_vfallp_fixture",
        )
        self.assertEqual(
            enabled["internal_authorization"]["authorization_id"],
            "internal_pre_vfallp_fixture",
        )
        self.assertEqual(
            enabled["provenance_status"],
            "internal_authorized_source_unverified",
        )
        self.assertEqual(
            enabled["internal_authorization"]["evidence"]["kind"],
            "cvat_export_archive",
        )
        self.assertFalse(remaining["eligibility"])
        self.assertIn("source_unknown", remaining["exclusion_reasons"])

    def test_media_inventory_authorization_requires_unchanged_listed_media(self) -> None:
        authorized_path = self._write(
            "data/external/Pre_VFallp/authorized_subset/one.mp4",
            b"authorized-pre-vfallp-video",
        )
        inventory_path = Path(
            "data/manifests/pre_vfallp_authorized_fixture_inventory.jsonl"
        )
        inventory = write_pre_vfallp_media_inventory(
            self.repo,
            inventory_path,
            inventory_id="pre_vfallp_authorized_fixture_inventory",
            subsets=["authorized_subset"],
        )
        config = self.repo / "configs/data/fall_risk_internal_authorizations.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            "\n".join(
                [
                    "schema_version: fall-risk-internal-authorizations-v2",
                    "authorizations:",
                    "  - authorization_id: internal_pre_vfallp_media_fixture",
                    "    approval_reference: test_internal_authorization",
                    "    approved_at: '2026-07-22'",
                    "    source_uri: internal://authorization/internal_pre_vfallp_media_fixture",
                    "    evidence:",
                    "      kind: local_media_inventory",
                    f"      name: {inventory_path.name}",
                    f"      sha256: {inventory.sha256}",
                    f"      path: {inventory_path.as_posix()}",
                    "      inventory_id: pre_vfallp_authorized_fixture_inventory",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        rows = build_fall_risk_manifest(self.repo, probe_video=self._probe_video).rows
        enabled = next(row for row in rows if row["path"] == authorized_path.relative_to(self.repo).as_posix())
        untouched = next(
            row
            for row in rows
            if row["dataset"] == "pre_vfallp" and row["path"].endswith("sample.mp4")
        )
        self.assertTrue(enabled["eligibility"])
        self.assertEqual(
            enabled["internal_authorization"]["evidence"]["kind"],
            "local_media_inventory",
        )
        self.assertEqual(
            enabled["internal_authorization"]["evidence"]["path"],
            inventory_path.as_posix(),
        )
        self.assertFalse(untouched["eligibility"])

        authorized_path.write_bytes(b"mutated-pre-vfallp-video")
        with self.assertRaisesRegex(ValueError, "media checksum"):
            build_fall_risk_manifest(self.repo, probe_video=self._probe_video)

    def test_repeated_build_is_byte_for_byte_deterministic(self) -> None:
        first = build_fall_risk_manifest(self.repo, probe_video=self._probe_video)
        second = build_fall_risk_manifest(self.repo, probe_video=self._probe_video)

        self.assertEqual(first.content, second.content)
        self.assertEqual(first.rows, second.rows)
        self.assertEqual(first.manifest_sha256, second.manifest_sha256)
        self.assertEqual(
            first.manifest_sha256, hashlib.sha256(first.content).hexdigest()
        )
        self.assertEqual(first.summary["manifest_sha256"], first.manifest_sha256)
        self.assertEqual(first.summary["asset_count"], len(first.rows))

    def test_atomic_writer_refuses_overwrite_by_default(self) -> None:
        output = self.repo / "data/manifests/fall_risk_video_manifest.jsonl"
        first = write_fall_risk_manifest(
            self.repo,
            output,
            probe_video=self._probe_video,
        )
        original = output.read_bytes()

        with self.assertRaises(FileExistsError):
            write_fall_risk_manifest(
                self.repo,
                output,
                probe_video=self._probe_video,
            )

        self.assertEqual(output.read_bytes(), original)
        second = write_fall_risk_manifest(
            self.repo,
            output,
            overwrite=True,
            probe_video=self._probe_video,
        )
        self.assertEqual(output.read_bytes(), second.content)
        self.assertEqual(first.manifest_sha256, second.manifest_sha256)

    def test_ffprobe_json_parser_preserves_rational_fps_and_ignores_stderr(self) -> None:
        calls: list[list[str]] = []

        def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
            calls.append(command)
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "streams": [
                            {
                                "avg_frame_rate": "500000/20833",
                                "r_frame_rate": "24/1",
                                "nb_frames": "321",
                                "width": 320,
                                "height": 240,
                            }
                        ],
                        "format": {"duration": "13.374786"},
                    }
                ),
                stderr="decoder warning that must not corrupt JSON",
            )

        metadata = probe_video_metadata(
            Path("sample.avi"), runner=fake_run, ffprobe_bin="ffprobe-test"
        )

        self.assertEqual(metadata.fps_num, 500_000)
        self.assertEqual(metadata.fps_den, 20_833)
        self.assertAlmostEqual(metadata.fps, 500_000 / 20_833)
        self.assertEqual(metadata.frame_count, 321)
        self.assertEqual(metadata.duration_sec, 13.374786)
        self.assertEqual((metadata.width, metadata.height), (320, 240))
        self.assertEqual(calls[0][0], "ffprobe-test")
        self.assertIn("json", calls[0])

    def _build_fixture(self) -> None:
        self._write(
            "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi",
            b"le2i-video",
        )
        self._write(
            "data/external/le2i_imvia/raw/FallDataset/Home_01/Annotation_files/video (1).txt",
            b"1\n2\n",
        )
        self._write(
            "data/external/le2i_imvia/raw/FallDataset/Lecture room/video (1).avi",
            b"le2i-lecture-video",
        )

        fall_dir = (
            "data/external/fall_detection_2017/raw/VideoDataset/Fall/"
            "SBJ_01_LOC1/ACT1_F_1"
        )
        self._write(f"{fall_dir}/camera_a.mp4", b"fall-camera-a")
        self._write(f"{fall_dir}/camera_b.mp4", b"fall-camera-b")
        self._write(
            f"{fall_dir}/metadata.json",
            json.dumps(
                {
                    "subjectId": 7,
                    "locationId": 3,
                    "actionId": 11,
                    "side": "F",
                    "attempt": 2,
                }
            ).encode(),
        )

        self._write("data/external/ur_fall/raw/fall-01-cam0.mp4", b"ur-cam0")
        self._write("data/external/ur_fall/raw/fall-01-cam1.mp4", b"ur-cam1")
        self._write("data/external/ur_fall/raw/fall-01-data.csv", b"sync")
        self._write("data/external/ur_fall/raw/fall-01-acc.csv", b"accelerometer")
        self._write("data/external/ur_fall/raw/urfall-cam0-falls.csv", b"fall-index")
        self._write("data/external/ur_fall/raw/urfall-cam0-adls.csv", b"adl-index")
        self._write("data/external/ur_fall/raw/fall-01-cam0-rgb.zip", b"raw-zip")

        self._write("data/external/toaga/raw/Videos/OAW01-top.mp4", b"toaga-top")
        self._write("data/external/toaga/raw/Videos/OAW01-bottom.mp4", b"toaga-bottom")
        self._write("data/external/toaga/raw/Table_1.xlsx", b"participant-table")
        self._write(
            "data/external/toaga/raw/Pose Tracking/OpenPose/01/"
            "OAW01-OpenPose-top-front-1.csv",
            b"openpose-keypoints",
        )
        self._write(
            "data/external/toaga/raw/Pose Tracking/Detectron/01/"
            "OAW01-Detectron-bottom-back-2.csv",
            b"detectron-keypoints",
        )
        self._write("data/external/toaga/raw/PoseTracking.zip", b"raw-zip")

        duplicate = b"same-imu-content"
        self._write(
            "data/external/gstride/raw/GSTRIDE_database/_IMU/V001.csv", duplicate
        )
        self._write(
            "data/external/gstride/raw/GSTRIDE_database/_IMU/V002.csv", duplicate
        )
        self._write("data/external/gstride/raw/GSTRIDE_DDBB.xlsx", b"gstride-table")
        self._write("data/external/gstride/raw/GSTRIDE_database.zip", b"raw-zip")

        self._write(
            "data/external/ltmm/raw/ClinicalDemogData_COFL.xlsx", b"ltmm-table"
        )
        self._write(
            "data/external/Pre_VFallp/dizziness_fall_forward/sample.mp4",
            b"pre-vfallp-video",
        )
        self._write(
            "data/external/caucafall/raw/Subject.1/WalkS1.avi",
            b"caucafall-walk",
        )
        self._write(
            "data/external/caucafall/raw/Subject.1/FallForwardS1.avi",
            b"caucafall-forward-fall",
        )

    def _write(self, relative_path: str, content: bytes) -> Path:
        path = self.repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    @staticmethod
    def _probe_video(path: Path) -> VideoMetadata:
        if path.name == "video (1).avi":
            return VideoMetadata(
                fps_num=24_000,
                fps_den=1_001,
                fps=24_000 / 1_001,
                frame_count=240,
                duration_sec=10.01,
                width=320,
                height=240,
            )
        return VideoMetadata(
            fps_num=30,
            fps_den=1,
            fps=30.0,
            frame_count=300,
            duration_sec=10.0,
            width=640,
            height=480,
        )


if __name__ == "__main__":
    unittest.main()
