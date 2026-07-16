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
    probe_video_metadata,
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
            },
        )
        self.assertEqual(len(rows), 20)
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
            self.assertIn("license_id", row)
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

    def test_missing_license_pre_vfallp_and_duplicate_content_are_ineligible(self) -> None:
        rows = build_fall_risk_manifest(
            self.repo, probe_video=self._probe_video
        ).rows

        for dataset in ("le2i_imvia", "toaga", "pre_vfallp"):
            affected = [row for row in rows if row["dataset"] == dataset]
            self.assertTrue(affected)
            self.assertTrue(all(row["license_id"] is None for row in affected))
            self.assertTrue(all(row["eligibility"] is False for row in affected))
            self.assertTrue(
                all("license_unknown" in row["exclusion_reasons"] for row in affected)
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
        self.assertTrue(all(row["license_id"] == "CC-BY-NC-SA-4.0" for row in ur_fall))
        self.assertTrue(all(row["eligibility"] is True for row in ur_fall))

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
