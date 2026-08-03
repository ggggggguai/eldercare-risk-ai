from __future__ import annotations

import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts.prepare.prepare_fall_pose_cache import (
    build_pose_cache_jobs,
    ensure_cache_contract,
    main,
)


def make_manifest(
    video_id: str | None,
    *,
    dataset: str = "le2i_imvia",
    eligibility: bool = True,
    media_type: str | None = None,
    path: str | None = None,
) -> dict[str, object]:
    return {
        "asset_id": f"asset_{video_id}",
        "video_id": video_id,
        "dataset": dataset,
        "media_type": media_type or ("video" if video_id is not None else "tabular"),
        "eligibility": eligibility,
        "path": path or (f"unused/{video_id}.avi" if video_id else "unused/asset.csv"),
        "scene_region": "home",
        "source_group_id": "source_1",
        "fps": 25.0,
        "frame_count": 50,
        "duration_sec": 2.0,
        "width": 640,
        "height": 480,
        "sha256": f"sha256_{video_id}",
    }


class FallPoseCachePreparationTest(unittest.TestCase):
    def test_build_jobs_filters_dataset_and_ignores_non_video_assets(self) -> None:
        rows = [
            make_manifest(None),
            make_manifest("le2i_2"),
            make_manifest("le2i_1"),
            make_manifest("le2i_ineligible", eligibility=False),
            make_manifest("le2i_timeseries", media_type="timeseries"),
            make_manifest("ntu_1", dataset="ntu_rgbd"),
        ]

        jobs = build_pose_cache_jobs(rows, datasets=["le2i_imvia"])

        self.assertEqual([job["video_id"] for job in jobs], ["le2i_1", "le2i_2"])
        self.assertTrue(all(job["dataset"] == "le2i_imvia" for job in jobs))
        self.assertTrue(all("max_frames" not in job for job in jobs))

    def test_build_jobs_rejects_duplicate_non_empty_video_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate pose-cache manifest video_id: le2i_1"):
            build_pose_cache_jobs(
                [make_manifest("le2i_1"), make_manifest("le2i_1")],
                datasets=["le2i_imvia"],
            )

    def test_cache_contract_rejects_incompatible_parameters(self) -> None:
        contract = {
            "schema_version": "fall-pose-cache-v1",
            "manifest_sha256": "manifest-a",
            "model_sha256": "model-a",
            "confidence": 0.25,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            ensure_cache_contract(output_dir, contract)
            ensure_cache_contract(output_dir, dict(contract))

            with self.assertRaisesRegex(ValueError, "cache contract mismatch"):
                ensure_cache_contract(output_dir, {**contract, "confidence": 0.30})

    def test_cache_contract_accepts_new_manifest_revision(self) -> None:
        contract = {
            "schema_version": "fall-pose-cache-v1",
            "manifest_path": "data/manifests/fall_risk_video_manifest.jsonl",
            "manifest_sha256": "manifest-a",
            "model_sha256": "model-a",
            "confidence": 0.25,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            ensure_cache_contract(output_dir, contract)

            resolved = ensure_cache_contract(
                output_dir,
                {**contract, "manifest_sha256": "manifest-b"},
            )

            self.assertEqual(resolved, contract)

    def test_main_writes_full_video_cache_and_resumes_completed_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path = root / "pose.pt"
            model_path.write_bytes(b"model")
            video_path = root / "video.avi"
            video_path.write_bytes(b"video")
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(make_manifest("le2i_1", path=str(video_path))) + "\n",
                encoding="utf-8",
            )
            output_dir = root / "cache"
            calls: list[dict[str, object]] = []

            def fake_pose(**kwargs: object) -> int:
                calls.append(dict(kwargs))
                Path(kwargs["output_path"]).write_text(
                    json.dumps(
                        {
                            "frame_id": 0,
                            "timestamp_sec": 0.0,
                            "person_id": "le2i_1_001",
                            "track_id": 1,
                            "scene_region": "home",
                            "keypoints": [],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return 1

            fake_ultralytics = types.SimpleNamespace(YOLO=lambda _: object())
            argv = [
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(output_dir),
                "--model",
                str(model_path),
                "--dataset",
                "le2i_imvia",
                "--device",
                "mps",
                "--batch-id",
                "le2i_imvia",
            ]
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                with mock.patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                    with mock.patch(
                        "scripts.prepare.prepare_fall_pose_cache.run_yolov8_pose",
                        side_effect=fake_pose,
                    ):
                        with mock.patch(
                            "scripts.prepare.prepare_fall_pose_cache.process_pose_records",
                            side_effect=lambda records, **_: records,
                        ):
                            first_exit = main(argv)
                            second_exit = main(argv)

            self.assertEqual(first_exit, 0)
            self.assertEqual(second_exit, 0)
            self.assertEqual(len(calls), 1)
            self.assertIsNone(calls[0]["max_frames"])
            self.assertFalse(calls[0]["persist_tracker"])
            self.assertTrue((output_dir / "raw/le2i_1.jsonl").is_file())
            self.assertTrue((output_dir / "cleaned/le2i_1.jsonl").is_file())
            state = json.loads((output_dir / "state/le2i_1.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "completed")
            summary = json.loads(
                (output_dir / "batches/le2i_imvia/preparation_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["status_counts"], {"skipped_existing": 1})
            console_summary = json.loads(stdout.getvalue().splitlines()[-1])
            self.assertNotIn("jobs", console_summary)
            self.assertEqual(console_summary["processed_job_count"], 1)
            self.assertEqual(
                console_summary["summary_path"],
                (output_dir / "batches/le2i_imvia/preparation_summary.json").as_posix(),
            )


if __name__ == "__main__":
    unittest.main()
