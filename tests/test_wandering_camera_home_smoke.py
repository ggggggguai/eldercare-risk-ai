from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from elderly_monitoring.modules.mental_health.wandering.camera_home_smoke import (
    build_camera_home_smoke_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


def test_home_smoke_refuses_existing_output_before_expensive_video_work(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        build_camera_home_smoke_bundle(
            project_root=ROOT,
            input_video_path=tmp_path / "missing.mp4",
            output_dir=output,
            run_id="home-smoke-fixture",
            source_video_id="home-fixture",
            source_group_id="home-fixture-group",
            device_id="home-fixture-device",
            setup_id="home-fixture-setup",
            stream_epoch="full-recording",
            media_ref="external/home-fixture.mp4",
            participant_id="P-HOME-01",
            session_id="S-HOME-01",
            clock_domain_id="CLOCK-HOME-01",
            timezone_name=None,
        )


def test_home_smoke_cli_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/wandering/run_camera_home_smoke.py"),
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--input-video" in completed.stdout
    assert "--home-annotation-status" in completed.stdout
    assert "--provider-mode" in completed.stdout


@pytest.mark.parametrize("media_ref", ["/absolute/home.mp4", "../escape.mp4"])
def test_home_smoke_rejects_unsafe_media_ref_before_video_decode(
    tmp_path: Path, media_ref: str
) -> None:
    video = tmp_path / "fixture.mp4"
    video.write_bytes(b"not-decoded")

    with pytest.raises(ValueError, match="safe relative POSIX"):
        build_camera_home_smoke_bundle(
            project_root=ROOT,
            input_video_path=video,
            output_dir=tmp_path / "output",
            run_id="home-smoke-fixture",
            source_video_id="home-fixture",
            source_group_id="home-fixture-group",
            device_id="home-fixture-device",
            setup_id="home-fixture-setup",
            stream_epoch="full-recording",
            media_ref=media_ref,
            participant_id="P-HOME-01",
            session_id="S-HOME-01",
            clock_domain_id="CLOCK-HOME-01",
            timezone_name=None,
        )
