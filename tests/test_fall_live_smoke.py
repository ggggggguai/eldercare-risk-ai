from __future__ import annotations

import json

from scripts.collect.run_fall_live_smoke import (
    _assess_samples,
    _build_report,
    _redacted_source,
)


def _sample(
    *,
    status: str = "running",
    last_frame_at: str | None = "2026-07-30T01:00:00+00:00",
    stream_epoch: int = 1,
    processed_frames: int = 10,
    primary_pose_count: int = 8,
    analysis_count: int = 4,
    branch_status: str = "valid",
) -> dict:
    return {
        "status": status,
        "last_frame_at": last_frame_at,
        "stream_epoch": stream_epoch,
        "last_error": None,
        "frame_diagnostics": {},
        "runtime_diagnostics": {
            "processed_frames": processed_frames,
            "primary_pose_count": primary_pose_count,
            "analysis_count": analysis_count,
            "last_frame": {
                "window": {
                    "branches": {
                        "gait": {"status": branch_status},
                        "sit_stand": {"status": "unavailable"},
                        "near_fall": {"status": "unavailable"},
                        "fall_state": {"status": "valid"},
                    }
                }
            },
        },
    }


def test_redacted_source_drops_path_query_and_credentials() -> None:
    source = _redacted_source(
        "rtmp://user:secret@rtmp.example:1935/v3/openlive/device"
        "?expire=123&t=signed-token"
    )

    assert source == {
        "scheme": "rtmp",
        "host": "rtmp.example",
        "port": 1935,
    }


def test_assessment_passes_stable_transport_and_executed_branches() -> None:
    samples = [
        _sample(last_frame_at="2026-07-30T01:00:00+00:00", processed_frames=2),
        _sample(last_frame_at="2026-07-30T01:00:01+00:00", processed_frames=10),
    ]

    assessment = _assess_samples(samples, completed_full_duration=True)

    assert assessment["overall_status"] == "passed"
    assert assessment["transport"]["status"] == "passed"
    assert assessment["algorithm_pipeline"]["status"] == "passed"
    assert assessment["algorithm_pipeline"]["branch_statuses"]["sit_stand"] == "unavailable"


def test_assessment_fails_early_disconnect_and_missing_pose() -> None:
    samples = [
        _sample(
            status="running",
            processed_frames=1,
            primary_pose_count=0,
            analysis_count=0,
        ),
        _sample(
            status="failed",
            last_frame_at="2026-07-30T01:00:00+00:00",
            stream_epoch=4,
            processed_frames=0,
            primary_pose_count=0,
            analysis_count=0,
        ),
    ]
    samples[0]["runtime_diagnostics"]["last_frame"].pop("window")
    samples[1]["runtime_diagnostics"]["last_frame"].pop("window")
    samples[1]["last_error"] = "stream reconnect attempts exhausted"

    assessment = _assess_samples(samples, completed_full_duration=False)

    assert assessment["overall_status"] == "failed"
    assert "monitoring_ended_before_requested_duration" in assessment["transport"]["reasons"]
    assert "session_failed" in assessment["transport"]["reasons"]
    assert "no_primary_pose" in assessment["algorithm_pipeline"]["reasons"]
    assert "missing_branch_diagnostics" in assessment["algorithm_pipeline"]["reasons"]


def test_report_never_contains_stream_path_or_query() -> None:
    stream_url = (
        "rtmp://rtmp.example:1935/v3/openlive/private-device"
        "?expire=123&t=signed-token"
    )
    samples = [
        _sample(last_frame_at="2026-07-30T01:00:00+00:00", processed_frames=2),
        _sample(last_frame_at="2026-07-30T01:00:01+00:00", processed_frames=10),
    ]

    report = _build_report(
        stream_url=stream_url,
        requested_duration_sec=2.0,
        actual_duration_sec=2.1,
        completed_full_duration=True,
        samples=samples,
        callback_payloads=[],
        stop_status="stopped",
    )
    encoded = json.dumps(report)

    assert report["input"]["source"] == {
        "scheme": "rtmp",
        "host": "rtmp.example",
        "port": 1935,
    }
    assert "private-device" not in encoded
    assert "signed-token" not in encoded
