from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi.testclient import TestClient

from elderly_monitoring.service.app import create_app
from elderly_monitoring.service.settings import ServiceSettings


class _CallbackRecorder:
    def __init__(self, token: str) -> None:
        self.token = token
        self.payloads: list[dict[str, Any]] = []
        self.invalid_requests = 0
        self._lock = threading.Lock()

    def handler_type(self) -> type[BaseHTTPRequestHandler]:
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.headers.get("Authorization") != f"Bearer {recorder.token}":
                    with recorder._lock:
                        recorder.invalid_requests += 1
                    self.send_response(401)
                    self.end_headers()
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(payload, dict):
                        raise ValueError("callback payload must be an object")
                except (ValueError, json.JSONDecodeError):
                    with recorder._lock:
                        recorder.invalid_requests += 1
                    self.send_response(400)
                    self.end_headers()
                    return
                with recorder._lock:
                    recorder.payloads.append(payload)
                self.send_response(204)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                return

        return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fall-risk HTTP service against a real live stream without "
            "a business backend. The stream URL is read from an environment variable."
        )
    )
    parser.add_argument("--stream-url-env", default="EZVIZ_STREAM_URL")
    parser.add_argument("--model", type=Path, default=Path("yolov8n-pose.pt"))
    parser.add_argument("--duration-sec", type=float, default=120.0)
    parser.add_argument("--poll-interval-sec", type=float, default=0.5)
    parser.add_argument("--device-id", default="ezviz-smoke-camera")
    parser.add_argument("--person-id", default="ezviz-smoke-person")
    parser.add_argument("--scene-region", default="home")
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.duration_sec <= 0:
        raise SystemExit("--duration-sec must be positive")
    if args.poll_interval_sec <= 0:
        raise SystemExit("--poll-interval-sec must be positive")
    stream_url = os.environ.get(args.stream_url_env, "").strip()
    if not stream_url:
        raise SystemExit(f"stream URL environment variable is empty: {args.stream_url_env}")
    if not args.model.exists():
        raise SystemExit(f"pose model does not exist: {args.model}")

    report = run_live_smoke(
        stream_url=stream_url,
        model_path=args.model,
        requested_duration_sec=args.duration_sec,
        poll_interval_sec=args.poll_interval_sec,
        device_id=args.device_id,
        person_id=args.person_id,
        scene_region=args.scene_region,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    result = report["result"]
    transport = result["transport"]
    pipeline = result["algorithm_pipeline"]
    print(
        "live smoke "
        f"{result['overall_status']}: transport={transport['status']}, "
        f"pipeline={pipeline['status']}, frames={pipeline['max_processed_frames']}, "
        f"primary_poses={pipeline['max_primary_pose_count']}, "
        f"analyses={pipeline['max_analysis_count']}, "
        f"callbacks={report['callbacks']['count']}"
    )
    if transport["reasons"]:
        print(f"transport_reasons={','.join(transport['reasons'])}")
    if pipeline["reasons"]:
        print(f"pipeline_reasons={','.join(pipeline['reasons'])}")
    return 0 if result["overall_status"] == "passed" else 2


def run_live_smoke(
    *,
    stream_url: str,
    model_path: Path,
    requested_duration_sec: float,
    poll_interval_sec: float,
    device_id: str,
    person_id: str,
    scene_region: str,
) -> dict[str, Any]:
    api_token = "live-smoke-api-token"
    callback_token = "live-smoke-callback-token"
    base_settings = ServiceSettings.load()
    settings = replace(
        base_settings,
        model_path=model_path,
        api_token=api_token,
        callback_token=callback_token,
    )
    recorder = _CallbackRecorder(callback_token)
    callback_server = ThreadingHTTPServer(
        ("127.0.0.1", 0), recorder.handler_type()
    )
    callback_thread = threading.Thread(
        target=callback_server.serve_forever,
        daemon=True,
        name="fall-live-smoke-callback",
    )
    callback_thread.start()

    client = TestClient(create_app(settings=settings))
    headers = {"Authorization": f"Bearer {api_token}"}
    samples: list[dict[str, Any]] = []
    session_id: str | None = None
    stop_status: str | None = None
    completed_full_duration = False
    started = time.monotonic()
    try:
        response = client.post(
            "/v1/monitoring/sessions",
            headers=headers,
            json={
                "request_id": f"ezviz-live-smoke-{time.time_ns()}",
                "stream_url": stream_url,
                "device_id": device_id,
                "person_id": person_id,
                "scene_region": scene_region,
                "callback_url": (
                    f"http://127.0.0.1:{callback_server.server_port}/events"
                ),
            },
        )
        response.raise_for_status()
        session_id = str(response.json()["session_id"])
        deadline = started + requested_duration_sec
        while True:
            now = time.monotonic()
            if now >= deadline:
                completed_full_duration = True
                break
            status_response = client.get(
                f"/v1/monitoring/sessions/{session_id}", headers=headers
            )
            status_response.raise_for_status()
            sample = status_response.json()
            samples.append(sample)
            if sample.get("status") in {"failed", "stopped"}:
                break
            time.sleep(min(poll_interval_sec, max(0.0, deadline - now)))
    finally:
        actual_duration_sec = time.monotonic() - started
        if session_id is not None:
            stop_response = client.post(
                f"/v1/monitoring/sessions/{session_id}/stop", headers=headers
            )
            if stop_response.status_code == 202:
                stop_status = str(stop_response.json().get("status"))
            final_response = client.get(
                f"/v1/monitoring/sessions/{session_id}", headers=headers
            )
            if final_response.status_code == 200:
                samples.append(final_response.json())
        client.close()
        callback_server.shutdown()
        callback_server.server_close()
        callback_thread.join(timeout=1.0)

    return _build_report(
        stream_url=stream_url,
        requested_duration_sec=requested_duration_sec,
        actual_duration_sec=actual_duration_sec,
        completed_full_duration=completed_full_duration,
        samples=samples,
        callback_payloads=recorder.payloads,
        stop_status=stop_status,
        invalid_callback_requests=recorder.invalid_requests,
    )


def _assess_samples(
    samples: list[dict[str, Any]], *, completed_full_duration: bool
) -> dict[str, Any]:
    statuses = [str(sample.get("status", "unknown")) for sample in samples]
    status_history = [
        status for index, status in enumerate(statuses)
        if index == 0 or status != statuses[index - 1]
    ]
    frame_times = {
        str(sample["last_frame_at"])
        for sample in samples
        if sample.get("last_frame_at") is not None
    }
    max_epoch = max((int(sample.get("stream_epoch") or 0) for sample in samples), default=0)
    runtime_samples = [
        sample.get("runtime_diagnostics", {})
        for sample in samples
        if isinstance(sample.get("runtime_diagnostics"), dict)
    ]
    max_processed = max(
        (int(runtime.get("processed_frames") or 0) for runtime in runtime_samples),
        default=0,
    )
    max_primary = max(
        (int(runtime.get("primary_pose_count") or 0) for runtime in runtime_samples),
        default=0,
    )
    max_analyses = max(
        (int(runtime.get("analysis_count") or 0) for runtime in runtime_samples),
        default=0,
    )
    branch_statuses: dict[str, str] = {}
    for runtime in runtime_samples:
        last_frame = runtime.get("last_frame", {})
        window = last_frame.get("window", {}) if isinstance(last_frame, dict) else {}
        branches = window.get("branches", {}) if isinstance(window, dict) else {}
        if isinstance(branches, dict) and branches:
            branch_statuses = {
                str(name): str(diagnostic.get("status", "unknown"))
                for name, diagnostic in branches.items()
                if isinstance(diagnostic, dict)
            }

    transport_reasons: list[str] = []
    if not completed_full_duration:
        transport_reasons.append("monitoring_ended_before_requested_duration")
    if "running" not in statuses:
        transport_reasons.append("session_never_reached_running")
    if len(frame_times) < 2:
        transport_reasons.append("insufficient_distinct_frame_updates")
    if "failed" in statuses:
        transport_reasons.append("session_failed")
    if max_epoch > 1:
        transport_reasons.append("stream_reconnected")

    pipeline_reasons: list[str] = []
    if max_processed < 1:
        pipeline_reasons.append("no_processed_frames")
    if max_primary < 1:
        pipeline_reasons.append("no_primary_pose")
    if max_analyses < 1:
        pipeline_reasons.append("no_feature_analysis")
    if not branch_statuses:
        pipeline_reasons.append("missing_branch_diagnostics")
    if any(status == "inference_error" for status in branch_statuses.values()):
        pipeline_reasons.append("branch_inference_error")

    transport_status = "passed" if not transport_reasons else "failed"
    pipeline_status = "passed" if not pipeline_reasons else "failed"
    return {
        "overall_status": (
            "passed"
            if transport_status == "passed" and pipeline_status == "passed"
            else "failed"
        ),
        "transport": {
            "status": transport_status,
            "reasons": transport_reasons,
            "status_history": status_history,
            "distinct_frame_updates": len(frame_times),
            "max_stream_epoch": max_epoch,
        },
        "algorithm_pipeline": {
            "status": pipeline_status,
            "reasons": pipeline_reasons,
            "max_processed_frames": max_processed,
            "max_primary_pose_count": max_primary,
            "max_analysis_count": max_analyses,
            "branch_statuses": branch_statuses,
        },
    }


def _build_report(
    *,
    stream_url: str,
    requested_duration_sec: float,
    actual_duration_sec: float,
    completed_full_duration: bool,
    samples: list[dict[str, Any]],
    callback_payloads: list[dict[str, Any]],
    stop_status: str | None,
    invalid_callback_requests: int = 0,
) -> dict[str, Any]:
    assessment = _assess_samples(
        samples, completed_full_duration=completed_full_duration
    )
    last_sample = samples[-1] if samples else {}
    last_error = next(
        (
            str(sample["last_error"])
            for sample in reversed(samples)
            if sample.get("last_error")
        ),
        None,
    )
    callback_summaries = [
        {
            key: payload.get(key)
            for key in (
                "event_id",
                "episode_id",
                "risk_level",
                "trigger_event",
                "lifecycle_state",
                "version_kind",
            )
            if key in payload
        }
        for payload in callback_payloads
    ]
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "real_live_stream_engineering_smoke_only",
        "input": {
            "source": _redacted_source(stream_url),
            "requested_duration_sec": requested_duration_sec,
        },
        "result": assessment,
        "session": {
            "actual_duration_sec": round(actual_duration_sec, 3),
            "completed_full_duration": completed_full_duration,
            "stop_status": stop_status,
            "last_error": last_error,
            "final_frame_diagnostics": last_sample.get("frame_diagnostics", {}),
            "final_runtime_diagnostics": last_sample.get(
                "runtime_diagnostics", {}
            ),
        },
        "callbacks": {
            "count": len(callback_payloads),
            "invalid_request_count": invalid_callback_requests,
            "events": callback_summaries,
        },
        "claims": {
            "algorithm_accuracy": "not_evaluated",
            "clinical_validity": "not_evaluated",
            "backend_integration": "not_evaluated",
        },
    }


def _redacted_source(stream_url: str) -> dict[str, Any]:
    parsed = urlsplit(stream_url)
    return {
        "scheme": parsed.scheme.lower(),
        "host": parsed.hostname,
        "port": parsed.port,
    }


if __name__ == "__main__":
    raise SystemExit(main())
