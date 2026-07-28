from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
from fastapi.testclient import TestClient

from elderly_monitoring.service.app import create_app
from elderly_monitoring.service.session import SessionManager
from elderly_monitoring.service.settings import ServiceSettings


class _CallbackHandler(BaseHTTPRequestHandler):
    count = 0

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.__class__.count += 1
        self.send_response(204)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the HTTP fall-risk service against a finite local video.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.input.exists():
        raise SystemExit(f"input video does not exist: {args.input}")

    readers = []
    _CallbackHandler.count = 0

    class LocalVideoReader:
        def __init__(self, _: str, **kwargs: object) -> None:
            self.capture = None
            self.count = 0
            self.source_pts_sec = None
            readers.append(self)

        def open(self) -> None:
            self.capture = cv2.VideoCapture(str(args.input))
            if not self.capture.isOpened():
                raise RuntimeError("local smoke video could not be opened")

        def read(self):
            if self.count >= args.max_frames:
                return None
            time.sleep(1.0 / 8.0)
            ok, frame = self.capture.read()
            if not ok:
                return None
            self.count += 1
            self.source_pts_sec = max(
                0.0, float(self.capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            )
            return frame

        def release(self) -> None:
            if self.capture is not None:
                self.capture.release()
                self.capture = None

    callback_server = ThreadingHTTPServer(("127.0.0.1", 0), _CallbackHandler)
    threading.Thread(target=callback_server.serve_forever, daemon=True).start()
    manager = SessionManager(
        reader_factory=LocalVideoReader,
        model_path=args.model,
        reconnect_attempts=0,
        reconnect_delay_sec=0.0,
        callback_token="smoke-callback-token",
        scene_risk_scores={"home": 0.1},
    )
    settings = ServiceSettings(model_path=args.model, api_token="smoke-api-token", callback_token="smoke-callback-token")
    client = TestClient(create_app(settings=settings, session_manager=manager))
    headers = {"Authorization": "Bearer smoke-api-token"}
    response = client.post(
        "/v1/monitoring/sessions",
        headers=headers,
        json={
            "request_id": "local-service-smoke",
            "stream_url": "https://local-smoke.invalid/live",
            "device_id": "smoke-camera",
            "person_id": "smoke-person",
            "scene_region": "home",
            "callback_url": f"http://127.0.0.1:{callback_server.server_port}/events",
        },
    )
    response.raise_for_status()
    session_id = response.json()["session_id"]
    deadline = time.monotonic() + 120.0
    saw_running = False
    processed = 0
    primary_poses = 0
    analyses = 0
    pre_stop_status = "unknown"
    pre_stop_error = None
    while time.monotonic() < deadline:
        session = manager.get(session_id)
        if session and session.status.value == "running":
            saw_running = True
        if session and session.engine is not None:
            processed = int(getattr(session.engine, "frame_id", 0))
            primary_poses = int(getattr(session.engine, "primary_pose_count", 0))
            analyses = int(getattr(getattr(session.engine, "assembler", None), "analysis_count", 0))
        if readers and readers[0].count >= args.max_frames:
            break
        if session and session.status.value == "failed":
            pre_stop_status = session.status.value
            pre_stop_error = session.last_error
            break
        time.sleep(0.05)
    stop_started = time.monotonic()
    stop_response = client.post(f"/v1/monitoring/sessions/{session_id}/stop", headers=headers)
    stop_duration_sec = time.monotonic() - stop_started
    stop_response.raise_for_status()
    callback_server.shutdown()
    session = manager.get(session_id)
    if not saw_running or processed < 1 or primary_poses < 1 or analyses < 1 or session is None or session.status.value != "stopped":
        raise SystemExit(
            f"service smoke failed: running={saw_running}, processed={processed}, "
            f"primary_poses={primary_poses}, analyses={analyses}, pre_stop_status={pre_stop_status}, "
            f"error={pre_stop_error!r}, status={getattr(session, 'status', None)}"
        )
    processed = int(session.runtime_diagnostics.get("processed_frames", processed))
    primary_poses = int(
        session.runtime_diagnostics.get("primary_pose_count", primary_poses)
    )
    analyses = int(session.runtime_diagnostics.get("analysis_count", analyses))
    sampling = session.runtime_diagnostics.get("sampling", {})
    last_frame = session.runtime_diagnostics.get("last_frame", {})
    window = last_frame.get("window", {}) if isinstance(last_frame, dict) else {}
    branches = window.get("branches", {}) if isinstance(window, dict) else {}
    target = last_frame.get("target", {}) if isinstance(last_frame, dict) else {}
    invalid_branch_states = {
        str(name): diagnostic.get("status")
        for name, diagnostic in branches.items()
        if not isinstance(diagnostic, dict)
        or diagnostic.get("status") not in {"valid", "unavailable", "inference_error"}
    }
    if not branches or invalid_branch_states:
        raise SystemExit(
            "service smoke failed: missing or invalid stage-2 branch diagnostics: "
            f"{invalid_branch_states!r}"
        )
    if target.get("state") not in {"bound", "unbound", "ambiguous", "lost"}:
        raise SystemExit(
            f"service smoke failed: invalid target state: {target.get('state')!r}"
        )
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "engineering_regression_only",
        "input": {
            "path": str(args.input),
            "sha256": _sha256_file(args.input),
            "max_frames": args.max_frames,
        },
        "result": {
            "status": "passed",
            "saw_running": saw_running,
            "processed_frames": processed,
            "primary_pose_count": primary_poses,
            "analysis_count": analyses,
            "callback_count": _CallbackHandler.count,
            "stream_epoch": session.stream_epoch,
            "frame_queue": session.frame_diagnostics,
            "epoch_history": session.epoch_history,
            "sampling": sampling,
            "stage2": {
                "target": target,
                "window": window,
                "stage_timings_ms": last_frame.get("stage_timings_ms", {}),
                "epoch_resets": session.runtime_diagnostics.get("epoch_resets", []),
                "close": session.runtime_diagnostics.get("close"),
            },
            "stop_duration_sec": round(stop_duration_sec, 6),
        },
        "claims": {
            "algorithm_accuracy": "not_evaluated",
            "clinical_validity": "not_evaluated",
            "ezviz_live_link": "not_evaluated",
        },
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        f"service smoke passed: frames={processed}, primary_poses={primary_poses}, "
        f"analyses={analyses}, callbacks={_CallbackHandler.count}, "
        f"epoch={session.stream_epoch}, stop_sec={stop_duration_sec:.3f}, status=stopped"
    )
    return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
