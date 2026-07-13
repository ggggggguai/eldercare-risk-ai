from __future__ import annotations

import argparse
import threading
import time
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.input.exists():
        raise SystemExit(f"input video does not exist: {args.input}")

    readers = []

    class LocalVideoReader:
        def __init__(self, _: str, **kwargs: object) -> None:
            self.capture = None
            self.count = 0
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
    stop_response = client.post(f"/v1/monitoring/sessions/{session_id}/stop", headers=headers)
    stop_response.raise_for_status()
    callback_server.shutdown()
    session = manager.get(session_id)
    if not saw_running or processed < 1 or primary_poses < 1 or analyses < 1 or session is None or session.status.value != "stopped":
        raise SystemExit(
            f"service smoke failed: running={saw_running}, processed={processed}, "
            f"primary_poses={primary_poses}, analyses={analyses}, pre_stop_status={pre_stop_status}, "
            f"error={pre_stop_error!r}, status={getattr(session, 'status', None)}"
        )
    print(f"service smoke passed: frames={processed}, primary_poses={primary_poses}, analyses={analyses}, callbacks={_CallbackHandler.count}, status=stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
