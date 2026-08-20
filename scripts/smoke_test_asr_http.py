"""Exercise startup prewarm, authentication, and real ASR through FastAPI."""

from __future__ import annotations

import argparse
import base64
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from elderly_monitoring.modules.asr.api import create_asr_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real standalone ASR HTTP smoke test")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.requests < 1 or args.concurrency < 1:
        raise ValueError("requests and concurrency must be positive")
    audio_path = args.audio.expanduser().resolve()
    audio_format = audio_path.suffix.lower().lstrip(".")
    if audio_format not in {"wav", "mp3", "m4a"}:
        raise ValueError("smoke test audio must be WAV, MP3, or M4A")
    payload = {
        "schema_version": "asr_request_v1",
        "request_id": "asr-http-smoke-001",
        "audio_input": {
            "source_type": "base64",
            "source": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
            "format": audio_format,
            "sample_rate": None,
            "channels": None,
        },
    }
    app = create_asr_app(api_token="smoke-token", prewarm=True)
    started = time.perf_counter()
    with TestClient(app) as client:
        startup_seconds = time.perf_counter() - started
        ready = client.get("/health/ready")
        unauthorized = client.post("/v1/asr/transcribe", json=payload)
        def submit(index: int):
            request_payload = dict(payload)
            request_payload["request_id"] = f"asr-http-smoke-{index + 1:03d}"
            started_at = time.perf_counter()
            response = client.post(
                "/v1/asr/transcribe",
                json=request_payload,
                headers={"Authorization": "Bearer smoke-token"},
            )
            return response, time.perf_counter() - started_at

        inference_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(args.concurrency, args.requests)) as executor:
            results = list(executor.map(submit, range(args.requests)))
        inference_seconds = time.perf_counter() - inference_started
    responses = [response for response, _ in results]
    bodies = [response.json() for response in responses]
    request_seconds = [elapsed for _, elapsed in results]
    print(f"startup_status={ready.status_code}")
    print(f"unauthorized_status={unauthorized.status_code}")
    print(f"response_statuses={[response.status_code for response in responses]}")
    print(f"asr_statuses={[body.get('status') for body in bodies]}")
    print(f"startup_seconds={startup_seconds:.3f}")
    print(f"wall_inference_seconds={inference_seconds:.3f}")
    print(f"request_seconds={[round(value, 3) for value in request_seconds]}")
    print(f"segments={[len(body.get('segments', [])) for body in bodies]}")
    successful = all(
        response.status_code == 200
        and body.get("status") in {"completed", "completed_empty_speech"}
        for response, body in zip(responses, bodies, strict=True)
    )
    return 0 if ready.status_code == 200 and successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
