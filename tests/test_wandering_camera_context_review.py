from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import cv2
import httpx
import numpy as np
import pytest

from elderly_monitoring.modules.mental_health.wandering.camera_context_review import (
    CONTEXT_LABELS,
    CAMERA_CONTEXT_REVIEW_SCHEMA_VERSION,
    ContextFrame,
    ContextReviewRequest,
    DeterministicFakeContextProvider,
    DisabledContextProvider,
    OpenAICompatibleContextProvider,
    _build_context_review_row,
    _validate_context_review,
    extract_episode_frames,
    load_camera_context_review_config,
    select_eligible_episode_results,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/modules/wandering_camera_context_review_v1.yaml"
CLI = ROOT / "scripts/wandering/run_camera_context_review.py"


def _episode(
    *,
    episode_id: str = "episode-001",
    status: str = "ready",
    binary_label: str = "direct_or_non_wandering",
) -> dict[str, object]:
    return {
        "schema_version": "wandering-handoff-episode-result-v1",
        "module": "mental_health",
        "record_id": f"record-{episode_id}",
        "episode_id": episode_id,
        "person_id": "person-001",
        "session_id": "session-001",
        "source_video_id": "video-001",
        "start_time": None,
        "end_time": None,
        "start_sec": 1.0,
        "end_sec_exclusive": 4.0,
        "status": status,
        "binary": {
            "predicted_label": binary_label,
            "class_order": ["direct_or_non_wandering", "wandering_like"],
            "probabilities": [0.2, 0.8],
            "decision_threshold": 0.5,
        },
        "identity": {
            "model_id": "shape-model",
            "model_sha256": "1" * 64,
            "config_id": "shape-config",
            "config_sha256": "2" * 64,
            "policy_id": "shape-policy",
            "policy_sha256": "3" * 64,
        },
        "source_refs": [
            {
                "ref_type": "tracking",
                "ref_id": "video-001",
                "artifact_path": "/inputs/tracking.jsonl",
                "sha256": "4" * 64,
            }
        ],
    }


def _ready_frame(position: str, index: int) -> ContextFrame:
    image = np.full((8, 12, 3), 40 + index, dtype=np.uint8)
    ok, payload = cv2.imencode(".jpg", image)
    assert ok
    data = payload.tobytes()
    return ContextFrame(
        position=position,
        requested_timestamp_sec=float(index),
        selected_timestamp_sec=float(index),
        frame_index=index,
        track_id=1,
        bbox_xyxy=(1.0, 1.0, 7.0, 7.0),
        status="ready",
        error_code=None,
        scene_ref={
            "ref_type": "context_scene_frame",
            "ref_id": f"episode-001:{position}:scene",
            "artifact_path": f"/frames/{position}-scene.jpg",
            "sha256": "5" * 64,
        },
        person_crop_ref={
            "ref_type": "context_person_crop",
            "ref_id": f"episode-001:{position}:crop",
            "artifact_path": f"/frames/{position}-crop.jpg",
            "sha256": "6" * 64,
        },
        quality_flags=(),
        scene_bytes=data,
        person_crop_bytes=data,
    )


def test_trigger_selects_wandering_or_uncertain_without_mutating_input() -> None:
    rows = [
        _episode(episode_id="direct"),
        _episode(episode_id="wandering", binary_label="wandering_like"),
        _episode(episode_id="uncertain", status="uncertain"),
    ]
    before = copy.deepcopy(rows)

    selected, skipped = select_eligible_episode_results(rows)

    assert [row["episode_id"] for row in selected] == ["uncertain", "wandering"]
    assert skipped == {"not_context_eligible": 1}
    assert rows == before


def test_disabled_and_fake_providers_have_explicit_contracts() -> None:
    request = ContextReviewRequest(
        episode_id="episode-001",
        source_video_id="video-001",
        frames=tuple(
            _ready_frame(position, index)
            for index, position in enumerate(("start", "middle", "end"), start=1)
        ),
        tracking_qc_summary={"status": "ready"},
    )

    disabled = DisabledContextProvider().review(request)
    assert disabled.status == "unavailable"
    assert disabled.context_label == "unknown"
    assert disabled.confidence is None
    assert disabled.rationale is None
    assert disabled.error_code == "provider_disabled"

    first = DeterministicFakeContextProvider().review(request)
    second = DeterministicFakeContextProvider().review(request)
    assert first == second
    assert first.status == "ready"
    assert first.context_label in CONTEXT_LABELS - {"unknown"}
    assert first.error_code is None


def test_openai_compatible_adapter_serializes_images_and_parses_strict_json() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "context_label": "searching",
                                    "confidence": 0.8,
                                    "rationale": "Repeated visual search around the room.",
                                }
                            )
                        }
                    }
                ]
            },
        )

    provider = OpenAICompatibleContextProvider(
        endpoint="https://provider.invalid/v1/chat/completions",
        api_key="secret",
        model="vision-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    request = ContextReviewRequest(
        episode_id="episode-001",
        source_video_id="video-001",
        frames=tuple(
            _ready_frame(position, index)
            for index, position in enumerate(("start", "middle", "end"), start=1)
        ),
        tracking_qc_summary={"status": "ready"},
    )

    result = provider.review(request)

    assert result.status == "ready"
    assert result.context_label == "searching"
    assert result.confidence == 0.8
    assert result.error_code is None
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    image_parts = [
        part
        for part in payload["messages"][1]["content"]
        if part["type"] == "image_url"
    ]
    assert len(image_parts) == 6
    assert all(part["image_url"]["url"].startswith("data:image/jpeg;base64,") for part in image_parts)
    assert requests[0].headers["Authorization"] == "Bearer secret"


@pytest.mark.parametrize(
    ("response", "expected_code"),
    [
        (httpx.Response(503, text="unavailable"), "provider_http_error"),
        (
            httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"context_label":"not-allowed"}'}}
                    ]
                },
            ),
            "provider_invalid_response",
        ),
    ],
)
def test_openai_compatible_adapter_degrades_http_and_parse_failures(
    response: httpx.Response, expected_code: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        response.request = request
        return response

    provider = OpenAICompatibleContextProvider(
        endpoint="https://provider.invalid/v1/chat/completions",
        api_key="secret",
        model="vision-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    request = ContextReviewRequest(
        episode_id="episode-001",
        source_video_id="video-001",
        frames=tuple(
            _ready_frame(position, index)
            for index, position in enumerate(("start", "middle", "end"), start=1)
        ),
        tracking_qc_summary={},
    )

    result = provider.review(request)

    assert result.status == "unavailable"
    assert result.context_label == "unknown"
    assert result.error_code == expected_code


def test_openai_compatible_adapter_degrades_timeout_connection_and_missing_key() -> None:
    def timeout(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    def connection_failed(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed")

    request = ContextReviewRequest(
        episode_id="episode-001",
        source_video_id="video-001",
        frames=tuple(
            _ready_frame(position, index)
            for index, position in enumerate(("start", "middle", "end"), start=1)
        ),
        tracking_qc_summary={},
    )
    timeout_provider = OpenAICompatibleContextProvider(
        endpoint="https://provider.invalid/v1/chat/completions",
        api_key="secret",
        model="vision-model",
        client=httpx.Client(transport=httpx.MockTransport(timeout)),
    )
    connection_provider = OpenAICompatibleContextProvider(
        endpoint="https://provider.invalid/v1/chat/completions",
        api_key="secret",
        model="vision-model",
        client=httpx.Client(transport=httpx.MockTransport(connection_failed)),
    )
    missing_key_provider = OpenAICompatibleContextProvider(
        endpoint="https://provider.invalid/v1/chat/completions",
        api_key=None,
        model="vision-model",
    )

    assert timeout_provider.review(request).error_code == "provider_timeout"
    assert connection_provider.review(request).error_code == "provider_request_error"
    assert missing_key_provider.review(request).error_code == "provider_api_key_missing"


class _FakeCapture:
    def __init__(self, _path: str) -> None:
        self.fps = 2.0
        self.frame_count = 10
        self.timestamp = 0.0
        self.index = 0

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_POS_MSEC:
            self.timestamp = value / 1000.0
            self.index = min(self.frame_count - 1, max(0, round(self.timestamp * self.fps)))
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        image = np.full((20, 30, 3), self.index * 10, dtype=np.uint8)
        return True, image

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.frame_count)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.index + 1)
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.index / self.fps * 1000.0
        return 0.0

    def release(self) -> None:
        pass


def test_three_frame_extraction_records_scene_crop_time_index_and_hash(tmp_path: Path) -> None:
    tracking = [
        {
            "timestamp_sec": value,
            "frame_id": int(value * 2),
            "track_id": 7,
            "bbox": [3.0, 2.0, 18.0, 16.0],
            "track_confidence": 0.9,
        }
        for value in (1.0, 2.5, 3.5)
    ]
    final_root = tmp_path / "final"
    stage_root = tmp_path / "stage"
    stage_root.mkdir()

    frames = extract_episode_frames(
        episode=_episode(binary_label="wandering_like"),
        video_path=tmp_path / "fixture.mp4",
        tracking_rows=tracking,
        stage_output_root=stage_root,
        final_output_root=final_root,
        jpeg_quality=90,
        bbox_max_gap_seconds=0.6,
        crop_padding_ratio=0.1,
        capture_factory=_FakeCapture,
    )

    assert [frame.position for frame in frames] == ["start", "middle", "end"]
    assert all(frame.status == "ready" for frame in frames)
    assert all(frame.frame_index is not None for frame in frames)
    assert all(frame.selected_timestamp_sec is not None for frame in frames)
    assert all(frame.scene_ref["sha256"] for frame in frames)
    assert all(frame.person_crop_ref["sha256"] for frame in frames)
    assert all(Path(frame.scene_ref["artifact_path"]).is_relative_to(final_root) for frame in frames)
    assert len(list(stage_root.rglob("*.jpg"))) == 6


def test_context_row_is_schema_valid_and_preserves_episode_snapshot() -> None:
    episode = _episode(status="uncertain")
    before = copy.deepcopy(episode)
    frames = tuple(
        _ready_frame(position, index)
        for index, position in enumerate(("start", "middle", "end"), start=1)
    )

    row = _build_context_review_row(
        episode=episode,
        frames=frames,
        provider=DeterministicFakeContextProvider(),
        config_id="context-config",
        config_sha256="7" * 64,
        policy_id="context-trigger-v1",
        policy_sha256="8" * 64,
        source_refs=episode["source_refs"],
    )

    assert row["schema_version"] == CAMERA_CONTEXT_REVIEW_SCHEMA_VERSION
    assert row["episode_id"] == episode["episode_id"]
    assert row["status"] == "ready"
    assert row["error_code"] is None
    assert row["episode_identity"] == episode["identity"]
    assert len(row["frame_refs"]) == 3
    assert episode == before
    _validate_context_review(row)


def test_missing_frame_skips_provider_but_keeps_one_unavailable_row() -> None:
    episode = _episode(binary_label="wandering_like")
    frames = list(
        _ready_frame(position, index)
        for index, position in enumerate(("start", "middle", "end"), start=1)
    )
    frames[1] = ContextFrame.unavailable(
        position="middle",
        requested_timestamp_sec=2.5,
        error_code="frame_decode_failed",
    )

    row = _build_context_review_row(
        episode=episode,
        frames=tuple(frames),
        provider=DeterministicFakeContextProvider(),
        config_id="context-config",
        config_sha256="7" * 64,
        policy_id="context-trigger-v1",
        policy_sha256="8" * 64,
        source_refs=episode["source_refs"],
    )

    assert row["status"] == "unavailable"
    assert row["context_label"] == "unknown"
    assert row["error_code"] == "frame_decode_failed"
    assert row["confidence"] is None
    _validate_context_review(row)


def test_production_config_and_cli_help() -> None:
    config = load_camera_context_review_config(CONFIG)
    assert config["schema_version"] == "wandering-camera-context-review-config-v1"
    assert config["context_review_id"] == "w5d03a-b01-b02-context-core-v1"
    assert config["home_input_status"] == "awaiting_input"
    assert config["home_smoke_status"] == "not_run_input_unavailable"
    assert config["validation_scope"] == "b01_b02_development"
    assert config["provider"]["default_mode"] == "fake"

    completed = subprocess.run(
        [sys.executable, str(CLI), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--provider-mode" in completed.stdout
    assert "--output-dir" in completed.stdout
    assert "--source-video-id" in completed.stdout


def test_existing_output_is_rejected_before_any_provider_call(tmp_path: Path) -> None:
    from elderly_monitoring.modules.mental_health.wandering.camera_context_review import (
        build_camera_context_review_bundle,
    )

    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        build_camera_context_review_bundle(
            project_root=ROOT,
            config_path=CONFIG,
            output_dir=output,
        )
