"""HTTP-only client for the independently deployed ASR service."""

from __future__ import annotations

import base64
from typing import Protocol

import httpx

from elderly_monitoring.modules.asr.schemas import ASRRequest, ASRTranscript, AudioInput
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.media import (
    DecodedAudioMedia,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.schemas import (
    AudioMediaInput,
)


ASR_TIMEOUT_SECONDS = 20.0
ASR_MODEL_VERSION = "asr-paraformer-zh-v1.0"


class ASRClientProtocol(Protocol):
    def transcribe(
        self,
        *,
        request_id: str,
        audio: DecodedAudioMedia,
        declaration: AudioMediaInput,
    ) -> ASRTranscript: ...


class ASRClientError(RuntimeError):
    """Raised for timeout, transport, status, or response-contract failures."""


class ASRHttpClient:
    def __init__(
        self,
        *,
        url: str,
        api_token: str,
        timeout_seconds: float = ASR_TIMEOUT_SECONDS,
        client: httpx.Client | None = None,
    ) -> None:
        self.url = url
        self.api_token = api_token
        self.timeout_seconds = float(timeout_seconds)
        self._client = client

    def transcribe(
        self,
        *,
        request_id: str,
        audio: DecodedAudioMedia,
        declaration: AudioMediaInput,
    ) -> ASRTranscript:
        payload = ASRRequest(
            schema_version="asr_request_v1",
            request_id=f"{request_id}:asr",
            audio_input=AudioInput(
                source_type="base64",
                source=base64.b64encode(audio.payload).decode("ascii"),
                format=declaration.format,
                sample_rate=declaration.sample_rate,
                channels=declaration.channels,
            ),
            language="zh",
            enable_vad=True,
            enable_punctuation=True,
            return_timestamps=True,
            model_version=ASR_MODEL_VERSION,
        ).model_dump(mode="json")
        headers = {"Authorization": f"Bearer {self.api_token}"}
        try:
            if self._client is None:
                with httpx.Client(timeout=self.timeout_seconds, trust_env=False) as client:
                    response = client.post(self.url, json=payload, headers=headers)
            else:
                response = self._client.post(
                    self.url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
        except httpx.TimeoutException as exc:
            raise ASRClientError("ASR request timed out") from exc
        except httpx.HTTPError as exc:
            raise ASRClientError("ASR request failed") from exc
        if response.status_code != 200:
            raise ASRClientError(f"ASR returned HTTP {response.status_code}")
        try:
            return ASRTranscript.model_validate(response.json())
        except (ValueError, TypeError) as exc:
            raise ASRClientError("ASR returned an invalid response") from exc


__all__ = [
    "ASR_MODEL_VERSION",
    "ASR_TIMEOUT_SECONDS",
    "ASRClientError",
    "ASRClientProtocol",
    "ASRHttpClient",
]
