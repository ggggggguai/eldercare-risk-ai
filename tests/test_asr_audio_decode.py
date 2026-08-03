from __future__ import annotations

import base64
import io
import shutil
import subprocess
import wave

import pytest

from elderly_monitoring.modules.asr.audio_decode import decode_audio_input, decode_audio_path
from elderly_monitoring.modules.asr.schemas import AudioInput
from elderly_monitoring.modules.asr.settings import ASRSettings


def _wav_bytes(*, duration_ms: int = 3000, sample_rate: int = 8000, channels: int = 2) -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * channels * (sample_rate * duration_ms // 1000))
    return stream.getvalue()


def test_declared_metadata_is_checked_against_decoded_wav() -> None:
    source = base64.b64encode(_wav_bytes()).decode("ascii")
    decoded = decode_audio_input(
        AudioInput(
            source_type="base64",
            source=source,
            format="wav",
            sample_rate=8000,
            channels=2,
        ),
        ASRSettings(),
    )
    assert decoded.sample_rate == 16000
    assert decoded.original_sample_rate == 8000
    assert decoded.original_channels == 2
    with pytest.raises(ValueError, match="sample rate"):
        decode_audio_input(
            AudioInput(
                source_type="base64",
                source=source,
                format="wav",
                sample_rate=16000,
                channels=2,
            ),
            ASRSettings(),
        )
    with pytest.raises(ValueError, match="declared audio format"):
        decode_audio_input(
            AudioInput(
                source_type="base64",
                source=source,
                format="mp3",
                sample_rate=8000,
                channels=2,
            ),
            ASRSettings(),
        )


def test_audio_duration_must_be_between_three_and_sixty_seconds() -> None:
    source = base64.b64encode(_wav_bytes(duration_ms=1000)).decode("ascii")
    with pytest.raises(ValueError, match="duration"):
        decode_audio_input(
            AudioInput(
                source_type="base64",
                source=source,
                format="wav",
                sample_rate=None,
                channels=None,
            ),
            ASRSettings(),
        )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_m4a_aac_is_decoded_through_ffmpeg(tmp_path) -> None:
    output = tmp_path / "sample.m4a"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=3",
            "-c:a",
            "aac",
            "-y",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    decoded = decode_audio_path(output, ASRSettings())
    assert decoded.source_format == "m4a"
    assert decoded.sample_rate == 16000
    assert decoded.original_channels in {1, 2}
    assert 2900 <= decoded.duration_ms <= 3100
