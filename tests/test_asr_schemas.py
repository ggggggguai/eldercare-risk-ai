import pytest
from pydantic import ValidationError

from elderly_monitoring.modules.asr.schemas import ASRRequest


def test_asr_request_accepts_versioned_base64_input() -> None:
    request = ASRRequest(
        request_id="asr-001",
        audio_input={
            "source_type": "base64",
            "source": "UklGRg==",
            "format": "wav",
            "sample_rate": None,
            "channels": None,
        },
    )
    assert request.schema_version == "asr_request_v1"
    assert request.model_version == "asr-paraformer-zh-v1.0"


@pytest.mark.parametrize("field", ["task_id", "prompt", "device_id", "file_path"])
def test_asr_request_rejects_business_and_storage_fields(field: str) -> None:
    payload = {
        "request_id": "asr-001",
        "audio_input": {
            "source_type": "base64",
            "source": "UklGRg==",
            "format": "wav",
            "sample_rate": None,
            "channels": None,
        },
        field: "not-part-of-asr",
    }
    with pytest.raises(ValidationError):
        ASRRequest.model_validate(payload)


def test_external_audio_input_rejects_local_path_source_type() -> None:
    with pytest.raises(ValidationError):
        ASRRequest(
            request_id="asr-001",
            audio_input={"source_type": "path", "source": "C:/sample.wav"},
        )


def test_first_stage_contract_keeps_vad_and_punctuation_enabled() -> None:
    with pytest.raises(ValidationError):
        ASRRequest(
            request_id="asr-001",
            audio_input={
                "source_type": "base64",
                "source": "UklGRg==",
                "format": "wav",
                "sample_rate": None,
                "channels": None,
            },
            enable_vad=False,
        )

    with pytest.raises(ValidationError):
        ASRRequest(
            request_id="asr-001",
            audio_input={
                "source_type": "base64",
                "source": "UklGRg==",
                "format": "wav",
                "sample_rate": None,
                "channels": None,
            },
            return_timestamps=False,
        )


def test_audio_declaration_matches_v3_3_ranges() -> None:
    request = ASRRequest(
        request_id="asr-001",
        audio_input={
            "source_type": "base64",
            "source": "UklGRg==",
            "format": "m4a",
            "sample_rate": 48000,
            "channels": 2,
        },
    )
    assert request.audio_input.format == "m4a"
    with pytest.raises(ValidationError):
        ASRRequest(
            request_id="asr-002",
            audio_input={
                "source_type": "base64",
                "source": "UklGRg==",
                "format": "aac",
                "sample_rate": 96000,
                "channels": 6,
            },
        )


def test_audio_metadata_fields_are_present_even_when_unknown() -> None:
    with pytest.raises(ValidationError):
        ASRRequest(
            request_id="asr-003",
            audio_input={
                "source_type": "base64",
                "source": "UklGRg==",
                "format": "wav",
            },
        )
