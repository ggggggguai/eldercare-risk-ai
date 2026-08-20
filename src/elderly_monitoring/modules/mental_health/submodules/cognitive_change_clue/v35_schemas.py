"""Subject-level HTTP contract for the frozen V3.5 cognitive model."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.schemas import (
    ASRStatus,
    AudioMediaInput,
    CaptureWindow,
    CognitiveASROptions,
    CognitiveLevel,
    CognitiveModalityQuality,
    CognitiveSchemaModel,
    CognitiveStatus,
    CognitiveWarning,
    FaceMediaInput,
    ModalityName,
    Probability,
    RequestId,
)


V35_MODEL_VERSION = "cognitive-mm-v3.5.0"
V35_PICTURE_TASK_PROTOCOL = "cogpic_picture_description_3task_v1"
V35_VOICE_TASK_PROTOCOL = "cogvoice_scene_narration_3task_v1"
# Backwards-compatible alias for clients referring to the validated default.
V35_TASK_PROTOCOL = V35_PICTURE_TASK_PROTOCOL
V35_REQUEST_SCHEMA_VERSION = "cognitive_subject_infer_request_v1"
V35_RESPONSE_SCHEMA_VERSION = "cognitive_subject_infer_response_v1"


class CognitiveV35TaskInput(CognitiveSchemaModel):
    task_slot: Literal[1, 2, 3]
    capture_window: CaptureWindow
    audio_input: AudioMediaInput | None
    face_input: FaceMediaInput | None


class CognitiveV35InferRequest(CognitiveSchemaModel):
    schema_version: Literal["cognitive_subject_infer_request_v1"]
    request_id: RequestId
    task_protocol: Literal[
        "cogpic_picture_description_3task_v1",
        "cogvoice_scene_narration_3task_v1",
    ]
    stimulus_mode: Literal["visual_image", "audio_prompt"] = "visual_image"
    tasks: list[CognitiveV35TaskInput] = Field(min_length=3, max_length=3)
    asr_options: CognitiveASROptions
    model_version: Literal["cognitive-mm-v3.5.0"] = V35_MODEL_VERSION

    @model_validator(mode="after")
    def validate_task_slots(self) -> "CognitiveV35InferRequest":
        if [task.task_slot for task in self.tasks] != [1, 2, 3]:
            raise ValueError("tasks must contain task_slot 1, 2, 3 in order")
        expected_mode = (
            "audio_prompt"
            if self.task_protocol == V35_VOICE_TASK_PROTOCOL
            else "visual_image"
        )
        if self.stimulus_mode != expected_mode:
            raise ValueError("stimulus_mode does not match task_protocol")
        return self


class CognitiveV35TaskQuality(CognitiveSchemaModel):
    task_slot: Literal[1, 2, 3]
    status: Literal["completed", "degraded", "insufficient_input"]
    used_modalities: list[ModalityName]
    modality_quality: CognitiveModalityQuality
    asr_status: ASRStatus
    warnings: list[CognitiveWarning]

    @model_validator(mode="after")
    def validate_task_shape(self) -> "CognitiveV35TaskQuality":
        order = ["audio", "text", "face"]
        if self.used_modalities != sorted(set(self.used_modalities), key=order.index):
            raise ValueError("task used_modalities must be unique and ordered")
        if self.status == "completed" and self.used_modalities != order:
            raise ValueError("completed task requires audio, text, and face")
        if self.status == "insufficient_input" and self.used_modalities:
            raise ValueError("insufficient task must not expose used modalities")
        return self


class CognitiveV35InferResponse(CognitiveSchemaModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["cognitive_subject_infer_response_v1"]
    request_id: RequestId
    task_protocol: Literal[
        "cogpic_picture_description_3task_v1",
        "cogvoice_scene_narration_3task_v1",
    ]
    stimulus_mode: Literal["visual_image", "audio_prompt"]
    adaptation_mode: Literal[
        "standard_picture_description",
        "voice_prompt_adaptation",
    ]
    status: CognitiveStatus
    cognitive_clue_score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
    )
    cognitive_clue_level: CognitiveLevel | None
    confidence: Probability | None
    research_outputs: None = None
    used_modalities: list[ModalityName]
    modality_quality: CognitiveModalityQuality
    task_quality: list[CognitiveV35TaskQuality] = Field(min_length=3, max_length=3)
    asr_status: ASRStatus
    model_version: Literal["cognitive-mm-v3.5.0"]
    warnings: list[CognitiveWarning]

    @model_validator(mode="after")
    def validate_response_shape(self) -> "CognitiveV35InferResponse":
        expected = (
            ("audio_prompt", "voice_prompt_adaptation")
            if self.task_protocol == V35_VOICE_TASK_PROTOCOL
            else ("visual_image", "standard_picture_description")
        )
        if (self.stimulus_mode, self.adaptation_mode) != expected:
            raise ValueError("stimulus/adaptation mode does not match task_protocol")
        if self.adaptation_mode == "voice_prompt_adaptation" and not any(
            item.code == "voice_prompt_adaptation" for item in self.warnings
        ):
            raise ValueError("voice-prompt response must expose adaptation warning")
        if (
            self.adaptation_mode == "voice_prompt_adaptation"
            and self.confidence is not None
            and self.confidence > 0.6
        ):
            raise ValueError("voice-prompt confidence must not exceed 0.600")
        order = ["audio", "text", "face"]
        if self.used_modalities != sorted(set(self.used_modalities), key=order.index):
            raise ValueError("used_modalities must be unique and ordered")
        if [item.task_slot for item in self.task_quality] != [1, 2, 3]:
            raise ValueError("task_quality must use task_slot 1, 2, 3 in order")
        warning_codes = [item.code for item in self.warnings]
        if len(warning_codes) != len(set(warning_codes)):
            raise ValueError("global warning codes must not repeat")
        scores = (
            self.cognitive_clue_score,
            self.cognitive_clue_level,
            self.confidence,
        )
        if self.status == "insufficient_input":
            if any(value is not None for value in scores) or self.used_modalities:
                raise ValueError("insufficient_input must not expose model results")
            return self
        if any(value is None for value in scores) or "audio" not in self.used_modalities:
            raise ValueError("available V3.5 result requires audio and model results")
        all_tasks_completed = all(item.status == "completed" for item in self.task_quality)
        if self.status == "completed" and (
            self.used_modalities != order or not all_tasks_completed
        ):
            raise ValueError("completed V3.5 result requires all modalities in all tasks")
        if self.status == "degraded" and all_tasks_completed:
            raise ValueError("degraded V3.5 result must contain a degraded task")
        return self


__all__ = [
    "V35_MODEL_VERSION",
    "V35_PICTURE_TASK_PROTOCOL",
    "V35_TASK_PROTOCOL",
    "V35_VOICE_TASK_PROTOCOL",
    "V35_REQUEST_SCHEMA_VERSION",
    "V35_RESPONSE_SCHEMA_VERSION",
    "CognitiveV35InferRequest",
    "CognitiveV35InferResponse",
    "CognitiveV35TaskInput",
    "CognitiveV35TaskQuality",
]
