"""V3.3 cognitive-function change clue feature pipeline."""

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.preprocess import (
    AudioWindow,
    FacePreprocessResult,
    make_audio_windows,
    normalize_cognitive_text,
    preprocess_face_frames,
    sample_frame_positions,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.quality import (
    ModalityQuality,
    compute_audio_quality,
    compute_face_quality,
    compute_text_quality,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.fusion import (
    MODALITY_ORDER,
    QualityAwareFusion,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.model import (
    CognitiveChangeClueModel,
    CognitiveModelOutput,
)

__all__ = [
    "AudioWindow",
    "FacePreprocessResult",
    "CognitiveChangeClueModel",
    "CognitiveModelOutput",
    "MODALITY_ORDER",
    "ModalityQuality",
    "QualityAwareFusion",
    "compute_audio_quality",
    "compute_face_quality",
    "compute_text_quality",
    "make_audio_windows",
    "normalize_cognitive_text",
    "preprocess_face_frames",
    "sample_frame_positions",
]
