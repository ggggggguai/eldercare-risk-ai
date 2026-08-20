"""Lazy, thread-safe FunASR Paraformer runtime adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from elderly_monitoring.modules.asr.audio_decode import DecodedAudio
from elderly_monitoring.modules.asr.integrity import (
    ASRPackageIntegrityError,
    verify_model_package,
)
from elderly_monitoring.modules.asr.settings import ASRSettings


class ASRModelUnavailableError(RuntimeError):
    """Raised when the local model package or ASR runtime is unavailable."""


@dataclass(frozen=True)
class EngineResult:
    text: str
    sentence_info: list[dict[str, Any]] = field(default_factory=list)
    timestamps: list[list[int]] = field(default_factory=list)


class ParaformerEngine:
    def __init__(self, settings: ASRSettings | None = None) -> None:
        self.settings = settings or ASRSettings.load()
        self._model: Any | None = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def warmup(self) -> None:
        """Load all local model components before the service accepts traffic."""

        self._get_model()

    def transcribe(self, audio: DecodedAudio) -> EngineResult:
        model = self._get_model()
        try:
            with self._inference_lock:
                raw = model.generate(
                    input=audio.samples,
                    cache={},
                    batch_size_s=self.settings.batch_size_seconds,
                    merge_vad=True,
                    merge_length_s=self.settings.merge_length_seconds,
                    pred_timestamp=True,
                    sentence_timestamp=True,
                )
        except Exception as exc:
            raise RuntimeError("FunASR inference failed") from exc
        if not isinstance(raw, list) or not raw or not isinstance(raw[0], dict):
            return EngineResult(text="")
        result = raw[0]
        sentence_info = result.get("sentence_info")
        timestamps = result.get("timestamp")
        return EngineResult(
            text=str(result.get("text") or "").strip(),
            sentence_info=list(sentence_info) if isinstance(sentence_info, list) else [],
            timestamps=list(timestamps) if isinstance(timestamps, list) else [],
        )

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            paths = {
                "model": self.settings.model_root / "paraformer",
                "vad_model": self.settings.model_root / "fsmn-vad",
                "punc_model": self.settings.model_root / "ct-punc",
            }
            missing = [
                str(directory)
                for directory in paths.values()
                if not (directory / "config.yaml").is_file()
                or not (directory / "model.pt").is_file()
            ]
            if missing:
                raise ASRModelUnavailableError(
                    "local ASR model package is incomplete: " + ", ".join(missing)
                )
            try:
                verify_model_package(self.settings.model_root)
            except ASRPackageIntegrityError as exc:
                raise ASRModelUnavailableError(
                    "local ASR model package integrity check failed"
                ) from exc
            try:
                import torch
                from funasr import AutoModel
            except Exception as exc:
                raise ASRModelUnavailableError("FunASR runtime is unavailable") from exc
            device = self.settings.device
            if device == "auto":
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            try:
                self._model = AutoModel(
                    model=str(paths["model"]),
                    vad_model=str(paths["vad_model"]),
                    punc_model=str(paths["punc_model"]),
                    device=device,
                    disable_update=True,
                    check_latest=False,
                    vad_kwargs={"check_latest": False},
                    punc_kwargs={"check_latest": False},
                )
            except Exception as exc:
                raise ASRModelUnavailableError("local ASR model could not be loaded") from exc
            return self._model


_DEFAULT_ENGINES: dict[ASRSettings, ParaformerEngine] = {}
_DEFAULT_ENGINES_LOCK = Lock()


def get_default_engine(settings: ASRSettings | None = None) -> ParaformerEngine:
    """Return one engine per effective runtime configuration for this process."""

    runtime_settings = settings or ASRSettings.load()
    with _DEFAULT_ENGINES_LOCK:
        engine = _DEFAULT_ENGINES.get(runtime_settings)
        if engine is None:
            engine = ParaformerEngine(runtime_settings)
            _DEFAULT_ENGINES[runtime_settings] = engine
        return engine


def prewarm_default_engine(settings: ASRSettings | None = None) -> ParaformerEngine:
    engine = get_default_engine(settings)
    engine.warmup()
    return engine
