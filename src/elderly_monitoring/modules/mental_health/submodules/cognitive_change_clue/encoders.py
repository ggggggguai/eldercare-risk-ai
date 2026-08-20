"""Frozen offline encoders for V3.3 feature generation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.preprocess import (
    AudioWindow,
    make_audio_windows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_ASSET_ROOT = (
    PROJECT_ROOT / "models" / "mental_health" / "cognitive_change_clue" / "v3.3.0"
)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    root = Path(path)
    entries: list[str] = []
    for file_path in sorted((item for item in root.rglob("*") if item.is_file())):
        relative = file_path.relative_to(root).as_posix()
        entries.append(f"{sha256_file(file_path)}  {relative}\n")
    return hashlib.sha256("".join(entries).encode("utf-8")).hexdigest()


def _freeze(model: Any) -> Any:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class AudioEncoder:
    output_dim = 768

    def __init__(
        self,
        asset_path: str | Path | None = None,
        *,
        device: str = "cpu",
        model: Any | None = None,
    ) -> None:
        import torch

        self.device = torch.device(device)
        self.asset_path = Path(
            asset_path or DEFAULT_ASSET_ROOT / "encoders" / "wavlm_base_plus.pth"
        )
        if model is None:
            from torchaudio.pipelines import WAVLM_BASE_PLUS
            from torchaudio.pipelines._wav2vec2 import utils as wav2vec2_utils

            if not self.asset_path.is_file():
                raise FileNotFoundError(f"WavLM asset is missing: {self.asset_path}")
            model = wav2vec2_utils._get_model(
                WAVLM_BASE_PLUS._model_type,
                WAVLM_BASE_PLUS._params,
            )
            state_dict = torch.load(self.asset_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=True)
        self.model = _freeze(model).to(self.device)

    def encode(self, samples: np.ndarray, *, duration_ms: int | None = None) -> np.ndarray:
        windows = make_audio_windows(samples, duration_ms=duration_ms)
        embeddings = [self._encode_window(window) for window in windows]
        weights = np.asarray([window.weight_ms for window in windows], dtype=np.float64)
        pooled = np.average(np.stack(embeddings), axis=0, weights=weights)
        result = np.asarray(pooled, dtype=np.float32)
        if result.shape != (self.output_dim,):
            raise RuntimeError(f"WavLM returned unexpected shape {result.shape}")
        return result

    def encode_statistics(
        self, samples: np.ndarray, *, duration_ms: int | None = None
    ) -> np.ndarray:
        """Return weighted WavLM mean and standard deviation without fine-tuning."""

        windows = make_audio_windows(samples, duration_ms=duration_ms)
        means: list[np.ndarray] = []
        second_moments: list[np.ndarray] = []
        for window in windows:
            frames = self._encode_window_frames(window)
            means.append(frames.mean(axis=0, dtype=np.float64))
            second_moments.append(np.square(frames, dtype=np.float64).mean(axis=0))
        weights = np.asarray([window.weight_ms for window in windows], dtype=np.float64)
        pooled_mean = np.average(np.stack(means), axis=0, weights=weights)
        pooled_second = np.average(np.stack(second_moments), axis=0, weights=weights)
        pooled_std = np.sqrt(np.maximum(0.0, pooled_second - np.square(pooled_mean)))
        result = np.concatenate((pooled_mean, pooled_std)).astype(np.float32, copy=False)
        if result.shape != (self.output_dim * 2,) or not np.isfinite(result).all():
            raise RuntimeError(f"WavLM statistics pooling returned invalid shape {result.shape}")
        return result

    def _encode_window(self, window: AudioWindow) -> np.ndarray:
        return self._encode_window_frames(window).mean(axis=0).astype(np.float32, copy=False)

    def _encode_window_frames(self, window: AudioWindow) -> np.ndarray:
        import torch

        # Torchaudio's locked WavLM attention rejects padding masks. A single
        # unpadded window exposes exactly the real samples that a true mask
        # would expose, while retaining the frozen V3.3 window boundaries.
        waveform = (
            torch.from_numpy(window.waveform[: window.valid_samples])
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.inference_mode():
            features, feature_lengths = self.model.extract_features(waveform)
            last_layer = features[-1]
            valid_frames = (
                int(feature_lengths[0].item())
                if feature_lengths is not None
                else int(last_layer.shape[1])
            )
            if valid_frames <= 0:
                raise RuntimeError("WavLM returned no valid frames")
            frames = last_layer[0, :valid_frames]
        result = frames.detach().cpu().numpy().astype(np.float32, copy=False)
        if result.ndim != 2 or result.shape[0] <= 0 or result.shape[1] != self.output_dim:
            raise RuntimeError(f"WavLM returned unexpected frame shape {result.shape}")
        return result


class TextEncoder:
    output_dim = 768

    def __init__(
        self,
        asset_path: str | Path | None = None,
        *,
        device: str = "cpu",
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        import torch

        self.device = torch.device(device)
        self.asset_path = Path(asset_path or DEFAULT_ASSET_ROOT / "encoders" / "roberta")
        if tokenizer is None or model is None:
            from transformers import AutoModel, AutoTokenizer

            if not self.asset_path.is_dir():
                raise FileNotFoundError(f"RoBERTa asset is missing: {self.asset_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                self.asset_path,
                local_files_only=True,
                use_fast=True,
                do_lower_case=False,
            )
            model = AutoModel.from_pretrained(
                self.asset_path,
                local_files_only=True,
            )
        tokenizer.truncation_side = "right"
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        self.model = _freeze(model).to(self.device)

    def encode(self, texts: str | Sequence[str]) -> np.ndarray:
        import torch

        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        if not values:
            return np.empty((0, self.output_dim), dtype=np.float32)
        tokens = self.tokenizer(
            values,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        tokens = {name: value.to(self.device) for name, value in tokens.items()}
        with torch.inference_mode():
            output = self.model(**tokens).last_hidden_state[:, 0, :]
        result = output.detach().cpu().numpy().astype(np.float32, copy=False)
        if result.ndim != 2 or result.shape[1] != self.output_dim:
            raise RuntimeError(f"RoBERTa returned unexpected shape {result.shape}")
        return result[0] if single else result

    def encode_mean(self, texts: str | Sequence[str]) -> np.ndarray:
        """Mean-pool non-padding, non-special RoBERTa token representations."""

        import torch

        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        if not values:
            return np.empty((0, self.output_dim), dtype=np.float32)
        tokens = self.tokenizer(
            values,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        special = tokens.pop("special_tokens_mask").bool()
        tokens = {name: value.to(self.device) for name, value in tokens.items()}
        special = special.to(self.device)
        with torch.inference_mode():
            hidden = self.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].bool() & ~special
            if not torch.all(mask.any(dim=1)):
                raise RuntimeError("RoBERTa mean pooling received no content tokens")
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            output = (hidden * weights).sum(dim=1) / weights.sum(dim=1)
        result = output.detach().cpu().numpy().astype(np.float32, copy=False)
        if result.ndim != 2 or result.shape[1] != self.output_dim or not np.isfinite(result).all():
            raise RuntimeError(f"RoBERTa mean pooling returned invalid shape {result.shape}")
        return result[0] if single else result


class FaceEncoder:
    output_dim = 512

    def __init__(
        self,
        asset_path: str | Path | None = None,
        *,
        device: str = "cpu",
        model: Any | None = None,
    ) -> None:
        import torch

        self.device = torch.device(device)
        self.asset_path = Path(
            asset_path or DEFAULT_ASSET_ROOT / "encoders" / "resnet18_imagenet1k_v1.pth"
        )
        if model is None:
            from torchvision.models import resnet18

            if not self.asset_path.is_file():
                raise FileNotFoundError(f"ResNet18 asset is missing: {self.asset_path}")
            model = resnet18(weights=None)
            state_dict = torch.load(self.asset_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            model.fc = torch.nn.Identity()
        self.model = _freeze(model).to(self.device)

    def encode(self, tensors: Iterable[np.ndarray], *, batch_size: int = 16) -> np.ndarray:
        import torch

        values = tuple(np.asarray(value, dtype=np.float32) for value in tensors)
        if not values:
            raise ValueError("at least one valid face tensor is required")
        outputs: list[Any] = []
        for offset in range(0, len(values), max(1, int(batch_size))):
            batch = torch.from_numpy(np.stack(values[offset : offset + batch_size])).to(self.device)
            with torch.inference_mode():
                outputs.append(self.model(batch).detach().cpu())
        pooled = torch.cat(outputs, dim=0).mean(dim=0)
        result = pooled.numpy().astype(np.float32, copy=False)
        if result.shape != (self.output_dim,):
            raise RuntimeError(f"ResNet18 returned unexpected shape {result.shape}")
        return result


__all__ = [
    "AudioEncoder",
    "DEFAULT_ASSET_ROOT",
    "FaceEncoder",
    "PROJECT_ROOT",
    "TextEncoder",
    "sha256_directory",
    "sha256_file",
]
