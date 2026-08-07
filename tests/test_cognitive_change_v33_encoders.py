import numpy as np
import pytest
import torch
from types import SimpleNamespace

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.encoders import (
    AudioEncoder,
    FaceEncoder,
    TextEncoder,
)


class _FakeWavLM(torch.nn.Module):
    def extract_features(self, waveform, lengths=None):
        assert lengths is None
        value = float(waveform.shape[1])
        features = torch.full((1, 2, 768), value, device=waveform.device)
        return [features], None


def test_audio_encoder_uses_valid_unpadded_samples_and_duration_weights() -> None:
    encoder = AudioEncoder(device="cpu", model=_FakeWavLM())
    result = encoder.encode(np.ones(25 * 16_000, dtype=np.float32), duration_ms=25_000)
    expected = (320_000 * 10_000 + 240_000 * 15_000) / 25_000
    assert result.shape == (768,)
    assert result[0] == pytest.approx(expected)
    assert np.all(result == result[0])


def test_audio_encoder_statistics_pooling_adds_finite_standard_deviation() -> None:
    encoder = AudioEncoder(device="cpu", model=_FakeWavLM())
    result = encoder.encode_statistics(
        np.ones(25 * 16_000, dtype=np.float32), duration_ms=25_000
    )
    assert result.shape == (1536,)
    assert np.isfinite(result).all()
    assert np.all(result[768:] > 0.0)


class _FakeResNet(torch.nn.Module):
    def forward(self, values):
        means = values.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return means.repeat(1, 512)


def test_face_encoder_means_valid_frame_embeddings() -> None:
    encoder = FaceEncoder(device="cpu", model=_FakeResNet())
    result = encoder.encode(
        (
            np.zeros((3, 224, 224), dtype=np.float32),
            np.ones((3, 224, 224), dtype=np.float32),
        )
    )
    assert result.shape == (512,)
    assert np.allclose(result, 0.5)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.truncation_side = "left"
        self.padding_side = "left"
        self.calls = []

    def __call__(self, values, **kwargs):
        self.calls.append((list(values), dict(kwargs)))
        batch_size = len(values)
        input_ids = torch.arange(256).repeat(batch_size, 1)
        attention_mask = torch.ones((batch_size, 256), dtype=torch.long)
        result = {"input_ids": input_ids, "attention_mask": attention_mask}
        if kwargs.get("return_special_tokens_mask"):
            special = torch.zeros((batch_size, 256), dtype=torch.long)
            special[:, 0] = 1
            special[:, -1] = 1
            result["special_tokens_mask"] = special
        return result


class _FakeRoBERTa(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        batch_size, sequence_length = input_ids.shape
        values = torch.arange(768, dtype=torch.float32).reshape(1, 1, 768)
        hidden = values.repeat(batch_size, sequence_length, 1)
        return SimpleNamespace(last_hidden_state=hidden)


def test_text_encoder_uses_fixed_truncation_and_deterministic_cls_vector() -> None:
    tokenizer = _FakeTokenizer()
    encoder = TextEncoder(
        device="cpu",
        tokenizer=tokenizer,
        model=_FakeRoBERTa(),
    )
    first = encoder.encode("老人描述图片" * 100)
    second = encoder.encode("老人描述图片" * 100)
    assert tokenizer.truncation_side == "right"
    assert tokenizer.padding_side == "right"
    assert tokenizer.calls[0][1] == {
        "padding": "max_length",
        "truncation": True,
        "max_length": 256,
        "return_tensors": "pt",
    }
    assert first.shape == (768,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


class _PositionRoBERTa(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        hidden = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 768)
        return SimpleNamespace(last_hidden_state=hidden)


def test_text_encoder_mean_pooling_excludes_special_tokens() -> None:
    tokenizer = _FakeTokenizer()
    encoder = TextEncoder(
        device="cpu",
        tokenizer=tokenizer,
        model=_PositionRoBERTa(),
    )
    result = encoder.encode_mean("老人描述图片")
    assert result.shape == (768,)
    assert np.isfinite(result).all()
    assert result[0] == pytest.approx(np.mean(np.arange(1, 255)))
    assert tokenizer.calls[0][1]["return_special_tokens_mask"] is True
