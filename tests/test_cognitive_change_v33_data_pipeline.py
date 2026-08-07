import json
import wave
from pathlib import Path

import numpy as np
import pandas as pd

from elderly_monitoring.modules.asr.schemas import (
    ASRQuality,
    ASRSegment,
    ASRTranscript,
)
from training.cognitive_change_clue import build_cogpic_manifest as manifest_module
from training.cognitive_change_clue import build_subject_split as split_module
from training.cognitive_change_clue import cache_features as feature_module
from training.cognitive_change_clue import common
from training.cognitive_change_clue import run_asr_cache as asr_module


def _write_wav(path: Path, seconds: float = 3.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(int(16_000 * seconds), dtype="<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(samples.tobytes())


def _make_cogpic_fixture(root: Path) -> None:
    for official in ("Train", "Test"):
        numbers = (
            {"AD": "001", "MCI": "002", "HC": "003"}
            if official == "Train"
            else {"AD": "101", "MCI": "102", "HC": "103"}
        )
        for diagnosis, number in numbers.items():
            subject = root / official / diagnosis / f"{diagnosis}_subj_{number}_1_70_1_28_24"
            for task_number in (1, 2, 3):
                task = subject / f"pic_{task_number}"
                _write_wav(task / "audio.wav")
                (task / "audio.txt").write_text("老人描述图片", encoding="utf-8")
                face = task / "frames_face"
                face.mkdir(parents=True)
                (face / "frame_0001.jpg").write_bytes(b"fixture")


def test_manifest_and_subject_split_are_deterministic_and_leak_free(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(common, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(manifest_module, "DEFAULT_PROCESSED_ROOT", tmp_path / "processed")
    dataset_root = tmp_path / "CogPic"
    _make_cogpic_fixture(dataset_root)
    manifest_path = tmp_path / "processed" / "manifest.parquet"
    report_path = tmp_path / "processed" / "report.json"
    report = manifest_module.build_manifest(
        dataset_root=dataset_root,
        output_path=manifest_path,
        report_path=report_path,
        split_path=tmp_path / "missing-split.json",
        expected_counts=None,
    )
    assert report["counts"]["subjects"] == 6
    assert report["counts"]["tasks"] == 18
    frame = pd.read_parquet(manifest_path)
    assert frame["sample_id"].nunique() == 18
    assert set(frame["subject_id"]) == {
        "AD_subj_001",
        "MCI_subj_002",
        "HC_subj_003",
        "AD_subj_101",
        "MCI_subj_102",
        "HC_subj_103",
    }
    assert all(str(path).startswith("CogPic/") for path in frame["audio_path"])

    split_path = tmp_path / "processed" / "split.json"
    first = split_module.build_subject_split(
        manifest_path=manifest_path,
        output_path=split_path,
        strict_counts=False,
        validation_total=1,
    )
    second = split_module.build_subject_split(
        manifest_path=manifest_path,
        output_path=split_path,
        strict_counts=False,
        validation_total=1,
    )
    assert first["splits"] == second["splits"]
    assert first["counts"] == {"train": 2, "validation": 1, "test": 3}
    updated = pd.read_parquet(manifest_path)
    assert updated["derived_split"].isna().sum() == 0
    assert updated.groupby("subject_id")["derived_split"].nunique().max() == 1


def test_asr_cache_resumes_from_hash_valid_index(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(common, "WORKSPACE_ROOT", tmp_path)
    audio_path = tmp_path / "source" / "audio.wav"
    original_path = tmp_path / "source" / "audio.txt"
    transcript_path = tmp_path / "cache" / "transcript.json"
    text_path = tmp_path / "cache" / "text.txt"
    _write_wav(audio_path)
    original_path.write_text("老人描述图片", encoding="utf-8")
    manifest_path = tmp_path / "manifest.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": "HC_subj_001__pic_1",
                "subject_id": "HC_subj_001",
                "audio_path": "source/audio.wav",
                "original_text_path": "source/audio.txt",
                "asr_text_path": "cache/text.txt",
                "asr_transcript_path": "cache/transcript.json",
                "asr_status": None,
                "asr_quality": None,
            }
        ]
    ).to_parquet(manifest_path, index=False)
    calls = []

    def fake_transcribe(audio_source, *, request_id, settings):
        calls.append((audio_source, request_id, settings.device))
        return ASRTranscript(
            request_id=request_id,
            status="completed",
            text="老人描述图片",
            segments=[
                ASRSegment(
                    segment_id="seg-001",
                    start_ms=0,
                    end_ms=2500,
                    text="老人描述图片",
                    confidence=0.8,
                )
            ],
            quality=ASRQuality(
                audio_duration_ms=3000,
                speech_duration_ms=2500,
                speech_ratio=0.8333,
                mean_confidence=0.8,
                decode_status="ok",
                vad_status="ok",
            ),
        )

    monkeypatch.setattr(asr_module, "transcribe_audio", fake_transcribe)
    index_path = tmp_path / "cache" / "index.jsonl"
    report_path = tmp_path / "cache" / "report.json"
    first = asr_module.run_asr_cache(
        manifest_path=manifest_path,
        index_path=index_path,
        report_path=report_path,
        device="cpu",
        checkpoint_every=1,
    )
    second = asr_module.run_asr_cache(
        manifest_path=manifest_path,
        index_path=index_path,
        report_path=report_path,
        device="cpu",
        checkpoint_every=1,
    )
    assert len(calls) == 1
    assert first["complete"] is True and second["complete"] is True
    assert json.loads(transcript_path.read_text(encoding="utf-8"))["text"] == "老人描述图片"
    assert text_path.read_text(encoding="utf-8") == "老人描述图片"
    updated = pd.read_parquet(manifest_path)
    assert updated.loc[0, "asr_status"] == "completed"


class _FakeAudioEncoder:
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, samples, *, duration_ms):
        type(self).calls += 1
        return np.ones(768, dtype=np.float32)


def test_feature_cache_resumes_without_reencoding_valid_npz(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(common, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(feature_module, "AudioEncoder", _FakeAudioEncoder)
    _FakeAudioEncoder.calls = 0
    audio_path = tmp_path / "source" / "audio.wav"
    transcript_path = tmp_path / "source" / "transcript.json"
    _write_wav(audio_path)
    transcript = ASRTranscript(
        request_id="fixture",
        status="completed",
        text="老人描述图片",
        quality=ASRQuality(
            audio_duration_ms=3000,
            speech_duration_ms=2500,
            speech_ratio=0.8333,
            decode_status="ok",
            vad_status="ok",
        ),
    )
    transcript_path.write_text(transcript.model_dump_json(), encoding="utf-8")
    manifest_path = tmp_path / "manifest.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": "HC_subj_001__pic_1",
                "subject_id": "HC_subj_001",
                "derived_split": "train",
                "audio_path": "source/audio.wav",
                "asr_transcript_path": "source/transcript.json",
                "face_path": "source/faces",
            }
        ]
    ).to_parquet(manifest_path, index=False)
    split_path = tmp_path / "split.json"
    split_path.write_text("{}\n", encoding="utf-8")
    asset_root = tmp_path / "assets"
    for path in (
        asset_root / "encoders" / "wavlm_base_plus.pth",
        asset_root / "encoders" / "roberta" / "config.json",
        asset_root / "encoders" / "resnet18_imagenet1k_v1.pth",
        asset_root / "detectors" / "mediapipe_face_detection_short_range.tflite",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    feature_root = tmp_path / "features"
    first = feature_module.cache_features(
        manifest_path=manifest_path,
        split_path=split_path,
        feature_root=feature_root,
        asset_root=asset_root,
        modalities=("audio",),
        device="cpu",
        checkpoint_every=1,
    )
    second = feature_module.cache_features(
        manifest_path=manifest_path,
        split_path=split_path,
        feature_root=feature_root,
        asset_root=asset_root,
        modalities=("audio",),
        device="cpu",
        checkpoint_every=1,
    )
    assert _FakeAudioEncoder.calls == 1
    assert first["modalities"]["audio"]["ready_for_training"] is True
    assert second["modalities"]["audio"]["ready_for_training"] is True
    index = [json.loads(line) for line in (feature_root / "audio_index.jsonl").read_text().splitlines()]
    assert index[0]["output_dim"] == 768
    assert index[0]["quality"] == 0.8
    assert np.load(feature_root / "audio" / "HC_subj_001__pic_1.npz")["embedding"].shape == (
        768,
    )


def test_atomic_write_retries_short_lived_windows_reader_lock(tmp_path, monkeypatch) -> None:
    target = tmp_path / "index.jsonl"
    real_replace = common.os.replace
    attempts = []

    def flaky_replace(source, destination):
        attempts.append((source, destination))
        if len(attempts) < 3:
            raise PermissionError("simulated Windows sharing violation")
        return real_replace(source, destination)

    monkeypatch.setattr(common.os, "replace", flaky_replace)
    monkeypatch.setattr(common.time, "sleep", lambda _seconds: None)
    common.atomic_write_jsonl(target, ({"sample_id": "sample-001"},))
    assert len(attempts) == 3
    assert json.loads(target.read_text(encoding="utf-8"))["sample_id"] == "sample-001"
