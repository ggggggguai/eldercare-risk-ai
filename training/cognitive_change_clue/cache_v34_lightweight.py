"""Cache label-free V3.4 prosody, eGeMAPS, and text statistics.

The cache is intentionally restricted to CogPic official Train.  Raw values are
stored here; fold-specific normalization belongs to model training so outer-fold
information cannot leak into an inner training set.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import soundfile as sf

try:
    from .common import (
        ALGORITHM_ROOT,
        DEFAULT_MANIFEST_PATH,
        WORKSPACE_ROOT,
        atomic_write_json,
        atomic_write_jsonl,
        atomic_write_npz,
        resolve_workspace_path,
        sha256_file,
        workspace_relative,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        ALGORITHM_ROOT,
        DEFAULT_MANIFEST_PATH,
        WORKSPACE_ROOT,
        atomic_write_json,
        atomic_write_jsonl,
        atomic_write_npz,
        resolve_workspace_path,
        sha256_file,
        workspace_relative,
    )


SCHEMA_VERSION = "cognitive_v34_lightweight_feature_cache_v1"
RECORD_SCHEMA_VERSION = "cognitive_v34_lightweight_feature_record_v1"
FEATURE_VERSION = "cognitive_lightweight_features_v3.4.0"
DEFAULT_OUTPUT_ROOT = (
    ALGORITHM_ROOT
    / "data"
    / "processed"
    / "cognitive_change_clue"
    / "v3.4.0"
    / "lightweight_features"
)
ASR_INDEX_PATH = (
    ALGORITHM_ROOT
    / "data"
    / "processed"
    / "cognitive_change_clue"
    / "v3.3.0"
    / "asr_cache_index.jsonl"
)

AUDIO_BASIC_NAMES = (
    "duration_seconds",
    "speech_ratio",
    "silence_ratio",
    "voiced_seconds",
    "pause_count",
    "mean_pause_seconds",
    "longest_pause_seconds",
    "characters_per_second",
    "characters_per_voiced_second",
)
TEXT_STAT_NAMES = (
    "valid_character_count",
    "characters_per_second",
    "characters_per_voiced_second",
    "token_count",
    "lexical_diversity",
    "repetition_ratio",
    "self_correction_ratio",
    "filler_ratio",
    "mean_sentence_length",
    "content_word_ratio",
    "adjacent_semantic_coherence_proxy",
    "sentence_count",
)

_VALID_CHARACTER = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
_TOKEN_CHARACTER = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
_SENTENCE_BREAK = re.compile(r"[。！？!?；;]+")
_SELF_CORRECTIONS = ("不对", "不是", "我是说", "应该说", "改口", "说错了")
_FILLER_PATTERN = re.compile(r"嗯+|呃+|额+|啊+|那个|这个|就是")
_CONTENT_POS_PREFIXES = ("n", "v", "a", "m", "q", "r", "d")
_SMILE: Any | None = None


class V34LightweightFeatureError(RuntimeError):
    pass


def _finite_vector(values: Iterable[float], *, expected_dim: int, name: str) -> np.ndarray:
    vector = np.asarray(list(values), dtype=np.float32)
    if vector.shape != (expected_dim,) or not np.isfinite(vector).all():
        raise V34LightweightFeatureError(
            f"{name} must be a finite vector with shape ({expected_dim},)"
        )
    return vector


def _read_jsonl_index(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        if sample_id in records:
            raise V34LightweightFeatureError(f"duplicate ASR sample: {sample_id}")
        records[sample_id] = row
    return records


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    signal, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(signal, axis=1, dtype=np.float32)
    if mono.size == 0 or int(sample_rate) <= 0 or not np.isfinite(mono).all():
        raise V34LightweightFeatureError(f"invalid audio: {path}")
    return mono.astype(np.float32, copy=False), int(sample_rate)


def _resample_16k(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == 16000:
        return signal
    import librosa

    value = librosa.resample(signal, orig_sr=sample_rate, target_sr=16000)
    return np.asarray(value, dtype=np.float32)


def _vad_flags(signal_16k: np.ndarray, *, aggressiveness: int = 2) -> list[bool]:
    import webrtcvad

    frame_samples = 480  # 30 ms at 16 kHz
    clipped = np.clip(signal_16k, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2", copy=False)
    if pcm.size % frame_samples:
        pcm = np.pad(pcm, (0, frame_samples - pcm.size % frame_samples))
    vad = webrtcvad.Vad(int(aggressiveness))
    return [
        bool(vad.is_speech(pcm[start : start + frame_samples].tobytes(), 16000))
        for start in range(0, pcm.size, frame_samples)
    ]


def _pause_stats(
    voiced_flags: Sequence[bool],
    *,
    frame_seconds: float = 0.03,
    minimum_pause_seconds: float = 0.30,
) -> tuple[int, float, float, float]:
    voiced_indices = [index for index, value in enumerate(voiced_flags) if value]
    if not voiced_indices:
        return 0, 0.0, 0.0, 0.0
    voiced_seconds = len(voiced_indices) * frame_seconds
    pauses: list[float] = []
    current = 0
    for value in voiced_flags[voiced_indices[0] : voiced_indices[-1] + 1]:
        if value:
            if current * frame_seconds >= minimum_pause_seconds:
                pauses.append(current * frame_seconds)
            current = 0
        else:
            current += 1
    return (
        len(pauses),
        float(np.mean(pauses)) if pauses else 0.0,
        max(pauses, default=0.0),
        voiced_seconds,
    )


def _valid_characters(text: str) -> list[str]:
    return _VALID_CHARACTER.findall(text)


def _tokens(text: str) -> list[str]:
    import jieba

    return [
        token.strip().lower()
        for token in jieba.lcut(text, cut_all=False)
        if token.strip() and _TOKEN_CHARACTER.search(token)
    ]


def _content_word_ratio(text: str) -> float:
    import jieba.posseg as pseg

    tagged = [
        pair
        for pair in pseg.cut(text)
        if pair.word.strip() and _TOKEN_CHARACTER.search(pair.word)
    ]
    if not tagged:
        return 0.0
    content = sum(pair.flag.startswith(_CONTENT_POS_PREFIXES) for pair in tagged)
    return content / len(tagged)


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    chars = _valid_characters(text)
    if len(chars) < n:
        return set(chars)
    return {"".join(chars[index : index + n]) for index in range(len(chars) - n + 1)}


def _adjacent_coherence(sentences: Sequence[str]) -> float:
    if len(sentences) < 2:
        return 0.0
    scores = []
    for left, right in zip(sentences, sentences[1:]):
        left_set = _char_ngrams(left)
        right_set = _char_ngrams(right)
        union = left_set | right_set
        scores.append(len(left_set & right_set) / len(union) if union else 0.0)
    return float(np.mean(scores))


def text_statistics(
    text: str,
    *,
    duration_seconds: float,
    voiced_seconds: float,
) -> np.ndarray:
    chars = _valid_characters(text)
    tokens = _tokens(text)
    token_counts = Counter(tokens)
    sentences = [
        value.strip()
        for value in _SENTENCE_BREAK.split(text)
        if _valid_characters(value)
    ]
    repeated = sum(max(0, count - 1) for count in token_counts.values())
    correction_count = sum(text.count(marker) for marker in _SELF_CORRECTIONS)
    filler_count = len(_FILLER_PATTERN.findall(text))
    token_denominator = max(1, len(tokens))
    values = (
        float(len(chars)),
        len(chars) / max(duration_seconds, 1e-6),
        len(chars) / max(voiced_seconds, 1e-6),
        float(len(tokens)),
        len(token_counts) / token_denominator,
        repeated / token_denominator,
        correction_count / token_denominator,
        filler_count / token_denominator,
        float(np.mean([len(_valid_characters(value)) for value in sentences]))
        if sentences
        else 0.0,
        _content_word_ratio(text),
        _adjacent_coherence(sentences),
        float(len(sentences)),
    )
    return _finite_vector(values, expected_dim=len(TEXT_STAT_NAMES), name="text_stats")


def audio_basic_statistics(
    *,
    duration_seconds: float,
    voiced_flags: Sequence[bool],
    valid_character_count: int,
) -> np.ndarray:
    pause_count, mean_pause, longest_pause, voiced_seconds = _pause_stats(voiced_flags)
    speech_ratio = min(1.0, voiced_seconds / max(duration_seconds, 1e-6))
    values = (
        duration_seconds,
        speech_ratio,
        1.0 - speech_ratio,
        voiced_seconds,
        float(pause_count),
        mean_pause,
        longest_pause,
        valid_character_count / max(duration_seconds, 1e-6),
        valid_character_count / max(voiced_seconds, 1e-6),
    )
    return _finite_vector(values, expected_dim=len(AUDIO_BASIC_NAMES), name="audio_basic")


def _egemaps(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, tuple[str, ...]]:
    import opensmile

    global _SMILE
    if _SMILE is None:
        _SMILE = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    frame = _SMILE.process_signal(signal, sample_rate)
    if frame.shape != (1, 88):
        raise V34LightweightFeatureError(f"unexpected eGeMAPS shape: {frame.shape}")
    names = tuple(str(value) for value in frame.columns)
    values = _finite_vector(frame.iloc[0].to_numpy(), expected_dim=88, name="egemaps")
    return values, names


def _dependency_versions() -> dict[str, str]:
    packages = ("jieba", "librosa", "opensmile", "soundfile", "webrtcvad-wheels")
    return {name: importlib.metadata.version(name) for name in packages}


def _audio_source_path(row: Mapping[str, Any]) -> Path:
    audio_path = resolve_workspace_path(str(row["audio_path"]))
    if not audio_path.is_file():
        raise V34LightweightFeatureError(f"source is missing for {row['sample_id']}")
    return audio_path


def cache_features(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    asr_index_path: Path = ASR_INDEX_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    frame = pd.read_parquet(manifest_path, engine="pyarrow")
    train = frame.loc[frame["official_split"].astype(str).eq("train")].copy()
    train = train.sort_values("sample_id", kind="stable").reset_index(drop=True)
    if len(train) != 1377 or train["subject_id"].astype(str).nunique() != 459:
        raise V34LightweightFeatureError("official Train must contain 1377 tasks / 459 subjects")
    if limit is not None:
        train = train.iloc[: int(limit)].copy()
    asr_index = _read_jsonl_index(asr_index_path)
    feature_dir = output_root / "records"
    index_path = output_root / "lightweight_index.jsonl"
    rows: list[dict[str, Any]] = []
    missing_counts = Counter()
    egemaps_names: tuple[str, ...] | None = None
    for row in train.to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        asr = asr_index.get(sample_id)
        text_available = bool(asr and str(asr.get("status")) == "completed")
        audio_path = _audio_source_path(row)
        text_path = (
            resolve_workspace_path(str(asr["text_path"])) if text_available else None
        )
        transcript_path = (
            resolve_workspace_path(str(asr["transcript_path"])) if text_available else None
        )
        text_available = bool(
            text_available
            and text_path is not None
            and text_path.is_file()
            and transcript_path is not None
            and transcript_path.is_file()
        )
        source_hashes = {
            "audio": sha256_file(audio_path),
            "text": sha256_file(text_path) if text_available and text_path else None,
            "transcript": (
                sha256_file(transcript_path)
                if text_available and transcript_path
                else None
            ),
        }
        if source_hashes["audio"] != str(row["audio_sha256"]):
            raise V34LightweightFeatureError(f"audio hash mismatch: {sample_id}")
        if text_available and source_hashes["text"] != str(asr["text_sha256"]):
            raise V34LightweightFeatureError(f"text hash mismatch: {sample_id}")
        output_path = feature_dir / f"{sample_id}.npz"
        if output_path.is_file() and not force:
            with np.load(output_path, allow_pickle=False) as payload:
                audio_basic = np.asarray(payload["audio_basic"], dtype=np.float32)
                egemaps = np.asarray(payload["egemaps"], dtype=np.float32)
                text_stats = np.asarray(payload["text_stats"], dtype=np.float32)
                cached_hashes = json.loads(str(payload["source_hashes_json"].item()))
                cached_egemaps_names = tuple(
                    json.loads(str(payload["egemaps_names_json"].item()))
                )
            if cached_hashes != source_hashes:
                raise V34LightweightFeatureError(f"stale feature cache: {sample_id}")
            egemaps_names = egemaps_names or cached_egemaps_names
            if egemaps_names != cached_egemaps_names:
                raise V34LightweightFeatureError("eGeMAPS columns changed across cache rows")
        else:
            signal, sample_rate = _load_mono(audio_path)
            duration_seconds = signal.size / sample_rate
            signal_16k = _resample_16k(signal, sample_rate)
            flags = _vad_flags(signal_16k)
            text = text_path.read_text(encoding="utf-8").strip() if text_available and text_path else ""
            valid_count = len(_valid_characters(text))
            audio_basic = audio_basic_statistics(
                duration_seconds=duration_seconds,
                voiced_flags=flags,
                valid_character_count=valid_count,
            )
            voiced_seconds = float(audio_basic[AUDIO_BASIC_NAMES.index("voiced_seconds")])
            text_stats = (
                text_statistics(
                    text,
                    duration_seconds=duration_seconds,
                    voiced_seconds=voiced_seconds,
                )
                if text_available
                else np.zeros(len(TEXT_STAT_NAMES), dtype=np.float32)
            )
            egemaps, current_names = _egemaps(signal, sample_rate)
            egemaps_names = egemaps_names or current_names
            if egemaps_names != current_names:
                raise V34LightweightFeatureError("eGeMAPS columns changed across cache rows")
            atomic_write_npz(
                output_path,
                audio_basic=audio_basic,
                egemaps=egemaps,
                text_stats=text_stats,
                source_hashes_json=np.asarray(
                    json.dumps(source_hashes, ensure_ascii=False, sort_keys=True)
                ),
                egemaps_names_json=np.asarray(json.dumps(egemaps_names)),
            )
        for name, value, dimension in (
            ("audio_basic", audio_basic, len(AUDIO_BASIC_NAMES)),
            ("egemaps", egemaps, 88),
            ("text_stats", text_stats, len(TEXT_STAT_NAMES)),
        ):
            _finite_vector(value, expected_dim=dimension, name=name)
        if not text_available:
            missing_counts["text_stats"] += 1
        rows.append(
            {
                "schema_version": RECORD_SCHEMA_VERSION,
                "feature_version": FEATURE_VERSION,
                "sample_id": sample_id,
                "subject_id": str(row["subject_id"]),
                "official_split": "train",
                "feature_path": workspace_relative(output_path),
                "feature_sha256": sha256_file(output_path),
                "source_hashes": source_hashes,
                "dimensions": {
                    "audio_basic": len(AUDIO_BASIC_NAMES),
                    "egemaps": 88,
                    "text_stats": len(TEXT_STAT_NAMES),
                },
                "missing": {
                    "audio_basic": False,
                    "egemaps": False,
                    "text_stats": not text_available,
                },
                "missing_reasons": {
                    "audio_basic": None,
                    "egemaps": None,
                    "text_stats": None if text_available else "asr_text_unavailable",
                },
            }
        )
    atomic_write_jsonl(index_path, rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "status": "complete",
        "source_scope": "cogpic_official_train_only",
        "official_test_media_read": False,
        "expected_subjects": 459 if limit is None else len(set(train["subject_id"].astype(str))),
        "expected_rows": 1377 if limit is None else len(train),
        "indexed_rows": len(rows),
        "missing_counts": {
            "audio_basic": 0,
            "egemaps": 0,
            "text_stats": int(missing_counts["text_stats"]),
        },
        "manifest_path": workspace_relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "asr_index_path": workspace_relative(asr_index_path),
        "asr_index_sha256": sha256_file(asr_index_path),
        "index_path": workspace_relative(index_path),
        "index_sha256": sha256_file(index_path),
        "dependencies": _dependency_versions(),
        "extraction": {
            "vad": {
                "implementation": "webrtcvad",
                "sample_rate": 16000,
                "frame_ms": 30,
                "aggressiveness": 2,
                "minimum_internal_pause_ms": 300,
            },
            "egemaps": {
                "feature_set": "eGeMAPSv02",
                "feature_level": "Functionals",
            },
            "text": {
                "tokenizer": "jieba_precise",
                "coherence": "adjacent_clause_character_bigram_jaccard_proxy",
            },
        },
        "feature_groups": {
            "audio_basic": {"dimension": len(AUDIO_BASIC_NAMES), "names": list(AUDIO_BASIC_NAMES)},
            "egemaps": {"dimension": 88, "names": list(egemaps_names or ())},
            "text_stats": {"dimension": len(TEXT_STAT_NAMES), "names": list(TEXT_STAT_NAMES)},
        },
    }
    atomic_write_json(output_root / "cache_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--asr-index", type=Path, default=ASR_INDEX_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = cache_features(
            manifest_path=args.manifest,
            asr_index_path=args.asr_index,
            output_root=args.output_root,
            force=args.force,
            limit=args.limit,
        )
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIO_BASIC_NAMES",
    "DEFAULT_OUTPUT_ROOT",
    "FEATURE_VERSION",
    "TEXT_STAT_NAMES",
    "V34LightweightFeatureError",
    "_pause_stats",
    "audio_basic_statistics",
    "cache_features",
    "text_statistics",
]
