from __future__ import annotations

import numpy as np

from training.cognitive_change_clue.cache_v34_lightweight import (
    AUDIO_BASIC_NAMES,
    TEXT_STAT_NAMES,
    _pause_stats,
    audio_basic_statistics,
    text_statistics,
)


def test_pause_stats_count_only_internal_runs_at_least_300ms() -> None:
    flags = [False] * 3 + [True] * 4 + [False] * 9 + [True] * 2 + [False] * 10 + [True]
    count, mean_seconds, longest_seconds, voiced_seconds = _pause_stats(flags)
    assert count == 1
    assert mean_seconds == 0.3
    assert longest_seconds == 0.3
    assert voiced_seconds == 0.21


def test_audio_basic_statistics_are_finite_and_length_normalized() -> None:
    vector = audio_basic_statistics(
        duration_seconds=2.0,
        voiced_flags=[True] * 20 + [False] * 10 + [True] * 20,
        valid_character_count=10,
    )
    values = dict(zip(AUDIO_BASIC_NAMES, vector.tolist()))
    assert vector.shape == (len(AUDIO_BASIC_NAMES),)
    assert np.isfinite(vector).all()
    assert values["characters_per_second"] == 5.0
    assert values["pause_count"] == 1.0


def test_text_statistics_do_not_use_task_or_label_fields() -> None:
    vector = text_statistics(
        "嗯，这个孩子在洗碗。不对，他在擦盘子。",
        duration_seconds=10.0,
        voiced_seconds=8.0,
    )
    values = dict(zip(TEXT_STAT_NAMES, vector.tolist()))
    assert vector.shape == (len(TEXT_STAT_NAMES),)
    assert np.isfinite(vector).all()
    assert values["valid_character_count"] > 0
    assert values["characters_per_second"] > 0
    assert values["self_correction_ratio"] > 0
    assert values["filler_ratio"] > 0
    assert values["sentence_count"] == 2.0
