from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_episode_boundary_development import (
    CameraEpisodeBoundaryDevelopmentError,
    load_camera_episode_segmenter_search_config,
    rank_segmenter_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/modules/wandering_camera_segmenter_search_v1.yaml"
CONFIG_V2 = ROOT / "configs/modules/wandering_camera_segmenter_search_v2.yaml"


def _candidate(
    candidate_id: str,
    *,
    gate: bool = False,
    recall: float,
    minimum_batch_recall: float,
    coverage: float,
    f1: float,
    tiou: float,
    minimum_batch_f1: float | None = None,
    serious: int = 0,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "gate_satisfied": gate,
        "pooled_recall": recall,
        "minimum_batch_recall": minimum_batch_recall,
        "known_episode_candidate_coverage": coverage,
        "pooled_f1": f1,
        "minimum_batch_f1": f1 if minimum_batch_f1 is None else minimum_batch_f1,
        "matched_mean_tiou": tiou,
        "technical_hard_break_crossing_count": 0,
        "serious_split_merge_count": serious,
        "candidate_support": 10,
    }


def test_recall_first_ranking_prefers_complete_gate_then_stable_batch_floor() -> None:
    rows = [
        _candidate(
            "fallback-higher-recall",
            recall=0.90,
            minimum_batch_recall=0.50,
            coverage=0.95,
            f1=0.70,
            tiou=0.80,
        ),
        _candidate(
            "gate-lower-batch-floor",
            gate=True,
            recall=0.80,
            minimum_batch_recall=0.74,
            coverage=0.90,
            f1=0.75,
            tiou=0.70,
        ),
        _candidate(
            "gate-stable",
            gate=True,
            recall=0.80,
            minimum_batch_recall=0.78,
            coverage=0.88,
            f1=0.73,
            tiou=0.68,
        ),
    ]

    ranked = rank_segmenter_candidates(rows)

    assert [row["candidate_id"] for row in ranked] == [
        "gate-stable",
        "gate-lower-batch-floor",
        "fallback-higher-recall",
    ]


def test_recall_first_fallback_and_final_id_tie_break_are_deterministic() -> None:
    rows = [
        _candidate(
            "candidate-b",
            recall=0.82,
            minimum_batch_recall=0.76,
            coverage=0.91,
            f1=0.74,
            tiou=0.69,
            serious=2,
        ),
        _candidate(
            "lower-recall",
            recall=0.81,
            minimum_batch_recall=0.80,
            coverage=0.99,
            f1=0.90,
            tiou=0.90,
        ),
        _candidate(
            "candidate-a",
            recall=0.82,
            minimum_batch_recall=0.76,
            coverage=0.91,
            f1=0.74,
            tiou=0.69,
            serious=2,
        ),
    ]

    first = rank_segmenter_candidates(rows)
    second = rank_segmenter_candidates(list(reversed(rows)))

    expected = ["candidate-a", "candidate-b", "lower-recall"]
    assert [row["candidate_id"] for row in first] == expected
    assert [row["candidate_id"] for row in second] == expected


def test_search_config_rejects_undeclared_candidate_parameter_drift(
    tmp_path: Path,
) -> None:
    config = load_camera_episode_segmenter_search_config(CONFIG)
    drifted = deepcopy(config)
    drifted["candidates"][0]["undeclared_parameter"] = 1
    path = tmp_path / "drifted.yaml"
    path.write_text(yaml.safe_dump(drifted, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        CameraEpisodeBoundaryDevelopmentError,
        match="candidate fields drifted",
    ):
        load_camera_episode_segmenter_search_config(path)


def test_second_round_search_config_is_stationary_closing_only() -> None:
    config = load_camera_episode_segmenter_search_config(CONFIG_V2)

    assert config["search_id"] == "w5d01-b01-b02-stationary-closing-v2"
    assert len(config["candidates"]) == 16
    assert {
        candidate["parameter_group"] for candidate in config["candidates"][1:]
    } == {"stationary_closing"}


def test_segmenter_search_cli_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/wandering/tune_camera_episode_boundaries.py"),
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--work-dir" in completed.stdout
    assert "--output-dir" in completed.stdout
    assert "--config" in completed.stdout
