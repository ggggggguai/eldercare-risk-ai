"""API-003 package decision and fallback-loader tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from elderly_monitoring.modules.mental_health.mood_social.package_selection import (
    DEFAULT_ART002_DECISION_PATH,
    PackageSelectionError,
    load_active_package_selection,
)
from elderly_monitoring.modules.mental_health.mood_social.pipeline import (
    MoodSocialPipeline,
)


ROOT = Path(__file__).resolve().parents[1]


def test_art002_no_promotion_resolves_the_read_only_art001_package() -> None:
    selection = load_active_package_selection(repository_root=ROOT)
    assert selection.promotion_passed is False
    assert selection.package_run_id == "MH-20260802-013"
    assert selection.model_version == "mood-fusion-v3.3.3"
    assert selection.package_directory.name == "MH-20260802-013"
    pipeline = MoodSocialPipeline.from_package()
    assert pipeline.package.package_directory == selection.package_directory
    assert pipeline.package.manifest["run_id"] == "MH-20260802-013"


def test_package_selection_rejects_a_false_promotion_claim(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_ART002_DECISION_PATH.read_text(encoding="utf-8"))
    payload["promotion_passed"] = True
    drifted = tmp_path / "decision.json"
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PackageSelectionError, match="no-promotion"):
        load_active_package_selection(drifted, repository_root=ROOT)


def test_package_selection_rejects_a_path_outside_the_model_root(
    tmp_path: Path,
) -> None:
    payload = json.loads(DEFAULT_ART002_DECISION_PATH.read_text(encoding="utf-8"))
    payload["active_package_directory"] = str(tmp_path)
    drifted = tmp_path / "decision.json"
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PackageSelectionError, match="escapes"):
        load_active_package_selection(drifted, repository_root=ROOT)
