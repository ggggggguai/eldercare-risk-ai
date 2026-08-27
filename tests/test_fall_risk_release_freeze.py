from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


RELEASE_MANIFEST = Path("configs/modules/fall_risk_release_v1.yaml")
RELEASE_V2_MANIFEST = Path("configs/modules/fall_risk_release_v2.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_fall_risk_competition_release_is_frozen_and_hash_bound() -> None:
    release = yaml.safe_load(RELEASE_MANIFEST.read_text(encoding="utf-8"))

    assert release["status"] == "competition_release_frozen"
    assert release["stage"] == "formal_model_replacement_complete"
    assert len(release["runtime_source_revision"]) == 40
    assert release["change_policy"]["requires_new_release_id"] is True
    assert release["safety_policy"]["rule_fallback_retained"] is True
    assert release["evidence_policy"]["approval_does_not_create_missing_evidence"] is True

    locked_files = [release["service_config"], *release["artifacts"]]
    assert locked_files
    for item in locked_files:
        path = Path(item["path"])
        assert path.is_file(), path
        assert item["sha256"] == _sha256(path), path


def test_release_freezes_the_current_hybrid_runtime_roles() -> None:
    release = yaml.safe_load(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    roles = {item["role"] for item in release["artifacts"]}

    assert roles == {
        "pose_detection_and_tracking",
        "gait_primary",
        "sit_stand_primary",
        "fall_event_ensemble_seed_42",
        "fall_event_ensemble_seed_43",
        "fall_event_ensemble_seed_44",
    }
    assert release["runtime_branches"]["near_fall"] == "rule_primary"
    assert release["runtime_branches"]["personal_baseline"] == "statistical_fail_closed"
    assert release["runtime_branches"]["risk_fusion"] == "rule_primary"


def test_v2_release_is_frozen_and_hash_binds_runtime_implementation() -> None:
    release = yaml.safe_load(RELEASE_V2_MANIFEST.read_text(encoding="utf-8"))

    assert release["release_id"] == "fall-risk-competition-v2-20260827"
    assert release["status"] == "competition_release_frozen"
    assert release["based_on_release"] == "fall-risk-competition-v1-20260819"
    assert release["change_policy"]["requires_new_release_id"] is True
    assert release["safety_policy"]["rule_fallback_retained"] is True
    locked_files = [
        release["service_config"],
        *release["artifacts"],
        *release["runtime_implementation"],
        release["evidence"]["near_fall_selection_record"],
    ]
    for item in locked_files:
        path = Path(item["path"])
        assert path.is_file(), path
        assert item["sha256"] == _sha256(path), path


def test_v2_promotes_near_fall_but_preserves_continuous_sit_stand() -> None:
    release = yaml.safe_load(RELEASE_V2_MANIFEST.read_text(encoding="utf-8"))
    roles = {item["role"] for item in release["artifacts"]}

    assert roles == {
        "pose_detection_and_tracking",
        "gait_primary",
        "sit_stand_primary",
        "near_fall_tabular_seed_42",
        "fall_event_ensemble_seed_42",
        "fall_event_ensemble_seed_43",
        "fall_event_ensemble_seed_44",
    }
    assert release["near_fall_policy"]["score_threshold"] == 0.3815
    assert release["near_fall_policy"]["alert_cooldown_sec"] == 15.0
    assert release["near_fall_policy"]["inference_failure_behavior"] == "rule_fallback"
    assert release["sit_stand_policy"]["clip_presence_random_forest_promoted"] is False
    assert release["runtime_branches"]["sit_stand"].startswith("continuous_tcn_seed_42")
