from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


RELEASE_MANIFEST = Path("configs/modules/fall_risk_release_v1.yaml")


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
