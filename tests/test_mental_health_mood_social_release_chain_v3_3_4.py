"""ART-002 and API-003 no-promotion release-chain tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from elderly_monitoring.modules.mental_health.mood_social.model_package import (
    load_mood_social_model_package,
)
from elderly_monitoring.modules.mental_health.mood_social.package_selection import (
    load_active_package_selection,
)


ROOT = Path(__file__).resolve().parents[1]
ART_DECISION = (
    ROOT
    / "models/mental_health/mood_social/v3.3.3/candidates/MH-20260804-023/art002_decision_manifest.json"
)
API_DECISION = (
    ROOT
    / "models/mental_health/mood_social/v3.3.3/candidates/MH-20260804-024/api003_decision_manifest.json"
)


def test_art002_failed_promotion_preserves_art001_without_fake_package() -> None:
    decision = _json(ART_DECISION)

    assert decision["task_id"] == "ART-002"
    assert decision["run_id"] == "MH-20260804-023"
    assert decision["promotion_passed"] is False
    assert decision["new_online_package_created"] is False
    assert decision["active_package_run_id"] == "MH-20260802-013"
    assert not (
        ROOT / "models/mental_health/mood_social/v3.3.4/packages/MH-20260804-023"
    ).exists()


def test_art002_hash_chain_binds_candidate_workpoint_and_fallback() -> None:
    decision = _json(ART_DECISION)
    upstream = decision["upstream"]
    paths = {
        "candidate_model_sha256": ROOT
        / "models/mental_health/mood_social/v3.3.3/candidates/MH-20260804-021/mood_fusion_v3_3_4_candidate.joblib",
        "candidate_manifest_sha256": ROOT
        / "models/mental_health/mood_social/v3.3.3/candidates/MH-20260804-021/mood_fusion_v3_3_4_candidate_manifest.json",
        "workpoint_manifest_sha256": ROOT
        / "models/mental_health/mood_social/v3.3.3/candidates/MH-20260804-022/competition_workpoint_manifest.json",
        "promotion_sha256": ROOT
        / "reports/mental_health/mood_social/MH-20260804-021/promotion.json",
    }
    for key, path in paths.items():
        assert _sha256(path) == upstream[key]

    fallback = ROOT / decision["active_package_directory"]
    assert _sha256(fallback / "manifest.json") == upstream["fallback_manifest_sha256"]
    assert _sha256(fallback / "SHA256SUMS") == upstream["fallback_checksums_sha256"]
    package = load_mood_social_model_package(fallback)
    assert len(package.online_models) == 7
    assert not package.offline_models
    assert package.manifest["model006_online"] is False


def test_api003_loader_and_backend_contract_keep_v333_active() -> None:
    decision = _json(API_DECISION)
    selection = load_active_package_selection(repository_root=ROOT)

    assert decision["task_id"] == "API-003"
    assert decision["run_id"] == "MH-20260804-024"
    assert decision["promotion_passed"] is False
    assert decision["new_package_connected"] is False
    assert decision["no_evidence_http_status"] == 200
    assert decision["unavailable_package_http_status"] == 503
    assert decision["backend_algorithm_implementation"] is False
    assert selection.decision_run_id == "MH-20260804-023"
    assert selection.package_run_id == "MH-20260802-013"
    assert selection.model_version == "mood-fusion-v3.3.3"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
