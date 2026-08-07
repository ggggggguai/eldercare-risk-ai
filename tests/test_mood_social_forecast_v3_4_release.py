from __future__ import annotations

import hashlib
import json
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ALGORITHM_ROOT.parents[1]
RUN_ROOT = (
    ALGORITHM_ROOT
    / "artifacts"
    / "mental_health"
    / "mood_social"
    / "forecast_v3.4"
    / "optimization_candidates"
    / "MH-20260807-FOPT-002"
)
RELEASE_ROOT = RUN_ROOT / "release"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_manifest_preserves_the_no_promotion_boundary() -> None:
    manifest = _json(RELEASE_ROOT / "release_manifest.json")
    assert manifest["task_id"] == "FORECAST-OPT-001G"
    assert manifest["status"] == "release_completed"
    assert manifest["promotion_status"] == "retain_v3_4_baseline"
    assert manifest["recommended_fallback"] == "MH-20260805-FCAST-001"
    assert manifest["product_integration_performed"] is False
    assert manifest["product_visible"] is False
    assert manifest["execution_mode"] == "offline_only"


def test_release_cards_are_audit_only_and_cover_both_horizons() -> None:
    cards = []
    for task_id in ("forecast_1m", "forecast_2m"):
        card = _json(RELEASE_ROOT / f"{task_id}_001d_candidate_model_card.json")
        assert card["task_id"] == task_id
        assert card["candidate_status"] == "audit_only_not_promoted"
        assert card["recommended_fallback"] == "MH-20260805-FCAST-001"
        assert card["product_visible"] is False
        assert set(card["baseline"]) >= {"auprc", "auroc", "brier", "ece"}
        assert set(card["candidate"]) >= {"auprc", "auroc", "brier", "ece"}
        cards.append(task_id)
    assert cards == ["forecast_1m", "forecast_2m"]


def test_release_isolation_hashes_bind_v33_and_backend() -> None:
    isolation = _json(RELEASE_ROOT / "isolation_report.json")
    assert isolation["forecast_backend_hits"] == []
    assert isolation["online_model_version"] == "mood-fusion-v3.3.3"
    assert isolation["online_package_run_id"] == "MH-20260802-013"
    assert isolation["product_integration_performed"] is False
    acceptance = (
        ALGORITHM_ROOT
        / "models"
        / "mental_health"
        / "mood_social"
        / "v3.3.3"
        / "acceptance.json"
    )
    assert _sha256(acceptance) == isolation["v3_3"]["acceptance_sha256"]
    for relative, digest in isolation["backend_hashes"].items():
        assert _sha256(WORKSPACE_ROOT / "backend" / relative) == digest


def test_release_checksums_cover_validation_report() -> None:
    listed = {}
    for line in (RELEASE_ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        listed[relative] = digest
    assert "validation_report.json" in listed
    actual = {
        path.relative_to(RELEASE_ROOT).as_posix()
        for path in RELEASE_ROOT.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert set(listed) == actual
    assert all(
        _sha256(RELEASE_ROOT / relative) == digest
        for relative, digest in listed.items()
    )
