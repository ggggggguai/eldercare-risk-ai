"""R7 full-fit expert package, checksum validation and explicit candidate API."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from elderly_monitoring.modules.mental_health.mood_social.feature_mapper import map_mood_social_features
from elderly_monitoring.modules.mental_health.mood_social.pipeline import _build_expert_frame
from elderly_monitoring.modules.mental_health.mood_social.r4.calibration import fit_calibration
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import CandidateSpec, fit_model, predict_model
from elderly_monitoring.modules.mental_health.mood_social.r7.contract import DEFAULT_DATA_RELATIVE, R7_EXISTING_EXPERT_SEEDS, repository_root, sha256_file
from elderly_monitoring.modules.mental_health.mood_social.r7.independent_experts import FEATURES, FIXED_SPECS, load_domain_frame
from elderly_monitoring.modules.mental_health.mood_social.r7.modeling import FittedExpertBundle, write_json
from elderly_monitoring.modules.mental_health.mood_social.r7.personal_change import physiology_support, runtime_personal_changes
from elderly_monitoring.modules.mental_health.mood_social.r7.phq_history_expert import HISTORY_FEATURES, HistoricalPHQ, build_history_features, build_psyche_history_frame
from elderly_monitoring.modules.mental_health.mood_social.r7.rule_fusion import DomainEvidence, HistoryEvidence, fuse_current_state, rule_contract_sha256
from elderly_monitoring.modules.mental_health.mood_social.schemas import MoodSocialInferRequest


R7_PACKAGE_RUN_ID = "MH-20260813-R7-001"
R7_MODEL_VERSION = "mood-social-current-state-v3.3.3-r7"
R7_PACKAGE_RELATIVE = Path(f"models/mental_health/mood_social/v3.3.3-r7/packages/{R7_PACKAGE_RUN_ID}")
R7_REPORT_RELATIVE = Path("reports/mental_health/mood_social/v3.3.3-r7/OPT-V333-R7-007-package")
MOOD_SOCIAL_R7_CANDIDATE_INFER_PATH = "/v1/mental-health/mood-social/r7-candidate/infer"


class R7PackageError(RuntimeError):
    pass


class R7CandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str
    request_id: str
    person_id: str
    target_date: str
    model_version: str
    package_run_id: str
    available: bool
    current_state_only: bool
    diagnosis: bool
    expert_assessments: dict[str, Any]
    passive_current_state: dict[str, Any]
    history_informed_current_state: dict[str, Any]
    rule_fusion: dict[str, Any]
    probability_used_sources: list[str]
    supporting_evidence_sources: list[str]
    fallback_model_version: str | None
    fallback_reason: str | None
    release_status: str
    limitations: list[str]


def _crossfit_calibrator(frame: pd.DataFrame, features: tuple[str, ...], spec: CandidateSpec, target: str) -> tuple[Any, Any, float]:
    probability = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold in range(5):
        train = frame.loc[frame["outer_fold"].ne(fold)].copy()
        validation = frame.loc[frame["outer_fold"].eq(fold)].copy()
        train["binary_target"] = train[target].astype(int)
        model = fit_model(train, features, spec, participant_equal=True)
        probability.loc[validation.index] = predict_model(model, validation, features)
    work = frame.copy()
    work["binary_target"] = work[target].astype(int)
    calibrator = fit_calibration(work, probability.to_numpy(float), "platt")
    model = fit_model(work, features, spec, participant_equal=True)
    threshold = float(np.quantile(calibrator.predict(probability), 0.5))
    return model, calibrator, threshold


def _fit_existing_bundle(root: Path, domain: str) -> FittedExpertBundle:
    frame, features = load_domain_frame(root, domain, R7_EXISTING_EXPERT_SEEDS[0])
    if domain == "activity" or domain == "profile":
        spec = CandidateSpec(f"{domain}_logistic", "elasticnet", {"C": 0.1, "l1_ratio": 0.2}, R7_EXISTING_EXPERT_SEEDS[0])
    else:
        family, params = FIXED_SPECS[domain]
        spec = CandidateSpec(f"{domain}_promoted", family, params, R7_EXISTING_EXPERT_SEEDS[0])
    models: dict[str, Any] = {}
    calibrators: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    for head in ("ge5", "ge10"):
        model, calibrator, threshold = _crossfit_calibrator(frame, features, spec, f"phq9_{head}_target")
        models[head] = model
        calibrators[head] = calibrator
        thresholds[head] = threshold
    return FittedExpertBundle(domain, features, models["ge5"], models["ge10"], calibrators["ge5"], calibrators["ge10"], thresholds["ge5"], thresholds["ge10"], "adaptive-reused-offline", ("no real-device parity",), f"{domain}-expert-r7")


def _fit_history_bundle(root: Path) -> FittedExpertBundle:
    from elderly_monitoring.modules.mental_health.mood_social.r4.data import load_r4_development_frame
    frame = build_psyche_history_frame(load_r4_development_frame(repository_root=root))
    split = pd.read_parquet(root / DEFAULT_DATA_RELATIVE / "protocol/existing_splits" / f"seed-{R7_EXISTING_EXPERT_SEEDS[0]}.parquet")
    frame = frame.merge(split[["r4_row_id", "outer_fold"]], on="r4_row_id", validate="one_to_one")
    spec = CandidateSpec("history_logistic", "elasticnet", {"C": 0.1, "l1_ratio": 0.2}, R7_EXISTING_EXPERT_SEEDS[0])
    models: dict[str, Any] = {}; calibrators: dict[str, Any] = {}; thresholds: dict[str, float] = {}
    for head in ("ge5", "ge10"):
        model, calibrator, threshold = _crossfit_calibrator(frame, HISTORY_FEATURES, spec, f"phq9_{head}_target")
        models[head] = model; calibrators[head] = calibrator; thresholds[head] = threshold
    return FittedExpertBundle("phq_history", HISTORY_FEATURES, models["ge5"], models["ge10"], calibrators["ge5"], calibrators["ge10"], thresholds["ge5"], thresholds["ge10"], "adaptive-reused-history-informed", ("not passive-only", "nominal-wave evaluation"), "phq-history-expert-r7")


def _checksums(package: Path) -> None:
    paths = sorted((path for path in package.iterdir() if path.is_file() and path.name != "SHA256SUMS"), key=lambda path: path.name)
    (package / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in paths), encoding="ascii", newline="\n")


def build_r7_package(*, repository_root_value: Path | None = None, overwrite: bool = False) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    package = root / R7_PACKAGE_RELATIVE
    report = root / R7_REPORT_RELATIVE
    if package.exists() and not overwrite:
        raise FileExistsError(package)
    package.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    bundles = {domain: _fit_existing_bundle(root, domain) for domain in ("activity", "sleep", "profile")}
    bundles["phq_history"] = _fit_history_bundle(root)
    joblib.dump(bundles, package / "r7_experts.joblib", compress=3)
    manifest = {
        "version": "mood-social-r7-modular-expert-package-v1", "run_id": R7_PACKAGE_RUN_ID,
        "model_version": R7_MODEL_VERSION, "status": "offline-validated/integration-ready/device-validation-pending",
        "promotion_authorized": False, "online_default_changed": False, "current_online_default": "MH-20260812-R5-001",
        "current_state_only": True, "forecast_changed": False,
        "experts": {name: {"model_version": bundle.model_version, "features": list(bundle.features), "evidence_grade": bundle.evidence_grade, "limitations": list(bundle.limitations)} for name, bundle in bundles.items()},
        "social": {"status": "anomaly-only", "phq_probability_enabled": False},
        "physiology": {"status": "support-only", "phq_probability_enabled": False},
        "rule_fusion": {"trained": False, "rule_contract_sha256": rule_contract_sha256(), "attention_index_is_phq_probability": False},
        "fallback": {"primary": "MH-20260812-R5-001", "legacy": "MH-20260802-013"},
        "license": {"DepreST-CAT": "CC BY-NC-SA academic; commercial/default release blocked pending review"},
    }
    write_json(package / "manifest.json", manifest, overwrite=True)
    write_json(package / "device_input_schema.json", MoodSocialInferRequest.model_json_schema(), overwrite=True)
    _checksums(package)
    result = {"status": manifest["status"], "package_run_id": R7_PACKAGE_RUN_ID, "model_sha256": sha256_file(package / "r7_experts.joblib"), "manifest_sha256": sha256_file(package / "manifest.json"), "checksums_sha256": sha256_file(package / "SHA256SUMS"), "online_default_changed": False}
    write_json(report / "package_build_report.json", result, overwrite=True)
    return result


def _verify(package: Path) -> None:
    listed: set[str] = set()
    for line in (package / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", 1); path = (package / name).resolve(); path.relative_to(package)
        if name in listed or not path.is_file() or sha256_file(path) != expected: raise R7PackageError(f"checksum failed: {name}")
        listed.add(name)
    expected = {path.name for path in package.iterdir() if path.is_file() and path.name != "SHA256SUMS"}
    if listed != expected: raise R7PackageError("checksum coverage incomplete")


@lru_cache(maxsize=4)
def _load_verified_package(package_text: str) -> tuple[dict[str, Any], dict[str, FittedExpertBundle]]:
    package = Path(package_text)
    _verify(package)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    bundles = joblib.load(package / "r7_experts.joblib")
    return manifest, bundles


def _expert_result(bundle: FittedExpertBundle, frame: pd.DataFrame, *, available: bool, valid_days: int, personal: Mapping[str, object] | None = None) -> tuple[dict[str, Any], DomainEvidence | None]:
    if not available:
        return {"domain": bundle.domain, "available": False, "fallback_reason": "insufficient_domain_evidence", "model_version": bundle.model_version}, None
    p5, p10 = bundle.predict_frame(frame); probability5 = float(p5[0]); probability10 = float(p10[0])
    level = 2 if probability10 >= bundle.threshold_ge10 else 1 if probability5 >= bundle.threshold_ge5 else 0
    personal_level = None if not personal or not personal.get("available") else int(personal.get("level", 0))
    evidence = DomainEvidence(bundle.domain, True, level, min(valid_days / 7.0, 1.0), bool(personal and personal.get("persistent")), personal_level)  # type: ignore[arg-type]
    return {"domain": bundle.domain, "available": True, "evidence_grade": bundle.evidence_grade, "probability_phq_ge5": probability5, "probability_phq_ge10": probability10, "domain_level": evidence.effective_level(), "confidence": evidence.confidence, "valid_days": valid_days, "baseline_status": "stable" if personal and personal.get("available") else "cold_start", "personal_change": dict(personal or {}), "model_version": bundle.model_version, "limitations": list(bundle.limitations)}, evidence


def infer_mood_social_r7_candidate(request: MoodSocialInferRequest | Mapping[str, Any], *, package_directory: Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[6]; package = (package_directory or root / R7_PACKAGE_RELATIVE).resolve()
    manifest, bundles = _load_verified_package(str(package))
    parsed = request if isinstance(request, MoodSocialInferRequest) else MoodSocialInferRequest.model_validate(request)
    mapped = map_mood_social_features(parsed); frame = _build_expert_frame(mapped); changes = runtime_personal_changes(parsed)
    activity_result, activity_vote = _expert_result(bundles["activity"], frame, available=int(mapped.activity.value("valid_days") or 0) >= 7, valid_days=int(mapped.activity.value("valid_days") or 0), personal=changes["activity"])
    sleep_result, sleep_vote = _expert_result(bundles["sleep"], frame, available=int(mapped.sleep.value("valid_nights") or 0) >= 7, valid_days=int(mapped.sleep.value("valid_nights") or 0), personal=changes["sleep"])
    profile_available = any(mapped.social_context.feature_mask)
    profile_result, _ = _expert_result(bundles["profile"], frame, available=profile_available, valid_days=0)
    if profile_result.get("available"):
        profile_result["background_vulnerability"] = profile_result.pop("domain_level")
        profile_result.pop("probability_phq_ge5", None)
        profile_result.pop("probability_phq_ge10", None)
        profile_result["phq_probability_available"] = False
        profile_result["limitations"] = [
            *profile_result.get("limitations", []),
            "background-only; never counted as a dynamic vote",
        ]
    history_records = [
        HistoricalPHQ(item.date, item.score, item.known_at)
        for item in parsed.historical_phq9_assessments
        if item.known_at is not None
    ]
    history_features, history_audit = build_history_features(history_records, target_date=parsed.target_date)
    history_audit["excluded_missing_known_at"] = sum(
        item.known_at is None for item in parsed.historical_phq9_assessments
    )
    history_result: dict[str, Any]; history_vote: HistoryEvidence | None = None
    if history_features is None:
        history_result = {"domain": "phq_history", **history_audit, "model_version": "phq-history-expert-r7"}
    else:
        history_frame = pd.DataFrame([history_features]); p5, p10 = bundles["phq_history"].predict_frame(history_frame); level = 2 if p10[0] >= bundles["phq_history"].threshold_ge10 else 1 if p5[0] >= bundles["phq_history"].threshold_ge5 else 0
        history_vote = HistoryEvidence(True, level, int(history_audit["last_age_days"]), high=level >= 2, low=level == 0)
        history_result = {"domain": "phq_history", **history_audit, "probability_phq_ge5": float(p5[0]), "probability_phq_ge10": float(p10[0]), "domain_level": level, "model_version": "phq-history-expert-r7", "limitations": ["history-informed, not passive-only"]}
    social_change = changes["social"]
    social_vote = DomainEvidence("social", True, int(social_change.get("level", 0)), 0.5, bool(social_change.get("persistent")), int(social_change.get("level", 0))) if social_change.get("available") else None
    votes = [value for value in (activity_vote, sleep_vote, social_vote) if value is not None]
    physiology = physiology_support(parsed)
    fused = fuse_current_state(votes, history=history_vote, profile_background={"available": profile_available, "vulnerability": profile_result.get("background_vulnerability")}, physiology_support=physiology)
    available = bool(fused["passive_current_state"]["available"] or fused["history_informed_current_state"]["available"])
    return {"schema_version": "mood_social_r7_candidate_response_v1", "request_id": parsed.request_id, "person_id": parsed.person_id, "target_date": parsed.target_date.isoformat(), "model_version": R7_MODEL_VERSION, "package_run_id": R7_PACKAGE_RUN_ID, "available": available, "current_state_only": True, "diagnosis": False, "expert_assessments": {"activity": activity_result, "sleep": sleep_result, "social": {"domain": "social", "available": bool(social_vote), "phq_probability_available": False, "personal_change": social_change, "limitations": ["DepreST social probability gate failed; anomaly-only"]}, "physiology": physiology, "profile": profile_result, "phq_history": history_result}, **fused, "probability_used_sources": [name for name, result in (("camera", activity_result), ("sleep_device", sleep_result), ("profile", profile_result), ("historical_phq9", history_result)) if result.get("available") and name != "profile"], "supporting_evidence_sources": [name for name, enabled in (("profile", profile_available), ("physiology", physiology.get("available", False)), ("s10", bool(social_vote))) if enabled], "fallback_model_version": "mood-social-current-state-v3.3.3-r5" if not available else None, "fallback_reason": "no_valid_r7_dynamic_or_history_evidence" if not available else None, "release_status": manifest["status"], "limitations": ["offline/reused evidence; no real-device validation", "rule attention index is not a PHQ probability"]}


__all__ = ["MOOD_SOCIAL_R7_CANDIDATE_INFER_PATH", "R7CandidateResponse", "R7PackageError", "R7_PACKAGE_RUN_ID", "build_r7_package", "infer_mood_social_r7_candidate"]
