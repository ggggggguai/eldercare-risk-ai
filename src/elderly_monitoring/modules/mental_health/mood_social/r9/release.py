"""R9 promoted-node package and explicit candidate inference."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.feature_mapper import map_mood_social_features
from elderly_monitoring.modules.mental_health.mood_social.pipeline import _build_expert_frame
from elderly_monitoring.modules.mental_health.mood_social.r4.calibration import fit_calibration
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import CandidateSpec, fit_model, predict_model
from elderly_monitoring.modules.mental_health.mood_social.r7.modeling import FittedExpertBundle, select_specificity_threshold
from elderly_monitoring.modules.mental_health.mood_social.r7.phq_history_expert import HistoricalPHQ, build_history_features
from elderly_monitoring.modules.mental_health.mood_social.r7.rule_fusion import DomainEvidence
from elderly_monitoring.modules.mental_health.mood_social.r8.release import (
    R8_PACKAGE_RELATIVE,
    R8_PACKAGE_RUN_ID,
    infer_mood_social_r8_candidate,
)

from .contract import DEFAULT_DATA_RELATIVE, DEFAULT_REPORT_RELATIVE, R9_PROTOCOL_VERSION, repository_root, sha256_file, write_json
from .evidence_graph import graph_contract_sha256
from .evaluation import EVALUATION_REPORT_RELATIVE
from .modeling import HEAD_TARGETS, training_weights
from .rule_fusion import fuse_r9_current_state, rule_contract_sha256
from .schemas import R9CandidateRequest, R9CandidateResponse, R9_RESPONSE_SCHEMA
from .selection import FEATURES, SELECTION_REPORT_RELATIVE, WEIGHT_MODE, load_selection_frames


R9_PACKAGE_RUN_ID = "MH-20260814-R9-001"
R9_MODEL_VERSION = "mood-social-current-state-v3.3.3-r9"
R9_PACKAGE_RELATIVE = Path(f"models/mental_health/mood_social/v3.3.3-r9/packages/{R9_PACKAGE_RUN_ID}")
R9_REPORT_RELATIVE = DEFAULT_REPORT_RELATIVE / "package"
MOOD_SOCIAL_R9_CANDIDATE_INFER_PATH = "/v1/mental-health/mood-social/r9-candidate/infer"


class R9PackageError(RuntimeError):
    pass


def _modal_spec(lock: dict[str, Any], node: str, head: str) -> CandidateSpec:
    rows = [lock["nodes"][node]["outer_fold_selection"][str(fold)][head]["candidate"] for fold in range(5)]
    candidate_id = Counter(row["candidate_id"] for row in rows).most_common(1)[0][0]
    payload = next(row for row in rows if row["candidate_id"] == candidate_id)
    return CandidateSpec(payload["candidate_id"], payload["family"], payload["params"], payload["seed"])


def _fit_bundle(frame: pd.DataFrame, node: str, lock: dict[str, Any]) -> FittedExpertBundle:
    models: dict[str, Any] = {}
    calibrators: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    for head, target in HEAD_TARGETS.items():
        spec = _modal_spec(lock, node, head)
        oof = pd.Series(np.nan, index=frame.index, dtype=float)
        for fold in range(5):
            train = frame.loc[frame["outer_fold"].ne(fold)].copy()
            valid = frame.loc[frame["outer_fold"].eq(fold)].copy()
            train["binary_target"] = train[target].astype(int)
            model = fit_model(train, FEATURES[node], spec, participant_equal=False, sample_weight=training_weights(train, WEIGHT_MODE[node]))
            oof.loc[valid.index] = predict_model(model, valid, FEATURES[node])
        if oof.isna().any():
            raise R9PackageError(f"full-fit crossfit OOF incomplete for {node}/{head}")
        work = frame.copy()
        work["binary_target"] = work[target].astype(int)
        calibrator = fit_calibration(work, oof.to_numpy(float), "platt")
        models[head] = fit_model(work, FEATURES[node], spec, participant_equal=False, sample_weight=training_weights(work, WEIGHT_MODE[node]))
        calibrators[head] = calibrator
        thresholds[head] = select_specificity_threshold(work[target].astype(int), calibrator.predict(oof.to_numpy(float)), 0.80)
    return FittedExpertBundle(
        domain=node,
        features=FEATURES[node],
        model_ge5=models["ge5"],
        model_ge10=models["ge10"],
        calibration_ge5=calibrators["ge5"],
        calibration_ge10=calibrators["ge10"],
        threshold_ge5=float(thresholds["ge5"]),
        threshold_ge10=float(thresholds["ge10"]),
        evidence_grade="adaptive-reused-locked-outer",
        limitations=("no new blind cohort", "no real-device parity", "current-state screening not diagnosis"),
        model_version=f"{node}-expert-r9",
    )


def _checksums(package: Path) -> None:
    files = sorted((path for path in package.iterdir() if path.is_file() and path.name != "SHA256SUMS"), key=lambda path: path.name)
    (package / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="ascii", newline="\n")


def build_r9_package(*, repository_root_value: Path | None = None, overwrite: bool = False) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    protocol_path = root / DEFAULT_DATA_RELATIVE / "protocol/r9_protocol_manifest.json"
    evaluation_path = root / EVALUATION_REPORT_RELATIVE / "evaluation_report.json"
    lock_path = root / SELECTION_REPORT_RELATIVE / "selection_lock.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    required = {"phq_history", "activity_sleep", "sleep_history"}
    if protocol.get("status") != "evaluation-complete" or not required.issubset(evaluation.get("promoted_nodes", [])):
        raise R9PackageError("R9 evaluation has not authorized the promoted-node package")
    package = root / R9_PACKAGE_RELATIVE
    if package.exists() and not overwrite:
        raise FileExistsError(package)
    package.mkdir(parents=True, exist_ok=True)
    frames = load_selection_frames(root)
    bundles = {node: _fit_bundle(frames[node], node, lock) for node in sorted(required)}
    joblib.dump(bundles, package / "r9_promoted_experts.joblib", compress=3)
    manifest = {
        "version": "mood-social-r9-promoted-node-package-v1",
        "protocol_version": R9_PROTOCOL_VERSION,
        "run_id": R9_PACKAGE_RUN_ID,
        "model_version": R9_MODEL_VERSION,
        "status": "offline-validated/integration-ready/device-validation-pending",
        "current_state_only": True,
        "diagnosis": False,
        "promotion_authorized": False,
        "online_default_changed": False,
        "current_online_default": "MH-20260812-R5-001",
        "r8_core": {"run_id": R8_PACKAGE_RUN_ID, "path": R8_PACKAGE_RELATIVE.as_posix(), "fallback_role": "independent experts and facial support"},
        "promoted_nodes": sorted(required),
        "fallback_nodes": {"activity": "R7 via R8", "sleep": "R7 via R8", "social": "anomaly-only", "profile": "background-only", "physiology": "support-only"},
        "experts": {name: {"features": list(bundle.features), "model_version": bundle.model_version, "evidence_grade": bundle.evidence_grade, "limitations": list(bundle.limitations)} for name, bundle in bundles.items()},
        "evidence_graph": {"trained": False, "contract_sha256": graph_contract_sha256()},
        "rule_fusion": {"trained": False, "contract_sha256": rule_contract_sha256(), "attention_index_is_phq_probability": False},
        "fallback_chain": [R8_PACKAGE_RUN_ID, "MH-20260812-R5-001", "MH-20260802-013"],
        "release_limits": ["no complete seven-domain same-person cohort", "no complete R9 AUPRC", "no real-device observations", "no paired PHQ-video training data"],
        "protocol_manifest_sha256": sha256_file(protocol_path),
        "selection_lock_sha256": sha256_file(lock_path),
        "evaluation_report_sha256": sha256_file(evaluation_path),
    }
    write_json(package / "manifest.json", manifest, overwrite=True)
    write_json(package / "request_schema.json", R9CandidateRequest.model_json_schema(), overwrite=True)
    write_json(package / "response_schema.json", R9CandidateResponse.model_json_schema(), overwrite=True)
    _checksums(package)
    result = {"status": manifest["status"], "package_run_id": R9_PACKAGE_RUN_ID, "model_sha256": sha256_file(package / "r9_promoted_experts.joblib"), "manifest_sha256": sha256_file(package / "manifest.json"), "checksums_sha256": sha256_file(package / "SHA256SUMS"), "promoted_nodes": sorted(required), "online_default_changed": False}
    write_json(root / R9_REPORT_RELATIVE / "package_build_report.json", result, overwrite=True)
    return result


def _verify(package: Path) -> dict[str, Any]:
    try:
        lines = (package / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise R9PackageError("R9 package checksums unavailable") from exc
    listed: set[str] = set()
    for line in lines:
        try:
            expected, name = line.split("  ", 1)
            path = (package / name).resolve()
            path.relative_to(package.resolve())
        except (ValueError, OSError) as exc:
            raise R9PackageError("invalid R9 checksum entry") from exc
        if name in listed or not path.is_file() or sha256_file(path) != expected:
            raise R9PackageError(f"R9 checksum failed: {name}")
        listed.add(name)
    expected_files = {path.name for path in package.iterdir() if path.is_file() and path.name != "SHA256SUMS"}
    if listed != expected_files:
        raise R9PackageError("R9 checksum coverage incomplete")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("run_id") != R9_PACKAGE_RUN_ID or manifest.get("evidence_graph", {}).get("contract_sha256") != graph_contract_sha256() or manifest.get("rule_fusion", {}).get("contract_sha256") != rule_contract_sha256():
        raise R9PackageError("R9 package manifest contract drift")
    return manifest


@lru_cache(maxsize=4)
def _load_verified_package(package_text: str) -> tuple[dict[str, Any], dict[str, FittedExpertBundle]]:
    package = Path(package_text)
    manifest = _verify(package)
    return manifest, joblib.load(package / "r9_promoted_experts.joblib")


def _freshness_features(features: dict[str, float]) -> dict[str, float]:
    result = dict(features)
    age = float(result["history.age_days"])
    weight = 0.0 if age < 14 or age > 100 else 1.0 if age <= 30 else float(np.exp(-(age - 30.0) / 70.0))
    result["history.freshness_weight"] = weight
    result["history.decayed_last_score"] = float(result["history.last_score"]) * weight
    return result


def _assessment(bundle: FittedExpertBundle, frame: pd.DataFrame) -> dict[str, Any]:
    p5, p10 = bundle.predict_frame(frame)
    early, elevated = float(p5[0]), float(p10[0])
    level = 2 if elevated >= bundle.threshold_ge10 else 1 if early >= bundle.threshold_ge5 else 0
    return {"domain": bundle.domain, "role": "history_probability" if bundle.domain == "phq_history" else "local_joint_probability", "available": True, "probability_phq_ge5": early, "probability_phq_ge10": elevated, "domain_level": level, "evidence_grade": bundle.evidence_grade, "model_version": bundle.model_version, "limitations": list(bundle.limitations)}


def _domain_vote(name: str, result: Mapping[str, Any]) -> DomainEvidence | None:
    if not result.get("available"):
        return None
    level = result.get("domain_level")
    if level is None and name == "social":
        level = result.get("personal_change", {}).get("level")
    if level is None:
        return None
    personal = result.get("personal_change", {})
    return DomainEvidence(name, True, int(level), float(result.get("confidence", 0.5)), bool(personal.get("persistent")), int(personal.get("level", 0)) if personal.get("available") else None)  # type: ignore[arg-type]


def infer_mood_social_r9_candidate(
    request: R9CandidateRequest | Mapping[str, Any],
    *,
    package_directory: Path | None = None,
    r8_package_directory: Path | None = None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[6]
    package = (package_directory or root / R9_PACKAGE_RELATIVE).resolve()
    manifest, bundles = _load_verified_package(str(package))
    parsed = request if isinstance(request, R9CandidateRequest) else R9CandidateRequest.model_validate(request)
    core = infer_mood_social_r8_candidate(parsed.r8_payload(), package_directory=r8_package_directory or root / R8_PACKAGE_RELATIVE)
    mapped = map_mood_social_features(parsed)
    frame = _build_expert_frame(mapped)
    experts = dict(core["expert_assessments"])
    for domain, role in {
        "activity": "dynamic_probability",
        "sleep": "dynamic_probability",
        "social": "anomaly_only",
        "profile": "background_only",
        "physiology": "support_only",
        "facial_affect": "facial_support",
    }.items():
        if isinstance(experts.get(domain), dict):
            experts[domain] = {"domain": domain, **experts[domain], "role": role}
    joint: dict[str, Any] = {}
    fallbacks: dict[str, Any] = {"activity": "R7 via R8", "sleep": "R7 via R8"}
    activity_available = bool(experts.get("activity", {}).get("available"))
    sleep_available = bool(experts.get("sleep", {}).get("available"))
    if activity_available and sleep_available:
        joint["activity_sleep"] = _assessment(bundles["activity_sleep"], frame)
    else:
        joint["activity_sleep"] = {"domain": "activity_sleep", "role": "local_joint_probability", "available": False, "fallback_reason": "requires_activity_and_sleep", "fallback_nodes": ["activity", "sleep"]}
    records = [HistoricalPHQ(item.date, item.score, item.known_at) for item in parsed.historical_phq9_assessments if item.known_at is not None]
    history_features, history_audit = build_history_features(records, target_date=parsed.target_date)
    history_eligible = bool(history_features is not None and 14 <= float(history_features["history.age_days"]) <= 100)
    if history_eligible:
        history_features = _freshness_features(history_features)
        history_frame = pd.DataFrame([history_features])
        history_result = {**_assessment(bundles["phq_history"], history_frame), **history_audit, "freshness_interval": "14_30_strong" if history_audit["last_age_days"] <= 30 else "31_100_decayed"}
        experts["phq_history"] = history_result
        if sleep_available:
            combined = frame.copy()
            for key, value in history_features.items():
                combined[key] = value
            joint["sleep_history"] = _assessment(bundles["sleep_history"], combined)
        else:
            joint["sleep_history"] = {"domain": "sleep_history", "role": "local_joint_probability", "available": False, "fallback_reason": "requires_sleep_and_14_100_day_history", "fallback_nodes": ["sleep", "phq_history"]}
    else:
        age = None if history_features is None else float(history_features["history.age_days"])
        interval = "unavailable" if age is None else "0_13_sensitivity_only" if age < 14 else "101_180_background" if age <= 180 else "over_180_abstain"
        history_result = {"domain": "phq_history", "role": "history_probability", "available": False, **history_audit, "freshness_interval": interval, "fallback_reason": "outside_r9_14_100_primary_runtime_window"}
        experts["phq_history"] = history_result
        joint["sleep_history"] = {"domain": "sleep_history", "role": "local_joint_probability", "available": False, "fallback_reason": "requires_sleep_and_14_100_day_history", "fallback_nodes": ["sleep", "phq_history"]}
    votes = [vote for vote in (_domain_vote("activity", experts.get("activity", {})), _domain_vote("sleep", experts.get("sleep", {})), _domain_vote("social", experts.get("social", {}))) if vote is not None]
    history_vote = None
    if history_result.get("available"):
        history_vote = {"available": True, "level": history_result["domain_level"], "high": history_result["domain_level"] >= 2, "low": history_result["domain_level"] == 0, "last_age_days": history_audit["last_age_days"]}
    fused = fuse_r9_current_state(
        votes,
        history=history_vote,
        facial=experts.get("facial_affect", {}),
        profile_background=experts.get("profile", {}),
        physiology_support=experts.get("physiology", {}),
        local_joint_assessments=joint,
    )
    # The rule helper keeps a diagnostic copy; the public contract exposes the
    # same content only once under ``joint_expert_assessments``.
    fused.pop("local_joint_assessments", None)
    route = fused["evidence_graph"]
    probability_nodes = list(route["probability_nodes"])
    available = bool(fused["operational_attention_state"].get("available"))
    observed = list(dict.fromkeys(core.get("observed_sources", [])))
    fallbacks.update({
        "activity_sleep": None if joint["activity_sleep"].get("available") else "activity+sleep independent R8 core",
        "sleep_history": None if joint["sleep_history"].get("available") else "sleep/history independent R8 core",
        "phq_history": None if history_result.get("available") else history_result.get("fallback_reason"),
        "production": "MH-20260812-R5-001",
        "legacy": "MH-20260802-013",
    })
    return {
        "schema_version": R9_RESPONSE_SCHEMA,
        "request_id": parsed.request_id,
        "person_id": parsed.person_id,
        "target_date": parsed.target_date.isoformat(),
        "model_version": R9_MODEL_VERSION,
        "package_run_id": R9_PACKAGE_RUN_ID,
        "core_package_run_id": R8_PACKAGE_RUN_ID,
        "available": available,
        "current_state_only": True,
        "diagnosis": False,
        "expert_assessments": experts,
        "joint_expert_assessments": joint,
        **fused,
        "fusion_mode": route["fusion_mode"],
        "evidence_signature": route["signature"],
        "probability_used_sources": probability_nodes,
        "supporting_evidence_sources": list(core.get("supporting_evidence_sources", [])),
        "observed_sources": observed,
        "quality": {"activity_available": activity_available, "sleep_available": sleep_available, "facial_available": bool(experts.get("facial_affect", {}).get("available")), "no_cross_modal_imputation": True},
        "freshness": {"phq_history_interval": history_result.get("freshness_interval"), "last_phq_age_days": history_audit.get("last_age_days"), "vote_window_days": [14, 100]},
        "lineage": {"r9_package_run_id": R9_PACKAGE_RUN_ID, "r8_core_package_run_id": R8_PACKAGE_RUN_ID, "graph_contract_sha256": graph_contract_sha256(), "rule_contract_sha256": rule_contract_sha256(), "evaluation_report_sha256": manifest["evaluation_report_sha256"]},
        "abstain_reason": None if available else "insufficient_dynamic_and_eligible_history_evidence",
        "fallback_chain": fallbacks,
        "fallback_model_version": None if available else "mood-social-current-state-v3.3.3-r8",
        "fallback_reason": None if available else "r9_no_eligible_evidence",
        "release_status": manifest["status"],
        "limitations": ["adaptive/reused locked-procedure evidence; not a new blind test", "no complete seven-domain R9 AUPRC", "no real-device validation", "facial-affect classes are not PHQ probabilities", "rule-derived attention is not diagnosis or a calibrated PHQ probability"],
    }


__all__ = ["MOOD_SOCIAL_R9_CANDIDATE_INFER_PATH", "R9CandidateRequest", "R9CandidateResponse", "R9PackageError", "R9_MODEL_VERSION", "R9_PACKAGE_RELATIVE", "R9_PACKAGE_RUN_ID", "build_r9_package", "infer_mood_social_r9_candidate"]
