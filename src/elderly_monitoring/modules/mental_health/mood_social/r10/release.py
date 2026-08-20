"""R10 candidate package and explicit current-state inference."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.feature_mapper import map_mood_social_features
from elderly_monitoring.modules.mental_health.mood_social.pipeline import _build_expert_frame
from elderly_monitoring.modules.mental_health.mood_social.r7.modeling import (
    project_heads,
    select_specificity_threshold,
)
from elderly_monitoring.modules.mental_health.mood_social.r7.phq_history_expert import (
    HistoricalPHQ,
    build_history_features,
)
from elderly_monitoring.modules.mental_health.mood_social.r7.rule_fusion import DomainEvidence
from elderly_monitoring.modules.mental_health.mood_social.r9.release import (
    R9_PACKAGE_RELATIVE,
    R9_PACKAGE_RUN_ID,
    infer_mood_social_r9_candidate,
)

from .baseline import load_r10_frames
from .contract import (
    DEFAULT_DATA_RELATIVE,
    DEFAULT_REPORT_RELATIVE,
    R10_PROTOCOL_VERSION,
    repository_root,
    sha256_file,
    write_json,
)
from .evaluation import EVALUATION_REPORT_RELATIVE
from .evidence_graph import graph_contract_sha256
from .features import add_activity_sleep_interactions
from .modeling import (
    HEAD_TARGETS,
    FittedCandidate,
    apply_calibration,
    candidate_inner_predictions,
    fit_final_calibration,
    fit_selected_candidate,
)
from .rule_fusion import fuse_r10_current_state, rule_contract_sha256
from .schemas import R10CandidateRequest, R10CandidateResponse, R10_RESPONSE_SCHEMA
from .selection import SELECTION_REPORT_RELATIVE


R10_PACKAGE_RUN_ID = "MH-20260814-R10-001"
R10_MODEL_VERSION = "mood-social-current-state-v3.3.3-r10"
R10_PACKAGE_RELATIVE = Path(
    f"models/mental_health/mood_social/v3.3.3-r10/packages/{R10_PACKAGE_RUN_ID}"
)
R10_REPORT_RELATIVE = DEFAULT_REPORT_RELATIVE / "package"
MOOD_SOCIAL_R10_CANDIDATE_INFER_PATH = "/v1/mental-health/mood-social/r10-candidate/infer"


class R10PackageError(RuntimeError):
    pass


@dataclass
class R10JointBundle:
    node: str
    candidate_ge5: FittedCandidate
    candidate_ge10: FittedCandidate
    calibration_ge5: Any
    calibration_ge10: Any
    calibration_method_ge5: str
    calibration_method_ge10: str
    threshold_ge5: float
    threshold_ge10: float

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p5 = apply_calibration(
            self.calibration_ge5,
            self.candidate_ge5.predict_raw(frame),
            self.calibration_method_ge5,
        )
        p10 = apply_calibration(
            self.calibration_ge10,
            self.candidate_ge10.predict_raw(frame),
            self.calibration_method_ge10,
        )
        return project_heads(p5, p10)


def _modal_selection(lock: dict[str, Any], node: str, head: str) -> dict[str, Any]:
    rows = [
        lock["nodes"][node][str(repeat)][str(fold)][head]
        for repeat in range(3)
        for fold in range(5)
    ]
    candidate = Counter(row["candidate_id"] for row in rows).most_common(1)[0][0]
    selected_rows = [row for row in rows if row["candidate_id"] == candidate]
    calibration = Counter(row["calibration"] for row in selected_rows).most_common(1)[0][0]
    metadata: dict[str, Any] = {}
    if candidate.startswith("residual_"):
        values = [float(row["metadata"]["lambda"]) for row in selected_rows]
        metadata["lambda"] = Counter(values).most_common(1)[0][0]
    return {"candidate_id": candidate, "calibration": calibration, "metadata": metadata}


def _fit_bundle(frame: pd.DataFrame, node: str, lock: dict[str, Any]) -> R10JointBundle:
    work = frame.loc[frame["repeat"].eq(0)].copy().reset_index(drop=True)
    folds = work["outer_fold"].astype(int)
    fitted: dict[str, FittedCandidate] = {}
    calibrators: dict[str, Any] = {}
    methods: dict[str, str] = {}
    thresholds: dict[str, float] = {}
    for head, target in HEAD_TARGETS.items():
        selection = _modal_selection(lock, node, head)
        probabilities, metadata = candidate_inner_predictions(
            node, work, folds, head, 20260910 + (0 if head == "ge5" else 100)
        )
        candidate = selection["candidate_id"]
        if candidate not in probabilities:
            raise R10PackageError(f"modal candidate unavailable in full fit: {node}/{head}")
        fit_selection = dict(selection)
        fit_selection["metadata"] = dict(metadata[candidate])
        if candidate.startswith("residual_"):
            fit_selection["metadata"]["lambda"] = selection["metadata"]["lambda"]
        fitted[head] = fit_selected_candidate(
            node, work, folds, head, 20260920, fit_selection, metadata
        )
        methods[head] = selection["calibration"]
        calibrators[head] = fit_final_calibration(
            work, probabilities[candidate], target, methods[head]
        )
        calibrated = apply_calibration(
            calibrators[head], probabilities[candidate], methods[head]
        )
        thresholds[head] = select_specificity_threshold(work[target], calibrated, 0.80)
    return R10JointBundle(
        node=node,
        candidate_ge5=fitted["ge5"],
        candidate_ge10=fitted["ge10"],
        calibration_ge5=calibrators["ge5"],
        calibration_ge10=calibrators["ge10"],
        calibration_method_ge5=methods["ge5"],
        calibration_method_ge10=methods["ge10"],
        threshold_ge5=float(thresholds["ge5"]),
        threshold_ge10=float(thresholds["ge10"]),
    )


def _write_checksums(package: Path) -> None:
    files = sorted(
        (path for path in package.iterdir() if path.is_file() and path.name != "SHA256SUMS"),
        key=lambda path: path.name,
    )
    (package / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files),
        encoding="ascii",
        newline="\n",
    )


def build_r10_package(
    *, repository_root_value: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    protocol_path = root / DEFAULT_DATA_RELATIVE / "protocol/r10_protocol_manifest.json"
    evaluation_path = root / EVALUATION_REPORT_RELATIVE / "evaluation_report.json"
    lock_path = root / SELECTION_REPORT_RELATIVE / "selection_lock.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    promoted = list(evaluation.get("promoted_nodes", []))
    if protocol.get("status") != "evaluation-complete" or not promoted:
        raise R10PackageError("no R10 node passed promotion and engineering gates")
    package = root / R10_PACKAGE_RELATIVE
    if package.exists() and not overwrite:
        raise FileExistsError(package)
    package.mkdir(parents=True, exist_ok=True)
    frames = load_r10_frames(root)
    bundles = {node: _fit_bundle(frames[node], node, lock) for node in promoted}
    joblib.dump(bundles, package / "r10_joint_experts.joblib", compress=3)
    manifest = {
        "version": "mood-social-r10-candidate-package-v1",
        "protocol_version": R10_PROTOCOL_VERSION,
        "run_id": R10_PACKAGE_RUN_ID,
        "model_version": R10_MODEL_VERSION,
        "status": "offline-validated/integration-ready/device-validation-pending",
        "evidence_status": "adaptive-development/reused-benchmark/locked-procedure-estimate",
        "current_state_only": True,
        "diagnosis": False,
        "online_default_changed": False,
        "current_online_default": "MH-20260812-R5-001",
        "promoted_nodes": promoted,
        "r9_core": {"run_id": R9_PACKAGE_RUN_ID, "path": R9_PACKAGE_RELATIVE.as_posix()},
        "evidence_graph_sha256": graph_contract_sha256(),
        "rule_sha256": rule_contract_sha256(),
        "selection_lock_sha256": sha256_file(lock_path),
        "evaluation_report_sha256": sha256_file(evaluation_path),
        "release_limits": [
            "no new blind cohort",
            "no real-device observations",
            "no complete seven-domain same-person cohort",
            "no paired S10 facial-video/current-PHQ observations",
        ],
    }
    write_json(package / "manifest.json", manifest, overwrite=True)
    write_json(package / "request_schema.json", R10CandidateRequest.model_json_schema(), overwrite=True)
    write_json(package / "response_schema.json", R10CandidateResponse.model_json_schema(), overwrite=True)
    _write_checksums(package)
    result = {
        "status": manifest["status"],
        "package_run_id": R10_PACKAGE_RUN_ID,
        "promoted_nodes": promoted,
        "model_sha256": sha256_file(package / "r10_joint_experts.joblib"),
        "manifest_sha256": sha256_file(package / "manifest.json"),
        "checksums_sha256": sha256_file(package / "SHA256SUMS"),
        "online_default_changed": False,
    }
    write_json(root / R10_REPORT_RELATIVE / "package_build_report.json", result, overwrite=True)
    return result


def _verify(package: Path) -> dict[str, Any]:
    try:
        lines = (package / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise R10PackageError("R10 package checksums unavailable") from exc
    listed: set[str] = set()
    for line in lines:
        expected, name = line.split("  ", 1)
        path = (package / name).resolve()
        path.relative_to(package.resolve())
        if name in listed or not path.is_file() or sha256_file(path) != expected:
            raise R10PackageError(f"R10 checksum failed: {name}")
        listed.add(name)
    expected_files = {
        path.name for path in package.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    }
    if listed != expected_files:
        raise R10PackageError("R10 checksum coverage incomplete")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("run_id") != R10_PACKAGE_RUN_ID
        or manifest.get("evidence_graph_sha256") != graph_contract_sha256()
        or manifest.get("rule_sha256") != rule_contract_sha256()
    ):
        raise R10PackageError("R10 package contract drift")
    return manifest


@lru_cache(maxsize=4)
def _load(package_text: str) -> tuple[dict[str, Any], dict[str, R10JointBundle]]:
    package = Path(package_text)
    return _verify(package), joblib.load(package / "r10_joint_experts.joblib")


def _assessment(bundle: R10JointBundle, frame: pd.DataFrame) -> dict[str, Any]:
    p5, p10 = bundle.predict(frame)
    early, elevated = float(p5[0]), float(p10[0])
    level = 3 if elevated >= bundle.threshold_ge10 else 1 if early >= bundle.threshold_ge5 else 0
    return {
        "domain": bundle.node,
        "role": "local_joint_probability",
        "available": True,
        "probability_phq_ge5": early,
        "probability_phq_ge10": elevated,
        "threshold_ge5": bundle.threshold_ge5,
        "threshold_ge10": bundle.threshold_ge10,
        "domain_level": level,
        "model_version": f"{bundle.node}-r10",
        "evidence_grade": "adaptive-reused-locked-procedure",
        "limitations": ["not blind", "not real-device validated"],
    }


def _vote(name: str, value: Mapping[str, Any]) -> DomainEvidence | None:
    if not value.get("available"):
        return None
    level = value.get("domain_level")
    if level is None and name == "social":
        level = value.get("personal_change", {}).get("level")
    if level is None:
        return None
    return DomainEvidence(name, True, int(level), float(value.get("confidence", 0.5)))  # type: ignore[arg-type]


def infer_mood_social_r10_candidate(
    request: R10CandidateRequest | Mapping[str, Any],
    *,
    package_directory: Path | None = None,
    r9_package_directory: Path | None = None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[6]
    package = (package_directory or root / R10_PACKAGE_RELATIVE).resolve()
    manifest, bundles = _load(str(package))
    parsed = request if isinstance(request, R10CandidateRequest) else R10CandidateRequest.model_validate(request)
    core = infer_mood_social_r9_candidate(
        parsed.r9_payload(),
        package_directory=r9_package_directory or root / R9_PACKAGE_RELATIVE,
    )
    mapped = map_mood_social_features(parsed)
    frame = add_activity_sleep_interactions(_build_expert_frame(mapped))
    experts = dict(core["expert_assessments"])
    activity_available = bool(experts.get("activity", {}).get("available"))
    sleep_available = bool(experts.get("sleep", {}).get("available"))
    joint: dict[str, Any] = {}
    if "activity_sleep" in bundles and activity_available and sleep_available:
        joint["activity_sleep_v2"] = _assessment(bundles["activity_sleep"], frame)
    else:
        joint["activity_sleep_v2"] = {
            "domain": "activity_sleep_v2",
            "role": "local_joint_probability",
            "available": False,
            "fallback_reason": "node_not_promoted_or_signature_incomplete",
        }
    records = [
        HistoricalPHQ(item.date, item.score, item.known_at)
        for item in parsed.historical_phq9_assessments
        if item.known_at is not None
    ]
    history_features, history_audit = build_history_features(records, target_date=parsed.target_date)
    history_eligible = bool(
        history_features is not None
        and 76 <= float(history_features["history.age_days"]) <= 100
    )
    history_vote: dict[str, Any] = {
        "available": False,
        "low": False,
        "high": False,
        "last_age_days": history_audit.get("last_age_days"),
    }
    if history_eligible and "sleep_history" in bundles and sleep_available:
        age = float(history_features["history.age_days"])
        history_features["history.freshness_weight"] = float(np.exp(-(age - 30.0) / 90.0))
        history_features["history.decayed_last_score"] = (
            history_features["history.last_score"] * history_features["history.freshness_weight"]
        )
        combined = frame.copy()
        for key, value in history_features.items():
            combined[key] = value
        joint["sleep_history_v2"] = _assessment(bundles["sleep_history"], combined)
        history_vote = {
            "available": True,
            "domain_level": joint["sleep_history_v2"]["domain_level"],
            "low": bool(history_features.get("history.last_ge10", 0) == 0),
            "high": bool(history_features.get("history.last_ge10", 0) == 1),
            "last_age_days": age,
        }
    else:
        joint["sleep_history_v2"] = {
            "domain": "sleep_history_v2",
            "role": "local_joint_probability",
            "available": False,
            "fallback_reason": "requires_promoted_node_sleep_and_validated_76_100_day_history",
        }
    votes = [
        value
        for value in (
            _vote("activity", experts.get("activity", {})),
            _vote("sleep", experts.get("sleep", {})),
            _vote("social", experts.get("social", {})),
        )
        if value is not None
    ]
    fused = fuse_r10_current_state(
        votes,
        activity_sleep_joint=joint["activity_sleep_v2"],
        sleep_history_joint=joint["sleep_history_v2"],
        history=history_vote,
        facial=experts.get("facial_affect", {}),
        profile_background=experts.get("profile", {}),
        physiology_support=experts.get("physiology", {}),
    )
    route = fused["evidence_graph"]
    selected = route["selected_probability_node"]
    probability_sources = [selected] if selected else list(route["independent_probability_nodes"])
    available = bool(fused["operational_attention_state"]["available"])
    response = {
        "schema_version": R10_RESPONSE_SCHEMA,
        "request_id": parsed.request_id,
        "person_id": parsed.person_id,
        "target_date": parsed.target_date.isoformat(),
        "model_version": R10_MODEL_VERSION,
        "package_run_id": R10_PACKAGE_RUN_ID,
        "core_package_run_id": R9_PACKAGE_RUN_ID,
        "available": available,
        "current_state_only": True,
        "diagnosis": False,
        "expert_assessments": experts,
        "joint_expert_assessments": joint,
        "passive_current_state": core["passive_current_state"],
        "interaction_enhanced_current_state": core["interaction_enhanced_current_state"],
        "history_informed_integrated_state": fused["operational_attention_state"],
        "operational_attention_state": fused["operational_attention_state"],
        "facial_affect_assessment": core["facial_affect_assessment"],
        **fused,
        "fusion_mode": route["fusion_mode"],
        "evidence_signature": route["signature"],
        "probability_used_sources": probability_sources,
        "supporting_evidence_sources": list(route["supporting_nodes"]),
        "observed_sources": list(core.get("observed_sources", [])),
        "quality": {
            "activity_available": activity_available,
            "sleep_available": sleep_available,
            "no_cross_modal_imputation": True,
            "joint_probability_used_in_decision": selected is not None,
        },
        "freshness": {
            "history_probability_interval_days": [76, 100],
            "last_phq_age_days": history_audit.get("last_age_days"),
            "history_probability_eligible": history_eligible,
        },
        "lineage": {
            "r10_package_run_id": R10_PACKAGE_RUN_ID,
            "r9_core_package_run_id": R9_PACKAGE_RUN_ID,
            "package_manifest_sha256": sha256_file(package / "manifest.json"),
            "graph_contract_sha256": graph_contract_sha256(),
            "rule_contract_sha256": rule_contract_sha256(),
            "evaluation_report_sha256": manifest["evaluation_report_sha256"],
        },
        "abstain_reason": None if available else "insufficient_eligible_current_state_evidence",
        "fallback_chain": {
            "activity_sleep_v2": joint["activity_sleep_v2"].get("fallback_reason"),
            "sleep_history_v2": joint["sleep_history_v2"].get("fallback_reason"),
            "r9": R9_PACKAGE_RUN_ID,
            "production": "MH-20260812-R5-001",
        },
        "fallback_model_version": None if available else "mood-social-current-state-v3.3.3-r9",
        "fallback_reason": None if available else "r10_no_eligible_evidence",
        "release_status": manifest["status"],
        "limitations": manifest["release_limits"],
    }
    R10CandidateResponse.model_validate(response)
    return response


__all__ = [
    "MOOD_SOCIAL_R10_CANDIDATE_INFER_PATH",
    "R10CandidateRequest",
    "R10CandidateResponse",
    "R10_MODEL_VERSION",
    "R10_PACKAGE_RELATIVE",
    "R10_PACKAGE_RUN_ID",
    "R10PackageError",
    "build_r10_package",
    "infer_mood_social_r10_candidate",
]
