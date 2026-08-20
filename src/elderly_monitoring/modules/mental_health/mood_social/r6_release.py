"""Full-fit r6 research package and explicit fail-closed replay API.

R6 did not pass its development/offline promotion gates.  This artifact exists
for reproducible local integration and model preservation only; it never
changes the production ``/infer`` package selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from elderly_monitoring.modules.mental_health.mood_social.feature_mapper import map_mood_social_features
from elderly_monitoring.modules.mental_health.mood_social.package_selection import (
    EXPECTED_ACTIVE_RUN_ID,
    load_active_package_selection,
)
from elderly_monitoring.modules.mental_health.mood_social.pipeline import _build_expert_frame
from elderly_monitoring.modules.mental_health.mood_social.r4.calibration import FittedCalibration, fit_calibration
from elderly_monitoring.modules.mental_health.mood_social.r4.features import deployable_features_for_signature
from elderly_monitoring.modules.mental_health.mood_social.r5.fusion import select_train_only_workpoints
from elderly_monitoring.modules.mental_health.mood_social.r5.objectives import conditional_ordered_heads, project_independent_heads
from elderly_monitoring.modules.mental_health.mood_social.r5_release import runtime_signature
from elderly_monitoring.modules.mental_health.mood_social.r6.baseline import load_r6_track_frame
from elderly_monitoring.modules.mental_health.mood_social.r6.competition import _track_frame, predict_candidate
from elderly_monitoring.modules.mental_health.mood_social.r6.confirmation import (
    DEFAULT_CONFIRMATION_RELATIVE,
    confirmation_state,
    locked_confirmation_recipe,
)
from elderly_monitoring.modules.mental_health.mood_social.r6.contract import (
    R6_CONFIRMATION_SEED,
    R6_PROTOCOL_VERSION,
)
from elderly_monitoring.modules.mental_health.mood_social.r6.finalization import _select_calibration
from elderly_monitoring.modules.mental_health.mood_social.r6.objectives import r6_training_weights
from elderly_monitoring.modules.mental_health.mood_social.r6.selection import (
    R6CandidateSpec,
    fit_regular_model,
    predict_regular_model,
)
from elderly_monitoring.modules.mental_health.mood_social.schemas import MoodSocialInferRequest


R6_PACKAGE_RUN_ID = "MH-20260812-R6-001"
R6_MODEL_VERSION = "mood-social-current-state-v3.3.3-r6"
R6_PACKAGE_VERSION = "mood-social-r6-research-package-v1"
MOOD_SOCIAL_R6_CANDIDATE_INFER_PATH = "/v1/mental-health/mood-social/r6-candidate/infer"
R6_PACKAGE_RELATIVE = Path(
    "models/mental_health/mood_social/v3.3.3-r6/packages/MH-20260812-R6-001"
)
R6_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r6/OPT-V333-R6-007-package"
)
ONLINE_FALLBACK_RUN_ID = EXPECTED_ACTIVE_RUN_ID
REPLAY_ALLOWED_SIGNATURES = ("profile", "sleep")
ROUTE_BLOCKED_SIGNATURES = ("joint",)


class R6PackageError(RuntimeError):
    """Raised when the research package cannot be trusted or used safely."""


class R6CandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    request_id: str
    person_id: str
    target_date: str
    model_version: str
    package_run_id: str
    current_state_only: bool
    diagnosis: bool
    evidence_signature: str
    route_pattern: str
    historical_phq_used: bool
    history_attention_used: bool
    online_default_changed: bool
    online_fallback_package_run_id: str
    evidence_level: str
    release_status: str
    promotion_authorized: bool
    available: bool
    candidate_status: str
    fallback_required: bool
    reason: str | None
    probability_ge5: float | None
    probability_ge10: float | None
    workpoints: dict[str, Any] | None


@dataclass
class R6FittedHead:
    model: Any
    alpha: float
    prior: float

    def predict(self, frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
        probability = predict_regular_model(self.model, frame, features)
        if self.alpha < 1.0:
            probability = self.alpha * probability + (1.0 - self.alpha) * self.prior
        return np.clip(probability, 1.0e-6, 1.0 - 1.0e-6)


@dataclass
class R6CurrentStateBundle:
    deployable_models: Mapping[str, R6FittedHead]
    deployable_calibrators: Mapping[str, Mapping[str, FittedCalibration]]
    deployable_workpoints: Mapping[str, Mapping[str, Any]]
    deployable_specs: Mapping[str, Mapping[str, Any]]
    psyche_models: Mapping[str, R6FittedHead]
    psyche_calibrators: Mapping[str, FittedCalibration]
    psyche_workpoints: Mapping[str, Any]
    psyche_features: tuple[str, ...]
    psyche_spec: Mapping[str, Any]

    def predict_deployable(self, frame: pd.DataFrame, signature: str) -> tuple[np.ndarray, np.ndarray]:
        if signature not in {"profile", "sleep", "joint"}:
            raise R6PackageError(f"unsupported r6 runtime signature: {signature}")
        features = deployable_features_for_signature(signature)
        raw5 = self.deployable_models[f"{signature}:ge5"].predict(frame, features)
        conditional = self.deployable_models[f"{signature}:conditional_ge10"].predict(frame, features)
        raw5, raw10 = conditional_ordered_heads(raw5, conditional)
        p5 = self.deployable_calibrators[signature]["ge5"].predict(raw5)
        p10 = self.deployable_calibrators[signature]["ge10"].predict(raw10)
        return project_independent_heads(p5, p10)

    def predict_psyche_research(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raw5 = self.psyche_models["ge5"].predict(frame, self.psyche_features)
        conditional = self.psyche_models["conditional_ge10"].predict(frame, self.psyche_features)
        raw5, raw10 = conditional_ordered_heads(raw5, conditional)
        p5 = self.psyche_calibrators["ge5"].predict(raw5)
        p10 = self.psyche_calibrators["ge10"].predict(raw10)
        return project_independent_heads(p5, p10)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((item for item in directory.rglob("*") if item.is_file()), key=lambda item: item.relative_to(directory).as_posix()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _spec(lock: Mapping[str, Any], track: str) -> R6CandidateSpec:
    value = dict(lock["recipe"]["candidates"][track])
    value["seed"] = R6_CONFIRMATION_SEED
    return R6CandidateSpec(**value)


def _fit_head(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    spec: R6CandidateSpec,
    target: str,
) -> R6FittedHead:
    fit = frame.copy()
    fit["binary_target"] = fit[target].astype(int)
    weight = r6_training_weights(fit, "capped_class")
    model = fit_regular_model(fit, features, spec, sample_weight=weight)
    alpha = float(spec.params.get("alpha", 1.0)) if spec.family == "shrinkage_route_experts" else 1.0
    prior = float(np.average(fit[target], weights=weight))
    return R6FittedHead(model=model, alpha=alpha, prior=prior)


def _track_calibration(
    frame: pd.DataFrame,
    features: tuple[str, ...] | None,
    spec: R6CandidateSpec,
    track: str,
) -> tuple[dict[str, FittedCalibration], dict[str, Any], dict[str, Any]]:
    raw5 = pd.Series(np.nan, index=frame.index, dtype=float)
    raw10 = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold in range(5):
        train = frame.loc[frame["outer_fold"].ne(fold)]
        validation = frame.loc[frame["outer_fold"].eq(fold)]
        p5, p10 = predict_candidate(train, validation, features, spec, track)
        raw5.loc[validation.index] = p5
        raw10.loc[validation.index] = p10
    if raw5.isna().any() or raw10.isna().any():
        raise R6PackageError(f"r6 full-fit calibration OOF is incomplete for {track}")
    calibration_frame = frame.copy()
    calibration_frame["inner_fold"] = calibration_frame["outer_fold"].astype(int)
    calibrators: dict[str, FittedCalibration] = {}
    calibrated: dict[str, np.ndarray] = {}
    audit: dict[str, Any] = {}
    for head, raw in (("ge5", raw5.to_numpy(float)), ("ge10", raw10.to_numpy(float))):
        method, crossfit, selection_audit = _select_calibration(calibration_frame, raw, head)
        fit = calibration_frame[["global_participant_id", f"phq9_{head}_r3_target"]].copy()
        fit["binary_target"] = fit[f"phq9_{head}_r3_target"].astype(int)
        calibrators[head] = fit_calibration(fit, raw, method)  # type: ignore[arg-type]
        calibrated[head] = crossfit
        audit[head] = selection_audit
    p5, p10 = project_independent_heads(calibrated["ge5"], calibrated["ge10"])
    workpoints = {
        "ge5": select_train_only_workpoints(frame["phq9_ge5_r3_target"].to_numpy(int), p5),
        "ge10": select_train_only_workpoints(frame["phq9_ge10_r3_target"].to_numpy(int), p10),
    }
    return calibrators, workpoints, audit


def _device_schema() -> dict[str, Any]:
    schema = MoodSocialInferRequest.model_json_schema()
    schema["$id"] = "mood-social-r6-device-input-v1"
    schema["x-r6-current-state-contract"] = {
        "target": "target-day D current PHQ symptom attention only",
        "historical_phq": "accepted for shared request compatibility but never used by r6",
        "history_attention": "accepted for shared request compatibility but never used by r6",
        "forbidden_model_inputs": [
            "current or historical PHQ score/items",
            "history attention index",
            "concurrent questionnaires",
            "future records",
            "participant/source/dataset identifiers",
            "route/mask/coverage as learned risk features",
        ],
        "research_only_fields": "PSYCHE-D source__ fields are not exposed in this device schema",
        "no_evidence": "fail closed and require MH-20260802-013 fallback",
    }
    return schema


def build_r6_fullfit_package(
    *,
    repository_root: Path,
    package_directory: Path | None = None,
    report_directory: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    package = (package_directory or root / R6_PACKAGE_RELATIVE).resolve()
    report = (report_directory or root / R6_REPORT_RELATIVE).resolve()
    if (package.exists() or report.exists()) and not overwrite:
        raise FileExistsError("r6 full-fit package/report already exists")
    package.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    confirmation = root / DEFAULT_CONFIRMATION_RELATIVE
    state = confirmation_state(root)
    if not state["sealed"]:
        raise R6PackageError("r6 confirmation must be sealed before packaging")
    evaluation_path = confirmation / "confirmation_evaluation.json"
    lodo_path = confirmation / "lodo/lodo_report.json"
    seal_path = confirmation / "SEALED.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    lodo = json.loads(lodo_path.read_text(encoding="utf-8"))
    if evaluation.get("status") != "research_candidate_not_promoted":
        raise R6PackageError("r6 expected non-promotion status drifted")
    if evaluation.get("same_information_model_stacking_stop") is not True:
        raise R6PackageError("r6 frozen stopping decision drifted")
    if lodo.get("head_accounting", {}).get("passed") != 8:
        raise R6PackageError("r6 complete LODO evidence is unavailable")
    fallback = load_active_package_selection(repository_root=root)
    if fallback.package_run_id != ONLINE_FALLBACK_RUN_ID:
        raise R6PackageError("r6 production fallback identity drifted")
    lock = locked_confirmation_recipe(root)
    specs = {track: _spec(lock, track) for track in lock["recipe"]["candidates"]}

    multi, _ = load_r6_track_frame(root, seed=R6_CONFIRMATION_SEED, track="multisource_deployable")
    psyche, psyche_features = load_r6_track_frame(root, seed=R6_CONFIRMATION_SEED, track="psyche_d_single_source_research")
    assert psyche_features is not None
    joint, _ = _track_frame(root, R6_CONFIRMATION_SEED, "joint_route_recovery")
    calibrators: dict[str, Mapping[str, FittedCalibration]] = {}
    workpoints: dict[str, Mapping[str, Any]] = {}
    calibration_audit: dict[str, Any] = {}
    for signature, track_frame, track, spec in (
        ("profile", multi.loc[multi["feature_signature"].eq("profile")].copy(), "multisource_deployable", specs["multisource_deployable"]),
        ("sleep", multi.loc[multi["feature_signature"].eq("sleep")].copy(), "multisource_deployable", specs["multisource_deployable"]),
        ("joint", joint, "joint_route_recovery", specs["joint_route_recovery"]),
    ):
        track_frame["feature_signature"] = signature
        values = _track_calibration(track_frame, None, spec, track)
        calibrators[signature], workpoints[signature], calibration_audit[signature] = values
    psyche_calibrators, psyche_workpoints, calibration_audit["psyche_research"] = _track_calibration(
        psyche, psyche_features, specs["psyche_d_single_source_research"], "psyche_d_single_source_research"
    )

    deployable_models: dict[str, R6FittedHead] = {}
    for signature, part, spec in (
        ("profile", multi.loc[multi["feature_signature"].eq("profile")].copy(), specs["multisource_deployable"]),
        ("sleep", multi.loc[multi["feature_signature"].eq("sleep")].copy(), specs["multisource_deployable"]),
        ("joint", joint, specs["joint_route_recovery"]),
    ):
        features = deployable_features_for_signature(signature)
        deployable_models[f"{signature}:ge5"] = _fit_head(part, features, spec, "phq9_ge5_r3_target")
        deployable_models[f"{signature}:conditional_ge10"] = _fit_head(
            part.loc[part["phq9_ge5_r3_target"].eq(1)], features, spec, "phq9_ge10_r3_target"
        )
    psyche_models = {
        "ge5": _fit_head(psyche, psyche_features, specs["psyche_d_single_source_research"], "phq9_ge5_r3_target"),
        "conditional_ge10": _fit_head(
            psyche.loc[psyche["phq9_ge5_r3_target"].eq(1)],
            psyche_features,
            specs["psyche_d_single_source_research"],
            "phq9_ge10_r3_target",
        ),
    }
    bundle = R6CurrentStateBundle(
        deployable_models=deployable_models,
        deployable_calibrators=calibrators,
        deployable_workpoints=workpoints,
        deployable_specs={key: dict(value) for key, value in lock["recipe"]["candidates"].items()},
        psyche_models=psyche_models,
        psyche_calibrators=psyche_calibrators,
        psyche_workpoints=psyche_workpoints,
        psyche_features=tuple(psyche_features),
        psyche_spec=dict(lock["recipe"]["candidates"]["psyche_d_single_source_research"]),
    )
    model_path = package / "r6_current_state_bundle.joblib"
    joblib.dump(bundle, model_path, compress=3)
    schema_path = package / "device_input_schema.json"
    _write_json(schema_path, _device_schema())
    fallback_manifest = fallback.package_directory / "manifest.json"
    fallback_checksums = fallback.package_directory / "SHA256SUMS"
    manifest = {
        "version": R6_PACKAGE_VERSION,
        "run_id": R6_PACKAGE_RUN_ID,
        "model_version": R6_MODEL_VERSION,
        "protocol_version": R6_PROTOCOL_VERSION,
        "status": "research-only/integration-ready/shadow-blocked",
        "promotion_authorized": False,
        "promotion_failure": "all r6 development gates failed; best confirmation delta AUPRC < 0.002",
        "evidence_level": "sealed reused-cohort confirmation; no new subjects and no real-device parity",
        "current_state_only": True,
        "historical_or_current_phq_used_as_model_input": False,
        "history_attention_used_as_model_input": False,
        "online_default_changed": False,
        "online_default_package_run_id": ONLINE_FALLBACK_RUN_ID,
        "candidate_model_path": model_path.name,
        "candidate_model_sha256": _sha256(model_path),
        "device_schema_path": schema_path.name,
        "device_schema_sha256": _sha256(schema_path),
        "replay_allowed_signatures": list(REPLAY_ALLOWED_SIGNATURES),
        "route_blocked_signatures": list(ROUTE_BLOCKED_SIGNATURES),
        "replay_always_requires_fallback": True,
        "unsupported_signature_policy": "fail_closed_and_require_MH-20260802-013_fallback",
        "research_track": {
            "track": "psyche_d_single_source_research",
            "device_api_eligible": False,
            "reason": "source__ fields have no audited device generator parity",
        },
        "joint_route": {
            "device_api_eligible": False,
            "lodo_unblock_threshold_passed": bool(lodo["joint_nhanes_unblock_gate"]["pass"]),
            "reason": "joint confirmation main table regressed and r6 development gate failed",
        },
        "fallback": {
            "run_id": fallback.package_run_id,
            "package_directory": fallback.package_directory.relative_to(root).as_posix(),
            "manifest_sha256": _sha256(fallback_manifest),
            "checksums_sha256": _sha256(fallback_checksums),
            "verified_during_build": True,
        },
        "confirmation": {
            "recipe_lock_sha256": _sha256(confirmation / "candidate_recipe_lock.json"),
            "evaluation_sha256": _sha256(evaluation_path),
            "lodo_sha256": _sha256(lodo_path),
            "seal_sha256": _sha256(seal_path),
        },
        "calibration": calibration_audit,
        "workpoints": {"source": "post-seal full-fit cross-fitted training OOF only", **workpoints, "psyche_research": psyche_workpoints},
    }
    manifest_path = package / "manifest.json"
    _write_json(manifest_path, manifest)
    selection = {
        "version": "mood-social-r6-research-selection-v1",
        "r6_package_run_id": R6_PACKAGE_RUN_ID,
        "r6_status": manifest["status"],
        "r6_online_default": False,
        "active_online_package_run_id": ONLINE_FALLBACK_RUN_ID,
        "real_device_data_available": False,
        "new_subjects_available": False,
        "shadow_started": False,
        "promotion_authorized": False,
    }
    _write_json(package / "selection.json", selection)
    checksum_paths = sorted((path for path in package.iterdir() if path.is_file() and path.name != "SHA256SUMS"), key=lambda path: path.name)
    (package / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="ascii",
        newline="\n",
    )
    result = {
        "status": manifest["status"],
        "package_run_id": R6_PACKAGE_RUN_ID,
        "package_directory": package.relative_to(root).as_posix(),
        "package_tree_sha256": _tree_sha256(package),
        "model_sha256": _sha256(model_path),
        "manifest_sha256": _sha256(manifest_path),
        "checksums_sha256": _sha256(package / "SHA256SUMS"),
        "online_default_package_run_id": ONLINE_FALLBACK_RUN_ID,
        "online_default_changed": False,
    }
    _write_json(report / "package_build_report.json", result)
    _write_json(report / "fullfit_calibration_audit.json", calibration_audit)
    return result


def _verify_checksums(package: Path) -> None:
    checksum_path = package / "SHA256SUMS"
    if not checksum_path.is_file():
        raise R6PackageError("r6 package SHA256SUMS is missing")
    listed: set[str] = set()
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        try:
            expected, name = line.split("  ", 1)
        except ValueError as error:
            raise R6PackageError("r6 package SHA256SUMS is malformed") from error
        path = (package / name).resolve()
        try:
            path.relative_to(package)
        except ValueError as error:
            raise R6PackageError("r6 package checksum path escapes package") from error
        if name in listed or not path.is_file() or _sha256(path) != expected:
            raise R6PackageError(f"r6 package checksum failed: {name}")
        listed.add(name)
    expected_files = {path.name for path in package.iterdir() if path.is_file() and path.name != "SHA256SUMS"}
    if listed != expected_files:
        raise R6PackageError("r6 package checksum coverage is incomplete")


def load_r6_fullfit_package(package_directory: Path) -> tuple[R6CurrentStateBundle, dict[str, Any]]:
    package = Path(package_directory).resolve()
    _verify_checksums(package)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("version") != R6_PACKAGE_VERSION
        or manifest.get("run_id") != R6_PACKAGE_RUN_ID
        or manifest.get("promotion_authorized") is not False
        or manifest.get("online_default_changed") is not False
        or manifest.get("online_default_package_run_id") != ONLINE_FALLBACK_RUN_ID
    ):
        raise R6PackageError("r6 package identity or fallback policy changed")
    model_path = package / str(manifest["candidate_model_path"])
    if _sha256(model_path) != manifest["candidate_model_sha256"]:
        raise R6PackageError("r6 candidate model hash changed")
    bundle = joblib.load(model_path)
    if not isinstance(bundle, R6CurrentStateBundle):
        raise R6PackageError("r6 candidate model bundle type changed")
    return bundle, manifest


@dataclass(frozen=True)
class R6IntegrationRunner:
    package_directory: Path
    bundle: R6CurrentStateBundle
    manifest: Mapping[str, Any]

    @classmethod
    def from_package(cls, package_directory: Path) -> "R6IntegrationRunner":
        bundle, manifest = load_r6_fullfit_package(package_directory)
        return cls(Path(package_directory).resolve(), bundle, manifest)

    def predict(self, request: MoodSocialInferRequest | Mapping[str, Any]) -> dict[str, Any]:
        parsed = request if isinstance(request, MoodSocialInferRequest) else MoodSocialInferRequest.model_validate(request)
        mapped = map_mood_social_features(parsed)
        frame = _build_expert_frame(mapped)
        signature, route_pattern = runtime_signature(frame)
        base = {
            "schema_version": "mood_social_r6_candidate_response_v1",
            "request_id": parsed.request_id,
            "person_id": parsed.person_id,
            "target_date": parsed.target_date.isoformat(),
            "model_version": R6_MODEL_VERSION,
            "package_run_id": R6_PACKAGE_RUN_ID,
            "current_state_only": True,
            "diagnosis": False,
            "evidence_signature": signature,
            "route_pattern": route_pattern,
            "historical_phq_used": False,
            "history_attention_used": False,
            "online_default_changed": False,
            "online_fallback_package_run_id": ONLINE_FALLBACK_RUN_ID,
            "evidence_level": "research replay only; no new-subject or real-device confirmation",
            "release_status": "research-only/integration-ready/shadow-blocked",
            "promotion_authorized": False,
        }
        if signature in ROUTE_BLOCKED_SIGNATURES:
            return {
                **base, "available": False, "candidate_status": "route_blocked",
                "fallback_required": True,
                "reason": "r6 joint confirmation regressed; use MH-20260802-013",
                "probability_ge5": None, "probability_ge10": None, "workpoints": None,
            }
        if signature not in REPLAY_ALLOWED_SIGNATURES:
            return {
                **base, "available": False, "candidate_status": "unsupported_signature",
                "fallback_required": True,
                "reason": "runtime evidence has no r6 replay-eligible signature",
                "probability_ge5": None, "probability_ge10": None, "workpoints": None,
            }
        p5, p10 = self.bundle.predict_deployable(frame, signature)
        probability5, probability10 = float(p5[0]), float(p10[0])
        if not (0.0 <= probability10 <= probability5 <= 1.0):
            raise R6PackageError("r6 runtime dual-head probability is invalid")
        workpoint_output: dict[str, Any] = {}
        for head, probability in (("ge5", probability5), ("ge10", probability10)):
            workpoint_output[head] = {
                point: {"threshold": float(values["threshold"]), "positive": bool(probability >= float(values["threshold"]))}
                for point, values in self.bundle.deployable_workpoints[signature][head].items()
                if point in {"competition", "safety"}
            }
        return {
            **base,
            "available": True,
            "candidate_status": "research_shadow_replay_available",
            "fallback_required": True,
            "reason": "r6 did not pass promotion gates; probability is research-only",
            "probability_ge5": probability5,
            "probability_ge10": probability10,
            "workpoints": workpoint_output,
        }


def infer_mood_social_r6_candidate(
    request: MoodSocialInferRequest | Mapping[str, Any],
    *,
    package_directory: Path | None = None,
) -> dict[str, Any]:
    """Explicit algorithm-side research replay; never changes production default."""

    root = Path(__file__).resolve().parents[5]
    runner = R6IntegrationRunner.from_package(package_directory or root / R6_PACKAGE_RELATIVE)
    return runner.predict(request)


__all__ = [
    "MOOD_SOCIAL_R6_CANDIDATE_INFER_PATH", "ONLINE_FALLBACK_RUN_ID",
    "REPLAY_ALLOWED_SIGNATURES", "ROUTE_BLOCKED_SIGNATURES", "R6CandidateResponse",
    "R6CurrentStateBundle", "R6IntegrationRunner", "R6PackageError", "R6FittedHead",
    "R6_PACKAGE_RELATIVE", "R6_PACKAGE_RUN_ID", "build_r6_fullfit_package",
    "infer_mood_social_r6_candidate", "load_r6_fullfit_package",
]
