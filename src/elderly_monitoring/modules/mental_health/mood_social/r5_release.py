"""Full-fit r5 current-state package and fail-closed integration replay API.

This module is deliberately outside the sealed ``r5`` experiment package so
post-confirmation engineering cannot change the immutable evaluation graph.
It does not replace the active MH-20260802-013 online package.
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

from elderly_monitoring.modules.mental_health.mood_social.feature_mapper import (
    map_mood_social_features,
)
from elderly_monitoring.modules.mental_health.mood_social.pipeline import (
    _build_expert_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.package_selection import (
    EXPECTED_ACTIVE_RUN_ID,
    load_active_package_selection,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.calibration import (
    FittedCalibration,
    fit_calibration,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.features import (
    DEPLOYABLE_ACTIVITY_FEATURES,
    DEPLOYABLE_PROFILE_FEATURES,
    DEPLOYABLE_SLEEP_FEATURES,
    deployable_features_for_signature,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    participant_equal_weights,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.competition import (
    _load_track_frame,
    _selected_structure,
    _spec_from_dict,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.confirmation import (
    locked_confirmation_recipe,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.contract import (
    R5_CONFIRMATION_SEED,
    R5_PROTOCOL_VERSION,
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.fusion import (
    select_dual_calibration,
    select_train_only_workpoints,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.selection import (
    R5CandidateSpec,
    fit_r5_model,
    predict_r5_model,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.structure import (
    conditional_ordered_heads,
    project_independent_heads,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.weights import (
    r5_training_weights,
)
from elderly_monitoring.modules.mental_health.mood_social.schemas import (
    MoodSocialInferRequest,
)


R5_PACKAGE_RUN_ID = "MH-20260812-R5-001"
R5_MODEL_VERSION = "mood-social-current-state-v3.3.3-r5"
MOOD_SOCIAL_R5_CANDIDATE_INFER_PATH = (
    "/v1/mental-health/mood-social/r5-candidate/infer"
)
R5_PACKAGE_VERSION = "mood-social-r5-integration-package-v1"
R5_PACKAGE_RELATIVE = Path(
    "models/mental_health/mood_social/v3.3.3-r5/packages/MH-20260812-R5-001"
)
R5_CONFIRMATION_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/OPT-V333-R5-006-confirmation"
)
R5_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/OPT-V333-R5-007-package"
)
ONLINE_FALLBACK_RUN_ID = EXPECTED_ACTIVE_RUN_ID
INTEGRATION_ALLOWED_SIGNATURES = ("profile", "sleep")
ROUTE_BLOCKED_SIGNATURES = ("joint",)


class R5PackageError(RuntimeError):
    """Raised when r5 package selection or contents fail closed."""


class R5CandidateResponse(BaseModel):
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
    available: bool
    candidate_status: str
    fallback_required: bool
    reason: str | None
    probability_ge5: float | None
    probability_ge10: float | None
    workpoints: dict[str, Any] | None


@dataclass
class R5CurrentStateBundle:
    multisource_models: Mapping[str, Any]
    multisource_calibrators: Mapping[str, FittedCalibration]
    multisource_workpoints: Mapping[str, Any]
    multisource_spec: Mapping[str, Any]
    psyche_models: Mapping[str, Any]
    psyche_calibrators: Mapping[str, FittedCalibration]
    psyche_workpoints: Mapping[str, Any]
    psyche_features: tuple[str, ...]
    psyche_spec: Mapping[str, Any]

    def predict_multisource(
        self, frame: pd.DataFrame, signature: str
    ) -> tuple[np.ndarray, np.ndarray]:
        if signature not in {"profile", "sleep", "joint"}:
            raise R5PackageError(f"unsupported r5 runtime signature: {signature}")
        features = deployable_features_for_signature(signature)
        raw5 = predict_r5_model(
            self.multisource_models[f"{signature}:ge5"], frame, features
        )
        raw10 = predict_r5_model(
            self.multisource_models[f"{signature}:ge10"], frame, features
        )
        p5 = self.multisource_calibrators["ge5"].predict(raw5)
        p10 = self.multisource_calibrators["ge10"].predict(raw10)
        return project_independent_heads(p5, p10)

    def predict_psyche_research(
        self, frame: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        raw5 = predict_r5_model(
            self.psyche_models["ge5"], frame, self.psyche_features
        )
        conditional = predict_r5_model(
            self.psyche_models["conditional_ge10"], frame, self.psyche_features
        )
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
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (item for item in directory.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(directory).as_posix(),
    ):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _fit_binary(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    spec: R5CandidateSpec,
    target_column: str,
    *,
    psyche_weight_scheme: str | None = None,
) -> Any:
    fit = frame.copy()
    fit["binary_target"] = fit[target_column].astype(int)
    weight = (
        participant_equal_weights(fit)
        if psyche_weight_scheme is None
        else r5_training_weights(fit, psyche_weight_scheme)  # type: ignore[arg-type]
    )
    return fit_r5_model(fit, features, spec, sample_weight=weight)


def _fullfit_calibration(
    frame: pd.DataFrame,
    oof: pd.DataFrame,
) -> tuple[dict[str, FittedCalibration], dict[str, Any], dict[str, Any]]:
    aligned = frame[
        [
            "r5_row_id",
            "global_participant_id",
            "outer_fold",
            "phq9_ge5_r3_target",
            "phq9_ge10_r3_target",
        ]
    ].merge(
        oof[
            [
                "r5_row_id",
                "raw_probability_ge5",
                "raw_probability_ge10",
            ]
        ],
        on="r5_row_id",
        how="left",
        validate="one_to_one",
    )
    if aligned[["raw_probability_ge5", "raw_probability_ge10"]].isna().any().any():
        raise R5PackageError("r5 full-fit calibration OOF alignment is incomplete")
    aligned["inner_fold"] = aligned["outer_fold"].astype(int)
    raw5 = aligned["raw_probability_ge5"].to_numpy(float)
    raw10 = aligned["raw_probability_ge10"].to_numpy(float)
    (calibrated5, calibrated10), _outer, audit = select_dual_calibration(
        aligned, (raw5, raw10), (raw5, raw10)
    )
    methods = audit["selected"]["methods"]
    calibrators: dict[str, FittedCalibration] = {}
    for head, target_column, raw in (
        ("ge5", "phq9_ge5_r3_target", raw5),
        ("ge10", "phq9_ge10_r3_target", raw10),
    ):
        fit = aligned.copy()
        fit["binary_target"] = fit[target_column].astype(int)
        calibrators[head] = fit_calibration(fit, raw, methods[head])
    workpoints = {
        "ge5": select_train_only_workpoints(
            aligned["phq9_ge5_r3_target"].to_numpy(int), calibrated5
        ),
        "ge10": select_train_only_workpoints(
            aligned["phq9_ge10_r3_target"].to_numpy(int), calibrated10
        ),
    }
    return calibrators, workpoints, audit


def _device_schema() -> dict[str, Any]:
    schema = MoodSocialInferRequest.model_json_schema()
    schema["$id"] = "mood-social-r5-device-input-v1"
    schema["x-r5-current-state-contract"] = {
        "timezone": "Asia/Shanghai",
        "target": "target-day D current PHQ symptom attention only",
        "state_window": "D-6 through D completed natural days",
        "history_window": "D-28 through D-1 passive history; D is current_daily_features",
        "forbidden_model_inputs": [
            "historical_phq9_assessments",
            "history_attention_indices",
            "current PHQ score/items",
            "other concurrent questionnaires",
            "participant/source/dataset identifiers",
            "route/mask/coverage values as learned risk features",
            "future records",
        ],
        "accepted_but_not_used_by_r5": [
            "historical_phq9_assessments",
            "history_attention_indices",
        ],
        "missingness": "Pydantic semantic validation then signature routing; unsupported evidence fails closed",
        "order": "history dates must be unique, ascending and before target_date",
        "units": "minutes, minute-of-day, bpm, milliseconds, counts and normalized ratios as field names specify",
    }
    return schema


def build_r5_fullfit_package(
    *,
    repository_root: Path,
    package_directory: Path | None = None,
    report_directory: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    package = (package_directory or root / R5_PACKAGE_RELATIVE).resolve()
    report = (report_directory or root / R5_REPORT_RELATIVE).resolve()
    if (package.exists() or report.exists()) and not overwrite:
        raise FileExistsError("r5 full-fit package/report already exists")
    package.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    confirmation = root / R5_CONFIRMATION_RELATIVE
    evaluation_path = confirmation / "confirmation_evaluation.json"
    lodo_path = confirmation / "lodo/lodo_report.json"
    lock_path = confirmation / "candidate_recipe_lock.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    lodo = json.loads(lodo_path.read_text(encoding="utf-8"))
    lock = locked_confirmation_recipe(root)
    if evaluation.get("status") != "offline_candidate_pass":
        raise R5PackageError("r5 confirmation did not pass the frozen offline gate")
    if lodo.get("status") not in {"pass", "complete_with_route_local_warnings"}:
        raise R5PackageError("r5 complete LODO evidence is unavailable")
    if lodo["sources"]["nhanes"]["route_warning"] != "block_related_source_route":
        raise R5PackageError("r5 frozen joint-route warning is missing")
    fallback = load_active_package_selection(repository_root=root)
    if fallback.package_run_id != ONLINE_FALLBACK_RUN_ID:
        raise R5PackageError("r5 fallback package identity drifted")

    specs = {
        track: _spec_from_dict(value)
        for track, value in lock["recipe"]["candidates"].items()
    }
    multisource, _ = _load_track_frame(
        root, R5_CONFIRMATION_SEED, "multisource_deployable"
    )
    psyche, psyche_features = _load_track_frame(
        root, R5_CONFIRMATION_SEED, "psyche_d_single_source_research"
    )
    assert psyche_features is not None
    multi_oof = pd.read_parquet(
        confirmation
        / "candidate/multisource_deployable"
        / f"seed-{R5_CONFIRMATION_SEED}.parquet"
    )
    psyche_oof = pd.read_parquet(
        confirmation
        / "candidate/psyche_d_single_source_research"
        / f"seed-{R5_CONFIRMATION_SEED}.parquet"
    )
    multi_calibrators, multi_workpoints, multi_calibration_audit = _fullfit_calibration(
        multisource, multi_oof
    )
    psyche_calibrators, psyche_workpoints, psyche_calibration_audit = _fullfit_calibration(
        psyche, psyche_oof
    )
    multi_models: dict[str, Any] = {}
    for signature in ("profile", "sleep", "joint"):
        part = multisource.loc[multisource["feature_signature"].eq(signature)].copy()
        features = deployable_features_for_signature(signature)
        for head, target_column in (
            ("ge5", "phq9_ge5_r3_target"),
            ("ge10", "phq9_ge10_r3_target"),
        ):
            multi_models[f"{signature}:{head}"] = _fit_binary(
                part, features, specs["multisource_deployable"], target_column
            )
    structure = _selected_structure(root)
    psyche_models = {
        "ge5": _fit_binary(
            psyche,
            psyche_features,
            specs["psyche_d_single_source_research"],
            "phq9_ge5_r3_target",
            psyche_weight_scheme=structure["weight_scheme"],
        ),
        "conditional_ge10": _fit_binary(
            psyche.loc[psyche["phq9_ge5_r3_target"].eq(1)],
            psyche_features,
            specs["psyche_d_single_source_research"],
            "phq9_ge10_r3_target",
            psyche_weight_scheme=structure["weight_scheme"],
        ),
    }
    bundle = R5CurrentStateBundle(
        multisource_models=multi_models,
        multisource_calibrators=multi_calibrators,
        multisource_workpoints=multi_workpoints,
        multisource_spec=lock["recipe"]["candidates"]["multisource_deployable"],
        psyche_models=psyche_models,
        psyche_calibrators=psyche_calibrators,
        psyche_workpoints=psyche_workpoints,
        psyche_features=tuple(psyche_features),
        psyche_spec=lock["recipe"]["candidates"]["psyche_d_single_source_research"],
    )
    model_path = package / "r5_current_state_bundle.joblib"
    joblib.dump(bundle, model_path, compress=3)
    schema_path = package / "device_input_schema.json"
    _write_json(schema_path, _device_schema())
    fallback_manifest = fallback.package_directory / "manifest.json"
    fallback_checksums = fallback.package_directory / "SHA256SUMS"
    manifest = {
        "version": R5_PACKAGE_VERSION,
        "run_id": R5_PACKAGE_RUN_ID,
        "model_version": R5_MODEL_VERSION,
        "protocol_version": R5_PROTOCOL_VERSION,
        "status": "integration-ready/shadow-blocked",
        "evidence_level": "sealed reused-cohort confirmation; no real-device parity",
        "current_state_only": True,
        "historical_or_current_phq_used_as_model_input": False,
        "online_default_changed": False,
        "online_default_package_run_id": ONLINE_FALLBACK_RUN_ID,
        "candidate_model_path": model_path.name,
        "candidate_model_sha256": _sha256(model_path),
        "device_schema_path": schema_path.name,
        "device_schema_sha256": _sha256(schema_path),
        "integration_allowed_signatures": list(INTEGRATION_ALLOWED_SIGNATURES),
        "route_blocked_signatures": list(ROUTE_BLOCKED_SIGNATURES),
        "unsupported_signature_policy": "fail_closed_and_require_versioned_fallback",
        "research_track": {
            "track": "psyche_d_single_source_research",
            "device_api_eligible": False,
            "reason": "audited local longitudinal source fields lack real-device generator parity",
        },
        "fallback": {
            "run_id": fallback.package_run_id,
            "package_directory": fallback.package_directory.relative_to(root).as_posix(),
            "manifest_sha256": _sha256(fallback_manifest),
            "checksums_sha256": _sha256(fallback_checksums),
            "verified_during_build": True,
        },
        "confirmation": {
            "recipe_lock_sha256": _sha256(lock_path),
            "evaluation_sha256": _sha256(evaluation_path),
            "lodo_sha256": _sha256(lodo_path),
            "candidate_oof_sha256": {
                "multisource_deployable": sha256_file(
                    confirmation / "candidate/multisource_deployable" / f"seed-{R5_CONFIRMATION_SEED}.parquet"
                ),
                "psyche_d_single_source_research": sha256_file(
                    confirmation / "candidate/psyche_d_single_source_research" / f"seed-{R5_CONFIRMATION_SEED}.parquet"
                ),
            },
        },
        "calibration": {
            "multisource": multi_calibration_audit["selected"],
            "psyche_research": psyche_calibration_audit["selected"],
        },
        "workpoints": {
            "source": "sealed confirmation cross-fitted OOF, train-only for full-fit package",
            "multisource": multi_workpoints,
            "psyche_research": psyche_workpoints,
        },
    }
    manifest_path = package / "manifest.json"
    _write_json(manifest_path, manifest)
    checksum_paths = sorted(
        (path for path in package.iterdir() if path.is_file() and path.name != "SHA256SUMS"),
        key=lambda path: path.name,
    )
    (package / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="ascii",
        newline="\n",
    )
    selection = {
        "version": "mood-social-r5-integration-selection-v1",
        "r5_package_run_id": R5_PACKAGE_RUN_ID,
        "r5_package_directory": package.relative_to(root).as_posix(),
        "r5_status": "integration-ready/shadow-blocked",
        "r5_online_default": False,
        "active_online_package_run_id": ONLINE_FALLBACK_RUN_ID,
        "real_device_data_available": False,
        "shadow_started": False,
        "joint_route_enabled": False,
        "profile_sleep_local_replay_enabled": True,
    }
    _write_json(package / "selection.json", selection)
    # selection was appended after the first checksum assembly; regenerate it.
    checksum_paths = sorted(
        (path for path in package.iterdir() if path.is_file() and path.name != "SHA256SUMS"),
        key=lambda path: path.name,
    )
    (package / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="ascii",
        newline="\n",
    )
    result = {
        "status": "integration-ready/shadow-blocked",
        "package_run_id": R5_PACKAGE_RUN_ID,
        "package_directory": package.relative_to(root).as_posix(),
        "package_tree_sha256": _tree_sha256(package),
        "model_sha256": _sha256(model_path),
        "manifest_sha256": _sha256(manifest_path),
        "checksums_sha256": _sha256(package / "SHA256SUMS"),
        "online_default_package_run_id": ONLINE_FALLBACK_RUN_ID,
        "online_default_changed": False,
    }
    _write_json(report / "package_build_report.json", result)
    _write_json(report / "fullfit_calibration_audit.json", {
        "multisource": multi_calibration_audit,
        "psyche_research": psyche_calibration_audit,
    })
    return result


def _verify_checksums(package: Path) -> None:
    checksum_path = package / "SHA256SUMS"
    if not checksum_path.is_file():
        raise R5PackageError("r5 package SHA256SUMS is missing")
    listed: set[str] = set()
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        try:
            expected, name = line.split("  ", 1)
        except ValueError as exc:
            raise R5PackageError("r5 package SHA256SUMS is malformed") from exc
        path = (package / name).resolve()
        try:
            path.relative_to(package)
        except ValueError as exc:
            raise R5PackageError("r5 package checksum path escapes package") from exc
        if name in listed or not path.is_file() or _sha256(path) != expected:
            raise R5PackageError(f"r5 package checksum failed: {name}")
        listed.add(name)
    expected_files = {
        path.name for path in package.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    }
    if listed != expected_files:
        raise R5PackageError("r5 package checksum coverage is incomplete")


def load_r5_fullfit_package(package_directory: Path) -> tuple[R5CurrentStateBundle, dict[str, Any]]:
    package = Path(package_directory).resolve()
    _verify_checksums(package)
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("version") != R5_PACKAGE_VERSION
        or manifest.get("run_id") != R5_PACKAGE_RUN_ID
        or manifest.get("online_default_changed") is not False
        or manifest.get("online_default_package_run_id") != ONLINE_FALLBACK_RUN_ID
    ):
        raise R5PackageError("r5 package identity or fallback policy changed")
    model_path = package / str(manifest["candidate_model_path"])
    if _sha256(model_path) != manifest["candidate_model_sha256"]:
        raise R5PackageError("r5 candidate model hash changed")
    bundle = joblib.load(model_path)
    if not isinstance(bundle, R5CurrentStateBundle):
        raise R5PackageError("r5 candidate model bundle type changed")
    return bundle, manifest


def runtime_signature(frame: pd.DataFrame) -> tuple[str, str]:
    def available(features: tuple[str, ...]) -> bool:
        masks = [f"feature_mask.{name}" for name in features]
        return any(int(frame.iloc[0].get(name, 0) or 0) == 1 for name in masks)

    activity = available(DEPLOYABLE_ACTIVITY_FEATURES)
    sleep = available(DEPLOYABLE_SLEEP_FEATURES)
    profile = available(DEPLOYABLE_PROFILE_FEATURES)
    bits = f"{int(activity)}{int(sleep)}{int(profile)}"
    if bits == "001":
        return "profile", bits
    if bits == "010":
        return "sleep", bits
    if bits in {"101", "111"}:
        return "joint", bits
    return "unsupported", bits


@dataclass(frozen=True)
class R5IntegrationRunner:
    package_directory: Path
    bundle: R5CurrentStateBundle
    manifest: Mapping[str, Any]

    @classmethod
    def from_package(cls, package_directory: Path) -> "R5IntegrationRunner":
        bundle, manifest = load_r5_fullfit_package(package_directory)
        return cls(Path(package_directory).resolve(), bundle, manifest)

    def predict(
        self, request: MoodSocialInferRequest | Mapping[str, Any]
    ) -> dict[str, Any]:
        parsed = (
            request
            if isinstance(request, MoodSocialInferRequest)
            else MoodSocialInferRequest.model_validate(request)
        )
        mapped = map_mood_social_features(parsed)
        frame = _build_expert_frame(mapped)
        signature, route_pattern = runtime_signature(frame)
        base = {
            "schema_version": "mood_social_r5_candidate_response_v1",
            "request_id": parsed.request_id,
            "person_id": parsed.person_id,
            "target_date": parsed.target_date.isoformat(),
            "model_version": R5_MODEL_VERSION,
            "package_run_id": R5_PACKAGE_RUN_ID,
            "current_state_only": True,
            "diagnosis": False,
            "evidence_signature": signature,
            "route_pattern": route_pattern,
            "historical_phq_used": False,
            "history_attention_used": False,
            "online_default_changed": False,
            "online_fallback_package_run_id": ONLINE_FALLBACK_RUN_ID,
            "evidence_level": "integration replay only; no real-device parity",
        }
        if signature in ROUTE_BLOCKED_SIGNATURES:
            return {
                **base,
                "available": False,
                "candidate_status": "route_blocked",
                "fallback_required": True,
                "reason": "r5 confirmation LODO blocked the joint signature",
                "probability_ge5": None,
                "probability_ge10": None,
                "workpoints": None,
            }
        if signature not in INTEGRATION_ALLOWED_SIGNATURES:
            return {
                **base,
                "available": False,
                "candidate_status": "unsupported_signature",
                "fallback_required": True,
                "reason": "runtime evidence has no r5 deployable signature",
                "probability_ge5": None,
                "probability_ge10": None,
                "workpoints": None,
            }
        p5, p10 = self.bundle.predict_multisource(frame, signature)
        probability5 = float(p5[0])
        probability10 = float(p10[0])
        if not (0.0 <= probability10 <= probability5 <= 1.0):
            raise R5PackageError("r5 runtime dual-head probability is invalid")
        workpoint_output: dict[str, Any] = {}
        for head, probability in (("ge5", probability5), ("ge10", probability10)):
            workpoint_output[head] = {
                point: {
                    "threshold": float(values["threshold"]),
                    "positive": bool(probability >= float(values["threshold"])),
                }
                for point, values in self.bundle.multisource_workpoints[head].items()
                if point in {"competition", "safety"}
            }
        return {
            **base,
            "available": True,
            "candidate_status": "integration_replay_available",
            "fallback_required": False,
            "reason": None,
            "probability_ge5": probability5,
            "probability_ge10": probability10,
            "workpoints": workpoint_output,
        }


def infer_mood_social_r5_candidate(
    request: MoodSocialInferRequest | Mapping[str, Any],
    *,
    package_directory: Path | None = None,
) -> dict[str, Any]:
    """Algorithm-side integration replay API; it never changes online default."""

    root = Path(__file__).resolve().parents[5]
    runner = R5IntegrationRunner.from_package(
        package_directory or root / R5_PACKAGE_RELATIVE
    )
    return runner.predict(request)


__all__ = [
    "INTEGRATION_ALLOWED_SIGNATURES",
    "ONLINE_FALLBACK_RUN_ID",
    "R5IntegrationRunner",
    "R5CandidateResponse",
    "R5PackageError",
    "R5CurrentStateBundle",
    "R5_PACKAGE_RELATIVE",
    "R5_PACKAGE_RUN_ID",
    "MOOD_SOCIAL_R5_CANDIDATE_INFER_PATH",
    "ROUTE_BLOCKED_SIGNATURES",
    "build_r5_fullfit_package",
    "infer_mood_social_r5_candidate",
    "load_r5_fullfit_package",
    "runtime_signature",
]
