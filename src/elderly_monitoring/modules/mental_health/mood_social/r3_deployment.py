"""Train, package, verify, and run the promoted V3.3.3-r3 model.

This module is intentionally separate from the frozen formal evaluator.  A
package can be built only after OPT-V333-007 reports that every metric gate
passed; the active package selector is changed only after API regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from elderly_monitoring.modules.mental_health.mood_social.config import (
    MoodSocialConfig,
    load_mood_social_config,
)
from elderly_monitoring.modules.mental_health.mood_social.feature_mapper import (
    map_mood_social_features,
)
from elderly_monitoring.modules.mental_health.mood_social.pipeline import (
    _attention_trend,
    _build_expert_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.baseline import (
    BASE_WEIGHT,
    LEGACY_EXPERTS,
    _ProbabilityCalibrator,
    _FusionCalibration,
    _crossfit_expert_calibration,
    _expert_raw_oof,
    _fit_calibrator as _fit_legacy_calibrator,
    _fit_fusion as _fit_legacy_fusion,
    _fit_fusion_calibration as _fit_legacy_fusion_calibration,
    _fusion_probability as _legacy_fusion_probability,
    _observed_fraction as _legacy_observed_fraction,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.calibration import (
    FittedCalibrator,
    crossfit_calibrator,
    fit_calibrator,
    select_workpoints,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.contract import (
    ACTIVITY_RISK_FEATURES,
    PROFILE_RISK_FEATURES,
    R3_PROTOCOL_VERSION,
    SLEEP_RISK_FEATURES,
    assert_deployable_feature_names,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.data import (
    eligible_rows,
    load_r3_training_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.fusion import (
    EXPERT_IDS,
    FusionSpec,
    PARTICIPANT_WEIGHT,
    _direct_probability,
    _fit_residual,
    _logit,
    _merge_meta_features,
    _stack_matrix,
    candidate_expert_train_and_predict,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.modeling import (
    ModelSpec,
    WeightSpec,
    binary_metrics,
    fit_classifier,
    predict_probability,
    sample_weights,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.selection import (
    EXPERT_DEFINITIONS,
)
from elderly_monitoring.modules.mental_health.mood_social.schemas import (
    MoodSocialInferRequest,
    MoodSocialInferResponse,
)


R3_MODEL_VERSION = "mood-fusion-v3.3.3-r3"
R3_PACKAGE_VERSION = "mood-social-v3.3.3-r3-package-v1"
DEFAULT_RUN_ID = "MH-20260811-R3-001"
DEFAULT_FORMAL_REPORT = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-007-formal-outer/formal_evaluation.json"
)
DEFAULT_FORMAL_MANIFEST = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-007-formal-outer/artifact_manifest.json"
)
DEFAULT_EXPERT_SELECTION = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-003-004-selection/selected_experts.json"
)
DEFAULT_FUSION_SELECTION = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-005-fusion/selected_fusions.json"
)
DEFAULT_CALIBRATION_SELECTION = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-006-calibration/calibration_and_workpoints.json"
)
DEFAULT_BASELINE_OOF = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-002-baseline-replay/strict_baseline_oof.parquet"
)


class R3PackageError(RuntimeError):
    """Raised when a promoted package is missing, changed, or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise R3PackageError(f"JSON object required: {path.name}")
    return value


def _model_spec(payload: Mapping[str, Any]) -> ModelSpec:
    return ModelSpec(
        candidate_id=str(payload["candidate_id"]),
        family=str(payload["family"]),  # type: ignore[arg-type]
        params=dict(payload["params"]),
        seed=int(payload.get("seed", 20260728)),
    )


def _modal_expert_choices(selection: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    outer_choices = selection["outer_choices"]
    for expert_id in EXPERT_IDS:
        values = [dict(item[expert_id]) for item in outer_choices.values()]
        candidates: list[tuple[int, float, str, dict[str, Any]]] = []
        for candidate_id in sorted({str(item["candidate_id"]) for item in values}):
            matching = [item for item in values if str(item["candidate_id"]) == candidate_id]
            score = float(
                np.mean(
                    [float(item["inner_oof_metrics"]["selection_score"]) for item in matching]
                )
            )
            candidates.append((len(matching), score, candidate_id, matching[0]))
        result[expert_id] = max(
            candidates, key=lambda item: (item[0], item[1], item[2])
        )[3]
    return result


def _modal_fusion_spec(selection: Mapping[str, Any]) -> FusionSpec:
    values = [dict(item["selected_spec"]) for item in selection["outer_choices"].values()]
    counts: dict[str, int] = {}
    for value in values:
        candidate_id = str(value["candidate_id"])
        counts[candidate_id] = counts.get(candidate_id, 0) + 1
    candidate_id = max(sorted(counts), key=lambda item: (counts[item], item))
    selected = next(value for value in values if str(value["candidate_id"]) == candidate_id)
    return FusionSpec(candidate_id, str(selected["kind"]), dict(selected["params"]))


def _modal_calibration_method(selection: Mapping[str, Any]) -> str:
    values = [str(item["selected_method"]) for item in selection["outer_choices"].values()]
    return max(sorted(set(values)), key=lambda item: (values.count(item), item))


@dataclass
class R3ExpertRuntime:
    expert_id: str
    features: tuple[str, ...]
    routes: tuple[str, ...]
    models: tuple[Any, ...]
    blend_kind: str
    rank_references: tuple[np.ndarray, ...]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        result = np.full(len(frame), np.nan, dtype=float)
        selected = frame["route_pattern"].astype(str).isin(self.routes).to_numpy()
        if not selected.any():
            return result
        values = np.vstack(
            [predict_probability(model, frame.loc[selected], self.features) for model in self.models]
        )
        if self.blend_kind == "probability":
            probability = values.mean(axis=0)
        elif self.blend_kind == "raw_logit":
            probability = 1.0 / (
                1.0 + np.exp(-np.clip(np.mean(_logit(values), axis=0), -30.0, 30.0))
            )
        elif self.blend_kind == "rank":
            ranked = np.vstack(
                [
                    np.searchsorted(reference, member, side="right") / float(len(reference))
                    for reference, member in zip(self.rank_references, values, strict=True)
                ]
            )
            probability = ranked.mean(axis=0)
        else:
            raise R3PackageError(f"unsupported expert blend: {self.blend_kind}")
        result[selected] = np.clip(probability, 1.0e-6, 1.0 - 1.0e-6)
        return result


@dataclass
class LegacyRuntime:
    expert_models: Mapping[str, Any]
    expert_calibrators: Mapping[str, _ProbabilityCalibrator]
    fusion_model: LogisticRegression
    fusion_calibration: _FusionCalibration

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        features = frame[
            ["route_pattern"]
        ].copy()
        for definition in LEGACY_EXPERTS:
            available = frame["route_pattern"].astype(str).isin(definition.routes).to_numpy()
            raw = np.full(len(frame), np.nan, dtype=float)
            calibrated = np.full(len(frame), np.nan, dtype=float)
            if available.any():
                raw[available] = predict_probability(
                    self.expert_models[definition.expert_id],
                    frame.loc[available],
                    definition.features,
                )
                calibrated[available] = self.expert_calibrators[
                    definition.expert_id
                ].predict(raw[available])
            features[f"{definition.expert_id}_raw"] = raw
            features[f"{definition.expert_id}_probability"] = calibrated
            features[f"{definition.expert_id}_confidence"] = np.where(
                available,
                definition.reliability * _legacy_observed_fraction(frame, definition),
                0.0,
            )
        raw_fusion = _legacy_fusion_probability(self.fusion_model, features)
        return self.fusion_calibration.predict(raw_fusion, frame["route_pattern"])


@dataclass
class CandidateFusionRuntime:
    spec: FusionSpec
    residual_model: Any | None = None
    stack_model: LogisticRegression | None = None

    @classmethod
    def fit(cls, spec: FusionSpec, train: pd.DataFrame) -> "CandidateFusionRuntime":
        direct = _direct_probability(train, float(spec.params.get("profile_weight", 0.2)))
        if spec.kind == "residual":
            return cls(
                spec,
                residual_model=_fit_residual(train, direct, float(spec.params["l2"])),
            )
        if spec.kind == "stack":
            model = LogisticRegression(
                solver="liblinear",
                C=float(spec.params["C"]),
                max_iter=4000,
                random_state=20260728,
            )
            model.fit(
                _stack_matrix(train),
                train["binary_target"].to_numpy(int),
                sample_weight=sample_weights(train, PARTICIPANT_WEIGHT),
            )
            return cls(spec, stack_model=model)
        return cls(spec)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        direct = _direct_probability(
            frame, float(self.spec.params.get("profile_weight", 0.2))
        )
        if self.spec.kind == "direct":
            return direct
        if self.spec.kind in {"convex_probability", "convex_logit"}:
            old_weight = float(self.spec.params["old_weight"])
            old = frame["baseline_probability"].to_numpy(float)
            if self.spec.kind == "convex_probability":
                result = old_weight * old + (1.0 - old_weight) * direct
            else:
                value = old_weight * _logit(old) + (1.0 - old_weight) * _logit(direct)
                result = 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))
            return np.clip(result, 1.0e-6, 1.0 - 1.0e-6)
        if self.spec.kind == "residual" and self.residual_model is not None:
            return self.residual_model.predict(
                frame["baseline_probability"].to_numpy(float), direct
            )
        if self.spec.kind == "stack" and self.stack_model is not None:
            return np.clip(
                self.stack_model.predict_proba(_stack_matrix(frame))[:, 1],
                1.0e-6,
                1.0 - 1.0e-6,
            )
        raise R3PackageError("fitted fusion runtime is incomplete")


@dataclass
class R3RuntimeBundle:
    model_version: str
    run_id: str
    experts: Mapping[str, R3ExpertRuntime]
    legacy: LegacyRuntime
    fusion: CandidateFusionRuntime
    calibrator: FittedCalibrator
    workpoints: Mapping[str, Any]

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        route = frame["route_pattern"].astype(str)
        if route.eq("000").any():
            raise R3PackageError("no-evidence rows must abstain before model inference")
        meta = frame[["route_pattern"]].copy()
        for expert_id in EXPERT_IDS:
            meta[f"expert_{expert_id}_probability"] = self.experts[expert_id].predict(frame)
        meta["baseline_probability"] = self.legacy.predict(frame)
        uncalibrated = self.fusion.predict(meta)
        calibration_input = meta.copy()
        calibration_input["probability"] = uncalibrated
        probability = self.calibrator.predict(calibration_input)
        result = pd.DataFrame(
            {
                "route_pattern": route.to_numpy(),
                "baseline_probability": meta["baseline_probability"].to_numpy(float),
                "uncalibrated_probability": uncalibrated,
                "probability": probability,
            }
        )
        for expert_id in EXPERT_IDS:
            result[f"expert_{expert_id}_probability"] = meta[
                f"expert_{expert_id}_probability"
            ].to_numpy(float)
        return result


def _fit_expert_runtimes(
    frame: pd.DataFrame, choices: Mapping[str, Mapping[str, Any]]
) -> dict[str, R3ExpertRuntime]:
    result: dict[str, R3ExpertRuntime] = {}
    for definition in EXPERT_DEFINITIONS:
        choice = choices[definition.expert_id]
        members = tuple(_model_spec(value) for value in choice["members"])
        weight = WeightSpec(**dict(choice["weight_spec"]))
        train = frame.loc[frame["route_pattern"].isin(definition.required_routes)].copy()
        models = tuple(
            fit_classifier(train, definition.features, member, weight) for member in members
        )
        blend = str(choice.get("blend_kind", "probability"))
        references = (
            tuple(
                np.sort(predict_probability(model, train, definition.features), kind="stable")
                for model in models
            )
            if blend == "rank"
            else ()
        )
        result[definition.expert_id] = R3ExpertRuntime(
            definition.expert_id,
            assert_deployable_feature_names(definition.features),
            definition.required_routes,
            models,
            blend,
            references,
        )
    return result


def _fit_legacy_runtime(
    frame: pd.DataFrame, baseline_oof: pd.DataFrame
) -> LegacyRuntime:
    fold = frame["outer_fold"].astype(int)
    raw_oof = _expert_raw_oof(frame, fold)
    fusion_features = _crossfit_expert_calibration(raw_oof)
    expert_models: dict[str, Any] = {}
    expert_calibrators: dict[str, _ProbabilityCalibrator] = {}
    for definition in LEGACY_EXPERTS:
        train = frame.loc[frame["route_pattern"].isin(definition.routes)].copy()
        expert_models[definition.expert_id] = fit_classifier(
            train, definition.features, definition.model_spec, BASE_WEIGHT
        )
        expert_calibrators[definition.expert_id] = _fit_legacy_calibrator(
            raw_oof, f"{definition.expert_id}_raw"
        )
    calibration_frame = baseline_oof.rename(
        columns={"baseline_raw_probability": "fusion_raw_probability"}
    )
    return LegacyRuntime(
        expert_models,
        expert_calibrators,
        _fit_legacy_fusion(fusion_features),
        _fit_legacy_fusion_calibration(calibration_frame),
    )


def train_r3_runtime_bundle(
    *, repository_root: Path, run_id: str = DEFAULT_RUN_ID
) -> tuple[R3RuntimeBundle, dict[str, Any]]:
    root = Path(repository_root).resolve()
    frame = eligible_rows(load_r3_training_frame(repository_root=root))
    expert_selection = _read_json(root / DEFAULT_EXPERT_SELECTION)
    fusion_selection = _read_json(root / DEFAULT_FUSION_SELECTION)
    calibration_selection = _read_json(root / DEFAULT_CALIBRATION_SELECTION)
    choices = _modal_expert_choices(expert_selection)
    fusion_spec = _modal_fusion_spec(fusion_selection)
    calibration_method = _modal_calibration_method(calibration_selection)
    baseline_oof = pd.read_parquet(root / DEFAULT_BASELINE_OOF)

    empty = frame.iloc[:0].copy()
    new_oof, _ = candidate_expert_train_and_predict(
        frame, frame["outer_fold"].astype(int), empty, choices
    )
    old = baseline_oof[["r3_row_id", "baseline_probability"]]
    meta = _merge_meta_features(new_oof, old)
    raw_oof = np.full(len(meta), np.nan, dtype=float)
    fold = meta["fusion_validation_fold"].to_numpy(int)
    for validation_fold in sorted(np.unique(fold)):
        runtime = CandidateFusionRuntime.fit(fusion_spec, meta.loc[fold != validation_fold])
        raw_oof[fold == validation_fold] = runtime.predict(meta.loc[fold == validation_fold])
    if not np.isfinite(raw_oof).all():
        raise R3PackageError("production fusion OOF is incomplete")
    calibration_frame = meta.copy()
    calibration_frame["probability"] = raw_oof
    calibrated_oof, calibration_audit = crossfit_calibrator(
        calibration_frame, calibration_method
    )
    workpoints = select_workpoints(
        calibration_frame["binary_target"], calibrated_oof
    )
    experts = _fit_expert_runtimes(frame, choices)
    legacy = _fit_legacy_runtime(frame, baseline_oof)
    bundle = R3RuntimeBundle(
        R3_MODEL_VERSION,
        run_id,
        experts,
        legacy,
        CandidateFusionRuntime.fit(fusion_spec, meta),
        fit_calibrator(calibration_frame, calibration_method),
        workpoints,
    )
    audit = {
        "row_count": int(len(frame)),
        "participant_count": int(frame["global_participant_id"].nunique()),
        "expert_candidate_ids": {
            key: str(value["candidate_id"]) for key, value in choices.items()
        },
        "fusion_spec": {
            "candidate_id": fusion_spec.candidate_id,
            "kind": fusion_spec.kind,
            "params": dict(fusion_spec.params),
        },
        "calibration_method": calibration_method,
        "calibration_crossfit_audit": calibration_audit,
        "production_oof_metrics": binary_metrics(
            calibration_frame["binary_target"], calibrated_oof
        ),
        "workpoints": workpoints,
    }
    return bundle, audit


@dataclass(frozen=True)
class R3ModelPackage:
    root: Path
    manifest: Mapping[str, Any]
    bundle: R3RuntimeBundle


def load_r3_model_package(package_directory: str | Path) -> R3ModelPackage:
    root = Path(package_directory).resolve()
    expected = {"manifest.json", "model.joblib", "SHA256SUMS"}
    actual = {path.name for path in root.iterdir() if path.is_file()} if root.is_dir() else set()
    if actual != expected:
        raise R3PackageError("r3 package file set is incomplete or contains extras")
    checksums: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    if set(checksums) != {"manifest.json", "model.joblib"}:
        raise R3PackageError("r3 checksum manifest has the wrong file set")
    for name, digest in checksums.items():
        if _sha256_file(root / name) != digest:
            raise R3PackageError(f"r3 package checksum mismatch: {name}")
    manifest = _read_json(root / "manifest.json")
    if (
        manifest.get("package_version") != R3_PACKAGE_VERSION
        or manifest.get("model_version") != R3_MODEL_VERSION
        or manifest.get("protocol_version") != R3_PROTOCOL_VERSION
    ):
        raise R3PackageError("r3 package identity mismatch")
    bundle = joblib.load(root / "model.joblib")
    if not isinstance(bundle, R3RuntimeBundle):
        raise R3PackageError("r3 model payload type mismatch")
    if bundle.run_id != manifest.get("run_id") or bundle.model_version != R3_MODEL_VERSION:
        raise R3PackageError("r3 model payload disagrees with manifest")
    return R3ModelPackage(root, manifest, bundle)


def build_r3_model_package(
    *, repository_root: Path, run_id: str = DEFAULT_RUN_ID
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    formal_report = _read_json(root / DEFAULT_FORMAL_REPORT)
    formal_manifest = _read_json(root / DEFAULT_FORMAL_MANIFEST)
    if formal_report.get("metric_gate_pass") is not True:
        raise R3PackageError("formal r3 metric gates did not pass; package creation forbidden")
    if formal_manifest.get("metric_gate_pass") is not True:
        raise R3PackageError("formal r3 manifest does not authorize package creation")
    destination = (
        root
        / "models/mental_health/mood_social/v3.3.3-r3/candidates"
        / run_id
    )
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite r3 candidate package: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle, training_audit = train_r3_runtime_bundle(
        repository_root=root, run_id=run_id
    )
    stage = Path(tempfile.mkdtemp(prefix=f"{run_id}-", dir=destination.parent))
    try:
        joblib.dump(bundle, stage / "model.joblib", compress=3)
        manifest = {
            "package_version": R3_PACKAGE_VERSION,
            "protocol_version": R3_PROTOCOL_VERSION,
            "model_version": R3_MODEL_VERSION,
            "run_id": run_id,
            "status": "candidate_pending_API_regression",
            "network_access_required": False,
            "no_evidence_policy": "abstain_without_probability",
            "formal_report": {
                "path": DEFAULT_FORMAL_REPORT.as_posix(),
                "sha256": _sha256_file(root / DEFAULT_FORMAL_REPORT),
            },
            "formal_manifest": {
                "path": DEFAULT_FORMAL_MANIFEST.as_posix(),
                "sha256": _sha256_file(root / DEFAULT_FORMAL_MANIFEST),
            },
            "split_manifest_sha256": _sha256_file(
                root
                / "data/processed/mental_health/mood_social/v3.3.3-r2/splits/split_manifest.json"
            ),
            "training_config_sha256": _sha256_file(
                root / "configs/training/mood_social_v3_3_3_r3.yaml"
            ),
            "training_audit": training_audit,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        checksum_lines = [
            f"{_sha256_file(stage / name)}  {name}"
            for name in ("manifest.json", "model.joblib")
        ]
        (stage / "SHA256SUMS").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )
        stage.replace(destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    loaded = load_r3_model_package(destination)
    return {
        "status": "pass",
        "run_id": run_id,
        "model_version": R3_MODEL_VERSION,
        "package_directory": destination.relative_to(root).as_posix(),
        "manifest_sha256": _sha256_file(destination / "manifest.json"),
        "model_sha256": _sha256_file(destination / "model.joblib"),
        "checksums_sha256": _sha256_file(destination / "SHA256SUMS"),
        "training_audit": loaded.manifest["training_audit"],
    }


def _route_pattern(frame: pd.DataFrame) -> str:
    def observed(features: Sequence[str]) -> bool:
        return any(int(frame.iloc[0].get(f"feature_mask.{name}", 0) or 0) == 1 for name in features)

    return "".join(
        "1" if value else "0"
        for value in (
            observed(ACTIVITY_RISK_FEATURES),
            observed(SLEEP_RISK_FEATURES),
            observed(PROFILE_RISK_FEATURES),
        )
    )


@dataclass(frozen=True)
class R3MoodSocialPipeline:
    package: R3ModelPackage
    config: MoodSocialConfig

    @classmethod
    def from_package(
        cls, package_directory: str | Path, *, config: MoodSocialConfig | None = None
    ) -> "R3MoodSocialPipeline":
        return cls(load_r3_model_package(package_directory), config or load_mood_social_config())

    def predict(
        self, request: MoodSocialInferRequest | Mapping[str, Any]
    ) -> MoodSocialInferResponse:
        parsed = (
            request
            if isinstance(request, MoodSocialInferRequest)
            else MoodSocialInferRequest.model_validate(request)
        )
        mapped = map_mood_social_features(parsed, config=self.config)
        frame = _build_expert_frame(mapped)
        route = _route_pattern(frame)
        if route == "000":
            return MoodSocialInferResponse.model_validate(
                {
                    "schema_version": "mood_social_infer_response_v3",
                    "request_id": parsed.request_id,
                    "person_id": parsed.person_id,
                    "target_date": parsed.target_date,
                    "module": "mood_social_attention",
                    "model_version": R3_MODEL_VERSION,
                    "available": False,
                    "evidence_scope": "insufficient_data",
                    "attention_index": None,
                    "attention_score": None,
                    "attention_level": None,
                    "confidence": 0.0,
                    "used_sources": [],
                    "covered_domains": [],
                    "domain_scores": None,
                    "model_contributions": [],
                    "trend": "unknown",
                    "summary": "当前未获得可用的情绪与社交关注证据。",
                    "limitations": ["insufficient_data", "r3_no_production_csp_evidence"],
                    "diagnosis": False,
                }
            )
        frame["route_pattern"] = route
        prediction = self.package.bundle.predict_frame(frame).iloc[0]
        probability = float(prediction["probability"])
        covered_domains: list[str] = []
        if route[0] == "1":
            covered_domains.append("activity")
        if route[1] == "1":
            covered_domains.append("sleep")
        if route[:2] == "11":
            covered_domains.append("activity_sleep_joint")
        if route[2] == "1":
            covered_domains.append("social_context")
        used_sources = [
            source
            for source, present in (
                ("camera", route[0] == "1"),
                ("sleep_device", route[1] == "1"),
                ("profile", route[2] == "1"),
            )
            if present
        ]
        domain_scores = {
            "activity": (
                float(prediction["expert_activity_probability"])
                if route[0] == "1"
                else None
            ),
            "sleep": (
                float(prediction["expert_sleep_probability"])
                if route[1] == "1"
                else None
            ),
            "physiology": None,
            "activity_sleep_joint": (
                float(prediction["expert_joint_probability"])
                if route[:2] == "11"
                else None
            ),
            "social_context": (
                float(prediction["expert_profile_probability"])
                if route[2] == "1"
                else None
            ),
            "personal_change": {"activity": None, "sleep": None, "social": None},
        }
        score = int(
            (Decimal(str(probability)) * Decimal("100")).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
        )
        level = 0 if probability < 0.25 else 1 if probability < 0.45 else 2 if probability < 0.65 else 3
        return MoodSocialInferResponse.model_validate(
            {
                "schema_version": "mood_social_infer_response_v3",
                "request_id": parsed.request_id,
                "person_id": parsed.person_id,
                "target_date": parsed.target_date,
                "module": "mood_social_attention",
                "model_version": R3_MODEL_VERSION,
                "available": True,
                "evidence_scope": "proxy_label_supported",
                "attention_index": probability,
                "attention_score": score,
                "attention_level": level,
                "confidence": 1.0,
                "used_sources": used_sources,
                "covered_domains": covered_domains,
                "domain_scores": domain_scores,
                "model_contributions": [],
                "trend": _attention_trend(parsed, probability, self.config),
                "summary": "当前模型结果可用，暂未识别出明确的主要关注因素，建议继续观察近期趋势。",
                "limitations": [
                    "仅用于情绪低落与社交退缩关注趋势，不构成医学诊断。",
                    "概率仅在存在冻结的 C/S/P 生产证据时输出。",
                ],
                "diagnosis": False,
            }
        )


__all__ = [
    "DEFAULT_RUN_ID",
    "R3_MODEL_VERSION",
    "R3MoodSocialPipeline",
    "R3PackageError",
    "build_r3_model_package",
    "load_r3_model_package",
    "train_r3_runtime_bundle",
]
