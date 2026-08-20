"""Strict OOF fusion candidates for the V3.3.4 optimization chain.

The implementation deliberately keeps V3.3.3 artifacts immutable.  It adapts
the frozen FUSION-001 row table with the selected OOF outputs from
OPT-CALIB-001, OPT-EXPERT-001 and OPT-TREND-001, then delegates the audited
low-order candidate fitting protocol to the V3.3.3 deployment optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml

from .deployment_optimization import (
    CANDIDATE_IDS,
    OUTER_FOLDS,
    RANDOM_SEED,
    _metrics,
    _nested_search,
    _source_stability,
    _write_json,
    _write_parquet,
)
from .calibration_optimization import _ece
from .masked_stacking import (
    INPUT_FEATURES,
    effective_evidence,
    evidence_combination,
)


TASK_ID = "OPT-FUSION-003"
RUN_ID = "MH-20260804-021"
TRAINING_VERSION = "mood-social-fusion-optimization-v3.3.4-v1"
MODEL_VERSION = "mood-social-fusion-candidate-v3.3.4-v1"
MANIFEST_VERSION = "mood-social-fusion-optimization-v3.3.4-manifest-v1"
REPORT_VERSION = "mood-social-fusion-optimization-v3.3.4-report-v1"
EPSILON = 1.0e-6


class FusionV334OptimizationError(RuntimeError):
    """Raised when the frozen V3.3.4 fusion boundary is violated."""


@dataclass(frozen=True)
class V334FusionModel:
    """Versioned wrapper around the audited source-independent fusion model."""

    task_id: str
    run_id: str
    model_version: str
    candidate_id: str
    inner_model: Any

    def validate(self) -> None:
        if self.task_id != TASK_ID or self.run_id != RUN_ID:
            raise FusionV334OptimizationError("V3.3.4 model identity changed")
        if (
            self.model_version != MODEL_VERSION
            or self.candidate_id not in CANDIDATE_IDS
        ):
            raise FusionV334OptimizationError("V3.3.4 model version changed")
        if not hasattr(self.inner_model, "predict_frame"):
            raise FusionV334OptimizationError("inner fusion model is invalid")

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.validate()
        result = self.inner_model.predict_frame(frame).copy()
        result["model_version"] = self.model_version
        result["candidate_id"] = self.candidate_id
        return result


@dataclass(frozen=True)
class FusionV334Config:
    repository_root: Path
    payload: Mapping[str, Any]
    config_path: Path

    @property
    def report_directory(self) -> Path:
        return _resolve(
            self.repository_root, self.payload["output"]["report_directory"]
        )

    @property
    def model_path(self) -> Path:
        return _resolve(self.repository_root, self.payload["output"]["model_path"])

    @property
    def manifest_path(self) -> Path:
        return _resolve(self.repository_root, self.payload["output"]["manifest_path"])


def load_v334_config(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> FusionV334Config:
    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise FusionV334OptimizationError(
            "OPT-FUSION-003 config is unreadable"
        ) from exc
    if not isinstance(payload, Mapping):
        raise FusionV334OptimizationError("OPT-FUSION-003 config must be a mapping")
    if payload.get("task_id") != TASK_ID or payload.get("run_id") != RUN_ID:
        raise FusionV334OptimizationError("OPT-FUSION-003 task identity changed")
    if int(payload.get("random_seed", -1)) != RANDOM_SEED:
        raise FusionV334OptimizationError("OPT-FUSION-003 random seed changed")
    if tuple(payload.get("candidates", {}).get("ids", ())) != CANDIDATE_IDS:
        raise FusionV334OptimizationError("candidate grid changed")
    boundary = payload.get("production_boundary", {})
    required_false = (
        "consume_public_dataset_id",
        "select_production_threshold",
        "change_active_experts",
        "change_personal_trend",
        "include_model006_predictions",
        "change_http_behavior",
        "change_attention_levels",
        "overwrite_existing_models",
    )
    if any(boundary.get(key) is not False for key in required_false):
        raise FusionV334OptimizationError("production boundary changed")
    root = Path(repository_root or config_path.parents[2]).resolve()
    return FusionV334Config(root, payload, config_path)


def optimize_v334_fusion(
    config: FusionV334Config,
    *,
    overwrite: bool = False,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    frame, assignments, protection = load_v334_inputs(config)
    print(f"[{TASK_ID}] inputs aligned: {len(frame)} rows", flush=True)
    _validate_outputs(config, overwrite=overwrite)
    stage = config.report_directory.with_name(f".{config.report_directory.name}.stage")
    stage_meta = stage / "stage.json"
    if stage_meta.is_file():
        metadata = json.loads(stage_meta.read_text(encoding="utf-8"))
        if metadata.get("input_hashes") != {
            "fusion": protection["fusion_oof_table_sha256"],
            "baseline": protection["baseline_oof_sha256"],
            "calibration": protection["calibration_oof_sha256"],
            "expert": protection["expert_oof_sha256"],
            "trend": protection["trend_oof_sha256"],
        }:
            raise FusionV334OptimizationError("nested search checkpoint input drifted")
        search = pd.read_parquet(stage / "candidate_search.parquet")
        oof = pd.read_parquet(stage / "oof_predictions.parquet")
        selected_id = str(metadata["selected_candidate_id"])
        inner_model = joblib.load(stage / "inner_model.joblib")
        print(f"[{TASK_ID}] reused nested search checkpoint: {selected_id}", flush=True)
    else:
        search, oof, selected_id, inner_model = _nested_search(frame, assignments)
        stage.mkdir(parents=True, exist_ok=True)
        _write_parquet(search, stage / "candidate_search.parquet")
        _write_parquet(oof, stage / "oof_predictions.parquet")
        joblib.dump(inner_model, stage / "inner_model.joblib", compress=0, protocol=4)
        _write_json(
            stage_meta,
            {
                "task_id": TASK_ID,
                "run_id": RUN_ID,
                "selected_candidate_id": selected_id,
                "input_hashes": {
                    "fusion": protection["fusion_oof_table_sha256"],
                    "baseline": protection["baseline_oof_sha256"],
                    "calibration": protection["calibration_oof_sha256"],
                    "expert": protection["expert_oof_sha256"],
                    "trend": protection["trend_oof_sha256"],
                },
            },
        )
        print(f"[{TASK_ID}] nested search complete: {selected_id}", flush=True)
    calibration = _metrics(
        frame["binary_target"].to_numpy(dtype=int),
        frame["calibration_probability"].to_numpy(dtype=float),
    )
    calibration["ece"] = _ece(
        frame["binary_target"].to_numpy(dtype=int),
        frame["calibration_probability"].to_numpy(dtype=float),
    )
    overall = _overall_metrics(frame, oof, calibration)
    outer = _outer_metrics(frame, oof)
    source = _source_stability(frame, selected_id)
    print(f"[{TASK_ID}] leave-one-source evaluation complete", flush=True)
    device = _group_metrics(frame, oof, "device_combination")
    missing = _group_metrics(frame, oof, "mask_pattern")
    model = V334FusionModel(TASK_ID, RUN_ID, MODEL_VERSION, selected_id, inner_model)
    model.validate()
    ablation = _weak_branch_ablation(frame, model)
    promotion = _promotion_decision(config, overall, outer, source)
    print(
        f"[{TASK_ID}] promotion gate: {promotion['promotion_passed']}",
        flush=True,
    )
    summary = {
        "status": "pass",
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "selected_candidate_id": selected_id,
        "promotion_passed": bool(promotion["promotion_passed"]),
        "package_default": promotion["package_default"],
        "baseline_metrics": _row_dict(overall, "current_online_baseline"),
        "candidate_metrics": _row_dict(overall, "v334_candidate"),
        "corrected_calibration_metrics": _row_dict(
            overall, "corrected_calibration_reference"
        ),
        "new_expert_oof": protection["new_expert_oof"],
        "new_trend_oof": protection["new_trend_oof"],
        "social_evidence_scope": "engineering_proxy_no_direct_s10_phq9_validation",
        "strict_oof": True,
    }
    _publish(
        config,
        frame,
        search,
        oof,
        overall,
        outer,
        source,
        device,
        missing,
        ablation,
        promotion,
        summary,
        model,
        protection,
        command or (),
    )
    if stage.exists():
        shutil.rmtree(stage)
    print(f"[{TASK_ID}] artifacts published", flush=True)
    return {
        "status": "pass",
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "selected_candidate_id": selected_id,
        "promotion_passed": bool(promotion["promotion_passed"]),
        "model_path": str(config.model_path),
        "report_directory": str(config.report_directory),
    }


def load_v334_inputs(
    config: FusionV334Config,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    inputs = config.payload["input"]
    protected_paths = {
        _resolve(config.repository_root, inputs["fusion_oof_table_path"]): inputs[
            "fusion_oof_table_sha256"
        ],
        _resolve(config.repository_root, inputs["baseline_oof_path"]): inputs[
            "baseline_oof_sha256"
        ],
        _resolve(config.repository_root, inputs["calibration_oof_path"]): inputs[
            "calibration_oof_sha256"
        ],
        _resolve(config.repository_root, inputs["expert_oof_path"]): inputs[
            "expert_oof_sha256"
        ],
        _resolve(config.repository_root, inputs["trend_oof_path"]): inputs[
            "trend_oof_sha256"
        ],
        _resolve(config.repository_root, inputs["split"]["path"]): inputs["split"][
            "sha256"
        ],
    }
    for path, expected in protected_paths.items():
        if not path.is_file() or _sha256_file(path) != str(expected):
            raise FusionV334OptimizationError(f"protected input drifted: {path}")

    table_path = _resolve(config.repository_root, inputs["fusion_oof_table_path"])
    frame = pd.read_parquet(table_path).reset_index(drop=True)
    if len(frame) != 22191 or frame["global_participant_id"].nunique() != 15361:
        raise FusionV334OptimizationError("FUSION-001 table dimensions changed")
    keys = ["dataset_id", "canonical_row_index"]

    baseline = pd.read_parquet(
        _resolve(config.repository_root, inputs["baseline_oof_path"])
    )
    baseline_column = str(inputs.get("baseline_probability_column", ""))
    if baseline_column != "deployment_candidate_probability":
        raise FusionV334OptimizationError(
            "current online baseline must bind deployment_candidate_probability"
        )
    frame = _merge_one(
        frame,
        baseline,
        keys,
        baseline_column,
        "baseline_probability",
    )
    calibration = pd.read_parquet(
        _resolve(config.repository_root, inputs["calibration_oof_path"])
    )
    frame = _merge_one(
        frame,
        calibration,
        keys,
        "probability__corrected_coverage_platt",
        "calibration_probability",
    )
    # The shared audited search publishes this compatibility column.  Its
    # V3.3.4 meaning is the corrected OPT-CALIB-001 strict OOF reference.
    frame["offline_reference_probability"] = frame["calibration_probability"]
    experts = pd.read_parquet(
        _resolve(config.repository_root, inputs["expert_oof_path"])
    )
    _merge_expert_candidates(frame, experts, keys)
    trends = pd.read_parquet(_resolve(config.repository_root, inputs["trend_oof_path"]))
    _merge_trend_candidates(frame, trends, keys)

    if frame["baseline_probability"].notna().sum() != 22188:
        raise FusionV334OptimizationError("current online OOF alignment changed")
    if frame["calibration_probability"].notna().sum() != 22188:
        raise FusionV334OptimizationError("calibration OOF alignment changed")

    available = effective_evidence(frame) > 0.0
    frame = frame.loc[available].reset_index(drop=True)
    if len(frame) != 22188 or frame["binary_target"].nunique() != 2:
        raise FusionV334OptimizationError("V3.3.4 available OOF dimensions changed")
    split_path = _resolve(config.repository_root, inputs["split"]["path"])
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    assignments = pd.DataFrame(split_payload.get("participant_assignments", []))
    required = {
        "global_participant_id",
        "dataset_id",
        "outer_fold",
        "inner_validation_fold_by_outer_fold",
    }
    if assignments.empty or not required.issubset(assignments.columns):
        raise FusionV334OptimizationError("DATA-007 assignments are incomplete")
    if assignments["global_participant_id"].duplicated().any():
        raise FusionV334OptimizationError("DATA-007 assignments are duplicated")
    assignments["global_participant_id"] = assignments["global_participant_id"].astype(
        str
    )
    assignments = assignments.set_index("global_participant_id")
    if set(frame["global_participant_id"].astype(str)) - set(assignments.index):
        raise FusionV334OptimizationError("DATA-007 participant coverage changed")
    protection = {
        "fusion_oof_table_sha256": str(inputs["fusion_oof_table_sha256"]),
        "baseline_oof_sha256": str(inputs["baseline_oof_sha256"]),
        "baseline_probability_column": baseline_column,
        "calibration_oof_sha256": str(inputs["calibration_oof_sha256"]),
        "expert_oof_sha256": str(inputs["expert_oof_sha256"]),
        "trend_oof_sha256": str(inputs["trend_oof_sha256"]),
        "split_sha256": str(inputs["split"]["sha256"]),
        "public_dataset_id_as_model_input": False,
        "model006_enters_fusion": False,
        "new_expert_oof": {
            "run_id": "MH-20260804-015",
            "selected": "selected_probability",
        },
        "new_trend_oof": {
            "run_id": "MH-20260804-016",
            "selected": "selected_probability",
        },
        "calibration_candidate": {
            "run_id": "MH-20260804-014",
            "selected": "probability__corrected_coverage_platt",
        },
    }
    return frame, assignments, protection


def _merge_one(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    keys: list[str],
    source_column: str,
    output_column: str,
) -> pd.DataFrame:
    if source[keys].duplicated().any() or source_column not in source:
        raise FusionV334OptimizationError(f"invalid OOF source column: {source_column}")
    values = source[keys + [source_column]].rename(
        columns={source_column: output_column}
    )
    return frame.merge(values, on=keys, how="left", validate="one_to_one")


def _merge_expert_candidates(
    frame: pd.DataFrame, source: pd.DataFrame, keys: list[str]
) -> None:
    for expert in ("activity", "sleep", "joint"):
        rows = source.loc[source["expert"].eq(expert)].copy()
        if rows[keys].duplicated().any():
            raise FusionV334OptimizationError(f"duplicate {expert} expert OOF rows")
        values = rows[keys + ["selected_probability", "expert_mask"]].rename(
            columns={
                "selected_probability": "_new_probability",
                "expert_mask": "_new_mask",
            }
        )
        merged = frame.merge(values, on=keys, how="left", validate="one_to_one")
        frame["expert_" + expert + "_current_probability"] = merged["_new_probability"]
        frame["expert_" + expert + "_expert_mask"] = (
            pd.to_numeric(merged["_new_mask"], errors="coerce").fillna(0).astype(int)
        )
        confidence = pd.to_numeric(
            frame["expert_" + expert + "_confidence"], errors="coerce"
        ).fillna(0.0)
        frame["expert_" + expert + "_confidence"] = confidence * frame[
            "expert_" + expert + "_expert_mask"
        ].to_numpy(dtype=float)


def _merge_trend_candidates(
    frame: pd.DataFrame, source: pd.DataFrame, keys: list[str]
) -> None:
    for branch in ("activity", "sleep", "social"):
        rows = source.loc[source["branch"].eq(branch)].copy()
        if rows[keys].duplicated().any():
            raise FusionV334OptimizationError(f"duplicate {branch} trend OOF rows")
        reliability_column = (
            "optimized_reliability"
            if "optimized_reliability" in rows
            else "reliability"
        )
        values = rows[
            keys + ["selected_probability", "personal_change_mask", reliability_column]
        ].rename(
            columns={
                "selected_probability": "_new_probability",
                "personal_change_mask": "_new_mask",
                reliability_column: "_new_reliability",
            }
        )
        merged = frame.merge(values, on=keys, how="left", validate="one_to_one")
        mask_column = f"trend_{branch}_personal_change_mask"
        evidence_column = f"trend_{branch}_personal_change_evidence"
        reliability_name = f"trend_{branch}_reliability"
        incoming_mask = (
            pd.to_numeric(merged["_new_mask"], errors="coerce").fillna(0).astype(int)
        )
        frame[mask_column] = incoming_mask
        frame[evidence_column] = merged["_new_probability"]
        frame[reliability_name] = pd.to_numeric(
            merged["_new_reliability"], errors="coerce"
        ).fillna(0.0)


def _overall_metrics(
    frame: pd.DataFrame, oof: pd.DataFrame, calibration: Mapping[str, Any]
) -> pd.DataFrame:
    target = frame["binary_target"].to_numpy(dtype=int)
    baseline_probability = oof["baseline_probability"].to_numpy(dtype=float)
    candidate_probability = oof["deployment_candidate_probability"].to_numpy(
        dtype=float
    )
    baseline = _metrics(target, baseline_probability)
    baseline["ece"] = _ece(target, baseline_probability)
    candidate = _metrics(target, candidate_probability)
    candidate["ece"] = _ece(target, candidate_probability)
    rows = [
        {"model": "current_online_baseline", **baseline},
        {"model": "corrected_calibration_reference", **dict(calibration)},
        {"model": "v334_candidate", **candidate},
    ]
    return pd.DataFrame(rows)


def _outer_metrics(frame: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold in range(OUTER_FOLDS):
        selected = frame["outer_fold"].astype(int).eq(fold).to_numpy()
        target = frame.loc[selected, "binary_target"].to_numpy(dtype=int)
        baseline = _metrics(target, oof.loc[selected, "baseline_probability"])
        candidate = _metrics(
            target, oof.loc[selected, "deployment_candidate_probability"]
        )
        rows.append(
            {
                "outer_fold": fold,
                "row_count": int(selected.sum()),
                "positive_rows": int(target.sum()),
                "baseline_auprc": baseline["auprc"],
                "candidate_auprc": candidate["auprc"],
                "auprc_delta": candidate["auprc"] - baseline["auprc"],
                "baseline_auroc": baseline["auroc"],
                "candidate_auroc": candidate["auroc"],
                "baseline_brier": baseline["brier"],
                "candidate_brier": candidate["brier"],
            }
        )
    return pd.DataFrame(rows)


def _group_metrics(frame: pd.DataFrame, oof: pd.DataFrame, group: str) -> pd.DataFrame:
    if group == "device_combination":
        values = evidence_combination(frame).astype(str).to_numpy()
    else:
        values = frame[group].astype(str).to_numpy()
    rows = []
    for value in sorted(set(values), key=lambda item: item.encode("utf-8")):
        selected = values == value
        target = frame.loc[selected, "binary_target"].to_numpy(dtype=int)
        for model, column in (
            ("current_online_baseline", "baseline_probability"),
            ("v334_candidate", "deployment_candidate_probability"),
        ):
            rows.append(
                {
                    "group": group,
                    "value": value,
                    "model": model,
                    **_metrics(target, oof.loc[selected, column]),
                }
            )
    return pd.DataFrame(rows)


def _weak_branch_ablation(frame: pd.DataFrame, model: V334FusionModel) -> pd.DataFrame:
    rows = []
    for branch in ("physiology", "social_context", "trend_social"):
        ablated = frame.copy()
        if branch.startswith("trend_"):
            name = branch.removeprefix("trend_")
            ablated[f"trend_{name}_personal_change_mask"] = 0
            ablated[f"trend_{name}_reliability"] = 0.0
            ablated[f"trend_{name}_personal_change_evidence"] = np.nan
        else:
            ablated[f"expert_{branch}_expert_mask"] = 0
            ablated[f"expert_{branch}_confidence"] = 0.0
            ablated[f"expert_{branch}_current_probability"] = np.nan
        prediction = model.predict_frame(ablated)
        available = prediction["available"].to_numpy(dtype=bool)
        probability = prediction["fusion_probability"].to_numpy(dtype=float)
        available = available & np.isfinite(probability)
        target = frame.loc[available, "binary_target"].to_numpy(dtype=int)
        values: dict[str, Any] = {
            "removed_branch": branch,
            "available_row_count": int(available.sum()),
            "coverage": float(available.mean()),
        }
        if available.any() and target.size and np.unique(target).size == 2:
            values.update(_metrics(target, probability[available]))
        else:
            values.update(
                {
                    "row_count": int(target.size),
                    "positive_rows": int(target.sum()),
                    "auprc": None,
                    "auroc": None,
                    "brier": None,
                }
            )
        rows.append(values)
    return pd.DataFrame(rows)


def _promotion_decision(
    config: FusionV334Config,
    overall: pd.DataFrame,
    outer: pd.DataFrame,
    source: pd.DataFrame,
) -> dict[str, Any]:
    gate = config.payload["promotion_gate"]
    baseline = overall.set_index("model").loc["current_online_baseline"]
    candidate = overall.set_index("model").loc["v334_candidate"]
    eligible = source[
        source["positive_rows"].ge(int(gate["source_minimum_positive_rows"]))
    ]
    checks = {
        "natural_auprc_delta": bool(
            candidate["auprc"] - baseline["auprc"]
            >= float(gate["minimum_natural_auprc_delta"])
        ),
        "natural_auroc_non_degraded": bool(
            candidate["auroc"] - baseline["auroc"]
            >= float(gate["minimum_natural_auroc_delta"])
        ),
        "natural_brier_within_tolerance": bool(
            candidate["brier"] - baseline["brier"]
            <= float(gate["maximum_natural_brier_increase"])
        ),
        "outer_fold_stability": bool(
            int(
                outer["auprc_delta"]
                .ge(-float(gate["outer_fold_auprc_tolerance"]))
                .sum()
            )
            >= int(gate["minimum_non_degraded_outer_folds"])
        ),
        "leave_one_source_stability": bool(
            eligible.empty
            or bool(
                eligible["auprc_delta"]
                .ge(-float(gate["maximum_leave_one_source_auprc_drop"]))
                .all()
            )
        ),
    }
    passed = all(checks.values())
    return {
        "version": "mood-social-v3.3.4-promotion-gate-v1",
        "promotion_passed": passed,
        "checks": checks,
        "thresholds": dict(gate),
        "observed": {
            "natural_auprc_delta": float(candidate["auprc"] - baseline["auprc"]),
            "natural_auroc_delta": float(candidate["auroc"] - baseline["auroc"]),
            "natural_brier_increase": float(candidate["brier"] - baseline["brier"]),
            "non_degraded_outer_folds": int(outer["auprc_delta"].ge(0.0).sum()),
            "eligible_leave_one_source_units": int(len(eligible)),
            "worst_eligible_leave_one_source_auprc_delta": float(
                eligible["auprc_delta"].min()
            )
            if not eligible.empty
            else None,
        },
        "package_default": "v334_candidate" if passed else "MH-20260802-013",
        "fallback_package": "MH-20260802-013",
    }


def _row_dict(frame: pd.DataFrame, model: str) -> dict[str, Any]:
    row = frame.loc[frame["model"].eq(model)].iloc[0]
    return {
        key: (
            None
            if pd.isna(value)
            else float(value)
            if isinstance(value, (float, np.floating))
            else int(value)
            if isinstance(value, (int, np.integer))
            else value
        )
        for key, value in row.to_dict().items()
        if key != "model"
    }


def _publish(
    config: FusionV334Config,
    frame: pd.DataFrame,
    search: pd.DataFrame,
    oof: pd.DataFrame,
    overall: pd.DataFrame,
    outer: pd.DataFrame,
    source: pd.DataFrame,
    device: pd.DataFrame,
    missing: pd.DataFrame,
    ablation: pd.DataFrame,
    promotion: Mapping[str, Any],
    summary: Mapping[str, Any],
    model: V334FusionModel,
    protection: Mapping[str, Any],
    command: Sequence[str],
) -> None:
    report = config.report_directory
    report.mkdir(parents=True, exist_ok=True)
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.model_path, compress=0, protocol=4)
    _write_parquet(search, report / "candidate_search.parquet")
    _write_parquet(oof, report / "oof_predictions.parquet")
    _write_parquet(overall, report / "overall_metrics.parquet")
    _write_parquet(outer, report / "outer_fold_stability.parquet")
    _write_parquet(source, report / "leave_one_source_stability.parquet")
    _write_parquet(device, report / "device_combination_metrics.parquet")
    _write_parquet(missing, report / "missing_modality_metrics.parquet")
    _write_parquet(ablation, report / "weak_branch_ablation.parquet")
    _write_json(report / "promotion.json", promotion)
    _write_json(report / "summary.json", summary)
    _write_json(report / "upstream_protection.json", protection)
    _write_json(report / "config.json", config.payload)
    _write_json(
        report / "run.json",
        {
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "training_version": TRAINING_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": list(command),
            "strict_oof": True,
        },
    )
    _write_json(
        report / "model_card.json",
        {
            "version": REPORT_VERSION,
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "model_version": MODEL_VERSION,
            "candidate_id": model.candidate_id,
            "strict_oof": True,
            "offline_only": True,
            "promotion_passed": bool(promotion["promotion_passed"]),
            "uses_public_dataset_id": False,
            "production_threshold_selected": False,
            "medical_diagnosis": False,
        },
    )
    (report / "model_card.md").write_text(
        "# MoodSocial V3.3.4 Fusion Candidate\n\n"
        f"- Run: `{RUN_ID}`\n- Candidate: `{model.candidate_id}`\n"
        f"- Promotion passed: `{str(bool(promotion['promotion_passed'])).lower()}`\n"
        "- Strict nested participant OOF; public `dataset_id` is audit metadata only.\n"
        "- Candidate remains offline until ART-002/API-003/TEST-002 complete.\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "version": MANIFEST_VERSION,
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "training_version": TRAINING_VERSION,
        "model_version": MODEL_VERSION,
        "strict_oof": True,
        "selected_candidate_id": model.candidate_id,
        "promotion_passed": bool(promotion["promotion_passed"]),
        "package_default": promotion["package_default"],
        "uses_public_dataset_id": False,
        "input_features": list(INPUT_FEATURES),
        "row_count": int(len(frame)),
        "positive_rows": int(frame["binary_target"].sum()),
        "model_sha256": _sha256_file(config.model_path),
        "oof_sha256": _sha256_file(report / "oof_predictions.parquet"),
        "upstream_protection": dict(protection),
        "production_boundary": {
            "consume_public_dataset_id": False,
            "select_production_threshold": False,
            "change_active_experts": False,
            "change_personal_trend": False,
            "include_model006_predictions": False,
            "change_http_behavior": False,
            "change_attention_levels": False,
            "overwrite_existing_models": False,
        },
    }
    _write_json(config.manifest_path, manifest)
    report_core = _report_core_sha256(report)
    manifest["report_core_sha256"] = report_core
    _write_json(config.manifest_path, manifest)
    checksums = (
        f"{_sha256_file(config.model_path)}  {config.model_path.name}\n"
        f"{_sha256_file(config.manifest_path)}  {config.manifest_path.name}\n"
    )
    (config.model_path.parent / "SHA256SUMS").write_text(
        checksums,
        encoding="utf-8",
        newline="\n",
    )
    artifact_rows = []
    for path in sorted(report.iterdir(), key=lambda item: item.name.encode("utf-8")):
        if path.is_file() and path.name not in {"artifacts.json"}:
            artifact_rows.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    _write_json(
        report / "artifacts.json",
        {
            "version": "mood-social-opt-fusion-003-artifacts-v1",
            "report_core_sha256": report_core,
            "artifacts": artifact_rows,
        },
    )


def _report_core_sha256(report: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (
            item
            for item in report.iterdir()
            if item.is_file() and item.name not in {"run.json", "artifacts.json"}
        ),
        key=lambda item: item.name.encode("utf-8"),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_outputs(config: FusionV334Config, *, overwrite: bool) -> None:
    if not overwrite and (
        config.model_path.exists()
        or config.manifest_path.exists()
        or config.report_directory.exists()
    ):
        raise FusionV334OptimizationError("OPT-FUSION-003 output exists; use overwrite")


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "FusionV334Config",
    "FusionV334OptimizationError",
    "RUN_ID",
    "TASK_ID",
    "V334FusionModel",
    "load_v334_config",
    "load_v334_inputs",
    "optimize_v334_fusion",
]
