"""Build the strict out-of-fold evidence table for FUSION-001.

This module only aligns already-published expert and PersonalTrend OOF rows.
It does not fit a stacking model, choose a threshold, or call production HTTP
code.  Labels are retained only as an audit/training column; they never enter
the evidence feature columns.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
import yaml
from sklearn.metrics import roc_auc_score

if TYPE_CHECKING:
    from elderly_monitoring.modules.mental_health.mood_social.offline_auxiliary import (
        OfflineAuxiliaryInputs,
    )


TASK_ID = "FUSION-001"
RUN_ID = "MH-20260802-009"
TABLE_VERSION = "mood-social-fusion-oof-table-v3.3.3-v1"
MANIFEST_VERSION = "mood-social-fusion-table-manifest-v1"
DIAGNOSTICS_VERSION = "mood-social-fusion-alignment-diagnostics-v1"
OUTER_FOLD_COUNT = 5
SPLIT_ID = "mood-social-v3.3.3-participant-nested-5x5-seed-20260728-v1"
SPLIT_SHA256 = "e9915dbc590a6ea454c34d26558ac5866c57ee8a8e78f44cdcc79c859df77ee3"
FEATURE_SCHEMA_SHA256 = (
    "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
)
EVALUATION_CORE_SHA256 = (
    "07bbcb88b7e41137d151e6c1775bcb805b0cd7e138476b802542a2b635e3a71e"
)
BASELINE_PROTECTION_SHA256 = (
    "ba403f4ed6f75e2bd612e198db8af5893a01afd09bfd5bfb99730b9f736428f6"
)
MODEL006_MODEL_SHA256 = (
    "35b36840edf542349a128f770cb02fdcbdb87a371f606ee053ff1bb201d93835"
)
MODEL006_MANIFEST_SHA256 = (
    "134d29c0174d483c0c343e20c48c9f0205c53511100199f69c12716f3a0c8806"
)
MODEL006_REPORT_CORE_SHA256 = (
    "a576f1d0835487e32d56678e1bb3bd61cc66afa97e332c69afbc6e86f637d883"
)
TREND_MODEL_SHA256 = "e924e2342077d5847650c3b650fb2f1ee0fa40ea2da5c4d406c26b0ba93ba58f"
TREND_MANIFEST_SHA256 = (
    "d9858a725c1bcd4cd883efadb03a379a1072b5b886c2e5ff232715aaf61d1846"
)
TREND_OOF_SHA256 = "f23d79b7be21a6a863414e139f537d61c8a9cb06e9a64e061dc60627893d408a"
TREND_REPORT_CORE_SHA256 = (
    "bedeffc520f6fbff2df347554b8d40287ec32baff2f83e6b98ef48b172baff20"
)
EXPERT_ORDER = ("activity", "sleep", "joint", "physiology", "social_context")
TREND_ORDER = ("activity", "sleep", "social")
EXPERT_SOURCE_SCOPE = {
    "activity": ("psyche_d", "resilient", "nhanes"),
    "sleep": ("psyche_d", "resilient", "nhanes"),
    "joint": ("psyche_d", "resilient", "nhanes"),
    "physiology": ("resilient",),
    "social_context": (
        "resilient",
        "nhanes",
        "shenzhen_elderly",
        "nhanes_ssq_2005_2008",
    ),
}
TREND_SOURCE_SCOPE = {
    "activity": ("psyche_d",),
    "sleep": ("psyche_d",),
    "social": ("shenzhen_elderly",),
}
_SAFE_WINDOW_COLUMNS = (
    "timescale_semantics",
    "window_start_date",
    "window_end_date",
    "window_calendar_days",
    "nominal_month",
    "cycle",
    "survey_cycle",
    "window_semantics",
    "window_number_values_json",
)
_REQUIRED_EXPERT_COLUMNS = {
    "prediction_schema_version",
    "prediction_id",
    "dataset_id",
    "global_participant_id",
    "canonical_row_index",
    "binary_target",
    "outer_fold",
    "raw_probability",
    "calibrated_probability",
    "expert_mask",
    "observed_feature_count",
    "observed_feature_fraction",
}
_REQUIRED_TREND_COLUMNS = {
    "oof_version",
    "branch",
    "dataset_id",
    "prediction_id",
    "canonical_row_index",
    "global_participant_id",
    "outer_fold",
    "target",
    "available",
    "personal_change_mask",
    "reliability",
    "valid_history_days",
    "timescale_semantics",
    "probability",
}


class FusionOOFError(RuntimeError):
    """Raised when frozen fusion OOF inputs or alignment are invalid."""


@dataclass(frozen=True)
class FusionOOFConfig:
    config_path: Path
    config_sha256: str
    payload: Mapping[str, Any]
    repository_root: Path
    report_directory: Path
    table_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class FusionOOFInputs:
    config: FusionOOFConfig
    trend_inputs: OfflineAuxiliaryInputs
    expert_frames: Mapping[str, pd.DataFrame]
    trend_frame: pd.DataFrame
    expert_feature_names: Mapping[str, tuple[str, ...]]
    upstream_protection: Mapping[str, Any]


def load_fusion_oof_config(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> FusionOOFConfig:
    config_path = Path(path).resolve()
    root = Path(repository_root or config_path.parents[2]).resolve()
    if not config_path.is_file():
        raise FusionOOFError(f"FUSION-001 config is missing: {config_path}")
    raw = config_path.read_bytes()
    try:
        payload = yaml.safe_load(raw.decode("utf-8"))
    except yaml.YAMLError as exc:
        raise FusionOOFError("FUSION-001 config is not valid YAML") from exc
    if not isinstance(payload, Mapping):
        raise FusionOOFError("FUSION-001 config root must be a mapping")
    _require(payload.get("task_id") == TASK_ID, "task ID changed")
    _require(payload.get("run_id") == RUN_ID, "run ID changed")
    _require(payload.get("table_version") == TABLE_VERSION, "table version changed")
    _require(payload.get("frozen_document_version") == "V3.3.3", "document changed")
    _require(payload.get("random_seed") == 20260728, "random seed changed")
    schema = _mapping(payload, "feature_schema")
    _require(
        schema.get("version") == "mood_social_feature_schema_v3_3_3",
        "schema version changed",
    )
    _require(schema.get("sha256") == FEATURE_SCHEMA_SHA256, "schema hash changed")
    split = _mapping(payload, "split")
    _require(split.get("split_id") == SPLIT_ID, "split ID changed")
    _require(split.get("sha256") == SPLIT_SHA256, "split hash changed")
    _require(split.get("outer_folds") == OUTER_FOLD_COUNT, "outer fold count changed")
    boundary = _mapping(payload, "production_boundary")
    _require(
        not any(bool(value) for value in boundary.values()),
        "production boundary changed",
    )
    confidence = _mapping(payload, "confidence")
    _require(
        confidence.get("probability_representation") == "current_calibrated",
        "probability representation changed",
    )
    reliability = _mapping(confidence, "model_reliability")
    _require(
        reliability.get("method") == "outer_fold_excluded_source_auroc_excess",
        "reliability method changed",
    )
    _require(reliability.get("minimum_source_rows") == 100, "source minimum changed")
    _require(
        reliability.get("fallback") == "outer_fold_excluded_expert_pooled",
        "reliability fallback changed",
    )
    output = _mapping(payload, "output")
    report_directory = _resolve(root, output.get("report_directory"))
    table_path = report_directory / str(output.get("table_file"))
    manifest_path = report_directory / str(output.get("manifest_file"))
    return FusionOOFConfig(
        config_path=config_path,
        config_sha256=_sha256_file(config_path),
        payload=payload,
        repository_root=root,
        report_directory=report_directory,
        table_path=table_path,
        manifest_path=manifest_path,
    )


def load_fusion_oof_inputs(config: FusionOOFConfig) -> FusionOOFInputs:
    from elderly_monitoring.modules.mental_health.mood_social.personal_trend import (
        load_personal_trend_config,
        load_personal_trend_inputs,
    )

    root = config.repository_root
    upstream = _mapping(config.payload, "upstream")
    trend_config = load_personal_trend_config(
        _resolve(root, upstream.get("personal_trend_config_path")),
        repository_root=root,
    )
    trend_inputs = load_personal_trend_inputs(trend_config)
    _require(
        _sha256_file(trend_config.model_path) == TREND_MODEL_SHA256,
        "TREND model hash drifted",
    )
    _require(
        _sha256_file(trend_config.manifest_path) == TREND_MANIFEST_SHA256,
        "TREND manifest hash drifted",
    )
    trend_spec = _mapping(upstream, "personal_trend")
    trend_oof_path = _resolve(root, trend_spec.get("oof_path"))
    _require(_sha256_file(trend_oof_path) == TREND_OOF_SHA256, "TREND OOF hash drifted")
    _require(
        trend_spec.get("report_core_sha256") == TREND_REPORT_CORE_SHA256,
        "TREND report binding changed",
    )
    _require(
        _sha256_file(_resolve(root, trend_spec.get("model_path")))
        == TREND_MODEL_SHA256,
        "TREND config model binding changed",
    )
    trend_frame = pd.read_parquet(trend_oof_path)
    _require(
        _REQUIRED_TREND_COLUMNS.issubset(trend_frame.columns),
        "TREND OOF schema is incomplete",
    )
    _require(
        set(trend_frame["branch"].astype(str)) == set(TREND_ORDER),
        "TREND branches changed",
    )
    expert_frames: dict[str, pd.DataFrame] = {}
    feature_names: dict[str, tuple[str, ...]] = {}
    expert_specs = _mapping(upstream, "experts")
    for expert in EXPERT_ORDER:
        spec = _mapping(expert_specs, expert)
        model_path = _resolve(root, spec.get("model_path"))
        manifest_path = _resolve(root, spec.get("manifest_path"))
        oof_path = _resolve(root, spec.get("oof_path"))
        _require(
            _sha256_file(model_path) == str(spec.get("model_sha256")),
            f"{expert} model hash drifted",
        )
        _require(
            _sha256_file(manifest_path) == str(spec.get("manifest_sha256")),
            f"{expert} manifest hash drifted",
        )
        _require(
            _sha256_file(oof_path) == str(spec.get("oof_sha256")),
            f"{expert} OOF hash drifted",
        )
        manifest = _read_json(manifest_path)
        names = tuple(str(item) for item in manifest.get("input_feature_names", ()))
        _require(
            names and len(names) == len(set(names)),
            f"{expert} input feature binding changed",
        )
        feature_names[expert] = names
        frame = pd.read_parquet(oof_path)
        _require(
            _REQUIRED_EXPERT_COLUMNS.issubset(frame.columns),
            f"{expert} OOF schema is incomplete",
        )
        _require(
            set(frame["prediction_schema_version"].astype(str))
            == {str(spec.get("prediction_schema_version"))},
            f"{expert} prediction schema changed",
        )
        expert_frames[expert] = frame
    model006 = _mapping(upstream, "model006")
    _require(model006.get("enters_fusion") is False, "MODEL-006 entered fusion")
    _require(
        model006.get("model_sha256") == MODEL006_MODEL_SHA256,
        "MODEL-006 model binding changed",
    )
    _require(
        model006.get("manifest_sha256") == MODEL006_MANIFEST_SHA256,
        "MODEL-006 manifest binding changed",
    )
    _require(
        model006.get("report_core_sha256") == MODEL006_REPORT_CORE_SHA256,
        "MODEL-006 report binding changed",
    )
    protection = {
        "version": "mood-social-fusion-upstream-protection-v1",
        "split_sha256": SPLIT_SHA256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "baseline_protection_sha256": BASELINE_PROTECTION_SHA256,
        "evaluation_core_sha256": EVALUATION_CORE_SHA256,
        "model006": dict(model006),
        "personal_trend": dict(trend_spec),
        "experts": {name: dict(_mapping(expert_specs, name)) for name in EXPERT_ORDER},
    }
    return FusionOOFInputs(
        config=config,
        trend_inputs=trend_inputs,
        expert_frames=expert_frames,
        trend_frame=trend_frame,
        expert_feature_names=feature_names,
        upstream_protection=protection,
    )


def build_fusion_oof_table(
    inputs: FusionOOFInputs,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Build the deterministic union table and alignment diagnostics."""

    base = _build_base_frame(inputs.trend_inputs)
    diagnostics: dict[str, Any] = {
        "version": DIAGNOSTICS_VERSION,
        "task_id": TASK_ID,
        "strict_oof": True,
        "stacking_fitted": False,
        "threshold_selected": False,
        "model006_columns_consumed": [],
        "expected_row_count": int(len(base)),
        "expected_participant_count": int(base["global_participant_id"].nunique()),
        "expected_outer_fold_counts": {
            str(key): int(value)
            for key, value in base["outer_fold"].value_counts().sort_index().items()
        },
        "expert_alignment": {},
        "trend_alignment": {},
        "warnings": [
            "day_mask is a public aggregate-window observation proxy, not a runtime D-6..D vector",
            "social PersonalTrend is a Shenzhen marital-status engineering proxy without direct S10/PHQ-9 longitudinal validation",
        ],
    }
    for expert in EXPERT_ORDER:
        base = _merge_expert(base, inputs, expert, diagnostics)
    for branch in TREND_ORDER:
        base = _merge_trend(base, inputs, branch, diagnostics)
    base = _finalize_table(base)
    coverage = _coverage_summary(base)
    reliability = _reliability_summary(base)
    diagnostics["observed_row_count"] = int(len(base))
    diagnostics["observed_participant_count"] = int(
        base["global_participant_id"].nunique()
    )
    diagnostics["coverage_pattern_count"] = int(base["mask_pattern"].nunique())
    return base, diagnostics, coverage, reliability


def train_fusion_oof_table(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    overwrite: bool = False,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    root = Path(repository_root).resolve()
    config = load_fusion_oof_config(config_path, repository_root=root)
    _validate_output_paths(config, overwrite=overwrite)
    inputs = load_fusion_oof_inputs(config)
    table, diagnostics, coverage, reliability = build_fusion_oof_table(inputs)
    report = _publish(
        config,
        inputs,
        table,
        diagnostics,
        coverage,
        reliability,
        started=started,
        ended=datetime.now(timezone.utc),
        overwrite=overwrite,
        command=command or (),
    )
    return report


def _build_base_frame(inputs: OfflineAuxiliaryInputs) -> pd.DataFrame:
    assignments = inputs.assignments.copy()
    assignment_map = {
        str(row.global_participant_id): int(row.outer_fold)
        for row in assignments.itertuples(index=False)
    }
    records: list[dict[str, Any]] = []
    for dataset_id in sorted(inputs.frames):
        frame = inputs.frames[dataset_id]
        for row_index, row in frame.iterrows():
            global_id = str(row.get("global_participant_id"))
            _require(
                global_id in assignment_map,
                f"{dataset_id} row {row_index} has no split assignment",
            )
            target = _finite(row.get("binary_target"))
            _require(
                target in (0.0, 1.0), f"{dataset_id} row {row_index} target is invalid"
            )
            masks = _all_feature_masks(row)
            day_masks = _day_mask_summary(row)
            context = {
                key: _json_scalar(row.get(key))
                for key in _SAFE_WINDOW_COLUMNS
                if key in frame.columns and _json_scalar(row.get(key)) is not None
            }
            prediction_id = f"{dataset_id}::row={int(row_index)}"
            records.append(
                {
                    "fusion_schema_version": TABLE_VERSION,
                    "fusion_row_id": prediction_id,
                    "dataset_id": dataset_id,
                    "global_participant_id": global_id,
                    "canonical_row_index": int(row_index),
                    "outer_fold": assignment_map[global_id],
                    "binary_target": int(target),
                    "window_key": prediction_id,
                    "window_context_json": _canonical_json(context),
                    "feature_mask_json": _canonical_json(masks),
                    "day_mask_json": _canonical_json(day_masks),
                    "feature_mask_count": int(sum(masks.values())),
                    "feature_mask": int(bool(masks and any(masks.values()))),
                    "day_mask": _day_mask_scalar(day_masks),
                    "day_mask_semantics": "aggregate_public_window_observed_proxy",
                    "_feature_masks": masks,
                    "_day_masks": day_masks,
                    "_source_row": row.to_dict(),
                }
            )
    result = pd.DataFrame(records)
    _require(len(result) == 22191, f"canonical fusion row count changed: {len(result)}")
    _require(
        not result["fusion_row_id"].duplicated().any(), "fusion row IDs are duplicated"
    )
    return result


def _merge_expert(
    base: pd.DataFrame,
    inputs: FusionOOFInputs,
    expert: str,
    diagnostics: dict[str, Any],
) -> pd.DataFrame:
    frame = inputs.expert_frames[expert].copy()
    source_scope = set(EXPERT_SOURCE_SCOPE[expert])
    expected = {
        (str(row.dataset_id), int(row.canonical_row_index))
        for row in base.itertuples(index=False)
        if str(row.dataset_id) in source_scope
    }
    actual = {
        (str(row.dataset_id), int(row.canonical_row_index))
        for row in frame.itertuples(index=False)
    }
    _require(
        actual == expected, f"{expert} OOF row coverage does not match canonical scope"
    )
    _require(
        not frame["prediction_id"].duplicated().any(),
        f"{expert} OOF IDs are duplicated",
    )
    names = inputs.expert_feature_names[expert]
    reliability_map, scope_map = _reliability_maps(frame)
    lookup = {
        (str(row.dataset_id), int(row.canonical_row_index)): row
        for row in frame.itertuples(index=False)
    }
    prefix = f"expert_{expert}_"
    values: dict[str, list[Any]] = {
        name: []
        for name in (
            "applicable",
            "raw_probability",
            "current_probability",
            "model_reliability",
            "feature_coverage",
            "confidence",
            "feature_mask_json",
            "feature_mask_count",
            "feature_mask",
            "day_mask",
            "day_mask_count",
            "day_mask_semantics",
            "expert_mask",
            "observed_feature_count",
            "upstream_prediction_id",
            "reliability_scope",
        )
    }
    for row in base.itertuples(index=False):
        key = (str(row.dataset_id), int(row.canonical_row_index))
        if key not in lookup:
            _append_unavailable(values)
            continue
        source = inputs.trend_inputs.frames[str(row.dataset_id)].iloc[
            int(row.canonical_row_index)
        ]
        item = lookup[key]
        observed_count = int(item.observed_feature_count)
        observed_fraction = _finite(item.observed_feature_fraction)
        _require(
            observed_fraction is not None and 0 <= observed_fraction <= 1,
            f"{expert} feature coverage is invalid",
        )
        masks = {name: int(_feature_observed(source, name)) for name in names}
        count = int(sum(masks.values()))
        _require(
            count == observed_count,
            f"{expert} observed feature count disagrees with canonical mask",
        )
        _require(
            abs(count / len(names) - float(observed_fraction)) < 1e-9,
            f"{expert} feature coverage disagrees with canonical mask",
        )
        day = _branch_day_mask(source, expert)
        mask = int(item.expert_mask)
        _require(mask in (0, 1), f"{expert} expert mask is invalid")
        raw = _finite(item.raw_probability)
        current = _finite(item.calibrated_probability)
        if mask == 1:
            _require(
                raw is not None and current is not None,
                f"{expert} available row has null probability",
            )
        else:
            _require(
                raw is None and current is None,
                f"{expert} unavailable row has probability",
            )
        rel, rel_scope = (
            reliability_map[(int(item.outer_fold), str(item.dataset_id))],
            scope_map[(int(item.outer_fold), str(item.dataset_id))],
        )
        confidence = (
            float(np.clip(rel * float(observed_fraction), 0.0, 1.0)) if mask else 0.0
        )
        values["applicable"].append(1)
        values["raw_probability"].append(raw)
        values["current_probability"].append(current)
        values["model_reliability"].append(float(rel))
        values["feature_coverage"].append(float(observed_fraction))
        values["confidence"].append(confidence)
        values["feature_mask_json"].append(_canonical_json(masks))
        values["feature_mask_count"].append(count)
        values["feature_mask"].append(int(count > 0))
        values["day_mask"].append(day["mask"])
        values["day_mask_count"].append(day["count"])
        values["day_mask_semantics"].append(day["semantics"])
        values["expert_mask"].append(mask)
        values["observed_feature_count"].append(observed_count)
        values["upstream_prediction_id"].append(str(item.prediction_id))
        values["reliability_scope"].append(rel_scope)
    addition = pd.DataFrame(
        {f"{prefix}{name}": column for name, column in values.items()},
        index=base.index,
    )
    base = pd.concat([base, addition], axis=1)
    diagnostics["expert_alignment"][expert] = {
        "expected_rows": len(expected),
        "matched_rows": len(actual),
        "source_scope": sorted(source_scope),
        "prediction_schema_versions": sorted(
            frame["prediction_schema_version"].astype(str).unique()
        ),
        "outer_fold_values": sorted(frame["outer_fold"].astype(int).unique().tolist()),
        "mismatch_count": 0,
    }
    return base


def _merge_trend(
    base: pd.DataFrame,
    inputs: FusionOOFInputs,
    branch: str,
    diagnostics: dict[str, Any],
) -> pd.DataFrame:
    frame = inputs.trend_frame[
        inputs.trend_frame["branch"].astype(str).eq(branch)
    ].copy()
    expected = {
        (str(row.dataset_id), int(row.canonical_row_index))
        for row in base.itertuples(index=False)
        if str(row.dataset_id) in set(TREND_SOURCE_SCOPE[branch])
    }
    actual = {
        (str(row.dataset_id), int(row.canonical_row_index))
        for row in frame.itertuples(index=False)
    }
    _require(
        actual == expected,
        f"PersonalTrend {branch} OOF coverage does not match canonical scope",
    )
    _require(
        not frame["prediction_id"].duplicated().any(),
        f"PersonalTrend {branch} IDs are duplicated",
    )
    lookup = {
        (str(row.dataset_id), int(row.canonical_row_index)): row
        for row in frame.itertuples(index=False)
    }
    prefix = f"trend_{branch}_"
    fields = (
        "applicable",
        "personal_change_evidence",
        "reliability",
        "feature_mask",
        "day_mask",
        "valid_history_days",
        "personal_change_mask",
        "timescale_semantics",
        "upstream_prediction_id",
    )
    values = {name: [] for name in fields}
    component_names = (
        "robust_deviation",
        "risk_slope",
        "isolation_forest",
        "change_point",
        "persistence",
        "history_coverage",
    )
    component_values = {name: [] for name in component_names}
    for row in base.itertuples(index=False):
        key = (str(row.dataset_id), int(row.canonical_row_index))
        if key not in lookup:
            _append_trend_unavailable(values, component_values)
            continue
        item = lookup[key]
        mask = int(item.personal_change_mask)
        available = bool(item.available)
        probability = _finite(item.probability)
        reliability = _finite(item.reliability)
        history = int(item.valid_history_days)
        _require(mask in (0, 1), f"PersonalTrend {branch} mask is invalid")
        _require(
            reliability is not None and 0 <= reliability <= 1,
            f"PersonalTrend {branch} reliability is invalid",
        )
        _require(
            (mask == 1) == available, f"PersonalTrend {branch} available/mask mismatch"
        )
        _require(
            (probability is not None) == available,
            f"PersonalTrend {branch} probability/mask mismatch",
        )
        source = inputs.trend_inputs.frames[str(row.dataset_id)].iloc[
            int(row.canonical_row_index)
        ]
        raw_mask = _branch_feature_mask(source, branch)
        day = _branch_day_mask(source, branch)
        values["applicable"].append(1)
        values["personal_change_evidence"].append(probability)
        values["reliability"].append(float(reliability))
        values["feature_mask"].append(int(raw_mask > 0))
        values["day_mask"].append(day["mask"])
        values["valid_history_days"].append(history)
        values["personal_change_mask"].append(mask)
        values["timescale_semantics"].append(str(item.timescale_semantics))
        values["upstream_prediction_id"].append(str(item.prediction_id))
        for component in component_names:
            value = _finite(getattr(item, component, None))
            _require(
                value is not None and 0 <= value <= 1,
                f"PersonalTrend {branch} component is invalid",
            )
            component_values[component].append(value)
    addition = pd.DataFrame(
        {
            **{f"{prefix}{name}": column for name, column in values.items()},
            **{f"{prefix}{name}": column for name, column in component_values.items()},
        },
        index=base.index,
    )
    base = pd.concat([base, addition], axis=1)
    diagnostics["trend_alignment"][branch] = {
        "expected_rows": len(expected),
        "matched_rows": len(actual),
        "source_scope": sorted(TREND_SOURCE_SCOPE[branch]),
        "oof_version": sorted(frame["oof_version"].astype(str).unique()),
        "outer_fold_values": sorted(frame["outer_fold"].astype(int).unique().tolist()),
        "mismatch_count": 0,
    }
    return base


def _finalize_table(base: pd.DataFrame) -> pd.DataFrame:
    drop = ["_feature_masks", "_day_masks", "_source_row"]
    table = base.drop(columns=drop)
    pattern_columns = [f"expert_{name}_expert_mask" for name in EXPERT_ORDER] + [
        f"trend_{name}_personal_change_mask" for name in TREND_ORDER
    ]
    table["mask_pattern"] = (
        table[pattern_columns].fillna(0).astype(int).astype(str).agg("".join, axis=1)
    )
    table = table.sort_values(
        ["outer_fold", "dataset_id", "canonical_row_index"], kind="stable"
    ).reset_index(drop=True)
    return table


def _coverage_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_id, group in table.groupby("dataset_id", sort=True):
        for fold, fold_group in group.groupby("outer_fold", sort=True):
            row: dict[str, Any] = {
                "dataset_id": str(dataset_id),
                "outer_fold": int(fold),
                "row_count": int(len(fold_group)),
                "positive_row_count": int(fold_group["binary_target"].sum()),
            }
            for expert in EXPERT_ORDER:
                row[f"{expert}_available_count"] = int(
                    fold_group[f"expert_{expert}_expert_mask"].fillna(0).sum()
                )
            for branch in TREND_ORDER:
                row[f"trend_{branch}_available_count"] = int(
                    fold_group[f"trend_{branch}_personal_change_mask"].fillna(0).sum()
                )
            rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["dataset_id", "outer_fold"], kind="stable")
        .reset_index(drop=True)
    )


def _reliability_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for expert in EXPERT_ORDER:
        prefix = f"expert_{expert}_"
        subset = table[table[f"{prefix}applicable"].eq(1)]
        for scope, group in subset.groupby(f"{prefix}reliability_scope", sort=True):
            rows.append(
                {
                    "branch": expert,
                    "reliability_scope": str(scope),
                    "row_count": int(len(group)),
                    "mean_model_reliability": float(
                        group[f"{prefix}model_reliability"].mean()
                    ),
                    "mean_feature_coverage": float(
                        group[f"{prefix}feature_coverage"].mean()
                    ),
                    "mean_confidence": float(group[f"{prefix}confidence"].mean()),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["branch", "reliability_scope"], kind="stable")
        .reset_index(drop=True)
    )


def _reliability_maps(
    frame: pd.DataFrame,
) -> tuple[dict[tuple[int, str], float], dict[tuple[int, str], str]]:
    reliability: dict[tuple[int, str], float] = {}
    scopes: dict[tuple[int, str], str] = {}
    minimum_rows = 100
    minimum_positive = 5
    minimum_negative = 5
    for fold in range(OUTER_FOLD_COUNT):
        train = frame[
            (frame["outer_fold"].astype(int) != fold)
            & (frame["expert_mask"].astype(int) == 1)
        ].copy()
        train = train[
            np.isfinite(pd.to_numeric(train["calibrated_probability"], errors="coerce"))
        ]
        pooled = _auroc_reliability(train)
        for source in sorted(frame["dataset_id"].astype(str).unique()):
            group = train[train["dataset_id"].astype(str).eq(source)]
            enough = (
                len(group) >= minimum_rows
                and int(group["binary_target"].sum()) >= minimum_positive
                and int((1 - group["binary_target"]).sum()) >= minimum_negative
            )
            if enough:
                reliability[(fold, source)] = _auroc_reliability(group)
                scopes[(fold, source)] = "source_excluded_outer_fold"
            else:
                reliability[(fold, source)] = pooled
                scopes[(fold, source)] = "expert_pooled_excluded_outer_fold"
    return reliability, scopes


def _auroc_reliability(frame: pd.DataFrame) -> float:
    if len(frame) == 0:
        return 0.0
    target = frame["binary_target"].to_numpy(dtype="int64")
    probability = frame["calibrated_probability"].to_numpy(dtype="float64")
    if np.unique(target).size < 2:
        return 0.0
    try:
        auroc = float(roc_auc_score(target, probability))
    except ValueError:
        return 0.0
    return float(np.clip(2.0 * (auroc - 0.5), 0.0, 1.0))


def _all_feature_masks(row: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in row.items():
        if not str(key).startswith("feature_mask."):
            continue
        mask = _finite(value)
        if mask is None:
            continue
        result[str(key)[len("feature_mask.") :]] = int(mask == 1.0)
    return dict(sorted(result.items()))


def _branch_feature_mask(row: Mapping[str, Any], branch: str) -> int:
    if branch == "social":
        names = ("social_context.marital_status",)
    elif branch == "activity":
        names = tuple(
            key[len("feature_mask.") :]
            for key in row
            if str(key).startswith("feature_mask.activity.")
            and not str(key).endswith("feature_coverage")
        )
    elif branch == "sleep":
        names = tuple(
            key[len("feature_mask.") :]
            for key in row
            if str(key).startswith("feature_mask.sleep.")
            and not str(key).endswith("feature_coverage")
        )
    else:
        names = tuple(
            key[len("feature_mask.") :]
            for key in row
            if str(key).startswith("feature_mask.physiology.")
            and not str(key).endswith("feature_coverage")
        )
    return sum(int(_feature_observed(row, name)) for name in names)


def _feature_observed(row: Mapping[str, Any], name: str) -> bool:
    """Apply the canonical mask, including fold-fitted activity ECDF input."""

    if name == "activity.activity_volume_norm":
        mask = _finite(row.get("x_source_mask"))
        value = row.get("x_source_value")
    else:
        mask = _finite(row.get(f"feature_mask.{name}"))
        value = row.get(name)
    if mask != 1.0:
        return False
    if value is None or value is pd.NA:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return _finite(value) is not None


def _branch_day_mask(row: Mapping[str, Any], branch: str) -> dict[str, Any]:
    if branch == "activity":
        count = _masked_count(row, "activity.valid_days")
        return {
            "mask": None if count is None else int(count > 0),
            "count": count,
            "semantics": "aggregate_public_window_observed_proxy",
        }
    if branch == "sleep":
        count = _masked_count(row, "sleep.valid_nights")
        return {
            "mask": None if count is None else int(count > 0),
            "count": count,
            "semantics": "aggregate_public_window_observed_proxy",
        }
    if branch == "joint":
        activity = _masked_count(row, "activity.valid_days")
        sleep = _masked_count(row, "sleep.valid_nights")
        if activity is None or sleep is None:
            return {
                "mask": None,
                "count": None,
                "semantics": "aggregate_public_window_joint_proxy",
            }
        return {
            "mask": int(activity > 0 and sleep > 0),
            "count": min(activity, sleep),
            "semantics": "aggregate_public_window_joint_proxy",
        }
    if branch == "physiology":
        count = _masked_count(row, "physiology.valid_nights")
        return {
            "mask": None if count is None else int(count > 0),
            "count": count,
            "semantics": "aggregate_public_window_observed_proxy",
        }
    return {"mask": None, "count": None, "semantics": "static_context_not_applicable"}


def _day_mask_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        domain: _branch_day_mask(row, domain)
        for domain in ("activity", "sleep", "physiology", "social_context")
    }


def _day_mask_scalar(summary: Mapping[str, Mapping[str, Any]]) -> int | None:
    values = [
        item.get("mask") for item in summary.values() if item.get("mask") is not None
    ]
    if not values:
        return None
    return int(any(values))


def _masked_count(row: Mapping[str, Any], name: str) -> int | None:
    value = _finite(row.get(name))
    mask = _finite(row.get(f"feature_mask.{name}"))
    if value is None or mask != 1.0:
        return None
    return int(round(value))


def _append_unavailable(values: dict[str, list[Any]]) -> None:
    values["applicable"].append(0)
    values["raw_probability"].append(None)
    values["current_probability"].append(None)
    values["model_reliability"].append(None)
    values["feature_coverage"].append(None)
    values["confidence"].append(0.0)
    values["feature_mask_json"].append(None)
    values["feature_mask_count"].append(0)
    values["feature_mask"].append(0)
    values["day_mask"].append(None)
    values["day_mask_count"].append(None)
    values["day_mask_semantics"].append("not_applicable_source_scope")
    values["expert_mask"].append(0)
    values["observed_feature_count"].append(0)
    values["upstream_prediction_id"].append(None)
    values["reliability_scope"].append("not_applicable_source_scope")


def _append_trend_unavailable(
    values: dict[str, list[Any]], components: dict[str, list[Any]]
) -> None:
    values["applicable"].append(0)
    values["personal_change_evidence"].append(None)
    values["reliability"].append(0.0)
    values["feature_mask"].append(0)
    values["day_mask"].append(None)
    values["valid_history_days"].append(0)
    values["personal_change_mask"].append(0)
    values["timescale_semantics"].append(None)
    values["upstream_prediction_id"].append(None)
    for name in components:
        components[name].append(None)


def _publish(
    config: FusionOOFConfig,
    inputs: FusionOOFInputs,
    table: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    coverage: pd.DataFrame,
    reliability: pd.DataFrame,
    *,
    started: datetime,
    ended: datetime,
    overwrite: bool,
    command: Sequence[str],
) -> dict[str, Any]:
    report = config.report_directory
    stage = Path(tempfile.mkdtemp(prefix=f".{report.name}.", dir=str(report.parent)))
    try:
        _write_parquet(table, stage / "fusion_oof_table.parquet")
        _write_parquet(coverage, stage / "coverage_summary.parquet")
        _write_parquet(reliability, stage / "confidence_reliability.parquet")
        _write_json(stage / "alignment_diagnostics.json", diagnostics)
        _write_json(
            stage / "warnings.json", {"warnings": diagnostics.get("warnings", [])}
        )
        _write_json(stage / "upstream_protection.json", inputs.upstream_protection)
        shutil.copyfile(config.config_path, stage / "fusion_table_config.yaml")
        manifest = {
            "version": MANIFEST_VERSION,
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "table_version": TABLE_VERSION,
            "config_sha256": config.config_sha256,
            "split_id": SPLIT_ID,
            "split_sha256": SPLIT_SHA256,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "row_count": int(len(table)),
            "participant_count": int(table["global_participant_id"].nunique()),
            "positive_row_count": int(table["binary_target"].sum()),
            "outer_folds": OUTER_FOLD_COUNT,
            "strict_oof": True,
            "model006_included": False,
            "production_boundary": dict(
                _mapping(config.payload, "production_boundary")
            ),
            "mask_semantics": dict(_mapping(config.payload, "mask_semantics")),
            "artifacts": {},
        }
        manifest["artifacts"] = {
            item["path"]: item
            for item in _file_hashes(
                stage,
                exclude={"artifacts.json", "run.json", "fusion_table_manifest.json"},
            )
        }
        _write_json(stage / "fusion_table_manifest.json", manifest)
        report_core = _report_core_sha256(stage)
        run = {
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "command": list(command),
            "started_at_utc": started.isoformat(),
            "ended_at_utc": ended.isoformat(),
            "duration_seconds": (ended - started).total_seconds(),
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "report_core_sha256": report_core,
            "conclusion": "FUSION-001 strict OOF evidence table completed; stacking remains deferred to FUSION-002.",
        }
        _write_json(stage / "run.json", run)
        artifacts = {
            "version": "mood-social-fusion-artifacts-v1",
            "report_core_sha256": report_core,
            "files": _file_hashes(stage, exclude={"artifacts.json", "run.json"}),
        }
        _write_json(stage / "artifacts.json", artifacts)
        if report.exists():
            if not overwrite:
                raise FusionOOFError(f"FUSION-001 report already exists: {report}")
            shutil.rmtree(report)
        stage.replace(report)
        return {
            "run_id": RUN_ID,
            "report_directory": str(report),
            "table_sha256": _sha256_file(report / "fusion_oof_table.parquet"),
            "table_row_count": len(table),
            "report_core_sha256": report_core,
            "diagnostics": dict(diagnostics),
        }
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _validate_output_paths(config: FusionOOFConfig, *, overwrite: bool) -> None:
    if config.report_directory.exists() and not overwrite:
        raise FusionOOFError(
            f"FUSION-001 report already exists: {config.report_directory}"
        )
    if (
        config.report_directory == config.repository_root
        or config.report_directory.parent == config.repository_root.parent
    ):
        raise FusionOOFError("FUSION-001 report directory is too broad")


def _report_core_sha256(directory: Path) -> str:
    names = (
        "alignment_diagnostics.json",
        "confidence_reliability.parquet",
        "coverage_summary.parquet",
        "fusion_oof_table.parquet",
        "fusion_table_config.yaml",
        "fusion_table_manifest.json",
        "upstream_protection.json",
        "warnings.json",
    )
    digest = hashlib.sha256()
    for name in names:
        path = directory / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_hashes(directory: Path, *, exclude: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name.encode("utf-8")):
        if path.name in exclude or not path.is_file():
            continue
        rows.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=3,
        use_dictionary=False,
        write_statistics=False,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(_canonical_json(payload) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FusionOOFError(f"JSON object expected: {path}")
    return value


def _resolve(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise FusionOOFError("path binding is missing")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise FusionOOFError(f"mapping is missing: {key}")
    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FusionOOFError(message)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
