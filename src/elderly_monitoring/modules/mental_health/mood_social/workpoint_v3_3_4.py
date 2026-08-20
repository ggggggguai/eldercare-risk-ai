"""Offline competition-demo workpoint selection for V3.3.4."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .runtime_policy import (
    RuntimePolicyObservation,
    evaluate_runtime_policy,
    load_runtime_policy_config,
)


TASK_ID = "OPT-WORKPOINT-001"
RUN_ID = "MH-20260804-022"
MODEL_VERSION = "mood-social-workpoint-v3.3.4-v1"
SCOPE = "competition_demo"
EPSILON = 1.0e-12


class WorkpointV334Error(RuntimeError):
    """Raised when the offline workpoint boundary is violated."""


def load_workpoint_config(
    path: str | Path, *, repository_root: str | Path | None = None
) -> tuple[Path, Mapping[str, Any], Path]:
    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise WorkpointV334Error("OPT-WORKPOINT-001 config is unreadable") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("task_id") != TASK_ID
        or payload.get("run_id") != RUN_ID
    ):
        raise WorkpointV334Error("workpoint task identity changed")
    if (
        payload.get("scope") != SCOPE
        or payload.get("production_threshold_selected") is not False
    ):
        raise WorkpointV334Error("workpoint must remain competition_demo only")
    root = Path(repository_root or config_path.parents[2]).resolve()
    return config_path, payload, root


def select_workpoint(
    config_path: str | Path,
    *,
    repository_root: str | Path | None = None,
    overwrite: bool = False,
    command: Sequence[str] = (),
) -> dict[str, Any]:
    config_file, config, root = load_workpoint_config(
        config_path, repository_root=repository_root
    )
    report = _resolve(root, config["output"]["report_directory"])
    manifest_path = _resolve(root, config["output"]["manifest_path"])
    if not overwrite and (report.exists() or manifest_path.exists()):
        raise WorkpointV334Error("OPT-WORKPOINT-001 output exists; use overwrite")
    frame, protection = _load_inputs(root, config)
    curves = pd.concat(
        [
            _threshold_curve(frame, "current_online_baseline", "baseline_probability"),
            _threshold_curve(
                frame, "v334_candidate", "deployment_candidate_probability"
            ),
        ],
        ignore_index=True,
    )
    selected_source = str(
        config["selection"]["candidate_source"]
        if protection["promotion_passed"]
        else config["selection"]["fallback_source"]
    )
    selected = _select_threshold(curves, selected_source, config["selection"])
    outer = _outer_threshold_curve(frame, selected_source, selected["threshold"])
    replay = _replay_fixture(config, root)
    summary = {
        "status": "pass",
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "scope": SCOPE,
        "production_threshold_selected": False,
        "selected_source": selected_source,
        "selected_threshold": selected,
        "candidate_promotion_passed": bool(protection["promotion_passed"]),
        "replay_fixture": "synthetic_contiguous_day_cases_v1",
        "replay_cases": int(len(replay)),
        "runtime_policy_config_sha256": protection["runtime_policy_config_sha256"],
    }
    _publish(
        config_file,
        config,
        report,
        manifest_path,
        curves,
        outer,
        replay,
        summary,
        protection,
        command,
    )
    return {
        "status": "pass",
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "selected_source": selected_source,
        "selected_threshold": selected["threshold"],
        "report_directory": str(report),
    }


def _load_inputs(
    root: Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    inputs = config["input"]
    paths = {
        "oof": _resolve(root, inputs["oof_path"]),
        "promotion": _resolve(root, inputs["promotion_path"]),
        "split": _resolve(root, inputs["split_path"]),
    }
    for key, path in paths.items():
        expected = str(inputs[f"{key}_sha256"])
        if not path.is_file() or _sha256_file(path) != expected:
            raise WorkpointV334Error(f"workpoint input hash failed: {key}")
    frame = pd.read_parquet(paths["oof"]).reset_index(drop=True)
    required = {
        "binary_target",
        "outer_fold",
        "baseline_probability",
        "deployment_candidate_probability",
    }
    if len(frame) != 22188 or not required.issubset(frame.columns):
        raise WorkpointV334Error("OPT-FUSION-003 OOF contract changed")
    if (
        frame["binary_target"].nunique() != 2
        or frame[["baseline_probability", "deployment_candidate_probability"]]
        .isna()
        .any()
        .any()
    ):
        raise WorkpointV334Error("workpoint OOF contains invalid probabilities")
    promotion = json.loads(paths["promotion"].read_text(encoding="utf-8"))
    runtime_path = _resolve(root, config["input"]["runtime_policy_path"])
    if _sha256_file(runtime_path) != str(config["input"]["runtime_policy_file_sha256"]):
        raise WorkpointV334Error("runtime policy file hash failed")
    runtime = load_runtime_policy_config(runtime_path)
    if (
        runtime.scope != "offline_demo"
        or runtime.expected_model_version != "mood-fusion-v3.3.3"
    ):
        raise WorkpointV334Error("frozen runtime policy binding changed")
    protection = {
        "oof_sha256": str(inputs["oof_sha256"]),
        "promotion_sha256": str(inputs["promotion_sha256"]),
        "split_sha256": str(inputs["split_sha256"]),
        "promotion_passed": bool(promotion.get("promotion_passed")),
        "fallback_package": str(promotion.get("fallback_package", "MH-20260802-013")),
        "runtime_policy_config_sha256": runtime.config_sha256,
        "runtime_policy_path": str(config["input"]["runtime_policy_path"]),
        "synthetic_fixture": True,
    }
    return frame, protection


def _threshold_curve(frame: pd.DataFrame, source: str, column: str) -> pd.DataFrame:
    target = frame["binary_target"].to_numpy(dtype=int)
    values = frame[column].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for threshold in np.linspace(0.01, 0.99, 99):
        predicted = values >= threshold
        tp = int(np.sum(predicted & (target == 1)))
        fp = int(np.sum(predicted & (target == 0)))
        fn = int(np.sum(~predicted & (target == 1)))
        tn = int(np.sum(~predicted & (target == 0)))
        rows.append(
            {
                "source": source,
                "threshold": float(round(threshold, 2)),
                **_binary_metrics(tp, fp, fn, tn),
            }
        )
    return pd.DataFrame(rows)


def _outer_threshold_curve(
    frame: pd.DataFrame, source: str, threshold: float
) -> pd.DataFrame:
    column = (
        "baseline_probability"
        if source == "current_online_baseline"
        else "deployment_candidate_probability"
    )
    rows = []
    for fold in sorted(frame["outer_fold"].unique()):
        subset = frame.loc[frame["outer_fold"].eq(fold)]
        target = subset["binary_target"].to_numpy(dtype=int)
        predicted = subset[column].to_numpy(dtype=float) >= threshold
        tp = int(np.sum(predicted & (target == 1)))
        fp = int(np.sum(predicted & (target == 0)))
        fn = int(np.sum(~predicted & (target == 1)))
        tn = int(np.sum(~predicted & (target == 0)))
        rows.append(
            {
                "source": source,
                "outer_fold": int(fold),
                "threshold": threshold,
                **_binary_metrics(tp, fp, fn, tn),
            }
        )
    return pd.DataFrame(rows)


def _select_threshold(
    curves: pd.DataFrame, source: str, selection: Mapping[str, Any]
) -> dict[str, Any]:
    subset = curves.loc[curves["source"].eq(source)].copy()
    eligible = subset.loc[
        subset["sensitivity"] >= float(selection["minimum_sensitivity"])
    ]
    if eligible.empty:
        eligible = subset
    chosen = eligible.sort_values(
        ["f1", "alert_rate_per_100_person_days", "threshold"],
        ascending=[False, True, True],
    ).iloc[0]
    return {
        "source": source,
        "threshold": float(chosen["threshold"]),
        "selection_rule": str(selection["rule"]),
        "sensitivity": float(chosen["sensitivity"]),
        "specificity": float(chosen["specificity"]),
        "precision": float(chosen["precision"]),
        "f1": float(chosen["f1"]),
        "alert_rate_per_100_person_days": float(
            chosen["alert_rate_per_100_person_days"]
        ),
        "false_alert_rate_per_100_negative_days": float(
            chosen["false_alert_rate_per_100_negative_days"]
        ),
        "eligible_threshold_count": int(len(eligible)),
    }


def _binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2.0 * precision * sensitivity / max(precision + sensitivity, EPSILON)
    total = tp + fp + fn + tn
    negatives = fp + tn
    return {
        "row_count": total,
        "positive_rows": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "alert_rate_per_100_person_days": 100.0 * (tp + fp) / max(total, 1),
        "false_alert_rate_per_100_negative_days": 100.0 * fp / max(negatives, 1),
    }


def _replay_fixture(config: Mapping[str, Any], root: Path) -> pd.DataFrame:
    runtime = load_runtime_policy_config(
        _resolve(root, config["input"]["runtime_policy_path"])
    )
    model_version = runtime.expected_model_version
    cases = {
        "stable_low": [0.20, 0.20, 0.20],
        "one_day_spike": [0.20, 0.20, 0.20, 0.80, 0.20, 0.20],
        "upgrade_after_two_days": [0.20, 0.20, 0.20, 0.35, 0.35, 0.35],
        "evidence_interruption": [0.20, 0.20, None, 0.80, 0.80, 0.80],
        "downgrade_hysteresis": [0.50, 0.50, 0.50, 0.39, 0.39, 0.39],
    }
    rows = []
    start = date(2026, 8, 1)
    for case, values in cases.items():
        observations = tuple(
            RuntimePolicyObservation(
                date=start + timedelta(days=index),
                model_version=model_version,
                available=value is not None,
                attention_index=value,
            )
            for index, value in enumerate(values)
        )
        result = evaluate_runtime_policy(observations, config=runtime)
        payload = asdict(result)
        payload["case"] = case
        payload["fixture_scope"] = "synthetic_contiguous_day_cases_v1"
        payload["synthetic_fixture"] = True
        payload["target_date"] = result.target_date.isoformat()
        payload["reason_codes"] = list(result.reason_codes)
        if case == "evidence_interruption":
            gap_result = evaluate_runtime_policy(
                observations[:3],
                target_date=start + timedelta(days=2),
                config=runtime,
            )
            payload["interruption_result_available"] = bool(gap_result.available)
            payload["interruption_evidence_status"] = gap_result.evidence_status
            payload["interruption_reason_codes"] = list(gap_result.reason_codes)
        else:
            payload["interruption_result_available"] = None
            payload["interruption_evidence_status"] = None
            payload["interruption_reason_codes"] = []
        rows.append(payload)
    return pd.DataFrame(rows)


def _publish(
    config_file: Path,
    config: Mapping[str, Any],
    report: Path,
    manifest_path: Path,
    curves: pd.DataFrame,
    outer: pd.DataFrame,
    replay: pd.DataFrame,
    summary: Mapping[str, Any],
    protection: Mapping[str, Any],
    command: Sequence[str],
) -> None:
    report.mkdir(parents=True, exist_ok=True)
    _write_parquet(curves, report / "threshold_curve.parquet")
    _write_parquet(outer, report / "outer_fold_selected_threshold.parquet")
    _write_parquet(replay, report / "stability_replay.parquet")
    _write_json(report / "selection.json", summary["selected_threshold"])
    _write_json(report / "summary.json", summary)
    _write_json(report / "upstream_protection.json", protection)
    _write_json(report / "config.json", config)
    _write_json(
        report / "run.json",
        {
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "scope": SCOPE,
            "command": list(command),
            "strict_oof": True,
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    )
    (report / "model_card.md").write_text(
        "# OPT-WORKPOINT-001\n\n- Scope: `competition_demo`\n- This workpoint is offline-only and is not a production threshold.\n- Synthetic replay fixtures are not real device deployment data.\n",
        encoding="utf-8",
        newline="\n",
    )
    report_core = _report_core_sha256(report)
    manifest = {
        "version": "mood-social-workpoint-v3.3.4-manifest-v1",
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "model_version": MODEL_VERSION,
        "scope": SCOPE,
        "production_threshold_selected": False,
        "selected_source": summary["selected_source"],
        "selected_threshold": summary["selected_threshold"],
        "report_core_sha256": report_core,
        "upstream_protection": protection,
    }
    _write_json(manifest_path, manifest)
    artifact_rows = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in sorted(report.iterdir(), key=lambda item: item.name.encode("utf-8"))
        if path.is_file() and path.name != "artifacts.json"
    ]
    _write_json(
        report / "artifacts.json",
        {
            "version": "mood-social-workpoint-artifacts-v1",
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


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


__all__ = [
    "RUN_ID",
    "TASK_ID",
    "WorkpointV334Error",
    "load_workpoint_config",
    "select_workpoint",
]
