"""Independently validate MODEL-006 artifacts, OOF coverage and metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from elderly_monitoring.modules.mental_health.mood_social.offline_auxiliary import (
    BUNDLE_VERSION,
    EVALUATION_CORE_SHA256,
    GRADE_IDS,
    MANIFEST_VERSION,
    MODEL_FAMILY_BY_TASK,
    OOF_VERSION,
    RUN_ID,
    TASK_ID,
    TRAINING_VERSION,
    OfflineAuxiliaryBundle,
    OfflineAuxiliaryError,
    build_task_frame,
    load_offline_auxiliary_config,
    load_offline_auxiliary_inputs,
)


class ValidationFailure(RuntimeError):
    pass


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        self.count += 1
        if not condition:
            raise ValidationFailure(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/mood_social_offline_auxiliary_v3_3_3.yaml"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    checks = Checks()
    try:
        config = load_offline_auxiliary_config(config_path, repository_root=root)
        inputs = load_offline_auxiliary_inputs(config)
        result = validate(root, config, inputs, checks)
    except (OfflineAuxiliaryError, ValidationFailure, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {"status": "pass", "checks": checks.count, **result},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def validate(root: Path, config: Any, inputs: Any, checks: Checks) -> dict[str, Any]:
    checks.require(config.model_path.is_file(), "MODEL-006 bundle is missing")
    checks.require(config.manifest_path.is_file(), "MODEL-006 manifest is missing")
    checks.require(config.report_directory.is_dir(), "MODEL-006 report is missing")
    manifest = _read_json(config.manifest_path)
    checks.require(
        manifest.get("manifest_version") == MANIFEST_VERSION, "manifest version drifted"
    )
    checks.require(
        manifest.get("training_version") == TRAINING_VERSION, "training version drifted"
    )
    checks.require(manifest.get("task_id") == TASK_ID, "manifest task ID drifted")
    checks.require(manifest.get("run_id") == RUN_ID, "manifest run ID drifted")
    checks.require(
        manifest.get("status") == "offline_auxiliary_not_active",
        "offline status drifted",
    )
    checks.require(
        manifest.get("model_sha256") == _sha256_file(config.model_path),
        "bundle hash drifted",
    )
    checks.require(
        manifest.get("training_config_sha256") == config.config_sha256,
        "config hash drifted",
    )
    checks.require(
        manifest.get("upstream_evaluation_core_sha256") == EVALUATION_CORE_SHA256,
        "EVAL core binding drifted",
    )
    boundary = manifest.get("production_boundary", {})
    checks.require(
        not any(bool(value) for value in boundary.values()),
        "offline boundary contains a true flag",
    )
    bundle = joblib.load(config.model_path)
    checks.require(isinstance(bundle, OfflineAuxiliaryBundle), "bundle type drifted")
    bundle.validate()
    checks.require(bundle.bundle_version == BUNDLE_VERSION, "bundle version drifted")
    checks.require(
        tuple(bundle.submodels) == tuple(MODEL_FAMILY_BY_TASK), "submodel order drifted"
    )
    report = config.report_directory
    artifacts = _read_json(report / "artifacts.json")
    rows = artifacts.get("artifacts")
    checks.require(isinstance(rows, list), "artifact list is missing")
    checks.require(
        artifacts.get("artifact_count") == len(rows), "artifact count drifted"
    )
    for row in rows:
        path = report / str(row["path"])
        checks.require(path.is_file(), f"report artifact missing: {row['path']}")
        checks.require(
            path.stat().st_size == int(row["bytes"]),
            f"artifact size drifted: {row['path']}",
        )
        checks.require(
            _sha256_file(path) == str(row["sha256"]),
            f"artifact hash drifted: {row['path']}",
        )
    expected_core = _rows_sha256([row for row in rows if row["path"] != "run.json"])
    checks.require(
        artifacts.get("report_core_sha256") == expected_core, "report core hash drifted"
    )
    predictions = pd.read_parquet(report / "predictions.parquet")
    required_columns = {
        "oof_version",
        "task_key",
        "dataset_id",
        "objective",
        "model_family",
        "prediction_id",
        "canonical_row_index",
        "global_participant_id",
        "outer_fold",
        "target_score",
        "target_normalized",
        "target_grade",
        "available",
        "prediction_normalized",
        "prediction_score",
        "prediction_grade",
        *(f"probability_grade_{grade}" for grade in GRADE_IDS),
    }
    checks.require(
        required_columns.issubset(predictions.columns), "OOF columns are incomplete"
    )
    checks.require(
        set(predictions["oof_version"]) == {OOF_VERSION}, "OOF version drifted"
    )
    expected_rows = sum(
        len(inputs.frames[task.split("::", 1)[0]]) for task in MODEL_FAMILY_BY_TASK
    )
    checks.require(len(predictions) == expected_rows, "OOF total row count drifted")
    metrics = _read_json(report / "metrics.json")
    metrics_by_task = {row["task_key"]: row for row in metrics["tasks"]}
    checks.require(
        tuple(metrics_by_task) == tuple(MODEL_FAMILY_BY_TASK),
        "metric task order drifted",
    )
    search = _read_json(report / "search_results.json")
    search_by_task = {row["task_key"]: row for row in search["tasks"]}
    checks.require(
        tuple(search_by_task) == tuple(MODEL_FAMILY_BY_TASK),
        "search task order drifted",
    )
    preprocessing = _read_json(report / "preprocessing.json")
    preprocessing_by_task = {row["task_key"]: row for row in preprocessing["tasks"]}
    checks.require(
        tuple(preprocessing_by_task) == tuple(MODEL_FAMILY_BY_TASK),
        "preprocessing task order drifted",
    )
    assignment_fold = inputs.assignments.set_index("global_participant_id")[
        "outer_fold"
    ].astype("int64")
    for task_key, family in MODEL_FAMILY_BY_TASK.items():
        dataset_id, objective = task_key.split("::", 1)
        part = predictions[predictions["task_key"].eq(task_key)].copy()
        canonical = build_task_frame(inputs, dataset_id)
        checks.require(len(part) == len(canonical), f"{task_key} OOF row count drifted")
        checks.require(
            not part["prediction_id"].duplicated().any(),
            f"{task_key} duplicate predictions",
        )
        checks.require(
            set(part["dataset_id"]) == {dataset_id}, f"{task_key} dataset drifted"
        )
        checks.require(
            set(part["objective"]) == {objective}, f"{task_key} objective drifted"
        )
        checks.require(
            set(part["model_family"]) == {family}, f"{task_key} family drifted"
        )
        part = part.sort_values("canonical_row_index", kind="stable").reset_index(
            drop=True
        )
        canonical = canonical.sort_values(
            "canonical_row_index", kind="stable"
        ).reset_index(drop=True)
        checks.require(
            part["prediction_id"].tolist() == canonical["prediction_id"].tolist(),
            f"{task_key} identity order drifted",
        )
        checks.require(
            np.array_equal(part["target_score"], canonical["target_score"]),
            f"{task_key} target score drifted",
        )
        checks.require(
            np.array_equal(part["target_grade"], canonical["target_grade"]),
            f"{task_key} target grade drifted",
        )
        expected_fold = (
            part["global_participant_id"].map(assignment_fold).to_numpy(dtype="int64")
        )
        checks.require(
            np.array_equal(part["outer_fold"], expected_fold),
            f"{task_key} outer folds drifted",
        )
        available = part[part["available"]].copy()
        unavailable = part[~part["available"]].copy()
        checks.require(not available.empty, f"{task_key} has no evaluable OOF rows")
        checks.require(
            unavailable["prediction_normalized"].isna().all(),
            f"{task_key} unavailable predictions are non-null",
        )
        if objective == "regression":
            checks.require(
                available["prediction_normalized"].between(0, 1).all(),
                f"{task_key} regression range drifted",
            )
            checks.require(
                np.allclose(
                    available["prediction_score"],
                    available["prediction_normalized"] * 27,
                ),
                f"{task_key} score scaling drifted",
            )
            recomputed = _regression_metrics(available)
        else:
            probability_columns = [f"probability_grade_{grade}" for grade in GRADE_IDS]
            prob = available[probability_columns].to_numpy(dtype="float64")
            checks.require(
                np.allclose(prob.sum(axis=1), 1.0, atol=1e-8, rtol=0),
                f"{task_key} probability sums drifted",
            )
            checks.require(
                np.array_equal(
                    np.argmax(prob, axis=1),
                    available["prediction_grade"].to_numpy(dtype="int64"),
                ),
                f"{task_key} predicted grades drifted",
            )
            recomputed = _multiclass_metrics(available)
        _compare_nested(
            recomputed,
            metrics_by_task[task_key]["overall"],
            checks,
            prefix=f"{task_key}.metrics",
        )
        task_search = search_by_task[task_key]
        checks.require(
            len(task_search["outer_searches"]) == 5,
            f"{task_key} search fold count drifted",
        )
        expected_candidates = (
            20
            if family == "elasticnet_regression"
            else 16
            if family == "catboost_regressor"
            else 8
        )
        for outer in task_search["outer_searches"]:
            checks.require(
                outer["candidate_count"] == expected_candidates,
                f"{task_key} candidate count drifted",
            )
            checks.require(
                bool(outer["outer_test_metrics_not_used_for_selection"]),
                f"{task_key} outer-test selection guard drifted",
            )
        production = task_search["production_selection"]
        checks.require(
            production["candidate_id"]
            == metrics_by_task[task_key]["final_candidate_id"],
            f"{task_key} production selection drifted",
        )
        submodel = bundle.submodels[task_key]
        checks.require(
            dict(submodel.hyperparameters) == dict(production["params"]),
            f"{task_key} bundle parameters drifted",
        )
        checks.require(
            not any(
                "phq" in name.lower()
                or "target" in name.lower()
                or name.startswith("source__")
                for name in submodel.input_features
            ),
            f"{task_key} forbidden input feature detected",
        )
        folds = preprocessing_by_task[task_key]["folds"]
        checks.require(len(folds) == 6, f"{task_key} preprocessing scope count drifted")
        checks.require(
            folds[-1]["scope"] == "production",
            f"{task_key} production preprocessing missing",
        )
        sample = inputs.frames[dataset_id].head(5)
        smoke = submodel.predict_frame(sample)
        checks.require(
            smoke["available"].shape == (len(sample),),
            f"{task_key} smoke availability drifted",
        )
        if objective == "regression" and smoke["available"].any():
            checks.require(
                np.isfinite(smoke["prediction_normalized"][smoke["available"]]).all(),
                f"{task_key} smoke prediction invalid",
            )
        if objective == "multiclass" and smoke["available"].any():
            checks.require(
                np.allclose(
                    smoke["probabilities"][smoke["available"]].sum(axis=1), 1.0
                ),
                f"{task_key} smoke probabilities invalid",
            )
    checks.require(
        manifest["submodel_count"] == len(MODEL_FAMILY_BY_TASK),
        "manifest submodel count drifted",
    )
    checks.require(
        set(manifest["submodels"]) == set(MODEL_FAMILY_BY_TASK),
        "manifest submodel set drifted",
    )
    checks.require(
        _sha256_file(config.manifest_path) == _sha256_file(config.manifest_path),
        "manifest hash unreadable",
    )
    return {
        "run_id": RUN_ID,
        "oof_rows": len(predictions),
        "available_rows": int(predictions["available"].sum()),
        "submodels": len(bundle.submodels),
        "model_sha256": _sha256_file(config.model_path),
        "manifest_sha256": _sha256_file(config.manifest_path),
        "report_core_sha256": expected_core,
    }


def _regression_weights(frame: pd.DataFrame) -> np.ndarray:
    ids = frame["global_participant_id"].astype(str)
    counts = ids.value_counts()
    weight = ids.map(lambda value: 1.0 / counts[value]).to_numpy(dtype="float64")
    return weight / weight.mean()


def _multiclass_weights(frame: pd.DataFrame) -> np.ndarray:
    grade = frame["target_grade"].astype("int64")
    ids = frame["global_participant_id"].astype(str)
    unit = ids + "::" + grade.astype(str)
    present = sorted(set(grade))
    weight = np.zeros(len(frame), dtype="float64")
    for value in present:
        mask = grade.to_numpy() == value
        units = unit[mask]
        counts = units.value_counts()
        indices = np.flatnonzero(mask)
        for index, key in zip(indices, units.tolist(), strict=True):
            weight[index] = (1.0 / len(present)) / len(counts) / counts[key]
    return weight / weight.mean()


def _regression_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    y = frame["target_normalized"].to_numpy(dtype="float64")
    p = frame["prediction_normalized"].to_numpy(dtype="float64")
    w = _regression_weights(frame)
    w = w / w.sum()
    error = p - y
    mae = float(np.sum(w * np.abs(error)))
    rmse = float(np.sqrt(np.sum(w * error**2)))
    stat = spearmanr(y, p).statistic if np.unique(p).size > 1 else math.nan
    return {
        "mae_normalized": mae,
        "rmse_normalized": rmse,
        "mae_score": mae * 27,
        "rmse_score": rmse * 27,
        "spearman": float(stat) if math.isfinite(stat) else None,
    }


def _multiclass_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    y = frame["target_grade"].to_numpy(dtype="int64")
    prob = frame[[f"probability_grade_{grade}" for grade in GRADE_IDS]].to_numpy(
        dtype="float64"
    )
    pred = np.argmax(prob, axis=1)
    w = _multiclass_weights(frame)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=list(GRADE_IDS), sample_weight=w, zero_division=0
    )
    return {
        "macro_f1": float(
            f1_score(
                y,
                pred,
                labels=list(GRADE_IDS),
                average="macro",
                sample_weight=w,
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y, pred, sample_weight=w)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred, sample_weight=w)),
        "log_loss": float(log_loss(y, prob, labels=list(GRADE_IDS), sample_weight=w)),
        "ordinal_mae": float(np.average(np.abs(pred - y), weights=w)),
        "per_class": [
            {
                "grade": grade,
                "precision": float(precision[grade]),
                "recall": float(recall[grade]),
                "f1": float(f1[grade]),
                "weighted_support": float(support[grade]),
            }
            for grade in GRADE_IDS
        ],
        "confusion_matrix": confusion_matrix(
            y, pred, labels=list(GRADE_IDS), sample_weight=w
        ).tolist(),
    }


def _compare_nested(actual: Any, expected: Any, checks: Checks, *, prefix: str) -> None:
    if isinstance(actual, Mapping):
        checks.require(set(actual) == set(expected), f"{prefix} keys drifted")
        for key in actual:
            _compare_nested(
                actual[key], expected[key], checks, prefix=f"{prefix}.{key}"
            )
    elif isinstance(actual, list):
        checks.require(len(actual) == len(expected), f"{prefix} length drifted")
        for index, value in enumerate(actual):
            _compare_nested(value, expected[index], checks, prefix=f"{prefix}[{index}]")
    elif actual is None:
        checks.require(expected is None, f"{prefix} null drifted")
    elif isinstance(actual, (int, float)):
        checks.require(
            math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-10),
            f"{prefix} value drifted",
        )
    else:
        checks.require(actual == expected, f"{prefix} value drifted")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rows_sha256(rows: list[Mapping[str, Any]]) -> str:
    payload = "".join(
        f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n" for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
