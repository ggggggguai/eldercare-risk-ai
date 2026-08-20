"""External auxiliary experiments for FORECAST-OPT-001E."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ALPHAS = (0.1, 1.0, 10.0, 100.0)


class ForecastExternalAuxiliaryError(ValueError):
    """Raised when an external auxiliary protocol is invalid."""


@dataclass
class RidgeAuxiliaryModel:
    feature_names: tuple[str, ...]
    alpha: float
    pipeline: Pipeline

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        missing = sorted(set(self.feature_names).difference(frame.columns))
        if missing:
            raise ForecastExternalAuxiliaryError(
                f"auxiliary features are missing: {missing}"
            )
        return np.asarray(
            self.pipeline.predict(frame.loc[:, list(self.feature_names)]),
            dtype="float64",
        )


def _pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def _metrics(target: Sequence[float], prediction: Sequence[float]) -> dict[str, float]:
    y = np.asarray(target, dtype="float64")
    p = np.asarray(prediction, dtype="float64")
    error = p - y
    ranked_y = pd.Series(y).rank(method="average").to_numpy(dtype="float64")
    ranked_p = pd.Series(p).rank(method="average").to_numpy(dtype="float64")
    spearman = (
        float(np.corrcoef(ranked_y, ranked_p)[0, 1])
        if np.std(ranked_p) > 0 and np.std(ranked_y) > 0
        else np.nan
    )
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "spearman": spearman,
        "target_mean": float(np.mean(y)),
        "prediction_mean": float(np.mean(p)),
    }


def _select_alpha(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    target_name: str,
    groups: pd.Series | None,
    splits: int,
) -> tuple[float, list[dict[str, float]]]:
    if groups is not None:
        unique_groups = groups.nunique()
        splitter = GroupKFold(n_splits=min(splits, unique_groups))
        split_iterator = splitter.split(frame, groups=groups)
    else:
        splitter = KFold(
            n_splits=min(splits, len(frame)), shuffle=True, random_state=20260728
        )
        split_iterator = splitter.split(frame)
    partitions = list(split_iterator)
    records = []
    for alpha_index, alpha in enumerate(ALPHAS):
        errors = []
        for train_index, validation_index in partitions:
            model = _pipeline(alpha)
            model.fit(
                frame.iloc[train_index][list(feature_names)],
                frame.iloc[train_index][target_name],
            )
            prediction = model.predict(
                frame.iloc[validation_index][list(feature_names)]
            )
            target = frame.iloc[validation_index][target_name].to_numpy(dtype="float64")
            errors.extend(np.abs(prediction - target).tolist())
        records.append(
            {
                "alpha_index": alpha_index,
                "alpha": alpha,
                "inner_mae": float(np.mean(errors)),
            }
        )
    selected = min(records, key=lambda row: (row["inner_mae"], row["alpha_index"]))
    return float(selected["alpha"]), records


def nested_group_ridge_oof(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    target_name: str,
    group_name: str,
    *,
    outer_splits: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate participant-isolated OOF predictions with inner alpha search."""

    if frame[group_name].nunique() < outer_splits:
        raise ForecastExternalAuxiliaryError("too few groups for nested GroupKFold")
    splitter = GroupKFold(n_splits=outer_splits)
    output = frame[[group_name, target_name]].copy()
    output["prediction"] = np.nan
    output["null_prediction"] = np.nan
    output["outer_fold_id"] = -1
    fold_audits = []
    for outer_fold, (train_index, test_index) in enumerate(
        splitter.split(frame, groups=frame[group_name])
    ):
        train = frame.iloc[train_index]
        test = frame.iloc[test_index]
        if set(train[group_name]).intersection(test[group_name]):
            raise ForecastExternalAuxiliaryError(
                "participant leaked across external fold"
            )
        alpha, inner_records = _select_alpha(
            train,
            feature_names,
            target_name,
            train[group_name],
            splits=4,
        )
        model = _pipeline(alpha)
        model.fit(train[list(feature_names)], train[target_name])
        output.loc[frame.index[test_index], "prediction"] = model.predict(
            test[list(feature_names)]
        )
        output.loc[frame.index[test_index], "null_prediction"] = float(
            train[target_name].mean()
        )
        output.loc[frame.index[test_index], "outer_fold_id"] = outer_fold
        fold_audits.append(
            {
                "outer_fold_id": outer_fold,
                "selected_alpha": alpha,
                "train_participant_count": train[group_name].nunique(),
                "test_participant_count": test[group_name].nunique(),
                "inner_search": inner_records,
            }
        )
    if output["prediction"].isna().any() or output["outer_fold_id"].lt(0).any():
        raise ForecastExternalAuxiliaryError("external OOF coverage is incomplete")
    metrics = _metrics(output[target_name], output["prediction"])
    null_metrics = _metrics(output[target_name], output["null_prediction"])
    return output.reset_index(drop=True), {
        "target": target_name,
        "row_count": len(output),
        "participant_count": frame[group_name].nunique(),
        "features": list(feature_names),
        "metrics": metrics,
        "null_metrics": null_metrics,
        "mae_delta_vs_null": metrics["mae"] - null_metrics["mae"],
        "fold_audits": fold_audits,
    }


def nested_leave_one_out_ridge(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    target_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run nested LOOCV for the 23-person OBF MADRS change task."""

    if len(frame) < 10:
        raise ForecastExternalAuxiliaryError("too few rows for nested LOOCV")
    output = frame[
        ["global_participant_id", "collection_start_year", target_name]
    ].copy()
    output["prediction"] = np.nan
    output["null_prediction"] = np.nan
    output["selected_alpha"] = np.nan
    for train_index, test_index in LeaveOneOut().split(frame):
        train = frame.iloc[train_index]
        test = frame.iloc[test_index]
        alpha, _ = _select_alpha(train, feature_names, target_name, None, splits=5)
        model = _pipeline(alpha)
        model.fit(train[list(feature_names)], train[target_name])
        output.loc[frame.index[test_index], "prediction"] = model.predict(
            test[list(feature_names)]
        )
        output.loc[frame.index[test_index], "null_prediction"] = float(
            train[target_name].mean()
        )
        output.loc[frame.index[test_index], "selected_alpha"] = alpha
    metrics = _metrics(output[target_name], output["prediction"])
    null_metrics = _metrics(output[target_name], output["null_prediction"])
    return output.reset_index(drop=True), {
        "target": target_name,
        "row_count": len(output),
        "features": list(feature_names),
        "metrics": metrics,
        "null_metrics": null_metrics,
        "mae_delta_vs_null": metrics["mae"] - null_metrics["mae"],
    }


def leave_year_out_ridge(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    target_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    years = sorted(frame["collection_start_year"].astype(int).unique().tolist())
    output = frame[
        ["global_participant_id", "collection_start_year", target_name]
    ].copy()
    output["prediction"] = np.nan
    output["null_prediction"] = np.nan
    audits = []
    for year in years:
        test_mask = frame["collection_start_year"].astype(int).eq(year)
        train = frame.loc[~test_mask]
        test = frame.loc[test_mask]
        if train.empty or test.empty:
            continue
        alpha, _ = _select_alpha(train, feature_names, target_name, None, splits=5)
        model = _pipeline(alpha)
        model.fit(train[list(feature_names)], train[target_name])
        output.loc[test.index, "prediction"] = model.predict(test[list(feature_names)])
        output.loc[test.index, "null_prediction"] = float(train[target_name].mean())
        audits.append(
            {
                "held_out_year": year,
                "train_rows": len(train),
                "test_rows": len(test),
                "selected_alpha": alpha,
            }
        )
    complete = output["prediction"].notna()
    if not complete.all():
        raise ForecastExternalAuxiliaryError("leave-year-out coverage is incomplete")
    metrics = _metrics(output[target_name], output["prediction"])
    null_metrics = _metrics(output[target_name], output["null_prediction"])
    return output.reset_index(drop=True), {
        "years": years,
        "metrics": metrics,
        "null_metrics": null_metrics,
        "mae_delta_vs_null": metrics["mae"] - null_metrics["mae"],
        "audits": audits,
    }


def fit_final_ridge(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    target_name: str,
    group_name: str,
) -> RidgeAuxiliaryModel:
    alpha, _ = _select_alpha(
        frame, feature_names, target_name, frame[group_name], splits=5
    )
    model = _pipeline(alpha)
    model.fit(frame[list(feature_names)], frame[target_name])
    return RidgeAuxiliaryModel(tuple(feature_names), alpha, model)


def apply_conscious_proxy_to_psyche(
    model: RidgeAuxiliaryModel,
    samples_by_task: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Apply a fixed external model without reading any PSYCHE-D label column."""

    parts = []
    anchor_names = [f"{name}__anchor" for name in model.feature_names]
    for task_id, source in samples_by_task.items():
        required = {
            "global_participant_id",
            "target_window_id",
            "outer_fold_id",
            *anchor_names,
        }
        missing = sorted(required.difference(source.columns))
        if missing:
            raise ForecastExternalAuxiliaryError(
                f"PSYCHE semantic bridge is missing: {missing}"
            )
        safe = source.loc[
            :,
            [
                "global_participant_id",
                "target_window_id",
                "outer_fold_id",
                *anchor_names,
            ],
        ].copy()
        safe = safe.rename(
            columns=dict(zip(anchor_names, model.feature_names, strict=True))
        )
        output = safe[
            ["global_participant_id", "target_window_id", "outer_fold_id"]
        ].copy()
        output.insert(0, "task_id", task_id)
        output["conscious_phq9_proxy"] = model.predict(safe)
        output["proxy_source"] = "fixed_external_conscious_model"
        parts.append(output)
    return pd.concat(parts, ignore_index=True)


__all__ = [
    "ALPHAS",
    "ForecastExternalAuxiliaryError",
    "RidgeAuxiliaryModel",
    "apply_conscious_proxy_to_psyche",
    "fit_final_ridge",
    "leave_year_out_ridge",
    "nested_group_ridge_oof",
    "nested_leave_one_out_ridge",
]
