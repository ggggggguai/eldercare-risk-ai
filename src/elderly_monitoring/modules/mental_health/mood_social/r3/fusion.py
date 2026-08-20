"""Strict second-level route fusion selection for OPT-V333-005."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from elderly_monitoring.modules.mental_health.mood_social.r3.baseline import (
    legacy_train_and_predict,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.contract import (
    R3_PROTOCOL_VERSION,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.data import (
    eligible_rows,
    inner_fold_series,
    load_r3_training_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.modeling import (
    ModelSpec,
    WeightSpec,
    binary_metrics,
    sample_weights,
    weighted_binary_metrics,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.selection import (
    EXPERT_DEFINITIONS,
    _fit_predict_members,
)


FUSION_VERSION = "mood-social-v3.3.3-r3-strict-route-fusion-v1"
DEFAULT_SELECTION_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-003-004-selection/selected_experts.json"
)
DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-005-fusion"
)
EPSILON = 1.0e-6
PARTICIPANT_WEIGHT = WeightSpec("participant", participant_equal=True)
EXPERT_IDS = tuple(item.expert_id for item in EXPERT_DEFINITIONS)


@dataclass(frozen=True)
class FusionSpec:
    candidate_id: str
    kind: str
    params: Mapping[str, float]


def fusion_specs() -> tuple[FusionSpec, ...]:
    result: list[FusionSpec] = []
    for profile_weight in (0.0, 0.1, 0.2, 0.3, 0.5):
        result.append(
            FusionSpec(
                f"direct_profile{int(profile_weight * 100):02d}",
                "direct",
                {"profile_weight": profile_weight},
            )
        )
    for old_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        result.extend(
            [
                FusionSpec(
                    f"convex_probability_old{int(old_weight * 100):03d}",
                    "convex_probability",
                    {"old_weight": old_weight, "profile_weight": 0.2},
                ),
                FusionSpec(
                    f"convex_logit_old{int(old_weight * 100):03d}",
                    "convex_logit",
                    {"old_weight": old_weight, "profile_weight": 0.2},
                ),
            ]
        )
    for l2 in (0.1, 1.0, 10.0):
        result.append(
            FusionSpec(
                f"residual_l2_{str(l2).replace('.', 'p')}",
                "residual",
                {"l2": l2, "profile_weight": 0.2},
            )
        )
    for c_value in (0.1, 1.0, 10.0):
        result.append(FusionSpec(f"stack_c{str(c_value).replace('.', 'p')}", "stack", {"C": c_value}))
    return tuple(result)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member(payload: Mapping[str, Any]) -> ModelSpec:
    return ModelSpec(
        candidate_id=str(payload["candidate_id"]),
        family=str(payload["family"]),  # type: ignore[arg-type]
        params=dict(payload["params"]),
        seed=int(payload.get("seed", 20260728)),
    )


def _expert_choices(selection: Mapping[str, Any], outer_fold: int) -> dict[str, dict[str, Any]]:
    return {
        expert_id: dict(selection["outer_choices"][str(outer_fold)][expert_id])
        for expert_id in EXPERT_IDS
    }


def candidate_expert_train_and_predict(
    train: pd.DataFrame,
    train_fold: pd.Series,
    predict: pd.DataFrame,
    choices: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First-level OOF expert features and full-train test features."""

    fold = pd.Series(train_fold, index=train.index).astype(int)
    train_output = train[
        ["r3_row_id", "dataset_id", "global_participant_id", "binary_target", "route_pattern"]
    ].copy()
    train_output["fusion_validation_fold"] = fold.to_numpy()
    predict_output = predict[
        ["r3_row_id", "dataset_id", "global_participant_id", "binary_target", "route_pattern"]
    ].copy()
    for definition in EXPERT_DEFINITIONS:
        choice = choices[definition.expert_id]
        members = tuple(_member(item) for item in choice["members"])
        weight_spec = WeightSpec(**dict(choice["weight_spec"]))
        blend_kind = str(choice.get("blend_kind", "probability"))
        train_probability = np.full(len(train), np.nan, dtype=float)
        for validation_fold in sorted(fold.unique()):
            fit_rows = train.loc[fold.ne(validation_fold)]
            validation_rows = train.loc[fold.eq(validation_fold)]
            fit_rows = fit_rows.loc[
                fit_rows["route_pattern"].isin(definition.required_routes)
            ]
            available = validation_rows["route_pattern"].isin(
                definition.required_routes
            ).to_numpy()
            if available.any():
                values = _fit_predict_members(
                    fit_rows,
                    validation_rows.loc[available],
                    definition.features,
                    members,
                    weight_spec,
                    blend_kind,
                )
                positions = np.flatnonzero(fold.eq(validation_fold).to_numpy())
                train_probability[positions[available]] = values
        available_train = train["route_pattern"].isin(definition.required_routes).to_numpy()
        if not np.isfinite(train_probability[available_train]).all():
            raise ValueError(f"candidate {definition.expert_id} train OOF incomplete")
        train_output[f"expert_{definition.expert_id}_probability"] = train_probability

        full_train = train.loc[
            train["route_pattern"].isin(definition.required_routes)
        ]
        predict_probability = np.full(len(predict), np.nan, dtype=float)
        available_predict = predict["route_pattern"].isin(
            definition.required_routes
        ).to_numpy()
        if available_predict.any():
            predict_probability[available_predict] = _fit_predict_members(
                full_train,
                predict.loc[available_predict],
                definition.features,
                members,
                weight_spec,
                blend_kind,
            )
        predict_output[f"expert_{definition.expert_id}_probability"] = predict_probability
    return train_output.reset_index(drop=True), predict_output.reset_index(drop=True)


def _logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    return np.log(value / (1.0 - value))


def _direct_probability(frame: pd.DataFrame, profile_weight: float) -> np.ndarray:
    result = np.full(len(frame), np.nan, dtype=float)
    route = frame["route_pattern"].astype(str).to_numpy()
    values = {
        expert_id: frame[f"expert_{expert_id}_probability"].to_numpy(float)
        for expert_id in EXPERT_IDS
    }
    for index, pattern in enumerate(route):
        candidates: list[tuple[str, float]]
        if pattern in {"111", "110"}:
            candidates = [("joint", 1.0 - profile_weight)]
            if pattern.endswith("1"):
                candidates.append(("profile", profile_weight))
        elif pattern.startswith("1"):
            candidates = [("activity", 1.0 - profile_weight)]
            if pattern.endswith("1"):
                candidates.append(("profile", profile_weight))
        elif pattern[1] == "1":
            candidates = [("sleep", 1.0 - profile_weight)]
            if pattern.endswith("1"):
                candidates.append(("profile", profile_weight))
        else:
            candidates = [("profile", 1.0)]
        observed = [(values[name][index], weight) for name, weight in candidates if np.isfinite(values[name][index])]
        if not observed:
            # This can only happen if a route/expert contract drifted.
            raise ValueError(f"direct route {pattern} has no expert evidence")
        weights = np.asarray([item[1] for item in observed], dtype=float)
        if weights.sum() <= 0.0:
            weights = np.ones_like(weights)
        result[index] = float(
            np.average(np.asarray([item[0] for item in observed]), weights=weights)
        )
    return np.clip(result, EPSILON, 1.0 - EPSILON)


def _stack_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = [_logit(frame["baseline_probability"].to_numpy(float))]
    for expert_id in EXPERT_IDS:
        probability = frame[f"expert_{expert_id}_probability"].to_numpy(float)
        columns.append(_logit(np.nan_to_num(probability, nan=0.5)))
    return np.column_stack(columns)


@dataclass
class _ResidualModel:
    intercept: float
    coefficient: float

    def predict(self, old_probability: np.ndarray, direct_probability: np.ndarray) -> np.ndarray:
        value = _logit(old_probability) + self.intercept + self.coefficient * _logit(direct_probability)
        return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _fit_residual(
    frame: pd.DataFrame, direct: np.ndarray, l2: float
) -> _ResidualModel:
    target = frame["binary_target"].to_numpy(float)
    old_logit = _logit(frame["baseline_probability"].to_numpy(float))
    new_logit = _logit(direct)
    weight = sample_weights(frame, PARTICIPANT_WEIGHT)

    parameter = np.zeros(2, dtype=float)
    normalized_weight = weight / weight.sum()
    for _ in range(100):
        linear = old_logit + parameter[0] + parameter[1] * new_logit
        probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -30.0, 30.0)))
        error = normalized_weight * (probability - target)
        gradient = np.asarray(
            [error.sum(), np.sum(error * new_logit) + l2 * parameter[1]]
        )
        curvature = normalized_weight * probability * (1.0 - probability)
        hessian = np.asarray(
            [
                [curvature.sum(), np.sum(curvature * new_logit)],
                [
                    np.sum(curvature * new_logit),
                    np.sum(curvature * np.square(new_logit)) + l2,
                ],
            ]
        )
        try:
            step = np.linalg.solve(hessian + np.eye(2) * 1.0e-8, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError("residual Newton Hessian is singular") from exc
        parameter -= step
        if float(np.max(np.abs(step))) < 1.0e-8:
            break
    if not np.isfinite(parameter).all():
        raise ValueError("residual Newton optimizer produced non-finite parameters")
    return _ResidualModel(float(parameter[0]), float(parameter[1]))


def _fit_predict_fusion(
    spec: FusionSpec,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> np.ndarray:
    profile_weight = float(spec.params.get("profile_weight", 0.2))
    direct_train = _direct_probability(train, profile_weight)
    direct_validation = _direct_probability(validation, profile_weight)
    if spec.kind == "direct":
        return direct_validation
    if spec.kind in {"convex_probability", "convex_logit"}:
        old_weight = float(spec.params["old_weight"])
        old = validation["baseline_probability"].to_numpy(float)
        if spec.kind == "convex_probability":
            result = old_weight * old + (1.0 - old_weight) * direct_validation
        else:
            value = old_weight * _logit(old) + (1.0 - old_weight) * _logit(direct_validation)
            result = 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))
        return np.clip(result, EPSILON, 1.0 - EPSILON)
    if spec.kind == "residual":
        model = _fit_residual(train, direct_train, float(spec.params["l2"]))
        return model.predict(
            validation["baseline_probability"].to_numpy(float), direct_validation
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
        return np.clip(
            model.predict_proba(_stack_matrix(validation))[:, 1],
            EPSILON,
            1.0 - EPSILON,
        )
    raise ValueError(f"unknown fusion kind: {spec.kind}")


def _merge_meta_features(
    new: pd.DataFrame,
    old: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["r3_row_id", "baseline_probability"]
    result = new.merge(old[columns], on="r3_row_id", how="left", validate="one_to_one")
    if result["baseline_probability"].isna().any():
        raise ValueError("legacy baseline meta feature is incomplete")
    return result


def select_fusion_for_outer_context(
    frame: pd.DataFrame,
    *,
    outer_fold: int,
    choices: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[FusionSpec] | None = None,
    fold_checkpoint_directory: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    outer_train = frame.loc[frame["outer_fold"].ne(int(outer_fold))].copy()
    inner = inner_fold_series(frame, outer_fold).loc[outer_train.index].astype(int)
    specs = tuple(candidates or fusion_specs())
    candidate_parts: dict[str, list[pd.DataFrame]] = {
        spec.candidate_id: [] for spec in specs
    }
    for fusion_validation_fold in sorted(inner.unique()):
        train_cache: Path | None = None
        validation_cache: Path | None = None
        if fold_checkpoint_directory is not None:
            fold_checkpoint_directory.mkdir(parents=True, exist_ok=True)
            train_cache = fold_checkpoint_directory / f"fold{int(fusion_validation_fold)}_train.parquet"
            validation_cache = fold_checkpoint_directory / f"fold{int(fusion_validation_fold)}_validation.parquet"
        if train_cache is not None and train_cache.exists() and validation_cache is not None and validation_cache.exists():
            train_meta = pd.read_parquet(train_cache)
            validation_meta = pd.read_parquet(validation_cache)
        else:
            fusion_train = outer_train.loc[inner.ne(fusion_validation_fold)].copy()
            fusion_validation = outer_train.loc[inner.eq(fusion_validation_fold)].copy()
            subfold = inner.loc[fusion_train.index]
            new_train, new_validation = candidate_expert_train_and_predict(
                fusion_train, subfold, fusion_validation, choices
            )
            old_train, _, old_validation_probability = legacy_train_and_predict(
                fusion_train, subfold, fusion_validation
            )
            train_meta = _merge_meta_features(new_train, old_train)
            validation_old = fusion_validation[["r3_row_id"]].copy()
            validation_old["baseline_probability"] = old_validation_probability
            validation_meta = _merge_meta_features(new_validation, validation_old)
            if train_cache is not None and validation_cache is not None:
                train_meta.to_parquet(train_cache, index=False)
                validation_meta.to_parquet(validation_cache, index=False)
        for spec in specs:
            probability = _fit_predict_fusion(spec, train_meta, validation_meta)
            part = validation_meta[
                [
                    "r3_row_id",
                    "dataset_id",
                    "global_participant_id",
                    "binary_target",
                    "route_pattern",
                    "baseline_probability",
                ]
            ].copy()
            part["fusion_validation_fold"] = int(fusion_validation_fold)
            part["candidate_id"] = spec.candidate_id
            part["probability"] = probability
            candidate_parts[spec.candidate_id].append(part)
    records: list[dict[str, Any]] = []
    candidate_oof: dict[str, pd.DataFrame] = {}
    for spec in specs:
        oof = pd.concat(candidate_parts[spec.candidate_id], ignore_index=True)
        if len(oof) != len(outer_train) or oof["r3_row_id"].duplicated().any():
            raise ValueError(f"fusion candidate OOF incomplete: {spec.candidate_id}")
        natural = binary_metrics(oof["binary_target"], oof["probability"])
        participant = weighted_binary_metrics(
            oof["binary_target"],
            oof["probability"],
            sample_weights(oof, PARTICIPANT_WEIGHT),
        )
        records.append(
            {
                "outer_context": int(outer_fold),
                "candidate_id": spec.candidate_id,
                "kind": spec.kind,
                "params": dict(spec.params),
                "row_count": int(len(oof)),
                "natural_auprc": natural["auprc"],
                "natural_auroc": natural["auroc"],
                "natural_brier": natural["brier"],
                "participant_auprc": participant["auprc"],
                "participant_auroc": participant["auroc"],
                "participant_brier": participant["brier"],
                "selection_score": float(
                    0.6 * natural["auprc"] + 0.4 * participant["auprc"]
                ),
                "status": "pass",
            }
        )
        candidate_oof[spec.candidate_id] = oof
    search = pd.DataFrame(records)
    selected_row = search.sort_values(
        ["selection_score", "participant_auprc", "natural_brier", "candidate_id"],
        ascending=[False, False, True, True],
        kind="stable",
    ).iloc[0]
    selected_id = str(selected_row["candidate_id"])
    selected_oof = candidate_oof[selected_id].copy()
    choice = {
        "outer_context": int(outer_fold),
        "selected_candidate_id": selected_id,
        "selected_spec": asdict(next(spec for spec in specs if spec.candidate_id == selected_id)),
        "selection_metric": "0.6 natural inner-OOF AUPRC + 0.4 participant-equal inner-OOF AUPRC",
        "outer_results_opened": False,
    }
    return search, selected_oof, choice


def run_fusion_selection(
    *,
    repository_root: Path,
    selection_path: Path | None = None,
    report_directory: Path | None = None,
    outer_folds: Iterable[int] = range(5),
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    selection_file = selection_path or root / DEFAULT_SELECTION_RELATIVE
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    output = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output / "_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    frame = eligible_rows(load_r3_training_frame(repository_root=root))
    searches: list[pd.DataFrame] = []
    selected_tables: list[pd.DataFrame] = []
    choices: dict[str, Any] = {}
    for outer_fold in tuple(int(value) for value in outer_folds):
        search_path = checkpoint_root / f"outer{outer_fold}_search.parquet"
        oof_path = checkpoint_root / f"outer{outer_fold}_selected_oof.parquet"
        choice_path = checkpoint_root / f"outer{outer_fold}_choice.json"
        if search_path.exists() and oof_path.exists() and choice_path.exists() and not overwrite:
            search = pd.read_parquet(search_path)
            selected_oof = pd.read_parquet(oof_path)
            choice = json.loads(choice_path.read_text(encoding="utf-8"))
        else:
            search, selected_oof, choice = select_fusion_for_outer_context(
                frame,
                outer_fold=outer_fold,
                choices=_expert_choices(selection, outer_fold),
                fold_checkpoint_directory=checkpoint_root / f"outer{outer_fold}_meta",
            )
            search.to_parquet(search_path, index=False)
            selected_oof.to_parquet(oof_path, index=False)
            choice_path.write_text(
                json.dumps(choice, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
        searches.append(search)
        selected_oof.insert(0, "outer_context", outer_fold)
        selected_tables.append(selected_oof)
        choices[str(outer_fold)] = choice
    search_table = pd.concat(searches, ignore_index=True)
    selected_table = pd.concat(selected_tables, ignore_index=True)
    search_path = output / "fusion_candidate_search.parquet"
    oof_path = output / "selected_fusion_inner_oof.parquet"
    choices_path = output / "selected_fusions.json"
    for path in (search_path, oof_path, choices_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite fusion artifact: {path}")
    search_table.to_parquet(search_path, index=False)
    selected_table.to_parquet(oof_path, index=False)
    choices_path.write_text(
        json.dumps(
            {
                "protocol_version": R3_PROTOCOL_VERSION,
                "fusion_version": FUSION_VERSION,
                "outer_results_opened": False,
                "outer_choices": choices,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol_version": R3_PROTOCOL_VERSION,
        "fusion_version": FUSION_VERSION,
        "task_id": "OPT-V333-005",
        "status": "pass",
        "outer_results_opened": False,
        "strict_second_level_oof": True,
        "selection_path": selection_file.relative_to(root).as_posix(),
        "selection_sha256": _sha256_file(selection_file),
        "candidate_search": {
            "path": search_path.relative_to(root).as_posix(),
            "sha256": _sha256_file(search_path),
            "rows": int(len(search_table)),
        },
        "selected_inner_oof": {
            "path": oof_path.relative_to(root).as_posix(),
            "sha256": _sha256_file(oof_path),
            "rows": int(len(selected_table)),
        },
        "selected_fusions": {
            "path": choices_path.relative_to(root).as_posix(),
            "sha256": _sha256_file(choices_path),
        },
    }
    manifest_path = output / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "FUSION_VERSION",
    "FusionSpec",
    "candidate_expert_train_and_predict",
    "fusion_specs",
    "run_fusion_selection",
    "select_fusion_for_outer_context",
]
