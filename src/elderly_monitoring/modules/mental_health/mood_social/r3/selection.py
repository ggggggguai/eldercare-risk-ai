"""Outer-sealed expert selection for OPT-V333-003/004.

This module never predicts an outer-test row.  Every reported score is an
inner-OOF score inside one frozen outer-training partition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r3.contract import (
    ACTIVITY_MODEL_FEATURES,
    JOINT_MODEL_FEATURES,
    PROFILE_MODEL_FEATURES,
    R3_PROTOCOL_VERSION,
    SLEEP_MODEL_FEATURES,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.data import (
    eligible_rows,
    inner_fold_series,
    load_r3_training_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.modeling import (
    CandidateUnavailable,
    DEFAULT_WEIGHT_SPECS,
    ModelSpec,
    WeightSpec,
    binary_metrics,
    default_model_specs,
    fit_classifier,
    predict_probability,
    sample_weights,
    weighted_binary_metrics,
)


DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-003-004-selection"
)
TRAINING_CONFIG_RELATIVE = Path(
    "configs/training/mood_social_v3_3_3_r3.yaml"
)


@dataclass(frozen=True)
class ExpertDefinition:
    expert_id: str
    features: tuple[str, ...]
    required_routes: tuple[str, ...]


EXPERT_DEFINITIONS: tuple[ExpertDefinition, ...] = (
    ExpertDefinition(
        "activity", ACTIVITY_MODEL_FEATURES, ("100", "101", "110", "111")
    ),
    ExpertDefinition("sleep", SLEEP_MODEL_FEATURES, ("010", "011", "110", "111")),
    ExpertDefinition("joint", JOINT_MODEL_FEATURES, ("110", "111")),
    ExpertDefinition(
        "profile", PROFILE_MODEL_FEATURES, ("001", "011", "101", "111")
    ),
)


@dataclass(frozen=True)
class CandidateChoice:
    candidate_id: str
    members: tuple[ModelSpec, ...]
    weight_spec: WeightSpec
    blend_kind: str = "probability"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "members": [member.to_dict() for member in self.members],
            "weight_spec": self.weight_spec.to_dict(),
            "blend_kind": self.blend_kind,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite selection artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _applicable(frame: pd.DataFrame, definition: ExpertDefinition) -> pd.DataFrame:
    return frame[frame["route_pattern"].isin(definition.required_routes)].copy()


def _ensemble_candidates(specs: Sequence[ModelSpec]) -> list[tuple[str, tuple[ModelSpec, ...]]]:
    candidates = [(spec.candidate_id, (spec,)) for spec in specs]
    for family in ("lightgbm", "catboost", "xgboost"):
        members = tuple(spec for spec in specs if spec.family == family)
        if len(members) >= 2:
            candidates.append((f"{family}_multiseed_ensemble", members))
    return candidates


def _fit_predict_members(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
    members: Sequence[ModelSpec],
    weight_spec: WeightSpec,
    blend_kind: str = "probability",
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    references: list[np.ndarray] = []
    for member in members:
        model = fit_classifier(train, features, member, weight_spec)
        predictions.append(predict_probability(model, validation, features))
        if blend_kind == "rank":
            references.append(
                np.sort(predict_probability(model, train, features), kind="stable")
            )
    matrix = np.vstack(predictions)
    if blend_kind == "probability":
        return np.mean(matrix, axis=0)
    if blend_kind == "raw_logit":
        logits = np.log(matrix / (1.0 - matrix))
        value = np.mean(logits, axis=0)
        return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))
    if blend_kind == "rank":
        ranked = np.vstack(
            [
                np.searchsorted(reference, probability, side="right")
                / float(len(reference))
                for reference, probability in zip(references, predictions, strict=True)
            ]
        )
        return np.clip(np.mean(ranked, axis=0), 1.0e-6, 1.0 - 1.0e-6)
    raise ValueError(f"unsupported ensemble blend: {blend_kind}")


def evaluate_candidate_inner_oof(
    frame: pd.DataFrame,
    *,
    outer_fold: int,
    definition: ExpertDefinition,
    members: Sequence[ModelSpec],
    weight_spec: WeightSpec,
    blend_kind: str = "probability",
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate one expert candidate without touching its outer-test rows."""

    outer_train = frame[frame["outer_fold"].ne(int(outer_fold))].copy()
    outer_train["_inner_fold"] = inner_fold_series(frame, outer_fold).loc[
        outer_train.index
    ].astype(int)
    working = _applicable(outer_train, definition)
    if working.empty or working["binary_target"].nunique() < 2:
        raise ValueError(f"insufficient classes for {definition.expert_id}")
    predictions: list[pd.DataFrame] = []
    for validation_fold in sorted(working["_inner_fold"].unique()):
        train = working[working["_inner_fold"].ne(validation_fold)]
        validation = working[working["_inner_fold"].eq(validation_fold)]
        if validation.empty:
            continue
        if train["binary_target"].nunique() < 2:
            raise ValueError(
                f"inner training fold has one class: outer={outer_fold}, inner={validation_fold}"
            )
        probability = _fit_predict_members(
            train,
            validation,
            definition.features,
            members,
            weight_spec,
            blend_kind,
        )
        predictions.append(
            pd.DataFrame(
                {
                    "r3_row_id": validation["r3_row_id"].to_numpy(),
                    "global_participant_id": validation[
                        "global_participant_id"
                    ].to_numpy(),
                    "dataset_id": validation["dataset_id"].to_numpy(),
                    "binary_target": validation["binary_target"].astype(int).to_numpy(),
                    "inner_fold": int(validation_fold),
                    "probability": probability,
                }
            )
        )
    oof = pd.concat(predictions, ignore_index=True)
    if len(oof) != len(working) or oof["r3_row_id"].duplicated().any():
        raise ValueError("inner OOF predictions do not cover the expert working set once")
    natural = binary_metrics(oof["binary_target"], oof["probability"])
    participant_weight = sample_weights(
        oof,
        WeightSpec("participant_eval", participant_equal=True),
    )
    participant = weighted_binary_metrics(
        oof["binary_target"], oof["probability"], participant_weight
    )
    metrics = {
        "row_count": float(len(oof)),
        "participant_count": float(oof["global_participant_id"].nunique()),
        "positive_row_count": float(oof["binary_target"].sum()),
        "prevalence": float(oof["binary_target"].mean()),
        "natural_auprc": natural["auprc"],
        "natural_auroc": natural["auroc"],
        "natural_brier": natural["brier"],
        "participant_auprc": participant["auprc"],
        "participant_auroc": participant["auroc"],
        "participant_brier": participant["brier"],
        "selection_score": float(
            0.6 * natural["auprc"] + 0.4 * participant["auprc"]
        ),
    }
    return metrics, oof


def _best_record(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    successful = [row for row in records if row.get("status") == "pass"]
    if not successful:
        raise ValueError("all preregistered expert candidates failed")
    return max(
        successful,
        key=lambda row: (
            float(row["selection_score"]),
            float(row["participant_auprc"]),
            -float(row["natural_brier"]),
            str(row["candidate_id"]),
        ),
    )


def _choice_from_record(
    record: Mapping[str, Any], candidates: Mapping[str, tuple[ModelSpec, ...]]
) -> CandidateChoice:
    weight = WeightSpec(**dict(record["weight_spec"]))
    return CandidateChoice(
        candidate_id=str(record["candidate_id"]),
        members=candidates[str(record["model_candidate_id"])],
        weight_spec=weight,
        blend_kind=str(record.get("blend_kind", "probability")),
    )


def _top_two_family_members(
    records: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, tuple[ModelSpec, ...]],
) -> tuple[ModelSpec, ...]:
    """Choose two successful single-seed candidates from distinct families."""

    ranked = sorted(
        (
            row
            for row in records
            if row.get("status") == "pass"
            and len(candidates[str(row["model_candidate_id"])]) == 1
        ),
        key=lambda row: (
            -float(row["selection_score"]),
            -float(row["participant_auprc"]),
            float(row["natural_brier"]),
            str(row["candidate_id"]),
        ),
    )
    selected: list[ModelSpec] = []
    families: set[str] = set()
    for row in ranked:
        member = candidates[str(row["model_candidate_id"])][0]
        if member.family in families:
            continue
        selected.append(member)
        families.add(member.family)
        if len(selected) == 2:
            break
    return tuple(selected)


def run_expert_selection(
    *,
    repository_root: Path,
    report_directory: Path | None = None,
    model_specs: Sequence[ModelSpec] | None = None,
    weight_specs: Sequence[WeightSpec] = DEFAULT_WEIGHT_SPECS,
    outer_folds: Iterable[int] = range(5),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the complete outer-sealed expert family/weight selection."""

    root = Path(repository_root).resolve()
    output = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"selection directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frame = eligible_rows(load_r3_training_frame(repository_root=root))
    specs = tuple(model_specs or default_model_specs())
    base_model_candidates = dict(_ensemble_candidates(specs))
    base_weight = WeightSpec("participant", participant_equal=True)
    search_records: list[dict[str, Any]] = []
    choices: dict[str, dict[str, Any]] = {}
    selected_oof_frames: list[pd.DataFrame] = []
    checkpoint_root = output / "_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    for outer_fold in tuple(int(value) for value in outer_folds):
        choices[str(outer_fold)] = {}
        for definition in EXPERT_DEFINITIONS:
            model_candidates = dict(base_model_candidates)
            checkpoint_prefix = checkpoint_root / f"outer{outer_fold}_{definition.expert_id}"
            checkpoint_records = checkpoint_prefix.with_suffix(".records.json")
            checkpoint_choice = checkpoint_prefix.with_suffix(".choice.json")
            checkpoint_oof = checkpoint_prefix.with_suffix(".oof.parquet")
            if (
                checkpoint_records.exists()
                and checkpoint_choice.exists()
                and checkpoint_oof.exists()
                and not overwrite
            ):
                restored_records = json.loads(
                    checkpoint_records.read_text(encoding="utf-8")
                )
                restored_choice = json.loads(
                    checkpoint_choice.read_text(encoding="utf-8")
                )
                search_records.extend(restored_records)
                choices[str(outer_fold)][definition.expert_id] = restored_choice
                selected_oof_frames.append(pd.read_parquet(checkpoint_oof))
                continue
            family_records: list[dict[str, Any]] = []
            for candidate_id, members in model_candidates.items():
                record: dict[str, Any] = {
                    "stage": "model_family",
                    "outer_fold": outer_fold,
                    "expert_id": definition.expert_id,
                    "candidate_id": candidate_id,
                    "model_candidate_id": candidate_id,
                    "member_ids": [member.candidate_id for member in members],
                    "weight_spec": base_weight.to_dict(),
                    "blend_kind": "probability",
                }
                try:
                    metrics, _ = evaluate_candidate_inner_oof(
                        frame,
                        outer_fold=outer_fold,
                        definition=definition,
                        members=members,
                        weight_spec=base_weight,
                        blend_kind="probability",
                    )
                    record.update(metrics)
                    record["status"] = "pass"
                except (CandidateUnavailable, ValueError, RuntimeError) as exc:
                    record.update({"status": "failed", "failure": str(exc)})
                family_records.append(record)
                search_records.append(record)
            top_two = _top_two_family_members(family_records, model_candidates)
            if len(top_two) == 2:
                top_two_id = "top2__" + "__".join(
                    member.candidate_id for member in top_two
                )
                model_candidates[top_two_id] = top_two
                for blend_kind in ("probability", "raw_logit", "rank"):
                    candidate_id = f"{top_two_id}__{blend_kind}"
                    record = {
                        "stage": "top2_blend",
                        "outer_fold": outer_fold,
                        "expert_id": definition.expert_id,
                        "candidate_id": candidate_id,
                        "model_candidate_id": top_two_id,
                        "member_ids": [member.candidate_id for member in top_two],
                        "weight_spec": base_weight.to_dict(),
                        "blend_kind": blend_kind,
                    }
                    try:
                        metrics, _ = evaluate_candidate_inner_oof(
                            frame,
                            outer_fold=outer_fold,
                            definition=definition,
                            members=top_two,
                            weight_spec=base_weight,
                            blend_kind=blend_kind,
                        )
                        record.update(metrics)
                        record["status"] = "pass"
                    except (CandidateUnavailable, ValueError, RuntimeError) as exc:
                        record.update({"status": "failed", "failure": str(exc)})
                    family_records.append(record)
                    search_records.append(record)
            best_family = _best_record(family_records)
            members = model_candidates[str(best_family["model_candidate_id"])]
            blend_kind = str(best_family.get("blend_kind", "probability"))

            weight_records: list[dict[str, Any]] = []
            for weight_spec in weight_specs:
                candidate_id = f"{best_family['model_candidate_id']}__{weight_spec.weight_id}"
                record = {
                    "stage": "weight",
                    "outer_fold": outer_fold,
                    "expert_id": definition.expert_id,
                    "candidate_id": candidate_id,
                    "model_candidate_id": str(best_family["model_candidate_id"]),
                    "member_ids": [member.candidate_id for member in members],
                    "weight_spec": weight_spec.to_dict(),
                    "blend_kind": blend_kind,
                }
                try:
                    metrics, _ = evaluate_candidate_inner_oof(
                        frame,
                        outer_fold=outer_fold,
                        definition=definition,
                        members=members,
                        weight_spec=weight_spec,
                        blend_kind=blend_kind,
                    )
                    record.update(metrics)
                    record["status"] = "pass"
                except (CandidateUnavailable, ValueError, RuntimeError) as exc:
                    record.update({"status": "failed", "failure": str(exc)})
                weight_records.append(record)
                search_records.append(record)
            best_weight = _best_record(weight_records)
            choice = _choice_from_record(best_weight, model_candidates)
            choices[str(outer_fold)][definition.expert_id] = choice.to_dict()
            metrics, selected_oof = evaluate_candidate_inner_oof(
                frame,
                outer_fold=outer_fold,
                definition=definition,
                members=choice.members,
                weight_spec=choice.weight_spec,
                blend_kind=choice.blend_kind,
            )
            selected_oof.insert(0, "expert_id", definition.expert_id)
            selected_oof.insert(0, "outer_context", outer_fold)
            selected_oof["selected_candidate_id"] = choice.candidate_id
            selected_oof_frames.append(selected_oof)
            choices[str(outer_fold)][definition.expert_id]["inner_oof_metrics"] = metrics
            start_index = len(search_records) - len(family_records) - len(weight_records)
            expert_records = search_records[start_index:]
            checkpoint_records.write_text(
                json.dumps(
                    expert_records,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            checkpoint_choice.write_text(
                json.dumps(
                    choices[str(outer_fold)][definition.expert_id],
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            selected_oof.to_parquet(checkpoint_oof, index=False)

    search = pd.DataFrame(search_records)
    selected_oof_table = pd.concat(selected_oof_frames, ignore_index=True)
    search_path = output / "candidate_search.parquet"
    oof_path = output / "selected_expert_inner_oof.parquet"
    choices_path = output / "selected_experts.json"
    if not overwrite:
        for path in (search_path, oof_path, choices_path):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite selection artifact: {path}")
    search.to_parquet(search_path, index=False)
    selected_oof_table.to_parquet(oof_path, index=False)
    _write_json(
        choices_path,
        {
            "protocol_version": R3_PROTOCOL_VERSION,
            "outer_results_opened": False,
            "selection_metric": "0.6 * natural inner-OOF AUPRC + 0.4 * participant-equal inner-OOF AUPRC",
            "outer_choices": choices,
        },
        overwrite=overwrite,
    )
    manifest = {
        "protocol_version": R3_PROTOCOL_VERSION,
        "task_ids": ["OPT-V333-003", "OPT-V333-004"],
        "status": "pass",
        "outer_results_opened": False,
        "training_config": {
            "path": TRAINING_CONFIG_RELATIVE.as_posix(),
            "sha256": _sha256_file(root / TRAINING_CONFIG_RELATIVE),
        },
        "outer_contexts": sorted(int(value) for value in outer_folds),
        "candidate_search": {
            "path": search_path.relative_to(root).as_posix(),
            "sha256": _sha256_file(search_path),
            "row_count": int(len(search)),
            "failed_count": int(search["status"].eq("failed").sum()),
        },
        "selected_expert_inner_oof": {
            "path": oof_path.relative_to(root).as_posix(),
            "sha256": _sha256_file(oof_path),
            "row_count": int(len(selected_oof_table)),
        },
        "selected_experts": {
            "path": choices_path.relative_to(root).as_posix(),
            "sha256": _sha256_file(choices_path),
        },
    }
    manifest_path = output / "artifact_manifest.json"
    _write_json(manifest_path, manifest, overwrite=overwrite)
    manifest["manifest_path"] = manifest_path.relative_to(root).as_posix()
    manifest["manifest_sha256"] = _sha256_file(manifest_path)
    return manifest


__all__ = [
    "CandidateChoice",
    "EXPERT_DEFINITIONS",
    "ExpertDefinition",
    "evaluate_candidate_inner_oof",
    "run_expert_selection",
]
