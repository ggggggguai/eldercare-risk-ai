"""Subject-safe Audio/Text late fusion for the V34-A3 candidate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.model import (
    CognitiveChangeClueModel,
)

try:
    from .build_subject_cv_v34 import DEFAULT_AUDIT_PATH, DEFAULT_OUTPUT_PATH
    from .common import sha256_file, workspace_relative
    from .dataset import compute_moca_statistics, compute_training_weights
    from .train import (
        atomic_torch_save,
        capture_rng_state,
        configure_determinism,
        restore_rng_state,
        train_one_epoch,
        write_json,
        write_jsonl,
    )
    from .train_v34 import (
        CONFIG_PATH,
        PREDICTION_FILES,
        REPORT_ROOT,
        V34TrainingError,
        _candidate_loss_weights,
        _candidate_masking_plan,
        _complete_rows,
        _coverage,
        _fold_summary,
        _loader,
        _new_grad_scaler,
        _raw_metric_report,
        _write_prediction_bundle,
        load_config,
        predict_raw,
        verify_prediction_bundle,
    )
    from .v34_data import SubjectFeatureDataset, load_fold, load_official_train_pool
    from .v34_metrics import fit_platt, temporary_platt_evaluation
except ImportError:
    from build_subject_cv_v34 import DEFAULT_AUDIT_PATH, DEFAULT_OUTPUT_PATH  # type: ignore[no-redef]
    from common import sha256_file, workspace_relative  # type: ignore[no-redef]
    from dataset import compute_moca_statistics, compute_training_weights  # type: ignore[no-redef]
    from train import (  # type: ignore[no-redef]
        atomic_torch_save,
        capture_rng_state,
        configure_determinism,
        restore_rng_state,
        train_one_epoch,
        write_json,
        write_jsonl,
    )
    from train_v34 import (  # type: ignore[no-redef]
        CONFIG_PATH,
        PREDICTION_FILES,
        REPORT_ROOT,
        V34TrainingError,
        _candidate_loss_weights,
        _candidate_masking_plan,
        _complete_rows,
        _coverage,
        _fold_summary,
        _loader,
        _new_grad_scaler,
        _raw_metric_report,
        _write_prediction_bundle,
        load_config,
        predict_raw,
        verify_prediction_bundle,
    )
    from v34_data import SubjectFeatureDataset, load_fold, load_official_train_pool  # type: ignore[no-redef]
    from v34_metrics import fit_platt, temporary_platt_evaluation  # type: ignore[no-redef]


CANDIDATE_ID = "V34-A3"
COMPONENT_IDS = {"audio": "V34-A3-Audio", "text": "V34-A3-Text"}
INNER_FOLDS = 5


def _subject_hash(subject_ids: Sequence[str]) -> str:
    payload = json.dumps(sorted(subject_ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _component_config_hash(config: Mapping[str, Any], modality: str) -> str:
    payload = json.dumps(
        {
            "model": config["model"],
            "training": config["training"],
            "masking": config["masking"],
            "modality": modality,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_inner_subject_folds(
    records: Sequence[Mapping[str, Any]],
    subject_ids: Sequence[str],
    *,
    outer_fold: int,
) -> list[dict[str, Any]]:
    labels_by_subject: dict[str, str] = {}
    requested = set(subject_ids)
    for row in records:
        subject_id = str(row["subject_id"])
        if subject_id not in requested:
            continue
        label = str(row["diagnosis_label"])
        previous = labels_by_subject.setdefault(subject_id, label)
        if previous != label:
            raise V34TrainingError(f"subject has inconsistent diagnosis: {subject_id}")
    ordered = sorted(requested)
    if set(ordered) != set(labels_by_subject):
        raise V34TrainingError("inner OOF subject table is incomplete")
    labels = [labels_by_subject[subject_id] for subject_id in ordered]
    seed = 20260805 + 100 + int(outer_fold)
    splitter = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    folds = []
    seen: set[str] = set()
    for inner_fold, (train_indices, evaluation_indices) in enumerate(
        splitter.split(np.zeros(len(ordered), dtype=np.int8), labels)
    ):
        train_subjects = [ordered[index] for index in train_indices]
        evaluation_subjects = [ordered[index] for index in evaluation_indices]
        if set(train_subjects).intersection(evaluation_subjects):
            raise V34TrainingError("inner OOF fold leaks subjects")
        seen.update(evaluation_subjects)
        folds.append(
            {
                "inner_fold": inner_fold,
                "seed": seed,
                "train_subjects": train_subjects,
                "evaluation_subjects": evaluation_subjects,
                "train_subject_hash": _subject_hash(train_subjects),
                "evaluation_subject_hash": _subject_hash(evaluation_subjects),
            }
        )
    if seen != set(ordered) or sum(len(item["evaluation_subjects"]) for item in folds) != len(ordered):
        raise V34TrainingError("inner OOF does not cover each subject exactly once")
    return folds


def _component_config(
    base_config: Mapping[str, Any], *, seed: int, modality: str
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base_config))
    config["training"] = copy.deepcopy(dict(base_config["training"]))
    config["training"]["seed"] = int(seed)
    config["model"] = copy.deepcopy(dict(base_config["model"]))
    config["model"]["loss_weights"] = _candidate_loss_weights(
        base_config["model"]["loss_weights"], "main_head"
    )
    config["masking"] = copy.deepcopy(dict(base_config["masking"]))
    config["masking"]["seed_base"] = int(seed) + 5000
    config["component"] = {"candidate_id": COMPONENT_IDS[modality], "modality": modality}
    return config


def train_fixed_component(
    records: Sequence[Mapping[str, Any]],
    subject_ids: Sequence[str],
    *,
    modality: str,
    epochs: int,
    seed: int,
    config: Mapping[str, Any],
    device: torch.device,
    checkpoint_path: Path,
) -> CognitiveChangeClueModel:
    if modality not in COMPONENT_IDS or epochs < 1:
        raise V34TrainingError("invalid fixed component request")
    expected_subject_hash = _subject_hash(subject_ids)
    expected_config_hash = _component_config_hash(config, modality)
    if checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            payload.get("schema_version") != "cognitive_v34_fixed_component_v1"
            or payload.get("modality") != modality
            or int(payload.get("epochs", -1)) != int(epochs)
            or payload.get("train_subject_hash") != expected_subject_hash
            or payload.get("config_sha256") != expected_config_hash
            or int(payload.get("seed", -1)) != int(seed)
        ):
            raise V34TrainingError(f"fixed component checkpoint mismatch: {checkpoint_path}")
        model = CognitiveChangeClueModel(
            projection_dim=int(config["model"]["projection_dim"]),
            fusion_hidden_dims=tuple(config["model"]["fusion_hidden_dims"]),
            dropout=float(config["model"]["dropout"]),
        )
        model.load_state_dict(payload["model_state_dict"])
        return model.to(device).eval()

    component_config = _component_config(config, seed=seed, modality=modality)
    configure_determinism(component_config)
    dataset = SubjectFeatureDataset(
        records, subject_ids, available_modalities=(modality,)
    )
    weights = compute_training_weights(dataset.records)
    moca_statistics = compute_moca_statistics(dataset.records)
    model = CognitiveChangeClueModel(
        projection_dim=int(component_config["model"]["projection_dim"]),
        fusion_hidden_dims=tuple(component_config["model"]["fusion_hidden_dims"]),
        dropout=float(component_config["model"]["dropout"]),
    ).to(device)
    training = component_config["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=tuple(float(value) for value in training["adam_betas"]),
        eps=float(training["adam_eps"]),
    )
    amp_enabled = bool(training["amp"]) and device.type == "cuda"
    scaler = _new_grad_scaler(amp_enabled)
    epoch_rows = []
    for epoch in range(1, int(epochs) + 1):
        plan, _ = _candidate_masking_plan(
            dataset.records,
            modalities=(modality,),
            epoch=epoch,
            seed_base=int(component_config["masking"]["seed_base"]),
        )
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        scaler_state = copy.deepcopy(scaler.state_dict())
        rng_state = capture_rng_state()
        recoveries = []
        for recovery_index in range(9):
            try:
                metrics = train_one_epoch(
                    model,
                    _loader(dataset, component_config, train=True, epoch=epoch),
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                    masking_plan=plan,
                    weights=weights,
                    moca_statistics=moca_statistics,
                    loss_weights=component_config["model"]["loss_weights"],
                    accumulation_steps=int(training["gradient_accumulation_steps"]),
                    max_grad_norm=float(training["max_grad_norm"]),
                    amp_enabled=amp_enabled,
                )
                break
            except Exception as exc:
                if (
                    not amp_enabled
                    or not str(exc).startswith("gradient is non-finite:")
                    or recovery_index >= 8
                ):
                    raise
                model.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)
                # GradScaler.state_dict() does not include its per-optimizer
                # UNSCALED/STEPPED stage. Recreate it before retrying an epoch
                # that failed after unscale_(), otherwise the retry raises a
                # misleading duplicate-unscale error.
                scaler = _new_grad_scaler(amp_enabled)
                scaler.load_state_dict(scaler_state)
                restore_rng_state(rng_state)
                previous_scale = float(scaler.get_scale())
                new_scale = max(1.0, previous_scale * 0.5)
                state = scaler.state_dict()
                state["scale"] = new_scale
                scaler.load_state_dict(state)
                scaler_state = copy.deepcopy(scaler.state_dict())
                recoveries.append(
                    {
                        "reason": str(exc),
                        "previous_scale": previous_scale,
                        "new_scale": new_scale,
                    }
                )
        epoch_rows.append({"epoch": epoch, "train": metrics, "recoveries": recoveries})
    payload = {
        "schema_version": "cognitive_v34_fixed_component_v1",
        "candidate_id": COMPONENT_IDS[modality],
        "modality": modality,
        "epochs": int(epochs),
        "seed": int(seed),
        "train_subject_hash": expected_subject_hash,
        "train_subject_count": len(set(subject_ids)),
        "config_sha256": expected_config_hash,
        "model_state_dict": model.state_dict(),
        "training_weights": weights.to_dict(),
        "moca_statistics": moca_statistics.to_dict(),
        "epoch_metrics": epoch_rows,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(checkpoint_path, payload)
    return model.eval()


def _component_rows(
    model: CognitiveChangeClueModel,
    records: Sequence[Mapping[str, Any]],
    subject_ids: Sequence[str],
    *,
    modality: str,
    config: Mapping[str, Any],
    device: torch.device,
    outer_fold: int,
    split_role: str,
) -> list[dict[str, Any]]:
    dataset = SubjectFeatureDataset(records, subject_ids)
    prediction_config = copy.deepcopy(dict(config))
    prediction_config["training"] = copy.deepcopy(dict(config["training"]))
    prediction_config["training"].setdefault(
        "seed", int(config["seed"]) + int(outer_fold)
    )
    return predict_raw(
        model,
        dataset,
        config=prediction_config,
        device=device,
        candidate_id=COMPONENT_IDS[modality],
        outer_fold=outer_fold,
        split_role=split_role,
    )


def _fit_late_fusion(
    audio_rows: Sequence[Mapping[str, Any]], text_rows: Sequence[Mapping[str, Any]]
) -> tuple[LogisticRegression, list[dict[str, Any]]]:
    audio = {str(row["sample_id"]): row for row in audio_rows}
    text = {str(row["sample_id"]): row for row in text_rows}
    if set(audio) != set(text):
        raise V34TrainingError("A3 OOF component sample sets differ")
    fit_rows = []
    for sample_id in sorted(audio):
        a_row, t_row = audio[sample_id], text[sample_id]
        if str(a_row["subject_id"]) != str(t_row["subject_id"]):
            raise V34TrainingError("A3 OOF component identities differ")
        if a_row["raw_logit"] is None or t_row["raw_logit"] is None:
            continue
        fit_rows.append(
            {
                "sample_id": sample_id,
                "subject_id": str(a_row["subject_id"]),
                "label": int(a_row["label_hc_vs_non_hc"]),
                "audio_logit": float(a_row["raw_logit"]),
                "text_logit": float(t_row["raw_logit"]),
                "q_audio": float(a_row["q_audio"]),
                "q_text": float(t_row["q_text"]),
            }
        )
    labels = np.asarray([row["label"] for row in fit_rows], dtype=np.int64)
    if len(fit_rows) == 0 or len(np.unique(labels)) != 2:
        raise V34TrainingError("A3 inner OOF cannot fit the binary late fusion")
    features = np.asarray(
        [
            [row["audio_logit"], row["text_logit"], row["q_audio"], row["q_text"]]
            for row in fit_rows
        ],
        dtype=np.float64,
    )
    classifier = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000,
    )
    classifier.fit(features, labels)
    return classifier, fit_rows


def combine_component_rows(
    audio_rows: Sequence[Mapping[str, Any]],
    text_rows: Sequence[Mapping[str, Any]],
    classifier: LogisticRegression,
    *,
    outer_fold: int,
    split_role: str,
    schema_version: str,
) -> list[dict[str, Any]]:
    audio = {str(row["sample_id"]): row for row in audio_rows}
    text = {str(row["sample_id"]): row for row in text_rows}
    if set(audio) != set(text):
        raise V34TrainingError("A3 component prediction sample sets differ")
    result = []
    for sample_id in sorted(audio):
        a_row, t_row = audio[sample_id], text[sample_id]
        audio_logit = None if a_row["raw_logit"] is None else float(a_row["raw_logit"])
        text_logit = None if t_row["raw_logit"] is None else float(t_row["raw_logit"])
        if audio_logit is not None and text_logit is not None:
            raw_logit = float(
                classifier.decision_function(
                    [[audio_logit, text_logit, float(a_row["q_audio"]), float(t_row["q_text"])]]
                )[0]
            )
        elif audio_logit is not None:
            raw_logit = audio_logit
        else:
            raw_logit = text_logit
        result.append(
            {
                "schema_version": schema_version,
                "candidate_id": CANDIDATE_ID,
                "outer_fold": int(outer_fold),
                "split_role": split_role,
                "sample_id": sample_id,
                "subject_id": str(a_row["subject_id"]),
                "diagnosis": str(a_row["diagnosis"]),
                "label_hc_vs_non_hc": int(a_row["label_hc_vs_non_hc"]),
                "raw_logit": raw_logit,
                "audio_logit": audio_logit,
                "text_logit": text_logit,
                "face_logit": None,
                "q_audio": float(a_row["q_audio"]),
                "q_text": float(t_row["q_text"]),
                "q_face": float(a_row["q_face"]),
                "audio_missing_mask": int(a_row["audio_missing_mask"]),
                "text_missing_mask": int(t_row["text_missing_mask"]),
                "face_missing_mask": int(a_row["face_missing_mask"]),
            }
        )
    return result


def _degradation_calibration(
    calibration_rows: Sequence[Mapping[str, Any]],
    outer_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for branch in ("audio", "text"):
        other = "text" if branch == "audio" else "audio"
        calibration = [
            row
            for row in calibration_rows
            if row[f"{branch}_logit"] is not None and row[f"{other}_logit"] is None
        ]
        outer = [
            row
            for row in outer_rows
            if row[f"{branch}_logit"] is not None and row[f"{other}_logit"] is None
        ]
        labels = {int(row["label_hc_vs_non_hc"]) for row in calibration}
        if labels != {0, 1}:
            report[branch] = {
                "status": "invalid" if outer else "not_observed_in_outer_evaluation",
                "reason": (
                    "inner_calibration_lacks_both_binary_classes"
                    if outer
                    else "no_outer_evaluation_rows_for_branch"
                ),
                "inner_calibration_task_count": len(calibration),
                "outer_evaluation_task_count": len(outer),
            }
            continue
        if not outer:
            report[branch] = {
                "status": "valid_no_outer_evaluation_rows",
                "inner_calibration_task_count": len(calibration),
                "outer_evaluation_task_count": 0,
                "temporary_platt_parameters": fit_platt(
                    [float(row["raw_logit"]) for row in calibration],
                    [int(row["label_hc_vs_non_hc"]) for row in calibration],
                ),
            }
            continue
        report[branch] = {
            "status": "valid",
            "inner_calibration_task_count": len(calibration),
            "outer_evaluation_task_count": len(outer),
            "temporary_platt": temporary_platt_evaluation(calibration, outer),
        }
    return report


def _best_epochs(base_runs: Mapping[str, Path], outer_fold: int) -> dict[str, int]:
    result = {}
    for modality, run_dir in base_runs.items():
        path = run_dir / f"fold_{outer_fold}" / COMPONENT_IDS[modality] / "fold_metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[modality] = int(payload["best_epoch"])
    return result


def build_fold(
    *,
    outer_fold: int,
    run_dir: Path,
    base_runs: Mapping[str, Path],
    records: Sequence[Mapping[str, Any]],
    input_hashes: Mapping[str, str],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    candidate_dir = run_dir / f"fold_{outer_fold}" / CANDIDATE_ID
    candidate_dir.mkdir(parents=True, exist_ok=True)
    fold = load_fold(outer_fold)
    split_sha = str(fold["split_sha256"])
    existing = candidate_dir / "fold_metrics.json"
    if existing.is_file() and (candidate_dir / "prediction_manifest.json").is_file():
        verify_prediction_bundle(candidate_dir, expected_split_sha256=split_sha)
        return json.loads(existing.read_text(encoding="utf-8"))
    started = time.perf_counter()
    selection_subjects = sorted(
        set(fold["inner_train_subjects"]).union(fold["inner_validation_subjects"])
    )
    inner_folds = build_inner_subject_folds(records, selection_subjects, outer_fold=outer_fold)
    write_json(
        candidate_dir / "inner_oof_split_audit.json",
        {
            "schema_version": "cognitive_v34_a3_inner_oof_v1",
            "outer_fold": outer_fold,
            "seed": 20260805 + 100 + outer_fold,
            "subject_count": len(selection_subjects),
            "subject_hash": _subject_hash(selection_subjects),
            "folds": inner_folds,
            "status": "passed",
        },
    )
    epochs = _best_epochs(base_runs, outer_fold)
    oof_by_modality: dict[str, list[dict[str, Any]]] = {"audio": [], "text": []}
    component_assets: dict[str, Any] = {"inner_oof": {}, "final": {}}
    for inner in inner_folds:
        inner_index = int(inner["inner_fold"])
        component_assets["inner_oof"][str(inner_index)] = {}
        for modality_index, modality in enumerate(("audio", "text")):
            checkpoint = candidate_dir / "components" / f"inner_{inner_index}" / f"{modality}.pt"
            model = train_fixed_component(
                records,
                inner["train_subjects"],
                modality=modality,
                epochs=epochs[modality],
                seed=20260805 + 10000 * outer_fold + 100 * inner_index + modality_index,
                config=config,
                device=device,
                checkpoint_path=checkpoint,
            )
            rows = _component_rows(
                model,
                records,
                inner["evaluation_subjects"],
                modality=modality,
                config=config,
                device=device,
                outer_fold=outer_fold,
                split_role="inner_validation",
            )
            oof_by_modality[modality].extend(rows)
            prediction_path = checkpoint.with_suffix(".predictions.jsonl")
            write_jsonl(prediction_path, rows)
            component_assets["inner_oof"][str(inner_index)][modality] = {
                "checkpoint": workspace_relative(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "predictions": workspace_relative(prediction_path),
                "predictions_sha256": sha256_file(prediction_path),
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    classifier, fit_rows = _fit_late_fusion(oof_by_modality["audio"], oof_by_modality["text"])
    write_jsonl(candidate_dir / "inner_selection_oof_fusion_inputs.jsonl", fit_rows)

    final_models: dict[str, CognitiveChangeClueModel] = {}
    for modality_index, modality in enumerate(("audio", "text")):
        checkpoint = candidate_dir / "components" / "final" / f"{modality}.pt"
        final_models[modality] = train_fixed_component(
            records,
            selection_subjects,
            modality=modality,
            epochs=epochs[modality],
            seed=20260805 + 10000 * outer_fold + 9000 + modality_index,
            config=config,
            device=device,
            checkpoint_path=checkpoint,
        )
        component_assets["final"][modality] = {
            "checkpoint": workspace_relative(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
        }

    validation_subjects = set(fold["inner_validation_subjects"])
    validation_components = {
        modality: [
            row for row in rows if str(row["subject_id"]) in validation_subjects
        ]
        for modality, rows in oof_by_modality.items()
    }
    prediction_rows = {
        "inner_validation": combine_component_rows(
            validation_components["audio"],
            validation_components["text"],
            classifier,
            outer_fold=outer_fold,
            split_role="inner_validation",
            schema_version=str(config["outputs"]["prediction_schema_version"]),
        )
    }
    for role, subject_key in (
        ("inner_calibration", "inner_calibration_subjects"),
        ("outer_evaluation", "outer_evaluation_subjects"),
    ):
        component_rows = {
            modality: _component_rows(
                final_models[modality],
                records,
                fold[subject_key],
                modality=modality,
                config=config,
                device=device,
                outer_fold=outer_fold,
                split_role=role,
            )
            for modality in ("audio", "text")
        }
        prediction_rows[role] = combine_component_rows(
            component_rows["audio"],
            component_rows["text"],
            classifier,
            outer_fold=outer_fold,
            split_role=role,
            schema_version=str(config["outputs"]["prediction_schema_version"]),
        )

    bundle = {
        "schema_version": "cognitive_v34_a3_fold_bundle_v1",
        "candidate_id": CANDIDATE_ID,
        "outer_fold": outer_fold,
        "split_sha256": split_sha,
        "best_epochs": epochs,
        "base_fold_metrics": {
            modality: {
                "path": workspace_relative(
                    base_runs[modality]
                    / f"fold_{outer_fold}"
                    / COMPONENT_IDS[modality]
                    / "fold_metrics.json"
                ),
                "sha256": sha256_file(
                    base_runs[modality]
                    / f"fold_{outer_fold}"
                    / COMPONENT_IDS[modality]
                    / "fold_metrics.json"
                ),
            }
            for modality in ("audio", "text")
        },
        "fusion": {
            "type": "LogisticRegression",
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 1000,
            "feature_order": ["audio_logit", "text_logit", "q_audio", "q_text"],
            "classes": [int(value) for value in classifier.classes_],
            "coef": classifier.coef_.tolist(),
            "intercept": classifier.intercept_.tolist(),
            "fit_task_count": len(fit_rows),
            "fit_subject_count": len({row["subject_id"] for row in fit_rows}),
        },
        "components": component_assets,
    }
    bundle_path = candidate_dir / "late_fusion_bundle.json"
    write_json(bundle_path, bundle)
    manifest = _write_prediction_bundle(
        candidate_dir=candidate_dir,
        candidate_id=CANDIDATE_ID,
        outer_fold=outer_fold,
        split_sha256=split_sha,
        checkpoint_path=bundle_path,
        input_hashes={
            **dict(input_hashes),
            "v34_split": split_sha,
            "audio_base_fold_metrics": bundle["base_fold_metrics"]["audio"]["sha256"],
            "text_base_fold_metrics": bundle["base_fold_metrics"]["text"]["sha256"],
        },
        prediction_rows=prediction_rows,
    )
    complete = {
        role: _complete_rows(rows, ("audio", "text", "face"))
        for role, rows in prediction_rows.items()
    }
    raw = {role: _raw_metric_report(rows) for role, rows in complete.items()}
    temporary = temporary_platt_evaluation(
        complete["inner_calibration"], complete["outer_evaluation"]
    )
    degradation = _degradation_calibration(
        prediction_rows["inner_calibration"],
        prediction_rows["outer_evaluation"],
    )
    summary = {
        "schema_version": "cognitive_v34_fold_metrics_v1",
        "candidate_id": CANDIDATE_ID,
        "outer_fold": outer_fold,
        "selection_population": "same_original_complete_audio_text_face_tasks",
        "selection_modalities": ["audio", "text", "face"],
        "best_epochs": epochs,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "split_sha256": split_sha,
        "complete_task_counts": {role: len(rows) for role, rows in complete.items()},
        "raw_metrics": {
            role: {key: value for key, value in report.items() if key != "subject_rows"}
            for role, report in raw.items()
        },
        "temporary_calibration": temporary,
        "degradation_calibration": degradation,
        "candidate_valid": all(
            record["status"] != "invalid" for record in degradation.values()
        ),
        "coverage": {role: _coverage(rows) for role, rows in prediction_rows.items()},
        "duration_seconds": time.perf_counter() - started,
        "official_test_evaluated": False,
    }
    write_json(candidate_dir / "fold_metrics.json", summary)
    write_json(
        candidate_dir / "run_manifest.json",
        {
            "schema_version": "cognitive_v34_fold_run_v1",
            "status": "completed",
            "candidate_id": CANDIDATE_ID,
            "outer_fold": outer_fold,
            "best_epochs": epochs,
            "prediction_manifest": workspace_relative(candidate_dir / "prediction_manifest.json"),
            "official_test_evaluated": False,
        },
    )
    return summary


def aggregate(run_dir: Path) -> dict[str, Any]:
    folds = [
        json.loads((run_dir / f"fold_{index}" / CANDIDATE_ID / "fold_metrics.json").read_text(encoding="utf-8"))
        for index in range(5)
    ]
    subject_auc = [float(item["raw_metrics"]["outer_evaluation"]["subject"]["roc_auc"]) for item in folds]
    task_auc = [float(item["raw_metrics"]["outer_evaluation"]["task"]["roc_auc"]) for item in folds]
    calibrated_auc = [float(item["temporary_calibration"]["outer_evaluation_subject_metrics"]["roc_auc"]) for item in folds]
    calibrated_brier = [float(item["temporary_calibration"]["outer_evaluation_subject_metrics"]["brier"]) for item in folds]
    calibrated_ece = [float(item["temporary_calibration"]["outer_evaluation_subject_metrics"]["ece_10_equal_width"]) for item in folds]
    confusion = {key: 0 for key in ("tp", "fn", "tn", "fp")}
    false_positive_counts: Counter[str] = Counter()
    for item in folds:
        matrix = item["temporary_calibration"]["outer_evaluation_subject_metrics"]["confusion_matrix"]
        for key in confusion:
            confusion[key] += int(matrix[key])
        false_positive_counts.update(item["temporary_calibration"]["false_positive_subjects"])
    sensitivity = confusion["tp"] / (confusion["tp"] + confusion["fn"])
    specificity = confusion["tn"] / (confusion["tn"] + confusion["fp"])
    report = {
        "schema_version": "cognitive_v34_candidate_cv_summary_v1",
        "candidate_id": CANDIDATE_ID,
        "run_id": run_dir.name,
        "fold_count": 5,
        "development_subject_count": 459,
        "official_test_evaluated": False,
        "selection_population": "same_original_complete_audio_text_face_tasks",
        "candidate_valid": all(bool(item.get("candidate_valid", False)) for item in folds),
        "best_epochs": [item["best_epochs"] for item in folds],
        "task_raw_auc": {"folds": task_auc, "mean": mean(task_auc), "population_std": pstdev(task_auc), "worst": min(task_auc)},
        "subject_raw_auc": {"folds": subject_auc, "mean": mean(subject_auc), "population_std": pstdev(subject_auc), "worst": min(subject_auc)},
        "subject_temporary_platt_auc": {"folds": calibrated_auc, "mean": mean(calibrated_auc), "population_std": pstdev(calibrated_auc), "worst": min(calibrated_auc)},
        "subject_temporary_platt_brier": {"folds": calibrated_brier, "mean": mean(calibrated_brier), "population_std": pstdev(calibrated_brier)},
        "subject_temporary_platt_ece": {"folds": calibrated_ece, "mean": mean(calibrated_ece), "population_std": pstdev(calibrated_ece)},
        "pooled_temporary_workpoint": {
            "confusion_matrix": confusion,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": (sensitivity + specificity) / 2.0,
        },
        "false_positive_analysis": {
            "population": "outer_evaluation_complete_audio_text_face_hc_subjects",
            "false_positive_count": sum(false_positive_counts.values()),
            "unique_false_positive_subject_count": len(false_positive_counts),
            "subjects": [
                {"subject_id": subject_id, "fold_occurrences": count}
                for subject_id, count in sorted(false_positive_counts.items())
            ],
        },
        "folds": folds,
        "limitations": [
            "This is five-fold development evidence on CogPic official Train only.",
            "It is not an independent external Test result.",
        ],
    }
    write_json(run_dir / f"{CANDIDATE_ID}_cv_summary.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    if sklearn_version != "1.9.0":
        raise V34TrainingError(f"scikit-learn 1.9.0 is required, got {sklearn_version}")
    config = load_config(args.config)
    audit = json.loads(DEFAULT_AUDIT_PATH.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or int(audit.get("official_test_overlap_count", -1)) != 0:
        raise V34TrainingError("V3.4 split audit did not pass")
    resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(resolved_device if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise V34TrainingError("CUDA was requested but is unavailable")
    records, input_hashes = load_official_train_pool(verify_hashes=True)
    base_runs = {"audio": Path(args.audio_run), "text": Path(args.text_run)}
    for modality, path in base_runs.items():
        if not path.is_absolute():
            path = REPORT_ROOT / "runs" / path
            base_runs[modality] = path
        for outer_fold in range(5):
            required = path / f"fold_{outer_fold}" / COMPONENT_IDS[modality] / "fold_metrics.json"
            if not required.is_file():
                raise V34TrainingError(f"A3 base run is incomplete: {required}")
    run_dir = REPORT_ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": "cognitive_v34_candidate_run_v1",
            "status": "running",
            "run_id": args.run_id,
            "candidate_id": CANDIDATE_ID,
            "started_or_resumed_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "audio_base_run": workspace_relative(base_runs["audio"]),
            "text_base_run": workspace_relative(base_runs["text"]),
            "split_sha256": sha256_file(DEFAULT_OUTPUT_PATH),
            "config_sha256": sha256_file(args.config),
            "official_test_evaluated": False,
        },
    )
    requested = range(5) if args.folds == "all" else [int(value) for value in args.folds.split(",")]
    for outer_fold in requested:
        build_fold(
            outer_fold=outer_fold,
            run_dir=run_dir,
            base_runs=base_runs,
            records=records,
            input_hashes=input_hashes,
            config=config,
            device=device,
        )
    completed = [index for index in range(5) if (run_dir / f"fold_{index}" / CANDIDATE_ID / "fold_metrics.json").is_file()]
    result: dict[str, Any] = {"run_id": args.run_id, "candidate_id": CANDIDATE_ID, "completed_folds": completed, "official_test_evaluated": False}
    if len(completed) == 5:
        result["cv_summary"] = aggregate(run_dir)
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        manifest.update({"status": "completed", "ended_at": datetime.now(timezone.utc).astimezone().isoformat(), "summary": workspace_relative(run_dir / f"{CANDIDATE_ID}_cv_summary.json")})
        write_json(run_dir / "run_manifest.json", manifest)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--audio-run", required=True)
    parser.add_argument("--text-run", required=True)
    parser.add_argument("--folds", default="all")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_inner_subject_folds",
    "combine_component_rows",
    "train_fixed_component",
]
