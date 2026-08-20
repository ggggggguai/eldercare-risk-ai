from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import torch

from .casme2_apexfusion import ApexFusionConfig, ApexFusionNet, load_checkpoint, save_checkpoint
from .casme2_apexfusion_evaluation import (
    FUSION_WEIGHTS,
    apply_calibrator,
    equal_seed_probability_ensemble,
    fit_calibrator,
    selection_score,
)
from .casme2_apexfusion_manifest import sha256_file
from .casme2_apexfusion_training import (
    CANDIDATES,
    MemoryArtifactStore,
    TrainingCandidate,
    _augment,
    _classification_loss,
    _loader,
    _ssl_pretrain,
    _to_device,
    predict,
    set_seed,
)
from .casme2_traditional import TraditionalConfig, build_pipeline, feature_vector, load_traditional, save_traditional
from .metrics import classification_metrics


FORMAL_SCHEMA = "opt_me_002_formal_v1"
RESOLVED_SCHEMA = "opt_me_002_resolved_formal_contract_v1"
EXPECTED_CANDIDATE_ID = "ME2-FROZEN-NESTED-V1"
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260809


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sequence_sha256(values: Sequence[str]) -> str:
    return sha256(canonical_json(list(values)).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return sha256_file(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(probabilities: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1.0)
    return value / value.sum(axis=1, keepdims=True)


def candidate_from_mapping(value: Mapping[str, Any]) -> TrainingCandidate:
    payload = dict(value)
    payload["enabled_routes"] = tuple(payload["enabled_routes"])
    return TrainingCandidate(**payload)


def _selected_deep_result(selection_outer: Mapping[str, Any]) -> Mapping[str, Any]:
    return next(
        row
        for row in selection_outer["deep_candidates"]
        if row["candidate"]["candidate_id"] == selection_outer["selected_deep_candidate"]
    )


def _selected_traditional_result(selection_outer: Mapping[str, Any]) -> Mapping[str, Any]:
    return next(
        row
        for row in selection_outer["traditional_candidates"]
        if row["candidate"]["candidate_id"] == selection_outer["selected_traditional_candidate"]
    )


def materialize_application_contracts(
    *,
    selection: Mapping[str, Any],
    freeze: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selection_by_fold = {int(row["outer_fold"]): row for row in selection["outer_results"]}
    contracts: list[dict[str, Any]] = []
    for frozen in freeze["outer_contracts"]:
        outer_fold = int(frozen["outer_fold"])
        selected = selection_by_fold[outer_fold]
        deep = _selected_deep_result(selected)
        traditional = _selected_traditional_result(selected)
        epoch_scores = []
        for epoch in range(1, int(config["training_budget"]["supervised_epochs"]) + 1):
            scores = [float(log["history"][epoch - 1]["selection_score"]) for log in deep["inner_logs"]]
            epoch_scores.append({"epoch": epoch, "mean_inner_selection_score": float(np.mean(scores)), "fold_values": scores})
        final_epoch = max(epoch_scores, key=lambda row: (row["mean_inner_selection_score"], -row["epoch"]))["epoch"]

        deep_predictions = sorted(deep["predictions"], key=lambda row: str(row["sample_id"]))
        traditional_predictions = sorted(traditional["predictions"], key=lambda row: str(row["sample_id"]))
        if [row["sample_id"] for row in deep_predictions] != [row["sample_id"] for row in traditional_predictions]:
            raise RuntimeError(f"Outer fold {outer_fold} inner OOF populations differ")
        labels = np.asarray([row["true_label"] for row in deep_predictions], dtype=np.int64)
        raw_deep = np.asarray([row["probabilities"] for row in deep_predictions], dtype=np.float64)
        raw_traditional = np.asarray([row["probabilities"] for row in traditional_predictions], dtype=np.float64)
        calibrator = fit_calibrator(str(frozen["selected_calibration"]), labels, raw_deep)
        calibrated_deep = apply_calibrator(calibrator, raw_deep)
        fusion_scores = []
        for weight in FUSION_WEIGHTS:
            probabilities = normalize(weight * calibrated_deep + (1.0 - weight) * raw_traditional)
            score = selection_score(labels, probabilities)
            fusion_scores.append({"deep_weight": float(weight), "traditional_weight": float(1.0 - weight), "selection_score": list(score)})
        selected_fusion = max(
            fusion_scores,
            key=lambda row: (tuple(row["selection_score"]), -abs(row["deep_weight"] - 0.5)),
        )
        contracts.append(
            {
                "outer_fold": outer_fold,
                "selected_deep_candidate": selected["selected_deep_candidate"],
                "deep_candidate": deep["candidate"],
                "selected_traditional_candidate": selected["selected_traditional_candidate"],
                "traditional_candidate": traditional["candidate"],
                "selected_calibration": frozen["selected_calibration"],
                "final_calibrator": calibrator,
                "selected_output": frozen["selected_output"],
                "final_fusion": selected_fusion,
                "epoch_selection": {"rule": config["checkpoint_rule"], "scores": epoch_scores, "final_epoch": int(final_epoch)},
                "inner_oof_sample_count": len(labels),
                "inner_oof_sample_ids_sha256": sequence_sha256([str(row["sample_id"]) for row in deep_predictions]),
                "outer_test_used": False,
            }
        )
    return sorted(contracts, key=lambda row: row["outer_fold"])


def verify_freeze_gate(
    *,
    config_path: Path,
    freeze_path: Path,
    audit_path: Path,
    selection_path: Path,
    split_path: Path,
    artifact_manifest_path: Path,
) -> dict[str, Any]:
    config, freeze, audit = read_json(config_path), read_json(freeze_path), read_json(audit_path)
    selection, split = read_json(selection_path), read_json(split_path)
    errors: list[str] = []
    config_hash, freeze_hash = sha256_file(config_path), sha256_file(freeze_path)
    selection_hash, split_hash = sha256_file(selection_path), sha256_file(split_path)
    if freeze.get("candidate_id") != EXPECTED_CANDIDATE_ID or freeze.get("status") != "frozen_pre_formal_evaluation":
        errors.append("candidate_not_uniquely_frozen")
    if audit.get("status") != "passed" or audit.get("error_count") != 0:
        errors.append("ens_audit_not_passed")
    if audit.get("formal_evaluation_config_sha256") != config_hash or freeze.get("formal_evaluation_config_sha256") != config_hash:
        errors.append("formal_config_hash_mismatch")
    if audit.get("frozen_candidate_sha256") != freeze_hash:
        errors.append("frozen_candidate_hash_mismatch")
    if freeze.get("selection_report_sha256") != selection_hash or selection.get("split_sha256") != split_hash:
        errors.append("selection_or_split_hash_mismatch")
    if selection.get("artifact_manifest_sha256") != sha256_file(artifact_manifest_path):
        errors.append("artifact_manifest_hash_mismatch")
    if len(freeze.get("outer_contracts", [])) != 24 or len(split.get("outer_folds", [])) != 24:
        errors.append("outer_contract_count")
    if config.get("final_seeds") != [20260806, 20260817, 20260829]:
        errors.append("final_seed_contract")
    if config.get("formal_evaluation_started") is not False or any(audit.get("guards", {}).values()):
        errors.append("formal_evaluation_already_started")
    if errors:
        raise RuntimeError(f"Formal freeze gate failed: {errors}")
    contracts = materialize_application_contracts(selection=selection, freeze=freeze, config=config)
    return {
        "schema_version": RESOLVED_SCHEMA,
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "status": "resolved_before_formal_evaluation",
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "paths": {
            "formal_config": config_path.resolve().as_posix(),
            "frozen_candidate": freeze_path.resolve().as_posix(),
            "ens_independent_audit": audit_path.resolve().as_posix(),
            "selection_report": selection_path.resolve().as_posix(),
            "split": split_path.resolve().as_posix(),
            "artifact_manifest": artifact_manifest_path.resolve().as_posix(),
        },
        "hashes": {
            "formal_config_sha256": config_hash,
            "frozen_candidate_sha256": freeze_hash,
            "ens_independent_audit_sha256": sha256_file(audit_path),
            "selection_report_sha256": selection_hash,
            "split_sha256": split_hash,
            "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
        },
        "formal_config": config,
        "application_contracts": contracts,
        "formal_evaluation_started": False,
        "outer_test_metrics_generated": False,
        "outer_test_predictions_generated": False,
    }


def train_deep_final(
    store: MemoryArtifactStore,
    *,
    train_ids: Sequence[str],
    candidate: TrainingCandidate,
    seed: int,
    epochs: int,
    ssl_epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[ApexFusionNet, list[dict[str, Any]]]:
    set_seed(seed)
    dataset = store.dataset(train_ids, candidate)
    loader = _loader(dataset, batch_size=batch_size, seed=seed, training=True)
    model = ApexFusionNet(ApexFusionConfig(temporal=candidate.temporal, temporal_layers=1)).to(device)
    _ssl_pretrain(model, loader, candidate, device, seed, ssl_epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    labels = [int(store.rows[sample_id]["three_class_id"]) for sample_id in train_ids]
    counts = torch.tensor([labels.count(label) for label in range(3)], dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(seed + 17)
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for raw in loader:
            batch = _to_device(raw, device)
            _augment(batch, candidate, generator)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            main = _classification_loss(output["main_logits"], batch["label"], counts, candidate.loss_kind)
            auxiliary = torch.nn.functional.cross_entropy(output["aux_logits"], batch["aux_label"])
            from .casme2_apexfusion import multitask_loss

            supcon = multitask_loss(output, batch["label"], batch["aux_label"], aux_weight=0.0, supcon_weight=1.0)["supcon"]
            loss = main + candidate.aux_weight * auxiliary + candidate.supcon_weight * supcon
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch["label"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(dataset),
                "gradient_norm_last_batch": float(gradient_norm.detach().cpu()),
                "sample_count": len(dataset),
            }
        )
    return model, history


def strict_state_reload_difference(left: ApexFusionNet, right: ApexFusionNet) -> float:
    differences = []
    for key, value in left.state_dict().items():
        other = right.state_dict()[key]
        differences.append(float((value.detach().cpu() - other.detach().cpu()).abs().max()))
    return max(differences, default=0.0)


def fit_traditional_final(
    store: MemoryArtifactStore,
    *,
    train_ids: Sequence[str],
    candidate: Mapping[str, Any],
) -> tuple[Any, np.ndarray, np.ndarray]:
    groups = tuple(str(value) for value in candidate["feature_groups"])
    x_train = np.stack([feature_vector(store.arrays[sample_id], groups) for sample_id in train_ids])
    y_train = np.asarray([store.rows[sample_id]["three_class_id"] for sample_id in train_ids], dtype=np.int64)
    config = TraditionalConfig(feature_groups=groups, pca_components=16, kernel=str(candidate["kernel"]))
    pipeline = build_pipeline(config, max_samples=len(x_train))
    pipeline.fit(x_train, y_train)
    return pipeline, x_train, y_train


def aligned_traditional_probabilities(pipeline: Any, matrix: np.ndarray) -> np.ndarray:
    probabilities = pipeline.predict_proba(matrix)
    classes = [int(value) for value in pipeline.named_steps["classifier"].classes_]
    aligned = np.zeros((len(probabilities), 3), dtype=np.float64)
    aligned[:, classes] = probabilities
    return normalize(aligned)


def metrics_from_rows(rows: Sequence[Mapping[str, Any]], probability_key: str = "probabilities") -> dict[str, Any]:
    labels = [int(row["true_label"]) for row in rows]
    predictions = [int(np.argmax(row[probability_key])) for row in rows]
    return classification_metrics(labels, predictions)


def subject_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    *,
    probability_key: str = "probabilities",
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, list[float]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["subject_id"])].append(row)
    subjects = sorted(grouped)
    rng = np.random.default_rng(seed)
    values = {"uf1": [], "uar": [], "accuracy": []}
    for _ in range(iterations):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        sample_rows = [row for subject in sampled for row in grouped[str(subject)]]
        metrics = metrics_from_rows(sample_rows, probability_key)
        for metric in values:
            values[metric].append(float(metrics[metric]))
    return {
        metric: [float(np.quantile(metric_values, 0.025)), float(np.quantile(metric_values, 0.975))]
        for metric, metric_values in values.items()
    }


def aggregate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    probability_key: str = "probabilities",
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    pooled = metrics_from_rows(rows, probability_key)
    subjects = sorted({str(row["subject_id"]) for row in rows})
    per_subject = {
        subject: metrics_from_rows([row for row in rows if str(row["subject_id"]) == subject], probability_key)
        for subject in subjects
    }
    return {
        "sample_count": len(rows),
        "subject_count": len(subjects),
        "pooled": pooled,
        "per_subject": per_subject,
        "zero_score_subjects": [subject for subject, value in per_subject.items() if value["uf1"] <= 0.0 or value["uar"] <= 0.0],
        "bootstrap_95_ci": subject_bootstrap_ci(rows, probability_key=probability_key, seed=bootstrap_seed),
    }


def ensemble_prediction_rows(seed_rows: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    if len(seed_rows) != 3:
        raise ValueError("Three seed prediction populations are required")
    maps = [{str(row["sample_id"]): row for row in rows} for rows in seed_rows]
    sample_ids = sorted(maps[0])
    if any(sorted(mapping) != sample_ids for mapping in maps[1:]):
        raise ValueError("Seed prediction populations are not aligned")
    output = []
    for sample_id in sample_ids:
        source = maps[0][sample_id]
        probabilities = equal_seed_probability_ensemble(
            [np.asarray(mapping[sample_id]["probabilities"], dtype=np.float64)[None, :] for mapping in maps]
        )[0]
        output.append(
            {
                "sample_id": sample_id,
                "subject_id": source["subject_id"],
                "true_label": int(source["true_label"]),
                "predicted_label": int(np.argmax(probabilities)),
                "probabilities": probabilities.tolist(),
                "source_seeds": [int(mapping[sample_id]["seed"]) for mapping in maps],
                "ensemble": "equal_probability_mean",
            }
        )
    return output


def selection_ablation_summary(selection: Mapping[str, Any]) -> dict[str, Any]:
    def summarize(key: str) -> list[dict[str, Any]]:
        registry_key = "candidate_registry" if key == "deep_candidates" else "traditional_registry"
        id_key = "candidate_id"
        rows = []
        for registry in selection[registry_key]:
            candidate_id = registry[id_key]
            metrics = [
                next(value for value in outer[key] if value["candidate"]["candidate_id"] == candidate_id)["metrics"]
                for outer in selection["outer_results"]
            ]
            entry = {"candidate": registry}
            for metric in ("uf1", "uar", "accuracy"):
                values = [float(value[metric]) for value in metrics]
                entry[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}
            entry["positive_f1_mean"] = float(np.mean([value["per_class"]["positive"]["f1"] for value in metrics]))
            entry["positive_recall_mean"] = float(np.mean([value["per_class"]["positive"]["recall"] for value in metrics]))
            rows.append(entry)
        return rows

    return {
        "scope": "training_side_outer_train_inner_oof_only_not_formal_outer_test",
        "deep_candidates": summarize("deep_candidates"),
        "traditional_candidates": summarize("traditional_candidates"),
    }


def result_is_complete_and_valid(result_path: Path, *, resolved_sha256: str) -> bool:
    if not result_path.exists():
        return False
    result = read_json(result_path)
    if result.get("status") != "completed" or result.get("resolved_contract_sha256") != resolved_sha256:
        raise RuntimeError(f"Existing formal result is incompatible: {result_path}")
    for key in ("deep_checkpoint", "traditional_checkpoint", "epoch_log", "prediction_file"):
        payload = result[key]
        path = Path(payload["path"])
        if not path.exists() or sha256_file(path) != payload["sha256"]:
            raise RuntimeError(f"Existing formal result artifact is corrupted: {path}")
    return True


def run_fold_seed(
    *,
    store: MemoryArtifactStore,
    fold: Mapping[str, Any],
    application_contract: Mapping[str, Any],
    seed: int,
    device: torch.device,
    output_dir: Path,
    resolved_sha256: str,
    formal_config: Mapping[str, Any],
) -> dict[str, Any]:
    result_path = output_dir / "result.json"
    if result_is_complete_and_valid(result_path, resolved_sha256=resolved_sha256):
        return read_json(result_path)
    recovery_archive: str | None = None
    if output_dir.exists() and any(output_dir.iterdir()):
        recovery_root = output_dir.parent / "_technical_recovery"
        recovery_root.mkdir(parents=True, exist_ok=True)
        attempt = 1
        archived = recovery_root / f"{output_dir.name}_attempt_{attempt:02d}"
        while archived.exists():
            attempt += 1
            archived = recovery_root / f"{output_dir.name}_attempt_{attempt:02d}"
        shutil.move(str(output_dir), str(archived))
        recovery_archive = archived.resolve().as_posix()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train_ids = list(fold["outer_train_sample_ids"])
    test_ids = list(fold["outer_test_sample_ids"])
    candidate = candidate_from_mapping(application_contract["deep_candidate"])
    final_epoch = int(application_contract["epoch_selection"]["final_epoch"])
    ssl_epochs = int(formal_config["training_budget"]["ssl_epochs"]) if candidate.ssl != "none" else 0
    model, history = train_deep_final(
        store,
        train_ids=train_ids,
        candidate=candidate,
        seed=seed,
        epochs=final_epoch,
        ssl_epochs=ssl_epochs,
        batch_size=int(formal_config["training_budget"]["batch_size"]),
        device=device,
    )
    epoch_path = output_dir / "epoch_log.jsonl"
    epoch_hash = write_jsonl(epoch_path, history)
    deep_checkpoint = output_dir / "deep_checkpoint.pt"
    save_checkpoint(
        deep_checkpoint,
        model,
        metadata={
            "task_id": "EVAL-ME2-001",
            "candidate_id": EXPECTED_CANDIDATE_ID,
            "outer_fold": int(fold["outer_fold"]),
            "seed": int(seed),
            "train_sample_ids_sha256": sequence_sha256(train_ids),
            "test_sample_ids_sha256": sequence_sha256(test_ids),
            "test_used_for_selection": False,
            "resolved_contract_sha256": resolved_sha256,
        },
    )
    loaded_model, loaded_metadata = load_checkpoint(deep_checkpoint, map_location=device)
    loaded_model = loaded_model.to(device)
    strict_difference = strict_state_reload_difference(model, loaded_model)
    if strict_difference != 0.0 or loaded_metadata["resolved_contract_sha256"] != resolved_sha256:
        raise RuntimeError("Deep checkpoint strict reload failed")

    pipeline, x_train, _ = fit_traditional_final(store, train_ids=train_ids, candidate=application_contract["traditional_candidate"])
    traditional_checkpoint = output_dir / "traditional_checkpoint.joblib"
    traditional_config = TraditionalConfig(
        feature_groups=tuple(application_contract["traditional_candidate"]["feature_groups"]),
        pca_components=16,
        kernel=str(application_contract["traditional_candidate"]["kernel"]),
    )
    save_traditional(
        traditional_checkpoint,
        pipeline,
        traditional_config,
        {
            "task_id": "EVAL-ME2-001",
            "candidate_id": EXPECTED_CANDIDATE_ID,
            "outer_fold": int(fold["outer_fold"]),
            "seed": int(seed),
            "train_sample_ids_sha256": sequence_sha256(train_ids),
            "test_sample_ids_sha256": sequence_sha256(test_ids),
            "test_used_for_selection": False,
            "resolved_contract_sha256": resolved_sha256,
        },
    )
    loaded_traditional = load_traditional(traditional_checkpoint)
    original_svc = pipeline.named_steps["classifier"]
    loaded_svc = loaded_traditional["pipeline"].named_steps["classifier"]
    for attribute in ("support_", "support_vectors_", "dual_coef_", "intercept_", "probA_", "probB_", "classes_"):
        if not np.array_equal(getattr(original_svc, attribute), getattr(loaded_svc, attribute)):
            raise RuntimeError(f"Traditional checkpoint state mismatch: {attribute}")
    probe_left = aligned_traditional_probabilities(pipeline, x_train[: min(4, len(x_train))])
    probe_right = aligned_traditional_probabilities(loaded_traditional["pipeline"], x_train[: min(4, len(x_train))])
    traditional_reload_difference = float(np.max(np.abs(probe_left - probe_right)))
    # libsvm's probability coupling can differ at sub-micro precision after a
    # joblib roundtrip even when support vectors and dual coefficients are bitwise
    # identical.  The tolerance is fixed before formal evaluation and remains far
    # below any value capable of changing a three-class decision here.
    if traditional_reload_difference > 1e-6:
        raise RuntimeError("Traditional checkpoint strict reload failed")

    test_dataset = store.dataset(test_ids, candidate)
    test_loader = _loader(test_dataset, batch_size=int(formal_config["training_budget"]["batch_size"]), seed=seed, training=False)
    deep_rows = predict(loaded_model, test_loader, device)
    groups = tuple(application_contract["traditional_candidate"]["feature_groups"])
    x_test = np.stack([feature_vector(store.arrays[sample_id], groups) for sample_id in test_ids])
    traditional_probabilities = aligned_traditional_probabilities(loaded_traditional["pipeline"], x_test)
    deep_by_id = {row["sample_id"]: row for row in deep_rows}
    raw_deep = np.stack([deep_by_id[sample_id]["probabilities"] for sample_id in test_ids])
    calibrated_deep = apply_calibrator(application_contract["final_calibrator"], raw_deep)
    deep_weight = float(application_contract["final_fusion"]["deep_weight"])
    fusion = normalize(deep_weight * calibrated_deep + (1.0 - deep_weight) * traditional_probabilities)
    selected_output = str(application_contract["selected_output"])
    selected = {"deep": calibrated_deep, "traditional": traditional_probabilities, "deep_traditional_fusion": fusion}[selected_output]
    prediction_rows = []
    for index, sample_id in enumerate(test_ids):
        row = store.rows[sample_id]
        probability = selected[index]
        prediction_rows.append(
            {
                "sample_id": sample_id,
                "subject_id": row["subject_id"],
                "true_label": int(row["three_class_id"]),
                "predicted_label": int(np.argmax(probability)),
                "probabilities": probability.tolist(),
                "raw_deep_probabilities": normalize(raw_deep)[index].tolist(),
                "calibrated_deep_probabilities": calibrated_deep[index].tolist(),
                "traditional_probabilities": traditional_probabilities[index].tolist(),
                "fusion_probabilities": fusion[index].tolist(),
                "selected_output": selected_output,
                "outer_fold": int(fold["outer_fold"]),
                "seed": int(seed),
                "test_used_for_selection": False,
                "outer_test_forward_count": 1,
            }
        )
    prediction_path = output_dir / "test_predictions.jsonl"
    prediction_hash = write_jsonl(prediction_path, prediction_rows)
    result = {
        "schema_version": FORMAL_SCHEMA,
        "task_id": "EVAL-ME2-001",
        "status": "completed",
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "outer_fold": int(fold["outer_fold"]),
        "seed": int(seed),
        "test_subject": fold["test_subject"],
        "selected_deep_candidate": application_contract["selected_deep_candidate"],
        "selected_traditional_candidate": application_contract["selected_traditional_candidate"],
        "selected_calibration": application_contract["selected_calibration"],
        "selected_output": selected_output,
        "final_epoch": final_epoch,
        "final_fusion": application_contract["final_fusion"],
        "partitions": {
            "train": {"sample_count": len(train_ids), "sample_ids_sha256": sequence_sha256(train_ids), "subject_ids": sorted({store.rows[sample_id]["subject_id"] for sample_id in train_ids})},
            "test": {"sample_count": len(test_ids), "sample_ids_sha256": sequence_sha256(test_ids), "subject_ids": [fold["test_subject"]]},
        },
        "metrics": metrics_from_rows(prediction_rows),
        "component_metrics": {
            "deep": metrics_from_rows(prediction_rows, "calibrated_deep_probabilities"),
            "traditional": metrics_from_rows(prediction_rows, "traditional_probabilities"),
            "deep_traditional_fusion": metrics_from_rows(prediction_rows, "fusion_probabilities"),
        },
        "deep_checkpoint": {"path": deep_checkpoint.resolve().as_posix(), "sha256": sha256_file(deep_checkpoint), "strict_reload_max_state_difference": strict_difference},
        "traditional_checkpoint": {"path": traditional_checkpoint.resolve().as_posix(), "sha256": sha256_file(traditional_checkpoint), "strict_reload_max_probability_difference": traditional_reload_difference},
        "epoch_log": {"path": epoch_path.resolve().as_posix(), "sha256": epoch_hash, "epoch_count": len(history)},
        "prediction_file": {"path": prediction_path.resolve().as_posix(), "sha256": prediction_hash, "sample_count": len(prediction_rows)},
        "resolved_contract_sha256": resolved_sha256,
        "test_used_for_selection": False,
        "outer_test_forward_count_per_sample": 1,
        "technical_recovery_archive": recovery_archive,
        "duration_seconds": time.perf_counter() - started,
        "environment": {"device": str(device), "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "torch_version": torch.__version__},
    }
    write_json(result_path, result)
    del model, loaded_model, pipeline, loaded_traditional
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
