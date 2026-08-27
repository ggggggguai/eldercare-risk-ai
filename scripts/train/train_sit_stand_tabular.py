from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from elderly_monitoring.modules.fall_risk.sit_stand_tabular import (
    MODEL_VERSION,
    SCHEMA_VERSION,
    aggregate_event_scores,
    binary_event_metrics,
    select_f1_threshold,
    split_validation_groups,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and confirm the scoped sit-stand clip-presence candidate."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=600)
    return parser


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if metadata.get("dataset_sha256") != _sha256(args.data):
        raise ValueError("sit-stand dataset SHA-256 mismatch")
    if metadata.get("task") != "sit_stand_event_presence_proxy_v1":
        raise ValueError("sit-stand dataset task is incompatible")
    if metadata.get("test_pose_read") is not False:
        raise ValueError("sit-stand dataset violates the locked-test contract")
    with np.load(args.data, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    partitions = arrays["partitions"].astype(str)
    if set(partitions.tolist()) != {"train", "validation"}:
        raise ValueError("sit-stand candidate accepts train/validation tensors only")
    train_indices = np.flatnonzero(partitions == "train")
    validation_indices = np.flatnonzero(partitions == "validation")
    selection_mask, confirmation_mask = split_validation_groups(
        arrays["split_group_ids"][validation_indices],
        arrays["labels"][validation_indices],
    )
    selection_indices = validation_indices[selection_mask]
    confirmation_indices = validation_indices[confirmation_mask]

    model = RandomForestClassifier(
        n_estimators=args.trees,
        min_samples_leaf=2,
        max_features=None,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(
        arrays["tabular_features"][train_indices],
        arrays["labels"][train_indices],
        sample_weight=arrays["sample_weights"][train_indices],
    )
    scores = model.predict_proba(arrays["tabular_features"])[:, 1]
    shared = {
        "labels": arrays["labels"],
        "event_ids": arrays["event_ids"],
        "split_group_ids": arrays["split_group_ids"],
        "datasets": arrays["datasets"],
        "action_ids": arrays["action_ids"],
    }
    selection_rows = aggregate_event_scores(
        selection_indices, scores[selection_indices], **shared
    )
    confirmation_rows = aggregate_event_scores(
        confirmation_indices, scores[confirmation_indices], **shared
    )
    threshold, selection_metrics = select_f1_threshold(selection_rows)
    confirmation_metrics = binary_event_metrics(confirmation_rows, threshold)

    args.output_dir.mkdir(parents=True)
    checkpoint_path = args.output_dir / "best_model.joblib"
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "model": model,
        "threshold": threshold,
        "feature_names": metadata["tabular_feature_names"],
        "training_contract": {
            "seed": args.seed,
            "tree_count": args.trees,
            "dataset_sha256": _sha256(args.data),
            "train_only_fit": True,
            "selection_only_threshold": True,
            "confirmation_excluded_from_fit_and_threshold": True,
            "test_pose_read": False,
            "test_evaluated": False,
        },
    }
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(checkpoint, temporary, compress=3)
        os.replace(temporary, checkpoint_path)
    finally:
        temporary.unlink(missing_ok=True)
    _write_jsonl(args.output_dir / "selection_predictions.jsonl", selection_rows)
    _write_jsonl(args.output_dir / "confirmation_predictions.jsonl", confirmation_rows)
    metrics = {
        "schema_version": "sit-stand-tabular-training-metrics-v1",
        "status": "development_provisional",
        "task": "sit_stand_clip_presence_vs_explicit_hard_negatives",
        "model_version": MODEL_VERSION,
        "selection": selection_metrics,
        "confirmation": confirmation_metrics,
        "confirmation_gate": {
            "precision_min": 0.85,
            "recall_min": 0.90,
            "f1_min": 0.90,
            "pr_auc_min": 0.95,
            "passed": bool(
                confirmation_metrics["precision"] >= 0.85
                and confirmation_metrics["recall"] >= 0.90
                and confirmation_metrics["f1"] >= 0.90
                and confirmation_metrics["pr_auc"] >= 0.95
            ),
        },
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dataset_sha256": _sha256(args.data),
        "test_pose_read": False,
        "test_evaluated": False,
        "limitations": [
            "The task is action-clip presence, not continuous temporal localization.",
            "The confirmation partition is internal development evidence, not the locked test.",
            "The model is not runtime-compatible with the streaming sit-stand branch.",
        ],
    }
    _write_json(args.output_dir / "metrics.json", metrics)
    return {
        "checkpoint_path": checkpoint_path.as_posix(),
        "metrics_path": (args.output_dir / "metrics.json").as_posix(),
        "confirmation_f1": confirmation_metrics["f1"],
        "confirmation_gate_passed": metrics["confirmation_gate"]["passed"],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = train(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
