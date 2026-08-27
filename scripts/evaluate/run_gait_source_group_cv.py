#!/usr/bin/env python3
"""Run leakage-controlled source-group-out evaluation for gait instability."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from elderly_monitoring.modules.fall_risk.gait_action_pretraining import (
    ActionTCNPretrainingConfig,
    train_action_pretraining_tcn,
)
from elderly_monitoring.modules.fall_risk.gait_tcn import (
    GaitTCNTrainingConfig,
    LightweightGaitTCN,
    _binary_metrics,
    train_gait_tcn,
)


DEFAULT_GAIT_DATASET = Path(
    "data/processed/fall_risk/gait_observable_context_v2/"
    "splitv3-3342705-scfaux-v1/dataset.npz"
)
DEFAULT_ACTION_DATASET = Path(
    "data/processed/fall_risk/action_pretraining_v1/"
    "splitv3-3342705-scfaux-v1/dataset.npz"
)
DEFAULT_OUTPUT = Path("reports/fall_risk/gait_source_group_cv_v1")
POSITIVE_ACTION_IDS = {"B02", "B03", "B04"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gait-dataset", type=Path, default=DEFAULT_GAIT_DATASET)
    parser.add_argument("--action-dataset", type=Path, default=DEFAULT_ACTION_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--action-epochs", type=int, default=30)
    parser.add_argument("--gait-epochs", type=int, default=50)
    parser.add_argument(
        "--holdout-source-group",
        action="append",
        default=[],
        help="run only the named holdout group; repeat for multiple groups",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    return normalized or hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def positive_source_groups(arrays: Mapping[str, np.ndarray]) -> list[str]:
    source_groups = arrays["source_group_ids"].astype(str)
    labels = arrays["labels"].astype(np.int64)
    primary = arrays.get(
        "primary_evaluation_mask", np.ones(len(labels), dtype=np.uint8)
    ).astype(bool)
    return sorted(set(source_groups[(labels == 1) & primary].tolist()))


def derive_action_partitions(
    arrays: Mapping[str, np.ndarray], holdout_source_group: str
) -> np.ndarray:
    original = arrays["partitions"].astype(str)
    source_groups = arrays["source_group_ids"].astype(str)
    partitions = original.copy()
    partitions[source_groups == holdout_source_group] = "excluded"
    if np.any((source_groups == holdout_source_group) & (partitions != "excluded")):
        raise AssertionError("holdout source entered action pretraining")
    if not np.any(partitions == "train") or not np.any(partitions == "validation"):
        raise ValueError("derived action fold requires train and internal validation rows")
    return partitions


def derive_gait_partitions(
    arrays: Mapping[str, np.ndarray], holdout_source_group: str
) -> np.ndarray:
    source_groups = arrays["source_group_ids"].astype(str)
    partitions = np.full(len(source_groups), "train", dtype="<U10")
    partitions[source_groups == holdout_source_group] = "validation"
    if not np.any(partitions == "validation"):
        raise ValueError(f"unknown gait holdout source group: {holdout_source_group}")
    return partitions


def _write_derived_dataset(
    *,
    source_path: Path,
    arrays: Mapping[str, np.ndarray],
    partitions: np.ndarray,
    destination: Path,
    holdout_source_group: str,
    dataset_role: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    dataset_path = destination / "dataset.npz"
    metadata_path = destination / "metadata.json"
    if not overwrite and (dataset_path.exists() or metadata_path.exists()):
        if dataset_path.exists() and metadata_path.exists():
            return dataset_path, metadata_path
        raise FileExistsError(f"incomplete derived fold dataset: {destination}")

    derived = dict(arrays)
    derived["partitions"] = partitions
    np.savez_compressed(dataset_path, **derived)

    source_metadata_path = source_path.with_name("metadata.json")
    metadata = deepcopy(json.loads(source_metadata_path.read_text(encoding="utf-8")))
    metadata.update(
        {
            "dataset_sha256": _sha256(dataset_path),
            "split_protocol": "source_group_out_cv_v1",
            "source_split_id": None,
            "source_split_report_path": None,
            "source_split_report_sha256": None,
            "partition_schemes": {"frozen": "partitions"},
            "test_pose_read": False,
            "test_tensor_generated": False,
            "test_evaluated": False,
            "cv_holdout_source_group": holdout_source_group,
            "cv_dataset_role": dataset_role,
            "cv_source_dataset_path": source_path.as_posix(),
            "cv_source_dataset_sha256": _sha256(source_path),
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return dataset_path, metadata_path


def _score_holdout(
    checkpoint_path: Path,
    arrays: Mapping[str, np.ndarray],
    holdout_source_group: str,
    *,
    device: str,
    threshold: float,
    batch_size: int = 128,
) -> list[dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = LightweightGaitTCN(**dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    selected_device = _select_device(device)
    model.to(selected_device).eval()

    labels = arrays["labels"].astype(np.int64)
    source_groups = arrays["source_group_ids"].astype(str)
    primary = arrays.get(
        "primary_evaluation_mask", np.ones(len(labels), dtype=np.uint8)
    ).astype(bool)
    indices = np.flatnonzero((source_groups == holdout_source_group) & primary)
    if set(labels[indices].tolist()) != {0, 1}:
        raise ValueError(f"holdout source lacks both gait classes: {holdout_source_group}")

    features = arrays["features"].astype(np.float32, copy=False)
    masks = arrays.get(
        "label_span_masks", np.ones(features.shape[:2], dtype=np.float32)
    ).astype(np.float32, copy=False)
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            gait_logits, walking_logits = model.forward_heads(
                torch.from_numpy(features[batch_indices]).to(selected_device),
                torch.from_numpy(masks[batch_indices]).to(selected_device),
            )
            gait_probability = torch.softmax(gait_logits, dim=1)[:, 1]
            if walking_logits is not None:
                gait_probability *= torch.softmax(walking_logits, dim=1)[:, 1]
            probabilities.append(gait_probability.cpu().numpy())
    scores = np.concatenate(probabilities)

    grouped: dict[str, dict[str, Any]] = {}
    segment_ids = arrays["action_segment_ids"].astype(str)
    action_ids = arrays["action_ids"].astype(str)
    datasets = arrays.get("datasets", np.full(len(labels), "unknown")).astype(str)
    for index, score in zip(indices, scores, strict=True):
        segment_id = segment_ids[index]
        row = grouped.setdefault(
            segment_id,
            {
                "action_segment_id": segment_id,
                "source_group_id": holdout_source_group,
                "dataset": datasets[index],
                "action_id": action_ids[index],
                "label": int(labels[index]),
                "window_probabilities": [],
            },
        )
        if row["label"] != int(labels[index]) or row["action_id"] != action_ids[index]:
            raise ValueError(f"inconsistent gait segment metadata: {segment_id}")
        row["window_probabilities"].append(float(score))

    output: list[dict[str, Any]] = []
    for segment_id in sorted(grouped):
        row = grouped[segment_id]
        probability = float(np.mean(row.pop("window_probabilities")))
        row.update(
            {
                "probability": probability,
                "predicted_label": int(probability >= threshold),
                "threshold": threshold,
            }
        )
        output.append(row)
    return output


def _select_device(requested: str) -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(normalized)


def evaluation_metrics(
    rows: Sequence[Mapping[str, Any]], *, threshold: float
) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["probability"]) for row in rows], dtype=np.float64)
    metrics = _binary_metrics(labels, scores, threshold=threshold)
    metrics["pr_auc"] = (
        float(average_precision_score(labels, scores))
        if set(labels.tolist()) == {0, 1}
        else None
    )
    metrics["roc_auc"] = (
        float(roc_auc_score(labels, scores))
        if set(labels.tolist()) == {0, 1}
        else None
    )
    return metrics


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be within (0, 1)")
    gait_arrays = _load_npz(args.gait_dataset)
    action_arrays = _load_npz(args.action_dataset)
    holdout_groups = positive_source_groups(gait_arrays)
    if len(holdout_groups) < 3:
        raise ValueError("source-group CV requires at least three positive source groups")
    if args.holdout_source_group:
        unknown = sorted(set(args.holdout_source_group) - set(holdout_groups))
        if unknown:
            raise ValueError(f"unknown positive holdout source groups: {unknown}")
        holdout_groups = [
            group for group in holdout_groups if group in set(args.holdout_source_group)
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for fold_index, holdout_group in enumerate(holdout_groups, start=1):
        fold_dir = args.output_dir / f"fold-{fold_index:02d}-{_slug(holdout_group)}"
        action_dataset, action_metadata = _write_derived_dataset(
            source_path=args.action_dataset,
            arrays=action_arrays,
            partitions=derive_action_partitions(action_arrays, holdout_group),
            destination=fold_dir / "action_dataset",
            holdout_source_group=holdout_group,
            dataset_role="encoder_pretraining_without_holdout_source",
            overwrite=args.overwrite,
        )
        gait_dataset, gait_metadata = _write_derived_dataset(
            source_path=args.gait_dataset,
            arrays=gait_arrays,
            partitions=derive_gait_partitions(gait_arrays, holdout_group),
            destination=fold_dir / "gait_dataset",
            holdout_source_group=holdout_group,
            dataset_role="gait_source_group_holdout",
            overwrite=args.overwrite,
        )

        action_result = train_action_pretraining_tcn(
            action_dataset,
            fold_dir / "action_pretraining",
            metadata_path=action_metadata,
            config=ActionTCNPretrainingConfig(
                epochs=args.action_epochs,
                patience=min(6, args.action_epochs),
                seed=42,
                device=args.device,
            ),
            overwrite=args.overwrite,
        )
        gait_result = train_gait_tcn(
            gait_dataset,
            fold_dir / "gait_tcn",
            metadata_path=gait_metadata,
            config=GaitTCNTrainingConfig(
                epochs=args.gait_epochs,
                patience=min(8, args.gait_epochs),
                seed=43,
                device=args.device,
                evaluate_test=False,
                pretrained_checkpoint=action_result["checkpoint_path"],
                freeze_encoder_epochs=min(5, args.gait_epochs - 1),
                hierarchical_walking_gate=True,
            ),
            overwrite=args.overwrite,
        )
        rows = _score_holdout(
            Path(gait_result["checkpoint_path"]),
            gait_arrays,
            holdout_group,
            device=args.device,
            threshold=args.threshold,
        )
        all_rows.extend(rows)
        fold_reports.append(
            {
                "fold": fold_index,
                "holdout_source_group": holdout_group,
                "metrics": evaluation_metrics(rows, threshold=args.threshold),
                "positive_subtype_support": {
                    action_id: sum(
                        int(row["label"]) == 1 and row["action_id"] == action_id
                        for row in rows
                    )
                    for action_id in sorted(POSITIVE_ACTION_IDS)
                },
                "action_pretraining": action_result,
                "gait_training": gait_result,
            }
        )
        print(
            json.dumps(
                {
                    "fold": fold_index,
                    "holdout": holdout_group,
                    "metrics": fold_reports[-1]["metrics"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    overall = evaluation_metrics(all_rows, threshold=args.threshold)
    report = {
        "schema_version": "gait-source-group-cv-report-v1",
        "protocol": {
            "evaluation_unit": "action_segment",
            "split": "leave_one_positive_source_group_out",
            "encoder_holdout_policy": (
                "holdout source excluded from encoder fit and encoder early stopping"
            ),
            "gait_holdout_policy": "holdout source used only for fold evaluation",
            "threshold": args.threshold,
            "threshold_policy": "fixed_before_oof_scoring",
            "test_pose_read": False,
            "test_tensor_generated": False,
            "test_evaluated": False,
        },
        "source_datasets": {
            "gait_path": args.gait_dataset.as_posix(),
            "gait_sha256": _sha256(args.gait_dataset),
            "action_path": args.action_dataset.as_posix(),
            "action_sha256": _sha256(args.action_dataset),
        },
        "holdout_source_groups": holdout_groups,
        "overall_oof": overall,
        "positive_subtype_support": {
            action_id: sum(
                int(row["label"]) == 1 and row["action_id"] == action_id
                for row in all_rows
            )
            for action_id in sorted(POSITIVE_ACTION_IDS)
        },
        "folds": fold_reports,
    }
    _write_jsonl(args.output_dir / "oof_predictions.jsonl", all_rows)
    _write_json(args.output_dir / "report.json", report)
    print(json.dumps({"overall_oof": overall}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
