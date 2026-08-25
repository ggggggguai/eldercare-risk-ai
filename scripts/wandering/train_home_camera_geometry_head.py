from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterInput,
    canonical_json_bytes,
    load_camera_inputs,
    weighted_bucket_observations,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    load_camera_config,
)


ROOT = Path(__file__).resolve().parents[2]
CLASS_ORDER = ("direct", "pacing", "lapping", "random")
FEATURE_NAMES = (
    "log_duration_seconds",
    "log_bucket_count",
    "path_body_heights",
    "path_body_heights_per_second",
    "extent_body_heights",
    "net_body_heights",
    "net_to_path_ratio",
    "closure_to_extent_ratio",
    "minor_major_axis_ratio",
    "pause_ratio_010",
    "pause_ratio_015",
    "pause_ratio_020",
    "pause_ratio_030",
    "reversal_count_per_minute",
    "reversal_step_fraction",
    "mean_absolute_turn_radians",
    "large_turn_fraction",
    "radial_distance_cv",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a lightweight home camera geometry head from temporal/behavior "
            "development labels; rectangles are not tracking truth."
        )
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--labeled-import-root", type=Path, required=True)
    parser.add_argument("--boundary-eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.project_root.resolve(strict=True)
    tracking_root = args.tracking_root.resolve(strict=True)
    labeled_root = args.labeled_import_root.resolve(strict=True)
    boundary_eval_root = args.boundary_eval_root.resolve(strict=True)
    output = args.output_dir.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"geometry-head output already exists: {output}")
    camera_config = load_camera_config(
        root / "configs/modules/wandering_camera_v1.yaml"
    )
    adapters: dict[str, CameraAdapterInput] = {}
    truth_by_episode: dict[str, dict[str, Any]] = {}
    boundaries_by_episode: dict[str, dict[str, Any]] = {}
    proposals_by_id: dict[str, dict[str, Any]] = {}
    for task_dir in sorted((labeled_root / "tasks").iterdir()):
        if not task_dir.is_dir():
            continue
        video_id = task_dir.name
        adapters[video_id] = load_camera_inputs(
            tracking_root / video_id / "inputs/tracking.jsonl",
            task_dir / "media_sidecar.labeled.json",
            camera_config,
        )
        for row in _load_jsonl(task_dir / "episode_truth.jsonl"):
            truth_by_episode[str(row["episode_id"])] = row
        for row in _load_jsonl(task_dir / "episode_boundaries.jsonl"):
            boundaries_by_episode[str(row["episode_id"])] = row
        for row in _load_jsonl(task_dir / "proposals/proposals.jsonl"):
            if row["proposal_status"] != "rejected_by_qc":
                proposals_by_id[str(row["proposal_id"])] = row

    samples: list[dict[str, Any]] = []
    for episode_id, truth in sorted(truth_by_episode.items()):
        if str(truth["observable_pattern"]) not in CLASS_ORDER:
            continue
        boundary = boundaries_by_episode[episode_id]
        feature = _geometry_features(
            adapters[str(boundary["source_video_id"])],
            track_id=int(boundary["target_track_id"]),
            start_sec=float(boundary["start_sec"]),
            end_sec=float(boundary["end_sec_exclusive"]),
        )
        if feature is not None:
            samples.append(
                _sample_row(
                    sample_id=f"oracle:{episode_id}",
                    sample_role="oracle_temporal_development",
                    source_video_id=str(boundary["source_video_id"]),
                    label=str(truth["observable_pattern"]),
                    features=feature,
                )
            )

    matched_episode_ids: set[str] = set()
    for match in _load_jsonl(boundary_eval_root / "matches.jsonl"):
        if match["view"] != "all_locomotion_candidates":
            continue
        episode_id = str(match["episode_id"])
        proposal_id = str(match["proposal_id"])
        if episode_id in matched_episode_ids:
            continue
        matched_episode_ids.add(episode_id)
        proposal = proposals_by_id[proposal_id]
        truth = truth_by_episode[episode_id]
        if str(truth["observable_pattern"]) not in CLASS_ORDER:
            continue
        feature = _geometry_features(
            adapters[str(proposal["source_video_id"])],
            track_id=int(proposal["track_id"]),
            start_sec=float(proposal["start_sec"]),
            end_sec=float(proposal["end_sec_exclusive"]),
        )
        if feature is not None:
            samples.append(
                _sample_row(
                    sample_id=f"automatic:{proposal_id}",
                    sample_role="matched_automatic_development",
                    source_video_id=str(proposal["source_video_id"]),
                    label=str(truth["observable_pattern"]),
                    features=feature,
                )
            )

    x = np.asarray([row["features"] for row in samples], dtype=np.float64)
    y = np.asarray([row["label"] for row in samples], dtype=object)
    groups = np.asarray([row["source_video_id"] for row in samples], dtype=object)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260822)
    candidates = []
    for regularization in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        predicted = np.empty(len(y), dtype=object)
        for train, test in splitter.split(x, y, groups):
            scaler = StandardScaler().fit(x[train])
            model = LogisticRegression(
                C=regularization,
                class_weight="balanced",
                max_iter=5000,
                random_state=20260822,
            ).fit(scaler.transform(x[train]), y[train])
            predicted[test] = model.predict(scaler.transform(x[test]))
        recalls = recall_score(
            y,
            predicted,
            labels=list(CLASS_ORDER),
            average=None,
            zero_division=0,
        )
        candidates.append(
            {
                "C": regularization,
                "macro_f1": f1_score(
                    y,
                    predicted,
                    labels=list(CLASS_ORDER),
                    average="macro",
                    zero_division=0,
                ),
                "recall_by_class": {
                    name: float(value)
                    for name, value in zip(CLASS_ORDER, recalls, strict=True)
                },
                "confusion": confusion_matrix(
                    y, predicted, labels=list(CLASS_ORDER)
                ).tolist(),
            }
        )
    best = max(
        candidates,
        key=lambda row: (
            row["macro_f1"],
            row["recall_by_class"]["pacing"],
            row["recall_by_class"]["direct"],
        ),
    )
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=float(best["C"]),
        class_weight="balanced",
        max_iter=5000,
        random_state=20260822,
    ).fit(scaler.transform(x), y)
    artifact = {
        "schema_version": "wandering-camera-geometry-head-v1",
        "model_id": "home-camera-geometry-logistic-v1",
        "validation_scope": "same_participant_home_development",
        "minimum_track_confidence": 0.70,
        "bucket_seconds": 0.5,
        "feature_names": list(FEATURE_NAMES),
        "class_order": [str(value) for value in model.classes_],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": model.coef_.tolist(),
        "intercepts": model.intercept_.tolist(),
        "training_sample_count": len(samples),
        "training_roles": sorted({str(row["sample_role"]) for row in samples}),
        "behavior_truth_consumed_by_training": True,
        "behavior_truth_consumed_at_inference": False,
        "cvat_rectangle_consumed_as_tracking_truth": False,
        "cross_validation": {
            "split": "stratified_group_5fold_by_source_video",
            "best": best,
            "candidates": candidates,
        },
    }
    target_rows = []
    for proposal_id, proposal in sorted(proposals_by_id.items()):
        if str(proposal["source_video_id"]) not in {
            "wand_H01_base_2",
            "wand_P01_hall_1",
            "wand_P01_hall_2",
            "wand_P01_dining_1",
        }:
            continue
        feature = _geometry_features(
            adapters[str(proposal["source_video_id"])],
            track_id=int(proposal["track_id"]),
            start_sec=float(proposal["start_sec"]),
            end_sec=float(proposal["end_sec_exclusive"]),
        )
        if feature is None:
            continue
        probabilities = model.predict_proba(
            scaler.transform(np.asarray([feature], dtype=np.float64))
        )[0]
        target_rows.append(
            {
                "source_video_id": proposal["source_video_id"],
                "proposal_id": proposal_id,
                "start_sec": proposal["start_sec"],
                "end_sec_exclusive": proposal["end_sec_exclusive"],
                "predicted_pattern": str(model.classes_[int(np.argmax(probabilities))]),
                "probabilities": {
                    str(name): float(value)
                    for name, value in zip(model.classes_, probabilities, strict=True)
                },
            }
        )
    report = {
        "schema_version": "wandering-camera-geometry-head-development-report-v1",
        "sample_count": len(samples),
        "label_counts": {
            name: int(np.sum(y == name)) for name in CLASS_ORDER
        },
        "best_cross_validation": best,
        "target_automatic_predictions": target_rows,
    }
    output.mkdir(parents=True)
    (output / "model.json").write_bytes(canonical_json_bytes(artifact))
    (output / "report.json").write_bytes(canonical_json_bytes(report))
    print(json.dumps(report["best_cross_validation"], sort_keys=True))
    return 0


def _sample_row(
    *,
    sample_id: str,
    sample_role: str,
    source_video_id: str,
    label: str,
    features: Sequence[float],
) -> dict[str, Any]:
    if label not in CLASS_ORDER:
        raise ValueError(f"unexpected geometry label: {label}")
    return {
        "sample_id": sample_id,
        "sample_role": sample_role,
        "source_video_id": source_video_id,
        "label": label,
        "features": list(features),
    }


def _geometry_features(
    adapter: CameraAdapterInput,
    *,
    track_id: int,
    start_sec: float,
    end_sec: float,
) -> list[float] | None:
    selected = tuple(
        row
        for row in adapter.observations
        if row.track_id == track_id
        and start_sec <= row.timestamp_sec < end_sec
        and row.track_confidence >= 0.70
    )
    buckets = weighted_bucket_observations(
        selected,
        minimum_track_confidence=0.70,
        bucket_seconds=0.5,
    )
    if len(buckets) < 3:
        return None
    height = float(np.median([row.bbox_height for row in buckets]))
    if not math.isfinite(height) or height <= 0.0:
        return None
    points = np.asarray([row.bbox_bottom_point for row in buckets], dtype=np.float64)
    points = (points - points[0]) / height
    vectors = np.diff(points, axis=0)
    steps = np.linalg.norm(vectors, axis=1)
    path = float(steps.sum())
    duration = max(0.5, float(end_sec - start_sec))
    pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    extent = float(pairwise.max())
    net = float(np.linalg.norm(points[-1] - points[0]))
    centered = points - points.mean(axis=0)
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    major_vector = eigenvectors[:, int(np.argmax(eigenvalues))]
    projection_steps = np.diff(centered @ major_vector)
    active_projection = projection_steps[np.abs(projection_steps) >= 0.02]
    reversal_count = int(
        np.sum(np.sign(active_projection[1:]) != np.sign(active_projection[:-1]))
    ) if len(active_projection) >= 2 else 0
    active_vectors = vectors[steps >= 0.01]
    turns = []
    for left, right in zip(active_vectors, active_vectors[1:]):
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 0.0:
            continue
        turns.append(
            math.acos(float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0)))
        )
    turns_array = np.asarray(turns, dtype=np.float64)
    radial = np.linalg.norm(centered, axis=1)
    radial_mean = float(radial.mean())
    return [
        math.log1p(duration),
        math.log1p(len(buckets)),
        path,
        path / duration,
        extent,
        net,
        net / max(path, 1e-9),
        net / max(extent, 1e-9),
        float(eigenvalues.min() / max(eigenvalues.max(), 1e-9)),
        float(np.mean(steps <= 0.010)),
        float(np.mean(steps <= 0.015)),
        float(np.mean(steps <= 0.020)),
        float(np.mean(steps <= 0.030)),
        reversal_count * 60.0 / duration,
        reversal_count / max(1, len(active_projection) - 1),
        float(turns_array.mean()) if len(turns_array) else 0.0,
        float(np.mean(turns_array >= 2.0)) if len(turns_array) else 0.0,
        float(radial.std() / max(radial_mean, 1e-9)),
    ]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
