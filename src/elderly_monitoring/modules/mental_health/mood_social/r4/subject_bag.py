"""Postlocked causal DeepSets and limited pairwise-loss diagnostic."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import ap_context_metrics
from elderly_monitoring.modules.mental_health.mood_social.r4.data import (
    load_r4_development_frame,
    select_label_task,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    participant_equal_weights,
)


DEFAULT_OUTPUT = Path(
    "reports/mental_health/mood_social/v3.3.3-r4/"
    "OPT-V333-R4-005-postlocked-deepsets"
)
BAG_FEATURES = (
    "source__steps_awake_mean",
    "source__steps_mvpa_sum_recent",
    "source__steps_lpa_sum_recent",
    "source__sleep_asleep_mean_recent",
    "source__sleep_in_bed_mean_recent",
    "source__sleep_ratio_asleep_in_bed_mean_recent",
)


class CausalDeepSets(nn.Module):
    def __init__(self, feature_count: int, hidden: int = 16) -> None:
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(feature_count, hidden), nn.ReLU())
        self.rho = nn.Sequential(
            nn.Linear(hidden + feature_count, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(
        self, sequence: torch.Tensor, mask: torch.Tensor, current: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.phi(sequence)
        weight = mask.unsqueeze(-1)
        pooled = (encoded * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        return self.rho(torch.cat([pooled, current], dim=1)).squeeze(1)


class CausalOrderedDeepSets(nn.Module):
    """Shared encoder with a conditional elevated-risk probability."""

    def __init__(self, feature_count: int, hidden: int = 16) -> None:
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(feature_count, hidden), nn.ReLU())
        self.shared = nn.Sequential(nn.Linear(hidden + feature_count, hidden), nn.ReLU())
        self.early_head = nn.Linear(hidden, 1)
        self.conditional_elevated_head = nn.Linear(hidden, 1)

    def forward(
        self, sequence: torch.Tensor, mask: torch.Tensor, current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.phi(sequence)
        weight = mask.unsqueeze(-1)
        pooled = (encoded * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        shared = self.shared(torch.cat([pooled, current], dim=1))
        early = torch.sigmoid(self.early_head(shared).squeeze(1))
        conditional = torch.sigmoid(self.conditional_elevated_head(shared).squeeze(1))
        return early, early * conditional


@dataclass(frozen=True)
class BagTensors:
    sequence: torch.Tensor
    mask: torch.Tensor
    current: torch.Tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def fit_scaler(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = frame[list(BAG_FEATURES)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    median = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    scale = np.where(q75 - q25 > 1.0e-8, q75 - q25, 1.0)
    return median, scale


def build_causal_bag_tensors(
    frame: pd.DataFrame, median: np.ndarray, scale: np.ndarray
) -> BagTensors:
    if frame.duplicated(["global_participant_id", "nominal_month"]).any():
        raise ValueError("DeepSets input contains duplicate participant/month rows")
    maximum = int(frame.groupby("global_participant_id").size().max())
    sequence = np.zeros((len(frame), maximum, len(BAG_FEATURES)), dtype=np.float32)
    mask = np.zeros((len(frame), maximum), dtype=np.float32)
    current = np.zeros((len(frame), len(BAG_FEATURES)), dtype=np.float32)
    position = {index: offset for offset, index in enumerate(frame.index)}
    for _person, part in frame.groupby("global_participant_id", sort=False):
        ordered = part.sort_values("nominal_month", kind="stable")
        values = ordered[list(BAG_FEATURES)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        values = np.where(np.isfinite(values), values, median)
        values = ((values - median) / scale).astype(np.float32)
        for order_index, row_index in enumerate(ordered.index):
            row = position[row_index]
            prefix = values[: order_index + 1]
            sequence[row, : len(prefix)] = prefix
            mask[row, : len(prefix)] = 1.0
            current[row] = values[order_index]
    return BagTensors(
        torch.from_numpy(sequence), torch.from_numpy(mask), torch.from_numpy(current)
    )


def _ranking_pairs(frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    position = {index: offset for offset, index in enumerate(frame.index)}
    positive: list[int] = []
    negative: list[int] = []
    for _person, part in frame.groupby("global_participant_id", sort=False):
        positives = [position[index] for index in part.index[part["binary_target"].eq(1)]]
        negatives = [position[index] for index in part.index[part["binary_target"].eq(0)]]
        for pos in positives:
            for neg in negatives:
                positive.append(pos)
                negative.append(neg)
    return torch.tensor(positive, dtype=torch.long), torch.tensor(negative, dtype=torch.long)


def fit_deepsets(
    frame: pd.DataFrame,
    tensors: BagTensors,
    *,
    pairwise_weight: float,
    seed: int,
    epochs: int = 40,
) -> CausalDeepSets:
    torch.manual_seed(seed)
    model = CausalDeepSets(len(BAG_FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1.0e-3)
    target = torch.tensor(frame["binary_target"].to_numpy(float), dtype=torch.float32)
    weight = torch.tensor(participant_equal_weights(frame), dtype=torch.float32)
    positive, negative = _ranking_pairs(frame)
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logit = model(tensors.sequence, tensors.mask, tensors.current)
        pointwise = F.binary_cross_entropy_with_logits(logit, target, weight=weight)
        pairwise = (
            F.softplus(-(logit[positive] - logit[negative])).mean()
            if pairwise_weight > 0 and len(positive) > 0
            else torch.zeros((), dtype=logit.dtype)
        )
        loss = pointwise + pairwise_weight * pairwise
        loss.backward()
        optimizer.step()
    return model.eval()


def predict_deepsets(model: CausalDeepSets, tensors: BagTensors) -> np.ndarray:
    with torch.no_grad():
        probability = torch.sigmoid(
            model(tensors.sequence, tensors.mask, tensors.current)
        ).numpy()
    return np.clip(probability, 1.0e-6, 1.0 - 1.0e-6)


def fit_ordered_deepsets(
    frame: pd.DataFrame,
    tensors: BagTensors,
    *,
    seed: int,
    epochs: int = 40,
) -> CausalOrderedDeepSets:
    torch.manual_seed(seed)
    model = CausalOrderedDeepSets(len(BAG_FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1.0e-3)
    early_target = torch.tensor(frame["phq9_ge5_r3_target"].to_numpy(float), dtype=torch.float32)
    elevated_target = torch.tensor(frame["phq9_ge10_r3_target"].to_numpy(float), dtype=torch.float32)
    weight = torch.tensor(participant_equal_weights(frame), dtype=torch.float32)
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        early, elevated = model(tensors.sequence, tensors.mask, tensors.current)
        loss = F.binary_cross_entropy(early, early_target, weight=weight) + F.binary_cross_entropy(
            elevated, elevated_target, weight=weight
        )
        loss.backward()
        optimizer.step()
    return model.eval()


def _ordered_outer_oof(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    early_prediction = pd.Series(np.nan, index=frame.index, dtype=float)
    elevated_prediction = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold in sorted(frame["outer_fold"].astype(int).unique()):
        train = frame.loc[frame["outer_fold"].ne(fold)].copy()
        test = frame.loc[frame["outer_fold"].eq(fold)].copy()
        median, scale = fit_scaler(train)
        train_tensor = build_causal_bag_tensors(train, median, scale)
        test_tensor = build_causal_bag_tensors(test, median, scale)
        model = fit_ordered_deepsets(
            train, train_tensor, seed=20260832 + fold
        )
        with torch.no_grad():
            early, elevated = model(
                test_tensor.sequence, test_tensor.mask, test_tensor.current
            )
        early_prediction.loc[test.index] = early.numpy()
        elevated_prediction.loc[test.index] = elevated.numpy()
    if early_prediction.isna().any() or elevated_prediction.isna().any():
        raise ValueError("ordered DeepSets OOF is incomplete")
    early = np.clip(early_prediction.loc[frame.index].to_numpy(float), 1.0e-6, 1.0 - 1.0e-6)
    elevated = np.clip(
        elevated_prediction.loc[frame.index].to_numpy(float), 1.0e-6, 1.0 - 1.0e-6
    )
    if np.any(elevated > early + 1.0e-12):
        raise ValueError("ordered DeepSets violated probability monotonicity")
    return early, elevated


def _outer_oof(frame: pd.DataFrame, pairwise_weight: float) -> np.ndarray:
    prediction = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold in sorted(frame["outer_fold"].astype(int).unique()):
        train = frame.loc[frame["outer_fold"].ne(fold)].copy()
        test = frame.loc[frame["outer_fold"].eq(fold)].copy()
        median, scale = fit_scaler(train)
        train_tensor = build_causal_bag_tensors(train, median, scale)
        test_tensor = build_causal_bag_tensors(test, median, scale)
        model = fit_deepsets(
            train,
            train_tensor,
            pairwise_weight=pairwise_weight,
            seed=20260812 + fold,
        )
        prediction.loc[test.index] = predict_deepsets(model, test_tensor)
    if prediction.isna().any():
        raise ValueError("DeepSets OOF is incomplete")
    return prediction.loc[frame.index].to_numpy(float)


def run_subject_bag_diagnostic(repository_root: Path, output: Path | None = None) -> dict[str, Any]:
    root = repository_root.resolve()
    destination = (output or root / DEFAULT_OUTPUT).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite DeepSets diagnostic: {destination}")
    base = load_r4_development_frame(repository_root=root)
    psyche = base.loc[base["dataset_id"].eq("psyche_d")].copy()
    reports: dict[str, Any] = {}
    predictions: list[pd.DataFrame] = []
    for task in ("phq_ge5_current", "phq_ge10_current"):
        frame = select_label_task(psyche, task)
        for name, pairwise_weight in (("pointwise", 0.0), ("pointwise_plus_pairwise", 0.1)):
            probability = _outer_oof(frame, pairwise_weight)
            reports[f"{task}:{name}"] = {
                "metrics": ap_context_metrics(frame["binary_target"], probability),
                "participant_equal": ap_context_metrics(
                    frame["binary_target"], probability,
                    sample_weight=participant_equal_weights(frame),
                ),
                "pairwise_weight": pairwise_weight,
                "ranking_pairs": int(len(_ranking_pairs(frame)[0])),
            }
            part = frame[
                ["r4_row_id", "global_participant_id", "nominal_month", "outer_fold", "binary_target"]
            ].copy()
            part["r4_task_id"] = task
            part["model"] = name
            part["probability"] = probability
            predictions.append(part)
    shared_early, shared_elevated = _ordered_outer_oof(psyche)
    for task, target_column, probability in (
        ("phq_ge5_current", "phq9_ge5_r3_target", shared_early),
        ("phq_ge10_current", "phq9_ge10_r3_target", shared_elevated),
    ):
        frame = select_label_task(psyche, task)
        reports[f"{task}:shared_ordered"] = {
            "metrics": ap_context_metrics(frame["binary_target"], probability),
            "participant_equal": ap_context_metrics(
                frame["binary_target"], probability,
                sample_weight=participant_equal_weights(frame),
            ),
            "shared_encoder": True,
            "monotonic_by_construction": True,
        }
        part = frame[
            ["r4_row_id", "global_participant_id", "nominal_month", "outer_fold", "binary_target"]
        ].copy()
        part["r4_task_id"] = task
        part["model"] = "shared_ordered"
        part["probability"] = probability
        predictions.append(part)
    prediction_path = destination / "deepsets_pairwise_oof.parquet"
    report_path = destination / "deepsets_pairwise_metrics.json"
    pd.concat(predictions, ignore_index=True).to_parquet(prediction_path, index=False)
    _dump(
        report_path,
        {
            "protocol_version": "mood-social-v3.3.3-r4",
            "evidence_level": "postlocked_diagnostic_only_not_model_selection",
            "architecture": "causal_deepsets_phi16_mean_pool_current_rho16_and_shared_ordered_heads",
            "features": BAG_FEATURES,
            "epochs": 40,
            "future_rows_used": False,
            "formal_locked_candidate_changed": False,
            "results": reports,
        },
    )
    manifest = {
        "status": "pass",
        "prediction_path": prediction_path.relative_to(root).as_posix(),
        "prediction_sha256": _sha256(prediction_path),
        "report_path": report_path.relative_to(root).as_posix(),
        "report_sha256": _sha256(report_path),
    }
    manifest_path = destination / "artifact_manifest.json"
    _dump(manifest_path, manifest)
    return manifest


__all__ = [
    "BAG_FEATURES",
    "CausalDeepSets",
    "CausalOrderedDeepSets",
    "build_causal_bag_tensors",
    "run_subject_bag_diagnostic",
]
