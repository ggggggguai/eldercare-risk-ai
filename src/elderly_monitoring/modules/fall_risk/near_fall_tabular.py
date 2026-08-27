from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from .near_fall_training import (
    NearFallDatasetConfig,
    _has_causal_context,
    _left_pad_causal_tensor,
    _resample_causal_window,
    _window_quality,
    apply_normalization,
    build_near_fall_tensor,
)
from .near_fall import extract_near_fall_events


SCHEMA_VERSION = "near-fall-tabular-checkpoint-v1"
MODEL_VERSION = "near-fall-tabular-extratrees-v1"


def extract_near_fall_tabular_features(tensor: np.ndarray) -> np.ndarray:
    values = np.asarray(tensor, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (10, 8):
        raise ValueError("near-fall tabular input must have shape [T,10,8]")
    valid = values[..., 7] > 0
    features: list[float] = []
    for channel in range(6):
        for joint in range(10):
            selected = values[:, joint, channel][valid[:, joint]]
            if len(selected) == 0:
                features.extend((0.0, 0.0, 0.0))
            else:
                features.extend(
                    (
                        float(np.mean(selected)),
                        float(np.std(selected)),
                        float(np.max(np.abs(selected))),
                    )
                )
    features.extend(float(np.mean(valid[:, joint])) for joint in range(10))
    features.extend(
        float(np.mean(values[:, joint, 6][valid[:, joint]]))
        if np.any(valid[:, joint])
        else 0.0
        for joint in range(10)
    )
    output = np.asarray(features, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("near-fall tabular features contain non-finite values")
    return output


def build_causal_near_fall_tensor(
    records: Sequence[Mapping[str, Any]],
    *,
    anchor_time_sec: float,
    config: NearFallDatasetConfig | None = None,
) -> np.ndarray | None:
    preparation = config or NearFallDatasetConfig(fallback_window_secs=(2.0,))
    if not records:
        return None
    ordered = sorted(
        (dict(row) for row in records),
        key=lambda row: (float(row.get("timestamp_sec", 0.0)), int(row.get("frame_id", -1))),
    )
    anchor = min(
        ordered,
        key=lambda row: (
            abs(float(row.get("timestamp_sec", 0.0)) - anchor_time_sec),
            int(row.get("frame_id", -1)),
        ),
    )
    anchor_time = float(anchor.get("timestamp_sec", 0.0))
    anchor_frame = int(anchor.get("frame_id", -1))
    for context_sec in preparation.context_window_secs:
        context = replace(
            preparation,
            window_sec=context_sec,
            fallback_window_secs=(),
        )
        if not _has_causal_context(
            ordered,
            anchor_time_sec=anchor_time,
            window_frames=context.window_frames,
            target_fps=context.target_fps,
        ):
            continue
        slots = _resample_causal_window(
            ordered,
            anchor_time_sec=anchor_time,
            anchor_frame=anchor_frame,
            config=context,
        )
        quality = _window_quality(slots, context.window_frames)
        if quality["observed_frame_count"] < preparation.min_observed_frames:
            continue
        if (
            quality["usable_frame_ratio"] < preparation.min_usable_frame_ratio
            or quality["joint_coverage"] < preparation.min_joint_coverage
            or quality["mean_joint_quality"] < preparation.min_mean_joint_quality
        ):
            continue
        tensor = build_near_fall_tensor(
            slots,
            window_frames=context.window_frames,
            target_fps=context.target_fps,
        )
        return _left_pad_causal_tensor(tensor, preparation.window_frames)
    return None


class NearFallTabularPredictor:
    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        checkpoint = joblib.load(self.checkpoint_path)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("near-fall tabular checkpoint must be a mapping")
        if checkpoint.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported near-fall tabular checkpoint schema")
        if checkpoint.get("model_version") != MODEL_VERSION:
            raise ValueError("near-fall tabular model version mismatch")
        contract = checkpoint.get("input_contract")
        if not isinstance(contract, Mapping):
            raise ValueError("near-fall tabular input contract is missing")
        self.model = checkpoint["model"]
        self.model_version = MODEL_VERSION
        self.threshold = float(checkpoint["validation_threshold"])
        self.window_frames = int(contract["window_frames"])
        self.feature_count = int(contract["feature_count"])
        self.normalization = {
            "mean": np.asarray(contract["normalization"]["mean"], dtype=np.float32),
            "std": np.asarray(contract["normalization"]["std"], dtype=np.float32),
        }

    def predict_tensor(self, tensor: np.ndarray) -> dict[str, Any]:
        values = np.asarray(tensor, dtype=np.float32)
        if values.shape != (self.window_frames, 10, 8):
            raise ValueError(
                f"near-fall tabular predictor expected {(self.window_frames, 10, 8)}, got {values.shape}"
            )
        if not np.any(values[..., 7] > 0):
            return {
                "status": "unavailable",
                "reason": "empty_valid_mask",
                "near_fall_event_score": None,
            }
        normalized = apply_normalization(values[None, ...], self.normalization)[0]
        features = extract_near_fall_tabular_features(normalized)
        if len(features) != self.feature_count:
            raise ValueError("near-fall tabular feature contract mismatch")
        score = float(self.model.predict_proba(features[None, :])[0, 1])
        return {
            "status": "valid",
            "near_fall_event_score": score,
            "model_version": self.model_version,
        }


def rescore_near_fall_events(
    events: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    predictor: NearFallTabularPredictor,
    fallback_to_rule: bool,
) -> list[dict[str, Any]]:
    rescored: list[dict[str, Any]] = []
    for source in events:
        event = dict(source)
        rule_score = event.get("near_fall_event_score")
        event["rule_near_fall_event_score"] = rule_score
        try:
            tensor = build_causal_near_fall_tensor(
                records,
                anchor_time_sec=float(event["end_time"]),
            )
            if tensor is None:
                raise ValueError("insufficient_causal_context_or_quality")
            prediction = predictor.predict_tensor(tensor)
            score = prediction.get("near_fall_event_score")
            if prediction.get("status") != "valid" or score is None:
                raise ValueError(
                    str(prediction.get("reason") or "tabular_prediction_unavailable")
                )
        except Exception as exc:
            reason = str(exc) or type(exc).__name__
            event["tabular_status"] = "unavailable"
            event["tabular_reason"] = reason
            if fallback_to_rule:
                event["near_fall_event_score"] = rule_score
                event["score_source"] = "rule_fallback"
                event["fallback_reason"] = reason
                event["model_version"] = predictor.model_version
            else:
                event["near_fall_event_score"] = None
            rescored.append(event)
            continue
        event["near_fall_event_score"] = float(score)
        event["tabular_status"] = "valid"
        event["tabular_reason"] = None
        event["score_source"] = "tabular_rescorer"
        event["model_version"] = predictor.model_version
        rescored.append(event)
    return rescored


class NearFallTabularRuntimePredictor:
    """Rescore causal rule candidates while retaining a rule safety fallback."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        self.predictor = NearFallTabularPredictor(checkpoint_path)
        self.model_version = self.predictor.model_version

    def predict_records(
        self, records: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        source = [dict(record) for record in records]
        return rescore_near_fall_events(
            extract_near_fall_events(source),
            source,
            predictor=self.predictor,
            fallback_to_rule=True,
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
