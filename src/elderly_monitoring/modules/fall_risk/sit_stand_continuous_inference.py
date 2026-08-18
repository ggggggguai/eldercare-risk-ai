from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from elderly_monitoring.modules.fall_risk.sit_stand import (
    MODEL_VERSION as RULE_MODEL_VERSION,
    extract_sit_stand_events,
)
from elderly_monitoring.modules.fall_risk.sit_stand_continuous import (
    SitStandContinuousConfig,
    build_sit_stand_causal_window,
)
from elderly_monitoring.modules.fall_risk.sit_stand_continuous_tcn import (
    TASK,
    ContinuousSitStandTCN,
    SitStandStreamDecoderConfig,
    decode_sit_stand_stream,
)


class ExperimentalSitStandTCNPredictor:
    """Adapt the provisional causal TCN to the runtime sit-stand contract.

    This predictor is opt-in. When the candidate emits no event or inference
    fails, the established rule extractor remains the safety fallback.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        batch_size: int = 128,
    ) -> None:
        if batch_size < 1:
            raise ValueError("sit-stand TCN batch_size must be positive")
        self.model, self.device, self.checkpoint_sha256 = load_continuous_sit_stand_tcn(
            checkpoint_path,
            device=device,
        )
        self.batch_size = batch_size
        self.model_version = f"sit-stand-continuous-tcn-v1:{self.checkpoint_sha256[:12]}"

    def predict_records(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        source = [dict(record) for record in records]
        try:
            predictions = predict_tcn_sit_stand_video(
                source,
                video_id="runtime",
                model=self.model,
                device=self.device,
                checkpoint_sha256=self.checkpoint_sha256,
                batch_size=self.batch_size,
                latest_only=True,
            )["predictions"]
        except (OSError, RuntimeError, TypeError, ValueError):
            predictions = []
        if predictions:
            return [
                {
                    "person_id": str(prediction["stream_id"]).split(":", 1)[0],
                    "track_id": str(prediction["stream_id"]).split(":", 1)[1],
                    "start_time": float(prediction["onset_time"]),
                    "end_time": float(prediction["offset_time"]),
                    "transition_type": prediction["transition_type"],
                    "sit_stand_risk_score": float(prediction["score"]),
                    "score_source": "tcn",
                    "model_version": prediction["model_version"],
                    "risk_factors": ["experimental_sit_stand_tcn_event"],
                }
                for prediction in predictions
            ]

        fallback = extract_sit_stand_events(source)
        for event in fallback:
            event["score_source"] = "rule_fallback"
            event["fallback_reason"] = "tcn_no_event_or_inference_failure"
            event["model_version"] = self.model_version
        return fallback


def load_continuous_sit_stand_tcn(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[ContinuousSitStandTCN, torch.device, str]:
    source = Path(checkpoint_path)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if checkpoint.get("task") != TASK:
        raise ValueError("checkpoint task is not continuous sit-stand localization")
    if checkpoint.get("status") != "development_provisional":
        raise ValueError("continuous sit-stand checkpoint status is invalid")
    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(model_config, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("continuous sit-stand checkpoint is incomplete")
    selected_device = _device(device)
    model = ContinuousSitStandTCN(**dict(model_config))
    model.load_state_dict(state_dict)
    model.to(selected_device).eval()
    return model, selected_device, _sha256(source)


def predict_tcn_sit_stand_video(
    records: Iterable[Mapping[str, Any]],
    *,
    video_id: str,
    model: ContinuousSitStandTCN,
    device: torch.device,
    checkpoint_sha256: str,
    continuous_config: SitStandContinuousConfig | None = None,
    decoder_config: SitStandStreamDecoderConfig | None = None,
    batch_size: int = 128,
    latest_only: bool = False,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    window_config = continuous_config or SitStandContinuousConfig()
    decoder = decoder_config or SitStandStreamDecoderConfig()
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for value in records:
        row = dict(value)
        person_id = str(row.get("person_id", "unknown"))
        track_id = "none" if row.get("track_id") is None else str(row.get("track_id"))
        groups[(person_id, track_id)].append(row)

    predictions: list[dict[str, Any]] = []
    diagnostics = {
        "video_id": video_id,
        "stream_count": len(groups),
        "cutoff_count": 0,
        "valid_window_count": 0,
        "unavailable_window_count": 0,
        "invalid_window_count": 0,
        "decoded_event_count": 0,
    }
    model_version = f"sit-stand-continuous-tcn-v1:{checkpoint_sha256[:12]}"
    for identity, group_rows in sorted(groups.items()):
        ordered = sorted(
            group_rows,
            key=lambda row: (
                float(row.get("timestamp_sec", float("inf"))),
                int(row.get("frame_id", 0) or 0),
            ),
        )
        all_cutoffs = sorted(
            {
                float(row["timestamp_sec"])
                for row in ordered
                if isinstance(row.get("timestamp_sec"), (int, float))
                and not isinstance(row.get("timestamp_sec"), bool)
                and np.isfinite(float(row["timestamp_sec"]))
                and float(row["timestamp_sec"]) >= 0
            }
        )
        cutoffs = all_cutoffs[-1:] if latest_only else all_cutoffs
        diagnostics["cutoff_count"] += len(cutoffs)
        tensors: list[np.ndarray] = []
        valid_cutoffs: list[float] = []
        for cutoff in cutoffs:
            try:
                window = build_sit_stand_causal_window(
                    ordered,
                    cutoff_time_sec=cutoff,
                    config=window_config,
                    required_observed_frames=window_config.min_partial_observed_frames,
                )
            except ValueError:
                diagnostics["invalid_window_count"] += 1
                continue
            if window.status != "valid":
                diagnostics["unavailable_window_count"] += 1
                continue
            tensors.append(window.tensor)
            valid_cutoffs.append(cutoff)

        diagnostics["valid_window_count"] += len(tensors)
        probability_rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for start in range(0, len(tensors), batch_size):
                features = torch.from_numpy(np.stack(tensors[start : start + batch_size])).to(
                    device
                )
                outputs = model(features)
                frame = torch.softmax(outputs["frame_logits"][:, -1], dim=1).cpu().numpy()
                boundary = torch.sigmoid(outputs["boundary_logits"][:, -1]).cpu().numpy()
                presence = torch.softmax(outputs["presence_logits"], dim=1)[:, 1].cpu().numpy()
                direction = torch.softmax(outputs["direction_logits"], dim=1).cpu().numpy()
                for local_index in range(len(features)):
                    probability_rows.append(
                        {
                            "timestamp_sec": valid_cutoffs[start + local_index],
                            "presence_score": float(presence[local_index]),
                            "frame_probabilities": frame[local_index].tolist(),
                            "boundary_probabilities": boundary[local_index].tolist(),
                            "direction_probabilities": direction[local_index].tolist(),
                        }
                    )
        stream_id = f"{identity[0]}:{identity[1]}"
        predictions.extend(
            decode_sit_stand_stream(
                probability_rows,
                video_id=video_id,
                stream_id=stream_id,
                config=decoder,
                model_version=model_version,
            )
        )
    diagnostics["decoded_event_count"] = len(predictions)
    return {
        "predictions": sorted(
            predictions,
            key=lambda row: (
                float(row["onset_time"]),
                str(row["stream_id"]),
                str(row["prediction_id"]),
            ),
        ),
        "diagnostics": diagnostics,
        "test_access": {"test_pose_read": False, "test_evaluated": False},
    }


def predict_rule_sit_stand_video(
    records: Iterable[Mapping[str, Any]], *, video_id: str
) -> dict[str, Any]:
    source = [dict(row) for row in records]
    predictions: list[dict[str, Any]] = []
    rejected = 0
    for event in extract_sit_stand_events(source):
        transition = event.get("transition_type")
        if transition not in {"sit_to_stand", "stand_to_sit"}:
            rejected += 1
            continue
        if event.get("quality_coverage", {}).get("insufficient_sit_stand_quality"):
            rejected += 1
            continue
        onset = float(event["start_time"])
        offset = float(event["end_time"])
        if offset <= onset:
            rejected += 1
            continue
        stream_id = f"{event.get('person_id', 'unknown')}:{event.get('track_id', 'none')}"
        identity = f"{video_id}|{stream_id}|{transition}|{onset:.6f}|{offset:.6f}"
        predictions.append(
            {
                "prediction_id": "sitstandrule_"
                + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
                "video_id": video_id,
                "stream_id": stream_id,
                "transition_type": transition,
                "onset_time": round(onset, 6),
                "offset_time": round(offset, 6),
                "score": 1.0,
                "quality_state": "valid",
                "model_version": RULE_MODEL_VERSION,
            }
        )
    return {
        "predictions": predictions,
        "diagnostics": {
            "video_id": video_id,
            "input_record_count": len(source),
            "decoded_event_count": len(predictions),
            "rejected_event_count": rejected,
        },
        "test_access": {"test_pose_read": False, "test_evaluated": False},
    }


def _device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be cpu, cuda, mps or auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    return torch.device(requested)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
