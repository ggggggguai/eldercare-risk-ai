from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from elderly_monitoring.modules.fall_risk.sit_stand_continuous import (
    SitStandContinuousConfig,
)
from elderly_monitoring.modules.fall_risk.sit_stand_continuous_inference import (
    load_continuous_sit_stand_tcn,
    predict_rule_sit_stand_video,
    predict_tcn_sit_stand_video,
)
from elderly_monitoring.modules.fall_risk.sit_stand_continuous_tcn import (
    SitStandStreamDecoderConfig,
)
from elderly_monitoring.modules.fall_risk.sit_stand_event_evaluation import (
    evaluate_sit_stand_events,
    write_sit_stand_evaluation_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run rule or TCN sit-stand detection on full development pose streams."
    )
    parser.add_argument("--mode", choices=("rule", "tcn"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/annotations/fall_risk/sit_stand_event_labels_v1.jsonl"),
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path(
            "data/splits/fall_risk/sit_stand_event_v1_pose_balanced/assignments.jsonl"
        ),
    )
    parser.add_argument(
        "--pose-dir",
        type=Path,
        default=Path("data/processed/fall_risk/pose_quality_y8n_v1/cleaned"),
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/training/sit_stand_event_v1.yaml"),
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=Path("configs/evaluation/sit_stand_event_v1.provisional.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output_dir.exists():
            raise FileExistsError(f"streaming evaluation output exists: {args.output_dir}")
        if args.mode == "tcn" and args.checkpoint is None:
            raise ValueError("--checkpoint is required for TCN evaluation")
        if args.mode == "rule" and args.checkpoint is not None:
            raise ValueError("--checkpoint is not valid for rule evaluation")
        assignments = _read_jsonl(args.assignments)
        if any(row.get("partition") == "test" for row in assignments):
            raise ValueError("streaming development evaluation must not receive test assignments")
        validation_ids = {
            str(row["label_id"])
            for row in assignments
            if row.get("partition") == "validation"
        }
        labels = [
            row for row in _read_jsonl(args.labels) if str(row.get("label_id")) in validation_ids
        ]
        if not labels:
            raise ValueError("validation ground truth is empty")
        video_ids = sorted({str(row["video_id"]) for row in labels})
        evaluation_config = _read_yaml(args.evaluation_config)
        training_config = _read_yaml(args.training_config)
        continuous_values = dict(training_config["continuous_dataset"])
        for key in (
            "window_frames",
            "partial_context_policy",
            "multi_person_policy",
            "joints",
            "base_channels",
            "timing_channels",
            "negative_policy",
            "test_pose_allowed",
        ):
            continuous_values.pop(key, None)
        continuous_config = SitStandContinuousConfig(**continuous_values)
        decoder_config = SitStandStreamDecoderConfig(
            presence_threshold=float(evaluation_config["score_threshold"]),
            state_threshold=float(evaluation_config["score_threshold"]),
            boundary_threshold=float(evaluation_config["score_threshold"]),
            confirmation_frames=int(evaluation_config["confirmation_frames"]),
            event_merge_gap_sec=float(evaluation_config["event_merge_gap_sec"]),
            max_sequence_gap_sec=continuous_config.max_gap_sec,
            minimum_duration_sec=1.0 / continuous_config.target_fps,
        )
        model = selected_device = checkpoint_sha256 = None
        if args.mode == "tcn":
            model, selected_device, checkpoint_sha256 = load_continuous_sit_stand_tcn(
                args.checkpoint, device=args.device
            )

        predictions: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for video_id in video_ids:
            pose_path = args.pose_dir / f"{video_id}.jsonl"
            if not pose_path.is_file():
                raise FileNotFoundError(f"validation pose is missing: {pose_path}")
            records = _read_jsonl(pose_path)
            if args.mode == "rule":
                result = predict_rule_sit_stand_video(records, video_id=video_id)
            else:
                result = predict_tcn_sit_stand_video(
                    records,
                    video_id=video_id,
                    model=model,
                    device=selected_device,
                    checkpoint_sha256=str(checkpoint_sha256),
                    continuous_config=continuous_config,
                    decoder_config=decoder_config,
                    batch_size=args.batch_size,
                )
            predictions.extend(result["predictions"])
            diagnostics.append(result["diagnostics"])

        evaluation = evaluate_sit_stand_events(
            labels, predictions, config=evaluation_config
        )
        args.output_dir.mkdir(parents=True)
        _write_jsonl(args.output_dir / "predictions.jsonl", predictions)
        _write_jsonl(args.output_dir / "stream_diagnostics.jsonl", diagnostics)
        _write_json(
            args.output_dir / "run.json",
            {
                "schema_version": "sit-stand-streaming-validation-run-v1",
                "status": "development_provisional",
                "mode": args.mode,
                "validation_video_count": len(video_ids),
                "prediction_count": len(predictions),
                "checkpoint_sha256": checkpoint_sha256,
                "test_access": {
                    "test_pose_read": False,
                    "test_features_generated": False,
                    "test_evaluated": False,
                },
            },
        )
        write_sit_stand_evaluation_bundle(evaluation, args.output_dir / "evaluation")
    except (KeyError, OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(evaluation["metrics"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSON object required at {path}:{line_number}")
        rows.append(row)
    return rows


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML mapping required: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
