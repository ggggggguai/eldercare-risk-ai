from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import yaml

from elderly_monitoring.modules.fall_risk.baseline_longitudinal import (
    build_longitudinal_split,
)


def _period(person_index: int, day: int, *, labelled: bool) -> dict[str, Any]:
    person_id = f"synthetic-person-{person_index:03d}"
    start = datetime(2026, 6, 1, tzinfo=timezone(timedelta(hours=8))) + timedelta(days=day - 1)
    end = start + timedelta(days=1) - timedelta(seconds=1)
    return {
        "observation_id": f"synthetic-observation-{person_index:03d}-{day:02d}",
        "record_type": "fall_baseline_period_features",
        "schema_version": "fall-baseline-period-features-v1",
        "asset_id": f"synthetic-longitudinal-asset-{person_index:03d}",
        "person_id": person_id,
        "device_id": "synthetic-device-1",
        "camera_profile_id": "synthetic-camera-1",
        "source_group_id": f"synthetic-home-{person_index:03d}",
        "period_id": f"synthetic-day-{day:02d}",
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "timezone": "Asia/Shanghai",
        "completed": True,
        "aggregation_version": "synthetic-period-aggregation-v1",
        "upstream_versions": {"gait": "synthetic-gait-v1"},
        "input_summary": {"source_record_count": 100},
        "valid_monitoring_hours": 8.0,
        "quality_state": "valid",
        "no_personal_baseline_score": 0.45,
        "no_personal_baseline_available_weight": 0.68,
        "fusion_config_version": "fall-risk-fusion-synthetic-v1",
        "metric_quality": {
            "mean_gait_speed": {
                "available": True,
                "observation_count": 10,
                "quality": 0.9,
                "coverage": 0.9,
                "exposure_hours": 8.0,
                "missing_reason": None,
            }
        },
        "gait_stability_features": {
            "mean_center_speed_norm_per_sec": (
                0.20 if day == 13 and person_index % 2 == 0 else 0.42
            )
        },
        "outcome_label_id": (
            f"synthetic-risk-{person_index:03d}-{day:02d}" if labelled else None
        ),
    }


def _label_and_reviews(
    observation: Mapping[str, Any], person_index: int, day: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    positive = day == 13 and person_index % 2 == 0
    label_id = str(observation["outcome_label_id"])
    label = {
        "label_id": label_id,
        "asset_id": observation["asset_id"],
        "task_type": "longitudinal_baseline",
        "subject_id": observation["person_id"],
        "start_time": datetime.fromisoformat(str(observation["period_end"])).timestamp() + 3600.0,
        "end_time": datetime.fromisoformat(str(observation["period_end"])).timestamp() + 7200.0,
        "risk_level": 3 if positive else 0,
        "risk_factors": ["synthetic_state_change"] if positive else [],
        "label_source": "manual_consensus",
    }
    reviews = [
        {
            "review_id": f"synthetic-review-{person_index:03d}-{day:02d}-{reviewer}",
            "label_id": label_id,
            "reviewer_id": f"synthetic-reviewer-{reviewer}",
            "decision": "accepted",
            "reviewed_at": "2026-08-12T08:00:00+08:00",
            "evidence_reference": f"synthetic-evidence-{person_index:03d}-{day:02d}",
        }
        for reviewer in (1, 2)
    ]
    return label, reviews


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def build_fixture(output_dir: Path, config_path: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("evaluation config must be a YAML mapping")

    observations: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    profiles = {"schema_version": "fall-risk-subject-profiles-v2", "subjects": []}
    for person_index in range(1, 61):
        person_id = f"synthetic-person-{person_index:03d}"
        profiles["subjects"].append(
            {
                "subject_id": person_id,
                "profile_version": "synthetic-profile-v1",
                "profile_source": "synthetic_fixture",
                "consent_id": f"synthetic-consent-{person_index:03d}",
                "features": {
                    "source_group_id": f"synthetic-home-{person_index:03d}",
                    "authorized_device_ids": ["synthetic-device-1"],
                    "authorized_camera_profile_ids": ["synthetic-camera-1"],
                },
            }
        )
        for day in range(1, 14):
            observation = _period(person_index, day, labelled=day > 10)
            observations.append(observation)
            if day > 10:
                label, label_reviews = _label_and_reviews(observation, person_index, day)
                labels.append(label)
                reviews.extend(label_reviews)

    artifact = build_longitudinal_split(
        observations, labels, profiles, reviews, protocol
    )
    if artifact["metadata"]["status"] == "blocked":
        raise ValueError(f"synthetic fixture is blocked: {artifact['metadata']['blockers']}")
    validation_observation_ids = {
        row["observation_id"]
        for row in artifact["assignments"]
        if row["partition"] == "validation"
    }
    validation_scoring_ids = {
        row["observation_id"]
        for row in artifact["assignments"]
        if row["partition"] == "validation" and row["role"] == "scoring"
    }
    validation_observations = [
        row for row in observations if row["observation_id"] in validation_observation_ids
    ]
    validation_label_ids = {
        str(row["outcome_label_id"])
        for row in validation_observations
        if row["observation_id"] in validation_scoring_ids
    }
    validation_labels = [
        row for row in labels if row["label_id"] in validation_label_ids
    ]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent))
    try:
        (staging / "observations.jsonl").write_text(_jsonl(observations), encoding="utf-8")
        (staging / "risk_labels.jsonl").write_text(_jsonl(labels), encoding="utf-8")
        (staging / "subject_profiles.json").write_text(
            json.dumps(profiles, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "review_log.jsonl").write_text(_jsonl(reviews), encoding="utf-8")
        (staging / "assignments.jsonl").write_text(
            _jsonl(artifact["assignments"]), encoding="utf-8"
        )
        (staging / "split.json").write_text(
            json.dumps(artifact["metadata"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "observations.validation.jsonl").write_text(
            _jsonl(validation_observations), encoding="utf-8"
        )
        (staging / "risk_labels.validation.jsonl").write_text(
            _jsonl(validation_labels), encoding="utf-8"
        )
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "split_id": artifact["metadata"]["split_id"],
        "subjects": artifact["metadata"]["eligible_subject_count"],
        "scoring_periods": artifact["metadata"]["scoring_period_count"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic synthetic longitudinal-baseline evaluation fixture."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/fall_baseline_longitudinal_v1.provisional.yaml"),
    )
    args = parser.parse_args(argv)
    try:
        result = build_fixture(args.output_dir, args.config)
    except (FileExistsError, FileNotFoundError, OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
