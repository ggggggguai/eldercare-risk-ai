from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

from elderly_monitoring.modules.fall_risk.baseline import (
    build_personal_baselines,
    score_baseline_deviation,
)


DISCLOSURE = "仅用于算法机制演示，不代表真人历史采集。"
PERSON_ID = "live-demo-person"
DEVICE_ID = "live-demo-camera"
CAMERA_PROFILE_ID = "synthetic-demo-camera-profile"
METRICS = (
    "mean_gait_speed",
    "mean_sit_stand_duration",
    "near_fall_rate_per_hour",
    "nighttime_activity_rate_per_hour",
    "activity_volume",
)


def _period(
    day: int,
    *,
    gait_speed: float,
    sit_stand_duration: float,
    near_fall_count: int,
    nighttime_activity_count: int,
    activity_volume: float,
) -> dict[str, Any]:
    start = datetime(2026, 8, 17, tzinfo=timezone(timedelta(hours=8))) + timedelta(
        days=day
    )
    return {
        "record_type": "fall_baseline_period_features",
        "schema_version": "fall-baseline-period-features-v1",
        "person_id": PERSON_ID,
        "device_id": DEVICE_ID,
        "camera_profile_id": CAMERA_PROFILE_ID,
        "period_id": f"synthetic-normal-202608{17 + day:02d}",
        "period_start": start.isoformat(),
        "period_end": (start + timedelta(days=1) - timedelta(seconds=1)).isoformat(),
        "timezone": "Asia/Shanghai",
        "completed": True,
        "aggregation_version": "fall-baseline-synthetic-demo-v1",
        "upstream_versions": {"source": "deterministic-synthetic-normal-v1"},
        "input_summary": {"source_record_count": 2, "source_type": "synthetic"},
        "record_count": 2,
        "valid_monitoring_hours": 2.0,
        "quality_state": "synthetic_demo",
        "gait_stability_features": {
            "mean_center_speed_norm_per_sec": gait_speed
        },
        "mean_sit_stand_duration": sit_stand_duration,
        "near_fall_event_count": near_fall_count,
        "nighttime_activity_count": nighttime_activity_count,
        "activity_volume": activity_volume,
        "dominant_scene_region": "home",
        "scene_region_distribution": {"home": 1.0},
        "metric_quality": {
            metric: {
                "available": True,
                "observation_count": 2,
                "quality": 0.92,
                "coverage": 0.9,
                "exposure_hours": 2.0,
                "missing_reason": None,
            }
            for metric in METRICS
        },
        "data_provenance": {
            "data_type": "synthetic",
            "display_label": "合成正常基线",
            "disclosure": DISCLOSURE,
            "generation_method": "deterministic_fixture",
        },
    }


def build_records() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    normal_values = (
        (0.42, 2.35, 0, 0, 100),
        (0.43, 2.30, 0, 0, 103),
        (0.41, 2.40, 0, 1, 98),
        (0.42, 2.32, 0, 0, 101),
        (0.44, 2.28, 0, 0, 105),
        (0.41, 2.38, 1, 0, 99),
        (0.43, 2.31, 0, 1, 102),
        (0.42, 2.36, 0, 0, 100),
        (0.41, 2.29, 0, 0, 97),
        (0.43, 2.34, 0, 0, 104),
    )
    history = [
        _period(
            index,
            gait_speed=values[0],
            sit_stand_duration=values[1],
            near_fall_count=values[2],
            nighttime_activity_count=values[3],
            activity_volume=values[4],
        )
        for index, values in enumerate(normal_values, start=1)
    ]
    normal_current = _period(
        11,
        gait_speed=0.42,
        sit_stand_duration=2.33,
        near_fall_count=0,
        nighttime_activity_count=0,
        activity_volume=101,
    )
    normal_current["period_id"] = "synthetic-current-normal-20260828"
    deviation_current = _period(
        12,
        gait_speed=0.25,
        sit_stand_duration=4.1,
        near_fall_count=3,
        nighttime_activity_count=2,
        activity_volume=52,
    )
    deviation_current["period_id"] = "synthetic-current-deviation-20260829"
    return history, normal_current, deviation_current


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def write_fixture(output_dir: Path) -> dict[str, Any]:
    history, normal_current, deviation_current = build_records()
    baselines = build_personal_baselines(history)
    [normal_score] = score_baseline_deviation([normal_current], baselines)
    [deviation_score] = score_baseline_deviation([deviation_current], baselines)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "synthetic-normal-history.jsonl").write_text(
        "".join(_json_line(row) for row in history), encoding="utf-8"
    )
    (output_dir / "synthetic-current-normal.json").write_text(
        json.dumps(normal_current, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "synthetic-current-normal-request.json").write_text(
        json.dumps(
            {"period": normal_current}, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "synthetic-current-deviation.json").write_text(
        json.dumps(deviation_current, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "synthetic-current-deviation-request.json").write_text(
        json.dumps(
            {"period": deviation_current},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "data_type": "synthetic",
        "display_label": "合成正常基线",
        "disclosure": DISCLOSURE,
        "history_period_count": len(history),
        "live_default_period_id": normal_current["period_id"],
        "live_default_expected_comparison": normal_score,
        "mechanism_only_period_id": deviation_current["period_id"],
        "mechanism_only_expected_comparison": deviation_score,
        "usage_rule": (
            "正式直播对照只加载 live_default；mechanism_only 只能单独展示，"
            "不得混入正常/失衡/跌倒实拍结果。"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a disclosed synthetic personal-baseline fixture for the live demo."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = write_fixture(args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
