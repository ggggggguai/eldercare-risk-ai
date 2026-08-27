from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


VIDEO_COLUMNS = [
    "test_case_id", "relative_path", "sha256", "dataset_version", "run_id",
    "model_versions", "decode_status", "frame_count", "processed_frame_count",
    "person_detection_coverage", "pose_valid_frame_ratio", "track_break_count",
    "expected_actions", "expected_result", "inference_status", "actual_result",
    "max_risk_level", "max_gait_score", "max_sit_stand_score",
    "max_near_fall_score", "max_fall_score", "predicted_gait_event_count",
    "predicted_sit_stand_event_count", "predicted_near_fall_event_count",
    "predicted_fall_event_count", "confidence_or_quality", "rejection_reason",
    "error_class", "latency_ms", "realtime_factor", "pass_fail", "evidence_path",
]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object per line: {path}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, columns: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _interval_overlap(left_start: float, left_end: float, right_start: float, right_end: float) -> bool:
    return left_end >= right_start and right_end >= left_start


def _branch_payload(directory: Path, branch: str) -> dict[str, Any]:
    metrics_path = directory / f"metrics_{branch}.json"
    prediction_name = "predictions_near_fall_event.jsonl" if branch == "near_fall_event" else "predictions_fall_event.jsonl"
    payload = _read_json(metrics_path)
    payload["predictions"] = _read_jsonl(directory / prediction_name)
    return payload


def _index_by_video(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        indexed[str(row["video_id"])].append(dict(row))
    return dict(indexed)


def _match_indexes(payload: Mapping[str, Any]) -> tuple[set[str], set[str], set[str]]:
    matched_predictions = {str(row["prediction_id"]) for row in payload.get("matches", [])}
    matched_labels = {str(row["label_id"]) for row in payload.get("matches", [])}
    false_positive_ids = {str(row["prediction_id"]) for row in payload.get("false_positives", [])}
    return matched_predictions, matched_labels, false_positive_ids


def _operational_alarm_summary(
    truth_rows: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize alert-time hits separately from boundary-based event scoring."""
    truths = [row for row in truth_rows if row.get("event_type") != "uncertain"]
    preds = [row for row in predictions if row.get("status") == "emitted"]
    hit_labels: set[str] = set()
    hit_predictions: set[str] = set()
    for truth in truths:
        video_id = str(truth.get("video_id"))
        start = float(truth["start_time"])
        end = float(truth["end_time"])
        for prediction in preds:
            if str(prediction.get("video_id")) != video_id:
                continue
            alert = prediction.get("alert_time")
            if alert is None:
                continue
            if start <= float(alert) <= end:
                hit_labels.add(str(truth.get("label_id")))
                hit_predictions.add(str(prediction.get("prediction_id")))
    video_truth = {str(row.get("video_id")) for row in truths}
    video_hits = {
        str(row.get("video_id"))
        for row in truths
        if str(row.get("label_id")) in hit_labels
    }
    return {
        "truth_event_count": len(truths),
        "truth_event_hit_count": len(hit_labels),
        "truth_event_hit_recall": len(hit_labels) / len(truths) if truths else None,
        "truth_video_count": len(video_truth),
        "truth_video_hit_count": len(video_hits),
        "truth_video_hit_recall": len(video_hits) / len(video_truth) if video_truth else None,
        "alert_count_inside_truth_intervals": len(hit_predictions),
        "definition": "alert_time lies within the exact annotated interval; diagnostic only",
    }


def build_bundle(
    *,
    manifest_path: Path,
    ground_truth_path: Path,
    action_labels_path: Path,
    fall_eval_dir: Path,
    near_fall_eval_dir: Path,
    output_dir: Path,
    run_id: str,
    dataset_version: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output exists: {output_dir}")
    manifest = _read_jsonl(manifest_path)
    truth = _read_jsonl(ground_truth_path)
    actions = _read_jsonl(action_labels_path)
    manifest_by_video = {str(row["video_id"]): row for row in manifest}
    truth_by_branch = {
        branch: _index_by_video(row for row in truth if row.get("task_type") == branch)
        for branch in ("fall_event", "near_fall_event")
    }
    actions_by_video = _index_by_video(actions)
    fall = _branch_payload(fall_eval_dir, "fall_event")
    near = _branch_payload(near_fall_eval_dir, "near_fall_event")
    branch_payloads = {"fall_event": fall, "near_fall_event": near}
    truth_by_label = {
        str(row["label_id"]): row
        for row in truth
        if row.get("label_id") not in (None, "")
    }
    predictions_by_branch = {
        branch: _index_by_video(payload["predictions"])
        for branch, payload in branch_payloads.items()
    }

    output_dir.mkdir(parents=True)
    evidence_dir = output_dir / "10_evidence_examples"
    evidence_dir.mkdir()
    contracts = {
        "fall": _read_json(fall_eval_dir / "contract.json"),
        "near_fall": _read_json(near_fall_eval_dir / "contract.json"),
    }
    _write_json(output_dir / "01_run_manifest.json", {
        "schema_version": "fall-self-evaluation-run-manifest-v1",
        "run_id": run_id,
        "dataset_version": dataset_version,
        "evaluation_role": "engineering_replay_development_candidate",
        "status": "development_provisional",
        "input_manifest": str(manifest_path),
        "ground_truth": str(ground_truth_path),
        "action_labels": str(action_labels_path),
        "branch_contracts": contracts,
        "environment_mode": "disabled",
        "test_pose_read": False,
        "test_evaluated": False,
        "independent_test_claim_allowed": False,
        "known_boundary": "decoder and near-fall track parameters were explored on this batch",
    })

    test_case_rows = []
    for video_id, item in sorted(manifest_by_video.items()):
        video_actions = actions_by_video.get(video_id, [])
        test_case_rows.append({
            "test_case_id": video_id,
            "relative_path": item.get("path", ""),
            "sha256": item.get("content_sha256", item.get("sha256", "")),
            "unique_content_id": item.get("asset_id", ""),
            "source_group_id": item.get("source_group_id", ""),
            "media_form": item.get("media_type", "rgb_video"),
            "participant_code": item.get("subject_id", "unknown"),
            "scene_id": item.get("scene_region", "unknown"),
            "camera_setup_id": item.get("view", "unknown"),
            "annotation_path": item.get("annotation_path", ""),
            "expected_actions": _json_text(sorted({str(a.get("action_id")) for a in video_actions})),
            "expected_intervals": _json_text([
                {"action_id": a.get("action_id"), "start_time": a.get("start_time"), "end_time": a.get("end_time")}
                for a in video_actions
            ]),
            "evaluation_role": "engineering_replay",
            "review_status": "manual_cvat_import",
            "eligible_for_metric": "true",
            "exclusion_reason": "",
        })
    _write_csv(output_dir / "02_test_case_manifest.csv", list(test_case_rows[0]), test_case_rows)

    match_indexes = {branch: _match_indexes(payload) for branch, payload in branch_payloads.items()}
    video_rows = []
    for video_id, item in sorted(manifest_by_video.items()):
        video_json = _read_json(fall_eval_dir / "videos" / f"{video_id}.json")
        pose = video_json.get("pose", {})
        fall_events = predictions_by_branch["fall_event"].get(video_id, [])
        near_events = predictions_by_branch["near_fall_event"].get(video_id, [])
        expected = actions_by_video.get(video_id, [])
        expected_types = sorted({str(a.get("action_id")) for a in expected})
        max_quality = pose.get("mean_core_keypoint_quality")
        all_pred_ids = {str(p.get("prediction_id")) for p in fall_events + near_events}
        matched_ids = match_indexes["fall_event"][0] | match_indexes["near_fall_event"][0]
        false_ids = match_indexes["fall_event"][2] | match_indexes["near_fall_event"][2]
        truth_ids = {
            str(t.get("label_id"))
            for branch in truth_by_branch
            for t in truth_by_branch[branch].get(video_id, [])
        }
        matched_truth = match_indexes["fall_event"][1] | match_indexes["near_fall_event"][1]
        has_error = bool(false_ids or (truth_ids - matched_truth))
        video_rows.append({
            "test_case_id": video_id,
            "relative_path": item.get("path", ""),
            "sha256": item.get("content_sha256", item.get("sha256", "")),
            "dataset_version": dataset_version,
            "run_id": run_id,
            "model_versions": _json_text(sorted({str(p.get("model_version")) for p in fall_events + near_events})),
            "decode_status": video_json.get("status", "unknown"),
            "frame_count": item.get("frame_count", ""),
            "processed_frame_count": pose.get("detected_frame_count", ""),
            "person_detection_coverage": pose.get("detected_frame_coverage", ""),
            "pose_valid_frame_ratio": pose.get("detected_frame_coverage", ""),
            "track_break_count": max(0, int(pose.get("track_count", 0) or 0) - 1),
            "expected_actions": _json_text(expected_types),
            "expected_result": "positive" if any(str(a.get("action_id", "")).startswith(("C", "D")) for a in expected) else "background_or_unlabelled",
            "inference_status": video_json.get("status", "unknown"),
            "actual_result": _json_text(sorted({str(p.get("event_type")) for p in fall_events + near_events})),
            "max_risk_level": max([int(p.get("risk_level", 0) or 0) for p in fall_events + near_events] or [0]),
            "max_gait_score": "not_available",
            "max_sit_stand_score": "not_available",
            "max_near_fall_score": max([float(p.get("score", 0.0)) for p in near_events] or [0.0]),
            "max_fall_score": max([float(p.get("score", 0.0)) for p in fall_events] or [0.0]),
            "predicted_gait_event_count": "not_available",
            "predicted_sit_stand_event_count": "not_available",
            "predicted_near_fall_event_count": len(near_events),
            "predicted_fall_event_count": len(fall_events),
            "confidence_or_quality": max_quality if max_quality is not None else "not_available",
            "rejection_reason": "",
            "error_class": "false_positive_or_false_negative" if has_error else "",
            "latency_ms": "not_available",
            "realtime_factor": "not_available",
            "pass_fail": "fail" if has_error else "pass",
            "evidence_path": str(fall_eval_dir / "videos" / f"{video_id}.json"),
        })
    _write_csv(output_dir / "03_video_results.csv", VIDEO_COLUMNS, video_rows)

    event_rows: list[dict[str, Any]] = []
    for branch, payload in branch_payloads.items():
        matches_by_pred = {str(row["prediction_id"]): row for row in payload.get("matches", [])}
        for truth_row in payload.get("false_negatives", []):
            event_rows.append({"test_case_id": truth_row.get("video_id"), "event_id": truth_row.get("label_id"), "branch": branch, "truth_action": truth_row.get("event_type"), "truth_start": truth_row.get("start_time"), "truth_end": truth_row.get("end_time"), "predicted_type": "", "predicted_start": "", "predicted_end": "", "score": "", "threshold": payload.get("metrics", {}).get("score_threshold"), "match_rule": "unmatched_ground_truth", "overlap_or_time_error": "", "matched": False, "error_type": "false_negative", "false_positive": False, "false_negative": True, "quality_status": truth_row.get("quality"), "rejection_reason": ""})
        for prediction in payload.get("predictions", []):
            pred_id = str(prediction.get("prediction_id"))
            match = matches_by_pred.get(pred_id)
            matched_truth = truth_by_label.get(str(match.get("label_id"))) if match else None
            event_rows.append({"test_case_id": prediction.get("video_id"), "event_id": pred_id, "branch": branch, "truth_action": match.get("ground_truth_event_type", "") if match else "", "truth_start": matched_truth.get("start_time", "") if matched_truth else "", "truth_end": matched_truth.get("end_time", "") if matched_truth else "", "predicted_type": prediction.get("event_type"), "predicted_start": prediction.get("start_time"), "predicted_end": prediction.get("end_time"), "score": prediction.get("score"), "threshold": payload.get("metrics", {}).get("score_threshold"), "match_rule": "iou_or_onset" if match else "unmatched_prediction", "overlap_or_time_error": match.get("boundary_iou", "") if match else "", "matched": bool(match), "error_type": "" if match else "false_positive", "false_positive": not bool(match), "false_negative": False, "quality_status": prediction.get("quality_state"), "rejection_reason": "" if match else "unmatched_prediction"})
    _write_jsonl(output_dir / "04_event_results.jsonl", event_rows)

    metrics_payload = {
        "schema_version": "fall-self-evaluation-bundle-metrics-v1",
        "run_id": run_id,
        "dataset_version": dataset_version,
        "status": "development_provisional",
        "video_count": len(manifest_by_video),
        "unique_content_count": len({str(row.get("content_sha256", row.get("sha256", ""))) for row in manifest}),
        "branches": {
            branch: {
                "metrics": payload.get("metrics", {}),
                **(
                    {
                        "operational_alarm_in_interval": _operational_alarm_summary(
                            [row for row in truth if row.get("task_type") == branch],
                            payload.get("predictions", []),
                        )
                    }
                    if branch == "fall_event"
                    else {}
                ),
                "confidence_intervals_95": payload.get("confidence_intervals_95", {}),
                "protocol_version": payload.get("protocol_version"),
                "protocol_status": payload.get("protocol_status"),
            }
            for branch, payload in branch_payloads.items()
        },
        "metric_boundary": "fall and near_fall remain separate; C03-C05 are development proxy actions in this batch",
        "independent_test_claim_allowed": False,
    }
    _write_json(output_dir / "05_metrics.json", metrics_payload)

    slice_rows = []
    for key_name in ("participant_code", "scene_id"):
        values = defaultdict(list)
        for row in test_case_rows:
            value = row.get(key_name, "unknown")
            values[str(value)].append(row["test_case_id"])
        for value, ids in sorted(values.items()):
            id_set = set(ids)
            slice_rows.append({"slice_type": key_name, "slice_value": value, "video_count": len(ids), **{f"{branch}_{metric}": sum(1 for item in branch_payloads[branch].get("matches", []) if str(item.get("video_id")) in id_set) if metric == "tp" else sum(1 for item in branch_payloads[branch].get("false_positives", []) if str(item.get("video_id")) in id_set) if metric == "fp" else sum(1 for item in branch_payloads[branch].get("false_negatives", []) if str(item.get("video_id")) in id_set) for branch in branch_payloads for metric in ("tp", "fp", "fn")}})
    _write_csv(output_dir / "06_metrics_by_slice.csv", list(slice_rows[0]) if slice_rows else ["slice_type", "slice_value", "video_count"], slice_rows)

    failures = []
    for path in (fall_eval_dir / "failures.jsonl", near_fall_eval_dir / "failures.jsonl"):
        if path.is_file():
            failures.extend(_read_jsonl(path))
    _write_csv(output_dir / "07_failures_and_exclusions.csv", ["video_id", "status", "error_type", "error", "source"], [{**row, "source": str(path)} for path in (fall_eval_dir, near_fall_eval_dir) for row in (_read_jsonl(path / "failures.jsonl") if (path / "failures.jsonl").is_file() else [])])
    _write_csv(output_dir / "08_latency_summary.csv", ["metric", "value", "status"], [{"metric": key, "value": "not_available", "status": "not_recorded_by_evaluator"} for key in ("p50_ms", "p95_ms", "max_ms", "realtime_factor")])
    _write_json(output_dir / "11_runtime_environment.json", {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "environment": "conda eldercare-ai", "environment_mode": "disabled"})

    fall_metrics = fall.get("metrics", {})
    near_metrics = near.get("metrics", {})
    fall_alarm = metrics_payload["branches"]["fall_event"].get("operational_alarm_in_interval", {})
    summary = f"""# 自采视频评测候选结果\n\n- 运行 ID：`{run_id}`\n- 唯一视频：`{len(manifest_by_video)}`\n- 评测状态：`development_provisional`\n\n## 结果\n\n| 分支 | TP | FP | FN | Precision | Recall | F1 |\n|---|---:|---:|---:|---:|---:|---:|\n| 跌倒事件（IoU/onset 正式口径） | {fall_metrics.get("tp")} | {fall_metrics.get("fp")} | {fall_metrics.get("fn")} | {fall_metrics.get("precision")} | {fall_metrics.get("recall")} | {fall_metrics.get("f1")} |\n| 近跌倒代理事件 | {near_metrics.get("tp")} | {near_metrics.get("fp")} | {near_metrics.get("fn")} | {near_metrics.get("precision")} | {near_metrics.get("recall")} | {near_metrics.get("f1")} |\n\n跌倒分支另有告警时刻诊断：{fall_alarm.get("truth_event_hit_count")}/{fall_alarm.get("truth_event_count")} 个真值事件的 `alert_time` 落在人工区间内（Recall={fall_alarm.get("truth_event_hit_recall")}）。该数值不替代正式事件边界 F1。候选解码参数、逐视频输出和完整指标见 `01_run_manifest.json`、`03_video_results.csv`、`04_event_results.jsonl` 和 `05_metrics.json`。\n\n本包把跌倒与近跌倒分开统计，不把代理动作当作临床或独立测试真值。当前数据已经被查看并用于候选比较，因此只能作为工程回放/开发候选证据，不能写成独立测试准确率或老人域泛化结论。\n"""
    (output_dir / "00_README.md").write_text(summary, encoding="utf-8")
    (output_dir / "09_summary.md").write_text(summary, encoding="utf-8")

    sums = []
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
        sums.append(f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="ascii")
    return metrics_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a traceable self-collected fall evaluation bundle.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--action-labels", type=Path, required=True)
    parser.add_argument("--fall-eval-dir", type=Path, required=True)
    parser.add_argument("--near-fall-eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-version", default="fall-nearfall-v1-self-collected-20260826")
    args = parser.parse_args(argv)
    build_bundle(
        manifest_path=args.manifest,
        ground_truth_path=args.ground_truth,
        action_labels_path=args.action_labels,
        fall_eval_dir=args.fall_eval_dir,
        near_fall_eval_dir=args.near_fall_eval_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        dataset_version=args.dataset_version,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
