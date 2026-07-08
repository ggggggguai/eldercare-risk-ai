from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.fall_risk.annotations import write_fall_label_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 CVAT for video 1.1 导出转换为跌倒风险统一 action/event JSONL。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CVAT 导出的 annotations.xml 或 .zip 路径。",
    )
    parser.add_argument(
        "--action-output",
        type=Path,
        default=Path("data/annotations/fall_risk/action_labels.jsonl"),
        help="输出动作级 action_labels.jsonl 路径。",
    )
    parser.add_argument(
        "--event-output",
        type=Path,
        default=Path("data/annotations/fall_risk/event_labels.jsonl"),
        help="输出事件级 event_labels.jsonl 路径。",
    )
    parser.add_argument("--fps", type=float, default=24.0, help="视频帧率，用于把帧号换算为秒。")
    parser.add_argument(
        "--file-root",
        type=Path,
        default=None,
        help="原始视频目录；设置后会与 CVAT source 文件名拼成 file_path。",
    )
    parser.add_argument("--labeler", default="unknown", help="写入 action_labels.jsonl 的标注员 ID。")
    parser.add_argument(
        "--review-status",
        default="pending",
        choices=("pending", "reviewed", "final"),
        help="写入 JSONL 的复核状态。",
    )
    parser.add_argument("--subject-id", default="unknown", help="默认 subject_id。")
    parser.add_argument("--scene", default="home", help="默认场景。")
    parser.add_argument("--view", default="fixed_camera", help="默认视角。")
    args = parser.parse_args()

    counts = write_fall_label_jsonl(
        args.input,
        action_output_path=args.action_output,
        event_output_path=args.event_output,
        fps=args.fps,
        file_root=args.file_root,
        labeler=args.labeler,
        review_status=args.review_status,
        default_subject_id=args.subject_id,
        default_scene=args.scene,
        default_view=args.view,
    )
    print(f"已写入 {counts['action_labels']} 条动作标签：{args.action_output}")
    print(f"已写入 {counts['event_labels']} 条事件标签：{args.event_output}")


if __name__ == "__main__":
    main()
