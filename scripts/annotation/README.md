# 标注脚本目录

只放标注格式转换、标注质检、一致性统计脚本。

跌倒风险和心理健康风险应分别统计标签一致性，不能混用评估口径。

## CVAT 跌倒风险标注转换

将 CVAT `CVAT for video 1.1` 导出的 XML 或 ZIP 转为统一 JSONL：

```bash
conda run -n eldercare-ai python scripts/annotation/convert_cvat_fall_labels.py \
  --input data/annotations/fall_risk/cvat_exports/raw/le2i_home_01_first_2_videos_cvat.zip \
  --action-output data/annotations/fall_risk/action_labels.jsonl \
  --event-output data/annotations/fall_risk/event_labels.jsonl \
  --file-root "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos" \
  --fps 24 \
  --labeler eldercare \
  --review-status reviewed
```

转换脚本只处理人工动作/事件标签，不导入 LE2I 官方 txt。官方 txt 的跌倒窗口应由单独导入脚本写入 `event_labels.jsonl`，并保留 `label_source = le2i_txt`。
