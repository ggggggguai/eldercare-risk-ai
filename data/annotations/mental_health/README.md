# 心理健康风险标注目录

当前目录只预留心理健康风险真值标签，尚无可用于正式风险模型评估的量表、专家意见或纵向标签。建议后续补充：

```text
daily_rhythm_labels.jsonl
sleep_rhythm_labels.jsonl
social_interaction_labels.jsonl
mental_health_risk_labels.jsonl
self_report_scores.jsonl
review_log.jsonl
```

现有 `mental_health.wandering` 子包及步骤 2 转换产物属于隔离的轨迹数据适配与人工联系表复核，不是本目录的风险真值，也未接入日级评分主链。正式 split、轨迹分类模型和摄像头域验证完成前，不应把这些产物写成心理健康或徘徊识别标签。

心理健康标签必须谨慎处理。若没有量表、专家意见或稳定行为证据，不建议直接给出强标签。
