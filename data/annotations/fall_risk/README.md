# 跌倒风险标注目录

根目录文件是统一标签契约，不是转换脚本的默认落点：

```text
action_labels.jsonl
event_labels.jsonl
risk_labels.jsonl
subject_profiles.json
annotation_review_log.jsonl
```

其中 `risk_labels.jsonl` 和 `annotation_review_log.jsonl` 在没有真实人工结果时必须保持空 JSONL（零条记录，不写 `[]` 或示例行）；`subject_profiles.json` 的空模板为：

```json
{
  "schema_version": "fall-risk-subject-profiles-v1",
  "subjects": []
}
```

`generated/v1/` 保存按来源隔离的自动转换候选，例如：

```text
generated/v1/cvat_<export-id>/action_labels.jsonl
generated/v1/cvat_<export-id>/event_labels.jsonl
generated/v1/le2i_official/action_labels.jsonl
generated/v1/le2i_official/event_labels.jsonl
generated/v1/le2i_official/import_report.json
```

`le2i_official/action_labels.jsonl` 是零条记录的配套审计输入；LE2I 导入器本身只生成官方事件和导入报告，不会把 TXT 人体框伪装成动作标签。

目录名只标识源导出批次，不能代替记录关联的 manifest subset。现有 `generated/v1/cvat_coffee_01_02/` 实际覆盖 `Coffee_room_01`、`Coffee_room_02` 和 `Home_02`；不得因目录名而把其中 `Home_02` 候选误归为 Coffee。

生成记录必须保持 `pending` 或 `auto_imported`、`eligibility=false` 和空 `review_evidence_ids`。通过 `audit` 只证明结构和来源可检查，不代表已经成为正式真值；未经人工确认、双人独立复核和 `formal` 校验，不得复制、拼接或覆盖根目录标签。

`cvat_exports/raw/` 保存不可变的原始 CVAT 导出。转换时若发现账号、邮箱等身份元数据，只记录脱敏风险，不得在标签、报告或文档中复制具体值。

标注规则以 `docs/modules/fall_risk/data/数据集标注规范.md`、`docs/modules/fall_risk/data/数据标注SOP.md` 和 `docs/modules/fall_risk/data/跌倒风险标签字典.md` 为准。

`quarantine/` 保存尚无法关联到本地原始视频的记录。这些记录不属于训练或评估输入；只有找回可验证的原始导出并完成重新映射后，才能回写到正式 JSONL。
