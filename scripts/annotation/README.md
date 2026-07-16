# 标注脚本目录

本目录只放标注格式转换、官方标签导入、标注质检和一致性统计脚本。跌倒风险和心理健康风险必须分别统计，不能混用标签或评估口径。

以下命令都必须在仓库根目录执行，并使用项目 conda 环境。正式处理前先确认 editable 安装指向当前仓库：

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
```

## 1. 构建统一 manifest

```bash
conda run -n eldercare-ai python scripts/annotation/build_fall_risk_manifest.py \
  --repo-root . \
  --output data/manifests/fall_risk_video_manifest.jsonl \
  --ffprobe-bin ffprobe
```

构建器逐资产记录哈希和来源信息，并从每条视频读取真实 `fps_num/fps_den`、帧数、时长和分辨率。默认拒绝覆盖已有 manifest；重建正式版本前应先完成版本决策，不要随意追加 `--overwrite`。

## 2. 转换 CVAT 候选标签

CVAT `CVAT for video 1.1` XML 或 ZIP 必须先转换到来源专属候选目录：

```bash
conda run -n eldercare-ai python scripts/annotation/convert_cvat_fall_labels.py \
  --input data/annotations/fall_risk/cvat_exports/raw/le2i_home_01_first_2_videos_cvat.zip \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --output-dir data/annotations/fall_risk/generated/v1/cvat_home_01 \
  --labeler labeler_fall_01
```

输出为该目录下的 `action_labels.jsonl` 和 `event_labels.jsonl`。转换器按 `video_id` 使用 manifest 中每条视频的有理 FPS，不得给正式批处理硬编码单一 FPS。`--fps` 和 `--file-root` 只允许配合 `--development-override` 做旧测试夹具兼容，不能用于正式候选生成。

转换结果固定为 `review_status=pending`、`eligibility=false`、`review_evidence_ids=[]`。转换器不会把候选提升为 `reviewed/final`，也不会覆盖已有输出。不同导出必须使用不同的来源目录；不要把 `--action-output` 或 `--event-output` 指向根目录正式标签。

候选目录按源导出批次命名，不是数据集 subset 的事实源。经审计，现有 `generated/v1/cvat_coffee_01_02/` 在 100 条视频上各含 514 条动作候选和映射事件，其中按记录计为 `Coffee_room_01=233`、`Coffee_room_02=150`、`Home_02=131`；各视频 FPS 已分别从 manifest 读取。这个目录不能被整体归类为 Coffee，统计和合并必须使用每条记录关联的 manifest subset。

若源 CVAT 导出含账号或邮箱等身份元数据，转换器只输出脱敏告警，不复制其值。不得把这些值写入文档、报告或标签。

## 3. 导入 LE2I 官方窗口候选

LE2I TXT 与人工 CVAT 标签保持独立来源，使用单独文件：

```bash
conda run -n eldercare-ai python scripts/annotation/import_le2i_fall_labels.py \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --event-output data/annotations/fall_risk/generated/v1/le2i_official/event_labels.jsonl \
  --report-output data/annotations/fall_risk/generated/v1/le2i_official/import_report.json
```

导入器只写 TXT 明确支持的跌倒窗口，保留 LE2I 的 1-based 源帧和统一 JSONL 的 0-based 帧；`0/0` 记录为显式无跌倒窗口，不生成事件。`Lecture room` 和 `Office` 没有官方 TXT，不进入官方有监督事件候选。输出固定为 `label_source=le2i_txt`、`review_status=auto_imported`、`eligibility=false`。

导入器默认拒绝覆盖事件和报告。上述路径已经存在时不要原地重跑；复现新版本应选择新的版本目录并保留旧输出哈希。

## 4. 严格校验

先对来源专属 CVAT 候选做审计；空的风险、复核和人员模板仍使用根目录文件：

```bash
conda run -n eldercare-ai python scripts/annotation/validate_fall_risk_labels.py \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --action-labels data/annotations/fall_risk/generated/v1/cvat_home_01/action_labels.jsonl \
  --event-labels data/annotations/fall_risk/generated/v1/cvat_home_01/event_labels.jsonl \
  --risk-labels data/annotations/fall_risk/risk_labels.jsonl \
  --subject-profiles data/annotations/fall_risk/subject_profiles.json \
  --review-log data/annotations/fall_risk/annotation_review_log.jsonl \
  --config configs/data/fall_risk_label_validation_v1.yaml \
  --mode audit \
  --report-output reports/fall_risk/cvat_le2i_home_01_candidate_validation.json
```

`audit` 模式会保留缺少人工复核等 blocker，但只有结构或来源错误才令命令失败。报告默认拒绝覆盖，重复运行时应使用新的版本化报告名。

只有人工确认、双人独立复核、来源和许可门槛全部满足后，才能对根目录统一标签运行正式校验：

```bash
conda run -n eldercare-ai python scripts/annotation/validate_fall_risk_labels.py \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --action-labels data/annotations/fall_risk/action_labels.jsonl \
  --event-labels data/annotations/fall_risk/event_labels.jsonl \
  --risk-labels data/annotations/fall_risk/risk_labels.jsonl \
  --subject-profiles data/annotations/fall_risk/subject_profiles.json \
  --review-log data/annotations/fall_risk/annotation_review_log.jsonl \
  --config configs/data/fall_risk_label_validation_v1.yaml \
  --mode formal \
  --report-output reports/fall_risk/annotation_validation_formal_v1.json
```

`formal` 模式把 `pending`、`uncertain`、未知许可、已有人员画像缺少同意引用、缺少独立复核或来源不完整均视为阻塞。候选目录中的文件即使通过结构审计，也不得自动复制、拼接或提升到根目录；根目录变更必须是可审计的人工数据发布动作。

当前字段契约、复核哈希和冲突仲裁规则见 `docs/modules/fall_risk/data/跌倒风险标签字典.md` 与 `docs/modules/fall_risk/data/数据标注SOP.md`。
