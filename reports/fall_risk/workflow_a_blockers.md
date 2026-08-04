# 工作流 A 当前阻塞

核验日期：2026-08-04

跌倒标签契约已升级为 v2。明确标注只要 schema、媒体关联、来源文件和 hash 正确即可使用。重复内容、未获内部授权的来源未知数据、媒体探测失败、隔离数据和 `U01/uncertain` 仍被排除。

| ID | 当前事实 | 解除动作 | 完成证据 |
|---|---|---|---|
| A-B01 | Coffee 原始 CVAT ZIP 已找回且 hash 与 514+514 条转换标签一致，但包含 CVAT 用户名元数据 | 将原件放在受控位置；如需进入版本库，先生成脱敏导出并重新转换 | 身份字段扫描结果和新来源 hash |
| A-B02 | v2 根标签已发布为 9,314 条动作、6,338 条事件；NTU 媒体路径已恢复，Fall Detection 2017 批次已接入候选链；全库 formal 有 285 个 blocker | 处理其他来源的 129 条 `U01/uncertain`（动作/事件层共 258 个 blocker）、27 条 manifest 技术排除关联，并完成剩余数据治理 | v2 formal 报告 `formal_ready=true` |
| A-B03 | 当前标签中仍有未知人员或保守源组，不能声明个体泛化 | 使用可靠元数据恢复脱敏人员 ID；无法恢复时继续使用保守 `source_group_id` | 人员映射或源组说明 |
| A-B04 | `risk_labels.jsonl` 和 subject profiles 为空，没有功能 proxy 与纵向状态变化真值 | 定义功能量表/状态变化终点并取得有 consent 的连续数据 | 非空风险标签、人员画像和任务 formal 报告 |
| A-B05 | 尚未指定测试集保管人和一次性发布流程 | 指定保管角色、冻结日期和测试执行规则 | 盲测治理记录与冻结 split ID |
| A-B06 | 事件评估配置仍为 provisional | 固定 IoU、onset、搜索窗口、阈值、最小样本量和 10,000 次聚类 bootstrap | frozen 配置、hash 和评审记录 |
| A-B07 | 连续监控合格时长为 0 | 采集或确认连续摄像机时长和家庭日 | manifest 连续分母及评估输出 |
| A-B09 | Fall Detection 2017 人工 CVAT 批次已导入 2,977 条动作/事件并进入 v2/v3 候选链，但 `provenance_status=project_collected_manual_cvat_unverified`、策略为 `candidate_requires_qc_review`；2 个源技术排除、1 个可用源未匹配 | 完成项目来源授权/追溯、缺失源复核、U01 语义复核和人工 QC；未完成前只能作为候选训练数据，不得进入 frozen 正式评估 | 批次审计报告、来源决定、缺失源处理记录和 formal 校验更新 |

## 2026-08-04 已解除

- `A-B08` 的事件监督阻塞已解除：哈希绑定的项目负责人裁决生成 962 条 near-fall positive、1,774 条 fall negative 和 4,201 条 near-fall negative，全部 15 类 task-specific hard negative 有 primary 覆盖；18,812 条 assignment 的统一 split 无泄漏，`training_ready.fall_event=true`、`training_ready.near_fall_event=true`。`training_ready.action_type=false` 仍记录为动作类型开发限制，但不再阻塞两个事件任务。
- 该解除只覆盖标签与 split 数据门槛。`A-B03` 老人/人员泛化、`A-B04` 功能与纵向真值、`A-B05/A-B06` 盲测和冻结协议、`A-B07` 连续监控分母及 `A-B09` 来源治理仍未解除。

验证命令：

```bash
conda run -n eldercare-ai python scripts/annotation/validate_fall_risk_labels.py \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --action-labels data/annotations/fall_risk/action_labels.jsonl \
  --event-labels data/annotations/fall_risk/event_labels.jsonl \
  --risk-labels data/annotations/fall_risk/risk_labels.jsonl \
  --subject-profiles data/annotations/fall_risk/subject_profiles.json \
  --config configs/data/fall_risk_label_validation_v2.yaml \
  --mode formal \
  --report-output reports/fall_risk/label_validation_formal_v2.json
```
