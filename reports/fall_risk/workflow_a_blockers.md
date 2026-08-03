# 工作流 A 当前阻塞

核验日期：2026-08-02

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
| A-B08 | 模型训练标签 v3 已生成 9,314 条动作、2,303 个 fall positive、6 个 `squat_or_kneel` fall negative 和 258 个 task-specific ignore；统一 split 有 11,881 条 assignment、6,516 个资产和 184 个保守泄漏组。六条负例全部在 train，另六类 fall hard negative、near-fall positive 和部分动作三分区覆盖仍缺，三项 `training_ready` 均为 false | 按 v3 字典逐窗补齐其余 fall/near-fall hard negative；安全采集 C03-C05 并双人复核恢复与未跌倒结局；补足不破坏源组隔离的动作与负样本分区覆盖 | `training-labels-v3-validation.json` 中目标任务 `training_ready=true`，并有来源、复核和类别分布证据 |
| A-B09 | Fall Detection 2017 人工 CVAT 批次已导入 2,977 条动作/事件并进入 v2/v3 候选链，但 `provenance_status=project_collected_manual_cvat_unverified`、策略为 `candidate_requires_qc_review`；2 个源技术排除、1 个可用源未匹配 | 完成项目来源授权/追溯、缺失源复核、U01 语义复核和人工 QC；未完成前只能作为候选训练数据，不得进入 frozen 正式评估 | 批次审计报告、来源决定、缺失源处理记录和 formal 校验更新 |

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
