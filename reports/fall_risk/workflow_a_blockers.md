# 工作流 A 当前阻塞

核验日期：2026-07-23

跌倒标签契约已升级为 v2。明确标注只要 schema、媒体关联、来源文件和 hash 正确即可使用。重复内容、未获内部授权的来源未知数据、媒体探测失败、隔离数据和 `U01/uncertain` 仍被排除。

| ID | 当前事实 | 解除动作 | 完成证据 |
|---|---|---|---|
| A-B01 | Coffee 原始 CVAT ZIP 已找回且 hash 与 514+514 条转换标签一致，但包含 CVAT 用户名元数据 | 将原件放在受控位置；如需进入版本库，先生成脱敏导出并重新转换 | 身份字段扫描结果和新来源 hash |
| A-B02 | v2 根标签已发布为 1,515 条动作、1,515 条事件；108 个 Pre_VFallp 视频贡献 246/246 条 CVAT 标签，CaucaFall 100 个视频贡献 311/311 条人工 CVAT 标签；formal 校验 `errors=0`，仍有 113 个 blocker | 处理 43 条 `U01/uncertain`（动作/事件层共 86 个 blocker）、27 条其他来源 manifest 技术不可用关联，并完成剩余数据治理 | v2 formal 报告 `formal_ready=true` |
| A-B03 | 当前标签中仍有未知人员或保守源组，不能声明个体泛化 | 使用可靠元数据恢复脱敏人员 ID；无法恢复时继续使用保守 `source_group_id` | 人员映射或源组说明 |
| A-B04 | `risk_labels.jsonl` 和 subject profiles 为空，没有功能 proxy 与纵向状态变化真值 | 定义功能量表/状态变化终点并取得有 consent 的连续数据 | 非空风险标签、人员画像和任务 formal 报告 |
| A-B05 | 尚未指定测试集保管人和一次性发布流程 | 指定保管角色、冻结日期和测试执行规则 | 盲测治理记录与冻结 split ID |
| A-B06 | 事件评估配置仍为 provisional | 固定 IoU、onset、搜索窗口、阈值、最小样本量和 10,000 次聚类 bootstrap | frozen 配置、hash 和评审记录 |
| A-B07 | 连续监控合格时长为 0 | 采集或确认连续摄像机时长和家庭日 | manifest 连续分母及评估输出 |
| A-B08 | 模型训练标签 v3 已生成 1,515 条动作、252 个 fall positive 和 86 个 task-specific ignore；CaucaFall 的 311 条人工轨迹已进入 v3；人工 event negative=0、near-fall positive=0，且 primary `slow_walk` 在 test 分区为 0，三项 `training_ready` 均为 false | 按 v3 字典逐窗补齐 fall/near-fall hard negative；安全采集 C03-C05 并双人复核恢复与未跌倒结局；补足不破坏源组隔离的动作测试覆盖 | `training-labels-v3-validation.json` 中目标任务 `training_ready=true`，并有来源、复核和类别分布证据 |

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
