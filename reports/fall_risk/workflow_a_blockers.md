# 工作流 A 人工与外部阻塞清单

核验日期：2026-07-15

本文只记录自动化无法合法代替、且已由仓库文件或本地审计确认的阻塞。自动化工具完成不等于这些阻塞已经解除。

## 当前结论

`fall-risk-data-v1` 目前只能作为发布候选，不能标记为正式冻结版本。自动化底座已经生成 manifest、候选标签、严格审计、四类 blocked split 和合成评估证据包；现有真实标签仍未达到正式评估资格，正式 split 和比赛指标不得生成。

## 阻塞项

| ID | 已核验事实 | 责任角色 | 所需输入或动作 | 解除证据 |
|---|---|---|---|---|
| A-B01 | 两个已受 Git 跟踪的 CVAT 原始导出和一个未跟踪导出包含非空的 CVAT 用户名/邮箱类元数据。审计未输出具体值。 | 数据管理员、仓库管理员 | 决定原件的受控留存位置、生成脱敏副本，并决定是否取消跟踪及治理历史。任何删除、替换或历史改写均需项目负责人明确批准。 | 脱敏检查记录；版本控制中不再发布身份字段；原件访问与留存说明。 |
| A-B02 | 根 `action_labels.jsonl` 与 `event_labels.jsonl` 各 922 条，全部为 `pending`；204 条 C/D 高风险动作没有复核日志证据。严格 audit 另确认根标签有 2,766 个 v1 schema 错误，不能直接提升状态。 | 两名独立标注员、复核人员、独立仲裁人员 | 从 source-specific 候选完成双人独立复核；迁移来源链，冲突只能由未参与前序复核的人员仲裁。不得覆盖根标签或由脚本自动升级。 | 两个不同 `reviewer_id` 的最终记录 hash 证据；有效冲突链；formal 校验 0 error/0 blocker。 |
| A-B03 | 922 条动作标签的 `subject_id` 全为 `unknown`。LE2I 当前仓库证据不能恢复可靠人员身份。 | 数据管理员、数据集负责人 | 依据官方元数据恢复脱敏人员 ID；无法恢复时确认 LE2I 仅作事件开发/外部测试，并接受不能声明人员泛化的限制。 | 人员映射来源记录，或签字确认的保守 source-group 使用说明。 |
| A-B04 | manifest 对 LE2I 190、TOAGA 423、Pre_VFallp 108 个资产保守记录 `license_unknown`；这表示仓库内缺少可供自动确认的法律证据，不表示对数据集法律状态作结论。Pre_VFallp 另缺可靠来源与标签语义。 | 合规/法务负责人、数据管理员 | 依据官方材料核对来源、版本、许可文本和比赛/研究用途；在确认前保持不可用。 | 许可证清单、官方来源链接/版本、用途确认和 manifest 更新审计。 |
| A-B05 | 根标签缺少 v1 所需的稳定来源记录、导出 hash、资格与复核外键；Lecture/Office 又没有官方 TXT。现有旧记录不能恢复完整来源链。 | 数据管理员 | 找回对应原始 CVAT 导出并先脱敏，再重新生成 source-specific 候选；无法找回的记录保持不可追溯、不可用于正式指标。 | 导出 SHA-256、稳定来源 ID、manifest 外键和 0-error audit 报告。 |
| A-B06 | `risk_labels.jsonl`、`annotation_review_log.jsonl` 和 subject profiles 目前为空模板；没有功能 proxy 参考终点或纵向状态变化真值，LTMM 本地没有长期原始 `.dat` 信号。 | 临床/功能终点负责人、数据负责人 | 预注册序数风险标签、功能量表映射和纵向变化终点；取得合法长期信号与真人知情同意。不得用模型预测代替真值。 | 有 consent 的 profile、双人复核 risk 标签、终点定义和相应任务 formal 校验。 |
| A-B07 | 目前没有测试集保管人、调参人员隔离记录，也没有证据证明候选测试标签未被用于调参。 | 项目负责人、测试集保管人 | 指定角色、冻结日期和一次性正式评估流程；若测试集已污染则另建盲测。 | 盲测治理记录、冻结 split ID 和访问记录。 |
| A-B08 | IoU、onset 容忍、搜索窗口、合并/复位、主指标、最小样本量与 10,000 次 bootstrap 方案尚未由协议负责人预注册；现有两份 evaluation YAML 明确为 provisional。 | 评估协议负责人 | 审核开发配置并创建新的 frozen 配置版本；不得根据测试结果反向选择。 | 评审记录、冻结配置 SHA-256、协议版本与 clean-Git 复现。 |
| A-B09 | LE2I 的 130 份官方 TXT 实测为 99 个跌倒窗口、31 个明确 `0/0`、0 个 bbox-only。`0/0` 当前只表示没有官方跌倒窗口，导入器不生成正事件。 | 数据负责人、协议负责人 | 若要将 31 个 `0/0` 用作正式负样本，预注册负样本单位和连续暴露定义；否则保持为“无正窗口”且不提供传统 FPR 分母。 | 负样本协议与复核记录，或明确排除说明。 |
| A-B10 | manifest 中 `continuous_monitoring_eligible=true` 为 0，合法连续监控时长和家庭日分母均为 0。 | 数据采集负责人、评估协议负责人 | 采集或确认合法连续监控区间、摄像机时长、家庭日和负样本单位。短事件剪辑不得折算成连续小时。 | 带来源与许可的连续分母字段；validator 与评估报告通过。 |

## 解除阻塞后的验证命令

完成相应人工输入后，先运行严格校验，再构建 split；不要直接修改发布候选状态：

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
  --report-output reports/fall_risk/label_validation_formal.json

# 先由协议负责人创建并批准下列 versioned frozen 配置，不要原地改 provisional 文件。
conda run -n eldercare-ai python scripts/split/build_fall_risk_splits.py \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --annotations-dir data/annotations/fall_risk \
  --config configs/data/fall_risk_splits_v1.frozen.yaml \
  --validation-report reports/fall_risk/label_validation_formal.json \
  --validation-config configs/data/fall_risk_label_validation_v1.yaml \
  --output-dir data/splits/fall_risk-frozen-v1
```

只有 split 已冻结、评估协议已冻结且测试集治理记录齐全时，才可运行正式评估。当前 provisional 配置只允许合成或开发烟测。
