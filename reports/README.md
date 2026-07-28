# 实验报告目录

本目录只保存新的算法实验结果、复现记录和失败案例，不保存会议汇报或开发日志。历史材料已移至 [`docs/archive/reports/`](../docs/archive/reports/)。

建议按模块组织：

```text
reports/
  fall_risk/       跌倒检测、近跌倒、步态、坐站、基线和实时性能
  mental_health/   日级风险分层、人工复核一致性和长期趋势案例
  reproducibility/ 环境、配置、数据版本和复现实验记录
```

每份报告至少记录代码版本、配置、数据清单、划分方法、指标定义、结果、失败案例和复现命令。目标值与实测值必须明确区分。

Workflow A 当前入口：

- [`fall_risk/data_audit.md`](fall_risk/data_audit.md)：真实本地数据与标签审计。
- [`fall_risk/workflow_a_blockers.md`](fall_risk/workflow_a_blockers.md)：来源、隐私、真值与数据门槛。
- [`fall_risk/fall-risk-data-v2-release-candidate.md`](fall_risk/fall-risk-data-v2-release-candidate.md)：发布候选验收结论。
- [`fall_risk/training-labels-v3-migration.json`](fall_risk/training-labels-v3-migration.json)：v2 到模型训练标签 v3 的确定性迁移计数与 hash。
- [`fall_risk/training-labels-v3-validation.json`](fall_risk/training-labels-v3-validation.json)：v3 schema、引用、训练等级和训练门禁结果。
- [`fall_risk/runtime/README.md`](fall_risk/runtime/README.md)：实时链路阶段 0/1 的固定输入、配置指纹和离线服务回放；不是算法效果报告。
- [`reproducibility/dataset_and_split_versions.md`](reproducibility/dataset_and_split_versions.md)：数据、split、配置和合成证据包哈希。

模型实验入口：

- [`fall_risk/kinecal_gait_tcn/README.md`](fall_risk/kinecal_gait_tcn/README.md)：KINECAL 14 点轻量步态 TCN 的固定划分 baseline、失败结论和复现命令。
