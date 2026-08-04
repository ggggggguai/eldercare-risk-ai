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
- [`fall_risk/training-labels-v3-validation.json`](fall_risk/training-labels-v3-validation.json)：v3 schema、引用、训练等级和训练门禁结果；当前两个事件任务通过，动作类型任务未通过。
- [`fall_risk/runtime/README.md`](fall_risk/runtime/README.md)：实时链路阶段 0/1 的固定输入、配置指纹和离线服务回放；不是算法效果报告。
- [`reproducibility/dataset_and_split_versions.md`](reproducibility/dataset_and_split_versions.md)：数据、split、配置和合成证据包哈希。

模型实验入口：

- [`fall_risk/kinecal_gait_tcn/README.md`](fall_risk/kinecal_gait_tcn/README.md)：KINECAL 14 点轻量步态 TCN 的固定划分 baseline、失败结论和复现命令。
- [`fall_risk/gait_window_v4_hierarchical_masked/development-20260803/README.md`](fall_risk/gait_window_v4_hierarchical_masked/development-20260803/README.md)：不增加数据条件下的质量捷径隔离、walking gate、三个 seed 结果和不替换主路径的结论。
- [`fall_risk/gait_window_v5_effect_first/development-20260803/README.md`](fall_risk/gait_window_v5_effect_first/development-20260803/README.md)：B01-B04 functional proxy 的六 seed 训练与概率集成结果；F1 `0.308`，test 未读取，未替换规则主路径。
- [`fall_risk/sit_stand_event_v1/README.md`](fall_risk/sit_stand_event_v1/README.md)：坐站 candidate-clip Logistic/TCN provisional validation；不提供连续事件定位或 test 结论。
- [`fall_risk/near_fall_event_v1/README.md`](fall_risk/near_fall_event_v1/README.md)：近跌倒恢复确认 TCN 三 seed provisional pilot；validation F1 `0.990-0.997`，窗口全部来自 NTU，test 未读取，未替换规则主路径。
- [`fall_risk/fall_event_proxy_v2_v3split/README.md`](fall_risk/fall_event_proxy_v2_v3split/README.md)：当前 v3 split 的跌倒 candidate-clip TCN 三 seed provisional pilot；validation F1 `0.958-0.963`，test 未读取，未替换规则主路径。
