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
- [`fall_risk/self-collected-data-audit-20260810.md`](fall_risk/self-collected-data-audit-20260810.md)：2026-08-10 自采 P01/P02、P03 与未归属来源的完整性、重复、媒体、授权和治理状态审查；全部暂不进入训练或正式评测。
- [`fall_risk/self-collected-data-audit-20260811.md`](fall_risk/self-collected-data-audit-20260811.md)：新交付 P01–P05 的哈希对账、CVAT 脱敏与一一对应校验、P04/P05 增量整理和当前入链门禁。
- [`fall_risk/self_collected_scf_mvp_v1/README.md`](fall_risk/self_collected_scf_mvp_v1/README.md)：SCF_MVP_V1 候选审计、根标签发布、262 段冻结回放和 G2 门禁；P01/P02/P04 已进入 train，P03/P05 保持隔离。G2 近跌倒 checkpoint 仍为 No-Go，步态运行启用另见下一项。
- [`fall_risk/gait_runtime_activation_20260818.md`](fall_risk/gait_runtime_activation_20260818.md)：步态 pretrained seed 43 的比赛交付受控启用、配置路径、fallback、验证和回滚记录。
- [`fall_risk/workflow_a_blockers.md`](fall_risk/workflow_a_blockers.md)：来源、隐私、真值与数据门槛。
- [`fall_risk/fall-risk-data-v2-release-candidate.md`](fall_risk/fall-risk-data-v2-release-candidate.md)：发布候选验收结论。
- [`fall_risk/training-labels-v3-migration.json`](fall_risk/training-labels-v3-migration.json)：v2 到模型训练标签 v3 的确定性迁移计数与 hash。
- [`fall_risk/training-labels-v3-validation.json`](fall_risk/training-labels-v3-validation.json)：v3 schema、引用、训练等级和训练门禁结果；当前两个事件任务通过，动作类型任务未通过。
- [`fall_risk/runtime/README.md`](fall_risk/runtime/README.md)：实时链路阶段 0/1 的固定输入、配置指纹和离线服务回放；不是算法效果报告。
- [`reproducibility/dataset_and_split_versions.md`](reproducibility/dataset_and_split_versions.md)：数据、split、配置和合成证据包哈希。

模型实验入口：

- [`fall_risk/fall_risk_release_freeze_20260819.md`](fall_risk/fall_risk_release_freeze_20260819.md)：当前混合模型的正式比赛交付冻结决定、release 边界、证据解释和完整性测试入口。
- [`fall_risk/fall_nearfall_v1/README.md`](fall_risk/fall_nearfall_v1/README.md)：115 条居家模拟动作视频上的冻结 v1 跌倒/近跌倒开发性工程评测、事件级指标、误报检查和复现命令；不是 frozen test 或老人域泛化证据。
- [`fall_risk/fall_nearfall_adaptation_v1/README.md`](fall_risk/fall_nearfall_adaptation_v1/README.md)：将 11 条自采跌倒正例加入开发训练后，在 44 条未见居家视频上的域内适配对照；不替换冻结 v1，也不是跨主体泛化证据。

- [`fall_risk/kinecal_gait_tcn/README.md`](fall_risk/kinecal_gait_tcn/README.md)：KINECAL 14 点轻量步态 TCN 的固定划分 baseline、失败结论和复现命令。
- [`fall_risk/gait_window_v4_hierarchical_masked/development-20260803/README.md`](fall_risk/gait_window_v4_hierarchical_masked/development-20260803/README.md)：不增加数据条件下的质量捷径隔离、walking gate、三个 seed 结果和不替换主路径的结论。
- [`fall_risk/gait_window_v5_effect_first/development-20260803/README.md`](fall_risk/gait_window_v5_effect_first/development-20260803/README.md)：B01-B04 functional proxy 的六 seed 训练与概率集成结果；F1 `0.308`，test 未读取，未替换规则主路径。
- [`fall_risk/gait_observable_v1/splitv3-e71a045/README.md`](fall_risk/gait_observable_v1/splitv3-e71a045/README.md)：当前 split 的 train/validation-only 双构建、rule/结构化 baseline 与单 seed TCN smoke；validation 正类规模门禁未通过，test 未读取。
- [`fall_risk/gait_observable_context_v2/splitv3-e71a045/README.md`](fall_risk/gait_observable_context_v2/splitv3-e71a045/README.md)：同轨真实上下文、label-span 掩码和证据分层的受限消融；train 正类监督段增至 92，正式 validation 门禁仍未通过。
- [`fall_risk/sit_stand_event_v1/README.md`](fall_risk/sit_stand_event_v1/README.md)：坐站 candidate-clip Logistic/TCN provisional validation；不提供连续事件定位或 test 结论。
- [`fall_risk/sit_stand_training_audit.md`](fall_risk/sit_stand_training_audit.md)与[`fall_risk/sit_stand_training_blockers.md`](fall_risk/sit_stand_training_blockers.md)：连续坐站 P0 hash/test 隔离审计和真实标签、背景、冻结协议阻塞；当前为 `infrastructure_only`。
- [`fall_risk/sit_stand_event_v2/README.md`](fall_risk/sit_stand_event_v2/README.md)：SCF 发布后的坐站专项 v2 治理、物化感知保护组 split、11,605 个多 cutoff 因果窗口和同协议 TCN/规则完整流开发评估；正式模型门禁仍为 No-Go，比赛期默认运行已临时切换为 TCN-first。
- [`fall_risk/sit_stand_runtime_activation_20260818.md`](fall_risk/sit_stand_runtime_activation_20260818.md)：比赛期坐站 TCN-first 默认启用记录；保留规则 fallback，正式模型门禁仍为 No-Go。
- [`fall_risk/near_fall_event_v1/README.md`](fall_risk/near_fall_event_v1/README.md)：当前 v3 split 的近跌倒恢复确认 TCN 三 seed 开发训练；validation 事件 F1 `0.9933-0.9967`，test 未读取，困难负例/连续背景/老人域门禁仍阻塞，未替换规则主路径。
- [`fall_risk/near_fall_event_v2/README.md`](fall_risk/near_fall_event_v2/README.md)：SCF 发布后按完整同轨上下文和 3 秒/2 秒回退重建的近跌倒数据；E1 仅加入受审负例，最终为 3,842 个窗口、1,695 个事件，三 seed P05 结果仍为 No-Go。
- [`fall_risk/fall_event_proxy_v2_v3split/README.md`](fall_risk/fall_event_proxy_v2_v3split/README.md)：当前 v3 split 的跌倒 candidate-clip TCN 三 seed provisional pilot；validation F1 `0.958-0.963`，test 未读取，未替换规则主路径。
- [`fall_risk/fall_event_continuous_governance_v1/README.md`](fall_risk/fall_event_continuous_governance_v1/README.md)：历史 `splitv3_c7fd...` 的连续跌倒训练监督治理；已由当前 split 的 v2 治理取代，只作追溯。
- [`fall_risk/fall_event_continuous_governance_v2/README.md`](fall_risk/fall_event_continuous_governance_v2/README.md)：当前 `splitv3_334...` 的 7,701 条连续跌倒开发监督、姿态覆盖、平衡权重和 test 隔离审计。
- [`fall_risk/fall_event_continuous_tcn_v2/README.md`](fall_risk/fall_event_continuous_tcn_v2/README.md)：当前 `splitv3_334...` 上重新治理的 7,504 个因果窗口、三 seed TCN、validation 概率集成和分层误报审计；当前为 No-Go，test 未读取，未替换规则主路径。
- [`fall_risk/runtime/ezviz-live-smoke-20260812.md`](fall_risk/runtime/ezviz-live-smoke-20260812.md)：真实萤石 HTTPS-FLV 的 120 秒算法端严格烟测；业务回调、地址刷新、弱网和一小时资源验收仍未完成。
- [`fall_risk/baseline_longitudinal_phase2.md`](fall_risk/baseline_longitudinal_phase2.md)：个体基线 Phase 2 的 longitudinal schema、outcome-blind 前向 split、四组消融评估基础设施、合成验收和真实空数据 blocker；不是老人域效果报告。

心理健康实验入口：

- [`mental_health/wandering_step2/README.md`](mental_health/wandering_step2/README.md)：WanderingPatterns/SmartCare 步骤 2 的来源哈希、转换计数、异常、确定性验证和人工联系表入口；不是正式 split、模型效果或现有心理健康评分能力。
- [`mental_health/wandering_step2/HUMAN_REVIEW.md`](mental_health/wandering_step2/HUMAN_REVIEW.md)：六类固定抽样联系表的人工复核记录与后续限制。
