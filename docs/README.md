# 项目文档

更新时间：2026-08-27

本页是文档唯一总入口。现行文档描述当前代码和接口；计划文档描述目标；归档文档只保留历史上下文，不能作为当前实现依据。

## 推荐阅读顺序

1. [项目 README](../README.md)：范围、环境和服务启动方式。
2. [算法工程架构](architecture/算法工程骨架.md)：代码分层、两条算法链和当前实现状态。
3. [算法事件输出接口](interfaces/算法事件输出接口.md)：两个模块独立输出的 `AlgorithmEvent` 契约。
4. [跌倒风险模块](modules/fall_risk/README.md)或[心理健康模块](modules/mental_health/README.md)：模块入口、算法规则和运行方式。
5. [当前任务](tasks/README.md)：尚未完成、需要验证或需要校准的工作。
6. [协作规则](../AGENTS.md)：环境、测试、事实源、文档维护和模型替换门禁。

## 现行文档

| 类别 | 文档 | 用途 |
|---|---|---|
| 架构 | [算法工程架构](architecture/算法工程骨架.md) | 工程边界、代码分层和实现状态 |
| 架构 | [实时视频监测链路](architecture/实时视频监测前后端算法链路.md) | 跌倒直播服务与业务后端的数据流 |
| 接口 | [算法事件输出接口](interfaces/算法事件输出接口.md) | 通用事件字段和风险等级编码 |
| 接口 | [跌倒风险服务对接说明](interfaces/跌倒风险算法服务后端对接说明.md) | HTTP 会话、鉴权和回调契约 |
| 协作 | [AGENTS.md](../AGENTS.md) | 项目环境、开发流程、事实源和文档同步规则 |
| 交付 | [跌倒风险算法模块（含运行环境）比赛交付初稿](deliverables/跌倒风险算法模块（含运行环境）-比赛交付初稿.md) | 面向评审专家的模块章节草稿、图源与待补证据清单 |
| 交付 | [跌倒风险模块创新点总结](deliverables/跌倒风险模块创新点总结.md) | 比赛提交报告可直接引用的跌倒模块创新点与表述边界 |
| 跌倒 | [模块 README](modules/fall_risk/README.md) | 当前能力、命令、字段和限制 |
| 跌倒 | [正式比赛交付冻结决定](../reports/fall_risk/fall_risk_release_freeze_20260819.md) | v1 模型与规则组合、哈希冻结边界和证据解释 |
| 跌倒 | [正式比赛交付 v2 冻结记录](../reports/fall_risk/fall_risk_release_freeze_20260827.md) | v2 近跌倒运行晋级、坐站模型取舍、哈希冻结边界和证据限制 |
| 跌倒 | [协作开发指南](modules/fall_risk/guides/跌倒风险算法协作开发指南.md) | 开发约束、验证方式和代码职责 |
| 跌倒 | [研发计划](modules/fall_risk/plans/跌倒风险算法研发计划.md) | 目标路线、实验设计和阶段计划，不等于完成状态 |
| 跌倒 | [演示方案](modules/fall_risk/plans/跌倒风险演示方案.md) | 初审视频脚本、镜头顺序、素材清单和注意事项 |
| 跌倒 | [演示执行稿](modules/fall_risk/plans/跌倒风险演示执行稿.md) | 面向导演、现场统筹、摄制和后期的逐镜头执行、口令、验收和表述边界 |
| 跌倒 | [本地 Web 演示台](../src/elderly_monitoring/service/static/fall-risk.html) | `/demo/fall-risk` 初审演示入口；默认明确标记的模拟回放，可切换真实直播；实时态展示任务追踪、实际处理进度和回调状态，仓库外业务闭环明确标为未接入；录屏服务端口为 `8000` |
| 跌倒 | [下一阶段执行计划](modules/fall_risk/plans/跌倒风险下一阶段执行计划.md) | 当前实时链路与算法语义质量冲刺：时钟与采样契约、分支可用性、会话边界、异步回调、变形回放和运行可观测性 |
| 跌倒 | [算法技术方案](modules/fall_risk/plans/跌倒风险算法技术方案.md) | 分层算法链路、技术选型、模块输入输出、训练评估门槛和研发阶段路线 |
| 跌倒 | [挑战杯冲奖增强计划](modules/fall_risk/plans/挑战杯揭榜挂帅冲奖增强计划.md) | 官方评分映射、七周执行计划、验收门槛和提交证据 |
| 跌倒 | [算法链路剩余模型训练计划](modules/fall_risk/plans/算法链路剩余模型训练计划-比赛效果优先.md) | 以比赛模型效果为核心的步态、坐站、个体基线和风险融合训练路线、指标、消融与倒排计划 |
| 跌倒 | [个体化行为基线算法增强计划](modules/fall_risk/plans/个体化行为基线算法增强计划.md) | 因果周期基线、鲁棒状态机、纵向前向 split、消融评估和真实数据门禁；计划状态不等于真实效果 |
| 跌倒 | [环境因素多模态增强设计方案](modules/fall_risk/plans/环境因素多模态增强设计方案.md) | 环境增强的目标架构、输入契约、时间对齐、特征计算、融合边界和模块职责 |
| 跌倒 | [环境因素多模态增强实施计划](modules/fall_risk/plans/环境因素多模态增强实施计划.md) | ESP32-S3、照度与水浸传感接入，人-环境交互特征、规则融合、受控实验和验收边界 |
| 跌倒 | [工作流 A Codex 执行任务书](modules/fall_risk/plans/工作流A-Codex执行任务书.md) | 数据、标注与评估底座的代理执行范围、阶段门槛和验收条件 |
| 跌倒 | [工作流 B Codex 执行任务书](modules/fall_risk/plans/工作流B-Codex执行任务书.md) | 算法增强、实验矩阵、数据门禁和冻结交接；不作为已实现或实测结果证明 |
| 跌倒 | [跌倒模型训练计划：Codex 执行任务书](modules/fall_risk/plans/跌倒模型训练计划-Codex执行任务书.md) | fall/near-fall 事件模型训练、连续视频评估、shadow 和晋级门禁；不把 provisional 候选写成正式能力 |
| 跌倒 | [模型选型矩阵](modules/fall_risk/plans/跌倒风险各任务模型调研与选型矩阵.md) | 候选模型和启用门槛，不等于已接入模型 |
| 跌倒 | [步态模型训练方案](modules/fall_risk/plans/步态模型训练方案.md) | 基于现有数据的步态二分类、多任务 TCN、跨数据源评估、基线对照和主链替换门槛 |
| 跌倒 | [步态模型训练计划：Codex 执行任务书](modules/fall_risk/plans/步态模型训练计划-Codex执行任务书.md) | 步态专项审计、test 隔离、数据重建、训练矩阵、冻结评估、shadow 和代理停止条件 |
| 跌倒 | [坐站模型训练方案](modules/fall_risk/plans/坐站模型训练方案.md) | 坐站事件定位、逐帧相位分割、动作与功能 proxy、无泄漏评估和规则 fallback 的训练方案 |
| 跌倒 | [坐站模型训练计划：Codex 执行任务书](modules/fall_risk/plans/坐站模型训练计划-Codex执行任务书.md) | 连续坐站事件标签、专项 split、因果 TCN、事件评估、一次性 test、shadow 和代理停止条件 |
| 跌倒 | [近跌倒模型训练方案](modules/fall_risk/plans/近跌倒模型训练方案.md) | 近跌倒事件监督门禁、因果 TCN、困难负样本、事件级评估和规则 fallback 的训练方案 |
| 跌倒 | [自采数据模型增强计划](modules/fall_risk/plans/自采数据模型增强计划.md) | 基于当前步态、坐站、近跌倒和跌倒事件训练进度，安全接入 SCF_MVP_V1 的优先级、实验矩阵、分区和门禁 |
| 跌倒 | [跌倒事件模型训练方案](modules/fall_risk/plans/跌倒事件模型训练方案.md) | 跌倒事件边界、因果 TCN、困难负样本、事件级评估、校准和规则安全覆盖的训练方案 |
| 跌倒 | [动作与事件标签 v3 训练版设计稿](modules/fall_risk/plans/跌倒风险事件级标签v3设计评审稿.md) | 已实现的动作/事件训练语义、subtype 门槛、hard negative、ignore 和防泄漏设计；机器契约以 v3 schema 和校验器为准 |
| 数据 | [标注 SOP](modules/fall_risk/data/数据标注SOP.md) | 标注执行和质检流程 |
| 数据 | [跌倒标签目录说明](../data/annotations/fall_risk/README.md) | v2 根标签、v3 训练标签、来源批次、split 和机器事实源 |
| 数据 | [数据集标注规范](modules/fall_risk/data/数据集标注规范.md) | 数据集到统一标注格式的映射 |
| 数据 | [数据集处理状态](modules/fall_risk/data/数据集处理状态.md) | `data/external/` 下各数据集的实际接入阶段、产物和缺口 |
| 数据 | [数据集与划分说明](modules/fall_risk/数据集与划分说明.md) | 当前正式训练数据源、各子模型训练窗口和统一 v3 split |
| 数据 | [标签字典](modules/fall_risk/data/跌倒风险标签字典.md) | 动作、事件和风险标签定义 |
| 数据 | [Windows CVAT 教程](modules/fall_risk/data/Windows本地部署CVAT标注员教程.md) | 标注员本地工具部署 |
| 审计 | [工作流 A 数据审计](../reports/fall_risk/data_audit.md) | manifest、标签来源、时间轴和评估分母的实测事实 |
| 审计 | [工作流 A 阻塞清单](../reports/fall_risk/workflow_a_blockers.md) | 已核验的人工、法律和数据阻断，以及解除证据要求 |
| 审计 | [fall-risk-data-v2 发布候选](../reports/fall_risk/fall-risk-data-v2-release-candidate.md) | 自动化验收结果与不可声明结论 |
| 复现 | [数据与 split 版本](../reports/reproducibility/dataset_and_split_versions.md) | 数据、配置、split 和合成证据包哈希 |
| 审计 | [根标签校验报告](../reports/fall_risk/label_validation_formal_v2.json) | v2 根标签的机器可读 formal 结果；当前仍有 blocker |
| 审计 | [训练标签 v3 迁移报告](../reports/fall_risk/training-labels-v3-migration.json) | 动作/事件训练标签的确定性迁移计数、去重和输入输出 hash |
| 审计 | [训练标签 v3 校验报告](../reports/fall_risk/training-labels-v3-validation.json) | v3 结构与 split 合法；fall/near-fall 事件训练门禁通过，动作类型门禁仍未通过 |
| 审计 | [跌倒事件训练 P0 审计](../reports/fall_risk/fall_event_training_audit.md) | 当前 hash、split、formal、连续背景、老人域、协议和 test 治理门禁；当前为 `infrastructure_only` |
| 审计 | [跌倒事件训练阻塞清单](../reports/fall_risk/fall_event_blockers.md) | formal、样本规模、连续分母、老人域、冻结协议和保管人缺口 |
| 审计 | [跌倒连续训练数据治理 v2](../reports/fall_risk/fall_event_continuous_governance_v2/README.md) | 当前 `splitv3_334...` 的 7,701 条显式监督、双姿态缓存覆盖、分层权重和 test 隔离 |
| 评估 | [跌倒连续因果 TCN v2](../reports/fall_risk/fall_event_continuous_tcn_v2/README.md) | 7,504 个因果窗口、三 seed validation、分层误报和训练门禁 No-Go；运行配置另有 provisional 主评分开关 |
| 复现 | [跌倒事件训练 manifest](../reports/reproducibility/fall_event_training_manifest.json) | P0 审计的机器可读输入 hash、门禁结果和执行授权边界 |
| 审计 | [坐站训练 P0 审计](../reports/fall_risk/sit_stand_training_audit.md) | 连续坐站输入 hash、专项标签/背景、四类门禁和 test 隔离；当前为 `infrastructure_only` |
| 审计 | [坐站训练阻塞清单](../reports/fall_risk/sit_stand_training_blockers.md) | 连续事件真值、显式背景、动作/相位/功能监督和冻结审批缺口 |
| 评估 | [坐站训练数据 v2 治理与开发评估](../reports/fall_risk/sit_stand_event_v2/README.md) | SCF 发布后的专项标签、物化感知 split、多 cutoff 因果窗口、TCN/规则完整流对照和主链 No-Go 结论 |
| 复现 | [坐站训练 manifest](../reports/reproducibility/sit_stand_training_manifest.json) | 坐站 P0 审计的机器可读 hash、事实、门禁和执行授权边界 |
| 跌倒 | [个体化行为基线算法增强计划](modules/fall_risk/plans/个体化行为基线算法增强计划.md) | 因果周期契约、鲁棒统计、冷启动、防污染、纵向数据、消融评估和轻量模型晋级路线 |
| 评估 | [工作流 A 合成烟测报告](../reports/fall_risk/workflow_a_synthetic_evaluation/bundle/report.md) | 事件评估 bundle 的开发链路证据；不是比赛指标或真实效果 |
| 评估 | [居家模拟动作 `fall_nearfall_v1`](../reports/fall_risk/fall_nearfall_v1/README.md) | 冻结 v1 在 115 条单人居家候选视频上的开发性跌倒/近跌倒事件指标；不是 frozen test 或老人域泛化证据 |
| 评估 | [居家数据域内适配 `fall_nearfall_adaptation_v1`](../reports/fall_risk/fall_nearfall_adaptation_v1/README.md) | 加入 11 条自采跌倒正例后的三 seed 适配模型与 44 条未见视频对照；不替换冻结 v1，不支持跨主体泛化结论 |
| 评估 | [步态 observable-context v2 TCN](../reports/fall_risk/gait_observable_context_v2/splitv3-e71a045/README.md) | 当前唯一保留的步态实验候选；validation-only，未接入主路径 |
| 评估 | [步态 TCN v5 六 seed 集成实验](../reports/fall_risk/gait_window_v5_effect_first/development-20260803/README.md) | B01-B04 functional proxy 的六 seed 训练、集成结果和 test 隔离状态 |
| 评估 | [坐站首轮 provisional 训练](../reports/fall_risk/sit_stand_event_v1/README.md) | 坐站训练前审计、clip-level E0/E2、候选双头 TCN smoke、test 隔离、预估偏差与正式事件定位阻塞 |
| 评估 | [跌倒连续因果 TCN 阻塞清单](../reports/fall_risk/fall_event_blockers.md) | 当前模型方向的连续背景、老人域、冻结协议和正式晋级门禁；临时运行开关不等于门禁解除 |
| 评估 | [近跌倒恢复确认 TCN pilot](../reports/fall_risk/near_fall_event_v1/README.md) | 当前 v3 split 的三 seed validation、窗口来源限制和 test 隔离状态；不是 onset-time 预警指标 |
| 评估 | [SCF_MVP_V1 基线回放与近跌倒增强](../reports/fall_risk/self_collected_scf_mvp_v1/README.md) | 自采隔离 candidate、262 段四分支回放、E1-E3 九次训练与 P05 challenge；三组均 No-Go，未替换主路径 |
| 评估 | [个体基线 Phase 2 状态报告](../reports/fall_risk/baseline_longitudinal_phase2.md) | 纵向 schema、前向 split、四组消融基础设施和真实空数据 blocker；不提供真实效果结论 |
| 复现 | [实时链路工程回归](../reports/fall_risk/runtime/README.md) | 阶段 0/1 的固定输入 hash、离线服务烟测与时钟/队列诊断；不是效果指标 |
| 心理健康 | [模块 README](modules/mental_health/README.md) | 日级聚合、基线、评分、离线 CLI，以及隔离徘徊数据管线的当前边界 |
| 心理健康 | [徘徊样行为识别技术方案](modules/mental_health/plans/徘徊样行为识别技术方案.md) | 徘徊轨迹分类、摄像头接入、日级基线、评估门槛和个人工作范围；属于目标设计，不等于已实现 |
| 心理健康 | [徘徊模块协作交接与职责边界](modules/mental_health/徘徊模块协作交接与职责边界.md) | 按角色分配徘徊模块上下游交付、验收标准和阻塞条件 |
| 复现 | [徘徊步骤 2 转换与复核](../reports/mental_health/wandering_step2/README.md) | WanderingPatterns/SmartCare 来源哈希、转换统计、异常、确定性证据和人工联系表入口 |
| 任务 | [当前任务](tasks/README.md) | 项目待办和验证缺口 |

## 目录约定

```text
docs/
  architecture/          当前工程架构和运行链路
  interfaces/            当前对外接口契约
  modules/               模块说明、有效计划和数据规范
  tasks/                 当前待办；完成后从这里移除
  reference/             外部资料原件
  archive/               历史评审、汇报和已结束计划

reports/                 新实验指标、复现记录和失败案例
```

## 维护规则

- 当前能力以模块 README、工程架构和代码为准，不从研发计划或归档报告反推。
- 工作流 A 的自动化状态见跌倒模块 README；真实数据门禁见审计报告和任务清单。合成评估结果只能用于基础设施烟测。
- 跌倒风险和心理健康分别评分、分别输出事件；接口文档只定义共享字段。
- 新增、删除或移动现行文档时同步更新本页。
- 状态发生变化时更新 `tasks/README.md`，不要在多个计划或汇报中维护重复待办。
- 阶段汇报、旧评审和已执行计划移入 `archive/`，不继续在原文上滚动维护。
