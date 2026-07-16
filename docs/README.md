# 项目文档

更新时间：2026-07-15

本页是文档唯一总入口。现行文档描述当前代码和接口；计划文档描述目标；归档文档只保留历史上下文，不能作为当前实现依据。

## 推荐阅读顺序

1. [项目 README](../README.md)：范围、环境和服务启动方式。
2. [算法工程架构](architecture/算法工程骨架.md)：代码分层、两条算法链和当前实现状态。
3. [算法事件输出接口](interfaces/算法事件输出接口.md)：两个模块独立输出的 `AlgorithmEvent` 契约。
4. [跌倒风险模块](modules/fall_risk/README.md)或[心理健康模块](modules/mental_health/README.md)：模块入口、算法规则和运行方式。
5. [当前任务](tasks/README.md)：尚未完成、需要验证或需要校准的工作。

## 现行文档

| 类别 | 文档 | 用途 |
|---|---|---|
| 架构 | [算法工程架构](architecture/算法工程骨架.md) | 工程边界、代码分层和实现状态 |
| 架构 | [实时视频监测链路](architecture/实时视频监测前后端算法链路.md) | 跌倒直播服务与业务后端的数据流 |
| 接口 | [算法事件输出接口](interfaces/算法事件输出接口.md) | 通用事件字段和风险等级编码 |
| 接口 | [跌倒风险服务对接说明](interfaces/跌倒风险算法服务后端对接说明.md) | HTTP 会话、鉴权和回调契约 |
| 跌倒 | [模块 README](modules/fall_risk/README.md) | 当前能力、命令、字段和限制 |
| 跌倒 | [协作开发指南](modules/fall_risk/guides/跌倒风险算法协作开发指南.md) | 开发约束、验证方式和代码职责 |
| 跌倒 | [研发计划](modules/fall_risk/plans/跌倒风险算法研发计划.md) | 目标路线、实验设计和阶段计划，不等于完成状态 |
| 跌倒 | [挑战杯冲奖增强计划](modules/fall_risk/plans/挑战杯揭榜挂帅冲奖增强计划.md) | 官方评分映射、七周执行计划、验收门槛和提交证据 |
| 跌倒 | [工作流 A Codex 执行任务书](modules/fall_risk/plans/工作流A-Codex执行任务书.md) | 数据、标注与评估底座的代理执行范围、阶段门槛和验收条件 |
| 跌倒 | [工作流 B Codex 执行任务书](modules/fall_risk/plans/工作流B-Codex执行任务书.md) | 算法增强、实验矩阵、数据门禁和冻结交接；不作为已实现或实测结果证明 |
| 跌倒 | [模型选型矩阵](modules/fall_risk/plans/跌倒风险各任务模型调研与选型矩阵.md) | 候选模型和启用门槛，不等于已接入模型 |
| 数据 | [标注 SOP](modules/fall_risk/data/数据标注SOP.md) | 标注执行和质检流程 |
| 数据 | [数据集标注规范](modules/fall_risk/data/数据集标注规范.md) | 数据集到统一标注格式的映射 |
| 数据 | [标签字典](modules/fall_risk/data/跌倒风险标签字典.md) | 动作、事件和风险标签定义 |
| 数据 | [Windows CVAT 教程](modules/fall_risk/data/Windows本地部署CVAT标注员教程.md) | 标注员本地工具部署 |
| 审计 | [工作流 A 数据审计](../reports/fall_risk/data_audit.md) | manifest、标签、许可、时间轴和评估分母的实测事实 |
| 审计 | [工作流 A 阻塞清单](../reports/fall_risk/workflow_a_blockers.md) | 已核验的人工、法律和数据阻断，以及解除证据要求 |
| 审计 | [fall-risk-data-v1 发布候选](../reports/fall_risk/fall-risk-data-v1-release-candidate.md) | 自动化验收结果与不可声明结论 |
| 复现 | [数据与 split 版本](../reports/reproducibility/dataset_and_split_versions.md) | 数据、配置、split 和合成证据包哈希 |
| 审计 | [根标签校验报告](../reports/fall_risk/label_validation_audit.json) | 当前真实根标签的机器可读 audit 结果；未通过 formal 门禁 |
| 评估 | [工作流 A 合成烟测报告](../reports/fall_risk/workflow_a_synthetic_evaluation/bundle/report.md) | 事件评估 bundle 的开发链路证据；不是比赛指标或真实效果 |
| 心理健康 | [模块 README](modules/mental_health/README.md) | 日级聚合、基线、评分和离线 CLI |
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
