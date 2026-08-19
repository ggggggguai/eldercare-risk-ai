# 项目文档

更新时间：2026-08-18

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
| 跌倒 | [模块 README](modules/fall_risk/README.md) | 当前能力、命令、字段和限制 |
| 跌倒 | [协作开发指南](modules/fall_risk/guides/跌倒风险算法协作开发指南.md) | 开发约束、验证方式和代码职责 |
| 跌倒 | [研发计划](modules/fall_risk/plans/跌倒风险算法研发计划.md) | 目标路线、实验设计和阶段计划，不等于完成状态 |
| 跌倒 | [下一阶段执行计划](modules/fall_risk/plans/跌倒风险下一阶段执行计划.md) | 当前实时链路与算法语义质量冲刺：时钟与采样契约、分支可用性、会话边界、异步回调、变形回放和运行可观测性 |
| 跌倒 | [算法技术方案](modules/fall_risk/plans/跌倒风险算法技术方案.md) | 分层算法链路、技术选型、模块输入输出、训练评估门槛和研发阶段路线 |
| 跌倒 | [挑战杯冲奖增强计划](modules/fall_risk/plans/挑战杯揭榜挂帅冲奖增强计划.md) | 官方评分映射、七周执行计划、验收门槛和提交证据 |
| 跌倒 | [环境因素多模态增强设计方案](modules/fall_risk/plans/环境因素多模态增强设计方案.md) | 环境增强的目标架构、输入契约、时间对齐、特征计算、融合边界和模块职责 |
| 跌倒 | [环境因素多模态增强实施计划](modules/fall_risk/plans/环境因素多模态增强实施计划.md) | ESP32-S3、照度与水浸传感接入，人-环境交互特征、规则融合、受控实验和验收边界 |
| 跌倒 | [工作流 A Codex 执行任务书](modules/fall_risk/plans/工作流A-Codex执行任务书.md) | 数据、标注与评估底座的代理执行范围、阶段门槛和验收条件 |
| 跌倒 | [工作流 B Codex 执行任务书](modules/fall_risk/plans/工作流B-Codex执行任务书.md) | 算法增强、实验矩阵、数据门禁和冻结交接；不作为已实现或实测结果证明 |
| 跌倒 | [模型选型矩阵](modules/fall_risk/plans/跌倒风险各任务模型调研与选型矩阵.md) | 候选模型和启用门槛，不等于已接入模型 |
| 跌倒 | [步态模型训练方案](modules/fall_risk/plans/步态模型训练方案.md) | 基于现有数据的步态二分类、多任务 TCN、跨数据源评估、基线对照和主链替换门槛 |
| 跌倒 | [坐站模型训练方案](modules/fall_risk/plans/坐站模型训练方案.md) | 坐站事件定位、逐帧相位分割、动作与功能 proxy、无泄漏评估和规则 fallback 的训练方案 |
| 跌倒 | [近跌倒模型训练方案](modules/fall_risk/plans/近跌倒模型训练方案.md) | 近跌倒事件监督门禁、因果 TCN、困难负样本、事件级评估和规则 fallback 的训练方案 |
| 跌倒 | [跌倒事件模型训练方案](modules/fall_risk/plans/跌倒事件模型训练方案.md) | 跌倒事件边界、因果 TCN、困难负样本、事件级评估、校准和规则安全覆盖的训练方案 |
| 跌倒 | [动作与事件标签 v3 训练版设计稿](modules/fall_risk/plans/跌倒风险事件级标签v3设计评审稿.md) | 已实现的动作/事件训练语义、subtype 门槛、hard negative、ignore 和防泄漏设计；机器契约以 v3 schema 和校验器为准 |
| 跌倒 | [KINECAL 骨架数据下载设计](modules/fall_risk/plans/KINECAL骨架数据下载设计.md) | KINECAL 风险组骨架子集的下载范围、数据治理和验证标准 |
| 跌倒 | [KINECAL 骨架数据下载实施计划](modules/fall_risk/plans/KINECAL骨架数据下载实施计划.md) | KINECAL 下载器、测试、文档和正式下载的执行步骤 |
| 数据 | [标注 SOP](modules/fall_risk/data/数据标注SOP.md) | 标注执行和质检流程 |
| 数据 | [跌倒标签目录说明](../data/annotations/fall_risk/README.md) | v2 根标签、v3 训练标签、来源批次、split 和机器事实源 |
| 数据 | [数据集标注规范](modules/fall_risk/data/数据集标注规范.md) | 数据集到统一标注格式的映射 |
| 数据 | [数据集处理状态](modules/fall_risk/data/数据集处理状态.md) | `data/external/` 下各数据集的实际接入阶段、产物和缺口 |
| 数据 | [标签字典](modules/fall_risk/data/跌倒风险标签字典.md) | 动作、事件和风险标签定义 |
| 数据 | [Windows CVAT 教程](modules/fall_risk/data/Windows本地部署CVAT标注员教程.md) | 标注员本地工具部署 |
| 审计 | [工作流 A 数据审计](../reports/fall_risk/data_audit.md) | manifest、标签来源、时间轴和评估分母的实测事实 |
| 审计 | [工作流 A 阻塞清单](../reports/fall_risk/workflow_a_blockers.md) | 已核验的人工、法律和数据阻断，以及解除证据要求 |
| 审计 | [fall-risk-data-v2 发布候选](../reports/fall_risk/fall-risk-data-v2-release-candidate.md) | 自动化验收结果与不可声明结论 |
| 复现 | [数据与 split 版本](../reports/reproducibility/dataset_and_split_versions.md) | 数据、配置、split 和合成证据包哈希 |
| 审计 | [根标签校验报告](../reports/fall_risk/label_validation_formal_v2.json) | v2 根标签的机器可读 formal 结果；当前仍有 blocker |
| 审计 | [训练标签 v3 迁移报告](../reports/fall_risk/training-labels-v3-migration.json) | 动作/事件训练标签的确定性迁移计数、去重和输入输出 hash |
| 审计 | [训练标签 v3 校验报告](../reports/fall_risk/training-labels-v3-validation.json) | v3 结构与 split 合法；fall/near-fall 事件训练门禁通过，动作类型门禁仍未通过 |
| 评估 | [工作流 A 合成烟测报告](../reports/fall_risk/workflow_a_synthetic_evaluation/bundle/report.md) | 事件评估 bundle 的开发链路证据；不是比赛指标或真实效果 |
| 评估 | [KINECAL 轻量步态 TCN baseline](../reports/fall_risk/kinecal_gait_tcn/README.md) | 14 点时序模型的固定划分实测、失败结论和复现方式 |
| 评估 | [步态 TCN v5 六 seed 集成实验](../reports/fall_risk/gait_window_v5_effect_first/development-20260803/README.md) | B01-B04 functional proxy 的六 seed 训练、集成结果和 test 隔离状态 |
| 评估 | [坐站首轮 provisional 训练](../reports/fall_risk/sit_stand_event_v1/README.md) | 坐站训练前审计、clip-level E0/E2、候选双头 TCN smoke、test 隔离、预估偏差与正式事件定位阻塞 |
| 评估 | [跌倒 candidate-clip TCN v3 pilot](../reports/fall_risk/fall_event_proxy_v2_v3split/README.md) | 当前 v3 split 的三 seed validation、分域结果和 test 隔离状态；不是连续跌倒定位指标 |
| 评估 | [近跌倒恢复确认 TCN pilot](../reports/fall_risk/near_fall_event_v1/README.md) | 当前 v3 split 的三 seed validation、窗口来源限制和 test 隔离状态；不是 onset-time 预警指标 |
| 复现 | [实时链路工程回归](../reports/fall_risk/runtime/README.md) | 阶段 0/1 的固定输入 hash、离线服务烟测与时钟/队列诊断；不是效果指标 |
| 心理健康 | [模块 README](modules/mental_health/README.md) | 日级聚合、基线、评分、离线 CLI，以及隔离徘徊数据管线的当前边界 |
| 心理健康 | [徘徊识别技术文档2](modules/mental_health/plans/徘徊识别技术文档2.md) | 现行五日技术路线：B01+B02 联合 development、automatic episode、camera shape、多模态 context、真实日报、滚动个人基线和算法对接契约 |
| 心理健康 | [M0-CAM-5D 快速交付总任务书](modules/mental_health/plans/M0-CAM-5D快速交付总任务书.md) | 当前唯一执行入口；五天完成 camera MP4 到 episode/context/daily/baseline/handoff 的算法闭环，后端和前端不在本轮 |
| 心理健康 | [W5D-T0 交付契约](modules/mental_health/plans/M0-CAM-5D-T0交付契约与运行基线.md) | 固定 B01+B02 development index、candidate/runtime identity、输出 schema 和 fresh/non-overwrite 规则 |
| 心理健康 | [徘徊五日算法交付契约](interfaces/徘徊五日算法交付契约.md) | episode/context/daily/baseline/handoff 文件名、统一状态、source reference、identity 和 JSON Schema 入口；不等于 AlgorithmEvent |
| 心理健康 | [W5D-T1 分段器联合优化](modules/mental_health/plans/M0-CAM-5D-T1-B01B02分段器联合优化.md) | B01+B02 可反复参数搜索，recall-first，支持 0.25 秒局部边界 refinement、失败清单和唯一 profile |
| 心理健康 | [W5D-T2 自动 Episode+Shape](modules/mental_health/plans/M0-CAM-5D-T2自动Episode与轨迹识别闭环.md) | tracking/sidecar 到一 proposal 一 result；binary 为主，四类诊断，必要时才启用轻量 camera 模型调整 |
| 心理健康 | [W5D-T3 居家视频与 Context](modules/mental_health/plans/M0-CAM-5D-T3居家视频与多模态上下文.md) | 先以 B01+B02/fixture 完成三帧多模态 context core；居家 MP4 到位后补 truth-free smoke，视频或标签暂缺不阻塞后续 |
| 心理健康 | [W5D-T4 真实日报与个人基线](modules/mental_health/plans/M0-CAM-5D-T4真实日报与个人基线.md) | 真实 person/session/timezone/presence adapter，日级聚合与 3/7/14 日 baseline readiness/deviation |
| 心理健康 | [W5D-T5 算法对接与 E2E](modules/mental_health/plans/M0-CAM-5D-T5算法对接包与整体验收.md) | 交付 episode/context/daily/baseline JSONL、manifest、运行命令和失败降级；不实现后端/前端 |
| 心理健康 | [M0-CAM-EP1B 执行任务书](modules/mental_health/plans/M0-CAM-EP1B执行任务书.md) | 历史执行记录：B01/B02 manual-CVAT oracle-boundary descriptive pilot；其旧独立证据门已被 W5D 取代 |
| 心理健康 | [M0-CAM-EP2A-S0 执行任务书](modules/mental_health/plans/M0-CAM-EP2A-S0执行任务书.md) | 历史实现契约：continuous episode boundary proposal；当前分段调优见 W5D-T1 |
| 心理健康 | [M0-CAM-EP2A-S1A 执行任务书](modules/mental_health/plans/M0-CAM-EP2A-S1A执行任务书.md) | 历史 evaluator 契约：matching、boundary metrics 和 failures；当前用于 B01+B02 development |
| 心理健康 | [M0-CAM-EP2A-S2A 执行任务书](modules/mental_health/plans/M0-CAM-EP2A-S2A执行任务书.md) | 历史 proposal-shape v2 契约；当前 automatic episode+shape 见 W5D-T2 |
| 心理健康 | [M0-CAM-EP2A-S2B 执行任务书](modules/mental_health/plans/M0-CAM-EP2A-S2B执行任务书.md) | 历史 review-pack 契约；当前多模态上下文见 W5D-T3 |
| 心理健康 | [M0-CAM EP1B / EP2A 历史交接](modules/mental_health/plans/M0-CAM-EP1B-EP2A-S1B人工证据交接.md) | 已被 W5D 取代；只保留旧 B01/B02 cohort、holdout 和 v2 指标追溯 |
| 心理健康 | [萤石 C6C 办公室视频拍摄与标注手册](modules/mental_health/拍摄与标注/萤石C6C办公室视频拍摄与标注手册.md) | 面向拍摄人员的完整 episode 优先、自然活动长片、legacy 40 秒诊断、质量检查和命名指南 |
| 心理健康 | [Windows 本地部署 CVAT 徘徊识别标注员教程](modules/mental_health/拍摄与标注/Windows本地部署CVAT徘徊识别标注员教程.md) | 面向标注同学的本地 CVAT 部署、真实 episode Rectangle Track、轨迹与目的分离、导出和交接指南 |
| 心理健康 | [M0-CAM-PORTABLE 执行任务书](modules/mental_health/plans/M0-CAM-PORTABLE执行任务书.md) | PORTABLE 首次实现的固定资产、离线恢复、零视频 preflight 与隔离验收协议；首次实现已结束，formal 仍 blocked |
| 心理健康 | [M0-CAM-PORTABLE-F1 执行任务书](modules/mental_health/plans/M0-CAM-PORTABLE-F1执行任务书.md) | F1 原执行设计与历史测试证据；当前作为跨机部署/正式发布前 `rework_deferred` 债务，不阻塞本机 episode-first 开发 |
| 心理健康 | [M0-CAM-MVP-1 执行任务书](modules/mental_health/plans/M0-CAM-MVP-1执行任务书.md) | session-level WanderingEvidence 实现协议；主功能与 F1 exact-schema 接受门均已完成 |
| 心理健康 | [M0-CAM-MVP-1-F1 执行任务书](modules/mental_health/plans/M0-CAM-MVP-1-F1执行任务书.md) | 已完成 primary/product exact-schema 和非空风险/事件字段边界收口；未展开日级风险 |
| 心理健康 | [M0-CAM-MVP-2S 执行任务书](modules/mental_health/plans/M0-CAM-MVP-2S执行任务书.md) | 已完成：用已验收 synthetic session bundle 和显式 binding 构建 exact-schema 日级摘要；不接真实 identity、基线、风险或事件 |
| 心理健康 | [M0-CAM-MVP-3S 执行任务书](modules/mental_health/plans/M0-CAM-MVP-3S执行任务书.md) | 主体与 F1 接受门已完成；v2 profile 携带可独立重算的 reference-day evidence，现行接受状态为 `passed` |
| 心理健康 | [M0-CAM-MVP-3S-F1 执行任务书](modules/mental_health/plans/M0-CAM-MVP-3S-F1执行任务书.md) | 已完成：真实校验冻结 producer prefix，并由 public loader 重算 v2 window/readiness/reference stats |
| 心理健康 | [M0-CAM-MVP-3D 执行任务书](modules/mental_health/plans/M0-CAM-MVP-3D执行任务书.md) | 已完成：用严格早于 observation day 的 synthetic profile 生成九项 signed deviation preview；接受状态 passed，不输出风险或事件 |
| 心理健康 | [徘徊模块协作交接与职责边界](modules/mental_health/徘徊模块协作交接与职责边界.md) | 算法侧 tracking/episode/shape/context/daily/baseline/handoff 职责；后端和前端后续接入 |
| 复现 | [徘徊步骤 2 转换与复核](../reports/mental_health/wandering_step2/README.md) | WanderingPatterns/SmartCare 来源哈希、转换统计、异常、确定性证据和人工联系表入口 |
| 复现 | [徘徊步骤 3 固定 split](../reports/mental_health/wandering_step3/README.md) | 1,810 条来源专用固定分配、official 封存、重复/近邻审计、五产物和完整哈希 |
| 复现 | [徘徊步骤 4 预处理与可视化](../reports/mental_health/wandering_step4/README.md) | 已完成：1,790 条非 sealed 留痕、1,775 ready/15 unavailable、14 通道、train-only 统计、确定性 bundle 和 `human_review_passed` 诊断图审 |
| 评估 | [徘徊步骤 5 RF 对照基线](../reports/mental_health/wandering_step5/README.md) | 已完成并独立复建复核：strict bundle loader、固定 26 维特征、二/四分类五 seed RF、development/frozen WP test 信任链、确定性与分来源真实指标；仅为 `comparison_only` |
| 评估 | [徘徊步骤 6 纯 TCN 对照基线](../reports/mental_health/wandering_step6/README.md) | 已完成：two-task 独立 CPU TCN、10 个安全 NPZ、validation 早停、固定 WP test、RF 同 seed 差值、development/test 字节级复建和 batch=1 基准；仅为 `comparison_only` |
| 复现 | [徘徊步骤 7 bbox-only 离线链](../reports/mental_health/wandering_step7/README.md) | 已完成：严格 media/tracking 契约、Camera QC、高度补偿、步骤 4 共享核心、RF/TCN 四组独立预测、合成 camera contract 与双构建确定性；未做真实摄像头效果验证 |
| 评估 | [徘徊步骤 8 合成污染兼容性](../reports/mental_health/wandering_step8/visual_review/v3/README.md) | 已完成：v3 图审与兼容性证据链关闭，正式结论为 `model_compatibility_warning`，不是“全部模型通过观察线”；详细 JSON/JSONL 按 `.gitignore` 本地保留 |
| 复现 | [徘徊步骤 9/9a TopoWander-MPT 前向契约](../reports/mental_health/wandering_step9/README.md) | 第 9a.6 节修正、三条路径级 LF 与唯一 Git 检查点已形成；步骤 9 本身是纯前向契约 |
| 复现 | [徘徊步骤 10 runtime 前置治理](../reports/mental_health/wandering_step10_runtime/README.md) | Step10-A 的历史 runtime/source 治理记录；不再作为常规模型性能开发门禁 |
| 评估 | [TopoWander-MPT M0-S 三种子稳定性](../reports/mental_health/wandering_performance/m0s_three_seed_stability_v1/README.md) | 固定三 seed development validation、异常中断恢复、fresh reload、同种子 RF/TCN paired 对照、成本与 primary-seed 冻结；后续固定候选计分见 M0-RS 报告 |
| 复现 | [TopoWander-MPT M0-R Release-Prep v2](../reports/mental_health/wandering_performance/m0r_release_prep_v1/README.md) | M0-S primary 的 local-only inference bundle、外部 manifest SHA、phase-aware evaluator 与 validation parity；没有执行当前候选 WP public-holdout 计分 |
| 复现 | [TopoWander-MPT M0-RH score-entry hardening v3](../reports/mental_health/wandering_performance/m0r_score_entry_hardening_v1/README.md) | 唯一 `score-frozen-wp` 正式入口、固定 CPU runtime、source/archive/双身份 preflight、validation parity 与拒绝覆盖原子提交；作为 M0-RS 的冻结前置证据 |
| 评估 | [TopoWander-MPT M0-RS fixed WP public-holdout score](../reports/mental_health/wandering_performance/m0rs_public_holdout_score_report_v1/README.md) | 固定 primary 候选的一次 public-shape 计分、六文件原子输出、独立复算与预注册门槛判定；`target_met`，不是首次盲测或 camera/老人域证据 |
| 复现 | [TopoWander-MPT M0-CAM-E / M0-CAM-H primary camera 工程闭环](../reports/mental_health/wandering_m0cam_engineering_v1/README.md) | 固定 primary 接入 Step7 adapter/QC/prepared arrays，并完成 synthetic-only、二分类平票与 active-source 三项加固；仅为 `synthetic_contract_only`，当前数据工具与授权评估见任务表 |
| 复现 | [TopoWander-MPT M0-CAM-EP1A oracle-boundary episode inference](../reports/mental_health/wandering_camera_episode_v1/README.md) | truth-separated CVAT/whole-clip boundary、episode 内 QC、80 点 frozen forward，以及 D01 12 条和 S00 三段正式入口回归；不是 automatic boundary 或 camera accuracy 证据 |
| 复现 | [W5D-00 delivery contract bundle](../reports/mental_health/wandering_camera_5d_contract_v3/output_contract.md) | 48 条 B01+B02 development index、owner-confirmed legacy setup alias、fixed primary CPU load、fresh/non-overwrite 与九份 schema 的机器产物入口 |
| 评估 | [W5D-01 B01+B02 segmenter search](../reports/mental_health/wandering_camera_segmenter_search_v2/README.md) | 两轮共 29 个 full-cohort development 候选后的 recall-first fallback、B01/B02/pooled 指标、top-3 确定性复放、硬断审计和完整失败留存；recall/F1 未达门 |
| 复现 | [W5D-02 automatic episode-to-shape pipeline](../reports/mental_health/wandering_camera_episode_pipeline_w5d02_v4/README.md) | 48 条 B01+B02 development 视频、129 条一一对应 result、proposal/QC/80 点/shape 自动闭环；coverage 0.897436 通过最低门但未达 0.90，binary 主结果、oracle/automatic miss 分开，fixed primary/0.5 保留 |
| 复现 | [W5D-03A three-frame context core](../reports/mental_health/wandering_camera_context_core_w5d03a_v2/README.md) | B01+B02 的 82/82 eligible context rows、246/246 起中末 scene/crop 时点和 492 个 Git 外 JPEG refs；fake 只验证 ready contract，不是 context accuracy 或居家验证 |
| 复现 | [W5D-03A disabled degradation](../reports/mental_health/wandering_camera_context_core_w5d03a_disabled_v2/README.md) | B01/B02 两个登记视频的 3/3 trigger 在 provider disabled 时成功输出 `unknown/unavailable/provider_disabled`，不阻塞主链 |
| 复现 | [W5D-04 B01+B02 real-development daily/baseline](../reports/mental_health/wandering_camera_daily_baseline_w5d04_v1/README.md) | 48 个显式 video binding、129 条 episode 汇总为 P-OFFICE-01 的 2 个自然日；presence/coverage/四态/context 守恒，2/2 profile 保持 warming-up，capture clock 缺失显式标记 declared schedule |
| 复现 | [W5D-04 15-day readiness replay](../reports/mental_health/wandering_camera_daily_baseline_w5d04_replay_v1/README.md) | deterministic replay 验证 3 个 warming-up、4 个 initial-ready、8 个 stable-ready 和最新 14 usable day 窗口；不是实际长期观察 |
| 复现 | [W5D-05 B01+B02 algorithm handoff v2](../reports/mental_health/wandering_camera_handoff_w5d05_v2/README.md) | 严格 8 文件、129/82/2/2/2、manifest/run-summary v2、跨阶段 identity/hash/count/duration 校验、public loader 和单命令 cached-tracking E2E；状态为 home-smoke pending，不是 home-validated |
| 复现 | [W5D-05 disabled degradation v2](../reports/mental_health/wandering_camera_handoff_w5d05_disabled_v2/README.md) | provider disabled 时 3 条 context 保留 `unknown/unavailable`，同时继续交付 129 条 episode 和完整 2/2/2 daily/profile/deviation；不把 provider 缺席升级为算法 run error |
| 复现 | [TopoWander-MPT M0-CAM-EP1B oracle-boundary batch evaluator](../reports/mental_health/wandering_camera_episode_eval_v1/README.md) | evaluator 实现与早期 AI-preannotation provisional diagnostic；正式 B01/B02 manual-CVAT 成绩见后续 development/holdout 报告 |
| 复现 | [M0-CAM-EP2A-S0 boundary proposal implementation](../reports/mental_health/wandering_camera_episode_boundary_proposal_v1/README.md) | 独立 proposal schema、五态状态机、Camera QC hard-break 复用、同 bucket hard-break/慢漂移审计修复、H02/H03 truth-free smoke；不是 accepted boundary 或 automatic-boundary 性能 |
| 复现 | [M0-CAM-EP2A-S1A boundary evaluator implementation](../reports/mental_health/wandering_camera_episode_boundary_eval_v1/README.md) | 只读 proposal/human-boundary loader、完整 scope binding、共享 matching、双视图 metrics/coverage/diagnostics/failures 和 deterministic synthetic CLI；不是人工或 automatic-boundary 性能 |
| 复现 | [M0-CAM-EP2A-S2A proposal-shape bridge](../reports/mental_health/wandering_camera_episode_proposal_inference_v1/README.md) | 原始 proposal endpoint 到 EP1A 同数学 QC/80 点预处理/frozen forward 的薄 bridge；旧 proposed-only smoke 保留为历史，当前 v2 为 proposed+uncertain forward、rejected skip，仍不生成 accepted boundary |
| 复现 | [M0-CAM-EP2A-S2A proposal-shape 验证记录](../reports/mental_health/wandering_camera_episode_proposal_inference_v1/VERIFICATION.md) | 初始 synthetic/H02/H03 历史 smoke 与 2026-08-17 candidate-inclusive v2 聚焦回归记录 |
| 复现 | [M0-CAM-EP2A-S2B proposal-shape review pack](../reports/mental_health/wandering_camera_episode_proposal_review_v1/README.md) | proposal/prediction/tracking/video identity 的只读连接、逐 proposal SVG、空白人工模板与 H02/H03 四条 uncertain review smoke；不是 truth 或性能证据 |
| 评估 | [M0-CAM B01/B02 intake 与 cohort freeze](../reports/mental_health/wandering_camera_b01_b02_intake_v1/README.md) | 50 条视频/80 条 manual CVAT track 的只读审计、owner identity/blinding 确认、track 56 裁决、B01 development 与 B02 frozen-video holdout 固化 |
| 评估 | [M0-CAM B01 development / B02 historical frozen-video holdout](../reports/mental_health/wandering_camera_b01_development_v1/README.md) | EP1B manual-CVAT oracle pilot、S1B policy freeze/旧 proposed-only 一次性 holdout；另含 v2 post-hoc addendum，不是跨人/跨机位证据 |
| 评估 | [M0-CAM B02 candidate-inclusive development](../reports/mental_health/wandering_camera_b02_candidate_inclusive_development_v1/README.md) | 负责人授权复用 B02 后的 post-hoc exploratory v2：17/18 ready、binary all-eligible accuracy/macro-F1=`0.944444/0.970588`；不是 held-out，uncertain 未变成 accepted boundary |
| 复现 | [TopoWander-MPT M0-CAM-RD camera development 数据工具](../reports/mental_health/wandering_camera_data_readiness_v1/README.md) | 初版 240 条 synthetic observation、schema/readiness 与历史工具骨架；制品保持不变，真实 readiness 仍为 `not_ready` |
| 复现 | [TopoWander-MPT M0-CAM-RD-F / RD-F2 development entry hardening](../reports/mental_health/wandering_camera_rd_f_v1/README.md) | receipt-first、C3、session-aware person-hours、evaluator、最大基数 matching 和 authorized API provenance 已收口；证据仍为 `synthetic_schema_contract_only`、`readiness_status=not_ready`，当前工作见任务表 |
| 复现 | [TopoWander-MPT M0-CAM-C01-PREP receipt-first C1 handoff](../reports/mental_health/wandering_camera_c01_prep_v1/README.md) | canonical tracking/sidecar round-trip、receipt-first 视频→共享 tracker→C1 pair 主体及 owner 清单；原定初版缺口由 C01-F 处理，后续完整关闭见任务表中的 F2 |
| 复现 | [TopoWander-MPT M0-CAM-C01-F provenance and final-binding hardening](../reports/mental_health/wandering_camera_c01_f_v1/README.md) | C01-F 原定公共 context、原子 safe binding、portable metadata 与视频 SHA 范围已完成；后续 active-root/resolver/source/component 审计已由 F2 收口，仅为 `synthetic_schema_contract_only`，实时状态见任务表 |
| 复现 | [TopoWander-MPT M0-CAM-C01-F2 handoff integrity hardening](../reports/mental_health/wandering_camera_c01_f2_v1/README.md) | 正式 collection/receipt 跨机交接的历史加固证据；本机 episode-first 小样开发不再以继续扩展该审计链为前置 |
| 复现 | [TopoWander-MPT M0-CAM-PORTABLE runtime assets and zero-video preflight](../reports/mental_health/wandering_camera_portable_v1/README.md) | 首次实现与 F1 执行时测试快照；当前接受状态由任务表和后续复审决定，formal 仍 blocked，不是 camera 性能证据 |
| 复现 | [TopoWander-MPT M0-CAM-MVP-1 session evidence prototype](../reports/mental_health/wandering_camera_mvp1_v1/README.md) | 首次实现与 F1 synthetic 验证记录；当前接受状态为 passed，但不输出风险或 AlgorithmEvent |
| 审计 | [M0-CAM-MVP-1 复审与 F1 关闭记录](../reports/mental_health/wandering_camera_mvp1_v1/MVP1_REAUDIT_20260814.md) | exact-schema/decision-boundary P1 复现、F1 修复证据与接受状态恢复 |
| 复现 | [M0-CAM-MVP-2S synthetic 日级摘要](../reports/mental_health/wandering_camera_mvp2s_v1/README.md) | 显式 synthetic binding、epoch/IANA 自然日、presence/status coverage、episode 日级切分与空 baseline/risk/event 边界 |
| 复现 | [M0-CAM-MVP-3S synthetic 个人基线预览](../reports/mental_health/wandering_camera_mvp3s_v1/README.md) | v2 builder、reference-day readback 与 F1 自动化/synthetic 收据；接受状态 passed |
| 审计 | [M0-CAM-MVP-3S 独立复审与关闭记录](../reports/mental_health/wandering_camera_mvp3s_v1/MVP3S_REAUDIT_20260814.md) | producer prefix identity 与 rehashed reference stats 两个假通过的复现、影响和 F1 关闭证据 |
| 复现 | [M0-CAM-MVP-3D synthetic baseline deviation preview](../reports/mental_health/wandering_camera_mvp3d_v1/README.md) | strictly-prior-day、九项 signed delta、exact-schema loader 与 Git 外 synthetic 攻击收据；接受状态 passed |
| 历史 | [徘徊样行为识别旧方案](../文档/徘徊样行为识别技术方案（历史归档-2026-08-12）.md) | 截至 2026-08-12 的旧步骤、报告和精确协议，仅供追溯，不代表现行路线 |
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
文档/                    中文参考资料和按项目要求保留的历史原文；不直接作为现行事实源
```

## 维护规则

- 当前能力以模块 README、工程架构和代码为准，不从研发计划或归档报告反推。
- 工作流 A 的自动化状态见跌倒模块 README；真实数据门禁见审计报告和任务清单。合成评估结果只能用于基础设施烟测。
- 跌倒风险和心理健康分别评分、分别输出事件；接口文档只定义共享字段。
- 新增、删除或移动现行文档时同步更新本页。
- 状态发生变化时更新 `tasks/README.md`，不要在多个计划或汇报中维护重复待办。
- 阶段汇报、旧评审和已执行计划移入 `archive/`，不继续在原文上滚动维护。
- 根目录 `文档/` 中的历史原文只有在本页明确标为“现行”时才可作为路线；本次旧徘徊方案仅供追溯。
