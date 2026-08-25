# 当前任务

更新时间：2026-08-22

本文件只记录尚未完成的工作。已经落地的能力写入模块 README；阶段结论和旧待办移入 `docs/archive/`。当前跌倒风险模块处于模型化增强阶段：规则 baseline 仍作为对照和安全 fallback，新增时序模型必须经过数据、split、评估和部署门禁后才能替换主路径。

## P0：形成可验证交付

工作流 A 已实现统一 manifest、v2 标注导入/发布、模型训练标签 v3 迁移/统一 split/校验、四任务 split builder 和事件评估器，并用合成数据跑通 bundle。项目负责人于 2026-08-04 裁决现有规范动作标签可用于任务级正负样本：当前 v3 有 9,314 条动作、9,498 条事件和 18,812 条 split assignment，fall/near-fall hard negative 类型覆盖完整，962 条 C03 已转为 near-fall positive，446 条当前 NTU 全片跌倒按首帧/尾帧生成精确边界；校验 `valid=true`、事件任务 `training_ready=true`。fall 和 near-fall 已在锁定 test 的前提下完成三 seed train/validation 开发实验。当前剩余工作是处理 v2 formal blocker、冻结事件标签与评估协议、补充连续背景/老人域/跨来源证据并建立一次性 test 发布流程；动作类型任务因稀有类别分区覆盖不足仍为 `training_ready=false`。

### 徘徊当前状态：居家分段与分类修复已完成（development）

`current_code_task=none`，`current_data_task=none`，`current_delivery_status=home_camera_repair_development_completed`，`next_human_task=adjudicate wand_H02_dining_1_exercise / CVAT track 100 evaluation_role`。负责人批准的 0.70 可信轨迹门槛、分段召回与连续性、统一送模门禁和 pacing subtype 修复均已完成；逐项状态见[居家视频分段与分类修复任务台账](徘徊居家视频分段与分类修复.md)，数值和失败边界见[修复报告](../../reports/mental_health/wandering_camera_home_repair_v1/README.md)。CVAT rectangle 与 `evaluation_role` 只作时间段/行为语义，不作为检测或跟踪真值。

人工标注 project 导入已实现累计 task frame offset 归一化、70 视频/sidecar/tracking 身份校验和单主体时序 Track ID 派生。负责人于 2026-08-22 裁决 `wand_H02_dining_2_sweep` 的 task 1657 / CVAT track 108：保留 `observable_pattern=direct`，将 `evaluation_role` 改为 `ordinary_negative`；原 XML 以 SHA-256 绑定备份保存。修复前 v3 automatic-boundary 主视图 precision/recall/F1=`0.566038/0.666667/0.612245`，oracle pacing recall=0。修复后 parent precision/recall/F1=`0.802920/0.814815/0.808824`，resolved boundary recall/F1=`0.938931/0.656000`，全链 direct/pacing recall=`0.933962/0.888889`；random recall=`0.25` 仍是主要短板。结果只代表同一 participant 的 development。历史审计见 [W5D-03B](../../reports/mental_health/wandering_camera_home_smoke_w5d03b_v1/README.md)，修复证据见[居家修复报告](../../reports/mental_health/wandering_camera_home_repair_v1/README.md)。

W5D-00 至 W5D-05 的既有核心仍已完成：contract、segmenter fallback、automatic episode→shape、context core、真实 development 日报/个人基线、严格 8 文件 handoff 和单命令 cached-tracking E2E。正式包原样包含 129 条 episode、82 条 context、2 条 daily、2 条 profile 和 2 条 deviation；95 条可用 shape、34 条 unavailable、0 条 error、19 条 wandering-like 和 617.8 秒 duration 守恒。两个真实 profile/deviation 均为 `warming_up`；15 日 replay 只验证 3/7/14 readiness 和最新 14 日窗口。现行任务书为 [M0-CAM-5D 快速交付总任务书](../modules/mental_health/plans/M0-CAM-5D快速交付总任务书.md)。

负责人最新决定：B01 和 B02 统一作为可反复使用的 camera development 数据，可持续用于分段器、边界 refinement、送模策略、camera 预处理、binary 识别和必要的轻量模型调整。旧 B02 holdout/v1/v2 结果只保留为历史快照，不再阻止当前开发。W5D-03A 已用 B01+B02/fixture 完成 input-independent context core；新到位的居家 MP4 已补完 W5D-03B truth-free smoke，独立人工标注已按“仅作时间段/行为标签”完成 development 复核。新数据不回滚 W5D-03A、W5D-04、W5D-05 的既有完成状态，也不自动证明跨人、跨机位或临床效果。

全部当前及后续提供的视频、截图、标注和内部研发使用默认已经完整授权；consent、receipt 和隐私审批不再是当前任务门禁。仍保留匿名 person/session/setup/video binding、原视频不进 Git和输出不覆盖等工程约束。

主路线为：

```text
video/tracking
  → technical tracklet QC
  → episode boundary
  → complete episode arc-length resample to 80
  → TopoWander shape classification
  → optional three-frame multimodal purpose/context
  → real session/day aggregation
  → rolling personal baseline
  → backend-facing JSON/JSONL handoff
```

轨迹模型只判断 shape。电话、找东西、清洁、锻炼等困难行为不单独建立轨迹模型，由多模态 context reviewer 补充；reviewer 失败输出 `unknown/unavailable`，不阻塞轨迹、日报或基线。后端和前端不在本轮实现，算法只交付可对接契约。

五日核心代码任务、`W5D-03B` truth-free smoke、居家 labeled-development 复核及本轮分段/分类修复已经完成。当前 `home_input_status=available`、`home_smoke_status=ready`；70 视频 fresh repair run 为 137/137 parent prediction ready、0 unavailable，完整相机回归 `677 passed`。这些 development 结果不得写成独立测试、跨人泛化、检测/跟踪准确率或 `home_validated`。

40 秒路径、C4 sealed、跨 participant/setup、PORTABLE 和长期临床验证均为后续增强，不是五日门禁。最低不可省规则是：人工标签不能根据预测修改；shape 与 purpose/context 分离；technical hard break 不跨越；原视频不进 Git；输出不覆盖；模型/配置/数据身份可追溯；局部改动先跑窄测。

| 任务 | 完成标准 |
|---|---|
| 使跌倒根标签通过 formal 校验 | 从 `generated/v2/` 合并来源文件和 hash 完整的明确标签；`U01/uncertain`、来源缺失和技术隔离记录不进入正式评估 |
| 复核并冻结事件训练标签 v3 | 当前 fall/near-fall 事件门槛已通过；仍需数据负责人检查来源/场景分布、确认现有数据使用范围并冻结版本，且不得把未标注背景自动写成 negative |
| 冻结四个跌倒任务 split | 当前 v2 根事件的 fall/near-fall split 仅为 provisional ready，v3 统一 split 也未冻结；四任务分别取得合格样本、稳定 `split_id`、无泄漏报告和冻结记录。没有真实参考终点的功能/纵向任务继续明确阻塞，不制造空壳正式 split |
| 完成跌倒风险正式评估 | 预注册并冻结事件匹配与统计协议，指定测试集保管人与一次性发布流程；在真实冻结 split 上输出 Precision、Recall、F1、PR-AUC、合法分母下的误报指标、提前量、95% CI 和失败案例 bundle |
| 解除 Workflow A 数据阻断 | 完成 CVAT 身份元数据处置、人员或保守源组说明、功能与纵向参考终点确认；解除证据写入 `reports/fall_risk/workflow_a_blockers.md` |
| 居家剩余标注语义复核 | `wand_H02_dining_1_exercise` / CVAT track 100 为 `pacing + purposeful + ordinary_negative`，与标注手册 role 规则不符；负责人只需裁决 evaluation role。该项不改写已完成的 observable-pattern shape 指标。 |

## P1：提高实时可靠性

实时链路阶段 0-3 的核心代码已完成：固定离线输入和配置 hash、latest-frame 有界队列、源 PTS/单调时钟分域、采样/gap/帧龄诊断、目标四态、分支三态与融合 mask、epoch 清理、episode 生命周期、不可变事件版本和进程内有界异步 outbox 均已有自动化回归。当前证据只覆盖运行状态与投递一致性，不覆盖模型效果、真实萤石、长时资源验收或进程重启后的事件恢复。

| 任务 | 完成标准 |
|---|---|
| 统一跌倒配置来源 | 风险融合阈值从配置传到实际运行对象并有回归测试；主轨迹丢失时间、跌倒状态参数和分支质量门禁已接入服务配置 |
| 完成实时窗口增量计算与资源验收 | 相同窗口 memoization 已完成；仍需对新增姿态后的质量平滑和分支分析做逐层 golden 等价的增量计算，并在固定硬件上给出连续运行、分段延迟、资源峰值和吞吐测试 |
| 校准近跌倒误报 | 增加弯腰、快速坐下、转身、遮挡和多人场景负样本，报告阈值曲线与失败案例 |
| 完善个体基线冷启动 | 明确无历史、初始基线和稳定基线阶段的分数上限、置信度和更新策略 |
| 建立心理健康运行调度 | W5D-05 已交付可重复运行的徘徊算法 runner、日级 JSONL、严格 handoff 和重跑说明；定时触发、HTTP 服务和业务调度仍由后端团队后续实现 |
| 接入摄像头徘徊证据 | W5D-00 至 W5D-05 的 B01+B02 development 算法包、W5D-03B 单段居家 truth-free smoke 及 70-task 时间段/行为标签 development 复核已完成；后续由后端消费独立 wandering evidence，并单独决定是否映射到 `routine_irregularity_score`。CVAT rectangle 不作为检测或跟踪真值；track 100 的 evaluation role 仍待人工裁决 |

## P2：数据和模型增强

- 当前阶段主线：为步态、坐站、近跌倒/跌倒事件构建时序模型训练任务，并在同一数据、同一 split、同一指标协议下与规则 baseline 对照；优先使用 TCN、MS-TCN++ 或 ST-GCN，保留规则安全覆盖。
- 步态已完成目标重构、有效权重、全动作共享预训练、质量捷径隔离、walking gate、条件异常头和物理一致增强。B01-B04 functional proxy 上已补跑六个 seed；单模型 F1 为 `0.145-0.253`，六 seed 动作段概率集成 validation F1 为 `0.308`、balanced accuracy 为 `0.733`、正常误报约 `40.8/小时`。验证集只有 12 个正动作段，test 未读取，集成仅作离线开发候选，运行时继续规则 fallback 且默认 checkpoint 保持 `null`。见[步态 TCN v5 六 seed 集成实验](../../reports/fall_risk/gait_window_v5_effect_first/development-20260803/README.md)。
- 坐站已完成 provisional 数据审计、train/validation-only 派生 split、`[16,14,7]` 数据集、E0、Logistic smoke/pilot，以及候选双头 clip TCN smoke/pilot；test 姿态与指标保持锁定。TCN pilot 的 presence balanced accuracy/F1 为 0.981/0.986，方向 balanced accuracy/macro-F1 为 0.994/0.994，但 1,308/1,409 个事件来自 NTU，不能据此启动正式 E4 或替换规则。下一项最小行动是人工复核一批连续视频并补充显式背景和真实 onset/offset；若保持预裁剪任务，只能继续做标记为 provisional 的分析，不能宣称连续事件定位。见[坐站首轮 provisional 训练](../../reports/fall_risk/sit_stand_event_v1/README.md)。
- 跌倒事件已基于当前 v3 action split 重跑 provisional candidate-clip TCN：`D01/D02/D03/D05` 正类、`A03/A05/A06/A09` 明确非跌倒 proxy，train/validation 生成 2,533 个 `[32,14,7]` 窗口，test 684 条标签保持锁定。三 seed validation F1 为 `0.958/0.961/0.963`，balanced accuracy 为 `0.958/0.960/0.962`，独立 evaluator 复算一致；方向头保持 `not_trained`。validation 保护组只有 9 个且以 NTU 为主，Fall Detection 2017 分域 balanced accuracy 为 `0.826-0.840`。该模型仍是预裁剪动作 proxy，不是连续事件定位或正式效果；连续背景分母、真实 onset/offset 和老人域证据仍缺，TCN 继续只作 shadow。见[跌倒 candidate-clip TCN v3 pilot](../../reports/fall_risk/fall_event_proxy_v2_v3split/README.md)。
- 近跌倒已完成“恢复后确认”训练基础设施、synthetic 小样本烟测和新标签三 seed provisional pilot；2026-08-04 的项目负责人裁决将 962 条 C03 生成为 `stumble_recovery` 正例，并从已确认动作/跌倒事件生成八类 hard negative。当前 primary near-fall 正/负为 948/1,906，train/validation/test 均有正负监督，`training_ready.near_fall_event=true`。本轮只生成 876 个 train/validation 窗口，全部来自 NTU，test 未读取；validation F1 `0.990-0.997` 仅为来源受限开发结果，实际负例窗口类别不完整。下一步应补齐跨来源姿态/连续背景和 hard-negative 窗口覆盖，完成正式协议、连续背景误报验证和老人域外部验证前不得替换 `near-fall-rule-v0.1` 主路径。见[近跌倒恢复确认 TCN provisional pilot](../../reports/fall_risk/near_fall_event_v1/README.md)。
- 模型替换门槛：来源完整标签、人员/源组无泄漏划分、冻结验证协议、误报漏报分析、推理延迟和低质量输入降级证据全部齐备。
- 个体基线和最终风险融合暂不强行监督训练；需要连续个人数据或非空 `risk_labels.jsonl` 后再选择 EWMA/CUSUM、Logistic Regression、LightGBM 或时序融合模型。
- 扩充老人域、近跌倒、低光、遮挡、辅助器具和跨房间数据。
- 通用全视频姿态缓存已覆盖当前 manifest 的全部 6,512 个 eligible RGB 视频：`le2i_imvia` 184 个、`caucafall` 100 个、`ntu_rgbd` 3,914 个、`fall_detection_2017` 2,012 个、`fall_tiktok` 66 个、`pre_vfallp` 108 个、`toaga` 28 个和 `ur_fall` 100 个，剩余 0，且各批均通过完整解析验收。`gstride`、`ltmm` 以及 ToAGa/UR Fall 的表格、时序资产不属于 RGB 姿态缓存。UR Fall 的 `adl-07-cam0.mp4` 源归档本身截断，只缓存了 35 个可解码帧，不能视为完整 180 帧样本；`pre_vfallp` 的姿态缓存也不解除数据集隔离状态。姿态缓存完成不等于训练窗口或模型训练已就绪。
- NTU RGB+D 媒体路径已按仓库内 `data/external/ntu` 重建，3,924 个 AVI 和主 manifest 纳入的 3,914 个资产均可访问；A043 S001-S017 的 938 个已标视频已接入。S016/C003/P008/R001 job revision 已以严格文件名绑定叠加到完整 S016 project；S013/C001、S015/C003 和两组三视角 A05 裁决已写入决定文件及 JSONL。当前 v3 中 446 条全片跌倒已按项目负责人裁决使用首帧 onset、尾帧 offset；后续仍需补齐 S002 缺少的 10 个 C001 任务并复核跨视角方向差异，未标注 A043 不按文件名直接导入。
- 抖音/B站跌倒视频整理批次已按项目负责人决定记录为项目自采并授权内部训练；后续若取得人员对应关系，应把当前单一保守来源池细化为脱敏 subject group 后重新冻结 split。该决定不作为公开再分发授权。
- KINECAL 轻量 TCN 固定划分 baseline 已跑通，但 7 人测试集 balanced accuracy 仅 `0.333`、ROC-AUC `0.583`；进入主链前必须完成重复参与者级交叉验证、RGB 姿态微调和独立外部测试。
- 补消融实验：完整跌倒管线、去掉个体基线、去掉近跌倒、仅事件检测。
- 建立版本化实验报告、复现配置、误报漏报案例和资源成本记录。

## 更新规则

1. 每项任务必须有可检查的完成标准，不能只写“优化”或“完善”。
2. 完成后从本文件移除，并更新对应模块 README 或实验报告。
3. 新模型只有在固定验证集上优于 baseline，且延迟和稳定性满足要求后，才能进入主路径。
