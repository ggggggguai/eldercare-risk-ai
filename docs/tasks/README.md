# 当前任务

更新时间：2026-08-15

本文件只记录尚未完成的工作。已经落地的能力写入模块 README；阶段结论和旧待办移入 `docs/archive/`。当前跌倒风险模块处于模型化增强阶段：规则 baseline 仍作为对照和安全 fallback，新增时序模型必须经过数据、split、评估和部署门禁后才能替换主路径。

## P0：形成可验证交付

工作流 A 已实现统一 manifest、v2 标注导入/发布、模型训练标签 v3 迁移/统一 split/校验、四任务 split builder 和事件评估器，并用合成数据跑通 bundle。项目负责人于 2026-08-04 裁决现有规范动作标签可用于任务级正负样本：当前 v3 有 9,314 条动作、9,498 条事件和 18,812 条 split assignment，fall/near-fall hard negative 类型覆盖完整，962 条 C03 已转为 near-fall positive，446 条当前 NTU 全片跌倒按首帧/尾帧生成精确边界；校验 `valid=true`、事件任务 `training_ready=true`。fall 和 near-fall 已在锁定 test 的前提下完成三 seed train/validation 开发实验。当前剩余工作是处理 v2 formal blocker、冻结事件标签与评估协议、补充连续背景/老人域/跨来源证据并建立一次性 test 发布流程；动作类型任务因稀有类别分区覆盖不足仍为 `training_ready=false`。

### 徘徊当前主任务：M0-CAM-EP1B

`current_code_task=M0-CAM-EP1B`。公开轨迹模型、冻结候选、tracking、Camera QC、弧长重采样、camera forward、evaluation primitives 和 synthetic 日级/baseline preview 都已存在，不推倒重来。EP1A 已把完整 episode 路径提升为正式 config/module/importer/CLI，并由 D01/S00 回归确认。EP1B 的代码、指标分母和停止条件见 [执行任务书](../modules/mental_health/plans/M0-CAM-EP1B执行任务书.md)。

当前同时保留两条明确分层的路径：EP1A 正式入口执行 **人工/CVAT/whole-clip 边界 → episode 内 QC → 弧长 80 点 → frozen shape forward**；旧 `camera_episode.py` 仍执行 **40 秒滑窗预测 → 相邻同预测类别窗口合并**，只作 legacy long-context diagnostic。D01 12 条 whole-clip 与 S00 三条 XML episode 已从正式入口复现 direct；S00 合并的 0–40 秒仍为 lapping-like。本机已有 P01 4 条、L01 3 条、R01 3 条、H01–H05 共 9 条、N01 1 条和 Q01–Q03 共 3 条剪辑，但除 S00 外尚未形成 XML/tracking/正式 episode 输出。下一缺口是先标注、跟踪并评价这些现有样片，再按有效 support 补拍，不是先调模型和阈值。

主路线固定为：

```text
video/tracking
  → technical tracklet QC
  → episode boundary
  → complete episode arc-length resample to 80
  → frozen TopoWander shape classification
  → episode sequence duration/repetition + observable purpose/context
  → alert candidate（后续）
```

40 秒路径保留为 `legacy long-context shape diagnostic`，不覆盖 episode 真值，不直接作为告警。PORTABLE/F1、逐文件 hash、source anchor 和重复 exact-schema 审计均降为跨机部署/正式发布前债务，不阻塞本机 EP1/EP2 开发。

最低不可省规则只有：人工标签不能根据预测修改；participant/session/setup 先分组再派生 episode；原视频不进 Git；冻结模型/阈值作为基线保留；输出不覆盖；局部改动跑窄测并回归 D01/S00。

| 任务 | 完成标准 |
|---|---|
| 使跌倒根标签通过 formal 校验 | 从 `generated/v2/` 合并来源文件和 hash 完整的明确标签；`U01/uncertain`、来源缺失和技术隔离记录不进入正式评估 |
| 复核并冻结事件训练标签 v3 | 当前 fall/near-fall 事件门槛已通过；仍需数据负责人检查来源/场景分布、确认现有数据使用范围并冻结版本，且不得把未标注背景自动写成 negative |
| 冻结四个跌倒任务 split | 当前 v2 根事件的 fall/near-fall split 仅为 provisional ready，v3 统一 split 也未冻结；四任务分别取得合格样本、稳定 `split_id`、无泄漏报告和冻结记录。没有真实参考终点的功能/纵向任务继续明确阻塞，不制造空壳正式 split |
| 完成跌倒风险正式评估 | 预注册并冻结事件匹配与统计协议，指定测试集保管人与一次性发布流程；在真实冻结 split 上输出 Precision、Recall、F1、PR-AUC、合法分母下的误报指标、提前量、95% CI 和失败案例 bundle |
| 解除 Workflow A 数据阻断 | 完成 CVAT 身份元数据处置、人员或保守源组说明、功能与纵向参考终点确认；解除证据写入 `reports/fall_risk/workflow_a_blockers.md` |
| `M0-CAM-EP1B`：摄像头 shape 小样本 | 先实现轻量批量 evaluator，并标注/跟踪现有 P01/L01/R01；prediction 与独立 truth 一对一连接，以全部 shape-eligible truth 为主分母，非 ready 不得静默删除。先用每类约 2 条跑通，再把 pacing/lapping/random 补到各约 10 个有效 episode；报告四分类/shape-binary precision、recall、F1、support、QC coverage、时长分层、分组支持和 purposeful 诊断。该规模只用于路线选择，不宣称现实 95%。详见 [EP1B 任务书](../modules/mental_health/plans/M0-CAM-EP1B执行任务书.md)。 |
| `M0-CAM-EP2A`：连续视频自动/半自动边界 | 用 track start/end、出画重入、长 gap、持续停留后活动切换等信号产生 boundary；一次转向/折返/闭环不能直接切段。报告 boundary tIoU、起止误差、漏切、多切，以及自动切段后的端到端 shape F1；不拿 oracle-boundary 成绩冒充自动识别。 |
| 自然负例与困难负例 | 先收集 30–60 分钟自然活动，再逐步扩充多小时、多 participant/setup。坐、站、办公等非行走时段进入独立 `continuous_negative_timeline`，不能伪标 direct。先报 shape-candidate false activations / eligible negative person-hour；最终 alert FAR 等告警决策层实现后再报。 |
| `M0-CAM-EP2B/EP3`：长上下文与时间/上下文增强 | 比较 episode-only 与 episode + legacy 40 秒诊断。先把 duration、dwell、重复 episode、回访和 purpose/context 作为模型外证据；只有留出 development 数据证明冻结模型或决策层不足时，才新建时间感知/域适配候选并重训，不直接翻开当前禁用的时间通道。 |
| 正式数据与发布债务 | 日常 EP1/EP2 只需匿名编号、独立标签、分组防泄漏、非覆盖输出和真实样片回归。现有 collection/receipt/exact-schema/hash/PORTABLE 工具只在跨机交接、数据/模型冻结、sealed 评估或正式发布时集中启用；没有实际失败时不继续扩展审计。 |
| 建立心理健康评估口径 | 固定日级验证样本、人工复核标签和分层一致性指标，不使用医学诊断表述 |
| 完成真实萤石链路联调 | 使用真实设备或开放平台直播地址启动会话，后端收到并验收风险回调 |
| 固定接口契约 | 后端确认字段、鉴权、时间格式、幂等规则和风险动作编码，并保存联调记录 |
| 完成数据合规材料 | 在采集真人数据前准备知情同意、脱敏编号、访问控制、保留周期和退出删除流程 |
| 固化开发环境与数据版本证据 | 保持 `eldercare-ai` 的 editable 安装指向当前仓库；记录环境、代码、manifest、标签、split 和配置 hash，并用文档中的 conda 命令完成全量复现 |

### 徘徊连续执行队列

| 顺序 | 当前事务 | 完成标志 |
|---|---|---|
| W1 | EP1B 四类 camera episode 小样本 | 先消费现有 P/L/R/H 视频并完成批量 evaluator；每类有可评分完整 episode，输出 all-eligible 主指标、ready-only 条件指标、QC coverage、时长/分组支持和失败案例。工具完成但数据不足时只写 `implementation_complete + data_pending`。 |
| W2 | EP2A 连续混合视频与 automatic boundary | 报 boundary tIoU/onset-offset、漏切/多切和自动端到端 shape 指标；与 oracle 上限分开。 |
| W3 | 自然活动和 purposeful hard negatives | 建立 continuous-negative timeline；报告 candidate false activations/person-hour、QC coverage 和主要误触发。 |
| W4 | EP2B/EP3 决策 | 只有留出数据证明 episode-only 不足时才采用 40 秒辅助、外部 duration/dwell/repetition/context 或新训练候选；每次只验证一个主要变化。 |
| W5 | Development freeze 与 C4 | 冻结 episode producer、shape candidate、boundary/uncertain/matching 和告警决策；最后在独立 participant/session/setup 上评价，sealed 结果不回调同一版本。 |
| W6 | 产品接入 | 复用已有 session/day/baseline preview，将稳定 episode 序列接入日级 evidence；真实 person binding、风险策略和 `AlgorithmEvent` 分别验证。 |
| R | 发布/跨机债务 | 仅在实际部署或正式交付前集中关闭 PORTABLE、资产恢复和必要 hash；不在 W1–W5 期间继续扩展 synthetic 审计。 |

## P1：提高实时可靠性

实时链路阶段 0-3 的核心代码已完成：固定离线输入和配置 hash、latest-frame 有界队列、源 PTS/单调时钟分域、采样/gap/帧龄诊断、目标四态、分支三态与融合 mask、epoch 清理、episode 生命周期、不可变事件版本和进程内有界异步 outbox 均已有自动化回归。当前证据只覆盖运行状态与投递一致性，不覆盖模型效果、真实萤石、长时资源验收或进程重启后的事件恢复。

| 任务 | 完成标准 |
|---|---|
| 统一跌倒配置来源 | 风险融合阈值从配置传到实际运行对象并有回归测试；主轨迹丢失时间、跌倒状态参数和分支质量门禁已接入服务配置 |
| 完成实时窗口增量计算与资源验收 | 相同窗口 memoization 已完成；仍需对新增姿态后的质量平滑和分支分析做逐层 golden 等价的增量计算，并在固定硬件上给出连续运行、分段延迟、资源峰值和吞吐测试 |
| 校准近跌倒误报 | 增加弯腰、快速坐下、转身、遮挡和多人场景负样本，报告阈值曲线与失败案例 |
| 完善个体基线冷启动 | 明确无历史、初始基线和稳定基线阶段的分数上限、置信度和更新策略 |
| 建立心理健康运行调度 | 明确日级任务由谁触发、输入从何处读取、结果如何交付和重跑 |
| 接入摄像头徘徊证据 | 先完成 episode-first runner，并分别报告 oracle-boundary shape、automatic boundary+shape、legacy 40 秒诊断、candidate false activations/person-hour、延迟和覆盖率。只有 episode producer、目的/持续性决策和独立 sealed session 证据齐全后才接入 `routine_irregularity_score` |

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
