# 当前任务

更新时间：2026-08-12

本文件只记录尚未完成的工作。已经落地的能力写入模块 README；阶段结论和旧待办移入 `docs/archive/`。当前跌倒风险模块处于模型化增强阶段：规则 baseline 仍作为对照和安全 fallback，新增时序模型必须经过数据、split、评估和部署门禁后才能替换主路径。

## P0：形成可验证交付

工作流 A 已实现统一 manifest、v2/v3 标签、统一 split 和校验。2026-08-17 接纳 SCF_MVP_V1 的 P01/P02/P04 后，v3 有 9,612 条动作、9,651 条事件和 19,263 条 assignment；SCF 451 条全部位于 train。校验 `valid=true`，事件任务 `training_ready=true`，动作类型仍为 false；v2 formal 仍有 285 个既有 blocker。

| 任务 | 完成标准 |
|---|---|
| 使跌倒根标签通过 formal 校验 | 从 `generated/v2/` 合并来源文件和 hash 完整的明确标签；`U01/uncertain`、来源缺失和技术隔离记录不进入正式评估 |
| 复核并冻结事件训练标签 v3 | 当前 fall/near-fall 事件门槛已通过；仍需数据负责人检查来源/场景分布、确认现有数据使用范围并冻结版本，且不得把未标注背景自动写成 negative |
| 冻结四个跌倒任务 split | 当前 v2 根事件的 fall/near-fall split 仅为 provisional ready，v3 统一 split 也未冻结；四任务分别取得合格样本、稳定 `split_id`、无泄漏报告和冻结记录。没有真实参考终点的功能/纵向任务继续明确阻塞，不制造空壳正式 split |
| 完成跌倒风险正式评估 | 预注册并冻结事件匹配与统计协议，指定测试集保管人与一次性发布流程；在真实冻结 split 上输出 Precision、Recall、F1、PR-AUC、合法分母下的误报指标、提前量、95% CI 和失败案例 bundle |
| 解除跌倒事件训练 P0 门禁 | 以 `reports/fall_risk/fall_event_blockers.md` 为当前清单：formal 通过；validation 至少 30 个独立 fall；七类 hard negative 各至少 20 个独立保护组；至少 100 camera-hour 连续背景；老人 ADL 至少 20 人/30 camera-hour；冻结 split/协议并指定 test 保管人。门禁解除前不启动正式训练 |
| 解除 Workflow A 数据阻断 | 完成 CVAT 身份元数据处置、人员或保守源组说明、功能与纵向参考终点确认；解除证据写入 `reports/fall_risk/workflow_a_blockers.md` |
| 完成徘徊方案步骤 3 固定 split | 按[徘徊样行为识别技术方案第 11 节](../modules/mental_health/plans/徘徊样行为识别技术方案.md)先补测试，再实现来源专用 builder：校验步骤 2 六个固定输入哈希；WP 每类精确分成 `280/60/60`（合计 `1120/240/240`，只称 `public_shape_benchmark`）；SmartCare 开发池按自然日替代组固定为 train 152、validation 38，官方 20 条全部且只进入 `sealed_external_test`；输出 split、assignments、近邻审计、报告及哈希，两次全新构建逐字节一致。步骤 3 完成前不开始预处理或模型训练 |
| 建立心理健康评估口径 | 固定日级验证样本、人工复核标签和分层一致性指标，不使用医学诊断表述 |
| 完成真实萤石业务闭环联调 | 2026-08-12 算法端 120 秒严格烟测已通过；仍需业务后端收到并验收风险回调，并验证直播地址刷新 |
| 固定接口契约 | 后端确认字段、鉴权、时间格式、幂等规则和风险动作编码，并保存联调记录 |
| 完成数据合规材料 | 在采集真人数据前准备知情同意、脱敏编号、访问控制、保留周期和退出删除流程 |
| 固化开发环境与数据版本证据 | 保持 `eldercare-ai` 的 editable 安装指向当前仓库；记录环境、代码、manifest、标签、split 和配置 hash，并用文档中的 conda 命令完成全量复现 |

## P1：提高实时可靠性

实时链路阶段 0-3 的核心代码已完成：固定离线输入和配置 hash、latest-frame 有界队列、源/接收时钟分域、采样/gap/帧龄诊断、目标四态、分支三态与融合 mask、epoch 清理、episode 生命周期、不可变事件版本和进程内有界异步 outbox 均已有自动化回归。2026-08-12 真实萤石算法端 120 秒严格烟测已通过；当前仍不覆盖模型效果、业务后端闭环、直播地址刷新、弱网、长时资源验收或进程重启后的事件恢复。

| 任务 | 完成标准 |
|---|---|
| 统一跌倒配置来源 | 风险融合阈值从配置传到实际运行对象并有回归测试；主轨迹丢失时间、跌倒状态参数和分支质量门禁已接入服务配置 |
| 完成实时窗口增量计算与资源验收 | 相同窗口 memoization 已完成；仍需对新增姿态后的质量平滑和分支分析做逐层 golden 等价的增量计算，并在固定硬件上给出连续运行、分段延迟、资源峰值和吞吐测试 |
| 校准近跌倒误报 | 增加弯腰、快速坐下、转身、遮挡和多人场景负样本，报告阈值曲线与失败案例 |
| 建立心理健康运行调度 | 明确日级任务由谁触发、输入从何处读取、结果如何交付和重跑 |
| 接入摄像头徘徊证据 | 从现有跟踪/姿态质量输出提取单镜头轨迹，完成窗口拒识、片段合并、日级字段和摄像头域评估；先旁路验证，只有固定测试证明有效后才接入 `routine_irregularity_score` |

## P2：数据和模型增强

- 当前阶段主线：为步态、坐站、近跌倒/跌倒事件构建时序模型训练任务，并在同一数据、同一 split、同一指标协议下与规则 baseline 对照；优先使用 TCN、MS-TCN++ 或 ST-GCN，保留规则安全覆盖。
- 步态 observable-context v2 已按 `splitv3_3342705b7b1ac51570148337` 重建并完成三 seed。2026-08-18 项目负责人基于比赛时间约束批准 transfer seed 43 进入默认实时步态主分支；服务加载路径、CPU、16 帧契约已固定，输入不足或推理失败自动回退规则。该决定解决工程接入，不解除 test、老人域、冻结协议和样本规模门禁，证据等级仍为 provisional。见[运行启用记录](../../reports/fall_risk/gait_runtime_activation_20260818.md)。
- 坐站 v2 已重用全部可用受审动作并接入 SCF P01/P02/P04 auxiliary train：发布 1,942 个事件、5,512 个显式背景和 229 个 ignore，物化感知保护组 split 生成 11,373 个多 cutoff 因果窗口。seed 42 pilot 的完整 validation 流 event F1/Recall/FP-hour 为 `0.449/0.419/158.8`，recall-balanced 复评为 `0.380/0.507/419.7`；均未达到 0.75/0.85 门禁。2026-08-18 按最高执行人比赛期决定，默认分支切到 `experimental_tcn`，最新 cutoff 推理失败或无事件时回退规则；证据等级仍为 `development_provisional`，test 未读取。见[启用记录](../../reports/fall_risk/sit_stand_runtime_activation_20260818.md)与[v2 治理与训练报告](../../reports/fall_risk/sit_stand_event_v2/README.md)。
- 跌倒事件 candidate-clip TCN pilot 已退休；连续治理 v2 已重新绑定当前 `splitv3_3342705b7b1ac51570148337`，7,701 条开发监督物化为 7,504 个 `[32,17,20]` 因果窗口，train/validation=`5,721/1,783`。因果 residual TCN 已完成 seed 42/43/44，固定阈值 0.5 的 validation F1/PR-AUC 均值为 `0.6745/0.6782`；三模型平均概率 F1/Recall=`0.6824/0.7360`，但普通背景误报率 `0.2116`，onset validation 仅 7 条，SCF 仅在 train。2026-08-18 为比赛交付将三枚 checkpoint 以 `experimental_tcn` 接入实时主评分，TCN 命中沿用强触发契约，窗口不足/推理失败回退规则；raw 分数、来源和 fallback 写入诊断。连续背景、老人域、冻结协议和独立阈值校准仍未通过，test 未读取，证据等级保持 `development_provisional`。见[训练报告](../../reports/fall_risk/fall_event_continuous_tcn_v2/README.md)。
- 近跌倒 v2 的 `splitv3_e71a045.../c7fd...` 产物和三 seed 指标均为历史 provisional 结果，不绑定当前 `splitv3_3342705b7b1ac51570148337`；SCF 发布后的窗口、事件和困难负例需重建，test 仍未读取。当前不扩 seed、不晋级 checkpoint，优先补连续老人域背景及新人员困难负例。见[v2 治理与训练报告](../../reports/fall_risk/near_fall_event_v2/README.md)。
- SCF_MVP_V1 的 P01/P02/P04 已进入训练根标签；P05 challenge、P03 excluded。E1-E3 仍为 No-Go，不替换 checkpoint 或规则主路径。见[执行报告](../../reports/fall_risk/self_collected_scf_mvp_v1/README.md)。
- 模型替换门槛：来源完整标签、人员/源组无泄漏划分、冻结验证协议、误报漏报分析、推理延迟和低质量输入降级证据全部齐备。
- 个体基线 Phase 0-1 与 Phase 2 算法基础设施已完成合成验收；已接入实时主链路的显式周期提交入口 `POST /v1/monitoring/sessions/{session_id}/baseline-period`，没有有效周期输入时仍失败关闭。当前待办是采集真实连续周期，补非空且有授权的 subject profiles、独立 `risk_labels` 与双审记录，在 validation 前冻结晋级阈值、最小样本量、split 和协议；这些门禁前真实 builder 保持 `blocked`，不得用 synthetic 指标声称个体化有效或接入默认主路径。见[Phase 2 状态报告](../../reports/fall_risk/baseline_longitudinal_phase2.md)。
- 扩充老人域、近跌倒、低光、遮挡、辅助器具和跨房间数据。
- 主 pose cache 的 8 个批次覆盖 6,512 个 eligible RGB 视频（674,889 源帧、708,177 条 raw/cleaned 记录），SCF P01/P02/P04 的 150 个 train 视频复用独立 cleaned pose；步态数据准备通过多 pose root 合并读取，当前 6,662 个 eligible RGB 视频均有 pose。`fall_detection_2017` 的 2 个 `completed_no_pose` 视频经核查为空场景；`gstride`、`ltmm` 及表格/时序资产不属于 RGB pose cache。姿态覆盖不等于正式训练或主链路替换已就绪。
- NTU RGB+D 媒体路径已按仓库内 `data/external/ntu` 重建，3,924 个 AVI 和主 manifest 纳入的 3,914 个资产均可访问；A043 S001-S017 的 938 个已标视频已接入。S016/C003/P008/R001 job revision 已以严格文件名绑定叠加到完整 S016 project；S013/C001、S015/C003 和两组三视角 A05 裁决已写入决定文件及 JSONL。当前 v3 中 446 条全片跌倒已按项目负责人裁决使用首帧 onset、尾帧 offset；后续仍需补齐 S002 缺少的 10 个 C001 任务并复核跨视角方向差异，未标注 A043 不按文件名直接导入。
- 抖音/B站跌倒视频整理批次已按项目负责人决定记录为项目自采并授权内部训练；后续若取得人员对应关系，应把当前单一保守来源池细化为脱敏 subject group 后重新冻结 split。该决定不作为公开再分发授权。
- KINECAL 轻量 TCN baseline 已退休；当前步态只启用 observable-context v2 pretrained seed 43，规则作为失败降级和解释分保留。
- 补消融实验：完整跌倒管线、去掉个体基线、去掉近跌倒、仅事件检测。
- 建立版本化实验报告、复现配置、误报漏报案例和资源成本记录。

## 更新规则

1. 每项任务必须有可检查的完成标准，不能只写“优化”或“完善”。
2. 完成后从本文件移除，并更新对应模块 README 或实验报告。
3. 新模型只有在固定验证集上优于 baseline，且延迟和稳定性满足要求后，才能进入主路径。
