# 当前任务

更新时间：2026-08-19

本文件只记录尚未完成的工作。已经落地的能力写入模块 README；阶段结论和旧待办移入 `docs/archive/`。当前跌倒风险模块已完成正式比赛交付模型替换并冻结 v1。以下数据、研究评估和工程任务服务于后续 release 或证据完善，不再阻塞 v1；任何模型、阈值、特征或服务配置更新必须新建 release ID。

## P0：完善后续 release 证据

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

环境因素分支的接收契约、有界缓存、因果对齐、弱光/水浸交互特征和非阻塞 shadow 日志已落地，默认配置保持 `environment.mode=disabled`。后续任务仍是采集配对视频/环境数据、冻结 ROI/照度标定、完成 shadow 数据质量与 assist 消融评估；在准入报告完成前不得打开 assist 或宣称环境增强效果。

| 任务 | 完成标准 |
|---|---|
| 统一跌倒配置来源 | 风险融合阈值从配置传到实际运行对象并有回归测试；主轨迹丢失时间、跌倒状态参数和分支质量门禁已接入服务配置 |
| 完成实时窗口增量计算与资源验收 | 相同窗口 memoization 已完成；仍需对新增姿态后的质量平滑和分支分析做逐层 golden 等价的增量计算，并在固定硬件上给出连续运行、分段延迟、资源峰值和吞吐测试 |
| 校准近跌倒误报 | 增加弯腰、快速坐下、转身、遮挡和多人场景负样本，报告阈值曲线与失败案例 |
| 建立心理健康运行调度 | 明确日级任务由谁触发、输入从何处读取、结果如何交付和重跑 |
| 接入摄像头徘徊证据 | 从现有跟踪/姿态质量输出提取单镜头轨迹，完成窗口拒识、片段合并、日级字段和摄像头域评估；先旁路验证，只有固定测试证明有效后才接入 `routine_irregularity_score` |

## P2：数据和模型增强

- 当前阶段主线：保持 `fall-risk-competition-v1-20260819` 不变；新的步态、坐站、近跌倒/跌倒事件实验只作为下一 release 候选，在同一数据、split 和指标协议下与 v1 对照，并保留规则安全覆盖。
- v1 已冻结步态 seed 43、坐站 seed 42 和跌倒事件三 seed checkpoint；这些文件只作为现行交付基线使用，后续训练结果不得覆盖。历史 validation 指标和缺失 test/老人域/连续背景证据继续见各自训练报告。
- 近跌倒 v1 明确冻结规则主路径。现有 TCN 三 seed 与动作辅助消融只作为下一 release 的历史候选，不写入 v1 checkpoint 清单；若继续优化，优先补连续老人域背景及新人员困难负例。见[v2 治理与训练报告](../../reports/fall_risk/near_fall_event_v2/README.md)。
- SCF_MVP_V1 的 P01/P02/P04 已进入训练根标签；P05 challenge、P03 excluded。E1-E3 仍为 No-Go，不替换 checkpoint 或规则主路径。见[执行报告](../../reports/fall_risk/self_collected_scf_mvp_v1/README.md)。
- 后续 release 晋级门槛：来源完整标签、人员/源组无泄漏划分、冻结验证协议、误报漏报分析、推理延迟和低质量输入降级证据全部齐备。该门槛不追溯撤销负责人对 v1 的比赛交付冻结决定。
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
