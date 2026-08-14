# 当前任务

更新时间：2026-08-12

本文件只记录尚未完成的工作。已经落地的能力写入模块 README；阶段结论和旧待办移入 `docs/archive/`。当前跌倒风险模块处于模型化增强阶段：规则 baseline 仍作为对照和安全 fallback，新增时序模型必须经过数据、split、评估和部署门禁后才能替换主路径。

## P0：形成可验证交付

工作流 A 已实现统一 manifest、v2 标注导入/发布、模型训练标签 v3 迁移/统一 split/校验、四任务 split builder、事件评估器和跌倒事件训练 P0 审计，并用合成数据跑通 bundle。项目负责人于 2026-08-04 裁决现有规范动作标签可用于任务级正负样本：当前 v3 有 9,314 条动作、9,498 条事件和 18,812 条 split assignment，fall/near-fall hard negative 类型覆盖完整，962 条 C03 已转为 near-fall positive，446 条当前 NTU 全片跌倒按首帧/尾帧生成精确边界；校验 `valid=true`、事件任务 `training_ready=true`。2026-08-08 P0 复核恢复了 v2 formal 报告并确认当前 split/hash 无漂移和无机器报告泄漏，但总状态仍为 `infrastructure_only`：formal 有 285 个 blocker，validation 只有 7 条 primary fall 标签记录且独立事件数未验，连续背景与显式老人 ADL 均为 0，split/协议未冻结且无 test 保管授权；旧 candidate 报告绑定旧 split，不能复用。当前剩余工作是解除这些门禁并建立一次性 test 发布流程；动作类型任务因稀有类别分区覆盖不足仍为 `training_ready=false`。

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
- 步态 P0-P3 与真实上下文 v2 受限消融已完成：专项审计继续绑定当前 labels/manifest/assignments/split/config hash并封锁 test。v2 只用同轨真实帧补上下文，以 `label_span_mask` 防止标注外帧被当成正/负证据；train 正类二分类监督段由旧协议 61 增至 92，另有 9 段只保留不施加 binary loss。双构建逐字节一致，单 seed、3 epoch TCN 只证明链路可运行；validation B02/B03/B04 仍仅 2/4/3 个主证据段、1 个 source group，正式训练、校准、三 seed、test 和主路径替换继续阻塞。详见[真实上下文 v2 报告](../../reports/fall_risk/gait_observable_context_v2/splitv3-e71a045/README.md)、[原始 v1 报告](../../reports/fall_risk/gait_observable_v1/splitv3-e71a045/README.md)和[步态训练审计](../../reports/fall_risk/gait_training_audit.md)。
- 坐站连续链已按现有人工边界发布 1,866 个事件、2,220 个显式背景和 192 个 ignore。过滤优化将可用窗口由 2,218 提升到 3,759，pose-aware split `sitstandsplit_d751c5c4807698fc6882b593` 双构建一致。边界正样本加权、流式事件解码、长 gap fail-closed 和 bootstrap 背景组修复已完成；seed 42、40 epoch pilot 的窗口级 presence/frame/boundary exact F1 为 0.791/0.619/0.205。667 个完整 validation 姿态流上，TCN event F1 0.351、Recall 0.403、FP/hour 331.9，优于规则的 0.135/0.374/1890.2，但远低于 0.75/0.85 门禁且来源/困难负例退化明显，因此 seed 43/44 No-Go。默认 checkpoint 继续为 `null`，test 未读取，规则主路径保持 `sit-stand-risk-rule-v0.1`。见[坐站连续事件训练记录](../../reports/fall_risk/sit_stand_training_audit.md)和[阻塞清单](../../reports/fall_risk/sit_stand_training_blockers.md)。
- 跌倒事件 candidate-clip TCN pilot 已退休；当前仅保留 17 点/20 通道连续因果输入和审计基础设施。真实连续 dataset、因果 TCN、校准和 predictor 均未实现，规则主路径保持不变。见[P0 审计](../../reports/fall_risk/fall_event_training_audit.md)。
- 近跌倒已完成“恢复后确认”训练基础设施、synthetic 小样本烟测和当前 v3 split 的三 seed 开发训练。标注 track 与姿态 track 命名空间误配已修复，确定性数据集包含 1,105 个窗口、822 个事件；train 覆盖 5 个来源和 7 类困难负例，validation 的 69 个负事件仍全部是 `fast_but_controlled_sit`。seed 42/43/44 的 validation 事件 F1 为 `0.9967/0.9967/0.9933`，但 test 未读取，连续背景、老人域、冻结协议和 test 治理均未完成。下一步不是追加 epoch，而是补齐 validation 七类困难负例、train 的 `progressed_to_fall`、连续背景和老人域数据；这些门禁通过前不得替换 `near-fall-rule-v0.1` 主路径。见[近跌倒恢复确认 TCN 开发训练报告](../../reports/fall_risk/near_fall_event_v1/README.md)。
- SCF_MVP_V1 已完成来源专用 importer、隔离 candidate、262 段四分支 G1 回放和 G2 E1-E3 九次训练。冻结分区为 P01/P02/P04 train、P05 challenge、P03 excluded。E2 的困难负例触发率相对 E0 三 seed 一致下降，但 C03-C05 代理检出同步明显下降；E1-E3 均为 No-Go，不替换 checkpoint 或规则主路径。下一步应围绕 E2 做类别采样/损失权重和阈值联合调优，而不是直接扩大正负样本。见[SCF_MVP_V1 执行报告](../../reports/fall_risk/self_collected_scf_mvp_v1/README.md)。
- 模型替换门槛：来源完整标签、人员/源组无泄漏划分、冻结验证协议、误报漏报分析、推理延迟和低质量输入降级证据全部齐备。
- 个体基线 Phase 0-1 与 Phase 2 算法基础设施已完成合成验收：因果周期契约、鲁棒 reference、状态机、防污染、纵向 observation/review schema、按保守来源组隔离且同人时间前向的 split、泄漏审计、四组固定消融、分层/CI/失败案例和 test release 哈希门禁均已实现。当前待办是采集真实连续周期，补非空且有授权的 subject profiles、独立 `risk_labels` 与双审记录，在 validation 前冻结晋级阈值、最小样本量、split 和协议；这些门禁前真实 builder 保持 `blocked`，不得用 synthetic 指标声称个体化有效或接入主路径。见[Phase 2 状态报告](../../reports/fall_risk/baseline_longitudinal_phase2.md)。
- 扩充老人域、近跌倒、低光、遮挡、辅助器具和跨房间数据。
- 通用全视频姿态缓存已覆盖当前 manifest 的全部 6,512 个 eligible RGB 视频（674,889 源帧、708,177 条 raw/cleaned 姿态记录）；8 个数据集批次均通过严格校验，`errors=0`、无残留 `.part`。`fall_detection_2017` 的 2 个 `completed_no_pose` 视频经核查为空场景；UR Fall 的 `adl-07-cam0.mp4` 已按当前完整 180 帧源修订重提。`gstride`、`ltmm` 以及 ToAGa/UR Fall 的表格、时序资产不属于 RGB 姿态缓存；manifest 中 8 个 `duplicate_content` 视频为 `eligibility=false`，也不进入提取范围。Pre_VFallp 当前 108 个内部授权资产已取消数据集级隔离；公开来源和公开再分发权仍未验证，未知人员继续使用单一保守源组。姿态缓存完成本身不等于训练窗口或模型训练已就绪。
- NTU RGB+D 媒体路径已按仓库内 `data/external/ntu` 重建，3,924 个 AVI 和主 manifest 纳入的 3,914 个资产均可访问；A043 S001-S017 的 938 个已标视频已接入。S016/C003/P008/R001 job revision 已以严格文件名绑定叠加到完整 S016 project；S013/C001、S015/C003 和两组三视角 A05 裁决已写入决定文件及 JSONL。当前 v3 中 446 条全片跌倒已按项目负责人裁决使用首帧 onset、尾帧 offset；后续仍需补齐 S002 缺少的 10 个 C001 任务并复核跨视角方向差异，未标注 A043 不按文件名直接导入。
- 抖音/B站跌倒视频整理批次已按项目负责人决定记录为项目自采并授权内部训练；后续若取得人员对应关系，应把当前单一保守来源池细化为脱敏 subject group 后重新冻结 split。该决定不作为公开再分发授权。
- KINECAL 轻量 TCN baseline 已退休；当前步态只保留 observable-context v2 TCN validation-only 候选，规则主路径不变。
- 补消融实验：完整跌倒管线、去掉个体基线、去掉近跌倒、仅事件检测。
- 建立版本化实验报告、复现配置、误报漏报案例和资源成本记录。

## 更新规则

1. 每项任务必须有可检查的完成标准，不能只写“优化”或“完善”。
2. 完成后从本文件移除，并更新对应模块 README 或实验报告。
3. 新模型只有在固定验证集上优于 baseline，且延迟和稳定性满足要求后，才能进入主路径。
