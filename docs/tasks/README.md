# 当前任务

更新时间：2026-08-03

本文件只记录尚未完成的工作。已经落地的能力写入模块 README；阶段结论和旧待办移入 `docs/archive/`。当前跌倒风险模块处于模型化增强阶段：规则 baseline 仍作为对照和安全 fallback，新增时序模型必须经过数据、split、评估和部署门禁后才能替换主路径。

## P0：形成可验证交付

工作流 A 已实现统一 manifest、v2 标注导入/发布、模型训练标签 v3 迁移/统一 split/校验、四任务 split builder 和事件评估器，并用合成数据跑通 bundle。当前 v3 结构与 split 合法，但没有人工 event negative，near-fall positive 为 0；当前剩余工作是补齐模型监督数据、处理 v2 formal blocker并冻结评估协议。

| 任务 | 完成标准 |
|---|---|
| 使跌倒根标签通过 formal 校验 | 从 `generated/v2/` 合并来源文件和 hash 完整的明确标签；`U01/uncertain`、来源缺失和技术隔离记录不进入正式评估 |
| 使训练标签 v3 达到模型门槛 | 对 fall/near-fall 分别逐窗补齐 hard negative；安全采集并双人复核 near-fall positive；校验报告中目标任务 `training_ready=true`，且不把未标注背景自动写成 negative |
| 冻结四个跌倒任务 split | `fall_event_v1`、`near_fall_event_v1`、`functional_proxy_v1` 和 `longitudinal_baseline_v1` 分别取得合格样本、稳定 `split_id` 和无泄漏报告；没有真实参考终点的任务继续明确阻塞，不制造空壳正式 split |
| 完成跌倒风险正式评估 | 预注册并冻结事件匹配与统计协议，指定测试集保管人与一次性发布流程；在真实冻结 split 上输出 Precision、Recall、F1、PR-AUC、合法分母下的误报指标、提前量、95% CI 和失败案例 bundle |
| 解除 Workflow A 数据与法律阻断 | 完成公开数据来源/许可证确认、CVAT 身份元数据处置、人员或保守源组说明、功能与纵向参考终点确认；解除证据写入 `reports/fall_risk/workflow_a_blockers.md` |
| 完成情绪与社交关注 V3.3.3 生产链 | MH-002、CAM-001、MH-003、五套数据适配、DATA-007 参与者级嵌套五折 split，以及 MODEL-001 ActivityExpert、MODEL-002 SleepExpert、MODEL-003 ActivitySleepJointExpert、MODEL-004 PhysiologyExpert 已完成；下一步执行 MODEL-005 SocialContextExpert，再完成 PersonalTrend、Masked Logistic Stacking、模型包和真实推理。全部折内预处理、校准和融合必须绑定 split ID `mood-social-v3.3.3-participant-nested-5x5-seed-20260728-v1`，模型包不可用时不得回退 legacy 评分卡 |
| 完成真实萤石链路联调 | 使用真实设备或开放平台直播地址启动会话，后端收到并验收风险回调 |
| 完成 S10 认知三模态接线 | ASR 已完成 V3.3 Python 入口、进程级单例/预热、独立 HTTP、WAV/MP3/M4A、媒体约束和 `q_text` 对齐；后续使用 S10 主动小测真实音频联调音频、文本和面部三分支，视频通话音轨作为第二阶段 |
| 固定接口契约 | 后端确认字段、鉴权、时间格式、幂等规则和风险动作编码，并保存联调记录 |
| 完成数据合规材料 | 在采集真人数据前准备知情同意、脱敏编号、访问控制、保留周期和退出删除流程 |
| 固化开发环境与数据版本证据 | 保持 `eldercare-ai` 的 editable 安装指向当前仓库；记录环境、代码、manifest、标签、split 和配置 hash，并用文档中的 conda 命令完成全量复现 |

## P1：提高实时可靠性

实时链路阶段 0-3 的核心代码已完成：固定离线输入和配置 hash、latest-frame 有界队列、源 PTS/单调时钟分域、采样/gap/帧龄诊断、目标四态、分支三态与融合 mask、epoch 清理、episode 生命周期、不可变事件版本和进程内有界异步 outbox 均已有自动化回归。当前证据只覆盖运行状态与投递一致性，不覆盖模型效果、真实萤石、长时资源验收或进程重启后的事件恢复。

| 任务 | 完成标准 |
|---|---|
| 统一跌倒配置来源 | 风险融合阈值从配置传到实际运行对象并有回归测试；主轨迹丢失时间、跌倒状态参数和分支质量门禁已接入服务配置 |
| 完成实时窗口增量计算与资源验收 | 相同窗口 memoization 已完成；仍需对新增姿态后的质量平滑和分支分析做逐层 golden 等价的增量计算，并在固定硬件上给出连续运行、分段延迟、资源峰值和吞吐测试 |
| 校准近跌倒误报 | 增加弯腰、快速坐下、转身、遮挡和多人场景负样本，报告阈值曲线与失败案例 |
| 完善个体基线冷启动 | 明确无历史、初始基线和稳定基线阶段的分数上限、置信度和更新策略 |
| 建立心理健康运行调度 | 明确日级任务由谁触发、输入从何处读取、结果如何交付和重跑 |

## P2：数据和模型增强

- 当前阶段主线：为步态、坐站、近跌倒/跌倒事件构建时序模型训练任务，并在同一数据、同一 split、同一指标协议下与规则 baseline 对照；优先使用 TCN、MS-TCN++ 或 ST-GCN，保留规则安全覆盖。
- 步态运行框架已支持统一 `[T,14,5]` 输入、TCN 常驻推理、规则解释分和显式降级；剩余门禁是训练出任务匹配且经过无泄漏验证、校准和外部测试的 RGB 步态 checkpoint。checkpoint 为空时继续标明 `rule_fallback`，不能把框架完成写成模型有效。
- 模型替换门槛：来源完整标签、人员/源组无泄漏划分、冻结验证协议、误报漏报分析、推理延迟和低质量输入降级证据全部齐备。
- 个体基线和最终风险融合暂不强行监督训练；需要连续个人数据或非空 `risk_labels.jsonl` 后再选择 EWMA/CUSUM、Logistic Regression、LightGBM 或时序融合模型。
- 扩充老人域、近跌倒、低光、遮挡、辅助器具和跨房间数据。
- 恢复 NTU RGB+D 的可用媒体路径：当前 2,976 条精确动作标签及无泄漏 split 已生成，但外部 manifest 指向的旧解压目录不存在；需从 `ntu.zip` 重新解压或重建 manifest 后再做训练。A043 的 948 条已明确排除，不列为待标。
- KINECAL 轻量 TCN 固定划分 baseline 已跑通，但 7 人测试集 balanced accuracy 仅 `0.333`、ROC-AUC `0.583`；进入主链前必须完成重复参与者级交叉验证、RGB 姿态微调和独立外部测试。
- 在规则 baseline 和独立测试集稳定后，再比较 Logistic Regression、LightGBM、TCN 或姿态时序模型。
- RESILIENT 先做一次冻结模型的老人域反向时间迁移敏感性测试；保存结果后才可用全部 73 人做内部嵌套交叉验证适配。48 对 ACE-III 纵向样本只作简单模型探索。
- 补消融实验：完整跌倒管线、去掉个体基线、去掉近跌倒、仅事件检测。
- 建立版本化实验报告、复现配置、误报漏报案例和资源成本记录。

## 更新规则

1. 每项任务必须有可检查的完成标准，不能只写“优化”或“完善”。
2. 完成后从本文件移除，并更新对应模块 README 或实验报告。
3. 新模型只有在固定验证集上优于 baseline，且延迟和稳定性满足要求后，才能进入主路径。
