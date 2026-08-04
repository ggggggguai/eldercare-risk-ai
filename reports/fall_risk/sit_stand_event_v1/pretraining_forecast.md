# 坐站事件首轮训练前预估

生成时间：2026-08-03

生成时点：E0 规则开发评估已完成，E2 smoke/pilot 尚未启动。为避免事后回填，本文不为已经执行的 E0 伪造数值预测区间；E0 实测单独列出。

## 1. 训练就绪结论

当前结论为 **provisional**，不是 formal。

- 可训练：`sit_stand_event_presence_proxy_v1`，仅做显式动作 clip 上的坐站事件存在性 E2 Logistic Regression；只使用 train/validation。
- 不可训练：正式 `sit_stand_event_localization_v1` E4。显式连续背景标签为 0、`linked_event_id` 为 0、经确认的真实 onset/offset 为 0；1,896 条 `exact` 仅表示动作 clip 边界，不能当事件边界。
- 不可训练：`normal/slow/failed` 三分类。`B06` 无 action-type primary，`C01` 无 test primary。
- 不可训练：MS-TCN++ 相位分割和 functional proxy。逐帧相位标签与独立功能真值均为空。

所有 test 姿态和 test 特征保持未读取。当前 1,276 条锁定 test 候选仅计数，不进入 dataset、模型选择、阈值或指标。

## 2. 审计与数据规模

输入事实源：

- labels SHA-256：`9955ec8e0c08ff91f397e8e07cb7e90270ac02accbadbba35f7b2870f60371d1`
- manifest SHA-256：`38f15531110ca5401e10575ade5aaf6b7baf497dc04c143a1005d9f013bc61db`
- source split SHA-256：`13b6f42f7232285816ea0f7b76017604168453e230d19145c1bcf7eaf409ef68`
- source assignments SHA-256：`83433d0f62e2a2b24fccccc5ca66154da72d6b3fb6495ba025f84645b29ddb4c`
- prepared dataset SHA-256：`5293cca410459694b37268160fcad38966558c5fb14babae1c88d66d7c0317bf`
- derived split SHA-256：`73f2e745b7a6eaeea7d289fec8dc3cd8f2f4d14261897db920422ac26655c2ed`

| Partition | 窗口 | 独立事件 | 保护组 | 正/负事件 | 来源事件覆盖 |
|---|---:|---:|---:|---:|---|
| train | 1,646 | 1,574 | 121 | 868 / 706 | NTU 1,158；Fall Detection 2017 189；CaucaFall 103；UR Fall 70；LE2I 54 |
| validation | 1,439 | 1,409 | 11 | 973 / 436 | NTU 1,308；LE2I 69；Fall Detection 2017 32 |
| test | 未构建 | 未读取 | 未读取 | 不评估 | 1,276 条标签候选保持锁定 |

动作覆盖：正类为 `A03/A04`；负类为具有 primary 的显式 `A05/A06/A09/D01/D02/D03/D05`。`A07/A11/D04` 在当前门禁后没有进入接受样本。共拒绝 593 个候选窗：观测帧不足 562、质量不足 29、无匹配轨迹 1、标签时间内无姿态 1。

接受窗口质量较高但分布过于集中：usable frame ratio 中位数 1.0、均值 0.997；核心点覆盖中位数 1.0、均值 0.995；关键点质量均值 0.951。主要偏差不是平均质量，而是正常坐站正类和 validation 高度集中于 NTU、validation 只有 11 个保护组、clip 边界/数据源捷径，以及困难老人域覆盖不足。

## 3. 训练前指标预估

以下区间只针对 validation 的 clip-level proxy，不是事件 onset/offset、FP/hour 或正式 test 表现。

| 实验 | 指标预估 | 依据与假设 | 置信度 | 最可能失败模式 |
|---|---|---|---|---|
| E0 规则状态机 | 不可可靠预估；E2 启动前已实测：balanced accuracy 0.270、F1 0.246、PR-AUC 0.636、Recall 0.181 | 执行 E0 前没有独立数值先验，因此不事后构造区间；实测混淆矩阵为 `[[157,279],[797,176]]` | 实测值高；泛化含义低 | 视角/尺度阈值失配，正常动作漏检，跌倒高度变化误报，目标轨迹选择错误 |
| E2 Logistic | balanced accuracy 0.55-0.75；F1 0.60-0.82；PR-AUC 0.70-0.90 | 垂直位移、腿伸展和速度特征应优于 E0 固定阈值；但 validation 正类率 69.1%、NTU 占 92.8%，且只有 11 个保护组 | 低到中 | 学到数据源/clip 长度/姿态质量捷径；跨来源表现分裂；规则派生特征在线性模型中不可分 |
| E4 事件 TCN | 不可可靠预估，也不得训练 | 显式连续背景、真实 onset/offset 和事件关联均为 0；当前张量只能支撑 clip proxy，不能形成合法定位监督 | 高 | 若强行训练会把 clip 边界当事件边界，得到不可解释的虚高分和连续视频高误报 |

## 4. 资源预估

当前 dataset 为 3,085 x 16 x 14 x 7，磁盘约 14 MiB，训练设备为 CPU（当前受控进程中 MPS unavailable）。

| 运行 | 预计时长 | 峰值内存 | 新增磁盘 |
|---|---:|---:|---:|
| E2 smoke，seed 42，`max_iter=25` | 1-10 秒 | 0.25-0.60 GiB | < 5 MiB |
| E2 pilot，seed 42，`max_iter=1000` | 1-30 秒 | 0.25-0.60 GiB | < 5 MiB |
| E4 TCN | 不估计 | 不估计 | 不估计 |

## 5. 目标门槛

以下来自训练方案，是补齐正式事件定位数据后的研发目标，不是本轮预估，也不适用于 clip proxy：

- validation event F1 >= 0.75。
- validation boundary IoU median >= 0.60。
- onset/offset median absolute error <= 0.50 sec。
- 每个主要数据源 event recall >= 0.60。
- 相对规则 baseline，F1 提升 >= 0.05，或相同 recall 下 FP/hour 明显下降。

本轮 E2 即使超过 0.75，也不能据此宣布达到事件定位门槛，因为当前没有边界 IoU、onset/offset 误差、连续背景 FP/hour 或冻结 test 结果。

## 6. 不可预测结论

当前不能预测或声明：正式 test 表现、连续视频事件定位能力、真实老人跨家庭泛化、临床坐站功能、未来跌倒概率、MS-TCN++ 相位能力或主链替换收益。
