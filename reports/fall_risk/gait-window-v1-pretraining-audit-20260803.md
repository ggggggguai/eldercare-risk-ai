# 步态窗口 v1 训练前审计

日期：2026-08-03

状态：工程链路与数据分布审计，不是正式模型评估。当前 v3 校验报告仍为 `training_ready.action_type=false`，因此本文的训练烟测不能作为 checkpoint 替换或效果声明依据。

## 1. 冻结输入

- 标签：`data/annotations/fall_risk/action_labels_v3.jsonl`
- manifest：`data/manifests/fall_risk_video_manifest.jsonl`
- assignments：`data/splits/fall_risk/training_labels_v3/assignments.jsonl`
- split：`splitv3_32db8736e890c9fad95a8292`
- 姿态缓存：`data/processed/fall_risk/pose_quality_y8n_v1/cleaned`
- 标签 SHA-256：`9955ec8e0c08ff91f397e8e07cb7e90270ac02accbadbba35f7b2870f60371d1`
- manifest SHA-256：`38f15531110ca5401e10575ade5aaf6b7baf497dc04c143a1005d9f013bc61db`
- assignments SHA-256：`83433d0f62e2a2b24fccccc5ca66154da72d6b3fb6495ba025f84645b29ddb4c`
- 数据集 SHA-256：`33590a640e8033c3deaf12ad16dd8a46eaa3fa465262fcc99bbf16d095f25fea`

构建契约固定为 `4 FPS x 4 秒 = 16 帧`，步长 8 帧，最大时间 gap 0.5 秒，最少 10 个观测帧，每个动作段最多 4 个窗口。primary 动作段的窗口权重和为 1.0，auxiliary 为 0.35。

## 2. 选择与构建结果

v3 选择器得到 5,581 个动作段：5,169 个 normal activity、412 个 gait instability。正常类包含 A01-A12；异常类包含 B01-B04。最终 1,473 个动作段生成 2,097 个窗口，其中 1,609 个负类、488 个正类。

| 数据源 | 生成窗口的动作段 |
|---|---:|
| CaucaFall | 82 |
| 抖音/B站整理 | 29 |
| LE2I/IMViA | 194 |
| NTU RGB+D | 960 |
| Pre_VFallp | 143 |
| TOAGA | 27 |
| UR Fall | 38 |

拒绝计数为：观测帧不足 4,134 个候选窗口，步态质量不足 56 个候选窗口。同一源观测只能占用一个目标时间槽，不能通过最近邻复制虚增观测数。构建器同时验证标签、manifest、assignments 的 SHA-256，验证冻结 assignment 字段，并拒绝跨 partition 的 split/source/sample group。

## 3. 分区分布

| 方案 | 分区 | 窗口 | 动作段 | 负类窗口 | 正类窗口 |
|---|---|---:|---:|---:|---:|
| frozen | train | 676 | 525 | 658 | 18 |
| frozen | validation | 710 | 643 | 689 | 21 |
| frozen | test | 675 | 282 | 231 | 444 |
| frozen | excluded | 36 | 23 | 31 | 5 |
| Fold A | train | 506 | 314 | 451 | 55 |
| Fold A | validation: Pre_VFallp | 507 | 143 | 95 | 412 |
| Fold B | train | 716 | 263 | 304 | 412 |
| Fold B | validation: LE2I | 253 | 167 | 210 | 43 |

frozen 分区虽包含两个类别，但 train/validation 的正例分别只有 18/21 个窗口，而 test 集中 444 个正例，不适合开发期模型选择。Fold A/B 用于跨来源开发验证；两折的类别比例明显不对称，必须分别报告并按最差折选择候选，不得只报告合并平均。

Fold A/B 的 action segment、split group、source group 和 sample group 的 train/validation 交集均为 0。抖音/B站、NTU 和 TOAGA 不进入 Fold A/B 训练或阈值选择。

## 4. 训练烟测

两折均完成以下工程烟测：

- 规则、LightGBM、EBM：LightGBM 10 轮，EBM 10 轮。
- TCN：CPU、1 epoch、8 hidden channels、dilation 1/2。
- 两类训练器均返回 `test_metrics=null` 或 `test_group_metrics=null`。
- 输出目录没有 test prediction 文件，证明正式数据默认锁定测试评估。

烟测指标没有模型选择含义。1 epoch TCN 的 Fold A/Fold B 动作段 ROC-AUC 分别约为 0.484/0.480，接近随机，仅证明代码路径可运行。

## 5. 当前阻塞

1. `reports/fall_risk/training-labels-v3-validation.json` 仍为 `training_ready.action_type=false`。
2. B01 `slow_walk` 只能解释为可观察的功能风险 proxy，不能作为“步态不稳”或医学异常的直接真值。
3. frozen 分区不适合作开发选模；必须使用 Fold A/B，并在模型和阈值冻结后才允许显式 `--evaluate-test`。
4. 尚未完成三 seed 正式训练、分组置信区间、校准、正常视频误报、固定硬件延迟和真实视频端到端验收。
