# SCF_MVP_V1 自采数据基线回放与近跌倒增强门禁报告

初次执行日期：2026-08-12；训练治理更新：2026-08-17
执行范围：`自采数据模型增强计划` 的 G0、G1 和 G2
结论：G0 候选构建、G1 回放和 G2 训练已完成；2026-08-17 起 P01/P02/P04 的受审标签进入训练根标签。步态 action-pretrained hierarchical TCN 已完成三 seed provisional 训练，但尚不满足主链路替换门禁；近跌倒三组候选 checkpoint 仍均为 No-Go。

## 1. 边界

- 规则 baseline 仍是运行主路径和低质量输入 fallback，所有候选 checkpoint 仅用于冻结回放。
- 262 段输入均为动作中心短片，不报告 FP/hour，也不把短片拼接成连续背景。
- 文件名动作只用于压力分层，不替代 onset/peak/recovery、方向或事件边界的 canonical truth。
- 项目负责人已确认该批次为自采数据、允许内部开发训练，且 262 个文件标注已经复核。
- 冻结开发分区为 P01/P02/P04 auxiliary train、P05 challenge、P03 excluded；重复内容按 SHA-256 分组。
- 根 manifest、v2/v3 和统一 split 已于 2026-08-17 重建；SCF assignment 只进入 train，现有 validation/test 不接纳 SCF。
- `test_pose_read=false`、`test_evaluated=false`。
- 发布策略为 `training_tier_cap=auxiliary`、`split_partition=train`；SCF 的 gait role 仅为 `action_pretraining_and_walking_gate_only`，不作为 conditional abnormal head 的正/负监督。

## 2. G0 候选构建

来源专用导入器严格绑定交付清单和 5 份脱敏 CVAT ZIP 的 SHA-256，处理 P02 4 倍坐标缩放、P03 `-90` 度显示坐标转换、显式帧映射和内容哈希保护组。两次全新构建的四个输出逐字节一致。

| 项目 | 结果 |
|---|---:|
| 视频 / 唯一内容 | 262 / 258 |
| CVAT 动作轨迹 | 786 |
| 近跌倒候选 | 294 |
| 可进入 loss / challenge-only | 124 / 52 个候选（P03 排除） |

G0 已 `ready`。P01/P02/P04 共 150 个视频、298 条动作已发布到根标签链；P03 整组排除，P05 保持 challenge-only。

## 3. G1 冻结回放

姿态阶段使用 `yolov8n-pose.pt`（SHA-256 `c6fa93dd1ee4...641d212`）和现有质量控制。262 段均生成姿态，合计 43,454 条记录；平均/中位检测帧覆盖率为 `95.09%/100%`，25 段低于 80%，101 段出现多轨迹，5 段存在超过 0.75 秒的源 gap，平均核心关键点质量为 `0.9174`。session、rotation 和上述质量字段已写入机器汇总；相机视角、低光和遮挡没有可靠源标注，显式记为 `unavailable`。

| 分支 | 冻结对象 | 短片回放结果 |
|---|---|---|
| 步态 | `gait-risk-rule-v0.2`；context-v2 seed 42 TCN | 683 个窗口；规则在 228/262 段达到 0.5，TCN 在 81/262 段达到冻结阈值 `0.3992367` |
| 坐站 | `sit-stand-risk-rule-v0.1`；continuous seed 42 TCN | TCN 有 37,398 个有效因果窗口；规则/TCN 分别解码 164/272 个事件 |
| 近跌倒 | `near-fall-rule-v0.1`；seed 42/43/44 TCN | 2,153 个 3 秒因果窗口；规则触发 32/262 段，TCN 分别触发 190/193/191 段 |
| 跌倒事件 | `fall-state-rule-v0.1` | 0/262 段触发；未复用旧 candidate-clip TCN |

这些数字是固定阈值下的压力回放触发数，不是准确率。步态规则在非步行动作上普遍高分，说明其 walking gate 不足；坐站脚本动作没有双审事件边界，不能据此计算 Precision/Recall；跌倒规则 0 触发也不能证明召回，因为本批只有 1 条未复核 D03 且文件名动作不是跌倒真值。

来源分层显示明显域差异：步态 TCN 在 P05 触发 `49/56`，在 P03 仅 `2/56`。该差异要求后续做相机、旋转、分辨率和参与者捷径消融，不能当作行为差异。

## 4. G2 近跌倒结果

E1-E3 各运行 seed 42/43/44，共 9 次。自采有效训练窗口为 E1 `152`、E2 `171`、E3 `195`；现有 validation 固定为 `444` 个窗口、`367` 个事件。C03/C04/C05 使用 `reviewed_action_interval_end_proxy` 粗恢复锚点，不代表精确 recovery 标注。

| 实验 | validation event F1（42/43/44） | P05 六类负例总体触发率 | P05 C03-C05 代理检出率 | 决策 |
|---|---|---|---|---|
| E0 | `0.9967/0.9967/0.9933` | `94.1%/94.1%/94.1%` | `93.8%/100%/100%` | 冻结基线 |
| E1 | `0.9966/0.9867/0.9900` | `88.2%/94.1%/79.4%` | `87.5%/100%/75.0%` | No-Go |
| E2 | `0.9966/0.9900/0.9950` | `79.4%/88.2%/79.4%` | `75.0%/87.5%/75.0%` | No-Go |
| E3 | `0.9933/0.9900/0.9900` | `88.2%/88.2%/70.6%` | `87.5%/81.2%/62.5%` | No-Go |

9 次运行都通过现有 validation F1 相对 E0 下降不超过 1 个百分点的门槛，但 P05 显示明显敏感度交换。E2 的困难负例方向最稳定，三个 seed 分别下降 `14.7/5.9/14.7` 个百分点；同时正例代理检出下降 `18.8/12.5/25.0` 个百分点，因此不能升级 checkpoint。

逐动作、Wilson 95% CI 和运行证据见 `near_fall_augmentation/evaluation.json`。P05 共 50 个有效事件、145 个窗口，另有 2 个候选因 3 秒因果上下文不足被拒绝。短动作片段未报告 FP/hour。

六类计划负例共 126 段，其中 112 段至少有一个有效 TCN 窗口。规则触发 `16/126`；三 seed 在有效窗口分母上的短片触发如下：

| 动作 | 分母 | seed 42 | seed 43 | seed 44 |
|---|---:|---:|---:|---:|
| A02 正常转身 | 9 | 8 | 9 | 9 |
| A03 可控坐下 | 29 | 13 | 13 | 14 |
| A05 可控下蹲 | 8 | 3 | 4 | 5 |
| A06 可控弯腰 | 29 | 23 | 23 | 20 |
| A08 日常扶物 | 27 | 27 | 27 | 27 |
| A10 正常调步 | 10 | 10 | 10 | 10 |
| 合计 | 112 | 84 (75.0%) | 86 (76.8%) | 85 (75.9%) |

对应 Wilson 95% CI 分别为 `0.662-0.821`、`0.682-0.836`、`0.672-0.829`。75 段由三个 seed 一致触发，11 段由两个触发，8 段由一个触发，32 段均不触发。P05 的六类计划负例中，三个 checkpoint 均对全部 27 个有效短片触发，构成严重的自采域偏移或来源捷径证据。

C03/C04/C05 原始触发只用于复核排队：有效窗口分母为 28/30/8，seed 42 触发 28/30/8，seed 43 为 28/27/8，seed 44 为 28/26/8。因为尚无双审 canonical 恢复事件，这些数字不是 Recall。

G2 最终决策为：E0 `frozen_baseline_available`，E1/E2/E3 `No-Go`。近跌倒数据集、checkpoint 和评估产物均隔离保存；步态 TCN 的单独运行启用见 `gait_runtime_activation_20260818.md`，不改变近跌倒 No-Go 结论。

## 5. 步态 TCN 训练治理（2026-08-17）

SCF 发布后按 `splitv3_3342705b7b1ac51570148337` 继承既有 6,516 个资产分区，只新增 150 个 SCF train 资产和 451 条 assignment；P03/P05 在 root/v3/split 中均为 0。步态 observable-context v2 重新物化 2,372 个窗口（train/validation `1,849/523`，独立正段 `15`），其中 SCF 260 个窗口只参与 shared encoder/walking gate，conditional abnormal head 权重为 0。

| 方案 | seed 42 | seed 43 | seed 44 | 均值 | 决策 |
|---|---:|---:|---:|---:|---|
| scratch TCN validation F1 | 0.692 | 0.720 | 0.643 | 0.685 | 对照 |
| action-pretrained hierarchical TCN validation F1 | 0.621 | 0.857 | 0.857 | 0.778 | provisional 首选 |

迁移方案 ROC-AUC 为 `0.994/0.997/0.998`，Brier 为 `0.0125/0.0138/0.0150`；正常窗口误报为 `44.1/12.0/12.0` 次/hour。seed 方差和误报量仍偏高，且仅有 2 个 source groups、没有老人域连续背景验证。test pose/tensor/metrics 均未读取，当前状态为 `development_provisional`；2026-08-18 仅 pretrained seed 43 按受控运行决定启用，规则 baseline 继续作为 fallback。

步态产物：

- `data/processed/fall_risk/action_pretraining_v1/splitv3-3342705-scfaux-v1/`：2,619 windows，dataset SHA `7eab581f5d4819a20b8e770a24ea1c0d46a4c23a75d5fa7e2a71c3014948468c`。
- `data/processed/fall_risk/gait_observable_context_v2/splitv3-3342705-scfaux-v1/`：2,372 windows，dataset SHA `37d8f93e18e41de50a6a1039845a082e815b2446804f9daf1442c05e53adfa1c`，metadata SHA `3601872cd9fafc2b7aa7fccd628ad223e64ca8b3ac8b723ca7a46702d09a5717`。
- `reports/fall_risk/gait_training_audit_scfaux_v1.md`：当前审计结论 `development_provisional`，`runtime_replacement_allowed=false`。

主链路替换仍需补齐：至少 30 个独立 validation 正段、每个异常子类至少 10 段、至少 3 个 source groups，连续背景/老人域验证，冻结 split 和评估协议，test 保管释放，阈值/校准、延迟和低质量输入降级证据。

## 6. 机器证据与复现

本地机器证据：

- `baseline_replay/summary.json`：SHA-256 `4e20e9128c95...6d1b8f9`
- `g1_baseline_replay/summary.json`：SHA-256 `649e7383632322d6f80f00004f2092455027f4d23967fb5d5a1e320e9f059325`
- `near_fall_augmentation/gate.json`：SHA-256 `882cdd0daf3b...86276`

上述 JSON、姿态缓存和逐视频输出含本地数据派生信息，按仓库规则不提交。可提交的 importer、回放脚本、门禁配置、隔离 candidate 和本报告共同记录执行契约。

```bash
conda run -n eldercare-ai python scripts/evaluate/replay_self_collected_g1.py \
  --manifest data/annotations/fall_risk/generated/v2/self_collected_scf_mvp_v1_candidate/manifest.jsonl \
  --action-labels data/annotations/fall_risk/generated/v2/self_collected_scf_mvp_v1_candidate/action_labels.jsonl \
  --pose-dir reports/fall_risk/self_collected_scf_mvp_v1/baseline_replay/cleaned_pose \
  --gait-checkpoint reports/fall_risk/gait_observable_context_v2/splitv3-e71a045/development-seed-42/best_model.pt \
  --gait-metrics reports/fall_risk/gait_observable_context_v2/splitv3-e71a045/development-seed-42/metrics.json \
  --sit-stand-checkpoint reports/fall_risk/sit_stand_continuous_tcn_v1/sitstandsplit_d751c5c4807698fc6882b593/seed42-pilot-boundary-v2/best_model.pt \
  --sit-stand-training-config configs/training/sit_stand_event_v1.yaml \
  --sit-stand-evaluation-config configs/evaluation/sit_stand_event_v1.provisional.yaml \
  --output-dir reports/fall_risk/self_collected_scf_mvp_v1/g1_baseline_replay \
  --device cpu
```

下一步以 E2 为受控研究基线，联合评估类别采样、损失权重和阈值校准；P05 继续保持独立 challenge，不回流训练。只有负例触发下降且 C03-C05 代理检出不再明显退化时，才允许提出新的候选 checkpoint。
