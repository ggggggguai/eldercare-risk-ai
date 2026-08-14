# 坐站连续事件训练执行记录

日期：2026-08-11
状态：`development_provisional`，seed 42 连续流 pilot 已完成，三 seed No-Go

## 数据与门禁

现有人工边界和事件类型发布为 4,278 条开发标签：事件 1,866、显式困难背景 2,220、ignore 192；未新增复核人、未推断未标注背景、未发布 test 真值。

过滤优化采用两项保守策略：短标注区间按长度使用 6-16 个真实观测帧下限并保留 `frame_mask`；同帧多人仅在主轨覆盖率至少 0.60、相对第二轨领先至少 1.35 倍时选择目标，否则拒绝。可用窗口由 2,218 增至 3,759，其中 2,294 个为显式部分上下文，453 个使用占优轨迹，183 个目标歧义、97 个观测不足、15 个关节质量不足、32 个不连续多轨继续拒绝。

pose-aware development split 为 `sitstandsplit_d751c5c4807698fc6882b593`。validation 两方向为 sit-to-stand 64、stand-to-sit 54；controlled_bend/controlled_squat/fall_or_rapid_descent/bed_transfer_or_lying 保护组为 16/10/14/10；NTU validation 占比 0。`training-a` 与 `training-b` 双构建一致，materialization gate 为 `true`。

## 边界修复与 seed 42 pilot

原 smoke 的边界正帧只占监督位置约 3%，未加权 BCE 在 2 epoch 学成全负。训练改为固定 `boundary_positive_weight=12`、±1 帧训练容差，validation 同时保留 exact 和 tolerant 指标；选模加入 tolerant boundary F1，pilot 使用 patience 8。2 epoch 修复 smoke 的 exact/tolerant boundary F1 为 0.103/0.190，不再塌缩。

seed 42、CPU、40 epoch pilot 的最佳 epoch 为 35：

| 指标 | validation |
|---|---:|
| direction macro-F1 | 0.8625 |
| presence F1 | 0.7913 |
| supervised-frame macro-F1 | 0.6190 |
| boundary exact F1 / AP | 0.2050 / 0.1449 |
| boundary ±1 frame F1 | 0.3484 |

## 完整 validation 姿态流评估

新增因果流扫描和事件状态机：按完整 validation 姿态流逐 cutoff 推理，连续 2 帧确认，0.5 秒断点 fail-closed，同方向事件最多合并 0.5 秒。扫描不使用人工 onset/offset 决定窗口位置。规则与 TCN 共享 `sit-stand-event-eval-v1` 的真值、阈值、显式背景分母和一对一匹配器，共覆盖 667 个 validation 视频；test 未读取。

| 指标 | S0 规则 | seed 42 TCN |
|---|---:|---:|
| event Precision | 0.0825 | 0.3111 |
| event Recall | 0.3741 | 0.4029 |
| event F1 | 0.1352 | 0.3511 |
| direction macro-F1 | 0.2515 | 0.3483 |
| boundary IoU median | 0.4747 | 0.4635 |
| onset / offset median error | 0.1667 / 0.3667 秒 | 0.6333 / 0.0583 秒 |
| FP/hour | 1890.16 | 331.91 |
| 36 组 bootstrap event F1 95% CI | [0.0578, 0.2262] | [0.2430, 0.4217] |

TCN 相对规则 event F1 提升 21.6 pp 且 FP/hour 下降，但绝对门禁失败：event F1 低于 0.75、Recall 低于 0.85、onset 中位误差高于 0.50 秒，困难负例误报率除快速下降外均高于 10%。多个 UR Fall 来源组 F1 为 0，CAUCAFall 来源组仅 0.167-0.286；未排除来源退化。显式背景只有 0.286 camera-hour，FP/hour 不稳定。

结论为三 seed `No-Go`。不运行 seed 43/44，不读取 test，不做阈值事后调优，不接入运行主路径。规则 `sit-stand-risk-rule-v0.1` 继续默认，配置 checkpoint 保持 `null`。

## 机器产物

| 产物 | SHA-256 |
|---|---|
| pose-aware split | `8a6c880dabb74b2ba14b3b362b84e81c3c37bee93c1097f5b9b7f00729afd4a0` |
| `training-a/dataset.npz` | `a417d24b817815d2cbba8f1ff303291f29f0a763614af4cfdf379598d36d756e` |
| `training-a/metadata.json` | `ece02a528a1ccc1a20245dc18ec3c7f07b1d644c26fec2400e61604ccfd086d7` |
| seed 42 smoke checkpoint | `ba31ecf551ac421479c365a68cf9401a4dc32449c9938b52468e046b0451ede9` |
| seed 42 smoke metrics | `bcebf8b1b80c1b65b0106b0c3241a9d598a8292627b4579e4a961ebfc30b19be` |
| seed 42 weighted pilot checkpoint | `8bf024c71075ce2802db582266df8c26469c6c235c3efb22927e184d9b37aa5c` |
| seed 42 weighted pilot metrics | `7fba001c68b8c1a447a87f7281cb0d20e072eec02983d19ce7dcd17771eba70a` |
| S0 流评估 | `fcb9584bdf68c3372c19156c3e93ec03fdb90e1ce93eb8d098164020eeabf87d` |
| seed 42 TCN 流评估 | `8f8bd2aeefa4c9127ce23d895591925dd4a72e6b3f52118bd1c2ba07c0b5b9b5` |

`test_pose_read=false`、`test_features_generated=false`、`test_evaluated=false`。
