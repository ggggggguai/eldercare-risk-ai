# 坐站 TCN 训练数据 v2 治理与开发评估

状态：`development_provisional`，主链替换 `No-Go`。更新时间：2026-08-17。

## 结论

SCF_MVP_V1 发布后，坐站专项不覆盖根 v2/v3 或旧 v1 产物，而是新建 v2 标签、保护组 split 和因果窗口。数据门禁已通过，TCN 在同一 v2 validation 上明显优于规则，但绝对事件指标、连续背景时长、老人域和冻结协议仍不满足替换条件，运行主路径继续使用规则。

| 项目 | v2 结果 |
|---|---:|
| 发布标签 | 7,683 |
| 事件 / 显式背景 / ignore | 1,942 / 5,512 / 229 |
| 起身 / 坐下事件 | 1,024 / 918 |
| SCF train-only 标签 | 298 |
| 物化窗口 | 11,373 |
| train / validation 窗口 | 10,081 / 1,292 |
| train / validation 保护组 | 133 / 37 |
| test pose / features / evaluation | 0 / 0 / 0 |

## 治理规则

- 输入绑定 `action_labels_v3` SHA-256 `c439acdc...2509f59` 和根 split `splitv3_3342705b7b1ac51570148337`。
- A03/A04/B06/C01 是事件；已受审的步行、转向、弯腰、下蹲、卧床转移、支撑接触、近跌倒恢复和跌倒区间只作为显式背景或困难负例。未标注时间不推断为背景。
- SCF P01/P02/P04 强制为 auxiliary、train-only，并按 0.5 降权；P05 challenge 和 P03 excluded 不进入发布。根 v3 与坐站专项均保留 auxiliary tier。
- split 按 subject/source/sample/video/event/source-action 保护组重建。首轮非物化 split 因 validation 可用卧床转移组仅 6 组而失败；最终 `sitstandsplit_11e163684e3bb898a5c3508d` 使用首轮样本的 pose 可物化性清单重平衡，10 组门禁通过。
- 事件在 50%、75%、100% 进度生成因果 cutoff；仅在不跨越下一条已标注动作时增加 0.5 秒 post-event cutoff。显式背景取 50% 和 100% cutoff。每个窗口只读取 cutoff 及之前姿态。
- 权重先按事件和保护组归一，再做来源平方根平衡、SCF auxiliary 降权，最后固定背景/起身/坐下总权重为 50%/25%/25%。presence 和 direction loss 已改为实际消费样本权重。

最终物化门禁覆盖 train/validation 起身 `902/63`、坐下 `818/52`；validation 的弯腰、下蹲、跌倒快速下降、卧床转移可用保护组分别为 `16/10/14/10`。拒绝计数按 cutoff 统计：观测帧不足 6,246、目标轨迹歧义 749、多轨 54、关节质量不足 24。

## 开发结果

seed 42 loss-scale-fixed pilot 在 epoch 38 最优。窗口级 presence/direction/frame/tolerant-boundary F1 为 `0.498/0.897/0.651/0.243`。完整 v2 validation 流式结果：

| 指标 | TCN | 规则 |
|---|---:|---:|
| Event F1 | 0.449 | 0.168 |
| Precision | 0.483 | 0.108 |
| Recall | 0.419 | 0.382 |
| Direction macro-F1 | 0.427 | 0.232 |
| FP/hour | 158.80 | 1142.82 |
| 显式背景 camera-hour | 0.353 | 0.353 |

TCN 相对规则有明显增益，但仍低于 event F1 `0.75`、Recall `0.85` 门禁；坐下 F1 只有 `0.360`，卧床转移、跪地活动和跳跃误报率分别为 `29.4%`、`50%` 和 `100%`。FP/hour 分母远低于稳定结论所需 5 camera-hour，更低于正式 validation 目标 20 camera-hour。因此不继续在同一 validation 上扫阈值，不运行 seed 43/44，不读取 test，不配置默认 checkpoint。

## Recall-balanced 采样复评

为验证召回优先的数据策略，在冻结根 split、TCN 结构、`pilot` profile 和流式评估协议不变的前提下，单独调整了采样：事件 cutoff 改为 `25/50/75/100%`，post-event 改为 `0.25/0.5s`，背景/起身/坐下权重改为 `40/30/30`。新数据集包含 13,147 个窗口（train/validation 为 11,754/1,393），门禁通过，SCF 物化窗口仍全部 train-only，test pose/features/evaluation 均为 0。

seed 42 在第 19 个 epoch 达到最佳窗口选择分数，随后早停。完整 validation 流结果如下：

| 指标 | recall-balanced TCN | 当前 TCN 候选 | 规则 |
|---|---:|---:|---:|
| Event F1 | 0.380 | 0.449 | 0.168 |
| Precision | 0.304 | 0.483 | 0.108 |
| Recall | 0.507 | 0.419 | 0.382 |
| Direction macro-F1 | 0.383 | 0.427 | 0.232 |
| FP/hour | 419.70 | 158.80 | 1142.82 |

该策略确实提高召回，但误报约为当前候选的 2.6 倍，event F1 下降；因此不晋级主链，也不追加 seed 43/44。机器产物：

- 数据集：`data/processed/fall_risk/sit_stand_continuous_v2/splitv3_3342705b-materialized/recall-balanced-v2/`
- 训练：`reports/fall_risk/sit_stand_continuous_tcn_v2/sitstandsplit_11e163684e3bb898a5c3508d/seed42-pilot-recall-balanced-v2/`
- 流式评估：同目录下 `seed42-pilot-recall-balanced-v2-stream-evaluation/` 和 `recall-balanced-rule-stream-evaluation/`

机器产物：

- 标签发布：`reports/fall_risk/sit-stand-event-label-publication-v2.json`
- 最终 split：`data/splits/fall_risk/sit_stand_event_v2_splitv3_3342705b_materialized/`
- 最终数据：`data/processed/fall_risk/sit_stand_continuous_v2/splitv3_3342705b-materialized/balanced-causal-v2-class-balanced/`
- pilot：`reports/fall_risk/sit_stand_continuous_tcn_v2/sitstandsplit_11e163684e3bb898a5c3508d/seed42-pilot-loss-scale-fixed/`
- 同协议流评估：同目录下 `seed42-pilot-loss-scale-fixed-stream-evaluation/` 与 `rule-stream-evaluation/`

下一轮数据投入只做三件事：补连续老人域背景至至少 5 camera-hour，补不同人员的卧床转移/跪地/坐下困难负例，修复 SCF 根 v3 tier 后冻结新 split 与协议。没有这些数据时继续扩模型或追加 epoch，预期收益低于过拟合风险。

## 比赛期运行覆盖

2026-08-18，项目最高执行人批准比赛期将默认坐站分支切换为原始 seed 42 TCN。该决定只改变运行配置，不改变上述评估结论：模型仍为 `development_provisional`，正式替换门禁仍 `No-Go`。运行适配只推理最新 causal cutoff；无事件、输入不足或推理异常时回退规则，并在诊断中记录来源和 fallback 原因。

启用记录见 `reports/fall_risk/sit_stand_runtime_activation_20260818.md`。
