# W5D-03B 居家 truth-free smoke 与标注接入审计

日期：2026-08-22

## 结论

W5D-03B 的真实居家输入烟测已经完成。烟测从原始
`wand_R01_hall_1.mp4` 重新执行 YOLOv8 + ByteTrack、`closing-s008-d04`
automatic proposal、fixed TopoWander shape forward 和三帧 context 工程契约，最终
`home_smoke_status=ready`、`delivery_status=completed_with_home_smoke`。本次没有读取人工
truth，不输出准确率/F1，不发出 `AlgorithmEvent`，也不把单段复放写成 `home_validated`。

独立 CVAT project 标注已完成裁决、全量导入和 development 评价：70 个 task、135 条
`source=manual` 的 `wandering_episode` track 均有完整八属性。task 1657 / track 108 已按
负责人裁决改为 `direct + ordinary_negative`；原 XML 有哈希备份。automatic-boundary 与
oracle-boundary shape 指标已分别计算，人工标签仅作为时间段/行为语义 truth，CVAT 框没有
作为检测或跟踪真值。

## 输入事实

- 原始目录：`C:\Users\lenovo\Desktop\测试文件夹`
- 原始 MP4：70 个，共 4,078,709,265 bytes
- 既有 tracking/media sidecar：70/70；只读盘点，不作为本次 fresh smoke 的 tracking 输入
- CVAT XML：`wandering标注\annotations.xml`
- 原 CVAT XML SHA-256：`94ad9aee2169e6f0960cb6b740a7b62ad4f0d4f1ea780f9b4235c23a1b911137`
- 裁决后 CVAT XML SHA-256：`8aa9911c3b542aee8b9c0ae9eaa2ccdd3cabbed83715dcebb2986f55b3103327`
- CVAT XML：70 task、135 manual episode；所有 task 均至少有一条 episode，八属性无
  `not_set`

人工标签分布为：

- shape：direct 106、pacing 9、lapping 8、random 8、unknown 4
- evaluation role：ordinary negative 107、wandering-like positive 24、uncertain 4
- visibility：good 135；tracking issue：not checked 135

## Truth-free smoke

运行 ID：`w5d03b-home-r01-hall1-20260821-v2`

输入视频 `wand_R01_hall_1.mp4` 的 SHA-256 为
`ceac87747f3c4dc97f71a2e9337a903c8d31d386c10d96da9084dcd3df546a7c`。fresh tracking
SHA-256 为 `e9d112bdbb240071f32d9a4122bd0bac32dbdac0d88d792535237bdb9eef800c`。

复放输出：

- 487 条 fresh tracking observation
- 1 条 automatic proposal
- 1 次 fixed TopoWander forward
- 1 条 episode result，状态 `uncertain`
- 1 条 context review，三帧 scene/crop 共 6 个 JPEG 引用均可读且哈希一致
- 0 条 human truth input
- `home_smoke_status=ready`

该 episode 的 automatic binary 结果为 `wandering_like`，分数
`0.9610769152641296`；四分类诊断为 `random`，分数 `0.5760876536369324`。这只是
truth-free 自动输出，不是正确率或真实标签结论。context 使用 deterministic fake provider
验证契约，`exercise/0.5` 不是模型能力或真实目的判断。

完整、未入 Git 的复放 bundle 位于
`tmp/wandering_home_w5d03b_20260821_v1/home_smoke_r01_hall1_v2/`。最终目录不含临时
shape batch index，文本产物中也没有残留 staging 路径。原视频、fresh tracking、
帧图和中间产物继续留在 Git 外；本报告只记录身份、计数、哈希和证据边界。

## 人工标注裁决与全量导入

### 2026-08-22 负责人裁决

负责人确认 task 1657 / CVAT track 108 的 `observable_pattern=direct` 正确，并授权将
`evaluation_role` 从 `purposeful_hard_negative` 改为 `ordinary_negative`。负责人同时确认
全部居家 CVAT 标注只作为时间段/行为语义标签，CVAT rectangle 不作为检测或跟踪真值。
后续导入仍保留几何审计，但机器 Track ID 只能标记为单主体时序覆盖派生，不能据此计算或
宣称检测/跟踪性能。

原始 XML 已备份为
`annotations.pre_track108_adjudication_20260822.sha256-94ad9aee.xml`；其 SHA-256 与裁决前
XML 相同。track 108 的 50 个重复关键帧属性全部一致更新，`observable_pattern=direct`
保持不变。

### 裁决前阻断记录

正式 project XML 导入在以下记录 fail closed：

- task ID：1657
- video：`wand_H02_dining_2_sweep`
- CVAT track ID：108
- observable pattern：`direct`
- purpose context：`purposeful`
- purpose evidence：`observed_context`
- evaluation role：`purposeful_hard_negative`
- script type：`cleaning`

现行契约要求 `purposeful_hard_negative` 为 purposeful 且 shape 不能是 `direct`。因此需要
标注人确认：若 `direct` 正确，应改 evaluation role；若 purposeful hard negative 正确，应
把 observable pattern 改成实际的非 direct shape。算法侧不替标注人裁决。

几何审计还显示 CVAT 框不能作为机器 Track ID 真值：135 条 episode 的 CVAT-machine bbox
IoU 中位数为 0、均值为 0.0021375，134/135 低于 0.1，0/135 达到 0.5。另一方面，按
单主体时序覆盖派生的机器轨迹覆盖中位数为 1.0、均值为 0.995687，134/135 至少 0.8。
因此当前只能把标注解释为时间段/行为语义标签，并把 Track ID 绑定标记为
`single_subject_temporal_dominance_derived`；不能声称人工框验证了检测或跟踪。

裁决后 v3 导入结果为 70/70 task、135/135 episode；134/135 达到时序覆盖门，唯一未达项
是 `wand_R01_dining_1` / CVAT track 131，覆盖率 `0.539851`，其 oracle prediction 按 QC
输出 `unavailable/boundary_tracking_gap`，没有调用模型。导入状态因此保持
`labeled_development_intake_uncertain_track_alignment`，不是检测/跟踪验收通过。

## Labeled-development 结果

### Automatic boundary

按独立人工时间段、冻结 matching policy 评价 70 个视频：135 条 ready truth，325 条原始
candidate 中 159 条进入 all-locomotion 主视图，匹配 90 条。主视图 precision/recall/F1 为
`0.566038/0.666667/0.612245`，匹配项 temporal IoU 中位数/均值为
`0.720114/0.734842`。严格 proposed-only 条件视图只有 19 条 candidate，precision/recall/F1
为 `0.736842/0.103704/0.181818`。因此分段器在本居家 development 数据上明显未达可靠自动
边界水平；不能用高条件 precision 掩盖极低 recall。

### Oracle-boundary shape

固定 `topowander-m0s-seed20260731-epoch0005` 在 135 条人工时间段上运行，推理不读取 truth：
126 ready、9 unavailable、0 inference error；131 条 shape 可评分，其中 122 条 prediction
ready，覆盖率 `0.931298`。

- all-shape-eligible（pipeline miss 计入）：binary accuracy/macro-F1=
  `0.923664/0.930021`；四分类 accuracy/macro-F1=`0.839695/0.492788`。
- ready-only conditional：binary accuracy/macro-F1=`0.991803/0.985342`；四分类
  accuracy/macro-F1=`0.901639/0.523205`。
- pacing support 9，但模型预测 pacing 为 0，recall/F1 均为 0；四分类 subtype 仍不能视为
  可用。
- `dark_1` 只有 2/9 shape-eligible episode ready；低照 QC 是主要覆盖失败。

另发现 `wand_H02_dining_1_exercise` / CVAT track 100 为
`pacing + purposeful + ordinary_negative`，与标注手册的 purposeful non-direct role 规则不符。
该问题不改写 observable-pattern shape 指标，但 purposeful-hard-negative diagnostic 仍需负责人
另行裁决，当前不报告该诊断为干净结果。

## 证据边界

- 已证明：真实居家 MP4 能走通 fresh tracking -> automatic episode -> shape -> 三帧
  context 工程闭环；70-task 时间段/行为 truth 可用于 development boundary 与 oracle-shape 评价。
- 已量化但未达可用：automatic-boundary 主视图 F1 `0.612245`；oracle 四分类 macro-F1
  `0.492788`，pacing recall 为 0。
- 未证明：检测/跟踪性能、跨 participant/setup 泛化、context accuracy、长期个人 baseline 或
  临床效果。所有 70 个视频仍属于同一 development participant，不能写成独立测试集。
