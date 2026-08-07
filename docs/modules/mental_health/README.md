# 心理健康风险算法模块

本模块输出行为与睡眠变化的工程特征和独立心理健康风险事件，用于风险预警和人工复核，不输出医学诊断。当前已完成数据适配、日级聚合、个人基线、持续异常、风险评分和离线日级 CLI。

跌倒风险与心理健康风险共享上游感知、人员身份和姿态质量数据，但分别评分并分别输出各自的 `AlgorithmEvent`。心理健康管线只产生 `module=mental_health` 的独立事件。

专项目标设计见[徘徊样行为识别技术方案](plans/徘徊样行为识别技术方案.md)，跨团队分工、交付物和验收条件见[徘徊模块协作交接与职责边界](徘徊模块协作交接与职责边界.md)。当前已建立隔离的 `mental_health.wandering` 实验子包，提供版本化 `TrajectorySample` 契约、严格且原子化的安全 JSONL 读写，以及 `wandering-source-manifest-v1`、`wandering-conversion-report-v1`、`wandering-split-v1` 三项严格契约。manifest/report 会绑定相对来源路径、来源和输出 SHA-256、转换器版本、样本计数、稳定 sample ID 与结构化警告；split 会绑定来源 manifest 哈希，拒绝重复、未知、遗漏和跨分区 sample ID，并把 `sealed_external_test` 作为独立互斥分区。所有 JSON 采用固定排序和原子写入，原始来源只读。

方案步骤 2 已于 2026-08-03 完成。两个隔离转换器和真实产物通过自动验证：WanderingPatterns 写出 1,600 条，SmartCare 写出 210 条并拒绝 10 条；当前严格 reader 共读回 1,810 条，两次全新运行的 9 个 bundle 文件逐文件 SHA-256 一致。SmartCare 的 9 条极短轨迹、2 条越界轨迹及 1 条重叠关系均有结构化记录；WanderingPatterns 固定哈希 DataFrame 的实际列中没有人员、会话或其他受支持 group 字段。固定种子为 6 个类别各生成 20 条联系表，用户已确认 6 张均可通过，详见[步骤 2 转换与复核记录](../../../reports/mental_health/wandering_step2/README.md)和[人工复核记录](../../../reports/mental_health/wandering_step2/HUMAN_REVIEW.md)。

方案步骤 3 已于 2026-08-03 完成。`configs/data/wandering_split_v1.yaml` 固定六个输入哈希、`seed=20260731`、`group_policy=source_specific_v1` 和 SmartCare validation 日期；来源专用 builder 在分配前校验全部输入，已有输出目录时拒绝覆盖。1,810 个 ID 全部且只分配一次：WP 四类各 `280/60/60`，SmartCare 开发池 train 152/validation 38，官方 20 条全部且只进入 `sealed_external_test`。五个机器产物位于 `data/splits/mental_health/wandering/v1/`，split 语义哈希为 `4ac4a3877a056809066562cb09e4d30aa1d38baafcb4600f1f8a8a672776adbf`；两个全新目录重建逐字节一致，正式记录见[步骤 3 固定 split 复现记录](../../../reports/mental_health/wandering_step3/README.md)。

方案步骤 4 的代码和机器产物已于 2026-08-03 完成。`configs/data/wandering_preprocessing_v1.yaml` 同时钉死两份公开来源、全部 split 相关文件和 canonical split 哈希；builder 只读取 WP samples、SmartCare train pool、split 和 assignments，未打开 official validation。正式 bundle 一对一保留 1,790 个非 sealed ID：1,775 ready，15 条 SmartCare train 为 `unavailable/too_few_valid_points`（normal 10、wandering-like 5）；ready 固定为 train 1,257、validation 278、WP test 240。每条 ready 记录含 `T=80` 来源/shape 视图、严格 `[80,14]` 的 `raw_features/model_features`、全 1 mask 和拓扑诊断；WP 不伪造 image 画布，当前时间两通道全 0。train-only 统计使用 100,560 个有效位置，四个机器文件在两个新目录逐字节一致。完整哈希、命令与边界见[步骤 4 复现记录](../../../reports/mental_health/wandering_step4/README.md)。

步骤 4 的人工门禁也已关闭。项目用户于 2026-08-03 完成 48 条固定 ready train 和全部 15 条 unavailable 短轨迹复核，未记录异常，[HUMAN_REVIEW.md](../../../reports/mental_health/wandering_step4/HUMAN_REVIEW.md) 状态为 `human_review_passed`。本轮复验中，步骤 4 窄测试为 `25 passed, 20 subtests passed`，全部徘徊回归为 `73 passed, 74 subtests passed`；另在全新目录重建四个机器文件并与正式 bundle 逐字节一致。因此步骤 4 已满足进入步骤 5 的技术和人工门禁。

方案步骤 5 已于 2026-08-03 完成，并于 2026-08-04 通过独立复核。strict loader 同时绑定步骤 4 四个机器文件、canonical split、assignments、近邻审计和 `human_review_passed` 原始字节；固定 26 维手工特征与两个 Random Forest 对照任务按五个预注册 seed 运行。development 只保存 1,535 条 train+validation ready 特征、只在 train 拟合；外部 manifest SHA 明确传入后，evaluator 才对 240 条 WP test 生成特征和指标。RF config SHA 为 `d29db531f42ae5d7cf4ed540a85681ddab78e483695633ccc8b382d98e745e35`，development manifest 为 `fff6340e868de32bee2021ec1000f166b8caabe5caeb1132abb8ab822bfaaaf2`，public benchmark manifest 为 `8b89b0b4f14e2c9e4054b8121c9ee8428eab8b6311c32a411b699f46cb02167c`。正式产物的文件哈希、样本/标签计数、逐条预测、10 个模型指纹和解释输出均独立复算一致；另在全新目录完整训练并重放 evaluator，两个 manifest 与正式结果逐字节一致。窄测试为 `22 passed, 21 subtests passed`，全部徘徊回归为 `95 passed, 95 subtests passed`；完整测试为 `473 passed, 1 failed, 162 subtests passed`，唯一失败是跌倒模块固定视频缺失。四分类 validation 五 seed macro-F1 为 `0.9726±0.0033`，二分类 WP/SmartCare 来源等权 macro-F1 为 `0.9507±0.0069`；完整分来源指标、安全加载、解释、失败案例、CPU 延迟与确定性证据见[步骤 5 RF 对照报告](../../../reports/mental_health/wandering_step5/README.md)。

方案步骤 6 已于 2026-08-04 完成。`configs/modules/wandering_tcn_v1.yaml` 同时绑定 RF config、RF development/public manifest、步骤 4 bundle/split/schema 和全部结构/训练参数；development 继续调用步骤 5 strict loader。WP 四分类与 WP+SmartCare 二分类分别从随机初始化训练五个互不共享任何参数或优化器的纯 TCN，共 10 个单线程 CPU checkpoint。输入只含 `[80,14] model_features + point_mask`，零基索引 10/11 时间通道保持 0、索引 12 与外部 mask 一致；15 条 unavailable 未进入 tensor/loss，SmartCare 未进入 four-class，official 20 条没有步骤 6 入口。checkpoint 使用固定 ZIP 元数据、排序 state key、仅含有限 float32 数组的 NPZ，加载不调用 `torch.load`；外部 manifest SHA、全部 artifact、配置/源码/环境、NPZ key/dtype/shape/finite 和 semantic fingerprint 全部通过后才允许 forward。

步骤 6 development manifest 为 `0f4c48d948f0f4355dd577c89330b050ecb1bb513ca83ef1b25234897e27a10e`，public benchmark manifest 为 `1b0c67f86b291dfba4d48314d470d5bc09f84089aac7e23c718e3cd77ad8c2c4`。两个独立 development 目录各 46 个文件、两个固定 test 目录各 14 个文件，逐文件长度和 SHA-256 差异均为 0。validation 五 seed 四分类 macro-F1 为 `0.9411±0.0105`，二分类来源等权 macro-F1 为 `0.9967±0.0011`；WP public shape benchmark 分别为 `0.9380±0.0106` 和 `0.9844±0.0075`。相对同 seed RF，四分类 validation/test 分别低 `0.0315/0.0328`，二分类分别高 `0.0460/0.0242`；没有为改善结果修改 split、seed、early stopping 或结构。完整指标、训练历史、安全加载、失败案例和 CPU 基准见[步骤 6 纯 TCN 对照报告](../../../reports/mental_health/wandering_step6/README.md)。

方案步骤 7 已于 2026-08-04 完成，并在同日关闭工程评审发现的三个边界问题。`configs/modules/wandering_camera_v1.yaml` 以 exact fields 固定步骤 4/5/6 信任根、2 Hz bucket、40 秒/80 点窗口、20 秒 stride、Camera QC、高度补偿和 `primary_seed=20260731`。新 camera 子层严格校验 `wandering-media-v1` 与来源 tracking SHA，只消费 `frame_id/track_id/bbox/track_confidence/timestamp_sec`；上游临时 `person_id`、bbox 中心和像素速度不进入徘徊身份或轨迹。adapter 以完整 `source_group/video/device/setup/epoch/track` 隔离，并将授权状态冻结为三个 exact 枚举；QC 以过滤前合法观测范围保留候选窗，使低置信度边缘和全低置信度轨迹留下 `unavailable` 审计记录。ready 窗口再复用步骤 4 的弧长重采样、稳健各向同性归一化、14 通道、拓扑和冻结 feature stats。

步骤 7 的唯一 CLI 支持互斥的 `tracking JSONL + sidecar` 或本地有限 MP4；MP4 模式直接复用既有 YOLO/ByteTrack，不复制检测跟踪。RF 与 TCN 只经各自安全 loader 加载主 seed，并为每个 ready 窗口分别输出四分类/二分类四组未校准概率，固定 `comparison_only` 且不融合。只有明确的模型 forward/输出异常降级为单窗 `inference_error`，其他契约错误使整次运行失败。修复后固定 80 条合成 bbox 的 tracking+sidecar 两次全新构建得到相同 8 文件和 manifest SHA-256 `fe56b2b9d394ee5e3e34e96514485a45daeb5f9ad9207ed2dbe214c86c20d98f`；步骤 7 窄测试 `40 passed`，全部徘徊测试 `148 passed, 104 subtests passed`，完整测试 `526 passed, 1 failed, 171 subtests passed`，唯一失败仍为跌倒模块固定视频缺失。完整实现、故障覆盖、产物哈希和复现边界见[步骤 7 报告](../../../reports/mental_health/wandering_step7/README.md)。

证据边界没有扩大到现实效果：RF 与 TCN 仍是 `comparison_only` 且概率未经校准；步骤 7 只证明 `synthetic_camera_contract`，步骤 8 只增加 `synthetic_camera_corruption` 下的确定性增强、Camera QC 压力测试和冻结模型兼容性报告。没有授权 MP4/目标摄像头真值，不报告摄像头准确率、召回率或误报率。该子包仍未从本模块顶层重新导出，也未接入现有聚合、评分或运行时；TopoWander-MPT、校准/OOD、episode、日级徘徊字段、共享心理风险主链和正式现实效果指标仍未实现。

方案步骤 8 已按 v3 关闭。诊断 v1 保持 `diagnostic_invalid_pacing_gate`；v2 因 train WP `direct/medium selected=34`、`pacing/medium selected=26` 低于每层 50 而固定为 `diagnostic_quota_blocked_precommit`，没有 v2 bundle。v3 config SHA-256 为 `b6f6b682b2c4b908a83c60db6da04f39cfbab4e92fe748b8e6acaf84752c89d8`；正式 A 为 `0e255e0f89493c1ad389ac6d183c82843cbe0efc2978d5be7fbae66ce8dc4b1a`，两次全新目录的六个文件逐字节一致，含 1,491 个 train pairs、834 个 frozen validation pressure views、240 个 gated faults 和 48 个 limitation controls。负责人实际复核固定 440 对并签署，V/H 分别为 `31dcff92d3114d237b304953f467f20dd070e46091f08c4dde595f1f5faa6205`、`d9390010ec5dbb873945467f30bd4a05715342f9b446510ab1de8309bbc2be3a`。正式 C 为 `e27aa0716bb1cd39f6c37ac98c31b3d201d08b7673f61aadaa2ba1355f58bdbf`，绑定 A/V/H/RF/TCN，生成 4,144 条预测、24 个分组和 12 条 `model_compatibility_warning`。WP RF 二分类 low/medium 标签一致率为 0.990/0.953；RF 四分类为 0.950/0.844；TCN 二分类为 0.861/0.797，TCN 四分类为 0.545/0.578。主预注册聚合包含所有 Camera-QC-ready pressure，包括 semantic accepted 与 rejected；因此准确结论是“TCN，尤其四分类 TCN，在冻结通用压力协议下明显敏感，RF 二分类在主聚合下相对最稳”，不能把全部 warning 简化成纯 label-preserving 视图排名。warning 保持原样，不通过调参消除。机器指标见[步骤 8 compatibility metrics](../../../reports/mental_health/wandering_step8/compatibility/v3/metrics.json)。

下一项算法任务不是步骤 8a，而是已在技术方案冻结、尚未实现的步骤 9 P0。它的实现文件只新增 TopoWander-MPT exact config、`model.py` 和 `test_wandering_model.py`，完成后另写复现 README；用手算 fixture 验证三输入、mask-aware TCN、19 个语义 patch、relation-bias Transformer、分层 logits、梯度和安全 NPZ。不读取任何数据 bundle、不训练、不生成性能指标。完成状态最多为 `step9_forward_contract_verified`。开始代码前先由项目负责人决定步骤 8 的版本库检查点，避免把仍未跟踪的步骤 8 与步骤 9 混在一起。

版本控制现场：步骤 2–7 已由提交 `94ca337` 纳入 `feat/wandering-data-pipeline`，当前 HEAD `33e5380` 与远端同名分支同步。步骤 8 的 v1/v2/v3 配置、三个模块、三个脚本和三份测试仍未跟踪，现行 Markdown 有修改；正式 `augmentation/v3/`、确定性复建、visual/H、compatibility 报告及两次失效证据归档均已存在。步骤 8 三组窄测 `42 passed`，全部徘徊测试 `190 passed, 104 subtests passed`；完整测试 `568 passed, 1 failed, 171 subtests passed`，唯一失败仍是跌倒模块固定视频 `data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi` 缺失。下一位协作者不得清理正式或归档证据、回改 warning、关闭 loader 校验或把合成兼容性写成目标摄像头效果；暂存或提交仍需项目负责人明确决定，并在前后复核全部固定输入与 manifest SHA-256。

## 徘徊步骤 7 复现命令（已完成）

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_adapter.py \
  tests/test_wandering_camera_qc.py \
  tests/test_wandering_camera_inference.py -q

conda run -n eldercare-ai python scripts/wandering/run_camera_inference.py \
  --config configs/modules/wandering_camera_v1.yaml \
  --project-root . \
  --tracking-jsonl "<tracking-jsonl>" \
  --media-sidecar "<wandering-media-v1.json>" \
  --rf-development-dir reports/mental_health/wandering_step5/development/v1 \
  --expected-rf-development-manifest-sha256 fff6340e868de32bee2021ec1000f166b8caabe5caeb1132abb8ab822bfaaaf2 \
  --tcn-development-dir reports/mental_health/wandering_step6/development/v1 \
  --expected-tcn-development-manifest-sha256 0f4c48d948f0f4355dd577c89330b050ecb1bb513ca83ef1b25234897e27a10e \
  --output "<new-step7-output-directory>"
```

输出目录必须事先不存在。tracking+sidecar 是确定性验收事实输入；MP4 模式只保证其下游 camera core 可回放，不承诺不同依赖、权重或硬件生成逐字节相同的原始 tracking。

## 徘徊步骤 4 复现命令（已完成）

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_anchorless_normalization.py \
  tests/test_wandering_preprocessing.py \
  tests/test_wandering_topology.py -q

conda run -n eldercare-ai python scripts/wandering/build_preprocessing.py \
  --config configs/data/wandering_preprocessing_v1.yaml \
  --project-root . \
  --output "<new-preprocessing-output-directory>"

conda run -n eldercare-ai python scripts/wandering/visualize_preprocessing.py \
  --config configs/data/wandering_preprocessing_v1.yaml \
  --project-root . \
  --bundle "<new-preprocessing-output-directory>" \
  --output "<new-step4-review-directory>"
```

两个输出目录必须事先不存在。visualizer 只使用 builder 报告中的固定抽样 ID，不接受重新抽样参数。

## 徘徊步骤 3 命令

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_manifests.py \
  tests/test_wandering_splits.py -q

conda run -n eldercare-ai python scripts/wandering/build_split.py \
  --config configs/data/wandering_split_v1.yaml \
  --project-root . \
  --output data/splits/mental_health/wandering/v1
```

构建输出必须是不存在的新目录。预处理和训练脚本后续必须显式读取冻结的 `split.json`，不得现场切分。

## 徘徊步骤 2 命令

所有命令必须在当前仓库的 `eldercare-ai` 环境中运行，并使用不存在的新输出目录；转换器不会覆盖现有目录。WanderingPatterns pickle 入口只支持 Linux/WSL 的 `unshare` 隔离。

```bash
conda run -n eldercare-ai python scripts/wandering/convert_smartcare.py \
  --source-root "<SmartCare source root>" \
  --output "<new SmartCare output>"

conda run -n eldercare-ai python scripts/wandering/convert_wandering_patterns.py \
  --source-root "<WanderingPatterns source root>" \
  --output "<new WanderingPatterns output>"

conda run -n eldercare-ai python scripts/wandering/visualize_converted.py \
  --input "<strict JSONL>" \
  --source-name "<source name>" \
  --input-role "<role>" \
  --output "<new visual-review output>" \
  --samples-per-class 20 \
  --seed 20260801
```

## 数据来源与身份前提

行为输入复用现有 YOLOv8 Pose、ByteTrack、`PoseObservation` 和姿态质量控制结果，不创建第二套人体检测、姿态模型或实时循环。每条记录必须包含上游已经绑定的非空业务 `person_id`。`track_id` 只表示单路视频内关联，不能代替 `person_id`，也不能用于跨设备合并。

单设备记录可以省略 `device_id`；进入日级时间线的同一人员记录只要出现具名设备，其他记录也必须提供 `device_id`。同一时刻的多设备记录先按有效性、姿态质量和稳定设备键保留一个来源；异步区间仍重叠时，再按同样原则选择唯一来源。重叠会产生 `overlapping_device_observations` 标记，且不会直接制造场景转移。

## 绝对时间契约

行为记录必须通过以下一种方式得到事件时间：

- 带时区的 ISO-8601 `observed_at`；
- 带时区的 ISO-8601 `session_start_time` 加有限、非负的 `timestamp_sec`。

两者同时提供时以 `observed_at` 为准，并按配置容差核对推导时间。冲突记录保留在合理观测时长中，但不会进入有效观测时长。无时区、非法时间或只有相对秒的记录不会被猜测归入某个自然日；适配器会保留不可用标记，日级聚合跳过这些记录且不把标记归到其他日期。某人员完全没有有效绝对时间时快速失败。所有分桶先转换到 `aggregation.timezone`，默认 `Asia/Shanghai`；跨午夜区间在本地午夜拆分。

事件时间取当前人员和评估日最后一条合法输入的 `observed_at`。当没有合法事件时间时，CLI 要求显式传入带时区的 `evaluation_time`；它只可作为 `insufficient_data` 事件的时间回退，不能替代可评分风险结论的事件时间，也不会读取系统当前时间。

## 行为字段与计算

输入沿用命名关键点列表，每个关键点包含 `name`、归一化 `x/y` 和 `score`。若已经经过姿态质量控制，聚合器优先使用有效的 `x_smooth/y_smooth`，并保留上游质量标记。像素坐标、非有限坐标、低质量端点和被质量控制拒绝的记录不能进入有效时长。

聚合按 `person_id + date` 输出，计算规则如下：

- `observation_seconds`：只累加同一人员、同一设备、相邻且 `0 < delta <= max_gap_seconds` 的区间；不使用首尾时间差。
- `valid_observation_seconds`：区间两端质量合格、坐标有效、未被质量控制拒绝，且共同可见核心点数达标时才累加。
- `normalized_motion_proxy`：共同可见核心点的归一化欧氏位移中位数除以区间秒数。它是图像运动代理量，不是真实米/秒。
- `activity_volume`：有效区间的 `normalized_motion_proxy * delta_seconds` 之和。
- `active_ratio`：有效活动秒数除以有效观测秒数；分母为零时为 `None`。
- `nighttime_activity_ratio`：夜间有效活动秒数除以夜间有效观测秒数；默认夜间为本地时间 `[22:00, 06:00)`。
- `observation_coverage`：有效观测秒数除以合理观测秒数，不等同于后续评分阶段的模态覆盖率。
- `scene_region_distribution`：采用左端点场景的有效秒数分布；未知场景不产生转移。
- `scene_transition_count`：只统计质量有效、时间连续、无设备重叠的相邻记录场景变化。

当分母为零时，相应比例返回 `None` 并添加机器可读质量标记；缺失数据不填成正常值 `0`。

## 睡眠适配

睡眠适配只是标准字段校验，不代表已经接入萤石或其他真实设备协议。

| 字段 | 规则 |
|---|---|
| `person_id` | 非空业务人员 ID |
| `date` | 严格 `YYYY-MM-DD`；也可用带时区 `observed_at` / `timestamp` 转换 |
| `sleep_onset_latency` | 分钟，有限数值，`0-720`，可缺失 |
| `night_awakenings` | 整数，`0-100`，可缺失 |
| `sleep_efficiency` | 比例，`0.0-1.0`，可缺失；不接受或转换百分数 |
| `device_source` | 可选非空字符串 |
| `quality_score` | 可选有限数值，`0.0-1.0` |
| `quality_flags` | 可选非空字符串列表 |

缺失的三个睡眠指标保持 `None` 并写入质量标记。非法单位、范围或类型会抛出包含记录号和字段名的 `MentalHealthDataError`，不会静默裁剪。

## 可选自评输入

自评记录使用 JSON 或 JSONL，每条记录包含稳定 `person_id`、严格 `YYYY-MM-DD` 日期，或带时区的 `observed_at` / `timestamp`。可选分数字段为 `social_withdrawal_score`、`negative_affect_score` 和 `self_report_risk_score`，均必须是 `0.0-1.0` 的有限数值；`manual_emergency_flag` 必须是显式布尔值。缺失字段保持不可用，不默认成 0。

## 配置与调用

默认配置位于 `configs/modules/mental_health.yaml`：

```yaml
aggregation:
  timezone: Asia/Shanghai
  max_gap_seconds: 5.0
  timestamp_conflict_tolerance_seconds: 1.0
  min_keypoint_quality: 0.45
  min_common_core_keypoints: 4
  active_motion_threshold: 0.02
  night_start: "22:00"
  night_end: "06:00"

baseline:
  initial_days: 3
  stable_days: 7
  max_window_days: 14
  abnormal_score_threshold: 0.6

scoring:
  thresholds:
    level_1: 0.25
    level_2: 0.45
    level_3: 0.65
  min_persistent_days_for_level_3: 3
  passive_max_level: 3
```

这些值是算法原型的工程默认值，未经临床校准。Python 调用入口如下：

```python
from elderly_monitoring.modules.mental_health import (
    MentalHealthRiskPipeline,
    adapt_sleep_records,
    aggregate_daily_behavior,
    score_daily_mental_health,
)

daily_features = aggregate_daily_behavior(behavior_records)
sleep_records = adapt_sleep_records(raw_sleep_records)
baseline_features = score_daily_mental_health(history_days, current_days)
event = MentalHealthRiskPipeline().predict_from_features(baseline_features[0])
```

## 离线日级 CLI

历史行为和当前行为必须使用 JSONL，每个非空行是一个行为对象。睡眠和自评可使用单对象 JSON、对象数组 JSON 或 JSONL。睡眠/自评中与当前 `person_id + date` 匹配的记录进入当前日，其余记录作为历史模态；提供 `evaluation_time` 时，该自然日的可选模态也可建立当前人员日。

示例文件位于 `examples/features/mental_health_*`。以下命令只运行心理健康分支并输出 `module=mental_health` 事件：

```bash
PYTHONPATH=src conda run -n eldercare-ai python -m elderly_monitoring.inference.run_features \
  --module mental_health \
  --history-behavior examples/features/mental_health_history_behavior.jsonl \
  --current-behavior examples/features/mental_health_current_behavior.jsonl \
  --sleep examples/features/mental_health_sleep.json \
  --self-report examples/features/mental_health_self_report.json \
  --output /tmp/mental_health_daily.jsonl
```

省略 `--output` 时写到 stdout。当前行为没有合法事件时间时必须增加例如 `--evaluation-time 2026-07-04T18:30:00+08:00`。旧的单特征入口仍保留：

```bash
PYTHONPATH=src conda run -n eldercare-ai python -m elderly_monitoring.inference.run_features \
  --module mental_health \
  --input examples/features/mental_health_sample.json
```

日级输出按 `date + person_id` 排序，每个 `person_id + date` 一行：

```json
{"person_id":"p01","date":"2026-07-04","daily_features":{},"baseline_features":{},"event":{}}
```

`daily_features` 是行为、睡眠和自评的当日标准化结果；`baseline_features` 包含个人历史偏离、持续天数、覆盖率、质量与两个独立窗口；`event` 是统一 `AlgorithmEvent`，且 `module` 固定为 `mental_health`。输出不包含当前处理时间。相同输入、配置和代码产生字节级稳定 JSONL。所有文件会先完整解析、验证和计算；任一 JSONL 记录错误会报告文件、行号和字段，退出码非 0，输出路径采用临时文件原子替换，不留下部分结果。

## 个人基线与偏离分

基线按业务 `person_id` 和自然日独立建立，当前评估日不会进入自身基线。同一自然日重复输入只计为一个历史日；初始基线需要 3 个合格自然日，7 日后视为稳定，统计窗口最多保留最近 14 个合格自然日。

每个标量特征同时计算风险方向上的标准化差异、相对变化和分位数越界，取三者最大值。活动量和活跃比例只对下降评分；入睡潜伏期和夜间觉醒只对上升评分；睡眠效率只对下降评分；夜间活动比例和场景转移次数使用双侧偏离。零方差历史使用相对与绝对缩放下限，避免真实突变被零标准差掩盖。

`persistent_abnormal_days` 只累计达到异常阈值的连续、合格自然日。日历日期缺失或当日质量不合格会中断证据链，不会被当作正常日，也不会增加持续天数。`baseline_window` 记录历史参考范围，`evidence_window` 单独记录当前结论的连续异常范围。

## 缺失模态与风险评分

默认覆盖率分母只包含活动下降、睡眠扰动和规律性偏离三个预期特征，并按 YAML 中对应权重计算。可选的社交退缩、负面情感和自评分数仅在合法提供时参与风险评分；缺失值保持不可用，不以 0 分参与。风险总分只在当前可用特征之间重新归一化权重。

普通风险等级先按 YAML 阈值得到 0-3 级候选值，再依次应用覆盖率、基线成熟度和持续天数上限。纯摄像头与睡眠数据最高为 3 级。4 级只允许由达到阈值的合法自评风险分或显式 `manual_emergency_flag: true` 触发，3-4 级建议动作均为 `manual_review`。强证据不会被被动数据上限降级，但覆盖不足、基线不足和持续证据不足仍会降低置信度并写入 metadata。

没有任何可评分特征时，为兼容公共 schema 输出 `risk_score=0.0`，同时固定为 `risk_level=0`、`confidence=0.0` 和 `trigger_event=insufficient_data`；该兼容值不表示已经判断为正常。置信度表示特征覆盖、基线质量和持续性证据的支持程度，不表示医学结论正确率。

## 当前边界

- 不从摄像头推断负面情绪、社交退缩或医学状态。
- 不实现跨镜 ReID、人脸识别或身份数据库。
- 不实现真实睡眠设备协议、外部 API、数据库、推送或处置系统。
- 尚未经过真实设备数据、临床标签或临床有效性验证。
