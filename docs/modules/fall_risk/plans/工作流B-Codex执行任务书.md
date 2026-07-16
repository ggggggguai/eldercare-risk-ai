# 工作流 B：算法增强与实验矩阵 Codex 执行任务书

你是一名负责时序行为建模、机器学习实验和模型验证的高级 Python 工程师。请在以下仓库中直接实施工作流 B，而不是只输出建议、罗列候选模型或重新写一份计划：

```text
/Users/guai/Documents/project/挑战杯/algorithm
```

## 目标

在工作流 A 提供的可追溯数据、冻结 split 和评价协议之上，建立一套质量感知、无数据泄漏、可复现、可校准、可消融的跌倒风险算法实验链路，并形成一个可交给工作流 C 评估实时集成的冻结候选。

最终自动化交付版本名为 `fall-risk-workflow-b-v1`。它表示实验基础设施和冻结候选流程完成，不表示复杂模型必然优于规则 baseline，也不表示任何内部冲奖目标已经达到。

工作流 B 必须分别处理四类证据：

```text
跌倒与近跌倒事件
步态与坐站功能评估
个体基线纵向变化
同人同时段配对风险融合（仅在真实配对数据存在时）
```

四类证据不得混成一个数据集、一个 split 或一个综合 Accuracy。只有自动化工具、测试、验证集实验、校准、消融、复现证据和冻结交接全部通过，且人工或数据依赖被明确完成或明确列为阻塞时，才能声明相应阶段完成。

## Context (carry forward)

- 项目使用 Python 3.11、`src/` 布局和 conda 环境 `eldercare-ai`。
- 所有 Python 脚本和测试必须通过 `conda run -n eldercare-ai python ...` 执行，禁止使用裸 `python`、裸 `pytest` 或 base 环境。
- 跌倒风险和心理健康风险必须独立评分、独立验证、独立输出。本任务只处理 `fall_risk`，不得修改心理健康模块行为，也不得增加跨模块综合风险。
- 工作流 B 的原始要求位于 `docs/modules/fall_risk/plans/挑战杯揭榜挂帅冲奖增强计划.md` 第 6 节，并受其中 G0-G4 门槛、四类证据族和模型升级决策规则约束。
- 工作流 A 是正式实验的硬前置。没有可追溯 manifest、`reviewed/final` 标签、分组 split、泄漏检查和冻结评价协议时，只能实现基础设施、合成测试和阻塞报告，不得训练正式候选或生成比赛指标。
- 当前能力和限制以模块 README、工程架构、任务清单、实际代码和实际数据为准。计划文档和模型候选清单不能作为“已经实现”或“已经有效”的证据。
- 当前仓库可能有用户尚未提交的修改，且工作流 A 可能正在并行实施。必须先检查 `git status`、相关 diff 和新文件；不得回退、覆盖或格式化与本任务无关的用户修改。
- 测试集不得用于选型。模型、超参数、阈值、校准器和拒识策略只能用训练数据、分组交叉验证和冻结验证集确定；冻结测试集只能由独立保管人对冻结候选运行一次。
- “共享缓存”有两层含义：
  - 下游模型公平比较必须使用同一个冻结上游缓存。
  - 上游姿态或平滑消融必须在同一原始输入和同一协议下分别生成版本化缓存，每轮只替换一个上游因素。
- 个体基线持久化只指算法侧本地版本化状态、序列化接口和离线产物。本仓库不实现业务数据库、账号、设备管理、消息、工单或人工处置。
- 2026-07-15 的已知观察值如下，但开始实施时必须重新核验，不能盲信：
  - YOLOv8-Pose + ByteTrack、RTMPose 适配器、姿态质量控制、步态、坐站、近跌倒、个人基线和规则融合已有 baseline。
  - 当前姿态平滑主要是固定 alpha 的 EMA 类实现；尚未形成 EMA、One Euro 和 Kalman 的公平对照框架。
  - 当前步态、坐站、近跌倒和个人基线输出均为未使用真实老人风险标签校准的规则或统计 proxy。
  - 当前规则融合主要消费若干 0-1 局部分数；缺失值在部分路径中可能被当成 0，不能直接作为缺失感知学习模型的训练输入。
  - 尚未看到统一 `[T,V,C]`/特征时序缓存、训练实验注册表、校准报告、分支消融报告或冻结模型候选。
  - `pyproject.toml` 的 `ml` 可选依赖包含 NumPy、scikit-learn 和 LightGBM；EBM、PyOD、ruptures、骨架网络框架和完整 RTMPose 运行依赖未作为标准环境的一部分锁定。
  - 本机偶然已安装的包不等于可复现项目依赖。未写入 `environment.yml`/`pyproject.toml` 并验证的依赖不得成为正式主线的隐含前提。

## 开始前必须阅读

完整阅读以下文件后再设计或编辑代码：

```text
AGENTS.md
README.md
environment.yml
environment-reference.txt
pyproject.toml
.gitignore
docs/README.md
docs/architecture/算法工程骨架.md
docs/interfaces/算法事件输出接口.md
docs/tasks/README.md
docs/modules/fall_risk/README.md
docs/modules/fall_risk/plans/挑战杯揭榜挂帅冲奖增强计划.md
docs/modules/fall_risk/plans/工作流A-Codex执行任务书.md
docs/modules/fall_risk/plans/跌倒风险算法研发计划.md
docs/modules/fall_risk/plans/跌倒风险各任务模型调研与选型矩阵.md
docs/modules/fall_risk/data/数据集标注规范.md
docs/modules/fall_risk/data/跌倒风险标签字典.md
data/annotations/fall_risk/README.md
scripts/collect/README.md
scripts/evaluate/README.md
reports/README.md
```

还必须检查以下实现、配置、脚本和测试：

```text
configs/modules/fall_risk.yaml
configs/evaluation/
src/elderly_monitoring/modules/fall_risk/pose.py
src/elderly_monitoring/modules/fall_risk/pose_quality.py
src/elderly_monitoring/modules/fall_risk/gait.py
src/elderly_monitoring/modules/fall_risk/sit_stand.py
src/elderly_monitoring/modules/fall_risk/near_fall.py
src/elderly_monitoring/modules/fall_risk/baseline.py
src/elderly_monitoring/modules/fall_risk/features.py
src/elderly_monitoring/modules/fall_risk/pipeline.py
src/elderly_monitoring/modules/fall_risk/evaluation.py
src/elderly_monitoring/modules/fall_risk/data_manifest.py
src/elderly_monitoring/modules/fall_risk/splits.py
src/elderly_monitoring/runtime/feature_assembly.py
configs/data/
tests/test_fall_risk_pose.py
tests/test_fall_risk_pose_quality.py
tests/test_fall_risk_gait.py
tests/test_fall_risk_sit_stand.py
tests/test_fall_risk_near_fall.py
tests/test_fall_risk_baseline.py
tests/test_fall_risk_pipeline.py
tests/test_fall_risk_event_evaluation.py
tests/test_fall_risk_data_manifest.py
tests/test_fall_risk_splits.py
```

若文件尚不存在，记录为事实，不要创建空占位文件来假装前置工作已完成。

## 执行方式

1. 先重新核验仓库、环境、数据、标签、split、协议和依赖现状，输出精简实施计划，再立即进入不受阻塞的实现。
2. 采用测试驱动：每个行为变更先补失败测试，再实现最小正确改动。
3. 每完成一个阶段，向用户更新：`完成内容 / 修改文件 / 验证结果 / 数据门禁 / 剩余风险`。
4. 先运行最相关的窄范围测试；阶段完成后运行完整测试集和规定的真实视频烟测。
5. 遇到失败时先定位根因并保留证据，不得通过放宽校验、临时 `PYTHONPATH`、删除困难样本、改测试、改变 split 或降低指标口径来掩盖问题。
6. 每个实验必须由配置驱动，记录代码、环境、数据、split、缓存、配置和预测 hash；禁止把关键阈值、随机种子或特征列表只藏在代码中。
7. 每轮只改变一个因素。上游感知、平滑、分支模型、融合模型和校准器不得同时替换后再把变化归因给其中某一项。
8. 自动化阶段之间应持续执行。只有命中本文停止条件时才暂停相关动作并请求用户决定；不受影响的基础设施工作继续完成。

## 允许修改的范围

优先复用现有模块和目录。可以在确有必要时修改或新增：

```text
src/elderly_monitoring/modules/fall_risk/pose.py
src/elderly_monitoring/modules/fall_risk/pose_quality.py
src/elderly_monitoring/modules/fall_risk/gait.py
src/elderly_monitoring/modules/fall_risk/sit_stand.py
src/elderly_monitoring/modules/fall_risk/near_fall.py
src/elderly_monitoring/modules/fall_risk/baseline.py
src/elderly_monitoring/modules/fall_risk/features.py
src/elderly_monitoring/modules/fall_risk/pipeline.py
src/elderly_monitoring/modules/fall_risk/temporal_samples.py
src/elderly_monitoring/modules/fall_risk/perception_benchmark.py
src/elderly_monitoring/modules/fall_risk/model_candidate.py
src/elderly_monitoring/modules/fall_risk/experiments/
scripts/collect/                                # 仅统一缓存和离线算法产物
scripts/train/                                  # 仅离线训练、校准和候选冻结
scripts/evaluate/                               # 只做预测评估、感知对照、消融汇总和盲测工具
configs/experiments/fall_risk/
configs/evaluation/                             # 只新增 B 所需协议，不覆盖 A 的冻结协议
tests/
data/processed/fall_risk/temporal_cache/        # 本地生成，受 .gitignore 管理
data/processed/fall_risk/raw_observations/      # 本地生成，受 .gitignore 管理
data/processed/fall_risk/task_bundles/          # 本地生成，受 .gitignore 管理
data/processed/fall_risk/models/                # 本地生成，受 .gitignore 管理
reports/fall_risk/
reports/reproducibility/
docs/modules/fall_risk/README.md
docs/architecture/算法工程骨架.md
docs/tasks/README.md
docs/README.md
```

若已有等价模块，必须复用并避免创建平行实现。只有确实减少复杂度或匹配现有结构时才新增模块；不要为了“框架完整”创建无行为的注册表、基类、插件系统或占位模型。

`src/elderly_monitoring/runtime/` 和 `src/elderly_monitoring/service/` 属于工作流 C 的实时工程范围。工作流 B 只能读取它们以核对输入输出和延迟语义；如需修改，必须先满足冻结候选门槛并请求用户批准。

工作流 B 的所有新模型、平滑器和实验实现必须显式 opt-in。现有公开函数签名、默认配置、规则阈值和默认输出必须通过 golden regression 保持兼容；不得因为修改了被实时层直接导入的领域模块，就让候选模型或新行为静默进入实时默认路径。实时默认 scorer 的切换只由工作流 C 在门禁通过并获得批准后实施。

## 禁止事项

- 不得修改 `src/elderly_monitoring/modules/mental_health/`、心理健康配置或其接口行为。
- 不得增加综合模块、跨模块融合分数或跨模块动作协调逻辑。
- 不得在工作流 A 未就绪时绕过门禁训练“正式模型”或生成比赛结论。
- 不得读取冻结测试标签来选择模型、特征、超参数、阈值、校准器、拒识阈值或失败案例修复方向。
- 不得按窗口随机划分；同一人员、家庭、原视频、原事件、多机位、相邻窗口或内容副本不得跨集合。
- 不得把不同数据集中的不同受试者拼成所谓多模态或多分支配对样本。
- 不得把跌倒视频事件标签当作长期风险、功能能力或临床结局真值。
- 不得把算法规则输出当作人工真值，也不得用同一规则生成标签后再证明该规则有效。
- 不得把 `pending`、`uncertain`、源文件缺失、许可不明或未冻结 split 的记录用于正式训练、校准或指标。
- 不得把低质量、缺失、插值失败、ID 冲突或相机移动静默填成正常值；训练和推理必须保留显式 mask 与拒识状态。
- scaler、imputer、特征选择、异常检测器和基线统计只能在 train 拟合；校准器只能使用 train 内 OOF 或 train 内独立 calibration 子集；val/test 只能按冻结对象 transform 或 predict。
- 不得让个体历史特征使用预测时点之后的数据；面向实时的候选不得使用未计入延迟的未来帧、双向平滑或未来窗口统计。
- 不得把无关键点真值时的抖动、骨长稳定性或覆盖率写成“关键点准确率”。
- 没有家具/扶手/床边 ROI 和接触标签时，腕部证据必须继续命名为 `support_contact_proxy`，不得升级为真实接触判断。
- 不得为了冲奖目标反复查看测试集、删除负结果、隐藏困难子集或重划数据。
- 不得上传数据、日志、关键点、模型输入或标注到外部服务。
- 未经用户批准，不得下载数据或权重、安装或升级依赖、修改 `environment.yml`/`pyproject.toml`、改 `AlgorithmEvent` schema、创建提交或推送远端。
- 不得在报告、配置、模型元数据、测试夹具或最终回答中暴露邮箱、令牌、直播地址、真实姓名、原始人脸或其他身份信息。

## 阶段 0：基线核验与数据门禁

### 0.1 仓库与环境保护

1. 运行 `git status --short`，阅读相关 diff，识别既有用户修改和工作流 A 的在途产物。
2. 运行：

   ```bash
   conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
   ```

   editable 安装必须指向当前仓库；若不是，按 `AGENTS.md` 使用当前项目重新安装后再验证。
3. 核对 `environment.yml`、`pyproject.toml` 与实际导入能力，区分“标准环境声明”“本机偶然安装”和“候选模型缺失依赖”。
4. 记录当前规则 baseline、模型版本、配置 hash、现有测试结果和本地缓存，不覆盖任何原始或冻结产物。
5. 对本次实验实际使用的源码和配置生成规范化 `reports/reproducibility/source_manifest.json`：记录相对路径、文件 SHA-256、git commit 和工作树状态，并对整个清单生成 `source_state_hash`。未提交和相关未跟踪源码必须纳入清单，不能只写 `dirty=true`。
6. 从标准 conda 环境的显式包列表和项目环境定义生成脱敏环境锁 hash；不得把本机偶然安装但未声明的包静默纳入正式候选。

### 0.2 四类证据就绪矩阵

对每个证据族分别检查并记录：

```text
manifest 路径与 hash
标签路径、schema、review_status 和独立真值来源
观察单位、label horizon 和任务语义
train/val/test split ID、分组键和泄漏检查结果
评价协议路径、状态和 hash
规则 baseline 预测路径与 hash
连续监控时长或纵向观察时长的合法分母
可用特征、质量字段、缺失比例和上游版本
允许运行的候选与缺失依赖
```

当前 Codex 只能读取 train/val 标签。test 标签的 schema、数量、复核状态、真值来源和 hash 必须由独立保管人通过不含逐样本标签的聚合审计文件提供。若 train/val/test 标签仍混在一个 Codex 可读文件中，必须停止读取并要求保管人完成物理隔离；不得先打开全文件、再按 split 过滤后声称测试标签未暴露。

同时冻结工作流 A 到 B 的 join 契约，至少包含：

```text
canonical_task_id
asset_id
assignment_id
partition                  # train / val / test 的唯一规范字段
split_id
subject_id / source_group_id / original_event_id
label_id（train/val 可见；test 只由保管人持有）
```

不得在 `task_type`、配置协议 ID 和 split ID 之间混用 `fall_event`、`fall_event_v1` 等近似名称。若工作流 A 当前字段仍是 `split_name`、`partition` 或其他命名，先选定唯一规范字段并提供显式兼容读取和冲突失败测试，不能靠字符串猜测关联。

每个证据族只能处于以下状态之一：

| 状态 | 含义 | 允许动作 |
|---|---|---|
| `ready_for_validation` | 标签、split 和协议可追溯，规则 baseline 实现路径明确 | 先生成 baseline；在 train 内层开发后，对冻结 val 运行一次短名单比较 |
| `infrastructure_only` | schema 或数据存在，但正式标签、split 或协议未冻结 | 只实现工具、合成测试和开发烟测，不输出实测结论 |
| `blocked` | 数据来源、许可、真值、时间轴或泄漏治理不可靠 | 只记录阻塞，不消费该数据 |

必须生成：

```text
reports/fall_risk/workflow_b_readiness.md
reports/fall_risk/workflow_b_blockers.md
```

阻塞报告只写经过核验的事实，至少包含责任角色、所需输入、解除证据和解除后的准确命令，不写推测。

### 0.3 预注册项

在运行任何正式候选前，逐证据族确认以下值来自冻结配置，而不是 Codex 临时选择：

```text
主指标和次指标
观察单位和正负类定义
事件 IoU / onset 容忍 / 合并与复位规则
阈值选择规则和固定召回或固定误报工作点
label horizon、预测截止时间和因果性要求
分组 bootstrap 单位、次数、随机种子和最小独立组数
缺失数据和拒识计数方式
候选比较与并列决策规则
实时延迟预算和目标硬件
```

缺失项使用 `null` 和明确 blocker；不得悄悄填入“常用默认值”后称为预注册协议。

退出条件：环境绑定正确，四类证据各有状态和证据，冻结测试集未暴露，允许继续的阶段清晰，用户资产未被修改。

## 阶段 1：统一时序样本与版本化缓存

实现一个确定性的时序样本构建器。推荐落在：

```text
src/elderly_monitoring/modules/fall_risk/temporal_samples.py
scripts/collect/build_fall_temporal_cache.py
configs/experiments/fall_risk/temporal_cache_v1.yaml
tests/test_fall_risk_temporal_samples.py
```

如果仓库已有同等职责实现，复用现有路径，不创建重复入口。

### 1.1 缓存布局

缓存必须拆成三层，不能把标签、split 和上游张量混在一个索引中：

```text
data/processed/fall_risk/raw_observations/<raw_cache_id>/
  raw_manifest.json
  detections.jsonl
  poses.jsonl

data/processed/fall_risk/temporal_cache/<cache_id>/
  cache_manifest.json
  feature_index.jsonl
  shards/
    <shard_id>.npz

data/processed/fall_risk/task_bundles/<task_bundle_id>/
  task_manifest.json
  samples_index.jsonl
```

`raw_cache_id` 由源 manifest、帧选择协议、检测/跟踪/姿态后端、权重 hash、上游配置、`source_state_hash` 和环境锁 hash 决定。平滑器不得进入 raw cache；同一 raw pose 才能公平比较多个平滑器。

`cache_id` 只标识标签无关的时序特征，由 `raw_cache_id`、平滑器、重采样/特征配置、`source_state_hash` 和环境锁 hash 决定。它不得包含 split、标签版本或真值边界。

`task_bundle_id` 由 `cache_id`、canonical task ID、split ID、标签版本/保管人提供的密封 test 标签 hash、取窗协议和评价协议共同决定。现有 raw cache、feature cache 和 task bundle 均不得原地覆盖。

`feature_index.jsonl` 每条记录至少包含：

```text
feature_record_id
asset_id
subject_id / source_group_id / original_event_id
video_id / track_id
source_start_time / source_end_time
scene_region / view
tensor_path / tensor_key / tensor_hash / shape / dtype
joint_order / bone_edges
target_hz / window_sec / causal
coordinate_system / scale_source
pose_backend / tracker_backend / smoother
upstream_model_version / upstream_config_hash
manifest_hash / source_state_hash / environment_lock_hash
quality_summary / missing_ratio / interpolation_ratio / id_switch_flags
```

`samples_index.jsonl` 每条任务记录至少包含：

```text
sample_id
canonical_task_id / evidence_family
asset_id / assignment_id / feature_record_ids
subject_id / source_group_id / original_event_id
start_time / end_time / prediction_cutoff_time
sampling_strategy
partition / split_id
label_ref / label_version             # 仅 train/val 对 Codex 可见
cache_id / task_bundle_id
```

标签引用和标签派生字段不得混入 raw observation 或 temporal feature 张量。train/val 可按冻结协议使用真值构造监督样本；test 预测取窗必须来自完整流、固定时间网格或冻结候选生成器，绝不能使用 test 真值事件边界。test 真值只由独立保管人在预测完成后用于匹配评估。

### 1.2 张量契约

固定 COCO 17 点顺序，并在配置和缓存中显式记录。至少保存：

```text
timestamps          [T]
joint               [T, V, 3]   # x_norm, y_norm, confidence
joint_body          [T, V, 2]   # 可靠时使用身体尺度中心化坐标
body_scale          [T]
body_scale_mask     [T]
valid_mask          [T, V]
interpolated_mask   [T, V]
jump_mask           [T, V]
joint_motion        [T, V, 4]   # vx, vy, ax, ay，按真实秒计算
bone                [T, E, 3]   # dx, dy, length
bone_motion         [T, E, 2]
feature_sequence    [T, F]       # 仅特征级任务使用
feature_mask        [T, F]
```

要求：

- 所有样本按真实 `timestamp_sec` 重采样；不得用帧号差直接冒充时间差。
- `target_hz`、`window_sec`、步长、关节顺序、骨连接、通道顺序、dtype 和坐标系必须来自版本化配置。
- 原始归一化图像坐标必须保留。身体尺度使用躯干、人体框或经批准的场景标定；尺度不可用时设置 mask，不得除以任意常数。
- 速度和加速度按秒计算。时间戳重复、倒退、间隔异常或跨轨迹时必须拒绝或明确分段。
- 只允许在配置的短缺失范围内插值；长缺失保留为无效 mask。不得跨 ID switch、流 epoch、事件边界或预测截止时间插值。
- 实时候选必须使用 `causal=true` 的缓存和因果特征。离线双向平滑只能作为单独离线对照，并明确计入用途和附加延迟。
- 保存 confidence、valid、插值来源、异常跳变、ID 切换、上游模型和质量摘要；缺失不得变成全零“正常人”。
- 同一输入、配置和环境重复运行时，样本顺序、规范化内容 hash 和张量内容 hash 必须一致。
- task bundle 构建器必须通过 `asset_id/assignment_id` 关联工作流 A 的 `partition/split_id`，并断言人员、源组、原事件和内容 hash 无交叉；不得用文件名或近似 task 字符串猜测关联。

测试至少覆盖：非均匀时间戳重采样、真实秒差分、固定关节顺序、短缺失插值、长缺失 mask、ID switch 分段、因果截止、身体尺度缺失、raw/feature/task 三层 ID 变化规则、源码脏状态 hash、重复运行确定性、test 真值边界不可用于取窗，以及跨 split 泄漏拒绝。

退出条件：规则、传统模型和时序模型可读取同一冻结 task bundle；上游候选消融能从受控 raw cache 生成独立版本 feature cache；任何标签变化只改变 task bundle，不改变标签无关 cache；三层产物均可由 hash 复现。

## 阶段 2：上游感知与质量控制公平对照

### 2.1 固定工程 baseline

保留 `YOLOv8-Pose + ByteTrack + 当前 EMA` 为工程 baseline。不得因为模型矩阵中存在更多候选就顺手扩展检测器或跟踪器。

先冻结原视频到检测框、轨迹分配和未平滑关键点的 raw observation cache，再生成各平滑/特征版本。不得先平滑后才尝试重建“原始”对照。

先实现统一 benchmark 入口和配置，推荐：

```text
src/elderly_monitoring/modules/fall_risk/perception_benchmark.py
scripts/evaluate/run_fall_perception_benchmark.py
configs/experiments/fall_risk/perception_v1.yaml
tests/test_fall_risk_perception_benchmark.py
```

### 2.2 姿态后端对照

在同一原视频、同一人体目标协议、同一评估窗口和同一下游版本上比较：

```text
YOLOv8-Pose baseline
RTMPose-S
RTMPose-M（资源和依赖允许时）
```

姿态后端比较必须区分两种协议：

- `pose_only`：复用同一冻结人体框和轨迹分配，将同一 bbox 输入各姿态后端，用于归因姿态模型差异。
- `end_to_end`：各后端按其标准检测/姿态链运行，只能评价整条感知链，不能把差异单独归因于姿态模型。

若某后端无法消费相同 bbox，只能运行 `end_to_end` 并披露限制，不得把同时变化的检测器、姿态器和跟踪结果称为单变量对照。

有域内人工关键点真值时，可以报告 PCK/OKS 或预注册关键点误差；没有真值时，只能报告：

```text
核心点有效覆盖率
低质量和拒识比例
时序抖动与骨长变异 proxy
事件峰值保留和 onset 偏移
ID switch 后恢复情况
下游固定规则指标
P50/P95 单帧附加延迟、吞吐和内存
```

这些 proxy 不得命名为“姿态准确率”。RTMPose 依赖、配置、权重或许可未锁定时，先完成适配器测试和 blocker；不得自动下载或修改环境。

当前 RTMPose 适配器在缺少稳定 `track_id` 时可能使用帧内序号。它不能直接被视为 ByteTrack 的跨帧轨迹；公平对照前必须接入相同跟踪协议，或把比较明确限制为单帧姿态质量并披露该限制。

### 2.3 平滑器对照

固定同一份原始关键点缓存，比较：

```text
不平滑
当前 EMA
One Euro
Kalman
```

每个平滑器必须实现统一接口，保留原始坐标、平滑坐标、因果标志、参数和版本。评价至少包括：

```text
有真值时的关键点误差
无真值时的时序抖动和骨长变异 proxy
近跌倒/跌倒运动峰值衰减
onset 时间偏移
有效覆盖和拒识变化
附加 P50/P95 延迟
固定下游规则的事件指标变化
```

不能只因输出“更平滑”就选型。若平滑器降低抖动但削弱事件峰值或增加不可接受延迟，必须保留负结果。

### 2.4 尺度与接触证据

- 为速度、位移、摆动和下降特征提供显式尺度来源：身体尺度、人体框尺度或经批准的场景标定。
- 每条特征记录 `scale_source`、尺度值、有效性和版本；不得混用像素/帧、归一化坐标/秒和真实米制单位。
- 没有场景标定时只能输出 image-normalized 或 body-normalized proxy，不能写成米、米每秒或真实步幅。
- 家具、扶手、床边 ROI 和接触标签没有就绪时，不新增真实接触分类器；腕部逻辑保持 `support_contact_proxy`。

选型只在冻结验证集完成。若新姿态器或平滑器没有稳定增益，工程 baseline 保持不变。测试集不得用于决定是否替换。

退出条件：每个上游对照只改变一个因素，有完整配置、缓存 hash、指标、延迟和失败案例；没有真值的结论被准确限定。

## 阶段 3：分支增强实验

实现统一但不过度抽象的实验运行器，推荐入口：

```text
src/elderly_monitoring/modules/fall_risk/experiments/
scripts/train/train_fall_branch_model.py
scripts/evaluate/evaluate_fall_branch_model.py
tests/test_fall_risk_experiments.py
```

运行器必须支持：固定 task bundle、训练、分组内层验证、预测 JSONL、指标 bundle、配置/hash 记录和负结果保留。训练入口不得放入只负责指标评估的 `scripts/evaluate/`。不得用一个通用分类器接口抹平事件匹配、功能分层和纵向变化的不同语义。

### 3.1 跌倒与近跌倒事件证据族

事后跌倒和近跌倒必须分别形成预测、状态和指标，不能只实现近跌倒增强。

事后跌倒保留当前规则状态机 baseline，并核对以下可观测状态：

```text
suspected
confirmed_static
recovered
unresolved
```

算法不得自行输出 `confirmed_false_alarm`。必须评估疑似跌倒、倒地静止确认、恢复和超时未决的完整转换，并保存关联事件 ID、证据窗、质量和状态时间；标签不足时只完成状态机回归测试和 blocker。

近跌倒先保留当前高召回规则候选，再按数据条件比较：

```text
规则候选 baseline
规则候选 + Isolation Forest
规则候选 + One-Class SVM
规则候选 + hard-negative 分类器（存在独立正例和困难负例时）
ECOD（PyOD 获得批准并锁定后）
TCN / ST-GCN++（正例、事件边界和依赖达到门槛后）
```

要求：

- 单类模型只能用训练 split 中经审计的正常窗口拟合；污染比例、特征和 scaler 只从训练数据确定。
- hard-negative mining 只能使用 train 内层 group split 或 train OOF 错误；冻结 val 和 test 的失败案例均不得反馈训练。
- 弯腰、快速坐下、正常转身、下蹲、遮挡、多人、助行器和低光必须按冻结标签作为困难负样本分层。
- 异常分数只能称为 anomaly score；没有近跌倒正例和人工复核时不得校准成近跌倒概率。
- 恢复且未稳定倒地是近跌倒语义的一部分；事件输出必须保留 evidence window、event type、质量和恢复约束。

事后跌倒至少报告：事件级 Precision/Recall/F1、高风险事件 Recall、每摄像机小时 FP、疑似跌倒 detection latency、倒地静止确认延迟、恢复召回、错误恢复率、`unresolved` 比例、拒识覆盖和按人员/源组 bootstrap 95% CI。

近跌倒至少报告：事件级 Precision/Recall/F1、事件 PR-AUC、边界 IoU、onset detection latency、固定召回下 FP/摄像机小时、恢复约束正确率、拒识覆盖和按人员/源组 bootstrap 95% CI。

只有存在后续参考跌倒事件时才计算：

```text
lead_time = reference_event_start - first_level_3_or_higher_alert
```

正值表示提前，负值表示事件发生后才报警。没有后续参考跌倒事件时，近跌倒只报告相对自身 onset 的检测延迟，不得称为 pre-fall lead time。

### 3.2 步态与坐站功能证据族

步态按数据条件比较：

```text
可解释规则 baseline
Logistic Regression
LightGBM
EBM（依赖获批并锁定后）
轻量 TCN（标签、样本独立性和依赖达标后）
ST-GCN++ / CTR-GCN（只作数据充足后的增强）
```

坐站必须拆开三个任务：

```text
状态机事件定位
事件级局部能力/风险分层
逐帧相位分割
```

- 事件 TCN 只有在事件边界和独立功能 proxy 就绪时启动。
- MS-TCN++ 只有在逐帧相位标签就绪时启动；只有窗口标签时不得伪造逐帧监督。
- 功能模型必须按人员或家庭分组；同一人的相邻窗口不能跨集合。
- 只支持功能风险或 proxy 关联结论，不能由横断面 TUG/BBS/POMA/SPPB 结果声称“提前发现天数”。
- 2D 单目步速、步幅、角度和支撑接触必须保持 proxy 语义。

指标必须按子任务绑定：

- 步态和坐站局部能力/风险分层：AUROC、PR-AUC、Brier、预注册阈值下的灵敏度/特异度/F1、分层一致性、拒识覆盖和按人员 bootstrap 95% CI。Accuracy 只能作为次指标。
- 坐站事件定位：事件级 Precision/Recall/F1、边界 IoU、onset/offset 误差、重复/漏检和每观察小时 FP。
- 坐站逐帧相位分割：frame F1、segmental F1@10/25/50、edit score 和 mIoU；不得只报容易被长背景支配的 frame accuracy。

### 3.3 个体基线纵向证据族

在当前均值、标准差和分位数 baseline 上按顺序实现：

```text
Median / MAD
EWMA
CUSUM
PELT（ruptures 获批并锁定后）
混合效应模型（独立人数和重复测量达到门槛后）
```

必须定义并测试：

- `no_history`、`initializing`、`stable` 三个状态及其风险分/置信度上限。
- 只使用预测时点之前的历史；同日未来记录不得泄漏。
- 高风险事件、低质量窗口、相机移动、ID 冲突和上游版本变化不得写回正常基线。
- 日/小时聚合粒度、时区、自然日边界、昼夜、工作日/周末、场景条件和最小支持天数来自配置。
- 上游模型或特征版本变化后执行版本隔离、重建或经验证的迁移，不得无条件沿用旧分布。
- 输出当前值、期望范围、偏离方向/幅度、变化点、支持天数、质量和版本。

只有存在同一人员连续多日观测以及重复功能测量、专家确认的状态变化日期或前瞻结局时，才能评价纵向变化和提前发现。否则只完成算法、合成测试和 blocker，不得把规则自我偏离当真值。

指标至少包括变化检测 Precision/Recall、固定误报下召回、每人日误报、检测延迟或提前发现天数、缺失/拒识覆盖和按人员 bootstrap 95% CI。没有参考变化日期时不得计算提前量。

### 3.4 配对风险融合证据族

只有以下条件全部满足时才训练融合模型：

```text
同一 person_id
同一预注册观察时段
各输入来源和时间范围可追溯
独立 risk label 或前瞻结局
配对样本自己的 group split
质量、缺失 mask 和上游版本齐全
```

不同数据集、不同人员或不同时间段不满足上述条件时，融合状态必须为 `not_evaluable`；只实现 schema、校验器和 blocker，不得合成伪配对样本。

有配对数据时统一比较：

```text
固定规则评分卡
Logistic Regression
LightGBM
EBM / GAM（依赖获批并锁定后）
```

融合输入必须包含局部分支特征、局部分数、质量、缺失 mask、场景、基线支持度、来源和版本，不能只拼若干风险分数。缺失指示不得被模型默认解释为正常。

跌倒、倒地静止和强近跌倒规则继续作为安全覆盖。学习模型的 `risk_score`、`risk_level`、`confidence`、拒识和强规则覆盖必须语义一致；不得为了提高普通分类指标削弱 4 级紧急事件。

指标至少包括 PR-AUC、AUROC、Brier、ECE/reliability、固定误报下高风险召回、拒识覆盖、分层鲁棒性、P50/P95 离线推理延迟和按人员/家庭 bootstrap 95% CI。

退出条件：每个可评估证据族都有规则 baseline、至少一个合理轻量候选、相同 task bundle 下的公平对照和原始预测；不可评估证据族有明确 blocker 而不是伪结果。

## 阶段 4：校准、拒识、消融与冻结选型

### 4.1 防泄漏训练流程

所有候选必须遵守：

1. train 内层 group split/OOF 完成特征选择、hard-negative mining、超参数比较，并拟合特征处理、模型和异常检测器。
2. 校准使用 train 内分组 OOF 预测或 train 内独立分组 calibration 子集。
3. 冻结 val 只对预先列明的短名单候选运行一次，用于最终候选比较、阈值、拒识工作点和 guardrail 检查；val 结果不得反馈训练或扩大候选搜索。
4. test split 不参与以上任何步骤。
5. 时间任务的所有统计只使用预测截止时间之前的数据。

如果现有 split 不能提供无泄漏校准，停止正式校准并记录 blocker；不得把测试集临时改名为 calibration。

### 4.2 校准与拒识

对有概率语义的候选比较未校准、sigmoid 和 isotonic。样本不足以稳定使用 isotonic 时必须报告限制并保留更简单方案。

至少输出：

```text
Brier score
log loss（适用时）
ECE 和 reliability 数据
拒识覆盖率
coverage-risk / coverage-performance 曲线
低质量、缺失、域外和模态冲突分层
```

`confidence` 反映输入可靠性和模型适用性，不是医疗确定性。低质量样本必须被拒识、降置信或限制等级；不能通过填零看起来“正常且高置信”。

### 4.3 消融

按四类证据族分别执行，不制作跨数据集 A-F 总消融。至少包括：

```text
事件：去质量门控、去恢复约束、去 hard-negative 分支
功能：规则 vs 结构化特征模型 vs 时序模型；去 mask/质量特征
纵向：群体阈值 vs 个人基线；去污染保护；去条件基线
配对融合：去场景、去活动节律、去质量拒识、去个人基线（仅有配对数据时）
```

每次消融只改一个因素，固定 task bundle、训练预算、阈值规则和评价协议。没有配对融合数据时将融合消融标为 `not_evaluable`，不得从不同证据族拼表。

### 4.4 冻结候选决策

每个证据族的候选进入冻结交接必须同时满足：

- 在冻结验证集和预注册主指标上相对规则 baseline 有稳定、可重复的实际增益，或在指标相当时显著降低复杂度/延迟。
- 校准、拒识、困难负样本、关键场景和缺失分层没有不可接受退化。
- P95 离线推理延迟满足预注册预算，并在标准环境可复现。
- 模型、配置、特征 schema、上游缓存和所有依赖已锁定且许可可用。
- 强事件安全覆盖、机器可读风险因子和规则 fallback 保留。
- 负结果、失败案例和未达内部目标的指标完整保留。

“内部目标达到”不是选择过程中反复查看测试集的理由。若复杂模型没有稳定增益，冻结规则主线或更简单候选，并如实报告负结果。

### 4.5 模型 artifact 与候选 bundle

每个冻结候选必须生成可执行 artifact：

```text
data/processed/fall_risk/models/<candidate_id>/
  model_manifest.json
  feature_schema.json
  model.<approved_format>
  calibrator.<approved_format>       # 不适用时在 manifest 中为 null
  threshold_policy.yaml
  checksums.json
```

`model_manifest.json` 至少绑定：证据族、模型类、序列化格式、包版本、`source_state_hash`、环境锁 hash、权重 hash、feature schema hash、raw/cache/task bundle ID、split ID、训练/校准/阈值配置 hash、强规则覆盖、质量拒识和 fallback 版本。加载器必须先验证全部 hash，拒绝未知格式、缺失文件和 schema 不匹配；不得静默加载不受信任的 pickle 或任意代码对象。

在 `model_candidate.py` 提供最小稳定 API：加载候选、校验输入 schema、输出分数/概率/质量状态/拒识原因和模型版本。该 API 必须是显式 opt-in，不改变 `FallRiskPipeline` 的默认规则行为。

工作流 B 冻结的是一个版本化 `candidate_bundle`，不是假定存在一个覆盖四类证据的总模型。bundle manifest 必须逐证据族记录：`candidate_id`、规则 fallback、状态（`selected`/`rule_only`/`not_evaluable`）、是否属于工作流 C 的实时子集以及已知限制。

必须生成冻结摘要：

```text
reports/fall_risk/workflow_b_candidate_freeze.md
reports/reproducibility/model_and_config_hashes.md
```

摘要至少记录 `candidate_bundle_id`、逐证据族候选、代码 commit、`source_state_hash`、环境锁 hash、manifest hash、split ID、raw/cache/task bundle ID、训练/校准/阈值配置 hash、模型/权重 hash、验证集预测 hash、实时子集、已知限制和回退版本。

退出条件：选型完全基于 train 内层协议和一次冻结 val，candidate bundle、各 artifact 和规则 fallback 可重建，测试集仍未用于调参。

## 阶段 5：盲测包与工作流 C 交接

### 5.1 独立盲测包

Codex 只生成冻结盲测命令和只读输入契约，不自行读取受保护测试标签。盲测包必须：

- 验证所有冻结 hash，不匹配时拒绝运行。
- 只读取 test 输入并输出原始预测，不在运行时训练、调参或选阈值。
- 预测文件写入新版本目录，默认拒绝覆盖。
- 由未参与调参且持有测试标签的独立成员执行匹配和指标评估。
- 一次运行后保存命令、环境、stdout/stderr、原始预测、指标和时间戳。
- test 标签一旦解封并完成评估，该 test 永久标记为 `consumed`。盲测失败只能进入下一候选版本；v2 若需新的正式结论，必须使用新的封存 test。复用已消费 test 只能标为探索性复核。
- 不得删除失败结果、重划已消费 test 或用 test 继续调参。

如果当前 Codex 会话已经读取测试标签，该结果只能标为“非盲内部复核”，不得称为独立盲测。

### 5.2 向工作流 C 交接

交接只提供 candidate bundle 中标记为实时子集的算法候选和稳定接口，不直接改实时服务：

```text
输入特征/时序 schema
缺失与质量 mask 语义
因果窗口和状态重置要求
模型加载与版本校验
规则安全覆盖和 fallback
risk_score / confidence / risk_factors 语义
单样本和批量离线推理入口
目标硬件延迟预算
已知不支持场景
```

若需要修改 `AlgorithmEvent`、`runtime/` 或 `service/` 才能接入，先给出最小接口差异和回归风险并请求用户批准。不得在 B 中顺手实现 latest-frame、stream epoch、outbox、HTTP 回调或业务处置。独立盲测通过前，工作流 C 只能做显式 opt-in 的集成和性能评估，不得把候选晋升为实时默认主线。

退出条件：独立成员可用一条冻结命令执行盲测；工作流 C 可在不猜测模型语义的情况下评估实时集成。

## 阶段 6：报告、文档与发布候选

对可评估的证据族生成或更新：

```text
reports/fall_risk/fall_event_metrics.md
reports/fall_risk/near_fall_metrics.md
reports/fall_risk/gait_sit_stand_metrics.md
reports/fall_risk/baseline_longitudinal_metrics.md
reports/fall_risk/fusion_calibration.md
reports/fall_risk/ablation_report.md
reports/fall_risk/robustness_report.md
reports/fall_risk/failure_casebook.md
reports/fall_risk/model_card.md
reports/fall_risk/workflow_b_candidate_freeze.md
reports/reproducibility/model_and_config_hashes.md
```

每份报告必须遵循 `reports/README.md`，至少记录：git commit、`source_state_hash`、环境锁 hash、数据清单、split ID、raw/cache/task bundle ID、配置、指标定义、结果、95% CI、拒识、失败案例、原始预测路径/hash 和复现命令。

报告必须明确区分：

```text
已实现自动化
已在合成数据验证
已在 train 内层开发 / 已完成一次冻结 val 验证
独立盲测结果
探索性分析
未达目标的负结果
人工/数据/依赖阻塞
```

目标值和实测值必须分开。没有正式数据时，合成测试结果不得进入 `*_metrics.md` 的实测结论表。

只有代码行为或当前能力实际变化后，才更新当前状态文档：

```text
docs/modules/fall_risk/README.md
docs/architecture/算法工程骨架.md
docs/tasks/README.md
```

新增、移动或删除现行文档时，无论代码状态是否变化，都必须同步更新唯一总入口 `docs/README.md`。

当前实现状态只写入模块 README、工程架构和任务清单；不要把计划文档或本任务书改成完成证明。

## 人工与外部环节处理规则

Codex 无法代替以下工作：

```text
双人标签复核与冲突仲裁
真人知情同意和安全动作采集
subject_id 人工恢复
数据、代码和权重许可证确认
功能评估或纵向参考终点确认
主指标、label horizon 和临床/业务阈值预注册
固定比赛硬件确认
测试标签独立保管和正式盲测执行
模型进入实时主线的负责人签字
```

对这些环节必须：

1. 生成结构化模板、校验器、只读运行命令和明确操作清单。
2. 在 `workflow_b_blockers.md` 写明责任角色、所需输入、完成证据和解除后的准确命令。
3. 继续完成不依赖该人工输入的基础设施、单元测试和合成烟测。
4. 绝不能伪造标签、审批、盲测隔离、临床结论、许可证或硬件结果。

## 完成标准

完成判定分为四层，不得混写。

### 自动化基础设施完成

- [ ] editable 安装指向当前仓库，标准环境依赖可复现。
- [ ] 四类证据就绪矩阵和 blocker 可由命令重新生成。
- [ ] raw observation、标签无关 feature cache 和 task bundle 三层可独立版本化并按 hash 复现。
- [ ] 时序缓存按真实时间重采样，保留 mask、来源、版本和因果截止。
- [ ] task bundle 泄漏检查覆盖人员、家庭/源组、原事件、多机位、相邻窗口和内容 hash。
- [ ] 上游感知和平滑 benchmark 能在同一协议下做单变量对照。
- [ ] 分支实验运行器能输出配置化训练、预测、指标、校准、消融和复现 bundle。
- [ ] 特征选择、hard-negative、预处理、模型和异常检测器只使用 train 内层数据；校准只使用 train OOF/calibration；冻结 val 只运行一次短名单比较和阈值/guardrail。
- [ ] 相关窄范围测试、完整测试集、真实视频烟测和 `git diff --check` 通过。

### 单个证据族完成

- [ ] 使用独立真值、冻结 split、冻结协议和同一 task bundle。
- [ ] 规则 baseline 与候选均有原始预测、主指标、95% CI、拒识和失败案例。
- [ ] 候选搜索只使用 train 内层协议，冻结 val 只运行一次；负结果未删除。
- [ ] 报告结论不超出该证据族的真值能力。
- [ ] 无法评估时明确标为 `infrastructure_only` 或 `not_evaluable`，没有伪指标。

### 工作流 B 冻结候选完成

- [ ] 所有可评估证据族已完成，不能评估的证据族有真实 blocker 和责任人。
- [ ] 至少一个核心机制在重复验证中产生可复现增益，并通过预注册的 G4 判定规则。
- [ ] 只有真实配对数据存在时才完成融合；否则融合明确标为 `not_evaluable`。
- [ ] 唯一版本化 candidate bundle、逐证据族 artifact、规则 fallback、模型卡、配置/hash 和盲测包齐全。
- [ ] 工作流 C 交接契约完整，但未越界修改实时服务。
- [ ] 文档没有把计划目标、合成测试或验证集结果写成独立盲测/临床有效性事实。

### 独立盲测验证完成

- [ ] test 在候选冻结前保持物理隔离，调参人员和当前 Codex 未读取逐样本标签。
- [ ] 独立保管人校验 candidate bundle、源码、环境、数据、task bundle 和模型全部 hash 后，只运行冻结预测/评估命令一次。
- [ ] 原始预测、指标、日志、时间戳、失败案例和 test `consumed` 状态完整保存。
- [ ] 结果通过预注册的 G3/G4 判定；未通过时不在同一 test 上继续调参。
- [ ] 只有本层完成后，candidate bundle 中的实时子集才具备提交工作流 C 审批为默认主线的资格。

若数据、真值、预注册或盲测仍未完成，最终状态必须准确写成例如：

```text
工作流 B 自动化基础设施和冻结候选已完成；事件证据族已完成一次冻结 val 验证；
纵向与配对融合仍受以下数据/人工门槛阻塞：...
当前冻结候选尚未完成独立盲测，不得称为实时主模型或比赛最终指标。
```

不得为了满足任务措辞而宣称整体完成。

如果没有任何核心机制产生稳定增益，仍必须冻结规则主线 candidate bundle 并交付完整负结果；此时只能声明自动化和相应证据族实验已完成，工作流 B 冻结候选层仍未通过 G4。

## 验证命令

根据实际实现补齐 CLI 参数，并把最终可直接运行的准确命令写回相关 README。所有 Python 命令必须使用项目 conda 环境。至少执行：

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms

conda run -n eldercare-ai python -m pytest tests/test_fall_risk_temporal_samples.py -q
conda run -n eldercare-ai python -m pytest tests/test_fall_risk_perception_benchmark.py -q
conda run -n eldercare-ai python -m pytest tests/test_fall_risk_experiments.py -q
conda run -n eldercare-ai python -m pytest tests/test_fall_risk_calibration.py -q
conda run -n eldercare-ai python -m pytest tests/test_fall_risk_longitudinal.py -q
conda run -n eldercare-ai python -m pytest tests/test_fall_risk_model_inference.py -q

conda run -n eldercare-ai python -m pytest \
  tests/test_fall_risk_pose.py \
  tests/test_fall_risk_pose_quality.py \
  tests/test_fall_risk_gait.py \
  tests/test_fall_risk_sit_stand.py \
  tests/test_fall_risk_near_fall.py \
  tests/test_fall_risk_baseline.py \
  tests/test_fall_risk_pipeline.py \
  tests/test_fall_risk_data_manifest.py \
  tests/test_fall_risk_splits.py \
  tests/test_fall_risk_event_evaluation.py \
  -q

conda run -n eldercare-ai python -m pytest -q

conda run -n eldercare-ai python scripts/collect/run_fall_tracking.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --output /tmp/workflow_b_fall_tracks_smoke.jsonl \
  --model yolov8n.pt \
  --scene-region home \
  --max-frames 5

conda run -n eldercare-ai python scripts/collect/run_fall_pose.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --output /tmp/workflow_b_fall_poses_smoke.jsonl \
  --model yolov8n-pose.pt \
  --scene-region home \
  --max-frames 5

git diff --check
```

对于尚未创建的测试文件，使用与实际实现一致的名称，但测试覆盖不得减少。正式冻结测试集评估命令不得由参与调参的 Codex 会话自动执行；只生成给独立保管人的冻结命令。

## 停止并询问用户的条件

执行到以下情况时停止相关动作并请求批准；不受影响的其他阶段可以继续：

- 需要删除、移动、重命名或整文件替换现有文件，或者会丢失既有内容、覆盖标签/缓存/模型/冻结报告。允许范围内、保留既有用户修改的增量 `apply_patch` 不触发本条。
- 需要修改本任务允许范围之外的代码，尤其是心理健康、`runtime/`、`service/` 或业务接口。
- 需要新增、升级或移除依赖，修改环境定义，或下载模型权重/大型数据。
- 需要改变 `AlgorithmEvent`、上游 JSONL、时序样本或冻结 split 的公共 schema。
- 需要人工确定主指标、label horizon、IoU/onset 容忍、阈值、拒识工作点、最小样本量或实时硬件预算。
- 数据许可、人员身份、配对关系、功能/纵向参考终点或标签复核状态无法从仓库证据确认。
- 发现 train/val/test 泄漏、测试标签已被调参人员或当前 Codex 读取，或盲测治理已经失效。
- 需要把不同数据源拼接为同人同时段样本，或需要用算法输出代替人工真值。
- 需要把低质量样本默认填成正常值，或改变强事件安全覆盖才能让候选指标变好。
- 新模型没有稳定验证增益但任务似乎要求继续扩大搜索；此时保留规则主线并请求是否开展新的探索版本。
- 发现凭据、个人身份信息、未经授权的真人数据或可识别人脸可能进入版本控制、日志、模型或报告。
- 两条实现路径会改变数据契约、实验语义、因果性或长期维护成本，且仓库现有模式不能给出明确选择。
- 同一阻塞连续出现且没有安全替代路径。

## 最终交付格式

完成后用简洁中文报告：

1. 自动化基础设施、各证据族和工作流整体的独立状态。
2. 已实现的代码、CLI、配置、缓存 schema 和模型推理入口。
3. 实际生成的 source/environment hash、raw/cache/task bundle ID、split ID、candidate bundle ID、模型/配置 hash、原始预测和报告。
4. 实际运行的验证命令和结果；独立盲测是否由合格保管人执行必须单列。
5. 候选相对规则 baseline 的验证集结果、校准、拒识、延迟、失败案例和负结果。
6. 尚未完成的人工、数据、依赖或硬件门槛，不得混入“已完成”。
7. 是否具备交给工作流 C 的冻结候选；若不具备，写明回退版本和解除阻塞的准确命令。

不要只给文件清单；必须逐项说明完成条件是否通过。若未通过，写明证据、影响和下一责任人，不得用“基本完成”“效果较好”等模糊措辞掩盖。

> 本任务书面向具有真实文件系统和终端权限的 Codex。执行前必须确认仓库路径、允许修改范围、禁止事项和停止条件与当前环境一致。
