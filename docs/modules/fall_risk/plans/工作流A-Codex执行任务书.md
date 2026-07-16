# 工作流 A：数据、标注与评估底座 Codex 执行任务书

你是一名负责机器学习数据治理与评估基础设施的高级 Python 工程师。请在以下仓库中直接实施工作流 A，而不是只输出建议或重新写一份计划：

```text
/Users/guai/Documents/project/挑战杯/algorithm
```

## 目标

建立一个可追溯、可验证、可冻结、可复现的跌倒风险数据与评估底座，使后续规则、轻量模型和时序模型只能在相同数据版本、相同 split 和相同评价协议下比较。

最终目标版本名为 `fall-risk-data-v1`。只有自动化工具、测试、数据审计和可复现入口全部通过，且人工依赖被明确完成或明确列为阻塞时，才能声明工作流 A 完成。

## Context (carry forward)

- 项目使用 Python 3.11、`src/` 布局和 conda 环境 `eldercare-ai`。
- 所有 Python 脚本和测试必须通过 `conda run -n eldercare-ai python ...` 执行，禁止使用裸 `python`、裸 `pytest` 或 base 环境。
- 跌倒风险和心理健康风险必须独立评分、独立验证、独立输出。本任务只处理 `fall_risk`，不得修改心理健康模块行为。
- 工作流 A 的原始要求位于 `docs/modules/fall_risk/plans/挑战杯揭榜挂帅冲奖增强计划.md` 第 5 节，并受其中 G0-G3 门槛约束。
- 当前能力和限制以模块 README、工程架构、任务清单和实际代码为准，计划文档不能作为“已经实现”的证据。
- 当前仓库可能有用户尚未提交的修改。必须先检查 `git status` 和相关 diff；不得回退、覆盖或格式化与本任务无关的用户修改。
- 2026-07-15 的已知观察值如下，但开始实施时必须重新核验，不能盲信：
  - `action_labels.jsonl` 和 `event_labels.jsonl` 各有 922 条记录，均为 `pending`。
  - 当前标签的 `subject_id` 均为 `unknown`。
  - 现有标签中 `C03/C05` 为 0，`C04` 仅 1 条，不能支撑正式近跌倒结论。
  - `fall_risk_video_manifest.jsonl`、`risk_labels.jsonl`、`subject_profiles.json`、`annotation_review_log.jsonl` 和正式 split 尚未建立。
  - 当前 CVAT 转换逻辑对一次导出使用单一 FPS；LE2I Home 约为 24 FPS，Coffee/Lecture/Office 为 25 FPS。
  - `scripts/evaluate/` 当前只有 README，没有可执行事件评估器。
  - LTMM 本地主要只有表格和索引，长期原始信号大部分不可用。
- 当前标签、公开数据和 CVAT 原始导出都可能包含错误路径、来源不明记录或身份元数据。不得直接把它们视为正式真值或提交材料。

## 开始前必须阅读

完整阅读以下文件后再设计或编辑代码：

```text
AGENTS.md
README.md
environment.yml
environment-reference.txt
docs/README.md
docs/architecture/算法工程骨架.md
docs/interfaces/算法事件输出接口.md
docs/tasks/README.md
docs/modules/fall_risk/README.md
docs/modules/fall_risk/plans/挑战杯揭榜挂帅冲奖增强计划.md
docs/modules/fall_risk/plans/跌倒风险算法研发计划.md
docs/modules/fall_risk/data/数据集标注规范.md
docs/modules/fall_risk/data/数据标注SOP.md
docs/modules/fall_risk/data/跌倒风险标签字典.md
data/README.md
data/annotations/fall_risk/README.md
scripts/annotation/README.md
scripts/evaluate/README.md
reports/README.md
```

还必须检查下列实现及其测试：

```text
src/elderly_monitoring/modules/fall_risk/annotations.py
src/elderly_monitoring/modules/fall_risk/ltmm.py
scripts/annotation/convert_cvat_fall_labels.py
scripts/annotation/build_ltmm_manifest.py
tests/test_fall_risk_annotation_conversion.py
tests/test_ltmm_manifest.py
```

## 执行方式

1. 先重新核验仓库、数据和环境现状，给出精简实施计划，再立即进入实现；不要停在计划阶段。
2. 采用测试驱动：每个行为变更先补失败测试，再实现最小正确改动。
3. 每完成一个阶段，向用户更新：`完成内容 / 修改文件 / 验证结果 / 剩余风险`。
4. 先运行最相关的窄范围测试；阶段完成后运行完整测试集。
5. 遇到失败时先定位根因并保留证据，不得通过放宽校验、临时 `PYTHONPATH`、删除样本或改测试来掩盖问题。
6. 自动化阶段之间应持续执行。只有命中本文的停止条件时才暂停并请求用户决定。

## 允许修改的范围

优先复用现有模块和目录。可以在确有必要时修改或新增：

```text
src/elderly_monitoring/modules/fall_risk/annotations.py
src/elderly_monitoring/modules/fall_risk/              # 仅数据治理、split、评估相关模块
scripts/annotation/
scripts/evaluate/
scripts/split/                                          # 仅在现有目录无法合理承载时新增
tests/
configs/data/
configs/evaluation/
data/manifests/
data/splits/
data/annotations/fall_risk/quarantine/
data/annotations/fall_risk/generated/                  # 候选重导结果，禁止静默覆盖正式标签
reports/fall_risk/
reports/reproducibility/
docs/modules/fall_risk/data/
docs/modules/fall_risk/README.md
docs/architecture/算法工程骨架.md
docs/tasks/README.md
docs/README.md
```

只有确实减少复杂度或匹配现有结构时才新增模块。不要为了“架构完整”创建无行为的抽象层、注册表或占位类。

## 禁止事项

- 不得修改 `src/elderly_monitoring/modules/mental_health/`、心理健康配置或其接口行为。
- 不得修改实时服务、回调、风险融合或模型行为，除非它们阻断预测 JSONL 的读取；遇到这种情况先说明并请求批准。
- 不得删除、移动、重命名或覆盖原始视频、第三方数据、CVAT ZIP/XML、现有标签或用户未提交修改。
- 不得根据画面、文件顺序或猜测伪造 `subject_id`、人员关系、事件边界、风险标签、许可状态或知情同意状态。
- 不得把 `pending`、`uncertain`、源文件缺失或许可不明记录自动改成 `reviewed/final`。
- 不得把目录级 fall/ADL 标签伪装成精确事件起止时间。
- 不得把算法预测当成人工真值或用规则输出验证规则本身。
- 不得把公开跌倒视频用于证明长期个人风险预测或临床有效性。
- 不得把人工序数真值写成虚构的连续 `risk_score`。连续风险分属于模型预测，不属于人工标签。
- 不得上传任何数据、日志或标注到外部服务。
- 未经用户批准，不得下载大型数据、安装依赖、改 `environment.yml`、改 `pyproject.toml`、创建提交或推送远端。
- 不得在日志、报告、测试夹具或最终回答中暴露邮箱、令牌、直播地址、真实姓名或其他身份信息。

## 阶段 0：基线核验与保护

完成以下检查并记录事实：

1. 运行 `git status --short`，识别相关文件中的既有修改并保留它们。
2. 运行：

   ```bash
   conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
   ```

   editable 安装必须指向当前仓库；若不是，按 `AGENTS.md` 使用当前项目重新安装后再验证。
3. 统计数据集文件数量、视频数量、标签数量、复核状态、subject 覆盖、标签类别、质量类别和 quarantine 数量。
4. 检查 CVAT 导出是否包含身份元数据。不得修改原件；只在报告中以脱敏方式记录风险。
5. 对现有标签、manifest 和 split 计算初始 checksum，写入本地审计结果。
6. 建立 `reports/fall_risk/workflow_a_blockers.md`。只记录经过核验的人工依赖和外部阻塞，不写推测。

退出条件：环境路径正确，当前数据状态有可复现统计，原始资产未被修改，人工依赖有明确清单。

## 阶段 1：统一 manifest

实现确定性的全量 manifest 构建器，至少支持当前本地存在的：

```text
LE2I/IMViA
Fall Detection 2017
UR Fall
TOAGA
GSTRIDE
LTMM
Pre_VFallp
```

manifest 每条资产至少包含：

```text
asset_id
dataset
subset
path
sha256
media_type 或 modality
fps_num / fps_den / fps                  # 非视频允许 null
frame_count / duration_sec               # 非视频允许 null
width / height                           # 非视频允许 null
subject_id
source_group_id
original_event_id
scene_region
view
label_source
annotation_path
license_id
consent_id
review_status
eligibility
exclusion_reasons
```

要求：

- 路径必须相对仓库，禁止写机器相关绝对路径。
- `asset_id` 全局唯一且稳定；相同输入重复运行必须字节级一致。
- 使用结构化元数据解析和 `ffprobe`/OpenCV 等可靠接口，不得从随意字符串切片猜测关键字段。
- 保存真实 FPS 有理数和浮点表示；不得用全局默认 FPS 覆盖视频元数据。
- 同一原事件的多机位必须共享 `original_event_id/source_group_id`。
- `subject_id` 无法从官方元数据恢复时保持 `unknown`，并设置保守的 `source_group_id`。
- `license_id` 或来源无法确认的资产必须 `eligibility=false`，进入审计，不进入正式 split。
- Pre_VFallp 在来源、许可和标签语义未核验前必须隔离。
- manifest 按稳定键排序，并生成 manifest SHA-256 和版本摘要。

目标输出：

```text
data/manifests/fall_risk_video_manifest.jsonl
reports/fall_risk/data_audit.md
reports/reproducibility/dataset_and_split_versions.md
```

测试至少覆盖：稳定 ID、路径存在、hash、逐视频 FPS、重复文件、非视频资产、缺失许可、同事件多机位分组和重复运行确定性。

## 阶段 2：标注导入、转换与验证

### 2.1 修复 CVAT 转换

- 让转换器从 manifest 按 `video_id` 获取逐视频 FPS、路径、帧数和时长。
- 可以保留显式 `--fps` 作为向后兼容的开发覆盖项，但正式批处理必须使用 manifest；发生冲突时应失败或清晰告警，不能静默继续。
- 校验 `start_frame/end_frame` 在视频范围内，时间由真实 FPS 或可用 PTS 换算。
- 校验 `start_time/end_time` 与帧号一致，容差必须配置化并有测试。
- 保留人工动作标签与映射事件的来源链，不能覆盖官方事件来源。
- 写文件必须采用临时文件加原子替换；默认拒绝覆盖已有正式标签。
- 候选重导结果先写入 `data/annotations/fall_risk/generated/v1/`，通过验证且获得人工确认前不得替换根目录正式标签。

### 2.2 导入 LE2I 官方标注

实现独立的 LE2I TXT 导入器：

- 只把官方 TXT 支持的跌倒窗口写为 `event_type=fall`。
- `label_source=le2i_txt`，不得写成 `manual_reviewed`。
- 官方边界和人工边界同时保留，通过稳定来源 ID 关联，不互相覆盖。
- `Lecture room` 和 `Office` 没有官方 TXT，不得进入官方有监督事件指标。
- Home_02 必须保留原始 `video (31)` 至 `video (60)` 编号，不得重编为 1-30。

### 2.3 标注校验器

实现严格校验 CLI，至少检查：

- JSONL 可解析、字段完整、枚举合法、ID 唯一或具有合法复合键。
- `start_frame <= end_frame`、`start_time <= end_time`，且均未超视频边界。
- `U01/uncertain` 有原因说明。
- 标签源、review 状态、文件存在性和 manifest 关联一致。
- `pending/uncertain/missing/license_unknown` 不具备正式评估资格。
- 动作映射事件、官方事件和人工事件能区分来源。
- 高风险动作、跌倒和冲突记录是否具备人工复核证据；缺失时报告 blocker，禁止自动补全。
- 输出类别分布、数据集分布、场景分布、人员/组分布、质量分布、边界异常和仲裁比例。

必须为 `risk_labels.jsonl`、`subject_profiles.json` 和 `annotation_review_log.jsonl` 定义并验证 schema，但没有人工结果时只生成空模板或示例 schema，禁止生成假记录。

## 阶段 3：版本化 split

实现按任务独立的 split builder：

```text
fall_event_v1
near_fall_event_v1
functional_proxy_v1
longitudinal_baseline_v1
```

共同规则：

- 只接收 `eligibility=true` 且标签为 `reviewed/final` 的正式样本。
- `subject_id` 已知时按人分组；未知时按 `source_group_id` 保守分组。
- 同一原事件、同一人的多机位、相邻窗口和派生副本不得跨集合。
- Fall Detection 2017 按 `SBJ_*`；UR Fall 按事件编号且 cam0/cam1 同组；TOAGA 按 `OAWxx`；GSTRIDE 按 `Vxxx`；自采数据按脱敏 `subject_id`。
- LE2I 无可靠人员 ID 时不得支持人员泛化声明。优先作为事件开发集或外部事件测试，并在报告中披露限制。
- 不同任务不得因“样本多”而强行合并。没有功能或纵向参考终点时必须报告缺口。
- 比例、随机种子和分层字段写入 YAML；不得在代码中隐藏阈值。
- split 输出按稳定键排序，对规范化内容生成 `split_id` 和 SHA-256。
- 冻结后只创建新版本，禁止原地重划测试集。

实现泄漏检测，至少断言：

```text
subject_id 交集为空
source_group_id 交集为空
original_event_id 交集为空
内容 hash 交集为空
同一事件的多机位和相邻窗口没有跨集合
```

正式测试标签应与预测运行入口分离。测试集首次正式评估前，只允许做 schema 和基础设施烟测，不允许根据测试标签调整阈值。

## 阶段 4：事件评估器

先完整实现跌倒/近跌倒事件评估；功能 proxy 和纵向评估只有在真实标签存在时实现正式指标，否则只完成 schema、split 和阻塞报告。

### 4.1 预测 JSONL 契约

至少包含：

```text
video_id
task_type
event_type
prediction_id
score
start_time
end_time
onset_time
status
quality_state
model_version
config_hash
split_id
```

### 4.2 匹配协议

- 只有同一 `video_id`、同一 `task_type`、允许匹配的 `event_type` 才能成为候选。
- IoU 阈值、onset 容忍范围、搜索窗口、事件合并和复位规则全部从冻结 YAML 读取。
- 一个预测最多匹配一个真值，一个真值最多匹配一个预测。
- 匹配必须确定性；重复预测除首次合法匹配外计 FP。
- PR 曲线按 score 阈值扫描，并在每个阈值重新执行同一匹配规则。
- 未注册的默认值只能标记为 `development/provisional`，不得作为正式协议静默固化。

### 4.3 指标

至少输出：

```text
事件级 Precision / Recall / F1
事件级 PR-AUC
TP / FP / FN
漏报率
FDR = FP / (TP + FP)
传统 FPR（只有合法负样本分母时）
每摄像机小时 FP
每家庭日 FP（只有合法连续监控数据时）
事件边界 IoU
onset_detection_latency
跌倒 detection_latency
恢复召回率和错误恢复率（存在恢复真值时）
人工复核工作量
```

提前量统一定义为：

```text
lead_time = reference_event_start - first_level_3_or_higher_alert
```

正值表示提前，负值表示事件发生后才报警。没有后续参考跌倒事件时，近跌倒只报告相对自身 onset 的检测延迟，不得称为 pre-fall lead time。

### 4.4 统计与证据输出

- 按 `subject_id` 聚类 bootstrap；未知时按 `source_group_id`。
- bootstrap 次数和随机种子配置化；开发默认可较小，正式报告使用 10,000 次。
- 输出 95% CI，并在样本不足时明确标为探索性。
- 每次评估输出：

  ```text
  metrics.json
  matches.jsonl
  false_positives.jsonl
  false_negatives.jsonl
  excluded_samples.jsonl
  threshold_curve.csv
  report.md
  ```

- 报告必须记录代码版本、环境、manifest hash、split ID、标签版本、配置 hash、预测 hash、指标定义和复现命令。
- 短事件剪辑不得伪装成连续监控小时。FP/摄像机小时的分母只能来自 manifest 中明确标记的可评估连续时长。

测试至少覆盖：完美匹配、无预测、无真值、重复告警、一对多、多对一、IoU 边界、onset 容忍边界、跨类型不匹配、`uncertain` 排除、提前量正负号、连续时长分母、bootstrap 确定性和 split ID 不一致。

## 阶段 5：审计、文档与发布候选

1. 使用真实本地数据运行 manifest builder 和严格校验器，但不得自动提升人工复核状态。
2. 用合成预测和开发集跑通评估器全链路；在正式测试集冻结前，不得把结果写成比赛正式指标。
3. 生成或更新：

   ```text
   reports/fall_risk/data_audit.md
   reports/fall_risk/workflow_a_blockers.md
   reports/reproducibility/dataset_and_split_versions.md
   docs/modules/fall_risk/README.md
   docs/architecture/算法工程骨架.md
   docs/tasks/README.md
   docs/README.md
   ```

4. 当前实现状态只写入模块 README、工程架构和任务清单；不要把计划文档改成完成证明。
5. 报告中必须明确区分：已实现自动化、已验证真实数据、待人工复核、待采集、许可阻塞和探索性结果。
6. 生成 `fall-risk-data-v1` 发布候选摘要，但只有全部完成条件满足时才标记为正式冻结版本。

## 人工环节处理规则

Codex 无法代替以下工作：

```text
双人独立标注
冲突仲裁
真人知情同意
健康成年人安全动作采集
subject_id 人工恢复
许可证法律确认
测试集保管人与调参人员职责隔离
临床或功能参考终点确认
```

对这些环节必须：

1. 生成结构化模板、校验器、统计脚本和明确操作清单。
2. 在 `workflow_a_blockers.md` 写明责任角色、所需输入、完成证据和解除阻塞后的命令。
3. 继续完成不依赖该人工输入的自动化工作。
4. 绝不能伪造记录、自动签字、自动提升状态或用算法结果代替人工结论。

## 完成标准

只有以下条件全部满足，才能声明自动化部分完成：

- [ ] editable 安装指向当前仓库。
- [ ] 全量 manifest 可重复生成，路径、hash、媒体元数据和许可状态校验通过。
- [ ] Coffee/Lecture/Office 不再按 24 FPS 错误换算；逐视频时间轴测试通过。
- [ ] LE2I 官方事件和人工事件来源独立、可追溯、不会互相覆盖。
- [ ] 严格校验器能阻止 pending、uncertain、源文件缺失和许可不明数据进入正式 split。
- [ ] 四类任务拥有独立 split schema；有数据的任务生成稳定 split ID，无数据的任务输出明确 blocker。
- [ ] 泄漏检查覆盖人员、源组、原事件、多机位、相邻窗口和内容 hash。
- [ ] 事件评估器按照冻结配置输出匹配明细、指标、95% CI 和失败案例。
- [ ] 提前量符号、FP/摄像机小时分母和事件一对一匹配都有回归测试。
- [ ] 生成结果可以通过文档中的一组 conda 命令从头复现。
- [ ] 相关窄范围测试和完整测试集通过。
- [ ] `git diff --check` 通过，没有无关改动、秘密信息或原始数据修改。
- [ ] 文档没有把待人工完成或探索性结果写成已实现事实。

若人工标注、采集、许可或纵向真值尚未完成，最终状态必须写成：

```text
自动化底座已完成；工作流 A 整体仍受以下人工/数据门槛阻塞：...
```

不得为了满足任务措辞而宣称整体完成。

## 验证命令

根据最终 CLI 名称调整参数，并把可直接运行的准确命令写回相关 README。至少执行：

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
conda run -n eldercare-ai python -m pytest tests/test_fall_risk_annotation_conversion.py -q
conda run -n eldercare-ai python -m pytest tests/test_fall_risk_data_manifest.py -q
conda run -n eldercare-ai python -m pytest tests/test_fall_risk_data_validation.py -q
conda run -n eldercare-ai python -m pytest tests/test_fall_risk_splits.py -q
conda run -n eldercare-ai python -m pytest tests/test_fall_risk_event_evaluation.py -q
conda run -n eldercare-ai python -m pytest -q
git diff --check
```

对于尚未创建的测试文件，使用与实际实现一致的名称，但测试覆盖不得减少。

## 停止并询问用户的条件

执行到以下情况时停止相关动作并请求批准；不受影响的其他阶段可以继续：

- 需要删除、移动、重命名或覆盖现有文件。
- 需要修改本任务允许范围之外的代码。
- 需要新增或升级依赖、修改环境定义。
- 需要下载大型数据、上传数据或调用外部标注服务。
- 需要把候选标签提升为 `reviewed/final` 或替换正式标签。
- 需要人工决定正式 IoU、onset 容忍、搜索窗口、样本量或统计主指标。
- 数据许可、人员身份或参考终点无法从仓库证据确认。
- 测试集已经可能被用于调参，需要重新定义盲测治理。
- 发现凭据、个人身份信息或未经授权的真人数据可能进入版本控制或报告。
- 同一阻塞连续出现且没有安全替代路径。

## 最终交付格式

完成后用简洁中文报告：

1. 已实现的代码和 CLI。
2. 已生成的数据、split、配置和报告及其版本/hash。
3. 实际运行的验证命令和结果。
4. 尚未完成的人工或外部依赖，不得混入“已完成”。
5. 关键风险、数据边界和下一位负责人应执行的准确命令。

不要只给文件清单；必须说明每个完成条件是否通过。若未通过，写明证据和下一步，不得用模糊措辞掩盖。

> 本任务书面向具有真实文件系统和终端权限的 Codex。执行前必须确认仓库路径、允许修改范围和停止条件与当前环境一致。
