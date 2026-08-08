# AGENTS.md

## 项目环境

本项目的测试和脚本运行必须使用项目 conda 环境，不要使用默认 shell 里的 Python。

统一使用：

```bash
conda run -n eldercare-ai python -m pytest -q
```

运行项目脚本时也必须进入同一个环境：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_tracking.py --help
conda run -n eldercare-ai python -m elderly_monitoring.inference.run_features --help
```

除非是在专门排查环境问题，否则不要使用裸 `python`、裸 `pytest` 或 base conda 环境做验证。

标准环境定义文件：

```text
environment.yml
```

本机已验证环境说明：

```text
environment-reference.txt
```

项目使用 `src/` 布局。运行脚本前应确认 `elderly-monitoring-algorithms` 的 editable 安装指向当前仓库，而不是其他旧目录：

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
```

如果 editable 路径不是当前仓库，使用当前项目重新安装后再验证，不要长期用临时 `PYTHONPATH` 掩盖环境绑定问题：

```bash
conda run -n eldercare-ai python -m pip install -e ".[vision,service]"
```

徘徊步骤 10 有一项冻结的正式运行例外：红测、绿测和仓库 pytest 仍统一使用 `eldercare-ai`；production config 外部 SHA 形成并取得负责人授权后的六文件 R2M 也只允许在该标准环境调用 byte-only public materializer，不建模或训练。正式流程只建立两个由 `configs/runtime/wandering_step10_runtime_v1.yml` 创建的全新 Linux/x86_64 环境 A/B。每个环境启动一个训练 CLI fresh process，由唯一 builder 在同一进程内部先完成 production preflight、再执行五 seed 构建；随后用一个 fresh process 执行 full verifier，并用五个 fresh process 分别执行显式 seed safe loader，最后跨 A/B 比较。每个环境创建后，只允许从规范化的 active checkout 根执行一次 `python -m pip install --no-deps --no-build-isolation -e .` 接入源码；现场来源集合必须恰为锁定的 27 个 Conda 工件、30 个 pip wheel 加这一项 `elderly-monitoring-algorithms==0.2.0` local editable。标准 `eldercare-ai`、临时 `PYTHONPATH`、额外依赖或任何 runtime gate 跳过方式都不得产出正式步骤 10 bundle。完整契约见心理健康徘徊技术方案第 10.0、10.1 和 10.9 节。

## 开发流程

非平凡功能开发、缺陷修复、重构和发布准备需要遵循较完整的工程流程：

- 需求或目标行为不清楚时，先澄清再实现。
- 多步骤工作先写简短计划。
- 功能和缺陷修复优先采用测试驱动或先补回归测试。
- 遇到失败时做系统化排查，不凭猜测改代码。
- 宣称完成前，必须运行相关测试或验证命令。

措辞调整、文档补充、小配置改动可以保持轻量流程。

## 测试规则

修改 Python 代码后，先运行最相关的窄范围测试；条件允许时再运行完整测试集。

常用命令：

```bash
conda run -n eldercare-ai python -m pytest tests/test_fall_risk_tracking.py -q
conda run -n eldercare-ai python -m pytest -q
```

涉及视觉检测或跟踪模块时，需要额外跑一个真实视频烟测：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_tracking.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --output /tmp/fall_tracks_smoke.jsonl \
  --model yolov8n.pt \
  --scene-region home \
  --max-frames 5
```

涉及姿态关键点模块时，需要额外跑一个真实视频烟测：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_pose.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --output /tmp/fall_poses_smoke.jsonl \
  --model yolov8n-pose.pt \
  --scene-region home \
  --max-frames 5
```

如果测试无法运行，必须说明失败原因，并写出已经尝试过的具体命令。

## 协作风格

- 优先追求正确性、证据和有用的分歧，不为了表面一致而回避问题。
- 做技术评审时区分事实、推断和观点。
- 评审代码或方案时，先说风险、缺陷、缺失约束和更稳妥的替代方案。
- 没有检查相关证据前，不要宣称某个模块已经满足研发计划要求。

## 项目边界

本仓库只覆盖老年人跌倒风险和心理健康风险预警的算法原型研发。

本仓库不实现：

- 家属端 App、社区端后台或可视化看板。
- 账号、权限、设备管理。
- 消息推送、电话通知、工单流转和线下处置流程。

两个算法模块共享 `AlgorithmEvent` 字段契约，但必须分别评分、分别验证、分别输出：

- 跌倒风险只输出 `module=fall_risk` 事件。
- 心理健康风险只输出 `module=mental_health` 事件。
- 不增加综合模块、综合风险等级或跨模块动作协调逻辑。
- 业务后端按 `module` 分别持久化和处置算法事件。

项目工作首先需要对齐：

```text
docs/README.md
docs/architecture/算法工程骨架.md
docs/interfaces/算法事件输出接口.md
docs/tasks/README.md
```

跌倒风险相关工作还需要对齐：

```text
docs/modules/fall_risk/plans/跌倒风险算法研发计划.md
docs/modules/fall_risk/README.md
```

心理健康相关工作还需要对齐：

```text
docs/modules/mental_health/README.md
docs/modules/mental_health/plans/徘徊样行为识别技术方案.md
configs/modules/mental_health.yaml
```

## 当前研发阶段

当前项目处于跌倒风险算法的模型化增强阶段。步态、坐站、近跌倒和跌倒事件已经存在数据准备、训练、validation 复评或 shadow 推理实现，但全部仍是 provisional 候选；规则 baseline 继续作为运行主路径、安全覆盖和低质量输入 fallback。不得再把这些任务写成“尚未开始构建训练链”，也不得把候选实验写成已替换主路径。

当前 v3 训练标签的 fall/near-fall 事件监督门禁通过，动作类型门禁仍未通过，且所有 split 尚未冻结。模型候选只有在来源完整的标注、无泄漏且冻结的 train/validation/test split、连续背景与老人域验证、冻结评估协议和延迟/稳定性证据具备后，才能替换主路径。个体行为基线和最终风险融合仍需对应的连续个人数据或 `risk_labels` 真值；在这些条件满足前，保留统计/规则 fallback。

心理健康日级 baseline 已实现；徘徊专项步骤 2–8 已关闭。步骤 9/9a 已形成独立 Git 检查点：三条路径级 `eol=lf` 已生效，stage/新 checkout SHA 复核通过，最终 `model.py` SHA-256 为 `11fd7732f49d7392f0dab8edaa3023ebb1ba32355b24fc2157349b7fff80b331`，状态为 `step9_forward_contract_verified` 与 `step9a_exact_config_fail_closed_verified`；检查点 commit=`0a7db4e`。步骤 10 的 config/artifact 字段级 exact schema、wrapper 调用图和 canonical bytes 已冻结；Linux/x86_64 `2.13.0+cu130` runtime fact source=`configs/runtime/wandering_step10_runtime_v1.yml / 11426 bytes / 034d603a…e2f1cda` 已在候选 commit `03079a5` 的全新 checkout/环境中通过 27 个 Conda 工件与 30 个 pip wheel 的精确来源集合、`pip check` 及 runtime/thread/deterministic 探针，状态为 `wandering_step10_runtime_fact_source_verified`。但 22-source closure 中 12 个 auxiliary source 尚无路径级 LF/新检查点，当前为 `i/lf,w/crlf,attr/`；下一项只能先按第 10.0/10.9 节关闭 G0，fresh checkout SHA 通过后才建立失败测试和生产源码。production config 外部 SHA 形成前不得读取正式训练数据、创建正式数据路径的 production optimizer 或生成 checkpoint，合成单测中的 AdamW 不受此禁令。目标模型训练效果、片段状态机、日级接入和授权摄像头验证完成前，不把这些工程产物写成正式徘徊识别能力。

## 文档维护规则

- `docs/README.md` 是文档唯一总入口；新增、移动或删除现行文档时必须同步更新。
- 根 `README.md` 是仓库入口，`AGENTS.md` 是协作约束入口；项目阶段、事实源、标准命令或边界变化时必须同步更新，不能只改 `docs/` 下的模块文档。
- 当前实现状态只写入工程架构、模块 README 和 `docs/tasks/README.md`，不要在多个计划或汇报中重复维护。
- `plans/` 描述目标路线和实验设计，不能把计划内容当作已经实现的证据。
- `docs/archive/` 只保存历史评审、阶段汇报、早期方案和已结束计划；归档内容不作为当前代码事实源。
- 新实验指标、复现记录和失败案例写入 `reports/`，会议汇报和开发日志不要放入该目录。
- 修改代码行为、字段、阈值、运行命令或已知限制时，同步更新对应模块 README、接口文档和任务状态。
- 文档移动后必须检查 Markdown 链接和代码路径引用，不能留下指向不存在文件的入口。

## 跌倒标注事实源

- v2 是根标签与发布候选契约：以 `configs/data/fall_risk_label_validation_v2.yaml`、当前标签字典、`data/annotations/fall_risk/action_labels.jsonl`、`event_labels.jsonl` 和现行发布/校验代码为准。v2 formal 未通过时，不得称为 frozen 数据发布。
- v3 是独立的模型训练契约：以 `configs/data/fall_risk_action_label_schema_v3.json`、`configs/data/fall_risk_event_label_schema_v3.json`、`configs/data/fall_risk_training_decision_20260804.json`、两份 `*_labels_v3.jsonl`、`data/splits/fall_risk/training_labels_v3/` 和现行迁移/split/校验代码为准。v3 不覆盖 v2，也不能反向充当根标签发布事实源。
- 当前计数、hash、split 和 `training_ready` 状态以 `reports/fall_risk/training-labels-v3-validation.json`、`reports/reproducibility/dataset_and_split_versions.md` 及机器产物为准，不在协作规则中复制长期维护。
- 不得从 Git 历史或 diff、`docs/archive/`、原始 CVAT 工具项目名或旧生成目录推断当前标注 schema；这些内容只用于追溯，不是现状依据。
- 修改任一层 schema 时，必须同步检查来源候选、根标签、受审决策、发布报告、split 和所有哈希引用。

## 跌倒风险固定算法路线

后续跌倒风险模块的实现、文档、测试和代码评审，都需要按以下主线对齐：

```text
萤石设备或开放平台视频流
  ↓
人体检测与跟踪
  ↓
人体姿态关键点提取
  ↓
关键点质量控制与时序平滑
  ↓
步态稳定性分析
  ↓
坐站转换能力分析
  ↓
近跌倒事件检测
  ↓
个体化行为基线建模
  ↓
轻量风险融合模型 + 规则校准
  ↓
跌倒风险等级 + 置信度 + 可解释风险因子 + 预警动作建议
```

实现时不要跳过中间层直接从视频给最终风险结论。若某一层暂时使用规则、轻量 baseline 或占位实现，必须在文档和结果说明中标明当前状态。

这里的“风险融合”仅指跌倒模块内部对步态、坐站、近跌倒、个人基线、场景和活动节律特征的组合，不改变两个算法模块独立输出的边界。
