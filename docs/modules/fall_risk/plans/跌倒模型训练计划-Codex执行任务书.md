# 跌倒模型训练计划：Codex 执行任务书

> 本文件是给 Codex 等具有仓库和终端访问权限的 agent 使用的执行任务书。它描述目标路线和执行门禁，不是训练结果证明，也不是已冻结数据发布。

## 1. 任务目标

你是资深计算机视觉与机器学习工程师。请在仓库
`/Users/guai/Documents/project/挑战杯/algorithm` 中，基于真实代码、数据产物和文档，建立并验证跌倒事件模型的训练链路。

本任务的主线是 `fall_event` 连续事件模型；`near_fall_event` 必须保持独立训练、独立评估、独立输出。步态、坐站、个体基线和风险融合只能作为后续依赖或对照，不能合并为综合模型。最终事件只能使用 `module=fall_risk`。

完成标准不是得到一个高窗口分数，而是得到可追溯、无泄漏、可冻结、可复现、能在连续视频中评估并可安全 shadow 运行的候选模型。规则 `FallStateDetector` 始终是当前主路径、安全覆盖和低质量输入 fallback，候选不得自动替换它。

## 2. Context（必须继承）

- 项目使用 `src/` 布局；所有 Python、测试和脚本命令必须使用 `conda run -n eldercare-ai python ...`，禁止裸 `python`、裸 `pytest` 或临时 `PYTHONPATH`。
- v2 是根标签/发布候选契约；v3 是独立训练契约，二者不能互相替代。
- 当前 v3 事件标签校验为 `valid=true`，`training_ready.fall_event=true`、`training_ready.near_fall_event=true`；动作类型门禁仍未通过。
- 当前统一 v3 split 为 provisional，不能称为 frozen。启动时必须重新读取当前 split、manifest、标签和哈希；不得盲信旧报告。
- 当前跌倒 TCN 是 4 秒预裁剪 candidate-clip 动作存在性 proxy，不是连续事件定位器；三 seed validation 高分不能解释为连续召回率、误报率、临床有效性或老人域效果。
- 当前评估器和协议为 `development_provisional`，真实连续背景分母、老人域验证和一次性 test 发布流程尚未完成。
- `risk_labels.jsonl` 和 subject profile 为空时，不训练最终个体风险融合；保留统计/规则 fallback。
- 不实现家属端、社区后台、账号权限、消息推送、工单或跨模块协调。

## 3. 开始前必须阅读

完整阅读并引用证据路径：

```text
AGENTS.md
docs/README.md
docs/architecture/算法工程骨架.md
docs/interfaces/算法事件输出接口.md
docs/tasks/README.md
docs/modules/fall_risk/README.md
docs/modules/fall_risk/plans/跌倒风险算法研发计划.md
docs/modules/fall_risk/plans/跌倒事件模型训练方案.md
configs/data/fall_risk_label_validation_v2.yaml
configs/data/fall_risk_action_label_schema_v3.json
configs/data/fall_risk_event_label_schema_v3.json
configs/data/fall_risk_training_decision_20260804.json
reports/fall_risk/training-labels-v3-validation.json
reports/reproducibility/dataset_and_split_versions.md
configs/training/fall_event_v1.yaml
configs/training/near_fall_event_v1.yaml
configs/evaluation/fall_event_v1.provisional.yaml
configs/evaluation/near_fall_event_v1.provisional.yaml
```

使用 `rg --files` 和 `rg` 检查相关的 manifest、迁移、split、数据准备、训练、评估、shadow 推理、运行时 fallback 和测试入口。若文档引用的文件不存在，记录为缺口；不得创建空文件伪装前置工作完成。

## 4. 代理边界

### 允许

- 读取仓库文件、数据元数据、报告和代码；检查 `git status`、diff、文件哈希和环境绑定。
- 在既有目录中新增或修改跌倒数据、split、训练、评估、校准、shadow 和测试代码。
- 使用当前项目 conda 环境运行窄范围测试、静态检查、CLI `--help` 和必要的短 smoke。
- 生成版本化报告、模型卡、失败案例和复现清单；新产物必须使用新目录或显式版本号。

### 禁止

- 不得回退、覆盖、删除或移动用户已有修改、原始视频、第三方数据、正式标签或冻结产物。
- 不得读取或使用 test 真值进行选模、调参、阈值、校准、困难样本挖掘或返工；未冻结前不得运行正式 test。
- 不得下载大型数据/权重、安装依赖、修改 `environment.yml`/`pyproject.toml`、修改 `AlgorithmEvent` schema、创建提交或推送远端，除非先获得用户批准。
- 不得把未标注背景自动变成 negative；不确定、严重遮挡、目标不明、ID 冲突或边界不可靠样本必须 `ignore` 或进入人工复核。
- 不得把规则输出当人工真值，不得将 candidate-clip 指标写成连续事件指标。
- 不得修改心理健康模块或新增综合风险模块。

## 5. 执行步骤

执行前先输出不超过 10 项的简短计划，然后立即处理所有不受阻塞的工作。功能和缺陷修复采用测试驱动：先补失败测试，再实现最小正确改动；先运行最相关的窄范围测试，阶段完成后再运行完整测试集。遇到失败先保留命令、输出和根因证据，不得通过改测试、删样本、放宽门禁或临时路径绕过。当前 P0 审计只覆盖 `fall_event` 分支；该分支 P0 未通过时，只能继续审计、工具、测试、合成数据和短 smoke，不得启动真实数据训练。`near_fall_event` 必须通过自己的标签、hash、实际窗口覆盖和 test 锁定审计后，才可运行明确标记为 `development_provisional` 的 train/validation 实验；这不解除 formal、frozen validation、shadow 或 test 门禁。某一分支命中人工阻塞时，只暂停该分支，其他独立基础设施工作继续。

### P0：数据、split 和协议门禁

1. 运行 `git status --short`，记录与本任务相关的既有修改；确认 editable 安装指向当前仓库。
2. 重新读取 v2/v3 标签、manifest、`training_labels_v3` split 和所有输入 SHA-256，输出“当前事实清单”。重点检查旧报告 `split_id` 是否仍匹配当前 split。
3. 恢复或重生成 `reports/fall_risk/label_validation_formal_v2.json`，逐项核验 v2 formal blockers；没有 `formal_ready=true` 前不得声称数据冻结。
4. 按 subject/source/event/content 的连通保护组重建或确认 split；禁止随机窗口切分、同一事件多视角跨分区和重叠窗口跨分区。
5. 补齐连续背景、exact onset/offset、七类 fall hard negative 和来源/质量复核；未具备来源完整证据的样本不得进入正式分母。
6. 预注册连续事件协议：IoU、onset 容差、阈值规则、合并/复位、bootstrap 分组、拒识计数和 test 保管人。当前 provisional YAML 只能作为开发默认值。

P0 产物：

```text
reports/fall_risk/fall_event_training_audit.md
reports/fall_risk/fall_event_blockers.md
reports/reproducibility/fall_event_training_manifest.json
data/splits/fall_risk/training_labels_v3/<new frozen version>/
configs/evaluation/fall_event_v2.frozen.yaml  # 仅门禁具备且负责人批准后生成
```

P0 建议数据门槛：validation/test 各至少 30 个独立 fall 正例；七类 hard negative 各至少 20 个独立保护组；至少 100 camera-hour 合格连续背景；老人域至少 20 名独立老人、30 camera-hour ADL。不能达到时必须标记 `infrastructure_only` 或 `development_provisional`，不得降低口径冒充正式证据。

### P1：baseline、连续候选和校准

1. 在同一连续 validation 上复现规则 `FallStateDetector`，固定 E0 结果和合法背景分母。
2. 按当前 split/hash 重建现有 candidate-clip TCN，作为 E1 数据/训练链路 sanity 和 warm-start；旧 `f898...` 报告不能直接充当当前版本结果。
3. 实现或复用连续事件 dataset builder 和因果预测入口。候选输入至少保留时间戳、17 点骨架、motion、bone、quality/mask、插值/跳变和目标连续性；模型只能使用预测截止时刻以前的帧。
4. 按实验矩阵训练三 seed 轻量因果 TCN；比较 class-balanced BCE/focal、来源均衡采样和必要的 onset 辅助头。subtype 头只有在动作类型门禁通过后才可启用。
5. 只使用 train/validation 拟合阈值和概率校准；输出事件级 Precision/Recall/F1/PR-AUC、IoU、onset 误差、P50/P95 告警延迟、FP/camera-hour、Brier/ECE、分域指标、失败案例和 P50/P95 推理耗时。

P1 通过建议：Recall 的 95% CI 下界≥0.90，F1 至少较 E0 高 0.05，FP≤0.1/camera-hour，P95 告警延迟≤1 秒，最差来源 Recall≥0.85，ECE≤0.05，三 seed F1 标准差≤0.02。任一关键来源/质量层退化时保留规则主路径。

### P2：盲测、shadow 和晋级申请

1. 冻结 checkpoint、阈值、校准器、代码/环境/数据 hash 和配置后，由独立保管人一次性运行 test；Codex 不得自行打开 test 标签。
2. 在目标硬件完成至少 72 小时 shadow；检查长视频误报、低质量拒识、NaN/Inf、崩溃、内存、队列积压、重复 episode 和规则 fallback。
3. 只有在事件效果、分域稳定性、延迟、资源、校准、复现和 fallback 全部合格后，才提交模型替换申请。申请只涉及 `fall_risk`，不修改跨模块边界。

P2 产物：一次性 test bundle、shadow 报告、模型卡、失败案例、运行资源报告和候选晋级/拒绝记录。未通过则保留 `provisional_shadow`，不得写入默认 checkpoint。

## 6. 最小实验矩阵

| 编号 | 实验 | 目的 |
|---|---|---|
| E0 | 规则状态机 | 主路径 baseline、安全覆盖 |
| E1 | 现有 candidate-clip TCN | 验证当前 split 重建和 warm-start，不作正式结论 |
| E2 | 因果 TCN：坐标 | 最小学习基线 |
| E3 | E2 + motion/bone | 验证快速下降和人体结构信息 |
| E4 | E3 + quality/mask/timing | 验证遮挡、丢帧和姿态噪声鲁棒性 |
| E5 | E4 + onset 辅助头 | 验证定位与低延迟净收益 |
| E6 | ST-GCN++ 同输入 | 骨架拓扑强对照 |
| OOD | leave-one-source-out、老人 ADL、低光/遮挡/辅具/床边压力集 | 验证跨来源和实际误报 |

所有实验必须使用同一标签版本、split、姿态版本和冻结协议；每轮只改变一个因素。`near_fall_event`、步态、坐站、个人基线和风险融合分别记录，不合并 Accuracy。

## 7. 检查点与输出格式

每完成一个阶段，输出：

```text
✅ 阶段与完成内容
修改文件及新增产物
执行的确切命令（Python 命令必须带 conda run -n eldercare-ai）
测试/校验结果
数据门禁状态
未解决风险和下一检查点
```

最终输出中文 Markdown，包含：范围判定、事实/推断/建议区分、修改文件、命令和结果、各阶段门禁、实验矩阵、失败原因、未完成项。不要输出“已完成”而没有对应报告、哈希或测试证据。

## 8. 强制停止条件

立即停止相关动作并请求用户决定：

- 需要删除、覆盖、移动用户文件或冻结产物。
- 需要下载数据/权重、安装依赖、修改环境或接口契约。
- 需要读取 test 真值、重划已冻结 split 或放宽标签门禁。
- 数据来源、人员授权、连续背景或老人域证据不足以支持下一阶段。
- 发现两个会改变架构或评估口径的有效方案。
- 同一错误连续尝试两次仍未解决。

无论如何，不得为了得到指标而绕过门禁。规则 baseline 必须一直可回退，模型候选必须显式标记 `provisional_shadow`。

> 这是一个具有真实文件系统和终端访问权限的 agentic 任务。粘贴前请确认仓库路径、权限、conda 环境和允许修改范围与实际项目一致。
