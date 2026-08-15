# M0-CAM-EP1B 执行任务书

日期：2026-08-15

状态：`ready_to_execute`

当前代码任务：`M0-CAM-EP1B`

推荐执行模型：`GPT-5.6 Codex / gpt-5.6-sol`

推荐智商（推理强度）：**最高（max）**

## 1. 本阶段只完成什么

把 EP1A 已经跑通的单视频 oracle-boundary 推理，整理成一条可批量复跑、可正确计算分母的摄像头四类小样本评估线：

```text
EP1A prediction bundle + 独立 truth view + 匿名分组信息
  → 一对一连接并检查边界身份
  → 全部 shape-eligible episode 为主分母
  → 四分类与二分类指标
  → QC coverage、时长分层、目的性困难负例诊断
  → 失败清单与下一轮补拍建议
```

完成标准是“工具、测试和真实标注小样报告可以复跑”，不是达到某个分数。当前样本量只支持 descriptive development pilot，不能宣称 camera 95%、自动边界、最终告警、真实老人或临床效果。

## 2. 已确认的起点

EP1A 已完成并可作为本阶段稳定底座：

- 已有 truth-free boundary importer、whole-clip/CVAT/JSONL 入口、episode 内 QC、80×14 预处理和同一冻结 TopoWander forward；
- D01 的 12 条 whole-clip 为 `12/12 ready/direct`；
- S00 的三个 XML episode 为 `3/3 ready/direct`；
- S00 组合后的 legacy 0–40 秒仍为 lapping-like，但只属于 long-context diagnostic；
- 推理不读取 `observable_pattern`、purpose 或 evaluation role；
- 当前模型时间通道关闭，episode 原始 duration 只是模型外元数据；
- EP1A 聚焦测试已独立复跑通过。

EP1A 可以维持 `completed`，但证据范围仅是 **oracle-boundary engineering scope**。当前 HEAD 尚未包含这些 untracked 新文件；这是版本交接风险，不是重新审计 EP1A 的理由。没有 commit/push 授权时保持工作树，不擅自提交。

截至 2026-08-15，本机现有原始剪辑库存为：D01 12 条、P01 4 条、L01 3 条、R01 3 条、H01–H05 共 9 条、N01 1 条、Q01–Q03 共 3 条，另有 S00 1 条及其 XML。当前只发现 S00 具备 XML；P/L/R/H/N/Q 还没有正式 tracking、truth 和 EP1A prediction bundle。下一步应先消费这些已有视频，不要在没有看过和标注它们之前盲目补拍。

目录名和拍摄脚本不是正式真值。尤其 D01 中若实际出现明显折返、回到起点或混合行为，必须按画面复核并重新定 episode；不能因为文件位于 `D01` 就批量写成 direct。

## 3. EP1B 的范围

### 3.1 必须做

1. 实现轻量的多 bundle oracle-boundary 评估入口；
2. 复用 EP1A 输出，不复制 tracker、Camera QC、预处理或冻结模型 forward；
3. prediction 与 truth 只在推理完成后连接；
4. 输出四分类/二分类 precision、recall、F1、support、confusion 和原始计数；
5. 主指标保留全部 shape-eligible truth，不通过删除 unavailable/error 美化成绩；
6. 另报 ready-only 条件指标、QC coverage、时长分层和失败案例；
7. purposeful hard negative 保留实际形态，在 shape 指标中仍按 pacing/lapping/random 计；
8. 用 D01/S00 做回归，并在真实 P/L/R 小样到位后形成首份四类 pilot 报告。

### 3.2 明确不做

- 不实现 automatic/semiautomatic boundary；
- 不调用旧 40 秒 window merge 作为 episode 真值；
- 不重训、不微调、不改模型权重；
- 不改 0.5 二分类阈值，不增加 OOD/uncertain 阈值；
- 不按预测修改 XML、truth、边界或标签；
- 不把 purposeful pacing/lapping/random 改成 direct；
- 不输出风险、告警或 `AlgorithmEvent`；
- 不扩展 receipt、PORTABLE、逐文件 hash、字节对齐、source anchor 或 exact-schema 攻击审计；
- 不因数据不足提前启动 EP2。

## 4. 最小代码线

建议新增以下最小文件，实际命名可在不改变职责的前提下微调：

```text
configs/modules/wandering_camera_episode_eval_v1.yaml
src/elderly_monitoring/modules/mental_health/wandering/camera_episode_evaluation.py
scripts/wandering/run_camera_episode_evaluation.py
tests/test_wandering_camera_episode_evaluation.py
reports/mental_health/wandering_camera_episode_eval_v1/
```

对现有 `camera_episode_import.py` 只做一个小扩展：公开 `validate_episode_truth()` / `load_episode_truth()`，复用现有 `wandering-camera-episode-truth-v1`，不要另造 truth v2。评估器不应各自复制一份标签校验逻辑。

只有当逐条运行 EP1A CLI 明显影响复跑时，才增加薄的 `run_camera_episode_batch.py`。它只能编排既有 importer/inference，不得复制 tracking 或模型逻辑。

评估配置保持轻量，只记录会改变指标口径的内容：四类顺序、二分类映射、shape eligibility、非 ready 处分和时长分层。不要嵌入文件 SHA，不要把配置 loader 写成逐字节 trust-root。

建议固定的描述性时长分层为：

```text
short:  0 <= duration < 15 s
medium: 15 <= duration < 40 s
long:   duration >= 40 s
```

这些只用于报告，不是模型窗口、标签规则或徘徊判定阈值。必须在查看 EP1B 四类预测成绩前固定；如果确有更合理分层，先在任务书和配置中说明理由，再运行正式 pilot。

## 5. 输入与连接方式

### 5.1 轻量批量索引

评估入口应能显式接收多个 EP1A prediction bundle，并为每个 bundle 指定独立 truth 文件和最少分组信息。推荐一行一个 bundle：

```json
{"bundle_id":"office-p01-s01-d01-01","prediction_bundle_dir":".../d01/01","truth_jsonl":".../d01_truth.jsonl","participant_id":"P01","session_id":"S01","camera_setup_id":"C6C_OFFICE_01"}
```

这是本机 development 索引，不是授权或发布 manifest。路径可以指向 Git 外数据；原视频不得复制进仓库。匿名 ID 必须稳定，但不需要层层审批或新哈希链。

### 5.2 一对一连接

oracle-boundary 模式不需要时间 IoU matching。使用以下身份做一对一检查：

```text
episode_id
source_video_id
target_track_id / prediction track_id
start_sec
end_sec_exclusive
```

要求：

- `episode_id` 在本批 truth 和 prediction 中分别唯一；
- prediction 和 truth 数量、身份、区间一致；
- 重复 ID、缺 truth、缺 prediction、错视频、错 track 或区间漂移必须明确失败，不能静默跳过；
- inference bundle 的 summary 必须声明 `truth_labels_consumed_by_inference=false`、`automatic_boundary_inference=false`；
- 同一评估批次的 candidate/model/class order 必须一致，只比较记录的 identity，不重新建设资产哈希闭包。

D01 只有在存在独立人工 truth 记录时才能进入指标；“模型预测 direct”不能反向生成 direct 真值。若当前只有负责人说明而没有可读 truth 文件，D01 继续作为工程回归，并在数据缺口中明确写 `truth_file_pending`。

## 6. shape 真值与 purpose 真值必须拆开

现有 CVAT 工作流会在 purpose/evidence 不清楚时把 `evaluation_role` 和 `annotation_status` 记为 uncertain。这个 uncertain 主要表示目的/告警用途不确定，不能自动抹掉已经清楚标出的轨迹形态。

EP1B evaluator 必须派生独立的 `shape_truth_status`，但不得改写 XML 或原 truth：

- `observable_pattern` 为 direct/pacing/lapping/random，且没有被明确 `excluded`：形态可评分；
- purpose 为 unknown、role/status 为 uncertain，但 shape 清楚：仍可进入 **shape-only** 指标；不得进入 purpose/alert 指标；
- `observable_pattern=unknown`：形态不可评分；
- 错人、边界不可判、整段不可用或明确 `excluded`：形态不可评分；
- purposeful pacing/lapping/random：在四分类仍按原类，在二分类 shape 任务仍属于 wandering-like；只在单独的 purposeful-hard-negative 诊断中说明它不能直接触发告警。

评估结果必须同时保留原始 `annotation_status/evaluation_role` 和派生 `shape_truth_status/reason`，不能偷偷改变人工标签。

## 7. 指标分母

### 7.1 主指标：all shape-eligible episodes

主表以全部 `shape_truth_status=eligible` 的 episode 为分母：

- `ready` 且有合法类别：进入四类和二类 confusion；
- `unavailable`、`boundary_uncertain`、`inference_error` 或明确 abstention：分别计数，并作为该真值类别未正确识别影响 recall/F1；
- confusion 可以增加 `pipeline_miss` 列，但它不是第五种行为标签；
- 不允许把非 ready episode 从主 support 中删除。

四类：`direct/pacing/lapping/random`。

二类映射：

```text
direct → direct_or_non_wandering
pacing/lapping/random → wandering_like
```

### 7.2 次指标：ready-only conditional metrics

可以另报只看 ready prediction 的 conditional confusion/F1，用来判断“QC 通过后分类头表现怎样”，但标题必须含 `ready_only` 或 `conditional`，不能作为 EP1B 主成绩。

### 7.3 类别缺失

若某一类 support 为 0：

- 仍输出各已有类 support 和原始计数；
- 四分类 macro-F1 标为 `not_computable`，不能用缺失类补 0 或忽略缺失类后冒充完整四类成绩；
- D01/S00 direct-only 回归只验证工具和 direct 行为，不形成 camera 四分类成绩。

### 7.4 必须一起报告

- 四类与二类 per-class precision/recall/F1/support；
- macro-F1、accuracy 和 confusion；
- truth 总数、shape eligible/excluded/unknown 数；
- prediction 的 ready/unavailable/boundary_uncertain/inference_error 数；
- overall QC coverage 与各类别 coverage；
- short/medium/long 分层的 support、coverage 和描述性指标；
- participant/session/setup 数量与每组支持数；
- purposeful hard negative 的形态预测分布，明确 `alert_metrics_available=false`；
- 概率仍为 `probability_calibrated=false`；
- misclassified、pipeline miss、身份/输入错误和不可评分真值的失败清单。

## 8. 建议输出

输出目录必须是新目录，最少包含：

```text
summary.json
metrics.json
confusion.json
episode_results.jsonl
failures.jsonl
README.md
```

`episode_results.jsonl` 用于误差分析，允许在推理完成后同时携带 truth 与 prediction。推理输入和 EP1A prediction bundle 仍必须保持 truth-free。

输出只需确定性排序、字段可读、拒绝覆盖和可重新计算。不要求每个文件新增 SHA、canonical byte audit 或隔离 checkout 复放。

## 9. EP1A 顺手补的两个 P1

这两项应在计算 EP1B 指标前完成，但不得借机重构整条链：

1. 为 `build_cvat_episode_import_bundle()` 和 `build_camera_episode_inference_bundle()` 补至少一个真正调用 builder 的合成端到端回归，并验证 fresh output/non-overwrite；现有 CLI `--help` 测试不能替代 builder 回归。
2. 增加“边缘只有低于 minimum confidence 的 observation”测试。`minimum_raw_detections` 和 boundary gap 不能被模型不会消费的低置信度点虚假满足；统一为 accepted observation/bucket 口径，并用 D01/S00 回归确认无退化。

## 10. 数据节奏

### Pilot 1：先跑整链

- 复用现有 D01/S00；
- pacing/lapping/random 各约 2 条自然完整 episode；
- 每条实际完成就结束，不强凑 40 秒；
- 先检查 importer、identity join、QC、指标和失败输出。

### Pilot 2：形成 EP1B 小样

- pacing/lapping/random 各逐步扩到约 10 条可评分完整 episode；
- direct 复用已有合格样本，必要时补少量不同起终点和速度；
- 同时积累打电话、找东西、清洁、锻炼、搬运等 purposeful hard negatives，但不要求它们阻塞第一版四类 shape 报告；
- 单人、单办公室、单机位仍只能称 descriptive development pilot。

不要过分规定人物怎样走。脚本只给大致形态和活动目的，最终标签按画面实际形成的轨迹，不按计划动作硬贴。

现有视频的用途必须分开：

- P01/L01/R01：优先形成人工 truth、共享 tracker 输出和 EP1A bundle，进入 core shape pilot；
- H01–H05：只把人工划出的清楚 P/L/R episode 纳入 shape 指标，并单列 purposeful-hard-negative subgroup；
- N01：坐、站、办公等非行走时间不能伪标 direct。它留给 EP2 automatic producer 后的 continuous-negative/person-hour 评价；
- Q01–Q03：用于 QC/质量失败诊断，不混入清晰样本主 F1；
- D01：逐条看实际画面后才决定 shape 与边界，不能由文件夹名或既有预测批量生成真值。

## 11. 最低测试

优先测试行为，不扩大审计面：

1. truth 不进入 EP1A inference；
2. 重复/缺失 episode ID、错视频、错 track、区间漂移拒绝；
3. 四个 runtime 状态分别计数，非 ready 不从主分母消失；
4. all-eligible 主指标与 ready-only 条件指标不同且计算正确；
5. purpose unknown + 清楚 shape 仍可用于 shape-only 指标；
6. purposeful hard negative 不被改成 direct；
7. duration band 边界 15 秒、40 秒行为正确；
8. 缺类别时 macro-F1 为 not computable；
9. builder 端到端和 fresh output/non-overwrite；
10. 低置信度边缘 observation 不掩盖 QC 缺口；
11. D01/S00 保持既有 ready/direct 概率回归；
12. CLI `--help`、Python compile、Markdown 链接和 `git diff --check`。

修改 Python 后先跑新 evaluator/EP1A 聚焦测试，再跑相邻 camera 测试。只有合并或交付前才跑全部 `test_wandering_camera_*.py`；普通局部失败不触发全仓审计。

## 12. 状态与停止条件

### 可以自主继续

- 阅读和修改徘徊范围内本地代码、配置、测试和文档；
- 新建 Git 外临时输出目录；
- 修复局部路径、CLI、测试、fixture 和指标实现；
- 运行 `eldercare-ai` 中的窄测与真实授权样片回归；
- 数据尚未齐全时完成 evaluator，并输出真实 data gap。

### 必须停下请负责人决定

- 改人工标签、episode 边界、split 或科学目标；
- 修改冻结模型、阈值、类别定义或训练路线；
- 从预测反向生成/改写 truth；
- 访问 sealed camera、SmartCare official/raw 或未授权数据；
- 删除/覆盖既有视频、标注、报告或证据；
- 修改跌倒或其他模块；
- commit、push、发布或对外传输；
- 数据缺失到无法形成真实四类报告时，先交付 `implementation_complete + data_pending`，不要伪造样本或自动启动 EP2。

## 13. 完成判定

可以标记 `m0cam_ep1b_status=completed`，必须同时满足：

- 批量 evaluator、CLI 和测试已落地；
- 两个 EP1A P1 回归已关闭；
- 至少一批真实、独立标注的 D/P/L/R episode 从 EP1A bundle 进入 evaluator；
- 当前任务表计划的 pacing/lapping/random 各约 10 条已形成可评分库存，或负责人明确接受更小的 descriptive pilot 作为本阶段终点；
- 主/条件指标、coverage、duration、group support、purposeful diagnostics 和 failures 均输出；
- 报告准确限制结论，不写 95%、FAR、自动边界或告警；
- D01/S00 相邻回归没有退化。

只有工具完成但真实 P/L/R 数据不足时，状态写：

```text
m0cam_ep1b_implementation_status=completed
m0cam_ep1b_data_status=pending
m0cam_ep1b_status=not_completed
```

## 14. 可直接交给 Codex 的目标提示词

```text
你是本项目 M0-CAM-EP1B 的主开发 AI。请在当前仓库持续完成“摄像头完整 episode 的四类小样本评估线”，以实现可用的徘徊模块为最高优先级。

AI 设置：
- 模型：优先 GPT-5.6 Codex / gpt-5.6-sol
- 智商（推理强度）：最高（max）
- 工作方式：质量优先、可长时持续执行；可并行做只读审查和测试分析，但主代理必须亲自核对关键代码与结果。

开始前必须完整阅读：
1. AGENTS.md
2. docs/tasks/README.md
3. docs/modules/mental_health/plans/徘徊识别技术文档2.md
4. docs/modules/mental_health/plans/M0-CAM-EP1B执行任务书.md
5. docs/modules/mental_health/README.md
6. reports/mental_health/wandering_camera_episode_v1/README.md
7. reports/mental_health/wandering_camera_episode_v1/VERIFICATION.md
8. EP1A 的 config/importer/inference/CLI/tests 实际源码

当前事实：M0-CAM-EP1A 已完成，且只证明 oracle-boundary engineering scope。D01 12 条 whole-clip 与 S00 三条 CVAT episode 均由正式入口得到 ready/direct；S00 0–40 秒 legacy 结果仍为 lapping-like，但不得用于覆盖三个 direct episode 真值。当前唯一代码任务是 M0-CAM-EP1B，不要重开 EP1A、PORTABLE、receipt、hash 或 exact-schema 审计。

你的目标：
1. 先独立核对当前工作树和 EP1A 输出，保护所有既有 dirty/untracked 改动，不 reset/checkout/clean/stash，不覆盖输出。
2. 以测试驱动实现轻量批量 oracle-boundary evaluator：读取多个 EP1A prediction bundle、独立 truth view 和匿名 participant/session/setup 索引；推理阶段永远看不到 truth。
3. prediction 与 truth 按 episode/video/track/start/end 一对一连接；重复、缺失或漂移明确失败。
4. 派生独立 shape_truth_status：清楚的 D/P/L/R 即使 purpose unknown、role/status uncertain，仍可进入 shape-only 指标；unknown shape 和明确 excluded 不进入。不要改原 XML/truth。
5. 主指标以全部 shape-eligible episode 为分母。unavailable、boundary_uncertain、inference_error、abstention 分别报告并作为 pipeline miss 影响 recall；另报 ready-only 条件指标，但不能把它当主成绩。
6. 输出四类/二类 precision、recall、F1、support、confusion、raw counts、QC coverage、15/40 秒时长分层、participant/session/setup support、purposeful-hard-negative 形态诊断和逐 episode failures。类别缺失时完整 macro-F1 必须 not_computable。
7. 复用 EP1A 和已有简单 metric 原语，不复制 tracker/QC/preprocessing/model。必要时新建轻量 eval config/module/CLI/test；不要增加文件 SHA、字节对齐、授权收据或攻击型 exact-schema 工程。
8. 顺手关闭两个 P1：为两个 EP1A bundle builder 补真正端到端的 synthetic/non-overwrite 测试；补低置信度边缘 observation 测试并让 raw detection/boundary-gap QC 与 accepted observation/bucket 口径一致。改动必须小，并回归 D01/S00。
9. 先用 D01/S00 加现有真实 P/L/R 小样运行；如果 P/L/R 尚未到位，完成工具与可复跑的 direct-only smoke，准确报告 implementation_complete + data_pending，列出最少缺什么，不伪造标签、不自动开始 EP2。
10. 更新任务表、技术路线、mental-health README、docs 索引和本阶段 report；只有真实四类小样到位并满足任务书完成门，才把 EP1B 标为 completed。

硬边界：不训练、不调参、不改冻结候选/0.5 阈值/类别定义，不按模型结果改标签，不实现 automatic boundary/告警/AlgorithmEvent，不访问 sealed 或未授权数据，不修改其他模块，不 commit/push。普通本地代码、路径、测试和可回滚实现问题由你自主解决，不要逐步请示。

环境与验证：所有 Python 和 pytest 必须使用 WSL 项目 conda 环境 eldercare-ai；先确认 editable 安装指向当前仓库。修改 Python 后先跑最窄相关测试，再跑 EP1A 与相邻 camera 回归；不要用裸 python/pytest。真实视频只使用负责人已经提供的本地 development 数据，原视频不进 Git，输出使用全新目录。

最终交付必须包含：实现文件、测试命令与实际结果、真实数据覆盖表、主指标与 ready-only 指标、失败案例、未解决的数据缺口、准确的证据范围、下一步建议。先讲是否真正完成；不能用测试通过冒充真实四类数据完成，不能用 oracle-boundary 成绩冒充自动端到端效果。
```
