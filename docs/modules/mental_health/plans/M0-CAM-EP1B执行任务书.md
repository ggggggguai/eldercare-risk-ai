# M0-CAM-EP1B 执行任务书

> 历史执行记录：本任务的实现和指标继续有效，但其后继“等待独立 participant/setup”门禁已被 [M0-CAM-5D 快速交付总任务书](M0-CAM-5D快速交付总任务书.md)取代。B01+B02 现在可反复用于 development。

日期：2026-08-15；研究语义修订：2026-08-16；完成记录：2026-08-17

状态：`completed_manual_cvat_oracle_boundary_descriptive_pilot`

历史后继记录：`closed; old_next=M0-CAM-EP2A independent participant/setup validation`

现行任务：`M0-CAM-5D W5D-00 -> W5D-05`

推荐执行模型：`GPT-5.6 Codex / gpt-5.6-sol`

推荐智商（推理强度）：**最高（max）**

## 1. 本阶段只完成什么

把 EP1A 已经跑通的单视频 oracle-boundary 推理，整理成一条可批量复跑、可正确计算分母的摄像头 shape 小样本评估线。摄像头侧以独立 binary head 的 wandering-like shape 为主要结果，四分类保留为 subtype diagnostic：

```text
EP1A prediction bundle + 独立 truth view + 匿名分组信息
  → 一对一连接并检查边界身份
  → 全部 shape-eligible episode 为主分母
  → 主要 shape-binary 指标 + 四分类 subtype diagnostic
  → QC coverage、时长分层、目的性困难负例诊断
  → 失败清单与下一轮补拍建议
```

完成标准是“工具、测试和真实标注小样报告可以复跑”，不是达到某个分数。当前样本量只支持 descriptive development pilot，不能宣称 camera 95%、自动边界、最终告警、真实老人或临床效果。

本阶段只评价 Martino-Saltzman 层面的可观察轨迹形态，不评价 Algase 层面的临床徘徊综合征。`wandering_like` 只能解释为 shape binary，不能直接改名为 wandering/alert。目的、持续性和告警继续分层。

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

截至 2026-08-16，本机现有原始剪辑库存仍为：D01 12 条、P01 4 条、L01 3 条、R01 3 条、H01–H05 共 9 条、N01 1 条、Q01–Q03 共 3 条，另有 S00 1 条及其 XML。P01/L01/R01/H01/H04/H05 的 tracking、whole-clip boundary 和 EP1A prediction bundle 已形成；但对应 15 条标签是执行 AI 根据原始画面、contact sheet 和 truth-free 轨迹生成的预标注，未经人工标注者确认，不能称为独立 truth。S00 三条 direct 来自既有 CVAT XML。

2026-08-17 确认 `D:\徘徊数据集\自采数据\B01/B02` 有 50 条真实视频和 80 条 `source=manual` CVAT track，raw D/P/L/R=`46/12/5/14`。负责人随后确认 participant/session/setup/clock、CVAT 盲标事实和 development/held-out，并裁决 B01 out-of-frame track 56；裁决只应用于派生 reviewed XML。B01 development 与 B02 一次性 frozen-video holdout 已完成，详见 [intake/cohort freeze](../../../../reports/mental_health/wandering_camera_b01_b02_intake_v1/README.md)和[正式结果](../../../../reports/mental_health/wandering_camera_b01_development_v1/README.md)。

当前 `0.260417/0.756410` 只是一份 AI-preannotation provisional diagnostic。它证明 evaluator、分母和失败清单可工作，也暴露 pacing/random 在当前机位的 subtype 坍缩；它不构成真实 camera 性能。下一步必须先完成人工 truth 复核，不按现有预测或 AI 建议回改标签。

目录名和拍摄脚本不是正式真值。尤其 D01 中若实际出现明显折返、回到起点或混合行为，必须按画面复核并重新定 episode；不能因为文件位于 `D01` 就批量写成 direct。

## 3. EP1B 的范围

### 3.1 必须做

1. 实现轻量的多 bundle oracle-boundary 评估入口；
2. 复用 EP1A 输出，不复制 tracker、Camera QC、预处理或冻结模型 forward；
3. prediction 与 truth 只在推理完成后连接；
4. 输出主要 shape-binary 与四分类 subtype diagnostic 的 precision、recall、F1、support、confusion 和原始计数；
5. 主指标保留全部 shape-eligible truth，不通过删除 unavailable/error 美化成绩；
6. 另报 ready-only 条件指标、QC coverage、时长分层和失败案例；
7. purposeful hard negative 保留实际形态，在 shape 指标中仍按 pacing/lapping/random 计；
8. 用 D01/S00 做回归，并在人工确认的真实 P/L/R 小样到位后形成首份 shape pilot 报告；不要求高四分类分数作为完成门。

### 3.2 明确不做

- 不实现 automatic/semiautomatic boundary；
- 不调用旧 40 秒 window merge 作为 episode 真值；
- 不重训、不微调、不改模型权重；
- 不改 0.5 二分类阈值，不增加 OOD/uncertain 阈值；
- 不按预测修改 XML、truth、边界或标签；
- 不把 purposeful pacing/lapping/random 改成 direct；
- 不输出风险、告警或 `AlgorithmEvent`；
- 不扩展 receipt、PORTABLE、逐文件 hash、字节对齐、source anchor 或 exact-schema 攻击审计；
- 不因数据不足降低 EP1B 的人工 truth 与完成门；独立的 EP2A-S0 implementation-only 可以并行，但不能借此计算无真值成绩或宣称 EP1B/EP2A 完成。

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

### 6.1 人工 episode 边界协议

EP1B 使用 oracle boundary，因此边界本身必须先有统一人工协议：

- 先完整观看视频，不看模型预测，不把目录名、拍摄脚本或 AI 预标注当答案；
- 大约连续迈出三步、行走已经成立时开始；入场、站位和准备动作不计入；
- 数秒停顿、犹豫、转向、折返和闭环仍属于同一 locomotion episode，不在这些位置切段；
- 同一地点持续停留约 15 秒、坐下或明确开始另一活动时结束；
- track end、出画、长 gap、ID switch 和 session end 是硬边界；持续静止或活动改变是软边界；
- 若 A→B 后立即返回且没有可观察停留或任务切换，不能为了得到两个 direct 强行切开；保留完整 episode，目的另标；
- 起终点接近只是形态特征之一，不能单独决定 lapping、wandering-like 或告警；
- 边界不可可靠判定时标 `boundary_uncertain`，不伪造精确边界。

三步和约 15 秒是人工协议锚点，不是冻结模型输入、类别定义或自动 producer 的既定阈值。EP2A 只能在 development 标注上另行验证自动状态机。

## 7. 指标分母

### 7.1 主指标：all shape-eligible episodes

主表以全部 `shape_truth_status=eligible` 的 episode 为分母：

- `ready` 且有合法类别：进入四类和二类 confusion；
- `unavailable`、`boundary_uncertain`、`inference_error` 或明确 abstention：分别计数，并作为该真值类别未正确识别影响 recall/F1；
- confusion 可以增加 `pipeline_miss` 列，但它不是第五种行为标签；
- 不允许把非 ready episode 从主 support 中删除。

四类：`direct/pacing/lapping/random`，用于 subtype diagnostic。

二类映射：

```text
direct → direct_or_non_wandering
pacing/lapping/random → wandering_like
```

摄像头侧报告优先级固定为：

1. all-shape-eligible shape-binary precision/recall/F1、macro-F1、support 和 confusion；
2. ready-only shape-binary conditional diagnostic；
3. all-eligible 与 ready-only 四分类 subtype diagnostic；
4. QC、coverage、duration、group、purposeful 和 failure 分层。

四分类仍必须完整输出，但 pacing/lapping/random 互混不再单独阻止 camera binary 工具线完成。公开 WP 的四分类目标和冻结类别定义保持不变。

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
- 先对现有 P01/L01/R01/H01/H04/H05 共 15 条 AI 预标注进行盲于模型预测的人工复核；
- pacing/lapping/random 至少各保留一批边界和形态清楚的人工确认 episode；
- 每条实际完成就结束，不强凑 40 秒；
- 先检查 importer、identity join、QC、指标和失败输出。

### Pilot 2：形成 EP1B 小样

- 若人工复核后 D/P/L/R 均有清楚支持，先形成 binary-primary、four-class-diagnostic 的 descriptive pilot；
- pacing/lapping/random 各逐步扩到约 10 条可评分完整 episode，作为 subtype 诊断与采集质量目标，不作为 binary 工具线的硬完成门；
- direct 复用已有合格样本，必要时补少量不同起终点和速度；
- 同时积累打电话、找东西、清洁、锻炼、搬运等 purposeful hard negatives，但不要求它们阻塞第一版 binary-primary shape 报告；
- 单人、单办公室、单机位仍只能称 descriptive development pilot。

不要过分规定人物怎样走。脚本只给大致形态和活动目的，最终标签按画面实际形成的轨迹，不按计划动作硬贴。

为得到清楚的 camera subtype 诊断样本，拍摄提示可以比自然场景更明确，但不能变成标签规则：

- pacing 优先让往返轴横跨画面，并形成约 3 次清楚方向反转；
- lapping 围绕可见参照物完成约 2 圈；
- random 访问至少 3 个分离区域并多次改变方向，避免退化成单轴往返或固定绕圈；
- 正式标签仍按实际画面；动作未形成时改标实际类别或 unknown，不按脚本硬贴。

现有视频的用途必须分开：

- P01/L01/R01：现有 tracking/bundle 保留，先独立人工复核 AI 预标注；清楚样本进入 core shape pilot，不清楚样本保留 uncertain；
- H01–H05：只把人工划出的清楚 P/L/R episode 纳入 shape 指标，并单列 purposeful-hard-negative subgroup；现有 H01/H04/H05 也仍需人工确认；
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
- EP1B 冻结模型、阈值、类别和人工 truth 不得根据 B02 prediction 回调，也不得把 oracle-boundary EP1B 成绩写成 automatic shape。2026-08-17 负责人只授权了独立版本的 S2A 非模型送模门禁调整；它不授权修改 EP1B。

## 13. 完成判定

可以标记 `m0cam_ep1b_status=completed`，必须同时满足：

- 批量 evaluator、CLI 和测试已落地；
- 两个 EP1A P1 回归已关闭；
- 至少一批真实、独立标注的 D/P/L/R episode 从 EP1A bundle 进入 evaluator；
- 旧 AI 预标注若进入正式 cohort，必须由人工逐条确认/修改；否则明确排除在正式 batch 之外，不得覆盖原预标注或 XML；
- all-eligible shape-binary 主要指标、ready-only 条件指标、四分类 subtype diagnostic、coverage、duration、group support、purposeful diagnostics 和 failures 均输出；
- 报告准确限制结论，不写 95%、FAR、自动边界或告警；
- D01/S00 相邻回归没有退化。

用户已明确把目标摄像头的四分类降为 subtype diagnostic，因此“每个 pacing/lapping/random 约 10 条”保留为后续扩样目标，不再单独阻塞 EP1B 完成。若人工复核后某一类没有任何清楚样本，则仍不满足真实 D/P/L/R 门，应继续 data pending。

2026-08-17 完成状态：

```text
m0cam_ep1b_implementation_status=completed
m0cam_ep1b_human_truth_status=reviewed_manual_cvat
m0cam_ep1b_evaluation_status=completed_descriptive_pilot
m0cam_ep1b_data_status=manual_cvat_b01_development_plus_b02_video_holdout
m0cam_ep1b_status=completed
```

B01 development 的 D/P/L/R support 为 `36/9/3/9`，binary all-eligible accuracy/macro-F1=`0.807018/0.817375`；B02 frozen-video holdout support 为 `9/2/2/5`，binary=`1.0/1.0`。旧 15 条 AI 预标注没有进入正式成绩。该完成状态只覆盖人工 CVAT、oracle-boundary、shape-only descriptive pilot，不覆盖 automatic boundary/shape、alert、跨 participant/setup、老人域或临床效果。完整结果见 [B01 development / B02 frozen-video holdout](../../../../reports/mental_health/wandering_camera_b01_development_v1/README.md)。

## 14. 后续执行入口

EP1B 不再是当前开发任务。后续 AI 的完整目标提示词维护在 [M0-CAM EP1B / EP2A 证据交接](M0-CAM-EP1B-EP2A-S1B人工证据交接.md)，实时状态只看 [任务表](../../../tasks/README.md)。任何复跑必须保留 B01/B02 历史结果；负责人授权的 B02 复用只覆盖版本化 S2A v2 送模策略，不覆盖 EP1B 模型/阈值/类别/truth。不得把 oracle-boundary shape 成绩冒充 automatic pipeline 性能。
