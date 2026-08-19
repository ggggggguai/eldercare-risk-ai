# M0-CAM-EP1B verification

日期：2026-08-15；功能复审更新：2026-08-16

环境：所有 Python/pytest 使用 WSL `eldercare-ai`；editable project location 指向 `/mnt/c/Users/lenovo/Desktop/心理算法`。

## 聚焦回归

```bash
conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_episode_evaluation.py \
  tests/test_wandering_camera_episode_import.py \
  tests/test_wandering_camera_episode_inference.py
```

初版结果为 `25 passed in 12.61s`；补充 prediction row 与 summary candidate/model 身份一致性拒绝并完成 import 清理后，最终结果为 `28 passed in 11.99s`。

覆盖 identity join 拒绝、all-eligible/ready-only 分母、四种 pipeline miss、purpose/shape 分离、缺类 macro-F1、15/40 秒边界、builder E2E/non-overwrite 与低置信度 observation QC。

2026-08-16 功能复审先以新增回归暴露 QC/prediction-ready 混用、EP1A summary 语义未校验、全批 runtime 状态缺失、purposeful role mismatch 静默漏诊断和失败上下文不足，然后完成修正：

```bash
conda run -n eldercare-ai python -m pytest -q tests/test_wandering_camera_episode_evaluation.py
```

最终结果：`40 passed in 4.10s`。覆盖真实 producer 风格的 `inference_error/abstention + qc_status=ready`、非法 QC/runtime/reason 状态矩阵、冻结/oracle summary、全批与 eligible 状态计数、独立 binary head 与精确 0.5 平票规则、four-class 概率/argmax 和 binary/four 层级关系、同一物理 episode 换 ID 或重叠区间的拒绝、跨 bundle group identity 一致性、显式 group count/逐类支持、purposeful role mismatch、不可评分 truth，以及 CLI 的结构化 identity/missing-index rejection 且失败时不写成功 bundle。

evaluator + truth importer 聚焦结果：`48 passed in 19.71s`；EP1B + EP1A 聚焦结果：`59 passed in 17.92s`。非字符串 truth/QC/boundary enum 现在返回领域错误，不再泄漏裸 `TypeError`。

## AI 预标注诊断运行

- 19 条 P/L/R/H 原始视频完成 contact-sheet 与 truth-free target-track 轨迹审查；
- 执行 AI 在 shape forward 前生成 15 条 whole-clip 预标注，并通过 `load_episode_boundaries()` 与 `load_episode_truth()`；它们未根据预测生成，但尚未经过人工复核，因此不是独立 truth；
- 15 个新 EP1A bundle 为 `14 ready / 1 unavailable`；
- 加入 S00 三条既有 CVAT direct 后，evaluator 严格连接 18/18 label-view/prediction；
- 输出 `18 shape eligible / 17 ready / 12 failures`；2026-08-16 完成 binary-head、层级概率、overlap/group identity 与 enum 输入加固后的当前代码 machine result 位于 `tmp/m0cam_ep1b_pilot_20260815_v1/evaluation_current_code_20260816_03/`。复跑时必须另选尚不存在的 fresh output 目录，不能复用或覆盖该目录。
- 新目录复跑保持 all-eligible 四分类/shape-binary macro-F1=`0.260417/0.756410`、ready-only=`0.260417/0.773333` 和 QC coverage=`17/18` 不变；17 条 ready 的 binary head 与 four-class 派生二类恰好 0 分歧，但 evaluator 已改为使用独立 binary head。失败拆分为 11 个 four-class mismatch、3 个 binary mismatch 和 1 个 pipeline miss（可重叠，共 12 个 episode）。这些 mismatch 相对于 AI 预标注计算，只证明 evaluator 行为；人工 truth 到位前不能称为模型错误率。另新增 prediction-ready coverage、全批 runtime 状态、group counts/support、purposeful role mismatch 和完整失败上下文。

## 相邻与完整 camera 回归

EP1A 加 adapter/QC/primary/legacy/corruption 相邻回归：

```bash
conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_episode_evaluation.py \
  tests/test_wandering_camera_episode_import.py \
  tests/test_wandering_camera_episode_inference.py \
  tests/test_wandering_camera_qc.py \
  tests/test_wandering_camera_primary_inference.py \
  tests/test_wandering_camera_episode.py \
  tests/test_wandering_camera_inference.py \
  tests/test_wandering_camera_adapter.py \
  tests/test_wandering_camera_corruption.py
```

历史结果：`142 passed in 73.91s`。2026-08-16 当前实现复审结果：`176 passed in 108.18s`。

全部 camera 回归：

```bash
conda run -n eldercare-ai python -m pytest -q tests/test_wandering_camera_*.py
```

初版结果为 `536 passed, 1 warning in 142.11s`；身份一致性加固后为 `539 passed, 1 warning in 144.91s`；2026-08-16 当前实现复审为 `570 passed, 1 warning in 232.89s`。唯一 warning 来自既有 PORTABLE 重复 ZIP 条目拒绝测试。

## EP1A 真实样片回归

在全新 `tmp/m0cam_ep1b_ep1a_regression_20260815_v1/` 重新运行 D01 12 条 whole-clip 与 S00 三段：

```text
D01: 12/12 ready/direct
S00: 3/3 ready/direct
common_fields_exact=true
accepted_observation_count_equals_source=true
```

与 `tmp/m0cam_ep1a_formal_20260815_v2/` 逐 JSON 字段比较，除新增 `accepted_observation_count` 外所有 prediction 和 summary 字段完全一致。

## 交付卫生

```text
evaluation CLI --help: passed
episode inference CLI --help: passed
compileall for changed modules/CLIs: passed
new EP1B Markdown links: 5 checked, 0 broken
git diff --check: passed
```

全量扫描 `docs/README.md` 另发现 3 个既有 fall-risk machine-report 链接目标缺失；它们不是本轮新增链接且属于其他模块，本任务未改动。本轮新增 EP1B 链接及新报告内部链接均可解析。

## 证据解释

当前索引有 2 个匿名 participant group、7 个 session、1 个 setup，但 15/18 episode 集中在同一 participant group。更重要的是，这 15 条仍是 AI 预标注，因此当前结果不是 descriptive model pilot。测试通过只证明 evaluator 与 EP1A 工程行为；human truth 未满足 EP1B completion gate，不能把 `implementation_complete` 写成 camera 模型完成。后续允许并行开发 EP2A-S0 proposal 工具，但不能把 proposal、truth-free smoke 或测试通过写成 EP2A 已评价或 EP1B 已完成。
