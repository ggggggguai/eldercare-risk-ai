# 评估脚本目录

当前已实现跌倒/近跌倒事件评估器，以及坐站连续因果 TCN 与规则对照的 validation-only 评估入口。功能 proxy 与纵向任务只有 schema 和 split 门禁；在真实参考终点存在前，不生成正式指标。

## 事件评估契约

预测 JSONL 至少包含：

```text
video_id, task_type, event_type, prediction_id, score,
start_time, end_time, onset_time, status, quality_state,
model_version, config_hash, split_id
```

评估器按同一 `video_id/task_type` 和配置允许的事件类型做确定性一对一匹配。PR 曲线在每个 score 阈值重新匹配；AP 方法固定记录为 `average_precision_step`。只有 `eligibility=true` 且 `reviewed/final` 的真值会进入指标。

输出目录固定包含：

```text
metrics.json
matches.jsonl
false_positives.jsonl
false_negatives.jsonl
excluded_samples.jsonl
threshold_curve.csv
report.md
```

传统 FPR 只有注册的 TN 单位才计算；`FP/摄像机小时` 和 `FP/家庭日` 只接受 manifest 中明确合格的连续分母。短事件剪辑不会折算为连续监控时间。fall detection latency、near-fall onset latency 和 recovery 指标分别统计。

## 合成烟测

先在不存在的临时路径生成确定性 fixture：

```bash
conda run -n eldercare-ai python scripts/evaluate/build_synthetic_fall_event_fixture.py \
  --output-dir /tmp/fall-risk-synthetic-input \
  --evaluation-config configs/evaluation/fall_event_v1.provisional.yaml
```

再运行 provisional 评估：

```bash
conda run -n eldercare-ai python scripts/evaluate/evaluate_fall_events.py \
  --ground-truth /tmp/fall-risk-synthetic-input/ground_truth.jsonl \
  --predictions /tmp/fall-risk-synthetic-input/predictions.jsonl \
  --manifest /tmp/fall-risk-synthetic-input/manifest.jsonl \
  --split /tmp/fall-risk-synthetic-input/split.json \
  --assignments /tmp/fall-risk-synthetic-input/assignments.jsonl \
  --partition validation \
  --config configs/evaluation/fall_event_v1.provisional.yaml \
  --output-dir /tmp/fall-risk-synthetic-bundle \
  --label-version synthetic-labels-v1 \
  --allow-provisional
```

`--allow-provisional` 只允许非正式的 train/validation 开发烟测。仓库证据包位于 `reports/fall_risk/workflow_a_synthetic_evaluation/`；其中满分结果只是 perfect-match fixture，不是模型或比赛性能。

## 候选模型复评

跌倒连续因果链当前只保留输入契约和审计入口，尚无可晋级 checkpoint；旧 candidate-clip pilot 仅保留在历史报告中：

```bash
conda run -n eldercare-ai python scripts/evaluate/evaluate_fall_event_tcn.py \
  --data data/processed/fall_risk/fall_event_proxy_v2_v3split/dataset.npz \
  --metadata data/processed/fall_risk/fall_event_proxy_v2_v3split/metadata.json \
  --checkpoint reports/fall_risk/fall_event_proxy_v2_v3split/pilot-seed42/best_model.pt \
  --output /tmp/fall_event_validation.json
```

坐站只保留连续因果 TCN 及同协议规则对照的 validation 复评：

```bash
conda run -n eldercare-ai python scripts/evaluate/evaluate_sit_stand_streaming.py --help
```

这些入口会拒绝 test 评估或只接受 `partition=validation`。模型权重和 `.joblib` 是本地忽略产物；命令路径用于复现实验，不表示仓库发布 checkpoint。候选 clip 指标不等于连续事件定位、连续背景误报率、老人域泛化或正式测试结果。

## 个体基线纵向消融

Phase 2 入口固定评估无个人基线、mean/std、median/MAD 和完整鲁棒候选四组，要求 prediction coverage 完全一致。先生成纯合成 fixture，再只评估 validation：

```bash
conda run -n eldercare-ai python scripts/evaluate/build_synthetic_fall_baseline_longitudinal_fixture.py \
  --output-dir /tmp/fall_baseline_longitudinal_synthetic

conda run -n eldercare-ai python scripts/evaluate/generate_fall_baseline_longitudinal_predictions.py \
  --observations /tmp/fall_baseline_longitudinal_synthetic/observations.validation.jsonl \
  --assignments /tmp/fall_baseline_longitudinal_synthetic/assignments.jsonl \
  --split /tmp/fall_baseline_longitudinal_synthetic/split.json \
  --partition validation \
  --output /tmp/fall_baseline_longitudinal_synthetic/replay-predictions.validation.jsonl

conda run -n eldercare-ai python scripts/evaluate/evaluate_fall_baseline_longitudinal.py \
  --observations /tmp/fall_baseline_longitudinal_synthetic/observations.validation.jsonl \
  --risk-labels /tmp/fall_baseline_longitudinal_synthetic/risk_labels.validation.jsonl \
  --assignments /tmp/fall_baseline_longitudinal_synthetic/assignments.jsonl \
  --split /tmp/fall_baseline_longitudinal_synthetic/split.json \
  --predictions /tmp/fall_baseline_longitudinal_synthetic/replay-predictions.validation.jsonl \
  --partition validation \
  --data-status synthetic \
  --output-dir /tmp/fall_baseline_longitudinal_bundle
```

该结果只证明前向 split、泄漏门禁、分层指标、人员级 bootstrap 和失败案例 bundle 可运行。真实 observation、`risk_labels` 与 subject profiles 当前为空；`--data-status formal` 只接受 frozen 协议和 frozen split，test 还要求绑定 split/协议/预测/标签/commit/授权角色/run ID 的 `--test-release-ack`，普通布尔开关不能开放 test。

## 正式门禁

正式评估必须同时满足：

- split 与 evaluation protocol 均为 frozen；
- formal validation/test 的 Git 工作区干净并记录 commit，bootstrap 不少于 10,000 次；
- manifest、assignments、标签、split 根哈希和环境绑定全部匹配；
- split 构建时，`--formal-validation-report` 中的 `risk_labels`、subject profiles 文件哈希与 CLI 从实际文件字节计算的 SHA-256 完全一致；
- test 分区提供绑定配置、预测、标签、commit 和唯一 run ID 的保管人授权记录；
- run ID 的单次消费由保管人工作流登记；无状态评估器只校验授权记录的哈希绑定，不能单独证明该 ID 未被重复使用；
- 测试集未用于阈值选择或调参。

当前仓库不满足这些条件，不得运行或宣称正式测试指标。
