# 评估脚本目录

当前已实现跌倒/近跌倒事件评估器，以及跌倒 candidate-clip TCN、坐站规则/Logistic/候选 TCN 的 validation-only 复评入口。功能 proxy 与纵向任务只有 schema 和 split 门禁；在真实参考终点存在前，不生成正式指标。

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

跌倒 candidate-clip TCN 使用当前 v3 split 生成的数据集，并只在 metadata 记录的 validation 分区复算固定 checkpoint：

```bash
conda run -n eldercare-ai python scripts/evaluate/evaluate_fall_event_tcn.py \
  --data data/processed/fall_risk/fall_event_proxy_v2_v3split/dataset.npz \
  --metadata data/processed/fall_risk/fall_event_proxy_v2_v3split/metadata.json \
  --checkpoint reports/fall_risk/fall_event_proxy_v2_v3split/pilot-seed42/best_model.pt \
  --output /tmp/fall_event_validation.json
```

坐站 provisional 数据集分别提供规则 E0、Logistic 和候选双头 TCN 的 validation 复评：

```bash
conda run -n eldercare-ai python scripts/evaluate/evaluate_sit_stand_rule.py \
  --data data/processed/fall_risk/sit_stand_event_v1/provisional-20260803-seed42-v3/dataset.npz \
  --metadata data/processed/fall_risk/sit_stand_event_v1/provisional-20260803-seed42-v3/metadata.json \
  --output-dir /tmp/sit_stand_rule_validation

conda run -n eldercare-ai python scripts/evaluate/evaluate_sit_stand.py \
  --data data/processed/fall_risk/sit_stand_event_v1/provisional-20260803-seed42-v3/dataset.npz \
  --metadata data/processed/fall_risk/sit_stand_event_v1/provisional-20260803-seed42-v3/metadata.json \
  --checkpoint reports/fall_risk/sit_stand_event_v1/pilot-logreg-provisional-20260803-seed42-v2/checkpoint.joblib \
  --output /tmp/sit_stand_logistic_validation.json

conda run -n eldercare-ai python scripts/evaluate/evaluate_sit_stand_tcn.py \
  --data data/processed/fall_risk/sit_stand_event_v1/provisional-20260803-seed42-v3/dataset.npz \
  --metadata data/processed/fall_risk/sit_stand_event_v1/provisional-20260803-seed42-v3/metadata.json \
  --checkpoint reports/fall_risk/sit_stand_event_v1/pilot-candidate-tcn-provisional-20260803-seed42-v1/best_model.pt \
  --output /tmp/sit_stand_tcn_validation.json
```

这些入口会拒绝 test 评估或只接受 `partition=validation`。模型权重和 `.joblib` 是本地忽略产物；命令路径用于复现实验，不表示仓库发布 checkpoint。候选 clip 指标不等于连续事件定位、连续背景误报率、老人域泛化或正式测试结果。

## 正式门禁

正式评估必须同时满足：

- split 与 evaluation protocol 均为 frozen；
- Git 工作区干净，bootstrap 不少于 10,000 次；
- manifest、assignments、标签、split 根哈希和环境绑定全部匹配；
- `--validation-report` 指向的实际 formal report 哈希与 frozen split 完全一致；
- test 分区提供绑定配置、预测、标签、commit 和唯一 run ID 的保管人授权记录；
- 测试集未用于阈值选择或调参。

当前仓库不满足这些条件，不得运行或宣称正式测试指标。
