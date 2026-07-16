# 跌倒风险数据与 Split 版本记录

记录日期：2026-07-15

目标名：`fall-risk-data-v1`

发布状态：`release_candidate_blocked`，不是 frozen 版本。

## 数据版本

| 项目 | 版本/状态 | SHA-256 |
|---|---|---|
| 全量 manifest | `fall-risk-data-v1-candidate`，3,454 assets / 2,440 videos | `6cbb45d7f5f11aabe0239426022e9df31ba9c4383078ae05dfce870bb5a7e049` |
| 根 action 标签 | 922 pending，旧 schema | `e72eeed3067d8c1bc75ddca732deffc5a5defefe37fbd618e9ba2e05bccc885a` |
| 根 event 标签 | 922 pending，旧 schema | `a200a39b7119d7468232e487ef45c0a4e8dc728c2dc0610a15add1dbe64bede9` |
| risk 标签模板 | 0 records | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| review log 模板 | 0 records | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| subject profile 模板 | 0 subjects | `aa92c571bf1316e58af81866c6a3028b99f7ad3d8c03686f7333a513ec68c8f2` |
| 标签校验配置 | `fall-risk-label-validation-v1` | `5fba8c6db22960c60ecc9ebcb1ebff474b13fb0e377d01173c6ddb197b448d4f` |
| split 开发配置 | provisional | `66230c65e6b79ceab581033a41fbd62900fa1e2b6cdf349eb0c8734dec63bd0c` |

## Split 版本

所有 assignments 文件均为空文件 SHA-256 `e3b0c442...b855`。这是带 blocker 的事实产物，不是可评估 split。

| 任务 | 状态 | 样本 | split_id | split.json SHA-256 |
|---|---|---:|---|---|
| `fall_event_v1` | blocked | 0 | `null` | `846617cee6925674d36d08d1947b3f54fadf696e18af47ec6a37b6df8bee0a7c` |
| `near_fall_event_v1` | blocked | 0 | `null` | `908acaee1f6897c3f9117753aece373c730363f2e262e02ba362b89e24725655` |
| `functional_proxy_v1` | blocked | 0 | `null` | `7bbb156bbb085f01ed1e9dbf66be4fcf25e929ede1b83aa61456d0872e895a75` |
| `longitudinal_baseline_v1` | blocked | 0 | `null` | `59679ffdcdf10fe26912e443bd91cd2fd242e15617ec0028b469ba8a99bd9709` |

共同 blocker：没有同时满足 manifest 可用、标签 `eligibility=true`、`reviewed/final`、双人独立复核证据和保守泄漏标识的样本。blocked 产物绑定真实 manifest SHA；不会生成假的 split 根哈希。

## 评估协议

| 协议 | 状态 | 配置 SHA-256 |
|---|---|---|
| `fall-event-eval-v1` | `development_provisional` | `b898915107ae6cc5e54509988ddfb102aeff4dbbd8c8eded700277bc3e15eb60` |
| `near-fall-event-eval-v1` | `development_provisional` | `06818cb4eeb32e4161fe70e004ce36a5ed2d5ce01d701ea10d0c93d39931434e` |

正式协议必须由负责人预注册并另存为 frozen 版本；不得根据测试标签或结果回改当前阈值。frozen split 入口会同时校验 manifest、六类标签/复核输入、validation config 和 formal validation report 的文件哈希，并拒绝不安全的校验配置。

## 合成证据包

- synthetic split ID：`fall_event_v1_synthetic:sha256:5bb3684c004f73394b321a77c11bf3cba7c86606602801e391da1e2c618a505d`
- truth SHA-256：`fe81fa0a387baf25f29671678e089704e4db78f154df4cd39248f3adafa86513`
- prediction SHA-256：`0661d70ce40d1cb5963abda1542fd25fb42851e01a02ddff171348fd43f10290`
- manifest SHA-256：`d13f70410d0cd6c1e46ff92d65efb06d0dd8689df82d6c7b30a5acc08e574baf`
- evaluation implementation SHA-256：`22bd1dbde51f56ae36db68640429449d7b23b6e1a15c6b11cc001eb7fa00015e`
- 结果：1 TP、0 FP、0 FN、F1 1.0、传统 FPR/FP 每小时/FP 每家庭日均为 `null`。

该包只证明 CLI、哈希绑定、匹配和证据输出可复现，不是实验指标。

## 复现入口

```bash
conda run -n eldercare-ai python scripts/split/build_fall_risk_splits.py \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --annotations-dir data/annotations/fall_risk \
  --config configs/data/fall_risk_splits_v1.yaml \
  --validation-config configs/data/fall_risk_label_validation_v1.yaml \
  --output-dir /tmp/fall-risk-splits-rebuild

conda run -n eldercare-ai python scripts/evaluate/build_synthetic_fall_event_fixture.py \
  --output-dir /tmp/fall-risk-synthetic-input \
  --evaluation-config configs/evaluation/fall_event_v1.provisional.yaml

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

所有 `/tmp` 输出路径必须在运行前不存在；默认 no-overwrite 行为是版本保护的一部分。
