# M0-RS verification

日期：2026-08-13

## 执行前身份与现场

按 `AGENTS.md` 使用 WSL `Ubuntu-22.04` 的 `eldercare-ai`。`pip show elderly-monitoring-algorithms` 的 editable location 指向当前桌面 checkout。正式命令前只读核验：

- v3 manifest 为 12,642 bytes，SHA-256 `3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7`；candidate 为 `seed=20260731 / best epoch=5`；
- model state、forward config、performance config、RF config、training identity、release identity 与 M0-RH 登记一致；
- v3 source archive 为 355,058 bytes，SHA-256 `30ffec61c620643591b155a66bff3b753ff3a3ff7772eddc5e9d423b98434c0a`，精确 7-entry bytes 与 active source 一致；
- Git HEAD 为 `2981583736a74b1328e0fa52bf39028d6a3f8f7f`，初始 Git staged 集合为空；用户既有脏改动保留；
- 没有同名 score 进程、正式输出目录或 `.m0r_public_holdout_score_v1-*` staging。

## 宿主传参与正式执行

Windows PowerShell 没有裸 `conda`，因此使用 WSL 环境。第一次宿主调用因 PowerShell→WSL 嵌套引号重建问题返回 0，但没有 CLI JSON、进程、final、staging 或任何结果文件。冻结 CLI 的成功路径只有在 `run_authorized_frozen_wp_score()` 返回、final 已原子提交后才会打印 `formal_score_committed=true`，所以该现场证明第一次调用没有进入正式 handler，也没有形成或观察分数。

随后用 Base64 只传输 WSL shell 脚本文本，先运行无数据的 `score-frozen-wp --help` 探针，确认只暴露四个冻结参数。之后从仓库根执行与授权文本完全相同的唯一命令行。该次 handler 在约 46 秒后返回：

```json
{"candidate_manifest_sha256":"3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7","formal_score_committed":true,"metrics_not_printed":true,"mode":"score-frozen-wp"}
```

提交后没有再次调用 scorer、accessor 或模型推理。

## 正式六文件核验

正式目录 `reports/mental_health/wandering_performance/m0r_public_holdout_score_v1/` 精确包含：

| File | bytes | SHA-256 |
|---|---:|---|
| `artifact_manifest.json` | 733 | `0444c141b8878619c88c86ab05ddb48ed4c2d27b194292fc43e04060fe77a17c` |
| `execution.json` | 10,847 | `8d5f920a59de103d0df14b59c65ee3fa83e346e567bda7ebbfc883e1c447470f` |
| `predictions.jsonl` | 141,700 | `dee9d61aed13f04cf30d54bc47498f48e2291a9bbfe88979a318e46112f35195` |
| `metrics.json` | 1,392 | `ba9ffa370957ed662e202e504b241d469d16141dfd701eed9f9556272f2d2c42` |
| `confusion.json` | 162 | `ea97efcc8cec7cb2d24c918f9af68dd7ce8aa7b7f67a1e7d9284554fa6613c37` |
| `errors.jsonl` | 896 | `58f181c78d61591064c6dd631c7865f75c38d5382fb0eb55fb960b8953488aa7` |

Artifact manifest 登记的五个 payload 文件集合、size、SHA 全部与磁盘一致，`exact_artifact_set_verified_before_commit=true`、`same_filesystem_atomic_commit=true`。所有 JSON/JSONL 可逐行解析，正式目录没有被添加 sidecar，父目录没有残留 staging。

## 独立复算

使用只读 PowerShell 从 `predictions.jsonl` 独立重建真值/预测 confusion、每类 precision/recall/F1、macro-F1 和错误集合，没有调用项目 evaluator：

- 240 条、ID 唯一、全部 `source_dataset=wandering_patterns`、`split=test`；
- 四类真值各 60，binary `0/1=60/180`；
- 四分类 macro-F1 `0.9833310178753357`，recall 为 `1.0 / 0.9833333333333333 / 0.9833333333333333 / 0.9666666666666667`；
- 二分类 macro-F1 `0.9890099825991392`，recall 为 `1.0 / 0.9888888888888889`；
- 四分类 confusion `[[60,0,0,0],[0,59,1,0],[0,1,59,0],[2,0,0,58]]`；binary confusion `[[60,0],[2,178]]`；
- predictions 派生的 4 个错误 sample ID 与 `errors.jsonl` 精确一致，其中四分类错误 4 个、binary 错误 2 个；
- 每条 binary 决策都等于 `probability >= 0.5`，四分类决策都等于固定顺序上的层级概率 argmax，四分类概率和在 `1e-12` 内为 1；
- 按 accessor 顺序从 predictions 重算 ordered sample-ID SHA 为 `18edd572fb3224c0103ca8b751d5f8befe9a1c8dbbdba6943090bf2081e760d4`，与 execution 一致。

六个 recall 全部 `>=0.90`，两项 macro-F1 全部 `>=0.95`，所以 `performance_status=target_met`。

## Execution/preflight 审计

`execution.json` 确认：

- `phase=test`，accessor mode `frozen_wp_test`，records method `records_for_split`；
- canonical records SHA `ffc88ac6f7937ca53128488bcc62749f3eafffaad4c897c0a503722a17bf9ae4` 与 ordered sample-ID SHA 非空；
- active execution source、source archive、training/release identity cross-binding、training active/snapshot source、RF/upstream descriptors 全部 verified；
- `preflight_completed_before_accessor=true`、`preflight_completed_before_model_inference=true`；
- runtime 为 CPU `8/1`、batch64、workers0、pin-memory false，`eval=true`、`torch_inference_mode=true`；
- `fit=false`、`optimizer=false`、`threshold_search=false`、`model_or_threshold_modified=false`；
- 仅 `wp_public_holdout=true`；WP raw、SmartCare official/raw、sealed camera 均为 false。

## Post-hoc primary-seed 对照

只有正式结果提交后，才读取 Step5 RF 和 Step6 TCN 已保存的 primary `seed=20260731` test predictions。四组均为同一 240 条 sample ID 和真值。逐样本正确性 contingency 为：RF four-class `229/7/4/0`，RF binary `231/7/2/0`，TCN four-class `219/17/2/2`，TCN binary `236/2/2/0`，顺序是“两者都对/仅 MPT 对/仅 baseline 对/两者都错”。没有读取其他 seed 做选择，没有重训 baseline，也没有把 paired 解释反馈成模型修改。

## 验收与副作用

- 计分代码、候选和配置未修改，因此按任务约束没有重复完整 pytest；验收限定为正式输出独立复算、preflight evidence、文档链接、sidecar 和 `git diff --check`。
- 未执行 M1、重训、ensemble、seed/epoch/checkpoint 重选、阈值/标签/split/协议修改。
- 未访问 WP raw、SmartCare official/raw 或 sealed camera。
- 正式六文件未移动、改写或添加文件；v1/v2/v3 和其他旧证据未覆盖。
- 没有 stash、reset、commit、push、发布或部署；用户既有脏改动保持不动。
- 下一阶段仅在任务/文档中交接到 camera development，本 Goal 没有实现或运行摄像头阶段。
