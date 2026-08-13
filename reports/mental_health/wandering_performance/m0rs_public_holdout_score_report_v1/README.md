# TopoWander-MPT M0-RS fixed WP public-holdout score

日期：2026-08-13

完成状态：`wandering_m0rs_public_holdout_scored`

性能状态：`performance_status=target_met`

## 结论

负责人针对精确 v3 manifest 给出明确授权后，M0-S primary `seed=20260731 / best epoch=5` 已通过唯一正式入口 `score-frozen-wp` 完成一份固定 WP public-holdout 候选级计分。候选、模型字节、forward/performance config、阈值、类别规则、split 和标签均未改变；没有重训、ensemble、checkpoint 重选或 M1。

240 条 WP public holdout 上，WP 四分类 macro-F1 为 `0.9833310178753357`，WP 二分类 macro-F1 为 `0.9890099825991392`。两项 macro-F1 均达到 `>=0.95`，四个形状类别和 binary 两类的 recall 均达到 `>=0.90`，因此预注册判定为 `target_met`。该结果已经固化，不根据 test 错误调参或重跑。

科学边界：WP public holdout 历史上已经被契约测试解析，并被 Step5 RF 和 Step6 TCN 评分。本结果只说明固定 TopoWander-MPT 候选在 public-shape benchmark 上的表现，不是项目首次盲测，也不是目标摄像头、未见人员、真实老人、产品或临床有效性证据。

## 固定候选与 cohort 身份

- Candidate：TopoWander-MPT，M0-S `seed=20260731 / best epoch=5`
- v3 candidate manifest：12,642 bytes，SHA-256 `3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7`
- 正式 phase：`test`
- Source：`wandering_patterns`
- Sample count：240，sample ID 全部唯一
- 四类真值分布：`direct/pacing/lapping/random = 60/60/60/60`
- Binary 真值分布：`0/1 = 60/180`
- Canonical records SHA-256：`ffc88ac6f7937ca53128488bcc62749f3eafffaad4c897c0a503722a17bf9ae4`
- Ordered sample-ID SHA-256：`18edd572fb3224c0103ca8b751d5f8befe9a1c8dbbdba6943090bf2081e760d4`

Ordered sample-ID SHA 已从正式 predictions 的原始顺序独立复算一致。Canonical records SHA 由受信 accessor 返回的完整 canonical record bytes 在推理前生成并写入 execution evidence；叙述性审计没有再次打开 accessor。

## 正式机器输出

正式目录保持为 `../m0r_public_holdout_score_v1/`，精确包含以下六个文件，没有添加报告或 sidecar：

| File | bytes | SHA-256 |
|---|---:|---|
| `artifact_manifest.json` | 733 | `0444c141b8878619c88c86ab05ddb48ed4c2d27b194292fc43e04060fe77a17c` |
| `execution.json` | 10,847 | `8d5f920a59de103d0df14b59c65ee3fa83e346e567bda7ebbfc883e1c447470f` |
| `predictions.jsonl` | 141,700 | `dee9d61aed13f04cf30d54bc47498f48e2291a9bbfe88979a318e46112f35195` |
| `metrics.json` | 1,392 | `ba9ffa370957ed662e202e504b241d469d16141dfd701eed9f9556272f2d2c42` |
| `confusion.json` | 162 | `ea97efcc8cec7cb2d24c918f9af68dd7ce8aa7b7f67a1e7d9284554fa6613c37` |
| `errors.jsonl` | 896 | `58f181c78d61591064c6dd631c7865f75c38d5382fb0eb55fb960b8953488aa7` |

`artifact_manifest.json` 中登记的五个 payload 文件集合、bytes 和 SHA 全部与磁盘一致；六文件 JSON/JSONL 均可解析，正式目录旁没有残留同名前缀 staging。

## 指标、recall 与 confusion

| Task / class | Recall |
|---|---:|
| WP four-class macro-F1 | `0.9833310178753357` |
| direct | `1.0000000000000000` |
| pacing | `0.9833333333333333` |
| lapping | `0.9833333333333333` |
| random | `0.9666666666666667` |
| WP binary macro-F1 | `0.9890099825991392` |
| direct_or_non_wandering | `1.0000000000000000` |
| wandering_like | `0.9888888888888889` |

四分类 confusion，行是真值、列是预测，顺序为 `[direct,pacing,lapping,random]`：

```text
[[60, 0, 0, 0],
 [ 0,59, 1, 0],
 [ 0, 1,59, 0],
 [ 2, 0, 0,58]]
```

Binary confusion，顺序为 `[direct_or_non_wandering,wandering_like]`：

```text
[[60,  0],
 [ 2,178]]
```

`errors.jsonl` 有 4 个唯一 sample：2 个 random 被判为 direct，构成全部 2 个 binary 错误；另有 pacing→lapping 和 lapping→pacing 各 1 个。四分类错误数为 4，binary 错误数为 2。独立复算确认 predictions、metrics、recall、confusion 和 errors 完全自洽，binary `sigmoid >= 0.5` 与四分类层级概率 argmax 决策也逐条一致。

## Runtime 与 preflight

正式执行固定为 CPU、intra-op `8`、inter-op `1`、batch `64`、workers `0`、pin-memory `false`，使用 `eval()` 和 `torch.inference_mode()`。Execution evidence 确认以下检查均在 accessor 或模型推理前完成：

- 外部 manifest 原始 SHA、schema 和 candidate artifacts；
- active CLI/release/model/performance/preprocessing/RF source 路径与字节；
- v3 source archive 外层 SHA、精确 7-entry 集合及逐 entry bytes；
- training/release identity 原文件、内嵌身份、active/snapshot training source 和交叉绑定；
- RF config 及 11 个 upstream preprocessing/split/input descriptors；
- 固定 CPU runtime 设置与回读。

Accessor mode 为 `frozen_wp_test`，只调用 `records_for_split("test")`。Evidence 同时记录 `wp_raw=false`、`smartcare_official_or_raw=false`、`sealed_camera=false`、`fit=false`、`optimizer=false`、`threshold_search=false`、`model_or_threshold_modified=false`。正式结果通过同文件系统 staging 完整验证后原子提交，拒绝覆盖。

## Post-hoc paired 描述

分数固化后，只读取 Step5 RF 与 Step6 TCN 已保存的 primary `seed=20260731` 同 cohort predictions。四组 sample ID 与真值均精确一致；下表只是逐样本正确性描述，不是 seed 选择、重训或新的统计主张。

| Baseline / task | 两者都对 | 仅 MPT 对 | 仅 baseline 对 | 两者都错 | MPT / baseline correct |
|---|---:|---:|---:|---:|---:|
| RF four-class | 229 | 7 | 4 | 0 | 236 / 233 |
| RF binary | 231 | 7 | 2 | 0 | 238 / 233 |
| TCN four-class | 219 | 17 | 2 | 2 | 236 / 221 |
| TCN binary | 236 | 2 | 2 | 0 | 238 / 238 |

这项 post-hoc 解释不改变 `target_met` 判定，也不会反馈为 public-holdout 调参。

## 下一模型主线

M0-RS 至此结束。下一任务是把已固化的 primary inference bundle 接入既有 Step7 camera tracklet/Camera QC 路径，在授权的 camera development 数据上验证轨迹输入、QC/fallback、逐窗置信度和 episode 聚合，并记录目标机位域差异。该阶段不得把 public-shape 分数当作 camera 性能；真正独立泛化结论留给后续 C4 sealed camera。本 Goal 没有启动或实现摄像头阶段。

详细命令、宿主传参诊断和复算证据见 `VERIFICATION.md` 与 `terminal_audit.json`。
