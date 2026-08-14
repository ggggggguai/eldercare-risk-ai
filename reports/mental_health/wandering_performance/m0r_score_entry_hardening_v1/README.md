# TopoWander-MPT M0-RH formal score-entry hardening

日期：2026-08-12

状态：`wandering_m0r_score_entry_hardened_waiting_public_holdout_authorization`

## 结论

M0-RH 已完成。新的 v3 scoring candidate 仍严格绑定 M0-S primary `seed=20260731 / best epoch=5`，没有重训、ensemble、M1、权重/forward config/performance config/阈值/split/标签/选点变更。v1/v2 candidate、诊断和 M0/M0-S 证据均保留且未覆盖。

本轮只完成正式计分入口加固和 development validation parity。没有执行 `score-frozen-wp`，没有让新增 controller 调用真实 frozen-WP accessor，没有形成 WP public-holdout predictions、metrics、confusion 或 errors。v3 manifest 中 canonical test records SHA 与 ordered sample-ID SHA 均保持 `null`；它们只能在未来取得明确授权并由正式 accessor 返回固定 cohort 后写入当次 execution evidence。

## v3 固定身份

| 层 | 路径 | bytes | SHA-256 |
|---|---|---:|---|
| Training candidate identity | `../m0r_release_prep_v1/identity/training_candidate_identity.json` | 5,697 | `c4e8576fce46a69bca3cebf81306197c9abfa439ff720bb3ced95ea74f580228` |
| M0-S model state | v3 bundle `model_state.npz` | 838,934 | `94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031` |
| Release implementation identity v3 | `identity/release_implementation_identity_v3.json` | 1,864 | `10a2bd6997b3cc8979eecfaf1b9adf58560878376ab7c512e65b30384939615a` |
| Release source archive v3 | `artifacts/release_implementation_source_v3.zip` | 355,058 | `30ffec61c620643591b155a66bff3b753ff3a3ff7772eddc5e9d423b98434c0a` |
| Candidate scoring manifest v3 | `artifacts/topowander_m0r_candidate_v3/candidate_manifest.json` | 12,642 | `3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7` |

v3 bundle 中的 model state、forward config、performance config 与 v2 SHA 完全相同。新增的 `frozen_wp_rf_config.yaml` SHA 为 `d29db531f42ae5d7cf4ed540a85681ddab78e483695633ccc8b382d98e745e35`，用于绑定受信 preprocessing/split/input descriptors；它不是模型参数或新训练配置。

所有身份与二进制制品均为 `current_machine_local_only`。Git HEAD `2981583736a74b1328e0fa52bf39028d6a3f8f7f` 只作为现场上下文记录；本轮没有 commit 或 push。

## 唯一正式计分入口

生产 test 入口只有 `score-frozen-wp`。它只接受：

- `--project-root`
- `--manifest`
- `--expected-manifest-sha256`
- `--output-dir`

它不接受 records、split、batch、threshold、seed、threads、类别顺序或标签映射。既有 `evaluate-records` 已限制为 validation-only；内部 evaluator 仍可用于 synthetic test fixture，但不能成为第二个 production test 入口。

正式 controller 的固定 preflight 顺序为：

1. 在任何其他动作前拒绝已经存在的 final output；
2. 验证外部传入的 manifest SHA 和 manifest schema；
3. 验证 bundle 内 model/forward/performance/RF config descriptors，并 safe-load 同一完整 model state；
4. 验证当前实际执行的 `release.py`、CLI、`model.py`、`performance.py`、`preprocessing_bundle.py` 和 RF config 路径/字节；
5. 验证 v3 source ZIP 外层 SHA、精确 7-entry 集合、逐 entry bytes；
6. 验证项目内 training/release identity 原始文件与 manifest 内嵌身份一致，并核对 M0-S active/snapshot 源码、model/config/reference 和 training/release 交叉绑定；
7. 验证 RF config 与全部上游 preprocessing/split/input descriptors，设置并回读 CPU 8/1、batch64、workers0、pin-memory=false；
8. 只有全部 preflight 通过后才内部调用 `load_preprocessing_bundle(..., mode=BUNDLE_MODE_FROZEN_WP_TEST)`；
9. 只取 `bundle.records_for_split("test")`，验证 240 条、source/split、唯一 ID、四类各60、binary 60/180，并生成 canonical records/order SHA；
10. 在 `torch.inference_mode()`/`eval()` 下推理，完整写入同文件系统 staging、逐字节验证 artifact set 后原子提交到此前不存在的 final 目录。

任何 preflight 失败均发生在 accessor 与模型推理之前；accessor 返回非法 cohort 时也会在模型推理前拒绝。

## v3 development validation parity

证据目录：`validation_parity_v3/`。

| 检查 | 结果 |
|---|---:|
| WP validation sample count | 240 |
| sample ID / true label | exact |
| predicted label / confusion / raw counts | exact |
| maximum absolute probability difference | `0.0` |
| maximum absolute WP-only metric difference | `0.0` |
| tolerance | `1e-7` |
| WP 四分类 macro-F1 | `0.9874973951700553` |
| WP 二分类 macro-F1 | `0.9944750109348742` |
| M0-RH candidate/controller WP test accessor / inference / scoring | false / false / false |

四分类 confusion（`direct,pacing,lapping,random`）：

```text
[[60,0,0,0],
 [0,60,0,0],
 [0,1,59,0],
 [1,1,0,58]]
```

二分类 confusion（`direct_or_non_wandering,wandering_like`）：

```text
[[60,0],
 [1,179]]
```

Parity SHA 为 `cd073d9762e8e2029a2c9f179b7a35da0045c671c5c8439ed28affe66e873d23`；validation predictions SHA 仍为 `dd8f1d6f32434dd010ba84c04c5a5a94ebd4a66b16f6691d57f15495960300ec`。没有使用 SmartCare source-equal 指标替代 WP binary 指标，也没有概率舍入、后处理、复制输出、阈值或权重变化。

## 验证和边界

- M0-RH release 窄测：15 passed + 4 subtests。
- Release + performance：27 passed + 4 subtests；1 个既有 CUDA 驱动警告。
- 全部 `tests/test_wandering_*.py`：244 passed + 164 subtests；1 个既有 CUDA 驱动警告。
- editable 安装指向当前桌面仓库；formal CLI help 只显示四个参数；`evaluate-records --expected-split test` 在 argparse 阶段拒绝。
- v3 source ZIP 已实际解压，7 个 entry 的路径、size、SHA 与 active files 一致；M0-S 6 个 training source 的 active/snapshot 双副本一致。
- v2 manifest、v2 source ZIP 和 v1 batch-layout 诊断 SHA 复核不变。

完整命令与文件级证据见 `VERIFICATION.md` 和 `terminal_audit.json`。

本结果只证明正式计分入口的 fail-closed 工程性质及固定公开轨迹 development validation 的可复现性，不是 WP public-holdout 分数、目标摄像头、真实老人、产品或临床有效性证据。完整 wandering 回归中的既有 accessor contract 测试可按既有范围解析 WP test 做 schema/count/access 检查；这些记录没有进入 M0-RH candidate inference、scoring、artifact 或设计反馈。

## 未来唯一命令（仅展示，未执行）

只有在下一次取得明确负责人授权后，才允许从仓库根运行：

```bash
conda run -n eldercare-ai python scripts/wandering/release_wandering_candidate.py score-frozen-wp \
  --project-root . \
  --manifest reports/mental_health/wandering_performance/m0r_score_entry_hardening_v1/artifacts/topowander_m0r_candidate_v3/candidate_manifest.json \
  --expected-manifest-sha256 3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7 \
  --output-dir reports/mental_health/wandering_performance/m0r_public_holdout_score_v1
```

当前 `m0r_public_holdout_score_v1/` 不存在。本轮在该人工授权边界前停止。
