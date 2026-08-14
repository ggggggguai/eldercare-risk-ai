# M0-RH verification

日期：2026-08-12

## 环境

按 `AGENTS.md` 使用 WSL `Ubuntu-22.04` 的 `eldercare-ai` 环境。开始时执行：

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
```

结果确认 `Editable project location` 指向当前 `C:\Users\lenovo\Desktop\心理算法` checkout。未修改 `environment.yml`、依赖锁或共享环境包。

## 测试先行证据

新增 M0-RH 测试首次运行在 collection 阶段失败，原因是旧 release 模块没有 `RELEASE_IMPLEMENTATION_SOURCE_FILES`、canonical cohort identity、固定运行时回读和 formal controller API。实现后最终结果：

```bash
conda run -n eldercare-ai python -m pytest tests/test_wandering_release.py -q
# 15 passed, 4 subtests passed

conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_release.py tests/test_wandering_performance.py -q
# 27 passed, 4 subtests passed, 1 warning

conda run -n eldercare-ai python -m pytest tests/test_wandering_*.py -q
# 244 passed, 164 subtests passed, 1 warning
```

完整 wandering 回归第一次使用 6 分钟外层上限时超时；检查发现原 pytest 仍在运行，未并行重启。原进程退出后因输出通道无法恢复最终码，使用 15 分钟上限复验并取得 green 汇总。冻结精确源码后再次运行完整回归，最终为 `244 passed, 164 subtests passed in 368.72s`。唯一 warning 是已有 PyTorch CUDA driver version 提示；固定 M0-RH runtime 为 CPU。

CLI 参数面：

```bash
conda run -n eldercare-ai python \
  scripts/wandering/release_wandering_candidate.py score-frozen-wp --help
```

只显示 `--project-root`、`--manifest`、`--expected-manifest-sha256`、`--output-dir`。使用 `evaluate-records --expected-split test` 会在 argparse 阶段以 invalid choice 拒绝，没有打开 manifest、records 或 accessor。

## v3 source archive

外层：

- 路径：`artifacts/release_implementation_source_v3.zip`
- bytes：355,058
- SHA-256：`30ffec61c620643591b155a66bff3b753ff3a3ff7772eddc5e9d423b98434c0a`

精确 entry 集合：

| Entry | bytes | SHA-256 |
|---|---:|---|
| `configs/modules/wandering_rf_v1.yaml` | 3,398 | `d29db531f42ae5d7cf4ed540a85681ddab78e483695633ccc8b382d98e745e35` |
| `scripts/wandering/release_wandering_candidate.py` | 7,348 | `b60e8c44ad8673b045ddaba563794d923fd661ef73c5d057a904b6af5d611ce1` |
| `src/elderly_monitoring/modules/mental_health/wandering/model.py` | 54,947 | `aecf165568bddc6c2e08bc36ff6ba1e89a4b8d4446aac1a58ff7b9081d617726` |
| `src/elderly_monitoring/modules/mental_health/wandering/performance.py` | 126,453 | `f9a7535400a850417700a25e0e0cbce443707a9f65d33387493062f9f0bee955` |
| `src/elderly_monitoring/modules/mental_health/wandering/preprocessing_bundle.py` | 35,026 | `ce382567974acd60d07a0bcb941a1e9fb058e5cac7935269449496134a14e4c1` |
| `src/elderly_monitoring/modules/mental_health/wandering/release.py` | 92,544 | `761e4ea17fa88e05a8b448260e8b9ab3b547af61527b09a2006665a632b77df8` |
| `tests/test_wandering_release.py` | 34,008 | `dbb7847618795388022ee120f8f08e72fb57b15a12fbf6c13abca4d02437ba66` |

归档已实际解压到 workspace `tmp/` 下的独立目录，逐 entry 与 identity 和 active file 比较后全部一致；随后只清理该临时验证目录。归档、identity 和正式 artifact 均未删除或覆盖。

## v3 candidate bundle

目录：`artifacts/topowander_m0r_candidate_v3/`

| File | bytes | SHA-256 |
|---|---:|---|
| `candidate_manifest.json` | 12,642 | `3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7` |
| `forward_config.yaml` | 4,469 | `debd3adcf0c84d266ca594fd9ba2716a5a854e074e6e2092570b1ec656ed80f7` |
| `frozen_wp_rf_config.yaml` | 3,398 | `d29db531f42ae5d7cf4ed540a85681ddab78e483695633ccc8b382d98e745e35` |
| `model_state.npz` | 838,934 | `94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031` |
| `performance_config.yaml` | 1,840 | `ecc4c5a00dc9c30b1a4a16b943d39d9de85cce27077c8ccba1ba223c529de5f2` |

`model_state.npz`、`forward_config.yaml`、`performance_config.yaml` 三个模型相关文件与 v2 的 SHA 逐项一致。bundle 不含 optimizer、RNG、history、其他 seed checkpoint 或 test records。manifest 的 `canonical_records_sha256` 和 `ordered_sample_ids_sha256` 为 JSON `null`。

## Validation parity v3

`validation_parity_v3/` 通过拒绝覆盖的 staging/atomic writer 生成，精确 artifact set 为：

| File | bytes | SHA-256 |
|---|---:|---|
| `artifact_manifest.json` | 841 | `7bc7e151184fbdc3c8f2eb37954cdaf66e6df3144177ba68709d72a07ccbbf95` |
| `confusion.json` | 168 | `b00d7d73c3b518aacd4ccb61a4c382d7d5b13009c60a344b8857a2e3525bde76` |
| `errors.jsonl` | 671 | `c29e92c740ba3284fa2894f1f0a182d1fbc2440b6e8f2c776b244c0a23ff08af` |
| `execution.json` | 802 | `9015069732040656d240627439878acb5118bff2c9881654d4df44c412303830` |
| `metrics.json` | 1,345 | `80a995c839474a4338e3e527e3475f0b7ae00ee3918057704afa29a453b97bdc` |
| `parity.json` | 1,205 | `cd073d9762e8e2029a2c9f179b7a35da0045c671c5c8439ed28affe66e873d23` |
| `predictions.jsonl` | 143,004 | `dd8f1d6f32434dd010ba84c04c5a5a94ebd4a66b16f6691d57f15495960300ec` |

全部 240 条 prediction 的 split 为 `validation`。Execution 回读 runtime 为 CPU、intra-op 8、inter-op 1、batch64、workers0、pin-memory false；test accessor/inference/scoring 均为 false。Probability 和 WP-only metric max-abs 都是 `0.0`。

## Fail-closed coverage

自动化测试覆盖：

- final output 已存在时，即使 manifest SHA 错误也先拒绝，accessor/inference 调用均为0；
- manifest SHA、bundle path escape、model/forward/performance/RF config 漂移均在 accessor 前拒绝；
- active CLI 路径、active source descriptor、source ZIP extra entry 漂移均在 accessor 前拒绝；
- project identity file 与 manifest embedded identity 不一致时在 accessor 前拒绝；
- synthetic fake accessor 是 formal controller 的唯一 record 来源；返回 239 条时 accessor 调用1、inference调用0；
- fresh process 回读 CPU 8/1 和固定 batch/workers/pin-memory；
- synthetic 240 条 cohort 的 source/split、unique ID、四类各60、binary 60/180；
- canonical records SHA 与 ordered sample-ID SHA 对顺序交换敏感；
- binary `sigmoid >= 0.5`、层级概率、四分类 argmax 和固定类别顺序；
- staging 完整性、artifact manifest、原子提交与拒绝覆盖；
- execution 明确 `fit=false`、`optimizer=false`、`threshold_search=false`、`model_or_threshold_modified=false`。

## 数据与 Git 边界

- 新增 M0-RH controller 的开发/测试只使用 synthetic fixture、fake accessor 和 WP development validation。
- 未执行正式 `score-frozen-wp`；未来 output `reports/mental_health/wandering_performance/m0r_public_holdout_score_v1/` 当前不存在。
- 没有 M0-R public-holdout predictions、metrics、confusion 或 errors。
- 完整回归中的既有 accessor contract 测试可能按既有范围解析固定 WP test，但没有将记录交给 M0-RH candidate inference/scoring，也没有生成 canonical test records/order SHA 或设计反馈。
- 未读取 WP raw、SmartCare official/raw 或 sealed camera；未读取历史 RF/TCN test predictions。
- 未重训、未执行 M1、未改 seed/epoch/model/config/threshold/split/label。
- HEAD 保持 `2981583736a74b1328e0fa52bf39028d6a3f8f7f`；没有 commit、push、发布或部署。
- `git diff --check` 在终端审计中通过。
