# 徘徊方案步骤 3 固定 split 复现记录

日期：2026-08-03

本记录只覆盖已验收公开轨迹的固定划分、逐样本分配、重复/近邻审计和哈希绑定，不代表轨迹预处理、分类模型、摄像头域、真实老人泛化或临床有效性已经完成。步骤 2 的转换产物、人工复核文件和原始数据均未修改、移动或删除。

## 结论

- `wandering-split-v1` 已按 `group_policy=source_specific_v1`、`seed=20260731` 冻结。
- 1,810 个 sample ID 全部且只分配一次：`train=1272`、`validation=278`、`test=240`、`sealed_external_test=20`，四个分区两两无交集。
- WanderingPatterns 四类各为 `280/60/60`，总计 `1120/240/240`；其 test 只命名为 `public_shape_benchmark`。
- SmartCare 开发池按固定自然日替代组划分：train 152（normal 73、wandering-like 79），validation 38（normal 17、wandering-like 21）。validation 日期为 `2020-09-20` 和 `2020-10-07`；这里仅证明自然日替代组不重叠，自然日不是人员或真实 session 标识。
- SmartCare official 20 条全部且只进入 `sealed_external_test`。builder 先校验整文件 SHA-256，随后只流式提取 `sample_id`，不解析其坐标或标签；逐样本 assignments 也不写入其标签。开发 loader 默认只返回 train，validation/test 需显式打开，始终拒绝 sealed 分区。
- 正向完全重复、反向完全重复在 WanderingPatterns 和 SmartCare 开发池中均为 0 组。正式 Procrustes 审计精确复现技术方案预审的 `<0.05` 9,061 对，其中同类 9,055 对、跨类 6 对；跨分区 4,054 对，所以公开形状 test 不能解释为人员级或真实场景防泄漏验证。

## 固定输入

builder 在读取样本和分配之前逐项校验以下六个文件；任何一项缺失或 SHA-256 漂移都会在创建输出目录前失败。

| 输入 | SHA-256 |
|---|---|
| `data/processed/wandering/wandering_patterns/manifest.json` | `04bb88849eddd15ef12de7883612f3b03a20004dfbcd9684e1ef059f06875b9e` |
| `data/processed/wandering/wandering_patterns/samples.jsonl` | `498a0c39af27240bd38ab82367fdc1a13373ba3a22e8fede915419fea54e6ee7` |
| `data/processed/wandering/smartcare/manifest.json` | `c74bb729cf2c353311fef2013ae6abf72a59f70dab5a1969ffb48a37262736bc` |
| `data/processed/wandering/smartcare/train_pool.jsonl` | `01c5e3b8ecabd594837e5154ab64e52a1d364bcc15096a8193e8b627aa6fb780` |
| `data/processed/wandering/smartcare/official_validation.jsonl` | `78ac0568089ca1aa5aedd4e38f77d37bc5d0d40c3c1ba14d51ceaa4a0ead7b40` |
| `reports/mental_health/wandering_step2/HUMAN_REVIEW.md` | `9577803dde1a4a9cb6b97ad09bb4fbc573817ba9d1e8954d44ad22d312cb0cd0` |

配置 `configs/data/wandering_split_v1.yaml` 的 SHA-256 为 `579ad16e13b72f2914c5a2d14dfa73ef4a10a9a50f7ecc13bc20a67380a2e8ca`，已写入 `split.json.source_hashes.split_config`。

## 审计结果

| 阈值 | 总对数 | 同类 | 跨类 | 跨分区 | 分区内 |
|---|---:|---:|---:|---:|---:|
| `<0.02` | 1,598 | 1,598 | 0 | 757 | 841 |
| `<0.05` | 9,061 | 9,055 | 6 | 4,054 | 5,007 |

距离流程固定为：弧长重采样到 32 点，减去坐标均值，以单一 Frobenius 范数作各向同性缩放，在前向/时间反向以及旋转/镜像中取最小归一化 full-Procrustes 距离。实现没有再除以 `sqrt(32)`，所以该值不是逐点 RMS。近邻只用于暴露公开模板相似性，不改写固定分配。

## 机器产物与哈希

正式产物目录为 `data/splits/mental_health/wandering/v1/`。

| 产物 | 字节数 | 文件 SHA-256 |
|---|---:|---|
| `split.json` | 82,260 | `7fa934c8041f538ff033260d722c21f5a4dc0e6cd8d23fa42836aa2e24ed808f` |
| `split.sha256` | 65 | `426175e2c5a7f822706f91445b2bd80c37704fb4b9943891a5fcfcd1b1efb96f` |
| `assignments.jsonl` | 697,190 | `993307c13cb30484dba9fa1359c9c4fce714bb45b7d776e9a36ac6dcbee6eac6` |
| `near_neighbor_audit.json` | 1,598 | `72d6f9f10f092fcadcd81f71e424d51f3b26938d6bfccbdcecced5852f851d10` |
| `split_report.json` | 3,734 | `ef149a76c8f1f344aeb7d9e9b26d6daa6f155deaa249a28f3880f7c3de125a1b` |

- canonical split payload SHA-256：`4ac4a3877a056809066562cb09e4d30aa1d38baafcb4600f1f8a8a672776adbf`；`split.sha256` 保存该值。
- canonical report payload SHA-256：`daaa8c073ae1a4fec3836f4956634ef542cfeb7732f7420668ac0d1c86f6e935`。
- `split_report.json` 内绑定六个输入哈希、配置哈希、split 语义哈希和其余四个产物的文件哈希；本记录补充报告文件自身哈希。

## 测试先行与复现命令

实现前先新增 `tests/test_wandering_splits.py`，首次窄测试按预期在收集阶段失败：`ModuleNotFoundError: No module named 'elderly_monitoring.modules.mental_health.wandering.splits'`。实现后固定窄测试为 `26 passed, 38 subtests passed`，全部徘徊相关回归为 `48 passed, 54 subtests passed`。验收测试会在两个不存在的临时目录中分别执行完整构建，五个产物逐字节比较一致，并验证已有目录拒绝覆盖；正式仓库产物又与一个独立 `/tmp` 全新构建逐文件执行 `cmp`，结果 `BYTE_IDENTICAL=5`。机器产物不写绝对路径或墙上时钟。

完整测试集结果为 `426 passed, 1 failed, 121 subtests passed`。唯一失败是既有 `tests/test_fall_runtime_fingerprint.py` 找不到固定输入 `data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi`；这不是徘徊步骤 3 回归，本步骤未修改跌倒代码或验收配置。测试还报告一次既有 CUDA 驱动版本警告，但不影响上述 CPU 测试结论。

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_manifests.py \
  tests/test_wandering_splits.py -q

conda run -n eldercare-ai python scripts/wandering/build_split.py \
  --config configs/data/wandering_split_v1.yaml \
  --project-root . \
  --output data/splits/mental_health/wandering/v1
```

输出目录必须事先不存在。训练和预处理代码后续必须显式读取本步骤冻结的 `split.json`，不得现场随机切分；步骤 4 才开始实现 QC、重采样、双视图、14 通道和对应可视化。
