# 徘徊方案步骤 4 公开轨迹预处理与可视化记录

日期：2026-08-03

本记录只覆盖步骤 3 冻结的 1,790 条非 sealed 公开轨迹的质量门、弧长重采样、无锚点 shape 归一化、14 通道、拓扑诊断、train-only 统计和固定图审抽样。它不代表分类模型、摄像头域、真实老人泛化、日级心理风险或临床有效性已经完成。

## 结论

- 一对一输出 1,790 条，`sample_id=parent_sample_id`；1,775 条 `ready`，15 条 SmartCare train 为 `unavailable/too_few_valid_points`，其中 normal 10、wandering-like 5。
- 原 split 未改变：原始 `train=1272 / validation=278 / test=240`；质量门后 ready 为 `train=1257 / validation=278 / WP test=240`。
- 20 个 `sealed_external_test` ID 只通过冻结 split/assignments 证明排除；builder 和 visualizer 均未打开 SmartCare official validation 文件，sealed ID、坐标和标签未写入步骤 4 bundle。
- 每条 ready 记录保存 `[80,2]` 来源视图、`[80,2]` shape 视图、`[80]` 全 1 mask 和两份有限 `[80,14]` 特征；SmartCare 的 image 视图与来源重采样逐点相同，WanderingPatterns 的 `image_normalized_points=null`。
- 当前公开来源没有逐点时间，全部 `normalized_dt=0`、`time_available=0`；没有从点序伪造 FPS、速度或持续时间。
- `feature_stats.json` 只使用 1,257 条 ready train 的 100,560 个 `point_mask=1` 位置计算通道 1–10 的 median/IQR，并绑定两份来源、全部 split 输入、canonical split 和预处理配置哈希。
- builder 固定选出 WP 四类和 SmartCare 二类各 8 条 ready train，共 48 条；visualizer 没有二次抽样，并另行展示全部 15 条 unavailable 原始短轨迹。

## 固定输入

| 输入 | SHA-256 |
|---|---|
| `data/processed/wandering/wandering_patterns/samples.jsonl` | `498a0c39af27240bd38ab82367fdc1a13373ba3a22e8fede915419fea54e6ee7` |
| `data/processed/wandering/smartcare/train_pool.jsonl` | `01c5e3b8ecabd594837e5154ab64e52a1d364bcc15096a8193e8b627aa6fb780` |
| `data/splits/mental_health/wandering/v1/split.json` 文件 | `7fa934c8041f538ff033260d722c21f5a4dc0e6cd8d23fa42836aa2e24ed808f` |
| canonical `split_sha256` | `4ac4a3877a056809066562cb09e4d30aa1d38baafcb4600f1f8a8a672776adbf` |
| `data/splits/mental_health/wandering/v1/split.sha256` 文件 | `426175e2c5a7f822706f91445b2bd80c37704fb4b9943891a5fcfcd1b1efb96f` |
| `data/splits/mental_health/wandering/v1/assignments.jsonl` | `993307c13cb30484dba9fa1359c9c4fce714bb45b7d776e9a36ac6dcbee6eac6` |
| `configs/data/wandering_split_v1.yaml` | `579ad16e13b72f2914c5a2d14dfa73ef4a10a9a50f7ecc13bc20a67380a2e8ca` |
| `configs/data/wandering_preprocessing_v1.yaml` | `5b69243337cddeeec8beed4f081330b06c47eadeaee06dc3b3e4ba026f36ff45` |

构建前后重新计算的步骤 2/3 六个文件哈希均与上表一致；没有修改、移动或删除步骤 2/3 文件。

## 机器产物与哈希

正式目录：`data/processed/wandering/preprocessing/v1/`

| 产物 | 字节数 | SHA-256 |
|---|---:|---|
| `samples.jsonl` | 78,920,126 | `323ec1258a03ed2daab1e84e3291edc62644c68b1c292dc93a332c01faa08cda` |
| `feature_stats.json` | 2,844 | `249ad8383043dba1d95452533ddc6004dfecc2c9e99be43f788dcfcbfb434c8d` |
| `preprocessing_report.json` | 4,210 | `d600fbd16efdc8c2b6e89db8c35c965fbb8322f099c464e67600abff5d8804e8` |
| `manifest.json` | 1,278 | `242072bdfe4b969d320a091ecc499aff445c30a937dfb1e6739ad869aa938593` |

生成顺序固定为 `samples.jsonl → feature_stats.json → preprocessing_report.json → manifest.json`。manifest 只保存前三个机器文件的字节数和 SHA-256；前三个文件不反向保存 manifest 哈希，因此不存在循环哈希。manifest 自身哈希只记录在本人工 README。

四个机器文件均为 canonical JSON/JSONL、恰好一个结尾 LF，不包含绝对路径或墙上时钟。固定集成测试在两个不存在的新目录分别完整构建，四个文件逐字节一致；已有目录拒绝覆盖，模拟目录提交失败后没有残留最终目录或临时目录。

## 测试先行和命令

先新增三份测试后运行固定窄命令，首次按预期在收集阶段失败：三个错误均为对应实现模块尚不存在的 `ModuleNotFoundError`。实现后固定窄测试为 `25 passed, 20 subtests passed`；全部徘徊相关回归为 `73 passed, 74 subtests passed`。

最终代码状态的完整测试集为 `451 passed, 1 failed, 141 subtests passed`。唯一失败是既有 `tests/test_fall_runtime_fingerprint.py` 找不到固定输入 `data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi`；这不是步骤 4 回归，本步骤没有修改跌倒代码或验收配置。测试另报告一次既有 CUDA 驱动版本警告，不影响上述 CPU 预处理测试。

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_anchorless_normalization.py \
  tests/test_wandering_preprocessing.py \
  tests/test_wandering_topology.py -q

conda run -n eldercare-ai python scripts/wandering/build_preprocessing.py \
  --config configs/data/wandering_preprocessing_v1.yaml \
  --project-root . \
  --output data/processed/wandering/preprocessing/v1

conda run -n eldercare-ai python scripts/wandering/visualize_preprocessing.py \
  --config configs/data/wandering_preprocessing_v1.yaml \
  --project-root . \
  --bundle data/processed/wandering/preprocessing/v1 \
  --output reports/mental_health/wandering_step4
```

两个输出目录都必须事先不存在。visualizer 会重新校验两份公开来源 JSONL，只读取 `preprocessing_report.json` 的固定 ready train ID，并从公开开发源单独加载 15 条 unavailable 原始轨迹；CLI 没有抽样参数。

## 人工图审状态

当前状态：`human_review_passed`。

七张诊断图和完整固定 ID 已生成在 `diagnostics/` 与 `HUMAN_REVIEW.md`。项目用户于 2026-08-03 检查全部 48 条 ready train 与 15 条 unavailable 短轨迹，确认未发现异常；正式签字记录见 [HUMAN_REVIEW.md](HUMAN_REVIEW.md)。步骤 4 的人工图审门禁已关闭，但不改变下述能力边界。

## 能力边界

本步骤没有实现 bbox 高度平滑/补偿、按秒切 gap、ID switch、摄像头 media sidecar、数据增强、RF/TCN 训练、official 20 条评估、日级聚合或心理风险接入。公开轨迹结果只能称为可复现的 `shape_benchmark` 输入准备，不能称为摄像头验证、真实老人验证、心理状态判断或医学诊断。
