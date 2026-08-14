# 近跌倒恢复确认 TCN 开发训练报告

日期：2026-08-09

## 结论与风险

本轮已在当前 `splitv3_e71a045eb58489f43dc5fd11` 上完成近跌倒恢复确认 TCN 的 train/validation 数据重建、确定性复建和 seed `42/43/44` 三次训练。三组事件级 validation F1 为 `0.9967/0.9967/0.9933`，均未读取 test 姿态、标签内容或指标。

这些数值只能作为 `development_provisional` 的候选窗口分类结果，不能解释为连续视频、老人域或正式泛化效果。validation 的 69 个负事件全部是 `fast_but_controlled_sit`，缺少其余七类困难负例窗口；三组 seed 的共同误报也集中在 NTU `A008` 坐下片段。模型在观察到 `recovery_frame` 后确认候选，不是 onset-time 提前预警器。规则 `near-fall-rule-v0.1` 继续作为实时主路径、安全覆盖和低质量 fallback，当前 checkpoint 不写入默认配置。

## 修复与数据构建

旧数据构建器把 CVAT 标注 `track_id` 与 ByteTrack 姿态 `track_id` 当作同一命名空间，导致大量非 NTU 标签因字符串不相等而被拒绝。本轮改为在标注区间内选择覆盖帧数最多、质量最高的主导姿态轨迹，并在样本中同时保留 `annotation_track_id`、`pose_track_id` 和 `track_match_method`。1,105 个窗口中有 229 个窗口的两类 track ID 不同，这一差异现在被显式审计。

训练器仍对真实 `event_id/source_group_id/sample_group_id/split_group_id` 做跨分区泄漏拒绝；只忽略 `subject_id=unknown|none|null` 这类跨数据集占位值，不会放松真实保护组隔离。

## 数据与门禁

- source split：`splitv3_e71a045eb58489f43dc5fd11`，当前机器校验无跨 partition 泄漏。
- 开发 dataset：`1,105` 个 `[24,10,8]` 窗口、`822` 个事件；train `661` 窗口/`455` 事件，validation `444` 窗口/`367` 事件。
- train：正/负窗口 `347/314`，正/负事件 `347/108`；39 个非占位 subject。
- validation：正/负窗口 `298/146`，正/负事件 `298/69`；6 个非占位 subject。
- train 来源：CaucaFall `171`、Fall TikTok `8`、LE2I `9`、NTU RGB+D `445`、UR Fall `28`。
- validation 来源：LE2I `13`、NTU RGB+D `431`。
- test 锁定：只记录 `733` 条 test 标签数量；`test_pose_read=false`、`test_evaluated=false`、训练结果中的 `test=null`。
- 拒绝原因：因果上下文不足 `1,295`、观测帧不足 `1`、姿态质量不足 `6`、无可用姿态轨迹 `1`。
- 正例语义：在 `recovery_frame` 结束的因果窗口内确认 `recovered_without_fall`；不是近跌倒 onset 定位或提前预警。

实际困难负例窗口覆盖如下：

| partition | controlled bend | squat | exercise | controlled sit | step adjustment | turn | support contact | progressed to fall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 29 | 82 | 40 | 136 | 7 | 18 | 2 | 0 |
| validation | 0 | 0 | 0 | 146 | 0 | 0 | 0 | 0 |

因此标签层的八类 hard-negative 门禁虽已通过，窗口层仍未通过正式训练门禁。

## 可复现性

第二次构建输出到 `/tmp/near-fall-rebuild-e71a045-trackmap-v2`，以下三个文件与仓库内第一次构建结果逐字节相同：

| 产物 | SHA-256 |
|---|---|
| `dataset.npz` | `1804adfcfc29fa44469eb47dd878ad118727321949fa3a13089406b4f9f82083` |
| `metadata.json` | `876b81a031224d7f2fc90fe291e1adcf9bca61cee4fb48282f95b8ead86a5517` |
| `samples.jsonl` | `7c7685401d32e5cd25f6ad461ffc0939c64a6c1b40fd13f22eca9b003046d269` |

数据输入绑定：manifest `539274...503a`、v3 event labels `7b263e...8f82`、assignments `878a3b...f715`、split `02aaae...3b80`、v3 validation report `bd4f56...5b9f`。完整哈希保存在 `metadata.json`，每个姿态输入文件也单独绑定 SHA-256。

## 三 seed 结果

指标由各自 best checkpoint 在 validation 分区以阈值 `0.5` 独立复算。事件概率取同一事件全部窗口的最大概率；事件计算不使用窗口权重。

| seed | best epoch | epochs | Precision | Recall | F1 | Balanced accuracy | PR-AUC | ROC-AUC | TN/FP/FN/TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 15 | 24 | 0.9933 | 1.0000 | 0.9967 | 0.9855 | 0.9992 | 0.9970 | 67/2/0/298 |
| 43 | 20 | 29 | 0.9933 | 1.0000 | 0.9967 | 0.9855 | 0.9999 | 0.9998 | 67/2/0/298 |
| 44 | 20 | 29 | 0.9900 | 0.9966 | 0.9933 | 0.9766 | 0.9982 | 0.9933 | 66/3/1/297 |

三 seed 事件级 F1 均值/总体标准差为 `0.9955/0.0016`，PR-AUC 均值为 `0.9991`。模型参数量为 `4,802`。在当前 Darwin arm64 CPU 上，以 seed 42 checkpoint、batch size 1、50 次预热和 500 次纯 forward 测量，中位数 `1.34 ms`、P95 `1.42 ms`；该数值不包含姿态提取、数据准备、状态机或视频 I/O，不能作为端到端实时延迟。

## 失败案例

- seed 42/43 共同把 `ntu_rgbd_s003_p016_r002_a008_c001` 和 `..._c003` 的快速可控坐下判成近跌倒。
- seed 44 同样误判 `..._a008_c001`，并误判 `..._a008_c002` 与 `ntu_rgbd_s017_p003_r002_a008_c002`；另漏检 `ntu_rgbd_s013_p016_r001_a042_c001` 的一个恢复事件。
- validation 负例来源和动作类型过窄，当前混淆矩阵不能外推到弯腰、转身、支撑接触、遮挡、多人、连续背景或最终跌倒场景。
- train 达到事件级 F1 `1.0`，说明模型容量足以完全拟合当前训练事件；在补齐跨来源困难负例和连续背景前，应把高 validation 分数视为窄分布可分性证据，而非上线证据。

每个运行目录中的 `validation_predictions.jsonl` 保存窗口级原始概率和样本 provenance，便于逐例复核。

## 运行产物

数据目录：`data/processed/fall_risk/near_fall_event_v1/splitv3-e71a045-trackmap-v2/`。

| seed | 目录 | best checkpoint SHA-256 |
|---:|---|---|
| 42 | `development-splitv3-e71a045-trackmap-v2-seed42/` | `59c5f26fbd2467f47417c7029fcd0bc76b5567d0060704d6afc571801f1f0e5b` |
| 43 | `development-splitv3-e71a045-trackmap-v2-seed43/` | `130bbcc7011d6f0d39b69f413b23105b7eda9760d34471827f3ab29b02c5eacf` |
| 44 | `development-splitv3-e71a045-trackmap-v2-seed44/` | `00e65e4748dd5d13915447f30ae072e0bb3ceffcdde056cdfc2e4eddc83d52b7` |

每个目录包含 `best_model.pt`、`last_model.pt`、`config.json`、`metrics.json` 和 `validation_predictions.jsonl`。所有 checkpoint 和指标均标记为 `development_provisional`。

## 复现命令

```bash
conda run -n eldercare-ai python scripts/prepare/prepare_near_fall_event_dataset.py \
  --output-dir data/processed/fall_risk/near_fall_event_v1/splitv3-e71a045-trackmap-v2

conda run -n eldercare-ai python scripts/train/train_near_fall_tcn.py \
  --data data/processed/fall_risk/near_fall_event_v1/splitv3-e71a045-trackmap-v2/dataset.npz \
  --metadata data/processed/fall_risk/near_fall_event_v1/splitv3-e71a045-trackmap-v2/metadata.json \
  --config configs/training/near_fall_event_v1.yaml \
  --profile pilot \
  --seed 42 \
  --device cpu \
  --output-dir reports/fall_risk/near_fall_event_v1/development-splitv3-e71a045-trackmap-v2-seed42
```

seed 43/44 只替换 `--seed` 和输出目录。训练器在输出目录已存在时默认拒绝覆盖，避免污染历史运行。

## 尚未解除的正式门禁

代码内可修复的问题和本轮开发训练已完成，但正式模型仍缺少外部证据：v2 formal 还有 285 个 blocker，连续背景为 `0/100 camera-hour`，明确老人 ADL 为 `0/20` 人和 `0/30 camera-hour`，split/评估协议未冻结，且未指定 test 保管人与一次性发布流程。validation 还需补齐七类困难负例窗口，train 需补 `progressed_to_fall`。

这些门禁需要新增或复核真实数据及项目治理决策，不能由训练代码自动生成。解除前不得读取 test、启动告警驱动 shadow、替换规则主路径或把本报告数值写成正式比赛/临床效果。
