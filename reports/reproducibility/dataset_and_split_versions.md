# 跌倒风险数据与 Split 版本记录

记录日期：2026-07-25

目标名：`fall-risk-data-v2`

发布状态：`release_candidate_blocked`，不是 frozen 版本。

## 数据版本

| 项目 | 版本/状态 | SHA-256 |
|---|---|---|
| 全量 manifest | `fall-risk-data-v2-candidate`，6,530 assets / 5,516 videos；含 2,976 条人工精确 NTU 动作片段，旧解压媒体路径当前不可用 | `deac5fa3e271238087cdad5c883afa52aedc9a72cacf7eee8ab9bd0724b8111a` |
| Pre_VFallp 内部授权配置 | 5 个 CVAT 压缩包授权组，覆盖 36+18+18+18+18 个视频 | `6258bcb36b64241894382b551ec467139eeebdadc8875895770d176c52add2bd` |
| Pre_VFallp 历史 72 条媒体清单 | `pre_vfallp_remaining_media_20260722`，仅作取得四个 CVAT 包前的历史审计证据 | `b63ef3f589861531aefe0540f07d3ca0a74e9d0845edd845148413010961f95f` |
| 根 action 标签 | 4,759，v2 来源链；含 2,976 条 NTU 人工精确动作和 268 条 UR Fall 人工动作 | `08a13d283e39f3aa1d7f91cf4077253a7eac923014201fb33e2426e9ed8ce23c` |
| 根 event 标签 | 1,783，官方窗口优先；NTU 不派生事件 | `eb431bedf9104c9cfbd7bf166b1555f1c824b1d23a2a1d21784bb54ab88fd8ea` |
| risk 标签模板 | 0 records | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| subject profile 模板 | 0 subjects | `0c1d2de8b9d1c18e9bc35855082907d42fe0e64bc781d70c285f37b4d91805b4` |
| 标签校验配置 | `fall-risk-label-validation-v2` | `eb759ae8cabfbe5c2112cc6e0d32a1534c9e558000a7e9857437211b57d6650a` |
| split 开发配置 | v2 provisional | `eb7fb40c0e21e81eda06fe6f4ef107d1f8a648f4f1559e39e3c16ef27ec53a04` |
| v2 根标签发布报告 | 4,759 action / 1,783 event；排除 71 条重叠 fall，A043 不导入 | `bbaa72fba32a55576a68fb465e5a3b417739f821f5be620caa3b71a18637a868` |
| v2 formal 校验报告 | `errors=5,952`（NTU 媒体路径缺失）、`blockers=179`、`formal_ready=false` | `e27818dda18a4fc78565fa2b8e145cd4de267e47e480c352e2c8733bce081518` |
| v3 action 训练标签 | 4,759；父类 primary=4,297、auxiliary=349、ignore=113 | `ba787841b5146c6c462ba70f55499d4d2b0a47e5397055a0a7949ca4b6235310` |
| v3 event 训练标签 | 464；fall positive=312、ignore=152、negative=0 | `4cbbda2f57bbeadc496c49f1de0732f685a5a71b1d78fc762348fecabc392cd9` |
| v3 action schema | `fall-risk-action-label-v3` | `1937070abb1cf230ad44185b91569f1f3a76f101a34b28b40168f76ef7f7abfd` |
| v3 event schema | `fall-risk-event-label-v3` | `6335b6b79699ebf1da59b68d1ec533c416cd0d48f4cc052117562c1af07ddbbf` |

## 模型训练标签 v3

v3 是从上述 v2 根标签确定性生成的独立训练视图，不改变 `fall-risk-data-v2` 发布候选。迁移报告 SHA-256 为 `cba37c43717ae12044fee6518be29d0a5d2087ca2f15a914acde46d947d07ba9`，校验报告 SHA-256 为 `7c47ed1a2b23dd97ce2ace669cf2ce8a3e65095b34d0887aea8ccd8d87b9abf2`。

迁移结果合并 71 个 LE2I/CVAT 重叠 fall，159 个 D04 均与唯一父 fall 双向关联。NTU 的 2,976 条动作全部为 `primary/exact/single_annotated`，具体动作 tier 也为 primary；A042/C03 不会自动升级为 v3 near-fall 训练正例。v3 统一 split 有 5,223 条标签分配、3,501 个资产和 154 个保守泄漏组，`split_id=splitv3_23e61c44a3a44dd8ffba4b40`，校验未发现跨 partition 泄漏。primary fall 正例按 train/validation/test 分为 74/14/7；部分动作类别没有覆盖三个 partition，且人工 event negative 均为 0，因此三项 `training_ready` 均为 false。

复现命令：

```bash
conda run -n eldercare-ai python scripts/annotation/migrate_fall_labels_v2_to_v3.py --overwrite
conda run -n eldercare-ai python scripts/annotation/build_fall_training_split_v3.py --overwrite
conda run -n eldercare-ai python scripts/annotation/validate_fall_labels_v3.py --overwrite
```

## Split 版本

v3 训练标签共用一份统一 split，不能复用下方绑定 v2 根标签的任务 split：

| split | 状态 | 标签/资产/组 | split_id | assignments SHA-256 | split.json SHA-256 |
|---|---|---:|---|---|---|
| `training_labels_v3` | split valid（provisional；训练门禁 false） | 5,223 / 3,501 / 154 | `splitv3_23e61c44a3a44dd8ffba4b40` | `6b7ed71c5442425f7d4112d52b64dfa0a8cd3ce9424f1d72144748e0dffd36af` | `77488b36521c5dd34b7f910d0e41d1e71759f62c833d83843d68509e8d4c3c9a` |

split 产物已按 v2 根标签重建；`fall_event_v1` 是任务协议标识，不是旧标签版本。

| 任务 | 状态 | 样本 | split_id | split.json SHA-256 |
|---|---|---:|---|---|
| `fall_event_v1` | ready（provisional） | 248 | `fall_event_v1:sha256:a929a3edae413ea07d97836403478eceb19330458dd89702de5cc8f912f5ca0b` | `b3b5fac31c2b35e269e4e87a4635fcf466cc1a21f4a9da4e6649ecdea8aec18b` |
| `near_fall_event_v1` | ready（provisional；13 条均在 validation） | 13 | `near_fall_event_v1:sha256:e4383bd6581cd534be874bed5e3d13735926a1172532dee1b5c3b75c8073f218` | `f8a081556d1889564d1bbfd4827ffacf98f8600907ee11021fee9e0466ef23d8` |
| `functional_proxy_v1` | blocked | 0 | `null` | `b8830d09e56ae58e5dffc2084b67d8d942cf4a90a0243d3f31aff62d9e3e7b9f` |
| `longitudinal_baseline_v1` | blocked | 0 | `null` | `826d1bcadfd3474a7c3aea8a0a4f85da78431653512f66b329336010a1ec6bbb` |

共同 blocker：formal 校验仍有 113 个 blocker；新增 1 条 CaucaFall `U01` 记录因来源没有提供人工原因而按 uncertain 处理。v2 的 13 条 C03 映射事件可形成 provisional near-fall split，但不满足 v3 near-fall positive 的恢复结局和双审要求；功能/纵向任务仍没有真实参考终点。所有产物绑定当前 manifest 与 v2 标签 hash；不会生成假的 frozen split。

## 评估协议

| 协议 | 状态 | 配置 SHA-256 |
|---|---|---|
| `fall-event-eval-v1` | `development_provisional` | `b898915107ae6cc5e54509988ddfb102aeff4dbbd8c8eded700277bc3e15eb60` |
| `near-fall-event-eval-v1` | `development_provisional` | `06818cb4eeb32e4161fe70e004ce36a5ed2d5ce01d701ea10d0c93d39931434e` |

正式协议必须由负责人预注册并另存为 frozen 版本；不得根据测试标签或结果回改当前阈值。frozen split 入口会同时校验 manifest、标签、人员画像、validation config 和 formal validation report 的文件哈希。

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
  --validation-config configs/data/fall_risk_label_validation_v2.yaml \
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
