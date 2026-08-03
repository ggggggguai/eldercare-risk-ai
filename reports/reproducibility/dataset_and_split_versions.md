# 跌倒风险数据与 Split 版本记录

记录日期：2026-08-02

目标名：`fall-risk-data-v2`

发布状态：`release_candidate_blocked`，不是 frozen 版本。

## 数据版本

| 项目 | 版本/状态 | SHA-256 |
|---|---|---|
| 全量 manifest | `fall-risk-data-v2-candidate`，7,534 assets / 6,520 videos；NTU 外部 manifest 已按 `data/external/ntu` 重建，主链纳入的 3,914 个视频均可访问 | `38f15531110ca5401e10575ade5aaf6b7baf497dc04c143a1005d9f013bc61db` |
| Pre_VFallp 内部授权配置 | 5 个 CVAT 压缩包授权组，覆盖 36+18+18+18+18 个视频 | `6258bcb36b64241894382b551ec467139eeebdadc8875895770d176c52add2bd` |
| Pre_VFallp 历史 72 条媒体清单 | `pre_vfallp_remaining_media_20260722`，仅作取得四个 CVAT 包前的历史审计证据 | `b63ef3f589861531aefe0540f07d3ca0a74e9d0845edd845148413010961f95f` |
| 根 action 标签 | 9,314，v2 来源链；含 4,404 条 NTU 动作、2,977 条 Fall Detection 2017 动作和 268 条 UR Fall 人工动作 | `058540ce76076b373bfa86c70d0b157fa10af5cbb521aa7c4c58a3ac7a1b3af6` |
| 根 event 标签 | 6,338，官方窗口优先；含 1,428 条 NTU A043 人工映射事件和 2,977 条 Fall Detection 2017 映射事件 | `fff793d735d89a7675e833cd97969c9c58fe3595abdcb57b734a7d3e43232099` |
| risk 标签模板 | 0 records | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| subject profile 模板 | 0 subjects | `0c1d2de8b9d1c18e9bc35855082907d42fe0e64bc781d70c285f37b4d91805b4` |
| 标签校验配置 | `fall-risk-label-validation-v2` | `eb759ae8cabfbe5c2112cc6e0d32a1534c9e558000a7e9857437211b57d6650a` |
| split 开发配置 | v2 provisional | `eb7fb40c0e21e81eda06fe6f4ef107d1f8a648f4f1559e39e3c16ef27ec53a04` |
| v2 根标签发布报告 | 9,314 action / 6,338 event；排除 71 条重叠 fall，A043 仅按人工 CVAT 决定接入，Fall Detection 2017 批次保留候选 QC 状态；隔离目录名与报告 `batch_id` 不一致的候选批次 | `bfe98754b9f7a09a2186474e8bdc517d9fb2cf8ca6c6d0e194bfb293c2de8538` |
| v2 formal 校验报告 | `errors=0`、`blockers=285`、`formal_ready=false` | `386356a4aa8643264b2206e2e9be88d0508865a96d5ffd2998474089a655689d` |
| v3 action 训练标签 | 9,314；父类 primary=8,444、auxiliary=702、ignore=168 | `9955ec8e0c08ff91f397e8e07cb7e90270ac02accbadbba35f7b2870f60371d1` |
| v3 event 训练标签 | 2,567；fall positive=2,303、ignore=258、negative=6 | `de4700d8b31c723742b3d29fb7538ce91a01b75f5ce801602864db69bbcf29b2` |
| v3 action schema | `fall-risk-action-label-v3` | `1937070abb1cf230ad44185b91569f1f3a76f101a34b28b40168f76ef7f7abfd` |
| v3 event schema | `fall-risk-event-label-v3` | `6335b6b79699ebf1da59b68d1ec533c416cd0d48f4cc052117562c1af07ddbbf` |

## 模型训练标签 v3

v3 是从上述 v2 根标签确定性生成的独立训练视图，不改变 `fall-risk-data-v2` 发布候选。迁移报告 SHA-256 为 `4917400a4db59e37b6fc89fe776e9b783f17a6416fd00dfc17fcfd914f6f3dfd`，校验报告 SHA-256 为 `c1441c9221c655c1da5a6d23b303b4d4eed1839265f3431961e46fb397df9ae4`。

迁移结果合并 71 个 LE2I/CVAT 重叠 fall，220 个 D04 均与唯一父 fall 双向关联。NTU 的 2,976 条精确全片动作全部为 `primary/exact/single_annotated`，具体动作 tier 也为 primary；Fall Detection 2017 贡献 2,977 条 v3 动作和 1,097 条事件窗口；A042/C03 不会自动升级为 v3 near-fall 训练正例。A043 决定文件生成 6 条 `manual_v3/adjudicated` 的 `squat_or_kneel` fall negative，裁决零漏配。v3 统一 split 有 11,881 条标签分配、6,516 个资产和 184 个保守泄漏组，`split_id=splitv3_32db8736e890c9fad95a8292`，校验未发现跨 partition 泄漏。primary fall 正例按 train/validation/test 分为 74/14/7，negative 为 6/0/0；部分动作类别没有覆盖三个 partition，另外六类 fall hard negative 和 near-fall positive 仍缺，因此三项 `training_ready` 均为 false。

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
| `training_labels_v3` | split valid（provisional；训练门禁 false） | 11,881 / 6,516 / 184 | `splitv3_32db8736e890c9fad95a8292` | `83433d0f62e2a2b24fccccc5ca66154da72d6b3fb6495ba025f84645b29ddb4c` | `13b6f42f7232285816ea0f7b76017604168453e230d19145c1bcf7eaf409ef68` |

split 产物已按 v2 根标签重建；`fall_event_v1` 是任务协议标识，不是旧标签版本。

| 任务 | 状态 | 样本 | split_id | split.json SHA-256 |
|---|---|---:|---|---|
| `fall_event_v1` | ready（provisional；train=1,391、validation=526、test=374） | 2,291 | `fall_event_v1:sha256:e52bf881939e63f9f734d5ee591a3bcbc9f83029fdaba1e6f2b8dfd028ba9069` | `0415219d0ec3b8dd573b55f08442d99f657edc3c4b566acacb00d5bf58c6cc35` |
| `near_fall_event_v1` | ready（provisional；train=1、validation=13、test=0） | 14 | `near_fall_event_v1:sha256:332d26602d64d105aeca6dad6713a9cf79025a6504253fb65fee6bead5263065` | `61417a962766b025a227bf812e02b466002d85aed5157e538f6dad6850eefb60` |
| `functional_proxy_v1` | blocked | 0 | `null` | `dad00606716ed19bf3ea664a91a91716cb44ec275e6d6e023850bda76871ff51` |
| `longitudinal_baseline_v1` | blocked | 0 | `null` | `bcaa09b8c24b409ca714e615f8576df8cb9a4429c18c0e07524a629672233c3d` |

共同 blocker：formal 校验仍有 285 个 blocker，其中 27 条为技术排除关联、258 条为 `U01/uncertain`。v2 的 14 条 C03 映射事件可形成 provisional near-fall split，但不满足 v3 near-fall positive 的恢复结局和双审要求；功能/纵向任务仍没有真实参考终点。所有产物绑定当前 manifest 与 v2 标签 hash；不会生成假的 frozen split。

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
