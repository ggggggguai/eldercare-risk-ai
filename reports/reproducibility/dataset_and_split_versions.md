# 跌倒风险数据与 Split 版本记录

记录日期：2026-08-04

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
| v3 受审训练决策 | 绑定当前 v2 action hash；NTU 全片边界、C03 正例、hard negative 与 UR Fall A07 覆盖规则 | `d87bfc260f904508f799dc29239e02ca34085cac4c5a0cde269c605836b3db90` |
| v3 action 训练标签 | 9,314；父类 primary=8,444、auxiliary=702、ignore=168 | `894c6d052159be5f6aad19778b3484aa49e663e29b5030465134fc906b554e5f` |
| v3 event 训练标签 | 9,498；fall 正/负=2,303/1,774、near-fall 正/负=962/4,201、ignore=258 | `7b263e8c1b2d2eae5d8e88cdc23316beb8d41a0e6479a76c620f127a372b8f82` |
| v3 action schema | `fall-risk-action-label-v3` | `1937070abb1cf230ad44185b91569f1f3a76f101a34b28b40168f76ef7f7abfd` |
| v3 event schema | `fall-risk-event-label-v3` | `30542206f41f8704717d6b8baf34c835ed8b37d36a884f4a764d196826a3c6ee` |

## 模型训练标签 v3

v3 是从上述 v2 根标签和哈希绑定受审决策确定性生成的独立训练视图，不改变 `fall-risk-data-v2` 发布候选。迁移报告 SHA-256 为 `d2eab56a42e1d3aa79f130e6eb4cc44884538f175bc82cf9bf63b4faf79567a4`，校验报告 SHA-256 为 `6334434a6b5ae3af71666591a7b31c62c4ad1c46a6394204d0321c4d85832634`。

迁移结果合并 71 个 LE2I/CVAT 重叠 fall；218 个 D04 与唯一父 fall 双向关联，2 个无唯一父事件的 D04 保持 ignore。项目裁决的 36 个展开指令全部匹配：446 条 NTU 全片跌倒统一为 `[0, frame_count)` 精确边界，962 条 C03 生成 near-fall 正例，明确动作/跌倒事件生成 5,975 条负例，7 类 fall 和 8 类 near-fall hard negative 均有 primary 覆盖。v3 统一 split 有 18,812 条标签分配、6,516 个资产和 184 个保守泄漏组，`split_id=splitv3_f89832f6cad5b9c5f630d00b`，校验未发现跨 partition 泄漏。primary fall 正/负按 train/validation/test 分为 `74/7/14` 和 `958/396/369`；primary near-fall 正/负为 `348/300/300` 和 `1109/364/433`。`training_ready.fall_event=true`、`training_ready.near_fall_event=true`；稀有动作类型三分区覆盖不足使 `action_type=false`。

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
| `training_labels_v3` | split valid（provisional；事件门禁 true、动作类型门禁 false） | 18,812 / 6,516 / 184 | `splitv3_f89832f6cad5b9c5f630d00b` | `878a3b46afc570ca32baf37220579301b5cdb420009249318012fc79ac68f715` | `13e7f6dd6321a0fde391e6012282a283297f9114f1fcac2821645c71f70ce71e` |

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
