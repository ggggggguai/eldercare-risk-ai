# 跌倒风险数据与 Split 版本记录

记录日期：2026-08-17

目标名：`fall-risk-data-v2`

发布状态：`release_candidate_blocked`，不是 frozen 版本。

## 数据版本

| 项目 | 版本/状态 | SHA-256 |
|---|---|---|
| 全量 manifest | `fall-risk-data-v2-candidate`，7,684 assets / 6,670 videos；包含 SCF P01/P02/P04 150 个 train 视频，P03/P05 未纳入 | `e09ba07a6ae74c012f5bc2eb0bbb55bffc24763989e55fc16ac4df8e46e6f629` |
| Pre_VFallp 内部授权配置 | 5 个 CVAT 压缩包授权组，覆盖 36+18+18+18+18 个视频 | `6258bcb36b64241894382b551ec467139eeebdadc8875895770d176c52add2bd` |
| Pre_VFallp 历史 72 条媒体清单 | `pre_vfallp_remaining_media_20260722`，仅作取得四个 CVAT 包前的历史审计证据 | `b63ef3f589861531aefe0540f07d3ca0a74e9d0845edd845148413010961f95f` |
| 根 action 标签 | 9,612；SCF P01/P02/P04 新增 298 条 | `52ac8f6a371db36d7b3120135a459f48d377bd095518850e62690b1c9ca29b11` |
| 根 event 标签 | 6,338，官方窗口优先；含 1,428 条 NTU A043 人工映射事件和 2,977 条 Fall Detection 2017 映射事件 | `fff793d735d89a7675e833cd97969c9c58fe3595abdcb57b734a7d3e43232099` |
| risk 标签模板 | 0 records | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| subject profile 模板 | 0 subjects | `0c1d2de8b9d1c18e9bc35855082907d42fe0e64bc781d70c285f37b4d91805b4` |
| 标签校验配置 | `fall-risk-label-validation-v2` | `eb759ae8cabfbe5c2112cc6e0d32a1534c9e558000a7e9857437211b57d6650a` |
| split 开发配置 | v2 provisional | `eb7fb40c0e21e81eda06fe6f4ef107d1f8a648f4f1559e39e3c16ef27ec53a04` |
| v2 根标签发布报告 | 9,612 action / 6,338 event；SCF 发布 298 条动作 | `984bd0b883dd15bb8b4add91b59e2d8020574efcdf2176ace7e0bd4cebb87644` |
| v2 formal 校验报告 | `errors=0`、`blockers=285`、`formal_ready=false` | `7c6bb7dc68cb73c35cde5e945d1a95b2e55729a4d52ce2ceb93e4942230f2764` |
| v3 受审训练决策 | 绑定当前 v2 action hash；NTU 全片边界、C03 正例、hard negative 与 UR Fall A07 覆盖规则 | `8157579420c4bb078c5d4efa7a7c0296afeeb6421562ff6e51cba940f47eaac3` |
| v3 action 训练标签 | 9,612；primary=8,444、auxiliary=1,000、ignore=168；SCF 298 条全部 auxiliary | `c439acdcb0bb712dce2ce64f72eb19203fc71fa6639d27f44e6aabdbb2509f59` |
| v3 event 训练标签 | 9,651；fall 正/负=2,303/1,827、near-fall 正/负=981/4,282、ignore=258 | `992017e2da198e87f9ade9f8279c1797c7bfeeb00c1d5067b14489958f2b1d9c` |
| v3 action schema | `fall-risk-action-label-v3` | `1937070abb1cf230ad44185b91569f1f3a76f101a34b28b40168f76ef7f7abfd` |
| v3 event schema | `fall-risk-event-label-v3` | `30542206f41f8704717d6b8baf34c835ed8b37d36a884f4a764d196826a3c6ee` |

## 模型训练标签 v3

v3 是从上述 v2 根标签和哈希绑定受审决策确定性生成的独立训练视图，不改变 `fall-risk-data-v2` 发布候选。迁移报告 SHA-256 为 `41a33840ac0de6a584d63cde218d9c770b6613ef86aa64aeeffbce8fbb3a894f`，校验报告 SHA-256 为 `b9ac8f1f99ea70f2136675cec21ba24c060b1c8dc12ead8bf6fb6e6525d25eb1`。

当前统一 assignments 中共有 403 条 Pre_VFallp action/event 分配，全部位于 train，并由单一 `pre_vfallp_unresolved` 保护组隔离。旧步态预训练报告中“选中的 246 条全部位于 test”只属于旧 split；相关数据集、checkpoint 和指标在当前 split 上均需重建，不能混用。

迁移结果合并 71 个 LE2I/CVAT 重叠 fall；218 个 D04 与唯一父 fall 双向关联，2 个无唯一父事件的 D04 保持 ignore。2026-08-17 接纳 SCF P01/P02/P04 后，v3 统一 split 有 19,263 条标签分配、6,666 个资产和 187 个保守泄漏组，`split_id=splitv3_3342705b7b1ac51570148337`；继承旧 split 的 6,516 个资产，仅 SCF 451 条 assignment 新增且全部位于 train。校验未发现跨 partition 泄漏，`training_ready.fall_event=true` 、`training_ready.near_fall_event=true`；`action_type=false`。

复现命令：

```bash
conda run -n eldercare-ai python scripts/annotation/migrate_fall_labels_v2_to_v3.py --overwrite
conda run -n eldercare-ai python scripts/annotation/build_fall_training_split_v3.py \\
  --inherit-assignments <base-assignments-jsonl> --overwrite
conda run -n eldercare-ai python scripts/annotation/validate_fall_labels_v3.py --overwrite
```

## 步态 observable-context v2 训练

2026-08-17 将 `SCF_MVP_V1` 的 P01/P02/P04 纳入当前 v3 train 后，按既有 6,516 个资产的 partition 增量继承并追加 150 个 SCF 资产、451 条 assignment；P03/P05 未进入 root/v3/split。步态数据只把 SCF 用作 shared action encoder 和 walking gate 的 auxiliary 信号，conditional abnormal head 的 SCF 权重固定为 0，防止“来源=正常步态”捷径。

| 产物 | 状态/结果 | SHA-256 |
|---|---|---|
| `data/splits/fall_risk/training_labels_v3/assignments.jsonl` | 19,263 assignments；跨 partition 泄漏 0 | `f4f9ee9a1e58c9a840abea7dfd847dc5db5ce558de58536f899ddb5d4349d7b8` |
| `data/splits/fall_risk/training_labels_v3/split.json` | `splitv3_3342705b7b1ac51570148337`；6,666 assets / 187 groups | `668feae4bb17ec898a77dadd92495879cedb18572c5bfc1a8c149637d2095439` |
| action pretraining `dataset.npz` | 2,619 windows；SCF 138；train/validation 1,849/737 | `7eab581f5d4819a20b8e770a24ea1c0d46a4c23a75d5fa7e2a71c3014948468c` |
| gait context `dataset.npz` | 2,372 windows；train/validation 1,849/523；positive 202 | `37d8f93e18e41de50a6a1039845a082e815b2446804f9daf1442c05e53adfa1c` |
| gait context `metadata.json` | `development_provisional`；SCF 260 windows，conditional weight sum 0；test 未读取 | `3601872cd9fafc2b7aa7fccd628ad223e64ca8b3ac8b723ca7a46702d09a5717` |
| gait training manifest | hash-bound audit；`runtime_main_path=rule_fallback` | `4dba63f7267d062660075ec5e4573288f5bf65ba2c6283f6da6f30309b7e4f01` |

三 seed validation group F1：scratch `0.692/0.720/0.643`，action-pretrained transfer `0.621/0.857/0.857`；transfer 均值约 `0.778`，ROC-AUC 均值约 `0.996`。transfer 是当前 provisional 首选，但 validation 只有 15 个独立正段、2 个 source groups，且冻结回放的正常窗口误报为 `12.0-44.1/hour`。test pose/tensor/metrics 均未读取，split 和评估协议仍是 provisional，因此正式替换门禁未通过；2026-08-18 另有比赛受控运行启用记录，规则 fallback 保留。

## 跌倒事件训练 P0 审计

2026-08-08 的 fail-closed 审计状态为 `infrastructure_only`，`manifest_id=fall_event_training_manifest_ebb9deccdd37d70c402b9a4b`。当前 hash 绑定、v3 fall 监督和无泄漏机器报告通过；v2 formal、独立事件/困难负例保护组规模、连续背景、老人 ADL、frozen split/协议和 test 保管门禁未通过。该历史 candidate-clip 报告声明的 source split 为 `splitv3_f89832f6cad5b9c5f630d00b`，不能充当 2026-08-17 当前 `splitv3_3342705b7b1ac51570148337` 的结果，需按当前 split 重建。

| 产物 | SHA-256 |
|---|---|
| `reports/reproducibility/fall_event_training_manifest.json` | `611f80aecafe649e7bcb0b285f872524c452a13343784463f3e5458b03ab2cf7` |
| `reports/fall_risk/fall_event_training_audit.md` | `64ac0dbe453d63b2718652ee9b1bff82ee5b906e8f2fcd6999022c4d044f8f39` |
| `reports/fall_risk/fall_event_blockers.md` | `20843e2cf9f3d2a746be24425d30f6a659c3d95cb94d009f9f58d0b52d7d65af` |
| `src/elderly_monitoring/modules/fall_risk/fall_event_training_audit.py` | `a2a4a1550f6515264edcc1425ec7a53de91360d812600d303c2ffdc2b50bf72c` |
| `scripts/audit/audit_fall_event_training.py` | `d098c7af67e3eed2d82da7359bb5b3ad83938b9873d78091b750df7ae1f29b60` |
| `src/elderly_monitoring/modules/fall_risk/fall_event_continuous.py` | `db21b3529a284d217ef425110574a3801767c6d70aa9add21492b1480487f4bb` |

`fall_event_continuous.py` 当前只实现经合成测试的 `[T,17,20]` cutoff-only 输入契约；它没有读取真实 test、构建正式连续 dataset、训练 checkpoint 或接入规则主路径。

## 坐站连续事件 P0-P3 数据发布与构建

2026-08-11 坐站专项状态为 `development_provisional`。复用现有人工边界和事件类型发布 1,866 个事件、2,220 个显式背景和 192 个 ignore。过滤优化后使用 pose 可物化性重新平衡 development split，得到 `sitstandsplit_d751c5c4807698fc6882b593`；3,759 个窗口双构建一致，materialization gate 通过。边界正样本加权、流式事件解码、同协议规则比较和 36 组 bootstrap 修复已完成。seed 42 pilot 有相对增益但绝对门禁失败，seed 43/44 No-Go。`test_pose_read/test_features_generated/test_evaluated` 均为 `false`。

| 产物 | 状态 | SHA-256 |
|---|---|---|
| `reports/reproducibility/sit_stand_training_manifest.json` | P0 机器事实与执行授权 | `02d6d46d52cc833c0ad10c6b525d233e0a115feb4c2f4bcc6698e24d06ce3f79` |
| `reports/fall_risk/sit_stand_training_audit.md` | 过滤优化、seed 42 pilot 与完整流评估 | `af601b7b9b32728d8ebad790faa1428c8054dd8ee9688f0a0a34238e7518eeba` |
| `reports/fall_risk/sit_stand_training_blockers.md` | 三 seed No-Go 与 test 阻塞 | `f1279198f6d1ecdf15b55c3ee7233360d94274b08c6a81b5623f19ebd4cf03e1` |
| `configs/data/fall_risk_sit_stand_event_schema_v1.json` | 独立连续事件/背景/ignore 契约 | `e1b2a812c29cf63ca847a7e026a994db69aa25301dff788cb4c8bbd5292ff1b7` |
| `configs/training/sit_stand_event_v1.yaml` | 加权边界连续因果 TCN 配置，默认 checkpoint `null` | `235767671fc6630b026f49449c4659b634c2c757a3a95b549f722ebfe3c590f9` |
| `configs/evaluation/sit_stand_event_v1.provisional.yaml` | development provisional 一对一事件协议 | `05e00a2c62440acfdc828ffbd9dcddcb17e7cf88bfdcdd75a070cc5498989c71` |
| `data/annotations/fall_risk/sit_stand_event_labels_v1.jsonl` | 4,278 条 development 标签 | `39dc55aeadc72f5cd13089c3c383f97c3c05c9a8dfb5cc85094e9ac33a9ab513` |
| `data/splits/fall_risk/sit_stand_event_v1/split.json` | 专项保护 split | `15012cba7b2c7938f7a3f62cb53525b808751d6ed9db3e56d296b8ad6caefa84` |
| `build-h/dataset.npz` | 2,218 个 `[64,14,9]` train/validation 窗口 | `256035767380c2098ca31736118c3e7760f26daad49bd426e5969e1e4f79c207` |
| `build-h/metadata.json` | materialization gate `false` | `0fc97a076aa831e9c2127cb5e9bcce943308734f1ffd8229f095f73edd3d74fe` |
| `sit_stand_event_v1_pose_balanced/split.json` | pose-aware development split | `8a6c880dabb74b2ba14b3b362b84e81c3c37bee93c1097f5b9b7f00729afd4a0` |
| `training-a/dataset.npz` | 3,759 个 `[64,14,9]` 窗口及逐帧/边界监督 | `a417d24b817815d2cbba8f1ff303291f29f0a763614af4cfdf379598d36d756e` |
| `sit_stand_continuous_tcn.py` | 加权 boundary loss、早停与流解码 | `7029e2cc0c25442f4ba0a1dbbf005c3338ca92a32382a9b7d6e9f026a516933c` |
| `sit_stand_continuous_inference.py` | 完整姿态流 TCN/规则推理 | `91beebcbf0b170a0763061af843ea359d8bf169e91a363f0da298f571806dd91` |
| `sit_stand_event_evaluation.py` | 背景组 bootstrap 与来源分层 | `243fefd62c27e71b172e5b083208348d961b1fd303bbe0c467e018fde3a10660` |
| seed 42 smoke checkpoint | 2 epoch，development provisional | `ba31ecf551ac421479c365a68cf9401a4dc32449c9938b52468e046b0451ede9` |
| seed 42 weighted pilot checkpoint | 40 epoch，best epoch 35 | `8bf024c71075ce2802db582266df8c26469c6c235c3efb22927e184d9b37aa5c` |
| S0 完整流评估 | event F1 `0.1352` | `fcb9584bdf68c3372c19156c3e93ec03fdb90e1ce93eb8d098164020eeabf87d` |
| seed 42 TCN 完整流评估 | event F1 `0.3511`，三 seed No-Go | `8f8bd2aeefa4c9127ce23d895591925dd4a72e6b3f52118bd1c2ba07c0b5b9b5` |

最终 `training-a` 与 `training-b` 逐字节一致。validation 两方向为 64/54，四类困难负例保护组为 16/10/14/10。seed 42 pilot 的窗口级 direction/presence/frame/boundary exact F1 为 0.8625/0.7913/0.6190/0.2050。完整流 TCN event F1/Recall/FP-hour 为 0.3511/0.4029/331.9，规则为 0.1352/0.3741/1890.2；TCN 未达到绝对门禁且来源/困难负例退化明显，尚未形成可晋级候选。

## 近跌倒恢复确认开发训练

历史 `splitv3_e71a045eb58489f43dc5fd11` 的 train/validation 姿态窗口曾完成两次干净构建，数据集包含 1,105 个窗口、822 个事件；这些窗口、checkpoint 和指标不绑定当前 SCF 发布后的 split，需重建后才能作为当前近跌倒结果。test 只记录 733 条标签数量，姿态和指标未读取。

| 产物 | 状态/结果 | SHA-256 |
|---|---|---|
| `dataset.npz` | `[1105,24,10,8]`，train/validation only | `1804adfcfc29fa44469eb47dd878ad118727321949fa3a13089406b4f9f82083` |
| `metadata.json` | `development_provisional`，`test_pose_read=false` | `876b81a031224d7f2fc90fe291e1adcf9bca61cee4fb48282f95b8ead86a5517` |
| `samples.jsonl` | 1,105 条样本 provenance | `7c7685401d32e5cd25f6ad461ffc0939c64a6c1b40fd13f22eca9b003046d269` |
| seed 42 best checkpoint | validation 事件 F1 `0.9967` | `59c5f26fbd2467f47417c7029fcd0bc76b5567d0060704d6afc571801f1f0e5b` |
| seed 43 best checkpoint | validation 事件 F1 `0.9967` | `130bbcc7011d6f0d39b69f413b23105b7eda9760d34471827f3ab29b02c5eacf` |
| seed 44 best checkpoint | validation 事件 F1 `0.9933` | `00e65e4748dd5d13915447f30ae072e0bb3ceffcdde056cdfc2e4eddc83d52b7` |

三组结果均为 `development_provisional`、`test=null`、`test_evaluated=false`，不进入默认 checkpoint。完整覆盖、失败案例、预测文件和复现命令见[近跌倒恢复确认 TCN 开发训练报告](../fall_risk/near_fall_event_v1/README.md)。

## 跌倒连续训练数据治理

历史 v1 治理绑定 `splitv3_c7fd01cf4473c02a57c68fbe`，不作为当前 SCF 发布后的训练结果。v2 已重新绑定当前 `splitv3_3342705b7b1ac51570148337` 及 action/event/manifest/assignments/split/validation 六个输入哈希；生成 7,701 条开发监督、覆盖 5,338 个视频，train/validation=`5,888/1,813`，姿态缺失 0。3,730 条 test 标签在语义解析和姿态查找前封锁。

| 产物 | 状态/结果 | SHA-256 |
|---|---|---|
| `configs/data/fall_event_continuous_training_governance_v2.json` | 当前输入 hash、监督映射、loss/采样与因果姿态策略 | `89e681d554e0dbb464c6f70590be7918d5cb2b93146a988e36cda33a305e508c` |
| `fall_event_continuous_governance_v2/training_manifest.jsonl` | 7,701 条当前 train/validation 监督 | `14c5f88f5f25c4264efc1725b475f36de69b8ccd6d669b73140809aa54dfbc04` |
| `fall_event_continuous_governance_v2/audit.json` | 当前 split、姿态覆盖、采样质量和 test 封锁 | `415fec5125159b7c5c05ea1f1fa24777b3e45892d2ae8652ba9bdc2f384b3ea3` |
| `fall_event_continuous_dataset_v2/dataset.npz` | 7,504 个 `[32,17,20]` 因果窗口；train/validation=`5,721/1,783` | `34993ae9a6b599ce9452b59e4bf68cde192116bb0d3f3bad25e70275b6c62f71` |
| `fall_event_continuous_dataset_v2/metadata.json` | `future_observation_count=0`；test 未读取 | `062166ec39f030b1c9dd6acdb9c0e004c4a067c9e527be24ecd61578e1ae348c` |
| `fall_event_continuous_dataset_v2/samples.jsonl` | 7,504 条逐样本来源和 cutoff 审计 | `88e0ad0f1cc5ec95a1a0e0ee94564d05377dc29189228160e92b7b18786d5dc5` |
| `fall_event_continuous_tcn_v2/aggregate_metrics.json` | 三 seed F1/PR-AUC 均值 `0.6745/0.6782`；No-Go | `af9c5cb2a9615ed505afc6ffc878c1bd0d9e8a12bdd5d561f5059b04ce2ec4f2` |

完整配置、三个 checkpoint 哈希、分层误报和复现命令见[连续跌倒事件 TCN v2 开发训练报告](../fall_risk/fall_event_continuous_tcn_v2/README.md)。普通背景误报、onset 样本量、连续 camera-hour、老人域和冻结协议仍未通过，不授权运行主路径替换。

## Split 版本

v3 训练标签共用一份统一 split，不能复用下方绑定 v2 根标签的任务 split：

| split | 状态 | 标签/资产/组 | split_id | assignments SHA-256 | split.json SHA-256 |
|---|---|---:|---|---|---|
| `training_labels_v3` | split valid（继承旧 6,516 个资产；事件门禁 true、动作类型门禁 false） | 19,263 / 6,666 / 187 | `splitv3_3342705b7b1ac51570148337` | `f4f9ee9a1e58c9a840abea7dfd847dc5db5ce558de58536f899ddb5d4349d7b8` | `668feae4bb17ec898a77dadd92495879cedb18572c5bfc1a8c149637d2095439` |

split 产物已按 v2 根标签重建；`fall_event_v1` 是任务协议标识，不是旧标签版本。

| 任务 | 状态 | 样本 | split_id | split.json SHA-256 |
|---|---|---:|---|---|
| `fall_event_v1` | ready（provisional；train=1,391、validation=526、test=374） | 2,291 | `fall_event_v1:sha256:9f4b52f179ed013aa7d30ceeb3c7da35123085cbd8cdaa91dfd7c4ad11e16ca2` | `37ae87ea4d32f512369a193244d135097fede90105384e7d9d553819fb468585` |
| `near_fall_event_v1` | ready（provisional；train=1、validation=13、test=0） | 14 | `near_fall_event_v1:sha256:973a8332bd02f8b472eb00b67b0ec74bec0e4a31840ff13232dc65720fe84be0` | `c3dbb2051c6c3688cae65523f9185f70fcc8dfb59791b815c09d2b15bd6ee6c5` |
| `functional_proxy_v1` | blocked | 0 | `null` | `06067c0f0ed5c22cc74bcdcc898cbc24915edca48b4267d4ca41b5bd42215ec6` |
| `longitudinal_baseline_v1` | blocked | 0 | `null` | `334fbbde1f1539a0a3702ba2416f77cb8cd5cf451a49d38a530ccc839abf13f5` |

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
