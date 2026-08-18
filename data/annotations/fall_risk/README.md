# 跌倒风险标注目录

根目录统一标签包括：

```text
action_labels.jsonl
event_labels.jsonl
risk_labels.jsonl
subject_profiles.json
```

v2 是根标签与发布候选契约。当前根目录有 9,612 条动作和 6,338 条事件；audit 结构错误为 0，但 formal 仍有 285 个 blocker，因此不能称为 frozen 数据发布。

供动作分类和 fall/near-fall 事件模型使用的训练版 v3 标签独立保存为：

```text
action_labels_v3.jsonl
event_labels_v3.jsonl
```

v3 不覆盖 v2。当前迁移结果为 9,612 条动作和 9,651 条事件窗口：动作 primary=8,699、auxiliary=745、ignore=168；事件 positive=3,284、negative=6,109、ignore=258，其中 fall 正/负=2,303/1,827，near-fall 正/负=981/4,282。

连续坐站事件使用独立契约，不回写根 v2/v3。旧 `sit_stand_event_labels_v1.jsonl` 保留用于历史复现；2026-08-17 的 `sit_stand_event_labels_v2.jsonl` 和 review log 复用当前受审动作边界，发布 1,942 个事件、5,512 个显式背景和 229 个 ignore。SCF P01/P02/P04 的 298 条标签强制 auxiliary/train-only，P03/P05 不进入发布；未标注背景和 test 真值不生成。最终物化感知 split 为 `sitstandsplit_11e163684e3bb898a5c3508d`，test 姿态、特征和评估均未读取。

v3 任务级正负样本只来自 `configs/data/fall_risk_training_decision_20260804.json` 的哈希绑定项目裁决，不是按动作名临时推断。当前 962 条 `stumble_recovery` 正例由 948 条 NTU C03、13 条 Pre_VFallp C03 和 1 条 fall_tiktok C03 组成；未标注背景、LE2I `0/0`、U01、重遮挡和出画没有转成 negative。未来新增 C04/C05 或背景窗仍需独立人工复核。

CaucaFall 的 100 个 AVI 已完成手工 CVAT 标注并进入主标签链，manifest 均为 `eligibility=true`、`label_source=cvat_manual`，并绑定 10 个脱敏任务 ZIP。导入生成 311 条动作和 311 条映射事件；100 个视频、10 名受试者均保留，目录动作名只保留为来源元数据。v2/v3 迁移对少量 v3 标签名做了显式别名归一，原始标签和映射记录在 `generated/v2/caucafall_manual/import_report.json` 中。

UR Fall 的人工 CVAT 导出已作为 `generated/v2/cvat_ur_fall/` 接入 v2 根标签：99 个视频生成 268 条动作和 268 条映射事件，`adl-07-cam0.mp4` 按人工决定不标注。包含账号/邮箱的原始 ZIP 不入库；仓库只保存规范化任务名、补齐 3 条空 `U01` 原因并移除身份节点后的脱敏 ZIP，原始与脱敏 SHA-256 及变更记录见批次 `import_report.json`。该批次现已随根标签迁移进入 v3 和统一 split。

抖音/B站跌倒视频整理批次已按原始稀疏编号升序使用 `1.mp4` 至 `66.mp4`，旧名到新名的完整映射固定在 `configs/data/fall_tiktok_source_map_v1.json`。CVAT 的 66 个标注剪辑保存在 `annotated_clips/1.mp4` 至 `66.mp4`，当前 manifest 仅保留这 66 个 clip；根目录原始视频已移入废纸篓，不再作为媒体资产或训练输入。CVAT 的 66 个任务生成 150 条动作和 150 条映射事件，并已发布到 v2 根标签。项目负责人已在 `configs/data/fall_tiktok_collection_decision_v1.json` 确认为项目自采并授权内部训练。未知人员共用 `fall_tiktok_project_collection_pool`，本决定不授予公开再分发权。迁移后的动作标签为 primary=126、auxiliary=21、ignore=3，事件标签为 auxiliary=65、ignore=6；U01 与现行质量/类别门槛仍保持 ignore。统一 split 为防止未知人员泄漏，将该来源组的 221 条动作/事件 assignment 全部放在 test；取得人员对应关系并重新分组前，不把它们强行拆到 train。原始 CVAT 包不入库，脱敏副本和哈希见 `generated/v2/fall_tiktok_manual/import_report.json`。

NTU RGB+D 单动作来源按项目负责人 2026-07-25 的人工复核决定接入主标签链：`generated/v2/ntu_rgbd_clip_labels/` 含 2,976 条人工精确全片动作，映射与决策元数据固定在 `configs/data/ntu_rgbd_clip_label_map_v2.json`。它们对应 `A008 -> A03` 948 条、`A009 -> A04` 948 条、`A042 -> C03` 948 条、`A080 -> A05` 132 条；v3 的 `training_tier`、`action_type_training_tier` 均为 `primary`，`boundary_precision=exact`、`review_status=single_annotated`。`A043` 仍禁止按文件名直接映射，只有出现在已接受人工 CVAT 批次中的视频才能接入。v2 来源只产生动作标签；v3 迁移依据受审训练决定把这 948 条 C03 生成为 near-fall 正例，而不是由文件名或迁移器默认规则自动派生。外部 manifest 已按仓库内 `data/external/ntu` 重建，3,924 个 AVI 均可访问；主 manifest 纳入的 3,914 个 NTU 资产已通过文件存在性校验。

S001-S017 A043 CVAT 导出已按 `configs/data/ntu_rgbd_a043_cvat_decision_v1.json` 的项目负责人决定接入主标签链。`generated/v2/ntu_rgbd_a043_cvat_review/` 覆盖 938 个视频，生成 1,428 条动作和 1,428 条映射事件；主 manifest 只白名单导入这 938 个 `video_id`，不会自动导入未标注 A043。S016/C003/P008/R001 的严格命名 job revision 叠加在原完整 S016 project 上；合并 ZIP 已删除身份元数据，并重建为含完整 task/source 元数据的规范化 project。S013/C001/P018/R001 裁决为 D01，S015/C003/P015/R001 裁决为 D02；S011/P015/R001 与 S017/P020/R001 的六个视角裁决为 A05 fall hard negative。该批次仍缺少 S002 的 10 个 C001 任务。原 CVAT 导入把 446 个全片跌倒记录为没有片内 onset；现行 v3 受审决定已统一使用首帧 onset、媒体尾帧 offset，写为 `[0, frame_count)` 精确边界。306 个完整三视角组中有 303 组一致含跌倒、3 组一致非跌倒、0 组跌倒覆盖冲突；51 组方向不一致、36 组原 CVAT 边界差超过 5 帧，均保留为 QC 事实。

Fall Detection 2017 的人工 CVAT 批次已接入 v2/v3 候选链：2,011 个源成功导入，生成 2,977 条 v2 动作/事件、2,977 条 v3 动作和 1,097 条 v3 事件窗口。该批次仍标记为 `project_collected_manual_cvat_unverified`，训练策略为 `candidate_requires_qc_review`；2 个源技术排除，另有 1 个 manifest 可用源尚未匹配。来源、归一化和 QC 警告固定在 `generated/v2/fall_detection_2017_manual/import_report.json`，不能据此宣称正式训练就绪。

SCF_MVP_V1 的 262 个自采视频已完成对账。P01/P02/P04 中 150 个视频、298 条受审动作已进入根 manifest、v2/v3 和 train split；P03 继续 excluded，P05 继续 challenge-only。来源候选仍完整保留供审计。

动作父类与具体动作分别使用 `training_tier` 和 `action_type_training_tier`。具体动作少于 10 个独立 sample group 时忽略，10-29 个或来源少于 3 个 source group 时只作 auxiliary，至少 30 个 sample group且至少 3 个 source group才可作 primary；训练代码必须从 `action_labels_v3.jsonl` 读取该层级，不能只读取 split assignment 中的父类 tier。

动作与事件共用 `data/splits/fall_risk/training_labels_v3/` 下的 v3 split：19,263 条分配覆盖 6,666 个资产、187 个保守泄漏组，跨分区泄漏为 0。SCF 的 451 条分配全部位于 train；P03/P05 不在该 split。

当前 v3 schema、引用和 split 校验为 `valid=true`、`issues=[]`，`training_ready.fall_event=true`、`training_ready.near_fall_event=true`；部分 primary 动作类别仍未覆盖全部分区，故 `training_ready.action_type=false`。事件监督门禁通过只表示现有标签可按该 split 做开发训练，不代表 split 已冻结、模型已通过连续背景/老人域/正式 test 验证，也不代表可以替换规则主路径。

相关入口：

```text
configs/data/fall_risk_action_label_schema_v3.json
configs/data/fall_risk_event_label_schema_v3.json
scripts/annotation/migrate_fall_labels_v2_to_v3.py
scripts/annotation/build_fall_training_split_v3.py
scripts/annotation/validate_fall_labels_v3.py
data/splits/fall_risk/training_labels_v3/assignments.jsonl
data/splits/fall_risk/training_labels_v3/split.json
reports/fall_risk/training-labels-v3-migration.json
reports/fall_risk/training-labels-v3-validation.json
```

`generated/` 保存按来源批次隔离的转换结果：

```text
generated/v2/cvat_<export-id>/action_labels.jsonl
generated/v2/cvat_<export-id>/event_labels.jsonl
generated/v2/fall_detection_2017_manual/action_labels.jsonl
generated/v2/fall_detection_2017_manual/event_labels.jsonl
generated/v2/fall_detection_2017_manual/import_report.json
generated/v2/fall_detection_2017_manual/source_annotations.zip
generated/v2/le2i_official/event_labels.jsonl
generated/v2/le2i_official/import_report.json
generated/v2/toaga_official_walking/action_labels.jsonl
generated/v2/toaga_official_walking/import_report.json
generated/v2/pre_vfallp_confusion_delirium/action_labels.jsonl
generated/v2/pre_vfallp_confusion_delirium/event_labels.jsonl
generated/v2/pre_vfallp_confusion_delirium/import_report.json
generated/v2/pre_vfallp_confusion_nph/action_labels.jsonl
generated/v2/pre_vfallp_confusion_nph/event_labels.jsonl
generated/v2/pre_vfallp_confusion_nph/import_report.json
generated/v2/pre_vfallp_dizziness_fall_forward_side/action_labels.jsonl
generated/v2/pre_vfallp_dizziness_fall_forward_side/event_labels.jsonl
generated/v2/pre_vfallp_dizziness_fall_forward_side/import_report.json
generated/v2/pre_vfallp_weakness_fall_forward/action_labels.jsonl
generated/v2/pre_vfallp_weakness_fall_forward/event_labels.jsonl
generated/v2/pre_vfallp_weakness_fall_forward/import_report.json
generated/v2/pre_vfallp_weakness_fall_side/action_labels.jsonl
generated/v2/pre_vfallp_weakness_fall_side/event_labels.jsonl
generated/v2/pre_vfallp_weakness_fall_side/import_report.json
generated/v2/ntu_rgbd_clip_labels/action_labels.jsonl
generated/v2/ntu_rgbd_clip_labels/import_report.json
generated/v2/ntu_rgbd_a043_cvat_review/action_labels.jsonl
generated/v2/ntu_rgbd_a043_cvat_review/event_labels.jsonl
generated/v2/ntu_rgbd_a043_cvat_review/import_report.json
generated/v2/ntu_rgbd_a043_cvat_review/source_annotations.zip
generated/v2/caucafall_manual/action_labels.jsonl
generated/v2/caucafall_manual/event_labels.jsonl
generated/v2/caucafall_manual/import_report.json
generated/v2/cvat_ur_fall/action_labels.jsonl
generated/v2/cvat_ur_fall/event_labels.jsonl
generated/v2/cvat_ur_fall/import_report.json
generated/v2/fall_tiktok_manual/action_labels.jsonl
generated/v2/fall_tiktok_manual/event_labels.jsonl
generated/v2/fall_tiktok_manual/import_report.json
```

`cvat_exports/raw/` 保存生成标签所依据的原始 CVAT ZIP/XML。转换器绑定来源路径与 SHA-256；原始文件缺失或 hash 不一致时标签校验失败。包含身份元数据的原件不得提交到公开版本库。

含 CVAT 身份元数据的外部导出应先记录原始 SHA-256，再使用 `scripts/annotation/redact_cvat_identity.xslt` 生成脱敏 ZIP；转换和校验只引用脱敏副本。原始包不修改、不复制进仓库，脱敏包与原始 hash 的对应关系写入数据审计报告。

原始 CVAT 文件中的项目名、导出工具元数据和历史任务备注不定义当前标签契约。根标签与发布候选以 v2 schema、映射和校验配置为准；模型训练以 v3 schema、哈希绑定训练决定、v3 标签、统一 split 和校验器为准。原始文件只作为不可变追溯证据保存，v2 与 v3 不能互相替代。

`quarantine/` 保存媒体或来源无法可靠关联的记录，不进入数据划分。当前 108 条 Pre_VFallp 已通过内部授权例外解除技术隔离，但仍不是已验证的公开来源；未来未列入授权清单的媒体仍保持隔离。技术可用性仍受 manifest 的重复内容、来源缺失、媒体探测失败和隔离状态约束。

字段和时间边界见 `docs/modules/fall_risk/data/跌倒风险标签字典.md`。

CVAT 项目标签配置见 `configs/data/fall_risk_cvat_labels_v2.json`。
