# 标注脚本目录

本目录负责跌倒风险 manifest、CVAT 标签转换、LE2I 官方窗口、TOAGA、CaucaFall 和 NTU RGB+D 来源动作标签导入，以及严格校验。结构正确、来源文件可验证且对应 manifest 技术状态可用的明确标签默认可使用。

所有命令必须使用项目 conda 环境。先确认 editable 安装指向当前仓库：

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
```

## 1. 构建 manifest

```bash
conda run -n eldercare-ai python scripts/annotation/build_fall_risk_manifest.py \
  --repo-root . \
  --output data/manifests/fall_risk_video_manifest.jsonl \
  --ffprobe-bin ffprobe
```

manifest 记录媒体 hash、真实 FPS、时长、人员或保守源组、来源地址和技术排除原因。重复内容、来源缺失、媒体探测失败或隔离数据保持 `eligibility=false`。CaucaFall 的 100 个 AVI 均为技术可用媒体；完成导入后记录为 `label_source=cvat_manual`，并按受试者绑定脱敏任务 ZIP。当前 Pre_VFallp 的 108 条媒体均由 `configs/data/fall_risk_internal_authorizations.yaml` 启用，并分别绑定 5 个用户授权的 CVAT 压缩包：dizziness forward/side 36 条，其余 4 个批次各 18 条。内部授权只解除技术隔离，不等于公开来源已经验证。

`data/manifests/pre_vfallp_remaining_media_inventory_20260722.jsonl` 保留为四个 18 条子集在取得 CVAT 压缩包前的历史媒体清单，不再是当前授权事实源。需要复核这份历史清单时可运行：

```bash
conda run -n eldercare-ai python scripts/annotation/build_pre_vfallp_media_inventory.py \
  --repo-root . \
  --output data/manifests/pre_vfallp_remaining_media_inventory_20260722.jsonl \
  --inventory-id pre_vfallp_remaining_media_20260722 \
  --subset confusion_delirium \
  --subset confusion_nph \
  --subset weakness_fall_forward \
  --subset weakness_fall_side
```

## 2. 转换 CVAT 标签

含账号或邮箱元数据的导出先用 `scripts/annotation/redact_cvat_identity.xslt` 删除身份节点，并记录原始包与脱敏包的 SHA-256；不要直接把原始身份导出提交到公开版本库。该规则还会把旧格式 `fall_risk_office__...` task 名规范化为 manifest 可解析的 LE2I Office 名称。

```bash
conda run -n eldercare-ai python scripts/annotation/convert_cvat_fall_labels.py \
  --input data/annotations/fall_risk/cvat_exports/raw/<export>.zip \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --output-dir data/annotations/fall_risk/generated/v2/<batch-id> \
  --labeler labeler_fall_01
```

转换器按每条视频的真实 FPS 生成动作和映射事件 JSONL，并绑定原始导出路径与 SHA-256。不同导出使用不同目录，现有输出不会被覆盖。原始导出包含身份元数据时只输出告警，不将身份值复制进标签。

`fall_detection_2017` 的两个外部包需要使用专用合并器。它用 `flat_mp4修改.zip` 替换 `ADL_20240921` 项目，将 `ADL_20240922` 的 image 漏标任务 552 和 `Fall_20240919` 的 job 漏标任务 832 注入基础项目；同时把 640x360 框缩放到 manifest 的 1920x1080。四个 XML 227/230 帧任务按端点线性重采样到真实 57/58 帧，具体映射写入 `import_report.json`，不会把时间轴差异当作坐标缩放处理。

```bash
conda run -n eldercare-ai python scripts/annotation/import_fall_detection_2017_cvat_labels.py \
  --base-archive "/absolute/path/to/flat_mp4.zip" \
  --revision-archive "/absolute/path/to/flat_mp4修改.zip" \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --output-dir data/annotations/fall_risk/generated/v2/fall_detection_2017_manual \
  --overwrite
```

输出为候选 `action_labels.jsonl`、`event_labels.jsonl`、脱敏规范化 `source_annotations.zip` 和导入报告。当前批次导入 2,011 个 eligible 视频、2,977 条动作/映射事件；`20240917123819.mp4` 没有标注，两个重复内容视频按 manifest 技术排除。`20240922115152.mp4` 的 `U01` 备注“挥手”原样保留，并在报告中标为需要人工复核的语义警告。

Pre_VFallp 使用专用导入器。它支持压缩包目录内嵌 task ZIP、CVAT 多任务项目导出、项目全局帧到单视频局部帧的还原，以及在唯一媒体匹配且帧数相同时将 `_resized.mp4` 来源映射回 manifest 原视频。导入器会生成脱敏 task ZIP、候选 action/event JSONL 和导入报告：

```bash
conda run -n eldercare-ai python scripts/annotation/import_pre_vfallp_cvat_labels.py \
  --input "/absolute/path/to/<archive>.zip" \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --redacted-export-dir data/annotations/fall_risk/cvat_exports/raw/<batch-id> \
  --output-dir data/annotations/fall_risk/generated/v2/<batch-id> \
  --overwrite
```

当前 5 个批次共覆盖 108 个视频，导入 246 条动作和 246 条映射事件。只有单个 `outside=1` 框、没有任何可见区间的删除残留会作为 `outside_only` 写入 `ignored_tracks`；其他空轨迹或格式错误仍会阻断导入。

## 3. 导入 LE2I 官方窗口

```bash
conda run -n eldercare-ai python scripts/annotation/import_le2i_fall_labels.py \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --event-output data/annotations/fall_risk/generated/v2/le2i_official/event_labels.jsonl \
  --report-output data/annotations/fall_risk/generated/v2/le2i_official/import_report.json
```

导入器保留原始 TXT 路径、hash 和 1-based 源帧，同时输出 0-based 统一帧。`0/0` 只表示没有官方跌倒窗口，不生成事件。

## 4. 导入 TOAGA 正常步行

```bash
conda run -n eldercare-ai python scripts/annotation/import_toaga_normal_walk_labels.py \
  --manifest data/manifests/fall_risk_video_manifest.jsonl
```

导入器只读取 manifest 中 `dataset=toaga`、`subset=walking`、`media_type=video` 且 `eligibility=true` 的 RGB 视频。当前导入 14 位老人各一个 top/bottom 视角，共 28 条全片 `A01/normal_walk` 动作候选，来源绑定 `Table_1.xlsx` 的 SHA-256。它们没有 CVAT 轨迹或 bbox，也不生成 fall/near-fall 事件；迁移到 v3 后固定为 `auxiliary`、`source_verified` 和 `boundary_precision=unknown`，同一 `OAWxx` 的双视角保持在同一 source group。

## 5. 导入 CaucaFall 人工 CVAT 标注

```bash
conda run -n eldercare-ai python scripts/annotation/import_caucafall_cvat_labels.py \
  --input "/absolute/path/to/CAUCAFall标注.zip" \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --overwrite
```

导入器读取 10 个内层 CVAT ZIP，删除账号、邮箱和本地 URL 元数据，规范化 task 名称和受试者标识，并将 v3 风格标签别名显式归一到 v2 根标签。当前导入 100 个视频、311 条动作和 311 条映射事件；原始压缩包不复制到仓库，脱敏 ZIP 写入 `cvat_exports/raw/caucafall_manual/`，导入报告记录原始 SHA-256、别名统计和脱敏统计。

## 6. 导入 NTU RGB+D 单动作视频

```bash
conda run -n eldercare-ai python scripts/annotation/import_ntu_rgbd_clip_labels.py \
  --source-root data/external/ntu \
  --overwrite \
  --workers 8
```

该命令从仓库内 `data/external/ntu` 生成 `data/manifests/ntu_rgbd_clip_manifest.jsonl`，并把动作标签写入 `generated/v2/ntu_rgbd_clip_labels/`。`configs/data/ntu_rgbd_clip_label_map_v2.json` 记录 2026-07-25 项目负责人的人工精确边界决定：`A008 -> A03`、`A009 -> A04`、`A042 -> C03`、`A080 -> A05` 共导入 2,976 条；`A043` 不在这里按文件名直接导入。导入行使用 `source=ntu_rgbd_manual_clip_label`，进入 v2 根标签及 v3/split，并迁移为 `primary/exact/single_annotated`；只生成动作标签，不从 C03 自动生成 near-fall 事件。manifest 仍记录绝对媒体路径，因此移动仓库后必须用新位置重建本文件和全部下游 hash 绑定产物。

## 7. 导入 NTU RGB+D A043 人工 CVAT 批次

```bash
conda run -n eldercare-ai python scripts/annotation/import_ntu_rgbd_a043_cvat_labels.py \
  --source "/path/to/nturgbd_rgb_s001.zip" \
  --source "/path/to/nturgbd_rgb_s002" \
  --source "/path/to/nturgbd_rgb_s003" \
  --source "/path/to/nturgbd_rgb_s004" \
  --source "/path/to/nturgbd_rgb_s005" \
  --source "/path/to/nturgbd_rgb_s006.zip" \
  --source "/path/to/nturgbd_rgb_s007.zip" \
  --source "/path/to/nturgbd_rgb_s008.zip" \
  --source "/path/to/nturgbd_rgb_s009.zip" \
  --source "/path/to/nturgbd_rgb_s010.zip" \
  --source "/path/to/ntu s11-s15.zip" \
  --source "/path/to/nturgbd_rgb_s016.zip" \
  --source "/path/to/nturgbd_rgb_s017.zip" \
  --revision "/path/to/s06-s10/修改的标注.zip" \
  --revision "/path/to/S006C003P007R002A043_rgb.avi.zip" \
  --revision "/path/to/S008C001P015R002A043_rgb.avi.zip" \
  --revision "/path/to/S016C003P008R001A043.zip" \
  --allow-incomplete
```

该导入器按源文件名和帧数关联 `ntu_rgbd_clip_manifest.jsonl`，支持 CVAT project/task/job ZIP 及 AVI/MP4 source 名；`--revision` 只能替换基础包中同名任务，原始空任务若没有修订会阻断。job XML 本身没有源视频名，因此 job ZIP 文件名必须严格匹配 `SxxxCxxxPxxxRxxxA043[ _rgb].zip`，否则拒绝导入。导入报告同时记录基础 project、revision job 的 SHA-256 和唯一源名；生成的 `source_annotations.zip` 是带完整 project task/source 元数据的规范化重建包，不冒充 CVAT 原生 project 再导出。导入器验证连续、无重叠动作序列，支持正常行走后跌倒、末帧 `outside=1`、跌倒后坐站、受控躺下非跌倒、受控下蹲困难负样本和全片跌倒协议；生成的脱敏项目用显式 `project_global` 帧坐标，补齐受试者 ID，并删除账号、邮箱和本地 URL。

当前 S001-S017 共 938 个视频，生成 1,428 条动作和 1,428 条映射事件；S016 的 `S016C003P008R001A043` job revision 已覆盖原 project 中的 D03，并按三视角裁决为 D02。S013/P018/R001/C001 裁决为 D01，S015/P015/R001/C003 裁决为 D02；S011/P015/R001 和 S017/P020/R001 的六个三视角下蹲视频裁决为 A05，并授权为 fall hard negative。产物与脱敏合并 ZIP 位于 `generated/v2/ntu_rgbd_a043_cvat_review/`。主 manifest 从该批次动作 JSONL 提取 `video_id` 白名单，发布器只接纳目录名与报告 `batch_id` 一致的批次；未标注 A043 和候选目录不会进入根标签。S002 仍缺 10 个 C001 任务，因此真实导入需要显式 `--allow-incomplete`；原导入中 446 个全片跌倒没有片内 onset，已由 2026-08-04 裁决在 v3 统一为首帧 onset、媒体尾帧 offset。

脱敏合并包中的媒体名规范为原始 `.avi`，导入器同时接受原始 CVAT `.mp4` 名和该受控 `.avi` 名，因此 `source_annotations.zip` 可用于确定性重放；动作码、帧数和 manifest 唯一匹配规则不变。

## 8. 导入抖音/B站跌倒视频 CVAT 标注

原始目录中的 66 个稀疏编号视频已按旧编号升序重命名为 `1.mp4` 至 `66.mp4`，可逆映射保存在 `configs/data/fall_tiktok_source_map_v1.json`。CVAT 实际标注的是 66 个剪辑，统一保存为 `data/external/抖音b站跌倒视频整理/annotated_clips/1.mp4` 至 `66.mp4`。

```bash
conda run -n eldercare-ai python scripts/annotation/import_fall_tiktok_cvat_labels.py \
  --input "/Users/guai/Documents/qq files/fall_tiktok.zip" \
  --overwrite
```

导入器按 source map 将 `001.mp4` 至 `066.mp4` 的 CVAT 任务绑定到 `fall_tiktok_clip_001` 至 `fall_tiktok_clip_066`，移除身份元数据，并输出脱敏 CVAT ZIP、150 条动作、150 条映射事件和导入报告。`configs/data/fall_tiktok_collection_decision_v1.json` 记录项目负责人确认的项目自采及内部训练授权；该决定不授予公开再分发权，未知人员仍合并到单一保守 source group。manifest 中 66 个标注剪辑和 64 个非重复原视频 eligible，2 个重复原视频保持技术排除。v3 迁移结果为动作 primary=126、auxiliary=21、ignore=3，事件 auxiliary=65、ignore=6。

## 9. 发布 v2 根标签

```bash
conda run -n eldercare-ai python scripts/annotation/publish_v2_fall_labels.py \
  --source-root data/annotations/fall_risk/generated/v2 \
  --overwrite
```

发布器保留全部 v2 动作标签；官方 LE2I TXT 跌倒窗口优先，排除与其重叠的 CVAT `fall` 事件，并在 `reports/fall_risk/fall-risk-data-v2-root-publish.json` 记录来源和输出 hash。

## 10. 严格校验

```bash
conda run -n eldercare-ai python scripts/annotation/validate_fall_risk_labels.py \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --action-labels data/annotations/fall_risk/action_labels.jsonl \
  --event-labels data/annotations/fall_risk/event_labels.jsonl \
  --risk-labels data/annotations/fall_risk/risk_labels.jsonl \
  --subject-profiles data/annotations/fall_risk/subject_profiles.json \
  --config configs/data/fall_risk_label_validation_v2.yaml \
  --mode formal \
  --report-output reports/fall_risk/label_validation_formal_v2.json
```

校验器检查 schema、媒体与标注来源、hash、时间/帧换算、动作到事件映射、人员标识、技术排除和 `uncertain/U01`。

## 11. 生成模型训练标签 v3

v3 训练标签从现行 v2 根标签确定性生成，不覆盖 v2：

```bash
conda run -n eldercare-ai python scripts/annotation/migrate_fall_labels_v2_to_v3.py --overwrite
```

迁移器执行以下训练语义转换：

- v2 inclusive end 转为 v3 half-open end。
- 动作改为 `action_family + action_type` 层级目标。
- 父类与具体动作分别生成 `training_tier` 和 `action_type_training_tier`；具体动作层级按 sample/source group 数量计算并受父类层级上限约束。
- C03 仅依据默认的哈希绑定项目裁决升级为 `stumble_recovery` near-fall positive；C04/C05 不自动升级。
- D04 改为 `post_fall_immobile` 并关联父 fall。
- LE2I/CVAT 重叠 fall 合并为一个 event，多个来源写入 `source_refs`。
- U01 为 fall/near-fall 分别生成 ignore mask。
- 默认读取 `fall_risk_training_decision_20260804.json`；它绑定当前 v2 action SHA-256，固定 NTU 全片边界、C03 正例、任务级 hard-negative 映射和 UR Fall A07 视频覆盖规则。标签 hash 漂移、重复 action/task 目标或裁决零匹配都会被报告或拒绝。
- 明确动作只按裁决映射为 task-specific negative；`partial_occlusion` 可在事件任务中由复核决定成为 primary `occlusion_or_camera_motion`，但动作任务层级仍保持 auxiliary。重遮挡、出画、多人不确定和 U01 不生成 negative。
- 未标注背景和 LE2I `0/0` 不自动生成 negative。

当前输出为 9,612 条动作和 9,651 条事件窗口，其中 event positive=3,284、negative=6,109、ignore=258；fall 正/负=2,303/1,827，near-fall 正/负=981/4,282。SCF 的 451 条 assignment 全部固定在 train。

### 11.1 发布新增人工近跌倒 v3 候选

当前受审 C03 已通过上述项目裁决进入正式 v3。对未来新增 C04/C05 或独立背景窗口，先由标注员逐视频确认并生成 `near-fall-manual-decision-v1` JSONL。每一行都必须绑定一个不可变人工复核导出文件及其 SHA-256，不能把规则结果、动作码、目录名或未标注背景作为来源。正例必须满足：

- `label_role=positive`、`target_status=confirmed`。
- `onset_frame <= peak_frame <= recovery_frame`；`peak_frame` 可为 `null`，但 `recovery_frame` 不可缺失。
- `review_status=double_reviewed` 或 `adjudicated`，且 `reviewer_ids` 至少有两个不同人员。
- `physical_event_id` 唯一，`event_subtype` 为五类 recovery subtype 之一。
- 首版 `linked_action_ids=[]`；当前发布器不改 action 标签，不能生成不对称引用。

负例必须逐窗人工确认，至少一名复核员，所有 event/point/physical event 字段为 `null`。除显式人工背景窗外，训练门禁要求以下八类全部有覆盖：`normal_turn`、`normal_step_adjustment`、`routine_support_contact`、`fast_but_controlled_sit`、`controlled_squat`、`controlled_bend`、`exercise_or_stretch`、`progressed_to_fall`。人工背景窗不能替代这八类；`progressed_to_fall` 首版是 near-fall binary negative。

输入行的共同字段和一个正例示例如下；JSONL 实际保存时每条记录占一行：

```json
{"schema_version":"near-fall-manual-decision-v1","decision_id":"near_fall_positive_001","source_type":"manual_near_fall_v1","video_id":"video_1","track_id":"person_1","label_role":"positive","start_frame":100,"end_frame_exclusive":181,"frame_index_base":0,"target_status":"confirmed","boundary_precision":"exact","quality_flags":[],"annotator_id":"annotator_01","reviewer_ids":["reviewer_01","reviewer_02"],"review_status":"double_reviewed","note":"Confirmed recovery without a fall.","source_annotation_path":"data/annotations/fall_risk/reviews/near_fall_batch_001.json","source_annotation_sha256":"<64 lowercase hex>","physical_event_id":"physical_<24 lowercase hex>","event_subtype":"stumble_recovery","hard_negative_type":null,"onset_frame":112,"peak_frame":138,"impact_frame":null,"recovery_frame":172,"linked_action_ids":[],"contact_evidence":"observed"}
```

发布器只写独立候选文件，不允许直接覆盖 v3 事实源：

```bash
conda run -n eldercare-ai python scripts/annotation/publish_near_fall_labels_v3.py \
  --decisions data/annotations/fall_risk/near_fall_manual_decisions_v1.jsonl \
  --output-event-labels data/annotations/fall_risk/event_labels_v3.near_fall_candidate.jsonl \
  --report reports/fall_risk/near-fall-label-publication-v1.json
```

命令会验证人工来源、来源 hash、manifest 资格、half-open 边界、双审、恢复点、低质量排除、稳定 ID 和重复项。候选需由数据负责人逐行审核后才能进入版本化事实源；发布器不会自动执行该批准动作。

### 11.2 重建候选 split 并校验门禁

生成动作和事件共用的 v3 防泄漏 split：

```bash
conda run -n eldercare-ai python scripts/annotation/build_fall_training_split_v3.py \
  --event-labels data/annotations/fall_risk/event_labels_v3.near_fall_candidate.jsonl \
  --assignments-output /tmp/near_fall_v3_candidate_split/assignments.jsonl \
  --report-output /tmp/near_fall_v3_candidate_split/split.json

conda run -n eldercare-ai python scripts/annotation/validate_fall_labels_v3.py \
  --event-labels data/annotations/fall_risk/event_labels_v3.near_fall_candidate.jsonl \
  --split-assignments /tmp/near_fall_v3_candidate_split/assignments.jsonl \
  --split-report /tmp/near_fall_v3_candidate_split/split.json \
  --report-output /tmp/near_fall_v3_candidate_validation.json
```

只有候选报告同时满足 `valid=true` 和 `training_ready.near_fall_event=true`，且人工检查确认 train/validation 的人员、来源、场景和八类 hard negative 覆盖足够，才能准备真实 train/validation 数据。此步骤不读取 test 姿态、不输出 test 指标，也不能通过拆分同一受试者或来源组改善数字。

正式 v3 split 有 19,263 条标签分配、6,666 个资产和 187 个保守泄漏组，校验未发现跨 partition 泄漏。SCF P01/P02/P04 强制位于 train，P03/P05 不进入该 split。

## 12. 校验模型训练标签 v3

```bash
conda run -n eldercare-ai python scripts/annotation/validate_fall_labels_v3.py
```

校验器检查闭合 schema、half-open 边界、manifest/hash、动作 taxonomy、具体动作训练层级、positive/negative/ignore 条件、near-fall recovery/双审、双向父子引用、重复 physical event，以及 v3 split 的输入 hash、`split_id` 和跨分区泄漏。结构合法不等于数据可训练；当前报告会明确返回：

```text
valid=true
training_ready.action_type=false
training_ready.fall_event=true
training_ready.near_fall_event=true
```

`action_type=false` 的直接原因是部分稀有 primary 类仍未覆盖全部分区。两个事件门槛为 true 只表示现有事件监督可按统一 split 开发训练；未取得连续背景分母、老人域验证和冻结评估协议前，不得把它解释为正式模型效果，也不得用自动背景窗或拆散保守源组改善数字。

## 13. 模型候选收敛

KINECAL 下载和准备入口已退休。当前各任务只保留一个 TCN 实验候选，规则 baseline 继续作为运行主路径和低质量输入 fallback。
