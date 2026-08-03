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

当前 S001-S017 共 938 个视频，生成 1,428 条动作和 1,428 条映射事件；S016 的 `S016C003P008R001A043` job revision 已覆盖原 project 中的 D03，并按三视角裁决为 D02。S013/P018/R001/C001 裁决为 D01，S015/P015/R001/C003 裁决为 D02；S011/P015/R001 和 S017/P020/R001 的六个三视角下蹲视频裁决为 A05，并授权为 fall hard negative。产物与脱敏合并 ZIP 位于 `generated/v2/ntu_rgbd_a043_cvat_review/`。主 manifest 从该批次动作 JSONL 提取 `video_id` 白名单，发布器只接纳目录名与报告 `batch_id` 一致的批次；未标注 A043 和候选目录不会进入根标签。S002 仍缺 10 个 C001 任务，因此真实导入需要显式 `--allow-incomplete`；446 个全片跌倒标签没有人工 onset。

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
conda run -n eldercare-ai python scripts/annotation/migrate_fall_labels_v2_to_v3.py
```

迁移器执行以下训练语义转换：

- v2 inclusive end 转为 v3 half-open end。
- 动作改为 `action_family + action_type` 层级目标。
- 父类与具体动作分别生成 `training_tier` 和 `action_type_training_tier`；具体动作层级按 sample/source group 数量计算并受父类层级上限约束。
- C03-C05 不自动升级为 near-fall positive。
- D04 改为 `post_fall_immobile` 并关联父 fall。
- LE2I/CVAT 重叠 fall 合并为一个 event，多个来源写入 `source_refs`。
- U01 为 fall/near-fall 分别生成 ignore mask。
- `ntu_rgbd_a043_cvat_decision_v1.json` 中经三视角裁决并授权的 A05 窗口生成 `manual_v3/adjudicated` 的 `squat_or_kneel` fall hard negative；裁决文件及 SHA-256 写入 source refs。
- 未标注背景和 LE2I `0/0` 不自动生成 negative。

当前输出为 6,337 条动作和 1,470 条事件窗口，其中 action primary=5,841、auxiliary=380、ignore=116；event positive=1,306、negative=6、ignore=158、near-fall positive=0。六条 negative 均为人工裁决的 `squat_or_kneel`，裁决零漏配。NTU 的 2,976 条人工精确全片动作保持原有 tier；A043 人工 CVAT 标签按接受决定和边界精度迁移；fall_tiktok 自采批次按人工标注与现行层级规则进入训练。Pre_VFallp 的 14 条 v2 `near_fall` 映射和 NTU 的 948 条 C03 动作，按 v3 契约都不会自动升级为 near-fall 训练正例。

生成动作和事件共用的 v3 防泄漏 split：

```bash
conda run -n eldercare-ai python scripts/annotation/build_fall_training_split_v3.py
```

当前 split 有 7,807 条标签分配、4,505 个资产和 155 个保守泄漏组，校验未发现跨 partition 泄漏，primary fall 正例按 train/validation/test 分为 74/14/7，primary fall negative 为 6/0/0。NTU 按受试者组不跨 partition；因此不能为了让六条负样本覆盖 validation/test 而拆散 P015/P020 所属保护组。Pre_VFallp 维持一个保守源组；CaucaFall 按 10 名受试者分组。旧 v2 `fall_event_v1` split 不适用于 v3。

## 12. 校验模型训练标签 v3

```bash
conda run -n eldercare-ai python scripts/annotation/validate_fall_labels_v3.py
```

校验器检查闭合 schema、half-open 边界、manifest/hash、动作 taxonomy、具体动作训练层级、positive/negative/ignore 条件、near-fall recovery/双审、双向父子引用、重复 physical event，以及 v3 split 的输入 hash、`split_id` 和跨分区泄漏。结构合法不等于数据可训练；当前报告会明确返回：

```text
valid=true
training_ready.action_type=false
training_ready.fall_event=false
training_ready.near_fall_event=false
```

`action_type=false` 的直接原因是部分 primary 类仍未覆盖全部分区；在人工补齐 hard negative、near-fall 正例并恢复每个 primary 类的三分区覆盖前，不得用自动背景窗或拆散保守源组绕过门禁。

## 13. 下载 KINECAL 风险组骨架

```bash
conda run -n eldercare-ai python scripts/annotation/download_kinecal.py \
  --output-dir data/external/kinecal/raw \
  --workers 32 \
  --retries 3 \
  --allow-missing
```

该命令只下载 `NF/FHs/FHm` 三个官方风险组在 3 米步行、TUG 和 STS-5 动作中的 Kinect 骨架文本，不下载逐帧深度二进制文件。KINECAL v1.0.3 源站在该范围内有 17 个动作目录不存在，因此正式下载显式使用 `--allow-missing`，随后仍必须运行本地 manifest 校验。下载范围、源数据缺口、年龄冲突和 TCN 使用方式见 `data/external/kinecal/README.md`。

本地校验：

```bash
conda run -n eldercare-ai python scripts/annotation/download_kinecal.py \
  --output-dir data/external/kinecal/raw \
  --verify-only
```
