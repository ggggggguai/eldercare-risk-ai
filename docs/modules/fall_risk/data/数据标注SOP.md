# 跌倒风险数据标注 SOP

更新时间：2026-07-15

本文是跌倒风险视频数据标注的执行 SOP，面向标注员、复核人员和工程转换人员。标签定义以 `docs/modules/fall_risk/data/跌倒风险标签字典.md` 为准，数据集选择和统一输出格式以 `docs/modules/fall_risk/data/数据集标注规范.md` 为准。本文只说明怎么把云端标注工作从账号注册、项目创建、实际标注、复核、导出到归档完整执行起来。

## 1. 总原则

- 标注员只记录视频中能看见的事实，不做医学诊断，不判断老人真实健康状态。
- 第一阶段优先标动作级标签，风险级标签只由复核人员或项目负责人确认。
- 不临时发明标签。所有标签必须来自 `跌倒风险标签字典.md`。
- 看不清、遮挡严重、人物离开画面或目标老人无法确认时，使用 `U01`，不要猜。
- 公开视频、官方 txt、人工标注、表格 proxy 和自动算法结果必须区分来源，不能混成同一种真值。
- CVAT 中的矩形框主要用于承载“动作片段时间段 + 目标对象身份”，不是本项目最终的人体检测真值。

## 2. 角色分工

| 角色    | 主要职责                             | 不做什么               |
| ----- | -------------------------------- | ------------------ |
| 数据管理员 | 准备视频、manifest、CVAT 项目和任务，管理导出文件  | 不擅自修改标签字典          |
| 标注员   | 按 SOP 标动作片段、质量标记和备注              | 不标最终风险等级           |
| 复核人员  | 抽查、双人复核、仲裁冲突、确认事件级和风险级标签         | 不跳过证据直接给高风险        |
| 工程人员  | 将 CVAT 导出转换为统一 JSONL，做格式检查和一致性统计 | 不把原始 CVAT 导出直接送进训练 |

## 3. 工具选择

主工具使用 **CVAT Online 云端版**。

适用范围：

| 数据类型                                          | 是否可上传云端 | 说明                     |
| --------------------------------------------- | ------- | ---------------------- |
| LE2I/IMViA、UR Fall、Fall Detection 2017 等公开数据集 | 可以      | 第一阶段优先使用云端标注           |
| TOAGA 公开视频                                    | 可以      | 只标可见步行动作，不标跌倒事件        |
| 自采老人视频                                        | 默认不上传   | 必须先完成授权、脱敏、访问控制和删除机制确认 |
| 含人脸、家庭环境、身份信息的未脱敏视频                           | 不上传     | 不进入本 SOP 的云端标注流程       |

## 4. 注册并进入云端标注工具

### 4.1 使用前检查

每台标注电脑至少满足：

| 项目  | 要求                          |
| --- | --------------------------- |
| 系统  | macOS、Windows 10/11 或 Linux |
| 浏览器 | Chrome 或 Edge               |
| 网络  | 能稳定访问 CVAT Online 并上传视频     |
| 屏幕  | 建议 13 英寸以上，最好外接显示器          |

标注员无需安装 Docker、Git 或任何本地服务。

### 4.2 注册 CVAT Online 账号

1. 打开 CVAT 官网：<https://www.cvat.ai/>
2. 点击进入 CVAT Online 或直接访问：

```text
https://app.cvat.ai/
```

3. 使用项目统一邮箱注册账号。
4. 邮箱验证完成后登录。
5. 登录后确认能看到 `Projects`、`Tasks` 等页面。

账号命名建议：

```text
labeler_fall_01
labeler_fall_02
reviewer_fall_01
admin_fall_risk
```

### 4.3 项目权限

数据管理员负责创建项目并邀请成员。

| 角色    | CVAT 权限建议           | 说明                   |
| ----- | ------------------- | -------------------- |
| 数据管理员 | Owner/Admin         | 创建项目、标签、任务和导出        |
| 复核人员  | Maintainer/Reviewer | 复核、记录决定和导出源标注；不能仅凭 CVAT 状态提升正式资格 |
| 标注员   | Worker/Annotator    | 只负责分配给自己的 task       |

标注员不要自行新建标签、删除 task 或修改项目配置。发现标签缺失时，在工作群或 issue 中反馈给数据管理员。

### 4.4 上传前合规检查

上传到 CVAT Online 前，数据管理员必须确认：

- 数据属于公开数据集，或已经完成授权和脱敏。
- 文件名不包含真实姓名、手机号、住址、身份证号等个人信息。
- 视频中没有不应公开的敏感画面；若有，先不上传。
- 上传清单已经记录到 manifest 或批次表。

第一阶段只上传公开数据集和经过确认可上传的教学样本。

## 5. 标注前数据准备

### 5.1 第一批标注顺序

第一批不要从 TOAGA、GSTRIDE 或 Fall Detection 2017 开始。按以下顺序执行：

| 顺序  | 数据                                       | 用途                  |
| ---:| ---------------------------------------- | ------------------- |
| 1   | LE2I/IMViA `Home_01/video (1).avi`       | 培训示范，先人工标，再对照官方 txt |
| 2   | LE2I/IMViA `Home_01/video (2)-(6).avi`   | 标注员练习               |
| 3   | LE2I/IMViA `Home_01` 全部视频                | 第一批正式动作级标注          |
| 4   | LE2I/IMViA `Home_02`、`Coffee_room_01/02` | 扩大事件评估              |
| 5   | 已授权且已脱敏的自采视频                             | 完整动作、事件、风险三级标注      |

当前教学样本路径：

```text
data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi
```

官方 txt 路径：

```text
data/external/le2i_imvia/raw/FallDataset/Home_01/Annotation_files/video (1).txt
```

培训时先不要让标注员看官方 txt。先独立标注，再由复核人员对照 txt 讲解跌倒开始和结束边界。

### 5.2 视频 ID 命名

统一使用无空格、可追踪的 `video_id`。

示例：

```text
le2i_home_01_video_1
le2i_home_01_video_2
self_livingroom_p01_B03_side_t01
```

CVAT 任务名使用：

```text
fall_risk__{dataset}__{subset}__{video_id}
```

示例：

```text
fall_risk__le2i_imvia__home_01__le2i_home_01_video_1
```

### 5.3 原始视频和标注导出目录

原始视频不要改名、不要覆盖。统一标注结果落到：

```text
data/annotations/fall_risk/
```

CVAT 原始导出按批次归档到不可变来源目录：

```text
data/annotations/fall_risk/cvat_exports/raw/<batch-id>/
```

转换结果先落到来源专属候选目录：

```text
data/annotations/fall_risk/generated/v1/cvat_<export-id>/
data/annotations/fall_risk/generated/v1/le2i_official/
```

候选目录按源导出批次命名，不按 subset 自动拆分。现有 `cvat_coffee_01_02/` 实际同时包含 `Coffee_room_01`、`Coffee_room_02` 和 `Home_02`；必须按每条记录关联的 manifest subset 统计，不能从目录名推断数据归属。

通过人工确认、双人独立复核和正式校验后，发布版本才使用根目录统一契约：

```text
data/annotations/fall_risk/action_labels.jsonl
data/annotations/fall_risk/event_labels.jsonl
data/annotations/fall_risk/risk_labels.jsonl
data/annotations/fall_risk/subject_profiles.json
data/annotations/fall_risk/annotation_review_log.jsonl
```

自动生成候选固定为 `pending` 或 `auto_imported`、`eligibility=false`、`review_evidence_ids=[]`。不要让训练脚本直接读取 CVAT XML、ZIP、人工 Excel 或未经发布的候选，也不得让转换脚本直接覆盖根目录标签。

### 5.4 AVI 无法上传时的处理

如果 CVAT 或浏览器无法正常预览 `.avi`，可以生成一个仅用于标注的 `.mp4` 副本。不要删除原始 AVI。

示例：

```bash
mkdir -p data/annotations/fall_risk/annotation_videos/le2i_home_01
ffmpeg -i "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  -c:v libx264 -pix_fmt yuv420p -an \
  "data/annotations/fall_risk/annotation_videos/le2i_home_01/le2i_home_01_video_1.mp4"
```

转换后在 manifest 中保留原始路径和标注用路径，不要把标注用 MP4 当成新的数据来源。

## 6. 创建 CVAT Online 项目

### 6.1 新建 Project

在 CVAT Online 首页：

1. 点击 `Projects`。
2. 点击 `+` 创建项目。
3. 项目名填写：

```text
fall_risk_action_annotation_v1
```

4. 描述填写：

```text
跌倒风险动作级标注。标注员只标视频可见动作事实，不标医学诊断和最终风险等级。
```

### 6.2 标签配置

在 CVAT 中每个动作标签建成一个 rectangle label。标注时用 rectangle track 承载动作片段。

| 标签    | 英文名                        | 中文名       |
| ----- | -------------------------- | --------- |
| `A01` | `normal_walk`              | 正常行走      |
| `A02` | `normal_turn`              | 正常转身      |
| `A03` | `normal_sit`               | 正常坐下      |
| `A04` | `normal_stand`             | 正常起身      |
| `B01` | `slow_walk`                | 缓慢行走      |
| `B02` | `dragging_walk`            | 拖步        |
| `B03` | `shuffling_walk`           | 小碎步       |
| `B04` | `swaying_walk`             | 行走摇晃      |
| `B05` | `unstable_turn`            | 转身不稳      |
| `B06` | `slow_sit_to_stand`        | 起身缓慢      |
| `C01` | `failed_sit_to_stand`      | 起身失败      |
| `C02` | `wall_support_walk`        | 扶墙/扶物行走   |
| `C03` | `stumble_recovery`         | 踉跄后恢复     |
| `C04` | `rapid_support_contact`    | 快速扶物      |
| `C05` | `rapid_body_drop_recovery` | 身体快速下沉后恢复 |
| `D01` | `forward_fall`             | 向前跌倒      |
| `D02` | `lateral_fall`             | 侧向跌倒      |
| `D03` | `backward_fall`            | 向后跌倒      |
| `D04` | `long_static_after_fall`   | 跌倒后静止     |
| `U01` | `unable_to_judge`          | 无法判断      |

建议标签显示名使用：

```text
A01_normal_walk
B05_unstable_turn
C03_stumble_recovery
D01_forward_fall
U01_unable_to_judge
```

这样导出后容易自动解析 `action_id`。

### 6.3 标签属性

每个标签都建议配置以下属性：

| 属性               | 类型  | 可选值或填写规则                                                                                        |
| ---------------- | --- | ----------------------------------------------------------------------------------------------- |
| `quality`        | 单选  | `clear`、`partial_occlusion`、`heavy_occlusion`、`low_light`、`off_screen`、`multi_person_uncertain` |
| `target_subject` | 文本  | 不知道填 `unknown`；自采视频填 `p01`、`p02` 等                                                              |
| `note`           | 文本  | 必要时填写原因，例如“遮挡严重”“疑似扶墙但手部不可见”                                                                    |

`quality` 不替代动作标签。画面质量差但仍能判断动作时，动作标签照常标；无法判断时使用 `U01`。

## 7. 创建 CVAT Online Task

每个视频建一个 task，不要把多个视频混进同一个 task。

操作步骤：

1. 进入项目 `fall_risk_action_annotation_v1`。
2. 点击 `+` 创建 task。
3. 填写 task 名：

```text
fall_risk__le2i_imvia__home_01__le2i_home_01_video_1
```

4. 上传视频文件。
5. 选择当前项目的标签集。
6. 保存 task。
7. 打开 task，检查视频能否正常播放、逐帧前进和回退。

任务备注中填写：

```text
dataset=le2i_imvia
subset=Home_01
video_id=le2i_home_01_video_1
scene_region=home
view=fixed_camera
label_source=manual_action
```

## 8. 标注操作流程

### 8.1 每条视频的固定流程

每条视频必须按以下顺序：

1. 完整播放一遍，不标注，只理解动作过程。
2. 第二遍记录主要动作切换点。
3. 第三遍开始创建 action track。
4. 标完后从头回看一次，检查时间边界、漏标和标签混淆。
5. 提交给复核人员。

不要边第一次看边标，容易漏掉前后动作关系。

### 8.2 用 rectangle track 标动作片段

CVAT 中的每个动作片段用一个 rectangle track 表示。

操作方法：

1. 跳到动作开始帧。
2. 选择对应标签，例如 `B05_unstable_turn`。
3. 使用 rectangle track 框住目标老人可见身体区域。
4. 沿时间轴播放或逐帧移动，保持 track 覆盖同一个目标老人。
5. 到动作结束帧后，将该 track 结束或设置 outside。
6. 动作切换时，新建下一个 action track。

矩形框要求：

- 框住目标老人主要可见身体区域。
- 不要求像人体检测框一样逐帧精确，但不能框到其他人。
- 人物离开画面时不要硬延长 track。
- 多人场景只标目标老人；无法确认目标时标 `U01` 并写备注。

### 8.3 时间边界规则

| 动作    | 开始                  | 结束                  |
| ----- | ------------------- | ------------------- |
| 行走    | 第一步明显开始移动           | 停止行走或切换到转身、坐下、跌倒等动作 |
| 转身    | 身体或脚步开始改变朝向         | 朝向稳定且不再继续转动         |
| 坐下    | 身体明显开始向下坐           | 坐稳，身体不再继续下降         |
| 起身    | 身体离开座位或开始上升         | 站稳，身体不再明显晃动         |
| 近跌倒   | 失衡、急停、快速下沉或突然扶物开始   | 恢复稳定，或转为真正跌倒        |
| 跌倒    | 身体开始失去支撑并倒向地面、床边或椅旁 | 身体接触并稳定在倒地或倒卧状态     |
| 跌倒后静止 | 跌倒动作结束后开始静止         | 明显起身、移动或视频结束        |

时间边界先精确到 `0.1s` 或相邻几帧。复核时允许小于 `0.5s` 的边界偏差由复核人员统一修正。

### 8.4 标签选择规则

优先选择一个主要动作标签。

常见判断：

| 情况            | 标注            |
| ------------- | ------------- |
| 稳定连续走路        | `A01`         |
| 明显慢但稳定        | `B01`         |
| 脚抬不起来、拖着走     | `B02`         |
| 步幅很小、密集挪动     | `B03`         |
| 行走中左右晃但未差点摔倒  | `B04`         |
| 转身明显晃动或停顿     | `B05`         |
| 起身慢但最终一次成功    | `B06`         |
| 多次尝试起身失败或未成功  | `C01`         |
| 持续扶墙、扶桌、扶家具移动 | `C02`         |
| 踉跄、脚步错乱但最后站住  | `C03`         |
| 快失衡时突然扶物恢复    | `C04`         |
| 身体突然快速下沉但恢复   | `C05`         |
| 明确失去支撑并倒下     | `D01/D02/D03` |
| 跌倒后持续不动       | `D04`         |
| 看不清或无法确认      | `U01`         |

不要把“走得慢”直接标成高风险。标注员只标动作事实，最终风险由复核和模型融合决定。

### 8.5 质量属性填写

| 画面情况            | `quality`                |
| --------------- | ------------------------ |
| 人体主体清楚，动作边界明确   | `clear`                  |
| 身体局部遮挡，但主要动作可判断 | `partial_occlusion`      |
| 关键动作被挡住，通常无法判断  | `heavy_occlusion`        |
| 光线过暗，动作判断困难     | `low_light`              |
| 人物部分或完全离开画面     | `off_screen`             |
| 多人场景无法确认目标老人    | `multi_person_uncertain` |

`U01` 必须写 note。例如：

```text
人物被桌子遮挡，无法确认是否倒地
多人交叉，无法确认目标老人
光线不足，无法判断是否扶墙
```

## 9. 复核流程

### 9.1 复核比例

| 阶段               | 复核要求                             |
| ---------------- | -------------------------------- |
| 标注员培训前 10 条      | 100% 复核，逐条讲解                     |
| 正式批量前 10%        | 双人标注，统计一致率                       |
| 跌倒、近跌倒、起身失败      | 全量复核                             |
| `U01` 超过 20% 的视频 | 必须复核                             |
| 冲突样本             | 写入 `annotation_review_log.jsonl` |
| 拟发布为 `eligibility=true` 的标签 | 不同 reviewer 双人独立复核 |

### 9.2 复核检查项

复核人员逐条检查：

- 是否漏标明显动作片段。
- 标签是否来自标签字典。
- 起止时间是否覆盖完整动作。
- `C03/C04/C05` 是否被误标成普通慢走。
- `D01/D02/D03` 是否确实失去支撑并倒下。
- `D04` 是否只覆盖跌倒后静止，不包含跌倒过程本身。
- `U01` 是否有明确原因。
- 多人场景是否目标一致。

### 9.3 冲突仲裁

冲突处理优先级：

1. 时间边界偏差小于 `0.5s`：复核人员统一修正。
2. 标签类别冲突：回看视频后仲裁，例如 `C03` vs `D01`。
3. 正常动作 vs `U01`：以可观察证据为准。
4. 仍无法判断：保留 `U01` 或事件级 `uncertain`，不要强行判定。

所有批准、冲突和仲裁记录写入：

```text
data/annotations/fall_risk/annotation_review_log.jsonl
```

每条 review 必填：

```text
review_id                 label_id
label_type                reviewer_id
decision                  reviewed_at
reason_code               note
```

可选链路字段为 `previous_record_sha256`、`result_record_sha256` 和 `supersedes_review_id`。`approve/adjudicate` 必须写 `result_record_sha256`，并与最终完整标签的规范化 JSON SHA-256 一致。标签内的 `review_evidence_ids` 必须精确列出当前有效的批准/仲裁 review ID。

正式标签至少需要两个不同 `reviewer_id` 的有效决定；同一人提交两个 review ID 不算双人复核。出现冲突时按以下链路处理：

1. 冲突 review 用 `supersedes_review_id` 指向前一 review，并让 `previous_record_sha256` 等于前一 review 的 `result_record_sha256`。
2. `decision=conflict` 不能作为正式证据，必须由直接后继的 `decision=adjudicate` 解决。
3. 仲裁人的 `reviewer_id` 必须不同于整条前置链的所有 reviewer，`reviewed_at` 必须更晚。
4. 仲裁完成后还需另一名独立 reviewer 对同一最终记录作有效 `approve`，才能达到双人门槛。
5. 标签任一字段改变都会改变 `result_record_sha256`；变更后必须重新复核，不能只改 review log。

`risk_labels.jsonl` 与 `annotation_review_log.jsonl` 没有真实人工结果时保持空 JSONL，不放示例行。完整 schema 见 `跌倒风险标签字典.md`。

## 10. 导出 CVAT 标注

### 10.1 导出时机

每个 task 完成以下状态后才能导出：

- 标注员自检完成。
- 复核人员确认通过。
- 所有 `U01` 都有 note。
- 跌倒、近跌倒、起身失败片段已全量复核。

### 10.2 导出格式

在 CVAT task 页面：

1. 点击 `Actions`。
2. 选择 `Export task dataset` 或 `Export annotations`。
3. 格式优先选择 CVAT 原生格式。
4. 不需要导出图片帧，除非复核人员要求留存证据图。

导出文件命名：

```text
cvat_export__{task_name}__{export_id}__v{YYYYMMDD}.zip
```

示例：

```text
cvat_export__fall_risk__le2i_imvia__home_01__le2i_home_01_video_1__exp01__v20260715.zip
```

保存位置：

```text
data/annotations/fall_risk/cvat_exports/raw/<batch-id>/
```

文件名或 CVAT 页面状态只描述源导出版本，不代表统一标签已经获得 `reviewed/final` 资格。原始 ZIP/XML 归档后视为不可变输入；不得覆盖或就地清理，发现身份元数据时只记录脱敏风险并按数据治理流程处理。

### 10.3 云端数据留存

每批标注导出后，数据管理员需要记录云端 task 状态：

| 状态                     | 什么时候用            | 处理方式                               |
| ---------------------- | ---------------- | ---------------------------------- |
| `keep_online`          | 公开数据集，后续还要复核     | 保留 task，并记录导出版本                    |
| `archive_after_export` | 批次已完成，短期不再修改     | 下载导出文件并归档，本地 JSONL 通过检查后再归档云端任务    |
| `delete_after_export`  | 已授权但敏感度较高的脱敏自采数据 | 导出、复核和 JSONL 检查完成后，由数据管理员删除云端 task |

任何含敏感信息的视频不得因为“标注方便”长期留在云端。删除前必须确认 CVAT 导出 ZIP、统一 JSONL 和复核记录均已归档。

## 11. 构建、转换与校验

工程人员必须按“manifest -> 来源候选 -> 审计 -> 人工复核/仲裁 -> 正式校验 -> 发布”的顺序处理。转换成功不等于可以进入训练。

### 11.1 构建统一 manifest

先确认 editable 安装指向当前仓库，再构建 manifest：

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
conda run -n eldercare-ai python scripts/annotation/build_fall_risk_manifest.py \
  --repo-root . \
  --output data/manifests/fall_risk_video_manifest.jsonl \
  --ffprobe-bin ffprobe
```

manifest 构建器读取逐视频真实 `fps_num/fps_den`、帧数、时长和分辨率，同时记录资产哈希、来源、许可、人员/组和资格状态。默认拒绝覆盖；重建正式 manifest 前先做版本决策。

### 11.2 转换 CVAT 候选

每个源导出使用独立输出目录：

```bash
conda run -n eldercare-ai python scripts/annotation/convert_cvat_fall_labels.py \
  --input data/annotations/fall_risk/cvat_exports/raw/le2i_home_01_first_2_videos_cvat.zip \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --output-dir data/annotations/fall_risk/generated/v1/cvat_home_01 \
  --labeler labeler_fall_01
```

输出是 `action_labels.jsonl` 和从动作确定性映射的 `event_labels.jsonl`。转换器按 `video_id` 读取 manifest 的逐视频有理 FPS；正式批处理不得使用全局 `--fps` 或 `--file-root`。这两个参数只在同时显式给出 `--development-override` 时用于旧测试夹具。

候选固定为 `review_status=pending`、`eligibility=false`、`review_evidence_ids=[]`。脚本默认拒绝覆盖输出，也拒绝 `--review-status reviewed/final`。不得把 `--action-output/--event-output` 指向根目录文件。

当前 `generated/v1/cvat_coffee_01_02/` 在 100 条视频上各含 514 条动作候选和映射事件，按记录计为 `Coffee_room_01=233`、`Coffee_room_02=150`、`Home_02=131`，且各视频 FPS 分别来自 manifest。该目录按源导出命名，不得整体标成 Coffee。

### 11.3 导入 LE2I 官方窗口候选

官方 TXT 与人工 CVAT 边界独立保存：

```bash
conda run -n eldercare-ai python scripts/annotation/import_le2i_fall_labels.py \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --event-output data/annotations/fall_risk/generated/v1/le2i_official/event_labels.jsonl \
  --report-output data/annotations/fall_risk/generated/v1/le2i_official/import_report.json
```

导入器只生成 TXT 明确支持的 `event_type=fall`，并保留 `label_source=le2i_txt`、1-based 源帧与 0-based 统一帧。`0/0` 只计入报告的显式无跌倒窗口；`Lecture room/Office` 无 TXT，不生成官方事件。官方候选固定为 `auto_imported/false/[]`，不能替代人工动作标注。

### 11.4 审计候选与正式标签

对 CVAT 来源候选执行审计：

```bash
conda run -n eldercare-ai python scripts/annotation/validate_fall_risk_labels.py \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --action-labels data/annotations/fall_risk/generated/v1/cvat_home_01/action_labels.jsonl \
  --event-labels data/annotations/fall_risk/generated/v1/cvat_home_01/event_labels.jsonl \
  --risk-labels data/annotations/fall_risk/risk_labels.jsonl \
  --subject-profiles data/annotations/fall_risk/subject_profiles.json \
  --review-log data/annotations/fall_risk/annotation_review_log.jsonl \
  --config configs/data/fall_risk_label_validation_v1.yaml \
  --mode audit \
  --report-output reports/fall_risk/cvat_le2i_home_01_candidate_validation.json
```

`audit` 会报告缺少人工复核等 blocker，但结构、边界、来源哈希或 manifest 关联错误仍会失败。报告默认拒绝覆盖，复跑使用新的版本化文件名。

完成真实人工复核和受控发布后，对根目录统一标签执行：

```bash
conda run -n eldercare-ai python scripts/annotation/validate_fall_risk_labels.py \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --action-labels data/annotations/fall_risk/action_labels.jsonl \
  --event-labels data/annotations/fall_risk/event_labels.jsonl \
  --risk-labels data/annotations/fall_risk/risk_labels.jsonl \
  --subject-profiles data/annotations/fall_risk/subject_profiles.json \
  --review-log data/annotations/fall_risk/annotation_review_log.jsonl \
  --config configs/data/fall_risk_label_validation_v1.yaml \
  --mode formal \
  --report-output reports/fall_risk/annotation_validation_formal_v1.json
```

`formal` 将 `pending/uncertain`、未知许可、已有人员画像缺少同意引用、缺少独立复核、来源不完整或 manifest 不合格视为阻塞。只有 formal 报告为 `valid=true` 的冻结发布版本才具备正式评估资格。

action/event/risk/review/profile 的完整字段契约见 `跌倒风险标签字典.md`。其中人工 `risk_labels.jsonl` 禁止 `risk_score`；没有真实风险标签和 review 时两个 JSONL 保持零条记录，`subject_profiles.json` 保持 `fall-risk-subject-profiles-v1` 的空 `subjects` 模板。

## 12. 质检清单

导出前标注员自检：

- 每个动作片段都有标签。
- 标签名称没有拼写错误。
- `U01` 都有 note。
- 动作切换处没有明显漏标。
- 近跌倒和跌倒没有混淆。
- 跌倒后静止 `D04` 没有覆盖跌倒过程。

复核人员质检：

- 前 10 条训练样本 100% 复核。
- `C01-C05` 和 `D01-D04` 全量复核。
- `U01` 比例超过 20% 的视频必须回看。
- 标注边界偏差大于 `0.5s` 的片段必须修正。
- 标签冲突必须写入 `annotation_review_log.jsonl`。
- 正式标签至少有两个不同 reviewer 的有效批准/仲裁，且 review 结果哈希与最终记录一致。
- 冲突仲裁人独立于前置链所有 reviewer，冲突链的前后哈希和时间顺序完整。

工程人员质检：

- JSONL 每行都是合法 JSON。
- `start_time <= end_time`，且动作/事件帧号、时间和 manifest 有理 FPS 一致。
- `action_id` 必须在标签字典中。
- `asset_id` 必须能在 manifest 中找到；视频标签的 `video_id` 还必须匹配同一资产。
- 来源文件存在且 `source_annotation_sha256` 与实际文件一致。
- 人工风险标签没有 `risk_score`，空模板没有示例或伪造记录。
- 训练、验证、测试划分按人员或样本组，不按窗口随机切分。
- 不把 CVAT XML、官方 txt 或 Excel 直接作为训练输入。

## 13. 常见问题

### 13.1 CVAT Online 无法打开或无法登录

优先检查：

- 浏览器是否为 Chrome 或 Edge。
- 网络是否能访问 `https://app.cvat.ai/`。
- 账号是否已完成邮箱验证。
- 是否使用了项目管理员邀请的账号登录。
- 浏览器是否拦截了第三方登录、弹窗或必要 cookie。

如果仍无法登录，标注员不要重新创建私人项目，应把脱敏截图、账号内部代号和报错时间发给数据管理员处理，不要把邮箱写入公共日志或文档。

### 13.2 视频上传后无法播放

优先检查：

- 文件是否损坏。
- 浏览器是否为 Chrome 或 Edge。
- 是否为 AVI 编码兼容问题。

如果是 AVI 兼容问题，按第 5.4 节生成 MP4 标注副本。

### 13.3 不知道该标慢走还是拖步

判断原则：

- 只是速度慢但脚能正常抬起：`B01`。
- 脚明显抬不起来、脚尖或脚底拖着地面：`B02`。
- 步幅很小、密集挪动：`B03`。

仍不确定时，先标最保守的可见事实，并写 note，交复核人员判断。

### 13.4 近跌倒和跌倒分不清

判断原则：

- 最后站住了，没有失去支撑接触地面：`C03/C04/C05`。
- 明确失去支撑并倒下：`D01/D02/D03`。
- 倒地后持续不动：跌倒动作结束后另标 `D04`。

### 13.5 有官方 txt，是否还要人工标

要。LE2I 官方 txt 主要提供跌倒窗口和人体框，不能替代动作级人工标注。

正确做法：

1. 标注员先独立标动作片段。
2. 复核人员用官方 txt 对照跌倒窗口。
3. 工程人员把官方 TXT 导入 `generated/v1/le2i_official/event_labels.jsonl`，并保留 `label_source=le2i_txt`。
4. 官方窗口与人工边界独立复核，不能互相覆盖或自动提升。

## 14. 每日交付物

每个标注日结束前，标注员提交：

```text
完成 task 列表
有疑问的视频和时间点
U01 比例较高的视频
导出的 CVAT ZIP 文件路径
```

复核人员提交：

```text
复核通过 task 列表
冲突样本列表
annotation_review_log.jsonl 新增记录
需要返工的视频和原因
```

工程人员提交：

```text
来源专属 generated/v1 候选路径
manifest 与源导出 SHA-256
audit 报告路径和 blocker 摘要
标签分布与质量统计
```

## 15. 完成标准

一批标注数据只有同时满足以下条件，才算完成：

- CVAT 原始导出已归档。
- 来源专属动作/事件候选已生成且没有覆盖根目录文件。
- 高风险动作和跌倒事件已由不同人员双人复核，结果哈希绑定最终记录。
- 冲突样本已由独立仲裁人解决并形成完整 review 链。
- 严格 schema、来源、manifest 和边界审计通过。
- 标签分布和 `U01` 比例已统计。
- 数据划分不泄漏同一人员或同一样本组。
- 只有受控发布且 `formal` 报告 `valid=true` 时，才可声明具备正式评估资格。

## 16. 外部工具参考

- CVAT 官网：<https://www.cvat.ai/>
- CVAT Online：<https://app.cvat.ai/>
- CVAT 用户手册：<https://docs.cvat.ai/docs/manual/>
