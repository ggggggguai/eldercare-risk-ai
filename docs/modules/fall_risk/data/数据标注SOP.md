# 跌倒风险数据标注 SOP

更新时间：2026-07-22


## 1. 总原则

- 标注员只记录视频中能看见的事实，不做医学诊断，不判断老人真实健康状态。
- 不临时发明标签。所有标签必须来自 `跌倒风险标签字典.md`。
- 看不清、遮挡严重、人物离开画面或目标老人无法确认时，使用 `U01`，不要猜。
- 公开视频、官方 txt、人工标注、表格 proxy 和自动算法结果必须区分来源，不能混成同一种真值。
- CVAT 中的矩形框主要用于承载“动作片段时间段 + 目标对象身份”，不是本项目最终的人体检测真值。
- v2 CVAT/root 标签记录来源事实；模型训练使用独立 v3 标签，禁止把未标注背景直接当负样本。
- v3 中 C03-C05 先标可见失衡动作，是否形成 near-fall 必须再确认恢复且未跌倒。

## 2. 角色分工

| 角色    | 主要职责                             | 不做什么               |
| ----- | -------------------------------- | ------------------ |
| 数据管理员 | 准备视频、manifest、CVAT 项目和任务，管理导出文件  | 不擅自修改标签字典          |
| 标注员   | 按 SOP 标动作片段、质量标记和备注              | 不标最终风险等级           |
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
admin_fall_risk
```

### 4.3 项目权限

数据管理员负责创建项目并邀请成员。

| 角色    | CVAT 权限建议           | 说明                   |
| ----- | ------------------- | -------------------- |
| 数据管理员 | Owner/Admin         | 创建项目、标签、任务和导出        |
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
data/annotations/fall_risk/generated/v2/cvat_<export-id>/
data/annotations/fall_risk/generated/v2/le2i_official/
```

候选目录按源导出批次命名，不按 subset 自动拆分。现有 `cvat_coffee_01_02/` 实际同时包含 `Coffee_room_01`、`Coffee_room_02` 和 `Home_02`；必须按每条记录关联的 manifest subset 统计，不能从目录名推断数据归属。


```text
data/annotations/fall_risk/action_labels.jsonl
data/annotations/fall_risk/event_labels.jsonl
data/annotations/fall_risk/risk_labels.jsonl
data/annotations/fall_risk/subject_profiles.json
```


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
fall_risk_action_annotation_v2
```

4. 描述填写：

```text
跌倒风险动作级标注。标注员只标视频可见动作事实，不标医学诊断和最终风险等级。
```

### 6.2 标签配置

在 CVAT 中每个动作标签建成一个 rectangle label。标注时用 rectangle track 承载动作片段。

标签配置文件为 `configs/data/fall_risk_cvat_labels_v2.json`，共 29 个动作标签。

| 标签    | 英文名                        | 中文名       |
| ----- | -------------------------- | --------- |
| `A01` | `normal_walk`              | 正常行走      |
| `A02` | `normal_turn`              | 正常转身      |
| `A03` | `controlled_sit_down`      | 正常坐下      |
| `A04` | `normal_sit_to_stand`      | 正常起身      |
| `A05` | `controlled_squat`         | 正常下蹲      |
| `A06` | `controlled_bend`          | 正常俯身/弯腰  |
| `A07` | `controlled_lie_down`      | 主动可控躺下  |
| `A08` | `routine_support_contact`  | 日常扶椅/扶物  |
| `A09` | `kneel_or_floor_activity`  | 正常跪下/地面活动 |
| `A10` | `normal_step_adjustment`   | 正常调步/绕障  |
| `A11` | `assisted_sit_or_lowering` | 被协助坐下或降低身体 |
| `A12` | `normal_hop`               | 正常单脚跳/跳跃 |
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
| `D05` | `seated_fall`              | 从坐姿发生跌倒   |
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

`A07-A11` 已进入通用 v2 CVAT 配置，并在 v3 中保留为具体困难负样本类型。不得把主动躺下、日常扶物、跪地、正常调步或被协助降低身体合并成无具体类别的背景片段。

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

1. 进入项目 `fall_risk_action_annotation_v2`。
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
| 下蹲    | 身体开始可控屈膝并向下移动         | 蹲稳或恢复站立，且不再继续下沉         |
| 俯身/弯腰 | 躯干或髋部开始可控地向前屈曲         | 恢复直立，或达到稳定俯身姿态且不再继续下沉         |
| 近跌倒   | 失衡、急停、快速下沉或突然扶物开始   | 恢复稳定，或转为真正跌倒        |
| 跌倒    | 身体开始失去支撑并倒向地面、床边或椅旁 | 身体接触并稳定在倒地或倒卧状态     |
| 跌倒后静止 | 跌倒动作结束后开始静止         | 明显起身、移动或视频结束        |


### 8.4 标签选择规则

优先选择一个主要动作标签。

常见判断：

| 情况            | 标注            |
| ------------- | ------------- |
| 稳定连续走路        | `A01`         |
| 正常可控下蹲        | `A05`         |
| 正常可控俯身/弯腰    | `A06`         |
| 主动可控躺下        | `A07`         |
| 日常扶椅或正常扶物    | `A08`         |
| 跪地或地面活动       | `A09`         |
| 正常调步或绕障       | `A10`         |
| 被协助坐下或降低身体   | `A11`         |
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

在 v3 中，C03/C04/C05 的动作 track 只回答发生了什么失衡机制。工程人员只有在视频中能看到恢复稳定且未形成 fall 时，才创建 near-fall positive；否则写 ignore 或保留在待复核清单。D04 只表示 `post_fall_immobile`，视频结束不能自动证明达到长静止阈值。


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



| ---------------- | -------------------------------- |
| 正式批量前 10%        | 双人标注，统计一致率                       |



- 是否漏标明显动作片段。
- 标签是否来自标签字典。
- 起止时间是否覆盖完整动作。
- `C03/C04/C05` 是否被误标成普通慢走。
- `D01/D02/D03` 是否确实失去支撑并倒下。
- `D04` 是否只覆盖跌倒后静止，不包含跌倒过程本身。
- `U01` 是否有明确原因。
- 多人场景是否目标一致。


冲突处理优先级：

3. 正常动作 vs `U01`：以可观察证据为准。
4. 仍无法判断：保留 `U01` 或事件级 `uncertain`，不要强行判定。


```text
```


```text
reason_code               note
```





## 10. 导出 CVAT 标注

### 10.1 导出时机

每个 task 完成以下状态后才能导出：

- 标注员自检完成。
- 所有 `U01` 都有 note。

### 10.2 导出格式

在 CVAT task 页面：

1. 点击 `Actions`。
2. 选择 `Export task dataset` 或 `Export annotations`。
3. 格式优先选择 CVAT 原生格式。

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


### 10.3 云端数据留存

每批标注导出后，数据管理员需要记录云端 task 状态：

| 状态                     | 什么时候用            | 处理方式                               |
| ---------------------- | ---------------- | ---------------------------------- |
| `archive_after_export` | 批次已完成，短期不再修改     | 下载导出文件并归档，本地 JSONL 通过检查后再归档云端任务    |


## 11. 构建、转换与校验


### 11.1 构建统一 manifest

先确认 editable 安装指向当前仓库，再构建 manifest：

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
conda run -n eldercare-ai python scripts/annotation/build_fall_risk_manifest.py \
  --repo-root . \
  --output data/manifests/fall_risk_video_manifest.jsonl \
  --ffprobe-bin ffprobe
```


### 11.2 转换 CVAT 候选

每个源导出使用独立输出目录：

```bash
conda run -n eldercare-ai python scripts/annotation/convert_cvat_fall_labels.py \
  --input data/annotations/fall_risk/cvat_exports/raw/le2i_home_01_first_2_videos_cvat.zip \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --output-dir data/annotations/fall_risk/generated/v2/cvat_home_01 \
  --labeler labeler_fall_01
```

输出是 `action_labels.jsonl` 和从动作确定性映射的 `event_labels.jsonl`。转换器按 `video_id` 读取 manifest 的逐视频有理 FPS；正式批处理不得使用全局 `--fps` 或 `--file-root`。这两个参数只在同时显式给出 `--development-override` 时用于旧测试夹具。


当前 `generated/v2/cvat_coffee_01_02/` 在 100 条视频上各含 514 条动作候选和映射事件，按记录计为 `Coffee_room_01=233`、`Coffee_room_02=150`、`Home_02=131`，且各视频 FPS 分别来自 manifest。该目录按源导出命名，不得整体标成 Coffee。

当前 `generated/v2/cvat_lecture_room/` 含 250 条动作候选和 250 条映射事件，覆盖 `Lecture room` 的 27 条视频。同一外部 ZIP 还包含 33 个空的 `Office` 任务元数据；这些空任务没有生成标签。

该批次使用脱敏来源 `data/annotations/fall_risk/cvat_exports/raw/le2i_lecture_room_cvat_redacted.zip`。原始外部 ZIP 的 SHA-256 与脱敏 ZIP 的 SHA-256 必须记录在 `reports/fall_risk/data_audit.md`，转换标签只绑定脱敏来源 hash。

当前 `generated/v2/cvat_office/` 另含 158 条动作候选和 158 条映射事件，覆盖 `Office` 的 33 条视频。该批次来自独立 Office 导出，使用脱敏来源 `data/annotations/fall_risk/cvat_exports/raw/le2i_office_cvat_redacted.zip`；脱敏时同时把旧格式 task 名规范化为 manifest 约定格式，来源哈希记录在数据审计报告中。

### 11.3 导入 LE2I 官方窗口候选

官方 TXT 与人工 CVAT 边界独立保存：

```bash
conda run -n eldercare-ai python scripts/annotation/import_le2i_fall_labels.py \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --event-output data/annotations/fall_risk/generated/v2/le2i_official/event_labels.jsonl \
  --report-output data/annotations/fall_risk/generated/v2/le2i_official/import_report.json
```

导入器只生成 TXT 明确支持的 `event_type=fall`，并保留 `label_source=le2i_txt`、1-based 源帧与 0-based 统一帧。`0/0` 只计入报告的显式无跌倒窗口；`Lecture room/Office` 无 TXT，不生成官方事件。官方候选固定为 `auto_imported/false/[]`，不能替代人工动作标注。

### 11.4 审计候选与正式标签

对 CVAT 来源候选执行审计：

```bash
conda run -n eldercare-ai python scripts/annotation/validate_fall_risk_labels.py \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --action-labels data/annotations/fall_risk/generated/v2/cvat_home_01/action_labels.jsonl \
  --event-labels data/annotations/fall_risk/generated/v2/cvat_home_01/event_labels.jsonl \
  --risk-labels data/annotations/fall_risk/risk_labels.jsonl \
  --subject-profiles data/annotations/fall_risk/subject_profiles.json \
  --config configs/data/fall_risk_label_validation_v2.yaml \
  --mode audit \
  --report-output reports/fall_risk/cvat_le2i_home_01_candidate_validation.json
```


### 11.5 生成并校验模型训练标签 v3

从现行 v2 根标签生成独立 v3 文件：

```bash
conda run -n eldercare-ai python scripts/annotation/migrate_fall_labels_v2_to_v3.py
conda run -n eldercare-ai python scripts/annotation/build_fall_training_split_v3.py
conda run -n eldercare-ai python scripts/annotation/validate_fall_labels_v3.py
```

迁移器只自动完成字段、half-open 边界、来源合并、D04 父事件关联和 U01 ignore。以下内容必须人工完成：

- C03-C05 是否形成 near-fall positive。
- fall/near-fall hard-negative 窗口。
- near-fall recovery frame 和双人复核。
- 来源冲突、目标身份冲突和不确定边界。

迁移器会根据独立 `sample_group_id/source_group_id` 数量生成 `action_type_training_tier`：少于 10 个 sample group 为 ignore；10-29 个，或来源少于 3 个 source group，为 auxiliary；至少 30 个 sample group且至少 3 个 source group才为 primary。训练具体动作头时必须读取该字段，不能直接复用父类 `training_tier`。

当前 v3 统一 split 覆盖 1,853 条动作/事件标签和 426 个资产，按保守关系形成 31 个泄漏组，校验未发现跨 partition 泄漏；primary fall 正例按 train/validation/test 分为 74/14/7。Pre_VFallp 的未知人员视频共用保守源组，不得为改善分区分布而拆散。CaucaFall 的 100 个视频按 10 名受试者分组，同一受试者不得跨 partition。标签或 manifest 改变后必须依次重跑迁移、split 和校验；旧 v2 split 不得复用。

当前 v3 校验结构合法且 split 有效，但 primary `slow_walk` 在 test 分区没有样本，因此 `training_ready.action_type=false`。event negative=0、near-fall positive=0，两个事件任务也均为 `training_ready=false`；不得通过随机抽未标注背景、复制 C04 或拆散保守源组来消除提示。



```bash
conda run -n eldercare-ai python scripts/annotation/validate_fall_risk_labels.py \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --action-labels data/annotations/fall_risk/action_labels.jsonl \
  --event-labels data/annotations/fall_risk/event_labels.jsonl \
  --risk-labels data/annotations/fall_risk/risk_labels.jsonl \
  --subject-profiles data/annotations/fall_risk/subject_profiles.json \
  --config configs/data/fall_risk_label_validation_v2.yaml \
  --mode formal \
  --report-output reports/fall_risk/annotation_validation_formal_v2.json
```



## 12. 质检清单

导出前标注员自检：

- 每个动作片段都有标签。
- 标签名称没有拼写错误。
- `U01` 都有 note。
- 动作切换处没有明显漏标。
- 近跌倒和跌倒没有混淆。
- 跌倒后静止 `D04` 没有覆盖跌倒过程。


- `U01` 比例超过 20% 的视频必须回看。
- 标注边界偏差大于 `0.5s` 的片段必须修正。

工程人员质检：

- JSONL 每行都是合法 JSON。
- `start_time <= end_time`，且动作/事件帧号、时间和 manifest 有理 FPS 一致。
- `action_id` 必须在标签字典中。
- `asset_id` 必须能在 manifest 中找到；视频标签的 `video_id` 还必须匹配同一资产。
- 来源文件存在且 `source_annotation_sha256` 与实际文件一致。
- v3 action 的 `action_family/action_type` 与 action ID 一致。
- v3 action 的 `action_type_training_tier` 符合当前类别 sample/source group 门槛，且不高于父类 `training_tier`。
- v3 event 的 positive/negative/ignore 条件字段互斥。
- near-fall primary 具有 recovery frame 和双人复核。
- `primary/auxiliary/ignore` 在训练中不会被默认等权消费。
- `content_sha256/sample_group_id/physical_event_id` 不跨数据分区。
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


### 13.4 近跌倒和跌倒分不清

判断原则：

- 最后站住了，没有失去支撑接触地面：`C03/C04/C05`。
- 明确失去支撑并倒下：`D01/D02/D03`。
- 倒地后持续不动：跌倒动作结束后另标 `D04`。

### 13.5 有官方 txt，是否还要人工标

要。LE2I 官方 txt 主要提供跌倒窗口和人体框，不能替代动作级人工标注。

正确做法：

1. 标注员先独立标动作片段。
3. 工程人员把官方 TXT 导入 `generated/v2/le2i_official/event_labels.jsonl`，并保留 `label_source=le2i_txt`。

## 14. 每日交付物

每个标注日结束前，标注员提交：

```text
完成 task 列表
有疑问的视频和时间点
U01 比例较高的视频
导出的 CVAT ZIP 文件路径
```


```text
冲突样本列表
需要返工的视频和原因
```

工程人员提交：

```text
来源专属 generated/v2 候选路径
manifest 与源导出 SHA-256
audit 报告路径和 blocker 摘要
标签分布与质量统计
v3 positive/negative/ignore 与 training tier 统计
v3 校验 warnings 和 training_ready 状态
```

## 15. 完成标准

一批标注数据只有同时满足以下条件，才算完成：

- CVAT 原始导出已归档。
- 来源专属动作/事件候选已生成且没有覆盖根目录文件。
- 严格 schema、来源、manifest 和边界审计通过。
- 标签分布和 `U01` 比例已统计。
- 数据划分不泄漏同一人员或同一样本组。
- 只有受控发布且 `formal` 报告 `valid=true` 时，才可声明具备正式评估资格。
- v3 `valid=true` 只表示结构合法；只有目标任务同时具备人工 positive、negative、ignore 和无泄漏 split 时，才可声明训练数据就绪。

## 16. 外部工具参考

- CVAT 官网：<https://www.cvat.ai/>
- CVAT Online：<https://app.cvat.ai/>
- CVAT 用户手册：<https://docs.cvat.ai/docs/manual/>
