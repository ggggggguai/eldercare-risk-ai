# 跌倒风险数据审计

核验日期：2026-08-02

## Manifest

- 资产总数：7,534。
- 视频：6,520。
- `eligibility=true`：7,520。
- `eligibility=false`：14。
- 排除原因：`duplicate_content=14`。
- manifest 保存媒体 hash、FPS、时长、人员/源组、来源地址和技术排除信息。

Pre_VFallp 的 108 条视频已按用户内部授权例外全部解除技术隔离，并由 `configs/data/fall_risk_internal_authorizations.yaml` 中 5 个 CVAT 压缩包授权组逐条绑定：dizziness forward/side 36 条，其余 confusion delirium、confusion NPH、weakness forward 和 weakness side 各 18 条。授权配置 SHA-256 为 `6258bcb36b64241894382b551ec467139eeebdadc8875895770d176c52add2bd`。108 条均保留 `provenance_status=internal_authorized_source_unverified`，不等于公开来源已验证，并继续共用 `source_group_id=pre_vfallp_unresolved`。旧的 72 条媒体清单只保留为取得四个 CVAT 压缩包前的历史审计证据，不再是现行授权事实源。

## 标签来源

| 批次 | 动作 | 事件 | 状态 |
|---|---:|---:|---|
| `generated/v2/cvat_coffee_01_02` | 514 | 514 | 原始 ZIP 已找回，来源 hash 匹配 |
| `generated/v2/cvat_home_01` | 8 | 8 | 来源 hash 匹配 |
| `generated/v2/cvat_lecture_room` | 250 | 250 | 27 条 Lecture room 视频；脱敏导出 hash 匹配 |
| `generated/v2/cvat_office` | 158 | 158 | 33 条 Office 视频；脱敏导出 hash 匹配 |
| `generated/v2/le2i_official` | 0 | 99 | 官方 TXT 来源 hash 匹配 |
| `generated/v2/toaga_official_walking` | 28 | 0 | 14 位老人 top/bottom 全片 `A01/normal_walk`；`Table_1.xlsx` hash 匹配 |
| `generated/v2/pre_vfallp_confusion_delirium` | 37 | 37 | 18 个 CVAT 视频；候选 formal 校验 `valid=true` |
| `generated/v2/pre_vfallp_confusion_nph` | 28 | 28 | 18 个 CVAT 视频；1 个 outside-only 删除残留已审计忽略；候选 formal 校验 `valid=true` |
| `generated/v2/pre_vfallp_dizziness_fall_forward_side` | 75 | 75 | 36 个 CVAT 视频；候选 formal 校验 `valid=true` |
| `generated/v2/pre_vfallp_weakness_fall_forward` | 54 | 54 | 18 个 CVAT 视频；候选 formal 校验 `valid=true` |
| `generated/v2/pre_vfallp_weakness_fall_side` | 52 | 52 | 18 个 CVAT 视频；候选 formal 校验 `valid=true` |
| `generated/v2/caucafall_manual` | 311 | 311 | 100 个视频、10 名受试者；脱敏 CVAT 导入，原始包不入库 |
| `generated/v2/cvat_ur_fall` | 268 | 268 | 99 个视频；`adl-07-cam0.mp4` 按人工决定不标注；规范化脱敏 CVAT 导入 |
| `generated/v2/ntu_rgbd_clip_labels` | 2,976 | 0 | 2026-07-25 人工确认精确全片边界；A043 不按文件名直接导入 |
| `generated/v2/ntu_rgbd_a043_cvat_review` | 1,428 | 1,428 | 938 个 S001-S017 A043 人工 CVAT 视频按 2026-07-30 决定接受；未标注 A043 不导入 |
| `generated/v2/fall_detection_2017_manual` | 2,977 | 2,977 | 2,011 个源成功导入；项目采集人工 CVAT，来源仍未完成治理，候选策略要求 QC 复核 |
| `generated/v2/fall_tiktok_manual` | 150 | 150 | 66 个任务；项目自采并授权内部训练，候选校验通过 |

LE2I 官方 TXT 共 130 份，其中 99 个跌倒窗口、31 个 `0/0` 无跌倒窗口。Lecture room 导出包含 27 个有轨迹任务和 250 条轨迹；同一项目导出中的 33 个 Office 任务为空。Office 的 158 条轨迹来自独立的 33 任务导出。TOAGA 的 28 条动作只使用来源派生全片 `A01`，不伪造 CVAT 框或事件。CaucaFall 新增 311 条人工动作和 311 条映射事件，UR Fall 新增 268/268。NTU RGB+D 的 2,976 条精确全片动作和 A043 人工 CVAT 的 1,428/1,428 已按各自决定接入；Fall Detection 2017 新增 2,977/2,977 v2 标签，抖音/B站自采批次新增 150/150 并按现行层级进入内部训练。当前 v2 根标签为 9,314 条动作和 6,338 条事件；71 条与官方窗口重叠的 CVAT `fall` 事件按官方来源优先级排除。目录名与报告 `batch_id` 不一致的 S001-S017 候选目录被隔离，不参与发布。发布明细和输入/输出 hash 见 `reports/fall_risk/fall-risk-data-v2-root-publish.json`。

### CaucaFall 来源与限制

- 官方来源：Mendeley Data v4，DOI `10.17632/7w7fccy7ky.4`，CC BY 4.0。
- 本地 100 个 AVI 均完成媒体探测和 hash，按 10 名受试者建立保守 source group，全部 `eligibility=true`。
- 100 条 manifest 记录均为 `label_source=cvat_manual`，按受试者绑定 10 个脱敏任务 ZIP；`source_action_code` 仍只保留目录来源信息。
- 导入产生 311 条动作、311 条映射事件和 20,087 个逐帧 bbox；无结构、帧边界或 bbox 越界问题。
- 原始包含账号/邮箱元数据，导入器只保留脱敏 ZIP，原始压缩包不复制进仓库；v3 标签别名归一和 1 条 U01 缺失原因均记录在导入报告。

### UR Fall 来源与限制

- manifest 含 100 个视频；本批次按人工决定只接收 99 个 CVAT 任务，`adl-07-cam0.mp4` 不标注。
- 导入生成 268 条动作和 268 条映射事件，覆盖 99 个视频；动作包含 60 条明确跌倒和 33 条 `U01`。
- 原始外部 ZIP SHA-256 为 `505112280219b7a24dde757d21a4f37ae54dbb55963f6145f6dc01d9298ef257`，因含身份元数据不复制进仓库。
- 仓库内脱敏 ZIP SHA-256 为 `3ee9760d9718841988757674c1cb52e27aa155bd42ecdc1da331e4348b19e7a1`；规范化 99 个任务名、为 track `46/169/191` 补充 `U01` 原因并移除身份节点，其他标签语义按人工验收结果保留。
- 本批次已随根标签迁移进入 v3 和统一训练 split。

### 抖音/B站跌倒视频整理来源与限制

- 原始批次的 66 个稀疏编号已按旧编号升序映射为 `1.mp4` 至 `66.mp4`；根目录原始视频已移入废纸篓，当前媒体只保留 `annotated_clips/1.mp4` 至 `66.mp4`。旧名到新名的 66 条可逆映射仍固定在 `configs/data/fall_tiktok_source_map_v1.json`。
- CVAT 实际标注使用的 66 个剪辑保存为 `annotated_clips/1.mp4` 至 `66.mp4`。内容关联核验 65 组通过自动特征阈值，第 39 组因竖屏字幕与横屏裁剪得分偏低，人工画面复核确认同源，因此 66/66 组均已确认。
- 66 个任务包含 150 条轨迹，生成 150 条动作和 150 条映射事件。来源级 audit 报告 `reports/fall_risk/fall_tiktok_candidate_validation.json` 为 `valid=true`、`errors=0`、`blockers=0`；formal 模式仍会把 3 条 U01 在动作/事件两层记为 6 个预期 blocker。
- 外部原始 `fall_tiktok.zip` SHA-256 为 `e1879e18a25a943831f6360f4668d313aa1e44a306d16252cffe95008a3133ee`，不复制进仓库；脱敏副本 SHA-256 为 `e8114a9ce011e2653aa1f6e11c79df031e22224dc9496bd0df3159948a280d64`，共移除 134 个身份元素。
- 项目负责人于 2026-07-28 确认为项目自采并授权内部训练，决定文件为 `configs/data/fall_tiktok_collection_decision_v1.json`，SHA-256 为 `b7070e235c25b54754c0169b0e8e3abc8e343f199be2762ceb55e50aa432a02c`。该决定不授予公开再分发权，也不虚构未记录的 consent ID。
- 当前 manifest 仅有 66 个标注剪辑 eligible；原始视频不再作为媒体资产，旧的 2 条重复排除记录只保留在来源审计中。未知人员共用 `fall_tiktok_project_collection_pool`，禁止为了分区平衡拆散。v3 动作为 primary=126、auxiliary=21、ignore=3，事件为 auxiliary=65、ignore=6；统一 split 将该来源组的 221 条 assignment 全部放在 test，取得人员对应关系并重新分组前不进入 train。

### NTU RGB+D A043 人工决定与限制

- 2026-07-30 接受决定固定在 `configs/data/ntu_rgbd_a043_cvat_decision_v1.json`，SHA-256 为 `6c90ce9bd9137aa28ba473504ca7afe75b5f50e15fa3a3b85054f6ea04867492`。只有正式人工 CVAT 批次出现的 938 个 A043 `video_id` 进入主 manifest；未标注 A043 仍禁止按文件名直接导入。
- S001-S017 批次生成 1,428 条动作和 1,428 条映射事件。S016/C003/P008/R001 的 job ZIP 不含源名，只允许由严格 ZIP 文件名绑定到原完整 S016 project 的唯一任务；报告分别保留 project/job SHA-256，并生成含完整 task/source 元数据的规范化 project ZIP。脱敏合并 ZIP SHA-256 为 `ad7fa9c316c57eadc964c31438cd7e35e6956a14eaf0c43310005c998c28eac2`。
- 三视角裁决结果为：S013/P018/R001/C001=D01，S015/P015/R001/C003=D02，S016/P008/R001/C003=D02；S011/P015/R001 和 S017/P020/R001 的六个视角均为 A05 fall hard negative。来源批次 audit/formal 均为 `errors=0`、`blockers=0`、`warnings=0`。
- S002 仍缺 10 个 C001 任务；446 个全片跌倒任务没有精确 onset。306 个完整三视角组中有 303 组一致含跌倒、3 组一致非跌倒、0 组跌倒覆盖冲突；51 个方向不一致、36 个边界差超过 5 帧，这些 QC 限制不因接受决定而消失。

### Lecture room 来源追溯

- 原始外部 ZIP（仅保留文件名，不在仓库保存原始身份元数据）：`lecture_room (1).zip`
- 原始 ZIP SHA-256：`a81a105c54c6dca055349dbb80f0e22c85dd03e10823a25184a9bbd3a4e38ddf`
- 仓库内脱敏来源：`data/annotations/fall_risk/cvat_exports/raw/le2i_lecture_room_cvat_redacted.zip`
- 脱敏 ZIP SHA-256：`c480f86939b7acc16aaf394e682fb99d33be7eca2778d5908a0368a25133cf63`
- 脱敏只移除 CVAT `owner/assignee/username/email` 元数据；任务、轨迹、帧和标签数量保持 60/250。
- 候选审计 `reports/fall_risk/cvat_lecture_room_candidate_validation.json`：`valid=true`、`errors=0`、`blockers=0`。

### Office 来源追溯

- 原始外部 ZIP（仅保留文件名，不在仓库保存原始身份元数据）：`office标注.zip`
- 原始 ZIP SHA-256：`1d8afd68f8c1f86b424a99bbaeca7e7259346d1b4025e5f0cf1a08497803691a`
- 仓库内脱敏来源：`data/annotations/fall_risk/cvat_exports/raw/le2i_office_cvat_redacted.zip`
- 脱敏 ZIP SHA-256：`004ad0dafcef38051c0d559d052564f30f3f2c025abadeb0d47b442a6b4fde36`
- 脱敏时同时将 33 个旧格式 `fall_risk_office__...` task 名规范化为 manifest 约定格式；任务、轨迹、帧和标签数量保持 33/158。
- 候选审计 `reports/fall_risk/cvat_office_candidate_validation.json`：`valid=true`、`errors=0`、`blockers=0`。

### Pre_VFallp 内部授权与标注覆盖

- `dizziness_fall_forward + side.zip`：36 个视频，75 条动作和 75 条映射事件，原始 SHA-256 为 `6ef811ebaebdf227596b7019440dd4f4d6c8c7bb3cc4266de15d8c93eab5da85`。
- `confusion_delirium已经标注.zip`：18 个视频，37 条动作和 37 条映射事件，原始 SHA-256 为 `cbb4d24f20a779fa66389d7942ef9845f4ed9de4ffc7c1794930afe86f025412`。
- `confusion_nph已经标注.zip`：18 个视频，28 条动作和 28 条映射事件，原始 SHA-256 为 `2e263a26b5edff57c2165c3a585cd39b3b4fad81424832ef2681d9386df12a82`。任务 32 的 track 1 只有一个 `outside=1` 框，没有可见区间，按删除残留记入 `ignored_tracks`，未生成标签。
- `weakness_fall_forward.zip`：18 个视频，54 条动作和 54 条映射事件，原始 SHA-256 为 `25441d0854e02a0df00e8bacf3f85d9033061d8ee2678e9965b36a7be7333bf5`。
- `weakness_fall_side.zip`：18 个视频，52 条动作和 52 条映射事件，原始 SHA-256 为 `806cb9212b64e49ba11f473755d8f124e74559b008cb8c7743c16e8d0ba3983d`。
- 五个批次合计覆盖 108/108 个视频，产生 246 条动作和 246 条映射事件。动作分布为 `A01=25`、`B01=72`、`B02=20`、`B03=1`、`B04=43`、`C03=13`、`D01=36`、`D02=36`；事件分布为 fall=72、gait_instability=136、near_fall=13、normal_activity=25。
- 原始外部 ZIP 不复制进仓库；108 个仓库内 task ZIP 已移除 `owner/assignee/username/email` 身份节点，复扫未发现身份标签或邮箱字符串。五份候选 formal 报告均为 `valid=true`、`errors=0`、`blockers=0`。
- 内部授权只解除 manifest 技术隔离；未知人员、未知公开来源和单一保守源组限制继续保留。

## 当前结论

v2 根标签已完成重建，NTU 外部 manifest 已重定位到仓库内 `data/external/ntu`；3,924 个 NTU AVI 和主 manifest 纳入的 3,914 个资产均可访问。Fall Detection 2017 人工批次已进入 v2 根标签，但仍保留项目来源和 QC 门禁。全库正式校验结果为 `errors=0`、`blockers=285`、`formal_ready=false`，其中 27 条为 `formal_manifest_ineligible`、258 条为 `formal_uncertain`。`risk_labels.jsonl` 和 subject profiles 仍为空，因此功能/纵向任务继续阻塞。当前仍是可追溯发布候选，不是 frozen 数据版本。

## 模型训练标签 v3

现有 v3 由当前 v2 根标签确定性生成，未覆盖 v2；已包含 UR Fall 和 NTU RGB+D：

| 产物 | 数量 | 结果 |
|---|---:|---|
| `action_labels_v3.jsonl` | 9,314 | primary=8,444、auxiliary=702、ignore=168 |
| `event_labels_v3.jsonl` | 2,567 | fall positive=2,303、fall negative=6、task-specific ignore=258 |

迁移合并了 71 个 LE2I/CVAT 重叠 fall，220 个 D04 均与唯一父 fall 双向关联；没有把 C03-C05 自动升级为 near-fall，也没有从未标注背景或 LE2I `0/0` 自动生成 negative。NTU 的 2,976 条精确全片动作保持既定 tier，A043 按人工边界精度迁移；Fall Detection 2017 的 2,977 条动作进入同一 v3 训练视图并贡献 1,097 条事件窗口；六条 A05 裁决通过决定文件及其 SHA-256 生成 `manual_v3/adjudicated` 的 `squat_or_kneel` fall negative，裁决零漏配。TOAGA 的 28 条 `normal_walk` 保持 `auxiliary/source_verified/boundary_precision=unknown`。抖音/B站自采来源的动作按 primary=126、auxiliary=21、ignore=3 接入，事件按 auxiliary=65、ignore=6 接入。统一 v3 split 有 11,881 条标签分配、6,516 个资产和 184 个保守泄漏组，跨分区泄漏为 0，primary fall 正例按 train/validation/test 分为 74/14/7，negative 为 6/0/0。v3 校验 `valid=true`、`issues=[]`，但 primary 动作类别未完整覆盖各分区，另六类 fall hard negative、negative 分区覆盖和 near-fall positive 仍缺，因此三项 `training_ready` 均为 false。

迁移与校验证据：

```text
reports/fall_risk/training-labels-v3-migration.json
reports/fall_risk/training-labels-v3-validation.json
data/splits/fall_risk/training_labels_v3/assignments.jsonl
data/splits/fall_risk/training_labels_v3/split.json
```

## 根标签 hash

| 文件 | 记录数 | SHA-256 |
|---|---:|---|
| `action_labels.jsonl` | 9,314 | `058540ce76076b373bfa86c70d0b157fa10af5cbb521aa7c4c58a3ac7a1b3af6` |
| `event_labels.jsonl` | 6,338 | `fff793d735d89a7675e833cd97969c9c58fe3595abdcb57b734a7d3e43232099` |

正式校验报告由以下命令生成到 `reports/fall_risk/label_validation_formal_v2.json`；该报告只反映当前输入，不代表模型指标或临床有效性。
