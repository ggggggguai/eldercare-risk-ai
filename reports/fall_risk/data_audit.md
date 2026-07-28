# 跌倒风险数据审计

核验日期：2026-07-25

## Manifest

- 资产总数：3,554。
- 视频：2,540。
- 技术可用资产：3,540。
- 技术排除资产：14。
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
| `generated/v2/ntu_rgbd_clip_labels` | 2,976 | 0 | 2026-07-25 人工确认精确全片边界；进入根标签，A043 948 条排除 |

LE2I 官方 TXT 共 130 份，其中 99 个跌倒窗口、31 个 `0/0` 无跌倒窗口。Lecture room 导出包含 27 个有轨迹任务和 250 条轨迹；同一项目导出中的 33 个 Office 任务为空。Office 的 158 条轨迹来自独立的 33 任务导出。TOAGA 的 28 条动作只使用来源派生全片 `A01`，不伪造 CVAT 框或事件。CaucaFall 新增 311 条人工动作和 311 条映射事件，UR Fall 新增 268 条人工动作和 268 条映射事件。NTU RGB+D 新增 2,976 条人工精确全片动作，不生成事件；A043 的 948 条明确排除。可发布 v2 批次合并为 4,759 条根动作和 1,783 条根事件；71 条与官方窗口重叠的 CVAT `fall` 事件按官方来源优先级排除。发布明细和输入/输出 hash 见 `reports/fall_risk/fall-risk-data-v2-root-publish.json`。

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

v2 根标签已完成重建，正式校验结果为 `errors=5,952`、`blockers=179`、`formal_ready=false`。5,952 个 error 来自 2,976 条 NTU 记录的旧解压媒体路径不存在，在 manifest 和 action 层各计一次；这不否定人工边界，但在重新解压或重建路径前不能用于训练。blocker 还包括 `U01/uncertain` 和现有技术/治理限制；`risk_labels.jsonl` 和 subject profiles 仍为空，因此功能/纵向任务继续阻塞。当前仍是可追溯发布候选，不是 frozen 数据版本。

## 模型训练标签 v3

现有 v3 由当前 v2 根标签确定性生成，未覆盖 v2；已包含 UR Fall 和 NTU RGB+D：

| 产物 | 数量 | 结果 |
|---|---:|---|
| `action_labels_v3.jsonl` | 4,759 | primary=4,297、auxiliary=349、ignore=113 |
| `event_labels_v3.jsonl` | 464 | fall positive=312、task-specific ignore=152 |

迁移合并了 71 个 LE2I/CVAT 重叠 fall，159 个 D04 均与唯一父 fall 双向关联；没有把 C03-C05 自动升级为 near-fall，也没有从未标注背景或 LE2I `0/0` 自动生成 negative。NTU 的 2,976 条动作全部为 `primary/exact/single_annotated`，具体动作 tier 也全部为 primary；TOAGA 的 28 条 `normal_walk` 保持 `auxiliary/source_verified/boundary_precision=unknown`。统一 v3 split 有 5,223 条标签分配、3,501 个资产和 154 个保守泄漏组，跨分区泄漏为 0，primary fall 正例按 train/validation/test 分为 74/14/7。v3 校验 `valid=true`、`issues=[]`，但 primary 动作类别未完整覆盖各分区，因此 `training_ready.action_type=false`；event negative=0、near-fall positive=0，两个事件任务也均为 false。

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
| `action_labels.jsonl` | 4,759 | `08a13d283e39f3aa1d7f91cf4077253a7eac923014201fb33e2426e9ed8ce23c` |
| `event_labels.jsonl` | 1,783 | `eb431bedf9104c9cfbd7bf166b1555f1c824b1d23a2a1d21784bb54ab88fd8ea` |

正式校验报告由以下命令生成到 `reports/fall_risk/label_validation_formal_v2.json`；该报告只反映当前输入，不代表模型指标或临床有效性。
