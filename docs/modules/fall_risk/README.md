# 跌倒风险算法模块

更新时间：2026-08-12

本模块承接 `docs/modules/fall_risk/plans/跌倒风险算法研发计划.md`。

面向 2026 年挑战杯揭榜挂帅赛题 `XH-202617` 的冲奖优先级、七周里程碑和验收门槛，见[挑战杯揭榜挂帅冲奖增强计划](plans/挑战杯揭榜挂帅冲奖增强计划.md)。该计划只描述未来工作，不能作为当前能力或实测指标的证据。

拟议的 ESP32-S3、照度与水浸传感接入，以及人-环境交互特征和受控实验，见[环境因素多模态增强实施计划](plans/环境因素多模态增强实施计划.md)。该计划不表示环境设备或融合逻辑已经接入当前主链。

各任务的检测、跟踪、姿态、平滑、步态、坐站、近跌倒、个体基线和风险融合模型候选，见 `docs/modules/fall_risk/plans/跌倒风险各任务模型调研与选型矩阵.md`。该文档区分当前主线、短期对照和数据充足后的增强实验，不能把候选清单理解为已实现能力。

新同事加入或开始较大改动前，建议先阅读 `docs/modules/fall_risk/guides/跌倒风险算法协作开发指南.md`。该文档汇总了当前实现状态、固定算法路线、开发注意事项、验证命令和后续优化方向。

## 当前研发阶段：模型化增强

当前模块不是停留在规则 baseline 阶段，而是进入模型化增强阶段。增强主线是使用来源完整、无泄漏划分的数据，将步态、坐站、近跌倒/跌倒事件等规则分支逐步替换为 TCN、MS-TCN++ 或 ST-GCN 等时序模型，并与现有规则结果做受控对照。

当前仍保留规则分支用于 baseline、紧急事件安全覆盖、低质量输入降级和结果解释。个体行为基线及最终风险融合只有在连续个人数据或 `risk_labels` 真值、冻结 split 和正式评估协议具备后，才进入模型替换；计划中的候选模型不能写成已经完成的主链能力。

## 算法主线

```text
视频或离线特征
  -> 人体检测与跟踪
  -> 姿态关键点
  -> 关键点质量控制与时序平滑
  -> 步态/坐站/转身/近跌倒特征
  -> 个体化行为基线
  -> 规则 baseline / 时序模型增强
  -> 风险事件 JSON
```

## 当前实现原则

- 已保留结构化特征单次推理，并跑通视频姿态、内存滑窗、风险事件和 HTTP 回调原型。
- 实时主线使用 YOLOv8-Pose + ByteTrack；RTMPose 作为可选离线后端。
- 实时演示路径优先稳定，复杂模型只做增强实验。
- 输出是风险提示/预警事件，不是医疗诊断结果。

## 实时链路阶段 0-3

实时链路质量冲刺的阶段 0-3 已完成核心工程实现和自动化单元/集成回归；2026-08-12 真实萤石算法端 120 秒严格烟测已通过，业务回调、地址刷新、弱网和长时资源仍未验收：

- `configs/modules/fall_risk_runtime_acceptance.yaml` 冻结了离线回放输入、时钟域、latest-frame 容量、采样门禁和外部验收阻塞项；`scripts/evaluate/build_fall_runtime_fingerprint.py` 输出配置和输入 hash。
- 会话使用有界 latest-frame 缓冲；队列满时丢弃最旧帧并记录计数与源 PTS。`stream_epoch` 在每次成功打开流时递增，取帧队列、推理限频与窗口状态不会跨 epoch 复用。
- 源 PTS 用于姿态/动作窗口；接收等待、推理节流、帧龄和停止预算使用单调时钟。采样诊断分别报告源有效采样率、接收有效采样率、最大 gap、窗口覆盖和每个时序分支的采样门禁。
- 步态、坐站、近跌倒、跌倒状态和个人基线分别输出 `valid`、`unavailable` 或 `inference_error`，并记录输入/有效帧数、核心与分支关键点覆盖、窗口时长、有效 FPS、最大 gap、版本、原因和耗时。不可用分数为 `null`，融合只消费显式 `fusion_mask` 并按可用权重归一化，不再把缺失分支当作零风险；场景分只作为行为或基线信号的上下文，不能单独触发风险。
- 目标选择显式记录 `unbound`、`bound`、`ambiguous` 和 `lost`。未绑定时多人输入不会按最大框自动绑定；既有轨迹仍存在时保持绑定；丢失超过 TTL 后的接替会重置旧窗口并记录原因。`person_id` 仍只是专属流业务先验，`binding_verified=false`。
- 每个 epoch 会重置跟踪器、姿态窗口、跌倒状态、事件节流和采样状态，并记录丢弃的轨迹/窗口及边界原因；URL 更新使用 `stream_url_updated`，普通重连使用 `stream_reconnected`。已生成事件在 `metadata.stream_epoch` 保留来源 epoch。
- 风险过程状态机区分 `idle`、`suspected`、`confirmed_static`、`recovered` 和 `unresolved`。状态边保存来源 epoch、源 PTS、单调进入时间、峰值、最后有效观测、TTL、触发条件和清理原因；疑似信号在躯干角回到不超过 45 度且运动分不低于 0.05 时释放，默认恢复确认窗为 3 秒源时间、无有效观测 TTL 为 30 秒单调时间。低质量观测不推进静止时长。这些值只冻结工程生命周期语义，不是识别效果或医学阈值。`confirmed_static` 只表示配置的静止条件满足。
- 同一 episode 的首次、升级、静止条件满足、恢复和 unresolved 使用稳定 `episode_id` 和各自不同的 `event_id`；同一版本重试或显式重放复用原 `event_id` 与冻结 JSON。普通 level-0 不回调，但 `recovered/unresolved` 作为 episode 终止版本即使 level 为 0 也投递。
- `process_frame()` 不执行 HTTP、重试等待或证据生成等待，只做不可变版本创建和非阻塞入队。默认容量 32 的进程内 outbox 独立记录 `pending/retrying/delivered/failed`；队列满时优先保留 4 级和唯一终止版本，所有替换或拒绝都有计数和最后决策。重连不会取消旧 epoch 事件。
- 停止给 outbox 默认 3 秒排空预算、会话线程默认 5 秒停止预算；诊断记录 delivered、failed、unfinished、rejected 和投递线程预算结束状态。HTTP 单次尝试有超时，因此推理线程不受网络等待影响；若停止预算短于在途 HTTP，API 按预算返回并明确报告仍在结束的投递线程。
- `GET /v1/monitoring/sessions/{session_id}` 可查询 `stream_epoch`、frame queue、目标、分支窗口、阶段耗时和 epoch 清理诊断；不返回直播地址、token 或原始视频。
- [`reports/fall_risk/runtime/`](../../../reports/fall_risk/runtime/README.md) 中现有 LE2I 报告仍是先前阶段证据，不能据此宣称阶段 3 的真实设备、长时资源、识别准确率、泛化或临床有效性。

当前固定窗口对完全相同输入使用 memoization；每次加入新姿态后仍需重做质量平滑和分支分析，真正的增量时序计算及固定硬件资源收益证据尚未完成。YOLOv8-Pose/ByteTrack 后端目前只能提供检测、跟踪、姿态合并耗时，无法从该后端拆出三个独立计时。outbox 是进程内实现，不提供数据库持久化或进程重启恢复。

## 工作流 A：数据、标注与评估底座

截至 2026-08-09，工作流 A 的自动化底座、v2 根标签发布、模型训练标签 v3 结构和跌倒事件 P0 fail-closed 审计已实现。现有标签经项目负责人裁决后，fall/near-fall 事件监督已通过 v3 数据与 split 门禁；动作类型、v2 formal、连续背景分母、老人域泛化和正式评估仍未通过。当前事实以实际产物和审计结果为准，不以[工作流 A 执行任务书](plans/工作流A-Codex执行任务书.md)中的计划项作为完成证据。

| 能力 | 当前状态 | 证据与限制 |
|---|---|---|
| 统一数据 manifest | 已实现并对本地数据运行 | `data/manifests/fall_risk_video_manifest.jsonl` 当前有 7,684 条资产，其中 6,670 条是视频；包含 SCF P01/P02/P04 的 150 个 auxiliary train 视频，以及 3,914 条已接受的 NTU RGB+D 视频。SCF P03/P05 未进入根 manifest |
| 通用全视频姿态缓存 | 当前训练相关视频覆盖完整 | 主 pose cache 的 8 个批次覆盖 6,512 个 eligible RGB 视频、674,889 源帧和 708,177 条 raw/cleaned 记录；SCF P01/P02/P04 的 150 个训练视频复用 `baseline_replay/cleaned_pose` 独立 cache。步态 builder 通过重复 `--additional-pose-dir` 合并读取，当前 6,662 个 eligible RGB 视频均有可用 pose，未复制或重跑 SCF 姿态。该覆盖不等于正式训练或主链路替换已就绪 |
| 跌倒动作候选 TCN | 历史 pilot，已退休 | 旧 candidate-clip 仅作报告追溯；当前不训练、不评估、不接入实时主路径。连续因果输入基础设施另行保留 |
| 跌倒连续因果 TCN | provisional `experimental_tcn` 已进实时主评分；规则保留 fallback | v2 治理重新绑定 `splitv3_3342705b7b1ac51570148337`，生成 7,701 条开发监督并物化 7,504 个 `[32,17,20]` 因果窗口，train/validation=`5,721/1,783`，test 语义和姿态未读取。三 seed F1@0.5 均值 `0.6745`、PR-AUC 均值 `0.6782`；平均概率 F1/Recall=`0.6824/0.7360`，普通背景误报率 `0.2116`，onset validation 仅 7 条。配置已将 TCN 有效窗口作为 `fall_event_score` 主来源，命中按现有 0.9 强触发契约进入融合/事件状态机；窗口不足或推理失败回退规则，shadow/raw 分数仍写入诊断。连续背景、老人域和冻结协议仍缺失，证据等级为 `development_provisional`。见[训练报告](../../../reports/fall_risk/fall_event_continuous_tcn_v2/README.md) |
| 跌倒事件训练 P0 审计 | 已实现；主路径门禁仍阻塞 | `scripts/audit/audit_fall_event_training.py` 不解析 test 标签内容，只绑定 manifest、v3 标签、split、formal/v3 报告和配置 hash。现有 P0 报告早于 SCF 发布，需按当前 `splitv3_3342705b7b1ac51570148337` 重建；formal、连续背景、老人域、冻结 split/协议和 test 保管门禁仍未通过。旧 candidate 报告不能作为当前 split 结果。见[P0 审计](../../../reports/fall_risk/fall_event_training_audit.md)和[阻塞清单](../../../reports/fall_risk/fall_event_blockers.md) |
| 步态窗口、训练与运行 | provisional seed 43 已受控进入主链；规则保留 fallback | `splitv3_3342705b7b1ac51570148337` 的 observable-context v2 有 2,372 个窗口，SCF 260 个窗口只进入 shared encoder/walking gate。transfer seed 43 validation F1=`0.857`、walking-gate F1=`0.919`、正常窗口误报约 `12.0/hour`。2026-08-18 因比赛交付时间约束，由项目负责人批准将该 checkpoint 写入默认服务配置；输入不足、任务/窗口契约不兼容或推理失败时自动使用规则分。test/老人域/冻结协议仍未完成，因此证据等级保持 `development_provisional`。见[运行启用记录](../../../reports/fall_risk/gait_runtime_activation_20260818.md) |
| 坐站训练准备与因果 TCN | v2 数据治理与 seed 42 完整流 pilot 完成；No-Go | SCF 发布后专项 v2 发布 1,942 个事件、5,512 个显式背景和 229 个 ignore；298 条 SCF 标签强制 auxiliary/train-only。物化感知 split 生成 11,373 个多 cutoff `[64,14,9]` 窗口并通过开发门禁，test 未读取。同 split 完整流上，TCN event F1/Recall/FP-hour 为 0.449/0.419/158.8，规则为 0.168/0.382/1142.8；但仍低于 0.75/0.85 门禁，显式背景仅 0.353 camera-hour，卧床转移与跪地误报严重，故不运行 seed 43/44。默认 checkpoint 仍为 `null`，规则主路径不变。见[v2 治理与训练报告](../../../reports/fall_risk/sit_stand_event_v2/README.md) |
| 近跌倒恢复确认 TCN | v2 三 seed及 A01/A04 辅助消融完成；主链路 No-Go | base 使用 3 秒/2 秒回退，SCF E1 最终构建 3,842 个窗口和 1,695 个事件，665 条 test 标签未读取；三 seed validation F1 为 `0.8395-0.8772`，P05 负例为 `13/34`、`28/34`、`22/34`。A01/A04 又提供 802 个低权重 train 窗口，467 个 validation 窗口独立 challenge；0.10/0.25/0.35 seed 42 的 primary validation F1 提高到 `0.8906/0.9003/0.8973`，但 P05 负例恶化为 `25/34`、`24/34`、`21/34`。故不扩 seed、不晋级 checkpoint。见[v2 治理与训练报告](../../../reports/fall_risk/near_fall_event_v2/README.md)；规则主路径保持不变 |
| SCF_MVP_V1 自采增强 | P01/P02/P04 已进入训练根标签；E1-E3 No-Go | 150 个视频、298 条动作进入 v2/v3，派生 153 条任务事件；451 条 assignment 全部固定在 train。P05 challenge、P03 excluded 不变；规则和原 checkpoint 保持主路径。见[执行报告](../../../reports/fall_risk/self_collected_scf_mvp_v1/README.md) |
| 标注导入与严格校验 | v2 根标签已发布；formal 阻塞 | 根标签为 9,612 条动作、6,338 条事件；SCF 新增 298 条动作。结构错误为 0；formal 为 `errors=0`、`blockers=285`、`formal_ready=false` |
| CaucaFall 人工标注 | 已进入主标签链 | 官方 DOI 为 `10.17632/7w7fccy7ky.4`；100 个视频、10 名受试者、311 条人工 CVAT 动作和 311 条映射事件已接入。manifest 标为 `label_source=cvat_manual`，10 个脱敏任务 ZIP 位于 `cvat_exports/raw/caucafall_manual/`；原始 ZIP 不入库，别名和脱敏记录见 `generated/v2/caucafall_manual/import_report.json` |
| NTU RGB+D 人工标注 | 已进入主标签链 | 外部 NTU manifest 共 3,924 个 RGB 视频且均可访问；2,976 条 A008/A009/A042/A080 片段按 2026-07-25 人工决定进入 v2/v3/split。S001-S017 的 938 个已标 A043 视频按 2026-07-30 接受决定接入，生成 1,428 条动作和 1,428 条映射事件；未标注 A043 不按文件名导入。S016/C003/P008/R001 job revision 以严格 ZIP 文件名绑定并叠加到完整 S016 project，规范化 project ZIP 保留源名和两层 SHA-256。2026-08-04 裁决将当前 v3 的 446 条全片跌倒设为首帧 onset、尾帧 offset，并确认全部 NTU C03 为近跌倒。该批次仍缺 S002 的 10 个 C001 任务；306 个完整三视角组中有 51 个方向不一致、36 个原 CVAT 边界差超过 5 帧 |
| 抖音/B站跌倒视频整理 | 项目自采；标签已具备训练 tier | 66 个标注剪辑保存在 `annotated_clips/1.mp4` 至 `66.mp4`，当前 manifest 仅保留这 66 个 clip；根目录原始视频已移入废纸篓。150 条动作和 150 条映射事件已进入 manifest/v2；v3 动作为 126 primary、21 auxiliary、3 ignore，事件为 65 auxiliary、6 ignore。未知人员按单一保守来源组防泄漏，当前 221 条 assignment 全部在 test；决定不授予公开再分发权 |
| 模型训练标签 v3 | 事件监督门禁通过；动作类型门禁仍阻塞 | 独立输出 9,612 条动作和 9,651 条事件：fall 正/负=2,303/1,827，near-fall 正/负=981/4,282，ignore=258。统一 split 覆盖 19,263 条标签、6,666 个资产、187 个泄漏组且跨分区泄漏为 0；SCF 451 条均在 train。`training_ready.fall_event=true`、`training_ready.near_fall_event=true`，`action_type=false` |
| 四任务独立 split | 两个 provisional ready、两个 blocked | 基于 v2 根事件的专项 split 与 19,263 条 assignment 的 v3 统一训练 split 用途不同，当前均不是 frozen split |
| 跌倒/近跌倒事件评估器 | 已实现；仅完成合成烟测 | 入口为 `scripts/evaluate/evaluate_fall_events.py`，开发协议位于 `configs/evaluation/`；`reports/fall_risk/workflow_a_synthetic_evaluation/bundle/` 证明 bundle 生成链路可运行，但协议是 `development_provisional`、输入是合成数据，任何数值都不是比赛指标或真实模型效果 |
| 正式数据版本与指标 | 未就绪 | 需要先取得合格标签、正式校验报告、非空冻结 split、冻结评估协议和盲测治理证据，才能生成正式指标 |

### 全视频姿态缓存批次

以下批次统一使用仓库内 `models/yolov8n-pose.pt`、ByteTrack、CPU 和相同质量控制契约；每批报告位于 `data/processed/fall_risk/pose_quality_y8n_v1/batches/<batch_id>/`。

| batch_id | 视频 | 源帧 | raw/cleaned 姿态 | 检测帧覆盖率 | 有效关键点率 | 下肢有效率 | 步态可用率 | 验收 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `le2i_imvia` | 184 | 74,375 | 72,231 | 89.3163% | 83.4954% | 89.4570% | 87.7573% | `passed` |
| `caucafall` | 100 | 19,877 | 17,847 | 89.1030% | 90.9802% | 98.4386% | 98.1341% | `passed` |
| `ntu_rgbd` | 3,914 | 296,511 | 307,137 | 99.8664% | 94.0833% | 97.7042% | 96.9444% | `passed` |
| `fall_detection_2017` | 2,012 | 113,651 | 112,624 | 96.1523% | 90.6755% | 96.1518% | 94.4745% | `passed` |
| `fall_tiktok` | 66 | 10,945 | 13,895 | 88.4605% | 81.2912% | 91.0004% | 88.7370% | `passed` |
| `pre_vfallp` | 108 | 84,089 | 82,994 | 98.6847% | 98.0639% | 99.9733% | 100.0000% | `passed` |
| `toaga` | 28 | 60,510 | 79,950 | 90.7420% | 84.0798% | 99.6100% | 99.4296% | `passed` |
| `ur_fall` | 100 | 14,931 | 21,499 | 82.8880% | 89.0288% | 94.9951% | 93.6230% | `passed` |

`fall_detection_2017` 有 2 个 `completed_no_pose` 视频，经全片抽帧确认均为空场景。UR Fall 的 `adl-07-cam0.mp4` 已按当前 SHA-256 `0b346e78321c0cdfd281a19f3f4fd9cb6d61ed27f4caa1a955ac182c5b66a954` 重提完整 180 帧并重新通过批次严格校验；该视频仍按既有人工决定不标注，不进入 v3 监督。Pre_VFallp 当前 108 个资产已依据 `configs/data/fall_risk_internal_authorizations.yaml` 取消数据集级隔离，全部为 `eligibility=true`，可用于项目内部研发；这不等于公开来源或公开再分发权已经验证，未知人员仍共用 `pre_vfallp_unresolved` 保守源组。manifest 中另有 8 个 `duplicate_content` 视频为 `eligibility=false`，不属于本缓存的输入范围。

当前数据阻断包括：129 条 `U01/uncertain`、技术排除资产关联、Fall Detection 2017 项目来源治理、人员或保守源组治理、功能与纵向参考终点、测试集保管职责隔离，以及评估协议预注册。责任角色和解除条件见[工作流 A 阻塞清单](../../../reports/fall_risk/workflow_a_blockers.md)。

自动化入口统一通过项目 conda 环境运行。先确认 editable 安装指向当前仓库：

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
```

使用非现行输出路径复跑 manifest，避免覆盖当前审计输入：

```bash
conda run -n eldercare-ai python scripts/annotation/build_fall_risk_manifest.py \
  --repo-root . \
  --output /tmp/fall_risk_video_manifest.jsonl
```

通用姿态缓存按数据源分批运行；相同契约下重跑会校验 state 和文件大小后跳过已完成视频：

```bash
conda run -n eldercare-ai python scripts/prepare/prepare_fall_pose_cache.py \
  --dataset ntu_rgbd \
  --batch-id ntu_rgbd \
  --model models/yolov8n-pose.pt \
  --device cpu \
  --continue-on-error

conda run -n eldercare-ai python scripts/prepare/validate_fall_pose_cache.py \
  --batch-id ntu_rgbd
```

缓存根目录的 `cache_manifest.json` 固定创建缓存时的 manifest 哈希，并严格固定模型、阈值、tracker 和姿态质量参数。后续仅 manifest 内容修订时可以复用同一缓存；每个视频仍按 manifest 中的 `source_sha256` 与 state 校验是否需要重提。模型或提取参数变化时必须使用新的输出目录，不能混写已有缓存。

审计当前根标签；当前命令会明确报告 formal blocker，直到 `formal_ready=true` 前预期以非零状态退出：

```bash
conda run -n eldercare-ai python scripts/annotation/validate_fall_risk_labels.py \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --action-labels data/annotations/fall_risk/action_labels.jsonl \
  --event-labels data/annotations/fall_risk/event_labels.jsonl \
  --risk-labels data/annotations/fall_risk/risk_labels.jsonl \
  --subject-profiles data/annotations/fall_risk/subject_profiles.json \
  --config configs/data/fall_risk_label_validation_v2.yaml \
  --mode audit \
  --report-output /tmp/fall_risk_label_validation_audit.json
```

生成绑定当前输入 hash 的跌倒事件训练 P0 审计；输出存在时默认拒绝覆盖：

```bash
conda run -n eldercare-ai python scripts/audit/audit_fall_event_training.py
```

当前结果为 `infrastructure_only`，不授权正式训练、frozen 协议生成或 test 评估。

生成并校验不覆盖 v2 的模型训练标签：

```bash
conda run -n eldercare-ai python scripts/annotation/migrate_fall_labels_v2_to_v3.py --overwrite
conda run -n eldercare-ai python scripts/annotation/build_fall_training_split_v3.py --overwrite
conda run -n eldercare-ai python scripts/annotation/validate_fall_labels_v3.py --overwrite
```

`valid=true` 只表示 schema、边界、来源和引用合法。现行校验还要求事件任务具备完整 hard-negative 类型和三分区正负覆盖；当前 `fall_event`、`near_fall_event` 已通过该监督门禁，`action_type` 仍因部分稀有动作缺少三分区覆盖而关闭。事件监督可用不代表模型、连续背景误报分母、老人域泛化或正式 test 已就绪。

当前配置可复现四任务开发 split：`fall_event_v1`、`near_fall_event_v1` 生成 provisional `ready` 产物，功能 proxy 和纵向基线继续生成 `blocked` 产物。`ready` 只表示当前 v2 根事件可构建无泄漏开发划分，不表示 formal、frozen 或正式评估就绪：

```bash
conda run -n eldercare-ai python scripts/split/build_fall_risk_splits.py \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --annotations-dir data/annotations/fall_risk \
  --config configs/data/fall_risk_splits_v1.yaml \
  --output-dir /tmp/fall_risk_splits
```

事件评估器的完整 bundle 只能先用合成 fixture 做开发烟测：

```bash
conda run -n eldercare-ai python scripts/evaluate/build_synthetic_fall_event_fixture.py \
  --output-dir /tmp/workflow_a_synthetic/input

conda run -n eldercare-ai python scripts/evaluate/evaluate_fall_events.py \
  --ground-truth /tmp/workflow_a_synthetic/input/ground_truth.jsonl \
  --predictions /tmp/workflow_a_synthetic/input/predictions.jsonl \
  --manifest /tmp/workflow_a_synthetic/input/manifest.jsonl \
  --split /tmp/workflow_a_synthetic/input/split.json \
  --assignments /tmp/workflow_a_synthetic/input/assignments.jsonl \
  --partition validation \
  --config configs/evaluation/fall_event_v1.provisional.yaml \
  --output-dir /tmp/workflow_a_synthetic/bundle \
  --label-version synthetic-labels-v1 \
  --allow-provisional
```

## 目录对应

```text
src/elderly_monitoring/modules/fall_risk/features.py
  跌倒风险特征整理

src/elderly_monitoring/modules/fall_risk/pipeline.py
  跌倒风险推理主流程

src/elderly_monitoring/modules/fall_risk/tracking.py
  人体检测与跟踪，输出人体框和轨迹

src/elderly_monitoring/modules/fall_risk/pose.py
  姿态关键点提取，输出人体骨架序列

src/elderly_monitoring/modules/fall_risk/pose_quality.py
  关键点质量控制与时序平滑，输出带质量标记的稳定姿态序列

src/elderly_monitoring/modules/fall_risk/gait.py
  步态稳定性特征提取和规则 baseline，输出 gait_risk_score

src/elderly_monitoring/modules/fall_risk/sit_stand.py
  坐站转换能力特征提取和规则 baseline，输出 sit_stand_risk_score

src/elderly_monitoring/modules/fall_risk/near_fall.py
  近跌倒事件检测规则 baseline，输出 near_fall_event_score

src/elderly_monitoring/modules/fall_risk/fall_event_training.py
  跌倒动作 proxy 数据审计与 train/validation 窗口准备

src/elderly_monitoring/modules/fall_risk/fall_event_continuous.py
  跌倒连续因果 17 点输入张量和 cutoff-only 重采样

src/elderly_monitoring/modules/fall_risk/fall_event_continuous_dataset.py
  屏蔽未来插值并物化 `[32,17,20]` train/validation 因果窗口

src/elderly_monitoring/modules/fall_risk/fall_event_continuous_tcn.py
  development-only causal residual TCN 训练、早停和 provisional checkpoint

src/elderly_monitoring/modules/fall_risk/fall_event_tcn.py
  provisional 跌倒候选片段 TCN 训练、复评和 shadow 推理

src/elderly_monitoring/modules/fall_risk/fall_event_training_audit.py
  跌倒事件训练 P0 hash、数据、协议和 test 治理门禁审计

src/elderly_monitoring/modules/fall_risk/baseline.py
  个体化行为基线建模，输出 baseline_deviation_score

src/elderly_monitoring/runtime/
  实时姿态跟踪、特征窗口、跌倒状态和事件节流

src/elderly_monitoring/service/
  直播流读取、单会话 HTTP 服务和风险回调

data/annotations/fall_risk/
  动作级、事件级、风险级标注

data/processed/fall_risk/
  关键点、轨迹、步态、坐站、近跌倒和个体基线特征
```

动作级标签字典统一包含 29 个 canonical 标签：截图中的 27 个标签加 `A12_normal_hop` 和 `D05_seated_fall`。可控下蹲/弯腰分别标 `A05/A06`，主动躺下和被协助降低身体分别标 `A07/A11`，快速下沉后恢复仍按 `C05` 处理。

## 人体检测与姿态关键点

人体检测与跟踪：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_tracking.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --output data/processed/fall_risk/tracks/home_01_video_1_tracks.jsonl \
  --model yolov8n.pt \
  --scene-region home
```

跟踪 JSONL 的 `bbox` 和 `center` 始终使用像素坐标，`speed_px_per_sec` 也以像素/秒表示；它与下面姿态 JSONL 的默认归一化 `bbox` 是两个独立契约。

姿态关键点提取默认仍使用已验证的 YOLOv8-pose 后端：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_pose.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --output data/processed/fall_risk/poses/home_01_video_1_poses.jsonl \
  --backend yolov8-pose \
  --model models/yolov8n-pose.pt \
  --scene-region home
```

研发计划中的目标姿态模型是 RTMPose。当前工程已新增 RTMPose/MMPose 可选后端，输出仍复用同一套 pose JSONL 契约：`frame_id`、`person_id`、`track_id`、`bbox`、`bbox_pixels`、`coordinate_system`、`scene_region`、`keypoints`、`pose_confidence`、`keypoint_quality` 和 `timestamp_sec`。默认模式下，姿态记录的 `bbox` 与 `keypoints[].x/y` 都是相对于整幅图像的 `0-1` 归一化坐标，且 `coordinate_system` 为 `image_normalized_0_1`；`bbox_pixels` 始终保留检测器原始的 `[x1, y1, x2, y2]` 像素框，供标注匹配和诊断使用。增加 `--absolute-coordinates` 后，`bbox` 与关键点改为像素坐标，`coordinate_system` 为 `image_pixels`，`bbox_pixels` 仍保留同一像素框。

现有质量控制、步态、坐站和近跌倒规则默认消费 `image_normalized_0_1`；绝对坐标输出主要用于导出或诊断，进入这些下游模块前必须先转换回归一化坐标。

RTMPose 后端当前只做预训练权重推理，不包含训练或微调流程。若已在 `eldercare-ai` 环境安装 MMPose、MMCV 和 MMEngine，可直接运行：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_pose.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --output data/processed/fall_risk/poses/home_01_video_1_rtmpose_poses.jsonl \
  --backend rtmpose \
  --pose-config human \
  --device cpu \
  --scene-region home
```

也可以通过 `--pose-config` 指向具体 RTMPose config 或 MMPose 模型别名，通过 `--pose-checkpoint` 指向对应预训练权重。当前 RTMPose CLI 使用 MMPose 推理结果直接适配为统一 JSONL；若 MMPose 输出中没有稳定 `track_id`，会按帧内人体序号生成 `track_id/person_id`，后续接入 tracking JSONL 时可复用同一适配层替换为稳定轨迹 ID。若这些可选依赖未安装，脚本会在选择 `--backend rtmpose` 时抛出清晰的 `RuntimeError`，提示安装 MMPose 相关依赖；YOLOv8-pose 后端和普通姿态单元测试不受影响。

关键点质量控制与时序平滑：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_pose_quality.py \
  --input data/processed/fall_risk/poses/home_01_video_1_poses.jsonl \
  --output data/processed/fall_risk/poses/home_01_video_1_poses_cleaned.jsonl
```

当前质量控制层是规则/统计 baseline，不输出 `risk_level`、`risk_score` 或 `recommended_action`。它只消费姿态 JSONL，并按 `person_id + track_id` 分组生成后续步态、坐站和近跌倒模块可复用的稳定关键点序列。

新增输出字段：

| 字段 | 含义 |
|---|---|
| `core_keypoint_quality` | 面向下游动作分析的核心关键点质量分 |
| `valid_core_count` | 有效核心关键点数量 |
| `missing_core_names` | 当前帧缺失或低置信度的核心关键点名称 |
| `quality_state` | `usable`、`missing_core`、`low_quality` 或 `low_quality_run` |
| `low_quality_run_length` | 当前连续低质量帧长度 |
| `keypoints[].valid` | 该关键点是否可作为有效观测或短缺失插值 |
| `keypoints[].source` | `observed`、`low_confidence`、`missing` 或 `interpolated` |
| `keypoints[].x_smooth` / `keypoints[].y_smooth` | 指数平滑后的坐标，不覆盖原始 `x/y` |
| `keypoints[].is_jump_outlier` | 是否触发归一化坐标异常跳变标记 |
| `window_quality` | 当前时间窗质量摘要和下游模块可用性 |

默认规则：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `min_keypoint_score` | `0.30` | 低于该置信度的关键点标记为低置信度 |
| `low_quality_threshold` | `0.45` | 低于该核心质量分的帧标记为低质量 |
| `low_quality_run_frames` | `3` | 连续低质量帧达到该长度后标记为 `low_quality_run` |
| `max_interp_gap_frames` | `2` | 最多对 1-2 帧短缺失片段做线性插值 |
| `alpha` | `0.40` | 指数平滑系数 |
| `jump_threshold_norm` | `0.18` | 核心关键点相邻帧异常跳变阈值 |
| `window_sec` | `1.0` | 窗口质量摘要默认秒级窗口 |

下游模块应优先使用 `x_smooth/y_smooth`、`valid` 和 `window_quality` 判断关键点是否适合计算步态、坐站转换或近跌倒特征；异常跳变点不会被删除，但会被标记并在平滑中降权。

## 步态稳定性分析

步态稳定性分析对应固定算法路线中的第 5 步：

```text
关键点质量控制与时序平滑
  -> 步态稳定性分析
  -> 坐站转换能力分析
```

当前已实现“轻量层级 TCN 主预测接口 + 人工特征解释/规则 fallback”双分支。新 checkpoint 把连续 `quality` 隔离在编码器之外，只用于 mask/池化；walking gate 先估计活动适用性，条件异常头只在 walking 标签上训练，最终输出两者概率乘积。TCN 按 checkpoint 契约把 source-time 姿态重采样为 `[16,14,5]`，并在观测不足时拒识。默认服务配置已于 2026-08-18 启用 pretrained seed 43；规则分仍同步保留为解释分，并在模型不可用或推理失败时接管输出。

旧开发 profile 的正类包含 B01-B04；当前推荐的 `observable_instability_b02_b04` profile 只将 B02-B04 作为异常步态正类，负类仍为 A01-A12 正常活动/hard negative。B01 `slow_walk` 单列为可观察功能风险 proxy，不能把慢速直接解释为步态不稳或临床异常。当前训练/运行输入契约固定为 4 FPS、4 秒、16 帧、最多 0.5 秒 gap、最少 10 个观测帧。

为了使用其他动作资源，训练侧保留五类语义动作共享编码器预训练，并按当前 split 重建。新 action-pretraining 数据有 2,619 个窗口，其中 SCF 贡献 138 个；迁移 encoder 仅用于 provisional gait candidate，不把坐站、近跌倒或跌倒标签改成步态正负类。旧 split 的预训练数据、checkpoint 和指标均不再复用。

输入来自 cleaned pose JSONL，优先使用：

- `x_smooth` / `y_smooth`
- `valid`
- `window_quality.usable_for_gait`

运行命令：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_gait.py \
  --input data/processed/fall_risk/pose_quality_y8n_v1/cleaned/le2i_home_01_video_1.jsonl \
  --output data/processed/fall_risk/features/home_01_video_1_gait.jsonl \
  --tcn-checkpoint reports/fall_risk/gait_observable_context_v2/splitv3-3342705-scfaux-v1/pretrained-seed43/best_model.pt \
  --tcn-window-frames 16
```

省略 `--tcn-checkpoint` 时命令仍可运行，但 `score_source=rule_fallback`、`fallback_reason=model_unavailable`。当前步态候选仅接受 observable-context v2 任务契约，其他历史 checkpoint 不进入运行链路。

输出每条记录表示一个步态分析窗口，核心字段包括：

| 字段 | 含义 |
|---|---|
| `gait_risk_score` | 0-1 当前主分支分数；有兼容模型时取 TCN 概率，否则取规则 fallback |
| `model_score` / `fallback_score` | 分别保留模型连续分和规则对照分，便于校准与失败分析 |
| `score_source` / `fallback_reason` | 标明实际使用 `tcn`、`rule_fallback` 或 `unavailable` 及降级原因 |
| `gait_stability_features` | 步速、步频、步幅、对称性、重心摆动、躯干倾角、路径和转身稳定性 proxy |
| `quality_coverage` | 可用帧比例、髋膝踝关键点覆盖率、插值比例、跳变数和质量不足标记 |
| `risk_factors` | 规则命中的可解释风险因子 |

当前八类核心指标为 `gait_speed_norm_per_sec`、`cadence_steps_per_min_proxy`、`stride_length_norm_proxy`、`left_right_asymmetry`、`center_lateral_sway`、`trunk_tilt_mean_deg`、`path_deviation` 和 `turn_instability_proxy`。规则分支还保留速度变异、停顿比例、脚踝运动范围、插值比例和关键点覆盖率等诊断字段。

局限：

- 当前 TCN 概率和规则分均未经过真实老人步态风险标签及外部测试校准。
- 旧 Fold A/B 会重分配 frozen test，已被训练入口拒绝；相关历史数值不能作为无泄漏证据。
- 当前 v3 校验报告为 `training_ready.action_type=false`；真实数据集开发训练和敏感性实验仅用于选模与错误分析，不是正式泛化证据。
- 单目归一化二维坐标没有相机标定，步速、步幅和转身稳定性只能是视觉 proxy，不能写成米/秒、厘米或临床量表结果。
- 相机视角、遮挡、下肢出画和跟踪 ID 切换会显著影响特征。
- 当前输出可用于工程链路、规则 baseline 和误差分析，不能解释为临床结论。

### 步态 TCN 实验收敛状态

旧 KINECAL 数据下载、准备和 baseline 入口已移除。当前只保留 observable-context v2 TCN；迁移 encoder seed 43 已按比赛交付决定进入默认步态主分支，但样本、来源域、冻结协议和老人域门禁仍未满足，不能表述为正式泛化验收。

## 坐站转换能力分析

坐站转换能力分析对应固定算法路线中的第 6 步：

```text
步态稳定性分析
  -> 坐站转换能力分析
  -> 近跌倒事件检测
```

2026-08-18 起，按最高执行人比赛期决定，默认运行分支切换为 `experimental_tcn`，使用原始 seed 42 checkpoint；它只在最新 cutoff 上推理，TCN 无事件或推理失败时回退规则，并在分支诊断中标明 `score_source`。该切换是 `development_provisional` 的比赛交付覆盖，不代表模型通过正式替换门禁；v2 seed 42 完整流 event F1 为 0.449、Recall 为 0.419，新 recall-balanced 采样复评为 0.380/0.507，均仍属 provisional。旧 Logistic 和预裁剪候选 TCN 训练/评估入口已移除。

实验链路配置字段：

- `sit_stand_runtime_mode`: `experimental_tcn`（比赛期默认）或 `rule_baseline`
- `sit_stand_model_path`: 已训练坐站 TCN checkpoint 路径
- `sit_stand_model_device` / `sit_stand_model_batch_size`: 推理设备和批大小

规则主路径从 cleaned pose JSONL 中读取肩、髋、膝、踝关键点，优先使用：

- `x_smooth` / `y_smooth`
- `valid`
- `is_jump_outlier`
- `core_keypoint_quality`
- `window_quality.usable_for_sit_stand`

腕部关键点只用于“疑似支撑使用”的弱 proxy；腕部缺失不会阻断基础坐站指标计算。

运行命令：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_sit_stand.py \
  --input data/processed/fall_risk/poses/home_01_video_1_poses_cleaned.jsonl \
  --output data/processed/fall_risk/features/home_01_video_1_sit_stand.jsonl
```

provisional 训练准备和复现入口：

```bash
conda run -n eldercare-ai python scripts/audit/audit_sit_stand_training.py

conda run -n eldercare-ai python scripts/prepare/prepare_sit_stand_event_dataset.py \
  --output-dir data/processed/fall_risk/sit_stand_event_v1/provisional-20260803-seed42-v3 \
  --allow-provisional

conda run -n eldercare-ai python scripts/train/train_sit_stand_continuous_tcn.py \
  --data data/processed/fall_risk/sit_stand_event_v1/provisional-20260803-seed42-v3/dataset.npz \
  --output-dir reports/fall_risk/sit_stand_continuous_tcn_v1 \
  --allow-provisional
```

输出每条记录表示一次候选坐站转换事件，核心字段包括：

| 字段 | 含义 |
|---|---|
| `transition_type` | `sit_to_stand`、`stand_to_sit` 或 `unknown_transition` |
| `sit_stand_risk_score` | 0-1 坐站局部风险分，越高表示坐站能力下降线索越强 |
| `duration` | 起身或坐下耗时，单位秒 |
| `failed_attempts` | 疑似起身失败次数 proxy |
| `trunk_forward_angle` | 转换过程中躯干前倾角 proxy |
| `post_stand_sway` | 起身后横向摇晃程度 proxy |
| `support_usage` | 疑似借助支撑 proxy 及证据字段 |
| `stabilization_time` | 起身后站稳所需时间，单位秒 |
| `sit_stand_features` | 髋部垂直位移、腿部伸展、评分组件等中间特征 |
| `quality_coverage` | 可用帧比例、关键点覆盖率、插值比例、跳变数和质量不足标记 |
| `risk_factors` | 规则命中的可解释风险因子 |

低质量窗口不会被当成高风险：当 `window_quality.usable_for_sit_stand` 不足、核心点覆盖率不足或有效帧过少时，模块输出 `sit_stand_risk_score = 0.0`，并在 `risk_factors` 中标记 `insufficient_sit_stand_quality`。

局部风险解释建议：

| `sit_stand_risk_score` | 局部含义 |
|---:|---|
| `< 0.25` | 未见明显坐站困难 |
| `0.25 - 0.50` | 轻微坐站异常 |
| `0.50 - 0.70` | 疑似坐站困难 |
| `>= 0.70` | 明显坐站困难，需融合层重点关注 |

局限：

- 该分数尚未经过真实老人坐站风险标签校准。
- 当前 E2 高分是来源偏斜的 clip-level provisional 结果，不是连续事件定位、临床功能或正式 test 证据。
- 归一化 2D 坐标不能直接代表真实世界距离、速度或角度。
- 相机视角、遮挡、下肢出画和跟踪 ID 切换会显著影响髋部高度、躯干角和摇晃指标。
- 没有家具、墙面、扶手检测或真实接触标签时，`support_usage` 只能表示“疑似支撑使用”，不能可靠判断扶了什么。
- `sit_stand_risk_score` 是坐站局部风险分，应交给后续融合层结合步态、近跌倒、个体基线和场景信号生成最终跌倒风险事件。

## 近跌倒事件检测

近跌倒事件检测对应固定算法路线中的第 7 步：

```text
坐站转换能力分析
  -> 近跌倒事件检测
  -> 个体化行为基线建模
```

当前实现是可解释规则/统计 baseline，不训练时序深度模型，不输出真正跌倒事件分数，也不直接输出最终 `risk_level` 或 `recommended_action`。它从 cleaned pose JSONL 中读取肩、髋、膝、踝关键点，优先使用：

- `x_smooth` / `y_smooth`
- `valid`
- `is_jump_outlier`
- `core_keypoint_quality`
- `window_quality.usable_for_near_fall`

训练侧另有隔离的“恢复后近跌倒确认”候选链：`publish_near_fall_labels_v3.py` 发布人工 v3 候选，`prepare_near_fall_event_dataset.py` 构建因果 `[T,10,8]` train/validation 数据，`train_near_fall_tcn.py` 训练 binary TCN。它尚未接入这里的规则主路径。现行真实 v3 事件监督门禁已经通过，当前 split 的三 seed开发训练已完成；train 已覆盖 5 个来源，但 validation 的困难负例仍只有快速可控坐下一类，不能替代连续背景和老人域验证。人工决策格式和门禁顺序见 `scripts/annotation/README.md`。

腕部关键点只用于“疑似支撑接触”的弱 proxy；腕部缺失不会阻断横向失衡、快速下沉恢复、急停恢复等基础近跌倒线索识别。

运行命令：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_near_fall.py \
  --input data/processed/fall_risk/poses/home_01_video_1_poses_cleaned.jsonl \
  --output data/processed/fall_risk/features/home_01_video_1_near_fall.jsonl
```

输出每条记录表示一个近跌倒候选事件或一个质量不足窗口，核心字段包括：

| 字段 | 含义 |
|---|---|
| `near_fall_event_score` | 0-1 近跌倒局部分数，越高表示短时近跌倒 proxy 线索越强 |
| `event_type` | `stumble_or_lateral_loss`、`rapid_descent_recovery`、`sudden_stop_recovery`、`support_contact_proxy`、`abnormal_crouch_recovery` 或 `unknown_near_fall` |
| `near_fall_features` | 横向速度/加速度、髋部下沉恢复、路径偏移、躯干角变化、急停恢复、支撑 proxy 和评分组件 |
| `quality_coverage` | 可用帧比例、核心点覆盖率、腕部覆盖率、插值比例、跳变数和质量不足标记 |
| `risk_factors` | 规则命中的机器可读可解释风险因子 |
| `evidence` | 结构化触发证据片段 |

低质量窗口不会被当成高风险：当 `window_quality.usable_for_near_fall` 不足、核心点覆盖率不足或有效帧过少时，模块输出 `near_fall_event_score = 0.0`、`event_type = unknown_near_fall`，并在 `risk_factors` 中标记 `insufficient_near_fall_quality`。

局部分数解释建议：

| `near_fall_event_score` | 局部含义 |
|---:|---|
| `< 0.25` | 未见明显近跌倒事件 |
| `0.25 - 0.50` | 轻微信号或弱证据，需要观察 |
| `0.50 - 0.70` | 疑似近跌倒事件，建议融合层关注 |
| `>= 0.70` | 近跌倒强证据，融合层可触发高风险逻辑 |

局限：

- 该分数尚未经过真实老人近跌倒标签校准。
- 横向速度、加速度、位移和躯干角都是归一化 2D 图像坐标 proxy，不能解释为真实世界物理量。
- 相机视角、遮挡、跟踪 ID 切换、正常转身/绕障/下蹲/挥手等动作都可能影响近跌倒 proxy。
- `support_contact_proxy` 没有家具、墙面或真实接触标签，只能作为弱证据，不能单独强判高风险。
- `near_fall_event_score` 是局部事件分数，可被 `FallRiskPipeline` 消费；近跌倒模块本身不直接调用融合流程。

## 个体化行为基线建模

个体化行为基线建模对应固定算法路线中的第 8 步：

```text
近跌倒事件检测
  -> 个体化行为基线建模
  -> 轻量风险融合模型 + 规则校准
```

当前实现是 `fall-personal-baseline-robust-v1` 规则/统计候选，不训练深度模型，也不直接输出最终 `risk_level`、`risk_score` 或 `recommended_action`。它只消费上游已经形成的完整日/小时周期记录，以 `person_id + camera_profile_id` 隔离 reference，并为每个当前周期按 `period_end < current.period_start` 重新选择历史；均值/标准差保留为诊断对照，主评分使用 median/MAD、分位数、相对变化和近期变化。10 秒 cleaned pose 窗口不是周期记录，实时链路在没有上游周期聚合结果时将该分支标为 `unavailable`，不会产生伪零分或增加融合覆盖率。

运行命令：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_baseline.py \
  --baseline-input data/processed/fall_risk/features/history_features.jsonl \
  --current-input data/processed/fall_risk/features/current_features.jsonl \
  --output data/processed/fall_risk/features/current_baseline.jsonl \
  --config configs/modules/fall_risk_baseline.yaml
```

每条输入必须声明 `record_type=fall_baseline_period_features`、`schema_version=fall-baseline-period-features-v1`、`completed=true`，并包含 `person_id`、`device_id`、`camera_profile_id`、`period_id`、带时区的 `period_start/period_end`、`timezone`、`valid_monitoring_hours` 和逐指标 `metric_quality`。次数类近跌倒与夜间活动先按有效监测小时归一化；空/`unknown` 身份、未完成周期、无时区时间或缺少相机域的记录失败关闭。同一身份/相机/周期的逐字相同记录去重，内容冲突则拒绝。

输出每条记录表示一个按天或按小时聚合的当前观测窗口，核心字段包括：

| 字段 | 含义 |
|---|---|
| `baseline_deviation_score` | 0-1 个体基线偏离分；无可用历史/指标时为 `null` |
| `baseline_state` | `none/cold/initial/stable/drift_suspected/recovery` |
| `baseline_confidence` | 由历史量、历史/当前质量形成的工程置信状态 |
| `available_metric_mask` | 逐指标可用性，不可用指标不参与归一化 |
| `metric_deviation_scores` | 逐指标偏离分；不可用项为 `null` |
| `legacy_mean_std_metric_scores` | 旧 mean/std 规则的逐指标对照分，不作为主评分 |
| `baseline_fusion_weight` | `none/cold=0`、`initial=0.25`、`stable=1.0` 的 provisional 内部融合权重 |
| `baseline_features` | 当前窗口聚合后的平均步速、坐站耗时、近跌倒频率、夜间活动、活动量和场景分布 |
| `baseline_reference` | median、MAD、winsorized mean、均值/标准差对照、分位数、曝光和质量摘要 |
| `deviation_factors` | 机器可读偏离因子，如 `gait_speed_drop_from_baseline` |
| `baseline_quality` | 历史天数、历史记录数、当前质量、基线置信和低样本/低质量标记 |

当前可解释偏离因子包括：

- `gait_speed_drop_from_baseline`
- `sit_stand_duration_increase_from_baseline`
- `near_fall_frequency_increase`
- `nighttime_activity_increase`
- `activity_volume_drop`
- `scene_region_pattern_shift`
- `insufficient_baseline_history`
- `reduced_baseline_quality`

低样本量和低质量数据不会被当成高风险：`none/cold` 阶段输出 `baseline_deviation_score=null`；`initial` 阶段按版本化配置限制分数；低质量历史/当前周期降置信并限制分数。`PersonalBaselineTracker` 另外提供快 EWMA、单向 CUSUM、连续异常冻结和三周期恢复候选规则；这些阈值均为 provisional，只由合成序列验证，不能解释为真实效果结论。

Phase 2 算法侧基础设施已经实现，但真实效果仍未验收：

- `configs/data/fall_risk_longitudinal_observation_schema_v1.json` 定义完成周期到未来 outcome 的连接契约；`configs/data/fall_risk_longitudinal_review_schema_v1.json` 固定人工复核证据。
- `scripts/split/build_fall_baseline_longitudinal_split.py` 按家庭/保守来源组做外层隔离，在每个 `person_id + camera_profile_id` 内严格先 reference、后 scoring。assignment 不保存 outcome ID 或风险真值，且绑定协议哈希。
- `scripts/evaluate/evaluate_fall_baseline_longitudinal.py` 强制无个人基线、mean/std、median/MAD、完整鲁棒候选四组使用完全相同的评分样本，并输出整体、人员/设备/相机/场景/质量/曝光分层、人员级 bootstrap 95% CI 和失败案例。
- validation 可以使用 provisional 协议；`data_status=formal` 要求 frozen 协议和 frozen split，test 还要求绑定 split、协议、预测、标签、commit、授权角色和 run ID 的保管人 release acknowledgement。代码只校验绑定关系，run ID 的单次消费由保管人流程负责。

真实门禁审计命令如下。当前预期退出码为 3，输出 `blocked`，因为 observation 0 条、`risk_labels` 0 条、subject profiles 0 人：

```bash
conda run -n eldercare-ai python scripts/split/build_fall_baseline_longitudinal_split.py \
  --output-dir /tmp/fall_baseline_longitudinal_gate
```

合成评估只能验证基础设施：

```bash
conda run -n eldercare-ai python scripts/evaluate/build_synthetic_fall_baseline_longitudinal_fixture.py \
  --output-dir /tmp/fall_baseline_longitudinal_synthetic

conda run -n eldercare-ai python scripts/evaluate/generate_fall_baseline_longitudinal_predictions.py \
  --observations /tmp/fall_baseline_longitudinal_synthetic/observations.validation.jsonl \
  --assignments /tmp/fall_baseline_longitudinal_synthetic/assignments.jsonl \
  --split /tmp/fall_baseline_longitudinal_synthetic/split.json \
  --partition validation \
  --output /tmp/fall_baseline_longitudinal_synthetic/replay-predictions.validation.jsonl

conda run -n eldercare-ai python scripts/evaluate/evaluate_fall_baseline_longitudinal.py \
  --observations /tmp/fall_baseline_longitudinal_synthetic/observations.validation.jsonl \
  --risk-labels /tmp/fall_baseline_longitudinal_synthetic/risk_labels.validation.jsonl \
  --assignments /tmp/fall_baseline_longitudinal_synthetic/assignments.jsonl \
  --split /tmp/fall_baseline_longitudinal_synthetic/split.json \
  --predictions /tmp/fall_baseline_longitudinal_synthetic/replay-predictions.validation.jsonl \
  --partition validation \
  --data-status synthetic \
  --output-dir /tmp/fall_baseline_longitudinal_bundle
```

局限：

- 当前基线特征是工程 proxy，尚未经过真实老人长期数据标定。
- 当前仓库没有生成上述周期记录的在线聚合调度；默认没有周期输入时实时 pose 链路中的个人基线保持 `unavailable`。上游聚合器可通过服务的 `POST /v1/monitoring/sessions/{session_id}/baseline-period` 提交已完成周期，之后结果才进入实时融合。
- Phase 2 真实数据、冻结 split 和真实四组消融仍未完成；该接入只打开已提交周期的 provisional 特征融合，不证明个人基线优于固定群体阈值，也不改变无周期/低质量输入时的规则 fallback。
- 平均步速、转身稳定性和活动量受相机角度、遮挡、采样策略和上游聚合方式影响。
- 场景模式变化只表示相对个人历史区域分布异常，不等同于危险场景判断。
- 本模块只输出 `baseline_deviation_score` 和解释性偏离因子，不能解释为医疗诊断结论。

## 最小可用版本输入字段

当前最小可用版本从结构化特征推理，不直接消费原始视频。视频、姿态关键点、IMU 等模态应先转成下列 0-1 风险特征：

| 字段 | 含义 |
|---|---|
| `gait_risk_score` | 步态不稳、步速/步幅异常等风险 |
| `sit_stand_risk_score` | 坐站转换困难、起身失败或耗时增加 |
| `near_fall_event_score` | 近跌倒、踉跄恢复、快速扶物等前置事件 |
| `baseline_deviation_score` | 相对个体行为基线的异常偏离 |
| `scene_risk_score` | 夜间、床边、浴室等场景风险 |
| `activity_rhythm_score` | 活动节律下降或昼夜活动模式改变 |
| `fall_event_score` | 疑似跌倒事件强触发分数 |
| `long_static_score` | 疑似跌倒后长时间静止强触发分数 |
| `keypoint_quality` | 姿态/检测质量，用于估计置信度 |
| `feature_coverage` | 上游特征覆盖率；缺省时按核心特征自动估计 |

样例输入见 `examples/features/fall_risk_sample.json`。该样例是工程验证用特征样例，不代表真实老人健康数据。

## 最小可用版本输出字段

入口函数：

```python
from elderly_monitoring.modules.fall_risk import FallRiskPipeline

event = FallRiskPipeline().predict_from_features(sample)
payload = event.to_dict()
```

输出遵循全局 `AlgorithmEvent` schema，不在本模块另建接口：

| 字段 | 含义 |
|---|---|
| `risk_score` | 0-1 风险分数，越高表示风险提示越强 |
| `risk_level` | 0-4 整数等级：0 正常，1 低风险，2 中风险，3 高风险，4 紧急风险 |
| `risk_factors` | 触发风险提示的解释性因子 |
| `confidence` | 0-1 置信度，综合关键点质量、特征覆盖率和风险强度 |
| `trigger_event` | 主触发事件，如 `near_fall`、`fall_or_long_static` |
| `recommended_action` | 建议动作编码，如 `notify_guardian`、`emergency_alert` |
| `model_version` | 当前规则评分卡版本 |

如果前端或报告需要 0-100 分或 `low / medium / high` 文案，应在展示层从现有 schema 映射，不改变算法事件接口。

## 当前算法

第一版使用可解释规则评分卡：

- 加权融合步态、坐站、近跌倒、个体基线、场景和活动节律风险。
- `near_fall_event_score >= 0.70` 触发 3 级高风险提示。
- `fall_event_score >= 0.80` 或 `long_static_score >= 0.80` 触发 4 级紧急风险提示。
- 置信度由 `keypoint_quality`、`feature_coverage` 和风险强度估计。

## 跌倒事件候选模型

当前保留连续因果输入张量、真实监督治理清单和 P0 审计基础设施；旧 candidate-clip TCN 训练/评估入口已退休。连续窗口 dataset、在线因果姿态清洗、冻结协议、老人域验证和 test 治理完成前，不进行正式训练或接入跌倒 TCN，规则主路径保持不变。

当前治理以 v3 `fall_event` 的 primary/auxiliary 正负事件为主监督，并额外接纳明确动作区间：`A01/A02/A04/A08/A10/A12`、`B01-B06`、`C01/C02` 作为普通/功能背景，`C03-C05` 作为恢复型近跌倒难负样本。未标注时间段仍不会被推断成负类。SCF_MVP_V1 的 298 条可用监督全部进入 train，其中 43 条恢复型近跌倒、53 条显式困难负例，其余为普通/功能背景；这三名主体的数据不能证明老人域泛化。

现行 v3 的 `training_ready.fall_event=true`，但这只解除事件监督结构门禁。旧 pilot 的 validation 保护组只有 9 个、来源以 NTU 为主，且未读取 test；其分数不能写成当前 split 结果、“已确认跌倒概率”、连续视频召回率、误报率或临床有效性。该 profile 的 `subtype_loss_weight=0.0`，方向头没有训练；报告显式返回 `subtype.status=not_trained`，不能从 checkpoint 推断跌倒方向。

`fall_event_continuous.py` 建立 `[T,17,20]` 因果输入：joint、motion/acceleration、bone、image y/bbox height、quality/valid/interpolated/jump/frame mask、实际 source `delta_t`、相对时间、有效 FPS、gap 和 track continuity。时刻 `t` 的 slot 只可使用 `<=t` 的观测；重复采样、缺帧、长 gap 或跳变不会伪造速度。现有 cleaned pose 的短 gap 插值可能使用未来帧，因此治理契约要求屏蔽插值坐标及其派生 motion；完成这一策略和真实连续窗口 builder 前，不应称为连续跌倒模型。

```bash
conda run -n eldercare-ai python scripts/prepare/govern_fall_event_training_data.py
```

## 判断频率与输出策略

`FallRiskPipeline.predict_from_features()` 仍保留单次特征推理 API；实时入口已经实现内存滑窗调度，使用以下策略：

| 场景 | 策略 |
|---|---|
| 轨迹和姿态关键点 | 同一次 YOLOv8-pose + ByteTrack 推理生成内存对象，默认最多 10 FPS；目标硬件仍需实测有效采样率不低于关键分支 6 FPS 门槛 |
| 动作识别 | 使用 1-2 秒滑窗滚动判断 |
| 风险融合 | 默认每 2 秒更新一次内部风险状态 |
| 普通状态 | 0 级不回调；已有 episode 的 `recovered/unresolved` 终止版本例外 |
| 风险等级升高或触发事件变化 | 立即输出事件 JSON |
| 近跌倒、疑似跌倒、长时间静止 | 不等待节流周期，立即输出 |

HTTP 入口为 `elderly_monitoring.service.app:app`，支持创建、查询、更新地址和幂等停止单路直播会话。流读取默认使用 FFmpeg，以兼容萤石 HEVC-over-FLV；系统 `ffprobe` 探测分辨率，系统 `ffmpeg` 解码为 BGR 并按 `ffmpeg_scale_width` 缩放。FFmpeg 与运行时都使用同一 `max_inference_fps` 上限：解码层只选择原始帧，不会为低帧率直播补造重复帧；运行时继续按接收时钟防止过密推理。该上限必须显著高于关键分支的 6 FPS 门槛，并在目标硬件上以最终有效采样率验收。超时或进程退出会主动清理子进程并进入现有重连流程。只有达到 `reconnect_stable_after_sec` 或 `reconnect_stable_after_frames` 的连接才会清零连续失败预算，防止可打开但立即 EOF 的短流永久重连；显式更新临时直播地址会重新开始预算。rawvideo 管道不提供源 PTS，因此该后端的 `source_pts_sec` 为 `null`。环境变量 `STREAM_READER_BACKEND`、`FFMPEG_SCALE_WIDTH`、`POSE_INFERENCE_SIZE`、`RECONNECT_STABLE_AFTER_SEC` 和 `RECONNECT_STABLE_AFTER_FRAMES` 可覆盖配置，非法后端会在启动时失败关闭；选择 FFmpeg 后端时 readiness 同时检查 `ffmpeg` 与 `ffprobe`。`pose_inference_size` 当前保持模型默认 640，变更后必须重新验证关键点覆盖率与四个关键分支的采样门禁。解码错误会脱敏当前直播地址后再写入会话状态。

真实视频全生命周期烟测：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_service_smoke.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --model models/yolov8n-pose.pt \
  --max-frames 30
```

业务后端尚未接入时，可以先用本机临时回调接收器验证真实直播算法链。直播地址只通过环境变量传入，报告不保存路径、查询签名或 Token：

```bash
EZVIZ_STREAM_URL='rtmp://example.invalid/live?temporary-signature' \
conda run -n eldercare-ai python scripts/collect/run_fall_live_smoke.py \
  --model models/yolov8n-pose.pt \
  --duration-sec 120 \
  --scene-region living_room \
  --report /tmp/ezviz_live_smoke.json
```

该命令分别报告持续出帧/重连状态与姿态、特征分支状态；四个关键分支必须在测试周期内达到 `valid`，没有风险事件本身不算失败。它不替代业务回调联调，也不构成算法准确率证据。2026-07-30 首次 RTMP 地址只返回约 14 帧后结束；2026-08-12 新授权 HTTPS-FLV 地址在修复后的默认 FFmpeg 链上通过 120 秒严格算法端烟测，单一 epoch、末端有效采样率 6.8514 FPS、四个关键分支有效且停止受控，详见[2026-07-30 失败记录](../../../reports/fall_risk/runtime/ezviz-live-smoke-20260730.md)和[2026-08-12 通过记录](../../../reports/fall_risk/runtime/ezviz-live-smoke-20260812.md)。业务后端回调、地址刷新、弱网和一小时固定硬件验收仍未完成。服务只接受可直接解码的 `rtsp`、`rtmp`、`http` 或 `https` 地址，不转换 `ezopen` 地址。

## 运行方式

```bash
conda run -n eldercare-ai python -m elderly_monitoring.inference.run_features \
  --module fall_risk \
  --input examples/features/fall_risk_sample.json
```

测试：

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_fall_risk_pipeline.py \
  tests/test_fall_risk_tracking.py \
  tests/test_fall_risk_pose.py \
  tests/test_fall_risk_pose_quality.py \
  tests/test_fall_risk_gait.py \
  tests/test_fall_risk_sit_stand.py \
  tests/test_fall_risk_near_fall.py \
  tests/test_fall_risk_baseline.py \
  -q
```

## 局限与升级路线

当前版本没有使用真实风险标签训练，不应解释为诊断模型或临床结论。LE2I/IMViA 样例可用于跌倒检测和跟踪链路验证，但它主要提供跌倒/非跌倒事件，不等同于长期跌倒风险标签。

后续接入真实数据时，建议固定 `predict_from_features()` 的输入输出契约：

1. 先从 `generated/v2/` 重新发布来源完整的根标签，使 formal 校验通过；再用现有 split builder 生成带稳定 `split_id` 的非空冻结划分。
2. 在冻结协议和盲测治理下，用当前规则评分卡作为可复现 baseline，报告事件级 Precision、Recall、F1、PR-AUC、合法分母下的误报指标和提前预警时间；合成 bundle 不得进入结果表。
3. 在不改变 `AlgorithmEvent` schema 的前提下，将内部 scorer 替换为 Logistic Regression、LightGBM 或姿态时序模型。
4. 保留 `risk_factors`，通过规则命中、特征贡献或 SHAP/重要性分数提供解释。
5. 对真实个人健康数据只保存脱敏 ID、聚合特征和授权记录，避免在样例文件和日志中写入个人身份信息。
