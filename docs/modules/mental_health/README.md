# 心理健康风险算法模块

本模块输出行为与睡眠变化的工程特征和独立心理健康风险事件，用于风险预警和人工复核，不输出医学诊断。当前已完成数据适配、日级聚合、个人基线、持续异常、风险评分和离线日级 CLI。

## ASR 与认知文本接入

中文 ASR 已作为独立算法模块落在 `src/elderly_monitoring/modules/asr/`，采用 Paraformer、FSMN-VAD 和 CT-Punc。认知 V3.3 通过 `modules.asr.api.transcribe()` 获取 `ASRTranscript v1`，再由 `cognitive_tasks.asr_adapter` 完成冻结文本规范化、`char_count`、`q_text` 和缺失模态掩码。ASR 不读取任务题目、用户、设备、数据库或留存字段，认知模块也不加载 FunASR 内部对象。

`elderly_monitoring.modules.asr.api:app` 提供可在 `eldercare-asr` 环境独立启动的 `POST /v1/asr/transcribe`，启动阶段预热进程级模型单例。对外只接受临时 URL 或 base64 的 WAV/MP3/M4A，范围为 3–60 秒和 50 MiB。首阶段以 S10 主动小测和同进程 Python 调用为主；视频通话音轨、任务条件建模和完整认知三模态推理属于后续接线内容。字段和联调方式见[ASR 与认知模块开发协作文档](guides/ASR与认知模块开发协作文档V1.0.md)。

跌倒风险与心理健康风险共享上游感知、人员身份和姿态质量数据，但分别评分并分别输出各自的 `AlgorithmEvent`。心理健康管线只产生 `module=mental_health` 的独立事件。

五个 standalone 专家、PersonalTrend、严格 OOF 融合、ART-001 完整模型包和 V3 正式推理接入均已完成。当前活动包为 `MH-20260802-013`，对外版本为 `mood-fusion-v3.3.3`。V3.3.4 优化候选 `MH-20260804-021` 未通过冻结晋级门，因此 ART-002/API-003 保持该活动包，不创建虚假 V3.3.4 在线包。2026-07-25 的三套数据审计是历史事实记录，不再单独决定当前训练角色；情绪与社交关注主线以项目级冻结文档、版本化模型包和实现级接口契约为准。

## V3.3.3 情绪与社交关注接口状态

独立 `mood_social` 包已经完成 V3.3.3 配置，以及 `mood_social_infer_request_v3`、`mood_social_infer_response_v3`、`mood_social_error_v1` 的严格 Pydantic 模型。CAM-001 已完成严格 V3 摄像头日级活动和场景化原始步态上游；MH-003 已完成统一日级/7 日特征 Schema、确定性映射、`feature_mask`、`day_mask` 和原始步态趋势上下文。`POST /v1/mental-health/mood-social/infer` 已接入版本化模型包与完整 V3 推理编排：

- 所有请求、响应和错误嵌套对象均拒绝未知字段；
- V1、V2 和其他请求 schema 返回结构化 422；
- 新路由的 401、422、500、503 使用独立 `mood_social_error_v1`；
- 有有效证据且模型包通过 manifest、`SHA256SUMS` 和 payload hash 校验时，使用 ART-001 `MH-20260802-013` 完成五专家、PersonalTrend、融合和非诊断输出；
- 没有有效证据时返回 HTTP 200、`available=false`，不调用旧 `MentalHealthRiskPipeline`、规则评分卡或 `/daily-risk`；
- 模型包缺失、损坏或 hash 不一致时返回结构化 `MODEL_ARTIFACT_UNAVAILABLE` 503；
- API-003 读取 ART-002 的不晋级决定并继续解析到 ART-001，对外模型版本保持 `mood-fusion-v3.3.3`；
- 旧 `/v1/mental-health/daily-risk` 及本文件后续所述评分卡/CLI 仍是 legacy 兼容能力，不是 V3.3.3 新生产结果。

V3.3.3 默认配置仍位于项目根目录 `configs/modules/mood_social_v3_3_3.yaml`。源码运行和普通安装运行都会从当前工作目录及其父目录发现该外部配置；部署需要显式指定根目录时使用 `ELDERLY_MONITORING_PROJECT_ROOT`，只覆盖配置文件时使用 `MOOD_SOCIAL_CONFIG_PATH`。模型相对目录始终按项目根目录解析。

实现位置：

```text
src/elderly_monitoring/modules/mental_health/mood_social/
src/elderly_monitoring/modules/mental_health/feature_extraction/activity/mood_social_v3.py
src/elderly_monitoring/service/app.py
src/elderly_monitoring/service/schemas.py
tests/test_mental_health_mood_social_api.py
tests/test_mental_health_mood_social_camera.py
tests/test_mental_health_mood_social_camera_gait.py
tests/test_mental_health_mood_social_feature_schema.py
tests/test_mental_health_mood_social_feature_mapper.py
src/elderly_monitoring/modules/mental_health/mood_social/experts/activity.py
configs/training/mood_social_activity_expert_v3_3_3.yaml
scripts/train_mood_social_activity_expert_v3_3_3.py
scripts/validate_mood_social_activity_expert_v3_3_3.py
tests/test_mental_health_mood_social_activity_expert.py
src/elderly_monitoring/modules/mental_health/mood_social/experts/sleep.py
configs/training/mood_social_sleep_expert_v3_3_3.yaml
scripts/train_mood_social_sleep_expert_v3_3_3.py
scripts/validate_mood_social_sleep_expert_v3_3_3.py
tests/test_mental_health_mood_social_sleep_expert.py
src/elderly_monitoring/modules/mental_health/mood_social/experts/joint.py
configs/training/mood_social_activity_sleep_joint_expert_v3_3_3.yaml
scripts/train_mood_social_activity_sleep_joint_expert_v3_3_3.py
scripts/validate_mood_social_activity_sleep_joint_expert_v3_3_3.py
tests/test_mental_health_mood_social_activity_sleep_joint_expert.py
src/elderly_monitoring/modules/mental_health/mood_social/experts/physiology.py
configs/training/mood_social_physiology_expert_v3_3_3.yaml
scripts/train_mood_social_physiology_expert_v3_3_3.py
scripts/validate_mood_social_physiology_expert_v3_3_3.py
tests/test_mental_health_mood_social_physiology_expert.py
```

统一特征入口为：

```python
from elderly_monitoring.modules.mental_health.mood_social import (
    feature_schema_manifest,
    map_mood_social_features,
)

manifest = feature_schema_manifest()
mapped = map_mood_social_features(request)
```

mapper 始终输出自然日 D-6 至 D 的七个槽，缺失日保持 null/mask 0，不用更早记录补位。活动比例按字段自己的有效日期和分钟分母合并；睡眠时刻使用跨午夜中点和圆周编码；夜间生理按字段有效夜聚合；S10 区分无记录、未完成观测和真实零。`feature_coverage` 保留用于专家置信度但不进入监督专家列，`valid_days` / `valid_nights` 仍按冻结口径进入专家。D-28 至 D 的 `camera_gait_metrics` 按 `date + camera_id + scene_version` 稳定保存到内部上下文，其键、四个原始指标及风险方向由同一 manifest 冻结；这里不计算 Q10/Q90 或个人归一化。当前 HTTP 路由尚未调用该 mapper，真实接线属于 API-001/API-002。

## V3.3.3 公开数据适配状态

DATA-002 至 DATA-006 已完成五个当前训练来源的严格 canonical 适配。所有适配器位于 `src/elderly_monitoring/datasets/adapters/`；历史 `modules.mental_health.validation.public_datasets` 路径仅保留兼容重导出。DATA-007 已生成冻结的参与者级嵌套五折 split；MODEL-001 至 MODEL-004 已分别生成 ActivityExpert、SleepExpert、ActivitySleepJointExpert 和 PhysiologyExpert 的训练折预处理、模型、校准器和完整 OOF，其余专家仍未训练。

| dataset_id | 来源 | 当前产物边界 |
|---|---|---|
| `psyche_d` | PSYCHE-D | 名义月份窗口；睡眠严格映射；步数只保留为训练折 `x_source` |
| `resilient` | RESILIENT | 首个有效传感器日起 14 个 Europe/London 日历日；步数只保留为训练折 `x_source` |
| `nhanes` | NHANES 2011-2014 DPQ+PAM | G/H 周期内连接；最多 7 个有效 PAM window；`M10VALUE` 只保留为训练折 `x_source` |
| `shenzhen` | 深圳社区老人 | 横断面 SocialContext；冲突键整组排除；不创建 `x_source` 或 ECDF |
| `nhanes_ssq_2005_2008` | NHANES 2005-2008 DPQ+SSQ | D/E 周期内连接；横断面 SocialContext；SSQ 只作受控来源字段；不创建 `x_source` 或 ECDF |

DATA-006 的 12 个 CDC 官方 XPT/codebook 文件冻结在工作区 `数据集/心理/NHANES-SSQ-2005-2008/`。source manifest SHA-256 为 `5d95ffd6e86bd83e2221d30a91fe727deac7eed1e6ed7744e1bf624569577baa`，来源集合 SHA-256 为 `48b47b9904cbe91e679046e10ede1340b4401c1cb674fcaf1ff047cccfd31608`。正式 canonical 为 3,150 人、156 列，`PHQ-9>=10` 阳性 185；只映射年龄段、性别、婚姻和教育四个严格同义的 `social_context` 字段，收入代码及全部 SSQ 字段不进入生产特征，也不映射为 S10 `social_contact.*`。

DATA-007 的正式 split 位于 `data/processed/mental_health/mood_social/v3.3.3/splits/split_manifest.json`。split ID 为 `mood-social-v3.3.3-participant-nested-5x5-seed-20260728-v1`，文件共 4,184,759 bytes，SHA-256 为 `e9915dbc590a6ea454c34d26558ac5866c57ee8a8e78f44cdcc79c859df77ee3`。它绑定五套 canonical、artifact manifest、mapping、逻辑 frame hash 和 MH-003 Schema，覆盖 15,361 名参与者、22,191 行、4,079 个阳性行。

划分以 `dataset_id::participant_id` 为唯一单元，先按“数据集、参与者窗口数、参与者阳性窗口数”分层，再按随机种子 20260728 的 SHA-256 顺序和确定性分层偏移轮转到五折；每个外层训练集合都单独生成内层五折交叉拟合分配。DATA-007 没有拟合 ECDF、填补器、编码器、标准化器、特征选择器、校准器或模型。

MODEL-001 运行 `MH-20260731-001` 只使用 `psyche_d`、`resilient`、`nhanes`，硬排除深圳和 `nhanes_ssq_2005_2008`。完整 OOF 为 13,714 行，其中 13,705 行可评价、9 行无活动证据固定 `expert_mask=0` 和概率 null。生产 bundle 使用 LightGBM + Isotonic，选中 `activity_volume_norm`、`relative_amplitude`、`interdaily_stability`、`intradaily_variability`、`activity_variability`、`valid_days`；不将 missing indicator、mask 或 `feature_coverage` 用作风险特征。

正式模型为 `models/mental_health/mood_social/v3.3.3/activity_expert.joblib`，SHA-256 `906a3cc2b74cdd3beeba446ca32885c7d2f0388be230ab793a2728bfd711bc3a`；同目录 manifest 绑定 split、Schema、输入、代码和配置。报告位于 `reports/mental_health/mood_social/MH-20260731-001/`，汇总校准 AUPRC `0.3372883872`、Brier `0.2668652581`。该工件尚未接入 HTTP，也不等于 ART-001 完整 V3.3.3 模型包。

MODEL-002 运行 `MH-20260731-002` 使用同三套来源并硬排除深圳自报睡眠和 `nhanes_ssq_2005_2008`。完整 OOF 为 13,714 行，其中 13,591 行可评价、123 行无睡眠证据固定 `expert_mask=0` 和概率 null。生产 bundle 使用 LightGBM + Isotonic，直接读取 canonical 同语义睡眠字段，并选中睡眠/在床时长、效率、入睡/起床/中点周期编码、碎片化、规律性和 `valid_nights` 共 12 个字段；`night_exit_count_mean` 因三来源均无训练证据被排除。

正式睡眠模型为 `models/mental_health/mood_social/v3.3.3/sleep_expert.joblib`，SHA-256 `966ba9ecd32ac18ac63ed9a6733a2e84f859a657cdb664bc2e034f0857056ade`。报告位于 `reports/mental_health/mood_social/MH-20260731-002/`，汇总校准 AUPRC `0.3315803561`、Brier `0.2587572644`。两次正式构建的模型、OOF、指标和搜索结果 hash 一致；独立校验 92 项通过。该工件同样尚未接入 HTTP，也不等于完整模型包。

MODEL-003 运行 `MH-20260731-003` 只使用上述三套来源同一 canonical 行的真实活动与睡眠证据。完整 OOF 为 13,714 行，其中 13,585 行联合可评价；129 行因至少一侧无证据固定 `expert_mask=0` 和概率 null。活动侧 ECDF、联合字段选择和填补均严格在对应训练折拟合；睡眠侧使用 canonical 同语义字段。联合输入不读取 standalone Activity/Sleep 概率，也不使用 missing indicator、mask 或 `feature_coverage`，且 `expert_mask=1` 必须要求活动和睡眠两侧均有真实模型输入。

正式联合模型为 `models/mental_health/mood_social/v3.3.3/activity_sleep_joint_expert.joblib`，SHA-256 `2152e7a62ba6382d64ddb1ca865410fa99e0acf9b73db54ed4c25622c46e70c8`。报告位于 `reports/mental_health/mood_social/MH-20260731-003/`，汇总校准 AUPRC `0.3954505791`、AUROC `0.6356911435`、Brier `0.2659286996`。模型、manifest、OOF、指标、搜索结果和配置经无代码变化覆盖重建后 hash 一致；独立校验 149 项通过。该工件仍未接入 HTTP，也不等于 ART-001 完整模型包。

MODEL-004 运行 `MH-20260731-004` 只使用 RESILIENT canonical 中真实存在的夜间生理字段；其他四个来源被硬排除，不能以全缺失填补行进入训练。完整 OOF 为 73 行，其中 71 行可评价、9 个可评价阳性，2 行无真实生理证据固定 `expert_mask=0` 和概率 null。各内外层训练集合独立完成有限非恒定字段选择、中位数填补、均值/标准差标准化、完整 20 候选 ElasticNet Logistic 搜索和 Platt 校准；两个无阳性的内层验证折保留 `AUPRC=null`，候选使用完整 pooled 严格内层 OOF 排名。最终选择 7 个真实字段，结构性全缺失的 `respiratory_abnormal_ratio`、`valid_nights` 被排除；missing indicator、mask、`feature_coverage`、活动和睡眠字段均不作为风险输入。

正式生理模型为 `models/mental_health/mood_social/v3.3.3/physiology_expert.joblib`，SHA-256 `1596e4ecae67319818c672f0cfc1ca7177fb4481f21ca66363154e8290c69a9e`；manifest SHA-256 为 `2afdb2320f748932ff25d017ff60dd61e54e6e55e5104c72a3e7fec1d483192f`。报告位于 `reports/mental_health/mood_social/MH-20260731-004/`，汇总校准 AUPRC `0.1528603721`、AUROC `0.4964157706`、Brier `0.2381505785`。固定 5000 次上限的收敛审计和小样本警告均已保存；无性能晋级门槛。格式化后覆盖重训，模型、OOF、指标和搜索结果保持不变；随后 32 个确定性核心文件经无代码变化覆盖重建 hash 一致，独立校验 142 项通过。该工件仍未接入 HTTP，也不等于 ART-001 完整模型包。

正式构建和独立校验入口：

```bash
conda run -n eldercare-ai python scripts/build_nhanes_ssq_v3_3_3.py
conda run -n eldercare-ai python scripts/validate_nhanes_ssq_v3_3_3.py
conda run -n eldercare-ai python scripts/build_mood_social_splits_v3_3_3.py
conda run -n eldercare-ai python scripts/validate_mood_social_splits_v3_3_3.py
conda run -n eldercare-ai python scripts/train_mood_social_activity_expert_v3_3_3.py
conda run -n eldercare-ai python scripts/validate_mood_social_activity_expert_v3_3_3.py
conda run -n eldercare-ai python scripts/train_mood_social_sleep_expert_v3_3_3.py
conda run -n eldercare-ai python scripts/validate_mood_social_sleep_expert_v3_3_3.py
conda run -n eldercare-ai python scripts/train_mood_social_activity_sleep_joint_expert_v3_3_3.py
conda run -n eldercare-ai python scripts/validate_mood_social_activity_sleep_joint_expert_v3_3_3.py
conda run -n eldercare-ai python scripts/train_mood_social_physiology_expert_v3_3_3.py
conda run -n eldercare-ai python scripts/validate_mood_social_physiology_expert_v3_3_3.py
```

数据产物位于 `data/processed/mental_health/mood_social/v3.3.3/`。DATA-006 校验器直接从冻结 XPT 重建键、筛选、标签、字段、mask、Arrow Schema 和 artifact manifest；DATA-007 校验器不导入 split 构建实现，独立重算输入绑定、全部外层/内层分配、覆盖、互斥、统计和完整性 hash；MODEL-001/002/003/004 独立校验器分别复核配置、模型、完整 OOF、指标、搜索结果、折内审计及推理硬门控，共 128/92/149/142 项。

## 数据来源与身份前提

行为输入复用现有 YOLOv8 Pose、ByteTrack、`PoseObservation` 和姿态质量控制结果，不创建第二套人体检测、姿态模型或实时循环。每条记录必须包含上游已经绑定的非空业务 `person_id`。`track_id` 只表示单路视频内关联，不能代替 `person_id`，也不能用于跨设备合并。

单设备记录可以省略 `device_id`；进入日级时间线的同一人员记录只要出现具名设备，其他记录也必须提供 `device_id`。同一时刻的多设备记录先按有效性、姿态质量和稳定设备键保留一个来源；异步区间仍重叠时，再按同样原则选择唯一来源。重叠会产生 `overlapping_device_observations` 标记，且不会直接制造场景转移。

## 绝对时间契约

行为记录必须通过以下一种方式得到事件时间：

- 带时区的 ISO-8601 `observed_at`；
- 带时区的 ISO-8601 `session_start_time` 加有限、非负的 `timestamp_sec`。

两者同时提供时以 `observed_at` 为准，并按配置容差核对推导时间。冲突记录保留在合理观测时长中，但不会进入有效观测时长。无时区、非法时间或只有相对秒的记录不会被猜测归入某个自然日；适配器会保留不可用标记，日级聚合跳过这些记录且不把标记归到其他日期。某人员完全没有有效绝对时间时快速失败。所有分桶先转换到 `aggregation.timezone`，默认 `Asia/Shanghai`；跨午夜区间在本地午夜拆分。

事件时间取当前人员和评估日最后一条合法输入的 `observed_at`。当没有合法事件时间时，CLI 要求显式传入带时区的 `evaluation_time`；它只可作为 `insufficient_data` 事件的时间回退，不能替代可评分风险结论的事件时间，也不会读取系统当前时间。

## 行为字段与计算

输入沿用命名关键点列表，每个关键点包含 `name`、归一化 `x/y` 和 `score`。若已经经过姿态质量控制，聚合器优先使用有效的 `x_smooth/y_smooth`，并保留上游质量标记。像素坐标、非有限坐标、低质量端点和被质量控制拒绝的记录不能进入有效时长。

聚合按 `person_id + date` 输出，计算规则如下：

- `observation_seconds`：只累加同一人员、同一设备、相邻且 `0 < delta <= max_gap_seconds` 的区间；不使用首尾时间差。
- `valid_observation_seconds`：区间两端质量合格、坐标有效、未被质量控制拒绝，且共同可见核心点数达标时才累加。
- `normalized_motion_proxy`：共同可见核心点的归一化欧氏位移中位数除以区间秒数。它是图像运动代理量，不是真实米/秒。
- `activity_volume`：有效区间的 `normalized_motion_proxy * delta_seconds` 之和。
- `active_ratio`：有效活动秒数除以有效观测秒数；分母为零时为 `None`。
- `nighttime_activity_ratio`：夜间有效活动秒数除以夜间有效观测秒数；默认夜间为本地时间 `[22:00, 06:00)`。
- `observation_coverage`：有效观测秒数除以合理观测秒数，不等同于后续评分阶段的模态覆盖率。
- `scene_region_distribution`：采用左端点场景的有效秒数分布；未知场景不产生转移。
- `scene_transition_count`：只统计质量有效、时间连续、无设备重叠的相邻记录场景变化。

当分母为零时，相应比例返回 `None` 并添加机器可读质量标记；缺失数据不填成正常值 `0`。

## 日间活动特征工程

`feature_extraction.activity` 同时提供严格 V3.3.3 生产格式 Python 上游和 V1 兼容路径。V3 路径使用以下入口：

```python
from elderly_monitoring.modules.mental_health.feature_extraction.activity import (
    aggregate_mood_social_camera_daily,
    aggregate_mood_social_camera_windows,
    extract_mood_social_camera_features,
)
```

V3 路径只计算本地 `[06:00, 18:00)` 的 10 秒窗口；严格区分完全无日记录的 `activity=null`、有记录但零有效覆盖的对象，以及有效观测下的真实 0。零覆盖对象中 `valid_daytime_detection_minutes=0`、小时覆盖全 0，其余日级活动语义和小时强度为 null、步态为空。原始帧覆盖按一秒支持区间并集计算，重复同一时刻的帧不能伪造覆盖；窗口必须达到 0.60 覆盖、至少两个不同时刻的有效样本，并同时具备中心和姿态运动量。

多摄像头先分别生成 `camera_id + scene_version` 候选，再按身份置信度、核心姿态有效率、跟踪置信度和稳定场景键选择唯一活动来源。低活动只由 `active_score <= 0.20` 定义，缺测、无效和非低活动槽都会中断连续片段，达到 30 分钟才计入片段字段。两组小时数组和日级汇总来自同一组去重窗口；活动峰值按分钟活动质量取最大值、并列取最早分钟。

步速、坐站、转身和姿态稳定性按 `camera_id + scene_version` 分别取日级中位数。图像速度不裁剪为 0–1，也不解释为米/秒；CAM-001 不计算个人 Q10/Q90 或 `walking_speed_norm_camera`。严格输出在返回前通过 `MoodSocialActivity` 自校验，并拒绝全天步数、V1 久坐/小时字段、`gait_speed_mps` 和预计算个人归一化值。完整输入、输出和自主冻结细节见源码同目录 [activity/README.md](../../../src/elderly_monitoring/modules/mental_health/feature_extraction/activity/README.md)。

### V1 兼容入口

旧入口继续供规则评分卡、ROI 调试和 `/v1/mental-health/daytime-activity` 使用。它不训练新的心理模型，而是把摄像头结构化结果先聚合为 10 秒活动窗口，再聚合为旧日间行为特征：

Python 调用入口：

```python
from elderly_monitoring.modules.mental_health.feature_extraction.activity import (
    aggregate_activity_windows,
    aggregate_daytime_activity_from_windows,
    extract_daytime_activity_features,
)

windows = aggregate_activity_windows(frame_or_second_records)
daily_activity = aggregate_daytime_activity_from_windows(windows, sleep_records=sleep_records)
```

帧/秒级输入可包含 `person_id`、带时区 `timestamp` 或 `observed_at`、`camera_id`、`bbox`、`bbox_confidence`、`keypoints`、`keypoint_confidence`、`zone`、`room`、`posture` 和 `tracking_confidence`。`bbox` 默认按 `[x, y, width, height]` 解释；如输入为 `[x1, y1, x2, y2]`，需显式传入 `bbox_format: "xyxy"`。

10 秒窗口输出 `active_score`、`motion_state`、`room`、`zone`、`posture`、`valid_detection_ratio` 和 `data_quality`。默认活动分公式为：

```text
active_score =
  0.55 * center_motion_score
+ 0.30 * pose_motion_score
+ 0.10 * zone_transition_score
+ 0.05 * posture_change_score
```

V1 日级输出包括 `daytime_active_minutes`、`weighted_daytime_activity`、`sedentary_*`、`daytime_bed_*`、`room_transition_count`、`bedroom_stay_ratio`、`outdoor_*`、`wake_activation_delay_minutes`、`routine_stability_score` 和进餐时段相关活动字段。低质量、离线、遮挡和身份不确定窗口只进入质量标记，不会被当作低运动。V1 输出不能直接提交到严格 V3 `activity` 对象。

## 睡眠适配

睡眠适配只是标准字段校验，不代表已经接入萤石或其他真实设备协议。

| 字段 | 规则 |
|---|---|
| `person_id` | 非空业务人员 ID |
| `date` | 严格 `YYYY-MM-DD`；也可用带时区 `observed_at` / `timestamp` 转换 |
| `sleep_onset_latency` | 分钟，有限数值，`0-720`，可缺失 |
| `night_awakenings` | 整数，`0-100`，可缺失 |
| `sleep_efficiency` | 比例，`0.0-1.0`，可缺失；不接受或转换百分数 |
| `device_source` | 可选非空字符串 |
| `quality_score` | 可选有限数值，`0.0-1.0` |
| `quality_flags` | 可选非空字符串列表 |

缺失的三个睡眠指标保持 `None` 并写入质量标记。非法单位、范围或类型会抛出包含记录号和字段名的 `MentalHealthDataError`，不会静默裁剪。

## 可选自评输入

自评记录使用 JSON 或 JSONL，每条记录包含稳定 `person_id`、严格 `YYYY-MM-DD` 日期，或带时区的 `observed_at` / `timestamp`。可选分数字段为 `social_withdrawal_score`、`negative_affect_score` 和 `self_report_risk_score`，均必须是 `0.0-1.0` 的有限数值；`manual_emergency_flag` 必须是显式布尔值。缺失字段保持不可用，不默认成 0。

## 徘徊行为线索

认知功能变化线索模块已提供 MVP 徘徊检测规则。输入是上游跟踪输出的中心点轨迹，输出是可解释的行为线索，不包含医学诊断：

```python
from elderly_monitoring.modules.mental_health import (
    aggregate_daily_wandering,
    detect_wandering_events,
)

events = detect_wandering_events(track_points)
daily = aggregate_daily_wandering(events, history_daily_features=history_days)
```

默认规则按 120 秒窗口和 30 秒步长计算路径效率、转向次数、重复网格比例、闭环得分、长条往返比例和跟踪质量。事件分为 `wandering_candidate` 与 `wandering_event`，形态分为 `pacing`、`lapping`、`random` 和 `mixed`。低质量轨迹只记录低置信度候选，不进入确认事件。日级输出包含夜间徘徊次数、总分钟数、形态计数、连续夜间次数和个人历史基线偏离量，可作为评分卡中的安全行为线索之一。

## 配置与调用

默认配置位于 `configs/modules/mental_health.yaml`：

```yaml
aggregation:
  timezone: Asia/Shanghai
  max_gap_seconds: 5.0
  timestamp_conflict_tolerance_seconds: 1.0
  min_keypoint_quality: 0.45
  min_common_core_keypoints: 4
  active_motion_threshold: 0.02
  night_start: "22:00"
  night_end: "06:00"

baseline:
  initial_days: 3
  stable_days: 7
  max_window_days: 14
  abnormal_score_threshold: 0.6

scoring:
  thresholds:
    level_1: 0.25
    level_2: 0.45
    level_3: 0.65
  min_persistent_days_for_level_3: 3
  passive_max_level: 3
```

这些值是算法原型的工程默认值，未经临床校准。Python 调用入口如下：

```python
from elderly_monitoring.modules.mental_health import (
    MentalHealthRiskPipeline,
    adapt_sleep_records,
    aggregate_daily_behavior,
    score_daily_mental_health,
)

daily_features = aggregate_daily_behavior(behavior_records)
sleep_records = adapt_sleep_records(raw_sleep_records)
baseline_features = score_daily_mental_health(history_days, current_days)
event = MentalHealthRiskPipeline().predict_from_features(baseline_features[0])
```

## 离线日级 CLI

历史行为和当前行为必须使用 JSONL，每个非空行是一个行为对象。睡眠和自评可使用单对象 JSON、对象数组 JSON 或 JSONL。睡眠/自评中与当前 `person_id + date` 匹配的记录进入当前日，其余记录作为历史模态；提供 `evaluation_time` 时，该自然日的可选模态也可建立当前人员日。

示例文件位于 `examples/features/mental_health_*`。以下命令只运行心理健康分支并输出 `module=mental_health` 事件：

```bash
PYTHONPATH=src conda run -n eldercare-ai python -m elderly_monitoring.inference.run_features \
  --module mental_health \
  --history-behavior examples/features/mental_health_history_behavior.jsonl \
  --current-behavior examples/features/mental_health_current_behavior.jsonl \
  --sleep examples/features/mental_health_sleep.json \
  --self-report examples/features/mental_health_self_report.json \
  --output /tmp/mental_health_daily.jsonl
```

省略 `--output` 时写到 stdout。当前行为没有合法事件时间时必须增加例如 `--evaluation-time 2026-07-04T18:30:00+08:00`。旧的单特征入口仍保留：

```bash
PYTHONPATH=src conda run -n eldercare-ai python -m elderly_monitoring.inference.run_features \
  --module mental_health \
  --input examples/features/mental_health_sample.json
```

日级输出按 `date + person_id` 排序，每个 `person_id + date` 一行：

```json
{"person_id":"p01","date":"2026-07-04","daily_features":{},"baseline_features":{},"event":{}}
```

`daily_features` 是行为、睡眠和自评的当日标准化结果；`baseline_features` 包含个人历史偏离、持续天数、覆盖率、质量与两个独立窗口；`event` 是统一 `AlgorithmEvent`，且 `module` 固定为 `mental_health`。输出不包含当前处理时间。相同输入、配置和代码产生字节级稳定 JSONL。所有文件会先完整解析、验证和计算；任一 JSONL 记录错误会报告文件、行号和字段，退出码非 0，输出路径采用临时文件原子替换，不留下部分结果。

## 个人基线与偏离分

基线按业务 `person_id` 和自然日独立建立，当前评估日不会进入自身基线。同一自然日重复输入只计为一个历史日；初始基线需要 3 个合格自然日，7 日后视为稳定，统计窗口最多保留最近 14 个合格自然日。

每个标量特征同时计算风险方向上的标准化差异、相对变化和分位数越界，取三者最大值。活动量和活跃比例只对下降评分；入睡潜伏期和夜间觉醒只对上升评分；睡眠效率只对下降评分；夜间活动比例和场景转移次数使用双侧偏离。零方差历史使用相对与绝对缩放下限，避免真实突变被零标准差掩盖。

`persistent_abnormal_days` 只累计达到异常阈值的连续、合格自然日。日历日期缺失或当日质量不合格会中断证据链，不会被当作正常日，也不会增加持续天数。`baseline_window` 记录历史参考范围，`evidence_window` 单独记录当前结论的连续异常范围。

## 缺失模态与风险评分

默认覆盖率分母只包含活动下降、睡眠扰动和规律性偏离三个预期特征，并按 YAML 中对应权重计算。可选的社交退缩、负面情感和自评分数仅在合法提供时参与风险评分；缺失值保持不可用，不以 0 分参与。风险总分只在当前可用特征之间重新归一化权重。

普通风险等级先按 YAML 阈值得到 0-3 级候选值，再依次应用覆盖率、基线成熟度和持续天数上限。纯摄像头与睡眠数据最高为 3 级。4 级只允许由达到阈值的合法自评风险分或显式 `manual_emergency_flag: true` 触发，3-4 级建议动作均为 `manual_review`。强证据不会被被动数据上限降级，但覆盖不足、基线不足和持续证据不足仍会降低置信度并写入 metadata。

没有任何可评分特征时，为兼容公共 schema 输出 `risk_score=0.0`，同时固定为 `risk_level=0`、`confidence=0.0` 和 `trigger_event=insufficient_data`；该兼容值不表示已经判断为正常。置信度表示特征覆盖、基线质量和持续性证据的支持程度，不表示医学结论正确率。

## 当前边界

- 不从摄像头推断负面情绪、社交退缩或医学状态。
- 不实现跨镜 ReID、人脸识别或身份数据库。
- 不实现真实睡眠设备协议、外部 API、数据库、推送或处置系统。
- 尚未经过真实设备数据、临床标签或临床有效性验证。

## Motor-Cognitive Gait Clues

`feature_extraction.gait_transfer` provides the cognitive-change clue module's gait feature entry points. It reuses upstream pose records and the fall-risk gait/sit-stand feature extractors, but produces independent `mental_health`-side daily features rather than fall-risk events or diagnosis labels.

Python entry points:

```python
from elderly_monitoring.modules.mental_health import (
    CognitiveGaitConfig,
    detect_turn_events,
    extract_cognitive_gait_features,
)

daily_features = extract_cognitive_gait_features(pose_records)
```

Daily output fields include `gait_speed_norm_per_sec`, optional `gait_speed_mps` when a per-scene meter scale is provided, `sit_stand_duration_seconds`, `turn_duration_seconds`, `turn_stability_score`, `gait_cycle_stability_score`, `motor_cognitive_clue_score`, event/window counts, quality flags, `diagnosis: false`, and `model_version`.

This optional `gait_speed_mps` belongs only to the legacy motor-cognitive clue path when an explicit scene scale exists. The V3.3.3 mood-social `camera_gait_metrics` producer is separate, never carries this field, and always preserves image-scale speed by `camera_id + scene_version`.

The current implementation is an engineering baseline for behavioral trend clues. It does not infer dementia, cognitive impairment, depression, or any medical diagnosis.
