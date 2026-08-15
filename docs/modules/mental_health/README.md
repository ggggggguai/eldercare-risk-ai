# 心理健康风险算法模块

更新时间：2026-08-15

本模块输出行为与睡眠变化的工程特征和独立心理健康风险事件，用于风险预警和人工复核，不输出医学诊断。心理健康与跌倒模块共享上游感知和 `AlgorithmEvent` 字段契约，但分别评分、验证和输出；本模块只产生 `module=mental_health` 事件。

- 现行徘徊路线：[徘徊识别技术文档2](plans/徘徊识别技术文档2.md)
- 拍摄手册：[萤石 C6C 办公室视频拍摄与标注手册](拍摄与标注/萤石C6C办公室视频拍摄与标注手册.md)
- CVAT 标注员教程：[Windows 本地部署 CVAT 徘徊识别标注员教程](拍摄与标注/Windows本地部署CVAT徘徊识别标注员教程.md)
- 当前任务：[任务表](../../tasks/README.md)
- 跨角色接口：[徘徊模块协作交接与职责边界](徘徊模块协作交接与职责边界.md)
- 旧完整方案：[历史归档](../../../文档/徘徊样行为识别技术方案（历史归档-2026-08-12）.md)

本文只维护已经实现的能力和当前限制，不保存下一步路线、分支、逐次测试计数或制品哈希；精确实验事实以对应 `reports/mental_health/` 报告为准。

## 1. 已实现的通用心理健康能力

当前代码已经支持：

- 行为、睡眠和自评数据适配；
- 日级聚合与缺失模态处理；
- 个人基线与持续异常判断；
- 独立心理健康风险评分；
- 离线日级 CLI；
- `AlgorithmEvent(module=mental_health)` 输出。

尚未建立在线日级调度、真实萤石会话接入或业务回调闭环。徘徊轨迹分类也尚未接入日级心理健康主链。

## 2. 已实现的徘徊研发底座

徘徊专项目前具备以下隔离能力：

1. WanderingPatterns 和 SmartCare 的来源适配、结构校验与统一样本格式；
2. 来源专用 train/validation/test 分配，SmartCare official 保持 sealed；
3. 固定 80 点、14 通道、mask-aware 的轨迹预处理与质量状态；
4. 26 维手工拓扑特征 Random Forest 对照；
5. 纯 TCN 对照；
6. bbox-bottom 轨迹的合成摄像头适配、Camera QC 和离线推理链；
7. 合成跟踪污染与视觉人工复核证据；
8. `TopoWanderMPT` 的 TCN + semantic patch + relation-aware Transformer 前向结构；
9. Step10-A 五 seed pair-aware 预训练候选；
10. `TopoWanderMPT` M0 的共享 trunk + binary/subtype 双头监督训练、joint evaluator、overfit smoke、best/last checkpoint、完整 checkpoint 加载、fresh reload 和 validation 报告；
11. M0-S 固定三种子训练、显式 seed 绑定、崩溃窗口恢复、逐 seed fresh reload、RF/TCN 同种子 paired 对照、成本汇总和 development primary-seed 冻结；
12. M0-RH v3 的受信 frozen-WP 正式计分入口、固定 CPU 运行时、accessor 前 source/archive/双身份 preflight、拒绝覆盖和原子提交协议；
13. M0-RS 固定 primary 候选的一次 WP public-holdout 计分、正式六文件固化、独立指标复算和预注册门槛判定；
14. M0-CAM-E 将同一固定 primary 通过独立 wrapper 接入 Step7 camera adapter/QC/prepared arrays，形成固定 40 秒逐窗概率和同预测类别窗口事后聚合。该路径现在明确为 legacy long-context shape diagnostic；`camera_episode.py` 尚不是模型前的行为分段器。
15. M0-CAM-RD-F/RD-F2 已完成独立 authorized-development 入口加固：三个 CLI receipt-first、primary loader threshold/evidence envelope 兼容、实际 tracking↔C3 participant/session/setup/clock 唯一绑定、session-aware eligible negative person-hours、truth mask/abstention/unavailable/error/miss 分离，以及最大基数优先的确定性 episode matching；非 fixture Python 调用会在最早入口拒绝 hooks/fake loader，公开逐窗 predictor 固定为 synthetic-only，authorized scope 只由 receipt-gated controller 的未导出内部路径产生；
16. M0-CAM-C01-PREP/F/F2 已完成正式 collection/receipt 场景下的 C1 tracking/sidecar 准备和交接加固；这些工具保留给数据冻结、跨机交接和 sealed 评估，不继续作为本机 episode-first 小样开发的前置工作。
17. M0-CAM-PORTABLE 已有运行资产恢复原型，F1 后续复审仍留有跨机恢复/rollback 等发布缺口，状态为 `rework_deferred`。它不阻塞当前机器上的 EP1/EP2；到实际跨机部署或正式交付前集中关闭。
18. M0-CAM-MVP-1 已复用同一 fixed-primary builder 形成薄的 session-level 产品切片：单命令 synthetic CLI 输出完整 `primary/`、`wandering-session-evidence-v1` 和产品 manifest，保留三态、QC、episode 数量与 duration sum，并固定 person/risk/alert/diagnosis/`AlgorithmEvent` 为空。F1 已关闭后续复审发现的 descriptor-aware missing/extra、嵌套 probability/summary、非空决策字段与 final 顶层集合缺口，当前为 `implementation_complete + acceptance=passed`；证据仍仅为 `synthetic_contract_only`。
19. M0-CAM-MVP-2S 已形成独立 `WanderingDailySummary`：完整回读一个或多个 MVP-1-F1 product final/primary，以显式 `wandering-daily-binding-v1` 绑定 synthetic person/session/timezone/track/presence，按 epoch elapsed seconds 与 IANA 自然日汇总 presence、三态 coverage、status overlap 和跨日 episode。final 固定为 canonical `daily_summary.jsonl + manifest.json`，person binding、baseline、risk/action/diagnosis/`AlgorithmEvent` 均保持 false/null；没有改动 generic daily/baseline/pipeline/config 或 fixed primary producer。
20. M0-CAM-MVP-3S 已形成独立 `WanderingBaselineProfilePreview` v2 builder：完整回读一个或多个 MVP-2S final，按 synthetic person 隔离，使用 3/7 日 readiness、最近 14 个机械可用日和 fixed reference stats 输出 canonical `baseline_profiles.jsonl + manifest.json`；timezone/candidate/model/class/threshold/calibration/episode/night/config/duration identity 漂移会返回 `baseline_version_reset_required`。F1 已真实绑定 marker 前冻结 MVP-2S producer prefix，并让 profile 携带 exact selected `reference_days`，public loader 会重算 window/readiness 与全部统计；v1 final 明确需要重新审计，不能进入 v2 downstream。当前接受状态为 `passed`。person binding、真实 baseline/deviation、risk/action/diagnosis/`AlgorithmEvent` 均保持 false/null；没有改动 generic baseline/daily/pipeline/config、fixed candidate 或 PORTABLE。
21. M0-CAM-MVP-3D 已形成独立 `WanderingBaselineDeviationPreview` builder 与 public loader：完整回读一个 MVP-3S v2 baseline final 和一个或多个 MVP-2S observation daily final，强制唯一 person/timezone profile、strictly-prior reference date、reference/observation manifest 隔离和完整 identity 一致；九项 metric 只输出 observation value、reference median/P90 与 signed delta。warming、无 coverage、current null 或 reference count 0 均传播 null，candidate-duration rate 允许大于 3600。producer prefix 与 consumer-only loader 从首版分隔，final 固定为 canonical `baseline_deviation_preview.jsonl + manifest.json`；接受状态为 `passed`。不输出 abnormal、综合分、概率、风险、动作或 `AlgorithmEvent`，也未接入 generic pipeline。
22. M0-CAM-EP1A 已形成独立 `wandering_camera_episode_v1.yaml`、episode inference/importer 和两个 CLI。入口支持 truth-free 简化 boundary JSONL、CVAT for video 1.1 XML 和显式 `target_track_id` 的 whole-clip 短片；CVAT track ID 与机器 track ID 分层，boundary/truth 分文件。episode 内复用既有技术 track-break、0.5 秒 bucket、短缺口质量权重、bbox-height compensation、`prepare_camera_window()`、train-only stats 和同一 frozen candidate。`ready/unavailable/boundary_uncertain/inference_error` 分开，旧 40 秒配置和 `camera_episode.py` 未改语义。

2026-08-15 的 EP1A 正式入口回归使用负责人提供的本地 development 样片派生 tracking：D01 12 条 whole-clip 均为 `ready/direct`，S00 三个 CVAT episode（0–18.60、18.60–39.73、39.73–44.73 秒）均为 `ready/direct`，逐条概率与先前临时诊断一致。S00 0–40 秒 legacy 路径仍单独为 lapping-like。成绩名称仅为 **oracle-boundary shape classification**；它证明给定人工/whole-clip 边界后的工程分类行为，不是 automatic boundary、现实 camera accuracy、告警或临床证据。详见 [EP1A 报告](../../../reports/mental_health/wandering_camera_episode_v1/README.md)。

当前 EP1B 先消费本机已有 P01 4 条、L01 3 条、R01 3 条及 H01–H05 9 条剪辑，补齐实际画面复核、tracking、truth 和 EP1A 输出，再用独立批量 evaluator 计算 all-eligible 主指标与 ready-only 条件指标。除 S00 外这些剪辑当前尚无 XML/正式 episode bundle，目录名不能直接充当真值。实现与完成门见 [EP1B 执行任务书](plans/M0-CAM-EP1B执行任务书.md)。

这些能力彼此隔离，不自动表示已经形成可部署的徘徊模型。

## 3. 当前数据边界

现有 ready 数据口径：

| 分区 | WanderingPatterns | SmartCare | 用途 |
| --- | ---: | ---: | --- |
| train | 1,120 | 137 | 模型拟合 |
| validation | 240 | 38 | 开发评估和早停 |
| WP public holdout（历史名称 frozen WP test，同一 240 条） | 240 | 0 | 固定候选级计分，不用于日常调参 |
| SmartCare official | 0 | 20 | sealed external test，不用于训练、调参或误差分析 |

另外 15 条 SmartCare train 因有效点过少保持 `unavailable`，不进入模型 tensor、loss 或指标。WanderingPatterns 缺少可用 participant/session 分组字段，公开数据结果存在近邻和泛化边界，不能等同于真实摄像头或未见人员效果。

详细来源、split 和预处理证据：

- [步骤 2 转换与人工复核](../../../reports/mental_health/wandering_step2/README.md)
- [步骤 3 固定 split](../../../reports/mental_health/wandering_step3/README.md)
- [步骤 4 预处理与人工复核](../../../reports/mental_health/wandering_step4/README.md)

## 4. 已有性能参照

固定 validation 上的已保存参照为：

| 模型 | WP 四分类 macro-F1 | WP+SmartCare 二分类 source-equal macro-F1 | 定位 |
| --- | ---: | ---: | --- |
| 26 维特征 + Random Forest | 约 0.9726 | 约 0.9507 | 四分类强基线 |
| 纯 TCN | 约 0.9411 | 约 0.9967 | 二分类强基线 |
| TopoWander-MPT M0，seed 20260731 | 0.995833 | 1.000000 | 单 seed 监督候选，development validation |
| TopoWander-MPT M0-S，3 seed | 0.987497 ± 0.000000 | 0.992774 ± 0.006312 | 固定三 seed，primary `20260731` 已冻结，development validation |

这些指标来自不同任务口径，不能相互替代，也不能证明摄像头效果。详细指标、逐类结果和预测见：

M0-RS 固定 primary 在 240 条 WP public holdout 上的 WP 四分类 macro-F1 为 `0.983331`，WP 二分类 macro-F1 为 `0.989010`，六个已报告 recall 均不低于 `0.966667`，预注册状态为 `target_met`。这里的 binary 是 WP-only，不是上表的 WP+SmartCare source-equal 口径。

- [步骤 5 Random Forest 报告](../../../reports/mental_health/wandering_step5/README.md)
- [步骤 6 纯 TCN 报告](../../../reports/mental_health/wandering_step6/README.md)
- [TopoWander-MPT M0 监督训练报告](../../../reports/mental_health/wandering_performance/m0_seed20260731_v1/README.md)
- [TopoWander-MPT M0-S 三种子稳定性报告](../../../reports/mental_health/wandering_performance/m0s_three_seed_stability_v1/README.md)
- [TopoWander-MPT M0-R Release-Prep v2 报告](../../../reports/mental_health/wandering_performance/m0r_release_prep_v1/README.md)
- [TopoWander-MPT M0-RH score-entry hardening v3 报告](../../../reports/mental_health/wandering_performance/m0r_score_entry_hardening_v1/README.md)
- [TopoWander-MPT M0-RS fixed WP public-holdout score 报告](../../../reports/mental_health/wandering_performance/m0rs_public_holdout_score_report_v1/README.md)
- [TopoWander-MPT M0-CAM-E / M0-CAM-H primary camera 工程报告](../../../reports/mental_health/wandering_m0cam_engineering_v1/README.md)
- [TopoWander-MPT M0-CAM-RD camera development 数据工具报告](../../../reports/mental_health/wandering_camera_data_readiness_v1/README.md)
- [TopoWander-MPT M0-CAM-RD-F development entry/evaluator 加固报告](../../../reports/mental_health/wandering_camera_rd_f_v1/README.md)
- [TopoWander-MPT M0-CAM-MVP-1 session evidence 原型报告](../../../reports/mental_health/wandering_camera_mvp1_v1/README.md)
- [TopoWander-MPT M0-CAM-MVP-2S synthetic 日级摘要报告](../../../reports/mental_health/wandering_camera_mvp2s_v1/README.md)
- [TopoWander-MPT M0-CAM-MVP-3S synthetic 个人基线预览报告](../../../reports/mental_health/wandering_camera_mvp3s_v1/README.md)
- [M0-CAM-MVP-3S 独立复审](../../../reports/mental_health/wandering_camera_mvp3s_v1/MVP3S_REAUDIT_20260814.md)
- [M0-CAM-MVP-3D synthetic baseline deviation preview 报告](../../../reports/mental_health/wandering_camera_mvp3d_v1/README.md)

合成 Camera QC 与污染压力只用于工程兼容性和鲁棒性诊断：

- [步骤 7 bbox-only 合成摄像头链](../../../reports/mental_health/wandering_step7/README.md)
- [步骤 8 合成污染兼容性](../../../reports/mental_health/wandering_step8/visual_review/v3/README.md)

## 5. TopoWander-MPT 当前证据

`TopoWanderMPT` 前向主体已经实现，包含 mask-aware TCN、semantic patch、relation-aware Transformer、binary/subtype 分类头和 projection 分支。历史 M0 使用同结构 scratch 初始化完成单 seed 端到端监督训练：1,257 条 train、278 条 validation，best epoch 13，last epoch 21；fresh reload 的 WP 四分类 / 来源等权二分类 macro-F1 为 `0.995833 / 1.000000`。该原始报告保持不变，作为 M0-S 的历史输入。

M0-S 用同一最终源码按固定顺序运行 seeds `20260731/20260801/20260802`，三次 WP 四分类 macro-F1 均为 `0.987497`；来源等权二分类 macro-F1 为 `0.997238 / 0.983848 / 0.997238`，mean `0.992774`、population std `0.006312`，所有已报告类别 recall 的三 seed 最小值为 `0.941176`。三次均通过两项 macro-F1 `>=0.95` 和全类别 recall `>=0.90` 门禁，primary seed 按预注册规则冻结为 `20260731`，未执行 M1。四分类错误 intersection=union=3；二分类错误 intersection=1、union=2。RF/TCN 对照直接复用同种子已保存 validation predictions，没有重训。

M0-R Release-Prep v2 已把 primary `seed=20260731 / best epoch=5` 导出为本机 local-only 的紧凑 inference bundle，并实现外部 manifest SHA 绑定、完整 model state 检查、CPU/`eval`/`inference_mode` loader 与 phase-aware WP-only evaluator。240 条 WP validation 的 sample ID/label、逐样本概率和 WP-only 指标相对 M0-S reference 均为零差异；精确 manifest 身份见 M0-R 报告。该 v2 只作为 Release-Prep 证据，没有执行 WP public-holdout 候选推理或计分，也没有重训、改权重、改阈值或执行 M1。

M0-RH v3 已在保持相同 M0-S model/forward/performance bytes 的前提下补齐正式 score 入口。唯一 production test 命令为 `score-frozen-wp`：不接收外部 records/split/batch/threshold/thread 参数，先验证 final output 不存在、外部 manifest SHA、bundle descriptors、active execution source、v3 source archive、training/release 双身份、RF/upstream descriptors 和 CPU `8/1` + batch64 runtime，再内部调用受信 `BUNDLE_MODE_FROZEN_WP_TEST` accessor。`evaluate-records` 已限制为 validation-only。v3 development parity 的 240 条 ID/label、概率、WP-only 指标与冻结 reference 均为零差异。M0-RH 新增 controller/候选路径没有调用 accessor 做 public-holdout 推理或计分；完整 wandering 回归中的既有 accessor contract 测试仅解析 access/count/schema，其记录未进入候选推理、计分、制品或设计反馈。canonical test records/order SHA 在 M0-RH 结束时保持未生成；实时授权和下一项任务只看任务表。

M0-RS 已在负责人明确授权后使用同一 primary 和唯一正式入口完成一次 WP public-holdout 计分。正式六文件通过同文件系统 staging 验证后原子提交，canonical records/order SHA 已形成；独立复算与 artifact descriptor 全部一致，结果为 `wandering_m0rs_public_holdout_scored`、`performance_status=target_met`。该结果不会用于回调模型、阈值、split、标签或协议；其后的 M0-CAM-E 仅复用该固定候选做 camera 工程接线。

M0-CAM-E 已把固定 primary candidate 接入既有 Step7 adapter、Camera QC 和 prepared-array 路径，并以 CPU float32 `eval`/`inference_mode` 直接 forward ready 40 秒 window。逐窗输出保留 scope/tracklet/time/QC、binary/subtype/four-class 未校准概率和不可用零调用语义；所谓 episode candidate 只在相同 scope、parent tracklet 与预测 shape 内合并这些 window，不是 episode-first。它不输出 alert、risk 或 `AlgorithmEvent`，只属于 synthetic/legacy camera 工程证据。

M0-CAM-RD/RD-F/F2 已实现正式 collection/annotation validator、session-aware person-hours、episode matching 和 evaluator 原语。正式 C0-C3 数据包尚未冻结，但用户提供的 D01/S00 已用于本机 development diagnostic；这些结果不能直接晋升为正式 camera 性能。EP1 只复用有收益的 importer、matching 和 metric 原语，不继续扩展 receipt/provenance 审计。

M0-CAM-PORTABLE-F1 的执行时证据为聚焦 `100 passed`、受影响窄测 `79 passed, 4 subtests passed`、完整 camera `297 passed`、七文件 DrvFS `git archive` overlay 的 `unshare -Urn` 回归 `100 passed`，以及固定 candidate manifest-bound CPU/eval safe-load 的 forward/network attempt 0。完整 camera 回归包含 synthetic tracking 上的真实 fixed-candidate forward 和 QC，只有 production preflight 与独立 safe-load可以表述为 forward=0。后续复审证明这些数字没有覆盖 4 个可构造 P0，因此当前是 `m0cam_portable_f1_acceptance_status=rework_deferred`、`portable_software_audit_status=rework_required`、`portable_review_blocker=post_f1_findings_open`；formal 继续 blocked。这些仍不产生 camera 性能证据。见 [PORTABLE 执行时报告](../../../reports/mental_health/wandering_camera_portable_v1/README.md)和 [F1 后续复审](../../../reports/mental_health/wandering_camera_portable_v1/F1_REAUDIT_20260814.md)。

独立审计从 278 条 development predictions 复算出了相同 validation 指标，并确认 materialized train/validation 的 sample ID、parent ID 和精确特征无重合。边界口径需要准确理解：共享 preprocessing container 会为完整性解析其中的 WP test 行，M0-S 的训练、validation、fresh reload 与选点路径没有使用 test；SmartCare official/raw 与 sealed camera 没有进入本次模型流程。仓库契约测试会读取 frozen accessor，Step5/Step6 也已对 WP test 做过历史评分，因此 M0-RS 只能称为固定候选的 public-holdout 计分，而不是项目级首次盲测。WanderingPatterns 没有可靠 participant/session 分组，且近邻审计有 4,054 对 `<0.05` 跨分区形状近邻，所以当前高分只属于固定公开轨迹 benchmark。

Step10-A 的五 seed pair-aware 预训练仍只在外部活动工作树形成候选，其 total proxy 相对初始值的平均改善约为 0.0437%，没有直接训练分类头，也没有参与本次 M0。

因此目前只能表述为：

- M0-S 已形成可训练、可加载、可重载和可报告的公开轨迹 development 冻结候选；每个 run 都保存 seed-bound model/AdamW/CPU RNG checkpoint、history、fresh predictions 和 metrics；
- resume 已以“完整 epoch checkpoint 已提交但 history/latest/best 仍落后”的故障注入验证，恢复逻辑以连续且校验通过的 checkpoint 为事实源重建可变索引；正式配置固定 CPU，并明确不声称 CUDA resume；
- 外部工作树中有一个未来可作为初始化对照的 encoder/trunk，但当前 checkout 没有稳定 bundle 路径或 trunk-only loader，尚未证明它能提高 accuracy、macro-F1 或 recall；
- Step10-B 和旧双环境逐字节复现不再阻塞监督性能开发；
- M0-S 使用同结构、同初始化方法的 seed-controlled scratch 初始化；A-init 仅在外部候选已登记并补齐 trunk-only loader 后作为可选对照。

历史前向和 runtime 证据：

- [步骤 9/9a 前向契约](../../../reports/mental_health/wandering_step9/README.md)
- [步骤 10 runtime 历史记录](../../../reports/mental_health/wandering_step10_runtime/README.md)

## 6. 当前能力限制

以下能力尚未完成或未获得证据：

- automatic/semiautomatic episode boundary producer 尚未实现。技术 tracklet、转弯和行为 episode 不是同一概念；不能靠调旧 `episode_merge_gap_seconds` 解决；
- oracle-boundary inference 与 legacy 40 秒 diagnostic 已在 EP1A 输出和报告中分离；automatic boundary+shape 尚未实现，四类 camera episode F1、boundary 指标和 candidate false activations/person-hour 仍需补齐；
- 当前时间通道被预处理、模型配置和 forward 共同关闭。duration/speed/dwell/repetition 只能先作为模型外元数据；若以后进入模型，必须建立新配置、统计和训练候选；
- 除 D01/S00 oracle-boundary development 回归外，已有 P/L/R/H 原始剪辑尚未完成实际画面复核、tracking、truth 和批量评估，仍缺足量可评分 pacing/lapping/random、连续自然负例和长时徘徊；不能报告 camera 95%、真实老人效果或最终 FAR；
- M0-CAM-RD-F2、M0-CAM-C01-PREP、C01-F 与 C01-F2 已完成；[M0-CAM-MVP-1](plans/M0-CAM-MVP-1执行任务书.md)、[MVP-1-F1](plans/M0-CAM-MVP-1-F1执行任务书.md)、synthetic [MVP-2S](plans/M0-CAM-MVP-2S执行任务书.md)、[MVP-3S/F1](plans/M0-CAM-MVP-3S-F1执行任务书.md) 与 [MVP-3D](plans/M0-CAM-MVP-3D执行任务书.md) 已通过各自软件契约门。MVP-2S/3S/3D 仍固定 `person_binding_verified=false`、真实 baseline/deviation/risk/event 为空；产品代码线停在 deviation preview，真实 MVP-2R/3R/4 未启动。PORTABLE-F1 接受状态仍为 `rework_deferred`；
- PORTABLE/F1 仍有跨机恢复和发布债务，不能声称 fresh clone 已正式可交付；这些债务不阻塞当前机器上的 EP1/EP2 开发，到实际部署/发布节点集中处理；
- 目标摄像头授权、episode 真值和独立 sealed camera 评估；
- 目的性行为与真正告警需求的上下文决策层；
- 由授权标注数据支持的概率校准、OOD/uncertain 策略和冻结 episode 合并策略；
- 稳定业务 person identity、日级聚合与心理风险主链接入；
- 真实老人、自然居家或临床有效性。

公开轨迹、合成污染和授权成人 pilot 是不同证据等级。任何报告都必须说明自己证明了哪一层，不能把开发指标写成现实部署效果。

## 7. 常用验证入口

常规开发使用 `eldercare-ai`，并先确认 editable 安装指向当前源码根：

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
conda run -n eldercare-ai python -m pytest tests/test_wandering_*.py -q
conda run -n eldercare-ai python scripts/wandering/preflight_camera_runtime.py --project-root .
```

完整测试只在合并或发布候选时运行；单次性能实验按技术文档2使用窄测、overfit smoke 和固定 validation evaluator。
