# M0-CAM EP1B / EP2A 历史证据交接

更新时间：2026-08-17

状态：`superseded_by_m0cam_5d_fast_delivery`

> 本文不再是当前执行入口。负责人已将路线调整为 B01+B02 可反复用于 development，优先完成分段、轨迹识别、居家视频、日报、个人基线和算法对接闭环。当前任务见 [M0-CAM-5D 快速交付总任务书](M0-CAM-5D快速交付总任务书.md)和 `docs/tasks/README.md`。

## 1. 保留的历史事实

- B01/B02 是同一 participant、同一办公室 setup 的两个 session；
- B01 旧 development 有 36 条可用视频；B02 旧视频级 holdout 有 11 条视频；
- EP1B oracle-boundary binary all-eligible macro-F1：B01 `0.817375`、B02 `1.0`；
- EP2A boundary F1：B01 `0.598131`、B02 `0.971429`；
- candidate-inclusive v2 在 B02 上 ready=`17/18`、all-eligible binary macro-F1=`0.970588`；
- v2 保持 uncertain endpoint/status/reason，rejected_by_qc 继续跳过；
- 旧结果和历史输出不覆盖、不删除。

精确 cohort、指标和 hash 继续以以下报告为准：

- [B01/B02 intake](../../../../reports/mental_health/wandering_camera_b01_b02_intake_v1/README.md)
- [B01 development / B02 历史结果](../../../../reports/mental_health/wandering_camera_b01_development_v1/README.md)
- [B02 candidate-inclusive v2](../../../../reports/mental_health/wandering_camera_b02_candidate_inclusive_development_v1/README.md)

## 2. 已被取代的门禁

以下旧要求已经失效，不得再据此暂停当前开发：

```text
current_code_task=none
waiting_for_independent_camera_generalization_evidence
B01/B02 不得继续调参
新 participant/setup 前不得启动下游
C4 sealed 必须先于日报和个人基线
```

新规则是：

- B01+B02 合并为可持续使用的 camera development 数据；
- 可调分段器、边界 refinement、送模策略、camera preprocessing、binary threshold 和必要的轻量 camera 模型；
- 居家视频作为下一真实场景 smoke/验收，后续若参与调参则标为 development；
- 先形成 episode -> shape -> context -> daily -> baseline -> handoff；
- 独立泛化、C4 sealed 和 PORTABLE 为后续增强，不是五日门禁。

## 3. 仍有效的技术约束

- shape 与 purpose/context 分离；
- 多模态 context 不改轨迹类别或人工 truth；
- technical hard break 不跨越；
- uncertain 不伪装成人工 accepted boundary；
- 原视频、截图和模型响应不进 Git；
- 输出 fresh/non-overwrite；
- development 指标不宣称临床效果或跨人泛化。
