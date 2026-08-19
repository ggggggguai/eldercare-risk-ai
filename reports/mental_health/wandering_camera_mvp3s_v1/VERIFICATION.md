# M0-CAM-MVP-3S 验证记录

日期：2026-08-14

初版执行状态曾为 `passed`，后续独立复审重开两个接受门缺口；M0-CAM-MVP-3S-F1 已按 [MVP3S_REAUDIT_20260814.md](MVP3S_REAUDIT_20260814.md) 的复现关闭缺口，现行 `m0cam_mvp3s_acceptance_status=passed`。

## 旧缺口红测

F1 首先将独立复审攻击固化为回归。修复前精确运行两项攻击得到 `2 failed`，均为 `DID NOT RAISE`：marker 前同长度 producer mutation 仍返回冻结 SHA；rehashed negative reference stats 在同步更新 canonical bytes、size 与 SHA 后仍被 public loader 接受。

随后扩展的攻击测试在篡改后重算 canonical bytes、artifact size 和 SHA，覆盖：

- MVP-2S daily row/manifest missing、extra、descriptor/count 与 rogue top-level；
- profile/manifest missing、extra、metric order、nested stats 与 rogue top-level；
- 非空 deviation/risk/`AlgorithmEvent`；
- duplicate manifest/person-day 与 timezone 漂移；
- candidate/model/seed/epoch/class/threshold/calibration/episode/night/config/duration identity 漂移；
- fresh-only、owned staging cleanup、validated readback、CLI 参数边界与 15 日滚动 E2E。
- producer marker 缺失/重复、边界多/少 byte、同长度前缀漂移和 active source readback；
- exact reference-day fields、日期排序/重复、数值范围、window/readiness/count 漂移、负数与范围内错误 stats、active builder source 和 v1 downstream 拒绝。

## 自动化证据

全部 Python/pytest/CLI 命令均在 `eldercare-ai` 中运行。

| 门 | 结果 |
|---|---|
| MVP-3S-F1 + daily 聚焦 | `115 passed` |
| MVP-2S/MVP-1/camera 相邻回归 | `247 passed` |
| generic daily/baseline/pipeline/CLI 隔离回归 | `60 passed, 16 subtests passed` |
| 直接 CLI help | 通过；只暴露 `--daily-bundle` 与 `--output` 任务参数 |
| Git 外 15 日 v2 synthetic E2E + 两类 descriptor-aware 攻击 | `1 passed` |

UTF-8 round-trip 与本次相关 Markdown 链接测试通过；`git diff --check` 通过（只有既有工作树的 LF/CRLF 提示）；editable 安装指向当前仓库；最终 branch 为 `feat/wandering-data-pipeline`、HEAD 为 `bfc3329ad14d145e6b4c3fb836a0a525f30e970b`、stage 为空。表中软件门不能解释为真人 baseline 或 camera 性能证据。

## Synthetic E2E

Git 工作树外 pytest 临时目录构建 15 个 synthetic 自然日，full validated loader 回读 v2 final：只含 `baseline_profiles.jsonl + manifest.json`，profile 使用最近 14 个可用日，首日 `2026-07-02`、末日 `2026-07-15`、readiness 为 stable preview。测试不调用 production stats helper，从 `reference_days` 独立线性插值复算全部九项 `count/median/p90/min/max` 并逐项相等；随后复制同一合法 final，分别注入负数和 `[0,1]` 内错误 trajectory stats，重算 JSONL descriptor 后均得到 `reference_stats_mismatch`，原合法 final 字节与回读不受影响。风险/事件边界保持空值，未读取任何真人媒体。

## 未获得的证据

本次未获得真实 person binding、真实 baseline acceptance、baseline deviation、风险/动作/诊断/告警、`AlgorithmEvent`、目标摄像头性能、真实老人或临床效果证据。

PORTABLE-F1 仍为 `rework_deferred`；`C0=false`、`C1=false`、`authorized_camera_data_consumed=false`、`m0cam_d_started=false`。MVP-2R、MVP-3R 与 MVP-4 未启动。

```text
status=wandering_m0cam_mvp3s_f1_complete
current_code_task=none
m0cam_mvp3s_status=complete
m0cam_mvp3s_implementation_complete=true
m0cam_mvp3s_acceptance_status=passed
m0cam_mvp3s_review_blocker=none
synthetic_baseline_profile_contract_ready=true
mvp3s_evidence_scope=synthetic_contract_only
m0cam_mvp3s_f1_status=complete
m0cam_mvp3s_f1_acceptance_status=passed
person_binding_verified=false
real_baseline_ready=false
eligible_for_risk=false
algorithm_event_emitted=false
```
