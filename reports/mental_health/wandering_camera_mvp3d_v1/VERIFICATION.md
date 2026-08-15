# M0-CAM-MVP-3D 验证记录

日期：2026-08-14

全部完成门已通过，现行 `m0cam_mvp3d_acceptance_status=passed`。

## 红绿测

实现前先新增 MVP-3D 回归并运行，pytest 在收集阶段按预期失败：

```text
ModuleNotFoundError: No module named
'elderly_monitoring.modules.mental_health.wandering.camera_baseline_deviation_preview'
```

首次最小实现后得到 `35 passed, 2 failed`。两个失败都证明实现额外把日级 rate 舍入到 6 位，偏离 MVP-3S 既有公式的原始浮点语义；删除额外舍入并重算 producer prefix SHA 后，MVP-3D 当时的 37 项聚焦测试全部转绿。随后增加 timezone drift 与独立 Git 外 E2E，最终整组计数见下表。

攻击回归覆盖：

- same-day、future reference、空 reference window、reference manifest 复用；
- missing profile、duplicate person/day、timezone 与 candidate/model/seed/epoch/class/threshold/calibration/episode/night/config/duration identity 漂移；
- warming、observation 无 coverage、night current null、reference count 0 与 candidate-duration rate 大于 3600；
- row/manifest/nested metric missing/extra、非空 risk/event、rogue final 顶层文件；
- 重算 canonical JSONL、artifact byte count 与 SHA 后的错误 nested signed delta；
- producer prefix byte 漂移、fresh-only、owned staging cleanup、public readback 和 CLI 参数边界。

## 自动化证据

全部 Python/pytest/CLI 命令均使用 `eldercare-ai`。

| 门 | 结果 |
|---|---|
| MVP-3D + MVP-3S + daily 聚焦 | `155 passed` |
| MVP-1/camera 相邻回归 | `287 passed` |
| generic daily/baseline/pipeline/CLI 隔离回归 | `60 passed, 16 subtests passed` |
| 直接 CLI help | 通过；只暴露 baseline bundle、repeatable observation daily bundle 与 output 参数 |
| Git 外 7 日 reference + 后续 observation E2E/攻击 | `1 passed` |

这些数字只证明 synthetic 软件契约，不是 person binding、真实 baseline、camera 性能或临床证据。

## Git 外 synthetic E2E

固定 Git 工作树外目录 `/tmp/m0cam_mvp3d_synthetic_e2e_20260814` 已运行一个 7 reference-day + 1 strictly-later observation-day 的 CLI E2E，结果 `1 passed`：

- CLI 生成并由 public loader 完整回读只含两个文件的合法 final；
- 测试不调用 production deviation helper，从 observation daily row 独立计算九项 metric，并逐项复算 signed delta-from-median/P90；
- same-day observation 与 baseline input manifest 复用均返回对应结构化错误且无 final；
- 合法 final 的复制件被篡改 nested `delta_from_p90`，同步重算 canonical bytes、byte count 与 SHA 后仍被 loader 拒绝；原合法 final 继续完整回读。

## 未获得的证据

本次未获得真实 person binding、真实 baseline acceptance、真实 baseline deviation、异常/阈值/综合分、概率、风险/动作/诊断/告警、`AlgorithmEvent`、目标摄像头性能、真实老人或临床效果证据。

PORTABLE-F1 仍为 `rework_deferred`；`C0=false`、`C1=false`、`authorized_camera_data_consumed=false`、`m0cam_d_started=false`。MVP-2R、MVP-3R 与 MVP-4 未启动。

```text
status=wandering_m0cam_mvp3d_complete
current_code_task=none
m0cam_mvp3d_status=complete
m0cam_mvp3d_acceptance_status=passed
baseline_deviation_preview_contract_ready=true
mvp3d_evidence_scope=synthetic_contract_only
person_binding_verified=false
real_baseline_ready=false
eligible_for_risk=false
algorithm_event_emitted=false
```
