# TopoWander-MPT M0-CAM-MVP-1 session evidence prototype

> **当前接受状态：** 后续复审曾以可复现 P1 重开 exact-schema 接受门；MVP-1-F1 已通过 descriptor-aware 红测、最小 consumer 修复和 fresh synthetic E2E 关闭该缺口。现行状态为 `implementation_complete + acceptance=passed`，见 [MVP1_REAUDIT_20260814.md](MVP1_REAUDIT_20260814.md)和任务表。

更新时间：2026-08-14

## 结论

M0-CAM-MVP-1 与 F1 关闭后的现行状态为：

```text
status=wandering_m0cam_mvp1_complete
m0cam_mvp1_complete=true
m0cam_mvp1_acceptance_status=passed
m0cam_mvp1_f1_complete=true
m0cam_mvp1_review_blocker=none
session_evidence_contract_ready=true
product_stage=session_evidence_prototype
mvp1_evidence_scope=synthetic_contract_only
algorithm_event_emitted=false
```

本阶段复用现有 tracking/sidecar → Camera QC → fixed-primary → window prediction → episode candidate 主链，新增薄的 session-level 产品适配层和单命令 synthetic CLI。产品 controller 内部只调用一次既有 `build_primary_camera_inference_bundle()`；没有复制 adapter、QC、preprocessing、fixed-primary forward 或 episode 算法。

本报告只证明 synthetic contract 下的产品软件接线、严格汇总、失败闭锁和制品绑定。它不是目标摄像头、真人、老人域、产品性能或临床有效性证据。

## 实现

- 产品适配层：[camera_product.py](../../../src/elderly_monitoring/modules/mental_health/wandering/camera_product.py)
- 单命令 CLI：[run_wandering_camera_product.py](../../../scripts/wandering/run_wandering_camera_product.py)
- 回归测试：[test_wandering_camera_product.py](../../../tests/test_wandering_camera_product.py)

fresh 输出固定为：

```text
<output>/
  primary/
  wandering_evidence.json
  manifest.json
```

`wandering_evidence.json` 使用 `wandering-session-evidence-v1`，保留 source scope、evidence/validation scope、observation/tracklet/window 三态计数、ready ratio、QC reason/quality flag、episode 四类型计数与 `*_duration_sum_seconds`、固定 candidate/model 身份和 development-unfrozen merge policy。状态规则为：有 ready 即 `ready`；ready 与失败混合时 `degraded=true`；无 ready 且有 inference error 时为 `inference_error`；其余为 `unavailable`。F1 consumer 对 primary 十类 artifact、相关嵌套对象、产品 evidence/manifest 和 final 三项顶层集合执行 exact-key 校验，并复用 episode aggregator 重算 canonical episode bytes。

产品 manifest 使用 `wandering-camera-product-manifest-v1`，以相对路径、byte count 和 SHA-256 绑定 `wandering_evidence.json` 与 `primary/manifest.json`；primary manifest 继续绑定其内部九个 artifact。汇总前会重新核验 primary file set、canonical JSON/JSONL、artifact size/SHA、schema、source/scope、三态/原因/count、fixed model identity、episode contributor 与 policy。

所有产品输出固定：

```text
person_id=null
person_binding_verified=false
probability_calibrated=false
risk_level=null
risk_score=null
recommended_action=null
alert_decision=null
medical_diagnosis=null
algorithm_event=null
algorithm_event_status=not_ready_session_only
```

因此 `track_id` 没有被提升为 person identity，失败窗口没有被计为 direct、ordinary negative 或 wandering，episode duration sum 也没有被解释为 presence time。

## 自动化与 synthetic 证据

- 首次红灯：新测试在 `camera_product` 尚不存在时收集失败；随后才实现生产代码。
- 首次实现产品聚焦：`17 passed`；F1 红测在旧 validator 上为 `42 failed, 17 passed`，最终聚焦为 `59 passed`。
- F1 产品 + primary + episode + QC + inference 相邻回归：`132 passed`。
- 日级聚合 + baseline + mental-health pipeline + CLI 隔离回归：`60 passed, 16 subtests passed`。
- 产品 CLI `--help`：exit 0，只有 tracking JSONL、media sidecar、显式 development merge gap 和 output；不提供 candidate、threshold、calibration、risk、download、video 或 tracker 参数。

首次 Git 外 fresh E2E 使用 `/tmp/wandering_mvp1_e2e_20260814_a1/` 下的 pytest `tmp_path`；F1 另在 fresh `/tmp/m0cam_mvp1_f1_e2e_20260814_01/` 复跑 `1 passed`。输入均为 80 条 synthetic tracking observation，无视频、detector 或 tracker 调用。首次结果为：

- observation `80`；tracklet `1`；window `1`；
- ready `1`、unavailable `0`、inference_error `0`；
- session status `ready`、`degraded=false`；
- episode candidate `1`，本次 fixed candidate 输出为 `random`，duration sum `40.0 s`；
- `wandering_evidence.json` SHA-256：`5d1afe741b5e35fe211a0148a59ccc556584126f8add90a3de2e6594a0edf782`；
- product `manifest.json` SHA-256：`2612cf19480984874e44c63f22ea6afc7aedb2b849df1c328e41ad803ddfc21d`；
- `primary/manifest.json` SHA-256：`53cf3ec7eb4361d3c9808494a85d4a1726e1512b1628626b61bf6196c287f3b7`。

这些 SHA 只标识本次 fresh synthetic run；primary execution 仍含真实工程 latency，不主张不同完整运行逐字节相同。对于同一固定 primary bundle，canonical evidence 重复汇总 bytes 相同。

详细命令和检查见 [VERIFICATION.md](VERIFICATION.md)。

## 未获得的证据与下一阶段

- 没有读取真人媒体、receipt、collection、annotation、WP raw、SmartCare official/raw 或 sealed camera；
- 没有运行 detector、tracker、M0-CAM-D、重训、微调、校准或域适配；
- 没有稳定 person/session/absolute-time/presence binding，不能做日级 evidence；
- 没有冻结 wandering→risk/action policy，不输出风险、告警、医学诊断或 `AlgorithmEvent`；
- PORTABLE F1 继续为 `rework_deferred/rework_required`，formal 仍 blocked；C0/C1 仍为 false。

MVP-2（`WanderingDailyEvidence`）、MVP-3（个人基线）和 MVP-4（风险映射与 `AlgorithmEvent`）均未开始。它们分别等待稳定 person/time/presence binding、合法日级 evidence 和负责人冻结风险/动作策略；不得用本阶段 synthetic 软件证据顶替这些前置。
