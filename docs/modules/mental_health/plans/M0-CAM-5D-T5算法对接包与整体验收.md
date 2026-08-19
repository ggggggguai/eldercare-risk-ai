# M0-CAM-5D-T5 算法对接包与整体验收

状态：`completed_algorithm_ready_with_home_smoke_pending`

计划：Day 5

上游：[T3 input-independent context core](M0-CAM-5D-T3居家视频与多模态上下文.md)、[T4 真实日报与个人基线](M0-CAM-5D-T4真实日报与个人基线.md)

## 完成事实（2026-08-18）

- `camera_handoff.py`、`run_camera_handoff.py` 和 manifest/run-summary v2 schema 已落地；producer 使用 fresh staging + atomic commit，public loader 对最终 8 文件独立重读；
- 正式 B01+B02 v2 包严格只有本任务书指定的 8 个文件，原样透传 129 条 episode、82 条 context、2 条 daily、2 条 profile 和 2 条 deviation；
- 组装时验证上游 partial manifest SHA、artifact descriptor、record count、现行 row schema、context→episode identity/snapshot、daily→episode/context source hash、daily/profile/deviation day/profile identity，以及 count/duration 守恒；
- 守恒结果为 129 条 episode、95 条可用 shape、34 条 unavailable、0 条 error、19 条 wandering-like 和 617.8 秒 wandering-like duration；daily context bucket 为 129；
- 单命令 cached-tracking E2E 已实际重跑 48 条 B01+B02 视频，重新得到 129/82/2/2/2，并由最终 public loader 通过；E2E 目录为 `tmp/wandering_camera_5d_runs/w5d05-e2e-b01-b02-v1/`；
- provider-disabled 路径以 3 条 `unknown/unavailable` context 继续生成 129 条 episode 和完整 2/2/2 daily/profile/deviation，最终状态仍为 `algorithm_ready_with_home_smoke_pending`，不是 run error；
- 同一输入的业务 record ID 和显式 run ID 稳定；已有输出目录、descriptor 篡改、stage manifest 篡改和跨阶段身份篡改均失败关闭；
- manifest 与 run summary 均固定 `module=mental_health`、`algorithm_event_emitted=false`、`medical_diagnosis_emitted=false`、`backend_or_frontend_implemented=false`；居家状态保持 `awaiting_input/not_run_input_unavailable`，未声明 `home_validated`。
- 最终测试为 W5D-05 聚焦 `10 passed`、全 camera `660 passed`、wandering + mental_health 联合 `929 passed, 205 subtests passed`；两条 warning 为既有重复 ZIP 防御用例和 PyTorch CUDA 驱动探测，CPU 验收无失败。

正式证据见 [W5D-05 B01+B02 handoff v2](../../../../reports/mental_health/wandering_camera_handoff_w5d05_v2/README.md) 和 [W5D-05 disabled degradation v2](../../../../reports/mental_health/wandering_camera_handoff_w5d05_disabled_v2/README.md)。

## 目标

交付一个后端团队可以直接消费、开发人员可以重复运行、失败可以定位的算法包。只实现算法 runner 和文件/object contract，不实现后端或前端。

## 最终运行入口

应提供一条命令，或在 README 中提供不超过三个连续命令，覆盖：

```text
MP4 or tracking/sidecar
  -> automatic episode
  -> trajectory shape
  -> optional context
  -> daily report
  -> baseline profile/deviation
  -> handoff bundle
```

若 MP4 tracking 的运行成本较高，允许入口从已经生成且身份匹配的 tracking/sidecar 恢复。五日核心验收至少使用一条已登记的 B01/B02 真实 MP4 证明 MP4 -> tracking 入口，完整回归可从缓存 tracking/sidecar 开始。居家 MP4 到位后再以同一入口补做 truth-free smoke，不把素材缺席视为交付失败。

## 验收范围与状态

| 范围 | 当前是否必需 | 结果语义 |
|---|---|---|
| B01+B02 episode/context/daily/baseline E2E | 是 | 五日算法闭环工程验收 |
| provider-disabled/fake/error fallback | 是 | 多模态非阻塞验收 |
| 多日 fixture/replay | 是 | 3/7/14 日机制验收，不冒充真实长期观察 |
| 居家 MP4 truth-free smoke | 素材到位后补充 | 真实场景工程 smoke，不等于性能评价 |
| 居家人工标签定量评价 | 否 | 有独立标签后才形成 development 指标 |

无居家输入的 bundle 必须携带 `home_input_status=awaiting_input`、`home_smoke_status=not_run_input_unavailable` 和实际 `validation_scope`。该状态允许算法包标记为 `algorithm_ready_with_home_smoke_pending`；不得标记为 `home_validated`。

## 交付目录

```text
handoff/
  handoff_manifest.json
  episode_results.jsonl
  context_reviews.jsonl
  daily_reports.jsonl
  baseline_profiles.jsonl
  baseline_deviations.jsonl
  run_summary.json
  README.md
```

## 后端对接最小语义

- `module` 固定为 `mental_health`；
- `person_id/session_id/source_video_id` 显式给出；
- 时间使用带时区 ISO 8601 或明确 local_date/timezone；
- 每条记录有 schema version、model/config/policy identity；
- status/quality/error 不可丢失；
- context 结果与 shape 结果分字段；
- baseline readiness 明确；
- 同一输入重跑具有稳定 ID，fresh output 不覆盖已有目录。

本轮可以预留 `AlgorithmEvent` 映射说明，但不得把单个 wandering-like shape 或多模态 label 直接写成医学诊断。后端何时产生业务事件由后续接口确认。

## 完成门

1. B01+B02 回归能生成完整 handoff bundle；居家输入缺席时生成明确 pending 状态，到位后可原样复放。
2. 无人工修改中间 JSON。
3. 多模态 provider 不可用时仍生成 episode/daily/baseline 输出。
4. 关键计数、duration、person/day binding 经人工抽查一致。
5. 输出 descriptor、record count 和 source references 自洽。
6. fresh/non-overwrite、错误降级和恢复路径有测试。
7. 聚焦测试通过后，完整 wandering 测试通过。
8. README 给出当前机器可运行命令、输入要求、输出字段和已知限制。
9. 核心交付结论与证据范围一致：允许 `algorithm_ready_with_home_smoke_pending`，禁止在未运行时声称 `home_validated`。

## 最终验证命令

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
conda run -n eldercare-ai python -m pytest tests/test_wandering_camera_*.py -q
conda run -n eldercare-ai python -m pytest tests/test_wandering_*.py tests/test_mental_health_*.py -q
```

涉及 tracking 新改动时，先用一条已登记的 B01/B02 授权真实视频跑小帧数或短片 smoke，并记录实际命令和输出目录；居家视频到位后补跑，不等待人工标签。

## 明确不交付

- 后端 HTTP 服务；
- 前端页面；
- 账号、权限、设备管理；
- 消息推送、电话、工单；
- 临床诊断或跨人泛化声明；
- C4 sealed、PORTABLE 或正式跨机发布。
