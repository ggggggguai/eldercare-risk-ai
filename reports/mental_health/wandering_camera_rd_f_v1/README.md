# TopoWander-MPT M0-CAM-RD-F/RD-F2 development entry hardening

## 结论

M0-CAM-RD-F 主体实现与本轮验证已完成，当时记录状态为：

```text
status=wandering_m0cam_development_entry_hardened_waiting_c0_c1
evidence_scope=synthetic_schema_contract_only
readiness_status=not_ready
authorized_camera_data_consumed=false
m0cam_d_started=false
sealed_camera_accessed=false
```

本轮只修复并验证真实 development 入口与 evaluator 的软件契约。没有读取真人或授权 camera 数据，没有启动 M0-CAM-D，没有访问 WP raw、SmartCare official/raw 或 sealed camera，没有重训或改变固定候选、权重、阈值、标签、split 或科学指标。原 M0-CAM-RD 的 240 条 synthetic observation、readiness 和既有制品保持不变。

## 完成后 API 边界审计

本报告中的 synthetic/test-fixture 结果、固定候选身份和测试结果仍有效；但完成后的直接 Python API 审计发现两个进入真实 C1 前必须收口的问题：

1. `run_authorized_camera_development()` 的 `_test_hooks` 必须与 `_candidate_loader` 一样，仅在 `_test_fixture=True` 时可用；否则库调用者可能绕过真实 candidate，却生成 authorized 外观的 execution；
2. exported `predict_primary_camera_window()` 不应接受普通调用者指定 authorized `evidence_scope`；authorized scope 必须只由 receipt-gated development controller 内部产生。

这两项记为 `M0-CAM-RD-F2`，并已在后续小型收口中完成；完成记录见下文。此审计不表示本轮读取过真人数据，也不否定既有 `synthetic_schema_contract_only` 证据。

## RD-F2 API provenance 收口

RD-F2 已于 2026-08-13 完成，当前状态为：

```text
status=wandering_m0cam_development_entry_hardened_waiting_c0_c1
evidence_scope=synthetic_schema_contract_only
readiness_status=not_ready
authorized_camera_data_consumed=false
m0cam_d_started=false
rd_f2_complete=true
```

- `run_authorized_camera_development()` 在非 `_test_fixture=True` 时，只要 `_test_hooks is not None`（包括空 mapping）或传入 candidate loader，就在 output/config/receipt、受保护输入、loader/forward 和 staging/final 之前拒绝；
- 公开 `predict_primary_camera_window()` 只接受 `synthetic_camera_contract / synthetic_contract_only`，ready、unavailable 和 inference-error 的原 synthetic 行为保持兼容；
- 实际 forward 抽到未进入 `__all__` 的私有 helper；只有 receipt-gated development controller 在既有前置通过后由内部逻辑传入 authorized scope，pytest fixture 仍只能产生 `test_fixture_only`；
- 新的核心安全回归使用 `tmp_path`、结构化临时 metadata 和内存 fake runtime，不读取真人或受保护 camera 数据，也不生成新的正式 evidence artifact。
- 最终 primary+dataset+development 聚焦回归共 `74` 项，均包含在最终全部 camera 回归 `139 passed` 中；三个 RD CLI `--help` 均退出 0，`git diff --check` 退出 0（仅既有 autocrlf 提示）。

## 已修复范围

| 修复项 | 实际代码路径 | 验证结果 |
|---|---|---|
| primary loader config 与 evidence envelope | `configs/modules/wandering_camera_development_v1.yaml`、`camera_primary_inference.py`、`camera_episode.py`、`camera_development.py` | development config 显式绑定 binary threshold `0.5`；window、episode、execution 和 final manifest 使用同一 evidence scope；旧 primary 默认仍为 `synthetic_contract_only` |
| 三个 receipt-first CLI | `camera_dataset.py`、`prepare_camera_session.py`、`build_camera_annotations.py`、`run_camera_development.py` | receipt 缺失、失效、operation/scope 不符时在 protected tracking/annotation/cohort、candidate loader、QC/forward 和输出前失败；CLI 无 synthetic/fake/bypass 参数 |
| C3 完整绑定 | `camera_dataset.py`、`camera_development.py` | 实际 `(source_group_id, source_video_id, device_id, setup_id, stream_epoch, track_id)` 必须唯一绑定匿名 participant/session/camera setup/clock domain；未绑定、错绑、重复、stale 或缺 clock 均 fail closed |
| eligible negative person-hours | `camera_development.py` | 按 participant + session + clock domain 分组；同域 union、跨 session 求和；只使用显式 presence + ordinary-negative；positive、truth mask、QC、tracking 和 inference error 分开；prediction uncertain 不缩短分母 |
| annotation/evaluator 语义 | `camera_dataset.py`、`camera_development.py` | accepted truth 必须有 direct/pacing/lapping/random shape；truth-masked、explicit abstention、unavailable、inference error 和 no-prediction miss 分离；purposeful pacing 不改写 shape |
| 最大基数 episode matching | `camera_development.py` | 确定性一对一最小费用最大流：先最大匹配数，再总 IoU、onset+offset 差和稳定 ID；零重叠拒绝；policy ID 非空 |
| production-branch 回归 | `tests/test_wandering_camera_dataset.py`、`tests/test_wandering_camera_development.py` | engineering smoke 与 labeled evaluation fixture 均完整经过 production validators；只在 manifest-bound candidate loader 边界注入 deterministic fake model；输出仅为 `test_fixture_only` 且只写 pytest `tmp_path` |

## 验证摘要

- editable 安装：`elderly-monitoring-algorithms` 指向当前项目根（终端中文路径显示有编码噪声）；
- 原聚焦基线：`22 passed`；
- 新回归红测：新增 API 尚未实现时，2 个 test module 在 collection 阶段按预期失败；
- RD-F 聚焦测试：`35 passed in 6.74s`；
- 全部 camera 回归：`118 passed in 69.62s`；
- 三个 CLI `--help`：全部退出 0，正式参数均要求 receipt/collection，未暴露 fake runtime 或 loader bypass；
- `git diff --check`：退出 0；仅显示工作树原有 autocrlf 提示。

完整命令和零调用证据见 [VERIFICATION.md](VERIFICATION.md)。结构化 `verification.json` 属于本机生成证据，受 `.gitignore` 管理；关键状态与摘要已固化在本报告中。

## 当前门禁与下一步

当前仍是 `readiness_status=not_ready`，RD-F2 软件门已通过。下一步 M0-CAM-D engineering smoke 的剩余启动条件是：

1. 负责人或外部授权流程提供 approved、active、未过期、`camera_development` purpose 的真实 C0 receipt，并覆盖 `run_development`、participant/session/setup/source-group；
2. C1 提供经授权成人、固定 setup 的 tracking JSONL + `wandering-media-v1` sidecar，且 collection、sidecar、实际 tracking scope 和 receipt 全部一致。

有标签 development evaluation 还必须增加：

3. C2 人工 episode annotation，accepted truth 具有可评分 shape，uncertain/excluded 保持真值遮罩；
4. C3 为每个实际 tracklet 提供唯一 participant/session/setup/clock-domain 绑定、participant-present 区间和显式 clock alignment；
5. development-only matching、uncertain 与 episode-merge policy 的非空 policy ID 和明确参数。

在 C0+C1 前不得运行 M0-CAM-D；在 C2+C3 前不得输出 camera accuracy/F1/FAR。synthetic 与 pytest fixture 不是 camera、真人、老人、产品或临床性能证据。

## Git 与制品边界

- 新建本目录；没有覆盖 `wandering_camera_data_readiness_v1/` 或任何旧报告/制品；
- 保留用户原有脏工作树；没有 stash、reset、checkout 丢弃、删除、stage、commit 或 push；
- pytest production-path fixture 只写 WSL `/tmp`，不写入 `reports/`，其 evidence scope 为 `test_fixture_only`；
- 本报告不包含 C0 receipt、真人数据、媒体路径、凭据或 authorized development 指标。
