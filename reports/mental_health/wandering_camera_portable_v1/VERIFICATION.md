# M0-CAM-PORTABLE verification

> 本文件是 F1 执行时验证快照。测试计数仍是真实结果，但后续独立复审发现其未覆盖的 P0，因此当前接受状态见 [F1_REAUDIT_20260814.md](F1_REAUDIT_20260814.md)和任务表；不要单凭本快照写成软件审计已通过。

## 1. 现场与资产

验证环境：WSL `Ubuntu-22.04`，conda 环境 `eldercare-ai`，base HEAD：

```text
bfc3329ad14d145e6b4c3fb836a0a525f30e970b
```

只读核验使用了任务书列出的 candidate 五件套、preprocessing config/manifest/feature_stats 和 Ultralytics 包内 ByteTrack resource。没有搜索媒体目录、未知权重缓存或用户目录，没有读取 camera receipt/collection/tracking/sidecar/annotation/video。

## 2. 聚焦与完整 camera 回归

```bash
conda run --no-capture-output -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_portability.py \
  tests/test_wandering_camera_collection.py -q
```

结果：`100 passed`。

修复前新增回归直接复现两条 P0：同一 HEAD 下同时修改 runtime source 与 contract descriptor 时，旧实现错误返回 `ready`；loader/tracker 捕获网络拒绝异常并返回时，旧 guard 也能正常退出。F1 修复后，这两类回归均稳定 blocked 且无 controller final。

```bash
conda run --no-capture-output -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_portability.py \
  tests/test_wandering_camera_adapter.py \
  tests/test_wandering_camera_collection.py \
  tests/test_wandering_camera_corruption.py \
  tests/test_wandering_camera_dataset.py \
  tests/test_wandering_camera_development.py \
  tests/test_wandering_camera_episode.py \
  tests/test_wandering_camera_inference.py \
  tests/test_wandering_camera_primary_inference.py \
  tests/test_wandering_camera_qc.py -q
```

结果：`297 passed`。唯一 warning 来自测试故意构造 duplicate ZIP member，正是被 restore 拒绝的攻击 fixture，不是 production warning。

受影响的 release/candidate/primary/development 窄测另有 `79 passed, 4 subtests passed`。297 项完整 camera 回归属于自动化/synthetic camera 兼容性证据，并非全部 fake/in-memory forward；既有 primary 测试包含在 synthetic tracking 上使用真实固定 candidate 执行 forward 和 QC 的路径。只有 production preflight 与第 5 节独立 candidate safe-load可以表述为 `forward=0`；它们也不产生 camera 性能证据。

## 3. 攻击与失败原子覆盖

自动化回归覆盖：

- absolute、traversal、drive-relative、Windows absolute、UNC、反斜杠、URL、credential marker、控制符、超长 member/destination；
- duplicate YAML key、duplicate/extra/missing/case-colliding member/destination、ZIP symlink、metadata drift；
- archive/member size 或 SHA 漂移、同长度篡改、解压尺寸上限；
- fake/editable checkout、Git commit blob 与 contract/descriptor/active source 三方漂移、candidate model/release source cross-binding、PEP 440 dependency drift；
- adapter/QC/inference/episode/preprocessing 与正式 CLI 的 runtime trust closure；
- builder 两项输出、三个 restore group、ready marker、preflight report 的 final 竞争目标 no-replace；提交中途失败、KeyboardInterrupt、best-effort rollback 与逻辑 residue 报告；
- canonical 父目录 link/reparse、case alias、外部 checkout identity、UNC/device/URL/credential source；
- candidate safe-load、preprocessing cross-binding、detector missing、ByteTrack identity 任一失败时 fail closed；
- 在模块导入前安装 socket/connect/create_connection、HTTP 和 subprocess bomb；candidate loader、detector loader 与 `runtime.run()` 分别吞掉拒绝异常并返回时，作用域退出仍按 attempt delta fail closed，socket 函数恢复，controller 不提交 final；
- detector 首次解析产生冻结 token，第二次校验分别拒绝 contract、ready marker、weight/path/inode 漂移；ready marker 拒绝重复键、非 canonical bytes、额外或漂移字段；
- CLI `ready=0`、预期 deployability `blocked=2`、参数/contract/internal error `=1`；三条新增 PORTABLE CLI 不提供 download、URL、video、tracking 或 forward bypass。既有 camera CLI 的正常 video/tracking 输入参数不属于 bypass，不能被这句话否定。

本次独立复验的七个 CLI `--help` 均为 exit 0：

```text
scripts/wandering/build_camera_runtime_asset_bundle.py
scripts/wandering/restore_camera_runtime_assets.py
scripts/wandering/preflight_camera_runtime.py
scripts/wandering/build_camera_tracking_pair.py
scripts/wandering/prepare_camera_session.py
scripts/wandering/run_camera_development.py
scripts/wandering/run_topowander_camera_inference.py
```

## 4. 隔离 overlay

隔离源码由 `git archive` 的 base HEAD 加精确七文件 allowlist 构成：

```text
modified:
  src/elderly_monitoring/modules/mental_health/wandering/camera_collection.py
  tests/test_wandering_camera_collection.py
new:
  src/elderly_monitoring/modules/mental_health/wandering/camera_portability.py
  scripts/wandering/build_camera_runtime_asset_bundle.py
  scripts/wandering/restore_camera_runtime_assets.py
  scripts/wandering/preflight_camera_runtime.py
  tests/test_wandering_camera_portability.py
```

两份 tracked 修改的 binary diff 综合 SHA-256（F1 完成时现场值）：

```text
bcb2aa1af308f8a8d1a0d64fcd974c0444b2bdeb2284a4d86dd840c6580fe9df
```

七文件当前 identity：

| 文件 | size | SHA-256 |
|---|---:|---|
| `camera_portability.py` | 86,861 | `4db838b85e7e142cb98af473857bfe50ae29bb07f7656a4352b02aa228c0eb67` |
| `camera_collection.py` | 20,748 | `1b4c0f070a8de06b2f62e19d976a291e99d3f4a6a7176f230b35e7455cb0118b` |
| `build_camera_runtime_asset_bundle.py` | 1,889 | `638a5a30a752f4cfc696aa7b4fd2285f32ec4d885797176b2b2c237c5563cd16` |
| `restore_camera_runtime_assets.py` | 1,594 | `d7643be0e3a6f9c7af8e7b15c5ac3f4f6036d708873aba3fda013f8d2e3a6207` |
| `preflight_camera_runtime.py` | 1,291 | `eb9605b4f9789843f7ee15c690bda3a51a40dd9c778e79dad2c78a3ad4d98348` |
| `test_wandering_camera_portability.py` | 52,739 | `259a3ee158c656f03cec9f400d061e9f88690cbcd4f771fc010bd1fbc89d4520` |
| `test_wandering_camera_collection.py` | 36,553 | `0dd7332b4dd36fc5d40bdeddebc0af43516573b924833f1efd06cd6453411f9d` |

在 `unshare -Urn` 系统级断网 namespace 内，从该 overlay 创建独立 `--system-site-packages` venv，再执行本地 `pip install --no-build-isolation --no-deps -e` 和两份聚焦测试。结果为 `100 passed`、无 skip；唯一 warning 是 duplicate ZIP 攻击 fixture。本次 overlay 证明 F1 软件回归可在系统级断网边界复现，但不是不可变/负责人批准的持久 source trust root，因此 formal 状态仍有 `source_integration_pending`。

## 5. 真实 safe-load 与生产 blocked preflight

真实 candidate safe-load 在系统级 `unshare -Urn` 和应用 `deny_network_access()` 双重保护下执行，调用既有：

```python
load_candidate_from_manifest(
    manifest_path,
    expected_manifest_sha256="3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7",
)
```

结果：

```json
{"candidate_id":"topowander-m0s-seed20260731-epoch0005","device_types":["cpu"],"forward_calls":0,"manifest_sha256":"3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7","model_training":false,"network_attempt_count":0,"network_denial_guard_enforced":true}
```

production preflight 的当前结构化结果：

```json
{"blocker_code":"asset_source_unavailable","checks":[{"check_id":"active_editable_checkout","status":"ready"},{"blocker_code":"asset_source_unavailable","check_id":"canonical_contract_and_source_trust","status":"blocked"}],"network_attempt_count":0,"network_denial_guard_enforced":true,"purpose":"deployability_only","safeguards":{"changes_c0_c3":false,"reads_video":false,"runs_forward":false,"runs_qc":false,"runs_tracker":false},"schema_version":"wandering-camera-runtime-preflight-v1","status":"blocked"}
```

这是预期 deployability blocker，不是参数或内部错误。直接 Python CLI 契约测试确认其应用退出码为 2；conda wrapper 只将非零子进程报告为失败，不改变结构化 blocker。

## 6. F1 入口审计发现（历史）

首次验证之后的独立源码审计和最小运行复现确认：

1. `source_identity.descriptors` 来自 canonical contract，自身又被用来验证同一工作树；`git_tree` 只比较 HEAD 字符串，未把当前 contract/runtime files 与该 commit 的 blobs 独立比较；`approved_patch_archive.base_git_commit` 没有参与实际验证。这意味着同一 dirty overlay 可修改源码与 descriptors 后自签；
2. 应用 network guard 只在连接入口记录并抛出异常，context 退出时不检查本次 attempt delta。最小复现由下游吞掉拒绝异常后得到：

   ```text
   guard_exited_successfully=true
   attempt_delta=1
   ```

   因此 `network_attempt_count>0` 仍可能伴随成功退出，存在 preflight/controller 假 `ready` 通路；
3. fresh-only 最终提交使用 `Path.replace()`，目标若在前置检查后出现可能被覆盖；runtime trust-root closure、canonical 父目录身份、可捕获失败 rollback、detector 不可变二次身份和部分 blocker 映射也未完整关闭。

这些发现是 F1 的红测入口，不再代表修复后的软件状态。

## 7. F1 执行时结论（后续复审未接受）

F1 当时以最小文件面实现了以下修复意图；后续复审证明其测试没有覆盖全部可构造路径，因此本节不能再作为当前通过结论：

- `git_tree` 独立读取指定 commit blobs，逐字节绑定 canonical contract 与全部 runtime trust roots；无外部 approval anchor 的 patch 模式 fail closed；
- network guard 按每次作用域 attempt delta 退出失败，preflight 只在最外 guard 成功退出后设为 ready；
- fresh final 使用 atomic no-replace，进程内可捕获失败执行全项 best-effort rollback；不声称断电/进程被杀后的 crash atomicity；
- detector 使用冻结 token 二次核验，marker 使用无重复键的 exact canonical JSON bytes，预期 deployability 缺失返回结构化 blocker/exit 2。

执行时曾记录的现场结论：

```text
m0cam_portable_f1_complete=true
portable_software_audit_status=passed
portable_review_blocker=none
m0cam_portable_status=blocked
portable_runtime_blocker_code=asset_source_unavailable
source_integration_pending=true
C0=false
C1=false
m0cam_d_started=false
```

## 8. 未执行项

- 未创建 production canonical contract；
- 未创建或声称 owner-approved detector/archive；
- 未运行 production restore 成功路径；
- 未读取 camera 数据；
- production restore/preflight 和独立 safe-load 未运行 tracker、QC、candidate forward、evaluator 或 M0-CAM-D；完整 297 项 synthetic 兼容回归包含真实 fixed-candidate forward/QC，不能无条件写成全局“未运行”；
- 未修改 `model.py`、`release.py`、共享 tracker、candidate bytes、threshold、labels、splits、environment 或共享依赖；
- 未 stage、commit 或 push。
