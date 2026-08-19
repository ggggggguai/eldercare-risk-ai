# M0-CAM-PORTABLE 执行任务书

> 历史发布债务：PORTABLE 仅在跨机/正式发布前处理，不阻塞当前机器的五日算法闭环。现行任务见 [M0-CAM-5D 总任务书](M0-CAM-5D快速交付总任务书.md)。

版本：1.1

更新时间：2026-08-14

状态：首次实现已结束；F1 后续复审为 `rework_deferred`，formal PORTABLE 继续 blocked

> **2026-08-14 后续复审注记：** F1 的执行时测试证据保留，但“软件审计通过”已被后续独立复审撤销。当前产品代码转入 `M0-CAM-MVP-1`；PORTABLE rework 延后到产品实际依赖 restore/preflight、fresh-clone 部署、正式交付或 M0-CAM-D 前。实时结论见[任务表](../../../tasks/README.md)和 [F1 复审记录](../../../../reports/mental_health/wandering_camera_portable_v1/F1_REAUDIT_20260814.md)。

## 1. 文档角色与事实源

本文是 [徘徊识别技术文档2](徘徊识别技术文档2.md) 下的有界执行任务书，只展开 `M0-CAM-PORTABLE`，不是第二条模型路线，也不改变固定候选或摄像头数据门禁。

- 实时任务状态只写 [当前任务表](../../../tasks/README.md)；
- 稳定路线和门禁只写技术文档2；
- 已实现能力在实现和验证完成后才写 [心理健康模块 README](../README.md)；
- 实际命令、测试、隔离验收和结果已写入 `reports/mental_health/wandering_camera_portable_v1/`；
- 如果本文与任务表、技术文档2或当前代码冲突，以任务表、技术文档2和当前代码为准，并先报告冲突。

本文中的路径、大小和 SHA 是 2026-08-14 制定任务时的只读基线，执行 AI 必须现场复核，不能只相信本文。

后续 F1 对本文首次实现做过 source trust、network、no-replace、运行闭包、路径/回滚、detector token 和 blocker/exactness 修复并形成自动化证据；再复审仍发现 4 个 P0。当前为 `m0cam_portable_f1_acceptance_status=rework_deferred`、`portable_software_audit_status=rework_required`、`portable_review_blocker=post_f1_findings_open`。formal 仍 `blocked/asset_source_unavailable + source_integration_pending`。

## 2. 当前上下文

制定本文时，已确认：

```text
branch=feat/wandering-data-pipeline
HEAD=bfc3329ad14d145e6b4c3fb836a0a525f30e970b
stage=empty
camera_handoff_status=wandering_m0cam_c0_c1_handoff_hardened_waiting_owner_inputs
m0cam_portable_phase_status=wandering_m0cam_portable_blocked
m0cam_portable_status=blocked
portable_software_audit_status=rework_required
portable_review_blocker=post_f1_findings_open
portable_runtime_blocker_code=asset_source_unavailable
source_integration_pending=true
c01_f_complete=true
c01_f2_complete=true
evidence_scope=synthetic_schema_contract_only
C0=false
C1=false
authorized_camera_data_consumed=false
m0cam_d_started=false
```

`M0-CAM-C01-PREP/F/F2` 已完成并完成本地 Git 集成。`M0-CAM-PORTABLE` 首次实现、F1 修复尝试、隔离 overlay 和当前 candidate safe-load 已形成；但 [M0-CAM-PORTABLE-F1](M0-CAM-PORTABLE-F1执行任务书.md) 后续复审未接受。当前无 owner-approved 资产时停止 PORTABLE 代码追加，先推进不依赖它的 MVP-1；拍摄线可并行准备 C0+C1，但各线不能互相冒充完成。

当前软件审计待返工，而且 fresh clone formal 仍不能确定性运行：

1. 固定 candidate v3 五文件存在于本机，但属于 ignored/local-only 资产；
2. preprocessing manifest 和 feature stats 存在于本机，但同样被 ignore；
3. portable contract 要求的固定仓库目标路径没有 `yolov8n.pt`，也没有负责人批准的 detector size、SHA 或持久恢复来源；不得为找一个可用文件而搜索未知缓存或用户目录；
4. video-direct controller 只向共享 tracker 传入不可变 token 二次核验后的本地绝对 detector 路径；应用层 guard 对被下游吞掉的网络拒绝异常也按 attempt delta fail closed；
5. portable contract、显式 restore/preflight 和隔离验收机制已经存在；F1 关闭过部分已知缺口，但后续复审仍发现 source identity、实际 runtime closure、restore commit-time exactness 和 rollback ownership 四个 P0；
6. production canonical contract、approved detector/persistent archive 和持久 source identity 仍缺失，formal closure 不能启动。

因此，F1 的 `297 passed` camera 回归、`100 passed` 隔离回归和 candidate safe-load 只证明当时被覆盖路径的自动化结果及当前候选字节 loadability；它们不足以证明软件审计通过，也不能证明 PORTABLE formal pass、fresh-clone deployability 或 camera 性能。完整状态以任务表、F1 后续复审和当前代码为准。

## 3. 本任务目标

本任务只回答：

> 在完全不读取 camera 数据、不运行 tracker/QC/model forward 的前提下，一个不含 ignored 运行资产的 fresh clone 或等价隔离源码，能否通过明确、离线、SHA 绑定、拒绝覆盖的方式恢复固定运行资产，并完成真实 candidate safe-load 和 runtime deployability preflight？

必须形成：

1. exact-schema 的外层 portable asset contract；
2. 明确离线来源的 deterministic bundle/restore 能力；
3. `ready|blocked` 的零视频 runtime preflight；
4. video-direct 路径在共享 tracker 调用前对显式本地 detector 权重做 size/SHA 校验，消除正常运行路径中的 basename 自动下载依赖；同进程恶意替换和复杂 TOCTOU 仍作为明确的外部信任边界；
5. clean checkout、`git archive` 加精确补丁或等价隔离源码中的恢复与 safe-load 证据；
6. 缺失、漂移、篡改、路径攻击、依赖不符和网络尝试均 fail closed 的回归证据。

## 4. 明确非目标

本任务不得：

- 在 production restore/preflight 或隔离验收中读取、搜索、哈希或运行视频、receipt、collection、tracking、sidecar、annotation 或 sealed 数据；
- 在 production restore/preflight 或隔离验收中运行 tracker、Camera QC、window preparation、模型 `forward`、evaluator 或 M0-CAM-D；
- 重训、执行 M1、换 candidate，或修改模型、权重、阈值、标签、split、类别顺序、matching、uncertain、episode merge policy；
- 修改冻结 candidate v3 manifest 或五个 bundle 文件的字节；
- 重新生成 preprocessing manifest/stats 代替冻结副本；
- 修改共享 `fall_risk/tracking.py`、`environment.yml` 或共享依赖；
- 依赖 Ultralytics 静默下载、未知缓存、搜索任意用户目录或未声明 fallback；
- 生成、签署或代填 C0 receipt，也不得改变 C0-C3、`evidence_scope` 或 owner acceptance；
- 把 preflight、safe-load、fixture 或隔离恢复描述为 camera、真实老人、产品或临床证据；
- 未经用户明确授权进行 stage、commit、push、stash、reset、发布、Git LFS 或远端制品上传。

自动化回归的边界单独计算：既有和新增测试可以在 `tmp_path`、in-memory 中使用立即清理的 synthetic receipt/collection/tracking/sidecar、fake tracker/QC 和 fake/in-memory forward，以证明兼容性和零调用顺序；这些只属于自动化软件证据，不得保存为 authorized input、PORTABLE 资产或 camera 性能。PORTABLE 专用 preflight 测试仍必须证明其自身的 video/tracker/QC/forward 调用为零。

## 5. 冻结身份与最小资产闭包

### 5.1 固定 candidate

固定候选继续是：

```text
candidate_id=topowander-m0s-seed20260731-epoch0005
seed=20260731
best_epoch=5
candidate_manifest_schema=wandering-m0rh-scoring-candidate-manifest-v3
```

制定任务时的 candidate manifest：

```text
path=reports/mental_health/wandering_performance/m0r_score_entry_hardening_v1/artifacts/topowander_m0r_candidate_v3/candidate_manifest.json
size=12642
sha256=3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7
```

candidate 五文件闭包为：

| 成员 | size | SHA-256 |
|---|---:|---|
| `candidate_manifest.json` | 12,642 | `3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7` |
| `model_state.npz` | 838,934 | `94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031` |
| `forward_config.yaml` | 4,469 | `debd3adcf0c84d266ca594fd9ba2716a5a854e074e6e2092570b1ec656ed80f7` |
| `performance_config.yaml` | 1,840 | `ecc4c5a00dc9c30b1a4a16b943d39d9de85cce27077c8ccba1ba223c529de5f2` |
| `frozen_wp_rf_config.yaml` | 3,398 | `d29db531f42ae5d7cf4ed540a85681ddab78e483695633ccc8b382d98e745e35` |

必须复用 `release.py` 中现有 `load_candidate_from_manifest()` 做 manifest-bound safe-load；不得复制一套宽松 loader，也不得修改 `model.py`、`release.py` 或 candidate bytes 来绕过校验。

### 5.2 preprocessing

最小 preprocessing 闭包只有：

| 成员 | 状态 | SHA-256 |
|---|---|---|
| `configs/data/wandering_preprocessing_v1.yaml` | tracked | `5b69243337cddeeec8beed4f081330b06c47eadeaee06dc3b3e4ba026f36ff45` |
| `data/processed/wandering/preprocessing/v1/manifest.json` | ignored/local-only | `242072bdfe4b969d320a091ecc499aff445c30a937dfb1e6739ad869aa938593` |
| `data/processed/wandering/preprocessing/v1/feature_stats.json` | ignored/local-only | `249ad8383043dba1d95452533ddc6004dfecc2c9e99be43f788dcfcbfb434c8d` |

79 MB `samples.jsonl`、preprocessing report、WP upstream、训练 checkpoint、source archive、RF/TCN development manifests 不属于 camera runtime 最小闭包，不得为了 PORTABLE 读取或打包。

### 5.3 detector、tracker 和依赖

- `yolov8n.pt` 是必需运行资产；目前没有获批字节、size、SHA 或恢复来源，不能猜测，未经负责人明确授权也不得下载；未来即使批准官方不可变来源，也只能显式取得、审定 SHA 后转为离线资产，不能恢复 runtime download fallback；
- Ultralytics 8.4.78 内置 `bytetrack.yaml` 制定任务时为 856 bytes、SHA-256 `395701d947a749179dee3e327b1181b730c4ca98e7cac5a2ab05280aae573b8b`；执行时必须从已绑定包中定位并验证完整 SHA，而不是只检查文件名；
- Python、项目 distribution、NumPy、Torch、OpenCV、PyYAML、Ultralytics 等版本必须与项目锁定环境相符；重建标准以 `environment.yml` 为准，项目版本以 `pyproject.toml` 和 `importlib.metadata` 的 distribution metadata 为准，editable 路径用 samefile 核验。`environment-reference.txt` 只是本机说明，其中陈旧的项目版本不得作为 gate；依赖按 PEP 440/锁定 distribution 版本比较，合法 runtime local suffix（例如 Torch 的 `+cu130`）应另行记录而不能机械判为漂移；
- tracked camera、primary、development、collection 配置和 active `model.py/release.py` 作为 trust roots 参与 preflight，但不重复打包为 ignored runtime 资产。

### 5.4 源码 trust root

candidate safe loader 只验证 manifest 与四个 bundle artifact，不能替代运行源码身份。PORTABLE 的完整 source trust root 还必须绑定：

- 不可变 Git commit/tree，或负责人认可、SHA 绑定、可持久恢复的 patch archive；
- `camera_collection.py`、`camera_component.py`、`camera_dataset.py`、`camera_development.py`、`camera_primary_inference.py`；
- candidate 自身已冻结的 `model.py`、`release.py`；
- 只读复用的共享 `fall_risk/tracking.py`；
- 对应 scripts/configs 的 exact descriptors。

仅用未提交工作树或临时 overlay 验证实现机制，不足以证明一个新的 fresh clone 能恢复同一源码。

## 6. 外层 portable asset contract

不能修改冻结 v3 manifest 中的 `current_machine_local_only`；新增外层 contract 负责描述“如何恢复这些原始字节”。canonical contract 至少包含：

- exact schema/version 和 `purpose=deployability_only`；
- candidate ID、seed、epoch、manifest path/size/SHA；
- archive 的固定 logical ID、文件名、size、SHA 和精确成员集合；
- 每个 runtime asset 的 role、固定仓库相对 destination、size、SHA；
- candidate 五文件和 preprocessing 三项的交叉绑定；
- detector 的获批 provenance token、固定 basename、size、SHA；
- tracked configs、active source 和 package ByteTrack config 的 path/size/SHA 或版本绑定；
- 必要依赖版本；
- source commit/tree 或批准 patch archive，以及关键 runtime 文件 descriptors；
- 明确声明 `reads_video=false`、`runs_tracker=false`、`runs_qc=false`、`runs_forward=false`、`changes_c0_c3=false`。

archive 编码也属于契约：必须冻结 archive format/version、成员排序、timestamp、mode、compression 和 host metadata，且 canonical contract 自身不得放入被其 SHA 绑定的 archive 形成哈希循环。builder 使用明确的 owner-approved source descriptor 和代码内冻结的编码规则先生成 archive/observed descriptor；随后人工或独立复算 archive identity，再冻结供 restore 使用的 canonical contract。两类 descriptor 不得混淆。

contract 不得包含 URL credential、token、用户目录、机器绝对路径或可触发自动下载的 fallback。若 detector descriptor 尚未由负责人批准，schema/parser 和测试只使用 `tmp_path` 临时 contract；不得生成 canonical `configs/modules/wandering_camera_portable_v1.yaml` 的 fixture/placeholder。production CLI 在 canonical contract 不存在时必须确定性返回 `blocked/asset_source_unavailable`，不得输出 `ready`。

## 7. 首次实现文件面（已形成，待 F1 修复）

首次实现形成了以下文件面；它不是当前 AI 重新创建文件的清单，当前修改范围以 F1 任务书为准：

```text
# 仍未形成；只在 detector exact identity 与批准 archive 已到位后允许形成：
configs/modules/wandering_camera_portable_v1.yaml
src/elderly_monitoring/modules/mental_health/wandering/camera_portability.py
scripts/wandering/build_camera_runtime_asset_bundle.py
scripts/wandering/restore_camera_runtime_assets.py
scripts/wandering/preflight_camera_runtime.py
tests/test_wandering_camera_portability.py
```

为关闭 video-direct 的静默下载风险，可在必要时最小修改：

```text
src/elderly_monitoring/modules/mental_health/wandering/camera_collection.py
tests/test_wandering_camera_collection.py
```

首次实现已新增：

```text
reports/mental_health/wandering_camera_portable_v1/README.md
reports/mental_health/wandering_camera_portable_v1/VERIFICATION.md
```

如果必须修改 `camera_primary_inference.py` 或 `camera_development.py`，只能提取/复用窄校验函数，并先证明现有接口无法安全复用。需要修改 `release.py`、`model.py`、共享 tracker、其他模块或依赖时必须暂停请求授权。

## 8. 首次实现顺序（已执行，禁止作为当前任务重跑）

本节保留首次实现的设计基线。独立审计已经证明，仅重复这些测试和实现步骤不足以关闭 source self-sign、swallowed network attempt、no-clobber 和完整 runtime trust closure；这些缺口已由 F1 任务书收口，不得继续堆叠无关 synthetic 治理。

### P0：现场复核

先核对 AGENTS、任务表、技术文档2、本文、模块 README、F2 报告、operator checklist、Git 现场、editable 安装和当前资产。不得搜索媒体目录。

### P1：失败测试先行

先建立 exact schema、固定 candidate、CLI 参数面、`ready|blocked` 状态和禁止操作的失败回归。至少覆盖：

- absolute/traversal、Windows drive/UNC、反斜杠、URL/credential、控制符、超长 path；
- duplicate member/destination、DrvFS case alias、symlink、额外或缺失成员；
- archive/member size 或 SHA 漂移、同长度篡改、解压尺寸上限；
- fake checkout、外部 config copy、editable root 或依赖版本不符；
- final/target 已存在、partial destination、staging/commit 中途失败；
- candidate、preprocessing、detector、ByteTrack config 任一缺失或漂移；
- production preflight 的 network/download helper、视频、tracker、QC、candidate forward、receipt/collection/annotation reader 全部零调用；missing detector 必须在导入或构造 `YOLO` 前 blocked；
- preflight CLI 的 stdout、结构化报告与退出码一致：`ready=0`、预期 deployability `blocked=2`、参数/内部错误 `=1`。

### P2：离线 bundle 与 restore

builder 只接受明确存在且 SHA 匹配的本地固定资产，不下载、不搜索未知缓存；输出只允许 Git 外 fresh 路径。archive 必须按 contract 冻结的 format/version、成员顺序、timestamp、mode、compression 和 host metadata 确定性编码。缺 detector 或任一资产漂移时，必须在创建 final 前失败。

restore 固定顺序：

1. 校验 active checkout 和 canonical contract；
2. 只读验证 archive 自身及全部成员；
3. 拒绝 path escape、symlink、重复、case collision、额外/缺失成员和尺寸攻击；
4. 确认所有由 restore 管理的 ignored destination 都不存在；tracked configs 不属于 restore destination；
5. 对 candidate、preprocessing、detector 各原子组分别在其 final 的同文件系统 sibling staging 写入并回读 size/SHA；
6. 全部 staging 都通过后再按固定顺序逐组 atomic rename，并记录本次调用创建的每个目标；
7. 后续组提交失败时只回滚本次调用已经创建的目标并复核无残留；全部组成功后最后通过 sibling staging、回读和 atomic rename 写入 content-bound ready marker。marker 写入、回读或 rename 失败也必须回滚本次创建的全部资产组和 marker，并复核零残留。

这是 failure-atomic transaction，不得描述成跨多个目录的一次 filesystem-level 原子 rename。restore 不覆盖、合并或“修复”已有目标。已有正确资产由 preflight 验证；部分 restore-managed 目标已存在时整体 blocked。

### P3：零视频 runtime preflight

固定检查顺序：

1. active checkout、caller root 和 editable installation 身份；
2. canonical portable contract、source trust root 与 tracked configs；
3. Python/package distribution 依赖与 PEP 440 版本约束；
4. runtime asset regular-file、destination、size、SHA；
5. preprocessing 交叉绑定；
6. active runtime critical files、`model.py/release.py` 与 candidate source identity；
7. 使用现有 loader 做真实 candidate CPU safe-load；
8. 显式本地 detector weight 的 hash 与 loadability；
9. package ByteTrack config 的 identity 与可解析性。

safe-load 不是 inference。PORTABLE 专用测试必须在模块导入前对 `socket.connect`/`socket.create_connection`、HTTP/download helper 和可疑 subprocess 建立 bomb，并证明 candidate `forward`、detector inference、tracker、视频、QC 和 network attempt 都为零。正式隔离验收还必须在系统级 network-denied 环境运行，不能只靠硬编码字段声称无网络。

preflight 输出只包含结构化 deployability 状态、固定 check ID、`ready|blocked`、安全 logical identifiers、预期/观察 size/SHA 和固定 blocker code。不得输出本机绝对路径、credential、accuracy、F1、FAR、风险等级或 C0-C4 晋级字段。若写文件，必须使用 fresh output、staging 回读和原子提交。

CLI 状态与退出码固定为：`ready=exit 0`、预期 deployability `blocked=exit 2`、参数/contract 语法/内部错误 `=exit 1`。报告必须同时记录 `network_denial_guard_enforced=true|false` 与 `network_attempt_count`；只有 guard 实际生效且 attempt count 为 0 时，才能表述“本次隔离验收未观察到网络访问”。

### P4：video-direct detector 闭锁

在 wandering 专用 controller 内，保持既有顺序：

```text
active checkout/config
  → receipt
  → collection/scope/source
  → output/operator metadata
  → portable detector contract + local weight size/SHA
  → video hash/probe/runtime/tracker
  → private atomic C1 commit
```

模块 import 时不得预读 portable contract 或 detector。无效 receipt/collection/source 时，portable detector 文件也不得被 resolve/stat/read/hash。只有全部 owner gate 通过后，才惰性形成两个不同值：`detector_model_id="yolov8n.pt"` 只用于 sidecar/summary/component validation，`detector_model_path=<verified absolute path>` 只传给共享 `run_yolov8_bytetrack()`。不得把现有 portable metadata 常量直接替换成绝对路径。

在调用 `YOLO` 前应再次 stat/hash detector；network guard 必须覆盖 preflight loadability 和正式 controller 的整个 `runtime.run()` 生命周期，因为共享 tracker 会在其中再次构造 `YOLO(model_path)`。不为此扩展成复杂 `openat`/同进程恶意替换防护。portability 失败必须被 controller 映射为既有 `CameraCollectionPreparationError`（或保持其既有可捕获错误层级），且不能污染 final provenance。

### P5：隔离源码验收

不得污染或清理当前用户工作树。无 commit 授权时，可用 `git archive HEAD` 加精确 allowlist patch/new files 构造临时隔离源码；验收报告必须记录 base HEAD、patch SHA、每个新增文件 SHA、精确 allowlist，并证明隔离树没有额外源码文件。获得 commit 授权后可用 clean checkout。两种方式都不得复制 ignored runtime 资产作为源码内容。

临时 overlay 只能证明“当前实现机制在隔离环境可验证”。在代码进入不可变 Git commit/tree，或形成负责人认可且可持续恢复的 SHA-bound patch archive 前，formal PORTABLE 仍必须 `blocked/source_integration_pending`，不得把临时补丁等同 fresh clone 的持久 source trust root。

隔离验收至少证明：

1. 未恢复资产时返回明确 `blocked`；
2. 从获批离线 bundle 恢复后 candidate 真实 safe-load、detector 本地 loadability、preprocessing 和 ByteTrack binding 全过；
3. 分别删除、错 SHA、同长度篡改一个成员时都返回 `blocked`；
4. 在系统级 network-denied guard 下运行，报告 guard 状态与 attempt count；全程无媒体、无 tracker、无 QC、无模型 forward；
5. 如临时把 editable install 指向隔离源码，必须在 `finally` 恢复到主 checkout，并再次用 `pip show` 和模块路径核验。

如果没有获批 detector、持久 archive 或持久 source identity，仍应完成可独立验证的代码与负向隔离测试，但 formal PORTABLE 状态保持 `blocked`。

### P6：回归和证据

先运行新窄测，再运行 release/candidate loader、collection、primary inference、development 相关窄测，最后按风险运行九个 camera 文件：

```text
tests/test_wandering_camera_adapter.py
tests/test_wandering_camera_collection.py
tests/test_wandering_camera_corruption.py
tests/test_wandering_camera_dataset.py
tests/test_wandering_camera_development.py
tests/test_wandering_camera_episode.py
tests/test_wandering_camera_inference.py
tests/test_wandering_camera_primary_inference.py
tests/test_wandering_camera_qc.py
```

新增 CLI 与既有相关 CLI 的 `--help` 必须退出码 0，最后运行 `git diff --check`。九文件回归允许 `tmp_path`/in-memory synthetic fixture、fake tracker/QC/forward；只能报告为兼容性自动化证据，不能削弱 production preflight 和隔离验收的零调用要求。完整仓库 pytest 不是本任务硬门。

## 9. 状态机与完成标准

### 9.1 未开始

```text
status=wandering_m0cam_c0_c1_handoff_hardened_waiting_owner_inputs
m0cam_portable_status=not_started
```

### 9.2 执行中

```text
m0cam_portable_status=in_progress
portable_restore_status=not_run|blocked|ready
portable_preflight_status=not_run|blocked|ready
portable_isolated_safe_load=false|true
portable_candidate_safe_load=false|true
portable_detector_asset_verified=false|true
network_denial_guard_enforced=false|true
network_attempt_count=<non-negative integer>
```

### 9.3 实现完成但外部资产阻塞

允许以下诚实终态：

```text
m0cam_portable_status=blocked
portable_blocker_code=asset_source_unavailable|detector_weight_missing|asset_sha_mismatch|dependency_version_mismatch|editable_checkout_mismatch|active_source_or_config_drift|isolated_restore_failed|candidate_safe_load_failed|network_guard_unavailable|source_integration_pending
```

这表示软件可以正确拒绝运行，不表示 PORTABLE pass。

### 9.4 通过

只有代码已有不可变/获批持久 source identity，并且获批持久资产源、detector exact identity、隔离 restore、candidate safe-load、detector loadability、preprocessing/ByteTrack binding 和全部负向门禁都通过，才可写：

```text
m0cam_portable_status=passed
portable_restore_status=ready
portable_preflight_status=ready
portable_isolated_safe_load=true
portable_candidate_safe_load=true
portable_detector_asset_verified=true
network_denial_guard_enforced=true
network_attempt_count=0
m0cam_d_started=false
```

全局 `status` 必须根据任务表中 PORTABLE 与并行拍摄线的实时组合派生，不能由代码 AI硬编码成 `waiting_owner_inputs`。每次更新任务表时必须保留拍摄线当时的真实 C0/C1 值；代码 AI 不得自行覆写。`evidence_scope` 继续保持当前事实值，PORTABLE deployability 不创造 camera evidence。

### 9.5 F1 实现尝试与后续复审（未接受）

首次实现之后的 F1 曾按原任务书完成修复和自动化验证，但后续独立复审仍构造出四个 P0，因此不能接受为软件审计通过。实时状态继续区分待返工的软件审计与 production runtime 两个维度：

```text
m0cam_portable_phase_status=wandering_m0cam_portable_blocked
m0cam_portable_status=blocked
portable_software_audit_status=rework_required
portable_review_blocker=post_f1_findings_open
portable_runtime_blocker_code=asset_source_unavailable
source_integration_pending=true
m0cam_portable_f1_acceptance_status=rework_deferred
```

F1 执行时测试数字仍是历史证据，但没有覆盖 source identity 自引用/replace-ref、实际 runtime import closure、restore 假 `ready` 和 rollback ownership。当前先推进不依赖 PORTABLE 的 `M0-CAM-MVP-1`；进入 fresh-clone 部署、正式交付或 M0-CAM-D 前，必须恢复最小返工并完成 formal closure。formal closure 另外仍受 owner-approved detector/persistent archive、精确外部路径访问授权和持久源码身份约束。

## 10. 必须暂停并请求负责人输入

只有以下情况需要暂停；普通代码、路径、Windows/WSL、editable、timeout、局部测试和文档问题由执行 AI 自主诊断、修复、复测并继续：

- 需要获批 `yolov8n.pt` 字节、exact size/SHA、官方不可变下载授权或持久外部资产位置；
- 需要改变 candidate/model/threshold/label/split、preprocessing bytes 或科学协议；
- 需要访问任何 camera/WP raw/SmartCare official/raw/sealed 数据；
- 需要修改共享 tracker、其他模块、`environment.yml` 或共享依赖；
- 需要覆盖/删除既有资产或证据；
- 需要 stage、commit、push、发布、Git LFS 或远端制品上传。

## 11. 执行 AI 验收输出格式

最终交接必须分开报告：

1. **计划/设计**：本次采用的 contract、恢复和 preflight 设计；
2. **已实现代码**：精确文件与行为；
3. **自动化/synthetic 证据**：测试命令、通过数、攻击矩阵和隔离负向结果；
4. **真实 camera/产品/临床证据**：本任务应明确为“无”；
5. **PORTABLE 状态**：`passed` 或带固定 blocker code 的 `blocked`；
6. **Git 现场**：branch、HEAD、stage、dirty/untracked、是否 commit/push；
7. **下一输入**：只有确实缺失时列出 owner 必须提供的最小资产材料。

不得用“代码已实现”替代“隔离恢复已通过”，也不得用“preflight ready”替代 C0/C1 或 M0-CAM-D。

## 12. 首次实现提示词（已退役）

本文 1.0 版曾在此提供首次实现提示词。该提示词已经执行完毕，独立审计又发现了它没有覆盖的假 `ready`、no-clobber、runtime closure 和路径/事务缺口，因此不得继续复制或让新 AI 重做原 P0-P6。

F1 已按 [M0-CAM-PORTABLE-F1 执行任务书](M0-CAM-PORTABLE-F1执行任务书.md)完成。如果 owner detector/持久资产源、精确路径访问授权或持久源码集成授权仍缺失，只列缺失材料并停止；不得生成 placeholder、搜索缓存、下载权重或继续添加无关 synthetic 治理。
