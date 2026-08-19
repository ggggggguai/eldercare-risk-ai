# TopoWander-MPT M0-CAM-PORTABLE

> **当前接受状态已被后续复审更新：** 本文保留 F1 执行时命令、测试和观察结果，不再作为 `portable_software_audit_status=passed` 的实时依据。现行状态为 `rework_deferred/rework_required/post_f1_findings_open`，见 [F1_REAUDIT_20260814.md](F1_REAUDIT_20260814.md)和任务表。

## 1. 结论

以下是 F1 执行结束时记录的结论快照；后续复审已经撤销其中的“软件审计通过”，formal PORTABLE 也仍未通过：

```text
status=wandering_m0cam_portable_blocked
evidence_scope=synthetic_schema_contract_only
portable_restore_status=blocked
portable_preflight_status=blocked
portable_candidate_safe_load_verified=true
portable_detector_asset_verified=false
network_denial_guard_enforced=true
network_attempt_count=0
portable_runtime_blocker_code=asset_source_unavailable
source_integration_pending=true
m0cam_portable_f1_complete=true
portable_software_audit_status=passed
portable_review_blocker=none
C0=false
C1=false
authorized_camera_data_consumed=false
m0cam_d_started=false
```

F1 的实现意图是关闭 source self-sign、swallowed network attempt、final overwrite race、runtime trust closure、canonical path/rollback、detector token 和 blocker/exactness 缺口；后续复审证明 source identity、完整 runtime closure、restore commit-time exactness 和 rollback ownership 仍未闭合。除此之外，formal 还有两个独立阻断：没有负责人批准的 `yolov8n.pt` exact bytes、size、SHA-256、provenance token 和可持久恢复的离线 archive 来源，所以 production CLI 仍为 `asset_source_unavailable`；本次源码仍是未提交 overlay，在进入不可变 Git tree 或形成负责人批准的持久 SHA-bound patch archive 前，继续 `source_integration_pending=true`。canonical config 被故意保持不存在；实现没有搜索缓存、下载或生成 placeholder 权重。

## 2. F1 后的软件面及审计状态

- `camera_portability.py` 提供 exact-schema canonical contract 与 owner source descriptor parser；拒绝 absolute/traversal、Windows drive/UNC、反斜杠、URL/credential、控制符、超长路径、重复键、重复或大小写碰撞 member/destination；
- builder 只接受显式存在且 size/SHA 匹配的本地资产，不搜索、不下载；ZIP 固定为 stored compression、lexicographic member order、1980 timestamp、`0600` regular-file mode 和 stripped host metadata，并输出供独立复算的 observed descriptor；
- restore 先验证 archive 和全部成员，再为 candidate、preprocessing、detector 三组建立 sibling staging，全部回读通过后按固定顺序提交；builder、三组、ready marker 和 preflight report 的 final commit 都使用 atomic no-replace。对进程内可捕获失败执行 best-effort 清理并按逻辑角色报告 residue；不声称断电或进程被杀后的 crash atomicity；
- `git_tree` preflight 将 active HEAD、canonical contract 当前字节和每个 runtime trust root 分别与指定 commit blob 独立比对，contract descriptor 只作交叉检查；冻结 candidate manifest 继续交叉绑定 `model.py/release.py`。没有 contract 外 approval anchor 的 patch 模式明确 `source_integration_pending`；
- `camera_collection.py` 在 receipt、collection/scope/source、output 和 operator metadata 全部通过后才解析不可变 detector token。tracker 前只按同一 token 二次核验 path/size/SHA/contract/marker identity，sidecar 仍只记录 `yolov8n.pt`；network guard 在作用域退出时检查本次 attempt delta，即使下游吞掉拒绝异常也 fail closed；
- canonical/asset 路径拒绝 symlink/reparse、case alias、escape 和非精确 active identity；ready marker 要求无重复键的 exact canonical JSON bytes；预期 archive/ByteTrack/source 缺失映射为结构化 deployability blocker；
- production restore/preflight 不导入或读取 camera receipt、collection、tracking、sidecar、annotation 或视频，不运行 tracker、Camera QC、window preparation、candidate forward、evaluator 或 M0-CAM-D。

## 3. 固定身份复核

候选五件套在本机只读复核全部匹配任务书：

| 文件 | size | SHA-256 |
|---|---:|---|
| `candidate_manifest.json` | 12,642 | `3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7` |
| `model_state.npz` | 838,934 | `94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031` |
| `forward_config.yaml` | 4,469 | `debd3adcf0c84d266ca594fd9ba2716a5a854e074e6e2092570b1ec656ed80f7` |
| `performance_config.yaml` | 1,840 | `ecc4c5a00dc9c30b1a4a16b943d39d9de85cce27077c8ccba1ba223c529de5f2` |
| `frozen_wp_rf_config.yaml` | 3,398 | `d29db531f42ae5d7cf4ed540a85681ddab78e483695633ccc8b382d98e745e35` |

预处理最小闭包也匹配：tracked config `5b6924...ff45`，ignored manifest `242072...8593`，ignored feature stats `249ad8...4c8d`。没有读取或打包 79 MB `samples.jsonl`、preprocessing report、WP upstream、checkpoint 或其他训练资产。

项目 editable 安装指向当前 checkout；Python `3.11.15`、项目 distribution `0.2.0`、PyYAML `6.0.3`、NumPy `2.4.6`、Torch `2.13.0`、OpenCV `4.13.0.92`、Ultralytics `8.4.78` 均匹配锁定要求。包内 `cfg/trackers/bytetrack.yaml` 为 856 bytes，SHA-256 为 `395701d947a749179dee3e327b1181b730c4ca98e7cac5a2ab05280aae573b8b`。

## 4. 验收事实

- 聚焦 PORTABLE + collection：`100 passed`；旧实现下新增 source-self-sign 回归直接得到错误 `ready`，swallowed-attempt 回归也暴露 guard 退出未检查 delta；修复后均通过；
- 受影响的 release/candidate/primary/development 窄测：`79 passed, 4 subtests passed`；
- PORTABLE 加任务书指定九个既有 camera 文件：`297 passed`；唯一 warning 来自故意构造 duplicate ZIP member 的攻击 fixture。这些属于自动化/synthetic 软件兼容性证据，完整 camera 回归包含 synthetic tracking 上的真实 fixed-candidate forward 和 QC，不能写成全部 fake/in-memory forward；
- 本次隔离验收使用 `git archive` base HEAD `bfc3329ad14d145e6b4c3fb836a0a525f30e970b` 加七文件精确 overlay，在 `unshare -Urn` 系统级断网 namespace 中完成独立 venv、`--no-deps --no-build-isolation` 本地 editable 安装和测试：`100 passed`，无 skip；未复制 ignored runtime asset；
- 当前固定 candidate 在 `unshare -Urn` 与应用网络拒绝器双重保护下由既有 `load_candidate_from_manifest()` 真实 CPU safe-load：manifest SHA 精确匹配、`model.training=false`、参数设备只有 CPU、forward 调用 0、network attempt 0；
- production preflight 返回 `blocked/asset_source_unavailable`：active/editable checkout 通过，canonical contract/source trust 检查阻塞，后续 candidate/detector/ByteTrack 与任何 camera 路径均未触发，network attempt 为 0。专门回归已证明 swallowed attempt 也会阻止 preflight/controller final。

以上三项是执行时曾记录、现已被后续复审取代的结论。当前接受状态是 `m0cam_portable_f1_acceptance_status=rework_deferred`、`portable_software_audit_status=rework_required`、`portable_review_blocker=post_f1_findings_open`；formal blocker 也保持不变。

更详细的命令与攻击覆盖见 [VERIFICATION.md](VERIFICATION.md)。

## 5. 不能据此声称的内容

本报告只证明列出的实现和自动化命令在当时得到相应结果；后续复审证明这些证据不足以通过 F1 软件审计。它也不证明 detector 正式资产已恢复、fresh clone formal PORTABLE 已通过，或 C0/C1、目标机位、camera accuracy/F1/FAR、真实 episode、产品延迟、老人域或临床有效性。截至当前仍不允许启动 M0-CAM-D。

下一次合法推进是等待负责人提供获批 detector exact identity 与持久离线资产源，并取得精确外部路径访问和持久源码集成授权；材料未齐时停止。材料齐备后才执行 FORMAL-CLOSURE：独立复算 archive、冻结 canonical contract，并在不可变/获批持久 source identity 的隔离源码中执行成功 restore、candidate safe-load、detector local loadability 和完整 preflight。不得通过下载、缓存搜索、改 SHA、改 candidate、降低 gate 或使用 placeholder 制造 `ready`。
