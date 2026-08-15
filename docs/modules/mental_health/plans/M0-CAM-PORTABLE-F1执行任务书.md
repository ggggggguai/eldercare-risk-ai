# M0-CAM-PORTABLE-F1 执行任务书

版本：1.1

更新时间：2026-08-14

状态：实现尝试完成；后续独立复审未接受，`rework_deferred`

> **2026-08-14 后续复审注记：** 本文记录的实现过程和测试数字仍是执行时事实，但不能继续作为当前 `passed` 结论。后续复审构造出 source identity 自引用/replace-ref、runtime import closure、restore 假 `ready` 和 rollback ownership 四个 P0。现行状态以[任务表](../../../tasks/README.md)和 [F1 复审记录](../../../../reports/mental_health/wandering_camera_portable_v1/F1_REAUDIT_20260814.md)为准；rework 延后到正式部署或 M0-CAM-D 前，不阻塞 `M0-CAM-MVP-1`。

## 1. 事务定位

`M0-CAM-PORTABLE-F1` 的全名是：

> source-trust、network-zero 与 fresh-restore 审计收口

它不是新建第二套 portability，也不是继续堆叠无关 synthetic 治理。它只修复 `M0-CAM-PORTABLE` 首次实现中已经被独立审计证实、可能产生假 `ready`、覆盖竞争目标或留下不诚实证据的缺陷。

本文是 [M0-CAM-PORTABLE 首次实现任务书](M0-CAM-PORTABLE执行任务书.md)的下位修复任务。实时状态仍只看[任务表](../../../tasks/README.md)；稳定路线看[技术文档2](徘徊识别技术文档2.md)；当前实现能力和限制看[模块 README](../README.md)；实际命令与结果写入 [PORTABLE 报告](../../../../reports/mental_health/wandering_camera_portable_v1/README.md)。

执行时状态曾记录为：

```text
m0cam_portable_f1_implementation_attempt_complete=true
m0cam_portable_f1_acceptance_status=rework_deferred
portable_software_audit_status=rework_required
portable_review_blocker=post_f1_findings_open
m0cam_portable_status=blocked
portable_runtime_blocker_code=asset_source_unavailable
source_integration_pending=true
C0=false
C1=false
authorized_camera_data_consumed=false
m0cam_d_started=false
```

现场验证为聚焦 `100 passed`、相关窄测 `79 passed, 4 subtests passed`、完整 camera `297 passed`、七 CLI help 通过、七文件 overlay 的 `unshare -Urn` 回归 `100 passed`，以及固定 candidate CPU/eval safe-load 的 forward/network attempt 均为 0。这些数字继续保留，但后续复审已证明其没有覆盖全部完成门；本文后续内容属于原执行设计，不覆盖当前 `rework_deferred` 状态。

## 2. 已核验现场与证据边界

制定本文时的现场为：

```text
branch=feat/wandering-data-pipeline
HEAD=bfc3329ad14d145e6b4c3fb836a0a525f30e970b
stage=empty
camera_handoff_status=wandering_m0cam_c0_c1_handoff_hardened_waiting_owner_inputs
m0cam_portable_phase_status=wandering_m0cam_portable_blocked
m0cam_portable_status=blocked
portable_software_audit_status=rework_required
portable_review_blocker=software_audit_findings_open
portable_runtime_blocker_code=asset_source_unavailable
source_integration_pending=true
C0=false
C1=false
authorized_camera_data_consumed=false
m0cam_d_started=false
```

已复现但不能越级解释的证据：

- PORTABLE + 九个 camera 文件：`265 passed`；这是自动化/synthetic 兼容性证据，其中既有 primary 测试会在 synthetic tracking 上运行真实 fixed-candidate forward 和 QC；
- base HEAD + 七文件 overlay 的系统级断网隔离：`67 passed, 1 skipped`；skip 只因 Linux `/tmp` 无 DrvFS case alias；
- 当前固定 candidate 在 CPU/eval 下独立 safe-load，`forward=0`、`network_attempt_count=0`；这只证明当前字节可加载；
- production preflight 因 canonical contract 不存在返回 `blocked/asset_source_unavailable`；
- 没有 production canonical contract、approved detector、`READY.json`、C0/C1、真实 camera、产品或临床证据。

265、67+1 和 safe-load 均不能覆盖后续红队发现，也不能把 PORTABLE 写为 passed。

## 3. 为什么必须重开

### 3.1 P0：source identity 可以自签

当前 canonical contract 自带 source descriptors，校验又只按同一 contract 检查当前工作树；`git_tree` 只比较 HEAD 字符串，没有将关键源码和 canonical contract 字节与指定 Git object/tree 独立比对。`approved_patch_archive.base_git_commit` 也没有实际参与信任验证。

结果是：同一脏工作树可以同时修改代码和 contract，再填写自身 SHA，存在制造 `ready` 的通路。

### 3.2 P0：被吞掉的网络尝试可以继续 ready

当前应用 guard 会记录并抛出网络拒绝异常，但退出时不检查本次作用域的 attempt delta。如果 candidate loader、detector loader 或 tracker 捕获该异常后继续返回，preflight/controller 仍可能成功。独立最小复现已经观察到：

```text
guard_exited_successfully=true
attempt_delta=1
```

### 3.3 P1/P2：事务和身份闭包不完整

已确认的后续缺口包括：

- runtime trust-root 清单没有覆盖实际 development import graph 中的 adapter、QC、inference、episode、preprocessing 和正式 CLI；
- fresh-only 最终提交使用可覆盖目标的 `Path.replace()`，存在检查后目标出现的竞态；
- rollback 对可捕获中断、单项清理失败和最终残留复核不完整，文档对 failure-atomic 的措辞过强；
- canonical contract 只检查文件自身 symlink，未完整关闭父目录 symlink/junction、DrvFS case alias 和最终 active-checkout 身份；
- detector 二次校验会重新加载可变 contract，没有绑定第一次得到的 path/size/SHA/contract/marker identity；
- missing archive、ByteTrack resource 等预期 deployability 失败不一定稳定映射为 `blocked/exit 2`；
- ready marker 只比较解析后的 JSON 语义，没有验证 canonical bytes、重复键和 exact encoding；
- PORTABLE 报告把完整 camera 回归误写成全部 fake/in-memory forward，并把“未运行 QC/forward”的作用域写得过宽。

## 4. 唯一目标

F1 只完成以下目标：

1. 让 source trust 不能由当前 contract/工作树自签；
2. 让任一被观察到的网络尝试即使被下游吞掉，也必然阻止 `ready` 和 final；
3. 让所有 fresh-only final 使用真正 no-clobber 的提交语义；
4. 补全实际 runtime source/config/package closure，并保持与冻结 candidate source identity 的交叉绑定；
5. 关闭 canonical path alias、可捕获失败回滚和 detector 二次身份漂移；
6. 让预期 deployability blocker、CLI 退出码、ready marker exactness 和报告措辞一致；
7. 形成先红后绿、可独立复现的软件证据。

F1 完成后仍必须保持 formal PORTABLE blocked，直到 owner-approved detector/persistent archive、精确外部路径访问授权和持久源码身份全部真实到位。

## 5. 非目标与禁止项

F1 不得：

- 读取、搜索、哈希或运行真人/授权 camera、receipt、collection、tracking、sidecar、annotation、WP raw、SmartCare official/raw 或 sealed 数据；
- 下载、搜索未知缓存或猜测 `yolov8n.pt` 的 bytes/size/SHA/provenance；
- 生成 production canonical placeholder、fake detector、owner acceptance 或 `READY.json`；
- 运行真实 tracker、M0-CAM-D 或 evaluator；
- 重训、换 candidate、改 seed/epoch/model/threshold/labels/split/matching/uncertain/episode policy；
- 修改 `model.py`、`release.py`、共享 `fall_risk/tracking.py`、其他模块、`environment.yml` 或共享依赖；
- 建设 PKI、签名服务、远端 registry、Git LFS、通用资产平台或与已确认缺陷无关的新治理框架；
- 删除或覆盖既有资产、报告、用户工作树或不明 untracked；
- 未经当前用户明确授权 stage、commit、push 或发布；
- 把 fixture、pytest、overlay 或 safe-load 描述为 camera、老人域、产品或临床证据。

既有完整 camera 回归可以在 synthetic tracking 上运行当前 fixed candidate/QC；必须明确标为自动化兼容性证据。F1 production preflight、独立 safe-load 和隔离 portability 验收本身仍必须保持 video/tracker/QC/forward 为零。

## 6. 允许修改的最小文件面

默认允许：

```text
src/elderly_monitoring/modules/mental_health/wandering/camera_portability.py
src/elderly_monitoring/modules/mental_health/wandering/camera_collection.py
scripts/wandering/build_camera_runtime_asset_bundle.py
scripts/wandering/restore_camera_runtime_assets.py
scripts/wandering/preflight_camera_runtime.py
tests/test_wandering_camera_portability.py
tests/test_wandering_camera_collection.py
reports/mental_health/wandering_camera_portable_v1/README.md
reports/mental_health/wandering_camera_portable_v1/VERIFICATION.md
docs/tasks/README.md
docs/modules/mental_health/README.md
docs/modules/mental_health/plans/徘徊识别技术文档2.md
docs/modules/mental_health/plans/M0-CAM-PORTABLE执行任务书.md
docs/modules/mental_health/plans/M0-CAM-PORTABLE-F1执行任务书.md
docs/README.md
README.md
```

如果失败回归证明必须复用 `camera_primary_inference.py` 或 `camera_development.py` 中既有窄校验，可以先报告原因后做最小提取，不得改变推理、QC、threshold 或 evaluator 行为。任何需要修改冻结/共享文件或扩大到其他模块的方案都必须暂停。

## 7. TDD 执行顺序

### F1-P0：现场复核

完整阅读 AGENTS、任务表、技术文档2、两份 PORTABLE 任务书、模块 README、PORTABLE/C01-F2 报告、operator checklist 和相关源码/测试。先执行 Git、stage、HEAD、editable 安装和导入路径的只读检查；保留用户现场。

### F1-P1：先建立旧实现失败回归

任何修复前，至少建立以下能在旧实现上失败的最小测试：

1. 同一 HEAD 下同时修改 runtime source 与 contract descriptor，必须 blocked；
2. `git_tree` descriptor 与指定 commit blob 不一致，必须 blocked；canonical contract 自身漂移也必须 blocked；
3. patch base commit、allowlist、patch SHA 或应用后文件任一不符，必须 blocked；若 v1 无法得到 contract 外的持久 approval anchor，则 patch 模式必须明确 fail closed；
4. adapter/QC/inference/episode/preprocessing/正式 CLI 任一关键 source 漂移，在 candidate/detector load、媒体或 tracker 前 blocked；
5. candidate loader、detector loader 和 `runtime.run()` 分别尝试网络、捕获拒绝异常并返回时，preflight/controller 仍必须 blocked，且 controller 无 final；
6. candidate、preprocessing、detector、builder output、ready marker、preflight report 在最终提交前出现竞争目标时，原字节保留且整体失败；
7. 第二/第三资产组、marker 写入/回读/提交、可捕获中断和单项 rollback 清理失败时，结果与残留报告必须诚实；
8. canonical 父目录 symlink/junction、DrvFS case alias、外部 config copy、UNC/device namespace source 必须拒绝；
9. 第一次 detector 解析后切换 contract/marker/weight，第二次校验必须拒绝；
10. missing archive、ByteTrack missing/unreadable、Git/source drift 等预期 deployability 失败必须为结构化 `blocked/exit 2`；
11. ready marker 重复键、非 canonical bytes、额外或漂移字段必须拒绝。

不得先改实现，再补只会顺从新实现的测试。最终报告需保存“旧实现失败 → 修复后通过”的最小证据。

### F1-P2：非循环 source trust

`git_tree` 至少满足：

- active HEAD 与指定 commit 一致；
- canonical contract 当前字节必须与指定 commit 中的对应 blob 一致，不能由自身内容证明自身可信；
- 每个 runtime trust root 的当前字节必须与指定 commit/tree 中的 blob 独立比对；contract descriptor 只能作为交叉检查，不能作为唯一信任根；
- `model.py/release.py` 继续与冻结 candidate manifest 的 source identity 交叉核验；
- dirty/untracked overlay 只能得到 `source_integration_pending`，不能 formal ready。

`approved_patch_archive` 至少绑定 base commit、外部批准 SHA、精确 allowlist、patch bytes 和应用后结果。若当前输入中没有独立 approval anchor，最安全的 F1 行为是明确 blocked，而不是用同一 contract 内字段代替 owner approval。

### F1-P3：network-zero

应用 guard 必须记录进入时的计数与本次作用域 attempt delta。无论内部异常是否传播，只要 delta 大于零：

- preflight 不得 `ready`；
- detector/candidate loadability 不得记为 ready；
- video-direct controller 必须转换为既有可捕获 preparation error；
- tracking/sidecar/summary 不得提交；
- socket 函数必须在退出时恢复。

应用 guard 只证明受控 Python 连接入口的拒绝。正式隔离证据仍要求 `unshare -Urn` 或等价系统级断网，不能把应用 monkeypatch 描述成操作系统级全网络隔离。

### F1-P4：no-clobber、rollback 与路径

- 不得使用会覆盖现有目标的 `Path.replace()`/`os.replace()` 作为 fresh-only 最终提交；
- builder archive/descriptor、三个 restore group、ready marker 和 preflight report 都要使用 atomic no-replace 或等价 fail-closed 协议；
- rollback 只能删除本次确认创建的对象，不能删除竞争者目标；
- 对可捕获 `BaseException` 先 best-effort 清理全部项目并复核残留，再重新抛出；单项清理失败不得跳过其余清理；
- 如果不实现断电/进程被杀后的恢复，文档必须把保证限定为“进程内可捕获失败且文件系统允许清理”，不得声称 crash-atomic；
- canonical contract 每级父目录、最终文件和 active checkout 采用 samefile/真实文件身份与精确大小写边界；拒绝 symlink/junction、case alias、escape、UNC/device/URL/credential source；
- 继续保持 Git 外 fresh output、同文件系统 sibling staging 和各组提交，不宣称跨目录单次 filesystem atomic rename。

### F1-P5：不可变 detector token、blocker 与 exactness

第一次 portable detector 解析必须产生不可变 token，至少携带 path、size、SHA、contract identity 和 ready-marker identity。tracker 前只按同一 token 二次 stat/hash，不得重新加载变化后的 contract 接受另一资产。provenance 继续只保存 `detector_model_id=yolov8n.pt`，绝对路径只传共享 tracker。

预期 deployability 缺失或漂移统一映射为固定 `blocked/exit 2`；参数、contract 语法或真正内部错误才是 `exit 1`。Ready marker 使用 canonical JSON bytes 验证并拒绝重复键、额外字段和非规范编码。

### F1-P6：回归、隔离和文档

验证顺序：

1. PORTABLE + collection 聚焦测试；
2. 受影响的 release/candidate/primary/development 窄测；
3. PORTABLE 加九个 camera 文件；
4. 七个相关 CLI `--help` 和必要 blocked exit contract；
5. `git archive HEAD + 精确 allowlist overlay` 或获批 clean source 的隔离断网验证；
6. 当前 fixed candidate CPU/eval safe-load，forward bomb、network attempt 0；
7. production preflight 继续诚实 `blocked/asset_source_unavailable`；
8. Markdown UTF-8、本地链接、`git diff --check`、editable 路径、stage 和最终 Git 状态。

不要预设修复后的测试数量仍为 265/67+1，以现场新增回归后的实际计数为准。

七个 CLI 为：

```text
scripts/wandering/build_camera_runtime_asset_bundle.py
scripts/wandering/restore_camera_runtime_assets.py
scripts/wandering/preflight_camera_runtime.py
scripts/wandering/build_camera_tracking_pair.py
scripts/wandering/prepare_camera_session.py
scripts/wandering/run_camera_development.py
scripts/wandering/run_topowander_camera_inference.py
```

## 8. 原定完成门（后续复审证明未满足）

以下是 F1 执行时采用的原定完成门，仅用于追溯。后续复审已证明这些条件没有被全部满足，因此当前禁止写出下列三项完成状态；实时接受状态仍是 `rework_deferred/rework_required/post_f1_findings_open`。

只有同时满足以下条件，才可写：

```text
m0cam_portable_f1_complete=true
portable_software_audit_status=passed
portable_review_blocker=none
```

完成条件：

- 两条 P0 都有旧实现失败、修复后通过的直接回归；
- 本文既定 no-clobber、trust closure、path、rollback、detector token、blocker/exactness 均有回归；
- 没有未解释的新 P0/P1；
- 聚焦、相关窄测、完整 camera 回归、七 CLI 和隔离断网验证通过，skip/warning 逐项解释；
- production canonical config、detector 和 ready marker 仍不存在时，真实 CLI 稳定 `blocked/asset_source_unavailable`、network attempt 0，后续 loader 零触发；
- 没有读取 camera/official/sealed 数据，没有运行真实 tracker/M0-CAM-D；
- candidate、科学协议、共享代码与依赖没有变化；
- 文档准确区分首次实现、F1 修复、自动化证据、safe-load、formal 状态和 camera 证据；
- 未 stage、commit、push。

F1 完成后仍必须保持：

```text
m0cam_portable_phase_status=wandering_m0cam_portable_blocked
m0cam_portable_status=blocked
portable_runtime_blocker_code=asset_source_unavailable
source_integration_pending=true
C0=false
C1=false
authorized_camera_data_consumed=false
m0cam_d_started=false
```

只有 owner-approved detector/persistent archive、精确外部资产路径访问授权和持久源码身份全部到位，才另行启动 `M0-CAM-PORTABLE-FORMAL-CLOSURE`。

## 9. 必须暂停的情况

普通路径、Windows/WSL、editable、timeout、局部代码、测试和文档问题自主修复并继续。以下情况必须暂停：

- 需要 owner-approved detector bytes/size/SHA/provenance、持久 archive 或访问其外部路径；
- 需要生成 production canonical contract、`READY.json` 或声称 owner acceptance；
- 需要修改冻结 candidate、`model.py`、`release.py`、共享 tracker、其他模块、`environment.yml` 或共享依赖；
- 需要改变 threshold、labels、split、matching、uncertain、episode policy 或科学指标；
- 需要访问 camera、WP raw、SmartCare official/raw 或 sealed 数据；
- 需要删除、覆盖或人工清理既有资产/证据/不明用户文件；
- 需要 stage、commit、push、上传或发布。

## 10. 执行 AI 选择

推荐使用：**最高推理（max）**。

原因是本任务同时涉及供应链信任根、Git object 身份、Windows/DrvFS 路径别名、跨文件系统提交语义、异常吞噬和证据口径，错误修复可能制造不易观察的假 `ready`。如果执行入口不支持 `max`，使用 **极高（xhigh）**；不建议只使用“高”。

## 11. 原执行提示词（已退役，禁止复用）

下列提示词只保存当时执行上下文。它已被后续复审和 `M0-CAM-MVP-1` 当前优先级取代，不得再次交给工作 AI；新的工作 AI 只使用 [MVP-1 任务书](M0-CAM-MVP-1执行任务书.md)中的现行提示词。

```text
你接任 C:\Users\lenovo\Desktop\心理算法 的 M0-CAM-PORTABLE-F1 代码任务。

推理等级：最高（max）。如果当前模型不支持 max，使用极高（xhigh）；不要只用 high。

你的唯一目标是：修复现有 M0-CAM-PORTABLE 首次实现中已经确认的 source identity 自签、被吞网络尝试假 ready、fresh-only 覆盖竞态、runtime trust closure、canonical path/rollback、detector 二次身份和 blocker/exactness 缺口。不要重做 PORTABLE 首次实现，不要扩大功能，不要启动 FORMAL-CLOSURE 或 M0-CAM-D。

第一步必须完整阅读并现场核对：
1. AGENTS.md
2. docs/tasks/README.md（唯一实时任务状态源）
3. docs/modules/mental_health/plans/徘徊识别技术文档2.md
4. docs/modules/mental_health/plans/M0-CAM-PORTABLE执行任务书.md
5. docs/modules/mental_health/plans/M0-CAM-PORTABLE-F1执行任务书.md
6. docs/modules/mental_health/README.md
7. reports/mental_health/wandering_camera_portable_v1/README.md
8. reports/mental_health/wandering_camera_portable_v1/VERIFICATION.md
9. reports/mental_health/wandering_camera_c01_f2_v1/README.md 与 VERIFICATION.md
10. reports/mental_health/wandering_camera_c01_prep_v1/OPERATOR_CHECKLIST.md
11. camera_portability.py、camera_collection.py、release.py、model.py、camera_primary_inference.py、camera_development.py
12. 三条 PORTABLE CLI、四条相关 camera CLI、相关 configs 和 camera tests
13. pyproject.toml、environment.yml、environment-reference.txt

先执行只读检查：
- git status --short
- git diff --cached --name-only
- git rev-parse --abbrev-ref HEAD
- git rev-parse HEAD
- git diff --check
- conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
- 用 eldercare-ai 确认实际导入模块位于当前 checkout

保留用户全部现场。禁止 stash/reset/checkout、清理不明目录、stage、commit 或 push。若交接与现场冲突，先报告，以当前代码、任务表和技术文档2为准。

固定候选不得变化：candidate_id=topowander-m0s-seed20260731-epoch0005、seed=20260731、best_epoch=5、candidate manifest SHA=3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7。不得修改冻结 candidate 五文件、model.py 或 release.py。

必须先红后绿。任何实现修复前，先增加能在旧实现上失败的最小回归，至少覆盖：
1. 同一 HEAD 下修改源码并同步修改 contract descriptor，仍必须 blocked；当前源码/contract 必须与独立 Git object/tree 或外部获批持久 identity 交叉验证，不能自签；
2. approved patch 的 base commit、外部批准 SHA、allowlist、patch bytes 或应用结果任一不符必须 blocked；没有 contract 外 approval anchor 时该模式 fail closed；
3. adapter/QC/inference/episode/preprocessing/正式 CLI 任一 runtime trust root 漂移必须在 load/media/tracker 前 blocked；model.py/release.py 继续与 candidate manifest 交叉绑定；
4. candidate loader、detector loader、runtime.run() 各自尝试网络并吞掉拒绝异常后，preflight/controller 仍 blocked；只要本次 attempt delta>0 就绝不能 ready 或提交 final；
5. builder output、三个 restore group、ready marker、preflight report 在最终提交前出现竞争目标时，不得覆盖或删除竞争者目标；
6. 第二/第三组、marker、可捕获中断、单项 rollback 清理失败时，继续 best-effort 清理其余项目并诚实报告 residue；不要声称 crash-atomic；
7. canonical 父目录 symlink/junction、DrvFS case alias、外部 config copy、UNC/device/URL/credential source 必须拒绝；
8. 第一次 detector 解析后切换 contract/marker/weight，二次校验必须按第一次不可变 token 拒绝；
9. missing archive、ByteTrack missing/unreadable、source drift 等预期 deployability 失败必须 blocked/exit 2；ready marker 必须 canonical bytes、exact-schema、无重复键。

然后只做关闭这些测试所需的最小修复。不要建设 PKI、签名服务、远端 registry、LFS 或新通用框架。git_tree 必须把当前 contract 和 runtime roots 与指定 commit/tree blobs 独立比对；approved patch 如果没有可信外部 anchor，就保持 source_integration_pending。应用 network guard 必须在作用域退出时检查本次 attempt delta，即使下游吞掉异常也 fail closed；正式隔离仍使用系统级 network-denied 环境。fresh-only final 不得使用可覆盖目标的 replace 语义。

默认只修改 F1 任务书的最小 allowlist。若必须修改 camera_primary_inference.py 或 camera_development.py，只能复用窄身份校验且先报告原因；需要修改 model.py、release.py、共享 tracker、其他模块、environment.yml 或共享依赖时暂停。

测试全部使用 eldercare-ai，禁止裸 python/pytest。顺序为：
1. tests/test_wandering_camera_portability.py + tests/test_wandering_camera_collection.py；
2. 受影响 release/candidate/primary/development 窄测；
3. PORTABLE + 九个 camera 文件；
4. 七个相关 CLI --help 和 blocked exit contract；
5. git archive HEAD + 精确 allowlist overlay 的 unshare -Urn 隔离回归；
6. fixed candidate CPU/eval 独立 safe-load，forward=0、network attempt=0；
7. production preflight 继续 blocked/asset_source_unavailable；
8. Markdown UTF-8/链接、git diff --check、editable、stage 和最终 Git 状态。

不要预设测试计数。完整 camera 回归可以在 synthetic tracking 上运行真实 fixed candidate/QC，但只能写自动化/synthetic 兼容性证据；production preflight、独立 safe-load 和 PORTABLE 隔离验收本身必须 video/tracker/QC/forward=0。真实 camera、产品、老人域和临床证据必须写“无”。

当前没有 owner-approved yolov8n.pt、持久 archive 或持久 source identity。不得搜索缓存、下载、猜 SHA、生成 canonical placeholder 或 READY。F1 即使全部通过，也只能写 m0cam_portable_f1_complete=true、portable_software_audit_status=passed；formal m0cam_portable_status 继续 blocked/asset_source_unavailable + source_integration_pending，C0/C1 保持现场真实值，M0-CAM-D 不启动。

必须暂停：需要 owner asset/外部路径、持久源码集成、共享或冻结文件修改、科学协议变化、camera/official/sealed 数据、删除/覆盖证据、stage/commit/push/发布。普通代码、路径、Windows/WSL、editable、timeout、测试和文档问题自主修复并继续。

最终分栏报告：
1. 现场与冲突；
2. 红队发现和旧实现失败回归；
3. 已实现修复及精确文件；
4. 自动化/synthetic 证据；
5. 隔离网络与 source identity 证据；
6. candidate safe-load；
7. 真实 camera/产品/临床证据（应为无）；
8. F1 与 formal PORTABLE 状态及全部 blocker；
9. Git 现场；
10. 下一 owner 输入。

任何“无网络”“无 forward”“fresh-only”“immutable source”声明都必须写明作用域和直接证据。不得因为测试数量多而忽略仍可构造的假 ready。
```
