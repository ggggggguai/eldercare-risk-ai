# M0-CAM-RD-F verification

## 环境和入口门禁

Windows 原生 PATH 没有 Conda；验证使用项目既定的 WSL `Ubuntu-22.04` 和 `eldercare-ai`。editable 安装显示当前项目根：

```text
Name: elderly-monitoring-algorithms
Version: 0.2.0
Editable project location: <project-root>
```

开始时分支为 `feat/wandering-data-pipeline`，相对 origin ahead 9；工作树已有多项用户改动，staged 集合为空。所有既有改动和制品均保留。

## 测试驱动证据

修改生产实现前，原 RD 聚焦测试为：

```text
22 passed in 5.26s
```

加入 RD-F 回归但尚未实现新 API 后，pytest collection 按预期失败：

```text
ImportError: cannot import name 'prepare_authorized_camera_session'
ImportError: cannot import name 'evaluate_labeled_episodes'
2 errors during collection
```

最终聚焦命令：

```text
conda run --no-capture-output -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_dataset.py \
  tests/test_wandering_camera_development.py -q
35 passed in 6.74s
```

最终全部 camera 回归：

```text
conda run --no-capture-output -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_adapter.py \
  tests/test_wandering_camera_inference.py \
  tests/test_wandering_camera_qc.py \
  tests/test_wandering_camera_corruption.py \
  tests/test_wandering_camera_primary_inference.py \
  tests/test_wandering_camera_episode.py \
  tests/test_wandering_camera_dataset.py \
  tests/test_wandering_camera_development.py -q
118 passed in 69.62s
```

三个 CLI help 均在同一环境退出 0：

```text
python scripts/wandering/prepare_camera_session.py --help
python scripts/wandering/build_camera_annotations.py --help
python scripts/wandering/run_camera_development.py --help
```

`git diff --check` 退出 0；只报告工作树原有的 LF/CRLF 转换提示，没有 whitespace error。

## receipt-first 零调用证据

回归测试分别覆盖 receipt 文件缺失、expired、inactive、operation 不允许、collection receipt ID 不符、scope 不符和 sealed role。失败发生在 protected tracking/annotation/cohort、candidate loader、QC/preprocessing、model forward、staging/final 之前；断言包括：

```text
protected input read count = 0
candidate loader count = 0
forward count = 0
final exists = false
same-prefix staging count = 0
```

`prepare_camera_session.py` 与 `build_camera_annotations.py` 现在都要求 `--receipt` 和 `--collection`。`run_camera_development.py` 在 receipt 通过后才读取 evaluation policy。三个正式 CLI 都没有 `--allow-synthetic`、`--fake-runtime`、loader bypass 或同义参数。

## production-branch fixture 边界

engineering smoke 和 labeled evaluation 成功测试使用旧 synthetic tracking bytes 加结构化授权 metadata fixture，但不使用 `_test_only=True`。两条路径都经过：

```text
fresh-output preflight
→ exact development config
→ real receipt validator
→ collection + sidecar validators and scope binding
→ active source/candidate preflight
→ real tracking adapter
→ labeled path C2/C3 validators
→ manifest-bound primary loader compatibility check
→ Camera QC + preprocessing
→ deterministic fake model at candidate-loader boundary only
→ episode/evaluator
→ staging + atomic final
```

fixture 输出只写 pytest `tmp_path`，所有 window、episode、execution 和 manifest 均为 `evidence_scope=test_fixture_only`，并写 `authorized_camera_data_consumed=false`。它们没有进入本报告目录，也不构成 authorized evidence。

## C3、person-hours、matching 和 evaluator

- C3 回归覆盖未绑定、错 setup/session、重复绑定、stale binding、缺 clock alignment 和 annotation 指向未观察 tracklet；均在 candidate loader 前拒绝。
- matching 保留 source group/video/device/setup/epoch/track + participant/session/camera setup/clock domain；附件指定的 A1/A2/P1/P2 反例得到 2 对匹配。
- person-hours 回归证明两个相同相对时间但不同 session 的 1 小时区间合计为 2 person-hours；同 session/clock domain 先 union；prediction uncertain 不缩短分母。
- accepted unknown truth 被拒绝；truth-masked prediction 不进入主错误计数；matched low-confidence episode 计 explicit abstention；QC unavailable、inference error 和无预测 miss 分开；未匹配 accepted truth 不再伪装成 uncertain。
- purposeful pacing 在 four-class 中仍为 pacing、binary 中仍为 wandering-like，只进入 purpose/alert 诊断。
- matching、uncertain 和 merge policy 的 `policy_id` 必须为非空字符串；零时间重叠不合法。

## 未触碰项

- 固定 candidate manifest SHA-256：`3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7`；
- 固定 model SHA-256：`94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031`；
- primary seed/best epoch：`20260731 / 5`；
- 没有训练、M1、阈值/标签/split/指标修改；
- 没有真人 camera、WP raw、SmartCare official/raw 或 sealed camera 访问；
- 没有 M0-CAM-D、commit、push、发布或部署。

## 完成后独立 API 审计

本文件记录的 35 项聚焦测试、118 项 camera 回归、候选 SHA 和数据访问边界均继续成立。随后审计发现两个未被这些测试覆盖的库调用边界：非 fixture `_test_hooks` 尚未全面拒绝；公开逐窗预测 API 尚可由调用者指定 authorized `evidence_scope`。两项归入后续 `M0-CAM-RD-F2`；原记录不改写，后续完成证据如下。

## RD-F2 完成验证

开始时分支仍为 `feat/wandering-data-pipeline`，HEAD 为 `d3c101bc459234418832a5100c0b5c29ac518dc3`，相对 origin ahead 12；既有用户脏改动保持不变。editable 安装再次确认指向当前项目根。

先写回归、未改生产实现时，两个缺口稳定得到：

```text
17 failed
```

最小实现后，primary + development 聚焦测试为：

```text
63 passed in 14.05s
```

全部可运行 camera 回归为：

```text
139 passed in 69.78s
```

三个 RD CLI `--help` 均退出 0，且无 synthetic、fake runtime/loader、validation/evidence scope、test hook/fixture 或 bypass selector。直接 Python 安全回归另证明：

```text
non-fixture _test_hooks={}       -> zero preflight/read/loader/forward/output
non-fixture nonempty hooks       -> zero preflight/read/loader/forward/output
non-fixture candidate loader     -> zero preflight/read/loader/forward/output
public non-synthetic scopes      -> zero model forward
controller authorized scope      -> only after receipt/collection/sidecar/source/tracking/loader/QC gates
fixture evidence scope           -> test_fixture_only
```

最终 primary+dataset+development 聚焦用例共 74 项，均包含在上述最终 139 项 camera 回归中；`git diff --check` 退出 0，仅显示既有 LF/CRLF 转换提示。

RD-F2 没有读取授权 camera、WP raw、SmartCare official/raw 或 sealed 数据，没有修改 fixed candidate/model/threshold/labels/split，没有重训或启动 M0-CAM-D，也没有 stage、commit 或 push。
