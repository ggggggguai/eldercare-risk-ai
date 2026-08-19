# M0-CAM-MVP-1 verification

> 本文件保留首次实现验证，并追加 F1 exact-schema 关闭证据。当前接受状态见 [MVP1_REAUDIT_20260814.md](MVP1_REAUDIT_20260814.md)和任务表；所有结果仍只属于自动化与 synthetic contract。

日期：2026-08-14

## 环境与入口

所有 Python 和 pytest 均使用 WSL Ubuntu-22.04 的 `eldercare-ai`：

```bash
/home/lenovo/miniconda3/bin/conda run -n eldercare-ai python ...
```

`python -m pip show elderly-monitoring-algorithms` 返回 editable project location 为当前桌面仓的 WSL 挂载路径。入口审计记录 branch `feat/wandering-data-pipeline`、HEAD `bfc3329ad14d145e6b4c3fb836a0a525f30e970b`，stage 为空；既有 dirty/untracked PORTABLE、C01 与文档改动均保留，未 stash/reset/checkout/clean。

## TDD 记录

实现前运行：

```bash
conda run -n eldercare-ai python -m pytest tests/test_wandering_camera_product.py -q
```

结果按预期在收集阶段失败：`ImportError: cannot import name 'camera_product'`。新增生产模块后第一次运行有 9 个失败，原因是测试夹具的 `normalized_tracking_sha256` 没有绑定夹具 bytes；修正夹具 hash 后，没有降低生产 validator 或断言，聚焦测试转绿。

F1 在修改生产 validator 前，对合法 primary descriptor-aware 重算 canonical bytes、size 与 SHA 后执行 missing/extra、嵌套 probability/summary、`data_access`、非空 decision field 及 rogue product 顶层攻击，旧实现结果为 `42 failed, 17 passed in 18.42s`。生产修复后相同套件全部转绿；所有失败构建均无 final 或本次 staging residue。

## 验证结果

### 产品聚焦

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_product.py -q
```

首次完整绿灯结果：`17 passed in 14.80s`。补强 episode contributor/source-preflight 一致性校验后的首次实现最终复跑为 `17 passed in 15.00s`。F1 exact-schema 最终复跑为 `59 passed in 21.57s`。

### camera 相邻回归

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_product.py \
  tests/test_wandering_camera_primary_inference.py \
  tests/test_wandering_camera_episode.py \
  tests/test_wandering_camera_qc.py \
  tests/test_wandering_camera_inference.py -q
```

首次结果为 `90 passed in 55.57s`；首次实现补强后的复跑为 `90 passed in 67.98s`。F1 最终相邻回归为 `132 passed in 82.27s`。

### 日级与风险主链隔离回归

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_mental_health_daily_aggregation.py \
  tests/test_mental_health_baseline.py \
  tests/test_mental_health_pipeline.py \
  tests/test_mental_health_cli.py -q
```

首次结果：`60 passed, 16 subtests passed in 5.51s`。F1 最终隔离回归为 `60 passed, 16 subtests passed in 7.90s`。

### CLI help

```bash
conda run -n eldercare-ai python \
  scripts/wandering/run_wandering_camera_product.py --help
```

结果：exit 0。公开参数只有：

```text
--tracking-jsonl
--media-sidecar
--episode-merge-gap-seconds
--output
```

### Git 外 fresh synthetic E2E

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_product.py::test_cli_help_parameter_errors_and_fixed_candidate_synthetic_e2e \
  -q --basetemp /tmp/wandering_mvp1_e2e_20260814_a1
```

首次结果：`1 passed in 14.26s`。F1 使用 fresh `--basetemp /tmp/m0cam_mvp1_f1_e2e_20260814_01` 再次运行，结果为 `1 passed in 18.83s`。输出保存在 Git 外 temporary test root，包含完整 10-file primary bundle、`wandering_evidence.json` 和产品 manifest。输入仅为 synthetic tracking/sidecar；fixed candidate 在 ready window 上 forward 一次，没有视频、detector、tracker 或网络入口。

## 契约覆盖

自动化回归覆盖：

1. all-ready、mixed degraded、error-only、unavailable-only 和零窗口状态；
2. failure window 不进入 episode/type/duration；
3. source/setup/epoch scope 漂移 fail closed；
4. `track_id` 不进入 `person_id`；
5. risk/alert/diagnosis/`AlgorithmEvent` 固定为空；
6. primary artifact 缺失、hash、schema 和 count drift 无 product final；
7. fixed primary bundle 的 canonical evidence bytes 可重复；
8. final 已存在拒绝覆盖，可捕获失败清理本次 sibling staging 且不删除无关 staging；
9. existing primary builder 只调用一次；不调用 RF/TCN comparison-only、detector 或 tracker；
10. CLI help、参数错误和 fixed-candidate synthetic E2E。
11. 每类 primary artifact 顶层 exact field set、detector/tracker、active-source preflight、runtime/latency/environment 嵌套 field set；
12. probability/summary exact keys、固定 class order、有限归一化、label/threshold 一致性，以及 episode aggregator canonical bytes 重算；
13. fixed synthetic-only `data_access`、产品 evidence/manifest exact fields、final 三项顶层集合和 primary 内非空 risk/diagnosis/action/`AlgorithmEvent` 拒绝。

## 边界

这些检查是软件结构和 synthetic contract 证据，不是 camera accuracy/F1/FAR、真实人/老人域、产品延迟、临床有效性或业务告警证据。PORTABLE/C0/C1/M0-CAM-D 状态没有因本次测试改变。

## 最终结构检查

- `git diff --check`：exit 0；只有当前 Windows worktree 的 LF→CRLF 提示，没有 whitespace error；
- MVP-1 修改文件均可用 strict UTF-8 解码；新增/更新的 MVP-1 本地链接全部存在；
- 仓库级旧链接扫描仍会报告三个 HEAD 中已经存在、当前机器缺失的 fall-risk JSON 目标：`label_validation_formal_v2.json`、`training-labels-v3-migration.json`、`training-labels-v3-validation.json`。本任务未修改这些链接或越界生成制品；
- fixed primary/QC/episode/inference、PORTABLE、mental-health pipeline/baseline 和 `environment.yml` 的 diff allowlist 检查为空；
- 最终 branch `feat/wandering-data-pipeline`、HEAD `bfc3329ad14d145e6b4c3fb836a0a525f30e970b`，stage 为空；没有 commit 或 push。
