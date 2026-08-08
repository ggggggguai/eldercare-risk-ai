# 徘徊步骤 10 runtime 前置治理

日期：2026-08-08

状态：`wandering_step10_runtime_fact_source_verified`

## 冻结事实源

步骤 10 把下面文件视为 opaque `verified_snapshot_v1` regular-file；训练、第二次重建和公共安全 loader 只按外部 descriptor 对同一份原始字节核对，不从其内容反推现场版本：

```text
path: configs/runtime/wandering_step10_runtime_v1.yml
byte_count: 11426
sha256: 034d603ae722e1723189e03fbab05a3e8c110809d0f49011ef3e9ece8e2f1cda
eol: lf
candidate_checkpoint_commit: 03079a5257f69069218d07a73faf0ce797185897
```

该 YAML 只面向 Linux/x86_64，并把环境分成两个互不冒充的来源集合：

- 27 个 Conda 工件全部使用具体 conda-forge 工件 URL 和 MD5，不再把 Python/pip 以外的基础库交给求解器动态选择；集合包含 Python、pip、OpenSSL、libgcc、libstdcxx、`packaging`、`setuptools`、`wheel` 等。
- 30 个 pip 工件全部使用具体 PyPI wheel URL 和 SHA-256，并启用 `--require-hashes` 与 binary-only；集合覆盖 Torch、NumPy、PyYAML 及 Torch 的 pip 传递依赖。
- `setuptools==83.0.0`、`wheel==0.47.0`、`packaging==26.3` 只来自 Conda；YAML 不声明或声称安装 PyPI setuptools wheel。

发行包 pin 为 `torch==2.13.0`，目标 wheel 的 `torch.__version__` 为 `2.13.0+cu130`。

## 候选阶段工件级重建

候选最终字节曾使用事先不存在的第三个环境名重建：

```text
/home/lenovo/miniconda3/bin/conda env create \
  -n eldercare-ai-wandering-runtime-v1-rebuild3-20260808 \
  -f /mnt/c/Users/lenovo/Desktop/心理算法/configs/runtime/wandering_step10_runtime_v1.yml
```

重建后没有只依据版本探针判定成功，而是逐项比较锁文件与安装现场：Conda 侧读取全部 `conda-meta/*.json` 的工件 URL/MD5；pip 侧读取全部 distribution 的 `direct_url.json`、`INSTALLER` 和 URL/SHA。两个集合均要求双向完全相等，结果为：

```text
conda_artifacts_exact=27
pip_wheels_exact=30
setuptools_source=conda
wheel_source=conda
packaging_source=conda
```

锁文件自身还通过 exact 顶层键/顺序、27 个 Conda 直链唯一性、30 个 pip 分发名与直链唯一性、MD5/SHA-256 格式和 pip 集合不含 setuptools 的静态断言；不能用重复行掩盖 set-equality。`python -m pip check` 输出 `No broken requirements found.`。fresh-process runtime/thread/deterministic 探针与下节结果相同。

候选工作树的步骤 9 模型窄测为 `27 passed, 1 warning, 56 subtests passed`，全部徘徊测试为 `217 passed, 1 warning, 160 subtests passed`；唯一 warning 是既有 CUDA 驱动探测，测试与步骤 10 协议仍使用 CPU。runtime 修正没有修改 `model.py`。

## 候选检查点与 fresh-checkout 复验

负责人明确要求并授权两提交协议：第一提交继续保持“候选、待 fresh-checkout 复验”，不得 amend；全部门禁通过后再由 docs-only 提交晋升状态。候选检查点为：

```text
commit: 03079a5257f69069218d07a73faf0ce797185897
subject: chore(mental-health): checkpoint wandering step10 runtime candidate
fresh_worktree: C:\Users\lenovo\AppData\Local\Temp\wandering-step10-runtime-fresh-03079a5-20260808
fresh_environment: eldercare-ai-wandering-runtime-v1-freshcheckout-03079a5-20260808
```

worktree 目标在创建前不存在；它以 detached HEAD 指向候选检查点且 `git status --short` 为空。只从该 checkout 读取 runtime YAML，逐项结果为：

```text
runtime_path=configs/runtime/wandering_step10_runtime_v1.yml
runtime_bytes=11426
runtime_sha256=034d603ae722e1723189e03fbab05a3e8c110809d0f49011ef3e9ece8e2f1cda
runtime_cr=0
runtime_lf=65
git_check_attr=text:set,eol:lf
git_ls_files_eol=i/lf,w/lf,attr/text_eol=lf
model_sha256=11fd7732f49d7392f0dab8edaa3023ebb1ba32355b24fc2157349b7fff80b331
```

fresh environment 名在创建前不存在。它从上述 fresh-checkout YAML 完整重建后，再次通过锁文件与安装现场的双向来源集合审计：

```text
conda_artifacts_exact=27
pip_wheels_exact=30
setuptools_source=conda
wheel_source=conda
packaging_source=conda
pip_check=No broken requirements found.
```

来源审计首次调用曾因只读检查器误解 `dict.setdefault()` 首次插入的返回值而在集合比较前失败；这不是工件漂移，没有修改候选 commit、runtime 字节或 expected。修正检查器表达式后，在同一 fresh environment 上完成了上面的双向完全相等审计。

最后在该环境的全新进程中、任何 tensor/model/并行工作前设置并读回 deterministic/thread 状态，结果为：

```json
{"runtime":{"implementation":"cpython","numpy_version":"2.4.6","platform_machine":"x86_64","platform_system":"Linux","python_version":"3.11.15","pyyaml_version":"6.0.3","torch_version":"2.13.0+cu130"},"torch_state":{"deterministic_algorithms":true,"deterministic_warn_only":false,"inter_op_threads":1,"intra_op_threads":1}}
```

上述门禁全部通过，且没有重算 expected、amend 候选提交或修改 `model.py`，因此 runtime 状态晋升为 `wandering_step10_runtime_fact_source_verified`。

## 已撤回候选

旧 descriptor `8506 bytes / 52a76abc3dff2a4e6f08e25c0a8ce8ad459bf7806085fc2d7fd56a89832a464d` 只精确锁定 Python、pip 两个 Conda 工件，其余 25 个 Conda 工件仍由求解器选择；旧报告还错误声称 31 个 wheel 均由 URL/SHA 安装，而现场 setuptools 实际来自 Conda。该候选已撤回，不得写入步骤 10 production config、artifact 或安全 loader。

更早的格式候选还曾把未加引号的 `--only-binary=:all:` 解析成 YAML mapping；该残缺环境已删除，也不构成验证证据。

## 边界与下一步

本任务没有读取项目数据、创建步骤 10 红测/生产源码/production YAML、打开 optimizer、训练模型或生成 checkpoint。`2.13.0+cu130` 是 CPU 协议使用的 wheel runtime 标识，不表示 CUDA 可用或步骤 10 使用 GPU。

步骤 9+9a 检查点、最终 Step 9 绑定和 runtime fact source 三项前置门禁现已闭合；runtime 候选本身继续 verified。后续 source-closure 审计又发现 22 个项目 source 中 12 个辅助源码在当前工作树为 CRLF、Git blob 为 LF，因此下一项不是红测，而是按技术方案第 10.0/10.9 节为这 12 个 exact path 形成 LF 属性、Git 检查点和 fresh-checkout SHA 证据；完成后在本报告追加独立 post-runtime source-closure 小节，再进入红测。production config 外部 SHA 形成前仍不得读取正式训练数据、创建正式数据路径的 production optimizer 或生成 checkpoint；未来合成单测中的 AdamW 不受此禁令。
