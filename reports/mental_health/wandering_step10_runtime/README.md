# 徘徊步骤 10 runtime 前置治理

日期：2026-08-08

状态：`wandering_step10_runtime_candidate_verified_git_checkpoint_pending`

## 候选事实源

下面的 Linux/x86_64 runtime 候选已经完成工件级重建验证，但当前文件和本报告尚未进入 Git，HEAD 仍为 `745ed6c28958b004f23ccd3758054eef4820b692`。因此本节只记录候选 descriptor，不把它称为已冻结的 `verified_snapshot_v1`，也不据此进入步骤 10 红测：

```text
path: configs/runtime/wandering_step10_runtime_v1.yml
byte_count: 11426
sha256: 034d603ae722e1723189e03fbab05a3e8c110809d0f49011ef3e9ece8e2f1cda
eol: lf
```

该 YAML 只面向 Linux/x86_64，并把环境分成两个互不冒充的来源集合：

- 27 个 Conda 工件全部使用具体 conda-forge 工件 URL 和 MD5，不再把 Python/pip 以外的基础库交给求解器动态选择；集合包含 Python、pip、OpenSSL、libgcc、libstdcxx、`packaging`、`setuptools`、`wheel` 等。
- 30 个 pip 工件全部使用具体 PyPI wheel URL 和 SHA-256，并启用 `--require-hashes` 与 binary-only；集合覆盖 Torch、NumPy、PyYAML 及 Torch 的 pip 传递依赖。
- `setuptools==83.0.0`、`wheel==0.47.0`、`packaging==26.3` 只来自 Conda；YAML 不再声明或声称安装 PyPI setuptools wheel。

发行包 pin 为 `torch==2.13.0`，目标 wheel 的 `torch.__version__` 为 `2.13.0+cu130`。

## 第三个全新环境与严格来源审计

候选最终字节使用事先不存在的第三个环境名重建：

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

锁文件自身还通过 exact 顶层键/顺序、27 个 Conda 直链唯一性、30 个 pip 分发名与直链唯一性、MD5/SHA-256 格式和 pip 集合不含 setuptools 的静态断言；不能用重复行掩盖 set-equality。

`python -m pip check` 输出 `No broken requirements found.`。在该环境的全新进程中，任何 tensor/model/并行工作前设置并读回 deterministic/thread 状态，结果为：

```json
{"runtime":{"implementation":"cpython","numpy_version":"2.4.6","platform_machine":"x86_64","platform_system":"Linux","python_version":"3.11.15","pyyaml_version":"6.0.3","torch_version":"2.13.0+cu130"},"torch_state":{"deterministic_algorithms":true,"deterministic_warn_only":false,"inter_op_threads":1,"intra_op_threads":1}}
```

当前工作树的步骤 9 模型窄测为 `27 passed, 1 warning, 56 subtests passed`，全部徘徊测试为 `217 passed, 1 warning, 160 subtests passed`；唯一 warning 是既有 CUDA 驱动探测，测试与步骤 10 协议仍使用 CPU。runtime 修正没有修改 `model.py`，其冻结 SHA-256 仍为 `11fd7732f49d7392f0dab8edaa3023ebb1ba32355b24fc2157349b7fff80b331`。

## 已撤回候选

旧 descriptor `8506 bytes / 52a76abc3dff2a4e6f08e25c0a8ce8ad459bf7806085fc2d7fd56a89832a464d` 只精确锁定 Python、pip 两个 Conda 工件，其余 25 个 Conda 工件仍由求解器选择；报告还错误声称 31 个 wheel 均由 URL/SHA 安装，而现场 setuptools 实际来自 Conda。该候选已撤回，不得写入步骤 10 production config、artifact 或安全 loader。

更早的格式候选还曾把未加引号的 `--only-binary=:all:` 解析成 YAML mapping；该残缺环境已删除，也不构成验证证据。

## 边界与待关闭门禁

本任务没有读取项目数据、创建步骤 10 红测/生产源码/production YAML、打开 optimizer、训练模型或生成 checkpoint。`2.13.0+cu130` 是 CPU 协议使用的 wheel runtime 标识，不表示 CUDA 可用或步骤 10 使用 GPU。

当前技术验证已经证明候选字节可按精确工件集合重建，但正式 runtime 门禁仍未关闭。还必须在负责人明确授权后把 runtime YAML、报告及同步状态文件纳入一个 Git 检查点，再从该检查点创建全新 checkout，复核相对路径、`11426` bytes、SHA-256、纯 LF、27/30 来源集合、`pip check` 和 runtime/thread/deterministic 探针。完成该检查点复验前，不得开始第 10.9 节红测；完成后才可把状态改为 `wandering_step10_runtime_fact_source_verified`。
