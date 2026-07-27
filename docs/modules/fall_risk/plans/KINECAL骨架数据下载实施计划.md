# KINECAL 骨架数据下载实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 下载 KINECAL v1.0.3 的 57 个官方风险组参与者在三类目标动作中的逐帧骨架，并提供可复现下载、校验和训练前使用说明。

**Architecture:** 下载器使用 Python 标准库解析官方 `register.csv` 和目录索引，筛选 `NF/FHs/FHm` 参与者及三个动作，再以有限并发下载 `skel/*.txt`；环境安装 `httpx` 时复用 HTTP 连接，否则回退到标准库传输。原始数据保存在 Git 忽略目录，下载器生成本地 SHA-256 manifest 和汇总文件，README 说明骨架格式、分组语义和 TCN 使用边界。

**Tech Stack:** Python 3.11 标准库、pytest、PhysioNet HTTPS 目录索引、JSONL/JSON、项目 `eldercare-ai` conda 环境。

---

### Task 1: 下载器解析与选择规则

**Files:**
- Create: `tests/test_kinecal_download.py`
- Create: `src/elderly_monitoring/modules/fall_risk/kinecal.py`
- Create: `scripts/annotation/download_kinecal.py`

- [ ] **Step 1: 编写失败测试**

测试提供内存中的 `register.csv` 和目录 HTML，断言：

```python
assert [row["part_id"] for row in select_risk_group_participants(rows)] == ["SPPB40", "SPPB62"]
assert parse_directory_files(html, suffix=".txt") == [("100.txt", 1673)]
assert participant_number("SPPB700") == "700"
```

- [ ] **Step 2: 验证测试因模块不存在而失败**

Run:

```bash
conda run -n eldercare-ai python -m pytest tests/test_kinecal_download.py -q
```

Expected: collection fails because `elderly_monitoring.modules.fall_risk.kinecal` does not exist.

- [ ] **Step 3: 实现最小解析函数**

实现：

```python
def read_register_csv(text: str) -> list[dict[str, str]]: ...
def select_risk_group_participants(rows: Iterable[Mapping[str, str]]) -> list[dict[str, str]]: ...
def participant_number(part_id: str) -> str: ...
def parse_directory_files(html: str, *, suffix: str) -> list[RemoteFile]: ...
```

- [ ] **Step 4: 运行窄范围测试并确认通过**

Run:

```bash
conda run -n eldercare-ai python -m pytest tests/test_kinecal_download.py -q
```

Expected: parsing tests pass.

### Task 2: 下载、断点续传与摘要

**Files:**
- Modify: `tests/test_kinecal_download.py`
- Modify: `src/elderly_monitoring/modules/fall_risk/kinecal.py`
- Modify: `scripts/annotation/download_kinecal.py`

- [ ] **Step 1: 为下载行为编写失败测试**

使用临时目录和本地 HTTP server，验证：

```python
assert download_file(url, output, expected_size=4).status == "downloaded"
assert download_file(url, output, expected_size=4).status == "skipped"
assert sha256_file(output) == hashlib.sha256(b"pose").hexdigest()
```

并验证汇总包含 `33/15/9` 分组、年龄不一致记录、文件数和字节数。

- [ ] **Step 2: 运行测试并确认因下载函数不存在而失败**

Run the same narrow pytest command. Expected: missing download/summary APIs.

- [ ] **Step 3: 实现有限并发下载器和 CLI**

CLI：

```bash
conda run -n eldercare-ai python scripts/annotation/download_kinecal.py \
  --output-dir data/external/kinecal/raw \
  --workers 8 \
  --retries 3
```

实现要求：

- 只选择 `NF/FHs/FHm`。
- 只枚举 `3m-walk-Front-View`、`Get-Up-And-Go-Front-View`、`STS-5`。
- 只下载 `skel/*.txt`、`register.csv` 和 `LICENSE.txt`。
- 已存在且大小匹配时跳过。
- 使用临时 `.part` 文件后原子替换。
- 输出 `download_manifest.jsonl` 和 `download_summary.json`。
- 严格模式下有缺失目录、空目录或下载失败时返回非零状态。

- [ ] **Step 4: 运行窄范围测试并确认通过**

Run the same narrow pytest command. Expected: all KINECAL downloader tests pass.

### Task 3: 数据集 README 与仓库入口

**Files:**
- Create: `data/external/kinecal/README.md`
- Modify: `.gitignore`
- Modify: `scripts/annotation/README.md`
- Modify: `docs/modules/fall_risk/data/数据集标注规范.md`
- Modify: `docs/README.md`

- [ ] **Step 1: 更新忽略规则**

保留 `data/external/kinecal/README.md`，继续忽略 `raw/`：

```gitignore
!data/external/kinecal/
data/external/kinecal/*
!data/external/kinecal/README.md
```

- [ ] **Step 2: 编写使用说明**

README 必须包含来源、版本、许可证、下载命令、恢复下载、目录结构、57 人分组、4 条年龄不一致、骨架文件格式、TCN 输入转换和按人划分要求。

- [ ] **Step 3: 更新现行文档入口**

在数据标注规范中增加 KINECAL 的标签语义和禁止事项，并在脚本 README 中列出下载入口。

- [ ] **Step 4: 检查 Markdown 路径和格式**

Run:

```bash
git diff --check
test -f data/external/kinecal/README.md
test -f scripts/annotation/download_kinecal.py
```

Expected: exit 0.

### Task 4: 正式下载与验证

**Files:**
- Generate, ignored: `data/external/kinecal/raw/**`

- [ ] **Step 1: 确认 editable 安装**

Run:

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
```

Expected: editable project location points to the current repository.

- [ ] **Step 2: 执行正式下载**

Run the downloader CLI from Task 2. Keep the process running until it exits.

- [ ] **Step 3: 验证本地数据**

Run:

```bash
conda run -n eldercare-ai python scripts/annotation/download_kinecal.py \
  --output-dir data/external/kinecal/raw \
  --verify-only
```

Expected: 57 participants, groups `33/15/9`, 154 available recordings and 17 source-missing recordings, no `.bin`, no failed available files, and all manifest hashes match.

- [ ] **Step 4: 运行相关和完整测试**

Run:

```bash
conda run -n eldercare-ai python -m pytest tests/test_kinecal_download.py -q
conda run -n eldercare-ai python -m pytest -q
```

Expected: all tests pass.
