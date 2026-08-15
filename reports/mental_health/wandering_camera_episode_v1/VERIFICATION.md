# M0-CAM-EP1A verification

日期：2026-08-15

环境：所有 Python/pytest 命令均使用 `eldercare-ai`；editable project location 为 `/mnt/c/Users/lenovo/Desktop/心理算法`。

## 红测

实现前运行：

```bash
conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_episode_inference.py \
  tests/test_wandering_camera_episode_import.py
```

结果：collection 出现 2 个 import error，原因分别为 `camera_episode_inference` 和 `camera_episode_import` 尚不存在。这是预期的功能缺口复现。

## 绿测

聚焦：

```bash
conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_episode_inference.py \
  tests/test_wandering_camera_episode_import.py
```

结果：`14 passed in 11.54s`。

EP1A 加冻结 corruption trust root：

```bash
conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_episode_import.py \
  tests/test_wandering_camera_episode_inference.py \
  tests/test_wandering_camera_corruption.py
```

结果：`26 passed in 24.99s`。EP1A 直接复用既有 QC 原语，最终 `camera_qc.py` 与 Git 基线字节一致，旧 trust-root SHA 不需要重锚。

legacy/primary 相邻回归：

```bash
conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_qc.py \
  tests/test_wandering_camera_primary_inference.py \
  tests/test_wandering_camera_episode.py \
  tests/test_wandering_camera_inference.py \
  tests/test_wandering_camera_adapter.py
```

结果：`105 passed in 47.98s`。

authorized development/dataset：

```bash
conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_development.py \
  tests/test_wandering_camera_dataset.py
```

结果：`80 passed in 8.60s`。

共享 preprocessing/model：

```bash
conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_preprocessing.py \
  tests/test_wandering_model.py
```

结果：`43 passed, 73 subtests passed in 28.43s`。唯一 warning 是当前 PyTorch 探测到本机 CUDA driver 版本不匹配；本轮固定候选和测试均使用 CPU，测试通过。

全部 camera 相邻回归：

```bash
conda run -n eldercare-ai python -m pytest -q tests/test_wandering_camera_*.py
```

结果：`525 passed in 135.14s`。唯一 warning 来自 PORTABLE 的 duplicate ZIP entry 拒绝测试；无失败。

## 真实 development 回归

完整命令见 [README.md](README.md)。

- D01：12 个 whole-clip 正式入口，`12 ready / 12 direct`，与旧诊断概率最大差 0；
- S00 importer：3 个 CVAT episode，boundary/truth 分离；
- S00 inference：`3 ready / 3 direct`，边界为 0–18.60、18.60–39.7333、39.7333–44.7333 秒，与旧 XML 诊断概率最大差 0；
- S00 legacy：0–40 秒仍为 lapping，`P(lapping)=0.997431`，保存在旧 long-context diagnostic，不混入 episode 输出。

原视频未复制进 Git；最终源码 formal EP1A 输出位于 Git 忽略的 `tmp/m0cam_ep1a_formal_20260815_v2/`。该目录与前一轮 `v1` 证据逐文件比较无差异（`diff -rq` 退出 0）。

## 证据解释

`authorized_development_smoke` 只说明本机授权样片、给定 boundary 后的 episode QC/preprocessing/frozen forward 跑通。没有 automatic boundary、跨人/跨 setup camera performance、自然负例时长、概率校准、风险或告警证据。

## 2026-08-15 独立完成复核

在不修改 EP1A 代码的情况下再次确认：

- editable 安装指向当前仓库；EP1A 聚焦复跑为 `14 passed`，EP1A 加 corruption trust-root 独立复跑为 `26 passed`；
- Git 外 v2 制品实际存在，聚合仍为 D01 `12/12 ready/direct`、S00 `3/3 ready/direct`，legacy 0–40 秒仍为 `lapping, P=0.99743098`；
- boundary/truth 物理分文件，推理只加载 boundary；旧 `camera_qc.py` 与 HEAD 无差异；
- 未发现阻塞 completed 状态的 P0，因此 `m0cam_ep1a_status=completed` 保持不变，范围仍仅限 oracle-boundary engineering scope。

三个非阻塞 P1 转入 EP1B：为两个 bundle builder 补真实端到端/non-overwrite 回归；验证低置信度边缘 observation 不能虚假满足 raw-detection/boundary-gap QC；在批量 evaluator 中正确区分 shape eligibility 与 purpose/alert eligibility。EP1A 新文件当前仍为 untracked，说明 HEAD 尚未包含该交付；这属于后续版本交接事项，不需要为此扩展哈希审计。
