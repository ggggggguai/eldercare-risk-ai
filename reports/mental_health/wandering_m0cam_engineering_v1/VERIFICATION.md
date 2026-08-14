# M0-CAM-E / M0-CAM-H 验证记录

## 只读身份检查

- Git HEAD：`2981583736a74b1328e0fa52bf39028d6a3f8f7f`
- 初始 staged 集合：空
- editable project：当前仓库
- candidate manifest：12642 bytes，SHA-256 `3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7`
- model state SHA-256：`94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031`
- candidate：TopoWander-MPT，seed `20260731`，best epoch `5`，非 ensemble、非重训
- active `model.py`：54947 bytes，SHA-256 `aecf165568bddc6c2e08bc36ff6ba1e89a4b8d4446aac1a58ff7b9081d617726`，与 training/release identity 同时一致
- active `release.py`：92544 bytes，SHA-256 `761e4ea17fa88e05a8b448260e8b9ab3b547af61527b09a2006665a632b77df8`，与 release identity 一致

## 测试

所有 Python/pytest 均使用 WSL `Ubuntu-22.04` 中的 `eldercare-ai`。

### 新增窄测

```text
python -m pytest -q \
  tests/test_wandering_camera_episode.py \
  tests/test_wandering_camera_primary_inference.py

31 passed
```

除原有 fixed manifest loader、真实候选 forward、失败闭锁、episode 隔离、拒绝覆盖与 fresh CLI E2E 外，新增覆盖：两个非 synthetic 授权枚举均在 candidate loader/tracking/forward/staging 前拒绝；`[0.5,0.5]` 归为 `wandering_like`、低于 0.5 归为 direct、标签不一致 fail closed；binary manifest 契约漂移在 forward 前失败；active `model.py/release.py` 任一导入身份漂移均在 loader 前失败；source preflight 证据写入 execution/model bindings。

### 相关回归

```text
python -m pytest -q \
  tests/test_wandering_camera_adapter.py \
  tests/test_wandering_camera_qc.py \
  tests/test_wandering_camera_inference.py \
  tests/test_wandering_release.py \
  tests/test_wandering_camera_episode.py \
  tests/test_wandering_camera_primary_inference.py

86 passed, 4 subtests passed
```

以上为本次 M0-CAM-H 硬门。完整 wandering 回归未重复运行；下述线程隔离与完整 wandering 数字是 M0-CAM-E 既有历史验证，不冒充本次新鲜结果。

线程隔离修复后额外定向验证：

```text
python -m pytest -q \
  tests/test_wandering_camera_episode.py \
  tests/test_wandering_camera_primary_inference.py \
  tests/test_wandering_model.py

49 passed, 56 subtests passed
```

### 完整 wandering 回归

```text
python -m pytest -q tests/test_wandering_*.py

266 passed, 164 subtests passed
```

唯一 warning 是既有 PyTorch backward 测试触发的本机 CUDA driver 版本提示；全部 M0-CAM-E 与冻结 CPU 契约测试通过。

## fresh synthetic v4 独立核验

- output：`artifacts/synthetic_e2e_v4/`
- output 在运行前不存在，同名前缀 staging 为空；运行后用同一命令确认入口以 `FileExistsError` 拒绝覆盖
- manifest SHA-256：`e90b976fbc88bfd72b0c33488e2e5e769a69a50298c0cb30ad98d4941cd6c1a6`
- artifact descriptors：9/9 byte count + SHA-256 匹配
- ready/unavailable/inference_error：`2 / 1 / 0`
- model forwards：2
- episode candidates：2
- probability consistency：通过；`binary[0] == four[direct]`、`binary[1] == sum(four wandering subtypes)`、`four subtype == binary[1] * subtype`
- float32 层级公式独立复算最大绝对偏差 `1.0667875527392567e-08`，低于 `1e-7` 验收容限；binary/subtype/four-class 标签与冻结规则一致
- unavailable：binary/subtype/four-class 全为 null，`model_invocation_skipped=true`
- candidate `predict_logits()`：未调用
- WP prefix padding：未使用
- observed CPU threads：`8/1`
- observed batch sizes：`[1, 1]`
- model-forward-only p50/p95：`93.341036 / 137.4264038 ms`
- episode end：显式 `episode_end_sec_exclusive`
- episode policy：`development_unfrozen`
- alert decision：null
- sidecar authorization/validation scope：`synthetic_fixture / synthetic_camera_contract`
- source preflight：`passed=true`、`verified_before_candidate_loader=true`；manifest 与 active `model.py/release.py` 的 expected/observed 描述完全一致
- v1/v2/v3 历史目录分别保持 10 个文件，前后组合 SHA-256 依次为 `fe89e9e60f4594c74693f0a89aae284dc4599a4f5e7f6346c39629dfe5eb7ceb`、`3c0c71b3d09a5a5e295c33ba18e0121588c1ff1eba9fc3e8070ac06b68797335`、`c3fc63b4e287eceef5f4c622e275ef3ac9f4c7ade14685e1cc55f1b11a0490a2`

## Git 与边界

- `git diff --check`：通过；
- 本任务涉及的 Markdown 为严格 UTF-8，代码围栏成对，新增 M0-CAM-E 报告链接全部可解析；
- 全文档链接扫描另发现 3 个在 Git HEAD 中已存在的 fall-risk JSON 缺失链接：`label_validation_formal_v2.json`、`training-labels-v3-migration.json`、`training-labels-v3-validation.json`。它们不由本任务引入，且属于其他模块，本 Goal 未越界修改；
- v4 中记录的 6 个源码 SHA 与当前文件全部一致，JSON/JSONL 全部可解析；
- 没有 stash、reset、commit、push 或外部发布；
- 没有修改 `AGENTS.md`、Step7 comparison-only 实现、fall-risk tracking、共享环境、candidate、manifest 或固定模型配置；
- 保留所有用户初始脏改动；
- v1/v2/v3 中间 synthetic 输出保留，未覆盖或删除；
- 最终证据仅为 `synthetic_contract_only`。
