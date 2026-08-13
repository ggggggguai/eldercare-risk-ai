# M0 验证记录

所有项目测试与脚本均使用 `eldercare-ai`。正式训练在 WSL ext4 的 `/home/lenovo/work/eldercare-wandering-m0-20260812` 源码副本与 `/home/lenovo/runs/eldercare-wandering/m0_seed20260731_v1` 输出目录运行；桌面报告目录为无覆盖复制。

## 通过项

- 性能单元测试：`9 passed`。
- 最终 `eldercare-ai` editable 已恢复为当前桌面仓；import 路径为当前仓 `src/elderly_monitoring/__init__.py`，恢复后性能测试再次为 `9 passed`。
- M0 + TopoWander 前向测试（ext4）：`36 passed, 56 subtests passed`。
- 性能 + 前向 + preprocessing bundle 窄测（桌面事实源）：`43 passed, 73 subtests passed`；在 development-only bundle 改造后性能测试单独复跑为 `9 passed`。
- overfit smoke：`PASS`，16 条四类 train 样本，loss `2.309692 -> 0.712978`，binary/subtype/trunk 梯度非零且 finite，trunk 更新量非零，mask 与 checkpoint reload 最大差均为 `0`。
- M0 正式训练：21 epoch 后 validation early stopping，best 13、last 21；每 epoch checkpoint 均存在。
- fresh-process reload：`PASS`；best checkpoint 复算指标与训练保存值一致，四项绝对差均为 `0`。
- resume probe：`PASS`；加载 last 的模型、AdamW、CPU torch RNG 和 epoch，识别 run 已完成，无新 checkpoint、无覆盖。该 probe 没有模拟中途崩溃后继续下一 epoch，不能作为任意中断点恢复已经验证的证据。
- 全部 wandering 测试：`226 passed, 160 subtests passed, 482 deselected`。
- `git diff --check`：无 whitespace error；终端只报告仓库既有/当前工作树的 Windows 换行转换 warning。

## 最终全仓 pytest

命令：

```bash
conda run -n eldercare-ai python -m pytest tests -q
```

结果：`707 passed, 239 subtests passed, 1 failed`。唯一失败为明确范围外的跌倒模块测试：

```text
tests/test_fall_runtime_fingerprint.py::FallRuntimeFingerprintTest::test_repository_acceptance_config_has_hashable_fixed_inputs
FileNotFoundError: fixed input is unavailable:
data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi
```

未修改跌倒模块，也未补造或下载该固定媒体输入。

## 环境说明

WSL 的 PyTorch 为 `2.13.0+cu130`，宿主 NVIDIA driver 对应 CUDA driver version `12070`，组合不兼容，`torch.cuda.is_available()` 为 false。M0 performance config 固定 `device: cpu`、8 个 intra-op 线程；没有升级/卸载包，没有修改 `environment.yml` 或共享依赖锁。测试 warning 仅来自 PyTorch 的 CUDA 可用性探测。

## 数据消费边界

训练执行平面仅复制由 development accessor 物化的 1,535 条记录：train 1,257、validation 278。bundle manifest 的历史字段为 `test_split_opened=false`；其准确语义是 WP test 未被 accessor 暴露，也未进入 tensor、推理、指标、训练或选点。上游 preprocessing bundle 会为完整性解析同时保存 train/validation/test 的共享 `samples.jsonl`，因此不能把该字段解释成 test 所在容器字节从未读取。`smartcare_official_opened=false`、`sealed_camera_opened=false` 的零读取口径成立。正式 run、checkpoint metadata、fresh report 和 resume probe 均保持这一开发边界。
