# TopoWander-MPT M0 监督训练

状态：`M0 engineering loop complete; public validation target attained`

本实验使用 scratch 初始化、固定 seed `20260731`、一个共享 mask-aware TCN + semantic patch + relation-aware Transformer trunk、一个 binary head、一个 subtype head 和一个 AdamW optimizer。训练只使用 materialized development bundle 中的 1,257 条 train，选点和报告只使用 278 条 validation；WP frozen test 未被 development accessor 暴露，也未用于推理、计分、训练或选点，SmartCare official/raw 与 sealed camera 数据未被读取。

## 结果

| 指标 | 结果 |
| --- | ---: |
| best epoch / last epoch | 13 / 21 |
| WP 四分类 macro-F1 | 0.995833 |
| WP binary macro-F1 | 1.000000 |
| SmartCare binary macro-F1 | 1.000000 |
| 来源等权 binary macro-F1 | 1.000000 |
| 联合 selection score | 0.995833 |
| 总训练时间 | 687.486 s |
| 平均 epoch 时间 | 32.737 s |

WP 四分类逐类结果：direct、pacing、lapping、random 的 recall 分别为 `1.000000 / 1.000000 / 0.983333 / 1.000000`。唯一四分类错误为 1 条 lapping 被预测为 pacing，混淆矩阵为：

```text
[[60, 0,  0, 0],
 [ 0, 60, 0, 0],
 [ 0, 1, 59, 0],
 [ 0, 0,  0, 60]]
```

两项目标 macro-F1 均达到 `0.95`，因此按本次 goal 的有界规则不执行 M1，也不启动多 seed、网格搜索或 frozen test。

## 闭环证据

- best checkpoint：`checkpoints/epoch-0013/`
- last/resume checkpoint：`checkpoints/epoch-0021/`
- training history：`training_history.jsonl`
- fresh-process metrics：`fresh_reload/metrics.json`
- validation predictions：`fresh_reload/predictions.jsonl`
- confusion matrices：`fresh_reload/confusion_matrix.json`
- single-sample inference：`fresh_reload/single_sample_inference.json`
- fresh report：`fresh_reload/README.md`
- resume probe：`resume_probe.json`
- test/environment verification：`VERIFICATION.md`

fresh reload 从 best checkpoint 重建同一模型后，四分类 macro-F1、来源等权二分类 macro-F1、selection score 和 validation loss 与训练时保存值的绝对差均为 `0`。resume 探针加载 last 的完整模型、AdamW 状态、torch RNG 和 epoch，识别 run 已完成并保持 21 个 epoch checkpoint 与 best/last 指针不变。

## 成本与限制

本机 WSL PyTorch 构建与宿主 NVIDIA 驱动不兼容，CUDA 不可用；本实验在 WSL ext4 的 CPU 执行平面使用 8 个 intra-op 线程完成，没有修改共享依赖。结果仅来自单 seed 的公开轨迹 development validation。WanderingPatterns 缺少可靠 participant/session 分组，且现有 near-neighbor 审计提示泛化边界；本结果不能解释为 frozen-test、目标摄像头、真实老人、产品或临床效果。

当前最小短板是 lapping 与 pacing 的单样本混淆。它不触发本次 M1；若未来把 M0 作为冻结候选，应另立任务做多 seed 稳定性、成本复核和一次性 WP frozen-test 发布流程。

## 2026-08-12 独立审计补充

- 直接读取 `fresh_reload/predictions.jsonl` 独立复算：278 行与 278 个唯一 validation ID 完全对应；WP 四分类 macro-F1 为 `0.995833043961386`，WP/SmartCare 二分类 macro-F1 均为 `1.0`，来源等权结果为 `1.0`，与报告一致；
- materialized bundle 的 train/validation sample ID、parent ID 和精确特征均无重合；但 WanderingPatterns 缺少可靠 participant/session 分组，近邻审计存在 4,054 对 `<0.05` 跨分区形状近邻，因此当前分数可能高估真正独立人员或目标域泛化；
- preprocessing bundle 把 train/validation/test 保存于同一个 `samples.jsonl`。development 模式会为完整性解析该容器，再只暴露 train/validation；所以准确边界是“WP test 未暴露、未推理、未计分、未训练、未参与选点”，不是“其所在容器字节从未读取”；
- resume probe 证明正常完成的 epoch-21 checkpoint 能完整加载 model、AdamW、CPU torch RNG 和 epoch，并保持已有文件不变；它没有模拟进程在 checkpoint/history/pointer 更新之间异常退出，也没有实际续训下一 epoch。因此本报告只主张 checkpoint 完整加载，不主张任意崩溃点恢复已经验证。
