# 步态 observable_instability_b02_b04：P2/P3 provisional 结果

更新时间：2026-08-10

## 状态

本报告只证明当前 train/validation 数据准备、结构化 baseline 和单 seed TCN smoke 链路可运行。协议状态为 `development_provisional`，不包含 test 姿态、test tensor、test 指标或部署候选；规则步态分支继续是运行主路径和 fallback。

当前输入绑定：

| 输入 | SHA-256 / 标识 |
|---|---|
| source split | `splitv3_e71a045eb58489f43dc5fd11` |
| dataset.npz | `61c54e44038aa1c99eb8a0cfc007261f7affb6edaa57c706f9d4362768a01640` |
| action labels v3 | `894c6d052159be5f6aad19778b3484aa49e663e29b5030465134fc906b554e5f` |
| manifest | `539274222b8a3b91249398a0415f304b73edf3dc5066150304102a7ca773503a` |
| assignments | `878a3b46afc570ca32baf37220579301b5cdb420009249318012fc79ac68f715` |
| split report | `02aaaee6870aad05a20e0a9633bd28e8c2e900336b01baf36a45b6f6d1153b80` |

双构建 `splitv3-e71a045-build-a` / `splitv3-e71a045-build-b` 的 `dataset.npz`、`metadata.json`、`samples.jsonl`、assignments 和 split sidecar 逐字节一致。共生成 1,353 个窗口：train 1,010、validation 343；test 标签只做计数，记录 1,065 条，未读取对应姿态。

姿态过滤后 validation 主任务独立动作段只有 9 个，B02/B03/B04 分别为 2/4/3，正类仅来自 1 个 source group。根据任务书的预注册停止规则，禁止完整超参搜索、概率校准、三 seed 候选冻结和任何 test 访问。

## P3 结果

validation 窗口/动作段指标如下；由于样本规模门禁失败，所有点估计只作工程烟测。

| 实验 | balanced accuracy | F1 | 正常误报（/hour） |
|---|---:|---:|---:|
| rule（exclude_quality） | 0.502 | 0.062 | 1324.0 |
| Logistic（exclude_quality） | 0.894 | 0.600 | 40.1 |
| LightGBM（exclude_quality） | 0.867 | 0.200 | 353.1 |
| EBM（exclude_quality） | 0.854 | 0.185 | 389.2 |
| rule（exclude_quality_and_scale） | 0.502 | 0.062 | 1324.0 |
| Logistic（exclude_quality_and_scale） | 0.894 | 0.600 | 40.1 |
| LightGBM（exclude_quality_and_scale） | 0.864 | 0.247 | 236.7 |
| EBM（exclude_quality_and_scale） | 0.817 | 0.222 | 240.7 |
| TCN smoke，seed 42，3 epoch | 0.858 | 0.216 | 228.7 |

TCN checkpoint 参数量 14,212，validation 动作段 balanced accuracy 为 0.858；这些数字不能证明跨来源泛化或部署适用性。TCN smoke 仅用于验证数据、训练、checkpoint、阈值和 metrics 链路，不用于模型选择。

## 复现命令

```bash
conda run -n eldercare-ai python scripts/prepare/prepare_gait_window_dataset.py \
  --output-dir data/processed/fall_risk/gait_observable_v1/splitv3-e71a045-build-a \
  --target-profile observable_instability_b02_b04

conda run -n eldercare-ai python scripts/train/train_gait_tcn.py \
  --data data/processed/fall_risk/gait_observable_v1/splitv3-e71a045-build-a/dataset.npz \
  --output-dir reports/fall_risk/gait_observable_v1/splitv3-e71a045/tcn-smoke-seed-42 \
  --epochs 3 --patience 2 --batch-size 16 --seed 42 --device cpu
```

## 未解除阻塞

- B02/B03/B04 validation 独立动作段与 source group 不足；B03 `action_type_training_tier` 无 primary，不能训练 subtype 头。
- split 仍 provisional，评估协议未冻结；未指定 test 保管人和 release 文件。
- 连续正常背景、目标老人域、拒识/延迟/内存和失败案例证据未达到正式门槛。
- 不得复用旧 split 的 gait dataset、encoder、checkpoint、阈值或指标；不得修改默认 checkpoint 或规则主路径。
