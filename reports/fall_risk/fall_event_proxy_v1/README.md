# 跌倒识别 TCN 三 seed 集成实验

日期：2026-08-09

## 结论

当前效果最好的跌倒识别模型是 seed 42/43/44 三个轻量 TCN 的平均概率集成。validation 选择阈值 `0.42490479350090027` 后，699 个候选动作窗口上的 Precision 为 `0.9779`、Recall 为 `0.9699`、F1 为 `0.9739`、balanced accuracy 为 `0.9730`。三模型均已训练完成，默认推理入口已切换到该集成配置。

这组指标衡量 4 秒姿态候选窗口是否包含跌倒动作，不是连续视频逐帧定位指标。阈值也在同一 validation 上选择，因此结果应解释为当前开发集效果，不解释为独立测试集效果。

## 数据与模型

- split：`splitv3_e71a045eb58489f43dc5fd11`。
- 数据目录：`data/processed/fall_risk/fall_event_proxy_v1/splitv3-e71a045/`。
- 输入：8 FPS x 4 秒，14 个姿态点，7 个通道，张量形状 `[2533,32,14,7]`。
- train：1,834 个窗口，正类 1,102、负类 732。
- validation：699 个窗口，正类 365、负类 334。
- dataset SHA-256：`91ab4f332f0b63267b95ba56b911dedaa9fc5a8f6b7886ed855a6ff633077b39`。
- 模型：4 层 dilation TCN，单模型 8,134 个参数；训练时启用左右镜像增强，二分类损失权重为 1，方向辅助损失为 0。
- 干净目录重建与当前 `dataset.npz`、`metadata.json`、`samples.jsonl` 逐字节一致。

## 单模型结果

下表均使用各 checkpoint 内的 `0.5` 阈值：

| seed | Precision | Recall | F1 | Balanced accuracy | PR-AUC | best epoch | checkpoint SHA-256 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.9745 | 0.9425 | 0.9582 | 0.9578 | 0.9906 | 28 | `fcb7d1390bc6ceff4cba5448d6eedab261ac6fad1249961c3780b8d64ccf6d2c` |
| 43 | 0.9746 | 0.9479 | 0.9611 | 0.9605 | 0.9947 | 34 | `1af35ad36c2834f4ba40185c4bcc2769089739c19c809d052f81386b08f89ffa` |
| 44 | 0.9694 | 0.9562 | 0.9628 | 0.9616 | 0.9877 | 23 | `1c5659cc0b457c322f3c8e9c62bc310cc5306a2683468bfcbccb7bf325381578` |

best single 为 seed 44。三份 `independent-validation.json` 均与训练报告逐项一致。

## 集成结果

集成方式为三个模型的跌倒概率算术平均。

| 配置 | Precision | Recall | F1 | Balanced accuracy | PR-AUC | ROC-AUC | 混淆矩阵 `[[TN,FP],[FN,TP]]` |
|---|---:|---:|---:|---:|---:|---:|---|
| 三 seed，阈值 0.5 | 0.9776 | 0.9562 | 0.9668 | 0.9661 | 0.9934 | 0.9936 | `[[326,8],[16,349]]` |
| 三 seed，阈值 0.4249047935 | 0.9779 | 0.9699 | **0.9739** | **0.9730** | 0.9934 | 0.9936 | `[[326,8],[11,354]]` |

机器复算产物：`reports/fall_risk/fall_event_proxy_v1/ensemble-validation.json`，SHA-256 为 `96f8ab4373910b99ea2bcb21ba6eb65e0c864b0567468da6a4c663c8520ce431`，集成签名为 `ensemble:588267bf1a6a208388c033f48ba35a43522404a09a5be44a35b121f7c3494be0`。

分来源结果：

| 来源 | 样本数 | Precision | Recall | F1 | Balanced accuracy | 错误 |
|---|---:|---:|---:|---:|---:|---|
| Fall Detection 2017 | 95 | 0.8987 | 1.0000 | 0.9467 | 0.8333 | 8 个误报 |
| LE2I | 18 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| NTU RGB+D | 586 | 1.0000 | 0.9617 | 0.9805 | 0.9808 | 11 个漏报 |

剩余 19 个错误高度集中：8 个误报全部是 Fall Detection 2017 的 `A03` 受控坐下，11 个漏报全部是 NTU 的 `D01/D02` 前向或侧向跌倒。这两类样本是下一轮提升识别效果时最值得定向补强的数据。

## 消融结果

seed 44 额外训练了方向辅助损失权重 `0.2` 的多任务版本。其跌倒二分类 F1 为 `0.9615`、balanced accuracy 为 `0.9600`，低于纯二分类 seed 44；方向 macro-F1 也只有 `0.4079`。因此当前不使用方向辅助头，也不继续训练其余多任务 seed。

## 真实视频烟测

输入 `le2i_home_01_video_1.jsonl` 的官方跌倒开始时间约为 `5.958 s`。默认集成推理共得到 15 个有效滑窗，其中 8 个触发跌倒：

- 首个触发窗口：`[3.5, 7.5] s`，分数 `0.7760`，窗口覆盖真实跌倒开始点。
- 最高分窗口：`[6.9582, 10.9582] s`，分数 `0.9938`。
- 输出 SHA-256：`b217fb4f1055bb1f594d5225f3a5e23bebd394f82ed42d05418e48c17f013e20`。

该烟测证明默认推理确实加载了新集成并能在一个真实跌倒视频上触发；它不是多视频连续事件召回率或误报率统计。

## 复现命令

训练三个 seed：

```bash
conda run -n eldercare-ai python scripts/train/train_fall_event_tcn.py \
  --data data/processed/fall_risk/fall_event_proxy_v1/splitv3-e71a045/dataset.npz \
  --metadata data/processed/fall_risk/fall_event_proxy_v1/splitv3-e71a045/metadata.json \
  --profile pilot --seed 42 --device cpu --subtype-loss-weight 0 \
  --output-dir reports/fall_risk/fall_event_proxy_v1/development-splitv3-e71a045-seed42 \
  --allow-provisional
```

seed 43/44 使用同一命令，只替换 `--seed` 和 `--output-dir`。

独立复算三模型集成：

```bash
conda run -n eldercare-ai python scripts/evaluate/evaluate_fall_event_tcn.py \
  --data data/processed/fall_risk/fall_event_proxy_v1/splitv3-e71a045/dataset.npz \
  --metadata data/processed/fall_risk/fall_event_proxy_v1/splitv3-e71a045/metadata.json \
  --checkpoint reports/fall_risk/fall_event_proxy_v1/development-splitv3-e71a045-seed42/best_model.pt \
  --checkpoint reports/fall_risk/fall_event_proxy_v1/development-splitv3-e71a045-seed43/best_model.pt \
  --checkpoint reports/fall_risk/fall_event_proxy_v1/development-splitv3-e71a045-seed44/best_model.pt \
  --threshold 0.42490479350090027 \
  --output reports/fall_risk/fall_event_proxy_v1/ensemble-validation.json \
  --device cpu
```

默认推理已经内置上述三个 checkpoint 和阈值：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_event_model.py \
  --input data/processed/fall_risk/pose_quality_y8n_v1/cleaned/le2i_home_01_video_1.jsonl \
  --output /tmp/fall_event_predictions.jsonl \
  --device cpu
```
