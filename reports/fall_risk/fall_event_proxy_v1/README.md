# 跌倒动作候选 TCN provisional pilot

日期：2026-08-03

## 结论

本轮按照现有 v3 primary 动作标签完成了 `fall_event_candidate_clip_tcn_v1` 的 train/validation 闭环。模型判断预先截取的 4 秒姿态窗口是否包含跌倒动作，只作为 `provisional_shadow` 候选模型；它不是连续视频跌倒定位器，也不替换实时 `fall_state` 规则和 `confirmed_static` 安全覆盖。

本报告生成时使用的旧 v3 监督只有 6 条正式负例且全部在 train，因此本轮结果不是正式模型证据。现行 `training-labels-v3-validation.json` 已为 `training_ready.fall_event=true`，但本报告没有按新标签和 split 重训，历史分数不能因此解释为“已确认跌倒概率”，也不能作为正式 test、连续视频召回率、误报率或临床有效性证据。

## 数据

- 正类：`D01/D02/D03/D05` 明确跌倒动作。
- proxy 负类：`A03/A05/A06/A09` 明确非跌倒动作；未标注背景不推断为负类。
- 输入：8 FPS x 4 秒 = 32 帧，14 点，7 通道，张量形状 `[2323,32,14,7]`。
- train：1,264 个窗口，正/负 650/614，124 个保护组。
- validation：1,059 个窗口，正/负 540/519，11 个保护组。
- test：1,060 条标签只计数；姿态、特征和指标均未读取。
- 拒绝：观测帧不足 195、质量不足 23、无匹配轨迹 1。
- dataset SHA-256：`31237b05036f47abd094e8d5101bf5fc12925b7b89b1ce500d7033b7d92c2959`。
- derived split SHA-256：`444338aae47d7cf510a0ffafdc670303350f2a974a0ea77d65dd70398cbfd3a7`。

## Pilot 结果

运行目录：`reports/fall_risk/fall_event_proxy_v1/presence-pilot-seed42-v2/`。配置为 seed 42、CPU、40 epoch 上限、patience 8；epoch 30 early stopping，best epoch 为 22，用时 137.396 秒，参数量 8,134。

| validation 指标 | 实测值 |
|---|---:|
| Balanced accuracy | 0.944 |
| F1 | 0.945 |
| Precision / Recall | 0.955 / 0.935 |
| PR-AUC | 0.959 |
| ROC-AUC | 0.976 |
| 混淆矩阵 `[[TN,FP],[FN,TP]]` | `[[495,24],[35,505]]` |

独立 evaluator 复算结果完全一致。共保存 59 个 presence 二分类失败案例，其中误报 24、漏报 35。`subtype_loss_weight=0.0`，方向头没有训练；训练报告和独立评估均返回 `subtype.status=not_trained`，未训练方向不计入失败案例，也不对外输出方向结论。

分域 balanced accuracy 为：Fall Detection 2017 `0.692`、LE2I `0.963`、NTU RGB+D `0.956`。总体高分不能掩盖 Fall Detection 2017 的明显泛化下降。

- checkpoint SHA-256：`0eaab61587d3f26f41686b4265551c4143ac84803f1ba1d0aa61af1b8f6441a5`。
- run config SHA-256：`266d9ac392ba9cbe13a334822f8187638cac396b5588eca92c9b9a407dd82ead`。
- code fingerprint：`0a87561ae31bb71a3b3d9bf7fc1e54c148cad4c10f30c3bc2b551d0facab738f`。

## 复现命令

```bash
conda run -n eldercare-ai python scripts/prepare/prepare_fall_event_dataset.py \
  --allow-provisional

conda run -n eldercare-ai python scripts/train/train_fall_event_tcn.py \
  --profile pilot \
  --output-dir reports/fall_risk/fall_event_proxy_v1/presence-pilot-seed42-v2 \
  --allow-provisional

conda run -n eldercare-ai python scripts/evaluate/evaluate_fall_event_tcn.py \
  --data data/processed/fall_risk/fall_event_proxy_v1/dataset.npz \
  --metadata data/processed/fall_risk/fall_event_proxy_v1/metadata.json \
  --checkpoint reports/fall_risk/fall_event_proxy_v1/presence-pilot-seed42-v2/best_model.pt \
  --output reports/fall_risk/fall_event_proxy_v1/presence-pilot-seed42-v2/independent-validation.json \
  --device cpu
```

## 真实视频烟测

LE2I `Home_01/video (1).avi` 的前 100 帧依次经过 YOLOv8n-Pose、姿态质控/平滑和 TCN，生成 2 个可用 `provisional_shadow` 窗口；核心点覆盖率为 0.965/0.957，跌倒分数为 0.00143/0.00155，均未触发候选跌倒。该烟测只证明工程链路和低质量拒识可运行，不是效果评估。

## 下一门禁

补齐 train/validation/test 的正式 fall hard negative、连续背景和人工复核 onset/offset，冻结连续事件评估协议后，才能训练并评估正式 `fall_event_v1`。门禁通过前不读取 test、不让该 TCN 驱动实时告警。
