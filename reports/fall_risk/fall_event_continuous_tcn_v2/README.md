# 连续跌倒事件 TCN v2 开发训练报告

状态：`development_provisional`。训练评估结论仍为 **No-Go**：不读取 test，不能据此解除正式模型晋级门禁。2026-08-18 的比赛交付配置另以显式 `experimental_tcn` 开关启用三枚 checkpoint 的临时主评分；这不是训练结论改判，规则仍作为窗口不可用/推理失败 fallback。

## 当前数据

- 事实源 split：`splitv3_3342705b7b1ac51570148337`。
- 治理清单：7,701 条开发监督、5,338 个视频，train/validation=`5,888/1,813`，姿态缺失 0。
- test 封锁：3,730 条标签在语义解析和姿态访问前跳过。
- 因果窗口：7,504 个 `[32,17,20]` 样本，train/validation=`5,721/1,783`。
- train 正/负=`1,442/4,279`；validation 正/负=`394/1,389`。
- onset 监督 train/validation=`66/7`。
- `future_observation_count=0`；cleaned pose 中插值坐标及其派生运动已屏蔽。
- SCF_MVP_V1 的 298 条监督全部进入 train，不能据此估计自采域 validation 泛化。

旧 `fall_event_continuous_governance_v1` 绑定 `splitv3_c7fd...`，不作为当前结果。本报告和 v2 产物重新绑定当前 action/event/manifest/assignments/split/validation 六个哈希。

## 训练结果

三组仅改变 seed。固定配置为 4 秒、8 FPS、causal residual Conv1D、hidden channels 48、dilation `1/2/4/8`、dropout 0.1、batch size 64、AdamW、learning rate `1e-3`、最多 25 epoch、patience 6、CPU。归一化只拟合 train，frame mask 保持 0/1。

| seed | best epoch | F1@0.5 | precision | recall | PR-AUC | onset accuracy (n=7) |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 16 | 0.6829 | 0.6573 | 0.7107 | 0.6750 | 0.4286 |
| 43 | 11 | 0.6732 | 0.5897 | 0.7843 | 0.6530 | 0.1429 |
| 44 | 8 | 0.6675 | 0.6625 | 0.6726 | 0.7066 | 0.2857 |
| mean | - | 0.6745 | 0.6365 | 0.7225 | 0.6782 | 0.2857 |
| population std | - | 0.0064 | 0.0332 | 0.0464 | 0.0220 | 0.1166 |

三模型平均概率在固定阈值 0.5 下为 F1 `0.6824`、precision `0.6360`、recall `0.7360`、PR-AUC `0.6895`、误报率 `0.1195`。在同一 validation 上选择最大 F1 阈值 `0.27422556` 后，F1/recall 提高到 `0.7084/0.9188`，但 precision 降至 `0.5764`、误报率升至 `0.1915`。validation 同时用于早停和阈值选择，因此后者是偏乐观的开发指标。

## 分层与失败案例

固定阈值 0.5 的分层结果：

| 分层 | 样本 | 正/负 | F1 | recall | false-positive rate | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| fall_detection_2017 | 298 | 94/204 | 0.7273 | 0.7660 | 0.1569 | 0.7982 |
| LE2I | 151 | 7/144 | 0.3784 | 1.0000 | 0.1597 | 0.6085 |
| NTU RGB+D | 1,334 | 293/1,041 | 0.6862 | 0.7201 | 0.1066 | 0.6820 |
| full context | 225 | 7/218 | 0.3784 | 1.0000 | 0.1055 | 0.6085 |
| partial context | 1,558 | 387/1,171 | 0.6962 | 0.7313 | 0.1221 | 0.7154 |

固定阈值下有 166 个 false positive 和 104 个 false negative：

- 普通背景误报 146/690（`0.2116`），其中 `A01/normal_walk` 占 135 条，是首要失败族。
- 显式困难负例误报 20/402（`0.0498`），包括 controlled sit-down 10 条和遮挡/相机运动 10 条。
- 恢复型近跌倒 297 条在阈值 0.5 下无误报；降至 `0.27422556` 后出现 6 条误报。
- 104 个漏报均来自 auxiliary fall，其中 Fall Detection 2017 为 22 条、NTU RGB+D 为 82 条。
- primary fall/onset validation 只有 7 条，无法据其 100% presence recall 或多数投票 onset accuracy `0.1429` 声称泛化通过。

## 运行链路状态

- `configs/modules/fall_risk_service.yaml` 已配置 seed42/43/44 三枚 checkpoint。
- `FallRiskSessionEngine` 在清洗姿态形成 4 秒因果窗口后计算平均概率；`experimental_tcn` 模式下写入 `fall_event_tcn_score`，有效窗口成为 `fall_event_score` 主来源。
- 模型命中沿用现有 0.9 强触发契约，进入融合、episode 状态机和 `AlgorithmEvent` 回调；原始概率、来源和 onset 保留在诊断元数据。
- 低于 4 秒历史、关键点质量不足、窗口异常或 checkpoint 加载失败时，TCN 输出 `unavailable/inference_error`，`fall_event_score` 自动回退规则状态机。
- 该接入证明的是可运行的 provisional 主链路和降级稳定性，不代表训练评估门禁或正式替换授权已通过。

## No-Go 原因

1. 普通行走背景误报仍高，尚无连续 camera-hour 分母、跨房间长视频或真实流事件级评估。
2. onset validation 只有 7 条，三 seed onset accuracy 低且不稳定。
3. validation 同时承担早停和阈值选择；split 与协议尚未冻结，test 保持封锁。
4. SCF 只在 train，缺少独立自采、老人、辅助器具、低光和遮挡 validation。
5. 下一步优先补 A01 连续行走背景与精确 onset 标注，并做事件状态机和每小时误报评估；在这些证据前保持 provisional 开关和规则 fallback，不宣称正式模型替换。

## 复现

```bash
conda run -n eldercare-ai python scripts/prepare/govern_fall_event_training_data.py \
  --governance-config configs/data/fall_event_continuous_training_governance_v2.json \
  --output-dir reports/fall_risk/fall_event_continuous_governance_v2

conda run -n eldercare-ai python scripts/prepare/prepare_fall_event_continuous_dataset.py \
  --governance-manifest reports/fall_risk/fall_event_continuous_governance_v2/training_manifest.jsonl \
  --output-dir reports/fall_risk/fall_event_continuous_dataset_v2 \
  --window-sec 4 --target-fps 8 --max-gap-sec 0.25 \
  --min-observed-frames 16 --min-partial-observed-frames 8 \
  --min-valid-joint-ratio 0.5 --max-interpolated-joint-ratio 0

conda run -n eldercare-ai python scripts/train/train_fall_event_continuous_tcn.py \
  --data reports/fall_risk/fall_event_continuous_dataset_v2/dataset.npz \
  --metadata reports/fall_risk/fall_event_continuous_dataset_v2/metadata.json \
  --output-dir reports/fall_risk/fall_event_continuous_tcn_v2/seed42 \
  --seed 42 --epochs 25 --patience 6 --batch-size 64 \
  --hidden-channels 48 --device cpu

conda run -n eldercare-ai python scripts/evaluate/evaluate_fall_event_continuous_tcn.py \
  --data reports/fall_risk/fall_event_continuous_dataset_v2/dataset.npz \
  --samples reports/fall_risk/fall_event_continuous_dataset_v2/samples.jsonl \
  --checkpoint reports/fall_risk/fall_event_continuous_tcn_v2/seed42/best_model.pt \
  --checkpoint reports/fall_risk/fall_event_continuous_tcn_v2/seed43/best_model.pt \
  --checkpoint reports/fall_risk/fall_event_continuous_tcn_v2/seed44/best_model.pt \
  --output-dir reports/fall_risk/fall_event_continuous_tcn_v2 \
  --batch-size 128 --device cpu
```

## 产物哈希

| 产物 | SHA-256 |
|---|---|
| governance config | `89e681d554e0dbb464c6f70590be7918d5cb2b93146a988e36cda33a305e508c` |
| governance manifest | `14c5f88f5f25c4264efc1725b475f36de69b8ccd6d669b73140809aa54dfbc04` |
| dataset.npz | `34993ae9a6b599ce9452b59e4bf68cde192116bb0d3f3bad25e70275b6c62f71` |
| samples.jsonl | `88e0ad0f1cc5ec95a1a0e0ee94564d05377dc29189228160e92b7b18786d5dc5` |
| seed42/best_model.pt | `650ab9016d3d4c6f682f524c85c3ed09f02bba526d2e83f7cdc8fac87b08d99c` |
| seed43/best_model.pt | `b4c60ab009730ca83bb0f0d694af1580d89cd9d1a05876c7599a0e71c214fc79` |
| seed44/best_model.pt | `c1602d9d1a2c12037b6a4ebae4ed560e0b9e6c5216fe263afcafb23046ad852e` |
| aggregate_metrics.json | `af9c5cb2a9615ed505afc6ffc878c1bd0d9e8a12bdd5d561f5059b04ce2ec4f2` |
