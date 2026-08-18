# 近跌倒恢复确认 TCN v2 数据治理、三 seed 与动作辅助负例消融

执行日期：2026-08-17  
统一 split：`splitv3_c7fd01cf4473c02a57c68fbe`  
结论：v2 数据治理、SCF E1 多尺度负例增强和三 seed 开发训练完成；A01/A04 动作辅助负例又完成 seed 42 三档权重消融。辅助数据提高了 primary validation F1，但 P05 误触发明显增加，仍为 `No-Go`，不替换 `near-fall-rule-v0.1` 主路径。

## 1. 数据治理

v2 先在人工标签区间内选择 dominant pose track，再从完整姿态 JSONL 回取同一 track 的因果前序上下文，不跨 track 拼接。窗口优先使用 3 秒；3 秒不足时回退到 2 秒，并在固定 24 帧模型输入中左侧填充无效 mask。每个样本记录 `context_sec`、`context_frames`、`context_length` 和 `padding_frames`，每个标签另写 `audit.jsonl`。

base 构建器读取统一 v3 事件和公共姿态缓存。全部 2,854 条 primary 事件标签都有审计记录，其中 665 条 test 标签仅记为 locked，未读取姿态或真值；2,189 条 train/validation 标签中 1,626 条物化成功，563 条被拒绝。SCF 最终根发布是动作标签，不会自动变成近跌倒事件监督；因此另通过 E1 的受审映射仅加入 A02/A03/A05/A06/A08/A10 负例，C03-C05 粗代理不进入正例 loss。

| 项目 | 数量 |
|---|---:|
| train / validation 窗口 | 2,410 / 1,432 |
| 3 秒 / 2 秒窗口 | 1,930 / 1,912 |
| 物化事件 | 1,695 |
| SCF train 窗口 / 事件 | 242 / 69，全部为受审负例 |
| base 上下文不足 / 质量不足 | 558 / 16 个窗口尝试 |
| test 标签 | 665，`test_pose_read=false` |

最终 E1 多尺度数据集 SHA-256：`f9daf22e1a10e2d270f68683f8dbc45caa83ba1e6d91dae35e8752ab7d25f8d9`。输入事件标签 SHA-256 前缀为 `992017e2`；旧 `near_fall_event_v1` 和构建期间的过渡产物均保留，最终结果使用带该输入前缀的独立目录。base 与 E1 增强各执行两次，dataset、samples 和 base audit 的对应 SHA-256 完全一致。

## 2. 三 seed 结果

固定阈值为 0.5。校准阈值只使用 validation，并选择满足事件 Recall ≥0.90 时 Precision 最高的阈值；P05 不参与阈值拟合。

| seed | validation event F1 / P / R | 校准阈值 | P05 负例触发 | P05 正例代理检出 |
|---:|---:|---:|---:|---:|
| 42 | 0.8772 / 0.8410 / 0.9167 | 0.5340 | 13/34 | 10/16 |
| 43 | 0.8395 / 0.7520 / 0.9500 | 0.6808 | 28/34 | 14/16 |
| 44 | 0.8567 / 0.7824 / 0.9467 | 0.7910 | 22/34 | 11/16 |

旧 E0 在 P05 的 seed 42 固定阈值结果为负例 `32/34`、正例代理 `15/16`。v2+E1 多尺度的 seed 42 降到 `13/34`，但正例代理也降到 `10/16`；seed 43/44 又回到 `28/34` 和 `22/34` 负例触发。改善不能跨 seed 稳定复现，且三个 seed 都有不同程度的代理正例损失。

旧 v1 validation F1 接近 1 的主要原因是大量短标签未物化且 validation 困难负例单一。v2 的较低 F1 来自更完整、更困难的数据口径，不能直接解释为模型退步，也不能与旧口径做同分母排名。

## 3. A01/A04 动作辅助负例消融

当前 v3 动作标签中，A01 `normal_walk` 和 A04 `normal_sit_to_stand` 只作为单人标注动作区间，不能冒充双审 near-fall primary negative。新物化器把它们标为 `supervision_tier=auxiliary`，test 仅审计不读姿态；action-train 可按低权重进入 loss，action-validation 单独物化为 challenge，不参与训练、早停或阈值校准。

按当前标签与 split 实际筛出 2,670 条 A01/A04 primary 动作标签，其中 485 条 test 锁定。开发集有 523 条标签通过同轨因果上下文和姿态质量门禁，生成 1,269 个窗口：802 个 train 窗口进入消融，467 个 validation 窗口只进入独立 challenge 产物；1,662 个窗口尝试因上下文不足拒绝，1 个因质量不足拒绝。训练数据共 4,644 个窗口，primary validation 的 668 个事件成员保持不变。

| action auxiliary 权重 | primary validation event F1 / P / R | 校准阈值 | P05 负例触发 | P05 正例代理检出 |
|---:|---:|---:|---:|---:|
| 0，原 SCF E1 seed 42 | 0.8772 / 0.8410 / 0.9167 | 0.5340 | 13/34 | 10/16 |
| 0.10 | 0.8906 / 0.8489 / 0.9367 | 0.7379 | 25/34 | 14/16 |
| 0.25 | 0.9003 / 0.8696 / 0.9333 | 0.6563 | 24/34 | 14/16 |
| 0.35 | 0.8973 / 0.8529 / 0.9467 | 0.8018 | 21/34 | 12/16 |

0.25 在 primary validation 上最佳，0.35 在三档辅助实验中 P05 负例最少，但三档都明显差于原 SCF E1 的 `13/34`。这说明 A01/A04 能改善同分布普通动作边界，却没有覆盖 P05 的 A02/A03/A05/A06/A08/A10 困难动作分布；增加正例代理检出不能抵消误触发退化。因此不运行辅助实验 seed 43/44，不将任一辅助 checkpoint 晋级。

首次试跑曾把 467 个 action-validation 窗口混入早停和阈值选择；该口径会改变 primary validation 成员，不能与原实验公平比较，已明确排除。目录 `splitv3-c7fd01cf-input992017e2-scf-e1-ms-action-aux` 和训练目录 `action-aux-seed42` 只保留为失败协议追溯，不进入上表或任何晋级判断。

0.25 train-only 数据集 SHA-256 为 `5b3915be7d55beba5e92de159eca2c5e3148dd884f63838d703cb38023535cce`。0.10 和 0.35 的 SHA-256 分别为 `46f34a16120898d13caf24e493be0412f26aa4376e45a4a8a0485f930a3ccfc3`、`3a0ed4f11543118a76aec98fa8c5aba7e1c1ec9c15c4923caf6917debd28f320`。所有实验均 `test_pose_read=false`、`test_evaluated=false`。

## 4. 决策

- 当前没有满足稳定性和敏感度要求的可晋级 checkpoint；seed 42 只保留为低误触发方向的研究候选。
- 主链路替换：`No-Go`。规则继续作为运行主路径、低质量 fallback 和安全覆盖。
- test：未读取、未评估。
- 下一优先级：补连续老人域背景和 P05 同类但新人员的 A02/A03/A05/A06/A08/A10 困难负例；A01/A04 已证明不是当前误触发的主要缺口。冻结 split 与事件协议后再做阈值和模型晋级。
- 当前短动作片不支持 FP/hour，C03/C04/C05 仍是粗粒度动作末端代理，不是精确 recovery 真值。

## 5. 复现

```bash
conda run -n eldercare-ai python scripts/prepare/prepare_near_fall_event_dataset.py \
  --additional-pose-dir reports/fall_risk/self_collected_scf_mvp_v1/baseline_replay/cleaned_pose \
  --output-dir data/processed/fall_risk/near_fall_event_v2/splitv3-c7fd01cf-input992017e2-base

conda run -n eldercare-ai python scripts/prepare/prepare_self_collected_near_fall_augmentation.py \
  --experiment E1 \
  --base-dir data/processed/fall_risk/near_fall_event_v2/splitv3-c7fd01cf-input992017e2-base \
  --output-dir data/processed/fall_risk/near_fall_event_v2/splitv3-c7fd01cf-input992017e2-scf-e1-ms

conda run -n eldercare-ai python scripts/train/train_near_fall_tcn.py \
  --data data/processed/fall_risk/near_fall_event_v2/splitv3-c7fd01cf-input992017e2-scf-e1-ms/dataset.npz \
  --metadata data/processed/fall_risk/near_fall_event_v2/splitv3-c7fd01cf-input992017e2-scf-e1-ms/metadata.json \
  --config configs/training/near_fall_event_v2.yaml \
  --profile pilot --seed 42 \
  --output-dir reports/fall_risk/near_fall_event_v2/splitv3-c7fd01cf-input992017e2-scf-e1-ms/seed42

conda run -n eldercare-ai python scripts/prepare/prepare_near_fall_action_auxiliary.py \
  --base-dir data/processed/fall_risk/near_fall_event_v2/splitv3-c7fd01cf-input992017e2-scf-e1-ms \
  --auxiliary-loss-weight 0.25 \
  --output-dir data/processed/fall_risk/near_fall_event_v2/splitv3-c7fd01cf-input992017e2-scf-e1-ms-action-aux-trainonly
```
