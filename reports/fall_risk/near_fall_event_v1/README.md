# 近跌倒恢复确认 TCN provisional pilot

日期：2026-08-04

## 结论

本轮完成了基于现行 v3 标签和 `splitv3_f89832f6cad5b9c5f630d00b` 的近跌倒恢复确认 TCN train/validation 开发训练，固定 seed `42/43/44` 各运行一次。模型只确认**已经看到恢复帧后的近跌倒**，不是 onset-time 提前预警器；test 姿态、标签和指标均未读取，规则 `near-fall-rule-v0.1` 仍是实时主路径。

三 seed 的 validation 指标较高，但不能解释为连续视频效果：开发窗口全部来自 NTU RGB+D，且实际可用的负例窗口只有 `fast_but_controlled_sit` 和 `controlled_squat`。其余 hard-negative 类型虽在标签层有 primary 覆盖，但因无匹配姿态或因果上下文未进入本轮窗口。因此本轮产物保持 `provisional_shadow`，不接入告警、不替换规则。

## 数据与门禁

- v3 validation：`valid=true`，`training_ready.near_fall_event=true`。
- primary 标签：near-fall 正例 `948`，负例 `1,906`；八类 hard negative 均有标签层覆盖。
- source split：`splitv3_f89832f6cad5b9c5f630d00b`，跨 subject/source/event/sample 保护组无泄漏。
- 开发 dataset：`876` 个窗口、`756` 个事件、`35` 个 subject；train `445`（347/98），validation `431`（298/133）。
- test 锁定：`733` 条 test 标签只计数；`test_pose_read=false`、`test_evaluated=false`。
- 来源：876 个窗口全部为 `ntu_rgbd`。
- 实际负例窗口：`fast_but_controlled_sit=225`、`controlled_squat=6`。
- 拒绝窗口：因果上下文不足 `696`、无匹配姿态轨迹 `666`、质量不足 `3`。
- dataset SHA-256：`2d0eefb2a62f60ce14beb8f7098bc98e37cda849c354b8d281212898f57a108b`。
- metadata SHA-256：`a4867cedc64329a27f619fdefffd6aa9322274a7e7634968aa1ac6eab2ca0eb7`。

## 三 seed 结果

指标由固定 checkpoint 在 validation 分区按阈值 `0.5` 独立复算；PR-AUC、ROC-AUC 和 Brier 仅是本开发分区统计，不是正式评估协议结果。

| seed | best epoch | epochs | Precision | Recall | F1 | Balanced accuracy | PR-AUC | ROC-AUC | Brier |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 14 | 23 | 0.993 | 1.000 | 0.997 | 0.992 | 1.000 | 0.999 | 0.0043 |
| 43 | 21 | 30 | 0.987 | 1.000 | 0.993 | 0.985 | 1.000 | 1.000 | 0.0062 |
| 44 | 24 | 33 | 0.980 | 1.000 | 0.990 | 0.977 | 1.000 | 1.000 | 0.0086 |

训练器原始 validation loss/accuracy：seed 42 为 `0.0100/0.9984`，seed 43 为 `0.0141/0.9952`，seed 44 为 `0.0160/0.9931`。三次运行均 `synthetic=false`、`test_evaluated=false`。

## 运行产物

- 数据：`data/processed/fall_risk/near_fall_event_v1/splitv3-f89832/`
- seed 42：`reports/fall_risk/near_fall_event_v1/pilot-seed42-splitv3-f89832/`
- seed 43：`reports/fall_risk/near_fall_event_v1/pilot-seed43-splitv3-f89832/`
- seed 44：`reports/fall_risk/near_fall_event_v1/pilot-seed44-splitv3-f89832/`

各目录保存 checkpoint、last checkpoint、训练 config、metrics 和数据 provenance。checkpoint SHA-256 分别为：

- seed 42：`be4d24b9c72a84bbe3bf1c8f9be707009ffd333352d462ef21ec13560eb9b01e`
- seed 43：`8dd0bd7eb6ca0eaf45b28ba48899ed0364ff4e8203d779c1868a5a8048ad98f3`
- seed 44：`d7229c48933560221ea3dbb01633d6ee385ab21ef61fa63eeb036e84e5dc19a`

## 复现命令

```bash
conda run -n eldercare-ai python scripts/prepare/prepare_near_fall_event_dataset.py \
  --output-dir data/processed/fall_risk/near_fall_event_v1/splitv3-f89832

conda run -n eldercare-ai python scripts/train/train_near_fall_tcn.py \
  --data data/processed/fall_risk/near_fall_event_v1/splitv3-f89832/dataset.npz \
  --metadata data/processed/fall_risk/near_fall_event_v1/splitv3-f89832/metadata.json \
  --profile pilot \
  --output-dir reports/fall_risk/near_fall_event_v1/pilot-seed42-splitv3-f89832
```

seed 43/44 使用相同输入、配置和 pilot 超参数，仅将训练 seed 改为 `43/44`；三次运行均未读取 test。

## 下一门禁

下一步不是继续追加 epoch，而是补齐能生成实际窗口的跨来源 hard negative、连续背景和非 NTU 姿态证据，复核窗口拒绝原因并冻结 validation 协议。取得连续背景分母、老人域外部验证和正式 test 发布流程前，模型不得替换规则主路径或驱动实时告警。
