# 跌倒风险比赛交付 v2 冻结记录

日期：2026-08-27

发布 ID：`fall-risk-competition-v2-20260827`

状态：`competition_release_frozen`

## 冻结结论

项目负责人要求按当前证据和运行契约重新冻结。v2 在 v1 基础上正式晋级近跌倒 ExtraTrees 重评分链路；步态、连续坐站和跌倒事件 checkpoint 保持不变。

近跌倒实时链路为：规则生成候选 -> 3 秒因果窗口（不足时尝试 2 秒）-> seed 42 ExtraTrees 重评分 -> 阈值 `0.3815` -> 近跌倒告警冷却 15 秒。模型窗口不可用或推理失败时回退规则候选，不把异常输入伪装成模型低风险。

坐站 Random Forest 不进入 v2 主链。其内部确认集片段存在性 F1 为 `0.9543`，但不输出动作方向和事件起止边界；统一 115 段连续回放坐站事件 F1 为 `0.2738`。v2 继续使用能够满足连续因果事件契约的 seed 42 TCN，并保留规则 fallback。

机器事实源为 `configs/modules/fall_risk_release_v2.yaml`，默认服务配置为 `configs/modules/fall_risk_service_v2.yaml`。旧 `fall-risk-competition-v1-20260819` 及其 `fall_risk_service.yaml` 保持不可变，可用于回滚和历史追溯。

## 冻结组合

| 环节 | v2 冻结实现 |
| --- | --- |
| 人体/姿态/跟踪 | YOLOv8n-Pose + ByteTrack |
| 步态 | TCN seed 43 + 规则 fallback |
| 坐站 | 连续 TCN seed 42 + 规则 fallback |
| 近跌倒 | 规则候选 + ExtraTrees seed 42 重评分 + 规则 fallback |
| 跌倒事件 | 三 seed 连续 TCN 集成 + 规则 fallback |
| 个体基线 | 统计实现，缺少有效周期时 fail closed |
| 最终融合 | 规则融合 |

## 证据边界

近跌倒候选在暗光开发验证为 F1=`0.800`、Recall=`1.000`；固定参数后的 hall 开发挑战为 F1=`0.714`；115 段完整工程回放为 Precision=`0.7419`、Recall=`0.8519`、F1=`0.7931`。这些视频来自单一成人、单一家庭和同一采集批次，hall 场景也不是未触碰 test，因此冻结只表示负责人接受其作为比赛交付版本，不证明跨人员、老人域或临床泛化。

`event_hold_sec=0.75` 只用于离线事件匹配区间，不是实时未来观察或告警延迟。实时告警只使用当前及历史姿态形成的因果证据。

历史 v2 formal blocker、未冻结研究 split/协议、未读取锁定 test、连续背景和老人域缺口继续保留，不因本次冻结而改变。

## 变更规则

v2 不可原地覆盖。更换 checkpoint、路径、阈值、特征契约、服务运行配置或冻结源码后，必须创建新的 release ID 并重新验证。

## 冻结验证

环境绑定确认：`elderly-monitoring-algorithms` editable 安装指向当前仓库。

```bash
conda run -n eldercare-ai python -m pytest -q
```

结果：`696 passed, 1 skipped, 98 subtests passed`。

真实视频姿态烟测：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_pose.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --output /tmp/fall_poses_v2_freeze_smoke.jsonl \
  --model yolov8n-pose.pt \
  --scene-region home \
  --max-frames 5
```

结果：成功输出 5 条姿态关键点记录。发布清单中的 16 个配置、权重、证据和关键源码绑定项全部通过 SHA-256 自检；交付 ZIP 完整性检查无错误。
