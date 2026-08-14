# 跌倒风险实时链路工程回归

本目录保存实时链路运行指纹、离线回放结果和真实直播失败记录。烟测脚本已经增加阶段 2 的目标四态、分支三态、窗口质量、融合 mask、阶段耗时和 epoch 清理断言；但当前 `offline-service-smoke.json` 仍是阶段 0/1 生成的旧报告，不能作为阶段 2 回放通过的证据。

2026-07-30 已执行首次真实萤石 RTMP 算法端烟测。地址可打开并解码，但每次连接只返回约 14 帧后结束，70 秒内产生 4 个 epoch，未形成主目标姿态或特征窗口，停止读取线程也未在 5 秒预算内退出。详见[萤石真实直播算法端烟测](ezviz-live-smoke-20260730.md)。该失败记录不表示真实萤石联调完成。

2026-08-12 使用新的授权 HTTPS-FLV 地址和修复后的 FFmpeg 服务链完成 120 秒严格算法端烟测：单一 epoch、732 个处理帧、732 个主目标姿态、末端有效采样率 6.8514 FPS，四个关键分支均有效，0.520196 秒内停止。详见[萤石真实直播算法端烟测](ezviz-live-smoke-20260812.md)。该证据不替代业务后端回调、URL 刷新、弱网和一小时固定硬件验收。

当前冻结配置：`configs/modules/fall_risk_runtime_acceptance.yaml`。

复现命令：

```bash
conda run -n eldercare-ai python scripts/evaluate/build_fall_runtime_fingerprint.py \
  --config configs/modules/fall_risk_runtime_acceptance.yaml \
  --output reports/fall_risk/runtime/runtime-fingerprint.json

conda run -n eldercare-ai python scripts/collect/run_fall_service_smoke.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --model models/yolov8n-pose.pt \
  --max-frames 30 \
  --report reports/fall_risk/runtime/offline-service-smoke.json
```

检测、跟踪和姿态由当前 YOLOv8-Pose/ByteTrack 后端一次调用完成，报告只记录合并后端耗时，不能伪装成三个独立阶段的计时。新增帧后的真正增量窗口计算、固定硬件一小时连续运行、弱网、直播地址刷新和外部业务回调验收仍为阻塞项。
