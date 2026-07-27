# 跌倒风险实时链路工程回归

本目录保存实时链路运行指纹和离线回放结果。烟测脚本已经增加阶段 2 的目标四态、分支三态、窗口质量、融合 mask、阶段耗时和 epoch 清理断言；但当前 `offline-service-smoke.json` 仍是阶段 0/1 生成的旧报告，本轮因执行环境禁止 localhost 回调监听且提权审批服务不可用，未能重新生成。它不能作为阶段 2 回放通过的证据。

当前冻结配置：`configs/modules/fall_risk_runtime_acceptance.yaml`。

复现命令：

```bash
conda run -n eldercare-ai python scripts/evaluate/build_fall_runtime_fingerprint.py \
  --config configs/modules/fall_risk_runtime_acceptance.yaml \
  --output reports/fall_risk/runtime/runtime-fingerprint.json

conda run -n eldercare-ai python scripts/collect/run_fall_service_smoke.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --model yolov8n-pose.pt \
  --max-frames 30 \
  --report reports/fall_risk/runtime/offline-service-smoke.json
```

检测、跟踪和姿态由当前 YOLOv8-Pose/ByteTrack 后端一次调用完成，报告只记录合并后端耗时，不能伪装成三个独立阶段的计时。新增帧后的真正增量窗口计算、真实萤石直播、固定硬件一小时连续运行、弱网和外部回调验收仍为阻塞项，不在离线结果中标记完成。
