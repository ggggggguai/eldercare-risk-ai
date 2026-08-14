# 坐站连续事件训练阻塞清单

日期：2026-08-11
当前状态：`development_provisional`，seed 42 pilot 未达线，三 seed No-Go

- `event_gate`：完整流 event F1 0.351、Recall 0.403，低于 0.75/0.85 门禁。
- `boundary_head`：加权后 exact F1 0.205、±1 帧 F1 0.348；完整流 onset 中位误差 0.633 秒，仍未达线。
- `hard_negatives`：bed transfer、bend、squat、kneel/floor 误报率分别为 35.3%、37.1%、17.7%、66.7%，未达到 `<10%`。
- `source_robustness`：多个 UR Fall 小来源组 F1 为 0，CAUCAFall 组仅 0.167-0.286，尚无稳定最差来源证据。
- `partial_context`：3,759 个窗口中 2,294 个使用部分上下文，需要完整上下文消融和分层指标。
- `ambiguous_targets`：183 个同帧多人区间仍无法可靠绑定标注目标，保持拒绝。
- `continuous_background`：完整 validation 流评估只有 0.286 camera-hour 显式背景，低于稳定 FP/hour 结论所需的 5 camera-hour。
- `frozen_approval`：split、评估协议和 checkpoint 均未冻结，默认 checkpoint 保持 `null`。
- `test_release`：没有独立 release ID；test 姿态、特征和指标继续锁定。

已完成项：边界正样本加权、连续事件解码、长 gap fail-closed、同协议 S0 规则比较、36 组 bootstrap 修复和 seed 42 完整 pilot。

下一步不是追加 seed 或 epoch，而是补足至少 5 camera-hour 连续显式背景，针对弯腰/蹲下/床边转移补困难负例，并修复来源退化；保持同一冻结协议重新跑 seed 42，达到绝对门禁后才允许 seed 43/44。
