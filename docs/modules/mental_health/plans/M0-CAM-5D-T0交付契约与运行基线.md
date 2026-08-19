# M0-CAM-5D-T0 交付契约与运行基线

状态：`completed`

计划：Day 1 上午

上游：现有 camera/episode/model/synthetic product 底座

下游：[T1 B01+B02 分段器联合优化](M0-CAM-5D-T1-B01B02分段器联合优化.md)

## 目标

在开始调参前固定本轮算法交付范围、可重复运行入口、开发数据索引和输出 schema，避免五日冲刺中再次扩展审计或产品范围。

## 必须完成

1. 确认 editable 安装指向当前仓库。
2. 建立 B01+B02 联合 development index；两批均允许反复运行和调参。
3. 记录 fixed primary candidate、80 点输入、现有 `0.5` binary threshold 和 camera preprocessing identity。
4. 约定 fresh output 根目录和不覆盖规则。
5. 固定最终文件名：episode、context、daily、baseline、manifest、summary。
6. 定义统一状态 `ready/uncertain/unavailable/error` 和 source reference。
7. 将居家 MP4 登记为后续真实场景补充 smoke 输入；未到位时以 `awaiting_input` 记录，不阻塞任何 input-independent core、日报、基线或 handoff 工作。

## 输出

```text
development_index.jsonl
run_manifest.json
output_contract.md or schema definitions
baseline_verification.txt
```

实现与证据：

- builder：`scripts/wandering/build_camera_delivery_contract.py`；
- 配置：`configs/modules/wandering_camera_5d_contract_v1.yaml`；
- 接口契约：[徘徊五日算法交付契约](../../../interfaces/徘徊五日算法交付契约.md)；
- JSON Schema：`configs/schemas/wandering_5d_v1/`；
- 机器产物：`reports/mental_health/wandering_camera_5d_contract_v3/`。v1/v2 保留为 non-overwrite 历史运行；v2 首次显式记录 B02-0008 的 owner-confirmed legacy setup alias，v3 进一步固定 episode 的 technical segment、duration、reason 和 `auto_accepted/uncertain/rejected` 运行决策。

2026-08-18 最终 v3 构建得到 48 条 development index：B01 36 条、B02 12 条，48 个 `source_video_id` 唯一；video、tracking、sidecar、truth、CVAT XML 全部路径和 SHA-256 通过。47 条 setup 与 owner binding 一致，B02-0008 的旧 `office-setup-01` 显式记为 `owner_confirmed_legacy_alias`。fixed primary 在 CPU 以 `eval` 安全加载，manifest SHA-256 为 `225015cf05cbb12d177aba7168de86851713884ac659f306a0c957fdeb076409`。居家输入登记为 `awaiting_input`。后续执行决定已明确：该状态不阻塞 W5D-03A、W5D-04 或 W5D-05，居家 smoke 在素材到位后以 fresh 补充运行完成。

## 完成门

- candidate 能在 CPU 安全加载；
- B01/B02 index 中每条视频、tracking、sidecar、truth/CVAT 路径可解析；
- 输出目录 fresh 且不会覆盖历史证据；
- 当前相关聚焦测试通过；
- 任务表更新为 `W5D-01 in_progress`。

## 验证命令

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_primary_inference.py \
  tests/test_wandering_camera_episode_inference.py \
  tests/test_wandering_camera_episode_evaluation.py -q
```

## 不做

- 不实现后端或前端；
- 不重训模型；
- 不建立新的 receipt/privacy 审批；
- 不运行 C4、sealed 或 PORTABLE。

## 完成验证

```text
editable_project_location=/mnt/c/Users/lenovo/Desktop/心理算法
development_record_count=48
batch_counts=B01:36,B02:12
candidate_cpu_load=passed
runtime_device=cpu
input_point_count=80
binary_decision_threshold=0.5
schema_count=9
contract_tests=5 passed
full_camera_tests=624 passed, 1 expected negative-fixture warning
```
