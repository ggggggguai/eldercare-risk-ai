# Verification

环境：WSL `Ubuntu-22.04`，conda `eldercare-ai`。

已运行的核心验证：

```text
python -m pytest tests/test_wandering_camera_episode_boundary.py \
  tests/test_wandering_camera_episode_proposal_inference.py \
  tests/test_wandering_camera_geometry_head.py -q
33 passed
```

```text
run_camera_home_repair.py --tracking-root <home-output> --output-dir <all70_v5>
video_count=70, proposal_count=137, ready_count=137, unavailable_count=0
```

```text
evaluate_camera_episode_boundaries.py <labeled_v2>
parent: 137 candidates, 110/135 matched
```

```text
evaluate_home_repair_classification.py <all70_v5 resolved>
resolved: 244 candidates, 123/131 matched
direct recall=0.933962, pacing recall=0.888889
```

```text
python -m pytest tests/test_wandering_camera*.py -q
677 passed, 1 warning in 209.46s
```

唯一警告来自 portability 防篡改测试主动构造重复的
`candidate/candidate_manifest.json` ZIP entry；测试本身通过，不是本次修复失败。

仓库全量附加检查：

```text
python -m pytest -q
1 failed, 1350 passed, 2 warnings, 243 subtests passed in 377.93s
```

唯一失败为
`tests/test_fall_runtime_fingerprint.py::FallRuntimeFingerprintTest::test_repository_acceptance_config_has_hashable_fixed_inputs`：
跌倒配置指定的
`data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi`
在当前 checkout 不存在。该失败属于跌倒固定外部输入缺失，不涉及本轮徘徊代码；本轮未修改
跌倒配置，也未制造占位视频规避门禁。
