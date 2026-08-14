# M0-CAM-C01-F2 verification

验证日期：2026-08-13

## 环境与结果

- editable install：`elderly-monitoring-algorithms 0.2.0`，指向当前 `<repo>` checkout。
- focused：`118 passed in 16.41s`。
- 全部九个 camera 文件：`234 passed in 103.91s`。
- 六个 CLI `--help`：全部 exit 0；未新增 authorization/evidence/scope/context/fake/test/skip/bypass 参数。
- `camera_adapter.py` SHA-256：`a2617dd688e0e200ff07d28ab7b9e93e3d66ff3e5d01023146eb3f5ddd75f331`，与 Step8 冻结信任根一致。

执行命令均使用项目环境：

```text
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
conda run -n eldercare-ai python -m pytest tests/test_wandering_camera_adapter.py tests/test_wandering_camera_dataset.py tests/test_wandering_camera_collection.py -q
conda run -n eldercare-ai python -m pytest tests/test_wandering_camera_adapter.py tests/test_wandering_camera_collection.py tests/test_wandering_camera_corruption.py tests/test_wandering_camera_dataset.py tests/test_wandering_camera_development.py tests/test_wandering_camera_episode.py tests/test_wandering_camera_inference.py tests/test_wandering_camera_primary_inference.py tests/test_wandering_camera_qc.py -q
```

CLI help 逐项运行：`build_camera_tracking_pair.py`、`prepare_camera_session.py`、`build_camera_annotations.py`、`run_camera_development.py`、`run_camera_inference.py`、`run_topowander_camera_inference.py`。

## 攻击覆盖

1. fake second checkout 在 receipt/data touch 前拒绝；
2. 正确 root、`.` 与 samefile symlink alias 接受；
3. DrvFS case alias、symlink 和 `..` 回仓库拒绝；
4. 两份外部同内容 config 均拒绝，samefile config alias 接受；
5. missing/pending/inactive/expired/operation-invalid receipt 之后所有非 receipt 角色零触碰；
6. valid receipt + invalid collection/scope/source 之后 video/tracking/sidecar/output 零触碰；
7. pair-only final 只含统一 SHA 与 declaration basis；
8. video-direct final 只含统一 SHA 与 controller pre/post basis；
9. basis/SHA/source/flags/provenance mismatch 均无 final；
10. return summary 等于 disk summary，三文件同 staging 重载并原子提交；
11. backend/version path、URL、credential、control、超长攻击在 raw temp/tracker 前拒绝；
12. `8.3.0`、`1.0+cpu`、`test-version`、`2026.08-rc1` 接受；
13. final summary 与 sidecar 递归扫描无绝对路径、URL、credential 或 fixture 注入；
14. public authorized-sidecar/context bypass 继续关闭；
15. tracker 后同长度媒体变化清理 raw 且无 final；
16. owner markdown templates 继续 formal-reject；
17. public API/CLI 无新增授权、证据、scope、context、fake/test/skip/bypass 参数；
18. existing、in-checkout、symlink/alias 和 dangling output 拒绝，失败路径清理 staging/raw。

## Git 与副作用

- 验证基线 HEAD：`d3c101bc459234418832a5100c0b5c29ac518dc3`。
- F2 功能验证完成时未 commit、push、stash、reset，且暂存区为空；后续本地集成状态以 Git 历史和任务表为准。
- PREP/F 原报告与 post-audit 记录未覆盖；Step8 配置、结果和冻结 adapter SHA 未改。
- F2 开始前的工作树已有 PREP/F/RD-F 等未提交变更；F2 验证过程保留这些变更，并只增量修改所需文件。
