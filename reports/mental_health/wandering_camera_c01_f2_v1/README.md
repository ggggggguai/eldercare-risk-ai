# M0-CAM-C01-F2 handoff integrity hardening

完成日期：2026-08-13

## 结论

```text
status=wandering_m0cam_c0_c1_handoff_hardened_waiting_owner_inputs
c01_f_complete=true
c01_f2_complete=true
evidence_scope=synthetic_schema_contract_only
C0=false
C1=false
authorized_camera_data_consumed=false
m0cam_d_started=false
```

F2 已在不读取真实或授权相机数据的前提下关闭四组完整性缺口。当前只表示 C0/C1 软件交接入口已加固；不表示负责人授权成立、C1 已采集、QC/模型已运行或取得 camera 性能证据。

## 实现结果

- active checkout 从实际导入的 `camera_dataset.py` 向上定位唯一 `pyproject.toml + src/elderly_monitoring` 根；caller `project_root` 和两份公开配置路径只作 `samefile()` 身份断言。生产 receipt、collection、tracking、sidecar、video、output 使用最近既存祖先与 samefile-aware 边界检查，拒绝 fake root、外部配置副本、大小写别名、`..` 与 symlink 回仓库。
- pair-only 与 video-direct 均恢复 receipt-first 外部路径触碰顺序。role-aware spy 证明 invalid receipt 之后 collection/tracking/sidecar/video/output 的 resolve/stat/read/hash/open/runtime/temp/tracker 为零；invalid collection/scope/source 之后所有后续路径触碰为零。
- final `source_binding` 统一为 `source_sha256 + source_sha256_basis`。pair-only 只能写 `validated_sidecar_declaration` 且 execution flags 为 false；video-direct 只能写 `controller_observed_video_pre_and_post_tracking`，SHA 来自 tracker 前后相等的实际视频哈希，execution flags 为 true。旧 `observed_source_sha256`、`binding_basis` 与 video metadata 重复 SHA 不再落盘。
- 新增共享 portable component validator，synthetic、authorized pair-only 与 video-direct 共用；backend/version 的路径、URL、credential、控制符和超长攻击在 protected work 前闭锁，普通 semver、`+cpu`、test/release-candidate token 可通过。Step8 冻结的 `camera_adapter.py` 保持原 SHA，未改旧 augmentation 配置或结果。
- tracking、sidecar 与 summary 仍在同一 staging 中重载后原子提交；返回 summary 与磁盘 summary 相等。basis/SHA/source/flag 篡改、同长度 tracker 后媒体变化、既存/dangling output 均无 final，staging/raw temp 清理。

## 数据与执行边界

测试只使用 `tmp_path`、synthetic/in-memory metadata、symlink 和 monkeypatch fake tracker/video，临时制品均即时消费或清理。未 resolve、搜索、读取、哈希或运行任何真实/授权 camera 数据；未访问 WP raw、SmartCare official 或 sealed 数据；未运行真实 tracker、Camera QC、RF/TCN、fixed primary、evaluator 或 M0-CAM-D；未修改 schema、shared tracker、candidate、model、release、threshold、label、split 或环境。

## 本阶段修改文件

- 实现：`camera_component.py`、`camera_dataset.py`、`camera_collection.py`；
- 回归：`test_wandering_camera_adapter.py`、`test_wandering_camera_dataset.py`、`test_wandering_camera_collection.py`；
- 状态与证据：本报告、任务表、心理健康模块 README、技术文档2完成记录、根 README 与文档索引。

`camera_adapter.py` 最终未修改，保留 Step8 冻结 SHA；PREP/F 原报告未覆盖。

## 下一步

软件线下一项是零视频的 `M0-CAM-PORTABLE` 运行资产与 preflight；本报告不实现或验证该阶段。真实数据线继续等待负责人/R5 提供 approved、active、unexpired 的 C0 receipt、授权外部视频和匿名 development collection，随后 R2 才可通过 PREP/F/F2 链形成正式 C1 tracking/sidecar/summary。只有 PORTABLE 通过且 C0+C1 合法、scope/hash 校验通过后，才可启动一次无标签 M0-CAM-D engineering smoke；有标签评估还需 C2+C3。

精确命令、测试结果和攻击覆盖见 [VERIFICATION.md](VERIFICATION.md)。
