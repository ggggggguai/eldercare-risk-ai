# 隔离标注

本目录中的记录不进入训练、评估或正式标签统计。

`le2i_home_02_missing_source_action_labels.jsonl` 和 `le2i_home_02_missing_source_event_labels.jsonl` 于 2026-07-15 从正式标签中隔离：它们引用的 `Home_02/video (1)-(30).avi` 不存在于本地 LE2I 原始集，且现有 CVAT 导出无法给出可验证的重映射关系。

若找回对应 CVAT 导出或原始视频，先按任务元数据验证 `task_id`、源文件名和视频内容，再重新导入；不得将编号直接加 30。
