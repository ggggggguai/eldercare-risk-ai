# 数据目录

本目录只保存算法研发相关数据，不保存系统业务数据。

```text
data/
  external/              公开数据或第三方数据说明
  raw_videos/            原始视频，按模块分开
  annotations/           标注文件，按模块分开
  processed/             关键点、特征、中间结果
  splits/                训练、验证、测试、盲测划分
  manifests/             数据索引和版本清单
  documents/             授权、采集记录、脱敏说明
```

跌倒标注分为两个不可混用的层级：`annotations/fall_risk/*_labels.jsonl` 是 v2 根标签与发布候选，`*_labels_v3.jsonl` 和 `splits/fall_risk/training_labels_v3/` 是由 v2 与受审决策确定性生成的模型训练视图。当前数量、hash 和门禁状态必须读取对应机器报告，不能从旧目录或历史文档推断。

`processed/` 和本地实验目录可包含可再生成的大文件或模型产物；`.pt`、`.joblib` 等本地模型文件默认不进入版本库。`external/` 中存在本地副本不等于具备公开再分发授权，使用范围以 manifest、来源决定和数据审计为准。

含人的视频、音频、文本、问卷和推理结果都按敏感数据处理。对外展示优先使用骨架、统计特征、脱敏截图或聚合指标。
