# 项目文档索引

本文档是项目资料入口。根目录只保留项目入口、环境和协作规则；研发计划、规范、接口、评审和参考材料统一放在 `docs/` 下。

## 推荐阅读顺序

1. `../README.md`：项目定位、工程结构和运行环境。
2. `architecture/算法工程骨架.md`：算法工程边界和模块划分。
3. `interfaces/算法事件输出接口.md`：算法模块统一输出的事件 JSON。
4. `interfaces/跌倒风险算法服务后端对接说明.md`：风险回调契约及后端调用算法服务的方法。
5. `modules/fall_risk/guides/跌倒风险算法协作开发指南.md`：跌倒风险方向的开发入口。
6. `modules/fall_risk/plans/跌倒风险算法研发计划.md`：跌倒风险算法主计划。
7. `modules/mental_health/README.md`：心理健康风险模块当前边界。
8. `tasks/待解决问题.md`：仍需落实或验证的问题。

## 目录分布

```text
docs/
  architecture/          工程边界、架构和模块划分
  interfaces/            对外 JSON 事件接口
  modules/
    fall_risk/
      guides/            协作指南和开发入口
      plans/             研发计划和子模块执行计划
      data/              数据集、采集、标注和训练方案
    mental_health/       心理健康风险模块文档
  reference/competition/ 比赛方案、外部参考材料
  reviews/               技术审查和竞争力评估报告
  tasks/                 待解决问题和后续行动清单
```

## 跌倒风险核心文档

- `modules/fall_risk/README.md`：模块说明、运行命令和字段契约。
- `modules/fall_risk/guides/跌倒风险算法协作开发指南.md`：新同事或新任务的第一入口。
- `modules/fall_risk/plans/跌倒风险算法研发计划.md`：完整算法路线、研发阶段和评价指标。
- `modules/fall_risk/data/数据集标注规范.md`：每个数据集怎么标、谁来标、产出什么统一标注文件。
- `modules/fall_risk/data/跌倒风险标签字典.md`：动作级、事件级、风险级标签和判定规则。

## 根目录保留文件

- `README.md`：项目总入口。
- `AGENTS.md`：协作代理和自动化工具规则。
- `environment.yml`：标准 conda 环境定义。
- `environment-reference.txt`：本机已验证环境记录。
