# 徘徊步骤 4 预处理人工图审

状态：`human_review_passed`

本文件只记录人工复核状态。自动测试和生成诊断图不能替代人工签字；本次状态由项目用户完成全部指定图审后更新。

## 人工确认结果

- [x] 48 条 ready train 固定样本均同时核对来源轨迹、T=80 来源视图、shape 视图、曲率、反向点和回访连线。
- [x] 15 条 SmartCare train 短轨迹均保持原路径与 `too_few_valid_points`，未补点训练、删除或重分组。
- [x] WanderingPatterns 没有伪造 image-normalized 画布，图中没有 bbox 高度补偿内容。
- [x] 记录复核人、日期、异常 sample ID 与最终结论。

复核人：项目用户，通过当前 Codex 任务确认

复核日期：2026-08-03（Asia/Shanghai）

异常 sample ID：无

结论：通过；用户确认全部 48 条 ready train 固定样本和 15 条 unavailable 短轨迹均未发现问题。此结论仅关闭步骤 4 的预处理人工图审门禁，不构成模型效果、摄像头域、真实老人场景或临床有效性验证。

## 固定 ready train ID

- `smartcare/normal`: `smartcare_177ad988979694256cd7bda2`, `smartcare_3a85a84e98c865958ab7a1b1`, `smartcare_4a975fd862eb4ffa62cb1851`, `smartcare_7c2883df05073ce1a0cffaac`, `smartcare_860499e9464047f1fcae0eae`, `smartcare_cbdc7fe5ac4d9680d2a5f0cc`, `smartcare_df036d6b2575855f94f8f61d`, `smartcare_e8aa579f1c5707f4ec69272d`
- `smartcare/wandering_like`: `smartcare_1546765218074e5195f8669f`, `smartcare_9b94f780c5e40cac299c3700`, `smartcare_a5a52a98083ab18e1004a74e`, `smartcare_cb33f36cde90150a1fb61c5a`, `smartcare_d9ebfd912171165deecf0c66`, `smartcare_e3b2e812959911c602859c58`, `smartcare_e4646591788f19e34f72d8ee`, `smartcare_ed1ad34eece2a0c1cf9e3293`
- `wandering_patterns/direct`: `wandering_patterns_04eff340b7f7788ab7eff461`, `wandering_patterns_3de203012eccdf10d858c826`, `wandering_patterns_400af71fe97cbd68e96e9e33`, `wandering_patterns_559a84869154cf17410ed45f`, `wandering_patterns_5693cc79098c11831f78a41f`, `wandering_patterns_57c7213385608a3589f6c3c5`, `wandering_patterns_776985de8301115386a4f314`, `wandering_patterns_9efe4117aa89a18505c8cf59`
- `wandering_patterns/lapping`: `wandering_patterns_1e4d59d99cb760fbf8073694`, `wandering_patterns_3abe08026cf01c8fa9bbea57`, `wandering_patterns_5076a735a6ad106e453fad26`, `wandering_patterns_adba67256e131a6426e51dc0`, `wandering_patterns_b2a56a30730fb2bc6aae5fef`, `wandering_patterns_d151fad30c1ba8aac9b6abd2`, `wandering_patterns_d70c9598ddd7e1fd8649da9c`, `wandering_patterns_eb7fec3452f6586085437def`
- `wandering_patterns/pacing`: `wandering_patterns_15f2ba0ee9c69328cf989936`, `wandering_patterns_2d2ebdf505fb0c2efd55a35a`, `wandering_patterns_3e110f0f4f990b3e514076c9`, `wandering_patterns_43841132c425da238aed11ef`, `wandering_patterns_81cf0942fae20bb20d8cc886`, `wandering_patterns_bb836c0dfdc55b4330b8f83b`, `wandering_patterns_e8a98536b690292cb047e595`, `wandering_patterns_f8dca5d0e683ca9010e5751f`
- `wandering_patterns/random`: `wandering_patterns_0ff63d093ccdd631941e9951`, `wandering_patterns_16eede2c44c00c226bbf043a`, `wandering_patterns_19d1811a2bdf633b84dd0f35`, `wandering_patterns_7d5490e400455771c62bedfd`, `wandering_patterns_9779e576ca214e8ea0294a43`, `wandering_patterns_c0b00107a07b6b096b1cf6b4`, `wandering_patterns_cbedf9757da840273c1b8ad9`, `wandering_patterns_cc0ae07858d237f99755fc37`

## 全部 unavailable 短轨迹 ID

- `smartcare_16d08d4b3b6d7f348b2d89f8`, `smartcare_1c7387d8922ea4421a100e74`, `smartcare_22a88a75ca2f8cc4882d77ae`, `smartcare_2cb716af8bc7baa6b908d96c`, `smartcare_4604c8838ec3481f5780c4ff`, `smartcare_743ec26cad9cc800b64ecdcc`, `smartcare_7b1fa02866da691d91047ffc`, `smartcare_81a1868df2bf51e6fcd01abb`, `smartcare_8dadb97ab4a3c36134480aec`, `smartcare_91cefb4d6c6229e1b16786ef`, `smartcare_9c8a47117e79913ca27d1a5a`, `smartcare_a6d9f9807493d159c8c3f4d0`, `smartcare_baf601c83e42d45d0b0118a8`, `smartcare_c85193405156a787deaf7c62`, `smartcare_cd79fcef0e97a338cddce756`
