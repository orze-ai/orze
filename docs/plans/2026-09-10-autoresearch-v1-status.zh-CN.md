# V1 实施账本

方案：[冻结的第一版实施方案](2026-09-10-autoresearch-v1.zh-CN.md)。只在证据支持的范围内标记完成。

| 项目 | 状态 | 证据与剩余工作 |
|---|---|---|
| V1-00 | 基线已核实 | core 起点 1279 passed / 6 optional Pro skips；Pro 离线 304 passed，授权边界使用测试替身 |
| V1-01A：champion 恢复 | 已修复、机制已验证 | 修复 `c9f01bf`；11 个目标红测转绿；真实启动回归；最终 core 1302 passed / 6 optional Pro skips；另以两仓源码运行相关跨仓测试 30 passed |
| V1-01B：研究上下文资格 | 已修复、机制已验证 | Pro `1477bf8`；48 排名、4 统计、15 独立复核红测转绿；最终 Pro 371 passed；core 跨仓相关 30 passed。手工记录不混排，完整项目配置优先 |
| V1-01C：声明式排序 | 已修复、机制已验证 | Core `14065b5` / Pro `139d404`；真实 report/rebuild/sweep/研究上下文排序一致；最终 core 1344 passed / 6 optional Pro skips，Pro 377 passed，跨仓相关 30 passed |
| V1-01D：摘要资格与真实交付 | 已修复、机制已验证 | Core `8802d03` / Pro `d0d27f1`；真实 producer→role command→CLI→prompt，消费时重新核实，历史笔记不覆盖；最终 core 1368 passed / 6 optional Pro skips，Pro 394 passed，跨仓相关 30 passed |
| V1-01E：完成记录与通知边界 | 已修复、机制已验证 | Core `f871eaf`；31 冻结测试在旧模块 29 red / 2 pass，当前全绿；通知关闭仍更新合格记录，observer 不授予生命周期；最终 core 1399 passed / 6 optional Pro skips，Pro 394 passed，跨仓相关 30 passed |
| V1-01F：晋升资格与显式异常策略 | 已修复、机制已验证 | Core `2ef6b5b`；58 冻结行为测试旧模块 48 red / 10 pass，当前全绿，另 3 真实存储回归；最终 core 1460 passed / 6 optional Pro skips，Pro 394 passed，跨仓相关 30 passed |
| V1-01G1：评估资格与真实完成 | 已修复、机制已验证 | Core `637ef59`；87 冻结测试旧模块 69 red / 18 pass，当前全绿；既有输出/同步/异步共用契约，None 不再被猜作成功；最终 core 1547 passed / 6 optional Pro skips，Pro 394 passed，跨仓相关 30 passed |
| V1-01 整体 | 进行中 | 冠军选择与进步/重复观察语义、无 finished 时撤销、report/admin 与旧 verifier 消费者仍待收口；G1 不等于 eval-only retry 已完成 |
| V1-02 至 V1-07 | 未验收完成 | 后续按方案逐项核实与修复；已有主干能力也必须提供对应验收证据 |

[V1-01A 机器可读证据](../evidence/2026-09-10-v1-01a-champion-recovery.json)包含基线、修复提交、红测内容哈希、命令、退出码、通过/跳过数量和适用边界。方案定稿、代码推送、机制验证、真实研究收益是四种不同状态。

V1-01B 的完整证据保存在 Pro 仓 `docs/evidence/2026-09-10-v1-01b-research-ranking.json`。其资格证明针对指标样本与最佳结果 ID，不意味着 `error_analysis.json` 内容或配置标签的执行来源已全部验证。

[V1-01C 机器可读证据](../evidence/2026-09-10-v1-01c-objective-ordering.json)记录两仓成对提交、各组行为红测、最终全量与跨仓回归。源码版本更新及最低依赖声明不是发布；没有运行真实训练或付费 provider。

[V1-01D 机器可读证据](../evidence/2026-09-10-v1-01d-digest-delivery.json)记录摘要交付的行为红测、最终全量与边界检查。自动摘要已接通不等于所有旧 SOP 笔记流已迁移，也不意味着整个 prompt 预算或全局原子快照已解决。

[V1-01E 机器可读证据](../evidence/2026-09-10-v1-01e-notification-authority.json)提供固定旧模块重现命令、原封不动的测试哈希及最终全量结果。合格完成记账与通知开关已分离，但不声称重复投递、过期尝试或科学进步判定已完成。

[V1-01F 机器可读证据](../evidence/2026-09-10-v1-01f-promotion-policy.json)记录显式策略、共享资格及 SQLite 并发/故障验证；[兼容说明](../champion-policy.md)明确默认值变化、旧历史保留和复验边界。异常检测不是科学判决或收益证明。

[V1-01G1 机器可读证据](../evidence/2026-09-10-v1-01g1-evaluation-contract.json)记录三入口、调度保留、失败来源/缓存、封存与回执故障的冻结测试、固定旧模块重放命令和全量结果；[评估契约](../evaluation-contract.md)明确无目标任务、metrics 原位评估兼容及已知恢复边界。没有真实 GPU/LLM 或科研效率收益声明。
