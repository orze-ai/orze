# V1 实施账本

方案：[冻结的第一版实施方案](2026-09-10-autoresearch-v1.zh-CN.md)。只在证据支持的范围内标记完成。

| 项目 | 状态 | 证据与剩余工作 |
|---|---|---|
| V1-00 | 基线已核实 | core 起点 1279 passed / 6 optional Pro skips；Pro 离线 304 passed，授权边界使用测试替身 |
| V1-01A：champion 恢复 | 已修复、机制已验证 | 修复 `c9f01bf`；11 个目标红测转绿；真实启动回归；最终 core 1302 passed / 6 optional Pro skips；另以两仓源码运行相关跨仓测试 30 passed |
| V1-01B：研究上下文资格 | 已修复、机制已验证 | Pro `1477bf8`；48 排名、4 统计、15 独立复核红测转绿；最终 Pro 371 passed；core 跨仓相关 30 passed。手工记录不混排，完整项目配置优先 |
| V1-01C：声明式排序 | 已修复、机制已验证 | Core `14065b5` / Pro `139d404`；真实 report/rebuild/sweep/研究上下文排序一致；最终 core 1344 passed / 6 optional Pro skips，Pro 377 passed，跨仓相关 30 passed |
| V1-01 整体 | 进行中 | digest 写入/读取路径与生命周期资格、champion_guard 的代理指标/策略、通知开关与 plateau 状态耦合、未资格通知回退及 eval source 校验仍待收口 |
| V1-02 至 V1-07 | 未验收完成 | 后续按方案逐项核实与修复；已有主干能力也必须提供对应验收证据 |

[V1-01A 机器可读证据](../evidence/2026-09-10-v1-01a-champion-recovery.json)包含基线、修复提交、红测内容哈希、命令、退出码、通过/跳过数量和适用边界。方案定稿、代码推送、机制验证、真实研究收益是四种不同状态。

V1-01B 的完整证据保存在 Pro 仓 `docs/evidence/2026-09-10-v1-01b-research-ranking.json`。其资格证明针对指标样本与最佳结果 ID，不意味着 `error_analysis.json` 内容或配置标签的执行来源已全部验证。

[V1-01C 机器可读证据](../evidence/2026-09-10-v1-01c-objective-ordering.json)记录两仓成对提交、各组行为红测、最终全量与跨仓回归。源码版本更新及最低依赖声明不是发布；没有运行真实训练或付费 provider。
