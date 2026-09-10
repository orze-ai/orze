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
| V1-01G2 / V1-05A：显式补评 | 已实现、机制已验证 | Core `1f375c4`；73 新机制验收与 G1 的 87 测试合计 160 passed；另有 5 个草稿边界红测及 1 个旧发布函数行为红测可重放；最终 core 1620 passed / 6 optional Pro skips，Pro 394 passed，跨仓相关 30 passed |
| V1-01H1：选择不等于局部改善 | 已修复、机制已验证 | Core `858dc31`；31 冻结测试旧模块 24 red / 7 pass，当前全绿；idle tick 重验/撤销，stable ID、缺测及失格替换不冒充改善；最终 core 1651 passed / 6 optional Pro skips，Pro 394 passed，跨仓相关 30 passed |
| V1-01I1：native report 统一资格 | 已修复、机制已验证 | Core `7b76484`；35 冻结测试旧模块 26 行为 red + 4 新字段验收失败 / 5 pass，当前全绿；完整 cfg、逐行 identity、缓存不授予资格，旧 offline API 明确未验证；最终 core 1686 passed / 6 optional Pro skips，Pro 394 passed，跨仓相关 30 passed |
| V1-01I2：只读 catalog 与任务状态显示 | 已修复、机制已验证 | Core `b5db211`；29 公共入口测试旧模块 25 行为 red + 1 新字段失败 / 3 pass，另 6 旧发布器 red、5 草稿边界 red 与 23 新机制验收，共 63 passed；最终 core 1749 passed / 6 optional Pro skips，Pro 394 passed，跨仓相关 30 passed |
| V1-01J1：显式研究节奏的证据资格 | 已修复、机制已验证 | Pro `efa3329`；16 公共调度测试旧模块 15 行为 red / 1 pass，另 18 新机制/控制验收，共 34 passed；最终 core 1749 passed / 6 optional Pro skips，Pro 428 passed，跨仓相关 30 passed |
| V1-01I3：lineage 观察者不补审 | 已修复、机制已验证 | Core `781fc31`；18 冻结公共入口测试旧模块 14 行为 red / 4 pass，另 19 新 API 边界，共 37 passed；最终 core 1786 passed / 6 optional Pro skips，Pro 428 passed，跨仓相关 30 passed |
| V1-01J2：director 状态、目标与动作接线 | 已修复、机制已验证 | Pro `6f3915c`；50 冻结公共测试完整旧源码隔离重放 46 行为 red / 4 pass，另 10 草稿行为 red、12 新机制/兼容验收，共 72 passed；最终 core 1786 passed / 6 optional Pro skips，Pro 500 passed，跨仓相关 30 passed |
| V1-01K1：显式、版本绑定的队列审核 | 已修复、机制已验证 | Core `5fe292b` / Pro `9e7df19`；13 旧行为 red、2 旧控制、8 草稿行为 red 与 89 新机制验收，共 112 项；最终 core 1847 passed / 7 optional Pro skips、Pro 550 passed、真实跨仓 31 passed；registry optional-dependency 测试声明修订保留原快照并重放，非产品红测 |
| V1-01 整体 | 进行中 | 历史修订/重复观察/重启计数、skills/thinker、其他消费者的阶段一致性仍待收口；J1 仅关闭 research 显式节奏，J2 不代表通用依赖和全部 director 资源安全已完成，K1 不代表统一审核预算或科学判断已完成 |
| V1-02 至 V1-07 | 未验收完成 | 后续按方案逐项核实与修复；已有主干能力也必须提供对应验收证据 |

[V1-01A 机器可读证据](../evidence/2026-09-10-v1-01a-champion-recovery.json)包含基线、修复提交、红测内容哈希、命令、退出码、通过/跳过数量和适用边界。方案定稿、代码推送、机制验证、真实研究收益是四种不同状态。

V1-01B 的完整证据保存在 Pro 仓 `docs/evidence/2026-09-10-v1-01b-research-ranking.json`。其资格证明针对指标样本与最佳结果 ID，不意味着 `error_analysis.json` 内容或配置标签的执行来源已全部验证。

[V1-01C 机器可读证据](../evidence/2026-09-10-v1-01c-objective-ordering.json)记录两仓成对提交、各组行为红测、最终全量与跨仓回归。源码版本更新及最低依赖声明不是发布；没有运行真实训练或付费 provider。

[V1-01D 机器可读证据](../evidence/2026-09-10-v1-01d-digest-delivery.json)记录摘要交付的行为红测、最终全量与边界检查。自动摘要已接通不等于所有旧 SOP 笔记流已迁移，也不意味着整个 prompt 预算或全局原子快照已解决。

[V1-01E 机器可读证据](../evidence/2026-09-10-v1-01e-notification-authority.json)提供固定旧模块重现命令、原封不动的测试哈希及最终全量结果。合格完成记账与通知开关已分离，但不声称重复投递、过期尝试或科学进步判定已完成。

[V1-01F 机器可读证据](../evidence/2026-09-10-v1-01f-promotion-policy.json)记录显式策略、共享资格及 SQLite 并发/故障验证；[兼容说明](../champion-policy.md)明确默认值变化、旧历史保留和复验边界。异常检测不是科学判决或收益证明。

[V1-01G1 机器可读证据](../evidence/2026-09-10-v1-01g1-evaluation-contract.json)记录三入口、调度保留、失败来源/缓存、封存与回执故障的冻结测试、固定旧模块重放命令和全量结果；[评估契约](../evaluation-contract.md)明确无目标任务、metrics 原位评估兼容及已知恢复边界。没有真实 GPU/LLM 或科研效率收益声明。

[V1-01G2 机器可读证据](../evidence/2026-09-10-v1-01g2-evaluation-retry.json)记录显式补评的状态事务、文件恢复、重启调度、真实 CLI 和 benchmark look/nonce 验收；[使用与边界](../evaluation-retry.md)说明归档位置和拒绝条件。此项是紧接 G1 提前收口的 V1-05A 依赖切片，不代表跳过 V1-02 至 V1-04，也不代表整个 V1-05 完成。新增 API 的缺失检查与真正行为红测分开记录；独立 generation、复验、过期 attempt fencing、预约后未启动的预算处理和调度公平性仍未验收。

[V1-01H1 机器可读证据](../evidence/2026-09-10-v1-01h1-objective-progress.json)提供公共通知路径的冻结红绿重放、真实 guard history 边界与最终全量结果；[语义与限制](../objective-progress.md)区分当前局部比较、历史修订与科学判决。空闲 tick 重验增加读 I/O，不声称效率净收益；旧 mtime 恢复和逐 host 状态尚未改为唯一 observation 计数。

[V1-01I1 机器可读证据](../evidence/2026-09-10-v1-01i1-native-report.json)记录 native authority、cache/identity 与旧离线兼容的冻结红绿测试和测试前提更正；[范围说明](../report-authority.md)区分原生排名、未验证离线展示及 Pipeline 覆盖。I1 验收当时，配置 DB 的直接 report 入口不会自动迁移，但旧 CLI 构造器仍待修复；不能把当时的 API 验收当作 CLI 非迁移读或 admin 队列状态已完成。

[V1-01I2 机器可读证据](../evidence/2026-09-10-v1-01i2-observer-catalog.json)补上 I1 当时未完成的真实 CLI 非迁移读取、完整目录和 admin producer→API 任务状态显示。原始公开入口红测、旧发布器在新 CLI 下的 red、未提交 CLI 草稿的控制根目录 red 分别登记，未混称旧版本行为。63 项最终测试哈希、单文件写后核实/故障恢复及两仓全量命令均可重放；[观察入口契约](../observer-catalog.md)明确默认路径兼容、配置读取预算、缓存 TTL 和非原子边界。完整 V1 仍在执行，不将只读观察修复冒充持久 attempt/observation 或研究效率收益。

V1-01J1 的完整证据保存在 Pro 仓 `docs/evidence/2026-09-10-v1-01j1-evidence-cadence.json`，语义见该仓 `docs/evidence-cadence.md`。真实调度在锁边界验证完整资格、声明源/方向、可比较 distinct task ID、缺失 secondary、逐主机状态恢复及停用/backoff；新基线不读写旧 `_best_metric_*`。这是一项默认关闭的操作性节奏策略，不是科学收敛、独立样本或共享 observation/attempt 账本，其他旧文本/计数消费者尚未统一。

[V1-01I3 机器可读证据](../evidence/2026-09-10-v1-01i3-lineage-observers.json)补充审查 director 时发现的共享资格读写混用：原观察调用会重扫 manifest、写新收据或补建 compute receipt 目录。冻结公共入口红测真实记录文件树/字节及审计副作用，修复保留模型哈希、终态和 benchmark 绑定；生成路径仍由训练启动/封存负责。[语义与边界](../lineage-observer-boundary.md)明确 I2 当时未覆盖这个 opt-in 缺口、旧 offline warm cache 和元数据信任模型。该修正切片在 J2 前收口，不代表 director、V1-01 整体或 V1 已完成。

V1-01J2 的完整证据保存在 Pro 仓 `docs/evidence/2026-09-10-v1-01j2-director-objectives.json`，语义见该仓 `docs/director-evidence.md`。完整旧 Pro worktree 实际提供被复制的 director 源码，不把当前新脚本伪装为旧复制路径；两份草稿快照分别重放控制目录/拒绝不 ACK/诊断链接与阶段交集缺口。无 objective 仍可读取合法排队工作，legacy 0 保持未知；自定义 PID/sentinel 路径一致，拒绝后可恢复重试。未具可信产物绑定的检测源切换已明确禁用，不宣称通用依赖已实现；真实进程 ownership/停止确认、持久 attempt/ACK 和全局原子性仍属后续条目。

[V1-01K1 机器可读证据](../evidence/2026-09-10-v1-01k1-queue-review.json)在两仓各保留一份，记录固定旧 runner/插件行为重放、三份草稿快照、完整测试哈希、入口闭环和最终全量。[队列审核契约](../queue-review.md)说明显式 opt-in、canonical registry、只在内存传递配置、默认不连带启用其他 bundled 自动化，以及任务 revision/当前 queued 一致性/事务收据的执行权边界。相同配置/seed 和规则笔记不再自动删任务，审核不改 metrics。审核收据不等于科学结论；provider 调用预算、跨文件遥测原子性、策略版本迁移和独立 attempt/observation 身份仍未完成。
