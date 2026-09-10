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
| V1-01L1：thinker 观察与启动确认 | 已修复、机制已验证 | Pro `4ed67d1`；21 冻结公共行为旧源码重放 16 red / 5 pass，38 新机制及 1 草稿状态边界 red，共 60 passed；最终 Pro 610 passed，core 1847 passed / 7 optional Pro skips，真实跨仓 31 passed |
| V1-01L2：逐技能激活与实际提示词交付 | 已修复、机制已验证 | Core `8da25dc` / Pro `b7cd5be`；22 冻结公共行为旧源码 16 red / 6 pass，另 2 草稿 I/O 边界 red 与 76 新机制，共 100 项；最终 Core 1876 passed / 7 optional Pro skips、Pro 681 passed，真实跨仓 60 passed；全部 60 项 L1 冻结回归也通过 |
| V1-01 整体 | 进行中 | 历史修订/重复观察/重启计数仍待收口；共享完成阶段一致性由 V1-02B 补齐。J1/L1/L2 仅关闭相应角色/技能的显式证据节奏，J2 不代表通用依赖和全部 director 资源安全已完成，K1 不代表统一审核预算或科学判断已完成 |
| V1-02A：持久触发交付 | 已实现、机制已验证 | Core `105c58e` / Pro `99f3489`；3 旧公共行为 red、2 草稿绑定 red 与 80 新机制，共 85 passed；最终 Core 1932 passed / 7 optional Pro skips、Pro 710 passed、真实跨仓 60 passed；旧 workflow fixture 显式版本化保留原快照，最终 v3 旧源码仍重现 2 red |
| V1-02B：共享完成阶段一致性 | 已修复、机制已验证 | Core `3c92571` / Pro 验收 `4672fbb`；33 旧行为 red、1 新 helper 草稿 red、1 新并发事务机制与 18 兼容控制，共 53 passed；最终 Core 1975 passed / 7 optional Pro skips、Pro 720 passed、真实跨仓 60 passed |
| V1-02C：提案交接与不可覆盖入队 | 已修复、机制已验证 | Core `d869acf` / Pro 验收 `598905e`；22 旧行为 red、3 草稿 red、31 新机制与 3 兼容控制，共 59 passed；最终 Core 2031 passed / 7 optional Pro skips、Pro 723 passed、真实跨仓 60 passed。两套 source 测试夹具显式版本化，保留原快照 |
| V1-02D1：执行停止确认 | 已修复、机制已验证 | Core `7c33787` / Pro 契约 `72158a5`；25 旧行为 red、15 旧兼容控制、4 草稿 red 与 33 新机制验收，共 77 passed；最终 Core 2108 passed / 7 optional Pro skips、Pro 723 passed、真实跨仓 60 passed。四份旧 fixture 完整快照保留，只迁移明确停止成功的测试替身，不修改业务断言 |
| V1-02D2：原生 attempt 与发布边界 | 已实现、机制已验证 | Core `caecc61` / Pro 契约 `3e8f724` 已推送并读回精确远端 ref；冻结源码最终 Core 2524 passed / 7 optional Pro skips、Pro 723 passed、真实配对 60 passed；474/160 个源码、测试及构建声明文件前后 SHA 完全相同。[合同](../execution-attempt-authority.md)及[机器证据](../evidence/2026-09-10-v1-02d2-attempt-authority.json)区分旧开发提交/草稿红测、新机制及显式 fixture 迁移，不累加重叠 target。原生 adoption、repair worker、原生 resume admission、独立 observation 和产物隔离仍未完成 |
| V1-02 整体 | 进行中 | A 关闭触发交付与保守启动恢复，B 关闭共享已记录阶段的完成资格，C 关闭原生提案源的并发交接/不可覆盖 admission，D1 关闭已接线的强制停止确认与持久停止 HOLD；D2 关闭已接线当前 attempt 的实际发布与消费边界。原生重启 adoption、未知结果裁定、完整跨崩溃交接和科学任务独立 observation 仍待收口 |
| V1-03A1：共享配额与完成记账 | 已修复、机制已验证 | Pro `6d22d7e` / Core 契约 `dcba7b5` 已推送并核实；21 个新目标测试为 11 旧行为 red、4 旧控制、6 新机制。最终 Pro 744 passed、真实配对 60 passed；Core 474 文件与 D2 完全一致，明确复用 D2 的 2524 passed / 7 optional Pro skips，而非声称新跑。[证据](../evidence/2026-09-10-v1-03a1-shared-quota.json)与[契约](../shared-quota-completion.md)保留旧混合状态一次迁移及同主机范围 |
| V1-03B：持久需求与真实消费 | 已修复、机制已验证 | Core `3ea04e6` / Pro `19f9bf5` 已推送并读回精确远端 ref；47 个新用例区分 13 旧行为 red、3 旧控制、22 新 API 机制、7 草稿 red、2 草稿控制。最终 Core 2552 passed / 7 optional Pro skips、Pro 763 passed、真实配对 60 passed；477/172 个源码、测试与构建文件前后 SHA 相同。[证据](../evidence/2026-09-10-v1-03b-persistent-demand.json)与[契约](../persistent-research-demand.md)覆盖只读持久积压、实际门控及最终提示词；不等于执行就绪或原子容量预约 |
| V1-03C：预算并发准入与故障 | 已修复、机制已验证 | Pro `1273a02` / Core 契约 `566f092` 已推送并核实；26 新用例为 8 旧行为 red、3 旧控制、2 草稿 red、1 草稿控制、12 新机制。最终 Pro 789 passed、真实配对 60 passed；Core 477 文件与 B 一致，明确复用 B 全量 2552 passed / 7 optional Pro skips；Pro 178 文件前后 SHA 相同。[证据](../evidence/2026-09-10-v1-03c-budget-admission.json)及[契约](../research-budget-admission.md)保留实际扣费、不自动接管未知 owner，不声称 provider 账单、掉电原子性或科研收益 |
| V1-03D：原生结果分类与产出归属 | 已修复、机制已验证 | Core `85598d4` / Pro `59bc0e8` 已推送并核实；31 新用例区分 5 旧行为 red、4 旧控制、22 新机制，固定五模块重放仍为 5 failed / 4 passed。最终 Core 2565 passed / 7 optional Pro skips、Pro 807 passed、真实配对 60 passed；481/184 文件前后 SHA 相同。[证据](../evidence/2026-09-10-v1-03d-result-classification.json)及[契约](../native-research-results.md)覆盖真实 CLI→本批提案→Core 终态→Pro usage，accepted 不等于 Lake admission、执行或测量有效 |
| V1-03E：可选角色预设 | 已修复、机制已验证 | Core `7741e23` / Pro `64684d8` 已推送并读回；46 新用例为 15 旧公共行为 red、10 旧控制、21 新机制。冻结源码最终 Core 2588 passed / 7 optional Pro skips、Pro 830 passed、真实配对 60 passed；484/188 文件 SHA 前后一致。[证据](../evidence/2026-09-10-v1-03e-optional-presets.json)及[契约](../optional-role-presets.md)覆盖真实配置→派发及跨项目隔离；两份旧测试显式迁移，42 条原 assert AST 相同且保留原快照 |
| V1-03F：Provider 返回分类 | 已修复、机制已验证 | Core 契约 `f3a6612` / Pro `28ac911` 已推送并读回；25 新用例区分 9 旧公共行为 red、3 新稳定原因要求失败、3 旧控制、4 新逐调用机制及 6 草稿 red。固定旧两模块重放为 12 failed / 3 passed。最终 Pro 855 passed、真实配对 60 passed；Core 484 文件与 E 完全相同，明确复用 E 的 2588 passed / 7 optional Pro skips，Pro 194 文件全量前后相同。[证据](../evidence/2026-09-10-v1-03f-provider-outcomes.json)及[契约](../provider-outcomes.md)连接真实响应解析→cycle→CLI→持久终态→usage；缺终止元数据仍为 legacy unknown |
| V1-03 整体 | 本版机制已验收 | A1 关闭同控制器共享配额记账，B 关闭持久需求消费者，C 关闭预算准入边界，D 关闭本批提案结果与共享 inbox 误归属，E 关闭默认凭据/GOAL 扩组和隐式停滞触发，F 关闭已支持结构化 provider 拒绝/截断/完整空响应与传输失败的实际分类。按主机保存不等于跨主机共享配额；角色预设不等于热撤销或通用策略闭环；离线机制不是线上 provider 验收或研究收益，整个 V1 尚未完成 |
| V1-04A：方法状态、来源与默认策略 | 已修复、机制已验证 | Core 契约 `588f606` / Pro `d1f17e7` 已推送并读回；43 新用例为 22 旧公共行为 red、7 旧控制、11 新机制、3 草稿 red。固定旧四模块重放 21 failed / 7 passed / 1 新机制排除，另 mixed-key 旧行为 1 failed。最终 Pro 898 passed、真实配对 60 passed；Core 484 文件不变，明确复用 E 的 2588 passed / 7 optional Pro skips，Pro 205 文件前后相同。[证据](../evidence/2026-09-10-v1-04a-method-context.json)及[契约](../method-context.md)保留真实原生交付、候选失效历史与严格来源/验证区分。旧 70% 分支已不可达，只作死代码清理 |
| V1-04B：整份提示预算与实际输入清单 | 已实现、机制已验证 | Core `ec2f981` / Pro `9c82209` 已推送并读回；57 个新增验收区分 1 旧公共行为 red、2 旧控制、18 新配置要求失败、34 新机制及 2 草稿 red。固定旧两模块重放 1 failed / 2 passed，最终同测 3 passed。最终 Core 2606 passed / 7 optional Pro skips（新跑全量）、Pro 937 passed、真实配对 60 passed；486/213 文件前后 SHA 相同。[证据](../evidence/2026-09-10-v1-04b-prompt-budget.json)与[契约](../prompt-budget.md)覆盖完整 UTF-8 提示、必需原文、范围/哈希、省略原因、真实原生派发前发布与失败阻断；19 条旧规则测试只迁移读取计数位置，18 个 assert AST 完全相同且保留原快照 |
| V1-04 整体 | 本版机制已验收 | A 关闭方法笔记的退役配方、错误 proven 标签、max-mtime 缓存及默认强制家族/ML 策略；B 关闭完整 prompt 预算、必需契约完整性、可选源清单及 provider 入参绑定。原生 prepared 清单不是实际 HTTP 发送或科研收益证明；生成上下文底层的全部旧 I/O、全局原子快照及独立科学 observation 不属于已证明的提示字节上限。整个 V1 尚未完成 |
| V1-05B1：声明产物、物理快照与 occurrence 登记 | 已实现、机制已验证 | Core `64c1a04` 已推送并读回；85 个新增验收为 18 个旧校验器不满足的新配置要求、67 个新机制，不冒称 85 个历史运行缺陷。最终新跑 Core 2691 passed / 7 optional Pro skips、Pro 937 passed（虽源码未改，仍对新 Core 重跑）、配对 60 passed；494/213 文件前后 SHA 一致。[证据](../evidence/2026-09-10-v1-05b1-artifact-publication.json)与[契约](../artifact-records.md)覆盖启动前绑定、独立 inode、锁外大文件复制、事务内完整登记/终态、失败 HOLD。独立并发夹具因确定性暂存契约迁移，原文保留且 57 个 assert AST 不变。尚无评估隔离、observation、显式复验或真实 CPU 闭环；未接受暂存仍需显式处理 |
| V1-05B2：隔离评估、观察记录与绑定补评 | 已实现、机制已验证 | Core `8ff845c` 已推送并读回；101 新用例区分 28 个新协议要求失败、2 控制、63 新机制与 8 草稿缺陷，不冒称历史运行缺陷。最终新跑 Core 2792 passed / 7 optional Pro skips、Pro 937 passed、配对 60 passed；507/213 文件前后 SHA 一致。[证据](../evidence/2026-09-10-v1-05b2-observation-publication.json)及[契约](../observation-records.md)覆盖独立评估目录、固定输入/协议、0..32 observation、同事务结果登记与终态、仅评估重试；真实文件/SQLite 故障与独立复核通过。原 fixture 的 3 个安全拒绝预期错误单列，原文保留，不算产品红测。尚无 observation 排名、显式复验、重启接管或实际 CPU 闭环 |
| V1-05B3：显式同规格复验与独立执行槽 | 已实现、机制已验证 | Core `956db91` / Pro `c88161e` 已推送并读回；64 个新增验收为 4 个新保留字段要求、4 控制、48 新机制与 8 新测试草稿红；另 1 个普通启动草稿兼容缺陷由未改旧测试复现，不增加 64。最终新跑 Core 2849 passed / 7 optional Pro skips、Pro 944 passed、配对 60 passed；519/214 文件前后 SHA 一致。[证据](../evidence/2026-09-10-v1-05b3-explicit-replication.json)及[契约](../explicit-replication.md)覆盖真实 CLI→原子新 task/request→空 inbox 调度→claim/launch/terminal/artifact；同配置/seed/spec 不加盐，原 owner 不变，同请求不重置。错误配置修复后重跑全量，前一中断 epoch 不计最终证明。尚不支持新 replica 的 interrupted checkpoint resume，外置策略仍可否决，独立 occurrence 不等于统计独立 |
| V1-05 至 V1-07 | 未验收完成 | B1/B2/B3 关闭声明产物、隔离评估、独立观察、绑定补评与显式复验的本版机制；继续清理归属、恢复、消费记账及通用 CPU 闭环，不能以局部机制替代整版验收 |

[V1-02D1 机器可读证据](../evidence/2026-09-10-v1-02d1-termination-authority.json)与[执行停止契约](../execution-termination-authority.md)在两仓各保留一份。固定旧代码重放证明主进程退出被误当作停止确认、失败初始化/槽位注册释放权限，以及补评接受残留写入者产物的路径；修复要求持久请求、停止器严格 True 和整数退出码，再发布绑定请求哈希的确认。未确认停止跨对象丢失和已接线恢复入口保持 HOLD，不自动重试。四个草稿边界红测及四份兼容 fixture 原始快照均可重放。首次 start/stop 均未落盘时的崩溃/存储故障、普通自然退出的完整后代证明、stale-attempt CAS 和产物代际隔离仍未完成；仍不代表整个 V1 已完成。

[V1-02C 机器可读证据](../evidence/2026-09-10-v1-02c-proposal-handoff.json)与[提案源契约](../proposal-source-handoff.md)在两仓各保留一份。固定旧 fs/parser/IdeaLake/phase 重放与实际 Pro producer→Core consumer 重放证明追加丢失、同 ID 覆写、失败后不能 ACK、未完成 finalizer 被提前入库等缺陷。新入口只创建或精确重放，不把 config duplicate/解析失败当成删除源的权限；主文件原字节与未处理区块保留，已提交但删源失败可恢复。源 owner 不再按 60 秒年龄被抢占；磁盘错误恢复失败和二次 close 错误保留不确定 owner。marker/未知 owner 不提供自动恢复工具，其他 legacy/portfolio 写入口及科学复验语义没有被宣称已统一；仍不代表 V1 完成。

[V1-02B 机器可读证据](../evidence/2026-09-10-v1-02b-stage-agreement.json)与[阶段资格契约](../completed-stage-agreement.md)在两仓各保留一份。真实缺失阶段历史保持兼容，已存在的 NULL/未知/非终态不能再混入已完成结果；SQL 与 Python 使用精确状态值，结构/身份歧义不被 set/dict 去重掩盖。通知缓存只在自有写事务内重新检查并更新，不提交调用者尚未提交的工作。冻结测试覆盖真实 report 热缓存撤销、Pro 排名/调度、两个 SQLite 连接及故障触发器。Pipeline 计数、其他扩展阶段和非 COMPLETE 生命周期语义未扩大；本项不是科学任务身份、不可变 observation 或科研收益验收。

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

V1-01L1 的完整证据保存在 Pro 仓 `docs/evidence/2026-09-10-v1-01l1-thinker-evidence.json`，语义与限制见该仓 `docs/thinker-evidence.md`。thinker 不再数 Markdown 行或读写旧 best 字段；当前合格可比较 ID 的观察基线与启动确认分离，只有进程创建及登记成功才消费启动前 receipt。手动/周期/失败级联保留独立语义，失格或未知不冒充 plateau；陈旧 scope/reference/generation、预算/构建/登记失败不会覆盖或确认本次机会。同主机的有界状态并非共享 observation/attempt 账本，手动 trigger 的 claim-before-Popen 窗口及 L2 skill 门控仍待修复。

[V1-01L2 机器可读证据](../evidence/2026-09-10-v1-01l2-skill-activation.json)在两仓各保留一份；[技能契约](../skill-activation.md)区分 legacy string API、strict native composition、逐 source 周期/合格证据基线，以及原生 research 内容寻址文件/子进程校验。只有实际进入提示词的技能在 Popen 与 RoleProcess 登记成功后确认；全部未激活不会预留预算或启动。目录故障的两份草稿函数快照、原封不动的行为测试、最终代码哈希与全量命令可重放。输出/watchdog 回执仍按声明技能推导，内容寻址文件清理与整体输入预算未关闭；没有把启动确认冒充持久 exactly-once 交付。下一项继续 V1-02 的 payload/lease/进程不确定性，而非宣称 V1 已完成。

[V1-02A 机器可读证据](../evidence/2026-09-10-v1-02a-trigger-delivery.json)与[交付契约](../trigger-delivery.md)在两仓各保留一份。原生四类入口共用持久 inbox，实际领取的不可变 payload 到达 Script env/args 与 Claude/research 最终提示词；文件保留不等于未消费、文件消失不等于完成。确定未 exec 可重试，LAUNCHING/STARTED/IN_DOUBT 不因过期或内存清空而重放；匹配角色/nonce/attempt 的实际完成先落库再清收据。固定旧代码重放证明 A→B 丢请求、明确未启动后丢请求与升级误删，独立草稿快照证明坏绑定抢先终态的缺陷。旧测试中的假完成/固定 attempt/claim-unlink 依赖显式版本化，不靠放宽安全合同换绿。未实现不确定任务的自动裁定、完整执行协议指纹或所有研究任务身份；没有科研收益或线上完成声明。
