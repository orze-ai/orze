# 已记录阶段与完成资格

原生排名、研究证据节奏、有界 lifecycle 查询、catalog 展示和通知指标缓存刷新，必须同时尊重全局 FSM、旧 status 镜像，以及已经记录的 training/evaluation 阶段。

| 已记录阶段 | 可与全局 COMPLETE 一致的值 |
|---|---|
| training | COMPLETE |
| evaluation | COMPLETE 或 SKIPPED |
| 确实缺少历史阶段表或某阶段行 | 保留旧任务兼容；不补写成功记录 |

已存在行的 NULL、未知值、非终态不能被当作“未记录”。其他扩展阶段不被赋予臆造的失败语义。状态值精确比较，不由历史表的大小写/尾空格 collation 放宽。

共享实现位于 `orze.reporting.lifecycle_stages`。只读入口在同一个 SQLite 读事务内验证真实表、必要列和唯一身份，再读取状态。缺必要结构、view、NULL 身份或重复身份使整次读取不可用；历史无主键但实际身份唯一的表仍可读取。普通任务的已记录阶段冲突只撤销该任务的完成资格；有界精确 ID 集请求中任何一个请求任务存在冲突时，整个请求拒绝。

catalog 将阶段冲突显示为 UNKNOWN，但不修复或改写数据库。Pipeline 计数保留原来的全局 FSM 口径，不改成阶段合格数量。结果文件与热缓存不能恢复已经撤销的资格。

通知的 `eval_metrics` 仍只是可变诊断镜像。刷新只在自有 `BEGIN IMMEDIATE` 事务内检查结构与唯一身份，并在 UPDATE 自身重新检查状态/阶段。如果调用者已经开启事务，则跳过这次 best-effort 刷新，不提交或回滚调用者的其他修改。失败只回滚本次拥有的事务。

本项不证明指标本身正确、独立 observation、统计显著性或科研进步；不修改排队/失败状态机、自动恢复任务，也不提供数据库与结果文件之间的原子快照。真正缺失阶段历史的兼容性不是补造 provenance。V1 的任务/attempt/observation 身份和真实 CPU 闭环仍需后续验收。
