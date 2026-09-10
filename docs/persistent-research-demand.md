# 持久研究需求：观察、门控与提示词（V1-03B）

需求来自已接收入库的生命周期记录，不来自提案 inbox 是否为空，也不由一次调度查询顺便改变任务状态。本切片复用当前 SQLite 目录，只读、不创建或迁移数据库、不调用 `get_unclaimed`、不读取配置正文/指标/大产物。

## 快照的含义

`load_persistent_demand(db_path)` 返回不可变 `DemandSnapshot` 和不可变 `DemandCounts`。一次读事务、固定大小分块读取规范化的小型状态值；不截取前 128/2000 项冒充总数。内存有界不等于读取时间与历史规模无关，也不代表跨角色共享缓存已实现。

| 类别 | 已记录含义 |
|---|---|
| queued | mirror 与全局 QUEUED 一致；阶段缺历史或待启动 |
| evaluation_pending | IN_PROGRESS 且训练 COMPLETE、评估 PENDING，含首次评估与显式补评 |
| claimed / in_progress | 已领取或明确记录执行阶段；不算等待池 |
| inactive | 一致的终态记录；不是有效科学观测的判决 |
| unknown | 身份、mirror/全局/已记录阶段冲突，或运行阶段信息不足 |

全局已终态但阶段仍 IN_PROGRESS 也为 unknown。NULL/错误类型不是合法状态，真正缺失历史与存在但无效的记录分开处理。FAILED/SKIPPED/ARCHIVED 的历史 PENDING 不被擅自改写。领域产物与计算收据仍由派发/评估边界核实。

只有 available 且 complete 时，`queue_count` 才是精确排队数，`waiting_count` 才是 queued + evaluation_pending。否则两者为 None；在可读取但不完整的快照中，`counts.queued` 仍是确定的下界。数据库不可读时的零下界不能称为“没有工作”。

## 实际消费者

原生 `mode: research` 的生产者不因角色改名而绕过需求门控；旧名 `research` 的 script/Claude 入口保留同一路径。已知排队下界超过 `max_queue_size` 时，在预算/Popen 前推迟，手动请求和不足 60 秒的近期周期也不能绕过这个已知积压上限。该上限仍沿用 queued 口径与 `>` 比较，不偷偷改为全部活动任务。

只有完整等待数且已知正资源容量时才推导空闲/深队列节奏；待评估积压不会被空 inbox 隐藏。无库或未知需求不提供饥饿加速，但仍保留正常到期和显式请求的兼容语义；不能据此声称未知队列已被证明小于上限。配额和自身失败门控继续独立生效。当前 GPU 容量策略未在本切片改造成 CPU 执行器。

同一生产者用于门控的需求快照继续进入实际 command/skill/rules-hash/研究 CLI 提示词。`queued`、`waiting`、`evaluation_pending` 的未知展示为 unknown；另有 typed count、available、complete、reason 等字段。通知汇总使用同一只读接口，不会把缺少策略文件的任务写成 skipped。其他旧展示计数（例如 completed 模板变量）没有被本切片宣称全部统一。

配置含 `idea_lake_db` 键即声明目录路由，显式 None/空值按共享默认路径解释，不能让另一个 supplied Lake 覆盖。只有没有 DB 声明的 legacy 调用可以显式提供 Lake。路由不确定时返回未知，不猜测另一个项目的需求。

## 验收与限制

[机器证据](evidence/2026-09-10-v1-03b-persistent-demand.json)记录固定旧开发提交重放、新 API 机制、草稿边界、最终全量、独立复核和推送 SHA。原有业务测试未为本切片改断言。

这是某个时点的已记录积压，不是可立即执行的 ready 列表，不是跨进程原子队列容量预约，不核准 checkpoint/source/lease，也不保证所有未入库提案已计入。多角色仍各自读取快照，未新增缓存一致性协议。预算预约的并发/故障处理、结果分类、默认角色组与完整 CPU autoresearch 闭环继续后续实施；没有 GPU/provider/研究效率净收益声明。
