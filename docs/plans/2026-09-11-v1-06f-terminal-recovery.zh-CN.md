# V1-06F：已确认 CPU 终态的重启续结算

日期：2026-09-11。本片承接原实施方案的可恢复执行要求，基线为已闭合并推送的 Core `55c2ed321dc759b6b96d721e0b3b423151bbc49b` / Pro `af8e3c7f4c83a1883d11ef4756efd4c4a8ee48cc`。V1-07B 原留出验收已经在旧冻结 Core 上完成；本片不回改该挑战、其应用、旧测试或历史完成声明。

## 当前缺口与先验状态

只读源码显示 native_cpu_action.harvest 在 execution_transaction 完整退出之后才调用 budget.settle。该处控制器退出，可能留下当前 TERMINAL、真实已确认 effect/TREE_CLOSED 和仍为 BOUND 的预算记录。新控制器的内存 handles 为空，启动路径不消费该持久终态；后续普通队列会持续因占槽而 Wait。底层显式 settle 已具精确幂等结算能力，缺少的是产品重启消费者。

上述是静态定位，不是已运行红测。必须先以完整基线真实 CLI 故障复现，观察恢复要求的行为断言失败；不能把新增 API 不存在或导入错误当成产品红测。

## 最小实现边界

在既有 budget 模块新增有界的已确认终态续结算入口，并接到 CPU 启动的正常预算初始化之后、任何新 reserve/GO 之前。只枚举本作用域至多 slots 个活预约（显式检测超界）；不把原有 _totals 全历史扫描冒称总开销有界。

仅考虑 BOUND、完整当前 AttemptRef、phase=action、kind/origin=native_cpu_action、原 scope/database/permit/attempt 绑定完全一致的 TERMINAL。实际结算必须复用现有 settle 的每任务 effect guard、已确认 effect 和真实 closure 元数据校验、同库事务及提交后核实。不造旧执行句柄，不根据旧 PID/年龄取权，不再解释 Domain、不新发 observation、不重复 GO、不退款、不重置 task/generation 或已有 Stop。

RESERVED、LAUNCHING/RUNNING/未知状态、NOT_STARTED 和不确定结果不由本片接管；健康对端的活预约继续保守占槽等待。坏/丢失/替换的 effect、树证明、Ref、scope、持久停止不确定性或残留未知 owner 均不能释放槽位。budget_storage_unconfirmed 的持久 HOLD 不作为自动重试权限；普通已记录 Stop 可以继续保留，续结算不得使它恢复执行。

## 真实红绿验收

新增独立临时 CPU 项目及其自有 controller 监督器，真实 cli.main→Orze→原生 worker。故障仅透明截在 budget.settle 的首次入口：先只读确认无打开事务、实际当前 TERMINAL/fullRef/terminal 一致、真实 prepared/committed effect 完整且原 guard 已释放，输出并 flush 故障边界，再真实 os._exit(86)。不先调用 settle、不写伪造生命周期、不用测试框架替产品发布或结算。

父进程持有实际监督句柄并收集 controller TREE_CLOSED，要求无 forced cleanup；不会扫描或杀 host 进程。只读保存原 attempt、产物、effect 和 BOUND 记录，通过公开普通准入增加另一个合法任务。新解释器使用同一配置，不带故障开关，再运行真实 CLI。

旧行为应体现原 BOUND 未释放、下一任务未执行；修复后应只续结算原终态、正常执行下一任务。要求原 fullRef/terminal/artifact/effect 原字节不变、原 worker 仅一次、新 worker 一次、无新原任务 generation、额度累计不退款、最终正确 SETTLED，并检查再次重启幂等。单个行为 red 与新增机制/反例数量分开记录。

## 反例、兼容与证据

覆盖开放预约保留、当前引用/permit/type 绑定、普通 Stop 不解禁、存储不确定 HOLD、effect/closure/guard 拒绝、并发精确重放和提交确认失败。只使用临时数据库及测试自有进程；不跑 GPU/provider/生产迁移，不接管未知活任务。

保留实际红测、完整候选快照、原运行报告与前后文件 SHA；冻结测试后全量 Core/Pro/配对及原 A/B 验收重跑，独立复核后分片 commit/push 并核对远端。底层无进程的 NOT_STARTED 单元夹具不能冒充真实 terminal/tree/controller 崩溃证据。

本片不是完整 RUNNING adoption、publication HOLD 裁定、原生 repair、Director 正向交接或全局历史记录开销优化。原计划与整个 V1 仍按证据逐项收口。
