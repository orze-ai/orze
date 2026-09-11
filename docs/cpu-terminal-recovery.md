# 已确认 CPU 终态的重启续结算

本契约限定 V1-06F 的自动恢复范围，不授予旧进程接管或不确定任务重跑权限。

## 产品入口与效果

正常 CPU CLI 启动在同一 IdeaLake 预算初始化之后、任何新的预算预约和 GO 之前调用 `reconcile_confirmed_terminals(lake, scope)`。因此，若原控制器在 native 终态/effect 确认之后、预算 settle 之前退出，新控制器可以释放该终态仍占用的槽位，再由正常策略、准入、claim 和执行路径处理后续任务。

原 task/generation/full AttemptRef、terminal、artifact、observation 和 effect 不重写；原 worker 不重跑。预留 wall 时间仍全额累计，不退款，实际耗时也不替代预算契约。新控制器不会重新调用旧 Domain 的 prepare/interpret，不从元数据重造旧句柄。

返回值是有界、脱离内部对象的诊断：`schema / examined / settled / already_settled / retained`，仅含预约 ID 和保留原因，不是执行许可。枚举至多声明 slots 个活预约并检测超界；原有 `_totals` 仍检查历史预约，本片不宣称整个预算系统的历史开销有界。

## 资格和停止边界

只有完整当前 TERMINAL，且原生 CPU action 的 phase、scope、database、permit、Ref、claim/config、来源、原发布声明、生命周期、真实整树闭合以及 prepared/committed effect 相互一致，才可自动续结算。资格不仅在枚举时检查，也在真正的同库结算 writer 及独立提交读回中复验。

RESERVED、LAUNCHING、RUNNING、IN_DOUBT、NOT_STARTED 均保留，不自动释放。任何预存 Stop 都不准自动续结算；普通 Stop 保持原义，storage uncertainty 明确 HOLD。旧的公开显式 `settle` 保持其 NOT_STARTED 和 Stop 下的既有兼容语义；它不等于自动恢复入口的授权。

自动恢复不触发训练、GPU/provider、原生 repair、Director 或任何 host PID 扫描。操作系统进程号只作为原持久证据中的字段比较，不用于重新定位并杀死进程。

## 持久恢复状态

新增 `cpu_action_recovery` 表，每 scope 至多一行：绑定 scope、此次新 nonce、IN_PROGRESS/COMPLETE 和有界摘要。仅正常 initialize 的原同库事务幂等创建表；新只读 API 不暗中迁移缺失的新 schema。现有三表的记录不重置。

确认存在可恢复候选后，新 recovery gate 排除并发恢复消费者；先提交并独立读回 IN_PROGRESS，再逐项通过原 effect guard 严格结算。全部内层 guard 与外层 gate 的退出均确认后，才按精确 nonce CAS 到 COMPLETE 并读回。没有候选时可只返回诊断，但仍须先检查已有恢复状态和未知 gate，不能用零 BOUND 掩盖未知。

内层退出失败或外层目录已删除但 fsync 失败，都不能变成恢复成功；持久 IN_PROGRESS 阻断后续启动，包括所有预算行已经 SETTLED 的情况。新版本已有控制器的 reserve、bind、require_permit 也检查恢复障碍，不可利用已释放槽位绕过未确认恢复。

最终 COMPLETE 的 ACK 丢失与中途 IN_PROGRESS 不同：只有在全部物理退出已经确认之后才可能提交 COMPLETE；精确已提交 COMPLETE 可以被新读者识别，不降回 IN_PROGRESS，不据此重复旧执行。已有 Stop 不被自动解除。

## 证据与非目标

真实产品验收使用测试自有 CPU worker 和新的 CLI 解释器。原控制器在真实终态事务与 effect guard 已退出、尚未调用原 settle 的边界实际退出 86；新控制器必须正常完成第二个公开准入任务，再由第三个新控制器验证空闲重启幂等。成功 worker 和非零失败 worker 是同一历史缺口的两个参数案例，不冒称两个独立缺陷。

另外用临时文件/数据库受控故障验证不合格证据、TOCTOU、提交确认失败、锁退出不确定和并发消费。真实 TREE 恢复证据与没有创建任何进程的 RESERVED/NOT_STARTED 兼容单元控制严格分开。日志一致性校验不是独立执行签名，也不能补造没有保留的运行数据。

新障碍只约束遵循本协议的新版本控制器；旧版本常驻进程不会因多了一张表而追溯遵守新门禁。本片不是混版本滚动运行协议，没有部署或生产迁移。它也不涵盖 RUNNING adoption、不确定副作用裁定、全局资源回收、科研收益或普遍效率提升。
