# 原生执行 attempt 与副作用发布契约（V1-02D2）

本文说明开发分支的边界，不是版本发布或整个 V1 的完成声明。Core 提供实现；Pro 使用同一 Core 的执行与完成事件，不另建一套 attempt 权威。

## 身份与职责

通用存储只认识 task、phase、attempt、generation、状态和 JSON 绑定，不认识 GPU、ASR、指标优劣、假设或科学结论。训练、评估及 post-script 是接入它的适配器；`launch_failure_report` 是控制器报告动作，不是一次 OS 执行或科学观察。

`AttemptRef(task_id, phase, attempt_id, generation)` 固定一次尝试的身份。同一任务/阶段的后继尝试使用新 ID 和递增 generation；旧回调不能通过查询“当前 attempt”取得新尝试的权限。结果目录的 `_execution_catalog.json` 只声明数据库路由，不是授权凭证。显式 Lake、claim 中的 DB 绑定与声明必须一致；缺失对象、错误数据库、无效声明不能降级成 legacy 写入。

| 记录 | 含义 |
|---|---|
| `LAUNCHING` | 已提交启动意图；不是“没有进程”，也不允许按超时盲目重放 |
| `RUNNING` | 框架已观察并登记执行创建；不是进程此刻仍存活的证明 |
| `NOT_STARTED` | 该适配器已明确确认未执行；不是失败的科学测量 |
| `TERMINAL` | 这个 attempt 的终态被接受；成功与否及领域有效性另行判断 |
| `IN_DOUBT` / publication HOLD | 现有证据不足以确认副作用；不自动释放、接管或重试 |

只读存储入口不创建表；写入入口要求调用者已经拥有 SQLite 事务，不自行 BEGIN/COMMIT/ROLLBACK。JSON 身份按规范编码精确比较，不把 bool/int 或 int/float 当作相同绑定。schema、主键与记录歧义不自动“修复”成成功。

## 短发布边界

每个任务使用 nonce/身份绑定的 `_attempt_effect.lock`，不按年龄接管。嵌套使用必须显式传入同一 lease。原生启动先在短事务内提交 intent，然后释放锁再 Popen；执行创建后用新短事务确认身份、start 收据及生命周期。

大产物哈希、领域评估验证、GPU 等待、Popen、进程停止和 provider 调用不在这个短锁/SQLite 事务内。lineage/interruption 的慢准备与小型发布分离；发布前后检查准备时的输入及路径身份。当前新的短发布接口对目录 checkpoint/模型有明确能力限制，不能伪装为已验证；旧 wrapper 的目录兼容不等于原生接口支持。

改变文件的终态事务先写不可覆盖的意图：

```text
_execution_effects/<attempt_id>/prepared.json
_execution_effects/<attempt_id>/committed.json
```

prepared 绑定完整 attempt ref 和有界操作说明。终态行绑定 prepared 原始字节的 SHA-256。协调器在提交前后核对当前 attempt、明确的生命周期绑定及依赖行，并且只有 SQL 提交被确认后才写 committed 文件。文件同步、精确读回、父目录同步、事务提交或清理不确定时保留 HOLD。committed 文件本身不是 SQLite 提交证明，文件与 SQLite 也不是一个物理原子事务。

`watch_attempt` 用于本次 intent/start 的当前行与显式 lifecycle；`watch_dependency` 固定一个仍为当前、已闭合的源 attempt 的完整行，不要求它的历史全局 FSM 永远等于现在的 FSM。例如训练结束后评估改变全局状态是合法的。`lifecycle_phase` 可让控制器动作显式约束 training 生命周期，不为它编造同名阶段。

计算收据是分配事实，不是 observation。原生校验除了文件读回，还核对调用者的 phase、process PID、资源、规范化 start clock、outcome、reason 与严格整数 return code；RUNNING 终态要求已存在且一致的 start。LAUNCHING 的已确认停止可以缺少 start，但不能忽略一份已经存在的冲突 start。显式 pre-native completion import 保留旧时钟兼容，不冒充 Popen 前的原生 intent。

## 已接线的消费边界

训练及评估的正常结束、失败、受控强停和已确认的初始化失败通过当前 attempt 发布；重复/过期回调不再写当前 metrics、FSM、计算终态或重复交付 finished。活动槽位只移除仍属于该对象的条目。失败初始化的原始 ref 在清理确认后随原异常交给报告动作；它不会在迟到时替新 claim 报错。

claim/reset/孤儿清理及显式评估 retry 共用任务发布 guard，并检查所有阶段尚未闭合的 attempt，不仅检查 evaluation。已有 post-script intent 也会阻止替换它依赖的评估结果。

完成值保留二元迭代/索引/比较兼容，同时携带不可变 ref；它不再是可附加任意属性的 tuple。phase、常驻 loop、once、auto-GPU 和通知消费者保留并复核这个 ref。数据库 backlog 是一次新的调度请求，可以选择当前已接受的训练源，但不能把旧完成事件重新解释为 B 的完成。`launch_eval=None` 不能生成第二份原生完成事件。

原生 post-script 在 Popen 前有独立、绑定 source ref 与命令的 intent；每脚本的确定性动作身份用于防止自动重复。未知 Popen/停止结果保留非终态与 HOLD。自然退出的终态仍不证明所有逃逸后代及领域产物均安全；这个限制没有因加了 attempt 表而消失。

原生 graceful shutdown/atexit 先确认实际 attempt，再停止和发布。配置缺失的 atexit 只记录分配/生命周期，不编造评估输出或可恢复 checkpoint。停止不确定或发布失败保留 handle。允许 detach 的训练仍保留 claim/start/RUNNING；这不是重启后自动重建 native handle 已经完成的证据。

旧 startup 恢复器在真实文件/数据库发布边界持短 guard 重新检查 legacy 资格，不得接管存在 native catalog/history 的任务，即使 PID 消失或 metrics 已写。原生 adoption 必须由后续明确协议处理。旧 resume admission 同样不能绕过原生所有权；拒绝意味着需要原生恢复动作，不意味着 checkpoint 可以被删除或任务已经安全重试。

默认无库目录兼容仅适用于配置加载器实际补入默认 DB、且不存在 claim/native 绑定或声明的旧任务。内部默认来源标记由加载器重新计算，用户提供该标记不能获得权限；显式配置但缺失的 DB（即使路径与默认相同）仍拒绝恢复受理。

## 兼容迁移与明确未完成事项

既有无 native 历史的调用保留显式兼容路径；并非所有 legacy helper 被宣称具有新的原生保证。测试中没有真实 idea admission 的孤立 FSM、直接改持久 launch clock 的假时钟、无真实 claim/start 的训练监控替身，以及把持久 publication HOLD 当作可自动重试的预期，均以保留原文件、记录新契约的方式迁移；不能将这些迁移统计为旧产品行为红测。

原生失败不会在发布锁内调用 LLM 修改共享代码。启用 repair 的普通失败记录 `pending_explicit_action`；启动失败报告也标注待显式动作处理。这是诊断字段，不是已经接通的持久 repair 工作队列。自动 repair worker 与隔离的补丁发布仍待实现。

本切片不承诺：

- 原生重启 adoption、未知结果的人工裁定工具或自动解除 HOLD；
- 每消费者的持久 ACK、跨崩溃 exactly-once 通知、全局失败/完成计数重建；
- 独立 artifact/observation 身份、同 ID 历史修订、复验协议或统计推断；
- surviving worker 对 canonical output 的物理写入隔离，或所有 legacy/posthoc/pre-script 路径的统一迁移；
- 任意规模的历史读取/清理。文件收据有明确上限，存储历史校验和保留策略仍需容量设计；
- CPU 通用研究闭环、跨领域留出验收或研究效率净收益。

最终验收、源代码 SHA、明确的旧发布版/草稿红测区别、fixture 迁移及远端 commit 由配套机器证据和实施账本记录。上述未完成项目必须继续实施，不能因本切片测试通过而将整个 V1 标记完成。
