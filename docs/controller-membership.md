# 内部控制器登记与成员账本（V1-05C2e2b）

这是 C2e2 正向停止确认的基础组件，不是已完成的控制器停止/重启功能。
本片不接入 `Orze.run`、CLI 或 Director，不提供 `confirmed`、ACK、接管、
解锁或重启 API。既有 C2e1 公共停止/重启入口仍按其保守拒绝合同工作。

## 三个独立义务

1. 在执行可能发生之前保存成员意图，绑定本次控制器身份。
2. 在实际 READY/GO/TREE_CLOSED 边界登记 OS 状态。退出码不能代替整树闭合。
3. 在数据库事务、文件效果确认及资源收尾完成后登记动作结算。

所有持久记录复用当前 IdeaLake SQLite。没有新增执行服务、第二套 runner、
另一个生命周期数据库或以公开 active maps 为权威的成员清点机制。

## 登记

内部 `register_controller(lake, scope)` 只接受真实、持久的 IdeaLake。
它检查实际 `main` 路由、数据库策略、普通单链接文件、目录身份和明确的表结构；
拒绝调用者已有事务，不提交调用者的工作。

登记绑定随机 controller ID、主机/boot ID、自身 PID/start ticks、scope 和数据库
路径及 inode。公开 `identity` 是副本，不可通过修改副本重路由。
同进程只能保持一个强 context；新线程也可见，fork 在获取继承锁之前即拒绝。
控制读取使用现有路径的新连接，不把线程不安全的 `IdeaLake.conn` 交给控制线程。

同 scope 的持久 nonce 所有权复用已审计的 source-lock 协议。旧 owner 即使已退出、
另一调用改用不同数据库，也不能通过年龄、PID 文件或空成员列表接管。
部分登记或提交响应丢失保留强 HOLD；没有清空 global 后重试的生产接口。
本片刻意不释放该 owner；进程正常退出不等于它的义务全部完成。

`quiesce(request_id)` 仅限制这个 context 的新工作。请求 ID 必须精确匹配，
不同请求不能覆盖已接受的请求。它不确认已运行工作闭合，也不授予重启权限。

## 成员接线

六类 native OS 动作是 training、evaluation、posthoc、post_script、pre_script、
artifact_preflight。成员意图随完整 AttemptRef 在调用者同一 SQLite 事务插入。
三个 failure-report 动作没有自己的 OS 工作，仍须有完整 source ref 和动作结算。
`legacy_import` 不会自动获得新控制器的执行所有权。

通用监督原语在可能创建进程之前绑定成员；READY、GO 和实际整树闭合更新同一成员。
强内部成员/handle 关联不依赖公开进程字段或 active slot。
本片每实例最多保留 4096 个成员（包含已结算成员），达到上限即拒绝新的成员，
不会驱逐未知 owner。这是内部有界验收范围；不能将它直接当作无限期产品运行策略，
后续生命周期接线还需处理已结算历史与运行期限。
quiesce 后的已准备工作只发送一次 STOP，不再 GO；调用者仍需等待真实闭合。
控制检查本身不并发接收监督消息、不递归调用 `poll → stop → wait`。

native 的 `finish_attempt` 只是结算候选。必须等整个 `execution_transaction`
退出，包括 SQL 提交、文件效果确认、SQLite timeout 恢复和 effect guard 释放，
再核实终态、登记动作 SETTLED。提交或收尾不确定不能被整数退出码清除。
role 要等确切交付结算和 owned lock release；bounded executor 和 probe 还须完成
输出处理及本地描述符收尾。成员账本本身不判断科研结果是否有效。

## 本机探针

`run_probe` 是对现有监督和有界流收集器的薄适配，不是修复器策略。
没有 context 时原样委托当前 `subprocess.run`，保留旧调用和测试边界。
有 context 时只支持已声明的本机、无交互、有限 timeout 子集；不支持的路径不降级
为裸进程。默认 GPU 空闲检查、GPU/进程诊断、本机 fleet 查询及 kernel probe
共用该边界。远端 SSH 探针在执行前拒绝。

完整 stdout 与 stderr 合计上限 64 KiB；返回普通结果之前要求整树闭合、管道 EOF
和描述符关闭。持久元数据只有分流哈希与计数的绑定摘要，不保存输出正文、argv 或环境。
STOP 后即使真实返回 0，也不返回普通成功探针；输出不完整不能变成可用 GPU 的空清单。

## 尚未关闭的边界

- 没有实际控制请求读取、控制器资源释放、ACK 发布、观察者预先捕获的 pidfd 退出证明，
  或一次性重启消费。`members_snapshot` 仅用于诊断，不是这些权限的替代品。
- 尚未完成静态拒绝分支的 source-qualified 无执行证明，也未开放 quiesce 后的新派生
  failure report。缺 handle 的 NOT_STARTED 不会被猜作已闭合。
- 内部 bounded executor 目前要求其 cwd identity 与登记 scope 一致；没有借用
  任意项目路径或已有 native repair 历史获得新的执行许可。
- 产品 profile 仍待在执行前拒绝未迁移的 bot、housekeeper、自定义 cleanup、内部升级、
  LLM metric fallback、legacy 裸执行及远端工作。本片未声称这些已被全局拦截。
- 这不是任意 Python 扩展或恶意同 UID 代码的沙箱，也不是跨主机协调、崩溃恢复或
  自动裁定未知执行的系统。SQLite/文件系统的同步能力仍依赖部署环境。
- 内部真实 SQLite 和小型 CPU 进程测试不等于真实 `Orze.run` 闭环，不证明 GPU/付费
  provider 运行、通用 CPU 调度或研究效率提升。通用 autoresearch 的后续验收仍按原计划推进。
