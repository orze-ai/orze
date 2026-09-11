# 角色进程树监督：V1-05C2e2a

本片关闭新 Pro `run_role_step` 的角色执行归属与结果消费边界，是正向控制器停止确认的前置项，不是整个控制器 ACK、跨重启接管或科研效率验收。

## 启动与闭合

在已有角色锁中先持久化 schema 2 `role-process.json` INTENT，再创建阻塞于 READY 的 worker。当前进程的强 owner 保留确切监督句柄，构造并绑定 RoleProcess、确认同一触发尝试 STARTED 后，才写 GO_REQUESTED 并发送 GO。技能、thinker 等激活 ACK 在成功 GO 后发生。命令、环境、原始 nonce 和触发正文不落入此回执；它保存绑定身份及哈希。

角色使用既有 Linux subreaper/pidfd 监督原语。主进程退出不代表整树结束；需要真实 TREE_CLOSED、匹配 READY、严格整数退出码及原语的 ECHILD_WALL 证明。STOP 或强制清理之后的退出 0 仍是操作失败，不升级为有效研究结果。

结果收割、graceful shutdown、atexit 和内部升级的角色门禁都识别新 owner。监督、存储或身份不确定时保留 owner、活动项和锁，不返回 finished、记 usage、递增 cycle 或启动同角色下一次工作。HOLD 一旦由 owner 锁定，不因后续可读、进程退出或超时自动解除。进展诊断仍复用旧的日志、产物元数据和 `/proc` CPU 观察；这些观察不授予停止或闭合权限。

## 交付与锁释放

终态使用启动时捕获的数据库路径与完整触发尝试 token，不从消费时可变 RoleProcess 字段重新拼装。消费前持续检查公开对象是否仍与捕获身份一致。只有当前同代、同尝试、精确 outcome/exit_code/cleanup_verified 的终态，才允许释放；没有触发任务的角色也必须有整树闭合证明。

新 owner 将整个已验证锁目录以 no-replace rename 隔离后执行有界回收，不先删除回执再调用无条件 `_fs_unlock`。成功的 `settle_role_delivery` 已经释放新锁，调用者不能再次裸解锁。关闭只删除本次捕获且确认结束的原对象，不能覆盖期间被替换的活动槽位。

明确未进入 prepare，或 READY 后未尝试 GO 且已收到真实 STOP/TREE_CLOSED，才可证明从未执行；触发任务还需精确 LAUNCHING→PENDING 后才能释放并重试。启动确认实际已写但响应丢失、GO 不确定、未知 prepare 交接都不满足该条件。单纯的终态函数显式拒写、且 owner 没有进入 sticky HOLD，可以保留 STARTED 后重试真实终态写入；这不等于清除未知执行状态。

## 兼容与限制

旧 schema 1/裸 RoleProcess 保持旧路径，不能被描述为已获得新整树证明。重启看到 schema 2 回执时必须 HOLD，不能用 nonce 扫描、PID 消失、年龄或锁超时自动接管。当前强 owner 是进程内权威，不是跨主机或崩溃恢复协议。

本片没有实现控制器登记、停止请求消费、全体成员封闭集合、控制器退出证明或一次性 restart ACK。内部升级的角色门禁只阻止带未闭合角色的 exec；之前的安装动作、旧训练/评估升级路径、bot、自定义 cleanup 和外部服务管理器没有因此获得安全认证。既有关闭 sentinel/PID 清理也不是 ACK。

测试区分真实 CPU/SQLite 行为、故障注入和显式模拟监督夹具。旧 Popen 替身迁移不等于真实 READY/GO 证明；既有业务断言的合同修订单独记录。没有使用模型、数据集、GPU 或付费 provider，也没有测量或宣称研究收益。
