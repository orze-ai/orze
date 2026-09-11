# 同配置、同作用域的单次控制器交接（V1-05C2e2d）

这是显式、有限的本机交接协议，不是任意旧进程接管、崩溃恢复或无限运行。
沿用原 Python CLI、Orze、IdeaLake、成员监督与执行路径，不增加第二套 runner。
限定机制已通过离线验收；[机器可读证据](evidence/2026-09-11-v1-05c2e2d-controller-handoff.json)
区分实际执行、故障重放、夹具和未开放边界，不代表整个 V1 完成。

## 配置与调用

只在全新登记作用域显式设置：

```yaml
controller_control:
  version: 2
  profile: local_handoff_v1
```

其余严格范围与 [停止 ACK](controller-stop-ack.md) 相同：原工作目录、完整已解析
配置指纹、实际配置文件路径及字节 witness、数据库与结果目录 inode、真实物理 GPU
集合全部固定。后台 admin、远程执行、自动升级和未受监督的可选入口仍不开放。
这不是完整项目配置，也没有把 CPU 动作伪装成 GPU 工作。

首个实例仍由原前台 `orze -c CONFIG` 启动。调用者必须回到同一工作目录：

```sh
orze restart -c CONFIG --request-id operation-001 --timeout 60
orze -c CONFIG --restart --request-id operation-001 --timeout 60
```

操作 request ID 必须明确提供，限定为 1..128 个 ASCII 字母、数字及 `_.:-`。
它与复验任务的 request ID 不共用解析字段。重复投递使用同一 ID；改 ID 不会
绕过“同一来源只能产生一个子代”的约束。timeout 是控制操作等待界限，不改变训练配置。

旧 `{version: 1, profile: local_stop_v1}` 保持原 schema 与 stop-only 合同；
不会在线升级已有登记、删除 owner 或凭已退出 PID 开一个新实例。

## 停止证据与启动权限

独立命令／API 先捕获旧控制器的真实 pidfd，并复验 PID birth identity。
沿用停止协议，必须同时具备匹配 ACK、完整成员／动作／原资源收尾和同一 pidfd
的真实退出。手工构造 `CompletedControllerStop` 或磁盘 ACK 不能发行交接。

在同一个 SQLite 中，`controller_scope_heads` 指定唯一当前实例及 pending grant；
每代保留永久 anchor 下独立的 create-only owner。旧实例、请求、ACK、成员、终态与
effect receipt 不删除、不重新排队、不按年龄失效。身份按连续 generation、
predecessor 和已确认前代 STARTED 链核对。

持久操作依次经过 RESERVED、SPAWNING、ISSUED、CONSUMED、PREPARED、STARTED。
SPAWNING 的写入和新连接读回发生在唯一一次真实 Popen 之前。启动异常、命令崩溃
或等待超时不回滚成可再次 Popen 的状态。

子进程执行原解释器的 `-m orze.cli -c CONFIG`，只继承显式传递的 bootstrap
socket，不继承旧 GPU 租约 FD。UNIX SEQPACKET 每帧核验 SCM_CREDENTIALS；
HELLO 绑定实际子进程 birth identity、随机 nonce、配置及运行入口 witness。
父进程捕获自己的实际子进程 pidfd 后才持久化 ISSUED、发送带随机 token 的 GO。
token 只在存活的私有通道中传递；数据库保存其哈希。

原 CLI 校验启动模式并装载完整配置后接受 GO；没有实际通道和强进程内 admission
不能跨过永久 namespace。子代在一个真实事务内消费 grant、更新 current head、
插入实例和 Session binding，提交后重新读取确认。

## 资源准备与真正开始

子代取得原 Lake 连接、自己的 PID 文件和 GPU 租约，尚不进入第一条探针／研究循环。
PREPARED 只是“资源已准备”，不是启动成功。父进程核实真实子代登记、绑定与
资源准备记录，再发送 COMMIT。子代必须在提交前重新核实实际资源与当前准入，
持久化并读回 STARTED，随后沿用原循环。

父进程只有收到匹配最终 STARTED、重新核实持久记录并确认捕获的子进程仍活着，
才返回 `CompletedControllerHandoff`／CLI 0。此返回对象只是信息，不是另一张启动票据。

最终回执丢失时，操作调用者返回 HOLD／75；若子代已经真实提交 STARTED，它可
独立继续。不能把丢回执当成第二次启动理由，也不能把已提交的 STARTED 降级后重做。
新的调用者只可重新捕获当前活着的子代并核实同一操作；未知、未提交或已退出则 HOLD。
bootstrap EOF 本身从不证明完成，也不授权未提交的子代进入执行。

这里的“独立”仅指后继不依赖本片已验证的外部观察命令继续存活，不是脱离任意
宿主祖先监督器、cgroup 或服务管理器的证明。`current_controller() is None`
也不证明调用者不属于另一棵监督树。受监督 role／script 内直接调用本 API 不在
支持范围：`start_new_session` 不改变后代归属，角色的整树 STOP 仍可能终止其后继。
因此 Pro Director 的默认受监督角色不能直接接入本 API；必须先提供明确的控制
动作归属与交接边界，不能用 setsid、双重 fork 或清空环境绕过监督。

## 有界性与未开放能力

发行来源 generation 必须小于 64；每次发行检查的跨代已闭合历史最多 4096 个成员，
不丢弃历史以腾出预算。新实例仍受自身 2048 软停止／4096 硬限制约束，其新成员
可能使下一次交接超出历史上限而 HOLD。这是有限资格范围，不承诺
长期无限研究或历史压缩。未知／孤立／不一致的任务 FSM、未闭合 attempt／claim／
stage 和未核实 effect 一律阻止交接；合法排队、PENDING 与已确认终态按原协议保留。

同配置重启不允许 `enable`、离线 resume、重新分配 GPU、升级安装包或任意 owner
恢复。Pro Director 暂不把这个 API 的存在当作已接线；其动作仍保留拒绝合同，待
独立验收真实调用者、权威配置载入、稳定操作 ID 与确认消费。

离线实际进程测试仅运行合成 CPU 子进程，GPU 元数据／容量、私有租约目录和 license
是明确夹具边界；文件锁、SQLite、CLI、进程创建、pidfd 和协议消息需要保持真实。
这不能证明线上 GPU／provider 可用、科学结论正确或研究效率提高。

兼容性细节：停止协议的核心请求／ACK／退出接受条件保持不变；提取共用等待函数后，
停止等待窗口从观察者捕获完成后起计。它不是覆盖路径读取、观察者构造及全部外部 I/O
的硬实时总截止时间，也不应把这次重构描述为原函数完整时序逐字节不变。
