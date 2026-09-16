# 原生 CPU 控制器的一次性交接

显式 `local_cpu_handoff_v1` 将原生 CPU 主循环接入已有的单后继协议。
它支持保持相同配置与预算作用域的控制器交接；不会重置预算、清除旧任务的 claim，
也不会自动重试失败任务。原 `local_cpu_stop_v1` 继续只支持停止。

## 配置与入口

从新的执行作用域启用：

```yaml
controller_control:
  version: 2
  profile: local_cpu_handoff_v1
execution:
  version: 1
  resource: cpu
  slots: 1
  wall_budget_seconds: 120
telemetry: false
auto_upgrade: false
max_fix_attempts: 0
metric_harvest:
  enabled: false
```

其他路径和 CPU 动作配置沿用[原生 CPU 动作](cpu-actions.md)。执行预算也可单独声明
`execution.version: 2`、`wall_budget_seconds: null`；单动作仍需要有限运行边界。
控制器协议版本与执行预算版本互不替代。

使用同一工作目录和配置启动，然后从外部进程发起交接：

```sh
orze -c /absolute/path/orze.yaml
orze restart -c /absolute/path/orze.yaml --request-id handoff-001 --timeout 60
orze stop -c /absolute/path/orze.yaml --timeout 60
```

旧形式 `orze -c CONFIG --restart --request-id handoff-001` 使用同一协议。
timeout 是等待控制操作的预算，不修改动作预算或设置研究总期限。
`start`、`resume`、`enable`、`restart --foreground` 不由该声明授权。
已有停止模式的登记不能通过修改配置自动升级为交接模式。

## 关闭、准备与启动

1. 外部发起者捕获并核验旧登记进程的 pidfd，发出持久停止请求。
2. 旧实例停止自有 worker 及后代，核验动作终态、effect 回执与独立 CPU 预算结算，
   关闭原数据库连接并写 ACK。发起者还必须等待旧进程实际退出。
3. 在原数据库中保留一次性的启动意图；未知结果不会清掉意图并再次启动。
4. 后继实例通过继承的内核凭证通道接收授权，事务性消费登记，再运行原 CPU 初始化。
   预算作用域实际绑定、原历史重验后，才发布 PREPARED。
5. 收到 COMMIT 后再次核验预算、历史和资源，写入 STARTED，然后才进入策略与动作循环。

CPU 的 STARTED 记录绑定实际预算作用域，不伪造 GPU 租约或 PID 文件。
停止确认不单独授予下一次普通启动权；后继必须消费同一个有效交接意图。

同一 request-id 的重放只核对已有结果，不产生第二个后继。若最后的 STARTED 回复丢失，
结果可能是未知；相同 request-id 可以重新核验。未完成的 PREPARED 或失去所有权不能
通过换一个 request-id 自动接管。返回 75 表示没有获得完整确认。

## 跨代预算证明

停止模式的预算摘要覆盖整个单实例作用域，不能直接用于交接。
新模式使用 schema 2：每个 ACK 的预算摘要只覆盖该控制器的精确预留和 AttemptRef，
并核验对应终态及原始 effect。后续合法预留不会改变旧摘要。

关闭和交接准入仍检查全部历史：所有预留必须对应登记成员，没有活跃或孤立预留。
历史读取可以核验旧任务代际；实际结算入口仍要求当前代际，不授予旧执行权。
旧的 wall 预留全部保留，不因控制器换代退款。

沿用既有协议上限：最多 64 代，交接时全部历史最多 4096 个成员。超过上限保留拒绝，
不会丢弃历史或未知资源。有限验收或控制操作的超时不是自主研究的总期限。

## 边界

这不是 service/watchdog 集成、进程收养或生产迁移。发起者必须独立于被关闭的控制器；
进程创建新 session 也不证明它脱离服务管理器的 cgroup 生命周期。服务隔离、manager
环境、迁移和回滚仍需分别验证。对同一 UID 的任意外部恶意写入不提供隔离。

本能力解决控制器连续运行的所有权和预算约束，不提供研究质量或速度提升证据。
