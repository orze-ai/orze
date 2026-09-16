# 原生 CPU 控制器的注册停止

`local_cpu_stop_v1` 将原生 CPU 动作接入既有控制器登记、成员和停止协议。
主入口仍为 `orze -c CONFIG`，使用原来的 CPU 策略、预算、执行记录和监督器。
这是默认关闭的本机停止能力。

## 配置与操作

在 CPU 项目中显式声明：

```yaml
controller_control:
  version: 1
  profile: local_cpu_stop_v1
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

原有 CPU 配置、路径和动作合同仍然适用。该配置不接受物理 GPU 范围、旧角色、
managed run-idea、远端执行或自定义清理脚本。也支持显式
`execution.version: 2`、`wall_budget_seconds: null` 和有限动作运行期租约；
控制器协议版本与执行预算版本是分别声明的。

从相同工作目录、使用相同配置运行：

```sh
orze -c /absolute/path/orze.yaml
orze stop -c /absolute/path/orze.yaml --timeout 60
```

旧形式 `orze -c CONFIG --stop --timeout 60` 使用同一个停止观察者。
这里的 timeout 是等待停止确认的时间，不修改动作的 timeout 或预算。
只有确认旧控制器已经完成收尾并实际退出，停止命令才返回 0；未知返回 75。

## 确认内容

登记绑定真实数据库、结果目录、工作目录、完整配置、主机／启动身份和进程出生身份。
CPU 路径不写兼容 PID 文件；停止观察者先捕获并核验登记进程的 pidfd，随后发送持久
停止请求，并等待同一个内核句柄对应的进程退出。

控制器在发布 ACK 前必须确认：

- 停止请求线程已退出，全部自有动作及其后代已经通过原监督器关闭。
- 完整动作终态和 effect 回执已提交，控制器成员已结算。
- **CPU 预算也已单独结算**。该步骤发生在动作提交之后，不能从成员结算推断。
  全作用域预算预留与本控制器动作逐项对应；缺少 attempt 的 RESERVED 行同样阻止 ACK。
- 原数据库连接已关闭，GPU 资源始终未分配。

ACK 包含预算作用域身份、预留数量和完整预留摘要。发布 ACK 的事务再次核验预算；
停止观察者还会独立重读预算、完整终态和回执。丢失数据库关闭响应、结算哈希变化、
未确认进程关闭或未确认预算提交均保留拒绝状态。

## 恢复边界

首次使用需要新执行作用域。允许预置排队任务；已有执行记录、旧控制标记或未知 owner
不能自动导入。登记及停止历史保留，普通第二次启动会拒绝接管。
每实例沿用既有控制器成员上限，达到准入阈值时请求停止，不丢弃未知成员。

该配置只支持停止。`restart`、后台 `start`、`resume`、`enable` 和 service/watchdog
交接尚未接入。预算 ACK 是当前单实例全作用域摘要，不是跨代交接历史格式；
不能将 `CompletedControllerStop` 当作新实例启动许可。
同一 UID 的任意外部写入者不在本协议隔离范围内。

测试使用实际 CLI、原生 CPU worker 及后代、SQLite、pidfd 和完整预算结算；
没有使用真实模型、GPU 或已有服务。它证明停止可靠性，不提供研究质量或速度提升证据。
