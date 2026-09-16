# 已关闭 CPU 服务的显式恢复

服务宿主退出后，可以在**同一配置、执行作用域、数据库及运行包版本**上创建新的宿主，
继续使用原研究历史与累计预算。旧服务配置、宿主登记、控制器登记及关闭记录全部保留。

本入口面向已经完整关闭的 `local_cpu_handoff_v1` 服务。没有完整 ACK、存在未结算动作、
未知进程或未完成交接时仍返回 HOLD；不会把删除锁目录当作恢复方法。
实际 systemd、机器重启、跨机器／跨版本迁移、旧 PID 服务迁移与备份回滚仍需独立验收。

需要保存关闭时点的数据时，可先使用[服务备份与数据恢复](service-backup.md)。
备份副本用于核验和后续迁移准备，不替代这里的同作用域执行恢复。

## 准备新宿主

显式提供旧配置、新配置路径和一次性请求标识：

```sh
orze service recover \
  --source-service-config /absolute/project/service.json \
  --service-config /absolute/project/service-next.json \
  --request-id recovery-001
```

默认只准备新配置，不创建控制器。新文件和新宿主登记路径必须不存在，新配置放在 results
目录之外。准备记录固定旧服务文件、关闭快照、来源控制器和请求标识。
不要覆盖旧文件，移动登记目录或清理已经开始的恢复意图。

对于使用 `method: process` 的宿主，可显式启动准备好的新配置：

```sh
python -m orze.service.host --service-config /absolute/project/service-next.json
```

新宿主启动时再次核验原件、所有关闭历史和实际进程状态，再通过已有一次性交接协议
创建自己的后继控制器。准备成功不保证稍后的启动仍满足条件。

## systemd 安装路径

在使用相同运行包的前提下，先通过已有显式卸载命令关闭旧宿主并移除它的三个 unit：

```sh
orze service uninstall --service-config /absolute/project/service.json
orze service recover \
  --source-service-config /absolute/project/service.json \
  --service-config /absolute/project/service-next.json \
  --request-id recovery-001 --install
```

`--install` 在准备后调用已有的项目级安装器，安装新配置对应的 main、watchdog 和 timer。
旧三个 unit 必须完全不存在，或处于与原始生成内容一致的停用状态；旧 main 和 timer
不能仍然 enabled。属性缺失、masked、混合安装状态、非空 cgroup 或剩余主进程均不能据此恢复。
原服务配置及研究目录由卸载操作保留。

核验分别检查加载、活动和安装状态，不把 `not-found` 单独当作进程关闭证明。
这些状态在 [systemd 的接口定义](https://github.com/systemd/systemd/blob/main/man/org.freedesktop.systemd1.xml)
中将加载与活动状态分开定义，并另列安装状态；实现还核验控制器及宿主的本地进程身份。
此安装路径目前只有管理器替身验证，尚未在实际 user manager 上验收。

## 恢复的实际含义

- 新宿主持有自己创建的实际子进程及 pidfd；旧 PID 只用于核实旧进程是否仍然存在，
  从不因此获得发信号或收养权限。异地主机和不可判断的进程状态拒绝恢复。
- 新控制器消费原数据库里的唯一 grant，沿用原配置、累计预算及控制器代际。
  两个已准备宿主竞争同一来源时，只允许一个后继。
- 原有有限预算耗尽后，恢复不会增加额度。预算版本 2 的 `wall_budget_seconds: null`
  继续表示没有研究总期限；每个实际 CPU 动作仍有自己的有限时限。
- `.orze_disabled`、`.orze_stop_all`、`.orze_shutdown` 继续阻止新工作。
- 新宿主正常关闭后，可将它作为下一次恢复来源，继续保留整条历史。
- 缺失回复或不确定启动保留原意图，不能通过重用配置文件自动重试。状态及文件需要核实时，
  同一请求标识也不会绕过唯一 grant 创建另一个后继。

该入口解决已关闭宿主的延续执行，不证明科研质量或效率提升，也不提供非正常崩溃后的自动修复。
