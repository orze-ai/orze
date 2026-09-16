# 独立项目的常驻服务宿主

显式服务宿主负责创建、持有和回收一个项目的控制器。watchdog 是短时客户端：
它向仍在运行的宿主请求交接，后继控制器继续由同一个宿主持有。
这避免把后继放在即将退出的 watchdog 生命周期内。

本入口已完成隔离 CPU 产品验证。实际 systemd、GPU、旧服务迁移、冷恢复与生产切换
尚未验收；本功能不自动迁移已有全局服务。

## 配置与管理

项目需要支持交接的显式控制器声明，例如
[CPU 交接配置](cpu-controller-handoff.md)。使用新的执行作用域，提前创建 results
目录和服务配置的父目录。服务配置放在 results 之外；运行环境中应已安装可直接导入的
Core／Pro 包，安装过程会固定解释器和包内容身份。

以下命令会创建配置及该项目的三个 user unit，并立即启动服务：

```sh
orze service install -c /absolute/project/orze.yaml --method systemd --service-config /absolute/project/service.json
orze service status --service-config /absolute/project/service.json
orze service audit --service-config /absolute/project/service.json
orze service logs --service-config /absolute/project/service.json
```

同一个配置路径选择同一组 main service、watchdog service 和 timer。不同项目使用不同
配置路径；不要移动或改写正在运行的配置。配置与 unit 文件必须不存在，安装不覆盖已有文件。
新入口不支持 crontab。省略 `--service-config` 仍指向旧全局入口。

向宿主请求交接或停止：

```sh
python -m orze.service.host --service-config /absolute/project/service.json --operation status
python -m orze.service.host --service-config /absolute/project/service.json --operation restart --request-id handoff-001 --source-controller-id CURRENT_ID
python -m orze.service.host --service-config /absolute/project/service.json --operation stop
```

`CURRENT_ID` 来自该宿主的 status 输出。使用宿主入口，让它持有实际后继子进程。
同一个请求标识可复核已有交接结果，不重复启动。最后回复丢失时，宿主保留原请求；
后续 watchdog 重试原请求，其他请求被拒绝，直到结果能够核实。

卸载能够确认关闭的服务：

```sh
orze service uninstall --service-config /absolute/project/service.json
```

宿主仍在运行时，卸载核验三个 unit 的当前配置，再请求注册停止，复查后禁用、删除
选中的 unit。宿主已写入完整关闭记录并退出时，入口只读核对原配置、运行包和研究记录，
还须确认三个 unit 都处于 inactive、没有剩余主进程或 cgroup，且有效配置与生成内容一致。
这一路径仅禁用启动链接、删除选中的 unit，不根据存盘 PID 发送停止命令。

服务配置、宿主登记、关闭记录及研究状态保留。任一记录或有效 unit 状态无法核实时拒绝继续；
不要将删除登记目录当作恢复办法。停止后重新安装及冷恢复仍待完成。

## 研究任务自然结束

控制器按策略正常结束并完成结算后，宿主保留该子进程的原始内核句柄，status 返回
`state: stopped` 与核实后的关闭记录。宿主保持运行，watchdog 不会因此重新启动研究；
`stop` 可关闭宿主。没有完整 ACK、存在未结算记录或进程身份不属于该宿主时，不报告完成。

初始控制器在首次 status 前结束，或后继在 STARTED 回复送达前结束，也可核实。
已完成交接的原请求仍能重放确认，回复丢失后不会启动第二个后继。新的 restart 请求不能
让已经自然结束的项目重新开始。宿主退出后，service status 可报告已核实的关闭记录；
这只是记录核验，不代表重新取得进程控制权。

## 生命周期与确认

宿主的 `service.json.host.lock` 在首次创建子进程前保存启动意图，不按年龄回收。
客户端通过内核进程凭证和捕获的进程身份确认宿主，不用 PID 文件授予控制权。
初始子进程加载真实项目配置后，还须取得宿主的输入确认，才进入普通控制器初始化。

后续交接沿用控制器协议：旧实例停止自有工作、完成结算、关闭数据库、写 ACK，
宿主确认旧进程退出后，后继才可消费一次性授权。预算和研究历史不因交接重置。
宿主收到停止信号后关闭新工作准入，已经创建的子进程仍被记录和跟踪。

systemd 模式还核验宿主 MainPID、管理器 cgroup、三个 unit 的目标、启动命令、环境及
额外生命周期命令；安装后等待认证 status，再启用 timer。有效配置核验是逐次观察，
不是对同 UID 任意恶意并发改写的原子隔离。实际管理器行为和特殊路径的属性显示仍待部署验证。

本轮 CPU 产品采用 `method: process`，直接运行同一宿主及真实 CLI；管理命令使用隔离替身。
旧的未注册 watchdog 路径仍存在，不能据此宣称全局 PID 所有权问题已解决。
初始实现见[服务宿主验证记录](plans/2026-09-16-service-host-results.zh-CN.md)，
自然结束和已关闭服务管理见[生命周期验证记录](plans/2026-09-16-service-lifecycle-results.zh-CN.md)。
