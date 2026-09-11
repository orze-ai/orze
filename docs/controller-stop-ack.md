# 单机注册控制器的停止 ACK（V1-05C2e2c）

本片把 C2e2b 的内部登记、强成员和探针接入真实 `Orze.run()`；
只开放停止确认，不开放启动交接、重启、接管或跨主机控制。
默认未启用的旧配置继续采用既有停止请求／HOLD 合同。

## 显式支持范围

在已经完整配置的项目中使用以下配置片段：

```yaml
controller_control:
  version: 1
  profile: local_stop_v1
gpu_scheduling:
  allowed_gpus: [0]  # 实际部署必须填写本实例的真实物理范围
role_presets: []
telemetry: false
auto_upgrade: false
max_fix_attempts: 0
bot: null
telegram_bot: null
notifications:
  enabled: false
retrospection:
  enabled: false
metric_harvest:
  enabled: false
substrate:
  elo_ranking_enabled: false
```

这不是一个完整训练项目或 CPU 调度配置。训练入口、数据、预算、角色和其他
项目设置仍须按各自合同提供；声明物理 GPU 不会创造设备或 CPU 执行槽。

直接以前台 `orze -c CONFIG` 运行；此模式不启动默认 admin 线程。
`orze stop -c CONFIG --timeout 60` 与旧形式 `orze -c CONFIG --stop --timeout 60`
共用停止观察者，只有实际验证完成才返回 0，未知状态返回 75。
停止等待时间不再覆盖被冻结的训练 timeout。

首版要求同一工作目录、相同已解析配置、单一本机、持久 IdeaLake、明确 GPU
范围和新执行作用域。允许预置排队条目；已有 execution attempt、旧任务／owner
目录、已知 PID／leader／停止标记不自动导入。没有按 PID 探活后清空历史的捷径。
目录检查不是对宿主机所有写入者的证明；旧版本和其他同 UID 程序不属于该实例的
停止证书，也没有被本协议沙箱化。

`start`、`run-idea`、role-only、独立 admin、远端／容器／fleet、自定义 cleanup
script、隐式修复、运行时 strategy_team 注入不在该模式内。启动期 Lake 迁移或
初始化失败明确拒绝，不降级到无 Lake 继续运行。

## 实际权力来源

1. `ControllerSession` 在 PID 文件、GPU 租约和本实例的第一条探针之前登记。
   登记绑定真实 Lake 对象／连接、数据库与 scope 的 inode、host／boot／自身
   PID 与 start ticks、no-age 所有权目录。失败后不释放或重建未知 owner。
2. 构造期先验证 profile，注册前精确匹配 Loader 指纹，执行期持续重验；公开配置、实际控制路径、
   7 个 `_env_ORZE_*` 路由、物理 GPU 范围、工作目录与会话 holder 都不能静默换代。
   本片新增的会话／成员摘要保存哈希，不保存配置密钥、命令环境或输出正文；
   这不改变既有执行日志、配置产物和其他记录各自的存储合同。
3. 前台主循环继续使用原 scheduler、原执行 attempt、同一个 SQLite 和原 native／
   role 消费者。后台停止检查线程只读请求并请求 QUIESCING、设置唤醒标志；
   它不读取监督协议、不抢收进程消息、不代替主线程提交执行结果。
4. 已固定请求的轮询保持只读，不能反复申请 SQLite 写锁打断 native 收尾事务。
   信号处理器只设置内存状态，数据库操作由停止检查线程在之后执行。

`controller_sessions` 在同一个数据库中保存固定 `binding_json`、单个
`request_json` 和最终 `ack_json`。请求绑定 controller ID、完整 binding 哈希与
随机 request ID。请求／登记／路径／所有权被替换，或存储结果无法核实，都拒绝 ACK。
仍活着的同一实例可以由新观察者重新捕获 pidfd 后继续等待已有的同一请求；
已经退出的实例不能靠磁盘 ACK 被新观察者接管。

## 停止的两个必要条件

控制器的 ACK 必须在主线程完成以下检查之后发布：

- QUIESCING 下，持久成员集合与强私有 owner 集合逐项一致，没有未结束的执行事务。
- 每个成员的 OS 树已真实闭合，或有明确的未执行／无需 OS 证明；动作必须 SETTLED。
  Native 项还须核对完整历史 AttemptRef、终态内容哈希和已经确认的 effect receipt。
  更新 generation 不会使真正已完成的旧成员失去历史资格。
- 角色必须经过原交付消费者、精确锁释放；训练／评估／posthoc 必须经过原 native
  停止发布器。它返回 True、公开 map 为空或 slot 被替换都不能单独提供完成证明。
- 停止检查线程已 join；真实原 Lake 连接已关闭；捕获的 GPU 租约集合、对象和 FD
  witness 不变，实际 close 完成后逐个验证 FD 已关闭；精确自有 PID 文件才可删除并 fsync。
  后加到租约列表的描述符不能被顺便关闭。原 leader 的过期接管／宽松 release 不参与。

ACK 包含请求与 binding 哈希、成员数量及规范化摘要、资源收尾类别。
观察者必须在请求之前捕获并重新核实该控制器的真实 pidfd；看到 ACK 后仍要等
这个 pidfd 可读（旧控制器退出），再重读登记、请求、ACK、成员、native 终态与
effect receipt。`CompletedControllerStop` 是已验证停止的返回记录，不是可重启的能力票据。
普通退出码、stdout、PID 文件、布尔值以及手工构造的返回对象都不会让旧 restart
路径获得新权限。

注册目录与历史记录始终保留。再次启动、`restart`、`--restart`、`--enable` 不会
因此清标记、重新安装包、调用 detached Popen 或重新分配 GPU；有资格的一次性交接
与恢复仍是后续工作。

## 取消与未知必须区分

READY 后的匹配停止只发一次 STOP，不再 GO。Pro 不确认未实际 GO 的激活机会，
保留原交付／预算／角色 holder 给真实闭合消费者。返回码 0 不会抹去 STOP。

对已经自然退出、但 TREE_CLOSED 尚未消费的进程，唯一协议读取者先收取真实帧；
STOP 发送的精确断连竞争也只在收到合法 TREE_CLOSED 且 supervisor 实际正常退出时
收敛，不把 EOF 或发送异常当成闭合证明。

探针只有两种确定取消可以进入 `ControllerQuiescing` 路径：适配器最初准入检查
明确拒绝（尚未建立意图），或已经整树闭合、输出 EOF／关闭、动作 SETTLED 的匹配停止。
`ControllerProbeCancelled` 同时兼容原 `ControllerProbeHOLD` 捕获；它绝不返回一个
正常的空 GPU inventory。溢出、未知监督、丢输出及未核实提交仍 HOLD。

已准入来源的三类失败报告可以在 QUIESCING 下收尾，但必须绑定本 controller
已经结算的精确源 Ref；没有自由的 `cleanup=True` 新工作豁免。
静态 artifact preflight 的 NOT_STARTED 需要 producer 在 prepare 之前发出的
私有证明，缺少 handle 本身不构成未执行证据。

## 有界性与未完成内容

每个实例最多保留 4096 个生命周期成员，包括已结算探针；主循环在达到 2048 时
请求本地停止，为来源绑定报告保留空间，不删除／淘汰未知 owner。
单次长 tick 仍可能触及硬上限并 HOLD。这不是无限运行、历史压缩或自动恢复方案。

该模式绕过旧 startup recovery、过期 leader 接管、内部升级 sentinel、hot reload、
周期／应急清理、housekeeper、legacy SOP `--help` 执行、Pro 自动安装，以及角色入口的
布局迁移／动态角色注入／bot／telemetry bootstrap。显式 native 调度和显式角色保留；
这些可选旧路径没有被“登记了一部分成员”追认为受监督工作。

并非任意时序都保证正向 ACK。特别是意图建立后、真正 prepare 之前的未知取消，
存储争用／失败、丢失监督及未核实动作仍可能要求显式恢复。保守拒绝不是已完成恢复。

验证包含真实 `load_project_config → Orze → run → _run_leased`、真实 CLI stop、
SQLite、CPU worker／逃逸后代、监督协议和隔离临时目录中的真实 flock／FD。
GPU 探测、容量、遥测是明确测试边界，未运行实际 GPU、模型、数据集或付费 provider。
Session 单元测试的简化 host 另列，不冒充产品主循环；真实 CPU 子进程也不等于产品
已经有 CPU executor／通用 autoresearch 循环。整版 V1 和研究效率收益仍未验收。

机器可读结果见 [本片证据](evidence/2026-09-11-v1-05c2e2c-controller-stop-ack.json)。
