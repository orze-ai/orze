# 已关闭 CPU 服务的运行包切换

`orze service upgrade` 为已关闭的 CPU 服务准备一个使用当前运行包的新宿主。
旧、新运行包可以不同；项目配置、Python 解释器、工作目录、results 和数据库保持原来的
路径与文件身份。原生历史、累计预留和执行代次继续使用原库，不从空库重新开始。

当前适配器是 `local_cpu_state_v1`：读取并验证现有 `local_cpu_handoff_v1` 的关闭状态。
它不做数据库格式转换，也不从恢复目录启动。升级准备成功不表示生产部署或任意版本组合
已通过验收；实际验证的版本与边界见[本次结果](plans/2026-09-16-service-upgrade-results.zh-CN.md)。

## 准备和启动

保留旧运行包的完整目录，并先在旧环境中通过注册停止流程关闭旧控制器及宿主。
旧管理单元也必须停止并取消自动启动。随后在旧环境生成备份：

```sh
orze service backup create \
  --service-config /absolute/project/old-service.json \
  --destination /absolute/backups/before-upgrade
```

单独保存返回的 `manifest_sha256`。在能导入目标运行包的环境中准备新配置：

```sh
orze service upgrade \
  --source-service-config /absolute/project/old-service.json \
  --service-config /absolute/project/new-service.json \
  --request-id upgrade-001 \
  --backup /absolute/backups/before-upgrade \
  --manifest-sha256 SAVED_SHA256
```

命令只创建新的服务配置，不启动进程、不安装管理单元。目标路径必须不存在；旧服务
配置和关闭记录保留。项目的 `controller_runtime` 声明仍须通过当前版本校验；命令不会
删除或改写它来绕过固定版本要求。需要修改项目配置的切换暂不属于此适配器支持范围。

对于明确使用 `method: process` 的隔离服务，可在同一目标环境启动宿主：

```sh
python3 -m orze.service.host --service-config /absolute/project/new-service.json
```

systemd 的安装与实际切换需要另外受控执行。本命令没有隐式安装选项，不能把一次本地
宿主启动当作 systemd 或机器重启验收。

## 运行包与状态如何核对

源服务的运行包按已保存的根目录、文件数和内容哈希重新读取，不导入目标解释器。
适配器核验旧服务的当前关闭状态，并与固定备份中的关闭证明、服务、数据库、控制器和
包身份匹配。旧服务曾恢复过时，其已完成交接使用当时的实际解释器／CLI 身份核验。
普通 `recover` 仍要求当前运行包一致，不能借本功能默认接受版本变化。

新宿主启动前再次检查源包、备份、停止标记、旧进程缺席、管理单元和历史结算。
授予后继使用既有数据库中的唯一源控制器约束和当前代次事务；两个升级请求，或旧版恢复
与新版升级同时竞争，都只能有一个取得交接。所有新进程握手和执行准入继续检查目标
当前运行包。历史版本匹配只用于读取已经完成的源交接，不是新进程的执行许可。

成功切换后可以按原来的注册停止和同版本 `recover` 流程管理新服务。原预算不会因升级
或后续恢复增加；备份也不会自动包含切换后的新结果和新支出。

## 中断和退回

新宿主尚未取得交接时，可以在保留的旧环境中执行普通 `recover`，继续使用旧运行包。
旧版恢复取得的新代次会使原升级声明过期，之后启动该声明会被拒绝。保留未使用的配置
和备份，不能把它们当作可重复使用的授权。

一旦启动意图已写入，连接丢失、Stop 或其他不确定结果会保留原交接状态并返回 HOLD。
不要删除宿主锁、交接行或 `pending_grant` 来重试；一次调用失败不证明新进程从未出现。

新版已运行之后，退回不理解这套过渡协议的旧二进制仍未实现。恢复旧数据库会丢失
新结果并退还已用预算，因此不受支持。后续完整回滚必须从当前关闭状态进行一次兼容的
反向交接，并保留全部累计成本；本功能不宣称已完成这一验收。
