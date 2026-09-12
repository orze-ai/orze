# CPU 运行期租约

新 native CPU action 默认获得不可续期的运行授权，期限为动作的 `timeout_seconds`。计时从持久 INTENT 内开始，包含工作目录、READY/GO 准备和结果解释、发布时间；不再只从 GO 后计时。

可在 CPU 项目配置中缩短期限：

```yaml
cpu_runtime_lease:
  version: 1
  ttl_seconds: 2
```

TTL 必须为正有限数，不能超过所派发动作的 timeout。缺省仍启用默认期限；显式 null、false、0 或未知字段是错误，不是禁用开关。配置绑定调用指纹，不能热重载；它不改变预算 namespace、动作内容或去重身份。

授权使用同一主机、同一次启动的 Linux CLOCK_BOOTTIME，包含系统挂起时间。原监督器使用严格 v2 READY/CLOSED 协议，自主检查期限并停止自己拥有的进程树；父端不轮询不再使 worker 无限运行。旧非 CPU 监督协议保持 v1。此机制不是沙箱、分布式接管或硬实时调度保证。

到期后的返回码 0 也不是成功。已有 RUNNING owner 只有在真实整树闭合后才能写入 `interrupted / cpu_runtime_lease_expired`，不登记成功产物或观察；确认终态后释放槽位，已预留 wall 额度不退还。自然结束但结果解释或消费晚于期限，同样不得正常发布。closure 的真实 STOP 标记保持不变，terminal 的独立租约元数据解释这类中断。

如果到期发生在 effect prepare、SQL commit 或回执确认过程中，保留真实 poststate 和 HOLD，不改写第二个终态、不自动重跑。监督器丢失、时钟/boot 不明、身份变化也不能靠到期解除未知状态。确认后的历史终态只按存档协议恢复结算，不恢复旧执行权；新旧版本均保留原 effect guard 和恢复屏障。

实施与验收边界见 [C3 合同](plans/2026-09-12-c3-runtime-lease.zh-CN.md)。不代表 GPU/provider、线上部署或研究收益已验证。
