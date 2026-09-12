# 自主继续实施与部署边界（2026-09-12）

研究总期限不再是必填项。Core 的持续授权、可恢复 Pause 与 Pro 的证据准入已实现；这不等于任意研究的最优停止器，也不等于生产已经切换。两边沿用原主循环、执行链与持久账本，没有另建 runner 或第二套状态真相。

## 已实现的行为

Core 源码固定为 `6007f6efe49964f14317cd620f0b177615c35da8`。新项目可显式声明：

```yaml
execution:
  version: 2
  resource: cpu
  slots: 1
  wall_budget_seconds: null
```

这是“无累计执行额度上限”，不是把上限写成巨大数字。原 v1 有限额度保持原意，同一持久 scope 不准升级清账。每个动作仍须有有限 timeout、有效 runtime lease 和实际资源许可；已有 operator Stop、存储或执行 HOLD 不会被取消。

Policy 可决定继续工作、Wait、Pause 或 Stop。Pause 必须在确认无活动预约时落入原决策表，只结束这次前台调用、不写永久 Stop。之后再次进入同一项目会沿用原账本重评；本片没有新增后台自动唤醒器，不能把两次独立 CLI 说成已验证自动调度恢复。Stop 是持久结束，队列耗尽不是科学收敛。

Pro 源码固定为 `81f8a52bdca288c8f39ed213cd08887467d52036`。仅在项目显式启用 research 批次决策合同时，父进程在预算处理前读取真实 receipt，区分 Continue / Wait / StopProposing / HOLD，子进程在上下文／模型调用前复查。等待结果、已关闭提案方向或不确定状态不再为重复检查而启动提案调用；已准入实验仍可完成。该门禁不是全局 Stop，也不是费用授权；父子检查不构成横跨预算预约的原子事务。

CPU v2 不自动启用 Pro／GPU 角色；Pro 门禁仍属于原有角色执行链。本片没有把两个执行配置宣称为已统一的模型驱动 CPU 研究接口。

## 已完成的真实执行验证

新 Core wheel SHA-256 为 `9898fd3038fb0a723a7fbbb19aa4b9c027032e89a8b8499606f42c70ee6de43d`。Git archive、wheel 与隔离安装逐字节核对 252 个包文件；沿用锁定的 23 个依赖 wheel，不修改旧环境。

另有[新的 Core＋Pro 配对制品](../evidence/2026-09-12-autonomous-paired-package.json)：固定 Pro 提交构建的 wheel SHA-256 为 `defa50b2cb4c3e526a784d752dbc92de3e48812a3d77ea945c6887e4cb6c23c0`，90 个 Pro 包文件与 Git／安装内容一致。12 条实际构建、离线安装、依赖检查与 Core help 命令全部成功；没有执行 Pro 许可链或 worker。两包仍使用原版本号，选用本轮候选必须核对目录与哈希，不能只看版本号。新配对环境是 `/hot-data/fsx/workspace/erik/orze-autonomous-paired-package-2026-09-12.rq3d9cB2/paired-venv`，不是现网环境。

[完整运行证据](../evidence/2026-09-12-autonomous-runtime.json) 包含 8 次真实 CLI、60 个不同原生 CPU 动作：

- 持续模式：40 个动作，累计预约记账 80 秒，剩余额度为 null，最后由队列 Policy 判断 queue_drained。
- 旧有限模式对照：只执行 2 个动作、累计预约 4 秒，另一个任务仍排队，按原额度规则结束。
- 排序／压缩两个领域的默认与反事实共 4 个项目：均实际执行并独立复验，两个默认项目还执行了 analyze；两个反事实项目各只有 3 个 measure 动作。随后由同一有限示例 Policy 判断 confirmed_selection。不是任意领域或无限候选研究证明。
- 同一数据库／配置／scope 的两次前台调用各完成一个动作再 Pause，累计记账 4 秒、2 个 SETTLED；原记录保留，stop_json 一直为空。

8 个控制器均有 TREE_CLOSED / ECHILD 闭合证明；60 个原生动作均有闭合证明及 SETTLED 结算记录。外层故障看护没有触发。80 秒是保守预约记账值，不是实际消耗 80 秒 CPU。原始材料 524 文件完整归档；首次制品检查脚本错误要求打包未声明前端开发文件，失败与修正均保留，没有改 wheel 来迁就检查器。

## 完整回归与不能忽略的部署问题

[Pro 完整回归](https://github.com/orze-ai/orze-pro/blob/feat/research-production-validation/docs/evidence/2026-09-12-pro-autonomy-full.json)已实跑 1009 项通过，另有[配对 Core 可选集成](https://github.com/orze-ai/orze-pro/blob/feat/research-production-validation/docs/evidence/2026-09-12-autonomy-paired.json) 31 项通过。它们使用明确的测试专用许可替身，不证明真实 Pro 许可有效。Core 第一轮完整回归使用长 CephFS 临时路径，实际为 4459 passed / 97 failed / 7 skipped，失败原样保存于 [首轮报告](../evidence/2026-09-12-core-autonomy-cephfs-full.json)。相同冻结源码的[短 /tmp 完整对照](../evidence/2026-09-12-core-autonomy-ext4-full.json)实际为 **4556 passed / 7 skipped / 2 warnings，861.87 秒**；7 个跳过项为 Core-only 环境没有真实 Pro 的可选导入，另有上述 31 项配对集成核验。两次完整运行前后源码／测试／示例／pyproject 指纹均为 `1f8fa8ba38ab3443c82ac997a200db33e76e3c993b8058e9c426fe8d0b8a0f65`，原测试没有删改。

[97 项逐条归因与真实探针](../evidence/2026-09-12-core-filesystem-independent-review.json)：82 项明确报测试夹具的 AF_UNIX 路径过长；另 5 项有真实原子不覆盖重命名拒绝栈，9 项 GC 和 1 项角色完成为相关路径的间接失败。实际私有目录探针确认：此主机 CephFS 对不存在目标的 RENAME_NOREPLACE 返回 EINVAL，产品 API 因此拒绝；/tmp 的 ext4 成功。短 Unix socket 在两边都成功，超长路径在两边都失败。直接错误、共同调用链推断与对照必须分别看待。

[两轮完整归档的独立复核](../evidence/2026-09-12-core-autonomy-full-archive-review.json)确认 4563 个唯一测试条目完全相同，首轮 97 个失败在短 /tmp 轮全部通过，原 4459 个通过和 7 个跳过状态不变。两份完整日志及 100 份原始领域／恢复 JSON（20,097,880 字节）逐条按原大小和 SHA 核查后归档，无路径重复；这证明记录一致性，不证明 CephFS 兼容。

这不是把首轮错误改成跳过，也不是用 /tmp 的成功掩盖 CephFS 生产限制。当前 CephFS 上的 GC／角色释放流程不能宣称可部署；目标盘必须另做实际能力检查。普通可覆盖 rename 的正控成功不提供安全替代，不能降级原子不覆盖语义。

## 仍未完成的范围

CPU 公共 evidence view 目前最多 32 个结果，尚无长程分页／持久增量摘要；累计账本查询也未证明任意长运行的有界成本。示例 Policy 的有限候选复验不能外推为全局科学收敛。Pro 父子进程配置仍未建立强字节绑定；本片未解决该既有边界。

研究速度的实测仍以 [原 48 次配对结果](2026-09-12-research-efficiency-results.zh-CN.md) 为准；本片没有重跑或改写那轮对照，不声称取消总期限本身能提速，也没有真实付费模型／GPU 收益数据。

尚未指定生产服务与状态目录，未切换全局 Python／服务、修改用户 ASR 项目、启用模型或 GPU 账户、读取生产许可、发布包或合并 main。研究总期限不再是待补输入；生产目标、存储能力、现有所有者闭合与资源授权仍是上线前置条件。
