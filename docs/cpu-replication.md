# V1-06C：显式 CPU 复验

本片延续 [CPU 动作](cpu-actions.md) 和 [领域/策略接口](research-interfaces.md)，沿既有 `orze replicate`、`replication_requests` 表和同一 Orze 主循环接线。复验是一个明确请求，不是去重失效，也不是自动重试未知副作用。本文的机制验收状态以本片机器证据和实施账本为准；不代表整个 V1 已完成。

## 明确请求，不改原任务

先有同一 IdeaLake / results scope 中当前已确认完成的原生 CPU action，再调用：

```sh
orze replicate SOURCE_TASK --request-id repeat-check-001 --reason "复查同一请求" -c /project/orze.yaml
```

所选配置必须显式启用 CPU。CLI 只准入，不启动 worker、不发现 GPU，不要求 CPU 项目提供训练脚本或 base config。普通前台 `orze -c CONFIG` 后续从同一持久队列领取副本并执行。

一次请求生成一个新 task ID，但逐字节复制原始 task config。用途、输入、命令或 Domain payload、种子、超时和输出声明不加盐、不篡改；reason 是请求控制元数据，不改科研 specification。每次实际执行有自己的 attempt、声明产物和 observation occurrence，并重新预留自己的 wall envelope。相同数值和 protocol 不会合并为一个 observation，也不自动证明统计独立性。

原有普通提案去重原样保留：仅更换 task ID 提交相同配置依然不会自动执行，未准入的 Markdown 保留。不提供公开 skip-dedup 标志，也不把复验任务回灌 inbox。协调器只在受验证的事务中调用既有队列/FSM 插入原语。

同 request ID 的精确重放返回同一个 task ID，状态为 `already_requested`；不重置排队、执行中或已完成的任务。同 key 改来源或 reason、损坏/不确定记录不能当作新请求。不同 key 是用户或策略明确请求的不同副本，各自仍受执行准入约束。数据库提交或响应不确定时，应重用原 key；不要通过生成新 key 掩盖不确定性。

## 两种授权不混用

沿用同一请求表及其唯一 request/task 映射，不迁移旧表、不放宽 schema 1 的训练契约。新增 schema 2 明确 `adapter: native_cpu_action`、来源 fullRef 的 `phase: action`，记录原配置字节 SHA、完整来源 row/terminal、产物/观察集合摘要、执行动作 fingerprint、Domain 元数据摘要和原产物绑定。

CPU 没有伪造 `source_file_sha256` 或训练执行身份。无 Domain 的 A 动作显式绑定空 Domain；B 动作绑定已捕获 Domain 声明、实现版本标签、准备动作及来源元数据。启动时重新准备的动作、领域和输出绑定必须一致；领域实现版本切换不能悄悄借用旧请求。实现标签不是源码字节证明，外部可执行文件、导入、网络与环境也没有因此变成 hermetic。

schema 1 请求不能通过给目标贴 CPU kind 来取得 CPU 执行；schema 2 也不能进入训练消费者。历史 request JSON 和字符串 ID 本身都不是执行能力，必须在真实当前 Lake 中核实。

来源必须是当前 TERMINAL/completed、effect 已确认的真实 CPU attempt。允许零产物、零观察，以及显式 valid/invalid/unknown 观察；执行失败或未确认终态不能伪装为完成来源。这里验证有界元数据和确认收据，不读取大产物内容、重新调用领域解释器或裁定科学正确性。若任务分析既有来源，执行时仍重新走 B 的完整来源捕获/验证。

## 事务与动态检查

创建 request 和新 QUEUED task 在同一个已有执行事务中。源任务 effect guard 与目标执行 guard 是两个明确角色；专用 `watch_cpu_replication` 只读核验同数据库中该请求、原始目标配置和当前来源，在登记时、COMMIT 前后及请求返回前复核。它不接受任意回调，也不授予其他 task 的写锁。

前台派发在 reservation/claim 之前核请求；native intent、READY、GO 前与终态发布再次核实并捕获该授权。已经捕获的请求被删、改或无授权任务后来被挂上请求，都不能静默降级为普通执行。合法新 grant 不绕过 controller Stop、CPU 槽或 wall allowance。额度不足的副本留在队列，不获得免费执行。

READY 后授权撤销不得发送 GO；执行后的授权变化或终态事务故障保持 HOLD，不发布成功、不退还未确认 reservation，不自动重跑。仅看 SQLite 事务内的一次读回不足以确认持久提交。没有承诺对任意外部副作用 exactly-once，也没有自动接管失联 worker、重启恢复或未知结果裁定。

## 策略选择复验

本地注册 Policy 可以在当前捕获的 `recorded_evidence.results` 中选择一个 completed action occurrence：

```python
return {
    "kind": "Replicate",
    "source_ref": result["ref"],
    "request_id": "stable-repeat-key",
    "reason": "复核当前已记录结果",
}
```

这四个字段必须精确匹配合同。fullRef 包含 task/phase/attempt/generation，不能只凭 task ID 偷换来源代际；策略回调修改收到的数据，也不能扩张原捕获集合。协调器在实际准入事务中再次核对这个预期 fullRef。

Replicate 只生成持久排队任务，随后记录带未来 wakeup 的 Wait；下一轮再由 Policy 选择普通 Execute。相同 key 的幂等重放也会等待，不形成无间隔的重复准入循环。`--once` 若本轮只请求复验，就结束该轮，副本需下一次运行执行。策略/执行路径的读取或授权异常进入 HOLD；显式复验 CLI 准入失败保留既有 JSON 错误与退出码 2，不把错误响应当作终态或回滚证明。

来源、key、reason 和目标关系在请求账本中持久保存；这不是记录任意完整 Policy 内存或决策过程。队列准入不扣一次执行额度，实际执行仍逐次预留。可信 Python Policy/Domain 回调与控制元数据操作并未纳入子进程 wall budget，也没有无限队列/全局内存上限或统一信息增益保证。

## 验收边界

验收区分新增 schema/策略/CPU 入口要求、既有跨 adapter 缺口、实施草稿故障以及新测试前提纠正。真实 CLI 复验覆盖 A 零产物和 B 多观察路径，验证原配置不变、新任务/尝试/记录独立；另有普通去重、请求冲突、额度耗尽、来源/目标变化、READY 和终态故障等负例。重叠套件和重放不累加为新增用例或额外缺陷。

机制正确性不等于科研收益提高；本片不优化 ASR，不运行 GPU/付费 provider 或线上实验。两类异构领域以及接口冻结后的独立留出验收仍须单独完成，留出内容在本片尚未读取。
