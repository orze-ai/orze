# V1-06B：同一执行链路中的领域、策略与观察

本片沿 [CPU 动作入口](cpu-actions.md)继续接线：仍是 `orze -c CONFIG`、同一个 `Orze._run_leased` 主循环、同一 IdeaLake、CPU 额度与原生监督器。没有第二个 runner、后台服务或观察数据库。本文只描述 V1-06B，不代表整个 V1-06/V1-07 已验收，接口尚未作为留出验收方案冻结。

## 选择与职责

不配置 `action_domain` 时，V1-06A 的七字段 `action` 任务及默认 queue 路径继续工作。显式领域模式使用另一种任务声明 `domain_request`，不能在同一任务混放两个入口。

```yaml
execution:
  version: 1
  resource: cpu
  slots: 1
  wall_budget_seconds: 60
action_domain:
  version: 1
  kind: json_observations
  config: {}
action_policy:
  version: 1
  kind: queue
  idle: wait
  wait_seconds: 1
```

Core 负责身份、准入、执行边界、来源、事务与产物。Domain 的 `prepare(request, sources)` 物化实际 command 动作，`interpret(prepared, envelope)` 解释真实输出并返回显式 tuple；Policy 的 `decide(snapshot, budget)` 选择现有队列任务、等待或停止。接口实际参与 CLI 派发与收尾，不只是类型声明。

通过 `orze.core.research_interfaces.register_domain(name, implementation_id, factory)` / `register_policy(...)` 注册本地 Python 实现，再在同一应用启动代码中调用既有 `orze.cli.main()`。领域 factory 接收脱离的 `config`，策略 factory 接收脱离的完整策略声明；自定义策略可额外声明 `config: {…}`。名称必须预先注册，重复注册拒绝；任务不能指定任意 import 路径。当前不提供自动包发现或插件安装流程。

每次 invocation 捕获选定实现、配置与绑定的方法；排队任务不能替换这些实现。`implementation_id` 是明确的实现版本标签，不是源码字节证明。回调接收有界、脱离的 JSON 数据，不接收 Lake、当前 AttemptRef 的写权限或 Executor。它们是受信任的本地应用代码：这不是 OS 沙箱，也不强制回调纯函数、禁止网络或限制任意 Python 回调的运行时间。CPU timeout/额度约束监督子进程，不包含策略/准备/解释器任意代码的 CPU 时间。

内置 `command` Domain 只执行命令并解释为零观察；`json_observations` Domain 解释显式 JSON 观察声明，不代替领域科学有效性判断。自定义 Domain 不需要用户在任务里写 command，可以根据自己的 payload 物化命令、解释输出格式。实际验收也覆盖了这种自定义实现。

## 通用任务 envelope 与结果

新的任务 config 精确为 `kind: native_cpu_action` 和 `domain_request`。request 精确包含下面七个字段；`payload` 属于 Domain，Core 不按 ASR、训练参数或指标名推断语义。

```yaml
kind: native_cpu_action
domain_request:
  version: 1
  purpose: Analyze two previously produced artifacts
  inputs: {}
  timeout_seconds: 5
  outputs:
    result:
      path: result.json
      max_bytes: 4096
  input_artifact_ids: [artifact-id-a, artifact-id-b]
  payload:
    command: [python3, analyze.py]
    specification: {subject: declared-analysis-subject}
    protocol: {id: arithmetic-v1}
    result_output: result
```

这是 `json_observations` payload 的形状示例，不是可直接运行的项目：需提供真实 artifact IDs 和实际可执行命令。Domain 返回 `{action, observation}`；action 仍必须满足既有七字段 command 契约。准备过程不能修改 request 声明的 purpose、timeout 或 outputs；可以物化 command 和 inline inputs。原始任务配置的字节 SHA、准备动作 specification SHA、领域 subject SHA、protocol SHA 分开绑定，不用其中一个冒充其他身份。

未声明观察输出时必须显式声明 `observation: None`，解释器仍必须返回 `()`。声明观察输出时指定现有 output 的 logical name 及 subject/protocol/adapter 身份，该文件上限不得超过 1 MiB；解释结果也可以为零条。内置 JSON Domain 的输出精确为：

```json
{"version":1,"observations":[{"name":"size","values":{"bytes":120},"validation":{"status":"valid","reason_code":"domain_checked"},"comparison_scope":"roundtrip-v1"}]}
```

`observations: []` 是合法零观察，不是零分；没有声明产物的 command Domain 也合法。缺失、坏 JSON、非法 envelope、解释器返回 None、未声明却产生观察都不能降级为空成功。初始化、策略决策、来源捕获和准备阶段的普通异常通过前台 HOLD 边界返回，不先领取任务或预扣执行额度。真实执行后的解释或发布失败则保留已有 attempt 与 reservation，不发布成功、不退款。

每次最多 32 条观察、名称和 ID 唯一，保留已有单记录 JSON 限额。Domain 声明的 valid/invalid/unknown 及 comparison_scope 原样受契约约束保存，Core 不猜统计等价、独立性或排行资格。执行失败/STOP 是负的执行结果，不自动生成科学观察或伪造数值。

## 来源、观察与事务

来源只接受同一真实 IdeaLake、同 results scope 中当前 TERMINAL/completed 且 effect 已确认的 artifact。逐项核完整 producer AttemptRef、terminal 的 artifact membership、原 spec、完整元数据、路径链、文件内容与身份。不同来源可以有不同 specification；不能伪造一个共同 spec。

最多 32 个来源，实际内容合计最多 16 MiB，持久来源快照最多 32 KiB；数量未超也可能先触及元数据或既有完整 attempt JSON 限额。读取/hash 在 SQLite writer 外，进入 READY/GO 和发布时重新核验。worker 得到独立、真正封印、偏移为零的只读 FD；环境变量 `ORZE_ACTION_SOURCE_FDS` 是 JSON 的 artifact ID→FD 映射。来源 FD 与 `ORZE_ACTION_INPUT_FD` 的 inline JSON 分开，不能把 metadata 路径当作来源读取授权。

观察 schema 2 显式携带 `input_artifact_bindings`，逐 ID 保存原 producer fullRef、原 specification 和内容摘要；该集合与 input IDs 精确对应。schema 1 的单 subject-spec 输入约束保持不变，没有删除旧检查或迁移旧表。观察 subject 与分析动作自己的输出 artifact spec 不要求相同；结果 artifact 必须由本次分析 attempt 实际产生。

只有真实 TREE_CLOSED 后才在 writer 外准备产物、读取有界结果和调用解释器。同一个 effect 事务登记本次产物、完整观察集合和终态，最后以预期集合复核 artifact、observation 和跨任务只读来源。提交前后都复核，确认 effect 后才结算 CPU 槽。跨任务 source watch 不授予其他任务的写权，也不放宽原同 task 的 dependency/artifact watch。

READY 后同内容新 inode 替换、实际注册后 terminal SQL trigger 篡改来源/观察等不确定情况保持 HOLD；不发布成功、不退额度、不自动重做。运行中子进程拿到的是已捕获只读字节，但仍不承诺可执行程序、导入、环境及所有外部资源 hermetic。

## 策略看到什么，仍缺什么

自定义接口模式的队列窗口最多 32 项；策略同时收到 `recorded_evidence`：`results`、`unavailable`、`more_available`。它从当前持久状态读取，不只依赖这个 controller 的内存完成缓存，因此可看到另一 invocation 的已确认记录。失败、interrupted、invalid、unknown 不被压成零分或成功。

这是有界的已记录元数据视图，不执行大 artifact 内容 hash，不授予来源/GO 权限。实际分析仍在派发时执行上述来源检查。单结果过大显式报告 unavailable/更多结果，不截断一条观察来伪装完整；组合策略输入仍受既有 JSON 边界约束，无法完整容纳时拒绝调用，不默默丢必需声明。顺序窗口不是全局知识库、无限历史遍历或统一信息增益估计器。

Execute 只能选择本次捕获队列中的 task；Wait 必须带理由和未来一小时内的 wakeup；Stop 是本预算 scope 持久的停止准入。策略的科研决策意义由实现及证据决定，不由 Core 的角色名、调用数或停滞启发式代替。

本片验证了不同用途的两个分析任务对相同 subject/protocol 产生相同数值时仍保留不同 attempt/observation ID。它们的任务请求不同；不能据此声称相同请求的显式 CPU replica 已实现。仅更换 task ID 的相同配置仍会被既有去重保留在提案源中。下一片继续接显式 CPU replica，然后再做两类异构领域和接口冻结后的独立留出验收。自动恢复/未知副作用裁定、回调资源隔离、按实耗退款、研究收益与线上 GPU/provider 验收均未在此关闭。
