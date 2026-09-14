# CPU Policy 的有界提案历史分页

提案历史读取可以显式加入已有 Policy v2。它只扩展历史回执的可见范围，
不改变正常提案准入、四类持久 outcome、执行权限或累计预算。
未声明此字段的旧 v2，以及 v1，继续使用原来前 32 条的
`recorded_proposals = {results, more_available}` 形状。

先按[接口注册方式](research-interfaces.md)注册受信任的自定义 Policy，再声明：

```yaml
action_policy:
  version: 2
  kind: my_registered_policy
  idle: wait
  wait_seconds: 1
  evidence_page_size: 8
  proposal_page_size: 8
  config: {}
```

`proposal_page_size` 是可选的 exact integer 1..32；null、布尔值、浮点数
或字符串均不接受。不能用于 v1 或内置 queue。上例不自动注册 Policy。
此配置属于 invocation 指纹；不能运行中改变它来重置权限。

## 决策和返回值

第一回调收到首个历史页。下页 token 非 null 时返回精确两字段：

```python
return {"kind": "ReadProposals", "cursor": snapshot["proposal_page"]["next_cursor"]}
```

定向读取同 scope 内的 1..32 个唯一请求 ID：

```python
return {"kind": "SelectProposals", "request_ids": ["request-a", "request-b"]}
```

不接受 Policy 自带的结果、原始 source snapshot 或 SQL 位置。定向结果遵循
请求 ID 列表顺序；缺失 ID 放入 `missing_request_ids`，不合成 rejected outcome。
选择不推进扫描位置，也不消费下一页 token。

`recorded_proposals.results` 仍只包含原始 compact outcome：
`request_id/task_id/status/reason/existing_id`。额外 `proposal_page` 精确包含：

- `schema: 1`、本通道的 `scan_id`、`mode: scan|selection`；
- `page`、`seen`：已扫描页数和历史回执数，定向选择不增加；
- `traversal_end`、`next_cursor`：本读版本的枚举是否结束和一次性下一页 token；
- `missing_request_ids`：定向读取未找到的请求，扫描页为空列表。

扫描按 `request_id` 的 BINARY 顺序，不是时间顺序。尾页只表示本版本枚举结束，
不是科研收敛，也不意味着当前展示包含此前所有页。后续页、定向选择等情况下，
即使 `traversal_end=true`，`more_available` 仍可为 true。

## 与证据读取共享的边界

两通道共享同一个进程、Lake 连接、scope/database 路由和 SQLite 读版本检查，
各自保留有界页面、扫描位置和一次性 token。不能跨通道、跨 invocation、
跨连接或重放已消费的 token。可检测的本连接写入（含回滚）及导致 SQLite
读版本变化的 peer 提交会使扫描 HOLD；无实际变化的同值写入不保证产生新版本。

`ReadEvidence/SelectEvidence/ReadProposals/SelectProposals` 都留在原 CPU
决策循环内，不重复 ingress、不记 Wait、不预约预算、不领取任务、不启动 worker，
也不消费 `--once` 的正常决定机会。正常决定和关闭 invocation 时释放扫描状态。
不存在持久游标、全历史缓存、第二数据库或自动清账。

切到提案读取时，当前证据页的结果和 unavailable 条目以完整 action Ref 重新交给
原 `EvidencePager.read(refs=...)` 核验；它不消费证据扫描 token，返回的
`evidence_page.mode` 如实为 `selection`。空证据页只复验共享 guard。
切回证据读取不推进提案通道；历史页只在共享读版本仍有效时保留。

历史回执完整校验原封印格式与 scope/database，但不要求旧来源或目标仍处于
当初状态，也不调用 Domain。历史 inserted、already_present_exact、
config_duplicate、conflict 都不是当前输入授权或 source ACK。
正常 Propose 仍从回调之外的私有证据快照选源，并重新核验 actual current Ref、
内容、effect 和原事务边界。相同 request ID 改内容仍 HOLD，不改写原回执；
查询损坏或失败也不产生拒绝记录。Stop、HOLD、单次租约和预算不退款语义不变。

## 有界不等于任意载荷都能显示

两个 page_size 都只是记录数上限。完整回调 snapshot 仍受原来的 64 KiB /
2048 JSON 节点等约束；两页与队列相加过界会 HOLD，不扩大限制或默默删除结果。
读取器不持有长时间 SQLite 读事务，不重验每条历史来源的大文件内容；
选中来源的实际准入检查仍另外执行。

本片不证明长期科研收敛、恒定全循环成本或所有读取不确定性可自动恢复。
累计预算读取、首次 ingress 的历史成本及持久研究摘要仍是独立问题。
