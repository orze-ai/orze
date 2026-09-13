# CPU Policy 的有界证据分页

这是显式的 `action_policy.version: 2`，与控制累计执行额度的
`execution.version: 2` 是两个独立版本。后者允许不设研究总时限；前者解决
自定义 Policy 只能看见前 32 个当前代终态结果的问题。默认 queue 和旧 Policy
v1 不会被自动升级，原 `recorded_evidence` 三字段契约保持不变。

先按[接口注册方式](research-interfaces.md)注册自己的受信任本地 Policy，再声明：

```yaml
action_policy:
  version: 2
  kind: my_registered_policy
  idle: wait
  wait_seconds: 1
  evidence_page_size: 8
  config: {}
```

`evidence_page_size` 必须为整数 1..32，不能用于内置 queue。上例只是配置形状，
不是已注册实现。每次回调仍收到当前页的 `recorded_evidence`，以及额外的
`evidence_page` 元数据：

| 字段 | 含义 |
| --- | --- |
| `scan_id` | 本次只读扫描的随机身份；改变后应重置本次遍历状态 |
| `mode` | `scan` 或 `selection` |
| `page` / `seen` | 已枚举页数 / 当前代终态候选数，包括不可用条目 |
| `unavailable_seen` | 扫描中累计不可用条目数；定向选择不增加它 |
| `traversal_end` | 本读版本的当前代终态枚举结束，不表示科研收敛 |
| `next_cursor` | 下一页的一次性 opaque token，尾页为 null |

需要继续查阅时返回精确两字段决策：

```python
return {"kind": "ReadEvidence", "cursor": snapshot["evidence_page"]["next_cursor"]}
```

只在 token 非 null 时调用。需要同时分析少量跨页候选时，可保存其 fullRef，随后返回：

```python
return {"kind": "SelectEvidence", "refs": selected_full_refs}
```

每个 ref 精确包含 `task_id`、`phase`、`attempt_id`、`generation`；只接受
1..32 个不同的 action ref。下一回调收到 Core 重新核验的选择结果。允许查询尚未
遍历过的 ref，但不存在、非当前代、跨 namespace 或不确定的记录不会成为来源授权。
不接受 Policy 自带的 artifact/observation records。选择不推进或消费扫描游标。

这两类读取不领取任务、不预扣额度、不创建 worker，也不向决策账本追加一次执行。
`--once` 允许先完成只读查询，再作一次正常决定。协调器保留的只是当前有界队列和
扫描位置；不会把所有历史记录放入每轮输入、另建账本或持有长时间 SQLite 读事务。
每页仍受条数、字节和 JSON 节点上限约束，整份回调输入也仍受原有限额约束。

游标绑定当前进程、Lake 连接、数据库与 results 路径身份及 SQLite 读版本。
本连接写入（包括回滚）、其他连接提交、schema 或路径身份变化都会使扫描明确
失效/HOLD，不静默重启后跳过结果。回调之后、任何正常决定之前再次核验读版本。
正常 Execute/Propose/Replicate/Wait/Pause/Stop 后释放扫描；下一轮或新 invocation
从新 `scan_id` 开始，不能拿旧 token 续权。并发执行完成也可能使正在翻页的扫描失效。

`more_available` 仍表示当前展示不完整：即使到了尾页，前面页被省略、定向选择或
超大记录仍可使它为 true。不能把尾页或该标记当作科学结论。失败、interrupted、
invalid、unknown 和 unavailable 原样保留；旧页的文件不会因遍历到尾页而永久有效。
最终 Propose、Replicate、Execute 仍走原来源身份、内容、代际、effect 与预算校验。

## 明确未解决的部分

这不是任意历史代恢复、持久研究摘要或全局信息增益算法。Policy 仍是受信任 Python
代码，不是资源沙箱；读取本身不受 action timeout 约束。Pro 自己的研究上下文、
CPU `recorded_proposals` 前 32 条窗口、累计预算与首次 ingress 的历史扫描成本
尚未在此消除，因此不能声称整个长程循环已变成恒定时间或自动获得科学收敛能力。
