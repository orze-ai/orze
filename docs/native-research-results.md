# 原生研究角色结果契约（V1-03D）

本项区分角色本次的空输出、拒绝、阻塞、失败及部分提案产出。它不是科学判决，也不证明任务已入库、已执行或测量有效。

## 本次结果的来源

原生 `mode: research` 在预算预约之后、trigger lease/Popen 之前建立结果引用，绑定本次预算 attempt ID、角色名、process nonce 哈希、项目/结果目录及提案源文件。引用随实际子进程环境传递，并由 `RoleProcess` 捕获；结束时不从可热更新的角色配置或共享队列推断身份。

子进程实际 CLI 核实范围后运行同一个研究 cycle。结果使用有界 JSON 发布到 `.orze/state/agent_results/<attempt_id>.json`，包含状态、稳定原因、本批 accepted IDs/数量及拒绝原因计数。写入与关闭临时文件、不可覆盖发布、同步和读回必须成功；发布失败使 CLI 非零退出，不删除已写提案、不退款。

这里的 accepted 仅指通过本批过滤且成功写入提案源，不等于后续持久 admission。源消费时仍可能去重或拒绝。partial 仅指本批部分提案被接受，不能解释成部分测量有效。

## 统一完成结论

| 结果状态 | 正常退出且清理已确认时的操作性分类 |
|---|---|
| accepted / partial | OK；仅计本批 accepted 数量 |
| empty / rejected / blocked | SOFT_FAILURE；不借用其他 writer 的产出 |
| error / 缺失或不匹配回执 | ERROR |

Core 在真实 trigger 终态提交**之前**验证结果并形成结论。Pro 从同一个已捕获进程对象取得结果，记录 `agent_usage.jsonl.native_result` 与角色状态的 `native_result`；不再用共享 inbox 的大小、mtime 或条目差值改判原生角色。

非零退出、明确配额退出、超时、清理不明和终态存储失败不能被“好回执”升级成成功。保留原始结果文件；操作性详情可以说明为什么本次没有获得成功确认。配额分类仍进入原有共享账户冷却路径，不变成普通程序错误。

## 兼容和限制

- `run_research_cycle` 保留整数返回 API，增加可选 `result_out`。独立 CLI 也依据详细结果选择退出码，但没有原生控制器引用时不生成结果文件。
- 旧 generic script 角色保留原兼容检查。原生 research 不再生成旧的逐技能 mtime 输出证明，并清理其临时快照；技能的启动/激活 ACK 仍保留。这些旧证明不能继续被当作原生角色的执行来源。
- 回执最多 64 KiB、512 个 accepted ID，拒绝原因和字段也有界；过大的批次在原生写入前拒绝。没有自动回执清理或全局 exactly-once 外部副作用承诺。
- 当前字符串 provider API 对“真正空响应”与被后端吞掉的传输错误不能可靠细分，明确记为 `provider_response_unknown`。不把自由文本猜作官方 refusal，也不声称已完成所有 provider 的结构化拒绝/截断集成。
- 结果文件是框架子进程的操作性回执，身份绑定不证明提案的科学质量，也不抵御可任意改写控制目录的恶意外部 writer。
- 本项不重建控制器重启后丢失的 `RoleProcess` 引用，不授权自动重放。usage 追加仍是 best-effort，不与 SQLite 终态作跨存储原子提交。接口属于本开发分支，提交和推送不代表已发布。

证据：[V1-03D 结果分类](evidence/2026-09-10-v1-03d-result-classification.json)。最终状态由冻结源码测试、独立复核及远端 ref 核对决定。
