# 无预设研究期限：自主继续、暂缓与结束

用户要求：没有研究总时限，由 Orze 自己决定。总期限不再作为部署或启动研究的必填输入；这不授权未指定的 GPU/provider 无限花费，也不取消单个失控动作的终止机制。

本片分三项实施，沿用现有主循环和账本，不另建 runner：

1. CPU 显式 `execution.version: 2`、`wall_budget_seconds: null` 表示持续授权。保留有限 slots、每 action 有限 timeout/runtime lease、同一持久 scope 和累计 reservation 审计。v1 原样兼容；不能将同一旧 scope 改成 v2 清账、用巨大数伪装持续授权或为 unknown 执行退款。
2. Policy 增加 `Pause`：当前 scope 无活动预约时，可持久记录暂缓并退出本次前台调用，不写永久 Stop；后续调用根据新的事实重评。`Wait` 是本次调用内的暂时等待，`Stop` 是持久结束，operator Stop 和存储/执行 HOLD 仍独立。Pause 不是解除任何 guard、占用或 Stop 的权限。
3. Pro 在启动 research 提案器、申请预算之前读取既有批次合同事实，投影为 Continue / Wait / StopProposing / HOLD；子进程内保留重复检查。待结果、已验证关闭的方向、未确认状态不继续空跑 provider。StopProposing 仅停止该提案方向，已准入实验照常 drain；不写全局 Stop，不把空响应或基础设施失败当科学收敛。

验收采用新测试实际 red→最小实现→原样 green，保留旧有限模式、并发/事务/执行生命周期回归。CPU 另在新的私有项目运行真实非 pytest、无 GPU/provider 的前台 CLI，验证持续账本、由 Policy 自主 Stop、Pause 后同账本重新评估。外层测试进程的故障看护期限不写进研究合同；实际研究结束原因必须来自运行记录。

科学判断仍由选定 Domain/Policy 和预先声明的研究合同负责：继续探索、分析、复验与结束要能指向实际证据，不用固定轮数或计时器假称收敛。当前 CPU 公共 evidence view 最多 32 个结果；本片不把截断视图视为全貌，未实现分页/持久增量摘要之前不能声称任意长程证据推理已解决。Pro 的 typed gate 也不是跨所有研究方向的全局最优停止器。

当前生产目标尚未明确；本片不修改既有 ASR 工作树、全局服务、模型账户或授权缓存，不发布包或合并 main。
