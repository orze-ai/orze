# 长期证据访问与存储准入实施片

延续用户“没有研究总时限，由 Orze 自己决定”的要求；本轮不建立新研究总期限，不改 ASR 项目，不切换生产服务或付费账户。

1. 保留 CPU v1 Policy 的原三字段、前 32 个当前代终态证据契约。显式 Policy v2 增加有界页大小、只读 ReadEvidence 和 SelectEvidence 决策；当前页仍经过 Core 私有捕获，分析／复验仍走原来源内容、代际、effect、预算和执行校验。不会把旧 attempt 恢复为当前来源。
2. 分页 reader 仅由协调器持有，游标为当前进程／连接／namespace 的一次性 opaque token，不是可自行拼装的授权。采用完整当前 Ref 的 BINARY keyset，页间 SQLite data_version／total_changes／schema_version 或路径身份改变就明确 stale/HOLD。重新进入同库需新开扫描，不凭旧游标恢复权限。
3. 每页和定向选择仍有 JSON／条数上限；不累积全部记录进每轮 prompt。尾页只说明本读视图枚举结束，unavailable、超大记录仍保留，不能当作科学收敛或所有历史文件永久有效。定向重读用于少量跨页候选，最终行为前重验来源。
4. 对实际依赖原子不覆盖目录重命名的角色／GC路由，在准入前运行私有能力探针，不等执行后释放才发现不兼容。CPU-only 不因 CephFS 名称被误禁用。保留实际 rename/HOLD，不降级为普通可覆盖 rename，不搬迁用户状态。

验收先保留新测试的真实 red（缺少新 API 与产品行为分别说明），最小实现后同测试 green，加独立对抗测试、原邻域、冻结完整回归及新私有真实 CLI。CephFS 提前拒绝不是已支持 CephFS；分页是当前代终态访问，不是任意历史代的执行权限、Pro 长程规划、持久摘要或全局科学收敛器。Pro 自身研究上下文、CPU proposal 历史查询以及累计预算扫描的长期成本不在本片已完成的承诺中。
