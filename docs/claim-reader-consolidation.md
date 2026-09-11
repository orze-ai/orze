# Claim 读取收敛（S2）

公共入口为 engine.claim_authority.read_claim_snapshot(path, *, limit=65536, required=False)，返回解析对象与同一次原始字节读取的 SHA256；仅最初不存在且允许缺失时返回 None。它是只读元数据快照，不授予执行、发布或恢复权限。

- CPU action 与已确认终态续结算显式使用 8192 字节上限、required=True；两处训练私有读取器依赖消除。
- 旧 read_claim 只投影 dict/None，并保留原 claim_* 的 InDoubt 拒绝；snapshot 的纯校验拒绝为 Busy。这个区别保持 pre-effect 回滚正常释放与持久不确定锁保留的既有语义。
- 同一份有界读取保留原始 JSON 编码/数值/未知字段规则、重复键拒绝、单链接普通文件、无符号链接检查及 FD/path 身份复验。两条路径的严格项合并：legacy 增加 mode，CPU 增加完整长度与已有 claim 路径结构检查。不是所有非法输入异常文案逐字兼容。
- required 缺失及真实 OS 错误保持拒绝，不把中途消失伪装为初始缺失。读后 unlink 会先改变 FD nlink，因而可能先报 Busy；读取前真实 open 失败仍传播 FileNotFoundError。新测试该处的初版错误预期和修正都有记录。
- 不修改训练通用读取器、其余 claim_authority 函数、预算/事务/来源判定、配置或 Pro 实现。没有第三套 reader、新模块、外部依赖或持久协议。

生产源码净增加 25 行，换取显式公共契约和异常兼容投影；不是全仓 reader 减少或整个技术栈代码量下降。旧 read_claim 现在也计算一次最多64 KiB的 SHA；没有声称运行更快或已证明维护/科研效率收益。

只验证固定开发分支及离线测试；没有部署、合并 main、真实 GPU/provider 或新独立留出研究声明。之前不可核验状态的 HOLD、训练 checkpoint 和一般性旧进程接管边界不变。
