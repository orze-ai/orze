# ingress 重复配置：复用当前相同文本的解析结果

本片基于 Core `080edd8199e5772353f1c7e03924ff7dc1e13b13`、Pro `d25754a33884e1d640930d84f2c0674f777d7c68`。只在 Core `_dedup_owner` 增加当前配置文本相等时的分支；Pro 产品源码和既有测试不变。继续降低上一片 128 条真实重复配置对照的成本，未完成全部 P1。

## 改变与保留的约束

提案准备阶段已解析 YAML 并计算配置身份；调用者自有事务入口还会重算、核验传入身份。当前写事务读出的已存配置与这份文本完全相同时，直接复用准备结果，不再解析第二次。没有跨提案／跨事务 cache。

SQL 当前状态与任务种类过滤、候选数量上限、先前不可用记录的拒绝、首个语义匹配者、同 ID 完整源身份判断、逐条 `BEGIN IMMEDIATE` 和源文件 ACK 保持原样。仅格式不同但语义相同的 YAML 继续解析。数据库、预算、锁、permit、执行、恢复和 schema 代码没有修改。

这项修改是重复工作的消除，不是旧业务错误的修复。新增业务检查在修正夹具后也通过旧版：同批次另一连接改变配置／状态／kind 后仍重新裁定；1,024／1,025 候选边界；前置 blob／超限配置；不同表示的首个语义匹配；调用者伪造身份仍拒绝。

## 成本与负对照

先保留[剖析摘要](../evidence/runs/2026-09-15-dedup-exact-config/profile-summary.txt)及 pstats。该剖析包含旧 ingress cache 对照和 profiler／tracemalloc 开销，只用于定位重复解析；不以其中耗时计算收益。

正式测量使用相同新 ingress、相同源锁和数据库，两组仅切换已归档旧 admission 与新 admission。每个项目一批 128 条跨 ID 重复提案，历史中只有一个匹配者。其余记录的身份已准备好。两轮在不同目录重建输入，预热后新旧交替各七次；源与数据库逻辑摘要始终不变。

| 历史条数／配置表示 | 第一轮旧 → 新，中位 ms | 第二轮旧 → 新，中位 ms |
| --- | --- | --- |
| 1／完全相同文本 | 111.69 → 94.17 | 136.91 → 118.80 |
| 100／完全相同文本 | 119.00 → 99.77 | 142.07 → 123.99 |
| 1,000／完全相同文本 | 173.08 → 154.47 | 196.27 → 177.58 |
| 5,000／完全相同文本 | 427.00 → 406.76 | 446.82 → 431.35 |
| 1／仅语义相同 | 113.70 → 113.78 | 136.21 → 135.60 |
| 5,000／仅语义相同 | 428.53 → 428.80 | 445.87 → 446.10 |

128 条完全相同配置的 YAML 解析 **384 → 256**，两组各保留 **128 次写事务／回滚**。一条历史的局部耗时下降约 16%／13%；5,000 条时只有约 5%／3%。不同文本的负对照几乎无收益，部分样本略慢；SQL 随历史增长的成本仍在。该数据库的[当前查询计划](../evidence/runs/2026-09-15-dedup-exact-config/query-plan.json)只利用索引的 status 前缀并为 rowid 顺序排序，不能把本片解释成已消除候选历史扫描。

第一轮一条历史的独立 Python 跟踪分配峰值为 244,823 → 235,018 字节；100 条时反而为 236,180 → 242,366 字节，不能宣称内存普遍下降。两轮耗时有主机负载差异；目录为本机私有 `/tmp`，主机非独占，部分测量与回归重叠；逐条 warning 输出在两组均抑制。原始[第一轮](../evidence/runs/2026-09-15-dedup-exact-config/benchmark-v1.json)、[第二轮](../evidence/runs/2026-09-15-dedup-exact-config/benchmark-v2.json)保留每次样本、I/O、内存和独立计数。合成 completed 历史不是实际 worker 证据，解析次数和局部耗时也不是科研收益。

## 验证与保留的失败

- 首次新增夹具漏填 `ideas.raw_markdown`，旧版检查得到 **4 failed、6 passed**，首次新代码定向检查得到 **4 failed、103 passed**。保留[原夹具](../evidence/runs/2026-09-15-dedup-exact-config/test-script-v1.py)、[旧版失败](../evidence/runs/2026-09-15-dedup-exact-config/baseline-tests.log)和[首次定向失败](../evidence/runs/2026-09-15-dedup-exact-config/targeted-v1.log)；只补充新增夹具必填字段和简短参数 ID，原有测试不变。
- 恢复原产品源码后，修正夹具的旧版 **10 passed**：[基线日志](../evidence/runs/2026-09-15-dedup-exact-config/baseline-tests-v2.log)。优化版冻结输入的定向集合 **107 passed，21.70 秒**：[完整记录](../evidence/runs/2026-09-15-dedup-exact-config/targeted/run.json)。
- Core 全量：**4,809 passed、7 个可选 Pro skipped、2 个既有 warning，950.16 秒**；输入冻结：[完整记录](../evidence/runs/2026-09-15-dedup-exact-config/core-full/run.json)、[日志](../evidence/runs/2026-09-15-dedup-exact-config/core-full/stdout.log)、[JUnit](../evidence/runs/2026-09-15-dedup-exact-config/core-full/junit.xml)。
- Pro 全量：**1,066 passed，284.12 秒**；运行前后两仓输入指纹一致。
- Core／Pro 可选配对：**31 passed，0.82 秒**，两仓输入冻结；原件留在私有 Pro 仓，固定私有提交 `6c12fc898ec0b71b93f61240fdc10a1ea6f5899e` 的[13 份原件 SHA256 索引](../evidence/2026-09-15-dedup-exact-config-pro-private-index.json)逐项匹配 Git 内容。集合重叠，不能相加。

复用现有[真实 CLI 验证脚本](../evidence/checks/2026-09-15-ingress-product.py)，在新目录执行两个全新 CLI 进程：第一轮从 sidecar 接纳一个真实 CPU 动作，拒绝另一 ID 的相同配置；第二轮重启重放不新增任务／attempt／reservation／artifact。源 sidecar 完整保留，实际排序产物为 `[1, 3, 5]`。一个真实 worker 的 TREE_CLOSED／ECHILD_WALL、TERMINAL 与 SETTLED 哈希已由脚本核对：[产品原件](../evidence/runs/2026-09-15-dedup-exact-config/product.json)。没有使用 provider、GPU 或现有服务。

[归档索引](../evidence/runs/2026-09-15-dedup-exact-config/archives.json)保存剖析项目、两轮性能项目及真实产品项目共 60 个普通文件，另保留失败日志、源码基线、测试输入、运行器指纹与回归原件。[独立进程机械核验通过](../evidence/runs/2026-09-15-dedup-exact-config/verification.json)：产品 AST 只新增当前文本相等的分支，278 个其余 Core 源码、511 个旧 Core 测试、全部 91 个 Pro 源码和 141 个 Pro 测试匹配基线；四组冻结回归、60 个归档普通文件及实际 worker 的 ref／关闭／结算与产物均一致。它不是另一位作者的审查。

## 后续

逐条事务、候选历史扫描、sidecar 目录／前缀／全循环扫描继续待办。另留一个[只读 SQL 改写探针](../evidence/runs/2026-09-15-dedup-exact-config/next-query-probe.json)：同一个 5,000 条合成历史中，把已匹配 hash 与缺失身份候选拆成不重叠的两支，可利用现有两个索引。七次交替、每次 128 次纯查询的中位耗时为 296.50 → 2.47 ms；单个数据集的结果相等。它没有进入本片产品代码，也未包含 source／锁／事务／解析或坏输入与并发资格验证；不是完整入队性能或等价性验收。后续必须验证 null／损坏身份、重复分支、status／kind、rowid 顺序、容量拒绝和同事务变化，再考虑采用。

Pro 其他 consumer、全局 qualification 基线与联合载荷、持久记忆、真实模型／许可、CephFS 角色／GC 兼容、受控上线和跨领域端到端研究验收仍未关闭。没有发布安装包或切换生产服务。
