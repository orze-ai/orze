# Proposal 候选索引：同一写事务中的有界两组读取

本片基于 Core `c17ce2e8573bd14d0153cb7e2f030914e57a36fd`、Pro `6c12fc898ec0b71b93f61240fdc10a1ea6f5899e`。只改变 Core 正常 admission 的候选读取；Pro 产品源码和既有测试不变。它把前片只读 SQL 探针推进到实际 ingress，继续降低 P1 长历史成本。

## 最终实现

原 OR 查询只能利用复合索引的 status 前缀，每次提案仍扫描大量无关历史。现在在同一个写事务内读取两组：匹配 hash 的候选，以及缺失身份、且未落入第一组的候选。分别使用已有 hash 索引和缺失身份索引，不改 schema、不强制指定索引名称、不增加 cache。

两组共享至多 **1,025 条返回记录**的额度。第一组已超 1,024 时直接保留原容量拒绝；否则第二组只读取余下额度。总量超限时在解析任何历史记录前拒绝。只有总量未超限、两组均完整时，才合并各自 rowid 顺序，保持首个语义匹配者及前置不可用记录的处理。每条候选仍受原 64 KiB 候选配置读取检查约束。

普通 `insert(if_absent=True)` 的 writer 和调用者自有事务入口保持不变；新增实际第二连接检查确认两次查询之间不能提交配置变更，writer 关闭后下一次入队则能看到该变更。状态／kind、当前 YAML、派生身份校验、同 ID 全部源字段、源锁／ACK、预算／permit、执行和恢复代码均保留。正常情形候选 SELECT 从一条变为两条，溢出首组时只有一条；逐提案事务数量没有减少。

返回记录有界不等于每次总 I/O 有硬上限：数据库仍可能读取和排序大量相关候选，缺失身份或失去索引时仍可能昂贵。ingress 之前的缺失 hash 准备也仍可能扫描／修补历史。

## 经过对照放弃的两版

- 最初直接 `UNION ALL` 的分支查询通过业务检查，但 5,000 条全部为候选时约 **450.34 → 556.30 ms**，慢约 24%。保留[源码](../evidence/runs/2026-09-15-dedup-candidate-query/before-branch-limits/proposal_admission.py)和[原件](../evidence/runs/2026-09-15-dedup-candidate-query/benchmark-v1.json)。
- 给每个 SQL 分支单独加限额后，5,000 条超限场景改善，但 1,000 条有效候选约 **292.36 → 376.72 ms**，慢约 29%。保留[源码](../evidence/runs/2026-09-15-dedup-candidate-query/nested-branch-limits/proposal_admission.py)和[原件](../evidence/runs/2026-09-15-dedup-candidate-query/benchmark-v2.json)。
- 最终采用同一 writer 内两组读取、共享额度和有序合并。前两版的语义检查没有被冒充为最终源码的完整回归。

## 最终同机交替测量

每个独立项目一批 128 条保留源提案；旧版为本片基线 admission，新版为最终 admission。真实 source 锁、读取、解析、SQLite writer 和 ACK 判断均经过原 ingress。两轮各自重建数据库，预热后新旧交替各七次；源内容、返回内容与数据库逻辑摘要始终相等。合成 completed 行不是实际 worker 证据。

| 历史／候选情形 | 第一轮旧 → 新，中位 ms | 第二轮旧 → 新，中位 ms |
| --- | --- | --- |
| 1 条历史，一个相同文本匹配 | 110.59 → 110.24 | 110.23 → 113.20 |
| 100 条历史，一个相同文本匹配 | 116.57 → 113.90 | 114.80 → 112.22 |
| 1,000 条历史，一个相同文本匹配 | 170.77 → 115.88 | 170.83 → 112.54 |
| 5,000 条历史，一个相同文本匹配 | 424.96 → 113.45 | 406.97 → 97.95 |
| 1 条历史，一个仅语义相同匹配 | 130.95 → 134.65 | 114.08 → 118.25 |
| 5,000 条历史，一个仅语义相同匹配 | 451.31 → 136.92 | 429.32 → 118.96 |
| 1,000 条全部为候选，保持重复拒绝 | 311.56 → 316.54 | 291.94 → 298.62 |
| 5,000 条全部为候选，保持容量拒绝 | 455.95 → 320.94 | 439.58 → 304.19 |

5,000 条稀疏、相同文本候选的完整 ingress 批次约下降 73%／76%；语义匹配约下降 70%／72%。小规模和 1,000 条密集候选仍有约 2%–4% 的轻微退化，不能宣称所有规模改善。两轮 1,000 条密集候选的 Python 跟踪分配峰值约 **417 KB → 435 KB**，有所增加；5,000 条容量拒绝约 **1.59 MB → 1.53 MB**。不是 RSS 或总内存硬上限。

两组均保留 128 次 writer／回滚，YAML 解析次数也相同；变化来自候选读取。I/O、独立 SQLite VM 计数、实际查询计划、逐次分布和独立 tracemalloc 原件见[最终第一轮](../evidence/runs/2026-09-15-dedup-candidate-query/benchmark-v3.json)、[最终第二轮](../evidence/runs/2026-09-15-dedup-candidate-query/benchmark-v4.json)。计数／内存跟踪不放入正常计时；warning 输出在两组均抑制。本机私有 `/tmp`，主机非独占，部分测量与其他验证重叠；未测生产共享存储或研究收益。

## 业务、差分与真实执行

旧代码上的 29 项新增业务基线通过；最初 UNION 版定向 136 项通过。最终新增两项查询间实际 peer 写入检查后，冻结输入的 **138 项定向通过，27.24 秒**：[记录](../evidence/runs/2026-09-15-dedup-candidate-query/targeted/run.json)。覆盖 rowid 顺序、重叠／null／blob 身份、当前 YAML、状态、不可用记录、容量、缺失索引和调用者事务回滚。

独立差分脚本在 64 组带固定种子的混合历史上调用普通入口及调用者自有事务入口，形成 128 组新旧对照、256 次实际 admission 调用。最终结果、完整数据库逻辑投影和回滚结果均相等：每种入口各有 31 个 duplicate、28 个 unavailable 拒绝、5 个新入队。固定时钟只用于比较合成数据库；没有伪装成执行记录。保留[初版差分](../evidence/runs/2026-09-15-dedup-candidate-query/differential-v1.json)和[最终差分](../evidence/runs/2026-09-15-dedup-candidate-query/differential-v2.json)。

另在新项目复用现有真实 CLI 脚本：第一轮从 sidecar 接纳一个真实排序 CPU 动作，另一 ID 的相同配置保留为 duplicate；第二次启动重放不增加 ideas／attempt／reservation／artifact。原 sidecar 保留，真实产物 `[1, 3, 5]`、TREE_CLOSED／ECHILD_WALL、TERMINAL 与 SETTLED 哈希均由脚本核验：[产品原件](../evidence/runs/2026-09-15-dedup-candidate-query/product.json)。没有调用模型、使用 GPU 或切换服务。

## 完整回归与保留的运行问题

- Core 首次全量：**4,836 passed、4 failed、7 skipped、2 个既有 warning，1,124.33 秒**；源码／测试输入指纹一致，但受到下述中断请求影响：[原始记录](../evidence/runs/2026-09-15-dedup-candidate-query/core-full/run.json)、[日志](../evidence/runs/2026-09-15-dedup-candidate-query/core-full/stdout.log)。三个失败标记被单独复现为测试临时目录太长，Unix socket 在产品启动前绑定失败。首次定位得到 **3 failed、1 passed**；改用短临时路径后相同四项 **4 passed，7.04 秒**。产品源码和业务断言均未修改，原[失败日志](../evidence/runs/2026-09-15-dedup-candidate-query/failure-reproduction.log)与[短路径复跑](../evidence/runs/2026-09-15-dedup-candidate-query/failure-reproduction-v2.log)保留。
- 为提前定位曾向确认为本任务的 pytest 子进程发送一次 SIGINT；进程继续运行，随后第四项在最初 iteration 报 `execution: controller is stopping`，没有到达该测试准备注入的 rollback。保留[中断记录](../evidence/runs/2026-09-15-dedup-candidate-query/interruption-note.json)，不把首次运行称为未受中断影响的全量回归。
- Core 短路径完整重跑：**4,840 passed、7 个可选 Pro skipped、2 个既有 warning，1,003.41 秒**，输入冻结且未发送中断：[完整记录](../evidence/runs/2026-09-15-dedup-candidate-query/core-full-v2/run.json)、[日志](../evidence/runs/2026-09-15-dedup-candidate-query/core-full-v2/stdout.log)、[JUnit](../evidence/runs/2026-09-15-dedup-candidate-query/core-full-v2/junit.xml)。
- Pro 全量：**1,066 passed，285.69 秒**，两仓输入冻结。
- Core／Pro 可选配对：**31 passed，0.80 秒**，两仓输入冻结。私有原件固定提交 `83e1bfdcf8a20a599866b7dcb832c55fa1a345b7` 的[13 份 SHA256 索引](../evidence/2026-09-15-dedup-candidate-query-pro-private-index.json)逐项匹配 Git 内容；集合重叠，不能相加。

[七个归档和逐文件索引](../evidence/runs/2026-09-15-dedup-candidate-query/archives.json)保留全部四轮成本项目、两轮差分项目和真实产品，共 759 个普通文件。完整回归、被放弃源码、原始日志和脚本另存本片目录。[独立进程机械核验通过](../evidence/runs/2026-09-15-dedup-candidate-query/verification.json)：278 个其余 Core 源码、512 个旧 Core 测试、全部 91 个 Pro 源码及 141 个 Pro 测试匹配基线；资格循环与其余产品 AST 保留，四组最终冻结回归、首次失败回归、759 个归档原件、两版差分及实际 worker 的 ref／关闭／结算／产物核对一致。它不是另一位作者的审查。

## 尚未关闭

稀疏且已有身份的候选检索不再随无关历史逐条扫描，但缺失身份修补、相关候选排序、逐条事务和 sidecar 目录／前缀／全循环扫描仍有成本。Pro 其他 consumer、全局 qualification 基线、evidence／proposal／预算联合载荷、持久记忆、真实模型／许可、CephFS 角色／GC、受控上线及跨领域研究验证继续待办。局部检索改善不等于整个 P1 或产品目标完成。
