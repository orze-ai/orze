# ingress：当前事务去重与 sidecar 分批读取

本片继续机器交接的 P1，基于 Core `ad7ed16632897ab3d4ad62fb717fec53d6795f34` 和 Pro `520971ecf57370a8c3d8d4017da74dca99b88f55`。两仓分支为 `codex/p1-history-validation`。它修复旧配置 cache 错误压制合法提案，并降低大量保留 sidecar 的首批读取成本；全循环重复扫描与 Pro consumer 仍未完成。

## 行为与界限

ingress 不再加载全局配置 JSON cache，也不以派生 hash 查询的返回值直接跳过提案。同 ID 身份和跨 ID 配置去重均交由现有 `IdeaLake.insert(if_absent=True)` 写事务核验当前状态、任务种类与实际 YAML。因此历史配置改变或任务转为 failed／archived 后，旧 cache 不再阻止合法提案；仍为已接纳配置的真实重复提案继续被拒绝，源内容保留。

本片没有修改事务去重、源锁、ACK 发布、预算、permit、执行或恢复模块。旧记录缺失配置 hash 的准备调用仍保留，可能扫描／修补历史，不能把“不加载 JSON cache”解释成数据库操作均与历史长度无关。

sidecar 改为延迟解析。主文件满 128 条时不读取 sidecar；其余页面仅保留所选窗口及必要的前瞻条目。所选候选至多 128 条，原始区块合计至多 4 MiB；每个 ingress sidecar 使用主文件已有的新鲜读取、普通单链接文件、身份复核和 4 MiB 上限。已观察到的符号链接、超限、无法确认的 sidecar 留在原处并记录警告，其他可读文件仍能入队。sidecar 从不被消费删除。

主文件优先、文件名顺序和第一个有效 sidecar 定义优先保持不变。游标只是检查提示；主文件变化重置游标，每次重新读取 sidecar。为在编辑后保持优先级，后页仍会读取、解析前缀，并保留文件名排序／已见 ID 元数据。整轮最坏情况下仍有二次增长的重复工作。这不是每页总 I/O 或总内存硬上限，也没有引入 mtime／配置缓存或持久 iterator。完整 legacy overlay 的其他调用保留原行为。详见[更新后的源交接契约](../proposal-source-handoff.md)。

## 保留的失败与测试迁移

- 初始 9 项新检查运行于旧代码，得到 **8 failed、1 passed**：[基线日志](../evidence/runs/2026-09-15-ingress-bounds/baseline.log)。随后新增并发配置变化、符号链接和读取不确定性检查，新文件最终收集 13 项。
- 新实现首次运行旧成本夹具，得到 **5 failed、62 passed**：[首次日志](../evidence/runs/2026-09-15-ingress-bounds/first-targeted.log)。四项参数化结果仍期待已删除的全局 cache 调用；一项锁丢失注入挂在该旧调用上。迁移只将三处调用计数断言改为零，并将实际锁目录替换注入移至仍存在的定向 hash 查询；锁、数据库和删源业务断言保持原件。原文件已归档，机械检查按严格文本变换核对，未放宽业务结果。
- 第一次独立 CLI harness 错配 `QueuePolicy` 与需要分页策略的 version 2，产品在创建数据库前正确拒绝，随后 harness 读取不存在的数据库失败。保留了[原脚本](../evidence/runs/2026-09-15-ingress-bounds/product-script-v1.py)、[失败日志](../evidence/runs/2026-09-15-ingress-bounds/product-v1.log)及完整失败项目；修正为 QueuePolicy version 1 后在新目录重跑。执行 scope 仍为 version 2 unlimited。

## 回归结果

| 集合 | 结果 |
| --- | --- |
| Core 一次全量 | 4,799 passed、7 可选 Pro skipped、2 既有 warning；949.10 秒 |
| Pro 全量 | 1,037 passed；275.25 秒 |
| Core／Pro 可选配对 | 31 passed；0.90 秒 |
| 定向 | 132 passed；45.34 秒 |

三个完整回归均记录运行前后相应源码／测试指纹，输入一致；Pro 两次运行同时记录两仓。Core 原件：[run.json](../evidence/runs/2026-09-15-ingress-bounds/core-full/run.json)、[日志](../evidence/runs/2026-09-15-ingress-bounds/core-full/stdout.log)、[JUnit](../evidence/runs/2026-09-15-ingress-bounds/core-full/junit.xml)。Pro 日志和源码清单留在私有仓固定提交 `cd4d708`，公开[私有证据索引](../evidence/2026-09-15-ingress-bounds-pro-private-index.json)从该 Git 提交逐项读取并核验 13 份原件。

定向集合覆盖真实不可覆盖入队、锁丢失、ACK、实际重复配置、读取大小、后页到达、并发配置变化和 CPU 产品链：[日志](../evidence/runs/2026-09-15-ingress-bounds/targeted.log)、[JUnit](../evidence/runs/2026-09-15-ingress-bounds/targeted.xml)。不同集合有重叠，不能相加。

[独立进程机械复核通过](../evidence/runs/2026-09-15-ingress-bounds/verification.json)：277 个其余 Core 源码文件、509 个其余既有测试文件匹配基线；成本测试严格匹配上述迁移。Pro 90 个源码文件与 140 个测试文件未变。回归输入、日志、JUnit、五个项目归档共 12,253 个普通文件及其原件哈希均一致，真实 CPU attempt、关闭证明、结算哈希和产物也已关联核验。它不是另一位作者的审查。

## 同机交替成本测量

构造 100／1,000／5,000 条合成历史及同 ID、不同标题的保留 sidecar，每条约 2 KB。旧版完整 overlay 与新版分批读取使用真实源锁和 SQLite 入队路径。每种规模分别测首批和完整遍历，预热后新旧交替各五次；两轮在不同目录重建输入。每次返回相同记录和原始内容摘要，数据库逻辑指纹不变。合成 completed 行并非实际 worker 证据。

| sidecar 数量 | 首批中位 ms，第一轮旧 → 新 | 第二轮旧 → 新 | 完整遍历中位 ms，第一轮旧 → 新 | 第二轮旧 → 新 |
| --- | --- | --- | --- | --- |
| 100 | 79.10 → 83.45 | 86.90 → 93.69 | 77.76 → 81.76 | 90.80 → 93.50 |
| 1,000 | 260.42 → 105.91 | 265.81 → 118.55 | 2,070.92 → 1,634.19 | 2,163.64 → 1,699.61 |
| 5,000 | 1,027.00 → 133.32 | 1,037.14 → 135.57 | 41,624.92 → 28,342.40 | 41,867.36 → 28,136.89 |

5,000 条时两轮首批局部耗时均约下降 87%，完整遍历约下降 32%／33%，后者仍耗时数十秒。第一轮首批 Python 跟踪分配峰值由 16,430,759 降至 2,223,921 字节。100 条时新鲜文件检查使成本略增；不能声称所有规模都加速。

另设 **128 条全部为真实跨 ID 重复配置** 的负对照，预热后交替七次。旧版用 cache 提前跳过，新版逐条进入当前事务，首批中位耗时 **59.20 → 115.31 ms**；Python 跟踪分配峰值 230,312 → 244,891 字节。源文件保持原样，没有新增任务，数据库逻辑内容不变。[负对照完整分布](../evidence/runs/2026-09-15-ingress-bounds/duplicate-control.json)保留了这项退化。后续若批量优化事务核验，必须继续保证当前配置／状态和并发变化的正确性，不能恢复由 stale cache 裁定的行为。

逐次耗时、读取计数、独立内存跟踪见[第一轮](../evidence/runs/2026-09-15-ingress-bounds/benchmark-v1.json)与[第二轮](../evidence/runs/2026-09-15-ingress-bounds/benchmark-v2.json)。tracemalloc 仅用于独立首批测量，不是 RSS；I/O 包含 `/proc/self/io` 采样成本。双方均抑制逐提案 warning 的输出开销。使用私有 `/tmp`，主机非独占；没有测共享生产存储。Core-only harness 显式屏蔽可选 Pro，加载器的 `role_runner` 缺失诊断保留在原始日志，不是 CPU 动作失败。

## 真实产品与原件

独立产品项目运行两个全新的 CLI 子进程。第一轮从 sidecar 接纳并执行一个实际排序 CPU action，另一个不同 ID 的相同配置被拒绝；第二次启动重放后，ideas／attempt／reservation／artifact 投影不变，没有新增执行。第二次启动可能追加 Wait 等决策记录，因此不声称整个数据库字节或全部表不变。原始 sidecar 保留，实际输出为 `[1, 3, 5]`。

一个真实 worker 的 TERMINAL、TREE_CLOSED／ECHILD_WALL 与 SETTLED 哈希均可关联。两轮性能项目、真实产品、错误 harness 项目与重复配置对照归档于[压缩包和逐文件哈希索引](../evidence/runs/2026-09-15-ingress-bounds/archives.json)。脚本：[成本测量](../evidence/checks/2026-09-15-ingress-benchmark.py)、[重复配置对照](../evidence/checks/2026-09-15-ingress-duplicate-control.py)、[实际 CLI 验证](../evidence/checks/2026-09-15-ingress-product.py)、[归档](../evidence/checks/2026-09-15-ingress-archive.py)、[机械复核](../evidence/checks/2026-09-15-ingress-verify.py)。机械复核在单独进程中运行，不是另一位审查者。

本片是源码验收；没有发布安装包、修改全局环境或切换服务。Pro 许可只在测试进程中使用替身，没有调用真实模型、占用 GPU 或取得生产验收。仍需继续处理目录／前缀扫描和重复事务成本、Pro 真实有界检索与联合载荷、持久记忆、角色／GC 的 CephFS 兼容、受控上线及端到端科研收益。这些局部测量不证明整个产品已完成或研究效率普遍提高。
