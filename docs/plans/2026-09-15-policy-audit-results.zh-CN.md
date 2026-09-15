# Policy 只读续页：合并重复预算审计

本片继续机器交接的 P1，起点 Core `cdfbe064849af73b89300d0778be514044c31087`，配对 Pro `bac1fdb`，两仓工作分支仍为 `codex/p1-history-validation`。

## 改动与边界

只读 evidence／proposal 续页不执行 ingress，也不创建预算、claim 或 worker。因此将其完整预算审计放在 Policy callback 前，单次续页从两次完整审计减少到一次。没有保存或复用预算 snapshot，每一页都重新读取全部预算历史；非续页循环仍在 ingress 写入前单独完整审计。

`require_admission` 返回当次审计得到的独立预算视图，并在 callback 前检查当前 invocation、配置、Stop/HOLD。分页 revision 在 evidence／proposal／budget 联合读取完成后及 callback 返回后分别复核。并发提交、同连接回滚、schema 变化等使旧 revision 失效；`data_version` 只用于拒绝混合视图，不授予任何执行权。

预算模块、事务与执行前 `require_permit` 完全保持原件。source qualification、reserve、bind、settle、恢复及实际 GO 路径保留原复核。没有加入全局预算缓存、schema 迁移、退款或研究总期限。

## 失败基线与验证

新增 11 项检查先运行于原实现，得到 8 项预期失败、3 项通过：[原始日志](../evidence/runs/2026-09-15-policy-audit/baseline.log)。一项失败显示重复扫描；其余失败表明部分读取期变化仍能到达 callback。旧版已有的 callback 后 revision 检查仍会拒绝 SQL 变化后的决定，本片增加 callback 前边界，不能把基线失败解释成已证明发生了越权执行。

定向结果为 261 passed、1 项 pytest helper 导入顺序 warning，耗时 157.34 秒：[日志](../evidence/runs/2026-09-15-policy-audit/targeted.log)、[JUnit](../evidence/runs/2026-09-15-policy-audit/targeted.xml)。覆盖预算坏历史、精确累计、槽位冲突、并发、回滚、Stop/HOLD、分页、真实 CPU action 与恢复。

| 集合 | 结果 |
| --- | --- |
| Core 全量 | 4,786 passed、7 可选 Pro skipped、2 既有 warning；994.85 秒 |
| Pro 全量 | 1,037 passed；303.64 秒 |
| Core／Pro 可选配对 | 31 passed；0.97 秒 |
| 定向预算／分页／真实 CPU／恢复 | 261 passed；157.34 秒 |

三个完整回归记录的运行前后源码／测试指纹一致，Core 此次是一次完整全绿运行：[原始 run.json](../evidence/runs/2026-09-15-policy-audit/core-full/run.json)、[日志](../evidence/runs/2026-09-15-policy-audit/core-full/stdout.log)、[JUnit](../evidence/runs/2026-09-15-policy-audit/core-full/junit.xml)。不同集合重叠，不能相加。

Pro 完整原件保留在私有仓固定提交 `520971e`；[私有证据索引](../evidence/2026-09-15-policy-audit-pro-private-index.json)逐项从 Git 提交读取并校验 13 份入口原件。完整 Pro 日志和源码清单不复制到公开 Core 仓。

[独立进程机械复核已通过](../evidence/runs/2026-09-15-policy-audit/verification.json)：其余 278 个 Core 源码文件、509 个既有 Core 测试文件，以及 Pro 90 个源码和 140 个测试文件均匹配 Git 原件；回归日志、JUnit、输入指纹和全部归档哈希一致。它还将五次 Policy trace 的预算值与对应 SQLite 行逐项核对，并关联了两次真实 worker 的关闭证明与结算哈希。

## 同机交替测量

第一轮在 100／1,000／5,000 条合成 SETTLED 预算记录上，各版本预热后交替运行七次。所有调用走实际 Orze iteration、BoundPolicy 和 EvidencePager，另有 33 个明确标识的元数据前缀。它们不是实际 worker 或结算历史。

| 预算记录数 | 第一轮续页中位耗时 ms：旧 → 新 | 第二轮：旧 → 新 |
| --- | --- | --- |
| 100 | 55.45 → 45.72 | 60.30 → 46.21 |
| 1,000 | 252.78 → 142.00 | 256.34 → 145.27 |
| 5,000 | 1,130.51 → 583.46 | 1,139.42 → 586.99 |

每个版本逐次输出相同，SQLite 逻辑内容前后指纹相同，完整预算审计次数为 2→1。5,000 条记录时局部耗时下降约 48%。新增 callback 前校验有少量开销，Python 跟踪分配峰值在第一轮为 146,618→158,916 字节；没有声称内存下降。

原始耗时分布、I/O 计数和独立内存跟踪保存在[第一轮报告](../evidence/runs/2026-09-15-policy-audit/benchmark-v1.json)和[第二轮报告](../evidence/runs/2026-09-15-policy-audit/benchmark-v2.json)。计数、内存跟踪和准备下一条续页不进入计时；I/O 包括读取 `/proc/self/io` 的开销，tracemalloc 峰值不是 RSS。测量使用私有 `/tmp`，主机并非独占，也未测共享生产存储吞吐。

两轮各 406 个项目文件（实际合成 SQLite、metadata effect 与原报告）分别归档；[压缩包与逐文件哈希索引](../evidence/runs/2026-09-15-policy-audit/benchmark-archives.json)保留完整输入，不能把其中合成 SETTLED 行解释为原生执行成功。

两轮 harness 均屏蔽可选 Pro 导入，扩展加载器记录了无法加载 `role_runner` 的诊断信息，原始日志保留。这是 Core-only 测量中预期的可选扩展拒绝，不是 CPU iteration 失败；没有调用模型或真实许可。第二轮提前至所有导入前设置屏蔽，仍保留同样的扩展诊断，CPU 产品代码保持相同。

## 真实产品记录

原有产品检查走实际 CLI／Orze loop：先产生一个真实 source，读四次后页，再执行一个真实 CPU action。两个 worker 的 TERMINAL、TREE_CLOSED／ECHILD_WALL 和 SETTLED 记录均保留；33 个无进程树的前缀仅为元数据夹具。保存了完整项目的 152 个普通文件，包括实际 SQLite、effect 记录与五次 Policy trace：[压缩原件](../evidence/runs/2026-09-15-policy-audit/native-paging-product.tar.gz)、[逐文件哈希](../evidence/runs/2026-09-15-policy-audit/product-archive.json)。

这属于源码运行验收。本片未发布包、修改全局安装或切换生产服务。Pro 测试许可替身不代替真实许可／provider 验收。

## 剩余工作

非空 ingress 的全局 config cache、源文件／sidecar 总量与循环扫描仍未关闭。Pro legacy report consumer 仍需自己的有界检索与联合载荷合同，不能将本次 Core native CPU 续页优化当作其实现。持久记忆、CephFS 角色／GC 支持、受控生产上线及端到端研究收益继续保留为未完成事项。

复现脚本：[交替测量](../evidence/checks/2026-09-15-policy-audit-benchmark.py)、[独立进程机械复核](../evidence/checks/2026-09-15-policy-audit-verify.py)。后者是单独进程中的机械核实，不是另一位作者或人工审查。
