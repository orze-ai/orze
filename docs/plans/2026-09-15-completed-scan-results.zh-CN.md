# 全局合格基线分批聚合与 CPU 决策合约兼容

本片将 Pro research／code-evolution 的 prospective decision-contract 基线和只取最佳值／计数的 evolution 调用，接到逐批生命周期读取与标量聚合。所有完成候选仍经过原 qualification，不用页面最佳值替代全局最佳值。并修复真实 CLI 暴露的合法 CPU command 参数列表被决策合约误判为隐式参数扫描的问题。

最终两轮 5,000 条完成历史中，聚合调用的 Python 峰值分配约从 1.61 MB 降至 69–74 KB；完整扫描约 660–663 ms，旧版约 659–660 ms。两个带合约的实际 CPU 提案完成执行、关闭、结算及重启重放。没有整体研究提速或产品整体完成主张。

## 实现与范围

- Core 新增 `reporting/completed_scan.py`。默认每次最多读 128 个候选，允许 1–256；使用 binary keyset 顺序，沿用完整 schema／身份唯一性与训练、评估阶段协议。
- 查询用 `CROSS JOIN` 固定从 idea ID 顺序出发，避免当前标准 schema 的状态索引优先计划在每页重新排序剩余完成历史。不同索引、collation 或历史 schema 的成本仍可能不同。
- 每页查询关闭 read transaction 后才交给调用者资格核验，不在文件读取期间占用数据库读事务。数据库 revision、schema、PID、路径身份及重定向在读取和正常退出时复核；检测到变化即丢弃整个聚合。
- 提前停止遍历不能作为完整扫描退出。回滚和 SQLite 实际未写入的 no-op UPDATE 可以继续；真正修改并提交后再恢复原值仍拒绝。
- Pro `retain_values=False` 只返回最佳值／计数及配置、数据库作用域；需要家族统计的默认调用仍保留各 ID 和分数。扫描结束重新读取项目配置，配置变化拒绝整个结果。
- decision contract 下，仅结构准确且通过现有 action／domain-request 校验的 `native_cpu_action`，可把 argv、inline inputs 和来源引用视为单个任务的数据。单独写 kind 标签、非法或混合 envelope、额外 root 参数、损坏动作均拒绝。训练配置仍走原隐式 sweep 检测，其他 research policy 限制仍继续执行。

原 qualification、预算、admission、执行、恢复实现没有修改。Core 原有源码仅改 research-policy 验证函数；Pro 只改 shared snapshot、最佳值包装及 decision baseline 三个函数，research 与 code-evolution 共享后者。已有测试保持原件。

候选集合在 Python 中按 O(page_size) 保留；这不是总 I/O、SQLite 工作、单 ID、单文件或整个 prompt 的字节硬上限。每次仍核验所有完成候选。文件按原资格器逐个读取，不声称所有文件构成同时刻原子快照。家族统计等需要完整映射的消费方仍为 O(N) 保留。CPU 合约通过不授予预算、进程或执行所有权，也不自动满足科研阈值。

## 等覆盖成本对照

冻结旧 Pro `83e1bfd` 的 guards 源码，在同一私有合成项目中交替执行旧完整 snapshot、新完整 snapshot、新标量聚合。每轮 5 次计时；另测 Python traced peak 和进程 I/O。四次独立项目运行，前两次对应未采用的查询顺序，后两次对应最终源码。旧／新完整分数映射、计数及最佳值一致，每个完成候选均 qualification 一次，数据库逻辑摘要不变。

| 历史／完成候选数 | 最终第一轮旧→聚合 ms | 最终第二轮旧→聚合 ms | 最终第一轮旧→聚合峰值 bytes |
|---:|---:|---:|---:|
| 1／1 | 1.149→1.862 | 1.129→1.841 | 15,109→20,713 |
| 100／100 | 14.096→14.670 | 13.995→14.686 | 38,142→35,447 |
| 1,000／1,000 | 130.871→132.118 | 130.413→132.334 | 226,947→57,543 |
| 5,000／5,000 | 659.231→660.069 | 660.155→663.241 | 1,611,219→73,536 |
| 5,000／1 | 1.149→3.152 | 1.137→3.139 | 14,236→19,750 |

5,000 个完成候选的聚合峰值下降约 95.4%–95.7%，最终完整扫描接近旧版，不能当作稳定提速。保留完整映射的新 snapshot 约 667–670 ms／980 KB。小历史和稀疏完成历史的固定检查／ID 遍历开销增加；稀疏行并不获得内存收益。Python traced peak 不是 RSS；夹具为本地合成报告，没有模型调用或科研收益测量。

未采用的首版：两轮 5,000 条约 664–665→798–799 ms，约慢 20%。独立只读 profile 发现 40 个候选查询各使用临时排序；SQL 对照固定读取顺序后候选检索约 139→6.9 ms，结果完全一致。但只含 1／50 个完成项的 SQL 负对照反而更慢，原件均保留。SQL 单项时间没有代替最终完整 qualification 对照。

## 实际消费链

最终用保留的原失败 harness，在两个新私有项目中各导入 260 条历史。两个 research CLI 在各九次显式分页请求中始终携带完整全局基线；末条正常时为 −259，末条 taint 时为 −258。离线确定性 provider 提交一个声明式 CPU action 和一个 prospective decision contract，两个真实 receipt 均 admitted。

之后两个 Core CLI 各执行一个实际 worker：输出来自正确后页来源，TREE_CLOSED／ECHILD_WALL 关闭，预算 SETTLED。两次额外 Core 重启重放后，ideas、attempt、reservation、artifact 及决策 receipt 摘要不变；每个项目仍仅一次执行、一次结算和一个产物。18 份 prompt 清单、全局基线、receipt、attempt／terminal／settlement 及输出都保留。动作完成不等于其科研阈值已满足；未使用真实模型、生产许可或 GPU。

另保留早期两个普通提案的合约提交（18 个请求、未执行），以及两个未开启 decision contract 的 CPU 对照执行（8 个请求）。这两组不替代最终带合约的执行验证。

## 验证与失败记录

- 旧 Pro 基线 25 项通过；最初误写不存在的测试文件导致 exit 4／0 项，保留原件。
- 原 Core 定向 no-op UPDATE 必然推进 revision 的错误假设：1 失败／46 通过。原测试、日志和修正后的复跑均保留；没有修改旧测试。
- 新 CPU 合约行为基线：18 失败／6 通过。最终这 24 项全部通过；动作结构、训练 sweep 和其他策略限制均有验证。
- 最终 Core 定向 172 项、Pro 定向 69 项通过。
- 首版冻结全量：Core 4,869 通过／7 可选跳过、Pro 1,074 通过、配对 31 通过。它对应改进查询顺序和 CPU 合约修复之前的源码，不替代最终回归。
- 最终 Core 全量 4,893 通过、7 可选跳过、2 既有 warning（953.23 s）；Pro 全量 1,074、配对 31 通过。两仓 before／after 输入指纹一致。
- 单独进程机械核验通过：原件 Git／AST 范围、全部冻结运行、四次成本对照、27,852 个归档文件、四个实际 worker、44 份成功 prompt 清单及两次带合约重放均关联核对。它不是另一位审阅者。

产品 v1 保留 CPU 合约拒绝；v2 普通提案已 admitted，但 harness 误查询未创建的执行表导致失败；v3 修正无执行状态后成功。最终另用原 v1 脚本的相同字节重新运行，证明 CPU 合约修复，没有删去合约或绕过 preflight。各版本脚本、项目及失败日志均保留。

公开 Core 原件在 [completed-scan runs](../evidence/runs/2026-09-15-completed-scan/)。Pro 源码、prompt、完整测试输入清单、项目归档和审计在私有仓 `docs/evidence/runs/2026-09-15-completed-scan/`；公开端仅保留结果摘要及[已提交 Pro 文件哈希索引](../evidence/2026-09-15-completed-scan-pro-private-index.json)。

## 未关闭的工作

继续联合 evidence／proposal／预算总载荷、其他全历史 consumer、源文件容量及稀疏场景成本治理。持久记忆、CephFS／服务上线、真实许可和模型、跨领域等质量端到端验收均未由本片完成。

最终 Core run SHA256：`fa73800d403ac82aa3359214124ea55a009ca8b23bcf74549ceec295967eb825`；私有最终机械核验 SHA256：`d6bd908d04c9afa4569943f750dff6615d0d228f0a218e55fa005ea5ce617194`。
