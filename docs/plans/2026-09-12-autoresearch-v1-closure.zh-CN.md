# 原始 autoresearch V1：补缺后的最终机制验收

日期：2026-09-12。范围：冻结的 [V1-00～07 原方案](2026-09-10-autoresearch-v1.zh-CN.md)，以及重新审计发现的 C1/C2/C3 必须补缺。结论：本版实施与有限机制验证完成；研究收益未验证。提交位置为两仓 `feat/autoresearch-v1`，不是 main 合并、发布或部署。

本页是新的验收记录，不恢复 2026-09-11 被撤回的整体结论。原方案、旧汇总/检查器、历史全量及失败快照均保留原文。新的完成判断依赖本次补缺和固定源码的完整回归，而非只校验旧记录自洽。

## 重新审计的三项缺口

| 条目 | 完成的产品行为 | 可核验证据 |
|---|---|---|
| C1：领域无关的覆盖资格 | 显式覆盖声明进入实际 qualifier、报告、缓存/历史身份和 Pro 消费者；同值换指标名不改变资格。聚合列不填足数据集覆盖，缺失/重复/非有限值拒绝；不保留 `wer_*` 资格后门 | Core 修复 `7d4cabf`；[原消费者红测](../evidence/2026-09-12-s3-consumers-baseline.json)、[声明边界](../evidence/2026-09-12-s3-boundaries-baseline.json)、[兼容迁移](../evidence/2026-09-12-s3-core-fixture-migration.json)、[C1/C2 全量](../evidence/2026-09-12-c1c2-core-full.json) |
| C2：配额不阻断实验 | 同一个真实 Orze/Pro 控制循环中，同账户退避、健康账户独立、CPU-backed 实验实际完成。该联测发现并修复退出容器接口错误；shutdown 只移除入口捕获且身份未变的已确认对象，保留 HOLD/回调替换 | Core 修复 `e449bf8`；[8 failed / 2 passed 原边界](../evidence/2026-09-12-c2-shutdown-slots-baseline.json)、[52 项回归](../evidence/2026-09-12-c2-target-result.json)、[独立复核](../evidence/2026-09-12-c2-shutdown-slots-independent-review.json)；Pro `tests/test_quota_data_plane_integration.py`，当前完整全量包含该真实联测 |
| C3：运行中租约 | 复用原 attempt、预算与监督树；持久同主机/boot 的不可变期限，CPU v2 监督器在父进程不轮询时仍自主停树；终态发布前后检查授权。已确认终态可仅续结算，未确认/未知副作用仍 HOLD、不重跑或释放预约 | Core 修复 `f62495df9359cb85a128be7ff70ca78dbfcd7f03`；[原行为 2 failed / 1 passed](../evidence/2026-09-12-runtime-lease-original-gap.json)、[91 项新目标](../evidence/2026-09-12-c3-target-result.json)、[独立监督复核](../evidence/2026-09-12-c3-supervision-review.json)、[发布边界](../evidence/2026-09-12-c3-publication-author.json)、[恢复适配](../evidence/2026-09-12-c3-budget-recovery-author.json) |

C1 的明确兼容取舍：无覆盖门的旧配置继续工作；依赖隐式数据集覆盖的配置必须声明覆盖语义。不自动改用户项目，也不把原来的错误启发式兼容为资格后门。C3 默认期限从 INTENT 捕获，包含启动准备及发布；正常 worker 已结束但迟到发布也可能失去授权，这是 [冻结合同](2026-09-12-c3-runtime-lease.zh-CN.md)公开的新边界，不冒称原产品已有的 wall timeout 语义。

C3 未新增持久表、第二套状态机或外部依赖；复用原 effect guard 完成发布授权。它增加必要的协议和边界检查，不声称全栈净减行或复杂度已经消失。

## 固定源码上的最终实测

| 验证 | 实际结果 | 原始证据 |
|---|---|---|
| Core 完整 `tests/`，无选择排除 | 4456 passed、7 skipped、2 warnings；816.36 秒；exit 0 | [完整命令、12 个原始输出块与前后指纹](../evidence/2026-09-12-c3-core-full.json)、[日志](../evidence/runs/2026-09-12-c3-core-full.log)、[原始 JUnit](../evidence/runs/2026-09-12-c3-core-full.junit.xml) |
| Pro 完整 `tests/` | 983 passed；173.44 秒；exit 0 | [固定 Pro 证据](https://github.com/orze-ai/orze-pro/blob/da2d7d92da5281ab49362509fad71fc3d78e427a/docs/evidence/2026-09-12-c3-pro-verification.json) |
| 原跨仓配对回归 | 60 passed；0.82 秒；exit 0 | 同一 Pro 记录中的独立命令、原始输出和 JUnit |
| C3 新增目标 | 91 passed；27.47 秒；exit 0 | [目标记录](../evidence/2026-09-12-c3-target-result.json)；这些测试已经包含在 Core 全量内，不累加 |
| 额外真实终态崩溃补验 | 2 个场景均通过；6 次真实新 CLI、2 次 native action | [完整重启证据](../evidence/2026-09-12-c3-terminal-restart-review.json)；不属于 pytest 收集数 |

7 项跳过均为 Core 独立环境缺少可选 Pro；2 项为已导入辅助模块的 pytest assertion-rewrite 警告。均原样列明，不称“零跳过零警告”。Pro 使用真实两仓源码和测试进程内授权替身，未修改生产授权，不能证明生产 license 或外部服务可用。

Core `f62495d` 与 Pro `da2d7d9` 的 777 + 226 = 1003 个源码/测试/示例/构建文件，在目标、两仓全量前后均为相同字节；期间仅提交元数据和证据文档变化。Pro 生产实现仍为 `1384c12d40069c98a430d0e25647de102a7e38de`。额外导入的旧 native 快照不在 1003 中，单列 SHA-256 `a3af30c67e83afb0cc8eb311c26b574a4193a91e24dedd8d3653d1317a38e92e`，新测试加载前检查字面哈希，它也属于固定 Core 提交。

完整 Core JUnit SHA-256：`f3e3fe0edb1e755ec0d76dd88e3226963df817260d3c765d6183fe785f3f0daf`。另见 [独立 JUnit/原始报告核验](../evidence/2026-09-12-c3-full-independent-review.json)、[固定 Git 对象复核](../evidence/2026-09-12-c3-fixed-source-review.json)和[机器索引](../evidence/2026-09-12-autoresearch-v1-closure.json)。

两个补充崩溃窗口分别证明：

- 真正租约到期、终态/effect 已确认且 guard 释放后，在 settle 第一入口前实际 exit 86；后续两次新 CLI 只将原预约 BOUND→SETTLED 一次，不重跑 worker 或改写产物。
- 真正写入 confirm 回执后、协调器授权后检前实际 exit 86；过期后两次新 CLI 都保持原 guard、BOUND 与 HOLD，不把“有回执”误判为授权完成，不按年龄接管。

后一个场景的 SQL 终态和 artifact 确实已经提交；HOLD 不是撤销这些事实，也不承诺跨文件/SQL 的全局回滚。

## 原 V1 条款不缩小

V1-00 的原始环境/基线记录保留；V1-01 的 Objective、有效性、报告/上下文一致性由原分片加 C1 覆盖；V1-02 的持久交付、幂等/并发、当前 attempt 隔离与恢复由原分片加 C3 覆盖；V1-03 的配额、分类、需求/门控由原分片加 C2 的同循环证据覆盖；V1-04 的实际方法交付与整份上下文预算继续完整回归；V1-05 的产物/评估分离、同源补评、独立复验和自有清理由完整回归覆盖；V1-06 的已有 CLI/Orze 真实 CPU 策略、领域、wait/stop/复验/分析闭环继续运行；V1-07 的两个异构任务和原独立留出记录保留，并在本次固定源码复跑兼容。

各片的真实路径、原红测和源码提交见 [逐片账本](2026-09-10-autoresearch-v1-status.zh-CN.md)。原独立留出结论仍绑定当时冻结的 Core；本次公开后重跑不能重新获得“未见任务”资格。全量日志里的 4 份 acceptance、6 份 holdout、40 份 recovery 原始 JSON 由独立复核归档，报告/用例/项目/进程数不能互相替代或累加成独立科研实验。

## 失败历史与适用边界

C3 原产品诊断、新候选不可变 descriptor 缺陷、首次候选 10 failed / 17 passed、第二候选 1 failed / 27 passed、监督夹具和预算 1 ns 边界均分别保留。真实实现失败与新夹具前提错误有明确区分，原文件和命令不被后来的绿测覆盖。C1/C2 的旧夹具迁移及 Pro 首次全量 1 failed / 982 passed 也未删改。

本次完成的是原计划中可恢复、可验证、可替换策略的通用 autoresearch 最小闭环及其离线机制验证，不是针对 ASR 的调参优化。没有据此证明普遍研究提速、科学显著性、最优解、特定 GPU 节省比例或所有未知缺陷消失。

远程集群接管、未知非幂等副作用自动裁定、通用历史进程 adoption、自动 repair、训练 checkpoint resume、任意 Director 交接仍未实现；真实 GPU、付费 provider、生产授权、在线迁移和部署仍未验证。这些边界不改名为成功，也不自动扩为本轮的新授权。
