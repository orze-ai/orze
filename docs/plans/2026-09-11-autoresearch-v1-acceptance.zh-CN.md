# 原始 autoresearch V1 汇总验收

日期：2026-09-11。2026-09-12 更正：**整版完成结论已撤回，继续实施**。见[重新审计与剩余工作](2026-09-12-v1-reopened.zh-CN.md)。下文保留原验收论证和历史结果，不再作为整个 V1 完成的有效签核。

依据：[冻结的原始 V1 方案](2026-09-10-autoresearch-v1.zh-CN.md)与[逐片实施账本](2026-09-10-autoresearch-v1-status.zh-CN.md)。本文只汇总原始 V1-00～07，不新增实施范围。V1-00 的原始命令、两仓 SHA、环境失败与离线替身记录见 [Pro 固定基线入口](https://github.com/orze-ai/orze-pro/blob/1384c12d40069c98a430d0e25647de102a7e38de/docs/plans/2026-09-10-autoresearch-v1.zh-CN.md)。

## 要求与代表性证据

下列测试路径是可定位的代表性机制断言，不是全部测试清单；历史 JSON 保留各自的代码版本、命令、失败分类和独立复核。测试数量不能替代行为证据，重叠回归不能累计为独立缺陷。

表中未标 Pro 的测试属于 Core；Pro 路径相对于独立的 orze-pro 仓。

| 原始条目 | 本次汇总核对的条件 | 代表性当前测试路径 | 历史证据入口 |
|---|---|---|---|
| V1-00 | 两仓基线、导入路由、离线环境及授权替身与产品验收分开记录 | Core `tests/conftest.py`；Pro 固定基线入口中的 `unittest.mock.patch` 与 `PYTHONPATH` 命令（环境设置，不冒充产品行为断言） | [原始起点及基线记录](2026-09-10-autoresearch-v1-status.zh-CN.md)；[S2 冻结基线](../evidence/2026-09-11-s2-claim-reader-baseline.json)仅证明本次回归起点，不替代最初基线 |
| V1-01 | 声明最小化/最大化，零/负数、缺测/非有限、partial、协议及上下文资格；确定排序不等于统计等价 | `tests/test_objective_ordering_contract.py`、`tests/test_observation_report_protocol_binding.py`、`tests/test_holdout_product_review.py` | [Objective 排序](../evidence/2026-09-10-v1-01c-objective-ordering.json)、[局部改善](../evidence/2026-09-10-v1-01h1-objective-progress.json)、[观察发布](../evidence/2026-09-10-v1-05b2-observation-publication.json) |
| V1-02 | 合法状态、幂等交接、并发提案、实际 payload、已证未启动恢复、过期尝试隔离；可核验终态续结算与未知副作用 HOLD | `tests/test_trigger_delivery.py`、`tests/test_proposal_admission.py`、`tests/test_role_supervision.py`、`tests/test_cpu_terminal_settlement_recovery.py`；Pro `tests/test_trigger_payload_consumer.py` | [提案交接](../evidence/2026-09-10-v1-02c-proposal-handoff.json)、[原生 attempt](../evidence/2026-09-10-v1-02d2-attempt-authority.json)、[终态重启续结算](../evidence/2026-09-11-v1-06f-terminal-recovery.json) |
| V1-03 | 同账户配额背压不牵连健康账户/实验；拒绝、空输出、崩溃分开；持久需求与显式角色门控 | Pro `tests/test_shared_quota_completion.py`、`tests/test_persistent_research_demand.py`、`tests/test_research_result_classification.py` | [共享配额](../evidence/2026-09-10-v1-03a1-shared-quota.json)、[持久需求](../evidence/2026-09-10-v1-03b-persistent-demand.json)、[Provider 分类](../evidence/2026-09-10-v1-03f-provider-outcomes.json)、[可选角色](../evidence/2026-09-10-v1-03e-optional-presets.json) |
| V1-04 | 退役/未证实方法不升级为配方；保留来源/原因；必需契约完整、整份提示有界、项目隔离 | Pro `tests/test_method_context_authority.py`、`tests/test_native_method_context_delivery.py`、`tests/test_prompt_budget_boundaries.py`、`tests/test_native_prompt_manifest.py` | [方法上下文](../evidence/2026-09-10-v1-04a-method-context.json)、[整份提示预算](../evidence/2026-09-10-v1-04b-prompt-budget.json) |
| V1-05 | 产物与科学有效性分离；同源补评不重跑生成；显式复验有独立身份；清理限定自有范围 | `tests/test_observation_snapshot_contract.py`、`tests/test_cpu_replication_product.py`、`tests/test_cleanup_pattern_containment.py`、`tests/test_holdout_product_review.py` | [产物发布](../evidence/2026-09-10-v1-05b1-artifact-publication.json)、[显式复验](../evidence/2026-09-10-v1-05b3-explicit-replication.json)、[受限清理](../evidence/2026-09-10-v1-05c1-contained-cleanup.json)、[留出补评](../evidence/2026-09-11-v1-07b-holdout.json) |
| V1-06 | 原 CLI/Orze 主循环中的显式 CPU action；可替换领域/策略、真实 Wait/Stop/Propose/Replicate/分析及 0..N 观察 | `tests/test_cpu_product_loop.py`、`tests/test_cpu_domain_product.py`、`tests/test_cpu_proposal_product.py`、`tests/test_cpu_replication_product.py`、`tests/test_acceptance_product.py` | [CPU 执行基础](../evidence/2026-09-11-v1-06a-cpu-actions.json)及[实施账本中的 B～F](2026-09-10-autoresearch-v1-status.zh-CN.md)、[实际跨领域闭环](../evidence/2026-09-11-v1-07a-cross-domain.json) |
| V1-07 | 两个异构 CPU 任务及冻结后独立留出共用 Core；兼容回归、真实故障/负结果、逐项固定提交 | `tests/test_acceptance_product.py`、`tests/test_acceptance_policy_review.py`、`tests/test_holdout_product.py`、`tests/test_holdout_product_review.py` | [已知领域验收](../evidence/2026-09-11-v1-07a-cross-domain.json)、[原始留出验收](../evidence/2026-09-11-v1-07b-holdout.json)；当前兼容性见 [S2 最终记录](../evidence/2026-09-11-s2-claim-reader.json) |

## 恢复与未知状态的准确含义

动态条件不仅在入队检查：`test_cpu_domain_runtime_review.py` 真实 READY 后换源 inode 拒绝 GO；`test_trigger_delivery.py` 用实际 SQLite 与受控时钟验证启动边界重查租约。运行中 `test_native_pre_script_supervision.py` 使用真实 1 秒超时及 CPU 后代，STOP 返回 0 仍不得成功，实际 supervisor 丢失后保持 HOLD、不二次 STOP 或重派；[原始监督故障证据](../evidence/2026-09-10-v1-05c2d2-native-pre-script-supervision.json)保留其范围与完整失败历史。受控时钟不证明分布式 RUNNING 续租接管。

已明确未执行的启动失败可按所属协议重新受理；评估失败可以新尝试读取原 artifact，不能把原失败 occurrence 重写为成功。

V1-06F 验证真实控制器崩溃后，对完整当前 TERMINAL、整树闭合、effect 和来源一致的 CPU 预约续结算；原 worker、原 artifact/observation 不重跑或改写。

LAUNCHING/RUNNING/IN_DOUBT、未知 owner、未确认发布及持久 Stop 不因年龄、PID 消失或空 map 自动释放。HOLD 是明确的不确定状态，不是“恢复成功”；也不能把所有 HOLD 都解释为漏实现了原计划所需的自动重试。

通用历史进程 adoption、自动 repair、训练 checkpoint resume、任意 Director 正向交接不是原始最小闭环的独立必做验收项。它们仍是能力边界，不能借本汇总宣称已经实现；后续新增需求应另立协议和证据。

## 历史记录与最终封口

旧分片文档中的“整个 V1 尚未完成”是当时正确的日期性结论，原文及当时缺项继续保留。最终汇总只在已被后续证据支持的原始条件上更新总体状态，不倒改历史、不扩大各分片原有支持范围。

S1/S2 是规则与 reader 收敛及兼容验收，不是另一个研究收益实验。当前再次运行 A/B 测试只证明回归兼容；公开过的同一留出任务不重新获得“独立未见”资格，最初留出结论仍绑定其冻结 Core epoch。

最终源码配对：Core `92e388b9f7e1a1bc822695c2f1c52d4e43ab3b79` / Pro `1384c12d40069c98a430d0e25647de102a7e38de`。实际全量为 Core **4297 passed / 7 skipped / 2 warnings，775.40 秒**；Pro **978 passed，171.83 秒**；配对 **60 passed，0.85 秒**。7 项为可选 Pro 跳过，2 项为已导入辅助模块的 pytest 重写警告；配对回归另使用真实两仓源码与测试进程内授权替身，不验证线上授权。

764/224 个源码、测试、示例和构建声明文件全量前后及固定源码提交精确一致；472/134 份旧测试未改。[S2 原始日志与 JUnit](../evidence/2026-09-11-s2-claim-reader.json)、[固定提交与独立复核](../evidence/2026-09-11-s2-root-validation.json)和[独立 JUnit 审核](../evidence/2026-09-11-s2-junit-independent-review.json)已归档。S2 证据远端 Core `d92f5e69771057ffa83cd1e8574b23ec6e1be7c1`、Pro `331b701a821d269cfd18fb1242c095fb2a7acb77` 已精确读回；证据提交与源码提交分开，未合并 main 或部署。

[汇总机器索引](../evidence/2026-09-11-autoresearch-v1-acceptance.json)将 8 个原始条目映射到 41 个代表性测试函数、此次 JUnit 中的 81 个通过实例及 23 个历史证据引用（引用可重复，不等于 23 项独立实验）。检查器 `PYTHONDONTWRITEBYTECODE=1 python3 docs/evidence/checks/v1-acceptance-check.py` 从 Core 根目录运行，相邻目录须为配套 orze-pro；验证固定源码、历史文件与当前通过实例的一致性，不自动证明测试语义或科学收益。历史目标/full、单元/产品、同一 fixture 的多个断言和重跑仍分开计量。

## 验收结论与边界

“原始 V1-00～07 在声明支持范围内的最小 autoresearch 闭环机制验收完成；研究收益未验证。”

该表述限定于有记录的离线私有项目、真实本地 CPU 执行和明确故障注入。它不等于全技术栈无缺陷、全领域科学平台、统计独立/显著性、效率普遍提升或节省特定比例 GPU 时间。

真实 GPU、付费 provider、生产授权、在线迁移及部署均未由本轮验收证明；提交和推送不等于发布。需要科研收益结论时，另行授权并冻结预算、协议和重复对照。

原始 V1 的这一轮实施与有限机制验收已收口。未知状态仍显式 HOLD，未验证能力不改名为已完成；后续生产部署和研究收益实验须各自提供新授权、冻结条件与证据。
