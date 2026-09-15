# Pro research consumer 接入报告证据分页

本片基于 Core `0836960`、Pro `cd4d708`。Core 产品源码／测试不变，Pro 修改真实 research CLI／`run_research_cycle`，两仓分支仍为 `codex/p1-history-validation`。完整 Pro 原件留在私有仓，公开仓只保存摘要与提交索引。

## 实际行为

显式 `research_evidence.version: 1` 启用新响应协议。模型可继续读取后页、定向选择未见 ID，并在下一页重新核验最多 32 个保留的支持／反例。每页最多 32 个候选，SQL 使用 keyset 查询，不先 qualification 全部 completed 历史。比较仅限当前页与保留 ID，没有全局排名或科学收敛声明。未启用的原单次调用接口保持兼容。

必需的证据块完整携带报告 Objective、策略版本、比较／覆盖范围、内容身份、当前 attempt 引用、资格状态和本次请求额度。单记录显示至多 12 KiB、证据视图至多 32 KiB，与其他规则／任务内容一起受既有总 prompt 字节上限控制；放不下的记录保留 unavailable，不留下脱离资格的分数。

数据库提交／schema／路径身份、配置文件、Stop/HOLD及当前记录变化会拒绝本次读取或提案追加。原生 observation 协议、benchmark、覆盖与未知结果不降级为缓存评分。每次 provider 请求都有独立 create-only prompt 清单，原第一份文件名兼容。请求上限和既有 token envelope 都继续约束本次调用，耗尽不表示研究结束。

真实 CLI 验证还发现并修复了数据库作用域不一致：分页按配置读取了 Lake，但谱系过滤此前仍依赖显式 `--lake-db`。分页模式现在让证据、去重和谱系共用同一配置数据库，没有降低 parent 校验。

## 验证结果

| 集合 | 最终结果 |
| --- | --- |
| Core 全量 | 4,799 passed、7 可选 Pro skipped、2 既有 warning；1,078.81 秒 |
| Pro 全量 | 1,066 passed；296.44 秒 |
| Pro 定向 | 189 passed；59.49 秒 |
| Core／Pro 可选配对 | 31 passed；0.87 秒 |

最终 Pro 三组回归均冻结两仓输入，原有 Pro 测试不变。Core 全量原件保存在[run.json](../evidence/runs/2026-09-15-research-paging/core-full/run.json)、[日志](../evidence/runs/2026-09-15-research-paging/core-full/stdout.log)和[JUnit](../evidence/runs/2026-09-15-research-paging/core-full/junit.xml)。集合重叠，不能相加。私有实现与完整证据已提交为 `d25754a33884e1d640930d84f2c0674f777d7c68`，见[私有证据 SHA256 索引](../evidence/2026-09-15-pro-research-paging-private-index.json)。

[独立进程机械核验通过](../evidence/2026-09-15-pro-research-paging-verification.json)：279 个 Core 源码、511 个 Core 测试、其余 87 个 Pro 源码和 140 个旧 Pro 测试匹配基线；最终与中间回归、24,536 个归档普通文件、八份 prompt 清单、两个真实 worker 的关闭／结算及产物、token envelope 载荷均核对一致。完整核验原件留在私有仓，公开摘要记录其 SHA256。机械核验不是另一位作者的独立审查。

原始失败完整保留：旧版 7 项新检查失败；两处新增夹具预期错误；首次 Pro 全量 1,065 项通过后，真实 CLI 仍暴露 parent 作用域拒绝；修正后才得到最终 1,066 项全量。首次 benchmark harness 的缺失必填字段失败也保留。不能把中间全绿当成最终产品通过。

两个独立产品项目使用合成历史和确定性离线 provider，各自经过真实 research CLI、四次分页请求、原生 result／prompt 清单、不可覆盖入队，以及一个真实 CPU worker 的关闭结算。后页合格与 taint 被拒绝时选择不同的当前来源。保留四个实际 CLI 进程、两个 TREE_CLOSED／ECHILD_WALL、TERMINAL 与 SETTLED 哈希及产物。这是消费／执行路径验收，不是实际模型判断或科学收益。

另外经过真实 provider 路由、仅替换 HTTP 的检查确认：首次请求扣满 10,000 token envelope 后，第二次没有到达 HTTP 边界，也未追加提案。测试替身不构成真实许可、provider 或 token 账单验收。

## 首视图收益与全遍历退化

两轮独立目录、各入口预热后交替五次。在 5,000 条合成报告历史上：

| 入口 | 第一轮中位 ms | 第二轮中位 ms |
| --- | --- | --- |
| 旧全历史资格排序 | 3,456.80 | 3,437.87 |
| 新首个 16 候选证据页 | 31.74 | 31.64 |
| 新完整分页枚举 | 9,561.66 | 9,507.39 |

首视图提供的信息量不同，不能当作等信息的加速。完整分页恢复了相同合格证据集合，但重复核验使耗时约为旧排序的 **2.77 倍**，还可能增加模型请求。第一轮 Python 跟踪分配峰值依次为 5,478,604／1,129,229／2,564,546 字节，不是 RSS。测量未包含模型、谱系、失败分析和完整 prompt 组装成本，也未测共享生产存储。正负对照均保留在私有原件。

## 仍未完成

其他 Pro consumer、全局 prospective decision-contract 基线扫描、上游大文件资格读取、ID／谱系／失败和 config 历史扫描尚未统一治理。本片请求额度不是完整 CPU／财务预算视图；evidence／proposal／执行预算联合 snapshot、持久记忆、生产模型／许可、CephFS 角色／GC、受控上线及跨领域科研收益继续列为未完成。本片未发布安装包、切换生产服务或使用 GPU。
