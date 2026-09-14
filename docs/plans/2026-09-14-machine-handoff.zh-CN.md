# Orze 换机接续 TODO

交接日期：2026-09-14。这是开发工作的接续清单，不是现有运行任务的跨机器接管授权。

## 1. 目标和不可改变的约束

- 把 **Orze 产品**做成通用 autoresearch 框架，不对 ASR 单个 research 过度优化，不把 WER／领域名硬编码回通用决策。
- 用户要求持续推进：逐项修复、验证、独立核实、提交推送，拿原始证据证明完成；不要把“一个分片完成”写成“整个产品完成”。
- **没有研究总时限，研究何时继续／等待／暂停／停止由 Orze 的策略和证据决定。** 保留每个 action 的 timeout、运行租约、资源约束及人工 Stop/HOLD；不能把分页结束当作研究收敛。
- 早期 CPU 对照协议中的 10 秒累计额度、外层 30 秒／15 分钟，是当时有限实验的参数，不是当前产品的研究总期限。后续明确支持的 CPU execution v2 可用 `wall_budget_seconds: null`。
- 未知提交、未知副作用或所有权不明必须 HOLD；不自动退款、不按年龄／PID 消失清理 guard、不重新执行已有但不确定的动作。
- Core 仓公开，Pro 仓私有。Pro 源码、完整测试清单／日志、制品元数据保留在私有仓；公开交接仅保留结果、哈希和引用。不搬运密钥或许可缓存到仓库。
- 不自动覆盖用户 ASR 项目、全局安装或现有服务；不自动合并 main、发布 PyPI、使用真实付费账户／GPU或进行生产切换。生产目标及资源授权须明确。

## 2. 从哪里继续

两仓都在 `feat/research-production-validation`，不是 main。本文件新增前已交付的固定状态如下；新增本交接文档的提交只改文档。

| 用途 | Core | Pro |
| --- | --- | --- |
| 含最新验收及原始证据的交付提交 | `5ca32f68388530abcdf78543293ef9480cce0546` | `ac937e413c2d71d1a38d23998910358077f8cdfc` |
| 本轮测试／制品使用的固定候选 | `df446ab35e745daf02cc7c4a19cadffd57518103` | `879859cb383d7f113f58ba1da054c6785e4fce6a` |
| 交接前读回的 main，仅作历史定位 | `5ca2680dbd0139e63cd35a4f83da438b5b47ab99` | `57feb6e0af19dcf62a80b1e447f0771936d39afe` |

仓库：[orze-ai/orze](https://github.com/orze-ai/orze)、[orze-ai/orze-pro（私有）](https://github.com/orze-ai/orze-pro)。本轮交付时两工作树干净，已推送并读回远端确认。到新机后仍须重新检查分支和工作树，不能假定远端永不变化。

在一个自己选择的、持久存储的全新父目录中执行；下面目录已存在时应停下，不覆盖旧工作：

```bash
set -e
mkdir orze-continuation-2026-09-14
cd orze-continuation-2026-09-14
git clone --branch feat/research-production-validation https://github.com/orze-ai/orze.git orze
git clone --branch feat/research-production-validation https://github.com/orze-ai/orze-pro.git orze-pro
git -C orze merge-base --is-ancestor 5ca32f68388530abcdf78543293ef9480cce0546 HEAD
git -C orze-pro merge-base --is-ancestor ac937e413c2d71d1a38d23998910358077f8cdfc HEAD
git -C orze status --short
git -C orze-pro status --short
```

Pro clone 需要自己的合法仓库访问权限；失败不要换成公开同名占位包。两仓保持相邻，便于原配对测试。若分支在交接后已有新的源码／测试修改，先审查差异，不能把旧绿测套用到新代码。若要逐字节复现旧验收，另建指向表中固定提交的 worktree，不重置已有脏工作树。

旧机正确工作树仅作历史定位：

```text
/hot-data/fsx/workspace/erik/orze-production-validation-2026-09-12.UgS3uV/orze
/hot-data/fsx/workspace/erik/orze-production-validation-2026-09-12.UgS3uV/orze-pro
```

不要误用旧 `orze-implementation-2026-09-10.SyXgzC` 工作树；不要在用户有未提交修改的 `auto-research-1.7b` 中进行框架改动。

## 3. 已完成：不要重新作为未解决缺陷修一遍

- [x] 原 V1-00～07 通用最小闭环及 C1/C2/C3 补缺：领域无关资格、配额与实验数据面隔离、运行租约与保守恢复。机制验收不是普遍科研收益证明。
- [x] CPU 当前终态 evidence 分页／定向读取、proposal 历史分页／定向读取；正常准入仍独立验证来源。前 32 条信息窗口问题不能再笼统列为未修。
- [x] Pro 显式 `retrospection_window=head_tail` 的有界首尾笔记读取。它不是已经实现持久研究记忆或完整历史证据检索。
- [x] 预算 N+1 查询合并为一次读取，保留逐行校验和任意精度整数累计；同时修复 SQL ID 与 permit ID 错配可能漏审本行的问题。
- [x] ingress 空解析批次跳过历史 ID／配置缓存读取；非空只查本批最多 128 个 ID，保留真实 insert/exact/conflict 与 ACK 顺序。
- [x] 不兼容存储的提前拒绝；CPU-only 在旧机 CephFS 上真实 canary 通过。不能把此项误写为“CephFS 角色／GC 已兼容”，也不能写成“CephFS 上所有 Core 都不能运行”。
- [x] 最新固定两仓回归、离线包／安装验证及独立证据核验；没有生产服务切换。

最新验证：Core **4,733 passed / 7 可选 Pro skipped / 2 既有 warnings**；Pro **1,037 passed**；可选配对 **31 passed**；新安装 Core **35 passed**。安装组含一个真实 CPU worker 的 TREE/ECHILD、effect、产物及 SETTLED 证明。不同测试集合重叠，不能相加。

Core 的 505 个旧测试文件有 504 个字节不变，唯一例外仅四处并发注入点迁移，原 26 个断言 AST 不变；Pro 的 140 个旧测试文件全部字节不变。不要声称这轮所有旧测试都完全未改，也不要把该夹具例外扩大到其他断言。

## 4. 未完成 TODO：建议接续顺序

### P0：恢复新机开发与验证基线

- [ ] 检查新环境的 AGENTS.md／仓库指令、两仓提交、工作树及私有仓访问权限；确认不是旧占位 Pro。
- [ ] 建立新的隔离 Python 环境，记录解释器、依赖和文件系统。旧验收使用 Python 3.10 / Linux x86_64；项目声明的其他 Python 版本并未因此获得同等现场验收。
- [ ] 从固定源码／锁重建制品和测试环境，核对实际模块导入路径；不得复用旧机 venv 路径或因版本号一样就混用旧 wheel。
- [ ] 在新机短路径、私有测试目录中复跑针对性基线，再决定完整回归；将环境缺失、真实产品失败和检查器错误分开留证。

### P1：降低长历史成本，保持相同拒绝边界

- [ ] 分别 profile 预算行校验里的重复 JSON 规范化，以及同一次 Policy 只读分页／选择中的重复全历史 budget snapshot。先量化贡献，再定下一小片合同。
- [ ] 评估是否能减少同一行重复解析／复制；若评估同版本 snapshot 复用，先证明完整失效条件，不直接加入全局预算缓存。
- [ ] 失效和回归覆盖：同／异连接写入、回滚、schema／连接／路径替换、Stop、进程内 `_HELD`、损坏历史行、active slot 冲突、精确累计、恢复及 GO 前检查。不能只检查 `data_version`。
- [ ] 继续评估非空 ingress 的 legacy 配置缓存、源文件／sidecar 总量及全循环重复扫描；不能把已经消除的全表 ID 查询再算一次新成果。
- [ ] 对 100／1,000／5,000 及更长历史做同机交替对照，留原始分布、查询次数、内存／I/O和行为差分；实际产品链另验，不用合成 SETTLED 行冒充真实执行。

当前有限实测：5,000 条预算行的 SQL 从 5,001→1，汇总中位数 999.41→931.20 ms；这组合成 SETTLED 行仍执行 25,000 次 decode，完整审计仍 O(N)。不是整体研究吞吐提升证明。

入口：`src/orze/core/cpu_action_budget.py`、`src/orze/engine/cpu_phase.py`、`cpu_policy_evidence.py`、`cpu_proposals.py`、`idea_ingress.py`、`src/orze/idea_lake.py`、`src/orze/core/integrity.py`。

验收：合法决策和结果不退化，坏输入仍拒绝，历史预算／Stop／HOLD不被绕过；成本收益来自可复核测量而非只数 SQL。

### P1：Pro 长程合格证据检索与大 snapshot 规划

- [ ] 检查当前全历史 qualification 后再裁剪上下文的流程；为真实 research consumer 设计有界查询／分页／定向选择，不只是给 Core 增加一个未接入的 API。
- [ ] 保留 Objective、comparison scope、覆盖声明、协议／来源版本及当前 attempt 身份；invalid／unknown 不能因排序、截断或缓存变成 valid。
- [ ] 设计 evidence／proposal／预算联合 snapshot 的总载荷预算：单个大记录、多个记录合并、跨页遗漏均要显式处理，不无限增加 prompt 上限。
- [ ] 验证关键历史位于窗口之后、后页含反例／冲突、来源变更及跨轮检索时，真实策略能发现并做正确后续决定；没有足够证据时不得声称已收敛。

私有 Pro 入口：`src/orze_pro/agents/evidence_context.py` 的 `ranked_evidence` → `research_context.py` 的 `build_context` → `research.py` 的 `run_research_cycle`。已有 `load_current_digest` 会重建当前证据视图，不能把它当作本项已经全部完成。

### P2：持久研究摘要／记忆

- [ ] 先定记录身份、格式／版本、容量和失效协议：保存问题、假设、决策理由、已证实／被反驳／未确认事项，以及可追溯的原始来源。
- [ ] 摘要必须区分研究者笔记、派生视图和合格证据；不得通过摘要自我认证，必须能返回当前原件核实。
- [ ] 重启后的研究上下文可恢复，但不能因此获得旧进程或副作用接管权。来源换代、删除、损坏或资格规则变更时应失效／降级。
- [ ] 做真实多轮及重新启动的研究验证，证明不丢关键反例、不重复已确认无效尝试，不以累积全文替代有界记忆。

### P2：生产存储与受控上线

- [ ] 在新机实际配置的锁、checkpoint、GC 源／目标路由重新预检。旧机 CephFS 缺少所需原子 no-replace 操作的结果，不是新机文件系统能力证书。
- [ ] 解决角色 release／GC 的存储兼容：明确受支持的部署存储，或单独设计等强度协议；不得降级为覆盖 rename、先检查后覆盖，或随意移锁到 `/tmp` 绕过所有权保护。
- [ ] 保留碰撞、FD 身份、路径替换、同步失败、Stop／HOLD和未知副作用等边界；CPU-only 路径不得被无关角色／GC限制误禁。
- [ ] 明确目标机器／服务／项目、实际制品及存储、旧 owner 闭合、状态兼容、备份和回滚；核对真实 Pro 许可及实际模型／GPU／费用授权。
- [ ] 审查现有 service installer 的全局服务名和即时启动行为，不能把它当隔离 canary 安装器直接执行。
- [ ] 先独立环境 canary，再受控切换，并实际演练启动、停止、重启、故障恢复、回滚；形成真实目标环境的验收证据。

入口：[存储合同](../storage-preflight.md)、`src/orze/engine/storage_preflight.py`、`gc_safety.py`、`gc_tree.py`、`src/orze/service/install.py`；Pro 侧 `src/orze_pro/engine/role_runner.py`。这些动作未完成，不能因 wheel 安装或离线许可替身测试成功而打勾。

### P3：端到端研究提速验证

- [ ] 在以上机制可用后，冻结跨领域、等质量的对照；记录到有效证据／确认选择的延迟、调用和资源成本、无效／未知结果及失败率。
- [ ] 保留负对照和无收益任务；通用规则不能暗中依赖 ASR 名称或某个示例的预知答案。
- [ ] 真实 provider／GPU 研究须使用明确授权的资源；不设置产品研究总截止时间。有限验收运行的停止／取消规则须与产品自主决策分开说明。

已有一次 48 项目／24 配对的有限 CPU 对照：两个默认示例少一次分析，两个反事实没有同样收益；不能声称从未做过对照，也不能泛化为通用科研已经提速。见[原正式结果](2026-09-12-research-efficiency-results.zh-CN.md)。

## 5. 证据与制品：新机如何核实

优先阅读：

1. [最新性能验收与边界](2026-09-14-history-cost-results.zh-CN.md)。
2. [长程访问结果](2026-09-14-long-history-results.zh-CN.md)：其中“下一片预算／ingress”已由上述最新结果部分关闭，不要照抄为全未做。
3. [最终独立核验](../evidence/2026-09-14-cost-final-independent-review.json)，SHA256 `64027e20c533d9631e339df534cd5d48fe17af5d476b9a53a7001cef30744f88`。
4. [Pro 私有原件固定提交索引](../evidence/2026-09-14-cost-pro-private-index.json)。
5. [存储合同](../storage-preflight.md)与[生产预检清单](2026-09-12-production-readiness.zh-CN.md)：后者是历史部署清单，不是最新制品版本说明。
6. 原目标追溯：[V1 实施方案](2026-09-10-autoresearch-v1.zh-CN.md)、[V1 补缺机制验收](2026-09-12-autoresearch-v1-closure.zh-CN.md)。

最新制品 SHA256：

```text
Core wheel: 1070fd9417ece854e368d01a96dccd3c0723bd74bf8d0d1b6058cadc999dca74
Pro wheel:  6509c7449e0aee3a6b62d56cbe95566011df545e4c4c74e3c962f64a244274b5
```

两包版本仍为 4.6.2／0.13.1，不能靠版本号选包。完整配对制品在私有 Pro 的 `docs/evidence/runs/2026-09-14-cost-package/package-payload.tar.gz`，SHA256 `086981d62c0a2141af7a5e2f8f484ef24c05b4f9e1f0e93d4502b6af114b6a31`。对应索引、锁、构建／安装原记录在同目录，包含 23 个外部依赖 wheel；不同平台应重新构建并建立自己的验收，不能盲用旧二进制。

新机注意：

- 历史记录中的 `/hot-data/...`、`/tmp/...`、已安装路径和机器身份都是历史事实。部分检查／打包脚本依赖这些绝对路径；先读脚本，复制到新验证目录做显式适配，不覆盖旧脚本、旧日志或历史结果。
- 不复制整个 venv 当迁机安装。离线安装使用核对过的精确 wheel／hash 锁；源码测试环境与正式制品环境分开。
- `docs/evidence/runs/2026-09-14-cost-validation/run_frozen.py` 可记录新的命令、stdout／stderr和前后指纹，输出目录必须全新；`check_original_tests.py` 用于特定基线及精确夹具例外，不是永久允许修改旧测试的豁免。
- Core 常规 pytest 从 `tests/` 收集，不要执行整个 evidence 目录中的旧测试快照。新机建议短路径 basetemp，避免把 AF_UNIX 路径限制误判为框架逻辑错误。
- Pro 历史回归使用测试进程内许可替身，只能作为离线测试方法；不能把它复制进生产代码或生产验收。
- 安装验证必须确认加载的是安装的 Orze 模块。根 `conftest.py` 会插入工作区 `src`；既有 installed harness 因此只排除该根文件，保留 `tests/conftest` 并逐模块核对 wheel/Git。不能一面从源码导入，一面声称验证了安装包。
- Core 的 `.gitignore` 默认忽略 `.log`。提交新证据时显式核查已跟踪的原始日志；只对确认过的本轮证据文件使用 `git add -f -- <精确路径>`，不批量强加临时文件或秘密。

## 6. 不能随迁机顺手完成的事情

复制的历史 SQLite／owner／claim／receipt 是证据或备份，不是可直接启动的 live namespace。不得重签其中的路径、dev/inode、PID、start_ticks、boot、nonce 来绕过 HOLD；旧 FD、锁和控制器授权不能迁移复用。新机使用全新私有测试项目，真实运行态迁移另定协议。

远程集群接管、未知非幂等副作用自动裁定、通用旧进程 adoption、训练 checkpoint resume、任意 Director 交接等仍不是本版已提供的通用保证；不要将“恢复已有上下文”扩成这些能力。

## 7. 交给下一位执行者的工作方式

从 P0 开始核对，再优先定位 P1 的真实成本和 Pro 消费链。每次只冻结一个可验证小改动：保旧基线／失败 → 实现 → 定向与真实产品验证 → 两仓回归 → 制品核验 → 独立审查 → commit/push → 远端读回。原件与派生汇总分开，元数据夹具与真实 worker 分开。

没有新的方向选择时按此清单继续推进，不需要用户逐次催促；涉及目标服务切换、真实账户／费用／设备、所有权或备份破坏的新增决定时，明确指出依赖并取得必要方向，不能自行扩大授权。
