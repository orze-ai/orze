# 预算重复规范化：实施与验收

本片继续机器交接的第一项 P1，处理每行预算校验中的重复 JSON 解析／规范化，并量化真实 Policy 续页中的重复预算审计。通用 autoresearch 的其余 P1、持久记忆及生产上线仍未完成。

起点为 Core `558f8c69db03eb7a5087d8200f54d4ce0fe46dcd`，配对 Pro `ac937e4`。工作分支为两仓的 `codex/p1-history-validation`。Pro 本片仅增加验证记录，不改源码和测试。

固定实现提交为 Core `7b6e0b8b950365beffae717865b368b5bb2a3e27`；[提交后制品复核](../evidence/runs/2026-09-14-budget-normalization/verification-committed.json)再次从 Git 读取所有 wheel 包文件，比对实际安装与原始验证记录，结果通过。

## 实现边界

预算的 declaration、scope、permit 字段检查与生成独立 canonical 副本分开。扫描先解析真实 SQL 行中的完整 permit JSON，再校验其字段，不再为了丢弃的副本而重复编码／解析同一对象。对外归一化入口继续返回独立 canonical 对象。

每次扫描仍读取并校验全部历史。canonical JSON、重复键／大小限制、scope 哈希、SQL／permit 身份、状态、ref、槽位冲突及 Python 任意精度累计继续保留。没有引入预算缓存、snapshot 复用、数据库版本授权、schema 迁移或退款。正常 reserve、bind、settle、Stop/HOLD、恢复与执行前 require_permit 仍经过原有事务／复核路径。

[独立进程检查器](../evidence/checks/2026-09-14-budget-normalization-verify.py)用固定 Git 原件检查 AST：37 个其余函数及所有模块级声明保持原结构，三个校验器保留原字段检查分支，扫描唯一变化是省去重复副本。它同时校验原测试、原始日志、JUnit、制品与实际 SQLite。这里的“独立进程”指机械复核，不代表另一位作者或人工审查。

## 同机交替对照

第二轮每组先预热，每个版本交替运行五次；100／1,000／5,000 条当前 scope 行之外各有等量其他 scope 行。记录的原始分布、I/O 计数、tracemalloc 和 cProfile 均在[报告](../evidence/runs/2026-09-14-budget-normalization/benchmark-v2.json)。计数、内存跟踪和 profile 不进入耗时区间。

| 当前 scope 行数 | 预算汇总中位数 ms：旧 → 新 | Policy 只读续页中位数 ms：旧 → 新 |
| --- | --- | --- |
| 100 | 18.33 → 11.28 | 70.91 → 57.27 |
| 1,000 | 187.50 → 114.52 | 401.79 → 260.84 |
| 5,000 | 919.77 → 572.52 | 1,885.35 → 1,175.65 |

5,000 行时，两项局部耗时均下降约 38%。[第一轮原件](../evidence/runs/2026-09-14-budget-normalization/benchmark-v1.json)也得到类似结果（汇总 916.86→568.06 ms；续页 1,881.99→1,169.49 ms）。第一轮启动时全局 Pro 占位包被尝试导入并拒绝；第二轮在诊断 harness 中屏蔽可选 Pro 导入，CPU 产品实现不替换。

每条 SETTLED 元数据行的 `_decode` 为 5→2，`_json` 为 13→7；RESERVED 行只解析一次 permit，BOUND／SETTLED 另解析一次 ref。5,000 行汇总的 Python 跟踪峰值为 26,009→16,038 字节，续页为 183,760→161,994 字节。这是单独跟踪运行的 Python 分配峰值，不是进程 RSS 或机器总内存。

真实 `Orze.iteration`、BoundPolicy 和 EvidencePager 的续页仍各做两次完整预算扫描：一次 admission，一次 callback 预算输入。第二轮 5,000 行续页的解析为 50,012→20,006 次，规范化为 130,034→70,022 次，完整审计次数仍为 2。因此这片没有消除跨入口或跨页的重复全历史审计。

这些是私有 `/tmp` 合成账本上的协调开销。33 个分页前缀也明确为元数据夹具；没有把它们计为真实 worker、结算历史或研究收益。机器上同时存在回归测试，未声称独占主机或生产共享文件系统吞吐。两个版本逐次输出一致，前后 SQLite 逻辑内容指纹一致。I/O 数据保留原值，其中包含读取 `/proc/self/io` 自身的开销。

## 验证状态

Core 全量执行后，两个目录名夹具错误通过移动工作副本、原测试定向复跑闭合。固定源码／测试的最终覆盖为 4,775 passed、7 个可选 Pro skipped；这是一次全量加两项复跑，不冒充一次全绿的全量运行。Pro 全量与可选配对分别完整通过。

| 集合 | 已完成结果 |
| --- | --- |
| 新增检查在旧代码上 | 3 个预期成本失败、39 通过 |
| 预算、并发、原生执行、Stop/HOLD 与恢复定向 | 179 通过；57.87 秒 |
| Core 全量原始结果 | 4,773 通过、2 个目录名夹具失败、7 可选 Pro 跳过、2 既有 warning；827.65 秒 |
| 移到真实 `orze` 目录后原两项测试复跑 | 2 通过；0.13 秒；与全量相同的源码／测试指纹 |
| Pro 全量，测试期许可门替身 | 1,037 通过；259.85 秒 |
| Core／Pro 可选配对，测试期许可门替身 | 31 通过；0.89 秒 |
| 修正工作副本布局后的控制器／交接集成 | 16 通过；27.46 秒 |
| 新安装 Core 上的预算与真实 CPU 产品检查 | 58 通过；43.83 秒 |

以上集合重叠，不能相加。Core 测试目录中的 508 个既有文件（506 个 Python、2 个 JSON）及 Pro 140 个既有测试文件逐字节保留；只新增一个 Core 测试文件。安装测试保留原 `tests/conftest.py`，只排除会插入工作区 src 的根 conftest，并使用外部 pytest harness。111 个实际加载 Core 模块必须匹配新 wheel 与源文件字节。五个实际产品项目共六次 native CPU 执行，检查器逐一关联复核其 TREE_CLOSED／ECHILD_WALL、TERMINAL 与 SETTLED 的 ref 和 terminal 哈希。

新 wheel SHA256：`1f3aef56bdfa3525af3e2af4f2ae6f30121be33c5a4d0cf619102233f33e1860`。新 venv 使用已有离线依赖完成安装、`pip check` 和实际 CLI help；全部包文件及安装 RECORD 逐项比对。没有修改全局安装、用户研究仓库或已有服务。

保留的验证错误如下：首次 Core 全量因任务目录使用 `core/pro`，而旧集成夹具固定读取相邻 `orze-pro`，误加载了全局占位包；该次运行被中断，原始日志不含完整汇总，不作为通过证据。新增任务内 `orze/orze-pro` 路径后，原 16 项集成测试通过并重新全量。这次全量的两项 `test_launch_policy_latency.py` 仍要求解析后的真实路径以 `/orze/src/orze/cli.py` 结尾；随后把 Core worktree 移到真实 `orze` 目录，原两项断言直接通过。最终检查器核对所有测试身份、两次输入指纹及精确失败原因，不隐藏失败或修改旧断言。安装首次遇到系统 `ensurepip` 缺失，改为 `venv --without-pip` 加已有离线 pip wheel；第一次 help 检查误用无 `__main__` 的 `python -m orze`，修正为实际入口 `python -m orze.cli`。这些为验证环境／检查器问题，原始失败日志及初版脚本保留，没有修改产品来迎合夹具。

## 原件与剩余事项

- [预算基线与定向原始日志](../evidence/runs/2026-09-14-budget-normalization/targeted.log)
- [安装产品测试原件](../evidence/runs/2026-09-14-budget-normalization/installed-product.json)
- [Core 全量原件](../evidence/runs/2026-09-14-budget-normalization/core-full-corrected/run.json)与[两项路径复跑](../evidence/runs/2026-09-14-budget-normalization/core-path-rerun/run.json)
- [独立进程机械复核结果](../evidence/runs/2026-09-14-budget-normalization/verification.json)
- [压缩原始项目、SQLite、包与逐文件哈希索引](../evidence/runs/2026-09-14-budget-normalization/archive-index.json)
- [可重复运行的对照脚本](../evidence/checks/2026-09-14-budget-normalization-benchmark.py)

Pro 完整日志、源码清单与配对记录保留在私有 Pro 仓，不复制到公开 Core 仓；[固定私有提交索引](../evidence/2026-09-14-budget-normalization-pro-private-index.json)对七份入口原件从 Git 重新读取并核对哈希。未执行真实 Pro 许可／付费 provider 验收，没有 GPU 计算、生产切换或包发布。

下一片仍应处理一次只读决策中的重复预算审计（若考虑复用，必须证明全部失效条件）；非空 ingress 的全局缓存／源与 sidecar 总量成本也未关闭。另一项 P1 必须把有界检索接入 Pro 的真实 research consumer，并联合限制 evidence／proposal／budget 的上下文载荷。当前 Pro `ranked_evidence` 仍逐个资格校验全部 completed 历史再排序；Core 原生 CPU pager 与这条 legacy report 消费路径并非相同数据合同，不能直接接上 pager 名称就标为完成。
