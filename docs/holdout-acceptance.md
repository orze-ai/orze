# 原始排程 holdout：冻结 Core 的声明式跨领域验收

本片依照[冻结执行方案](plans/2026-09-11-v1-07b-holdout.zh-CN.md)，使用[原始独立挑战](evidence/challenges/v1-07b-original.md)。挑战在 A 闭合后首次打开，原文未改；公开后的同一实例不再称作新的未见任务。

## 实际接入与结果边界

Core 产品源码仍为 `8f61ae5f68c0846d5149aceb4df55c69263c9df2`，Pro 实现不变。新增 `examples/holdout` 应用：登记 SchedulingDomain，调用原 CLI；producer 输出原始排程 artifact、零 observation，evaluator 通过现有封印来源 FD 读取候选并独立产生测量。没有第二套研究执行器、SQL 写入旁路或新增 Policy。

使用既有 QueuePolicy（idle=wait）、公开 IdeaLake.insert(if_absent=True) 普通准入及 CLI replicate。验收程序显式声明任务次序；这证明声明式任务跨领域接入，不证明 CommonPolicy 对 maximize 的无代码迁移、自治提出排程候选或完整 autoresearch 自主性。A 的共享 Policy 自产分析与反事实验收保持原样、单独成立。

原实例的真实结果为 baseline 有效 0、greedy challenger v1 有效 30；这两个值从实际候选重新计算，不是最优参考答案。原 artifact 在 v2（唯一变化为容量 1）下为 capacity_overload，无可排名数值。v1 观察及来源保持不变，v2 独立 protocol/comparison_scope，不能混榜。

主工作流 7 次 native action：两个 producer、baseline 评估、一次真实失败评估、恢复评估、v2 重评、一次显式复验。另五个私有项目各 2 次 action，分别覆盖精确 deadline / 半开 end-start 正控，以及重复 ID、缺失先决任务、容量超载、坏 JSON 四种单故障负控。invalid 是正常完成的领域判定，不是执行失败或 HOLD。

每个完整产品 epoch 共 17 次 native action、34 秒保守预留额度、22 次真实独立解释器 CLI 调用。CLI 准入 ACK 和三次空闲 Wait 不算研究动作；另有四项 supervised worker 单元测试，不并入这 17 次。执行计数来自实际 attempt 身份与预算记录，不用计划数替代。

## 失败、复验与空闲

故障开关仅在指定 evaluator 的实际子进程中生效：读取绑定来源后写出部分 evaluation.json，再真实 exit 71。原配置、purpose、动作输入和协议未加盐。旧 attempt 先有真实 TREE_CLOSED / ECHILD_WALL、failed 终态及 SETTLED；partial 没有 artifact/observation 发布。

新的 Python controller 以新的显式 task、完全相同 raw config 和来源恢复评估。旧失败 task 的同 ID 准入精确重放不重置旧状态。这不是接管旧 attempt，也不是控制器崩溃恢复。

对已完成 evaluator 的公开 replicate 先入队、后经普通预算和执行路径产生独立 occurrence；相同 request ID 重放不再创建或执行。相同结果只称数值相等，不称统计独立。三次真实空闲调度只新增 Wait 决策，不新增任务、attempt、预算预约或观察。

## 证据、计量与可重放性

完整原始报告保留 CLI 命令/输出/退出、controller 自身 PID 与 birth ticks、实际外层监督闭合证明、各步骤权威表快照、原 artifact 字节、来源/协议/观察绑定、失败 partial 和结算记录。外层监督器仅用于测试拥有的 controller 进程隔离；研究 worker 始终由原生 Orze 路径控制。没有 host PID 扫描、GPU 或 provider 调用。

独立领域判据使用事件边界的容量重算，与应用的整数 tick 扫描不同，且不导入应用 Domain 或 Core。其独立证明限定于原实例和列明候选；没有声称独立 checker 与 Domain 在所有配置边界等价（例如负 job value 及输入上限）。独立产品测试在首次产品运行前冻结，修复记录器时未修改其断言。

首个候选 epoch 85 passed / 3 failed：准入与执行快照标签重复，独立定位正确拒绝歧义。保留完整原轨迹、原 helper 和原产品测试；新增唯一性回归先红后绿，记录器仅加序号前缀。三处失败和该回归是同一测试记录缺陷，不是四个 Core 缺陷。正式目标为 89 passed，4 个新测试文件 / 171 条 literal assert；旧测试和已有 A 文件不改。详见[候选历史](evidence/2026-09-11-v1-07b-target-history.json)和[最终机器证据](evidence/2026-09-11-v1-07b-holdout.json)。

CLI wall 包括为重启验收而反复创建的控制器进程、启动、调度与持久化；worker body CPU/wall 不含解释器导入及最后 JSON 编码/写出。两种量测不是同一口径，预留 34 秒也不是实际 CPU 用量。不由有限、故意反复启动的本地验收推导稳态效率、GPU 节省或普遍研究收益；完整 target/full 两轮都保留，不挑较快者报告。

运行测试：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:. CUDA_VISIBLE_DEVICES= python3 -m pytest -q -p no:cacheprovider tests/test_holdout_product.py tests/test_holdout_product_review.py tests/test_holdout_domain.py tests/test_holdout_domain_review.py --tb=short`。重放会创建自己的临时项目及本地 CPU 子进程，不依赖原临时路径仍存在；归档 checker 则只读已保存的原字节。

## 仍未完成

本片不覆盖控制器丢失 RUNNING owner、未知 publication HOLD 的自动裁定、TERMINAL/effect 确认但预算未 settle 的崩溃窗口、原生 repair/adoption 或受监督 Director 正向交接。先记录的已确认 evaluator 失败后新任务重评不能替代这些保证。整个 V1 尚未完成。

历史 A 的完整证据 checker 固定自己的 746/223 文件集；新增 B 后应在其固定成对 worktree 重放 A 历史检查，不能删除新文件或修改旧 manifest 来造绿。B 的全量会再次运行原 A 测试，原始 A 报告另存并与 B 调用计数分开。
