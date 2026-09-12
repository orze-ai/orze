# 研究提效第一轮：实际结果与部署边界

第一项有界提效已实际运行：在两个默认 CPU 任务上，保持相同有效选择及独立 attempt 复验，少执行一次无法改善选择的 validation-only 分析。**不是通用科研效率已经解决，也不是生产已切换。**

## 冻结与运行

协议提交 `9120ff5`；完整应用/计量/测试在 `ff53afc9d9eee6ea468d17cca9ddd720485633ea` 冻结，干净工作树启动。正式轮仅 `formal-01`，按预注册顺序完成 24 对 / 48 个新 CLI 项目，未替换失败或挑选重跑。两臂均使用同一个原 Core 4.6.2 wheel（SHA `8c2cbb4f3def9a22ecfb4990c2675a9c2f84ee0550c54aad2ccee9075bf8cc4c`），非 editable 安装、无 Pro、GPU 或付费 provider。

实际命令与逐次工具输出见 [执行记录](../evidence/2026-09-12-efficiency-formal-execution.json)。启动工具 `12fdae`，session `2061`，最终 `d0737d` exit 0。首个 CLI 开始至最后一个结束为 165.111 秒。运行前后应用及已安装 Core Python 共 240 个文件，完整 keyset 与 SHA 一致。

## 主终点及实测耗时

A 为原 CommonPolicy，B 为显式 opt-in 的 DominancePruningPolicy。每行均为 6 个 AB/BA 配对；wall 列是各臂完整 CLI 耗时中位数，“配对差”是各对 B−A 的中位数，**不是两个中位数相减**。

| 原公开任务 | 动作数 A→B | 已结算保守额度 A→B | CLI wall 中位数 A→B | wall 配对差 |
| --- | --- | --- | --- | --- |
| 排序默认 | 5→4 | 10→8 秒 | 4.859→3.374 秒 | −1.525 秒 |
| 压缩默认 | 5→4 | 10→8 秒 | 4.944→3.455 秒 | −1.460 秒 |
| 排序反事实 | 3→3 | 6→6 秒 | 2.689→2.694 秒 | −0.026 秒 |
| 压缩反事实 | 3→3 | 6→6 秒 | 2.691→2.656 秒 | −0.035 秒 |

两个默认任务的动作数和保守额度均降低 20%；两个反事实任务没有该收益，不把其微小 wall 波动解释为优化成功。四个任务混合总量为 A 96 / B 84 个动作、A 192 / B 168 秒保守额度，即这四个任务等权混合下两项总量下降 12.5%；该混合取决于人为任务比例，不能将默认任务的 20% 泛化到整个任务集。额度不是实际 CPU 时间、API 费用或退款。

完整次终点在 [原始汇总](../evidence/runs/2026-09-12-research-efficiency/summary.json)：首次 valid 被 policy 消费、确认选择、worker CPU/wall 与 native elapsed 全部保留。首次 valid 没有一致改善；本次收益来自后续少做分析。CLI wall 含解释器、控制器、进程启动、存储和轮询等开销，不把它减去 worker wall 称为纯框架 CPU。

## 质量与覆盖代价

48 次均正常退出；180 个实际 native attempt 均 completed、TREE_CLOSED/ECHILD_WALL、effect 确认，180 个预约全部 SETTLED；未执行项、非零/未知退出、质量失败均为 0。计量逐项核对完整 Ref、捕获的 supervision binding、终态 SHA、已发布观察/产物和 trace，不以日志“成功”字样代替账本证据。

24 个配对的选择完全一致：排序默认 baseline/cost 9、压缩默认 baseline/cost 137、排序反事实 challenger/cost 0、压缩反事实 challenger/cost 12；协议、数据、比较范围和动作签名相同，均有真实新 attempt 复验。这里的质量是声明协议下的有效选择，不是所有可能候选中的全局最优。

覆盖代价必须保留：A 取得 72 valid / 12 invalid / 12 unknown 观察，B 为 60 / 12 / 12。B 少了 12 份对落后候选的有效分析观察，unknown 没有被改写成 valid，也没有伪造已执行的分析。它适合“找到并确认当前最佳选择”，不等同于完整鉴定全部候选。

## 复核和可复现证据

- 新策略及计量冻结测试独立复核：首稿实际 4F48P，修复后原样 52P；[完整 red→green](../evidence/2026-09-12-efficiency-measurement-review.json)。四个失败属于新计量程序，不冒称历史 Core bug。
- 包含原 acceptance 回归及包资源断言的运行：实际 114P / 14.13 秒，[记录](../evidence/2026-09-12-efficiency-preformal-regression.json)。其四个基线 CLI、先前两个开发 smoke、三个安装 canary 均不计入正式 48 次；不是全仓测试的新结论。
- [完整归档索引](../evidence/runs/2026-09-12-research-efficiency/index.json) 包含完整原始目录的压缩包，保留数据库、日志、48 个 run.json、产物和闭合回执；不是仅保留派生指标。
- [独立正式复算](../evidence/2026-09-12-efficiency-formal-independent-review.json) 使用不调用作者 qualify/summarize 的 stdlib 检查器：`9e821a` exit 0，48/48、180 个实际 attempt 与 prepared/committed 记录通过，四组全部中位数与配对差精确吻合。两次较早的检查器格式/namespace 前提误读及修正原样保留，不记为实验失败，也未重跑实验。

原始持久目录为 `/hot-data/fsx/workspace/erik/orze-production-validation-2026-09-12.UgS3uV/formal-01`。manifest SHA `b0a165ab2311deeb51ba4fd9c7b7200706683c495c0b3a55178e8d87f968c819`；summary SHA `4c586c565501fe45766c71ae827c44d121f4732adc9840660689fb071ac3e72e`。源码、固定数据、wheel 和依赖均由 manifest/部署证据定位。

## 对产品与上线意味着什么

本次在两个既定领域示例中复用了同一剪枝规则：协议明确保证仅验证、成本严格劣于可比 incumbent 时，不必为当前选择补做分析。新领域仍须独立确认 validation-only 协议前提。实现放在应用 Policy，Core/Domain/worker/数据/判据不变。当前代码仍适配有限 acceptance 候选布局，**不是任意科研任务开箱即用的优化器**；没有默认打开，也没有将领域名或 WER 写进 Core。

生产准备另见 [部署预检](2026-09-12-production-readiness.zh-CN.md)：修复了 wheel 漏打包操作指南，真实构建、哈希锁安装、pip check、安装入口与 Core-only canary 通过；运行时 Python 和依赖未改。修复版制品在独立 venv，原实验环境保持不变。

现有 [service installer](../../src/orze/service/install.py) 仍使用全局 `~/.orze_service.json`、固定 `orze.service` 并执行 `enable --now`，不是按项目隔离的 canary 安装器；本轮未调用。目标明确后还须处理服务隔离和回退，不能把 wheel 安装通过当成生产服务升级通过。

尚未完成的是指定目标的生产切换与真实模型/GPU研究对照。仍需明确机器/服务/项目、Pro 许可（如使用）、模型/GPU范围、费用和时长上限；再按旧 owner 闭合、状态备份、独立新环境、受控 canary、切换/重启/回退的顺序验证。当前 ASR 脏工作树、全局安装和服务未改，未发布包或合并 main。本轮 CPU 数据不能代替这些验收。
