# 跨领域 CPU autoresearch 验收

本片 V1-07A 是两个已知异构任务的应用接入验收，不是整个 V1 完成声明。产品实现固定在 Core `8f61ae5f68c0846d5149aceb4df55c69263c9df2`；源码、产品配置/schema、原有测试和 Pro 实现均不为示例修改。约束先于实现提交在[冻结协议](plans/2026-09-11-v1-07-acceptance.zh-CN.md)。

## 接入与实际闭环

`examples/acceptance/__main__.py` 仅登记两个 Domain 和一个 CommonPolicy，然后调用真实 `orze.cli.main()`。执行、提案入队、预算、claim、READY/GO、封印来源、产物/observation 发布及终态结算继续经过已有 Orze 主循环；没有第二 runner 或应用直接写 Lake 的路径。脚本 worker 只依赖 Python 标准库及同目录 common 模块。

两个 Domain 共用 Policy 源码、空策略参数、2 秒/动作上限和 10 秒/项目额度。探索的三个候选是明确的示例先验，赢家由 recorded valid/comparable cost 决定，不按领域名、预期数字或动作次数选择。Policy 是有限候选的确定性最小化策略；不是通用候选发现算法，也不是 Core 默认研究规则。

| 数据与任务 | baseline | challenger | unchecked 初测 → 独立分析 | 实际复验对象 | 动作数 |
|---|---:|---:|---|---|---:|
| 默认排序 | valid / 9 次比较 | invalid / 0 | unknown / 29 → 新 valid / 29 | baseline | 5 |
| 仅改为已排序输入 | valid / 8 | valid / 0 | 未提出、未执行 | challenger | 3 |
| 默认编码（含 NUL） | valid / 137 字节 | invalid / 12 | unknown / 266 → 新 valid / 266 | baseline | 5 |
| 仅删除输入 NUL | valid / 136 | valid / 12 | 未提出、未执行 | challenger | 3 |

默认路径的分析输入是本次 baseline/challenger/unchecked 刚发布的三个真实 artifact IDs，经封印 FD 读到绑定字节。分析产生新的 evaluator/observation，原 unknown 整份记录不被升级或改写。分析复用原算法测量，仅执行验证；不是新的算法测量。

Replicate 使用已选择的完整 Ref，raw config/action spec 不加盐，产生不同 task/attempt/artifact/observation。只有实际副本完成、整树闭合、effect 登记并 SETTLED 后，真实项目才主动 Stop。默认 Stop 虽剩余额度为零，原因仍须为 `confirmed_selection`；反事实 Stop 剩余 4 秒。独立 occurrence 不自动等于统计独立。

Policy 内部通过同 action spec 的另一个已完成 occurrence 关联确认，不独自提供 request ACK 或恢复授权；产品测试另核对持久 replication request 的 source Ref、request ID、目标 task 和终态。纯 Policy 反例若把 analysis 的 valid cost 改成最优，选的是 analysis evaluator：复验该分析并不等于重跑仍未知的源算法。

## 领域判据与身份

排序记录每次实际值比较，包括 False；不把循环条件、复制、交换或 Python 指令计入指标。独立重放完整 trace，另检查输出顺序与元素重数。有限输入通过不证明候选对所有输入正确；原样输出在已排序反事实下有效正是这个边界。

编码指标计 RAW1/RLE1/HEX1 的完整容器：4 字节 magic、4 字节大端原始长度和全部 payload。严格 decoder 拒绝不匹配长度、坏编码与尾随垃圾，再检查逐字节 round-trip。JSON 中的十六进制是传输表示，不把它的文本大小当容器大小。dataset_sha256 对 canonical JSON 求哈希；编码任务的 JSON 值是显式 hex 字符串，不是原始 bytes 的 SHA。

unchecked 的首次结果是完整、正常完成、尚未执行科学判据的 unknown，不是坏 envelope、缺产物或 HOLD。压缩应用草稿曾提前调用 round-trip 再覆盖状态；新增行为红测真实复现，完整草稿与原 30 项测试已封存。修复仅把 unknown 分支前移，原 decoder 与判据不变；这是新示例的语义缺陷，不冒称 Core 历史缺陷。

## 可重放入口

在源码仓可运行四个独立新测试文件：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. CUDA_VISIBLE_DEVICES= python3 -m pytest -q tests/test_acceptance_product.py tests/test_acceptance_policy_review.py tests/test_acceptance_sorting.py tests/test_acceptance_compression.py --tb=short
```

其中 9 个产品用例和 8 个独立 Policy 用例共用一次四项目 session fixture：16 次 native action、4 次成功 CLI 调用和 4 次持久 Stop 拒绝调用。剩余领域单元测试的独立本地 worker 不叠加进这 16 次；纯 snapshot 修改没有提交 observation 或执行任务。第二次 CLI 在同一解释器中调用，重新隔离应用登记；这是持久 Stop 的再次入口拒绝，不是控制器进程崩溃/重启验收。

`testing.py` 是测试插件，仅隔离注册、argv/cwd、信号环境，并把 GPU/legacy/provider 入口设为禁止调用；不替换执行器、Lake、预算、终态发布、Domain 或 Policy。插件在运行后输出 `ACCEPTANCE_REPORTS` 路径和 SHA，并保存完整原 stdout、解析后的真实 trace、九张表的只读行快照、产物 envelope、时间与第二次返回值。第一次尚无 exporter 的候选 17 项绿测不补造遗失的父进程计时/轨迹；最终证据用修复后实际重新运行并落盘的数据。

真实非测试入口也是 `PYTHONPATH=src:. python3 -m examples.acceptance -c /absolute/fresh-project/orze.yaml`。配置必须选择 `execution.resource: cpu`、`action_domain.kind: acceptance_sorting` 或 `acceptance_compression`，Domain config 只含 dataset；`action_policy.kind: acceptance`、config 为 `{}`。完整实测配置保存在归档报告的 cfg 字段，路径是当次私有项目，重放需使用新的独立路径，不复用已持久 Stop 的数据库。不应在现有 research 项目运行这个有限候选验收策略。

## 计量口径与限制

目标 70 项运行的实际诊断如下；数字不是速度回归门槛，也不是与旧系统的随机对照：

| 项目 | native 动作 / Policy 步数 | 已预留 wall 秒 | CLI 调用 wall 秒 | worker 主体 CPU 秒合计 | native elapsed wall 秒合计 | 首次看到 valid 秒 |
|---|---:|---:|---:|---:|---:|---:|
| sorting_default | 5 / 11 | 10 | 3.860514 | 0.000982 | 0.378688 | 0.772612 |
| sorting_counterfactual | 3 / 7 | 6 | 2.256074 | 0.000314 | 0.217555 | 0.737622 |
| compression_default | 5 / 11 | 10 | 3.834462 | 0.000579 | 0.368458 | 0.749075 |
| compression_counterfactual | 3 / 7 | 6 | 2.258680 | 0.000293 | 0.217240 | 0.753755 |

worker CPU/wall 从 worker main 内读输入之前，到组装输出 envelope 时采样；不含 Python 启动/import、最终 JSON 编码与写出。native elapsed 从发 GO 前开始，到 harvest 确认闭合后采样，包含观察闭合的轮询延迟，不含随后 Domain 解释/产物提交。CLI wall 是同解释器内本次调用计时，含隔离设置与真实 main，但不含新解释器启动及 helper 在起表前的 import。有效证据到达量是 Policy 第一次在实际 snapshot 看到相应 observation 的时间，不是精确数据库提交时刻。单调时钟只在同主机同次运行比较。

四个项目确实消耗 32 秒预留额度，不退款；10 秒上限不能逐项目全部当已消费。微任务执行主体很短，而控制路径、进程监督、封印/哈希、SQLite 事务和等待会占显著 wall 时间。上表提供分层量测，不能用 CLI 减 worker wall 计算纯框架 CPU 开销，更不能据此证明普遍 research 效率收益或节省百分比。

可信本地 Python 回调不是 OS 沙箱。此次成功路径保留已有失败/中断/篡改/HOLD 回归，但不代替恢复故障验收。留出内容仍未读；已知领域关闭后才打开原独立选择文件。原生修复/恢复、未知效果裁定和受监督 Director 正向交接仍按总账保留未完成，不能用此有限策略的成功宣称整个 V1 完成。
