# 固定研究对照的执行与采集

`examples.research_comparison` 可以执行明确指定的调度任务对照，并重新核验每个计划样本。
它补齐的是运行与采集入口。当前公开调度任务、离线 HTTP 替身和 A/A 测试不能证明
真实模型研究能力提升，也不是新的未见任务。

## 先固定输入

在包含 Core、Pro 和示例代码的 Python 环境运行：

```bash
python3 -m examples.research_comparison describe-runtime > runtime.json
```

此命令不调用模型。描述包含 Core/Pro 包的内容身份、实际导入路径、Python/依赖版本、
已加载启动模块的内容身份，以及公共采集器与核验器的代码身份。
两个 arm 可以使用不同的兼容 Core/Pro 代码；共同采集器／核验器必须固定。
子进程会在研究前后重新检查实际环境，不能用调用方填写的版本代替实际导入版本。
这不是密闭沙箱：它不穷尽操作系统状态、硬件性能或所有环境变量。

完整 specification 为 JSON 对象，字段恰好如下：

| 字段 | 内容 |
| --- | --- |
| `schema` | `1` |
| `protocol` | [现有对照协议](../examples/research_comparison/README.md#protocol-fields)，含目标任务、负面对照、AB/BA 顺序、重复次数和预算 |
| `shared` | `model`、`tools`、`environment` 的原始值，分别匹配协议中的哈希 |
| `tasks` | 以完整任务 ID 为键；每个值包含 `data`、`evaluator`、`instructions`、`initial_history`、`initial_memory` |
| `arms` | `A`、`B`，各包含对应的 `runtime` 描述和 `treatment` 配置 |

当前执行器仅支持：

- 新的 prospective 调度任务，最多 256 个计划样本，每个样本从空历史 `[]` 开始。
  `initial_memory: null` 不创建记忆；`initial_memory: {schema: 1, entries: []}` 则显式发布
  一个作用域绑定到本项目的空文档。开启 `research_memory` 时必须选择后者；A/B 共同输入
  采用相同初始化声明，关闭记忆的 arm 也会得到空存储，避免给另一边额外的初始结论。
  当前不接受预填记忆或导入历史。
- `model` 恰好包含 `backend`、`model`、`endpoint`，作为实际 Pro CLI 参数传入；需要正常 Pro 许可。
  后端的采样参数仍由固定的 Pro 传输实现决定。CLI 模型名称不证明远端模型修订；
  真正等条件实验还需固定供应商模型修订、传输设置并核对实际请求。
- `tools` 恰好包含 `workload: "scheduling-v1"`、`rounds`（1–16）、`num_ideas`（1–16）和
  `evaluation_protocol`（公共调度评价器支持的协议）。
- `data` 使用公共调度实例格式；`evaluator` 包含评价器文件的 `source_sha256` 和 `protocol`。
  `instructions` 明确提供实例和模型任务要求。种子只控制候选的评价顺序，不控制远端模型采样。
- `treatment` 可以添加研究配置，不能覆盖共同的实例、资源、队列、报告或项目路径设置。
- 正的 CPU 预留限额和 0–86,400 秒之间的正外层超时；所有其他 prospective 预算也必须明确。

协议及输入哈希固定字节；它们不证明登记时间，也不授予模型、账户或资源使用权限。
真实调用前需另行明确任务、账户和总费用／调用额度。执行器强制自有进程树的外层时限，
并使用原生 CPU 预算和 token envelope；它不提供供应商总账单硬上限。账户侧费用控制仍需独立设置。

## 执行和只读核验

先固定 specification 文件的 SHA-256，再运行显式执行命令：

```bash
python3 -m examples.research_comparison execute-campaign \
  --specification /absolute/specification.json \
  --specification-sha256 FILE_SHA256 \
  --output-dir /absolute/new-campaign
```

程序先核验全部 task/arm 输入，再创建新目录和固定 specification，最后按计划逐个执行。
输出目录存在时拒绝运行；中断目录保留，不自动续跑或重试。输出中的
`specification_sha256` 对应保存后的文件，供下一步使用。

每个样本按固定轮数调用 Pro，执行获准候选、独立评价并重新评价最终选择。
研究子进程先注册与 CPU 执行相同的调度 Domain，再进入正常 Pro research CLI；这使证据
分页中的执行预算检查能识别该任务。显式空记忆在项目初始化后、第一轮 research 前发布，
初始状态与各调用前后的记忆表随原有数据库快照保存。核验检查空文档、项目作用域、版本 1
及其发生位置；这些存储快照本身不证明模型理解或使用了条目。初始化开销包含在外层时间内。
每次评价后重新核验原生账本、观察和候选字节；选择策略使用核验后的有效价值，
并保存这次决策输入的引用和读取时刻。模型自报分数不参与选择。
正常失败可以保留已关闭的原生动作与成本；无法确认进程树关闭时停止后续样本。
外层 wall time 从创建样本目录后的请求写入前开始，到子进程树排空后结束，
包含启动、模型、评价、采集与关闭；不包含前置协议检查和最终跨样本报告生成。

```bash
python3 -m examples.research_comparison audit-campaign \
  --campaign-dir /absolute/new-campaign \
  --specification-sha256 SAVED_FILE_SHA256 \
  --output /absolute/new-report.json
```

核验会重读固定原件、复算候选质量与原生结算，校验实际命令、研究轮数、种子顺序和选择。
用量来自独立的 prompt 请求清单及用量日志；失败重试和缺失值继续可见。
这是受信任采集器产生的证据核验，不是对伪造全部记录的第三方提供密码学认证。
输出是一次性新文件；不会调用模型或接管历史进程。

## 解释报告

所有计划样本都保留。没有索引的样本为 missing；已有索引但原件缺失或核验失败为 unknown，
对应成本也未知。没有完整结果的超时样本不假定零成本。完整关闭且原生账本确实为空的
失败样本可以证明零原生动作；不能据此假定供应商零调用或零收费。

`confirmed_selection_seconds` 从外层请求写入前计时，终点为独立评价复验及核验确认有效
选择之后；包含启动、模型和 CPU 工作及本次确认核验。回执绑定选择／复验 attempt，
所有内层步骤必须落在 worker 及外层时钟区间内。无效、失败、回执或外层时钟缺失时为 `null`。
它不代表模型首次理解证据或自主决定收敛的时刻。细节见[确认延迟验证](plans/2026-09-16-campaign-timing-results.zh-CN.md)。

`first_valid_consumed_seconds` 使用原有 CPU 对照的操作定义：工作流决策代码首次读取
合格有效观察的时刻，减去外层采集起点。每份回执绑定原生评价 attempt、候选哈希、协议、
verdict 和评价后的账本快照；读取必须在该评价关闭之后、下一步启动之前。完整清单中的
首次有效输入才给出数值。全部无效、清单不完整或外层时钟缺失时为 `null`；后续调用失败
可以保留先前已核验的首次读取时间。此指标不证明 Pro 提案模型已理解或引用这份证据。
细节与新增核验成本见[决策输入计时](plans/2026-09-16-campaign-consumption-results.zh-CN.md)。

目前供应商美元费用和 GPU 用量仍为 `null`。
因此必须核验这些预算的 prospective 配对不会合格，也不会产生提速中位数。
已知数值只代表其列的可核实部分。`new_research_evidence: false` 表示报告重算本身不创造实验，
测试通过也不构成真实研究收益证明。

原始结果包含绝对路径，需保留运行目录。跨机器重定位、生产部署、旧版本兼容性和
真实研究任务的公平实验协议需要另外验收。
