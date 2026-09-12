# 研究效率对照与生产部署：执行协议 v1

日期：2026-09-12。新增用户授权：实际推进研究提速和生产部署。旧 V1 机制验收保持其固定版本与边界，不用本次实验倒改历史结论。

## 生产准备与边界

部署制品固定 Core `82012aeb81cf6b6b137b80f27319a53d5ddeb8db`（源码 `f62495d`）、Pro `da2d7d92da5281ab49362509fad71fc3d78e427a`。从 Git archive 构建 wheel，安装到全新 venv，记录 wheel/源码/依赖哈希，运行真实安装入口和私有 CPU 项目；不使用 pytest 的 monkeypatch 或授权替身完成部署预检。

默认用户安装实际为 Core 4.4.3、Pro 占位 0.0.1。当前 ASR 工作区有大量未提交修改，不将其推断为本次生产目标，不覆盖安装或重启。最终切换须明确机器/服务/项目、Pro 正式授权及模型/GPU预算。新的包预检、CPU 对照和 canary 是部署准备，不等于既有服务已升级。旧进程可能的 writer 必须按原 ownership/guard 合同排查；不按 PID 消失或服务 inactive 自动释放预算或删除 guard。

## 可检验的效率假设

当前应用策略 CommonPolicy 在已经观测到 unchecked 候选成本严格高于 valid incumbent 时，仍会提出来源绑定的 validation-only analysis。现有 sorting/compression 协议的该分析只验证并重发原成本，不降低成本。

新增应用层 DominancePruningPolicy，Core、Domain、worker、数据与判据均不修改。它只在声明允许的精确协议、同 comparison scope / dataset、unknown 状态及成本严格劣于 incumbent 时省掉此分析，仍使用原 Replicate 与合法 Stop。任何条件不满足均沿用原策略行为，不把 unknown 改写为 valid，不合成已执行的 analysis。

这是一项通用决策规则在两个已知协议上的有限验证，不按领域名/WER猜资格，也不是所有 unknown 结果都可被剪枝。显式代价：少取得一个无助于当前选择的候选验证结果，因此目标不是完整鉴定全部候选。

## 正式对照：运行前冻结

- 基线：原 CommonPolicy；处理：新 DominancePruningPolicy。两臂使用同一个已安装的固定 Core wheel，不与缺少 CPU 能力的旧 main 比速度。
- 任务：原 sorting / compression 各自 DEFAULT_DATASET 与 COUNTERFACTUAL_DATASET，共四个公开任务。不称新独立留出，不新增或修改数据以追求正结果。
- 每任务 6 个配对，共 24 对 / 48 个独立新解释器 CLI 项目。每对共享原数据/协议；第 0、2、4 对 AB，第 1、3、5 对 BA。顺序记录在运行清单；按任务交错执行，避免某个任务全部落在单一时间段。
- 两臂均单 CPU 槽、10 秒累计保守额度、2 秒每个 action。无真实 GPU、付费 provider、Pro 授权替身或用户项目。每个 CLI 外层最多 30 秒，整轮最多 15 分钟；超时/失败/HOLD 都进入原始结果，不能挑最快或只分析成功组。
- 运行之前提交本协议和完整实验脚本/测试；执行清单记录脚本提交、安装 wheel SHA、运行源码/worker文件 SHA、依赖及环境。任何正式开跑后的修改或重跑作为新轮次记录，原失败保留。
- 先验证质量：每臂必须真实合法结束，选择有效且具新 attempt 复验，原测量与复验可比；同一配对的选中动作签名/候选、成本与协议一致。不得用更差结果换速度。任何质量失败使“等质提效”结论不成立。
- 主终点：质量门满足时，实际 native action 数与已结算预约的保守额度之差。预期默认两组 5→4、反事实两组 3→3只是预注册假设，必须以真实账本核对。
- 次终点：完整 CLI wall、首次被 policy 消费的 valid 观察时间、确认选择决策时间、实际 worker 主体 CPU/wall。保留每次原始值，逐任务给中位数和配对差；不把 CLI wall 减 worker wall 叫作纯框架 CPU。
- 同时报告 failed / interrupted / invalid / unknown / analysis 数以及少取得的有效观察。原 trace 必须记录真实 snapshot，不删除原未知结果、虚造analysis或只保留处理臂有利证据。
- 不把 48 个项目、进程/观察数宣称为独立科研发现或统计显著性。只有实测 wall 支持时才报告该有限任务集的延迟改善；动作/预约节省本身不是模型推理加速或普遍研究收益。

## 下一阶段真实研究与线上切换

正式 provider 的质量/成本对照另需冻结账户/模型、同一候选任务与评价协议、提示/工具权限、调用/token/金额/时长上限以及重试与配额成本口径。GPU 实验另须明确物理设备、数据和运行窗口。不得用本次 CPU 结果填补这些观测。

生产发布不自动推送 PyPI、合并 main 或运行用户现有升级脚本；选择目标后采用精确制品、独立 venv、预检与受控 canary，确认旧 owner 闭合、备份及兼容策略，再切换服务并实际验证重启和回滚。真实授权失败必须如实保留，不改生产 gate。
