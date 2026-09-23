# 机器收回后的研究交接：结果、代码与恢复入口

更新：2026-09-23。用户已确认实验机器被收回，并要求整理成果、推送 remote main。本轮研究停止于资源交接；不要继续轮询旧机器或自动恢复已结束实验。

**结论：已有指定任务上的执行效率提升，尚未证明平台研究效率再提高 30%，也未证明通用研究能力提高。** 当前探索默认保持 ParallelRefine；已验证的并行准备执行方式保留。inquiry、混合研究模型和新原生对照都没有成为新的产品默认。

## 1. 从这里读起

- 本文：最终状态、值得保留的代码和下一机器的恢复顺序。
- [30% 目标、十一篇 RSI 论文及完整证据链](2026-09-22-rsi-efficiency-30.zh-CN.md)：目标不缩小，论文报告的 token/费用收益不能替代本地最终验收时间。
- [归档索引与离线校验](../evidence/2026-09-23-machine-retirement/README.zh-CN.md)：Core 保存公开结果，私有 Pro 保存完整方法/调用记录和实验脚本。已有文件原样保留，最新入口以本文为准。

## 2. 实验结果总表

时间均为每次研究实例启动到独立最终验收的墙钟时间，含模型等待、失败和评估；未达标按 120 分钟计入。不能只平均成功者，也不能把开发提升或基线预检算作验收成功。

| 比较 | 完成及验收 | 平均达标分钟 | 保留的结论 |
|---|---|---|---|
| [ASR：ParallelRefine / 组合探索](../evidence/real-gpu-research-20260921/README.md) | 16/16；两组各 5/8 | 60.38 / 73.63 | 组合方案更慢，保留 ParallelRefine |
| [ASR：原执行 / 改进执行](../evidence/real-gpu-efficiency-20260922/README.md) | 16/16；两组各 5/8 | 71.44 / 54.98 | 下降约 23%，按该轮预注册规则采用 prepared 执行；四个已知 ASR 语料上的结论，区间仍含零 |
| [跨领域：control / inquiry](../evidence/runs/2026-09-22-cross-domain-research/RESULTS.zh-CN.md) | 32/32；9/16 / 12/16 | 58.93 / 38.82 | 描述性下降 34.13%，仅 2/4 领域更快、区间含退步；不升级默认 |
| [新任务：Opus 5 / Opus 5.5](../evidence/runs/2026-09-23-rsi-prospective/RESULTS.zh-CN.md) | 32/32；11/16 / 13/16 | 54.62 / 36.52 | 描述性下降 33.15%，仅 2/4 领域更快、区间含退步；不升级默认 |
| [已观察任务：单模型 / 混合模型](../evidence/runs/2026-09-23-rsi-mixed-model/RESULTS.zh-CN.md) | 16/16；6/8 / 7/8 | 54.31 / 31.02 | 开发下降 42.88%，晋级候选；尚非独立迁移或原生入口效果 |
| [实际原生 Sonnet / 混合角色](../evidence/runs/2026-09-23-rsi-native-drain-review/RESULTS.zh-CN.md) | 计划 32，启动 10，完整结束 2，验收 0 | 不作均值判断 | 验收适配器竞态导致中止；修复已用固定样例检查，原付费试验退休 |

跨领域研究包括六个公开真实数据集和两个生成优化问题类；生成实例不是采集数据。四领域是四个不确定性单位，不能把 32 次研究视作 32 个独立领域。

### 必须连同成绩保留的反例

- inquiry 的 sentiment-1 一条高分方法依赖标签排序，打乱行序后不再达标；原统计不删行，不能把该条通过作为通用能力证据。[独立诊断](../evidence/runs/2026-09-22-research-validity/verification.json)
- 旧 TSP 两个开发集的 2% 门槛被数学下界排除；最终审计门槛仍不确定，不能把全部优化失败都归为不可能。[可达性分析](../evidence/runs/2026-09-22-rsi-feasibility/RESULTS.zh-CN.md)
- 首批比较后再确认只改善 6.72%；源码评审存在截断、换序不稳定和选中慢方法等问题，没有足够证据增加评审层。[确认时机](../evidence/runs/2026-09-23-rsi-confirmation-choice/RESULTS.zh-CN.md)、[源码评审](../evidence/runs/2026-09-23-rsi-source-order-diagnostic/RESULTS.zh-CN.md)
- 更快返回的模型可能提出更长的训练；混合方案的视觉任务明显变慢。代码有效、质量更高、模型等待更短，均不自动等于研究达标更快。

## 3. 有用代码的位置和状态

| 代码 | 用途及边界 |
|---|---|
| [src/orze/research/execution.py](../../src/orze/research/execution.py) | 并行准备、就绪方法执行及独立交付；省略 workers 时默认为 2。 |
| [src/orze/research/exploration.py](../../src/orze/research/exploration.py) | 分支局部历史与在线/回放共同循环；回放不能产生未知结果。 |
| [src/orze/research/exploration_policies.py](../../src/orze/research/exploration_policies.py) | ParallelRefine、组合及候选探索策略；现役默认未因本轮归档改变。 |
| [src/orze/research/data_splits.py](../../src/orze/research/data_splits.py) | 分组划分及避免保留标签排序的修复。 |
| [src/orze/reporting/completed_scan.py](../../src/orze/reporting/completed_scan.py) | 其他连接提交后重新读取完成证据，避免长期陈旧快照。 |
| [src/orze/core/gpu_lease.py](../../src/orze/core/gpu_lease.py) | 按实际物理设备解析 Slurm GPU，保留原生所有权与外部进程检查。 |

另一仓的实现和归档见[配套交接](https://github.com/orze-ai/orze-pro/blob/main/docs/plans/2026-09-23-machine-handoff.zh-CN.md)。


实验适配器的最新恢复起点在 **Pro** 的 [共同原生环境归档](https://github.com/orze-ai/orze-pro/tree/main/docs/evidence/runs/2026-09-23-rsi-native-shared-environment)。其中 `rsi-native-pyvrp-runtime-20260923/` 保存修正后的 `remote.py`、`drain.py`、`goal.py`、执行器与独立预测 worker；`rsi-stl10-reference-preflight-20260923/` 保存尚未启动的图像预检。不要从更早的中止试验目录取适配器覆盖它。

- [原生验收竞态与修复检查](../evidence/runs/2026-09-23-rsi-native-drain-review/RESULTS.zh-CN.md)：必须等 accepted IDs 全部入库并终态，再审计；不能以请求回执数量代替训练完成。
- [独立替换任务及准备代码](../evidence/runs/2026-09-23-rsi-native-replacements/PREPARED.zh-CN.md)：STL-10、CLINC150、Gas Turbine NOX、生成 CVRP 的下载、划分和评分代码在 Pro 对应归档。
- [CVRP 校准及共同参考](../evidence/runs/2026-09-23-rsi-cvrp-calibration/RESULTS.zh-CN.md)：PyVRP 0.14 固定 1000 次迭代的人工可达见证改善 3.8537%；不是研究模型产出或 30% 效率收益。两次失败校准也已保留。

### 原始 1.7B 项目的未提交工作

另已将原 `auto-research-1.7b` 项目 **255 个既有修改/新增源码文件**、权限和删除清单保存为[私有 Pro 增量快照](https://github.com/orze-ai/orze-pro/tree/main/docs/evidence/runs/2026-09-23-project-worktree-preservation)，基准 commit、逐文件哈希及恢复说明均在包内。该项目修改未作为产品代码合入，也未重新验证科学效果；保全不等于验收。日志、凭证、数据和权重不在增量包中。Core/Pro 的五个旧研究工作树均干净且其提交已经合入 main。

## 4. 机器收回时准确停在哪里

[14/16 参考清单](../evidence/runs/2026-09-23-rsi-cvrp-calibration/prepared-reference-inventory.json)：SVHN、DBpedia14、Online News Popularity、生成 regular MaxCut、CLINC150、Gas Turbine NOX、生成 CVRP 各两实例已准备；STL-10 两实例仍缺 GPU 参考训练与独立重载。未把 14 个就绪实例替代原定 16 个实例。

[共同环境检查](../evidence/runs/2026-09-23-rsi-native-shared-environment/RESULTS.zh-CN.md)：新增依赖只读挂载、1113 文件哈希及实际 Enroot CPU 隔离已验证；**该新增环境的 GPU 执行尚未验证**。STL-10 预检计划为 4 次训练、6 次独立预测，代码已准备，执行数为 0；新 32 次研究尚未冻结或启动。

机器收回前，本机 GPU 有常驻服务和剩余显存，但现有原生执行器要求没有其他计算进程；Slurm 也没有满足分配条件的资源。这是执行条件不满足，不能描述为所有 GPU 都在满负荷计算。用户现已确认机器被收回，无需继续重复资源探测。

## 5. 费用与进程结算

以[中止原生试验结算](../evidence/runs/2026-09-23-rsi-native-drain-review/budget-release.json)为最新累计账本：授权上限 **$5000**，保守占用 **$4659.467075**，未分配 **$340.532925**；26 条未知用量记录仍全额覆盖。这是预算上界，不是实际账单。

原生中止试验 33 个提供方进程已结束，32 次用量已知、1 次未知；45 个已解析付费方法中 17 个未执行。31 个 worker dispatch 有 26 个本地闭合回执，另 5 个由 Slurm 终态证明关闭，不能补造本地回执。旧 $320 池已退休，不得重复释放或恢复。最近的参考、校准、环境准备及本次归档没有新增模型请求。

历史定期监控已终态结束。没有本任务仍在等待的 GPU 作业；本文不是继续运行旧服务、监控或研究的启动指令。

## 6. 下一台机器如何恢复

1. 取得 Core 与私有 Pro 的最新 `main`，先运行各仓的离线归档校验。所有新工作放在新的 `/work/` 目录；旧根路径仅是来源标识。
2. 确认新机器的设备所有权、实际 GPU 型号/内存和隔离能力。旧 Enroot 根目录、Docker 镜像、本机地址、Slurm 作业号及日志中的 PID 都不可直接继承。
3. 从 Pro 的最新共同环境、替换任务和 CVRP 归档取代码；检查硬编码路径、镜像及依赖。按固定版本/来源 SHA256 重建依赖，双方使用同一环境。凭证单独配置，不从实验归档恢复。
4. Git 已保存源码、结果、协议、用量/闭合记录及哈希，**不含完整原始数据、特征数组、训练检查点、私有划分种子或运行环境二进制**。完整逐次重放还需要旧共享存储中的对应文件；文件缺失时需重建并重新校验。无法匹配原哈希的重建必须作为新准备记录，不能声称恢复了原冻结实例。
5. 先完成缺少的 STL-10 参考与共同原生 GPU 检查。当前预检脚本申请 4 张 GPU，正式设计使用 8 张；这是本次脚本/设计配置。新硬件或运行条件发生变化，须在双方任何研究请求前共同确定，不能沿用不匹配的墙钟结论。
6. 复核真实对照入口：实际项目 control 为 custom/sonnet、每次 3 提案/最多 2 次角色尝试；候选为 Opus 5 与 5.5 两角色、每次各 1 提案/最多各 3 次。双方角色 timeout=600 秒、stall=20 分钟、poll=30 秒、cooldown=900 秒。[已解析配置](../evidence/runs/2026-09-23-rsi-native-shared-environment/configuration-verification.json)
7. 16 实例参考与资源就绪后，重新冻结新 32 次研究及预算。EMNIST、Banking77、YearPredictionMSD、jobshop 已被原生研究观察；更早模型/策略试验中的任务也不能再次称为全新验证。不要续跑退休的付费目录。
8. 仍需同时达到：平均最终验收时间下降至少 30%、每领域成功数不下降、至少 3/4 领域更快、四领域配对 95% t 区间上界小于零、费用与进程完整核验、独立科学有效性审阅，以及胜出实现进入真实默认入口。未满足就保留现有默认。

## 7. 对研究效率最有用的教训

- 优先衡量到可靠答案的完整时间；更高分、更短模型延迟、更多提案都只是中间指标。
- 检查模型是否根据实际测量修改方法、检验竞争解释；研究式措辞和事后归因不能代替执行证据。
- 先验证输入隔离、行序随机化和目标改善空间，再花钱研究；可达性见证与候选模型的能力成绩分开。
- 比较真实现役入口。独立试验脚本中的赢法，必须再验证能否改善用户实际运行路径。
- 用现有回执、源码与数据库完成排空判断，保留失败和未使用提案；避免为局部问题叠加新的模型评审或调度服务。
- 机器和镜像准备只做支撑研究所需的最小工作；准备齐全不等于研究目标完成。恢复时从缺失的 GPU 预检接续，避免重复已验证的 CPU 准备。
