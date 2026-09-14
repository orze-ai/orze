# 预算与 ingress 性能片：验收记录

状态：本片实现、两仓完整回归、安装验证及最终独立核验均已通过。它不代表全部研究性能瓶颈已解决，也不是生产服务切换完成声明。

实施补充先提交推送为 `89b933e4703b3902a53211f0514cca3770c24a3f`。固定 Core 候选为 `df446ab35e745daf02cc7c4a19cadffd57518103`；配对 Pro 为 `879859cb383d7f113f58ba1da054c6785e4fce6a`，本片没有修改 Pro 源码、测试或构建声明。目标仍是通用 autoresearch 的协调成本与完整性，不是 ASR 专项优化。

## 实际改动与收益

- 预算两个读取入口共用纯行校验器。汇总一次 SELECT、流式逐行审计，消除每行按 permit ID 再查的 N+1；精确整数累计、原状态／引用检查及原七列返回值保留。
- 解析后为空的 ingress 批次不读历史 ID 和配置缓存；非空批次只查询本批最多 128 个 ID。源锁、fresh read、sidecar、游标、实际 insert/exact/conflict、跨 ID 去重与 ACK 次序不变。旧 `get_all_ids` 语义不变，轻量适配器只按本批逐项 `get`。
- 新扫描同时核对实际 SQL ID，修复旧扫描可能审错记录的问题。正常 SQL CHECK／三个索引下，A=2 秒、B=3 秒的合成历史原本共 5 秒；仅将 A 的 permit 改指向 B，旧公开 `snapshot` 接受并误报 6 秒，新版明确 `cpu_budget_permit_changed`。见[公开接口探针](../evidence/runs/2026-09-14-budget-public-alias/summary.json)。它是元数据损坏验证，不是实际执行、结算或公开写入 API 可产生坏记录的证明。

同机 `/tmp` 合成历史，每个当前 scope 另有等量其他 scope；预热后新旧交替各五次，表内为中位数。SQL trace 与解析调用计数另跑，不混入计时区间。

| 当前 scope 行数 | 汇总 SQL：旧 → 新 | 汇总毫秒：旧 → 新 | 空 ingress 毫秒：旧 → 新 |
| --- | --- | --- | --- |
| 100 | 101 → 1 | 19.81 → 18.35 | 15.82 → 15.64 |
| 1,000 | 1,001 → 1 | 196.08 → 183.16 | 16.61 → 15.64 |
| 5,000 | 5,001 → 1 | 999.41 → 931.20 | 20.18 → 15.65 |

这不是数量级提速：本组 SETTLED 合成行每行仍有五次 JSON decode，5,000 行仍为 25,000 次；预算完整审计仍为 O(N)。没有新增索引、汇总表、额度缓存、退款或研究总时限。非空 ingress 仍加载 legacy 配置缓存，sidecar 和源文件处理仍有总量成本。以上不是生产共享文件系统吞吐、有效研究证据延迟或科学研究收益百分比。见[独立差分与全部五次样本](../evidence/2026-09-14-cost-equivalence-review.json)。

## 验证及原件

| 集合 | 实际结果 |
| --- | --- |
| 预算作者定向，含既有 native／Stop／恢复邻域 | 141 passed，82.87 秒 |
| ingress 作者定向，含既有并发／准入邻域 | 103 passed，15.52 秒 |
| 整个旧预算模块与新版的独立差分 | 23 passed，5.12 秒；其中一项明确为 alias 拒绝加固 |
| 独立既有 native／恢复邻域 | 12 passed，16.93 秒 |
| Core 全量 | 4,733 passed，7 可选 Pro skipped，2 既有 warnings，954.67 秒 |
| Pro 全量，测试期许可门替身 | 1,037 passed，268.57 秒 |
| Core/Pro 可选配对补测，测试期许可门替身 | 31 passed，1.00 秒 |
| 新安装 Core 上的真实产品及本片回归 | 35 passed，26.42 秒 |

这些集合重叠，不能相加。定向原件见[预算作者报告](../evidence/2026-09-14-budget-scan-author.json)、[ingress 作者报告](../evidence/2026-09-14-ingress-cost-author.json)、[独立报告](../evidence/2026-09-14-cost-equivalence-review.json)。[Core 全量原始日志](../evidence/runs/2026-09-14-cost-validation/core-full/stdout.log)中的两项 warning 均为既有辅助模块的 pytest assertion rewrite 提示；可选 Pro 路径另经上述配对补测。冻结输入覆盖 Core 808 个、Pro 231 个源码／测试／示例、指定构建文件及本片差分测试所需旧模块；不声称整台机器或所有文档文件都未变化。

Core 本片起点的 505 个旧测试文件中，504 个逐字节未变；唯一文件严格等于原件加四处注入点替换，其全部 26 个原断言 AST 不变，真实第二连接竞争写入仍执行。Pro 的 140 个旧测试文件全部逐字节未变。[Core 逐文件审计](../evidence/runs/2026-09-14-cost-validation/core-original-tests/stdout.log)没有将例外文件跳过或声称全部文件不变。

[最终独立核验](../evidence/2026-09-14-cost-final-independent-review.json)重新核对原始 JUnit／footer／日志哈希、1,039 个输入与固定 Git、全部制品原件、111 个加载模块、四个产品项目原 SQLite，以及实际执行的 TREE/ECHILD、effect、产物和 SETTLED 记录。报告 SHA256 为 `64027e20c533d9631e339df534cd5d48fe17af5d476b9a53a7001cef30744f88`。

明确保留了旧版预算成本断言 2 failed/6 passed、旧 ingress 成本断言 7 failed/8 passed，以及新源码移除旧查询后原并发注入点失效的 1 failed/14 passed。后者是夹具迁移，不是产品行为回退。静态替换计数、独立回归 wrapper 路径和制品归档抄录的问题也分别留在对应报告中。最终检查器曾误设 installed site 必须位于路径首位、无 attempt 的项目必须已有 attempt 表；原始检查器和错误均保留，按真实 pytest 路径及原数据库投影契约纠正，未改产品或测试。不伪称产品红灯，不用后来的成功覆盖原始记录。

## 制品与运行边界

从固定提交重新离线构建 Core/Pro，23 个外部依赖 wheel 字节不变；两个新 venv 完成 22 条构建／安装命令、`pip check`、实际 Core help 隔离检查，以及独立 Git archive → wheel → 安装 RECORD 复核。

Core wheel SHA256：`1070fd9417ece854e368d01a96dccd3c0723bd74bf8d0d1b6058cadc999dca74`。

新 Core 安装上的 35 项回归使用外部 pytest harness，不声称纯 venv 解释器全量测试。显式排除仅负责插入工作区 `src` 的根 conftest，原 `tests/conftest` 保留；111 个实际加载的 Orze 模块必须逐一匹配新安装路径、wheel 与固定 Git。四个既有产品案例共一次真实 CPU worker，具备 TREE/ECHILD 闭合、产物和 SETTLED 记录；140 个 proposal 协调结果及合成历史不冒充 worker。原始报告与 155 个项目文件归档在[安装验证索引](../evidence/runs/2026-09-14-cost-validation/core-installed-payload/index.json)。

未改用户 ASR 工作区、全局安装或现有服务；未调用真实 Pro 许可、付费 provider 或 GPU；未合入 main、发布包或切换生产服务。安装可用性不等于生产部署完成，既有目标文件系统／服务切换边界不因本片消失。Pro 原始清单、日志及完整制品仅留私有 Pro 仓，见[固定私有提交的结果／哈希索引](../evidence/2026-09-14-cost-pro-private-index.json)；其七份入口原件均另从 Git 提交读出核对 SHA256。

## 剩余性能工作

测量已经表明单纯减少 SQL 数量不足以解决长历史延迟。下一步应分别量化重复 JSON 规范化和 policy 只读续页中的重复预算全审计，再选择最小改动。静态代码仍在 admission 和 policy snapshot 两处读取预算；这只是待验证的成本来源，不是已测端到端收益。

若评估同版本只读续页复用，必须同时证明 SQL 修订、同进程写入／回滚、连接／路径替换、Stop/HOLD（包括不落 SQL 的进程内 HOLD）均会失效；正常准入与 GO 仍独立重新核实。不能仅凭 `data_version` 或旧缓存直接授予执行权。本片没有加入这种缓存。
