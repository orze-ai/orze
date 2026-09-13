# 长期证据访问与存储准入：本轮验收

本轮完成两个具体缺口：自定义 CPU Policy 可有界访问第 32 条之后的当前代终态证据；依赖原子不覆盖重命名的角色／GC 在产生新的任务副作用前预检实际存储路由。没有加入研究总时限，没有改 ASR 项目，也没有切换生产服务。

源码候选固定为 Core `cce70af569e748f845c5b0286172f87cfbf9ffe4`、Pro `210fd1b1d7ef2fddd01f47579df050e54f3643b6`。后续证据归档／发布隔离只改文档，不改变这组执行代码。

## 实际完成的行为

- 显式 `action_policy.version: 2` 支持 ReadEvidence／SelectEvidence，页大小 1..32；旧 v1 契约不变。游标只属于本次进程、连接和读版本，不能复用为执行权限。读链不重复 ingress，不消费一次执行、不新增预算预约；正常决定后释放扫描。详见[接口说明](../cpu-evidence-paging.md)。
- 真实 CLI 从第 34 条取得实际来源，定向重读后经封印 FD 完成分析，再创建独立 replica。正控确实执行了来源、分析、复验三个 native 动作；前面的 33 条是明确标注的 metadata/effect 夹具，绝不是 33 个真实 worker。
- 伪造来源、改变来源字节、跨 invocation 复用旧游标均拒绝；回调期间另一连接实际修改 SQL 后，也不能记录永久 Stop。`--once` 可以先读完多页，再完成一次真正执行。完整 red／green、SQL、产物和闭合回执见[产品证据](../evidence/2026-09-13-evidence-paging-product-review.json)。
- 存储探针使用实际路由及已捕获的目录 FD，检查文件／目录移动和已有目标碰撞；不降级成可覆盖 rename。持久 Stop 优先于预检，已有 HOLD 不会因预检提前创建 archive。最终操作仍保留原 HOLD。详见[存储说明](../storage-preflight.md)。

## 冻结回归及保留的失败

| 验证 | 实际结果 |
| --- | --- |
| Core 完整回归，短 `/tmp` ext4 basetemp | 4,638 passed，7 skipped，2 warnings；935.25 秒 |
| Pro 完整回归，测试专用许可替身 | 1,010 passed；255.88 秒 |
| 原 Core／Pro 配对邻域 | 31 passed；1.06 秒 |
| 存储最终相关组合 | 191 passed；14.06 秒 |
| 独立分页／存储对抗用例 | 20 passed／10 passed |
| 新产品闭环／循环边界 | 4 passed／2 passed |

这些集合有重叠，不能相加作为独立样本数。Core 的 7 个跳过项是可选 Pro 路径；没有把它们计为通过。Core 完整运行覆盖 799 个源／测试／示例／构建文件的前后相同指纹；Pro 完整运行另覆盖 Core 800 项和 Pro 229 项。起始提交中 Core 的 495 个原有测试文件、Pro 的 137 个文件逐字节未变，见[原测试保留核验](../evidence/2026-09-13-original-tests-preserved.json)。

[Core 完整原始记录](../evidence/runs/2026-09-13-validation/core/run.json)、[Core 全量独审](../evidence/2026-09-13-core-full-independent-review.json)、[Pro 完整记录发布索引](../evidence/2026-09-13-pro-storage-full.json)、[配对独审](../evidence/2026-09-13-evidence-storage-paired.json)保留完整命令、输出、JUnit 与指纹或其私有原件引用。另有[分页独审](../evidence/2026-09-13-cpu-evidence-paging-review.json)、[存储独审](../evidence/2026-09-13-storage-preflight-independent-review.json)。

没有把新 API 不存在的 red 算成多个历史 bug。产品基线是正常 Stop、只执行来源任务而未完成后续分析的真实行为。存储候选的提前 archive 创建、相对路径探错盘、同名路径替代后先误移动再报错、Stop 前先创建管理目录，都保存了实际失败及对应候选源码；修复后保留业务断言。FD 透明测试探针的适配单独记录，未删减原断言。参见[作者证据发布索引](../evidence/2026-09-13-storage-preflight-author.json)。

## 固定制品与真实 CPU canary

新隔离候选根为 `/hot-data/fsx/workspace/erik/orze-paging-package-2026-09-13.ioUQcy08`，使用其中的 `core-only-venv`／`paired-venv`，没有覆盖旧环境。

- Core wheel SHA-256：`419ff0efc582e201cb9a7c1cbf8be5b3333fb2d8d05d16cce6c75e49af95076c`。
- Pro wheel SHA-256：`a9baee2fc30d4f38e6a72e94e736e344b6fd9e3472d8e1f9f47bf8eb1b4b0567`。

22 条实际构建、离线安装及检查命令通过；Git blob、wheel payload、安装内容与 RECORD 一致，23 个原依赖未升级。首次验证脚本因 Python 3.10 无 `tomllib` 失败的原记录也保留；这是验证工具兼容问题，不算产品执行失败。包版本号没有变化，必须按本轮提交和 wheel 哈希选择，不能混配旧同版本 Core。

新 Core-only wheel 在本机 CephFS 的全新私有目录完成一个真实 CPU 动作：产物为 `{"sum":55,"count":10}`，TREE_CLOSED／ECHILD_WALL、租约授权、effect 确认与 SETTLED 对应同一 fullRef；累计额度仍为 null，保留本动作 2 秒预约费用。外层故障保护计时器不是 Orze 研究总时限。它验证 CPU-only 没有被角色／GC 预检误禁，不是科学收敛或分页 Policy 的另一次留出实验。参见[制品发布索引](../evidence/2026-09-13-evidence-paging-package.json)和[独立制品核验](../evidence/2026-09-13-package-independent-review.json)。

## 发布边界与未关闭项

推送前实际确认 Core 仓公开、Pro 仓私有。完整 Pro 源码快照、含私有 README／构建元数据的包原始记录保存在私有 Pro 仓，原件字节和哈希不变；公开仓只保留对应发布索引和可核对引用。没有将本轮私有源码快照推到公开仓，也没有改写既有远端历史。

本机 CephFS 仍不支持所需的原子不覆盖重命名：这里完成的是提前拒绝，不是 CephFS 的角色／GC 兼容。探针目前不缓存，空闲但配置可运行的角色也可能产生固定预检 I/O。旧长路径 AF_UNIX 测试问题不因 ext4 回归通过而消失。

CPU proposal 历史仍有前 32 条窗口；Pro 自己的长程研究上下文、持久摘要，以及累计预算／首次 ingress 的历史扫描成本仍待处理。分页结束不等于科研收敛，当前代证据访问也不等于恢复任意历史代的权限。本轮不宣称新的研究吞吐百分比、全局恒定时间或所有 autoresearch 设计问题清零。生产服务切换、Pro 真实许可、模型账户和 GPU 验收没有执行。
