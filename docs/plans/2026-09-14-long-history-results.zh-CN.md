# 长程研究信息访问：本轮结果

本轮修复的是通用 autoresearch 的信息访问，不是 ASR 调参：显式启用的 CPU proposal 历史分页，以及 Pro 追加笔记的有界首尾读取。没有研究总时限，没有改 ASR 项目，没有合并 main 或切换生产服务。

实施合同已先推送为 `46ab33233be8a31a37e5ae991358ae08305c9d3e`。固定源码／制品候选为 Core `26643ed7d6fa42e1d72939c61eb6fc8d249ffa66`、Pro `1719502ad5e875f1d4e30e3dec04023fb1757418`；后续归档提交只添加文档与证据。

## 已验证的行为

- `action_policy.version: 2` 可显式增加 `proposal_page_size`，通过 `ReadProposals`／`SelectProposals` 读取后续回执。同循环、共享读版本、独立单次游标；读链不重复 ingress、不记执行、不预约预算、不消费 `--once`。旧 v1／未启用的新字段行为不变。详见[接口说明](../cpu-proposal-paging.md)。
- 旧实际 CLI 在 35 次真实接纳／重放／冲突回执下看不到后页，正常 Stop，却未完成目标动作。修复后原四测试保持全部 47 个断言：4 次翻页、定向选择后执行一次真实 CPU 动作，产物、TREE_CLOSED／ECHILD_WALL、effect 与 SETTLED 相互对应；后页冲突也能驱动 Stop 而不启动 worker／扣执行额度。这 35 次回执不是 35 次实验。见[独立产品证据](../evidence/2026-09-14-proposal-paging-product-review.json)。
- 新增历史定向读取本身的重复查询也已消除：32 个请求 ID 的实际记录 SELECT 从 32 次降为 1 次，仍逐行完整解码、核对数据库与 scope。它不等于全研究循环提速。65 条历史分为 32／32／1；旧 `records_view`／`recorded_proposals` 函数保持原样。见[作者证据](../evidence/2026-09-14-cpu-proposal-paging-author.json)。
- 切到 proposal 通道时，当前展示的 evidence 被重新定向核验而不消耗其扫描游标。坏 effect 变为 unavailable；改坏真实产物字节后，元数据读取不冒充内容 hash 校验，实际 Propose 仍被原来源准入拒绝。正控和两个负控均无额外 worker／预约。该[独立验收 probe](../evidence/2026-09-14-proposal-channel-review.json)不加入默认测试集合。
- Pro 的 `retrospection_window=head_tail` 经角色参数、CLI 和实际 research cycle 到达 prompt 装配。最多首／尾各 8 KiB，同 FD 身份检查；尾块不足空间则整体省略。短文件只一个块。实际 manifest 落盘先于测试 provider，区间、hash 与真实 prompt 字节一致，仍是 schema 1、未验证笔记而不是实验资格。默认 `prefix` 和 knowledge 目录行为不变。私有[用户说明](https://github.com/orze-ai/orze-pro/blob/feat/research-production-validation/docs/research-authored-notes.md)与[证据发布索引](../evidence/2026-09-14-pro-private-index.json)。

## 冻结回归

Core 最终全量实际完成：4,680 passed、7 skipped、2 warnings，1,001.40 秒，前后文件指纹一致。7 个跳过项是可选 Pro 路径，不计为通过；对应配对邻域另跑。完整命令、原始 stdout／stderr、JUnit 和 803 文件指纹见[Core 冻结运行记录](../evidence/runs/2026-09-14-validation/core-full-canonical/run.json)；另有[最终独立核验](../evidence/2026-09-14-long-history-final-independent-review.json)，重新核对原始文件 hash、完整 JUnit、两仓指纹、固定 Git blob 及旧测试字节。

| 验证 | 已取得结果 |
| --- | --- |
| Core 完整回归 | 4,680 passed，7 skipped，2 warnings；1,001.40 秒 |
| Core proposal／evidence 作者邻域 | 147 passed，66.69 秒 |
| Core 独立真实 proposal 产品用例 | 4 passed，7.30 秒 |
| 跨通道独立验收 probe | 3 passed，19.06 秒 |
| Pro 首尾读取及邻域 | 109 passed，18.26 秒 |
| Pro 最终配对完整回归 | 1,037 passed，272.40 秒；两仓指纹相同 |
| Core 可选 Pro 邻域 | 31 passed，1.05 秒；两仓指纹相同 |

这些集合重叠，不能相加。本轮起点中 Core 502 个旧测试文件、Pro 138 个旧测试文件全部逐字节未变；[Core 原测试字节审计](../evidence/runs/2026-09-14-validation/core-original-tests/stdout.log)与[Pro 原件索引](../evidence/2026-09-14-pro-private-index.json)保留逐文件依据。最终配对记录覆盖 Core 803 个、Pro 231 个 `src/tests/examples` 及指定构建文件；不声称整台主机未发生任何其他变化。

保留了以下未通过／非验收 epoch：默认 Core pytest 误收证据目录中的旧测试副本，3 个 collection 错误；仅在 `pyproject.toml` 增加正式 `tests` 发现路径后重跑，未删除旧测试。首次 Pro 全量虽 1,037 通过，但期间 Core 这一构建配置改变，paired 指纹不一致；因此另跑上述最终配对完整回归。新测试的异常类型／同值 SQL 写入前提、笔记夹具路径、只读产物故障注入权限、安装测试的源码路径防护失败都保留原始记录，未冒充历史产品缺陷。

## 隔离制品与部署边界

全新环境根：`/hot-data/fsx/workspace/erik/orze-long-history-package-2026-09-14.dxa7jzhz`。22 条实际离线构建／安装／检查命令和独立 Git → wheel → installed RECORD 复验通过；23 个依赖 wheel 未升级。

- Core wheel SHA-256：`0501beda118b0300f89a756ce5636c21b326876c00f83cb8be175c943687fc50`。
- Pro wheel SHA-256：`30c94b80f9d30f9dac8f5f4a78a6df8ad6f5437b0b4a2fc34faa7e64f40099d6`。

安装后的 Core payload 在外部 pytest 验证环境重跑原四产品用例：4 passed，8.53 秒，仍有一个真实 CPU worker；111 个实际加载的 Orze 模块逐一匹配新 wheel 和固定 Git 提交。它使用 test harness，不冒称纯 venv 解释器全量运行。根 conftest 的源码路径插入被显式排除，`tests/conftest` 及原安全夹具保留。见[安装产品证据](../evidence/2026-09-14-proposal-installed-review.json)和[补充独立核验](../evidence/2026-09-14-proposal-supplement-verification.json)。

Core 仓公开、Pro 仓私有，已重新只读核对；Pro 原始源码快照、测试清单及包元数据只归档在私有仓，公开仓保留结果、hash 与引用。版本号未变，必须按上述配对提交与 wheel hash 选择，不能混配同版本旧 wheel。没有全局安装、真实 Pro 许可、付费模型、GPU 或生产切换验收；CephFS 的角色／GC 仍只是提前拒绝，不是兼容性已解决。

## 下一片

[预算和 ingress 性能合同](2026-09-14-history-cost-next.zh-CN.md)已经细化：先抽共享原校验器、合并预算行读取，保持任意精度累计与全部逐行校验；再消除空批次历史读取并做本批 ID 定向查询。现有[规模诊断](../evidence/2026-09-14-history-cost-diagnostic.json)只是合成元数据测量，不能当成已完成的优化。

仍未关闭：持久研究摘要、Pro 长程 qualified evidence 检索、全循环历史扫描成本、联合 snapshot 大载荷规划、CephFS 角色／GC 兼容和真实生产环境验收。页结束不等于研究收敛；本轮没有证明科学收益或吞吐提升百分比，也不宣布整个 autoresearch 产品已完成。
