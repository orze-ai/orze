# S3 / C1：声明式、领域无关的数据集覆盖

日期：2026-09-12。冻结方案；修复和验证尚未完成。
依据：[原 V1 方案](2026-09-10-autoresearch-v1.zh-CN.md)及[重开审计](2026-09-12-v1-reopened.zh-CN.md)。

## 缺口与基线

Core `69ff7babb57371632fb3048db781b24a3545548a`、Pro `328cad547c12d2262971cbc20b921d17d31710f4`。当前 coverage 选择器按 `wer_` 猜测数据集；公共 qualifier 的实际同值重命名红对照见 [原始诊断](../evidence/2026-09-12-domain-coverage-diagnostic.json)。既有 S1 绿测是旧规则字符化，不代表原“Core 不按领域指标名猜有效性”合同满足。

## 冻结语义

1. 新的 `report.dataset_keys` 是有序、唯一、非空字符串名单，最多 256 项；每个键最多 1024 UTF-8 字节，不接受空白键。每个成员必须恰好对应一条显式 report column，复用其精确 source/dotpath，不增加 raw metrics 解析旁路。键名及显示顺序不暗示任何领域资格；primary 只有被明确列入时才计入。
2. 未给出该字段时，已有 `report.benchmark_contract.required_metrics` 可作为显式覆盖名单。显式字段为空/坏类型不得偷偷回退。两个声明同时存在时，dataset_keys 必须覆盖全部 required_metrics，且二者各自满足其已有合同；不隐式合并。
3. 无任一覆盖声明时，`min_datasets=0` 或缺省的旧配置继续；正门槛必须失败关闭并给出明确的缺声明原因。合法空名单仅能满足零门槛。不得从 `wer_`、其它指标名、显示列总数或重复列推断覆盖。
4. 计数仍只接受有限、非布尔数；声明源已读到缺失/null 时不得用 metrics.json 的同名代理填补。重复名单/歧义列不能重复计数或胜出。
5. 共享纯声明解析同时服务配置检查、直接 qualifier、计数及实际消费者。直接资格入口对坏/缺声明给出稳定拒绝原因，不抛未捕配置异常，不假装只是一个科学负结果。
6. leaderboard cache 和 champion-history objective scope 必须绑定完整覆盖声明及其语义版本；改变名单即撤销旧资格/历史共享，不能沿用原列名相同的旧缓存结果。
7. `legacy_archive_metric_value` 保持原body及非权威范围；历史档案数值不授予当前发布或排名资格。不新增绕过 live 声明的“legacy 资格模式”。
8. 兼容迁移明确且有限：原正覆盖 fixture 加入其真实数据集名单，不降阈值、不改源码产物、不删失败/无效断言；旧文件完整版本保留在固定基线 Git blob。S1 针对旧选择器的字符化预期显式版本化，不声称新规则与错误旧行为完全兼容。不修改任何用户项目/线上配置。

## 顺序与验收

先提交本方案与原红诊断；新增公共 live qualifier 重命名/缺声明/显式正控回归，冻结并记录旧源码结果。再做最小生产修改：优先仅 reporting.evidence、config 校验、leaderboard cache、champion_history；只有真实调用证据显示必要时才扩大，并记录原因。

独立边界验收覆盖声明类型/去重/来源歧义、benchmark 回退与冲突、零/负数/缺测/非有限、源覆盖规则及配置/直接调用一致性。产品验收经真实本地文件与 Lake 驱动 Core report/rebuild/研究摘要及 Pro 当前研究消费者；不只 spy helper。基线诊断、旧控制、新声明要求失败与草稿错误分别记账。

最终运行目标回归、两仓完整全量、可选 Pro 配对回归；记录源码/测试/示例/构建文件前后指纹、原始日志和 JUnit；检查固定提交、独立审查与远端读回。S3 关闭不关闭 C2/C3，不提前恢复整个 V1 的完成状态。
