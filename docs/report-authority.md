# Report 的权威范围

`update_report` 保留原有返回行和主/filtered leaderboard JSON 字段，新增独立的 `lifecycle_authority` 标签。它与 benchmark/local evidence 的 `mode` 是两回事。

原生项目（传入 native Lake，或配置声明 `idea_lake_db`）使用同一数据库路径解析、双状态一致的完成集合及共享的证据资格入口。配置的数据库缺失、重定向、不兼容或不可读时不降级为文件自证。只有配置数据库、没有传 Lake 对象且 inbox 为空时，会通过已关闭连接的只读 catalog snapshot 发现完整任务目录，不依赖旧 `_archived_index.json`。

原生排名服从完整 `cfg.report` 的指标、源文件、方向及覆盖要求；展示列的 `metric_harvest/default` 回退不能改变研究证据契约。每次排名重查 taint、clean-access、lineage、benchmark 等当前政策，并在读取前后核实逐行 `evidence_identity`。`_results_cache.json` 是派生展示，不能授予资格或提供分数，即使有人重新计算了它的行校验和。

没有声明数据库、也没有 native Lake 的旧离线调用仍保留 artifact-only 显示与原有缓存快路径，明确标注 `unverified_local_artifact`，不是 research steering 的权威入口。这个兼容接口不意味着加载完整项目配置后的 CLI 可以在数据库缺失时自动降级。

传入 native Lake 或原生 catalog snapshot 时，Pipeline 计数描述审计 FSM 的执行状态；排名还要求 legacy mirror 一致及证据有效。因此 Pipeline Completed 数和合格排名行数可能不同，不能据此认定矛盾。计数仍采用既有 FSM 口径，ARCHIVED/SKIPPED 不进入四状态执行总数；没有把这个计数改成指标合格数。Markdown 与主 JSON 的 `pipeline_scope` 区分 `lake_catalog`、`unavailable_lake_catalog` 和旧 offline 范围。

`--report-only` 已提前分流，不构造 `IdeaLake`、不初始化/迁移数据库，不触发 credential/star/扩展安装/计算启动路径。完整目录和管理面板队列的契约见[只读观察入口](observer-catalog.md)。

边界与后续工作：

- 这是逐行证据一致性，不是全报告的文件/数据库原子快照，也不是独立 observation/attempt 账本。
- Native 缓存暂时只保存展示，每轮重新读取、前后哈希会增加 I/O；本项不声称研究效率净收益。
- CLI 仍会写派生报告文件；“只读”指它不改变权威数据库和实验产物，不意味着完全没有输出。
- Admin queue 的任务状态已独立于训练 metrics；它仍是缓存观察值，不是执行授权或研究排名资格。其他 admin 入口、旧 verifier/director 和显式 convergence 的独立读取/决策路径仍待修复。
