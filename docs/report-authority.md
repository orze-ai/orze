# Report 的权威范围

`update_report` 保留原有返回行和主/filtered leaderboard JSON 字段，新增独立的 `lifecycle_authority` 标签。它与 benchmark/local evidence 的 `mode` 是两回事。

原生项目（传入 native Lake，或配置声明 `idea_lake_db`）使用同一数据库路径解析、双状态一致的完成集合及共享的证据资格入口。配置的数据库缺失、重定向、不兼容或不可读时不降级为文件自证。只有配置数据库、没有传 Lake 对象且 inbox 为空时，合格的已完成历史 ID 仍能进入候选。

原生排名服从完整 `cfg.report` 的指标、源文件、方向及覆盖要求；展示列的 `metric_harvest/default` 回退不能改变研究证据契约。每次排名重查 taint、clean-access、lineage、benchmark 等当前政策，并在读取前后核实逐行 `evidence_identity`。`_results_cache.json` 是派生展示，不能授予资格或提供分数，即使有人重新计算了它的行校验和。

没有声明数据库、也没有 native Lake 的旧离线调用仍保留 artifact-only 显示与原有缓存快路径，明确标注 `unverified_local_artifact`，不是 research steering 的权威入口。这个兼容接口不意味着加载完整项目配置后的 CLI 可以在数据库缺失时自动降级。

传入 native Lake 时，Pipeline 计数描述审计 FSM 的执行状态；排名还要求 legacy mirror 一致及证据有效。因此 Pipeline Completed 数和合格排名行数可能不同，不能据此认定矛盾。仅配置数据库但没有 Lake 对象时，当前 Pipeline 计数只覆盖传入 ideas 和合格生命周期的 completed 候选，并非完整 catalog；Markdown 与主 JSON 的 `pipeline_scope` 明确区分覆盖范围。完整只读 catalog 消费属于后续 reader 切片。

边界与后续工作：

- 这是逐行证据一致性，不是全报告的文件/数据库原子快照，也不是独立 observation/attempt 账本。
- Native 缓存暂时只保存展示，每轮重新读取、前后哈希会增加 I/O；本项不声称研究效率净收益。
- CLI `--report-only` 目前仍通过常规 `IdeaLake` 构造器打开既有数据库，其 bootstrap/迁移副作用尚未关闭；本项只验证 `update_report` 和实际 orchestrator 所用的排名入口，未将该 CLI 的只读打开算作完成。
- Admin queue、旧 verifier/director 和显式 convergence 的独立读取/决策路径仍待修复；新标签不会自动修好这些消费者。
