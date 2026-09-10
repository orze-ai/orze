# 只读观察入口：report-only 与 admin queue

`orze --report-only -c /path/to/orze.yaml` 使用完整配置和真实 report 生成器，但不进入运行时初始化。数据库必须已存在且符合共享 SQLite 策略；缺失、重定向、不兼容时不会新建、迁移或从 metrics 自证完成。配置无效/权威不可用返回 2；权威不可用时可以输出空排名及诊断。报告正常写出后返回 0。

配置必须存在。相对 results、显式 ideas 和自定义 DB 路径按所选配置目录解析，调用结束恢复原 cwd。保留现有配置加载器的默认值：默认 `.orze`、inbox 和 DB 位于 **results 的父目录**，即使 results 是嵌套或外置目录，也不把已有权威数据库搬到 config 旁边。显式其他子命令仍优先；report-only 与 stop/restart/admin 等全局动作旗标组合被拒绝。正常运行入口的路径处理不在本切片中重写。

`load_catalog_snapshot` 以 `mode=ro` / `query_only` 打开数据库，在一个只读事务内读取 schema 和任务目录，随后关闭连接。返回对象只含值，不持有可写 Lake 或数据库连接。副本修改不影响原快照，数据库后续变化也不会反向修改旧快照。

报告 JSON/Markdown 和 admin 缓存实际写入后会读回核实内容，避免底层静默跳过写入时宣称成功；未变化的报告仍保留 mtime/Updated 时间。单个输出失败不会回滚已成功发布的其他派生文件，因此这不是跨文件原子发布；存储恢复后可以重新生成。

管理面板的 producer 与实际 `GET /api/queue` 保留既有字段和分页形状，新增权威标签。配置声明 DB 的 native 队列合并持久目录与展开后的 inbox；因此 inbox 为空、原 Lake 已关闭时，持久化补评仍可见。任务状态来自一致的 FSM/legacy mirror；已完成任务若有明确矛盾的 training/evaluation stage，则标为 unknown。历史没有 stage 记录不伪造成功 stage；训练-only 的 evaluation SKIPPED 仍兼容。

| 权威任务状态 | 队列显示 |
|---|---|
| QUEUED | pending |
| CLAIMED / IN_PROGRESS（包括等待补评） | running |
| COMPLETE | completed |
| FAILED | failed |
| SKIPPED / ARCHIVED | skipped / archived |
| 缺记录、状态冲突、权威不可用 | unknown |

训练 `metrics.json` 的 COMPLETED 不再把未完成的评估显示成整个任务完成。单行冲突不隐藏其他正常任务；损坏 schema、重复 join identity 则让整个 catalog 不可用。没有声明 DB 的旧离线展示仍保留 artifact-only 行为，但 producer 和 API 都标为 `unverified_local_artifact`，不能用于研究决策。

普通报告只读轻量元数据，不选择或解析实验 config 列。Admin 为显示持久任务配置显式启用受限读取：每项最多 64 KiB，YAML 事件最多 4096，JSON 树最多 2048 节点、32 层、64 KiB 字符串；非字典、非 JSON 类型、非有限数和 alias-bearing YAML 不进入显示配置。此时 `config={}`、`config_available=false`，**只表示展示未加载，不判执行配置非法，也不改变任务状态**。这些限制针对持久配置的观察读取，不冒称已经约束整个运行时的 inbox/parser/prompt。

边界：

- Pipeline 仍是既有审计 FSM 计数，队列是状态一致性显示，排名是证据资格：三者不可混为一个计数。
- 本项不是全数据库/文件原子快照，不解决 observation 去重、历史修订、过期 attempt 或执行租约。
- 全目录快照仍是 O(任务数)，admin 配置预算是逐项预算，不是全目录总内存上限。API 仍读取派生缓存，保留既有 TTL，不保证实时状态。
- 本项只关闭 report-only 和队列路径，不代表所有 admin 端点、verifier、director、convergence 已迁移。
- 真实临时 SQLite、报告文件和进程内 HTTP API 验证不等于生产部署、GPU 训练、付费 provider 或科研效率收益验证。
