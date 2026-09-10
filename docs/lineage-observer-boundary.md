# Lineage：生成证据与读取证据的边界

启用 `model_lineage` 时，报告资格、评估资格、benchmark receipt 验证、公开排名资格、campaign lineage audit 和 managed run 结束验收都只消费已有数据隔离收据。它们不会补建 compute receipt 目录、重扫 train/evaluation manifest、等待审计锁、建立临时 fingerprint 索引或发布新的隔离收据。

`read_data_separation_receipt` 核对完整配置、manifest 路径与元数据、既有 receipt 的 envelope/hash/策略与计数约束；读取前后再次核对 manifest 元数据。缺失、损坏、策略改变或文件元数据变化即拒绝，不退回自动审计，也不省略 lineage 政策。只读验证仍会读取并校验模型产物，保留原有双次哈希、boundary/start/terminal、访问日志和 benchmark 绑定；因此不能把本项描述为“所有证据查询都是低成本”。

训练启动前的 `ensure_data_separation` 仍负责创建审计收据，产物封存仍使用原有生成路径。若收据丢失，观察者不会通过重建一个新时间戳收据来修复历史 lineage：新收据的 hash 与原 attempt 的绑定可能不同。恢复历史资格需要找回匹配的可信收据，或者按执行协议产生新证据，不能仅靠刷新报告。

兼容及证明范围：

- 未启用 lineage 的任务不因此被要求提供模型或数据隔离收据；独立声明 `managed_run.require_data_separation` 的结束验收同样只读。
- “只读”指这里的证据消费不修改实验或控制证据。报告和 admin 的派生文件仍可正常发布；没有承诺操作系统访问时间不变。
- 元数据检查沿用既有 receipt 缓存信任模型，并不是对可控制底层文件系统的攻击者提供新认证，也不是所有文件与 SQLite 的原子快照。
- 旧 offline 报表的 warm cache 没有覆盖 manifest 元数据变更，仍可能展示旧行；它是明确未验证的兼容展示。native 排名每次重新资格验证，不依赖该捷径。本项不宣称所有旧缓存消费者已统一。
- 既有 I2 测试证明的是 catalog/CLI 的权威数据库非迁移读取，未覆盖 opt-in lineage 引发的审计副作用；本项单独补充该缺口，不倒推扩张旧证据。
- 小型合成 manifest、临时真实 SQLite 和现有执行收据 fixture 验证机制；未运行真实训练、GPU、付费 provider 或生产科研收益对照。
