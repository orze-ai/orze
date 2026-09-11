# S2：CPU claim 读取去训练耦合

日期：2026-09-11。先固定契约，再新增测试和修改生产代码。

## 基线与范围

- Core：694754b0927a251ace72a7600e7338142caa4d49；Pro：1384c12d40069c98a430d0e25647de102a7e38de。
- 仅修改 Core 的 engine/claim_authority.py、engine/native_cpu_action.py、core/cpu_action_budget.py。Pro 产品和训练通用 training_attempts._read 原样保留。
- 在现有 claim_authority 中公开 read_claim_snapshot(path, *, limit=65536, required=False)，返回 (dict, 同一次读取原始 bytes 的 SHA256)，仅初始缺失且非必需时返回 None。
- 现有 read_claim 只作兼容投影，返回 dict/None。CPU 的两个消费者显式 limit=8192、required=True；不引入第三套 reader、新模块、schema 或执行协议。

## 必须保留的语义

- snapshot 的纯格式/身份拒绝为 AttemptEffectBusy；required=True 的初始缺失为 FileNotFoundError，OS 错误透传。
- legacy read_claim 把纯校验拒绝映射回原 AttemptEffectInDoubt 和 claim_* 原因，保留其现有事务保守策略。
- 不能只观察外层 CPUActionHOLD：execution_transaction 对 Busy 与 InDoubt 的持久锁保留不同。实际 CPU pre-effect 读取拒绝应回滚并正常释放 guard；已经 prepare/commit 不确定时仍按原事务逻辑保留。
- 单次打开、单次有界读取；parsed value 与 SHA 来自同一 raw，不重新序列化、不二次读取。沿用 JSON 编码、未知字段和数值接受规则。
- 汇合已有两套读取检查的严格项：普通单链接文件、无符号链接路径、大小上限、mode、dev/ino/nlink/size/mtime/ctime 与 FD/path 复验，实际长度须等于快照大小。
- mode 复验是旧 legacy claim 路径的加严；完整长度、父路径结构及 .. 拒绝是 CPU 路径的加严。明确记录这些 fail-closed 加严，不冒称逐异常完全相同。
- 新选项 limit 必须是 1..65536 的精确 int，required 必须是 bool；非法调用选项为 ValueError。CPU 内部原因由 training_* 改为 claim_*，不承诺私有错误文案不变；外层类型、事务效果和保留策略须兼容。
- 读取开始后消失或发生读取/关闭错误不得降级为初始缺失；不修改 safe_file 其他消费者、事务异常分类、预算和来源判定。

## 验收与证据

- 先在固定基线跑已有邻域与新增旧接口/CPU事务行为控制；这些应为绿色。另列新架构要求在基线的失败，不以缺 API/import 错误冒称历史产品 bug。
- 新 snapshot API 的机制用例在实现后执行并单独分类。所有测试先于生产修改冻结，已有测试保持原文。
- 覆盖 raw SHA/Unicode/空白、初始缺失与中途消失、8 KiB/64 KiB 边界、非法 JSON/重复键/非对象、重定向/硬链接/FIFO、短读/inode/mode 变化及 OS 错误。
- 两个真实 CPU 消费者接线：native 启动/完成路径与已确认终态预算恢复；用透明 spy 观察真实公共 reader，不 mock 掉其判据。另验证 pre-effect 坏 claim 不产生 worker、attempt 或持久 effect 锁。
- 定向、Core 全量、Pro 全量、跨仓配对回归；测试使用自有临时文件/SQLite/CPU worker，禁用 GPU，不触及 live 项目。
- 固定 src/tests/examples/构建声明的前后及提交 blob SHA；保留原始命令、结果、JUnit、所有失败分类和独立复核。复用 S1 已核查的记录校验规则，避免每片重新发明复杂证据框架。
- 提交并推送 feat/autoresearch-v1 后读回；不合并 main、不部署、不变更授权。

## 可声称的完成

CPU 对训练私有 reader 的直接依赖 2→0；scheduler claim 与这两处 CPU claim 共用一份读取实现，旧 read_claim 为兼容投影。全仓 reader 数量不据此下降，生产源码净行数如实记录，不保证负增长。不声称整个技术栈、所有训练耦合、原计划 V1 或科研效率已完成。
