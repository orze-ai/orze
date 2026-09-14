# 后续性能片：消除重复查询，不改变预算含义

这是下一片待实施合同，不是已完成的性能声明。依据是[实际成本诊断](../evidence/2026-09-14-history-cost-diagnostic.json)：合成的单 scope 100／1,000／5,000 行历史中，`_totals` 分别执行 101／1,001／5,001 次 SQL；5,000 行汇总约 1 秒。另有等量其他 scope 行。它不是线上吞吐、真实历史 TREE/effect 证明或研究加速百分比。

## 先实施的最小改动

1. 将 `_reservation` 的既有行校验抽成共享的纯行校验器。按 reservation ID 精确读取仍调用同一校验器；`_totals` 一次读取当前 scope 的完整所需列并逐行调用，不再每行按主键重读。
2. 保留每行 permit／scope／task／slot／ref／state／terminal 标记的现有一致性检查、canonical JSON 与长度限制。仍用 Python 任意精度整数累计 `reserved_nanoseconds`，仍审计全部历史，不做 SQL SUM／浮点汇总、不退款。
3. ingress 在原源锁、fresh read、sidecar、批次／游标解析完成之后，若本批为空则直接退出，不读全历史 ID 和配置缓存。
4. 非空 ingress 对最多 128 个本批 ID 定向查询。已存在的同 ID 仍到实际 insert/exact/conflict 原边界，不让配置缓存替它决定 ACK。跨 ID legacy 配置去重、CPU/legacy kind 分隔、源身份与 ACK 发布顺序不变。

## 不在这一片偷换的语义

- 不新增研究总时限、预算前缀缓存或自动回收额度；不修改已结算预约。
- `_totals` 的既有校验是账本元数据审计，不能声称它每次重跑全部 TREE/effect 证明。执行闭合及恢复的原证明链继续保留。
- 不以遇到损坏行时“跳过并继续”换吞吐，不把坏数据当零费用。
- 不把 schema 不确定性当作索引迁移机会；若需 `(scope,reservation_id)` 索引，单独核对 schema/迁移合同后再实施，不能借这片自动修改生产数据库。
- sidecar 文件枚举／读取仍可能依赖总量，旧 config cache 的加载、legacy repair 也仍有成本。因此最多证明消除局部 N+1 与空批次冗余，不能声称整个 ingress 或研究循环已为 O(batch)/O(1)。

## 验收

- 冻结旧 `_totals` 和 `_reservation` 全文，在私有 scratch SQLite 上比较新旧实际结果及拒绝行为：RESERVED/BOUND/SETTLED、超大精确整数、跨 scope、重复 slot、损坏 canonical JSON、不同 SQL 与 permit 身份、错 ref、非法状态。夹具只标为元数据测试，不伪称真实执行。
- 同 SQL trace 证明 100／1,000／5,000 行时 SELECT 数不再随 N 线性增加；保留原始多次耗时、输出与前后字节指纹。仍明确 CPU 逐行校验的 O(N) 成本。
- 用实际 CPU 执行与真实恢复邻域验证 debit／bind／settle、未知 commit、并发写入、HOLD／Stop、无上限累计额度不变。
- ingress 用真实文件锁／SQLite 验证：空源和空解析批次不查询历史；后页仍可到达；同 ID 精确重放 ACK、冲突不 ACK、sidecar 存在、不安全源路由、缓存误命中和锁丢失都保留原结果。
- 原有测试不删改；先保真实失败／性能基线，再跑定向和完整回归，独立核验后提交远端。

本片完成之后再评估跨 policy 只读续页复用同版本 budget snapshot 是否值得做；这需要独立证明，而不是顺便加入缓存。
