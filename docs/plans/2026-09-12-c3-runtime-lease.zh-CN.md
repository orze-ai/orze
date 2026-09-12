# C3：原生 CPU 运行期租约简化实施合同

日期：2026-09-12。状态：**根审实施合同；C1/C2 全量关闭后解锁实现，尚未验收通过**。

本文承接[原 V1 方案](2026-09-10-autoresearch-v1.zh-CN.md)的运行中租约超时要求和[重新审计](2026-09-12-v1-reopened.zh-CN.md)的 C3。保留[第一版提案](2026-09-12-c3-runtime-lease-proposal.zh-CN.md)作为设计历史；本稿不再采用其中的新表、新 nonce capability 或第二套状态机。C2 完成收口后才解锁生产实现，不并行改变前片冻结源码。

## 1. 范围与复用

新派发 native CPU action 默认获得不可续租的运行期授权。沿用同一 CLI/Orze、IdeaLake、attempt 完整 Ref、预算 Permit、强 Owner、现有 supervisor、terminal/effect 和 F 结算恢复。

不新增数据库表、JSONL、后台服务、runner、租约 ID/nonce 或 ACTIVE/EXPIRED/TERMINAL 副状态机。既有 LAUNCHING/RUNNING/TERMINAL/NOT_STARTED/IN_DOUBT 仍是执行状态权威；预约继续 RESERVED/BOUND/SETTLED，费用不退款。

已有身份足够：native 强 Owner 核真正 handle/process、scope、完整 Ref 与 canonical binding；Permit 绑定同一 reservation/ref；READY 还有原监督器 nonce 和实际进程身份。因此新 descriptor 是原 attempt 的不可变授权条件，脱离副本不是新执行凭证。未关闭旧代不能创建下一代；fresh controller 不从 deadline/数据库行重建旧 Owner。

旧非 CPU/legacy-profile 保留旧监督协议，不借本片声称它们获得同等运行期租约保障。不扩分布式接管、续租、人工强制释放或 checkpoint resume。

## 2. 配置和不可变 descriptor

项目顶层可选：

```yaml
cpu_runtime_lease:
  version: 1
  ttl_seconds: 2
```

- 字段缺省：有效 TTL 使用动作已声明的 `timeout_seconds`，所有新 native CPU 仍默认启用。
- 字段存在时须为 exact `{version, ttl_seconds}`；version 为 exact int 1，TTL 为非 bool、正、有限数，且不大于该动作 timeout。显式 null、false、0、未知字段或超范围均拒绝，不能作为禁用值。
- 仅在 `config._KNOWN_EXTRAS` 登记该键，不向 `DEFAULT_CONFIG` 添加 None，保留 absent 与显式 null 的区别。非 CPU 配置提供此字段须拒绝。
- `cpu_execution` invocation fingerprint 纳入规范化租约声明；有效 TTL 在动作派发时再次核实。不要修改原四字段预算 namespace、action spec/purpose、Domain request 或 replica config 来获得新额度/规避去重。
- 纳秒转换须有界、可重现且不向上超出声明 envelope；无法表示正纳秒或溢出时拒绝。不能以舍入生成零期限后默认为禁用。

在原 INTENT writer 内、创建 attempt 前采样时间，将下列 exact 对象持久到 `attempt.binding.runtime_lease`：

```text
{schema: 1, clock: 'CLOCK_BOOTTIME', hostname, boot_id, issued_ns, deadline_ns}
```

schema 为 exact int，hostname/boot_id 为严格验证的本机身份字符串，两个 ns 值为有界 exact int，`0 <= issued_ns < deadline_ns`。TTL 从该采样时刻起算，包含随后 bind/工作目录/READY/GO 的准备时间，不等同于旧的“GO 后才开始计时”。descriptor 连同原 binding 经实际 writer、commit 和独立读回确认后才允许 prepare/GO。后续 RUNNING/terminal binding 可按旧合同更新 lifecycle，但不得改、删除或续期 runtime_lease。

仅同 hostname/boot_id 使用 Linux CLOCK_BOOTTIME（包含 suspend 时间）。`now >= deadline_ns` 永久失去正常执行/发布权；时钟不可用、倒退到 issued_ns 之前、boot 不符、类型不明是 HOLD，不平移旧 deadline。活 Owner 一旦观察到期不得因后续读时钟值变化重新启用。正常同 boot 单调性、Owner 不可重建和旧 attempt 未关闭门共同阻止复活，无需另写一张 EXPIRED 表。

## 3. 启动、运行与监督协议

新 CPU 使用明确的新监督协议版本；旧版本的 exact 字段检查仍保留。新版本 READY 和 TREE_CLOSED 精确绑定完整 descriptor，父端必须同时核 descriptor 与本次 binding/current Ref/Permit，不允许新 CPU 用无期限的旧收据降级通过。

协议冻结为 `orze.linux_subreaper.v2`，保留旧 `PROTOCOL` 常量为 v1。v2 READY envelope 仍 exact `{event,binding}`，binding 为原七键加 `runtime_lease`，schema 为 exact int 2。v2 CLOSED 为原八键加 `lease_expired`（exact bool）、`lease_observed_ns`（有界 exact int，至少 issued_ns），schema 2；到期标记与观测是否达到 deadline 严格一致，到期必须已进入 stop_requested 状态。descriptor 仅经 binding 携带，不重复一份。GO/STOP 原 nonce 帧不变。旧非 CPU 调用仍使用原 exact schema 1。

纯 stdlib descriptor/clock helper 放在原 `supervisor_worker.py`，确保现有 `python -I` 执行可用。低层 prepare 和五位置参数构造器仅添加可选 keyword `runtime_lease=None`，None 仅表示旧调用；配置层显式 null 仍拒绝。Popen 前 fallback 与 Popen 后正常 handle 都保存相同 detached descriptor，不能移走原不确定性交接边界。

复用现有 supervisor executable 和 subreaper：

1. INTENT 持久绑定后、prepare 前核身份/Permit/动态 admission/当前期限；READY 后和真正 GO 前再核。
2. supervisor 经现有私有启动配置接收 descriptor，自行核 hostname/boot/clock；开 worker gate 前检查。到期后绝不 GO。
3. 同一 supervisor 在现有 select/drain loop 自主检查期限，即使 parent 卡在 callback、Policy 或 I/O 且控制通道仍开，也会到期停止其自有树。仍按原 TERM→KILL 有界升级和真实 ECHILD_WALL 证明闭合，不用 Lake 裸 PID 发信号，不扫描宿主进程。
4. 父端每次观察活 Owner 同样核期限；不能保留当前“poll 已返回非 None 就跳过 deadline”的正常发布漏洞。已知到期与身份/存储未知分流。

原 wall envelope 保守预留保持不变。更短租约、默认租约与 supervisor 的执行期限共同工作，而非只把原内存 wall timer 改名。

## 4. 执行权、正常发布权与停止收尾权

### 正常发布

已有强 Owner/当前 Ref 校验负责“是谁”；期限检查单独负责“此刻能否 GO 或正常发布”，不能把二者合并得使到期 Owner 连安全停止都做不到。

Domain interpret、文件 hash/复制等重工作仍在 writer 外，但消耗租约时间。不能用提前 FINISHING 豁免它们。正常成功/失败的 publication 必须保持未到期，包括：

- worker 闭合后开始解释/准备前，以及进入 terminal writer 时；
- artifact/observation 注册之后、最终 terminal CAS 前；
- 原 `execution_transaction` 的 commit 前、commit 后及 effect confirm 门。

正常 publication 的逻辑线性化点是现有短 writer 内、所有结果注册之后的最后 lease-aware terminal CAS：严格当前 Ref、捕获 binding、同 boot 且未到期才可提交这份 terminal。需要将时间检查接入现有 watch，不能只验证 attempt 行值没变。保留原 artifact/observation/source/lifecycle 全部末态围栏。

最后 gate 后提交/确认仍有时间窗口：跨期限或结果未知则 HOLD，不 confirm 正常 effect、不 SETTLED；已经发生的 SQL 提交不能伪称回滚或倒写。confirm 前后及其返回必须保持对应保守不确定性处理；若确已留下 committed receipt 但确认结果/时机未知，不能宣称它未发生。记录真实 poststate，沿原 effect HOLD/保留语义处理，不凭年龄再执行。这里不声称时钟、SQLite 和多个文件 fsync 是物理单原子，也不承诺纳秒级停止。

### 已知到期

在正常 `tx.prepare` 尚未开始时，真实活 Owner 可使用窄停止收尾权限：不再 GO/正常 publish，读取或一次请求停止 captured supervisor，核真实完整闭合后写专用 `interrupted` 和 typed expiry 证据。SIGTERM handler 返回 0 也不能 completed；无成功 artifacts/observations；原 terminal/effect 确认后才由原 budget.settle 释放槽位，不退 wall 额度。

若 worker 早已自然闭合，但 Domain interpret 消耗到期，同样不得正常发布。专用 expiry 终态必须如实保持原 closure 的 stop_requested，不能伪造监督器发过 STOP；用独立 typed expiry 元数据说明为何自然闭合仍 interrupted。

新 CPU terminal 精确增加 `runtime_lease: {schema: 1, status: 'authorized'|'expired', observed_ns: N}`；descriptor 由 attempt/READY 绑定，不在 terminal 复制。authorized 采样须在 issued/deadline 半开区间内，随后每个 gate 再采样；expired 采样须至少 deadline，outcome 必须 interrupted、reason_code 必须 `cpu_runtime_lease_expired`，artifact/observation 集合为空。原 prepared plan 同时绑定该元数据。该采样不冒充 commit 的物理时间，也不是跳过后续 gate 的凭证。正常关闭但解释后到期可以保持 closure.lease_expired=false，同时 terminal.status=expired。

窄终态收尾适用于已有 RUNNING 与可验证闭合的原 Owner。准备期到期而尚无 RUNNING/闭合证据时不得 GO，保守保持原 LAUNCHING/BOUND/HOLD，不虚构未启动收据来回收额度。已 RUNNING 的 GO 前已知到期可返回原 handle 供同一 harvest 窄收尾；未知 clock/身份仍 HOLD。

已进入正常 tx.prepare，或 commit/confirm 跨期限/未知时，保留原 prepared effect/锁及 HOLD，不开第二事务重写 interrupted。不能将“到期”当作消除未知副作用的证据。通用 IN_DOUBT 也不能为了便利再允许 finish；旧 finish 对 RUNNING 的要求不放宽。

## 5. 失联、结算及 F 兼容

监督器丢失、STOP/闭合未确认、descriptor/Ref/Permit 改变、clock/boot 不明：原强 Owner/BOUND/HOLD 保留。过期、PID 消失、空内存 map 或重启均不释放 reservation、不再 claim/GO、不退预算。测试拥有的清理不算产品完成。

旧已经 confirmed 的 terminal 继续按历史协议验证，无需与当前时间比较，也不补写 lease。新协议 terminal 的 expiry/nonexpiry 元数据须与 descriptor、实际 closure、terminal outcome/reason、artifact/observation 集合和 effect 完整一致。F 只能恢复这些已经确认的终态结算，不能重新解释 Domain、重发副作用、复活旧运行权或处理只有过期 RUNNING 的行。原恢复屏障、storage Stop 和零退款规则保持。

协议分支必须 exact typed/canonical；新字段不许靠允许任意 extra 兼容。历史确认日期过去不是证据失效，新旧协议也不可相互冒领。

若 confirm 已真实留下 committed.json、其返回之后 gate 才发现过期，保留真实 SQL/文件并由原 `AttemptEffectInDoubt` 保留 `_attempt_effect.lock`。F 即使能读取该收据，settle 仍必须取得同一 no-age guard，因此不得释放。进程在 confirm 与后检之间直接崩溃同样留下尚未退出的 guard。不新增第三份确认文件或试图抹除已发生的确认；对外结论是 HOLD，而不是声称文件从未写出。

## 6. 最小落点

- `core/config.py`、`core/cpu_execution.py`：配置存在性、规范化和 invocation fingerprint。
- `engine/native_cpu_action.py`：INTENT descriptor、强 Owner、prepare/GO/运行/解释/发布期限检查，known expiry 窄收尾与 unknown HOLD。
- `engine/supervised_process.py`、`engine/supervisor_worker.py`、`engine/process_supervision.py`：原监督器的新 CPU 期限协议、自主停止及 strict READY/CLOSED 验证。
- `engine/execution_authority.py`：原 terminal writer 的 lease-aware 最后 CAS/watch/commit/confirm 围栏；不新增跨 task 写权。
- `core/cpu_action_budget.py`：新版 terminal 和 F 的历史验证，不改变预算 namespace/状态或退费规则。
- 必要时 `engine/cpu_phase.py` 仅作期限错误路由/有界轮询衔接，不新增第二个控制循环。

共享的 descriptor/clock 验证可以是一个无持久状态的小 helper；本文不要求额外公开 API 或固定内部函数名。既有 consumers 若有 strict schema 依赖，实施前列清确切改动，不借机迁移所有 GPU/role adapters。原 A/B 应用、Domain/Policy、旧测试和原留出内容不得为 C3 特判；重跑公开留出只是兼容回归，不是新的未见验收。

## 7. 五类有限验收

各类少量正负参数即可，使用私有 CPU 项目、真实 SQLite/native/supervisor、立即认证的自有 pidfds；不 mock closure，也不填假 terminal。

1. **默认与运行中到期**：缺省配置真实走新期限；显式较短 TTL 的已 GO worker（含逃逸后代）到期真停止；STOP0 仍 interrupted、无成功结果、一次 SETTLED 且不退款。期限内正常完成控制，以及 null/false/0/超 envelope 拒绝。
2. **父端不轮询**：GO 后在测试私有 barrier 阻塞 parent，保持控制通道打开；supervisor 自主按持久绑定期限闭合整树。父端恢复不能凭旧返回码或 capability 正常发布；不得用父端主动 stop 冒充自主到期。
3. **真实最终 publication 窗口**：Domain/产物/观察实际注册后才使期限到达，最终 gate 拒绝正常 terminal/effect/settlement；另核慢解释与 commit/confirm 响应未知的真实 poststate。已 prepare 的失败保持 HOLD，不能二次改写为 interrupted。
4. **失联和陈旧授权**：杀测试立即捕获的 supervisor、保留自有 escaped writer；到期后重复观察/fresh CLI 仍 HOLD，无新 claim/attempt/GO、不释放槽位。换 Ref/descriptor/Permit、改 boot 不获得执行权。只用测试自有资源清理。
5. **存储、时钟与历史恢复**：严格 bool/float/int 和 binding 不可变；commit silent-rollback/已提交后响应丢失、不相关 peer 正常写入；真实已确认终态后 crash86 的 F 单次恢复；旧已确认协议/非 CPU 调用保持。compat 正控不能冒称运行中到期证据。

先保存完整 baseline/candidate 源和新测试，再实跑真实行为。缺新配置/API 或 import 错误不算旧产品红；当前 parent 不轮询继续运行与晚到正常 publication 才是可定位旧行为。记录真实 deadline、观察/停止耗时、精确引用、effect/预算 poststate 和所有失败；不宣称普遍研究收益或 exactly-once 外部副作用。

关闭仍需目标红绿/负控、完整旧兼容/双仓全量、固定提交与远端读回。本文冻结实施边界，不表示已经实现或完成；原三个运行诊断和其真实失败另存证据。
