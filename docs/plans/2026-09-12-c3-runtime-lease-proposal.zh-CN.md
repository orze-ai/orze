# C3：原生 CPU 运行期租约最小实施提案

日期：2026-09-12。状态：**待根审冻结；仅设计，尚未授权本片生产实现或宣告验收通过**。

本文承接[原始 V1 方案](2026-09-10-autoresearch-v1.zh-CN.md)第 41 行和[重新审计](2026-09-12-v1-reopened.zh-CN.md)的 C3。原方案要求运行中的租约超时与 worker 丢失处理；不能以启动前 LEASED 到期检查加上 worker wall timeout 代替。按 C1→C2→C3 顺序实施，不与前片并行修改相互依赖的生产行为。

## 1. 缺口与复用边界

当前已读实现的准确边界如下；行号为写提案时的位置，不是未来代码的固定接口：

| 位置 | 现有能力 | 不能据此宣称 |
|---|---|---|
| `engine/trigger_delivery.py:307,322` | begin_launch 检查 LEASED deadline，启动时清空 lease_until | STARTED 后持续有效、可到期的运行租约 |
| `engine/native_cpu_action.py:67,375,406` | owner 内存 deadline；harvest 在进程仍运行时检查 wall timeout 并请求停止 | 持久运行授权到期；controller 阻塞时监督器自行执行到期 |
| `core/cpu_action_budget.py:38,512` | RESERVED/BOUND/SETTLED、完整 Ref、slot、保守 wall 预留、Stop/恢复屏障 | permit 有运行期时间失效或续租语义 |
| `engine/attempt_effect_lock.py:1–6,36` | 短事务的 nonce/no-age 文件系统所有权 | 可按年龄失效的 process lease；本片不得修改为超龄夺锁 |
| `engine/supervisor_worker.py:183–228` | 原 subreaper 自主追踪整树；STOP/通道 EOF 后只清理其拥有的后代 | 当前存在授权 deadline 或 renewal 帧 |
| `engine/execution_authority.py:329–348` | 同 Lake BEGIN IMMEDIATE、末态/watch、commit 前后核实、随后 effect 确认 | 当前已经核查运行 lease 的有效性 |

复用同一 CLI、Orze、IdeaLake、native CPU owner、supervisor、effect/terminal 和预算结算，不新增 runner、后台服务、跨节点协调器或第二本预算账。

本片只给原生 CPU action 建立本地运行期授权。旧 GPU/role/脚本适配器保留原协议，不能因本片单独通过而宣称它们获得同一租约能力。既有失联 HOLD 与 no-age effect lock 不削弱。

## 2. 最小产品合同：默认启用、不可续租

根审取向已明确：每个**本片之后新派发的 native CPU action 默认必须持有运行 lease**，不能增加默认关闭开关后宣称满足一般保障。

- 默认 TTL 取该动作已声明的 `timeout_seconds`，不追加预算、不改变保守预留、不退款。
- 可选的项目级 `cpu_runtime_lease: {version: 1, ttl_seconds: N}` 只缩短 TTL；N 必须为非 bool、正、有限数，并且对每个被派发动作不大于其 wall envelope。没有 false/0/null 禁用值；字段缺省表示上述默认，而非无 lease。
- 不将该字段塞入已有四字段 `execution` 预算声明，不因更改 TTL 为同一 results/DB 开新额度 namespace。需把规范化租约声明加入加载配置指纹，并将**有效 TTL、完整当前 Ref 与 lease descriptor**写入本次 attempt 绑定。不得给 action specification、purpose 或 replica raw config 加盐。
- 非 CPU 配置若提供该 CPU 专用字段，明确拒绝，不猜模式。字段名和错误码仍须根审冻结；不实施隐式用户配置迁移。
- 版本 1 不支持 renewal、延长、换 owner 或接管。不可续租不等于无运行租约：本方案新增持久撤权、独立监督执行期限和发布围栏；不是给内存 wall timer 改名字。
- 默认 TTL 从 lease 成功发行前捕获的时刻计算，发行位于已有 reserve/claim/create-attempt/bind 之后、prepare_supervised 之前。因此 READY 准备也消耗授权 TTL；原执行 envelope 不被延长。这个更严格的时间边界必须写进最终兼容说明，不能声称与旧超时完全等价。

### 时间与身份

采用 Linux `CLOCK_BOOTTIME` 的整数纳秒，同一 host/boot 内使用统一时钟（包括机器 suspend 时间）；UTC 只作诊断。绑定 host 标识、内核 boot_id、issued_ns、deadline_ns；`deadline_ns = issued_ns + ttl_ns`，数值和加法有界、严格类型检查，`now >= deadline` 即到期。

父进程与 supervisor 独立核 host/boot/clock。时钟不可用、身份变化、反向到 issued_ns 之前、编码/范围不明均 HOLD，不将旧 deadline 平移到新 boot。host/PID 字段只是身份约束，不赋予发现、杀死或接管别的进程的权限。

## 3. 最小持久结构与 API

建议新增 `core/cpu_runtime_lease.py`，在**同一 supplied IdeaLake**中拥有一张严格表 `main.cpu_action_runtime_leases`，不改现有 reservation 状态集合：

- `lease_id`：随机 48hex 主键；不依赖时间/PID 推导。
- `reservation_id`：唯一，绑定原预算预约。
- `scope`、`ref_json`：scope 精确等原 budget scope；完整 task/phase/attempt/generation，Ref 唯一。
- `binding_json`：有界 canonical JSON，含 schema、原 budget_scope、原 permit 摘要、完整 Ref、owner_nonce_sha256、host/boot/clock、issued/deadline/ttl。DB 路径及 inode 继续由原 scope 约束。
- `state`：仅 `ACTIVE | EXPIRED | TERMINAL`。
- `expiry_json`：到期原因/观察时刻或撤权诊断；EXPIRED 后不可改回 ACTIVE。
- `terminal_sha256`：仅同事务精确终态关闭 lease 时记录。TERMINAL 只说明运行授权已消费，**不等于 effect 已确认或 budget 已 SETTLED**。

owner_nonce 明文只由实际活 owner 强持；可持久化 descriptor 不可重建可执行 capability。公开 snapshot/报告只提供脱离副本。不要把 dataclass 或 dict 本身当启动权。

拟用最小操作：

1. `initialize(lake)`：明确初始化入口唯一建表；只读验证/require 不偷偷建表或修 schema。
2. `issue(lake, permit, ref, ttl_seconds) -> strong lease handle`：独立短 writer；核原 BOUND、current LAUNCHING、scope/Stop/恢复屏障；create-only；commit 完整退出后独立只读回验本行。任何不确定不重复发行给相同 reservation。
3. `require_active(lake, handle, *, conn=None)`：核捕获身份/精确行/Ref/当前时间；返回已验证 descriptor。传入 conn 仅限同 Lake 的实际 writer；不 commit 调用者事务。
4. `expire(lake, handle_or_verified_record, reason)`：CAS ACTIVE→EXPIRED；读回 exact row；不改变任务/预算，不释放槽，不生成新 claim/attempt。到期事实不需要新 owner，但停止权限仍只来自活 owner 的确切进程句柄。
5. 终态 writer 的 `close_in_tx(..., ref, terminal, disposition)` 与只读 watch：与 terminal/effect prepared 信息在**同一事务**关闭租约；正常完成必须仍 ACTIVE 且未到期。EXPIRED 仅可用下述停止收尾权写 interrupted。
6. 历史验证函数：只读核完整 lease/Ref/terminal hash，给 budget.settle 与 F 使用；不能将历史 TERMINAL descriptor 当当前 GO 权。

每个事务只捕获并复验本次行/绑定及必要当前源，不要求整个 scope 的所有历史行或同行预约保持不变。BEGIN IMMEDIATE、schema/TEMP 大小写别名防护、commit 不确定与新连接读回必须沿用既有原则；不得以同连接缓存值证明提交。

## 4. GO、运行、正常发布与停止收尾

### 4.1 GO 与监督期限

已有 create-attempt、permit bind 后发行 lease；持久发行确认后才能 prepare。将 descriptor 通过现有私有父子通道交给 supervisor，READY 必须回显并强绑定同一 descriptor；记录 RUNNING 时固定该绑定。真正发 GO 前再次核 lease/permit/原动态 admission，supervisor 自己也在开 gate 前核 deadline。缺失/过期/改代均无 GO。

新 CPU 使用显式租约版监督协议。supervisor 仍是原 executable，仍持有自己的 subreaper/owned pidfds；在现有 select/drain loop 中检查 deadline，不依赖父进程 harvest、健康检查、Domain callback 或 Policy 返回。到期后不可接受 GO；已运行则按已有 TERM→KILL 有界升级规则停自有树，直到真实 ECHILD_WALL 才给闭合收据。它不读 Lake、不扫描 host、不根据表里的裸 PID 发信号。

### 4.2 正常运行与成功 publication

CPU loop 每次观察活句柄都核持久 lease；不能只在 prepare/GO 核一次。已知到期走专用 expiry→owned-stop；数据损坏、owner不明和提交结果不明仍是 HOLD，不把它们全部降格成普通到期。

正常 harvest 即使已读到 leader/树 return 0，也必须核 lease，不能保留当前“ret 非 None 就跳过 deadline”的缺口。Domain 解释、来源/产物哈希等重工作保持 writer 外，但消耗的时间不能豁免最终租约检查。

**不引入提前 FINISHING 或不限时 publication 豁免。**正常完成的线性化点是同一 terminal/effect writer 内、所有 artifact/observation/source 工作之后的最后 lease CAS：ACTIVE 且 `now < deadline`，当前完整 Ref、原 nonce 与 descriptor 一致，才可与这份精确 terminal 一起变为 TERMINAL。所有已有 watch 仍在 commit 前后执行；lease watch 必须加入此闭包，不能仅入事务时检查。

- 在 artifact/observation 注册后、terminal/最后 CAS 前到期，整个正常成功事务回滚；不留下合格成功结果，不允许 SETTLED。
- 在最后 gate 之后但提交/读回尚未确认时跨过期限，采用保守 HOLD；不得用“曾经检查有效”恢复正常 publication。
- 若 SQL 已提交但确认时已过期/响应未知，不倒写历史、也不伪称 rollback。保持 effect 未确认/不确定及 BOUND；记录真实 poststate。实际完整 terminal/effect 已在有效期限内确认后才是正常已完成，不因日后时间流逝废掉历史证据。
- effect 文件确认前仍需最后有效性核查。最终存储响应丢失不能借 lease 年龄自动判断已完成或补发新动作。所有组件间不宣称跨文件与 SQLite 的原子物理提交或硬实时时限。

最后一项存在不可消除的 I/O 时间窗口：实现必须保留 prepared/confirmed receipt 与原不确定性语义；若不能证明在授权内完成正常确认，结果为 HOLD。不能为了边界测试通过而补造精确 commit 时刻。

### 4.3 到期后的窄停止收尾权

过期撤销执行/正常成功发布权，但不撤销安全停止自有树的能力。活 owner 仍强持原 process/Ref/permit/lease identity，仅可：

1. 持久 CAS 过期；一次请求停止 captured supervisor。若 supervisor 已自主到期闭合，先读取实际闭合，不重复 STOP。
2. 核整树、实际 closure 与 lease descriptor 一致。SIGTERM handler 返回 0 仍是 interrupted，禁止 artifacts/observations 成功发布。
3. 用只针对该 Ref、该 expiry 和该闭合的停止收尾路径写 `outcome=interrupted`、typed `reason_code=cpu_runtime_lease_expired`，终态仍经过原 effect 事务及确认。
4. 只有 terminal/effect 确认且原 budget.settle 自身核实后释放 slot；reserved wall 不退款。未用的额度与原预约不得伪装成已消费 CPU 时间。

正常成功的失败事务若已留下不确定 effect，不能直接开启第二事务“改成 interrupted”掩盖；保持 HOLD。已知无副作用且完整 rollback 后才可进入窄收尾。陈旧 nonce/换 Ref/换数据库不能使用这个路径。

## 5. 失联、重启及兼容

- supervisor 丢失、closure 缺失、STOP 未确认、boot/clock 不符、lease/terminal/effect 提交未知：强 owner 和 BOUND 保留；fresh CLI 也不因期限已过、PID消失或空 map 自动放行该资源/任务。
- fresh controller 可只读识别已过期 ACTIVE 并安全记录 EXPIRED；不能恢复旧 capability、发现旧进程、重派旧任务或释放 reservation。无 controller 时 supervisor 的独立期限只能证明其自身停止动作；没有被当前 owner核实的闭合，不自动转产品终态。
- F 仅对已有完整当前 TERMINAL、effect 已确认、lease TERMINAL/终态hash一致的新版 action 继续原结算恢复；不重跑 worker/Domain、不重写 artifacts/observations。IN_PROGRESS 屏障与持久 Stop 原规则不变。过期但仅 RUNNING/EXPIRED 的行不属于 F 可恢复集合。
- 老的已确认 terminal/receipt 保持精确旧版验证，不能补造历史 lease。老 ACTIVE/BOUND 原生尝试也不在升级时追授新 lease或新GO。
- 监督器公共默认调用 `runtime_lease=None` 仅供原非 CPU/旧协议调用兼容；新 native CPU launch 必须传有效 lease，并严格拒缺失。拒绝新 CPU 通过 v1 receipt 降级完成。
- 新协议使用单独版本和 exact 字段集合，不能通过“允许任意 extra fields”放宽现 v1。READY/closure 的 lease descriptor、停止原因和时钟字段都严格 typed/canonical；布尔值不得冒充 schema/int 时间。
- 原研究 spec、Domain/Policy、A/B 应用与既有独立留出内容不改。C3 后重跑这些公开任务是兼容回归，不重新宣称未见留出验收。

## 6. 最小源码落点与隔离范围

| 文件/入口 | 计划责任 |
|---|---|
| 新 `core/cpu_runtime_lease.py` | 同 Lake lease 表、强 cap、发行/当前校验/expiry CAS/终态watch/历史验证 |
| `core/cpu_execution.py` 与配置验证入口 | 默认启用/可选缩短的规范化与 invocation 指纹；不改预算 namespace |
| `engine/native_cpu_action.py:launch,_admit,_owned,harvest,_terminate,stop` | 发行、READY/GO绑定、运行检查、期限与正常/停止权限分离、持有/退役条件 |
| `engine/cpu_phase.py:initialize,iteration,close` | 初始化、持续观察；已知expiry进专用停止；未知仍HOLD；Wait/Stop不新GO |
| `engine/supervised_process.py:prepare_supervised,_accept,start`；`engine/supervisor_worker.py:_run` | 同一 primitive 的严格租约版本、deadline自主停止及绑定收据；不新增进程发现方式 |
| `engine/process_supervision.py:ready_binding,bound_binding,require_closed` | exact 版本化资格检查；旧适配器不降级/不冒领 |
| `engine/execution_authority.py:ExecutionTransaction._verify_watches` | 同 writer 的运行lease read-only watch，commit前后围栏 |
| `core/cpu_action_budget.py:_terminal,_recovery_terminal,_settle` | lease新版闭合/终态证明及F兼容；不改变no-refund/唯一slot/恢复屏障 |

这是语义最小集合，不要求把每个辅助函数按上述命名拆出。其它公共 closure 消费者若因 strict schema 需要修改，作者必须先列出确切依赖并报告；不顺便迁移所有 role/GPU adapters。涉及当前 C1 文件时等待前片收口，不修改对方候选。

## 7. 五类真实验收及证据门

每类可以有少量参数控制；目标是有限、可归因的矩阵，不按断言数冒称独立故障。只用新私有 CPU 项目、真实 Lake/native/supervisor、立即捕获的自有 pidfds；不得通过 mock 整树证明或直接填假 terminal 造绿。

1. **运行到期与零退出码**：真实 CLI/Orze 已 GO 的 worker、含逃逸后代；有效 lease 先于执行 envelope 到期。确认持久不可逆 EXPIRED、owned tree真闭合、STOP0仍 interrupted、无成功观察、一次 SETTLED且不退款。另有期限内正常完成控制。默认不提供 lease字段的真实入口必须走新lease。
2. **controller 不轮询**：READY/GO 后用测试自有同步 barrier 阻塞 controller，supervisor 在控制通道仍打开时自主到期停止树；恢复controller后不得凭旧cap/返回0完成正常 publication。不能只在父进程里调用 `harvest` 或手动 `stop` 充当自主期限证明。
3. **最终发布窗口**：真实 Domain/产物/观察注册后，在 terminal前、最后writer gate前让期限到达；成功事务须拒绝。另测试已commit响应未知或确认跨期限，保存 actual row/effect/lease/BOUND，不能预设SQL必已回滚；旧lease不得再次正常发布。
4. **失联与不可重派**：用已捕获 supervisor pidfd 制造实际失联并保持自有后代活；到期后重复观察和fresh CLI仍HOLD，无第二STOP/GO、无新attempt/claim、槽和预算不释放。清理只用测试握手认证的自有资源，不把测试清理当产品完成。
5. **时钟/持久CAS/兼容恢复**：两个真实连接并发expire/finish不能都获正常权限；boot变化/时钟异常/row替换与commit silent-rollback拒绝；合法无关peer reservation不造成全scope误拒。真实已确认terminal后crash86的F恢复继续只结算一次；旧v1已确认历史与旧非CPU调用正控保持，不能把这些正控算新lease过期证明。

首个旧行为回归优先使用当前真实动作的 timeout 与“GO后父不轮询、worker越界继续/晚到正常完成”窗口；新 config/API 缺失或import错误只分类新需求接线，不能冒称旧业务红。各故障首测前保存完整相关候选源码/测试；新增协议机制 first-green如实记录。

关闭条件：目标红绿与负控、源/测试冻结前后、真实raw controller/worker/lease/terminal/effect/预算轨迹、独立语义复核、相关旧回归/双仓全量、固定提交与远端读回齐全。记录期限、实际观测延迟、停止耗时和框架动作；不要求硬实时纳秒响应，不声称 exactly-once 外部副作用、普遍研究收益或整版V1已完成。

## 8. 根审冻结前必须确认

1. 默认所有新 native CPU 启用、配置字段名与TTL从发行起计（可短于envelope）的明确兼容变化。
2. normal terminal/effect 同事务末次CAS、commit/确认跨期限的一律保守HOLD；禁止提前FINISHING豁免。
3. 严格版本化监督字段/clock实现、表状态和typed错误区分；新旧收据不能相互冒领。
4. 五类测试的实际baseline与new-mechanism分类、完整源码ownership清单；C1/C2收口后才解锁生产。

本提案未运行任何新 worker 或故障测试，未修改任何生产源码、旧测试、应用或原挑战。仅记录可实施设计，所有性能与行为仍待实测。

