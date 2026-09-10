# 持久触发交付（V1-02A）

本项把原生角色的触发请求变成持久 inbox，并绑定每次实际启动。它不代表整个 V1-02 完成，更不把进程退出当作研究成功。

## 接收与身份

Pro 的 triggered_by、thinker manual、professor 和普通角色入口共享一次 `observe_trigger`。稳定文件内容入库后，pending 记录中的原始 payload 决定本次命令；后续租约只 CAS 同一个 delivery ID/hash，不能领取 B 却继续执行按 A 构建的命令。

项目 scope 绑定规范化的 project/config/results/database 路径，role 另行隔离；不随 objective 或 role 配置变化而丢弃已接收任务。不同 specification、科学 task、artifact、evaluation/observation 身份仍是后续项目，本表的 delivery ID 只是触发消息身份。

单个触发文件限制为 64 KiB UTF-8、普通文件；同次 fd 读取前后及路径元数据必须一致。原生 intake 不删除 live 文件，避免 check-then-unlink 删除生产者刚替换的请求。文件仍在不表示有 pending，文件消失也不表示已完成。文件版本及持久记录去重；生产者在接收之前连续覆盖同一个单槽文件，仍可能覆盖尚未被观察的请求，本项没有把旧文件协议变成无损多生产者队列。

缺文件且缺 DB 的观察不创建数据库；稳定有效文件接收可以创建配置 DB 父目录及自身/legacy 表。已存在数据库不静默更换 journal 模式，不兼容 schema、重定向、不可读文件或损坏状态会拒绝。只观察已接收版本或文件已消失的持久队列时，使用只读连接。写事务短且不跨 Popen；CAS、attempt、transition 的写后结果必须可核验。

## 状态与恢复

| 状态 | 原生自动动作 |
|---|---|
| PENDING | 可领取特定消息 |
| LEASED | owner/generation 唯一；确定未启动可 defer，过期后可换代领取 |
| LAUNCHING | Popen 前已提交启动意图；不能因过期、文件或进程消失而重放 |
| STARTED | Popen 与 RoleProcess 登记后确认；等待匹配尝试的真实完成 |
| IN_DOUBT | 副作用或收据不确定，阻止该 scope/role 自动绕回 scheduled |
| TERMINAL | 匹配尝试的进程结果及清理已记录；不是科学成功判定 |

每次 lease 换代，旧 owner 不能启动或改变新尝试；attempt ID 不可复用，attempt 的 nonce/argv hash 不可变，状态边追加记录。只有直接 Popen 的 FileNotFoundError/PermissionError 等当前明确识别为未 exec 的路径回 PENDING；泛异常、登记失败或 STARTED 写入不确定保持 LAUNCHING/IN_DOUBT，不盲重放。过期租约恢复仅适用于尚未进入 LAUNCHING 的工作。

命令 hash 严格绑定最终 argv 的 JSON 编码，不代表 cwd、全部环境、代码文件或完整执行协议指纹。nonce hash 来自同一个实际传入子进程的随机 nonce；进程信号权限仍由已有 PID/start_ticks/nonce 身份核验负责，delivery 表中的 PID 仅用于诊断。

正常 `check_active_roles` 用 RoleProcess 捕获的 db/launch 身份完成记账，不读可变 role_state 的当前 attempt。先校验角色和 nonce 关联，再写终态，再删除匹配的进程收据和放锁。写失败、陈旧或坏绑定、不可确认清理时保留证据与不可重放状态。超时、非零退出或 rate limit 可以是操作性 TERMINAL，但并不证明任务目标达成。

直接 shutdown/upgrade/孤儿清理可能只证实进程已停，不能证明它没有产生外部副作用；账本中的 STARTED/LAUNCHING 仍不可重放。当前没有自动解除 IN_DOUBT 或人工裁定 API：未知任务需要显式核实与后续处理，不能通过删角色内存状态变成新任务。本项未承诺完整跨文件原子性或任意外部副作用 exactly-once。

## 真实消费者与兼容

Script 保留原配置 args 插值；所有模式把原始 payload、delivery ID、payload hash 通过专用环境变量交付。普通 scheduled 调用清除继承的这些变量。任意自定义脚本是否实际采用这个接口不由框架代为断言。

Claude/research 的有效技能提示词还会附加 JSON 编码的 Captured task request；research 的内容寻址 rules 文件及子进程单次哈希校验绑定最终任务内容。技能占位符也使用本次模板值。技能清单只描述技能片段，不把任务区伪装成技能。缺 skills 键的 legacy research 可接收任务区；有 skills 键却为空或全部未激活时仍在预算和 Popen 前停止，任务内容不能绕过门控。

L1/L2 的观察基线仍独立于 delivery；预算、锁、构建、租约或启动确认失败不做新的激活 ACK。预留预算后未启动的尝试仍可能计费，沿用现有保守预算规则，不声称退款。无外部触发的 scheduled 角色没有被本项改为完整持久 attempt 系统。

旧 `claim_trigger` 和 `trigger_consumptions` 保留。原生入队同事务写 legacy 占位与 native 映射；旧消费行没有映射时不猜作 PENDING 或 STARTED。匹配旧 orphan 文件会阻断，而不是重做未知任务。混跑旧二进制仍保留其旧 unlink 风险；成对升级才是本分支的验收范围。

原生 startup 升级清理传 `preserve_triggers=True`，保留尚未 intake 的请求；显式 legacy cleanup API 默认仍保留旧行为。科学状态一致性、并发提案追加、独立 generation/observation、统一 quota/budget、inbox/文件的保留策略及整体输入预算尚未在本项关闭。

验收使用临时文件、真实 SQLite/完成路径与模拟 OS/provider 边界；research dry-run 证明最终提示词，不是实际付费模型或科研效果。没有修改线上项目、调用 GPU/训练或发布软件包。
