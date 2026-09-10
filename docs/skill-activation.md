# 技能门控与实际交付

V1-01L2 只关闭技能组合、逐来源激活与原生 research 规则交付边界，不代表整个 V1 完成。

## 原生调用契约

`compose_skills_with_manifest(..., context_for_skill=..., strict=True)` 一次读取每个声明来源，返回实际组合的 `text` 和非空、已通过门控的 `included` 清单。每项包含 `source_key`、`requested_ref`、`resolved_source`、`trigger`、`source_sha256`，不复制正文。清单描述真正进入本轮技能提示词的片段，而不是所有声明的技能。相同来源重复声明仍保留重复正文，但一次启动只确认一次来源机会。

本地文件以解析后的绝对路径为键；Core/Pro 内置资源使用包内稳定键。frontmatter 的 id/name 不承担身份。哈希绑定同次读取的 UTF-8 文本，仅用于来源记录，不作为周期时钟键；仅修改正文不会重置已消费机会。

原字符串 `compose_skills()` API 保留旧的未知门控 warning/include、builtin 元数据兼容和空片段分隔行为。原生 Claude/research 使用严格模式：未知或无效 trigger、无效 frontmatter/计数/context、空正文不进入提示词。现有 `manual`、`plateau_or_new` 和两个零阈值表达式明确保留 always-included 兼容，并记录警告；它们不是新实现的手动或科学 plateau 判定。`on_file` 的路径/消费语义与 registry override 未在本项更改。

## 调度与确认

每个 role/source 分别维护 `periodic_research_cycles(N)` 时钟和 `on_plateau(N)` 的合格证据基线。首次周期时钟从 0 开始；成功启动只推进实际 included 的来源，不会让短周期技能重置长周期技能。预算拒绝、命令构建失败、Popen 失败或 RoleProcess 登记失败不确认机会。

所有正阈值 plateau 技能在同次组合共享一次当前资格读取。来源、完整 objective/qualification 配置与 agreed COMPLETE catalog 决定资格；Markdown 行数不提供证据。不同技能各自比较并确认当前合格、可比较、未消费的 distinct task ID。参考结果变更/撤回、资格缺失、状态超限或不可比较不冒充 plateau；这不是独立样本、统计收敛或科研进步证明。

ACK 只使用启动前捕获的清单、周期和 receipt，不重新读取技能或实验。已变化的 scope、状态代际或实际观察到的合法 trigger policy 拒绝旧 ACK。未知/不可读中间版本没有 ACK，也不重置既存时钟；恢复同一 gate 继续保留已消费历史。不声称记录全部文件修订历史。

状态上限为每角色 128 个曾跟踪的来源、总计 8192 个已保存及本轮待确认的 ID，单 plateau 仍受 L1 的 4096 ID 限制。未跟踪的普通/零阈值技能不占来源名额；超限不淘汰历史以制造新机会。状态上限不是总输入或扫描 I/O 上限，当前资格扫描仍可能 O(N)。

## Research 输入文件

存在 `skills` 键但显式空、类型无效、来源不可用/空白或全部未通过门控时，builder 返回 None；原生调度在预留预算和 Popen 前停止。没有 `skills` 键的旧 research 调用保留默认任务行为。

非空组合写入内容寻址的 `skill_composed_<sha256>.md`，不覆盖已存在的不同内容。目录/写入/核对失败走启动前拒绝并释放已拿到的角色锁。命令传递 `--rules-file` 和 `--rules-sha256`；子进程在上下文、决策协调与模型调用之前单次读取原始 bytes，要求哈希匹配、UTF-8 且非空，失败返回 -1，不回退默认任务。后续提示词复用首次读取的内容。旧 CLI 不传 hash 时仍保留原有可选规则行为。

## 未关闭边界

这些激活状态仍是逐主机调度状态，不是共享 durable delivery/attempt/observation 账本。进程已创建但登记失败的外部副作用不确定性、旧 trigger claim-before-Popen 窗口、跨文件原子性由 V1-02 继续处理。子进程发现规则失效时，父进程可能已预留预算和记录启动；此校验不承诺退款或未消耗父级启动机会。

既有输出回执/watchdog 仍按全部声明技能推导输出，不等于这里的实际 included 清单；没有声称两者已统一。内容寻址文件尚未做有界保留/清理，整个提示词/知识输入预算和来源扫描预算留待 V1-04。哈希不是签名或指令权限认证，也不证明技能科学合理。

验收未运行真实模型、GPU、训练或线上项目。成对源码分支推送不是软件包发布，测试通过不是 research 效率收益证明。
