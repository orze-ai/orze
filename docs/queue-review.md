# 队列审核：执行权与研究判断分离

V1-01K1 修复的是一个可选的队列策略入口，不是通用科研价值判定器。
同配置、同 seed、规则笔记匹配、缺少 hypothesis，都不能自动删除任务。
审核不写 `metrics.json`，不把跳过当作实验失败、完成或测量结果。

## 明确启用

Pro 的命名 professor 仅为可选 preset。没有完整配置或没有显式启用时，
旧 verifier 的过滤入口保持无操作；环境中存在 API key 也不会自动开始审核。

```yaml
idea_review:
  enabled: true
  backend: ollama
  model: operator-selected-model
  max_batch: 20
  allow_skip: false
  allow_prioritize: false
```

`backend` 和 `model` 必须明确配置；`enabled` 和权限字段必须是布尔值。
支持 openai、anthropic、gemini、kimi、ollama；不是选中哪个可用 key 就调用哪个。
默认仅允许 APPROVE：保留排队资格，不认证科研价值或执行正确性。
SKIP、PRIORITIZE 各需显式授权。审核使用指定 provider/model，Gemini 审核
不启用搜索或模型回退；其他现有 Gemini 调用保留原来的默认回退语义。
自定义 endpoint 仅支持 openai、kimi、ollama；其他 provider 显式配置它会拒绝，
不静默忽略地址。单条/多条完整 JSONL 与 JSON 数组共用严格校验。

这里不提供新的跨角色配额、持久调用预算或精确一次 LLM 请求保证。
脚本被终止或响应丢失后可能再次请求审核；共享额度、退避和调用凭据归属
仍属于 V1-02/V1-03，不因本切片而宣称完成。本文示例不会自动修改任何项目。

## 实际入口和流程选择

script 角色向子进程传递绝对 `ORZE_CONFIG_PATH`，不传整个配置内容。
runner 的选择顺序为 `--config/-c`、该环境变量、存在的当前目录 `orze.yaml`。
显式缺失/无效配置不回退；配置与 `--results-dir` 冲突时拒绝启动。
路径按配置文件目录解析，保留现有 results-parent 控制目录语义和显式数据库位置。
配置只进入 `Context.extras['cfg']`，不进入 FSM 状态、历史和活动日志。
未声明 objective 不从运行时默认值制造一个研究目标。

runner 和插件现在共用 `orze.fsm.engine` 的注册表。低层加载接口保持可用，
main 对自动发现的 bundled procedures 默认仅运行 activity_log；显式启用
idea_review 才加入 idea_verifier。其他 bundled procedures 必须通过
`fsm.procedures` 的无后缀名称列表选择；未经选择的旧状态 maintain 也不执行。
项目 `procedures/` 中显式文件仍可运行和覆盖同名内置流程，项目插件优先。
这避免修复注册表时连带启用原先断路的暂停、触发和维护动作。
已有项目自定义插件须使用 canonical import；不承诺旧 `fsm.engine` 独立注册表兼容。

## 批次与事务

core 提供 `review_batch(db_path, limit=20)` 和
`apply_review_decisions(db_path, batch, decisions, allow_skip=False, allow_prioritize=False)`。

- 读取现存、合规的数据库；不创建或迁移数据库。读时缺少审核表表示尚无收据。
- 只选 legacy queued/pending、FSM QUEUED 且没有已运行阶段的任务。
  缺失历史阶段可兼容，未知或活动/终态阶段排除；不是修复冲突状态。
- 每项 revision 绑定数据库绝对路径、实际配置和审核元数据、优先级、队列时钟、
  最新全局转换 ID。正常领取后重新排队的 ABA 不会复用原 revision。
- 模型只能返回本地捕获批次的唯一 ID；未知、重复、非法、越权、过期的任一决策
  都拒绝整批。允许只决定批次的一个子集；未决定者不记录已审核。
- `BEGIN IMMEDIATE` 内重新核对全部选中版本及排队状态，再提交动作和收据。
  SKIP 同步写全局状态、legacy 镜像和转换审计；PRIORITIZE 只改优先级；
  APPROVE 只记审核收据，不制造生命周期边。
- 收据与动作一起提交；关键更新核对实际写入及读回，不把 SQLite 静默忽略
  当作成功。任何失败回滚整批；已提交版本重复提交明确拒绝。
- 优先级动作同时记录输入和结果 revision，避免动作自身导致立即重新审核。
  旧 JSONL 建议不是权威收据，不会永久遮蔽修订后的任务。

批次是框架内部捕获的数据，不是允许不可信客户端自行铸造的安全 capability。
本切片没有独立 batch/attempt/observation ID；也不保证针对恶意数据库所有者、
跨文件替换或任意触发器篡改的安全隔离。审核收据绑定任务版本，尚不提供
更换审核策略后自动使所有历史审批失效的策略版本迁移。

## 有界上下文与证据边界

只组合窄队列审核 SOP，不混入搜索、改代码、全局 steering 等整套角色工作。
上下文按完整项目资格规则重查当前本地证据，并与阶段一致的 COMPLETE 目录交集；
report.md、缓存分数、原始 accuracy 和旧建议日志不提供权威成绩。
最多显示 10 条合格主指标；这不是独立样本数、等价性或统计收敛结论。
发送有界任务标题、说明、类型和配置键名，不发送整个任务配置值或项目配置。
标题/说明本身仍是用户提供的内容，不能声称通用秘密扫描或完全脱敏。

提示词和响应各限 64 KiB，批次最多 20；过大的提示词缩小批次，解析失败不部分应用。
数据库读取用有界页越过已审核前缀，每条元数据也有 SQL 和解析上限。
为了找到未审核任务仍可能扫描整个队列；资格排序也不是 O(20) 的总 I/O 保证。
不可显示的配置只从审核批次省略，不将该任务标记为非法或删除。

FSM 的审核计数仅在事务返回已提交决策后增加，是流程遥测；数据库收据才是动作事实。
数据库已提交而 FSM 状态文件未保存的崩溃窗口仍可能造成遥测漏计，不做跨文件原子承诺。
测试中的模型/provider、进程边界均被隔离；没有付费调用、GPU 训练、部署或实测效率提升。
