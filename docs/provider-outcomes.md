# Provider 返回状态与研究结果（V1-03F）

本项将已有后端返回的完成语义交付给真实研究 cycle。它不新增 provider、模型或 SDK，不证明线上账号/服务已验证。

## 来源和分类

`call_llm(..., result_out=details)` 及已有后端函数保留字符串返回，额外提供**逐调用**状态与稳定原因。详细结果只描述本次 provider 输出，不从共享日志或全局“上一次结果”取值；不保存拒绝正文或原始响应。

| Provider 输出 | Cycle 处理 |
|---|---|
| 明确拒绝/内容过滤 | blocked / provider_refused；不算被拒提案，不解析、不追加 |
| 已知截断/未完成 | error；即使文本可解析为合法提案，也不能追加 |
| 明确完成且文本为空 | empty / provider_empty_response |
| HTTP/传输失败 | error / provider_transport_failed |
| 本地 token 预算耗尽或配置错误 | error；不得借用前次成功详情 |
| 完整文本 | 进入原解析、校验和不可覆盖追加路径；不自动等于有效实验 |

明确拒绝与正常空完成不授权再试另一个模型。原来有界的传输重试/明确启用的供应商 fallback 仍须逐次预留预算；最终状态与最终文本来自同一次响应，不拼接失败尝试的片段。

缺少完成元数据的既有兼容响应不会被标为“明确完成”。旧字符串调用及未提供新 collector 的扩展保持兼容；这些兼容输出不是完成协议已验证的证据。明确的非最终状态不能被一段合法 JSON 升格为完整输出。

## 原生消费

分类在 cycle 的提案解析前发生，继续通过 V1-03D 的 attempt/角色/nonce/scope 绑定结果，先交 Core 形成持久 trigger 终态，再交 Pro 的 usage/状态详情。拒绝不是提案验证失败；截断不是部分科学观察。accepted 仍只代表本批提案通过过滤且写入源，不是入库、运行或测量有效。

## 官方字段依据与验证边界

OpenAI 官方 Chat schema 区分普通内容与 refusal，流式 delta 同样有独立 refusal 字段；结束原因也区分正常结束、长度上限、内容过滤及工具调用。本项只读取结构化字段，不猜自由文本的拒绝含义。[OpenAI Chat API reference](https://developers.openai.com/api/reference/resources/chat)

其他已有后端的终止字段依据：[Anthropic stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)、[Gemini GenerateContent](https://ai.google.dev/api/generate-content)、[Kimi Chat API](https://platform.kimi.ai/docs/api/chat)、[Ollama Generate](https://docs.ollama.com/api/generate)。供应商规范是字段含义的依据，不能替代真实账号集成测试。

离线验收在 HTTP 边界提供响应，保留实际解析/重试/预算/cycle/原生终态。无付费调用、GPU 或真实科研收益声明。所有失败数量区分旧操作性行为红测、新稳定原因字段验收和既有通过控制，不将新的 reason 尚不存在夸成旧版错误接受了输出。

证据：[V1-03F Provider 分类](evidence/2026-09-10-v1-03f-provider-outcomes.json)。只有冻结源码验收、独立复核及远端提交核对齐全后才标记机制完成。
