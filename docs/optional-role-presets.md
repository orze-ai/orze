# 显式可选角色预设（V1-03E）

角色配置是授权，GOAL 和环境凭据只是输入与能力。默认 `role_presets: []`：没有显式角色时，不因 GOAL、教授名称或 API key 自动扩展一组研究角色。

## 两个独立预设

- `role_presets: [environment_research]`：在配置加载时，根据既有三个环境 API key 补缺失的 `research_gemini`、`research_openai`、`research_anthropic`。不改变原 provider/model 默认值。即使已有别的角色，也只补缺名；显式同名项整项优先，包括 `enabled: false`。
- `role_presets: [strategy_team]`：允许 Pro 的既有团队 bootstrap。仍需 GOAL 文件或已配置的教授；它不是“无条件创建全队”。缺失教授可由 GOAL 补入；只有已启用教授才补缺失的分析、工程、思考和组合策略角色。明确禁用教授不补队友、不写初始审核请求；单模型约束仍禁止自动补组合策略角色。
- 两者可同时声明，但没有隐式互相启用。必须是仅包含这两个精确名称的列表；空值、布尔值、未知名称和重复名称拒绝。运行时直接接收无效配置的检查也不部分授权。

团队按当前项目配置中的角色名称幂等物化，不使用进程全局的“已注入一次”标记。因此项目 A 不会抑制或启用项目 B；重复 tick 不覆盖用户角色，也不重写已经存在的审核请求。

## 停滞触发

`triggered_by` 指定上游，不再自行授予“上游空转就启动”的权限。要保留该可选策略，必须显式配置：

```yaml
roles:
  worker:
    mode: script
    script: worker.py
    triggered_by: upstream
    auto_trigger_on_stall: true
    auto_trigger_stall_threshold: 10
```

开关必须是布尔 true，阈值必须是正整数且不是布尔值。上游计数不明不能授权启动。真实手动/持久请求仍走原交付路径；预算、共享账户配额、自身失败退避和启动确认不被绕过。停滞计数本身不是科研证据，也不是信息增益。

## 兼容、迁移与范围

- 已显式配置的角色、legacy `research:` 迁移及明确开启的周期/手动策略继续工作。不改写用户 YAML，也不自动给旧项目补 preset 来维持默认付费行为；依赖旧自动发现的项目需主动添加预设或显式角色。
- 既有团队初始审核仅在用户选了团队预设且实际新建分析角色时产生；它是显式预设动作，不是证明有新研究证据。已声明教授成功后的旧下游 fan-out 保留。
- 本项处理配置加载和团队物化，不承诺移除 preset 后热更新会撤销已物化角色或停止运行中的进程；更改此类声明应按受控重启流程重新加载。没有新的跨主机角色编排服务。
- 两份既有 Pro 测试仅迁移显式 preset/stall 前提，完整旧文件保存在 evidence/snapshots；全部原业务 assert 保留，不能把这类迁移算作缺陷红测。
- 所有验证离线进行；到达 Popen 边界不等于真实 provider 调用或科学产出。接口属于未发布的开发分支。

证据：[V1-03E 可选预设](evidence/2026-09-10-v1-03e-optional-presets.json)。
