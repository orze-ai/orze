# CPU 动作：V1-06A 的产品接入边界

本片接入实际 `orze -c CONFIG` 前台入口和既有 `Orze._run_leased` 主循环，不增加第二个 runner、服务或数据库。它是 V1-06 的执行基础切片，不是整个通用 autoresearch 闭环或 V1-07 的跨领域验收。通用接口尚未冻结，留出任务仍不得打开。

## 显式启用

项目配置示例：

```yaml
results_dir: ./results
execution:
  version: 1
  resource: cpu
  slots: 1
  wall_budget_seconds: 60
action_policy:
  version: 1
  kind: queue
  idle: stop
  wait_seconds: 1
```

`execution` 缺失时仍为旧 GPU 路径；空 `allowed_gpus` 不推导 CPU 模式。声明 CPU 时不探测、不租用 GPU，也不构造 GPU slot manager。旧的 `train_script`、`base_config` 默认值不成为 CPU 前置条件。CPU 模式不运行隐式 evaluator、角色组、自动修复、升级、canary、GPU 遥测、admin 后台线程、扫主机 PID 的恢复/清理或旧排行榜判定；本片不提供这些功能的 CPU 版本。

现有 GPU controller_control profiles、managed run-idea、后台 start、resume/restart、启用 legacy provider roles 不属于此入口。任务用自己的 timeout_seconds，不用 CLI --timeout；资源、策略及作用域路径绑定后不热重载。控制目录参与路径指纹。同一 IdeaLake 内改变额度声明，或替换已捕获 scope 的数据库身份，都不能自动取得新预算。显式配置一个全新数据库属于另一个预算 namespace；跨数据库统一额度或防重置不在本片范围。CPU foreground 不发布 GPU PID/租约 ACK；SIGINT/SIGTERM 与既有停止标记可以要求它收尾，不能据此宣称未知旧工作已经关闭。

## 动作声明与输入

任务仍经既有 ideas.md 入库和 ACK。示例：

````markdown
## idea-0001: Sort inline values

```yaml
kind: native_cpu_action
action:
  version: 1
  adapter: command
  purpose: Verify ordering of supplied values
  inputs:
    values: [3, 1, 2]
  command:
    - python3
    - -c
    - |
      import json, os
      from pathlib import Path
      values = json.loads(os.read(int(os.environ['ORZE_ACTION_INPUT_FD']), 65536))['values']
      Path('answer.json').write_text(json.dumps(sorted(values)))
  timeout_seconds: 5
  outputs:
    answer:
      path: answer.json
      max_bytes: 128
```
````

声明必须包含上述七个字段且不能加未知字段；版本、数值、JSON 类型、大小、输出路径都严格校验。整个 action 限于既有 64 KiB、深度/节点预算。`inputs` 必须是显式 JSON 对象，`outputs: {}` 明确允许没有产物。purpose 不得为空，不强制 hypothesis、统计检验或训练参数。`**Kind**` 与 config.kind 同时存在时必须一致；普通训练与 CPU 的去重域不同。

输入作为已封印且已读回的 memfd，只继承给本次 worker。`ORZE_ACTION_INPUT_FD` 是从偏移 0 开始可读取的描述符，`ORZE_ACTION_INPUT_SHA256` 是对应字节摘要；写入被 Linux seals 拒绝。命令直接使用 argv，无 shell 展开。工作目录是本次独立 `_action_attempts/<attempt_id>/work`，不写 train.py 或 idea_config.yaml。

这不是 OS 沙箱、CPU 核心隔离、网络限制或 hermetic 环境：任意命令依赖的可执行文件、导入、环境和外部路径尚未全部封存。CUDA/NVIDIA/HIP/ROCR visibility 清空只声明本动作不分配 GPU，不是阻止恶意命令访问设备的安全边界。标准输出/错误目前丢弃；需要诊断时应由命令写入显式有界输出文件并声明。输出大小在发布时检查，不等于运行中磁盘写入配额。

## 状态、预算与策略

同一 IdeaLake 保存 `native_cpu_action` task、独立 `phase=action` AttemptRef、action stage、CPU scope/reservation/decision 以及既有产物记录。没有 training/evaluation stage，也不伪造 GPU 0/-1。旧 GPU 调度/直接 launcher 必须拒绝 CPU 任务，不能把命令数组展开成训练 sweep。

预算是声明的执行 wall-time envelope 总和，单位不是实际 CPU 利用时间。reserve 在短 SQLite writer 中抢槽、预扣整段超时上限；精确读回和完整 AttemptRef 绑定之后才允许 READY/GO。命令的 timeout 必须等于实际持久预留，不允许小额度授权长动作。本版保守不退未用 wall 额度，确认闭合只释放槽位。初始化失败、提交/监督未知保留额度和身份，不凭年龄、PID 死亡或下个 tick 自动重试/退款。

新 CPU 动作的超时由[持久运行期租约](cpu-runtime-lease.md)和原监督器自主整树 STOP 执行，不依赖父端轮询；期限包含准备和结果发布时间。终止/收尾有额外时间，OS 调度及存储故障仍可延迟处理，没有硬实时终止或总进程 CPU 使用硬上限的承诺。活跃动作不能因磁盘健康退避而进入默认 30 秒无检查等待。终态记录的 elapsed_wall_seconds 是实际观察的 wall 耗时，不能当 CPU 计费。

真实 QueuePolicy 消费队列和当前预算：从已验证的有界队列选择首个预算可容纳的任务；资源不足/空队列可 Wait，持久 reason 与 UTC wakeup；额度已无法容纳排队任务且没有未结算预留时 Stop；idle=stop 在队列及活跃工作排空后 Stop。Wait 不创建 claim、attempt、子进程或新预留。Stop 是当前作用域持久的“不再准入”决定，重新运行不会自动清除。预算只是有限队列执行策略，不是研究收敛判断。

`--once` 做一次策略决策，若派发则等待该动作整树闭合并结算；它不继续领取第二个任务，也不把一次性的空队列 Wait 自动升级为永久 Stop。普通模式使用同一主循环的可中断事件等待；活跃 handle 的轮询不是反复启动研究角色。

## 产物、失败与尚未完成

主进程退出不是完成证明。只有既有 supervisor 的完整 TREE_CLOSED/ECHILD_WALL 后，才在锁外复制/hash 声明产物，再于同一个 effect 事务重验来源、登记产物和 action 终态，完整提交/回执确认后结算 CPU 槽。被 STOP 或强制清理的 exit 0 不算成功。非零退出可明确 FAILED；缺少声明产物或发布不确定不能冒充 completed。

本片只支持显式零 observation：terminal.observation_ids=[]。产物不是测量或科学判断。尚未交付 measurement envelope、可替换 Domain.prepare/Policy.decide 插件、analyze/显式 CPU replica、恢复接管、未知副作用裁定、按实耗退款、有界持久历史压缩及两个异构领域/独立留出验收。后续必须沿本条真实 CLI、预算、attempt 和产物链路继续接入，不能用此处 QueuePolicy 或声明校验器冒充完成这些目标。

测试证据只覆盖私有 CPU 子进程与本地 SQLite/文件机制；不代表真实 GPU/provider、线上部署、科学独立性或科研收益已经验证。
