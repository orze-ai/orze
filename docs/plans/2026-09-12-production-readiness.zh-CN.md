# 生产部署预检（2026-09-12）

本次已完成固定提交的真实 wheel 构建、隔离安装、依赖检查及 Core-only CPU canary。**尚未切换生产环境，也未验证生产 Pro 许可、真实模型账户或 GPU 预算。** 下述环境是候选制品，不是部署完成证明。

## 固定制品与源码

构建来源为 `git archive`，不依赖 editable 安装。构建工具为 setuptools 81.0.0、wheel 0.46.3、build 1.4.0。

| 制品 | 源码提交 |
| --- | --- |
| 原 Core 4.6.2 | `82012aeb81cf6b6b137b80f27319a53d5ddeb8db` |
| Pro 0.13.1 | `da2d7d92da5281ab49362509fad71fc3d78e427a` |
| 指南修复版 Core 4.6.2 | `b3fba2bfa73a2e42665ade83f6bd1f36eac9a0a6` |

wheel SHA-256：

- 原 Core：`8c2cbb4f3def9a22ecfb4990c2675a9c2f84ee0550c54aad2ccee9075bf8cc4c`。
- Pro：`f27d1ca0675ac2a76ea11f9ae2e42cd5dfead9246f31854bf652bf8eb1ea8b11`。
- 修复版 Core：`a454c23f6b3683964c5890ae925504dd5df92937821e959c702463f2a71e09b6`。

原 Core 包含 228 个 Python 文件和 23 个已有声明资源；Pro 包含 48 个 Python 文件和 41 个资源。文件内容、Python 文件集合与固定 Git blob 一致，wheel RECORD 已核验。原 Core 声明的 `orze/SKILL.md` 实际缺失，不能把“已有资源匹配”算成完整打包通过。

修复版只增加该指南，原 228 个 Python 文件、23 个资源、pyproject 和其他 wheel 元数据不变；新指南等于 Git blob `e4a7883ad052ff1b2a9a5c0f15a2d55ffeb4d023`，其 SHA-256 为 `a94f9fac23733ddeb10f8a6825a7a91fb545d48beabad941da17c3a535ac7b7d`。两版 wheel 文件名与版本相同，选择制品必须核对目录和哈希。

## 隔离环境及依赖

候选根目录：

`/hot-data/fsx/workspace/erik/orze-release-candidate-6BUecCLP`

- 原 Core + Pro：`/hot-data/fsx/workspace/erik/orze-release-candidate-6BUecCLP/venv`。
- 原 Core-only 对照环境：`/hot-data/fsx/workspace/erik/orze-release-candidate-6BUecCLP/core-only-venv`。
- 修复版 Core-only：`/hot-data/fsx/workspace/erik/orze-release-candidate-6BUecCLP/patched-core-only-venv`。

各环境使用自己的 `bin/python`、`bin/orze`。原对照环境和原报告保持不变；未将修复版写入原环境。原 wheel 位于候选根目录的 `wheelhouse/`，修复版位于 `patched-wheelhouse/`。

依赖先从本地缓存/标准 PyPI 解析并记录 URL、版本及哈希，再使用本地 wheelhouse、`--no-index --require-hashes` 安装。原 paired 锁含 25 个发行包；Core-only 仅去掉 Pro，其余版本一致。修复版锁仅替换 Core wheel 的哈希，未升级依赖。另有独立 pip 引导记录。本轮锁面向实际 Python 3.10/Linux x86_64，不声称跨平台或 GPU 兼容。

| 已归档锁文件 | SHA-256 |
| --- | --- |
| `dependency-lock.json` | `1e2ca43681064437bafbcbcc352afdfb986ac4e2a8ba45e83288504996eb89df` |
| `paired.lock` | `e35ba7e3a3d23a9ef7db18cca286ba83a80a76578ec2b6fa10e4a3c9b486e69f` |
| `core-only.lock` | `9e9434ce726dd618e72c562dff1ca112899cee8bdeb853894569dfe0d63be96e` |
| `patched-core-only.lock` | `df50e5b8dc0fb27174421be9819400ac423c3676419cb97b40e232732029ba50` |

## 实际验证与保留记录

三套环境的真实 `pip check`、`orze --help` 均退出 0；安装文件也逐字节对应所选 wheel。两个 Core-only 环境不安装 Pro、不伪造许可、不修改 HOME 或用户配置。

原版最终 canary 与修复版 canary 均通过真实安装入口执行一个私有 CPU action：产物为 `{"sum":55,"count":10}`，原生 v2 租约为 authorized，实际 TREE_CLOSED/ECHILD_WALL、effect 确认、预算 SETTLED，原 2 秒预约额度不退款。原版最终工具为 `9aedb2`；修复版命令组 `1791cd` 全部退出 0，独立制品复核 `b1e160` 退出 0。

初次构建器隔离选项隐藏用户 site 的 build、宿主缺 ensurepip、首次 canary 快照误假定空 observations 表存在，均保留原失败和后续修正记录。首次 canary 实际 action 已完成并结算，但快照脚本失败；另建私有项目重跑，未覆盖原项目。原版累计两个 canary action，修复版一个；它们不计入正式 48 次研究对照，也不证明研究提速。

## 证据与上线边界

[归档索引](../evidence/runs/2026-09-12-production-preflight/index.json) 收录九份文本副本，保留原路径、大小和 SHA，逐字节相同；包含原/修复版报告、独立复核、四份锁及两个最终 canary 的完整报告。原始逐命令 stdout/stderr 与安装器报告仍保存在候选目录，归档报告保留其路径和哈希。没有归档 wheel、依赖二进制或授权文件。

归档前仅检查这些已生成文件的敏感模式和带凭据 URL，未发现匹配；未读取环境密钥、许可缓存或生产运行目录。该检查不声称通用秘密检测能力。

生产仍须明确目标服务、工作目录、现有状态兼容、旧所有者闭合、备份/回退程序，以及 Pro 许可、模型账户和资源预算，再单独验证切换。安装 Pro 不等于授权通过；本次有意不调用会读取用户许可的 Pro 运行时。未发布包、未修改现有 ASR 进程或全局 Python，也未将候选切为生产目标。
