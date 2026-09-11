# S1：只读报告规则收敛

日期：2026-09-11。实施前冻结；不是功能完成声明。

## 基线与目的

- Core：dbba0a69d8f76994f15930d2f3b59a27186b3e4b。
- Pro：59cafe463360fe1256a5b2eb2831efa96006814b。
- 同一 feat/autoresearch-v1 分支，保护其他工作区与生产运行；不部署、不合并 main。
- 本片响应“更 robust 且更简洁”：删除重复实现和 Pro 对 rebuild_state 私有函数的依赖，不新增恢复功能或状态协议。

## 固定范围

1. 数据列选择仅保留 reporting.evidence.dataset_metric_keys 的既有实现；rebuild_state._report_dataset_keys 保留兼容别名，不留第二份函数体。Pro 直接使用公共函数。
2. 原 rebuild_state._eligible_metric 的历史归档数值过滤函数原样移入 reporting.legacy_metrics，公开命名 archived_metric_value；旧私有名保留兼容别名，Core 旧归档查询及 Pro archive-only 分支使用同一函数。
3. 此归档函数不是当前产物、生命周期或科学资格证明。Pro 正常 build_context 继续把同一 qualified_entries 传给历史配置统计；明确空集合不得退回旧数据库分数。
4. 不改列顺序、重复列、WER 兼容选择、缺测回退、bool/non-finite 拒绝、排序、错误传播、配置、数据库 schema、事务、预算、来源或执行授权。旧函数返回与异常行为均为兼容要求。
5. 不把所有旧归档规则改成新资格规则；二者语义不同，不以“统一”名义放宽或收紧。

## 先验与验收

- 先在基线跑新增的行为刻画和已有相关 Core/Pro 测试，应为绿色。纯重构不捏造历史产品红测。
- 新增结构验收在基线应失败：重复函数体仍存在，Pro 仍依赖 rebuild_state 私有函数；单列为新架构要求，不冒称运行故障。
- 变更后原刻画与所有旧测试原文件不改，结构验收转绿。精确比对归档函数 AST（仅函数名改变）与数据列选择输出；覆盖普通指标、WER 兼容、空/重复列、零/负数、bool、非有限数、无效输入、显式空 qualified_entries 和只读数据库。
- 真实 Pro build_context 接线测试确认调用公共入口，归档路径仍不获得 authoritative 资格。
- 两仓源码/测试清单测试前后 SHA 固定；运行目标、Core 全量、Pro 全量及 60 项配对回归，记录实际命令/退出码/数量与所有失败，不按测试数推导科研收益。
- 独立审查后分别提交代码、证据并推送读回。无需为本片重新复制之前的所有执行轨迹；只声称已有测试在固定源码上重跑，保留完整输出与新测证据。

## 简化完成指标

- 数据列选择函数体：2 → 1。
- Pro research_context 对 engine.rebuild_state 私有函数依赖：2 → 0。
- 归档过滤算法：仍只有 1 份；旧名称为同一对象的别名，不新增代理实现。
- 持久状态、外部依赖、执行分支：均不新增。
- 如实记录生产源码净行数与新模块数；不以搬文件或文件数量冒充全部架构已简化。

## 后续边界

本片只是可核实的首个减法切口。claim 读取归一、执行事务收敛、新旧路径隔离等须分别审查；不默认扩大为 native repair、checkpoint 或 Director 重写。整个 V1 的总验收仍独立于此片。
