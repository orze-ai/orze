# C2：配额与实验联合验证中的退出清理修复

日期：2026-09-12。冻结实施合同；尚未修复或验收完成。

真实 Orze/Pro 同轮 CPU-backed 联测已证明共享账户退避、健康账户独立及实验产物实际发布，但 `_run_leased` 在退出清理调用 `GpuSlotManager.clear()` 时抛错，后置断言未执行。原失败及一次单纯 worker mask fixture 修正见 Pro `docs/evidence/2026-09-12-quota-loop-baseline.json`。不得将产物存在说成该联测通过。

最小修复仅针对 legacy `graceful_shutdown` 的容器生命周期：入口捕获 training/evaluation/role 快照；只处理捕获对象；最后仅删除已处理、非 HOLD 且当前映射仍指向同一对象的条目。HOLD 对象原位保留，slot key、GPU bookkeeping 不重新分配；回调换入或新添对象不可被旧清理删除、覆盖或停止。复用现有 items/get/del 接口，不给 GpuSlotManager 增加伪装通用 MutableMapping 的抽象。

先冻结真实 GpuSlotManager/dict 的 kill/detach、训练/评估、HOLD/闭合、回调替换/新增边界单测并记录旧实现结果。这些映射单测可替代停止结果，不冒充真实进程证明。真实 Pro 联测保持已冻结源码和全部业务断言，修复后重跑，要求同轮配额、健康账户、原生发布、树闭合、退出及后置断言全部通过；不绕过 graceful_shutdown。

兼容回归覆盖现有 training/evaluation/role shutdown、termination HOLD、native publication 和 controller profile；最终两仓全量及配对回归仍必需。没有真实 GPU/provider、用户项目变更或部署。本片不改变 C3 的运行期租约合同。
