# 达成研究目标的效率：独立审计

固定目标：确认 RMSE 相对基线下降 50%；未达标计满 60 分钟。每题先平均两次重复，再对五题等权。

| 策略 | 封顶平均发现时间 | 15 分钟达标率 | 30 分钟达标率 | 60 分钟达标率 | 最终确认收益 |
|---|---:|---:|---:|---:|---:|
| 均匀探索 | 15.89 分钟 | 80.0% | 80.0% | 80.0% | 61.83% |
| Dream | 15.32 分钟 | 80.0% | 80.0% | 80.0% | 62.72% |
| 组合策略 | 14.33 分钟 | 80.0% | 80.0% | 80.0% | 63.23% |

Discovery time is retrospectively confirmed availability of a candidate selected only by development evidence. It is not an observed online stopping time or a measured reduction in complete user-goal delivery time. Confirmation, common preparation and audit delivery latency remain separately visible. Five synthetic families and one model do not establish universal research superiority.

原研究全程 8.85 小时；追加确认审计 11.81 分钟，98 次新增 CPU 程序评价，0 次新模型调用。

这是在已运行固定预算轨迹上审计首次有效候选的时间；没有在线提前停止实验，不包含将审计等待反事实删除后的端到端提速主张。
规则在运行中、独立确认前补充冻结；已见部分开发结果，不是研究启动前的完整预注册。默认决策须同时解释达标率、封顶均值、不确定性和最终质量。

- combined_minus_control：-1.57 分钟，问题层面 95% t 区间 [-3.75, +0.62]；参考 p=0.25。
- combined_minus_dream：-0.99 分钟，问题层面 95% t 区间 [-2.74, +0.75]；参考 p=0.375。
- dream_minus_control：-0.57 分钟，问题层面 95% t 区间 [-1.89, +0.75]；参考 p=0.375。
