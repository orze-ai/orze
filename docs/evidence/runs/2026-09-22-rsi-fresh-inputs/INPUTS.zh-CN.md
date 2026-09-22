# 下一轮研究效率验证的输入准备

这里只下载、解析、去重和划分数据，不运行研究候选，不观察候选得分。试验机制、预算与 30% 决策协议尚未冻结；输入准备成功不是研究能力提升证据。

## 数据与任务

| 任务 | 来源与口径 | 切分 |
|---|---|---|
| CIFAR-100 | [作者发布页](https://www.cs.toronto.edu/~kriz/cifar.html)，二进制版 MD5 与发布值一致；100 类图像 | 从官方训练划出 train/development/confirmation，原测试仅用于 audit；精确重复图像隔离 |
| KMNIST | [CODH](https://codh.rois.ac.jp/kmnist/index.html.en)，真实古籍手写字符、10 类 | 同上；不预设水平翻转保持标签 |
| AG News | [原论文作者仓库](https://github.com/zhangxiangxiao/Crepe)说明原始 CSV；本次采用 [PyTorch torchtext 指定镜像与校验值](https://github.com/pytorch/text/blob/main/torchtext/datasets/ag_news.py) | 相同规范化标题或正文的连通组隔离；原测试仅用于 audit，剔除与官方训练同组的 911 行 |
| TREC | [Li / Roth 发布页](https://cogcomp.seas.upenn.edu/Data/QA/QC/)，6 类问题分类 | 规范化文本组隔离；原测试中 11 行与训练同组，剩余 489 行 audit |
| Abalone | [UCI](https://archive.ics.uci.edu/dataset/1/abalone)，真实测量；目标 rings，sex 用三个指示列编码 | 无单独官方测试，按特征组随机划分四份 |
| SARCOS | [GPML 发布页](https://gaussianprocess.org/gpml/data/)，21 个状态输入预测第一个关节力矩 | 本次下载的官方测试大量重复训练，重新去重、分组、划分四份；其他六个力矩不作为特征 |
| TSP / QAP | 两类新随机生成实例，延续已声明几何/流量混合分布 | 各份独立生成；它们不是实际采集数据，也不是新问题类别 |

每任务两次划分，共 16 个输入世界，尚不意味着已经安排或运行 32 次研究。相同语料的两次划分不算两个独立领域。数据集在本平台既有开发计划中未找到使用记录，不声称基础模型未见过这些公开基准。

## 防止重复出现的评测问题

- 分组和行序均随机化，128-bit 切分种子只保存在 evaluator-private-seeds.json，与公开训练种子分离。候选只挂载许可的训练及预测特征文件，不能挂载 evaluator 文件。
- 特征组不得跨 train/development/confirmation/audit。独立校验同时重查实际特征摘要、标签范围、各份标签转换次数和数据文件 SHA256。
- 对 SARCOS 的双精度原始输入核验发现：4449 行源测试中 4428 行在源训练有完全相同输入，4425 行所有目标也相同。不是浮点降精度造成。相同任务观测先去重；有冲突目标的相同输入仍在同一组。详见 sarcos-source-overlap.json。
- 精确标题/正文/图像/输入去重不能保证不同报道事件、图像来源或机器人轨迹相互独立。没有时间标识就不声称 SARCOS 时间外推；组 bootstrap 也不能消除未观测相关性。
- 只在机制冻结后评价候选。后续共同参考模型须适配 CIFAR-100 输出维数及 KMNIST 方向性；在发研究请求前检查目标可达性的证据和计算限制，不把不可达目标的大量固定失败时间包装为策略优劣。

此前小型 TSP/QAP 的 2% 相对改进目标尚未得到可达性证明。因此，本目录的生成输入是候选评测材料，不是已确认合适的最终效能试验；必须先做同等条件的基线预检并明确这一不确定性，不能根据新候选表现事后修改目标。

## 文件

- downloads.json：来源、字节数、SHA256、可用的发布校验值。
- prepare.py：仅使用公开数据的解析与分组代码；不调用研究模型。
- preparation.json / data-manifest.json：生成统计与输入快照。
- verify_preparation.py / verification.json：独立的实际输入核验；不建立效能结论。

引用与许可：KMNIST 为 CODH / NIJL 的 CC BY-SA 4.0 数据；Abalone 为 UCI 的 CC BY 4.0。其他源的归属按上方发布页保留，不将镜像软件许可证等同新闻内容的版权许可。本仓库归档工具与摘要，不提交原始语料或图像。
