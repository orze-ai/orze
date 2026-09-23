# 后续独立验证的数据准备

已获取原始公开数据、检查完整性、完成分组和四份划分；没有基线预检、研究模型调用或 GPU 实验。划分核验见 [PREPARED.zh-CN.md](PREPARED.zh-CN.md)。当前混合模型开发实验仍冻结运行。新候选、优化任务、资源、共同参考及完整对照协议尚未冻结，不能声称已启动下一轮能力对照。

**重用纠正：原先只查 `tasks.json` 的范围不足。** 扩大到 Core/Pro 的历史计划、Markdown 证据和任务/计划 JSON，共检查 396 个文件后，确认 Bike Sharing 和 Superconductivity 已用于 9 月 18 日的研究。它们从后续新任务池移除；原下载及检查保留。证据路径、行号和文件哈希见 `novelty-review.json`。

替换后的六项为 EMNIST Balanced、SVHN、Banking77、DBpedia 14、YearPredictionMSD、Online News Popularity。在上述记录范围内未找到旧研究使用，不声称基础模型从未见过，也不声称历史记录完备。优化领域后续仍须保留，并明确区分生成问题与采集数据。

| 数据 | 一手来源 | 准备时须处理的问题 |
|---|---|---|
| EMNIST Balanced | [NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) | 47 类字符；只使用 balanced 子集，保留原 train/test 来源；检查方向、重复图像和标签范围。 |
| SVHN cropped digits | [作者站点](http://ufldl.stanford.edu/housenumbers/)、[torchvision 原始文件校验值](https://github.com/pytorch/vision/blob/main/torchvision/datasets/svhn.py) | 原标签 10 对应数字 0；优先作者下载，失败时镜像文件必须匹配公开 MD5。原始裁剪数据不提供足以确认所有图像来源独立的分组信息。 |
| Banking77 | [PolyAI 作者仓库](https://github.com/PolyAI-LDN/task-specific-datasets) | 固定作者仓库提交；相同或规范化后相同文本归组，检查重复文本的标签冲突，不按标签排序提供样本。CC BY 4.0。 |
| DBpedia 14 | [数据维护仓库](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | 固定数据提交，检查 title/content 的组合、类别与重复记录；保留原始划分来源。CC BY-SA 3.0。 |
| YearPredictionMSD | [UCI](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd) | 90 个音色统计量预测歌曲发行年份；保留作者规定的前 463,715 行训练、后 51,630 行测试边界，以保持原艺术家隔离。原文件无艺术家 ID，内部再划分不等于开发/确认也按艺术家隔离。CC BY 4.0。 |
| Online News Popularity | [UCI](https://archive.ics.uci.edu/dataset/332/online+news+popularity) | 采集的文章统计量与分享次数，不含原文。URL/采集间隔仅供分组；额外排除 12 个分享量聚合特征，因为其发布时可用性未核实，剩余 46 个输入。按发布日期、同 URL、同输入归组；随机日期划分不声称时间外推。CC BY 4.0。 |

已移出新任务池的 [Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) 和 [Superconductivity](https://archive.ics.uci.edu/dataset/464/superconductivty+data) 原始检查仍见 `raw-audit.json`；不因为历史重用问题而删除已经取得的记录。

EMNIST 原压缩包和 SVHN 文件的预期 MD5 来自已读取的 torchvision 数据集实现。所有实际下载再记录 SHA-256 和字节数；原九文件见 `downloads.json`，新增两文件见 `regression-downloads.json`。仅读取所需数据成员，不执行远端数据加载脚本。

只依据数据来源、任务覆盖、格式和泄漏风险准备输入。后续任何数据排除、划分或目标可达性调整都须发生在两组研究请求前，并保留原因和旧记录；不能依候选的结果挑任务或降低门槛。下载成功不代表数据划分合格，也不代表该任务的研究目标可达。
