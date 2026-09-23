# 后续独立验证的数据准备

仅获取原始公开数据并检查完整性；没有基线预检、研究模型调用或 GPU 实验。当前混合模型开发实验仍冻结运行。新候选、最终划分、资源、共同参考和门槛尚未冻结，不能声称已启动下一轮能力对照。

选择六个未在现有 `tasks.json` 中找到的数据集，覆盖视觉、文本和回归；这不是“基础模型从未见过”的证明。扫描日期为 2026-09-23，范围为本项目续作目录中的实验及归档任务声明。优化领域后续仍须保留，并明确区分生成问题与采集数据。

| 数据 | 一手来源 | 准备时须处理的问题 |
|---|---|---|
| EMNIST Balanced | [NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) | 47 类字符；只使用 balanced 子集，保留原 train/test 来源；检查方向、重复图像和标签范围。 |
| SVHN cropped digits | [作者站点](http://ufldl.stanford.edu/housenumbers/)、[torchvision 原始文件校验值](https://github.com/pytorch/vision/blob/main/torchvision/datasets/svhn.py) | 原标签 10 对应数字 0；优先作者下载，失败时镜像文件必须匹配公开 MD5。原始裁剪数据不提供足以确认所有图像来源独立的分组信息。 |
| Banking77 | [PolyAI 作者仓库](https://github.com/PolyAI-LDN/task-specific-datasets) | 固定作者仓库提交；相同或规范化后相同文本归组，检查重复文本的标签冲突，不按标签排序提供样本。CC BY 4.0。 |
| DBpedia 14 | [数据维护仓库](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | 固定数据提交，检查 title/content 的组合、类别与重复记录；保留原始划分来源。CC BY-SA 3.0。 |
| Bike Sharing hourly | [UCI](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) | 预测总租赁量时排除直接组成目标的 casual/registered；日期用于划分分组，去除行号；随机日期分组不能声称时间外推。CC BY 4.0。 |
| Superconductivity | [UCI](https://archive.ics.uci.edu/dataset/464/superconductivty+data) | 将特征表和化学组成表逐行核对，确保临界温度目标没有进入输入；相同组成/相同特征应归组，不能把重复材料分到训练与验收两边。CC BY 4.0。 |

EMNIST 原压缩包和 SVHN 文件的预期 MD5 来自已读取的 torchvision 数据集实现。所有实际下载再记录 SHA-256 和字节数；暂不解压整个归档或执行远端数据加载脚本。

只依据数据来源、任务覆盖、格式和泄漏风险准备输入。后续任何数据排除、划分或目标可达性调整都须发生在两组研究请求前，并保留原因和旧记录；不能依候选的结果挑任务或降低门槛。下载成功不代表数据划分合格，也不代表该任务的研究目标可达。
