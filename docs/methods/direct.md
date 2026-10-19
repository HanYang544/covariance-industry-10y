# 网络直接预测：输入与完整流程

[方法导航](README.md) · [五种输出变换](output_heads.md) · [网络架构](networks.md) · [CAB](cab.md) · [实验结果](../../experiment_results.md)

## 两种输入

| 输入 | 形状 | 处理方式 |
| --- | --- | --- |
| 风险状态序列 | 100×6 | 训练集标准化，截断[-3,3] |
| 历史协方差序列 | 100×10×10 | 每个矩阵是历史20日中心化样本协方差；按训练集元素标准化，不截断 |

历史矩阵的截止日期为t−99到t，每张矩阵只用截止当日及之前收益，整个输入覆盖119个收益日。标签使用t+1到t+20，没有把未来窗口放入输入。

## Transformer如何处理矩阵？

每张10×10矩阵展平成100维，投影为32维Token，加时间位置编码，经过一层4头Transformer（FFN64），取最后时点状态，再交给输出头。

这不是每个资产一个Token的时空双重Attention；当前实现是每张矩阵一个时间Token。

## 与动态融合的根本区别

动态融合学习专家权重，矩阵结构由专家提供；这里的网络直接产生矩阵参数。前四种输出头不使用专家融合。Residual有单个LW60旁路，但仍不是多专家动态融合。

## 如何阅读实验？

先固定Transformer比较[五种输出变换](output_heads.md)，再固定Residual比较[六种编码器](networks.md)。CAB是[并列论文架构](cab.md)，保留自身结构。学习率搜索在上述模型配置不变时单独改变学习率。

[特征输入模型](../../src/covariance_research/direct/model.py) · [协方差输入模型](../../src/covariance_research/direct/inputs/covariance.py) · [数据读取](../../src/covariance_research/common/preprocessing/frozen.py)
